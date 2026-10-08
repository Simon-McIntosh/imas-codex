"""Unified ingestion pipeline for all content types.

Graph-driven ingestion via CodeFile queue:
- Automatic deduplication (skips already-ingested files)
- Per-file atomic commits (interrupt-safe)
- Auto-updates FacilityPath status to 'explored'
- Links extracted MDSplus paths to SignalNode entities
- Routes files to appropriate splitters based on language/type

Replaces code_examples.pipeline.
"""

import asyncio
import hashlib
import logging
import re
import time as _time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from imas_codex.discovery.base.facility import get_facility
from imas_codex.graph import GraphClient

if TYPE_CHECKING:
    from imas_codex.embeddings.encoder import Encoder

from .chunkers import chunk_code, chunk_text
from .extractors import reference_handlers_for_systems
from .extractors.ids import extract_ids_references, extract_imas_path_references
from .graph import (
    link_chunks_to_data_nodes,
    link_chunks_to_edas_signals,
    link_chunks_to_ids_roots,
    link_chunks_to_imas_paths,
    link_examples_to_facility,
)
from .queue import get_pending_files
from .readers.remote import TEXT_SPLITTER_LANGUAGES, fetch_remote_files

logger = logging.getLogger(__name__)

# Progress callback type: (current, total, message) -> None
ProgressCallback = Callable[[int, int, str], None]

# Target chunk size in characters for code chunking.
# Matches embed worker's TARGET_EMBED_TEXT_CHARS to avoid segment splitting
# at embed time.  4000 chars ≈ 1K tokens for Qwen3-Embedding.
DEFAULT_CHUNK_MAX_CHARS = 4000


def _split_and_extract(
    content: str,
    language: str,
    metadata: dict[str, Any],
    max_chars: int = DEFAULT_CHUNK_MAX_CHARS,
    chunk_lines: int = 40,
    chunk_lines_overlap: int = 10,
    use_text_splitter: bool = False,
) -> list[dict[str, Any]]:
    """Split content into chunks and extract metadata.

    Splits text using tree-sitter or sliding windows, then runs the reference
    extractors named by the facility's configured data systems.

    Args:
        content: Source code text
        language: Programming language
        metadata: Base metadata to attach to each chunk
        max_chars: Maximum characters per chunk
        chunk_lines: Target lines per chunk
        chunk_lines_overlap: Overlap lines between chunks
        use_text_splitter: Force text-based splitting

    Returns:
        List of chunk dicts with text, metadata, and extracted references
    """
    if use_text_splitter or language in TEXT_SPLITTER_LANGUAGES:
        chunks = chunk_text(
            content,
            chunk_size=max_chars,
            chunk_overlap=chunk_lines_overlap * 60,
        )
    else:
        chunks = chunk_code(
            content,
            language=language,
            max_chars=max_chars,
            chunk_lines=chunk_lines,
            chunk_lines_overlap=chunk_lines_overlap,
        )

    facility_id = metadata.get("facility_id")
    handlers = (
        reference_handlers_for_systems(
            get_facility(facility_id).get("data_systems") or {}
        )
        if facility_id
        else ()
    )
    result: list[dict[str, Any]] = []
    for chunk in chunks:
        # Extract IDS references
        ids_refs = extract_ids_references(chunk.text)
        related_ids = sorted(ids_refs) if ids_refs else []
        imas_paths = extract_imas_path_references(chunk.text, ids_refs)

        mdsplus_paths: list[str] = []
        edas_ref_count = 0
        for handler in handlers:
            refs = handler.extractor(chunk.text)
            if handler.linker is link_chunks_to_data_nodes:
                mdsplus_paths = [r.path for r in refs]
            elif handler.linker is link_chunks_to_edas_signals:
                edas_ref_count = len(refs)

        chunk_dict: dict[str, Any] = {
            "text": chunk.text,
            "start_line": chunk.start_line,
            "end_line": chunk.end_line,
            **metadata,
        }
        if related_ids:
            chunk_dict["related_ids"] = related_ids
            chunk_dict["related_ids_count"] = len(related_ids)
        if imas_paths:
            chunk_dict["imas_paths"] = imas_paths
        if mdsplus_paths:
            chunk_dict["mdsplus_paths"] = mdsplus_paths
            chunk_dict["mdsplus_ref_count"] = len(mdsplus_paths)
        if edas_ref_count:
            chunk_dict["_edas_ref_count"] = edas_ref_count

        result.append(chunk_dict)

    return result


def _generate_example_id(facility: str, remote_path: str, content: str = "") -> str:
    """Generate the id for a code example.

    The id is derived from the facility, the file path and the file's content,
    so it is stable across a re-ingest of unchanged content and changes when the
    content changes.  A stable id lets the example write merge rather than
    raise; a changed id supersedes the file's previous example.
    """
    digest = hashlib.md5(  # noqa: S324 - id derivation, not a security hash
        f"{facility}:{remote_path}:{content}".encode()
    ).hexdigest()[:8]
    return f"{facility}:{Path(remote_path).stem}:{digest}"


def _supersede_stale_example(
    graph_client: GraphClient,
    facility: str,
    remote_path: str,
    example_id: str,
) -> None:
    """Remove a file's previous example and chunks when its content changed.

    The example id is content-derived, so unchanged content re-ingests onto the
    same id and simply merges.  Content that changed yields a new id, and the
    file's existing example is then stale and must go with its chunks, or the
    graph keeps both.  The traversal is the shared ``_CODE_CHUNK_CASCADE``;
    no second traversal is written here.  A file with no ``CodeFile`` node in
    the graph matches nothing and is left alone.
    """
    from imas_codex.discovery.base.reset import _CODE_CHUNK_CASCADE

    rows = graph_client.query(
        """
        MATCH (e:CodeExample {facility_id: $facility, source_file: $path})
        RETURN e.id AS id
        """,
        facility=facility,
        path=remote_path,
    )
    existing = {r["id"] for r in rows}
    if not existing or existing == {example_id}:
        return
    graph_client.query(
        f"""
        UNWIND $cf_ids AS cid
        MATCH (n:CodeFile {{id: cid}})
        {_CODE_CHUNK_CASCADE}
        RETURN count(n)
        """,
        cf_ids=[f"{facility}:{remote_path}"],
    )


def _write_file_example(
    graph_client: GraphClient,
    facility: str,
    file_info: dict[str, Any],
    chunks: list[dict[str, Any]],
    example_props: dict[str, Any],
    source_file_id: str | None,
    mdsplus_ref_count: int,
) -> int:
    """Write one file's example, chunks and links in a single guarded step.

    The example write is idempotent on its id: a merge replaces an existing
    example rather than raising the uniqueness violation a plain create raises.
    A superseded example (content changed → a different id) is removed with its
    chunks through the shared cascade before the fresh example is merged.

    Returns the number of MDSplus data nodes linked for this file.
    """
    remote_path = file_info["remote_path"]
    example_id = file_info["example_id"]
    from_file_id = f"{facility}:{remote_path}"

    _supersede_stale_example(graph_client, facility, remote_path, example_id)

    graph_client.query(
        """
        UNWIND $examples AS item
        MERGE (e:CodeExample {id: item.id})
        SET e += item
        WITH e, item
        OPTIONAL MATCH (cf:CodeFile {id: item.from_file})
        FOREACH (_ IN CASE WHEN cf IS NULL THEN [] ELSE [1] END |
            MERGE (e)-[:FROM_FILE]->(cf))
        WITH e, item
        OPTIONAL MATCH (f:Facility {id: item.facility_id})
        FOREACH (_ IN CASE WHEN f IS NULL THEN [] ELSE [1] END |
            MERGE (e)-[:AT_FACILITY]->(f))
        """,
        examples=[example_props],
    )

    # FacilityPath status update + HAS_EXAMPLE
    graph_client.query(
        """
        UNWIND $items AS item
        MATCH (p:FacilityPath {facility_id: $facility})
        WHERE item.source_file STARTS WITH p.path
        MATCH (e:CodeExample {id: item.example_id})
        SET p.status = 'explored',
            p.last_ingested_at = datetime(),
            p.files_ingested = coalesce(p.files_ingested, 0) + 1
        MERGE (p)-[:HAS_EXAMPLE]->(e)
        """,
        facility=facility,
        items=[{"source_file": remote_path, "example_id": example_id}],
    )

    # CodeFile → HAS_EXAMPLE
    graph_client.query(
        """
        MATCH (cf:CodeFile {id: $cf_id})
        MATCH (ce:CodeExample {id: $ce_id})
        MERGE (cf)-[:HAS_EXAMPLE]->(ce)
        """,
        cf_id=from_file_id,
        ce_id=example_id,
    )

    # The private extraction count selects the linker and is not graph data.
    edas_ref_count = sum(chunk.pop("_edas_ref_count", 0) for chunk in chunks)

    # CodeChunk nodes (relationships handled below)
    graph_client.create_nodes("CodeChunk", chunks, create_relationships=False)

    # HAS_CHUNK + AT_FACILITY for this example's chunks
    graph_client.query(
        """
        MATCH (c:CodeChunk)
        WHERE c.code_example_id = $example_id
        MATCH (e:CodeExample {id: $example_id})
        MERGE (e)-[:HAS_CHUNK]->(c)
        WITH c
        WHERE c.facility_id IS NOT NULL
        MATCH (f:Facility {id: c.facility_id})
        MERGE (c)-[:AT_FACILITY]->(f)
        """,
        example_id=example_id,
    )

    # CodeFile status update
    if source_file_id:
        now = datetime.now(UTC).isoformat()
        graph_client.query(
            """
            MATCH (sf:CodeFile {id: $sf_id})
            SET sf.status = 'ingested',
                sf.completed_at = $now,
                sf.code_example_id = $ce_id,
                sf.error = null
            """,
            sf_id=source_file_id,
            ce_id=example_id,
            now=now,
        )

    linked = 0
    for handler in reference_handlers_for_systems(
        get_facility(facility).get("data_systems") or {}
    ):
        if handler.linker is link_chunks_to_edas_signals and edas_ref_count:
            handler.linker(graph_client, example_ids=[example_id], facility_id=facility)
        elif handler.linker is link_chunks_to_data_nodes and mdsplus_ref_count:
            linked = handler.linker(graph_client, example_ids=[example_id])
    return linked


def _extract_author(path: str) -> str | None:
    """Extract username from path like /home/username/..."""
    match = re.match(r"/home/(\w+)/", path)
    return match.group(1) if match else None


def _check_already_ingested(
    graph_client: GraphClient,
    facility: str,
    remote_paths: list[str],
) -> tuple[list[str], list[str]]:
    """Check which files are already ingested."""
    result = graph_client.query(
        """
        MATCH (e:CodeExample)
        WHERE e.facility_id = $facility AND e.source_file IN $paths
        RETURN e.source_file AS path
        """,
        facility=facility,
        paths=remote_paths,
    )

    already_ingested = {r["path"] for r in result}
    to_ingest = [p for p in remote_paths if p not in already_ingested]

    return to_ingest, list(already_ingested)


async def ingest_files(
    facility: str,
    remote_paths: list[str] | None = None,
    description: str | None = None,
    progress_callback: ProgressCallback | None = None,
    force: bool = False,
    limit: int | None = None,
    encoder: "Encoder | None" = None,
) -> dict[str, int]:
    """Ingest files from a remote facility.

    Can be called in two modes:
    1. **Path list mode**: Provide remote_paths explicitly
    2. **Graph-driven mode**: Omit remote_paths to process queued CodeFile nodes

    Embedding is deferred: CodeChunk nodes are written with
    ``embedding = null``.  The ``embed_text_worker`` populates
    embeddings asynchronously on the GPU.

    Features:
    - Deduplication: Skips files that are already ingested (unless force=True)
    - Interrupt-safe: Each file is committed atomically
    - Auto status update: CodeFile nodes are marked 'ingested'
    - MDSplus linking: Extracted paths are linked to SignalNode entities

    Args:
        facility: Facility SSH host alias (e.g., "tcv")
        remote_paths: List of remote file paths to ingest (if None, uses graph queue)
        description: Optional description for all files
        progress_callback: Optional callback for progress reporting
        force: If True, re-ingest files even if already present
        limit: Maximum files to process from graph queue
        encoder: Unused (kept for backward compatibility).  Embedding
            is now handled by the ``embed_text_worker``.

    Returns:
        Dict with counts: files, chunks, ids_found, mdsplus_paths, skipped, data_nodes_linked
    """
    stats = {
        "files": 0,
        "chunks": 0,
        "ids_found": 0,
        "mdsplus_paths": 0,
        "skipped": 0,
        "data_nodes_linked": 0,
    }

    def report(current: int, total: int, message: str) -> None:
        if progress_callback:
            progress_callback(current, total, message)
        logger.info("[%d/%d] %s", current, total, message)

    # Determine source of files
    source_file_ids: dict[str, str] = {}
    # Each file's own terminal outcome, keyed by remote path.  Every path asked
    # for ends with exactly one outcome, so the caller marks files from this
    # rather than assuming every claimed file ingested.
    outcomes: dict[str, dict[str, Any]] = {}

    def settle() -> dict[str, Any]:
        """Return ``stats`` carrying every requested path's own outcome.

        Every exit from this function routes through here.  Paths settled before
        any fetch — already ingested, or refused by the gate — must still carry
        an outcome, because the caller marks a claimed file from this mapping and
        an absent entry reads as an unrecorded claim rather than as a settled one.
        """
        stats["outcomes"] = dict(outcomes)
        stats["failed"] = {
            path: outcome["reason"]
            for path, outcome in outcomes.items()
            if outcome["status"] == "failed"
        }
        stats["skipped_files"] = {
            path: outcome["reason"]
            for path, outcome in outcomes.items()
            if outcome["status"] == "skipped"
        }
        return stats

    if remote_paths is None:
        query_limit = limit if limit is not None else 10000
        pending = get_pending_files(facility, limit=query_limit)
        if not pending:
            report(0, 0, "No pending files in queue")
            return settle()

        remote_paths = [p["path"] for p in pending]
        source_file_ids = {p["path"]: p["id"] for p in pending}
        report(0, len(pending), f"Processing {len(pending)} queued files")

    total_files = len(remote_paths)
    report(0, total_files, f"Starting ingestion of {total_files} files")

    # Ensure Facility node exists so AT_FACILITY relationships don't fail
    with GraphClient() as gc:
        gc.ensure_facility(facility)

        # Ingestion gating: verify this graph allows the target facility
        try:
            from imas_codex.graph.meta import gate_ingestion

            gate_ingestion(gc, facility)
        except ValueError as e:
            logger.error("Ingestion gated: %s", e)
            report(0, 0, f"Ingestion blocked: {e}")
            return settle()

    # Deduplication check
    paths_to_ingest = remote_paths
    if not force:
        with GraphClient() as check_client:
            paths_to_ingest, already_ingested = _check_already_ingested(
                check_client, facility, remote_paths
            )
            stats["skipped"] = len(already_ingested)
            for path in already_ingested:
                outcomes[path] = {"status": "ingested", "reason": "already ingested"}
            if already_ingested:
                report(
                    0,
                    total_files,
                    f"Skipping {len(already_ingested)} already-ingested files",
                )

    if not paths_to_ingest:
        report(total_files, total_files, "All files already ingested")
        return settle()

    # Collect file content grouped by language
    files_by_language: dict[str, list[dict[str, Any]]] = {}
    file_metadata: dict[str, dict[str, Any]] = {}

    # Fetch files off the event loop thread (SSH blocks synchronously)
    fetched_files = await asyncio.to_thread(
        lambda: list(fetch_remote_files(facility, paths_to_ingest))
    )

    for idx, (remote_path, content, language) in enumerate(fetched_files):
        filename = Path(remote_path).name
        report(idx, len(paths_to_ingest), f"Fetched {filename} ({language})")

        example_id = _generate_example_id(facility, remote_path, content)
        author = _extract_author(remote_path)

        file_metadata[example_id] = {
            "facility_id": facility,
            "source_file": remote_path,
            "language": language,
            "title": filename,
            "description": description or f"Code example from {remote_path}",
            "author": author,
            "ingested_at": datetime.now(UTC).isoformat(),
        }

        file_info = {
            "content": content,
            "remote_path": remote_path,
            "example_id": example_id,
        }

        if language not in files_by_language:
            files_by_language[language] = []
        files_by_language[language].append(file_info)

    # A path the fetch never returned gets its own failed outcome, so the caller
    # does not leave it claimed and silent.
    for path in paths_to_ingest:
        if path not in outcomes and not any(
            fi["remote_path"] == path
            for flist in files_by_language.values()
            for fi in flist
        ):
            outcomes[path] = {"status": "failed", "reason": "fetch failed"}

    if not files_by_language:
        report(total_files, total_files, "No files to process")
        return settle()

    # Flatten all files — process together regardless of language.
    # Previous code grouped by language and ran separate chunk→embed→write
    # cycles per group. With 10 files across 5 languages, that meant 5
    # separate embedding calls and ~100 individual graph queries.
    all_files: list[dict[str, Any]] = []
    for language, file_list in files_by_language.items():
        for file_info in file_list:
            file_info["language"] = language
            all_files.append(file_info)

    total_to_process = len(all_files)
    stats["files"] = 0

    BATCH_SIZE = 20

    processed_files = 0
    for batch_start in range(0, total_to_process, BATCH_SIZE):
        batch_files = all_files[batch_start : batch_start + BATCH_SIZE]
        batch_end = min(batch_start + BATCH_SIZE, total_to_process)

        report(
            processed_files,
            total_to_process,
            f"Processing files {batch_start + 1}-{batch_end}/{total_to_process}",
        )

        # Split and extract for each file (language-aware per file).  A file
        # that yields no chunk — or whose extraction fails outright — is given
        # its own terminal outcome rather than blocking the rest of the batch.
        prepared: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []
        t_chunk_start = _time.monotonic()

        for file_info in batch_files:
            example_id = file_info["example_id"]
            remote_path = file_info["remote_path"]
            language = file_info["language"]
            chunk_metadata = {
                "source_file": remote_path,
                "facility_id": facility,
                "language": language,
                "code_example_id": example_id,
            }

            try:
                chunks = await asyncio.to_thread(
                    _split_and_extract,
                    file_info["content"],
                    language,
                    chunk_metadata,
                )
            except Exception:
                logger.warning(
                    "Failed to parse %s with tree-sitter, trying text splitter",
                    remote_path,
                )
                try:
                    chunks = await asyncio.to_thread(
                        _split_and_extract,
                        file_info["content"],
                        language,
                        chunk_metadata,
                        use_text_splitter=True,
                    )
                except Exception as e2:
                    logger.error("Failed to process %s: %s", remote_path, e2)
                    outcomes[remote_path] = {
                        "status": "failed",
                        "reason": f"extraction failed: {e2}",
                    }
                    continue

            if not chunks:
                # An admitted file whose extraction yields no chunk reaches a
                # terminal skipped state, instead of waiting at ``scored`` for a
                # claim that never comes.
                logger.info("No chunks extracted from %s; marking skipped", remote_path)
                outcomes[remote_path] = {
                    "status": "skipped",
                    "reason": "no chunks extracted",
                }
                continue

            # Generate chunk IDs
            for i, chunk in enumerate(chunks):
                content_hash = hashlib.md5(chunk["text"].encode()).hexdigest()[:8]
                chunk["id"] = f"{example_id}:chunk_{i}:{content_hash}"

            prepared.append((file_info, chunks))

        if not prepared:
            processed_files += len(batch_files)
            stats["files"] = processed_files
            continue

        t_chunk_elapsed = _time.monotonic() - t_chunk_start

        # Deferred embedding: write chunks to graph WITHOUT embeddings.
        # The embed_text_worker picks up CodeChunk nodes where
        # embedding IS NULL and embeds them asynchronously on the GPU.
        # This decouples ingestion throughput from embedding latency.

        # Count stats over the files that produced chunks.
        batch_ids_found = 0
        batch_mdsplus_paths = 0
        for _file_info, chunks in prepared:
            for chunk in chunks:
                batch_ids_found += len(chunk.get("related_ids", []))
                batch_mdsplus_paths += len(chunk.get("mdsplus_paths", []))

        batch_chunk_count = sum(len(chunks) for _fi, chunks in prepared)
        stats["chunks"] += batch_chunk_count
        stats["ids_found"] += batch_ids_found
        stats["mdsplus_paths"] += batch_mdsplus_paths

        # Per-file graph writes.  Each file's nodes are written in their own
        # guarded step, so a failure in one file's write — a uniqueness
        # violation, a bad property — fails only that file and leaves the rest
        # of the batch ingested.
        t_graph_start = _time.monotonic()
        step_times: dict[str, float] = {}
        with GraphClient() as graph_client:
            for file_info, chunks in prepared:
                remote_path = file_info["remote_path"]
                example_id = file_info["example_id"]
                meta = file_metadata.get(example_id)
                if not meta:
                    outcomes[remote_path] = {
                        "status": "failed",
                        "reason": "missing example metadata",
                    }
                    continue
                example_props = {
                    "id": example_id,
                    **meta,
                    "from_file": f"{facility}:{remote_path}",
                }
                t_s = _time.monotonic()
                try:
                    linked = _write_file_example(
                        graph_client,
                        facility,
                        file_info,
                        chunks,
                        example_props,
                        source_file_ids.get(remote_path),
                        batch_mdsplus_paths,
                    )
                except Exception as e:
                    logger.error("Ingest write failed for %s: %s", remote_path, e)
                    outcomes[remote_path] = {
                        "status": "failed",
                        "reason": str(e)[:200],
                    }
                    continue
                stats["data_nodes_linked"] += linked
                outcomes[remote_path] = {
                    "status": "ingested",
                    "example_id": example_id,
                }
                step_times["write_examples"] = (
                    step_times.get("write_examples", 0.0) + _time.monotonic() - t_s
                )

        t_graph_elapsed = _time.monotonic() - t_graph_start

        logger.info(
            "Batch %d-%d timing: chunk=%.1fs graph=%.1fs (%d chunks, %d files written)",
            batch_start + 1,
            batch_end,
            t_chunk_elapsed,
            t_graph_elapsed,
            batch_chunk_count,
            len(prepared),
        )

        processed_files += len(batch_files)
        stats["files"] = processed_files

    settle()

    # Final relationship linking — safety net for any relationships not
    # created in the per-batch step above (e.g. cross-batch references).
    all_example_ids = list(file_metadata.keys())
    report(processed_files, total_to_process, "Creating final graph relationships...")

    t_link_start = _time.monotonic()
    with GraphClient() as graph_client:
        link_chunks_to_ids_roots(graph_client, example_ids=all_example_ids)
        link_chunks_to_imas_paths(graph_client, example_ids=all_example_ids)
        handlers = reference_handlers_for_systems(
            get_facility(facility).get("data_systems") or {}
        )
        if stats["mdsplus_paths"] > 0 and any(
            handler.linker is link_chunks_to_data_nodes for handler in handlers
        ):
            link_chunks_to_data_nodes(graph_client, example_ids=all_example_ids)
        link_examples_to_facility(graph_client, example_ids=all_example_ids)
    t_link_elapsed = _time.monotonic() - t_link_start

    report(
        total_to_process,
        total_to_process,
        f"Completed: {stats['files']} files, {stats['chunks']} chunks, "
        f"{stats['skipped']} skipped, {stats['data_nodes_linked']} data nodes linked",
    )
    logger.info("Final linking: %.1fs", t_link_elapsed)
    return stats


__all__ = [
    "ProgressCallback",
    "ingest_files",
]
