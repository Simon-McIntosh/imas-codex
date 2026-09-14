"""Sync ISN grammar spec (segments, tokens, templates) to Neo4j.

The grammar spec is the canonical vocabulary for Standard Names and is
owned exclusively by the SN subsystem. This module provides the
library-level sync helper used by:

* ``sn run`` CLI — auto-sync at startup when the active grammar version
  differs from the installed ISN package (idempotent no-op otherwise)
* ``sn clear`` CLI — auto re-seed after a full subsystem wipe

The spec is loaded from the installed ``imas_standard_names`` package.
Writes are idempotent — re-running is a no-op at the database level.

Two pieces of state are owned by imas-codex rather than ISN and live
here:

1. Composite ``id`` properties — the LinkML schema declares ``id`` as
   the identifier slot for every grammar node using a deterministic
   composite format (e.g. ``{version}:{segment}:{value}``). ISN's
   sync_grammar keys nodes on the natural composite ``(version, name)``
   but does not project into a single ``id`` slot — we project here.
2. ``active`` flag rotation — "which grammar version is the running
   composition pipeline using" is imas-codex pipeline state (ADR-8).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from imas_codex.graph.client import GraphClient

logger = logging.getLogger(__name__)


_FINALISE_STATEMENTS: tuple[tuple[str, str], ...] = (
    (
        "set ISNGrammarVersion.id",
        "MATCH (v:ISNGrammarVersion) "
        "WHERE v.id IS NULL AND v.version IS NOT NULL "
        "SET v.id = v.version",
    ),
    (
        "set GrammarSegment.id",
        "MATCH (s:GrammarSegment) "
        "WHERE s.id IS NULL AND s.version IS NOT NULL AND s.name IS NOT NULL "
        "SET s.id = s.version + ':' + s.name",
    ),
    (
        "set GrammarToken.id",
        "MATCH (t:GrammarToken) "
        "WHERE t.id IS NULL AND t.version IS NOT NULL "
        "  AND t.segment IS NOT NULL AND t.value IS NOT NULL "
        "SET t.id = t.version + ':' + t.segment + ':' + t.value",
    ),
    (
        "set GrammarTemplate.id",
        "MATCH (tpl:GrammarTemplate) "
        "WHERE tpl.id IS NULL AND tpl.version IS NOT NULL AND tpl.name IS NOT NULL "
        "SET tpl.id = tpl.version + ':template:' + tpl.name",
    ),
    (
        "rotate ISNGrammarVersion.active flag",
        "MATCH (v:ISNGrammarVersion) SET v.active = (v.version = $version)",
    ),
)


def _grammar_input_paths() -> tuple[Path, ...]:
    """Return the installed files that the ISN grammar loader consumes.

    ``_GRAMMAR_SPEC_PATH`` is the loader's installed-package location, so this
    follows the runtime package rather than a checkout.  The vocabulary set is
    discovered from that specification's sibling directory; adding a YAML
    vocabulary therefore enters the digest without a codex-side file list.
    """
    from imas_standard_names.grammar_codegen import spec as grammar_spec

    specification = Path(grammar_spec._GRAMMAR_SPEC_PATH)
    vocabularies = tuple(sorted((specification.parent / "vocabularies").glob("*.yml")))
    if not vocabularies:
        raise RuntimeError(f"No grammar vocabularies found beside {specification}")
    return (specification, *vocabularies)


def _grammar_input_signature(paths: tuple[Path, ...]) -> list[dict[str, int | str]]:
    """Return the file metadata that invalidates a persisted digest."""
    return [
        {
            "path": str(path),
            "mtime_ns": path.stat().st_mtime_ns,
            "size": path.stat().st_size,
        }
        for path in paths
    ]


def _grammar_digest_cache_path() -> Path:
    """Return the user-local cache used by the startup grammar freshness check."""
    return Path.home() / ".cache" / "imas-codex" / "grammar-content-digest.json"


def _cached_grammar_digest(signature: list[dict[str, int | str]]) -> str | None:
    """Read a digest only when it was made from these exact file mtimes."""
    try:
        record = json.loads(_grammar_digest_cache_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if record.get("signature") != signature:
        return None
    digest = record.get("digest")
    return digest if isinstance(digest, str) else None


def _cache_grammar_digest(signature: list[dict[str, int | str]], digest: str) -> None:
    """Persist a digest atomically; a cache miss remains correct if this fails."""
    cache_path = _grammar_digest_cache_path()
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=cache_path.parent,
            prefix=f".{cache_path.name}.",
            delete=False,
        ) as handle:
            json.dump(
                {"signature": signature, "digest": digest}, handle, sort_keys=True
            )
            temporary_path = Path(handle.name)
        temporary_path.replace(cache_path)
    except OSError:
        logger.debug("grammar digest cache write failed", exc_info=True)


def grammar_content_digest() -> str:
    """Return a whitespace-stable digest of the installed grammar inputs."""
    from imas_standard_names.grammar_codegen.spec import IncludeLoader

    paths = _grammar_input_paths()
    signature = _grammar_input_signature(paths)
    if digest := _cached_grammar_digest(signature):
        return digest
    grammar_root = paths[0].parent
    hasher = hashlib.sha256()
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            loader = IncludeLoader if path == paths[0] else yaml.SafeLoader
            document = yaml.load(handle, Loader=loader)
        canonical = json.dumps(
            document,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        hasher.update(path.relative_to(grammar_root).as_posix().encode())
        hasher.update(b"\0")
        hasher.update(canonical.encode())
        hasher.update(b"\0")
    digest = f"sha256:{hasher.hexdigest()}"
    _cache_grammar_digest(signature, digest)
    return digest


_MERGE_CONTEXT_TOKENS = """
UNWIND $rows AS row
MATCH (s:GrammarSegment {name: row.segment, version: $version})
MERGE (t:GrammarToken {
    value: row.value,
    segment: row.segment,
    version: $version
})
ON CREATE SET t.aliases = []
MERGE (s)-[:HAS_TOKEN]->(t)
"""


def _grammar_context_token_rows(context: dict[str, Any]) -> list[dict[str, str]]:
    """Flatten the public ISN vocabulary sections into graph token rows."""
    declared_segments = set(context["segment_descriptions"])
    return [
        {"segment": str(section["segment"]), "value": str(token)}
        for section in context["vocabulary_sections"]
        if section["segment"] in declared_segments
        for token in section["tokens"]
    ]


def _sync_context_tokens(
    gc: GraphClient,
    *,
    version: str,
    rows: list[dict[str, str]],
    dry_run: bool,
) -> dict[str, Any]:
    """Ensure the graph mirrors every token exposed by the public context."""
    report: dict[str, Any] = {"rows": len(rows), "applied": not dry_run}
    if dry_run:
        report["planned_statement"] = _MERGE_CONTEXT_TOKENS
        return report
    gc.query(_MERGE_CONTEXT_TOKENS, version=version, rows=rows)
    return report


@dataclass
class GrammarSyncReport:
    """Result of a grammar sync run."""

    isn_version: str
    content_digest: str
    spec_version: str
    segments: int
    templates: int
    dry_run: bool
    applied: bool
    raw_report: dict[str, Any] = field(default_factory=dict)
    finalise_report: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _finalise_active_version(
    gc: GraphClient, version: str, content_digest: str, dry_run: bool
) -> dict[str, Any]:
    """Set composite IDs, snapshot content, and rotate the active version."""
    report: dict[str, Any] = {
        "target_version": version,
        "content_digest": content_digest,
        "applied": not dry_run,
    }

    if dry_run:
        report["planned_statements"] = list(_FINALISE_STATEMENTS)
        return report

    for label, cypher in _FINALISE_STATEMENTS:
        gc.query(cypher, version=version, content_digest=content_digest)
        report[label] = "ok"
    gc.query(
        "MATCH (v:ISNGrammarVersion {version: $version}) "
        "SET v.content_digest = $content_digest",
        version=version,
        content_digest=content_digest,
    )
    report["store ISNGrammarVersion.content_digest"] = "ok"
    return report


def sync_isn_grammar_to_graph(
    *,
    dry_run: bool = False,
    gc: GraphClient | None = None,
) -> GrammarSyncReport:
    """Sync the installed ISN grammar spec into Neo4j.

    Writes ``ISNGrammarVersion``, ``GrammarSegment``, ``GrammarToken``,
    ``GrammarTemplate`` nodes plus ``DEFINES`` / ``HAS_TOKEN`` / ``NEXT``
    / ``USES_TEMPLATE`` edges. Idempotent — safe to re-run.

    Parameters
    ----------
    dry_run:
        When True, return planned statements without touching the graph.
    gc:
        Optional open :class:`GraphClient`. When None, the function opens
        and closes its own client.

    Returns
    -------
    :class:`GrammarSyncReport` with ISN version, counts, and
    per-statement report.

    Raises
    ------
    RuntimeError
        If the ISN package is not installed or the sync fails.
    """
    try:
        from imas_standard_names import __version__ as isn_version, get_grammar_context
        from imas_standard_names.graph.spec import get_grammar_graph_spec
        from imas_standard_names.graph.sync import sync_grammar
    except ImportError as exc:
        raise RuntimeError(
            "imas_standard_names package not available — cannot sync grammar."
        ) from exc

    spec = get_grammar_graph_spec()
    content_digest = grammar_content_digest()
    spec_version = spec.get("version", "unknown")
    segments = len(spec["segments"])
    templates = len(spec["templates"])
    context_token_rows = _grammar_context_token_rows(get_grammar_context())

    logger.info(
        "Sync ISN grammar: isn=%s spec=%s segments=%d templates=%d dry_run=%s",
        isn_version,
        spec_version,
        segments,
        templates,
        dry_run,
    )

    owns_client = gc is None
    client_cm: GraphClient | None = None
    try:
        if owns_client:
            client_cm = GraphClient()
            client_cm.__enter__()
            gc_local: GraphClient = client_cm
        else:
            assert gc is not None
            gc_local = gc

        report = sync_grammar(gc_local, active_version=isn_version, dry_run=dry_run)
        context_report = _sync_context_tokens(
            gc_local,
            version=isn_version,
            rows=context_token_rows,
            dry_run=dry_run,
        )
        finalise_report = _finalise_active_version(
            gc_local,
            version=isn_version,
            content_digest=content_digest,
            dry_run=dry_run,
        )
        finalise_report["public context tokens"] = context_report
    except Exception as exc:  # noqa: BLE001 — surface as RuntimeError
        raise RuntimeError(f"Failed to sync grammar to Neo4j: {exc}") from exc
    finally:
        if owns_client and client_cm is not None:
            client_cm.__exit__(None, None, None)

    if dataclasses.is_dataclass(report):
        raw = dataclasses.asdict(report)
    elif hasattr(report, "__dict__"):
        raw = dict(report.__dict__)
    else:
        raw = dict(report)

    return GrammarSyncReport(
        isn_version=isn_version,
        content_digest=content_digest,
        spec_version=str(spec_version),
        segments=segments,
        templates=templates,
        dry_run=dry_run,
        applied=not dry_run,
        raw_report=raw,
        finalise_report=finalise_report,
    )
