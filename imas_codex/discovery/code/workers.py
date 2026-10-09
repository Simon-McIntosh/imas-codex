"""Async workers for parallel code discovery.

Workers that process code files through the pipeline:
- scan_worker: SSH file enumeration (FacilityPaths → CodeFile nodes)
- triage_worker: LLM dimension triage (discovered → triaged | skipped)
- enrich_worker: rg pattern matching + preview extraction (triaged → enriched)
- score_worker: LLM full scoring (enriched → scored)
- code_worker: Code ingestion — fetch, chunk, embed (scored → ingested)

Workers coordinate through graph_ops claim/mark functions using claimed_at timestamps.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from imas_codex.discovery.base import reachability
from imas_codex.discovery.base.claims import retry_on_deadlock
from imas_codex.discovery.base.reachability import (
    host_unreachable,
    wait_for_reachable_host,
)
from imas_codex.discovery.base.supervision import is_infrastructure_error

from .state import FileDiscoveryState

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)


def _render_score(value: Any) -> str:
    """Render a path score for the progress line, or ``-`` when unscored.

    A claimed FacilityPath may carry no score at all (``--min-score 0`` admits
    unscored paths), so a ``None`` must render as a placeholder rather than
    reaching ``str.format``.
    """
    if isinstance(value, bool) or not isinstance(value, int | float):
        return "-"
    return f"{float(value):.2f}"


def _scan_progress_message(paths: list[dict[str, Any]]) -> str:
    """Build the scan progress line, rendering unscored paths as ``-``."""
    scores = [_render_score(path.get("score")) for path in paths]
    return f"scanning {len(paths)} paths (scores: {', '.join(scores[:3])}...)"


# ============================================================================
# Scan Worker
# ============================================================================


async def scan_worker(
    state: FileDiscoveryState,
    on_progress: Callable | None = None,
    batch_size: int = 10,
) -> None:
    """Scan worker: SSH file enumeration from scored FacilityPaths.

    Claims FacilityPaths via files_claimed_at, runs batched SSH file listing
    (single SSH call per batch), creates CodeFile nodes.
    """
    from imas_codex.discovery.code.graph_ops import (
        claim_paths_for_file_scan,
        mark_path_file_scanned,
        release_path_file_scan_claim,
    )
    from imas_codex.discovery.code.scanner import (
        _persist_code_files,
        async_scan_remote_paths_batch,
    )
    from imas_codex.graph import GraphClient

    # Ensure Facility node exists
    with GraphClient() as gc:
        gc.ensure_facility(state.facility)

    ssh_retry_count = 0
    max_ssh_retries = 5
    unreachable_count = 0

    while not state.should_stop():
        # Claim paths atomically
        paths = await asyncio.to_thread(
            claim_paths_for_file_scan,
            state.facility,
            min_score=state.min_score,
            limit=batch_size,
            path_prefixes=state.path_prefixes,
        )

        if not paths:
            state.scan_phase.record_idle()
            if state.scan_phase.done:
                break
            if on_progress:
                on_progress("idle", state.scan_stats, None)
            await asyncio.sleep(2.0)
            continue

        state.scan_phase.record_activity(len(paths))

        # Build path list for batched SSH scan
        path_map = {p["path"]: p for p in paths}
        path_list = [p["path"] for p in paths]

        if on_progress:
            on_progress(_scan_progress_message(paths), state.scan_stats, None)

        try:
            # Single SSH call for the entire batch
            result_map = await async_scan_remote_paths_batch(
                state.facility,
                path_list,
                ssh_host=state.ssh_host,
            )

            # Reset retry counts on success
            ssh_retry_count = 0
            unreachable_count = 0

            # Process results per path
            for path, files in result_map.items():
                if state.should_stop():
                    break

                path_info = path_map.get(path)
                path_id = path_info["id"] if path_info else None

                if files:
                    persist_result = await asyncio.to_thread(
                        _persist_code_files,
                        state.facility,
                        files,
                        source_path_id=path_id,
                    )
                    state.scan_stats.processed += persist_result.get("discovered", 0)
                    state.scan_stats.record_batch(persist_result.get("discovered", 0))

                    # Mark path as scanned with file count
                    if path_id:
                        await asyncio.to_thread(
                            mark_path_file_scanned, path_id, len(files)
                        )

                    if on_progress:
                        path_info = path_map.get(path, {})
                        on_progress(
                            f"found {persist_result.get('discovered', 0)} files",
                            state.scan_stats,
                            [
                                {
                                    "path": path,
                                    "files_found": persist_result.get("discovered", 0),
                                    "score_composite": path_info.get("score"),
                                }
                            ],
                        )
                else:
                    # Mark path as scanned even with 0 files to prevent re-scanning
                    if path_id:
                        await asyncio.to_thread(mark_path_file_scanned, path_id, 0)
                    if on_progress:
                        path_info = path_map.get(path, {})
                        on_progress(
                            "no files",
                            state.scan_stats,
                            [
                                {
                                    "path": path,
                                    "files_found": 0,
                                    "score_composite": path_info.get("score"),
                                }
                            ],
                        )

                # Release claim after processing
                if path_id:
                    await asyncio.to_thread(release_path_file_scan_claim, path_id)

        except Exception as e:
            state.scan_stats.errors += len(paths)

            # Release all claims on error
            for p in paths:
                await asyncio.to_thread(release_path_file_scan_claim, p["id"])

            if host_unreachable(e):
                unreachable_count += 1
                logger.warning(
                    "SSH scan could not reach %s (%d in a row): %s; waiting "
                    "for the host",
                    state.facility,
                    unreachable_count,
                    e,
                )
                if on_progress:
                    on_progress(
                        f"host unreachable, waiting ({unreachable_count})",
                        state.scan_stats,
                        None,
                    )
                await wait_for_reachable_host(state, unreachable_count)
                continue

            ssh_retry_count += 1
            logger.warning(
                "SSH scan failed (%d/%d): %s", ssh_retry_count, max_ssh_retries, e
            )

            if ssh_retry_count >= max_ssh_retries:
                logger.error(
                    "File scan failed %d times with the host reachable. "
                    "Scan worker stopping.",
                    max_ssh_retries,
                )
                state.scan_phase.mark_done()
                if on_progress:
                    on_progress(
                        f"SSH failed: {str(e)[:100]}",
                        state.scan_stats,
                        None,
                    )
                break

            backoff = min(2**ssh_retry_count, 32)
            if on_progress:
                on_progress(
                    f"SSH retry {ssh_retry_count} in {backoff}s",
                    state.scan_stats,
                    None,
                )
                await reachability._sleep_unless_stopped(state, backoff)
            continue

        await asyncio.sleep(0.1)


# ============================================================================
# Triage Worker
# ============================================================================


async def triage_worker(
    state: FileDiscoveryState,
    on_progress: Callable | None = None,
    batch_size: int = 50,
    concurrency: int = 8,
) -> None:
    """Triage worker: names-arm relevance judgement of discovered CodeFiles.

    Claims discovered CodeFiles, builds each file's decision state (path,
    language, directory, directory description, sibling names and the
    facility's data-access patterns) and asks the decisions model the six
    triage questions.  A file whose relevance -- the largest of the four scope
    probabilities -- reaches the triage threshold becomes status='triaged' and
    proceeds to enrichment; the rest become status='skipped' with a skip reason
    naming their role and probabilities.

    Concurrency is bounded so the decisions endpoint is never asked for more
    than ``concurrency`` judgements at once.  A decisions failure leaves the
    file at its status and unclaimed so a later pass retries it (fail closed).
    """
    import time as _time

    from imas_codex.discovery.base.facility import get_facility
    from imas_codex.discovery.base.judgment import judge_rows
    from imas_codex.discovery.code.graph_ops import (
        claim_files_for_triage,
        release_file_triage_claims,
    )
    from imas_codex.discovery.code.scorer import (
        _group_files_by_parent,
        apply_name_relevance,
        build_triage_questions,
        build_triage_state,
        triage_relevance,
    )
    from imas_codex.settings import get_code_triage_threshold, get_model

    model = get_model("discovery-relevance")
    questions = build_triage_questions()
    threshold = get_code_triage_threshold()
    try:
        facility_config = get_facility(state.facility)
    except Exception as exc:  # noqa: BLE001 - absent facility block is data, not a crash
        logger.warning("triage_worker: facility config unavailable: %s", exc)
        facility_config = {}

    while not state.should_stop():
        if state.budget_exhausted:
            if on_progress:
                on_progress("budget exhausted", state.triage_stats, None)
            break

        files = await asyncio.to_thread(
            claim_files_for_triage,
            state.facility,
            limit=batch_size,
            path_prefixes=state.path_prefixes,
        )

        if not files:
            state.triage_phase.record_idle()
            if state.triage_phase.done:
                break
            if on_progress:
                on_progress("idle", state.triage_stats, None)
            await asyncio.sleep(2.0)
            continue

        state.triage_phase.record_activity(len(files))

        file_id_map = {f["path"]: f["id"] for f in files}

        # Sibling file names give the decision neighborhood context.
        groups = _group_files_by_parent(files, include_siblings=True)
        siblings_by_parent = {
            g["parent_path_id"]: g.get("sibling_names", []) for g in groups
        }

        if on_progress:
            on_progress(
                f"triaging {len(files)} files ({len(groups)} dirs)",
                state.triage_stats,
                None,
            )

        batch_start = _time.monotonic()

        decisions: list[dict[str, Any]] = []

        def state_for(
            file: dict[str, Any], siblings_by_parent=siblings_by_parent
        ) -> dict[str, Any]:
            row = dict(file)
            row["sibling_names"] = siblings_by_parent.get(
                file.get("parent_path_id"), []
            )
            return build_triage_state(
                row, state.facility, facility_config, with_content=False
            )

        async def apply(answered, cost, decisions=decisions, file_id_map=file_id_map):
            decisions.extend(
                {"path": row["path"], "answers": answers, "model": model, "cost": paid}
                for row, answers, paid in answered
            )
            return await asyncio.to_thread(
                apply_name_relevance,
                decisions,
                file_id_map,
                threshold=threshold,
                cost_total=cost,
            )

        triaged = skipped = 0
        try:
            applied, batch_cost, failed = await judge_rows(
                files,
                state_for,
                lambda: questions,
                apply,
                model=model,
                service="facility-discovery",
                concurrency=concurrency,
            )
            if applied:
                triaged, skipped = applied["triaged"], applied["skipped"]
        except Exception as e:
            logger.error("Triage persistence failed: %s", e)
            state.triage_stats.errors += len(files)
            await asyncio.to_thread(
                release_file_triage_claims, [f["id"] for f in files]
            )
            if is_infrastructure_error(e):
                raise
            continue
        state.triage_stats.cost += batch_cost
        failed_ids = [f["id"] for f in failed]

        batch_total = triaged + skipped
        state.triage_stats.processed += batch_total
        state.triage_stats.last_batch_time = _time.monotonic() - batch_start
        if batch_total:
            state.triage_stats.record_batch(batch_total)

        if failed_ids:
            state.triage_stats.errors += len(failed_ids)
            # Fail closed: clear the claim so a later pass retries the file.
            await asyncio.to_thread(release_file_triage_claims, failed_ids)

        if on_progress:
            triage_results = []
            for d in decisions:
                relevance = triage_relevance(d["answers"])
                role = (d["answers"].get("role") or {}).get("choice") or ""
                triage_results.append(
                    {
                        "path": d["path"],
                        "relevance": round(relevance, 3),
                        "category": role,
                        "description": role,
                        "skipped": relevance < threshold,
                    }
                )
            on_progress(
                f"triaged {triaged}, skipped {skipped} (${batch_cost:.3f})",
                state.triage_stats,
                triage_results,
            )

        await asyncio.sleep(0.1)


# ============================================================================
# Score Worker
# ============================================================================


async def score_worker(
    state: FileDiscoveryState,
    on_progress: Callable | None = None,
    batch_size: int = 50,
    concurrency: int = 8,
) -> None:
    """Score worker: Content-relevance judgement and description of enriched CodeFiles.

    Claims CodeFiles that have been triaged AND enriched
    (``status='triaged'``, ``is_enriched=true``).  A content-arm decisions call
    judges each file's content relevance from its evidence and preview; the
    local model then describes only the files whose content relevance reaches
    the ingest threshold.  A file is marked ``scored`` when its content
    decision succeeded, in one write carrying the relevance fields and, for an
    admitted file, the description.  A file whose content decision failed is
    released unclaimed at its prior status, so the next pass retries it.
    Whether the file is ingested is decided by its content relevance, not by
    the scorer.
    """
    from imas_codex.discovery.base.facility import get_facility
    from imas_codex.discovery.base.judgment import judge_rows
    from imas_codex.discovery.base.llm import call_llm_structured
    from imas_codex.discovery.code.graph_ops import (
        claim_files_for_scoring,
        fetch_file_chunk_text,
        release_file_score_claims,
    )
    from imas_codex.discovery.code.scorer import (
        FileScoreBatch,
        _build_score_system_prompt,
        _build_score_user_prompt,
        _group_files_by_parent,
        apply_file_scores,
        apply_ingested_rejudge,
        apply_stale_content_rejudge,
        build_triage_questions,
        build_triage_state,
        chunk_content_head,
        content_admits,
        triage_relevance,
    )
    from imas_codex.settings import (
        get_code_facet_admission_threshold,
        get_code_ingest_threshold,
        get_model,
        get_reasoning_effort,
    )

    model = get_model("discovery-score")
    relevance_model = get_model("discovery-relevance")
    # The content arm judges a file's content relevance from its preview text,
    # so it must ask the content question set: the graded relevance and the
    # four facet depths.  The names-arm question set omits all five, and a
    # response to a set that never asked them carries no answer to record.
    questions = build_triage_questions(with_content=True)
    try:
        facility_config = get_facility(state.facility)
    except Exception as exc:  # noqa: BLE001 - absent facility block is data, not a crash
        logger.warning("score_worker: facility config unavailable: %s", exc)
        facility_config = {}

    import time as _time

    _prompt_built_at = _time.monotonic()
    _PROMPT_REBUILD_INTERVAL = 60.0

    score_system_prompt = _build_score_system_prompt(
        facility=state.facility, focus=state.focus
    )

    while not state.should_stop():
        if state.budget_exhausted:
            if on_progress:
                on_progress("budget exhausted", state.score_stats, None)
            break

        # Rebuild system prompt periodically
        if (_time.monotonic() - _prompt_built_at) > _PROMPT_REBUILD_INTERVAL:
            score_system_prompt = _build_score_system_prompt(
                facility=state.facility, focus=state.focus
            )
            _prompt_built_at = _time.monotonic()

        files = await asyncio.to_thread(
            claim_files_for_scoring,
            state.facility,
            limit=batch_size,
            path_prefixes=state.path_prefixes,
        )

        if not files:
            state.score_phase.record_idle()
            if state.score_phase.done:
                break
            if on_progress:
                on_progress("idle", state.score_stats, None)
            await asyncio.sleep(2.0)
            continue

        state.score_phase.record_activity(len(files))

        file_id_map = {f["path"]: f["id"] for f in files}
        batch_ids = [f["id"] for f in files]

        # Group by parent path (no siblings needed — enrichment provides context)
        file_groups = _group_files_by_parent(files, include_siblings=False)

        if on_progress:
            on_progress(
                f"scoring {len(files)} files ({len(file_groups)} dirs)",
                state.score_stats,
                None,
            )

        batch_start = _time.monotonic()

        try:
            ingested_ids = [f["id"] for f in files if f.get("status") == "ingested"]
            chunks_by_file = await asyncio.to_thread(
                fetch_file_chunk_text, ingested_ids
            )
            described_relevance: dict[str, float] = {}
            parsed_results = []
            description_cost = 0.0
            file_by_path = {f["path"]: f for f in files}
            ingest_threshold = get_code_ingest_threshold()
            facet_threshold = get_code_facet_admission_threshold()

            def state_for(
                file: dict[str, Any], chunks_by_file=chunks_by_file
            ) -> dict[str, Any]:
                content = (
                    chunk_content_head(chunks_by_file.get(file["id"], [])) or None
                    if file.get("status") == "ingested"
                    else None
                )
                return build_triage_state(
                    file,
                    state.facility,
                    facility_config,
                    with_content=True,
                    content_head=content,
                )

            async def apply(
                answered,
                cost,
                file_by_path=file_by_path,
                ingest_threshold=ingest_threshold,
                facet_threshold=facet_threshold,
                described_relevance=described_relevance,
                score_system_prompt=score_system_prompt,
                file_id_map=file_id_map,
            ):
                nonlocal description_cost, parsed_results
                decisions = [
                    {
                        "path": row["path"],
                        "answers": answers,
                        "model": relevance_model,
                        "cost": paid,
                    }
                    for row, answers, paid in answered
                ]
                fresh = [
                    d
                    for d in decisions
                    if file_by_path[d["path"]].get("status", "triaged") == "triaged"
                ]
                stored = [
                    d
                    for d in decisions
                    if file_by_path[d["path"]].get("status") in {"scored", "skipped"}
                ]
                ingested = [
                    d
                    for d in decisions
                    if file_by_path[d["path"]].get("status") == "ingested"
                ]
                described_files = []
                for decision in fresh:
                    if content_admits(
                        decision["answers"], ingest_threshold, facet_threshold
                    ):
                        described_files.append(file_by_path[decision["path"]])
                        described_relevance[decision["path"]] = triage_relevance(
                            decision["answers"]
                        )
                if described_files:
                    groups = _group_files_by_parent(
                        described_files, include_siblings=False
                    )
                    parsed_raw, description_cost, _tokens = await asyncio.to_thread(
                        call_llm_structured,
                        model=model,
                        messages=[
                            {"role": "system", "content": score_system_prompt},
                            {
                                "role": "user",
                                "content": _build_score_user_prompt(groups),
                            },
                        ],
                        response_model=FileScoreBatch,
                        temperature=0.1,
                        service="facility-discovery",
                        reasoning_effort=get_reasoning_effort("discovery-score"),
                    )
                    assert isinstance(parsed_raw, FileScoreBatch)
                    parsed_results = parsed_raw.results
                fresh_result = (
                    await asyncio.to_thread(
                        apply_file_scores,
                        parsed_results,
                        file_id_map,
                        fresh,
                        batch_cost=description_cost,
                        content_cost=sum(d["cost"] for d in fresh),
                    )
                    if fresh
                    else {"scored": 0}
                )
                stored_result = (
                    await asyncio.to_thread(
                        apply_stale_content_rejudge, stored, file_id_map
                    )
                    if stored
                    else {"scored": 0, "skipped": 0}
                )
                if ingested:
                    await asyncio.to_thread(
                        apply_ingested_rejudge,
                        ingested,
                        file_id_map,
                        sum(d["cost"] for d in ingested),
                    )
                return {
                    "scored": fresh_result["scored"] + stored_result["scored"],
                    "skipped": stored_result["skipped"],
                    "ingested": len(ingested),
                }

            result, content_cost, failed = await judge_rows(
                files,
                state_for,
                lambda: questions,
                apply,
                model=relevance_model,
                service="facility-discovery",
                concurrency=concurrency,
            )
            state.score_stats.cost += content_cost + description_cost
            if failed:
                logger.info(
                    "content decision failed for %d of %d files",
                    len(failed),
                    len(files),
                )
                state.score_stats.errors += len(failed)
            result = result or {"scored": 0, "skipped": 0, "ingested": 0}
            batch_total = sum(result.values())
            state.score_stats.processed += batch_total
            state.score_stats.last_batch_time = _time.monotonic() - batch_start
            state.score_stats.record_batch(batch_total)

            await asyncio.to_thread(release_file_score_claims, batch_ids)

            if on_progress:
                # Stream the descriptions written for the admitted files.
                score_results = [
                    {
                        "path": r.path,
                        "score_composite": round(
                            described_relevance.get(r.path, 0.0), 3
                        ),
                        "category": "",
                        "description": r.description,
                        "skipped": False,
                    }
                    for r in parsed_results
                ]
                on_progress(
                    f"scored {result.get('scored', 0)} "
                    f"(${description_cost + content_cost:.3f})",
                    state.score_stats,
                    score_results,
                )

        except Exception as e:
            logger.error("Score batch failed: %s", e)
            state.score_stats.errors += 1
            await asyncio.to_thread(release_file_score_claims, batch_ids)
            if is_infrastructure_error(e):
                raise

        await asyncio.sleep(0.1)


# ============================================================================
# Re-judge worker (ingested files, from stored chunks)
# ============================================================================


async def rejudge_ingested_files(
    facility: str,
    *,
    path_prefixes: list[str] | None = None,
    batch_size: int = 10,
    concurrency: int = 8,
    cost_limit: int | float | None = None,
    on_progress: Callable | None = None,
) -> dict[str, Any]:
    """Re-judge ingested content-stage CodeFiles from their stored chunk text.

    The files this pass takes sit at ``status='ingested'`` with
    ``relevance_stage='content'`` and no recorded facet answer: their text is
    already in the graph as CodeChunks, so the pass rebuilds each file's content
    state from those chunks -- ordered by reading position and cut to the same
    length the fetched path uses -- and asks the content arm the same question
    set through the same seat.  The judgment fields are written back through the
    score arm's writer; the file's status, its CodeExample and its chunks are
    left as they are.  No file is fetched over the facility hop, and admission
    is not re-decided: a file whose new composite and facets both fall below
    their gates stays ``ingested`` and is reported.

    Returns a dict with ``rejudged``, the ``below_gate`` paths, the decision
    ``cost`` and the number of ``batches``.
    """
    from imas_codex.discovery.base.facility import get_facility
    from imas_codex.discovery.base.judgment import judge_rows
    from imas_codex.discovery.code.graph_ops import (
        claim_files_for_scoring,
        fetch_file_chunk_text,
        release_file_score_claims,
    )
    from imas_codex.discovery.code.scorer import (
        apply_ingested_rejudge,
        build_triage_questions,
        build_triage_state,
        chunk_content_head,
    )
    from imas_codex.settings import get_model

    model = get_model("discovery-relevance")
    questions = build_triage_questions(with_content=True)
    try:
        facility_config = get_facility(facility)
    except Exception as exc:  # noqa: BLE001 - absent facility block is data, not a crash
        logger.warning("rejudge_ingested_files: facility config unavailable: %s", exc)
        facility_config = {}

    rejudged = 0
    below_gate: list[str] = []
    spend = 0.0
    batches = 0
    attempted: set[str] = set()

    while batch := await asyncio.to_thread(
        claim_files_for_scoring,
        facility,
        limit=batch_size,
        path_prefixes=path_prefixes,
        ingested_rejudge=True,
    ):
        batch_ids = [f["id"] for f in batch]
        # A decision that failed leaves its file eligible for the same claim
        # (its facet confidence is still zero), so a later batch can mix files
        # this run already attempted with fresh ones.  Take only the fresh
        # files: the attempted ones are dropped from the batch, so no file is
        # judged twice in one pass and ``rejudged`` counts the distinct files
        # judged rather than the repeats.  A batch that brings back only
        # attempted files has no fresh work left, so release it and stop.
        fresh = [f for f in batch if f["id"] not in attempted]
        if not fresh:
            await asyncio.to_thread(release_file_score_claims, batch_ids)
            break
        attempted.update(f["id"] for f in fresh)
        batch = fresh
        file_id_map = {f["path"]: f["id"] for f in batch}
        batches += 1

        chunks_by_file = await asyncio.to_thread(
            fetch_file_chunk_text, [f["id"] for f in batch]
        )

        def state_for(
            file: dict[str, Any], chunks_by_file=chunks_by_file
        ) -> dict[str, Any]:
            return build_triage_state(
                file,
                facility,
                facility_config,
                with_content=True,
                content_head=chunk_content_head(chunks_by_file.get(file["id"], []))
                or None,
            )

        async def apply(answered, cost, file_id_map=file_id_map):
            decisions = [
                {"path": row["path"], "answers": answers, "model": model, "cost": paid}
                for row, answers, paid in answered
            ]
            return await asyncio.to_thread(
                apply_ingested_rejudge, decisions, file_id_map, cost
            )

        applied, batch_cost, failed_rows = await judge_rows(
            batch,
            state_for,
            lambda: questions,
            apply,
            model=model,
            service="facility-discovery",
            concurrency=concurrency,
        )
        spend += batch_cost
        failed = len(failed_rows)
        if applied:
            rejudged += applied["rejudged"]
            below_gate.extend(applied["below_gate"])

        await asyncio.to_thread(release_file_score_claims, batch_ids)

        if on_progress:
            on_progress(
                f"re-judged {rejudged} ingested files "
                f"({failed} failed this batch, ${spend:.3f})"
            )

        if cost_limit is not None and spend >= cost_limit:
            logger.info(
                "re-judge stopped at the cost limit ($%.3f >= $%.3f)",
                spend,
                cost_limit,
            )
            break

    return {
        "rejudged": rejudged,
        "below_gate": below_gate,
        "cost": spend,
        "batches": batches,
    }


# ============================================================================
# Code Worker (ingestion)
# ============================================================================


@retry_on_deadlock()
def _claim_code_files_for_ingestion(
    facility: str,
    limit: int = 20,
    min_relevance: float | None = None,
    min_facet_relevance: float | None = None,
    max_line_count: int = 10000,
    path_prefixes: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Claim scored CodeFiles for ingestion.

    Claims CodeFiles with status='scored' whose content decision admits them —
    the largest of the four scope probabilities reaches the ingest threshold,
    or the strongest facet reaches the facet threshold — and claims the highest
    relevance first.  Only a file whose relevance came from the content arm
    (``relevance_stage='content'``) is eligible, so a name-arm relevance can
    never carry a file into ingestion.  Skips files exceeding max_line_count
    to avoid tree-sitter hangs on very large auto-generated files.

    Dedup is handled *after* claiming — see ``_filter_duplicates()``.
    Keeping the claim query simple avoids expensive correlated subqueries
    that scale as O(candidates × ingested_hashes).

    When ``path_prefixes`` is given, only CodeFiles whose ``path`` starts with
    one of the prefixes are claimed, so a scoped run ingests only the named
    trees.

    Uses a claim_token two-step verify and @retry_on_deadlock decorator.
    """
    if min_relevance is None:
        from imas_codex.settings import get_code_ingest_threshold

        min_relevance = get_code_ingest_threshold()
    if min_facet_relevance is None:
        from imas_codex.settings import get_code_facet_admission_threshold

        min_facet_relevance = get_code_facet_admission_threshold()
    import uuid

    from imas_codex.config.discovery_config import build_facility_exclusion_filter
    from imas_codex.discovery.base.claims import DEFAULT_CLAIM_TIMEOUT_SECONDS
    from imas_codex.discovery.code.scorer import (
        RELEVANCE_STAGE_CONTENT,
        relevance_predicate,
    )
    from imas_codex.graph import GraphClient
    from imas_codex.graph.query_builder import build_path_prefix_filter

    prefix_clause, prefix_params = build_path_prefix_filter("sf", path_prefixes)
    excluded_clause, excluded_params = build_facility_exclusion_filter(facility, "sf")
    token = str(uuid.uuid4())
    cutoff = f"PT{DEFAULT_CLAIM_TIMEOUT_SECONDS}S"
    with GraphClient() as gc:
        # Serialize selection per facility. The token readback alone cannot
        # prevent two transactions from selecting the same unclaimed rows:
        # each can read its own token before the other transaction overwrites it.
        # The dependent write to the facility's existing name takes a lock
        # before candidates are read, without changing the facility record.
        gc.query(
            f"""
            MATCH (f:Facility {{id: $facility}})
            SET f.name = f.name
            WITH f
            MATCH (sf:CodeFile)-[:AT_FACILITY]->(f)
            WHERE sf.status = 'scored'
              AND {relevance_predicate("sf", RELEVANCE_STAGE_CONTENT, "$min_relevance", "$min_facet_relevance")}
              AND coalesce(sf.line_count, 0) <= $max_line_count
              {prefix_clause}
              {excluded_clause}
              AND (sf.claimed_at IS NULL
                   OR sf.claimed_at < datetime() - duration($cutoff))
            WITH sf, sf.score_composite AS relevance
            ORDER BY relevance DESC, rand()
            LIMIT $limit
            SET sf.claimed_at = datetime(), sf.claim_token = $token
            """,
            facility=facility,
            min_relevance=min_relevance,
            min_facet_relevance=min_facet_relevance,
            max_line_count=max_line_count,
            limit=limit,
            cutoff=cutoff,
            token=token,
            **prefix_params,
            **excluded_params,
        )
        # Read back by token to confirm ownership after the claim commits.
        result = gc.query(
            """
            MATCH (sf:CodeFile {claim_token: $token})
            RETURN sf.id AS id, sf.path AS path, sf.language AS language,
                   sf.score_composite AS score_composite,
                   sf.content_hash AS content_hash, sf.claim_token AS claim_token
            """,
            token=token,
        )
        return list(result)


def _refresh_ingestion_claims(claims: list[dict[str, str]]) -> None:
    """Extend only claims still owned by this ingestion batch."""
    from imas_codex.graph import GraphClient

    with GraphClient() as gc:
        result = gc.query(
            """
            UNWIND $claims AS claim
            MATCH (sf:CodeFile {id: claim.id})
            WHERE sf.claim_token = claim.token
            SET sf.claimed_at = CASE
                WHEN sf.status = 'scored' THEN datetime()
                ELSE sf.claimed_at
            END
            RETURN count(sf) AS owned
            """,
            claims=claims,
        )
    if not result or result[0]["owned"] != len(claims):
        raise RuntimeError("Code ingestion claim changed while its holder was active")


async def _keep_ingestion_claims_current(
    files: list[dict[str, Any]], interval: float | None = None
) -> None:
    """Keep a working batch from being reclaimed by the stale-claim cutoff."""
    from imas_codex.discovery.base.claims import DEFAULT_CLAIM_TIMEOUT_SECONDS

    claims = [{"id": file["id"], "token": file["claim_token"]} for file in files]
    delay = interval if interval is not None else DEFAULT_CLAIM_TIMEOUT_SECONDS / 3
    while True:
        await asyncio.sleep(delay)
        await asyncio.to_thread(_refresh_ingestion_claims, claims)


def _settle_unclaimable_files(
    facility: str,
    max_line_count: int = 10000,
    path_prefixes: list[str] | None = None,
) -> int:
    """Mark admitted files the ingestion claim can never reach as skipped.

    A file the claim would otherwise admit — content-stage, above the ingest
    relevance floor, inside the run's path scope and outside the facility's
    exclusion prefixes — is excluded by the claim's
    ``coalesce(sf.line_count, 0) <= $max_line_count`` predicate when it is
    oversized, so it waits at ``scored`` forever.  This gives exactly those rows
    a terminal ``skipped`` state with the reason.

    Returns the number of files settled.
    """
    from imas_codex.config.discovery_config import build_facility_exclusion_filter
    from imas_codex.discovery.code.scorer import (
        RELEVANCE_STAGE_CONTENT,
        relevance_predicate,
    )
    from imas_codex.graph import GraphClient
    from imas_codex.graph.query_builder import build_path_prefix_filter
    from imas_codex.settings import (
        get_code_facet_admission_threshold,
        get_code_ingest_threshold,
    )

    prefix_clause, prefix_params = build_path_prefix_filter("sf", path_prefixes)
    excluded_clause, excluded_params = build_facility_exclusion_filter(facility, "sf")
    with GraphClient() as gc:
        result = gc.query(
            f"""
            MATCH (sf:CodeFile)-[:AT_FACILITY]->(f:Facility {{id: $facility}})
            WHERE sf.status = 'scored'
              AND {relevance_predicate("sf", RELEVANCE_STAGE_CONTENT, "$min_relevance", "$min_facet_relevance")}
              AND coalesce(sf.line_count, 0) > $max_line_count
              {prefix_clause}
              {excluded_clause}
            SET sf.status = 'skipped',
                sf.skip_reason = 'exceeds max_line_count'
            RETURN count(sf) AS settled
            """,
            facility=facility,
            min_relevance=get_code_ingest_threshold(),
            min_facet_relevance=get_code_facet_admission_threshold(),
            max_line_count=max_line_count,
            **prefix_params,
            **excluded_params,
        )
        return result[0]["settled"] if result else 0


def _filter_duplicates(files: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Filter out files whose content_hash is already ingested.

    This is a cheap post-claim dedup check: collects the unique hashes
    from the claimed batch, does a single ``IN $hashes`` lookup against
    ingested CodeFiles, and returns only the files that need processing.

    Files filtered out are immediately marked 'skipped' so they won't
    be claimed again.

    Returns:
        List of files that still need ingestion.
    """
    from imas_codex.graph import GraphClient

    # Collect hashes from the batch (skip files without a hash)
    hashed = {f["content_hash"]: f for f in files if f.get("content_hash")}
    if not hashed:
        return files  # nothing to dedup

    with GraphClient() as gc:
        # Single indexed lookup: which of these hashes are already ingested?
        result = gc.query(
            """
            UNWIND $hashes AS h
            MATCH (dup:CodeFile {content_hash: h})
            WHERE dup.status = 'ingested'
            RETURN DISTINCT h AS hash
            """,
            hashes=list(hashed.keys()),
        )
        already_ingested = {r["hash"] for r in result}

    if not already_ingested:
        return files

    # Split into keep vs skip
    keep = []
    skip_ids = []
    for f in files:
        h = f.get("content_hash")
        if h and h in already_ingested:
            skip_ids.append(f["id"])
        else:
            keep.append(f)

    # Mark skipped files so they aren't reclaimed
    if skip_ids:
        with GraphClient() as gc:
            gc.query(
                """
                UNWIND $ids AS fid
                MATCH (sf:CodeFile {id: fid})
                SET sf.status = 'skipped',
                    sf.skip_reason = 'content already ingested',
                    sf.claimed_at = null,
                    sf.claim_token = null
                """,
                ids=skip_ids,
            )

    return keep


def _mark_files_ingested(file_ids: list[str]) -> int:
    """Mark CodeFiles as ingested after successful processing.

    Also marks content-identical duplicates (same content_hash) as
    skipped, since their content is now represented by the ingested copy.
    """
    from imas_codex.graph import GraphClient

    if not file_ids:
        return 0
    with GraphClient() as gc:
        result = gc.query(
            """
            UNWIND $ids AS fid
            MATCH (sf:CodeFile {id: fid})
            WHERE sf.status <> 'failed'
            SET sf.status = 'ingested',
                sf.ingested_at = datetime(),
                sf.claimed_at = null,
                sf.claim_token = null
            RETURN count(sf) AS updated
            """,
            ids=file_ids,
        )

        # Mark content-identical duplicates as skipped
        gc.query(
            """
            UNWIND $ids AS fid
            MATCH (sf:CodeFile {id: fid})
            WHERE sf.content_hash IS NOT NULL
            WITH sf
            MATCH (dup:CodeFile {content_hash: sf.content_hash})
            WHERE dup.id <> sf.id AND dup.status IN ['scored', 'triaged']
            SET dup.status = 'skipped',
                dup.skip_reason = 'duplicate of ' + sf.id,
                dup.claimed_at = null,
                dup.claim_token = null
            """,
            ids=file_ids,
        )

        return result[0]["updated"] if result else 0


def _mark_file_failed(file_id: str, error: str, claim_token: str) -> None:
    """Fail only a scored CodeFile still owned by this ingestion batch."""
    from imas_codex.graph import GraphClient

    with GraphClient() as gc:
        gc.query(
            """
            MATCH (sf:CodeFile {id: $id})
            WHERE sf.status = 'scored' AND sf.claim_token = $token
            SET sf.status = 'failed',
                sf.error = $error,
                sf.claimed_at = null,
                sf.claim_token = null
            """,
            id=file_id,
            error=error[:200],
            token=claim_token,
        )


def _mark_file_skipped(file_id: str, reason: str) -> None:
    """Mark a single CodeFile as skipped with a reason.

    A skipped file is terminal: it will not be reclaimed, and the reason records
    why no example was written for it.
    """
    from imas_codex.graph import GraphClient

    with GraphClient() as gc:
        gc.query(
            """
            MATCH (sf:CodeFile {id: $id})
            SET sf.status = 'skipped',
                sf.skip_reason = $reason,
                sf.claimed_at = null,
                sf.claim_token = null
            """,
            id=file_id,
            reason=reason[:200],
        )


async def code_worker(
    state: FileDiscoveryState,
    on_progress: Callable | None = None,
    batch_size: int = 10,
) -> None:
    """Code worker: Fetch, chunk, and link code files.

    Claims scored CodeFiles, runs the ingestion pipeline (tree-sitter
    chunking, entity extraction, graph writes).  Embedding is deferred
    to the ``embed_text_worker`` which populates embeddings
    asynchronously on the GPU.
    Transitions: scored → ingested | failed

    Uses small claim batches (default 10) so progress is reported
    frequently — this keeps the streamer display flowing and the
    rate calculation accurate.
    """
    import time as _time

    from imas_codex.ingestion.pipeline import ingest_files

    logger.info(
        "code_worker started (facility=%s, batch_size=%d, scan_only=%s, score_only=%s)",
        state.facility,
        batch_size,
        state.scan_only,
        state.score_only,
    )

    idle_log_interval = 10  # log every Nth consecutive idle poll
    # Give admitted files the claim can never reach a terminal state before the
    # first poll, so the pass settles rather than waiting on them.  Only an
    # ingest pass settles: a scan or score pass is not the stage that owns them.
    if not (state.scan_only or state.score_only):
        settled = await asyncio.to_thread(
            _settle_unclaimable_files,
            state.facility,
            10000,
            state.path_prefixes,
        )
        if settled:
            logger.info(
                "Settled %d admitted files as skipped (exceeds max_line_count)",
                settled,
            )
    consecutive_idle = 0
    batches_processed = 0

    while not state.should_stop():
        if state.scan_only or state.score_only:
            logger.info(
                "code_worker exiting: scan_only=%s, score_only=%s",
                state.scan_only,
                state.score_only,
            )
            break

        # Claim code files for ingestion (wrapped in try/except to survive
        # transient Neo4j errors without crashing the worker)
        try:
            files = await asyncio.to_thread(
                _claim_code_files_for_ingestion,
                state.facility,
                limit=batch_size,
                path_prefixes=state.path_prefixes,
            )
        except Exception as e:
            logger.warning("Code claim failed: %s", e)
            if is_infrastructure_error(e):
                raise
            await asyncio.sleep(2.0)
            continue

        if not files:
            consecutive_idle += 1
            state.code_phase.record_idle()
            if state.code_phase.done:
                logger.info(
                    "code_worker exiting: phase done after %d batches "
                    "(%d files processed, %d errors)",
                    batches_processed,
                    state.code_stats.processed,
                    state.code_stats.errors,
                )
                break
            if consecutive_idle == 1 or consecutive_idle % idle_log_interval == 0:
                logger.debug(
                    "code_worker idle (poll #%d, phase.idle=%s, "
                    "score_phase.done=%s, processed=%d)",
                    consecutive_idle,
                    state.code_phase.idle,
                    state.score_phase.done,
                    state.code_stats.processed,
                )
            if on_progress:
                on_progress("idle", state.code_stats, None)
            await asyncio.sleep(3.0)
            continue

        # Post-claim dedup: filter out files whose content is already ingested.
        # This is O(batch_size) with index rather than O(candidates × ingested)
        # in the claim query itself.
        try:
            files = await asyncio.to_thread(_filter_duplicates, files)
        except Exception as e:
            logger.warning("Dedup filter failed (proceeding with full batch): %s", e)
            if is_infrastructure_error(e):
                raise

        if not files:
            # Entire batch was duplicates — count as activity but skip processing
            state.code_phase.record_activity(0)
            continue

        consecutive_idle = 0
        state.code_phase.record_activity(len(files))

        if on_progress:
            on_progress(f"ingesting {len(files)} code files", state.code_stats, None)

        remote_paths = [f["path"] for f in files]
        scores = [f.get("score_composite", 0) for f in files]

        logger.info(
            "code_worker claimed %d files (scores %.2f–%.2f): %s",
            len(files),
            min(scores),
            max(scores),
            ", ".join(f["path"].rsplit("/", 1)[-1] for f in files[:3])
            + ("..." if len(files) > 3 else ""),
        )

        batch_start = _time.monotonic()

        try:
            keepalive = asyncio.create_task(_keep_ingestion_claims_current(files))
            try:
                # Ingestion makes synchronous graph calls. Run its event loop
                # off this worker's loop so claim renewal can keep running.
                ingest_stats = await asyncio.to_thread(
                    asyncio.run,
                    ingest_files(
                        facility=state.facility,
                        remote_paths=remote_paths,
                        force=False,
                    ),
                )
                if keepalive.done():
                    keepalive.result()
            finally:
                keepalive.cancel()
                await asyncio.gather(keepalive, return_exceptions=True)

            batch_elapsed = _time.monotonic() - batch_start

            ingested_count = ingest_stats.get("files", 0)
            skipped_count = ingest_stats.get("skipped", 0)
            chunks_count = ingest_stats.get("chunks", 0)
            outcomes = ingest_stats.get("outcomes", {})

            # Each file carries its own outcome, so one file's write failure
            # fails only that file.  A claimed file with no outcome is a
            # contract violation and is marked failed rather than silently
            # ingesting.
            ingested_ids: list[str] = []
            for f in files:
                outcome = outcomes.get(f["path"])
                if outcome is None:
                    await asyncio.to_thread(
                        _mark_file_failed,
                        f["id"],
                        "no ingestion outcome recorded",
                        f["claim_token"],
                    )
                elif outcome["status"] == "ingested":
                    ingested_ids.append(f["id"])
                elif outcome["status"] == "skipped":
                    await asyncio.to_thread(
                        _mark_file_skipped, f["id"], outcome.get("reason", "no chunks")
                    )
                else:
                    await asyncio.to_thread(
                        _mark_file_failed,
                        f["id"],
                        outcome.get("reason", "unknown failure"),
                        f["claim_token"],
                    )
            if ingested_ids:
                await asyncio.to_thread(_mark_files_ingested, ingested_ids)

            batch_total = ingested_count + skipped_count
            batches_processed += 1
            state.code_stats.processed += batch_total
            state.code_stats.last_batch_time = batch_elapsed
            state.code_stats.record_batch(batch_total)

            logger.info(
                "code_worker batch #%d: ingested=%d skipped=%d chunks=%d elapsed=%.1fs",
                batches_processed,
                ingested_count,
                skipped_count,
                chunks_count,
                batch_elapsed,
            )

            if on_progress:
                avg_chunks = chunks_count // max(ingested_count, 1)
                on_progress(
                    f"ingested {ingested_count}, {chunks_count} chunks",
                    state.code_stats,
                    [
                        {
                            "path": f["path"],
                            "language": f.get("language", ""),
                            "score_composite": f.get("score_composite"),
                            "chunks": avg_chunks,
                            "file_type": "code",
                        }
                        for f in files
                    ],
                )

        except Exception as e:
            if is_infrastructure_error(e):
                logger.warning(
                    "Code ingestion batch hit infrastructure failure (%d files): %s",
                    len(files),
                    e,
                )
                raise
            logger.error("Code ingestion batch failed (%d files): %s", len(files), e)
            state.code_stats.errors += 1
            # Mark individual files as failed
            for f in files:
                await asyncio.to_thread(
                    _mark_file_failed, f["id"], str(e)[:200], f["claim_token"]
                )

        await asyncio.sleep(0.1)

    logger.info(
        "code_worker stopped (facility=%s, batches=%d, "
        "processed=%d, errors=%d, should_stop=%s)",
        state.facility,
        batches_processed,
        state.code_stats.processed,
        state.code_stats.errors,
        state.should_stop(),
    )


# ============================================================================
# Enrich Worker (rg pattern matching on individual files)
# ============================================================================


async def enrich_worker(
    state: FileDiscoveryState,
    on_progress: Callable | None = None,
    batch_size: int = 100,
) -> None:
    """Enrich worker: rg pattern matching + preview extraction on triaged files.

    Claims triaged CodeFiles above the triage composite threshold,
    runs batched rg pattern matching and preview text extraction
    via SSH.  Stores pattern evidence on CodeFile nodes; preview text
    is NOT persisted but is available for the subsequent score worker
    via the graph claim query.

    Runs AFTER triage, BEFORE scoring.
    """
    import time as _time  # noqa: PLC0415

    from imas_codex.discovery.code.enrichment import (
        enrich_files,
        persist_file_enrichment,
    )
    from imas_codex.discovery.code.graph_ops import (
        claim_files_for_enrichment,
        release_file_enrich_claims,
    )

    while not state.should_stop():
        if state.scan_only:
            break

        files = await asyncio.to_thread(
            claim_files_for_enrichment,
            state.facility,
            limit=batch_size,
            min_relevance=state.min_relevance,
            path_prefixes=state.path_prefixes,
        )

        if not files:
            state.enrich_phase.record_idle()
            if state.enrich_phase.done:
                break
            if on_progress:
                on_progress("idle", state.enrich_stats, None)
            await asyncio.sleep(2.0)
            continue

        state.enrich_phase.record_activity(len(files))
        file_id_map = {f["path"]: f["id"] for f in files}
        batch_ids = [f["id"] for f in files]
        file_paths = [f["path"] for f in files]

        if on_progress:
            on_progress(
                f"enriching {len(files)} files",
                state.enrich_stats,
                None,
            )

        batch_start = _time.monotonic()

        try:
            results = await enrich_files(
                state.facility,
                file_paths,
            )

            enrich_counts = await asyncio.to_thread(
                persist_file_enrichment, results, file_id_map
            )
            enriched = enrich_counts["enriched"]
            failed = enrich_counts["failed"]
            processed = enriched + failed

            state.enrich_stats.processed += processed
            state.enrich_stats.last_batch_time = _time.monotonic() - batch_start
            state.enrich_stats.record_batch(processed)

            # Release claims
            await asyncio.to_thread(release_file_enrich_claims, batch_ids)

            if on_progress:
                # Stream enriched files with line count + pattern categories + preview
                enrich_results = []
                for r in results:
                    cats = r.get("pattern_categories", {})
                    # Find top pattern categories by count
                    top_cats = (
                        sorted(cats.items(), key=lambda x: x[1], reverse=True)[:4]
                        if cats
                        else []
                    )
                    # Extract a meaningful preview snippet (first non-blank, non-comment line)
                    preview = r.get("preview_text", "")
                    snippet = ""
                    if preview:
                        for line in preview.splitlines():
                            stripped = line.strip()
                            if stripped and not stripped.startswith(
                                ("#", "//", "/*", "*", "!", "C ", "c ")
                            ):
                                snippet = stripped[:80]
                                break
                    # Look up relevance from the claim data
                    file_id = file_id_map.get(r["path"])
                    relevance = None
                    for f in files:
                        if f["id"] == file_id:
                            relevance = f.get("relevance")
                            break
                    enrich_results.append(
                        {
                            "path": r["path"],
                            "relevance": relevance,
                            "patterns": r.get("total_pattern_matches", 0),
                            "line_count": r.get("line_count", 0),
                            "pattern_categories": dict(top_cats),
                            "preview_snippet": snippet,
                        }
                    )
                on_progress(
                    f"enriched {enriched}, failed {failed}",
                    state.enrich_stats,
                    enrich_results,
                )

        except Exception as e:
            logger.error("File enrichment batch failed: %s", e)
            state.enrich_stats.errors += 1
            await asyncio.to_thread(release_file_enrich_claims, batch_ids)
            if is_infrastructure_error(e):
                raise

        await asyncio.sleep(0.1)


# ============================================================================
# Link Worker (code evidence → signal propagation)
# ============================================================================


async def link_worker(
    state: FileDiscoveryState,
    on_progress: Callable | None = None,
) -> None:
    """Link worker: Propagate code evidence to FacilitySignals.

    After code ingestion creates DataReference → SignalNode links, this
    worker propagates evidence to FacilitySignals via the chain:
      DataReference → RESOLVES_TO_NODE → SignalNode ← HAS_DATA_SOURCE_NODE ← FacilitySignal

    Sets code_evidence_count and has_code_evidence on matched signals.
    Runs periodically while code workers are active, then one final pass.
    """
    from imas_codex.discovery.code.graph_ops import (
        has_pending_link_work,
        link_code_evidence_to_signals,
    )

    last_linked = 0

    while not state.should_stop():
        if state.scan_only or state.score_only:
            break

        has_work = await asyncio.to_thread(has_pending_link_work, state.facility)

        if not has_work:
            # If code phase is done, we're done too
            if state.code_phase.done:
                break
            await asyncio.sleep(5.0)
            continue

        if on_progress:
            on_progress("linking code evidence to signals", state.link_stats, None)

        try:
            result = await asyncio.to_thread(
                link_code_evidence_to_signals, state.facility
            )

            signals_linked = result.get("signals_linked", 0)
            refs_resolved = result.get("refs_resolved", 0)
            state.link_stats.processed += signals_linked
            last_linked = signals_linked

            if on_progress:
                on_progress(
                    f"linked {signals_linked} signals ({refs_resolved} refs resolved)",
                    state.link_stats,
                    None,
                )

        except Exception as e:
            logger.error("Code evidence linking failed: %s", e)
            state.link_stats.errors += 1
            if is_infrastructure_error(e):
                raise

        # Link is cheap, run every 10s
        await asyncio.sleep(10.0)

    # Final pass after all code ingestion is done.
    # Skip it when the graph is already drained; otherwise the worker can
    # spend minutes in a no-op relinking query after the UI has gone idle.
    if not (state.scan_only or state.score_only):
        try:
            final_has_work = await asyncio.to_thread(
                has_pending_link_work, state.facility
            )
            if final_has_work:
                result = await asyncio.to_thread(
                    link_code_evidence_to_signals, state.facility
                )
                final_linked = result.get("signals_linked", 0)
                if final_linked > last_linked and on_progress:
                    on_progress(
                        f"final link: {final_linked} signals",
                        state.link_stats,
                        None,
                    )
        except Exception as e:
            logger.error("Final code evidence linking failed: %s", e)
            if is_infrastructure_error(e):
                raise
