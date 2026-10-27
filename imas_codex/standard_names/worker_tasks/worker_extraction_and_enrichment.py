"""Extraction, enrichment, attachment, and comparator tasks."""

from __future__ import annotations

import logging
import re as _re
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from imas_codex.standard_names.budget import BudgetLease, BudgetManager
    from imas_codex.standard_names.state import StandardNameBuildState


async def extract_worker(
    _worker_dependency, /, state: StandardNameBuildState, **_kwargs
) -> None:
    """Extract candidate quantities from graph entities into batches.

    For DD source: queries IMASNode paths, groups by cluster/IDS/prefix.
    Skips sources already linked via HAS_STANDARD_NAME unless --force.
    Stores ExtractionBatch objects in ``state.extracted``.
    """
    from imas_codex.cli.logging import WorkerLogAdapter

    wlog = WorkerLogAdapter(logger, worker_name="sn_extract_worker")
    wlog.info("Starting extraction (source=%s)", state.source)

    def _on_status(text: str) -> None:
        state.extract_stats.status_text = text
        if state.loop_extract_stats is not None:
            state.loop_extract_stats.status_text = text
            try:
                state.loop_extract_stats.stream_queue.add(
                    [
                        {
                            "primary_text": "extract",
                            "primary_text_style": "dim cyan",
                            "description": text,
                        }
                    ]
                )
            except Exception:
                pass

    def _run() -> list:
        from imas_codex.standard_names.graph_store.graph_claim_scope import (
            get_existing_standard_names,
            get_named_source_ids,
        )
        from imas_codex.standard_names.sources.dd import extract_dd_candidates

        _on_status("loading existing names…")
        existing = get_existing_standard_names()
        if state.is_regen_mode() and state.source == "dd":
            from imas_codex.standard_names.graph_store.graph_lineage import (
                fetch_low_score_sources,
            )
            from imas_codex.standard_names.sources.dd import extract_specific_paths

            _on_status(f"loading sources below reviewer_score={state.min_score}…")
            regen_sources = fetch_low_score_sources(
                min_score=state.min_score,
                domain=state.domain_filter,
                ids=state.ids_filter,
                limit=state.limit,
                source_type="dd",
            )
            if not regen_sources:
                wlog.info(
                    "Regen mode (--min-score %s): no low-score sources found (domain=%s, ids=%s, limit=%s)",
                    state.min_score,
                    state.domain_filter,
                    state.ids_filter,
                    state.limit,
                )
                return []
            wlog.info(
                "Regen mode: %d sources below reviewer_score=%s (domain=%s, ids=%s, limit=%s)",
                len(regen_sources),
                state.min_score,
                state.domain_filter,
                state.ids_filter,
                state.limit,
            )
            paths = [r["source_id"] for r in regen_sources]
            batches = extract_specific_paths(
                paths=paths,
                existing_names=existing,
                on_status=_on_status,
                write_side_effects=not state.dry_run,
            )
            return batches
        named_ids: set[str] = set()
        if not state.force:
            named_ids = get_named_source_ids()
            if named_ids:
                wlog.info("Skipping %d already-named sources", len(named_ids))
        if state.source == "dd":
            from imas_codex.standard_names.batching import get_generate_batch_config

            batch_cfg = get_generate_batch_config()
            batches = extract_dd_candidates(
                ids_filter=state.ids_filter,
                domain_filter=state.domain_filter,
                limit=state.limit,
                existing_names=existing,
                on_status=_on_status,
                from_model=state.from_model,
                force=state.force,
                name_only=state.name_only,
                name_only_batch_size=state.name_only_batch_size,
                max_batch_size=batch_cfg["batch_size"],
                max_tokens=batch_cfg["max_tokens"],
                write_skipped=not state.dry_run,
            )
        else:
            wlog.error("Unknown source: %s", state.source)
            return []
        if named_ids and (not state.force):
            for batch in batches:
                batch.items = [
                    item
                    for item in batch.items
                    if item.get("path", item.get("signal_id")) not in named_ids
                ]
            batches = [b for b in batches if b.items]
        return batches

    batches = await _worker_dependency("asyncio").to_thread(_run)
    if state.force:
        use_rich = False

        def _get_mapping():
            from imas_codex.standard_names.graph_store.graph_lineage import (
                get_source_name_mapping,
            )

            return get_source_name_mapping(rich=use_rich)

        source_names = await _worker_dependency("asyncio").to_thread(_get_mapping)
        injected = 0
        for batch in batches:
            for item in batch.items:
                path = item.get("path", item.get("signal_id"))
                if path and path in source_names:
                    item["previous_name"] = source_names[path]
                    injected += 1
        if injected:
            wlog.info("Injected previous_name context for %d items", injected)

    def _get_feedback():
        from imas_codex.standard_names.graph_store.graph_lineage import (
            fetch_review_feedback_for_sources,
        )

        ids: set[str] = set()
        for batch in batches:
            for item in batch.items:
                path = item.get("path", item.get("signal_id"))
                if path:
                    ids.add(path)
        return fetch_review_feedback_for_sources(ids)

    feedback_map = await _worker_dependency("asyncio").to_thread(_get_feedback)
    fb_injected = 0
    for batch in batches:
        for item in batch.items:
            path = item.get("path", item.get("signal_id"))
            if path and path in feedback_map:
                item["review_feedback"] = feedback_map[path]
                fb_injected += 1
    if fb_injected:
        wlog.info("Injected review_feedback for %d items", fb_injected)

    def _get_reviewer_history():
        from imas_codex.standard_names.graph_store.graph_lineage import (
            fetch_reviewer_history_for_sources,
        )

        ids: set[str] = set()
        for batch in batches:
            for item in batch.items:
                path = item.get("path", item.get("signal_id"))
                if path:
                    ids.add(path)
        return fetch_reviewer_history_for_sources(ids)

    history_map = await _worker_dependency("asyncio").to_thread(_get_reviewer_history)
    hist_injected = 0
    for batch in batches:
        for item in batch.items:
            path = item.get("path", item.get("signal_id"))
            if path and path in history_map:
                item["reviewer_history"] = history_map[path]
                hist_injected += 1
    if hist_injected:
        wlog.info("Injected reviewer_history for %d items", hist_injected)
    if not state.dry_run and batches:
        from imas_codex.settings import get_dd_version
        from imas_codex.standard_names.graph_store.graph_source_bindings import (
            merge_standard_name_sources,
        )

        _dd_ver = get_dd_version()
        sources = []
        source_type = "dd" if state.source == "dd" else "signals"
        for batch in batches:
            for item in batch.items:
                path = item.get("path", item.get("signal_id"))
                if not path:
                    continue
                sources.append(
                    {
                        "id": f"{source_type}:{path}",
                        "source_type": source_type,
                        "source_id": path,
                        "dd_path": path if source_type == "dd" else None,
                        "batch_key": batch.group_key,
                        "status": "extracted",
                        "description": item.get("description")
                        or item.get("documentation")
                        or "",
                    }
                )
        if sources:
            written = await _worker_dependency("asyncio").to_thread(
                merge_standard_name_sources,
                sources,
                force=state.force,
                default_dd_version=_dd_ver,
            )
            wlog.info("Wrote %d StandardNameSource nodes to graph", written)
    total_items = sum(len(b.items) for b in batches)
    state.extracted = batches
    state.extract_stats.total = total_items
    state.extract_stats.processed = total_items
    state.extract_stats.record_batch(total_items)
    if state.loop_extract_stats is not None:
        state.loop_extract_stats.total = total_items
        state.loop_extract_stats.processed = total_items
        state.loop_extract_stats.record_batch(total_items)
    wlog.info("Extraction complete: %d batches, %d items", len(batches), total_items)
    state.stats["extract_batches"] = len(batches)
    state.stats["extract_count"] = total_items
    state.extract_stats.freeze_rate()
    state.extract_stats.status_text = ""
    state.extract_phase.mark_done()
    state.extract_stats.stream_queue.add(
        [
            {
                "primary_text": "extract",
                "description": f"{total_items} paths in {len(batches)} batches",
            }
        ]
    )
    if state.loop_extract_stats is not None:
        state.loop_extract_stats.freeze_rate()
        state.loop_extract_stats.status_text = ""
        state.loop_extract_stats.stream_queue.add(
            [
                {
                    "primary_text": "extract",
                    "description": f"{total_items} paths in {len(batches)} batches",
                }
            ]
        )


_NEARBY_COMPARATOR_HARD_CEILING = 60


def _search_nearby_names(_worker_dependency, /, query: str, k: int = 5) -> list[dict]:
    """Search for existing standard names near *query* for collision avoidance.

    Wraps :func:`imas_codex.standard_names.search.search_standard_names_vector`
    with graceful fallback — never raises, returns ``[]`` if graph or
    embeddings are unavailable.
    """
    try:
        from imas_codex.standard_names.search import search_standard_names_vector

        return search_standard_names_vector(query, k=k)
    except Exception:
        return []


def _collect_nearby_name_comparators(
    _worker_dependency,
    /,
    items: Sequence[dict[str, Any]],
    *,
    per_item_k: int = 5,
    cap: int = 30,
    search: Callable[..., list[dict]] | None = None,
) -> list[dict]:
    """Share a bounded nearby-name budget across the items in a batch.

    The first pass gives each item a share before a second pass spends any
    remaining capacity. Results remain deduplicated by standard-name id. The
    configured cap is a floor while the batch fits the hard ceiling: up to sixty
    items the comparator block grows linearly because composing against no cohort
    comparator is worse than one extra comparator line. Above the ceiling the
    allocation degrades to first-come, exactly the base behaviour, because the
    per-item guarantee cannot hold for an unbounded batch without an unbounded
    prompt and bounding the prompt wins.
    """
    if not items or cap <= 0:
        return []
    effective_cap = min(
        max(cap, len(items)), _worker_dependency("_NEARBY_COMPARATOR_HARD_CEILING")
    )
    search_names = search or _worker_dependency("_search_nearby_names")
    results_by_item = [
        search_names(item.get("description") or item.get("path", ""), k=per_item_k)
        for item in items
    ]
    per_item_allowance = max(1, cap // len(items))
    seen: set[str] = set()
    nearby: list[dict] = []
    next_result_indices: list[int] = []
    for item_results in results_by_item:
        added = 0
        next_index = 0
        for result in item_results:
            if len(nearby) >= effective_cap or added >= per_item_allowance:
                break
            next_index += 1
            result_id = result.get("id", "")
            if result_id and result_id not in seen:
                seen.add(result_id)
                nearby.append(result)
                added += 1
        next_result_indices.append(next_index)
    if len(nearby) < effective_cap:
        for item_results, next_index in zip(
            results_by_item, next_result_indices, strict=True
        ):
            for result in item_results[next_index:]:
                if len(nearby) >= effective_cap:
                    break
                result_id = result.get("id", "")
                if result_id and result_id not in seen:
                    seen.add(result_id)
                    nearby.append(result)
                if len(nearby) >= effective_cap:
                    break
    return nearby


_RERANK_LEASE_FLOOR_USD = 1e-06


async def _rerank_comparators_for_compose(
    _worker_dependency, /, items: Sequence[dict[str, Any]], nearby: list[dict]
) -> tuple[list[dict], float, str]:
    """Rerank the nearby-name comparator block when the switch is on.

    Off by default, so the block reaches the composed prompt exactly as the
    embedding retrieval produced it. When on, each item judges the block and
    it is reordered by judged relevance, keeping the embedding order on any
    failure.

    Ranking is not a gate, matching the sibling rerank callers: a raise out of
    the judgement call — an unresolvable decision model, a state builder that
    refuses — keeps the embedding order and logs why rather than aborting the
    compose batch.

    Returns:
        ``(block, cost, seat)``. ``seat`` is the model the call was billed
        under, empty when no call was made, so the caller charges the spend to
        the seat that raised it rather than to the compose model.
    """
    from imas_codex.standard_names.context import (
        comparator_rerank_enabled,
        rerank_name_comparators,
    )

    if not comparator_rerank_enabled():
        return (nearby, 0.0, "")
    try:
        ordered, cost, seat = await rerank_name_comparators(items, nearby)
    except Exception as exc:
        logger.warning("comparator rerank unavailable: %s", exc)
        return (nearby, 0.0, "")
    return (list(ordered), cost, seat)


def _book_rerank_spend(
    _worker_dependency, /, lease: BudgetLease, cost: float, *, seat: str, phase: str
) -> None:
    """Charge a rerank's spend on the lease opened before it ran.

    The lease is held before the call, so the charge never depends on a
    reservation taken afterwards, and ``charge_event`` records it whether or
    not the lease and the pool can fund it. The event names the seat that
    actually billed the call, which is the judge seat rather than the compose
    model.
    """
    if cost > 0:
        from imas_codex.standard_names.budget import LLMCostEvent

        lease.charge_event(
            cost,
            LLMCostEvent(
                model=seat,
                tokens_in=0,
                tokens_out=0,
                phase=phase,
                service="standard-names",
            ),
        )
    lease.release_unused()


async def _reranked_nearby_comparators(
    _worker_dependency,
    /,
    mgr: BudgetManager,
    items: Sequence[dict[str, Any]],
    nearby: list[dict],
    *,
    phase: str,
) -> tuple[list[dict], float]:
    """Rerank a compose batch's comparator block and book what it spent.

    The rerank is billed by the judge seat, and its price is only known once it
    returns, so its lease is opened before the call and the spend is charged
    through the lease already held. A pool that cannot fund even a floor lease
    runs the batch without a rerank, in the embedding order the block already
    has: nothing is charged for a call that was not made.

    The switch is read before the lease is opened, so the off path books
    nothing at all — no reservation, no charge — and a batch run without the
    rerank touches the budget exactly as it did before the switch existed.

    Returns ``(block, cost)``; ``cost`` is what the lease charged, ``0.0`` when
    the switch is off or no lease could be opened.
    """
    from imas_codex.standard_names.context import comparator_rerank_enabled

    if not comparator_rerank_enabled():
        return (nearby, 0.0)
    lease = mgr.reserve(_worker_dependency("_RERANK_LEASE_FLOOR_USD"), phase=phase)
    if lease is None:
        logger.debug("comparator rerank skipped: no reservable lease")
        return (nearby, 0.0)
    ordered, cost, seat = await _worker_dependency("_rerank_comparators_for_compose")(
        items, nearby
    )
    _worker_dependency("_book_rerank_spend")(lease, cost, seat=seat, phase=phase)
    return (ordered, cost)


def _compose_budget_phase(_worker_dependency, /, state: StandardNameBuildState) -> str:
    """The phase a compose batch's spend belongs to."""
    return getattr(state, "budget_phase_tag", "") or "generate_name"


async def _rerank_comparators_for_worker(
    _worker_dependency,
    /,
    state: StandardNameBuildState,
    items: Sequence[dict[str, Any]],
    nearby: list[dict],
) -> list[dict]:
    """Rerank a compose worker batch's block and book the spend it made.

    The same booking as the batch path, through the same helper, so the two
    cannot drift: a floor lease opened before the call, charged under the seat
    that billed it, recorded even where neither the lease nor the pool can fund
    it. The spend also goes to the compose cost the worker reports, which the
    batch path has no equivalent of.

    A worker running without a budget manager is unleased throughout — the
    compose call it goes on to make is unleased too — so the rerank still runs
    and the reported cost is its only record.
    """
    if state.budget_manager:
        ordered, cost = await _worker_dependency("_reranked_nearby_comparators")(
            state.budget_manager,
            items,
            nearby,
            phase=_worker_dependency("_compose_budget_phase")(state),
        )
    else:
        ordered, cost, _seat = await _worker_dependency(
            "_rerank_comparators_for_compose"
        )(items, nearby)
    if cost:
        state.compose_stats.cost += cost
    return ordered


_DD_CONTEXT_QUERY = "\nMATCH (n:IMASNode {id: $path})\nMATCH (current_dd_version:DDVersion {is_current: true})\nOPTIONAL MATCH (n)-[:HAS_IDENTIFIER_SCHEMA]->(ident:IdentifierSchema)\nOPTIONAL MATCH (n)-[:HAS_PARENT]->(parent:IMASNode)\nOPTIONAL MATCH (parent)-[:HAS_CHILD]->(sibling:IMASNode)\nWHERE sibling.id <> $path\n  AND NOT (sibling.data_type IN ['STRUCTURE', 'STRUCT_ARRAY'])\nWITH n, current_dd_version, ident, parent,\n     [(n)-[:HAS_UNIT]->(unit:Unit) | unit.id] AS unit_relations,\n     collect(DISTINCT {\n         path: sibling.id,\n         description: sibling.description,\n         data_type: sibling.data_type\n     })[0..8] AS sibling_fields\nRETURN n.coordinate1_same_as AS coordinate1,\n       n.coordinate2_same_as AS coordinate2,\n       n.coordinate3_same_as AS coordinate3,\n       n.timebasepath AS timebase,\n       n.cocos_transformation_type AS cocos_label,\n       n.cocos_transformation_expression AS cocos_expression,\n       n.lifecycle_status AS lifecycle_status,\n       n.data_type AS data_type,\n       n.description AS node_description,\n       n.documentation AS node_documentation,\n       ident.name AS identifier_schema_name,\n       ident.documentation AS identifier_schema_doc,\n       ident.options AS identifier_options,\n       current_dd_version.id AS dd_version,\n       n.unit AS raw_unit,\n       unit_relations,\n       CASE WHEN size(unit_relations) = 1\n            THEN head(unit_relations) ELSE null END AS unit_from_rel,\n       parent.id AS parent_path,\n       parent.description AS parent_description,\n       n.enrichment_source AS enrichment_source,\n       sibling_fields\n"
_ANCESTOR_CONTEXT_QUERY = "\nMATCH path = (n:IMASNode {id: $path})-[:HAS_PARENT*1..8]->(a:IMASNode)\nWHERE coalesce(trim(a.description), '') <> ''\n   OR coalesce(trim(a.documentation), '') <> ''\nRETURN length(path) AS depth,\n       a.id AS path,\n       a.description AS description,\n       a.documentation AS documentation,\n       a.enrichment_source AS enrichment_source\nORDER BY depth ASC\n"
_CROSS_IDS_QUERY = "\nMATCH (n:IMASNode {id: $path})-[:IN_CLUSTER]->(c:IMASSemanticCluster)\nWHERE c.scope IN ['global', 'domain']\nWITH c ORDER BY c.scope ASC LIMIT 3\nMATCH (member:IMASNode)-[:IN_CLUSTER]->(c)\nWHERE member.id <> $path\n  AND NOT (member.data_type IN ['STRUCTURE', 'STRUCT_ARRAY'])\nRETURN c.label AS cluster_label,\n       c.description AS cluster_description,\n       c.scope AS cluster_scope,\n       collect(DISTINCT member.id)[0..5] AS member_paths\n"
_VERSION_HISTORY_QUERY = "\nMATCH (vc:IMASNodeChange)-[:FOR_IMAS_PATH]->(n:IMASNode {id: $path})\nWHERE vc.change_type IN [\n    'path_added', 'cocos_transformation_type', 'sign_convention',\n    'units', 'path_renamed', 'definition_clarification'\n]\nRETURN vc.id AS change_id, vc.change_type AS change_type\n"
_ERROR_FIELDS_QUERY = "\nMATCH (d:IMASNode {id: $path})-[:HAS_ERROR]->(e:IMASNode)\nRETURN e.id AS error_path\n"
_MEMBER_DOC_QUERY = "\nUNWIND $paths AS p\nMATCH (n:IMASNode {id: p})\nRETURN n.id AS path, n.documentation AS documentation\n"
_MAX_MEMBER_DOCS = 8
_MEMBER_DOC_MAX_CHARS = 240
_IDS_CONTEXT_QUERY = "\nMATCH (ids:IDS {id: $ids_name})\nOPTIONAL MATCH (child:IMASNode)-[:IN_IDS]->(ids)\nWHERE child.id STARTS WITH $ids_prefix\n  AND child.data_type IN ['STRUCTURE', 'STRUCT_ARRAY', 'FLT_1D']\n  AND size([x IN split(child.id, '/') WHERE true]) = 2\nWITH ids,\n     collect(DISTINCT {\n         name: child.id, description: child.description, data_type: child.data_type\n     })[0..10] AS top_sections\nRETURN ids.description AS ids_description,\n       ids.documentation AS ids_documentation,\n       top_sections\n"


def _enrich_ids_context(_worker_dependency, /, ids_name: str) -> dict | None:
    """Fetch IDS-level context for batch header.

    Returns dict with ids_description, ids_documentation, top_sections,
    or None if IDS not found.
    """
    from imas_codex.graph.client import GraphClient

    with GraphClient() as gc:
        rows = list(
            gc.query(
                _worker_dependency("_IDS_CONTEXT_QUERY"),
                ids_name=ids_name,
                ids_prefix=f"{ids_name}/",
            )
        )
        if not rows:
            return None
        row = rows[0]
        sections = row.get("top_sections") or []
        valid = [s for s in sections if s.get("name")]
        return {
            "ids_description": row.get("ids_description") or "",
            "ids_documentation": row.get("ids_documentation") or "",
            "top_sections": valid,
        }


def _hybrid_search_neighbours_batch(
    _worker_dependency,
    /,
    gc: Any,
    items: list[tuple[str, str | None, str | None]],
    *,
    max_results: int = 15,
    search_k: int = 10,
) -> list[list[dict]]:
    """Batch-embed then search neighbours for multiple (path, desc, domain) items.

    Replaces the per-item N+1 embed pattern: collects all unique
    description-query texts, embeds them in **one** remote round-trip
    via :func:`embed_query_texts`, then fans out ``hybrid_dd_search``
    calls with pre-computed embeddings.

    Path-text queries (``"equilibrium/time_slice/..."``-style) don't
    need embedding — ``hybrid_dd_search`` detects them and uses
    text-only mode automatically.

    Args:
        gc: Active graph client.
        items: List of ``(path, description, physics_domain)`` tuples.
        max_results: Cap per-item neighbour count.
        search_k: ``k`` passed to ``hybrid_dd_search``.

    Returns:
        List of neighbour-lists in the same order as *items*.
    """
    from imas_codex.embeddings.description import embed_query_texts
    from imas_codex.graph.dd_search import hybrid_dd_search

    if not items:
        return []
    text_set: dict[str, None] = {}
    item_desc_queries: list[str] = []
    for _path, description, _domain in items:
        desc_query = (description or "")[:200].strip()
        item_desc_queries.append(desc_query)
        if desc_query:
            text_set.setdefault(desc_query, None)
    unique_texts = list(text_set)
    embed_cache: dict[str, list[float]] = {}
    if unique_texts:
        try:
            embeddings = embed_query_texts(unique_texts)
            for i, emb in enumerate(embeddings):
                if i < len(unique_texts):
                    embed_cache[unique_texts[i]] = emb
        except Exception:
            logger.warning(
                "Batch query embedding failed; falling back to text-only search",
                exc_info=True,
            )
    all_results: list[list[dict]] = []
    per_item_hits: list[dict[str, Any]] = []
    for idx, (path, _desc, physics_domain) in enumerate(items):
        item_hits: dict[str, Any] = {}
        desc_query = item_desc_queries[idx]
        if desc_query:
            pre_emb = embed_cache.get(desc_query)
            try:
                hits = hybrid_dd_search(
                    gc,
                    desc_query,
                    node_category="quantity",
                    physics_domain=physics_domain,
                    k=search_k,
                    embedding=pre_emb,
                )
                for h in hits:
                    if h.path != path:
                        item_hits[h.path] = h
            except Exception:
                logger.debug(
                    "Hybrid search (description) failed for %s", path, exc_info=True
                )
        try:
            hits = hybrid_dd_search(gc, path, node_category="quantity", k=search_k)
            for h in hits:
                if h.path != path and h.path not in item_hits:
                    item_hits[h.path] = h
        except Exception:
            logger.debug("Hybrid search (path) failed for %s", path, exc_info=True)
        per_item_hits.append(item_hits)
    all_hit_paths: set[str] = set()
    for item_hits in per_item_hits:
        all_hit_paths.update(item_hits)
    sn_map: dict[str, str | None] = {}
    if all_hit_paths:
        try:
            rows = gc.query(
                "\n                UNWIND $paths AS pid\n                MATCH (n:IMASNode {id: pid})\n                OPTIONAL MATCH (n)-[:HAS_STANDARD_NAME]->(sn:StandardName)\n                RETURN n.id AS path, sn.id AS sn_id\n                ",
                paths=list(all_hit_paths),
            )
            for r in rows or []:
                sn_map[r["path"]] = r.get("sn_id")
        except Exception:
            logger.debug("HAS_STANDARD_NAME batch pre-resolution failed", exc_info=True)
    for idx, (_path, _desc, _domain) in enumerate(items):
        item_hits = per_item_hits[idx]
        if not item_hits:
            all_results.append([])
            continue
        sorted_hits = sorted(item_hits.values(), key=lambda h: h.score, reverse=True)[
            :max_results
        ]
        neighbours: list[dict] = []
        for h in sorted_hits:
            sn_id = sn_map.get(h.path)
            tag = f"name:{sn_id}" if sn_id else f"dd:{h.path}"
            doc = (h.description or h.documentation or "")[:300]
            neighbours.append(
                {
                    "tag": tag,
                    "path": h.path,
                    "ids": h.ids_name,
                    "unit": h.units or "",
                    "physics_domain": h.physics_domain or "",
                    "doc_short": doc,
                    "cocos_label": h.cocos_transformation_type or "",
                    "lifecycle": h.lifecycle_status or "",
                    "node_type": h.node_type or "",
                    "score": float(h.score) if h.score is not None else None,
                }
            )
        all_results.append(neighbours)
    return all_results


def _hybrid_search_neighbours(
    _worker_dependency,
    /,
    gc: Any,
    path: str,
    description: str | None = None,
    physics_domain: str | None = None,
    max_results: int = 15,
    search_k: int = 10,
) -> list[dict]:
    """Single-item shim around :func:`_hybrid_search_neighbours_batch`.

    .. deprecated::
        Prefer :func:`_hybrid_search_neighbours_batch` for multi-item
        workloads.  This shim exists for any un-migrated callers.
    """
    results = _worker_dependency("_hybrid_search_neighbours_batch")(
        gc,
        [(path, description, physics_domain)],
        max_results=max_results,
        search_k=search_k,
    )
    return results[0] if results else []


_RELATED_MAX_RESULTS = 5


def _retry_attempts(_worker_dependency, /) -> int:
    from imas_codex.settings import get_sn_retry_attempts

    return get_sn_retry_attempts()


def _retry_k_expansion(_worker_dependency, /) -> int:
    from imas_codex.settings import get_sn_retry_k_expansion

    return get_sn_retry_k_expansion()


def _mark_vocab_gap_sources(
    _worker_dependency, /, batch: list[dict], error_detail: str, source_kind: str
) -> int:
    """Mark sources after a vocab-gap validation failure.

    When batch_size == 1 the offending source is known, so it is marked
    ``vocab_gap_compose`` (terminal — won't be reclaimed).  When
    batch_size > 1 we cannot identify which source triggered the failure,
    so all sources are marked ``failed`` with a retry-friendly reason —
    they will be reclaimed on the next ``sn run`` and retried in a
    different (possibly smaller) batch.
    """
    from imas_codex.standard_names.graph_store.graph_source_bindings import (
        persist_claimed_source_outcomes,
    )

    single_source = len(batch) == 1
    reason = "vocab_gap_compose" if single_source else "vocab_gap_batch"
    status = "skipped" if single_source else "extracted"
    outcomes = []
    for item in batch:
        sid = item.get("source_id") or item.get("path")
        if not sid:
            continue
        outcome = _worker_dependency("_claimed_source_outcome")(
            batch,
            sid,
            source_type=source_kind,
            status=status,
            last_error=error_detail[:300],
            skip_reason=reason,
            skip_reason_detail=error_detail[:300],
        )
        if outcome:
            outcomes.append(outcome)
    return len(persist_claimed_source_outcomes(outcomes))


def _related_path_neighbours(
    _worker_dependency,
    /,
    gc: Any,
    path: str,
    *,
    max_results: int = _RELATED_MAX_RESULTS,
) -> list[dict]:
    """Fetch explicit graph-relationship neighbours for a DD path.

    Calls :func:`related_dd_search` to find paths related via
    cluster membership, shared coordinates, matching units, identifier
    schemas, or COCOS transformation type.  Returns a compact list of
    dicts suitable for Jinja template injection.
    """
    from imas_codex.graph.dd_search import related_dd_search

    try:
        result = related_dd_search(
            gc, path, relationship_types="all", max_results=max_results
        )
    except Exception:
        logger.debug("related_dd_search failed for %s", path, exc_info=True)
        return []
    if not result.hits:
        return []
    neighbours: list[dict] = []
    for hit in result.hits:
        entry: dict[str, Any] = {
            "path": hit.path,
            "ids": hit.ids,
            "relationship_type": hit.relationship_type,
            "via": hit.via,
        }
        if hit.doc:
            entry["doc"] = hit.doc
        if hit.physics_domain:
            entry["physics_domain"] = hit.physics_domain
        neighbours.append(entry)
    return neighbours


def _enrich_batch_items(_worker_dependency, /, items: list[dict]) -> None:
    """Enrich batch items with rich DD context from the graph.

    Fetches coordinate specs, COCOS info, identifier schemas, sibling
    fields, cross-IDS cluster siblings, hybrid-search neighbours, and
    version history for each item. Modifies items in-place.
    """
    from imas_codex.graph.client import GraphClient

    with GraphClient() as gc:
        for item in items:
            path = item.get("path")
            if not path:
                continue
            prov_term, base_path = _worker_dependency("detect_value_provenance")(path)
            grounding_path = path
            if prov_term:
                item["value_provenance"] = prov_term
                item["provenance_base_path"] = base_path
                grounding_path = base_path
            rows = list(
                gc.query(_worker_dependency("_DD_CONTEXT_QUERY"), path=grounding_path)
            )
            if not rows:
                continue
            row = rows[0]
            from imas_codex.settings import get_dd_version
            from imas_codex.standard_names.dd_resolutions import resolve_dd_row

            prior_raw_context = item.get("raw_dd_context") or {}
            unit_from_rel = row.get("unit_from_rel")
            if "unit" in prior_raw_context:
                raw_unit = prior_raw_context["unit"]
            else:
                raw_unit = unit_from_rel or row.get("raw_unit") or item.get("unit")
            raw_context = {
                "path": path,
                "unit": raw_unit,
                "documentation": prior_raw_context.get("documentation")
                or row.get("node_documentation"),
                "data_type": row.get("data_type"),
                "physics_domain": row.get("physics_domain"),
                "cocos_transformation_type": row.get("cocos_label"),
                "cocos_transformation_expression": row.get("cocos_expression"),
                "lifecycle_status": row.get("lifecycle_status"),
            }
            resolved_context = resolve_dd_row(
                raw_context,
                dd_version=str(
                    row.get("dd_version") or item.get("dd_version") or get_dd_version()
                ),
            )
            item.update(resolved_context.as_pipeline_item())
            node_desc = (row.get("node_description") or "").strip()
            node_doc = (row.get("node_documentation") or "").strip()
            rich = node_desc or node_doc
            if rich:
                cur = (item.get("description") or "").strip()
                if (
                    not cur
                    or cur
                    == _worker_dependency(
                        "DETERMINISTIC_PARENT_DESCRIPTION_PLACEHOLDER"
                    )
                    or prov_term
                ):
                    item["description"] = rich
                item["documentation"] = node_doc or node_desc
            if not item.get("ancestor_context"):
                anc_rows = list(
                    gc.query(
                        _worker_dependency("_ANCESTOR_CONTEXT_QUERY"),
                        path=grounding_path,
                    )
                )
                lineage: list[dict[str, str]] = []
                seen_text: set[str] = set()
                for a in anc_rows:
                    a_desc = (a.get("description") or "").strip()
                    a_doc = (a.get("documentation") or "").strip()
                    text = a_desc or a_doc
                    if not text or text in seen_text:
                        continue
                    seen_text.add(text)
                    lineage.append({"path": a.get("path") or "", "text": text})
                    if len(lineage) >= 4:
                        break
                if lineage:
                    item["ancestor_context"] = lineage
            if item.get("unit") and path:
                from imas_codex.standard_names.unit_overrides import resolve_unit

                resolved, meta = resolve_unit(path, item["unit"])
                if meta and meta.get("rule") == "override":
                    item["unit"] = resolved
                elif meta and meta.get("rule") == "skip":
                    item["unit"] = None
            coords = []
            for key in ("coordinate1", "coordinate2", "coordinate3"):
                val = row.get(key)
                if val:
                    coords.append(val)
            if coords:
                item["coordinate_paths"] = coords
            timebase = row.get("timebase")
            if timebase:
                item["timebase"] = timebase
            cocos_label = row.get("cocos_label")
            cocos_expr = row.get("cocos_expression")
            if cocos_label:
                item["cocos_label"] = cocos_label
            if cocos_expr:
                item["cocos_expression"] = cocos_expr
            lifecycle = row.get("lifecycle_status")
            if lifecycle and lifecycle != "active":
                item["lifecycle_status"] = lifecycle
            parent_desc = row.get("parent_description")
            if parent_desc and (not item.get("parent_description")):
                item["parent_description"] = parent_desc
            ident_name = row.get("identifier_schema_name")
            if ident_name:
                item["identifier_schema"] = ident_name
                ident_doc = row.get("identifier_schema_doc")
                if ident_doc:
                    item["identifier_schema_doc"] = ident_doc
                raw_options = row.get("identifier_options")
                if raw_options:
                    try:
                        parsed = (
                            _worker_dependency("json").loads(raw_options)
                            if isinstance(raw_options, str)
                            else raw_options
                        )
                        if isinstance(parsed, list):
                            item["identifier_values"] = [
                                {
                                    "name": opt.get("name", ""),
                                    "index": opt.get("index", 0),
                                    "description": opt.get("description", ""),
                                }
                                for opt in parsed
                                if opt.get("name")
                            ]
                    except (_worker_dependency("json").JSONDecodeError, TypeError):
                        pass
            siblings = row.get("sibling_fields") or []
            if siblings and isinstance(siblings, list):
                valid = [s for s in siblings if s.get("path")]
                if valid:
                    item["sibling_fields"] = valid
            cross_rows = list(
                gc.query(_worker_dependency("_CROSS_IDS_QUERY"), path=path)
            )
            if cross_rows:
                clusters = []
                cross_ids_paths = []
                for cr in cross_rows:
                    label = cr.get("cluster_label")
                    members = cr.get("member_paths") or []
                    if label and members:
                        clusters.append(
                            {
                                "label": label,
                                "description": cr.get("cluster_description") or "",
                                "scope": cr.get("cluster_scope") or "",
                                "members": members,
                            }
                        )
                        cross_ids_paths.extend(members)
                if clusters:
                    item["clusters"] = clusters
                if cross_ids_paths:
                    seen = set()
                    unique = []
                    for p in cross_ids_paths:
                        if p not in seen:
                            seen.add(p)
                            unique.append(p)
                    item["cross_ids_paths"] = unique[:8]
            version_rows = list(
                gc.query(_worker_dependency("_VERSION_HISTORY_QUERY"), path=path)
            )
            if version_rows:
                valid_changes = []
                for vr in version_rows:
                    change_id = vr.get("change_id") or ""
                    change_type = vr.get("change_type") or ""
                    parts = change_id.rsplit(":", 1)
                    version = parts[-1] if len(parts) >= 2 else ""
                    if version and change_type:
                        valid_changes.append(
                            {"version": version, "change_type": change_type}
                        )
                if valid_changes:
                    item["version_history"] = valid_changes
            related = _worker_dependency("_related_path_neighbours")(gc, path)
            if related:
                item["related_neighbours"] = related
            error_rows = list(
                gc.query(_worker_dependency("_ERROR_FIELDS_QUERY"), path=path)
            )
            if error_rows:
                error_fields = [
                    ef["error_path"] for ef in error_rows if ef.get("error_path")
                ]
                if error_fields:
                    item["error_fields"] = error_fields
        batch_tuples: list[tuple[str, str | None, str | None]] = []
        batch_indices: list[int] = []
        for idx, item in enumerate(items):
            path = item.get("path")
            if not path:
                continue
            batch_tuples.append(
                (path, item.get("description"), item.get("physics_domain"))
            )
            batch_indices.append(idx)
        if batch_tuples:
            hybrid_results = _worker_dependency("_hybrid_search_neighbours_batch")(
                gc, batch_tuples
            )
            for bi, hr in zip(batch_indices, hybrid_results, strict=True):
                if hr:
                    items[bi]["hybrid_neighbours"] = hr
        member_sets: list[set[str]] = []
        all_members: set[str] = set()
        for item in items:
            primary = item.get("path")
            members: set[str] = set()
            for field in ("family_siblings", "cross_ids_paths"):
                vals = item.get(field) or []
                if isinstance(vals, list):
                    members.update(v for v in vals if v and v != primary)
            bounded = set(sorted(members)[: _worker_dependency("_MAX_MEMBER_DOCS")])
            member_sets.append(bounded)
            all_members.update(bounded)
        if all_members:
            doc_rows = list(
                gc.query(
                    _worker_dependency("_MEMBER_DOC_QUERY"), paths=sorted(all_members)
                )
            )
            member_docs: dict[str, str] = {}
            for r in doc_rows:
                p = r.get("path")
                doc = (r.get("documentation") or "").strip()
                if p and doc:
                    if len(doc) > _worker_dependency("_MEMBER_DOC_MAX_CHARS"):
                        doc = (
                            doc[: _worker_dependency("_MEMBER_DOC_MAX_CHARS")].rstrip()
                            + "…"
                        )
                    member_docs[p] = doc
            for item, members in zip(items, member_sets, strict=True):
                docs = {p: member_docs[p] for p in members if p in member_docs}
                if docs:
                    item["dd_paths_docs"] = docs
        try:
            from imas_codex.standard_names.enrichment import (
                attach_family_accepted_siblings,
            )

            attach_family_accepted_siblings(items, gc=gc)
        except Exception:
            logger.debug("family accepted-sibling enrichment failed", exc_info=True)


_TIME_DERIVATIVE_SEGMENT_RE = _re.compile("d(?:_|.+_)dt(?:_.*)?")
_AXIS_LEAVES = frozenset({"x", "y", "z", "r", "phi"})
_HARDWARE_LOCUS_WORDS = frozenset(
    {
        "coil",
        "probe",
        "sensor",
        "gauge",
        "camera",
        "detector",
        "launcher",
        "mirror",
        "antenna",
        "loop",
        "bolometer",
        "injector",
        "strap",
        "electrode",
        "magnet",
        "valve",
        "thermocouple",
        "interferometer",
        "polarimeter",
        "reflectometer",
        "spectrometer",
        "waveguide",
        "calorimeter",
        "manometer",
        "cryopump",
    }
)
_ORDINAL_POSITION_WORDS = frozenset(
    {
        "first",
        "second",
        "third",
        "fourth",
        "fifth",
        "start",
        "end",
        "intermediate",
        "middle",
        "centre",
        "centres",
        "center",
        "centers",
    }
)
_SOLID_GEOMETRY_PATH_SEGMENTS = frozenset(
    {"thick_line", "outline", "rectangle", "oblique", "arcs_of_circle"}
)
_OPTICAL_GEOMETRY_PATH_WORDS = frozenset({"beam", "sight"})


def _is_ordinal_point_sample(_worker_dependency, /, segment: str) -> bool:
    """True when *segment* names one ordinal sample of a single object.

    The DD samples one geometric object — a line of sight, a conductor path, a
    thick line — at successive positions and gives each position its own
    structure: ``line_of_sight/first_point``, ``line_of_sight/second_point``,
    ``conductor/elements/start_points``, ``…/elements/centres``. Recognised by
    shape: the DD's ``point``/``points`` sampling noun and/or an ordinal
    position word, rather than by an exhaustive path list.

    Where the sample sits along the object is a position on ONE quantity, not a
    second quantity — which is what separates it from a distinct vector FIELD of
    a device (``camera/direction`` vs ``camera/up``, a line of sight and an
    image-up orientation) that genuinely measures something else.
    """
    words = segment.split("_")
    if words[-1] in {"point", "points"}:
        words = words[:-1]
        if not words:
            return True
    return all(w in _worker_dependency("_ORDINAL_POSITION_WORDS") for w in words)


def _vector_fields_conflict(_worker_dependency, /, path_a: str, path_b: str) -> bool:
    """True when two paths are the SAME axis leaf of DIFFERENT vector fields
    of ONE DD device node.

    e.g. ``.../camera/direction/z`` vs ``.../camera/up/z`` — same leaf ``z``,
    differing vector-field parent (``direction`` vs ``up``), common device
    grandparent (``.../camera``). Derived from path structure, not a hardcoded
    device path.

    Ordinal samples of one object (``line_of_sight/first_point/r`` vs
    ``.../second_point/r``) are excluded: they are the same coordinate of the
    same object read at successive positions, so they belong on one name and the
    sample index never enters it.
    """
    sa = path_a.split("/")
    sb = path_b.split("/")
    if len(sa) < 3 or len(sb) < 3:
        return False
    if sa[-1] != sb[-1] or sa[-1] not in _worker_dependency("_AXIS_LEAVES"):
        return False
    if sa[-2] == sb[-2]:
        return False
    if _worker_dependency("_is_ordinal_point_sample")(sa[-2]) and _worker_dependency(
        "_is_ordinal_point_sample"
    )(sb[-2]):
        return False
    return sa[:-2] == sb[:-2]


def _content_tokens(_worker_dependency, /, text: str) -> set[str]:
    """Lower-cased content tokens with a simple plural stem (drop trailing s)."""
    raw = _re.split("[^a-z0-9]+", text.lower())
    return {t[:-1] if len(t) > 3 and t.endswith("s") else t for t in raw if t}


def _expanded_path_tokens(_worker_dependency, /, source_id: str) -> set[str]:
    """Content tokens of a DD path with its abbreviations spelled out.

    A DD path names a device in the abbreviated form the IDS uses (``nbi``,
    ``focs``, ``ec_launchers``, ``pf_active``) while a standard name must spell
    it in full (``neutral_beam_injector``, ``fiber_optic_current_sensor``,
    ``electron_cyclotron_launcher``, ``poloidal_field_coil``). Compared raw, the
    two share no token and a correct attachment reads as a device mismatch, so
    the path's tokens are expanded through the same abbreviation map the naming
    audit uses before any overlap test.
    """
    from imas_codex.standard_names.audits import _FORBIDDEN_ABBREVIATIONS

    tokens = _worker_dependency("_content_tokens")(source_id)
    expanded = set(tokens)
    for abbrev, full in _FORBIDDEN_ABBREVIATIONS:
        if abbrev.strip("_") in tokens:
            expanded |= _worker_dependency("_content_tokens")(full)
    for abbrev, full in _worker_dependency("_DD_DEVICE_EXPANSIONS").items():
        if tokens & _worker_dependency("_content_tokens")(abbrev):
            expanded |= _worker_dependency("_content_tokens")(full)
    return expanded


_DD_DEVICE_EXPANSIONS: dict[str, str] = {
    "focs": "fiber_optic_current_sensor",
    "nbi": "neutral_beam_injector",
    "ece": "electron_cyclotron_emission",
    "pf": "poloidal_field_coil",
    "tf": "toroidal_field_coil",
}


def _name_locus_token(_worker_dependency, /, sn_name: str) -> str | None:
    """Extract the trailing ``_of_<locus>`` / ``_at_<locus>`` locus token, or None."""
    of_i = sn_name.rfind("_of_")
    at_i = sn_name.rfind("_at_")
    i = max(of_i, at_i)
    if i < 0:
        return None
    locus = sn_name[i + 4 :]
    due = locus.find("_due_to_")
    if due >= 0:
        locus = locus[:due]
    return locus or None


def _geometry_representation_conflicts(
    _worker_dependency, /, source_id: str, locus: str | None
) -> bool:
    """Return whether DD geometry shape and name locus assert opposite media.

    A DD ``geometry/<solid primitive>/...`` subtree is a hardware conductor or
    component cross-section.  A line-of-sight/beam subtree is an optical path.
    Their scalar coordinate leaves commonly share unit ``m``, so dimensionality
    cannot distinguish them.  Compare only DD-owned path representation tokens
    with the words of the name's locus; no ISN vocabulary is enumerated here.
    """
    if not locus:
        return False
    segments = tuple(segment for segment in source_id.lower().split("/") if segment)
    solid_path = any(
        (
            segment == "geometry"
            and index + 1 < len(segments)
            and (
                segments[index + 1]
                in _worker_dependency("_SOLID_GEOMETRY_PATH_SEGMENTS")
            )
            for index, segment in enumerate(segments)
        )
    )
    optical_path = any(
        _worker_dependency("_content_tokens")(segment)
        & _worker_dependency("_OPTICAL_GEOMETRY_PATH_WORDS")
        for segment in segments
    )
    locus_words = _worker_dependency("_content_tokens")(locus)
    solid_locus_words = set().union(
        *(
            _worker_dependency("_content_tokens")(segment)
            for segment in _worker_dependency("_SOLID_GEOMETRY_PATH_SEGMENTS")
        )
    )
    optical_locus = bool(
        locus_words & _worker_dependency("_OPTICAL_GEOMETRY_PATH_WORDS")
    )
    solid_locus = bool(locus_words & solid_locus_words)
    return solid_path and optical_locus or (optical_path and solid_locus)


def _dd_standard_name_units_dimensionally_agree(
    _worker_dependency, /, sn_unit: str, dd_unit: str, source_id: str
) -> bool:
    """True when *sn_unit* and *dd_unit* canonicalise to the same dimension.

    Thin wrapper over the same registry the dimensionality rule below
    consults, for callers that need a unit-agreement verdict before that
    rule's own placement in the function.
    """
    from imas_codex.units.dd_unit_exceptions import canonical_or_none, units_agree

    return (
        canonical_or_none(dd_unit) is not None
        and canonical_or_none(sn_unit) is not None
        and units_agree(sn_unit, dd_unit, source_id)
    )


def _is_attachment_consistent(
    _worker_dependency,
    /,
    source_id: str,
    sn_name: str,
    existing_sources: Sequence[str] = (),
    *,
    dd_unit: str | None = None,
    sn_unit: str | None = None,
) -> tuple[bool, str]:
    """Reject attachments where the DD path tense disagrees with the SN tense.

    E.g. ``change_in_electron_density`` may not be attached to
    ``core_profiles/.../density`` (a base quantity, not a change). Symmetric:
    a base-quantity SN may not absorb an ``instant_changes`` path.

    ``existing_sources`` are the DD paths already carried by ``sn_name`` (bare,
    ``dd:``-prefix stripped) — used by the distinct-vector guard to reject a
    second vector field of one device node collapsing onto one scalar name.

    ``dd_unit`` / ``sn_unit`` enable the dimensionality rule. Both are optional
    because the lexical rules are unit-independent; a caller with no unit
    context gets exactly the pre-unit behaviour.
    """
    change_path_tokens = ("instant_changes", "/change/", "_delta", "tendency_")
    sn_claims_derivative = _worker_dependency("_name_claims_time_change")(sn_name)
    sn_is_rate = sn_claims_derivative or _worker_dependency("_name_denotes_rate")(
        sn_name
    )
    path_is_change = any(t in source_id for t in change_path_tokens) or any(
        _worker_dependency("_TIME_DERIVATIVE_SEGMENT_RE").fullmatch(seg)
        for seg in source_id.split("/")
    )
    if sn_claims_derivative and (not path_is_change):
        excused = (
            dd_unit
            and sn_unit
            and _worker_dependency("_time_change_claim_is_binary_scoped")(sn_name)
            and _worker_dependency("_dd_standard_name_units_dimensionally_agree")(
                sn_unit, dd_unit, source_id
            )
        )
        if not excused:
            return (
                False,
                f"tense mismatch: SN '{sn_name}' is a change/rate but path '{source_id}' is a base quantity",
            )
    if path_is_change and (not sn_is_rate):
        return (
            False,
            f"tense mismatch: path '{source_id}' is a change/rate but SN '{sn_name}' is a base quantity",
        )
    if _worker_dependency("_state_resolution_tokens")():
        sn_is_state = _worker_dependency("_name_denotes_state")(sn_name)
        path_is_state = "/state/" in source_id or source_id.endswith("/state")
        if path_is_state and (not sn_is_state):
            return (
                False,
                f"state-resolution mismatch: path '{source_id}' is state-resolved but SN '{sn_name}' is species-level",
            )
        if sn_is_state and (not path_is_state):
            return (
                False,
                f"state-resolution mismatch: SN '{sn_name}' is state-resolved but path '{source_id}' is species-level",
            )
    if any(
        b in sn_name for b in _worker_dependency("_shape_parameter_bases")()
    ) and any(f"_of_{s}" in sn_name for s in ("plasma_boundary", "flux_surface")):
        expected_surface = _worker_dependency("_shape_parameter_surface")(source_id)
        if f"_of_{expected_surface}" not in sn_name:
            return (
                False,
                f"shape-parameter surface mismatch: path '{source_id}' is of_{expected_surface} but SN '{sn_name}' carries a different surface",
            )
    for other in existing_sources:
        if (
            other
            and other != source_id
            and _worker_dependency("_vector_fields_conflict")(source_id, other)
        ):
            return (
                False,
                f"distinct-vector conflict: path '{source_id}' and already-sourced '{other}' are the same axis leaf of DIFFERENT vector fields of one device node — SN '{sn_name}' must not carry both",
            )
    from imas_codex.standard_names.segments import is_place_locus

    locus = _worker_dependency("_name_locus_token")(sn_name)
    if locus:
        if _worker_dependency("_geometry_representation_conflicts")(source_id, locus):
            return (
                False,
                f"geometry representation mismatch: path '{source_id}' and SN '{sn_name}' describe incompatible solid/optical geometry",
            )
        locus_tokens = _worker_dependency("_content_tokens")(locus)
        if locus_tokens & _worker_dependency("_HARDWARE_LOCUS_WORDS") and (
            not is_place_locus(locus)
        ):
            path_tokens = _worker_dependency("_expanded_path_tokens")(source_id)
            if not locus_tokens & path_tokens:
                return (
                    False,
                    f"locus/source device mismatch: SN '{sn_name}' has hardware locus '{locus}' but path '{source_id}' shares no device token with it",
                )
    if dd_unit:
        from imas_codex.graph.models import DDResolutionField, DDResolutionValueKind
        from imas_codex.settings import get_dd_version
        from imas_codex.standard_names.dd_resolutions import (
            DDResolutionValue,
            read_graph_dd_field,
        )

        graph_unit = read_graph_dd_field(
            path=source_id,
            dd_version=get_dd_version(),
            field=DDResolutionField.unit,
            graph_value=DDResolutionValue(
                kind=DDResolutionValueKind.string, value=dd_unit
            ),
        )
        dd_unit = graph_unit.effective.value
    if dd_unit and sn_unit:
        from imas_codex.units.dd_unit_exceptions import canonical_or_none, units_agree

        if (
            canonical_or_none(dd_unit) is not None
            and canonical_or_none(sn_unit) is not None
            and (not units_agree(sn_unit, dd_unit, source_id))
        ):
            return (
                False,
                f"unit dimensionality mismatch: path '{source_id}' declares '{dd_unit}' but SN '{sn_name}' declares '{sn_unit}' — physically distinct quantities",
            )
    return (True, "")


def _process_attachments_core(
    _worker_dependency,
    /,
    attachments: list,
    wlog: logging.LoggerAdapter | logging.Logger,
    *,
    source_claims: list[dict[str, Any]] | None = None,
    return_winner_ids: bool = False,
) -> dict[str, Any]:
    """Stateless attachment-processing helper used by both compose paths.

    Filters by tense consistency, writes ``HAS_STANDARD_NAME`` edges and
    appends ``source_paths``, and returns ``{"accepted", "rejected"}``
    counts.  Caller is responsible for any state-side stats updates.
    """
    existing_by_sn: dict[str, list[str]] = {}
    rejected: list[tuple[str, str, str]] = []
    accepted: list = []
    for a in attachments:
        existing = existing_by_sn.setdefault(a.standard_name, [])
        ok, reason = _worker_dependency("_is_attachment_consistent")(
            a.source_id, a.standard_name, existing_sources=existing
        )
        if ok:
            accepted.append(a)
            existing.append(a.source_id)
        else:
            rejected.append((a.source_id, a.standard_name, reason))
    for src, sn, why in rejected:
        wlog.warning("Rejected attachment %s → %s: %s", src, sn, why)
    counts: dict[str, Any] = {"accepted": 0, "rejected": len(rejected)}
    if return_winner_ids:
        counts["winner_ids"] = []
    if not accepted:
        return counts
    if source_claims is None:
        wlog.warning(
            "Rejected %d attachments without source ownership fences", len(accepted)
        )
        counts["rejected"] += len(accepted)
        return counts
    from imas_codex.standard_names.graph_store.graph_source_bindings import (
        persist_claimed_attachments,
    )

    fenced_batch = []
    for attachment in accepted:
        outcome = _worker_dependency("_claimed_source_outcome")(
            source_claims, attachment.source_id, source_type="dd", status="attached"
        )
        if outcome:
            fenced_batch.append(
                {
                    **outcome,
                    "source_id": attachment.source_id,
                    "standard_name": attachment.standard_name,
                }
            )
    winner_ids = set(persist_claimed_attachments(fenced_batch))
    if return_winner_ids:
        counts["winner_ids"] = sorted(winner_ids)
    accepted = [
        attachment
        for attachment in accepted
        if f"dd:{attachment.source_id}" in winner_ids
    ]
    counts["rejected"] += len(attachments) - len(rejected) - len(accepted)
    counts["accepted"] = len(accepted)
    for attachment in accepted:
        wlog.info(
            "Attached %s → %s (%s)",
            attachment.source_id,
            attachment.standard_name,
            attachment.reason,
        )
    return counts


def _process_attachments(
    _worker_dependency,
    /,
    attachments: list,
    state: StandardNameBuildState,
    wlog: logging.LoggerAdapter,
) -> None:
    """Attach DD paths to existing standard names without regeneration.

    Linear-path wrapper around :func:`_process_attachments_core` that
    folds counts into the build state's stats dictionary.
    """
    counts = _worker_dependency("_process_attachments_core")(attachments, wlog)
    if counts["accepted"]:
        prev = state.stats.get("attachments", 0)
        state.stats["attachments"] = prev + counts["accepted"]
    if counts["rejected"]:
        state.stats["attachments_rejected"] = (
            state.stats.get("attachments_rejected", 0) + counts["rejected"]
        )


def _update_sources_after_compose(
    _worker_dependency,
    /,
    candidates: list[dict],
    source: str,
    wlog: logging.LoggerAdapter,
) -> None:
    """Reject the unfenced legacy compose transition.

    Claimed composition is finalized inside
    :func:`graph_ops.persist_generated_name_batch`; source IDs alone are never
    sufficient authority for a graph mutation.
    """
    if candidates:
        wlog.warning(
            "Ignored %d unfenced %s compose source updates", len(candidates), source
        )


def _update_sources_after_attach(
    _worker_dependency, /, attachments: list, source: str, wlog: logging.LoggerAdapter
) -> None:
    """Reject the unfenced legacy attachment transition."""
    if attachments:
        wlog.warning(
            "Ignored %d unfenced %s attachment source updates", len(attachments), source
        )


def _update_sources_after_vocab_gap(
    _worker_dependency,
    /,
    vocab_gaps: list[dict],
    source: str,
    wlog: logging.LoggerAdapter,
    *,
    max_attempts: int = 3,
) -> None:
    """Reject the unfenced legacy vocabulary-gap transition."""
    del max_attempts
    if vocab_gaps:
        wlog.warning(
            "Ignored %d unfenced %s vocabulary-gap source updates",
            len(vocab_gaps),
            source,
        )


def _update_sources_after_skip(
    _worker_dependency,
    /,
    skipped_ids: list[str],
    source: str,
    wlog: logging.LoggerAdapter | logging.Logger,
) -> None:
    """Reject the unfenced legacy skip transition."""
    if skipped_ids:
        wlog.warning(
            "Ignored %d unfenced %s skip source updates", len(skipped_ids), source
        )


def _persist_claimed_skips(
    _worker_dependency,
    /,
    skipped_ids: list[str],
    batch: list[dict[str, Any]],
    *,
    source_type: str,
    reason: str,
    detail: str | None = None,
) -> list[str]:
    """Persist skips only for exact source-claim winners."""
    from imas_codex.standard_names.graph_store.graph_source_bindings import (
        persist_claimed_source_outcomes,
    )

    outcomes = []
    for source_id in skipped_ids:
        outcome = _worker_dependency("_claimed_source_outcome")(
            batch,
            source_id,
            source_type=source_type,
            status="skipped",
            skip_reason=reason,
            skip_reason_detail=detail,
        )
        if outcome:
            outcomes.append(outcome)
    return persist_claimed_source_outcomes(outcomes)


def _consume_claimed_vocab_gaps(
    _worker_dependency,
    /,
    gaps: list[dict[str, Any]],
    batch: list[dict[str, Any]],
    *,
    source_type: str,
    max_attempts: int = 3,
    return_winner_ids: bool = False,
) -> list[dict[str, Any]] | tuple[list[dict[str, Any]], list[str]]:
    """Persist vocabulary gaps while consuming their exact source claims."""
    from imas_codex.standard_names.graph_store.graph_source_bindings import (
        persist_claimed_vocab_gaps,
    )
    from imas_codex.standard_names.segments import is_actionable_gap, is_open_segment

    outcomes = []
    gap_by_sns: dict[str, list[dict[str, Any]]] = {}
    for gap in gaps:
        if is_open_segment(gap.get("segment")):
            continue
        source_id = gap.get("source_id")
        if not source_id:
            continue
        item = next(
            (
                claimed
                for claimed in batch
                if (claimed.get("path") or claimed.get("source_id")) == source_id
            ),
            None,
        )
        if not item:
            continue
        actionable = is_actionable_gap(gap.get("segment"), gap.get("token"))
        status = (
            "vocab_gap"
            if actionable
            else "failed"
            if int(item.get("attempt_count") or 0) >= max_attempts
            else "extracted"
        )
        outcome = _worker_dependency("_claimed_source_outcome")(
            batch,
            source_id,
            source_type=source_type,
            status=status,
            last_error=None
            if actionable
            else f"non-actionable vocabulary gap: {gap.get('reason', '')}"[:300],
        )
        if outcome:
            outcomes.append(outcome)
            gap_by_sns.setdefault(outcome["sns_id"], []).append(gap)
    winners = set(persist_claimed_vocab_gaps(gaps, outcomes, source_type=source_type))
    persisted_gaps = [gap for sns_id in winners for gap in gap_by_sns.get(sns_id, [])]
    if return_winner_ids:
        return (persisted_gaps, sorted(winners))
    return persisted_gaps


def _search_reference_exemplars(
    _worker_dependency, /, items: list[dict], domains: list[str], *, k: int = 5
) -> list[dict]:
    """Synthesise a query from batch items and return reference SN exemplars.

    Builds a query string from up to three item descriptions, calls
    :func:`search_standard_names_with_documentation`, and excludes any SN whose
    ``id`` matches an item already present in the batch (when items carry an
    ``existing_name``).  *domains* filters via embedding-quality semantics
    only (the search itself filters by ``validation_status='valid'``).

    Returns an empty list if no descriptions are available or the search
    backend is unavailable.
    """
    from imas_codex.standard_names.search import (
        search_standard_names_with_documentation,
    )

    desc_snippets = [
        item.get("description", "") for item in items[:3] if item.get("description")
    ]
    if not desc_snippets:
        return []
    synth_query = "; ".join(desc_snippets)
    exclude_ids = sorted(
        {item.get("existing_name") for item in items if item.get("existing_name")}
    )
    try:
        return search_standard_names_with_documentation(
            synth_query, k=k, exclude_ids=exclude_ids or None
        )
    except Exception:
        logger.debug("Reference exemplar search failed", exc_info=True)
        return []


async def _grammar_retry(
    _worker_dependency,
    /,
    original_name: str,
    parse_error: str,
    model: str,
    acall_fn,
    *,
    reasoning_effort: str | None = None,
) -> tuple[str | None, float, int, int]:
    """Single grammar-failure re-prompt.

    Asks the LLM to revise a name that failed grammar round-trip,
    providing the parse error and a grammar cheat-sheet fragment.

    Returns ``(revised_name | None, cost_usd, tokens_in, tokens_out)``.
    """
    from pydantic import BaseModel, Field

    class GrammarRetryResponse(BaseModel):
        revised_name: str = Field(
            description="The revised standard name that passes grammar parsing"
        )
        explanation: str = Field(description="Brief explanation of the fix")

    retry_prompt = f"The standard name `{original_name}` failed grammar parsing with error:\n  {parse_error}\n\nRevise ONLY the name to pass the grammar round-trip. Rules:\n- Pattern: [subject_][physical_base|geometric_base][_component][_position][_process][_object]\n- ALL segments are CLOSED including physical_base (80 registered tokens).\n  Only use tokens from the registries. If no token fits, emit a vocab_gap.\n- Closed-vocabulary tokens (e.g. toroidal, parallel, thermal,\n  e_cross_b_drift, normalized, fast_ion, volume_averaged) MUST be\n  placed in their closed segment slot. Never absorb them into\n  physical_base.\n- No abbreviations, no provenance verbs, no unit suffixes\n- Return the MINIMAL fix — keep the name as close to the original as possible.\n"
    try:
        llm_out = await acall_fn(
            model=model,
            messages=[{"role": "user", "content": retry_prompt}],
            response_model=GrammarRetryResponse,
            service="standard-names",
            reasoning_effort=reasoning_effort,
        )
        result, _cost, _tokens = llm_out
        return (
            result.revised_name if result else None,
            float(_cost or 0.0),
            getattr(llm_out, "input_tokens", 0) or 0,
            getattr(llm_out, "output_tokens", 0) or 0,
        )
    except Exception:
        return (None, 0.0, 0, 0)


async def _self_refine_candidate(
    _worker_dependency,
    /,
    name: str,
    description: str,
    segments: dict[str, Any] | None,
    dd_context: dict[str, Any] | None,
    model: str,
    acall_fn,
    *,
    reasoning_effort: str | None = None,
) -> tuple[str, str]:
    """Free local self-improvement pass over a freshly-composed candidate.

    Runs AFTER compose + grammar normalization and BEFORE persist (and so
    before the paid review quorum). The *local* compose model critiques its
    own ``name`` + ``description`` against deterministic grammar diagnostics
    and the compose rubric and may emit an improved candidate. The contract
    is **improve-or-no-op**: it never rejects.

    Safety: the suggested name is re-validated via the grammar round-trip
    (strict lossless parse → compose). If the rewrite
    fails grammar parsing, or loses canonical order (round-trips to a
    different string), or fails the well-formedness gate, the **original**
    ``(name, description)`` is returned unchanged. The local model is free
    (``get_model("sn-compose")`` served on the local GPU), so this charges
    ``$0``; callers do not account any cost.

    Returns ``(name, description)`` — improved when the rewrite is safe and
    the model elected to change it, otherwise the originals unchanged.
    """
    from pydantic import BaseModel, Field

    from imas_codex.llm.prompt_loader import render_prompt
    from imas_codex.standard_names.context import build_compose_context
    from imas_codex.standard_names.models import GrammarOperator

    class SelfRefineSegments(BaseModel):
        base_token: str = Field(description="Registered physical_base/geometry token")
        base_kind: str = Field(default="quantity", description="quantity | geometry")
        projection_axis: str | None = None
        qualifiers: list[str] = Field(default_factory=list)
        locus_token: str | None = None
        locus_relation: str | None = None
        locus_type: str | None = None
        process_token: str | None = None
        operators: list[GrammarOperator] = Field(default_factory=list)

    class SelfRefineResponse(BaseModel):
        changed: bool = Field(
            description="True only if name or description was improved"
        )
        name: str = Field(description="Improved (or unchanged) standard name")
        description: str = Field(
            default="", description="Improved (or unchanged) ≤120 char description"
        )
        segments: SelfRefineSegments | None = Field(
            default=None, description="IR grammar segments of the improved name"
        )

    diagnostics: list[str] = []
    try:
        from imas_codex.standard_names.grammar_adapter import parse_canonical_name

        parse_canonical_name(name)
        diagnostics.append(
            "Round-trip: OK — name parses and re-composes to canonical form."
        )
    except Exception:
        diagnostics.append(
            "Round-trip: WARNING — name did not parse cleanly; prefer the closest grammatical alternative."
        )
    try:
        context = await _worker_dependency("asyncio").to_thread(build_compose_context)
    except Exception:
        context = {}
    system_prompt = render_prompt("sn/self_refine_system", context)
    user_prompt = render_prompt(
        "sn/self_refine_user",
        {
            **context,
            "name": name,
            "description": description or "",
            "segments": segments or {},
            "dd_context": dd_context or {},
            "diagnostics": diagnostics,
        },
    )
    try:
        llm_out = await acall_fn(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_model=SelfRefineResponse,
            service="standard-names",
            reasoning_effort=reasoning_effort,
        )
        result, _cost, _tokens = llm_out
    except Exception:
        logger.debug("Self-refine LLM call failed for %r — keeping original", name)
        return (name, description)
    if result is None or not result.changed:
        return (name, description)
    new_name = _worker_dependency("normalize_spelling")((result.name or "").strip())
    if not new_name or new_name == name:
        new_desc = _worker_dependency("normalize_description_text")(
            (result.description or "").strip()
        )
        return (name, new_desc or description)
    _well_formed, _ = _worker_dependency("is_well_formed_candidate")(new_name)
    if not _well_formed:
        logger.debug(
            "Self-refine suggestion %r not well-formed — keeping original %r",
            new_name,
            name,
        )
        return (name, description)
    try:
        from imas_codex.standard_names.grammar_adapter import parse_canonical_name

        parse_canonical_name(new_name)
    except Exception:
        logger.debug(
            "Self-refine suggestion %r failed grammar round-trip — keeping %r",
            new_name,
            name,
        )
        return (name, description)
    new_desc = _worker_dependency("normalize_description_text")(
        (result.description or "").strip()
    )
    logger.info("Self-refine improved %r → %r", name, new_name)
    return (new_name, new_desc or description)


_COMPOSE_DOMAIN_STATE_QUERY = "\nUNWIND $ids AS name_id\nMATCH (sn:StandardName {id: name_id})\nOPTIONAL MATCH (src:StandardNameSource)-[:PRODUCED_NAME]->(sn)\nOPTIONAL MATCH (src)-[:FROM_DD_PATH]->(dd:IMASNode)\nRETURN sn.id AS id,\n       sn.description AS description,\n       [p IN collect(DISTINCT {path: dd.id, domain: dd.physics_domain})\n        WHERE p.path IS NOT NULL] AS sources\n"


async def _write_composed_name_domains(
    _worker_dependency, /, candidates: list[dict[str, Any]], winner_ids: set[str]
) -> int:
    """Give each newly persisted name its domain through the one writer.

    A composed name carries no domain of its own: its ``source_domains`` is the
    one writer's, and its ``physics_domain`` is the Jev decision. Each name the
    batch persisted is asked through ``domain_decision``'s Choice and then set
    by :func:`imas_codex.standard_names.domain_decision.write_name_domain`; a
    call that fails or a choice below the floor leaves the name undecided, per
    the writer's rule. ``winner_ids`` are the persisted sources' ids, so a
    composed name is one of *candidates* whose ``source_id`` is among them.

    Returns the number of names written.
    """
    from imas_codex.graph.client import GraphClient
    from imas_codex.settings import get_model
    from imas_codex.standard_names import domain_decision as _dd

    names = sorted(
        {
            str(c["id"])
            for c in candidates
            if c.get("id") and c.get("source_id") in winner_ids
        }
    )
    if not names:
        return 0
    ids_map = _dd.ids_domain_map()
    model = get_model(_dd.SEAT)
    with GraphClient() as gc:
        rows = [
            dict(row)
            for row in gc.query(
                _worker_dependency("_COMPOSE_DOMAIN_STATE_QUERY"), ids=names
            )
        ]
        semaphore = _worker_dependency("asyncio").Semaphore(_dd.CONCURRENCY)

        async def _decide(row: dict[str, Any]) -> Any:
            sources = {
                str(p["path"]): p.get("domain") or None
                for p in row.get("sources") or []
                if p and p.get("path")
            }
            async with semaphore:
                return await _dd.decide_domain(
                    row["id"],
                    row.get("description") or "",
                    sources,
                    ids_map,
                    model=model,
                )

        decisions = await _worker_dependency("asyncio").gather(
            *(_decide(row) for row in rows)
        )
        written = 0
        for row, decision in zip(rows, decisions, strict=True):
            await _worker_dependency("asyncio").to_thread(
                _dd.write_name_domain, gc, row["id"], decision
            )
            written += 1
    return written
