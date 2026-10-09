"""Graph state machine workers for the IMAS mapping pipeline.

Source-centric pipeline: each SignalSource group is the work item,
processed through four phases tracked via ``mapping_status`` on the
graph node:

  (enriched, no mapping_status) → assigned → mapped → validated

Workers claim batches from the graph using ``mapping_claimed_at`` +
``mapping_claim_token`` for coordination with orphan recovery via timeout.

Architecture follows ``discovery/base/engine.py``:
- Independent async workers claim batches from the graph
- ``@retry_on_deadlock()`` + ``ORDER BY rand()`` + claim_token
- ``PipelinePhase.set_has_work_fn`` for phase completion detection
- ``OrphanRecoverySpec`` for automatic stale claim release
- ``run_discovery_engine`` with supervised worker group

Workers:
- context_worker: Gathers IDS structure + semantic context (one-shot per IDS)
- assign_worker: LLM assigns sources to IMAS sections (per-IDS batch)
- map_worker: LLM maps per-source fields (claim loop)
- validate_worker: Validates + persists per-IDS (once all mapped)
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from imas_codex.cli.logging import WorkerLogAdapter
from imas_codex.discovery.base.claims import (
    claim_batch,
    has_pending,
    release_claim,
    release_claims_batch,
)
from imas_codex.discovery.base.engine import WorkerSpec, run_discovery_engine
from imas_codex.discovery.base.progress import WorkerStats
from imas_codex.discovery.base.state import DiscoveryStateBase
from imas_codex.discovery.base.supervision import (
    OrphanRecoverySpec,
    PipelinePhase,
)
from imas_codex.graph.client import GraphClient
from imas_codex.ids.candidates import (
    expand_cluster_siblings,
    judge_candidates,
    judgments_available,
    retrieve_candidates,
    route,
    route_ids,
)
from imas_codex.ids.graph_ops import (
    CandidateWriteError,
    delete_mapping,
    write_candidates,
)
from imas_codex.ids.mapping import PipelineCost
from imas_codex.settings import get_mapping_route_thresholds

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)

CLAIM_TIMEOUT_SECONDS = 300  # 5 minutes


# =============================================================================
# Discovery State — holds ALL IDS targets, not just one
# =============================================================================


@dataclass
class MappingDiscoveryState(DiscoveryStateBase):
    """Shared state for the mapping pipeline across all IDS targets."""

    # All IDS targets to map (resolved in pre-flight)
    target_ids_list: list[str] = field(default_factory=list)
    # Target info from discover_mappable_ids: [{ids_name, domains, source_count}]
    target_info: list[dict] = field(default_factory=list)

    dd_version: str | None = None
    dd_major: int | None = None
    model: str | None = None

    # Context cache: ids_name -> context dict (gathered by context_worker)
    contexts: dict[str, dict] = field(default_factory=dict)
    # Assignment results cache: ids_name -> TargetAssignmentBatch
    assignments: dict[str, Any] = field(default_factory=dict)
    # Mapping batches cache: ids_name -> list[(assignment, batch)]
    mapping_batches: dict[str, list] = field(default_factory=dict)
    # Claims still held by map_worker; release them after bounded shutdown.
    mapping_claims: set[str] = field(default_factory=set)

    # Pipeline cost tracking (cumulative across all IDS)
    cost: PipelineCost = field(default_factory=PipelineCost)

    # Aggregate counters (across all IDS)
    sources_total: int = 0
    sources_assigned: int = 0
    sources_mapped: int = 0
    sources_validated: int = 0
    bindings_total: int = 0
    bindings_passed: int = 0
    escalations: int = 0

    # Per-IDS results: ids_name -> {bindings, escalations}
    ids_results: dict[str, dict] = field(default_factory=dict)

    # Worker stats
    context_stats: WorkerStats = field(default_factory=WorkerStats)
    assign_stats: WorkerStats = field(default_factory=WorkerStats)
    map_stats: WorkerStats = field(default_factory=WorkerStats)
    validate_stats: WorkerStats = field(default_factory=WorkerStats)

    # Pipeline phases
    context_phase: PipelinePhase = field(init=False)
    assign_phase: PipelinePhase = field(init=False)
    map_phase: PipelinePhase = field(init=False)
    validate_phase: PipelinePhase = field(init=False)

    # Control
    persist: bool = True
    activate: bool = True
    clear: bool = False
    skip_errors: bool = False

    def __post_init__(self) -> None:
        self.context_phase = PipelinePhase("context")
        self.assign_phase = PipelinePhase("assign")
        self.map_phase = PipelinePhase("map")
        self.validate_phase = PipelinePhase("validate")

    @property
    def total_cost(self) -> float:
        return self.cost.total_usd

    def should_stop(self) -> bool:
        if super().should_stop():
            return True
        if self.budget_exhausted:
            return True
        return False


# =============================================================================
# Candidate Discovery State
# =============================================================================


@dataclass
class CandidateDiscoveryState(DiscoveryStateBase):
    """Shared state for the candidate judgment pipeline.

    One candidate worker claims enriched sources whose ``candidate_route`` is
    unset, routes each to its most probable IDSs, retrieves DD candidates
    within those IDSs, judges them, and writes the route and the candidate
    edges. The run ends when no unjudged source remains, the deadline passes,
    or the spend reaches the cost limit.
    """

    # Restrict to these physics domains (None = all).
    domains: list[str] = field(default_factory=list)
    # Restrict the routed IDSs to this set (None = any).
    ids_filter: list[str] = field(default_factory=list)
    # Restrict claims to these SignalSource identities (empty = all).
    focus_ids: list[str] = field(default_factory=list)
    # Maximum sources to judge this run (None = unbounded).
    source_limit: int | None = None
    # Sources claimed and judged per batch.
    batch_size: int = 10

    dd_version: int | None = None
    model: str | None = None

    # Cumulative candidate-stage cost; every decisions call's reported spend is
    # added to it as the call is made, so ``--cost-limit`` halts the loop.
    cost: PipelineCost = field(default_factory=PipelineCost)

    sources_judged: int = 0
    candidates_written: int = 0

    candidate_stats: WorkerStats = field(default_factory=WorkerStats)
    candidate_phase: PipelinePhase = field(init=False)

    def __post_init__(self) -> None:
        self.candidate_phase = PipelinePhase("candidate")

    @property
    def total_cost(self) -> float:
        return self.cost.total_usd

    def should_stop(self) -> bool:
        if super().should_stop():
            return True
        if self.budget_exhausted:
            return True
        if self.source_limit is not None and self.sources_judged >= self.source_limit:
            return True
        return False


# =============================================================================
# Graph Claim Operations
# =============================================================================


def _domain_filter(domains: list[str] | None) -> tuple[str, dict[str, Any]]:
    """Return the domain predicate fragment and its bound parameters."""
    if not domains:
        return "", {}
    return "AND n.physics_domain IN $domains", {"domains": domains}


_ASSIGNMENT_RETURN = """
OPTIONAL MATCH (m:FacilitySignal)-[:MEMBER_OF]->(n)
WITH n, count(m) AS member_count,
     collect(DISTINCT m.accessor)[..10] AS sample_accessors
OPTIONAL MATCH (rep:FacilitySignal {id: n.representative_id})
"""

_ASSIGNMENT_FIELDS = """
n.id AS id, n.group_key AS group_key,
n.description AS description,
n.keywords AS keywords,
n.physics_domain AS physics_domain,
member_count,
sample_accessors,
rep.description AS rep_description,
rep.unit AS rep_unit,
rep.sign_convention AS rep_sign_convention,
rep.cocos AS rep_cocos
"""

_REPRESENTATIVE = "OPTIONAL MATCH (rep:FacilitySignal {id: n.representative_id})"

_MAPPING_FIELDS = """
n.id AS id, n.group_key AS group_key,
n.description AS description,
n.keywords AS keywords,
n.physics_domain AS physics_domain,
rep.description AS rep_description,
rep.unit AS rep_unit,
rep.sign_convention AS rep_sign_convention,
rep.cocos AS rep_cocos
"""

_CANDIDATE_FIELDS = """
n.id AS id, n.group_key AS group_key,
n.description AS description,
n.keywords AS keywords,
n.physics_domain AS physics_domain,
rep.description AS rep_description,
rep.unit AS rep_unit,
rep.sign_convention AS rep_sign_convention,
rep.cocos AS rep_cocos
"""

_MAPPING_CLAIM_FIELDS = {
    "claimed_field": "mapping_claimed_at",
    "token_field": "mapping_claim_token",
}


def claim_sources_for_mapping(
    facility: str,
    ids_name: str,
    batch_size: int = 3,
) -> list[dict[str, Any]]:
    """Claim sources that selected a home in ``ids_name`` and are not yet bound there.

    A source is claimable by this IDS's pass while it carries a selected
    ``MAPPING_CANDIDATE`` edge into the IDS and no ``MAPS_TO_IMAS`` binding into
    it. A source with homes in two IDSs is therefore claimed once by each
    IDS's pass, and a pass skips a source it has already bound.
    """
    return claim_batch(
        "SignalSource",
        facility=facility,
        status_predicate=(
            "EXISTS { (n)-[c:MAPPING_CANDIDATE]->(:IMASNode) "
            "WHERE c.route = true AND c.ids = $ids_name } "
            "AND NOT EXISTS { (n)-[:MAPS_TO_IMAS]->(:IMASNode {ids: $ids_name}) }"
        ),
        status_params={"ids_name": ids_name},
        batch_size=batch_size,
        return_fields=_MAPPING_FIELDS,
        return_clause=_REPRESENTATIVE,
        timeout_seconds=CLAIM_TIMEOUT_SECONDS,
        **_MAPPING_CLAIM_FIELDS,
    )


def claim_sources_for_candidates(
    facility: str,
    domains: list[str] | None = None,
    batch_size: int = 10,
    focus_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Claim enriched sources whose candidates have not been judged yet.

    A source is claimable while its ``candidate_route`` is null; a route lands
    once its candidates are judged and written, so a judged source drops out of
    this set. The claim window bounds a source whose judgment failed: once the
    claim is stale it is reclaimed and retried.
    """
    domain_filter, domain_params = _domain_filter(domains)
    focus_filter = "AND n.id IN $focus_ids" if focus_ids else ""
    return claim_batch(
        "SignalSource",
        facility=facility,
        status_predicate=(
            f"n.status = 'enriched' AND n.candidate_route IS NULL "
            f"{domain_filter} {focus_filter}"
        ),
        status_params={
            **domain_params,
            **({"focus_ids": focus_ids} if focus_ids else {}),
        },
        batch_size=batch_size,
        return_fields=_CANDIDATE_FIELDS,
        return_clause=_REPRESENTATIVE,
        timeout_seconds=CLAIM_TIMEOUT_SECONDS,
        **_MAPPING_CLAIM_FIELDS,
    )


def record_mapping_verdict(
    source_id: str,
    disposition: str,
    evidence: str,
) -> None:
    """Record a chosen ``none`` verdict without touching ``mapping_status``.

    A verdict that names no listed path still deserves its disposition and
    reasoning recorded, so a later run neither claims nor re-asks the source.
    ``mapping_status`` is a phase summary — assigned, mapped, validated — and
    holds no disposition value, so a ``none`` verdict leaves it untouched.
    """
    with GraphClient() as gc:
        gc.query(
            """
            MATCH (sg:SignalSource {id: $id})
            SET sg.mapping_disposition = $disposition,
                sg.mapping_evidence = $evidence,
                sg.mapping_claimed_at = null,
                sg.mapping_claim_token = null
            """,
            id=source_id,
            disposition=disposition,
            evidence=evidence,
        )


def refresh_mapping_status(source_id: str, ids_name: str) -> str | None:
    """Set mapping_status for a source from its selected homes; clear its claim.

    The single owner of the ``assigned``/``mapped``/``validated`` rule. Both
    stages call it at the end of a pass and neither writes a status literal of
    its own. ``ids_name`` names the IDS the calling pass just handled, which is
    what separates the two stages at the same graph state:

    - ``validated`` once every IDS of the source's selected
      ``MAPPING_CANDIDATE`` edges carries a ``MAPS_TO_IMAS`` binding — written
      by the validate stage only after an IDS persists its bindings;
    - ``mapped`` while the pass's own IDS is one of the unbound IDSs, i.e. its
      field mappings have been generated but its bindings are not yet written;
      this is what ``has_pending_validation_work`` selects;
    - ``assigned`` while some selected IDS other than this pass's own is
      unbound, so a source mapped in two IDSs is not called validated when only
      one of them has been validated.

    ``mapping_status`` is left untouched when the source has no selected edge.
    The claim is cleared in every case, so a finished pass releases the source
    for the same IDS's next pass and for the other IDSs' passes.

    Args:
        source_id: ``SignalSource`` ID whose summary is refreshed.
        ids_name: IDS the calling pass just handled.

    Returns:
        The status written, or the source's unchanged status when it has no
        selected edge.
    """
    with GraphClient() as gc:
        rows = gc.query(
            """
            MATCH (sg:SignalSource {id: $id})
            OPTIONAL MATCH (sg)-[c:MAPPING_CANDIDATE]->(:IMASNode)
              WHERE c.route = true
            WITH sg, [i IN collect(DISTINCT c.ids) WHERE i IS NOT NULL] AS sel_ids
            WITH sg, sel_ids,
                 [i IN sel_ids WHERE NOT EXISTS {
                     (sg)-[:MAPS_TO_IMAS]->(:IMASNode {ids: i})
                 }] AS unbound
            SET sg.mapping_status = CASE
                    WHEN size(sel_ids) = 0 THEN sg.mapping_status
                    WHEN size(unbound) = 0 THEN 'validated'
                    WHEN $ids_name IN unbound THEN 'mapped'
                    ELSE 'assigned' END,
                sg.mapping_claimed_at = null,
                sg.mapping_claim_token = null
            RETURN sg.mapping_status AS status
            """,
            id=source_id,
            ids_name=ids_name,
        )
        return rows[0]["status"] if rows else None


def release_mapping_claim(source_id: str) -> None:
    """Release a mapping claim without changing status."""
    release_claim("SignalSource", source_id, **_MAPPING_CLAIM_FIELDS)


def release_mapping_claims_batch(source_ids: list[str]) -> None:
    """Release mapping claims on multiple sources."""
    release_claims_batch("SignalSource", source_ids, **_MAPPING_CLAIM_FIELDS)


def _escalated_source_predicate(
    ids_names: list[str] | None,
    domains: list[str] | None = None,
) -> tuple[str, dict[str, Any]]:
    """Select unresolved escalated sources with a candidate in a target IDS."""
    domain_filter, params = _domain_filter(domains)
    ids_clause = ""
    if ids_names:
        ids_clause = (
            "AND EXISTS { (n)-[r:MAPPING_CANDIDATE]->(:IMASNode) "
            "WHERE r.ids IN $ids_names } "
        )
        params["ids_names"] = list(ids_names)
    return (
        "n.candidate_route = 'escalated' "
        "AND n.mapping_disposition IS NULL " + ids_clause + domain_filter,
        params,
    )


def claim_sources_for_escalated(
    facility: str,
    ids_names: list[str] | None = None,
    domains: list[str] | None = None,
    batch_size: int = 20,
) -> list[dict[str, Any]]:
    """Claim escalated sources whose shortlist the reasoning seat must choose from.

    A source is claimable while its ``candidate_route`` is ``escalated`` and it
    carries at least one candidate the map run can target. ``ids_names``
    restricts the claim to sources with a candidate in one of the run's target
    IDSs. Choosing marks the picked edges ``selected`` and sets the route to
    ``selected`` in one statement, so a chosen source drops out of this set and
    is claimed next by the per-IDS map pass.
    """
    predicate, params = _escalated_source_predicate(ids_names, domains)
    return claim_batch(
        "SignalSource",
        facility=facility,
        status_predicate=predicate,
        status_params=params,
        batch_size=batch_size,
        return_fields=_ASSIGNMENT_FIELDS,
        return_clause=_ASSIGNMENT_RETURN,
        timeout_seconds=CLAIM_TIMEOUT_SECONDS,
        **_MAPPING_CLAIM_FIELDS,
    )


def has_pending_assignment_work(
    facility: str,
    ids_names: list[str] | None = None,
) -> bool:
    """Check if escalated sources with a target IDS candidate remain unselected."""
    predicate, params = _escalated_source_predicate(ids_names)
    return has_pending(
        "SignalSource",
        facility=facility,
        status_predicate=predicate,
        status_params=params,
    )


def has_pending_mapping_work(
    facility: str,
    ids_name: str | None = None,
    handled_source_ids: list[str] | None = None,
) -> bool:
    """Check for selected, unbound homes outside the already handled sources."""
    ids_filter = "AND c.ids = $ids_name " if ids_name else ""
    handled_filter = "AND NOT n.id IN $handled_source_ids" if handled_source_ids else ""
    return has_pending(
        "SignalSource",
        facility=facility,
        status_predicate=(
            "EXISTS { (n)-[c:MAPPING_CANDIDATE]->(ip:IMASNode) "
            f"WHERE c.route = true {ids_filter}"
            "AND NOT EXISTS { (n)-[:MAPS_TO_IMAS]->(:IMASNode {ids: ip.ids}) } } "
            f"{handled_filter}"
        ),
        status_params={
            **({"ids_name": ids_name} if ids_name else {}),
            **(
                {"handled_source_ids": handled_source_ids} if handled_source_ids else {}
            ),
        },
    )


def unmapped_selected_sources(facility: str, ids_name: str) -> list[str]:
    """Return selected sources without a persisted binding in this IDS."""
    with GraphClient() as gc:
        rows = gc.query(
            """
            MATCH (sg:SignalSource {facility_id: $facility})
                  -[c:MAPPING_CANDIDATE]->(ip:IMASNode)
            WHERE c.route = true AND c.ids = $ids_name
              AND NOT EXISTS {
                  (sg)-[:MAPS_TO_IMAS]->(:IMASNode {ids: $ids_name})
              }
            RETURN DISTINCT sg.id AS source_id
            ORDER BY source_id
            """,
            facility=facility,
            ids_name=ids_name,
        )
        return [row["source_id"] for row in rows]


def has_pending_validation_work(facility: str) -> bool:
    """Check if mapped-but-unvalidated sources exist."""
    return has_pending(
        "SignalSource",
        facility=facility,
        status_predicate="n.mapping_status = 'mapped'",
    )


def has_pending_candidate_work(
    facility: str,
    domains: list[str] | None = None,
    focus_ids: list[str] | None = None,
) -> bool:
    """Check if enriched sources remain whose candidates are unjudged."""
    domain_filter, domain_params = _domain_filter(domains)
    focus_filter = "AND n.id IN $focus_ids" if focus_ids else ""
    return has_pending(
        "SignalSource",
        facility=facility,
        status_predicate=(
            f"n.status = 'enriched' AND n.candidate_route IS NULL "
            f"{domain_filter} {focus_filter}"
        ),
        status_params={
            **domain_params,
            **({"focus_ids": focus_ids} if focus_ids else {}),
        },
    )


def clear_mappings_for_ids(facility: str, ids_names: list[str]) -> dict[str, int]:
    """Delete each IDS's whole mapping through the single delete owner.

    Routes the engine's clear path through ``delete_mapping``, so
    ``map run --clear`` removes the bindings, the ``MappingEvidence`` nodes and
    the ``IMASMapping`` node ``map clear`` removes instead of resetting source
    status alone and leaving a stale mapping behind.

    Args:
        facility: Facility ID.
        ids_names: IDS names whose mappings are deleted.

    Returns:
        ``{"mappings": n, "bindings": n, "evidence": n}`` summed over the IDSs.
    """
    counts = {"mappings": 0, "bindings": 0, "evidence": 0}
    if not ids_names:
        return counts
    with GraphClient() as gc:
        for ids_name in ids_names:
            deleted = delete_mapping(facility, ids_name, gc)
            for key in counts:
                counts[key] += deleted[key]
    return counts


def reset_mapping_state(
    facility: str,
    ids_names: list[str] | None = None,
) -> int:
    """Clear mapping_status on sources for fresh re-mapping."""
    params: dict[str, Any] = {"facility": facility}
    ids_filter = ""
    if ids_names:
        ids_filter = (
            "AND EXISTS { (sg)-[:MAPPING_CANDIDATE]->(ip:IMASNode) "
            "WHERE ip.ids IN $ids_names }"
        )
        params["ids_names"] = ids_names

    with GraphClient() as gc:
        result = gc.query(
            f"""
            MATCH (sg:SignalSource {{facility_id: $facility}})
            WHERE sg.mapping_status IS NOT NULL
              {ids_filter}
            SET sg.mapping_status = null,
                sg.mapping_claimed_at = null,
                sg.mapping_claim_token = null,
                sg.mapping_disposition = null,
                sg.mapping_evidence = null
            RETURN count(sg) AS cleared
            """,
            **params,
        )
        return result[0]["cleared"] if result else 0


# =============================================================================
# Workers
# =============================================================================


async def context_worker(
    state: MappingDiscoveryState,
    on_progress: Callable | None = None,
    **_kwargs,
) -> None:
    """Gather context for ALL IDS targets with shared embedding.

    Two phases:
    1. Shared: batch embed all sources, wiki/code context (ONCE)
    2. Per-IDS: vector queries using pre-computed embeddings (fast)
    """
    wlog = WorkerLogAdapter(logger, worker_name="context_worker")
    wlog.info(
        "Gathering context for %d IDS targets: %s",
        len(state.target_ids_list),
        state.target_ids_list,
    )

    from imas_codex.ids.mapping import gather_ids_context, gather_shared_context

    def _shared_progress(detail: str) -> None:
        if on_progress:
            on_progress(detail, state.context_stats, [{"detail": detail}])

    # Gather source embeddings and external context once for all IDS targets.
    shared = await asyncio.to_thread(
        gather_shared_context,
        state.facility,
        state.target_ids_list,
        gc=GraphClient(),
        dd_version=state.dd_major,
        on_progress=_shared_progress,
    )

    state.sources_total = len(shared["groups"])

    # Query each IDS using the shared embeddings.
    for i, ids_name in enumerate(state.target_ids_list):
        if state.should_stop():
            break

        def _ids_progress(detail: str, _ids=ids_name, _i=i) -> None:
            if on_progress:
                label = (
                    f"{_ids}: {detail}" if len(state.target_ids_list) > 1 else detail
                )
                on_progress(label, state.context_stats, [{"detail": label}])

        _ids_progress("subtree + filters")
        context = await asyncio.to_thread(
            gather_ids_context,
            state.facility,
            ids_name,
            shared,
            gc=GraphClient(),
            on_progress=_ids_progress,
        )
        state.contexts[ids_name] = context

        wlog.info(
            "Context for %s: %d candidates",
            ids_name,
            len(context.get("source_candidates", {})),
        )

    state.context_stats.processed = len(state.target_ids_list)
    wlog.info(
        "Context complete: %d IDS, %d total sources",
        len(state.contexts),
        state.sources_total,
    )
    state.context_phase.mark_done()


async def assign_worker(
    state: MappingDiscoveryState,
    on_progress: Callable | None = None,
    **_kwargs,
) -> None:
    """Choose target paths for each escalated source from its own shortlist.

    Claim loop: claims escalated sources, reads each one's candidate shortlist
    through :func:`read_candidates`, asks the reasoning seat which listed paths
    hold its values, and marks the picked edges selected through
    :func:`select_candidates`, which sets the source's route to ``selected``.
    A source the candidate stage already selected carries its marked edges and
    is not re-asked; a ``no_candidate`` source is skipped upstream. A source
    whose choice names no listed path has its disposition and reasoning
    recorded as its mapping evidence, so a later run does not claim and
    re-ask it.
    """
    wlog = WorkerLogAdapter(logger, worker_name="assign_worker")

    from imas_codex.ids.graph_ops import read_candidates, select_candidates
    from imas_codex.ids.mapping import achoose_targets, escalated_shortlist

    shortlist_size = get_mapping_route_thresholds().shortlist_size

    # Sources whose choice was refused or that picked no path are retried on a
    # later run, not within this one: a stable `handled` set keeps the claim
    # loop from re-reading a source that keeps returning the same answer.
    handled: set[str] = set()

    while not state.should_stop():
        sources = await asyncio.to_thread(
            claim_sources_for_escalated,
            state.facility,
            state.target_ids_list,
            batch_size=1,
        )
        sources = [s for s in sources if s["id"] not in handled]
        if not sources:
            state.assign_phase.record_idle()
            if state.assign_phase.done:
                break
            await asyncio.sleep(2.0)
            continue

        state.assign_phase.record_activity(len(sources))

        with GraphClient() as gc:
            candidate_edges = await asyncio.to_thread(
                read_candidates, [s["id"] for s in sources], gc
            )

            for source in sources:
                if state.should_stop():
                    return
                source_id = source["id"]
                handled.add(source_id)

                shortlist = escalated_shortlist(
                    candidate_edges.get(source_id, []), shortlist_size
                )
                if not shortlist:
                    wlog.warning(
                        "Escalated %s has no candidate to choose from, releasing",
                        source_id,
                    )
                    await asyncio.to_thread(release_mapping_claim, source_id)
                    continue

                try:
                    choice = await achoose_targets(
                        state.facility,
                        source,
                        shortlist,
                        model=state.model,
                        cost=state.cost,
                    )
                except Exception as e:
                    wlog.error("Choice failed for %s: %s", source_id, e)
                    state.assign_stats.errors += 1
                    await asyncio.to_thread(release_mapping_claim, source_id)
                    continue

                if not choice.paths:
                    # Persist the verdict so the source is not claimed and
                    # re-asked, and the reasoning seat not paid again, on a
                    # later run. The disposition and its reasoning become the
                    # source's mapping evidence; the candidate route is left
                    # as the candidate stage wrote it.
                    await asyncio.to_thread(
                        record_mapping_verdict,
                        source_id,
                        choice.disposition.value,
                        choice.reasoning,
                    )
                    state.assign_stats.processed += 1
                    wlog.info(
                        "No listed path for %s: disposition=%s recorded",
                        source_id,
                        choice.disposition.value,
                    )
                    continue

                marked = await asyncio.to_thread(
                    select_candidates, source_id, choice.paths, gc
                )
                # Assignment and mapping share the claim fields. Once the
                # selected edges are durable, release the source for mapping.
                await asyncio.to_thread(release_mapping_claim, source_id)
                state.sources_assigned += 1
                state.assign_stats.processed += 1

                wlog.info(
                    "Selected %d paths for %s, cost $%.4f",
                    marked,
                    source_id,
                    state.cost.total_usd,
                )

                if on_progress:
                    on_progress(
                        f"{source_id} -> {marked} selected",
                        state.assign_stats,
                        [
                            {
                                "source_id": source_id,
                                "target_path": ", ".join(choice.paths),
                                "physics_domain": source.get("physics_domain", ""),
                            }
                        ],
                    )

    state.assign_phase.mark_done()


async def map_worker(
    state: MappingDiscoveryState,
    on_progress: Callable | None = None,
    **_kwargs,
) -> None:
    """Claim each IDS's selected sources and generate field-level mappings.

    Claim loop: for each IDS target, claims sources that selected a home there
    and are not yet bound there, builds one :class:`TargetAssignment` per
    selected section from their selected edges, generates a mapping per
    section, and refreshes each source's ``mapping_status`` summary. A source
    already handled this run is not re-claimed, so a source with selected edges
    in two IDSs is handled once by each IDS's pass.
    """
    wlog = WorkerLogAdapter(logger, worker_name="map_worker")

    from imas_codex.ids.graph_ops import read_candidates
    from imas_codex.ids.mapping import (
        _acall_llm,
        _build_messages,
        _prepare_section_context,
        build_target_assignments,
    )
    from imas_codex.ids.models import SignalMappingBatch

    handled: set[tuple[str, str]] = set()

    while not state.should_stop():
        found_any = False
        for ids_name in state.target_ids_list:
            if state.should_stop():
                break

            claimed = await asyncio.to_thread(
                claim_sources_for_mapping,
                state.facility,
                ids_name,
                batch_size=3,
            )
            state.mapping_claims.update(source["id"] for source in claimed)
            # ``handled`` is keyed by (IDS, source): the same source claimed by
            # this IDS's pass is still eligible for every other IDS's pass,
            # and the claim this pass cannot use is released at once.
            sources = [s for s in claimed if (ids_name, s["id"]) not in handled]
            skipped = [s["id"] for s in claimed if (ids_name, s["id"]) in handled]
            if skipped:
                await asyncio.to_thread(release_mapping_claims_batch, skipped)
                state.mapping_claims.difference_update(skipped)
            if not sources:
                continue

            found_any = True
            handled.update((ids_name, s["id"]) for s in sources)
            state.map_phase.record_activity(len(sources))

            context = state.contexts.get(ids_name, {})

            with GraphClient() as gc:
                candidate_edges = await asyncio.to_thread(
                    read_candidates, [s["id"] for s in sources], gc
                )
            built = build_target_assignments(
                ids_name, [s["id"] for s in sources], candidate_edges
            )
            existing = state.assignments.get(ids_name)
            if existing is None:
                state.assignments[ids_name] = built
            else:
                existing.assignments.extend(built.assignments)

            for source in sources:
                if state.should_stop():
                    release_mapping_claims_batch([s["id"] for s in sources])
                    state.mapping_claims.difference_update(s["id"] for s in sources)
                    return

                source_id = source["id"]

                # One assignment per selected section: a source with selected
                # nodes in two sections of this IDS is mapped once per section.
                assignments = [a for a in built.assignments if a.source_id == source_id]

                if not assignments:
                    wlog.warning(
                        "No selected candidate for %s in %s, releasing",
                        source_id,
                        ids_name,
                    )
                    await asyncio.to_thread(release_mapping_claim, source_id)
                    state.mapping_claims.discard(source_id)
                    continue

                for assignment in assignments:
                    target_path = assignment.imas_target_path
                    try:
                        prep = await asyncio.to_thread(
                            _prepare_section_context,
                            state.facility,
                            ids_name,
                            assignment,
                            context,
                            gc=GraphClient(),
                            dd_version=context.get("dd_version"),
                        )
                        messages = _build_messages(
                            "signal_mapping_system",
                            prep["prompt"],
                        )
                        batch = await _acall_llm(
                            messages,
                            SignalMappingBatch,
                            model=state.model,
                            step_name=f"map_signals_{target_path}",
                            cost=state.cost,
                        )

                        state.mapping_batches.setdefault(ids_name, []).append(
                            (assignment, batch),
                        )
                        state.sources_mapped += 1
                        state.bindings_total += len(batch.mappings)
                        state.map_stats.processed += 1

                        await asyncio.to_thread(
                            refresh_mapping_status, source_id, ids_name
                        )
                        state.mapping_claims.discard(source_id)

                        wlog.info(
                            "Mapped %s -> %s: %d bindings",
                            source_id,
                            target_path,
                            len(batch.mappings),
                        )

                        if on_progress:
                            sg = next(
                                (
                                    g
                                    for g in context.get("groups", [])
                                    if g["id"] == source_id
                                ),
                                {},
                            )
                            on_progress(
                                f"{source_id} -> {target_path}",
                                state.map_stats,
                                [
                                    {
                                        "source_id": source_id,
                                        "target_path": target_path,
                                        "physics_domain": sg.get("physics_domain", ""),
                                        "bindings": len(batch.mappings),
                                    }
                                ],
                            )

                    except Exception as e:
                        wlog.error("Mapping failed for %s: %s", source_id, e)
                        await asyncio.to_thread(release_mapping_claim, source_id)
                        state.mapping_claims.discard(source_id)
                        state.map_stats.errors += 1
                        raise

        if not found_any:
            state.map_phase.record_idle()
            # A selected source can still carry an assign claim from an
            # earlier run. Keep polling until it is claimable or recovered.
            # Ignore sources already handled here: validation writes their
            # bindings only after this phase completes.
            pending = False
            for ids_name in state.target_ids_list:
                handled_ids = [
                    source_id
                    for handled_ids_name, source_id in handled
                    if handled_ids_name == ids_name
                ]
                if await asyncio.to_thread(
                    has_pending_mapping_work, state.facility, ids_name, handled_ids
                ):
                    pending = True
                    break
            if state.assign_phase.done and not pending:
                state.map_phase.mark_done()
                break
            await asyncio.sleep(2.0)


async def validate_worker(
    state: MappingDiscoveryState,
    on_progress: Callable | None = None,
    stop_reason: str | None = None,
    **_kwargs,
) -> None:
    """Validate mapped batches per IDS, including a bounded run's shutdown."""
    wlog = WorkerLogAdapter(logger, worker_name="validate_worker")

    from imas_codex.ids.mapping import (
        AssemblyBatch,
        adiscover_assembly,
        validate_mappings,
    )
    from imas_codex.ids.models import persist_mapping_result

    for ids_name in state.target_ids_list:
        if state.should_stop() and stop_reason is None:
            break
        if ids_name in state.ids_results:
            continue

        batches_for_ids = state.mapping_batches.get(ids_name, [])
        sections = state.assignments.get(ids_name)
        context = state.contexts.get(ids_name, {})

        if not batches_for_ids or not sections:
            wlog.info("No mappings for %s, skipping validation", ids_name)
            continue

        # A claimed batch may have assigned more sources than the map phase
        # completed. Validation must see only the batches that exist.
        sections = sections.model_copy(
            update={"assignments": [assignment for assignment, _ in batches_for_ids]}
        )
        mapped_sources = {assignment.source_id for assignment, _ in batches_for_ids}
        remaining_sources = (
            [
                source_id
                for source_id in await asyncio.to_thread(
                    unmapped_selected_sources, state.facility, ids_name
                )
                if source_id not in mapped_sources
            ]
            if stop_reason is not None
            else []
        )

        if on_progress:
            on_progress(f"validating {ids_name}", state.validate_stats)

        gc = GraphClient()
        field_batches = [b for _, b in batches_for_ids]

        try:
            # Assembly discovery
            configs = []
            async for _assignment, config in adiscover_assembly(
                state.facility,
                ids_name,
                sections,
                field_batches,
                context,
                gc=gc,
                model=state.model,
                cost=state.cost,
            ):
                configs.append(config)
                if on_progress:
                    on_progress(
                        f"assembly {len(configs)}/{len(sections.assignments)}",
                        state.validate_stats,
                        [
                            {
                                "target_path": config.target_path,
                                "pattern": (
                                    config.pattern.value
                                    if hasattr(config.pattern, "value")
                                    else str(config.pattern)
                                ),
                            }
                        ],
                    )

            assembly = AssemblyBatch(ids_name=ids_name, configs=configs)

            # Validation
            dd_version_str = state.dd_version or ""
            validated = await asyncio.to_thread(
                validate_mappings,
                state.facility,
                ids_name,
                dd_version_str,
                sections,
                field_batches,
                gc=gc,
            )

            # Derive error mappings from validated bindings unless skipped.
            if not state.skip_errors:
                from imas_codex.ids.mapping import derive_error_mappings

                error_bindings = await asyncio.to_thread(
                    derive_error_mappings,
                    validated.bindings,
                    gc=gc,
                    facility=state.facility,
                )
                if error_bindings:
                    validated.bindings.extend(error_bindings)
                    wlog.info(
                        "Derived %d error mappings for %s",
                        len(error_bindings),
                        ids_name,
                    )

            ids_passed = len(validated.bindings)
            ids_escalations = len(validated.escalations)

            if on_progress:
                on_progress(
                    f"{ids_name}: {ids_passed} passed, {ids_escalations} escalations",
                    state.validate_stats,
                    [
                        {
                            "target_path": ids_name,
                            "passed": ids_passed,
                            "escalations": ids_escalations,
                        }
                    ],
                )

            # Persist
            mapping_id = None
            if state.persist:
                status = (
                    "active" if state.activate and stop_reason is None else "generated"
                )
                mapping_id = await asyncio.to_thread(
                    persist_mapping_result,
                    validated,
                    assembly=assembly,
                    gc=gc,
                    status=status,
                    partial=bool(remaining_sources),
                    unmapped_sources=remaining_sources,
                    stop_reason=stop_reason if remaining_sources else None,
                )
                wlog.info(
                    "Persisted %s mapping %s (%s)",
                    ids_name,
                    mapping_id,
                    status,
                )

            # Refresh each source's status through the single owner. It writes
            # 'validated' only once every selected IDS of the source is bound,
            # and leaves 'assigned' while another selected IDS is still unbound.
            for a, _ in batches_for_ids:
                await asyncio.to_thread(
                    refresh_mapping_status,
                    a.source_id,
                    ids_name,
                )

            state.bindings_passed += ids_passed
            state.escalations += ids_escalations
            state.sources_validated += len(batches_for_ids)
            state.validate_stats.processed += 1
            state.ids_results[ids_name] = {
                "bindings": ids_passed,
                "escalations": ids_escalations,
            }

            wlog.info(
                "Validated %s: %d passed, %d escalations, cost $%.4f",
                ids_name,
                ids_passed,
                ids_escalations,
                state.cost.total_usd,
            )

        except Exception as e:
            wlog.error("Validation failed for %s: %s", ids_name, e)
            state.validate_stats.errors += 1
            raise

    state.validate_phase.mark_done()


def _candidate_records(
    judgments: list,
    candidates: list,
    selected: set[str],
) -> list[dict[str, Any]]:
    """Build the per-candidate records ``write_candidates`` persists.

    Records are ranked in Jev order (descending ``p_same_quantity``); the
    retrieval score and IDS are read back from the matching candidate, and its
    retrieval arms are emitted so a sibling can be told from a retrieval hit.
    A record is routed when its path is in ``selected``. A judgment whose path
    matches no retrieved candidate is refused, rather than written with empty
    arms, which would bypass ``write_candidates``' refusal of absent arms.
    """
    score_by_path = {
        candidate.hit.path: (candidate.hit.score, candidate.hit.ids_name)
        for candidate in candidates
    }
    arms_by_path = {
        candidate.hit.path: sorted(candidate.arms) for candidate in candidates
    }
    ordered = sorted(
        judgments, key=lambda judgment: judgment.p_same_quantity, reverse=True
    )
    records: list[dict[str, Any]] = []
    for rank, judgment in enumerate(ordered, start=1):
        if judgment.path not in score_by_path:
            raise CandidateWriteError(
                f"judgment path {judgment.path!r} has no retrieved candidate"
            )
        score, ids_name = score_by_path[judgment.path]
        records.append(
            {
                "path": judgment.path,
                "rank": rank,
                "retrieval_score": score,
                "ids": ids_name,
                "p_same_quantity": judgment.p_same_quantity,
                "model": judgment.model,
                "judged_at": judgment.judged_at,
                "route": judgment.path in selected,
                "arms": arms_by_path[judgment.path],
            }
        )
    return records


def _facility_block(facility: str) -> dict[str, Any]:
    """A compact facility block to ground a candidate judgment."""
    from imas_codex.discovery.base.facility import get_facility

    try:
        config = get_facility(facility)
    except Exception:
        return {"facility_id": facility}
    return {
        "facility_id": facility,
        "description": config.get("description", ""),
    }


async def candidate_worker(
    state: CandidateDiscoveryState,
    on_progress: Callable | None = None,
    **_kwargs,
) -> None:
    """Claim unjudged sources and route their DD candidates.

    Claim loop: claims batches of enriched sources whose ``candidate_route`` is
    unset, routes each source to its most probable IDSs, retrieves candidates
    within those IDSs, judges them and writes the route and edges. A decisions
    transport failure releases the source's claim so a later pass retries it.
    """
    wlog = WorkerLogAdapter(logger, worker_name="candidate_worker")

    # Without the decisions key every source the loop claims can only fail to
    # judge, so it would claim and release until its deadline. End the stage
    # before the first claim instead: ``judgments_available`` reports the
    # absence once, and every source stays unjudged for a later run with the
    # credential configured.
    if not judgments_available():
        state.candidate_phase.mark_done()
        return

    facility_block = _facility_block(state.facility)
    thresholds = get_mapping_route_thresholds()

    while not state.should_stop():
        batch_size = state.batch_size
        if state.source_limit is not None:
            remaining = state.source_limit - state.sources_judged
            if remaining <= 0:
                break
            batch_size = min(batch_size, remaining)

        claim_options = {"domains": state.domains or None, "batch_size": batch_size}
        if state.focus_ids:
            claim_options["focus_ids"] = state.focus_ids
        sources = await asyncio.to_thread(
            claim_sources_for_candidates, state.facility, **claim_options
        )
        if not sources:
            state.candidate_phase.record_idle()
            if state.candidate_phase.done:
                break
            await asyncio.sleep(2.0)
            continue

        state.candidate_phase.record_activity(len(sources))

        with GraphClient() as gc:
            descriptions: dict[str, str] = {}
            routed: dict[str, list[str]] = {}
            for source in sources:
                if state.should_stop():
                    break
                ids = await asyncio.to_thread(
                    route_ids,
                    source.get("description") or "",
                    gc=gc,
                    model=state.model,
                    cost=state.cost,
                )
                if ids is None:
                    await asyncio.to_thread(release_mapping_claim, source["id"])
                    continue
                if state.ids_filter:
                    ids = [name for name in ids if name in state.ids_filter]
                descriptions[source["id"]] = source.get("description") or ""
                routed[source["id"]] = ids

            if not routed:
                continue

            retrieved = await asyncio.to_thread(
                retrieve_candidates,
                descriptions,
                routed,
                gc=gc,
                dd_version=state.dd_version,
            )

            for source in sources:
                source_id = source["id"]
                if source_id not in routed:
                    continue
                if state.should_stop():
                    await asyncio.to_thread(release_mapping_claim, source_id)
                    continue

                candidates = retrieved.get(source_id, [])
                judgments = await asyncio.to_thread(
                    judge_candidates,
                    source,
                    facility_block,
                    candidates,
                    model=state.model,
                    cost=state.cost,
                )
                if judgments is None:
                    wlog.warning(
                        "No route for %s (decisions failed), releasing", source_id
                    )
                    await asyncio.to_thread(release_mapping_claim, source_id)
                    continue

                expansion = await asyncio.to_thread(
                    expand_cluster_siblings,
                    source,
                    facility_block,
                    candidates,
                    judgments,
                    gc=gc,
                    model=state.model,
                    cost=state.cost,
                    dd_version=state.dd_version,
                )
                if expansion is None:
                    wlog.warning(
                        "No cluster sibling judgment for %s (decisions failed), "
                        "releasing",
                        source_id,
                    )
                    await asyncio.to_thread(release_mapping_claim, source_id)
                    continue
                siblings, sibling_judgments = expansion
                all_candidates = list(candidates) + siblings
                all_judgments = list(judgments) + sibling_judgments
                decision = route(all_judgments, thresholds)
                if decision is None:
                    wlog.warning(
                        "No route for %s (decisions failed), releasing", source_id
                    )
                    await asyncio.to_thread(release_mapping_claim, source_id)
                    continue

                # A source with no candidate is skipped; its best path and
                # probability are logged as the evidence for the skip.
                if decision.decision == "no_candidate":
                    best = max(
                        all_judgments,
                        key=lambda j: j.p_same_quantity,
                        default=None,
                    )
                    wlog.info(
                        "No candidate for %s: best path %s "
                        "(p_same_quantity=%s), skipping",
                        source_id,
                        best.path if best else None,
                        f"{best.p_same_quantity:.3f}" if best else None,
                    )

                records = _candidate_records(
                    all_judgments, all_candidates, set(decision.selected)
                )
                written = await asyncio.to_thread(
                    write_candidates,
                    source_id,
                    records,
                    decision.decision,
                    gc,
                )
                state.sources_judged += 1
                state.candidates_written += written
                state.candidate_stats.processed += 1

                wlog.info(
                    "Judged %s: %s (%d candidates, cost $%.4f)",
                    source_id,
                    decision.decision,
                    written,
                    state.cost.total_usd,
                )

                if on_progress:
                    on_progress(
                        f"{source_id} -> {decision.decision}",
                        state.candidate_stats,
                        [
                            {
                                "source_id": source_id,
                                "route": decision.decision,
                                "candidates": written,
                            }
                        ],
                    )

    state.candidate_phase.mark_done()


# =============================================================================
# Engine Entry Point
# =============================================================================


async def run_mapping_engine(
    state: MappingDiscoveryState,
    *,
    stop_event: asyncio.Event | None = None,
    on_progress: Callable | None = None,
) -> None:
    """Run the mapping pipeline as a discovery engine.

    Wires up graph-based phase completion checks, orphan recovery,
    and supervised workers via ``run_discovery_engine``.
    """
    # Wire has_work_fn for phase completion detection
    state.assign_phase.set_has_work_fn(
        lambda: (
            has_pending_assignment_work(state.facility, state.target_ids_list)
            or not state.context_phase.done
        )
    )
    state.map_phase.set_has_work_fn(
        lambda: (
            any(
                has_pending_mapping_work(state.facility, ids_name)
                for ids_name in state.target_ids_list
            )
            or not state.assign_phase.done
        )
    )
    state.validate_phase.set_has_work_fn(
        lambda: has_pending_validation_work(state.facility) or not state.map_phase.done
    )

    # Clear previous mappings if requested: drop the whole mapping through its
    # delete owner, then reset source status for re-mapping.
    if state.clear:
        cleared_counts = await asyncio.to_thread(
            clear_mappings_for_ids,
            state.facility,
            state.target_ids_list,
        )
        logger.info(
            "Cleared %d mappings, %d bindings, %d evidence for %s",
            cleared_counts["mappings"],
            cleared_counts["bindings"],
            cleared_counts["evidence"],
            state.facility,
        )
        cleared = await asyncio.to_thread(
            reset_mapping_state,
            state.facility,
            state.target_ids_list,
        )
        if cleared:
            logger.info("Cleared mapping state for %d sources", cleared)

    workers = [
        WorkerSpec(
            "context",
            "context_phase",
            context_worker,
            on_progress=on_progress,
        ),
        WorkerSpec(
            "assign",
            "assign_phase",
            assign_worker,
            on_progress=on_progress,
            depends_on=["context_phase"],
        ),
        WorkerSpec(
            "map",
            "map_phase",
            map_worker,
            on_progress=on_progress,
            depends_on=["context_phase"],  # can overlap with assign
        ),
        WorkerSpec(
            "validate",
            "validate_phase",
            validate_worker,
            on_progress=on_progress,
            depends_on=["map_phase"],
        ),
    ]

    orphan_specs = [
        OrphanRecoverySpec(
            label="SignalSource",
            facility_field="facility_id",
            timeout_seconds=CLAIM_TIMEOUT_SECONDS,
            claimed_field="mapping_claimed_at",
        ),
    ]

    await run_discovery_engine(
        state,
        workers,
        stop_event=stop_event,
        orphan_specs=orphan_specs,
    )

    if state.mapping_claims:
        await asyncio.to_thread(
            release_mapping_claims_batch, sorted(state.mapping_claims)
        )
        state.mapping_claims.clear()

    stop_reason = (
        "deadline"
        if state.deadline_expired
        else "cost_limit"
        if state.budget_exhausted
        else None
    )
    if stop_reason and state.mapping_batches:
        await validate_worker(state, on_progress=on_progress, stop_reason=stop_reason)


async def run_candidate_engine(
    state: CandidateDiscoveryState,
    *,
    stop_event: asyncio.Event | None = None,
    on_progress: Callable | None = None,
) -> None:
    """Run the candidate judgment pipeline as a discovery engine.

    A single candidate worker claims unjudged sources until none remain; the
    phase's completion check reads the graph for unjudged sources, and orphan
    recovery releases candidate claims left stale on ``mapping_claimed_at``.
    """
    state.candidate_phase.set_has_work_fn(
        lambda: has_pending_candidate_work(
            state.facility, state.domains or None, state.focus_ids or None
        )
    )

    workers = [
        WorkerSpec(
            "candidate",
            "candidate_phase",
            candidate_worker,
            on_progress=on_progress,
        ),
    ]

    orphan_specs = [
        OrphanRecoverySpec(
            label="SignalSource",
            facility_field="facility_id",
            timeout_seconds=CLAIM_TIMEOUT_SECONDS,
            claimed_field="mapping_claimed_at",
        ),
    ]

    await run_discovery_engine(
        state,
        workers,
        stop_event=stop_event,
        orphan_specs=orphan_specs,
    )
