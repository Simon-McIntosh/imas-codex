"""SN loop — drives ``sn run`` via concurrent worker pools.

Primary entry point: :func:`run_sn_pools` (6-pool concurrent orchestrator).
"""

from __future__ import annotations

import asyncio
import json as _json
import logging
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from imas_codex.standard_names.defaults import DEFAULT_MIN_SCORE

logger = logging.getLogger(__name__)


@dataclass
class RunSummary:
    """Aggregated result of a ``sn run`` invocation."""

    run_id: str
    turn_number: int
    started_at: datetime
    stopped_at: datetime | None = None
    cost_spent: float = 0.0
    cost_limit: float = 0.0
    time_limit_s: float | None = None
    min_score: float | None = None
    names_composed: int = 0
    names_enriched: int = 0
    names_reviewed: int = 0
    names_regenerated: int = 0
    sources_reconciled: int = 0
    links_resolved: int = 0
    domains_touched: set[str] = field(default_factory=set)
    stop_reason: str = "completed"
    pass_records: list[dict[str, Any]] = field(default_factory=list)
    compose_cost: float = 0.0
    review_cost: float = 0.0


# ── Status mapping ────────────────────────────────────────────────────
# Map RunSummary.stop_reason to SNRun.status lifecycle values.
_STOP_TO_STATUS: dict[str, str] = {
    "completed": "completed",
    "budget_exhausted": "completed",
    "budget_saturated": "completed",
    "provider_budget_exhausted": "degraded",
    "time_limit_reached": "completed",
    "stalled": "completed",
    "no_work": "completed",
    "no_eligible_work": "completed",
    "dry_run": "completed",
    "interrupted": "interrupted",
    "failed": "failed",
    "degraded": "degraded",
}


def summary_table(summary: RunSummary) -> dict[str, Any]:
    """Flatten a :class:`RunSummary` for Rich display / JSON output."""
    return {
        "run_id": summary.run_id,
        "turn_number": summary.turn_number,
        "started_at": summary.started_at.isoformat(),
        "stopped_at": summary.stopped_at.isoformat() if summary.stopped_at else None,
        "elapsed_s": (
            (summary.stopped_at - summary.started_at).total_seconds()
            if summary.stopped_at
            else None
        ),
        "cost_spent": round(summary.cost_spent, 6),
        "cost_limit": summary.cost_limit,
        "min_score": summary.min_score,
        "names_composed": summary.names_composed,
        "names_enriched": summary.names_enriched,
        "names_reviewed": summary.names_reviewed,
        "names_regenerated": summary.names_regenerated,
        "sources_reconciled": summary.sources_reconciled,
        "links_resolved": summary.links_resolved,
        "domains_touched": sorted(summary.domains_touched),
        "stop_reason": summary.stop_reason,
    }


# ═══════════════════════════════════════════════════════════════════════
# Phase 8 — Pool-based orchestrator (replaces domain rotation)
# ═══════════════════════════════════════════════════════════════════════

# Default regen threshold when min_score is not explicitly provided.
# Imported from defaults.py — do not re-define here.


def _build_pool_specs(
    mgr: Any,
    stop_event: asyncio.Event,
    *,
    min_score: float | None = None,
    rotation_cap: int | None = None,
    escalation_model: str | None = None,
    review_name_backlog_cap: int | None = None,
    review_docs_backlog_cap: int | None = None,
    on_event: Callable[[dict[str, Any]], None] | None = None,
    only_domain: str | None = None,
    scope_run_id: str | None = None,
    names_only: bool = False,
    flush: bool = False,
    skip_review: bool = False,
) -> list[Any]:
    """Construct 7 :class:`PoolSpec` objects wiring claims → batch processors.

    Each pool gets two adapter closures:

    * **claim adapter** — runs the synchronous ``claim_*_seed_and_expand``
      graph function in a worker thread and returns the result wrapped in
      a dict (``{"items": [...]}``), or ``None`` on empty.
    * **process adapter** — unpacks the claimed batch and delegates to the
      corresponding ``process_*_batch`` async function, forwarding the
      shared :class:`BudgetManager` and ``stop_event``.

    After construction, backlog throttle wrappers are applied to upstream
    pools (generate_name, generate_docs, refine_name, refine_docs) so they
    pause when their downstream review queues exceed the configured cap.
    """
    from collections.abc import Awaitable

    from imas_codex.standard_names.defaults import (
        REVIEW_DOCS_BACKLOG_CAP,
        REVIEW_NAME_BACKLOG_CAP,
    )
    from imas_codex.standard_names.graph_ops import (
        claim_generate_docs_batch,
        claim_generate_name_batch,
        claim_refine_docs_batch,
        claim_refine_name_batch,
        claim_review_docs_batch,
        claim_review_name_batch,
        release_generate_docs_claims,
        release_generate_name_claims,
        release_refine_docs_claims,
        release_refine_name_claims,
        release_review_docs_claims,
        release_review_names_claims,
    )
    from imas_codex.standard_names.pools import PoolSpec
    from imas_codex.standard_names.workers import (
        process_generate_docs_batch,
        process_generate_name_batch,
        process_refine_docs_batch,
        process_refine_name_batch,
        process_review_docs_batch,
        process_review_name_batch,
    )

    regen_score = min_score if min_score is not None else DEFAULT_MIN_SCORE
    _rotation_cap_kwargs: dict[str, Any] = {}
    if rotation_cap is not None:
        _rotation_cap_kwargs["rotation_cap"] = rotation_cap
    _review_name_cap = (
        review_name_backlog_cap
        if review_name_backlog_cap is not None
        else REVIEW_NAME_BACKLOG_CAP
    )
    _review_docs_cap = (
        review_docs_backlog_cap
        if review_docs_backlog_cap is not None
        else REVIEW_DOCS_BACKLOG_CAP
    )

    # ── Adapter factories ─────────────────────────────────────────────

    def _make_claim_adapter(
        claim_fn: Callable[..., list[dict[str, Any]]],
        **kwargs: Any,
    ) -> Callable[[], Awaitable[dict[str, Any] | None]]:
        """Wrap a sync claim function as an async ``ClaimFn``."""

        async def _adapter() -> dict[str, Any] | None:
            items = await asyncio.to_thread(claim_fn, **kwargs)
            if not items:
                return None
            # Alias source_id → path for DD items so compose/grouping helpers
            # that key on `item["path"]` (a legacy convention from the
            # extract-time batch shape) work uniformly with claim-shaped items.
            for it in items:
                if it.get("source_type") == "dd" and "path" not in it:
                    sid = it.get("source_id")
                    if sid:
                        it["path"] = sid
            return {"items": items}

        return _adapter

    def _make_process_adapter(
        process_fn: Callable[
            [list[dict[str, Any]], Any, asyncio.Event],
            Awaitable[int],
        ],
    ) -> Callable[[dict[str, Any]], Awaitable[int]]:
        """Wrap a batch processor as a ``ProcessFn``."""

        async def _adapter(batch: dict[str, Any]) -> int:
            return await process_fn(batch["items"], mgr, stop_event, on_event=on_event)

        return _adapter

    def _make_release_adapter(
        release_fn: Callable[..., int],
        ids_kwarg: str = "sn_ids",
    ) -> Callable[[dict[str, Any]], Awaitable[None]]:
        """Wrap a token-aware release function as an async ``ReleaseFn``.

        Extracts ``id`` and ``claim_token`` from batch items and forwards
        them as keyword arguments to *release_fn*.  All items in a batch
        share the same ``claim_token`` (set atomically at claim time).

        Parameters
        ----------
        release_fn:
            Sync release function accepting keyword arguments
            *<ids_kwarg>* and *claim_token*.
        ids_kwarg:
            Name of the ids keyword argument (``"sn_ids"`` for
            :class:`StandardName` pools; ``"source_ids"`` for
            :class:`StandardNameSource` pools).
        """

        async def _adapter(batch: dict[str, Any]) -> None:
            items = batch.get("items", [])
            if not items:
                return
            ids = [item["id"] for item in items]
            token: str = items[0].get("claim_token") or ""
            await asyncio.to_thread(
                release_fn,
                **{ids_kwarg: ids, "claim_token": token},
            )

        return _adapter

    # ── PoolSpec construction ─────────────────────────────────────────

    # Optional scope_run_id kwargs for --focus mode.
    _scope_kwargs: dict[str, Any] = {}
    if scope_run_id:
        _scope_kwargs["scope_run_id"] = scope_run_id

    # Per-pool replica counts are config-driven via the
    # ``[tool.imas-codex.sn-pools]`` section (see
    # ``imas_codex.settings.get_pool_replicas``). Legacy installs without
    # that section still derive sensible defaults from
    # ``[sn-compose].max-concurrency`` inside the getter, so no caller
    # needs to fall back manually here.
    from imas_codex.settings import get_pool_replicas

    _gen_name_replicas = get_pool_replicas("generate_name")
    _review_name_replicas = get_pool_replicas("review_name")
    _refine_name_replicas = get_pool_replicas("refine_name")
    _gen_docs_replicas = get_pool_replicas("generate_docs")
    _review_docs_replicas = get_pool_replicas("review_docs")
    _refine_docs_replicas = get_pool_replicas("refine_docs")

    specs = [
        PoolSpec(
            name="generate_name",
            claim=_make_claim_adapter(
                claim_generate_name_batch,
                **({"domain": only_domain} if only_domain else {}),
                **_scope_kwargs,
            ),
            process=_make_process_adapter(process_generate_name_batch),
            release=_make_release_adapter(
                release_generate_name_claims, ids_kwarg="source_ids"
            ),
            replicas=_gen_name_replicas,
        ),
        PoolSpec(
            name="review_name",
            claim=_make_claim_adapter(
                claim_review_name_batch,
                **({"domain": only_domain} if only_domain else {}),
                **_scope_kwargs,
            ),
            process=_make_process_adapter(process_review_name_batch),
            release=_make_release_adapter(
                release_review_names_claims, ids_kwarg="sn_ids"
            ),
            replicas=_review_name_replicas,
        ),
        PoolSpec(
            name="refine_name",
            claim=_make_claim_adapter(
                claim_refine_name_batch,
                min_score=regen_score,
                **_rotation_cap_kwargs,
                **({"domain": only_domain} if only_domain else {}),
                **_scope_kwargs,
            ),
            process=_make_process_adapter(process_refine_name_batch),
            release=_make_release_adapter(
                release_refine_name_claims, ids_kwarg="sn_ids"
            ),
            replicas=_refine_name_replicas,
        ),
        PoolSpec(
            name="generate_docs",
            claim=_make_claim_adapter(
                claim_generate_docs_batch,
                **({"domain": only_domain} if only_domain else {}),
                **_scope_kwargs,
            ),
            process=_make_process_adapter(process_generate_docs_batch),
            release=_make_release_adapter(
                release_generate_docs_claims, ids_kwarg="sn_ids"
            ),
            replicas=_gen_docs_replicas,
        ),
        PoolSpec(
            name="review_docs",
            claim=_make_claim_adapter(
                claim_review_docs_batch,
                **({"domain": only_domain} if only_domain else {}),
                **_scope_kwargs,
            ),
            process=_make_process_adapter(process_review_docs_batch),
            release=_make_release_adapter(
                release_review_docs_claims, ids_kwarg="sn_ids"
            ),
            replicas=_review_docs_replicas,
        ),
        PoolSpec(
            name="refine_docs",
            claim=_make_claim_adapter(
                claim_refine_docs_batch,
                min_score=regen_score,
                **_rotation_cap_kwargs,
                **({"domain": only_domain} if only_domain else {}),
                **_scope_kwargs,
            ),
            process=_make_process_adapter(process_refine_docs_batch),
            release=_make_release_adapter(
                release_refine_docs_claims, ids_kwarg="sn_ids"
            ),
            replicas=_refine_docs_replicas,
        ),
    ]

    # ── Names-only filtering ─────────────────────────────────────────
    _DOCS_POOLS = {"generate_docs", "review_docs", "refine_docs"}
    if names_only:
        specs = [s for s in specs if s.name not in _DOCS_POOLS]

    # ── Flush filtering ──────────────────────────────────────────────
    # Flush mode drains existing work without generating new names.
    # Excludes generate_name so only review/refine pools run.
    if flush:
        specs = [s for s in specs if s.name != "generate_name"]

    # ── Skip-review filtering ────────────────────────────────────────
    # ``--only compose`` (and any generate-only selection) sets skip_review:
    # run the generate pools but no scoring/refinement. Drops the review AND
    # refine pools — refine has no work without review, and review is the only
    # paid (OpenRouter) stage, so this is the free, local-only zero-shot mode.
    if skip_review:
        _REVIEW_REFINE_POOLS = {
            "review_name",
            "review_docs",
            "refine_name",
            "refine_docs",
        }
        specs = [s for s in specs if s.name not in _REVIEW_REFINE_POOLS]

    # ── Backlog throttle wiring ───────────────────────────────────────
    # Upstream generators/refiners pause when their downstream review
    # queue exceeds the configured cap.  The throttle wraps the claim
    # adapter to return None (skip) when the downstream pool's
    # PoolHealth.pending_count is over cap, causing the pool to enter
    # its normal exponential backoff.  No blocking, no special yield.
    #
    # In focus mode (scope_run_id set), skip throttle entirely — the
    # focused set is 1-5 items and should never be blocked by global
    # review backlog.
    if not scope_run_id:
        specs_by_name = {s.name: s for s in specs}

        throttle_rules: list[tuple[str, str, int]] = [
            ("generate_name", "review_name", _review_name_cap),
            ("refine_name", "review_name", _review_name_cap),
            ("generate_docs", "review_docs", _review_docs_cap),
            ("refine_docs", "review_docs", _review_docs_cap),
        ]

        for upstream, downstream, cap in throttle_rules:
            if upstream not in specs_by_name or downstream not in specs_by_name:
                continue
            spec = specs_by_name[upstream]
            downstream_health = specs_by_name[downstream].health
            original_claim = spec.claim

            async def _throttled_claim(
                _orig: Callable[[], Awaitable[dict[str, Any] | None]] = original_claim,
                _health: Any = downstream_health,
                _cap: int = cap,
                _up: str = upstream,
                _down: str = downstream,
            ) -> dict[str, Any] | None:
                if _health.pending_count > _cap:
                    logger.debug(
                        "throttle: %s paused — %s backlog %d > cap %d",
                        _up,
                        _down,
                        _health.pending_count,
                        _cap,
                    )
                    return None
                return await _orig()

            spec.claim = _throttled_claim

    return specs


def _list_physics_domains_with_extractable_paths(source: str) -> list[str]:
    """Return distinct physics domains that have extractable DD paths.

    Queries the graph for distinct ``physics_domain`` values on
    ``IMASNode`` leaves that satisfy the same base filters used by
    :func:`~imas_codex.standard_names.sources.dd.extract_dd_candidates`:
    non-empty description, non-structure data type, not from
    ``core_instant_changes``.

    Only meaningful for ``source='dd'``; returns ``[]`` for other sources.
    """
    if source != "dd":
        return []

    from imas_codex.graph.client import GraphClient

    query = """
    MATCH (n:IMASNode)-[:IN_IDS]->(ids:IDS)
    WHERE n.description IS NOT NULL
      AND n.description <> ''
      AND NOT (n.data_type IN ['STRUCTURE', 'STRUCT_ARRAY'])
      AND ids.id <> 'core_instant_changes'
      AND n.physics_domain IS NOT NULL
      AND n.physics_domain <> ''
    RETURN DISTINCT n.physics_domain AS domain
    ORDER BY domain
    """
    with GraphClient() as gc:
        rows = list(gc.query(query))
    return [r["domain"] for r in rows if r.get("domain")]


async def _seed_all_domains(source: str, max_sources: int | None = None) -> int:
    """Seed sources from every physics domain except 'mixed'.

    'mixed' is skipped because mixed-unit DD paths cannot map to a single
    standard name (they violate the unit invariant per StandardName).
    """
    domains = await asyncio.to_thread(
        _list_physics_domains_with_extractable_paths, source
    )
    import random

    random.shuffle(domains)
    logger.info("Seeding %d domains (shuffled): %s", len(domains), domains)
    total = 0
    for d in domains:
        if d == "mixed":
            logger.info(
                "Skipping 'mixed' domain (mixed-unit sources are not standardisable)"
            )
            continue
        total += await _seed_domain_sources(domain=d, source=source)
        if max_sources and total >= max_sources:
            logger.warning("max_sources=%d reached; stopping seed sweep", max_sources)
            break
    if total > 1000:
        logger.warning(
            "Seeded %d sources — large queue; consider --max-sources to bound.",
            total,
        )
    return total


async def _seed_domain_sources(
    domain: str,
    source: str = "dd",
    stop_event: asyncio.Event | None = None,
    max_sources: int | None = None,
) -> int:
    """Seed the generate_name pool with StandardNameSource nodes for *domain*.

    Calls :func:`~imas_codex.standard_names.sources.dd.extract_dd_candidates`
    to discover DD paths for *domain* that have no existing StandardNameSource,
    then writes them via
    :func:`~imas_codex.standard_names.graph_ops.merge_standard_name_sources`.

    Returns the number of sources written (0 if none found or source != "dd").
    """
    if source != "dd":
        return 0

    from imas_codex.standard_names.graph_ops import (
        get_existing_standard_names,
        merge_standard_name_sources,
    )
    from imas_codex.standard_names.sources.dd import extract_dd_candidates

    existing = await asyncio.to_thread(get_existing_standard_names)
    batches = await asyncio.to_thread(
        extract_dd_candidates,
        domain_filter=domain,
        existing_names=existing,
        force=False,
        name_only=False,
    )

    sources = []
    for batch in batches:
        for item in batch.items:
            path = item.get("path")
            if not path:
                continue
            sources.append(
                {
                    "id": f"dd:{path}",
                    "source_type": "dd",
                    "source_id": path,
                    "dd_path": path,
                    "batch_key": batch.group_key,
                    "status": "extracted",
                    "description": item.get("description")
                    or item.get("documentation")
                    or "",
                    "physics_domain": item.get("physics_domain"),
                }
            )

    if not sources:
        return 0

    # Honour --max-sources for a single --domain seed too (previously only the
    # all-domains sweep capped, so --domain X --max-sources N seeded the whole
    # domain). Cap deterministically by path for a stable subset.
    if max_sources is not None and len(sources) > max_sources:
        sources.sort(key=lambda s: s["source_id"])
        logger.warning(
            "max_sources=%d reached for domain %r; seeding %d of %d candidates",
            max_sources,
            domain,
            max_sources,
            len(sources),
        )
        sources = sources[:max_sources]

    written = await asyncio.to_thread(merge_standard_name_sources, sources, force=False)
    return written


async def run_sn_pools(
    cost_limit: float,
    *,
    turn_number: int = 1,
    time_limit_s: float | None = None,
    min_score: float | None = None,
    rotation_cap: int | None = None,
    escalation_model: str | None = None,
    review_name_backlog_cap: int | None = None,
    review_docs_backlog_cap: int | None = None,
    source: str = "dd",
    only_domain: str | None = None,
    domains: tuple[str, ...] = (),
    max_sources: int | None = None,
    stop_event: asyncio.Event | None = None,
    loop_state: Any | None = None,
    pending_fn: Callable[[], dict[str, int]] | None = None,
    on_event: Callable[[dict[str, Any]], None] | None = None,
    display: Any | None = None,
    scope_run_id: str | None = None,
    names_only: bool = False,
    flush: bool = False,
    skip_review: bool = False,
) -> RunSummary:
    """Run the pool-based ``sn run`` orchestrator.

    Uses six concurrent worker pools that pull work from the graph
    independently and share a single :class:`BudgetManager`.

    When *names_only* is ``True``, the three docs pools
    (generate_docs, review_docs, refine_docs) are excluded so
    only name generation / review / refinement run.

    When *flush* is ``True``, the generate_name pool is excluded
    and auto-seeding is skipped.  Only review / refine / docs
    pools run, draining existing work without composing new names.

    Startup sequence:

    1. Create ``SNRun`` node and ``BudgetManager``.
    2. **Reconcile-once (B2)** — ``reconcile_standard_name_sources()``
       runs in a worker thread, completing before any pool issues its
       first claim.  This clears stale claims and revives sources
       whose upstream entities reappeared.
    3. Build 6 :class:`PoolSpec` objects (generate_name, review_name,
       refine_name, generate_docs, review_docs, refine_docs) with
       adapter closures and backlog throttle wiring.
    4. Delegate to :func:`~imas_codex.standard_names.pools.run_pools`
       which runs all pools concurrently with cooperative shutdown.
    5. Finalize ``SNRun`` with the actual stop reason and graph-derived
       cost.

    Args:
        cost_limit: Maximum LLM spend in USD.
        time_limit_s: Maximum wall-clock time in seconds.  When set,
            a background timer fires ``stop_event`` after this duration
            for a graceful shutdown.  ``None`` (default) means no time
            limit — only ``cost_limit`` and manual Ctrl-C stop the run.
        min_score: Review threshold.  Names with
            ``reviewer_score_name < min_score`` are routed to the
            refine_name pool; those above are eligible for review.
            Defaults to ``DEFAULT_MIN_SCORE`` when *None*.
        rotation_cap: Maximum REFINED_FROM chain depth before exhaustion.
            Defaults to ``DEFAULT_REFINE_ROTATIONS`` when *None*.
        escalation_model: Higher-capability model for final refine attempt.
            Defaults to ``DEFAULT_ESCALATION_MODEL`` when *None*.
        review_name_backlog_cap: Max pending review_name items before
            generate_name / refine_name pause.  Defaults to
            ``REVIEW_NAME_BACKLOG_CAP`` when *None*.
        review_docs_backlog_cap: Max pending review_docs items before
            generate_docs / refine_docs pause.  Defaults to
            ``REVIEW_DOCS_BACKLOG_CAP`` when *None*.
        source: ``"dd"`` or ``"signals"`` — scopes reconciliation.
        only_domain: Deprecated — use *domains* instead.
        domains: Tuple of physics domain names to seed.  When empty
            (default), all eligible domains are auto-seeded.
        max_sources: Cap on total StandardNameSource nodes seeded in
            the auto-seed sweep.  Prevents runaway queue growth.
        stop_event: Cooperative shutdown signal (set by the CLI harness).
        loop_state: Optional :class:`SNLoopState` for Rich progress.
        pending_fn: Optional callable ``() → dict[str, int]`` mapping
            pool names to pending counts.  When provided, a background
            watchdog polls this every 5 seconds to keep
            ``PoolHealth.pending_count`` current in headless / ``--quiet``
            mode where the Rich display ticker is absent.
        display: Optional :class:`~imas_codex.standard_names.display.SN6PoolDisplay`.
            When provided, the run's authoritative ``BudgetManager`` spend
            ledger (per-pool ``phase_spent`` + the graph-reconciled total)
            is wired into the display before the final render, so the COST
            gauge and ``print_summary`` report what was actually billed
            rather than the systematically undercounted sum of emitted
            ``on_event`` payloads (fanout / retry sub-charges emit no
            display event).
    """
    from imas_codex.standard_names.budget import BudgetManager
    from imas_codex.standard_names.pools import run_pools

    started = datetime.now(UTC)
    run_id = str(uuid.uuid4())
    summary = RunSummary(
        run_id=run_id,
        turn_number=turn_number,
        started_at=started,
        cost_limit=cost_limit,
        time_limit_s=time_limit_s,
        min_score=min_score,
    )

    if stop_event is None:
        stop_event = asyncio.Event()
    # Set when the idle-exhaustion watchdog detects sustained
    # zero-pending across all pools.  Lets the stop-reason logic
    # distinguish "out of eligible work" from "interrupted by user".
    idle_exhausted_event = asyncio.Event()
    # Set when the budget-saturation watchdog detects all pools have
    # consecutively failed to reserve budget SATURATION_THRESHOLD times.
    budget_saturated_event = asyncio.Event()
    # Set when the wall-clock deadline (--time-limit) fires.
    time_limit_event = asyncio.Event()
    # Set when any pool worker hits ``ProviderBudgetExhausted`` from the
    # upstream LLM provider (e.g. OpenRouter credit limit). Treated as a
    # peer stop signal — retrying against an exhausted account just
    # spins. The pool loop catches the exception, sets this event, and
    # propagates stop_event so all pools drain.
    provider_exhausted_event = asyncio.Event()

    # Shared BudgetManager — all six pools draw from the same pot.
    # Treat cost_limit <= 0 as unlimited (local GPU = zero cost).
    effective_budget = cost_limit if cost_limit > 0 else 1e9
    shared_mgr = BudgetManager(effective_budget, run_id=run_id)
    await shared_mgr.start()

    # Pre-create the SNRun node so LLMCost → FOR_RUN edges have a target.
    from imas_codex.standard_names.graph_ops import create_sn_run_open

    create_sn_run_open(
        run_id,
        started_at=started,
        cost_limit=cost_limit,
        min_score=min_score,
    )

    # Post-create assertion: verify the SNRun node exists in the graph.
    # Fail fast if the node wasn't persisted — without it, all LLMCost
    # FOR_RUN edges will be orphaned and telemetry is silently lost.
    from imas_codex.graph.client import GraphClient as _GC

    with _GC() as _gc:
        _sn_count = _gc.query(
            "MATCH (rr:SNRun {id: $rid}) RETURN count(rr) AS cnt",
            rid=run_id,
        )
        if not _sn_count or _sn_count[0]["cnt"] == 0:
            raise RuntimeError(
                f"SNRun {run_id} not found in graph after create_sn_run_open — "
                "aborting to prevent telemetry blackhole"
            )

    cost_is_exact = True

    # ── A3: LLM routing observability ─────────────────────────────
    import os

    from imas_codex.discovery.base.llm import _supports_cache_control
    from imas_codex.settings import get_model

    _a3_model = get_model("sn-compose")
    _a3_cache = _supports_cache_control(_a3_model)
    _a3_or_key = os.environ.get("OPENROUTER_API_KEY_STANDARD_NAMES") or ""
    _a3_or_key_src = "OPENROUTER_API_KEY_STANDARD_NAMES"
    if not _a3_or_key:
        _a3_or_key = os.environ.get("OPENROUTER_API_KEY_IMAS_CODEX") or ""
        _a3_or_key_src = "OPENROUTER_API_KEY_IMAS_CODEX"
    _a3_route = "direct" if (_a3_cache and _a3_or_key) else "proxy"
    logger.info(
        "run_sn_pools: model=%s supports_cache=%s route=%s api_key_source=%s",
        _a3_model,
        _a3_cache,
        _a3_route,
        _a3_or_key_src if _a3_or_key else "NONE",
    )

    try:
        # ── B2: Reconcile-once-at-startup ─────────────────────────
        # Must complete BEFORE any pool issues its first claim.
        from imas_codex.standard_names.graph_ops import (
            reconcile_standard_name_sources,
        )

        logger.info("run_sn_pools: reconciling sources (source=%s)…", source)
        recon_result = await asyncio.to_thread(reconcile_standard_name_sources, source)
        recon_total = sum(recon_result.values()) if recon_result else 0
        summary.sources_reconciled = recon_total
        logger.info(
            "run_sn_pools: reconcile complete — %d actions (%s)",
            recon_total,
            recon_result,
        )

        # ── B2b: Reconcile VocabGap nodes against current ISN vocab ───
        from imas_codex.standard_names.graph_ops import reconcile_vocab_gaps

        vg_result = await asyncio.to_thread(reconcile_vocab_gaps)
        if vg_result.get("checked", 0) > 0:
            deleted = (
                vg_result.get("deleted_false_positive", 0)
                + vg_result.get("deleted_invalid_segment", 0)
                + vg_result.get("deleted_open_segment", 0)
            )
            logger.info(
                "run_sn_pools: VocabGap reconcile — %d checked, %d deleted, "
                "%d reclassified, %d remaining",
                vg_result.get("checked", 0),
                deleted,
                vg_result.get("reclassified", 0),
                vg_result.get("remaining", 0),
            )

        # ── B2c: Reconcile provenance metadata ────────────────────────
        # NULL produced_sn_id scalars pointing at deleted names and delete
        # orphaned derived-parent scaffolding. Idempotent, provenance-only.
        from imas_codex.standard_names.graph_ops import reconcile_provenance

        prov_result = await asyncio.to_thread(reconcile_provenance)
        if prov_result.get("scalars_cleared", 0) or prov_result.get(
            "orphan_sources_deleted", 0
        ):
            logger.info(
                "run_sn_pools: provenance reconcile — %d stale scalar(s) cleared, "
                "%d orphaned derived-parent source(s) deleted",
                prov_result.get("scalars_cleared", 0),
                prov_result.get("orphan_sources_deleted", 0),
            )

        # ── B3: Domain extract (auto-seed) ────────────────────────
        # Skip auto-seeding in focus mode — sources are pre-seeded by CLI.
        # Skip auto-seeding in flush mode — only drain existing work.
        if scope_run_id:
            logger.info(
                "run_sn_pools: focus mode (run_id=%s…) — skipping auto-seed",
                scope_run_id[:8],
            )
            _domains = domains
        elif flush:
            logger.info("run_sn_pools: flush mode — skipping auto-seed")
            _domains = domains
        else:
            # Merge only_domain into domains tuple.
            _domains = domains
            if only_domain and not _domains:
                _domains = (only_domain,)

            if _domains:
                seeded = 0
                for d in _domains:
                    # max_sources is a GLOBAL cap across the domain list — pass
                    # the remaining budget so two domains can't each seed the cap.
                    _remaining = (
                        None if max_sources is None else max(0, max_sources - seeded)
                    )
                    if _remaining == 0:
                        logger.warning(
                            "max_sources=%d reached; skipping remaining domains",
                            max_sources,
                        )
                        break
                    seeded += await _seed_domain_sources(
                        domain=d,
                        source=source,
                        stop_event=stop_event,
                        max_sources=_remaining,
                    )
                logger.info(
                    "Auto-seeded %d sources from %d domain(s)", seeded, len(_domains)
                )
            else:
                seeded = await _seed_all_domains(source=source, max_sources=max_sources)
                logger.info("Auto-seeded %d sources from all eligible domains", seeded)

        # ── B3b: Rederive structural edges, seed parents, repair legacy drift ─
        # Backfill any missing HAS_PARENT / HAS_ERROR edges first so
        # ``seed_parent_sources`` can see every legitimate placeholder.
        # This catches two failure modes:
        #   1. Children written before ``_write_standard_name_edges``
        #      existed (no edges ever derived).
        #   2. ISN grammar revisions that newly derive HAS_PARENT
        #      edges absent at the original write time
        #      (e.g. ``flux_surface_mean_*``, ``total_plasma_current``).
        # Both classes leave parents structurally inaccessible to the
        # pipeline until the edges are re-derived. Idempotent (MERGE)
        # and fast (~1s for ~200 SNs) so safe to run on every loop.
        from imas_codex.standard_names.graph_ops import (
            normalize_derived_parent_lifecycle,
            rederive_structural_edges,
            seed_parent_sources,
        )

        edge_result = await asyncio.to_thread(rederive_structural_edges)
        logger.debug(
            "rederive_structural_edges processed %d SN(s)",
            edge_result.get("processed", 0),
        )
        if edge_result.get("migrated"):
            logger.info(
                "Migrated %d HAS_PARENT edges off superseded parents",
                edge_result["migrated"],
            )

        parent_count = await asyncio.to_thread(seed_parent_sources)
        if parent_count:
            logger.info("Seeded %d parent component sources", parent_count)
        repaired_parent_count = await asyncio.to_thread(
            normalize_derived_parent_lifecycle
        )
        if repaired_parent_count:
            logger.info(
                "Normalized %d derived parent lifecycle nodes",
                repaired_parent_count,
            )

        # ── Build pool specs ──────────────────────────────────────
        _only_domain_for_pools = _domains[0] if len(_domains) == 1 else None
        specs = _build_pool_specs(
            shared_mgr,
            stop_event,
            min_score=min_score,
            rotation_cap=rotation_cap,
            escalation_model=escalation_model,
            review_name_backlog_cap=review_name_backlog_cap,
            review_docs_backlog_cap=review_docs_backlog_cap,
            on_event=on_event,
            only_domain=_only_domain_for_pools,
            scope_run_id=scope_run_id,
            names_only=names_only,
            flush=flush,
            skip_review=skip_review,
        )

        # ── Wire pool health into display state ───────────────────
        if loop_state is not None and hasattr(loop_state, "set_pool_health"):
            for spec in specs:
                loop_state.set_pool_health(spec.name, spec.health)

        # ── Run pools + orphan sweep ──────────────────────────────
        from imas_codex.standard_names.defaults import (
            DEFAULT_ORPHAN_SWEEP_INTERVAL_S,
            DEFAULT_ORPHAN_SWEEP_TIMEOUT_S,
        )
        from imas_codex.standard_names.orphan_sweep import run_orphan_sweep_loop

        sweep_task = asyncio.create_task(
            run_orphan_sweep_loop(
                interval_s=DEFAULT_ORPHAN_SWEEP_INTERVAL_S,
                timeout_s=DEFAULT_ORPHAN_SWEEP_TIMEOUT_S,
                stop_event=stop_event,
            ),
            name="orphan_sweep",
        )

        # ── Embedding worker (reuses discovery infrastructure) ─────
        # Runs the shared embed_description_worker targeting StandardName
        # nodes.  It handles health gating, exponential backoff, and
        # batch persistence — no custom embed pool needed.
        from imas_codex.discovery.base.embed_worker import (
            embed_description_worker,
        )

        class _EmbedState:
            """Minimal state adapter for embed_description_worker."""

            stop_requested = False

            def should_stop(self) -> bool:
                return stop_event.is_set()

        embed_state = _EmbedState()
        embed_task = asyncio.create_task(
            embed_description_worker(
                embed_state,
                labels=["StandardName"],
                facility=None,
                batch_size=100,
            ),
            name="embed_sn",
        )

        # Periodic ``SNRun.cost_spent`` sync so ``imas-codex sn status``
        # reflects real spend even when the run is interrupted or crashes
        # before ``finalize_sn_run`` runs.
        async def _cost_spent_sync_loop() -> None:
            from imas_codex.standard_names.graph_ops import (
                update_sn_run_progress,
            )

            while not stop_event.is_set():
                try:
                    await asyncio.wait_for(stop_event.wait(), timeout=15.0)
                except TimeoutError:
                    pass
                try:
                    spent = max(
                        summary.cost_spent,
                        await shared_mgr.get_total_spent(),
                    )
                    summary.cost_spent = spent
                    await asyncio.to_thread(
                        update_sn_run_progress,
                        run_id,
                        cost_spent=spent,
                        cost_total=spent,
                        events_total=shared_mgr.batch_count,
                    )
                except Exception:  # noqa: BLE001 — never poison the loop
                    pass

        cost_sync_task = asyncio.create_task(
            _cost_spent_sync_loop(), name="cost_spent_sync"
        )

        # ── Deadline timer (--time-limit) ─────────────────────────
        deadline_task: asyncio.Task[None] | None = None
        if time_limit_s is not None and time_limit_s > 0:

            async def _deadline_timer() -> None:
                await asyncio.sleep(time_limit_s)
                logger.info(
                    "run_sn_pools: time limit reached (%.0fs) — requesting shutdown",
                    time_limit_s,
                )
                time_limit_event.set()
                stop_event.set()

            deadline_task = asyncio.create_task(
                _deadline_timer(), name="deadline_timer"
            )

        try:
            health_map = await run_pools(
                specs,
                shared_mgr,
                stop_event,
                pending_fn=pending_fn,
                idle_exhausted_event=idle_exhausted_event,
                budget_saturated_event=budget_saturated_event,
                provider_exhausted_event=provider_exhausted_event,
            )
        finally:
            if not sweep_task.done():
                sweep_task.cancel()
            if not cost_sync_task.done():
                cost_sync_task.cancel()
            if not embed_task.done():
                embed_task.cancel()
            if deadline_task is not None and not deadline_task.done():
                deadline_task.cancel()
            _gather_tasks = [sweep_task, cost_sync_task, embed_task]
            if deadline_task is not None:
                _gather_tasks.append(deadline_task)
            await asyncio.gather(*_gather_tasks, return_exceptions=True)
        logger.info("run_sn_pools: all pools exited — %s", health_map)

        # ── A3: per-pool cost observability ────────────────────────
        phase_spent = shared_mgr.phase_spent
        for pool_name, h in (health_map or {}).items():
            completed = getattr(h, "total_processed", 0) if h else 0
            spent = phase_spent.get(pool_name, 0.0)
            mean_cost = spent / completed if completed > 0 else 0.0
            logger.info(
                "run_sn_pools: pool=%s completed=%d spent=$%.4f mean_cost=$%.6f",
                pool_name,
                completed,
                spent,
                mean_cost,
            )
            if completed > 0 and mean_cost == 0.0 and _a3_route == "direct":
                logger.warning(
                    "run_sn_pools: pool=%s has %d completed items but mean_cost=0 "
                    "with expected route='direct' — cost tracking may be broken",
                    pool_name,
                    completed,
                )

        # Aggregate per-pool processed counts into RunSummary.
        def _total(name: str) -> int:
            h = health_map.get(name)
            return getattr(h, "total_processed", 0) if h is not None else 0

        summary.names_composed = _total("generate_name")
        summary.names_enriched = _total("generate_docs")
        summary.names_reviewed = _total("review_name") + _total("review_docs")
        summary.names_regenerated = _total("refine_name") + _total("refine_docs")

        # ── Async counter discrepancy check ───────────────────────
        # The SNRun node was bumped per-persist via bump_sn_run_counter.
        # Compare with pool-derived authoritative counts and log drift.
        try:
            from imas_codex.graph.client import GraphClient

            with GraphClient() as gc:
                _async_rows = gc.query(
                    "MATCH (rr:SNRun {id: $run_id}) "
                    "RETURN rr.names_composed AS nc, rr.names_enriched AS ne, "
                    "rr.names_reviewed AS nr, rr.names_regenerated AS ng",
                    run_id=run_id,
                )
            if _async_rows:
                _ar = _async_rows[0]
                _async_counters = {
                    "names_composed": int(_ar.get("nc") or 0),
                    "names_enriched": int(_ar.get("ne") or 0),
                    "names_reviewed": int(_ar.get("nr") or 0),
                    "names_regenerated": int(_ar.get("ng") or 0),
                }
                _auth_counters = {
                    "names_composed": summary.names_composed,
                    "names_enriched": summary.names_enriched,
                    "names_reviewed": summary.names_reviewed,
                    "names_regenerated": summary.names_regenerated,
                }
                _drifts = {
                    k: (_async_counters[k], _auth_counters[k])
                    for k in _auth_counters
                    if _async_counters[k] != _auth_counters[k]
                }
                if _drifts:
                    logger.debug(
                        "run_sn_pools: async counter drift detected — "
                        "overwriting with authoritative pool counts: %s",
                        ", ".join(
                            f"{k}(async={a}, auth={b})" for k, (a, b) in _drifts.items()
                        ),
                    )
                else:
                    logger.info(
                        "run_sn_pools: async counters match authoritative counts"
                    )
        except Exception:  # noqa: BLE001
            logger.debug(
                "run_sn_pools: async counter discrepancy check failed",
                exc_info=True,
            )

        # ── Determine stop reason ─────────────────────────────────
        # Check exhaustion before stop_event: the budget watchdog sets
        # stop_event when exhausted, so checking stop_event first would
        # misclassify budget-exhausted runs as "interrupted".
        # Likewise for the idle-exhaustion watchdog — when it fires, the
        # run finished its scope and must be classified as completed via
        # ``no_eligible_work`` rather than mistaken for a user interrupt.
        if shared_mgr.hard_exhausted():
            summary.stop_reason = "budget_exhausted"
        elif provider_exhausted_event.is_set():
            # Upstream LLM provider credits / billing limit hit — peer
            # to local cost_limit. Run is degraded: some work may have
            # completed before the failure, but the remaining queue is
            # blocked until the account is topped up.
            summary.stop_reason = "provider_budget_exhausted"
        elif budget_saturated_event.is_set():
            summary.stop_reason = "budget_saturated"
        elif time_limit_event.is_set():
            summary.stop_reason = "time_limit_reached"
        elif idle_exhausted_event.is_set():
            summary.stop_reason = "no_eligible_work"
        elif stop_event.is_set():
            summary.stop_reason = "interrupted"
        else:
            summary.stop_reason = "completed"

    except KeyboardInterrupt:
        summary.stop_reason = "interrupted"
        logger.warning("run_sn_pools interrupted by user")
    except Exception as exc:
        summary.stop_reason = "failed"
        logger.error("run_sn_pools failed: %s", exc, exc_info=True)
    finally:
        summary.stopped_at = datetime.now(UTC)

        # ── Shutdown timeouts ─────────────────────────────────────
        # Each sync graph call is wrapped in to_thread + wait_for so
        # a wedged Neo4j connection cannot block shutdown indefinitely.
        DRAIN_TIMEOUT = 30.0
        FINALIZE_TIMEOUT = 10.0
        ORPHAN_TIMEOUT = 10.0

        # Release any orphaned claims left by batches in flight at shutdown.
        try:
            from imas_codex.standard_names.graph_ops import release_all_orphan_claims

            orphan_counts = await asyncio.wait_for(
                asyncio.to_thread(release_all_orphan_claims),
                timeout=ORPHAN_TIMEOUT,
            )
            if orphan_counts.get("sn", 0) or orphan_counts.get("sns", 0):
                logger.info(
                    "run_sn_pools: orphan sweep released %d SN + %d SNS",
                    orphan_counts.get("sn", 0),
                    orphan_counts.get("sns", 0),
                )
        except TimeoutError:
            logger.warning(
                "run_sn_pools: orphan sweep timed out after %ds (non-fatal)",
                ORPHAN_TIMEOUT,
            )
        except Exception as _orphan_exc:  # noqa: BLE001
            logger.warning(
                "run_sn_pools: orphan sweep failed (non-fatal): %s", _orphan_exc
            )

        # ── Post-drain structural fixups ──────────────────────────
        # Re-derive structural edges (catches any new HAS_PARENT /
        # HAS_ERROR derivations from names composed during this run),
        # seed any parent placeholders whose children are now composed,
        # then resolve stale documentation links.  All non-LLM graph
        # operations — safe to run at shutdown.
        FIXUP_TIMEOUT = 30.0
        try:
            from imas_codex.standard_names.graph_ops import (
                rederive_structural_edges,
                seed_parent_sources,
            )

            await asyncio.wait_for(
                asyncio.to_thread(rederive_structural_edges),
                timeout=FIXUP_TIMEOUT,
            )
            _post_parents = await asyncio.wait_for(
                asyncio.to_thread(seed_parent_sources),
                timeout=FIXUP_TIMEOUT,
            )
            if _post_parents:
                logger.info(
                    "run_sn_pools: post-drain seeded %d parent SNs", _post_parents
                )
        except TimeoutError:
            logger.warning(
                "run_sn_pools: post-drain seed_parent_sources timed out (non-fatal)"
            )
        except Exception as _seed_exc:  # noqa: BLE001
            logger.warning(
                "run_sn_pools: post-drain seed_parent_sources failed: %s", _seed_exc
            )

        try:
            from imas_codex.standard_names.graph_ops import resolve_doc_links

            _link_stats = await asyncio.wait_for(
                asyncio.to_thread(resolve_doc_links),
                timeout=FIXUP_TIMEOUT,
            )
            _total_fixed = _link_stats.get("resolved", 0) + _link_stats.get(
                "removed", 0
            )
            if _total_fixed:
                summary.links_resolved = _total_fixed
                logger.info(
                    "run_sn_pools: resolved %d doc links (%d rewritten, %d removed)",
                    _total_fixed,
                    _link_stats.get("resolved", 0),
                    _link_stats.get("removed", 0),
                )
        except TimeoutError:
            logger.warning("run_sn_pools: resolve_doc_links timed out (non-fatal)")
        except Exception as _link_exc:  # noqa: BLE001
            logger.warning("run_sn_pools: resolve_doc_links failed: %s", _link_exc)

        # Drain pending LLMCost graph writes.  Bounded by DRAIN_TIMEOUT
        # so a wedged writer cannot block finalize_sn_run.
        cost_is_exact = True
        try:
            cost_is_exact = await asyncio.wait_for(
                asyncio.shield(shared_mgr.drain_pending()),
                timeout=DRAIN_TIMEOUT,
            )
        except TimeoutError:
            logger.error(
                "run_sn_pools: drain_pending timed out after %ds — "
                "cancelling writer and proceeding to finalize",
                DRAIN_TIMEOUT,
            )
            if shared_mgr._writer_task is not None:
                shared_mgr._writer_task.cancel()
            cost_is_exact = False
        except (asyncio.CancelledError, Exception) as _drain_exc:  # noqa: BLE001
            logger.warning(
                "run_sn_pools: drain_pending interrupted (%s); "
                "marking cost_is_exact=False and continuing finalization",
                _drain_exc,
            )
            cost_is_exact = False
        if not cost_is_exact and summary.stop_reason not in (
            "interrupted",
            "failed",
        ):
            summary.stop_reason = "degraded"

        # Refresh final cost from graph (best-effort under cancellation).
        try:
            graph_spent = await asyncio.wait_for(
                asyncio.to_thread(
                    lambda: shared_mgr._get_total_spent_sync(force_refresh=True)
                ),
                timeout=FINALIZE_TIMEOUT,
            )
            summary.cost_spent = max(summary.cost_spent, graph_spent)
        except TimeoutError:
            logger.warning(
                "run_sn_pools: get_total_spent timed out after %ds; "
                "using last-known cost_spent=$%.4f",
                FINALIZE_TIMEOUT,
                summary.cost_spent,
            )
        except (asyncio.CancelledError, Exception) as _spend_exc:  # noqa: BLE001
            logger.warning(
                "run_sn_pools: get_total_spent interrupted (%s); "
                "using last-known cost_spent=$%.4f",
                _spend_exc,
                summary.cost_spent,
            )

        # Phase-level cost breakdowns.
        phase_spent = shared_mgr.phase_spent
        summary.compose_cost = phase_spent.get("generate_name", 0.0) + phase_spent.get(
            "refine_name", 0.0
        )
        summary.review_cost = phase_spent.get("review_name", 0.0) + phase_spent.get(
            "review_docs", 0.0
        )

        # Reconcile the Rich display's COST figures to the authoritative
        # budget ledger.  Per-pool ``on_event`` payloads undercount real
        # spend (fanout / grammar-retry / acall retry sub-charges bill the
        # ledger without emitting a display event), so the final summary
        # must source TOTAL COST from ``phase_spent`` + the reconciled run
        # total — not from summed event payloads.  ``summary.cost_spent``
        # is the most authoritative total here (max of in-memory ledger
        # and the force-refreshed graph spend).
        if display is not None and hasattr(display, "set_budget_ledger"):
            try:
                display.set_budget_ledger(
                    phase_spent=phase_spent,
                    total=max(summary.cost_spent, shared_mgr.spent),
                )
            except Exception:  # noqa: BLE001 — display wiring is non-fatal
                pass

        # Compute pipeline hash — best-effort.
        _pipeline_hash: str | None = None
        _pipeline_hash_detail: str | None = None
        try:
            from imas_codex.standard_names.pipeline_version import (
                compute_pipeline_hash,
            )

            ph = compute_pipeline_hash()
            _pipeline_hash = ph["_composite"]
            _pipeline_hash_detail = _json.dumps(
                {k: v for k, v in ph.items() if k != "_composite"}
            )
        except Exception:  # noqa: BLE001
            pass

        # Finalize the SNRun node.  This is the *critical* write that
        # converts the open ``status='running'`` row into a closed run
        # — must run on every exit path (clean, budget-exhausted,
        # idle-exhausted, SIGINT, or task cancellation).
        from imas_codex.standard_names.graph_ops import finalize_sn_run

        try:
            await asyncio.wait_for(
                asyncio.to_thread(
                    finalize_sn_run,
                    run_id,
                    status=_STOP_TO_STATUS.get(summary.stop_reason, "completed"),
                    cost_spent=summary.cost_spent,
                    cost_is_exact=cost_is_exact,
                    stopped_at=summary.stopped_at,
                    elapsed_s=(summary.stopped_at - summary.started_at).total_seconds(),
                    cost_limit=round(summary.cost_limit, 6),
                    compose_cost=round(summary.compose_cost, 6),
                    review_cost=round(summary.review_cost, 6),
                    min_score=summary.min_score,
                    names_composed=summary.names_composed,
                    names_enriched=summary.names_enriched,
                    names_reviewed=summary.names_reviewed,
                    names_regenerated=summary.names_regenerated,
                    stop_reason=summary.stop_reason,
                    pipeline_hash=_pipeline_hash,
                    pipeline_hash_detail=_pipeline_hash_detail,
                ),
                timeout=FINALIZE_TIMEOUT,
            )
        except TimeoutError:
            logger.critical(
                "run_sn_pools: finalize_sn_run timed out after %ds for "
                "run_id=%s — SNRun row stays open; operator must reconcile "
                "manually",
                FINALIZE_TIMEOUT,
                run_id,
            )
        except Exception as _final_exc:  # noqa: BLE001
            logger.error(
                "run_sn_pools: finalize_sn_run failed for run_id=%s: %s",
                run_id,
                _final_exc,
                exc_info=True,
            )

        # Surface write failures even in Rich mode where loggers are
        # suppressed.  This ensures operators always see the warning.
        if shared_mgr.write_failed:
            logger.error(
                "run_sn_pools: LLMCost write failure detected — "
                "cost_is_exact=False. Check logs for details."
            )

    return summary
