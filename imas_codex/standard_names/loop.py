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
    drain_report: list[dict[str, Any]] = field(default_factory=list)


# ── Status mapping ────────────────────────────────────────────────────
# Map RunSummary.stop_reason to SNRun.status lifecycle values.
_STOP_TO_STATUS: dict[str, str] = {
    "completed": "completed",
    "budget_exhausted": "degraded",
    "budget_saturated": "degraded",
    "provider_budget_exhausted": "degraded",
    "time_limit_reached": "degraded",
    "stalled": "degraded",
    "pending_count_failed": "degraded",
    "terminal_scope_unproven": "degraded",
    "transient_scope_residue": "degraded",
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
        "drain_report": summary.drain_report,
    }


# ═══════════════════════════════════════════════════════════════════════
# Pool-based orchestrator — concurrent weighted pools over one shared budget,
# in place of a sequential rotation over physics domains.
# ═══════════════════════════════════════════════════════════════════════

# Default regen threshold when min_score is not explicitly provided.
# Imported from defaults.py — do not re-define here.


def _count_scope_names(
    scope_run_id: str | None = None, drain_scope_id: str | None = None
) -> int:
    """Return the number of StandardName nodes bound to *scope_run_id*.

    This is the size of a ``--focus`` scoped drain — the natural ceiling on how
    many nodes any pool can concurrently work. Exact-name scopes pass their
    already-known cardinality instead, so this fallback query is issued only for
    scope modes that do not know their size. Returns 0 on any query error.
    """
    from imas_codex.graph.client import GraphClient

    try:
        with GraphClient() as gc:
            if drain_scope_id:
                rows = list(
                    gc.query(
                        "MATCH (sn:StandardName {drain_scope_id: $sid}) "
                        "RETURN count(sn) AS n",
                        sid=drain_scope_id,
                    )
                )
            else:
                rows = list(
                    gc.query(
                        "MATCH (sn:StandardName {run_id: $rid}) RETURN count(sn) AS n",
                        rid=scope_run_id,
                    )
                )
        return int(rows[0]["n"]) if rows else 0
    except Exception as exc:  # noqa: BLE001 — best-effort sizing, never fatal
        logger.warning("_count_scope_names(%s) failed: %s", scope_run_id, exc)
        return 0


def _build_pool_specs(
    mgr: Any,
    stop_event: asyncio.Event,
    *,
    compose_model: str | None = None,
    min_score: float | None = None,
    rotation_cap: int | None = None,
    escalation_model: str | None = None,
    review_name_backlog_cap: int | None = None,
    review_docs_backlog_cap: int | None = None,
    on_event: Callable[[dict[str, Any]], None] | None = None,
    only_domain: str | None = None,
    scope_run_id: str | None = None,
    scope_size_hint: int | None = None,
    drain_scope_id: str | None = None,
    edits_only: bool = False,
    names_only: bool = False,
    docs_only: bool = False,
    flush: bool = False,
    skip_review: bool = False,
    skip_generate: bool = False,
    only_pool: str | None = None,
) -> list[Any]:
    """Construct 7 :class:`PoolSpec` objects wiring claims → batch processors.

    Each pool gets two adapter closures:

    * **claim adapter** — runs the synchronous ``claim_*_seed_and_expand``
      graph function in a worker thread and returns the result wrapped in
      a dict (``{"items": [...]}``), or ``None`` on empty.
    * **process adapter** — unpacks the claimed batch and delegates to the
      corresponding ``process_*_batch`` async function, forwarding the
      shared :class:`BudgetManager` and ``stop_event``.

    After construction, backlog throttle wrappers are applied to the upstream
    generate/refine pools (generate_name, generate_docs, refine_name,
    refine_docs) so they pause when their downstream review queues exceed the
    configured cap.  ``enrich_parents`` is intentionally NOT throttled.

    The ``enrich_parents`` pool drains the *existing* placeholder-derived-parent
    backlog, synthesising a children-grounded description and accepting the
    parent STRUCTURALLY (``name_stage='accepted'`` with an inherited score —
    it skips REVIEW_NAME, see ``persist_enriched_parent``).  It is a name-axis
    producer: it runs under ``names_only`` and ``flush`` (draining existing
    work, not seeding new), is dropped under ``docs_only``, and survives
    ``skip_review``.  These behaviours fall out of the existing ``_DOCS_POOLS``
    / flush / skip_review filters below; no special-casing needed.
    """
    import contextlib
    from collections.abc import Awaitable

    from imas_codex.standard_names.defaults import (
        REVIEW_DOCS_BACKLOG_CAP,
        REVIEW_NAME_BACKLOG_CAP,
    )
    from imas_codex.standard_names.graph_ops import (
        _CLAIM_HEARTBEAT_SECONDS,
        claim_enrich_parents_batch,
        claim_generate_docs_batch,
        claim_generate_name_batch,
        claim_refine_docs_batch,
        claim_refine_name_batch,
        claim_review_docs_batch,
        claim_review_name_batch,
        refresh_name_claims,
        release_enrich_parents_claims,
        release_generate_docs_claims,
        release_generate_name_claims,
        release_refine_docs_claims,
        release_refine_name_claims,
        release_review_docs_claims,
        release_review_names_claims,
    )
    from imas_codex.standard_names.pools import POOL_NAMES, PoolSpec
    from imas_codex.standard_names.workers import (
        process_enrich_parents_batch,
        process_generate_docs_batch,
        process_generate_name_batch,
        process_refine_docs_batch,
        process_refine_name_batch,
        process_review_docs_batch,
        process_review_name_batch,
    )

    if only_pool is not None and only_pool not in POOL_NAMES:
        raise ValueError(f"unknown standard-name pool: {only_pool}")
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

    async def _heartbeat_loop(
        sn_ids: list[str],
        token: str,
        stop: asyncio.Event,
        interval: float = _CLAIM_HEARTBEAT_SECONDS,
    ) -> None:
        """Refresh a held StandardName claim lease until *stop* is set.

        A quorum-consensus review (or an enrich batch) can outrun the claim
        TTL; without a heartbeat the lease expires mid-flight and a peer
        worker re-claims the same names, duplicating the paid LLM spend. This
        bumps ``claimed_at`` on the batch's names (compare-and-set on the
        token — a lease already lost to a peer is not stolen back) every
        ``interval`` seconds, comfortably inside the TTL.
        """
        while not stop.is_set():
            try:
                await asyncio.wait_for(stop.wait(), timeout=interval)
                return  # stopped between beats
            except TimeoutError:
                try:
                    await asyncio.to_thread(refresh_name_claims, sn_ids, token)
                except Exception as exc:  # noqa: BLE001 — heartbeat is best-effort
                    logger.debug("claim heartbeat refresh failed: %s", exc)

    def _make_process_adapter(
        process_fn: Callable[
            [list[dict[str, Any]], Any, asyncio.Event],
            Awaitable[int],
        ],
        *,
        heartbeat: bool = False,
        process_kwargs: dict[str, Any] | None = None,
    ) -> Callable[[dict[str, Any]], Awaitable[int]]:
        """Wrap a batch processor as a ``ProcessFn``.

        When ``heartbeat`` is set, a background task refreshes the batch's
        StandardName claim lease while the processor runs, so a long batch
        cannot have its lease expire and be re-claimed by a peer worker. The
        heartbeat is cancelled as soon as the processor returns or raises.
        """

        async def _adapter(batch: dict[str, Any]) -> int:
            kwargs = process_kwargs or {}
            if not heartbeat:
                return await process_fn(
                    batch["items"], mgr, stop_event, on_event=on_event, **kwargs
                )
            items = batch.get("items", [])
            sn_ids = [it["id"] for it in items if it.get("id")]
            token = items[0].get("claim_token") if items else None
            if not (sn_ids and token):
                return await process_fn(
                    items, mgr, stop_event, on_event=on_event, **kwargs
                )
            beat_stop = asyncio.Event()
            beat = asyncio.create_task(_heartbeat_loop(sn_ids, token, beat_stop))
            try:
                return await process_fn(
                    items, mgr, stop_event, on_event=on_event, **kwargs
                )
            finally:
                beat_stop.set()
                beat.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await beat

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

    # Optional scope kwargs threaded identically into every pool's claim
    # adapter: scope_run_id for --focus mode, edits_only for --edits mode
    # (scope the run to pending sn-edit successors, edit_status='open'). Both
    # may combine — the claim layer ANDs their predicates.
    _scope_kwargs: dict[str, Any] = {}
    if scope_run_id:
        _scope_kwargs["scope_run_id"] = scope_run_id
    if drain_scope_id:
        _scope_kwargs["drain_scope_id"] = drain_scope_id
    if edits_only:
        _scope_kwargs["edits_only"] = True

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
    _enrich_parents_replicas = get_pool_replicas("enrich_parents")

    # A scoped run has a finite actionable set. Never launch more claim loops
    # than names: excess replicas only amplify claim/readback traffic against
    # the same nodes. Exact-name callers already know cardinality and provide a
    # hint, avoiding a graph recount; other scoped modes retain one count query.
    if scope_size_hint is not None:
        if (
            not isinstance(scope_size_hint, int)
            or isinstance(scope_size_hint, bool)
            or scope_size_hint <= 0
        ):
            raise ValueError("scope_size_hint must be a positive integer")
        if not (scope_run_id or drain_scope_id):
            raise ValueError("scope_size_hint requires a bounded graph scope")
        _scope_size = scope_size_hint
    elif scope_run_id or drain_scope_id:
        _scope_size = _count_scope_names(scope_run_id, drain_scope_id)
    else:
        _scope_size = 0

    if _scope_size > 0:
        import math

        _gen_name_replicas = min(_gen_name_replicas, _scope_size)
        _review_name_replicas = min(_review_name_replicas, _scope_size)
        _refine_name_replicas = min(_refine_name_replicas, _scope_size)
        _enrich_parents_replicas = min(_enrich_parents_replicas, _scope_size)
        _docs_cap = max(1, math.ceil(_scope_size / 2))
        _gen_docs_replicas = min(_gen_docs_replicas, _docs_cap)
        _review_docs_replicas = min(_review_docs_replicas, _docs_cap)
        _refine_docs_replicas = min(_refine_docs_replicas, _docs_cap)
        logger.info(
            "scoped run (%s): %d names — every pool capped at cardinality; "
            "docs capped at %d",
            scope_run_id or drain_scope_id,
            _scope_size,
            _docs_cap,
        )

    specs = [
        PoolSpec(
            name="generate_name",
            claim=_make_claim_adapter(
                claim_generate_name_batch,
                **({"domain": only_domain} if only_domain else {}),
                **_scope_kwargs,
            ),
            process=_make_process_adapter(
                process_generate_name_batch,
                process_kwargs={"compose_model": compose_model},
            ),
            release=_make_release_adapter(
                release_generate_name_claims, ids_kwarg="source_ids"
            ),
            replicas=_gen_name_replicas,
        ),
        PoolSpec(
            name="review_name",
            claim=_make_claim_adapter(
                claim_review_name_batch,
                min_score=regen_score,
                **({"domain": only_domain} if only_domain else {}),
                **_scope_kwargs,
            ),
            process=_make_process_adapter(process_review_name_batch, heartbeat=True),
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
            process=_make_process_adapter(
                process_refine_name_batch,
                process_kwargs={"scope_run_id": scope_run_id},
            ),
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
                min_score=regen_score,
                **({"domain": only_domain} if only_domain else {}),
                **_scope_kwargs,
            ),
            process=_make_process_adapter(
                process_review_docs_batch,
                heartbeat=True,
            ),
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
        PoolSpec(
            name="enrich_parents",
            claim=_make_claim_adapter(
                claim_enrich_parents_batch,
                **({"domain": only_domain} if only_domain else {}),
                **_scope_kwargs,
            ),
            process=_make_process_adapter(process_enrich_parents_batch, heartbeat=True),
            release=_make_release_adapter(
                release_enrich_parents_claims, ids_kwarg="sn_ids"
            ),
            replicas=_enrich_parents_replicas,
        ),
    ]

    # ── Names-only / docs-only filtering ─────────────────────────────
    _DOCS_POOLS = {"generate_docs", "review_docs", "refine_docs"}
    if names_only:
        specs = [s for s in specs if s.name not in _DOCS_POOLS]
    if docs_only:
        # Inverse of names_only: run ONLY the docs pools (generate_docs,
        # review_docs, refine_docs) on already name-accepted names — so a
        # budget-capped docs rotation spends purely on documentation, not on
        # name compose/review.
        specs = [s for s in specs if s.name in _DOCS_POOLS]

    # ── Flush / skip-generate filtering ──────────────────────────────
    # Flush mode drains exactly the pinned downstream set without generating
    # new names.
    # ``--only`` phases that exclude the generate phase (skip_generate, e.g.
    # ``--only link`` / ``--only review``) drop the same pool. Both exclude
    # generate_name so only review/refine/docs pools run — no new composition.
    if flush:
        from imas_codex.standard_names.pools import FLUSH_POOL_NAMES

        specs = [s for s in specs if s.name in FLUSH_POOL_NAMES]
    elif skip_generate:
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

    # Exact action selectors are applied after the established coarse filters,
    # so names-only/docs-only and review suppression retain their safety
    # semantics. A contradictory combination fails instead of silently running
    # zero pools or widening to an adjacent review/refine action.
    if only_pool is not None:
        specs = [spec for spec in specs if spec.name == only_pool]
        if len(specs) != 1:
            raise ValueError(
                f"exact pool {only_pool!r} is excluded by another pool filter"
            )

    # ── Backlog throttle wiring ───────────────────────────────────────
    # Upstream generators/refiners pause when their downstream review
    # queue exceeds the configured cap.  The throttle wraps the claim
    # adapter to return None (skip) when the downstream pool's
    # PoolHealth.pending_count is over cap, causing the pool to enter
    # its normal exponential backoff.  No blocking, no special yield.
    #
    # In focus mode (scope_run_id set) or edits mode (edits_only), skip
    # throttle entirely — the scoped set is a bounded batch that should never
    # be blocked by the global review backlog.
    if not (scope_run_id or drain_scope_id or edits_only):
        specs_by_name = {s.name: s for s in specs}

        # NB: enrich_parents is NOT throttled. It accepts derived parents
        # structurally (skips REVIEW_NAME — see persist_enriched_parent), so it
        # no longer feeds the review_name bottleneck; it is cheap and drains a
        # finite backlog, and the accepted parents it produces queue durably in
        # generate_docs-pending (paced by the generate_docs↔review_docs throttle).
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
      AND coalesce(n.lifecycle_status, '') <> 'removed'
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

    # The DD version being extracted at pins only GENUINELY-NEW sources, passed
    # as the batch default rather than stamped per source. A source's pin is
    # immutable: an existing one keeps the version it was snapshotted at and is
    # read against that version's metadata, so a pin behind the current DD is
    # ordinary state, not staleness. Supplying it per source instead asserts a
    # version for every source and makes a re-seed collide with its own stored
    # pin. Extraction reads one current DDVersion, so the batches agree.
    extract_dd_version = next(
        (b.dd_version for b in batches if getattr(b, "dd_version", None)), None
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
                    or None,
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

    written = await asyncio.to_thread(
        merge_standard_name_sources,
        sources,
        force=False,
        default_dd_version=extract_dd_version,
    )
    return written


async def run_sn_pools(
    cost_limit: float,
    *,
    turn_number: int = 1,
    time_limit_s: float | None = None,
    compose_model: str | None = None,
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
    scope_size_hint: int | None = None,
    drain_scope_id: str | None = None,
    drain_paths: tuple[str, ...] = (),
    drain_dd_version: str | None = None,
    edits_only: bool = False,
    names_only: bool = False,
    docs_only: bool = False,
    flush: bool = False,
    skip_review: bool = False,
    skip_generate: bool = False,
    only_pool: str | None = None,
    attach_only: bool = False,
    reconcile_only: bool = False,
    skip_global_maintenance: bool = False,
    scope_started_callback: Callable[[], None] | None = None,
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
    2. **Reconcile-once at startup** — ``reconcile_standard_name_sources()``
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
        compose_model: Optional configured model override for pooled name
            composition. When absent, the production ``sn-compose`` seat is
            resolved by the worker.
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
        reconcile_only: Run the complete graph-maintenance sequence, including
            structural parent lifecycle repair, then return before constructing
            any operational worker pool.
        skip_global_maintenance: Bypass graph-wide startup, background, and
            post-drain maintenance while retaining the ordinary scoped worker
            pools and run audit. Requires *scope_run_id* and is incompatible
            with maintenance-only modes.
        only_pool: Restrict operational work to exactly one canonical worker
            pool. Broad ``--only`` phases leave this unset.
        scope_size_hint: Known positive cardinality for a bounded graph scope.
            Exact-name callers provide it to avoid recounting the graph.
    """
    from imas_codex.standard_names.budget import BudgetManager
    from imas_codex.standard_names.pools import POOL_NAMES, run_pools

    if only_pool is not None and only_pool not in POOL_NAMES:
        raise ValueError(f"unknown standard-name pool: {only_pool}")
    if scope_size_hint is not None:
        if (
            not isinstance(scope_size_hint, int)
            or isinstance(scope_size_hint, bool)
            or scope_size_hint <= 0
        ):
            raise ValueError("scope_size_hint must be a positive integer")
        if not (scope_run_id or drain_scope_id):
            raise ValueError("scope_size_hint requires a bounded graph scope")

    started = datetime.now(UTC)

    if drain_scope_id and (scope_run_id or edits_only):
        raise ValueError("drain_scope_id cannot be combined with another scope")
    if drain_scope_id and (not drain_paths or not drain_dd_version):
        raise ValueError("drain_scope_id requires exact paths and a DD version")
    skip_global_maintenance = skip_global_maintenance or bool(drain_scope_id)
    if skip_global_maintenance and not (scope_run_id or drain_scope_id):
        raise ValueError(
            "skip_global_maintenance requires scope_run_id or drain_scope_id"
        )
    if skip_global_maintenance and (attach_only or reconcile_only):
        raise ValueError(
            "skip_global_maintenance is incompatible with maintenance-only modes"
        )

    run_id = str(uuid.uuid4())
    summary = RunSummary(
        run_id=run_id,
        turn_number=turn_number,
        started_at=started,
        cost_limit=cost_limit,
        time_limit_s=time_limit_s,
        min_score=min_score,
    )

    # ── --only attach: focused, no-LLM DD-edge backfill ───────────────
    # Materialize the DD-side HAS_STANDARD_NAME projection from provenance and
    # reconcile the source_paths scalar, then return. No pools, no SNRun audit
    # row, no LLM-touching maintenance (source-drift steering, stranded-review
    # promotion) — just the two self-healing projections. The full maintenance
    # suite is `--only reconcile`; this is the one-shot edge backfill.
    if attach_only:
        from imas_codex.standard_names.graph_ops import (
            reconcile_standard_name_dd_edges,
            reconcile_standard_name_source_paths,
        )

        edge_res = await asyncio.to_thread(reconcile_standard_name_dd_edges)
        sp_res = await asyncio.to_thread(reconcile_standard_name_source_paths)
        logger.info(
            "run_sn_pools: --only attach — %d HAS_STANDARD_NAME edge(s) "
            "materialized, %d dropped on unit disagreement, %d source_paths "
            "scalar(s) reconciled",
            edge_res.get("edges_created", 0),
            edge_res.get("pairs_dropped", 0),
            sp_res.get("names_reconciled", 0),
        )
        summary.stopped_at = datetime.now(UTC)
        summary.sources_reconciled = edge_res.get("edges_created", 0)
        summary.stop_reason = "completed"
        return summary

    if stop_event is None:
        stop_event = asyncio.Event()
    # Set when the idle-exhaustion watchdog detects sustained
    # zero-pending across all pools.  Lets the stop-reason logic
    # distinguish "out of eligible work" from "interrupted by user".
    idle_exhausted_event = asyncio.Event()
    # Set when pending work remains but no pool advances for the liveness
    # window. This is incomplete work, not a clean empty drain.
    stalled_event = asyncio.Event()
    # Set when the graph-backed pending observation cannot prove the backlog.
    pending_count_failed_event = asyncio.Event()
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
    drain_heartbeat_task: asyncio.Task[None] | None = None

    # Shared BudgetManager — all six pools draw from the same pot.
    # Treat cost_limit <= 0 as unlimited (local GPU = zero cost).
    effective_budget = cost_limit if cost_limit > 0 else 1e9
    shared_mgr = BudgetManager(effective_budget, run_id=run_id)
    await shared_mgr.start()

    # Pre-create the SNRun node so LLMCost → FOR_RUN edges have a target.
    from imas_codex.settings import get_pool_replicas
    from imas_codex.standard_names.graph_ops import create_sn_run_open
    from imas_codex.standard_names.run_invocation import capture_run_invocation

    # Record how the run was asked to work alongside what it does, so an empty
    # or budget-starved run can be told apart from one whose scope excluded the
    # work. Captured from the caller's resolved arguments rather than re-parsed
    # from the command line, so non-CLI entry points are recorded identically.
    # Replica counts come from environment overrides as often as from config,
    # so the command line alone cannot explain a run that starved on budget:
    # the money a pool must hold at once is its replica count times its
    # per-request reservation. Record the resolved counts alongside the limit
    # they are spent against.
    invocation = capture_run_invocation(
        flags={
            "pool_replicas": {pool: get_pool_replicas(pool) for pool in POOL_NAMES},
            "cost_limit": cost_limit,
            "time_limit_s": time_limit_s,
            "turn_number": turn_number,
            "min_score": min_score,
            "compose_model": compose_model,
            "escalation_model": escalation_model,
            "rotation_cap": rotation_cap,
            "review_name_backlog_cap": review_name_backlog_cap,
            "review_docs_backlog_cap": review_docs_backlog_cap,
            "max_sources": max_sources,
            "source": source,
            "flush": flush,
            "names_only": names_only,
            "docs_only": docs_only,
            "skip_review": skip_review,
            "skip_generate": skip_generate,
        },
        scope={
            "domains": list(domains),
            "only_domain": only_domain,
            "only_pool": only_pool,
            "scope_run_id": scope_run_id,
            "scope_size_hint": scope_size_hint,
            "drain_scope_id": drain_scope_id,
            "drain_dd_version": drain_dd_version,
            "drain_path_count": len(drain_paths) or None,
            "edits_only": edits_only,
            "attach_only": attach_only,
            "reconcile_only": reconcile_only,
            "skip_global_maintenance": skip_global_maintenance,
        },
    )

    create_sn_run_open(
        run_id,
        started_at=started,
        cost_limit=cost_limit,
        min_score=min_score,
        **invocation,
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

    # Reconcile SNRun rows orphaned by a hard-killed prior process (finalize
    # runs in a finally block, so an open 'started' row means the process died
    # before it could close the run). Best-effort — never let a sweep failure
    # abort a fresh run.
    async def _global_maintenance_call(
        fn: Callable[..., Any],
        *args: Any,
        default: Any,
        **kwargs: Any,
    ) -> Any:
        """Run one graph-wide maintenance function unless explicitly bypassed."""
        if skip_global_maintenance:
            return default
        return await asyncio.to_thread(fn, *args, **kwargs)

    if not skip_global_maintenance:
        try:
            from imas_codex.standard_names.graph_ops import (
                mark_orphaned_standard_name_runs_stale,
            )

            mark_orphaned_standard_name_runs_stale(current_run_id=run_id)
        except Exception as _stale_exc:  # noqa: BLE001 — non-fatal reconciliation
            logger.warning(
                "run_sn_pools: orphaned-SNRun sweep failed (non-fatal): %s",
                _stale_exc,
            )

    cost_is_exact = True

    # ── Compose-model routing observability ───────────────────────
    import os

    from imas_codex.discovery.base.llm import _supports_cache_control
    from imas_codex.settings import get_model

    _a3_model = compose_model or get_model("sn-compose")
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

    primary_error: BaseException | None = None
    primary_traceback: Any | None = None
    cleanup_error: BaseException | None = None
    if drain_scope_id and scope_started_callback is not None:
        scope_started_callback()

    try:
        if skip_global_maintenance:
            logger.info(
                "run_sn_pools: scoped mode — bypassing global maintenance (run_id=%s)",
                scope_run_id,
            )
        # ── Reconcile-once-at-startup ─────────────────────────────
        # Must complete BEFORE any pool issues its first claim.
        from imas_codex.standard_names.graph_ops import (
            reconcile_standard_name_sources,
        )

        logger.info("run_sn_pools: reconciling sources (source=%s)…", source)
        recon_result = await _global_maintenance_call(
            reconcile_standard_name_sources, source, default={}
        )
        recon_total = sum(recon_result.values()) if recon_result else 0
        summary.sources_reconciled = recon_total
        logger.info(
            "run_sn_pools: reconcile complete — %d actions (%s)",
            recon_total,
            recon_result,
        )

        # ── Reconcile VocabGap nodes against current ISN vocab ────────
        from imas_codex.standard_names.graph_ops import reconcile_vocab_gaps

        vg_result = await _global_maintenance_call(reconcile_vocab_gaps, default={})
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

        # Re-evaluate durably-cached blocking decisions whose cause has since
        # been fixed upstream. Both are the same defect shape — a verdict
        # snapshotted onto the source and never revisited — so both run right
        # after the VocabGap pass and BEFORE seeding, so a revived source is
        # claimable the same run.
        #
        # A unit skip records the unit resolver as it stood at extraction; a
        # resolver fix (dimensionless is a unit, count pseudo-units, canonical
        # symbol order) does not lift it. Re-ask the extractor's own question.
        from imas_codex.standard_names.graph_ops import revive_unit_skipped_sources

        unit_result = await _global_maintenance_call(
            revive_unit_skipped_sources, default={}
        )
        if unit_result.get("revived", 0):
            logger.info(
                "run_sn_pools: unit-skip revival — %d of %d unit-skipped "
                "source(s) returned to 'extracted' (unit now parses)",
                unit_result.get("revived", 0),
                unit_result.get("checked", 0),
            )

        # A vocab_gap source is un-parked by reconcile_vocab_gaps only when the
        # exact token it asked for appears — but the request is often the wrong
        # spelling of a capability the vocabulary now expresses another way. Give
        # each parked source one retry per vocabulary change instead.
        from imas_codex.standard_names.graph_ops import (
            retry_vocab_gap_sources_on_grammar_change,
        )

        gap_retry = await _global_maintenance_call(
            retry_vocab_gap_sources_on_grammar_change, default={}
        )
        if gap_retry.get("revived", 0):
            logger.info(
                "run_sn_pools: vocabulary-bump retry — %d of %d parked "
                "source(s) returned to 'extracted' under the current ISN "
                "vocabulary",
                gap_retry.get("revived", 0),
                gap_retry.get("checked", 0),
            )

        # ── Reconcile provenance metadata ─────────────────────────────
        # Reattach live scalar/missing-edge desyncs, NULL produced_sn_id
        # scalars pointing at deleted names, and delete orphaned derived-parent
        # scaffolding. Idempotent, provenance-only. Then surface the ledger
        # orphan count (live names with no PRODUCED_NAME source) — the invariant
        # the ledger must hold; a non-zero count is silent provenance loss.
        from imas_codex.standard_names.graph_ops import (
            reconcile_grammar_segments,
            reconcile_provenance,
            reconcile_source_status_liveness,
            retire_unreachable_hint_edits,
        )
        from imas_codex.standard_names.ledger import find_provenance_orphans

        prov_result = await _global_maintenance_call(reconcile_provenance, default={})
        if (
            prov_result.get("edges_reattached", 0)
            or prov_result.get("scalars_cleared", 0)
            or prov_result.get("orphan_sources_deleted", 0)
        ):
            logger.info(
                "run_sn_pools: provenance reconcile — %d edge(s) reattached, "
                "%d stale scalar(s) cleared, "
                "%d orphaned derived-parent source(s) deleted",
                prov_result.get("edges_reattached", 0),
                prov_result.get("scalars_cleared", 0),
                prov_result.get("orphan_sources_deleted", 0),
            )

        source_status = await _global_maintenance_call(
            reconcile_source_status_liveness, default={}
        )
        if source_status.get("live_realigned", 0) or source_status.get(
            "orphaned_reset", 0
        ):
            logger.info(
                "run_sn_pools: source-status reconcile — %d aligned to a live "
                "target, %d returned to extracted",
                source_status.get("live_realigned", 0),
                source_status.get("orphaned_reset", 0),
            )

        retired_hints = await _global_maintenance_call(
            retire_unreachable_hint_edits, default=0
        )
        if retired_hints:
            logger.info(
                "run_sn_pools: retired %d terminal name-hint edit(s)",
                retired_hints,
            )

        # Realign grammar segment columns with each name's canonical id, so a
        # stale segment (e.g. position='pedestal' on an ..._at_pedestal_top
        # name written by a since-removed import path) self-heals and a
        # re-composition never diverges from the accepted id.
        seg_result = await _global_maintenance_call(
            reconcile_grammar_segments, default={}
        )
        if seg_result.get("names_realigned", 0):
            logger.info(
                "run_sn_pools: grammar-segment reconcile — %d name(s) realigned to canonical id",
                seg_result["names_realigned"],
            )

        # Advance any source-backed pipeline name stranded below the review
        # entry stage. A refine that converged onto a pre-existing placeholder
        # name kept it at 'pending' (its successor init was ON CREATE only), so
        # a valid, source-backed name could sit unreviewed forever — the name
        # review pool claims 'drafted'. persist_refined_name now advances such a
        # successor at write time; this heals any name already stranded and is
        # an idempotent net. Runs BEFORE the pools so the review pool sees the
        # advanced names the same run.
        from imas_codex.standard_names.graph_ops import (
            reconcile_reviewable_name_stage,
        )

        entry_result = await _global_maintenance_call(
            reconcile_reviewable_name_stage, default={}
        )
        if entry_result.get("names_advanced", 0):
            logger.info(
                "run_sn_pools: review-entry reconcile — %d stranded name(s) "
                "advanced to 'drafted'",
                entry_result["names_advanced"],
            )

        # Link COCOS-dependent names to the catalog's COCOS convention
        # (current DD version's — DDv4 → COCOS 17). Sets the cocos integer and
        # HAS_COCOS edge for any COCOS-dependent name missing it, across all
        # origins. Idempotent; no-op once the invariant holds.
        from imas_codex.standard_names.graph_ops import (
            reconcile_standard_name_cocos_links,
        )

        cocos_result = await _global_maintenance_call(
            reconcile_standard_name_cocos_links, default={}
        )
        if cocos_result.get("scalars_set", 0) or cocos_result.get("edges_created", 0):
            logger.info(
                "run_sn_pools: COCOS-link reconcile — %d cocos scalar(s) set, "
                "%d HAS_COCOS edge(s) created (COCOS %s)",
                cocos_result.get("scalars_set", 0),
                cocos_result.get("edges_created", 0),
                cocos_result.get("convention"),
            )

        # Apply registered self-contradiction unit corrections to the stored DD
        # graph. The build rewrites a DD unit when the exceptions registry flags
        # the path, but only at build time, so adding such an entry had no effect
        # on paths already stored and a full DD rebuild is far too expensive to
        # run for a unit correction. Units are DD-authoritative and flow into
        # every composed name, so this runs before anything that reads them.
        from imas_codex.graph.dd_graph_ops import reconcile_dd_unit_corrections

        dd_unit_result = await _global_maintenance_call(
            reconcile_dd_unit_corrections, default={}
        )
        if dd_unit_result.get("corrected", 0):
            logger.info(
                "run_sn_pools: DD-unit correction reconcile — %d stored unit(s) "
                "realigned with the exceptions registry",
                dd_unit_result["corrected"],
            )

        # Realign each name's HAS_UNIT edge set with its own unit scalar. A name
        # whose unit was written before the writers self-healed can carry the
        # superseded edge alongside the current one, leaving it with no single
        # dimensionality — and the attachment guard below compares dimensionality,
        # so an ambiguous name admits sources of either. Runs FIRST of the
        # structural passes for that reason. Idempotent.
        from imas_codex.standard_names.graph_ops import (
            reconcile_standard_name_unit_edges,
        )

        unit_edge_result = await _global_maintenance_call(
            reconcile_standard_name_unit_edges, default={}
        )
        if unit_edge_result.get("names_realigned", 0):
            logger.info(
                "run_sn_pools: unit-edge reconcile — %d name(s) realigned "
                "(%d stale edge(s) dropped, %d created)",
                unit_edge_result.get("names_realigned", 0),
                unit_edge_result.get("edges_dropped", 0),
                unit_edge_result.get("edges_created", 0),
            )

        # Re-ask the attachment guard of every source→name edge already stored.
        # The guard is consulted at compose time only, so an edge written before
        # a rule existed — or written by one of the paths that migrate a whole
        # source set wholesale (a refine successor, an edit cascade, a catalog
        # import) — is never revisited, and a decision cached durably and never
        # re-evaluated when the deciding logic improves stays permanently wrong.
        # Rejected edges are detached and their freed sources returned to
        # 'extracted' so the generate pool composes a correct name the same run.
        # Accepted names are catalog-authoritative and are reported, not
        # detached, without an explicit opt-in. Idempotent.
        from imas_codex.standard_names.attachment_audit import (
            AttachmentAuditResult,
            reconcile_attachment_consistency,
        )

        attach_result = (
            AttachmentAuditResult()
            if skip_global_maintenance
            else await asyncio.to_thread(reconcile_attachment_consistency)
        )
        if attach_result.detached or attach_result.rejected:
            logger.info(
                "run_sn_pools: attachment-consistency reconcile — %d of %d "
                "attachment(s) rejected, %d detached, %d source(s) rerouted "
                "(by rule: %s)",
                len(attach_result.rejected),
                attach_result.checked,
                attach_result.detached,
                attach_result.sources_rerouted,
                attach_result.by_rule(),
            )

        # Materialize the DD-side HAS_STANDARD_NAME edge from per-source
        # provenance, so a name reaches every DD path its provenance asserts —
        # not just the one source that seeded it. Gated on DD-eligibility and
        # units_agree; unit-disagreeing pairs are dropped and logged to the
        # unit-curation triage. Runs BEFORE the source_paths reconcile so the
        # scalar picks up the new edges the same run. Idempotent.
        from imas_codex.standard_names.graph_ops import (
            reconcile_standard_name_dd_edges,
        )

        dd_edge_result = await _global_maintenance_call(
            reconcile_standard_name_dd_edges, default={}
        )
        if dd_edge_result.get("edges_created", 0) or dd_edge_result.get(
            "pairs_dropped", 0
        ):
            logger.info(
                "run_sn_pools: DD-edge reconcile — %d HAS_STANDARD_NAME edge(s) "
                "materialized, %d pair(s) dropped on unit disagreement",
                dd_edge_result.get("edges_created", 0),
                dd_edge_result.get("pairs_dropped", 0),
            )

        # Materialize the denormalised source_paths scalar from live edges, so a
        # remapped/pruned/refined mapping can't leave the scalar stale (no other
        # path reconciles it). Edges are the source of truth; idempotent.
        from imas_codex.standard_names.graph_ops import (
            reconcile_standard_name_source_paths,
        )

        sp_result = await _global_maintenance_call(
            reconcile_standard_name_source_paths, default={}
        )
        if sp_result.get("names_reconciled", 0):
            logger.info(
                "run_sn_pools: source_paths reconcile — %d name(s) materialized "
                "from live edges",
                sp_result["names_reconciled"],
            )

        # Read-only ledger-health probe — a diagnostic, never fatal to the run.
        try:
            orphans = await asyncio.to_thread(find_provenance_orphans)
        except Exception as exc:  # noqa: BLE001 - diagnostic must not abort the run
            logger.debug("run_sn_pools: orphan-count probe skipped: %s", exc)
            orphans = []
        if orphans:
            logger.warning(
                "run_sn_pools: ledger invariant VIOLATED — %d live name(s) have NO "
                "source. Reattach ran this cycle, so these lack any recoverable "
                "producing source; investigate the origin. First few: %s",
                len(orphans),
                ", ".join(o["sn_id"] for o in orphans[:5]),
            )

        # Read-only grammar-gate probe — live names carrying a flux-surface
        # reduction operator on a flux-function base (constant_on_flux_surface)
        # can no longer be minted; any survivor is legacy debt to supersede.
        try:
            from imas_codex.standard_names.audits import (
                find_flux_surface_reduction_violations,
            )

            gate_violations = await asyncio.to_thread(
                find_flux_surface_reduction_violations
            )
        except Exception as exc:  # noqa: BLE001 - diagnostic must not abort the run
            logger.debug("run_sn_pools: gate-violation probe skipped: %s", exc)
            gate_violations = []
        if gate_violations:
            logger.warning(
                "run_sn_pools: flux-surface-reduction gate VIOLATED by %d live "
                "name(s) — reductions of flux functions are no-ops the grammar "
                "now rejects; supersede them onto the unreduced names. First "
                "few: %s",
                len(gate_violations),
                ", ".join(v["id"] for v in gate_violations[:5]),
            )

        # Read-only DD-version probe — live names fed by a DD path that the
        # current DD removed/renamed away can no longer be seeded; survivors
        # are pre-gate legacy debt to re-anchor or retire.
        try:
            from imas_codex.standard_names.audits import find_removed_dd_sources

            removed_srcs = await asyncio.to_thread(find_removed_dd_sources)
        except Exception as exc:  # noqa: BLE001 - diagnostic must not abort the run
            logger.debug("run_sn_pools: removed-dd-source probe skipped: %s", exc)
            removed_srcs = []
        if removed_srcs:
            logger.warning(
                "run_sn_pools: %d live name(s) still fed by DD paths absent "
                "from the current DD — re-anchor (renamed_to) or retire. "
                "First few: %s",
                len(removed_srcs),
                ", ".join(v["id"] for v in removed_srcs[:5]),
            )

        # ── DD source-drift refresh ───────────────────────────────────
        # Idempotent, always on: names record the DD-source snapshot
        # (unit/documentation) they were built against; any that no longer match
        # the live IMASNode (e.g. after a new DD version corrects a unit) are
        # steered through a docs refine carrying the exact DD delta as the edit
        # reason. Unstamped names are baselined first (no mass-refine on first
        # run). No-op when nothing drifted. Runs whenever docs are in scope —
        # skipped only in names-only mode, where the docs pools it feeds do not run.
        if not names_only:
            from imas_codex.standard_names.source_refresh import (
                refresh_drifted_sources,
            )

            sr = await _global_maintenance_call(refresh_drifted_sources, default={})
            if sr.get("baselined") or sr.get("detected"):
                logger.info(
                    "run_sn_pools: source-drift refresh — %d baselined, "
                    "%d drifted, %d steered, %d blocked",
                    sr.get("baselined", 0),
                    sr.get("detected", 0),
                    sr.get("steered", 0),
                    len(sr.get("blocked", [])),
                )

        # ── Stranded-reviewed promotion ───────────────────────────────
        # Idempotent, always on: a name is scored once and staged against the
        # threshold in force at review time. When the acceptance threshold is
        # later lowered, names that scored between the old and new thresholds
        # sit stuck at 'reviewed' — refine only claims BELOW-threshold names, so
        # a stored score that already clears the current threshold is never
        # re-touched. Promote those (both name and docs axes) to 'accepted'.
        # No-op when nothing is stranded. Names carrying an unapplied edit
        # (edit_status='open') are left for the normal accept path so their
        # rename / descendant cascade still applies.
        from imas_codex.standard_names.graph_ops import promote_stranded_reviewed

        _promote_min = min_score if min_score is not None else DEFAULT_MIN_SCORE
        promoted = await _global_maintenance_call(
            promote_stranded_reviewed,
            _promote_min,
            default={},
        )
        if promoted.get("name") or promoted.get("docs"):
            logger.info(
                "run_sn_pools: promoted %d stranded reviewed name(s) + %d "
                "docs to accepted (stored score >= %.3f)",
                promoted.get("name", 0),
                promoted.get("docs", 0),
                _promote_min,
            )

        # ── Domain extract and auto-seeding ───────────────────────
        # Skip auto-seeding in focus mode — sources are pre-seeded by CLI.
        # Skip auto-seeding in flush / docs-only mode — only drain existing.
        # Skip auto-seeding when the generate phase is excluded (skip_generate,
        # e.g. ``--only link``) — seeding new sources would be composed by a
        # generate pool that is not going to run.
        if scope_run_id or edits_only:
            _scope_label = (
                f"focus (run_id={scope_run_id[:8]}…)"
                if scope_run_id
                else "edits (edit_status='open')"
            )
            logger.info(
                "run_sn_pools: %s mode — skipping auto-seed",
                _scope_label,
            )
            _domains = domains
        elif flush or docs_only or skip_generate:
            logger.info(
                "run_sn_pools: %s mode — skipping auto-seed",
                "flush" if flush else ("docs-only" if docs_only else "skip-generate"),
            )
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

        # ── Structural-edge derivation and parent-source repair ────────────────
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
            structural_accept_derived_parents,
        )

        edge_result = await _global_maintenance_call(
            rederive_structural_edges, default={}
        )
        logger.debug(
            "rederive_structural_edges processed %d SN(s)",
            edge_result.get("processed", 0),
        )
        if edge_result.get("migrated"):
            logger.info(
                "Migrated %d HAS_PARENT edges off superseded parents",
                edge_result["migrated"],
            )

        parent_count = await _global_maintenance_call(seed_parent_sources, default=0)
        if parent_count:
            logger.info("Seeded %d parent component sources", parent_count)
        repaired_parent_count = await _global_maintenance_call(
            normalize_derived_parent_lifecycle,
            default=0,
        )
        if repaired_parent_count:
            logger.info(
                "Normalized %d derived parent lifecycle nodes",
                repaired_parent_count,
            )
        # Derived parents are never name-reviewed/refined; promote any that
        # reached drafted/reviewed/exhausted (via a child's refine or legacy
        # routing) to accepted structurally so they never strand on the name
        # axis. Self-healing — runs every startup.
        _structural_accepted = await _global_maintenance_call(
            structural_accept_derived_parents,
            default=0,
        )
        if _structural_accepted:
            logger.info(
                "Structurally accepted %d derived parent(s) on the name axis",
                _structural_accepted,
            )

        # Seed the structural provenance source for any parent that reached a
        # non-null name_stage before it was seeded (a child's refine producing
        # the parent-general name, or structural acceptance above) — those paths
        # accept the parent but write no PRODUCED_NAME source, stranding it as a
        # ledger orphan. Runs after acceptance so same-run parents are covered;
        # idempotent and self-healing.
        from imas_codex.standard_names.graph_ops import (
            reconcile_orphan_parent_sources,
        )

        _parent_sources = await _global_maintenance_call(
            reconcile_orphan_parent_sources, default=0
        )
        if _parent_sources:
            logger.info(
                "Seeded %d missing parent provenance source(s)",
                _parent_sources,
            )

        # Maintenance-only runs include the structural parent lifecycle work
        # above, then stop at the control-flow boundary before any claim-capable
        # pool or auxiliary worker is constructed.
        if reconcile_only:
            logger.info(
                "run_sn_pools: reconciliation and structural maintenance complete"
            )
            summary.stop_reason = "completed"
            return summary

        # ── Build pool specs ──────────────────────────────────────
        _only_domain_for_pools = _domains[0] if len(_domains) == 1 else None
        specs = _build_pool_specs(
            shared_mgr,
            stop_event,
            compose_model=compose_model,
            min_score=min_score,
            rotation_cap=rotation_cap,
            escalation_model=escalation_model,
            review_name_backlog_cap=review_name_backlog_cap,
            review_docs_backlog_cap=review_docs_backlog_cap,
            on_event=on_event,
            only_domain=_only_domain_for_pools,
            scope_run_id=scope_run_id,
            scope_size_hint=scope_size_hint,
            drain_scope_id=drain_scope_id,
            edits_only=edits_only,
            names_only=names_only,
            docs_only=docs_only,
            flush=flush,
            skip_review=skip_review,
            skip_generate=skip_generate,
            only_pool=only_pool,
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
        from imas_codex.standard_names.orphan_sweep import (
            run_manifest_drain_heartbeat_loop,
            run_orphan_sweep_loop,
        )

        sweep_task: asyncio.Task[None] | None = None
        if not skip_global_maintenance:
            sweep_task = asyncio.create_task(
                run_orphan_sweep_loop(
                    interval_s=DEFAULT_ORPHAN_SWEEP_INTERVAL_S,
                    timeout_s=DEFAULT_ORPHAN_SWEEP_TIMEOUT_S,
                    stop_event=stop_event,
                ),
                name="orphan_sweep",
            )
        elif drain_scope_id:
            drain_heartbeat_task = asyncio.create_task(
                run_manifest_drain_heartbeat_loop(
                    drain_scope_id=drain_scope_id,
                    interval_s=max(5, DEFAULT_ORPHAN_SWEEP_TIMEOUT_S // 3),
                    stop_event=stop_event,
                ),
                name="manifest_drain_heartbeat",
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
        embed_task: asyncio.Task[None] | None = None
        if not skip_global_maintenance:
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
                stalled_event=stalled_event,
                pending_count_failed_event=pending_count_failed_event,
            )
        finally:
            if sweep_task is not None and not sweep_task.done():
                sweep_task.cancel()
            if not cost_sync_task.done():
                cost_sync_task.cancel()
            if embed_task is not None and not embed_task.done():
                embed_task.cancel()
            if deadline_task is not None and not deadline_task.done():
                deadline_task.cancel()
            _gather_tasks = [cost_sync_task]
            if sweep_task is not None:
                _gather_tasks.append(sweep_task)
            if embed_task is not None:
                _gather_tasks.append(embed_task)
            if deadline_task is not None:
                _gather_tasks.append(deadline_task)
            await asyncio.gather(*_gather_tasks, return_exceptions=True)
        logger.info("run_sn_pools: all pools exited — %s", health_map)

        # ── Per-pool cost observability ────────────────────────────
        # NB: ``processed`` here is ``PoolHealth.total_processed`` — the number
        # of batch items a pool ATTEMPTED (each ``spec.process`` return value),
        # not the number that persisted. Paid LLM calls whose persist no-oped on
        # a claim-race (see the wasted-paid-call tripwire below) are counted as
        # processed here but did NOT advance graph state; the honest persisted
        # count lives in the SNRun ``names_*`` counters (bumped only on success).
        phase_spent = shared_mgr.phase_spent
        for pool_name, h in (health_map or {}).items():
            processed = getattr(h, "total_processed", 0) if h else 0
            spent = phase_spent.get(pool_name, 0.0)
            mean_cost = spent / processed if processed > 0 else 0.0
            logger.info(
                "run_sn_pools: pool=%s processed=%d spent=$%.4f mean_cost=$%.6f",
                pool_name,
                processed,
                spent,
                mean_cost,
            )
            if processed > 0 and mean_cost == 0.0 and _a3_route == "direct":
                logger.warning(
                    "run_sn_pools: pool=%s has %d processed items but mean_cost=0 "
                    "with expected route='direct' — cost tracking may be broken",
                    pool_name,
                    processed,
                )

        # ── Wasted-paid-call tripwire ──────────────────────────────
        # A paid LLM call whose persist no-oped means a concurrent replica had
        # already advanced the node past our claim: the LLM spend was pure
        # claim-race waste. Surface the per-pool ratio and warn when it exceeds
        # the tripwire threshold so a regressed run is visible in the summary
        # rather than hiding inside an inflated ``processed`` count.
        try:
            from imas_codex.standard_names.graph_ops import (
                persist_outcome_snapshot,
                reset_persist_outcomes,
            )

            _WASTE_TRIPWIRE = 0.02  # >2% wasted paid calls is a regression signal
            _outcomes = persist_outcome_snapshot(run_id)
            for pool_name, oc in sorted(_outcomes.items()):
                attempts = oc["attempts"]
                wasted = oc["wasted"]
                if attempts <= 0:
                    continue
                ratio = wasted / attempts
                logger.info(
                    "run_sn_pools: pool=%s paid_calls=%d wasted_persist=%d "
                    "(%.1f%% claim-race waste)",
                    pool_name,
                    attempts,
                    wasted,
                    ratio * 100.0,
                )
                if ratio > _WASTE_TRIPWIRE and wasted > 2:
                    logger.warning(
                        "run_sn_pools: pool=%s wasted %d/%d paid calls (%.1f%%) on "
                        "claim-race no-op persists — exceeds %.0f%% tripwire; check "
                        "replica scaling vs eligible-set size",
                        pool_name,
                        wasted,
                        attempts,
                        ratio * 100.0,
                        _WASTE_TRIPWIRE * 100.0,
                    )
            reset_persist_outcomes(run_id)
        except Exception:  # noqa: BLE001 — telemetry only, never fail the run
            logger.debug(
                "run_sn_pools: wasted-paid-call tripwire check failed", exc_info=True
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
        if pending_count_failed_event.is_set():
            summary.stop_reason = "pending_count_failed"
        elif stalled_event.is_set():
            summary.stop_reason = "stalled"
        elif shared_mgr.hard_exhausted():
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

        if summary.stop_reason in {"completed", "no_eligible_work"} and (
            scope_run_id or drain_scope_id
        ):
            from imas_codex.standard_names.graph_ops import scoped_terminal_residue

            try:
                terminal_residue = await asyncio.to_thread(
                    scoped_terminal_residue,
                    scope_run_id=scope_run_id,
                    drain_scope_id=drain_scope_id,
                )
            except Exception as terminal_exc:  # noqa: BLE001
                summary.stop_reason = "terminal_scope_unproven"
                logger.error(
                    "run_sn_pools: scoped terminal consistency query failed: %s",
                    terminal_exc,
                )
            else:
                if terminal_residue["total"]:
                    summary.stop_reason = "transient_scope_residue"
                    logger.error(
                        "run_sn_pools: scoped terminal residue refuses successful "
                        "completion: %s",
                        _json.dumps(terminal_residue, sort_keys=True, default=str),
                    )

    except KeyboardInterrupt:
        summary.stop_reason = "interrupted"
        logger.warning("run_sn_pools interrupted by user")
    except Exception as exc:
        primary_error = exc
        primary_traceback = exc.__traceback__
        summary.stop_reason = "failed"
        logger.error("run_sn_pools failed: %s", exc, exc_info=True)
    finally:
        summary.stopped_at = datetime.now(UTC)

        if drain_heartbeat_task is not None:
            drain_heartbeat_task.cancel()
            with __import__("contextlib").suppress(asyncio.CancelledError):
                await drain_heartbeat_task

        if drain_scope_id:
            from imas_codex.standard_names.graph_ops import (
                finalize_manifest_drain_scope,
            )

            try:
                cleanup_task = asyncio.create_task(
                    asyncio.to_thread(
                        finalize_manifest_drain_scope,
                        drain_scope_id,
                        paths=list(drain_paths),
                        dd_version=drain_dd_version,
                    )
                )
                summary.drain_report = await asyncio.shield(cleanup_task)
            except asyncio.CancelledError:
                logger.warning(
                    "bounded drain cancellation received; exact cleanup continues"
                )
                raise
            except Exception as finalization_exc:  # noqa: BLE001
                cleanup_error = finalization_exc
                logger.error("bounded drain finalization failed: %s", finalization_exc)

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
                _global_maintenance_call(
                    release_all_orphan_claims,
                    default={},
                ),
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
        # Re-derive structural edges from names composed during this run,
        # normalize the resulting parent lifecycle, and repair structural
        # provenance. Maintenance-only mode already completed this sequence
        # before its control-flow boundary, so it must not repeat mutating
        # work during shutdown.
        FIXUP_TIMEOUT = 30.0
        if not reconcile_only and not skip_global_maintenance:
            try:
                from imas_codex.standard_names.graph_ops import (
                    normalize_derived_parent_lifecycle,
                    reconcile_orphan_parent_sources,
                    rederive_structural_edges,
                )

                await asyncio.to_thread(rederive_structural_edges)
                _normalized_parents = await asyncio.to_thread(
                    normalize_derived_parent_lifecycle
                )
                _reconciled_parent_sources = await asyncio.to_thread(
                    reconcile_orphan_parent_sources
                )
                if _normalized_parents or _reconciled_parent_sources:
                    logger.info(
                        "run_sn_pools: post-drain normalized %d parent node(s) "
                        "and reconciled %d parent source(s)",
                        _normalized_parents,
                        _reconciled_parent_sources,
                    )
            except Exception as _structural_exc:  # noqa: BLE001
                logger.warning(
                    "run_sn_pools: post-drain structural maintenance failed: %s",
                    _structural_exc,
                )

        try:
            from imas_codex.standard_names.graph_ops import resolve_doc_links

            _link_stats = await asyncio.wait_for(
                _global_maintenance_call(resolve_doc_links, default={}),
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

        # Family-harmonization bookkeeping: refresh idempotency signatures
        # for every sibling family whose live members are all docs-accepted
        # (a member joined, or a member's docs changed and re-passed review).
        # Purely additive scalar writes; no-op when nothing changed.
        if not names_only and not skip_global_maintenance:
            try:
                from imas_codex.standard_names.harmonize import (
                    restamp_harmonized_families,
                )

                _fam_stats = await asyncio.wait_for(
                    asyncio.to_thread(restamp_harmonized_families),
                    timeout=FIXUP_TIMEOUT,
                )
                if _fam_stats.get("restamped"):
                    logger.info(
                        "run_sn_pools: restamped %d harmonized family(ies) "
                        "(%d unchanged, %d awaiting member docs)",
                        _fam_stats["restamped"],
                        _fam_stats.get("unchanged", 0),
                        _fam_stats.get("not_ready", 0),
                    )
            except TimeoutError:
                logger.warning("run_sn_pools: family restamp timed out (non-fatal)")
            except Exception as _fam_exc:  # noqa: BLE001
                logger.warning("run_sn_pools: family restamp failed: %s", _fam_exc)

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
            "pending_count_failed",
            "stalled",
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

    if primary_error is not None:
        raise primary_error.with_traceback(primary_traceback)
    if (
        cleanup_error is not None
        and primary_error is None
        and summary.stop_reason != "interrupted"
    ):
        raise cleanup_error
    return summary
