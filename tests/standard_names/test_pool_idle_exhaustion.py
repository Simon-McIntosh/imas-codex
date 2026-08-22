"""Tests for the idle-exhaustion watchdog and budget detection.

These tests guard against the idle-loop hang where SN pipeline workers keep
running after their work is exhausted:

1. ``_budget_watchdog`` sets ``stop_event`` on hard exhaustion.
2. ``_budget_saturation_watchdog`` sets ``stop_event`` when every pool
   that still has pending work has exceeded the consecutive
   reserve-failure threshold (idle pools, whose counters freeze below
   threshold, do not veto the gate).
3. ``_idle_exhaustion_watchdog`` sets ``stop_event`` only after a sustained
   fresh observation of zero pending counts, zero in-flight batches, and zero
   progress.
4. ``run_pools`` exits with the supplied ``idle_exhausted_event`` set
   when the idle watchdog fires.
"""

from __future__ import annotations

import asyncio

import pytest

from imas_codex.standard_names.budget import BudgetManager
from imas_codex.standard_names.pools import PoolSpec, run_pools

# ---------------------------------------------------------------------------
# Budget: MIN_VIABLE_TURN and near_exhausted must stay removed
# ---------------------------------------------------------------------------


class TestMinViableTurnRemoved:
    def test_min_viable_turn_constant_removed(self) -> None:
        """MIN_VIABLE_TURN must no longer exist in the budget module."""
        import imas_codex.standard_names.budget as budget_mod

        assert not hasattr(budget_mod, "MIN_VIABLE_TURN")

    def test_near_exhausted_method_removed(self) -> None:
        """near_exhausted must no longer exist on BudgetManager."""
        assert not hasattr(BudgetManager, "near_exhausted")


# ---------------------------------------------------------------------------
# Helpers — idle pool with controllable pending_count
# ---------------------------------------------------------------------------


def _make_idle_spec(name: str, *, pending: int = 0) -> PoolSpec:
    """Pool that always returns no work."""

    async def claim() -> None:
        await asyncio.sleep(0.01)
        return None

    async def process(batch: object) -> int:  # pragma: no cover
        return 0

    spec = PoolSpec(name=name, claim=claim, process=process)
    spec.health.pending_count = pending
    spec.backoff.base = 0.05
    spec.backoff.cap = 0.1
    spec.backoff.reset()
    return spec


# ---------------------------------------------------------------------------
# Idle-exhaustion watchdog
# ---------------------------------------------------------------------------


class TestIdleExhaustionWatchdog:
    """Verify the watchdog exits run_pools when all pools are idle."""

    @pytest.mark.asyncio
    async def test_run_pools_exits_after_idle_threshold(self) -> None:
        """All pools have pending_count==0 and never make progress —
        run_pools must exit with idle_exhausted_event set."""
        mgr = BudgetManager(total_budget=5.0)
        stop_event = asyncio.Event()
        idle_exhausted = asyncio.Event()

        pools = [_make_idle_spec("generate", pending=0)]

        # Tight idle params so the test runs in <1s.
        await asyncio.wait_for(
            run_pools(
                pools,
                mgr,
                stop_event,
                grace_period=0.5,
                weights={"generate": 1.0},
                idle_exhausted_event=idle_exhausted,
                idle_exhaustion_poll=0.05,
                idle_exhaustion_polls=3,
            ),
            timeout=5.0,
        )

        assert idle_exhausted.is_set(), (
            "idle_exhausted_event must be set when all pools sit idle"
        )
        assert stop_event.is_set()

    @pytest.mark.asyncio
    async def test_run_pools_without_pending_callback_exits_after_batch(self) -> None:
        """A completed batch must not disable natural idle shutdown."""
        mgr = BudgetManager(total_budget=5.0)
        stop_event = asyncio.Event()
        idle_exhausted = asyncio.Event()
        state = {"available": True}

        async def claim() -> dict[str, str] | None:
            if not state["available"]:
                return None
            state["available"] = False
            return {"id": "candidate"}

        async def process(batch: dict[str, str]) -> int:
            return 1

        spec = PoolSpec(name="generate", claim=claim, process=process)
        await asyncio.wait_for(
            run_pools(
                [spec],
                mgr,
                stop_event,
                grace_period=0.5,
                weights={"generate": 1.0},
                idle_exhausted_event=idle_exhausted,
                idle_exhaustion_poll=0.01,
                idle_exhaustion_polls=3,
                free_pool_set={"generate"},
            ),
            timeout=1.0,
        )

        assert spec.health.total_processed == 1
        assert idle_exhausted.is_set()

    @pytest.mark.asyncio
    async def test_pending_failure_stops_without_certifying_empty(self) -> None:
        """A failed pending query is terminally unproven, never empty."""
        mgr = BudgetManager(total_budget=5.0)
        stop_event = asyncio.Event()
        idle_exhausted = asyncio.Event()
        pending_failed = asyncio.Event()
        spec = _make_idle_spec("generate", pending=0)

        def pending_fn() -> dict[str, int]:
            raise RuntimeError("pending query unavailable")

        await asyncio.wait_for(
            run_pools(
                [spec],
                mgr,
                stop_event,
                pending_fn=pending_fn,
                pending_poll_interval=0.2,
                grace_period=0.5,
                weights={"generate": 1.0},
                idle_exhausted_event=idle_exhausted,
                pending_count_failed_event=pending_failed,
                idle_exhaustion_poll=0.01,
                idle_exhaustion_polls=3,
                free_pool_set={"generate"},
            ),
            timeout=1.0,
        )

        assert stop_event.is_set()
        assert pending_failed.is_set()
        assert not idle_exhausted.is_set()

    @pytest.mark.asyncio
    async def test_pending_positive_stall_has_distinct_signal(self) -> None:
        """A wedged backlog must not be reported as a clean drain."""
        mgr = BudgetManager(total_budget=5.0)
        stop_event = asyncio.Event()
        idle_exhausted = asyncio.Event()
        stalled = asyncio.Event()
        spec = _make_idle_spec("review_name", pending=4)

        await asyncio.wait_for(
            run_pools(
                [spec],
                mgr,
                stop_event,
                grace_period=0.5,
                weights={"review_name": 1.0},
                idle_exhausted_event=idle_exhausted,
                stalled_event=stalled,
                idle_exhaustion_poll=0.01,
                idle_exhaustion_polls=3,
                stall_seconds=0.05,
                free_pool_set={"review_name"},
            ),
            timeout=1.0,
        )

        assert stop_event.is_set()
        assert stalled.is_set()
        assert not idle_exhausted.is_set()

    @pytest.mark.asyncio
    async def test_pending_stall_waits_for_in_flight_batch(self) -> None:
        """A live batch may still turn a pending backlog into progress."""
        from imas_codex.standard_names.pools import _idle_exhaustion_watchdog

        stop_event = asyncio.Event()
        idle_exhausted = asyncio.Event()
        stalled = asyncio.Event()
        spec = _make_idle_spec("review_name", pending=4)
        spec.health.in_flight = 1

        watchdog = asyncio.create_task(
            _idle_exhaustion_watchdog(
                [spec],
                stop_event,
                idle_exhausted,
                stalled,
                poll=0.01,
                idle_polls=3,
                stall_seconds=0.03,
                in_flight_stall_seconds=0.2,
            )
        )
        try:
            await asyncio.sleep(0.08)
            assert not stop_event.is_set()
            assert not stalled.is_set()

            spec.health.in_flight = 0
            await asyncio.wait_for(watchdog, timeout=1.0)
        finally:
            if not watchdog.done():
                stop_event.set()
                await asyncio.gather(watchdog, return_exceptions=True)

        assert stop_event.is_set()
        assert stalled.is_set()
        assert not idle_exhausted.is_set()

    @pytest.mark.asyncio
    async def test_stuck_in_flight_batch_is_terminalized_after_age_bound(
        self,
    ) -> None:
        """A processor that never returns cannot suppress a typed stall forever."""
        mgr = BudgetManager(total_budget=5.0)
        stop_event = asyncio.Event()
        idle_exhausted = asyncio.Event()
        stalled = asyncio.Event()
        entered_process = asyncio.Event()
        process_cancelled = asyncio.Event()
        claimed = False

        async def claim() -> dict[str, str] | None:
            nonlocal claimed
            if claimed:
                return None
            claimed = True
            return {"id": "stuck-candidate"}

        async def process(batch: dict[str, str]) -> int:
            entered_process.set()
            try:
                await asyncio.Event().wait()
            finally:
                process_cancelled.set()
            return 0

        spec = PoolSpec(name="review_name", claim=claim, process=process)

        await asyncio.wait_for(
            run_pools(
                [spec],
                mgr,
                stop_event,
                pending_fn=lambda: {"review_name": 1},
                pending_poll_interval=0.01,
                grace_period=0.01,
                weights={"review_name": 1.0},
                idle_exhausted_event=idle_exhausted,
                stalled_event=stalled,
                idle_exhaustion_poll=0.01,
                idle_exhaustion_polls=3,
                stall_seconds=0.03,
                in_flight_stall_seconds=0.04,
                free_pool_set={"review_name"},
            ),
            timeout=1.0,
        )

        assert entered_process.is_set()
        assert process_cancelled.is_set()
        assert stop_event.is_set()
        assert stalled.is_set()
        assert not idle_exhausted.is_set()
        assert spec.health.in_flight == 0
        assert spec.health.oldest_in_flight_at is None

    def test_younger_replica_keeps_its_age_window_after_older_finishes(
        self,
    ) -> None:
        """Finishing an older batch must age a live replica from its own start."""
        spec = _make_idle_spec("review_name", pending=1)
        in_flight_age_limit = 0.2
        older_started_at = 10.0
        younger_started_at = 10.05

        older_batch = spec.health.mark_batch_started(started_at=older_started_at)
        younger_batch = spec.health.mark_batch_started(started_at=younger_started_at)
        spec.health.mark_batch_finished(older_batch)

        assert spec.health.in_flight == 1
        assert spec.health.oldest_in_flight_at == younger_started_at
        assert (
            spec.health.overdue_in_flight_age(
                age_limit=in_flight_age_limit,
                now=older_started_at + in_flight_age_limit + 0.01,
            )
            is None
        )
        assert spec.health.overdue_in_flight_age(
            age_limit=in_flight_age_limit,
            now=younger_started_at + in_flight_age_limit + 0.01,
        ) == pytest.approx(in_flight_age_limit + 0.01)

        spec.health.mark_batch_finished(younger_batch)
        assert spec.health.in_flight == 0
        assert spec.health.oldest_in_flight_at is None

    @pytest.mark.asyncio
    async def test_idle_watchdog_does_not_fire_with_pending_work(self) -> None:
        """Pools with pending_count > 0 must NOT be considered idle —
        external stop_event is the only exit path here."""
        mgr = BudgetManager(total_budget=5.0)
        stop_event = asyncio.Event()
        idle_exhausted = asyncio.Event()

        pools = [_make_idle_spec("generate", pending=5)]

        async def external_stop() -> None:
            # Long enough that >3 poll cycles elapse — ensures the
            # watchdog had ample opportunity to misfire if its
            # pending_count gate were broken.
            await asyncio.sleep(0.4)
            stop_event.set()

        stopper = asyncio.create_task(external_stop())
        try:
            await asyncio.wait_for(
                run_pools(
                    pools,
                    mgr,
                    stop_event,
                    grace_period=0.5,
                    weights={"generate": 1.0},
                    idle_exhausted_event=idle_exhausted,
                    idle_exhaustion_poll=0.05,
                    idle_exhaustion_polls=3,
                ),
                timeout=5.0,
            )
        finally:
            stopper.cancel()
            await asyncio.gather(stopper, return_exceptions=True)

        assert not idle_exhausted.is_set(), (
            "idle watchdog must not fire while any pool has pending work"
        )

    @pytest.mark.asyncio
    async def test_idle_watchdog_resets_on_progress(self) -> None:
        """A single forward step in ``total_processed`` must reset the
        idle counter so a transient lull does not stop the run."""
        mgr = BudgetManager(total_budget=5.0)
        stop_event = asyncio.Event()
        idle_exhausted = asyncio.Event()

        spec = _make_idle_spec("generate", pending=0)
        pools = [spec]

        async def bump_progress() -> None:
            # Bump progress repeatedly so the watchdog never accumulates
            # 3 consecutive idle polls.
            for _ in range(8):
                await asyncio.sleep(0.05)
                spec.health.total_processed += 1
            stop_event.set()  # exit cleanly via external signal

        bumper = asyncio.create_task(bump_progress())
        try:
            await asyncio.wait_for(
                run_pools(
                    pools,
                    mgr,
                    stop_event,
                    grace_period=0.5,
                    weights={"generate": 1.0},
                    idle_exhausted_event=idle_exhausted,
                    idle_exhaustion_poll=0.05,
                    idle_exhaustion_polls=3,
                ),
                timeout=5.0,
            )
        finally:
            bumper.cancel()
            await asyncio.gather(bumper, return_exceptions=True)

        assert not idle_exhausted.is_set(), (
            "idle watchdog must reset on progress and never fire here"
        )

    @pytest.mark.parametrize(
        ("producer_name", "consumer_name"),
        [
            ("refine_docs", "review_docs"),
            ("refine_name", "review_name"),
        ],
    )
    @pytest.mark.asyncio
    async def test_refine_handoff_reaches_review_before_idle_shutdown(
        self,
        producer_name: str,
        consumer_name: str,
    ) -> None:
        """A slow refine batch may create review work after the idle window."""
        mgr = BudgetManager(total_budget=5.0)
        stop_event = asyncio.Event()
        idle_exhausted = asyncio.Event()
        producer_started = asyncio.Event()
        finish_producer = asyncio.Event()
        downstream_ready = asyncio.Event()
        review_processed = asyncio.Event()
        state = {"refine": True, "drafted": False}

        async def claim_refine() -> dict[str, str] | None:
            if not state["refine"]:
                return None
            state["refine"] = False
            return {"id": "candidate"}

        async def process_refine(batch: dict[str, str]) -> int:
            producer_started.set()
            await finish_producer.wait()
            state["drafted"] = True
            downstream_ready.set()
            return 1

        async def claim_review() -> dict[str, str] | None:
            await downstream_ready.wait()
            if not state["drafted"]:
                return None
            state["drafted"] = False
            return {"id": "candidate"}

        async def process_review(batch: dict[str, str]) -> int:
            review_processed.set()
            return 1

        producer = PoolSpec(
            name=producer_name,
            claim=claim_refine,
            process=process_refine,
        )
        consumer = PoolSpec(
            name=consumer_name,
            claim=claim_review,
            process=process_review,
        )

        def pending_fn() -> dict[str, int]:
            return {
                producer_name: int(state["refine"]),
                consumer_name: int(state["drafted"]),
            }

        run_task = asyncio.create_task(
            run_pools(
                [producer, consumer],
                mgr,
                stop_event,
                pending_fn=pending_fn,
                pending_poll_interval=0.4,
                grace_period=0.5,
                weights={producer_name: 0.5, consumer_name: 0.5},
                idle_exhausted_event=idle_exhausted,
                idle_exhaustion_poll=0.01,
                idle_exhaustion_polls=3,
                free_pool_set={producer_name, consumer_name},
            )
        )
        try:
            await asyncio.wait_for(producer_started.wait(), timeout=1.0)
            await asyncio.sleep(0.08)
            assert not stop_event.is_set(), (
                "idle shutdown must not fire while refine work is in flight"
            )

            finish_producer.set()
            await asyncio.wait_for(review_processed.wait(), timeout=1.0)
            await asyncio.sleep(0.08)
            assert not stop_event.is_set(), (
                "a stale pending snapshot must not authorize idle shutdown"
            )

            await asyncio.wait_for(run_task, timeout=2.0)
        finally:
            finish_producer.set()
            if not run_task.done():
                stop_event.set()
                await asyncio.gather(run_task, return_exceptions=True)

        assert idle_exhausted.is_set()
        assert producer.health.total_processed == 1
        assert consumer.health.total_processed == 1


def test_scoped_terminal_residue_reports_transient_claims() -> None:
    """A refining claim remains terminally visible despite zero pending work."""
    from unittest.mock import MagicMock, patch

    from imas_codex.standard_names.graph_ops import scoped_terminal_residue

    gc = MagicMock()
    gc.__enter__.return_value = gc
    gc.__exit__.return_value = False
    gc.query.return_value = [
        {
            "name_count": 1,
            "source_count": 0,
            "names": [
                {
                    "id": "radial_outline_of_conductor_cross_section",
                    "name_stage": "refining",
                    "docs_stage": "pending",
                    "claim_token": "token",
                    "claimed_at": "2026-08-09T17:52:52Z",
                }
            ],
            "sources": [],
        }
    ]

    with patch("imas_codex.standard_names.graph_ops.GraphClient", return_value=gc):
        residue = scoped_terminal_residue(scope_run_id="bounded-run")

    assert residue["total"] == 1
    assert residue["names"][0]["name_stage"] == "refining"
    cypher = gc.query.call_args.args[0]
    assert "sn.run_id = $scope_id" in cypher
    assert "sns.run_id = $scope_id" in cypher
    assert "claim_token IS NOT NULL" in cypher
    assert "name_stage = 'refining'" in cypher


# ---------------------------------------------------------------------------
# Budget saturation watchdog (replaces the near-exhausted gate)
# ---------------------------------------------------------------------------


class TestBudgetSaturationWatchdog:
    @pytest.mark.asyncio
    async def test_watchdog_fires_on_budget_saturation(self) -> None:
        """When all pools exceed the consecutive reserve-failure threshold,
        the saturation watchdog must set stop_event."""
        mgr = BudgetManager(total_budget=3.0)
        stop_event = asyncio.Event()
        budget_saturated = asyncio.Event()

        pools = [_make_idle_spec("generate", pending=1)]

        async def saturate_budget() -> None:
            await asyncio.sleep(0.1)
            # The saturation watchdog checks pool names derived from the
            # actual PoolSpec list — here just "generate".
            mgr._consecutive_reserve_failures["generate"] = mgr.SATURATION_THRESHOLD

        saturator = asyncio.create_task(saturate_budget())
        try:
            await asyncio.wait_for(
                run_pools(
                    pools,
                    mgr,
                    stop_event,
                    grace_period=0.5,
                    weights={"generate": 1.0},
                    idle_exhausted_event=asyncio.Event(),
                    budget_saturated_event=budget_saturated,
                    # Disable idle watchdog by raising threshold high.
                    idle_exhaustion_poll=10.0,
                    idle_exhaustion_polls=1000,
                ),
                timeout=10.0,
            )
        finally:
            saturator.cancel()
            await asyncio.gather(saturator, return_exceptions=True)

        assert stop_event.is_set()
        assert budget_saturated.is_set()
        assert not mgr.hard_exhausted(), (
            "saturation watchdog should fire before hard exhaustion"
        )

    @pytest.mark.asyncio
    async def test_watchdog_fires_when_live_pools_saturated_despite_idle_pool(
        self,
    ) -> None:
        """Regression: the gate must fire when every pool with pending work
        is budget-saturated, even though another pool sits idle with its
        reserve-failure counter frozen at 0.

        This is the exact 0-token-spin scenario: a free ``generate_name``
        pool drained its sources (pending=0, never reserves → counter
        stays 0) while paid ``review_name`` is budget-blocked (pending>0,
        counter at threshold).  The old all-pools conjunction required the
        idle pool to be saturated too, so it never fired and the run spun
        for hours.  Scoping the gate to live pools fixes this.
        """
        from imas_codex.standard_names.pools import _budget_saturation_watchdog

        mgr = BudgetManager(total_budget=10.0)
        # Idle free pool: no pending work, counter frozen at 0 (never reserves).
        idle_pool = _make_idle_spec("generate_name", pending=0)
        # Live paid pool: pending work it cannot fund — saturated.
        live_pool = _make_idle_spec("review_name", pending=7)
        mgr._consecutive_reserve_failures["review_name"] = mgr.SATURATION_THRESHOLD

        # Sanity: the legacy all-pools predicate would NOT fire here, because
        # the idle generate_name pool never reaches the threshold.
        assert not mgr.all_pools_budget_saturated(("generate_name", "review_name"))

        stop_event = asyncio.Event()
        budget_saturated = asyncio.Event()
        await asyncio.wait_for(
            _budget_saturation_watchdog(
                mgr,
                [idle_pool, live_pool],
                stop_event,
                budget_saturated,
                poll=0.05,
            ),
            timeout=5.0,
        )

        assert stop_event.is_set()
        assert budget_saturated.is_set()
        assert not mgr.hard_exhausted()

    @pytest.mark.asyncio
    async def test_watchdog_does_not_fire_while_free_pool_productive(self) -> None:
        """The gate must NOT fire while a live pool is still unsaturated —
        e.g. a free local-GPU compose pool that keeps reserving successfully
        (counter resets to 0).  Killing it would waste affordable work."""
        from imas_codex.standard_names.pools import _budget_saturation_watchdog

        mgr = BudgetManager(total_budget=10.0)
        productive_pool = _make_idle_spec("generate_name", pending=5)  # not saturated
        blocked_pool = _make_idle_spec("review_name", pending=7)
        mgr._consecutive_reserve_failures["review_name"] = mgr.SATURATION_THRESHOLD

        stop_event = asyncio.Event()
        budget_saturated = asyncio.Event()
        wd = asyncio.create_task(
            _budget_saturation_watchdog(
                mgr,
                [productive_pool, blocked_pool],
                stop_event,
                budget_saturated,
                poll=0.05,
            )
        )
        await asyncio.sleep(0.3)  # several poll cycles
        assert not stop_event.is_set(), (
            "must not shut down while a live pool can still make paid progress"
        )
        assert not budget_saturated.is_set()
        stop_event.set()
        await asyncio.wait_for(wd, timeout=2.0)
