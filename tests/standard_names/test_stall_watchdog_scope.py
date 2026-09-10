"""The stall watchdog abandons a wedged call, not the run.

A stall watchdog that registers one overdue in-flight call — a single LLM
call held far past the pool's age bound — and answers by setting ``stop_event``
ends every pool, including the ones still holding claimable work. The recorded
instance: one ``review_name`` call in flight 711 s past a 600 s bound while six
items stayed pending, and the run stopped with ``stop_reason=stalled``, its
wedge residue stranded. The correct scope of the decision is the call, not the
run: the wedged call is abandoned in its own pool — cancelled, its row claim
released so the identity is reclaimable — the forward-progress clock is reset,
and the other pools keep claiming.

A watchdog that never stops is worse than one that stops too eagerly, so the
second half of the rule must hold too: a genuinely deadlocked run — pending
work, nothing in flight anywhere, no progress — still shuts down exactly as it
does today.

This file pins both halves by driving a wedged call through a real
``run_pools`` and showing the abandonment, not merely a green suite.
"""

from __future__ import annotations

import asyncio

import pytest

from imas_codex.standard_names.budget import BudgetManager
from imas_codex.standard_names.pools import PoolSpec, run_pools


async def _wait_for_all(*events: asyncio.Event, timeout: float = 2.0) -> None:
    """Spin until every supplied event is set (bounded by the caller)."""
    deadline = asyncio.get_running_loop().time() + timeout
    while not all(event.is_set() for event in events):
        if asyncio.get_running_loop().time() >= deadline:
            return
        await asyncio.sleep(0.005)


def _fast_backoff(spec: PoolSpec) -> None:
    """Keep empty-claim sleeps tiny so the run pace is set by the watchdog."""
    spec.backoff.base = 0.01
    spec.backoff.cap = 0.05
    spec.backoff.reset()


@pytest.mark.asyncio
async def test_one_wedged_call_leaves_other_pools_and_releases_its_claim() -> None:
    """One overdue in-flight call is abandoned in its own pool, not a stop.

    The review pool claims an identity whose batch processor wedges (never
    returns on its own). Once that call is overdue past the age bound it must
    be cancelled and its claim released so the identity is reclaimable, and
    the run must NOT stall: the generate pool's own claimable work still
    completes, and the released identity is re-claimed and persisted.
    """
    mgr = BudgetManager(total_budget=5.0)
    stop_event = asyncio.Event()
    idle_exhausted = asyncio.Event()
    stalled = asyncio.Event()

    wedge_released = asyncio.Event()
    wedge_persisted = asyncio.Event()
    generate_processed = asyncio.Event()

    wedge = {"released": False, "persisted": False}

    async def claim_wedged() -> dict[str, str] | None:
        # Claim the identity before it is released and again after release
        # (proving it is reclaimable), then stop handing it out.
        if not wedge["released"] or not wedge["persisted"]:
            return {"id": "wedged-identity"}
        return None

    async def process_wedged(batch: dict[str, str]) -> int:
        # Until the claim is released this invocation is the wedged one: it
        # blocks indefinitely and only the pool's abandonment unblocks it
        # (the wait is cancelled; the return below is unreachable). After the
        # release, a re-claim persists and reports progress.
        if not wedge["released"]:
            await asyncio.Event().wait()
            return 0
        wedge["persisted"] = True
        wedge_persisted.set()
        return 1

    async def release_wedged(batch: dict[str, str]) -> None:
        wedge["released"] = True
        wedge_released.set()

    async def claim_generate() -> dict[str, str] | None:
        if generate_processed.is_set():
            return None
        return {"id": "generate-identity"}

    async def process_generate(batch: dict[str, str]) -> int:
        generate_processed.set()
        return 1

    wedged = PoolSpec(
        name="review_name",
        claim=claim_wedged,
        process=process_wedged,
        release=release_wedged,
    )
    generate = PoolSpec(
        name="generate_name",
        claim=claim_generate,
        process=process_generate,
    )
    for spec in (wedged, generate):
        spec.health.pending_count = 1
        _fast_backoff(spec)

    run_task = asyncio.create_task(
        run_pools(
            [wedged, generate],
            mgr,
            stop_event,
            pending_fn=lambda: {"review_name": 1, "generate_name": 1},
            pending_poll_interval=0.01,
            grace_period=0.1,
            weights={"review_name": 1.0, "generate_name": 1.0},
            idle_exhausted_event=idle_exhausted,
            stalled_event=stalled,
            idle_exhaustion_poll=0.01,
            idle_exhaustion_polls=3,
            stall_seconds=0.05,
            in_flight_stall_seconds=0.05,
            free_pool_set={"review_name", "generate_name"},
        ),
        name="stall_scope_watchdog",
    )
    try:
        await asyncio.wait_for(
            _wait_for_all(wedge_released, wedge_persisted, generate_processed),
            timeout=3.0,
        )
    finally:
        stop_event.set()
        await asyncio.wait_for(
            asyncio.gather(run_task, return_exceptions=True), timeout=3.0
        )

    assert wedge_released.is_set()  # the wedged claim was released
    assert wedge_persisted.is_set()  # the released identity was reclaimed
    assert generate_processed.is_set()  # the other pool kept claiming
    assert not stalled.is_set()  # the run did not stall over one wedged call
    assert wedged.health.in_flight == 0


@pytest.mark.asyncio
async def test_genuine_deadlock_still_shuts_down() -> None:
    """Pending work, nothing in flight anywhere, no progress -> typed stall.

    The no-in-flight arm must keep working after the wedge-abandonment change:
    a run whose only remaining work can neither be claimed nor progressed is a
    deadlock, and the watchdog still signals the typed stall as it does today.
    """
    mgr = BudgetManager(total_budget=5.0)
    stop_event = asyncio.Event()
    idle_exhausted = asyncio.Event()
    stalled = asyncio.Event()

    async def claim() -> None:
        await asyncio.sleep(0.005)
        return None

    async def process(batch: object) -> int:  # pragma: no cover
        return 0

    spec = PoolSpec(name="review_name", claim=claim, process=process)
    spec.health.pending_count = 1
    _fast_backoff(spec)

    await asyncio.wait_for(
        run_pools(
            [spec],
            mgr,
            stop_event,
            pending_fn=lambda: {"review_name": 1},
            pending_poll_interval=0.01,
            grace_period=0.1,
            weights={"review_name": 1.0},
            idle_exhausted_event=idle_exhausted,
            stalled_event=stalled,
            idle_exhaustion_poll=0.01,
            idle_exhaustion_polls=3,
            stall_seconds=0.05,
            free_pool_set={"review_name"},
        ),
        timeout=3.0,
    )

    assert stop_event.is_set()
    assert stalled.is_set()
    assert not idle_exhausted.is_set()
