"""Tests for the global adaptive concurrency governor (AIMD backpressure).

The governor caps the number of in-flight LLM calls across the whole process.
Under healthy load it is a no-op (ceiling pinned at ``max_ceiling``); on
provider rate-limits it multiplicatively pulls the ceiling back, then recovers
*multiplicatively on wall-clock time* — even when no calls complete — once a
cooldown/settle window has elapsed.

Time is injected via ``time_fn`` so every timing assertion is deterministic and
uses no real sleeps (the async recovery test uses a sub-10ms settle to exercise
the acquire wait-timeout without meaningful wall-clock cost).

The heartbeat activity-tracker tests live here too since they were reopened
alongside the governor and share the deterministic ``Clock``.
"""

from __future__ import annotations

import asyncio

import pytest

from imas_codex.discovery.base.llm import (
    _ActivityState,
    _advance_cadence,
    _heartbeat_material_change,
    _heartbeat_should_warn,
)
from imas_codex.discovery.base.rate_governor import (
    AdaptiveConcurrencyGovernor,
    get_rate_governor,
    reset_rate_governor,
)


class Clock:
    """Deterministic monotonic-style clock for injection as ``time_fn``."""

    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, dt: float) -> None:
        self.now += dt


def make_governor(clock: Clock, **overrides) -> AdaptiveConcurrencyGovernor:
    params = {
        "max_ceiling": 128,
        "min_ceiling": 2,
        "decrease_factor": 0.5,
        "cooldown": 5.0,
        "settle": 1.0,
        "time_fn": clock,
        **overrides,
    }
    return AdaptiveConcurrencyGovernor(**params)


# ---------------------------------------------------------------------------
# No-op path: no 429s → ceiling never drops
# ---------------------------------------------------------------------------


def test_starts_at_max_ceiling():
    clock = Clock()
    gov = make_governor(clock)
    assert gov.ceiling == 128


def test_successes_without_ratelimit_never_reduce():
    clock = Clock()
    gov = make_governor(clock)
    for _ in range(50):
        clock.advance(2.0)
        gov.record_success()
    # Already at max — successes cannot push it above the ceiling.
    assert gov.ceiling == 128


def test_disabled_governor_is_unbounded():
    clock = Clock()
    gov = make_governor(clock, enabled=False)
    gov.record_rate_limited()
    # A disabled governor ignores backpressure entirely.
    assert gov.ceiling == 128
    assert gov.effective_ceiling() == float("inf")


# ---------------------------------------------------------------------------
# Multiplicative decrease on rate-limit
# ---------------------------------------------------------------------------


def test_rate_limited_halves_ceiling():
    clock = Clock()
    gov = make_governor(clock)
    gov.record_rate_limited()
    assert gov.ceiling == 64
    gov.record_rate_limited()
    assert gov.ceiling == 32


def test_rate_limited_uses_floor():
    clock = Clock()
    gov = make_governor(clock, max_ceiling=9, decrease_factor=0.5)
    gov.record_rate_limited()
    # floor(9 * 0.5) == 4
    assert gov.ceiling == 4


def test_min_ceiling_floor_respected():
    clock = Clock()
    gov = make_governor(clock, min_ceiling=2)
    for _ in range(20):
        gov.record_rate_limited()
    assert gov.ceiling == 2


# ---------------------------------------------------------------------------
# Multiplicative, time-based recovery after cooldown / settle
# ---------------------------------------------------------------------------


def test_no_increase_during_cooldown():
    clock = Clock()
    gov = make_governor(clock, cooldown=5.0, settle=1.0)
    gov.record_rate_limited()  # ceiling 64, cooldown until now+5
    assert gov.ceiling == 64
    clock.advance(1.0)  # still inside cooldown
    gov.record_success()
    assert gov.ceiling == 64


def test_multiplicative_ramp_after_cooldown():
    clock = Clock()
    gov = make_governor(clock, min_ceiling=8, cooldown=5.0, settle=1.0)
    for _ in range(4):
        gov.record_rate_limited()  # 128→64→32→16→8 (floor)
    assert gov.ceiling == 8
    clock.advance(6.0)  # past cooldown
    gov.record_success()
    assert gov.ceiling == 16  # doubled, not +1
    clock.advance(1.0)
    gov.record_success()
    assert gov.ceiling == 32
    clock.advance(1.0)
    gov.record_success()
    assert gov.ceiling == 64


def test_settle_gates_bursty_successes():
    clock = Clock()
    gov = make_governor(clock, min_ceiling=8, cooldown=5.0, settle=1.0)
    for _ in range(4):
        gov.record_rate_limited()  # → 8
    clock.advance(6.0)
    gov.record_success()
    assert gov.ceiling == 16
    # A burst of successes without advancing the clock past settle → no ramp.
    for _ in range(10):
        gov.record_success()
    assert gov.ceiling == 16


def test_time_based_recovery_without_any_successes():
    """Ceiling climbs to max on the clock with ZERO record_success calls.

    Simulates a blocked fleet: the acquire wait-timeout calls _maybe_recover
    each settle even though nothing is completing. Recovery is bounded to
    log2(max/min) multiplicative steps.
    """
    clock = Clock()
    gov = make_governor(clock, min_ceiling=8, cooldown=5.0, settle=1.0)
    for _ in range(4):
        gov.record_rate_limited()  # → 8
    assert gov.ceiling == 8
    clock.advance(5.0)  # cooldown boundary elapsed
    steps = 0
    while gov.ceiling < 128 and steps < 50:
        clock.advance(1.0)
        gov._maybe_recover(clock.now)
        steps += 1
    assert gov.ceiling == 128
    # log2(128 / 8) == 4 steps — the documented recovery bound.
    assert steps == 4


def test_ramp_recovers_to_max():
    clock = Clock()
    gov = make_governor(clock, min_ceiling=8, max_ceiling=64, cooldown=5.0, settle=1.0)
    for _ in range(4):
        gov.record_rate_limited()  # → 8
    clock.advance(6.0)
    for _ in range(200):
        gov.record_success()
        clock.advance(1.0)
    assert gov.ceiling == 64


# ---------------------------------------------------------------------------
# Async concurrency safety
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_concurrent_acquires_never_exceed_ceiling():
    clock = Clock()
    gov = make_governor(clock, max_ceiling=4)

    peak = 0
    in_flight = 0
    lock = asyncio.Lock()

    async def worker():
        nonlocal peak, in_flight
        async with gov.slot():
            async with lock:
                in_flight += 1
                peak = max(peak, in_flight)
            await asyncio.sleep(0)  # yield so others can pile up
            async with lock:
                in_flight -= 1

    await asyncio.gather(*(worker() for _ in range(40)))
    assert peak <= 4
    assert gov.in_flight == 0


@pytest.mark.asyncio
async def test_ceiling_drop_caps_new_acquires():
    clock = Clock()
    gov = make_governor(clock, max_ceiling=8)

    # Fill to the reduced ceiling and prove no more than that run at once.
    gov.record_rate_limited()  # ceiling 4
    assert gov.ceiling == 4

    peak = 0
    in_flight = 0
    lock = asyncio.Lock()
    release = asyncio.Event()

    async def worker():
        nonlocal peak, in_flight
        async with gov.slot():
            async with lock:
                in_flight += 1
                peak = max(peak, in_flight)
            await release.wait()
            async with lock:
                in_flight -= 1

    tasks = [asyncio.create_task(worker()) for _ in range(12)]
    # Let schedulable workers acquire up to the ceiling.
    for _ in range(20):
        await asyncio.sleep(0)
    assert peak <= 4
    assert gov.in_flight == 4
    release.set()
    await asyncio.gather(*tasks)
    assert gov.in_flight == 0


@pytest.mark.asyncio
async def test_disabled_governor_admits_all():
    clock = Clock()
    gov = make_governor(clock, max_ceiling=2, enabled=False)

    peak = 0
    in_flight = 0
    lock = asyncio.Lock()

    async def worker():
        nonlocal peak, in_flight
        async with gov.slot():
            async with lock:
                in_flight += 1
                peak = max(peak, in_flight)
            await asyncio.sleep(0)
            async with lock:
                in_flight -= 1

    await asyncio.gather(*(worker() for _ in range(10)))
    # Disabled → the cap does not apply; more than max_ceiling may run.
    assert peak > 2


@pytest.mark.asyncio
async def test_blocked_acquire_recovers_without_completions():
    """A blocked acquire climbs back via its wait-timeout — no releases needed.

    Sub-10ms settle keeps the real wall-clock cost negligible; the recovery
    DECISION is driven by the injected clock.
    """
    clock = Clock()
    # Small non-zero cooldown so a premature wait-timeout can't grow the ceiling
    # before the clock is advanced past the cooldown (keeps the test robust).
    gov = make_governor(clock, min_ceiling=8, cooldown=0.5, settle=0.01)
    for _ in range(4):
        gov.record_rate_limited()  # → 8
    assert gov.ceiling == 8

    # Take all 8 slots and never release them.
    for _ in range(8):
        await gov.acquire()
    assert gov.in_flight == 8

    blocked = asyncio.create_task(gov.acquire())
    await asyncio.sleep(0)
    assert not blocked.done()

    # No releases, no successes — only the clock advances (past the cooldown).
    # The blocked acquirer's settle-bounded wait must grow the ceiling and
    # admit it.
    clock.advance(1.0)
    await asyncio.wait_for(blocked, timeout=2.0)
    assert gov.in_flight == 9
    assert gov.ceiling >= 9


@pytest.mark.asyncio
async def test_record_success_wakes_blocked_acquirer():
    """A ceiling increase from the sync success hook wakes blocked acquirers.

    Settle is large so the acquire's own wait-timeout cannot fire in the test
    window — the only wake path exercised is the scheduled notify from
    record_success().
    """
    clock = Clock()
    gov = make_governor(clock, min_ceiling=8, cooldown=0.0, settle=100.0)
    for _ in range(4):
        gov.record_rate_limited()  # → 8
    for _ in range(8):
        await gov.acquire()

    blocked = asyncio.create_task(gov.acquire())
    await asyncio.sleep(0)
    assert not blocked.done()

    clock.advance(101.0)  # past settle so the success grows the ceiling
    gov.record_success()  # 8 → 16, schedules a wake of the blocked acquirer
    await asyncio.wait_for(blocked, timeout=2.0)
    assert gov.in_flight == 9


# ---------------------------------------------------------------------------
# LLM-activity heartbeat
# ---------------------------------------------------------------------------


def test_heartbeat_counters_and_stall_warning():
    clock = Clock()
    act = _ActivityState(time_fn=clock)
    act.record_started()
    act.add_spend(0.25)
    snap = act.snapshot()
    assert snap["started"] == 1
    assert snap["in_flight"] == 1
    assert snap["completed"] == 0
    assert snap["spend_usd"] == 0.25

    # In flight but within the stall threshold → no warning.
    clock.advance(60.0)
    assert not _heartbeat_should_warn(act.snapshot(), clock.now, stall_seconds=120.0)

    # In flight and no completion past the threshold → warn.
    clock.advance(70.0)
    assert _heartbeat_should_warn(act.snapshot(), clock.now, stall_seconds=120.0)

    # A completion clears in-flight and resets the stall clock.
    act.record_completed()
    snap = act.snapshot()
    assert snap["completed"] == 1
    assert snap["in_flight"] == 0
    # Nothing in flight → never warns however long the idle.
    clock.advance(10_000.0)
    assert not _heartbeat_should_warn(act.snapshot(), clock.now, stall_seconds=120.0)


def test_heartbeat_failed_decrements_in_flight():
    clock = Clock()
    act = _ActivityState(time_fn=clock)
    act.record_started()
    act.record_failed()
    snap = act.snapshot()
    assert snap["failed"] == 1
    assert snap["in_flight"] == 0
    assert snap["completed"] == 0


def test_heartbeat_new_burst_resets_stall_clock():
    # An idle gap since the last completion must not read as a stall when a
    # fresh burst starts.
    clock = Clock()
    act = _ActivityState(time_fn=clock)
    act.record_started()
    act.record_completed()  # last completion at t=start
    clock.advance(10_000.0)  # long idle
    act.record_started()  # new burst restarts the stall clock
    assert not _heartbeat_should_warn(act.snapshot(), clock.now, stall_seconds=120.0)
    clock.advance(130.0)
    assert _heartbeat_should_warn(act.snapshot(), clock.now, stall_seconds=120.0)


# ---------------------------------------------------------------------------
# Adaptive heartbeat cadence
# ---------------------------------------------------------------------------

_CAD = {"base": 15.0, "factor": 2.0, "cap": 120.0, "fast_beats": 3}


def test_cadence_ramp_stays_at_base_for_fast_beats():
    # The first fast_beats quiet beats all fire at the base interval.
    interval, beat = 15.0, 0
    seen = []
    for _ in range(_CAD["fast_beats"]):
        interval, beat = _advance_cadence(interval, beat, changed=False, **_CAD)
        seen.append(interval)
    assert seen == [15.0, 15.0, 15.0]
    assert beat == 3


def test_cadence_backs_off_geometrically_to_cap():
    # After the ramp, quiet beats double until clamped at the cap.
    interval, beat = 15.0, _CAD["fast_beats"]  # ramp already spent
    seen = []
    for _ in range(6):
        interval, beat = _advance_cadence(interval, beat, changed=False, **_CAD)
        seen.append(interval)
    # 30, 60, 120 (cap), then held at the cap.
    assert seen == [30.0, 60.0, 120.0, 120.0, 120.0, 120.0]


def test_cadence_change_restarts_the_ramp():
    # Deep in backoff, a material change snaps straight back to base and
    # re-earns fast_beats fast beats before backing off again.
    interval, beat = 120.0, 12
    interval, beat = _advance_cadence(interval, beat, changed=True, **_CAD)
    assert interval == 15.0
    assert beat == 1
    # The two following quiet beats stay fast (fast_beats=3), then back off.
    interval, beat = _advance_cadence(interval, beat, changed=False, **_CAD)
    assert interval == 15.0
    interval, beat = _advance_cadence(interval, beat, changed=False, **_CAD)
    assert interval == 15.0
    interval, beat = _advance_cadence(interval, beat, changed=False, **_CAD)
    assert interval == 30.0


def test_material_change_fresh_burst():
    assert _heartbeat_material_change(
        prev_busy=False,
        busy=True,
        prev_failed=0,
        failed=0,
        prev_ceiling=64.0,
        ceiling=64.0,
        stalling=False,
    )


def test_material_change_new_failure_and_stall():
    common = {
        "prev_busy": True,
        "busy": True,
        "prev_ceiling": 64.0,
        "ceiling": 64.0,
    }
    assert _heartbeat_material_change(prev_failed=2, failed=3, stalling=False, **common)
    assert _heartbeat_material_change(prev_failed=2, failed=2, stalling=True, **common)


def test_material_change_ceiling_move_but_nan_is_quiet():
    common = {
        "prev_busy": True,
        "busy": True,
        "prev_failed": 0,
        "failed": 0,
        "stalling": False,
    }
    # A real ceiling move (throttle) counts.
    assert _heartbeat_material_change(prev_ceiling=64.0, ceiling=32.0, **common)
    # A NaN ceiling read (governor blip) never counts as a move.
    nan = float("nan")
    assert not _heartbeat_material_change(prev_ceiling=64.0, ceiling=nan, **common)
    assert not _heartbeat_material_change(prev_ceiling=nan, ceiling=64.0, **common)
    # First observation (prev_ceiling None) is not a move on its own.
    assert not _heartbeat_material_change(prev_ceiling=None, ceiling=64.0, **common)


def test_material_change_steady_state_is_quiet():
    assert not _heartbeat_material_change(
        prev_busy=True,
        busy=True,
        prev_failed=5,
        failed=5,
        prev_ceiling=128.0,
        ceiling=128.0,
        stalling=False,
    )


class _FakeActivity:
    """Serves a scripted sequence of snapshots for the heartbeat loop."""

    def __init__(self, snaps):
        self._snaps = list(snaps)

    def _time_fn(self):
        return 0.0

    def snapshot(self):
        return self._snaps.pop(0)


def _snap(in_flight, failed=0):
    return {
        "in_flight": in_flight,
        "started": 10,
        "completed": 10 - in_flight,
        "failed": failed,
        "spend_usd": 1.23,
        # Recent completion → never a stall in these tests.
        "last_completion_monotonic": 0.0,
    }


async def test_heartbeat_loop_advertises_next_beat_and_idle_trailer(
    monkeypatch, caplog
):
    from imas_codex.discovery.base import llm

    intervals: list[float] = []

    async def fake_sleep(secs):
        intervals.append(secs)

    class _Gov:
        def effective_ceiling(self):
            return 64.0

    # Busy for three beats, then two idle observations → settle re-check + trailer.
    fake = _FakeActivity([_snap(5), _snap(5), _snap(5), _snap(0), _snap(0)])
    monkeypatch.setattr(llm.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(llm, "_ACTIVITY", fake)
    monkeypatch.setattr(llm, "get_rate_governor", lambda: _Gov())

    with caplog.at_level("INFO", logger="imas_codex.discovery.base.llm"):
        await llm._heartbeat_loop(
            15.0, max_interval=120.0, fast_beats=3, backoff_factor=2.0
        )

    beats = [
        r.getMessage() for r in caplog.records if "LLM heartbeat" in r.getMessage()
    ]
    # Five snapshots consumed → five beat lines, the last a trailer.
    assert len(beats) == 5
    # Every busy line advertises the next beat; the fast ramp stays at base.
    assert all("next_beat=" in b for b in beats)
    assert "next_beat=in 15s" in beats[0]
    # The fleet went idle → exactly one idle trailer, and the loop returned.
    assert beats[-1].endswith("next_beat=idle")
    assert sum("next_beat=idle" in b for b in beats) == 1


async def test_heartbeat_loop_backs_off_when_steady(monkeypatch, caplog):
    from imas_codex.discovery.base import llm

    async def fake_sleep(secs):
        pass

    class _Gov:
        def effective_ceiling(self):
            return 64.0

    # Six steady busy beats then idle-idle to terminate: base×3 then 30, 60, 120.
    fake = _FakeActivity([_snap(5)] * 6 + [_snap(0), _snap(0)])
    monkeypatch.setattr(llm.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(llm, "_ACTIVITY", fake)
    monkeypatch.setattr(llm, "get_rate_governor", lambda: _Gov())

    with caplog.at_level("INFO", logger="imas_codex.discovery.base.llm"):
        await llm._heartbeat_loop(
            15.0, max_interval=120.0, fast_beats=3, backoff_factor=2.0
        )

    beats = [
        r.getMessage() for r in caplog.records if "LLM heartbeat" in r.getMessage()
    ]
    advertised = [b.split("next_beat=")[1] for b in beats]
    assert advertised[:6] == [
        "in 15s",
        "in 15s",
        "in 15s",
        "in 30s",
        "in 60s",
        "in 120s",
    ]


# ---------------------------------------------------------------------------
# Process-global singleton
# ---------------------------------------------------------------------------


def test_singleton_is_stable():
    reset_rate_governor()
    a = get_rate_governor()
    b = get_rate_governor()
    assert a is b
    reset_rate_governor()
    c = get_rate_governor()
    assert c is not a


def test_singleton_reads_settings(monkeypatch):
    monkeypatch.setenv("IMAS_CODEX_RATE_GOVERNOR_MAX_CEILING", "16")
    reset_rate_governor()
    gov = get_rate_governor()
    assert gov.ceiling == 16
    reset_rate_governor()
    monkeypatch.delenv("IMAS_CODEX_RATE_GOVERNOR_MAX_CEILING", raising=False)
    reset_rate_governor()
