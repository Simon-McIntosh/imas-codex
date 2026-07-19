"""Tests for the global adaptive concurrency governor (AIMD backpressure).

The governor caps the number of in-flight LLM calls across the whole process.
Under healthy load it is a no-op (ceiling pinned at ``max_ceiling``); on
provider rate-limits it multiplicatively pulls the ceiling back, then additively
recovers once a cooldown/settle window has elapsed.

Time is injected via ``time_fn`` so every timing assertion is deterministic and
uses no real sleeps.
"""

from __future__ import annotations

import asyncio

import pytest

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
# Additive increase after cooldown / settle
# ---------------------------------------------------------------------------


def test_no_increase_during_cooldown():
    clock = Clock()
    gov = make_governor(clock, cooldown=5.0, settle=1.0)
    gov.record_rate_limited()  # ceiling 64, cooldown until now+5
    assert gov.ceiling == 64
    clock.advance(1.0)  # still inside cooldown
    gov.record_success()
    assert gov.ceiling == 64


def test_additive_ramp_after_cooldown():
    clock = Clock()
    gov = make_governor(clock, cooldown=5.0, settle=1.0)
    gov.record_rate_limited()  # 64
    clock.advance(6.0)  # past cooldown
    gov.record_success()
    assert gov.ceiling == 65
    clock.advance(1.0)
    gov.record_success()
    assert gov.ceiling == 66


def test_settle_gates_bursty_successes():
    clock = Clock()
    gov = make_governor(clock, cooldown=5.0, settle=1.0)
    gov.record_rate_limited()  # 64
    clock.advance(6.0)
    gov.record_success()
    assert gov.ceiling == 65
    # A burst of successes without advancing the clock past settle → no ramp.
    for _ in range(10):
        gov.record_success()
    assert gov.ceiling == 65


def test_ramp_recovers_to_max():
    clock = Clock()
    gov = make_governor(clock, max_ceiling=68, cooldown=5.0, settle=1.0)
    gov.record_rate_limited()  # 34
    clock.advance(6.0)
    for _ in range(200):
        gov.record_success()
        clock.advance(1.0)
    assert gov.ceiling == 68


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
