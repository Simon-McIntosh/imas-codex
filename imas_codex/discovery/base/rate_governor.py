"""Global adaptive concurrency governor for LLM calls (AIMD backpressure).

Review pools run up to dozens of concurrent replicas, every one of which calls
into :mod:`imas_codex.discovery.base.llm`. When an upstream provider rate-limits
(HTTP 429), the whole fleet keeps hammering it, turning throttling into a wall of
empty/failed responses. This module adds *global* backpressure: a single
process-wide ceiling on in-flight LLM calls that

* **multiplicatively decreases** the moment a 429 is observed, and
* **recovers on wall-clock time** back toward the maximum once the provider has
  settled — after a short cooldown, growing *multiplicatively* (double) at most
  once per settle interval.

This is an AIMD-style control law (as in TCP congestion control), but with one
deliberate departure that fixes a real production defect: recovery is **decoupled
from call completions**. A purely additive, completion-gated ramp (``+1`` per
successful call) crawls back at the rate of throughput — at the floor with slow
high-effort calls that is tens of minutes, and any fresh 429 mid-crawl re-halves
it, so a run gets pinned near the floor for its whole duration. Instead the
ceiling grows on the clock (driven both by :meth:`record_success` and, crucially,
by a blocked :meth:`acquire` on its wait timeout even when *nothing completes*),
multiplicatively, so it returns to full concurrency within
``cooldown + log2(max/min)*settle`` (~9 s with the defaults) regardless of load.

Under healthy load the governor is a **no-op**: the ceiling sits pinned at
``max_ceiling`` (high enough that it never binds) and every ``acquire`` returns
immediately.

The time source is injectable (``time_fn``) so cooldown/settle timing is fully
deterministic in tests with no real sleeps. The state-mutating hooks
(:meth:`record_rate_limited`, :meth:`record_success`) are plain synchronous
methods: they only touch integer/float fields and never ``await``, so on the
single-threaded event loop they are atomic with respect to other coroutines.
On a ceiling increase they schedule a wake of blocked acquirers; :meth:`release`
also wakes them, and a blocked ``acquire`` re-checks on its own settle-bounded
timeout — so a grown ceiling is never invisible to a waiter for longer than
``settle``.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable

__all__ = [
    "AdaptiveConcurrencyGovernor",
    "get_rate_governor",
    "reset_rate_governor",
    "set_rate_governor",
]


class AdaptiveConcurrencyGovernor:
    """Process-global ceiling on concurrent in-flight LLM calls (AIMD).

    Args:
        max_ceiling: Upper bound on concurrency; the ceiling starts here and
            never exceeds it. Chosen high so the governor is a no-op under
            healthy load.
        min_ceiling: Lower bound the multiplicative decrease can never breach.
        decrease_factor: Multiplier applied on each rate-limit
            (``ceiling = max(min_ceiling, floor(ceiling * factor))``).
        cooldown: Seconds after a rate-limit during which increases are
            suppressed (let the provider recover before probing up again).
        settle: Minimum seconds between additive increases, so a burst of
            successes ramps the ceiling up gradually rather than instantly.
        time_fn: Monotonic-style clock returning seconds. Injectable for
            deterministic tests; defaults to :func:`time.monotonic`.
        enabled: When ``False`` the governor is disabled — the ceiling is
            effectively unbounded and the record hooks are no-ops. Used by
            tests and non-throttled domains.
    """

    def __init__(
        self,
        *,
        max_ceiling: int = 128,
        min_ceiling: int = 8,
        decrease_factor: float = 0.5,
        cooldown: float = 5.0,
        settle: float = 1.0,
        time_fn: Callable[[], float] = time.monotonic,
        enabled: bool = True,
    ) -> None:
        if min_ceiling < 1:
            raise ValueError("min_ceiling must be >= 1")
        if max_ceiling < min_ceiling:
            raise ValueError("max_ceiling must be >= min_ceiling")
        if not 0.0 < decrease_factor < 1.0:
            raise ValueError("decrease_factor must be in (0, 1)")

        self._max_ceiling = int(max_ceiling)
        self._min_ceiling = int(min_ceiling)
        self._decrease_factor = float(decrease_factor)
        self._cooldown = float(cooldown)
        self._settle = float(settle)
        self._time_fn = time_fn
        self._enabled = bool(enabled)

        self._ceiling = self._max_ceiling
        self._in_flight = 0
        # No additive increase before this timestamp (set on each rate-limit).
        self._cooldown_until = 0.0
        # Timestamp of the last additive increase (gates bursty ramp-up).
        self._last_increase = 0.0

        # Lazily created so the governor can be constructed outside a running
        # loop (e.g. at import time by the singleton accessor). Rebuilt if the
        # running loop changes, since a process-global singleton may be driven
        # from successive event loops (one asyncio.run per CLI invocation) and
        # an asyncio.Condition is bound to the loop it was first awaited on.
        self._condition: asyncio.Condition | None = None
        self._condition_loop: asyncio.AbstractEventLoop | None = None

    # -- introspection -----------------------------------------------------

    @property
    def ceiling(self) -> int:
        """Current concurrency ceiling."""
        return self._ceiling

    @property
    def in_flight(self) -> int:
        """Number of slots currently held."""
        return self._in_flight

    @property
    def enabled(self) -> bool:
        return self._enabled

    def effective_ceiling(self) -> float:
        """Ceiling that ``acquire`` enforces (``inf`` when disabled)."""
        return float("inf") if not self._enabled else float(self._ceiling)

    # -- AIMD state transitions (sync, non-blocking) -----------------------

    def record_rate_limited(self) -> None:
        """Multiplicative decrease: halve the ceiling and start a cooldown."""
        if not self._enabled:
            return
        self._ceiling = max(
            self._min_ceiling, int(self._ceiling * self._decrease_factor)
        )
        self._cooldown_until = self._time_fn() + self._cooldown

    def _maybe_recover(self, now: float) -> bool:
        """Grow the ceiling toward max on a TIME basis; return True if it grew.

        Recovery is deliberately **decoupled from call completions**: once the
        post-rate-limit cooldown has elapsed and at least ``settle`` seconds have
        passed since the last increase, the ceiling grows *multiplicatively*
        (double, floored at +1) toward ``max_ceiling``. Multiplicative growth
        bounds recovery to ``log2(max/min)`` steps — from the floor to the max
        with the defaults (8 → 128) that is 4 steps of ``settle`` each, so a
        blocked fleet is back to full concurrency within
        ``cooldown + 4*settle`` (~9 s), not the tens of minutes a
        completion-gated +1 crawl took at the floor.

        No-op (returns False) when disabled, already at max, still inside the
        cooldown, or inside the settle interval since the last increase.
        """
        if not self._enabled:
            return False
        if self._ceiling >= self._max_ceiling:
            return False
        if now < self._cooldown_until:
            return False
        if now - self._last_increase < self._settle:
            return False
        self._ceiling = min(
            self._max_ceiling, max(self._ceiling + 1, self._ceiling * 2)
        )
        self._last_increase = now
        return True

    def record_success(self) -> None:
        """Opportunistically recover the ceiling after a completed call.

        Delegates to :meth:`_maybe_recover`; this is only one of two recovery
        drivers — a blocked :meth:`acquire` also recovers on its wait timeout, so
        the ceiling climbs on wall-clock even when zero calls complete. Wakes
        blocked acquirers when the ceiling actually grows.
        """
        if not self._enabled:
            return
        if self._maybe_recover(self._time_fn()):
            self._schedule_wake()

    def _schedule_wake(self) -> None:
        """Wake blocked acquirers after a sync ceiling increase.

        The record hooks are synchronous and cannot hold the async condition
        lock to notify directly, so schedule a tiny task that does. Safe to call
        with no running loop or before any acquire (no condition yet) — both are
        no-ops; a blocked acquirer would still recover on its own wait timeout.
        """
        if self._condition is None:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        loop.create_task(self._notify_waiters())

    async def _notify_waiters(self) -> None:
        cond = self._get_condition()
        async with cond:
            cond.notify_all()

    # -- slot admission ----------------------------------------------------

    def _get_condition(self) -> asyncio.Condition:
        # Bind the condition to the running loop on first use, and rebuild it if
        # a later call runs under a different loop. In-flight bookkeeping lives
        # on plain ints, so a fresh condition after a loop switch is safe (there
        # are no cross-loop waiters when the old loop has closed).
        loop = asyncio.get_running_loop()
        if self._condition is None or self._condition_loop is not loop:
            self._condition = asyncio.Condition()
            self._condition_loop = loop
        return self._condition

    async def acquire(self) -> None:
        """Wait until an in-flight slot is available, then take it.

        A disabled governor admits immediately. Otherwise blocks (without
        busy-waiting) until ``in_flight < ceiling``. The wait is bounded by
        ``settle`` so that a blocked fleet drives time-based recovery
        (:meth:`_maybe_recover`) even when **no calls are completing** — the
        core fix for a ceiling pinned at the floor by a completion-gated ramp.
        The predicate is also re-checked whenever a slot is released or the
        ceiling grows.
        """
        if not self._enabled:
            self._in_flight += 1
            return
        cond = self._get_condition()
        async with cond:
            while self._in_flight >= self._ceiling:
                try:
                    # Bounded wait: on timeout we re-evaluate recovery on the
                    # wall clock rather than waiting for a completion/notify.
                    await asyncio.wait_for(cond.wait(), timeout=self._settle)
                except TimeoutError:
                    # cond.wait() re-acquires the lock before the cancellation
                    # surfaces, so the condition state is safe to touch here.
                    if self._maybe_recover(self._time_fn()):
                        cond.notify_all()
            self._in_flight += 1

    async def release(self) -> None:
        """Release a previously acquired slot and wake waiters."""
        if not self._enabled:
            self._in_flight = max(0, self._in_flight - 1)
            return
        cond = self._get_condition()
        async with cond:
            self._in_flight = max(0, self._in_flight - 1)
            # Wake all waiters so a ceiling that grew while calls were in
            # flight admits every newly available slot, not just one.
            cond.notify_all()

    def slot(self) -> _Slot:
        """Return an async context manager that acquires/releases one slot::

        async with governor.slot():
            response = await litellm.acompletion(**kwargs)
        """
        return _Slot(self)


class _Slot:
    """Async context manager wrapping :meth:`acquire` / :meth:`release`."""

    def __init__(self, governor: AdaptiveConcurrencyGovernor) -> None:
        self._governor = governor

    async def __aenter__(self) -> AdaptiveConcurrencyGovernor:
        await self._governor.acquire()
        return self._governor

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self._governor.release()


# ---------------------------------------------------------------------------
# Process-global singleton
# ---------------------------------------------------------------------------

_GOVERNOR: AdaptiveConcurrencyGovernor | None = None


def get_rate_governor() -> AdaptiveConcurrencyGovernor:
    """Return the lazily-created process-global governor.

    Configuration (enabled flag, ceilings, decrease factor, cooldown, settle)
    is read from :mod:`imas_codex.settings` on first construction, so env
    overrides are honoured. Call :func:`reset_rate_governor` after changing the
    environment to force a rebuild.
    """
    global _GOVERNOR
    if _GOVERNOR is None:
        from imas_codex.settings import (
            get_rate_governor_cooldown,
            get_rate_governor_decrease_factor,
            get_rate_governor_enabled,
            get_rate_governor_max_ceiling,
            get_rate_governor_min_ceiling,
            get_rate_governor_settle,
        )

        _GOVERNOR = AdaptiveConcurrencyGovernor(
            max_ceiling=get_rate_governor_max_ceiling(),
            min_ceiling=get_rate_governor_min_ceiling(),
            decrease_factor=get_rate_governor_decrease_factor(),
            cooldown=get_rate_governor_cooldown(),
            settle=get_rate_governor_settle(),
            enabled=get_rate_governor_enabled(),
        )
    return _GOVERNOR


def set_rate_governor(governor: AdaptiveConcurrencyGovernor | None) -> None:
    """Install *governor* as the process-global instance (or clear with None)."""
    global _GOVERNOR
    _GOVERNOR = governor


def reset_rate_governor() -> None:
    """Drop the cached singleton so the next accessor rebuilds from settings."""
    global _GOVERNOR
    _GOVERNOR = None
