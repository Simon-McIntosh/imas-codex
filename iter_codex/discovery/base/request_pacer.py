"""Minimum spacing for synchronous and asynchronous discovery requests."""

from __future__ import annotations

import asyncio
import math
import threading
import time


class RequestPacer:
    """Space request starts using a monotonic clock, without an initial delay.

    Each client owns a pacer and uses either its synchronous or asynchronous
    entry point. The lock covers the sleep so concurrent callers cannot consume
    the same slot. A cancelled asynchronous wait consumes no slot.
    """

    def __init__(self, interval: float) -> None:
        if not math.isfinite(interval) or interval < 0:
            raise ValueError("The request interval must be finite and nonnegative")
        self.interval = interval
        self._last_start: float | None = None
        self._lock = threading.Lock()
        self._async_lock: asyncio.Lock | None = None

    def _delay(self) -> float:
        if self._last_start is None:
            return 0.0
        return max(0.0, self.interval - (time.monotonic() - self._last_start))

    def wait(self) -> None:
        """Reserve the next synchronous request start."""
        with self._lock:
            delay = self._delay()
            if delay:
                time.sleep(delay)
            self._last_start = time.monotonic()

    async def wait_async(self) -> None:
        """Reserve the next asynchronous request start without blocking a thread."""
        if self._async_lock is None:
            self._async_lock = asyncio.Lock()
        async with self._async_lock:
            delay = self._delay()
            if delay:
                await asyncio.sleep(delay)
            self._last_start = time.monotonic()
