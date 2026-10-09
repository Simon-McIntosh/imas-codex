"""Shared host reachability checks and worker backoff for discovery."""

from __future__ import annotations

import asyncio
import subprocess
from typing import Protocol


class ReachabilityState(Protocol):
    service_monitor: object | None

    def should_stop(self) -> bool: ...


UNREACHABLE_MAX_BACKOFF = 60.0


def host_unreachable(exc: Exception) -> bool:
    """Identify remote calls that could not establish a usable SSH response."""
    return isinstance(exc, subprocess.TimeoutExpired) or (
        isinstance(exc, subprocess.CalledProcessError) and exc.returncode == 255
    )


async def _sleep_unless_stopped(state: ReachabilityState, seconds: float) -> None:
    """Sleep in short steps so a stop request or deadline ends the wait."""
    loop = asyncio.get_running_loop()
    until = loop.time() + seconds
    while not state.should_stop():
        remaining = until - loop.time()
        if remaining <= 0:
            return
        await asyncio.sleep(min(remaining, 1.0))


async def wait_for_reachable_host(state: ReachabilityState, attempt: int) -> None:
    """Back off, then hold until the SSH monitor reports a healthy host."""
    await _sleep_unless_stopped(state, min(2.0**attempt, UNREACHABLE_MAX_BACKOFF))
    monitor = state.service_monitor
    if monitor is None:
        return
    while not state.should_stop() and not monitor.is_service_healthy("ssh"):
        await _sleep_unless_stopped(state, 5.0)
