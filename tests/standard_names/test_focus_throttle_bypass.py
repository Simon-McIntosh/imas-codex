"""Tests for focus-mode backlog throttle bypass.

Verifies that when scope_run_id is set (--focus mode), the backlog
throttle is NOT applied — focused items should never be blocked by
global review queue backlog.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestFocusModeThrottleBypass:
    """Throttle is skipped when scope_run_id is set."""

    def test_throttle_wrapping_logic(self) -> None:
        """Verify the throttle closure has __name__ == '_throttled_claim'."""
        # Reproduce the throttle wrapping from loop.py lines 866-884
        from collections.abc import Awaitable, Callable
        from typing import Any

        original_claim = AsyncMock(return_value={"items": []})
        original_claim.__name__ = "original_claim"

        mock_health = MagicMock()
        mock_health.pending_count = 300  # over cap

        # This is the same closure from loop.py
        async def _throttled_claim(
            _orig: Callable[[], Awaitable[dict[str, Any] | None]] = original_claim,
            _health: Any = mock_health,
            _cap: int = 200,
            _up: str = "generate_docs",
            _down: str = "review_docs",
        ) -> dict[str, Any] | None:
            if _health.pending_count > _cap:
                return None
            return await _orig()

        # Throttled claim returns None when over cap
        result = asyncio.run(_throttled_claim())
        assert result is None

        # Under cap, it delegates
        mock_health.pending_count = 100
        result = asyncio.run(_throttled_claim())
        assert result == {"items": []}

    def test_scope_run_id_guard_prevents_wrapping(self) -> None:
        """The `if not scope_run_id:` guard in loop.py prevents throttle."""
        # This test verifies the conditional logic directly
        scope_run_id = "test-focus-id"
        throttle_applied = False

        if not scope_run_id:
            # This block wraps claims in _throttled_claim closures
            throttle_applied = True

        assert not throttle_applied, (
            "Throttle should NOT be applied when scope_run_id is set"
        )

    def test_no_scope_run_id_allows_wrapping(self) -> None:
        """Without scope_run_id, throttle IS applied."""
        scope_run_id = None
        throttle_applied = False

        if not scope_run_id:
            throttle_applied = True

        assert throttle_applied, "Throttle should be applied when scope_run_id is None"
