"""The claim adapter lets a priority scope replace, not collide with, its caller's scope.

Under ``--focus`` the docs claim adapters receive ``scope_run_id`` through the
shared scope kwargs; ``generate_docs`` and ``review_docs`` also receive a
priority (drift) scope to try first. The priority pass must run against the
priority scope in place of the caller's scope, not alongside it — passing the
same keyword twice raises ``TypeError`` before the claim function is ever
reached, so a focus run that needs documentation work fails outright instead
of claiming. The caller's own scope still gets its turn on the fallback call
when the priority pass returns nothing.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

_GO = "imas_codex.standard_names.graph_ops"

_DRIFT_SCOPE = "drift-run-2"
_FOCUS_SCOPE = "focus-run-1"


def _docs_specs() -> dict[str, Any]:
    """Build the docs pools carrying both a focus and a priority scope."""
    from imas_codex.standard_names.loop import _build_pool_specs

    with patch("imas_codex.settings.get_pool_replicas", return_value=1):
        specs = _build_pool_specs(
            MagicMock(),
            asyncio.Event(),
            scope_run_id=_FOCUS_SCOPE,
            scope_size_hint=1,
            drift_scope_run_id=_DRIFT_SCOPE,
        )
    return {spec.name: spec for spec in specs}


@pytest.mark.asyncio
async def test_priority_scope_replaces_caller_scope_in_priority_pass() -> None:
    """Both scopes supplied: the priority pass claims its scope, not the caller's."""
    seen: list[str] = []

    def _claim(*, scope_run_id: str, **kwargs: Any) -> list[dict]:
        seen.append(scope_run_id)
        assert not kwargs  # no extra claim kwargs are expected here
        return [
            {"id": "sn-1", "claim_token": "tok", "source_type": "dd", "source_id": "p"}
        ]

    with patch(
        "imas_codex.standard_names.graph_store.graph_docs_persistence.claim_generate_docs_batch",
        side_effect=_claim,
    ):
        specs = _docs_specs()
        result = await specs["generate_docs"].claim()

    assert result is not None
    assert result["items"][0]["id"] == "sn-1"
    # The priority scope replaced the caller-supplied focus scope inside the call.
    assert seen == [_DRIFT_SCOPE]


@pytest.mark.asyncio
async def test_empty_priority_pass_falls_back_to_caller_scope() -> None:
    """Priority pass empty: the fallback call claims with the caller's scope."""
    seen: list[str] = []

    def _claim(*, scope_run_id: str, **kwargs: Any) -> list[dict]:
        seen.append(scope_run_id)
        if scope_run_id == _DRIFT_SCOPE:
            return []
        assert scope_run_id == _FOCUS_SCOPE
        return [
            {"id": "sn-9", "claim_token": "tok", "source_type": "dd", "source_id": "q"}
        ]

    with patch(
        "imas_codex.standard_names.graph_store.graph_docs_persistence.claim_generate_docs_batch",
        side_effect=_claim,
    ):
        specs = _docs_specs()
        result = await specs["generate_docs"].claim()

    assert result is not None
    assert result["items"][0]["id"] == "sn-9"
    assert seen == [_DRIFT_SCOPE, _FOCUS_SCOPE]


@pytest.mark.asyncio
async def test_review_docs_priority_scope_replaces_caller_scope() -> None:
    """The identical collision path on the review_docs adapter is also safe."""
    seen: list[str] = []

    def _claim(*, scope_run_id: str, **kwargs: Any) -> list[dict]:
        seen.append(scope_run_id)
        return [
            {"id": "sn-2", "claim_token": "tok", "source_type": "dd", "source_id": "r"}
        ]

    with patch(
        "imas_codex.standard_names.graph_store.graph_review_persistence.claim_review_docs_batch",
        side_effect=_claim,
    ):
        specs = _docs_specs()
        result = await specs["review_docs"].claim()

    assert result is not None
    assert result["items"][0]["id"] == "sn-2"
    assert seen == [_DRIFT_SCOPE]
