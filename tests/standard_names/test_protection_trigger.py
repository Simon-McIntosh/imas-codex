"""Approval-gate tests for protected Standard Name fields."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from imas_codex.standard_names.protection import filter_protected


def _graph_client(*, rows: list[dict[str, str]] | None = None) -> MagicMock:
    client = MagicMock()
    client.query.return_value = rows
    return client


def test_approved_name_is_protected() -> None:
    """Approval is sufficient and is the only predicate in the graph query."""
    client = _graph_client(rows=[{"id": "approved_name"}])
    with patch("imas_codex.graph.client.GraphClient") as graph_client:
        graph_client.return_value.__enter__.return_value = client
        filtered, skipped = filter_protected(
            [{"id": "approved_name", "description": "replacement"}]
        )

    query = client.query.call_args.args[0]
    assert "sn.name_stage = 'approved'" in query
    assert "sn.origin" not in query
    assert filtered == [{"id": "approved_name"}]
    assert skipped == ["approved_name"]


def test_unapproved_legacy_marker_is_not_protected() -> None:
    """An obsolete marker without approval cannot protect editorial fields."""
    client = _graph_client()
    client.query.side_effect = lambda query, **_params: (
        [{"id": "legacy_name"}] if "sn.origin" in query else []
    )
    with patch("imas_codex.graph.client.GraphClient") as graph_client:
        graph_client.return_value.__enter__.return_value = client
        item = {"id": "legacy_name", "description": "pipeline text"}
        filtered, skipped = filter_protected([item])

    assert filtered == [item]
    assert skipped == []


def test_query_failure_refuses_write() -> None:
    """Loss of approval authority propagates instead of disabling protection."""
    client = _graph_client()
    client.query.side_effect = RuntimeError("graph unavailable")
    with (
        patch("imas_codex.graph.client.GraphClient") as graph_client,
        pytest.raises(RuntimeError, match="graph unavailable"),
    ):
        graph_client.return_value.__enter__.return_value = client
        filter_protected([{"id": "unknown_name", "description": "replacement"}])
