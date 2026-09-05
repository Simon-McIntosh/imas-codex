"""A source parked at ``failed`` states why, or the write is refused.

A terminal source carrying no ``last_error`` cannot be triaged: nothing
distinguishes a transient fault from a permanent one, so the row is never
returned to the queue and never closed. These tests hold every write path
that can set a source status to ``failed`` to a non-empty reason.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from imas_codex.standard_names import graph_ops

REASONLESS = [None, "", "   ", "\n\t "]


def _outcome_client() -> MagicMock:
    """A GraphClient whose single query reports the outcome batch as a winner."""
    gc = MagicMock()
    gc.query.return_value = [{"id": "dd:wall/outline/r"}]
    client = MagicMock()
    client.__enter__.return_value = gc
    client.__exit__.return_value = False
    client.query = gc.query
    return client


def _gap_client() -> tuple[MagicMock, MagicMock]:
    transaction = MagicMock()
    transaction.closed = False
    transaction.run.side_effect = [
        [{"id": "dd:wall/outline/r", "source_id": "wall/outline/r"}],
        [{"id": "dd:wall/outline/r"}],
    ]
    session = MagicMock()
    session.begin_transaction.return_value = transaction

    @contextmanager
    def _session():
        yield session

    client = MagicMock()
    client.__enter__.return_value = client
    client.__exit__.return_value = False
    client.session = _session
    return client, transaction


def _failed_outcome(last_error: Any, *, omit: bool = False) -> dict[str, Any]:
    outcome: dict[str, Any] = {
        "sns_id": "dd:wall/outline/r",
        "source_id": "wall/outline/r",
        "claim_token": "winner",
        "claim_seq": 8,
        "status": "failed",
    }
    if not omit:
        outcome["last_error"] = last_error
    return outcome


@pytest.mark.parametrize("reason", REASONLESS)
def test_claimed_outcome_refuses_a_failed_status_without_a_reason(
    reason: Any,
) -> None:
    client = _outcome_client()

    with (
        patch.object(graph_ops, "GraphClient", return_value=client),
        pytest.raises(ValueError, match="non-empty reason"),
    ):
        graph_ops.persist_claimed_source_outcomes([_failed_outcome(reason)])

    client.query.assert_not_called()


def test_claimed_outcome_refuses_a_failed_status_with_no_error_key() -> None:
    client = _outcome_client()

    with (
        patch.object(graph_ops, "GraphClient", return_value=client),
        pytest.raises(ValueError, match="non-empty reason"),
    ):
        graph_ops.persist_claimed_source_outcomes([_failed_outcome(None, omit=True)])

    client.query.assert_not_called()


def test_claimed_outcome_carries_a_stated_failure_reason_to_the_write() -> None:
    client = _outcome_client()
    reason = "candidate pre-validation failed (unit disagreement): wall_radius"

    with patch.object(graph_ops, "GraphClient", return_value=client):
        assert graph_ops.persist_claimed_source_outcomes([_failed_outcome(reason)]) == [
            "dd:wall/outline/r"
        ]

    persisted = client.query.call_args.kwargs["batch"][0]
    assert persisted["status"] == "failed"
    assert persisted["last_error"] == reason


def test_claimed_outcome_trims_the_stated_failure_reason() -> None:
    client = _outcome_client()

    with patch.object(graph_ops, "GraphClient", return_value=client):
        graph_ops.persist_claimed_source_outcomes(
            [_failed_outcome("  compose returned no candidate  ")]
        )

    persisted = client.query.call_args.kwargs["batch"][0]
    assert persisted["last_error"] == "compose returned no candidate"


def test_a_non_terminal_outcome_still_needs_no_reason() -> None:
    """The refusal is scoped to ``failed``; a skip or a release is unaffected."""
    client = _outcome_client()
    outcome = _failed_outcome(None, omit=True) | {
        "status": "skipped",
        "skip_reason": "not_a_quantity",
    }

    with patch.object(graph_ops, "GraphClient", return_value=client):
        graph_ops.persist_claimed_source_outcomes([outcome])

    client.query.assert_called_once()


@pytest.mark.parametrize("reason", REASONLESS)
def test_gap_writeback_refuses_a_failed_status_without_a_reason(
    reason: Any,
) -> None:
    client, transaction = _gap_client()

    with (
        patch.object(graph_ops, "GraphClient", return_value=client),
        patch.object(graph_ops, "write_vocab_gaps", return_value=1),
        pytest.raises(ValueError, match="non-empty reason"),
    ):
        graph_ops.persist_claimed_vocab_gaps(
            [],
            [_failed_outcome(reason)],
            source_type="dd",
        )

    transaction.run.assert_not_called()
    transaction.commit.assert_not_called()


def test_gap_writeback_carries_a_stated_failure_reason_to_the_write() -> None:
    client, transaction = _gap_client()
    reason = "compose batch of 4 failed vocabulary validation"

    with (
        patch.object(graph_ops, "GraphClient", return_value=client),
        patch.object(graph_ops, "write_vocab_gaps", return_value=1),
    ):
        assert graph_ops.persist_claimed_vocab_gaps(
            [],
            [_failed_outcome(reason)],
            source_type="dd",
        ) == ["dd:wall/outline/r"]

    persisted = transaction.run.call_args_list[-1].kwargs["batch"][0]
    assert persisted["status"] == "failed"
    assert persisted["last_error"] == reason


@pytest.mark.parametrize("reason", REASONLESS)
def test_retry_budget_exhaustion_refuses_an_empty_reason(reason: Any) -> None:
    client = _outcome_client()

    with (
        patch.object(graph_ops, "GraphClient", return_value=client),
        pytest.raises(ValueError, match="non-empty reason"),
    ):
        graph_ops.mark_sources_failed("winner", ["dd:wall/outline/r"], reason)

    client.query.assert_not_called()


def test_retry_budget_exhaustion_records_a_stated_reason() -> None:
    client = _outcome_client()
    gc = client.__enter__.return_value
    gc.query.return_value = [{"affected": 1}]

    with patch.object(graph_ops, "GraphClient", return_value=client):
        assert (
            graph_ops.mark_sources_failed(
                "winner", ["dd:wall/outline/r"], "  model timed out  "
            )
            == 1
        )

    assert gc.query.call_args.kwargs["error"] == "model timed out"
