"""Terminal vocabulary-gap and pool-error states remain explicit failures."""

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from imas_codex.cli.sn import _require_terminal_drain
from imas_codex.standard_names import graph_ops, loop


def _vocab_gap_write_boundary() -> tuple[MagicMock, MagicMock]:
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


def _gap(*, reason: str | None) -> dict[str, str]:
    gap = {
        "source_id": "wall/outline/r",
        "segment": "geometric_base",
        "token": "annular_outline_radius",
    }
    if reason is not None:
        gap["reason"] = reason
    return gap


def _outcome() -> dict[str, object]:
    return {
        "sns_id": "dd:wall/outline/r",
        "source_id": "wall/outline/r",
        "claim_token": "winner",
        "claim_seq": 8,
        "status": "vocab_gap",
        "last_error": None,
    }


def test_vocab_gap_status_refuses_an_omitted_reason() -> None:
    client, transaction = _vocab_gap_write_boundary()

    with (
        patch.object(graph_ops, "GraphClient", return_value=client),
        patch.object(graph_ops, "write_vocab_gaps", return_value=1),
        pytest.raises(ValueError, match="non-empty reason"),
    ):
        graph_ops.persist_claimed_vocab_gaps(
            [_gap(reason=None)],
            [_outcome()],
            source_type="dd",
        )

    transaction.commit.assert_not_called()
    transaction.close.assert_called_once()


def test_vocab_gap_status_records_the_missing_vocabulary() -> None:
    client, transaction = _vocab_gap_write_boundary()

    with (
        patch.object(graph_ops, "GraphClient", return_value=client),
        patch.object(graph_ops, "write_vocab_gaps", return_value=1),
    ):
        assert graph_ops.persist_claimed_vocab_gaps(
            [_gap(reason="the public grammar has no token for this carrier")],
            [_outcome()],
            source_type="dd",
        ) == ["dd:wall/outline/r"]

    persisted = transaction.run.call_args_list[-1].kwargs["batch"][0]
    assert persisted["status"] == "vocab_gap"
    assert persisted["last_error"] == (
        "missing geometric_base vocabulary token 'annular_outline_radius': "
        "the public grammar has no token for this carrier"
    )


def test_counted_pool_error_refuses_a_successful_command_exit() -> None:
    classify = getattr(
        loop,
        "_pool_error_stop_reason",
        lambda stop_reason, health_map: (stop_reason, 0),
    )
    stop_reason, error_count = classify(
        "no_eligible_work",
        {
            "generate_name": SimpleNamespace(
                error_count=1,
                last_error="process: rich standard-name write rejected a winner",
            )
        },
    )

    assert error_count == 1
    assert stop_reason == "failed"
    with pytest.raises(SystemExit) as raised:
        _require_terminal_drain(stop_reason)
    assert raised.value.code == 1


def test_zero_pool_errors_preserve_a_proven_empty_exit() -> None:
    classify = getattr(
        loop,
        "_pool_error_stop_reason",
        lambda stop_reason, health_map: (stop_reason, 0),
    )
    stop_reason, error_count = classify(
        "no_eligible_work",
        {"generate_name": SimpleNamespace(error_count=0, last_error=None)},
    )

    assert error_count == 0
    assert stop_reason == "no_eligible_work"
    _require_terminal_drain(stop_reason)
