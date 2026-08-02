"""Atomic refinement persistence across attachment and provenance boundaries."""

from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest
from neo4j.exceptions import TransientError

from imas_codex.standard_names.attachment_audit import (
    AttachmentPairingGuardResult,
    AttachmentVerdict,
)
from imas_codex.standard_names.graph_ops import persist_refined_name


def _transaction(source_ids: list[str]) -> MagicMock:
    tx = MagicMock()

    def run(cypher: str, **_params):
        if "// REFINE_ATOMIC_PREFLIGHT" in cypher:
            return [
                {
                    "old_name": "electron_density",
                    "new_name": "volume_averaged_electron_density",
                    "source_ids": source_ids,
                }
            ]
        if "MERGE (new)-[:REFINED_FROM]->(old)" in cypher:
            return [
                {
                    "old_name": "electron_density",
                    "new_name": "volume_averaged_electron_density",
                }
            ]
        raise AssertionError(f"unexpected transaction query: {cypher}")

    tx.run.side_effect = run
    return tx


def _graph(transaction: MagicMock) -> MagicMock:
    graph = MagicMock()
    graph.__enter__.return_value = graph
    graph.__exit__.return_value = False
    session = MagicMock()
    session.begin_transaction.return_value = transaction
    graph.session.return_value.__enter__.return_value = session
    graph.session.return_value.__exit__.return_value = False
    return graph


@contextmanager
def _persistence_boundary(
    source_ids: list[str],
    guard_result: AttachmentPairingGuardResult,
):
    transaction = _transaction(source_ids)
    graph = _graph(transaction)
    with (
        patch("imas_codex.standard_names.graph_ops.GraphClient", return_value=graph),
        patch(
            "imas_codex.standard_names.attachment_audit.guard_source_pairings",
            return_value=guard_result,
        ) as guard,
        patch(
            "imas_codex.standard_names.provenance_lifecycle."
            "retarget_standard_name_sources",
            return_value=len(source_ids),
        ) as retarget,
        patch(
            "imas_codex.standard_names.provenance_lifecycle."
            "record_standard_name_change",
            return_value="sn-change:test",
        ) as record,
        patch("imas_codex.standard_names.graph_ops.bump_sn_run_counter"),
    ):
        yield transaction, guard, retarget, record


def _persist(**overrides):
    kwargs = {
        "old_name": "electron_density",
        "new_name": "volume_averaged_electron_density",
        "description": "Electron density averaged over volume.",
        "unit": "m^-3",
        "edit_mode": "rename",
        "edit_reason": "make the averaging scope explicit",
        "edit_origin": "human",
        "edit_status": "open",
        "run_id": "sn-edit-test",
        "expected_old_stage": "accepted",
    }
    kwargs.update(overrides)
    return persist_refined_name(**kwargs)


def test_full_source_set_and_ledger_commit_once() -> None:
    sources = ["dd:path/a", "dd:path/b"]
    admitted = AttachmentPairingGuardResult(tuple(sources), ())
    with _persistence_boundary(sources, admitted) as boundary:
        transaction, guard, retarget, record = boundary
        assert _persist() == {
            "old_name": "electron_density",
            "new_name": "volume_averaged_electron_density",
        }

    guard_handle = guard.call_args.args[0]
    assert guard_handle._transaction is transaction
    guard.assert_called_once_with(
        guard_handle, "volume_averaged_electron_density", sources
    )
    assert retarget.call_args.args[:3] == (
        guard_handle,
        "electron_density",
        "volume_averaged_electron_density",
    )
    assert retarget.call_args.kwargs["source_ids"] == sources
    assert retarget.call_args.kwargs["enforce_consistency"] is False
    assert record.call_count == 1
    assert record.call_args.args[0]._transaction is transaction
    assert record.call_args.kwargs["operation"] == "human_edit"
    assert record.call_args.kwargs["run_id"] == "sn-edit-test"
    transaction.commit.assert_called_once_with()
    transaction.rollback.assert_not_called()


@pytest.mark.parametrize("accepted", [("dd:path/a",), ()])
def test_any_rejected_source_rolls_back_everything(
    accepted: tuple[str, ...],
) -> None:
    sources = ["dd:path/a", "dd:path/b"]
    rejected_ids = sorted(set(sources) - set(accepted))
    rejected = tuple(
        AttachmentVerdict(
            source_id,
            source_id.removeprefix("dd:"),
            "volume_averaged_electron_density",
            "drafted",
            "tense mismatch",
        )
        for source_id in rejected_ids
    )
    guard_result = AttachmentPairingGuardResult(accepted, rejected)
    with _persistence_boundary(sources, guard_result) as boundary:
        transaction, _guard, retarget, record = boundary
        with pytest.raises(ValueError, match="rename rolled back"):
            _persist()

    transaction.rollback.assert_called_once_with()
    transaction.commit.assert_not_called()
    retarget.assert_not_called()
    record.assert_not_called()


def test_source_less_rename_still_records_human_edit() -> None:
    admitted = AttachmentPairingGuardResult((), ())
    with _persistence_boundary([], admitted) as boundary:
        transaction, _guard, retarget, record = boundary
        _persist()

    assert retarget.call_args.kwargs["source_ids"] == []
    record.assert_called_once()
    assert record.call_args.kwargs["operation"] == "human_edit"
    transaction.commit.assert_called_once_with()


def test_predecessor_compare_and_set_loss_has_no_mutation() -> None:
    transaction = _transaction([])
    transaction.run.side_effect = lambda *_args, **_kwargs: []
    graph = _graph(transaction)
    with (
        patch("imas_codex.standard_names.graph_ops.GraphClient", return_value=graph),
        patch(
            "imas_codex.standard_names.provenance_lifecycle.record_standard_name_change"
        ) as record,
    ):
        with pytest.raises(RuntimeError, match="predecessor stage"):
            _persist()

    transaction.rollback.assert_called_once_with()
    transaction.commit.assert_not_called()
    record.assert_not_called()


def test_preflight_fences_settled_edits_and_worker_claims() -> None:
    admitted = AttachmentPairingGuardResult((), ())
    with _persistence_boundary([], admitted) as boundary:
        transaction, _guard, _retarget, _record = boundary
        _persist(
            edit_mode=None,
            edit_status="closed",
            expected_old_stage=None,
            expected_claim_token="claim-token",
        )

    preflight = transaction.run.call_args_list[0]
    assert "old.claim_token IS NULL" in preflight.args[0]
    assert "old.claimed_at IS NULL" in preflight.args[0]
    assert "old.claim_token = $expected_claim_token" in preflight.args[0]
    assert preflight.kwargs["expected_claim_token"] == "claim-token"


def test_event_write_failure_rolls_back_graph_mutation() -> None:
    admitted = AttachmentPairingGuardResult(("dd:path/a",), ())
    with _persistence_boundary(["dd:path/a"], admitted) as boundary:
        transaction, _guard, _retarget, record = boundary
        record.side_effect = RuntimeError("event storage unavailable")
        with pytest.raises(RuntimeError, match="event storage"):
            _persist()

    transaction.rollback.assert_called_once_with()
    transaction.commit.assert_not_called()


def test_regular_refine_records_its_operation_once() -> None:
    admitted = AttachmentPairingGuardResult((), ())
    with _persistence_boundary([], admitted) as boundary:
        _transaction_mock, _guard, _retarget, record = boundary
        _persist(
            edit_mode=None,
            edit_reason="review requested clearer wording",
            edit_origin=None,
            edit_status="closed",
            expected_old_stage=None,
        )

    record.assert_called_once()
    assert record.call_args.kwargs["operation"] == "refine"


def test_transient_retry_rolls_back_first_event_and_commits_one() -> None:
    first = _transaction([])
    second = _transaction([])
    graphs = [_graph(first), _graph(second)]
    admitted = AttachmentPairingGuardResult((), ())
    event_calls = 0

    def record(*_args, **_kwargs):
        nonlocal event_calls
        event_calls += 1
        if event_calls == 1:
            raise TransientError("retry transaction")
        return "sn-change:committed"

    with (
        patch(
            "imas_codex.standard_names.graph_ops.GraphClient",
            side_effect=graphs,
        ),
        patch(
            "imas_codex.standard_names.attachment_audit.guard_source_pairings",
            return_value=admitted,
        ),
        patch(
            "imas_codex.standard_names.provenance_lifecycle."
            "retarget_standard_name_sources",
            return_value=0,
        ),
        patch(
            "imas_codex.standard_names.provenance_lifecycle."
            "record_standard_name_change",
            side_effect=record,
        ),
        patch("imas_codex.discovery.base.claims.time.sleep"),
        patch("imas_codex.standard_names.graph_ops.bump_sn_run_counter"),
    ):
        _persist()

    assert event_calls == 2
    first.rollback.assert_called_once_with()
    first.commit.assert_not_called()
    second.commit.assert_called_once_with()
    second.rollback.assert_not_called()
