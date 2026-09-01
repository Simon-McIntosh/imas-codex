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
from imas_codex.standard_names.graph_ops import (
    RefinedNamePersistenceRefusal,
    RefinedNamePersistenceRefusalReason,
    persist_refined_name,
)

_EDIT_FIELDS = (
    "edit_mode",
    "name_hint",
    "docs_hint",
    "edit_reason",
    "edit_origin",
    "edit_scope",
    "edit_status",
    "edit_requested_at",
    "edit_override_edits",
    "edit_include_accepted",
)


def _transaction(
    source_ids: list[str],
    *,
    source_edit_state: dict[str, object] | None = None,
    effective_edit_state: dict[str, object] | None = None,
) -> MagicMock:
    tx = MagicMock()
    source_edit_state = source_edit_state or {}
    effective_edit_state = effective_edit_state or {}

    def run(cypher: str, **_params):
        if "// REFINE_ATOMIC_PREFLIGHT" in cypher:
            return [
                {
                    "old_name": "electron_density",
                    "new_name": "volume_averaged_electron_density",
                    "source_ids": source_ids,
                    **{
                        f"source_{field}": source_edit_state.get(field)
                        for field in _EDIT_FIELDS
                    },
                    **{
                        f"effective_{field}": (
                            effective_edit_state[field]
                            if field in effective_edit_state
                            else _params.get(field)
                        )
                        for field in _EDIT_FIELDS
                    },
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


class _RacingEditTransaction:
    """Stateful transaction double that changes edit state after preflight."""

    def __init__(self) -> None:
        self.state = {
            "name_stage": "refining",
            "edit_mode": "rename",
            "edit_reason": "initial reason",
            "edit_status": "open",
        }
        self.successor: dict[str, object] = {}
        self.commit = MagicMock()
        self.rollback = MagicMock()

    def run(self, cypher: str, **params):
        if "// REFINE_ATOMIC_PREFLIGHT" in cypher:
            source_state = {field: self.state.get(field) for field in _EDIT_FIELDS}
            self.successor = dict(source_state)
            self.state["edit_reason"] = "concurrent replacement"
            return [
                {
                    "old_name": params["old_name"],
                    "new_name": params["new_name"],
                    "source_ids": ["dd:path/a"],
                    **{
                        f"source_{field}": value
                        for field, value in source_state.items()
                    },
                    **{
                        f"effective_{field}": value
                        for field, value in self.successor.items()
                    },
                }
            ]
        if "MERGE (new)-[:REFINED_FROM]->(old)" in cypher:
            matches_snapshot = all(
                self.state.get(field) == params.get(f"source_{field}")
                for field in _EDIT_FIELDS
            )
            return (
                [{"old_name": params["old_name"], "new_name": params["new_name"]}]
                if matches_snapshot
                else []
            )
        raise AssertionError(f"unexpected transaction query: {cypher}")


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
    *,
    source_edit_state: dict[str, object] | None = None,
    effective_edit_state: dict[str, object] | None = None,
):
    transaction = _transaction(
        source_ids,
        source_edit_state=source_edit_state,
        effective_edit_state=effective_edit_state,
    )
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

    def run(cypher: str, **_params):
        if "OPTIONAL MATCH (old:StandardName {id: $old_name})" in cypher:
            # The predecessor moved out of the stage the preflight demanded,
            # which is what left the compare-and-set matching nothing.
            return [{"old_exists": True, "old_stage": "refining"}]
        return []

    transaction.run.side_effect = run
    graph = _graph(transaction)
    with (
        patch("imas_codex.standard_names.graph_ops.GraphClient", return_value=graph),
        patch(
            "imas_codex.standard_names.provenance_lifecycle.record_standard_name_change"
        ) as record,
    ):
        with pytest.raises(RefinedNamePersistenceRefusal) as refusal:
            _persist()

    assert refusal.value.reason is RefinedNamePersistenceRefusalReason.PREDECESSOR_STAGE
    transaction.rollback.assert_called_once_with()
    transaction.commit.assert_not_called()
    record.assert_not_called()


def test_preflight_fences_settled_edits_and_worker_claims() -> None:
    admitted = AttachmentPairingGuardResult(("dd:path/a",), ())
    with _persistence_boundary(["dd:path/a"], admitted) as boundary:
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


def test_open_edit_propagation_uses_only_the_atomic_transaction() -> None:
    open_edit = {
        "edit_mode": "rename",
        "name_hint": "volume_averaged_electron_density",
        "edit_reason": "preserve the requested averaging scope",
        "edit_origin": "human",
        "edit_scope": "only_self",
        "edit_status": "open",
        "edit_requested_at": "2026-08-02T10:00:00+00:00",
        "edit_override_edits": False,
        "edit_include_accepted": True,
    }
    transaction = _transaction(
        ["dd:path/a"],
        source_edit_state=open_edit,
        effective_edit_state=open_edit,
    )
    graph = _graph(transaction)
    admitted = AttachmentPairingGuardResult(("dd:path/a",), ())
    with (
        patch(
            "imas_codex.standard_names.graph_ops.GraphClient",
            return_value=graph,
        ) as graph_client,
        patch(
            "imas_codex.standard_names.attachment_audit.guard_source_pairings",
            return_value=admitted,
        ),
        patch(
            "imas_codex.standard_names.provenance_lifecycle."
            "retarget_standard_name_sources",
            return_value=1,
        ),
        patch(
            "imas_codex.standard_names.provenance_lifecycle."
            "record_standard_name_change",
            return_value="sn-change:edit",
        ) as record,
        patch("imas_codex.standard_names.graph_ops.bump_sn_run_counter"),
    ):
        _persist(
            edit_mode=None,
            edit_reason=None,
            edit_origin=None,
            edit_status=None,
            expected_old_stage=None,
        )

    graph_client.assert_called_once_with()
    preflight = transaction.run.call_args_list[0]
    assert preflight.kwargs["inherit_open_edit"] is True
    assert "old.edit_status = 'open'" in preflight.args[0]
    assert "new.edit_reason       = successor_edit_reason" in preflight.args[0]
    record.assert_called_once()
    assert record.call_args.kwargs["operation"] == "human_edit"
    assert record.call_args.kwargs["reason"] == open_edit["edit_reason"]
    transaction.commit.assert_called_once_with()


def test_edit_state_race_fails_the_finalization_fence_and_rolls_back() -> None:
    transaction = _RacingEditTransaction()
    graph = _graph(transaction)
    admitted = AttachmentPairingGuardResult(("dd:path/a",), ())
    with (
        patch("imas_codex.standard_names.graph_ops.GraphClient", return_value=graph),
        patch(
            "imas_codex.standard_names.attachment_audit.guard_source_pairings",
            return_value=admitted,
        ),
        patch(
            "imas_codex.standard_names.provenance_lifecycle."
            "retarget_standard_name_sources"
        ) as retarget,
        patch(
            "imas_codex.standard_names.provenance_lifecycle.record_standard_name_change"
        ) as record,
    ):
        with pytest.raises(RefinedNamePersistenceRefusal) as refusal:
            _persist(
                edit_mode=None,
                edit_reason=None,
                edit_origin=None,
                edit_status=None,
                expected_old_stage=None,
            )

    # The finalization write carries the edit snapshot in its compare-and-set,
    # so a raced edit leaves it matching no row and no successor is written.
    assert (
        refusal.value.reason
        is RefinedNamePersistenceRefusalReason.SUCCESSOR_NOT_PERSISTED
    )

    transaction.rollback.assert_called_once_with()
    transaction.commit.assert_not_called()
    retarget.assert_not_called()
    record.assert_not_called()


def test_event_write_failure_rolls_back_graph_mutation() -> None:
    admitted = AttachmentPairingGuardResult(("dd:path/a",), ())
    open_edit = {
        "edit_mode": "rename",
        "edit_reason": "preserve the requested averaging scope",
        "edit_status": "open",
    }
    with _persistence_boundary(
        ["dd:path/a"],
        admitted,
        source_edit_state=open_edit,
        effective_edit_state=open_edit,
    ) as boundary:
        transaction, _guard, _retarget, record = boundary
        record.side_effect = RuntimeError("event storage unavailable")
        with pytest.raises(RuntimeError, match="event storage"):
            _persist(
                edit_mode=None,
                edit_reason=None,
                edit_origin=None,
                edit_status=None,
                expected_old_stage=None,
            )

    transaction.rollback.assert_called_once_with()
    transaction.commit.assert_not_called()
    assert record.call_args.kwargs["operation"] == "human_edit"
    assert record.call_args.kwargs["reason"] == open_edit["edit_reason"]


def test_regular_refine_records_its_operation_once() -> None:
    admitted = AttachmentPairingGuardResult(("dd:path/a",), ())
    with _persistence_boundary(["dd:path/a"], admitted) as boundary:
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
    first = _transaction(["dd:path/a"])
    second = _transaction(["dd:path/a"])
    graphs = [_graph(first), _graph(second)]
    admitted = AttachmentPairingGuardResult(("dd:path/a",), ())
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
            return_value=1,
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
