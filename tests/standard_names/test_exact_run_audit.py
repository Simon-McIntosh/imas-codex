"""Focused contracts for the exact standard-name run audit receipt."""

from __future__ import annotations

from copy import deepcopy
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from imas_codex.standard_names.run_audit import (
    _DD_EVIDENCE_QUERY,
    _RUN_EVIDENCE_QUERY,
    _TARGET_EVIDENCE_QUERY,
    audit_exact_standard_name_run,
)

_NAME_ID = "toroidal_offset_at_measurement_position"
_SCOPE = "4adf5428-8bee-4ce9-bc1d-637f84a9bed5"
_RUN_ID = "1e7be8c0-9c56-4fca-b649-182489d7f531"
_LAUNCH = "2026-08-04T00:00:00+00:00"
_COMPLETE = "2026-08-04T00:05:00+00:00"


def _target_row(*, element_id: str = "target-1") -> dict:
    return {
        "target": {
            "element_id": element_id,
            "id": _NAME_ID,
            "run_id": _SCOPE,
            "last_run_id": _RUN_ID,
            "name_stage": "accepted",
            "docs_stage": "pending",
            "status": "draft",
            "validation_status": "valid",
            "reviewer_score_name": 0.919,
            "reviewer_score_docs": None,
            "origin": "pipeline",
            "edit_status": "applied",
        },
        "sources": [
            {
                "source_id": "dd:magnetics/measurement/toroidal_offset",
                "source_type": "dd",
                "source_status": "attached",
                "produced_sn_id": _NAME_ID,
                "dd_snapshot": "4.1.1",
                "dd_snapshot_pinned": True,
                "dd_unit": "rad",
                "backing_id": "magnetics/measurement/toroidal_offset",
                "backing_unit": "rad",
                "backing_unit_ids": ["rad"],
                "projection_ids": ["projection-1"],
                "per_path_cocos_label": None,
                "west": False,
                "fixture": False,
            }
        ],
        "target_units": ["rad"],
    }


def _dd_row() -> dict:
    return {
        "version": {
            "id": "4.1.1",
            "is_current": True,
            "cocos": 17,
            "cocos_ids": [17],
        }
    }


def _run_row() -> dict:
    return {
        "run": {
            "id": _RUN_ID,
            "status": "completed",
            "stop_reason": "completed",
            "started_at": _LAUNCH,
            "stopped_at": _COMPLETE,
            "ended_at": _COMPLETE,
            "cost_spent": 0.064326,
            "cost_limit": 2.0,
            "cost_total": 0.064326,
            "cost_is_exact": True,
            "events_total": 2,
        },
        "costs": [
            {
                "id": "cost-1",
                "run_id": _RUN_ID,
                "linked_run_id": _RUN_ID,
                "pool": "review",
                "phase": "review_names",
                "llm_cost": 0.031,
                "overspend": 0.0,
            },
            {
                "id": "cost-2",
                "run_id": _RUN_ID,
                "linked_run_id": _RUN_ID,
                "pool": "review",
                "phase": "review_names",
                "llm_cost": 0.033326,
                "overspend": 0.0,
            },
        ],
        "ledger_cost": 0.064326,
        "overspend_cost": 0.0,
        "cost_events": 2,
        "reviews": [
            {
                "id": "review-0",
                "review_axis": "names",
                "cycle_index": 0,
                "review_group_id": "quorum",
                "resolution_role": "primary",
                "resolution_method": None,
                "score": 0.91,
                "linked_run_id": _RUN_ID,
                "linked_cost_ids": ["cost-1"],
            },
            {
                "id": "review-1",
                "review_axis": "names",
                "cycle_index": 1,
                "review_group_id": "quorum",
                "resolution_role": "secondary",
                "resolution_method": "consensus",
                "score": 0.928,
                "linked_run_id": _RUN_ID,
                "linked_cost_ids": ["cost-2"],
            },
        ],
        "review_count": 2,
        "predecessor_ids": ["toroidal_offset_at_measurement_location"],
        "refined_successor_ids": [],
        "docs_revision_ids": [],
        "internal_change_ids": [],
    }


def _audit(query_results: list[object]):
    client = MagicMock()
    client.query.side_effect = query_results
    with patch(
        "imas_codex.standard_names.grammar_segment_reconciliation._west_source_ids",
        return_value=frozenset(),
    ):
        receipt = audit_exact_standard_name_run(
            _NAME_ID,
            _SCOPE,
            _RUN_ID[:12],
            _LAUNCH,
            _COMPLETE,
            gc=client,
        )
    return receipt, client


def test_complete_receipt_preserves_evidence_and_exact_cost() -> None:
    receipt, client = _audit([[_target_row()], [_dd_row()], [_run_row()]])

    assert receipt.passed is True
    assert receipt.query_count == 3
    assert client.query.call_count == 3
    assert set(receipt.raw_rows) == {"target", "dd", "run"}
    assert receipt.ledger_cost == Decimal("0.064326")
    assert receipt.cumulative_cost == Decimal("0.064326")
    assert receipt.cost_total == Decimal("0.064326")
    assert receipt.cost_is_exact is True
    assert receipt.events_total == receipt.cost_event_count == 2
    assert receipt.target_scope_run_id == _SCOPE
    assert receipt.run_id == _RUN_ID
    assert receipt.pool_counts == {"review": 2}
    assert receipt.review_count == 2
    assert receipt.review_cycles == [0, 1]
    assert receipt.review_resolutions == ["consensus"]
    assert receipt.current_dd_versions == ["4.1.1"]
    assert receipt.global_cocos == [17]
    assert receipt.source_count == 1
    assert receipt.backing_count == 1
    assert receipt.projection_count == 1
    assert receipt.per_path_cocos_labels == {
        "magnetics/measurement/toroidal_offset": None
    }
    assert receipt.target_score_name == Decimal("0.919")
    assert receipt.predecessor_ids == ["toroidal_offset_at_measurement_location"]
    assert receipt.predecessor_count == 1
    assert receipt.refined_successor_count == 0
    assert receipt.docs_revision_count == 0
    assert receipt.new_name_count == 0


def test_missing_target_returns_failed_receipt_after_constant_queries() -> None:
    receipt, client = _audit([[], [_run_row()]])

    assert receipt.passed is False
    assert receipt.query_count == 2
    assert client.query.call_count == 2
    assert receipt.raw_rows["target"] == []
    assert receipt.raw_rows["run"] == [_run_row()]
    assert any("target identity resolved to 0" in item for item in receipt.diagnostics)


def test_ambiguous_target_and_run_are_reported_without_losing_rows() -> None:
    receipt, client = _audit(
        [
            [_target_row(), _target_row(element_id="target-2")],
            [_dd_row()],
            [_run_row(), _run_row()],
        ]
    )

    assert receipt.passed is False
    assert receipt.query_count == 3
    assert client.query.call_count == 3
    assert len(receipt.raw_rows["target"]) == 2
    assert len(receipt.raw_rows["run"]) == 2
    assert any("target identity resolved to 2" in item for item in receipt.diagnostics)
    assert any("run identity resolved to 2" in item for item in receipt.diagnostics)


def test_wrong_exact_scope_fails_without_conflating_the_sn_run() -> None:
    target = _target_row()
    target["target"]["run_id"] = "wrong-scope"

    receipt, _client = _audit([[target], [_dd_row()], [_run_row()]])

    assert receipt.passed is False
    assert receipt.target_scope_run_id == "wrong-scope"
    assert receipt.run_id == _RUN_ID
    assert any("does not match exact scope" in item for item in receipt.diagnostics)


def test_ambiguous_run_prefix_fails_with_both_raw_candidates_retained() -> None:
    receipt, _client = _audit([[_target_row()], [_dd_row()], [_run_row(), _run_row()]])

    assert receipt.passed is False
    assert len(receipt.raw_rows["run"]) == 2
    assert any("run identity resolved to 2" in item for item in receipt.diagnostics)


def test_unrelated_run_evidence_is_rejected_even_when_mock_returns_it() -> None:
    run_row = _run_row()
    run_row["run"]["id"] = "unrelated-run"

    receipt, _client = _audit([[_target_row()], [_dd_row()], [run_row]])

    assert receipt.passed is False
    assert any(
        "does not match supplied identity" in item for item in receipt.diagnostics
    )


@pytest.mark.parametrize("started_at", [None, "not-a-timestamp"])
def test_invalid_or_missing_run_start_time_fails_closed(started_at: object) -> None:
    run_row = _run_row()
    run_row["run"]["started_at"] = started_at

    receipt, client = _audit([[_target_row()], [_dd_row()], [run_row]])

    assert receipt.passed is False
    assert receipt.query_count == 3
    assert client.query.call_count == 3
    assert receipt.raw_rows["run"] == [run_row]
    assert any("invalid start timestamp" in item for item in receipt.diagnostics)


@pytest.mark.parametrize("evidence_kind", ["cost", "review"])
def test_unrelated_child_evidence_is_rejected(evidence_kind: str) -> None:
    run_row = _run_row()
    if evidence_kind == "cost":
        run_row["costs"][0]["linked_run_id"] = "unrelated-run"
    else:
        run_row["reviews"][0]["linked_cost_ids"] = ["unrelated-cost"]

    receipt, _client = _audit([[_target_row()], [_dd_row()], [run_row]])

    assert receipt.passed is False
    assert any(evidence_kind in item for item in receipt.diagnostics)


def test_review_linkage_reuses_bounded_costs_and_keeps_query_count_constant() -> None:
    run_row = _run_row()
    run_row["reviews"][0]["linked_cost_ids"] = []

    receipt, client = _audit([[_target_row()], [_dd_row()], [run_row]])

    assert receipt.passed is False
    assert receipt.query_count == 3
    assert client.query.call_count == 3
    assert any(
        "review evidence is not linked to the selected run" in item
        for item in receipt.diagnostics
    )
    assert _RUN_EVIDENCE_QUERY.count("[:FOR_RUN]->(run)") == 1
    assert "CALL (run, costs)" in _RUN_EVIDENCE_QUERY
    assert "[cost IN costs WHERE cost.run_id = run.id" in _RUN_EVIDENCE_QUERY


@pytest.mark.parametrize(
    ("field", "value", "diagnostic"),
    [
        ("cost_is_exact", None, "inexact cost"),
        ("cost_is_exact", False, "inexact cost"),
        ("cost_total", None, "no exact cost total"),
        ("cost_total", 0.064325, "differs from exact ledger sum"),
        ("events_total", None, "no event total"),
        ("events_total", 3, "event total differs"),
    ],
)
def test_missing_or_inconsistent_run_cost_fields_fail_closed(
    field: str, value: object, diagnostic: str
) -> None:
    run_row = _run_row()
    run_row["run"][field] = value

    receipt, _client = _audit([[_target_row()], [_dd_row()], [run_row]])

    assert receipt.passed is False
    assert receipt.raw_rows["run"] == [run_row]
    assert any(diagnostic in item for item in receipt.diagnostics)


@pytest.mark.parametrize("mismatch", ["aggregate", "cardinality", "duplicate"])
def test_cost_event_cardinality_mismatches_fail_closed(mismatch: str) -> None:
    run_row = _run_row()
    if mismatch == "aggregate":
        run_row["cost_events"] = 1
    elif mismatch == "cardinality":
        run_row["costs"].pop()
    else:
        run_row["costs"][1]["id"] = "cost-1"

    receipt, _client = _audit([[_target_row()], [_dd_row()], [run_row]])

    assert receipt.passed is False
    assert any("event total differs" in item for item in receipt.diagnostics)


def test_disagreeing_source_snapshots_skip_the_dd_query_and_fail_closed() -> None:
    target = _target_row()
    second_source = deepcopy(target["sources"][0])
    second_source["source_id"] = "dd:magnetics/measurement/other_offset"
    second_source["backing_id"] = "magnetics/measurement/other_offset"
    second_source["dd_snapshot"] = "4.1.0"
    target["sources"].append(second_source)

    receipt, client = _audit([[target], [_run_row()]])

    assert receipt.passed is False
    assert receipt.query_count == 2
    assert set(receipt.raw_rows) == {"target", "run"}
    assert (
        "EXACT_STANDARD_NAME_DD_EVIDENCE" not in client.query.call_args_list[1].args[0]
    )
    assert any(
        "snapshot identity is missing or ambiguous" in item
        for item in receipt.diagnostics
    )


def test_later_query_failure_retains_all_earlier_raw_evidence() -> None:
    receipt, client = _audit(
        [[_target_row()], [_dd_row()], RuntimeError("compile scope error")]
    )

    assert receipt.passed is False
    assert receipt.query_count == 3
    assert client.query.call_count == 3
    assert set(receipt.raw_rows) == {"target", "dd"}
    assert any("run evidence query failed" in item for item in receipt.diagnostics)


def test_queries_are_bounded_and_keep_aggregates_in_with_scope() -> None:
    queries = (_TARGET_EVIDENCE_QUERY, _DD_EVIDENCE_QUERY, _RUN_EVIDENCE_QUERY)

    for query in queries:
        assert "*0.." not in query
        assert "*1.." not in query
        assert "AllNodesScan" not in query
        assert "AllRelationshipsScan" not in query
        assert "MATCH (n)" not in query
        assert "MATCH ()-[" not in query
    assert "MATCH (target:StandardName {id: $name_id})" in _TARGET_EVIDENCE_QUERY
    assert "WITH target, source, backing" in _TARGET_EVIDENCE_QUERY
    assert "WITH source, backing, backing_unit_ids" in _TARGET_EVIDENCE_QUERY
    assert "DDVersion {is_current: true}" not in _TARGET_EVIDENCE_QUERY
    assert "MATCH (version:DDVersion {id: $dd_version})" in _DD_EVIDENCE_QUERY
    assert "WITH run, cost" in _RUN_EVIDENCE_QUERY
    assert _RUN_EVIDENCE_QUERY.count("[:FOR_RUN]->(run)") == 1
    assert "CALL (run, costs)" in _RUN_EVIDENCE_QUERY
    assert "AS ledger_cost" in _RUN_EVIDENCE_QUERY
    assert (
        "RETURN costs, ledger_cost, overspend_cost, cost_events" in _RUN_EVIDENCE_QUERY
    )
    assert "run.id STARTS WITH $run_id_prefix" in _RUN_EVIDENCE_QUERY
    for temporal_property in (
        "run.started_at",
        "cost.llm_at",
        "review.reviewed_at",
        "review.llm_at",
        "revision.created_at",
        "change.changed_at",
    ):
        assert f"datetime(toString({temporal_property}))" in _RUN_EVIDENCE_QUERY
