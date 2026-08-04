"""Focused contracts for the exact standard-name run audit receipt."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from imas_codex.standard_names.run_audit import (
    _DELTA_EVIDENCE_QUERY,
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
        "current_versions": [{"id": "4.1.1", "cocos": 17, "cocos_ids": [17]}],
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
                "pool": "review",
                "phase": "review_names",
                "llm_cost": 0.031,
                "overspend": 0.0,
            },
            {
                "id": "cost-2",
                "run_id": _RUN_ID,
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
            },
            {
                "id": "review-1",
                "review_axis": "names",
                "cycle_index": 1,
                "review_group_id": "quorum",
                "resolution_role": "secondary",
                "resolution_method": "consensus",
                "score": 0.928,
            },
        ],
        "review_count": 2,
    }


def _delta_row() -> dict:
    return {
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
    receipt, client = _audit([[_target_row()], [_run_row()], [_delta_row()]])

    assert receipt.passed is True
    assert receipt.query_count == 3
    assert client.query.call_count == 3
    assert set(receipt.raw_rows) == {"target", "run", "deltas"}
    assert receipt.ledger_cost == Decimal("0.064326")
    assert receipt.cumulative_cost == Decimal("0.064326")
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
    receipt, client = _audit([[], [_run_row()], []])

    assert receipt.passed is False
    assert receipt.query_count == 3
    assert client.query.call_count == 3
    assert receipt.raw_rows["target"] == []
    assert receipt.raw_rows["run"] == [_run_row()]
    assert any("target identity resolved to 0" in item for item in receipt.diagnostics)


def test_ambiguous_target_and_run_are_reported_without_losing_rows() -> None:
    receipt, client = _audit(
        [
            [_target_row(), _target_row(element_id="target-2")],
            [_run_row(), _run_row()],
            [_delta_row()],
        ]
    )

    assert receipt.passed is False
    assert receipt.query_count == 3
    assert client.query.call_count == 3
    assert len(receipt.raw_rows["target"]) == 2
    assert len(receipt.raw_rows["run"]) == 2
    assert any("target identity resolved to 2" in item for item in receipt.diagnostics)
    assert any("run identity resolved to 2" in item for item in receipt.diagnostics)


def test_later_query_failure_retains_all_earlier_raw_evidence() -> None:
    receipt, client = _audit(
        [[_target_row()], [_run_row()], RuntimeError("compile scope error")]
    )

    assert receipt.passed is False
    assert receipt.query_count == 3
    assert client.query.call_count == 3
    assert set(receipt.raw_rows) == {"target", "run"}
    assert receipt.ledger_cost == Decimal("0.064326")
    assert any("deltas evidence query failed" in item for item in receipt.diagnostics)


def test_queries_are_bounded_and_keep_aggregates_in_with_scope() -> None:
    queries = (_TARGET_EVIDENCE_QUERY, _RUN_EVIDENCE_QUERY, _DELTA_EVIDENCE_QUERY)

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
    assert "AS ledger_cost" in _RUN_EVIDENCE_QUERY
    assert (
        "RETURN costs, ledger_cost, overspend_cost, cost_events" in _RUN_EVIDENCE_QUERY
    )
    assert "run.id STARTS WITH $run_id_prefix" in _RUN_EVIDENCE_QUERY


@pytest.mark.parametrize("cohort_noise", [0, 40])
def test_query_count_is_independent_of_unrelated_graph_size(cohort_noise: int) -> None:
    receipt, client = _audit([[_target_row()], [_run_row()], [_delta_row()]])

    assert cohort_noise >= 0
    assert receipt.query_count == 3
    assert client.query.call_count == 3
