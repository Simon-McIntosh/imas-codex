"""Safe cancellation of an unaccepted rename proposal."""

from __future__ import annotations

from typing import Any

import pytest

from imas_codex.standard_names.provenance_lifecycle import cancel_staged_rename


class _Graph:
    def __init__(
        self,
        *,
        eligible: bool = True,
        predecessor_stage: str = "accepted",
    ) -> None:
        self.eligible = eligible
        self.predecessor_stage = predecessor_stage
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def query(self, cypher: str, **params: Any) -> list[dict[str, Any]]:
        self.calls.append((cypher, params))
        if len(self.calls) == 1:
            if not self.eligible:
                return []
            return [
                {
                    "successor": "new_name",
                    "successor_stage": "drafted",
                    "predecessor": "old_name",
                    "predecessor_stage": self.predecessor_stage,
                }
            ]
        return [
            {
                "predecessor": "old_name",
                "restored_stage": "accepted",
                "sources_restored": 2,
                "owners_restored": 1,
            }
        ]


def test_dry_run_checks_exact_open_rename_without_writing() -> None:
    graph = _Graph()

    result = cancel_staged_rename(
        graph,
        "new_name",
        reason="the successor changes the quantity semantics",
        dry_run=True,
    )

    assert result["ok"] is True
    assert result["predecessor"] == "old_name"
    assert len(graph.calls) == 1
    eligibility = graph.calls[0][0]
    assert "successor.edit_mode = 'rename'" in eligibility
    assert "successor.edit_status = 'open'" in eligibility
    assert "successor.name_stage IN ['drafted', 'reviewed', 'exhausted']" in eligibility
    assert "predecessor.reviewer_score_name >= $min_score" in eligibility
    assert graph.calls[0][1]["min_score"] == 0.85


def test_apply_restores_edges_and_records_deletion_atomically() -> None:
    graph = _Graph()

    result = cancel_staged_rename(
        graph,
        "new_name",
        reason="the successor changes the quantity semantics",
    )

    assert result == {
        "ok": True,
        "successor": "new_name",
        "predecessor": "old_name",
        "restored_stage": "accepted",
        "sources_restored": 2,
        "owners_restored": 1,
        "dry_run": False,
    }
    mutation, params = graph.calls[1]
    assert "operation: $deletion_operation" in mutation
    assert "MERGE (source)-[:PRODUCED_NAME]->(predecessor)" in mutation
    assert "MERGE (owner)-[:HAS_STANDARD_NAME]->(predecessor)" in mutation
    assert "DETACH DELETE successor" in mutation
    assert params["deletion_operation"] == "cancel_staged_rename"
    assert "predecessor.reviewer_score_name >= $min_score" in mutation


def test_dry_run_falls_back_to_reviewed_for_corrupt_transient_prior_stage() -> None:
    graph = _Graph(predecessor_stage="reviewed")

    result = cancel_staged_rename(
        graph,
        "new_name",
        reason="the successor changes the quantity semantics",
        dry_run=True,
    )

    assert result["predecessor_stage"] == "reviewed"


def test_non_open_or_accepted_successor_is_refused() -> None:
    graph = _Graph(eligible=False)

    result = cancel_staged_rename(
        graph,
        "new_name",
        reason="the successor changes the quantity semantics",
    )

    assert result["ok"] is False
    assert len(graph.calls) == 1


def test_reason_is_required() -> None:
    with pytest.raises(ValueError, match="non-empty reason"):
        cancel_staged_rename(_Graph(), "new_name", reason=" ")
