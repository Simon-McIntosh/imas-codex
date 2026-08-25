"""Structural authority replay for already-described derived parents."""

from __future__ import annotations

from unittest.mock import patch

from imas_codex.standard_names import graph_ops, parents


class _ReplayGraph:
    def query(self, cypher: str, **params: object) -> list[dict[str, object]]:
        if "RETURN parent.id AS id" in cypher:
            assert "parent.name_stage = 'accepted'" in cypher
            assert "parent.origin = 'derived'" in cypher
            assert "parent.validation_status = 'valid'" in cypher
            assert "parent.reviewer_score_name IS NULL" in cypher
            assert "parent.description <> $placeholder" in cypher
            return [{"id": "grounded-parent"}, {"id": "ungrounded-parent"}]
        if "_structural_authority_replay_lock" in cypher:
            assert params["parent_id"] in {"grounded-parent", "ungrounded-parent"}
            return [{"locked": 1}]
        if "_structural_authority_grounding_lock" in cypher:
            assert params["child_element_ids"] == ["element:accepted-child"]
            return [{"locked": 1}]
        raise AssertionError(f"unexpected query: {cypher}")


def _snapshot(parent_id: str, child_id: str, child_stage: str) -> dict[str, object]:
    return {
        "parent_id": parent_id,
        "parent_element_id": f"element:{parent_id}",
        "name_stage": "accepted",
        "origin": "derived",
        "claim_token": None,
        "reviewed_name_at": None,
        "docs_stage": "accepted",
        "validation_status": "valid",
        "embedded_at": None,
        "chain_length": 0,
        "authority_ids": [],
        "children": [
            {
                "id": child_id,
                "element_id": f"element:{child_id}",
                "name_stage": child_stage,
                "reviewer_score_name": 0.96 if child_stage == "accepted" else 0.7,
                "reviewer_model_name": "reviewer",
            }
        ],
    }


def test_replay_names_entailing_children_and_refuses_ungrounded_parent() -> None:
    snapshots = {
        "grounded-parent": _snapshot("grounded-parent", "accepted-child", "accepted"),
        "ungrounded-parent": _snapshot(
            "ungrounded-parent", "reviewed-child", "reviewed"
        ),
    }
    persisted_records: list[dict[str, object]] = []

    def persist(
        _gc: object,
        record: dict[str, object],
        **_kwargs: object,
    ) -> bool:
        persisted_records.append(record)
        return True

    with (
        patch.object(
            graph_ops,
            "_structural_authority_snapshot",
            side_effect=lambda _gc, parent_id: snapshots[parent_id],
        ),
        patch.object(
            graph_ops,
            "_persist_structural_authority",
            side_effect=persist,
        ),
    ):
        result = parents.replay_described_parent_authorities(gc=_ReplayGraph())

    assert result["candidate_count"] == 2
    assert result["replayed_count"] == 1
    assert result["refused_count"] == 1
    assert result["refused"] == [
        {
            "parent_id": "ungrounded-parent",
            "reason": "no accepted children",
            "live_child_ids": ["reviewed-child"],
        }
    ]
    assert len(persisted_records) == 1
    record = persisted_records[0]
    assert record["accepted_name_id"] == "grounded-parent"
    assert record["child_ids"] == ["accepted-child"]
    assert result["replayed"] == [
        {
            "parent_id": "grounded-parent",
            "authority_id": record["id"],
            "child_ids": ["accepted-child"],
            "accepted_child_ids": ["accepted-child"],
        }
    ]
