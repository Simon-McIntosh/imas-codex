"""Atomic persistence and replay behavior for structural name authorities."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import imas_codex.standard_names.graph_ops as graph_ops


def _snapshot(parent_id: str, *child_ids: str) -> dict:
    return {
        "parent_id": parent_id,
        "parent_element_id": "parent-element",
        "name_stage": "accepted",
        "origin": "derived",
        "claim_token": None,
        "reviewed_name_at": None,
        "docs_stage": "pending",
        "validation_status": "valid",
        "embedded_at": None,
        "chain_length": 0,
        "authority_ids": [],
        "children": [
            {
                "id": child_id,
                "element_id": f"child-element-{index}",
                "name_stage": "accepted",
                "reviewer_score_name": 0.95,
                "reviewer_model_name": "reviewer",
            }
            for index, child_id in enumerate(child_ids)
        ],
    }


def test_authority_signature_covers_exact_ordered_child_rows() -> None:
    first = graph_ops._structural_authority_record(
        _snapshot("parent", "child-a", "child-b"), accepting=False
    )
    changed = graph_ops._structural_authority_record(
        _snapshot("parent", "child-a", "child-c"), accepting=False
    )

    assert first["child_ids"] == ["child-a", "child-b"]
    assert first["signature"]["sha256"] == first["payload_sha256"]
    assert first["payload_sha256"] != changed["payload_sha256"]
    assert first["id"] != changed["id"]


def test_accept_and_authority_are_one_guarded_graph_statement() -> None:
    record = graph_ops._structural_authority_record(
        _snapshot("parent", "child-a", "child-b"), accepting=True
    )
    gc = MagicMock()
    gc.query.return_value = [{"name_stage": "accepted", "authority_id": record["id"]}]

    assert graph_ops._persist_structural_authority(
        gc,
        record,
        parent_updates={"name_stage": "accepted"},
    )

    cypher = gc.query.call_args.args[0]
    params = gc.query.call_args.kwargs
    assert "SET parent += $parent_updates" in cypher
    assert "CREATE (parent)-[:HAS_STRUCTURAL_AUTHORITY]->(authority)" in cypher
    assert "CREATE (authority)-[:ENTAILED_FROM_CHILD]->(entailing_child)" in cypher
    assert "= $child_ids" in cypher
    assert "= $child_element_ids" in cypher
    assert params["child_rows"] == record["child_rows"]
    assert params["child_element_ids"] == [
        child["element_id"] for child in record["child_rows"]
    ]
    assert params["parent_updates"] == {"name_stage": "accepted"}
    assert "reviewer_score_name" not in params["parent_updates"]


def test_backfill_names_every_write_and_second_run_writes_nothing(
    tmp_path: Path,
) -> None:
    client = MagicMock()
    client.query.side_effect = [
        [{"id": "childful-parent"}, {"id": "childless-parent"}],
        [],
    ]
    snapshots = {
        "childful-parent": _snapshot("childful-parent", "child-a", "child-b"),
        "childless-parent": _snapshot("childless-parent"),
    }
    receipt_path = tmp_path / "receipt.json"

    with (
        patch.object(
            graph_ops,
            "_structural_authority_snapshot",
            side_effect=lambda _gc, sn_id: snapshots[sn_id],
        ),
        patch.object(
            graph_ops, "_persist_structural_authority", return_value=True
        ) as persist,
    ):
        first = graph_ops.backfill_structural_name_authorities(
            receipt_path=receipt_path, gc=client
        )
        second = graph_ops.backfill_structural_name_authorities(gc=client)

    assert first["lacking_authority_before"] == 2
    assert first["eligible_before"] == 1
    assert first["written_count"] == 1
    assert [row["id"] for row in first["written"]] == ["childful-parent"]
    assert first["skipped"] == [
        {"id": "childless-parent", "reason": "no live children"}
    ]
    assert second["lacking_authority_before"] == 0
    assert second["written_count"] == 0
    assert persist.call_count == 1
    assert json.loads(receipt_path.read_text())["written"][0]["id"] == (
        "childful-parent"
    )
