"""Regression coverage for authoritative structural COCOS metadata."""

from typing import Any

import pytest

from imas_codex.standard_names.graph_ops import (
    _materialize_derived_parent_rows,
    _materialize_derived_parent_rows_batched,
    _ParentMaterializationCapture,
)


def _parent_row(cocos_transformation_type: str | None) -> dict[str, Any]:
    return {
        "parent_id": "electron_temperature",
        "authorized_unit": "eV",
        "edge_kinds": ["coordinate"],
        "child_data": [
            {
                "id": "electron_temperature_at_magnetic_axis",
                "unit": "eV",
                "cocos": cocos_transformation_type,
                "physics_domain": "transport",
                "op_kind": "coordinate",
            }
        ],
    }


def _assert_authoritative_cocos_query(cypher: str, value: str) -> None:
    normalized = " ".join(cypher.split())
    assert f"parent.cocos_transformation_type = {value}" in normalized
    assert f"WHEN {value} IS NULL THEN stored_cocos_edges" in normalized
    assert "OPTIONAL MATCH (parent)-[stored_cocos:HAS_COCOS]->(:COCOS)" in normalized
    assert "DELETE stored_cocos" in normalized


class _BatchCapture:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def query(self, cypher: str, **params: Any) -> list[dict[str, Any]]:
        self.calls.append({"cypher": cypher, "params": params})
        ids = [row["parent_id"] for row in params["rows"]]
        if "STRUCTURAL_CLOSURE_BATCH_MATERIALIZE" in cypher:
            return [
                {
                    "ids": ids,
                    "event_ids": [],
                    "event_links": [],
                    "event_count": 0,
                    "event_link_count": 0,
                }
            ]
        if "STRUCTURAL_CLOSURE_BATCH_PARENT_UNITS" in cypher:
            return [{"ids": ids}]
        raise AssertionError("unexpected structural materialization query")


@pytest.mark.parametrize("cocos_transformation_type", [None, "psi_like"])
def test_structural_cocos_metadata_is_authoritative(
    cocos_transformation_type: str | None,
) -> None:
    parent = _parent_row(cocos_transformation_type)

    single = _ParentMaterializationCapture()
    assert _materialize_derived_parent_rows(single, [parent]) == 1
    assert (
        single.calls[0]["params"]["cocos_transformation_type"]
        == cocos_transformation_type
    )
    _assert_authoritative_cocos_query(
        single.calls[0]["cypher"], "$cocos_transformation_type"
    )

    batched = _BatchCapture()
    assert _materialize_derived_parent_rows_batched(batched, [parent]) == 1
    assert (
        batched.calls[0]["params"]["rows"][0]["cocos_transformation_type"]
        == cocos_transformation_type
    )
    _assert_authoritative_cocos_query(
        batched.calls[0]["cypher"], "row.cocos_transformation_type"
    )
