"""Specification for fail-closed graph property coverage checks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from imas_codex.graph.schema import GraphSchema
from imas_codex.standard_names import graph_ops

SCHEMA_PATH = (
    Path(__file__).resolve().parents[2]
    / "imas_codex"
    / "schemas"
    / "standard_name.yaml"
)


class _Graph:
    def __init__(self, rows: list[dict[str, int]]) -> None:
        self.rows = rows
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def query(self, query: str, **params: Any) -> list[dict[str, int]]:
        self.calls.append((query, params))
        return self.rows


def _review_axis_slot(schema: GraphSchema) -> tuple[str, str]:
    matches = [
        (label, slot_name)
        for label in schema.node_labels
        for slot_name, metadata in schema.get_all_slots(label).items()
        if metadata["type"] == "StandardNameReviewMode"
    ]
    assert len(matches) == 1, (
        "LinkML must expose exactly one property with the review-mode range; "
        f"found {matches}"
    )
    return matches[0]


def _property_coverage(*args: Any, **kwargs: Any) -> dict[str, int]:
    helper = getattr(graph_ops, "property_coverage", None)
    assert callable(helper), (
        "fail-closed property coverage is not implemented: "
        "graph_ops.property_coverage is missing"
    )
    return helper(*args, **kwargs)


def test_property_coverage_reports_candidates_and_populated_property_together() -> None:
    schema = GraphSchema(SCHEMA_PATH)
    label, property_name = _review_axis_slot(schema)
    coverage_key = f"{property_name}_covered"
    graph = _Graph([{"candidates": 7, coverage_key: 5}])

    coverage = _property_coverage(
        label=label,
        properties=(property_name,),
        schema=schema,
        gc=graph,
    )

    assert coverage == {"candidates": 7, coverage_key: 5}
    assert len(graph.calls) == 1
    query, _params = graph.calls[0]
    assert f"MATCH (node:{label})" in query
    assert "count(node) AS candidates" in query
    assert f"count(node.{property_name}) AS {coverage_key}" in query


def test_property_coverage_fails_closed_when_candidates_lack_the_property() -> None:
    schema = GraphSchema(SCHEMA_PATH)
    label, property_name = _review_axis_slot(schema)
    assert property_name in schema.get_all_slots(label), (
        "the property used by the coverage fixture must exist in LinkML"
    )
    coverage_key = f"{property_name}_covered"
    graph = _Graph([{"candidates": 7, coverage_key: 0}])

    with pytest.raises(
        RuntimeError,
        match=(
            rf"{label}\.{property_name}.*candidates=7.*"
            rf"{coverage_key}=0"
        ),
    ):
        _property_coverage(
            label=label,
            properties=(property_name,),
            schema=schema,
            gc=graph,
        )


def test_property_coverage_rejects_a_property_absent_from_linkml() -> None:
    schema = GraphSchema(SCHEMA_PATH)
    label, property_name = _review_axis_slot(schema)
    misspelled = f"{property_name}_misspelled"
    declared = schema.get_all_slots(label)
    assert declared
    assert property_name in declared
    graph = _Graph([])

    with pytest.raises(
        ValueError,
        match=rf"Unknown property '{misspelled}' on {label}",
    ):
        _property_coverage(
            label=label,
            properties=(misspelled,),
            schema=schema,
            gc=graph,
        )

    assert not graph.calls, "schema rejection must happen before Cypher execution"
