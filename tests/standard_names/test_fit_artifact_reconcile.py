"""Constraint fit bookkeeping is excluded from nameable DD quantities."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from imas_codex.standard_names.graph_ops import reconcile_constraint_fit_artifacts

_TEST_PATH_MARKER = "test_fit_reconcile"
_TEST_PATH_ROOT = f"equilibrium/time_slice/constraints/{_TEST_PATH_MARKER}"


def test_reconcile_reports_changed_rows_and_then_no_changes() -> None:
    graph = MagicMock()
    graph.query.side_effect = [[{"reclassified": 3}], [{"reclassified": 0}]]

    assert reconcile_constraint_fit_artifacts(graph) == {"nodes_reclassified": 3}
    assert reconcile_constraint_fit_artifacts(graph) == {"nodes_reclassified": 0}

    query = graph.query.call_args_list[0].args[0]
    assert "node_category: 'quantity'" in query
    assert "node.is_leaf = true" in query
    assert "'/constraints/'" in query
    assert "last(split(node.id, '/')) = 'weight'" in query
    assert "segment ENDS WITH '_reconstructed'" in query
    assert "SET node.node_category = 'fit_artifact'" in query


@pytest.fixture()
def graph_client():
    try:
        from imas_codex.graph.client import GraphClient

        client = GraphClient()
        client.get_stats()
    except Exception as exc:  # pragma: no cover - environment-dependent
        pytest.skip(f"Neo4j not available: {exc}")

    client.query(
        "MATCH (node:IMASNode) WHERE node.id CONTAINS $marker DETACH DELETE node",
        marker=_TEST_PATH_MARKER,
    )
    yield client
    client.query(
        "MATCH (node:IMASNode) WHERE node.id CONTAINS $marker DETACH DELETE node",
        marker=_TEST_PATH_MARKER,
    )
    client.close()


@pytest.mark.graph
def test_reconcile_changes_only_constraint_fit_leaves(graph_client) -> None:
    targets = [
        f"{_TEST_PATH_ROOT}/pressure/weight",
        f"{_TEST_PATH_ROOT}/pressure/reconstructed",
        f"{_TEST_PATH_ROOT}/x_point/position_reconstructed/r",
    ]
    controls = [
        f"{_TEST_PATH_ROOT}/pressure/measured",
        f"{_TEST_PATH_ROOT}/pressure/weight_container",
        "equilibrium/time_slice/test_fit_reconcile/reconstructed",
    ]
    graph_client.query(
        """
        UNWIND $targets AS node_id
        CREATE (:IMASNode {
          id: node_id, node_category: 'quantity', is_leaf: true
        })
        """,
        targets=targets + controls,
    )

    first = reconcile_constraint_fit_artifacts(graph_client)
    second = reconcile_constraint_fit_artifacts(graph_client)
    rows = graph_client.query(
        """
        MATCH (node:IMASNode)
        WHERE node.id IN $ids
        RETURN node.id AS id, node.node_category AS category
        ORDER BY id
        """,
        ids=targets + controls,
    )

    categories = {row["id"]: row["category"] for row in rows}
    assert first["nodes_reclassified"] >= len(targets)
    assert second == {"nodes_reclassified": 0}
    assert {categories[node_id] for node_id in targets} == {"fit_artifact"}
    assert {categories[node_id] for node_id in controls} == {"quantity"}
