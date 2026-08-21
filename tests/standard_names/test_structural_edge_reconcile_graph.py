"""Disposable-graph contract for exact structural-edge reconciliation."""

from __future__ import annotations

import json
import os
from collections.abc import Iterator

import pytest
from neo4j import GraphDatabase

from imas_codex.graph.client import GraphClient
from imas_codex.graph.profiles import resolve_neo4j
from imas_codex.standard_names.graph_ops import (
    reconcile_structural_edges_for_standard_names,
)


@pytest.fixture(scope="module")
def disposable_neo4j() -> Iterator[tuple[str, str]]:
    uri = os.environ.get("IMAS_CODEX_TEST_NEO4J_URI")
    if not uri:
        pytest.skip("IMAS_CODEX_TEST_NEO4J_URI is not configured")
    if os.environ.get("IMAS_CODEX_TEST_NEO4J_EPHEMERAL") != "1":
        pytest.fail("structural-edge reconciliation requires a disposable graph")
    project_uri = resolve_neo4j(auto_tunnel=False).uri
    if uri.rstrip("/") == project_uri.rstrip("/"):
        pytest.fail("structural-edge reconciliation refuses the project graph URI")
    password = os.environ.get("IMAS_CODEX_TEST_NEO4J_PASSWORD", "")
    auth = ("neo4j", password) if password else None
    with GraphDatabase.driver(uri, auth=auth) as driver:
        driver.verify_connectivity()
        driver.execute_query("MATCH (node) DETACH DELETE node")
    yield uri, password


@pytest.fixture
def graph(disposable_neo4j: tuple[str, str]) -> Iterator[GraphClient]:
    uri, password = disposable_neo4j
    client = GraphClient(
        uri=uri,
        username="neo4j",
        password=password,
        graph_name="structural-edge-reconciliation",
    )
    client.query("MATCH (node) DETACH DELETE node")
    yield client
    client.query("MATCH (node) DETACH DELETE node")
    client.close()


def _seed_names(graph: GraphClient, *name_ids: str) -> None:
    graph.query(
        """
        UNWIND $ids AS id
        CREATE (:StandardName {id: id, name_stage: 'accepted'})
        """,
        ids=list(name_ids),
    )


def _relationships(graph: GraphClient) -> list[dict[str, object]]:
    return graph.query(
        """
        MATCH (source)-[relationship]->(target)
        RETURN source.id AS source,
               type(relationship) AS type,
               target.id AS target,
               properties(relationship) AS properties
        ORDER BY source, type, target
        """
    )


def _snapshot(graph: GraphClient) -> bytes:
    nodes = graph.query(
        """
        MATCH (node)
        RETURN elementId(node) AS element_id,
               labels(node) AS labels,
               properties(node) AS properties
        ORDER BY element_id
        """
    )
    relationships = graph.query(
        """
        MATCH (source)-[relationship]->(target)
        RETURN elementId(relationship) AS element_id,
               type(relationship) AS type,
               properties(relationship) AS properties,
               elementId(source) AS source_id,
               elementId(target) AS target_id
        ORDER BY element_id
        """
    )
    return json.dumps(
        {"nodes": nodes, "relationships": relationships},
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode()


@pytest.mark.graph
def test_mixed_cohort_writes_only_the_current_admitted_partition(
    graph: GraphClient,
) -> None:
    parent_child = "maximum_of_electron_temperature"
    error_sibling = "upper_uncertainty_of_electron_temperature"
    locus_name = "major_radius_of_magnetic_axis"
    unrelated_child = "minimum_of_electron_temperature"
    _seed_names(
        graph,
        parent_child,
        "electron_temperature",
        error_sibling,
        locus_name,
        unrelated_child,
        "ion_temperature",
    )
    graph.query(
        """
        MATCH (requested:StandardName {id: $requested})
        MATCH (wrong:StandardName {id: 'ion_temperature'})
        CREATE (requested)-[:HAS_PARENT {
            operator: 'stale', operator_kind: 'unary_prefix'
        }]->(wrong)
        WITH wrong
        MATCH (unrelated:StandardName {id: $unrelated})
        CREATE (unrelated)-[:HAS_PARENT {
            operator: 'stale', operator_kind: 'unary_prefix'
        }]->(wrong)
        """,
        requested=parent_child,
        unrelated=unrelated_child,
    )

    changed = reconcile_structural_edges_for_standard_names(
        graph,
        [parent_child, error_sibling, locus_name],
    )

    relationships = _relationships(graph)
    assert changed == 3
    assert {(row["source"], row["type"], row["target"]) for row in relationships} == {
        (parent_child, "HAS_PARENT", "electron_temperature"),
        ("electron_temperature", "HAS_ERROR", error_sibling),
        (locus_name, "HAS_LOCUS", "magnetic_axis"),
        (unrelated_child, "HAS_PARENT", "ion_temperature"),
    }
    properties = {
        (row["source"], row["type"], row["target"]): row["properties"]
        for row in relationships
    }
    assert properties[(parent_child, "HAS_PARENT", "electron_temperature")] == {
        "operator": "maximum",
        "operator_kind": "unary_prefix",
    }
    assert properties[("electron_temperature", "HAS_ERROR", error_sibling)] == {
        "error_type": "upper"
    }
    assert properties[(locus_name, "HAS_LOCUS", "magnetic_axis")] == {
        "locus_relation": "of",
        "locus_token": "magnetic_axis",
    }


@pytest.mark.graph
def test_duplicate_requested_ids_are_reconciled_once(graph: GraphClient) -> None:
    child = "maximum_of_electron_temperature"
    _seed_names(graph, child, "electron_temperature")

    changed = reconcile_structural_edges_for_standard_names(
        graph,
        [child, child, child],
    )

    assert changed == 1
    assert _relationships(graph) == [
        {
            "source": child,
            "type": "HAS_PARENT",
            "target": "electron_temperature",
            "properties": {
                "operator": "maximum",
                "operator_kind": "unary_prefix",
            },
        }
    ]


@pytest.mark.graph
def test_empty_request_is_an_admitted_no_op(graph: GraphClient) -> None:
    _seed_names(graph, "electron_temperature")
    before = _snapshot(graph)

    assert reconcile_structural_edges_for_standard_names(graph, []) == 0
    assert _snapshot(graph) == before


@pytest.mark.graph
def test_empty_identity_refuses_with_the_exact_live_reason(graph: GraphClient) -> None:
    _seed_names(graph, "electron_temperature")
    before = _snapshot(graph)

    with pytest.raises(ValueError) as raised:
        reconcile_structural_edges_for_standard_names(
            graph,
            ["electron_temperature", ""],
        )

    assert str(raised.value) == (
        "name_ids must contain only non-empty StandardName ids"
    )
    assert _snapshot(graph) == before


@pytest.mark.graph
def test_missing_identities_refuse_with_the_exact_sorted_live_reason(
    graph: GraphClient,
) -> None:
    _seed_names(graph, "electron_temperature")
    before = _snapshot(graph)

    with pytest.raises(ValueError) as raised:
        reconcile_structural_edges_for_standard_names(
            graph,
            ["missing_zeta", "electron_temperature", "missing_alpha"],
        )

    assert str(raised.value) == (
        "cannot reconcile structural edges for missing StandardName ids: "
        "'missing_alpha', 'missing_zeta'"
    )
    assert _snapshot(graph) == before


@pytest.mark.graph
def test_exact_replay_is_state_write_free(graph: GraphClient) -> None:
    child = "maximum_of_electron_temperature"
    error_sibling = "upper_uncertainty_of_electron_temperature"
    _seed_names(graph, child, "electron_temperature", error_sibling)

    assert (
        reconcile_structural_edges_for_standard_names(
            graph,
            [child, error_sibling],
        )
        == 2
    )
    before = _snapshot(graph)

    replayed = reconcile_structural_edges_for_standard_names(
        graph,
        [child, error_sibling],
    )

    assert replayed == 2
    assert _snapshot(graph) == before
