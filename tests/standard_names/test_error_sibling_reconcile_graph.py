"""Disposable-graph contract for deterministic error-sibling reconciliation."""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from unittest.mock import patch

import pytest
from neo4j import GraphDatabase

from imas_codex.graph.client import GraphClient
from imas_codex.graph.profiles import resolve_neo4j
from imas_codex.standard_names import graph_ops

_ERROR_MODEL = "deterministic:dd_error_modifier"
_ORPHAN_REASON = "orphaned error sibling (parent name deleted)"


@pytest.fixture(scope="module")
def disposable_neo4j() -> Iterator[tuple[str, str]]:
    uri = os.environ.get("IMAS_CODEX_TEST_NEO4J_URI")
    if not uri:
        pytest.skip("IMAS_CODEX_TEST_NEO4J_URI is not configured")
    if os.environ.get("IMAS_CODEX_TEST_NEO4J_EPHEMERAL") != "1":
        pytest.fail("error-sibling reconciliation requires a disposable graph")
    project_uri = resolve_neo4j(auto_tunnel=False).uri
    if uri.rstrip("/") == project_uri.rstrip("/"):
        pytest.fail("error-sibling reconciliation refuses the project graph URI")
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
        graph_name="error-sibling-reconciliation",
    )
    client.query("MATCH (node) DETACH DELETE node")
    yield client
    client.query("MATCH (node) DETACH DELETE node")
    client.close()


def _seed_names(graph: GraphClient, rows: list[dict[str, object]]) -> None:
    graph.query(
        """
        UNWIND $rows AS row
        CREATE (name:StandardName)
        SET name = row
        """,
        rows=rows,
    )


def _run_reconcile(endpoint: tuple[str, str]) -> dict[str, int]:
    uri, password = endpoint

    def client_factory() -> GraphClient:
        return GraphClient(
            uri=uri,
            username="neo4j",
            password=password,
            graph_name="error-sibling-reconciliation-run",
        )

    with patch.object(graph_ops, "GraphClient", side_effect=client_factory):
        return graph_ops.reconcile_error_siblings()


def _states(graph: GraphClient) -> dict[str, dict[str, object]]:
    rows = graph.query(
        """
        MATCH (name:StandardName)
        RETURN name.id AS id,
               name.validation_status AS validation_status,
               name.quarantine_reason AS quarantine_reason
        ORDER BY name.id
        """
    )
    return {
        str(row["id"]): {
            "validation_status": row["validation_status"],
            "quarantine_reason": row["quarantine_reason"],
        }
        for row in rows
    }


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
        MATCH (start)-[relationship]->(end)
        RETURN elementId(relationship) AS element_id,
               type(relationship) AS type,
               properties(relationship) AS properties,
               elementId(start) AS start_id,
               elementId(end) AS end_id
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
def test_mixed_cohort_marks_exactly_the_recognized_parentless_siblings(
    graph: GraphClient,
    disposable_neo4j: tuple[str, str],
) -> None:
    _seed_names(
        graph,
        [
            {"id": "plasma_current", "validation_status": "valid"},
            {
                "id": "upper_uncertainty_of_plasma_current",
                "model": _ERROR_MODEL,
                "validation_status": "valid",
            },
            {
                "id": "lower_uncertainty_of_missing_temperature",
                "model": _ERROR_MODEL,
                "validation_status": "valid",
            },
            {
                "id": "uncertainty_index_of_missing_pressure",
                "model": _ERROR_MODEL,
                "validation_status": "valid",
            },
            {
                "id": "median_uncertainty_of_missing_density",
                "model": _ERROR_MODEL,
                "validation_status": "valid",
            },
            {
                "id": "upper_uncertainty_of_missing_flux",
                "model": "reviewed:language-model",
                "validation_status": "valid",
            },
            {
                "id": "lower_uncertainty_of_missing_current",
                "model": _ERROR_MODEL,
                "validation_status": "quarantined",
                "quarantine_reason": "manual quarantine remains authoritative",
            },
        ],
    )

    result = _run_reconcile(disposable_neo4j)
    states = _states(graph)

    assert result == {"stale_marked": 2}
    assert {
        name_id
        for name_id, state in states.items()
        if state["quarantine_reason"] == _ORPHAN_REASON
    } == {
        "lower_uncertainty_of_missing_temperature",
        "uncertainty_index_of_missing_pressure",
    }
    assert states["upper_uncertainty_of_plasma_current"] == {
        "validation_status": "valid",
        "quarantine_reason": None,
    }
    assert states["median_uncertainty_of_missing_density"] == {
        "validation_status": "valid",
        "quarantine_reason": None,
    }
    assert states["upper_uncertainty_of_missing_flux"] == {
        "validation_status": "valid",
        "quarantine_reason": None,
    }
    assert states["lower_uncertainty_of_missing_current"] == {
        "validation_status": "quarantined",
        "quarantine_reason": "manual quarantine remains authoritative",
    }


@pytest.mark.graph
def test_upper_uncertainty_orphan_gets_the_exact_live_reason(
    graph: GraphClient,
    disposable_neo4j: tuple[str, str],
) -> None:
    name_id = "upper_uncertainty_of_missing_plasma_current"
    _seed_names(
        graph,
        [{"id": name_id, "model": _ERROR_MODEL, "validation_status": "valid"}],
    )

    assert _run_reconcile(disposable_neo4j) == {"stale_marked": 1}
    assert _states(graph)[name_id] == {
        "validation_status": "quarantined",
        "quarantine_reason": "orphaned error sibling (parent name deleted)",
    }


@pytest.mark.graph
def test_lower_uncertainty_orphan_gets_the_exact_live_reason(
    graph: GraphClient,
    disposable_neo4j: tuple[str, str],
) -> None:
    name_id = "lower_uncertainty_of_missing_electron_temperature"
    _seed_names(
        graph,
        [{"id": name_id, "model": _ERROR_MODEL, "validation_status": "valid"}],
    )

    assert _run_reconcile(disposable_neo4j) == {"stale_marked": 1}
    assert _states(graph)[name_id] == {
        "validation_status": "quarantined",
        "quarantine_reason": "orphaned error sibling (parent name deleted)",
    }


@pytest.mark.graph
def test_any_existing_parent_identity_refuses_orphan_quarantine(
    graph: GraphClient,
    disposable_neo4j: tuple[str, str],
) -> None:
    name_id = "upper_uncertainty_of_quarantined_parent"
    _seed_names(
        graph,
        [
            {
                "id": "quarantined_parent",
                "validation_status": "quarantined",
                "quarantine_reason": "parent is retained for separate review",
            },
            {"id": name_id, "model": _ERROR_MODEL, "validation_status": "valid"},
        ],
    )

    assert _run_reconcile(disposable_neo4j) == {"stale_marked": 0}
    state = _states(graph)[name_id]
    assert state == {"validation_status": "valid", "quarantine_reason": None}
    assert state["quarantine_reason"] != _ORPHAN_REASON


@pytest.mark.graph
def test_unknown_operator_prefix_refuses_orphan_quarantine(
    graph: GraphClient,
    disposable_neo4j: tuple[str, str],
) -> None:
    name_id = "median_uncertainty_of_missing_temperature"
    _seed_names(
        graph,
        [{"id": name_id, "model": _ERROR_MODEL, "validation_status": "valid"}],
    )

    assert _run_reconcile(disposable_neo4j) == {"stale_marked": 0}
    state = _states(graph)[name_id]
    assert state == {"validation_status": "valid", "quarantine_reason": None}
    assert state["quarantine_reason"] != _ORPHAN_REASON


@pytest.mark.graph
def test_existing_quarantine_reason_is_not_rewritten(
    graph: GraphClient,
    disposable_neo4j: tuple[str, str],
) -> None:
    name_id = "lower_uncertainty_of_missing_density"
    existing_reason = "manual quarantine remains authoritative"
    _seed_names(
        graph,
        [
            {
                "id": name_id,
                "model": _ERROR_MODEL,
                "validation_status": "quarantined",
                "quarantine_reason": existing_reason,
            }
        ],
    )

    assert _run_reconcile(disposable_neo4j) == {"stale_marked": 0}
    assert _states(graph)[name_id] == {
        "validation_status": "quarantined",
        "quarantine_reason": "manual quarantine remains authoritative",
    }


@pytest.mark.graph
def test_replay_after_quarantine_is_measured_write_free(
    graph: GraphClient,
    disposable_neo4j: tuple[str, str],
) -> None:
    _seed_names(
        graph,
        [
            {
                "id": "upper_uncertainty_of_missing_pressure",
                "model": _ERROR_MODEL,
                "validation_status": "valid",
            },
            {
                "id": "lower_uncertainty_of_existing_pressure",
                "model": _ERROR_MODEL,
                "validation_status": "valid",
            },
            {"id": "existing_pressure", "validation_status": "valid"},
        ],
    )

    assert _run_reconcile(disposable_neo4j) == {"stale_marked": 1}
    before = _snapshot(graph)
    replay = _run_reconcile(disposable_neo4j)
    after = _snapshot(graph)

    assert replay == {"stale_marked": 0}
    assert after == before
