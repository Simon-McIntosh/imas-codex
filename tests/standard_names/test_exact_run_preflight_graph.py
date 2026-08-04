"""Disposable-Neo4j receipt and access-plan checks for exact preflight."""

from __future__ import annotations

import os
import uuid
from typing import Any

import pytest

from imas_codex.standard_names.run_preflight import (
    _EXACT_STANDARD_NAME_PREFLIGHT_QUERY,
)

pytestmark = pytest.mark.graph


def _plan_nodes(plan: object) -> list[tuple[str, dict[str, Any]]]:
    if plan is None:
        return []
    if isinstance(plan, dict):
        operator = plan.get("operatorType") or plan.get("operator_type")
        arguments = plan.get("args") or plan.get("arguments") or {}
        children = plan.get("children", [])
    else:
        operator = getattr(plan, "operator_type", None)
        arguments = getattr(plan, "arguments", {})
        children = getattr(plan, "children", [])
    nodes = [(str(operator).partition("@")[0], dict(arguments))] if operator else []
    for child in children:
        nodes.extend(_plan_nodes(child))
    return nodes


def _params(name_id: str, dd_version: str) -> dict[str, object]:
    return {
        "name_id": name_id,
        "dd_version": dd_version,
        "min_score": 0.85,
        "rotation_cap": 3,
        "west_source_ids": [],
        "fixture_source_id_prefix": "dd:test_review_entry__",
    }


def test_disposable_graph_preserves_missing_receipt_and_uses_exact_seeks() -> None:
    """The real query retains one row and never falls back to a global scan."""
    from imas_codex.graph.client import GraphClient

    uri = os.environ.get("IMAS_CODEX_TEST_NEO4J_URI")
    if not uri:
        pytest.skip("IMAS_CODEX_TEST_NEO4J_URI is not configured")
    if os.environ.get("IMAS_CODEX_TEST_NEO4J_EPHEMERAL") != "1":
        pytest.fail("exact preflight graph test requires an ephemeral graph")

    suffix = uuid.uuid4().hex
    missing_name = f"exact_preflight_missing_{suffix}"
    ordinary_name = f"exact_preflight_ordinary_{suffix}"
    dd_version = f"4.1.1-exact-preflight-{suffix}"
    password = os.environ.get(
        "IMAS_CODEX_TEST_NEO4J_PASSWORD", os.environ.get("NEO4J_PASSWORD", "")
    )
    with GraphClient(
        uri=uri,
        username=os.environ.get("NEO4J_USERNAME", "neo4j"),
        password=password,
        graph_name="ephemeral-exact-refinement-preflight",
    ) as gc:
        with gc.session() as session:
            for statement in (
                "CREATE CONSTRAINT exact_preflight_standard_name_id IF NOT EXISTS "
                "FOR (node:StandardName) REQUIRE node.id IS UNIQUE",
                "CREATE CONSTRAINT exact_preflight_dd_version_id IF NOT EXISTS "
                "FOR (node:DDVersion) REQUIRE node.id IS UNIQUE",
            ):
                session.run(statement).consume()
            session.run(
                "MERGE (:DDVersion {id: $dd_version})",
                dd_version=dd_version,
            ).consume()

            missing_rows = list(
                session.run(
                    _EXACT_STANDARD_NAME_PREFLIGHT_QUERY,
                    **_params(missing_name, dd_version),
                )
            )
            session.run(
                "MERGE (:StandardName {id: $name_id})",
                name_id=ordinary_name,
            ).consume()
            ordinary_rows = list(
                session.run(
                    _EXACT_STANDARD_NAME_PREFLIGHT_QUERY,
                    **_params(ordinary_name, dd_version),
                )
            )
            explained = session.run(
                "EXPLAIN " + _EXACT_STANDARD_NAME_PREFLIGHT_QUERY,
                **_params(ordinary_name, dd_version),
            ).consume()

    assert len(missing_rows) == 1
    assert dict(missing_rows[0])["targets"] == []
    assert len(ordinary_rows) == 1
    assert [target["id"] for target in dict(ordinary_rows[0])["targets"]] == [
        ordinary_name
    ]

    plan_nodes = _plan_nodes(explained.plan)
    operators = [operator for operator, _arguments in plan_nodes]
    forbidden_scans = {
        "AllNodesScan",
        "NodeByLabelScan",
        "DirectedRelationshipTypeScan",
        "UndirectedRelationshipTypeScan",
        "UnionRelationshipTypesScan",
    }
    assert forbidden_scans.isdisjoint(operators)

    seek_details = [
        " ".join(str(value) for value in arguments.values())
        for operator, arguments in plan_nodes
        if operator == "NodeUniqueIndexSeek"
    ]
    assert any(
        "StandardName" in details and "id" in details for details in seek_details
    )
    assert any("DDVersion" in details and "id" in details for details in seek_details)

    variable_expansions = [
        operator for operator in operators if "VarLengthExpand" in operator
    ]
    assert variable_expansions
    variable_patterns = [
        line.strip()
        for line in _EXACT_STANDARD_NAME_PREFLIGHT_QUERY.splitlines()
        if "*" in line and "MATCH" in line
    ]
    assert variable_patterns == [
        "OPTIONAL MATCH (target)-[:REFINED_FROM*0..]->(prior:StandardName)",
        "OPTIONAL MATCH (later:StandardName)-[:REFINED_FROM*1..]->(target)",
        "OPTIONAL MATCH (target)-[:HAS_PARENT*0..]->(ancestor:StandardName)",
        "OPTIONAL MATCH (descendant:StandardName)-[:HAS_PARENT*1..]->(target)",
    ]
