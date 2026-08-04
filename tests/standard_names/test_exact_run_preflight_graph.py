"""Disposable-Neo4j receipt and access-plan checks for exact preflight."""

from __future__ import annotations

import os
import uuid
from typing import Any

import pytest

from imas_codex.standard_names.defaults import (
    DETERMINISTIC_PARENT_DESCRIPTION_PLACEHOLDER,
)
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
        "operation": "review_name",
        "min_score": 0.85,
        "rotation_cap": 3,
        "parent_desc_placeholder": DETERMINISTIC_PARENT_DESCRIPTION_PLACEHOLDER,
        "facility": None,
        "drain_scope_id": None,
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
    predecessor_name = f"exact_preflight_predecessor_{suffix}"
    fixture_source_id = f"dd:test_review_entry__{suffix}"
    target_path = f"exact/preflight/{suffix}"
    target_source_id = f"dd:{target_path}"
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
                "MERGE (target:StandardName {"
                "  id: $name_id, name_stage: 'drafted', "
                "  validation_status: 'valid', origin: 'pipeline', "
                "  description: 'A reviewable exact target.'}) "
                "SET target.source_paths = [$target_source_id], target.unit = 'm' "
                "REMOVE target.dd_version "
                "MERGE (target_source:StandardNameSource {id: $target_source_id}) "
                "SET target_source.source_id = $target_path, "
                "    target_source.source_type = 'dd', "
                "    target_source.status = 'composed', "
                "    target_source.produced_sn_id = $name_id, "
                "    target_source.dd_version = $dd_version, "
                "    target_source.dd_snapshot_pinned = true, "
                "    target_source.dd_unit = 'm', "
                "    target_source.dd_path = $target_path "
                "MERGE (backing:IMASNode {id: $target_path}) "
                "SET backing.unit = 'm' "
                "MERGE (unit:Unit {id: 'm'}) "
                "MERGE (target_source)-[:PRODUCED_NAME]->(target) "
                "MERGE (target_source)-[:FROM_DD_PATH]->(backing) "
                "MERGE (backing)-[:HAS_STANDARD_NAME]->(target) "
                "MERGE (backing)-[:HAS_UNIT]->(unit) "
                "MERGE (target)-[:HAS_UNIT]->(unit) "
                "MERGE (predecessor:StandardName {"
                "  id: $predecessor_name, name_stage: 'accepted'}) "
                "MERGE (target)-[:REFINED_FROM]->(predecessor) "
                "MERGE (source:StandardNameSource {id: $fixture_source_id}) "
                "MERGE (source)-[:PRODUCED_NAME]->(predecessor)",
                name_id=ordinary_name,
                predecessor_name=predecessor_name,
                fixture_source_id=fixture_source_id,
                target_path=target_path,
                target_source_id=target_source_id,
                dd_version=dd_version,
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
    missing_receipt = dict(missing_rows[0])
    assert missing_receipt["targets"] == []
    assert missing_receipt["action_count"] == 0
    assert missing_receipt["review_action_count"] == 0
    assert len(ordinary_rows) == 1
    ordinary_receipt = dict(ordinary_rows[0])
    assert [target["id"] for target in ordinary_receipt["targets"]] == [ordinary_name]
    assert ordinary_receipt["targets"][0]["source_paths"] == [target_source_id]
    assert ordinary_receipt["targets"][0]["dd_version"] is None
    assert [source["id"] for source in ordinary_receipt["sources"]] == [
        target_source_id
    ]
    assert ordinary_receipt["action_count"] == 1
    assert ordinary_receipt["review_action_count"] == 1
    assert ordinary_receipt["refine_action_count"] == 0
    assert ordinary_receipt["accepted_or_protected_lineage_ids"] == [predecessor_name]
    assert ordinary_receipt["refinement_protected_source_ids"] == [fixture_source_id]
    assert ordinary_receipt["protected_source_ids"] == []

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
