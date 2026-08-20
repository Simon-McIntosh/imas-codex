"""Disposable-graph coverage for atomic source release and retirement."""

from __future__ import annotations

import hashlib
import inspect
import json
import os
from collections.abc import Iterator
from pathlib import Path

import pytest
from neo4j import GraphDatabase

from imas_codex.graph.client import GraphClient
from imas_codex.settings import get_graph_uri
from imas_codex.standard_names.graph_ops import (
    retire_signed_dual_authority_targets,
)

_EVIDENCE_DIR = Path(__file__).parents[2] / "docs/evidence/sn-graph-wide-integrity"
_SOURCE_AUTHORITY_PATH = _EVIDENCE_DIR / "catalog-edit-dual-binding-adjudication.json"
_RETIREMENT_AUTHORITY_PATH = _EVIDENCE_DIR / "refused-target-orphan-adjudication.json"
_RETIRE_DISPOSITION = "retire_under_orphan_policy"


def _canonical_hash(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _authorities() -> tuple[
    dict[str, object],
    dict[str, object],
    str,
    list[dict[str, object]],
    list[dict[str, object]],
]:
    source_authority = json.loads(_SOURCE_AUTHORITY_PATH.read_text())
    retirement_authority = json.loads(_RETIREMENT_AUTHORITY_PATH.read_text())
    retirement_rows = [
        row
        for row in retirement_authority["rows"]
        if row["disposition"] == _RETIRE_DISPOSITION
    ]
    retirement_ids = {row["name"] for row in retirement_rows}
    source_rows = [
        row
        for row in source_authority["rows"]
        if retirement_ids.intersection(row["removed_targets"])
    ]
    assert len(retirement_rows) == 16
    assert len(source_rows) == 19
    assert (
        sum(
            target_id in retirement_ids
            for row in source_rows
            for target_id in row["removed_targets"]
        )
        == 20
    )
    return (
        source_authority,
        retirement_authority,
        _canonical_hash(retirement_authority),
        source_rows,
        retirement_rows,
    )


@pytest.fixture(scope="module")
def disposable_neo4j() -> Iterator[tuple[str, str]]:
    uri = os.environ.get("IMAS_CODEX_TEST_NEO4J_URI")
    if not uri:
        pytest.skip("IMAS_CODEX_TEST_NEO4J_URI is not configured")
    if os.environ.get("IMAS_CODEX_TEST_NEO4J_EPHEMERAL") != "1":
        pytest.fail("dual-authority retirement requires a disposable graph")
    if uri == (os.environ.get("IMAS_CODEX_TEST_PROJECT_NEO4J_URI") or get_graph_uri()):
        pytest.fail("dual-authority retirement refuses the project graph")
    password = os.environ.get("IMAS_CODEX_TEST_NEO4J_PASSWORD", "")
    auth = ("neo4j", password) if password else None
    print(f"GRAPH_ENDPOINT={uri}")
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
        graph_name="dual-authority-retirement",
    )
    client.query("MATCH (node) DETACH DELETE node")
    yield client
    client.query("MATCH (node) DETACH DELETE node")
    client.close()


def _seed_signed_closure(
    client: GraphClient,
    source_rows: list[dict[str, object]],
    retirement_rows: list[dict[str, object]],
) -> None:
    retirement_stage = {row["name"]: row["name_stage"] for row in retirement_rows}
    names = sorted(
        {
            target_id
            for row in source_rows
            for target_id in row["candidate_live_targets"]
        }
    )
    client.query(
        """
        UNWIND $rows AS row
        CREATE (:StandardName {
          id: row.id,
          name_stage: row.name_stage,
          status: 'active',
          origin: CASE WHEN row.retirement THEN 'catalog_edit' ELSE 'pipeline' END,
          validation_status: 'valid'
        })
        """,
        rows=[
            {
                "id": name,
                "name_stage": retirement_stage.get(name, "accepted"),
                "retirement": name in retirement_stage,
            }
            for name in names
        ],
    )
    client.query(
        """
        UNWIND $rows AS row
        CREATE (backing:IMASNode {id: substring(row.source_id, 3)})
        CREATE (source:StandardNameSource {
          id: row.source_id,
          source_type: 'dd',
          source_id: substring(row.source_id, 3),
          status: 'attached',
          produced_sn_id: row.prior_scalar_target
        })
        CREATE (source)-[:FROM_DD_PATH]->(backing)
        WITH row, source, backing
        UNWIND row.candidate_live_targets AS target_id
        MATCH (target:StandardName {id: target_id})
        CREATE (source)-[:PRODUCED_NAME]->(target)
        CREATE (backing)-[:HAS_STANDARD_NAME]->(target)
        """,
        rows=source_rows,
    )


def _preview_and_apply(
    client: GraphClient,
    source_authority: dict[str, object],
    retirement_authority: dict[str, object],
    retirement_hash: str,
) -> tuple[dict[str, object], dict[str, object]]:
    reason = "joint source and lifecycle authority identifies obsolete identities"
    preview = retire_signed_dual_authority_targets(
        source_authority,
        retirement_authority,
        retirement_authority_sha256=retirement_hash,
        reason=reason,
        gc=client,
    )
    assert preview["outcome"] == "would_apply"
    applied = retire_signed_dual_authority_targets(
        source_authority,
        retirement_authority,
        retirement_authority_sha256=retirement_hash,
        reason=reason,
        apply=True,
        manifest_sha256=preview["manifest_sha256"],
        gc=client,
    )
    return preview, applied


@pytest.mark.graph
def test_exact_signed_bindings_and_targets_change_with_per_identity_ledger(
    graph: GraphClient,
) -> None:
    source, retirement, retirement_hash, source_rows, retirement_rows = _authorities()
    _seed_signed_closure(graph, source_rows, retirement_rows)
    _, applied = _preview_and_apply(graph, source, retirement, retirement_hash)
    target_ids = [row["name"] for row in retirement_rows]

    assert applied["changed"] == 16
    assert applied["sources_reconciled"] == 19
    assert applied["bindings_released"] == 20
    assert applied["projections_released"] == 20
    assert applied["superseded"] == 16
    assert applied["ledger_rows"] == 16
    result = graph.query(
        """
        MATCH (target:StandardName)
        WHERE target.id IN $target_ids
        OPTIONAL MATCH (target)-[:HAS_INTERNAL_CHANGE]->
          (change:StandardNameChange {
            operation: 'retire_signed_dual_authority_target'
          })
        RETURN count(target) AS targets,
               count(CASE WHEN target.name_stage = 'superseded'
                                AND target.status = 'superseded'
                          THEN 1 END) AS superseded,
               count(change) AS ledger_rows,
               count(DISTINCT change.id) AS distinct_ledger_rows
        """,
        target_ids=target_ids,
    )
    assert result == [
        {
            "targets": 16,
            "superseded": 16,
            "ledger_rows": 16,
            "distinct_ledger_rows": 16,
        }
    ]


@pytest.mark.graph
def test_target_absent_from_signed_retire_set_cannot_enter_the_transaction(
    graph: GraphClient,
) -> None:
    source, retirement, retirement_hash, source_rows, retirement_rows = _authorities()
    _seed_signed_closure(graph, source_rows, retirement_rows)
    first_source = str(source_rows[0]["source_id"])
    path = first_source[3:]
    graph.query(
        """
        MATCH (source:StandardNameSource {id: $source_id})
        MATCH (source)-[:FROM_DD_PATH]->(backing:IMASNode {id: $path})
        CREATE (unsigned:StandardName {
          id: 'unsigned_retirement_target', name_stage: 'accepted', status: 'active'
        })
        CREATE (source)-[:PRODUCED_NAME]->(unsigned)
        CREATE (backing)-[:HAS_STANDARD_NAME]->(unsigned)
        """,
        source_id=first_source,
        path=path,
    )

    receipt = retire_signed_dual_authority_targets(
        source,
        retirement,
        retirement_authority_sha256=retirement_hash,
        reason="only the exact jointly signed cohort may change",
        gc=graph,
    )
    assert receipt["outcome"] == "refused"
    assert receipt["changed"] == 0
    assert (
        "name_ids"
        not in inspect.signature(retire_signed_dual_authority_targets).parameters
    )
    unsigned = graph.query(
        """
        MATCH (source:StandardNameSource {id: $source_id})
          -[:PRODUCED_NAME]->(target:StandardName {id: 'unsigned_retirement_target'})
        RETURN target.name_stage AS stage, count(*) AS bindings
        """,
        source_id=first_source,
    )
    assert unsigned == [{"stage": "accepted", "bindings": 1}]


@pytest.mark.graph
def test_new_live_structural_child_refuses_the_whole_cohort(
    graph: GraphClient,
) -> None:
    source, retirement, retirement_hash, source_rows, retirement_rows = _authorities()
    _seed_signed_closure(graph, source_rows, retirement_rows)
    target_id = str(retirement_rows[0]["name"])
    graph.query(
        """
        MATCH (target:StandardName {id: $target_id})
        CREATE (child:StandardName {
          id: 'new_structurally_legitimate_child',
          name_stage: 'accepted', status: 'active'
        })-[:HAS_PARENT]->(target)
        """,
        target_id=target_id,
    )

    receipt = retire_signed_dual_authority_targets(
        source,
        retirement,
        retirement_authority_sha256=retirement_hash,
        reason="structural legitimacy must remain authoritative",
        gc=graph,
    )
    assert receipt["outcome"] == "refused"
    assert receipt["changed"] == 0
    assert receipt["counts"]["targets"] == 16
    assert any(
        refusal.get("name_id") == target_id and "HAS_PARENT" in refusal["reason"]
        for refusal in receipt["refusals"]
    )
    unchanged = graph.query(
        """
        MATCH (target:StandardName)
        WHERE target.id IN $target_ids AND target.name_stage <> 'superseded'
        RETURN count(target) AS count
        """,
        target_ids=[row["name"] for row in retirement_rows],
    )
    assert unchanged == [{"count": 16}]


@pytest.mark.graph
def test_every_released_final_binding_is_superseded_in_the_same_commit(
    graph: GraphClient,
) -> None:
    source, retirement, retirement_hash, source_rows, retirement_rows = _authorities()
    _seed_signed_closure(graph, source_rows, retirement_rows)
    retirement_ids = {row["name"] for row in retirement_rows}
    retained_pairs = sorted(
        (row["source_id"], target_id)
        for row in source_rows
        for target_id in row["removed_targets"]
        if target_id not in retirement_ids
    )
    assert retained_pairs

    _, applied = _preview_and_apply(graph, source, retirement, retirement_hash)
    assert applied["bindings_released"] == 20
    postcondition = graph.query(
        """
        MATCH (target:StandardName)
        WHERE target.id IN $target_ids
        RETURN count(CASE WHEN target.name_stage = 'superseded'
                          AND NOT EXISTS {
                            (:StandardNameSource)-[:PRODUCED_NAME]->(target)
                          } THEN 1 END) AS retired_without_binding
        """,
        target_ids=sorted(retirement_ids),
    )
    assert postcondition == [{"retired_without_binding": 16}]
    retained = graph.query(
        """
        UNWIND $pairs AS expected
        MATCH (:StandardNameSource {id: expected.source_id})
          -[:PRODUCED_NAME]->(target:StandardName {id: expected.target_id})
        RETURN count(*) AS bindings,
               count(CASE WHEN target.name_stage <> 'superseded' THEN 1 END)
                 AS live_targets
        """,
        pairs=[
            {"source_id": source_id, "target_id": target_id}
            for source_id, target_id in retained_pairs
        ],
    )
    assert retained == [
        {"bindings": len(retained_pairs), "live_targets": len(retained_pairs)}
    ]


@pytest.mark.graph
def test_identical_manifest_replays_without_graph_writes(graph: GraphClient) -> None:
    source, retirement, retirement_hash, source_rows, retirement_rows = _authorities()
    _seed_signed_closure(graph, source_rows, retirement_rows)
    preview, applied = _preview_and_apply(graph, source, retirement, retirement_hash)
    before = _snapshot(graph)
    replay = retire_signed_dual_authority_targets(
        source,
        retirement,
        retirement_authority_sha256=retirement_hash,
        reason="joint source and lifecycle authority identifies obsolete identities",
        apply=True,
        manifest_sha256=preview["manifest_sha256"],
        gc=graph,
    )
    after = _snapshot(graph)
    assert applied["outcome"] == "applied"
    assert replay["outcome"] == "already_applied"
    assert replay["changed"] == 0
    assert replay["persistent_writes"] == 0
    assert replay["ledger_rows"] == 16
    assert after == before


def _snapshot(client: GraphClient) -> bytes:
    nodes = client.query(
        """
        MATCH (node)
        RETURN elementId(node) AS element_id, labels(node) AS labels,
               properties(node) AS properties
        ORDER BY element_id
        """
    )
    relationships = client.query(
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
