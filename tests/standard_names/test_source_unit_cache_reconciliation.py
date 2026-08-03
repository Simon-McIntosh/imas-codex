"""Disposable-graph contracts for DD source unit-cache reconciliation."""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
from hashlib import sha256
from pathlib import Path
from typing import Any

import pytest
from neo4j import GraphDatabase

from imas_codex.graph.client import GraphClient
from imas_codex.settings import get_graph_uri
from imas_codex.standard_names import source_authority_reconciliation as reconciliation
from imas_codex.standard_names.source_authority import (
    SOURCE_AUTHORITY_CLOSURE_QUERY,
    capture_source_authority_closure,
    participant_ids,
    payload_hash,
    read_source_authority_rows,
    read_source_target_protection_rows,
)

pytestmark = pytest.mark.graph


class _CountingTransaction:
    def __init__(self, transaction, counter: list[int]) -> None:
        self._transaction = transaction
        self._counter = counter

    def run(self, cypher: str, **params: Any):
        self._counter[0] += 1
        return self._transaction.run(cypher, **params)

    def rollback(self) -> None:
        self._transaction.rollback()


class _CountingSession:
    def __init__(self, session, counter: list[int]) -> None:
        self._session = session
        self._counter = counter

    def begin_transaction(self) -> _CountingTransaction:
        return _CountingTransaction(self._session.begin_transaction(), self._counter)


class _CountingClient:
    def __init__(self, client: GraphClient, counter: list[int]) -> None:
        self._client = client
        self._counter = counter

    @contextmanager
    def session(self):
        with self._client.session() as session:
            yield _CountingSession(session, self._counter)


@pytest.fixture(scope="module")
def ephemeral_uri() -> Iterator[str]:
    uri = os.environ.get("IMAS_CODEX_TEST_NEO4J_URI")
    if not uri:
        pytest.skip("IMAS_CODEX_TEST_NEO4J_URI is not configured")
    if os.environ.get("IMAS_CODEX_TEST_NEO4J_EPHEMERAL") != "1":
        pytest.fail("unit-cache reconciliation tests require an ephemeral graph")
    project_uri = os.environ.get("IMAS_CODEX_TEST_PROJECT_NEO4J_URI") or get_graph_uri()
    if uri == project_uri:
        pytest.fail("unit-cache reconciliation tests refuse the project graph")
    password = os.environ.get("IMAS_CODEX_TEST_NEO4J_PASSWORD")
    auth = ("neo4j", password) if password else None
    with GraphDatabase.driver(uri, auth=auth) as driver:
        driver.verify_connectivity()
    yield uri


@pytest.fixture()
def unit_cache_graph(ephemeral_uri: str) -> Iterator[GraphClient]:
    password = os.environ.get("IMAS_CODEX_TEST_NEO4J_PASSWORD", "")
    client = GraphClient(
        uri=ephemeral_uri,
        username="neo4j",
        password=password,
        graph_name="ephemeral-source-unit-cache",
    )
    client.query("MATCH (node) DETACH DELETE node")
    try:
        yield client
    finally:
        client.query("MATCH (node) DETACH DELETE node")
        assert client.query("MATCH (node) RETURN count(node) AS count") == [
            {"count": 0}
        ]
        client.close()


def _snapshot(unit: str) -> dict[str, object]:
    return {
        "dd_version": "4.1.0",
        "description": "Historical gas flow rate description",
        "physics_domain": "particle_sources",
        "dd_documentation": "Gas flow rate",
        "dd_snapshot_pinned": True,
        "dd_parent_path": None,
        "dd_parent_documentation": None,
        "dd_data_type": "FLT_0D",
        "dd_unit": unit,
        "dd_coordinates": [],
        "dd_lifecycle_status": "active",
        "dd_lifecycle_version": "4.1.1",
        "enhanced_description": "Gas flow rate",
        "enhancement_kind": "template",
    }


def _seed(graph: GraphClient) -> tuple[str, ...]:
    paths = (
        "spi/injector/fragmentation_gas/flow_rate",
        "spi/injector/propellant_gas/flow_rate",
    )
    graph.query(
        """
        CREATE (version:DDVersion {id: '4.1.1', is_current: true})
        CREATE (unit:Unit {id: 'Pa.m^3.s^-1'})
        WITH unit
        UNWIND $paths AS path
        CREATE (node:IMASNode {
          id: path, documentation: 'Gas flow rate', description: 'Gas flow rate',
          physics_domain: 'particle_sources', data_type: 'FLT_0D',
          unit: 'Pa.m^3.s^-1', node_category: 'quantity',
          lifecycle_status: 'active', lifecycle_version: '4.1.1',
          enrichment_source: 'template'
        })
        CREATE (node)-[:HAS_UNIT]->(unit)
        CREATE (source:StandardNameSource)
        SET source = $source_properties, source.id = 'dd:' + path,
            source.source_id = path
        CREATE (source)-[:FROM_DD_PATH {authority: 'dd'}]->(node)
        """,
        paths=list(paths),
        source_properties={
            "source_type": "dd",
            "status": "failed",
            **_snapshot("s^-1"),
        },
    )
    graph.query(
        """
        MATCH (source:StandardNameSource {
          id: 'dd:spi/injector/fragmentation_gas/flow_rate'
        })
        MATCH (node:IMASNode {id: 'spi/injector/fragmentation_gas/flow_rate'})
        CREATE (name:StandardName {id: 'gas_flow', name_stage: 'accepted'})
        CREATE (source)-[:PRODUCED_NAME]->(name)
        CREATE (node)-[:HAS_STANDARD_NAME]->(name)
        SET source.status = 'attached', source.produced_sn_id = name.id
        """
    )
    return paths


def _write_manifest(graph: GraphClient, path: Path, paths: tuple[str, ...]) -> str:
    source_ids = tuple(f"dd:{dd_path}" for dd_path in paths)
    with graph.session() as session:
        transaction = session.begin_transaction()
        rows = read_source_authority_rows(transaction, paths)
        protections = read_source_target_protection_rows(
            transaction,
            [
                {
                    "path": dd_path,
                    "source_ids": [source_id],
                    "prospective_target_ids": [],
                }
                for source_id, dd_path in zip(source_ids, paths, strict=True)
            ],
        )
        transaction.rollback()
    manifest_rows = []
    for row, protection in zip(rows, protections, strict=True):
        row["target_protection"] = protection
        normalized = reconciliation._without_authority_relationships(row)
        closure = capture_source_authority_closure(
            normalized,
            manifest_hash="planning",
            authorized_source_ids=frozenset(source_ids),
            mutable_source_fields=frozenset({"dd_unit"}),
        )
        authority_identity_hash, authority_relationships_hash = (
            reconciliation._unit_cache_authority_hashes(closure)
        )
        manifest_rows.append(
            {
                "source_id": f"dd:{row['path']}",
                "operation": reconciliation.RECONCILE_UNIT_CACHE,
                "expected_source_element_id": closure.source["element_id"],
                "expected_source_id": row["path"],
                "expected_from_dd_path": row["path"],
                "expected_before_snapshot_hash": payload_hash(closure.before_snapshot),
                "expected_authority_hash": closure.authority_hash,
                "expected_preserved_state_hash": closure.preserved_state_hash,
                "expected_participant_ids_hash": payload_hash(
                    tuple(sorted(participant_ids(row)))
                ),
                "expected_dd_unit": "s^-1",
                "expected_authority_identity_hash": authority_identity_hash,
                "expected_authority_relationships_hash": authority_relationships_hash,
                "west_intersection": 0,
                "test_intersection": 0,
            }
        )
    path.write_text(
        json.dumps(
            {
                "schema": "imas-codex.source-authority-reconciliation-manifest",
                "schema_version": 1,
                "rows": manifest_rows,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    )
    return sha256(path.read_bytes()).hexdigest()


def test_unit_cache_cohort_is_atomic_narrow_and_idempotent(
    unit_cache_graph: GraphClient, tmp_path: Path
) -> None:
    paths = _seed(unit_cache_graph)
    manifest = tmp_path / "unit-cache.json"
    manifest_hash = _write_manifest(unit_cache_graph, manifest, paths)
    before = unit_cache_graph.query(
        """
        MATCH (source:StandardNameSource)
        RETURN source.id AS id, properties(source) AS properties
        ORDER BY id
        """
    )

    dry_run = reconciliation.reconcile_source_authority(
        manifest,
        reason="align source unit caches with current DD authority",
        gc=unit_cache_graph,
    )
    assert dry_run["mode"] == "dry_run"
    assert dry_run["counts"] == {
        "allowlisted": 2,
        "planned": 2,
        "already_current": 0,
        "applied": 0,
        "refused": 0,
    }
    assert {
        row["source_id"]: {
            "source_dd_version": row["source_dd_version"],
            "authority_dd_version": row["authority_dd_version"],
            "from_unit": row["from_unit"],
            "to_unit": row["to_unit"],
        }
        for row in dry_run["rows"]
    } == {
        f"dd:{path}": {
            "source_dd_version": "4.1.0",
            "authority_dd_version": "4.1.1",
            "from_unit": "s^-1",
            "to_unit": "Pa.m^3.s^-1",
        }
        for path in paths
    }

    applied = reconciliation.reconcile_source_authority(
        manifest,
        reason="align source unit caches with current DD authority",
        apply=True,
        expected_manifest_hash=manifest_hash,
        run_id="source-unit-cache-test",
        gc=unit_cache_graph,
    )
    assert applied["mode"] == "applied"
    after = unit_cache_graph.query(
        """
        MATCH (source:StandardNameSource)
        RETURN source.id AS id, properties(source) AS properties
        ORDER BY id
        """
    )
    for before_row, after_row in zip(before, after, strict=True):
        expected = dict(before_row["properties"])
        expected["dd_unit"] = "Pa.m^3.s^-1"
        assert after_row == {"id": before_row["id"], "properties": expected}
    assert unit_cache_graph.query(
        """
        MATCH (:StandardNameSource)-[:HAS_UNIT_CACHE_CORRECTION]->
              (event:StandardNameSourceUnitCacheCorrection)
        WHERE event.id STARTS WITH 'source-unit-cache-reconciliation:'
        RETURN count(event) AS count,
               collect(DISTINCT event.source_dd_version) AS source_versions,
               collect(DISTINCT event.authority_dd_version) AS authority_versions,
               collect(DISTINCT event.from_unit) AS from_units,
               collect(DISTINCT event.to_unit) AS to_units,
               collect(DISTINCT event.manifest_hash) AS manifest_hashes
        """
    ) == [
        {
            "count": 2,
            "source_versions": ["4.1.0"],
            "authority_versions": ["4.1.1"],
            "from_units": ["s^-1"],
            "to_units": ["Pa.m^3.s^-1"],
            "manifest_hashes": [manifest_hash],
        }
    ]

    repeated = reconciliation.reconcile_source_authority(
        manifest,
        reason="align source unit caches with current DD authority",
        apply=True,
        expected_manifest_hash=manifest_hash,
        run_id="source-unit-cache-repeat",
        gc=unit_cache_graph,
    )
    assert repeated["mode"] == "already_current"
    assert repeated["counts"]["applied"] == 0


def test_unit_cache_authority_disagreement_rolls_back_whole_cohort(
    unit_cache_graph: GraphClient, tmp_path: Path
) -> None:
    paths = _seed(unit_cache_graph)
    manifest = tmp_path / "unit-cache.json"
    manifest_hash = _write_manifest(unit_cache_graph, manifest, paths)
    unit_cache_graph.query(
        """
        MATCH (node:IMASNode {id: $path})
        SET node.unit = 'A'
        """,
        path=paths[1],
    )

    refused = reconciliation.reconcile_source_authority(
        manifest,
        reason="align source unit caches with current DD authority",
        apply=True,
        expected_manifest_hash=manifest_hash,
        gc=unit_cache_graph,
    )

    assert refused["mode"] == "refused"
    assert refused["counts"]["applied"] == 0
    assert unit_cache_graph.query(
        """
        MATCH (source:StandardNameSource)
        RETURN collect(DISTINCT source.dd_unit) AS units
        """
    ) == [{"units": ["s^-1"]}]
    assert unit_cache_graph.query(
        "MATCH (event:StandardNameSourceUnitCacheCorrection) "
        "RETURN count(event) AS count"
    ) == [{"count": 0}]


def test_unit_cache_planning_queries_are_constant_and_seek_bounded(
    unit_cache_graph: GraphClient, tmp_path: Path
) -> None:
    paths = _seed(unit_cache_graph)
    counts = []
    for cohort in (paths[:1], paths):
        manifest = tmp_path / f"unit-cache-{len(cohort)}.json"
        _write_manifest(unit_cache_graph, manifest, cohort)
        counter = [0]
        receipt = reconciliation.reconcile_source_authority(
            manifest,
            reason="align source unit caches with current DD authority",
            gc=_CountingClient(unit_cache_graph, counter),
        )
        assert receipt["counts"]["planned"] == len(cohort)
        counts.append(counter[0])
    assert counts == [3, 3]

    unit_cache_graph.query(
        """
        CREATE CONSTRAINT source_unit_cache_source_id IF NOT EXISTS
        FOR (source:StandardNameSource) REQUIRE source.id IS UNIQUE
        """
    )
    unit_cache_graph.query(
        """
        CREATE CONSTRAINT source_unit_cache_node_id IF NOT EXISTS
        FOR (node:IMASNode) REQUIRE node.id IS UNIQUE
        """
    )
    with unit_cache_graph.session() as session:
        summary = session.run(
            "EXPLAIN " + SOURCE_AUTHORITY_CLOSURE_QUERY,
            paths=list(paths),
        ).consume()

    def operators(plan: dict[str, Any]) -> list[str]:
        return [str(plan.get("operatorType", ""))] + [
            operator
            for child in plan.get("children", [])
            for operator in operators(child)
        ]

    plan_operators = operators(summary.plan)
    assert any("IndexSeek" in operator for operator in plan_operators)
    assert not any("RelationshipTypeScan" in operator for operator in plan_operators)
