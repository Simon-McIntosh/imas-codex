"""Disposable-Neo4j checks for governed DD source snapshot migration."""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from neo4j import GraphDatabase

from imas_codex.graph.client import GraphClient
from imas_codex.settings import get_graph_uri
from imas_codex.standard_names import source_snapshot_migration as migration

pytestmark = pytest.mark.graph


@dataclass(frozen=True)
class _EphemeralNeo4j:
    uri: str

    def driver(self):
        return GraphDatabase.driver(self.uri, auth=None)

    def client(self, failure: str | None = None) -> _InjectedGraphClient:
        return _InjectedGraphClient(self.uri, failure)


@pytest.fixture(scope="module")
def ephemeral_neo4j() -> Iterator[_EphemeralNeo4j]:
    uri = os.environ.get("IMAS_CODEX_TEST_NEO4J_URI")
    if not uri:
        pytest.skip("IMAS_CODEX_TEST_NEO4J_URI is not configured")
    if os.environ.get("IMAS_CODEX_TEST_NEO4J_EPHEMERAL") != "1":
        pytest.fail("source snapshot tests require an ephemeral graph")
    project_uri = os.environ.get("IMAS_CODEX_TEST_PROJECT_NEO4J_URI") or get_graph_uri()
    if uri == project_uri:
        pytest.fail("source snapshot tests refuse the configured project graph")
    driver = GraphDatabase.driver(uri, auth=None)
    driver.verify_connectivity()
    try:
        yield _EphemeralNeo4j(uri)
    finally:
        driver.close()


class _InjectedTransaction:
    def __init__(self, transaction, failure: str | None) -> None:
        self._transaction = transaction
        self._failure = failure

    def run(self, cypher: str, **params: Any):
        if "SOURCE_SNAPSHOT_MIGRATION_LOCK" in cypher and self._failure == "edit":
            self._transaction.run(
                "MATCH (name:StandardName {id: 'name_byte'}) "
                "SET name.edit_status = 'applied'"
            ).consume()
        if "SOURCE_SNAPSHOT_MIGRATION_APPLY" in cypher and self._failure == "event":
            list(self._transaction.run(cypher, **params))
            raise RuntimeError("injected ledger failure")
        return self._transaction.run(cypher, **params)

    def commit(self) -> None:
        self._transaction.commit()

    def rollback(self) -> None:
        self._transaction.rollback()


class _InjectedSession:
    def __init__(self, session, failure: str | None) -> None:
        self._session = session
        self._failure = failure

    def begin_transaction(self) -> _InjectedTransaction:
        return _InjectedTransaction(self._session.begin_transaction(), self._failure)


class _InjectedGraphClient:
    def __init__(self, uri: str, failure: str | None) -> None:
        self._client = GraphClient(
            uri=uri,
            username="neo4j",
            password="",
            graph_name="ephemeral-source-snapshot",
        )
        self._failure = failure

    def close(self) -> None:
        self._client.close()

    @contextmanager
    def session(self):
        with self._client.session() as session:
            yield _InjectedSession(session, self._failure)


def _manifest(
    path: Path, paths: list[str], *, include_exclusions: bool = False
) -> Path:
    records: list[dict[str, Any]] = [
        {
            "scope_status": "executable",
            "next_operator": "bounded_review",
            "participants": {"source_ids": [f"dd:{dd_path}"], "name_ids": []},
            "scope_evidence": {},
        }
        for dd_path in paths
    ]
    special_checks: dict[str, Any] = {}
    if include_exclusions:
        records.extend(
            [
                {
                    "scope_status": "executable",
                    "next_operator": "bounded_review",
                    "participants": {
                        "source_ids": ["dd:west/excluded"],
                        "name_ids": [],
                    },
                    "scope_evidence": {"west_component_hits": ["component:west"]},
                },
                {
                    "scope_status": "executable",
                    "next_operator": "bounded_review",
                    "participants": {
                        "source_ids": ["dd:test/excluded"],
                        "name_ids": [],
                    },
                    "scope_evidence": {"test_component_hits": ["component:test"]},
                },
                {
                    "scope_status": "executable",
                    "next_operator": "DDGap_flag",
                    "participants": {
                        "source_ids": ["dd:defect/excluded"],
                        "name_ids": [],
                    },
                    "scope_evidence": {},
                },
            ]
        )
        special_checks = {
            "declared_defect": {
                "next_operator": "DDGap_flag",
                "source_id": "dd:defect/excluded",
            }
        }
    path.write_text(
        json.dumps(
            {
                "schema": "imas-codex.bounded-integrity-manifest",
                "schema_version": 2,
                "partitions": {"provenance": records},
                "special_checks": special_checks,
            }
        )
    )
    return path


def _seed(driver) -> None:
    with driver.session() as session:
        session.run("MATCH (node) DETACH DELETE node").consume()
        session.run(
            """
            CREATE (:DDVersion {id: 'old-dd', is_current: false})
            CREATE (:DDVersion {id: 'new-dd', is_current: true})
            WITH 1 AS ignored
            UNWIND [
              {path: 'byte/path', name: 'name_byte', old_unit: 'Pa', new_unit: 'Pa',
               old_domain: 'transport', new_domain: 'transport'},
              {path: 'semantic/path', name: 'name_semantic', old_unit: 'Hz',
               new_unit: 's^-1', old_domain: 'transport', new_domain: 'transport'},
              {path: 'changed/path', name: 'name_changed', old_unit: 'Pa',
               new_unit: 'Pa', old_domain: 'transport', new_domain: 'equilibrium'}
            ] AS item
            CREATE (unit:Unit {id: item.new_unit})
            CREATE (parent:IMASNode {id: item.path + '/parent', documentation: 'parent'})
            CREATE (coordinate:IMASNode {id: item.path + '/coordinate'})
            CREATE (node:IMASNode {
              id: item.path, documentation: 'documentation', data_type: 'FLT_1D',
              unit: item.new_unit, physics_domain: item.new_domain,
              lifecycle_status: 'active', lifecycle_version: 'new-dd',
              description: 'enhanced', enrichment_source: 'template'
            })
            CREATE (node)-[:HAS_UNIT]->(unit)
            CREATE (node)-[:HAS_PARENT]->(parent)
            CREATE (node)-[:HAS_COORDINATE]->(coordinate)
            CREATE (source:StandardNameSource {
              id: 'dd:' + item.path, source_type: 'dd', source_id: item.path,
              status: 'attached', attempt_count: 3, batch_key: 'preserved',
              produced_sn_id: item.name, dd_version: 'old-dd',
              description: 'documentation', physics_domain: item.old_domain,
              dd_documentation: 'documentation', dd_snapshot_pinned: true,
              dd_parent_path: item.path + '/parent',
              dd_parent_documentation: 'parent', dd_data_type: 'FLT_1D',
              dd_unit: item.old_unit, dd_coordinates: [item.path + '/coordinate'],
              dd_lifecycle_status: 'active', dd_lifecycle_version: 'new-dd',
              enhanced_description: 'enhanced', enhancement_kind: 'template'
            })
            CREATE (name:StandardName {
              id: item.name, name_stage: 'accepted', docs_stage: 'accepted',
              validation_status: 'valid', edit_status: 'open', edit_reason: 'preserve',
              description: 'name docs', unit: item.old_unit
            })
            CREATE (review:StandardNameReview {
              id: 'review:' + item.name, standard_name_id: item.name, score: 0.95
            })
            CREATE (source)-[:FROM_DD_PATH {owner: 'preserve'}]->(node)
            CREATE (source)-[:PRODUCED_NAME {owner: 'preserve'}]->(name)
            CREATE (node)-[:HAS_STANDARD_NAME {owner: 'preserve'}]->(name)
            CREATE (name)-[:HAS_REVIEW {owner: 'preserve'}]->(review)
            """
        ).consume()
        session.run(
            """
            UNWIND ['west/excluded', 'test/excluded', 'defect/excluded'] AS path
            CREATE (node:IMASNode {id: path, documentation: 'excluded'})
            CREATE (source:StandardNameSource {
              id: 'dd:' + path, source_type: 'dd', source_id: path,
              status: 'extracted', dd_version: 'old-dd', dd_snapshot_pinned: true
            })-[:FROM_DD_PATH]->(node)
            """
        ).consume()


def _query(driver, cypher: str) -> list[dict[str, Any]]:
    with driver.session() as session:
        return [dict(row) for row in session.run(cypher)]


def test_migrates_all_classifications_without_name_or_relationship_churn(
    ephemeral_neo4j: _EphemeralNeo4j, tmp_path: Path
) -> None:
    with ephemeral_neo4j.driver() as driver:
        _seed(driver)
        manifest = _manifest(
            tmp_path / "bounded.json",
            ["byte/path", "semantic/path", "changed/path"],
            include_exclusions=True,
        )
        before_names = _query(
            driver,
            "MATCH (name:StandardName) RETURN name.id AS id, properties(name) AS props "
            "ORDER BY id",
        )
        before_relationships = _query(
            driver,
            "MATCH (start)-[relationship]->(end) "
            "WHERE type(relationship) <> 'HAS_SNAPSHOT_CHANGE' "
            "RETURN start.id AS start, type(relationship) AS type, end.id AS end, "
            "properties(relationship) AS props ORDER BY start, type, end",
        )

        dry_run = migration.migrate_source_snapshots(
            manifest,
            expected_from_version="old-dd",
            reason="refresh immutable authority",
            gc=ephemeral_neo4j.client(),
        )

        assert dry_run["mode"] == "dry_run"
        assert (
            dry_run["counts"]
            | {
                "allowlisted": 3,
                "planned": 3,
                "byte_unchanged": 1,
                "semantic_unchanged": 1,
                "changed": 1,
            }
            == dry_run["counts"]
        )
        assert _query(
            driver,
            "MATCH (source:StandardNameSource) "
            "RETURN DISTINCT source.dd_version AS version ORDER BY version",
        ) == [{"version": "old-dd"}]

        applied = migration.migrate_source_snapshots(
            manifest,
            expected_from_version="old-dd",
            reason="refresh immutable authority",
            apply=True,
            run_id="source-snapshot-migration:test",
            gc=ephemeral_neo4j.client(),
        )

        assert applied["mode"] == "applied"
        assert applied["counts"]["applied"] == 3
        assert _query(
            driver,
            "MATCH (:StandardNameSource)-[:HAS_SNAPSHOT_CHANGE]->(event) "
            "RETURN count(event) AS count",
        ) == [{"count": 3}]
        assert (
            _query(
                driver,
                "MATCH (name:StandardName) RETURN name.id AS id, properties(name) AS props "
                "ORDER BY id",
            )
            == before_names
        )
        assert (
            _query(
                driver,
                "MATCH (start)-[relationship]->(end) "
                "WHERE type(relationship) <> 'HAS_SNAPSHOT_CHANGE' "
                "RETURN start.id AS start, type(relationship) AS type, end.id AS end, "
                "properties(relationship) AS props ORDER BY start, type, end",
            )
            == before_relationships
        )
        excluded = _query(
            driver,
            "MATCH (source:StandardNameSource) WHERE source.id ENDS WITH '/excluded' "
            "RETURN source.id AS id, source.dd_version AS version ORDER BY id",
        )
        assert {row["version"] for row in excluded} == {"old-dd"}

        repeated = migration.migrate_source_snapshots(
            manifest,
            expected_from_version="old-dd",
            reason="refresh immutable authority",
            apply=True,
            gc=ephemeral_neo4j.client(),
        )
        assert repeated["mode"] == "already_current"
        assert repeated["counts"]["already_current"] == 3
        assert _query(
            driver,
            "MATCH (:StandardNameSource)-[:HAS_SNAPSHOT_CHANGE]->(event) "
            "RETURN count(event) AS count",
        ) == [{"count": 3}]


@pytest.mark.parametrize(
    "mutation",
    [
        "source_identity",
        "unpinned",
        "missing_binding",
        "multiple_binding",
        "multiple_unit",
        "multiple_parent",
        "duplicate_coordinate",
        "multiple_current_version",
    ],
)
def test_ambiguous_topology_refuses_without_writes(
    ephemeral_neo4j: _EphemeralNeo4j, tmp_path: Path, mutation: str
) -> None:
    with ephemeral_neo4j.driver() as driver:
        _seed(driver)
        mutation_queries = {
            "source_identity": "MATCH (source {id: 'dd:byte/path'}) SET source.source_id = 'other'",
            "unpinned": "MATCH (source {id: 'dd:byte/path'}) SET source.dd_snapshot_pinned = false",
            "missing_binding": "MATCH (:StandardNameSource {id: 'dd:byte/path'})-[r:FROM_DD_PATH]->() DELETE r",
            "multiple_binding": "MATCH (source:StandardNameSource {id: 'dd:byte/path'}) CREATE (other:IMASNode {id: 'other/path'}) CREATE (source)-[:FROM_DD_PATH]->(other)",
            "multiple_unit": "MATCH (node:IMASNode {id: 'byte/path'}) CREATE (unit:Unit {id: 'bar'}) CREATE (node)-[:HAS_UNIT]->(unit)",
            "multiple_parent": "MATCH (node:IMASNode {id: 'byte/path'}) CREATE (parent:IMASNode {id: 'other/parent'}) CREATE (node)-[:HAS_PARENT]->(parent)",
            "duplicate_coordinate": "MATCH (node:IMASNode {id: 'byte/path'}), (coordinate:IMASNode {id: 'byte/path/coordinate'}) CREATE (node)-[:HAS_COORDINATE]->(coordinate)",
            "multiple_current_version": "CREATE (:DDVersion {id: 'other-current', is_current: true})",
        }
        with driver.session() as session:
            session.run(mutation_queries[mutation]).consume()
        manifest = _manifest(tmp_path / f"{mutation}.json", ["byte/path"])

        receipt = migration.migrate_source_snapshots(
            manifest,
            expected_from_version="old-dd",
            reason="refresh immutable authority",
            apply=True,
            gc=ephemeral_neo4j.client(),
        )

        assert receipt["mode"] == "refused"
        assert receipt["counts"]["refused"] == 1
        assert _query(
            driver,
            "MATCH (source:StandardNameSource {id: 'dd:byte/path'}) "
            "OPTIONAL MATCH (source)-[:HAS_SNAPSHOT_CHANGE]->(event) "
            "RETURN source.dd_version AS version, count(event) AS events",
        ) == [{"version": "old-dd", "events": 0}]


def test_active_source_and_name_claims_refuse(
    ephemeral_neo4j: _EphemeralNeo4j, tmp_path: Path
) -> None:
    with ephemeral_neo4j.driver() as driver:
        _seed(driver)
        with driver.session() as session:
            session.run(
                "MATCH (source:StandardNameSource {id: 'dd:byte/path'}) "
                "SET source.claim_token = 'busy'"
            ).consume()
            session.run(
                "MATCH (name:StandardName {id: 'name_semantic'}) "
                "SET name.drain_scope_id = 'busy'"
            ).consume()
        manifest = _manifest(tmp_path / "claims.json", ["byte/path", "semantic/path"])

        receipt = migration.migrate_source_snapshots(
            manifest,
            expected_from_version="old-dd",
            reason="refresh immutable authority",
            apply=True,
            gc=ephemeral_neo4j.client(),
        )

        assert receipt["mode"] == "refused"
        assert receipt["counts"]["refused"] == 2


@pytest.mark.parametrize("failure", ["edit", "event"])
def test_injected_cas_or_ledger_failure_rolls_back_source_and_event(
    ephemeral_neo4j: _EphemeralNeo4j, tmp_path: Path, failure: str
) -> None:
    with ephemeral_neo4j.driver() as driver:
        _seed(driver)
        manifest = _manifest(tmp_path / f"{failure}.json", ["byte/path"])

        with pytest.raises((RuntimeError, migration.SourceSnapshotMigrationConflict)):
            migration.migrate_source_snapshots(
                manifest,
                expected_from_version="old-dd",
                reason="refresh immutable authority",
                apply=True,
                gc=ephemeral_neo4j.client(failure),
            )

        assert _query(
            driver,
            "MATCH (source:StandardNameSource {id: 'dd:byte/path'}) "
            "OPTIONAL MATCH (source)-[:HAS_SNAPSHOT_CHANGE]->(event) "
            "MATCH (name:StandardName {id: 'name_byte'}) "
            "RETURN source.dd_version AS version, count(event) AS events, "
            "name.edit_status AS edit_status",
        ) == [{"version": "old-dd", "events": 0, "edit_status": "open"}]
