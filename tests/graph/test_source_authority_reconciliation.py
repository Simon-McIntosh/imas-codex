"""Disposable-Neo4j contracts for source-authority reconciliation."""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

import pytest
from neo4j import GraphDatabase

from imas_codex.graph.client import GraphClient
from imas_codex.settings import get_graph_uri
from imas_codex.standard_names import source_authority_reconciliation as reconciliation
from imas_codex.standard_names.source_authority import (
    SNAPSHOT_MUTABLE_FIELDS,
    capture_source_authority_closure,
    participant_ids,
    payload_hash,
    read_source_authority_rows,
)

pytestmark = pytest.mark.graph


@dataclass(frozen=True)
class _EphemeralNeo4j:
    uri: str

    def driver(self):
        return GraphDatabase.driver(self.uri, auth=None)

    def client(
        self, *, bad_cardinality: bool = False, relationship_drift: bool = False
    ):
        client = GraphClient(
            uri=self.uri,
            username="neo4j",
            password="",
            graph_name="ephemeral-source-authority",
        )
        if bad_cardinality:
            return _BadCardinalityClient(client)
        if relationship_drift:
            return _RelationshipDriftClient(client, self.uri)
        return client


@pytest.fixture(scope="module")
def ephemeral_neo4j() -> Iterator[_EphemeralNeo4j]:
    uri = os.environ.get("IMAS_CODEX_TEST_NEO4J_URI")
    if not uri:
        pytest.skip("IMAS_CODEX_TEST_NEO4J_URI is not configured")
    if os.environ.get("IMAS_CODEX_TEST_NEO4J_EPHEMERAL") != "1":
        pytest.fail("source-authority tests require an ephemeral graph")
    project_uri = os.environ.get("IMAS_CODEX_TEST_PROJECT_NEO4J_URI") or get_graph_uri()
    if uri == project_uri:
        pytest.fail("source-authority tests refuse the configured project graph")
    driver = GraphDatabase.driver(uri, auth=None)
    driver.verify_connectivity()
    try:
        yield _EphemeralNeo4j(uri)
    finally:
        driver.close()


class _BadCardinalityTransaction:
    def __init__(self, transaction) -> None:
        self._transaction = transaction

    def run(self, cypher: str, **params: Any):
        if "SOURCE_AUTHORITY_REPAIR_IDENTITY_APPLY" in cypher:
            list(self._transaction.run(cypher, **params))
            return []
        return self._transaction.run(cypher, **params)

    def commit(self) -> None:
        self._transaction.commit()

    def rollback(self) -> None:
        self._transaction.rollback()


class _BadCardinalitySession:
    def __init__(self, session) -> None:
        self._session = session

    def begin_transaction(self) -> _BadCardinalityTransaction:
        return _BadCardinalityTransaction(self._session.begin_transaction())


class _BadCardinalityClient:
    def __init__(self, client: GraphClient) -> None:
        self._client = client

    def close(self) -> None:
        self._client.close()

    @contextmanager
    def session(self):
        with self._client.session() as session:
            yield _BadCardinalitySession(session)


class _RelationshipDriftTransaction:
    def __init__(self, transaction, uri: str) -> None:
        self._transaction = transaction
        self._uri = uri
        self._drifted = False

    def run(self, cypher: str, **params: Any):
        if "SOURCE_SNAPSHOT_MIGRATION_LOCK" in cypher and not self._drifted:
            with GraphDatabase.driver(self._uri, auth=None) as driver:
                driver.execute_query(
                    """
                    MATCH (:StandardNameSource)-[backing:FROM_DD_PATH]->(:IMASNode)
                    SET backing.authority = 'concurrent-drift'
                    """
                )
            self._drifted = True
        return self._transaction.run(cypher, **params)

    def commit(self) -> None:
        self._transaction.commit()

    def rollback(self) -> None:
        self._transaction.rollback()


class _RelationshipDriftSession:
    def __init__(self, session, uri: str) -> None:
        self._session = session
        self._uri = uri

    def begin_transaction(self) -> _RelationshipDriftTransaction:
        return _RelationshipDriftTransaction(
            self._session.begin_transaction(), self._uri
        )


class _RelationshipDriftClient:
    def __init__(self, client: GraphClient, uri: str) -> None:
        self._client = client
        self._uri = uri

    def close(self) -> None:
        self._client.close()

    @contextmanager
    def session(self):
        with self._client.session() as session:
            yield _RelationshipDriftSession(session, self._uri)


def _clear(graph: _EphemeralNeo4j) -> None:
    with graph.driver() as driver, driver.session() as session:
        session.run("MATCH (node) DETACH DELETE node").consume()


def _current_snapshot() -> dict[str, Any]:
    return {
        "dd_version": "4.1.1",
        "description": "Authoritative DD documentation",
        "physics_domain": "diagnostics",
        "dd_documentation": "Authoritative DD documentation",
        "dd_snapshot_pinned": True,
        "dd_parent_path": None,
        "dd_parent_documentation": None,
        "dd_data_type": "FLT_0D",
        "dd_unit": "V",
        "dd_coordinates": [],
        "dd_lifecycle_status": "active",
        "dd_lifecycle_version": "4.1.1",
        "enhanced_description": "Enhanced DD description",
        "enhancement_kind": "template",
    }


def _seed(graph: _EphemeralNeo4j, operation: str) -> None:
    path = "diagnostic/channel/value"
    source_id_value: str | None = path
    source_snapshot: dict[str, Any] = _current_snapshot()
    category = (
        "structural"
        if operation == reconciliation.RETIRE_NONPARTICIPATING_SOURCE
        else "quantity"
    )
    if operation == reconciliation.REPAIR_IDENTITY_SCALAR:
        source_id_value = None
        source_snapshot = {
            "dd_version": "4.1.0",
            "description": "Older DD documentation",
            "dd_snapshot_pinned": True,
        }
    elif operation == reconciliation.ADOPT_CURRENT_SNAPSHOT:
        source_snapshot["description"] = "Stale operational mirror"
    elif operation == reconciliation.ADMIT_CURRENT_SNAPSHOT:
        source_snapshot = {}

    source_properties = {
        "id": f"dd:{path}",
        "source_type": "dd",
        "source_id": source_id_value,
        "status": "attached"
        if operation == reconciliation.RETIRE_NONPARTICIPATING_SOURCE
        else "failed",
        "batch_key": "preserved",
        **source_snapshot,
    }
    if operation == reconciliation.RETIRE_NONPARTICIPATING_SOURCE:
        source_properties["produced_sn_id"] = "null_placeholder"
    with graph.driver() as driver, driver.session() as session:
        session.run(
            """
            CREATE (version:DDVersion {id: '4.1.1', is_current: true})
            CREATE (unit:Unit {id: 'V'})
            CREATE (node:IMASNode {
              id: $path,
              documentation: 'Authoritative DD documentation',
              description: 'Enhanced DD description',
              physics_domain: 'diagnostics', data_type: 'FLT_0D', unit: 'V',
              node_category: $category, lifecycle_status: 'active',
              lifecycle_version: '4.1.1', enrichment_source: 'template'
            })
            CREATE (node)-[:HAS_UNIT]->(unit)
            CREATE (source:StandardNameSource)
            SET source = $source_properties
            CREATE (source)-[:FROM_DD_PATH {authority: 'preserved'}]->(node)
            """,
            path=path,
            category=category,
            source_properties=source_properties,
        ).consume()
        if operation == reconciliation.FOLD_DUPLICATE_SOURCE_IDENTITY:
            session.run(
                """
                MATCH (node:IMASNode {id: $path})
                CREATE (duplicate:StandardNameSource {
                  id: 'dd:legacy/duplicate', source_type: 'dd', source_id: $path,
                  status: 'failed', batch_key: 'duplicate-preserved'
                })
                CREATE (duplicate)-[:FROM_DD_PATH {authority: 'duplicate'}]->(node)
                """,
                path=path,
            ).consume()
        if operation == reconciliation.RETIRE_NONPARTICIPATING_SOURCE:
            session.run(
                """
                MATCH (source:StandardNameSource {id: $source_id})
                MATCH (node:IMASNode {id: $path})
                CREATE (name:StandardName {id: 'null_placeholder'})
                CREATE (source)-[:PRODUCED_NAME]->(name)
                CREATE (node)-[:HAS_STANDARD_NAME]->(name)
                """,
                source_id=f"dd:{path}",
                path=path,
            ).consume()


def _seed_additional_repair_source(graph: _EphemeralNeo4j, path: str) -> None:
    source_properties = {
        "id": f"dd:{path}",
        "source_type": "dd",
        "source_id": None,
        "status": "failed",
        "batch_key": "preserved",
        "dd_version": "4.1.0",
        "description": "Older DD documentation",
        "dd_snapshot_pinned": True,
    }
    with graph.driver() as driver, driver.session() as session:
        session.run(
            """
            MATCH (unit:Unit {id: 'V'})
            CREATE (node:IMASNode {
              id: $path,
              documentation: 'Authoritative DD documentation',
              description: 'Enhanced DD description',
              physics_domain: 'diagnostics', data_type: 'FLT_0D', unit: 'V',
              node_category: 'quantity', lifecycle_status: 'active',
              lifecycle_version: '4.1.1', enrichment_source: 'template'
            })
            CREATE (node)-[:HAS_UNIT]->(unit)
            CREATE (source:StandardNameSource)
            SET source = $source_properties
            CREATE (source)-[:FROM_DD_PATH {authority: 'preserved'}]->(node)
            """,
            path=path,
            source_properties=source_properties,
        ).consume()


def _write_live_repair_manifest(
    graph: _EphemeralNeo4j, path: Path, dd_paths: tuple[str, ...]
) -> Path:
    source_ids = tuple(f"dd:{dd_path}" for dd_path in dd_paths)
    client = graph.client()
    try:
        with client.session() as session:
            transaction = session.begin_transaction()
            rows = read_source_authority_rows(transaction, dd_paths)
            manifest_rows = []
            for row in rows:
                dd_path = row["path"]
                normalized = reconciliation._without_authority_relationships(row)
                closure = capture_source_authority_closure(
                    normalized,
                    manifest_hash="planning",
                    authorized_source_ids=frozenset(source_ids),
                    mutable_source_fields=frozenset({"source_id"}),
                )
                manifest_rows.append(
                    {
                        "source_id": f"dd:{dd_path}",
                        "operation": reconciliation.REPAIR_IDENTITY_SCALAR,
                        "expected_source_element_id": closure.source["element_id"],
                        "expected_source_id": None,
                        "expected_from_dd_path": dd_path,
                        "expected_before_snapshot_hash": payload_hash(
                            closure.before_snapshot
                        ),
                        "expected_authority_hash": closure.authority_hash,
                        "expected_preserved_state_hash": closure.preserved_state_hash,
                        "expected_participant_ids_hash": payload_hash(
                            tuple(sorted(participant_ids(row)))
                        ),
                        "west_intersection": 0,
                        "test_intersection": 0,
                    }
                )
            transaction.rollback()
    finally:
        client.close()
    path.write_text(
        json.dumps(
            {
                "schema": "imas-codex.source-authority-reconciliation-manifest",
                "schema_version": 1,
                "rows": manifest_rows,
            }
        )
    )
    return path


def _write_live_manifest(
    graph: _EphemeralNeo4j,
    path: Path,
    operation: str,
) -> Path:
    dd_path = "diagnostic/channel/value"
    source_id = f"dd:{dd_path}"
    client = graph.client()
    try:
        with client.session() as session:
            transaction = session.begin_transaction()
            rows = read_source_authority_rows(transaction, (dd_path,))
            row = rows[0]
            operational = reconciliation._without_authority_relationships(row)
            preserved = operational
            mutable = SNAPSHOT_MUTABLE_FIELDS
            extras: dict[str, Any] = {}
            if operation == reconciliation.REPAIR_IDENTITY_SCALAR:
                mutable = frozenset({"source_id"})
            elif operation == reconciliation.RETIRE_NONPARTICIPATING_SOURCE:
                preserved = reconciliation._retirement_preserved_row(
                    row, "null_placeholder"
                )
                mutable = frozenset(
                    {
                        "status",
                        "produced_sn_id",
                        "claimed_at",
                        "claim_token",
                        "drain_scope_id",
                        "drain_scope_claimed_at",
                        "drain_claim_scope_id",
                        "drain_scope_actionable",
                        "skip_reason",
                        "skip_reason_detail",
                    }
                )
                extras = {
                    "expected_node_category": "structural",
                    "expected_target_id": "null_placeholder",
                    "expected_retirement_destructive_closure_hash": payload_hash(
                        reconciliation._retirement_destructive_closure(
                            row, "null_placeholder"
                        )
                    ),
                }
            closure = capture_source_authority_closure(
                operational,
                manifest_hash="planning",
                authorized_source_ids=frozenset({source_id}),
                mutable_source_fields=mutable,
            )
            preserved_closure = capture_source_authority_closure(
                preserved,
                manifest_hash="planning",
                authorized_source_ids=frozenset({source_id}),
                mutable_source_fields=mutable,
            )
            if operation == reconciliation.FOLD_DUPLICATE_SOURCE_IDENTITY:
                duplicates = reconciliation._read_duplicates(
                    transaction, ("dd:legacy/duplicate",)
                )
                duplicate = duplicates["dd:legacy/duplicate"][0]
                extras = {
                    "duplicate_source_id": "dd:legacy/duplicate",
                    "expected_duplicate_source_element_id": duplicate["element_id"],
                    "expected_duplicate_source_id": dd_path,
                    "expected_duplicate_from_dd_path": dd_path,
                    "expected_duplicate_preserved_state_hash": payload_hash(
                        reconciliation._duplicate_preserved_state(duplicate)
                    ),
                    "expected_duplicate_destructive_closure_hash": payload_hash(
                        reconciliation._duplicate_destructive_closure(duplicate)
                    ),
                }
            transaction.rollback()
    finally:
        client.close()
    manifest_row = {
        "source_id": source_id,
        "operation": operation,
        "expected_source_element_id": closure.source["element_id"],
        "expected_source_id": closure.identity_payload["source_id"],
        "expected_from_dd_path": dd_path,
        "expected_before_snapshot_hash": payload_hash(closure.before_snapshot),
        "expected_authority_hash": closure.authority_hash,
        "expected_preserved_state_hash": preserved_closure.preserved_state_hash,
        "expected_participant_ids_hash": payload_hash(
            tuple(sorted(participant_ids(row)))
        ),
        "west_intersection": 0,
        "test_intersection": 0,
        **extras,
    }
    path.write_text(
        json.dumps(
            {
                "schema": "imas-codex.source-authority-reconciliation-manifest",
                "schema_version": 1,
                "rows": [manifest_row],
            }
        )
    )
    return path


def test_fold_manifest_participant_overlap_refuses_before_graph_mutation(
    ephemeral_neo4j: _EphemeralNeo4j, tmp_path: Path
) -> None:
    _clear(ephemeral_neo4j)
    _seed(ephemeral_neo4j, reconciliation.FOLD_DUPLICATE_SOURCE_IDENTITY)
    template_path = _write_live_manifest(
        ephemeral_neo4j,
        tmp_path / "fold-participant-template.json",
        reconciliation.FOLD_DUPLICATE_SOURCE_IDENTITY,
    )
    first = json.loads(template_path.read_text())["rows"][0]

    repeated = json.loads(json.dumps(first))
    repeated["source_id"] = "dd:second/path"
    repeated["expected_from_dd_path"] = "second/path"
    repeated["expected_duplicate_from_dd_path"] = "second/path"
    repeated_path = tmp_path / "fold-repeated-participant.json"
    repeated_path.write_text(
        json.dumps(
            {
                "schema": "imas-codex.source-authority-reconciliation-manifest",
                "schema_version": 1,
                "rows": [first, repeated],
            }
        )
    )

    overlap = json.loads(json.dumps(first))
    overlap["source_id"] = first["duplicate_source_id"]
    overlap["expected_from_dd_path"] = "legacy/duplicate"
    overlap["expected_duplicate_from_dd_path"] = "legacy/duplicate"
    overlap["duplicate_source_id"] = "dd:other/duplicate"
    overlap_path = tmp_path / "fold-overlapping-participant.json"
    overlap_path.write_text(
        json.dumps(
            {
                "schema": "imas-codex.source-authority-reconciliation-manifest",
                "schema_version": 1,
                "rows": [first, overlap],
            }
        )
    )

    client = ephemeral_neo4j.client()
    try:
        with pytest.raises(ValueError, match="globally unique"):
            reconciliation.reconcile_source_authority(
                repeated_path,
                reason="refuse repeated fold participant",
                gc=client,
            )
        with pytest.raises(ValueError, match="disjoint"):
            reconciliation.reconcile_source_authority(
                overlap_path,
                reason="refuse overlapping fold participant",
                gc=client,
            )
    finally:
        client.close()

    with ephemeral_neo4j.driver() as driver, driver.session() as session:
        row = session.run(
            """
            MATCH (source:StandardNameSource)
            OPTIONAL MATCH (event:StandardNameSourceIdentityFold)
            RETURN count(DISTINCT source) AS sources,
                   count(DISTINCT event) AS events
            """
        ).single()
    assert row["sources"] == 2
    assert row["events"] == 0


@pytest.mark.parametrize(
    "operation",
    [
        reconciliation.REPAIR_IDENTITY_SCALAR,
        reconciliation.ADOPT_CURRENT_SNAPSHOT,
        reconciliation.ADMIT_CURRENT_SNAPSHOT,
        reconciliation.FOLD_DUPLICATE_SOURCE_IDENTITY,
        reconciliation.RETIRE_NONPARTICIPATING_SOURCE,
    ],
)
def test_every_authority_operation_is_atomic_and_idempotent(
    ephemeral_neo4j: _EphemeralNeo4j,
    tmp_path: Path,
    operation: str,
) -> None:
    _clear(ephemeral_neo4j)
    _seed(ephemeral_neo4j, operation)
    manifest = _write_live_manifest(
        ephemeral_neo4j, tmp_path / f"{operation}.json", operation
    )
    manifest_hash = sha256(manifest.read_bytes()).hexdigest()
    client = ephemeral_neo4j.client()
    try:
        dry_run = reconciliation.reconcile_source_authority(
            manifest,
            reason="govern exact source authority",
            gc=client,
        )
        applied = reconciliation.reconcile_source_authority(
            manifest,
            reason="govern exact source authority",
            apply=True,
            expected_manifest_hash=manifest_hash,
            run_id="graph-test",
            gc=client,
        )
        repeated = reconciliation.reconcile_source_authority(
            manifest,
            reason="govern exact source authority",
            apply=True,
            expected_manifest_hash=manifest_hash,
            run_id="graph-test-repeat",
            gc=client,
        )
    finally:
        client.close()

    assert dry_run["mode"] == "dry_run", dry_run
    assert dry_run["counts"]["planned"] == 1
    assert applied["mode"] == "applied", applied
    assert applied["counts"]["applied"] == 1
    assert repeated["mode"] == "already_current", repeated
    assert repeated["counts"]["already_current"] == 1
    with ephemeral_neo4j.driver() as driver, driver.session() as session:
        event_count = session.run(
            "MATCH (event) WHERE $label IN labels(event) RETURN count(event) AS count",
            label=reconciliation._EVENT_LABELS[operation],
        ).single()["count"]
        assert event_count == 1
        name_count = session.run(
            "MATCH (name:StandardName) RETURN count(name) AS count"
        ).single()["count"]
        if operation == reconciliation.RETIRE_NONPARTICIPATING_SOURCE:
            assert name_count == 1, "retirement detaches but does not delete history"


def test_mutation_cardinality_failure_rolls_back_source_and_event(
    ephemeral_neo4j: _EphemeralNeo4j, tmp_path: Path
) -> None:
    _clear(ephemeral_neo4j)
    _seed(ephemeral_neo4j, reconciliation.REPAIR_IDENTITY_SCALAR)
    manifest = _write_live_manifest(
        ephemeral_neo4j,
        tmp_path / "rollback.json",
        reconciliation.REPAIR_IDENTITY_SCALAR,
    )
    client = ephemeral_neo4j.client(bad_cardinality=True)
    try:
        with pytest.raises(
            reconciliation.SourceAuthorityReconciliationConflict,
            match="cardinality",
        ):
            reconciliation.reconcile_source_authority(
                manifest,
                reason="repair exact identity",
                apply=True,
                expected_manifest_hash=sha256(manifest.read_bytes()).hexdigest(),
                gc=client,
            )
    finally:
        client.close()

    with ephemeral_neo4j.driver() as driver, driver.session() as session:
        row = session.run(
            """
            MATCH (source:StandardNameSource {id: 'dd:diagnostic/channel/value'})
            OPTIONAL MATCH (event:StandardNameSourceIdentityRepair)
            RETURN source.source_id AS source_id, count(event) AS events
            """
        ).single()
    assert row["source_id"] is None
    assert row["events"] == 0


def test_retirement_refuses_multiple_targets_without_partial_detachment(
    ephemeral_neo4j: _EphemeralNeo4j, tmp_path: Path
) -> None:
    _clear(ephemeral_neo4j)
    _seed(ephemeral_neo4j, reconciliation.RETIRE_NONPARTICIPATING_SOURCE)
    with ephemeral_neo4j.driver() as driver, driver.session() as session:
        session.run(
            """
            MATCH (source:StandardNameSource {id: 'dd:diagnostic/channel/value'})
            MATCH (node:IMASNode {id: 'diagnostic/channel/value'})
            CREATE (extra:StandardName {id: 'second_placeholder'})
            CREATE (source)-[:PRODUCED_NAME]->(extra)
            CREATE (node)-[:HAS_STANDARD_NAME]->(extra)
            """
        ).consume()
    manifest = _write_live_manifest(
        ephemeral_neo4j,
        tmp_path / "multi-live.json",
        reconciliation.RETIRE_NONPARTICIPATING_SOURCE,
    )
    client = ephemeral_neo4j.client()
    try:
        receipt = reconciliation.reconcile_source_authority(
            manifest,
            reason="retire nonparticipating source",
            gc=client,
        )
    finally:
        client.close()

    assert receipt["mode"] == "refused"
    assert any(
        "one exact null-lifecycle target" in reason
        for reason in receipt["refusals"][0]["reasons"]
    )
    with ephemeral_neo4j.driver() as driver, driver.session() as session:
        row = session.run(
            """
            MATCH (source:StandardNameSource {id: 'dd:diagnostic/channel/value'})
            OPTIONAL MATCH (source)-[:PRODUCED_NAME]->(name)
            RETURN source.status AS status, count(name) AS targets
            """
        ).single()
    assert row["status"] == "attached"
    assert row["targets"] == 2


def test_multirow_identity_repair_commits_and_repeats_atomically(
    ephemeral_neo4j: _EphemeralNeo4j, tmp_path: Path
) -> None:
    _clear(ephemeral_neo4j)
    _seed(ephemeral_neo4j, reconciliation.REPAIR_IDENTITY_SCALAR)
    paths = (
        "diagnostic/channel/value",
        "diagnostic/channel/value_two",
    )
    _seed_additional_repair_source(ephemeral_neo4j, paths[1])
    manifest = _write_live_repair_manifest(
        ephemeral_neo4j, tmp_path / "multirow-success.json", paths
    )
    manifest_hash = sha256(manifest.read_bytes()).hexdigest()
    client = ephemeral_neo4j.client()
    try:
        applied = reconciliation.reconcile_source_authority(
            manifest,
            reason="repair exact identity cohort",
            apply=True,
            expected_manifest_hash=manifest_hash,
            gc=client,
        )
        repeated = reconciliation.reconcile_source_authority(
            manifest,
            reason="repair exact identity cohort",
            apply=True,
            expected_manifest_hash=manifest_hash,
            gc=client,
        )
    finally:
        client.close()

    assert applied["mode"] == "applied", applied
    assert applied["counts"]["applied"] == 2
    assert repeated["mode"] == "already_current", repeated
    assert repeated["counts"]["already_current"] == 2
    with ephemeral_neo4j.driver() as driver, driver.session() as session:
        row = session.run(
            """
            MATCH (source:StandardNameSource)
            WHERE source.id IN $source_ids
            OPTIONAL MATCH (source)-[:HAS_IDENTITY_REPAIR]->(event)
            RETURN count(DISTINCT source) AS sources,
                   count(event) AS events,
                   collect(source.source_id) AS repaired_ids
            """,
            source_ids=[f"dd:{path}" for path in paths],
        ).single()
    assert row["sources"] == 2
    assert row["events"] == 2
    assert set(row["repaired_ids"]) == set(paths)


def test_multirow_late_failure_rolls_back_every_source_and_event(
    ephemeral_neo4j: _EphemeralNeo4j, tmp_path: Path
) -> None:
    _clear(ephemeral_neo4j)
    _seed(ephemeral_neo4j, reconciliation.REPAIR_IDENTITY_SCALAR)
    paths = (
        "diagnostic/channel/value",
        "diagnostic/channel/value_two",
    )
    _seed_additional_repair_source(ephemeral_neo4j, paths[1])
    manifest = _write_live_repair_manifest(
        ephemeral_neo4j, tmp_path / "multirow-rollback.json", paths
    )
    client = ephemeral_neo4j.client(bad_cardinality=True)
    try:
        with pytest.raises(
            reconciliation.SourceAuthorityReconciliationConflict,
            match="cardinality",
        ):
            reconciliation.reconcile_source_authority(
                manifest,
                reason="repair exact identity cohort",
                apply=True,
                expected_manifest_hash=sha256(manifest.read_bytes()).hexdigest(),
                gc=client,
            )
    finally:
        client.close()

    with ephemeral_neo4j.driver() as driver, driver.session() as session:
        row = session.run(
            """
            MATCH (source:StandardNameSource)
            WHERE source.id IN $source_ids
            OPTIONAL MATCH (event:StandardNameSourceIdentityRepair)
            RETURN count(source.source_id) AS repaired,
                   count(DISTINCT event) AS events
            """,
            source_ids=[f"dd:{path}" for path in paths],
        ).single()
    assert row["repaired"] == 0
    assert row["events"] == 0


def test_externally_committed_relationship_drift_refuses_before_mutation(
    ephemeral_neo4j: _EphemeralNeo4j, tmp_path: Path
) -> None:
    _clear(ephemeral_neo4j)
    _seed(ephemeral_neo4j, reconciliation.REPAIR_IDENTITY_SCALAR)
    manifest = _write_live_manifest(
        ephemeral_neo4j,
        tmp_path / "relationship-drift.json",
        reconciliation.REPAIR_IDENTITY_SCALAR,
    )
    client = ephemeral_neo4j.client(relationship_drift=True)
    try:
        with pytest.raises(
            reconciliation.SourceAuthorityReconciliationConflict,
            match="changed after locks",
        ):
            reconciliation.reconcile_source_authority(
                manifest,
                reason="repair exact identity",
                apply=True,
                expected_manifest_hash=sha256(manifest.read_bytes()).hexdigest(),
                gc=client,
            )
    finally:
        client.close()

    with ephemeral_neo4j.driver() as driver, driver.session() as session:
        row = session.run(
            """
            MATCH (source:StandardNameSource)-[backing:FROM_DD_PATH]->(:IMASNode)
            OPTIONAL MATCH (event:StandardNameSourceIdentityRepair)
            RETURN source.source_id AS source_id,
                   backing.authority AS authority,
                   count(event) AS events
            """
        ).single()
    assert row["source_id"] is None
    assert row["authority"] == "concurrent-drift"
    assert row["events"] == 0


def test_snapshot_admission_can_precede_nonparticipating_retirement(
    ephemeral_neo4j: _EphemeralNeo4j, tmp_path: Path
) -> None:
    _clear(ephemeral_neo4j)
    _seed(ephemeral_neo4j, reconciliation.RETIRE_NONPARTICIPATING_SOURCE)
    with ephemeral_neo4j.driver() as driver, driver.session() as session:
        session.run(
            """
            MATCH (source:StandardNameSource {id: 'dd:diagnostic/channel/value'})
            REMOVE source.dd_version, source.description, source.physics_domain,
                   source.dd_documentation, source.dd_snapshot_pinned,
                   source.dd_parent_path, source.dd_parent_documentation,
                   source.dd_data_type, source.dd_unit, source.dd_coordinates,
                   source.dd_lifecycle_status, source.dd_lifecycle_version,
                   source.enhanced_description, source.enhancement_kind
            """
        ).consume()
    admission_manifest = _write_live_manifest(
        ephemeral_neo4j,
        tmp_path / "admission-before-retirement.json",
        reconciliation.ADMIT_CURRENT_SNAPSHOT,
    )
    client = ephemeral_neo4j.client()
    try:
        admission = reconciliation.reconcile_source_authority(
            admission_manifest,
            reason="admit current snapshot",
            apply=True,
            expected_manifest_hash=sha256(admission_manifest.read_bytes()).hexdigest(),
            gc=client,
        )
        retirement_manifest = _write_live_manifest(
            ephemeral_neo4j,
            tmp_path / "retirement-after-admission.json",
            reconciliation.RETIRE_NONPARTICIPATING_SOURCE,
        )
        retirement = reconciliation.reconcile_source_authority(
            retirement_manifest,
            reason="retire nonparticipating source",
            apply=True,
            expected_manifest_hash=sha256(retirement_manifest.read_bytes()).hexdigest(),
            gc=client,
        )
    finally:
        client.close()

    assert admission["mode"] == "applied", admission
    assert retirement["mode"] == "applied", retirement
    with ephemeral_neo4j.driver() as driver, driver.session() as session:
        row = session.run(
            """
            MATCH (source:StandardNameSource {id: 'dd:diagnostic/channel/value'})
            OPTIONAL MATCH (source)-[:HAS_SNAPSHOT_ADMISSION]->(admission)
            OPTIONAL MATCH (source)-[:HAS_AUTHORITY_RETIREMENT]->(retirement)
            RETURN source.status AS status,
                   count(DISTINCT admission) AS admissions,
                   count(DISTINCT retirement) AS retirements
            """
        ).single()
    assert row["status"] == "stale"
    assert row["admissions"] == 1
    assert row["retirements"] == 1
