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

    def client(self, *, bad_cardinality: bool = False):
        client = GraphClient(
            uri=self.uri,
            username="neo4j",
            password="",
            graph_name="ephemeral-source-authority",
        )
        if bad_cardinality:
            return _BadCardinalityClient(client)
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
            normalized = reconciliation._without_authority_relationships(row)
            mutable = SNAPSHOT_MUTABLE_FIELDS
            extras: dict[str, Any] = {}
            if operation == reconciliation.REPAIR_IDENTITY_SCALAR:
                mutable = frozenset({"source_id"})
            elif operation == reconciliation.RETIRE_NONPARTICIPATING_SOURCE:
                normalized = reconciliation._retirement_preserved_row(
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
                }
            closure = capture_source_authority_closure(
                normalized,
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
        "expected_preserved_state_hash": closure.preserved_state_hash,
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
    assert "one exact null-lifecycle target" in receipt["refusals"][0]["reasons"]
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
