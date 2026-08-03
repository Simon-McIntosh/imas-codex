"""Contracts for manifest-bound structural closure reconciliation."""

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
from imas_codex.standard_names import structural_closure as closure
from imas_codex.standard_names.structural_closure import (
    EXCLUDE_NULL_SCAFFOLD,
    MATERIALIZE_ADMISSIBLE_PARENT,
    REFUSE_MISSING_UNIT_AUTHORITY,
    REFUSE_UNACCEPTED_CHILD_AUTHORITY,
    RETIRE_UNREACHABLE_CHAIN,
    SEED_ACCEPTED_PARENT_SOURCE,
    StructuralClosureConflict,
    build_structural_closure_manifest_row,
    load_structural_closure_manifest,
)


def _row(root_id: str = "velocity_due_to_convection") -> dict[str, object]:
    return {
        "root_id": root_id,
        "expected_actions": [MATERIALIZE_ADMISSIBLE_PARENT],
        "retire_ids": [],
        "scaffold_ids": [],
        "unit_override": None,
        "expected_closure_hash": "a" * 64,
        "expected_participant_ids_hash": "b" * 64,
        "expected_relationship_ids_hash": "c" * 64,
        "expected_admission_hash": "d" * 64,
        "west_intersection": 0,
        "test_intersection": 0,
        "reason": "resolve exact structural closure",
    }


def _write(path: Path, rows: list[dict[str, object]]) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema": "imas-codex.structural-closure-reconciliation-manifest",
                "schema_version": 1,
                "rows": rows,
            }
        )
    )
    return path


def test_manifest_is_exact_deterministic_and_hash_bound(tmp_path: Path) -> None:
    first = _row("velocity_due_to_convection")
    second = _row("convected_velocity")
    manifest_path = _write(tmp_path / "closure.json", [first, second])
    manifest = load_structural_closure_manifest(manifest_path)

    assert manifest.root_ids == ("convected_velocity", "velocity_due_to_convection")
    assert manifest.manifest_hash == sha256(manifest_path.read_bytes()).hexdigest()
    assert tuple(row["root_id"] for row in manifest.rows) == manifest.root_ids


def test_manifest_refuses_duplicates_unknown_actions_and_protected_rows(
    tmp_path: Path,
) -> None:
    duplicate = _write(tmp_path / "duplicate.json", [_row(), _row()])
    with pytest.raises(ValueError, match="duplicate"):
        load_structural_closure_manifest(duplicate)

    unknown_row = _row()
    unknown_row["expected_actions"] = ["invent_parent"]
    unknown = _write(tmp_path / "unknown.json", [unknown_row])
    with pytest.raises(ValueError, match="action"):
        load_structural_closure_manifest(unknown)

    protected_row = _row()
    protected_row["west_intersection"] = 1
    protected = _write(tmp_path / "protected.json", [protected_row])
    with pytest.raises(ValueError, match="intersections"):
        load_structural_closure_manifest(protected)


def test_manifest_refuses_overlapping_destructive_targets(tmp_path: Path) -> None:
    first = _row("first_parent")
    second = _row("second_parent")
    first["expected_actions"] = [EXCLUDE_NULL_SCAFFOLD]
    second["expected_actions"] = [EXCLUDE_NULL_SCAFFOLD]
    first["scaffold_ids"] = ["shared_scaffold"]
    second["scaffold_ids"] = ["shared_scaffold"]

    manifest = _write(tmp_path / "overlap.json", [first, second])
    with pytest.raises(ValueError, match="overlap"):
        load_structural_closure_manifest(manifest)


def test_unit_override_requires_reviewed_provenance(tmp_path: Path) -> None:
    row = _row()
    row["unit_override"] = {"unit": "m.s^-1", "provenance": ""}
    manifest = _write(tmp_path / "unit.json", [row])
    with pytest.raises(ValueError, match="provenance"):
        load_structural_closure_manifest(manifest)


def test_missing_unit_is_a_typed_refusal() -> None:
    assert REFUSE_MISSING_UNIT_AUTHORITY == "refuse_missing_unit_authority"
    assert issubclass(StructuralClosureConflict, RuntimeError)


def _synthetic_parent_closure(
    *, child_stage: str = "accepted", child_unit: str | None = "T"
) -> dict[str, Any]:
    child_properties: dict[str, Any] = {
        "id": "radial_magnetic_field",
        "name_stage": child_stage,
        "origin": "derived",
    }
    if child_unit is not None:
        child_properties["unit"] = child_unit
    child = {
        "element_id": "node-child",
        "labels": ["StandardName"],
        "properties": child_properties,
        "sources": [
            {
                "element_id": "source-child",
                "relationship_element_id": "binding-child",
                "properties": {
                    "id": "derived:radial_magnetic_field",
                    "status": "composed",
                },
            }
        ],
        "units": [],
        "dd_sources": [],
    }
    return {
        "root_id": "magnetic_field",
        "roots": [
            {
                "element_id": "node-parent",
                "labels": ["StandardName"],
                "properties": {"id": "magnetic_field", "origin": "derived"},
            }
        ],
        "names": [
            {
                "element_id": "node-parent",
                "labels": ["StandardName"],
                "properties": {"id": "magnetic_field", "origin": "derived"},
                "sources": [],
                "units": [],
                "dd_sources": [],
            },
            child,
        ],
        "parent_edges": [
            {
                "element_id": "edge-parent",
                "type": "HAS_PARENT",
                "start_id": "radial_magnetic_field",
                "end_id": "magnetic_field",
                "properties": {"operator_kind": "projection", "axis": "radial"},
            }
        ],
        "depth_truncated": False,
    }


@pytest.mark.parametrize(
    ("child_stage", "child_unit", "expected_action", "reason_fragment"),
    [
        (
            "accepted",
            None,
            REFUSE_MISSING_UNIT_AUTHORITY,
            "unit authority",
        ),
        (
            "reviewed",
            "T",
            REFUSE_UNACCEPTED_CHILD_AUTHORITY,
            "unaccepted child authority",
        ),
    ],
)
def test_materialization_refuses_missing_child_authority(
    child_stage: str,
    child_unit: str | None,
    expected_action: str,
    reason_fragment: str,
) -> None:
    row = _synthetic_parent_closure(child_stage=child_stage, child_unit=child_unit)
    manifest_row = build_structural_closure_manifest_row(
        row,
        root_id="magnetic_field",
        expected_actions=[expected_action],
        reason="preserve reviewed child authority",
    )

    plan = closure._plan_row(row, manifest_row, include_accepted=False)

    assert plan["status"] == "refused"
    assert plan["actions"] == [expected_action]
    assert any(reason_fragment in reason for reason in plan["unresolved"])


def test_accepted_retirement_requires_explicit_authorization() -> None:
    row = _synthetic_parent_closure()
    row["names"] = [row["names"][0]]
    row["parent_edges"] = []
    row["names"][0]["properties"]["name_stage"] = "accepted"
    row["roots"][0]["properties"]["name_stage"] = "accepted"
    manifest_row = build_structural_closure_manifest_row(
        row,
        root_id="magnetic_field",
        expected_actions=[RETIRE_UNREACHABLE_CHAIN],
        retire_ids=["magnetic_field"],
        reason="retire an exact unreachable materialized identity",
    )

    guarded = closure._plan_row(row, manifest_row, include_accepted=False)
    authorized = closure._plan_row(row, manifest_row, include_accepted=True)

    assert guarded["status"] == "refused"
    assert any("include_accepted" in reason for reason in guarded["unresolved"])
    assert authorized["status"] == "planned"


class _EphemeralNeo4j:
    def __init__(self, uri: str) -> None:
        self.uri = uri

    def client(self) -> GraphClient:
        return GraphClient(
            uri=self.uri,
            username="neo4j",
            password="",
            graph_name="ephemeral-structural-closure",
        )


@pytest.fixture(scope="module")
def ephemeral_neo4j() -> Iterator[_EphemeralNeo4j]:
    uri = os.environ.get("IMAS_CODEX_TEST_NEO4J_URI")
    if not uri:
        pytest.skip("IMAS_CODEX_TEST_NEO4J_URI is not configured")
    if os.environ.get("IMAS_CODEX_TEST_NEO4J_EPHEMERAL") != "1":
        pytest.fail("structural-closure tests require an ephemeral graph")
    project_uri = os.environ.get("IMAS_CODEX_TEST_PROJECT_NEO4J_URI") or get_graph_uri()
    if uri == project_uri:
        pytest.fail("structural-closure tests refuse the configured project graph")
    driver = GraphDatabase.driver(uri, auth=None)
    driver.verify_connectivity()
    try:
        yield _EphemeralNeo4j(uri)
    finally:
        driver.close()


@pytest.fixture()
def structural_graph(ephemeral_neo4j: _EphemeralNeo4j) -> Iterator[GraphClient]:
    client = ephemeral_neo4j.client()
    client.query("MATCH (node) DETACH DELETE node")
    try:
        yield client
    finally:
        client.query("MATCH (node) DETACH DELETE node")
        client.close()


def _snapshot(client: GraphClient, root_ids: list[str]) -> list[dict[str, Any]]:
    with client.session() as session:
        transaction = session.begin_transaction()
        try:
            return closure._read_rows(transaction, tuple(sorted(root_ids)))
        finally:
            transaction.rollback()


def _bound_manifest(
    tmp_path: Path,
    client: GraphClient,
    specifications: dict[str, dict[str, Any]],
) -> tuple[Path, str]:
    rows = _snapshot(client, list(specifications))
    by_root = {str(row["root_id"]): row for row in rows}
    manifest_rows = [
        build_structural_closure_manifest_row(
            by_root[root_id],
            root_id=root_id,
            reason="reconcile the exact audited structural closure",
            **specification,
        )
        for root_id, specification in sorted(specifications.items())
    ]
    path = _write(tmp_path / "structural-closure.json", manifest_rows)
    return path, sha256(path.read_bytes()).hexdigest()


def _seed_vector_parent(client: GraphClient) -> None:
    client.query(
        """
        CREATE (parent:StandardName {id: 'magnetic_field', origin: 'derived'})
        CREATE (radial:StandardName {
          id: 'radial_magnetic_field', origin: 'derived', name_stage: 'accepted',
          unit: 'T', physics_domain: 'magnetics'
        })
        CREATE (toroidal:StandardName {
          id: 'toroidal_magnetic_field', origin: 'derived', name_stage: 'accepted',
          unit: 'T', physics_domain: 'magnetics'
        })
        CREATE (unit:Unit {id: 'T'})
        CREATE (radial_source:StandardNameSource {
          id: 'derived:radial_magnetic_field', status: 'composed'
        })-[:PRODUCED_NAME]->(radial)
        CREATE (toroidal_source:StandardNameSource {
          id: 'derived:toroidal_magnetic_field', status: 'composed'
        })-[:PRODUCED_NAME]->(toroidal)
        CREATE (radial)-[:HAS_UNIT]->(unit)
        CREATE (toroidal)-[:HAS_UNIT]->(unit)
        CREATE (radial)-[:HAS_PARENT {
          operator_kind: 'projection', axis: 'radial'
        }]->(parent)
        CREATE (toroidal)-[:HAS_PARENT {
          operator_kind: 'projection', axis: 'toroidal'
        }]->(parent)
        """
    )


@pytest.mark.graph
def test_materialization_is_dry_run_safe_hash_bound_and_idempotent(
    tmp_path: Path, structural_graph: GraphClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_vector_parent(structural_graph)
    manifest, manifest_hash = _bound_manifest(
        tmp_path,
        structural_graph,
        {
            "magnetic_field": {
                "expected_actions": [MATERIALIZE_ADMISSIBLE_PARENT],
            }
        },
    )
    reads = 0
    original_read = closure._read_rows

    def counted_read(transaction: Any, roots: tuple[str, ...]):
        nonlocal reads
        reads += 1
        return original_read(transaction, roots)

    monkeypatch.setattr(closure, "_read_rows", counted_read)
    dry_receipt = closure.reconcile_structural_closure(
        manifest, dry_run=True, gc=structural_graph
    )
    assert dry_receipt["mode"] == "dry_run"
    assert dry_receipt["counts"]["planned"] == 1
    assert reads == 1
    assert structural_graph.query(
        "MATCH (parent:StandardName {id: 'magnetic_field'}) "
        "RETURN parent.name_stage AS stage"
    ) == [{"stage": None}]

    reads = 0
    applied = closure.reconcile_structural_closure(
        manifest,
        dry_run=False,
        expected_manifest_hash=manifest_hash,
        gc=structural_graph,
    )
    assert applied["mode"] == "applied"
    assert applied["counts"]["changed"] == 1
    assert reads == 3
    rows = structural_graph.query(
        """
        MATCH (source:StandardNameSource {id: 'derived:magnetic_field'})
              -[:PRODUCED_NAME]->(parent:StandardName {id: 'magnetic_field'})
        MATCH (parent)-[:HAS_UNIT]->(unit:Unit)
        RETURN parent.name_stage AS stage, parent.unit AS unit,
               unit.id AS linked_unit, count(source) AS sources
        """
    )
    assert rows == [
        {"stage": "accepted", "unit": "T", "linked_unit": "T", "sources": 1}
    ]

    repeated = closure.reconcile_structural_closure(
        manifest,
        dry_run=False,
        expected_manifest_hash=manifest_hash,
        gc=structural_graph,
    )
    assert repeated["mode"] == "already_current"
    assert repeated["counts"]["changed"] == 0


@pytest.mark.graph
def test_scaffold_exclusion_and_retirement_are_ledgered_by_lifecycle(
    tmp_path: Path, structural_graph: GraphClient
) -> None:
    structural_graph.query(
        """
        CREATE (:StandardName {id: 'temperature_over_scrape_off_layer',
                               name_stage: 'drafted', origin: 'derived'})
        CREATE (:StandardName {id: 'phase_of_fiber_optic_current_sensor',
                               origin: 'derived'})
        """
    )
    manifest, manifest_hash = _bound_manifest(
        tmp_path,
        structural_graph,
        {
            "phase_of_fiber_optic_current_sensor": {
                "expected_actions": [EXCLUDE_NULL_SCAFFOLD],
                "scaffold_ids": ["phase_of_fiber_optic_current_sensor"],
            },
            "temperature_over_scrape_off_layer": {
                "expected_actions": [RETIRE_UNREACHABLE_CHAIN],
                "retire_ids": ["temperature_over_scrape_off_layer"],
            },
        },
    )
    receipt = closure.reconcile_structural_closure(
        manifest,
        dry_run=False,
        expected_manifest_hash=manifest_hash,
        gc=structural_graph,
    )

    assert receipt["mode"] == "applied"
    assert receipt["counts"]["changed"] == 2
    assert structural_graph.query(
        "MATCH (name:StandardName) RETURN count(name) AS names"
    ) == [{"names": 0}]
    assert structural_graph.query(
        """
        MATCH (change:StandardNameChange)
        RETURN change.from_name AS name, change.operation AS operation,
               change.changed_at IS NOT NULL AS timestamped
        """
    ) == [
        {
            "name": "temperature_over_scrape_off_layer",
            "operation": "reconcile_structural_closure",
            "timestamped": True,
        }
    ]
    assert structural_graph.query(
        "MATCH (:StandardNameSource) RETURN count(*) AS sources"
    ) == [{"sources": 0}]


@pytest.mark.graph
def test_accepted_parent_source_seeding_preserves_catalog_state(
    tmp_path: Path, structural_graph: GraphClient
) -> None:
    _seed_vector_parent(structural_graph)
    structural_graph.query(
        """
        MATCH (parent:StandardName {id: 'magnetic_field'})
        SET parent.name_stage = 'accepted', parent.docs_stage = 'accepted',
            parent.description = 'Magnetic flux density.', parent.unit = 'T',
            parent.reviewer_score_name = 0.97
        """
    )
    before = structural_graph.query(
        """
        MATCH (parent:StandardName {id: 'magnetic_field'})
        RETURN properties(parent) AS properties
        """
    )[0]["properties"]
    manifest, manifest_hash = _bound_manifest(
        tmp_path,
        structural_graph,
        {
            "magnetic_field": {
                "expected_actions": [SEED_ACCEPTED_PARENT_SOURCE],
            }
        },
    )

    receipt = closure.reconcile_structural_closure(
        manifest,
        dry_run=False,
        expected_manifest_hash=manifest_hash,
        gc=structural_graph,
    )

    after = structural_graph.query(
        """
        MATCH (parent:StandardName {id: 'magnetic_field'})
        RETURN properties(parent) AS properties
        """
    )[0]["properties"]
    assert receipt["mode"] == "applied"
    assert after == before
    assert structural_graph.query(
        """
        MATCH (:StandardNameSource {id: 'derived:magnetic_field'})
              -[:PRODUCED_NAME]->(:StandardName {id: 'magnetic_field'})
        RETURN count(*) AS sources
        """
    ) == [{"sources": 1}]


class _RelationshipDriftTransaction:
    def __init__(self, transaction: Any, uri: str) -> None:
        self.transaction = transaction
        self.uri = uri
        self.drifted = False

    def run(self, cypher: str, **params: Any):
        if "SOURCE_SNAPSHOT_MIGRATION_LOCK" in cypher and not self.drifted:
            with GraphDatabase.driver(self.uri, auth=None) as driver:
                driver.execute_query(
                    """
                    MATCH (:StandardName {id: 'radial_magnetic_field'})
                          -[edge:HAS_PARENT]->(:StandardName {id: 'magnetic_field'})
                    SET edge.axis = 'vertical'
                    """
                )
            self.drifted = True
        return self.transaction.run(cypher, **params)

    def commit(self) -> None:
        self.transaction.commit()

    def rollback(self) -> None:
        self.transaction.rollback()


class _RelationshipDriftClient:
    def __init__(self, client: GraphClient, uri: str) -> None:
        self.client = client
        self.uri = uri

    @contextmanager
    def session(self):
        with self.client.session() as session:
            yield _RelationshipDriftSession(session, self.uri)


class _RelationshipDriftSession:
    def __init__(self, session: Any, uri: str) -> None:
        self.session = session
        self.uri = uri

    def begin_transaction(self) -> _RelationshipDriftTransaction:
        return _RelationshipDriftTransaction(self.session.begin_transaction(), self.uri)


@pytest.mark.graph
def test_relationship_drift_rolls_back_before_mutation(
    tmp_path: Path,
    structural_graph: GraphClient,
    ephemeral_neo4j: _EphemeralNeo4j,
) -> None:
    _seed_vector_parent(structural_graph)
    manifest, manifest_hash = _bound_manifest(
        tmp_path,
        structural_graph,
        {
            "magnetic_field": {
                "expected_actions": [MATERIALIZE_ADMISSIBLE_PARENT],
            }
        },
    )

    with pytest.raises(StructuralClosureConflict, match="changed after locks"):
        closure.reconcile_structural_closure(
            manifest,
            dry_run=False,
            expected_manifest_hash=manifest_hash,
            gc=_RelationshipDriftClient(structural_graph, ephemeral_neo4j.uri),
        )
    assert structural_graph.query(
        """
        MATCH (parent:StandardName {id: 'magnetic_field'})
        OPTIONAL MATCH (source:StandardNameSource {id: 'derived:magnetic_field'})
                       -[:PRODUCED_NAME]->(parent)
        RETURN parent.name_stage AS stage, count(source) AS sources
        """
    ) == [{"stage": None, "sources": 0}]
