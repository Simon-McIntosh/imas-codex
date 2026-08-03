"""Contracts for manifest-bound structural closure reconciliation."""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any

import pytest
from neo4j import GraphDatabase
from neo4j.time import DateTime

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


@pytest.mark.parametrize(
    ("instant", "epoch_seconds", "nanosecond"),
    [
        (datetime(1969, 12, 31, 23, 59, 59, 500_000, tzinfo=UTC), -1, 500_000_000),
        (datetime(1969, 12, 31, 23, 59, 59, tzinfo=UTC), -1, 0),
        (datetime(1970, 1, 1, tzinfo=UTC), 0, 0),
        (datetime(1970, 1, 1, 0, 0, 0, 999_999, tzinfo=UTC), 0, 999_999_000),
    ],
)
def test_event_timestamp_uses_exact_epoch_floor_semantics(
    instant: datetime, epoch_seconds: int, nanosecond: int
) -> None:
    assert closure._normalized_event_record({"changed_at": instant})["changed_at"] == {
        "epoch_seconds": epoch_seconds,
        "nanosecond": nanosecond,
    }


def test_event_hash_is_stable_across_equivalent_temporal_hydration() -> None:
    instant = datetime(2026, 8, 3, 16, 25, 40, 931156, tzinfo=UTC)
    hydrated = instant.astimezone(timezone(timedelta(hours=2)))

    assert closure._event_hash({"changed_at": instant}) == closure._event_hash(
        {"changed_at": hydrated}
    )


def test_event_timestamp_preserves_neo4j_nanosecond_precision() -> None:
    instant = DateTime(1969, 12, 31, 23, 59, 59, nanosecond=999_999_999, tzinfo=UTC)

    assert closure._normalized_event_record({"changed_at": instant})["changed_at"] == {
        "epoch_seconds": -1,
        "nanosecond": 999_999_999,
    }


def test_event_timestamp_refuses_missing_timezone_authority() -> None:
    with pytest.raises(
        StructuralClosureConflict, match="timestamp lost its timezone authority"
    ):
        closure._normalized_event_record(
            {"changed_at": datetime(1970, 1, 1, 0, 0, 0, 1)}
        )


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
        assert client.query("MATCH (node) RETURN count(node) AS nodes") == [
            {"nodes": 0}
        ]
        assert client.query(
            "MATCH ()-[relationship]->() RETURN count(relationship) AS relationships"
        ) == [{"relationships": 0}]
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


def _seed_electric_vector_parent(client: GraphClient) -> None:
    client.query(
        """
        CREATE (parent:StandardName {id: 'electric_field', origin: 'derived'})
        CREATE (radial:StandardName {
          id: 'radial_electric_field', origin: 'derived', name_stage: 'accepted',
          unit: 'V.m^-1', physics_domain: 'magnetics'
        })
        CREATE (toroidal:StandardName {
          id: 'toroidal_electric_field', origin: 'derived', name_stage: 'accepted',
          unit: 'V.m^-1', physics_domain: 'magnetics'
        })
        CREATE (unit:Unit {id: 'V.m^-1'})
        CREATE (radial_source:StandardNameSource {
          id: 'derived:radial_electric_field', status: 'composed'
        })-[:PRODUCED_NAME]->(radial)
        CREATE (toroidal_source:StandardNameSource {
          id: 'derived:toroidal_electric_field', status: 'composed'
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


def _seed_production_shaped_structural_cohort(
    client: GraphClient,
) -> dict[str, dict[str, Any]]:
    """Seed fourteen roots whose mutations emit twenty-eight ledger events."""
    _seed_vector_parent(client)
    _seed_electric_vector_parent(client)
    client.query(
        """
        CREATE (parent:StandardName {
          id: 'vacuum_magnetic_field', origin: 'derived', name_stage: 'accepted'
        })
        CREATE (radial:StandardName {
          id: 'radial_vacuum_magnetic_field', origin: 'derived',
          name_stage: 'accepted', unit: 'T'
        })
        CREATE (toroidal:StandardName {
          id: 'toroidal_vacuum_magnetic_field', origin: 'derived',
          name_stage: 'accepted', unit: 'T'
        })
        CREATE (unit:Unit {id: 'T'})
        CREATE (:StandardNameSource {
          id: 'derived:radial_vacuum_magnetic_field', status: 'composed'
        })-[:PRODUCED_NAME]->(radial)
        CREATE (:StandardNameSource {
          id: 'derived:toroidal_vacuum_magnetic_field', status: 'composed'
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
    retired_roots = [
        "count_due_to_gas_injection",
        "count_due_to_pellet_injection",
        "field_aligned_convection_velocity",
        "flux_due_to_perturbed_parallel_vector_potential",
        "flux_due_to_recycling",
        "opacity_at_ece_channel_emission_position",
        "phase_of_fiber_optic_current_sensor",
        "power_at_wall",
        "pressure_of_gyrokinetic_eigenmode",
        "temperature_over_scrape_off_layer",
        "volumetric_source_rate",
    ]
    rows = []
    for index, root_id in enumerate(retired_roots):
        scaffold_count = 2 if index < 3 else 1
        rows.append(
            {
                "root_id": root_id,
                "scaffold_ids": [
                    f"unmaterialized_operand_{word}_of_{root_id}"
                    for word in ("first", "second")[:scaffold_count]
                ],
            }
        )
    client.query(
        """
        UNWIND $rows AS row
        CREATE (root:StandardName {
          id: row.root_id, origin: 'derived', name_stage: 'drafted'
        })
        WITH root, row
        UNWIND row.scaffold_ids AS scaffold_id
        CREATE (scaffold:StandardName {id: scaffold_id, origin: 'derived'})
        CREATE (scaffold)-[:HAS_PARENT {operator_kind: 'binary'}]->(root)
        """,
        rows=rows,
    )
    specifications: dict[str, dict[str, Any]] = {
        "magnetic_field": {
            "expected_actions": [MATERIALIZE_ADMISSIBLE_PARENT],
        },
        "electric_field": {
            "expected_actions": [MATERIALIZE_ADMISSIBLE_PARENT],
        },
        "vacuum_magnetic_field": {
            "expected_actions": [SEED_ACCEPTED_PARENT_SOURCE],
        },
    }
    specifications.update(
        {
            row["root_id"]: {
                "expected_actions": [
                    EXCLUDE_NULL_SCAFFOLD,
                    RETIRE_UNREACHABLE_CHAIN,
                ],
                "retire_ids": [row["root_id"]],
                "scaffold_ids": row["scaffold_ids"],
            }
            for row in rows
        }
    )
    return specifications


class _CountingTransaction:
    def __init__(self, transaction: Any, counter: list[int]) -> None:
        self.transaction = transaction
        self.counter = counter

    def run(self, cypher: str, **params: Any):
        self.counter[0] += 1
        return self.transaction.run(cypher, **params)

    def commit(self) -> None:
        self.transaction.commit()

    def rollback(self) -> None:
        self.transaction.rollback()


class _CountingSession:
    def __init__(self, session: Any, counter: list[int]) -> None:
        self.session = session
        self.counter = counter

    def begin_transaction(self) -> _CountingTransaction:
        return _CountingTransaction(self.session.begin_transaction(), self.counter)


class _CountingClient:
    def __init__(self, client: GraphClient, counter: list[int]) -> None:
        self.client = client
        self.counter = counter

    @contextmanager
    def session(self):
        with self.client.session() as session:
            yield _CountingSession(session, self.counter)


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
    assert len(applied["events"]) == 1
    assert applied["events"][0]["action"] == MATERIALIZE_ADMISSIBLE_PARENT
    persisted_event = structural_graph.query(
        """
        MATCH (change:StandardNameChange {id: $event_id})
        RETURN properties(change) AS record
        """,
        event_id=applied["events"][0]["id"],
    )[0]["record"]
    assert closure._event_hash(persisted_event) == applied["events"][0]["hash"]
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
    assert repeated["events"] == []
    assert structural_graph.query(
        "MATCH (change:StandardNameChange) RETURN count(change) AS events"
    ) == [{"events": 1}]


@pytest.mark.graph
def test_apply_query_count_is_constant_for_single_multi_and_mixed_cohorts(
    tmp_path: Path, structural_graph: GraphClient
) -> None:
    def apply_count(specifications: dict[str, dict[str, Any]]) -> tuple[int, int]:
        manifest, manifest_hash = _bound_manifest(
            tmp_path, structural_graph, specifications
        )
        counter = [0]
        receipt = closure.reconcile_structural_closure(
            manifest,
            dry_run=False,
            expected_manifest_hash=manifest_hash,
            gc=_CountingClient(structural_graph, counter),
        )
        return counter[0], receipt["counts"]["changed"]

    _seed_vector_parent(structural_graph)
    single_count, single_changed = apply_count(
        {
            "magnetic_field": {
                "expected_actions": [MATERIALIZE_ADMISSIBLE_PARENT],
            }
        }
    )

    structural_graph.query("MATCH (node) DETACH DELETE node")
    _seed_vector_parent(structural_graph)
    _seed_electric_vector_parent(structural_graph)
    multi_count, multi_changed = apply_count(
        {
            "electric_field": {
                "expected_actions": [MATERIALIZE_ADMISSIBLE_PARENT],
            },
            "magnetic_field": {
                "expected_actions": [MATERIALIZE_ADMISSIBLE_PARENT],
            },
        }
    )

    structural_graph.query("MATCH (node) DETACH DELETE node")
    _seed_vector_parent(structural_graph)
    _seed_electric_vector_parent(structural_graph)
    structural_graph.query(
        """
        MATCH (parent:StandardName {id: 'electric_field'})
        SET parent.name_stage = 'accepted', parent.unit = 'V.m^-1'
        CREATE (:StandardName {id: 'temperature_over_scrape_off_layer',
                               name_stage: 'drafted', origin: 'derived'})
        """
    )
    mixed_count, mixed_changed = apply_count(
        {
            "electric_field": {
                "expected_actions": [SEED_ACCEPTED_PARENT_SOURCE],
            },
            "magnetic_field": {
                "expected_actions": [MATERIALIZE_ADMISSIBLE_PARENT],
            },
            "temperature_over_scrape_off_layer": {
                "expected_actions": [RETIRE_UNREACHABLE_CHAIN],
                "retire_ids": ["temperature_over_scrape_off_layer"],
            },
        }
    )

    assert (single_count, multi_count, mixed_count) == (10, 10, 10)
    assert (single_changed, multi_changed, mixed_changed) == (1, 2, 3)


@pytest.mark.graph
def test_shared_descendant_cohort_materializes_each_root_once(
    tmp_path: Path, structural_graph: GraphClient
) -> None:
    _seed_vector_parent(structural_graph)
    structural_graph.query(
        """
        MATCH (shared:StandardName {id: 'radial_magnetic_field'}),
              (unit:Unit {id: 'T'})
        CREATE (parent:StandardName {
          id: 'perturbed_magnetic_field', origin: 'derived'
        })
        CREATE (toroidal:StandardName {
          id: 'toroidal_perturbed_magnetic_field', origin: 'derived',
          name_stage: 'accepted', unit: 'T', physics_domain: 'magnetics'
        })
        CREATE (source:StandardNameSource {
          id: 'derived:toroidal_perturbed_magnetic_field', status: 'composed'
        })-[:PRODUCED_NAME]->(toroidal)
        CREATE (toroidal)-[:HAS_UNIT]->(unit)
        CREATE (shared)-[:HAS_PARENT {
          operator_kind: 'projection', axis: 'radial'
        }]->(parent)
        CREATE (toroidal)-[:HAS_PARENT {
          operator_kind: 'projection', axis: 'toroidal'
        }]->(parent)
        """
    )
    manifest, manifest_hash = _bound_manifest(
        tmp_path,
        structural_graph,
        {
            "magnetic_field": {
                "expected_actions": [MATERIALIZE_ADMISSIBLE_PARENT],
            },
            "perturbed_magnetic_field": {
                "expected_actions": [MATERIALIZE_ADMISSIBLE_PARENT],
            },
        },
    )

    receipt = closure.reconcile_structural_closure(
        manifest,
        dry_run=False,
        expected_manifest_hash=manifest_hash,
        gc=structural_graph,
    )

    assert receipt["counts"]["changed"] == 2
    assert len(receipt["events"]) == 2
    assert structural_graph.query(
        """
        MATCH (shared:StandardName {id: 'radial_magnetic_field'})
        RETURN count(shared) AS shared_nodes,
               size([(shared)-[:HAS_PARENT]->() | 1]) AS parent_count
        """
    ) == [{"shared_nodes": 1, "parent_count": 2}]


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
    event_rows = structural_graph.query(
        """
        MATCH (change:StandardNameChange)
        RETURN change.from_name AS name, change.operation AS operation,
               change.changed_at IS NOT NULL AS timestamped,
               change.action AS action
        ORDER BY name
        """
    )
    assert event_rows == [
        {
            "name": "phase_of_fiber_optic_current_sensor",
            "operation": "reconcile_structural_closure",
            "timestamped": True,
            "action": EXCLUDE_NULL_SCAFFOLD,
        },
        {
            "name": "temperature_over_scrape_off_layer",
            "operation": "reconcile_structural_closure",
            "timestamped": True,
            "action": RETIRE_UNREACHABLE_CHAIN,
        },
    ]
    assert len(receipt["events"]) == 2
    assert structural_graph.query(
        "MATCH (:StandardNameSource) RETURN count(*) AS sources"
    ) == [{"sources": 0}]


@pytest.mark.graph
def test_accepted_deletion_refuses_then_applies_only_with_authorization(
    tmp_path: Path, structural_graph: GraphClient
) -> None:
    structural_graph.query(
        """
        CREATE (:StandardName {id: 'temperature_over_scrape_off_layer',
                               name_stage: 'accepted', origin: 'derived'})
        """
    )
    manifest, manifest_hash = _bound_manifest(
        tmp_path,
        structural_graph,
        {
            "temperature_over_scrape_off_layer": {
                "expected_actions": [RETIRE_UNREACHABLE_CHAIN],
                "retire_ids": ["temperature_over_scrape_off_layer"],
            }
        },
    )

    guarded = closure.reconcile_structural_closure(
        manifest,
        dry_run=True,
        include_accepted=False,
        gc=structural_graph,
    )
    assert guarded["mode"] == "refused"
    assert structural_graph.query(
        "MATCH (:StandardName {id: 'temperature_over_scrape_off_layer'}) "
        "RETURN count(*) AS names"
    ) == [{"names": 1}]

    applied = closure.reconcile_structural_closure(
        manifest,
        dry_run=False,
        include_accepted=True,
        expected_manifest_hash=manifest_hash,
        gc=structural_graph,
    )
    assert applied["mode"] == "applied"
    assert applied["events"][0]["action"] == RETIRE_UNREACHABLE_CHAIN
    assert structural_graph.query(
        "MATCH (:StandardName {id: 'temperature_over_scrape_off_layer'}) "
        "RETURN count(*) AS names"
    ) == [{"names": 0}]


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
    assert len(receipt["events"]) == 1
    assert receipt["events"][0]["action"] == SEED_ACCEPTED_PARENT_SOURCE
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


class _ParticipantDriftTransaction:
    def __init__(self, transaction: Any, uri: str, statement: str) -> None:
        self.transaction = transaction
        self.uri = uri
        self.statement = statement
        self.drifted = False

    def run(self, cypher: str, **params: Any):
        if "SOURCE_SNAPSHOT_MIGRATION_LOCK" in cypher and not self.drifted:
            with GraphDatabase.driver(self.uri, auth=None) as driver:
                driver.execute_query(self.statement)
            self.drifted = True
        return self.transaction.run(cypher, **params)

    def commit(self) -> None:
        self.transaction.commit()

    def rollback(self) -> None:
        self.transaction.rollback()


class _ParticipantDriftSession:
    def __init__(self, session: Any, uri: str, statement: str) -> None:
        self.session = session
        self.uri = uri
        self.statement = statement

    def begin_transaction(self) -> _ParticipantDriftTransaction:
        return _ParticipantDriftTransaction(
            self.session.begin_transaction(), self.uri, self.statement
        )


class _ParticipantDriftClient:
    def __init__(self, client: GraphClient, uri: str, statement: str) -> None:
        self.client = client
        self.uri = uri
        self.statement = statement

    @contextmanager
    def session(self):
        with self.client.session() as session:
            yield _ParticipantDriftSession(session, self.uri, self.statement)


class _EventTamperTransaction:
    def __init__(self, transaction: Any) -> None:
        self.transaction = transaction
        self.tampered = False

    def run(self, cypher: str, **params: Any):
        if "STRUCTURAL_CLOSURE_EVENT_POSTFLIGHT" in cypher and not self.tampered:
            list(
                self.transaction.run(
                    """
                    MATCH (change:StandardNameChange)
                    WHERE change.id IN $event_ids
                    SET change.reason = 'tampered inside transaction'
                    """,
                    event_ids=params["event_ids"],
                )
            )
            self.tampered = True
        return self.transaction.run(cypher, **params)

    def commit(self) -> None:
        self.transaction.commit()

    def rollback(self) -> None:
        self.transaction.rollback()


class _EventTamperSession:
    def __init__(self, session: Any) -> None:
        self.session = session

    def begin_transaction(self) -> _EventTamperTransaction:
        return _EventTamperTransaction(self.session.begin_transaction())


class _EventTamperClient:
    def __init__(self, client: GraphClient) -> None:
        self.client = client

    @contextmanager
    def session(self):
        with self.client.session() as session:
            yield _EventTamperSession(session)


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


@pytest.mark.graph
@pytest.mark.parametrize("participant", ["unit", "dd"])
def test_claimed_unit_and_dd_participants_refuse_without_mutation(
    tmp_path: Path, structural_graph: GraphClient, participant: str
) -> None:
    _seed_vector_parent(structural_graph)
    if participant == "unit":
        structural_graph.query(
            "MATCH (unit:Unit {id: 'T'}) SET unit.claimed_at = datetime()"
        )
    else:
        structural_graph.query(
            """
            MATCH (child:StandardName {id: 'radial_magnetic_field'})
            CREATE (node:IMASNode {
              id: 'equilibrium/time_slice/magnetic_field/radial',
              claimed_at: datetime()
            })-[:HAS_STANDARD_NAME]->(child)
            """
        )
    manifest, _ = _bound_manifest(
        tmp_path,
        structural_graph,
        {
            "magnetic_field": {
                "expected_actions": [MATERIALIZE_ADMISSIBLE_PARENT],
            }
        },
    )

    receipt = closure.reconcile_structural_closure(
        manifest, dry_run=True, gc=structural_graph
    )

    assert receipt["mode"] == "refused"
    assert any(
        "active structural closure claims" in reason
        for reason in receipt["rows"][0]["unresolved"]
    )
    assert receipt["events"] == []


@pytest.mark.graph
def test_generic_protected_participant_property_refuses_without_mutation(
    tmp_path: Path, structural_graph: GraphClient
) -> None:
    _seed_vector_parent(structural_graph)
    structural_graph.query("MATCH (unit:Unit {id: 'T'}) SET unit.facility_id = 'west'")
    rows = _snapshot(structural_graph, ["magnetic_field"])
    manifest_row = build_structural_closure_manifest_row(
        rows[0],
        root_id="magnetic_field",
        expected_actions=[MATERIALIZE_ADMISSIBLE_PARENT],
        reason="refuse protected structural participants",
    )

    assert manifest_row["west_intersection"] == 1


@pytest.mark.graph
@pytest.mark.parametrize(
    "statement",
    [
        "MATCH (unit:Unit {id: 'T'}) SET unit.claimed_at = datetime()",
        "MATCH (node:IMASNode {id: 'equilibrium/time_slice/magnetic_field/radial'}) "
        "SET node.claimed_at = datetime()",
        "MATCH (unit:Unit {id: 'T'}) SET unit.facility_id = 'west'",
    ],
)
def test_unit_claim_and_generic_property_drift_roll_back_the_cohort(
    tmp_path: Path,
    structural_graph: GraphClient,
    ephemeral_neo4j: _EphemeralNeo4j,
    statement: str,
) -> None:
    _seed_vector_parent(structural_graph)
    structural_graph.query(
        """
        MATCH (child:StandardName {id: 'radial_magnetic_field'})
        CREATE (:IMASNode {
          id: 'equilibrium/time_slice/magnetic_field/radial'
        })-[:HAS_STANDARD_NAME]->(child)
        """
    )
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
            gc=_ParticipantDriftClient(
                structural_graph, ephemeral_neo4j.uri, statement
            ),
        )
    assert structural_graph.query(
        "MATCH (change:StandardNameChange) RETURN count(change) AS events"
    ) == [{"events": 0}]
    assert structural_graph.query(
        "MATCH (:StandardNameSource {id: 'derived:magnetic_field'}) "
        "RETURN count(*) AS sources"
    ) == [{"sources": 0}]


@pytest.mark.graph
def test_admission_drift_on_retirement_target_rolls_back_whole_cohort(
    tmp_path: Path,
    structural_graph: GraphClient,
    ephemeral_neo4j: _EphemeralNeo4j,
) -> None:
    structural_graph.query(
        """
        CREATE (root:StandardName {id: 'temperature_over_scrape_off_layer',
                                   name_stage: 'drafted', origin: 'derived'})
        CREATE (child:StandardName {
          id: 'average_temperature_over_scrape_off_layer',
          name_stage: 'drafted', origin: 'derived'
        })-[:HAS_PARENT {operator_kind: 'qualifier'}]->(root)
        """
    )
    targets = [
        "average_temperature_over_scrape_off_layer",
        "temperature_over_scrape_off_layer",
    ]
    manifest, manifest_hash = _bound_manifest(
        tmp_path,
        structural_graph,
        {
            "temperature_over_scrape_off_layer": {
                "expected_actions": [RETIRE_UNREACHABLE_CHAIN],
                "retire_ids": targets,
            }
        },
    )
    statement = """
        MATCH (:StandardName {id: 'average_temperature_over_scrape_off_layer'})
              -[edge:HAS_PARENT]->(:StandardName)
        SET edge.operator_kind = 'binary'
    """

    with pytest.raises(StructuralClosureConflict, match="changed after locks"):
        closure.reconcile_structural_closure(
            manifest,
            dry_run=False,
            expected_manifest_hash=manifest_hash,
            gc=_ParticipantDriftClient(
                structural_graph, ephemeral_neo4j.uri, statement
            ),
        )
    remaining_ids = structural_graph.query(
        "MATCH (name:StandardName) RETURN collect(name.id) AS ids"
    )[0]["ids"]
    assert sorted(remaining_ids) == sorted(targets)
    assert structural_graph.query(
        "MATCH (change:StandardNameChange) RETURN count(change) AS events"
    ) == [{"events": 0}]


@pytest.mark.graph
def test_event_record_tamper_fails_hash_postflight_and_rolls_back(
    tmp_path: Path, structural_graph: GraphClient
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

    with pytest.raises(StructuralClosureConflict, match="record hash changed"):
        closure.reconcile_structural_closure(
            manifest,
            dry_run=False,
            expected_manifest_hash=manifest_hash,
            gc=_EventTamperClient(structural_graph),
        )

    assert structural_graph.query(
        "MATCH (change:StandardNameChange) RETURN count(change) AS events"
    ) == [{"events": 0}]
    assert structural_graph.query(
        """
        MATCH (parent:StandardName {id: 'magnetic_field'})
        OPTIONAL MATCH (source:StandardNameSource {id: 'derived:magnetic_field'})
                       -[:PRODUCED_NAME]->(parent)
        RETURN parent.name_stage AS stage, count(source) AS sources
        """
    ) == [{"stage": None, "sources": 0}]


@pytest.mark.graph
def test_production_shaped_event_batch_round_trips_and_tamper_rolls_back(
    tmp_path: Path, structural_graph: GraphClient
) -> None:
    specifications = _seed_production_shaped_structural_cohort(structural_graph)
    manifest, manifest_hash = _bound_manifest(
        tmp_path, structural_graph, specifications
    )

    applied = closure.reconcile_structural_closure(
        manifest,
        dry_run=False,
        expected_manifest_hash=manifest_hash,
        gc=structural_graph,
    )

    assert applied["counts"]["allowlisted"] == 14
    assert applied["counts"]["changed"] == 28
    assert len(applied["events"]) == 28
    persisted = structural_graph.query(
        """
        MATCH (change:StandardNameChange)
        RETURN change.id AS id, properties(change) AS record
        ORDER BY id
        """
    )
    assert {row["id"]: closure._event_hash(row["record"]) for row in persisted} == {
        event["id"]: event["hash"] for event in applied["events"]
    }

    structural_graph.query("MATCH (node) DETACH DELETE node")
    specifications = _seed_production_shaped_structural_cohort(structural_graph)
    manifest, manifest_hash = _bound_manifest(
        tmp_path, structural_graph, specifications
    )
    with pytest.raises(StructuralClosureConflict, match="record hash changed"):
        closure.reconcile_structural_closure(
            manifest,
            dry_run=False,
            expected_manifest_hash=manifest_hash,
            gc=_EventTamperClient(structural_graph),
        )
    assert structural_graph.query(
        "MATCH (change:StandardNameChange) RETURN count(change) AS events"
    ) == [{"events": 0}]
