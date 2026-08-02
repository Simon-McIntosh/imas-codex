"""Stateful checks for atomic DD-gap identity reclassification."""

from __future__ import annotations

import os
from collections.abc import Iterator
from datetime import UTC, datetime
from uuid import uuid4

import pytest
from neo4j import GraphDatabase
from neo4j.exceptions import ConstraintError

from imas_codex.graph.schema import get_schema
from imas_codex.settings import get_graph_uri
from imas_codex.standard_names.dd_gaps import (
    _RECLASSIFY_REGISTRY_FACT_QUERY,
    _REGISTRY_FACTS_QUERY,
    _SYNC_REGISTRY_QUERY,
    _observation_id,
    _registry_migration_parameters,
    _registry_sync_plan,
)

pytestmark = pytest.mark.graph


@pytest.fixture(scope="module")
def ephemeral_driver() -> Iterator:
    uri = os.environ.get("IMAS_CODEX_TEST_NEO4J_URI")
    if not uri:
        pytest.skip("IMAS_CODEX_TEST_NEO4J_URI is not configured")
    if os.environ.get("IMAS_CODEX_TEST_NEO4J_EPHEMERAL") != "1":
        pytest.fail("stateful DD-gap tests require an explicitly ephemeral graph")
    project_uri = os.environ.get("IMAS_CODEX_TEST_PROJECT_NEO4J_URI") or get_graph_uri()
    if uri == project_uri:
        pytest.fail("stateful DD-gap tests refuse the configured project graph")

    driver = GraphDatabase.driver(uri, auth=None)
    driver.verify_connectivity()
    schema = get_schema()
    statements = {
        label: next(
            statement
            for statement in schema.constraint_statements()
            if f"FOR (n:{label})" in statement
        )
        for label in ("DDGap", "DDGapObservation", "DDGapIdentityChange")
    }
    with driver.session() as session:
        for statement in statements.values():
            session.run(statement).consume()
    try:
        yield driver
    finally:
        driver.close()


def _case() -> dict[str, object]:
    namespace = uuid4().hex
    path = f"test/{namespace}/*"
    source_path = f"test/{namespace}/one"
    old_id = f"dd_gap:{path}:unit_defect"
    new_id = f"dd_gap:{path}:self_contradiction"
    observed_at = datetime(2026, 8, 2, 10, 15, tzinfo=UTC).isoformat()
    observation = {
        "gap_id": old_id,
        "source_path": source_path,
        "reason": "the declared unit conflicts with the governed registry",
        "reporter": "registry_backfill",
        "observed_at": observed_at,
        "observed_dd_version": "4.1.0",
        "observed_value": "1",
        "expected_value": "Pa",
        "evidence_rule": "unit_equals_expected",
        "reference_path": None,
        "reference_value": None,
    }
    observation["observation_id"] = _observation_id(observation)
    target_observation = {
        **observation,
        "gap_id": new_id,
    }
    target_observation["observation_id"] = _observation_id(target_observation)
    target = {
        "id": new_id,
        "path": path,
        "kind": "self_contradiction",
        "status": "registered_exception",
        "registry_backend": "dd_unit_exceptions",
        "affected_path_count": 1,
        "triaged_at": observed_at,
        "observed_dd_version": "4.1.0",
        "observed_value": "1",
        "expected_value": "Pa",
        "evidence_rule": "unit_equals_expected",
        "upstream_url": None,
    }
    return {
        "namespace": namespace,
        "path": path,
        "source_path": source_path,
        "old_id": old_id,
        "new_id": new_id,
        "observed_at": observed_at,
        "observation": observation,
        "target_observation": target_observation,
        "target": target,
    }


def _seed_fact(tx, case: dict[str, object]) -> None:
    tx.run(
        """
        CREATE (source:IMASNode {id: $source_path})
        CREATE (name:StandardName {id: $name_id})
        CREATE (source)-[:HAS_STANDARD_NAME {confidence: 0.91}]->(name)
        CREATE (source_record:StandardNameSource {
            id: $source_record_id,
            source_type: 'dd',
            source_id: $source_path,
            status: 'attached'
        })
        CREATE (source_record)-[:PRODUCED_NAME {selected: true}]->(name)
        CREATE (gap:DDGap {
            id: $old_id,
            path: $path,
            kind: 'unit_defect',
            status: 'registered_exception',
            registry_backend: 'dd_unit_exceptions',
            affected_path_count: 1,
            example_count: 1,
            first_seen_at: datetime($observed_at),
            last_seen_at: datetime($observed_at),
            triaged_at: datetime($observed_at),
            observed_dd_version: '4.1.0',
            observed_value: '1',
            expected_value: 'Pa',
            evidence_rule: 'unit_equals_expected'
        })
        CREATE (source)-[:HAS_DD_GAP {
            reason: 'the declared unit conflicts with the governed registry',
            reporter: 'registry_backfill',
            first_observed_at: datetime($observed_at),
            last_observed_at: datetime($observed_at)
        }]->(gap)
        CREATE (observation:DDGapObservation {
            id: $observation_id,
            dd_gap_id: $old_id,
            source_path: $source_path,
            reason: 'the declared unit conflicts with the governed registry',
            reporter: 'registry_backfill',
            observed_dd_version: '4.1.0',
            observed_value: '1',
            expected_value: 'Pa',
            evidence_rule: 'unit_equals_expected',
            first_observed_at: datetime($observed_at),
            last_observed_at: datetime($observed_at)
        })
        CREATE (gap)-[:HAS_OBSERVATION {ordinal: 1}]->(observation)
        CREATE (state:DDGapStateChange {
            id: $state_id,
            dd_gap_id: $old_id,
            from_status: 'triaged',
            to_status: 'registered_exception',
            actor: 'registry-owner',
            reason: 'the registry governs this exception',
            changed_at: datetime($observed_at)
        })
        CREATE (gap)-[:HAS_STATE_CHANGE {source: 'curation'}]->(state)
        CREATE (prior:DDGapIdentityChange {
            id: $prior_identity_id,
            dd_gap_id: $old_id,
            old_id: $predecessor_id,
            new_id: $old_id,
            old_kind: 'semantic_gap',
            new_kind: 'unit_defect',
            changed_at: datetime($observed_at),
            changed_by: 'registry-sync',
            reason: 'an earlier governed classification changed'
        })
        CREATE (gap)-[:HAS_IDENTITY_CHANGE {sequence: 1}]->(prior)
        """,
        source_path=case["source_path"],
        name_id=f"standard_name:{case['namespace']}",
        source_record_id=f"dd:{case['source_path']}",
        old_id=case["old_id"],
        path=case["path"],
        observed_at=case["observed_at"],
        observation_id=case["observation"]["observation_id"],
        state_id=f"dd_gap_state_change:{case['namespace']}",
        prior_identity_id=f"dd_gap_identity_change:{case['namespace']}",
        predecessor_id=f"dd_gap:{case['path']}:semantic_gap",
    ).consume()


def _fact_snapshot(tx, gap_id: str) -> dict[str, object]:
    return next(
        dict(row) for row in tx.run(_REGISTRY_FACTS_QUERY) if row["id"] == gap_id
    )


def test_reclassification_preserves_fact_topology_and_identity_history(
    ephemeral_driver,
) -> None:
    case = _case()
    with ephemeral_driver.session() as session:
        tx = session.begin_transaction()
        try:
            _seed_fact(tx, case)
            old = _fact_snapshot(tx, str(case["old_id"]))
            plan = _registry_sync_plan(
                [case["target"]],
                [case["target_observation"]],
                [old],
            )

            migrated = list(
                tx.run(
                    _RECLASSIFY_REGISTRY_FACT_QUERY,
                    **_registry_migration_parameters(plan["reclassify"][0]),
                )
            )
            assert len(migrated) == 1
            tx.run(
                _SYNC_REGISTRY_QUERY,
                nodes=[case["target"]],
                observations=[case["target_observation"]],
            ).consume()

            result = tx.run(
                """
                MATCH (gap:DDGap {id: $new_id})
                OPTIONAL MATCH (source:IMASNode)-[path_link:HAS_DD_GAP]->(gap)
                WITH gap, source, path_link
                OPTIONAL MATCH (gap)-[:HAS_OBSERVATION]->(observation)
                WITH gap, source, path_link, collect(DISTINCT observation) AS observations
                OPTIONAL MATCH (gap)-[:HAS_STATE_CHANGE]->(state)
                WITH gap, source, path_link, observations,
                     collect(DISTINCT state) AS states
                OPTIONAL MATCH (gap)-[:HAS_IDENTITY_CHANGE]->(identity)
                WITH gap, source, path_link, observations, states,
                     collect(DISTINCT identity) AS identities
                OPTIONAL MATCH (source)-[direct:HAS_STANDARD_NAME]->(:StandardName)
                OPTIONAL MATCH (:StandardNameSource {source_id: source.id})
                               -[produced:PRODUCED_NAME]->(:StandardName)
                RETURN properties(gap) AS gap,
                       properties(path_link) AS path_link,
                       [item IN observations | properties(item)] AS observations,
                       [item IN states | properties(item)] AS states,
                       [item IN identities | properties(item)] AS identities,
                       count(DISTINCT direct) AS direct_name_links,
                       count(DISTINCT produced) AS source_name_links
                """,
                new_id=case["new_id"],
            ).single(strict=True)

            assert result["gap"]["status"] == "registered_exception"
            assert result["gap"]["id"] == case["new_id"]
            assert result["path_link"]["reporter"] == "registry_backfill"
            assert len(result["observations"]) == 1
            assert result["observations"][0]["dd_gap_id"] == case["new_id"]
            assert (
                result["observations"][0]["id"]
                == case["target_observation"]["observation_id"]
            )
            assert len(result["states"]) == 1
            assert result["states"][0]["dd_gap_id"] == case["old_id"]
            assert len(result["identities"]) == 2
            new_events = [
                item
                for item in result["identities"]
                if item["new_id"] == case["new_id"]
            ]
            assert len(new_events) == 1
            assert new_events[0]["old_id"] == case["old_id"]
            assert new_events[0]["old_kind"] == "unit_defect"
            assert new_events[0]["new_kind"] == "self_contradiction"
            assert result["direct_name_links"] == 1
            assert result["source_name_links"] == 1
            assert (
                tx.run(
                    "MATCH (gap:DDGap {id: $old_id}) RETURN count(gap) AS count",
                    old_id=case["old_id"],
                ).single(strict=True)["count"]
                == 0
            )
        finally:
            tx.rollback()

    with ephemeral_driver.session() as session:
        assert (
            session.run(
                "MATCH (node) WHERE node.id CONTAINS $namespace RETURN count(node) AS count",
                namespace=str(case["namespace"]),
            ).single(strict=True)["count"]
            == 0
        )


def test_reclassification_rejects_relationship_property_race(ephemeral_driver) -> None:
    case = _case()
    with ephemeral_driver.session() as session:
        tx = session.begin_transaction()
        try:
            _seed_fact(tx, case)
            old = _fact_snapshot(tx, str(case["old_id"]))
            plan = _registry_sync_plan(
                [case["target"]],
                [case["target_observation"]],
                [old],
            )
            tx.run(
                """
                MATCH (:IMASNode)-[link:HAS_DD_GAP]->(:DDGap {id: $old_id})
                SET link.reporter = 'concurrent-writer'
                """,
                old_id=case["old_id"],
            ).consume()

            migrated = list(
                tx.run(
                    _RECLASSIFY_REGISTRY_FACT_QUERY,
                    **_registry_migration_parameters(plan["reclassify"][0]),
                )
            )
            assert migrated == []
            assert (
                tx.run(
                    "MATCH (gap:DDGap {id: $old_id}) RETURN count(gap) AS count",
                    old_id=case["old_id"],
                ).single(strict=True)["count"]
                == 1
            )
        finally:
            tx.rollback()


def test_schema_constraint_rejects_duplicate_ddgap_identity(ephemeral_driver) -> None:
    duplicate_id = f"dd_gap:test/{uuid4().hex}:unit_defect"
    with ephemeral_driver.session() as session:
        tx = session.begin_transaction()
        try:
            tx.run("CREATE (:DDGap {id: $id})", id=duplicate_id).consume()
            with pytest.raises(ConstraintError):
                tx.run("CREATE (:DDGap {id: $id})", id=duplicate_id).consume()
        finally:
            tx.rollback()
