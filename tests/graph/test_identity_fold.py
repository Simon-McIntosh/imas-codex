"""Disposable-Neo4j checks for atomic standard-name identity folds."""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from typing import Any
from unittest.mock import patch

import pytest
from neo4j import GraphDatabase

from imas_codex.graph.client import GraphClient
from imas_codex.settings import get_graph_uri
from imas_codex.standard_names import edit

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
        pytest.fail("stateful identity-fold tests require an ephemeral graph")
    project_uri = os.environ.get("IMAS_CODEX_TEST_PROJECT_NEO4J_URI") or get_graph_uri()
    if uri == project_uri:
        pytest.fail("identity-fold tests refuse the configured project graph")
    driver = GraphDatabase.driver(uri, auth=None)
    driver.verify_connectivity()
    try:
        yield _EphemeralNeo4j(uri)
    finally:
        driver.close()


class _InjectedTransaction:
    def __init__(self, transaction, uri: str, failure: str | None) -> None:
        self._transaction = transaction
        self._uri = uri
        self._failure = failure

    def _commit_external_race(self, cypher: str) -> None:
        driver = GraphDatabase.driver(
            self._uri,
            auth=None,
            connection_timeout=5,
            connection_acquisition_timeout=5,
        )
        try:
            with driver.session() as session:
                transaction = session.begin_transaction(timeout=5)
                try:
                    transaction.run(cypher).consume()
                    transaction.commit()
                except BaseException:
                    transaction.rollback()
                    raise
        finally:
            driver.close()

    def run(self, cypher: str, **params: Any):
        if "ATOMIC_FOLD_EVENT" in cypher and self._failure == "event":
            list(self._transaction.run(cypher, **params))
            raise RuntimeError("injected event failure")
        if "ATOMIC_FOLD_LOCK" in cypher and self._failure == "competitor_race":
            self._commit_external_race(
                "MATCH (name:StandardName {id: 'retired_density'}) "
                "SET name.name_stage = 'accepted'"
            )
        if "ATOMIC_FOLD_LOCK" in cypher and self._failure == "unit_race":
            self._commit_external_race(
                "MATCH (unit:Unit {id: 'm^-3'}) SET unit.symbol = 'changed'"
            )
        if "ATOMIC_FOLD_LOCK" in cypher and self._failure == "temporal_type_race":
            self._commit_external_race(
                "MATCH (name:StandardName {id: 'electron_density'}) "
                "SET name.created_at = "
                "'2026-07-04T21:20:38.632000000+00:00'"
            )
        if "ATOMIC_FOLD_MOVE_SOURCES" in cypher and self._failure == "partial":
            list(self._transaction.run(cypher, **params))
            raise RuntimeError("injected partial source migration")
        return self._transaction.run(cypher, **params)

    def commit(self) -> None:
        self._transaction.commit()

    def rollback(self) -> None:
        self._transaction.rollback()


class _InjectedSession:
    def __init__(self, session, uri: str, failure: str | None) -> None:
        self._session = session
        self._uri = uri
        self._failure = failure

    def begin_transaction(self) -> _InjectedTransaction:
        return _InjectedTransaction(
            self._session.begin_transaction(), self._uri, self._failure
        )


class _InjectedGraphClient:
    def __init__(self, uri: str, failure: str | None) -> None:
        self._uri = uri
        self._client = GraphClient(
            uri=uri,
            username="neo4j",
            password="",
            graph_name="ephemeral-identity-fold",
        )
        self._failure = failure

    def __enter__(self) -> _InjectedGraphClient:
        return self

    def __exit__(self, *_args: Any) -> None:
        self._client.close()

    @contextmanager
    def session(self) -> Iterator[_InjectedSession]:
        with self._client.session() as session:
            yield _InjectedSession(session, self._uri, self._failure)


def _seed(driver, *, third_stage: str | None = None) -> None:
    with driver.session() as session:
        session.run("MATCH (node) DETACH DELETE node").consume()
        session.run(
            """
            CREATE (old:StandardName {
              id: 'invalid_duplicate', name_stage: 'accepted',
              validation_status: 'quarantined', unit: 'm^-3',
              source_paths: ['test:signal:density'], claim_token: null,
              claimed_at: null
            })
            CREATE (target:StandardName {
              id: 'electron_density', name_stage: 'accepted',
              validation_status: 'valid', unit: 'm^-3', source_paths: [],
              created_at: datetime('2026-07-04T21:20:38.632000000+00:00'),
              claim_token: null, claimed_at: null
            })
            CREATE (unit:Unit {id: 'm^-3', symbol: 'm^-3'})
            CREATE (target)-[:HAS_UNIT]->(unit)
            CREATE (signal:FacilitySignal {
              id: 'test:signal:density', unit: 'm^-3'
            })
            CREATE (signal)-[:HAS_UNIT]->(unit)
            CREATE (source:StandardNameSource {
              id: 'signals:test:density', source_type: 'signals',
              source_id: 'test:signal:density', status: 'composed',
              produced_sn_id: old.id, claim_token: null, claimed_at: null
            })
            CREATE (source)-[:FROM_SIGNAL]->(signal)
            CREATE (source)-[:PRODUCED_NAME]->(old)
            CREATE (signal)-[:HAS_STANDARD_NAME]->(old)
            WITH old, source, signal
            FOREACH (_ IN CASE WHEN $third_stage IS NULL THEN [] ELSE [1] END |
              CREATE (third:StandardName {
                id: 'retired_density', name_stage: $third_stage,
                validation_status: 'valid', unit: 'm^-3'
              })
              CREATE (source)-[:PRODUCED_NAME]->(third)
              CREATE (signal)-[:HAS_STANDARD_NAME]->(third)
            )
            """,
            third_stage=third_stage,
        ).consume()


def _seed_vorticity_shape(driver) -> None:
    with driver.session() as session:
        session.run("MATCH (node) DETACH DELETE node").consume()
        session.run(
            """
            CREATE (old:StandardName {
              id: 'vorticity_due_to_diamagnetic_drift_magnitude',
              name_stage: 'accepted', validation_status: 'quarantined',
              unit: 's^-1', source_paths: [
                'dd:plasma_profiles/ggd/vorticity/diamagnetic'
              ], claim_token: null, claimed_at: null,
              generated_at: datetime(
                '2026-07-27T14:18:52.914000000+00:00'
              ),
              created_at: datetime('2026-07-27T14:18:52.799000000+00:00'),
              embedded_at: datetime('2026-07-27T14:19:54.204000000+00:00'),
              reviewed_name_at: datetime(
                '2026-07-28T05:40:36.589000000+00:00'
              )
            })
            CREATE (target:StandardName {
              id: 'vorticity_due_to_diamagnetic_drift',
              name_stage: 'accepted', validation_status: 'valid', unit: 's^-1',
              source_paths: [
                'dd:edge_profiles/ggd/vorticity/diamagnetic',
                'dd:plasma_profiles/ggd/vorticity/diamagnetic'
              ], claim_token: null, claimed_at: null,
              imported_at: datetime('2026-07-04T21:21:17.079000000+00:00'),
              reviewed_docs_at: datetime(
                '2026-07-20T07:02:54.106000000+00:00'
              ),
              created_at: datetime('2026-07-04T21:20:38.632000000+00:00'),
              docs_generated_at: datetime(
                '2026-07-20T07:01:14.686000000+00:00'
              ),
              embedded_at: datetime('2026-07-05T16:15:21.078000000+00:00'),
              validated_at: datetime('2026-07-20T20:11:30.277000000+00:00')
            })
            CREATE (unit:Unit {id: 's^-1', symbol: 's^-1'})
            CREATE (old)-[:HAS_UNIT]->(unit)
            CREATE (target)-[:HAS_UNIT]->(unit)
            CREATE (plasma:IMASNode {
              id: 'plasma_profiles/ggd/vorticity/diamagnetic',
              unit: 's^-1', standard_name_id: target.id,
              node_category: 'quantity', active: true, shape: [1, 2]
            })
            CREATE (edge:IMASNode {
              id: 'edge_profiles/ggd/vorticity/diamagnetic',
              unit: 's^-1', standard_name_id: target.id,
              node_category: 'quantity', active: true, shape: [1, 2]
            })
            CREATE (plasma)-[:HAS_UNIT]->(unit)
            CREATE (edge)-[:HAS_UNIT]->(unit)
            CREATE (plasma_source:StandardNameSource {
              id: 'dd:plasma_profiles/ggd/vorticity/diamagnetic',
              source_type: 'dd',
              source_id: 'plasma_profiles/ggd/vorticity/diamagnetic',
              status: 'composed', produced_sn_id: target.id,
              attempt_count: 1, dd_snapshot_pinned: true,
              claim_token: null, claimed_at: null
            })
            CREATE (edge_source:StandardNameSource {
              id: 'dd:edge_profiles/ggd/vorticity/diamagnetic',
              source_type: 'dd',
              source_id: 'edge_profiles/ggd/vorticity/diamagnetic',
              status: 'composed', produced_sn_id: target.id,
              attempt_count: 1, dd_snapshot_pinned: true,
              claim_token: null, claimed_at: null
            })
            CREATE (plasma_source)-[:FROM_DD_PATH]->(plasma)
            CREATE (edge_source)-[:FROM_DD_PATH]->(edge)
            CREATE (plasma_source)-[:PRODUCED_NAME]->(old)
            CREATE (plasma_source)-[:PRODUCED_NAME]->(target)
            CREATE (edge_source)-[:PRODUCED_NAME]->(target)
            CREATE (plasma)-[:HAS_STANDARD_NAME]->(old)
            CREATE (plasma)-[:HAS_STANDARD_NAME]->(target)
            CREATE (edge)-[:HAS_STANDARD_NAME]->(target)
            """
        ).consume()


def _state(driver) -> dict[str, Any]:
    with driver.session() as session:
        nodes = [
            dict(record)
            for record in session.run(
                "MATCH (node) RETURN labels(node) AS labels, properties(node) AS props "
                "ORDER BY node.id"
            )
        ]
        relationships = [
            dict(record)
            for record in session.run(
                "MATCH (start)-[relationship]->(end) "
                "RETURN start.id AS start, type(relationship) AS type, "
                "end.id AS end, properties(relationship) AS props "
                "ORDER BY start, type, end"
            )
        ]
    return {"nodes": nodes, "relationships": relationships}


def _state_after_external_race(before: dict[str, Any], failure: str) -> dict[str, Any]:
    expected = deepcopy(before)
    if failure == "competitor_race":
        for node in expected["nodes"]:
            if node["props"].get("id") == "retired_density":
                node["props"]["name_stage"] = "accepted"
                break
    elif failure == "unit_race":
        for node in expected["nodes"]:
            if node["props"].get("id") == "m^-3":
                node["props"]["symbol"] = "changed"
                break
    elif failure == "temporal_type_race":
        for node in expected["nodes"]:
            if node["props"].get("id") == "electron_density":
                node["props"]["created_at"] = "2026-07-04T21:20:38.632000000+00:00"
                break
    return expected


def _invoke(
    instance: _EphemeralNeo4j,
    *,
    failure: str | None = None,
    old: str = "invalid_duplicate",
    into: str = "electron_density",
) -> dict[str, Any]:
    with (
        patch.object(edit, "GraphClient", side_effect=lambda: instance.client(failure)),
        patch.object(edit, "_isn_round_trip_ok", return_value=(True, None)),
    ):
        return edit.supersede_into(old, into)


def test_third_live_competitor_refuses_without_state_change(ephemeral_neo4j) -> None:
    with ephemeral_neo4j.driver() as driver:
        _seed(driver, third_stage="accepted")
        before = _state(driver)
        result = _invoke(ephemeral_neo4j)
        assert result["ok"] is False
        assert "third live" in result["reason"]
        assert _state(driver) == before


def test_live_shaped_temporal_properties_fold_atomically(ephemeral_neo4j) -> None:
    old = "vorticity_due_to_diamagnetic_drift_magnitude"
    target = "vorticity_due_to_diamagnetic_drift"
    with ephemeral_neo4j.driver() as driver:
        _seed_vorticity_shape(driver)
        with driver.session() as session:
            properties = session.run(
                """
                MATCH (old:StandardName {id: $old}),
                      (target:StandardName {id: $target})
                RETURN properties(old) AS old, properties(target) AS target
                """,
                old=old,
                target=target,
            ).single(strict=True)
            legacy_old_matches = session.run(
                "MATCH (name:StandardName {id: $id}) "
                "WHERE properties(name) = $properties RETURN count(name) AS matches",
                id=old,
                properties=properties["old"],
            ).single(strict=True)["matches"]
            legacy_target_matches = session.run(
                "MATCH (name:StandardName {id: $id}) "
                "WHERE properties(name) = $properties RETURN count(name) AS matches",
                id=target,
                properties=properties["target"],
            ).single(strict=True)["matches"]
        assert (legacy_old_matches, legacy_target_matches) == (0, 0)
        result = _invoke(ephemeral_neo4j, old=old, into=target)
        assert result["ok"] is True
        assert result["sources_carried"] == 1
        assert result["projections_carried"] == 1
        repeated = _invoke(ephemeral_neo4j, old=old, into=target)
        assert repeated["already_superseded"] is True
        with driver.session() as session:
            row = session.run(
                """
                MATCH (old:StandardName {id: $old}),
                      (target:StandardName {id: $target})
                CALL {
                  OPTIONAL MATCH (change:StandardNameChange {
                    operation: 'fold_identity', from_name: $old, to_name: $target
                  })
                  RETURN count(change) AS events
                }
                CALL {
                  OPTIONAL MATCH (:StandardName)-[link:HAS_INTERNAL_CHANGE]
                                 ->(:StandardNameChange {
                    operation: 'fold_identity', from_name: $old, to_name: $target
                  })
                  RETURN count(link) AS event_owner_edges
                }
                RETURN old.name_stage AS old_stage,
                       target.source_paths AS target_paths,
                       valueType(target.created_at) AS created_at_type,
                       events, event_owner_edges
                """,
                old=old,
                target=target,
            ).single(strict=True)
        assert dict(row) == {
            "old_stage": "superseded",
            "target_paths": [
                "dd:edge_profiles/ggd/vorticity/diamagnetic",
                "dd:plasma_profiles/ggd/vorticity/diamagnetic",
            ],
            "created_at_type": "ZONED DATETIME NOT NULL",
            "events": 1,
            "event_owner_edges": 2,
        }


@pytest.mark.parametrize(
    ("failure", "message"),
    [
        ("event", "event failure"),
        ("partial", "partial source migration"),
        ("competitor_race", "changed after preflight"),
        ("unit_race", "changed after preflight"),
        ("temporal_type_race", "changed after preflight"),
    ],
)
def test_failures_roll_back_the_complete_graph(
    ephemeral_neo4j, failure: str, message: str
) -> None:
    with ephemeral_neo4j.driver() as driver:
        _seed(
            driver,
            third_stage=(
                "superseded" if failure in {"competitor_race", "unit_race"} else None
            ),
        )
        before = _state(driver)
        with pytest.raises(RuntimeError, match=message):
            _invoke(ephemeral_neo4j, failure=failure)
        expected = (
            _state_after_external_race(before, failure)
            if failure in {"competitor_race", "unit_race", "temporal_type_race"}
            else before
        )
        assert _state(driver) == expected
