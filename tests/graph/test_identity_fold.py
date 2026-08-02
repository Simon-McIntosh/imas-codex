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
    return expected


def _invoke(
    instance: _EphemeralNeo4j,
    *,
    failure: str | None = None,
) -> dict[str, Any]:
    with (
        patch.object(edit, "GraphClient", side_effect=lambda: instance.client(failure)),
        patch.object(edit, "_isn_round_trip_ok", return_value=(True, None)),
        patch.object(edit, "_fold_west_dd_paths", return_value=frozenset()),
    ):
        return edit.supersede_into("invalid_duplicate", "electron_density")


def test_third_live_competitor_refuses_without_state_change(ephemeral_neo4j) -> None:
    with ephemeral_neo4j.driver() as driver:
        _seed(driver, third_stage="accepted")
        before = _state(driver)
        result = _invoke(ephemeral_neo4j)
        assert result["ok"] is False
        assert "third live" in result["reason"]
        assert _state(driver) == before


@pytest.mark.parametrize(
    ("failure", "message"),
    [
        ("event", "event failure"),
        ("partial", "partial source migration"),
        ("competitor_race", "changed after preflight"),
        ("unit_race", "changed after preflight"),
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
            if failure in {"competitor_race", "unit_race"}
            else before
        )
        assert _state(driver) == expected
