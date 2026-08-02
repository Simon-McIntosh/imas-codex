"""Disposable-graph checks for bounded manifest transaction semantics."""

from __future__ import annotations

import os
from collections.abc import Iterator
from uuid import uuid4

import pytest
from neo4j import GraphDatabase

from imas_codex.graph.client import GraphClient
from imas_codex.settings import get_graph_uri
from imas_codex.standard_names.graph_ops import (
    ManifestDrainConflict,
    build_manifest_drain_plan,
    claim_manifest_drain_scope,
    clear_manifest_drain_scope,
)
from imas_codex.standard_names.orphan_sweep import recover_manifest_drain_scope

pytestmark = pytest.mark.graph


@pytest.fixture(scope="module")
def ephemeral_driver() -> Iterator:
    uri = os.environ.get("IMAS_CODEX_TEST_NEO4J_URI")
    if not uri:
        pytest.skip("IMAS_CODEX_TEST_NEO4J_URI is not configured")
    if os.environ.get("IMAS_CODEX_TEST_NEO4J_EPHEMERAL") != "1":
        pytest.fail("stateful manifest-drain tests require an ephemeral graph")
    project_uri = os.environ.get("IMAS_CODEX_TEST_PROJECT_NEO4J_URI") or get_graph_uri()
    if uri == project_uri:
        pytest.fail("manifest-drain tests refuse the configured project graph")
    driver = GraphDatabase.driver(uri, auth=None)
    driver.verify_connectivity()
    try:
        yield driver
    finally:
        driver.close()


@pytest.fixture
def graph_case(ephemeral_driver) -> Iterator[dict[str, str]]:
    namespace = uuid4().hex
    version = f"test-{namespace}"
    paths = {
        "good": f"test/{namespace}/good",
        "bad": f"test/{namespace}/bad",
        "active": f"test/{namespace}/active",
    }
    with ephemeral_driver.session() as session:
        session.run(
            """
            CREATE (:DDVersion {id: $version, is_current: true})
            WITH 1 AS ready
            UNWIND $paths AS path
            CREATE (:IMASNode {
              id: path, node_category: 'quantity', documentation: 'test quantity',
              data_type: 'FLT_0D', lifecycle_status: 'active'
            })
            """,
            version=version,
            paths=list(paths.values()),
        ).consume()
    try:
        yield {"namespace": namespace, "version": version, **paths}
    finally:
        with ephemeral_driver.session() as session:
            session.run(
                """
                MATCH (node)
                WHERE node.id STARTS WITH $prefix OR node.id = $version
                   OR node.source_id STARTS WITH $prefix
                   OR node.drain_scope_id STARTS WITH $prefix
                DETACH DELETE node
                """,
                prefix=f"test/{namespace}",
                version=version,
            ).consume()


def _client() -> GraphClient:
    return GraphClient(
        uri=os.environ["IMAS_CODEX_TEST_NEO4J_URI"],
        username="neo4j",
        password="",
        graph_name="ephemeral-manifest-drain",
    )


def test_read_plan_is_zero_write_and_dd_gap_is_reporting_only(
    ephemeral_driver, graph_case
) -> None:
    with ephemeral_driver.session() as session:
        session.run(
            """
            MATCH (node:IMASNode {id: $path})
            CREATE (gap:DDGap {id: $gap_id, path: $path, kind: 'vocabulary_gap'})
            CREATE (node)-[:HAS_DD_GAP]->(gap)
            """,
            path=graph_case["good"],
            gap_id=f"test/{graph_case['namespace']}/gap",
        ).consume()
        before = session.run(
            "MATCH (node) WHERE node.id STARTS WITH $prefix RETURN count(node) AS n",
            prefix=f"test/{graph_case['namespace']}",
        ).single(strict=True)["n"]

    with _client() as gc:
        plan = build_manifest_drain_plan(
            [graph_case["good"]], dd_version=graph_case["version"], gc=gc
        )
    assert plan[0]["disposition"] == "genuine_gap"

    with ephemeral_driver.session() as session:
        after = session.run(
            "MATCH (node) WHERE node.id STARTS WITH $prefix RETURN count(node) AS n",
            prefix=f"test/{graph_case['namespace']}",
        ).single(strict=True)["n"]
    assert after == before


def test_ambiguity_rolls_back_the_complete_scope_stamp(
    ephemeral_driver, graph_case
) -> None:
    path = graph_case["bad"]
    with ephemeral_driver.session() as session:
        session.run(
            """
            MATCH (node:IMASNode {id: $path})
            CREATE (source:StandardNameSource {
              id: 'dd:' + $path, source_type: 'dd', source_id: $path,
              status: 'composed', dd_version: $version,
              dd_snapshot_pinned: true
            })-[:FROM_DD_PATH]->(node)
            CREATE (first:StandardName {
              id: $first, name_stage: 'drafted', docs_stage: 'pending',
              validation_status: 'valid'
            })
            CREATE (second:StandardName {
              id: $second, name_stage: 'reviewed', docs_stage: 'pending',
              validation_status: 'valid'
            })
            CREATE (source)-[:PRODUCED_NAME]->(first)
            CREATE (source)-[:PRODUCED_NAME]->(second)
            """,
            path=path,
            version=graph_case["version"],
            first=f"test/{graph_case['namespace']}/first",
            second=f"test/{graph_case['namespace']}/second",
        ).consume()

    with _client() as gc, pytest.raises(ManifestDrainConflict):
        claim_manifest_drain_scope(
            [graph_case["good"], path],
            dd_version=graph_case["version"],
            drain_scope_id=f"test/{graph_case['namespace']}/scope",
            gc=gc,
        )

    with ephemeral_driver.session() as session:
        row = session.run(
            """
            CALL {
              OPTIONAL MATCH (created:StandardNameSource {id: 'dd:' + $good})
              RETURN created IS NOT NULL AS created
            }
            CALL {
              MATCH (node) WHERE node.drain_scope_id = $scope
              RETURN count(node) AS scoped
            }
            RETURN created, scoped
            """,
            good=graph_case["good"],
            scope=f"test/{graph_case['namespace']}/scope",
        ).single(strict=True)
    assert dict(row) == {"created": False, "scoped": 0}


def test_exact_source_creation_and_cleanup_are_scope_owned(
    ephemeral_driver, graph_case
) -> None:
    scope = f"test/{graph_case['namespace']}/owned"
    with _client() as gc:
        scope_id, plan = claim_manifest_drain_scope(
            [graph_case["good"]],
            dd_version=graph_case["version"],
            drain_scope_id=scope,
            gc=gc,
        )
        assert scope_id == scope
        assert plan[0]["disposition"] == "genuine_gap"
        cleared = clear_manifest_drain_scope(scope, gc=gc)
    assert cleared == {"sources": 1, "names": 0}

    with ephemeral_driver.session() as session:
        row = session.run(
            """
            MATCH (source:StandardNameSource {id: 'dd:' + $path})
            RETURN source.status AS status, source.dd_version AS dd_version,
                   source.dd_snapshot_pinned AS pinned,
                   source.drain_scope_id AS scope
            """,
            path=graph_case["good"],
        ).single(strict=True)
    assert dict(row) == {
        "status": "extracted",
        "dd_version": graph_case["version"],
        "pinned": True,
        "scope": None,
    }


def test_stale_scope_recovery_preserves_a_fresh_worker_token(
    ephemeral_driver, graph_case
) -> None:
    scope = f"test/{graph_case['namespace']}/stale"
    fresh = f"test/{graph_case['namespace']}/fresh-worker"
    stale = f"test/{graph_case['namespace']}/stale-worker"
    with ephemeral_driver.session() as session:
        session.run(
            """
            CREATE (:StandardName {
              id: $fresh, name_stage: 'refining', docs_stage: 'pending',
              drain_scope_id: $scope,
              drain_scope_claimed_at: datetime() - duration('PT1H'),
              claim_token: 'external', claimed_at: datetime()
            })
            CREATE (:StandardName {
              id: $stale, name_stage: 'refining', docs_stage: 'pending',
              drain_scope_id: $scope,
              drain_scope_claimed_at: datetime() - duration('PT1H'),
              claim_token: 'expired',
              claimed_at: datetime() - duration('PT1H')
            })
            """,
            fresh=fresh,
            stale=stale,
            scope=scope,
        ).consume()

    with _client() as gc:
        result = recover_manifest_drain_scope(
            scope, scope_timeout_s=60, worker_timeout_s=60, gc=gc
        )
    assert result == {"sources": 0, "names": 2, "refining_reverted": 1}

    with ephemeral_driver.session() as session:
        rows = {
            row["id"]: dict(row)
            for row in session.run(
                """
                MATCH (name:StandardName) WHERE name.id IN [$fresh, $stale]
                RETURN name.id AS id, name.name_stage AS stage,
                       name.claim_token AS token, name.drain_scope_id AS scope
                """,
                fresh=fresh,
                stale=stale,
            )
        }
    assert rows[fresh] == {
        "id": fresh,
        "stage": "refining",
        "token": "external",
        "scope": None,
    }
    assert rows[stale] == {
        "id": stale,
        "stage": "reviewed",
        "token": None,
        "scope": None,
    }


def test_scope_takeover_recovers_only_stale_worker_state_and_preserves_status(
    ephemeral_driver, graph_case
) -> None:
    path = graph_case["active"]
    prior_scope = f"test/{graph_case['namespace']}/prior-scope"
    next_scope = f"test/{graph_case['namespace']}/next-scope"
    name_id = f"test/{graph_case['namespace']}/refining-name"
    with ephemeral_driver.session() as session:
        session.run(
            """
            MATCH (node:IMASNode {id: $path})
            CREATE (source:StandardNameSource {
              id: 'dd:' + $path, source_type: 'dd', source_id: $path,
              status: 'failed', dd_version: $version,
              dd_snapshot_pinned: true, produced_sn_id: $name_id,
              drain_scope_id: $prior_scope,
              drain_scope_claimed_at: datetime() - duration('PT1H')
            })-[:FROM_DD_PATH]->(node)
            CREATE (name:StandardName {
              id: $name_id, name_stage: 'refining', docs_stage: 'pending',
              validation_status: 'valid', claim_token: 'expired',
              claimed_at: datetime() - duration('PT1H'),
              drain_scope_id: $prior_scope,
              drain_scope_claimed_at: datetime() - duration('PT1H')
            })
            CREATE (source)-[:PRODUCED_NAME]->(name)
            """,
            path=path,
            version=graph_case["version"],
            name_id=name_id,
            prior_scope=prior_scope,
        ).consume()

    with _client() as gc:
        scope_id, plan = claim_manifest_drain_scope(
            [path],
            dd_version=graph_case["version"],
            drain_scope_id=next_scope,
            gc=gc,
        )
        assert scope_id == next_scope
        assert plan[0]["disposition"] == "active_in_flight"
        clear_manifest_drain_scope(next_scope, gc=gc)

    with ephemeral_driver.session() as session:
        row = session.run(
            """
            MATCH (source:StandardNameSource {id: 'dd:' + $path})
                  -[:PRODUCED_NAME]->(name:StandardName {id: $name_id})
            RETURN source.status AS source_status,
                   name.name_stage AS name_stage,
                   name.claim_token AS token,
                   source.drain_scope_id AS source_scope,
                   name.drain_scope_id AS name_scope
            """,
            path=path,
            name_id=name_id,
        ).single(strict=True)
    assert dict(row) == {
        "source_status": "failed",
        "name_stage": "reviewed",
        "token": None,
        "source_scope": None,
        "name_scope": None,
    }
