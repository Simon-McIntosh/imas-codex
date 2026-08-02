"""Disposable-graph checks for bounded manifest transaction semantics."""

from __future__ import annotations

import os
from collections.abc import Iterator
from uuid import uuid4

import pytest
from neo4j import GraphDatabase

from imas_codex.graph.client import GraphClient
from imas_codex.settings import get_graph_uri
from imas_codex.standard_names import graph_ops
from imas_codex.standard_names.graph_ops import (
    ManifestDrainConflict,
    build_manifest_drain_plan,
    claim_generate_name_batch,
    claim_manifest_drain_scope,
    clear_manifest_drain_scope,
    prepare_manifest_drain_scope,
    release_manifest_drain_claims,
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
        claimed = claim_generate_name_batch(drain_scope_id=scope)
        assert [item["id"] for item in claimed] == [f"dd:{graph_case['good']}"]
        marker = gc.query(
            """
            MATCH (source:StandardNameSource {id: $id})
            RETURN source.drain_claim_scope_id AS marker
            """,
            id=f"dd:{graph_case['good']}",
        )[0]["marker"]
        assert marker == scope
        assert release_manifest_drain_claims(scope, gc=gc) == {
            "sources": 1,
            "names": 0,
        }
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
            MATCH (node:IMASNode {id: $path})
            CREATE (source:StandardNameSource {
              id: 'dd:' + $path, source_type: 'dd', source_id: $path,
              status: 'composed', drain_scope_id: $scope,
              drain_scope_claimed_at: datetime() - duration('PT1H'),
              drain_scope_paths: [$path], drain_claim_scope_id: $scope,
              claim_token: 'source-expired',
              claimed_at: datetime() - duration('PT1H')
            })-[:FROM_DD_PATH]->(node)
            CREATE (fresh:StandardName {
              id: $fresh, name_stage: 'refining', docs_stage: 'pending',
              drain_scope_id: $scope,
              drain_scope_paths: [$path], drain_claim_scope_id: 'external-scope',
              drain_scope_claimed_at: datetime() - duration('PT1H'),
              claim_token: 'external', claimed_at: datetime()
            })
            CREATE (stale:StandardName {
              id: $stale, name_stage: 'refining', docs_stage: 'pending',
              drain_scope_id: $scope,
              drain_scope_paths: [$path], drain_claim_scope_id: $scope,
              drain_scope_claimed_at: datetime() - duration('PT1H'),
              claim_token: 'expired',
              claimed_at: datetime() - duration('PT1H')
            })
            CREATE (source)-[:PRODUCED_NAME]->(fresh)
            CREATE (source)-[:PRODUCED_NAME]->(stale)
            WITH source
            MATCH (unrelated_node:IMASNode {id: $unrelated_path})
            CREATE (:StandardNameSource {
              id: 'dd:' + $unrelated_path, source_type: 'dd',
              source_id: $unrelated_path, status: 'extracted',
              drain_scope_id: $scope,
              drain_scope_claimed_at: datetime() - duration('PT1H'),
              drain_scope_actionable: true,
              drain_scope_paths: [$unrelated_path],
              drain_claim_scope_id: 'external-scope',
              claim_token: 'unrelated-external', claimed_at: datetime()
            })-[:FROM_DD_PATH]->(unrelated_node)
            """,
            path=graph_case["active"],
            unrelated_path=graph_case["bad"],
            fresh=fresh,
            stale=stale,
            scope=scope,
        ).consume()

    with _client() as gc:
        result = recover_manifest_drain_scope(
            scope,
            scope_timeout_s=60,
            worker_timeout_s=60,
            paths=[graph_case["active"]],
            gc=gc,
        )
    assert result == {"sources": 1, "names": 2, "refining_reverted": 1}

    with ephemeral_driver.session() as session:
        source_token = session.run(
            """
            MATCH (source:StandardNameSource {id: 'dd:' + $path})
            RETURN source.claim_token AS token
            """,
            path=graph_case["active"],
        ).single(strict=True)["token"]
    assert source_token is None

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
    with ephemeral_driver.session() as session:
        unrelated = session.run(
            """
            MATCH (source:StandardNameSource {id: 'dd:' + $path})
            RETURN source.status AS status, source.claim_token AS token,
                   source.drain_scope_id AS scope,
                   source.drain_scope_actionable AS actionable
            """,
            path=graph_case["bad"],
        ).single(strict=True)
    assert dict(unrelated) == {
        "status": "extracted",
        "token": "unrelated-external",
        "scope": scope,
        "actionable": True,
    }


def test_scope_takeover_recovers_only_stale_worker_state_and_preserves_status(
    ephemeral_driver, graph_case
) -> None:
    path = graph_case["active"]
    prior_scope = f"test/{graph_case['namespace']}/prior-scope"
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
              drain_scope_paths: [$path],
              drain_scope_claimed_at: datetime() - duration('PT1H'),
              description: node.documentation,
              physics_domain: node.physics_domain,
              dd_documentation: node.documentation,
              dd_data_type: node.data_type,
              dd_lifecycle_status: node.lifecycle_status,
              dd_lifecycle_version: node.lifecycle_version,
              enhanced_description: node.description,
              enhancement_kind: node.enrichment_source
            })-[:FROM_DD_PATH]->(node)
            CREATE (name:StandardName {
              id: $name_id, name_stage: 'refining', docs_stage: 'pending',
              validation_status: 'valid', claim_token: 'expired',
              claimed_at: datetime() - duration('PT1H'),
              drain_scope_id: $prior_scope,
              drain_scope_paths: [$path], drain_claim_scope_id: $prior_scope,
              drain_scope_claimed_at: datetime() - duration('PT1H')
            })
            CREATE (source)-[:PRODUCED_NAME]->(name)
            """,
            path=path,
            version=graph_case["version"],
            name_id=name_id,
            prior_scope=prior_scope,
        ).consume()

    scope_id, plan = prepare_manifest_drain_scope(
        [path], dd_version=graph_case["version"], scope_timeout_seconds=60
    )
    assert scope_id != prior_scope
    assert plan[0]["disposition"] == "active_in_flight"
    with _client() as gc:
        clear_manifest_drain_scope(scope_id, gc=gc)

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


def test_recovery_clears_refined_predecessor_outside_current_source_closure(
    ephemeral_driver, graph_case
) -> None:
    path = graph_case["active"]
    scope = f"test/{graph_case['namespace']}/refined-scope"
    predecessor = f"test/{graph_case['namespace']}/predecessor"
    successor = f"test/{graph_case['namespace']}/successor"
    with ephemeral_driver.session() as session:
        session.run(
            """
            MATCH (node:IMASNode {id: $path})
            CREATE (source:StandardNameSource {
              id: 'dd:' + $path, source_type: 'dd', source_id: $path,
              status: 'composed', drain_scope_id: $scope,
              drain_scope_paths: [$path],
              drain_scope_claimed_at: datetime() - duration('PT1H')
            })-[:FROM_DD_PATH]->(node)
            CREATE (old:StandardName {
              id: $predecessor, name_stage: 'refining', docs_stage: 'pending',
              drain_scope_id: $scope, drain_scope_paths: [$path],
              drain_scope_claimed_at: datetime() - duration('PT1H'),
              drain_claim_scope_id: $scope, claim_token: 'owned-old',
              claimed_at: datetime() - duration('PT1H')
            })
            CREATE (new:StandardName {
              id: $successor, name_stage: 'drafted', docs_stage: 'pending',
              drain_scope_id: $scope, drain_scope_paths: [$path],
              drain_scope_claimed_at: datetime() - duration('PT1H')
            })
            CREATE (new)-[:REFINED_FROM]->(old)
            CREATE (source)-[:PRODUCED_NAME]->(new)
            """,
            path=path,
            scope=scope,
            predecessor=predecessor,
            successor=successor,
        ).consume()
    with _client() as gc:
        result = recover_manifest_drain_scope(
            scope,
            scope_timeout_s=60,
            worker_timeout_s=60,
            paths=[path],
            gc=gc,
        )
    assert result == {"sources": 1, "names": 2, "refining_reverted": 1}
    with ephemeral_driver.session() as session:
        rows = list(
            session.run(
                """
                MATCH (name:StandardName) WHERE name.id IN [$predecessor, $successor]
                RETURN name.id AS id, name.drain_scope_id AS scope,
                       name.claim_token AS token, name.name_stage AS stage
                ORDER BY id
                """,
                predecessor=predecessor,
                successor=successor,
            )
        )
    assert all(row["scope"] is None for row in rows)
    predecessor_row = next(row for row in rows if row["id"] == predecessor)
    assert predecessor_row["token"] is None
    assert predecessor_row["stage"] == "reviewed"


def test_terminal_roots_are_report_only_and_unclaimable(
    ephemeral_driver, graph_case
) -> None:
    accepted_path = graph_case["active"]
    metadata_path = f"test/{graph_case['namespace']}/metadata"
    terminal_paths = [
        f"test/{graph_case['namespace']}/{status}"
        for status in ("failed", "vocab_gap", "skipped", "attempt_cap")
    ]
    scope = f"test/{graph_case['namespace']}/terminal-scope"
    with ephemeral_driver.session() as session:
        session.run(
            """
            MATCH (accepted_node:IMASNode {id: $accepted_path})
            CREATE (accepted_source:StandardNameSource {
              id: 'dd:' + $accepted_path, source_type: 'dd',
              source_id: $accepted_path, status: 'extracted',
              dd_version: $version, dd_snapshot_pinned: true,
              description: accepted_node.documentation,
              physics_domain: accepted_node.physics_domain,
              dd_documentation: accepted_node.documentation,
              dd_data_type: accepted_node.data_type,
              dd_lifecycle_status: accepted_node.lifecycle_status
            })-[:FROM_DD_PATH]->(accepted_node)
            CREATE (accepted_name:StandardName {
              id: $accepted_name, name_stage: 'accepted', docs_stage: 'accepted',
              validation_status: 'valid'
            })
            CREATE (accepted_source)-[:PRODUCED_NAME]->(accepted_name)
            SET accepted_source.produced_sn_id = accepted_name.id
            CREATE (metadata:IMASNode {
              id: $metadata_path, node_category: 'metadata',
              documentation: 'metadata', data_type: 'STR_0D',
              lifecycle_status: 'active'
            })
            CREATE (:StandardNameSource {
              id: 'dd:' + $metadata_path, source_type: 'dd',
              source_id: $metadata_path, status: 'extracted',
              dd_version: $version, dd_snapshot_pinned: true,
              description: metadata.documentation,
              physics_domain: metadata.physics_domain,
              dd_documentation: metadata.documentation,
              dd_data_type: metadata.data_type,
              dd_lifecycle_status: metadata.lifecycle_status
            })-[:FROM_DD_PATH]->(metadata)
            WITH 1 AS ready
            UNWIND $terminal AS item
            CREATE (terminal_node:IMASNode {
              id: item.path, node_category: 'quantity',
              documentation: 'terminal', data_type: 'FLT_0D',
              lifecycle_status: 'active'
            })
            CREATE (:StandardNameSource {
              id: 'dd:' + item.path, source_type: 'dd', source_id: item.path,
              status: item.status, attempt_count: item.attempt_count,
              dd_version: $version, dd_snapshot_pinned: true,
              description: terminal_node.documentation,
              dd_documentation: terminal_node.documentation,
              dd_data_type: terminal_node.data_type,
              dd_lifecycle_status: terminal_node.lifecycle_status
            })-[:FROM_DD_PATH]->(terminal_node)
            """,
            accepted_path=accepted_path,
            accepted_name=f"test/{graph_case['namespace']}/accepted-name",
            metadata_path=metadata_path,
            version=graph_case["version"],
            terminal=[
                {
                    "path": path,
                    "status": "extracted"
                    if path.endswith("attempt_cap")
                    else path.rsplit("/", 1)[-1],
                    "attempt_count": 5 if path.endswith("attempt_cap") else 0,
                }
                for path in terminal_paths
            ],
        ).consume()

    with _client() as gc:
        _, plan = claim_manifest_drain_scope(
            [accepted_path, metadata_path, *terminal_paths],
            dd_version=graph_case["version"],
            drain_scope_id=scope,
            gc=gc,
        )
    dispositions = {item["path"]: item["disposition"] for item in plan}
    assert dispositions == {
        accepted_path: "accepted",
        metadata_path: "non_nameable",
        **dict.fromkeys(terminal_paths, "genuine_gap"),
    }
    assert claim_generate_name_batch(drain_scope_id=scope) == []
    with ephemeral_driver.session() as session:
        row = session.run(
            """
            MATCH (source:StandardNameSource)
            WHERE source.id IN ['dd:' + $accepted_path, 'dd:' + $metadata_path]
               OR source.source_id IN $terminal_paths
            RETURN count(CASE WHEN source.drain_scope_id IS NOT NULL THEN 1 END)
                   AS scoped,
                   count(CASE WHEN source.claim_token IS NOT NULL THEN 1 END)
                   AS claimed
            """,
            accepted_path=accepted_path,
            metadata_path=metadata_path,
            terminal_paths=terminal_paths,
        ).single(strict=True)
    assert dict(row) == {"scoped": 0, "claimed": 0}


@pytest.mark.parametrize(
    "mutation",
    [
        "documentation",
        "data_type",
        "physics_domain",
        "lifecycle",
        "enrichment",
        "unit",
        "coordinate",
        "parent",
        "configured_version",
    ],
)
def test_authority_drift_between_plan_and_cas_rolls_back(
    monkeypatch, ephemeral_driver, graph_case, mutation: str
) -> None:
    path = graph_case["good"]
    original_lock = graph_ops._lock_manifest_drain_authority

    def mutate_then_lock(gc, paths):
        with ephemeral_driver.session() as session:
            if mutation == "configured_version":
                session.run(
                    """
                    MATCH (version:DDVersion {id: $version})
                    SET version.is_current = false
                    """,
                    version=graph_case["version"],
                ).consume()
            elif mutation in {
                "documentation",
                "data_type",
                "physics_domain",
                "lifecycle",
                "enrichment",
            }:
                assignments = {
                    "documentation": "node.documentation = 'changed'",
                    "data_type": "node.data_type = 'INT_0D'",
                    "physics_domain": "node.physics_domain = 'magnetics'",
                    "lifecycle": (
                        "node.lifecycle_status = 'deprecated', "
                        "node.lifecycle_version = 'changed'"
                    ),
                    "enrichment": (
                        "node.description = 'changed', node.enrichment_source = 'test'"
                    ),
                }
                session.run(
                    f"MATCH (node:IMASNode {{id: $path}}) SET {assignments[mutation]}",
                    path=path,
                ).consume()
            else:
                labels = {
                    "unit": "Unit",
                    "coordinate": "Coordinate",
                    "parent": "IMASNode",
                }
                relationships = {
                    "unit": "HAS_UNIT",
                    "coordinate": "HAS_COORDINATE",
                    "parent": "HAS_PARENT",
                }
                authority_id = f"test/{graph_case['namespace']}/{mutation}"
                session.run(
                    f"""
                    MATCH (node:IMASNode {{id: $path}})
                    CREATE (authority:{labels[mutation]} {{id: $authority_id}})
                    CREATE (node)-[:{relationships[mutation]}]->(authority)
                    """,
                    path=path,
                    authority_id=authority_id,
                ).consume()
        original_lock(gc, paths)

    monkeypatch.setattr(graph_ops, "_lock_manifest_drain_authority", mutate_then_lock)
    with _client() as gc, pytest.raises(ManifestDrainConflict):
        claim_manifest_drain_scope(
            [path],
            dd_version=graph_case["version"],
            drain_scope_id=f"test/{graph_case['namespace']}/cas-{mutation}",
            gc=gc,
        )
    with ephemeral_driver.session() as session:
        row = session.run(
            """
            OPTIONAL MATCH (source:StandardNameSource {id: 'dd:' + $path})
            RETURN source IS NOT NULL AS created
            """,
            path=path,
        ).single(strict=True)
    assert row["created"] is False


@pytest.mark.parametrize(
    ("field", "changed"),
    [
        ("description", "changed"),
        ("physics_domain", "magnetics"),
        ("dd_documentation", "changed"),
        ("dd_parent_path", "changed/parent"),
        ("dd_parent_documentation", "changed"),
        ("dd_data_type", "INT_0D"),
        ("dd_unit", "A"),
        ("dd_coordinates", ["changed/coordinate"]),
        ("dd_lifecycle_status", "deprecated"),
        ("dd_lifecycle_version", "changed"),
        ("enhanced_description", "changed"),
        ("enhancement_kind", "test"),
    ],
)
def test_pinned_source_drift_between_plan_and_cas_rolls_back(
    monkeypatch, ephemeral_driver, graph_case, field: str, changed
) -> None:
    path = graph_case["good"]
    initial_scope = f"test/{graph_case['namespace']}/initial"
    next_scope = f"test/{graph_case['namespace']}/source-cas-{field}"
    with _client() as gc:
        claim_manifest_drain_scope(
            [path],
            dd_version=graph_case["version"],
            drain_scope_id=initial_scope,
            gc=gc,
        )
        clear_manifest_drain_scope(initial_scope, gc=gc)

    original_lock = graph_ops._lock_manifest_drain_authority

    def mutate_then_lock(gc, paths):
        with ephemeral_driver.session() as session:
            session.run(
                f"MATCH (source:StandardNameSource {{id: 'dd:' + $path}}) "
                f"SET source.{field} = $changed",
                path=path,
                changed=changed,
            ).consume()
        original_lock(gc, paths)

    monkeypatch.setattr(graph_ops, "_lock_manifest_drain_authority", mutate_then_lock)
    with _client() as gc, pytest.raises(ManifestDrainConflict):
        claim_manifest_drain_scope(
            [path],
            dd_version=graph_case["version"],
            drain_scope_id=next_scope,
            gc=gc,
        )
    with ephemeral_driver.session() as session:
        scoped = session.run(
            """
            MATCH (node) WHERE node.drain_scope_id = $scope
            RETURN count(node) AS count
            """,
            scope=next_scope,
        ).single(strict=True)["count"]
    assert scoped == 0


def test_exact_token_release_preserves_external_claim(
    ephemeral_driver, graph_case
) -> None:
    owned = f"test/{graph_case['namespace']}/owned-claim"
    external = f"test/{graph_case['namespace']}/external-claim"
    scope = f"test/{graph_case['namespace']}/claim-scope"
    with ephemeral_driver.session() as session:
        session.run(
            """
            CREATE (:StandardName {
              id: $owned, name_stage: 'refining', docs_stage: 'pending',
              claim_token: 'owned', claimed_at: datetime(),
              drain_scope_id: $scope, drain_claim_scope_id: $scope
            })
            CREATE (:StandardName {
              id: $external, name_stage: 'refining', docs_stage: 'pending',
              claim_token: 'external', claimed_at: datetime(),
              drain_scope_id: $scope, drain_claim_scope_id: 'external-scope'
            })
            """,
            owned=owned,
            external=external,
            scope=scope,
        ).consume()
    with _client() as gc:
        result = release_manifest_drain_claims(scope, gc=gc)
    assert result == {"sources": 0, "names": 1}
    with ephemeral_driver.session() as session:
        rows = {
            row["id"]: dict(row)
            for row in session.run(
                """
                MATCH (name:StandardName) WHERE name.id IN [$owned, $external]
                RETURN name.id AS id, name.name_stage AS stage,
                       name.claim_token AS token
                """,
                owned=owned,
                external=external,
            )
        }
    assert rows[owned] == {"id": owned, "stage": "reviewed", "token": None}
    assert rows[external] == {
        "id": external,
        "stage": "refining",
        "token": "external",
    }
