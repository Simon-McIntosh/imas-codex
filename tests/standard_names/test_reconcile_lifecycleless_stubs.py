"""Exact lifecycle-less StandardName stub reconciliation tests."""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest
from neo4j import GraphDatabase

from imas_codex.graph.client import GraphClient as RealGraphClient
from imas_codex.settings import get_graph_uri
from imas_codex.standard_names.graph_ops import (
    LifecyclelessStubConflict,
    _partition_lifecycleless_stub_rows,
    reconcile_lifecycleless_standard_name_stubs,
)


def _authority_rows() -> list[dict]:
    return [
        {
            "id": "electron_temperature",
            "properties": {"id": "electron_temperature"},
            "child_data": [
                {
                    "id": "maximum_of_electron_temperature",
                    "name_stage": "drafted",
                    "unit": "eV",
                    "cocos": None,
                    "physics_domain": "transport",
                    "kind": "scalar",
                    "op_kind": "unary_prefix",
                }
            ],
            "edge_kinds": ["unary_prefix"],
            "dd_sources": [],
        },
        {
            "id": "dead_link_endpoint",
            "properties": {"id": "dead_link_endpoint"},
            "child_data": [],
            "edge_kinds": [],
            "dd_sources": [],
        },
        {
            "id": "dd_backed_endpoint",
            "properties": {"id": "dd_backed_endpoint"},
            "child_data": [],
            "edge_kinds": [],
            "dd_sources": [
                {
                    "source_id": "dd:core_profiles/profiles_1d/electrons/temperature",
                    "expected_status": "composed",
                    "expected_scalar": "dd_backed_endpoint",
                    "expected_bindings": ["dd_backed_endpoint"],
                    "dd_path": "core_profiles/profiles_1d/electrons/temperature",
                    "unit": "eV",
                }
            ],
        },
    ]


def test_fresh_cohort_partitions_all_three_dispositions() -> None:
    partitions = _partition_lifecycleless_stub_rows(_authority_rows())

    assert [row["id"] for row in partitions["materialize-as-derived-parent"]] == [
        "electron_temperature"
    ]
    assert [row["id"] for row in partitions["delete-as-dead-link-stub"]] == [
        "dead_link_endpoint"
    ]
    assert [row["id"] for row in partitions["rebind-source"]] == ["dd_backed_endpoint"]
    assert partitions["refused"] == []


@pytest.mark.parametrize("missing", ["unit", "dd_path", "expected_status"])
def test_dd_backed_row_refuses_incomplete_authority(missing: str) -> None:
    row = _authority_rows()[2]
    row["dd_sources"][0][missing] = None

    partitions = _partition_lifecycleless_stub_rows([row])

    assert partitions["rebind-source"] == []
    assert partitions["refused"][0]["refusal_reason"] == (
        "incomplete DD source or unit authority"
    )


def test_parent_row_refuses_incomplete_unit_authority() -> None:
    row = _authority_rows()[0]
    row["child_data"][0]["unit"] = None

    partitions = _partition_lifecycleless_stub_rows([row])

    assert partitions["materialize-as-derived-parent"] == []
    assert partitions["refused"][0]["refusal_reason"] == (
        "incomplete child lifecycle or unit authority"
    )


@pytest.fixture(scope="module")
def disposable_neo4j() -> Iterator[tuple[str, str]]:
    uri = os.environ.get("IMAS_CODEX_TEST_NEO4J_URI")
    if not uri:
        pytest.skip("IMAS_CODEX_TEST_NEO4J_URI is not configured")
    if os.environ.get("IMAS_CODEX_TEST_NEO4J_EPHEMERAL") != "1":
        pytest.fail("stub reconciliation requires a disposable graph")
    if uri == (os.environ.get("IMAS_CODEX_TEST_PROJECT_NEO4J_URI") or get_graph_uri()):
        pytest.fail("stub reconciliation refuses the project graph")
    password = os.environ.get("IMAS_CODEX_TEST_NEO4J_PASSWORD", "")
    auth = ("neo4j", password) if password else None
    with GraphDatabase.driver(uri, auth=auth) as driver:
        driver.verify_connectivity()
    yield uri, password


@pytest.mark.graph
def test_transactional_dry_run_hash_apply_and_replay(
    disposable_neo4j: tuple[str, str],
) -> None:
    uri, password = disposable_neo4j
    child_id = "maximum_of_electron_temperature"
    parent_id = "electron_temperature"
    dead_id = "dead_link_endpoint"
    rebound_id = "dd_backed_endpoint"
    dd_path = "core_profiles/profiles_1d/electrons/temperature"
    source_id = f"dd:{dd_path}"
    ids = [child_id, parent_id, dead_id, rebound_id, dd_path, source_id]
    client = RealGraphClient(
        uri=uri,
        username="neo4j",
        password=password,
        graph_name="disposable-stub-reconciliation",
    )
    try:
        client.query(
            "CREATE (child:StandardName {id: $child_id, name_stage: 'drafted', "
            "unit: 'eV', kind: 'scalar', physics_domain: 'transport'}) "
            "CREATE (parent:StandardName {id: $parent_id}) "
            "CREATE (child)-[:HAS_PARENT {operator: 'maximum', "
            "operator_kind: 'unary_prefix'}]->(parent) "
            "CREATE (:StandardName {id: $dead_id}) "
            "CREATE (rebound:StandardName {id: $rebound_id}) "
            "CREATE (dd:IMASNode {id: $dd_path, units: 'eV'}) "
            "CREATE (source:StandardNameSource {id: $source_id, source_type: 'dd', "
            "source_id: $dd_path, status: 'composed', produced_sn_id: $rebound_id}) "
            "CREATE (source)-[:FROM_DD_PATH]->(dd) "
            "CREATE (source)-[:PRODUCED_NAME]->(rebound)",
            child_id=child_id,
            parent_id=parent_id,
            dead_id=dead_id,
            rebound_id=rebound_id,
            dd_path=dd_path,
            source_id=source_id,
        )

        preview = reconcile_lifecycleless_standard_name_stubs(gc=client)
        assert preview["counts"] == {
            "materialize-as-derived-parent": 1,
            "delete-as-dead-link-stub": 1,
            "rebind-source": 1,
            "refused": 0,
        }
        assert preview["changed"] == 0
        assert preview["would_change"] == 3
        assert (
            client.query(
                "MATCH (stub:StandardName) WHERE stub.id IN $ids "
                "RETURN count(stub) AS count",
                ids=[parent_id, dead_id, rebound_id],
            )[0]["count"]
            == 3
        )

        with pytest.raises(LifecyclelessStubConflict, match="fresh lifecycle-less"):
            reconcile_lifecycleless_standard_name_stubs(
                apply=True,
                manifest_sha256="0" * 64,
                gc=client,
            )

        applied = reconcile_lifecycleless_standard_name_stubs(
            apply=True,
            manifest_sha256=preview["manifest_sha256"],
            gc=client,
        )
        assert applied["changed"] == 3
        assert applied["sources_reset"] == 1
        state = client.query(
            "MATCH (parent:StandardName {id: $parent_id}) "
            "OPTIONAL MATCH (dead:StandardName {id: $dead_id}) "
            "OPTIONAL MATCH (rebound:StandardName {id: $rebound_id}) "
            "MATCH (source:StandardNameSource {id: $source_id}) "
            "RETURN parent.name_stage AS parent_stage, parent.origin AS origin, "
            "dead IS NULL AS dead_deleted, rebound IS NULL AS rebound_deleted, "
            "source.status AS source_status, source.produced_sn_id AS produced",
            parent_id=parent_id,
            dead_id=dead_id,
            rebound_id=rebound_id,
            source_id=source_id,
        )[0]
        assert state == {
            "parent_stage": "accepted",
            "origin": "derived",
            "dead_deleted": True,
            "rebound_deleted": True,
            "source_status": "extracted",
            "produced": None,
        }
        assert (
            client.query(
                "MATCH (change:StandardNameChange "
                "{manifest_sha256: $manifest_sha256}) RETURN count(change) AS count",
                manifest_sha256=preview["manifest_sha256"],
            )[0]["count"]
            == 3
        )

        replay = reconcile_lifecycleless_standard_name_stubs(
            apply=True,
            manifest_sha256=preview["manifest_sha256"],
            gc=client,
        )
        assert replay["outcome"] == "already_applied"
        assert replay["changed"] == 0
    finally:
        client.query(
            "MATCH (node) WHERE node.id IN $ids "
            "OR node.manifest_sha256 = $manifest_sha256 "
            "OR node.id STARTS WITH 'source-retry:source-reset:' "
            "DETACH DELETE node",
            ids=ids + [f"derived:{parent_id}"],
            manifest_sha256=locals().get("preview", {}).get("manifest_sha256"),
        )
