"""Exact lifecycle-less StandardName stub reconciliation tests."""

from __future__ import annotations

import os
from collections.abc import Iterator
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from neo4j import GraphDatabase

from imas_codex.graph.client import GraphClient as RealGraphClient
from imas_codex.settings import get_graph_uri
from imas_codex.standard_names.graph_ops import (
    LifecyclelessStubConflict,
    _lifecycleless_stub_manifest,
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


def _spurious_binding_row(
    *, sibling_stage: str = "accepted", sibling_validation: str = "valid"
) -> dict:
    stub_id = "fast_neutral_beam_motional_stark_wavelength"
    sibling_id = (
        "difference_of_fast_neutral_beam_motional_stark_wavelength_and_"
        "fast_neutral_beam_reference_wavelength_of_spectral_line"
    )
    return {
        "id": stub_id,
        "properties": {"id": stub_id},
        "child_data": [],
        "edge_kinds": [],
        "dd_sources": [
            {
                "source_id": "dd:charge_exchange/channel/bes/lorentz_shift",
                "expected_status": "composed",
                "expected_scalar": None,
                "expected_bindings": [stub_id, sibling_id],
                "binding_targets": [
                    {
                        "id": stub_id,
                        "name_stage": None,
                        "validation_status": None,
                    },
                    {
                        "id": sibling_id,
                        "name_stage": sibling_stage,
                        "validation_status": sibling_validation,
                    },
                ],
                "dd_path": "charge_exchange/channel/bes/lorentz_shift",
                "unit": "m",
            }
        ],
    }


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


def test_accepted_valid_sibling_authorizes_spurious_stub_deletion() -> None:
    row = _spurious_binding_row()

    partitions = _partition_lifecycleless_stub_rows([row])

    assert partitions["refused"] == []
    deleted = partitions["delete-as-dead-link-stub"]
    assert [item["id"] for item in deleted] == [row["id"]]
    assert deleted[0]["spurious_binding_authority"] == [
        {
            "source_id": "dd:charge_exchange/channel/bes/lorentz_shift",
            "expected_status": "composed",
            "expected_scalar": None,
            "expected_bindings": [
                "difference_of_fast_neutral_beam_motional_stark_wavelength_and_"
                "fast_neutral_beam_reference_wavelength_of_spectral_line",
                "fast_neutral_beam_motional_stark_wavelength",
            ],
            "authoritative_target_id": (
                "difference_of_fast_neutral_beam_motional_stark_wavelength_and_"
                "fast_neutral_beam_reference_wavelength_of_spectral_line"
            ),
            "authoritative_name_stage": "accepted",
            "authoritative_validation_status": "valid",
        }
    ]


def test_nonaccepted_sibling_does_not_authorize_spurious_stub_deletion() -> None:
    row = _spurious_binding_row(sibling_stage="drafted")

    partitions = _partition_lifecycleless_stub_rows([row])

    assert partitions["delete-as-dead-link-stub"] == []
    assert partitions["refused"][0]["refusal_reason"] == (
        "incomplete DD source or unit authority"
    )


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
        "no parent-owned unit authority"
    )


def test_nested_manifest_collections_are_order_insensitive() -> None:
    forward = _authority_rows()
    forward[0]["incident_relationships"] = [
        {"type": "HAS_UNIT", "related_properties": {"id": "eV"}},
        {"type": "HAS_LOCUS", "related_properties": {"id": "core"}},
    ]
    reverse = list(reversed(deepcopy(forward)))
    reverse[-1]["incident_relationships"] = list(
        reversed(reverse[-1]["incident_relationships"])
    )

    first = _lifecycleless_stub_manifest(_partition_lifecycleless_stub_rows(forward))
    second = _lifecycleless_stub_manifest(_partition_lifecycleless_stub_rows(reverse))

    assert first == second


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


def _disposable_client(endpoint: tuple[str, str], name: str) -> RealGraphClient:
    uri, password = endpoint
    return RealGraphClient(
        uri=uri,
        username="neo4j",
        password=password,
        graph_name=name,
    )


@pytest.mark.graph
def test_binary_only_child_refuses_parent_unit_authority(
    disposable_neo4j: tuple[str, str],
) -> None:
    parent_id = "diamagnetic_vorticity"
    child_id = "ratio_of_diamagnetic_vorticity_to_major_radius"
    client = _disposable_client(disposable_neo4j, "binary-unit-refusal")
    try:
        client.query(
            "CREATE (child:StandardName {id: $child, name_stage: 'drafted', "
            "unit: '1', kind: 'scalar', physics_domain: 'mhd'}) "
            "CREATE (parent:StandardName {id: $parent}) "
            "CREATE (child)-[:HAS_PARENT {operator: 'ratio', "
            "operator_kind: 'binary'}]->(parent)",
            child=child_id,
            parent=parent_id,
        )

        with pytest.raises(LifecyclelessStubConflict, match="parent-owned unit"):
            reconcile_lifecycleless_standard_name_stubs(gc=client)

        assert client.query(
            "MATCH (parent:StandardName {id: $parent}) "
            "RETURN parent.name_stage AS stage, parent.unit AS unit",
            parent=parent_id,
        ) == [{"stage": None, "unit": None}]
    finally:
        client.query(
            "MATCH (node) WHERE node.id IN $ids DETACH DELETE node",
            ids=[parent_id, child_id, f"derived:{parent_id}"],
        )


@pytest.mark.graph
def test_structural_admission_refusal_prevents_materialization(
    disposable_neo4j: tuple[str, str],
) -> None:
    parent_id = "electron_temperature"
    child_id = "maximum_of_electron_temperature"
    client = _disposable_client(disposable_neo4j, "admission-refusal")
    try:
        client.query(
            "CREATE (child:StandardName {id: $child, name_stage: 'drafted', "
            "unit: 'eV', kind: 'scalar', physics_domain: 'transport'}) "
            "CREATE (parent:StandardName {id: $parent}) "
            "CREATE (child)-[:HAS_PARENT {operator: 'maximum', "
            "operator_kind: 'unary_prefix'}]->(parent)",
            child=child_id,
            parent=parent_id,
        )
        with (
            patch(
                "imas_codex.standard_names.parents.is_admissible_parent_name",
                return_value=SimpleNamespace(
                    admit=False, reason="suppressed single-child shadow"
                ),
            ),
            pytest.raises(LifecyclelessStubConflict, match="admission refused"),
        ):
            reconcile_lifecycleless_standard_name_stubs(gc=client)

        assert client.query(
            "MATCH (parent:StandardName {id: $parent}) "
            "RETURN parent.name_stage AS stage",
            parent=parent_id,
        ) == [{"stage": None}]
    finally:
        client.query(
            "MATCH (node) WHERE node.id IN $ids DETACH DELETE node",
            ids=[parent_id, child_id, f"derived:{parent_id}"],
        )


@pytest.mark.graph
def test_new_incident_edge_invalidates_preview_hash(
    disposable_neo4j: tuple[str, str],
) -> None:
    stub_id = "dead_link_endpoint"
    locus_id = "core"
    client = _disposable_client(disposable_neo4j, "incident-hash-refusal")
    try:
        client.query("CREATE (:StandardName {id: $id})", id=stub_id)
        preview = reconcile_lifecycleless_standard_name_stubs(gc=client)
        client.query(
            "MATCH (stub:StandardName {id: $stub}) "
            "MERGE (locus:Locus {id: $locus}) "
            "CREATE (stub)-[:HAS_LOCUS]->(locus)",
            stub=stub_id,
            locus=locus_id,
        )

        with pytest.raises(LifecyclelessStubConflict, match="fresh lifecycle-less"):
            reconcile_lifecycleless_standard_name_stubs(
                apply=True,
                manifest_sha256=preview["manifest_sha256"],
                gc=client,
            )
        assert (
            client.query(
                "MATCH (:StandardName {id: $stub})-[edge:HAS_LOCUS]->(:Locus {id: $locus}) "
                "RETURN count(edge) AS count",
                stub=stub_id,
                locus=locus_id,
            )[0]["count"]
            == 1
        )
    finally:
        client.query(
            "MATCH (node) WHERE node.id IN $ids DETACH DELETE node",
            ids=[stub_id, locus_id],
        )


@pytest.mark.graph
def test_childful_rebind_preserves_parent_closure(
    disposable_neo4j: tuple[str, str],
) -> None:
    parent_id = "electron_temperature"
    child_id = "maximum_of_electron_temperature"
    dd_path = "core_profiles/profiles_1d/electrons/temperature"
    source_id = f"dd:{dd_path}"
    client = _disposable_client(disposable_neo4j, "childful-rebind")
    try:
        client.query(
            "CREATE (child:StandardName {id: $child, name_stage: 'drafted', "
            "unit: 'eV', kind: 'scalar', physics_domain: 'transport'}) "
            "CREATE (parent:StandardName {id: $parent}) "
            "CREATE (child)-[:HAS_PARENT {operator: 'maximum', "
            "operator_kind: 'unary_prefix'}]->(parent) "
            "CREATE (dd:IMASNode {id: $dd_path, units: 'eV'}) "
            "CREATE (source:StandardNameSource {id: $source, source_type: 'dd', "
            "source_id: $dd_path, status: 'composed', produced_sn_id: $parent}) "
            "CREATE (source)-[:FROM_DD_PATH]->(dd) "
            "CREATE (source)-[:PRODUCED_NAME]->(parent)",
            child=child_id,
            parent=parent_id,
            dd_path=dd_path,
            source=source_id,
        )
        preview = reconcile_lifecycleless_standard_name_stubs(gc=client)
        applied = reconcile_lifecycleless_standard_name_stubs(
            apply=True, manifest_sha256=preview["manifest_sha256"], gc=client
        )

        assert applied["changed"] == 1
        assert applied["sources_reset"] == 1
        assert client.query(
            "MATCH (:StandardName {id: $child})-[edge:HAS_PARENT]->"
            "(parent:StandardName {id: $parent}) "
            "RETURN count(edge) AS edges, parent.origin AS origin, "
            "parent.name_stage AS stage",
            child=child_id,
            parent=parent_id,
        ) == [{"edges": 1, "origin": "derived", "stage": "accepted"}]
    finally:
        client.query(
            "MATCH (node) WHERE node.id IN $ids "
            "OR node.id STARTS WITH 'source-retry:source-reset:' "
            "DETACH DELETE node",
            ids=[parent_id, child_id, dd_path, source_id, f"derived:{parent_id}"],
        )


@pytest.mark.graph
def test_late_failure_rolls_back_materialized_parent(
    disposable_neo4j: tuple[str, str],
) -> None:
    parent_id = "electron_temperature"
    child_id = "maximum_of_electron_temperature"
    dead_id = "dead_link_endpoint"
    client = _disposable_client(disposable_neo4j, "late-failure-rollback")
    try:
        client.query(
            "CREATE (child:StandardName {id: $child, name_stage: 'drafted', "
            "unit: 'eV', kind: 'scalar', physics_domain: 'transport'}) "
            "CREATE (parent:StandardName {id: $parent}) "
            "CREATE (child)-[:HAS_PARENT {operator: 'maximum', "
            "operator_kind: 'unary_prefix'}]->(parent) "
            "CREATE (:StandardName {id: $dead})",
            child=child_id,
            parent=parent_id,
            dead=dead_id,
        )
        preview = reconcile_lifecycleless_standard_name_stubs(gc=client)
        with (
            patch(
                "imas_codex.standard_names.graph_ops._delete_derived_parent_nodes",
                side_effect=RuntimeError("injected late failure"),
            ),
            pytest.raises(RuntimeError, match="injected late failure"),
        ):
            reconcile_lifecycleless_standard_name_stubs(
                apply=True,
                manifest_sha256=preview["manifest_sha256"],
                gc=client,
            )

        assert client.query(
            "MATCH (parent:StandardName {id: $parent}) "
            "RETURN parent.name_stage AS stage, parent.origin AS origin",
            parent=parent_id,
        )[0] == {"stage": None, "origin": None}
        assert (
            client.query(
                "MATCH (source:StandardNameSource {id: $source}) "
                "RETURN count(source) AS count",
                source=f"derived:{parent_id}",
            )[0]["count"]
            == 0
        )
        assert (
            client.query(
                "MATCH (:StandardName {id: $dead}) RETURN count(*) AS count",
                dead=dead_id,
            )[0]["count"]
            == 1
        )
    finally:
        client.query(
            "MATCH (node) WHERE node.id IN $ids DETACH DELETE node",
            ids=[parent_id, child_id, dead_id, f"derived:{parent_id}"],
        )


@pytest.mark.graph
def test_authoritative_sibling_deletes_spurious_stub_and_repairs_scalar(
    disposable_neo4j: tuple[str, str],
) -> None:
    stub_id = "fast_neutral_beam_motional_stark_wavelength"
    accepted_id = (
        "difference_of_fast_neutral_beam_motional_stark_wavelength_and_"
        "fast_neutral_beam_reference_wavelength_of_spectral_line"
    )
    dd_path = "charge_exchange/channel/bes/lorentz_shift"
    source_id = f"dd:{dd_path}"
    client = _disposable_client(disposable_neo4j, "spurious-stub-binding")
    try:
        client.query(
            "CREATE (stub:StandardName {id: $stub}) "
            "CREATE (accepted:StandardName {id: $accepted, name_stage: 'accepted', "
            "docs_stage: 'accepted', status: 'draft', origin: 'catalog_edit', "
            "validation_status: 'valid', reviewer_score_name: 1.0, "
            "description: 'Accepted wavelength difference.'}) "
            "CREATE (dd:IMASNode {id: $dd_path, units: 'm'}) "
            "CREATE (source:StandardNameSource {id: $source, source_type: 'dd', "
            "source_id: $dd_path, status: 'composed', produced_sn_id: null}) "
            "CREATE (source)-[:FROM_DD_PATH]->(dd) "
            "CREATE (source)-[:PRODUCED_NAME]->(stub) "
            "CREATE (source)-[:PRODUCED_NAME]->(accepted)",
            stub=stub_id,
            accepted=accepted_id,
            dd_path=dd_path,
            source=source_id,
        )

        preview = reconcile_lifecycleless_standard_name_stubs(gc=client)
        assert preview["counts"] == {
            "materialize-as-derived-parent": 0,
            "delete-as-dead-link-stub": 1,
            "rebind-source": 0,
            "refused": 0,
        }
        signed_row = preview["manifest"]["rows"]["delete-as-dead-link-stub"][0]
        assert (
            signed_row["spurious_binding_authority"][0]["authoritative_target_id"]
            == accepted_id
        )

        applied = reconcile_lifecycleless_standard_name_stubs(
            apply=True,
            manifest_sha256=preview["manifest_sha256"],
            gc=client,
        )

        assert applied["changed"] == 1
        assert applied["sources_reconciled"] == 1
        state = client.query(
            "MATCH (source:StandardNameSource {id: $source}) "
            "MATCH (accepted:StandardName {id: $accepted}) "
            "OPTIONAL MATCH (stub:StandardName {id: $stub}) "
            "RETURN stub IS NULL AS stub_deleted, "
            "source.produced_sn_id AS produced, "
            "COUNT { (source)-[:PRODUCED_NAME]->(accepted) } AS accepted_edges, "
            "accepted.name_stage AS accepted_stage, "
            "accepted.validation_status AS accepted_validation, "
            "accepted.description AS accepted_description",
            source=source_id,
            accepted=accepted_id,
            stub=stub_id,
        )[0]
        assert state == {
            "stub_deleted": True,
            "produced": accepted_id,
            "accepted_edges": 1,
            "accepted_stage": "accepted",
            "accepted_validation": "valid",
            "accepted_description": "Accepted wavelength difference.",
        }
    finally:
        client.query(
            "MATCH (node) WHERE node.id IN $ids "
            "OR node.manifest_sha256 = $manifest_sha256 DETACH DELETE node",
            ids=[stub_id, accepted_id, dd_path, source_id],
            manifest_sha256=locals().get("preview", {}).get("manifest_sha256"),
        )


@pytest.mark.graph
def test_nonaccepted_sibling_refuses_spurious_stub_deletion(
    disposable_neo4j: tuple[str, str],
) -> None:
    stub_id = "fast_neutral_beam_motional_stark_wavelength"
    sibling_id = "fast_neutral_beam_reference_wavelength_of_spectral_line"
    dd_path = "charge_exchange/channel/bes/lorentz_shift"
    source_id = f"dd:{dd_path}"
    client = _disposable_client(disposable_neo4j, "spurious-stub-refusal")
    try:
        client.query(
            "CREATE (stub:StandardName {id: $stub}) "
            "CREATE (sibling:StandardName {id: $sibling, name_stage: 'drafted', "
            "status: 'draft', origin: 'pipeline', validation_status: 'valid'}) "
            "CREATE (dd:IMASNode {id: $dd_path, units: 'm'}) "
            "CREATE (source:StandardNameSource {id: $source, source_type: 'dd', "
            "source_id: $dd_path, status: 'composed', produced_sn_id: null}) "
            "CREATE (source)-[:FROM_DD_PATH]->(dd) "
            "CREATE (source)-[:PRODUCED_NAME]->(stub) "
            "CREATE (source)-[:PRODUCED_NAME]->(sibling)",
            stub=stub_id,
            sibling=sibling_id,
            dd_path=dd_path,
            source=source_id,
        )

        with pytest.raises(
            LifecyclelessStubConflict, match="incomplete DD source or unit authority"
        ):
            reconcile_lifecycleless_standard_name_stubs(gc=client)

        state = client.query(
            "MATCH (source:StandardNameSource {id: $source}) "
            "MATCH (stub:StandardName {id: $stub}) "
            "RETURN source.produced_sn_id AS produced, "
            "COUNT { (source)-[:PRODUCED_NAME]->(:StandardName) } AS bindings, "
            "stub.name_stage AS stub_stage",
            source=source_id,
            stub=stub_id,
        )[0]
        assert state == {"produced": None, "bindings": 2, "stub_stage": None}
    finally:
        client.query(
            "MATCH (node) WHERE node.id IN $ids DETACH DELETE node",
            ids=[stub_id, sibling_id, dd_path, source_id],
        )
