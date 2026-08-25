"""Disposable-graph contract for governed successor-rewire retirement."""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from neo4j import GraphDatabase

from imas_codex.graph.client import GraphClient
from imas_codex.graph.profiles import resolve_neo4j
from imas_codex.standard_names.derivation import derive_edges
from imas_codex.standard_names.repair_authority import build_repair_authority
from imas_codex.standard_names.signed_manifest import apply_signed_manifest

_OPERATION = "retire_unauthorized_has_parent_relocations"
_REASON = "restore current structural derivation after successor relocation"
_SELECTION = {
    "id": "artifact-rows",
    "mode": "exact_complete_signed_cohort",
    "predicate": "artifact-rows",
}
_SPECTRAL_CHILD = "spectral_signal_to_noise_ratio_of_spectrometer_channel"
_SPECTRAL_PARENT = "signal_to_noise_ratio_of_spectrometer_channel"
_SPECTRAL_TIP = "logarithm_of_spectral_signal_to_noise_ratio_of_spectrometer_channel"
_LEGITIMATE_CHILD = "maximum_of_electron_temperature"
_LEGITIMATE_PARENT = "electron_temperature"
_MISSING_PATH_CHILD = "maximum_of_ion_temperature"
_MISSING_PATH_PARENT = "ion_temperature"


def _derived_properties(child_id: str) -> dict[str, Any]:
    edge = next(
        edge for edge in derive_edges(child_id) if edge.edge_type == "HAS_PARENT"
    )
    return {key: value for key, value in edge.props.items() if value is not None}


def _node(node_id: str) -> dict[str, str]:
    return {"id": node_id, "kind": "node", "graph_label": "StandardName"}


def _row(
    *,
    row_id: str,
    child_id: str,
    incumbent_id: str,
    replacement_id: str,
    relationship_id: str,
    properties: dict[str, Any],
) -> dict[str, Any]:
    return {
        "id": row_id,
        "identity": {
            "id": child_id,
            "kind": "standard_name",
            "target_id": replacement_id,
        },
        "participants": [
            _node(child_id),
            _node(incumbent_id),
            _node(replacement_id),
            {
                "id": relationship_id,
                "kind": "relationship",
                "graph_label": "HAS_PARENT",
            },
        ],
        "selection": _SELECTION,
        "mutations": [
            {
                "id": "restore-derived-parent",
                "order": 0,
                "kind": "recompute_projection",
                "participant_id": relationship_id,
                "arguments": {
                    "relationship_type": "HAS_PARENT",
                    "start_id": child_id,
                    "old_end_id": incumbent_id,
                    "new_end_id": replacement_id,
                    "properties": properties,
                },
            }
        ],
        "guards": [
            {
                "id": "current-structural-derivation",
                "kind": "semantic_authority",
                "implementation": "current-structural-derivation",
                "participant_ids": [],
            },
            {
                "id": "derivable-parent-path-preservation",
                "kind": "semantic_authority",
                "implementation": "derivable-parent-path-preservation",
                "participant_ids": [],
            },
            {
                "id": "out-of-allowlist-immutability",
                "kind": "collateral_immutability",
                "implementation": "out-of-allowlist-immutability",
                "participant_ids": [],
            },
        ],
        "orphan_policy": "refuse",
    }


@pytest.fixture(scope="module")
def disposable_neo4j() -> Iterator[tuple[str, str]]:
    uri = os.environ.get("IMAS_CODEX_TEST_NEO4J_URI")
    if not uri:
        pytest.skip("IMAS_CODEX_TEST_NEO4J_URI is not configured")
    if os.environ.get("IMAS_CODEX_TEST_NEO4J_EPHEMERAL") != "1":
        pytest.fail("successor-rewire retirement requires a disposable graph")
    project_uri = resolve_neo4j(auto_tunnel=False).uri
    if uri.rstrip("/") == project_uri.rstrip("/"):
        pytest.fail("successor-rewire retirement refuses the project graph URI")
    password = os.environ.get("IMAS_CODEX_TEST_NEO4J_PASSWORD", "")
    auth = ("neo4j", password) if password else None
    with GraphDatabase.driver(uri, auth=auth) as driver:
        driver.verify_connectivity()
        driver.execute_query("MATCH (node) DETACH DELETE node")
    yield uri, password


@pytest.fixture
def client(disposable_neo4j: tuple[str, str]) -> Iterator[GraphClient]:
    uri, password = disposable_neo4j
    graph = GraphClient(
        uri=uri,
        username="neo4j",
        password=password,
        graph_name="successor-rewire-retirement",
    )
    graph.query("MATCH (node) DETACH DELETE node")
    yield graph
    graph.query("MATCH (node) DETACH DELETE node")
    graph.close()


def _seed(client: GraphClient) -> None:
    node_ids = [
        _SPECTRAL_CHILD,
        _SPECTRAL_PARENT,
        _SPECTRAL_TIP,
        _LEGITIMATE_CHILD,
        _LEGITIMATE_PARENT,
        "electron_density",
        _MISSING_PATH_CHILD,
        _MISSING_PATH_PARENT,
        "ion_density",
        *[f"spectral_refinement_{index}" for index in range(1, 6)],
    ]
    client.query(
        """
        UNWIND $node_ids AS node_id
        CREATE (:StandardName {
          id: node_id, name_stage: 'accepted', validation_status: 'valid',
          status: 'draft'
        })
        """,
        node_ids=node_ids,
    )
    chain = [
        _SPECTRAL_TIP,
        "spectral_refinement_5",
        "spectral_refinement_4",
        "spectral_refinement_3",
        "spectral_refinement_2",
        "spectral_refinement_1",
        _SPECTRAL_PARENT,
    ]
    client.query(
        """
        UNWIND range(0, size($chain) - 2) AS position
        MATCH (successor:StandardName {id: $chain[position]}),
              (predecessor:StandardName {id: $chain[position + 1]})
        CREATE (successor)-[:REFINED_FROM]->(predecessor)
        SET predecessor.name_stage = 'superseded',
            predecessor.status = 'superseded'
        """,
        chain=chain,
    )
    pairs = [
        {
            "child": _SPECTRAL_CHILD,
            "parent": _SPECTRAL_TIP,
            "properties": _derived_properties(_SPECTRAL_CHILD),
        },
        {
            "child": _LEGITIMATE_CHILD,
            "parent": _LEGITIMATE_PARENT,
            "properties": _derived_properties(_LEGITIMATE_CHILD),
        },
        {
            "child": _MISSING_PATH_CHILD,
            "parent": "ion_density",
            "properties": _derived_properties(_MISSING_PATH_CHILD),
        },
    ]
    client.query(
        """
        UNWIND $pairs AS pair
        MATCH (child:StandardName {id: pair.child}),
              (parent:StandardName {id: pair.parent})
        CREATE (child)-[edge:HAS_PARENT]->(parent)
        SET edge = pair.properties
        """,
        pairs=pairs,
    )


def _authority_rows(client: GraphClient) -> list[dict[str, Any]]:
    requests = [
        {
            "row_id": "spectral-relocation",
            "child": _SPECTRAL_CHILD,
            "incumbent": _SPECTRAL_TIP,
            "replacement": _SPECTRAL_PARENT,
        },
        {
            "row_id": "legitimate-unary-parent",
            "child": _LEGITIMATE_CHILD,
            "incumbent": _LEGITIMATE_PARENT,
            "replacement": "electron_density",
        },
        {
            "row_id": "missing-derived-parent-path",
            "child": _MISSING_PATH_CHILD,
            "incumbent": "ion_density",
            "replacement": "electron_density",
        },
    ]
    rows = client.query(
        """
        UNWIND $requests AS request
        MATCH (:StandardName {id: request.child})-[edge:HAS_PARENT]->
              (:StandardName {id: request.incumbent})
        RETURN request, elementId(edge) AS relationship_id,
               properties(edge) AS properties
        ORDER BY request.row_id
        """,
        requests=requests,
    )
    return [
        _row(
            row_id=str(item["request"]["row_id"]),
            child_id=str(item["request"]["child"]),
            incumbent_id=str(item["request"]["incumbent"]),
            replacement_id=str(item["request"]["replacement"]),
            relationship_id=str(item["relationship_id"]),
            properties=dict(item["properties"]),
        )
        for item in rows
    ]


def _write_authority(client: GraphClient, path: Path):
    built = build_repair_authority(
        {
            "operation_id": _OPERATION,
            "authority_mode": "external_reviewed",
            "rows": _authority_rows(client),
            "selection": _SELECTION,
            "receipt_policy": {
                "id": "one-per-removed-successor-rewire-edge",
                "operation": _OPERATION,
                "cardinality": "per_target",
                "expected_count": "admitted_rows",
                "link_participant_kind": "standard_name",
                "replay_projection": ["manifest_sha256", "row_id"],
            },
            "orphan_policy": "refuse",
        }
    )
    path.write_bytes(built.content)
    return built


def _graph_snapshot(client: GraphClient) -> list[dict[str, Any]]:
    return client.query(
        """
        MATCH (node)
        OPTIONAL MATCH (node)-[relationship]->(other)
        RETURN elementId(node) AS node_id, properties(node) AS properties,
               collect({id: elementId(relationship), type: type(relationship),
                        properties: properties(relationship),
                        other: elementId(other)}) AS relationships
        ORDER BY node_id
        """
    )


@pytest.mark.graph
def test_closed_program_admits_only_unauthorized_relocation_and_replays_write_free(
    client: GraphClient, tmp_path: Path
) -> None:
    _seed(client)
    path = tmp_path / "successor-rewire-retirement.json"
    built = _write_authority(client, path)
    before_preview = _graph_snapshot(client)

    preview = apply_signed_manifest(
        path,
        authority_file_sha256=built.file_sha256,
        authority_payload_sha256=built.payload_sha256,
        reason=_REASON,
        gc=client,
    )

    assert preview["counts"] == {"authority_rows": 3, "admitted": 1, "refused": 2}
    assert preview["manifest"]["admitted_row_ids"] == ["spectral-relocation"]
    assert preview["refusals"] == [
        {
            "row_id": "legitimate-unary-parent",
            "reason": "current derivation still authorizes incumbent HAS_PARENT tip",
        },
        {
            "row_id": "missing-derived-parent-path",
            "reason": "removal would leave a derivable HAS_PARENT path absent",
        },
    ]
    assert _graph_snapshot(client) == before_preview

    applied = apply_signed_manifest(
        path,
        authority_file_sha256=built.file_sha256,
        authority_payload_sha256=built.payload_sha256,
        reason=_REASON,
        apply=True,
        manifest_sha256=preview["manifest_sha256"],
        gc=client,
    )

    assert applied["outcome"] == "applied"
    assert applied["changed"] == 1
    assert applied["receipt_rows"] == 1
    assert client.query(
        """
        MATCH (:StandardName {id: $child})-[:HAS_PARENT]->(parent:StandardName)
        RETURN collect(parent.id) AS parents
        """,
        child=_SPECTRAL_CHILD,
    ) == [{"parents": [_SPECTRAL_PARENT]}]
    receipts = client.query(
        """
        MATCH (change:StandardNameChange {
          operation: $operation, manifest_sha256: $manifest_sha256
        })
        RETURN change.row_id AS row_id
        """,
        operation=_OPERATION,
        manifest_sha256=preview["manifest_sha256"],
    )
    assert receipts == [{"row_id": "spectral-relocation"}]
    before_replay = _graph_snapshot(client)

    replay = apply_signed_manifest(
        path,
        authority_file_sha256=built.file_sha256,
        authority_payload_sha256=built.payload_sha256,
        reason=_REASON,
        apply=True,
        manifest_sha256=preview["manifest_sha256"],
        gc=client,
    )

    assert replay["outcome"] == "already_applied"
    assert replay["changed"] == 0
    assert replay["persistent_writes"] == 0
    assert _graph_snapshot(client) == before_replay
