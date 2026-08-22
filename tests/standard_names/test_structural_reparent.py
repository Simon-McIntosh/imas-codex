"""Disposable-graph contract for signed structural reparenting."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from neo4j import GraphDatabase

from imas_codex.graph.client import GraphClient
from imas_codex.graph.models import RepairMutationKind
from imas_codex.graph.profiles import resolve_neo4j
from imas_codex.standard_names import signed_manifest as operator
from imas_codex.standard_names.signed_manifest import (
    apply_signed_manifest,
    signed_payload_sha256,
)

_SELECTION = {
    "id": "artifact-rows",
    "mode": "exact_complete_signed_cohort",
    "predicate": "artifact-rows",
}
_OPERATION = "reparent_structural_standard_name_children"
_REASON = "relocate the exact independently adjudicated structural child cohort"
_OLD_PARENT = "electron_transport_coefficient"
_NEW_PARENT = "electron_diffusivity"
_CHILDREN = (
    "effective_electron_diffusivity",
    "parallel_electron_diffusivity",
    "poloidal_electron_diffusivity",
)


def _node(node_id: str) -> dict[str, Any]:
    return {"id": node_id, "kind": "node", "graph_label": "StandardName"}


def _guard() -> dict[str, Any]:
    return {
        "id": "out-of-allowlist-immutability",
        "kind": "collateral_immutability",
        "implementation": "out-of-allowlist-immutability",
        "participant_ids": [],
    }


def _row(
    child_id: str, relationship_id: str, relationship_properties: dict[str, Any]
) -> dict[str, Any]:
    return {
        "id": child_id,
        "identity": {
            "id": child_id,
            "kind": "standard_name",
            "target_id": _NEW_PARENT,
        },
        "participants": [
            _node(child_id),
            _node(_OLD_PARENT),
            _node(_NEW_PARENT),
            {
                "id": relationship_id,
                "kind": "relationship",
                "graph_label": "HAS_PARENT",
            },
        ],
        "selection": _SELECTION,
        "mutations": [
            {
                "id": "relocate-parent-edge",
                "order": 0,
                "kind": "recompute_projection",
                "participant_id": relationship_id,
                "arguments": {
                    "relationship_type": "HAS_PARENT",
                    "start_id": child_id,
                    "old_end_id": _OLD_PARENT,
                    "new_end_id": _NEW_PARENT,
                    "properties": relationship_properties,
                },
            }
        ],
        "guards": [_guard()],
        "orphan_policy": "refuse",
    }


def _write_authority(path: Path, rows: list[dict[str, Any]]) -> tuple[str, str]:
    authority: dict[str, Any] = {
        "schema": "imas-codex.repair-authority.v1",
        "operation_id": "structural-reparent",
        "authority_mode": "external_reviewed",
        "rows": rows,
        "repair_rows": [row["id"] for row in rows],
        "selection": _SELECTION,
        "receipt_policy": {
            "id": "one-per-structural-child-relocation",
            "operation": _OPERATION,
            "cardinality": "per_target",
            "expected_count": "admitted_rows",
            "link_participant_kind": "standard_name",
            "replay_projection": ["manifest_sha256", "row_id"],
        },
        "orphan_policy": "refuse",
    }
    payload_sha256 = signed_payload_sha256(authority)
    authority["signature"] = {
        "canonicalization": "json-sort-keys-v1",
        "sha256": payload_sha256,
    }
    content = json.dumps(authority, sort_keys=True, indent=2).encode()
    path.write_bytes(content)
    return hashlib.sha256(content).hexdigest(), payload_sha256


@pytest.fixture(scope="module")
def disposable_neo4j() -> Iterator[tuple[str, str]]:
    uri = os.environ.get("IMAS_CODEX_TEST_NEO4J_URI")
    if not uri:
        pytest.skip("IMAS_CODEX_TEST_NEO4J_URI is not configured")
    if os.environ.get("IMAS_CODEX_TEST_NEO4J_EPHEMERAL") != "1":
        pytest.fail("structural reparenting requires a disposable graph")
    project_uri = resolve_neo4j(auto_tunnel=False).uri
    if uri.rstrip("/") == project_uri.rstrip("/"):
        pytest.fail("structural reparenting refuses the project graph URI")
    password = os.environ.get("IMAS_CODEX_TEST_NEO4J_PASSWORD", "")
    auth = ("neo4j", password) if password else None
    print(f"disposable_bolt_uri={uri}")
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
        graph_name="structural-reparent",
    )
    graph.query("MATCH (node) DETACH DELETE node")
    yield graph
    graph.query("MATCH (node) DETACH DELETE node")
    graph.close()


def _seed(client: GraphClient) -> None:
    client.query(
        """
        CREATE (:StandardName {
          id: $old_parent, name_stage: 'accepted', validation_status: 'valid',
          status: 'draft', origin: 'derived', source_paths: ['derived:old']
        })
        CREATE (:StandardName {
          id: $new_parent, name_stage: 'accepted', validation_status: 'valid',
          status: 'draft', origin: 'derived', source_paths: ['derived:new']
        })
        WITH 1 AS ignored
        UNWIND range(0, size($child_ids) - 1) AS position
        CREATE (child:StandardName {
          id: $child_ids[position], name_stage: 'accepted',
          validation_status: 'valid', status: 'draft', origin: 'pipeline',
          source_paths: ['dd:transport/child/' + toString(position)],
          reviewer_score_name: 0.95 + position * 0.01
        })
        CREATE (source:StandardNameSource {
          id: 'dd:transport/child/' + toString(position), status: 'attached',
          source_type: 'dd', source_id: 'transport/child/' + toString(position),
          produced_sn_id: child.id, claimed_at: null, claim_token: null
        })
        WITH child, source, position
        MATCH (old:StandardName {id: $old_parent})
        CREATE (source)-[:PRODUCED_NAME {authority: 'dd'}]->(child)
        CREATE (child)-[:HAS_PARENT {
          operator_kind: 'unary', qualifier: 'transport', position: position,
          evidence: ['grammar', 'review']
        }]->(old)
        """,
        old_parent=_OLD_PARENT,
        new_parent=_NEW_PARENT,
        child_ids=list(_CHILDREN),
    )


def _authority_rows(client: GraphClient) -> list[dict[str, Any]]:
    rows = client.query(
        """
        UNWIND $child_ids AS child_id
        MATCH (:StandardName {id: child_id})-[parent:HAS_PARENT]->
              (:StandardName {id: $old_parent})
        RETURN child_id, elementId(parent) AS relationship_id,
               properties(parent) AS relationship_properties
        ORDER BY child_id
        """,
        child_ids=list(_CHILDREN),
        old_parent=_OLD_PARENT,
    )
    return [
        _row(
            str(item["child_id"]),
            str(item["relationship_id"]),
            dict(item["relationship_properties"]),
        )
        for item in rows
    ]


def _preview(
    path: Path, file_sha256: str, payload_sha256: str, client: GraphClient
) -> dict[str, Any]:
    return apply_signed_manifest(
        path,
        authority_file_sha256=file_sha256,
        authority_payload_sha256=payload_sha256,
        reason=_REASON,
        gc=client,
    )


def _apply(
    path: Path,
    file_sha256: str,
    payload_sha256: str,
    client: GraphClient,
    manifest_sha256: str,
) -> dict[str, Any]:
    return apply_signed_manifest(
        path,
        authority_file_sha256=file_sha256,
        authority_payload_sha256=payload_sha256,
        reason=_REASON,
        apply=True,
        manifest_sha256=manifest_sha256,
        gc=client,
    )


def _graph_snapshot(client: GraphClient) -> list[dict[str, Any]]:
    return client.query(
        """
        MATCH (node)
        OPTIONAL MATCH (node)-[relationship]->(other)
        RETURN elementId(node) AS node_id, labels(node) AS labels,
               properties(node) AS properties,
               collect({id: elementId(relationship), type: type(relationship),
                        properties: properties(relationship),
                        other: elementId(other)}) AS relationships
        ORDER BY node_id
        """
    )


def _child_authority(client: GraphClient) -> list[dict[str, Any]]:
    return client.query(
        """
        UNWIND $child_ids AS child_id
        MATCH (child:StandardName {id: child_id})
        OPTIONAL MATCH (source:StandardNameSource)-[binding:PRODUCED_NAME]->(child)
        RETURN child_id, properties(child) AS child_properties,
               collect({source_id: source.id, source_properties: properties(source),
                        binding_properties: properties(binding)}) AS producers
        ORDER BY child_id
        """,
        child_ids=list(_CHILDREN),
    )


def _parent_state(client: GraphClient) -> list[dict[str, Any]]:
    return client.query(
        """
        UNWIND $child_ids AS child_id
        MATCH (child:StandardName {id: child_id})
        OPTIONAL MATCH (child)-[parent:HAS_PARENT]->(target:StandardName)
        RETURN child_id, collect(target.id) AS parent_ids,
               collect(properties(parent)) AS relationship_properties
        ORDER BY child_id
        """,
        child_ids=list(_CHILDREN),
    )


def test_registry_and_loader_admit_the_closed_structural_program(
    tmp_path: Path,
) -> None:
    expected = {
        RepairMutationKind.set_properties.value,
        RepairMutationKind.delete.value,
        RepairMutationKind.supersede.value,
        RepairMutationKind.detach.value,
        RepairMutationKind.delete_relationship.value,
        RepairMutationKind.add_relationship.value,
        RepairMutationKind.recompute_projection.value,
    }
    assert operator._SIGNED_MANIFEST_MUTATION_KINDS == expected
    rows = [
        _row(child_id, f"relationship-{index}", {"position": index})
        for index, child_id in enumerate(_CHILDREN)
    ]
    authority = tmp_path / "structural-reparent-authority.json"
    file_sha256, payload_sha256 = _write_authority(authority, rows)

    loaded = operator._load_authority(
        authority,
        expected_file_sha256=file_sha256,
        expected_payload_sha256=payload_sha256,
    )

    assert len(loaded.rows) == 3
    assert all(operator._is_structural_reparent(row) for row in loaded.rows)


@pytest.mark.graph
def test_signed_preview_atomic_relocation_receipts_and_replay(
    client: GraphClient, tmp_path: Path
) -> None:
    _seed(client)
    authority_path = tmp_path / "structural-reparent-authority.json"
    file_sha256, payload_sha256 = _write_authority(
        authority_path, _authority_rows(client)
    )
    child_authority_before = _child_authority(client)
    edge_properties_before = {
        row["child_id"]: row["relationship_properties"][0]
        for row in _parent_state(client)
    }
    graph_before_preview = _graph_snapshot(client)

    preview = _preview(authority_path, file_sha256, payload_sha256, client)

    assert preview["outcome"] == "would_apply"
    assert preview["counts"] == {
        "authority_rows": 3,
        "admitted": 3,
        "refused": 0,
    }
    assert preview["would_change"] == 3
    assert preview["manifest"]["admitted_row_ids"] == sorted(_CHILDREN)
    assert _graph_snapshot(client) == graph_before_preview

    applied = _apply(
        authority_path,
        file_sha256,
        payload_sha256,
        client,
        preview["manifest_sha256"],
    )

    assert applied["outcome"] == "applied"
    assert applied["changed"] == 3
    assert applied["mutations"] == 3
    assert applied["receipt_rows"] == 3
    assert applied["persistent_writes"] == 6
    parent_state = _parent_state(client)
    assert [row["parent_ids"] for row in parent_state] == [[_NEW_PARENT]] * 3
    assert {
        row["child_id"]: row["relationship_properties"][0] for row in parent_state
    } == edge_properties_before
    assert _child_authority(client) == child_authority_before
    receipts_before = client.query(
        """
        MATCH (change:StandardNameChange {
          operation: $operation, manifest_sha256: $manifest_sha256
        })
        RETURN properties(change) AS properties
        ORDER BY change.row_id
        """,
        operation=_OPERATION,
        manifest_sha256=preview["manifest_sha256"],
    )
    assert [row["properties"]["row_id"] for row in receipts_before] == sorted(_CHILDREN)
    assert all(
        row["properties"]["mutation_kinds"] == ["recompute_projection"]
        for row in receipts_before
    )
    graph_before_replay = _graph_snapshot(client)

    replay = _apply(
        authority_path,
        file_sha256,
        payload_sha256,
        client,
        preview["manifest_sha256"],
    )

    assert replay["outcome"] == "already_applied"
    assert replay["changed"] == 0
    assert replay["persistent_writes"] == 0
    assert replay["receipt_rows"] == 3
    assert _graph_snapshot(client) == graph_before_replay
    assert (
        client.query(
            """
            MATCH (change:StandardNameChange {
              operation: $operation, manifest_sha256: $manifest_sha256
            })
            RETURN properties(change) AS properties
            ORDER BY change.row_id
            """,
            operation=_OPERATION,
            manifest_sha256=preview["manifest_sha256"],
        )
        == receipts_before
    )


@pytest.mark.graph
def test_one_verbatim_refusal_keeps_the_three_child_cohort_unchanged(
    client: GraphClient, tmp_path: Path
) -> None:
    _seed(client)
    authority_path = tmp_path / "structural-reparent-authority.json"
    file_sha256, payload_sha256 = _write_authority(
        authority_path, _authority_rows(client)
    )
    refused_child = _CHILDREN[1]
    client.query(
        """
        CREATE (extra:StandardName {
          id: 'ambiguous_transport_parent', name_stage: 'accepted',
          validation_status: 'valid', status: 'draft', origin: 'derived'
        })
        WITH extra
        MATCH (child:StandardName {id: $child_id})
        CREATE (child)-[:HAS_PARENT {operator_kind: 'unexpected'}]->(extra)
        """,
        child_id=refused_child,
    )
    parents_before = _parent_state(client)
    child_authority_before = _child_authority(client)

    preview = _preview(authority_path, file_sha256, payload_sha256, client)

    refusal_reason = (
        "signed structural reparent closure does not match exact incumbent parent"
    )
    assert preview["outcome"] == "refused"
    assert preview["would_change"] == 0
    assert preview["counts"] == {
        "authority_rows": 3,
        "admitted": 2,
        "refused": 1,
    }
    assert preview["refusals"] == [{"row_id": refused_child, "reason": refusal_reason}]

    refused = _apply(
        authority_path,
        file_sha256,
        payload_sha256,
        client,
        preview["manifest_sha256"],
    )

    assert refused["outcome"] == "refused"
    assert refused["changed"] == 0
    assert refused["would_change"] == 0
    assert refused["refusals"] == [{"row_id": refused_child, "reason": refusal_reason}]
    assert _parent_state(client) == parents_before
    assert _child_authority(client) == child_authority_before
    assert client.query(
        """
        MATCH (change:StandardNameChange {operation: $operation})
        RETURN count(change) AS count
        """,
        operation=_OPERATION,
    ) == [{"count": 0}]
