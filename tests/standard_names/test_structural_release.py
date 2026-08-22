"""Disposable-graph contract for signed structural parent release."""

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
    SignedManifestAuthorityError,
    apply_signed_manifest,
    signed_payload_sha256,
)

_SELECTION = {
    "id": "artifact-rows",
    "mode": "exact_complete_signed_cohort",
    "predicate": "artifact-rows",
}
_OPERATION = "release_structural_standard_name_children"
_REASON = "release the exact independently adjudicated structural child cohort"
_PARENT = "area_of_flux_surface"
_RELEASED_CHILD = "surface_area_of_flux_surface"
_OTHER_CHILD = (
    "derivative_with_respect_to_normalized_poloidal_flux_coordinate_of_"
    "area_of_flux_surface"
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


def _release_row(
    relationship_id: str, relationship_properties: dict[str, Any]
) -> dict[str, Any]:
    return {
        "id": _RELEASED_CHILD,
        "identity": {
            "id": _RELEASED_CHILD,
            "kind": "standard_name",
            "target_id": _RELEASED_CHILD,
        },
        "participants": [
            _node(_RELEASED_CHILD),
            _node(_PARENT),
            {
                "id": relationship_id,
                "kind": "relationship",
                "graph_label": "HAS_PARENT",
            },
        ],
        "selection": _SELECTION,
        "mutations": [
            {
                "id": "release-parent-edge",
                "order": 0,
                "kind": "recompute_projection",
                "participant_id": relationship_id,
                "arguments": {
                    "relationship_type": "HAS_PARENT",
                    "start_id": _RELEASED_CHILD,
                    "old_end_id": _PARENT,
                    "new_end_id": None,
                    "properties": relationship_properties,
                },
            }
        ],
        "guards": [_guard()],
        "orphan_policy": "refuse",
    }


def _reparent_row() -> dict[str, Any]:
    child_id = "synthetic_reparent_child"
    old_parent_id = "synthetic_old_parent"
    new_parent_id = "synthetic_new_parent"
    relationship_id = "synthetic-parent-edge"
    return {
        "id": child_id,
        "identity": {
            "id": child_id,
            "kind": "standard_name",
            "target_id": new_parent_id,
        },
        "participants": [
            _node(child_id),
            _node(old_parent_id),
            _node(new_parent_id),
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
                    "old_end_id": old_parent_id,
                    "new_end_id": new_parent_id,
                    "properties": {},
                },
            }
        ],
        "guards": [_guard()],
        "orphan_policy": "refuse",
    }


def _write_authority(path: Path, rows: list[dict[str, Any]]) -> tuple[str, str]:
    authority: dict[str, Any] = {
        "schema": "imas-codex.repair-authority.v1",
        "operation_id": "structural-release",
        "authority_mode": "external_reviewed",
        "rows": rows,
        "repair_rows": [row["id"] for row in rows],
        "selection": _SELECTION,
        "receipt_policy": {
            "id": "one-per-structural-child-release",
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
        pytest.fail("structural release requires a disposable graph")
    project_uri = resolve_neo4j(auto_tunnel=False).uri
    if uri.rstrip("/") == project_uri.rstrip("/"):
        pytest.fail("structural release refuses the project graph URI")
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
        graph_name="structural-release",
    )
    graph.query("MATCH (node) DETACH DELETE node")
    yield graph
    graph.query("MATCH (node) DETACH DELETE node")
    graph.close()


def _seed(client: GraphClient) -> None:
    client.query(
        """
        CREATE (parent:StandardName {
          id: $parent_id, name_stage: 'accepted', validation_status: 'valid',
          status: 'draft', origin: 'derived', source_paths: ['derived:area']
        })
        CREATE (released:StandardName {
          id: $released_child, name_stage: 'accepted', validation_status: 'valid',
          status: 'draft', origin: 'catalog_edit',
          source_paths: ['dd:equilibrium/grid/surface',
                         'dd:equilibrium/profiles_1d/surface'],
          reviewer_score_name: 0.97, reviewer_score_docs: 0.96
        })
        CREATE (other:StandardName {
          id: $other_child, name_stage: 'accepted', validation_status: 'valid',
          status: 'draft', origin: 'pipeline',
          source_paths: ['dd:equilibrium/profiles_1d/darea_dpsi']
        })
        CREATE (released)-[:HAS_PARENT {
          operator_kind: 'unary', operator: 'surface', evidence: ['review']
        }]->(parent)
        CREATE (other)-[:HAS_PARENT {
          operator_kind: 'unary', operator: 'derivative', evidence: ['dd']
        }]->(parent)
        WITH released, other
        UNWIND [
          {id: 'dd:equilibrium/grid/surface', path: 'equilibrium/grid/surface'},
          {id: 'dd:equilibrium/profiles_1d/surface',
           path: 'equilibrium/profiles_1d/surface'}
        ] AS producer
        CREATE (source:StandardNameSource {
          id: producer.id, status: 'attached', source_type: 'dd',
          source_id: producer.path, produced_sn_id: released.id,
          claimed_at: null, claim_token: null
        })
        CREATE (source)-[:PRODUCED_NAME {authority: 'dd'}]->(released)
        WITH other
        CREATE (other_source:StandardNameSource {
          id: 'dd:equilibrium/profiles_1d/darea_dpsi', status: 'attached',
          source_type: 'dd', source_id: 'equilibrium/profiles_1d/darea_dpsi',
          produced_sn_id: other.id, claimed_at: null, claim_token: null
        })
        CREATE (other_source)-[:PRODUCED_NAME {authority: 'dd'}]->(other)
        """,
        parent_id=_PARENT,
        released_child=_RELEASED_CHILD,
        other_child=_OTHER_CHILD,
    )


def _authority_row(client: GraphClient) -> dict[str, Any]:
    row = client.query(
        """
        MATCH (:StandardName {id: $child_id})-[parent:HAS_PARENT]->
              (:StandardName {id: $parent_id})
        RETURN elementId(parent) AS relationship_id,
               properties(parent) AS relationship_properties
        """,
        child_id=_RELEASED_CHILD,
        parent_id=_PARENT,
    )[0]
    return _release_row(
        str(row["relationship_id"]), dict(row["relationship_properties"])
    )


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
        MATCH (child:StandardName {id: $child_id})
        OPTIONAL MATCH (source:StandardNameSource)-[binding:PRODUCED_NAME]->(child)
        RETURN properties(child) AS child_properties,
               collect({source_id: source.id, source_properties: properties(source),
                        binding_properties: properties(binding)}) AS producers
        """,
        child_id=_RELEASED_CHILD,
    )


def _other_child_state(client: GraphClient) -> list[dict[str, Any]]:
    return client.query(
        """
        MATCH (child:StandardName {id: $child_id})-[parent:HAS_PARENT]->
              (target:StandardName)
        RETURN properties(child) AS child_properties,
               target.id AS parent_id, properties(parent) AS parent_properties
        """,
        child_id=_OTHER_CHILD,
    )


def _parent_ids(client: GraphClient) -> list[str]:
    rows = client.query(
        """
        MATCH (:StandardName {id: $child_id})-[:HAS_PARENT]->(parent:StandardName)
        RETURN parent.id AS parent_id ORDER BY parent_id
        """,
        child_id=_RELEASED_CHILD,
    )
    return [str(row["parent_id"]) for row in rows]


def test_registry_and_loader_admit_release_but_refuse_mixed_structural_programs(
    tmp_path: Path,
) -> None:
    existing_kinds = {
        RepairMutationKind.set_properties.value,
        RepairMutationKind.delete.value,
        RepairMutationKind.supersede.value,
        RepairMutationKind.detach.value,
        RepairMutationKind.delete_relationship.value,
        RepairMutationKind.add_relationship.value,
        RepairMutationKind.recompute_projection.value,
    }
    assert operator._SIGNED_MANIFEST_MUTATION_KINDS == existing_kinds
    release_row = _release_row("release-edge", {"operator": "surface"})
    authority = tmp_path / "structural-release-authority.json"
    file_sha256, payload_sha256 = _write_authority(authority, [release_row])

    loaded = operator._load_authority(
        authority,
        expected_file_sha256=file_sha256,
        expected_payload_sha256=payload_sha256,
    )

    assert len(loaded.rows) == 1
    assert operator._is_structural_release(loaded.rows[0])
    assert not operator._is_structural_reparent(loaded.rows[0])

    mixed_file_sha256, mixed_payload_sha256 = _write_authority(
        authority, [release_row, _reparent_row()]
    )
    with pytest.raises(
        SignedManifestAuthorityError,
        match="structural authority cannot mix reparent and release programs",
    ):
        operator._load_authority(
            authority,
            expected_file_sha256=mixed_file_sha256,
            expected_payload_sha256=mixed_payload_sha256,
        )


@pytest.mark.graph
def test_signed_preview_release_receipt_and_write_free_replay(
    client: GraphClient, tmp_path: Path
) -> None:
    _seed(client)
    authority_path = tmp_path / "structural-release-authority.json"
    file_sha256, payload_sha256 = _write_authority(
        authority_path, [_authority_row(client)]
    )
    child_authority_before = _child_authority(client)
    other_child_before = _other_child_state(client)
    graph_before_preview = _graph_snapshot(client)

    preview = _preview(authority_path, file_sha256, payload_sha256, client)

    assert preview["outcome"] == "would_apply"
    assert preview["counts"] == {
        "authority_rows": 1,
        "admitted": 1,
        "refused": 0,
    }
    assert preview["would_change"] == 1
    assert preview["manifest"]["admitted_row_ids"] == [_RELEASED_CHILD]
    assert _graph_snapshot(client) == graph_before_preview

    applied = _apply(
        authority_path,
        file_sha256,
        payload_sha256,
        client,
        preview["manifest_sha256"],
    )

    assert applied["outcome"] == "applied"
    assert applied["changed"] == 1
    assert applied["mutations"] == 1
    assert applied["receipt_rows"] == 1
    assert applied["persistent_writes"] == 2
    assert _parent_ids(client) == []
    assert _child_authority(client) == child_authority_before
    assert _other_child_state(client) == other_child_before
    receipts_before = client.query(
        """
        MATCH (child:StandardName {id: $child_id})-[:HAS_INTERNAL_CHANGE]->
              (change:StandardNameChange {
                operation: $operation, manifest_sha256: $manifest_sha256
              })
        RETURN properties(change) AS properties
        """,
        child_id=_RELEASED_CHILD,
        operation=_OPERATION,
        manifest_sha256=preview["manifest_sha256"],
    )
    assert len(receipts_before) == 1
    assert receipts_before[0]["properties"]["row_id"] == _RELEASED_CHILD
    assert receipts_before[0]["properties"]["mutation_kinds"] == [
        "recompute_projection"
    ]
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
    assert replay["receipt_rows"] == 1
    assert _graph_snapshot(client) == graph_before_replay
    assert (
        client.query(
            """
            MATCH (child:StandardName {id: $child_id})-[:HAS_INTERNAL_CHANGE]->
                  (change:StandardNameChange {
                    operation: $operation, manifest_sha256: $manifest_sha256
                  })
            RETURN properties(change) AS properties
            """,
            child_id=_RELEASED_CHILD,
            operation=_OPERATION,
            manifest_sha256=preview["manifest_sha256"],
        )
        == receipts_before
    )


@pytest.mark.graph
def test_verbatim_refusal_preserves_the_complete_family(
    client: GraphClient, tmp_path: Path
) -> None:
    _seed(client)
    authority_path = tmp_path / "structural-release-authority.json"
    file_sha256, payload_sha256 = _write_authority(
        authority_path, [_authority_row(client)]
    )
    client.query(
        """
        CREATE (extra:StandardName {
          id: 'unexpected_area_parent', name_stage: 'accepted',
          validation_status: 'valid', status: 'draft', origin: 'derived'
        })
        WITH extra
        MATCH (child:StandardName {id: $child_id})
        CREATE (child)-[:HAS_PARENT {operator: 'unexpected'}]->(extra)
        """,
        child_id=_RELEASED_CHILD,
    )
    graph_before = _graph_snapshot(client)

    preview = _preview(authority_path, file_sha256, payload_sha256, client)

    refusal_reason = (
        "signed structural release closure does not match exact incumbent parent"
    )
    assert preview["outcome"] == "refused"
    assert preview["would_change"] == 0
    assert preview["counts"] == {
        "authority_rows": 1,
        "admitted": 0,
        "refused": 1,
    }
    assert preview["refusals"] == [
        {"row_id": _RELEASED_CHILD, "reason": refusal_reason}
    ]

    refused = _apply(
        authority_path,
        file_sha256,
        payload_sha256,
        client,
        preview["manifest_sha256"],
    )

    assert refused["outcome"] == "refused"
    assert refused["changed"] == 0
    assert refused["refusals"] == [
        {"row_id": _RELEASED_CHILD, "reason": refusal_reason}
    ]
    assert _graph_snapshot(client) == graph_before
    assert client.query(
        """
        MATCH (change:StandardNameChange {operation: $operation})
        RETURN count(change) AS count
        """,
        operation=_OPERATION,
    ) == [{"count": 0}]
