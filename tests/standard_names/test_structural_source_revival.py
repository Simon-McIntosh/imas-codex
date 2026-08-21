"""Disposable-graph contract for signed structural-source revival."""

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
_OPERATION = "revive_structural_standard_name_sources"
_REASON = "revive the exact signed structural provenance sources"
_PARENT_IDS = ["electron_diffusivity", "ion_diffusivity"]


def _node(node_id: str, label: str) -> dict[str, Any]:
    return {"id": node_id, "kind": "node", "graph_label": label}


def _parent_relationship(element_id: str) -> dict[str, Any]:
    return {
        "id": element_id,
        "kind": "relationship",
        "graph_label": "HAS_PARENT",
    }


def _guard() -> dict[str, Any]:
    return {
        "id": "out-of-allowlist-immutability",
        "kind": "collateral_immutability",
        "implementation": "out-of-allowlist-immutability",
        "participant_ids": [],
    }


def _revival_row(
    *, parent_id: str, child_ids: list[str], parent_relationship_ids: list[str]
) -> dict[str, Any]:
    source_id = f"derived:{parent_id}"
    return {
        "id": source_id,
        "identity": {
            "id": source_id,
            "kind": "source",
            "source_id": source_id,
            "target_id": parent_id,
        },
        "participants": [
            _node(source_id, "StandardNameSource"),
            _node(parent_id, "StandardName"),
            *[_node(child_id, "StandardName") for child_id in child_ids],
            *[
                _parent_relationship(relationship_id)
                for relationship_id in parent_relationship_ids
            ],
        ],
        "selection": _SELECTION,
        "mutations": [
            {
                "id": "restore-binding",
                "order": 0,
                "kind": "add_relationship",
                "participant_id": parent_id,
                "arguments": {
                    "relationship_type": "PRODUCED_NAME",
                    "start_id": source_id,
                    "end_id": parent_id,
                },
            },
            {
                "id": "restore-source-lifecycle",
                "order": 1,
                "kind": "set_properties",
                "participant_id": source_id,
                "arguments": {
                    "properties": {
                        "status": "composed",
                        "source_type": "derived",
                        "source_id": parent_id,
                        "batch_key": "derived_parent",
                        "produced_sn_id": parent_id,
                        "claimed_at": None,
                        "claim_token": None,
                    }
                },
            },
        ],
        "guards": [_guard()],
        "orphan_policy": "refuse",
    }


def _write_authority(path: Path, rows: list[dict[str, Any]]) -> tuple[str, str]:
    authority: dict[str, Any] = {
        "schema": "imas-codex.repair-authority.v1",
        "operation_id": "structural-source-revival",
        "authority_mode": "external_reviewed",
        "rows": rows,
        "repair_rows": [row["id"] for row in rows],
        "selection": _SELECTION,
        "receipt_policy": {
            "id": "one-per-structural-source-revival",
            "operation": _OPERATION,
            "cardinality": "per_target",
            "expected_count": "admitted_rows",
            "link_participant_kind": "source",
            "replay_projection": ["manifest_sha256", "row_id"],
        },
        "orphan_policy": "refuse",
    }
    payload_sha256 = signed_payload_sha256(authority)
    authority["signature"] = {
        "canonicalization": "json-sort-keys-v1",
        "sha256": payload_sha256,
    }
    raw = json.dumps(authority, sort_keys=True, indent=2).encode()
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest(), payload_sha256


@pytest.fixture(scope="module")
def disposable_neo4j() -> Iterator[tuple[str, str]]:
    uri = os.environ.get("IMAS_CODEX_TEST_NEO4J_URI")
    if not uri:
        pytest.skip("IMAS_CODEX_TEST_NEO4J_URI is not configured")
    if os.environ.get("IMAS_CODEX_TEST_NEO4J_EPHEMERAL") != "1":
        pytest.fail("structural-source revival requires a disposable graph")
    project_uri = resolve_neo4j(auto_tunnel=False).uri
    if uri.rstrip("/") == project_uri.rstrip("/"):
        pytest.fail("structural-source revival refuses the project graph URI")
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
        graph_name="structural-source-revival",
    )
    graph.query("MATCH (node) DETACH DELETE node")
    yield graph
    graph.query("MATCH (node) DETACH DELETE node")
    graph.close()


def _seed_structural_cohort(client: GraphClient, parent_ids: list[str]) -> None:
    client.query(
        """
        UNWIND $parent_ids AS parent_id
        CREATE (parent:StandardName {
          id: parent_id, name_stage: 'accepted', status: 'draft', origin: 'derived'
        })
        CREATE (source:StandardNameSource {
          id: 'derived:' + parent_id,
          status: 'stale',
          source_type: 'derived',
          source_id: parent_id,
          batch_key: 'derived_parent',
          produced_sn_id: null,
          claimed_at: null,
          claim_token: null
        })
        FOREACH (qualifier IN ['effective', 'parallel', 'poloidal'] |
          CREATE (child:StandardName {
            id: qualifier + '_' + parent_id,
            name_stage: 'accepted',
            status: 'draft',
            origin: 'pipeline'
          })
          CREATE (child)-[:HAS_PARENT]->(parent)
        )
        """,
        parent_ids=parent_ids,
    )


def _authority_rows(client: GraphClient, parent_ids: list[str]) -> list[dict[str, Any]]:
    rows = client.query(
        """
        UNWIND $parent_ids AS parent_id
        MATCH (child:StandardName)-[parent:HAS_PARENT]->
              (:StandardName {id: parent_id})
        WITH parent_id, child, parent ORDER BY child.id
        RETURN parent_id,
               collect(child.id) AS child_ids,
               collect(elementId(parent)) AS relationship_ids
        ORDER BY parent_id
        """,
        parent_ids=parent_ids,
    )
    return [
        _revival_row(
            parent_id=str(row["parent_id"]),
            child_ids=list(row["child_ids"]),
            parent_relationship_ids=list(row["relationship_ids"]),
        )
        for row in rows
    ]


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


def test_generic_registry_admits_closed_structural_source_program() -> None:
    assert operator._GENERIC_MUTATION_KINDS == {
        RepairMutationKind.set_properties.value,
        RepairMutationKind.delete.value,
        RepairMutationKind.supersede.value,
        RepairMutationKind.detach.value,
        RepairMutationKind.delete_relationship.value,
        RepairMutationKind.add_relationship.value,
    }


@pytest.mark.graph
def test_signed_preview_apply_immutable_receipt_and_write_free_replay(
    client: GraphClient, tmp_path: Path
) -> None:
    _seed_structural_cohort(client, _PARENT_IDS)
    path = tmp_path / "structural-source-revival-authority.json"
    file_sha256, payload_sha256 = _write_authority(
        path, _authority_rows(client, _PARENT_IDS)
    )
    before_preview = _graph_snapshot(client)

    preview = _preview(path, file_sha256, payload_sha256, client)

    assert preview["outcome"] == "would_apply"
    assert preview["counts"] == {"authority_rows": 2, "admitted": 2, "refused": 0}
    assert preview["manifest"]["admitted_row_ids"] == [
        "derived:electron_diffusivity",
        "derived:ion_diffusivity",
    ]
    assert _graph_snapshot(client) == before_preview

    applied = _apply(
        path, file_sha256, payload_sha256, client, preview["manifest_sha256"]
    )

    assert applied["outcome"] == "applied"
    assert applied["changed"] == 2
    assert applied["mutations"] == 4
    assert applied["receipt_rows"] == 2
    assert applied["persistent_writes"] == 6
    assert client.query(
        """
        UNWIND $parent_ids AS parent_id
        MATCH (source:StandardNameSource {id: 'derived:' + parent_id})
        OPTIONAL MATCH (source)-[binding:PRODUCED_NAME]->(target:StandardName)
        RETURN source.id AS source_id, source.status AS status,
               source.produced_sn_id AS scalar,
               collect(target.id) AS targets, count(binding) AS bindings
        ORDER BY source_id
        """,
        parent_ids=_PARENT_IDS,
    ) == [
        {
            "source_id": "derived:electron_diffusivity",
            "status": "composed",
            "scalar": "electron_diffusivity",
            "targets": ["electron_diffusivity"],
            "bindings": 1,
        },
        {
            "source_id": "derived:ion_diffusivity",
            "status": "composed",
            "scalar": "ion_diffusivity",
            "targets": ["ion_diffusivity"],
            "bindings": 1,
        },
    ]
    receipt_before = client.query(
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
    graph_before_replay = _graph_snapshot(client)

    replay = _apply(
        path, file_sha256, payload_sha256, client, preview["manifest_sha256"]
    )

    assert replay["outcome"] == "already_applied"
    assert replay["changed"] == 0
    assert replay["persistent_writes"] == 0
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
        == receipt_before
    )


@pytest.mark.graph
def test_stale_state_refusal_is_verbatim(client: GraphClient, tmp_path: Path) -> None:
    _seed_structural_cohort(client, ["electron_diffusivity"])
    path = tmp_path / "changed-stale-state-authority.json"
    file_sha256, payload_sha256 = _write_authority(
        path, _authority_rows(client, ["electron_diffusivity"])
    )
    client.query(
        """
        MATCH (source:StandardNameSource {id: 'derived:electron_diffusivity'})
        SET source.status = 'composed'
        """
    )
    before = _graph_snapshot(client)

    preview = _preview(path, file_sha256, payload_sha256, client)

    assert preview["outcome"] == "refused"
    assert preview["counts"] == {"authority_rows": 1, "admitted": 0, "refused": 1}
    assert preview["refusals"] == [
        {
            "row_id": "derived:electron_diffusivity",
            "reason": "structural source status changed from signed stale state",
        }
    ]
    assert _graph_snapshot(client) == before
    assert client.query(
        "MATCH (change:StandardNameChange) RETURN count(change) AS count"
    ) == [{"count": 0}]
