"""Disposable-graph contract for signed source-target reconciliation."""

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
_OPERATION = "reconcile_standard_name_source_targets"
_REASON = "reconcile the exact independently adjudicated source target closure"


def _guard(implementation: str) -> dict[str, Any]:
    kind = {
        "last-producing-source": "semantic_authority",
        "out-of-allowlist-immutability": "collateral_immutability",
    }[implementation]
    return {
        "id": implementation,
        "kind": kind,
        "implementation": implementation,
        "participant_ids": [],
    }


def _node(node_id: str, label: str) -> dict[str, Any]:
    return {"id": node_id, "kind": "node", "graph_label": label}


def _binding(element_id: str) -> dict[str, Any]:
    return {
        "id": element_id,
        "kind": "relationship",
        "graph_label": "PRODUCED_NAME",
    }


def _reconciliation_row(
    *,
    source_id: str,
    survivor_id: str,
    target_bindings: dict[str, str],
) -> dict[str, Any]:
    losing_bindings = {
        target_id: binding_id
        for target_id, binding_id in target_bindings.items()
        if target_id != survivor_id
    }
    mutations = [
        {
            "id": f"remove:{target_id}",
            "order": order,
            "kind": "delete_relationship",
            "participant_id": binding_id,
        }
        for order, (target_id, binding_id) in enumerate(sorted(losing_bindings.items()))
    ]
    mutations.append(
        {
            "id": "select-survivor",
            "order": len(mutations),
            "kind": "set_properties",
            "participant_id": source_id,
            "arguments": {"properties": {"produced_sn_id": survivor_id}},
        }
    )
    return {
        "id": source_id,
        "identity": {
            "id": source_id,
            "kind": "source",
            "source_id": source_id,
            "target_id": survivor_id,
        },
        "participants": [
            _node(source_id, "StandardNameSource"),
            *[_node(target_id, "StandardName") for target_id in target_bindings],
            *[_binding(binding_id) for binding_id in target_bindings.values()],
        ],
        "selection": _SELECTION,
        "mutations": mutations,
        "guards": [
            _guard("last-producing-source"),
            _guard("out-of-allowlist-immutability"),
        ],
        "orphan_policy": "refuse",
    }


def _write_authority(path: Path, row: dict[str, Any]) -> tuple[str, str]:
    authority: dict[str, Any] = {
        "schema": "imas-codex.repair-authority.v1",
        "operation_id": "source-target-reconciliation",
        "authority_mode": "external_reviewed",
        "rows": [row],
        "repair_rows": [row["id"]],
        "selection": _SELECTION,
        "receipt_policy": {
            "id": "one-per-source-reconciliation",
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
        pytest.fail("source-target reconciliation requires a disposable graph")
    project_uri = resolve_neo4j(auto_tunnel=False).uri
    if uri.rstrip("/") == project_uri.rstrip("/"):
        pytest.fail("source-target reconciliation refuses the project graph URI")
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
        graph_name="source-target-reconciliation",
    )
    graph.query("MATCH (node) DETACH DELETE node")
    yield graph
    graph.query("MATCH (node) DETACH DELETE node")
    graph.close()


def _seed_name(client: GraphClient, name_id: str) -> None:
    client.query(
        """
        CREATE (:StandardName {
          id: $name_id,
          name_stage: 'accepted',
          status: 'draft',
          source_paths: []
        })
        """,
        name_id=name_id,
    )


def _seed_source(
    client: GraphClient,
    source_id: str,
    target_ids: list[str],
    *,
    scalar: str,
) -> dict[str, str]:
    client.query(
        """
        CREATE (source:StandardNameSource {
          id: $source_id,
          status: 'composed',
          produced_sn_id: $scalar
        })
        WITH source
        UNWIND $target_ids AS target_id
        MATCH (target:StandardName {id: target_id})
        CREATE (source)-[:PRODUCED_NAME]->(target)
        """,
        source_id=source_id,
        scalar=scalar,
        target_ids=target_ids,
    )
    return {
        str(row["target_id"]): str(row["binding_id"])
        for row in client.query(
            """
            MATCH (:StandardNameSource {id: $source_id})
                  -[binding:PRODUCED_NAME]->(target:StandardName)
            RETURN target.id AS target_id, elementId(binding) AS binding_id
            ORDER BY target_id
            """,
            source_id=source_id,
        )
    }


def _preview(
    path: Path,
    file_sha256: str,
    payload_sha256: str,
    client: GraphClient,
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
               collect({id: elementId(relationship),
                        type: type(relationship),
                        properties: properties(relationship),
                        other: elementId(other)}) AS relationships
        ORDER BY node_id
        """
    )


def test_generic_registry_admits_closed_source_target_program() -> None:
    assert operator._MUTATION_KINDS == {
        RepairMutationKind.set_properties.value,
        RepairMutationKind.delete.value,
        RepairMutationKind.supersede.value,
        RepairMutationKind.detach.value,
        RepairMutationKind.delete_relationship.value,
    }


@pytest.mark.graph
def test_signed_preview_apply_receipt_and_write_free_replay(
    client: GraphClient, tmp_path: Path
) -> None:
    target_ids = ["signed-survivor", "losing-alpha", "losing-beta"]
    for target_id in [*target_ids, "collateral-name"]:
        _seed_name(client, target_id)
    bindings = _seed_source(
        client,
        "source-under-reconciliation",
        target_ids,
        scalar="losing-alpha",
    )
    _seed_source(
        client,
        "retained-alpha-producer",
        ["losing-alpha"],
        scalar="losing-alpha",
    )
    _seed_source(
        client,
        "retained-beta-producer",
        ["losing-beta"],
        scalar="losing-beta",
    )
    row = _reconciliation_row(
        source_id="source-under-reconciliation",
        survivor_id="signed-survivor",
        target_bindings=bindings,
    )
    path = tmp_path / "source-target-authority.json"
    file_sha256, payload_sha256 = _write_authority(path, row)
    before_preview = _graph_snapshot(client)

    preview = _preview(path, file_sha256, payload_sha256, client)

    assert preview["outcome"] == "would_apply"
    assert preview["counts"] == {"authority_rows": 1, "admitted": 1, "refused": 0}
    assert preview["manifest"]["admitted_row_ids"] == ["source-under-reconciliation"]
    assert preview["manifest"]["rows"][0]["mutation_kinds"] == [
        "delete_relationship",
        "delete_relationship",
        "set_properties",
    ]
    assert _graph_snapshot(client) == before_preview

    applied = _apply(
        path,
        file_sha256,
        payload_sha256,
        client,
        preview["manifest_sha256"],
    )

    assert applied["outcome"] == "applied"
    assert applied["changed"] == 1
    assert applied["mutations"] == 3
    assert applied["receipt_rows"] == 1
    assert applied["persistent_writes"] == 4
    assert client.query(
        """
        MATCH (source:StandardNameSource {id: 'source-under-reconciliation'})
        OPTIONAL MATCH (source)-[:PRODUCED_NAME]->(target:StandardName)
        RETURN source.produced_sn_id AS scalar, collect(target.id) AS targets
        """
    ) == [{"scalar": "signed-survivor", "targets": ["signed-survivor"]}]
    receipt_before = client.query(
        """
        MATCH (change:StandardNameChange {
          operation: $operation,
          manifest_sha256: $manifest_sha256
        })
        RETURN properties(change) AS properties
        """,
        operation=_OPERATION,
        manifest_sha256=preview["manifest_sha256"],
    )
    graph_before_replay = _graph_snapshot(client)

    replay = _apply(
        path,
        file_sha256,
        payload_sha256,
        client,
        preview["manifest_sha256"],
    )

    assert replay["outcome"] == "already_applied"
    assert replay["changed"] == 0
    assert replay["persistent_writes"] == 0
    assert _graph_snapshot(client) == graph_before_replay
    assert (
        client.query(
            """
        MATCH (change:StandardNameChange {
          operation: $operation,
          manifest_sha256: $manifest_sha256
        })
        RETURN properties(change) AS properties
        """,
            operation=_OPERATION,
            manifest_sha256=preview["manifest_sha256"],
        )
        == receipt_before
    )


@pytest.mark.graph
def test_last_producing_source_refusal_is_verbatim(
    client: GraphClient, tmp_path: Path
) -> None:
    for target_id in ["kept-target", "would-be-orphan"]:
        _seed_name(client, target_id)
    bindings = _seed_source(
        client,
        "sole-losing-producer",
        ["kept-target", "would-be-orphan"],
        scalar="would-be-orphan",
    )
    row = _reconciliation_row(
        source_id="sole-losing-producer",
        survivor_id="kept-target",
        target_bindings=bindings,
    )
    path = tmp_path / "last-producer-authority.json"
    file_sha256, payload_sha256 = _write_authority(path, row)
    before = _graph_snapshot(client)

    preview = _preview(path, file_sha256, payload_sha256, client)

    assert preview["outcome"] == "refused"
    assert preview["counts"] == {"authority_rows": 1, "admitted": 0, "refused": 1}
    assert preview["refusals"] == [
        {
            "row_id": "sole-losing-producer",
            "reason": "target would lose its last producing source",
        }
    ]
    assert _graph_snapshot(client) == before
    assert client.query(
        "MATCH (change:StandardNameChange) RETURN count(change) AS count"
    ) == [{"count": 0}]


@pytest.mark.graph
def test_preview_refuses_an_omitted_live_target(
    client: GraphClient, tmp_path: Path
) -> None:
    for target_id in ["declared-survivor", "declared-loser", "omitted-target"]:
        _seed_name(client, target_id)
    bindings = _seed_source(
        client,
        "incompletely-signed-source",
        ["declared-survivor", "declared-loser", "omitted-target"],
        scalar="declared-loser",
    )
    _seed_source(
        client,
        "retained-declared-producer",
        ["declared-loser"],
        scalar="declared-loser",
    )
    row = _reconciliation_row(
        source_id="incompletely-signed-source",
        survivor_id="declared-survivor",
        target_bindings={
            target_id: binding_id
            for target_id, binding_id in bindings.items()
            if target_id != "omitted-target"
        },
    )
    path = tmp_path / "incomplete-target-authority.json"
    file_sha256, payload_sha256 = _write_authority(path, row)
    before = _graph_snapshot(client)

    preview = _preview(path, file_sha256, payload_sha256, client)

    assert preview["outcome"] == "refused"
    assert preview["refusals"] == [
        {
            "row_id": "incompletely-signed-source",
            "reason": (
                "signed source-target closure does not match complete live targets"
            ),
        }
    ]
    assert _graph_snapshot(client) == before
