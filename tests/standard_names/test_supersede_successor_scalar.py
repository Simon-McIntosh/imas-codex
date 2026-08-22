"""Contract for signed supersedes into a distinct canonical identity."""

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
from imas_codex.graph.profiles import resolve_neo4j
from imas_codex.standard_names.signed_manifest import (
    apply_signed_manifest,
    signed_payload_sha256,
)

_PREDECESSOR = "area_of_flux_surface"
_SUCCESSOR = "poloidal_plane_cross_sectional_area_of_flux_surface"
_ROW_ID = f"{_PREDECESSOR}=>{_SUCCESSOR}"
_OPERATION = "supersede_legacy_spelling"
_SELECTION = {
    "id": "artifact-rows",
    "mode": "exact_complete_signed_cohort",
    "predicate": "artifact-rows",
}


@pytest.fixture(scope="module")
def disposable_neo4j() -> Iterator[tuple[str, str]]:
    uri = os.environ.get("IMAS_CODEX_TEST_NEO4J_URI")
    if not uri:
        pytest.skip("IMAS_CODEX_TEST_NEO4J_URI is not configured")
    if os.environ.get("IMAS_CODEX_TEST_NEO4J_EPHEMERAL") != "1":
        pytest.fail("signed supersede tests require a disposable graph")
    project_uri = resolve_neo4j(auto_tunnel=False).uri
    if uri.rstrip("/") == project_uri.rstrip("/"):
        pytest.fail("signed supersede tests refuse the project graph URI")
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
        graph_name="signed-supersede-successor",
    )
    graph.query("MATCH (node) DETACH DELETE node")
    yield graph
    graph.query("MATCH (node) DETACH DELETE node")
    graph.close()


def _write_authority(path: Path) -> tuple[str, str]:
    authority: dict[str, Any] = {
        "schema": "imas-codex.repair-authority.v1",
        "operation_id": "flux-surface-umbrella-supersede",
        "authority_mode": "external_reviewed",
        "rows": [
            {
                "id": _ROW_ID,
                "identity": {
                    "id": _ROW_ID,
                    "kind": "standard_name",
                    "target_id": _PREDECESSOR,
                },
                "participants": [
                    {
                        "id": _PREDECESSOR,
                        "kind": "node",
                        "graph_label": "StandardName",
                    },
                    {
                        "id": _SUCCESSOR,
                        "kind": "node",
                        "graph_label": "StandardName",
                    },
                ],
                "selection": _SELECTION,
                "mutations": [
                    {
                        "id": f"{_PREDECESSOR}:supersede",
                        "order": 0,
                        "kind": "supersede",
                        "participant_id": _PREDECESSOR,
                        "arguments": {"successor_id": _SUCCESSOR},
                    }
                ],
                "guards": [
                    {
                        "id": f"{_ROW_ID}:structural-legitimacy",
                        "kind": "semantic_authority",
                        "implementation": "structural-legitimacy",
                        "participant_ids": [_PREDECESSOR],
                    },
                    {
                        "id": f"{_ROW_ID}:out-of-allowlist-immutability",
                        "kind": "collateral_immutability",
                        "implementation": "out-of-allowlist-immutability",
                        "participant_ids": [_PREDECESSOR, _SUCCESSOR],
                    },
                ],
                "orphan_policy": "refuse",
            }
        ],
        "repair_rows": [_ROW_ID],
        "selection": _SELECTION,
        "receipt_policy": {
            "id": "one-per-superseded-identity",
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
    raw = json.dumps(authority, sort_keys=True, indent=2).encode()
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest(), payload_sha256


def _collateral_bytes(client: GraphClient) -> bytes:
    rows = client.query(
        """
        MATCH (source:StandardNameSource {id: 'collateral-source'})
              -[binding:PRODUCED_NAME]->
              (target:StandardName {id: 'collateral-name'})
        RETURN properties(source) AS source,
               properties(binding) AS binding,
               properties(target) AS target,
               elementId(source) AS source_element_id,
               elementId(binding) AS binding_element_id,
               elementId(target) AS target_element_id
        """
    )
    return json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()


@pytest.mark.graph
def test_signed_supersede_writes_successor_receipt_and_replays_without_writes(
    client: GraphClient, tmp_path: Path
) -> None:
    client.query(
        """
        CREATE (:StandardName {
          id: $predecessor,
          name_stage: 'accepted',
          status: 'draft',
          validation_status: 'valid',
          source_paths: []
        })
        CREATE (:StandardName {
          id: $successor,
          name_stage: 'accepted',
          status: 'draft',
          validation_status: 'valid',
          source_paths: ['dd:profiles_1d/area']
        })
        CREATE (collateral:StandardName {
          id: 'collateral-name',
          name_stage: 'accepted',
          status: 'draft',
          validation_status: 'valid',
          source_paths: ['dd:collateral/path']
        })
        CREATE (:StandardNameSource {
          id: 'collateral-source',
          source_id: 'collateral/path',
          source_type: 'dd',
          status: 'composed',
          produced_sn_id: 'collateral-name'
        })-[:PRODUCED_NAME {mirror: 'unchanged'}]->(collateral)
        """,
        predecessor=_PREDECESSOR,
        successor=_SUCCESSOR,
    )
    collateral_before = _collateral_bytes(client)
    authority_path = tmp_path / "supersede-authority.json"
    file_sha256, payload_sha256 = _write_authority(authority_path)

    preview = apply_signed_manifest(
        authority_path,
        authority_file_sha256=file_sha256,
        authority_payload_sha256=payload_sha256,
        reason="replace a taxonomy umbrella with its canonical quantity",
        gc=client,
    )
    assert preview["outcome"] == "would_apply"
    assert preview["counts"] == {
        "authority_rows": 1,
        "admitted": 1,
        "refused": 0,
    }

    applied = apply_signed_manifest(
        authority_path,
        authority_file_sha256=file_sha256,
        authority_payload_sha256=payload_sha256,
        reason="replace a taxonomy umbrella with its canonical quantity",
        apply=True,
        manifest_sha256=preview["manifest_sha256"],
        gc=client,
    )
    assert applied["outcome"] == "applied"
    assert applied["changed"] == 1
    assert applied["receipt_rows"] == 1
    assert client.query(
        """
        MATCH (predecessor:StandardName {id: $predecessor})
        MATCH (receipt:StandardNameChange {
          operation: $operation,
          manifest_sha256: $manifest_sha256
        })
        RETURN predecessor.name_stage AS name_stage,
               predecessor.status AS status,
               predecessor.superseded_by AS superseded_by,
               receipt.from_name AS from_name,
               receipt.to_name AS to_name
        """,
        predecessor=_PREDECESSOR,
        operation=_OPERATION,
        manifest_sha256=preview["manifest_sha256"],
    ) == [
        {
            "name_stage": "superseded",
            "status": "superseded",
            "superseded_by": _SUCCESSOR,
            "from_name": _PREDECESSOR,
            "to_name": _SUCCESSOR,
        }
    ]
    assert _collateral_bytes(client) == collateral_before

    replay = apply_signed_manifest(
        authority_path,
        authority_file_sha256=file_sha256,
        authority_payload_sha256=payload_sha256,
        reason="replace a taxonomy umbrella with its canonical quantity",
        apply=True,
        manifest_sha256=preview["manifest_sha256"],
        gc=client,
    )
    assert replay["outcome"] == "already_applied"
    assert replay["changed"] == 0
    assert replay["persistent_writes"] == 0
    assert replay["receipt_rows"] == 1
    assert _collateral_bytes(client) == collateral_before
