"""Contract: the signed supersede mutation mints a receipt-stamped successor edge."""

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
from imas_codex.standard_names import signed_manifest
from imas_codex.standard_names.signed_manifest import (
    SignedManifestConflict,
    _change_id,
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
        pytest.skip("signed supersede edge tests require a disposable graph")
    project_uri = resolve_neo4j(auto_tunnel=False).uri
    if uri.rstrip("/") == project_uri.rstrip("/"):
        pytest.fail("signed supersede edge tests refuse the project graph URI")
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
        graph_name="signed-supersede-edge",
    )
    graph.query("MATCH (node) DETACH DELETE node")
    yield graph
    graph.query("MATCH (node) DETACH DELETE node")
    graph.close()


def _seed(client: GraphClient) -> None:
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
        """,
        predecessor=_PREDECESSOR,
        successor=_SUCCESSOR,
    )


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


def _apply(
    client: GraphClient, authority_path: Path, file_sha256: str, payload_sha256: str
) -> dict[str, Any]:
    preview = apply_signed_manifest(
        authority_path,
        authority_file_sha256=file_sha256,
        authority_payload_sha256=payload_sha256,
        reason="replace a taxonomy umbrella with its canonical quantity",
        gc=client,
    )
    assert preview["outcome"] == "would_apply"
    return apply_signed_manifest(
        authority_path,
        authority_file_sha256=file_sha256,
        authority_payload_sha256=payload_sha256,
        reason="replace a taxonomy umbrella with its canonical quantity",
        apply=True,
        manifest_sha256=preview["manifest_sha256"],
        gc=client,
    )


@pytest.mark.graph
def test_signed_supersede_mints_receipt_stamped_successor_edge(
    client: GraphClient, tmp_path: Path
) -> None:
    _seed(client)
    authority_path = tmp_path / "supersede-authority.json"
    file_sha256, payload_sha256 = _write_authority(authority_path)

    preview = apply_signed_manifest(
        authority_path,
        authority_file_sha256=file_sha256,
        authority_payload_sha256=payload_sha256,
        reason="replace a taxonomy umbrella with its canonical quantity",
        gc=client,
    )
    manifest_sha256 = preview["manifest_sha256"]
    applied = apply_signed_manifest(
        authority_path,
        authority_file_sha256=file_sha256,
        authority_payload_sha256=payload_sha256,
        reason="replace a taxonomy umbrella with its canonical quantity",
        apply=True,
        manifest_sha256=manifest_sha256,
        gc=client,
    )
    assert applied["outcome"] == "applied"

    rows = client.query(
        """
        MATCH (predecessor:StandardName {id: $predecessor})
              -[edge:HAS_SUCCESSOR]->
              (successor:StandardName {id: $successor})
        OPTIONAL MATCH (receipt:StandardNameChange {id: edge.receipt_id})
        RETURN edge.receipt_id AS receipt_id,
               edge.signed_at IS NOT NULL AS stamped,
               receipt.operation AS receipt_operation
        """,
        predecessor=_PREDECESSOR,
        successor=_SUCCESSOR,
    )
    expected_receipt_id = _change_id(manifest_sha256, _ROW_ID)
    assert rows == [
        {
            "receipt_id": expected_receipt_id,
            "stamped": True,
            "receipt_operation": _OPERATION,
        }
    ]

    # The signed edge is distinguishable from a derived one, which carries no
    # receipt stamp.
    unstamped = client.query(
        """
        MATCH (:StandardName)-[edge:HAS_SUCCESSOR]->(:StandardName)
        WHERE edge.receipt_id IS NULL
        RETURN count(edge) AS count
        """
    )
    assert int(unstamped[0]["count"]) == 0


@pytest.mark.graph
def test_supersede_postcondition_refuses_missing_successor_edge(
    client: GraphClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed(client)
    authority_path = tmp_path / "supersede-authority.json"
    file_sha256, payload_sha256 = _write_authority(authority_path)
    monkeypatch.setattr(
        signed_manifest, "_supersede_successor_edge_write", lambda _successor: ""
    )

    preview = apply_signed_manifest(
        authority_path,
        authority_file_sha256=file_sha256,
        authority_payload_sha256=payload_sha256,
        reason="replace a taxonomy umbrella with its canonical quantity",
        gc=client,
    )
    with pytest.raises(SignedManifestConflict) as excinfo:
        apply_signed_manifest(
            authority_path,
            authority_file_sha256=file_sha256,
            authority_payload_sha256=payload_sha256,
            reason="replace a taxonomy umbrella with its canonical quantity",
            apply=True,
            manifest_sha256=preview["manifest_sha256"],
            gc=client,
        )
    assert "successor edge" in str(excinfo.value)
