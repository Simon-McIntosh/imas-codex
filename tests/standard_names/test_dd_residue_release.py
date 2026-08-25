"""Disposable-graph contract for the legacy DD lifecycle release."""

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
from imas_codex.standard_names import signed_manifest as operator
from imas_codex.standard_names.repair_authority import build_repair_authority
from imas_codex.standard_names.signed_manifest import apply_signed_manifest

_SOURCE_IDS = (
    "dd:ntms/time_slice/mode",
    "dd:summary/pedestal_fits",
    "dd:waves/coherent_wave",
)
_OPERATION = "release_legacy_dd_source_lifecycle"
_REASON = "release the exact historical DD sources for ordinary recomposition"
_SELECTION = {
    "id": "artifact-rows",
    "mode": "exact_complete_signed_cohort",
    "predicate": "artifact-rows",
}
_RELEASE_PROPERTIES = {
    "status": "extracted",
    "attempt_count": 0,
    "claimed_at": None,
    "claim_token": None,
    "produced_sn_id": None,
    "composed_at": None,
}


@pytest.fixture(scope="module")
def disposable_neo4j() -> Iterator[tuple[str, str]]:
    uri = os.environ.get("IMAS_CODEX_TEST_NEO4J_URI")
    if not uri:
        pytest.skip("IMAS_CODEX_TEST_NEO4J_URI is not configured")
    if os.environ.get("IMAS_CODEX_TEST_NEO4J_EPHEMERAL") != "1":
        pytest.fail("legacy DD release requires a disposable graph")
    project_uri = resolve_neo4j(auto_tunnel=False).uri
    if uri.rstrip("/") == project_uri.rstrip("/"):
        pytest.fail("legacy DD release refuses the project graph URI")
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
        graph_name="legacy-dd-source-release",
    )
    graph.query("MATCH (node) DETACH DELETE node")
    yield graph
    graph.query("MATCH (node) DETACH DELETE node")
    graph.close()


def _seed_sources(client: GraphClient) -> None:
    client.query(
        """
        UNWIND $source_ids AS source_id
        CREATE (:StandardNameSource {
          id: source_id,
          source_id: substring(source_id, 3),
          source_type: 'dd',
          status: 'composed',
          attempt_count: 4,
          produced_sn_id: 'terminal-history',
          composed_at: datetime('2026-08-20T00:00:00Z')
        })
        """,
        source_ids=list(_SOURCE_IDS),
    )


def _seed_binding(
    client: GraphClient, source_id: str, target_id: str, *, live: bool
) -> None:
    client.query(
        """
        CREATE (target:StandardName {
          id: $target_id,
          name_stage: $name_stage,
          status: $status
        })
        WITH target
        MATCH (source:StandardNameSource {id: $source_id})
        CREATE (source)-[:PRODUCED_NAME]->(target)
        """,
        source_id=source_id,
        target_id=target_id,
        name_stage="accepted" if live else "superseded",
        status="draft" if live else "superseded",
    )


def _authority_row(client: GraphClient, source_id: str) -> dict[str, Any]:
    bindings = client.query(
        """
        MATCH (:StandardNameSource {id: $source_id})
              -[binding:PRODUCED_NAME]->(target:StandardName)
        RETURN elementId(binding) AS binding_id, target.id AS target_id
        ORDER BY target.id
        """,
        source_id=source_id,
    )
    participants = [
        {"id": source_id, "kind": "node", "graph_label": "StandardNameSource"}
    ]
    mutations: list[dict[str, Any]] = []
    for order, binding in enumerate(bindings):
        participants.extend(
            [
                {
                    "id": str(binding["target_id"]),
                    "kind": "node",
                    "graph_label": "StandardName",
                },
                {
                    "id": str(binding["binding_id"]),
                    "kind": "relationship",
                    "graph_label": "PRODUCED_NAME",
                },
            ]
        )
        mutations.append(
            {
                "id": f"release-binding:{order}",
                "order": order,
                "kind": "delete_relationship",
                "participant_id": str(binding["binding_id"]),
            }
        )
    mutations.append(
        {
            "id": "release-lifecycle",
            "order": len(mutations),
            "kind": "set_properties",
            "participant_id": source_id,
            "arguments": {"properties": _RELEASE_PROPERTIES},
        }
    )
    return {
        "id": source_id,
        "identity": {"id": source_id, "kind": "source", "source_id": source_id},
        "participants": participants,
        "selection": _SELECTION,
        "mutations": mutations,
        "guards": [
            {
                "id": "last-producing-source",
                "kind": "semantic_authority",
                "implementation": "last-producing-source",
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


def _write_authority(client: GraphClient, path: Path) -> tuple[str, str]:
    built = build_repair_authority(
        {
            "operation_id": _OPERATION,
            "authority_mode": "external_reviewed",
            "rows": [_authority_row(client, source_id) for source_id in _SOURCE_IDS],
            "selection": _SELECTION,
            "receipt_policy": {
                "id": "one-per-released-source",
                "operation": _OPERATION,
                "cardinality": "per_target",
                "expected_count": "admitted_rows",
                "link_participant_kind": "source",
                "replay_projection": ["manifest_sha256", "row_id"],
            },
            "orphan_policy": "refuse",
        }
    )
    path.write_bytes(built.content)
    return built.file_sha256, built.payload_sha256


def _preview(client: GraphClient, path: Path) -> dict[str, Any]:
    file_sha256, payload_sha256 = _write_authority(client, path)
    return apply_signed_manifest(
        path,
        authority_file_sha256=file_sha256,
        authority_payload_sha256=payload_sha256,
        reason=_REASON,
        gc=client,
    )


def _apply(client: GraphClient, path: Path, preview: dict[str, Any]) -> dict[str, Any]:
    raw = path.read_bytes()
    authority = json.loads(raw)
    return apply_signed_manifest(
        path,
        authority_file_sha256=hashlib.sha256(raw).hexdigest(),
        authority_payload_sha256=operator.signed_payload_sha256(authority),
        reason=_REASON,
        apply=True,
        manifest_sha256=preview["manifest_sha256"],
        gc=client,
    )


@pytest.mark.graph
def test_exact_residues_release_with_receipts_and_write_free_replay(
    client: GraphClient, tmp_path: Path
) -> None:
    _seed_sources(client)
    path = tmp_path / "legacy-dd-release.json"
    preview = _preview(client, path)

    assert preview["counts"] == {"authority_rows": 3, "admitted": 3, "refused": 0}
    assert preview["manifest"]["admitted_row_ids"] == sorted(_SOURCE_IDS)

    applied = _apply(client, path, preview)

    assert applied["outcome"] == "applied"
    assert applied["changed"] == 3
    assert applied["receipt_rows"] == 3
    rows = client.query(
        """
        MATCH (source:StandardNameSource)
        WHERE source.id IN $source_ids
        RETURN source.id AS source_id, source.status AS status,
               source.attempt_count AS attempt_count,
               source.produced_sn_id AS produced_sn_id,
               source.composed_at AS composed_at
        ORDER BY source_id
        """,
        source_ids=list(_SOURCE_IDS),
    )
    assert rows == [
        {
            "source_id": source_id,
            "status": "extracted",
            "attempt_count": 0,
            "produced_sn_id": None,
            "composed_at": None,
        }
        for source_id in sorted(_SOURCE_IDS)
    ]
    receipts = client.query(
        """
        MATCH (change:StandardNameChange {
          operation: $operation,
          manifest_sha256: $manifest_sha256
        })
        RETURN change.row_id AS row_id
        ORDER BY row_id
        """,
        operation=_OPERATION,
        manifest_sha256=preview["manifest_sha256"],
    )
    assert receipts == [{"row_id": source_id} for source_id in sorted(_SOURCE_IDS)]
    before_replay = client.query(
        "MATCH (node) RETURN elementId(node) AS id, properties(node) AS properties ORDER BY id"
    )

    replay = _apply(client, path, preview)

    assert replay["outcome"] == "already_applied"
    assert replay["changed"] == 0
    assert replay["persistent_writes"] == 0
    assert (
        client.query(
            "MATCH (node) RETURN elementId(node) AS id, properties(node) AS properties ORDER BY id"
        )
        == before_replay
    )


@pytest.mark.graph
def test_live_target_is_refused_verbatim(client: GraphClient, tmp_path: Path) -> None:
    _seed_sources(client)
    _seed_binding(client, _SOURCE_IDS[0], "still-live", live=True)

    preview = _preview(client, tmp_path / "live-target-refusal.json")

    assert preview["counts"] == {"authority_rows": 3, "admitted": 2, "refused": 1}
    assert preview["refusals"] == [
        {"row_id": _SOURCE_IDS[0], "reason": "source still has a live target"}
    ]


@pytest.mark.graph
def test_last_producing_source_is_refused_verbatim(
    client: GraphClient, tmp_path: Path
) -> None:
    _seed_sources(client)
    _seed_binding(client, _SOURCE_IDS[1], "terminal-history", live=False)

    preview = _preview(client, tmp_path / "last-producer-refusal.json")

    assert preview["counts"] == {"authority_rows": 3, "admitted": 2, "refused": 1}
    assert preview["refusals"] == [
        {
            "row_id": _SOURCE_IDS[1],
            "reason": "target would lose its last producing source",
        }
    ]
