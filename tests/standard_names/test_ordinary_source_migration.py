"""Disposable-graph contract for signed ordinary-source migration."""

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
from imas_codex.standard_names.signed_manifest import (
    apply_signed_manifest,
    signed_payload_sha256,
)

_SELECTION = {
    "id": "artifact-rows",
    "mode": "exact_complete_signed_cohort",
    "predicate": "artifact-rows",
}
_OPERATION = "migrate_ordinary_standard_name_source"
_REASON = "retarget the exact independently adjudicated ordinary source"
_OLD_NAME = "toroidal_angle_of_measurement_position"
_ROWS = (
    (
        "spectrometer_visible/channel/active_spatial_resolution/centre/phi",
        "toroidal_coordinate_of_active_spatial_resolution_zone",
    ),
    (
        "spectrometer_visible/channel/detector/centre/phi",
        "toroidal_coordinate_of_detector",
    ),
    (
        "spectrometer_visible/channel/polarizer/centre/phi",
        "toroidal_coordinate_of_polarizer",
    ),
)


def _node(node_id: str, label: str) -> dict[str, Any]:
    return {"id": node_id, "kind": "node", "graph_label": label}


def _row(source_id: str, target_id: str, binding_id: str) -> dict[str, Any]:
    return {
        "id": source_id,
        "identity": {
            "id": source_id,
            "kind": "source",
            "source_id": source_id,
            "target_id": target_id,
        },
        "participants": [
            _node(source_id, "StandardNameSource"),
            _node(_OLD_NAME, "StandardName"),
            _node(target_id, "StandardName"),
            {
                "id": binding_id,
                "kind": "relationship",
                "graph_label": "PRODUCED_NAME",
            },
        ],
        "selection": _SELECTION,
        "mutations": [
            {
                "id": "remove-incumbent-binding",
                "order": 1,
                "kind": "delete_relationship",
                "participant_id": binding_id,
            },
            {
                "id": "attach-authoritative-target",
                "order": 2,
                "kind": "add_relationship",
                "participant_id": target_id,
                "arguments": {
                    "relationship_type": "PRODUCED_NAME",
                    "start_id": source_id,
                    "end_id": target_id,
                },
            },
            {
                "id": "retarget-source-scalar",
                "order": 3,
                "kind": "set_properties",
                "participant_id": source_id,
                "arguments": {"properties": {"produced_sn_id": target_id}},
            },
        ],
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


def _write_authority(path: Path, rows: list[dict[str, Any]]) -> tuple[str, str]:
    authority: dict[str, Any] = {
        "schema": "imas-codex.repair-authority.v1",
        "operation_id": "ordinary-source-migration",
        "authority_mode": "external_reviewed",
        "rows": rows,
        "repair_rows": [row["id"] for row in rows],
        "selection": _SELECTION,
        "receipt_policy": {
            "id": "one-per-ordinary-source-migration",
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
        pytest.fail("ordinary-source migration requires a disposable graph")
    project_uri = resolve_neo4j(auto_tunnel=False).uri
    if uri.rstrip("/") == project_uri.rstrip("/"):
        pytest.fail("ordinary-source migration refuses the project graph URI")
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
        graph_name="ordinary-source-migration",
    )
    graph.query("MATCH (node) DETACH DELETE node")
    yield graph
    graph.query("MATCH (node) DETACH DELETE node")
    graph.close()


def _seed(client: GraphClient, path: str, target_id: str) -> str:
    source_id = f"dd:{path}"
    retained_path = "spectrometer_visible/channel/reference/centre/phi"
    client.query(
        """
        CREATE (old:StandardName {
          id: $old_name, name_stage: 'accepted', validation_status: 'valid',
          status: 'draft', source_paths: ['dd:' + $path, 'dd:' + $retained_path]
        })
        CREATE (new:StandardName {
          id: $target_id, name_stage: 'accepted', validation_status: 'valid',
          status: 'draft', source_paths: []
        })
        CREATE (dd:IMASNode {id: $path})
        CREATE (source:StandardNameSource {
          id: $source_id, source_type: 'dd', source_id: $path,
          status: 'attached', produced_sn_id: $old_name
        })
        CREATE (source)-[:FROM_DD_PATH]->(dd)
        CREATE (source)-[:PRODUCED_NAME]->(old)
        CREATE (dd)-[:HAS_STANDARD_NAME]->(old)
        CREATE (retained_dd:IMASNode {id: $retained_path})
        CREATE (retained:StandardNameSource {
          id: 'dd:' + $retained_path, source_type: 'dd', source_id: $retained_path,
          status: 'attached', produced_sn_id: $old_name
        })
        CREATE (retained)-[:FROM_DD_PATH]->(retained_dd)
        CREATE (retained)-[:PRODUCED_NAME]->(old)
        CREATE (retained_dd)-[:HAS_STANDARD_NAME]->(old)
        """,
        old_name=_OLD_NAME,
        target_id=target_id,
        path=path,
        source_id=source_id,
        retained_path=retained_path,
    )
    return str(
        client.query(
            """
            MATCH (:StandardNameSource {id: $source_id})
                  -[binding:PRODUCED_NAME]->(:StandardName {id: $old_name})
            RETURN elementId(binding) AS binding_id
            """,
            source_id=source_id,
            old_name=_OLD_NAME,
        )[0]["binding_id"]
    )


def _preview(
    path: Path, file_digest: str, payload_digest: str, client: GraphClient
) -> dict[str, Any]:
    return apply_signed_manifest(
        path,
        authority_file_sha256=file_digest,
        authority_payload_sha256=payload_digest,
        reason=_REASON,
        gc=client,
    )


def _apply(
    path: Path,
    file_digest: str,
    payload_digest: str,
    client: GraphClient,
    manifest_digest: str,
) -> dict[str, Any]:
    return apply_signed_manifest(
        path,
        authority_file_sha256=file_digest,
        authority_payload_sha256=payload_digest,
        reason=_REASON,
        apply=True,
        manifest_sha256=manifest_digest,
        gc=client,
    )


def test_three_spectrometer_rows_validate_through_loader(tmp_path: Path) -> None:
    rows = [
        _row(f"dd:{path}", target, f"binding-{index}")
        for index, (path, target) in enumerate(_ROWS)
    ]
    authority = tmp_path / "authority.json"
    file_digest, payload_digest = _write_authority(authority, rows)

    loaded = operator._load_authority(
        authority,
        expected_file_sha256=file_digest,
        expected_payload_sha256=payload_digest,
    )

    assert len(loaded.rows) == 3


@pytest.mark.graph
def test_preview_apply_receipt_and_write_free_replay(
    client: GraphClient, tmp_path: Path
) -> None:
    path, target_id = _ROWS[1]
    source_id = f"dd:{path}"
    binding_id = _seed(client, path, target_id)
    authority = tmp_path / "authority.json"
    file_digest, payload_digest = _write_authority(
        authority, [_row(source_id, target_id, binding_id)]
    )
    preview = _preview(authority, file_digest, payload_digest, client)
    assert preview["outcome"] == "would_apply"
    assert preview["counts"] == {"authority_rows": 1, "admitted": 1, "refused": 0}

    applied = _apply(
        authority,
        file_digest,
        payload_digest,
        client,
        preview["manifest_sha256"],
    )
    assert applied["outcome"] == "applied"
    assert applied["changed"] == 1
    assert applied["mutations"] == 3
    assert applied["receipt_rows"] == 1
    state = client.query(
        """
        MATCH (source:StandardNameSource {id: $source_id})
        OPTIONAL MATCH (source)-[:PRODUCED_NAME]->(bound:StandardName)
        OPTIONAL MATCH (source)-[:FROM_DD_PATH]->(dd:IMASNode)
        OPTIONAL MATCH (dd)-[:HAS_STANDARD_NAME]->(projected:StandardName)
        MATCH (old:StandardName {id: $old_name}), (new:StandardName {id: $target_id})
        RETURN source.produced_sn_id AS scalar,
               collect(DISTINCT bound.id) AS bindings,
               collect(DISTINCT projected.id) AS projections,
               old.source_paths AS old_paths, new.source_paths AS new_paths
        """,
        source_id=source_id,
        old_name=_OLD_NAME,
        target_id=target_id,
    )[0]
    assert state["scalar"] == target_id
    assert state["bindings"] == [target_id]
    assert state["projections"] == [target_id]
    assert f"dd:{path}" not in state["old_paths"]
    assert f"dd:{path}" in state["new_paths"]

    before = client.query(
        "MATCH (node) OPTIONAL MATCH (node)-[r]->() RETURN count(node) AS nodes, count(r) AS relationships"
    )[0]
    replay = _apply(
        authority,
        file_digest,
        payload_digest,
        client,
        preview["manifest_sha256"],
    )
    after = client.query(
        "MATCH (node) OPTIONAL MATCH (node)-[r]->() RETURN count(node) AS nodes, count(r) AS relationships"
    )[0]
    assert replay["outcome"] == "already_applied"
    assert replay["changed"] == 0
    assert replay["persistent_writes"] == 0
    assert after == before


@pytest.mark.graph
def test_last_producer_refusal_is_verbatim(client: GraphClient, tmp_path: Path) -> None:
    path, target_id = _ROWS[0]
    source_id = f"dd:{path}"
    binding_id = _seed(client, path, target_id)
    client.query(
        """
        MATCH (retained:StandardNameSource)-[binding:PRODUCED_NAME]->()
        WHERE retained.id <> $source_id
        DELETE binding
        """,
        source_id=source_id,
    )
    authority = tmp_path / "authority.json"
    file_digest, payload_digest = _write_authority(
        authority, [_row(source_id, target_id, binding_id)]
    )

    preview = _preview(authority, file_digest, payload_digest, client)

    assert preview["refusals"] == [
        {"row_id": source_id, "reason": "target would lose its last producing source"}
    ]


@pytest.mark.parametrize(
    ("drift", "reason"),
    [
        ("claim", "ordinary source has an active claim"),
        (
            "projection",
            "signed ordinary source closure does not match exact incumbent binding and projection",
        ),
    ],
)
@pytest.mark.graph
def test_claim_and_projection_closure_refusals(
    client: GraphClient, tmp_path: Path, drift: str, reason: str
) -> None:
    path, target_id = _ROWS[2]
    source_id = f"dd:{path}"
    binding_id = _seed(client, path, target_id)
    if drift == "claim":
        client.query(
            "MATCH (source:StandardNameSource {id: $source_id}) SET source.claim_token = 'busy'",
            source_id=source_id,
        )
    else:
        client.query(
            """
            MATCH (:StandardNameSource {id: $source_id})-[:FROM_DD_PATH]->
                  (dd:IMASNode)-[projection:HAS_STANDARD_NAME]->()
            DELETE projection
            """,
            source_id=source_id,
        )
    authority = tmp_path / "authority.json"
    file_digest, payload_digest = _write_authority(
        authority, [_row(source_id, target_id, binding_id)]
    )

    preview = _preview(authority, file_digest, payload_digest, client)

    assert preview["refusals"] == [{"row_id": source_id, "reason": reason}]


@pytest.mark.graph
def test_attachment_guard_refusal_is_verbatim(
    client: GraphClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from imas_codex.standard_names.attachment_audit import (
        AttachmentPairingGuardResult,
        AttachmentVerdict,
    )

    path, target_id = _ROWS[1]
    source_id = f"dd:{path}"
    binding_id = _seed(client, path, target_id)
    rejection = AttachmentVerdict(
        source_node_id=source_id,
        dd_path=path,
        sn_id=target_id,
        name_stage="accepted",
        reason="semantic mismatch",
    )
    monkeypatch.setattr(
        "imas_codex.standard_names.attachment_audit.guard_source_pairings",
        lambda *_args, **_kwargs: AttachmentPairingGuardResult((), (rejection,)),
    )
    authority = tmp_path / "authority.json"
    file_digest, payload_digest = _write_authority(
        authority, [_row(source_id, target_id, binding_id)]
    )

    preview = _preview(authority, file_digest, payload_digest, client)

    assert preview["refusals"] == [
        {
            "row_id": source_id,
            "reason": f"source migration attachment rejected: {source_id}: semantic mismatch",
        }
    ]
