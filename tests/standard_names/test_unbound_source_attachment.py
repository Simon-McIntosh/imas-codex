"""Disposable-graph contract for signed unbound ordinary-source attachment."""

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
    SignedManifestConflict,
    apply_signed_manifest,
    signed_payload_sha256,
)

_PATH = "spectrometer_visible/channel/detector/centre/phi"
_SOURCE_ID = f"dd:{_PATH}"
_TARGET_ID = "toroidal_coordinate_of_detector"
_OPERATION = "attach_unbound_standard_name_source"
_REASON = "attach the exact independently adjudicated unbound source"
_SELECTION = {
    "id": "artifact-rows",
    "mode": "exact_complete_signed_cohort",
    "predicate": "artifact-rows",
}


def _row() -> dict[str, Any]:
    return {
        "id": _SOURCE_ID,
        "identity": {
            "id": _SOURCE_ID,
            "kind": "source",
            "source_id": _SOURCE_ID,
            "target_id": _TARGET_ID,
        },
        "participants": [
            {
                "id": _SOURCE_ID,
                "kind": "node",
                "graph_label": "StandardNameSource",
            },
            {
                "id": _TARGET_ID,
                "kind": "node",
                "graph_label": "StandardName",
            },
        ],
        "selection": _SELECTION,
        "mutations": [
            {
                "id": "attach-authoritative-target",
                "order": 1,
                "kind": "add_relationship",
                "participant_id": _TARGET_ID,
                "arguments": {
                    "relationship_type": "PRODUCED_NAME",
                    "start_id": _SOURCE_ID,
                    "end_id": _TARGET_ID,
                },
            },
            {
                "id": "advance-source-lifecycle",
                "order": 2,
                "kind": "set_properties",
                "participant_id": _SOURCE_ID,
                "arguments": {
                    "properties": {
                        "status": "attached",
                        "produced_sn_id": _TARGET_ID,
                        "claimed_at": None,
                        "claim_token": None,
                        "last_error": None,
                    }
                },
            },
        ],
        "guards": [
            {
                "id": "out-of-allowlist-immutability",
                "kind": "collateral_immutability",
                "implementation": "out-of-allowlist-immutability",
                "participant_ids": [],
            }
        ],
        "orphan_policy": "refuse",
    }


def _write_authority(path: Path) -> tuple[str, str]:
    authority: dict[str, Any] = {
        "schema": "imas-codex.repair-authority.v1",
        "operation_id": "unbound-ordinary-source-attachment",
        "authority_mode": "external_reviewed",
        "rows": [_row()],
        "repair_rows": [_SOURCE_ID],
        "selection": _SELECTION,
        "receipt_policy": {
            "id": "one-per-unbound-source-attachment",
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
        pytest.fail("unbound-source attachment requires a disposable graph")
    project_uri = resolve_neo4j(auto_tunnel=False).uri
    if uri.rstrip("/") == project_uri.rstrip("/"):
        pytest.fail("unbound-source attachment refuses the project graph URI")
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
        graph_name="unbound-source-attachment",
    )
    graph.query("MATCH (node) DETACH DELETE node")
    yield graph
    graph.query("MATCH (node) DETACH DELETE node")
    graph.close()


def _seed(client: GraphClient, *, dd_unit: str = "rad") -> None:
    client.query(
        """
        CREATE (target:StandardName {
          id: $target_id, name_stage: 'accepted', validation_status: 'valid',
          status: 'draft', source_paths: ['dd:retained/path'], unit: 'rad'
        })
        CREATE (dd:IMASNode {id: $path, unit: $dd_unit})
        CREATE (source:StandardNameSource {
          id: $source_id, source_type: 'dd', source_id: $path,
          status: 'extracted', produced_sn_id: null, last_error: 'prior miss'
        })
        CREATE (source)-[:FROM_DD_PATH]->(dd)
        CREATE (:StandardName {
          id: 'collateral_name', name_stage: 'accepted',
          validation_status: 'valid', status: 'draft', source_paths: []
        })
        CREATE (:StandardNameSource {
          id: 'dd:collateral/path', source_type: 'dd',
          source_id: 'collateral/path', status: 'stale'
        })
        """,
        target_id=_TARGET_ID,
        path=_PATH,
        source_id=_SOURCE_ID,
        dd_unit=dd_unit,
    )


def _preview(
    client: GraphClient, tmp_path: Path
) -> tuple[Path, str, str, dict[str, Any]]:
    authority = tmp_path / "authority.json"
    file_digest, payload_digest = _write_authority(authority)
    preview = apply_signed_manifest(
        authority,
        authority_file_sha256=file_digest,
        authority_payload_sha256=payload_digest,
        reason=_REASON,
        gc=client,
    )
    return authority, file_digest, payload_digest, preview


def _apply(
    client: GraphClient,
    authority: Path,
    file_digest: str,
    payload_digest: str,
    manifest_digest: str,
) -> dict[str, Any]:
    return apply_signed_manifest(
        authority,
        authority_file_sha256=file_digest,
        authority_payload_sha256=payload_digest,
        reason=_REASON,
        apply=True,
        manifest_sha256=manifest_digest,
        gc=client,
    )


def _happy_apply(
    client: GraphClient, tmp_path: Path
) -> tuple[dict[str, Any], tuple[Any, ...]]:
    _seed(client)
    authority, file_digest, payload_digest, preview = _preview(client, tmp_path)
    applied = _apply(
        client,
        authority,
        file_digest,
        payload_digest,
        preview["manifest_sha256"],
    )
    return applied, (authority, file_digest, payload_digest, preview)


@pytest.mark.graph
def test_preview_reports_would_apply(client: GraphClient, tmp_path: Path) -> None:
    _seed(client)

    _, _, _, preview = _preview(client, tmp_path)

    assert preview["outcome"] == "would_apply"
    assert preview["counts"] == {"authority_rows": 1, "admitted": 1, "refused": 0}


@pytest.mark.graph
def test_apply_changed_equals_admitted_rows(
    client: GraphClient, tmp_path: Path
) -> None:
    applied, _ = _happy_apply(client, tmp_path)

    assert applied["outcome"] == "applied"
    assert applied["changed"] == applied["counts"]["admitted"] == 1
    state = client.query(
        """
        MATCH (source:StandardNameSource {id: $source_id})
              -[:PRODUCED_NAME]->(target:StandardName {id: $target_id})
        MATCH (source)-[:FROM_DD_PATH]->(dd:IMASNode)
              -[:HAS_STANDARD_NAME]->(target)
        RETURN source.status AS status, source.produced_sn_id AS scalar,
               target.source_paths AS source_paths
        """,
        source_id=_SOURCE_ID,
        target_id=_TARGET_ID,
    )[0]
    assert state == {
        "status": "attached",
        "scalar": _TARGET_ID,
        "source_paths": ["dd:retained/path", _SOURCE_ID],
    }


@pytest.mark.graph
def test_receipt_rows_equal_admitted_rows(client: GraphClient, tmp_path: Path) -> None:
    applied, _ = _happy_apply(client, tmp_path)

    assert applied["receipt_rows"] == applied["counts"]["admitted"] == 1
    assert client.query(
        "MATCH (change:StandardNameChange {operation: $operation}) RETURN count(change) AS count",
        operation=_OPERATION,
    ) == [{"count": 1}]


@pytest.mark.graph
def test_replay_is_write_free(client: GraphClient, tmp_path: Path) -> None:
    applied, authority_parts = _happy_apply(client, tmp_path)
    authority, file_digest, payload_digest, preview = authority_parts
    before = client.query(
        "MATCH (node) OPTIONAL MATCH (node)-[edge]->() RETURN count(node) AS nodes, count(edge) AS edges"
    )[0]

    replay = _apply(
        client,
        authority,
        file_digest,
        payload_digest,
        preview["manifest_sha256"],
    )
    after = client.query(
        "MATCH (node) OPTIONAL MATCH (node)-[edge]->() RETURN count(node) AS nodes, count(edge) AS edges"
    )[0]

    assert applied["outcome"] == "applied"
    assert replay["outcome"] == "already_applied"
    assert replay["changed"] == 0
    assert replay["persistent_writes"] == 0
    assert after == before


@pytest.mark.graph
def test_already_bound_source_refusal_is_exact(
    client: GraphClient, tmp_path: Path
) -> None:
    _seed(client)
    client.query(
        """
        CREATE (other:StandardName {
          id: 'other_name', name_stage: 'accepted', validation_status: 'valid',
          status: 'draft', source_paths: [$source_id]
        })
        WITH other
        MATCH (source:StandardNameSource {id: $source_id}), (dd:IMASNode {id: $path})
        CREATE (source)-[:PRODUCED_NAME]->(other)
        CREATE (dd)-[:HAS_STANDARD_NAME]->(other)
        SET source.status = 'attached', source.produced_sn_id = other.id
        """,
        source_id=_SOURCE_ID,
        path=_PATH,
    )

    _, _, _, preview = _preview(client, tmp_path)

    assert preview["refusals"] == [
        {"row_id": _SOURCE_ID, "reason": "ordinary source is already bound"}
    ]


@pytest.mark.graph
def test_compare_and_set_drift_refuses_apply(
    client: GraphClient, tmp_path: Path
) -> None:
    _seed(client)
    authority, file_digest, payload_digest, preview = _preview(client, tmp_path)
    client.query(
        "MATCH (source:StandardNameSource {id: $source_id}) SET source.last_error = 'drifted'",
        source_id=_SOURCE_ID,
    )

    with pytest.raises(
        SignedManifestConflict,
        match="^fresh signed-manifest closure does not match authorized SHA-256$",
    ):
        _apply(
            client,
            authority,
            file_digest,
            payload_digest,
            preview["manifest_sha256"],
        )


@pytest.mark.graph
def test_unit_disagreeing_pairing_refusal_is_exact(
    client: GraphClient, tmp_path: Path
) -> None:
    _seed(client, dd_unit="m")

    _, _, _, preview = _preview(client, tmp_path)

    assert preview["refusals"] == [
        {
            "row_id": _SOURCE_ID,
            "reason": (
                f"source attachment rejected: {_SOURCE_ID}: unit dimensionality "
                f"mismatch: path '{_PATH}' declares 'm' but SN '{_TARGET_ID}' "
                "declares 'rad' — physically distinct quantities"
            ),
        }
    ]


@pytest.mark.graph
def test_out_of_allowlist_rows_are_immutable(
    client: GraphClient, tmp_path: Path
) -> None:
    _seed(client)
    before = client.query(
        """
        MATCH (name:StandardName {id: 'collateral_name'}),
              (source:StandardNameSource {id: 'dd:collateral/path'})
        RETURN properties(name) AS name, properties(source) AS source
        """
    )[0]

    authority, file_digest, payload_digest, preview = _preview(client, tmp_path)
    _apply(
        client,
        authority,
        file_digest,
        payload_digest,
        preview["manifest_sha256"],
    )
    after = client.query(
        """
        MATCH (name:StandardName {id: 'collateral_name'}),
              (source:StandardNameSource {id: 'dd:collateral/path'})
        RETURN properties(name) AS name, properties(source) AS source
        """
    )[0]

    assert after == before
