"""Evidence that a documentation restore is admitted where set_properties is not."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from imas_codex.graph.client import GraphClient
from imas_codex.standard_names.signed_manifest import (
    _DOCS_RESTORE_ADAPTER,
    _DOCS_RESTORE_GUARDS,
    _DOCS_RESTORE_MUTATION,
    apply_signed_manifest,
    signed_payload_sha256,
)

_SELECTION = {
    "id": "artifact-rows",
    "mode": "exact_complete_signed_cohort",
    "predicate": "artifact-rows",
}


def _write_authority(
    path: Path, rows: list[dict[str, Any]], operation: str
) -> tuple[str, str]:
    authority: dict[str, Any] = {
        "schema": "imas-codex.docs-revision-restore.v1",
        "operation_id": operation,
        "rows": rows,
        "selection": _SELECTION,
    }
    payload_sha256 = signed_payload_sha256(authority)
    authority["signature"] = {
        "canonicalization": "json-sort-keys-v1",
        "sha256": payload_sha256,
    }
    raw = json.dumps(authority, sort_keys=True, indent=2).encode()
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest(), payload_sha256


def _row(
    row_id: str, sn_id: str, snapshot_id: str, properties: dict[str, Any] | None = None
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "id": row_id,
        "sn_id": sn_id,
        "snapshot_id": snapshot_id,
        "selection": _SELECTION,
    }
    if properties is not None:
        row["properties"] = properties
    return row


def _restore(
    path: Path,
    file_sha256: str,
    payload_sha256: str,
    client: GraphClient,
    *,
    apply: bool = False,
    manifest_sha256: str | None = None,
    guard_set: tuple[str, ...] = _DOCS_RESTORE_GUARDS,
) -> dict[str, Any]:
    return apply_signed_manifest(
        path,
        authority_file_sha256=file_sha256,
        authority_payload_sha256=payload_sha256,
        authority_adapter=_DOCS_RESTORE_ADAPTER,
        mutation_kind=_DOCS_RESTORE_MUTATION,
        guard_set=guard_set,
        reason="restore documentation from its own DocsRevision snapshot",
        apply=apply,
        manifest_sha256=manifest_sha256,
        gc=client,
    )


@pytest.fixture
def client(disposable_neo4j: tuple[str, str]) -> Iterator[GraphClient]:
    uri, password = disposable_neo4j
    graph = GraphClient(
        uri=uri,
        username="neo4j",
        password=password,
        graph_name="signed-docs-restore",
    )
    graph.query("MATCH (node) DETACH DELETE node")
    yield graph
    graph.query("MATCH (node) DETACH DELETE node")
    graph.close()


def _seed_name_with_live_child(client: GraphClient, sn_id: str) -> None:
    client.query(
        """
        CREATE (:StandardName {
          id: $sn_id,
          name_stage: 'accepted',
          status: 'draft',
          unit: '1',
          kind: 'scalar',
          physical_base: $sn_id,
          docs_stage: 'pending',
          documentation: 'reset text',
          description: 'reset description',
          docs_chain_length: 0,
          source_paths: []
        })
        """,
        sn_id=sn_id,
    )
    client.query(
        """
        MATCH (target:StandardName {id: $sn_id})
        CREATE (:StandardName {
          id: $sn_id + '_child',
          name_stage: 'accepted',
          status: 'draft',
          source_paths: []
        })-[:HAS_PARENT]->(target)
        """,
        sn_id=sn_id,
    )


def _seed_snapshot(client: GraphClient, sn_id: str, snapshot_id: str) -> None:
    client.query(
        """
        MATCH (target:StandardName {id: $sn_id})
        CREATE (target)-[:DOCS_REVISION_OF]->(:DocsRevision {
          id: $snapshot_id,
          standard_name_id: $sn_id,
          documentation: 'accepted documentation text',
          description: 'accepted description',
          revision_number: 3,
          reviewer_score_docs: 0.875,
          model: 'deterministic:test',
          created_at: datetime()
        })
        """,
        sn_id=sn_id,
        snapshot_id=snapshot_id,
    )


@pytest.mark.graph
def test_docs_restore_is_admitted_on_a_name_with_a_live_structural_child(
    client: GraphClient, tmp_path: Path
) -> None:
    sn_id = "docs-restore-target"
    snapshot_id = f"{sn_id}#rev-3"
    _seed_name_with_live_child(client, sn_id)
    _seed_snapshot(client, sn_id, snapshot_id)

    path = tmp_path / "docs-authority.json"
    file_sha256, payload_sha256 = _write_authority(
        path,
        [_row(f"docs:{sn_id}", sn_id, snapshot_id)],
        "restore_documentation_from_revision",
    )

    preview = _restore(path, file_sha256, payload_sha256, client)
    assert preview["outcome"] == "would_apply"
    apply_result = _restore(
        path,
        file_sha256,
        payload_sha256,
        client,
        apply=True,
        manifest_sha256=preview["manifest_sha256"],
    )
    assert apply_result["outcome"] == "applied"
    assert apply_result["docs_revision_ids"] == {sn_id: snapshot_id}

    restored = client.query(
        """
        MATCH (target:StandardName {id: $sn_id})
        RETURN target.documentation AS documentation,
               target.description AS description,
               target.docs_stage AS docs_stage,
               target.docs_chain_length AS docs_chain_length,
               target.unit AS unit,
               target.kind AS kind,
               target.physical_base AS physical_base
        """,
        sn_id=sn_id,
    )[0]
    assert restored["documentation"] == "accepted documentation text"
    assert restored["description"] == "accepted description"
    assert restored["docs_stage"] == "accepted"
    assert restored["docs_chain_length"] == 4
    assert restored["unit"] == "1"
    assert restored["kind"] == "scalar"
    assert restored["physical_base"] == sn_id

    receipts = client.query(
        """
        MATCH (change:StandardNameChange {docs_revision_id: $snapshot_id})
        RETURN change.docs_revision_id AS snapshot_id,
               change.operation AS operation,
               change.row_id AS row_id
        """,
        snapshot_id=snapshot_id,
    )
    assert receipts == [
        {
            "snapshot_id": snapshot_id,
            "operation": "restore_documentation_from_revision",
            "row_id": f"docs:{sn_id}",
        }
    ]

    replay = _restore(
        path,
        file_sha256,
        payload_sha256,
        client,
        apply=True,
        manifest_sha256=preview["manifest_sha256"],
    )
    assert replay["outcome"] == "already_applied"
    assert replay["changed"] == 0


@pytest.mark.graph
def test_set_properties_is_still_refused_on_the_same_live_child_name(
    client: GraphClient, tmp_path: Path
) -> None:
    sn_id = "structural-restore-target"
    _seed_name_with_live_child(client, sn_id)
    row = {
        "id": f"props:{sn_id}",
        "identity": {
            "id": f"props:{sn_id}",
            "kind": "standard_name",
            "target_id": sn_id,
        },
        "participants": [{"id": sn_id, "kind": "node", "graph_label": "StandardName"}],
        "selection": _SELECTION,
        "mutations": [
            {
                "id": f"props:{sn_id}:mutation",
                "order": 0,
                "kind": "set_properties",
                "participant_id": sn_id,
                "arguments": {"properties": {"docs_stage": "accepted"}},
            }
        ],
        "guards": [
            {
                "id": "structural-legitimacy",
                "kind": "semantic_authority",
                "implementation": "structural-legitimacy",
                "participant_ids": [sn_id],
            },
            {
                "id": "out-of-allowlist-immutability",
                "kind": "collateral_immutability",
                "implementation": "out-of-allowlist-immutability",
                "participant_ids": [sn_id],
            },
        ],
        "orphan_policy": "refuse",
    }
    authority: dict[str, Any] = {
        "schema": "imas-codex.repair-authority.v1",
        "operation_id": "compare-the-set-properties-path",
        "authority_mode": "external_reviewed",
        "rows": [row],
        "repair_rows": [row["id"]],
        "selection": _SELECTION,
        "receipt_policy": {
            "id": "one-per-logical-change",
            "operation": "compare_the_set_properties_path",
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
    path = tmp_path / "props-authority.json"
    raw = json.dumps(authority, sort_keys=True, indent=2).encode()
    path.write_bytes(raw)

    preview = apply_signed_manifest(
        path,
        authority_file_sha256=hashlib.sha256(raw).hexdigest(),
        authority_payload_sha256=payload_sha256,
        reason="show the structural guard still governs set_properties",
        gc=client,
    )
    assert preview["outcome"] == "refused"
    assert preview["refusals"] == [
        {
            "row_id": f"props:{sn_id}",
            "reason": "target has a live structural child",
        }
    ]


@pytest.mark.graph
def test_docs_restore_refuses_a_payload_carrying_a_structural_field(
    client: GraphClient, tmp_path: Path
) -> None:
    sn_id = "docs-restore-payload-target"
    snapshot_id = f"{sn_id}#rev-1"
    _seed_name_with_live_child(client, sn_id)
    _seed_snapshot(client, sn_id, snapshot_id)

    path = tmp_path / "docs-payload-authority.json"
    file_sha256, payload_sha256 = _write_authority(
        path,
        [_row(f"docs:{sn_id}", sn_id, snapshot_id, {"unit": "T"})],
        "restore_documentation_from_revision",
    )

    preview = _restore(path, file_sha256, payload_sha256, client)
    assert preview["outcome"] == "refused"
    assert preview["refusals"] == [
        {
            "row_id": f"docs:{sn_id}",
            "reason": ("restore payload carries a non-documentation field: unit"),
        }
    ]
    unchanged = client.query(
        "MATCH (target:StandardName {id: $sn_id}) "
        "RETURN target.docs_stage AS docs_stage, target.unit AS unit",
        sn_id=sn_id,
    )[0]
    assert unchanged == {"docs_stage": "pending", "unit": "1"}
