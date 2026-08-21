"""Canonical repair-authority builder contract."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from imas_codex.graph.models import RepairAuthorityArtifact
from imas_codex.standard_names import signed_manifest as operator
from imas_codex.standard_names.repair_authority import (
    ARTIFACT_ROWS_SELECTION,
    RepairAuthorityBuildError,
    build_repair_authority,
)
from imas_codex.standard_names.signed_manifest import signed_payload_sha256

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_EVIDENCE_ROOT = _REPOSITORY_ROOT / "docs/evidence/sn-graph-wide-integrity"
_COMMITTED_AUTHORITIES = {
    "catalog-edit-dual-binding-adjudication.json": (
        "5ca7761a7b022ac7889387d7bf63a027114a168cc3785ed4fdc8d31c08417b6e"
    ),
    "refused-target-orphan-adjudication.json": (
        "2c2d38f3241ec3057d24a5d05c27840f5e4ffe99520063059ab31c1e9d4bca36"
    ),
    "owner-geometry-rc66-partition.json": (
        "dbb37f7be12ba99d7e85bf13b9d63e6c19cb6c20bd35fe687e590f798e2dc85b"
    ),
    "stale-source-lifecycle.json": (
        "f2da3ff78d5427fe4477bc46c57a7dc33c8c2d6659d4a48e52f94a4014ae90ad"
    ),
}


def _node(node_id: str, label: str = "StandardName") -> dict[str, Any]:
    return {"id": node_id, "kind": "node", "graph_label": label}


def _relationship(
    relationship_id: str, relationship_type: str = "PRODUCED_NAME"
) -> dict[str, Any]:
    return {
        "id": relationship_id,
        "kind": "relationship",
        "graph_label": relationship_type,
    }


def _guard(implementation: str) -> dict[str, Any]:
    kind = (
        "collateral_immutability"
        if implementation == "out-of-allowlist-immutability"
        else "semantic_authority"
    )
    return {
        "id": implementation,
        "kind": kind,
        "implementation": implementation,
        "participant_ids": [],
    }


def _simple_row(mutation_kind: str) -> dict[str, Any]:
    row_id = f"row:{mutation_kind}"
    guard_names = ["out-of-allowlist-immutability"]
    if mutation_kind in {"delete", "supersede"}:
        guard_names.append("structural-legitimacy")
    if mutation_kind == "detach":
        guard_names.append("last-producing-source")
    return {
        "id": row_id,
        "identity": {
            "id": row_id,
            "kind": "standard_name",
            "target_id": row_id,
        },
        "participants": [_node(row_id)],
        "mutations": [
            {
                "id": f"mutation:{mutation_kind}",
                "order": 0,
                "kind": mutation_kind,
                "participant_id": row_id,
                "arguments": (
                    {"properties": {"status": "superseded"}}
                    if mutation_kind == "set_properties"
                    else None
                ),
            }
        ],
        "guards": [_guard(name) for name in guard_names],
        "orphan_policy": "refuse",
    }


def _relationship_delete_row() -> dict[str, Any]:
    source_id = "source:dual-binding"
    survivor_id = "survivor-name"
    losing_id = "losing-name"
    losing_binding = "binding:losing"
    survivor_binding = "binding:survivor"
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
            _node(survivor_id),
            _node(losing_id),
            _relationship(survivor_binding),
            _relationship(losing_binding),
        ],
        "mutations": [
            {
                "id": "remove-losing-binding",
                "order": 0,
                "kind": "delete_relationship",
                "participant_id": losing_binding,
            },
            {
                "id": "select-survivor",
                "order": 1,
                "kind": "set_properties",
                "participant_id": source_id,
                "arguments": {"properties": {"produced_sn_id": survivor_id}},
            },
        ],
        "guards": [
            _guard("last-producing-source"),
            _guard("out-of-allowlist-immutability"),
        ],
        "orphan_policy": "refuse",
    }


def _relationship_add_row() -> dict[str, Any]:
    target_id = "electron_diffusivity"
    source_id = f"derived:{target_id}"
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
            _node(target_id),
            _node("electron_heat_diffusivity"),
            _relationship("parent:electron-heat", "HAS_PARENT"),
        ],
        "mutations": [
            {
                "id": "restore-binding",
                "order": 0,
                "kind": "add_relationship",
                "participant_id": target_id,
                "arguments": {
                    "relationship_type": "PRODUCED_NAME",
                    "start_id": source_id,
                    "end_id": target_id,
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
                        "source_id": target_id,
                        "batch_key": "derived_parent",
                        "produced_sn_id": target_id,
                        "claimed_at": None,
                        "claim_token": None,
                    }
                },
            },
        ],
        "guards": [_guard("out-of-allowlist-immutability")],
        "orphan_policy": "refuse",
    }


def _row_for(mutation_kind: str) -> dict[str, Any]:
    if mutation_kind == "delete_relationship":
        return _relationship_delete_row()
    if mutation_kind == "add_relationship":
        return _relationship_add_row()
    return _simple_row(mutation_kind)


def _specification(mutation_kind: str) -> dict[str, Any]:
    return {
        "operation_id": f"exercise-{mutation_kind}",
        "authority_mode": "external_reviewed",
        "rows": [_row_for(mutation_kind)],
        "receipt_policy": {
            "id": "one-per-logical-change",
            "operation": f"exercise_{mutation_kind}",
            "cardinality": "per_target",
            "link_participant_kind": "standard_name",
            "replay_projection": ["manifest_sha256", "row_id"],
        },
        "orphan_policy": "refuse",
    }


@pytest.mark.parametrize("mutation_kind", sorted(operator._GENERIC_MUTATION_KINDS))
def test_builder_emits_loadable_authority_for_registered_mutation(
    mutation_kind: str, tmp_path: Path
) -> None:
    built = build_repair_authority(_specification(mutation_kind))
    data = json.loads(built.content)
    path = tmp_path / f"{mutation_kind}.json"
    path.write_bytes(built.content)

    loaded = operator._load_authority(
        path,
        expected_file_sha256=built.file_sha256,
        expected_payload_sha256=built.payload_sha256,
    )

    assert built.artifact.schema == "imas-codex.repair-authority.v1"
    assert loaded.rows[0].id == data["rows"][0]["id"]
    assert data["selection"]["id"] == ARTIFACT_ROWS_SELECTION
    assert data["selection"]["predicate"] == ARTIFACT_ROWS_SELECTION
    assert data["rows"][0]["selection"] == data["selection"]
    assert data["repair_rows"] == [data["rows"][0]["id"]]
    assert data["receipt_policy"]["expected_count"] == "admitted_rows"


def test_digests_are_derived_from_final_emitted_bytes() -> None:
    built = build_repair_authority(_specification("set_properties"))
    emitted = json.loads(built.content)

    assert built.file_sha256 == hashlib.sha256(built.content).hexdigest()
    assert built.payload_sha256 == signed_payload_sha256(emitted)
    assert emitted["signature"]["sha256"] == built.payload_sha256


def test_builder_rejects_top_level_all_or_nothing() -> None:
    specification = _specification("set_properties")
    specification["all_or_nothing"] = True

    with pytest.raises(
        RepairAuthorityBuildError,
        match="^top-level all_or_nothing is not part of a canonical repair authority$",
    ):
        build_repair_authority(specification)


@pytest.mark.parametrize("selection_role", ["artifact", "row"])
def test_builder_rejects_open_selection_predicate(selection_role: str) -> None:
    specification = _specification("set_properties")
    if selection_role == "artifact":
        specification["selection"] = {"predicate": "caller-supplied-query"}
    else:
        specification["rows"][0]["selection"] = {"predicate": "caller-supplied-query"}

    with pytest.raises(
        RepairAuthorityBuildError,
        match="selection predicate must be 'artifact-rows'",
    ):
        build_repair_authority(specification)


@pytest.mark.parametrize(
    ("filename", "expected_file_sha256"), _COMMITTED_AUTHORITIES.items()
)
def test_committed_authority_remains_loadable_without_resigning(
    filename: str, expected_file_sha256: str
) -> None:
    path = _EVIDENCE_ROOT / filename
    original = path.read_bytes()
    data = json.loads(original)
    signature = data.get("signature")

    RepairAuthorityArtifact.model_validate(data)

    assert path.read_bytes() == original
    assert hashlib.sha256(original).hexdigest() == expected_file_sha256
    assert json.loads(path.read_bytes()).get("signature") == signature
