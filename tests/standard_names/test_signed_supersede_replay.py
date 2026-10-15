"""Contract: a replayed signed supersede with a successor reads as applied.

A signed supersede that names a successor rewrites the predecessor's node
properties, and those properties are folded into the participant snapshot the
manifest digest covers. The next preview therefore computes a different digest
than the apply did, so a receipt lookup keyed on the digest would miss the
receipt and re-admit the row. The replay recognition is keyed on the authority
payload instead, and it verifies each admitted row's postcondition against the
digest its receipt was minted under.

The graph is stubbed: the stub answers the receipt lookup only when the query
carries the authority payload, so keying the lookup on the digest instead makes
the receipt invisible and the row re-admits.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

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
_MINTED_DIGEST = "0f" * 32


def _authority() -> dict[str, Any]:
    selection = {
        "id": "artifact-rows",
        "mode": "exact_complete_signed_cohort",
        "predicate": "artifact-rows",
    }
    row = {
        "id": _ROW_ID,
        "identity": {
            "id": _ROW_ID,
            "kind": "standard_name",
            "target_id": _PREDECESSOR,
        },
        "participants": [
            {"id": _PREDECESSOR, "kind": "node", "graph_label": "StandardName"},
            {"id": _SUCCESSOR, "kind": "node", "graph_label": "StandardName"},
        ],
        "selection": selection,
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
    authority = {
        "schema": "imas-codex.repair-authority.v1",
        "operation_id": "recover-recorded-successor-lineage",
        "authority_mode": "external_reviewed",
        "rows": [row],
        "repair_rows": [row["id"]],
        "selection": selection,
        "receipt_policy": {
            "id": "one-per-superseded-identity",
            "operation": "link_signed_successor",
            "cardinality": "per_target",
            "expected_count": "admitted_rows",
            "link_participant_kind": "standard_name",
            "replay_projection": ["manifest_sha256", "row_id"],
        },
        "orphan_policy": "refuse",
    }
    authority["signature"] = {
        "canonicalization": "json-sort-keys-v1",
        "sha256": signed_payload_sha256(authority),
    }
    return authority


class _StubTransaction:
    """Answers the receipt lookup only when it carries the authority payload.

    Every other query returns an empty result, so a row whose receipt is not
    recognised refuses at participant lookup rather than reading as admitted.
    """

    def __init__(
        self,
        payload_sha256: str,
        authority_file_sha256: str,
        successor_edge: dict[str, Any],
    ) -> None:
        self._payload_sha256 = payload_sha256
        self._authority_file_sha256 = authority_file_sha256
        self._successor_edge = successor_edge

    @property
    def closed(self) -> bool:
        return False

    def rollback(self) -> None:
        return None

    def commit(self) -> None:
        return None

    def run(self, cypher: str, **params: Any) -> list[dict[str, Any]]:
        if "StandardNameChange {" in cypher:
            values = {str(value) for value in params.values()}
            if self._payload_sha256 in values:
                return [{"properties": self._receipt()}]
            return []
        if "node.name_stage AS name_stage, node.status AS status" in cypher:
            return [{"name_stage": "superseded", "status": "superseded"}]
        if "collect(relationship.receipt_id) AS receipt_ids" in cypher:
            return [dict(self._successor_edge)]
        return []

    def _receipt(self) -> dict[str, Any]:
        return {
            "row_id": _ROW_ID,
            "from_name": _PREDECESSOR,
            "to_name": _SUCCESSOR,
            "operation": "link_signed_successor",
            "manifest_sha256": _MINTED_DIGEST,
            "authority_file_sha256": self._authority_file_sha256,
            "authority_payload_sha256": self._payload_sha256,
            "cohort_admitted_ids": [_ROW_ID],
        }


class _StubSession:
    def __init__(self, transaction: _StubTransaction) -> None:
        self._transaction = transaction

    def __enter__(self) -> _StubSession:
        return self

    def __exit__(self, *exc: object) -> bool:
        return False

    def begin_transaction(self) -> _StubTransaction:
        return self._transaction


class _StubClient:
    def __init__(self, transaction: _StubTransaction) -> None:
        self._transaction = transaction

    def session(self) -> _StubSession:
        return _StubSession(self._transaction)

    def close(self) -> None:
        return None


def _preview(
    tmp_path: Path,
    *,
    successor_edge_present: bool,
) -> dict[str, Any]:
    authority = _authority()
    raw = (json.dumps(authority, sort_keys=True, indent=2) + "\n").encode()
    path = tmp_path / "authority.json"
    path.write_bytes(raw)
    file_sha256 = hashlib.sha256(raw).hexdigest()
    payload_sha256 = signed_payload_sha256(authority)

    expected_receipt_id = _change_id(_MINTED_DIGEST, _ROW_ID)
    successor_edge = (
        {
            "receipt_ids": [expected_receipt_id],
            "successor_ids": [_SUCCESSOR],
            "count": 1,
        }
        if successor_edge_present
        else {"receipt_ids": [], "successor_ids": [], "count": 0}
    )
    transaction = _StubTransaction(payload_sha256, file_sha256, successor_edge)
    client = _StubClient(transaction)
    return apply_signed_manifest(
        path,
        authority_file_sha256=file_sha256,
        authority_payload_sha256=payload_sha256,
        reason="Recover successor links named by recorded lineage evidence",
        apply=False,
        client_factory=lambda: client,
    )


def test_replayed_supersede_with_successor_is_already_applied(tmp_path: Path) -> None:
    receipt = _preview(tmp_path, successor_edge_present=True)
    assert receipt["outcome"] == "already_applied"
    assert receipt["changed"] == 0


def test_receipt_without_successor_edge_is_not_already_applied(
    tmp_path: Path,
) -> None:
    with pytest.raises(SignedManifestConflict):
        _preview(tmp_path, successor_edge_present=False)
