"""Compatibility and expressivity checks for repair-authority artifacts.

All nine extension points recorded by
``docs/evidence/sn-repair-operator-consolidation/operator-characterisation.md``
are modeled, with none left out of scope:

* provenance adapters map to ``RepairAuthorityDigest`` and its signature profile;
* authority source modes map to ``RepairAuthorityMode``;
* typed row and participant identities map to ``RepairRowIdentity`` and
  ``RepairParticipant``;
* selection completeness maps to ``RepairSelection``;
* joined authority maps to ``RepairAuthorityJoin``;
* branching compound mutations map to ordered ``RepairMutation`` rows;
* semantic guard plug-ins map to named ``RepairGuard`` implementations;
* receipt cardinality and replay projection map to ``RepairReceiptPolicy``;
* permitted orphan hand-off maps to ``RepairOrphanPolicy``.

The compatibility envelope deliberately retains legacy JSON values without
normalizing them. Those values are immutable evidence, while the typed fields
define the canonical shape for new repair authorities.
"""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import pytest

from imas_codex.graph.models import (
    RepairAuthorityArtifact,
    RepairAuthorityRow,
    RepairMutationKind,
)

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


@pytest.mark.parametrize(
    ("filename", "expected_digest"), _COMMITTED_AUTHORITIES.items()
)
def test_committed_authority_validates_without_resigning(
    filename: str, expected_digest: str
) -> None:
    path = _EVIDENCE_ROOT / filename
    original_bytes = path.read_bytes()
    original_data = json.loads(original_bytes)

    validated = RepairAuthorityArtifact.model_validate(original_data)

    assert validated.schema == original_data["schema"]
    assert len(validated.rows) == len(original_data["rows"])
    assert original_data == json.loads(original_bytes)
    assert path.read_bytes() == original_bytes
    assert sha256(path.read_bytes()).hexdigest() == expected_digest


def test_canonical_authority_fields_cover_the_recorded_extensions() -> None:
    artifact_fields = RepairAuthorityArtifact.model_fields
    row_fields = RepairAuthorityRow.model_fields

    assert {
        "authority_mode",
        "authority_digests",
        "selection",
        "authority_joins",
        "repair_rows",
        "receipt_policy",
        "orphan_policy",
    } <= artifact_fields.keys()
    assert {
        "identity",
        "signatures",
        "participants",
        "selection",
        "mutations",
        "guards",
        "orphan_policy",
    } <= row_fields.keys()


def test_mutation_vocabulary_includes_lifecycle_and_edge_removal() -> None:
    assert {"delete", "supersede", "detach"} <= {
        member.value for member in RepairMutationKind
    }
