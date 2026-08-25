"""Generic transaction envelope for typed, signed graph-repair authorities.

The authority file chooses rows, mutations, and guards.  Callers supply only
the file and its two independently trusted digests; they cannot narrow or
expand the executable cohort.  The graph closure is derived again inside an
applying transaction, locked, and re-hashed before any mutation.

Only closed registries are interpreted here.  Authority artifacts never carry
Cypher.  The mutation registry contains ``set_properties``, ``delete``,
``supersede``, ``detach``, and two closed relationship programs:
``delete_relationship`` for source-target reconciliation,
``add_relationship`` for structural-source revival or one unbound ordinary-source
attachment, their exact delete/add/set combination for ordinary-source migration,
``restore_semantic_mirror`` for scalar and backing-projection restoration, and
``recompute_projection`` for an exact structural reparent or parent release.
The semantic guard registry contains last-producing-source,
structural-legitimacy, and out-of-allowlist-immutability.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from pydantic import ValidationError

from imas_codex.discovery.base.claims import retry_on_deadlock
from imas_codex.graph.client import GraphClient
from imas_codex.graph.models import (
    RepairAuthorityArtifact,
    RepairAuthorityDigest,
    RepairAuthorityRow,
    RepairGuard,
    RepairGuardKind,
    RepairMutation,
    RepairMutationKind,
    RepairParticipant,
    RepairParticipantKind,
    RepairReceiptPolicy,
    RepairRowIdentity,
    RepairSelection,
)

SIGNED_MANIFEST_SCHEMA = "imas-codex.signed-repair-manifest.v1"
SIGNED_MANIFEST_RECEIPT_SCHEMA = "imas-codex.signed-repair-receipt.v1"
SIGNED_AUTHORITY_CANONICALIZATION = "json-sort-keys-v1"

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_NODE_LABELS = frozenset({"StandardName", "StandardNameSource"})
_RELATIONSHIP_TYPES = frozenset({"PRODUCED_NAME", "HAS_PARENT"})
_MUTATION_KINDS = frozenset(
    {
        RepairMutationKind.set_properties.value,
        RepairMutationKind.delete.value,
        RepairMutationKind.supersede.value,
        RepairMutationKind.detach.value,
        RepairMutationKind.delete_relationship.value,
    }
)
_STRUCTURAL_REPARENT = RepairMutationKind.recompute_projection.value
_GENERIC_MUTATION_KINDS = _MUTATION_KINDS | {RepairMutationKind.add_relationship.value}
_SIGNED_MANIFEST_MUTATION_KINDS = _GENERIC_MUTATION_KINDS | {_STRUCTURAL_REPARENT}
_LAST_PRODUCER = "last-producing-source"
_STRUCTURAL_LEGITIMACY = "structural-legitimacy"
_COLLATERAL_IMMUTABILITY = "out-of-allowlist-immutability"
_SIGNED_LIFECYCLE = "signed-lifecycle-and-claim"
_NO_LIVE_PRODUCER = "no-live-producing-source"
_NO_LIVE_STRUCTURAL_CHILD = "no-live-structural-child"
_ERROR_SIBLING_PARENT_ABSENCE = "recognized-error-sibling-parent-absence"
_GUARD_KINDS = {
    _LAST_PRODUCER: RepairGuardKind.semantic_authority.value,
    _STRUCTURAL_LEGITIMACY: RepairGuardKind.semantic_authority.value,
    _COLLATERAL_IMMUTABILITY: RepairGuardKind.collateral_immutability.value,
    _SIGNED_LIFECYCLE: RepairGuardKind.semantic_authority.value,
    _NO_LIVE_PRODUCER: RepairGuardKind.semantic_authority.value,
    _NO_LIVE_STRUCTURAL_CHILD: RepairGuardKind.semantic_authority.value,
    _ERROR_SIBLING_PARENT_ABSENCE: RepairGuardKind.semantic_authority.value,
}

_REFUSED_TARGET_ORPHAN_ADAPTER = "refused-target-orphan"
_REFUSED_TARGET_ORPHAN_SCHEMA = "imas-codex.refused-target-orphan-adjudication.v2"
_REFUSED_TARGET_ORPHAN_DISPOSITION = "retire_under_orphan_policy"
_REFUSED_TARGET_ORPHAN_RECEIPT_SCHEMA = (
    "imas-codex.signed-provenance-orphan-retirement-receipt.v1"
)
_REFUSED_TARGET_ORPHAN_DISPOSITIONS = frozenset(
    {
        "preserve_as_structural_identity",
        "re_source_from_existing_dd_path",
        "retain_competing_binding",
        _REFUSED_TARGET_ORPHAN_DISPOSITION,
    }
)

_STALE_SOURCE_ADAPTER = "stale-source-lifecycle"
_STALE_SOURCE_LIFECYCLE_SCHEMA = "imas-codex.stale-source-lifecycle-disposition.v1"
_STALE_SOURCE_RECEIPT_SCHEMA = "imas-codex.signed-stale-source-detach-receipt.v1"
_STALE_SOURCE_GUARDS = (
    _SIGNED_LIFECYCLE,
    _LAST_PRODUCER,
    _COLLATERAL_IMMUTABILITY,
)

_ERROR_SIBLING_ADAPTER = "error-sibling-reconcile"
_ERROR_SIBLING_MODEL = "deterministic:dd_error_modifier"
_ERROR_SIBLING_REASON = "orphaned error sibling (parent name deleted)"
_ERROR_SIBLING_GUARDS = (
    _ERROR_SIBLING_PARENT_ABSENCE,
    _COLLATERAL_IMMUTABILITY,
)

_DUAL_AUTHORITY_ADAPTER = "dual-authority-retirement"
_DUAL_AUTHORITY_MUTATION = "release-and-supersede"
_AUTHORITY_JOIN = "authority-join"
_DUAL_AUTHORITY_GUARDS = (
    _AUTHORITY_JOIN,
    _SIGNED_LIFECYCLE,
    _NO_LIVE_STRUCTURAL_CHILD,
    _COLLATERAL_IMMUTABILITY,
)
_CATALOG_DISPOSITION_ADAPTER = "catalog-disposition"
_CATALOG_DISPOSITION_MUTATION = "select-survivor-and-release-bindings"
_CATALOG_DISPOSITION_GUARDS = (
    "signed-adjudication",
    _LAST_PRODUCER,
    _STRUCTURAL_LEGITIMACY,
    _COLLATERAL_IMMUTABILITY,
)
_INELIGIBLE_SOURCE_ADAPTER = "ineligible-source-retirement"
_INELIGIBLE_SOURCE_MUTATION = "release-ineligible-source-authority"
_INELIGIBLE_SOURCE_GUARDS = (
    _SIGNED_LIFECYCLE,
    "permitted-orphan-hand-off",
    _COLLATERAL_IMMUTABILITY,
)

_SEMANTIC_MIRROR_ADAPTER = "semantic-mirror-repair"
_SEMANTIC_MIRROR_MUTATION = "restore-semantic-mirror"
_SEMANTIC_MIRROR_GUARDS = (
    "sole-live-target-authority",
    "exact-upstream-backing",
    "out-of-allowlist-immutability",
)

_DD_RESIDUE_RELEASE_OPERATION = "release_legacy_dd_source_lifecycle"
_DD_RESIDUE_SOURCE_IDS = frozenset(
    {
        "dd:ntms/time_slice/mode",
        "dd:summary/pedestal_fits",
        "dd:waves/coherent_wave",
    }
)
_DD_RESIDUE_RELEASE_PROPERTIES = {
    "status": "extracted",
    "attempt_count": 0,
    "claimed_at": None,
    "claim_token": None,
    "produced_sn_id": None,
    "composed_at": None,
}


class SignedManifestAuthorityError(ValueError):
    """The authority bytes or typed repair program are invalid."""


class SignedManifestConflict(RuntimeError):
    """Current graph authority does not match the authorized manifest."""


class StaleSourceDetachConflict(SignedManifestConflict):
    """The signed stale-source closure no longer matches live graph authority."""


class _Query(Protocol):
    def query(self, cypher: str, **params: Any) -> list[dict[str, Any]]: ...


class _TransactionQuery:
    def __init__(self, transaction: Any) -> None:
        self._transaction = transaction

    def query(self, cypher: str, **params: Any) -> list[dict[str, Any]]:
        return [dict(record) for record in self._transaction.run(cypher, **params)]


@dataclass(frozen=True)
class _LoadedRow:
    id: str
    identity: dict[str, Any]
    participants: tuple[dict[str, Any], ...]
    mutations: tuple[dict[str, Any], ...]
    guards: tuple[dict[str, Any], ...]
    orphan_policy: str


@dataclass(frozen=True)
class _Authority:
    data: dict[str, Any]
    operation_id: str
    rows: tuple[_LoadedRow, ...]
    receipt_policy: dict[str, Any]
    file_sha256: str
    payload_sha256: str


@dataclass
class _Preview:
    manifest: dict[str, Any]
    manifest_sha256: str
    admitted: list[dict[str, Any]]
    refusals: list[dict[str, str]]
    collateral: list[dict[str, str]]


def _validate_dd_residue_release_authority(
    operation_id: str,
    rows: Sequence[_LoadedRow],
    receipt_policy: Mapping[str, Any],
) -> None:
    """Keep the historical DD lifecycle release closed to its exact cohort."""
    if operation_id != _DD_RESIDUE_RELEASE_OPERATION:
        return

    source_ids = {str(row.identity.get("source_id") or "") for row in rows}
    valid_rows = all(
        row.id == row.identity.get("source_id")
        and row.identity.get("kind") == "source"
        and row.identity.get("target_id") is None
        and len(
            [
                participant
                for participant in row.participants
                if participant.get("kind") == RepairParticipantKind.node.value
                and participant.get("graph_label") == "StandardNameSource"
                and participant.get("id") == row.id
            ]
        )
        == 1
        and sum(
            participant.get("kind") == RepairParticipantKind.node.value
            and participant.get("graph_label") == "StandardNameSource"
            for participant in row.participants
        )
        == 1
        and sum(
            participant.get("kind") == RepairParticipantKind.node.value
            and participant.get("graph_label") == "StandardName"
            for participant in row.participants
        )
        == sum(
            participant.get("kind") == RepairParticipantKind.relationship.value
            and participant.get("graph_label") == "PRODUCED_NAME"
            for participant in row.participants
        )
        and len(
            [
                mutation
                for mutation in row.mutations
                if mutation.get("kind") == RepairMutationKind.set_properties.value
                and mutation.get("participant_id") == row.id
                and mutation.get("arguments", {}).get("properties")
                == _DD_RESIDUE_RELEASE_PROPERTIES
            ]
        )
        == 1
        and {
            str(mutation["participant_id"])
            for mutation in row.mutations
            if mutation.get("kind") == RepairMutationKind.delete_relationship.value
        }
        == {
            str(participant["id"])
            for participant in row.participants
            if participant.get("kind") == RepairParticipantKind.relationship.value
            and participant.get("graph_label") == "PRODUCED_NAME"
        }
        and all(
            mutation.get("kind")
            in {
                RepairMutationKind.delete_relationship.value,
                RepairMutationKind.set_properties.value,
            }
            for mutation in row.mutations
        )
        and _guard_names(row) == {_LAST_PRODUCER, _COLLATERAL_IMMUTABILITY}
        and row.orphan_policy == "refuse"
        for row in rows
    )
    if (
        source_ids != _DD_RESIDUE_SOURCE_IDS
        or len(rows) != len(_DD_RESIDUE_SOURCE_IDS)
        or not valid_rows
        or receipt_policy.get("operation") != _DD_RESIDUE_RELEASE_OPERATION
    ):
        raise SignedManifestAuthorityError(
            "legacy DD lifecycle release requires its exact closed source cohort"
        )


def _validate_source_target_reconciliation_program(
    row_id: str,
    identity: dict[str, Any],
    participants: list[dict[str, Any]],
    mutations: list[dict[str, Any]],
) -> None:
    """Validate the sole closed program admitted for relationship deletion."""
    relationship_deletes = [
        mutation
        for mutation in mutations
        if mutation["kind"] == RepairMutationKind.delete_relationship.value
    ]
    if not relationship_deletes or any(
        mutation["kind"] == RepairMutationKind.add_relationship.value
        for mutation in mutations
    ):
        return
    if (
        identity.get("source_id") in _DD_RESIDUE_SOURCE_IDS
        and identity.get("target_id") is None
        and any(
            mutation.get("kind") == RepairMutationKind.set_properties.value
            and mutation.get("arguments", {}).get("properties")
            == _DD_RESIDUE_RELEASE_PROPERTIES
            for mutation in mutations
        )
    ):
        return

    source_nodes = [
        participant
        for participant in participants
        if participant["kind"] == RepairParticipantKind.node.value
        and participant["graph_label"] == "StandardNameSource"
    ]
    target_nodes = [
        participant
        for participant in participants
        if participant["kind"] == RepairParticipantKind.node.value
        and participant["graph_label"] == "StandardName"
    ]
    bindings = [
        participant
        for participant in participants
        if participant["kind"] == RepairParticipantKind.relationship.value
        and participant["graph_label"] == "PRODUCED_NAME"
    ]
    set_properties = [
        mutation
        for mutation in mutations
        if mutation["kind"] == RepairMutationKind.set_properties.value
    ]
    other_mutations = [
        mutation
        for mutation in mutations
        if mutation["kind"]
        not in {
            RepairMutationKind.delete_relationship.value,
            RepairMutationKind.set_properties.value,
        }
    ]
    survivor_id = identity.get("target_id")
    source_id = identity.get("source_id")
    expected_properties = {"produced_sn_id": survivor_id}
    deleted_participants = {
        str(mutation["participant_id"]) for mutation in relationship_deletes
    }
    binding_ids = {str(participant["id"]) for participant in bindings}

    valid = (
        identity.get("kind") == "source"
        and isinstance(source_id, str)
        and bool(source_id)
        and isinstance(survivor_id, str)
        and bool(survivor_id)
        and len(source_nodes) == 1
        and str(source_nodes[0]["id"]) == source_id
        and len(target_nodes) >= 2
        and survivor_id in {str(participant["id"]) for participant in target_nodes}
        and len(bindings) == len(target_nodes)
        and len(relationship_deletes) == len(bindings) - 1
        and deleted_participants < binding_ids
        and len(set_properties) == 1
        and str(set_properties[0]["participant_id"]) == source_id
        and set_properties[0].get("arguments", {}).get("properties")
        == expected_properties
        and len(mutations) == len(bindings)
        and not other_mutations
    )
    if not valid:
        raise SignedManifestAuthorityError(
            f"repair row {row_id!r} is not a closed source-target reconciliation program"
        )


def _validate_structural_source_revival_program(
    row_id: str,
    identity: dict[str, Any],
    participants: list[dict[str, Any]],
    mutations: list[dict[str, Any]],
) -> None:
    """Validate the sole closed program admitted for relationship creation."""
    relationship_adds = [
        mutation
        for mutation in mutations
        if mutation["kind"] == RepairMutationKind.add_relationship.value
    ]
    if not relationship_adds or any(
        mutation["kind"] == RepairMutationKind.delete_relationship.value
        for mutation in mutations
    ):
        return

    source_id = identity.get("source_id")
    target_id = identity.get("target_id")
    if source_id != f"derived:{target_id}":
        return

    source_nodes = [
        participant
        for participant in participants
        if participant["kind"] == RepairParticipantKind.node.value
        and participant["graph_label"] == "StandardNameSource"
    ]
    name_nodes = [
        participant
        for participant in participants
        if participant["kind"] == RepairParticipantKind.node.value
        and participant["graph_label"] == "StandardName"
    ]
    parent_relationships = [
        participant
        for participant in participants
        if participant["kind"] == RepairParticipantKind.relationship.value
        and participant["graph_label"] == "HAS_PARENT"
    ]
    property_updates = [
        mutation
        for mutation in mutations
        if mutation["kind"] == RepairMutationKind.set_properties.value
    ]
    expected_properties = {
        "status": "composed",
        "source_type": "derived",
        "source_id": target_id,
        "batch_key": "derived_parent",
        "produced_sn_id": target_id,
        "claimed_at": None,
        "claim_token": None,
    }
    expected_relationship = {
        "relationship_type": "PRODUCED_NAME",
        "start_id": source_id,
        "end_id": target_id,
    }
    valid = (
        identity.get("kind") == "source"
        and isinstance(source_id, str)
        and source_id == f"derived:{target_id}"
        and isinstance(target_id, str)
        and bool(target_id)
        and len(source_nodes) == 1
        and str(source_nodes[0]["id"]) == source_id
        and target_id in {str(participant["id"]) for participant in name_nodes}
        and len(name_nodes) >= 2
        and len(parent_relationships) >= 1
        and len(relationship_adds) == 1
        and str(relationship_adds[0]["participant_id"]) == target_id
        and relationship_adds[0].get("arguments") == expected_relationship
        and len(property_updates) == 1
        and str(property_updates[0]["participant_id"]) == source_id
        and property_updates[0].get("arguments", {}).get("properties")
        == expected_properties
        and len(mutations) == 2
    )
    if not valid:
        raise SignedManifestAuthorityError(
            f"repair row {row_id!r} is not a closed structural-source revival program"
        )


def _validate_unbound_source_attachment_program(
    row_id: str,
    identity: dict[str, Any],
    participants: list[dict[str, Any]],
    mutations: list[dict[str, Any]],
) -> None:
    """Validate one exact ordinary DD-source attachment program."""
    relationship_adds = [
        mutation
        for mutation in mutations
        if mutation["kind"] == RepairMutationKind.add_relationship.value
    ]
    if not relationship_adds or any(
        mutation["kind"] == RepairMutationKind.delete_relationship.value
        for mutation in mutations
    ):
        return

    source_id = identity.get("source_id")
    target_id = identity.get("target_id")
    if source_id == f"derived:{target_id}":
        return
    source_nodes = [
        participant
        for participant in participants
        if participant["kind"] == RepairParticipantKind.node.value
        and participant["graph_label"] == "StandardNameSource"
    ]
    name_nodes = [
        participant
        for participant in participants
        if participant["kind"] == RepairParticipantKind.node.value
        and participant["graph_label"] == "StandardName"
    ]
    relationship_participants = [
        participant
        for participant in participants
        if participant["kind"] == RepairParticipantKind.relationship.value
    ]
    property_updates = [
        mutation
        for mutation in mutations
        if mutation["kind"] == RepairMutationKind.set_properties.value
    ]
    expected_relationship = {
        "relationship_type": "PRODUCED_NAME",
        "start_id": source_id,
        "end_id": target_id,
    }
    expected_properties = {
        "status": "attached",
        "produced_sn_id": target_id,
        "claimed_at": None,
        "claim_token": None,
        "last_error": None,
    }
    valid = (
        identity.get("kind") == "source"
        and isinstance(source_id, str)
        and source_id.startswith("dd:")
        and len(source_id) > 3
        and isinstance(target_id, str)
        and bool(target_id)
        and len(source_nodes) == 1
        and str(source_nodes[0]["id"]) == source_id
        and len(name_nodes) == 1
        and str(name_nodes[0]["id"]) == target_id
        and not relationship_participants
        and len(relationship_adds) == 1
        and str(relationship_adds[0]["participant_id"]) == target_id
        and relationship_adds[0].get("arguments") == expected_relationship
        and len(property_updates) == 1
        and str(property_updates[0]["participant_id"]) == source_id
        and property_updates[0].get("arguments", {}).get("properties")
        == expected_properties
        and len(mutations) == 2
        and [int(mutation["order"]) for mutation in mutations] == [1, 2]
    )
    if not valid:
        raise SignedManifestAuthorityError(
            f"repair row {row_id!r} is not a closed unbound ordinary-source attachment program"
        )


def _validate_ordinary_source_migration_program(
    row_id: str,
    identity: dict[str, Any],
    participants: list[dict[str, Any]],
    mutations: list[dict[str, Any]],
) -> None:
    """Validate one exact edge, scalar, and backing-projection retarget."""
    relationship_deletes = [
        mutation
        for mutation in mutations
        if mutation["kind"] == RepairMutationKind.delete_relationship.value
    ]
    relationship_adds = [
        mutation
        for mutation in mutations
        if mutation["kind"] == RepairMutationKind.add_relationship.value
    ]
    if not relationship_deletes or not relationship_adds:
        return

    source_id = identity.get("source_id")
    target_id = identity.get("target_id")
    source_nodes = [
        participant
        for participant in participants
        if participant["kind"] == RepairParticipantKind.node.value
        and participant["graph_label"] == "StandardNameSource"
    ]
    name_nodes = [
        participant
        for participant in participants
        if participant["kind"] == RepairParticipantKind.node.value
        and participant["graph_label"] == "StandardName"
    ]
    bindings = [
        participant
        for participant in participants
        if participant["kind"] == RepairParticipantKind.relationship.value
        and participant["graph_label"] == "PRODUCED_NAME"
    ]
    property_updates = [
        mutation
        for mutation in mutations
        if mutation["kind"] == RepairMutationKind.set_properties.value
    ]
    expected_relationship = {
        "relationship_type": "PRODUCED_NAME",
        "start_id": source_id,
        "end_id": target_id,
    }
    valid = (
        identity.get("kind") == "source"
        and isinstance(source_id, str)
        and bool(source_id)
        and isinstance(target_id, str)
        and bool(target_id)
        and len(source_nodes) == 1
        and str(source_nodes[0]["id"]) == source_id
        and len(name_nodes) == 2
        and target_id in {str(participant["id"]) for participant in name_nodes}
        and len(bindings) == 1
        and len(relationship_deletes) == 1
        and str(relationship_deletes[0]["participant_id"]) == str(bindings[0]["id"])
        and len(relationship_adds) == 1
        and str(relationship_adds[0]["participant_id"]) == target_id
        and relationship_adds[0].get("arguments") == expected_relationship
        and len(property_updates) == 1
        and str(property_updates[0]["participant_id"]) == source_id
        and property_updates[0].get("arguments", {}).get("properties")
        == {"produced_sn_id": target_id}
        and len(mutations) == 3
        and [int(mutation["order"]) for mutation in mutations] == [1, 2, 3]
    )
    if not valid:
        raise SignedManifestAuthorityError(
            f"repair row {row_id!r} is not a closed ordinary-source migration program"
        )


def _validate_structural_reparent_program(
    row_id: str,
    identity: dict[str, Any],
    participants: list[dict[str, Any]],
    mutations: list[dict[str, Any]],
) -> None:
    """Validate one exact child-edge relocation between existing parents."""
    reparent_mutations = [
        mutation for mutation in mutations if mutation["kind"] == _STRUCTURAL_REPARENT
    ]
    if not reparent_mutations:
        return
    if all(
        (mutation.get("arguments") or {}).get("new_end_id") is None
        for mutation in reparent_mutations
    ):
        return

    name_nodes = [
        participant
        for participant in participants
        if participant["kind"] == RepairParticipantKind.node.value
        and participant["graph_label"] == "StandardName"
    ]
    parent_relationships = [
        participant
        for participant in participants
        if participant["kind"] == RepairParticipantKind.relationship.value
        and participant["graph_label"] == "HAS_PARENT"
    ]
    mutation = reparent_mutations[0] if len(reparent_mutations) == 1 else {}
    arguments = mutation.get("arguments") or {}
    child_id = str(identity.get("id") or "")
    old_parent_id = arguments.get("old_end_id")
    new_parent_id = arguments.get("new_end_id")
    relationship_properties = arguments.get("properties")
    node_ids = {str(participant["id"]) for participant in name_nodes}
    valid = (
        identity.get("kind") == "standard_name"
        and child_id
        and identity.get("target_id") == new_parent_id
        and len(name_nodes) == 3
        and len(node_ids) == 3
        and child_id in node_ids
        and isinstance(old_parent_id, str)
        and bool(old_parent_id)
        and old_parent_id in node_ids
        and isinstance(new_parent_id, str)
        and bool(new_parent_id)
        and new_parent_id in node_ids
        and old_parent_id != new_parent_id
        and len(parent_relationships) == 1
        and len(reparent_mutations) == 1
        and len(mutations) == 1
        and int(mutation.get("order", -1)) == 0
        and str(mutation.get("participant_id")) == str(parent_relationships[0]["id"])
        and set(arguments)
        == {
            "relationship_type",
            "start_id",
            "old_end_id",
            "new_end_id",
            "properties",
        }
        and arguments.get("relationship_type") == "HAS_PARENT"
        and arguments.get("start_id") == child_id
        and isinstance(relationship_properties, dict)
    )
    if not valid:
        raise SignedManifestAuthorityError(
            f"repair row {row_id!r} is not a closed structural reparent program"
        )


def _validate_structural_release_program(
    row_id: str,
    identity: dict[str, Any],
    participants: list[dict[str, Any]],
    mutations: list[dict[str, Any]],
) -> None:
    """Validate one exact child-edge release without a replacement parent."""
    release_mutations = [
        mutation for mutation in mutations if mutation["kind"] == _STRUCTURAL_REPARENT
    ]
    if not release_mutations:
        return
    if any(
        (mutation.get("arguments") or {}).get("new_end_id") is not None
        for mutation in release_mutations
    ):
        return

    name_nodes = [
        participant
        for participant in participants
        if participant["kind"] == RepairParticipantKind.node.value
        and participant["graph_label"] == "StandardName"
    ]
    parent_relationships = [
        participant
        for participant in participants
        if participant["kind"] == RepairParticipantKind.relationship.value
        and participant["graph_label"] == "HAS_PARENT"
    ]
    mutation = release_mutations[0] if len(release_mutations) == 1 else {}
    arguments = mutation.get("arguments") or {}
    child_id = str(identity.get("id") or "")
    old_parent_id = arguments.get("old_end_id")
    relationship_properties = arguments.get("properties")
    node_ids = {str(participant["id"]) for participant in name_nodes}
    valid = (
        identity.get("kind") == "standard_name"
        and child_id
        and identity.get("target_id") == child_id
        and len(name_nodes) == 2
        and len(node_ids) == 2
        and child_id in node_ids
        and isinstance(old_parent_id, str)
        and bool(old_parent_id)
        and old_parent_id in node_ids
        and len(parent_relationships) == 1
        and len(release_mutations) == 1
        and len(mutations) == 1
        and int(mutation.get("order", -1)) == 0
        and str(mutation.get("participant_id")) == str(parent_relationships[0]["id"])
        and set(arguments)
        == {
            "relationship_type",
            "start_id",
            "old_end_id",
            "new_end_id",
            "properties",
        }
        and arguments.get("relationship_type") == "HAS_PARENT"
        and arguments.get("start_id") == child_id
        and arguments.get("new_end_id") is None
        and isinstance(relationship_properties, dict)
    )
    if not valid:
        raise SignedManifestAuthorityError(
            f"repair row {row_id!r} is not a closed structural release program"
        )


def _validate_supersede_successor_program(
    row_id: str,
    identity: dict[str, Any],
    participants: list[dict[str, Any]],
    mutations: list[dict[str, Any]],
) -> None:
    """Validate an explicitly signed successor for a supersede mutation."""
    supersedes = [
        mutation
        for mutation in mutations
        if mutation["kind"] == RepairMutationKind.supersede.value
    ]
    signed_supersedes = [
        mutation for mutation in supersedes if mutation.get("arguments")
    ]
    if not signed_supersedes:
        return

    mutation = signed_supersedes[0] if len(signed_supersedes) == 1 else {}
    arguments = mutation.get("arguments") or {}
    predecessor_id = str(mutation.get("participant_id") or "")
    successor_id = arguments.get("successor_id")
    name_ids = {
        str(participant["id"])
        for participant in participants
        if participant["kind"] == RepairParticipantKind.node.value
        and participant["graph_label"] == "StandardName"
    }
    valid = (
        identity.get("kind") == "standard_name"
        and identity.get("target_id") == predecessor_id
        and len(supersedes) == 1
        and len(signed_supersedes) == 1
        and len(mutations) == 1
        and set(arguments) == {"successor_id"}
        and isinstance(successor_id, str)
        and bool(successor_id)
        and successor_id != predecessor_id
        and predecessor_id in name_ids
        and successor_id in name_ids
    )
    if not valid:
        raise SignedManifestAuthorityError(
            f"repair row {row_id!r} is not a closed signed-successor supersede program"
        )


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        default=str,
    ).encode()


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def signed_payload_sha256(authority: dict[str, Any]) -> str:
    """Return the canonical digest covered by the authority signature."""
    payload = {key: value for key, value in authority.items() if key != "signature"}
    return _digest(payload)


def _require_sha256(value: str, role: str) -> None:
    if _SHA256_RE.fullmatch(value) is None:
        raise SignedManifestAuthorityError(f"{role} must be a lowercase SHA-256 digest")


def _validate_model(model: type[Any], value: dict[str, Any], role: str) -> None:
    try:
        model.model_validate(value)
    except ValidationError as exc:
        raise SignedManifestAuthorityError(f"invalid {role}: {exc}") from exc


AUTHORITY_ARTIFACT_SCHEMA_WIRE_KEY = "schema"
AUTHORITY_ARTIFACT_SCHEMA_FIELD = "schema_id"


def authority_artifact_wire_projection(data: Mapping[str, Any]) -> dict[str, Any]:
    """Adapt the ``schema`` wire key to the ``RepairAuthorityArtifact.schema_id`` field.

    Committed authority bytes and every payload/file digest are always
    computed against the original ``schema`` key; only this
    validation-time projection is renamed, because the pydantic field is
    named ``schema_id`` so it does not shadow ``ConfiguredBaseModel.schema``.
    """
    projection = dict(data)
    if AUTHORITY_ARTIFACT_SCHEMA_WIRE_KEY in projection:
        projection[AUTHORITY_ARTIFACT_SCHEMA_FIELD] = projection.pop(
            AUTHORITY_ARTIFACT_SCHEMA_WIRE_KEY
        )
    return projection


def _load_authority(
    path: str | Path,
    *,
    expected_file_sha256: str,
    expected_payload_sha256: str,
) -> _Authority:
    _require_sha256(expected_file_sha256, "authority_file_sha256")
    _require_sha256(expected_payload_sha256, "authority_payload_sha256")
    raw = Path(path).read_bytes()
    file_sha256 = hashlib.sha256(raw).hexdigest()
    if file_sha256 != expected_file_sha256:
        raise SignedManifestAuthorityError("authority file SHA-256 mismatch")
    try:
        data = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SignedManifestAuthorityError(
            "authority file is not canonical JSON"
        ) from exc
    if not isinstance(data, dict):
        raise SignedManifestAuthorityError("authority root must be an object")
    payload_sha256 = signed_payload_sha256(data)
    if payload_sha256 != expected_payload_sha256:
        raise SignedManifestAuthorityError("canonical signed-payload SHA-256 mismatch")
    signature = data.get("signature")
    if (
        not isinstance(signature, dict)
        or signature.get("canonicalization") != SIGNED_AUTHORITY_CANONICALIZATION
        or signature.get("sha256") != payload_sha256
    ):
        raise SignedManifestAuthorityError(
            "authority signature does not match canonical signed payload"
        )

    raw_rows = data.get("rows")
    if not isinstance(raw_rows, list) or not raw_rows:
        raise SignedManifestAuthorityError("authority rows must be a non-empty array")
    selection = data.get("selection")
    if not isinstance(selection, dict):
        raise SignedManifestAuthorityError("authority selection must be an object")
    _validate_model(RepairSelection, selection, "authority selection")
    if selection.get("predicate") != "artifact-rows":
        raise SignedManifestAuthorityError(
            "authority selection predicate must be 'artifact-rows'"
        )

    receipt_policy = data.get("receipt_policy")
    if not isinstance(receipt_policy, dict):
        raise SignedManifestAuthorityError("receipt_policy must be an object")
    _validate_model(RepairReceiptPolicy, receipt_policy, "receipt policy")
    if receipt_policy.get("expected_count") != "admitted_rows":
        raise SignedManifestAuthorityError(
            "receipt_policy expected_count must be 'admitted_rows'"
        )

    digest_rows = data.get("authority_digests") or []
    if not isinstance(digest_rows, list):
        raise SignedManifestAuthorityError("authority_digests must be an array")
    for digest_row in digest_rows:
        if not isinstance(digest_row, dict):
            raise SignedManifestAuthorityError("authority digest must be an object")
        _validate_model(RepairAuthorityDigest, digest_row, "authority digest")

    loaded_rows: list[_LoadedRow] = []
    seen_row_ids: set[str] = set()
    mutated_participants: set[str] = set()
    for raw_row in raw_rows:
        if not isinstance(raw_row, dict):
            raise SignedManifestAuthorityError("repair row must be an object")
        identity = raw_row.get("identity")
        participants = raw_row.get("participants")
        mutations = raw_row.get("mutations")
        guards = raw_row.get("guards")
        row_selection = raw_row.get("selection")
        if not isinstance(identity, dict):
            raise SignedManifestAuthorityError("repair row identity must be an object")
        if not isinstance(participants, list) or not participants:
            raise SignedManifestAuthorityError(
                "repair row participants must be a non-empty array"
            )
        if not isinstance(mutations, list) or not mutations:
            raise SignedManifestAuthorityError(
                "repair row mutations must be a non-empty array"
            )
        if not isinstance(guards, list) or not guards:
            raise SignedManifestAuthorityError(
                "repair row guards must be a non-empty array"
            )
        if not isinstance(row_selection, dict):
            raise SignedManifestAuthorityError("repair row selection must be an object")

        _validate_model(RepairRowIdentity, identity, "repair row identity")
        _validate_model(RepairSelection, row_selection, "repair row selection")
        if row_selection != selection:
            raise SignedManifestAuthorityError(
                "repair row selection must equal the artifact selection"
            )
        for participant in participants:
            if not isinstance(participant, dict):
                raise SignedManifestAuthorityError(
                    "repair participant must be an object"
                )
            _validate_model(RepairParticipant, participant, "repair participant")
            label = str(participant["graph_label"])
            kind = str(participant["kind"])
            if kind == RepairParticipantKind.node.value and label not in _NODE_LABELS:
                raise SignedManifestAuthorityError(
                    f"unsupported repair participant node label: {label}"
                )
            if (
                kind == RepairParticipantKind.relationship.value
                and label not in _RELATIONSHIP_TYPES
            ):
                raise SignedManifestAuthorityError(
                    f"unsupported repair participant relationship type: {label}"
                )
        for mutation in mutations:
            if not isinstance(mutation, dict):
                raise SignedManifestAuthorityError("repair mutation must be an object")
            _validate_model(RepairMutation, mutation, "repair mutation")
            kind = str(mutation["kind"])
            if kind not in _SIGNED_MANIFEST_MUTATION_KINDS:
                raise SignedManifestAuthorityError(
                    f"unsupported repair mutation kind: {kind}"
                )
            participant_id = str(mutation["participant_id"])
            if participant_id in mutated_participants:
                raise SignedManifestAuthorityError(
                    "repair rows target the same mutation participant"
                )
            mutated_participants.add(participant_id)
        for guard in guards:
            if not isinstance(guard, dict):
                raise SignedManifestAuthorityError("repair guard must be an object")
            _validate_model(RepairGuard, guard, "repair guard")
            implementation = str(guard["implementation"])
            expected_kind = _GUARD_KINDS.get(implementation)
            if expected_kind is None:
                raise SignedManifestAuthorityError(
                    f"unsupported repair guard implementation: {implementation}"
                )
            if str(guard["kind"]) != expected_kind:
                raise SignedManifestAuthorityError(
                    f"repair guard kind does not match implementation: {implementation}"
                )

        row_id = str(raw_row.get("id", ""))
        if not row_id or row_id in seen_row_ids:
            raise SignedManifestAuthorityError(
                "repair row ids must be unique and non-empty"
            )
        seen_row_ids.add(row_id)
        participant_ids = {str(item["id"]) for item in participants}
        if len(participant_ids) != len(participants):
            raise SignedManifestAuthorityError(
                f"repair row {row_id!r} has duplicate participant ids"
            )
        if any(
            str(mutation["participant_id"]) not in participant_ids
            for mutation in mutations
        ):
            raise SignedManifestAuthorityError(
                f"repair row {row_id!r} mutates an undeclared participant"
            )
        mutation_kinds = {str(item["kind"]) for item in mutations}
        guard_names = {str(item["implementation"]) for item in guards}
        required_guards = {_COLLATERAL_IMMUTABILITY}
        if mutation_kinds & {
            RepairMutationKind.detach.value,
            RepairMutationKind.delete_relationship.value,
        }:
            required_guards.add(_LAST_PRODUCER)
        if mutation_kinds & {
            RepairMutationKind.delete.value,
            RepairMutationKind.supersede.value,
        }:
            required_guards.add(_STRUCTURAL_LEGITIMACY)
        missing_guards = sorted(required_guards - guard_names)
        if missing_guards:
            raise SignedManifestAuthorityError(
                f"repair row {row_id!r} is missing guards: {', '.join(missing_guards)}"
            )
        _validate_source_target_reconciliation_program(
            row_id, identity, participants, mutations
        )
        _validate_structural_source_revival_program(
            row_id, identity, participants, mutations
        )
        _validate_unbound_source_attachment_program(
            row_id, identity, participants, mutations
        )
        _validate_ordinary_source_migration_program(
            row_id, identity, participants, mutations
        )
        _validate_structural_reparent_program(row_id, identity, participants, mutations)
        _validate_structural_release_program(row_id, identity, participants, mutations)
        _validate_supersede_successor_program(row_id, identity, participants, mutations)

        projection = {
            **raw_row,
            "identity": str(identity["id"]),
            "participants": [str(item["id"]) for item in participants],
            "selection": str(row_selection["id"]),
            "mutations": [str(item["id"]) for item in mutations],
            "guards": [str(item["id"]) for item in guards],
        }
        _validate_model(RepairAuthorityRow, projection, "repair authority row")
        loaded_rows.append(
            _LoadedRow(
                id=row_id,
                identity=dict(identity),
                participants=tuple(dict(item) for item in participants),
                mutations=tuple(
                    sorted(
                        (dict(item) for item in mutations),
                        key=lambda item: (int(item["order"]), str(item["id"])),
                    )
                ),
                guards=tuple(dict(item) for item in guards),
                orphan_policy=str(raw_row["orphan_policy"]),
            )
        )

    structural_reparent_rows = [
        row for row in loaded_rows if _is_structural_reparent(row)
    ]
    structural_release_rows = [
        row for row in loaded_rows if _is_structural_release(row)
    ]
    structural_rows = structural_reparent_rows + structural_release_rows
    if structural_rows and (
        len(structural_rows) != len(loaded_rows)
        or bool(structural_reparent_rows) == bool(structural_release_rows)
    ):
        raise SignedManifestAuthorityError(
            "structural authority cannot mix reparent and release programs"
        )

    repair_rows = data.get("repair_rows")
    if repair_rows is not None and sorted(repair_rows) != sorted(seen_row_ids):
        raise SignedManifestAuthorityError(
            "repair_rows projection does not match authority rows"
        )
    operation_id = data.get("operation_id")
    if not isinstance(operation_id, str) or not operation_id.strip():
        raise SignedManifestAuthorityError("operation_id must be non-empty")
    _validate_dd_residue_release_authority(operation_id, loaded_rows, receipt_policy)
    artifact_projection = {
        **data,
        "authority_digests": [str(item["id"]) for item in digest_rows] or None,
        "selection": str(selection["id"]),
        "repair_rows": sorted(seen_row_ids),
        "receipt_policy": str(receipt_policy["id"]),
    }
    _validate_model(
        RepairAuthorityArtifact,
        authority_artifact_wire_projection(artifact_projection),
        "repair authority",
    )
    return _Authority(
        data=data,
        operation_id=operation_id,
        rows=tuple(sorted(loaded_rows, key=lambda row: row.id)),
        receipt_policy=dict(receipt_policy),
        file_sha256=file_sha256,
        payload_sha256=payload_sha256,
    )


def _load_refused_target_orphan_authority(
    source: str | Path | dict[str, Any],
    *,
    expected_sha256: str,
    mutation_kind: str | None,
    guard_set: tuple[str, ...] | None,
) -> _Authority:
    """Adapt the committed orphan adjudication without changing its bytes."""
    _require_sha256(expected_sha256, "authority_sha256")
    if isinstance(source, dict):
        data = source
    else:
        try:
            data = json.loads(Path(source).read_bytes())
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SignedManifestAuthorityError(
                "orphan authority file is not valid JSON"
            ) from exc
    if not isinstance(data, dict) or _digest(data) != expected_sha256:
        raise SignedManifestAuthorityError(
            "orphan retirement authority signature does not match"
        )
    if data.get("schema") != _REFUSED_TARGET_ORPHAN_SCHEMA:
        raise SignedManifestAuthorityError(
            "unsupported orphan retirement authority schema"
        )
    if data.get("read_only") is not True:
        raise SignedManifestAuthorityError(
            "orphan retirement authority must be read-only evidence"
        )
    raw_rows = data.get("rows")
    if not isinstance(raw_rows, list) or not raw_rows:
        raise SignedManifestAuthorityError(
            "orphan retirement authority requires target rows"
        )
    if any(
        not isinstance(row, dict)
        or row.get("disposition") not in _REFUSED_TARGET_ORPHAN_DISPOSITIONS
        for row in raw_rows
    ):
        raise SignedManifestAuthorityError(
            "orphan retirement authority has an unknown disposition"
        )
    names = [row.get("name") for row in raw_rows]
    if any(not isinstance(name, str) or not name for name in names):
        raise SignedManifestAuthorityError(
            "every orphan disposition requires an exact name"
        )
    if len(names) != len(set(names)):
        raise SignedManifestAuthorityError("orphan disposition names must be unique")
    disposition_counts = {
        disposition: sum(row["disposition"] == disposition for row in raw_rows)
        for disposition in sorted(_REFUSED_TARGET_ORPHAN_DISPOSITIONS)
    }
    summary = data.get("summary")
    if (
        not isinstance(summary, dict)
        or summary.get("targets") != len(raw_rows)
        or summary.get("disposition_sum") != len(raw_rows)
        or summary.get("disposition_counts") != disposition_counts
    ):
        raise SignedManifestAuthorityError(
            "orphan retirement authority summary does not match rows"
        )

    expected_guards = (
        _SIGNED_LIFECYCLE,
        _NO_LIVE_PRODUCER,
        _NO_LIVE_STRUCTURAL_CHILD,
        _COLLATERAL_IMMUTABILITY,
    )
    if mutation_kind != RepairMutationKind.supersede.value:
        raise SignedManifestAuthorityError(
            "orphan retirement requires the supersede mutation kind"
        )
    if guard_set != expected_guards:
        raise SignedManifestAuthorityError(
            "orphan retirement requires its exact signed guard set"
        )

    loaded_rows: list[_LoadedRow] = []
    for raw_row in raw_rows:
        if raw_row["disposition"] != _REFUSED_TARGET_ORPHAN_DISPOSITION:
            continue
        if raw_row.get("mutation_authority") != "classification_only":
            raise SignedManifestAuthorityError(
                "signed orphan evidence cannot directly grant source mutation"
            )
        name_stage = raw_row.get("name_stage")
        if name_stage not in {"accepted", "reviewed", "drafted", "pending"}:
            raise SignedManifestAuthorityError(
                "signed orphan target requires a live lifecycle stage"
            )
        closure = raw_row.get("structural_closure")
        if (
            not isinstance(closure, dict)
            or closure.get("classification") != "no_live_structural_descendant"
            or closure.get("has_live_has_parent_child") is not False
            or closure.get("has_live_refined_from_descendant") is not False
            or closure.get("live_has_parent_children") != []
            or closure.get("live_refined_from_descendants") != []
        ):
            raise SignedManifestAuthorityError(
                "signed orphan retirement requires an empty structural closure"
            )
        removed_bindings = raw_row.get("current_removed_bindings")
        if not isinstance(removed_bindings, list) or not removed_bindings:
            raise SignedManifestAuthorityError(
                "signed orphan retirement requires name-specific binding evidence"
            )
        name_id = str(raw_row["name"])
        participant = {
            "id": name_id,
            "kind": RepairParticipantKind.node.value,
            "graph_label": "StandardName",
            "expected_name_stage": name_stage,
            "authority_row_sha256": _digest(raw_row),
        }
        mutation = {
            "id": f"{name_id}:supersede",
            "order": 0,
            "kind": RepairMutationKind.supersede.value,
            "participant_id": name_id,
            "preserve_source_paths": True,
        }
        guards = tuple(
            {
                "id": implementation,
                "kind": _GUARD_KINDS[implementation],
                "implementation": implementation,
                "participant_ids": [name_id],
            }
            for implementation in expected_guards
        )
        loaded_rows.append(
            _LoadedRow(
                id=name_id,
                identity={
                    "id": name_id,
                    "kind": "standard_name",
                    "target_id": name_id,
                },
                participants=(participant,),
                mutations=(mutation,),
                guards=guards,
                orphan_policy="refuse",
            )
        )
    loaded_rows.sort(key=lambda row: row.id)
    if not loaded_rows:
        raise SignedManifestAuthorityError(
            "orphan authority contains no retirement dispositions"
        )
    if summary.get("remaining_retirements") != len(loaded_rows) or summary.get(
        "retirements_with_name_specific_evidence"
    ) != len(loaded_rows):
        raise SignedManifestAuthorityError(
            "orphan retirement summary does not match signed targets"
        )
    return _Authority(
        data={
            **data,
            "adapter": _REFUSED_TARGET_ORPHAN_ADAPTER,
            "all_or_nothing": True,
        },
        operation_id="signed-provenance-orphan-retirement",
        rows=tuple(loaded_rows),
        receipt_policy={
            "operation": "retire_signed_provenance_orphan",
            "expected_count": "admitted_rows",
        },
        file_sha256=expected_sha256,
        payload_sha256=expected_sha256,
    )


def _graph_json_value(value: Any) -> Any:
    """Convert Neo4j values to a stable JSON representation."""
    if isinstance(value, dict):
        return {
            str(key): _graph_json_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, list | tuple):
        return [_graph_json_value(item) for item in value]
    if hasattr(value, "iso_format"):
        return value.iso_format()
    if hasattr(value, "isoformat") and not isinstance(value, str):
        return value.isoformat()
    if value is None or isinstance(value, str | int | float | bool):
        return value
    return str(value)


def _graph_payload_hash(value: Any) -> str:
    payload = json.dumps(
        _graph_json_value(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _load_signed_stale_source_rows(
    authority_path: str | Path,
    source_ids: Sequence[str],
) -> tuple[str, str, list[dict[str, Any]]]:
    path = Path(authority_path).expanduser().resolve()
    raw = path.read_bytes()
    try:
        authority = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("stale-source lifecycle authority is not valid JSON") from exc
    if authority.get("schema") != _STALE_SOURCE_LIFECYCLE_SCHEMA:
        raise ValueError("unsupported stale-source lifecycle authority schema")
    signature = authority.get("signature")
    if not isinstance(signature, dict) or signature != {
        "algorithm": "sha256",
        "canonicalization": "jq -cS '.rows'",
        "scope": "rows",
        "digest": signature.get("digest") if isinstance(signature, dict) else None,
    }:
        raise ValueError("stale-source lifecycle signature contract is unsupported")
    declared_digest = signature.get("digest")
    if not isinstance(declared_digest, str) or len(declared_digest) != 64:
        raise ValueError("stale-source lifecycle signature requires a SHA-256 digest")
    rows = authority.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("stale-source lifecycle authority requires rows")
    canonical_rows = (
        json.dumps(rows, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    )
    if hashlib.sha256(canonical_rows.encode()).hexdigest() != declared_digest:
        raise ValueError("stale-source lifecycle rows signature does not match")

    requested = sorted(set(source_ids))
    if not requested or len(requested) != len(source_ids):
        raise ValueError("stale-source detach requires unique non-empty source ids")
    by_source = {row.get("source_id"): row for row in rows if isinstance(row, dict)}
    if any(source_id not in by_source for source_id in requested):
        raise ValueError("stale-source detach source is outside signed authority")
    selected = [dict(by_source[source_id]) for source_id in requested]
    for row in selected:
        source_id = row.get("source_id")
        source_type = row.get("source_type")
        live_target_ids = row.get("live_target_ids")
        scalar_target = row.get("scalar_target")
        source_shape_is_signed = (
            isinstance(source_id, str)
            and source_type in {"dd", "derived"}
            and isinstance(live_target_ids, list)
            and bool(live_target_ids)
            and all(isinstance(target_id, str) for target_id in live_target_ids)
            and len(set(live_target_ids)) == len(live_target_ids)
            and isinstance(scalar_target, str)
            and bool(scalar_target)
        )
        dd_shape_is_signed = source_type == "dd" and (
            source_id.startswith("dd:")
            and isinstance(row.get("source_dd_version"), str)
            and row.get("backing_lifecycle_status") == "removed"
        )
        derived_shape_is_signed = source_type == "derived" and (
            source_id.startswith("derived:")
            and row.get("source_dd_version") is None
            and row.get("backing_lifecycle_status") is None
        )
        if (
            not source_shape_is_signed
            or not (dd_shape_is_signed or derived_shape_is_signed)
            or row.get("disposition") != "detach"
            or row.get("configured_path_present") is not False
        ):
            raise ValueError("selected stale-source row lacks exact detach authority")
    return hashlib.sha256(raw).hexdigest(), declared_digest, selected


def _signed_stale_source_target_ids(row: dict[str, Any]) -> list[str]:
    """Return every target identity whose removal shape is signed by a row."""
    return sorted({*row["live_target_ids"], row["scalar_target"]})


def _stale_source_detach_closure(gc: Any, rows: list[dict[str, Any]]) -> dict[str, Any]:
    source_ids = [row["source_id"] for row in rows]
    target_ids = sorted(
        {target for row in rows for target in _signed_stale_source_target_ids(row)}
    )
    participants = [
        dict(row)
        for row in gc.query(
            """
            // SIGNED_STALE_SOURCE_DETACH_CLOSURE
            UNWIND $source_ids AS requested_id
            OPTIONAL MATCH (source:StandardNameSource {id: requested_id})
            RETURN requested_id,
                   elementId(source) AS source_element_id,
                   properties(source) AS source_properties,
                   CASE WHEN source IS NULL THEN [] ELSE
                     [(source)-[binding:PRODUCED_NAME]->(target:StandardName) |
                       {element_id: elementId(binding), properties: properties(binding),
                        target_element_id: elementId(target), target_id: target.id,
                        target_properties: properties(target)}]
                   END AS bindings,
                   CASE WHEN source IS NULL THEN [] ELSE
                     [(source)-[origin:FROM_DD_PATH]->(backing:IMASNode) |
                       {element_id: elementId(backing), properties: properties(backing),
                        origin_element_id: elementId(origin),
                        origin_properties: properties(origin),
                        projections: [(backing)-[projection:HAS_STANDARD_NAME]->
                          (target:StandardName) |
                          {element_id: elementId(projection),
                           properties: properties(projection),
                           target_element_id: elementId(target),
                           target_id: target.id}]}]
                   END AS backings
            ORDER BY requested_id
            """,
            source_ids=source_ids,
        )
    ]
    for row in participants:
        row["bindings"] = sorted(
            (dict(item) for item in row.get("bindings") or []),
            key=lambda item: (item["target_id"], item["element_id"]),
        )
        row["backings"] = sorted(
            (
                {
                    **dict(item),
                    "projections": sorted(
                        (
                            dict(projection)
                            for projection in item.get("projections") or []
                        ),
                        key=lambda projection: (
                            projection["target_id"],
                            projection["element_id"],
                        ),
                    ),
                }
                for item in row.get("backings") or []
            ),
            key=lambda item: item["element_id"],
        )
    target_closures = [
        dict(row)
        for row in gc.query(
            """
            UNWIND $target_ids AS requested_id
            OPTIONAL MATCH (target:StandardName {id: requested_id})
            RETURN requested_id,
                   elementId(target) AS target_element_id,
                   properties(target) AS target_properties,
                   CASE WHEN target IS NULL THEN [] ELSE
                     [(source:StandardNameSource)-[binding:PRODUCED_NAME]->(target) |
                       {source_element_id: elementId(source),
                        source_properties: properties(source),
                        binding_element_id: elementId(binding),
                        binding_properties: properties(binding)}]
                   END AS incoming_bindings
                   ,CASE WHEN target IS NULL THEN [] ELSE
                     [(child:StandardName)-[parent:HAS_PARENT]->(target)
                       WHERE coalesce(child.name_stage, '') <> 'superseded'
                         AND coalesce(child.status, '') <> 'superseded' |
                       {child_element_id: elementId(child),
                        child_properties: properties(child),
                        parent_element_id: elementId(parent),
                        parent_properties: properties(parent)}]
                   END AS live_children
            ORDER BY requested_id
            """,
            target_ids=target_ids,
        )
    ]
    for row in target_closures:
        row["incoming_bindings"] = sorted(
            (dict(item) for item in row.get("incoming_bindings") or []),
            key=lambda item: (
                (item.get("source_properties") or {}).get("id", ""),
                item["binding_element_id"],
            ),
        )
        row["live_children"] = sorted(
            (dict(item) for item in row.get("live_children") or []),
            key=lambda item: (
                (item.get("child_properties") or {}).get("id", ""),
                item["parent_element_id"],
            ),
        )
    versions = list(
        gc.query(
            """
            MATCH (version:DDVersion)
            WHERE version.is_current = true
            RETURN elementId(version) AS element_id,
                   properties(version) AS properties
            ORDER BY version.id
            """
        )
    )
    return {
        "participants": participants,
        "targets": target_closures,
        "current_versions": [dict(row) for row in versions],
    }


def _validate_stale_source_detach_closure(
    signed_rows: list[dict[str, Any]], closure: dict[str, Any]
) -> list[dict[str, Any]]:
    expected = {row["source_id"]: row for row in signed_rows}
    selected_ids = set(expected)
    targets = {row["requested_id"]: row for row in closure["targets"]}
    versions = closure["current_versions"]
    if (
        len(versions) != 1
        or (versions[0].get("properties") or {}).get("id") != "4.1.1"
        or (versions[0].get("properties") or {}).get("is_current") is not True
    ):
        raise StaleSourceDetachConflict("configured current DD authority changed")
    actions: list[dict[str, Any]] = []
    for participant in closure["participants"]:
        signed = expected[participant["requested_id"]]
        properties = participant.get("source_properties") or {}
        bindings = participant.get("bindings") or []
        backings = participant.get("backings") or []
        signed_targets = sorted(signed["live_target_ids"])
        authorized_targets = _signed_stale_source_target_ids(signed)
        binding_targets = [binding["target_id"] for binding in bindings]
        projections = [
            projection for backing in backings for projection in backing["projections"]
        ]
        projection_targets = [projection["target_id"] for projection in projections]
        common_shape_changed = (
            participant.get("source_element_id") is None
            or properties.get("status") != "stale"
            or properties.get("source_type") != signed["source_type"]
            or properties.get("dd_version") != signed["source_dd_version"]
            or properties.get("produced_sn_id") != signed["scalar_target"]
            or properties.get("claimed_at") is not None
            or properties.get("claim_token") is not None
            or not set(signed_targets).issubset(binding_targets)
            or not set(binding_targets).issubset(authorized_targets)
            or len(binding_targets) != len(set(binding_targets))
        )
        dd_shape_changed = signed["source_type"] == "dd" and (
            len(backings) != 1
            or (backings[0].get("properties") or {}).get("id")
            != signed["source_id"][3:]
            or (backings[0].get("properties") or {}).get("lifecycle_status")
            != signed["backing_lifecycle_status"]
            or sorted(projection_targets) != sorted(binding_targets)
            or len(projection_targets) != len(set(projection_targets))
        )
        derived_shape_changed = signed["source_type"] == "derived" and bool(backings)
        if common_shape_changed or dd_shape_changed or derived_shape_changed:
            raise StaleSourceDetachConflict(
                f"signed source closure changed for {signed['source_id']}"
            )
        for target_id in binding_targets:
            target = targets.get(target_id) or {}
            live_remaining = [
                incoming
                for incoming in target.get("incoming_bindings") or []
                if (incoming.get("source_properties") or {}).get("status") != "stale"
                and (incoming.get("source_properties") or {}).get("id")
                not in selected_ids
            ]
            if not live_remaining and not target.get("live_children"):
                raise StaleSourceDetachConflict(
                    f"detach would orphan target {target_id}"
                )
        actions.append(
            {
                "source_id": signed["source_id"],
                "source_element_id": participant["source_element_id"],
                "target_ids": binding_targets,
                "target_element_ids": [
                    binding["target_element_id"] for binding in bindings
                ],
                "binding_element_ids": [binding["element_id"] for binding in bindings],
                "backing_element_ids": [backing["element_id"] for backing in backings],
                "projection_element_ids": [
                    projection["element_id"] for projection in projections
                ],
                "scalar_target": signed["scalar_target"],
                "unblocks": signed["unblocks"],
            }
        )
    if len(actions) != len(signed_rows):
        raise StaleSourceDetachConflict("signed source cohort is incomplete")
    return actions


def _out_of_allowlist_source_hash(gc: Any, source_ids: list[str]) -> tuple[int, str]:
    rows = [
        dict(row)
        for row in gc.query(
            """
            MATCH (source:StandardNameSource)
            WHERE NOT (source.id IN $source_ids)
            RETURN source.id AS source_id,
                   properties(source) AS source_properties,
                   [(source)-[binding:PRODUCED_NAME]->(target:StandardName) |
                     {element_id: elementId(binding), properties: properties(binding),
                      target_id: target.id}] AS bindings,
                   [(source)-[origin:FROM_DD_PATH|FROM_SIGNAL]->(backing) |
                     {element_id: elementId(backing), origin_type: type(origin),
                      origin_element_id: elementId(origin),
                      origin_properties: properties(origin),
                      projections: [(backing)-[projection:HAS_STANDARD_NAME]->
                        (target:StandardName) |
                        {element_id: elementId(projection),
                         properties: properties(projection), target_id: target.id}]}]
                     AS backings
            ORDER BY source.id
            """,
            source_ids=source_ids,
        )
    ]
    for row in rows:
        row["bindings"] = sorted(
            (dict(item) for item in row.get("bindings") or []),
            key=lambda item: (item["target_id"], item["element_id"]),
        )
        row["backings"] = sorted(
            (
                {
                    **dict(item),
                    "projections": sorted(
                        (
                            dict(projection)
                            for projection in item.get("projections") or []
                        ),
                        key=lambda projection: (
                            projection["target_id"],
                            projection["element_id"],
                        ),
                    ),
                }
                for item in row.get("backings") or []
            ),
            key=lambda item: (item["origin_type"], item["element_id"]),
        )
    return len(rows), _graph_payload_hash(rows)


def _load_stale_source_authority(
    source: str | Path,
    source_ids: Sequence[str],
    *,
    mutation_kind: str | None,
    guard_set: tuple[str, ...] | None,
) -> _Authority:
    if mutation_kind != RepairMutationKind.detach.value:
        raise SignedManifestAuthorityError(
            "stale-source repair requires the detach mutation kind"
        )
    if guard_set != _STALE_SOURCE_GUARDS:
        raise SignedManifestAuthorityError(
            "stale-source repair requires its exact signed guard set"
        )
    file_sha256, rows_sha256, signed_rows = _load_signed_stale_source_rows(
        source, source_ids
    )
    loaded_rows = tuple(
        _LoadedRow(
            id=str(row["source_id"]),
            identity={
                "id": str(row["source_id"]),
                "kind": "standard_name_source",
                "source_id": str(row["source_id"]),
                "target_id": str(row["scalar_target"]),
            },
            participants=(
                {
                    "id": str(row["source_id"]),
                    "kind": RepairParticipantKind.node.value,
                    "graph_label": "StandardNameSource",
                },
            ),
            mutations=(
                {
                    "id": f"{row['source_id']}:detach",
                    "order": 0,
                    "kind": RepairMutationKind.detach.value,
                    "participant_id": str(row["source_id"]),
                    "arguments": {"implementation": _STALE_SOURCE_ADAPTER},
                },
            ),
            guards=tuple(
                {
                    "id": implementation,
                    "kind": _GUARD_KINDS[implementation],
                    "implementation": implementation,
                    "participant_ids": [str(row["source_id"])],
                }
                for implementation in _STALE_SOURCE_GUARDS
            ),
            orphan_policy="refuse",
        )
        for row in signed_rows
    )
    return _Authority(
        data={
            "adapter": _STALE_SOURCE_ADAPTER,
            "all_or_nothing": True,
            "signed_rows": signed_rows,
            "authority_file_sha256": file_sha256,
            "authority_rows_sha256": rows_sha256,
        },
        operation_id="signed-stale-source-detach",
        rows=loaded_rows,
        receipt_policy={
            "operation": "detach_stale_source_binding",
            "expected_count": "admitted_rows",
        },
        file_sha256=file_sha256,
        payload_sha256=rows_sha256,
    )


def _load_error_sibling_authority(
    *,
    mutation_kind: str | None,
    guard_set: tuple[str, ...] | None,
) -> _Authority:
    """Create authority metadata for the deterministic orphan predicate."""
    if mutation_kind != RepairMutationKind.set_properties.value:
        raise SignedManifestAuthorityError(
            "error-sibling reconcile requires the set-properties mutation kind"
        )
    if guard_set != _ERROR_SIBLING_GUARDS:
        raise SignedManifestAuthorityError(
            "error-sibling reconcile requires its exact deterministic guard set"
        )
    from imas_codex.standard_names.error_siblings import ERROR_SUFFIX_TO_OPERATOR

    authority_descriptor = {
        "adapter": _ERROR_SIBLING_ADAPTER,
        "model": _ERROR_SIBLING_MODEL,
        "operator_prefixes": sorted(
            f"{operator}_of_" for operator in ERROR_SUFFIX_TO_OPERATOR.values()
        ),
        "mutation_kind": mutation_kind,
        "guard_set": list(guard_set),
        "quarantine_reason": _ERROR_SIBLING_REASON,
    }
    authority_sha256 = _digest(authority_descriptor)
    return _Authority(
        data={
            **authority_descriptor,
            "signature_profile": "none",
            "all_or_nothing": False,
        },
        operation_id="deterministic-error-sibling-reconcile",
        rows=(),
        receipt_policy={
            "operation": "quarantine_orphaned_error_sibling",
            "expected_count": "admitted_rows",
        },
        file_sha256=authority_sha256,
        payload_sha256=authority_sha256,
    )


def _error_sibling_rows(query: _Query, reason: str) -> tuple[_LoadedRow, ...]:
    """Select the complete live cohort admitted by the orphan predicate."""
    from imas_codex.standard_names.error_siblings import error_sibling_parent_name

    candidates = query.query(
        """
        MATCH (name:StandardName)
        WHERE name.model = $model
          AND coalesce(name.validation_status, '') <> 'quarantined'
        RETURN name.id AS id
        ORDER BY name.id
        """,
        model=_ERROR_SIBLING_MODEL,
    )
    loaded_rows: list[_LoadedRow] = []
    for candidate in candidates:
        name_id = str(candidate["id"])
        parent_id = error_sibling_parent_name(name_id)
        if parent_id is None:
            continue
        parent = query.query(
            "MATCH (parent:StandardName {id: $parent_id}) RETURN parent.id LIMIT 1",
            parent_id=parent_id,
        )
        if parent:
            continue
        participant = {
            "id": name_id,
            "kind": RepairParticipantKind.node.value,
            "graph_label": "StandardName",
        }
        loaded_rows.append(
            _LoadedRow(
                id=name_id,
                identity={
                    "id": name_id,
                    "kind": "standard_name",
                    "target_id": name_id,
                },
                participants=(participant,),
                mutations=(
                    {
                        "id": f"{name_id}:quarantine",
                        "order": 0,
                        "kind": RepairMutationKind.set_properties.value,
                        "participant_id": name_id,
                        "arguments": {
                            "properties": {
                                "validation_status": "quarantined",
                                "quarantine_reason": reason,
                            }
                        },
                    },
                ),
                guards=tuple(
                    {
                        "id": implementation,
                        "kind": _GUARD_KINDS[implementation],
                        "implementation": implementation,
                        "participant_ids": [name_id],
                    }
                    for implementation in _ERROR_SIBLING_GUARDS
                ),
                orphan_policy="refuse",
            )
        )
    return tuple(loaded_rows)


def _runtime_authority_rows(authority: _Authority) -> tuple[_LoadedRow, ...]:
    rows = authority.data.get("runtime_rows")
    return tuple(rows) if rows is not None else authority.rows


def _apply_error_sibling_query_handle(query: _Query) -> dict[str, int]:
    """Preserve the lightweight query-handle contract used by unit doubles."""
    rows = query.query(
        """
        MATCH (sn:StandardName)
        WHERE sn.model = 'deterministic:dd_error_modifier'
          AND coalesce(sn.validation_status, '') <> 'quarantined'
        RETURN sn.id AS id
        """
    )
    from imas_codex.standard_names.error_siblings import error_sibling_parent_name

    orphan_ids: list[str] = []
    for row in rows or []:
        name_id = str(row["id"])
        parent_id = error_sibling_parent_name(name_id)
        if parent_id is None:
            continue
        parent = query.query(
            "MATCH (p:StandardName {id: $pid}) RETURN p.id LIMIT 1",
            pid=parent_id,
        )
        if not parent:
            orphan_ids.append(name_id)
    if orphan_ids:
        query.query(
            """
            UNWIND $ids AS sid
            MATCH (sn:StandardName {id: sid})
            SET sn.validation_status = 'quarantined',
                sn.quarantine_reason = 'orphaned error sibling (parent name deleted)'
            """,
            ids=orphan_ids,
        )
    return {"stale_marked": len(orphan_ids)}


def _scope_refusal(
    authority: _Authority,
    name_ids: list[str] | None,
    *,
    apply: bool,
) -> dict[str, Any] | None:
    if authority.data.get("adapter") != _REFUSED_TARGET_ORPHAN_ADAPTER:
        if name_ids is not None:
            raise SignedManifestAuthorityError(
                "generic signed authorities do not accept a caller row list"
            )
        return None
    signed_ids = [row.id for row in authority.rows]
    requested = signed_ids if name_ids is None else sorted(name_ids)
    if len(requested) != len(set(requested)):
        raise SignedManifestAuthorityError(
            "signed orphan retirement requires unique name ids"
        )
    outside = sorted(set(requested) - set(signed_ids))
    omitted = sorted(set(signed_ids) - set(requested))
    if not outside and not omitted:
        return None
    refusals = [
        {
            "name_id": name_id,
            "reason": "target is outside signed retirement authority",
        }
        for name_id in outside
    ] + [
        {
            "name_id": name_id,
            "reason": "signed retirement target was omitted",
        }
        for name_id in omitted
    ]
    return {
        "schema": _REFUSED_TARGET_ORPHAN_RECEIPT_SCHEMA,
        "outcome": "refused",
        "dry_run": not apply,
        "changed": 0,
        "would_change": 0,
        "counts": {
            "requested": len(requested),
            "admitted": 0,
            "refused": len(refusals),
        },
        "refusals": refusals,
        "authority_sha256": authority.payload_sha256,
    }


def _participant_snapshot(
    query: _Query, participant: dict[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]] | None] | None:
    kind = str(participant["kind"])
    participant_id = str(participant["id"])
    graph_label = str(participant["graph_label"])
    if kind == RepairParticipantKind.node.value:
        rows = query.query(
            """
            MATCH (node)
            WHERE node.id = $participant_id AND $graph_label IN labels(node)
            CALL {
                WITH node
                UNWIND keys(node) AS property_key
                WITH node, property_key ORDER BY property_key
                RETURN collect({
                    key: property_key,
                    type: valueType(node[property_key]),
                    value: CASE
                        WHEN valueType(node[property_key])
                             STARTS WITH 'ZONED DATETIME'
                        THEN toString(node[property_key].epochSeconds) + ':' +
                             toString(node[property_key].nanosecond)
                        WHEN valueType(node[property_key]) STARTS WITH 'LIST'
                        THEN [item IN node[property_key] | {
                            type: valueType(item),
                            value: CASE
                                WHEN valueType(item) STARTS WITH 'ZONED DATETIME'
                                THEN toString(item.epochSeconds) + ':' +
                                     toString(item.nanosecond)
                                ELSE toString(item)
                            END
                        }]
                        ELSE toString(node[property_key])
                    END
                }) AS property_fingerprint
            }
            RETURN elementId(node) AS element_id,
                   labels(node) AS labels,
                   properties(node) AS properties,
                   property_fingerprint
            """,
            participant_id=participant_id,
            graph_label=graph_label,
        )
    else:
        rows = query.query(
            """
            MATCH (start)-[relationship]->(end)
            WHERE elementId(relationship) = $participant_id
              AND type(relationship) = $graph_label
            RETURN elementId(relationship) AS element_id,
                   type(relationship) AS relationship_type,
                   properties(relationship) AS properties,
                   elementId(start) AS start_element_id,
                   labels(start) AS start_labels,
                   start.id AS start_id,
                   start.status AS start_status,
                   elementId(end) AS end_element_id,
                   labels(end) AS end_labels,
                   end.id AS end_id
            """,
            participant_id=participant_id,
            graph_label=graph_label,
        )
    if len(rows) != 1:
        return None
    snapshot = dict(rows[0])
    property_fingerprint = snapshot.pop("property_fingerprint", None)
    return snapshot, property_fingerprint


def _collateral_snapshot(
    query: _Query,
    *,
    excluded_node_ids: list[str],
    excluded_relationship_ids: list[str],
) -> list[dict[str, str]]:
    nodes = query.query(
        """
        MATCH (node)
        WHERE any(label IN labels(node) WHERE label IN $labels)
          AND NOT (elementId(node) IN $excluded_ids)
        RETURN elementId(node) AS element_id,
               labels(node) AS labels,
               properties(node) AS properties
        ORDER BY element_id
        """,
        labels=sorted(_NODE_LABELS),
        excluded_ids=excluded_node_ids,
    )
    relationships = query.query(
        """
        MATCH (start)-[relationship]->(end)
        WHERE type(relationship) IN $relationship_types
          AND NOT (elementId(relationship) IN $excluded_ids)
        RETURN elementId(relationship) AS element_id,
               type(relationship) AS relationship_type,
               properties(relationship) AS properties,
               elementId(start) AS start_element_id,
               elementId(end) AS end_element_id
        ORDER BY element_id
        """,
        relationship_types=sorted(_RELATIONSHIP_TYPES),
        excluded_ids=excluded_relationship_ids,
    )
    digests = [
        {"key": f"node:{row['element_id']}", "sha256": _digest(row)} for row in nodes
    ] + [
        {"key": f"relationship:{row['element_id']}", "sha256": _digest(row)}
        for row in relationships
    ]
    return sorted(digests, key=lambda row: row["key"])


def _guard_names(row: _LoadedRow) -> set[str]:
    return {str(guard["implementation"]) for guard in row.guards}


def _is_structural_reparent(row: _LoadedRow) -> bool:
    return (
        len(row.mutations) == 1
        and str(row.mutations[0]["kind"]) == _STRUCTURAL_REPARENT
        and isinstance((row.mutations[0].get("arguments") or {}).get("new_end_id"), str)
        and bool((row.mutations[0].get("arguments") or {}).get("new_end_id"))
    )


def _is_structural_release(row: _LoadedRow) -> bool:
    return (
        len(row.mutations) == 1
        and str(row.mutations[0]["kind"]) == _STRUCTURAL_REPARENT
        and "new_end_id" in (row.mutations[0].get("arguments") or {})
        and (row.mutations[0].get("arguments") or {}).get("new_end_id") is None
    )


def _signed_supersede_successor(mutation: Mapping[str, Any]) -> str | None:
    if str(mutation["kind"]) != RepairMutationKind.supersede.value:
        return None
    successor_id = (mutation.get("arguments") or {}).get("successor_id")
    return str(successor_id) if successor_id is not None else None


def _receipt_names(row: _LoadedRow) -> tuple[str, str]:
    from_name = str(
        row.identity.get("target_id") or row.identity.get("source_id") or row.id
    )
    successor_ids = [
        successor_id
        for mutation in row.mutations
        if (successor_id := _signed_supersede_successor(mutation)) is not None
    ]
    return from_name, successor_ids[0] if successor_ids else from_name


def _structural_reparent_authority(authority: _Authority) -> bool:
    rows = _runtime_authority_rows(authority)
    return bool(rows) and all(_is_structural_reparent(row) for row in rows)


def _structural_release_authority(authority: _Authority) -> bool:
    rows = _runtime_authority_rows(authority)
    return bool(rows) and all(_is_structural_release(row) for row in rows)


def _all_or_nothing(authority: _Authority) -> bool:
    return bool(authority.data.get("all_or_nothing")) or (
        _structural_reparent_authority(authority)
        or _structural_release_authority(authority)
    )


def _is_ordinary_source_migration(row: _LoadedRow) -> bool:
    kinds = {str(mutation["kind"]) for mutation in row.mutations}
    return {
        RepairMutationKind.delete_relationship.value,
        RepairMutationKind.add_relationship.value,
    } <= kinds


def _is_unbound_source_attachment(row: _LoadedRow) -> bool:
    kinds = [str(mutation["kind"]) for mutation in row.mutations]
    return kinds == [
        RepairMutationKind.add_relationship.value,
        RepairMutationKind.set_properties.value,
    ] and str(row.identity.get("source_id") or "").startswith("dd:")


def _is_dd_residue_release(row: _LoadedRow) -> bool:
    source_id = str(row.identity.get("source_id") or "")
    return (
        source_id in _DD_RESIDUE_SOURCE_IDS
        and row.identity.get("target_id") is None
        and any(
            mutation.get("kind") == RepairMutationKind.set_properties.value
            and mutation.get("arguments", {}).get("properties")
            == _DD_RESIDUE_RELEASE_PROPERTIES
            for mutation in row.mutations
        )
    )


def _dd_residue_release_refusal(query: _Query, action: dict[str, Any]) -> str | None:
    """Require an unclaimed legacy DD source with no live target."""
    row: _LoadedRow = action["row"]
    if not _is_dd_residue_release(row):
        return None
    source_id = str(row.identity["source_id"])
    source_snapshot = action["participant_snapshots"][source_id]
    properties = source_snapshot["properties"]
    if (
        properties.get("source_type") != "dd"
        or source_id != f"dd:{properties.get('source_id')}"
        or properties.get("status") not in {"composed", "attached"}
        or properties.get("claimed_at") is not None
        or properties.get("claim_token") is not None
    ):
        return "legacy DD source lifecycle does not match release authority"

    current = query.query(
        """
        MATCH (source:StandardNameSource)
        WHERE elementId(source) = $source_element_id
        OPTIONAL MATCH (source)-[binding:PRODUCED_NAME]->(target:StandardName)
        RETURN [item IN collect(CASE WHEN binding IS NULL THEN null ELSE {
          relationship_id: elementId(binding),
          target_element_id: elementId(target),
          target_id: target.id,
          target_live: NOT (coalesce(target.name_stage, '') IN
            ['superseded', 'exhausted', 'contested'])
            AND NOT (coalesce(target.status, '') IN
              ['deprecated', 'superseded'])
        } END) WHERE item IS NOT NULL] AS bindings
        """,
        source_element_id=source_snapshot["element_id"],
    )
    bindings = list(current[0].get("bindings") or []) if current else []
    if any(bool(binding.get("target_live")) for binding in bindings):
        return "source still has a live target"

    declared = sorted(
        (
            str(snapshot["element_id"]),
            str(snapshot["end_element_id"]),
            str(snapshot["end_id"]),
        )
        for snapshot in action["participant_snapshots"].values()
        if snapshot.get("relationship_type") == "PRODUCED_NAME"
    )
    observed = sorted(
        (
            str(binding["relationship_id"]),
            str(binding["target_element_id"]),
            str(binding["target_id"]),
        )
        for binding in bindings
    )
    if declared != observed:
        return "signed legacy DD source target closure changed"
    return None


def _structural_reparent_refusal(query: _Query, action: dict[str, Any]) -> str | None:
    """Require one exact live parent edge and preserve the child's authority."""
    row: _LoadedRow = action["row"]
    if not _is_structural_reparent(row):
        return None
    mutation = row.mutations[0]
    arguments = dict(mutation["arguments"])
    snapshots = action["participant_snapshots"]
    child_snapshot = snapshots[str(arguments["start_id"])]
    old_parent_snapshot = snapshots[str(arguments["old_end_id"])]
    new_parent_snapshot = snapshots[str(arguments["new_end_id"])]
    relationship_snapshot = snapshots[str(mutation["participant_id"])]
    child_properties = child_snapshot["properties"]
    if (
        child_properties.get("name_stage") == "superseded"
        or child_properties.get("status") in {"deprecated", "superseded"}
        or old_parent_snapshot["properties"].get("name_stage") != "accepted"
        or new_parent_snapshot["properties"].get("name_stage") != "accepted"
        or old_parent_snapshot["properties"].get("status")
        in {"deprecated", "superseded"}
        or new_parent_snapshot["properties"].get("status")
        in {"deprecated", "superseded"}
    ):
        return "signed structural reparent lifecycle is not live"

    closure_rows = query.query(
        """
        MATCH (child:StandardName), (old:StandardName), (new:StandardName)
        WHERE elementId(child) = $child_element_id
          AND elementId(old) = $old_parent_element_id
          AND elementId(new) = $new_parent_element_id
        RETURN properties(child) AS child_properties,
               [(child)-[parent:HAS_PARENT]->(current:StandardName) | {
                 relationship_id: elementId(parent),
                 relationship_properties: properties(parent),
                 parent_element_id: elementId(current),
                 parent_id: current.id
               }] AS parents,
               [(source:StandardNameSource)-[binding:PRODUCED_NAME]->(child) | {
                 source_element_id: elementId(source),
                 source_id: source.id,
                 source_properties: properties(source),
                 binding_element_id: elementId(binding),
                 binding_properties: properties(binding)
               }] AS producers
        """,
        child_element_id=child_snapshot["element_id"],
        old_parent_element_id=old_parent_snapshot["element_id"],
        new_parent_element_id=new_parent_snapshot["element_id"],
    )
    closure = dict(closure_rows[0]) if closure_rows else {}
    parents = sorted(
        (dict(item) for item in closure.get("parents") or []),
        key=lambda item: (str(item["parent_id"]), str(item["relationship_id"])),
    )
    producers = sorted(
        (dict(item) for item in closure.get("producers") or []),
        key=lambda item: (str(item["source_id"]), str(item["binding_element_id"])),
    )
    expected_parent = {
        "relationship_id": relationship_snapshot["element_id"],
        "relationship_properties": dict(arguments["properties"]),
        "parent_element_id": old_parent_snapshot["element_id"],
        "parent_id": str(arguments["old_end_id"]),
    }
    if (
        relationship_snapshot.get("relationship_type") != "HAS_PARENT"
        or relationship_snapshot.get("start_element_id") != child_snapshot["element_id"]
        or relationship_snapshot.get("end_element_id")
        != old_parent_snapshot["element_id"]
        or relationship_snapshot.get("properties") != arguments["properties"]
        or parents != [expected_parent]
    ):
        return (
            "signed structural reparent closure does not match exact incumbent parent"
        )
    signed_child_state = {
        "child_properties": closure.get("child_properties") or {},
        "producers": producers,
    }
    action["participant_digests"].append(
        {
            "participant_id": "structural-reparent-child-authority",
            "sha256": _digest(signed_child_state),
        }
    )
    action["structural_reparent_child_state"] = signed_child_state
    return None


def _structural_release_refusal(query: _Query, action: dict[str, Any]) -> str | None:
    """Require one exact parent edge and independent live child authority."""
    row: _LoadedRow = action["row"]
    if not _is_structural_release(row):
        return None
    mutation = row.mutations[0]
    arguments = dict(mutation["arguments"])
    snapshots = action["participant_snapshots"]
    child_snapshot = snapshots[str(arguments["start_id"])]
    old_parent_snapshot = snapshots[str(arguments["old_end_id"])]
    relationship_snapshot = snapshots[str(mutation["participant_id"])]
    child_properties = child_snapshot["properties"]
    if (
        child_properties.get("name_stage") != "accepted"
        or child_properties.get("validation_status") != "valid"
        or child_properties.get("status") in {"deprecated", "superseded"}
        or old_parent_snapshot["properties"].get("name_stage") != "accepted"
        or old_parent_snapshot["properties"].get("status")
        in {"deprecated", "superseded"}
    ):
        return "signed structural release lifecycle is not independently live"

    closure_rows = query.query(
        """
        MATCH (child:StandardName), (old:StandardName)
        WHERE elementId(child) = $child_element_id
          AND elementId(old) = $old_parent_element_id
        RETURN properties(child) AS child_properties,
               [(child)-[parent:HAS_PARENT]->(current:StandardName) | {
                 relationship_id: elementId(parent),
                 relationship_properties: properties(parent),
                 parent_element_id: elementId(current),
                 parent_id: current.id
               }] AS parents,
               [(source:StandardNameSource)-[binding:PRODUCED_NAME]->(child) | {
                 source_element_id: elementId(source),
                 source_id: source.id,
                 source_properties: properties(source),
                 binding_element_id: elementId(binding),
                 binding_properties: properties(binding)
               }] AS producers
        """,
        child_element_id=child_snapshot["element_id"],
        old_parent_element_id=old_parent_snapshot["element_id"],
    )
    closure = dict(closure_rows[0]) if closure_rows else {}
    parents = sorted(
        (dict(item) for item in closure.get("parents") or []),
        key=lambda item: (str(item["parent_id"]), str(item["relationship_id"])),
    )
    producers = sorted(
        (dict(item) for item in closure.get("producers") or []),
        key=lambda item: (str(item["source_id"]), str(item["binding_element_id"])),
    )
    expected_parent = {
        "relationship_id": relationship_snapshot["element_id"],
        "relationship_properties": dict(arguments["properties"]),
        "parent_element_id": old_parent_snapshot["element_id"],
        "parent_id": str(arguments["old_end_id"]),
    }
    if (
        relationship_snapshot.get("relationship_type") != "HAS_PARENT"
        or relationship_snapshot.get("start_element_id") != child_snapshot["element_id"]
        or relationship_snapshot.get("end_element_id")
        != old_parent_snapshot["element_id"]
        or relationship_snapshot.get("properties") != arguments["properties"]
        or parents != [expected_parent]
    ):
        return "signed structural release closure does not match exact incumbent parent"
    if not any(
        producer["source_properties"].get("status") != "stale" for producer in producers
    ):
        return "signed structural release child has no live producing source"
    signed_child_state = {
        "child_properties": closure.get("child_properties") or {},
        "producers": producers,
    }
    action["participant_digests"].append(
        {
            "participant_id": "structural-release-child-authority",
            "sha256": _digest(signed_child_state),
        }
    )
    action["structural_release_child_state"] = signed_child_state
    return None


def _ordinary_source_migration_refusal(
    query: _Query, action: dict[str, Any]
) -> str | None:
    """Bind one ordinary retarget to its complete live provenance closure."""
    row: _LoadedRow = action["row"]
    if not _is_ordinary_source_migration(row):
        return None
    snapshots = action["participant_snapshots"]
    source_id = str(row.identity["source_id"])
    target_id = str(row.identity["target_id"])
    source_snapshot = snapshots[source_id]
    target_snapshot = snapshots[target_id]
    binding_snapshot = next(
        snapshot
        for snapshot in snapshots.values()
        if snapshot.get("relationship_type") == "PRODUCED_NAME"
    )
    old_target_id = str(binding_snapshot["end_id"])
    source_properties = source_snapshot["properties"]
    target_properties = target_snapshot["properties"]
    if source_properties.get("status") == "stale":
        return "ordinary source status changed to stale"
    if (
        source_properties.get("claimed_at") is not None
        or source_properties.get("claim_token") is not None
    ):
        return "ordinary source has an active claim"
    if (
        source_properties.get("produced_sn_id") != old_target_id
        or target_properties.get("name_stage") != "accepted"
        or target_properties.get("validation_status") != "valid"
        or target_properties.get("status") in {"deprecated", "superseded"}
    ):
        return "signed ordinary source lifecycle does not match migration authority"

    closure_rows = query.query(
        """
        MATCH (source:StandardNameSource), (old:StandardName), (new:StandardName)
        WHERE elementId(source) = $source_element_id
          AND elementId(old) = $old_target_element_id
          AND elementId(new) = $new_target_element_id
        OPTIONAL MATCH (source)-[binding:PRODUCED_NAME]->(bound:StandardName)
        WITH source, old, new,
             [item IN collect(CASE WHEN binding IS NULL THEN null ELSE {
               relationship_id: elementId(binding),
               target_element_id: elementId(bound),
               target_id: bound.id
             } END) WHERE item IS NOT NULL] AS bindings
        OPTIONAL MATCH (source)-[origin:FROM_DD_PATH|FROM_SIGNAL]->(backing)
        WITH source, old, new, bindings,
             [item IN collect(CASE WHEN origin IS NULL THEN null ELSE {
               origin_id: elementId(origin),
               origin_type: type(origin),
               backing_element_id: elementId(backing),
               backing_id: backing.id,
               backing_labels: labels(backing)
             } END) WHERE item IS NOT NULL] AS backings
        OPTIONAL MATCH (backing_node)-[projection:HAS_STANDARD_NAME]->
                       (projected:StandardName)
        WHERE elementId(backing_node) IN
              [item IN backings | item.backing_element_id]
        RETURN bindings, backings,
               [item IN collect(CASE WHEN projection IS NULL THEN null ELSE {
                 projection_id: elementId(projection),
                 backing_element_id: elementId(backing_node),
                 target_element_id: elementId(projected),
                 target_id: projected.id
               } END) WHERE item IS NOT NULL] AS projections,
               old.source_paths AS old_source_paths,
               new.source_paths AS new_source_paths
        """,
        source_element_id=source_snapshot["element_id"],
        old_target_element_id=binding_snapshot["end_element_id"],
        new_target_element_id=target_snapshot["element_id"],
    )
    closure = closure_rows[0] if closure_rows else {}
    bindings = closure.get("bindings") or []
    backings = closure.get("backings") or []
    projections = closure.get("projections") or []
    if (
        bindings
        != [
            {
                "relationship_id": binding_snapshot["element_id"],
                "target_element_id": binding_snapshot["end_element_id"],
                "target_id": old_target_id,
            }
        ]
        or not backings
        or len(projections) != len(backings)
        or {item["backing_element_id"] for item in projections}
        != {item["backing_element_id"] for item in backings}
        or {item["target_id"] for item in projections} != {old_target_id}
        or any(
            item["target_element_id"] != binding_snapshot["end_element_id"]
            for item in projections
        )
    ):
        return "signed ordinary source closure does not match exact incumbent binding and projection"

    from imas_codex.standard_names.attachment_audit import guard_source_pairings

    guarded = guard_source_pairings(query, target_id, [source_id])
    if guarded.rejected or guarded.accepted_source_ids != (source_id,):
        detail = (
            ", ".join(
                f"{item.source_node_id}: {item.reason}" for item in guarded.rejected
            )
            or "pairing guard did not admit the exact source cohort"
        )
        return f"source migration attachment rejected: {detail}"
    action["participant_digests"].append(
        {
            "participant_id": "ordinary-source-provenance-closure",
            "sha256": _digest(closure),
        }
    )
    action["ordinary_source_closure"] = closure
    return None


def _unbound_source_attachment_refusal(
    query: _Query, action: dict[str, Any]
) -> str | None:
    """Admit one unbound DD source against its complete projection closure."""
    row: _LoadedRow = action["row"]
    if not _is_unbound_source_attachment(row):
        return None
    snapshots = action["participant_snapshots"]
    source_id = str(row.identity["source_id"])
    target_id = str(row.identity["target_id"])
    source_snapshot = snapshots[source_id]
    target_snapshot = snapshots[target_id]
    source_properties = source_snapshot["properties"]
    target_properties = target_snapshot["properties"]
    binding_count = query.query(
        """
        MATCH (:StandardNameSource {id: $source_id})-[:PRODUCED_NAME]->(:StandardName)
        RETURN count(*) AS bindings
        """,
        source_id=source_id,
    )[0]["bindings"]
    if int(binding_count):
        return "ordinary source is already bound"
    if (
        source_properties.get("status") != "extracted"
        or source_properties.get("source_type") != "dd"
        or source_properties.get("produced_sn_id") is not None
    ):
        return "signed ordinary source lifecycle does not match attachment authority"
    if (
        source_properties.get("claimed_at") is not None
        or source_properties.get("claim_token") is not None
    ):
        return "ordinary source has an active claim"
    if (
        target_properties.get("name_stage") != "accepted"
        or target_properties.get("validation_status") != "valid"
        or target_properties.get("status") in {"deprecated", "superseded"}
    ):
        return "signed attachment target is not an accepted live name"

    closure_rows = query.query(
        """
        MATCH (source:StandardNameSource), (target:StandardName)
        WHERE elementId(source) = $source_element_id
          AND elementId(target) = $target_element_id
        OPTIONAL MATCH (source)-[origin:FROM_DD_PATH|FROM_SIGNAL]->(backing)
        WITH source, target,
             [item IN collect(CASE WHEN origin IS NULL THEN null ELSE {
               origin_id: elementId(origin),
               origin_type: type(origin),
               backing_element_id: elementId(backing),
               backing_id: backing.id,
               backing_labels: labels(backing)
             } END) WHERE item IS NOT NULL] AS backings
        OPTIONAL MATCH (backing_node)-[projection:HAS_STANDARD_NAME]->
                       (projected:StandardName)
        WHERE elementId(backing_node) IN
              [item IN backings | item.backing_element_id]
        RETURN backings,
               [item IN collect(CASE WHEN projection IS NULL THEN null ELSE {
                 projection_id: elementId(projection),
                 backing_element_id: elementId(backing_node),
                 target_element_id: elementId(projected),
                 target_id: projected.id
               } END) WHERE item IS NOT NULL] AS projections,
               target.source_paths AS target_source_paths
        """,
        source_element_id=source_snapshot["element_id"],
        target_element_id=target_snapshot["element_id"],
    )
    closure = dict(closure_rows[0]) if closure_rows else {}
    backings = sorted(
        (dict(item) for item in closure.get("backings") or []),
        key=lambda item: (str(item["backing_id"]), str(item["origin_id"])),
    )
    projections = sorted(
        (dict(item) for item in closure.get("projections") or []),
        key=lambda item: (str(item["target_id"]), str(item["projection_id"])),
    )
    dd_path = str(source_properties.get("source_id") or "")
    if (
        source_id != f"dd:{dd_path}"
        or len(backings) != 1
        or backings[0]["origin_type"] != "FROM_DD_PATH"
        or "IMASNode" not in backings[0]["backing_labels"]
        or backings[0]["backing_id"] != dd_path
        or projections
    ):
        return "signed unbound ordinary source closure is not projection-free"

    from imas_codex.standard_names.attachment_audit import guard_source_pairings

    guarded = guard_source_pairings(query, target_id, [source_id])
    if guarded.rejected or guarded.accepted_source_ids != (source_id,):
        detail = (
            ", ".join(
                f"{item.source_node_id}: {item.reason}" for item in guarded.rejected
            )
            or "pairing guard did not admit the exact source cohort"
        )
        return f"source attachment rejected: {detail}"
    closure = {
        "backings": backings,
        "projections": projections,
        "target_source_paths": list(closure.get("target_source_paths") or []),
    }
    action["participant_digests"].append(
        {
            "participant_id": "unbound-ordinary-source-closure",
            "sha256": _digest(closure),
        }
    )
    action["unbound_source_attachment_closure"] = closure
    return None


def _source_target_reconciliation_refusal(
    query: _Query, action: dict[str, Any]
) -> str | None:
    """Require one signed survivor over the complete live target closure."""
    row: _LoadedRow = action["row"]
    if _is_ordinary_source_migration(row) or _is_dd_residue_release(row):
        return None
    if not any(
        mutation["kind"] == RepairMutationKind.delete_relationship.value
        for mutation in row.mutations
    ):
        return None
    snapshots = action["participant_snapshots"]
    source_snapshot = next(
        snapshot
        for snapshot in snapshots.values()
        if "StandardNameSource" in snapshot.get("labels", [])
    )
    binding_snapshots = [
        snapshot
        for snapshot in snapshots.values()
        if snapshot.get("relationship_type") == "PRODUCED_NAME"
    ]
    survivor_id = str(row.identity["target_id"])
    deleted_ids = {
        str(mutation["participant_id"])
        for mutation in row.mutations
        if mutation["kind"] == RepairMutationKind.delete_relationship.value
    }
    survivor_bindings = [
        snapshot
        for participant_id, snapshot in snapshots.items()
        if snapshot.get("relationship_type") == "PRODUCED_NAME"
        and participant_id not in deleted_ids
    ]
    live_bindings = query.query(
        """
        MATCH (source:StandardNameSource)-[binding:PRODUCED_NAME]->
              (target:StandardName)
        WHERE elementId(source) = $source_element_id
          AND coalesce(target.name_stage, '') <> 'superseded'
          AND NOT (coalesce(target.status, '') IN ['deprecated', 'superseded'])
        RETURN elementId(binding) AS relationship_id,
               elementId(target) AS target_element_id,
               target.id AS target_id
        ORDER BY relationship_id
        """,
        source_element_id=source_snapshot["element_id"],
    )
    declared = sorted(
        (
            str(snapshot["element_id"]),
            str(snapshot["end_element_id"]),
            str(snapshot["end_id"]),
        )
        for snapshot in binding_snapshots
    )
    current = sorted(
        (
            str(binding["relationship_id"]),
            str(binding["target_element_id"]),
            str(binding["target_id"]),
        )
        for binding in live_bindings
    )
    declared_target_ids = {
        str(snapshot["properties"]["id"])
        for snapshot in snapshots.values()
        if "StandardName" in snapshot.get("labels", [])
    }
    if (
        source_snapshot["properties"].get("status") == "stale"
        or declared != current
        or declared_target_ids != {binding[2] for binding in current}
        or len(survivor_bindings) != 1
        or survivor_bindings[0].get("end_id") != survivor_id
        or any(
            snapshot.get("start_element_id") != source_snapshot["element_id"]
            for snapshot in binding_snapshots
        )
    ):
        return "signed source-target closure does not match complete live targets"
    return None


def _structural_source_revival_refusal(
    query: _Query, action: dict[str, Any]
) -> str | None:
    """Require the complete signed bare-parent closure for explicit revival."""
    row: _LoadedRow = action["row"]
    if _is_ordinary_source_migration(row) or _is_unbound_source_attachment(row):
        return None
    if not any(
        mutation["kind"] == RepairMutationKind.add_relationship.value
        for mutation in row.mutations
    ):
        return None
    snapshots = action["participant_snapshots"]
    source_id = str(row.identity["source_id"])
    target_id = str(row.identity["target_id"])
    source_snapshot = snapshots[source_id]
    target_snapshot = snapshots[target_id]
    if source_snapshot["properties"].get("status") != "stale":
        return "structural source status changed from signed stale state"
    if (
        source_snapshot["properties"].get("produced_sn_id") is not None
        or source_snapshot["properties"].get("claimed_at") is not None
        or source_snapshot["properties"].get("claim_token") is not None
        or source_snapshot["properties"].get("source_type") != "derived"
        or source_snapshot["properties"].get("source_id") != target_id
        or target_snapshot["properties"].get("name_stage") != "accepted"
        or target_snapshot["properties"].get("status") in {"deprecated", "superseded"}
    ):
        return "signed structural source lifecycle does not match revival authority"

    closure = query.query(
        """
        MATCH (source:StandardNameSource), (target:StandardName)
        WHERE elementId(source) = $source_element_id
          AND elementId(target) = $target_element_id
        OPTIONAL MATCH (source)-[source_binding:PRODUCED_NAME]->(:StandardName)
        WITH source, target,
             [binding IN collect(source_binding)
              WHERE binding IS NOT NULL | elementId(binding)] AS source_bindings
        OPTIONAL MATCH (:StandardNameSource)-[producer:PRODUCED_NAME]->(target)
        WITH source, target, source_bindings,
             [binding IN collect(producer)
              WHERE binding IS NOT NULL | elementId(binding)] AS target_producers
        OPTIONAL MATCH (child:StandardName)-[parent:HAS_PARENT]->(target)
        WHERE coalesce(child.name_stage, '') <> 'superseded'
          AND NOT (coalesce(child.status, '') IN ['deprecated', 'superseded'])
        RETURN source_bindings, target_producers,
               [item IN collect(CASE WHEN parent IS NULL THEN null ELSE {
                 relationship_id: elementId(parent),
                 child_element_id: elementId(child),
                 child_id: child.id,
                 target_element_id: elementId(target)
               } END) WHERE item IS NOT NULL] AS live_children
        """,
        source_element_id=source_snapshot["element_id"],
        target_element_id=target_snapshot["element_id"],
    )
    state = closure[0] if closure else {}
    declared_children = sorted(
        (
            str(snapshot["element_id"]),
            str(snapshot["start_element_id"]),
            str(snapshot["start_id"]),
            str(snapshot["end_element_id"]),
        )
        for snapshot in snapshots.values()
        if snapshot.get("relationship_type") == "HAS_PARENT"
    )
    current_children = sorted(
        (
            str(child["relationship_id"]),
            str(child["child_element_id"]),
            str(child["child_id"]),
            str(child["target_element_id"]),
        )
        for child in state.get("live_children") or []
    )
    if (
        state.get("source_bindings")
        or state.get("target_producers")
        or not current_children
        or declared_children != current_children
        or any(
            child[3] != str(target_snapshot["element_id"]) for child in current_children
        )
    ):
        return "signed structural source closure does not match bare childful target"
    return None


def _removed_binding_snapshots(action: dict[str, Any]) -> list[dict[str, Any]]:
    row: _LoadedRow = action["row"]
    removed_participant_ids = {
        str(mutation["participant_id"])
        for mutation in row.mutations
        if mutation["kind"]
        in {
            RepairMutationKind.detach.value,
            RepairMutationKind.delete_relationship.value,
        }
    }
    return [
        snapshot
        for participant_id, snapshot in action["participant_snapshots"].items()
        if participant_id in removed_participant_ids
        and snapshot.get("relationship_type") == "PRODUCED_NAME"
    ]


def _target_snapshot(action: dict[str, Any]) -> dict[str, Any] | None:
    relationship = next(
        (
            snapshot
            for snapshot in action["participant_snapshots"].values()
            if snapshot.get("relationship_type") == "PRODUCED_NAME"
        ),
        None,
    )
    if relationship is not None:
        return {
            "element_id": relationship["end_element_id"],
            "id": relationship.get("end_id"),
        }
    mutation_ids = {
        str(mutation["participant_id"]) for mutation in action["row"].mutations
    }
    return next(
        (
            {
                "element_id": snapshot["element_id"],
                "id": snapshot["properties"].get("id"),
            }
            for participant_id, snapshot in action["participant_snapshots"].items()
            if participant_id in mutation_ids
            and "StandardName" in snapshot.get("labels", [])
        ),
        None,
    )


def _structural_refusal(query: _Query, action: dict[str, Any]) -> str | None:
    target = _target_snapshot(action)
    if target is None:
        return "structural target does not exist"
    rows = query.query(
        """
        MATCH (target:StandardName)
        WHERE elementId(target) = $target_element_id
        RETURN COUNT {
          (child:StandardName)-[:HAS_PARENT]->(target)
          WHERE coalesce(child.name_stage, '') <> 'superseded'
            AND coalesce(child.status, '') <> 'superseded'
        } AS live_children
        """,
        target_element_id=target["element_id"],
    )
    if not rows or int(rows[0].get("live_children") or 0) > 0:
        return "target has a live structural child"
    return None


def _producer_state(query: _Query, target_element_id: str) -> dict[str, Any]:
    rows = query.query(
        """
        MATCH (target:StandardName)
        WHERE elementId(target) = $target_element_id
        OPTIONAL MATCH (source:StandardNameSource)-[binding:PRODUCED_NAME]->(target)
        WITH target, source, binding
        ORDER BY source.id, elementId(binding)
        WITH target, collect(CASE WHEN binding IS NULL THEN null ELSE {
          relationship_id: elementId(binding),
          live: coalesce(source.status, '') <> 'stale'
        } END) AS producers
        RETURN [producer IN producers WHERE producer IS NOT NULL] AS producers,
               COUNT {
                 (child:StandardName)-[:HAS_PARENT]->(target)
                 WHERE coalesce(child.name_stage, '') <> 'superseded'
                   AND coalesce(child.status, '') <> 'superseded'
               } AS live_children
        """,
        target_element_id=target_element_id,
    )
    return dict(rows[0]) if rows else {"producers": [], "live_children": 0}


def _orphan_guard_refusal(query: _Query, action: dict[str, Any]) -> str | None:
    row: _LoadedRow = action["row"]
    guard_names = _guard_names(row)
    if not guard_names & {
        _SIGNED_LIFECYCLE,
        _NO_LIVE_PRODUCER,
        _NO_LIVE_STRUCTURAL_CHILD,
    }:
        return None
    target = _target_snapshot(action)
    if target is None:
        return "name does not exist"
    participant = row.participants[0]
    properties = action["participant_snapshots"][str(participant["id"])]["properties"]
    if _SIGNED_LIFECYCLE in guard_names:
        if properties.get("name_stage") != participant.get("expected_name_stage"):
            return "name lifecycle stage changed from signed authority"
        if (
            properties.get("claimed_at") is not None
            or properties.get("claim_token") is not None
        ):
            return "name has an active claim"
    rows = query.query(
        """
        MATCH (target:StandardName)
        WHERE elementId(target) = $target_element_id
        RETURN COUNT {
          (source:StandardNameSource)-[:PRODUCED_NAME]->(target)
          WHERE coalesce(source.status, '') <> 'stale'
        } AS live_producers,
        COUNT {
          (child:StandardName)-[:HAS_PARENT]->(target)
          WHERE child.name_stage <> 'superseded'
            AND NOT (coalesce(child.status, '') IN ['deprecated', 'superseded'])
        } AS live_children
        """,
        target_element_id=target["element_id"],
    )
    state = rows[0] if rows else {"live_producers": 0, "live_children": 0}
    if _NO_LIVE_PRODUCER in guard_names and int(state["live_producers"]) > 0:
        return "name has a live producing source"
    if _NO_LIVE_STRUCTURAL_CHILD in guard_names and int(state["live_children"]) > 0:
        return "name has a live HAS_PARENT child"
    return None


def _stale_source_participant_snapshots(
    closure: dict[str, Any], action: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    snapshots: dict[str, dict[str, Any]] = {}

    def add_node(element_id: str | None, labels: list[str]) -> None:
        if element_id is not None:
            snapshots[f"node:{element_id}"] = {
                "element_id": element_id,
                "labels": labels,
                "properties": {},
            }

    def add_relationship(
        element_id: str | None,
        relationship_type: str,
        start_element_id: str | None,
        end_element_id: str | None,
    ) -> None:
        if element_id is not None:
            snapshots[f"relationship:{element_id}"] = {
                "element_id": element_id,
                "relationship_type": relationship_type,
                "properties": {},
                "start_element_id": start_element_id,
                "end_element_id": end_element_id,
            }

    participant = next(
        row
        for row in closure["participants"]
        if row["requested_id"] == action["source_id"]
    )
    add_node(participant.get("source_element_id"), ["StandardNameSource"])
    for binding in participant.get("bindings") or []:
        add_node(binding.get("target_element_id"), ["StandardName"])
        add_relationship(
            binding.get("element_id"),
            "PRODUCED_NAME",
            participant.get("source_element_id"),
            binding.get("target_element_id"),
        )
    for backing in participant.get("backings") or []:
        add_node(backing.get("element_id"), ["IMASNode"])
        add_relationship(
            backing.get("origin_element_id"),
            "FROM_DD_PATH",
            participant.get("source_element_id"),
            backing.get("element_id"),
        )
        for projection in backing.get("projections") or []:
            add_node(projection.get("target_element_id"), ["StandardName"])
            add_relationship(
                projection.get("element_id"),
                "HAS_STANDARD_NAME",
                backing.get("element_id"),
                projection.get("target_element_id"),
            )
    target_ids = set(action["target_ids"])
    for target in closure["targets"]:
        if target["requested_id"] not in target_ids:
            continue
        add_node(target.get("target_element_id"), ["StandardName"])
        for incoming in target.get("incoming_bindings") or []:
            add_node(incoming.get("source_element_id"), ["StandardNameSource"])
            add_relationship(
                incoming.get("binding_element_id"),
                "PRODUCED_NAME",
                incoming.get("source_element_id"),
                target.get("target_element_id"),
            )
        for child in target.get("live_children") or []:
            add_node(child.get("child_element_id"), ["StandardName"])
            add_relationship(
                child.get("parent_element_id"),
                "HAS_PARENT",
                child.get("child_element_id"),
                target.get("target_element_id"),
            )
    for version in closure["current_versions"]:
        add_node(version.get("element_id"), ["DDVersion"])
    return snapshots


def _build_stale_source_preview(
    query: _Query, authority: _Authority, reason: str
) -> _Preview:
    signed_rows = list(authority.data["signed_rows"])
    closure = _stale_source_detach_closure(query, signed_rows)
    actions = _validate_stale_source_detach_closure(signed_rows, closure)
    selected_ids = [row["source_id"] for row in signed_rows]
    out_count, out_hash = _out_of_allowlist_source_hash(query, selected_ids)
    manifest = {
        "operation": "detach_signed_" + "stale_source_bindings",
        "reason": reason,
        "authority_file_sha256": authority.file_sha256,
        "authority_rows_sha256": authority.payload_sha256,
        "signed_rows": signed_rows,
        "closure": closure,
        "actions": actions,
        "out_of_allowlist": {"count": out_count, "sha256": out_hash},
    }
    rows_by_id = {row.id: row for row in authority.rows}
    admitted = [
        {
            "row": rows_by_id[action["source_id"]],
            "participant_snapshots": _stale_source_participant_snapshots(
                closure, action
            ),
            "participant_digests": [],
            "stale_action": action,
        }
        for action in actions
    ]
    node_ids = sorted(
        {
            snapshot["element_id"]
            for item in admitted
            for snapshot in item["participant_snapshots"].values()
            if "labels" in snapshot
        }
    )
    relationship_ids = sorted(
        {
            snapshot["element_id"]
            for item in admitted
            for snapshot in item["participant_snapshots"].values()
            if "relationship_type" in snapshot
        }
    )
    authority.data["stale_actions"] = actions
    authority.data["out_of_allowlist"] = manifest["out_of_allowlist"]
    return _Preview(
        manifest=manifest,
        manifest_sha256=_graph_payload_hash(manifest),
        admitted=admitted,
        refusals=[],
        collateral=_collateral_snapshot(
            query,
            excluded_node_ids=node_ids,
            excluded_relationship_ids=relationship_ids,
        ),
    )


def _build_preview(query: _Query, authority: _Authority, reason: str) -> _Preview:
    if authority.data.get("adapter") == _STALE_SOURCE_ADAPTER:
        return _build_stale_source_preview(query, authority, reason)
    if authority.data.get("adapter") == _ERROR_SIBLING_ADAPTER:
        authority.data["runtime_rows"] = _error_sibling_rows(query, reason)
    candidates: list[dict[str, Any]] = []
    refusals: list[dict[str, str]] = []
    for row in _runtime_authority_rows(authority):
        snapshots: dict[str, dict[str, Any]] = {}
        property_fingerprints: dict[str, list[dict[str, Any]]] = {}
        refusal: str | None = None
        for participant in row.participants:
            participant_id = str(participant["id"])
            participant_state = _participant_snapshot(query, participant)
            if participant_state is None:
                refusal = f"participant does not exist: {participant_id}"
                break
            snapshot, property_fingerprint = participant_state
            signature = participant.get("signature_sha256")
            if signature is not None and _digest(snapshot) != signature:
                refusal = f"participant signature mismatch: {participant_id}"
                break
            snapshots[participant_id] = snapshot
            if property_fingerprint is not None:
                property_fingerprints[participant_id] = property_fingerprint
        action = {
            "row": row,
            "participant_snapshots": snapshots,
            "property_fingerprints": property_fingerprints,
            "participant_digests": [
                {"participant_id": participant_id, "sha256": _digest(snapshot)}
                for participant_id, snapshot in sorted(snapshots.items())
            ],
        }
        if refusal is None:
            refusal = _dd_residue_release_refusal(query, action)
        if refusal is None:
            refusal = _orphan_guard_refusal(query, action)
        if refusal is None:
            refusal = _structural_reparent_refusal(query, action)
        if refusal is None:
            refusal = _structural_release_refusal(query, action)
        if refusal is None:
            refusal = _ordinary_source_migration_refusal(query, action)
        if refusal is None:
            refusal = _unbound_source_attachment_refusal(query, action)
        if refusal is None:
            refusal = _source_target_reconciliation_refusal(query, action)
        if refusal is None:
            refusal = _structural_source_revival_refusal(query, action)
        if refusal is None and _STRUCTURAL_LEGITIMACY in _guard_names(row):
            refusal = _structural_refusal(query, action)
        if refusal is not None:
            refusals.append({"row_id": row.id, "reason": refusal})
        else:
            candidates.append(action)

    admitted: list[dict[str, Any]] = []
    removed_relationship_ids: set[str] = set()
    producer_cache: dict[str, dict[str, Any]] = {}
    for action in candidates:
        row = action["row"]
        if _LAST_PRODUCER in _guard_names(row):
            removed_bindings = _removed_binding_snapshots(action)
            row_removed_ids = {
                str(relationship["element_id"]) for relationship in removed_bindings
            }
            strips_last_producer = False
            for relationship in removed_bindings:
                target_element_id = str(relationship["end_element_id"])
                producer_state = producer_cache.setdefault(
                    target_element_id, _producer_state(query, target_element_id)
                )
                remaining_live = [
                    producer
                    for producer in producer_state["producers"]
                    if producer.get("live")
                    and producer["relationship_id"] not in removed_relationship_ids
                    and producer["relationship_id"] not in row_removed_ids
                ]
                if (
                    not remaining_live
                    and int(producer_state.get("live_children") or 0) < 1
                ):
                    strips_last_producer = True
                    break
            if strips_last_producer:
                refusals.append(
                    {
                        "row_id": row.id,
                        "reason": "target would lose its last producing source",
                    }
                )
                continue
            removed_relationship_ids.update(row_removed_ids)
        admitted.append(action)

    admitted_node_ids = sorted(
        {
            snapshot["element_id"]
            for action in admitted
            for snapshot in action["participant_snapshots"].values()
            if "labels" in snapshot
        }
    )
    admitted_relationship_ids = sorted(
        {
            snapshot["element_id"]
            for action in admitted
            for snapshot in action["participant_snapshots"].values()
            if "relationship_type" in snapshot
        }
    )
    collateral = _collateral_snapshot(
        query,
        excluded_node_ids=admitted_node_ids,
        excluded_relationship_ids=admitted_relationship_ids,
    )
    manifest_rows = [
        {
            "row_id": action["row"].id,
            "identity": action["row"].identity,
            "mutation_kinds": [
                str(mutation["kind"]) for mutation in action["row"].mutations
            ],
            "participant_digests": action["participant_digests"],
            "closure_sha256": _digest(action["participant_digests"]),
        }
        for action in admitted
    ]
    refusals.sort(key=lambda item: (item["row_id"], item["reason"]))
    manifest = {
        "schema": SIGNED_MANIFEST_SCHEMA,
        "operation_id": authority.operation_id,
        "reason": reason,
        "authority_file_sha256": authority.file_sha256,
        "authority_payload_sha256": authority.payload_sha256,
        "rows": manifest_rows,
        "admitted_row_ids": [action["row"].id for action in admitted],
        "refusals": refusals,
        "collateral_rows": collateral,
        "collateral_sha256": _digest(collateral),
    }
    return _Preview(
        manifest=manifest,
        manifest_sha256=_digest(manifest),
        admitted=admitted,
        refusals=refusals,
        collateral=collateral,
    )


def _change_id(manifest_sha256: str, row_id: str) -> str:
    row_digest = hashlib.sha256(row_id.encode()).hexdigest()[:24]
    return f"sn-change:signed-manifest:{manifest_sha256}:{row_digest}"


def _receipt_rows(
    query: _Query, operation: str, manifest_sha256: str
) -> list[dict[str, Any]]:
    return query.query(
        """
        MATCH (change:StandardNameChange {
          operation: $operation,
          manifest_sha256: $manifest_sha256
        })
        RETURN properties(change) AS properties
        ORDER BY change.id
        """,
        operation=operation,
        manifest_sha256=manifest_sha256,
    )


def _verify_structural_reparent_postcondition(
    query: _Query,
    row: _LoadedRow,
    action: dict[str, Any] | None,
) -> None:
    mutation = row.mutations[0]
    arguments = dict(mutation["arguments"])
    states = query.query(
        """
        MATCH (child:StandardName {id: $child_id}),
              (old:StandardName {id: $old_parent_id}),
              (new:StandardName {id: $new_parent_id})
        RETURN properties(child) AS child_properties,
               [(child)-[parent:HAS_PARENT]->(current:StandardName) | {
                 relationship_properties: properties(parent),
                 parent_id: current.id
               }] AS parents,
               [(source:StandardNameSource)-[binding:PRODUCED_NAME]->(child) | {
                 source_element_id: elementId(source),
                 source_id: source.id,
                 source_properties: properties(source),
                 binding_element_id: elementId(binding),
                 binding_properties: properties(binding)
               }] AS producers
        """,
        child_id=arguments["start_id"],
        old_parent_id=arguments["old_end_id"],
        new_parent_id=arguments["new_end_id"],
    )
    state = dict(states[0]) if states else {}
    parents = sorted(
        (dict(item) for item in state.get("parents") or []),
        key=lambda item: str(item["parent_id"]),
    )
    expected_parents = [
        {
            "relationship_properties": dict(arguments["properties"]),
            "parent_id": str(arguments["new_end_id"]),
        }
    ]
    if parents != expected_parents:
        raise SignedManifestConflict(
            "recorded structural reparent lost its exact postcondition"
        )
    if action is None:
        return
    signed_child_state = action["structural_reparent_child_state"]
    producers = sorted(
        (dict(item) for item in state.get("producers") or []),
        key=lambda item: (str(item["source_id"]), str(item["binding_element_id"])),
    )
    if {
        "child_properties": state.get("child_properties") or {},
        "producers": producers,
    } != signed_child_state:
        raise SignedManifestConflict(
            "structural reparent changed child lifecycle or producing-source authority"
        )


def _verify_structural_release_postcondition(
    query: _Query,
    row: _LoadedRow,
    action: dict[str, Any] | None,
) -> None:
    mutation = row.mutations[0]
    arguments = dict(mutation["arguments"])
    states = query.query(
        """
        MATCH (child:StandardName {id: $child_id}),
              (old:StandardName {id: $old_parent_id})
        RETURN properties(child) AS child_properties,
               [(child)-[parent:HAS_PARENT]->(current:StandardName) | {
                 relationship_properties: properties(parent),
                 parent_id: current.id
               }] AS parents,
               [(source:StandardNameSource)-[binding:PRODUCED_NAME]->(child) | {
                 source_element_id: elementId(source),
                 source_id: source.id,
                 source_properties: properties(source),
                 binding_element_id: elementId(binding),
                 binding_properties: properties(binding)
               }] AS producers
        """,
        child_id=arguments["start_id"],
        old_parent_id=arguments["old_end_id"],
    )
    state = dict(states[0]) if states else {}
    if not states or state.get("parents"):
        raise SignedManifestConflict(
            "recorded structural release lost its exact parentless postcondition"
        )
    if action is None:
        return
    signed_child_state = action["structural_release_child_state"]
    producers = sorted(
        (dict(item) for item in state.get("producers") or []),
        key=lambda item: (str(item["source_id"]), str(item["binding_element_id"])),
    )
    if {
        "child_properties": state.get("child_properties") or {},
        "producers": producers,
    } != signed_child_state:
        raise SignedManifestConflict(
            "structural release changed child lifecycle or producing-source authority"
        )


def _verify_ordinary_source_migration_postcondition(
    query: _Query,
    row: _LoadedRow,
    participant_snapshots: dict[str, dict[str, Any]] | None,
) -> None:
    source_id = str(row.identity["source_id"])
    target_id = str(row.identity["target_id"])
    old_target_id: str | None = None
    if participant_snapshots is not None:
        deleted_binding_id = next(
            str(mutation["participant_id"])
            for mutation in row.mutations
            if mutation["kind"] == RepairMutationKind.delete_relationship.value
        )
        old_target_id = str(participant_snapshots[deleted_binding_id]["end_id"])
    states = query.query(
        """
        MATCH (source:StandardNameSource {id: $source_id}),
              (new:StandardName {id: $target_id})
        OPTIONAL MATCH (old:StandardName {id: $old_target_id})
        OPTIONAL MATCH (source)-[:PRODUCED_NAME]->(bound:StandardName)
        WITH source, old, new, collect(DISTINCT bound.id) AS bindings
        OPTIONAL MATCH (source)-[:FROM_DD_PATH|FROM_SIGNAL]->(backing)
        OPTIONAL MATCH (backing)-[:HAS_STANDARD_NAME]->(projected:StandardName)
        WITH source, old, new, bindings,
             collect(DISTINCT backing.id) AS backing_ids,
             collect(DISTINCT projected.id) AS projected_ids
        OPTIONAL MATCH (old_source:StandardNameSource)-[:PRODUCED_NAME]->(old)
        OPTIONAL MATCH (old_source)-[:FROM_DD_PATH]->(old_dd:IMASNode)
        OPTIONAL MATCH (old_source)-[:FROM_SIGNAL]->(old_signal:FacilitySignal)
        WITH source, old, new, bindings, backing_ids, projected_ids,
             collect(DISTINCT CASE
               WHEN old_source IS NULL THEN null
               WHEN old_dd IS NOT NULL THEN 'dd:' + old_dd.id
               WHEN old_signal IS NOT NULL THEN old_signal.id
               WHEN old_source.source_type = 'derived'
                AND old_source.source_id STARTS WITH 'derived:'
               THEN old_source.source_id ELSE old_source.id END) AS expected_old_paths
        OPTIONAL MATCH (new_source:StandardNameSource)-[:PRODUCED_NAME]->(new)
        OPTIONAL MATCH (new_source)-[:FROM_DD_PATH]->(new_dd:IMASNode)
        OPTIONAL MATCH (new_source)-[:FROM_SIGNAL]->(new_signal:FacilitySignal)
        RETURN source.produced_sn_id AS scalar, bindings, backing_ids, projected_ids,
               old.source_paths AS old_paths,
               new.source_paths AS new_paths,
               expected_old_paths,
               collect(DISTINCT CASE
                 WHEN new_source IS NULL THEN null
                 WHEN new_dd IS NOT NULL THEN 'dd:' + new_dd.id
                 WHEN new_signal IS NOT NULL THEN new_signal.id
                 WHEN new_source.source_type = 'derived'
                  AND new_source.source_id STARTS WITH 'derived:'
                 THEN new_source.source_id ELSE new_source.id END) AS expected_new_paths
        """,
        source_id=source_id,
        old_target_id=old_target_id,
        target_id=target_id,
    )
    state = states[0] if states else {}
    if (
        state.get("scalar") != target_id
        or state.get("bindings") != [target_id]
        or not state.get("backing_ids")
        or state.get("projected_ids") != [target_id]
        or (
            old_target_id is not None
            and sorted(state.get("old_paths") or [])
            != sorted(item for item in state.get("expected_old_paths") or [] if item)
        )
        or sorted(state.get("new_paths") or [])
        != sorted(item for item in state.get("expected_new_paths") or [] if item)
    ):
        raise SignedManifestConflict(
            "recorded ordinary-source migration lost its postcondition"
        )


def _verify_unbound_source_attachment_postcondition(
    query: _Query,
    row: _LoadedRow,
    action: dict[str, Any] | None,
) -> None:
    source_id = str(row.identity["source_id"])
    target_id = str(row.identity["target_id"])
    state_rows = query.query(
        """
        MATCH (source:StandardNameSource {id: $source_id}),
              (target:StandardName {id: $target_id})
        OPTIONAL MATCH (source)-[:PRODUCED_NAME]->(bound:StandardName)
        WITH source, target, collect(DISTINCT bound.id) AS bindings
        OPTIONAL MATCH (source)-[:FROM_DD_PATH]->(backing:IMASNode)
        OPTIONAL MATCH (backing)-[:HAS_STANDARD_NAME]->(projected:StandardName)
        RETURN source.status AS status,
               source.produced_sn_id AS scalar,
               source.source_id AS dd_path,
               bindings,
               collect(DISTINCT backing.id) AS backing_ids,
               collect(DISTINCT projected.id) AS projected_ids,
               target.source_paths AS target_source_paths
        """,
        source_id=source_id,
        target_id=target_id,
    )
    state = dict(state_rows[0]) if state_rows else {}
    dd_path = str(state.get("dd_path") or "")
    expected_uri = f"dd:{dd_path}"
    if (
        state.get("status") != "attached"
        or state.get("scalar") != target_id
        or state.get("bindings") != [target_id]
        or state.get("backing_ids") != [dd_path]
        or state.get("projected_ids") != [target_id]
        or expected_uri not in (state.get("target_source_paths") or [])
    ):
        raise SignedManifestConflict(
            "recorded unbound source attachment lost its exact four-mirror postcondition"
        )
    if action is None:
        return
    prior_paths = action["unbound_source_attachment_closure"]["target_source_paths"]
    expected_paths = (
        prior_paths if expected_uri in prior_paths else [*prior_paths, expected_uri]
    )
    if state.get("target_source_paths") != expected_paths:
        raise SignedManifestConflict(
            "unbound source attachment changed the target source-path mirror unexpectedly"
        )


def _verify_postconditions(
    query: _Query,
    authority: _Authority,
    row_ids: list[str],
    actions: list[dict[str, Any]] | None = None,
) -> None:
    if authority.data.get("adapter") == _STALE_SOURCE_ADAPTER:
        signed_rows = list(authority.data["signed_rows"])
        post = query.query(
            """
            UNWIND $rows AS expected
            MATCH (source:StandardNameSource {id: expected.source_id})
            OPTIONAL MATCH (source)-[:PRODUCED_NAME]->(bound:StandardName)
            WITH expected, source, collect(DISTINCT bound.id) AS bindings
            OPTIONAL MATCH (source)-[:FROM_DD_PATH]->(backing:IMASNode)
            OPTIONAL MATCH (backing)-[:HAS_STANDARD_NAME]->(projected:StandardName)
            WHERE projected.id IN expected.target_ids
            RETURN expected.source_id AS source_id,
                   source.produced_sn_id AS scalar,
                   bindings,
                   collect(DISTINCT projected.id) AS projections
            ORDER BY source_id
            """,
            rows=[
                {
                    "source_id": row["source_id"],
                    "target_ids": _signed_stale_source_target_ids(row),
                }
                for row in signed_rows
            ],
        )
        if len(post) != len(signed_rows) or any(
            row.get("scalar") is not None
            or row.get("bindings")
            or row.get("projections")
            for row in post
        ):
            raise StaleSourceDetachConflict("stale-source detach postcondition failed")
        actions = list(authority.data["stale_actions"])
        target_post = query.query(
            """
            UNWIND $target_ids AS target_id
            MATCH (target:StandardName {id: target_id})
            OPTIONAL MATCH (live:StandardNameSource)-[:PRODUCED_NAME]->(target)
            WHERE live.status <> 'stale'
            WITH target_id, target, count(DISTINCT live) AS live_producers
            OPTIONAL MATCH (child:StandardName)-[:HAS_PARENT]->(target)
            WHERE coalesce(child.name_stage, '') <> 'superseded'
              AND coalesce(child.status, '') <> 'superseded'
            RETURN target_id, live_producers,
                   count(DISTINCT child) AS live_children
            ORDER BY target_id
            """,
            target_ids=sorted(
                {target_id for action in actions for target_id in action["target_ids"]}
            ),
        )
        if any(
            int(row.get("live_producers") or 0) < 1
            and int(row.get("live_children") or 0) < 1
            for row in target_post
        ):
            raise StaleSourceDetachConflict(
                "stale-source target authority was stripped"
            )
        out_count, out_hash = _out_of_allowlist_source_hash(
            query, [row["source_id"] for row in signed_rows]
        )
        if {"count": out_count, "sha256": out_hash} != authority.data.get(
            "out_of_allowlist"
        ):
            raise StaleSourceDetachConflict("out-of-allowlist source closure changed")
        authority.data["target_post"] = target_post
        return
    by_id = {row.id: row for row in _runtime_authority_rows(authority)}
    actions_by_id = {action["row"].id: action for action in actions or []}
    for row_id in row_ids:
        row = by_id[row_id]
        if _is_structural_release(row):
            _verify_structural_release_postcondition(
                query, row, actions_by_id.get(row_id)
            )
            continue
        if _is_structural_reparent(row):
            _verify_structural_reparent_postcondition(
                query, row, actions_by_id.get(row_id)
            )
            continue
        if _is_ordinary_source_migration(row):
            action = actions_by_id.get(row_id)
            _verify_ordinary_source_migration_postcondition(
                query,
                row,
                action["participant_snapshots"] if action is not None else None,
            )
            continue
        if _is_unbound_source_attachment(row):
            _verify_unbound_source_attachment_postcondition(
                query, row, actions_by_id.get(row_id)
            )
            continue
        for mutation in row.mutations:
            participant = next(
                item
                for item in row.participants
                if item["id"] == mutation["participant_id"]
            )
            kind = str(mutation["kind"])
            if kind in {
                RepairMutationKind.detach.value,
                RepairMutationKind.delete_relationship.value,
            }:
                present = query.query(
                    """
                    MATCH ()-[relationship]->()
                    WHERE elementId(relationship) = $element_id
                    RETURN count(relationship) AS count
                    """,
                    element_id=participant["id"],
                )[0]["count"]
                if int(present) != 0:
                    raise SignedManifestConflict(
                        "recorded signed-manifest repair lost its postcondition"
                    )
            elif kind == RepairMutationKind.add_relationship.value:
                arguments = dict(mutation.get("arguments") or {})
                state = query.query(
                    """
                    MATCH (source:StandardNameSource {id: $source_id})
                          -[binding:PRODUCED_NAME]->
                          (target:StandardName {id: $target_id})
                    RETURN count(binding) AS count
                    """,
                    source_id=arguments["start_id"],
                    target_id=arguments["end_id"],
                )
                if not state or int(state[0].get("count") or 0) != 1:
                    raise SignedManifestConflict(
                        "recorded signed-manifest repair lost its postcondition"
                    )
            elif kind == RepairMutationKind.delete.value:
                present = query.query(
                    "MATCH (node) WHERE node.id = $id RETURN count(node) AS count",
                    id=participant["id"],
                )[0]["count"]
                if int(present) != 0:
                    raise SignedManifestConflict(
                        "recorded signed-manifest repair lost its postcondition"
                    )
            elif kind == RepairMutationKind.set_properties.value:
                expected = dict(mutation.get("arguments", {}).get("properties", {}))
                state = query.query(
                    "MATCH (node {id: $id}) RETURN properties(node) AS properties",
                    id=participant["id"],
                )
                if not state or any(
                    state[0]["properties"].get(key) != value
                    for key, value in expected.items()
                ):
                    raise SignedManifestConflict(
                        "recorded signed-manifest repair lost its postcondition"
                    )
            else:
                successor_id = _signed_supersede_successor(mutation)
                state = query.query(
                    """
                    MATCH (node:StandardName {id: $id})
                    RETURN node.name_stage AS name_stage, node.status AS status,
                           node.superseded_by AS superseded_by
                    """,
                    id=participant["id"],
                )
                expected = {
                    "name_stage": "superseded",
                    "status": "superseded",
                    "superseded_by": successor_id,
                }
                if not state or state[0] != expected:
                    raise SignedManifestConflict(
                        "recorded signed-manifest repair lost its postcondition"
                    )
        guard_names = _guard_names(row)
        if guard_names & {_NO_LIVE_PRODUCER, _NO_LIVE_STRUCTURAL_CHILD}:
            target_id = str(row.identity.get("target_id") or row.id)
            closure = query.query(
                """
                MATCH (target:StandardName {id: $target_id})
                RETURN COUNT {
                  (source:StandardNameSource)-[:PRODUCED_NAME]->(target)
                  WHERE coalesce(source.status, '') <> 'stale'
                } AS live_producers,
                COUNT {
                  (child:StandardName)-[:HAS_PARENT]->(target)
                  WHERE child.name_stage <> 'superseded'
                    AND NOT (coalesce(child.status, '') IN
                      ['deprecated', 'superseded'])
                } AS live_children
                """,
                target_id=target_id,
            )
            state = closure[0] if closure else {}
            if (
                _NO_LIVE_PRODUCER in guard_names
                and int(state.get("live_producers") or 0) != 0
            ) or (
                _NO_LIVE_STRUCTURAL_CHILD in guard_names
                and int(state.get("live_children") or 0) != 0
            ):
                raise SignedManifestConflict(
                    "recorded signed-manifest repair lost its postcondition"
                )


def _replay(
    query: _Query, authority: _Authority, manifest_sha256: str
) -> dict[str, Any] | None:
    if authority.data.get("adapter") == _STALE_SOURCE_ADAPTER:
        signed_rows = list(authority.data["signed_rows"])
        event_ids = {
            row["source_id"]: "sn-change:stale-source-detach:"
            + hashlib.sha256(
                f"{authority.payload_sha256}\0{row['source_id']}".encode()
            ).hexdigest()
            for row in signed_rows
        }
        replay = query.query(
            """
            UNWIND $rows AS expected
            OPTIONAL MATCH (event:StandardNameChange {id: expected.event_id})
            OPTIONAL MATCH (source:StandardNameSource {id: expected.source_id})
            OPTIONAL MATCH (source)-[:PRODUCED_NAME]->(target:StandardName)
            WITH expected, event, source,
                 collect(DISTINCT target.id) AS targets
            OPTIONAL MATCH (source)-[:FROM_DD_PATH]->(backing:IMASNode)
            OPTIONAL MATCH (backing)-[:HAS_STANDARD_NAME]->(projected:StandardName)
            WHERE projected.id IN expected.target_ids
            RETURN expected.source_id AS source_id,
                   event.id IS NOT NULL AS event_exists,
                   event.manifest_sha256 AS event_manifest_sha256,
                   event.authority_rows_sha256 AS event_authority_rows_sha256,
                   source.produced_sn_id AS scalar,
                   targets,
                   collect(DISTINCT projected.id) AS projections
            ORDER BY source_id
            """,
            rows=[
                {
                    "source_id": row["source_id"],
                    "target_ids": _signed_stale_source_target_ids(row),
                    "event_id": event_ids[row["source_id"]],
                }
                for row in signed_rows
            ],
        )
        recorded = [row for row in replay if row.get("event_exists")]
        if not recorded:
            return None
        if len(recorded) != len(signed_rows) or any(
            row.get("event_manifest_sha256") != manifest_sha256
            or row.get("event_authority_rows_sha256") != authority.payload_sha256
            or row.get("scalar") is not None
            or row.get("targets")
            or row.get("projections")
            for row in replay
        ):
            raise StaleSourceDetachConflict(
                "recorded stale-source detach lost its exact postcondition"
            )
        return {
            "schema": SIGNED_MANIFEST_RECEIPT_SCHEMA,
            "outcome": "already_applied",
            "changed": 0,
            "receipt_rows": len(recorded),
            "manifest_sha256": manifest_sha256,
        }
    operation = str(authority.receipt_policy["operation"])
    receipts = _receipt_rows(query, operation, manifest_sha256)
    if not receipts:
        return None
    properties = [dict(row["properties"]) for row in receipts]
    admitted_ids = sorted(properties[0].get("cohort_admitted_ids") or [])
    expected_ids = sorted(item.get("row_id") for item in properties)
    receipt_names = {
        row.id: _receipt_names(row) for row in _runtime_authority_rows(authority)
    }
    if (
        not admitted_ids
        or expected_ids != admitted_ids
        or len(properties) != len(admitted_ids)
        or any(
            sorted(item.get("cohort_admitted_ids") or []) != admitted_ids
            or item.get("authority_file_sha256") != authority.file_sha256
            or item.get("authority_payload_sha256") != authority.payload_sha256
            or (
                item.get("from_name"),
                item.get("to_name"),
            )
            != receipt_names.get(str(item.get("row_id")))
            for item in properties
        )
    ):
        raise SignedManifestConflict("signed-manifest receipt cohort is incomplete")
    _verify_postconditions(query, authority, admitted_ids)
    return {
        "schema": SIGNED_MANIFEST_RECEIPT_SCHEMA,
        "outcome": "already_applied",
        "changed": 0,
        "persistent_writes": 0,
        "receipt_rows": len(properties),
        "manifest_sha256": manifest_sha256,
        "admitted_row_ids": admitted_ids,
    }


def _lock_participants(query: _Query, preview: _Preview) -> None:
    node_ids = sorted(
        {
            snapshot["element_id"]
            for action in preview.admitted
            for snapshot in action["participant_snapshots"].values()
            if "labels" in snapshot
        }
    )
    relationship_ids = sorted(
        {
            snapshot["element_id"]
            for action in preview.admitted
            for snapshot in action["participant_snapshots"].values()
            if "relationship_type" in snapshot
        }
    )
    locked_nodes = query.query(
        """
        UNWIND $element_ids AS expected_id
        MATCH (node) WHERE elementId(node) = expected_id
        SET node += {}
        RETURN collect(elementId(node)) AS ids
        """,
        element_ids=node_ids,
    )[0]["ids"]
    locked_relationships = query.query(
        """
        UNWIND $element_ids AS expected_id
        MATCH ()-[relationship]->()
        WHERE elementId(relationship) = expected_id
        SET relationship += {}
        RETURN collect(elementId(relationship)) AS ids
        """,
        element_ids=relationship_ids,
    )[0]["ids"]
    if (
        sorted(locked_nodes) != node_ids
        or sorted(locked_relationships) != relationship_ids
    ):
        raise SignedManifestConflict(
            "signed-manifest participants changed while locking"
        )


def _added_relationship_ids(query: _Query, actions: list[dict[str, Any]]) -> list[str]:
    source_pairs = [
        {
            "source_id": mutation["arguments"]["start_id"],
            "target_id": mutation["arguments"]["end_id"],
        }
        for action in actions
        for mutation in action["row"].mutations
        if mutation["kind"] == RepairMutationKind.add_relationship.value
    ]
    reparent_pairs = [
        {
            "child_id": mutation["arguments"]["start_id"],
            "parent_id": mutation["arguments"]["new_end_id"],
        }
        for action in actions
        for mutation in action["row"].mutations
        if mutation["kind"] == _STRUCTURAL_REPARENT
        and _is_structural_reparent(action["row"])
    ]
    created_ids: list[str] = []
    if source_pairs:
        rows = query.query(
            """
            UNWIND $pairs AS pair
            MATCH (:StandardNameSource {id: pair.source_id})
                  -[binding:PRODUCED_NAME]->
                  (:StandardName {id: pair.target_id})
            RETURN collect(elementId(binding)) AS ids
            """,
            pairs=source_pairs,
        )
        created_ids.extend(str(item) for item in (rows[0].get("ids") or []) if item)
    if reparent_pairs:
        rows = query.query(
            """
            UNWIND $pairs AS pair
            MATCH (:StandardName {id: pair.child_id})
                  -[parent:HAS_PARENT]->
                  (:StandardName {id: pair.parent_id})
            RETURN collect(elementId(parent)) AS ids
            """,
            pairs=reparent_pairs,
        )
        created_ids.extend(str(item) for item in (rows[0].get("ids") or []) if item)
    return sorted(created_ids)


def _apply_structural_reparent(query: _Query, action: dict[str, Any]) -> int:
    """Relocate one exact parent edge while retaining its full property map."""
    row: _LoadedRow = action["row"]
    mutation = row.mutations[0]
    arguments = dict(mutation["arguments"])
    snapshots = action["participant_snapshots"]
    child_snapshot = snapshots[str(arguments["start_id"])]
    old_parent_snapshot = snapshots[str(arguments["old_end_id"])]
    new_parent_snapshot = snapshots[str(arguments["new_end_id"])]
    relationship_snapshot = snapshots[str(mutation["participant_id"])]
    changed = query.query(
        """
        MATCH (child:StandardName), (old:StandardName), (new:StandardName)
        WHERE elementId(child) = $child_element_id
          AND elementId(old) = $old_parent_element_id
          AND elementId(new) = $new_parent_element_id
          AND COUNT { (child)-[:HAS_PARENT]->(:StandardName) } = 1
        MATCH (child)-[prior:HAS_PARENT]->(old)
        WHERE elementId(prior) = $relationship_element_id
          AND properties(prior) = $relationship_properties
        DELETE prior
        CREATE (child)-[replacement:HAS_PARENT]->(new)
        SET replacement = $relationship_properties
        RETURN child.id AS child_id, elementId(replacement) AS relationship_id
        """,
        child_element_id=child_snapshot["element_id"],
        old_parent_element_id=old_parent_snapshot["element_id"],
        new_parent_element_id=new_parent_snapshot["element_id"],
        relationship_element_id=relationship_snapshot["element_id"],
        relationship_properties=dict(arguments["properties"]),
    )
    if len(changed) != 1 or changed[0].get("child_id") != arguments["start_id"]:
        raise SignedManifestConflict(
            f"structural reparent compare-and-set changed for row {row.id}"
        )
    action["created_relationship_id"] = str(changed[0]["relationship_id"])
    return 1


def _apply_structural_release(query: _Query, action: dict[str, Any]) -> int:
    """Remove one exact parent edge without creating a replacement."""
    row: _LoadedRow = action["row"]
    mutation = row.mutations[0]
    arguments = dict(mutation["arguments"])
    snapshots = action["participant_snapshots"]
    child_snapshot = snapshots[str(arguments["start_id"])]
    old_parent_snapshot = snapshots[str(arguments["old_end_id"])]
    relationship_snapshot = snapshots[str(mutation["participant_id"])]
    changed = query.query(
        """
        MATCH (child:StandardName), (old:StandardName)
        WHERE elementId(child) = $child_element_id
          AND elementId(old) = $old_parent_element_id
          AND COUNT { (child)-[:HAS_PARENT]->(:StandardName) } = 1
        MATCH (child)-[prior:HAS_PARENT]->(old)
        WHERE elementId(prior) = $relationship_element_id
          AND properties(prior) = $relationship_properties
        DELETE prior
        RETURN child.id AS child_id
        """,
        child_element_id=child_snapshot["element_id"],
        old_parent_element_id=old_parent_snapshot["element_id"],
        relationship_element_id=relationship_snapshot["element_id"],
        relationship_properties=dict(arguments["properties"]),
    )
    if len(changed) != 1 or changed[0].get("child_id") != arguments["start_id"]:
        raise SignedManifestConflict(
            f"structural release compare-and-set changed for row {row.id}"
        )
    return 1


def _apply_ordinary_source_migration(query: _Query, action: dict[str, Any]) -> int:
    """Atomically retarget the edge, scalar, backing projection, and paths."""
    row: _LoadedRow = action["row"]
    snapshots = action["participant_snapshots"]
    source_id = str(row.identity["source_id"])
    target_id = str(row.identity["target_id"])
    source_snapshot = snapshots[source_id]
    target_snapshot = snapshots[target_id]
    binding_snapshot = next(
        snapshot
        for snapshot in snapshots.values()
        if snapshot.get("relationship_type") == "PRODUCED_NAME"
    )
    closure = action["ordinary_source_closure"]
    result = query.query(
        """
        MATCH (source:StandardNameSource), (old:StandardName), (new:StandardName)
        WHERE elementId(source) = $source_element_id
          AND elementId(old) = $old_target_element_id
          AND elementId(new) = $new_target_element_id
          AND source.status <> 'stale'
          AND source.claimed_at IS NULL
          AND source.claim_token IS NULL
          AND source.produced_sn_id = old.id
          AND COUNT { (source)-[:PRODUCED_NAME]->(:StandardName) } = 1
        MATCH (source)-[prior:PRODUCED_NAME]->(old)
        WHERE elementId(prior) = $binding_element_id
        MATCH (source)-[origin:FROM_DD_PATH|FROM_SIGNAL]->(backing)
        WHERE elementId(origin) IN $origin_element_ids
          AND elementId(backing) IN $backing_element_ids
        MATCH (backing)-[projection:HAS_STANDARD_NAME]->(old)
        WHERE elementId(projection) IN $projection_element_ids
        WITH source, old, new, prior,
             collect(DISTINCT origin) AS origins,
             collect(DISTINCT backing) AS backings,
             collect(DISTINCT projection) AS projections
        WHERE size(origins) = size($origin_element_ids)
          AND size(backings) = size($backing_element_ids)
          AND size(projections) = size($projection_element_ids)
        DELETE prior
        FOREACH (item IN projections | DELETE item)
        CREATE (source)-[:PRODUCED_NAME]->(new)
        SET source.produced_sn_id = new.id
        FOREACH (item IN backings | MERGE (item)-[:HAS_STANDARD_NAME]->(new))
        WITH DISTINCT source, old, new
        OPTIONAL MATCH (remaining:StandardNameSource)-[:PRODUCED_NAME]->(old)
        OPTIONAL MATCH (remaining)-[:FROM_DD_PATH]->(remaining_dd:IMASNode)
        OPTIONAL MATCH (remaining)-[:FROM_SIGNAL]->
                       (remaining_signal:FacilitySignal)
        WITH source, old, new,
             collect(DISTINCT CASE
               WHEN remaining IS NULL THEN null
               WHEN remaining_dd IS NOT NULL THEN 'dd:' + remaining_dd.id
               WHEN remaining_signal IS NOT NULL THEN remaining_signal.id
               WHEN remaining.source_type = 'derived'
                AND remaining.source_id STARTS WITH 'derived:'
               THEN remaining.source_id ELSE remaining.id END) AS old_paths
        OPTIONAL MATCH (current:StandardNameSource)-[:PRODUCED_NAME]->(new)
        OPTIONAL MATCH (current)-[:FROM_DD_PATH]->(current_dd:IMASNode)
        OPTIONAL MATCH (current)-[:FROM_SIGNAL]->
                       (current_signal:FacilitySignal)
        WITH source, old, new, old_paths,
             collect(DISTINCT CASE
               WHEN current IS NULL THEN null
               WHEN current_dd IS NOT NULL THEN 'dd:' + current_dd.id
               WHEN current_signal IS NOT NULL THEN current_signal.id
               WHEN current.source_type = 'derived'
                AND current.source_id STARTS WITH 'derived:'
               THEN current.source_id ELSE current.id END) AS new_paths
        SET old.source_paths = [path IN old_paths WHERE path IS NOT NULL],
            new.source_paths = [path IN new_paths WHERE path IS NOT NULL]
        RETURN source.id AS source_id
        """,
        source_element_id=source_snapshot["element_id"],
        old_target_element_id=binding_snapshot["end_element_id"],
        new_target_element_id=target_snapshot["element_id"],
        binding_element_id=binding_snapshot["element_id"],
        origin_element_ids=[item["origin_id"] for item in closure["backings"]],
        backing_element_ids=[
            item["backing_element_id"] for item in closure["backings"]
        ],
        projection_element_ids=[
            item["projection_id"] for item in closure["projections"]
        ],
    )
    if len(result) != 1 or result[0].get("source_id") != source_id:
        raise SignedManifestConflict(
            f"ordinary-source migration compare-and-set changed for row {row.id}"
        )
    return len(row.mutations)


def _apply_unbound_source_attachment(query: _Query, action: dict[str, Any]) -> int:
    """Atomically write the source lifecycle, edge, projection, and path mirrors."""
    row: _LoadedRow = action["row"]
    snapshots = action["participant_snapshots"]
    source_id = str(row.identity["source_id"])
    target_id = str(row.identity["target_id"])
    source_snapshot = snapshots[source_id]
    target_snapshot = snapshots[target_id]
    closure = action["unbound_source_attachment_closure"]
    backing = closure["backings"][0]
    expected_paths = closure["target_source_paths"]
    result = query.query(
        """
        MATCH (source:StandardNameSource), (target:StandardName), (backing:IMASNode)
        WHERE elementId(source) = $source_element_id
          AND elementId(target) = $target_element_id
          AND elementId(backing) = $backing_element_id
          AND source.id = $source_id
          AND source.source_type = 'dd'
          AND source.source_id = $dd_path
          AND source.status = 'extracted'
          AND source.produced_sn_id IS NULL
          AND source.claimed_at IS NULL
          AND source.claim_token IS NULL
          AND target.name_stage = 'accepted'
          AND target.validation_status = 'valid'
          AND NOT (coalesce(target.status, '') IN ['deprecated', 'superseded'])
          AND coalesce(target.source_paths, []) = $expected_paths
          AND COUNT { (source)-[:PRODUCED_NAME]->(:StandardName) } = 0
          AND COUNT { (backing)-[:HAS_STANDARD_NAME]->(:StandardName) } = 0
        MATCH (source)-[origin:FROM_DD_PATH]->(backing)
        WHERE elementId(origin) = $origin_element_id
        CREATE (source)-[:PRODUCED_NAME]->(target)
        CREATE (backing)-[:HAS_STANDARD_NAME]->(target)
        SET source.status = 'attached',
            source.composed_at = datetime(),
            source.claimed_at = null,
            source.claim_token = null,
            source.produced_sn_id = target.id,
            source.last_error = null,
            target.source_paths = CASE
              WHEN $source_id IN coalesce(target.source_paths, [])
              THEN target.source_paths
              ELSE coalesce(target.source_paths, []) + $source_id END
        RETURN source.id AS source_id
        """,
        source_element_id=source_snapshot["element_id"],
        target_element_id=target_snapshot["element_id"],
        backing_element_id=backing["backing_element_id"],
        origin_element_id=backing["origin_id"],
        source_id=source_id,
        dd_path=backing["backing_id"],
        expected_paths=expected_paths,
    )
    if len(result) != 1 or result[0].get("source_id") != source_id:
        raise SignedManifestConflict(
            f"unbound source attachment compare-and-set changed for row {row.id}"
        )
    return len(row.mutations)


def _apply_mutation(query: _Query, action: dict[str, Any]) -> int:
    if "stale_action" in action:
        expected = action["stale_action"]
        changed = query.query(
            """
            MATCH (source:StandardNameSource {id: $row.source_id})
            WHERE elementId(source) = $row.source_element_id
              AND source.status = 'stale'
              AND source.produced_sn_id = $row.scalar_target
              AND source.claimed_at IS NULL
              AND source.claim_token IS NULL
            MATCH (source)-[binding:PRODUCED_NAME]->(target:StandardName)
            WHERE elementId(binding) IN $row.binding_element_ids
              AND elementId(target) IN $row.target_element_ids
            WITH source, collect(binding) AS bindings, collect(target) AS targets
            WHERE size(bindings) = size($row.binding_element_ids)
              AND size(targets) = size($row.target_element_ids)
            OPTIONAL MATCH (backing:IMASNode)-[projection:HAS_STANDARD_NAME]->
              (projected:StandardName)
            WHERE elementId(backing) IN $row.backing_element_ids
              AND elementId(projection) IN $row.projection_element_ids
              AND elementId(projected) IN $row.target_element_ids
            WITH source, bindings, targets, collect(projection) AS projections
            WHERE size(projections) = size($row.projection_element_ids)
            FOREACH (binding IN bindings | DELETE binding)
            FOREACH (projection IN projections | DELETE projection)
            SET source.produced_sn_id = null
            FOREACH (target IN targets |
              SET target.source_paths = [path IN coalesce(target.source_paths, [])
                WHERE NOT (path = source.id OR path = source.source_id
                           OR path = 'dd:' + source.source_id)])
            RETURN source.id AS source_id,
                   size(bindings) AS bindings_removed,
                   size(projections) AS projections_removed
            """,
            row=expected,
        )
        if len(changed) != 1 or changed[0].get("source_id") != expected["source_id"]:
            raise StaleSourceDetachConflict("stale-source compare-and-set changed")
        action["bindings_removed"] = int(changed[0]["bindings_removed"])
        action["projections_removed"] = int(changed[0]["projections_removed"])
        return 1
    changed = 0
    row: _LoadedRow = action["row"]
    if _is_structural_release(row):
        return _apply_structural_release(query, action)
    if _is_structural_reparent(row):
        return _apply_structural_reparent(query, action)
    if _is_ordinary_source_migration(row):
        return _apply_ordinary_source_migration(query, action)
    if _is_unbound_source_attachment(row):
        return _apply_unbound_source_attachment(query, action)
    snapshots = action["participant_snapshots"]
    for mutation in row.mutations:
        participant_id = str(mutation["participant_id"])
        snapshot = snapshots[participant_id]
        kind = str(mutation["kind"])
        if kind in {
            RepairMutationKind.detach.value,
            RepairMutationKind.delete_relationship.value,
        }:
            result = query.query(
                """
                MATCH (start)-[relationship:PRODUCED_NAME]->(end)
                WHERE elementId(relationship) = $relationship_id
                  AND elementId(start) = $start_id
                  AND elementId(end) = $end_id
                DELETE relationship
                RETURN count(*) AS changed
                """,
                relationship_id=snapshot["element_id"],
                start_id=snapshot["start_element_id"],
                end_id=snapshot["end_element_id"],
            )
        elif kind == RepairMutationKind.add_relationship.value:
            arguments = dict(mutation.get("arguments") or {})
            source_snapshot = snapshots[str(arguments["start_id"])]
            target_snapshot = snapshots[str(arguments["end_id"])]
            result = query.query(
                """
                MATCH (source:StandardNameSource), (target:StandardName)
                WHERE elementId(source) = $source_element_id
                  AND elementId(target) = $target_element_id
                  AND source.status = 'stale'
                  AND source.produced_sn_id IS NULL
                  AND source.claimed_at IS NULL
                  AND source.claim_token IS NULL
                  AND target.name_stage = 'accepted'
                  AND NOT (coalesce(target.status, '') IN
                    ['deprecated', 'superseded'])
                  AND NOT EXISTS {
                    MATCH (source)-[:PRODUCED_NAME]->(:StandardName)
                  }
                  AND NOT EXISTS {
                    MATCH (:StandardNameSource)-[:PRODUCED_NAME]->(target)
                  }
                CREATE (source)-[binding:PRODUCED_NAME]->(target)
                RETURN count(binding) AS changed
                """,
                source_element_id=source_snapshot["element_id"],
                target_element_id=target_snapshot["element_id"],
            )
        elif kind == RepairMutationKind.set_properties.value:
            properties = dict(mutation.get("arguments", {}).get("properties", {}))
            result = query.query(
                """
                MATCH (target)
                WHERE elementId(target) = $element_id
                CALL {
                    WITH target
                    UNWIND keys(target) AS property_key
                    WITH target, property_key ORDER BY property_key
                    RETURN collect({
                        key: property_key,
                        type: valueType(target[property_key]),
                        value: CASE
                            WHEN valueType(target[property_key])
                                 STARTS WITH 'ZONED DATETIME'
                            THEN toString(target[property_key].epochSeconds) + ':' +
                                 toString(target[property_key].nanosecond)
                            WHEN valueType(target[property_key]) STARTS WITH 'LIST'
                            THEN [item IN target[property_key] | {
                                type: valueType(item),
                                value: CASE
                                    WHEN valueType(item)
                                         STARTS WITH 'ZONED DATETIME'
                                    THEN toString(item.epochSeconds) + ':' +
                                         toString(item.nanosecond)
                                    ELSE toString(item)
                                END
                            }]
                            ELSE toString(target[property_key])
                        END
                    }) AS property_fingerprint
                }
                WITH target, property_fingerprint
                WHERE property_fingerprint = $expected_property_fingerprint
                SET target += $properties
                RETURN count(target) AS changed
                """,
                element_id=snapshot["element_id"],
                expected_property_fingerprint=action["property_fingerprints"][
                    participant_id
                ],
                properties=properties,
            )
        elif kind == RepairMutationKind.supersede.value:
            source_path_update = (
                ""
                if mutation.get("preserve_source_paths")
                else "target.source_paths = [],"
            )
            successor_id = _signed_supersede_successor(mutation)
            successor_update = (
                "target.superseded_by = $successor_id,"
                if successor_id is not None
                else ""
            )
            result = query.query(
                f"""
                MATCH (target:StandardName)
                WHERE elementId(target) = $element_id
                SET target.superseded_from_stage = coalesce(
                      target.superseded_from_stage, target.name_stage),
                    target.name_stage = 'superseded',
                    target.status = 'superseded',
                    {successor_update}
                    {source_path_update}
                    target.claimed_at = null,
                    target.claim_token = null
                RETURN count(target) AS changed
                """,  # noqa: S608 - the inserted fragment is selected locally
                element_id=snapshot["element_id"],
                successor_id=successor_id,
            )
        else:
            result = query.query(
                """
                MATCH (target)
                WHERE elementId(target) = $element_id
                  AND NOT (target)--()
                DELETE target
                RETURN count(*) AS changed
                """,
                element_id=snapshot["element_id"],
            )
        mutation_changed = int(result[0].get("changed") or 0) if result else 0
        if mutation_changed != 1:
            raise SignedManifestConflict(
                f"signed-manifest compare-and-set changed for row {row.id}"
            )
        changed += mutation_changed
    return changed


def _write_receipts(
    query: _Query,
    authority: _Authority,
    preview: _Preview,
    *,
    reason: str,
    run_id: str | None,
) -> list[str]:
    if authority.data.get("adapter") == _STALE_SOURCE_ADAPTER:
        rows = []
        for action in preview.admitted:
            expected = action["stale_action"]
            source_id = expected["source_id"]
            rows.append(
                {
                    "change_id": "sn-change:stale-source-detach:"
                    + hashlib.sha256(
                        f"{authority.payload_sha256}\0{source_id}".encode()
                    ).hexdigest(),
                    **expected,
                }
            )
        receipts = query.query(
            """
            UNWIND $rows AS row
            CREATE (change:StandardNameChange {id: row.change_id})
            SET change.from_name = row.scalar_target,
                change.to_name = row.scalar_target,
                change.operation = 'detach_stale_source_binding',
                change.reason = row.unblocks,
                change.origin = 'stale_source_lifecycle',
                change.run_id = $run_id,
                change.changed_at = datetime(),
                change.internal = true,
                change.source_id = row.source_id,
                change.detached_target_ids = row.target_ids,
                change.manifest_sha256 = $manifest_sha256,
                change.authority_rows_sha256 = $authority_rows_sha256
            WITH row, change
            UNWIND row.target_ids AS target_id
            MATCH (target:StandardName {id: target_id})
            MERGE (target)-[:HAS_INTERNAL_CHANGE]->(change)
            RETURN DISTINCT change.id AS change_id
            ORDER BY change_id
            """,
            rows=rows,
            run_id=run_id,
            manifest_sha256=preview.manifest_sha256,
            authority_rows_sha256=authority.payload_sha256,
        )
        return [str(row["change_id"]) for row in receipts]
    admitted_ids = [action["row"].id for action in preview.admitted]
    rows: list[dict[str, Any]] = []
    for action in preview.admitted:
        row: _LoadedRow = action["row"]
        from_name, to_name = _receipt_names(row)
        owner_element_id = next(
            (
                snapshot["element_id"]
                for participant_id, snapshot in action["participant_snapshots"].items()
                if "labels" in snapshot
                and not any(
                    mutation["participant_id"] == participant_id
                    and mutation["kind"] == RepairMutationKind.delete.value
                    for mutation in row.mutations
                )
            ),
            None,
        )
        rows.append(
            {
                "change_id": _change_id(preview.manifest_sha256, row.id),
                "row_id": row.id,
                "from_name": from_name,
                "to_name": to_name,
                "owner_element_id": owner_element_id,
                "mutation_kinds": [str(item["kind"]) for item in row.mutations],
            }
        )
    receipts = query.query(
        """
        UNWIND $rows AS row
        CREATE (change:StandardNameChange {
          id: row.change_id,
          from_name: row.from_name,
          to_name: row.to_name,
          operation: $operation,
          reason: $reason,
          origin: 'signed_manifest',
          run_id: $run_id,
          changed_at: datetime(),
          internal: true,
          row_id: row.row_id,
          mutation_kinds: row.mutation_kinds,
          manifest_sha256: $manifest_sha256,
          authority_file_sha256: $authority_file_sha256,
          authority_payload_sha256: $authority_payload_sha256,
          cohort_admitted_ids: $admitted_ids
        })
        WITH row, change
        OPTIONAL MATCH (owner)
        WHERE elementId(owner) = row.owner_element_id
        FOREACH (_ IN CASE WHEN owner IS NULL THEN [] ELSE [1] END |
          MERGE (owner)-[:HAS_INTERNAL_CHANGE]->(change))
        RETURN change.id AS change_id
        ORDER BY change.id
        """,
        rows=rows,
        operation=authority.receipt_policy["operation"],
        reason=reason,
        run_id=run_id,
        manifest_sha256=preview.manifest_sha256,
        authority_file_sha256=authority.file_sha256,
        authority_payload_sha256=authority.payload_sha256,
        admitted_ids=admitted_ids,
    )
    return [str(row["change_id"]) for row in receipts]


def _project_refused_target_orphan_receipt(
    authority: _Authority, receipt: dict[str, Any]
) -> dict[str, Any]:
    if authority.data.get("adapter") != _REFUSED_TARGET_ORPHAN_ADAPTER:
        return receipt
    counts = receipt.get("counts") or {}
    projected = {
        **receipt,
        "schema": _REFUSED_TARGET_ORPHAN_RECEIPT_SCHEMA,
        "counts": {
            "requested": int(counts.get("authority_rows") or len(authority.rows)),
            "admitted": int(counts.get("admitted") or 0),
            "refused": int(counts.get("refused") or 0),
        },
        "refusals": [
            {"name_id": row["row_id"], "reason": row["reason"]}
            for row in receipt.get("refusals") or []
        ],
    }
    outcome = receipt.get("outcome")
    if outcome == "would_apply":
        projected["dry_run"] = True
    elif outcome in {"applied", "already_applied"}:
        projected["dry_run"] = False
        projected["superseded"] = (
            len(authority.rows)
            if outcome == "already_applied"
            else int(receipt.get("changed") or 0)
        )
        projected["ledger_rows"] = int(receipt.get("receipt_rows") or 0)
        projected["persistent_writes"] = (
            0 if outcome == "already_applied" else int(receipt.get("changed") or 0) * 4
        )
    return projected


def _project_stale_source_receipt(
    authority: _Authority, receipt: dict[str, Any]
) -> dict[str, Any]:
    if authority.data.get("adapter") != _STALE_SOURCE_ADAPTER:
        return receipt
    outcome = str(receipt["outcome"])
    base = {
        "schema": _STALE_SOURCE_RECEIPT_SCHEMA,
        "outcome": outcome,
        "changed": int(receipt.get("changed") or 0),
        "receipt_rows": int(receipt.get("receipt_rows") or 0),
        "authority_file_sha256": authority.file_sha256,
        "authority_rows_sha256": authority.payload_sha256,
        "manifest_sha256": receipt.get("manifest_sha256"),
    }
    if outcome == "would_apply":
        actions = list(authority.data["stale_actions"])
        return {
            **base,
            "would_change": len(actions),
            "receipt_rows": len(actions),
            "bindings_to_remove": sum(
                len(action["binding_element_ids"]) for action in actions
            ),
            "projections_to_remove": sum(
                len(action["projection_element_ids"]) for action in actions
            ),
            "out_of_allowlist": authority.data["out_of_allowlist"],
        }
    if outcome == "already_applied":
        return base
    actions = list(authority.data["stale_actions"])
    counters_before = authority.data["counters_before"]
    counters_after = authority.data["counters_after"]
    target_post = list(authority.data["target_post"])
    return {
        **base,
        "change_ids": list(receipt.get("change_ids") or []),
        "bindings_removed": sum(
            int(action.get("bindings_removed") or 0)
            for action in receipt.get("admitted_actions") or []
        ),
        "projections_removed": sum(
            int(action.get("projections_removed") or 0)
            for action in receipt.get("admitted_actions") or []
        ),
        "minimum_live_producers_after": min(
            int(row["live_producers"]) for row in target_post
        ),
        "minimum_live_children_after": min(
            int(row["live_children"]) for row in target_post
        ),
        "StandardNameChange": {
            "before": int(counters_before["changes"]),
            "after": int(counters_after["changes"]),
            "delta": len(actions),
        },
        "LLMCost": {
            "before": int(counters_before["llm_costs"]),
            "after": int(counters_after["llm_costs"]),
            "delta": 0,
        },
        "out_of_allowlist": authority.data["out_of_allowlist"],
    }


def _project_receipt(authority: _Authority, receipt: dict[str, Any]) -> dict[str, Any]:
    if authority.data.get("adapter") == _ERROR_SIBLING_ADAPTER:
        return {"stale_marked": int(receipt.get("changed") or 0)}
    return _project_stale_source_receipt(
        authority, _project_refused_target_orphan_receipt(authority, receipt)
    )


def _apply_dual_authority_retirement(
    source_adjudication: dict[str, Any],
    retirement_authority: dict[str, Any],
    *,
    retirement_authority_sha256: str | None,
    mutation_kind: str | None,
    guard_set: tuple[str, ...] | None,
    reason: str,
    apply: bool,
    manifest_sha256: str | None,
    run_id: str | None,
    gc: Any | None,
) -> dict[str, Any]:
    """Run the closed source-release and target-supersession program."""
    from imas_codex.standard_names.graph_ops import (
        SIGNED_DUAL_AUTHORITY_RETIREMENT_RECEIPT_SCHEMA,
        SignedDualAuthorityRetirementConflict,
        SignedSourceDispositionConflict,
        _authority_payload_hash,
        _lock_signed_source_disposition_authority,
        _signed_dual_authority_change_id,
        _signed_dual_authority_retirement_manifest,
        _validate_signed_dual_authority_retirement,
    )

    if mutation_kind != _DUAL_AUTHORITY_MUTATION:
        raise SignedManifestAuthorityError(
            "dual-authority retirement requires its exact compound mutation"
        )
    if guard_set != _DUAL_AUTHORITY_GUARDS:
        raise SignedManifestAuthorityError(
            "dual-authority retirement requires its exact signed guard set"
        )
    if retirement_authority_sha256 is None:
        raise SignedManifestAuthorityError(
            "dual-authority retirement requires retirement_authority_sha256"
        )
    if not reason.strip():
        raise ValueError("dual-authority retirement requires a non-empty reason")
    if apply and manifest_sha256 is None:
        raise ValueError("apply requires manifest_sha256")
    if manifest_sha256 is not None:
        _require_sha256(manifest_sha256, "manifest_sha256")

    source_sha256, source_rows_sha256, source_rows, retirement_rows = (
        _validate_signed_dual_authority_retirement(
            source_adjudication,
            retirement_authority,
            retirement_authority_sha256,
        )
    )
    target_ids = [row["name"] for row in retirement_rows]
    signed_pairs = sorted(
        (binding["source_id"], row["name"])
        for row in retirement_rows
        for binding in row["current_removed_bindings"]
    )
    own_client = gc is None
    client: Any = GraphClient() if own_client else gc
    try:
        with client.session() as session:
            transaction = session.begin_transaction()
            query = _TransactionQuery(transaction)
            try:
                if apply:
                    replay_rows = query.query(
                        """
                        UNWIND $targets AS expected
                        OPTIONAL MATCH (target:StandardName {id: expected.target_id})
                        OPTIONAL MATCH (target)-[:HAS_INTERNAL_CHANGE]->
                          (change:StandardNameChange {
                            id: expected.event_id,
                            operation: 'retire_signed_dual_authority_target',
                            manifest_sha256: $manifest_sha256
                          })
                        RETURN collect({target_id: expected.target_id,
                          stage: target.name_stage,
                          status: target.status,
                          live_sources: COUNT {
                            (source:StandardNameSource)-[:PRODUCED_NAME]->(target)
                            WHERE coalesce(source.status, '') <> 'stale'
                          },
                          live_children: COUNT {
                            (child:StandardName)-[:HAS_PARENT]->(target)
                            WHERE child.name_stage <> 'superseded'
                              AND NOT (coalesce(child.status, '') IN
                                ['deprecated', 'superseded'])
                          }, change_id: change.id}) AS rows
                        """,
                        targets=[
                            {
                                "target_id": target_id,
                                "event_id": _signed_dual_authority_change_id(
                                    str(manifest_sha256), target_id
                                ),
                            }
                            for target_id in target_ids
                        ],
                        manifest_sha256=manifest_sha256,
                    )
                    replay = replay_rows[0].get("rows") if replay_rows else []
                    present = [row for row in replay if row.get("change_id")]
                    if present:
                        remaining_rows = query.query(
                            """
                            UNWIND $pairs AS expected
                            OPTIONAL MATCH (:StandardNameSource {id: expected.source_id})
                              -[binding:PRODUCED_NAME]->
                              (:StandardName {id: expected.target_id})
                            RETURN count(binding) AS remaining
                            """,
                            pairs=[
                                {"source_id": source_id, "target_id": target_id}
                                for source_id, target_id in signed_pairs
                            ],
                        )
                        remaining = int(remaining_rows[0].get("remaining") or 0)
                        if (
                            len(present) != len(target_ids)
                            or len(replay) != len(target_ids)
                            or remaining != 0
                            or any(
                                row.get("stage") != "superseded"
                                or row.get("status") != "superseded"
                                or int(row.get("live_sources") or 0) != 0
                                or int(row.get("live_children") or 0) != 0
                                for row in replay
                            )
                        ):
                            raise SignedDualAuthorityRetirementConflict(
                                "recorded retirement has lost its postcondition"
                            )
                        transaction.rollback()
                        return {
                            "schema": SIGNED_DUAL_AUTHORITY_RETIREMENT_RECEIPT_SCHEMA,
                            "outcome": "already_applied",
                            "dry_run": False,
                            "changed": 0,
                            "persistent_writes": 0,
                            "sources_reconciled": len(source_rows),
                            "bindings_released": len(signed_pairs),
                            "superseded": len(target_ids),
                            "ledger_rows": len(present),
                            "manifest_sha256": manifest_sha256,
                        }

                manifest, actions, refusals = (
                    _signed_dual_authority_retirement_manifest(
                        query,
                        source_rows,
                        retirement_rows,
                        source_sha256=source_sha256,
                        source_row_set_sha256=source_rows_sha256,
                        retirement_authority_sha256=retirement_authority_sha256,
                        reason=reason,
                    )
                )
                computed_hash = _authority_payload_hash(manifest)
                counts = {
                    "sources": len(source_rows),
                    "bindings": len(signed_pairs),
                    "targets": len(target_ids),
                    "admitted_sources": len(actions),
                    "refusals": len(refusals),
                }
                if apply and computed_hash != manifest_sha256:
                    raise SignedDualAuthorityRetirementConflict(
                        "fresh dual-authority manifest does not match signed hash"
                    )
                if refusals:
                    transaction.rollback()
                    return {
                        "schema": SIGNED_DUAL_AUTHORITY_RETIREMENT_RECEIPT_SCHEMA,
                        "outcome": "refused",
                        "dry_run": not apply,
                        "changed": 0,
                        "would_change": 0,
                        "counts": counts,
                        "refusals": refusals,
                        "manifest": manifest,
                        "manifest_sha256": computed_hash,
                    }
                if not apply:
                    transaction.rollback()
                    return {
                        "schema": SIGNED_DUAL_AUTHORITY_RETIREMENT_RECEIPT_SCHEMA,
                        "outcome": "would_apply",
                        "dry_run": True,
                        "changed": 0,
                        "would_change": len(target_ids),
                        "counts": counts,
                        "refusals": [],
                        "manifest": manifest,
                        "manifest_sha256": computed_hash,
                    }

                try:
                    _lock_signed_source_disposition_authority(query, manifest)
                except SignedSourceDispositionConflict as exc:
                    raise SignedDualAuthorityRetirementConflict(str(exc)) from exc
                locked_manifest, locked_actions, locked_refusals = (
                    _signed_dual_authority_retirement_manifest(
                        query,
                        source_rows,
                        retirement_rows,
                        source_sha256=source_sha256,
                        source_row_set_sha256=source_rows_sha256,
                        retirement_authority_sha256=retirement_authority_sha256,
                        reason=reason,
                    )
                )
                if (
                    locked_refusals
                    or _authority_payload_hash(locked_manifest) != computed_hash
                ):
                    raise SignedDualAuthorityRetirementConflict(
                        "dual authority changed while acquiring locks"
                    )

                scalar_rows = [
                    {
                        "source_id": action["source_id"],
                        "source_element_id": action["source_element_id"],
                        "prior_scalar_target": action["prior_scalar_target"],
                        "keep_target_id": action["keep_target_id"],
                    }
                    for action in locked_actions
                ]
                scalar_updates = query.query(
                    """
                    UNWIND $rows AS expected
                    MATCH (source:StandardNameSource {id: expected.source_id})
                    WHERE elementId(source) = expected.source_element_id
                      AND ((source.produced_sn_id IS NULL
                            AND expected.prior_scalar_target IS NULL)
                           OR source.produced_sn_id = expected.prior_scalar_target)
                      AND source.claimed_at IS NULL
                      AND source.claim_token IS NULL
                    SET source.produced_sn_id = expected.keep_target_id
                    RETURN collect(source.id) AS ids
                    """,
                    rows=scalar_rows,
                )
                if sorted(scalar_updates[0].get("ids") or []) != sorted(
                    row["source_id"] for row in source_rows
                ):
                    raise SignedDualAuthorityRetirementConflict(
                        "source scalar compare-and-set changed"
                    )

                mutation_rows = [
                    {
                        "source_id": action["source_id"],
                        "keep_target_id": action["keep_target_id"],
                        "remove_target_id": removal["target_id"],
                        "binding_element_id": removal["binding_element_id"],
                        "projection_element_id": removal["projection_element_id"],
                        "backing_element_id": removal["backing_element_id"],
                    }
                    for action in locked_actions
                    for removal in action["removals"]
                ]
                released = query.query(
                    """
                    UNWIND $rows AS expected
                    MATCH (source:StandardNameSource {id: expected.source_id})
                    MATCH (source)-[binding:PRODUCED_NAME]->
                      (target:StandardName {id: expected.remove_target_id})
                    WHERE elementId(binding) = expected.binding_element_id
                      AND source.produced_sn_id = expected.keep_target_id
                      AND EXISTS {
                        (source)-[:PRODUCED_NAME]->
                          (:StandardName {id: expected.keep_target_id})
                      }
                    MATCH (backing)-[projection:HAS_STANDARD_NAME]->(target)
                    WHERE elementId(backing) = expected.backing_element_id
                      AND elementId(projection) = expected.projection_element_id
                    DELETE binding, projection
                    RETURN collect(expected.source_id + '|' + expected.remove_target_id)
                      AS pairs
                    """,
                    rows=mutation_rows,
                )
                if len(released[0].get("pairs") or []) != len(signed_pairs):
                    raise SignedDualAuthorityRetirementConflict(
                        "signed binding closure changed during release"
                    )

                lifecycle_rows = [
                    {
                        "name_id": row["name"],
                        "prior_name_stage": row["name_stage"],
                        "event_id": _signed_dual_authority_change_id(
                            computed_hash, row["name"]
                        ),
                    }
                    for row in retirement_rows
                ]
                retired = query.query(
                    """
                    UNWIND $rows AS expected
                    MATCH (target:StandardName {id: expected.name_id})
                    WHERE target.name_stage = expected.prior_name_stage
                      AND target.claimed_at IS NULL
                      AND target.claim_token IS NULL
                      AND NOT EXISTS {
                        (source:StandardNameSource)-[:PRODUCED_NAME]->(target)
                        WHERE coalesce(source.status, '') <> 'stale'
                      }
                      AND NOT EXISTS {
                        (child:StandardName)-[:HAS_PARENT]->(target)
                        WHERE child.name_stage <> 'superseded'
                          AND NOT (coalesce(child.status, '') IN
                            ['deprecated', 'superseded'])
                      }
                    SET target.superseded_from_stage = coalesce(
                          target.superseded_from_stage, target.name_stage),
                        target.name_stage = 'superseded',
                        target.status = 'superseded',
                        target.source_paths = [],
                        target.claimed_at = null,
                        target.claim_token = null
                    CREATE (change:StandardNameChange {
                      id: expected.event_id,
                      from_name: expected.name_id,
                      to_name: expected.name_id,
                      operation: 'retire_signed_dual_authority_target',
                      reason: $reason,
                      origin: 'semantic_source_reconciliation',
                      run_id: $run_id,
                      changed_at: datetime(),
                      internal: true,
                      source_authority_sha256: $source_authority_sha256,
                      retirement_authority_sha256: $retirement_authority_sha256,
                      manifest_sha256: $manifest_sha256
                    })
                    CREATE (target)-[:HAS_INTERNAL_CHANGE]->(change)
                    RETURN target.id AS name_id, change.id AS change_id
                    ORDER BY name_id
                    """,
                    rows=lifecycle_rows,
                    reason=reason,
                    run_id=run_id,
                    source_authority_sha256=source_sha256,
                    retirement_authority_sha256=retirement_authority_sha256,
                    manifest_sha256=computed_hash,
                )
                if [row["name_id"] for row in retired] != target_ids:
                    raise SignedDualAuthorityRetirementConflict(
                        "lifecycle compare-and-set changed during retirement"
                    )
                transaction.commit()
                return {
                    "schema": SIGNED_DUAL_AUTHORITY_RETIREMENT_RECEIPT_SCHEMA,
                    "outcome": "applied",
                    "dry_run": False,
                    "changed": len(retired),
                    "persistent_writes": len(signed_pairs) * 3 + len(retired) * 4,
                    "sources_reconciled": len(source_rows),
                    "bindings_released": len(signed_pairs),
                    "projections_released": len(signed_pairs),
                    "superseded": len(retired),
                    "ledger_rows": len(retired),
                    "counts": counts,
                    "manifest": manifest,
                    "manifest_sha256": computed_hash,
                    "change_ids": [row["change_id"] for row in retired],
                }
            except BaseException:
                if not transaction.closed:
                    transaction.rollback()
                raise
    finally:
        if own_client:
            client.close()


def _apply_catalog_source_dispositions(
    adjudication: dict[str, Any],
    *,
    mutation_kind: str | None,
    guard_set: tuple[str, ...] | None,
    reason: str,
    apply: bool = False,
    manifest_sha256: str | None = None,
    run_id: str | None = None,
    admitted_subset: bool = False,
    structural_authority: dict[str, Any] | None = None,
    structural_authority_sha256: str | None = None,
    gc: Any | None = None,
) -> dict[str, Any]:
    """Apply signed one-survivor source dispositions under exact graph CAS."""
    from imas_codex.standard_names.graph_ops import (
        SIGNED_SOURCE_DISPOSITION_RECEIPT_SCHEMA,
        SignedSourceDispositionConflict,
        _authority_payload_hash,
        _lock_signed_source_disposition_authority,
        _signed_source_disposition_counts,
        _signed_source_disposition_execution_authority,
        _TransactionQuery,
        _validate_signed_source_adjudication,
        _validate_structural_legitimacy_authority,
    )

    if mutation_kind != _CATALOG_DISPOSITION_MUTATION:
        raise SignedManifestAuthorityError(
            "catalog disposition requires its exact compound mutation"
        )
    if guard_set != _CATALOG_DISPOSITION_GUARDS:
        raise SignedManifestAuthorityError(
            "catalog disposition requires its exact signed guard set"
        )
    if not reason.strip():
        raise ValueError("source disposition requires a non-empty reason")
    if apply and manifest_sha256 is None:
        raise ValueError("apply requires manifest_sha256")
    if manifest_sha256 is not None and not _SHA256_RE.fullmatch(manifest_sha256):
        raise ValueError("manifest_sha256 must be a lowercase SHA-256 digest")
    adjudication_sha256, adjudication_row_set_sha256, adjudication_rows = (
        _validate_signed_source_adjudication(adjudication)
    )
    structural_authority_sha256, structural_target_ids = (
        _validate_structural_legitimacy_authority(
            structural_authority,
            structural_authority_sha256,
            adjudication_rows,
        )
    )
    adjudicated_dispositions = {
        row["source_id"]: row["surviving_target"] for row in adjudication_rows
    }
    expected_dispositions = adjudicated_dispositions
    event_id = (
        f"sn-change:signed-source-disposition:{manifest_sha256}"
        if manifest_sha256
        else None
    )

    own = gc is None
    client: Any = GraphClient() if own else gc
    try:
        with client.session() as session:
            transaction = session.begin_transaction()
            query_handle = _TransactionQuery(transaction)
            try:
                if apply:
                    replay_rows = query_handle.query(
                        """
                        OPTIONAL MATCH (change:StandardNameChange {id: $event_id})
                        RETURN change.to_name AS disposition_json
                        """,
                        event_id=event_id,
                    )
                    disposition_json = (
                        replay_rows[0].get("disposition_json") if replay_rows else None
                    )
                    if disposition_json is not None:
                        recorded_dispositions = json.loads(disposition_json)
                        if admitted_subset:
                            recorded_matches_adjudication = bool(
                                recorded_dispositions
                            ) and all(
                                adjudicated_dispositions.get(source_id) == target_id
                                for source_id, target_id in recorded_dispositions.items()
                            )
                        else:
                            recorded_matches_adjudication = (
                                recorded_dispositions == adjudicated_dispositions
                            )
                        if not recorded_matches_adjudication:
                            raise SignedSourceDispositionConflict(
                                "recorded disposition covers a different source set"
                            )
                        expected_dispositions = recorded_dispositions
                        current_rows = query_handle.query(
                            """
                            UNWIND $rows AS expected
                            MATCH (source:StandardNameSource {id: expected.source_id})
                            OPTIONAL MATCH (source)-[:PRODUCED_NAME]->
                              (target:StandardName)
                            WHERE NOT coalesce(target.name_stage, '') IN
                              ['superseded', 'exhausted', 'contested']
                            WITH expected, source,
                                 collect(DISTINCT target.id) AS target_ids
                            RETURN collect({source_id: expected.source_id,
                              scalar: source.produced_sn_id,
                              target_ids: target_ids,
                              expected_target: expected.keep_target_id}) AS rows
                            """,
                            rows=[
                                {
                                    "source_id": source_id,
                                    "keep_target_id": target_id,
                                }
                                for source_id, target_id in expected_dispositions.items()
                            ],
                        )
                        current = current_rows[0].get("rows") if current_rows else []
                        if len(current) != len(expected_dispositions) or any(
                            row["scalar"] != row["expected_target"]
                            or row["target_ids"] != [row["expected_target"]]
                            for row in current
                        ):
                            raise SignedSourceDispositionConflict(
                                "recorded disposition has lost its postcondition"
                            )
                        transaction.rollback()
                        return {
                            "schema": SIGNED_SOURCE_DISPOSITION_RECEIPT_SCHEMA,
                            "outcome": "already_applied",
                            "dry_run": False,
                            "changed": 0,
                            "manifest_sha256": manifest_sha256,
                        }

                rows, manifest, actions, refusals = (
                    _signed_source_disposition_execution_authority(
                        query_handle,
                        adjudication_rows,
                        adjudication_sha256,
                        adjudication_row_set_sha256,
                        reason,
                        admitted_subset=admitted_subset,
                        structural_authority_sha256=structural_authority_sha256,
                        structural_target_ids=structural_target_ids,
                    )
                )
                expected_dispositions = {
                    row["source_id"]: row["surviving_target"] for row in rows
                }
                computed_hash = _authority_payload_hash(manifest)
                counts = _signed_source_disposition_counts(rows, actions, refusals)
                if apply and computed_hash != manifest_sha256:
                    raise SignedSourceDispositionConflict(
                        "fresh source-disposition manifest does not match signed hash"
                    )
                if refusals:
                    transaction.rollback()
                    return {
                        "schema": SIGNED_SOURCE_DISPOSITION_RECEIPT_SCHEMA,
                        "outcome": "refused",
                        "dry_run": not apply,
                        "changed": 0,
                        "would_change": 0,
                        "counts": counts,
                        "refusals": refusals,
                        "manifest": manifest,
                        "manifest_sha256": computed_hash,
                    }
                if not apply:
                    transaction.rollback()
                    return {
                        "schema": SIGNED_SOURCE_DISPOSITION_RECEIPT_SCHEMA,
                        "outcome": "would_apply",
                        "dry_run": True,
                        "changed": 0,
                        "would_change": 1,
                        "counts": counts,
                        "refusals": [],
                        "manifest": manifest,
                        "manifest_sha256": computed_hash,
                    }

                _lock_signed_source_disposition_authority(query_handle, manifest)
                (
                    locked_rows,
                    locked_manifest,
                    locked_actions,
                    locked_refusals,
                ) = _signed_source_disposition_execution_authority(
                    query_handle,
                    adjudication_rows,
                    adjudication_sha256,
                    adjudication_row_set_sha256,
                    reason,
                    admitted_subset=admitted_subset,
                    structural_authority_sha256=structural_authority_sha256,
                    structural_target_ids=structural_target_ids,
                )
                if (
                    [row["source_id"] for row in locked_rows]
                    != [row["source_id"] for row in rows]
                    or locked_refusals
                    or _authority_payload_hash(locked_manifest) != computed_hash
                ):
                    raise SignedSourceDispositionConflict(
                        "source disposition authority changed while acquiring locks"
                    )
                actions = locked_actions
                scalar_rows = [
                    {
                        "source_id": action["source_id"],
                        "source_element_id": action["source_element_id"],
                        "prior_scalar_target": action["prior_scalar_target"],
                        "keep_target_id": action["keep_target_id"],
                    }
                    for action in actions
                ]
                scalar_updates = query_handle.query(
                    """
                    UNWIND $rows AS expected
                    MATCH (source:StandardNameSource {id: expected.source_id})
                    WHERE elementId(source) = expected.source_element_id
                      AND ((source.produced_sn_id IS NULL
                            AND expected.prior_scalar_target IS NULL)
                           OR source.produced_sn_id = expected.prior_scalar_target)
                      AND source.claimed_at IS NULL
                      AND source.claim_token IS NULL
                    SET source.produced_sn_id = expected.keep_target_id
                    RETURN collect(source.id) AS ids
                    """,
                    rows=scalar_rows,
                )
                updated_ids = sorted(scalar_updates[0].get("ids") or [])
                if updated_ids != sorted(expected_dispositions):
                    raise SignedSourceDispositionConflict(
                        "source scalar or claim compare-and-set changed"
                    )

                mutation_rows = [
                    {
                        "source_id": action["source_id"],
                        "keep_target_id": action["keep_target_id"],
                        "remove_target_id": removal["target_id"],
                        "binding_element_id": removal["binding_element_id"],
                        "projection_element_id": removal["projection_element_id"],
                        "backing_element_id": removal["backing_element_id"],
                    }
                    for action in actions
                    for removal in action["removals"]
                ]
                mutated = query_handle.query(
                    """
                    UNWIND $rows AS expected
                    MATCH (source:StandardNameSource {id: expected.source_id})
                    MATCH (source)-[binding:PRODUCED_NAME]->
                      (removed:StandardName {id: expected.remove_target_id})
                    WHERE elementId(binding) = expected.binding_element_id
                      AND source.produced_sn_id = expected.keep_target_id
                      AND source.claimed_at IS NULL
                      AND source.claim_token IS NULL
                      AND EXISTS {
                        (source)-[:PRODUCED_NAME]->
                          (:StandardName {id: expected.keep_target_id})
                      }
                    MATCH (backing)-[projection:HAS_STANDARD_NAME]->(removed)
                    WHERE elementId(backing) = expected.backing_element_id
                      AND elementId(projection) = expected.projection_element_id
                    DELETE binding, projection
                    RETURN collect({source_id: expected.source_id,
                      binding_element_id: expected.binding_element_id,
                      projection_element_id: expected.projection_element_id}) AS rows
                    """,
                    rows=mutation_rows,
                )
                actual_mutations = mutated[0].get("rows") if mutated else []
                if len(actual_mutations) != len(mutation_rows):
                    raise SignedSourceDispositionConflict(
                        "signed binding closure changed during deletion"
                    )

                removed_target_ids = sorted(
                    {row["remove_target_id"] for row in mutation_rows}
                )
                remaining_bindings = query_handle.query(
                    """
                    UNWIND $target_ids AS target_id
                    MATCH (target:StandardName {id: target_id})
                    OPTIONAL MATCH (remaining:StandardNameSource)
                      -[:PRODUCED_NAME]->(target)
                    WHERE remaining.status <> 'stale'
                    WITH target_id, target, count(remaining) AS incoming_bindings
                    OPTIONAL MATCH (child:StandardName)-[:HAS_PARENT]->(target)
                    WHERE child.name_stage <> 'superseded'
                      AND NOT (coalesce(child.status, '') IN
                        ['deprecated', 'superseded'])
                    WITH target_id, incoming_bindings,
                         count(child) AS live_direct_children
                    RETURN collect({target_id: target_id,
                      incoming_bindings: incoming_bindings,
                      live_direct_children: live_direct_children}) AS rows
                    """,
                    target_ids=removed_target_ids,
                )
                remaining_rows = (
                    remaining_bindings[0].get("rows") if remaining_bindings else []
                )
                structurally_exempt_targets = {
                    row["target_id"] for row in manifest["structural_exemptions"]
                }
                if len(remaining_rows) != len(removed_target_ids) or any(
                    int(row["incoming_bindings"] or 0) == 0
                    and (
                        row["target_id"] not in structurally_exempt_targets
                        or int(row["live_direct_children"] or 0) == 0
                    )
                    for row in remaining_rows
                ):
                    raise SignedSourceDispositionConflict(
                        "removed target lost its final live authority"
                    )

                target_ids = sorted(
                    {
                        target_id
                        for action in actions
                        for target_id in [
                            action["keep_target_id"],
                            *(item["target_id"] for item in action["removals"]),
                        ]
                    }
                )
                query_handle.query(
                    """
                    UNWIND $target_ids AS target_id
                    MATCH (target:StandardName {id: target_id})
                    OPTIONAL MATCH (source:StandardNameSource)-[:PRODUCED_NAME]->(target)
                    OPTIONAL MATCH (source)-[:FROM_DD_PATH]->(dd:IMASNode)
                    OPTIONAL MATCH (source)-[:FROM_SIGNAL]->(signal:FacilitySignal)
                    WITH target, collect(DISTINCT CASE
                      WHEN source IS NULL THEN null
                      WHEN dd IS NOT NULL THEN 'dd:' + dd.id
                      WHEN signal IS NOT NULL THEN signal.id
                      WHEN source.source_type = 'derived'
                        AND source.source_id STARTS WITH 'derived:'
                      THEN source.source_id ELSE source.id END) AS paths
                    SET target.source_paths = [path IN paths WHERE path IS NOT NULL]
                    """,
                    target_ids=target_ids,
                )
                change_rows = query_handle.query(
                    """
                    MERGE (change:StandardNameChange {id: $event_id})
                    ON CREATE SET change.from_name = $removed_json,
                                  change.to_name = $disposition_json,
                                  change.operation =
                                    'apply_adjudicated_source_dispositions',
                                  change.reason = $reason,
                                  change.origin = 'semantic_source_reconciliation',
                                  change.run_id = $run_id,
                                  change.changed_at = datetime(),
                                  change.internal = true,
                                  change.manifest_sha256 = $manifest_sha256
                    WITH change
                    UNWIND $keep_target_ids AS keep_target_id
                    MATCH (kept:StandardName {id: keep_target_id})
                    MERGE (kept)-[:HAS_INTERNAL_CHANGE]->(change)
                    RETURN DISTINCT change.id AS change_id
                    """,
                    event_id=f"sn-change:signed-source-disposition:{computed_hash}",
                    removed_json=json.dumps(
                        {
                            action["source_id"]: [
                                item["target_id"] for item in action["removals"]
                            ]
                            for action in actions
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    disposition_json=json.dumps(
                        expected_dispositions, sort_keys=True, separators=(",", ":")
                    ),
                    reason=reason,
                    run_id=run_id,
                    manifest_sha256=computed_hash,
                    keep_target_ids=sorted(set(expected_dispositions.values())),
                )
                if len(change_rows) != 1:
                    raise SignedSourceDispositionConflict(
                        "source disposition receipt was not written exactly once"
                    )
                transaction.commit()
                return {
                    "schema": SIGNED_SOURCE_DISPOSITION_RECEIPT_SCHEMA,
                    "outcome": "applied",
                    "dry_run": False,
                    "changed": 1,
                    "sources_reconciled": len(actions),
                    "bindings_removed": len(mutation_rows),
                    "projections_removed": len(mutation_rows),
                    "counts": counts,
                    "refusals": [],
                    "manifest": manifest,
                    "manifest_sha256": computed_hash,
                    "change_id": change_rows[0]["change_id"],
                }
            except BaseException:
                if not transaction.closed:
                    transaction.rollback()
                raise
    finally:
        if own:
            client.close()


INELIGIBLE_SOURCE_RETIREMENT_MANIFEST_SCHEMA = (
    "imas-codex.ineligible-standard-name-source-retirement-manifest.v1"
)
INELIGIBLE_SOURCE_RETIREMENT_RECEIPT_SCHEMA = (
    "imas-codex.ineligible-standard-name-source-retirement-receipt.v1"
)


class IneligibleSourceRetirementConflict(RuntimeError):
    """The signed ineligible-source closure no longer matches graph authority."""


def _ineligible_source_retirement_authority(
    query_handle: _TransactionQuery,
    source_ids: list[str],
    reason: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, str]]]:
    from imas_codex.core.node_categories import SN_SOURCE_CATEGORIES

    rows = query_handle.query(
        """
        UNWIND $source_ids AS requested_id
        OPTIONAL MATCH (source:StandardNameSource {id: requested_id})
        RETURN requested_id,
               elementId(source) AS source_element_id,
               properties(source) AS source_properties,
               CASE WHEN source IS NULL THEN [] ELSE
                 [(source)-[binding:PRODUCED_NAME]->(target:StandardName) |
                   {element_id: elementId(binding),
                    properties: properties(binding),
                    target_element_id: elementId(target),
                    target_id: target.id,
                    target_properties: properties(target)}]
               END AS bindings,
               CASE WHEN source IS NULL THEN [] ELSE
                 [(source)-[origin:FROM_DD_PATH|FROM_SIGNAL]->(backing) |
                   {element_id: elementId(backing),
                    labels: labels(backing),
                    properties: properties(backing),
                    origin_element_id: elementId(origin),
                    origin_type: type(origin),
                    origin_properties: properties(origin),
                    projections: [(backing)-[projection:HAS_STANDARD_NAME]->
                      (projected:StandardName) |
                      {element_id: elementId(projection),
                       properties: properties(projection),
                       target_id: projected.id,
                       target_properties: properties(projected)}]}]
               END AS backings
        ORDER BY requested_id
        """,
        source_ids=source_ids,
    )
    eligible_categories = set(SN_SOURCE_CATEGORIES)
    participants: list[dict[str, Any]] = []
    actions: list[dict[str, Any]] = []
    refusals: list[dict[str, str]] = []
    for raw in rows:
        row = dict(raw)
        bindings = sorted(
            (dict(binding) for binding in row.get("bindings") or []),
            key=lambda binding: (binding["target_id"], binding["element_id"]),
        )
        backings = sorted(
            (
                {
                    **dict(backing),
                    "labels": sorted(backing.get("labels") or []),
                    "projections": sorted(
                        (dict(item) for item in backing.get("projections") or []),
                        key=lambda item: (item["target_id"], item["element_id"]),
                    ),
                }
                for backing in row.get("backings") or []
            ),
            key=lambda backing: backing["element_id"],
        )
        participant = {
            "source_id": row["requested_id"],
            "source_element_id": row.get("source_element_id"),
            "source_properties": row.get("source_properties"),
            "bindings": bindings,
            "backings": backings,
        }
        participants.append(participant)
        properties = row.get("source_properties")
        refusal: str | None = None
        if properties is None:
            refusal = "source does not exist"
        elif properties.get("source_type") != "dd":
            refusal = "source is not DD-backed"
        elif (
            properties.get("claimed_at") is not None
            or properties.get("claim_token") is not None
        ):
            refusal = "source has an active claim"
        elif len(backings) != 1 or backings[0].get("labels") != ["IMASNode"]:
            refusal = "source does not have exactly one DD backing node"
        elif backings[0].get("properties", {}).get("node_category") is None:
            refusal = "backing DD node has no node_category authority"
        elif backings[0]["properties"]["node_category"] in eligible_categories:
            category = backings[0]["properties"]["node_category"]
            refusal = f"backing DD node category {category!r} is SN-eligible"
        elif not bindings:
            refusal = "source has no PRODUCED_NAME bindings to retire"
        if refusal is not None:
            refusals.append({"source_id": row["requested_id"], "reason": refusal})
            continue

        target_ids = {binding["target_id"] for binding in bindings}
        projections = [
            {
                **projection,
                "backing_element_id": backings[0]["element_id"],
            }
            for projection in backings[0]["projections"]
            if projection["target_id"] in target_ids
        ]
        actions.append(
            {
                "source_id": row["requested_id"],
                "backing_element_id": backings[0]["element_id"],
                "backing_category": backings[0]["properties"]["node_category"],
                "bindings": bindings,
                "projections": projections,
            }
        )

    manifest = {
        "schema": INELIGIBLE_SOURCE_RETIREMENT_MANIFEST_SCHEMA,
        "operation": "retire_ineligible_standard_name_sources",
        "reason": reason,
        "source_ids": source_ids,
        "participants": participants,
        "actions": actions,
        "refusals": refusals,
    }
    return manifest, actions, refusals


def _ineligible_source_retirement_counts(
    source_ids: list[str],
    actions: list[dict[str, Any]],
    refusals: list[dict[str, str]],
) -> dict[str, int]:
    return {
        "requested": len(source_ids),
        "admitted": len(actions),
        "refused": len(refusals),
        "bindings_to_detach": sum(len(action["bindings"]) for action in actions),
        "projections_to_detach": sum(len(action["projections"]) for action in actions),
    }


def _apply_ineligible_source_retirement(
    source_ids: list[str],
    *,
    mutation_kind: str | None,
    guard_set: tuple[str, ...] | None,
    reason: str,
    apply: bool = False,
    manifest_sha256: str | None = None,
    run_id: str | None = None,
    gc: Any | None = None,
) -> dict[str, Any]:
    """Retire DD sources whose backing category cannot realize a name.

    Preview signs the complete source, binding, backing, and projection closure.
    Apply detaches every signed name realization, clears the scalar mirror, and
    parks the source as ``not_physical_quantity`` in one transaction. Names that
    lose their final producing source are returned for the orphan workflow and
    are never superseded here.
    """
    from imas_codex.standard_names.graph_ops import (
        _authority_payload_hash,
        _lock_scalar_selected_dedup_authority,
    )

    if mutation_kind != _INELIGIBLE_SOURCE_MUTATION:
        raise SignedManifestAuthorityError(
            "ineligible source retirement requires its exact compound mutation"
        )
    if guard_set != _INELIGIBLE_SOURCE_GUARDS:
        raise SignedManifestAuthorityError(
            "ineligible source retirement requires its exact signed guard set"
        )
    requested = sorted(set(source_ids))
    if not requested:
        raise ValueError("ineligible source retirement requires at least one source")
    if len(requested) != len(source_ids):
        raise ValueError("ineligible source retirement requires unique source ids")
    if not reason.strip():
        raise ValueError("ineligible source retirement requires a non-empty reason")
    if apply and manifest_sha256 is None:
        raise ValueError("apply requires manifest_sha256")
    if manifest_sha256 is not None and not _SHA256_RE.fullmatch(manifest_sha256):
        raise ValueError("manifest_sha256 must be a lowercase SHA-256 digest")

    own = gc is None
    client: Any = GraphClient() if own else gc
    event_id = (
        f"sn-change:ineligible-source-retirement:{manifest_sha256}"
        if manifest_sha256
        else None
    )
    try:
        with client.session() as session:
            transaction = session.begin_transaction()
            query_handle = _TransactionQuery(transaction)
            try:
                if apply:
                    replay_rows = query_handle.query(
                        """
                        OPTIONAL MATCH (change:StandardNameChange {id: $event_id})
                        RETURN change.to_name AS disposition_json
                        """,
                        event_id=event_id,
                    )
                    disposition_json = (
                        replay_rows[0].get("disposition_json") if replay_rows else None
                    )
                    if disposition_json is not None:
                        disposition = json.loads(disposition_json)
                        if sorted(disposition.get("source_ids") or []) != requested:
                            raise IneligibleSourceRetirementConflict(
                                "recorded retirement covers a different source set"
                            )
                        postcondition = query_handle.query(
                            """
                            UNWIND $source_ids AS source_id
                            MATCH (source:StandardNameSource {id: source_id})
                            RETURN collect({source_id: source.id,
                              status: source.status,
                              scalar: source.produced_sn_id,
                              bindings: COUNT {
                                (source)-[:PRODUCED_NAME]->(:StandardName)
                              }}) AS rows
                            """,
                            source_ids=requested,
                        )
                        current = postcondition[0].get("rows") if postcondition else []
                        if len(current) != len(requested) or any(
                            row["status"] != "not_physical_quantity"
                            or row["scalar"] is not None
                            or int(row["bindings"] or 0) != 0
                            for row in current
                        ):
                            raise IneligibleSourceRetirementConflict(
                                "recorded retirement has lost its postcondition"
                            )
                        transaction.rollback()
                        return {
                            "schema": INELIGIBLE_SOURCE_RETIREMENT_RECEIPT_SCHEMA,
                            "outcome": "already_applied",
                            "dry_run": False,
                            "changed": 0,
                            "manifest_sha256": manifest_sha256,
                            "orphaned_names": sorted(
                                disposition.get("orphaned_names") or []
                            ),
                        }

                manifest, actions, refusals = _ineligible_source_retirement_authority(
                    query_handle, requested, reason
                )
                computed_hash = _authority_payload_hash(manifest)
                counts = _ineligible_source_retirement_counts(
                    requested, actions, refusals
                )
                if apply and computed_hash != manifest_sha256:
                    raise IneligibleSourceRetirementConflict(
                        "fresh ineligible-source manifest does not match signed hash"
                    )
                if refusals:
                    transaction.rollback()
                    return {
                        "schema": INELIGIBLE_SOURCE_RETIREMENT_RECEIPT_SCHEMA,
                        "outcome": "refused",
                        "dry_run": not apply,
                        "changed": 0,
                        "would_change": 0,
                        "counts": counts,
                        "refusals": refusals,
                        "manifest": manifest,
                        "manifest_sha256": computed_hash,
                    }
                if not apply:
                    transaction.rollback()
                    return {
                        "schema": INELIGIBLE_SOURCE_RETIREMENT_RECEIPT_SCHEMA,
                        "outcome": "would_apply",
                        "dry_run": True,
                        "changed": 0,
                        "would_change": 1,
                        "counts": counts,
                        "manifest": manifest,
                        "manifest_sha256": computed_hash,
                    }

                _lock_scalar_selected_dedup_authority(query_handle, manifest)
                locked_manifest, locked_actions, locked_refusals = (
                    _ineligible_source_retirement_authority(
                        query_handle, requested, reason
                    )
                )
                if (
                    locked_refusals
                    or _authority_payload_hash(locked_manifest) != computed_hash
                ):
                    raise IneligibleSourceRetirementConflict(
                        "source authority changed while acquiring participant locks"
                    )
                actions = locked_actions
                binding_rows = [
                    {
                        "source_id": action["source_id"],
                        "target_id": binding["target_id"],
                        "element_id": binding["element_id"],
                    }
                    for action in actions
                    for binding in action["bindings"]
                ]
                projection_rows = [
                    {
                        "backing_element_id": projection["backing_element_id"],
                        "target_id": projection["target_id"],
                        "element_id": projection["element_id"],
                    }
                    for action in actions
                    for projection in action["projections"]
                ]
                deleted_bindings = query_handle.query(
                    """
                    UNWIND $rows AS expected
                    MATCH (source:StandardNameSource {id: expected.source_id})
                          -[binding:PRODUCED_NAME]->
                          (:StandardName {id: expected.target_id})
                    WHERE elementId(binding) = expected.element_id
                    DELETE binding
                    RETURN collect(expected.element_id) AS ids
                    """,
                    rows=binding_rows,
                )
                actual_binding_ids = sorted(
                    deleted_bindings[0].get("ids") or [] if deleted_bindings else []
                )
                if actual_binding_ids != sorted(
                    row["element_id"] for row in binding_rows
                ):
                    raise IneligibleSourceRetirementConflict(
                        "signed source binding set changed during retirement"
                    )
                if projection_rows:
                    deleted_projections = query_handle.query(
                        """
                        UNWIND $rows AS expected
                        MATCH (backing)-[projection:HAS_STANDARD_NAME]->
                              (:StandardName {id: expected.target_id})
                        WHERE elementId(backing) = expected.backing_element_id
                          AND elementId(projection) = expected.element_id
                        DELETE projection
                        RETURN collect(expected.element_id) AS ids
                        """,
                        rows=projection_rows,
                    )
                    actual_projection_ids = sorted(
                        deleted_projections[0].get("ids") or []
                        if deleted_projections
                        else []
                    )
                    if actual_projection_ids != sorted(
                        row["element_id"] for row in projection_rows
                    ):
                        raise IneligibleSourceRetirementConflict(
                            "signed backing projection set changed during retirement"
                        )

                participant_by_source = {
                    row["source_id"]: row for row in manifest["participants"]
                }
                state_rows = [
                    {
                        "source_id": action["source_id"],
                        "source_element_id": participant_by_source[action["source_id"]][
                            "source_element_id"
                        ],
                        "expected_status": participant_by_source[action["source_id"]][
                            "source_properties"
                        ].get("status"),
                        "expected_scalar": participant_by_source[action["source_id"]][
                            "source_properties"
                        ].get("produced_sn_id"),
                        "backing_element_id": action["backing_element_id"],
                        "backing_category": action["backing_category"],
                    }
                    for action in actions
                ]
                retired = query_handle.query(
                    """
                    UNWIND $rows AS expected
                    MATCH (source:StandardNameSource {id: expected.source_id})
                          -[:FROM_DD_PATH]->(backing:IMASNode)
                    WHERE elementId(source) = expected.source_element_id
                      AND elementId(backing) = expected.backing_element_id
                      AND source.status = expected.expected_status
                      AND source.produced_sn_id = expected.expected_scalar
                      AND source.claimed_at IS NULL
                      AND source.claim_token IS NULL
                      AND backing.node_category = expected.backing_category
                      AND COUNT {
                        (source)-[:PRODUCED_NAME]->(:StandardName)
                      } = 0
                    SET source.status = 'not_physical_quantity',
                        source.produced_sn_id = null,
                        source.claimed_at = null,
                        source.claim_token = null,
                        source.skip_reason = 'dd_node_category_ineligible',
                        source.skip_reason_detail =
                          'Backing DD node category ' + expected.backing_category +
                          ' cannot realize a StandardName'
                    RETURN collect(source.id) AS ids
                    """,
                    rows=state_rows,
                )
                actual_retired = sorted(retired[0].get("ids") or []) if retired else []
                if actual_retired != requested:
                    raise IneligibleSourceRetirementConflict(
                        "source lifecycle compare-and-set changed during retirement"
                    )

                target_ids = sorted(
                    {
                        binding["target_id"]
                        for action in actions
                        for binding in action["bindings"]
                    }
                )
                query_handle.query(
                    """
                    UNWIND $target_ids AS target_id
                    MATCH (target:StandardName {id: target_id})
                    OPTIONAL MATCH (source:StandardNameSource)
                      -[:PRODUCED_NAME]->(target)
                    OPTIONAL MATCH (source)-[:FROM_DD_PATH]->(dd:IMASNode)
                    OPTIONAL MATCH (source)-[:FROM_SIGNAL]->(signal:FacilitySignal)
                    WITH target, collect(DISTINCT CASE
                      WHEN source IS NULL THEN null
                      WHEN dd IS NOT NULL THEN 'dd:' + dd.id
                      WHEN signal IS NOT NULL THEN signal.id
                      WHEN source.source_type = 'derived'
                        AND source.source_id STARTS WITH 'derived:'
                      THEN source.source_id ELSE source.id END) AS paths
                    SET target.source_paths = [path IN paths WHERE path IS NOT NULL]
                    """,
                    target_ids=target_ids,
                )
                orphan_rows = query_handle.query(
                    """
                    UNWIND $target_ids AS target_id
                    MATCH (target:StandardName {id: target_id})
                    WHERE NOT coalesce(target.name_stage, '') IN
                      ['superseded', 'exhausted']
                      AND NOT EXISTS {
                        (:StandardNameSource)-[:PRODUCED_NAME]->(target)
                      }
                    RETURN target.id AS id ORDER BY id
                    """,
                    target_ids=target_ids,
                )
                orphaned_names = [row["id"] for row in orphan_rows]
                disposition = {
                    "source_ids": requested,
                    "status": "not_physical_quantity",
                    "orphaned_names": orphaned_names,
                }
                change_rows = query_handle.query(
                    """
                    MERGE (change:StandardNameChange {id: $event_id})
                    ON CREATE SET change.from_name = $removed_json,
                                  change.to_name = $disposition_json,
                                  change.operation =
                                    'retire_ineligible_standard_name_sources',
                                  change.reason = $reason,
                                  change.origin =
                                    'semantic_source_reconciliation',
                                  change.run_id = $run_id,
                                  change.changed_at = datetime(),
                                  change.internal = true,
                                  change.manifest_sha256 = $manifest_sha256
                    WITH change
                    UNWIND $target_ids AS target_id
                    MATCH (target:StandardName {id: target_id})
                    MERGE (target)-[:HAS_INTERNAL_CHANGE]->(change)
                    RETURN DISTINCT change.id AS change_id
                    """,
                    event_id=(
                        f"sn-change:ineligible-source-retirement:{computed_hash}"
                    ),
                    removed_json=json.dumps(
                        {
                            action["source_id"]: [
                                binding["target_id"] for binding in action["bindings"]
                            ]
                            for action in actions
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    disposition_json=json.dumps(
                        disposition, sort_keys=True, separators=(",", ":")
                    ),
                    reason=reason,
                    run_id=run_id,
                    manifest_sha256=computed_hash,
                    target_ids=target_ids,
                )
                if len(change_rows) != 1:
                    raise IneligibleSourceRetirementConflict(
                        "retirement change receipt was not written exactly once"
                    )
                transaction.commit()
                return {
                    "schema": INELIGIBLE_SOURCE_RETIREMENT_RECEIPT_SCHEMA,
                    "outcome": "applied",
                    "dry_run": False,
                    "changed": 1,
                    "sources_retired": len(actions),
                    "bindings_detached": len(binding_rows),
                    "projections_detached": len(projection_rows),
                    "orphaned_names": orphaned_names,
                    "counts": counts,
                    "manifest": manifest,
                    "manifest_sha256": computed_hash,
                    "change_id": change_rows[0]["change_id"],
                }
            except BaseException:
                if not transaction.closed:
                    transaction.rollback()
                raise
    finally:
        if own:
            client.close()


def _apply_semantic_mirror_repair(
    source_ids: list[str],
    *,
    reason: str,
    apply: bool = False,
    manifest_sha256: str | None = None,
    run_id: str | None = None,
    gc: Any | None = None,
) -> dict[str, Any]:
    """Repair scalar and upstream projections from sole live-edge authority."""
    from imas_codex.standard_names.graph_ops import (
        _PROJECTED_SEMANTIC_SOURCE_TYPES,
        SEMANTIC_MIRROR_REPAIR_RECEIPT_SCHEMA,
        SemanticMirrorRepairConflict,
        _authority_payload_hash,
        _lock_semantic_mirror_repair_authority,
        _semantic_mirror_repair_authority,
        _semantic_mirror_repair_counts,
        _TransactionQuery,
    )

    requested = sorted(set(source_ids))
    if not requested:
        raise ValueError("semantic mirror repair requires at least one source id")
    if len(requested) != len(source_ids):
        raise ValueError("semantic mirror repair requires unique source ids")
    if not reason.strip():
        raise ValueError("semantic mirror repair requires a non-empty reason")
    if apply and manifest_sha256 is None:
        raise ValueError("apply requires manifest_sha256")
    if manifest_sha256 is not None and not _SHA256_RE.fullmatch(manifest_sha256):
        raise ValueError("manifest_sha256 must be a lowercase SHA-256 digest")

    event_id = (
        f"sn-change:semantic-mirror-repair:{manifest_sha256}"
        if manifest_sha256
        else None
    )
    own = gc is None
    client: Any = GraphClient() if own else gc
    try:
        with client.session() as session:
            transaction = session.begin_transaction()
            query_handle = _TransactionQuery(transaction)
            try:
                if apply:
                    replay_rows = query_handle.query(
                        """
                        OPTIONAL MATCH (change:StandardNameChange {id: $event_id})
                        RETURN change.to_name AS target_json
                        """,
                        event_id=event_id,
                    )
                    target_json = (
                        replay_rows[0].get("target_json") if replay_rows else None
                    )
                    if target_json is not None:
                        expected_targets = json.loads(target_json)
                        if sorted(expected_targets) != requested:
                            raise SemanticMirrorRepairConflict(
                                "recorded repair covers a different source set"
                            )
                        current_rows = query_handle.query(
                            """
                            UNWIND $rows AS expected
                            MATCH (source:StandardNameSource {id: expected.source_id})
                            OPTIONAL MATCH (source)-[:PRODUCED_NAME]->
                              (target:StandardName)
                            WHERE NOT coalesce(target.name_stage, '') IN
                              ['superseded', 'exhausted', 'contested']
                            WITH expected, source,
                                 collect(DISTINCT target.id) AS target_ids
                            OPTIONAL MATCH (source)-[:FROM_DD_PATH|FROM_SIGNAL]->
                              (backing)
                            OPTIONAL MATCH (backing)-[:HAS_STANDARD_NAME]->
                              (mapped:StandardName)
                            WITH expected, source, target_ids,
                                 collect(DISTINCT mapped.id) AS mapped_ids
                            RETURN collect({source_id: expected.source_id,
                              source_type: source.source_type,
                              scalar: source.produced_sn_id,
                              target_ids: target_ids,
                              mapped_ids: mapped_ids,
                              expected_target: expected.target_id}) AS rows
                            """,
                            rows=[
                                {"source_id": source_id, "target_id": target_id}
                                for source_id, target_id in expected_targets.items()
                            ],
                        )
                        current = current_rows[0].get("rows") if current_rows else []
                        if len(current) != len(expected_targets) or any(
                            row["target_ids"] != [row["expected_target"]]
                            or row["scalar"] != row["expected_target"]
                            or (
                                row["source_type"] in _PROJECTED_SEMANTIC_SOURCE_TYPES
                                and row["expected_target"] not in row["mapped_ids"]
                            )
                            for row in current
                        ):
                            raise SemanticMirrorRepairConflict(
                                "recorded repair has lost its postcondition"
                            )
                        transaction.rollback()
                        return {
                            "schema": SEMANTIC_MIRROR_REPAIR_RECEIPT_SCHEMA,
                            "outcome": "already_applied",
                            "dry_run": False,
                            "changed": 0,
                            "manifest_sha256": manifest_sha256,
                        }

                manifest, actions, refusals, already_clean = (
                    _semantic_mirror_repair_authority(query_handle, requested, reason)
                )
                computed_hash = _authority_payload_hash(manifest)
                counts = _semantic_mirror_repair_counts(
                    requested, actions, refusals, already_clean
                )
                if apply and computed_hash != manifest_sha256:
                    raise SemanticMirrorRepairConflict(
                        "fresh semantic-mirror manifest does not match signed hash"
                    )
                if refusals:
                    transaction.rollback()
                    return {
                        "schema": SEMANTIC_MIRROR_REPAIR_RECEIPT_SCHEMA,
                        "outcome": "refused",
                        "dry_run": not apply,
                        "changed": 0,
                        "would_change": 0,
                        "counts": counts,
                        "refusals": refusals,
                        "manifest": manifest,
                        "manifest_sha256": computed_hash,
                    }
                if not actions:
                    transaction.rollback()
                    return {
                        "schema": SEMANTIC_MIRROR_REPAIR_RECEIPT_SCHEMA,
                        "outcome": "already_clean",
                        "dry_run": not apply,
                        "changed": 0,
                        "would_change": 0,
                        "counts": counts,
                        "refusals": [],
                        "manifest": manifest,
                        "manifest_sha256": computed_hash,
                    }
                if not apply:
                    transaction.rollback()
                    return {
                        "schema": SEMANTIC_MIRROR_REPAIR_RECEIPT_SCHEMA,
                        "outcome": "would_apply",
                        "dry_run": True,
                        "changed": 0,
                        "would_change": 1,
                        "counts": counts,
                        "refusals": [],
                        "manifest": manifest,
                        "manifest_sha256": computed_hash,
                    }

                _lock_semantic_mirror_repair_authority(query_handle, manifest)
                locked_manifest, locked_actions, locked_refusals, locked_clean = (
                    _semantic_mirror_repair_authority(query_handle, requested, reason)
                )
                if (
                    locked_refusals
                    or _authority_payload_hash(locked_manifest) != computed_hash
                ):
                    raise SemanticMirrorRepairConflict(
                        "semantic-mirror authority changed while acquiring locks"
                    )
                actions = locked_actions
                already_clean = locked_clean
                scalar_rows = [action for action in actions if action["scalar_change"]]
                if scalar_rows:
                    scalar_updates = query_handle.query(
                        """
                        UNWIND $rows AS expected
                        MATCH (source:StandardNameSource {id: expected.source_id})
                        WHERE elementId(source) = expected.source_element_id
                          AND ((source.produced_sn_id IS NULL
                                AND expected.prior_scalar_target IS NULL)
                               OR source.produced_sn_id =
                                  expected.prior_scalar_target)
                          AND source.claimed_at IS NULL
                          AND source.claim_token IS NULL
                          AND EXISTS {
                            (source)-[:PRODUCED_NAME]->(:StandardName {
                              id: expected.target_id})
                          }
                        SET source.produced_sn_id = expected.target_id
                        RETURN collect(source.id) AS ids
                        """,
                        rows=scalar_rows,
                    )
                    if sorted(scalar_updates[0].get("ids") or []) != sorted(
                        row["source_id"] for row in scalar_rows
                    ):
                        raise SemanticMirrorRepairConflict(
                            "source scalar compare-and-set changed"
                        )

                projection_rows = [
                    {
                        "source_id": action["source_id"],
                        **addition,
                    }
                    for action in actions
                    for addition in action["projection_additions"]
                ]
                if projection_rows:
                    projection_updates = query_handle.query(
                        """
                        UNWIND $rows AS expected
                        MATCH (source:StandardNameSource {id: expected.source_id})
                        MATCH (source)-[:FROM_DD_PATH|FROM_SIGNAL]->(backing)
                        WHERE elementId(backing) = expected.backing_element_id
                          AND source.produced_sn_id = expected.target_id
                          AND source.claimed_at IS NULL
                          AND source.claim_token IS NULL
                        MATCH (target:StandardName {id: expected.target_id})
                        WHERE elementId(target) = expected.target_element_id
                          AND NOT EXISTS {
                            (backing)-[:HAS_STANDARD_NAME]->(target)
                          }
                        CREATE (backing)-[:HAS_STANDARD_NAME]->(target)
                        RETURN collect(source.id) AS ids
                        """,
                        rows=projection_rows,
                    )
                    if sorted(projection_updates[0].get("ids") or []) != sorted(
                        row["source_id"] for row in projection_rows
                    ):
                        raise SemanticMirrorRepairConflict(
                            "backing projection compare-and-set changed"
                        )

                resolutions = [*actions, *already_clean]
                expected_targets = {
                    row["source_id"]: row["target_id"] for row in resolutions
                }
                change_rows = query_handle.query(
                    """
                    MERGE (change:StandardNameChange {id: $event_id})
                    ON CREATE SET change.from_name = $before_json,
                                  change.to_name = $target_json,
                                  change.operation =
                                    'repair_scalar_projection_mismatches',
                                  change.reason = $reason,
                                  change.origin =
                                    'semantic_source_reconciliation',
                                  change.run_id = $run_id,
                                  change.changed_at = datetime(),
                                  change.internal = true,
                                  change.manifest_sha256 = $manifest_sha256
                    WITH change
                    UNWIND $target_ids AS target_id
                    MATCH (target:StandardName {id: target_id})
                    MERGE (target)-[:HAS_INTERNAL_CHANGE]->(change)
                    RETURN DISTINCT change.id AS change_id
                    """,
                    event_id=f"sn-change:semantic-mirror-repair:{computed_hash}",
                    before_json=json.dumps(
                        {
                            action["source_id"]: {
                                "produced_sn_id": action["prior_scalar_target"],
                                "projection_missing": bool(
                                    action["projection_additions"]
                                ),
                            }
                            for action in actions
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    target_json=json.dumps(
                        expected_targets, sort_keys=True, separators=(",", ":")
                    ),
                    reason=reason,
                    run_id=run_id,
                    manifest_sha256=computed_hash,
                    target_ids=sorted(set(expected_targets.values())),
                )
                if len(change_rows) != 1:
                    raise SemanticMirrorRepairConflict(
                        "semantic-mirror receipt was not written exactly once"
                    )
                transaction.commit()
                return {
                    "schema": SEMANTIC_MIRROR_REPAIR_RECEIPT_SCHEMA,
                    "outcome": "applied",
                    "dry_run": False,
                    "changed": 1,
                    "sources_reconciled": len(actions),
                    "scalars_changed": len(scalar_rows),
                    "projections_added": len(projection_rows),
                    "counts": counts,
                    "refusals": [],
                    "manifest": manifest,
                    "manifest_sha256": computed_hash,
                    "change_id": change_rows[0]["change_id"],
                }
            except BaseException:
                if not transaction.closed:
                    transaction.rollback()
                raise
    finally:
        if own:
            client.close()


@retry_on_deadlock()
def apply_signed_manifest(
    authority_path: str | Path | dict[str, Any],
    *legacy_args: Any,
    authority_file_sha256: str | None = None,
    authority_payload_sha256: str | None = None,
    authority_sha256: str | None = None,
    retirement_authority_sha256: str | None = None,
    authority_adapter: str | None = None,
    mutation_kind: str | None = None,
    guard_set: tuple[str, ...] | None = None,
    admitted_subset: bool = False,
    structural_authority: dict[str, Any] | None = None,
    structural_authority_sha256: str | None = None,
    name_ids: list[str] | None = None,
    reason: str,
    apply: bool = False,
    manifest_sha256: str | None = None,
    run_id: str | None = None,
    gc: Any | None = None,
    client_factory: Callable[[], Any] | None = None,
) -> dict[str, Any]:
    """Preview or atomically apply the complete row set in a signed authority.

    ``manifest_sha256`` is the authorization returned by a prior preview.  It
    authorizes only the hash: participant closure, collateral rows, and counter
    baselines are always read again inside this invocation.
    """
    if authority_adapter == _SEMANTIC_MIRROR_ADAPTER:
        if (
            legacy_args
            or not isinstance(authority_path, Sequence)
            or isinstance(authority_path, str | bytes)
        ):
            raise SignedManifestAuthorityError(
                "semantic mirror repair requires an exact source-id sequence"
            )
        if mutation_kind != _SEMANTIC_MIRROR_MUTATION:
            raise SignedManifestAuthorityError(
                "semantic mirror repair requires its exact closed mutation program"
            )
        if guard_set != _SEMANTIC_MIRROR_GUARDS:
            raise SignedManifestAuthorityError(
                "semantic mirror repair requires its exact deterministic guard set"
            )
        return _apply_semantic_mirror_repair(
            list(authority_path),
            reason=reason,
            apply=apply,
            manifest_sha256=manifest_sha256,
            run_id=run_id,
            gc=gc,
        )
    if authority_adapter == _INELIGIBLE_SOURCE_ADAPTER:
        if (
            legacy_args
            or not isinstance(authority_path, Sequence)
            or isinstance(authority_path, str | bytes)
        ):
            raise SignedManifestAuthorityError(
                "ineligible source retirement requires an exact source-id sequence"
            )
        return _apply_ineligible_source_retirement(
            list(authority_path),
            mutation_kind=mutation_kind,
            guard_set=guard_set,
            reason=reason,
            apply=apply,
            manifest_sha256=manifest_sha256,
            run_id=run_id,
            gc=gc,
        )
    if authority_adapter == _CATALOG_DISPOSITION_ADAPTER:
        if legacy_args or not isinstance(authority_path, dict):
            raise SignedManifestAuthorityError(
                "catalog disposition requires one signed adjudication object"
            )
        return _apply_catalog_source_dispositions(
            authority_path,
            mutation_kind=mutation_kind,
            guard_set=guard_set,
            reason=reason,
            apply=apply,
            manifest_sha256=manifest_sha256,
            run_id=run_id,
            admitted_subset=admitted_subset,
            structural_authority=structural_authority,
            structural_authority_sha256=structural_authority_sha256,
            gc=gc,
        )
    if authority_adapter == _DUAL_AUTHORITY_ADAPTER:
        if len(legacy_args) != 1 or not isinstance(legacy_args[0], dict):
            raise SignedManifestAuthorityError(
                "dual-authority retirement requires both signed authorities"
            )
        if not isinstance(authority_path, dict):
            raise SignedManifestAuthorityError(
                "dual-authority retirement requires a source authority object"
            )
        return _apply_dual_authority_retirement(
            authority_path,
            legacy_args[0],
            retirement_authority_sha256=retirement_authority_sha256,
            mutation_kind=mutation_kind,
            guard_set=guard_set,
            reason=reason,
            apply=apply,
            manifest_sha256=manifest_sha256,
            run_id=run_id,
            gc=gc,
        )
    if not reason.strip():
        message = (
            "stale-source detach requires a non-empty reason"
            if authority_adapter == _STALE_SOURCE_ADAPTER
            else "signed-manifest apply requires a non-empty reason"
        )
        raise ValueError(message)
    deterministic_self_heal = authority_adapter == _ERROR_SIBLING_ADAPTER
    if apply and manifest_sha256 is None and not deterministic_self_heal:
        message = (
            "stale-source detach apply requires manifest_sha256"
            if authority_adapter == _STALE_SOURCE_ADAPTER
            else "signed-manifest apply requires manifest_sha256"
        )
        raise ValueError(message)
    if manifest_sha256 is not None:
        _require_sha256(manifest_sha256, "manifest_sha256")
    if authority_adapter == _STALE_SOURCE_ADAPTER:
        if len(legacy_args) != 2:
            raise SignedManifestAuthorityError(
                "stale-source repair requires graph client, authority path, and source ids"
            )
        if gc is not None:
            raise SignedManifestAuthorityError(
                "stale-source repair graph client was supplied twice"
            )
        gc = authority_path
        stale_authority_path, source_ids = legacy_args
        if not isinstance(source_ids, Sequence) or isinstance(source_ids, str | bytes):
            raise SignedManifestAuthorityError(
                "stale-source repair requires an exact source-id sequence"
            )
        authority = _load_stale_source_authority(
            stale_authority_path,
            source_ids,
            mutation_kind=mutation_kind,
            guard_set=guard_set,
        )
    elif authority_adapter == _REFUSED_TARGET_ORPHAN_ADAPTER:
        if legacy_args:
            raise SignedManifestAuthorityError(
                "orphan retirement does not accept positional adapter arguments"
            )
        if authority_sha256 is None:
            raise SignedManifestAuthorityError(
                "orphan retirement requires authority_sha256"
            )
        authority = _load_refused_target_orphan_authority(
            authority_path,
            expected_sha256=authority_sha256,
            mutation_kind=mutation_kind,
            guard_set=guard_set,
        )
    elif authority_adapter == _ERROR_SIBLING_ADAPTER:
        if legacy_args or authority_path != {}:
            raise SignedManifestAuthorityError(
                "error-sibling reconcile does not accept an authority artifact"
            )
        if reason != _ERROR_SIBLING_REASON:
            raise SignedManifestAuthorityError(
                "error-sibling reconcile requires its exact quarantine reason"
            )
        authority = _load_error_sibling_authority(
            mutation_kind=mutation_kind,
            guard_set=guard_set,
        )
    else:
        if legacy_args:
            raise SignedManifestAuthorityError(
                "generic signed authority does not accept positional adapter arguments"
            )
        if authority_adapter is not None:
            raise SignedManifestAuthorityError(
                f"unsupported signed authority adapter: {authority_adapter}"
            )
        if authority_file_sha256 is None or authority_payload_sha256 is None:
            raise SignedManifestAuthorityError(
                "generic signed authority requires file and payload digests"
            )
        authority = _load_authority(
            authority_path,
            expected_file_sha256=authority_file_sha256,
            expected_payload_sha256=authority_payload_sha256,
        )
    scope_refusal = _scope_refusal(authority, name_ids, apply=apply)
    if scope_refusal is not None:
        return scope_refusal
    if gc is not None and client_factory is not None:
        raise SignedManifestAuthorityError(
            "signed-manifest graph client and client factory are mutually exclusive"
        )
    own_client = gc is None
    client: Any = (
        client_factory()
        if client_factory is not None
        else GraphClient()
        if own_client
        else gc
    )
    try:
        if deterministic_self_heal and getattr(type(client), "session", None) is None:
            return _apply_error_sibling_query_handle(client)
        with client.session() as session:
            transaction = session.begin_transaction()
            query = _TransactionQuery(transaction)
            try:
                if apply and manifest_sha256 is not None:
                    replay = _replay(query, authority, str(manifest_sha256))
                    if replay is not None:
                        transaction.rollback()
                        return _project_receipt(authority, replay)

                preview = _build_preview(query, authority, reason)
                authorized_manifest_sha256 = (
                    preview.manifest_sha256
                    if deterministic_self_heal
                    else manifest_sha256
                )
                counts = {
                    "authority_rows": len(_runtime_authority_rows(authority)),
                    "admitted": len(preview.admitted),
                    "refused": len(preview.refusals),
                }
                if not apply:
                    transaction.rollback()
                    receipt = {
                        "schema": SIGNED_MANIFEST_RECEIPT_SCHEMA,
                        "outcome": (
                            "refused"
                            if _all_or_nothing(authority) and preview.refusals
                            else "would_apply"
                            if preview.admitted
                            else "refused"
                        ),
                        "changed": 0,
                        "would_change": (
                            0
                            if _all_or_nothing(authority) and preview.refusals
                            else len(preview.admitted)
                        ),
                        "counts": counts,
                        "refusals": preview.refusals,
                        "manifest": preview.manifest,
                        "manifest_sha256": preview.manifest_sha256,
                        "authority_file_sha256": authority.file_sha256,
                        "authority_payload_sha256": authority.payload_sha256,
                    }
                    return _project_receipt(authority, receipt)
                if preview.manifest_sha256 != authorized_manifest_sha256:
                    if authority.data.get("adapter") == _STALE_SOURCE_ADAPTER:
                        raise StaleSourceDetachConflict(
                            "fresh stale-source closure does not match manifest SHA-256"
                        )
                    raise SignedManifestConflict(
                        "fresh signed-manifest closure does not match authorized SHA-256"
                    )
                if not preview.admitted:
                    transaction.rollback()
                    receipt = {
                        "schema": SIGNED_MANIFEST_RECEIPT_SCHEMA,
                        "outcome": "refused",
                        "changed": 0,
                        "counts": counts,
                        "refusals": preview.refusals,
                        "manifest_sha256": preview.manifest_sha256,
                    }
                    return _project_receipt(authority, receipt)
                if _all_or_nothing(authority) and preview.refusals:
                    transaction.rollback()
                    return _project_receipt(
                        authority,
                        {
                            "schema": SIGNED_MANIFEST_RECEIPT_SCHEMA,
                            "outcome": "refused",
                            "changed": 0,
                            "would_change": 0,
                            "counts": counts,
                            "refusals": preview.refusals,
                            "manifest": preview.manifest,
                            "manifest_sha256": preview.manifest_sha256,
                        },
                    )

                _lock_participants(query, preview)
                locked_preview = _build_preview(query, authority, reason)
                if locked_preview.manifest_sha256 != preview.manifest_sha256:
                    if authority.data.get("adapter") == _STALE_SOURCE_ADAPTER:
                        raise StaleSourceDetachConflict(
                            "stale-source closure changed while locking"
                        )
                    raise SignedManifestConflict(
                        "signed-manifest closure changed while locking"
                    )
                counters_before = query.query(
                    """
                    RETURN COUNT { (:StandardNameChange) } AS changes,
                           COUNT { (:LLMCost) } AS llm_costs
                    """
                )[0]
                authority.data["counters_before"] = counters_before
                mutation_count = sum(
                    _apply_mutation(query, action) for action in locked_preview.admitted
                )
                change_ids = _write_receipts(
                    query,
                    authority,
                    locked_preview,
                    reason=reason,
                    run_id=run_id,
                )
                if len(change_ids) != len(locked_preview.admitted):
                    raise SignedManifestConflict(
                        "signed-manifest receipt cardinality changed"
                    )
                _verify_postconditions(
                    query,
                    authority,
                    [action["row"].id for action in locked_preview.admitted],
                    locked_preview.admitted,
                )
                collateral_after = _collateral_snapshot(
                    query,
                    excluded_node_ids=sorted(
                        {
                            snapshot["element_id"]
                            for action in locked_preview.admitted
                            for snapshot in action["participant_snapshots"].values()
                            if "labels" in snapshot
                        }
                    ),
                    excluded_relationship_ids=sorted(
                        {
                            snapshot["element_id"]
                            for action in locked_preview.admitted
                            for snapshot in action["participant_snapshots"].values()
                            if "relationship_type" in snapshot
                        }
                        | set(_added_relationship_ids(query, locked_preview.admitted))
                    ),
                )
                if collateral_after != locked_preview.collateral:
                    raise SignedManifestConflict("out-of-allowlist closure changed")
                counters_after = query.query(
                    """
                    RETURN COUNT { (:StandardNameChange) } AS changes,
                           COUNT { (:LLMCost) } AS llm_costs
                    """
                )[0]
                authority.data["counters_after"] = counters_after
                if (
                    int(counters_after["changes"]) - int(counters_before["changes"])
                    != len(change_ids)
                    or counters_after["llm_costs"] != counters_before["llm_costs"]
                ):
                    raise SignedManifestConflict(
                        "signed-manifest counter baseline changed unexpectedly"
                    )
                transaction.commit()
                receipt = {
                    "schema": SIGNED_MANIFEST_RECEIPT_SCHEMA,
                    "outcome": "applied",
                    "changed": len(locked_preview.admitted),
                    "mutations": mutation_count,
                    "receipt_rows": len(change_ids),
                    "persistent_writes": mutation_count + len(change_ids),
                    "counts": counts,
                    "refusals": locked_preview.refusals,
                    "manifest_sha256": locked_preview.manifest_sha256,
                    "authority_file_sha256": authority.file_sha256,
                    "authority_payload_sha256": authority.payload_sha256,
                    "change_ids": change_ids,
                    "admitted_actions": locked_preview.admitted,
                }
                return _project_receipt(authority, receipt)
            except BaseException:
                if not transaction.closed:
                    transaction.rollback()
                raise
    finally:
        if own_client:
            client.close()
