"""Generic transaction envelope for typed, signed graph-repair authorities.

The authority file chooses rows, mutations, and guards.  Callers supply only
the file and its two independently trusted digests; they cannot narrow or
expand the executable cohort.  The graph closure is derived again inside an
applying transaction, locked, and re-hashed before any mutation.

Only closed registries are interpreted here.  Authority artifacts never carry
Cypher.  The current mutation registry contains ``delete``, ``supersede``, and
``detach``.  The semantic guard registry contains last-producing-source,
structural-legitimacy, and out-of-allowlist-immutability.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Sequence
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

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_NODE_LABELS = frozenset({"StandardName", "StandardNameSource"})
_RELATIONSHIP_TYPES = frozenset({"PRODUCED_NAME", "HAS_PARENT"})
_MUTATION_KINDS = frozenset(
    {
        RepairMutationKind.delete.value,
        RepairMutationKind.supersede.value,
        RepairMutationKind.detach.value,
    }
)
_LAST_PRODUCER = "last-producing-source"
_STRUCTURAL_LEGITIMACY = "structural-legitimacy"
_COLLATERAL_IMMUTABILITY = "out-of-allowlist-immutability"
_SIGNED_LIFECYCLE = "signed-lifecycle-and-claim"
_NO_LIVE_PRODUCER = "no-live-producing-source"
_NO_LIVE_STRUCTURAL_CHILD = "no-live-structural-child"
_GUARD_KINDS = {
    _LAST_PRODUCER: RepairGuardKind.semantic_authority.value,
    _STRUCTURAL_LEGITIMACY: RepairGuardKind.semantic_authority.value,
    _COLLATERAL_IMMUTABILITY: RepairGuardKind.collateral_immutability.value,
    _SIGNED_LIFECYCLE: RepairGuardKind.semantic_authority.value,
    _NO_LIVE_PRODUCER: RepairGuardKind.semantic_authority.value,
    _NO_LIVE_STRUCTURAL_CHILD: RepairGuardKind.semantic_authority.value,
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
    if not isinstance(signature, dict) or signature.get("sha256") != payload_sha256:
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
            if kind not in _MUTATION_KINDS:
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
        if RepairMutationKind.detach.value in mutation_kinds:
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

    repair_rows = data.get("repair_rows")
    if repair_rows is not None and sorted(repair_rows) != sorted(seen_row_ids):
        raise SignedManifestAuthorityError(
            "repair_rows projection does not match authority rows"
        )
    operation_id = data.get("operation_id")
    if not isinstance(operation_id, str) or not operation_id.strip():
        raise SignedManifestAuthorityError("operation_id must be non-empty")
    artifact_projection = {
        **data,
        "authority_digests": [str(item["id"]) for item in digest_rows] or None,
        "selection": str(selection["id"]),
        "repair_rows": sorted(seen_row_ids),
        "receipt_policy": str(receipt_policy["id"]),
    }
    _validate_model(RepairAuthorityArtifact, artifact_projection, "repair authority")
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
) -> dict[str, Any] | None:
    kind = str(participant["kind"])
    participant_id = str(participant["id"])
    graph_label = str(participant["graph_label"])
    if kind == RepairParticipantKind.node.value:
        rows = query.query(
            """
            MATCH (node)
            WHERE node.id = $participant_id AND $graph_label IN labels(node)
            RETURN elementId(node) AS element_id,
                   labels(node) AS labels,
                   properties(node) AS properties
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
    return dict(rows[0])


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
    candidates: list[dict[str, Any]] = []
    refusals: list[dict[str, str]] = []
    for row in authority.rows:
        snapshots: dict[str, dict[str, Any]] = {}
        refusal: str | None = None
        for participant in row.participants:
            participant_id = str(participant["id"])
            snapshot = _participant_snapshot(query, participant)
            if snapshot is None:
                refusal = f"participant does not exist: {participant_id}"
                break
            signature = participant.get("signature_sha256")
            if signature is not None and _digest(snapshot) != signature:
                refusal = f"participant signature mismatch: {participant_id}"
                break
            snapshots[participant_id] = snapshot
        action = {
            "row": row,
            "participant_snapshots": snapshots,
            "participant_digests": [
                {"participant_id": participant_id, "sha256": _digest(snapshot)}
                for participant_id, snapshot in sorted(snapshots.items())
            ],
        }
        if refusal is None:
            refusal = _orphan_guard_refusal(query, action)
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
            target = _target_snapshot(action)
            relationship = next(
                snapshot
                for snapshot in action["participant_snapshots"].values()
                if snapshot.get("relationship_type") == "PRODUCED_NAME"
            )
            target_element_id = str(target["element_id"])
            producer_state = producer_cache.setdefault(
                target_element_id, _producer_state(query, target_element_id)
            )
            candidate_relationship_id = str(relationship["element_id"])
            remaining_live = [
                producer
                for producer in producer_state["producers"]
                if producer.get("live")
                and producer["relationship_id"] not in removed_relationship_ids
                and producer["relationship_id"] != candidate_relationship_id
            ]
            if not remaining_live and int(producer_state.get("live_children") or 0) < 1:
                refusals.append(
                    {
                        "row_id": row.id,
                        "reason": "target would lose its last producing source",
                    }
                )
                continue
            removed_relationship_ids.add(candidate_relationship_id)
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
        ORDER BY change.row_id
        """,
        operation=operation,
        manifest_sha256=manifest_sha256,
    )


def _verify_postconditions(
    query: _Query, authority: _Authority, row_ids: list[str]
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
    by_id = {row.id: row for row in authority.rows}
    for row_id in row_ids:
        row = by_id[row_id]
        for mutation in row.mutations:
            participant = next(
                item
                for item in row.participants
                if item["id"] == mutation["participant_id"]
            )
            kind = str(mutation["kind"])
            if kind == RepairMutationKind.detach.value:
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
            elif kind == RepairMutationKind.delete.value:
                present = query.query(
                    "MATCH (node) WHERE node.id = $id RETURN count(node) AS count",
                    id=participant["id"],
                )[0]["count"]
                if int(present) != 0:
                    raise SignedManifestConflict(
                        "recorded signed-manifest repair lost its postcondition"
                    )
            else:
                state = query.query(
                    """
                    MATCH (node:StandardName {id: $id})
                    RETURN node.name_stage AS name_stage, node.status AS status
                    """,
                    id=participant["id"],
                )
                if not state or state[0] != {
                    "name_stage": "superseded",
                    "status": "superseded",
                }:
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
    if (
        not admitted_ids
        or expected_ids != admitted_ids
        or len(properties) != len(admitted_ids)
        or any(
            sorted(item.get("cohort_admitted_ids") or []) != admitted_ids
            or item.get("authority_file_sha256") != authority.file_sha256
            or item.get("authority_payload_sha256") != authority.payload_sha256
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
    snapshots = action["participant_snapshots"]
    for mutation in row.mutations:
        participant_id = str(mutation["participant_id"])
        snapshot = snapshots[participant_id]
        kind = str(mutation["kind"])
        if kind == RepairMutationKind.detach.value:
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
        elif kind == RepairMutationKind.supersede.value:
            source_path_update = (
                ""
                if mutation.get("preserve_source_paths")
                else "target.source_paths = [],"
            )
            result = query.query(
                f"""
                MATCH (target:StandardName)
                WHERE elementId(target) = $element_id
                SET target.superseded_from_stage = coalesce(
                      target.superseded_from_stage, target.name_stage),
                    target.name_stage = 'superseded',
                    target.status = 'superseded',
                    {source_path_update}
                    target.claimed_at = null,
                    target.claim_token = null
                RETURN count(target) AS changed
                """,  # noqa: S608 - the inserted fragment is selected locally
                element_id=snapshot["element_id"],
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
                "from_name": row.identity.get("target_id")
                or row.identity.get("source_id")
                or row.id,
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
          to_name: row.from_name,
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
        ORDER BY change.row_id
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
    return _project_stale_source_receipt(
        authority, _project_refused_target_orphan_receipt(authority, receipt)
    )


@retry_on_deadlock()
def apply_signed_manifest(
    authority_path: str | Path | dict[str, Any],
    *legacy_args: Any,
    authority_file_sha256: str | None = None,
    authority_payload_sha256: str | None = None,
    authority_sha256: str | None = None,
    authority_adapter: str | None = None,
    mutation_kind: str | None = None,
    guard_set: tuple[str, ...] | None = None,
    name_ids: list[str] | None = None,
    reason: str,
    apply: bool = False,
    manifest_sha256: str | None = None,
    run_id: str | None = None,
    gc: Any | None = None,
) -> dict[str, Any]:
    """Preview or atomically apply the complete row set in a signed authority.

    ``manifest_sha256`` is the authorization returned by a prior preview.  It
    authorizes only the hash: participant closure, collateral rows, and counter
    baselines are always read again inside this invocation.
    """
    if not reason.strip():
        message = (
            "stale-source detach requires a non-empty reason"
            if authority_adapter == _STALE_SOURCE_ADAPTER
            else "signed-manifest apply requires a non-empty reason"
        )
        raise ValueError(message)
    if apply and manifest_sha256 is None:
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
    own_client = gc is None
    client: Any = GraphClient() if own_client else gc
    try:
        with client.session() as session:
            transaction = session.begin_transaction()
            query = _TransactionQuery(transaction)
            try:
                if apply:
                    replay = _replay(query, authority, str(manifest_sha256))
                    if replay is not None:
                        transaction.rollback()
                        return _project_receipt(authority, replay)

                preview = _build_preview(query, authority, reason)
                counts = {
                    "authority_rows": len(authority.rows),
                    "admitted": len(preview.admitted),
                    "refused": len(preview.refusals),
                }
                if not apply:
                    transaction.rollback()
                    receipt = {
                        "schema": SIGNED_MANIFEST_RECEIPT_SCHEMA,
                        "outcome": (
                            "refused"
                            if authority.data.get("all_or_nothing") and preview.refusals
                            else "would_apply"
                            if preview.admitted
                            else "refused"
                        ),
                        "changed": 0,
                        "would_change": (
                            0
                            if authority.data.get("all_or_nothing") and preview.refusals
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
                if preview.manifest_sha256 != manifest_sha256:
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
                if authority.data.get("all_or_nothing") and preview.refusals:
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
