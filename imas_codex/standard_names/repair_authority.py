"""Build canonical signed repair-authority artifacts from semantic inputs.

The builder owns the closed wire literals required by the signed-manifest
loader.  Callers provide repair rows and operation-specific receipt semantics;
they cannot substitute an open selection predicate or an undeclared execution
mode.  The returned digests are recomputed from the final emitted bytes.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from imas_codex.graph.models import RepairAuthorityArtifact
from imas_codex.standard_names.signed_manifest import (
    authority_artifact_wire_projection,
    signed_payload_sha256,
)

REPAIR_AUTHORITY_SCHEMA = "imas-codex.repair-authority.v1"
ARTIFACT_ROWS_SELECTION = "artifact-rows"
DEFAULT_SELECTION_MODE = "exact_complete_signed_cohort"
SIGNED_PAYLOAD_CANONICALIZATION = "json-sort-keys-v1"


class RepairAuthorityBuildError(ValueError):
    """The requested canonical repair authority cannot be emitted safely."""


@dataclass(frozen=True)
class BuiltRepairAuthority:
    """Validated authority bytes and the two independent required digests."""

    content: bytes
    file_sha256: str
    payload_sha256: str
    artifact: RepairAuthorityArtifact


def _selection(value: object, *, role: str) -> dict[str, Any]:
    if value is None:
        supplied: dict[str, Any] = {}
    elif isinstance(value, Mapping):
        supplied = copy.deepcopy(dict(value))
    else:
        raise RepairAuthorityBuildError(f"{role} selection must be an object")

    selection_id = supplied.get("id", ARTIFACT_ROWS_SELECTION)
    if selection_id != ARTIFACT_ROWS_SELECTION:
        raise RepairAuthorityBuildError(
            f"{role} selection id must be '{ARTIFACT_ROWS_SELECTION}'"
        )
    predicate = supplied.get("predicate", ARTIFACT_ROWS_SELECTION)
    if predicate != ARTIFACT_ROWS_SELECTION:
        raise RepairAuthorityBuildError(
            f"{role} selection predicate must be '{ARTIFACT_ROWS_SELECTION}'"
        )

    supplied["id"] = ARTIFACT_ROWS_SELECTION
    supplied["predicate"] = ARTIFACT_ROWS_SELECTION
    supplied.setdefault("mode", DEFAULT_SELECTION_MODE)
    return supplied


def _repair_rows(
    values: object, *, artifact_selection: dict[str, Any]
) -> list[dict[str, Any]]:
    if (
        not isinstance(values, Sequence)
        or isinstance(values, str | bytes)
        or not values
    ):
        raise RepairAuthorityBuildError("authority rows must be a non-empty array")

    rows: list[dict[str, Any]] = []
    row_ids: set[str] = set()
    for value in values:
        if not isinstance(value, Mapping):
            raise RepairAuthorityBuildError("every authority row must be an object")
        row = copy.deepcopy(dict(value))
        row_id = row.get("id")
        if not isinstance(row_id, str) or not row_id.strip() or row_id in row_ids:
            raise RepairAuthorityBuildError(
                "authority row ids must be unique non-empty strings"
            )
        row_ids.add(row_id)

        row_selection = _selection(row.get("selection"), role=f"row {row_id!r}")
        if row_selection != artifact_selection:
            raise RepairAuthorityBuildError(
                f"row {row_id!r} selection must equal the artifact selection"
            )
        row["selection"] = copy.deepcopy(artifact_selection)
        rows.append(row)
    return rows


def _receipt_policy(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise RepairAuthorityBuildError("receipt_policy must be an object")
    policy = copy.deepcopy(dict(value))
    expected_count = policy.get("expected_count", "admitted_rows")
    if expected_count != "admitted_rows":
        raise RepairAuthorityBuildError(
            "receipt_policy expected_count must be 'admitted_rows'"
        )
    policy["expected_count"] = "admitted_rows"
    return policy


def _serialized(data: dict[str, Any]) -> bytes:
    try:
        return (
            json.dumps(
                data,
                sort_keys=True,
                indent=2,
                allow_nan=False,
                default=str,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise RepairAuthorityBuildError(
            "repair authority contains a value that cannot be serialized"
        ) from exc


def build_repair_authority(specification: Mapping[str, Any]) -> BuiltRepairAuthority:
    """Return a signed, schema-valid canonical authority and its byte digests.

    ``selection``, row-level selections, ``repair_rows``, the receipt count
    expression, ``schema``, and ``signature`` are builder-owned.  Matching
    closed values may be present in an input assembled from an existing
    template, but callers never need to provide them.
    """
    if not isinstance(specification, Mapping):
        raise RepairAuthorityBuildError(
            "repair authority specification must be an object"
        )

    data = copy.deepcopy(dict(specification))
    if "all_or_nothing" in data:
        raise RepairAuthorityBuildError(
            "top-level all_or_nothing is not part of a canonical repair authority"
        )
    if "signature" in data:
        raise RepairAuthorityBuildError("the repair authority builder owns signature")

    supplied_schema = data.get("schema", REPAIR_AUTHORITY_SCHEMA)
    if supplied_schema != REPAIR_AUTHORITY_SCHEMA:
        raise RepairAuthorityBuildError(
            f"canonical repair authority schema must be '{REPAIR_AUTHORITY_SCHEMA}'"
        )
    data["schema"] = REPAIR_AUTHORITY_SCHEMA

    selection = _selection(data.get("selection"), role="artifact")
    rows = _repair_rows(data.get("rows"), artifact_selection=selection)
    row_ids = [str(row["id"]) for row in rows]
    supplied_projection = data.get("repair_rows")
    if supplied_projection is not None and list(supplied_projection) != row_ids:
        raise RepairAuthorityBuildError(
            "repair_rows projection must match authority row order"
        )

    data["selection"] = selection
    data["rows"] = rows
    data["repair_rows"] = row_ids
    data["receipt_policy"] = _receipt_policy(data.get("receipt_policy"))

    payload_sha256 = signed_payload_sha256(data)
    data["signature"] = {
        "canonicalization": SIGNED_PAYLOAD_CANONICALIZATION,
        "sha256": payload_sha256,
    }
    content = _serialized(data)

    try:
        emitted = json.loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:  # pragma: no cover
        raise RepairAuthorityBuildError(
            "emitted repair authority is not valid JSON"
        ) from exc
    emitted_payload_sha256 = signed_payload_sha256(emitted)
    if emitted_payload_sha256 != payload_sha256:  # pragma: no cover
        raise RepairAuthorityBuildError(
            "emitted repair authority changed its canonical signed payload"
        )
    schema_projection = {
        **emitted,
        "selection": str(emitted["selection"]["id"]),
        "repair_rows": [str(row["id"]) for row in emitted["rows"]],
        "receipt_policy": str(emitted["receipt_policy"]["id"]),
    }
    try:
        artifact = RepairAuthorityArtifact.model_validate(
            authority_artifact_wire_projection(schema_projection)
        )
    except ValidationError as exc:
        raise RepairAuthorityBuildError(f"invalid repair authority: {exc}") from exc

    return BuiltRepairAuthority(
        content=content,
        file_sha256=hashlib.sha256(content).hexdigest(),
        payload_sha256=emitted_payload_sha256,
        artifact=artifact,
    )
