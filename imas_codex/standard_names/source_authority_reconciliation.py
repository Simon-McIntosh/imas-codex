"""Governed source-authority reconciliation for exact DD source cohorts.

Each invocation accepts one hash-bound manifest containing one homogeneous
operation. Planning is read-only. Application locks the complete graph closure,
repeats the plan under the locks, writes one typed immutable event per source,
and proves the operation-specific post-state before committing. Standard-name
content and review lifecycle are never authority inputs that this surface may
rewrite.
"""

from __future__ import annotations

import copy
import hashlib
import hmac
import json
import uuid
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from imas_codex.core.node_categories import SN_SOURCE_CATEGORIES
from imas_codex.discovery.base.claims import retry_on_deadlock
from imas_codex.graph.client import GraphClient
from imas_codex.standard_names.source_authority import (
    SNAPSHOT_FIELDS,
    SNAPSHOT_MUTABLE_FIELDS,
    canonical_payload,
    capture_source_authority_closure,
    lock_participants,
    normalize_manifest_hash_binding,
    participant_ids,
    payload_hash,
    read_source_authority_rows,
    read_source_target_protection_rows,
    require_complete_paths,
    source_identity_payload,
    validate_source_id,
)
from imas_codex.standard_names.source_snapshot_migration import (
    classify_snapshot_change,
)

_MANIFEST_SCHEMA = "imas-codex.source-authority-reconciliation-manifest"
_MANIFEST_SCHEMA_VERSION = 1
_RECEIPT_SCHEMA = "imas-codex.source-authority-reconciliation-receipt"
_RECEIPT_SCHEMA_VERSION = 1
_RUN_PREFIX = "source-authority-reconciliation:"

REPAIR_IDENTITY_SCALAR = "repair_identity_scalar"
ADOPT_CURRENT_SNAPSHOT = "adopt_current_snapshot"
ADMIT_CURRENT_SNAPSHOT = "admit_current_snapshot"
FOLD_DUPLICATE_SOURCE_IDENTITY = "fold_duplicate_source_identity"
RETIRE_NONPARTICIPATING_SOURCE = "retire_nonparticipating_source"

_OPERATIONS = frozenset(
    {
        REPAIR_IDENTITY_SCALAR,
        ADOPT_CURRENT_SNAPSHOT,
        ADMIT_CURRENT_SNAPSHOT,
        FOLD_DUPLICATE_SOURCE_IDENTITY,
        RETIRE_NONPARTICIPATING_SOURCE,
    }
)

_AUTHORITY_RELATIONSHIPS = frozenset(
    {
        "HAS_IDENTITY_REPAIR",
        "HAS_SNAPSHOT_ADOPTION",
        "HAS_SNAPSHOT_ADMISSION",
        "HAS_IDENTITY_FOLD",
        "HAS_AUTHORITY_RETIREMENT",
    }
)

_EVENT_LABELS = {
    REPAIR_IDENTITY_SCALAR: "StandardNameSourceIdentityRepair",
    ADOPT_CURRENT_SNAPSHOT: "StandardNameSourceSnapshotAdoption",
    ADMIT_CURRENT_SNAPSHOT: "StandardNameSourceSnapshotAdmission",
    FOLD_DUPLICATE_SOURCE_IDENTITY: "StandardNameSourceIdentityFold",
    RETIRE_NONPARTICIPATING_SOURCE: "StandardNameSourceAuthorityRetirement",
}

_EVENT_PREFIXES = {
    REPAIR_IDENTITY_SCALAR: "source-identity-repair:",
    ADOPT_CURRENT_SNAPSHOT: "source-snapshot-adoption:",
    ADMIT_CURRENT_SNAPSHOT: "source-snapshot-admission:",
    FOLD_DUPLICATE_SOURCE_IDENTITY: "source-identity-fold:",
    RETIRE_NONPARTICIPATING_SOURCE: "source-authority-retirement:",
}

_COMMON_ROW_FIELDS = frozenset(
    {
        "source_id",
        "operation",
        "expected_source_element_id",
        "expected_source_id",
        "expected_from_dd_path",
        "expected_before_snapshot_hash",
        "expected_authority_hash",
        "expected_preserved_state_hash",
        "expected_participant_ids_hash",
        "west_intersection",
        "test_intersection",
    }
)

_EXTRA_ROW_FIELDS = {
    REPAIR_IDENTITY_SCALAR: frozenset(),
    ADOPT_CURRENT_SNAPSHOT: frozenset(),
    ADMIT_CURRENT_SNAPSHOT: frozenset(),
    FOLD_DUPLICATE_SOURCE_IDENTITY: frozenset(
        {
            "duplicate_source_id",
            "expected_duplicate_source_element_id",
            "expected_duplicate_source_id",
            "expected_duplicate_from_dd_path",
            "expected_duplicate_preserved_state_hash",
            "expected_duplicate_destructive_closure_hash",
        }
    ),
    RETIRE_NONPARTICIPATING_SOURCE: frozenset(
        {
            "expected_node_category",
            "expected_target_id",
            "expected_retirement_destructive_closure_hash",
        }
    ),
}

_AUTHORITY_EVENTS_QUERY = """
// SOURCE_AUTHORITY_RECONCILIATION_EVENTS
UNWIND $source_ids AS source_id
MATCH (source:StandardNameSource {id: source_id})
RETURN source_id,
  [(source)-[link]->(event)
    WHERE type(link) IN $relationship_types | {
      relationship_type: type(link),
      relationship_element_id: elementId(link),
      relationship_properties: properties(link),
      event_element_id: elementId(event),
      event_labels: labels(event),
      event_properties: properties(event)
  }] AS events
ORDER BY source_id
"""

_DUPLICATE_SOURCES_QUERY = """
// SOURCE_AUTHORITY_RECONCILIATION_DUPLICATES
UNWIND $source_ids AS source_id
OPTIONAL MATCH (source:StandardNameSource {id: source_id})
RETURN source_id,
  [candidate IN collect(DISTINCT source) WHERE candidate IS NOT NULL | {
    element_id: elementId(candidate), labels: labels(candidate),
    properties: properties(candidate),
    relationships: [(candidate)-[relationship]-(other) | {
      element_id: elementId(relationship), type: type(relationship),
      direction: CASE WHEN startNode(relationship) = candidate THEN 'out' ELSE 'in' END,
      properties: properties(relationship),
      other_element_id: elementId(other), other_labels: labels(other),
      other_id: other.id, other_properties: properties(other)
    }],
    names: [(candidate)-[binding:PRODUCED_NAME]->(name:StandardName) | {
      binding_element_id: elementId(binding), binding_properties: properties(binding),
      element_id: elementId(name), labels: labels(name), properties: properties(name),
      relationships: [(name)-[name_link]-(related) | {
        element_id: elementId(name_link), type: type(name_link),
        direction: CASE WHEN startNode(name_link) = name THEN 'out' ELSE 'in' END,
        properties: properties(name_link),
        other_element_id: elementId(related), other_labels: labels(related),
        other_id: related.id, other_properties: properties(related)
      }]
    }]
  }] AS sources
ORDER BY source_id
"""

_RELATIONSHIP_LOCK_QUERY = """
// SOURCE_AUTHORITY_RECONCILIATION_RELATIONSHIP_LOCK
MATCH ()-[relationship]->()
WHERE elementId(relationship) IN $element_ids
SET relationship._source_authority_lock = true
REMOVE relationship._source_authority_lock
RETURN count(relationship) AS locked
"""

_REPAIR_APPLY_QUERY = """
// SOURCE_AUTHORITY_REPAIR_IDENTITY_APPLY
UNWIND $items AS item
MATCH (source:StandardNameSource {id: item.source_id})
WHERE elementId(source) = item.source_element_id
CREATE (event:StandardNameSourceIdentityRepair)
SET event = item.event, event.changed_at = datetime(item.event.changed_at)
CREATE (source)-[:HAS_IDENTITY_REPAIR]->(event)
SET source.source_id = item.after_source_id
RETURN collect(source.id) AS source_ids, collect(event.id) AS event_ids
"""

_ADOPT_APPLY_QUERY = """
// SOURCE_AUTHORITY_ADOPT_SNAPSHOT_APPLY
UNWIND $items AS item
MATCH (source:StandardNameSource {id: item.source_id})
WHERE elementId(source) = item.source_element_id
CREATE (event:StandardNameSourceSnapshotAdoption)
SET event = item.event, event.adopted_at = datetime(item.event.adopted_at)
CREATE (source)-[:HAS_SNAPSHOT_ADOPTION]->(event)
SET source += item.after
RETURN collect(source.id) AS source_ids, collect(event.id) AS event_ids
"""

_ADMIT_APPLY_QUERY = """
// SOURCE_AUTHORITY_ADMIT_SNAPSHOT_APPLY
UNWIND $items AS item
MATCH (source:StandardNameSource {id: item.source_id})
WHERE elementId(source) = item.source_element_id
CREATE (event:StandardNameSourceSnapshotAdmission)
SET event = item.event, event.admitted_at = datetime(item.event.admitted_at)
CREATE (source)-[:HAS_SNAPSHOT_ADMISSION]->(event)
SET source += item.after
RETURN collect(source.id) AS source_ids, collect(event.id) AS event_ids
"""

_FOLD_APPLY_QUERY = """
// SOURCE_AUTHORITY_FOLD_DUPLICATE_APPLY
UNWIND $items AS item
MATCH (canonical:StandardNameSource {id: item.source_id})
MATCH (duplicate:StandardNameSource {id: item.duplicate_source_id})
MATCH (duplicate)-[backing:FROM_DD_PATH]->(node:IMASNode {id: item.path})
WHERE elementId(canonical) = item.source_element_id
  AND elementId(duplicate) = item.duplicate_source_element_id
  AND elementId(backing) = item.duplicate_backing_element_id
CREATE (event:StandardNameSourceIdentityFold)
SET event = item.event, event.folded_at = datetime(item.event.folded_at)
CREATE (canonical)-[:HAS_IDENTITY_FOLD]->(event)
DELETE backing
SET duplicate.status = 'stale',
    duplicate.skip_reason = 'duplicate_source_identity',
    duplicate.skip_reason_detail = item.reason
RETURN collect(canonical.id) AS source_ids, collect(event.id) AS event_ids
"""

_RETIRE_APPLY_QUERY = """
// SOURCE_AUTHORITY_RETIRE_NONPARTICIPATING_APPLY
UNWIND $items AS item
MATCH (source:StandardNameSource {id: item.source_id})
MATCH (node:IMASNode {id: item.path})
MATCH (source)-[binding:PRODUCED_NAME]->(name:StandardName {id: item.target_id})
MATCH (node)-[projection:HAS_STANDARD_NAME]->(name)
WHERE elementId(source) = item.source_element_id
  AND elementId(binding) = item.binding_element_id
  AND elementId(projection) = item.projection_element_id
CREATE (event:StandardNameSourceAuthorityRetirement)
SET event = item.event, event.retired_at = datetime(item.event.retired_at)
CREATE (source)-[:HAS_AUTHORITY_RETIREMENT]->(event)
DELETE binding, projection
SET source.status = 'stale',
    source.produced_sn_id = null,
    source.claimed_at = null,
    source.claim_token = null,
    source.drain_scope_id = null,
    source.drain_scope_claimed_at = null,
    source.drain_claim_scope_id = null,
    source.drain_scope_actionable = null,
    source.skip_reason = 'nonparticipating_dd_source',
    source.skip_reason_detail = item.reason
RETURN collect(source.id) AS source_ids, collect(event.id) AS event_ids
"""

_APPLY_QUERIES = {
    REPAIR_IDENTITY_SCALAR: _REPAIR_APPLY_QUERY,
    ADOPT_CURRENT_SNAPSHOT: _ADOPT_APPLY_QUERY,
    ADMIT_CURRENT_SNAPSHOT: _ADMIT_APPLY_QUERY,
    FOLD_DUPLICATE_SOURCE_IDENTITY: _FOLD_APPLY_QUERY,
    RETIRE_NONPARTICIPATING_SOURCE: _RETIRE_APPLY_QUERY,
}


class SourceAuthorityReconciliationConflict(RuntimeError):
    """The exact manifest-bound source-authority closure changed."""


@dataclass(frozen=True)
class SourceAuthorityManifest:
    """One exact homogeneous source-authority cohort."""

    path: Path
    manifest_hash: str
    operation: str
    rows: tuple[dict[str, Any], ...]
    source_ids: tuple[str, ...]
    paths: tuple[str, ...]
    allowlist_hash: str


def _hash(value: Any) -> str:
    return payload_hash(value)


def _sha_payload(value: Any) -> str:
    return hashlib.sha256(canonical_payload(value).encode()).hexdigest()


def _require_sha(value: Any, field: str) -> str:
    normalized = str(value or "").strip().casefold()
    if len(normalized) != 64 or any(
        char not in "0123456789abcdef" for char in normalized
    ):
        raise ValueError(f"{field} must be exactly one SHA-256 hex digest")
    return normalized


def load_source_authority_manifest(path: str | Path) -> SourceAuthorityManifest:
    """Load and validate one exact, homogeneous source-authority manifest."""
    manifest_path = Path(path).expanduser().resolve()
    raw = manifest_path.read_bytes()
    try:
        payload = json.loads(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"source-authority manifest is not valid JSON: {manifest_path}"
        ) from exc
    if not isinstance(payload, dict) or set(payload) != {
        "schema",
        "schema_version",
        "rows",
    }:
        raise ValueError(
            "source-authority manifest must contain only schema, schema_version, and rows"
        )
    if (
        payload.get("schema") != _MANIFEST_SCHEMA
        or payload.get("schema_version") != _MANIFEST_SCHEMA_VERSION
    ):
        raise ValueError("source-authority manifest schema is unsupported")
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("source-authority manifest requires a non-empty rows array")
    operations = {row.get("operation") for row in rows if isinstance(row, dict)}
    if len(operations) != 1:
        raise ValueError(
            "source-authority manifest must contain one homogeneous operation"
        )
    operation = next(iter(operations))
    if operation not in _OPERATIONS:
        raise ValueError(f"unsupported source-authority operation: {operation!r}")

    normalized_rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    duplicate_ids: list[str] = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("source-authority manifest rows must be objects")
        exact_fields = _COMMON_ROW_FIELDS | _EXTRA_ROW_FIELDS[operation]
        if set(row) != exact_fields:
            missing = sorted(exact_fields - set(row))
            extra = sorted(set(row) - exact_fields)
            raise ValueError(
                f"{operation} manifest row fields are not exact; missing={missing}, extra={extra}"
            )
        source_id = str(row["source_id"])
        path_value = validate_source_id(source_id)
        if source_id in seen:
            raise ValueError(f"duplicate source identity in manifest: {source_id}")
        seen.add(source_id)
        if row["operation"] != operation:
            raise ValueError(
                "source-authority manifest must contain one homogeneous operation"
            )
        if row["expected_from_dd_path"] != path_value:
            raise ValueError(
                "expected_from_dd_path must equal the path encoded by source_id"
            )
        if (
            not isinstance(row["expected_source_element_id"], str)
            or not row["expected_source_element_id"]
        ):
            raise ValueError("expected_source_element_id is required")
        for field in (
            "expected_before_snapshot_hash",
            "expected_authority_hash",
            "expected_preserved_state_hash",
            "expected_participant_ids_hash",
        ):
            _require_sha(row[field], field)
        if row["west_intersection"] != 0 or row["test_intersection"] != 0:
            raise ValueError("WEST and test intersections must both be exactly zero")
        if operation == FOLD_DUPLICATE_SOURCE_IDENTITY:
            duplicate_id = str(row["duplicate_source_id"])
            if duplicate_id == source_id:
                raise ValueError(
                    "duplicate_source_id must differ from the canonical source_id"
                )
            validate_source_id(duplicate_id)
            if (
                not isinstance(row["expected_duplicate_source_element_id"], str)
                or not row["expected_duplicate_source_element_id"]
            ):
                raise ValueError("expected_duplicate_source_element_id is required")
            if row["expected_duplicate_from_dd_path"] != path_value:
                raise ValueError("duplicate backing must equal the canonical DD path")
            _require_sha(
                row["expected_duplicate_preserved_state_hash"],
                "expected_duplicate_preserved_state_hash",
            )
            _require_sha(
                row["expected_duplicate_destructive_closure_hash"],
                "expected_duplicate_destructive_closure_hash",
            )
            duplicate_ids.append(duplicate_id)
        if operation == RETIRE_NONPARTICIPATING_SOURCE:
            if (
                not isinstance(row["expected_node_category"], str)
                or not row["expected_node_category"]
            ):
                raise ValueError("expected_node_category is required")
            if (
                not isinstance(row["expected_target_id"], str)
                or not row["expected_target_id"]
            ):
                raise ValueError("expected_target_id is required")
            _require_sha(
                row["expected_retirement_destructive_closure_hash"],
                "expected_retirement_destructive_closure_hash",
            )
        normalized_rows.append(copy.deepcopy(row))

    normalized_rows.sort(key=lambda row: row["source_id"])
    source_ids = tuple(row["source_id"] for row in normalized_rows)
    if operation == FOLD_DUPLICATE_SOURCE_IDENTITY:
        if len(duplicate_ids) != len(set(duplicate_ids)):
            raise ValueError("fold duplicate participants must be globally unique")
        if set(source_ids) & set(duplicate_ids):
            raise ValueError(
                "fold canonical and duplicate participants must be disjoint"
            )
    return SourceAuthorityManifest(
        path=manifest_path,
        manifest_hash=hashlib.sha256(raw).hexdigest(),
        operation=str(operation),
        rows=tuple(normalized_rows),
        source_ids=source_ids,
        paths=tuple(validate_source_id(source_id) for source_id in source_ids),
        allowlist_hash=_hash(source_ids),
    )


def _read_events(
    transaction: Any, source_ids: tuple[str, ...]
) -> dict[str, list[dict[str, Any]]]:
    rows = transaction.run(
        _AUTHORITY_EVENTS_QUERY,
        source_ids=list(source_ids),
        relationship_types=sorted(_AUTHORITY_RELATIONSHIPS),
    )
    return {str(row["source_id"]): list(row["events"] or []) for row in rows}


def _read_duplicates(
    transaction: Any, source_ids: tuple[str, ...]
) -> dict[str, list[dict[str, Any]]]:
    rows = transaction.run(_DUPLICATE_SOURCES_QUERY, source_ids=list(source_ids))
    return {str(row["source_id"]): list(row["sources"] or []) for row in rows}


def _claim_reasons(source: dict[str, Any], *, prefix: str = "source") -> list[str]:
    reasons: list[str] = []
    claim_fields = (
        "claimed_at",
        "claim_token",
        "drain_scope_id",
        "drain_scope_claimed_at",
        "drain_claim_scope_id",
    )
    properties = source.get("properties") or {}
    if any(properties.get(field) is not None for field in claim_fields):
        reasons.append(f"{prefix} has an active worker or bounded-drain claim")
    for name in source.get("names") or []:
        name_properties = name.get("properties") or {}
        if any(name_properties.get(field) is not None for field in claim_fields):
            reasons.append(
                f"produced name {name_properties.get('id')!r} has an active claim"
            )
    return reasons


def _protected_reasons(value: Any) -> list[str]:
    """Detect protected graph participants from identity-bearing fields only."""
    west = False
    test = False

    def visit(item: Any, key: str = "") -> None:
        nonlocal west, test
        if isinstance(item, dict):
            for child_key, child in item.items():
                visit(child, str(child_key))
            return
        if isinstance(item, list | tuple):
            for child in item:
                visit(child, key)
            return
        if not isinstance(item, str):
            return
        normalized = item.casefold()
        normalized_key = key.casefold()
        if normalized_key in {"facility_id", "facility"} and normalized == "west":
            west = True
        if normalized_key in {"id", "other_id", "source_id", "stable_id"}:
            if normalized.startswith("signals:west:") or normalized.startswith("west:"):
                west = True
            if normalized.startswith(("test:", "fixture:", "signals:test:")):
                test = True
        if normalized_key in {"source_type", "origin"} and normalized in {
            "test",
            "fixture",
        }:
            test = True

    visit(value)
    reasons = []
    if west:
        reasons.append("current graph closure intersects WEST")
    if test:
        reasons.append("current graph closure intersects test fixtures")
    return reasons


def _target_protection_reasons(protection: dict[str, Any]) -> list[str]:
    """Refuse incomplete target closures and direct or shared protected producers."""
    reasons: list[str] = []
    for direct_entry in protection.get("direct_sources") or []:
        requested_source_id = direct_entry.get("requested_source_id")
        matches = direct_entry.get("matches") or []
        if len(matches) != 1:
            reasons.append(
                f"protected closure source {requested_source_id!r} is missing or ambiguous"
            )
            continue
        direct_reasons = _protected_reasons(matches[0])
        reasons.extend(
            reason.replace("current graph closure", "direct source identity")
            for reason in direct_reasons
        )

    target_entries = protection.get("targets") or []
    prospective_ids = set(protection.get("prospective_target_ids") or [])
    seen_target_ids: set[str] = set()
    for target_entry in target_entries:
        requested_target_id = target_entry.get("requested_target_id")
        if isinstance(requested_target_id, str):
            if requested_target_id in seen_target_ids:
                reasons.append(
                    f"protected closure target {requested_target_id!r} is duplicated"
                )
            seen_target_ids.add(requested_target_id)
        matches = target_entry.get("matches") or []
        if len(matches) != 1:
            reasons.append(
                f"protected closure target {requested_target_id!r} is missing or ambiguous"
            )
            continue
        target = matches[0]
        producers = target.get("producers") or []
        producer_bindings = [
            (producer.get("source_element_id"), producer.get("binding_element_id"))
            for producer in producers
        ]
        if any(
            not source_id or not binding_id
            for source_id, binding_id in producer_bindings
        ):
            reasons.append(
                f"protected producer closure for target {requested_target_id!r} is incomplete"
            )
        if len(producer_bindings) != len(set(producer_bindings)):
            reasons.append(
                f"protected producer closure for target {requested_target_id!r} is ambiguous"
            )
        for producer in producers:
            producer_reasons = _protected_reasons(producer)
            reasons.extend(
                (
                    f"target {requested_target_id!r} has a protected producer: "
                    + reason.replace("current graph closure intersects ", "")
                )
                for reason in producer_reasons
            )
    missing_prospective = prospective_ids - seen_target_ids
    reasons.extend(
        f"prospective protected closure target {target_id!r} is missing"
        for target_id in sorted(missing_prospective)
    )
    return reasons


def _authority_topology_reasons(
    path: str, source: dict[str, Any], node: dict[str, Any]
) -> list[str]:
    reasons: list[str] = []
    identity = source_identity_payload(source)
    if identity["stable_id"] != f"dd:{path}" or identity["source_type"] != "dd":
        reasons.append("stable source identity or source type is inconsistent")
    from_dd = identity["from_dd_paths"]
    if len(from_dd) != 1 or from_dd[0].get("other_id") != path:
        reasons.append("FROM_DD_PATH is not one exact existing edge")
    if (node.get("properties") or {}).get("id") != path:
        reasons.append("backing IMASNode identity is inconsistent")
    units = node.get("units") or []
    if len(units) > 1 or len({item.get("id") for item in units}) != len(units):
        reasons.append("unit authority is ambiguous")
    elif units and node["properties"].get("unit") not in {None, units[0].get("id")}:
        reasons.append("unit scalar and relationship authority disagree")
    parents = node.get("parents") or []
    if len(parents) > 1 or len({item.get("id") for item in parents}) != len(parents):
        reasons.append("parent authority is ambiguous")
    coordinates = node.get("coordinates") or []
    coordinate_ids = [item.get("id") for item in coordinates]
    if None in coordinate_ids or len(coordinate_ids) != len(set(coordinate_ids)):
        reasons.append("coordinate authority is ambiguous")
    reasons.extend(_claim_reasons(source))
    return reasons


def _without_authority_relationships(row: dict[str, Any]) -> dict[str, Any]:
    normalized = copy.deepcopy(row)
    for source in normalized.get("sources") or []:
        source["relationships"] = [
            relationship
            for relationship in source.get("relationships") or []
            if relationship.get("type") not in _AUTHORITY_RELATIONSHIPS
        ]
    return normalized


def _retirement_preserved_row(row: dict[str, Any], target_id: str) -> dict[str, Any]:
    normalized = _without_authority_relationships(row)
    for source in normalized.get("sources") or []:
        source["relationships"] = [
            relationship
            for relationship in source.get("relationships") or []
            if not (
                relationship.get("type") == "PRODUCED_NAME"
                and relationship.get("other_id") == target_id
            )
        ]
        source["names"] = [
            name
            for name in source.get("names") or []
            if (name.get("properties") or {}).get("id") != target_id
        ]
    for node in normalized.get("nodes") or []:
        node["projections"] = [
            projection
            for projection in node.get("projections") or []
            if projection.get("id") != target_id
        ]
    return normalized


def _duplicate_preserved_state(source: dict[str, Any]) -> dict[str, Any]:
    mutable_fields = {
        "status",
        "skip_reason",
        "skip_reason_detail",
    }
    return {
        "element_id": source.get("element_id"),
        "labels": source.get("labels") or [],
        "properties": {
            key: value
            for key, value in (source.get("properties") or {}).items()
            if key not in mutable_fields
        },
        "relationships": [
            relationship
            for relationship in source.get("relationships") or []
            if relationship.get("type") not in _AUTHORITY_RELATIONSHIPS
            and relationship.get("type") != "FROM_DD_PATH"
        ],
        "names": source.get("names") or [],
    }


def _duplicate_destructive_closure(source: dict[str, Any]) -> dict[str, Any]:
    """Return every exact backing relationship a fold is authorized to remove."""
    return {
        "source_element_id": source.get("element_id"),
        "from_dd_paths": [
            relationship
            for relationship in source.get("relationships") or []
            if relationship.get("type") == "FROM_DD_PATH"
            and relationship.get("direction") == "out"
        ],
    }


def _retirement_destructive_closure(
    row: dict[str, Any], target_id: str
) -> dict[str, Any]:
    """Return the exact target and relationships a retirement may detach."""
    sources = row.get("sources") or []
    nodes = row.get("nodes") or []
    source = sources[0] if len(sources) == 1 else {}
    node = nodes[0] if len(nodes) == 1 else {}
    return {
        "source_element_id": source.get("element_id"),
        "targets": [
            name
            for name in source.get("names") or []
            if (name.get("properties") or {}).get("id") == target_id
        ],
        "bindings": [
            relationship
            for relationship in source.get("relationships") or []
            if relationship.get("type") == "PRODUCED_NAME"
            and relationship.get("direction") == "out"
            and relationship.get("other_id") == target_id
        ],
        "projections": [
            projection
            for projection in node.get("projections") or []
            if projection.get("id") == target_id
        ],
    }


def _events_for_operation(
    events: list[dict[str, Any]], operation: str
) -> list[dict[str, Any]]:
    label = _EVENT_LABELS[operation]
    return [event for event in events if label in (event.get("event_labels") or [])]


def _row_relationship_ids(row: dict[str, Any]) -> set[str]:
    """Return relationship element ids from one exact authority closure."""
    relationship_ids: set[str] = set()
    for source in row.get("sources") or []:
        relationship_ids.update(
            str(relationship["element_id"])
            for relationship in source.get("relationships") or []
            if relationship.get("element_id")
        )
        relationship_ids.update(
            str(entry["link_element_id"])
            for entry in source.get("ledger") or []
            if entry.get("link_element_id")
        )
        for name in source.get("names") or []:
            if name.get("binding_element_id"):
                relationship_ids.add(str(name["binding_element_id"]))
            relationship_ids.update(
                str(relationship["element_id"])
                for relationship in name.get("relationships") or []
                if relationship.get("element_id")
            )
    for node in row.get("nodes") or []:
        for key in ("units", "parents", "coordinates", "projections"):
            relationship_ids.update(
                str(relationship["relationship_element_id"])
                for relationship in node.get(key) or []
                if relationship.get("relationship_element_id")
            )
    protection = row.get("target_protection") or {}
    for direct_source in protection.get("direct_sources") or []:
        for source in direct_source.get("matches") or []:
            relationship_ids.update(
                str(binding["element_id"])
                for binding in source.get("bindings") or []
                if binding.get("element_id")
            )
    for target_entry in protection.get("targets") or []:
        for target in target_entry.get("matches") or []:
            relationship_ids.update(
                str(producer["binding_element_id"])
                for producer in target.get("producers") or []
                if producer.get("binding_element_id")
            )
    return relationship_ids


def _source_relationship_ids(source: dict[str, Any]) -> set[str]:
    relationship_ids = {
        str(relationship["element_id"])
        for relationship in source.get("relationships") or []
        if relationship.get("element_id")
    }
    for name in source.get("names") or []:
        if name.get("binding_element_id"):
            relationship_ids.add(str(name["binding_element_id"]))
        relationship_ids.update(
            str(relationship["element_id"])
            for relationship in name.get("relationships") or []
            if relationship.get("element_id")
        )
    return relationship_ids


def _lock_relationships(
    transaction: Any, relationship_ids: set[str] | list[str] | tuple[str, ...]
) -> tuple[str, ...]:
    """Write-lock every exact relationship and prove its cardinality."""
    exact_ids = tuple(sorted(set(relationship_ids)))
    rows = list(transaction.run(_RELATIONSHIP_LOCK_QUERY, element_ids=list(exact_ids)))
    locked = int(dict(rows[0]).get("locked") or 0) if rows else 0
    if locked != len(exact_ids):
        raise SourceAuthorityReconciliationConflict(
            "source-authority relationship set changed before locking"
        )
    return exact_ids


def _retirement_source_payload(
    source: dict[str, Any], target_ids: list[str]
) -> dict[str, Any]:
    return {
        "properties": copy.deepcopy(source.get("properties") or {}),
        "identity": source_identity_payload(source),
        "target_ids": sorted(target_ids),
    }


def _event_id(operation: str, identity: dict[str, Any]) -> str:
    return _EVENT_PREFIXES[operation] + _hash(identity)


def _matching_state_events(
    events: list[dict[str, Any]],
    operation: str,
    *,
    expected: dict[str, Any],
    identity_fields: tuple[str, ...],
) -> list[dict[str, Any]]:
    """Return exact events whose stored identity proves the current state."""
    label = _EVENT_LABELS[operation]
    matches: list[dict[str, Any]] = []
    for entry in events:
        if label not in (entry.get("event_labels") or []):
            continue
        event = entry.get("event_properties") or {}
        if any(event.get(field) != value for field, value in expected.items()):
            continue
        try:
            identity = {field: event[field] for field in identity_fields}
        except KeyError:
            continue
        if event.get("id") != _event_id(operation, identity):
            continue
        matches.append(event)
    return matches


def _expected_hash_reasons(
    manifest_row: dict[str, Any],
    *,
    closure: Any,
    preserved_state_hash: str,
    participant_ids_hash: str,
    before_snapshot_hash: str,
) -> list[str]:
    checks = {
        "expected_source_element_id": closure.source.get("element_id"),
        "expected_before_snapshot_hash": before_snapshot_hash,
        "expected_authority_hash": closure.authority_hash,
        "expected_preserved_state_hash": preserved_state_hash,
        "expected_participant_ids_hash": participant_ids_hash,
    }
    return [
        f"manifest {field} drifted"
        for field, current in checks.items()
        if manifest_row[field] != current
    ]


def _base_plan_context(
    row: dict[str, Any],
    manifest_row: dict[str, Any],
    *,
    manifest_hash: str,
    operation: str,
    authorized_source_ids: frozenset[str],
) -> tuple[
    Any | None,
    list[str],
    str | None,
    str | None,
    tuple[str, ...],
    str | None,
]:
    path = row.get("path")
    reasons: list[str] = []
    if len(row.get("versions") or []) != 1:
        reasons.append("current DDVersion is not unique")
    if len(row.get("sources") or []) != 1:
        reasons.append("exact StandardNameSource is not unique")
    if len(row.get("nodes") or []) != 1:
        reasons.append("current IMASNode is not unique")
    if reasons:
        return None, reasons, None, None, (), None

    operational_row = _without_authority_relationships(row)
    preserved_row = operational_row
    mutable_fields = SNAPSHOT_MUTABLE_FIELDS
    if operation == REPAIR_IDENTITY_SCALAR:
        mutable_fields = frozenset({"source_id"})
    elif operation == RETIRE_NONPARTICIPATING_SOURCE:
        preserved_row = _retirement_preserved_row(
            row, str(manifest_row["expected_target_id"])
        )
        mutable_fields = frozenset(
            {
                "status",
                "produced_sn_id",
                "claimed_at",
                "claim_token",
                "drain_scope_id",
                "drain_scope_claimed_at",
                "drain_claim_scope_id",
                "drain_scope_actionable",
                "skip_reason",
                "skip_reason_detail",
            }
        )
    closure = capture_source_authority_closure(
        operational_row,
        manifest_hash=manifest_hash,
        authorized_source_ids=authorized_source_ids,
        mutable_source_fields=mutable_fields,
    )
    preserved_closure = capture_source_authority_closure(
        preserved_row,
        manifest_hash=manifest_hash,
        authorized_source_ids=authorized_source_ids,
        mutable_source_fields=mutable_fields,
    )
    source = closure.source
    node = closure.node
    reasons.extend(_authority_topology_reasons(str(path), source, node))
    protection = row.get("target_protection") or {}
    if not protection:
        reasons.append("protected target closure is missing")
    else:
        reasons.extend(_target_protection_reasons(protection))
    current_version = closure.version["properties"].get("id")
    if not current_version:
        reasons.append("current DDVersion has no exact id")
    participant_set = tuple(sorted(participant_ids(row)))
    return (
        closure,
        reasons,
        str(current_version) if current_version else None,
        _hash(participant_set),
        participant_set,
        preserved_closure.preserved_state_hash,
    )


def _plan_repair(
    closure: Any,
    manifest_row: dict[str, Any],
    events: list[dict[str, Any]],
    *,
    reason: str,
    run_id: str | None,
    changed_at: str | None,
) -> tuple[str, dict[str, Any] | None, list[str], dict[str, Any]]:
    path = manifest_row["expected_from_dd_path"]
    before = closure.identity_payload
    after = copy.deepcopy(before)
    after["source_id"] = path
    before_hash = _hash(before)
    after_hash = _hash(after)
    event_identity = {
        "source_id": manifest_row["source_id"],
        "before_identity_hash": before_hash,
        "after_identity_hash": after_hash,
        "authority_hash": closure.authority_hash,
    }
    event_id = _event_id(REPAIR_IDENTITY_SCALAR, event_identity)
    if before.get("source_id") == path:
        matches = _matching_state_events(
            events,
            REPAIR_IDENTITY_SCALAR,
            expected={
                "source_id": manifest_row["source_id"],
                "after_identity_hash": after_hash,
                "authority_hash": closure.authority_hash,
            },
            identity_fields=(
                "source_id",
                "before_identity_hash",
                "after_identity_hash",
                "authority_hash",
            ),
        )
        if len(matches) == 1:
            return "already_current", matches[0], [], {"after_source_id": path}
        return "refused", None, ["exact identity lacks one matching repair event"], {}
    reasons = []
    if before.get("source_id") is not None:
        reasons.append("source.source_id is not null")
    if _events_for_operation(events, REPAIR_IDENTITY_SCALAR):
        reasons.append("source authority event exists before identity repair")
    event = {
        "id": event_id,
        "source_id": manifest_row["source_id"],
        "before_source_identity_payload": canonical_payload(before),
        "after_source_identity_payload": canonical_payload(after),
        "before_identity_hash": before_hash,
        "after_identity_hash": after_hash,
        "authority_hash": closure.authority_hash,
        "precondition_hash": closure.precondition_hash,
        "preserved_state_hash": closure.preserved_state_hash,
        "reason": reason,
        "run_id": run_id,
        "changed_at": changed_at,
    }
    return "planned", event, reasons, {"after_source_id": path}


def _plan_snapshot(
    operation: str,
    closure: Any,
    manifest_row: dict[str, Any],
    events: list[dict[str, Any]],
    *,
    current_version: str,
    reason: str,
    run_id: str | None,
    changed_at: str | None,
) -> tuple[str, dict[str, Any] | None, list[str], dict[str, Any]]:
    before = closure.before_snapshot
    after = closure.after_snapshot
    before_hash = _hash(before)
    after_hash = _hash(after)
    event_identity = {
        "source_id": manifest_row["source_id"],
        "before_snapshot_hash": before_hash,
        "after_snapshot_hash": after_hash,
        "authority_hash": closure.authority_hash,
    }
    event_id = _event_id(operation, event_identity)
    if before == after:
        matches = _matching_state_events(
            events,
            operation,
            expected={
                "source_id": manifest_row["source_id"],
                "after_snapshot_hash": after_hash,
                "authority_hash": closure.authority_hash,
            },
            identity_fields=(
                "source_id",
                "before_snapshot_hash",
                "after_snapshot_hash",
                "authority_hash",
            ),
        )
        if len(matches) == 1:
            return "already_current", matches[0], [], {"after": after}
        return (
            "refused",
            None,
            ["current authority payload lacks one matching event"],
            {},
        )

    properties = closure.source["properties"]
    reasons: list[str] = []
    if _events_for_operation(events, operation):
        reasons.append("source authority event exists before snapshot reconciliation")
    if operation == ADOPT_CURRENT_SNAPSHOT:
        if properties.get("dd_version") != current_version:
            reasons.append(
                "same-version adoption requires the unique current DD version"
            )
        if properties.get("dd_snapshot_pinned") is not True:
            reasons.append("same-version adoption requires a pinned source snapshot")
        event = {
            "id": event_id,
            "source_id": manifest_row["source_id"],
            "dd_version": current_version,
            "before_snapshot_hash": before_hash,
            "after_snapshot_hash": after_hash,
            "before_snapshot_payload": canonical_payload(before),
            "after_snapshot_payload": canonical_payload(after),
            "authority_hash": closure.authority_hash,
            "precondition_hash": closure.precondition_hash,
            "preserved_state_hash": closure.preserved_state_hash,
            "classification": classify_snapshot_change(
                manifest_row["expected_from_dd_path"], before, after
            ),
            "reason": reason,
            "run_id": run_id,
            "adopted_at": changed_at,
        }
    else:
        if any(before.get(field) is not None for field in SNAPSHOT_FIELDS):
            reasons.append(
                "snapshot admission requires every prior snapshot field to be null"
            )
        if properties.get("dd_version") is not None:
            reasons.append("snapshot admission cannot invent a prior DD version")
        if properties.get("dd_snapshot_pinned") is not None:
            reasons.append("snapshot admission requires a null pin, not false or true")
        event = {
            "id": event_id,
            "source_id": manifest_row["source_id"],
            "to_dd_version": current_version,
            "before_snapshot_hash": before_hash,
            "after_snapshot_hash": after_hash,
            "before_snapshot_payload": canonical_payload(before),
            "after_snapshot_payload": canonical_payload(after),
            "authority_hash": closure.authority_hash,
            "precondition_hash": closure.precondition_hash,
            "preserved_state_hash": closure.preserved_state_hash,
            "reason": reason,
            "run_id": run_id,
            "admitted_at": changed_at,
        }
    return "planned", event, reasons, {"after": after}


def _plan_fold(
    closure: Any,
    manifest_row: dict[str, Any],
    duplicate_sources: list[dict[str, Any]],
    events: list[dict[str, Any]],
    *,
    current_version: str,
    reason: str,
    run_id: str | None,
    changed_at: str | None,
) -> tuple[str, dict[str, Any] | None, list[str], dict[str, Any]]:
    reasons: list[str] = []
    if len(duplicate_sources) != 1:
        return "refused", None, ["duplicate StandardNameSource is not unique"], {}
    duplicate = duplicate_sources[0]
    duplicate_identity = source_identity_payload(duplicate)
    after_identity = copy.deepcopy(duplicate_identity)
    after_identity["from_dd_paths"] = []
    before_hash = _hash(duplicate_identity)
    after_hash = _hash(after_identity)
    event_identity = {
        "canonical_source_id": manifest_row["source_id"],
        "duplicate_source_id": manifest_row["duplicate_source_id"],
        "before_duplicate_identity_hash": before_hash,
        "after_duplicate_identity_hash": after_hash,
        "authority_hash": closure.authority_hash,
    }
    event_id = _event_id(FOLD_DUPLICATE_SOURCE_IDENTITY, event_identity)
    duplicate_properties = duplicate.get("properties") or {}
    from_dd = duplicate_identity["from_dd_paths"]
    if duplicate_properties.get("status") == "stale" and not from_dd:
        matches = _matching_state_events(
            events,
            FOLD_DUPLICATE_SOURCE_IDENTITY,
            expected={
                "canonical_source_id": manifest_row["source_id"],
                "duplicate_source_id": manifest_row["duplicate_source_id"],
                "after_duplicate_identity_hash": after_hash,
                "authority_hash": closure.authority_hash,
            },
            identity_fields=(
                "canonical_source_id",
                "duplicate_source_id",
                "before_duplicate_identity_hash",
                "after_duplicate_identity_hash",
                "authority_hash",
            ),
        )
        if len(matches) == 1:
            return "already_current", matches[0], [], {}
        return "refused", None, ["retired duplicate lacks one matching fold event"], {}
    if (
        duplicate.get("element_id")
        != manifest_row["expected_duplicate_source_element_id"]
    ):
        reasons.append("manifest expected_duplicate_source_element_id drifted")
    if (
        duplicate_identity.get("source_id")
        != manifest_row["expected_duplicate_source_id"]
    ):
        reasons.append("manifest expected_duplicate_source_id drifted")
    if duplicate_identity.get("source_type") != "dd":
        reasons.append("duplicate source type is not dd")
    if (
        len(from_dd) != 1
        or from_dd[0].get("other_id") != manifest_row["expected_duplicate_from_dd_path"]
    ):
        reasons.append("duplicate source does not have one exact canonical backing")
    if duplicate_properties.get("status") not in {"failed", "stale"}:
        reasons.append("duplicate source is not terminally failed or stale")
    if duplicate.get("names"):
        reasons.append("duplicate source owns a semantic target")
    reasons.extend(_claim_reasons(duplicate, prefix="duplicate source"))
    reasons.extend(_protected_reasons(duplicate))
    duplicate_preserved_hash = _hash(_duplicate_preserved_state(duplicate))
    if (
        duplicate_preserved_hash
        != manifest_row["expected_duplicate_preserved_state_hash"]
    ):
        reasons.append("manifest expected_duplicate_preserved_state_hash drifted")
    duplicate_destructive_hash = _hash(_duplicate_destructive_closure(duplicate))
    if (
        duplicate_destructive_hash
        != manifest_row["expected_duplicate_destructive_closure_hash"]
    ):
        reasons.append("manifest expected_duplicate_destructive_closure_hash drifted")
    if (
        closure.before_snapshot != closure.after_snapshot
        or closure.source["properties"].get("dd_version") != current_version
        or closure.source["properties"].get("dd_snapshot_pinned") is not True
    ):
        reasons.append("canonical source snapshot is not exact current authority")
    if _events_for_operation(events, FOLD_DUPLICATE_SOURCE_IDENTITY):
        reasons.append("canonical source authority event exists before duplicate fold")
    combined_precondition = _hash(
        {
            "canonical": closure.precondition_hash,
            "duplicate": duplicate,
        }
    )
    combined_preserved = _hash(
        {
            "canonical": closure.preserved_state_hash,
            "duplicate": _duplicate_preserved_state(duplicate),
        }
    )
    event = {
        "id": event_id,
        "canonical_source_id": manifest_row["source_id"],
        "duplicate_source_id": manifest_row["duplicate_source_id"],
        "before_duplicate_identity_payload": canonical_payload(duplicate_identity),
        "after_duplicate_identity_payload": canonical_payload(after_identity),
        "before_duplicate_identity_hash": before_hash,
        "after_duplicate_identity_hash": after_hash,
        "authority_hash": closure.authority_hash,
        "precondition_hash": combined_precondition,
        "preserved_state_hash": combined_preserved,
        "reason": reason,
        "run_id": run_id,
        "folded_at": changed_at,
    }
    mutation = {
        "duplicate_source_id": manifest_row["duplicate_source_id"],
        "duplicate_source_element_id": duplicate["element_id"],
        "duplicate_backing_element_id": from_dd[0]["element_id"]
        if len(from_dd) == 1
        else None,
        "reason": reason,
    }
    return "planned", event, reasons, mutation


def _plan_retirement(
    closure: Any,
    manifest_row: dict[str, Any],
    events: list[dict[str, Any]],
    *,
    reason: str,
    run_id: str | None,
    changed_at: str | None,
) -> tuple[str, dict[str, Any] | None, list[str], dict[str, Any]]:
    source = closure.source
    node = closure.node
    target_id = manifest_row["expected_target_id"]
    names = source.get("names") or []
    target_names = [
        name for name in names if (name.get("properties") or {}).get("id") == target_id
    ]
    node_category = (node.get("properties") or {}).get("node_category")
    before_payload = _retirement_source_payload(
        source,
        [(name.get("properties") or {}).get("id") for name in names],
    )
    after_source = copy.deepcopy(source)
    after_properties = after_source["properties"]
    after_properties.update(
        {
            "status": "stale",
            "skip_reason": "nonparticipating_dd_source",
            "skip_reason_detail": reason,
        }
    )
    for cleared_field in (
        "produced_sn_id",
        "claimed_at",
        "claim_token",
        "drain_scope_id",
        "drain_scope_claimed_at",
        "drain_claim_scope_id",
        "drain_scope_actionable",
    ):
        after_properties.pop(cleared_field, None)
    after_payload = _retirement_source_payload(after_source, [])
    before_hash = _hash(before_payload)
    after_hash = _hash(after_payload)
    event_identity = {
        "source_id": manifest_row["source_id"],
        "node_category": node_category,
        "removed_target_ids": [target_id],
        "before_source_hash": before_hash,
        "after_source_hash": after_hash,
        "authority_hash": closure.authority_hash,
    }
    event_id = _event_id(RETIRE_NONPARTICIPATING_SOURCE, event_identity)
    if not names and source["properties"].get("status") == "stale":
        if any(
            projection.get("id") == target_id
            for projection in node.get("projections") or []
        ):
            return (
                "refused",
                None,
                ["retired source retains its DD projection mirror"],
                {},
            )
        matches = _matching_state_events(
            events,
            RETIRE_NONPARTICIPATING_SOURCE,
            expected={
                "source_id": manifest_row["source_id"],
                "node_category": node_category,
                "removed_target_ids": [target_id],
                "after_source_hash": after_hash,
                "authority_hash": manifest_row["expected_authority_hash"],
            },
            identity_fields=(
                "source_id",
                "node_category",
                "removed_target_ids",
                "before_source_hash",
                "after_source_hash",
                "authority_hash",
            ),
        )
        if len(matches) == 1:
            return "already_current", matches[0], [], {}
        return (
            "refused",
            None,
            ["retired source lacks one matching authority event"],
            {},
        )
    reasons: list[str] = []
    if node_category != manifest_row["expected_node_category"]:
        reasons.append("manifest expected_node_category drifted")
    if node_category in SN_SOURCE_CATEGORIES:
        reasons.append("backing node category participates in standard-name generation")
    if len(names) != 1 or len(target_names) != 1:
        reasons.append("retirement requires one exact null-lifecycle target")
    target = target_names[0] if len(target_names) == 1 else None
    if (
        target is not None
        and (target.get("properties") or {}).get("name_stage") is not None
    ):
        reasons.append("retirement target has a materialized lifecycle")
    projections = [
        projection
        for projection in node.get("projections") or []
        if projection.get("id") == target_id
    ]
    if len(projections) != 1:
        reasons.append("retirement requires one exact DD projection mirror")
    if source["properties"].get("produced_sn_id") != target_id:
        reasons.append("produced_sn_id does not equal the exact retirement target")
    retirement_destructive_hash = _hash(
        _retirement_destructive_closure(
            {
                "sources": [source],
                "nodes": [node],
            },
            target_id,
        )
    )
    if (
        retirement_destructive_hash
        != manifest_row["expected_retirement_destructive_closure_hash"]
    ):
        reasons.append("manifest expected_retirement_destructive_closure_hash drifted")
    if _events_for_operation(events, RETIRE_NONPARTICIPATING_SOURCE):
        reasons.append("source authority event exists before retirement")
    event = {
        "id": event_id,
        "source_id": manifest_row["source_id"],
        "dd_path": manifest_row["expected_from_dd_path"],
        "node_category": node_category,
        "removed_target_ids": [target_id],
        "before_source_payload": canonical_payload(before_payload),
        "after_source_payload": canonical_payload(after_payload),
        "before_source_hash": before_hash,
        "after_source_hash": after_hash,
        "authority_hash": closure.authority_hash,
        "precondition_hash": closure.precondition_hash,
        "preserved_state_hash": closure.preserved_state_hash,
        "reason": reason,
        "run_id": run_id,
        "retired_at": changed_at,
    }
    mutation = {
        "target_id": target_id,
        "binding_element_id": target.get("binding_element_id") if target else None,
        "projection_element_id": projections[0].get("relationship_element_id")
        if len(projections) == 1
        else None,
        "reason": reason,
    }
    return "planned", event, reasons, mutation


def _plan_rows(
    rows: list[dict[str, Any]],
    manifest: SourceAuthorityManifest,
    *,
    events_by_source: dict[str, list[dict[str, Any]]],
    duplicates_by_source: dict[str, list[dict[str, Any]]],
    reason: str,
    run_id: str | None,
    changed_at: str | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    plans: list[dict[str, Any]] = []
    refusals: list[dict[str, Any]] = []
    manifest_by_source = {row["source_id"]: row for row in manifest.rows}
    authorized_source_ids = frozenset(manifest.source_ids)
    for row in rows:
        source_id = f"dd:{row['path']}"
        manifest_row = manifest_by_source[source_id]
        (
            closure,
            reasons,
            current_version,
            participant_hash,
            participant_set,
            preserved_state_hash,
        ) = _base_plan_context(
            row,
            manifest_row,
            manifest_hash=manifest.manifest_hash,
            operation=manifest.operation,
            authorized_source_ids=authorized_source_ids,
        )
        if closure is None:
            refusals.append({"source_id": source_id, "reasons": reasons})
            continue
        events = events_by_source.get(source_id, [])
        if manifest.operation == REPAIR_IDENTITY_SCALAR:
            status, event, operation_reasons, mutation = _plan_repair(
                closure,
                manifest_row,
                events,
                reason=reason,
                run_id=run_id,
                changed_at=changed_at,
            )
        elif manifest.operation in {ADOPT_CURRENT_SNAPSHOT, ADMIT_CURRENT_SNAPSHOT}:
            status, event, operation_reasons, mutation = _plan_snapshot(
                manifest.operation,
                closure,
                manifest_row,
                events,
                current_version=str(current_version),
                reason=reason,
                run_id=run_id,
                changed_at=changed_at,
            )
        elif manifest.operation == FOLD_DUPLICATE_SOURCE_IDENTITY:
            status, event, operation_reasons, mutation = _plan_fold(
                closure,
                manifest_row,
                duplicates_by_source.get(str(manifest_row["duplicate_source_id"]), []),
                events,
                current_version=str(current_version),
                reason=reason,
                run_id=run_id,
                changed_at=changed_at,
            )
        else:
            status, event, operation_reasons, mutation = _plan_retirement(
                closure,
                manifest_row,
                events,
                reason=reason,
                run_id=run_id,
                changed_at=changed_at,
            )
        reasons.extend(operation_reasons)
        if (
            status == "already_current"
            and event is not None
            and manifest.operation != FOLD_DUPLICATE_SOURCE_IDENTITY
            and event.get("preserved_state_hash") != preserved_state_hash
        ):
            reasons.append("matching event preserved state drifted")
        if (
            event is not None
            and status == "planned"
            and manifest.operation != FOLD_DUPLICATE_SOURCE_IDENTITY
        ):
            event["preserved_state_hash"] = preserved_state_hash
        before_snapshot_hash = _hash(closure.before_snapshot)
        if status == "planned":
            reasons.extend(
                _expected_hash_reasons(
                    manifest_row,
                    closure=closure,
                    preserved_state_hash=str(preserved_state_hash),
                    participant_ids_hash=str(participant_hash),
                    before_snapshot_hash=before_snapshot_hash,
                )
            )
            identity = closure.identity_payload
            if identity.get("source_id") != manifest_row["expected_source_id"]:
                reasons.append("manifest expected_source_id drifted")
        if reasons or status == "refused":
            refusals.append(
                {
                    "source_id": source_id,
                    "reasons": sorted(set(reasons or ["operation refused"])),
                }
            )
            continue
        plan_preserved_state_hash = (
            str((event or {}).get("preserved_state_hash"))
            if manifest.operation == FOLD_DUPLICATE_SOURCE_IDENTITY
            else str(preserved_state_hash)
        )
        plan_authority_hash = (
            str((event or {}).get("authority_hash"))
            if manifest.operation == RETIRE_NONPARTICIPATING_SOURCE
            and status == "already_current"
            else closure.authority_hash
        )
        plan = {
            "source_id": source_id,
            "path": row["path"],
            "operation": manifest.operation,
            "status": status,
            "source_element_id": closure.source["element_id"],
            "before_snapshot_hash": before_snapshot_hash,
            "after_snapshot_hash": _hash(closure.after_snapshot),
            "authority_hash": plan_authority_hash,
            "precondition_hash": _hash(
                {
                    "closure": row,
                    "events": events,
                    "duplicates": duplicates_by_source,
                }
            ),
            "preserved_state_hash": plan_preserved_state_hash,
            "participant_ids": list(participant_set),
            "relationship_ids": sorted(_row_relationship_ids(row)),
            "event": event,
            "mutation": mutation,
        }
        if manifest.operation == FOLD_DUPLICATE_SOURCE_IDENTITY:
            duplicate = duplicates_by_source[manifest_row["duplicate_source_id"]][0]
            plan["participant_ids"] = sorted(
                set(plan["participant_ids"])
                | {
                    duplicate["element_id"],
                    *(
                        relationship.get("other_element_id")
                        for relationship in duplicate.get("relationships") or []
                        if relationship.get("other_element_id")
                    ),
                }
            )
            plan["relationship_ids"] = sorted(
                set(plan["relationship_ids"]) | _source_relationship_ids(duplicate)
            )
        plans.append(plan)
    return plans, refusals


def _read_plan(
    transaction: Any,
    manifest: SourceAuthorityManifest,
    *,
    reason: str,
    run_id: str | None,
    changed_at: str | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = read_source_authority_rows(transaction, manifest.paths)
    require_complete_paths(
        rows,
        manifest.paths,
        conflict_type=SourceAuthorityReconciliationConflict,
    )
    manifest_by_source = {row["source_id"]: row for row in manifest.rows}
    protection_candidates = []
    for source_id, path in zip(manifest.source_ids, manifest.paths, strict=True):
        manifest_row = manifest_by_source[source_id]
        source_ids = [source_id]
        if manifest.operation == FOLD_DUPLICATE_SOURCE_IDENTITY:
            source_ids.append(str(manifest_row["duplicate_source_id"]))
        prospective_target_ids = []
        if manifest.operation == RETIRE_NONPARTICIPATING_SOURCE:
            prospective_target_ids.append(str(manifest_row["expected_target_id"]))
        protection_candidates.append(
            {
                "path": path,
                "source_ids": source_ids,
                "prospective_target_ids": prospective_target_ids,
            }
        )
    protection_rows = read_source_target_protection_rows(
        transaction, protection_candidates
    )
    require_complete_paths(
        protection_rows,
        manifest.paths,
        conflict_type=SourceAuthorityReconciliationConflict,
        message="protected target closure did not return the complete exact allowlist",
    )
    for row, protection in zip(rows, protection_rows, strict=True):
        row["target_protection"] = protection
    event_sources = list(manifest.source_ids)
    duplicate_ids: tuple[str, ...] = ()
    if manifest.operation == FOLD_DUPLICATE_SOURCE_IDENTITY:
        duplicate_ids = tuple(str(row["duplicate_source_id"]) for row in manifest.rows)
    events = _read_events(transaction, tuple(event_sources))
    duplicates = _read_duplicates(transaction, duplicate_ids) if duplicate_ids else {}
    return _plan_rows(
        rows,
        manifest,
        events_by_source=events,
        duplicates_by_source=duplicates,
        reason=reason,
        run_id=run_id,
        changed_at=changed_at,
    )


def _receipt(
    manifest: SourceAuthorityManifest,
    plans: list[dict[str, Any]],
    refusals: list[dict[str, Any]],
    *,
    apply: bool,
    run_id: str | None,
) -> dict[str, Any]:
    counts = Counter(plan["status"] for plan in plans)
    if refusals:
        mode = "refused"
    elif plans and counts["already_current"] == len(plans):
        mode = "already_current"
    else:
        mode = "applied" if apply else "dry_run"
    receipt = {
        "schema": _RECEIPT_SCHEMA,
        "schema_version": _RECEIPT_SCHEMA_VERSION,
        "mode": mode,
        "operation": manifest.operation,
        "manifest_path": str(manifest.path),
        "manifest_hash": manifest.manifest_hash,
        "allowlist_hash": manifest.allowlist_hash,
        "run_id": run_id if apply else None,
        "counts": {
            "allowlisted": len(manifest.source_ids),
            "planned": counts["planned"],
            "already_current": counts["already_current"],
            "applied": counts["planned"] if mode == "applied" else 0,
            "refused": len(refusals),
        },
        "rows": [
            {
                key: plan[key]
                for key in (
                    "source_id",
                    "path",
                    "operation",
                    "status",
                    "before_snapshot_hash",
                    "after_snapshot_hash",
                    "authority_hash",
                    "precondition_hash",
                    "preserved_state_hash",
                )
            }
            | {"event_id": (plan.get("event") or {}).get("id")}
            for plan in sorted(plans, key=lambda item: item["source_id"])
        ],
        "refusals": sorted(refusals, key=lambda item: item["source_id"]),
    }
    receipt["receipt_hash"] = _hash(receipt)
    return receipt


def plan_source_authority_reconciliation(
    manifest_path: str | Path,
    *,
    reason: str,
    expected_manifest_hash: str | None = None,
    gc: Any | None = None,
) -> dict[str, Any]:
    """Return a zero-write source-authority plan."""
    return reconcile_source_authority(
        manifest_path,
        reason=reason,
        apply=False,
        expected_manifest_hash=expected_manifest_hash,
        gc=gc,
    )


@retry_on_deadlock()
def reconcile_source_authority(
    manifest_path: str | Path,
    *,
    reason: str,
    apply: bool = False,
    expected_manifest_hash: str | None = None,
    run_id: str | None = None,
    gc: Any | None = None,
) -> dict[str, Any]:
    """Plan or atomically apply one exact source-authority cohort."""
    reason = (reason or "").strip()
    if not reason:
        raise ValueError("a source-authority reason is required")
    normalized_hash = normalize_manifest_hash_binding(
        expected_manifest_hash, apply=apply
    )
    manifest = load_source_authority_manifest(manifest_path)
    if normalized_hash is not None and not hmac.compare_digest(
        manifest.manifest_hash, normalized_hash
    ):
        raise ValueError("manifest SHA-256 does not match the exact parsed bytes")
    base_run_id = run_id or (_RUN_PREFIX + str(uuid.uuid4()) if apply else None)
    invocation_run_id = (
        f"{base_run_id}:manifest:{manifest.manifest_hash}"
        if base_run_id is not None
        else None
    )
    changed_at = datetime.now(UTC).isoformat() if apply else None
    own = gc is None
    client = GraphClient() if own else gc
    try:
        with client.session() as session:
            transaction = session.begin_transaction()
            try:
                plans, refusals = _read_plan(
                    transaction,
                    manifest,
                    reason=reason,
                    run_id=invocation_run_id,
                    changed_at=changed_at,
                )
                if refusals:
                    transaction.rollback()
                    return _receipt(
                        manifest,
                        plans,
                        refusals,
                        apply=apply,
                        run_id=invocation_run_id,
                    )
                pending = [plan for plan in plans if plan["status"] == "planned"]
                current = [
                    plan for plan in plans if plan["status"] == "already_current"
                ]
                if pending and current:
                    transaction.rollback()
                    return _receipt(
                        manifest,
                        plans,
                        [
                            {
                                "source_id": "<allowlist>",
                                "reasons": [
                                    "mixed pending and current rows cannot prove one atomic cohort"
                                ],
                            }
                        ],
                        apply=apply,
                        run_id=invocation_run_id,
                    )
                if not apply or not pending:
                    transaction.rollback()
                    return _receipt(
                        manifest,
                        plans,
                        [],
                        apply=apply,
                        run_id=invocation_run_id,
                    )

                lock_participants(
                    transaction,
                    {
                        participant
                        for plan in pending
                        for participant in plan["participant_ids"]
                    },
                    conflict_type=SourceAuthorityReconciliationConflict,
                    message="source-authority participant set changed before locking",
                )
                _lock_relationships(
                    transaction,
                    {
                        relationship
                        for plan in pending
                        for relationship in plan["relationship_ids"]
                    },
                )
                locked_plans, locked_refusals = _read_plan(
                    transaction,
                    manifest,
                    reason=reason,
                    run_id=invocation_run_id,
                    changed_at=changed_at,
                )
                if locked_refusals or [
                    plan["precondition_hash"] for plan in locked_plans
                ] != [plan["precondition_hash"] for plan in plans]:
                    raise SourceAuthorityReconciliationConflict(
                        "source, authority, target, event, or relationship state changed after locks"
                    )
                mutation_rows = list(
                    transaction.run(
                        _APPLY_QUERIES[manifest.operation],
                        items=[
                            {
                                "source_id": plan["source_id"],
                                "path": plan["path"],
                                "source_element_id": plan["source_element_id"],
                                "event": plan["event"],
                                **plan["mutation"],
                            }
                            for plan in pending
                        ],
                    )
                )
                if len(mutation_rows) != 1:
                    raise SourceAuthorityReconciliationConflict(
                        "source-authority mutation result cardinality changed inside the transaction"
                    )
                mutation = dict(mutation_rows[0])
                if set(mutation.get("source_ids") or []) != {
                    plan["source_id"] for plan in pending
                } or set(mutation.get("event_ids") or []) != {
                    plan["event"]["id"] for plan in pending
                }:
                    raise SourceAuthorityReconciliationConflict(
                        "source-authority mutation cardinality changed inside the transaction"
                    )
                post_plans, post_refusals = _read_plan(
                    transaction,
                    manifest,
                    reason=reason,
                    run_id=invocation_run_id,
                    changed_at=changed_at,
                )
                if post_refusals or any(
                    plan["status"] != "already_current" for plan in post_plans
                ):
                    raise SourceAuthorityReconciliationConflict(
                        "postflight source-authority and ledger proof did not hold"
                    )
                before = {plan["source_id"]: plan for plan in pending}
                if any(
                    plan["authority_hash"]
                    != before[plan["source_id"]]["authority_hash"]
                    or plan["preserved_state_hash"]
                    != before[plan["source_id"]]["preserved_state_hash"]
                    for plan in post_plans
                ):
                    raise SourceAuthorityReconciliationConflict(
                        "postflight authority or preserved graph state changed"
                    )
                transaction.commit()
                return _receipt(
                    manifest,
                    pending,
                    [],
                    apply=True,
                    run_id=invocation_run_id,
                )
            except BaseException:
                try:
                    transaction.rollback()
                except Exception:
                    pass
                raise
    finally:
        if own:
            client.close()
