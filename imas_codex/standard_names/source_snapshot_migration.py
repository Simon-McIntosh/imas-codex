"""Governed in-place migration of immutable DD source snapshots.

The stable ``dd:{path}`` source identity is preserved.  A bounded manifest is
used only to derive an exact allowlist; planning and application re-read every
authority from Neo4j.  Application locks and compares the complete source,
backing, projection, name, review, edit, and relationship state in one
transaction before changing snapshot fields and creating an immutable ledger
event.  This module never mutates a :class:`StandardName`.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import re
import uuid
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from imas_codex.discovery.base.claims import retry_on_deadlock
from imas_codex.graph.client import GraphClient
from imas_codex.units.dd_unit_exceptions import units_agree

_MANIFEST_SCHEMA = "imas-codex.bounded-integrity-manifest"
_MANIFEST_SCHEMA_VERSION = 2
_RECEIPT_SCHEMA = "imas-codex.source-snapshot-migration-receipt"
_RECEIPT_SCHEMA_VERSION = 1
_EVENT_PREFIX = "source-snapshot-change:"
_RUN_PREFIX = "source-snapshot-migration:"

_SNAPSHOT_FIELDS = (
    "dd_version",
    "description",
    "physics_domain",
    "dd_documentation",
    "dd_snapshot_pinned",
    "dd_parent_path",
    "dd_parent_documentation",
    "dd_data_type",
    "dd_unit",
    "dd_coordinates",
    "dd_lifecycle_status",
    "dd_lifecycle_version",
    "enhanced_description",
    "enhancement_kind",
)

_DD_AUTHORITY_CLASSIFICATION_FIELDS = (
    "physics_domain",
    "dd_documentation",
    "dd_snapshot_pinned",
    "dd_parent_path",
    "dd_parent_documentation",
    "dd_data_type",
    "dd_unit",
    "dd_coordinates",
    "dd_lifecycle_status",
    "dd_lifecycle_version",
    "enhanced_description",
    "enhancement_kind",
)

if set(_DD_AUTHORITY_CLASSIFICATION_FIELDS) != set(_SNAPSHOT_FIELDS) - {
    "dd_version",
    "description",
}:
    raise RuntimeError(
        "DD source classification fields must cover every authoritative snapshot field"
    )


@dataclass(frozen=True)
class SourceSnapshotAllowlist:
    """Exact DD identities authorized by one bounded manifest."""

    manifest_path: Path
    manifest_hash: str
    source_ids: tuple[str, ...]
    paths: tuple[str, ...]
    allowlist_hash: str
    excluded_counts: dict[str, int]
    excluded_source_ids: dict[str, tuple[str, ...]]


class SourceSnapshotMigrationConflict(RuntimeError):
    """The bounded migration preconditions do not hold exactly."""


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, set | frozenset):
        return sorted((_json_safe(item) for item in value), key=repr)
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, bytes):
        return {"bytes": value.hex()}
    if hasattr(value, "iso_format"):
        return {"temporal": value.iso_format()}
    if hasattr(value, "isoformat"):
        return {"temporal": value.isoformat()}
    return {
        "typed_repr": f"{type(value).__module__}.{type(value).__qualname__}:{value}"
    }


def _typed_value(value: Any, *, key: str = "") -> Any:
    if isinstance(value, dict):
        return [
            "mapping",
            [
                [str(name), _typed_value(item, key=str(name))]
                for name, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            ],
        ]
    if isinstance(value, list | tuple):
        items = [_typed_value(item) for item in value]
        if (
            key
            in {
                "coordinates",
                "dd_coordinates",
                "labels",
                "names",
                "nodes",
                "relationships",
                "sources",
                "versions",
            }
            or value
            and all(isinstance(item, dict) for item in value)
        ):
            items.sort(
                key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":"))
            )
        return [type(value).__name__, items]
    if isinstance(value, set | frozenset):
        items = [_typed_value(item) for item in value]
        items.sort(
            key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":"))
        )
        return [type(value).__name__, items]
    if value is None:
        return ["none"]
    if isinstance(value, bool):
        return ["bool", value]
    if isinstance(value, int):
        return ["int", str(value)]
    if isinstance(value, float):
        return ["float", repr(value)]
    if isinstance(value, str):
        return ["str", value]
    if isinstance(value, bytes):
        return ["bytes", value.hex()]
    scalar_type = f"{type(value).__module__}.{type(value).__qualname__}"
    if hasattr(value, "iso_format"):
        return [scalar_type, value.iso_format()]
    if hasattr(value, "isoformat"):
        return [scalar_type, value.isoformat()]
    return [scalar_type, str(value)]


def canonical_payload(value: Any) -> str:
    """Encode values with stable ordering while preserving scalar types."""
    return json.dumps(_typed_value(value), sort_keys=True, separators=(",", ":"))


def _hash(value: Any) -> str:
    return hashlib.sha256(canonical_payload(value).encode()).hexdigest()


def _validate_source_id(source_id: str) -> str:
    if not isinstance(source_id, str) or not source_id.startswith("dd:"):
        raise ValueError(f"exact DD source identity required, got {source_id!r}")
    path = source_id.removeprefix("dd:")
    if (
        not path
        or path.startswith("/")
        or path.endswith("/")
        or ":" in path
        or "*" in path
        or "?" in path
        or any(char.isspace() for char in path)
    ):
        raise ValueError(f"exact DD source identity required, got {source_id!r}")
    return path


def _record_source_ids(record: dict[str, Any]) -> list[str]:
    participants = record.get("participants") or {}
    result = list(participants.get("source_ids") or [])
    if record.get("source_id"):
        result.append(record["source_id"])
    return [str(source_id) for source_id in result]


def _dd_gap_source_ids(manifest: dict[str, Any]) -> set[str]:
    excluded: set[str] = set()
    for records in (manifest.get("partitions") or {}).values():
        for record in records or []:
            if str(record.get("next_operator") or "").casefold() == "ddgap_flag":
                excluded.update(_record_source_ids(record))
    for value in (manifest.get("special_checks") or {}).values():
        if not isinstance(value, dict):
            continue
        if str(value.get("next_operator") or "").casefold() == "ddgap_flag":
            excluded.update(_record_source_ids(value))
    return excluded


def _manifest_records(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for partition_records in manifest["partitions"].values():
        if not isinstance(partition_records, list):
            raise ValueError("bounded manifest partitions must contain record arrays")
        for record in partition_records:
            if not isinstance(record, dict):
                raise ValueError("bounded manifest records must be objects")
            records.append(record)
    for record in (manifest.get("special_checks") or {}).values():
        if isinstance(record, dict):
            records.append(record)
    return records


def _protected_source_ids(records: list[dict[str, Any]], protection: str) -> set[str]:
    protected: set[str] = set()
    for record in records:
        evidence = record.get("scope_evidence") or {}
        evidence_maps = [record, evidence] if isinstance(evidence, dict) else [record]
        state = " ".join(
            str(record.get(field) or "").casefold()
            for field in ("classification", "scope_status")
        )
        has_protected_closure = protection in state
        for mapping in evidence_maps:
            for key, value in mapping.items():
                normalized_key = str(key).casefold()
                if protection not in normalized_key or not value:
                    continue
                if any(
                    marker in normalized_key
                    for marker in (
                        "closure",
                        "source_hit",
                        "name_hit",
                        "component_hit",
                        "direct_",
                    )
                ):
                    has_protected_closure = True
                if "source" in normalized_key:
                    values = value if isinstance(value, list | tuple | set) else [value]
                    protected.update(
                        str(source_id)
                        for source_id in values
                        if str(source_id).startswith("dd:")
                    )
        if has_protected_closure:
            protected.update(_record_source_ids(record))
    return protected


def load_source_snapshot_allowlist(
    manifest_path: str | Path,
) -> SourceSnapshotAllowlist:
    """Derive a sorted exact DD allowlist from a bounded integrity manifest.

    Executable records are eligible only when their closure evidence is free
    of WEST and test participants.  Any source selected for DD-gap handling is
    removed independently, even if another executable record mentions it.
    """
    path = Path(manifest_path).expanduser().resolve()
    raw = path.read_bytes()
    try:
        manifest = json.loads(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"migration manifest is not valid JSON: {path}") from exc
    if (
        manifest.get("schema") != _MANIFEST_SCHEMA
        or manifest.get("schema_version") != _MANIFEST_SCHEMA_VERSION
        or not isinstance(manifest.get("partitions"), dict)
    ):
        raise ValueError(
            "migration requires a bounded integrity manifest with the supported schema"
        )

    records = _manifest_records(manifest)
    dd_gap_ids = _dd_gap_source_ids(manifest)
    west_ids = _protected_source_ids(records, "west")
    test_ids = _protected_source_ids(records, "test")
    selected: set[str] = set()
    excluded: dict[str, set[str]] = {}

    def exclude(reason: str, source_id: str) -> None:
        excluded.setdefault(reason, set()).add(source_id)

    for partition_records in manifest["partitions"].values():
        for record in partition_records:
            source_ids = _record_source_ids(record)
            if record.get("scope_status") != "executable":
                for source_id in source_ids:
                    exclude("non_executable", source_id)
                continue
            evidence = record.get("scope_evidence") or {}
            west = bool(
                evidence.get("direct_west_source_hits")
                or evidence.get("direct_west_name_hits")
                or evidence.get("west_component_hits")
            )
            test = bool(
                evidence.get("direct_test_source_hits")
                or evidence.get("direct_test_name_hits")
                or evidence.get("test_component_hits")
            )
            for source_id in source_ids:
                if not source_id.startswith("dd:"):
                    exclude("non_dd", source_id)
                elif west:
                    exclude("west", source_id)
                elif test:
                    exclude("test", source_id)
                elif source_id in dd_gap_ids:
                    exclude("dd_gap", source_id)
                else:
                    _validate_source_id(source_id)
                    selected.add(source_id)

    for reason, protected_ids in (
        ("dd_gap", dd_gap_ids),
        ("west", west_ids),
        ("test", test_ids),
    ):
        for source_id in protected_ids:
            exclude(reason, source_id)
        selected.difference_update(protected_ids)

    if not selected:
        raise ValueError(
            "bounded manifest resolved to zero migratable exact DD sources"
        )
    source_ids = tuple(sorted(selected))
    paths = tuple(_validate_source_id(source_id) for source_id in source_ids)
    return SourceSnapshotAllowlist(
        manifest_path=path,
        manifest_hash=hashlib.sha256(raw).hexdigest(),
        source_ids=source_ids,
        paths=paths,
        allowlist_hash=_hash(source_ids),
        excluded_counts={
            reason: len(source_ids) for reason, source_ids in sorted(excluded.items())
        },
        excluded_source_ids={
            reason: tuple(sorted(source_ids))
            for reason, source_ids in sorted(excluded.items())
        },
    )


def classify_snapshot_change(
    path: str, before: dict[str, Any], after: dict[str, Any]
) -> str:
    """Classify one canonical source snapshot without changing lifecycle state."""
    before_semantics = {
        field: before.get(field) for field in _DD_AUTHORITY_CLASSIFICATION_FIELDS
    }
    after_semantics = {
        field: after.get(field) for field in _DD_AUTHORITY_CLASSIFICATION_FIELDS
    }
    if canonical_payload(before_semantics) == canonical_payload(after_semantics):
        return "byte_unchanged"
    differing = {
        key
        for key in set(before_semantics) | set(after_semantics)
        if canonical_payload(before_semantics.get(key))
        != canonical_payload(after_semantics.get(key))
    }
    if differing <= {"dd_unit"} and units_agree(
        before_semantics.get("dd_unit"), after_semantics.get("dd_unit"), path
    ):
        return "semantic_unchanged"
    return "changed"


_SNAPSHOT_QUERY = """
// SOURCE_SNAPSHOT_MIGRATION_SNAPSHOT
UNWIND $paths AS path
CALL () {
  MATCH (version:DDVersion {is_current: true})
  RETURN collect({element_id: elementId(version), labels: labels(version),
                  properties: properties(version)}) AS versions
}
OPTIONAL MATCH (source:StandardNameSource {id: 'dd:' + path})
OPTIONAL MATCH (node:IMASNode {id: path})
WITH path, versions, collect(DISTINCT source) AS source_nodes,
     collect(DISTINCT node) AS authority_nodes
RETURN path, versions,
  [source IN source_nodes WHERE source IS NOT NULL | {
    element_id: elementId(source), labels: labels(source),
    properties: properties(source),
    relationships: [(source)-[relationship]-(other)
      WHERE type(relationship) <> 'HAS_SNAPSHOT_CHANGE' | {
        element_id: elementId(relationship), type: type(relationship),
        direction: CASE WHEN startNode(relationship) = source THEN 'out' ELSE 'in' END,
        properties: properties(relationship),
        other_element_id: elementId(other), other_labels: labels(other),
        other_id: other.id, other_properties: properties(other)
      }],
    ledger: [(source)-[link:HAS_SNAPSHOT_CHANGE]
                    ->(event:StandardNameSourceSnapshotChange) | {
      link_element_id: elementId(link), link_properties: properties(link),
      event_element_id: elementId(event), event_labels: labels(event),
      event_properties: properties(event)
    }],
    names: [(source)-[binding:PRODUCED_NAME]->(name:StandardName) | {
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
  }] AS sources,
  [authority IN authority_nodes WHERE authority IS NOT NULL | {
    element_id: elementId(authority), labels: labels(authority),
    properties: properties(authority),
    units: [(authority)-[unit_link:HAS_UNIT]->(unit:Unit) | {
      relationship_element_id: elementId(unit_link),
      relationship_properties: properties(unit_link),
      element_id: elementId(unit), labels: labels(unit), id: unit.id,
      properties: properties(unit)
    }],
    parents: [(authority)-[parent_link:HAS_PARENT]->(parent:IMASNode) | {
      relationship_element_id: elementId(parent_link),
      relationship_properties: properties(parent_link),
      element_id: elementId(parent), labels: labels(parent), id: parent.id,
      properties: properties(parent)
    }],
    coordinates: [(authority)-[coordinate_link:HAS_COORDINATE]->(coordinate) | {
      relationship_element_id: elementId(coordinate_link),
      relationship_properties: properties(coordinate_link),
      element_id: elementId(coordinate), labels: labels(coordinate), id: coordinate.id,
      properties: properties(coordinate)
    }],
    projections: [(authority)-[projection:HAS_STANDARD_NAME]
                              ->(name:StandardName) | {
      relationship_element_id: elementId(projection),
      relationship_properties: properties(projection),
      element_id: elementId(name), labels: labels(name), id: name.id,
      properties: properties(name)
    }]
  }] AS nodes
ORDER BY path
"""

_LOCK_QUERY = """
// SOURCE_SNAPSHOT_MIGRATION_LOCK
MATCH (participant)
WHERE elementId(participant) IN $element_ids AND participant.id IS NOT NULL
SET participant._source_snapshot_migration_lock = true
REMOVE participant._source_snapshot_migration_lock
RETURN count(participant) AS locked
"""

_APPLY_QUERY = """
// SOURCE_SNAPSHOT_MIGRATION_APPLY
UNWIND $items AS item
MATCH (source:StandardNameSource {id: item.source_id})
WHERE elementId(source) = item.source_element_id
CREATE (event:StandardNameSourceSnapshotChange {
  id: item.event.id,
  source_id: item.event.source_id,
  from_dd_version: item.event.from_dd_version,
  to_dd_version: item.event.to_dd_version,
  before_snapshot_hash: item.event.before_snapshot_hash,
  after_snapshot_hash: item.event.after_snapshot_hash,
  before_snapshot_payload: item.event.before_snapshot_payload,
  after_snapshot_payload: item.event.after_snapshot_payload,
  authority_hash: item.event.authority_hash,
  precondition_hash: item.event.precondition_hash,
  preserved_state_hash: item.event.preserved_state_hash,
  classification: item.event.classification,
  reason: item.event.reason,
  run_id: item.event.run_id,
  changed_at: datetime(item.event.changed_at)
})
CREATE (source)-[:HAS_SNAPSHOT_CHANGE]->(event)
SET source.dd_version = item.after.dd_version,
    source.description = item.after.description,
    source.physics_domain = item.after.physics_domain,
    source.dd_documentation = item.after.dd_documentation,
    source.dd_snapshot_pinned = item.after.dd_snapshot_pinned,
    source.dd_parent_path = item.after.dd_parent_path,
    source.dd_parent_documentation = item.after.dd_parent_documentation,
    source.dd_data_type = item.after.dd_data_type,
    source.dd_unit = item.after.dd_unit,
    source.dd_coordinates = item.after.dd_coordinates,
    source.dd_lifecycle_status = item.after.dd_lifecycle_status,
    source.dd_lifecycle_version = item.after.dd_lifecycle_version,
    source.enhanced_description = item.after.enhanced_description,
    source.enhancement_kind = item.after.enhancement_kind
RETURN collect(source.id) AS source_ids, collect(event.id) AS event_ids
"""


def _rows(transaction: Any, paths: tuple[str, ...]) -> list[dict[str, Any]]:
    return [dict(row) for row in transaction.run(_SNAPSHOT_QUERY, paths=list(paths))]


def _source_snapshot(properties: dict[str, Any]) -> dict[str, Any]:
    return {field: _json_safe(properties.get(field)) for field in _SNAPSHOT_FIELDS}


def _authority_snapshot(
    path: str, node: dict[str, Any], version: dict[str, Any]
) -> dict[str, Any]:
    properties = node["properties"]
    units = node.get("units") or []
    parents = node.get("parents") or []
    coordinates = node.get("coordinates") or []
    unit = properties.get("unit") or (units[0].get("id") if units else None)
    parent = parents[0] if parents else None
    return {
        "dd_version": version["properties"].get("id"),
        "description": properties.get("documentation"),
        "physics_domain": properties.get("physics_domain"),
        "dd_documentation": properties.get("documentation"),
        "dd_snapshot_pinned": True,
        "dd_parent_path": parent.get("id") if parent else None,
        "dd_parent_documentation": (
            parent.get("properties", {}).get("documentation") if parent else None
        ),
        "dd_data_type": properties.get("data_type"),
        "dd_unit": unit,
        "dd_coordinates": sorted(
            coordinate.get("id") for coordinate in coordinates if coordinate.get("id")
        ),
        "dd_lifecycle_status": properties.get("lifecycle_status"),
        "dd_lifecycle_version": properties.get("lifecycle_version"),
        "enhanced_description": properties.get("description"),
        "enhancement_kind": properties.get("enrichment_source"),
    }


def _preserved_state(source: dict[str, Any], node: dict[str, Any]) -> dict[str, Any]:
    properties = {
        key: value
        for key, value in source["properties"].items()
        if key not in _SNAPSHOT_FIELDS
    }
    names = _json_safe(source.get("names") or [])
    for name in names:
        for relationship in name.get("relationships") or []:
            if relationship.get("other_element_id") != source["element_id"]:
                continue
            relationship["other_properties"] = {
                key: value
                for key, value in (relationship.get("other_properties") or {}).items()
                if key not in _SNAPSHOT_FIELDS
            }
    return {
        "source_element_id": source["element_id"],
        "source_labels": source["labels"],
        "source_properties": properties,
        "source_relationships": source.get("relationships") or [],
        "names": names,
        "projections": node.get("projections") or [],
    }


def _participant_ids(row: dict[str, Any]) -> set[str]:
    ids = {
        item.get("element_id")
        for key in ("versions", "sources", "nodes")
        for item in row.get(key) or []
    }
    for source in row.get("sources") or []:
        for relationship in source.get("relationships") or []:
            ids.add(relationship.get("other_element_id"))
        for name in source.get("names") or []:
            ids.add(name.get("element_id"))
            for relationship in name.get("relationships") or []:
                ids.add(relationship.get("other_element_id"))
    for node in row.get("nodes") or []:
        for key in ("units", "parents", "coordinates", "projections"):
            ids.update(item.get("element_id") for item in node.get(key) or [])
    return {str(element_id) for element_id in ids if element_id}


def _event_id(event: dict[str, Any]) -> str:
    identity = {
        key: event[key]
        for key in (
            "source_id",
            "from_dd_version",
            "to_dd_version",
            "before_snapshot_hash",
            "after_snapshot_hash",
            "authority_hash",
        )
    }
    return _EVENT_PREFIX + _hash(identity)


def _claim_reason(source: dict[str, Any]) -> str | None:
    properties = source["properties"]
    if any(
        properties.get(field) is not None
        for field in (
            "claimed_at",
            "claim_token",
            "drain_scope_id",
            "drain_scope_claimed_at",
            "drain_claim_scope_id",
        )
    ):
        return "source has an active worker or bounded-drain claim"
    for name in source.get("names") or []:
        name_properties = name.get("properties") or {}
        if any(
            name_properties.get(field) is not None
            for field in (
                "claimed_at",
                "claim_token",
                "drain_scope_id",
                "drain_scope_claimed_at",
                "drain_claim_scope_id",
            )
        ):
            return f"produced name {name_properties.get('id')!r} has an active claim"
    return None


def _topology_reasons(
    path: str,
    source: dict[str, Any],
    node: dict[str, Any],
) -> list[str]:
    reasons: list[str] = []
    source_properties = source["properties"]
    if (
        source_properties.get("id") != f"dd:{path}"
        or source_properties.get("source_type") != "dd"
        or source_properties.get("source_id") != path
    ):
        reasons.append("exact source identity or type is inconsistent")
    if source_properties.get("dd_snapshot_pinned") is not True:
        reasons.append("source snapshot is not pinned")
    from_dd = [
        relationship
        for relationship in source.get("relationships") or []
        if relationship.get("type") == "FROM_DD_PATH"
        and relationship.get("direction") == "out"
    ]
    if len(from_dd) != 1 or from_dd[0].get("other_id") != path:
        reasons.append("FROM_DD_PATH is not exact")
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
    claim_reason = _claim_reason(source)
    if claim_reason:
        reasons.append(claim_reason)
    return reasons


def _ledger_idempotence(
    source: dict[str, Any],
    *,
    current_snapshot: dict[str, Any],
    authority_hash: str,
    target_version: str,
) -> dict[str, Any] | None:
    current_hash = _hash(current_snapshot)
    matches = []
    for entry in source.get("ledger") or []:
        event = entry.get("event_properties") or {}
        if (
            event.get("source_id") == source["properties"].get("id")
            and event.get("to_dd_version") == target_version
            and event.get("after_snapshot_hash") == current_hash
            and event.get("authority_hash") == authority_hash
        ):
            matches.append(event)
    if len(matches) != 1:
        return None
    event = matches[0]
    try:
        before_payload = event["before_snapshot_payload"]
        after_payload = event["after_snapshot_payload"]
    except KeyError:
        return None
    if (
        hashlib.sha256(before_payload.encode()).hexdigest()
        != event.get("before_snapshot_hash")
        or hashlib.sha256(after_payload.encode()).hexdigest()
        != event.get("after_snapshot_hash")
        or after_payload != canonical_payload(current_snapshot)
        or event.get("id") != _event_id(event)
    ):
        return None
    return event


def _plan_rows(
    rows: list[dict[str, Any]],
    *,
    manifest_hash: str,
    expected_from_version: str,
    reason: str,
    run_id: str | None,
    changed_at: str | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    planned: list[dict[str, Any]] = []
    refusals: list[dict[str, Any]] = []
    for row in rows:
        path = row["path"]
        row_reasons: list[str] = []
        versions = row.get("versions") or []
        sources = row.get("sources") or []
        nodes = row.get("nodes") or []
        if len(versions) != 1:
            row_reasons.append("current DDVersion is not unique")
        if len(sources) != 1:
            row_reasons.append("exact StandardNameSource is not unique")
        if len(nodes) != 1:
            row_reasons.append("current IMASNode is not unique")
        if row_reasons:
            refusals.append({"source_id": f"dd:{path}", "reasons": row_reasons})
            continue
        source = sources[0]
        node = nodes[0]
        version = versions[0]
        row_reasons.extend(_topology_reasons(path, source, node))
        target_version = version["properties"].get("id")
        if not target_version:
            row_reasons.append("current DDVersion has no exact id")
        source_properties = source["properties"]
        before = _source_snapshot(source_properties)
        after = _authority_snapshot(path, node, version)
        authority_hash = _hash({"path": path, "version": version, "node": node})
        preserved_state_hash = _hash(_preserved_state(source, node))
        precondition_hash = _hash(
            {"manifest_hash": manifest_hash, "graph_snapshot": row}
        )

        status = "planned"
        event: dict[str, Any] | None = None
        if source_properties.get("dd_version") == target_version:
            if before != after:
                row_reasons.append(
                    "source claims the current DD version but its snapshot differs"
                )
            else:
                event = _ledger_idempotence(
                    source,
                    current_snapshot=before,
                    authority_hash=authority_hash,
                    target_version=str(target_version),
                )
                if event is None:
                    row_reasons.append(
                        "current source lacks one exact matching migration ledger"
                    )
                else:
                    status = "already_current"
        else:
            if source_properties.get("dd_version") != expected_from_version:
                row_reasons.append("source DD version differs from the expected pin")
            if source.get("ledger"):
                row_reasons.append(
                    "migration ledger exists while the source remains on the old snapshot"
                )

        if row_reasons:
            refusals.append({"source_id": f"dd:{path}", "reasons": row_reasons})
            continue

        classification = (
            str(event["classification"])
            if status == "already_current" and event is not None
            else classify_snapshot_change(path, before, after)
        )
        before_payload = canonical_payload(before)
        after_payload = canonical_payload(after)
        event_properties = event
        if status == "planned":
            event_properties = {
                "source_id": f"dd:{path}",
                "from_dd_version": expected_from_version,
                "to_dd_version": target_version,
                "before_snapshot_hash": hashlib.sha256(
                    before_payload.encode()
                ).hexdigest(),
                "after_snapshot_hash": hashlib.sha256(
                    after_payload.encode()
                ).hexdigest(),
                "before_snapshot_payload": before_payload,
                "after_snapshot_payload": after_payload,
                "authority_hash": authority_hash,
                "precondition_hash": precondition_hash,
                "preserved_state_hash": preserved_state_hash,
                "classification": classification,
                "reason": reason,
                "run_id": run_id,
                "changed_at": changed_at,
            }
            event_properties["id"] = _event_id(event_properties)
        planned.append(
            {
                "source_id": f"dd:{path}",
                "path": path,
                "status": status,
                "classification": classification,
                "source_element_id": source["element_id"],
                "before": before,
                "after": after,
                "before_snapshot_hash": hashlib.sha256(
                    before_payload.encode()
                ).hexdigest(),
                "after_snapshot_hash": hashlib.sha256(
                    after_payload.encode()
                ).hexdigest(),
                "authority_hash": authority_hash,
                "precondition_hash": precondition_hash,
                "preserved_state_hash": preserved_state_hash,
                "event": event_properties,
                "participant_ids": sorted(_participant_ids(row)),
            }
        )
    return planned, refusals


def _receipt(
    allowlist: SourceSnapshotAllowlist,
    planned: list[dict[str, Any]],
    refusals: list[dict[str, Any]],
    *,
    apply: bool,
    run_id: str | None,
) -> dict[str, Any]:
    statuses = Counter(item["status"] for item in planned)
    classifications = Counter(item["classification"] for item in planned)
    if refusals:
        mode = "refused"
    elif planned and statuses["already_current"] == len(planned):
        mode = "already_current"
    else:
        mode = "applied" if apply else "dry_run"
    receipt = {
        "schema": _RECEIPT_SCHEMA,
        "schema_version": _RECEIPT_SCHEMA_VERSION,
        "mode": mode,
        "manifest_path": str(allowlist.manifest_path),
        "manifest_hash": allowlist.manifest_hash,
        "allowlist_hash": allowlist.allowlist_hash,
        "excluded_counts": allowlist.excluded_counts,
        "excluded_source_ids": allowlist.excluded_source_ids,
        "run_id": run_id if apply else None,
        "counts": {
            "allowlisted": len(allowlist.source_ids),
            "planned": statuses["planned"],
            "already_current": statuses["already_current"],
            "applied": statuses["planned"] if mode == "applied" else 0,
            "refused": len(refusals),
            "byte_unchanged": classifications["byte_unchanged"],
            "semantic_unchanged": classifications["semantic_unchanged"],
            "changed": classifications["changed"],
        },
        "rows": [
            {
                key: item[key]
                for key in (
                    "source_id",
                    "path",
                    "status",
                    "classification",
                    "before_snapshot_hash",
                    "after_snapshot_hash",
                    "authority_hash",
                    "precondition_hash",
                    "preserved_state_hash",
                )
            }
            | {"event_id": (item.get("event") or {}).get("id")}
            for item in sorted(planned, key=lambda value: value["source_id"])
        ],
        "refusals": sorted(refusals, key=lambda value: value["source_id"]),
    }
    receipt["receipt_hash"] = _hash(receipt)
    return receipt


def _read_plan(
    transaction: Any,
    allowlist: SourceSnapshotAllowlist,
    *,
    expected_from_version: str,
    reason: str,
    run_id: str | None,
    changed_at: str | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    rows = _rows(transaction, allowlist.paths)
    if [row.get("path") for row in rows] != list(allowlist.paths):
        raise SourceSnapshotMigrationConflict(
            "graph snapshot did not return the complete exact allowlist"
        )
    planned, refusals = _plan_rows(
        rows,
        manifest_hash=allowlist.manifest_hash,
        expected_from_version=expected_from_version,
        reason=reason,
        run_id=run_id,
        changed_at=changed_at,
    )
    return rows, planned, refusals


def plan_source_snapshot_migration(
    manifest_path: str | Path,
    *,
    expected_from_version: str,
    reason: str,
    expected_manifest_hash: str | None = None,
    gc: Any | None = None,
) -> dict[str, Any]:
    """Read and hash an exact migration plan, always rolling back the transaction."""
    return migrate_source_snapshots(
        manifest_path,
        expected_from_version=expected_from_version,
        reason=reason,
        expected_manifest_hash=expected_manifest_hash,
        apply=False,
        gc=gc,
    )


@retry_on_deadlock()
def migrate_source_snapshots(
    manifest_path: str | Path,
    *,
    expected_from_version: str,
    reason: str,
    apply: bool = False,
    expected_manifest_hash: str | None = None,
    run_id: str | None = None,
    gc: Any | None = None,
) -> dict[str, Any]:
    """Plan or atomically apply a bounded DD source snapshot migration."""
    expected_from_version = (expected_from_version or "").strip()
    reason = (reason or "").strip()
    if not expected_from_version:
        raise ValueError("an exact expected source DD version is required")
    if not reason:
        raise ValueError("a migration reason is required")
    normalized_manifest_hash = (
        expected_manifest_hash.strip().casefold()
        if isinstance(expected_manifest_hash, str)
        else None
    )
    if apply and normalized_manifest_hash is None:
        raise ValueError("apply requires an expected manifest SHA-256")
    if (
        normalized_manifest_hash is not None
        and re.fullmatch(r"[0-9a-f]{64}", normalized_manifest_hash) is None
    ):
        raise ValueError("expected manifest SHA-256 must be exactly 64 hex characters")
    allowlist = load_source_snapshot_allowlist(manifest_path)
    if normalized_manifest_hash is not None and not hmac.compare_digest(
        allowlist.manifest_hash, normalized_manifest_hash
    ):
        raise ValueError("manifest SHA-256 does not match the exact parsed bytes")
    base_run_id = run_id or (_RUN_PREFIX + str(uuid.uuid4()) if apply else None)
    invocation_run_id = (
        f"{base_run_id}:manifest:{allowlist.manifest_hash}"
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
                rows, planned, refusals = _read_plan(
                    transaction,
                    allowlist,
                    expected_from_version=expected_from_version,
                    reason=reason,
                    run_id=invocation_run_id,
                    changed_at=changed_at,
                )
                if refusals:
                    transaction.rollback()
                    return _receipt(
                        allowlist,
                        planned,
                        refusals,
                        apply=apply,
                        run_id=invocation_run_id,
                    )
                already_current = [
                    item for item in planned if item["status"] == "already_current"
                ]
                pending = [item for item in planned if item["status"] == "planned"]
                if already_current and pending:
                    transaction.rollback()
                    refusal = [
                        {
                            "source_id": "<allowlist>",
                            "reasons": [
                                "mixed current and old snapshots cannot prove one atomic cohort"
                            ],
                        }
                    ]
                    return _receipt(
                        allowlist,
                        planned,
                        refusal,
                        apply=apply,
                        run_id=invocation_run_id,
                    )
                if not apply or not pending:
                    transaction.rollback()
                    return _receipt(
                        allowlist,
                        planned,
                        [],
                        apply=apply,
                        run_id=invocation_run_id,
                    )

                participant_ids = sorted(
                    {
                        participant
                        for item in pending
                        for participant in item["participant_ids"]
                    }
                )
                lock_rows = list(
                    transaction.run(_LOCK_QUERY, element_ids=participant_ids)
                )
                locked = int(dict(lock_rows[0]).get("locked") or 0) if lock_rows else 0
                if locked != len(participant_ids):
                    raise SourceSnapshotMigrationConflict(
                        "snapshot migration participant set changed before locking"
                    )
                _, locked_plan, locked_refusals = _read_plan(
                    transaction,
                    allowlist,
                    expected_from_version=expected_from_version,
                    reason=reason,
                    run_id=invocation_run_id,
                    changed_at=changed_at,
                )
                if locked_refusals or [
                    item["precondition_hash"] for item in locked_plan
                ] != [item["precondition_hash"] for item in planned]:
                    raise SourceSnapshotMigrationConflict(
                        "source, authority, name, edit, or relationship state changed after locks"
                    )

                mutation_rows = list(
                    transaction.run(
                        _APPLY_QUERY,
                        items=[
                            {
                                "source_id": item["source_id"],
                                "source_element_id": item["source_element_id"],
                                "after": item["after"],
                                "event": item["event"],
                            }
                            for item in pending
                        ],
                    )
                )
                if len(mutation_rows) != 1:
                    raise SourceSnapshotMigrationConflict(
                        "snapshot mutation returned no atomic receipt"
                    )
                mutation = dict(mutation_rows[0])
                if set(mutation.get("source_ids") or []) != {
                    item["source_id"] for item in pending
                } or set(mutation.get("event_ids") or []) != {
                    item["event"]["id"] for item in pending
                }:
                    raise SourceSnapshotMigrationConflict(
                        "snapshot mutation cardinality changed inside the transaction"
                    )

                post_rows = _rows(transaction, allowlist.paths)
                post_plan, post_refusals = _plan_rows(
                    post_rows,
                    manifest_hash=allowlist.manifest_hash,
                    expected_from_version=expected_from_version,
                    reason=reason,
                    run_id=invocation_run_id,
                    changed_at=changed_at,
                )
                if post_refusals or any(
                    item["status"] != "already_current" for item in post_plan
                ):
                    raise SourceSnapshotMigrationConflict(
                        "postflight source and ledger proof did not hold"
                    )
                before_by_id = {item["source_id"]: item for item in pending}
                if any(
                    item["authority_hash"]
                    != before_by_id[item["source_id"]]["authority_hash"]
                    or item["preserved_state_hash"]
                    != before_by_id[item["source_id"]]["preserved_state_hash"]
                    for item in post_plan
                ):
                    raise SourceSnapshotMigrationConflict(
                        "postflight authority or preserved graph state changed"
                    )
                transaction.commit()
                return _receipt(
                    allowlist,
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
