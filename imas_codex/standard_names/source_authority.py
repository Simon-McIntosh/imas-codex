"""Canonical graph closure and compare-and-set primitives for DD sources.

Source-authority transitions share one typed representation of source identity,
DD snapshot authority, graph participants, and state that a transition must
preserve.  Keeping these primitives together ensures that each operator hashes
the same closure and locks the same participant set before it writes.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any

SNAPSHOT_FIELDS = (
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
SNAPSHOT_MUTABLE_FIELDS = frozenset(SNAPSHOT_FIELDS)

DD_AUTHORITY_CLASSIFICATION_FIELDS = (
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

if set(DD_AUTHORITY_CLASSIFICATION_FIELDS) != set(SNAPSHOT_FIELDS) - {
    "dd_version",
    "description",
}:
    raise RuntimeError(
        "DD source classification fields must cover every authoritative snapshot field"
    )


SOURCE_AUTHORITY_CLOSURE_QUERY = """
// SOURCE_AUTHORITY_CLOSURE_SNAPSHOT
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

SOURCE_TARGET_PROTECTION_QUERY = """
// SOURCE_AUTHORITY_TARGET_PROTECTION_CLOSURE
UNWIND $candidates AS candidate
CALL (candidate) {
  UNWIND candidate.source_ids AS requested_source_id
  OPTIONAL MATCH (source:StandardNameSource {id: requested_source_id})
  WITH requested_source_id, collect(DISTINCT source) AS source_nodes
  RETURN collect({
    requested_source_id: requested_source_id,
    matches: [source IN source_nodes WHERE source IS NOT NULL | {
      element_id: elementId(source), labels: labels(source),
      properties: properties(source),
      bindings: [(source)-[binding:PRODUCED_NAME]->(target:StandardName) | {
        element_id: elementId(binding), properties: properties(binding),
        target_element_id: elementId(target), target_id: target.id
      }]
    }]
  }) AS direct_sources
}
CALL (candidate) {
  UNWIND candidate.source_ids AS requested_source_id
  OPTIONAL MATCH (source:StandardNameSource {id: requested_source_id})
  OPTIONAL MATCH (source)-[:PRODUCED_NAME]->(current_target:StandardName)
  RETURN collect(DISTINCT current_target.id) AS current_target_ids
}
CALL (candidate, current_target_ids) {
  UNWIND current_target_ids + candidate.prospective_target_ids AS requested_target_id
  WITH DISTINCT requested_target_id
  WHERE requested_target_id IS NOT NULL
  OPTIONAL MATCH (target:StandardName {id: requested_target_id})
  WITH requested_target_id, collect(DISTINCT target) AS target_nodes
  RETURN collect({
    requested_target_id: requested_target_id,
    matches: [target IN target_nodes WHERE target IS NOT NULL | {
      element_id: elementId(target), labels: labels(target),
      properties: properties(target),
      producers: [(producer:StandardNameSource)-[binding:PRODUCED_NAME]->(target) | {
        source_element_id: elementId(producer), source_labels: labels(producer),
        source_properties: properties(producer),
        binding_element_id: elementId(binding),
        binding_properties: properties(binding)
      }]
    }]
  }) AS targets
}
RETURN candidate.path AS path,
       candidate.prospective_target_ids AS prospective_target_ids,
       direct_sources, targets
ORDER BY path
"""

PARTICIPANT_LOCK_QUERY = """
// SOURCE_AUTHORITY_PARTICIPANT_LOCK
// Touch every participant so the transaction holds their write locks for the
// rest of its statements; the property itself never survives the query.
MATCH (participant)
WHERE elementId(participant) IN $element_ids AND participant.id IS NOT NULL
SET participant._source_authority_participant_lock = true
REMOVE participant._source_authority_participant_lock
RETURN count(participant) AS locked
"""


class SourceAuthorityConflict(RuntimeError):
    """The exact source-authority closure changed during an operation."""


@dataclass(frozen=True)
class SourceAuthorityClosure:
    """Canonical payloads and hashes for one exact DD source closure."""

    source: dict[str, Any]
    node: dict[str, Any]
    version: dict[str, Any]
    before_snapshot: dict[str, Any]
    after_snapshot: dict[str, Any]
    identity_payload: dict[str, Any]
    authority_payload: dict[str, Any]
    authority_hash: str
    precondition_hash: str
    preserved_state_hash: str
    participant_ids: tuple[str, ...]
    participant_ids_hash: str


def json_safe(value: Any) -> Any:
    """Convert driver values into JSON-compatible values without losing type hints."""
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, set | frozenset):
        return sorted((json_safe(item) for item in value), key=repr)
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


def payload_hash(value: Any) -> str:
    """Hash a value through the canonical typed encoding."""
    return hashlib.sha256(canonical_payload(value).encode()).hexdigest()


def validate_source_id(source_id: str) -> str:
    """Return the path encoded by one exact stable DD source identity."""
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


def source_identity_payload(source: dict[str, Any]) -> dict[str, Any]:
    """Build the exact scalar and DD-edge identity payload for one source."""
    properties = source.get("properties") or {}
    edges = [
        {
            "element_id": relationship.get("element_id"),
            "properties": relationship.get("properties") or {},
            "other_element_id": relationship.get("other_element_id"),
            "other_labels": relationship.get("other_labels") or [],
            "other_id": relationship.get("other_id"),
        }
        for relationship in source.get("relationships") or []
        if relationship.get("type") == "FROM_DD_PATH"
        and relationship.get("direction") == "out"
    ]
    return {
        "stable_id": properties.get("id"),
        "source_type": properties.get("source_type"),
        "source_id": properties.get("source_id"),
        "from_dd_paths": edges,
    }


def source_snapshot(properties: dict[str, Any]) -> dict[str, Any]:
    """Build the complete immutable DD snapshot stored on a source."""
    return {field: json_safe(properties.get(field)) for field in SNAPSHOT_FIELDS}


def authority_snapshot(
    path: str, node: dict[str, Any], version: dict[str, Any]
) -> dict[str, Any]:
    """Build a source snapshot from the exact current DD authority closure."""
    properties = node["properties"]
    units = node.get("units") or []
    parents = node.get("parents") or []
    coordinates = node.get("coordinates") or []
    unit = properties.get("unit") or (units[0].get("id") if units else None)
    parent = parents[0] if parents else None
    raw = {
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
    from imas_codex.standard_names.dd_resolutions import resolve_dd_row

    context = resolve_dd_row(
        {
            "path": path,
            "unit": raw["dd_unit"],
            "documentation": raw["dd_documentation"],
            "data_type": raw["dd_data_type"],
            "physics_domain": raw["physics_domain"],
            "coordinates": raw["dd_coordinates"],
            "lifecycle_status": raw["dd_lifecycle_status"],
            "lifecycle_version": raw["dd_lifecycle_version"],
        },
        dd_version=raw["dd_version"],
    )
    effective = context.as_pipeline_item()
    raw.update(
        {
            "dd_unit": effective["unit"],
            "dd_documentation": effective["documentation"],
            "dd_data_type": effective["data_type"],
            "physics_domain": effective["physics_domain"],
            "dd_coordinates": effective["coordinates"],
            "dd_lifecycle_status": effective["lifecycle_status"],
            "dd_lifecycle_version": effective["lifecycle_version"],
            "raw_dd_context": effective["raw_dd_context"],
            "dd_resolution_ids": effective["dd_resolution_ids"],
            "dd_resolution_converged_ids": effective["dd_resolution_converged_ids"],
            "dd_resolution_manifest_digest": effective["dd_resolution_manifest_digest"],
            "_dd_resolution_marker": effective["_dd_resolution_marker"],
        }
    )
    return raw


def authority_payload(
    path: str, node: dict[str, Any], version: dict[str, Any]
) -> dict[str, Any]:
    """Build the exact graph payload whose hash establishes DD authority."""
    return {"path": path, "version": version, "node": node}


def preserved_state(
    source: dict[str, Any],
    node: dict[str, Any],
    *,
    authorized_source_ids: frozenset[str],
    mutable_source_fields: frozenset[str] = SNAPSHOT_MUTABLE_FIELDS,
) -> dict[str, Any]:
    """Build graph state that an authority transition is not allowed to change."""
    properties = {
        key: value
        for key, value in source["properties"].items()
        if key not in mutable_source_fields
    }
    names = json_safe(source.get("names") or [])
    for name in names:
        for relationship in name.get("relationships") or []:
            if (
                "StandardNameSource" not in (relationship.get("other_labels") or [])
                or relationship.get("other_id") not in authorized_source_ids
            ):
                continue
            relationship["other_properties"] = {
                key: value
                for key, value in (relationship.get("other_properties") or {}).items()
                if key not in mutable_source_fields
            }
    return {
        "source_element_id": source["element_id"],
        "source_labels": source["labels"],
        "source_properties": properties,
        "source_relationships": source.get("relationships") or [],
        "names": names,
        "projections": node.get("projections") or [],
    }


def participant_ids(row: dict[str, Any]) -> set[str]:
    """Return every node element id in the exact source-authority closure."""
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
    protection = row.get("target_protection") or {}
    for target_entry in protection.get("targets") or []:
        for target in target_entry.get("matches") or []:
            ids.add(target.get("element_id"))
            ids.update(
                producer.get("source_element_id")
                for producer in target.get("producers") or []
            )
    return {str(element_id) for element_id in ids if element_id}


def capture_source_authority_closure(
    row: dict[str, Any],
    *,
    manifest_hash: str,
    authorized_source_ids: frozenset[str],
    mutable_source_fields: frozenset[str] = SNAPSHOT_MUTABLE_FIELDS,
) -> SourceAuthorityClosure:
    """Canonicalize one row after its exact source, node, and version are proven."""
    source = row["sources"][0]
    node = row["nodes"][0]
    version = row["versions"][0]
    exact_participant_ids = tuple(sorted(participant_ids(row)))
    identity = source_identity_payload(source)
    authority = authority_payload(str(row["path"]), node, version)
    return SourceAuthorityClosure(
        source=source,
        node=node,
        version=version,
        before_snapshot=source_snapshot(source["properties"]),
        after_snapshot=authority_snapshot(str(row["path"]), node, version),
        identity_payload=identity,
        authority_payload=authority,
        authority_hash=payload_hash(authority),
        precondition_hash=payload_hash(
            {"manifest_hash": manifest_hash, "graph_snapshot": row}
        ),
        preserved_state_hash=payload_hash(
            preserved_state(
                source,
                node,
                authorized_source_ids=authorized_source_ids,
                mutable_source_fields=mutable_source_fields,
            )
        ),
        participant_ids=exact_participant_ids,
        participant_ids_hash=payload_hash(exact_participant_ids),
    )


def read_source_authority_rows(
    transaction: Any, paths: tuple[str, ...]
) -> list[dict[str, Any]]:
    """Read the complete exact graph closure for sorted DD paths."""
    return [
        dict(row)
        for row in transaction.run(SOURCE_AUTHORITY_CLOSURE_QUERY, paths=list(paths))
    ]


def read_source_target_protection_rows(
    transaction: Any, candidates: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Read protected producers for all current and prospective targets in one batch."""
    return [
        dict(row)
        for row in transaction.run(
            SOURCE_TARGET_PROTECTION_QUERY,
            candidates=candidates,
        )
    ]


def require_complete_paths(
    rows: list[dict[str, Any]],
    paths: tuple[str, ...],
    *,
    conflict_type: type[Exception] = SourceAuthorityConflict,
    message: str = "graph snapshot did not return the complete exact allowlist",
) -> None:
    """Refuse when a closure read is missing, duplicated, or reordered."""
    if [row.get("path") for row in rows] != list(paths):
        raise conflict_type(message)


def lock_participants(
    transaction: Any,
    element_ids: list[str] | tuple[str, ...] | set[str],
    *,
    conflict_type: type[Exception] = SourceAuthorityConflict,
    message: str = "source-authority participant set changed before locking",
) -> tuple[str, ...]:
    """Write-lock every exact participant and prove the lock cardinality."""
    exact_ids = tuple(sorted(set(element_ids)))
    lock_rows = list(
        transaction.run(PARTICIPANT_LOCK_QUERY, element_ids=list(exact_ids))
    )
    locked = int(dict(lock_rows[0]).get("locked") or 0) if lock_rows else 0
    if locked != len(exact_ids):
        raise conflict_type(message)
    return exact_ids


def normalize_manifest_hash_binding(
    expected_manifest_hash: str | None, *, apply: bool
) -> str | None:
    """Validate the expected exact-byte manifest digest before graph access."""
    normalized = (
        expected_manifest_hash.strip().casefold()
        if isinstance(expected_manifest_hash, str)
        else None
    )
    if apply and normalized is None:
        raise ValueError("apply requires an expected manifest SHA-256")
    if normalized is not None and re.fullmatch(r"[0-9a-f]{64}", normalized) is None:
        raise ValueError("expected manifest SHA-256 must be exactly 64 hex characters")
    return normalized
