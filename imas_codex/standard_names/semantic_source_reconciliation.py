"""Manifest-bound reconciliation of semantic source-to-name bindings.

The operator repairs only the three redundant representations of one semantic
binding: ``PRODUCED_NAME``, ``produced_sn_id``, and the backing entity's
``HAS_STANDARD_NAME`` projection.  Every authority and protection input is
captured in the manifest and compared again under write locks.  Name content,
review state, DD metadata, units, and COCOS transformation labels are read-only.
"""

from __future__ import annotations

import copy
import hashlib
import hmac
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from imas_codex.graph.client import GraphClient
from imas_codex.standard_names.source_authority import canonical_payload, payload_hash

MANIFEST_SCHEMA = "imas-codex.semantic-source-reconciliation-manifest"
RECEIPT_SCHEMA = "imas-codex.semantic-source-reconciliation-receipt"
EVENT_PREFIX = "sn-change:semantic-source-reconciliation:"
SUPPORTED_SOURCE_TYPES = frozenset({"dd", "facility_signal"})

MANIFEST_ROW_FIELDS = frozenset(
    {
        "source_id",
        "source_type",
        "prospective_target_id",
        "reviewed_override",
        "review_approval",
        "expected_closure_hash",
        "expected_source_identity_hash",
        "expected_lifecycle_hash",
        "expected_scalar_hash",
        "expected_targets_hash",
        "expected_backing_hash",
        "expected_projection_hash",
        "participant_ids",
        "participant_relationship_ids",
        "expected_protection_hash",
    }
)

CLOSURE_QUERY = """
// SEMANTIC_SOURCE_RECONCILIATION_CLOSURE
UNWIND $candidates AS candidate
OPTIONAL MATCH (source:StandardNameSource {id: candidate.source_id})
WITH candidate, collect(DISTINCT source) AS source_nodes
CALL (candidate) {
  OPTIONAL MATCH (target:StandardName {id: candidate.prospective_target_id})
  WITH collect(DISTINCT target) AS target_nodes
  RETURN [target IN target_nodes WHERE target IS NOT NULL | {
    element_id: elementId(target), labels: labels(target),
    properties: properties(target)}] AS prospective_targets
}
CALL () {
  MATCH (version:DDVersion {is_current: true})
  RETURN collect({element_id: elementId(version),
    properties: properties(version)}) AS dd_versions
}
CALL (candidate, source_nodes) {
  UNWIND source_nodes AS protected_source
  OPTIONAL MATCH (protected_source)-[:PRODUCED_NAME]->(current:StandardName)
  WITH candidate, collect(DISTINCT current.id) +
    [candidate.prospective_target_id] AS requested_ids
  UNWIND requested_ids AS requested_id
  WITH DISTINCT requested_id WHERE requested_id IS NOT NULL
  MATCH (requested:StandardName {id: requested_id})
  OPTIONAL MATCH path=(requested)-[:HAS_PARENT*0..]->(protected_name:StandardName)
  WITH DISTINCT protected_name, relationships(path) AS path_relationships
  RETURN collect({
    element_id: elementId(protected_name), properties: properties(protected_name),
    path_relationships: [relationship IN path_relationships | {
      relationship_element_id: elementId(relationship),
      start_element_id: elementId(startNode(relationship)),
      end_element_id: elementId(endNode(relationship)),
      properties: properties(relationship)}],
    producers: [(producer:StandardNameSource)-[binding:PRODUCED_NAME]->
      (protected_name) | {source_element_id: elementId(producer),
        source_properties: properties(producer),
        relationship_element_id: elementId(binding)}]
  }) AS protected_names
}
RETURN candidate.source_id AS source_id,
  candidate.prospective_target_id AS prospective_target_id,
  [source IN source_nodes WHERE source IS NOT NULL | {
    element_id: elementId(source), labels: labels(source), properties: properties(source),
    bindings: [(source)-[binding:PRODUCED_NAME]->(target:StandardName) | {
      relationship_element_id: elementId(binding), properties: properties(binding),
      target_element_id: elementId(target), target_id: target.id,
      target_properties: properties(target)}],
    dd_backings: [(source)-[backing:FROM_DD_PATH]->(entity:IMASNode) | {
      relationship_element_id: elementId(backing), relationship_properties: properties(backing),
      element_id: elementId(entity), labels: labels(entity), properties: properties(entity),
      projections: [(entity)-[projection:HAS_STANDARD_NAME]->(name:StandardName) | {
        relationship_element_id: elementId(projection), properties: properties(projection),
        target_element_id: elementId(name), target_id: name.id,
        target_properties: properties(name)}]}],
    signal_backings: [(source)-[backing:FROM_FACILITY_SIGNAL]->(entity:FacilitySignal) | {
      relationship_element_id: elementId(backing), relationship_properties: properties(backing),
      element_id: elementId(entity), labels: labels(entity), properties: properties(entity),
      projections: [(entity)-[projection:HAS_STANDARD_NAME]->(name:StandardName) | {
        relationship_element_id: elementId(projection), properties: properties(projection),
        target_element_id: elementId(name), target_id: name.id,
        target_properties: properties(name)}]}],
    events: [(source)-[event_link:HAS_INTERNAL_CHANGE]->
      (event:StandardNameChange) WHERE event.operation = 'reconcile_semantic_source' | {
        relationship_element_id: elementId(event_link), properties: properties(event_link),
        event_element_id: elementId(event), event_properties: properties(event)}]
  }] AS sources,
  prospective_targets,
  [(producer:StandardNameSource)-[binding:PRODUCED_NAME]->
      (target:StandardName {id: candidate.prospective_target_id}) | {
    source_element_id: elementId(producer), source_properties: properties(producer),
    relationship_element_id: elementId(binding), target_element_id: elementId(target),
    target_properties: properties(target)}] AS prospective_producers,
  dd_versions, protected_names
ORDER BY source_id
"""

PARTICIPANT_LOCK_QUERY = """
// SEMANTIC_SOURCE_RECONCILIATION_PARTICIPANT_LOCK
MATCH (participant) WHERE elementId(participant) IN $element_ids
SET participant._semantic_source_lock = true
REMOVE participant._semantic_source_lock
RETURN count(participant) AS locked
"""

RELATIONSHIP_LOCK_QUERY = """
// SEMANTIC_SOURCE_RECONCILIATION_RELATIONSHIP_LOCK
UNWIND $locks AS item
MATCH (anchor) WHERE elementId(anchor) = item.anchor_element_id
MATCH (anchor)-[relationship]-()
WHERE elementId(relationship) = item.relationship_element_id
SET relationship._semantic_source_lock = true
REMOVE relationship._semantic_source_lock
RETURN count(DISTINCT relationship) AS locked
"""

EVENT_COLLISION_QUERY = """
// SEMANTIC_SOURCE_RECONCILIATION_EVENT_COLLISIONS
UNWIND $event_ids AS event_id
OPTIONAL MATCH (event:StandardNameChange {id: event_id})
RETURN event_id, [item IN collect(event) WHERE item IS NOT NULL | {
  element_id: elementId(item), properties: properties(item),
  links: [(source:StandardNameSource)-[link:HAS_INTERNAL_CHANGE]->(item) | {
    source_id: source.id, relationship_element_id: elementId(link)}]}] AS matches
ORDER BY event_id
"""

APPLY_QUERY = """
// SEMANTIC_SOURCE_RECONCILIATION_APPLY
UNWIND $items AS item
MATCH (source:StandardNameSource {id: item.source_id})
WHERE elementId(source) = item.source_element_id
MATCH (target:StandardName {id: item.target_id})
WHERE elementId(target) = item.target_element_id
CALL (source) {
  OPTIONAL MATCH (source)-[binding:PRODUCED_NAME]->(:StandardName)
  DELETE binding
  RETURN count(*) AS ignored_bindings
}
MERGE (source)-[:PRODUCED_NAME]->(target)
SET source.produced_sn_id = item.target_id
WITH source, target, item
CALL (item, target) {
  OPTIONAL MATCH (dd:IMASNode) WHERE elementId(dd) = item.backing_element_id
  OPTIONAL MATCH (dd)-[old_projection:HAS_STANDARD_NAME]->(:StandardName)
  DELETE old_projection
  FOREACH (_ IN CASE WHEN dd IS NULL THEN [] ELSE [1] END |
    MERGE (dd)-[:HAS_STANDARD_NAME]->(target))
  RETURN count(*) AS ignored_dd
}
CALL (item, target) {
  OPTIONAL MATCH (signal:FacilitySignal)
  WHERE elementId(signal) = item.backing_element_id
  OPTIONAL MATCH (signal)-[old_projection:HAS_STANDARD_NAME]->(:StandardName)
  DELETE old_projection
  FOREACH (_ IN CASE WHEN signal IS NULL THEN [] ELSE [1] END |
    MERGE (signal)-[:HAS_STANDARD_NAME]->(target))
  RETURN count(*) AS ignored_signals
}
CREATE (event:StandardNameChange)
SET event = item.event
CREATE (source)-[:HAS_INTERNAL_CHANGE]->(event)
WITH source, target, event, item
CALL (item) {
  UNWIND item.affected_target_ids AS affected_id
  MATCH (affected:StandardName {id: affected_id})
  OPTIONAL MATCH (backing)-[:HAS_STANDARD_NAME]->(affected)
  WITH affected, backing,
    CASE WHEN backing:IMASNode THEN 'dd:' + backing.id ELSE backing.id END AS path
  ORDER BY path
  WITH affected, [value IN collect(DISTINCT path)
    WHERE value IS NOT NULL | value] AS paths
  SET affected.source_paths = paths
  RETURN count(*) AS rebuilt
}
RETURN collect(source.id) AS source_ids, collect(event.id) AS event_ids
"""


class SemanticSourceConflict(RuntimeError):
    """The exact semantic-source authority changed during reconciliation."""


@dataclass(frozen=True)
class SemanticSourceManifest:
    """Validated exact-byte authority for one homogeneous source cohort."""

    path: Path
    manifest_hash: str
    rows: tuple[dict[str, Any], ...]
    source_ids: tuple[str, ...]
    allowlist_hash: str


def _require_hash(value: Any, field: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{field} must be exactly 64 lowercase hex characters")
    return value


def _source_type(source_id: str) -> str:
    if source_id.startswith("dd:"):
        return "dd"
    if source_id.startswith("signals:"):
        return "facility_signal"
    return "derived"


def load_semantic_source_manifest(path: str | Path) -> SemanticSourceManifest:
    """Validate and hash exact manifest bytes before any graph access."""
    manifest_path = Path(path)
    raw = manifest_path.read_bytes()
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("semantic-source manifest is not valid JSON") from exc
    if not isinstance(payload, dict) or set(payload) != {
        "schema",
        "schema_version",
        "rows",
    }:
        raise ValueError("semantic-source manifest top-level fields are not exact")
    if payload["schema"] != MANIFEST_SCHEMA or payload["schema_version"] != 1:
        raise ValueError("semantic-source manifest schema is unsupported")
    rows = payload["rows"]
    if not isinstance(rows, list) or not rows:
        raise ValueError("semantic-source manifest requires rows")
    normalized: list[dict[str, Any]] = []
    seen_sources: set[str] = set()
    relationship_owners: dict[str, tuple[str, str]] = {}
    for item in rows:
        if not isinstance(item, dict) or set(item) != MANIFEST_ROW_FIELDS:
            raise ValueError("semantic-source manifest row fields are not exact")
        row = copy.deepcopy(item)
        source_id = row["source_id"]
        if not isinstance(source_id, str) or source_id in seen_sources:
            raise ValueError("semantic-source manifest source ids must be unique")
        seen_sources.add(source_id)
        inferred_type = _source_type(source_id)
        if (
            row["source_type"] not in SUPPORTED_SOURCE_TYPES
            or inferred_type != row["source_type"]
        ):
            raise ValueError(
                "semantic-source manifest has unsupported source authority"
            )
        target_id = row["prospective_target_id"]
        if not isinstance(target_id, str) or not target_id.strip():
            raise ValueError("semantic-source manifest requires an exact target id")
        if not isinstance(row["reviewed_override"], bool):
            raise ValueError("reviewed_override must be boolean")
        if row["reviewed_override"] != bool(row["review_approval"]):
            raise ValueError("reviewed overrides require independent approval")
        for field in MANIFEST_ROW_FIELDS:
            if field.startswith("expected_") and field.endswith("_hash"):
                _require_hash(row[field], field)
        for field in ("participant_ids", "participant_relationship_ids"):
            values = row[field]
            if (
                not isinstance(values, list)
                or values != sorted(set(values))
                or not all(isinstance(value, str) and value for value in values)
            ):
                raise ValueError(f"{field} must be a sorted unique exact list")
        for relationship_id in row["participant_relationship_ids"]:
            owner = relationship_owners.get(relationship_id)
            intended = (str(row["source_type"]), target_id)
            if owner is not None and owner != intended:
                raise ValueError(
                    "manifest rows overlap one relationship with conflicting intent"
                )
            relationship_owners[relationship_id] = intended
        normalized.append(row)
    normalized.sort(key=lambda row: row["source_id"])
    source_ids = tuple(str(row["source_id"]) for row in normalized)
    return SemanticSourceManifest(
        path=manifest_path,
        manifest_hash=hashlib.sha256(raw).hexdigest(),
        rows=tuple(normalized),
        source_ids=source_ids,
        allowlist_hash=payload_hash(source_ids),
    )


def _normalized_row(row: dict[str, Any]) -> dict[str, Any]:
    normalized = copy.deepcopy(row)
    for key in (
        "sources",
        "prospective_targets",
        "prospective_producers",
        "dd_versions",
        "protected_names",
    ):
        normalized[key] = sorted(
            normalized.get(key) or [],
            key=lambda item: (str(item.get("element_id")), canonical_payload(item)),
        )
    for source in normalized["sources"]:
        for key in ("bindings", "dd_backings", "signal_backings", "events"):
            source[key] = sorted(
                source.get(key) or [],
                key=lambda item: canonical_payload(item),
            )
        for backing in source["dd_backings"] + source["signal_backings"]:
            backing["projections"] = sorted(
                backing.get("projections") or [],
                key=lambda item: canonical_payload(item),
            )
    return normalized


def _identity(source: dict[str, Any]) -> dict[str, Any]:
    properties = source.get("properties") or {}
    return {
        "element_id": source.get("element_id"),
        "id": properties.get("id"),
        "source_type": properties.get("source_type"),
        "source_id": properties.get("source_id"),
        "backing_relationships": [
            {
                "relationship_element_id": backing.get("relationship_element_id"),
                "element_id": backing.get("element_id"),
                "id": (backing.get("properties") or {}).get("id"),
            }
            for backing in source.get("dd_backings", [])
            + source.get("signal_backings", [])
        ],
    }


def _lifecycle(source: dict[str, Any]) -> dict[str, Any]:
    properties = source.get("properties") or {}
    fields = (
        "status",
        "validation_status",
        "claimed_at",
        "claim_token",
        "drain_scope_id",
        "drain_scope_claimed_at",
        "drain_claim_scope_id",
        "open_edit_id",
    )
    return {field: properties.get(field) for field in fields}


def _scalar(source: dict[str, Any]) -> dict[str, Any]:
    properties = source.get("properties") or {}
    return {"produced_sn_id": properties.get("produced_sn_id")}


def _targets(source: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "relationship_element_id": binding.get("relationship_element_id"),
            "properties": binding.get("properties") or {},
            "target_element_id": binding.get("target_element_id"),
            "target_id": binding.get("target_id"),
            "target_stage": (binding.get("target_properties") or {}).get("name_stage"),
            "validation_status": (binding.get("target_properties") or {}).get(
                "validation_status"
            ),
        }
        for binding in source.get("bindings") or []
    ]


def _backings(source: dict[str, Any]) -> list[dict[str, Any]]:
    return copy.deepcopy(
        source.get("dd_backings", []) + source.get("signal_backings", [])
    )


def _backing_identity(source: dict[str, Any]) -> list[dict[str, Any]]:
    backings = _backings(source)
    for backing in backings:
        backing.pop("projections", None)
    return backings


def _projections(source: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        projection
        for backing in _backings(source)
        for projection in backing.get("projections") or []
    ]


def _protection(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "prospective_targets": row.get("prospective_targets") or [],
        "prospective_producers": row.get("prospective_producers") or [],
        "source": row.get("sources") or [],
        "protected_names": row.get("protected_names") or [],
    }


def _relationship_ids(row: dict[str, Any]) -> tuple[str, ...]:
    ids: set[str] = set()
    for source in row.get("sources") or []:
        for binding in source.get("bindings") or []:
            if binding.get("relationship_element_id"):
                ids.add(str(binding["relationship_element_id"]))
        for backing in _backings(source):
            if backing.get("relationship_element_id"):
                ids.add(str(backing["relationship_element_id"]))
            ids.update(
                str(projection["relationship_element_id"])
                for projection in backing.get("projections") or []
                if projection.get("relationship_element_id")
            )
        for event in source.get("events") or []:
            if event.get("relationship_element_id"):
                ids.add(str(event["relationship_element_id"]))
    ids.update(
        str(producer["relationship_element_id"])
        for producer in row.get("prospective_producers") or []
        if producer.get("relationship_element_id")
    )
    for protected_name in row.get("protected_names") or []:
        ids.update(
            str(relationship["relationship_element_id"])
            for relationship in protected_name.get("path_relationships") or []
            if relationship.get("relationship_element_id")
        )
        ids.update(
            str(producer["relationship_element_id"])
            for producer in protected_name.get("producers") or []
            if producer.get("relationship_element_id")
        )
    return tuple(sorted(ids))


def _relationship_lock_items(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Anchor exact relationship locks at already-bound participant nodes."""
    locks: dict[str, str] = {}

    def add(relationship_id: Any, anchor_id: Any) -> None:
        if relationship_id and anchor_id:
            normalized_relationship = str(relationship_id)
            normalized_anchor = str(anchor_id)
            prior = locks.get(normalized_relationship)
            if prior is not None and prior != normalized_anchor:
                raise SemanticSourceConflict(
                    "relationship closure has conflicting lock anchors"
                )
            locks[normalized_relationship] = normalized_anchor

    for row in rows:
        for source in row.get("sources") or []:
            source_element_id = source.get("element_id")
            for binding in source.get("bindings") or []:
                add(binding.get("relationship_element_id"), source_element_id)
            for backing in _backings(source):
                add(backing.get("relationship_element_id"), source_element_id)
                for projection in backing.get("projections") or []:
                    add(
                        projection.get("relationship_element_id"),
                        backing.get("element_id"),
                    )
            for event in source.get("events") or []:
                add(event.get("relationship_element_id"), source_element_id)
        for producer in row.get("prospective_producers") or []:
            add(
                producer.get("relationship_element_id"),
                producer.get("source_element_id"),
            )
        for protected_name in row.get("protected_names") or []:
            for relationship in protected_name.get("path_relationships") or []:
                add(
                    relationship.get("relationship_element_id"),
                    relationship.get("start_element_id"),
                )
            for producer in protected_name.get("producers") or []:
                add(
                    producer.get("relationship_element_id"),
                    producer.get("source_element_id"),
                )
    return [
        {"relationship_element_id": relationship_id, "anchor_element_id": anchor_id}
        for relationship_id, anchor_id in sorted(locks.items())
    ]


def _participant_ids(row: dict[str, Any]) -> tuple[str, ...]:
    ids: set[str] = set()
    for source in row.get("sources") or []:
        if source.get("element_id"):
            ids.add(str(source["element_id"]))
        for backing in _backings(source):
            if backing.get("element_id"):
                ids.add(str(backing["element_id"]))
            ids.update(
                str(projection["target_element_id"])
                for projection in backing.get("projections") or []
                if projection.get("target_element_id")
            )
        ids.update(
            str(binding["target_element_id"])
            for binding in source.get("bindings") or []
            if binding.get("target_element_id")
        )
    ids.update(
        str(target["element_id"])
        for target in row.get("prospective_targets") or []
        if target.get("element_id")
    )
    ids.update(
        str(producer["source_element_id"])
        for producer in row.get("prospective_producers") or []
        if producer.get("source_element_id")
    )
    for protected_name in row.get("protected_names") or []:
        if protected_name.get("element_id"):
            ids.add(str(protected_name["element_id"]))
        ids.update(
            str(producer["source_element_id"])
            for producer in protected_name.get("producers") or []
            if producer.get("source_element_id")
        )
    return tuple(sorted(ids))


def build_semantic_source_manifest_row(
    row: dict[str, Any],
    *,
    prospective_target_id: str,
    reviewed_override: bool = False,
    review_approval: str | None = None,
) -> dict[str, Any]:
    """Capture one complete semantic binding authority snapshot."""
    normalized = _normalized_row(row)
    sources = normalized.get("sources") or []
    if len(sources) != 1:
        raise ValueError("manifest planning requires one exact source")
    source = sources[0]
    source_id = str(normalized["source_id"])
    return {
        "source_id": source_id,
        "source_type": _source_type(source_id),
        "prospective_target_id": prospective_target_id,
        "reviewed_override": reviewed_override,
        "review_approval": review_approval,
        "expected_closure_hash": payload_hash(normalized),
        "expected_source_identity_hash": payload_hash(_identity(source)),
        "expected_lifecycle_hash": payload_hash(_lifecycle(source)),
        "expected_scalar_hash": payload_hash(_scalar(source)),
        "expected_targets_hash": payload_hash(_targets(source)),
        "expected_backing_hash": payload_hash(_backing_identity(source)),
        "expected_projection_hash": payload_hash(_projections(source)),
        "participant_ids": list(_participant_ids(normalized)),
        "participant_relationship_ids": list(_relationship_ids(normalized)),
        "expected_protection_hash": payload_hash(_protection(normalized)),
    }


def _walk_protection(value: Any, reasons: set[str], key: str = "") -> None:
    if isinstance(value, dict):
        for child_key, child in value.items():
            _walk_protection(child, reasons, str(child_key))
        return
    if isinstance(value, list | tuple):
        for child in value:
            _walk_protection(child, reasons, key)
        return
    normalized_key = key.casefold()
    if value is not None and normalized_key in {
        "claimed_at",
        "claim_token",
        "drain_scope_id",
        "drain_scope_claimed_at",
        "drain_claim_scope_id",
        "open_edit_id",
    }:
        reasons.add("protected closure has a current claim, drain scope, or open edit")
    if not isinstance(value, str):
        return
    normalized = value.casefold()
    if normalized_key in {"facility", "facility_id"} and normalized == "west":
        reasons.add("protected closure intersects WEST")
    if normalized_key in {"id", "source_id", "target_id"}:
        if normalized.startswith(("west:", "signals:west:")):
            reasons.add("protected closure intersects WEST")
        if normalized.startswith(("test:", "fixture:", "signals:test:")):
            reasons.add("protected closure intersects test fixtures")
    if normalized_key in {"origin", "source_type"} and normalized in {
        "test",
        "fixture",
    }:
        reasons.add("protected closure intersects test fixtures")


def _is_live_target(target: dict[str, Any]) -> bool:
    properties = target.get("target_properties") or target.get("properties") or {}
    name_stage = properties.get("name_stage", target.get("target_stage"))
    validation_status = properties.get(
        "validation_status", target.get("validation_status")
    )
    return (
        name_stage
        not in {
            "superseded",
            "exhausted",
        }
        and properties.get("status") not in {"superseded", "deprecated"}
        and validation_status != "quarantined"
    )


def _plan_row(
    row: dict[str, Any],
    manifest_row: dict[str, Any],
    *,
    reason: str | None = None,
) -> dict[str, Any]:
    normalized = _normalized_row(row)
    reasons: list[str] = []
    sources = normalized.get("sources") or []
    if len(sources) != 1:
        reasons.append("exact source is missing or ambiguous")
        return {
            "source_id": manifest_row["source_id"],
            "status": "refused",
            "unresolved": reasons,
        }
    source = sources[0]
    source_type = manifest_row["source_type"]
    backings = (
        source.get("dd_backings")
        if source_type == "dd"
        else source.get("signal_backings")
    )
    other_backings = (
        source.get("signal_backings")
        if source_type == "dd"
        else source.get("dd_backings")
    )
    if len(backings or []) != 1 or other_backings:
        reasons.append("source lacks one exact supported backing authority")
    if source_type != _source_type(manifest_row["source_id"]):
        reasons.append("source authority type drifted")
    if len(normalized.get("prospective_targets") or []) != 1:
        reasons.append("prospective target is missing or ambiguous")
    versions = normalized.get("dd_versions") or []
    if (
        len(versions) != 1
        or (versions[0].get("properties") or {}).get("id") != "4.1.1"
        or (versions[0].get("properties") or {}).get("cocos") != 17
    ):
        reasons.append("global DDVersion authority is not 4.1.1 with COCOS 17")
    protection_reasons: set[str] = set()
    _walk_protection(_protection(normalized), protection_reasons)
    reasons.extend(sorted(protection_reasons))
    current_targets = {item["target_id"] for item in _targets(source)}
    projection_targets = {item["target_id"] for item in _projections(source)}
    scalar_id = (source.get("properties") or {}).get("produced_sn_id")
    target_id = manifest_row["prospective_target_id"]
    already = (
        current_targets == {target_id}
        and projection_targets == {target_id}
        and scalar_id == target_id
    )
    checks = {
        "expected_closure_hash": payload_hash(normalized),
        "expected_source_identity_hash": payload_hash(_identity(source)),
        "expected_lifecycle_hash": payload_hash(_lifecycle(source)),
        "expected_scalar_hash": payload_hash(_scalar(source)),
        "expected_targets_hash": payload_hash(_targets(source)),
        "expected_backing_hash": payload_hash(_backing_identity(source)),
        "expected_projection_hash": payload_hash(_projections(source)),
        "participant_ids": list(_participant_ids(normalized)),
        "participant_relationship_ids": list(_relationship_ids(normalized)),
        "expected_protection_hash": payload_hash(_protection(normalized)),
    }
    immutable_fields = {
        "expected_source_identity_hash",
        "expected_lifecycle_hash",
        "expected_backing_hash",
    }
    reasons.extend(
        f"manifest {field} drifted"
        for field, actual in checks.items()
        if manifest_row[field] != actual and (not already or field in immutable_fields)
    )
    live_ids = {
        str(item["target_id"])
        for item in _targets(source) + _projections(source)
        if item.get("target_id") and _is_live_target(item)
    }
    if not manifest_row["reviewed_override"] and live_ids != {target_id}:
        reasons.append("automatic repair requires one sole live semantic identity")
    if manifest_row["reviewed_override"] and not manifest_row["review_approval"]:
        reasons.append("reviewed override lacks independent approval")
    events = source.get("events") or []
    event_id = _event_id(manifest_row)
    if already:
        matching = [
            event
            for event in events
            if _valid_event(event, manifest_row, event_id, reason=reason)
        ]
        if len(matching) != 1:
            reasons.append("already-current binding lacks one exact immutable event")
        return {
            "source_id": manifest_row["source_id"],
            "status": "refused" if reasons else "already_current",
            "unresolved": sorted(set(reasons)),
            "event": matching[0].get("event_properties")
            if len(matching) == 1
            else None,
        }
    if events:
        reasons.append("semantic reconciliation event exists before repair")
    if reasons:
        return {
            "source_id": manifest_row["source_id"],
            "status": "refused",
            "unresolved": sorted(set(reasons)),
        }
    target = normalized["prospective_targets"][0]
    event = _event_record(manifest_row, event_id, reason=reason)
    return {
        "source_id": manifest_row["source_id"],
        "status": "planned",
        "unresolved": [],
        "event": event,
        "mutation": {
            "source_id": manifest_row["source_id"],
            "source_element_id": source["element_id"],
            "target_id": target_id,
            "target_element_id": target["element_id"],
            "backing_element_id": backings[0]["element_id"],
            "affected_target_ids": sorted(
                current_targets | projection_targets | {target_id}
            ),
        },
    }


def _event_id(manifest_row: dict[str, Any]) -> str:
    identity = {
        "source_id": manifest_row["source_id"],
        "target_id": manifest_row["prospective_target_id"],
        "closure_hash": manifest_row["expected_closure_hash"],
        "protection_hash": manifest_row["expected_protection_hash"],
    }
    return EVENT_PREFIX + payload_hash(identity)


def _event_record(
    manifest_row: dict[str, Any], event_id: str, *, reason: str | None = None
) -> dict[str, Any]:
    record = {
        "id": event_id,
        "operation": "reconcile_semantic_source",
        "source_id": manifest_row["source_id"],
        "target_id": manifest_row["prospective_target_id"],
        "precondition_hash": manifest_row["expected_closure_hash"],
        "protection_hash": manifest_row["expected_protection_hash"],
        "reviewed_override": manifest_row["reviewed_override"],
        "review_approval": manifest_row["review_approval"],
        "reason": reason,
    }
    record["record_hash"] = payload_hash(record)
    return record


def _valid_event(
    entry: dict[str, Any],
    manifest_row: dict[str, Any],
    event_id: str,
    *,
    reason: str | None,
) -> bool:
    event = copy.deepcopy(entry.get("event_properties") or {})
    stored_hash = event.pop("record_hash", None)
    return (
        event.get("id") == event_id
        and event.get("source_id") == manifest_row["source_id"]
        and event.get("target_id") == manifest_row["prospective_target_id"]
        and event.get("reason") == reason
        and stored_hash == payload_hash(event)
    )


def plan_semantic_source_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return deterministic automatic candidates; ambiguous identities are excluded."""
    planned: list[dict[str, Any]] = []
    for row in rows:
        normalized = _normalized_row(row)
        sources = normalized.get("sources") or []
        if len(sources) != 1:
            continue
        source = sources[0]
        source_id = str(normalized.get("source_id") or "")
        if _source_type(source_id) not in SUPPORTED_SOURCE_TYPES:
            continue
        target_items = _targets(source) + _projections(source)
        live_ids = {
            str(item["target_id"])
            for item in target_items
            if item.get("target_id") and _is_live_target(item)
        }
        if len(live_ids) != 1:
            continue
        target_id = next(iter(live_ids))
        manifest_row = build_semantic_source_manifest_row(
            normalized, prospective_target_id=target_id
        )
        plan = _plan_row(normalized, manifest_row)
        if plan["status"] == "planned":
            planned.append(manifest_row)
    return planned


def _read_rows(
    transaction: Any, manifest: SemanticSourceManifest
) -> list[dict[str, Any]]:
    candidates = [
        {
            "source_id": row["source_id"],
            "prospective_target_id": row["prospective_target_id"],
        }
        for row in manifest.rows
    ]
    rows = [
        _normalized_row(dict(row))
        for row in transaction.run(CLOSURE_QUERY, candidates=candidates)
    ]
    rows.sort(key=lambda row: str(row.get("source_id")))
    if [row.get("source_id") for row in rows] != list(manifest.source_ids):
        raise SemanticSourceConflict("closure read omitted the exact source allowlist")
    return rows


def _plan(
    transaction: Any, manifest: SemanticSourceManifest, *, reason: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = _read_rows(transaction, manifest)
    by_source = {str(row["source_id"]): row for row in manifest.rows}
    return rows, [
        _plan_row(row, by_source[str(row["source_id"])], reason=reason) for row in rows
    ]


def _lock(transaction: Any, query: str, element_ids: set[str], message: str) -> None:
    rows = list(transaction.run(query, element_ids=sorted(element_ids)))
    count = int(dict(rows[0]).get("locked") or 0) if rows else 0
    if count != len(element_ids):
        raise SemanticSourceConflict(message)


def _lock_relationships(
    transaction: Any, rows: list[dict[str, Any]], expected_ids: set[str]
) -> None:
    locks = _relationship_lock_items(rows)
    if {item["relationship_element_id"] for item in locks} != expected_ids:
        raise SemanticSourceConflict("relationship lock closure changed")
    result = list(transaction.run(RELATIONSHIP_LOCK_QUERY, locks=locks))
    count = int(dict(result[0]).get("locked") or 0) if result else 0
    if count != len(locks):
        raise SemanticSourceConflict("relationship set changed before locking")


def _receipt(
    manifest: SemanticSourceManifest,
    plans: list[dict[str, Any]],
    *,
    applied: bool,
    query_count: int,
    transaction_count: int,
) -> dict[str, Any]:
    counts = Counter(plan["status"] for plan in plans)
    mode = (
        "refused"
        if counts["refused"]
        else "applied"
        if applied
        else "already_current"
        if counts["already_current"] == len(plans)
        else "planned"
    )
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "schema_version": 1,
        "mode": mode,
        "manifest_path": str(manifest.path),
        "manifest_hash": manifest.manifest_hash,
        "allowlist_hash": manifest.allowlist_hash,
        "counts": {
            "allowlisted": len(plans),
            "planned": counts["planned"],
            "applied": len(plans) if applied else 0,
            "already_current": counts["already_current"],
            "refused": counts["refused"],
        },
        "query_count": query_count,
        "transaction_count": transaction_count,
        "rows": [
            {
                "source_id": plan["source_id"],
                "status": plan["status"],
                "unresolved": plan["unresolved"],
                "event_id": (plan.get("event") or {}).get("id"),
                "event_hash": (plan.get("event") or {}).get("record_hash"),
            }
            for plan in plans
        ],
    }
    receipt["receipt_hash"] = payload_hash(receipt)
    return receipt


class _CountingTransaction:
    def __init__(self, transaction: Any) -> None:
        self.transaction = transaction
        self.query_count = 0

    def run(self, query: str, **params: Any) -> Any:
        self.query_count += 1
        return self.transaction.run(query, **params)

    def commit(self) -> None:
        self.transaction.commit()

    def rollback(self) -> None:
        self.transaction.rollback()


def reconcile_semantic_sources(
    manifest_path: str | Path,
    *,
    reason: str,
    apply: bool = False,
    expected_manifest_hash: str | None = None,
    gc: Any | None = None,
) -> dict[str, Any]:
    """Plan or atomically reconcile one exact semantic-source cohort."""
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("semantic-source reconciliation requires a non-empty reason")
    normalized_hash = (
        expected_manifest_hash.strip().casefold()
        if isinstance(expected_manifest_hash, str)
        else None
    )
    if apply and normalized_hash is None:
        raise ValueError("apply requires an expected manifest SHA-256")
    if (
        normalized_hash is not None
        and re.fullmatch(r"[0-9a-f]{64}", normalized_hash) is None
    ):
        raise ValueError("expected manifest SHA-256 must be exactly 64 hex characters")
    manifest = load_semantic_source_manifest(manifest_path)
    if normalized_hash is not None and not hmac.compare_digest(
        normalized_hash, manifest.manifest_hash
    ):
        raise ValueError("manifest SHA-256 does not match exact raw bytes")
    own = gc is None
    client = GraphClient() if own else gc
    try:
        with client.session() as session:
            counted = _CountingTransaction(session.begin_transaction())
            try:
                before_rows, plans = _plan(counted, manifest, reason=reason.strip())
                statuses = {plan["status"] for plan in plans}
                if (
                    "refused" in statuses
                    or len(statuses) > 1
                    or not apply
                    or statuses == {"already_current"}
                ):
                    counted.rollback()
                    if len(statuses) > 1 and "refused" not in statuses:
                        plans = [
                            {
                                **plan,
                                "status": "refused",
                                "unresolved": [
                                    "mixed pending and already-current cohort"
                                ],
                            }
                            for plan in plans
                        ]
                    return _receipt(
                        manifest,
                        plans,
                        applied=False,
                        query_count=counted.query_count,
                        transaction_count=1,
                    )
                participant_ids = {
                    item for row in manifest.rows for item in row["participant_ids"]
                }
                relationship_ids = {
                    item
                    for row in manifest.rows
                    for item in row["participant_relationship_ids"]
                }
                _lock(
                    counted,
                    PARTICIPANT_LOCK_QUERY,
                    participant_ids,
                    "participant set changed before locking",
                )
                _lock_relationships(counted, before_rows, relationship_ids)
                locked_rows, locked_plans = _plan(
                    counted, manifest, reason=reason.strip()
                )
                if payload_hash(locked_rows) != payload_hash(before_rows) or any(
                    plan["status"] != "planned" for plan in locked_plans
                ):
                    raise SemanticSourceConflict(
                        "semantic source closure changed after locks"
                    )
                event_ids = [_event_id(row) for row in manifest.rows]
                collision_rows = [
                    dict(row)
                    for row in counted.run(EVENT_COLLISION_QUERY, event_ids=event_ids)
                ]
                if any(row.get("matches") for row in collision_rows) or len(
                    collision_rows
                ) != len(event_ids):
                    raise SemanticSourceConflict(
                        "semantic change event identity already exists"
                    )
                items = []
                for plan in locked_plans:
                    mutation = copy.deepcopy(plan["mutation"])
                    mutation["event"] = copy.deepcopy(plan["event"])
                    items.append(mutation)
                result = [dict(row) for row in counted.run(APPLY_QUERY, items=items)]
                if (
                    len(result) != 1
                    or set(result[0].get("source_ids") or [])
                    != set(manifest.source_ids)
                    or set(result[0].get("event_ids") or []) != set(event_ids)
                ):
                    raise SemanticSourceConflict(
                        "semantic source apply cardinality changed"
                    )
                _, after_plans = _plan(counted, manifest, reason=reason.strip())
                if any(plan["status"] != "already_current" for plan in after_plans):
                    raise SemanticSourceConflict("semantic source postflight failed")
                counted.commit()
                return _receipt(
                    manifest,
                    after_plans,
                    applied=True,
                    query_count=counted.query_count,
                    transaction_count=1,
                )
            except Exception:
                counted.rollback()
                raise
    finally:
        if own:
            client.close()
