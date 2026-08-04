"""Exact manifest reconciliation for structural changes near protected sources.

The ordinary identity and source-authority operators intentionally refuse WEST
and persistent fixture closures.  This module does not weaken those guards.  It
admits a protected closure only as an immutable, hash-bound participant of one
exact transaction.  The only writable graph elements are listed in a manifest
row's mutation payload; every other participant is compared before and after.
"""

from __future__ import annotations

import copy
import hashlib
import hmac
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from imas_codex.discovery.base.claims import retry_on_deadlock
from imas_codex.graph.client import GraphClient
from imas_codex.standard_names import edit as identity_fold
from imas_codex.standard_names.grammar_segment_reconciliation import (
    ProtectedSourceSets,
    _read_protected_source_sets,
)
from imas_codex.standard_names.source_authority import (
    lock_participants,
    normalize_manifest_hash_binding,
    payload_hash,
)

PROTECTED_IDENTITY_FOLD = "protected_identity_fold"
RETIRE_STALE_SOURCE_BRANCH = "retire_stale_source_branch"
PROTECTED_STRUCTURAL_ACTIONS = frozenset(
    {PROTECTED_IDENTITY_FOLD, RETIRE_STALE_SOURCE_BRANCH}
)

_MANIFEST_SCHEMA = "imas-codex.protected-structural-reconciliation-manifest"
_RECEIPT_SCHEMA = "imas-codex.protected-structural-reconciliation-receipt"
_CURRENT_DD_VERSION = "4.1.1"
_CATALOG_COCOS = 17
_DOWNSTREAM_LABELS = ("ip_like", "psi_like")
_MINIMUM_AUTHORITY_POLICY_CONFIDENCE = 0.95
_SHA = re.compile(r"[0-9a-f]{64}")


def _batch_snapshot_query(query: str, marker: str) -> str:
    """Lift the proven single-pair fold snapshot into one cohort query."""
    lifted = query.replace(
        "MATCH (old:StandardName {id: $old_id}),",
        "UNWIND $pairs AS requested_pair\n"
        "MATCH (old:StandardName {id: requested_pair.old_id}),",
        1,
    )
    lifted = lifted.replace("$into_id", "requested_pair.target_id")
    return lifted.replace("// ATOMIC_FOLD_SNAPSHOT", f"// {marker}", 1).replace(
        "// ATOMIC_FOLD_POSTFLIGHT", f"// {marker}", 1
    )


PROTECTED_STRUCTURAL_SNAPSHOT_QUERY = _batch_snapshot_query(
    identity_fold._FOLD_SNAPSHOT_QUERY,
    "PROTECTED_STRUCTURAL_SNAPSHOT",
)
PROTECTED_STRUCTURAL_FOLD_POSTFLIGHT_QUERY = _batch_snapshot_query(
    identity_fold._FOLD_POSTFLIGHT_QUERY,
    "PROTECTED_STRUCTURAL_FOLD_POSTFLIGHT",
)

CATALOG_CONTRACT_QUERY = """
// PROTECTED_STRUCTURAL_CATALOG_CONTRACT
MATCH (version:DDVersion {is_current: true})
OPTIONAL MATCH (version)-[:HAS_COCOS|COCOS]->(cocos:COCOS)
RETURN collect(DISTINCT {
  element_id: elementId(version), properties: properties(version)
}) AS versions,
collect(DISTINCT CASE WHEN cocos IS NULL THEN null ELSE {
  element_id: elementId(cocos), properties: properties(cocos)
} END) AS cocos_nodes
"""

RELATIONSHIP_LOCK_QUERY = """
// PROTECTED_STRUCTURAL_RELATIONSHIP_LOCK
MATCH ()-[relationship]-()
WHERE elementId(relationship) IN $element_ids
SET relationship._protected_structural_lock = true
REMOVE relationship._protected_structural_lock
RETURN count(DISTINCT relationship) AS locked
"""

RETIREMENT_STATE_QUERY = """
// PROTECTED_STRUCTURAL_RETIREMENT_STATE
UNWIND $items AS item
OPTIONAL MATCH (old:StandardName {id: item.old_id})
OPTIONAL MATCH (target:StandardName {id: item.target_id})
CALL (item) {
  UNWIND item.source_ids AS source_id
  OPTIONAL MATCH (source:StandardNameSource {id: source_id})
  RETURN collect(CASE WHEN source IS NULL THEN null ELSE {
    element_id: elementId(source), labels: labels(source),
    properties: properties(source),
    relationships: [(source)-[relationship]-(other)
      WHERE type(relationship) IN ['PRODUCED_NAME', 'FROM_DD_PATH', 'FROM_SIGNAL'] | {
      element_id: elementId(relationship), type: type(relationship),
      properties: properties(relationship), other_element_id: elementId(other),
      other_id: other.id, other_labels: labels(other)
    }]
  } END) AS sources
}
CALL (item) {
  UNWIND item.backing_ids AS backing_id
  OPTIONAL MATCH (backing {id: backing_id})
  RETURN collect(CASE WHEN backing IS NULL THEN null ELSE {
    element_id: elementId(backing), labels: labels(backing),
    properties: properties(backing),
    relationships: [(backing)-[relationship]-(other)
      WHERE type(relationship) IN
            ['FROM_DD_PATH', 'FROM_SIGNAL', 'HAS_STANDARD_NAME', 'HAS_UNIT'] | {
      element_id: elementId(relationship), type: type(relationship),
      properties: properties(relationship), other_element_id: elementId(other),
      other_id: other.id, other_labels: labels(other)
    }]
  } END) AS backings
}
CALL (item) {
  UNWIND item.event_ids AS event_id
  OPTIONAL MATCH (event {id: event_id})
  RETURN collect(CASE WHEN event IS NULL THEN null ELSE {
    element_id: elementId(event), labels: labels(event), properties: properties(event)
  } END) AS events
}
RETURN item.row_key AS row_key,
       count(DISTINCT old) AS old_count,
       collect(DISTINCT CASE WHEN target IS NULL THEN null ELSE {
         element_id: elementId(target), labels: labels(target),
         properties: properties(target),
         relationships: [(target)-[relationship]-(other) | {
           element_id: elementId(relationship), type: type(relationship),
           properties: properties(relationship), other_element_id: elementId(other),
           other_id: other.id, other_labels: labels(other)
         }]
       } END) AS targets,
       sources, backings, events
ORDER BY row_key
"""

FOLD_APPLY_QUERY = """
// PROTECTED_STRUCTURAL_FOLD_APPLY
UNWIND $items AS item
MATCH (old:StandardName {id: item.old_id}),
      (target:StandardName {id: item.target_id})
WHERE elementId(old) = item.old_element_id
  AND elementId(target) = item.target_element_id
CREATE (change:StandardNameChange)
SET change = item.event, change.changed_at = datetime(item.event.changed_at)
CREATE (old)-[:HAS_INTERNAL_CHANGE]->(change)
CREATE (target)-[:HAS_INTERNAL_CHANGE]->(change)
CALL (item, target) {
  UNWIND item.sources AS expected
  MATCH (source:StandardNameSource {id: expected.id})
  WHERE elementId(source) = expected.element_id
  OPTIONAL MATCH (source)-[binding:PRODUCED_NAME]->(:StandardName)
  WHERE elementId(binding) IN expected.remove_binding_element_ids
  WITH source, target, collect(binding) AS bindings
  FOREACH (binding IN bindings | DELETE binding)
  CREATE (source)-[:PRODUCED_NAME]->(target)
  SET source.produced_sn_id = target.id
  RETURN count(DISTINCT source) AS moved_sources
}
CALL (item, target) {
  UNWIND item.backings AS expected
  MATCH (backing)
  WHERE elementId(backing) = expected.element_id
  OPTIONAL MATCH (backing)-[projection:HAS_STANDARD_NAME]->(:StandardName)
  WHERE elementId(projection) IN expected.remove_projection_element_ids
  WITH backing, target, expected, collect(projection) AS projections
  FOREACH (projection IN projections | DELETE projection)
  CREATE (backing)-[:HAS_STANDARD_NAME]->(target)
  SET backing.standard_name_id = CASE
    WHEN expected.has_standard_name_id THEN target.id
    ELSE backing.standard_name_id
  END
  RETURN count(DISTINCT backing) AS moved_backings
}
SET old.superseded_from_stage = item.predecessor_stage,
    old.name_stage = 'superseded', old.claim_token = null,
    old.claimed_at = null, old.source_paths = [],
    old.edit_status = CASE WHEN old.edit_status = 'open' THEN 'applied'
                           ELSE old.edit_status END,
    target.source_paths = item.target_paths
MERGE (target)-[:REFINED_FROM]->(old)
RETURN collect({row_key: item.row_key, event_id: change.id,
                sources: moved_sources, backings: moved_backings}) AS results
"""

RETIRE_APPLY_QUERY = """
// PROTECTED_STRUCTURAL_RETIRE_APPLY
UNWIND $items AS item
MATCH (old:StandardName {id: item.old_id}),
      (target:StandardName {id: item.target_id}),
      (source:StandardNameSource {id: item.source_id}),
      (backing {id: item.backing_id})
MATCH (source)-[binding:PRODUCED_NAME]->(old)
MATCH (backing)-[projection:HAS_STANDARD_NAME]->(old)
WHERE elementId(old) = item.old_element_id
  AND elementId(target) = item.target_element_id
  AND elementId(source) = item.source_element_id
  AND elementId(backing) = item.backing_element_id
  AND elementId(binding) = item.binding_element_id
  AND elementId(projection) = item.projection_element_id
CREATE (retirement:StandardNameSourceAuthorityRetirement)
SET retirement = item.retirement_event,
    retirement.retired_at = datetime(item.retirement_event.retired_at)
CREATE (source)-[:HAS_AUTHORITY_RETIREMENT]->(retirement)
CREATE (deletion:StandardNameChange)
SET deletion = item.deletion_event,
    deletion.changed_at = datetime(item.deletion_event.changed_at)
DELETE binding, projection
SET source.status = 'stale', source.produced_sn_id = null,
    source.claimed_at = null, source.claim_token = null,
    source.drain_scope_id = null, source.drain_scope_claimed_at = null,
    source.drain_claim_scope_id = null, source.drain_scope_actionable = null,
    source.skip_reason = 'stale_source_branch',
    source.skip_reason_detail = item.reason
DETACH DELETE old
RETURN collect({row_key: item.row_key, retirement_id: retirement.id,
                deletion_id: deletion.id}) AS results
"""


class ProtectedStructuralConflict(RuntimeError):
    """The exact manifest-bound closure changed or failed a safety rule."""


@dataclass(frozen=True)
class ProtectedStructuralManifest:
    """An exact homogeneous protected structural cohort."""

    path: Path
    manifest_hash: str
    action: str
    protected_set_hash: str
    catalog_contract: dict[str, Any]
    rows: tuple[dict[str, Any], ...]
    row_keys: tuple[str, ...]


def _require_sha(value: Any, field: str) -> str:
    normalized = str(value or "").strip().casefold()
    if _SHA.fullmatch(normalized) is None:
        raise ValueError(f"{field} must be exactly one SHA-256 hex digest")
    return normalized


def _canonical(value: Any) -> Any:
    return identity_fold._fold_normalize(value)


def _snapshot_hash(snapshot: dict[str, Any]) -> str:
    payload = copy.deepcopy(snapshot)
    payload.pop("_cas_signature", None)
    payload.pop("fold_events", None)
    return identity_fold._fold_cas_signature(payload)


def _relationship_ids(snapshot: dict[str, Any]) -> tuple[str, ...]:
    participants = set(identity_fold._fold_participant_ids(snapshot))
    element_ids: set[str] = set()

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            element_id = value.get("element_id")
            if isinstance(element_id, str):
                element_ids.add(element_id)
            for item in value.values():
                visit(item)
        elif isinstance(value, list | tuple):
            for item in value:
                visit(item)

    visit(snapshot)
    return tuple(sorted(element_ids - participants))


def _protected_subclosure(
    snapshot: dict[str, Any], protected: ProtectedSourceSets
) -> dict[str, Any]:
    protected_sources = [
        source
        for source in snapshot.get("sources") or []
        if source.get("id") in protected.present_source_ids
    ]
    protected_source_ids = {source.get("element_id") for source in protected_sources}
    backing_ids = {
        reference.get("backing_element_id")
        for source in protected_sources
        for reference in source.get("backing_refs") or []
    }
    protected_backings = [
        backing
        for backing in snapshot.get("backings") or []
        if backing.get("element_id") in backing_ids
        or any(
            owner.get("source_element_id") in protected_source_ids
            for owner in backing.get("owners") or []
        )
    ]
    protected_elements = {
        snapshot.get("target_element_id"),
        *protected_source_ids,
        *(backing.get("element_id") for backing in protected_backings),
    }
    relationships = [
        relationship
        for relationship in snapshot.get("relationships") or []
        if relationship.get("start_element_id") in protected_elements
        or relationship.get("end_element_id") in protected_elements
    ]
    return _canonical(
        {
            "target": {
                "element_id": snapshot.get("target_element_id"),
                "labels": snapshot.get("target_labels"),
                "properties": snapshot.get("target_properties"),
                "units": snapshot.get("target_units") or [],
            },
            "sources": protected_sources,
            "backings": protected_backings,
            "relationships": relationships,
        }
    )


def _relationship_view(
    *,
    relationship_id: str,
    relationship_type: str,
    properties: dict[str, Any],
    other_element_id: str,
    other_id: Any,
    other_labels: list[str],
) -> dict[str, Any]:
    return {
        "element_id": relationship_id,
        "type": relationship_type,
        "properties": properties,
        "other_element_id": other_element_id,
        "other_id": other_id,
        "other_labels": other_labels,
    }


def _retirement_protected_state(
    snapshot: dict[str, Any], protected: ProtectedSourceSets
) -> dict[str, Any]:
    """Project immutable target/WEST/fixture state in postflight query shape."""
    sources_by_element = {
        source["element_id"]: source for source in snapshot.get("sources") or []
    }
    backings_by_element = {
        backing["element_id"]: backing for backing in snapshot.get("backings") or []
    }
    protected_sources = [
        source
        for source in sources_by_element.values()
        if source.get("id") in protected.present_source_ids
    ]
    protected_source_elements = {source["element_id"] for source in protected_sources}
    protected_backings = [
        backing
        for backing in backings_by_element.values()
        if any(
            owner.get("source_element_id") in protected_source_elements
            for owner in backing.get("owners") or []
        )
    ]

    source_rows = []
    for source in protected_sources:
        relationships = []
        for binding in source.get("bindings") or []:
            relationships.append(
                _relationship_view(
                    relationship_id=binding["element_id"],
                    relationship_type="PRODUCED_NAME",
                    properties=binding.get("properties") or {},
                    other_element_id=binding["target_element_id"],
                    other_id=binding["target_id"],
                    other_labels=binding.get("target_labels") or [],
                )
            )
        for reference in source.get("backing_refs") or []:
            backing = backings_by_element[reference["backing_element_id"]]
            relationships.append(
                _relationship_view(
                    relationship_id=reference["element_id"],
                    relationship_type=reference["type"],
                    properties=reference.get("properties") or {},
                    other_element_id=backing["element_id"],
                    other_id=backing["id"],
                    other_labels=backing.get("labels") or [],
                )
            )
        source_rows.append(
            {
                "element_id": source["element_id"],
                "labels": source.get("labels") or [],
                "properties": source.get("properties") or {},
                "relationships": relationships,
            }
        )

    backing_rows = []
    for backing in protected_backings:
        relationships = []
        for owner in backing.get("owners") or []:
            source = sources_by_element[owner["source_element_id"]]
            relationships.append(
                _relationship_view(
                    relationship_id=owner["relationship_element_id"],
                    relationship_type=owner["relationship_type"],
                    properties=owner.get("relationship_properties") or {},
                    other_element_id=source["element_id"],
                    other_id=source["id"],
                    other_labels=source.get("labels") or [],
                )
            )
        for projection in backing.get("projections") or []:
            relationships.append(
                _relationship_view(
                    relationship_id=projection["element_id"],
                    relationship_type="HAS_STANDARD_NAME",
                    properties=projection.get("properties") or {},
                    other_element_id=projection["target_element_id"],
                    other_id=projection["target_id"],
                    other_labels=projection.get("target_labels") or [],
                )
            )
        for unit in backing.get("units") or []:
            relationships.append(
                _relationship_view(
                    relationship_id=unit["element_id"],
                    relationship_type="HAS_UNIT",
                    properties=unit.get("properties") or {},
                    other_element_id=unit["unit_element_id"],
                    other_id=unit["unit_id"],
                    other_labels=unit.get("unit_labels") or [],
                )
            )
        backing_rows.append(
            {
                "element_id": backing["element_id"],
                "labels": backing.get("labels") or [],
                "properties": backing.get("properties") or {},
                "relationships": relationships,
            }
        )

    target_relationships = []
    target_element_id = snapshot["target_element_id"]
    for relationship in snapshot.get("relationships") or []:
        if target_element_id not in {
            relationship.get("start_element_id"),
            relationship.get("end_element_id"),
        }:
            continue
        if relationship.get("start_element_id") == target_element_id:
            other_prefix = "end"
        else:
            other_prefix = "start"
        target_relationships.append(
            _relationship_view(
                relationship_id=relationship["element_id"],
                relationship_type=relationship["type"],
                properties=relationship.get("properties") or {},
                other_element_id=relationship[f"{other_prefix}_element_id"],
                other_id=relationship.get(f"{other_prefix}_id"),
                other_labels=relationship.get(f"{other_prefix}_labels") or [],
            )
        )
    return _canonical(
        {
            "targets": [
                {
                    "element_id": target_element_id,
                    "labels": snapshot.get("target_labels") or [],
                    "properties": snapshot.get("target_properties") or {},
                    "relationships": target_relationships,
                }
            ],
            "sources": source_rows,
            "backings": backing_rows,
        }
    )


def _retirement_post_protected_state(
    state: dict[str, Any], protected: ProtectedSourceSets
) -> dict[str, Any]:
    protected_sources = [
        source
        for source in state.get("sources") or []
        if source
        and (source.get("properties") or {}).get("id") in protected.present_source_ids
    ]
    protected_source_elements = {
        source.get("element_id") for source in protected_sources
    }
    protected_backings = [
        backing
        for backing in state.get("backings") or []
        if backing
        and any(
            relationship.get("other_element_id") in protected_source_elements
            and relationship.get("type") in {"FROM_DD_PATH", "FROM_SIGNAL"}
            for relationship in backing.get("relationships") or []
        )
    ]
    return _canonical(
        {
            "targets": [item for item in state.get("targets") or [] if item],
            "sources": protected_sources,
            "backings": protected_backings,
        }
    )


def _mutation_payload(
    snapshot: dict[str, Any], action: str, old_id: str
) -> dict[str, Any]:
    sources = identity_fold._fold_source_rows(snapshot, old_id)
    backings = identity_fold._fold_old_backings(snapshot, sources, old_id)
    if action == RETIRE_STALE_SOURCE_BRANCH:
        if len(sources) != 1 or len(backings) != 1:
            return {"source_ids": [], "backing_ids": [], "edge_ids": []}
    edge_ids = {
        binding.get("element_id")
        for source in sources
        for binding in source.get("bindings") or []
        if binding.get("target_id") == old_id
    } | {
        projection.get("element_id")
        for backing in backings
        for projection in backing.get("projections") or []
        if projection.get("target_id") == old_id
    }
    return {
        "source_ids": sorted(str(source["id"]) for source in sources),
        "backing_ids": sorted(str(backing["id"]) for backing in backings),
        "edge_ids": sorted(str(item) for item in edge_ids if item),
    }


def build_manifest_row(
    snapshot: dict[str, Any],
    *,
    action: str,
    protected: ProtectedSourceSets,
    reason: str,
    authority_evidence_sha256: str | None,
    event_timestamp: str,
) -> dict[str, Any]:
    """Bind one audited fold-shaped closure to an exact manifest row."""
    if action not in PROTECTED_STRUCTURAL_ACTIONS:
        raise ValueError(f"unknown protected structural action: {action!r}")
    old_id = str((snapshot.get("old_properties") or {}).get("id") or "")
    target_id = str((snapshot.get("target_properties") or {}).get("id") or "")
    if not old_id or not target_id or not reason.strip():
        raise ValueError("manifest rows require two identities and a reason")
    evidence = (
        _require_sha(authority_evidence_sha256, "authority_evidence_sha256")
        if action == PROTECTED_IDENTITY_FOLD
        else None
    )
    mutation = _mutation_payload(snapshot, action, old_id)
    row_key = payload_hash(
        {
            "action": action,
            "old_id": old_id,
            "target_id": target_id,
            "before_hash": _snapshot_hash(snapshot),
            "mutation": mutation,
            "authority_evidence_sha256": evidence,
        }
    )
    return {
        "row_key": row_key,
        "action": action,
        "old_id": old_id,
        "target_id": target_id,
        "source_ids": mutation["source_ids"],
        "backing_ids": mutation["backing_ids"],
        "expected_before_hash": _snapshot_hash(snapshot),
        "expected_participant_ids_hash": payload_hash(
            tuple(identity_fold._fold_participant_ids(snapshot))
        ),
        "expected_relationship_ids_hash": payload_hash(_relationship_ids(snapshot)),
        "expected_protected_subclosure_hash": payload_hash(
            _protected_subclosure(snapshot, protected)
        ),
        "expected_mutation_hash": payload_hash(mutation),
        "authority_evidence_sha256": evidence,
        "event_timestamp": event_timestamp,
        "reason": reason.strip(),
    }


def build_manifest_payload(
    rows: list[dict[str, Any]],
    *,
    protected_set_hash: str,
    authority_evidence_sha256: str,
    authority_evidence_path: str | Path | None = None,
    authority_verdict: str = "equivalent",
    minimum_authority_confidence: float = _MINIMUM_AUTHORITY_POLICY_CONFIDENCE,
) -> dict[str, Any]:
    """Build the exact JSON payload; callers serialize with sorted keys."""
    _require_sha(protected_set_hash, "protected_set_hash")
    evidence_hash = _require_sha(authority_evidence_sha256, "authority_evidence_sha256")
    return {
        "schema": _MANIFEST_SCHEMA,
        "schema_version": 1,
        "catalog_contract": {
            "dd_version": _CURRENT_DD_VERSION,
            "cocos": _CATALOG_COCOS,
            "downstream_labels": list(_DOWNSTREAM_LABELS),
            "authority_evidence_path": (
                str(Path(authority_evidence_path).expanduser().resolve())
                if authority_evidence_path is not None
                else None
            ),
            "authority_evidence_sha256": evidence_hash,
            "authority_verdict": authority_verdict,
            "minimum_authority_confidence": minimum_authority_confidence,
        },
        "protected_set_hash": protected_set_hash,
        "rows": sorted(copy.deepcopy(rows), key=lambda item: item["row_key"]),
    }


_ROW_FIELDS = {
    "row_key",
    "action",
    "old_id",
    "target_id",
    "source_ids",
    "backing_ids",
    "expected_before_hash",
    "expected_participant_ids_hash",
    "expected_relationship_ids_hash",
    "expected_protected_subclosure_hash",
    "expected_mutation_hash",
    "authority_evidence_sha256",
    "event_timestamp",
    "reason",
}


def load_protected_structural_manifest(
    path: str | Path,
) -> ProtectedStructuralManifest:
    """Load an exact-byte, homogeneous protected structural manifest."""
    manifest_path = Path(path).expanduser().resolve()
    raw = manifest_path.read_bytes()
    try:
        payload = json.loads(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("protected structural manifest is not valid JSON") from exc
    if not isinstance(payload, dict) or set(payload) != {
        "schema",
        "schema_version",
        "catalog_contract",
        "protected_set_hash",
        "rows",
    }:
        raise ValueError("protected structural manifest top-level fields are not exact")
    if payload["schema"] != _MANIFEST_SCHEMA or payload["schema_version"] != 1:
        raise ValueError("protected structural manifest schema is unsupported")
    contract = payload["catalog_contract"]
    if not isinstance(contract, dict) or set(contract) != {
        "dd_version",
        "cocos",
        "downstream_labels",
        "authority_evidence_path",
        "authority_evidence_sha256",
        "authority_verdict",
        "minimum_authority_confidence",
    }:
        raise ValueError("catalog contract fields are not exact")
    if (
        contract["dd_version"] != _CURRENT_DD_VERSION
        or contract["cocos"] != _CATALOG_COCOS
        or tuple(contract["downstream_labels"]) != _DOWNSTREAM_LABELS
    ):
        raise ValueError("manifest does not bind DD 4.1.1, COCOS 17, and labels")
    authority_hash = _require_sha(
        contract["authority_evidence_sha256"], "authority_evidence_sha256"
    )
    authority_verdict = str(contract["authority_verdict"] or "").strip().casefold()
    if not authority_verdict:
        raise ValueError("catalog contract requires an authority verdict")
    try:
        minimum_confidence = float(contract["minimum_authority_confidence"])
    except (TypeError, ValueError) as exc:
        raise ValueError("authority confidence policy must be numeric") from exc
    if (
        isinstance(contract["minimum_authority_confidence"], bool)
        or not 0.0 <= minimum_confidence <= 1.0
        or minimum_confidence < _MINIMUM_AUTHORITY_POLICY_CONFIDENCE
    ):
        raise ValueError("authority confidence policy is below the required floor")
    contract["authority_verdict"] = authority_verdict
    contract["minimum_authority_confidence"] = minimum_confidence
    protected_hash = _require_sha(payload["protected_set_hash"], "protected_set_hash")
    rows = payload["rows"]
    if not isinstance(rows, list) or not rows:
        raise ValueError("protected structural manifest requires rows")
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    actions: set[str] = set()
    mutable_ids: set[str] = set()
    for raw_row in rows:
        if not isinstance(raw_row, dict) or set(raw_row) != _ROW_FIELDS:
            raise ValueError("protected structural manifest row fields are not exact")
        row = copy.deepcopy(raw_row)
        row_key = _require_sha(row["row_key"], "row_key")
        if row_key in seen:
            raise ValueError("protected structural manifest row keys are duplicated")
        seen.add(row_key)
        action = str(row["action"])
        if action not in PROTECTED_STRUCTURAL_ACTIONS:
            raise ValueError(f"unknown protected structural action: {action!r}")
        actions.add(action)
        if not str(row["old_id"]).strip() or not str(row["target_id"]).strip():
            raise ValueError("manifest identities must be non-empty")
        if row["old_id"] == row["target_id"]:
            raise ValueError("manifest identities must be distinct")
        for field in ("source_ids", "backing_ids"):
            values = row[field]
            if not isinstance(values, list) or values != sorted(set(values)):
                raise ValueError(f"{field} must be a sorted unique list")
        overlap = mutable_ids & ({row["old_id"]} | set(row["source_ids"]))
        if overlap:
            raise ValueError(f"mutable participants overlap rows: {sorted(overlap)}")
        mutable_ids.update({row["old_id"], *row["source_ids"]})
        for field in (
            "expected_before_hash",
            "expected_participant_ids_hash",
            "expected_relationship_ids_hash",
            "expected_protected_subclosure_hash",
            "expected_mutation_hash",
        ):
            row[field] = _require_sha(row[field], field)
        if action == PROTECTED_IDENTITY_FOLD:
            row["authority_evidence_sha256"] = _require_sha(
                row["authority_evidence_sha256"], "authority_evidence_sha256"
            )
            if not hmac.compare_digest(
                row["authority_evidence_sha256"], authority_hash
            ):
                raise ValueError(
                    "fold row authority evidence differs from catalog contract"
                )
        elif row["authority_evidence_sha256"] is not None:
            raise ValueError("retirement rows do not carry semantic authority evidence")
        if not str(row["event_timestamp"]).strip() or not str(row["reason"]).strip():
            raise ValueError("manifest rows require event timestamp and reason")
        normalized.append(row)
    if len(actions) != 1:
        raise ValueError("protected structural manifest must be homogeneous")
    action = next(iter(actions))
    evidence_path = contract["authority_evidence_path"]
    if action == PROTECTED_IDENTITY_FOLD:
        if not isinstance(evidence_path, str) or not evidence_path.strip():
            raise ValueError("protected identity fold requires authority evidence path")
    elif evidence_path is not None:
        raise ValueError("retirement manifests do not carry authority evidence paths")
    normalized.sort(key=lambda item: item["row_key"])
    if [row["row_key"] for row in rows] != [row["row_key"] for row in normalized]:
        raise ValueError("protected structural rows must be sorted by row_key")
    return ProtectedStructuralManifest(
        path=manifest_path,
        manifest_hash=hashlib.sha256(raw).hexdigest(),
        action=action,
        protected_set_hash=protected_hash,
        catalog_contract=copy.deepcopy(contract),
        rows=tuple(normalized),
        row_keys=tuple(row["row_key"] for row in normalized),
    )


def _validate_authority_evidence(manifest: ProtectedStructuralManifest) -> None:
    """Validate the exact manifest-owned authority artifact for a protected fold."""
    if manifest.action != PROTECTED_IDENTITY_FOLD:
        return
    contract = manifest.catalog_contract
    evidence_path = Path(contract["authority_evidence_path"]).expanduser().resolve()
    try:
        raw = evidence_path.read_bytes()
    except OSError as exc:
        raise ValueError(
            "protected identity fold authority evidence is unavailable"
        ) from exc
    actual_hash = hashlib.sha256(raw).hexdigest()
    if not hmac.compare_digest(actual_hash, contract["authority_evidence_sha256"]):
        raise ValueError("authority evidence SHA-256 does not match the exact bytes")
    try:
        evidence = json.loads(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("authority evidence is not valid JSON") from exc
    if not isinstance(evidence, dict):
        raise ValueError("authority evidence must be a JSON object")

    verdict = evidence.get("authority_verdict") or {}
    cocos = evidence.get("cocos_contract") or {}
    graph_evidence = evidence.get("graph_evidence") or {}
    raw_evidence = graph_evidence.get("raw_evidence") or {}
    catalogs = raw_evidence.get("catalogs") or []
    current_catalogs = [
        item
        for item in catalogs
        if isinstance(item, dict) and item.get("is_current") is True
    ]
    artifact_verdict = str(
        verdict.get("verdict") or evidence.get("final_disposition") or ""
    ).casefold()
    try:
        confidence = float(verdict.get("confidence"))
    except (TypeError, ValueError) as exc:
        raise ValueError("authority evidence confidence is not numeric") from exc
    catalog_matches = (
        len(current_catalogs) == 1
        and current_catalogs[0].get("id") == contract["dd_version"]
        and current_catalogs[0].get("cocos") == contract["cocos"]
    )
    if (
        verdict.get("semantic_decision_remaining") is not False
        or not 0.0 <= confidence <= 1.0
        or confidence < contract["minimum_authority_confidence"]
        or not artifact_verdict.startswith(contract["authority_verdict"])
        or not catalog_matches
        or cocos.get("catalog_check_passed") is not True
        or cocos.get("catalog_constant") != contract["cocos"]
        or cocos.get("change_made") is not False
    ):
        raise ValueError(
            "authority evidence does not authorize current-DD semantic equivalence"
        )


def _read_snapshots(
    transaction: Any, manifest: ProtectedStructuralManifest
) -> dict[str, dict[str, Any]]:
    pairs = [
        {"old_id": row["old_id"], "target_id": row["target_id"]}
        for row in manifest.rows
    ]
    rows = list(
        transaction.run(
            PROTECTED_STRUCTURAL_SNAPSHOT_QUERY,
            pairs=pairs,
            live_stages=sorted(identity_fold._FOLD_LIVE_STAGES),
        )
    )
    snapshots: dict[str, dict[str, Any]] = {}
    by_pair = {(row["old_id"], row["target_id"]): row for row in manifest.rows}
    for graph_row in rows:
        raw = dict(graph_row)
        snapshot = _canonical(raw)
        snapshot["_cas_signature"] = identity_fold._fold_cas_signature(raw)
        old_id = str((snapshot.get("old_properties") or {}).get("id") or "")
        target_id = str((snapshot.get("target_properties") or {}).get("id") or "")
        manifest_row = by_pair.get((old_id, target_id))
        if manifest_row is None or manifest_row["row_key"] in snapshots:
            raise ProtectedStructuralConflict("snapshot returned an unexpected pair")
        snapshots[manifest_row["row_key"]] = snapshot
    return snapshots


def _read_catalog_contract(transaction: Any) -> dict[str, Any]:
    rows = [dict(row) for row in transaction.run(CATALOG_CONTRACT_QUERY)]
    if len(rows) != 1:
        raise ProtectedStructuralConflict("catalog query did not return one row")
    return _canonical(rows[0])


def _catalog_reasons(catalog: dict[str, Any]) -> list[str]:
    versions = catalog.get("versions") or []
    cocos_nodes = [node for node in catalog.get("cocos_nodes") or [] if node]
    reasons: list[str] = []
    if (
        len(versions) != 1
        or (versions[0].get("properties") or {}).get("id") != _CURRENT_DD_VERSION
    ):
        reasons.append("current DD version is not exactly 4.1.1")
    if (
        len(cocos_nodes) != 1
        or (cocos_nodes[0].get("properties") or {}).get("id") != _CATALOG_COCOS
    ):
        reasons.append("global catalog COCOS is not exactly 17")
    return reasons


def _row_reasons(
    row: dict[str, Any],
    snapshot: dict[str, Any] | None,
    protected: ProtectedSourceSets,
    catalog: dict[str, Any],
) -> list[str]:
    reasons = _catalog_reasons(catalog)
    if snapshot is None:
        return reasons + ["identity pair is missing or ambiguous"]
    if _snapshot_hash(snapshot) != row["expected_before_hash"]:
        reasons.append("manifest expected_before_hash drifted")
    participants = tuple(identity_fold._fold_participant_ids(snapshot))
    relationships = _relationship_ids(snapshot)
    if payload_hash(participants) != row["expected_participant_ids_hash"]:
        reasons.append("manifest expected_participant_ids_hash drifted")
    if payload_hash(relationships) != row["expected_relationship_ids_hash"]:
        reasons.append("manifest expected_relationship_ids_hash drifted")
    if (
        payload_hash(_protected_subclosure(snapshot, protected))
        != row["expected_protected_subclosure_hash"]
    ):
        reasons.append("protected subclosure hash drifted")
    mutation = _mutation_payload(snapshot, row["action"], row["old_id"])
    if payload_hash(mutation) != row["expected_mutation_hash"]:
        reasons.append("manifest expected_mutation_hash drifted")
    if (
        mutation["source_ids"] != row["source_ids"]
        or mutation["backing_ids"] != row["backing_ids"]
    ):
        reasons.append("manifest mutable allowlist drifted")
    claim_fields = {
        "claim_token",
        "claimed_at",
        "drain_scope_id",
        "drain_scope_claimed_at",
        "drain_claim_scope_id",
    }
    for properties in (
        snapshot.get("old_properties") or {},
        snapshot.get("target_properties") or {},
        *((source.get("properties") or {}) for source in snapshot.get("sources") or []),
    ):
        if any(properties.get(field) is not None for field in claim_fields):
            reasons.append("closure contains an active claim")
            break
    if snapshot.get("target_properties", {}).get("name_stage") != "accepted":
        reasons.append("protected target is not accepted")
    if snapshot.get("target_properties", {}).get("validation_status") != "valid":
        reasons.append("protected target is not valid")
    if snapshot.get("target_properties", {}).get("cocos") != _CATALOG_COCOS:
        reasons.append("protected target does not carry COCOS 17")
    labels = {
        (backing.get("properties") or {}).get("cocos_transformation_type")
        for backing in snapshot.get("backings") or []
    }
    if not labels <= {None, "one_like", *_DOWNSTREAM_LABELS}:
        reasons.append("per-path COCOS transformation label is unsupported")
    if row["action"] == PROTECTED_IDENTITY_FOLD:
        guard = identity_fold._fold_guard_reason(
            snapshot, row["old_id"], row["target_id"]
        )
        if guard:
            reasons.append(guard)
        if not row["authority_evidence_sha256"]:
            reasons.append("identity fold lacks authority evidence")
    else:
        sources = identity_fold._fold_source_rows(snapshot, row["old_id"])
        backings = identity_fold._fold_old_backings(snapshot, sources, row["old_id"])
        if len(sources) != 1 or len(backings) != 1:
            reasons.append("retirement requires one exact source and backing")
        elif (sources[0].get("properties") or {}).get("status") != "stale":
            reasons.append("retirement source is not stale")
        if snapshot.get("old_properties", {}).get("name_stage") != "pending":
            reasons.append("retirement identity is not pending")
        if snapshot.get("old_properties", {}).get("validation_status") != "quarantined":
            reasons.append("retirement identity is not quarantined")
        if any(
            {
                relationship.get("start_element_id"),
                relationship.get("end_element_id"),
            }
            == {snapshot.get("old_element_id"), snapshot.get("target_element_id")}
            for relationship in snapshot.get("relationships") or []
        ):
            reasons.append(
                "retirement would alter the accepted target relationship closure"
            )
    return sorted(set(reasons))


def _lock_relationships(transaction: Any, element_ids: set[str]) -> None:
    exact_ids = sorted(element_ids)
    rows = list(transaction.run(RELATIONSHIP_LOCK_QUERY, element_ids=exact_ids))
    locked = int(dict(rows[0]).get("locked") or 0) if rows else 0
    if locked != len(exact_ids):
        raise ProtectedStructuralConflict("relationship participant set changed")


def _event_identity(row: dict[str, Any], kind: str) -> str:
    return payload_hash({"row_key": row["row_key"], "kind": kind})


def _fold_item(row: dict[str, Any], snapshot: dict[str, Any]) -> dict[str, Any]:
    old_sources = identity_fold._fold_source_rows(snapshot, row["old_id"])
    old_backings = identity_fold._fold_old_backings(
        snapshot, old_sources, row["old_id"]
    )
    target_paths = identity_fold._fold_target_paths(
        snapshot, old_sources, old_backings, row["target_id"]
    )
    change_id = "sn-change:protected-fold:" + _event_identity(row, "fold")
    run_id = "sn-protected:" + row["row_key"]
    receipt_text, receipt = identity_fold._fold_receipt(
        snapshot,
        row["old_id"],
        row["target_id"],
        snapshot["old_properties"]["name_stage"],
        old_sources,
        old_backings,
        target_paths,
        change_id=change_id,
        run_id=run_id,
        changed_at=row["event_timestamp"],
    )
    event = {
        "id": change_id,
        "from_name": row["old_id"],
        "to_name": row["target_id"],
        "operation": "fold_identity",
        "reason": receipt_text,
        "origin": "catalog_edit",
        "run_id": run_id,
        "changed_at": row["event_timestamp"],
        "internal": True,
    }
    return {
        "row_key": row["row_key"],
        "old_id": row["old_id"],
        "target_id": row["target_id"],
        "old_element_id": snapshot["old_element_id"],
        "target_element_id": snapshot["target_element_id"],
        "predecessor_stage": snapshot["old_properties"]["name_stage"],
        "target_paths": target_paths,
        "sources": [
            {
                "id": source["id"],
                "element_id": source["element_id"],
                "remove_binding_element_ids": [
                    binding["element_id"]
                    for binding in source.get("bindings") or []
                    if binding.get("target_id") in {row["old_id"], row["target_id"]}
                ],
            }
            for source in old_sources
        ],
        "backings": [
            {
                "id": backing["id"],
                "element_id": backing["element_id"],
                "has_standard_name_id": "standard_name_id" in backing["properties"],
                "remove_projection_element_ids": [
                    projection["element_id"]
                    for projection in backing.get("projections") or []
                    if projection.get("target_id") in {row["old_id"], row["target_id"]}
                ],
            }
            for backing in old_backings
        ],
        "event": event,
        "expected_after": receipt["expected_after"],
    }


def _retirement_item(row: dict[str, Any], snapshot: dict[str, Any]) -> dict[str, Any]:
    sources = identity_fold._fold_source_rows(snapshot, row["old_id"])
    backings = identity_fold._fold_old_backings(snapshot, sources, row["old_id"])
    source = sources[0]
    backing = backings[0]
    binding = next(
        item for item in source["bindings"] if item.get("target_id") == row["old_id"]
    )
    projection = next(
        item
        for item in backing["projections"]
        if item.get("target_id") == row["old_id"]
    )
    run_id = "sn-protected:" + row["row_key"]
    retirement_id = "source-authority-retirement:" + _event_identity(row, "source")
    deletion_id = "sn-change:protected-retirement:" + _event_identity(row, "name")
    return {
        "row_key": row["row_key"],
        "old_id": row["old_id"],
        "target_id": row["target_id"],
        "source_id": source["id"],
        "backing_id": backing["id"],
        "old_element_id": snapshot["old_element_id"],
        "target_element_id": snapshot["target_element_id"],
        "source_element_id": source["element_id"],
        "backing_element_id": backing["element_id"],
        "binding_element_id": binding["element_id"],
        "projection_element_id": projection["element_id"],
        "postflight_source_ids": sorted(
            source_row["id"] for source_row in snapshot.get("sources") or []
        ),
        "postflight_backing_ids": sorted(
            backing_row["id"] for backing_row in snapshot.get("backings") or []
        ),
        "reason": row["reason"],
        "retirement_event": {
            "id": retirement_id,
            "source_id": source["id"],
            "dd_path": backing["id"],
            "removed_target_ids": [row["old_id"]],
            "reason": row["reason"],
            "run_id": run_id,
            "manifest_row_key": row["row_key"],
            "retired_at": row["event_timestamp"],
        },
        "deletion_event": {
            "id": deletion_id,
            "from_name": row["old_id"],
            "to_name": row["old_id"],
            "operation": "reconcile_structural_closure",
            "reason": row["reason"],
            "origin": "structural_reconciliation",
            "run_id": run_id,
            "manifest_row_key": row["row_key"],
            "changed_at": row["event_timestamp"],
            "internal": True,
        },
    }


def _receipt(
    manifest: ProtectedStructuralManifest,
    plans: list[dict[str, Any]],
    *,
    applied: bool,
    query_count: int,
) -> dict[str, Any]:
    receipt = {
        "schema": _RECEIPT_SCHEMA,
        "schema_version": 1,
        "manifest_hash": manifest.manifest_hash,
        "action": manifest.action,
        "mode": "applied" if applied else "dry_run",
        "counts": {
            "allowlisted": len(plans),
            "planned": sum(plan["status"] == "planned" for plan in plans),
            "already_current": sum(
                plan["status"] == "already_current" for plan in plans
            ),
            "refused": sum(plan["status"] == "refused" for plan in plans),
            "applied": sum(plan["status"] == "planned" for plan in plans)
            if applied
            else 0,
        },
        "rows": [
            {
                "row_key": plan["row_key"],
                "status": plan["status"],
                "unresolved": plan["unresolved"],
                "before_hash": plan.get("before_hash"),
                "protected_subclosure_hash": plan.get("protected_subclosure_hash"),
                "expected_event_ids": plan.get("event_ids", []),
            }
            for plan in plans
        ],
        "query_audit": {
            "query_count": query_count,
            "cohort_size_independent": True,
        },
        "safety": {
            "catalog_cocos": _CATALOG_COCOS,
            "current_dd_version": _CURRENT_DD_VERSION,
            "downstream_labels": list(_DOWNSTREAM_LABELS),
            "llm_calls": 0,
        },
    }
    receipt["receipt_hash"] = payload_hash(receipt)
    return receipt


def _plan(
    manifest: ProtectedStructuralManifest,
    snapshots: dict[str, dict[str, Any]],
    protected: ProtectedSourceSets,
    catalog: dict[str, Any],
) -> list[dict[str, Any]]:
    plans = []
    for row in manifest.rows:
        snapshot = snapshots.get(row["row_key"])
        reasons = _row_reasons(row, snapshot, protected, catalog)
        plan = {
            "row_key": row["row_key"],
            "status": "refused" if reasons else "planned",
            "unresolved": reasons,
            "participant_ids": identity_fold._fold_participant_ids(snapshot)
            if snapshot
            else [],
            "relationship_ids": list(_relationship_ids(snapshot)) if snapshot else [],
            "before_hash": _snapshot_hash(snapshot) if snapshot else None,
            "protected_subclosure_hash": payload_hash(
                _protected_subclosure(snapshot, protected)
            )
            if snapshot
            else None,
            "snapshot": snapshot,
            "event_ids": [],
            "retirement_protected_hash": (
                payload_hash(_retirement_protected_state(snapshot, protected))
                if snapshot and row["action"] == RETIRE_STALE_SOURCE_BRANCH
                else None
            ),
        }
        if not reasons and snapshot is not None:
            if row["action"] == PROTECTED_IDENTITY_FOLD:
                item = _fold_item(row, snapshot)
                plan["mutation"] = item
                plan["event_ids"] = [item["event"]["id"]]
            else:
                item = _retirement_item(row, snapshot)
                plan["mutation"] = item
                plan["event_ids"] = [
                    item["retirement_event"]["id"],
                    item["deletion_event"]["id"],
                ]
        plans.append(plan)
    return plans


def _read_preflight(
    transaction: Any, manifest: ProtectedStructuralManifest
) -> tuple[list[dict[str, Any]], ProtectedSourceSets, dict[str, Any]]:
    snapshots = _read_snapshots(transaction, manifest)
    catalog = _read_catalog_contract(transaction)
    protected = _read_protected_source_sets(transaction)
    if not hmac.compare_digest(
        protected.protected_set_hash, manifest.protected_set_hash
    ):
        raise ProtectedStructuralConflict("manifest protected_set_hash drifted")
    plans = _plan(manifest, snapshots, protected, catalog)
    return plans, protected, catalog


@retry_on_deadlock()
def reconcile_protected_structure(
    manifest_path: str | Path,
    *,
    apply: bool = False,
    expected_manifest_hash: str | None = None,
    gc: Any | None = None,
) -> dict[str, Any]:
    """Dry-run or atomically apply one exact protected structural cohort."""
    normalized_hash = normalize_manifest_hash_binding(
        expected_manifest_hash, apply=apply
    )
    manifest = load_protected_structural_manifest(manifest_path)
    if normalized_hash is not None and not hmac.compare_digest(
        normalized_hash, manifest.manifest_hash
    ):
        raise ValueError("manifest SHA-256 does not match the exact parsed bytes")
    _validate_authority_evidence(manifest)
    own = gc is None
    client = GraphClient() if own else gc
    query_count = 0
    try:
        with client.session() as session:
            transaction = session.begin_transaction()
            try:
                plans, _, _ = _read_preflight(transaction, manifest)
                query_count += 3
                if any(plan["status"] == "refused" for plan in plans) or not apply:
                    transaction.rollback()
                    return _receipt(
                        manifest, plans, applied=False, query_count=query_count
                    )
                lock_participants(
                    transaction,
                    {item for plan in plans for item in plan["participant_ids"]},
                    conflict_type=ProtectedStructuralConflict,
                    message="protected structural participant set changed",
                )
                _lock_relationships(
                    transaction,
                    {item for plan in plans for item in plan["relationship_ids"]},
                )
                query_count += 2
                locked, locked_protected, _ = _read_preflight(transaction, manifest)
                query_count += 3
                if any(plan["status"] != "planned" for plan in locked) or [
                    plan["before_hash"] for plan in locked
                ] != [plan["before_hash"] for plan in plans]:
                    raise ProtectedStructuralConflict(
                        "protected structural closure changed after locks"
                    )
                fold_items = [
                    plan["mutation"]
                    for plan in locked
                    if manifest.action == PROTECTED_IDENTITY_FOLD
                ]
                retire_items = [
                    plan["mutation"]
                    for plan in locked
                    if manifest.action == RETIRE_STALE_SOURCE_BRANCH
                ]
                fold_result = list(transaction.run(FOLD_APPLY_QUERY, items=fold_items))
                retire_result = list(
                    transaction.run(RETIRE_APPLY_QUERY, items=retire_items)
                )
                query_count += 2
                expected_rows = {plan["row_key"] for plan in locked}
                actual_rows = {
                    str(item["row_key"])
                    for result in (*fold_result, *retire_result)
                    for item in dict(result).get("results") or []
                }
                if actual_rows != expected_rows:
                    raise ProtectedStructuralConflict(
                        "protected structural mutation cardinality changed"
                    )
                if manifest.action == PROTECTED_IDENTITY_FOLD:
                    post_snapshots = _read_snapshots(transaction, manifest)
                    query_count += 1
                    for plan in locked:
                        post = post_snapshots.get(plan["row_key"])
                        change_id = plan["mutation"]["event"]["id"]
                        if (
                            post is None
                            or identity_fold._fold_verification_state(
                                post, fold_change_id=change_id
                            )
                            != plan["mutation"]["expected_after"]
                        ):
                            raise ProtectedStructuralConflict(
                                "protected fold exact postflight failed"
                            )
                else:
                    state_rows = list(
                        transaction.run(
                            RETIREMENT_STATE_QUERY,
                            items=[
                                {
                                    "row_key": plan["row_key"],
                                    "old_id": row["old_id"],
                                    "target_id": row["target_id"],
                                    "source_ids": plan["mutation"][
                                        "postflight_source_ids"
                                    ],
                                    "backing_ids": plan["mutation"][
                                        "postflight_backing_ids"
                                    ],
                                    "event_ids": plan["event_ids"],
                                }
                                for row, plan in zip(manifest.rows, locked, strict=True)
                            ],
                        )
                    )
                    query_count += 1
                    by_key = {
                        str(dict(item)["row_key"]): dict(item) for item in state_rows
                    }
                    for plan in locked:
                        state = by_key.get(plan["row_key"])
                        if (
                            state is None
                            or int(state.get("old_count") or 0) != 0
                            or len([item for item in state.get("events") or [] if item])
                            != 2
                            or payload_hash(
                                _retirement_post_protected_state(
                                    state, locked_protected
                                )
                            )
                            != plan["retirement_protected_hash"]
                        ):
                            raise ProtectedStructuralConflict(
                                "protected retirement exact postflight failed"
                            )
                post_catalog = _read_catalog_contract(transaction)
                post_protected = _read_protected_source_sets(transaction)
                query_count += 2
                if _catalog_reasons(post_catalog):
                    raise ProtectedStructuralConflict("catalog contract changed")
                if post_protected.protected_set_hash != manifest.protected_set_hash:
                    raise ProtectedStructuralConflict("protected source set changed")
                transaction.commit()
                return _receipt(manifest, locked, applied=True, query_count=query_count)
            except BaseException:
                transaction.rollback()
                raise
    finally:
        if own:
            client.close()
