"""Governed reconciliation of exact grammar compatibility projections.

Grammar segment properties are search-oriented mirrors of a standard name's
canonical identifier.  This module changes only those mirrors.  It binds an
exact allowlist to the current node, relationship, lifecycle, and protection
closure; parses each identifier once through the public ISN contract; and
records one immutable ``StandardNameChange`` for every applied projection.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import uuid
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from imas_codex.discovery.base.claims import retry_on_deadlock
from imas_codex.graph.client import GraphClient
from imas_codex.standard_names.source_authority import (
    normalize_manifest_hash_binding,
    payload_hash,
)
from imas_codex.standard_names.sources_manifest import load_sources_file

_MANIFEST_SCHEMA = "imas-codex.grammar-segment-reconciliation-manifest"
_RECEIPT_SCHEMA = "imas-codex.grammar-segment-reconciliation-receipt"
_SCHEMA_VERSION = 1
_OPERATION = "reconcile_grammar_segment_projection"
_ORIGIN = "governed_grammar_segment_reconciliation"
_EVENT_PREFIX = "sn-change:grammar-segment-reconciliation:"
_RUN_PREFIX = "grammar-segment-reconciliation:"
_FIXTURE_SOURCE_ID_PREFIX = "dd:test_review_entry__"
_FIXTURE_PATH_PREFIX = "test/"
_WEST_MANIFEST = Path(__file__).parent / "manifests" / "west_production_dd_paths.yaml"

_ROW_FIELDS = frozenset(
    {
        "name",
        "evidence_row_hash",
        "expected_before_hash",
        "expected_after_hash",
        "expected_identity_hash",
        "expected_protection_hash",
        "expected_participant_ids_hash",
        "expected_relationship_ids_hash",
        "west_intersection",
        "test_intersection",
    }
)

_CLOSURE_QUERY = """
// GRAMMAR_SEGMENT_RECONCILIATION_CLOSURE
UNWIND $names AS requested_name
OPTIONAL MATCH (name:StandardName {id: requested_name})
WITH requested_name, collect(DISTINCT name) AS matches
RETURN requested_name AS name,
       [candidate IN matches WHERE candidate IS NOT NULL | {
         element_id: elementId(candidate), labels: labels(candidate),
         properties: properties(candidate),
         relationships: [(candidate)-[relationship]-(other) | {
           element_id: elementId(relationship), type: type(relationship),
           direction: CASE WHEN startNode(relationship) = candidate
                           THEN 'out' ELSE 'in' END,
           properties: properties(relationship),
           other_element_id: elementId(other), other_labels: labels(other),
           other_id: other.id, other_properties: properties(other)
         }]
       }] AS matches
ORDER BY name
"""

_PROTECTED_SOURCES_QUERY = """
// GRAMMAR_SEGMENT_RECONCILIATION_PROTECTED_SOURCES
WITH $west_source_ids AS west_source_ids
CALL (west_source_ids) {
  UNWIND west_source_ids AS source_id
  OPTIONAL MATCH (source:StandardNameSource {id: source_id})
  RETURN collect(DISTINCT source.id) AS present_west_source_ids
}
CALL {
  MATCH (source:StandardNameSource)
  WHERE source.id STARTS WITH $fixture_source_id_prefix
  RETURN collect(DISTINCT source.id) AS fixture_source_ids
}
RETURN present_west_source_ids, fixture_source_ids
"""

_PARTICIPANT_LOCK_QUERY = """
// GRAMMAR_SEGMENT_RECONCILIATION_PARTICIPANT_LOCK
UNWIND $names AS requested_name
MATCH (name:StandardName {id: requested_name})
WITH name,
     [name] +
     [(source:StandardNameSource)-[:PRODUCED_NAME]->(name) | source] +
     [(node:IMASNode)-[:HAS_STANDARD_NAME]->(name) | node] +
     [(name)-[:HAS_UNIT]->(unit:Unit) | unit] +
     [(name)-[:HAS_PARENT]->(parent:StandardName) | parent] +
     [(child:StandardName)-[:HAS_PARENT]->(name) | child] +
     [(name)-[:REFINED_FROM]->(predecessor:StandardName) | predecessor] +
     [(successor:StandardName)-[:REFINED_FROM]->(name) | successor]
     AS participants
UNWIND participants AS participant
WITH DISTINCT participant
SET participant._grammar_segment_reconciliation_lock = true
REMOVE participant._grammar_segment_reconciliation_lock
RETURN collect(elementId(participant)) AS locked_element_ids
"""

_PROTECTED_SOURCE_LOCK_QUERY = """
// GRAMMAR_SEGMENT_RECONCILIATION_PROTECTED_SOURCE_LOCK
UNWIND $source_ids AS source_id
MATCH (source:StandardNameSource {id: source_id})
SET source._grammar_segment_reconciliation_lock = true
REMOVE source._grammar_segment_reconciliation_lock
RETURN collect(source.id) AS locked_source_ids
"""

_RELATIONSHIP_LOCK_QUERY = """
// GRAMMAR_SEGMENT_RECONCILIATION_RELATIONSHIP_LOCK
UNWIND $names AS requested_name
MATCH (name:StandardName {id: requested_name})
WITH name,
     [(source:StandardNameSource)-[binding:PRODUCED_NAME]->(name) | binding] +
     [(node:IMASNode)-[projection:HAS_STANDARD_NAME]->(name) | projection] +
     [(name)-[unit_link:HAS_UNIT]->(unit:Unit) | unit_link] +
     [(name)-[parent_link:HAS_PARENT]->(parent:StandardName) | parent_link] +
     [(child:StandardName)-[child_link:HAS_PARENT]->(name) | child_link] +
     [(name)-[prior_link:REFINED_FROM]->(predecessor:StandardName) | prior_link] +
     [(successor:StandardName)-[next_link:REFINED_FROM]->(name) | next_link]
     AS relationships
UNWIND relationships AS relationship
WITH DISTINCT relationship
SET relationship._grammar_segment_reconciliation_lock = true
REMOVE relationship._grammar_segment_reconciliation_lock
RETURN collect(elementId(relationship)) AS locked_element_ids
"""


class GrammarSegmentReconciliationConflict(RuntimeError):
    """The exact manifest-bound graph closure changed during reconciliation."""


@dataclass(frozen=True)
class GrammarSegmentManifest:
    """One exact hash-bound standard-name cohort."""

    path: Path
    manifest_hash: str
    source_manifest_hash: str
    catalog_contract_hash: str
    protected_set_hash: str
    rows: tuple[dict[str, Any], ...]
    names: tuple[str, ...]
    allowlist_hash: str


@dataclass(frozen=True)
class ProtectedSourceSets:
    """The identity sets one reconciliation transaction pins before it writes.

    Persistent fixture identities are immutable and refuse a row outright.
    The facility batch identities are locked bystanders: a transaction proves
    it left them as it found them, but membership alone refuses nothing.
    """

    west_source_ids: frozenset[str]
    fixture_source_ids: frozenset[str]
    present_source_ids: frozenset[str]
    protected_set_hash: str


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_sha(value: Any, field: str) -> str:
    normalized = str(value or "").strip().casefold()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError(f"{field} must be exactly one SHA-256 hex digest")
    return normalized


def _segment_projection(name: str) -> dict[str, Any]:
    """Parse one identifier once and return its runtime compatibility fields."""
    from imas_codex.standard_names.graph_ops import _parse_grammar

    parsed = _parse_grammar(name)
    metadata = {"grammar_parse_version", "validation_diagnostics_json"}
    return {key: value for key, value in parsed.items() if key not in metadata}


def _canonical_relationship(relationship: dict[str, Any]) -> dict[str, Any]:
    return {
        "element_id": relationship.get("element_id"),
        "type": relationship.get("type"),
        "direction": relationship.get("direction"),
        "properties": relationship.get("properties") or {},
        "other_element_id": relationship.get("other_element_id"),
        "other_labels": sorted(relationship.get("other_labels") or []),
        "other_id": relationship.get("other_id"),
        "other_properties": relationship.get("other_properties") or {},
    }


def _is_relevant_relationship(relationship: dict[str, Any]) -> bool:
    relationship_type = relationship.get("type")
    direction = relationship.get("direction")
    return (
        relationship_type in {"PRODUCED_NAME", "HAS_STANDARD_NAME"}
        and direction == "in"
    ) or relationship_type in {"HAS_UNIT", "HAS_PARENT", "REFINED_FROM"}


def _is_own_event_relationship(relationship: dict[str, Any]) -> bool:
    return (
        relationship.get("type") == "HAS_INTERNAL_CHANGE"
        and (relationship.get("other_properties") or {}).get("operation") == _OPERATION
    )


def _west_source_ids() -> frozenset[str]:
    """Load the facility batch source identities from the shipped manifest."""
    return frozenset(f"dd:{path}" for path in load_sources_file(_WEST_MANIFEST))


def _protected_set_hash(
    west_source_ids: frozenset[str], fixture_source_ids: frozenset[str]
) -> str:
    return payload_hash(
        {
            "west_source_ids": tuple(sorted(west_source_ids)),
            "fixture_source_ids": tuple(sorted(fixture_source_ids)),
            "fixture_identity_policy": {
                "source_id_prefix": _FIXTURE_SOURCE_ID_PREFIX,
                "source_path_prefix": _FIXTURE_PATH_PREFIX,
            },
        }
    )


def _read_protected_source_sets(transaction: Any) -> ProtectedSourceSets:
    west_source_ids = _west_source_ids()
    rows = list(
        transaction.run(
            _PROTECTED_SOURCES_QUERY,
            west_source_ids=sorted(west_source_ids),
            fixture_source_id_prefix=_FIXTURE_SOURCE_ID_PREFIX,
        )
    )
    if len(rows) != 1:
        raise GrammarSegmentReconciliationConflict(
            "protected source query did not return exactly one snapshot"
        )
    row = dict(rows[0])
    fixture_source_ids = frozenset(
        str(item) for item in row.get("fixture_source_ids") or []
    )
    present_west_source_ids = frozenset(
        str(item) for item in row.get("present_west_source_ids") or []
    )
    unexpected_west = present_west_source_ids - west_source_ids
    if unexpected_west:
        raise GrammarSegmentReconciliationConflict(
            "protected WEST source query returned identities outside its manifest"
        )
    return ProtectedSourceSets(
        west_source_ids=west_source_ids,
        fixture_source_ids=fixture_source_ids,
        present_source_ids=present_west_source_ids | fixture_source_ids,
        protected_set_hash=_protected_set_hash(west_source_ids, fixture_source_ids),
    )


def _protected_reasons(value: Any, protected: ProtectedSourceSets) -> list[str]:
    """Refuse closures whose identities are immutable.

    Persistent test fixtures are immutable.  Facility batch membership is
    ordinary repairable state and yields no reason.
    """
    test = False

    def visit(item: Any, key: str = "") -> None:
        nonlocal test
        if isinstance(item, dict):
            for child_key, child in item.items():
                visit(child, str(child_key))
        elif isinstance(item, list | tuple):
            for child in item:
                visit(child, key)
        elif isinstance(item, str):
            normalized = item.casefold()
            normalized_key = key.casefold()
            if normalized_key in {"id", "other_id", "source_id"}:
                test = test or normalized.startswith(
                    ("test:", "fixture:", "signals:test:")
                )
                test = test or normalized in protected.fixture_source_ids
                test = test or normalized.startswith(_FIXTURE_SOURCE_ID_PREFIX)
                if normalized_key == "source_id":
                    test = test or normalized.startswith(_FIXTURE_PATH_PREFIX)
            if normalized_key in {"origin", "source_type"}:
                test = test or normalized in {"test", "fixture"}

    visit(value)
    return ["current graph closure intersects test fixtures"] if test else []


def _snapshots(candidate: dict[str, Any], parsed: dict[str, Any]) -> dict[str, Any]:
    properties = candidate.get("properties") or {}
    columns = tuple(parsed)
    before = {column: properties.get(column) for column in columns}
    identity = {
        field: properties.get(field)
        for field in (
            "id",
            "name_stage",
            "docs_stage",
            "status",
            "validation_status",
            "origin",
            "claimed_at",
            "claim_token",
            "drain_scope_id",
            "drain_claim_scope_id",
        )
    }
    relationships = sorted(
        (
            _canonical_relationship(relationship)
            for relationship in candidate.get("relationships") or []
            if _is_relevant_relationship(relationship)
            and not _is_own_event_relationship(relationship)
        ),
        key=lambda item: str(item["element_id"]),
    )
    participant_ids = sorted(
        {
            str(candidate.get("element_id")),
            *(
                str(relationship["other_element_id"])
                for relationship in relationships
                if relationship.get("other_element_id")
            ),
        }
    )
    relationship_ids = sorted(
        str(relationship["element_id"])
        for relationship in relationships
        if relationship.get("element_id")
    )
    protection = {
        "name_element_id": candidate.get("element_id"),
        "labels": sorted(candidate.get("labels") or []),
        "relationships": relationships,
        "preserved_properties": {
            key: value for key, value in properties.items() if key not in columns
        },
    }
    return {
        "before": before,
        "after": parsed,
        "identity": identity,
        "protection": protection,
        "participant_ids": participant_ids,
        "relationship_ids": relationship_ids,
    }


def _read_closure(transaction: Any, names: tuple[str, ...]) -> list[dict[str, Any]]:
    rows = [dict(row) for row in transaction.run(_CLOSURE_QUERY, names=list(names))]
    if [row.get("name") for row in rows] != list(names):
        raise GrammarSegmentReconciliationConflict(
            "graph closure did not return the complete exact allowlist"
        )
    return rows


def _event_payload(
    *,
    name: str,
    reason: str,
    manifest_hash: str,
    before_hash: str,
    after_hash: str,
    identity_hash: str,
    protection_hash: str,
    participant_ids_hash: str,
    relationship_ids_hash: str,
    changed_at: str | None,
) -> dict[str, Any]:
    binding = {
        "operator_reason": reason,
        "manifest_hash": manifest_hash,
        "before_hash": before_hash,
        "after_hash": after_hash,
        "identity_hash": identity_hash,
        "protection_hash": protection_hash,
        "participant_ids_hash": participant_ids_hash,
        "relationship_ids_hash": relationship_ids_hash,
    }
    identity = {"name": name, **binding}
    return {
        "id": _EVENT_PREFIX + payload_hash(identity),
        "from_name": name,
        "to_name": name,
        "operation": _OPERATION,
        "reason": json.dumps(binding, sort_keys=True, separators=(",", ":")),
        "origin": _ORIGIN,
        "run_id": manifest_hash,
        "changed_at": changed_at,
        "internal": True,
    }


def _event_properties(candidate: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        relationship.get("other_properties") or {}
        for relationship in candidate.get("relationships") or []
        if _is_own_event_relationship(relationship)
    ]


def load_grammar_segment_manifest(path: str | Path) -> GrammarSegmentManifest:
    """Load one exact governed manifest without graph access."""
    manifest_path = Path(path).expanduser().resolve()
    raw = manifest_path.read_bytes()
    try:
        payload = json.loads(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"grammar-segment manifest is not valid JSON: {manifest_path}"
        ) from exc
    if not isinstance(payload, dict) or set(payload) != {
        "schema",
        "schema_version",
        "source_manifest_sha256",
        "catalog_contract_hash",
        "protected_set_hash",
        "rows",
    }:
        raise ValueError("grammar-segment manifest fields are not exact")
    if (
        payload.get("schema") != _MANIFEST_SCHEMA
        or payload.get("schema_version") != _SCHEMA_VERSION
    ):
        raise ValueError("grammar-segment manifest schema is unsupported")
    source_hash = _require_sha(
        payload.get("source_manifest_sha256"), "source_manifest_sha256"
    )
    catalog_contract_hash = _require_sha(
        payload.get("catalog_contract_hash"), "catalog_contract_hash"
    )
    protected_set_hash = _require_sha(
        payload.get("protected_set_hash"), "protected_set_hash"
    )
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("grammar-segment manifest requires a non-empty rows array")
    normalized: list[dict[str, Any]] = []
    names: list[str] = []
    for row in rows:
        if not isinstance(row, dict) or set(row) != _ROW_FIELDS:
            raise ValueError("grammar-segment manifest row fields are not exact")
        name = str(row.get("name") or "")
        if (
            not name
            or name != name.strip()
            or any(character.isspace() for character in name)
        ):
            raise ValueError(f"invalid exact standard name: {name!r}")
        if name in names:
            raise ValueError(f"duplicate standard name in manifest: {name}")
        for field in sorted(
            _ROW_FIELDS - {"name", "west_intersection", "test_intersection"}
        ):
            _require_sha(row.get(field), field)
        if row["test_intersection"] != 0:
            raise ValueError("test intersection must be exactly zero")
        names.append(name)
        normalized.append(dict(row))
    if names != sorted(names):
        raise ValueError("grammar-segment manifest rows must be sorted by name")
    return GrammarSegmentManifest(
        path=manifest_path,
        manifest_hash=_sha_bytes(raw),
        source_manifest_hash=source_hash,
        catalog_contract_hash=catalog_contract_hash,
        protected_set_hash=protected_set_hash,
        rows=tuple(normalized),
        names=tuple(names),
        allowlist_hash=payload_hash(tuple(names)),
    )


def _plan_rows(
    rows: list[dict[str, Any]],
    manifest: GrammarSegmentManifest,
    protected: ProtectedSourceSets,
    *,
    reason: str,
    changed_at: str | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    manifest_rows = {row["name"]: row for row in manifest.rows}
    plans: list[dict[str, Any]] = []
    refusals: list[dict[str, Any]] = []
    protected_set_drift = protected.protected_set_hash != manifest.protected_set_hash
    for row in rows:
        name = str(row["name"])
        manifest_row = manifest_rows[name]
        matches = row.get("matches") or []
        reasons: list[str] = []
        if len(matches) != 1:
            reasons.append("standard name is missing or ambiguous")
            refusals.append({"name": name, "reasons": reasons})
            continue
        candidate = matches[0]
        properties = candidate.get("properties") or {}
        if properties.get("name_stage") in {"superseded", "exhausted"}:
            reasons.append("standard name is not live")
        if protected_set_drift:
            reasons.append("manifest protected_set_hash drifted")
        parsed = _segment_projection(name)
        if not parsed.get("physical_base"):
            reasons.append("strict public ISN parser rejected the canonical name")
        snapshots = _snapshots(candidate, parsed)
        hashes = {
            "before_hash": payload_hash(snapshots["before"]),
            "after_hash": payload_hash(snapshots["after"]),
            "identity_hash": payload_hash(snapshots["identity"]),
            "protection_hash": payload_hash(snapshots["protection"]),
            "participant_ids_hash": payload_hash(tuple(snapshots["participant_ids"])),
            "relationship_ids_hash": payload_hash(tuple(snapshots["relationship_ids"])),
        }
        current = snapshots["before"] == snapshots["after"]
        for key, expected_key in (
            ("after_hash", "expected_after_hash"),
            ("identity_hash", "expected_identity_hash"),
            ("protection_hash", "expected_protection_hash"),
            ("participant_ids_hash", "expected_participant_ids_hash"),
            ("relationship_ids_hash", "expected_relationship_ids_hash"),
        ):
            if hashes[key] != manifest_row[expected_key]:
                reasons.append(f"manifest {expected_key} drifted")
        if (
            not current
            and hashes["before_hash"] != manifest_row["expected_before_hash"]
        ):
            reasons.append("manifest expected_before_hash drifted")
        reasons.extend(_protected_reasons(snapshots["protection"], protected))
        event = _event_payload(
            name=name,
            reason=reason,
            manifest_hash=manifest.manifest_hash,
            changed_at=changed_at,
            before_hash=manifest_row["expected_before_hash"],
            after_hash=manifest_row["expected_after_hash"],
            identity_hash=manifest_row["expected_identity_hash"],
            protection_hash=manifest_row["expected_protection_hash"],
            participant_ids_hash=manifest_row["expected_participant_ids_hash"],
            relationship_ids_hash=manifest_row["expected_relationship_ids_hash"],
        )
        events = _event_properties(candidate)
        exact_events = [
            item
            for item in events
            if all(
                item.get(key) == value
                for key, value in event.items()
                if key != "changed_at"
            )
        ]
        if events and (len(events) != 1 or len(exact_events) != 1):
            reasons.append("grammar reconciliation event is duplicate or tampered")
        elif current and len(exact_events) != 1:
            reasons.append("current grammar projection lacks its exact governed event")
        elif not current and events:
            reasons.append(
                "stale grammar projection already has a reconciliation event"
            )
        if reasons:
            refusals.append({"name": name, "reasons": sorted(set(reasons))})
            continue
        status = "already_current" if current else "planned"
        plans.append(
            {
                "name": name,
                "status": status,
                **hashes,
                "precondition_hash": payload_hash(
                    {"manifest_hash": manifest.manifest_hash, "candidate": candidate}
                ),
                "participant_ids": snapshots["participant_ids"],
                "relationship_ids": snapshots["relationship_ids"],
                "after": snapshots["after"],
                "event": event,
                "element_id": candidate["element_id"],
            }
        )
    return plans, refusals


def _read_plan(
    transaction: Any,
    manifest: GrammarSegmentManifest,
    *,
    reason: str,
    changed_at: str | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], ProtectedSourceSets]:
    protected = _read_protected_source_sets(transaction)
    plans, refusals = _plan_rows(
        _read_closure(transaction, manifest.names),
        manifest,
        protected,
        reason=reason,
        changed_at=changed_at,
    )
    return plans, refusals, protected


def _lock_participants(
    transaction: Any, names: tuple[str, ...], expected_element_ids: set[str]
) -> None:
    rows = list(transaction.run(_PARTICIPANT_LOCK_QUERY, names=list(names)))
    locked = {
        str(item) for row in rows for item in dict(row).get("locked_element_ids") or []
    }
    if locked != expected_element_ids:
        raise GrammarSegmentReconciliationConflict(
            "grammar reconciliation participant set changed before locking"
        )


def _lock_protected_sources(
    transaction: Any, expected_source_ids: frozenset[str]
) -> None:
    rows = list(
        transaction.run(
            _PROTECTED_SOURCE_LOCK_QUERY,
            source_ids=sorted(expected_source_ids),
        )
    )
    locked = {
        str(item) for row in rows for item in dict(row).get("locked_source_ids") or []
    }
    if locked != set(expected_source_ids):
        raise GrammarSegmentReconciliationConflict(
            "protected source set changed before locking"
        )


def _lock_relationships(
    transaction: Any, names: tuple[str, ...], expected_element_ids: set[str]
) -> None:
    rows = list(transaction.run(_RELATIONSHIP_LOCK_QUERY, names=list(names)))
    locked = {
        str(item) for row in rows for item in dict(row).get("locked_element_ids") or []
    }
    if locked != expected_element_ids:
        raise GrammarSegmentReconciliationConflict(
            "grammar reconciliation relationship set changed before locking"
        )


def _apply_query(columns: tuple[str, ...]) -> str:
    assignments = ", ".join(
        f"name.{column} = item.after.{column}" for column in columns
    )
    return f"""
// GRAMMAR_SEGMENT_RECONCILIATION_APPLY
UNWIND $items AS item
MATCH (name:StandardName {{id: item.name}})
WHERE elementId(name) = item.element_id
SET {assignments}
CREATE (event:StandardNameChange {{id: item.event.id}})
SET event.from_name = item.event.from_name,
    event.to_name = item.event.to_name,
    event.operation = item.event.operation,
    event.reason = item.event.reason,
    event.origin = item.event.origin,
    event.run_id = item.event.run_id,
    event.changed_at = datetime(item.event.changed_at),
    event.internal = item.event.internal
CREATE (name)-[:HAS_INTERNAL_CHANGE]->(event)
RETURN collect(name.id) AS names, collect(event.id) AS event_ids
"""


def _receipt(
    manifest: GrammarSegmentManifest,
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
    receipt: dict[str, Any] = {
        "schema": _RECEIPT_SCHEMA,
        "schema_version": _SCHEMA_VERSION,
        "mode": mode,
        "operation": _OPERATION,
        "manifest_path": str(manifest.path),
        "manifest_hash": manifest.manifest_hash,
        "source_manifest_hash": manifest.source_manifest_hash,
        "catalog_contract_hash": manifest.catalog_contract_hash,
        "protected_set_hash": manifest.protected_set_hash,
        "allowlist_hash": manifest.allowlist_hash,
        "run_id": run_id if apply else None,
        "counts": {
            "allowlisted": len(manifest.names),
            "planned": counts["planned"],
            "already_current": counts["already_current"],
            "applied": counts["planned"] if mode == "applied" else 0,
            "refused": len(refusals),
        },
        "rows": [
            {
                key: plan[key]
                for key in (
                    "name",
                    "status",
                    "before_hash",
                    "after_hash",
                    "identity_hash",
                    "protection_hash",
                    "participant_ids_hash",
                    "relationship_ids_hash",
                    "precondition_hash",
                )
            }
            | {"event_id": plan["event"]["id"]}
            for plan in sorted(plans, key=lambda item: item["name"])
        ],
        "refusals": sorted(refusals, key=lambda item: item["name"]),
        "query_audit": {
            "planning_reads": 2,
            "cohort_size_independent": True,
        },
        "safety": {
            "graph_writes": 1 if mode == "applied" else 0,
            "llm_calls": 0,
            "llm_cost_usd": 0.0,
            "parser_inputs": ["canonical_name"],
            "preserved_catalog_cocos": 17,
            "preserved_downstream_labels": ["psi_like", "ip_like"],
        },
    }
    receipt["receipt_hash"] = payload_hash(receipt)
    return receipt


def build_grammar_segment_manifest(
    source_manifest_path: str | Path,
    output_path: str | Path,
    *,
    expected_source_manifest_hash: str,
    gc: Any | None = None,
) -> dict[str, Any]:
    """Build a governed manifest from exact stamp-only adjudication rows."""
    source_path = Path(source_manifest_path).expanduser().resolve()
    source_raw = source_path.read_bytes()
    source_hash = _sha_bytes(source_raw)
    if not hmac.compare_digest(
        source_hash,
        _require_sha(expected_source_manifest_hash, "expected_source_manifest_hash"),
    ):
        raise ValueError("source manifest SHA-256 does not match the exact bytes")
    source = json.loads(source_raw)
    contract = source.get("catalog_contract") or {}
    versions = contract.get("dd_versions") or []
    if (
        contract.get("catalog_cocos") != 17
        or contract.get("psi_like_ip_like_retained") is not True
        or not {"psi_like", "ip_like"}.issubset(contract.get("downstream_labels") or [])
        or not any(
            version.get("id") == "4.1.1"
            and version.get("cocos") == 17
            and version.get("is_current") is True
            for version in versions
        )
    ):
        raise ValueError(
            "source manifest does not bind the DD 4.1.1, COCOS 17 catalog contract"
        )
    evidence_rows = [
        row
        for row in source.get("rows") or []
        if row.get("action_category") == "safe_stamp_only_reconcile"
    ]
    names = tuple(sorted(str(row["id"]) for row in evidence_rows))
    if not names:
        raise ValueError("source manifest contains no exact stamp-only rows")
    evidence_by_name = {str(row["id"]): row for row in evidence_rows}
    own = gc is None
    client = GraphClient() if own else gc
    try:
        with client.session() as session:
            transaction = session.begin_transaction()
            try:
                protected = _read_protected_source_sets(transaction)
                rows = _read_closure(transaction, names)
            finally:
                transaction.rollback()
    finally:
        if own:
            client.close()
    manifest_rows: list[dict[str, Any]] = []
    for row in rows:
        name = str(row["name"])
        matches = row.get("matches") or []
        if len(matches) != 1:
            raise GrammarSegmentReconciliationConflict(
                f"standard name {name!r} is missing or ambiguous"
            )
        parsed = _segment_projection(name)
        snapshots = _snapshots(matches[0], parsed)
        if not parsed.get("physical_base") or snapshots["before"] == snapshots["after"]:
            raise GrammarSegmentReconciliationConflict(
                f"standard name {name!r} is not one live parser-valid stale projection"
            )
        protection_reasons = _protected_reasons(snapshots["protection"], protected)
        if protection_reasons:
            raise GrammarSegmentReconciliationConflict("; ".join(protection_reasons))
        evidence = evidence_by_name[name]
        if evidence.get("stored_segments") != snapshots["before"]:
            raise GrammarSegmentReconciliationConflict(
                f"source evidence before projection drifted for {name!r}"
            )
        if (evidence.get("strict_parse") or {}).get("segments") != snapshots["after"]:
            raise GrammarSegmentReconciliationConflict(
                f"source evidence parser projection drifted for {name!r}"
            )
        manifest_rows.append(
            {
                "name": name,
                "evidence_row_hash": payload_hash(evidence_by_name[name]),
                "expected_before_hash": payload_hash(snapshots["before"]),
                "expected_after_hash": payload_hash(snapshots["after"]),
                "expected_identity_hash": payload_hash(snapshots["identity"]),
                "expected_protection_hash": payload_hash(snapshots["protection"]),
                "expected_participant_ids_hash": payload_hash(
                    tuple(snapshots["participant_ids"])
                ),
                "expected_relationship_ids_hash": payload_hash(
                    tuple(snapshots["relationship_ids"])
                ),
                "west_intersection": 0,
                "test_intersection": 0,
            }
        )
    payload = {
        "schema": _MANIFEST_SCHEMA,
        "schema_version": _SCHEMA_VERSION,
        "source_manifest_sha256": source_hash,
        "catalog_contract_hash": payload_hash(contract),
        "protected_set_hash": protected.protected_set_hash,
        "rows": manifest_rows,
    }
    rendered = json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    target = Path(output_path).expanduser().resolve()
    target.write_text(rendered)
    return {
        "manifest_path": str(target),
        "manifest_hash": _sha_bytes(rendered.encode()),
        "source_manifest_hash": source_hash,
        "catalog_contract_hash": payload_hash(contract),
        "protected_set_hash": protected.protected_set_hash,
        "allowlist_hash": payload_hash(names),
        "query_count": 2,
        "rows": len(manifest_rows),
    }


def plan_grammar_segment_reconciliation(
    manifest_path: str | Path,
    *,
    reason: str,
    expected_manifest_hash: str | None = None,
    gc: Any | None = None,
) -> dict[str, Any]:
    """Return a zero-write plan for one exact grammar projection cohort."""
    return reconcile_grammar_segments(
        manifest_path,
        reason=reason,
        apply=False,
        expected_manifest_hash=expected_manifest_hash,
        gc=gc,
    )


@retry_on_deadlock()
def reconcile_grammar_segments(
    manifest_path: str | Path,
    *,
    reason: str,
    apply: bool = False,
    expected_manifest_hash: str | None = None,
    run_id: str | None = None,
    gc: Any | None = None,
) -> dict[str, Any]:
    """Plan or atomically apply an exact grammar projection reconciliation."""
    normalized_reason = (reason or "").strip()
    if not normalized_reason:
        raise ValueError("a grammar-segment reconciliation reason is required")
    normalized_hash = normalize_manifest_hash_binding(
        expected_manifest_hash, apply=apply
    )
    manifest = load_grammar_segment_manifest(manifest_path)
    if normalized_hash is not None and not hmac.compare_digest(
        manifest.manifest_hash, normalized_hash
    ):
        raise ValueError("manifest SHA-256 does not match the exact parsed bytes")
    invocation_run_id = run_id or (_RUN_PREFIX + str(uuid.uuid4()) if apply else None)
    changed_at = datetime.now(UTC).isoformat() if apply else None
    own = gc is None
    client = GraphClient() if own else gc
    try:
        with client.session() as session:
            transaction = session.begin_transaction()
            try:
                plans, refusals, protected = _read_plan(
                    transaction,
                    manifest,
                    reason=normalized_reason,
                    changed_at=changed_at,
                )
                if refusals:
                    transaction.rollback()
                    return _receipt(
                        manifest, plans, refusals, apply=apply, run_id=invocation_run_id
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
                                "name": "<allowlist>",
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
                        manifest, plans, [], apply=apply, run_id=invocation_run_id
                    )
                _lock_participants(
                    transaction,
                    manifest.names,
                    {
                        participant
                        for plan in pending
                        for participant in plan["participant_ids"]
                    },
                )
                _lock_protected_sources(transaction, protected.present_source_ids)
                _lock_relationships(
                    transaction,
                    manifest.names,
                    {
                        relationship
                        for plan in pending
                        for relationship in plan["relationship_ids"]
                    },
                )
                locked_plans, locked_refusals, locked_protected = _read_plan(
                    transaction,
                    manifest,
                    reason=normalized_reason,
                    changed_at=changed_at,
                )
                if (
                    locked_protected != protected
                    or locked_refusals
                    or [plan["precondition_hash"] for plan in locked_plans]
                    != [plan["precondition_hash"] for plan in plans]
                ):
                    raise GrammarSegmentReconciliationConflict(
                        "name, event, lifecycle, protection, or relationship state changed after locks"
                    )
                mutation_rows = list(
                    transaction.run(
                        _apply_query(tuple(pending[0]["after"])),
                        items=[
                            {
                                "name": plan["name"],
                                "element_id": plan["element_id"],
                                "after": plan["after"],
                                "event": plan["event"],
                            }
                            for plan in pending
                        ],
                    )
                )
                if len(mutation_rows) != 1:
                    raise GrammarSegmentReconciliationConflict(
                        "grammar reconciliation mutation cardinality changed"
                    )
                mutation = dict(mutation_rows[0])
                if set(mutation.get("names") or []) != {
                    plan["name"] for plan in pending
                } or set(mutation.get("event_ids") or []) != {
                    plan["event"]["id"] for plan in pending
                }:
                    raise GrammarSegmentReconciliationConflict(
                        "grammar reconciliation mutation returned an incomplete cohort"
                    )
                post_plans, post_refusals, post_protected = _read_plan(
                    transaction,
                    manifest,
                    reason=normalized_reason,
                    changed_at=changed_at,
                )
                if (
                    post_protected != protected
                    or post_refusals
                    or any(plan["status"] != "already_current" for plan in post_plans)
                ):
                    raise GrammarSegmentReconciliationConflict(
                        "postflight projection or immutable event proof did not hold"
                    )
                transaction.commit()
                return _receipt(
                    manifest, pending, [], apply=True, run_id=invocation_run_id
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
