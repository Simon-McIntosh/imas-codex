"""Manifest-bound reconciliation of incomplete structural parent closures."""

from __future__ import annotations

import copy
import hashlib
import hmac
import json
import re
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from imas_codex.discovery.base.claims import retry_on_deadlock
from imas_codex.graph.client import GraphClient
from imas_codex.standard_names.graph_ops import (
    _materialize_derived_parent_rows_batched,
    reconcile_orphan_parent_sources_batched,
)
from imas_codex.standard_names.parents import is_admissible_parent_name
from imas_codex.standard_names.provenance_lifecycle import (
    deletion_change_params,
)
from imas_codex.standard_names.source_authority import (
    lock_participants,
    normalize_manifest_hash_binding,
    payload_hash,
)

EXCLUDE_NULL_SCAFFOLD = "exclude_null_scaffold"
MATERIALIZE_ADMISSIBLE_PARENT = "materialize_admissible_parent"
SEED_ACCEPTED_PARENT_SOURCE = "seed_accepted_parent_source"
RETIRE_UNREACHABLE_CHAIN = "retire_unreachable_chain"
REFUSE_MISSING_UNIT_AUTHORITY = "refuse_missing_unit_authority"
REFUSE_UNACCEPTED_CHILD_AUTHORITY = "refuse_unaccepted_child_authority"

STRUCTURAL_ACTIONS = frozenset(
    {
        EXCLUDE_NULL_SCAFFOLD,
        MATERIALIZE_ADMISSIBLE_PARENT,
        SEED_ACCEPTED_PARENT_SOURCE,
        RETIRE_UNREACHABLE_CHAIN,
        REFUSE_MISSING_UNIT_AUTHORITY,
        REFUSE_UNACCEPTED_CHILD_AUTHORITY,
    }
)
_MUTATING_ACTIONS = frozenset(
    {
        EXCLUDE_NULL_SCAFFOLD,
        MATERIALIZE_ADMISSIBLE_PARENT,
        SEED_ACCEPTED_PARENT_SOURCE,
        RETIRE_UNREACHABLE_CHAIN,
    }
)
_TERMINAL_STAGES = frozenset({"superseded", "exhausted", "contested"})
_MANIFEST_SCHEMA = "imas-codex.structural-closure-reconciliation-manifest"
_RECEIPT_SCHEMA = "imas-codex.structural-closure-reconciliation-receipt"
_EVENT_DIAGNOSTIC_SCHEMA = "imas-codex.structural-event-roundtrip-diagnostic"
_MAX_DESCENDANT_DEPTH = 12
_SAFE_EVENT_LITERAL_FIELDS = frozenset(
    {
        "action",
        "changed_at",
        "from_name",
        "id",
        "internal",
        "operation",
        "origin",
        "root_id",
        "target_id",
        "to_name",
    }
)

STRUCTURAL_CLOSURE_QUERY = f"""
// STRUCTURAL_CLOSURE_RECONCILIATION_SNAPSHOT
UNWIND $root_ids AS requested_root_id
OPTIONAL MATCH (root:StandardName {{id: requested_root_id}})
WITH requested_root_id, collect(DISTINCT root) AS root_nodes
CALL (requested_root_id, root_nodes) {{
  UNWIND CASE WHEN size(root_nodes) = 1 THEN root_nodes ELSE [null] END AS root
  OPTIONAL MATCH path=(descendant:StandardName)-[:HAS_PARENT*0..{_MAX_DESCENDANT_DEPTH}]->(root)
  WITH root, collect(DISTINCT descendant) AS names,
       collect(relationships(path)) AS relationship_lists
  RETURN [name IN names WHERE name IS NOT NULL | {{
    element_id: elementId(name), labels: labels(name), properties: properties(name),
    units: [(name)-[unit_link:HAS_UNIT]->(unit:Unit) | {{
      relationship_element_id: elementId(unit_link),
      relationship_properties: properties(unit_link),
      element_id: elementId(unit), labels: labels(unit), properties: properties(unit)
    }}],
    sources: [(source:StandardNameSource)-[binding:PRODUCED_NAME]->(name) | {{
      relationship_element_id: elementId(binding),
      relationship_properties: properties(binding),
      element_id: elementId(source), labels: labels(source), properties: properties(source)
    }}],
    dd_sources: [(dd:IMASNode)-[projection:HAS_STANDARD_NAME]->(name) | {{
      relationship_element_id: elementId(projection),
      relationship_properties: properties(projection),
      element_id: elementId(dd), labels: labels(dd), properties: properties(dd)
    }}]
  }}] AS names,
  [relationship IN reduce(all_relationships = [], batch IN relationship_lists |
                           all_relationships + batch)
    WHERE relationship IS NOT NULL | {{
      element_id: elementId(relationship), type: type(relationship),
      properties: properties(relationship),
      start_element_id: elementId(startNode(relationship)),
      start_id: startNode(relationship).id,
      end_element_id: elementId(endNode(relationship)),
      end_id: endNode(relationship).id
    }}] AS parent_edges,
  CASE WHEN root IS NULL THEN false ELSE EXISTS {{
    MATCH (:StandardName)-[:HAS_PARENT*{_MAX_DESCENDANT_DEPTH + 1}]->(root)
  }} END AS depth_truncated
}}
RETURN requested_root_id AS root_id,
       [root IN root_nodes | {{
         element_id: elementId(root), labels: labels(root), properties: properties(root)
       }}] AS roots,
       names, parent_edges, depth_truncated
ORDER BY root_id
"""

_RELATIONSHIP_LOCK_QUERY = f"""
// STRUCTURAL_CLOSURE_RELATIONSHIP_LOCK
UNWIND $root_ids AS root_id
MATCH (root:StandardName {{id: root_id}})
OPTIONAL MATCH path=(descendant:StandardName)
                    -[:HAS_PARENT*0..{_MAX_DESCENDANT_DEPTH}]->(root)
WITH collect(DISTINCT descendant) AS names,
     collect(relationships(path)) AS parent_relationship_lists
CALL (names) {{
  UNWIND names AS name
  OPTIONAL MATCH (name)-[relationship:HAS_UNIT|PRODUCED_NAME|HAS_STANDARD_NAME]-()
  RETURN collect(DISTINCT relationship) AS attached_relationships
}}
WITH attached_relationships +
     reduce(parent_relationships = [], batch IN parent_relationship_lists |
            parent_relationships + batch) AS relationships
UNWIND relationships AS relationship
WITH DISTINCT relationship
WHERE relationship IS NOT NULL AND elementId(relationship) IN $element_ids
SET relationship._structural_closure_lock = true
REMOVE relationship._structural_closure_lock
RETURN count(relationship) AS locked
"""

_DELETE_QUERY = """
// STRUCTURAL_CLOSURE_LEDGERED_RETIREMENT
UNWIND $items AS item
MATCH (name:StandardName {id: item.id})
WHERE NOT EXISTS { MATCH (:StandardNameSource)-[:PRODUCED_NAME]->(name) }
OPTIONAL MATCH (existing:StandardNameChange {id: item.event.id})
WITH name, item, collect(existing) AS existing_events
WHERE size(existing_events) = 0
CREATE (change:StandardNameChange)
SET change = item.event
WITH name, item, change
DETACH DELETE name
RETURN collect(item.id) AS deleted_ids,
       collect(change.id) AS event_ids
"""

_EVENT_READ_QUERY = """
// STRUCTURAL_CLOSURE_EVENT_POSTFLIGHT
UNWIND $event_ids AS event_id
OPTIONAL MATCH (event:StandardNameChange {id: event_id})
WITH event_id, collect(event) AS matches
RETURN event_id,
       [event IN matches WHERE event IS NOT NULL | properties(event)] AS records
ORDER BY event_id
"""

_DIAGNOSTIC_EVENT_ABSENCE_QUERY = """
// STRUCTURAL_CLOSURE_EVENT_ROUNDTRIP_ABSENCE
UNWIND $event_ids AS event_id
OPTIONAL MATCH (event:StandardNameChange {id: event_id})
RETURN event_id, count(event) AS matches
ORDER BY event_id
"""

_DIAGNOSTIC_EVENT_WRITE_QUERY = """
// STRUCTURAL_CLOSURE_EVENT_ROUNDTRIP_WRITE
UNWIND $records AS record
CREATE (change:StandardNameChange)
SET change = record
RETURN collect(change.id) AS event_ids
"""

_DIAGNOSTIC_EVENT_READ_QUERY = """
// STRUCTURAL_CLOSURE_EVENT_ROUNDTRIP_READ
UNWIND $event_ids AS event_id
OPTIONAL MATCH (event:StandardNameChange {id: event_id})
WITH event_id, collect(event) AS matches
RETURN event_id,
       [event IN matches WHERE event IS NOT NULL | properties(event)] AS records
ORDER BY event_id
"""

_DIAGNOSTIC_DURABILITY_QUERY = """
// STRUCTURAL_CLOSURE_EVENT_ROUNDTRIP_DURABILITY
UNWIND $event_ids AS event_id
OPTIONAL MATCH (event:StandardNameChange {id: event_id})
RETURN count(event) AS durable_events
"""


class StructuralClosureConflict(RuntimeError):
    """The exact manifest-bound structural closure changed."""

    def __init__(
        self, message: str, *, diagnostic: dict[str, Any] | None = None
    ) -> None:
        self.diagnostic = copy.deepcopy(diagnostic)
        if diagnostic is not None:
            message = f"{message}: {json.dumps(diagnostic, sort_keys=True)}"
        super().__init__(message)


@dataclass(frozen=True)
class StructuralClosureManifest:
    """One exact deterministic structural closure cohort."""

    path: Path
    manifest_hash: str
    rows: tuple[dict[str, Any], ...]
    root_ids: tuple[str, ...]
    allowlist_hash: str


def _require_sha(value: Any, field: str) -> str:
    normalized = str(value or "").strip().casefold()
    if re.fullmatch(r"[0-9a-f]{64}", normalized) is None:
        raise ValueError(f"{field} must be exactly one SHA-256 hex digest")
    return normalized


def load_structural_closure_manifest(
    path: str | Path,
) -> StructuralClosureManifest:
    """Load one exact-byte, deterministic structural closure manifest."""
    manifest_path = Path(path).expanduser().resolve()
    raw = manifest_path.read_bytes()
    try:
        payload = json.loads(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"structural closure manifest is not valid JSON: {path}"
        ) from exc
    if not isinstance(payload, dict) or set(payload) != {
        "schema",
        "schema_version",
        "rows",
    }:
        raise ValueError("structural closure manifest top-level fields are not exact")
    if payload.get("schema") != _MANIFEST_SCHEMA or payload.get("schema_version") != 1:
        raise ValueError("structural closure manifest schema is unsupported")
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("structural closure manifest requires rows")
    expected_fields = {
        "root_id",
        "expected_actions",
        "retire_ids",
        "scaffold_ids",
        "unit_override",
        "expected_closure_hash",
        "expected_participant_ids_hash",
        "expected_relationship_ids_hash",
        "expected_admission_hash",
        "west_intersection",
        "test_intersection",
        "reason",
    }
    normalized_rows: list[dict[str, Any]] = []
    seen_roots: set[str] = set()
    destructive_targets: set[str] = set()
    for row in rows:
        if not isinstance(row, dict) or set(row) != expected_fields:
            raise ValueError("structural closure manifest row fields are not exact")
        root_id = str(row["root_id"])
        if not root_id or root_id in seen_roots:
            raise ValueError(f"duplicate or empty structural root: {root_id!r}")
        seen_roots.add(root_id)
        actions = row["expected_actions"]
        if (
            not isinstance(actions, list)
            or not actions
            or actions != sorted(set(actions))
            or not set(actions) <= STRUCTURAL_ACTIONS
        ):
            raise ValueError(
                "structural closure actions must be known, unique, and sorted"
            )
        retire_ids = row["retire_ids"]
        scaffold_ids = row["scaffold_ids"]
        if not all(isinstance(values, list) for values in (retire_ids, scaffold_ids)):
            raise ValueError("structural destructive target lists are required")
        row_targets = [str(item) for item in retire_ids + scaffold_ids]
        if len(row_targets) != len(set(row_targets)):
            raise ValueError("structural destructive targets overlap within a row")
        overlap = destructive_targets & set(row_targets)
        if overlap:
            raise ValueError(
                f"structural destructive targets overlap: {sorted(overlap)}"
            )
        destructive_targets.update(row_targets)
        override = row["unit_override"]
        if override is not None and (
            not isinstance(override, dict)
            or set(override) != {"unit", "provenance"}
            or not str(override["unit"]).strip()
            or not str(override["provenance"]).strip()
        ):
            raise ValueError(
                "unit override requires exact unit and reviewed provenance"
            )
        for field in (
            "expected_closure_hash",
            "expected_participant_ids_hash",
            "expected_relationship_ids_hash",
            "expected_admission_hash",
        ):
            _require_sha(row[field], field)
        if row["test_intersection"] != 0:
            raise ValueError("test intersection must be exactly zero")
        if not str(row["reason"]).strip():
            raise ValueError("every structural closure row requires a reason")
        normalized_rows.append(copy.deepcopy(row))
    normalized_rows.sort(key=lambda item: item["root_id"])
    root_ids = tuple(str(row["root_id"]) for row in normalized_rows)
    return StructuralClosureManifest(
        path=manifest_path,
        manifest_hash=hashlib.sha256(raw).hexdigest(),
        rows=tuple(normalized_rows),
        root_ids=root_ids,
        allowlist_hash=payload_hash(root_ids),
    )


def _read_rows(transaction: Any, root_ids: tuple[str, ...]) -> list[dict[str, Any]]:
    return [
        _normalize_closure_row(dict(row))
        for row in transaction.run(STRUCTURAL_CLOSURE_QUERY, root_ids=list(root_ids))
    ]


def _normalize_closure_row(row: dict[str, Any]) -> dict[str, Any]:
    """Canonicalize unordered graph collections before hashing or comparison."""
    normalized = copy.deepcopy(row)
    normalized["roots"] = sorted(
        normalized.get("roots") or [], key=lambda item: str(item.get("element_id"))
    )
    names = normalized.get("names") or []
    for name in names:
        for key in ("units", "sources", "dd_sources"):
            name[key] = sorted(
                name.get(key) or [],
                key=lambda item: (
                    str(item.get("element_id")),
                    str(item.get("relationship_element_id")),
                ),
            )
    normalized["names"] = sorted(names, key=lambda item: str(item.get("element_id")))
    edges_by_id = {
        str(edge.get("element_id")): edge
        for edge in normalized.get("parent_edges") or []
        if edge.get("element_id")
    }
    normalized["parent_edges"] = [
        edges_by_id[element_id] for element_id in sorted(edges_by_id)
    ]
    return normalized


def _participant_ids(row: dict[str, Any]) -> tuple[str, ...]:
    ids: set[str] = set()
    for root in row.get("roots") or []:
        ids.add(str(root["element_id"]))
    for name in row.get("names") or []:
        ids.add(str(name["element_id"]))
        for key in ("units", "sources", "dd_sources"):
            ids.update(
                str(item["element_id"])
                for item in name.get(key) or []
                if item.get("element_id")
            )
    return tuple(sorted(ids))


def _relationship_ids(row: dict[str, Any]) -> tuple[str, ...]:
    ids = {
        str(edge["element_id"])
        for edge in row.get("parent_edges") or []
        if edge.get("element_id")
    }
    for name in row.get("names") or []:
        for key in ("units", "sources", "dd_sources"):
            ids.update(
                str(item["relationship_element_id"])
                for item in name.get(key) or []
                if item.get("relationship_element_id")
            )
    return tuple(sorted(ids))


def _protection_intersections(row: dict[str, Any]) -> dict[str, bool]:
    """Detect protected-corpus membership inside one closure snapshot."""
    west = False
    fixture = False

    def visit(value: Any, key: str = "") -> None:
        nonlocal west, fixture
        if isinstance(value, dict):
            for child_key, child in value.items():
                visit(child, str(child_key))
        elif isinstance(value, list | tuple):
            for child in value:
                visit(child, key)
        elif isinstance(value, str):
            normalized = value.casefold()
            normalized_key = key.casefold()
            if normalized_key in {"facility", "facility_id"} and normalized == "west":
                west = True
            if normalized_key in {"id", "source_id"}:
                west = west or normalized.startswith(("west:", "signals:west:"))
                fixture = fixture or normalized.startswith(
                    ("test:", "fixture:", "signals:test:")
                )
            if normalized_key in {"origin", "source_type"} and normalized in {
                "test",
                "fixture",
            }:
                fixture = True

    visit(row)
    return {"west": west, "fixture": fixture}


def _protected_reasons(row: dict[str, Any]) -> list[str]:
    """Refuse only the closures whose membership is immutable.

    Persistent test fixtures are immutable.  Facility batch membership is
    recorded as manifest evidence, but the batch is ordinary repairable state
    and never refuses a row.
    """
    intersections = _protection_intersections(row)
    return (
        ["structural closure intersects test fixtures"]
        if intersections["fixture"]
        else []
    )


def _claim_reasons(row: dict[str, Any]) -> list[str]:
    claim_fields = {
        "claimed_at",
        "claim_token",
        "drain_scope_id",
        "drain_scope_claimed_at",
        "drain_claim_scope_id",
    }
    claimed: list[str] = []
    for name in row.get("names") or []:
        participants = [
            (
                (name.get("properties") or {}).get("id"),
                name.get("properties") or {},
            )
        ]
        for key in ("sources", "units", "dd_sources"):
            participants.extend(
                (
                    (participant.get("properties") or {}).get("id")
                    or participant.get("element_id"),
                    participant.get("properties") or {},
                )
                for participant in name.get(key) or []
            )
        for participant_id, properties in participants:
            if any(properties.get(field) is not None for field in claim_fields):
                claimed.append(str(participant_id))
    return (
        [f"active structural closure claims: {sorted(set(claimed))}"] if claimed else []
    )


def _node_by_id(row: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str((name.get("properties") or {}).get("id")): name
        for name in row.get("names") or []
        if (name.get("properties") or {}).get("id")
    }


def _direct_children(row: dict[str, Any], root_id: str) -> list[dict[str, Any]]:
    nodes = _node_by_id(row)
    children: list[dict[str, Any]] = []
    seen: set[str] = set()
    for edge in row.get("parent_edges") or []:
        if edge.get("end_id") != root_id or edge.get("start_id") in seen:
            continue
        child_id = str(edge["start_id"])
        if child_id in nodes:
            children.append({"node": nodes[child_id], "edge": edge})
            seen.add(child_id)
    return sorted(children, key=lambda item: item["edge"]["start_id"])


class _ClosureTopologyProbe:
    """Serve the existing admission oracle entirely from one captured closure."""

    def __init__(self, row: dict[str, Any], root_id: str) -> None:
        self.children = _direct_children(row, root_id)
        self.root = _node_by_id(row).get(root_id) or {}

    def query(self, cypher: str, **params: Any) -> list[dict[str, Any]]:  # noqa: ARG002
        if "collect(DISTINCT child.id) AS child_ids" in cypher:
            return [{"child_ids": [item["edge"]["start_id"] for item in self.children]}]
        if "collect(DISTINCT r.axis) AS axes" in cypher:
            axes = sorted(
                {
                    (item["edge"].get("properties") or {}).get("axis")
                    for item in self.children
                    if (item["edge"].get("properties") or {}).get("operator_kind")
                    == "projection"
                    and (item["edge"].get("properties") or {}).get("axis")
                }
            )
            return [{"axes": axes}]
        if "collect(DISTINCT csrc.id) AS child_sources" in cypher:
            if len(self.children) != 1:
                return []
            child = self.children[0]["node"]
            child_properties = child.get("properties") or {}
            if child_properties.get("name_stage") in _TERMINAL_STAGES:
                return []
            if child_properties.get("origin", "pipeline") == "derived":
                return []
            return [
                {
                    "child_id": child_properties.get("id"),
                    "child_sources": sorted(
                        (item.get("properties") or {}).get("id")
                        for item in child.get("dd_sources") or []
                        if (item.get("properties") or {}).get("id")
                    ),
                    "parent_sources": sorted(
                        (item.get("properties") or {}).get("id")
                        for item in self.root.get("dd_sources") or []
                        if (item.get("properties") or {}).get("id")
                    ),
                }
            ]
        raise StructuralClosureConflict(
            "admission oracle requested uncaptured topology"
        )


def _admission_payload(row: dict[str, Any], root_id: str) -> dict[str, Any]:
    result = is_admissible_parent_name(root_id, _ClosureTopologyProbe(row, root_id))
    reason = result.reason
    if reason.startswith("required algebraic decomposition of "):
        reason = "required algebraic decomposition child"
    return {"admit": result.admit, "reason": reason, "clause": result.clause}


def _admission_closure(
    row: dict[str, Any], target_ids: set[str]
) -> dict[str, dict[str, Any]]:
    nodes = _node_by_id(row)
    return {
        target_id: (
            _admission_payload(row, target_id)
            if target_id in nodes
            else {"admit": False, "reason": "target absent", "clause": None}
        )
        for target_id in sorted(target_ids)
    }


def build_structural_closure_manifest_row(
    row: dict[str, Any],
    *,
    root_id: str,
    expected_actions: list[str],
    retire_ids: list[str] | None = None,
    scaffold_ids: list[str] | None = None,
    unit_override: dict[str, str] | None = None,
    reason: str,
) -> dict[str, Any]:
    """Bind an audited closure snapshot to one exact manifest row."""
    if str(row.get("root_id")) != root_id:
        raise ValueError("closure snapshot root does not match the manifest root")
    intersections = _protection_intersections(row)
    target_ids = {
        root_id,
        *(str(item) for item in retire_ids or []),
        *(str(item) for item in scaffold_ids or []),
    }
    admission = (
        _admission_closure(row, target_ids)
        if len(row.get("roots") or []) == 1
        else {
            target_id: {
                "admit": False,
                "reason": "root absent or ambiguous",
                "clause": None,
            }
            for target_id in sorted(target_ids)
        }
    )
    return {
        "root_id": root_id,
        "expected_actions": sorted(set(expected_actions)),
        "retire_ids": sorted(set(retire_ids or [])),
        "scaffold_ids": sorted(set(scaffold_ids or [])),
        "unit_override": copy.deepcopy(unit_override),
        "expected_closure_hash": payload_hash(row),
        "expected_participant_ids_hash": payload_hash(_participant_ids(row)),
        "expected_relationship_ids_hash": payload_hash(_relationship_ids(row)),
        "expected_admission_hash": payload_hash(admission),
        "west_intersection": int(intersections["west"]),
        "test_intersection": int(intersections["fixture"]),
        "reason": reason,
    }


def _unit_for_name(name: dict[str, Any]) -> tuple[str | None, list[str]]:
    properties = name.get("properties") or {}
    scalar = properties.get("unit")
    edges = {
        (unit.get("properties") or {}).get("id")
        for unit in name.get("units") or []
        if (unit.get("properties") or {}).get("id")
    }
    if len(edges) > 1 or scalar and edges and scalar not in edges:
        return None, [f"unit authority is ambiguous for {properties.get('id')!r}"]
    return str(scalar or next(iter(edges), "")) or None, []


def _materialization_context(
    row: dict[str, Any], manifest_row: dict[str, Any]
) -> tuple[dict[str, Any] | None, list[str], str]:
    root_id = str(manifest_row["root_id"])
    child_rows = _direct_children(row, root_id)
    accepted: list[dict[str, Any]] = []
    unaccepted: list[str] = []
    for child_row in child_rows:
        child = child_row["node"]
        properties = child.get("properties") or {}
        live_sources = [
            source
            for source in child.get("sources") or []
            if (source.get("properties") or {}).get("status")
            in {"composed", "attached"}
        ]
        if not live_sources or properties.get("name_stage") is None:
            continue
        if properties.get("name_stage") == "accepted":
            accepted.append(child_row)
        else:
            unaccepted.append(str(properties.get("id")))
    if unaccepted:
        return (
            None,
            [f"unaccepted child authority: {sorted(unaccepted)}"],
            REFUSE_UNACCEPTED_CHILD_AUTHORITY,
        )
    if not accepted:
        return (
            None,
            ["no accepted materialized child authority"],
            REFUSE_UNACCEPTED_CHILD_AUTHORITY,
        )
    units: set[str] = set()
    child_data: list[dict[str, Any]] = []
    reasons: list[str] = []
    for child_row in accepted:
        child = child_row["node"]
        properties = child.get("properties") or {}
        edge_properties = child_row["edge"].get("properties") or {}
        child_unit, unit_reasons = _unit_for_name(child)
        reasons.extend(unit_reasons)
        is_binary = edge_properties.get("operator_kind") == "binary"
        is_normalized_child = "normalized" in str(properties.get("id")).split(
            "_"
        ) and "normalized" not in root_id.split("_")
        if child_unit and not is_binary and not is_normalized_child:
            units.add(child_unit)
        child_data.append(
            {
                "id": properties.get("id"),
                "unit": child_unit,
                "cocos": properties.get("cocos_transformation_type"),
                "physics_domain": properties.get("physics_domain"),
                "op_kind": edge_properties.get("operator_kind"),
            }
        )
    override = manifest_row.get("unit_override")
    if override is not None:
        unit = str(override["unit"])
    elif len(units) == 1:
        unit = next(iter(units))
    else:
        if len(units) > 1:
            reasons.append(f"child unit authority is heterogeneous: {sorted(units)}")
        else:
            reasons.append("no dimensionally complete unique child unit authority")
        unit = ""
    if reasons or not unit:
        return None, reasons, REFUSE_MISSING_UNIT_AUTHORITY
    if override is not None:
        for child in child_data:
            if child["op_kind"] != "binary" and not (
                "normalized" in str(child["id"]).split("_")
                and "normalized" not in root_id.split("_")
            ):
                child["unit"] = unit
    edge_kinds = sorted(
        {
            str((child["edge"].get("properties") or {}).get("operator_kind"))
            for child in child_rows
            if (child["edge"].get("properties") or {}).get("operator_kind")
        }
    )
    return (
        {
            "parent_id": root_id,
            "child_data": child_data,
            "edge_kinds": edge_kinds,
            "origin": "derived",
            "name_stage": (
                _node_by_id(row).get(root_id, {}).get("properties") or {}
            ).get("name_stage"),
            "authorized_unit": unit,
        },
        [],
        MATERIALIZE_ADMISSIBLE_PARENT,
    )


def _plan_row(
    row: dict[str, Any], manifest_row: dict[str, Any], *, include_accepted: bool
) -> dict[str, Any]:
    root_id = str(manifest_row["root_id"])
    reasons: list[str] = []
    if row.get("depth_truncated"):
        reasons.append("structural descendant closure exceeds the bounded depth")
    roots = row.get("roots") or []
    if len(roots) > 1:
        reasons.append("structural root identity is ambiguous")
    reasons.extend(_claim_reasons(row))
    reasons.extend(_protected_reasons(row))
    nodes = _node_by_id(row)
    requested_retire = {str(item) for item in manifest_row["retire_ids"]}
    requested_scaffolds = {str(item) for item in manifest_row["scaffold_ids"]}
    requested_targets = requested_retire | requested_scaffolds
    unknown_targets = requested_targets - set(nodes)
    if unknown_targets and roots:
        reasons.append(
            f"destructive targets are outside the closure: {sorted(unknown_targets)}"
        )

    admission = (
        _admission_closure(row, {root_id, *requested_targets})
        if len(roots) == 1
        else {
            target_id: {
                "admit": False,
                "reason": "root absent or ambiguous",
                "clause": None,
            }
            for target_id in sorted({root_id, *requested_targets})
        }
    )
    participant_ids = _participant_ids(row)
    relationship_ids = _relationship_ids(row)
    current_hashes = {
        "expected_closure_hash": payload_hash(row),
        "expected_participant_ids_hash": payload_hash(participant_ids),
        "expected_relationship_ids_hash": payload_hash(relationship_ids),
        "expected_admission_hash": payload_hash(admission),
    }

    if not roots:
        if requested_targets and requested_targets == unknown_targets:
            return {
                "root_id": root_id,
                "status": "already_current",
                "actions": sorted(set(manifest_row["expected_actions"])),
                "unresolved": [],
                "participant_ids": [],
                "relationship_ids": [],
                "precondition_hash": payload_hash(row),
                "mutation": {},
            }
        reasons.append("structural root is missing")

    root = nodes.get(root_id) or {}
    root_properties = root.get("properties") or {}
    derived_sources = [
        source
        for source in root.get("sources") or []
        if (source.get("properties") or {}).get("id") == f"derived:{root_id}"
    ]
    if derived_sources and not requested_targets:
        return {
            "root_id": root_id,
            "status": "already_current",
            "actions": sorted(set(manifest_row["expected_actions"])),
            "unresolved": [],
            "participant_ids": list(participant_ids),
            "relationship_ids": list(relationship_ids),
            "precondition_hash": payload_hash(row),
            "mutation": {},
        }

    actions: list[str] = []
    mutation: dict[str, Any] = {}
    if requested_targets:
        sourced = sorted(
            target_id
            for target_id in requested_targets
            if (nodes.get(target_id) or {}).get("sources")
        )
        if sourced:
            reasons.append(f"destructive structural targets have producers: {sourced}")
        accepted_targets = sorted(
            target_id
            for target_id in requested_targets
            if ((nodes.get(target_id) or {}).get("properties") or {}).get("name_stage")
            == "accepted"
        )
        if accepted_targets and not include_accepted:
            reasons.append(
                f"accepted structural deletion requires include_accepted: {accepted_targets}"
            )
        for target_id in sorted(requested_retire):
            stage = ((nodes.get(target_id) or {}).get("properties") or {}).get(
                "name_stage"
            )
            if stage is None:
                reasons.append(
                    f"retirement target {target_id!r} is an id-only scaffold"
                )
        for target_id in sorted(requested_scaffolds):
            stage = ((nodes.get(target_id) or {}).get("properties") or {}).get(
                "name_stage"
            )
            if stage is not None:
                reasons.append(
                    f"scaffold target {target_id!r} has a materialized lifecycle"
                )
        if requested_retire:
            actions.append(RETIRE_UNREACHABLE_CHAIN)
        if requested_scaffolds:
            actions.append(EXCLUDE_NULL_SCAFFOLD)
        mutation = {"delete_ids": sorted(requested_targets)}
    elif root_properties.get("name_stage") == "accepted":
        if not admission[root_id]["admit"]:
            reasons.append(
                "accepted parent fails current admission: "
                f"{admission[root_id]['reason']}"
            )
        actions.append(SEED_ACCEPTED_PARENT_SOURCE)
        mutation = {"seed_parent_id": root_id}
    else:
        context, context_reasons, action = _materialization_context(row, manifest_row)
        reasons.extend(context_reasons)
        actions.append(action)
        if action == MATERIALIZE_ADMISSIBLE_PARENT:
            if not admission[root_id]["admit"]:
                reasons.append(
                    f"parent fails current admission: {admission[root_id]['reason']}"
                )
            mutation = {"materialize": context}

    if sorted(set(actions)) != sorted(set(manifest_row["expected_actions"])):
        reasons.append(
            "manifest expected_actions drifted: "
            f"expected={manifest_row['expected_actions']!r}, current={sorted(set(actions))!r}"
        )
    for field, current in current_hashes.items():
        if manifest_row[field] != current:
            reasons.append(f"manifest {field} drifted")
    status = "refused" if reasons else "planned"
    return {
        "root_id": root_id,
        "status": status,
        "actions": sorted(set(actions)),
        "unresolved": sorted(set(reasons)),
        "participant_ids": list(participant_ids),
        "relationship_ids": list(relationship_ids),
        "precondition_hash": payload_hash(
            {"row": row, "admission": admission, "actions": sorted(set(actions))}
        ),
        "mutation": mutation,
    }


def _read_plan(
    transaction: Any,
    manifest: StructuralClosureManifest,
    *,
    include_accepted: bool,
) -> list[dict[str, Any]]:
    rows = _read_rows(transaction, manifest.root_ids)
    if [row.get("root_id") for row in rows] != list(manifest.root_ids):
        raise StructuralClosureConflict("structural closure read omitted an exact root")
    by_root = {str(row["root_id"]): row for row in manifest.rows}
    return [
        _plan_row(row, by_root[str(row["root_id"])], include_accepted=include_accepted)
        for row in rows
    ]


class _TransactionQueryAdapter:
    def __init__(self, transaction: Any) -> None:
        self.transaction = transaction

    def query(self, cypher: str, **params: Any) -> list[dict[str, Any]]:
        return [dict(row) for row in self.transaction.run(cypher, **params)]


class _DiagnosticTransaction:
    def __init__(self, transaction: Any) -> None:
        self.transaction = transaction
        self.query_count = 0

    def run(self, cypher: str, **params: Any) -> Any:
        self.query_count += 1
        return self.transaction.run(cypher, **params)

    def rollback(self) -> None:
        self.transaction.rollback()


def _normalized_event_record(record: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(record)
    changed_at = normalized.get("changed_at")
    nanosecond = getattr(changed_at, "nanosecond", None)
    if hasattr(changed_at, "to_native"):
        changed_at = changed_at.to_native()
    if isinstance(changed_at, datetime):
        if changed_at.tzinfo is None or changed_at.utcoffset() is None:
            raise StructuralClosureConflict(
                "structural event timestamp lost its timezone authority"
            )
        utc_instant = changed_at.astimezone(UTC)
        since_epoch = utc_instant - datetime(1970, 1, 1, tzinfo=UTC)
        normalized["changed_at"] = {
            "epoch_seconds": since_epoch.days * 86_400 + since_epoch.seconds,
            "nanosecond": (
                int(nanosecond)
                if nanosecond is not None
                else utc_instant.microsecond * 1000
            ),
        }
    return normalized


def _event_hash(record: dict[str, Any]) -> str:
    return payload_hash(_normalized_event_record(record))


def _event_value_type(value: Any) -> str:
    if value is None:
        return "none"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "str"
    if isinstance(value, dict):
        return "mapping"
    if isinstance(value, list | tuple):
        return type(value).__name__
    return f"{type(value).__module__}.{type(value).__qualname__}"


def _event_field_summary(key: str, value: Any) -> dict[str, Any]:
    canonical_value = _normalized_event_record({key: value})[key]
    summary = {
        "type": _event_value_type(value),
        "canonical_hash": payload_hash(canonical_value),
    }
    if key in _SAFE_EVENT_LITERAL_FIELDS:
        summary["canonical_value"] = canonical_value
    return summary


def _event_record_comparison(
    event_id: str,
    expected_record: dict[str, Any],
    actual_records: list[dict[str, Any]],
) -> dict[str, Any]:
    normalized_expected = _normalized_event_record(expected_record)
    normalized_actual = [
        _normalized_event_record(dict(record)) for record in actual_records
    ]
    expected_hash = payload_hash(normalized_expected)
    actual_hashes = [payload_hash(record) for record in normalized_actual]
    actual_record = actual_records[0] if len(actual_records) == 1 else None
    normalized_single = normalized_actual[0] if len(normalized_actual) == 1 else {}
    expected_keys = set(expected_record)
    actual_keys = set(actual_record or {})
    fields = {}
    for key in sorted(expected_keys | actual_keys):
        expected = (
            _event_field_summary(key, expected_record[key])
            if key in expected_record
            else None
        )
        actual = (
            _event_field_summary(key, actual_record[key])
            if actual_record is not None and key in actual_record
            else None
        )
        fields[key] = {
            "expected": expected,
            "actual": actual,
            "matches": expected is not None
            and actual is not None
            and expected["canonical_hash"] == actual["canonical_hash"],
        }
    missing_keys = sorted(expected_keys - actual_keys)
    extra_keys = sorted(actual_keys - expected_keys)
    mismatch_fields = sorted(
        key for key, comparison in fields.items() if not comparison["matches"]
    )
    match = (
        len(actual_records) == 1
        and not missing_keys
        and not extra_keys
        and expected_hash == actual_hashes[0]
    )
    return {
        "event_id": event_id,
        "matches": match,
        "actual_record_count": len(actual_records),
        "expected_record_hash": expected_hash,
        "actual_record_hash": actual_hashes[0] if len(actual_hashes) == 1 else None,
        "actual_record_hashes": actual_hashes,
        "missing_keys": missing_keys,
        "extra_keys": extra_keys,
        "mismatch_fields": mismatch_fields,
        "fields": fields,
        "canonical_actual_keys": sorted(normalized_single),
    }


def _event_record_comparisons(
    records: dict[str, dict[str, Any]], rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    by_id = {str(row["event_id"]): row.get("records") or [] for row in rows}
    return [
        _event_record_comparison(
            event_id,
            records[event_id],
            [dict(record) for record in by_id.get(event_id, [])],
        )
        for event_id in sorted(records)
    ]


def _event_record(
    manifest: StructuralClosureManifest,
    *,
    action: str,
    root_id: str,
    target_id: str,
    reason: str,
    changed_at: datetime,
) -> dict[str, Any]:
    identity = {
        "manifest_hash": manifest.manifest_hash,
        "action": action,
        "root_id": root_id,
        "target_id": target_id,
    }
    operation = {
        MATERIALIZE_ADMISSIBLE_PARENT: "materialize_structural_parent",
        SEED_ACCEPTED_PARENT_SOURCE: "seed_structural_parent_source",
        RETIRE_UNREACHABLE_CHAIN: "reconcile_structural_closure",
        EXCLUDE_NULL_SCAFFOLD: "reconcile_structural_closure",
    }[action]
    return {
        "id": "sn-change:structural:" + payload_hash(identity),
        "from_name": target_id,
        "to_name": target_id,
        "operation": operation,
        "reason": reason,
        "origin": "structural_reconciliation",
        "run_id": "structural:" + manifest.manifest_hash,
        "changed_at": changed_at,
        "internal": True,
        **identity,
    }


def _event_bindings(
    manifest: StructuralClosureManifest,
    plans: list[dict[str, Any]],
    manifest_by_root: dict[str, dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    changed_at = datetime.now(UTC)
    records: dict[str, dict[str, Any]] = {}
    receipts: list[dict[str, Any]] = []
    for plan in sorted(plans, key=lambda item: item["root_id"]):
        root_id = str(plan["root_id"])
        manifest_row = manifest_by_root[root_id]
        targets: list[tuple[str, str]] = []
        if plan["mutation"].get("materialize"):
            targets.append((MATERIALIZE_ADMISSIBLE_PARENT, root_id))
        if plan["mutation"].get("seed_parent_id"):
            targets.append((SEED_ACCEPTED_PARENT_SOURCE, root_id))
        scaffold_ids = {str(item) for item in manifest_row["scaffold_ids"]}
        for target_id in plan["mutation"].get("delete_ids") or []:
            action = (
                EXCLUDE_NULL_SCAFFOLD
                if target_id in scaffold_ids
                else RETIRE_UNREACHABLE_CHAIN
            )
            targets.append((action, str(target_id)))
        for action, target_id in targets:
            record = _event_record(
                manifest,
                action=action,
                root_id=root_id,
                target_id=target_id,
                reason=str(manifest_row["reason"]),
                changed_at=changed_at,
            )
            event_id = str(record["id"])
            if event_id in records:
                raise StructuralClosureConflict("structural event identity collided")
            records[event_id] = record
            receipts.append(
                {
                    "id": event_id,
                    "hash": _event_hash(record),
                    "action": action,
                    "root_id": root_id,
                    "target_id": target_id,
                }
            )
    return records, sorted(receipts, key=lambda item: item["id"])


def _lock_relationships(
    transaction: Any, relationship_ids: set[str], root_ids: tuple[str, ...]
) -> None:
    exact_ids = sorted(relationship_ids)
    rows = list(
        transaction.run(
            _RELATIONSHIP_LOCK_QUERY,
            element_ids=exact_ids,
            root_ids=list(root_ids),
        )
    )
    locked = int(dict(rows[0]).get("locked") or 0) if rows else 0
    if locked != len(exact_ids):
        raise StructuralClosureConflict("structural closure relationship set changed")


def _apply_plans(
    transaction: Any,
    manifest: StructuralClosureManifest,
    plans: list[dict[str, Any]],
    manifest_by_root: dict[str, dict[str, Any]],
) -> tuple[int, list[dict[str, Any]], dict[str, dict[str, Any]]]:
    adapter = _TransactionQueryAdapter(transaction)
    records, event_receipts = _event_bindings(manifest, plans, manifest_by_root)
    events_by_target = {str(record["target_id"]): record for record in records.values()}
    materializations = [
        plan["mutation"]["materialize"]
        for plan in plans
        if plan["mutation"].get("materialize")
    ]
    source_seeds = [
        {
            "parent_id": plan["mutation"]["seed_parent_id"],
            "origin": "derived",
        }
        for plan in plans
        if plan["mutation"].get("seed_parent_id")
    ]
    try:
        materialized = _materialize_derived_parent_rows_batched(
            adapter,
            materializations,
            event_by_parent=events_by_target,
        )
        seeded = reconcile_orphan_parent_sources_batched(
            adapter,
            source_seeds,
            event_by_parent=events_by_target,
        )
    except RuntimeError as exc:
        raise StructuralClosureConflict(str(exc)) from exc
    delete_items = []
    for plan in plans:
        manifest_row = manifest_by_root[plan["root_id"]]
        for target_id in plan["mutation"].get("delete_ids") or []:
            deletion_change_params(
                "reconcile_structural_closure",
                reason=str(manifest_row["reason"]),
                origin="structural_reconciliation",
            )
            delete_items.append(
                {
                    "id": target_id,
                    "event": events_by_target[str(target_id)],
                }
            )
    rows = list(transaction.run(_DELETE_QUERY, items=delete_items))
    mutation = dict(rows[0]) if len(rows) == 1 else {}
    deleted = set(mutation.get("deleted_ids") or [])
    deletion_event_ids = set(mutation.get("event_ids") or [])
    if deleted != {item["id"] for item in delete_items} or deletion_event_ids != {
        item["event"]["id"] for item in delete_items
    }:
        raise StructuralClosureConflict("structural deletion cardinality changed")
    changed = materialized + seeded + len(deleted)
    return changed, event_receipts, records


def _verify_events(
    transaction: Any,
    records: dict[str, dict[str, Any]],
    event_receipts: list[dict[str, Any]],
) -> None:
    rows = [
        dict(row)
        for row in transaction.run(_EVENT_READ_QUERY, event_ids=sorted(records))
    ]
    comparisons = _event_record_comparisons(records, rows)
    expected_hashes = {str(item["id"]): str(item["hash"]) for item in event_receipts}
    mismatches = [
        comparison
        for comparison in comparisons
        if not comparison["matches"]
        or comparison["expected_record_hash"] != expected_hashes[comparison["event_id"]]
    ]
    if mismatches:
        raise StructuralClosureConflict(
            "structural event postflight record hash changed",
            diagnostic={
                "schema": _EVENT_DIAGNOSTIC_SCHEMA,
                "mismatches": mismatches,
            },
        )


def _receipt(
    manifest: StructuralClosureManifest,
    plans: list[dict[str, Any]],
    *,
    apply: bool,
    changed: int = 0,
    events: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    counts = Counter(plan["status"] for plan in plans)
    mode = (
        "refused"
        if counts["refused"]
        else "applied"
        if apply and changed
        else "already_current"
        if counts["already_current"] == len(plans)
        else "dry_run"
    )
    receipt = {
        "schema": _RECEIPT_SCHEMA,
        "schema_version": 1,
        "mode": mode,
        "manifest_path": str(manifest.path),
        "manifest_hash": manifest.manifest_hash,
        "allowlist_hash": manifest.allowlist_hash,
        "counts": {
            "allowlisted": len(plans),
            "planned": counts["planned"],
            "already_current": counts["already_current"],
            "refused": counts["refused"],
            "changed": changed,
        },
        "rows": [
            {
                "root_id": plan["root_id"],
                "status": plan["status"],
                "actions": plan["actions"],
                "unresolved": plan["unresolved"],
                "precondition_hash": plan["precondition_hash"],
            }
            for plan in plans
        ],
        "events": sorted(events or [], key=lambda item: item["id"]),
    }
    receipt["receipt_hash"] = payload_hash(receipt)
    return receipt


def diagnose_structural_event_roundtrip(
    manifest_path: str | Path,
    *,
    expected_manifest_hash: str,
    include_accepted: bool = False,
    gc: Any | None = None,
) -> dict[str, Any]:
    """Round-trip planned event records in one transaction that always rolls back."""
    normalized_hash = _require_sha(expected_manifest_hash, "expected_manifest_hash")
    manifest = load_structural_closure_manifest(manifest_path)
    if not hmac.compare_digest(normalized_hash, manifest.manifest_hash):
        raise ValueError("manifest SHA-256 does not match the exact parsed bytes")
    own = gc is None
    client = GraphClient() if own else gc
    event_ids: list[str] = []
    writes_started = False
    transaction_query_count = 0
    diagnostic: dict[str, Any] | None = None
    error: Exception | None = None
    try:
        with client.session() as session:
            transaction = _DiagnosticTransaction(session.begin_transaction())
            try:
                plans = _read_plan(
                    transaction, manifest, include_accepted=include_accepted
                )
                refused = [
                    str(plan["root_id"])
                    for plan in plans
                    if plan["status"] == "refused"
                ]
                if refused:
                    raise StructuralClosureConflict(
                        "structural event diagnostic refuses unresolved roots",
                        diagnostic={"root_ids": sorted(refused)},
                    )
                pending = [plan for plan in plans if plan["status"] == "planned"]
                by_root = {str(row["root_id"]): row for row in manifest.rows}
                records, event_receipts = _event_bindings(manifest, pending, by_root)
                event_ids = sorted(records)
                if not event_ids:
                    raise StructuralClosureConflict(
                        "structural event diagnostic requires planned events"
                    )
                absence_rows = [
                    dict(row)
                    for row in transaction.run(
                        _DIAGNOSTIC_EVENT_ABSENCE_QUERY,
                        event_ids=event_ids,
                    )
                ]
                existing = {
                    str(row["event_id"]): int(row.get("matches") or 0)
                    for row in absence_rows
                    if int(row.get("matches") or 0) != 0
                }
                returned_ids = {str(row["event_id"]) for row in absence_rows}
                if returned_ids != set(event_ids) or existing:
                    raise StructuralClosureConflict(
                        "structural event diagnostic requires absent event ids",
                        diagnostic={
                            "expected_event_ids": event_ids,
                            "existing_event_counts": existing,
                            "omitted_event_ids": sorted(set(event_ids) - returned_ids),
                        },
                    )
                writes_started = True
                write_rows = [
                    dict(row)
                    for row in transaction.run(
                        _DIAGNOSTIC_EVENT_WRITE_QUERY,
                        records=[records[event_id] for event_id in event_ids],
                    )
                ]
                written_ids = sorted(
                    str(event_id)
                    for row in write_rows
                    for event_id in row.get("event_ids") or []
                )
                if written_ids != event_ids:
                    raise StructuralClosureConflict(
                        "structural event diagnostic write cardinality changed",
                        diagnostic={
                            "expected_event_ids": event_ids,
                            "written_event_ids": written_ids,
                        },
                    )
                hydrated_rows = [
                    dict(row)
                    for row in transaction.run(
                        _DIAGNOSTIC_EVENT_READ_QUERY,
                        event_ids=event_ids,
                    )
                ]
                comparisons = _event_record_comparisons(records, hydrated_rows)
                diagnostic = {
                    "schema": _EVENT_DIAGNOSTIC_SCHEMA,
                    "schema_version": 1,
                    "manifest_hash": manifest.manifest_hash,
                    "allowlist_hash": manifest.allowlist_hash,
                    "planned_root_ids": sorted(
                        str(plan["root_id"]) for plan in pending
                    ),
                    "events": event_receipts,
                    "comparisons": comparisons,
                    "matching_event_ids": sorted(
                        comparison["event_id"]
                        for comparison in comparisons
                        if comparison["matches"]
                    ),
                    "mismatch_event_ids": sorted(
                        comparison["event_id"]
                        for comparison in comparisons
                        if not comparison["matches"]
                    ),
                    "all_records_match": all(
                        comparison["matches"] for comparison in comparisons
                    ),
                }
            except Exception as exc:
                error = exc
            finally:
                transaction_query_count = transaction.query_count
                transaction.rollback()

        durable_events = 0
        if writes_started:
            durability_rows = list(
                client.query(_DIAGNOSTIC_DURABILITY_QUERY, event_ids=event_ids)
            )
            durable_events = (
                int(durability_rows[0].get("durable_events") or 0)
                if len(durability_rows) == 1
                else -1
            )
            if durable_events != 0:
                raise StructuralClosureConflict(
                    "structural event diagnostic rollback left durable events",
                    diagnostic={
                        "event_ids": event_ids,
                        "durable_events": durable_events,
                    },
                ) from error
        if error is not None:
            raise error
        if diagnostic is None:
            raise StructuralClosureConflict(
                "structural event diagnostic produced no receipt"
            )
        diagnostic.update(
            {
                "rolled_back": True,
                "durable_event_count": durable_events,
                "diagnostic_transaction_queries": transaction_query_count,
                "postrollback_queries": 1,
                "query_count": transaction_query_count + 1,
                "transaction_count": 2,
                "rollback_count": 1,
                "commit_count": 0,
            }
        )
        diagnostic["receipt_hash"] = payload_hash(diagnostic)
        return diagnostic
    finally:
        if own:
            client.close()


@retry_on_deadlock()
def reconcile_structural_closure(
    manifest_path: str | Path,
    *,
    dry_run: bool = True,
    include_accepted: bool = False,
    expected_manifest_hash: str | None = None,
    gc: Any | None = None,
) -> dict[str, Any]:
    """Plan or atomically apply one exact structural closure cohort."""
    apply = not dry_run
    normalized_hash = normalize_manifest_hash_binding(
        expected_manifest_hash, apply=apply
    )
    manifest = load_structural_closure_manifest(manifest_path)
    if normalized_hash is not None and not hmac.compare_digest(
        normalized_hash, manifest.manifest_hash
    ):
        raise ValueError("manifest SHA-256 does not match the exact parsed bytes")
    own = gc is None
    client = GraphClient() if own else gc
    try:
        with client.session() as session:
            transaction = session.begin_transaction()
            try:
                plans = _read_plan(
                    transaction, manifest, include_accepted=include_accepted
                )
                if any(plan["status"] == "refused" for plan in plans) or dry_run:
                    transaction.rollback()
                    return _receipt(manifest, plans, apply=False)
                pending = [plan for plan in plans if plan["status"] == "planned"]
                if not pending:
                    transaction.rollback()
                    return _receipt(manifest, plans, apply=True)
                lock_participants(
                    transaction,
                    {
                        participant
                        for plan in pending
                        for participant in plan["participant_ids"]
                    },
                    conflict_type=StructuralClosureConflict,
                    message="structural closure participant set changed",
                )
                _lock_relationships(
                    transaction,
                    {
                        relationship
                        for plan in pending
                        for relationship in plan["relationship_ids"]
                    },
                    manifest.root_ids,
                )
                locked = _read_plan(
                    transaction, manifest, include_accepted=include_accepted
                )
                if [plan["precondition_hash"] for plan in locked] != [
                    plan["precondition_hash"] for plan in plans
                ]:
                    raise StructuralClosureConflict(
                        "structural closure changed after locks"
                    )
                by_root = {str(row["root_id"]): row for row in manifest.rows}
                changed, events, event_records = _apply_plans(
                    transaction, manifest, pending, by_root
                )
                _verify_events(transaction, event_records, events)
                post = _read_plan(
                    transaction, manifest, include_accepted=include_accepted
                )
                if any(
                    plan["status"] not in {"already_current", "refused"}
                    for plan in post
                ):
                    raise StructuralClosureConflict(
                        "structural closure postflight failed"
                    )
                before_unresolved = {
                    plan["root_id"]: plan["unresolved"]
                    for plan in plans
                    if plan["status"] == "refused"
                }
                after_unresolved = {
                    plan["root_id"]: plan["unresolved"]
                    for plan in post
                    if plan["status"] == "refused"
                }
                if before_unresolved != after_unresolved:
                    raise StructuralClosureConflict(
                        "structural closure unresolved set changed during apply"
                    )
                transaction.commit()
                return _receipt(
                    manifest,
                    post,
                    apply=True,
                    changed=changed,
                    events=events,
                )
            except Exception:
                transaction.rollback()
                raise
    finally:
        if own:
            client.close()
