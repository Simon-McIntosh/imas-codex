"""Typed, graph-backed interpretations of Data Dictionary declarations.

The live graph is the sole authority.  Each resolution is linked from one exact
``IMASNode``, links to one evidencing ``DDGap``, and records an upstream
reference (or ``none-yet``) together with who recorded it, when, and why.
Resolution reads fail closed when either gate or any exact-key invariant is not
satisfied.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from enum import Enum, StrEnum
from typing import Any, Protocol, Self

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from imas_codex.core.physics_domain import PhysicsDomain
from imas_codex.graph.models import (
    COCOSLabelTransformation,
    DDDataType,
    DDNodeType,
    DDResolutionField,
    DDResolutionStatus,
    DDResolutionValueKind,
    LifecycleStatus,
)

_RESOLUTION_MARKER = "resolved-dd-context"
_EXACT_VERSION_RE = re.compile(
    r"[0-9]+\.[0-9]+\.[0-9]+(?:[-+][0-9A-Za-z][0-9A-Za-z.-]*)?\Z"
)
_PATTERN_CHARACTERS = frozenset("*?[]{}\\^$|")
_NONE_YET = "none-yet"

_IONISATION_POTENTIAL_PARENTS = (
    "edge_profiles/ggd/ion/state/ionisation_potential",
    "plasma_profiles/ggd/ion/state/ionisation_potential",
)
_IONISATION_POTENTIAL_LEAVES = (
    "",
    "/coefficients",
    "/coefficients_error_lower",
    "/coefficients_error_upper",
    "/values",
    "/values_error_lower",
    "/values_error_upper",
)
IONISATION_POTENTIAL_RESOLUTION_PATHS = tuple(
    parent + leaf
    for parent in _IONISATION_POTENTIAL_PARENTS
    for leaf in _IONISATION_POTENTIAL_LEAVES
)
_IONISATION_POTENTIAL_UPSTREAM = (
    "https://github.com/iterorganization/IMAS-Data-Dictionary/pull/280"
)
_IONISATION_POTENTIAL_COMMIT = "commits:30a5ddd4b7037b9f93a8f00f7837809403349d99"


class DDResolutionError(RuntimeError):
    """Base class for fail-closed DD resolution errors."""


class DDResolutionManifestInvalid(DDResolutionError):
    """The graph authority is unavailable or structurally invalid."""


class DDResolutionCollision(DDResolutionError):
    """Multiple graph records claim one exact behavior key."""


class DDResolutionVersionMismatch(DDResolutionError):
    """A resolution exists for the field, but not this DD version."""


class DDResolutionStale(DDResolutionError):
    """The current value matches neither side of the recorded bridge."""


class DDResolutionEvidenceMismatch(DDResolutionError):
    """A graph resolution is missing required provenance or gate edges."""


class DDResolutionAmbiguity(DDResolutionError):
    """More than one prior record could certify convergence."""


class DDResolutionGraphPortConflict(DDResolutionError):
    """The graph cannot accept an exact resolution batch atomically."""


class DDResolutionGraphPathAction(StrEnum):
    """Mutation performed for one exact graph path."""

    corrected = "corrected"
    attached = "attached"
    unchanged = "unchanged"


class DDResolutionGraphPathReceipt(BaseModel):
    """Disposition of one exact path in a graph write batch."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    resolution_id: str
    path: str
    action: DDResolutionGraphPathAction


class DDResolutionGraphPortReceipt(BaseModel):
    """Aggregate receipt for an atomic graph write batch."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    expected: int
    writes: int
    nodes: int
    bridged_edges: int
    evidenced_edges: int
    version_edges: int
    path_receipts: tuple[DDResolutionGraphPathReceipt, ...]
    replay: bool
    receipt_hash: str


class DDResolutionCohortReceipt(BaseModel):
    """Evidence and graph receipts for the exact ionisation-potential cohort."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    evidence: Mapping[str, Any]
    graph: DDResolutionGraphPortReceipt
    existing: int
    added: int


def _enum_text(value: Any) -> str:
    return str(value.value if isinstance(value, Enum) else value)


def _canonical_json(value: Any) -> str:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _canonical_digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical_json(value).encode()).hexdigest()


def _validate_exact_path(path: str) -> str:
    clean = path.strip()
    if (
        not clean
        or clean.startswith("/")
        or "/" not in clean
        or any(character in clean for character in _PATTERN_CHARACTERS)
        or any(part in {"", ".", ".."} for part in clean.split("/"))
    ):
        raise ValueError("DD resolution path must be one exact IDS-prefixed path")
    return clean


def _validate_exact_version(version: str) -> str:
    clean = version.strip()
    if not _EXACT_VERSION_RE.fullmatch(clean):
        raise ValueError("DD resolution version must be one exact published version")
    return clean


class DDResolutionValue(BaseModel):
    """One canonical typed published or effective field value."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: DDResolutionValueKind
    value: str | tuple[str, ...] | None

    @model_validator(mode="after")
    def _validate_kind(self) -> Self:
        if self.kind == DDResolutionValueKind.string and not isinstance(
            self.value, str
        ):
            raise ValueError("kind='string' requires one string value")
        if self.kind == DDResolutionValueKind.string_list and (
            not isinstance(self.value, tuple)
            or any(not isinstance(item, str) for item in self.value)
        ):
            raise ValueError("kind='string_list' requires a list of strings")
        if self.kind == DDResolutionValueKind.null and self.value is not None:
            raise ValueError("kind='null' requires value=null")
        return self


_FIELD_VALUE_ENUMS: dict[DDResolutionField, type[Enum]] = {
    DDResolutionField.data_type: DDDataType,
    DDResolutionField.node_type: DDNodeType,
    DDResolutionField.physics_domain: PhysicsDomain,
    DDResolutionField.cocos_transformation_type: COCOSLabelTransformation,
    DDResolutionField.lifecycle_status: LifecycleStatus,
}


def _validate_field_value(field: DDResolutionField, value: DDResolutionValue) -> None:
    if value.kind == DDResolutionValueKind.string_list:
        if any(
            not item.strip()
            or any(character in item for character in _PATTERN_CHARACTERS)
            for item in value.value
        ):
            raise ValueError("coordinate identities must be nonempty exact strings")
        return
    if value.kind == DDResolutionValueKind.null:
        return
    if field == DDResolutionField.lifecycle_version:
        _validate_exact_version(value.value)
        return
    enum_type = _FIELD_VALUE_ENUMS.get(field)
    if enum_type is not None:
        try:
            enum_type(value.value)
        except ValueError as exc:
            raise ValueError(
                f"field={field.value!r} value {value.value!r} is not declared by "
                f"{enum_type.__name__}"
            ) from exc


class DDResolutionRecord(BaseModel):
    """One exact resolution bridge and its durable graph provenance."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    id: str
    gap_id: str
    path: str
    dd_version: str
    field: DDResolutionField
    observed: DDResolutionValue
    effective: DDResolutionValue
    reason: str
    recorded_by: str
    recorded_at: datetime
    upstream_reference: str
    upstream_commit_reference: str | None = None
    retiring_release: str
    state: DDResolutionStatus = DDResolutionStatus.active

    _exact_path = field_validator("path")(_validate_exact_path)
    _exact_version = field_validator("dd_version")(_validate_exact_version)

    @field_validator(
        "id",
        "gap_id",
        "reason",
        "recorded_by",
        "upstream_reference",
        "retiring_release",
    )
    @classmethod
    def _required_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("resolution provenance fields cannot be empty")
        return value

    @model_validator(mode="after")
    def _validate_contract(self) -> Self:
        allowed = (
            {DDResolutionValueKind.string_list}
            if self.field == DDResolutionField.coordinates
            else {DDResolutionValueKind.string, DDResolutionValueKind.null}
        )
        if self.observed.kind not in allowed or self.effective.kind not in allowed:
            raise ValueError(f"field={self.field.value!r} has incompatible value kind")
        _validate_field_value(self.field, self.observed)
        _validate_field_value(self.field, self.effective)
        if self.state == DDResolutionStatus.active and self.observed == self.effective:
            raise DDResolutionEvidenceMismatch(
                "an active resolution must bridge a change"
            )
        return self


class DDResolutionManifest(BaseModel):
    """Validated snapshot of the complete graph resolution authority."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    resolutions: tuple[DDResolutionRecord, ...]

    @model_validator(mode="after")
    def _validate_manifest(self) -> Self:
        ids: set[str] = set()
        keys: set[tuple[str, str, DDResolutionField]] = set()
        for record in self.resolutions:
            if record.id in ids:
                raise DDResolutionCollision(f"duplicate resolution id {record.id!r}")
            ids.add(record.id)
            if record.state != DDResolutionStatus.active:
                continue
            key = (record.path, record.dd_version, record.field)
            if key in keys:
                raise DDResolutionCollision(
                    f"multiple active resolutions claim {key!r}"
                )
            keys.add(key)
        return self

    @property
    def digest(self) -> str:
        """Stable digest of this graph snapshot, used only as read provenance."""
        return _canonical_digest(
            [
                record.model_dump(mode="json")
                for record in sorted(self.resolutions, key=lambda r: r.id)
            ]
        )


def _canonical_datetime(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("resolution timestamp must include a UTC offset")
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _graph_record(record: DDResolutionRecord) -> dict[str, Any]:
    return {
        "id": record.id,
        "path": record.path,
        "dd_version": record.dd_version,
        "field": record.field.value,
        "published_kind": record.observed.kind.value,
        "published_value": _canonical_json(record.observed.value),
        "effective_kind": record.effective.kind.value,
        "effective_value": _canonical_json(record.effective.value),
        "reason": record.reason,
        "recorded_by": record.recorded_by,
        "recorded_at": _canonical_datetime(record.recorded_at),
        "upstream_reference": record.upstream_reference,
        "upstream_commit_reference": record.upstream_commit_reference,
        "retiring_release": record.retiring_release,
        "source_manifest_digest": "graph-native",
        "status": record.state.value,
        "corrected_node": record.path,
        "evidence": record.gap_id,
        "for_dd_version": record.dd_version,
    }


def _resolution_id(path: str) -> str:
    digest = hashlib.sha256(
        _canonical_json(
            {
                "path": path,
                "dd_version": "4.1.1",
                "field": DDResolutionField.unit.value,
                "published": "e",
                "effective": "eV",
            }
        ).encode()
    ).hexdigest()
    return f"dd_resolution:{digest}"


def ionisation_potential_resolution_records(
    *,
    manifest: DDResolutionManifest,
    recorded_by: str,
    recorded_at: datetime,
    reason: str,
) -> tuple[dict[str, Any], ...]:
    """Build graph records for cohort paths without active exact authority."""
    existing = {
        record.path: record
        for record in manifest.resolutions
        if record.state == DDResolutionStatus.active
        and record.dd_version == "4.1.1"
        and record.field == DDResolutionField.unit
        and record.path in IONISATION_POTENTIAL_RESOLUTION_PATHS
    }
    invalid = [
        path
        for path, record in existing.items()
        if record.observed
        != DDResolutionValue(kind=DDResolutionValueKind.string, value="e")
        or record.effective
        != DDResolutionValue(kind=DDResolutionValueKind.string, value="eV")
        or record.upstream_reference != _IONISATION_POTENTIAL_UPSTREAM
        or record.upstream_commit_reference != _IONISATION_POTENTIAL_COMMIT
    ]
    if invalid:
        raise DDResolutionGraphPortConflict(
            "existing ionisation-potential authority disagrees on exact paths: "
            + ", ".join(sorted(invalid))
        )
    records = []
    for path in IONISATION_POTENTIAL_RESOLUTION_PATHS:
        if path in existing:
            continue
        record = DDResolutionRecord(
            id=_resolution_id(path),
            gap_id=f"dd_gap:{path}:self_contradiction",
            path=path,
            dd_version="4.1.1",
            field=DDResolutionField.unit,
            observed=DDResolutionValue(kind=DDResolutionValueKind.string, value="e"),
            effective=DDResolutionValue(kind=DDResolutionValueKind.string, value="eV"),
            reason=reason,
            recorded_by=recorded_by,
            recorded_at=recorded_at,
            upstream_reference=_IONISATION_POTENTIAL_UPSTREAM,
            upstream_commit_reference=_IONISATION_POTENTIAL_COMMIT,
            retiring_release="4.2.0",
            state=DDResolutionStatus.active,
        )
        records.append(_graph_record(record))
    return tuple(records)


class DDResolutionGraphPort(Protocol):
    """Typed boundary for one atomic graph materialization."""

    def apply(self, records: tuple[dict[str, Any], ...]) -> Mapping[str, Any]: ...


_GRAPH_PORT_PREFLIGHT_QUERY = """
UNWIND $records AS b
OPTIONAL MATCH (node:IMASNode {id: b.properties.corrected_node})
OPTIONAL MATCH (gap:DDGap {id: b.properties.evidence})
OPTIONAL MATCH (version:DDVersion {id: b.properties.for_dd_version})
OPTIONAL MATCH (effective_unit:Unit {id: b.effective_graph_value})
CALL {
    WITH node
    OPTIONAL MATCH (node)-[:HAS_UNIT]->(unit:Unit)
    RETURN [value IN collect(DISTINCT unit.id) WHERE value IS NOT NULL] AS unit_ids
}
CALL {
    WITH node, b
    OPTIONAL MATCH (node)-[:BRIDGED_BY]->(claim:DDResolution)
    WHERE claim.dd_version = b.properties.dd_version
      AND claim.field = b.properties.field
    RETURN [value IN collect(DISTINCT claim.id) WHERE value IS NOT NULL] AS claim_ids
}
OPTIONAL MATCH (resolution:DDResolution {id: b.properties.id})
CALL {
    WITH resolution
    OPTIONAL MATCH (source:IMASNode)-[:BRIDGED_BY]->(resolution)
    RETURN [value IN collect(DISTINCT source.id) WHERE value IS NOT NULL]
           AS corrected_nodes
}
CALL {
    WITH resolution
    OPTIONAL MATCH (resolution)-[:EVIDENCED_BY]->(linked_gap:DDGap)
    RETURN [value IN collect(DISTINCT linked_gap.id) WHERE value IS NOT NULL]
           AS evidence
}
CALL {
    WITH resolution
    OPTIONAL MATCH (resolution)-[:FOR_DD_VERSION]->(linked_version:DDVersion)
    RETURN [value IN collect(DISTINCT linked_version.id) WHERE value IS NOT NULL]
           AS dd_versions
}
RETURN b.properties.id AS id,
       count(DISTINCT node) AS node_count,
       count(DISTINCT gap) AS gap_count,
       count(DISTINCT version) AS version_count,
       count(DISTINCT effective_unit) AS effective_unit_count,
       collect(DISTINCT gap.path) AS gap_paths,
       collect(DISTINCT gap.kind) AS gap_kinds,
       collect(DISTINCT gap.observed_dd_version) AS gap_versions,
       collect(DISTINCT gap.observed_value) AS gap_observed_values,
       collect(DISTINCT gap.expected_value) AS gap_expected_values,
       node.unit AS graph_value, unit_ids, claim_ids,
       CASE WHEN resolution IS NULL THEN null
            ELSE resolution{.*, recorded_at: toString(resolution.recorded_at)} END
            AS properties,
       corrected_nodes, evidence, dd_versions
ORDER BY id
"""

_GRAPH_PORT_CORRECT_QUERY = """
UNWIND $records AS b
MATCH (node:IMASNode {id: b.properties.corrected_node})
CALL {
    WITH node
    OPTIONAL MATCH (node)-[:HAS_UNIT]->(unit:Unit)
    RETURN [value IN collect(DISTINCT unit.id) WHERE value IS NOT NULL] AS unit_ids
}
WITH node, b, unit_ids
WHERE node.unit = b.published_graph_value
  AND unit_ids = [b.published_graph_value]
MATCH (effective_unit:Unit {id: b.effective_graph_value})
OPTIONAL MATCH (node)-[old_unit:HAS_UNIT]->(:Unit)
WITH node, b, effective_unit, collect(old_unit) AS old_units
FOREACH (edge IN old_units | DELETE edge)
SET node.unit = b.effective_graph_value
MERGE (node)-[:HAS_UNIT]->(effective_unit)
RETURN node.id AS path
ORDER BY path
"""

_GRAPH_PORT_WRITE_QUERY = """
UNWIND $records AS b
MATCH (node:IMASNode {id: b.properties.corrected_node})
MATCH (gap:DDGap {id: b.properties.evidence})
MATCH (version:DDVersion {id: b.properties.for_dd_version})
CREATE (resolution:DDResolution)
SET resolution = b.properties,
    resolution.recorded_at = datetime(b.properties.recorded_at)
CREATE (node)-[:BRIDGED_BY]->(resolution)
CREATE (resolution)-[:EVIDENCED_BY]->(gap)
CREATE (resolution)-[:FOR_DD_VERSION]->(version)
RETURN count(resolution) AS written
"""

_GRAPH_PORT_COUNTS_QUERY = """
UNWIND $ids AS id
MATCH (resolution:DDResolution {id: id})
CALL {
    WITH resolution
    OPTIONAL MATCH (:IMASNode)-[bridge:BRIDGED_BY]->(resolution)
    RETURN count(bridge) AS bridged
}
CALL {
    WITH resolution
    OPTIONAL MATCH (resolution)-[evidence:EVIDENCED_BY]->(:DDGap)
    RETURN count(evidence) AS evidenced
}
CALL {
    WITH resolution
    OPTIONAL MATCH (resolution)-[version:FOR_DD_VERSION]->(:DDVersion)
    RETURN count(version) AS versioned
}
RETURN count(resolution) AS nodes, sum(bridged) AS bridged_edges,
       sum(evidenced) AS evidenced_edges, sum(versioned) AS version_edges
"""


def _graph_port_record_matches(
    current: Mapping[str, Any], expected: Mapping[str, Any]
) -> bool:
    properties = current.get("properties")
    if not isinstance(properties, Mapping):
        return False
    normalized = dict(properties)
    if normalized.get("recorded_at") is not None:
        try:
            normalized["recorded_at"] = _canonical_datetime(
                datetime.fromisoformat(
                    str(normalized["recorded_at"]).replace("Z", "+00:00")
                )
            )
        except ValueError:
            return False
    return (
        all(normalized.get(key) == value for key, value in expected.items())
        and current.get("corrected_nodes") == [expected["corrected_node"]]
        and current.get("evidence") == [expected["evidence"]]
        and current.get("dd_versions") == [expected["for_dd_version"]]
    )


def _classify_graph_port_preflight(
    rows: Sequence[Mapping[str, Any]],
    expected_by_id: Mapping[str, Mapping[str, Any]],
) -> dict[str, DDResolutionGraphPathAction]:
    """Classify exact graph state or refuse all mismatching paths together."""
    actions: dict[str, DDResolutionGraphPathAction] = {}
    invalid: list[str] = []
    returned_ids: set[str] = set()
    for row in rows:
        resolution_id = str(row.get("id"))
        returned_ids.add(resolution_id)
        expected = expected_by_id.get(resolution_id)
        if expected is None:
            invalid.append(f"unknown resolution {resolution_id!r}")
            continue
        path = str(expected["path"])
        published = json.loads(str(expected["published_value"]))
        effective = json.loads(str(expected["effective_value"]))
        existing_matches = _graph_port_record_matches(row, expected)
        claim_ids = list(row.get("claim_ids") or [])
        cardinality_matches = (
            int(row.get("node_count") or 0) == 1
            and int(row.get("gap_count") or 0) == 1
            and int(row.get("version_count") or 0) == 1
            and int(row.get("effective_unit_count") or 0) == 1
            and row.get("gap_paths") == [path]
            and row.get("gap_kinds") == ["self_contradiction"]
            and row.get("gap_versions") == [expected["dd_version"]]
            and row.get("gap_observed_values") == [published]
            and row.get("gap_expected_values") == [effective]
            and all(claim == resolution_id for claim in claim_ids)
        )
        if not cardinality_matches:
            invalid.append(f"{path} (incomplete evidence or conflicting authority)")
            continue
        if row.get("properties") is not None and not existing_matches:
            invalid.append(f"{path} (resolution id already has different content)")
            continue
        graph_value = row.get("graph_value")
        unit_ids = list(row.get("unit_ids") or [])
        if existing_matches and graph_value == effective and unit_ids == [effective]:
            actions[resolution_id] = DDResolutionGraphPathAction.unchanged
        elif graph_value == effective and unit_ids == [effective]:
            actions[resolution_id] = DDResolutionGraphPathAction.attached
        elif (
            row.get("properties") is None
            and graph_value == published
            and unit_ids == [published]
        ):
            actions[resolution_id] = DDResolutionGraphPathAction.corrected
        else:
            invalid.append(
                f"{path} (published={published!r}, effective={effective!r}, "
                f"observed={graph_value!r}, HAS_UNIT={unit_ids!r})"
            )
    if len(rows) != len(expected_by_id):
        invalid.extend(
            f"{record['path']} (missing preflight row)"
            for resolution_id, record in expected_by_id.items()
            if resolution_id not in returned_ids
        )
    if invalid:
        raise DDResolutionGraphPortConflict(
            "resolution graph port preflight refused exact paths: " + "; ".join(invalid)
        )
    return actions


class _LiveDDResolutionGraphPort:
    """Neo4j transaction boundary for an additive exact resolution batch."""

    def apply(self, records: tuple[dict[str, Any], ...]) -> Mapping[str, Any]:
        from imas_codex.graph.client import GraphClient

        parameters = tuple(
            {
                "properties": record,
                "published_graph_value": json.loads(record["published_value"]),
                "effective_graph_value": json.loads(record["effective_value"]),
            }
            for record in records
        )
        expected_by_id = {record["id"]: record for record in records}
        with GraphClient() as graph, graph.session() as session:
            transaction = session.begin_transaction()
            try:
                rows = [
                    dict(row)
                    for row in transaction.run(
                        _GRAPH_PORT_PREFLIGHT_QUERY, records=parameters
                    )
                ]
                actions = _classify_graph_port_preflight(rows, expected_by_id)
                correction_records = tuple(
                    parameter
                    for parameter in parameters
                    if actions[parameter["properties"]["id"]]
                    == DDResolutionGraphPathAction.corrected
                )
                if correction_records:
                    corrected = {
                        str(row["path"])
                        for row in transaction.run(
                            _GRAPH_PORT_CORRECT_QUERY, records=correction_records
                        )
                    }
                    expected_paths = {
                        str(item["properties"]["path"]) for item in correction_records
                    }
                    if corrected != expected_paths:
                        raise DDResolutionGraphPortConflict(
                            "resolution graph correction compare-and-set failed for "
                            + ", ".join(sorted(expected_paths - corrected))
                        )
                new_records = tuple(
                    parameter
                    for parameter in parameters
                    if actions[parameter["properties"]["id"]]
                    != DDResolutionGraphPathAction.unchanged
                )
                if new_records:
                    transaction.run(
                        _GRAPH_PORT_WRITE_QUERY, records=new_records
                    ).consume()
                verified_rows = [
                    dict(row)
                    for row in transaction.run(
                        _GRAPH_PORT_PREFLIGHT_QUERY, records=parameters
                    )
                ]
                verified = _classify_graph_port_preflight(verified_rows, expected_by_id)
                if any(
                    action != DDResolutionGraphPathAction.unchanged
                    for action in verified.values()
                ):
                    raise DDResolutionGraphPortConflict(
                        "resolution graph port postcondition left an incomplete path"
                    )
                count_rows = [
                    dict(row)
                    for row in transaction.run(
                        _GRAPH_PORT_COUNTS_QUERY, ids=sorted(expected_by_id)
                    )
                ]
                if len(count_rows) != 1:
                    raise DDResolutionGraphPortConflict(
                        "resolution graph port count verification failed"
                    )
                metrics = count_rows[0]
                if any(
                    int(metrics.get(key) or 0) != len(records)
                    for key in (
                        "nodes",
                        "bridged_edges",
                        "evidenced_edges",
                        "version_edges",
                    )
                ):
                    raise DDResolutionGraphPortConflict(
                        "resolution graph port did not produce exact scoped counts"
                    )
                transaction.commit()
            except Exception:
                transaction.rollback()
                raise
        receipts = tuple(
            {
                "resolution_id": record["id"],
                "path": record["path"],
                "action": actions[record["id"]],
            }
            for record in records
        )
        writes = sum(
            action != DDResolutionGraphPathAction.unchanged
            for action in actions.values()
        )
        payload = {
            "expected": len(records),
            "writes": writes,
            "nodes": int(metrics["nodes"]),
            "bridged_edges": int(metrics["bridged_edges"]),
            "evidenced_edges": int(metrics["evidenced_edges"]),
            "version_edges": int(metrics["version_edges"]),
            "path_receipts": receipts,
            "replay": writes == 0,
        }
        return {**payload, "receipt_hash": _canonical_digest(payload)}


def port_dd_resolution_records(
    records: Sequence[Mapping[str, Any]],
    *,
    graph_port: DDResolutionGraphPort | None = None,
) -> DDResolutionGraphPortReceipt:
    """Atomically add exact resolution records and correct published graph values."""
    normalized = tuple(dict(record) for record in records)
    if not normalized:
        payload = {
            "expected": 0,
            "writes": 0,
            "nodes": 0,
            "bridged_edges": 0,
            "evidenced_edges": 0,
            "version_edges": 0,
            "path_receipts": (),
            "replay": True,
        }
        return DDResolutionGraphPortReceipt(
            **payload, receipt_hash=_canonical_digest(payload)
        )
    ids = [str(record.get("id")) for record in normalized]
    paths = [str(record.get("path")) for record in normalized]
    if len(set(ids)) != len(ids) or len(set(paths)) != len(paths):
        raise DDResolutionGraphPortConflict(
            "resolution graph batch contains duplicate ids or paths"
        )
    result = (graph_port or _LiveDDResolutionGraphPort()).apply(normalized)
    return DDResolutionGraphPortReceipt.model_validate(result)


class DDResolutionGraphReader(Protocol):
    """Typed boundary for a complete graph resolution snapshot."""

    def read_resolutions(self) -> Sequence[Mapping[str, Any]]: ...


_GRAPH_RESOLUTION_QUERY = """
MATCH (resolution:DDResolution)
CALL {
    WITH resolution
    OPTIONAL MATCH (source:IMASNode)-[:BRIDGED_BY]->(resolution)
    RETURN collect(DISTINCT source.id) AS source_paths
}
CALL {
    WITH resolution
    OPTIONAL MATCH (resolution)-[:EVIDENCED_BY]->(gap:DDGap)
    RETURN collect(DISTINCT gap.id) AS gap_ids
}
CALL {
    WITH resolution
    OPTIONAL MATCH (resolution)-[:FOR_DD_VERSION]->(version:DDVersion)
    RETURN collect(DISTINCT version.id) AS version_ids
}
RETURN resolution{.*, recorded_at: toString(resolution.recorded_at)} AS properties,
       source_paths, gap_ids, version_ids
ORDER BY resolution.id
"""


class _LiveDDResolutionGraphReader:
    def read_resolutions(self) -> Sequence[Mapping[str, Any]]:
        from imas_codex.graph.client import GraphClient

        with GraphClient() as graph:
            return [dict(row) for row in graph.query(_GRAPH_RESOLUTION_QUERY)]


def dd_resolution_graph_reader() -> DDResolutionGraphReader:
    """Return the live graph reader boundary."""
    return _LiveDDResolutionGraphReader()


def _decode_graph_value(kind: Any, encoded: Any) -> DDResolutionValue:
    try:
        value = json.loads(str(encoded))
        return DDResolutionValue(kind=str(kind), value=value)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise DDResolutionManifestInvalid(
            "graph resolution carries an invalid typed value"
        ) from exc


def _record_from_graph(row: Mapping[str, Any]) -> DDResolutionRecord:
    properties = row.get("properties")
    if not isinstance(properties, Mapping):
        raise DDResolutionManifestInvalid(
            "graph resolution row has no property mapping"
        )
    source_paths = tuple(str(item) for item in row.get("source_paths") or ())
    gap_ids = tuple(str(item) for item in row.get("gap_ids") or ())
    version_ids = tuple(str(item) for item in row.get("version_ids") or ())
    path = str(properties.get("path") or "")
    version = str(properties.get("dd_version") or "")
    if source_paths != (path,):
        raise DDResolutionEvidenceMismatch(
            f"resolution {properties.get('id')!r} must have one bridge from {path!r}"
        )
    if len(gap_ids) != 1:
        raise DDResolutionEvidenceMismatch(
            f"resolution {properties.get('id')!r} must have one EVIDENCED_BY edge"
        )
    if version_ids != (version,):
        raise DDResolutionEvidenceMismatch(
            f"resolution {properties.get('id')!r} must have one exact DD-version edge"
        )
    upstream = str(properties.get("upstream_reference") or "").strip()
    if not upstream:
        raise DDResolutionEvidenceMismatch(
            f"resolution {properties.get('id')!r} requires an upstream reference or {_NONE_YET!r}"
        )
    try:
        return DDResolutionRecord(
            id=properties["id"],
            gap_id=gap_ids[0],
            path=path,
            dd_version=version,
            field=properties["field"],
            observed=_decode_graph_value(
                properties["published_kind"], properties["published_value"]
            ),
            effective=_decode_graph_value(
                properties["effective_kind"], properties["effective_value"]
            ),
            reason=properties["reason"],
            recorded_by=properties["recorded_by"],
            recorded_at=properties["recorded_at"],
            upstream_reference=upstream,
            upstream_commit_reference=properties.get("upstream_commit_reference"),
            retiring_release=properties["retiring_release"],
            state=properties.get("status", DDResolutionStatus.active),
        )
    except (KeyError, ValueError) as exc:
        raise DDResolutionManifestInvalid(
            f"graph resolution {properties.get('id')!r} is incomplete: {exc}"
        ) from exc


def load_dd_resolution_manifest(
    *, graph_reader: DDResolutionGraphReader | None = None
) -> DDResolutionManifest:
    """Load the complete graph authority, refusing unavailable or empty snapshots."""
    try:
        rows = tuple((graph_reader or dd_resolution_graph_reader()).read_resolutions())
    except DDResolutionError:
        raise
    except Exception as exc:
        raise DDResolutionManifestInvalid(
            "cannot read DD resolution graph authority"
        ) from exc
    if not rows:
        raise DDResolutionManifestInvalid("DD resolution graph authority is empty")
    return DDResolutionManifest(
        resolutions=tuple(_record_from_graph(row) for row in rows)
    )


def expand_ionisation_potential_resolution_cohort(
    *,
    recorded_by: str,
    reason: str,
    recorded_at: datetime | None = None,
    manifest: DDResolutionManifest | None = None,
    graph_port: DDResolutionGraphPort | None = None,
) -> DDResolutionCohortReceipt:
    """Materialize missing exact evidence and resolution bridges for the cohort."""
    authority = manifest or load_dd_resolution_manifest()
    timestamp = recorded_at or datetime.now(UTC)
    records = ionisation_potential_resolution_records(
        manifest=authority,
        recorded_by=recorded_by,
        recorded_at=timestamp,
        reason=reason,
    )
    from imas_codex.standard_names.dd_gaps import write_dd_gaps

    reports = [
        {
            "path": record["path"],
            "source_path": record["path"],
            "kind": "self_contradiction",
            "reason": reason,
            "reporter": recorded_by,
            "observed_at": _canonical_datetime(timestamp),
            "observed_dd_version": record["dd_version"],
            "observed_value": json.loads(record["published_value"]),
            "expected_value": json.loads(record["effective_value"]),
            "evidence_rule": "unit_equals_expected",
        }
        for record in records
    ]
    evidence = write_dd_gaps(reports)
    graph = port_dd_resolution_records(records, graph_port=graph_port)
    return DDResolutionCohortReceipt(
        evidence=evidence,
        graph=graph,
        existing=len(IONISATION_POTENTIAL_RESOLUTION_PATHS) - len(records),
        added=len(records),
    )


def effective_active_dd_resolutions(
    manifest: DDResolutionManifest | None = None,
) -> tuple[DDResolutionRecord, ...]:
    """Return active records from one complete graph snapshot."""
    authority = manifest or load_dd_resolution_manifest()
    return tuple(
        record
        for record in authority.resolutions
        if record.state == DDResolutionStatus.active
    )


def active_dd_resolution(
    *,
    path: str,
    dd_version: str,
    field: DDResolutionField,
    manifest: DDResolutionManifest | None = None,
) -> DDResolutionRecord | None:
    """Return the single active record for an exact behavior key."""
    exact_path = _validate_exact_path(path)
    exact_version = _validate_exact_version(dd_version)
    matches = tuple(
        record
        for record in effective_active_dd_resolutions(manifest)
        if record.path == exact_path
        and record.dd_version == exact_version
        and record.field == field
    )
    if len(matches) > 1:
        raise DDResolutionCollision(
            f"multiple active resolutions claim {(exact_path, exact_version, field.value)!r}"
        )
    return matches[0] if matches else None


class RawDDContext(BaseModel):
    """Immutable exact-version DD declarations presented to the resolver."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    path: str
    dd_version: str
    unit: str | None = None
    documentation: str | None = None
    data_type: str | None = None
    node_type: str | None = None
    physics_domain: str | None = None
    cocos_transformation_type: str | None = None
    cocos_transformation_expression: str | None = None
    coordinates: tuple[str, ...] = ()
    lifecycle_status: str | None = None
    lifecycle_version: str | None = None
    parents: tuple[RawDDContext, ...] = ()
    members: tuple[RawDDContext, ...] = ()
    _exact_path = field_validator("path")(_validate_exact_path)
    _exact_version = field_validator("dd_version")(_validate_exact_version)


class ResolvedDDField(BaseModel):
    """One published/effective field pair with graph provenance."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    field: DDResolutionField
    raw: DDResolutionValue
    effective: DDResolutionValue
    applied: bool
    resolution_id: str | None = None
    gap_id: str | None = None
    state: DDResolutionStatus | None = None
    converged: bool = False
    provenance: DDResolutionRecord | None = None


class ResolvedDDContext(BaseModel):
    """Published provenance beside direct values from one graph snapshot."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    raw: RawDDContext
    graph: RawDDContext
    published: RawDDContext
    unit: str | None
    documentation: str | None
    data_type: str | None
    node_type: str | None
    physics_domain: str | None
    cocos_transformation_type: str | None
    cocos_transformation_expression: str | None
    coordinates: tuple[str, ...]
    lifecycle_status: str | None
    lifecycle_version: str | None
    resolved_fields: tuple[ResolvedDDField, ...]
    applied_resolution_ids: tuple[str, ...]
    converged_resolution_ids: tuple[str, ...]
    resolution_provenance: tuple[DDResolutionRecord, ...]
    manifest_digest: str
    parents: tuple[ResolvedDDContext, ...] = ()
    members: tuple[ResolvedDDContext, ...] = ()

    def as_pipeline_item(self) -> dict[str, Any]:
        return {
            "path": self.raw.path,
            "dd_version": self.raw.dd_version,
            "unit": self.unit,
            "documentation": self.documentation,
            "data_type": self.data_type,
            "node_type": self.node_type,
            "physics_domain": self.physics_domain,
            "cocos_transformation_type": self.cocos_transformation_type,
            "cocos_transformation_expression": self.cocos_transformation_expression,
            "coordinates": list(self.coordinates),
            "lifecycle_status": self.lifecycle_status,
            "lifecycle_version": self.lifecycle_version,
            "raw_dd_context": self.raw.model_dump(mode="json"),
            "published_dd_context": self.published.model_dump(mode="json"),
            "dd_resolution_ids": list(self.applied_resolution_ids),
            "dd_resolution_converged_ids": list(self.converged_resolution_ids),
            "dd_resolution_manifest_digest": self.manifest_digest,
            "_dd_resolution_marker": _RESOLUTION_MARKER,
        }


def _same_value(left: DDResolutionValue, right: DDResolutionValue) -> bool:
    return _canonical_json(left) == _canonical_json(right)


def _resolved_field(
    field: DDResolutionField,
    raw: DDResolutionValue,
    effective: DDResolutionValue,
    record: DDResolutionRecord | None = None,
    *,
    applied: bool = False,
    converged: bool = False,
) -> ResolvedDDField:
    return ResolvedDDField(
        field=field,
        raw=raw,
        effective=effective,
        applied=applied,
        resolution_id=record.id if record else None,
        gap_id=record.gap_id if record else None,
        state=record.state if record else None,
        converged=converged,
        provenance=record,
    )


def resolve_dd_field(
    *,
    path: str,
    dd_version: str,
    field: DDResolutionField,
    raw_value: DDResolutionValue,
    manifest: DDResolutionManifest | None = None,
) -> ResolvedDDField:
    """Resolve one exact field without mutating its raw provenance."""
    exact_path = _validate_exact_path(path)
    exact_version = _validate_exact_version(dd_version)
    authority = manifest or load_dd_resolution_manifest()
    active = tuple(
        record
        for record in effective_active_dd_resolutions(authority)
        if record.path == exact_path and record.field == field
    )
    exact = tuple(record for record in active if record.dd_version == exact_version)
    if len(exact) > 1:
        raise DDResolutionCollision(
            f"multiple active resolutions claim {(exact_path, exact_version, field.value)!r}"
        )
    if exact:
        record = exact[0]
        if _same_value(raw_value, record.observed):
            return _resolved_field(
                field, raw_value, record.effective, record, applied=True
            )
        if _same_value(raw_value, record.effective):
            return _resolved_field(field, raw_value, raw_value, record, converged=True)
        raise DDResolutionStale(
            f"raw value for {(exact_path, exact_version, field.value)!r} matches neither recorded side"
        )
    if active:
        converged = tuple(
            record for record in active if _same_value(raw_value, record.effective)
        )
        if len(converged) > 1:
            raise DDResolutionAmbiguity(
                "multiple prior-version records could certify convergence"
            )
        if converged:
            return _resolved_field(
                field, raw_value, raw_value, converged[0], converged=True
            )
        versions = sorted({record.dd_version for record in active})
        raise DDResolutionVersionMismatch(
            f"resolution for {(exact_path, field.value)!r} was reviewed only for {versions!r}, not {exact_version!r}"
        )
    return _resolved_field(field, raw_value, raw_value)


def _read_graph_field(
    *,
    path: str,
    dd_version: str,
    field: DDResolutionField,
    graph_value: DDResolutionValue,
    manifest: DDResolutionManifest,
) -> ResolvedDDField:
    """Attach bridge provenance without replacing the graph field value."""
    active = tuple(
        record
        for record in effective_active_dd_resolutions(manifest)
        if record.path == path and record.field == field
    )
    exact = tuple(record for record in active if record.dd_version == dd_version)
    if len(exact) > 1:
        raise DDResolutionCollision(
            f"multiple active resolutions claim {(path, dd_version, field.value)!r}"
        )
    if exact:
        record = exact[0]
        if not _same_value(graph_value, record.effective):
            raise DDResolutionStale(
                f"graph value for {(path, dd_version, field.value)!r} does not "
                "match the resolution's effective value"
            )
        return _resolved_field(
            field,
            record.observed,
            graph_value,
            record,
            applied=True,
        )
    if active:
        converged = tuple(
            record for record in active if _same_value(graph_value, record.effective)
        )
        if len(converged) > 1:
            raise DDResolutionAmbiguity(
                "multiple prior-version records could certify convergence"
            )
        if converged:
            return _resolved_field(
                field,
                converged[0].observed,
                graph_value,
                converged[0],
                converged=True,
            )
        versions = sorted({record.dd_version for record in active})
        raise DDResolutionVersionMismatch(
            f"resolution for {(path, field.value)!r} was reviewed only for "
            f"{versions!r}, not {dd_version!r}"
        )
    return _resolved_field(field, graph_value, graph_value)


def read_graph_dd_field(
    *,
    path: str,
    dd_version: str,
    field: DDResolutionField,
    graph_value: DDResolutionValue,
    manifest: DDResolutionManifest | None = None,
) -> ResolvedDDField:
    """Validate one direct graph field and attach bridge provenance."""
    return _read_graph_field(
        path=_validate_exact_path(path),
        dd_version=_validate_exact_version(dd_version),
        field=field,
        graph_value=graph_value,
        manifest=manifest or load_dd_resolution_manifest(),
    )


_CONTEXT_FIELDS = tuple(DDResolutionField)


def _context_value(raw: RawDDContext, field: DDResolutionField) -> DDResolutionValue:
    value = getattr(raw, field.value)
    if field == DDResolutionField.coordinates:
        return DDResolutionValue(kind=DDResolutionValueKind.string_list, value=value)
    kind = DDResolutionValueKind.null if value is None else DDResolutionValueKind.string
    return DDResolutionValue(kind=kind, value=value)


def resolve_dd_context(
    raw: RawDDContext, *, manifest: DDResolutionManifest | None = None
) -> ResolvedDDContext:
    """Read graph values directly and attach their resolution provenance."""
    authority = manifest or load_dd_resolution_manifest()
    fields = tuple(
        _read_graph_field(
            path=raw.path,
            dd_version=raw.dd_version,
            field=field,
            graph_value=_context_value(raw, field),
            manifest=authority,
        )
        for field in _CONTEXT_FIELDS
    )
    values = {item.field: item.effective.value for item in fields}
    coordinates = values[DDResolutionField.coordinates]
    if not isinstance(coordinates, tuple):
        raise DDResolutionEvidenceMismatch(
            "coordinates must remain an ordered string list"
        )
    published = raw.model_copy(
        update={item.field.value: item.raw.value for item in fields}
    )
    provenance = tuple(
        sorted(
            (item.provenance for item in fields if item.provenance is not None),
            key=lambda record: record.id,
        )
    )
    return ResolvedDDContext(
        raw=raw,
        graph=raw,
        published=published,
        unit=values[DDResolutionField.unit],
        documentation=values[DDResolutionField.documentation],
        data_type=values[DDResolutionField.data_type],
        node_type=values[DDResolutionField.node_type],
        physics_domain=values[DDResolutionField.physics_domain],
        cocos_transformation_type=values[DDResolutionField.cocos_transformation_type],
        cocos_transformation_expression=values[
            DDResolutionField.cocos_transformation_expression
        ],
        coordinates=coordinates,
        lifecycle_status=values[DDResolutionField.lifecycle_status],
        lifecycle_version=values[DDResolutionField.lifecycle_version],
        resolved_fields=fields,
        applied_resolution_ids=tuple(
            sorted(item.resolution_id for item in fields if item.applied)
        ),
        converged_resolution_ids=tuple(
            sorted(item.resolution_id for item in fields if item.converged)
        ),
        resolution_provenance=provenance,
        manifest_digest=authority.digest,
        parents=tuple(
            resolve_dd_context(item, manifest=authority) for item in raw.parents
        ),
        members=tuple(
            resolve_dd_context(item, manifest=authority) for item in raw.members
        ),
    )


def resolve_dd_row(
    row: Mapping[str, Any],
    *,
    dd_version: str,
    manifest: DDResolutionManifest | None = None,
) -> ResolvedDDContext:
    """Resolve one raw graph projection while retaining exact provenance."""
    version = _validate_exact_version(dd_version)
    if "path" not in row:
        raise ValueError("DD row is missing exact path")
    return resolve_dd_context(
        RawDDContext(
            path=row["path"],
            dd_version=version,
            unit=row.get("unit"),
            documentation=row.get("documentation"),
            data_type=row.get("data_type"),
            node_type=row.get("node_type"),
            physics_domain=row.get("physics_domain"),
            cocos_transformation_type=row.get("cocos_transformation_type"),
            cocos_transformation_expression=row.get("cocos_transformation_expression"),
            coordinates=tuple(row.get("coordinates") or ()),
            lifecycle_status=row.get("lifecycle_status"),
            lifecycle_version=row.get("lifecycle_version"),
        ),
        manifest=manifest,
    )


def resolve_dd_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    dd_version: str,
    manifest: DDResolutionManifest | None = None,
) -> list[ResolvedDDContext]:
    """Resolve an all-or-error batch under one exact DD version."""
    authority = manifest or load_dd_resolution_manifest()
    return [
        resolve_dd_row(row, dd_version=dd_version, manifest=authority) for row in rows
    ]


DDResolutionState = DDResolutionStatus

__all__ = [
    "DDResolutionAmbiguity",
    "DDResolutionCollision",
    "DDResolutionError",
    "DDResolutionEvidenceMismatch",
    "DDResolutionField",
    "DDResolutionGraphReader",
    "DDResolutionManifest",
    "DDResolutionManifestInvalid",
    "DDResolutionRecord",
    "DDResolutionStale",
    "DDResolutionState",
    "DDResolutionStatus",
    "DDResolutionValue",
    "DDResolutionVersionMismatch",
    "RawDDContext",
    "ResolvedDDContext",
    "ResolvedDDField",
    "active_dd_resolution",
    "dd_resolution_graph_reader",
    "effective_active_dd_resolutions",
    "load_dd_resolution_manifest",
    "read_graph_dd_field",
    "resolve_dd_context",
    "resolve_dd_field",
    "resolve_dd_row",
    "resolve_dd_rows",
]
