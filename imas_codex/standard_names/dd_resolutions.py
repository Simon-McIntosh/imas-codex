"""Reviewed, version-bound interpretations of DD declarations.

The packaged resources remain the migration input until their later retirement.
The graph port materializes active records beside corrected DD values as
first-class provenance bridges. A resolution applies only to one exact
IDS-prefixed path, one exact published DD version, and one typed field. Any
ambiguity, staleness, or attempted cross-version reuse fails closed.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from enum import Enum, StrEnum
from functools import lru_cache
from importlib import resources
from pathlib import Path
from tempfile import NamedTemporaryFile, gettempdir
from typing import Any, Protocol, Self
from urllib.parse import urlparse

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    StrictInt,
    ValidationError,
    field_validator,
    model_validator,
)

from imas_codex.core.physics_domain import PhysicsDomain
from imas_codex.graph.models import (
    COCOSLabelTransformation,
    DDDataType,
    DDGapKind,
    DDNodeType,
    DDResolutionField,
    DDResolutionStatus,
    DDResolutionValueKind,
    LifecycleStatus,
)

_MANIFEST_RESOURCE = "dd_resolutions.yaml"
_MANIFEST_SCHEMA_VERSION = 1
_CANDIDATE_RESOURCE = "dd_resolution_candidates.yaml"
_CANDIDATE_SCHEMA_VERSION = 1
_RESOLUTION_MARKER = "resolved-dd-context"
_EXACT_VERSION_RE = re.compile(
    r"[0-9]+\.[0-9]+\.[0-9]+(?:[-+][0-9A-Za-z][0-9A-Za-z.-]*)?\Z"
)
_DIGEST_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
_RESOLUTION_ID_RE = re.compile(r"dd_resolution:[0-9a-f]{64}\Z")
_STATE_CHANGE_ID_RE = re.compile(r"dd-resolution-state-change:sha256:[0-9a-f]{64}\Z")
_OBSERVATION_ID_RE = re.compile(r"dd_gap_observation:[0-9a-f]{64}\Z")
_EVIDENCE_TOKEN_RE = re.compile(r"dd-gap-evidence:[0-9a-f]{64}\Z")
_UPSTREAM_COMMIT_RE = re.compile(r"[0-9a-f]{40}\Z")
_SOURCE_ROW_RE = re.compile(r"[UO][0-9]{2}\Z")
_UPSTREAM_CHANGE_KEY_RE = re.compile(r"[a-z][a-z0-9]*(?:-[a-z0-9]+)*\Z")
_PATTERN_CHARACTERS = frozenset("*?[]{}\\^$|")
_NONE_YET = "none-yet"

_CANDIDATE_MISSING_REQUIREMENTS = frozenset(
    {
        "approval_receipt",
        "approved_at",
        "approved_by",
        "fresh_evidence_token",
        "governed_decision_reason",
        "resolution_revision",
        "review_decision",
    }
)


class DDResolutionError(RuntimeError):
    """Base class for fail-closed DD resolution errors."""


class DDResolutionManifestInvalid(DDResolutionError):
    """The packaged manifest is absent, unreadable, or structurally invalid."""


class DDResolutionCollision(DDResolutionError):
    """Multiple manifest records claim one behavior or evidence identity."""


class DDResolutionVersionMismatch(DDResolutionError):
    """A reviewed resolution exists for this path and field, but not this version."""


class DDResolutionStale(DDResolutionError):
    """The exact-version raw value matches neither reviewed side of a resolution."""


class DDResolutionEvidenceMismatch(DDResolutionError):
    """Approval, DDGap evidence, or upstream provenance is incomplete or inconsistent."""


class DDResolutionAmbiguity(DDResolutionError):
    """A deterministic resolution receipt cannot be selected."""


class DDResolutionManifestConflict(DDResolutionError):
    """The tracked authority changed after the operator reviewed it."""


class DDResolutionGraphEvidenceMismatch(DDResolutionError):
    """Current graph evidence does not match the reviewed approval input."""


class DDResolutionGraphPortConflict(DDResolutionError):
    """The graph cannot accept the complete resolution port atomically."""


class DDResolutionGraphPathAction(StrEnum):
    """Mutation performed for one path by the graph port."""

    corrected = "corrected"
    attached = "attached"
    unchanged = "unchanged"


class DDResolutionGraphPathReceipt(BaseModel):
    """Deterministic disposition for one active resolution path."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    resolution_id: str
    path: str
    published_value: str
    effective_value: str
    action: DDResolutionGraphPathAction


class DDResolutionGraphPortReceipt(BaseModel):
    """Deterministic receipt for one complete active-resolution graph port."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    manifest_digest: str
    expected: int
    writes: int
    nodes: int
    bridged_edges: int
    evidenced_edges: int
    version_edges: int
    corrected: int
    attached: int
    unchanged: int
    path_receipts: tuple[DDResolutionGraphPathReceipt, ...]
    replay: bool
    receipt_hash: str


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
    RETURN [value IN collect(DISTINCT unit.id) WHERE value IS NOT NULL]
           AS unit_ids
}
RETURN b.properties.id AS id,
       count(DISTINCT node) AS node_count,
       count(DISTINCT gap) AS gap_count,
       count(DISTINCT version) AS version_count,
       count(DISTINCT effective_unit) AS effective_unit_count,
       collect(DISTINCT gap.path) AS gap_paths,
       node.unit AS graph_value,
       unit_ids
ORDER BY id
"""

_GRAPH_PORT_CORRECT_QUERY = """
UNWIND $records AS b
MATCH (node:IMASNode {id: b.properties.corrected_node})
CALL {
    WITH node
    OPTIONAL MATCH (node)-[:HAS_UNIT]->(unit:Unit)
    RETURN [value IN collect(DISTINCT unit.id) WHERE value IS NOT NULL]
           AS unit_ids
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

_GRAPH_PORT_CURRENT_QUERY = """
UNWIND $records AS b
OPTIONAL MATCH (resolution:DDResolution {id: b.id})
CALL {
    WITH resolution
    OPTIONAL MATCH (source:IMASNode)-[:BRIDGED_BY]->(resolution)
    RETURN [value IN collect(DISTINCT source.id) WHERE value IS NOT NULL]
           AS corrected_nodes
}
CALL {
    WITH resolution
    OPTIONAL MATCH (resolution)-[:EVIDENCED_BY]->(gap:DDGap)
    RETURN [value IN collect(DISTINCT gap.id) WHERE value IS NOT NULL] AS evidence
}
CALL {
    WITH resolution
    OPTIONAL MATCH (resolution)-[:FOR_DD_VERSION]->(version:DDVersion)
    RETURN [value IN collect(DISTINCT version.id) WHERE value IS NOT NULL]
           AS dd_versions
}
RETURN b.id AS id,
       CASE WHEN resolution IS NULL THEN null
            ELSE resolution{.*, recorded_at: toString(resolution.recorded_at)} END
            AS properties,
       corrected_nodes, evidence, dd_versions
ORDER BY id
"""

_GRAPH_PORT_UNKNOWN_QUERY = """
MATCH (resolution:DDResolution)
WHERE NOT (resolution.id IN $ids)
RETURN resolution.id AS id
ORDER BY id
"""

_GRAPH_PORT_CLEAR_QUERY = """
UNWIND $ids AS id
MATCH (resolution:DDResolution {id: id})
OPTIONAL MATCH (resolution)-[legacy:HAS_RESOLUTION|SUPPORTED_BY_OBSERVATION|
    HAS_STATE_CHANGE|BRIDGED_BY|EVIDENCED_BY|FOR_DD_VERSION]-()
DELETE legacy
"""

_GRAPH_PORT_WRITE_QUERY = """
UNWIND $records AS b
MATCH (node:IMASNode {id: b.properties.corrected_node})
MATCH (gap:DDGap {id: b.properties.evidence})
MATCH (version:DDVersion {id: b.properties.for_dd_version})
MERGE (resolution:DDResolution {id: b.properties.id})
SET resolution = b.properties,
    resolution.recorded_at = datetime(b.properties.recorded_at)
MERGE (node)-[:BRIDGED_BY]->(resolution)
MERGE (resolution)-[:EVIDENCED_BY]->(gap)
MERGE (resolution)-[:FOR_DD_VERSION]->(version)
RETURN count(DISTINCT resolution) AS written
"""

_GRAPH_PORT_COUNTS_QUERY = """
CALL {
    MATCH (resolution:DDResolution)
    RETURN count(resolution) AS nodes
}
CALL {
    MATCH (:IMASNode)-[bridge:BRIDGED_BY]->(:DDResolution)
    RETURN count(bridge) AS bridged_edges
}
CALL {
    MATCH (:DDResolution)-[evidence:EVIDENCED_BY]->(:DDGap)
    RETURN count(evidence) AS evidenced_edges
}
CALL {
    MATCH (:DDResolution)-[version:FOR_DD_VERSION]->(:DDVersion)
    RETURN count(version) AS version_edges
}
RETURN nodes, bridged_edges, evidenced_edges, version_edges
"""


def _graph_port_rows(result: Any) -> list[dict[str, Any]]:
    return [dict(row) for row in result]


def _graph_port_record_matches(current: Mapping[str, Any], expected: dict) -> bool:
    current_properties = current.get("properties")
    if not isinstance(current_properties, Mapping):
        return False
    normalized_properties = dict(current_properties)
    recorded_at = normalized_properties.get("recorded_at")
    if recorded_at is not None:
        try:
            normalized_properties["recorded_at"] = _canonical_datetime(str(recorded_at))
        except ValueError:
            return False
    return (
        normalized_properties == expected
        and current.get("corrected_nodes") == [expected["corrected_node"]]
        and current.get("evidence") == [expected["evidence"]]
        and current.get("dd_versions") == [expected["for_dd_version"]]
    )


def _classify_graph_port_preflight(
    rows: Sequence[Mapping[str, Any]],
    expected_by_id: Mapping[str, Mapping[str, Any]],
) -> dict[str, DDResolutionGraphPathAction]:
    """Classify exact graph state or refuse every mismatching path together."""
    actions: dict[str, DDResolutionGraphPathAction] = {}
    invalid: list[str] = []
    for row in rows:
        resolution_id = str(row.get("id"))
        expected = expected_by_id.get(resolution_id)
        if expected is None:
            invalid.append(f"unknown resolution {resolution_id!r}")
            continue
        path = str(expected["path"])
        published = json.loads(str(expected["published_value"]))
        effective = json.loads(str(expected["effective_value"]))
        graph_value = row.get("graph_value")
        unit_ids = list(row.get("unit_ids") or [])
        cardinality_matches = (
            int(row.get("node_count") or 0) == 1
            and int(row.get("gap_count") or 0) == 1
            and int(row.get("version_count") or 0) == 1
            and int(row.get("effective_unit_count") or 0) == 1
            and row.get("gap_paths") == [path]
        )
        if not cardinality_matches:
            invalid.append(
                f"{path} (node={row.get('node_count')!r}, "
                f"gap={row.get('gap_count')!r}, "
                f"version={row.get('version_count')!r}, "
                f"effective_unit={row.get('effective_unit_count')!r})"
            )
        elif graph_value == effective and unit_ids == [effective]:
            actions[resolution_id] = DDResolutionGraphPathAction.attached
        elif graph_value == published and unit_ids == [published]:
            actions[resolution_id] = DDResolutionGraphPathAction.corrected
        else:
            invalid.append(
                f"{path} (published={published!r}, effective={effective!r}, "
                f"observed={graph_value!r}, HAS_UNIT={unit_ids!r})"
            )
    if len(rows) != len(expected_by_id):
        returned_ids = {str(row.get("id")) for row in rows}
        invalid.extend(
            f"{expected['path']} (missing preflight row)"
            for resolution_id, expected in expected_by_id.items()
            if resolution_id not in returned_ids
        )
    if invalid:
        raise DDResolutionGraphPortConflict(
            "resolution graph port preflight refused exact paths: " + "; ".join(invalid)
        )
    return actions


def _require_complete_graph_correction(
    records: Sequence[Mapping[str, Any]], corrected_paths: Sequence[str]
) -> None:
    """Refuse and roll back when the write-time compare-and-set loses a row."""
    corrected = set(corrected_paths)
    missing = [
        str(record["path"]) for record in records if record["path"] not in corrected
    ]
    if missing or len(corrected) != len(records):
        raise DDResolutionGraphPortConflict(
            "resolution graph correction compare-and-set failed for exact paths: "
            + ", ".join(missing)
        )


class _LiveDDResolutionGraphPort:
    """Neo4j transaction boundary for the complete graph port."""

    def apply(self, records: tuple[dict[str, Any], ...]) -> Mapping[str, Any]:
        from imas_codex.graph.client import GraphClient

        parameters = [
            {
                "properties": record,
                "published_graph_value": json.loads(record["published_value"]),
                "effective_graph_value": json.loads(record["effective_value"]),
            }
            for record in records
        ]
        expected_by_id = {record["id"]: record for record in records}
        with GraphClient() as graph, graph.session() as session:
            transaction = session.begin_transaction()
            try:
                preflight = _graph_port_rows(
                    transaction.run(_GRAPH_PORT_PREFLIGHT_QUERY, records=parameters)
                )
                actions = _classify_graph_port_preflight(preflight, expected_by_id)

                unknown = [
                    str(row["id"])
                    for row in transaction.run(
                        _GRAPH_PORT_UNKNOWN_QUERY,
                        ids=sorted(expected_by_id),
                    )
                ]
                if unknown:
                    raise DDResolutionGraphPortConflict(
                        "graph contains DDResolution nodes outside the active port: "
                        + ", ".join(unknown)
                    )

                current = {
                    str(row["id"]): row
                    for row in _graph_port_rows(
                        transaction.run(_GRAPH_PORT_CURRENT_QUERY, records=records)
                    )
                }
                changed = tuple(
                    record
                    for record in records
                    if not _graph_port_record_matches(
                        current.get(record["id"], {}), record
                    )
                )
                correction_records = tuple(
                    parameter
                    for parameter in parameters
                    if actions[parameter["properties"]["id"]]
                    == DDResolutionGraphPathAction.corrected
                )
                if correction_records:
                    corrected_rows = _graph_port_rows(
                        transaction.run(
                            _GRAPH_PORT_CORRECT_QUERY,
                            records=correction_records,
                        )
                    )
                    _require_complete_graph_correction(
                        [item["properties"] for item in correction_records],
                        [str(row["path"]) for row in corrected_rows],
                    )
                if changed:
                    transaction.run(
                        _GRAPH_PORT_CLEAR_QUERY,
                        ids=[record["id"] for record in changed],
                    ).consume()
                    transaction.run(
                        _GRAPH_PORT_WRITE_QUERY,
                        records=[
                            {
                                "properties": record,
                                "effective_graph_value": json.loads(
                                    record["effective_value"]
                                ),
                            }
                            for record in changed
                        ],
                    ).consume()

                verified = {
                    str(row["id"]): row
                    for row in _graph_port_rows(
                        transaction.run(_GRAPH_PORT_CURRENT_QUERY, records=records)
                    )
                }
                mismatched = [
                    record["id"]
                    for record in records
                    if not _graph_port_record_matches(
                        verified.get(record["id"], {}), record
                    )
                ]
                postflight = _graph_port_rows(
                    transaction.run(_GRAPH_PORT_PREFLIGHT_QUERY, records=parameters)
                )
                post_actions = _classify_graph_port_preflight(
                    postflight, expected_by_id
                )
                not_effective = [
                    expected_by_id[resolution_id]["path"]
                    for resolution_id, action in post_actions.items()
                    if action == DDResolutionGraphPathAction.corrected
                ]
                if not_effective:
                    raise DDResolutionGraphPortConflict(
                        "resolution graph port left published graph values on paths: "
                        + ", ".join(not_effective)
                    )
                count_rows = _graph_port_rows(transaction.run(_GRAPH_PORT_COUNTS_QUERY))
                if mismatched or len(count_rows) != 1:
                    raise DDResolutionGraphPortConflict(
                        "resolution graph port postcondition failed for records: "
                        + ", ".join(mismatched)
                    )
                metrics = count_rows[0]
                expected_count = len(records)
                if any(
                    int(metrics.get(key) or 0) != expected_count
                    for key in (
                        "nodes",
                        "bridged_edges",
                        "evidenced_edges",
                        "version_edges",
                    )
                ):
                    raise DDResolutionGraphPortConflict(
                        "resolution graph port did not produce exact node and edge counts"
                    )
                transaction.commit()
            except Exception:
                transaction.rollback()
                raise
        changed_ids = {record["id"] for record in changed}
        corrected_ids = {item["properties"]["id"] for item in correction_records}
        return {
            "writes": len(changed_ids | corrected_ids),
            "nodes": int(metrics["nodes"]),
            "bridged_edges": int(metrics["bridged_edges"]),
            "evidenced_edges": int(metrics["evidenced_edges"]),
            "version_edges": int(metrics["version_edges"]),
            "path_receipts": [
                {
                    "resolution_id": record["id"],
                    "path": record["path"],
                    "published_value": record["published_value"],
                    "effective_value": record["effective_value"],
                    "action": (
                        DDResolutionGraphPathAction.corrected
                        if record["id"] in corrected_ids
                        else DDResolutionGraphPathAction.attached
                        if record["id"] in changed_ids
                        else DDResolutionGraphPathAction.unchanged
                    ),
                }
                for record in records
            ],
        }


class DDResolutionGraphReader(Protocol):
    """Typed read boundary for one exact current DDGap snapshot."""

    def get_gap(self, gap_id: str) -> Mapping[str, Any] | None: ...


class _LiveDDResolutionGraphReader:
    def get_gap(self, gap_id: str) -> Mapping[str, Any] | None:
        from imas_codex.standard_names.dd_gaps import get_dd_gap

        return get_dd_gap(gap_id)


def dd_resolution_graph_reader() -> DDResolutionGraphReader:
    """Return the production exact-snapshot reader."""
    return _LiveDDResolutionGraphReader()


class DDResolutionCandidateDisposition(StrEnum):
    """Review routing for evidence that has no behavior authority."""

    bounded_review_input = "bounded_review_input"
    broad_scope_hold = "broad_scope_hold"


class DDResolutionUpstreamStatus(StrEnum):
    """Official change state observed for review-only provenance."""

    open = "open"
    merged = "merged"


class _UniqueKeySafeLoader(yaml.SafeLoader):
    """Safe YAML loader that refuses duplicate mapping keys recursively."""


def _construct_unique_mapping(
    loader: _UniqueKeySafeLoader,
    node: yaml.MappingNode,
    deep: bool = False,
) -> dict[Any, Any]:
    loader.flatten_mapping(node)
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in mapping
        except TypeError as exc:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                "found an unhashable key",
                key_node.start_mark,
            ) from exc
        if duplicate:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"found duplicate key {key!r}",
                key_node.start_mark,
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeySafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def _enum_text(value: Any) -> str:
    return str(value.value if isinstance(value, Enum) else value)


def _canonical_datetime(value: datetime | str) -> str:
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError(f"invalid offset-aware timestamp {value!r}") from exc
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamp must include a UTC offset")
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _jsonable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return _jsonable(value.model_dump(mode="json"))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime):
        return _canonical_datetime(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_jsonable(item) for item in value]
    return value


def _canonical_json(value: Any) -> str:
    return json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"))


def _canonical_digest(value: Any) -> str:
    return f"sha256:{hashlib.sha256(_canonical_json(value).encode()).hexdigest()}"


def _record_identity_payload(record: Mapping[str, Any] | BaseModel) -> dict[str, Any]:
    raw = (
        record.model_dump(mode="json")
        if isinstance(record, BaseModel)
        else dict(record)
    )
    raw.pop("id", None)
    if "approved_at" in raw:
        raw["approved_at"] = _canonical_datetime(raw["approved_at"])
    if "observation_ids" in raw:
        raw["observation_ids"] = sorted(raw["observation_ids"])
    return _jsonable(raw)


def dd_resolution_value_hash(value: DDResolutionValue | Mapping[str, Any]) -> str:
    """Return the canonical hash stored beside one typed DD value."""
    validated = (
        value
        if isinstance(value, DDResolutionValue)
        else DDResolutionValue.model_validate(value)
    )
    return _canonical_digest(validated)


def content_addressed_resolution_id(record: Mapping[str, Any] | BaseModel) -> str:
    """Return the required identity for a canonical resolution record payload."""
    digest = _canonical_digest(_record_identity_payload(record)).removeprefix("sha256:")
    return f"dd_resolution:{digest}"


def _validate_exact_path(path: str) -> str:
    clean = path.strip()
    if not clean or clean.startswith("/") or clean.endswith("/") or "//" in clean:
        raise ValueError("DD resolution path must be a nonempty relative exact path")
    if "/" not in clean:
        raise ValueError(
            "DD resolution path must be IDS-prefixed, not an isolated leaf"
        )
    if any(character in clean for character in _PATTERN_CHARACTERS):
        raise ValueError("DD resolution path must be exact; patterns are forbidden")
    if any(segment in {".", ".."} for segment in clean.split("/")):
        raise ValueError("DD resolution path cannot contain relative aliases")
    return clean


def _validate_exact_version(version: str) -> str:
    clean = version.strip()
    if not _EXACT_VERSION_RE.fullmatch(clean):
        raise ValueError(
            "DD resolution version must be one exact published version "
            "(for example '4.1.0')"
        )
    return clean


class DDResolutionValue(BaseModel):
    """One canonical typed raw or effective DD field value."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: DDResolutionValueKind
    value: str | tuple[str, ...] | None

    @model_validator(mode="after")
    def _validate_kind(self) -> Self:
        kind = _enum_text(self.kind)
        if kind == "string" and not isinstance(self.value, str):
            raise ValueError("kind='string' requires one string value")
        if kind == "string_list":
            if not isinstance(self.value, tuple) or any(
                not isinstance(item, str) for item in self.value
            ):
                raise ValueError("kind='string_list' requires a list of strings")
        if kind == "null" and self.value is not None:
            raise ValueError("kind='null' requires value=null")
        return self


_FIELD_GAP_KINDS = {
    DDResolutionField.unit: frozenset(
        {DDGapKind.unit_defect, DDGapKind.self_contradiction}
    ),
    DDResolutionField.documentation: frozenset(
        {
            DDGapKind.doc_mismatch,
            DDGapKind.self_contradiction,
            DDGapKind.missing_declaration,
        }
    ),
    DDResolutionField.data_type: frozenset(
        {
            DDGapKind.type_wiring,
            DDGapKind.self_contradiction,
            DDGapKind.missing_declaration,
        }
    ),
    DDResolutionField.node_type: frozenset(
        {
            DDGapKind.type_wiring,
            DDGapKind.self_contradiction,
            DDGapKind.missing_declaration,
        }
    ),
    DDResolutionField.physics_domain: frozenset(
        {
            DDGapKind.doc_mismatch,
            DDGapKind.type_wiring,
            DDGapKind.self_contradiction,
            DDGapKind.missing_declaration,
        }
    ),
    DDResolutionField.cocos_transformation_type: frozenset(
        {
            DDGapKind.type_wiring,
            DDGapKind.self_contradiction,
            DDGapKind.missing_declaration,
        }
    ),
    DDResolutionField.cocos_transformation_expression: frozenset(
        {
            DDGapKind.type_wiring,
            DDGapKind.self_contradiction,
            DDGapKind.missing_declaration,
        }
    ),
    DDResolutionField.coordinates: frozenset(
        {
            DDGapKind.type_wiring,
            DDGapKind.self_contradiction,
            DDGapKind.missing_declaration,
        }
    ),
    DDResolutionField.lifecycle_status: frozenset(
        {
            DDGapKind.doc_mismatch,
            DDGapKind.self_contradiction,
            DDGapKind.missing_declaration,
        }
    ),
    DDResolutionField.lifecycle_version: frozenset(
        {
            DDGapKind.doc_mismatch,
            DDGapKind.self_contradiction,
            DDGapKind.missing_declaration,
        }
    ),
}

_FIELD_VALUE_ENUMS: dict[DDResolutionField, type[Enum]] = {
    DDResolutionField.data_type: DDDataType,
    DDResolutionField.node_type: DDNodeType,
    DDResolutionField.physics_domain: PhysicsDomain,
    DDResolutionField.cocos_transformation_type: COCOSLabelTransformation,
    DDResolutionField.lifecycle_status: LifecycleStatus,
}


def _validate_field_value(
    field: DDResolutionField,
    value: DDResolutionValue,
) -> None:
    if value.kind == DDResolutionValueKind.string_list:
        if any(
            not item.strip()
            or any(character in item for character in _PATTERN_CHARACTERS)
            for item in value.value
        ):
            raise ValueError(
                "coordinate identities must be nonempty exact strings, not patterns"
            )
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
    """One exact, reviewed, content-addressed local DD interpretation."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    id: str
    gap_id: str
    path: str
    dd_version: str
    field: DDResolutionField
    observed: DDResolutionValue
    observed_hash: str
    effective: DDResolutionValue
    resolution_revision: int
    reason: str
    observation_ids: tuple[str, ...]
    evidence_token: str
    approved_by: str
    approved_at: datetime
    approval_receipt: str
    upstream_url: str
    upstream_ref: str
    state: DDResolutionStatus

    _exact_path = field_validator("path")(_validate_exact_path)
    _exact_version = field_validator("dd_version")(_validate_exact_version)

    @field_validator(
        "reason",
        "approved_by",
        "approval_receipt",
        "upstream_ref",
    )
    @classmethod
    def _required_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("review and provenance text fields cannot be empty")
        return value

    @field_validator("approved_at")
    @classmethod
    def _offset_aware_approval(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("approved_at must include a UTC offset")
        return value

    @field_validator("resolution_revision")
    @classmethod
    def _positive_revision(cls, value: int) -> int:
        if value < 1:
            raise ValueError("resolution_revision must be positive")
        return value

    @field_validator("observation_ids", mode="before")
    @classmethod
    def _canonical_observation_ids(cls, value: Any) -> tuple[str, ...]:
        if isinstance(value, str | bytes) or not isinstance(value, Sequence):
            raise ValueError("observation_ids must be a nonempty sequence")
        items = tuple(str(item).strip() for item in value)
        if not items or any(not item for item in items):
            raise ValueError("observation_ids must contain nonempty identities")
        if len(items) != len(set(items)):
            raise DDResolutionCollision("duplicate observation identity in one record")
        return tuple(sorted(items))

    @model_validator(mode="after")
    def _validate_contract(self) -> Self:
        allowed_kinds = (
            {DDResolutionValueKind.string_list}
            if self.field == DDResolutionField.coordinates
            else {DDResolutionValueKind.string, DDResolutionValueKind.null}
        )
        if (
            self.observed.kind not in allowed_kinds
            or self.effective.kind not in allowed_kinds
        ):
            raise ValueError(
                f"field={self.field.value!r} requires value kinds "
                f"{sorted(item.value for item in allowed_kinds)!r}"
            )
        _validate_field_value(self.field, self.observed)
        _validate_field_value(self.field, self.effective)
        expected_observed_hash = dd_resolution_value_hash(self.observed)
        if self.observed_hash != expected_observed_hash:
            raise DDResolutionEvidenceMismatch(
                "observed_hash does not match the canonical observed typed value"
            )
        if self.state == DDResolutionStatus.active and self.observed == self.effective:
            raise DDResolutionEvidenceMismatch(
                "an active resolution must change the reviewed field value"
            )
        _verify_record_provenance(self)
        expected_id = content_addressed_resolution_id(self)
        if not _RESOLUTION_ID_RE.fullmatch(self.id) or self.id != expected_id:
            raise DDResolutionEvidenceMismatch(
                f"resolution id is not the content address {expected_id!r}"
            )
        return self


class DDResolutionStateChangeReceipt(BaseModel):
    """Immutable local receipt preserving one authority-state transition."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    id: str
    from_resolution_id: str
    to_resolution_id: str
    from_status: DDResolutionStatus
    to_status: DDResolutionStatus
    actor: str
    reason: str
    changed_at: datetime

    @field_validator("actor", "reason")
    @classmethod
    def _required_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("state-change authority and reason cannot be empty")
        return value

    @field_validator("changed_at")
    @classmethod
    def _offset_aware_change(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("changed_at must include a UTC offset")
        return value

    @model_validator(mode="after")
    def _validate_receipt(self) -> Self:
        if not _RESOLUTION_ID_RE.fullmatch(
            self.from_resolution_id
        ) or not _RESOLUTION_ID_RE.fullmatch(self.to_resolution_id):
            raise DDResolutionEvidenceMismatch(
                "state-change receipt must bind exact resolution identities"
            )
        if self.from_resolution_id == self.to_resolution_id:
            raise DDResolutionEvidenceMismatch(
                "state-change receipt must bind distinct immutable records"
            )
        if self.from_status == self.to_status:
            raise DDResolutionEvidenceMismatch(
                "state-change receipt must change durable status"
            )
        payload = self.model_dump(mode="json", exclude={"id"})
        expected = f"dd-resolution-state-change:{_canonical_digest(payload)}"
        if not _STATE_CHANGE_ID_RE.fullmatch(self.id) or self.id != expected:
            raise DDResolutionEvidenceMismatch(
                f"state-change id is not the content address {expected!r}"
            )
        return self


class DDResolutionManifest(BaseModel):
    """Complete validated package behavior authority."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int
    resolutions: tuple[DDResolutionRecord, ...]
    state_changes: tuple[DDResolutionStateChangeReceipt, ...] = ()

    @model_validator(mode="after")
    def _validate_manifest(self) -> Self:
        if self.schema_version != _MANIFEST_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported DD resolution schema_version {self.schema_version!r}"
            )
        seen_ids: set[str] = set()
        records_by_id: dict[str, DDResolutionRecord] = {}
        for record in self.resolutions:
            if record.id in seen_ids:
                raise DDResolutionCollision(f"duplicate resolution id {record.id!r}")
            seen_ids.add(record.id)
            records_by_id[record.id] = record

        seen_changes: set[str] = set()
        transitioned_ids: set[str] = set()
        for receipt in self.state_changes:
            if receipt.id in seen_changes:
                raise DDResolutionCollision(
                    f"duplicate resolution state-change id {receipt.id!r}"
                )
            seen_changes.add(receipt.id)
            if receipt.from_resolution_id in transitioned_ids:
                raise DDResolutionCollision(
                    f"resolution {receipt.from_resolution_id!r} has multiple transitions"
                )
            transitioned_ids.add(receipt.from_resolution_id)
            before = records_by_id.get(receipt.from_resolution_id)
            after = records_by_id.get(receipt.to_resolution_id)
            if before is None or after is None:
                raise DDResolutionEvidenceMismatch(
                    "state-change receipt references resolution history absent from "
                    "the manifest"
                )
            if before.state != receipt.from_status or after.state != receipt.to_status:
                raise DDResolutionEvidenceMismatch(
                    "state-change receipt statuses disagree with immutable records"
                )
            before_key = (before.path, before.dd_version, before.field)
            after_key = (after.path, after.dd_version, after.field)
            if before_key != after_key:
                raise DDResolutionEvidenceMismatch(
                    "state-change receipt crosses exact resolution keys"
                )

        revoked_ids = {
            receipt.from_resolution_id
            for receipt in self.state_changes
            if receipt.to_status == DDResolutionStatus.withdrawn
        }
        active_keys: dict[tuple[str, str, DDResolutionField], str] = {}
        observation_owners: dict[str, tuple[str, str, DDResolutionField]] = {}
        evidence_owners: dict[str, tuple[str, str, DDResolutionField]] = {}
        lifecycle_fields: dict[tuple[str, str], set[DDResolutionField]] = {}
        for record in self.resolutions:
            if record.state != DDResolutionStatus.active or record.id in revoked_ids:
                continue
            key = (record.path, record.dd_version, record.field)
            if key in active_keys:
                raise DDResolutionCollision(
                    f"multiple active DD resolutions claim exact key {key!r}"
                )
            active_keys[key] = record.id
            for observation_id in record.observation_ids:
                owner = observation_owners.setdefault(observation_id, key)
                if owner != key:
                    raise DDResolutionCollision(
                        f"observation {observation_id!r} is reused by {owner!r} and {key!r}"
                    )
            evidence_owner = evidence_owners.setdefault(record.evidence_token, key)
            if evidence_owner != key:
                raise DDResolutionCollision(
                    f"evidence token {record.evidence_token!r} has multiple owners"
                )
            if record.field in {
                DDResolutionField.lifecycle_status,
                DDResolutionField.lifecycle_version,
            }:
                lifecycle_fields.setdefault(
                    (record.path, record.dd_version), set()
                ).add(record.field)
        for key, fields in lifecycle_fields.items():
            if fields != {
                DDResolutionField.lifecycle_status,
                DDResolutionField.lifecycle_version,
            }:
                raise DDResolutionCollision(
                    f"lifecycle resolution {key!r} must review status and version together"
                )
        return self

    @property
    def digest(self) -> str:
        """Canonical digest of the complete packaged authority."""
        payload = {
            "schema_version": self.schema_version,
            "resolutions": [
                _jsonable(record)
                for record in sorted(self.resolutions, key=lambda item: item.id)
            ],
        }
        if self.state_changes:
            payload["state_changes"] = [
                _jsonable(receipt)
                for receipt in sorted(self.state_changes, key=lambda item: item.id)
            ]
        return _canonical_digest(payload)


class DDResolutionCandidateUpstreamChange(BaseModel):
    """Exact official change provenance without local review authority."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    change_url: str
    issue_url: str | None = None
    solution_commits: tuple[str, ...]
    status: DDResolutionUpstreamStatus
    merge_commit: str | None = None
    affected_since_dd_version: str | None = None
    proposed_change_dd_version: str | None = None
    fixed_dd_version: str | None = None

    @field_validator("change_url", "issue_url")
    @classmethod
    def _official_https_url(cls, value: str | None) -> str | None:
        if value is None:
            return None
        parsed = urlparse(value)
        if (
            parsed.scheme != "https"
            or parsed.netloc != "github.com"
            or parsed.username
            or parsed.password
            or not parsed.path.startswith("/iterorganization/IMAS-Data-Dictionary/")
        ):
            raise ValueError(
                "candidate provenance URLs must be credential-free official HTTPS URLs"
            )
        return value

    @field_validator("solution_commits", mode="before")
    @classmethod
    def _exact_solution_commits(cls, value: Any) -> tuple[str, ...]:
        if isinstance(value, str | bytes) or not isinstance(value, Sequence):
            raise ValueError("solution_commits must be a nonempty sequence")
        commits = tuple(str(item).strip() for item in value)
        if not commits or any(
            not _UPSTREAM_COMMIT_RE.fullmatch(item) for item in commits
        ):
            raise ValueError("every solution commit must be one full lowercase SHA")
        if len(commits) != len(set(commits)):
            raise ValueError("solution commits must be unique")
        return commits

    @field_validator("merge_commit")
    @classmethod
    def _exact_merge_commit(cls, value: str | None) -> str | None:
        if value is not None and not _UPSTREAM_COMMIT_RE.fullmatch(value):
            raise ValueError("merge_commit must be one full lowercase SHA")
        return value

    @field_validator(
        "affected_since_dd_version",
        "proposed_change_dd_version",
        "fixed_dd_version",
    )
    @classmethod
    def _exact_optional_version(cls, value: str | None) -> str | None:
        return _validate_exact_version(value) if value is not None else None

    @model_validator(mode="after")
    def _preserve_change_state(self) -> Self:
        if self.status == DDResolutionUpstreamStatus.merged and not self.merge_commit:
            raise ValueError("merged upstream provenance requires the merge commit")
        if self.status == DDResolutionUpstreamStatus.open and self.merge_commit:
            raise ValueError("an open upstream change cannot carry a merge commit")
        if self.status == DDResolutionUpstreamStatus.open and self.fixed_dd_version:
            raise ValueError("an open upstream change cannot claim a fixed DD release")
        return self


class DDResolutionCandidate(BaseModel):
    """One strictly non-runtime row mapping retained for governed review."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    source_row: str
    source_pattern: str
    dd_version: str
    field: DDResolutionField
    observed: DDResolutionValue
    proposed_effective: DDResolutionValue
    disposition: DDResolutionCandidateDisposition
    source_release_match_count: StrictInt
    exact_paths: tuple[str, ...]
    narrow_evidence_overlap_count: StrictInt | None = None
    upstream_change: str

    _exact_version = field_validator("dd_version")(_validate_exact_version)

    @field_validator("source_row")
    @classmethod
    def _source_row_identity(cls, value: str) -> str:
        if not _SOURCE_ROW_RE.fullmatch(value):
            raise ValueError("source_row must identify one audited legacy registry row")
        return value

    @field_validator("source_pattern")
    @classmethod
    def _source_pattern_present(cls, value: str) -> str:
        if not value:
            raise ValueError(
                "source_pattern must preserve the audited registry pattern"
            )
        return value

    @field_validator("exact_paths", mode="before")
    @classmethod
    def _canonical_exact_paths(cls, value: Any) -> tuple[str, ...]:
        if isinstance(value, str | bytes) or not isinstance(value, Sequence):
            raise ValueError("exact_paths must be a sequence")
        paths = tuple(_validate_exact_path(str(item)) for item in value)
        if len(paths) != len(set(paths)):
            raise ValueError("candidate exact paths must be unique")
        if paths != tuple(sorted(paths)):
            raise ValueError("candidate exact paths must be sorted")
        return paths

    @field_validator("upstream_change")
    @classmethod
    def _upstream_change_key(cls, value: str) -> str:
        if not _UPSTREAM_CHANGE_KEY_RE.fullmatch(value):
            raise ValueError("upstream_change must be a stable mechanism key")
        return value

    @model_validator(mode="after")
    def _validate_review_boundary(self) -> Self:
        if self.source_release_match_count < 1:
            raise ValueError("candidate source release match count must be positive")
        if self.observed == self.proposed_effective:
            raise ValueError("candidate input must preserve a proposed value change")
        allowed_kinds = (
            {DDResolutionValueKind.string_list}
            if self.field == DDResolutionField.coordinates
            else {DDResolutionValueKind.string, DDResolutionValueKind.null}
        )
        if (
            self.observed.kind not in allowed_kinds
            or self.proposed_effective.kind not in allowed_kinds
        ):
            raise ValueError(
                f"field={self.field.value!r} requires value kinds "
                f"{sorted(item.value for item in allowed_kinds)!r}"
            )
        _validate_field_value(self.field, self.observed)
        _validate_field_value(self.field, self.proposed_effective)
        if self.disposition == DDResolutionCandidateDisposition.bounded_review_input:
            if not self.exact_paths:
                raise ValueError("bounded review input requires exact paths")
            if len(self.exact_paths) > self.source_release_match_count:
                raise ValueError("bounded paths exceed the audited release cohort")
            if self.narrow_evidence_overlap_count is not None:
                raise ValueError(
                    "bounded review input cannot carry a broad overlap count"
                )
        else:
            if self.exact_paths:
                raise ValueError("broad scope holds cannot enumerate candidate paths")
            if self.narrow_evidence_overlap_count is None:
                raise ValueError(
                    "broad scope holds require the narrow evidence overlap"
                )
            if (
                not 0
                <= self.narrow_evidence_overlap_count
                < self.source_release_match_count
            ):
                raise ValueError(
                    "broad scope hold overlap must be smaller than its release cohort"
                )
        return self


class DDResolutionCandidateManifest(BaseModel):
    """Complete typed review input, deliberately separate from behavior authority."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    schema_version: StrictInt
    authority: str
    missing_requirements: tuple[str, ...]
    upstream_changes: dict[str, DDResolutionCandidateUpstreamChange]
    candidates: tuple[DDResolutionCandidate, ...]

    @field_validator("missing_requirements", mode="before")
    @classmethod
    def _canonical_missing_requirements(cls, value: Any) -> tuple[str, ...]:
        if isinstance(value, str | bytes) or not isinstance(value, Sequence):
            raise ValueError("missing_requirements must be a sequence")
        requirements = tuple(str(item).strip() for item in value)
        if len(requirements) != len(set(requirements)):
            raise ValueError("missing requirements must be unique")
        return tuple(sorted(requirements))

    @model_validator(mode="after")
    def _validate_review_manifest(self) -> Self:
        if self.schema_version != _CANDIDATE_SCHEMA_VERSION:
            raise ValueError(
                "unsupported DD resolution candidate schema_version "
                f"{self.schema_version!r}"
            )
        if self.authority != "review_input_only":
            raise ValueError("candidate resource authority must be review_input_only")
        if frozenset(self.missing_requirements) != _CANDIDATE_MISSING_REQUIREMENTS:
            raise ValueError(
                "candidate resource must enumerate every missing activation requirement"
            )
        if not self.candidates:
            raise ValueError("candidate resource must contain review input")
        source_rows = [candidate.source_row for candidate in self.candidates]
        if len(source_rows) != len(set(source_rows)):
            raise ValueError("candidate source rows must be unique")
        change_keys = set(self.upstream_changes)
        if any(not _UPSTREAM_CHANGE_KEY_RE.fullmatch(key) for key in change_keys):
            raise ValueError("upstream change keys must be stable mechanism keys")
        used_keys = {candidate.upstream_change for candidate in self.candidates}
        if used_keys != change_keys:
            raise ValueError("every declared upstream change must be used and defined")
        return self

    @property
    def digest(self) -> str:
        """Canonical digest of review input; never an active-manifest digest."""
        return _canonical_digest(self)


def _verify_record_provenance(record: DDResolutionRecord) -> None:
    prefix = f"dd_gap:{record.path}:"
    if not record.gap_id.startswith(prefix):
        raise DDResolutionEvidenceMismatch(
            "gap_id does not identify the resolution's exact DD path"
        )
    kind_text = record.gap_id.removeprefix(prefix)
    try:
        kind = DDGapKind(kind_text)
    except ValueError as exc:
        raise DDResolutionEvidenceMismatch(
            f"gap_id carries unknown DDGap kind {kind_text!r}"
        ) from exc
    allowed = _FIELD_GAP_KINDS.get(record.field, frozenset())
    if kind not in allowed:
        raise DDResolutionEvidenceMismatch(
            f"DDGap kind {kind.value!r} does not support field {record.field.value!r}"
        )
    if not _DIGEST_RE.fullmatch(record.observed_hash):
        raise DDResolutionEvidenceMismatch("observed_hash must be a canonical SHA-256")
    if not all(_OBSERVATION_ID_RE.fullmatch(item) for item in record.observation_ids):
        raise DDResolutionEvidenceMismatch(
            "every observation_id must be a content-addressed DDGap observation"
        )
    if not _EVIDENCE_TOKEN_RE.fullmatch(record.evidence_token):
        raise DDResolutionEvidenceMismatch(
            "evidence_token must identify one exact reviewed DDGap evidence set"
        )
    parsed = urlparse(record.upstream_url)
    if (
        parsed.scheme != "https"
        or not parsed.netloc
        or parsed.username
        or parsed.password
    ):
        raise DDResolutionEvidenceMismatch(
            "upstream_url must be an absolute credential-free HTTPS URL"
        )
    if not record.approved_by or not record.approval_receipt or not record.upstream_ref:
        raise DDResolutionEvidenceMismatch(
            "approval and exact upstream provenance are required"
        )


class RawDDContext(BaseModel):
    """Immutable exact-version raw DD declarations presented to the resolver."""

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
    """One raw/effective field pair plus its complete resolution receipt."""

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

    @model_validator(mode="after")
    def _validate_receipt(self) -> Self:
        if self.applied and self.converged:
            raise ValueError("a converged resolution is not an applied override")
        if self.applied or self.converged:
            if not self.resolution_id or not self.gap_id or self.provenance is None:
                raise ValueError(
                    "applied and converged fields require complete provenance"
                )
        elif any((self.resolution_id, self.gap_id, self.state, self.provenance)):
            raise ValueError("pass-through fields cannot claim resolution provenance")
        return self


class ResolvedDDContext(BaseModel):
    """Raw provenance and effective projections under one manifest digest."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    raw: RawDDContext
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
    manifest_digest: str
    parents: tuple[ResolvedDDContext, ...] = ()
    members: tuple[ResolvedDDContext, ...] = ()

    def as_pipeline_item(self) -> dict[str, Any]:
        """Project this context to a marked dictionary for legacy pipeline seams."""
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
            "dd_resolution_ids": list(self.applied_resolution_ids),
            "dd_resolution_converged_ids": list(self.converged_resolution_ids),
            "dd_resolution_manifest_digest": self.manifest_digest,
            "_dd_resolution_marker": _RESOLUTION_MARKER,
        }


@lru_cache(maxsize=8)
def _parse_manifest_content(content: str) -> DDResolutionManifest:
    try:
        document = yaml.safe_load(content)
    except yaml.YAMLError as exc:
        raise DDResolutionManifestInvalid(
            "DD resolution manifest is not valid YAML"
        ) from exc
    if not isinstance(document, dict):
        raise DDResolutionManifestInvalid(
            "DD resolution manifest top level must be a mapping"
        )
    try:
        return DDResolutionManifest.model_validate(document)
    except ValidationError as exc:
        raise DDResolutionManifestInvalid(
            f"DD resolution manifest is not schema-compliant: {exc}"
        ) from exc


def load_dd_resolution_manifest() -> DDResolutionManifest:
    """Load and validate the packaged manifest, cached by its exact content."""
    reference = dd_resolution_manifest_path()
    try:
        content = reference.read_text(encoding="utf-8")
    except (FileNotFoundError, OSError) as exc:
        raise DDResolutionManifestInvalid(
            f"packaged DD resolution manifest {_MANIFEST_RESOURCE!r} is unavailable"
        ) from exc
    return _parse_manifest_content(content)


def dd_resolution_manifest_path() -> Path:
    """Return the tracked manifest path used by the human lifecycle CLI."""
    reference = resources.files("imas_codex.standard_names.config").joinpath(
        _MANIFEST_RESOURCE
    )
    return Path(str(reference))


@lru_cache(maxsize=8)
def _parse_candidate_content(content: str) -> DDResolutionCandidateManifest:
    try:
        document = yaml.load(content, Loader=_UniqueKeySafeLoader)
    except yaml.YAMLError as exc:
        raise DDResolutionManifestInvalid(
            f"DD resolution candidate resource is not valid YAML: {exc}"
        ) from exc
    if not isinstance(document, dict):
        raise DDResolutionManifestInvalid(
            "DD resolution candidate top level must be a mapping"
        )
    try:
        return DDResolutionCandidateManifest.model_validate(document)
    except ValidationError as exc:
        raise DDResolutionManifestInvalid(
            f"DD resolution candidate resource is not schema-compliant: {exc}"
        ) from exc


def load_dd_resolution_candidates_for_review() -> DDResolutionCandidateManifest:
    """Load non-authoritative provenance for a governed review workflow."""
    reference = resources.files("imas_codex.standard_names.config").joinpath(
        _CANDIDATE_RESOURCE
    )
    try:
        content = reference.read_text(encoding="utf-8")
    except (FileNotFoundError, OSError) as exc:
        raise DDResolutionManifestInvalid(
            f"packaged DD resolution candidate resource {_CANDIDATE_RESOURCE!r} "
            "is unavailable"
        ) from exc
    return _parse_candidate_content(content)


def _load_manifest_from_path(path: Path) -> DDResolutionManifest:
    try:
        content = path.read_text(encoding="utf-8")
    except (FileNotFoundError, OSError) as exc:
        raise DDResolutionManifestInvalid(
            f"DD resolution manifest {str(path)!r} is unavailable"
        ) from exc
    return _parse_manifest_content(content)


def _manifest_lock_path(path: Path) -> Path:
    digest = hashlib.sha256(str(path.resolve()).encode()).hexdigest()
    return Path(gettempdir()) / f"imas-codex-dd-resolution-{digest}.lock"


@contextmanager
def _locked_manifest(
    path: Path,
    *,
    expected_digest: str,
):
    lock_path = _manifest_lock_path(path)
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            manifest = _load_manifest_from_path(path)
            if manifest.digest != expected_digest:
                raise DDResolutionManifestConflict(
                    "DD resolution manifest changed since review: "
                    f"expected {expected_digest!r}, found {manifest.digest!r}"
                )
            yield manifest
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _write_manifest(
    path: Path,
    manifest: DDResolutionManifest,
    *,
    expected_digest: str,
) -> None:
    current = _load_manifest_from_path(path)
    if current.digest != expected_digest:
        raise DDResolutionManifestConflict(
            "DD resolution manifest changed during guarded mutation: "
            f"expected {expected_digest!r}, found {current.digest!r}"
        )
    document: dict[str, Any] = {
        "schema_version": manifest.schema_version,
        "resolutions": [
            record.model_dump(mode="json") for record in manifest.resolutions
        ],
    }
    if manifest.state_changes:
        document["state_changes"] = [
            receipt.model_dump(mode="json") for receipt in manifest.state_changes
        ]
    content = yaml.safe_dump(document, sort_keys=False, allow_unicode=True)
    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
            temporary_path = Path(handle.name)
        os.replace(temporary_path, path)
    except OSError as exc:
        raise DDResolutionManifestInvalid(
            f"cannot atomically write DD resolution manifest {str(path)!r}: {exc}"
        ) from exc
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _effective_active_records(
    manifest: DDResolutionManifest,
) -> tuple[DDResolutionRecord, ...]:
    withdrawn = {
        receipt.from_resolution_id
        for receipt in manifest.state_changes
        if receipt.to_status == DDResolutionStatus.withdrawn
    }
    return tuple(
        record
        for record in manifest.resolutions
        if record.state == DDResolutionStatus.active and record.id not in withdrawn
    )


def effective_active_dd_resolutions(
    manifest: DDResolutionManifest | None = None,
) -> tuple[DDResolutionRecord, ...]:
    """Return the effective active records from a strict authority manifest."""
    return _effective_active_records(manifest or load_dd_resolution_manifest())


def _graph_port_retiring_release(
    record: DDResolutionRecord,
    review_input: DDResolutionCandidateManifest,
) -> str:
    receipt_parts = record.approval_receipt.split(":", 2)
    source_row = receipt_parts[1] if len(receipt_parts) == 3 else ""
    candidates = [
        candidate
        for candidate in review_input.candidates
        if candidate.source_row == source_row
        and record.path in candidate.exact_paths
        and record.dd_version == candidate.dd_version
        and record.field == candidate.field
        and record.observed == candidate.observed
        and record.effective == candidate.proposed_effective
    ]
    if len(candidates) != 1:
        raise DDResolutionGraphPortConflict(
            f"resolution {record.id!r} has {len(candidates)} exact candidate sources"
        )
    upstream = review_input.upstream_changes.get(candidates[0].upstream_change)
    if upstream is None:
        raise DDResolutionGraphPortConflict(
            f"resolution {record.id!r} has no upstream disposition"
        )
    if record.upstream_url and record.upstream_url != upstream.change_url:
        raise DDResolutionGraphPortConflict(
            f"resolution {record.id!r} disagrees with its upstream source"
        )
    return upstream.fixed_dd_version or upstream.proposed_change_dd_version or _NONE_YET


def _graph_port_records(
    manifest: DDResolutionManifest,
    review_input: DDResolutionCandidateManifest,
) -> tuple[dict[str, Any], ...]:
    records: list[dict[str, Any]] = []
    for record in effective_active_dd_resolutions(manifest):
        upstream_reference = record.upstream_url or _NONE_YET
        upstream_commit_reference = record.upstream_ref or _NONE_YET
        if not upstream_reference or not upstream_commit_reference:
            raise DDResolutionGraphPortConflict(
                f"resolution {record.id!r} has neither upstream provenance nor "
                "the explicit none-yet marker"
            )
        records.append(
            {
                "id": record.id,
                "path": record.path,
                "field": record.field.value,
                "published_kind": record.observed.kind.value,
                "published_value": _canonical_json(record.observed.value),
                "effective_kind": record.effective.kind.value,
                "effective_value": _canonical_json(record.effective.value),
                "dd_version": record.dd_version,
                "upstream_reference": upstream_reference,
                "upstream_commit_reference": upstream_commit_reference,
                "retiring_release": _graph_port_retiring_release(record, review_input),
                "recorded_by": record.approved_by,
                "recorded_at": _canonical_datetime(record.approved_at),
                "reason": record.reason,
                "source_manifest_digest": manifest.digest,
                "status": record.state.value,
                "corrected_node": record.path,
                "evidence": record.gap_id,
                "for_dd_version": record.dd_version,
            }
        )
    return tuple(sorted(records, key=lambda item: item["id"]))


def port_active_dd_resolutions_to_graph(
    *,
    graph_port: DDResolutionGraphPort | None = None,
) -> DDResolutionGraphPortReceipt:
    """Materialize every active packaged resolution as one graph bridge."""
    manifest = load_dd_resolution_manifest()
    review_input = load_dd_resolution_candidates_for_review()
    records = _graph_port_records(manifest, review_input)
    if not records:
        raise DDResolutionGraphPortConflict(
            "active DD resolution authority is empty; refusing a vacuous graph port"
        )
    metrics = dict((graph_port or _LiveDDResolutionGraphPort()).apply(records))
    expected = len(records)
    required_metrics = (
        "writes",
        "nodes",
        "bridged_edges",
        "evidenced_edges",
        "version_edges",
        "path_receipts",
    )
    if any(key not in metrics for key in required_metrics):
        raise DDResolutionGraphPortConflict(
            "graph port did not return complete node and edge metrics"
        )
    if any(
        int(metrics[key]) != expected
        for key in ("nodes", "bridged_edges", "evidenced_edges", "version_edges")
    ):
        raise DDResolutionGraphPortConflict(
            "graph port metrics do not match the complete active authority"
        )
    writes = int(metrics["writes"])
    if writes < 0 or writes > expected:
        raise DDResolutionGraphPortConflict(
            "graph port returned an invalid write count"
        )
    raw_path_receipts = metrics["path_receipts"]
    if not isinstance(raw_path_receipts, Sequence) or isinstance(
        raw_path_receipts, str | bytes
    ):
        raise DDResolutionGraphPortConflict(
            "graph port returned invalid per-path receipts"
        )
    try:
        path_receipts = tuple(
            DDResolutionGraphPathReceipt.model_validate(item)
            for item in raw_path_receipts
        )
    except ValidationError as exc:
        raise DDResolutionGraphPortConflict(
            "graph port returned invalid per-path receipts"
        ) from exc
    records_by_id = {record["id"]: record for record in records}
    receipts_by_id = {receipt.resolution_id: receipt for receipt in path_receipts}
    if (
        len(path_receipts) != expected
        or len(receipts_by_id) != expected
        or set(receipts_by_id) != set(records_by_id)
        or any(
            receipt.path != records_by_id[resolution_id]["path"]
            or receipt.published_value
            != records_by_id[resolution_id]["published_value"]
            or receipt.effective_value
            != records_by_id[resolution_id]["effective_value"]
            for resolution_id, receipt in receipts_by_id.items()
        )
    ):
        raise DDResolutionGraphPortConflict(
            "graph port per-path receipts do not match the active authority"
        )
    corrected = sum(
        receipt.action == DDResolutionGraphPathAction.corrected
        for receipt in path_receipts
    )
    attached = sum(
        receipt.action == DDResolutionGraphPathAction.attached
        for receipt in path_receipts
    )
    unchanged = sum(
        receipt.action == DDResolutionGraphPathAction.unchanged
        for receipt in path_receipts
    )
    if corrected + attached + unchanged != expected:
        raise DDResolutionGraphPortConflict(
            "graph port per-path actions are incomplete"
        )
    if (writes == 0) != (unchanged == expected):
        raise DDResolutionGraphPortConflict(
            "graph port replay state disagrees with per-path actions"
        )
    receipt_payload = {
        "manifest_digest": manifest.digest,
        "expected": expected,
        "writes": writes,
        "nodes": int(metrics["nodes"]),
        "bridged_edges": int(metrics["bridged_edges"]),
        "evidenced_edges": int(metrics["evidenced_edges"]),
        "version_edges": int(metrics["version_edges"]),
        "corrected": corrected,
        "attached": attached,
        "unchanged": unchanged,
        "path_receipts": [receipt.model_dump(mode="json") for receipt in path_receipts],
        "replay": writes == 0,
    }
    return DDResolutionGraphPortReceipt(
        **receipt_payload,
        receipt_hash=_canonical_digest(receipt_payload),
    )


def active_dd_resolution(
    *,
    path: str,
    dd_version: str,
    field: DDResolutionField,
    manifest: DDResolutionManifest | None = None,
) -> DDResolutionRecord | None:
    """Return the single effective active record for an exact authority key."""
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
            f"multiple active resolutions claim "
            f"{(exact_path, exact_version, field.value)!r}"
        )
    return matches[0] if matches else None


def approved_candidate_paths(
    candidate: DDResolutionCandidate,
    manifest: DDResolutionManifest,
    *,
    candidate_digest: str,
) -> tuple[str, ...]:
    """Return exact candidate paths carrying effective active authority."""
    receipt_prefix = (
        f"dd-resolution-approval:{candidate.source_row}:{candidate_digest}:"
    )
    keys = {
        (record.path, record.dd_version, record.field)
        for record in _effective_active_records(manifest)
        if record.approval_receipt.startswith(receipt_prefix)
    }
    return tuple(
        path
        for path in candidate.exact_paths
        if (path, candidate.dd_version, candidate.field) in keys
    )


def _verify_graph_evidence(
    snapshot: Mapping[str, Any] | None,
    *,
    gap_id: str,
    candidate: DDResolutionCandidate,
    expected_observation_ids: Sequence[str],
    expected_evidence_token: str,
) -> tuple[str, tuple[str, ...]]:
    from imas_codex.standard_names.dd_gaps import _evidence_token

    if snapshot is None:
        raise DDResolutionGraphEvidenceMismatch(
            f"current exact graph DDGap {gap_id!r} was not found"
        )
    exact_path = gap_id.removeprefix("dd_gap:").rsplit(":", 1)[0]
    exact_kind = gap_id.rsplit(":", 1)[1]
    if snapshot.get("id") != gap_id:
        raise DDResolutionGraphEvidenceMismatch(
            "current graph DDGap identity does not match the exact path and kind"
        )
    if snapshot.get("path") != exact_path:
        raise DDResolutionGraphEvidenceMismatch(
            "current graph DDGap path does not match the candidate path"
        )
    if snapshot.get("kind") != exact_kind:
        raise DDResolutionGraphEvidenceMismatch(
            "current graph DDGap kind does not match the reviewed kind"
        )
    if snapshot.get("observed_dd_version") != candidate.dd_version:
        raise DDResolutionGraphEvidenceMismatch(
            "current graph DDGap DD version does not match the candidate release"
        )
    if snapshot.get("observed_value") != candidate.observed.value:
        raise DDResolutionGraphEvidenceMismatch(
            "current graph DDGap observed value does not match the raw release fact"
        )
    observations = tuple(
        sorted(str(item.get("id") or "") for item in snapshot.get("observations") or ())
    )
    if not observations or any(not item for item in observations):
        raise DDResolutionGraphEvidenceMismatch(
            "current graph DDGap has no exact observation set"
        )
    expected_observations = tuple(
        sorted(str(item) for item in expected_observation_ids)
    )
    if observations != expected_observations:
        raise DDResolutionGraphEvidenceMismatch(
            "current graph DDGap observation set differs from the reviewed set"
        )
    canonical_token = _evidence_token(snapshot)
    if snapshot.get("evidence_token") not in (None, canonical_token):
        raise DDResolutionGraphEvidenceMismatch(
            "graph DDGap carried a noncanonical evidence token"
        )
    if canonical_token != expected_evidence_token:
        raise DDResolutionGraphEvidenceMismatch(
            "expected token does not match the canonical graph evidence token"
        )
    for observation in snapshot.get("observations") or ():
        source_path = observation.get("source_path")
        if source_path is not None and source_path != exact_path:
            raise DDResolutionGraphEvidenceMismatch(
                "graph observation source path does not match the candidate path"
            )
        observed_version = observation.get("observed_dd_version")
        if observed_version is not None and observed_version != candidate.dd_version:
            raise DDResolutionGraphEvidenceMismatch(
                "graph observation DD version does not match the candidate release"
            )
        observed_value = observation.get("observed_value")
        if observed_value is not None and observed_value != candidate.observed.value:
            raise DDResolutionGraphEvidenceMismatch(
                "graph observation value does not match the raw release fact"
            )
    return canonical_token, observations


def approve_dd_resolution_candidate(
    source_row: str,
    *,
    path: str,
    gap_kind: DDGapKind | str,
    observation_ids: Sequence[str],
    evidence_token: str,
    actor: str,
    reason: str,
    revision: int,
    expected_manifest_digest: str,
    graph_reader: DDResolutionGraphReader | None = None,
    approved_at: datetime | None = None,
) -> DDResolutionRecord:
    """Promote one exact reviewed candidate path into tracked authority."""
    review_input = load_dd_resolution_candidates_for_review()
    candidates = {
        candidate.source_row: candidate for candidate in review_input.candidates
    }
    candidate = candidates.get(source_row)
    if candidate is None:
        raise DDResolutionEvidenceMismatch(
            f"DD resolution candidate {source_row!r} was not found"
        )
    if candidate.disposition == DDResolutionCandidateDisposition.broad_scope_hold:
        raise DDResolutionEvidenceMismatch(
            f"candidate {source_row!r} is a broad-scope hold and cannot be approved"
        )
    if len(candidate.exact_paths) != candidate.source_release_match_count:
        raise DDResolutionEvidenceMismatch(
            f"candidate {source_row!r} has an unresolved graph conflict: "
            f"{len(candidate.exact_paths)} exact reviewed paths for "
            f"{candidate.source_release_match_count} release matches"
        )
    exact_path = _validate_exact_path(path)
    if exact_path not in candidate.exact_paths:
        raise DDResolutionEvidenceMismatch(
            f"path {exact_path!r} is not an exact reviewed path for candidate "
            f"{source_row!r}"
        )
    if isinstance(revision, bool) or revision < 1:
        raise DDResolutionEvidenceMismatch("resolution revision must be positive")
    if not actor.strip() or not reason.strip():
        raise DDResolutionEvidenceMismatch(
            "approval requires an explicit actor and governed decision reason"
        )
    upstream = review_input.upstream_changes.get(candidate.upstream_change)
    if upstream is None or not upstream.change_url or not upstream.solution_commits:
        raise DDResolutionEvidenceMismatch(
            f"candidate {source_row!r} lacks an official upstream solution reference"
        )

    manifest_path = dd_resolution_manifest_path()
    reader = graph_reader or dd_resolution_graph_reader()
    gap_id = f"dd_gap:{exact_path}:{_enum_text(gap_kind)}"
    with _locked_manifest(
        manifest_path, expected_digest=expected_manifest_digest
    ) as manifest:
        graph_token, graph_observations = _verify_graph_evidence(
            reader.get_gap(gap_id),
            gap_id=gap_id,
            candidate=candidate,
            expected_observation_ids=observation_ids,
            expected_evidence_token=evidence_token,
        )
        exact_key = (exact_path, candidate.dd_version, candidate.field)
        if any(
            (record.path, record.dd_version, record.field) == exact_key
            for record in _effective_active_records(manifest)
        ):
            raise DDResolutionCollision(
                f"an active DD resolution already claims exact key {exact_key!r}"
            )
        prior_revisions = [
            record.resolution_revision
            for record in manifest.resolutions
            if (record.path, record.dd_version, record.field) == exact_key
        ]
        if prior_revisions and revision <= max(prior_revisions):
            raise DDResolutionCollision(
                f"revision {revision} must exceed prior revision "
                f"{max(prior_revisions)} for exact key {exact_key!r}"
            )
        prior_observations = {
            observation_id
            for record in manifest.resolutions
            for observation_id in record.observation_ids
        }
        repeated_observations = prior_observations.intersection(graph_observations)
        if repeated_observations:
            raise DDResolutionEvidenceMismatch(
                "approval requires fresh DDGap observations; already used "
                f"identities: {sorted(repeated_observations)!r}"
            )
        if graph_token in {record.evidence_token for record in manifest.resolutions}:
            raise DDResolutionEvidenceMismatch(
                "approval requires a fresh DDGap evidence token"
            )

        timestamp = approved_at or datetime.now(UTC)
        receipt_payload = {
            "source_row": source_row,
            "path": exact_path,
            "dd_version": candidate.dd_version,
            "field": candidate.field,
            "actor": actor,
            "reason": reason,
            "revision": revision,
            "approved_at": timestamp,
            "observation_ids": graph_observations,
            "evidence_token": graph_token,
            "candidate_digest": review_input.digest,
        }
        approval_receipt = (
            f"dd-resolution-approval:{source_row}:{review_input.digest}:"
            f"{_canonical_digest(receipt_payload)}"
        )
        record_payload: dict[str, Any] = {
            "gap_id": gap_id,
            "path": exact_path,
            "dd_version": candidate.dd_version,
            "field": candidate.field,
            "observed": candidate.observed,
            "observed_hash": dd_resolution_value_hash(candidate.observed),
            "effective": candidate.proposed_effective,
            "resolution_revision": revision,
            "reason": reason,
            "observation_ids": graph_observations,
            "evidence_token": graph_token,
            "approved_by": actor,
            "approved_at": timestamp,
            "approval_receipt": approval_receipt,
            "upstream_url": upstream.change_url,
            "upstream_ref": "commits:" + ",".join(upstream.solution_commits),
            "state": DDResolutionStatus.active,
        }
        record_payload["id"] = content_addressed_resolution_id(record_payload)
        record = DDResolutionRecord.model_validate(record_payload)
        updated = DDResolutionManifest(
            schema_version=manifest.schema_version,
            resolutions=(*manifest.resolutions, record),
            state_changes=manifest.state_changes,
        )
        try:
            final_token, final_observations = _verify_graph_evidence(
                reader.get_gap(gap_id),
                gap_id=gap_id,
                candidate=candidate,
                expected_observation_ids=graph_observations,
                expected_evidence_token=graph_token,
            )
        except DDResolutionGraphEvidenceMismatch as exc:
            raise DDResolutionGraphEvidenceMismatch(
                f"current DDGap evidence changed during approval: {exc}"
            ) from exc
        if (final_token, final_observations) != (graph_token, graph_observations):
            raise DDResolutionGraphEvidenceMismatch(
                "current DDGap evidence changed during approval"
            )
        _write_manifest(
            manifest_path,
            updated,
            expected_digest=expected_manifest_digest,
        )
        return record


def revoke_dd_resolution(
    resolution_id: str,
    *,
    actor: str,
    reason: str,
    expected_manifest_digest: str,
    changed_at: datetime | None = None,
) -> DDResolutionStateChangeReceipt:
    """Withdraw active authority while retaining both immutable record states."""
    if not actor.strip() or not reason.strip():
        raise DDResolutionEvidenceMismatch(
            "revocation requires an explicit actor and reason"
        )
    manifest_path = dd_resolution_manifest_path()
    with _locked_manifest(
        manifest_path, expected_digest=expected_manifest_digest
    ) as manifest:
        active = {record.id: record for record in _effective_active_records(manifest)}
        record = active.get(resolution_id)
        if record is None:
            raise DDResolutionEvidenceMismatch(
                f"resolution {resolution_id!r} is not effective active authority"
            )

        withdrawn_payload = record.model_dump(mode="json", exclude={"id", "state"})
        withdrawn_payload["state"] = DDResolutionStatus.withdrawn
        withdrawn_payload["id"] = content_addressed_resolution_id(withdrawn_payload)
        withdrawn = DDResolutionRecord.model_validate(withdrawn_payload)
        timestamp = changed_at or datetime.now(UTC)
        receipt_payload: dict[str, Any] = {
            "from_resolution_id": record.id,
            "to_resolution_id": withdrawn.id,
            "from_status": record.state,
            "to_status": withdrawn.state,
            "actor": actor,
            "reason": reason,
            "changed_at": timestamp,
        }
        receipt_payload["id"] = (
            f"dd-resolution-state-change:{_canonical_digest(receipt_payload)}"
        )
        receipt = DDResolutionStateChangeReceipt.model_validate(receipt_payload)
        updated = DDResolutionManifest(
            schema_version=manifest.schema_version,
            resolutions=(*manifest.resolutions, withdrawn),
            state_changes=(*manifest.state_changes, receipt),
        )
        _write_manifest(
            manifest_path,
            updated,
            expected_digest=expected_manifest_digest,
        )
        return receipt


def _same_value(left: DDResolutionValue, right: DDResolutionValue) -> bool:
    return _canonical_json(left) == _canonical_json(right)


def _resolved_receipt(
    *,
    field: DDResolutionField,
    raw: DDResolutionValue,
    effective: DDResolutionValue,
    record: DDResolutionRecord | None = None,
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
    """Resolve one exact field without mutating raw input or external state."""
    exact_path = _validate_exact_path(path)
    exact_version = _validate_exact_version(dd_version)
    authority = manifest or load_dd_resolution_manifest()
    active = tuple(
        record
        for record in _effective_active_records(authority)
        if record.path == exact_path and record.field == field
    )
    exact = tuple(record for record in active if record.dd_version == exact_version)
    if len(exact) > 1:
        raise DDResolutionCollision(
            f"multiple active resolutions claim {(exact_path, exact_version, field.value)!r}"
        )
    if exact:
        record = exact[0]
        _verify_record_provenance(record)
        if record.observed_hash != dd_resolution_value_hash(record.observed):
            raise DDResolutionEvidenceMismatch(
                f"resolution {record.id!r} observed hash changed"
            )
        if _same_value(raw_value, record.observed):
            return _resolved_receipt(
                field=field,
                raw=raw_value,
                effective=record.effective,
                record=record,
                applied=True,
            )
        if _same_value(raw_value, record.effective):
            return _resolved_receipt(
                field=field,
                raw=raw_value,
                effective=raw_value,
                record=record,
                converged=True,
            )
        raise DDResolutionStale(
            f"raw value for {(exact_path, exact_version, field.value)!r} matches neither "
            f"reviewed observed nor effective value in {record.id!r}"
        )

    if active:
        converged = tuple(
            record for record in active if _same_value(raw_value, record.effective)
        )
        if len(converged) > 1:
            raise DDResolutionAmbiguity(
                f"multiple prior-version records could certify convergence for "
                f"{(exact_path, exact_version, field.value)!r}"
            )
        if converged:
            record = converged[0]
            _verify_record_provenance(record)
            return _resolved_receipt(
                field=field,
                raw=raw_value,
                effective=raw_value,
                record=record,
                converged=True,
            )
        reviewed_versions = sorted({record.dd_version for record in active})
        raise DDResolutionVersionMismatch(
            f"resolution for {(exact_path, field.value)!r} was reviewed only for "
            f"{reviewed_versions!r}, not exact version {exact_version!r}"
        )

    return _resolved_receipt(
        field=field,
        raw=raw_value,
        effective=raw_value,
    )


_CONTEXT_FIELDS = (
    DDResolutionField.unit,
    DDResolutionField.documentation,
    DDResolutionField.data_type,
    DDResolutionField.node_type,
    DDResolutionField.physics_domain,
    DDResolutionField.cocos_transformation_type,
    DDResolutionField.cocos_transformation_expression,
    DDResolutionField.coordinates,
    DDResolutionField.lifecycle_status,
    DDResolutionField.lifecycle_version,
)


def _context_raw_value(
    raw: RawDDContext, field: DDResolutionField
) -> DDResolutionValue:
    field_text = _enum_text(field)
    value = getattr(raw, field_text)
    if field == DDResolutionField.coordinates:
        return DDResolutionValue(kind=DDResolutionValueKind.string_list, value=value)
    kind = DDResolutionValueKind.null if value is None else DDResolutionValueKind.string
    return DDResolutionValue(kind=kind, value=value)


def _effective_projection(field: ResolvedDDField) -> str | tuple[str, ...] | None:
    return field.effective.value


def resolve_dd_context(
    raw: RawDDContext,
    *,
    manifest: DDResolutionManifest | None = None,
) -> ResolvedDDContext:
    """Resolve every authoritative field and nested DD grounding context."""
    authority = manifest or load_dd_resolution_manifest()
    fields = tuple(
        resolve_dd_field(
            path=raw.path,
            dd_version=raw.dd_version,
            field=field,
            raw_value=_context_raw_value(raw, field),
            manifest=authority,
        )
        for field in _CONTEXT_FIELDS
    )
    by_field = {item.field: _effective_projection(item) for item in fields}
    applied_ids = tuple(sorted(item.resolution_id for item in fields if item.applied))
    converged_ids = tuple(
        sorted(item.resolution_id for item in fields if item.converged)
    )
    coordinates = by_field[DDResolutionField.coordinates]
    if not isinstance(coordinates, tuple):
        raise DDResolutionEvidenceMismatch(
            "coordinates resolution did not produce an ordered string list"
        )
    return ResolvedDDContext(
        raw=raw,
        unit=by_field[DDResolutionField.unit],
        documentation=by_field[DDResolutionField.documentation],
        data_type=by_field[DDResolutionField.data_type],
        node_type=by_field[DDResolutionField.node_type],
        physics_domain=by_field[DDResolutionField.physics_domain],
        cocos_transformation_type=by_field[DDResolutionField.cocos_transformation_type],
        cocos_transformation_expression=by_field[
            DDResolutionField.cocos_transformation_expression
        ],
        coordinates=coordinates,
        lifecycle_status=by_field[DDResolutionField.lifecycle_status],
        lifecycle_version=by_field[DDResolutionField.lifecycle_version],
        resolved_fields=fields,
        applied_resolution_ids=applied_ids,
        converged_resolution_ids=converged_ids,
        manifest_digest=authority.digest,
        parents=tuple(
            resolve_dd_context(parent, manifest=authority) for parent in raw.parents
        ),
        members=tuple(
            resolve_dd_context(member, manifest=authority) for member in raw.members
        ),
    )


def resolve_dd_row(
    row: Mapping[str, Any],
    *,
    dd_version: str,
    manifest: DDResolutionManifest | None = None,
) -> ResolvedDDContext:
    """Resolve one raw graph projection while retaining exact provenance."""
    exact_version = _validate_exact_version(dd_version)
    row_version = row.get("dd_version")
    if row_version is not None and str(row_version) != exact_version:
        raise DDResolutionVersionMismatch(
            f"DD row carries version {row_version!r}, expected {exact_version!r}"
        )
    if "path" not in row:
        raise ValueError("DD row is missing exact path")
    return resolve_dd_context(
        RawDDContext(
            path=row["path"],
            dd_version=exact_version,
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
    """Resolve an all-or-error batch of raw graph rows under one exact DD version."""
    exact_version = _validate_exact_version(dd_version)
    authority = manifest or load_dd_resolution_manifest()
    contexts: list[ResolvedDDContext] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise TypeError(f"DD row {index} must be a mapping")
        try:
            contexts.append(
                resolve_dd_row(row, dd_version=exact_version, manifest=authority)
            )
        except (ValueError, DDResolutionError) as exc:
            raise type(exc)(f"DD row {index}: {exc}") from exc
    return contexts


# Alias exposes resolver terminology while the generated LinkML enum remains
# the single source of lifecycle values.
DDResolutionState = DDResolutionStatus

__all__ = [
    "DDResolutionAmbiguity",
    "DDResolutionCandidate",
    "DDResolutionCandidateDisposition",
    "DDResolutionCandidateManifest",
    "DDResolutionCandidateUpstreamChange",
    "DDResolutionCollision",
    "DDResolutionError",
    "DDResolutionEvidenceMismatch",
    "DDResolutionField",
    "DDResolutionGraphEvidenceMismatch",
    "DDResolutionGraphPathAction",
    "DDResolutionGraphPathReceipt",
    "DDResolutionGraphPort",
    "DDResolutionGraphPortConflict",
    "DDResolutionGraphPortReceipt",
    "DDResolutionGraphReader",
    "DDResolutionManifest",
    "DDResolutionManifestConflict",
    "DDResolutionManifestInvalid",
    "DDResolutionRecord",
    "DDResolutionStale",
    "DDResolutionState",
    "DDResolutionStatus",
    "DDResolutionStateChangeReceipt",
    "DDResolutionValue",
    "DDResolutionVersionMismatch",
    "DDResolutionUpstreamStatus",
    "RawDDContext",
    "ResolvedDDContext",
    "ResolvedDDField",
    "content_addressed_resolution_id",
    "dd_resolution_value_hash",
    "dd_resolution_graph_reader",
    "dd_resolution_manifest_path",
    "approved_candidate_paths",
    "approve_dd_resolution_candidate",
    "load_dd_resolution_candidates_for_review",
    "load_dd_resolution_manifest",
    "port_active_dd_resolutions_to_graph",
    "resolve_dd_context",
    "resolve_dd_field",
    "resolve_dd_row",
    "resolve_dd_rows",
    "revoke_dd_resolution",
]
