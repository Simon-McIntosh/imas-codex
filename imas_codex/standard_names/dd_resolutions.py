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
from datetime import datetime
from enum import Enum
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
            converged=True,
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
    resolution_ids = tuple(record.id for record in provenance)
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
        applied_resolution_ids=resolution_ids,
        converged_resolution_ids=resolution_ids,
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
    if row.get("dd_version") is not None and str(row["dd_version"]) != version:
        raise DDResolutionVersionMismatch(
            f"DD row carries version {row['dd_version']!r}, expected {version!r}"
        )
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
    "resolve_dd_context",
    "resolve_dd_field",
    "resolve_dd_row",
    "resolve_dd_rows",
]
