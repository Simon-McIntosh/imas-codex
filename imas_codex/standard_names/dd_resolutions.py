"""Reviewed, version-bound interpretations of immutable DD declarations.

The packaged YAML manifest is the only runtime behavior authority. Graph
``DDResolution`` nodes are lifecycle and provenance mirrors; this module never
queries or writes them. A resolution applies only to one exact IDS-prefixed
path, one exact published DD version, and one typed field. Any ambiguity,
staleness, or attempted cross-version reuse fails closed.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from enum import Enum, StrEnum
from functools import lru_cache
from importlib import resources
from typing import Any, Self
from urllib.parse import urlparse

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
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
_OBSERVATION_ID_RE = re.compile(r"dd_gap_observation:[0-9a-f]{64}\Z")
_EVIDENCE_TOKEN_RE = re.compile(r"dd-gap-evidence:[0-9a-f]{64}\Z")
_UPSTREAM_COMMIT_RE = re.compile(r"[0-9a-f]{40}\Z")
_SOURCE_ROW_RE = re.compile(r"[UO][0-9]{2}\Z")
_UPSTREAM_CHANGE_KEY_RE = re.compile(r"[a-z][a-z0-9]*(?:-[a-z0-9]+)*\Z")
_PATTERN_CHARACTERS = frozenset("*?[]{}\\^$|")

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


class DDResolutionCandidateDisposition(StrEnum):
    """Review routing for evidence that has no behavior authority."""

    bounded_review_input = "bounded_review_input"
    broad_scope_hold = "broad_scope_hold"


class DDResolutionUpstreamStatus(StrEnum):
    """Official change state observed for review-only provenance."""

    open = "open"
    merged = "merged"


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


class DDResolutionManifest(BaseModel):
    """Complete validated package behavior authority."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int
    resolutions: tuple[DDResolutionRecord, ...]

    @model_validator(mode="after")
    def _validate_manifest(self) -> Self:
        if self.schema_version != _MANIFEST_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported DD resolution schema_version {self.schema_version!r}"
            )
        seen_ids: set[str] = set()
        active_keys: dict[tuple[str, str, DDResolutionField], str] = {}
        observation_owners: dict[str, tuple[str, str, DDResolutionField]] = {}
        evidence_owners: dict[str, tuple[str, str, DDResolutionField]] = {}
        lifecycle_fields: dict[tuple[str, str], set[DDResolutionField]] = {}
        for record in self.resolutions:
            if record.id in seen_ids:
                raise DDResolutionCollision(f"duplicate resolution id {record.id!r}")
            seen_ids.add(record.id)
            if record.state != DDResolutionStatus.active:
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
    source_release_match_count: int
    exact_paths: tuple[str, ...]
    narrow_evidence_overlap_count: int | None = None
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

    schema_version: int
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
    reference = resources.files("imas_codex.standard_names.config").joinpath(
        _MANIFEST_RESOURCE
    )
    try:
        content = reference.read_text(encoding="utf-8")
    except (FileNotFoundError, OSError) as exc:
        raise DDResolutionManifestInvalid(
            f"packaged DD resolution manifest {_MANIFEST_RESOURCE!r} is unavailable"
        ) from exc
    return _parse_manifest_content(content)


@lru_cache(maxsize=8)
def _parse_candidate_content(content: str) -> DDResolutionCandidateManifest:
    try:
        document = yaml.safe_load(content)
    except yaml.YAMLError as exc:
        raise DDResolutionManifestInvalid(
            "DD resolution candidate resource is not valid YAML"
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
        for record in authority.resolutions
        if record.state == DDResolutionStatus.active
        and record.path == exact_path
        and record.field == field
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


def resolve_dd_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    dd_version: str,
    manifest: DDResolutionManifest | None = None,
) -> list[ResolvedDDContext]:
    """Resolve an all-or-error batch of raw graph rows under one exact DD version."""
    exact_version = _validate_exact_version(dd_version)
    authority = manifest or load_dd_resolution_manifest()
    raw_contexts: list[RawDDContext] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise TypeError(f"DD row {index} must be a mapping")
        row_version = row.get("dd_version")
        if row_version is not None and str(row_version) != exact_version:
            raise DDResolutionVersionMismatch(
                f"DD row {index} carries version {row_version!r}, expected {exact_version!r}"
            )
        if "path" not in row:
            raise ValueError(f"DD row {index} is missing exact path")
        raw_contexts.append(
            RawDDContext(
                path=row["path"],
                dd_version=exact_version,
                unit=row.get("unit"),
                documentation=row.get("documentation"),
                data_type=row.get("data_type"),
                node_type=row.get("node_type"),
                physics_domain=row.get("physics_domain"),
                cocos_transformation_type=row.get("cocos_transformation_type"),
                cocos_transformation_expression=row.get(
                    "cocos_transformation_expression"
                ),
                coordinates=tuple(row.get("coordinates") or ()),
                lifecycle_status=row.get("lifecycle_status"),
                lifecycle_version=row.get("lifecycle_version"),
            )
        )
    return [resolve_dd_context(raw, manifest=authority) for raw in raw_contexts]


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
    "DDResolutionManifest",
    "DDResolutionManifestInvalid",
    "DDResolutionRecord",
    "DDResolutionStale",
    "DDResolutionState",
    "DDResolutionStatus",
    "DDResolutionValue",
    "DDResolutionVersionMismatch",
    "DDResolutionUpstreamStatus",
    "RawDDContext",
    "ResolvedDDContext",
    "ResolvedDDField",
    "content_addressed_resolution_id",
    "dd_resolution_value_hash",
    "load_dd_resolution_candidates_for_review",
    "load_dd_resolution_manifest",
    "resolve_dd_context",
    "resolve_dd_field",
    "resolve_dd_rows",
]
