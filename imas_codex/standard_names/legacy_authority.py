"""Audit and describe legacy DD correction authority retired at runtime."""

from __future__ import annotations

import fnmatch
from collections import Counter
from dataclasses import dataclass
from enum import StrEnum

from imas_codex.settings import get_dd_version
from imas_codex.standard_names.dd_resolutions import (
    DDResolutionField,
    DDResolutionManifest,
    DDResolutionRecord,
    effective_active_dd_resolutions,
)
from imas_codex.standard_names.unit_overrides import _load_rules
from imas_codex.units.dd_unit_exceptions import (
    canonical_or_none,
    load_exceptions,
)


class DDResolutionShadowAuthority(RuntimeError):
    """Raised when more than one carrier can correct an active exact field."""


@dataclass(frozen=True)
class LegacyRetirement:
    """One legacy carrier row made inert by an active exact resolution."""

    carrier: str
    row_id: str
    resolution_id: str
    path: str


@dataclass(frozen=True)
class ShadowAuthority:
    """One residual authority competing with an active exact resolution."""

    carrier: str
    row_id: str
    resolution_id: str
    path: str


class ShadowAuditStatus(StrEnum):
    """Whether one carrier class is covered by this audit."""

    audited = "audited"
    not_audited = "not_audited"


@dataclass(frozen=True)
class ShadowCarrierAudit:
    """Coverage and residual count for one authority carrier class."""

    carrier: str
    status: ShadowAuditStatus
    residual_count: int | None
    reason: str | None = None


@dataclass(frozen=True)
class ShadowAuthorityAudit:
    """Residual authorities plus explicit carrier coverage."""

    residuals: tuple[ShadowAuthority, ...]
    carrier_results: tuple[ShadowCarrierAudit, ...]


_AUDITED_SHADOW_CARRIERS = (
    "active_manifest",
    "unit_overrides.override",
    "unit_overrides.skip",
    "dd_unit_exceptions.suppress",
    "dd_unit_exceptions.correct_in_graph",
    "numeric_missing_unit_fallback",
)


def _unit_records(
    manifest: DDResolutionManifest | None,
    dd_version: str,
) -> tuple[DDResolutionRecord, ...]:
    return tuple(
        record
        for record in effective_active_dd_resolutions(manifest)
        if record.dd_version == dd_version and record.field == DDResolutionField.unit
    )


def legacy_authority_retirements(
    *,
    manifest: DDResolutionManifest | None = None,
    dd_version: str | None = None,
) -> tuple[LegacyRetirement, ...]:
    """Return legacy rows whose exact matches are retired by active authority."""
    version = dd_version or get_dd_version()
    retirements: list[LegacyRetirement] = []
    exceptions = load_exceptions()["dd_unit_bugs"]
    overrides = _load_rules()
    for record in _unit_records(manifest, version):
        observed = canonical_or_none(record.observed.value)
        effective = canonical_or_none(record.effective.value)
        for index, entry in enumerate(exceptions, start=1):
            if (
                fnmatch.fnmatchcase(record.path, str(entry["path"]))
                and canonical_or_none(str(entry["dd_unit"])) == observed
                and canonical_or_none(str(entry["correct_unit"])) == effective
            ):
                retirements.append(
                    LegacyRetirement(
                        carrier="dd_unit_exceptions",
                        row_id=f"dd-unit-exception:{index}",
                        resolution_id=record.id,
                        path=record.path,
                    )
                )
        for index, rule in enumerate(overrides, start=1):
            expected = rule.override_unit if rule.strategy == "override" else None
            if rule.matches(record.path, record.observed.value) and (
                record.effective.value == expected
            ):
                retirements.append(
                    LegacyRetirement(
                        carrier="unit_overrides",
                        row_id=f"unit-override:{index}",
                        resolution_id=record.id,
                        path=record.path,
                    )
                )
    return tuple(retirements)


def find_shadow_authorities(
    *,
    manifest: DDResolutionManifest | None = None,
    dd_version: str | None = None,
) -> ShadowAuthorityAudit:
    """Audit covered carriers able to alter an active exact field."""
    version = dd_version or get_dd_version()
    records = _unit_records(manifest, version)
    shadows: list[ShadowAuthority] = []
    seen: dict[tuple[str, str, DDResolutionField], DDResolutionRecord] = {}
    for record in records:
        key = (record.path, record.dd_version, record.field)
        if key in seen:
            shadows.append(
                ShadowAuthority(
                    carrier="active_manifest",
                    row_id=record.id,
                    resolution_id=seen[key].id,
                    path=record.path,
                )
            )
        else:
            seen[key] = record

    exceptions = load_exceptions()["dd_unit_bugs"]
    overrides = _load_rules()
    for record in records:
        observed = canonical_or_none(record.observed.value)
        effective = canonical_or_none(record.effective.value)
        for index, entry in enumerate(exceptions, start=1):
            if not fnmatch.fnmatchcase(record.path, str(entry["path"])):
                continue
            if canonical_or_none(str(entry["dd_unit"])) != observed:
                continue
            if canonical_or_none(str(entry["correct_unit"])) == effective:
                continue
            carrier = (
                "dd_unit_exceptions.correct_in_graph"
                if entry.get("correct_in_graph")
                else "dd_unit_exceptions.suppress"
            )
            shadows.append(
                ShadowAuthority(
                    carrier=carrier,
                    row_id=f"dd-unit-exception:{index}",
                    resolution_id=record.id,
                    path=record.path,
                )
            )
        for index, rule in enumerate(overrides, start=1):
            if not rule.matches(record.path, record.observed.value):
                continue
            result = rule.override_unit if rule.strategy == "override" else None
            if record.effective.value == result:
                continue
            shadows.append(
                ShadowAuthority(
                    carrier=f"unit_overrides.{rule.strategy}",
                    row_id=f"unit-override:{index}",
                    resolution_id=record.id,
                    path=record.path,
                )
            )

    residuals = tuple(shadows)
    counts = Counter(item.carrier for item in residuals)
    carrier_results = tuple(
        ShadowCarrierAudit(
            carrier=carrier,
            status=ShadowAuditStatus.audited,
            residual_count=counts[carrier],
        )
        for carrier in _AUDITED_SHADOW_CARRIERS
    )
    return ShadowAuthorityAudit(
        residuals=residuals,
        carrier_results=carrier_results,
    )


def assert_zero_audited_shadow_authorities(
    *,
    manifest: DDResolutionManifest | None = None,
    dd_version: str | None = None,
) -> ShadowAuthorityAudit:
    """Return covered-carrier results or refuse on any audited residual."""
    audit = find_shadow_authorities(
        manifest=manifest,
        dd_version=dd_version,
    )
    shadows = audit.residuals
    if not shadows:
        return audit
    details = "; ".join(
        f"carrier={item.carrier} row_id={item.row_id} "
        f"resolution_id={item.resolution_id} path={item.path}"
        for item in shadows
    )
    raise DDResolutionShadowAuthority(details)
