"""Audit and describe legacy DD correction authority retired at runtime."""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass

from imas_codex.settings import get_dd_version
from imas_codex.standard_names.dd_resolutions import (
    DDResolutionField,
    DDResolutionManifest,
    DDResolutionRecord,
    effective_active_dd_resolutions,
)
from imas_codex.standard_names.unit_overrides import _load_rules, resolve_unit
from imas_codex.units.dd_unit_exceptions import (
    canonical_or_none,
    graph_unit_correction,
    load_exceptions,
    units_agree,
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
) -> tuple[ShadowAuthority, ...]:
    """Find every shipped carrier still able to alter an active exact field."""
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

    if shadows:
        return tuple(shadows)

    for retirement in legacy_authority_retirements(
        manifest=manifest, dd_version=version
    ):
        record = next(item for item in records if item.id == retirement.resolution_id)
        if retirement.carrier == "dd_unit_exceptions":
            correction = graph_unit_correction(
                record.path,
                record.observed.value,
                dd_version=version,
                manifest=manifest,
            )
            suppressed = units_agree(
                record.effective.value,
                record.observed.value,
                record.path,
                dd_version=version,
                manifest=manifest,
            )
            residual = correction is not None or suppressed
        else:
            _, metadata = resolve_unit(
                record.path,
                record.observed.value,
                dd_version=version,
                manifest=manifest,
            )
            residual = metadata is not None
        if residual:
            shadows.append(
                ShadowAuthority(
                    carrier=retirement.carrier,
                    row_id=retirement.row_id,
                    resolution_id=retirement.resolution_id,
                    path=retirement.path,
                )
            )

    for record in records:
        if record.observed.value is None and record.effective.value == "1":
            shadows.append(
                ShadowAuthority(
                    carrier="numeric_missing_unit_fallback",
                    row_id="numeric-no-unit",
                    resolution_id=record.id,
                    path=record.path,
                )
            )
    return tuple(shadows)


def assert_zero_shadow_authority(
    *,
    manifest: DDResolutionManifest | None = None,
    dd_version: str | None = None,
) -> None:
    """Refuse when any legacy or duplicate carrier shadows active authority."""
    shadows = find_shadow_authorities(manifest=manifest, dd_version=dd_version)
    if not shadows:
        return
    details = "; ".join(
        f"carrier={item.carrier} row_id={item.row_id} "
        f"resolution_id={item.resolution_id} path={item.path}"
        for item in shadows
    )
    raise DDResolutionShadowAuthority(details)
