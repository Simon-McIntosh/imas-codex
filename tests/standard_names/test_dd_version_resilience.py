"""Dictionary labels do not override byte-identical resolution content."""

from __future__ import annotations

from datetime import UTC, datetime

from imas_codex.graph.models import DDResolutionValueKind
from imas_codex.standard_names.dd_resolutions import (
    DDResolutionManifest,
    DDResolutionRecord,
    DDResolutionValue,
    DDResolutionVersionMismatch,
    resolve_dd_row,
)

_REVIEWED_VERSION = "4.1.1"
_AUTHORITATIVE_VERSION = "4.2.0"

_FIELD_VALUES = (
    ("unit", "m", "1"),
    ("documentation", "published documentation", "effective documentation"),
    ("data_type", "FLT_0D", "FLT_1D"),
    ("node_type", "static", "dynamic"),
    ("physics_domain", "transport", "magnetohydrodynamics"),
    ("cocos_transformation_type", "ip_like", "psi_like"),
    ("cocos_transformation_expression", "published expression", "effective expression"),
    (
        "coordinates",
        ("synthetic_resilience/grid/published",),
        ("synthetic_resilience/grid/effective",),
    ),
    ("lifecycle_status", "alpha", "active"),
    ("lifecycle_version", "4.1.0", "4.1.1"),
)


def _value(value: str | tuple[str, ...]) -> DDResolutionValue:
    kind = (
        DDResolutionValueKind.string_list
        if isinstance(value, tuple)
        else DDResolutionValueKind.string
    )
    return DDResolutionValue(kind=kind, value=value)


def _corpus() -> tuple[DDResolutionManifest, tuple[dict[str, object], ...]]:
    records = []
    rows = []
    for index, (field, observed, effective) in enumerate(_FIELD_VALUES):
        path = f"synthetic_resilience/source_{index}/value"
        records.append(
            DDResolutionRecord.model_validate(
                {
                    "id": f"dd_resolution:{index:064x}",
                    "gap_id": f"dd_gap:{path}:declaration_defect",
                    "path": path,
                    "dd_version": _REVIEWED_VERSION,
                    "field": field,
                    "observed": _value(observed),
                    "effective": _value(effective),
                    "reason": "Exercise content convergence across a dictionary label change.",
                    "recorded_by": "synthetic-test-maintainer",
                    "recorded_at": datetime(2026, 8, 21, tzinfo=UTC),
                    "upstream_reference": "none-yet",
                    "retiring_release": "none-yet",
                    "state": "active",
                }
            )
        )
        rows.append(
            {
                "path": path,
                "dd_version": _REVIEWED_VERSION,
                field: list(effective) if isinstance(effective, tuple) else effective,
            }
        )
    return DDResolutionManifest(resolutions=tuple(records)), tuple(rows)


def _resolved_count(
    rows: tuple[dict[str, object], ...], manifest: DDResolutionManifest
) -> int:
    resolved = 0
    for row in rows:
        try:
            resolve_dd_row(
                row,
                dd_version=_AUTHORITATIVE_VERSION,
                manifest=manifest,
            )
        except DDResolutionVersionMismatch:
            continue
        resolved += 1
    return resolved


def test_no_op_dictionary_bump_resolves_the_full_source_corpus() -> None:
    manifest, rows = _corpus()

    resolved = [
        resolve_dd_row(
            row,
            dd_version=_AUTHORITATIVE_VERSION,
            manifest=manifest,
        )
        for row in rows
    ]

    assert len(resolved) == len(_FIELD_VALUES)
    assert all(item.raw.dd_version == _AUTHORITATIVE_VERSION for item in resolved)
    assert all(item.converged_resolution_ids for item in resolved)


def test_unreviewed_content_still_refuses_after_a_dictionary_bump() -> None:
    manifest, rows = _corpus()
    changed = {**rows[0], "unit": "kg"}

    try:
        resolve_dd_row(
            changed,
            dd_version=_AUTHORITATIVE_VERSION,
            manifest=manifest,
        )
    except DDResolutionVersionMismatch as exc:
        assert "reviewed only" in str(exc)
    else:
        raise AssertionError("unreviewed content must refuse")


def test_recorded_label_mismatch_count_does_not_change_resolution_count() -> None:
    manifest, rows = _corpus()

    for mismatched_count in range(len(rows) + 1):
        relabelled = tuple(
            {
                **row,
                "dd_version": (
                    _REVIEWED_VERSION
                    if index < mismatched_count
                    else _AUTHORITATIVE_VERSION
                ),
            }
            for index, row in enumerate(rows)
        )
        assert _resolved_count(relabelled, manifest) == len(rows)
