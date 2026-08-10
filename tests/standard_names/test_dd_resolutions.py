from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from imas_codex.graph.models import (
    DDResolutionField,
    DDResolutionStatus,
    DDResolutionValueKind,
)
from imas_codex.standard_names.dd_resolutions import (
    DDResolutionAmbiguity,
    DDResolutionCollision,
    DDResolutionEvidenceMismatch,
    DDResolutionManifest,
    DDResolutionRecord,
    DDResolutionStale,
    DDResolutionValue,
    DDResolutionVersionMismatch,
    RawDDContext,
    content_addressed_resolution_id,
    dd_resolution_value_hash,
    load_dd_resolution_manifest,
    resolve_dd_context,
    resolve_dd_field,
    resolve_dd_rows,
)

_PATH = "equilibrium/time_slice/global_quantities/magnetic_axis/r"
_OBSERVATION_ID = f"dd_gap_observation:{'a' * 64}"
_EVIDENCE_TOKEN = f"dd-gap-evidence:{'b' * 64}"


def _value(value: str | None) -> DDResolutionValue:
    kind = DDResolutionValueKind.null if value is None else DDResolutionValueKind.string
    return DDResolutionValue(kind=kind, value=value)


def _record(**overrides: object) -> DDResolutionRecord:
    observed = overrides.pop("observed", _value("m"))
    effective = overrides.pop("effective", _value("cm"))
    payload: dict[str, object] = {
        "gap_id": f"dd_gap:{_PATH}:unit_defect",
        "path": _PATH,
        "dd_version": "4.1.0",
        "field": DDResolutionField.unit,
        "observed": observed,
        "observed_hash": dd_resolution_value_hash(observed),
        "effective": effective,
        "resolution_revision": 1,
        "reason": "The declaration contradicts the quantity dimensionality.",
        "observation_ids": (_OBSERVATION_ID,),
        "evidence_token": _EVIDENCE_TOKEN,
        "approved_by": "standard-names-review",
        "approved_at": datetime(2026, 8, 10, 9, 0, tzinfo=UTC),
        "approval_receipt": "review:dd-resolution:magnetic-axis-radius",
        "upstream_url": "https://github.com/iterorganization/IMAS-Data-Dictionary/pull/999",
        "upstream_ref": "pull/999",
        "state": DDResolutionStatus.active,
    }
    payload.update(overrides)
    if "observed_hash" not in overrides:
        payload["observed_hash"] = dd_resolution_value_hash(payload["observed"])
    if "id" not in overrides:
        payload["id"] = content_addressed_resolution_id(payload)
    return DDResolutionRecord.model_validate(payload)


def _manifest(*records: DDResolutionRecord) -> DDResolutionManifest:
    return DDResolutionManifest(schema_version=1, resolutions=records)


def test_packaged_manifest_is_reviewed_empty_authority() -> None:
    manifest = load_dd_resolution_manifest()

    assert manifest.schema_version == 1
    assert manifest.resolutions == ()
    assert manifest.digest.startswith("sha256:")


def test_manifest_digest_is_record_order_independent() -> None:
    first = _record()
    second = _record(
        dd_version="4.2.0",
        observation_ids=(f"dd_gap_observation:{'c' * 64}",),
        evidence_token=f"dd-gap-evidence:{'d' * 64}",
        resolution_revision=2,
    )

    assert _manifest(first, second).digest == _manifest(second, first).digest


@pytest.mark.parametrize(
    "path",
    [
        "equilibrium/**/r",
        "equilibrium/time_slice/?",
        "equilibrium/time_slice/[r]",
        "equilibrium/time_slice/{r,z}",
        "magnetic_axis_radius",
        "/equilibrium/time_slice/r",
        "equilibrium/../r",
    ],
)
def test_record_rejects_non_exact_or_ids_less_paths(path: str) -> None:
    with pytest.raises(ValidationError, match="path"):
        _record(path=path, gap_id=f"dd_gap:{path}:unit_defect")


@pytest.mark.parametrize("version", ["latest", ">=4.1.0", "4.1", "4.1.*", ""])
def test_record_rejects_non_exact_versions(version: str) -> None:
    with pytest.raises(ValidationError, match="exact published version"):
        _record(dd_version=version)


def test_record_rejects_wrong_value_kind_for_field() -> None:
    coordinates = DDResolutionValue(
        kind=DDResolutionValueKind.string_list,
        value=("equilibrium/time",),
    )

    with pytest.raises(ValidationError, match="requires value kinds"):
        _record(observed=coordinates)


def test_record_rejects_unknown_generated_dd_enum_value() -> None:
    with pytest.raises(ValidationError, match="DDDataType"):
        _record(
            field=DDResolutionField.data_type,
            gap_id=f"dd_gap:{_PATH}:type_wiring",
            observed=_value("FLOAT"),
            effective=_value("FLT_0D"),
        )


def test_record_rejects_pattern_coordinate_identity() -> None:
    observed = DDResolutionValue(
        kind=DDResolutionValueKind.string_list,
        value=("equilibrium/**/time",),
    )
    effective = DDResolutionValue(
        kind=DDResolutionValueKind.string_list,
        value=("equilibrium/time",),
    )

    with pytest.raises(ValidationError, match="coordinate identities"):
        _record(
            field=DDResolutionField.coordinates,
            gap_id=f"dd_gap:{_PATH}:type_wiring",
            observed=observed,
            effective=effective,
        )


def test_record_rejects_observed_hash_drift() -> None:
    with pytest.raises(DDResolutionEvidenceMismatch, match="observed_hash"):
        _record(observed_hash=f"sha256:{'0' * 64}")


def test_record_rejects_non_https_upstream_provenance() -> None:
    with pytest.raises(DDResolutionEvidenceMismatch, match="HTTPS"):
        _record(upstream_url="http://example.invalid/change")


def test_manifest_rejects_duplicate_active_exact_key() -> None:
    second = _record(
        reason="Independent review reached a conflicting local interpretation.",
        observation_ids=(f"dd_gap_observation:{'c' * 64}",),
        evidence_token=f"dd-gap-evidence:{'d' * 64}",
        resolution_revision=2,
    )

    with pytest.raises(DDResolutionCollision, match="exact key"):
        _manifest(_record(), second)


def test_manifest_rejects_evidence_reused_across_versions() -> None:
    with pytest.raises(DDResolutionCollision, match="observation"):
        _manifest(_record(), _record(dd_version="4.2.0", resolution_revision=2))


def test_manifest_requires_lifecycle_status_and_version_as_pair() -> None:
    record = _record(
        field=DDResolutionField.lifecycle_status,
        gap_id=f"dd_gap:{_PATH}:doc_mismatch",
        observed=_value("alpha"),
        effective=_value("active"),
    )

    with pytest.raises(DDResolutionCollision, match="status and version together"):
        _manifest(record)


def test_unrelated_field_passes_raw_unchanged() -> None:
    raw = _value("kg")

    result = resolve_dd_field(
        path="core_profiles/profiles_1d/electrons/density_thermal",
        dd_version="4.1.0",
        field=DDResolutionField.unit,
        raw_value=raw,
        manifest=_manifest(_record()),
    )

    assert result.raw is raw
    assert result.effective is raw
    assert result.applied is False
    assert result.converged is False
    assert result.resolution_id is None


def test_exact_observed_value_applies_reviewed_effective_value() -> None:
    record = _record()
    raw = _value("m")

    result = resolve_dd_field(
        path=_PATH,
        dd_version="4.1.0",
        field=DDResolutionField.unit,
        raw_value=raw,
        manifest=_manifest(record),
    )

    assert result.raw is raw
    assert result.effective.value == "cm"
    assert result.applied is True
    assert result.converged is False
    assert result.resolution_id == record.id
    assert result.provenance == record


def test_exact_raw_effective_value_converges_without_override() -> None:
    record = _record()
    raw = _value("cm")

    result = resolve_dd_field(
        path=_PATH,
        dd_version="4.1.0",
        field=DDResolutionField.unit,
        raw_value=raw,
        manifest=_manifest(record),
    )

    assert result.raw is raw
    assert result.effective is raw
    assert result.applied is False
    assert result.converged is True
    assert result.resolution_id == record.id


def test_new_version_equal_to_prior_effective_value_converges() -> None:
    record = _record()

    result = resolve_dd_field(
        path=_PATH,
        dd_version="4.2.0",
        field=DDResolutionField.unit,
        raw_value=_value("cm"),
        manifest=_manifest(record),
    )

    assert result.effective.value == "cm"
    assert result.applied is False
    assert result.converged is True


@pytest.mark.parametrize("raw", ["m", "mm"])
def test_new_version_never_reuses_prior_resolution(raw: str) -> None:
    with pytest.raises(DDResolutionVersionMismatch, match="reviewed only"):
        resolve_dd_field(
            path=_PATH,
            dd_version="4.2.0",
            field=DDResolutionField.unit,
            raw_value=_value(raw),
            manifest=_manifest(_record()),
        )


def test_exact_version_third_value_is_stale() -> None:
    with pytest.raises(DDResolutionStale, match="neither"):
        resolve_dd_field(
            path=_PATH,
            dd_version="4.1.0",
            field=DDResolutionField.unit,
            raw_value=_value("mm"),
            manifest=_manifest(_record()),
        )


def test_multiple_prior_convergence_receipts_fail_ambiguous() -> None:
    records = (
        _record(),
        _record(
            dd_version="4.0.0",
            observation_ids=(f"dd_gap_observation:{'c' * 64}",),
            evidence_token=f"dd-gap-evidence:{'d' * 64}",
            resolution_revision=2,
        ),
    )

    with pytest.raises(DDResolutionAmbiguity, match="multiple prior-version"):
        resolve_dd_field(
            path=_PATH,
            dd_version="4.2.0",
            field=DDResolutionField.unit,
            raw_value=_value("cm"),
            manifest=_manifest(*records),
        )


def test_withdrawn_record_never_changes_runtime_value() -> None:
    record = _record(state=DDResolutionStatus.withdrawn)

    result = resolve_dd_field(
        path=_PATH,
        dd_version="4.1.0",
        field=DDResolutionField.unit,
        raw_value=_value("m"),
        manifest=_manifest(record),
    )

    assert result.effective.value == "m"
    assert result.resolution_id is None


def test_context_resolves_nested_grounding_and_marks_projection() -> None:
    raw = RawDDContext(
        path=_PATH,
        dd_version="4.1.0",
        unit="m",
        parents=(
            RawDDContext(
                path="equilibrium/time_slice/global_quantities",
                dd_version="4.1.0",
                documentation="Global quantities.",
            ),
        ),
    )

    result = resolve_dd_context(raw, manifest=_manifest(_record()))
    projected = result.as_pipeline_item()

    assert result.raw is raw
    assert result.unit == "cm"
    assert result.applied_resolution_ids == (_record().id,)
    assert result.parents[0].documentation == "Global quantities."
    assert projected["raw_dd_context"]["unit"] == "m"
    assert projected["unit"] == "cm"
    assert projected["_dd_resolution_marker"] == "resolved-dd-context"
    assert projected["dd_resolution_manifest_digest"] == result.manifest_digest


def test_row_batch_rejects_embedded_version_mismatch() -> None:
    rows = [{"path": _PATH, "dd_version": "4.0.0", "unit": "m"}]

    with pytest.raises(DDResolutionVersionMismatch, match="carries version"):
        resolve_dd_rows(rows, dd_version="4.1.0", manifest=_manifest())
