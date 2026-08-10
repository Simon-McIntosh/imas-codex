from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from imas_codex.graph.models import (
    DDResolutionField,
    DDResolutionStatus,
    DDResolutionValueKind,
)
from imas_codex.standard_names import dd_resolutions as dd_resolution_module
from imas_codex.standard_names.dd_resolutions import (
    DDResolutionAmbiguity,
    DDResolutionCandidate,
    DDResolutionCandidateDisposition,
    DDResolutionCandidateManifest,
    DDResolutionCollision,
    DDResolutionEvidenceMismatch,
    DDResolutionManifest,
    DDResolutionManifestInvalid,
    DDResolutionRecord,
    DDResolutionStale,
    DDResolutionValue,
    DDResolutionVersionMismatch,
    RawDDContext,
    content_addressed_resolution_id,
    dd_resolution_value_hash,
    load_dd_resolution_candidates_for_review,
    load_dd_resolution_manifest,
    resolve_dd_context,
    resolve_dd_field,
    resolve_dd_rows,
)

_PATH = "equilibrium/time_slice/global_quantities/magnetic_axis/r"
_OBSERVATION_ID = f"dd_gap_observation:{'a' * 64}"
_EVIDENCE_TOKEN = f"dd-gap-evidence:{'b' * 64}"
_CANDIDATE_RESOURCE = (
    Path(__file__).parents[2]
    / "imas_codex"
    / "standard_names"
    / "config"
    / "dd_resolution_candidates.yaml"
)


def _candidate_content() -> str:
    return _CANDIDATE_RESOURCE.read_text()


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


def test_packaged_candidates_are_separate_review_only_input() -> None:
    active_before = load_dd_resolution_manifest()
    review_input = load_dd_resolution_candidates_for_review()
    active_after = load_dd_resolution_manifest()

    assert review_input.authority == "review_input_only"
    assert len(review_input.candidates) == 21
    assert active_after is active_before
    assert active_after.resolutions == ()
    assert active_after.digest == active_before.digest
    for candidate in review_input.candidates:
        with pytest.raises(ValidationError):
            DDResolutionRecord.model_validate(candidate.model_dump(mode="json"))


def test_candidate_type_has_no_activation_or_fresh_evidence_fields() -> None:
    assert set(DDResolutionCandidate.model_fields).isdisjoint(
        {
            "approval_receipt",
            "approved_at",
            "approved_by",
            "evidence_token",
            "gap_id",
            "id",
            "reason",
            "resolution_revision",
            "state",
        }
    )


@pytest.mark.parametrize(
    ("needle", "replacement"),
    [
        (
            "authority: review_input_only\n",
            "authority: review_input_only\nauthority: review_input_only\n",
        ),
        (
            "upstream_changes:\n",
            "upstream_changes:\n  geometric-vector-component-units: {}\n",
        ),
        (
            "    status: merged\n",
            "    status: merged\n    status: merged\n",
        ),
        (
            "  - source_row: U11\n",
            "  - source_row: U11\n    source_row: U11\n",
        ),
    ],
)
def test_candidate_parser_rejects_duplicate_keys_at_every_mapping_depth(
    needle: str,
    replacement: str,
) -> None:
    content = _candidate_content()
    assert content.count(needle) >= 1

    with pytest.raises(DDResolutionManifestInvalid, match="duplicate key") as caught:
        dd_resolution_module._parse_candidate_content(
            content.replace(needle, replacement, 1)
        )

    assert isinstance(caught.value.__cause__, yaml.constructor.ConstructorError)


def test_candidate_parser_rejects_unknown_fields() -> None:
    content = _candidate_content().replace(
        "authority: review_input_only\n",
        "authority: review_input_only\nunknown_authority_field: forbidden\n",
        1,
    )

    with pytest.raises(DDResolutionManifestInvalid) as caught:
        dd_resolution_module._parse_candidate_content(content)

    cause = caught.value.__cause__
    assert isinstance(cause, ValidationError)
    assert any(error["type"] == "extra_forbidden" for error in cause.errors())


@pytest.mark.parametrize(
    ("needle", "replacement"),
    [
        ("schema_version: 1\n", 'schema_version: "1"\n'),
        ("schema_version: 1\n", "schema_version: 1.0\n"),
        ("schema_version: 1\n", "schema_version: true\n"),
        (
            "    source_release_match_count: 12\n",
            '    source_release_match_count: "12"\n',
        ),
        (
            "    source_release_match_count: 12\n",
            "    source_release_match_count: 12.0\n",
        ),
        (
            "    source_release_match_count: 12\n",
            "    source_release_match_count: true\n",
        ),
        (
            "    narrow_evidence_overlap_count: 6\n",
            '    narrow_evidence_overlap_count: "6"\n',
        ),
        (
            "    narrow_evidence_overlap_count: 6\n",
            "    narrow_evidence_overlap_count: 6.0\n",
        ),
        (
            "    narrow_evidence_overlap_count: 6\n",
            "    narrow_evidence_overlap_count: false\n",
        ),
    ],
)
def test_candidate_parser_rejects_coercible_authority_integers(
    needle: str,
    replacement: str,
) -> None:
    content = _candidate_content()
    assert content.count(needle) >= 1

    with pytest.raises(DDResolutionManifestInvalid) as caught:
        dd_resolution_module._parse_candidate_content(
            content.replace(needle, replacement, 1)
        )

    cause = caught.value.__cause__
    assert isinstance(cause, ValidationError)
    assert any(error["type"] == "int_type" for error in cause.errors())


def test_candidate_resource_preserves_exact_upstream_change_state() -> None:
    review_input = load_dd_resolution_candidates_for_review()
    changes = review_input.upstream_changes

    assert set(changes) == {
        "geometric-vector-component-units",
        "ionization-potential-units",
        "neutral-energy-flux-wiring",
        "reconstructed-constraint-units",
    }
    assert changes["geometric-vector-component-units"].status.value == "merged"
    assert changes["geometric-vector-component-units"].merge_commit == (
        "cb0d86de388dbbdf62acca36de7b7f8c62bb9889"
    )
    assert changes["geometric-vector-component-units"].solution_commits == (
        "fd0c145cb897770738c20de4a426c27b2d8d1a2d",
        "721638233cd87f5ca3f9e71b36d66c46e146af2e",
    )
    assert changes["neutral-energy-flux-wiring"].status.value == "open"
    assert changes["neutral-energy-flux-wiring"].merge_commit is None
    assert changes["neutral-energy-flux-wiring"].solution_commits == (
        "f34c85d33497f2bd777db7eaf0f6fb93fddc66f2",
    )
    assert changes["ionization-potential-units"].status.value == "open"
    assert changes["ionization-potential-units"].proposed_change_dd_version == ("4.2.0")
    assert changes["ionization-potential-units"].solution_commits == (
        "30a5ddd4b7037b9f93a8f00f7837809403349d99",
    )
    assert changes["reconstructed-constraint-units"].status.value == "merged"
    assert changes["reconstructed-constraint-units"].merge_commit == (
        "d07172e814e91900cb4ed5d0b5f41547be3eef90"
    )
    assert changes["reconstructed-constraint-units"].solution_commits == (
        "35c146031bf98028911b8266d286dcdf6ee85e2e",
    )
    assert all(change.fixed_dd_version is None for change in changes.values())


def test_candidate_resource_is_exactly_row_bounded() -> None:
    review_input = load_dd_resolution_candidates_for_review()
    by_row = {candidate.source_row: candidate for candidate in review_input.candidates}

    assert set(by_row) == {
        "U11",
        "U12",
        "U13",
        "U14",
        "U15",
        "U16",
        "U19",
        "U21",
        "U22",
        "U25",
        "U26",
        "U27",
        "U28",
        "U29",
        "U32",
        "O17",
        "O20",
        "O21",
        "O22",
        "O23",
        "O24",
    }
    assert by_row["U19"].source_release_match_count == 14
    assert by_row["U19"].exact_paths == (
        "edge_profiles/ggd/ion/state/ionisation_potential",
        "edge_profiles/ggd/ion/state/ionisation_potential/coefficients",
        "edge_profiles/ggd/ion/state/ionisation_potential/values",
        "plasma_profiles/ggd/ion/state/ionisation_potential",
        "plasma_profiles/ggd/ion/state/ionisation_potential/coefficients",
        "plasma_profiles/ggd/ion/state/ionisation_potential/values",
    )

    held = {
        "O20": (1188, 6),
        "O21": (36, 12),
        "O22": (9, 3),
        "O23": (9, 3),
        "O24": (9, 3),
    }
    for row, (release_count, overlap_count) in held.items():
        candidate = by_row[row]
        assert (
            candidate.disposition == DDResolutionCandidateDisposition.broad_scope_hold
        )
        assert candidate.exact_paths == ()
        assert candidate.source_release_match_count == release_count
        assert candidate.narrow_evidence_overlap_count == overlap_count


def test_candidate_resource_preserves_contrary_semantic_boundaries() -> None:
    review_input = load_dd_resolution_candidates_for_review()
    exact_paths = {
        path for candidate in review_input.candidates for path in candidate.exact_paths
    }
    source_rows = {candidate.source_row for candidate in review_input.candidates}

    assert "ec_launchers/beam/direction/kphi" not in exact_paths
    assert not any(path.endswith("/position/psi") for path in exact_paths)
    assert "U17" not in source_rows
    assert "U33" not in source_rows
    assert not source_rows.intersection({"O12", "O13", "O14", "O15"})


def test_candidate_manifest_requires_every_missing_authority_field() -> None:
    review_input = load_dd_resolution_candidates_for_review()
    payload = review_input.model_dump(mode="json")
    payload["missing_requirements"].remove("approval_receipt")

    with pytest.raises(ValidationError, match="missing activation requirement"):
        DDResolutionCandidateManifest.model_validate(payload)


def test_broad_candidate_cannot_silently_gain_exact_paths() -> None:
    review_input = load_dd_resolution_candidates_for_review()
    held = next(
        candidate
        for candidate in review_input.candidates
        if candidate.disposition == DDResolutionCandidateDisposition.broad_scope_hold
    )
    payload = held.model_dump(mode="json")
    payload["exact_paths"] = [_PATH]

    with pytest.raises(ValidationError, match="cannot enumerate candidate paths"):
        DDResolutionCandidate.model_validate(payload)


def test_runtime_resolvers_never_consult_candidate_resource(monkeypatch) -> None:
    active_digest = load_dd_resolution_manifest().digest

    def _candidate_access_is_forbidden() -> None:
        raise AssertionError("runtime attempted to load review-only candidates")

    monkeypatch.setattr(
        dd_resolution_module,
        "load_dd_resolution_candidates_for_review",
        _candidate_access_is_forbidden,
    )
    field = resolve_dd_field(
        path="camera_ir/channel/camera/direction/x",
        dd_version="4.1.1",
        field=DDResolutionField.unit,
        raw_value=_value("m"),
    )
    context = resolve_dd_context(
        RawDDContext(
            path="camera_ir/channel/camera/direction/x",
            dd_version="4.1.1",
            unit="m",
        )
    )
    rows = resolve_dd_rows(
        [{"path": "camera_ir/channel/camera/direction/x", "unit": "m"}],
        dd_version="4.1.1",
    )

    assert field.effective.value == "m"
    assert field.applied is False
    assert context.unit == "m"
    assert context.applied_resolution_ids == ()
    assert rows[0].unit == "m"
    assert context.manifest_digest == active_digest
    assert rows[0].manifest_digest == active_digest
    assert not any("candidate" in key for key in context.as_pipeline_item())


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
