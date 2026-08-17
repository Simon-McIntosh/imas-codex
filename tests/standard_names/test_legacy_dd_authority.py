"""Regression coverage for singular DD correction authority."""

from importlib import import_module
from types import SimpleNamespace

import pytest

from imas_codex.graph.models import DDResolutionValueKind
from imas_codex.standard_names.dd_resolutions import (
    DDResolutionRecord,
    DDResolutionValue,
    content_addressed_resolution_id,
    dd_resolution_value_hash,
    load_dd_resolution_manifest,
)
from imas_codex.standard_names.unit_overrides import OverrideRule

_ACTIVE_DIRECTION_PATH = "camera_ir/channel/camera/direction/x"


def _replace_legacy_carriers(
    monkeypatch: pytest.MonkeyPatch,
    authority,
    *,
    exceptions: tuple[dict, ...] = (),
    overrides: tuple[OverrideRule, ...] = (),
) -> None:
    monkeypatch.setattr(
        authority,
        "load_exceptions",
        lambda: {"dd_unit_bugs": list(exceptions), "unit_equivalences": []},
    )
    monkeypatch.setattr(authority, "_load_rules", lambda: overrides)


def _manifest_with_unit_transition(
    observed: str | None,
    effective: str | None,
) -> SimpleNamespace:
    base = load_dd_resolution_manifest().resolutions[0]
    observed_value = DDResolutionValue(
        kind=(
            DDResolutionValueKind.null
            if observed is None
            else DDResolutionValueKind.string
        ),
        value=observed,
    )
    effective_value = DDResolutionValue(
        kind=(
            DDResolutionValueKind.null
            if effective is None
            else DDResolutionValueKind.string
        ),
        value=effective,
    )
    payload = base.model_dump()
    payload.update(
        observed=observed_value,
        observed_hash=dd_resolution_value_hash(observed_value),
        effective=effective_value,
    )
    payload["id"] = content_addressed_resolution_id(payload)
    record = DDResolutionRecord.model_validate(payload)
    return SimpleNamespace(resolutions=(record,), state_changes=())


def test_shipped_carriers_have_zero_shadow_authority() -> None:
    authority = import_module("imas_codex.standard_names.legacy_authority")

    assert authority.find_shadow_authorities() == ()


def test_shadow_guard_names_residual_carrier_and_row() -> None:
    authority = import_module("imas_codex.standard_names.legacy_authority")
    manifest = load_dd_resolution_manifest()
    duplicate = manifest.resolutions[0]
    double_authority = SimpleNamespace(
        resolutions=(*manifest.resolutions, duplicate),
        state_changes=manifest.state_changes,
    )

    with pytest.raises(
        authority.DDResolutionShadowAuthority,
        match=r"carrier=active_manifest row_id=dd_resolution:",
    ):
        authority.assert_zero_shadow_authority(manifest=double_authority)


def test_shadow_audit_flags_disagreeing_extraction_skip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority = import_module("imas_codex.standard_names.legacy_authority")
    _replace_legacy_carriers(
        monkeypatch,
        authority,
        overrides=(
            OverrideRule(
                path_pattern=_ACTIVE_DIRECTION_PATH,
                dd_unit="m",
                strategy="skip",
                override_unit=None,
                skip_reason="injected overlap",
                reason="exercise conflicting extraction authority",
            ),
        ),
    )

    shadows = authority.find_shadow_authorities()

    assert [(item.carrier, item.row_id) for item in shadows] == [
        ("unit_overrides.skip", "unit-override:1")
    ]


def test_shadow_audit_flags_disagreeing_extraction_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority = import_module("imas_codex.standard_names.legacy_authority")
    _replace_legacy_carriers(
        monkeypatch,
        authority,
        overrides=(
            OverrideRule(
                path_pattern=_ACTIVE_DIRECTION_PATH,
                dd_unit="m",
                strategy="override",
                override_unit="Pa",
                skip_reason=None,
                reason="exercise conflicting extraction authority",
            ),
        ),
    )

    shadows = authority.find_shadow_authorities()

    assert [(item.carrier, item.row_id) for item in shadows] == [
        ("unit_overrides.override", "unit-override:1")
    ]


def test_shadow_audit_flags_disagreeing_comparator_suppression(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority = import_module("imas_codex.standard_names.legacy_authority")
    _replace_legacy_carriers(
        monkeypatch,
        authority,
        exceptions=(
            {
                "path": _ACTIVE_DIRECTION_PATH,
                "dd_unit": "m",
                "correct_unit": "Pa",
                "reason": "exercise conflicting comparator authority",
            },
        ),
    )

    shadows = authority.find_shadow_authorities()

    assert [(item.carrier, item.row_id) for item in shadows] == [
        ("dd_unit_exceptions.suppress", "dd-unit-exception:1")
    ]


def test_shadow_audit_flags_disagreeing_graph_correction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority = import_module("imas_codex.standard_names.legacy_authority")
    _replace_legacy_carriers(
        monkeypatch,
        authority,
        exceptions=(
            {
                "path": _ACTIVE_DIRECTION_PATH,
                "dd_unit": "m",
                "correct_unit": "Pa",
                "correct_in_graph": True,
                "reason": "exercise conflicting graph authority",
            },
        ),
    )

    shadows = authority.find_shadow_authorities()

    assert [(item.carrier, item.row_id) for item in shadows] == [
        ("dd_unit_exceptions.correct_in_graph", "dd-unit-exception:1")
    ]


@pytest.mark.parametrize("effective", ["1", "Pa"])
def test_typed_unit_projection_preempts_numeric_fallback(
    monkeypatch: pytest.MonkeyPatch,
    effective: str,
) -> None:
    authority = import_module("imas_codex.standard_names.legacy_authority")
    _replace_legacy_carriers(monkeypatch, authority)
    manifest = _manifest_with_unit_transition(None, effective)

    shadows = authority.find_shadow_authorities(
        manifest=manifest,
        numeric_source_applicability=lambda _record: True,
    )

    assert shadows == ()


def test_nonnumeric_source_has_no_numeric_fallback_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority = import_module("imas_codex.standard_names.legacy_authority")
    _replace_legacy_carriers(monkeypatch, authority)
    manifest = _manifest_with_unit_transition("m", None)

    shadows = authority.find_shadow_authorities(
        manifest=manifest,
        numeric_source_applicability=lambda _record: False,
    )

    assert shadows == ()


def test_reachable_numeric_fallback_is_reported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority = import_module("imas_codex.standard_names.legacy_authority")
    _replace_legacy_carriers(monkeypatch, authority)
    manifest = _manifest_with_unit_transition("m", None)

    shadows = authority.find_shadow_authorities(
        manifest=manifest,
        numeric_source_applicability=lambda _record: True,
    )

    assert [(item.carrier, item.row_id) for item in shadows] == [
        ("numeric_missing_unit_fallback", "numeric-no-unit")
    ]
