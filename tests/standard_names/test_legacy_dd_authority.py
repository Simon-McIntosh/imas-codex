"""Regression coverage for singular DD correction authority."""

from importlib import import_module
from inspect import signature
from types import SimpleNamespace

import pytest

from imas_codex.standard_names.dd_resolutions import load_dd_resolution_manifest
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


def test_shipped_carriers_have_zero_shadow_authority() -> None:
    authority = import_module("imas_codex.standard_names.legacy_authority")

    audit = authority.assert_zero_audited_shadow_authorities()

    assert audit.residuals == ()
    assert {
        item.carrier: (item.status, item.residual_count)
        for item in audit.carrier_results
    } == {
        "active_manifest": (authority.ShadowAuditStatus.audited, 0),
        "unit_overrides.override": (authority.ShadowAuditStatus.audited, 0),
        "unit_overrides.skip": (authority.ShadowAuditStatus.audited, 0),
        "dd_unit_exceptions.suppress": (authority.ShadowAuditStatus.audited, 0),
        "dd_unit_exceptions.correct_in_graph": (
            authority.ShadowAuditStatus.audited,
            0,
        ),
        "numeric_missing_unit_fallback": (
            authority.ShadowAuditStatus.not_audited,
            None,
        ),
    }
    numeric = next(
        item
        for item in audit.carrier_results
        if item.carrier == "numeric_missing_unit_fallback"
    )
    assert "separate audited-absence policy" in numeric.reason


def test_public_shadow_audit_has_no_applicability_flag() -> None:
    authority = import_module("imas_codex.standard_names.legacy_authority")

    assert set(signature(authority.find_shadow_authorities).parameters) == {
        "manifest",
        "dd_version",
    }
    assert set(
        signature(authority.assert_zero_audited_shadow_authorities).parameters
    ) == {"manifest", "dd_version"}


def test_shadow_guard_names_residual_carrier_and_row() -> None:
    authority = import_module("imas_codex.standard_names.legacy_authority")
    manifest = load_dd_resolution_manifest()
    duplicate = manifest.resolutions[0]
    double_authority = SimpleNamespace(
        resolutions=(*manifest.resolutions, duplicate),
        state_changes=manifest.state_changes,
    )

    audit = authority.find_shadow_authorities(manifest=double_authority)

    assert [(item.carrier, item.row_id) for item in audit.residuals] == [
        ("active_manifest", duplicate.id)
    ]
    with pytest.raises(authority.DDResolutionShadowAuthority, match=duplicate.id):
        authority.assert_zero_audited_shadow_authorities(manifest=double_authority)


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

    audit = authority.find_shadow_authorities()

    assert [(item.carrier, item.row_id) for item in audit.residuals] == [
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

    audit = authority.find_shadow_authorities()

    assert [(item.carrier, item.row_id) for item in audit.residuals] == [
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

    audit = authority.find_shadow_authorities()

    assert [(item.carrier, item.row_id) for item in audit.residuals] == [
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

    audit = authority.find_shadow_authorities()

    assert [(item.carrier, item.row_id) for item in audit.residuals] == [
        ("dd_unit_exceptions.correct_in_graph", "dd-unit-exception:1")
    ]
