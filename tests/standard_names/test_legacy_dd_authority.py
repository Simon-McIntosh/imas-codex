"""Regression coverage for singular DD correction authority."""

from importlib import import_module
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
