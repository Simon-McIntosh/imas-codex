"""Regression coverage for singular DD correction authority."""

from importlib import import_module
from types import SimpleNamespace

import pytest

from imas_codex.standard_names.dd_resolutions import load_dd_resolution_manifest


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
