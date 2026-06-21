"""Tests for the tense-consistency guard on LLM-proposed attachments."""

import pytest

from imas_codex.standard_names.workers import _is_attachment_consistent


@pytest.mark.parametrize(
    "source_id,sn_name",
    [
        ("core_profiles/profiles_1d/electrons/density", "electron_density"),
        (
            "core_instant_changes/change/profiles_1d/electrons/density",
            "change_in_electron_density",
        ),
        (
            "core_profiles/profiles_1d/electrons/temperature",
            "electron_temperature",
        ),
        (
            "core_instant_changes/change/profiles_1d/electrons/temperature",
            "tendency_of_electron_temperature",
        ),
        (
            "equilibrium/time_slice/global_quantities/ip",
            "rate_of_change_of_plasma_current",
        ),  # rate path heuristic relies on SN prefix only — base path + rate SN flagged
    ],
)
def test_consistent_pairs(source_id: str, sn_name: str) -> None:
    ok, reason = _is_attachment_consistent(source_id, sn_name)
    if sn_name.startswith(
        ("change_in_", "tendency_of_", "rate_of_", "time_derivative_of_")
    ):
        # Path must contain a change/tendency token to pass.
        if any(
            tok in source_id
            for tok in ("instant_changes", "/change", "_delta", "tendency_")
        ):
            assert ok, reason
        else:
            assert not ok, "rate/change SN with base path must be rejected"
    else:
        assert ok, reason


@pytest.mark.parametrize(
    "source_id,sn_name",
    [
        # ``d_dt`` rate-marker paths: the time-derivative IDS structures expose
        # the rate explicitly. A ``time_derivative_of_X`` SN MUST attach.
        (
            "transport_solver_numerics/derivatives_1d/electrons/d_dt/pressure",
            "time_derivative_of_electron_pressure",
        ),
        (
            "transport_solver_numerics/derivatives_1d/d_dt/ion_density",
            "tendency_of_ion_density",
        ),
        # A ``_dt`` suffixed leaf (e.g. ``..._dt``) is also a rate marker.
        (
            "core_profiles/profiles_1d/electrons/temperature_dt",
            "time_derivative_of_electron_temperature",
        ),
    ],
)
def test_rate_marker_paths_accept_rate_names(source_id: str, sn_name: str) -> None:
    """A ``d_dt`` / ``derivatives_1d`` rate path matches a rate SN."""
    ok, reason = _is_attachment_consistent(source_id, sn_name)
    assert ok, reason


@pytest.mark.parametrize(
    "source_id,sn_name",
    [
        # A non-rate base path with a rate SN is STILL flagged: the d_dt fix
        # must not make every path accept a rate name.
        (
            "core_profiles/profiles_1d/electrons/pressure",
            "time_derivative_of_electron_pressure",
        ),
        (
            "core_profiles/profiles_1d/electrons/temperature",
            "tendency_of_electron_temperature",
        ),
    ],
)
def test_non_rate_path_still_rejects_rate_name(source_id: str, sn_name: str) -> None:
    """A base-quantity path with no rate marker rejects a rate SN."""
    ok, reason = _is_attachment_consistent(source_id, sn_name)
    assert not ok
    assert "tense mismatch" in reason


@pytest.mark.parametrize(
    "source_id,sn_name",
    [
        # Base path → change SN: must be rejected.
        ("core_profiles/profiles_1d/electrons/density", "change_in_electron_density"),
        (
            "core_profiles/profiles_1d/electrons/temperature",
            "tendency_of_electron_temperature",
        ),
        (
            "equilibrium/time_slice/global_quantities/ip",
            "rate_of_change_of_plasma_current",
        ),
        # Change path → base SN: must be rejected.
        (
            "core_instant_changes/change/profiles_1d/electrons/density",
            "electron_density",
        ),
        (
            "core_instant_changes/change/global_quantities/ip",
            "plasma_current",
        ),
    ],
)
def test_inconsistent_pairs_are_rejected(source_id: str, sn_name: str) -> None:
    ok, reason = _is_attachment_consistent(source_id, sn_name)
    assert not ok
    assert "tense mismatch" in reason
