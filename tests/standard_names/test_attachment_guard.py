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


# ---------------------------------------------------------------------------
# Locus <-> source device-compatibility guard
# ---------------------------------------------------------------------------


def test_locus_device_mismatch_rejected() -> None:
    """A camera path may not source a strain-gauge-locus name (zero token
    overlap between a concrete hardware locus and the path)."""
    ok, reason = _is_attachment_consistent(
        "camera_ir/channel/camera/direction/y",
        "y_direction_unit_vector_of_strain_gauge_sensor",
    )
    assert not ok
    assert "locus" in reason.lower()


def test_locus_device_match_accepted() -> None:
    """A camera path sourcing a camera-locus name shares the `camera` token."""
    ok, reason = _is_attachment_consistent(
        "camera_ir/channel/camera/up/x",
        "x_direction_unit_vector_of_camera",
    )
    assert ok, reason


def test_locus_device_hardware_property_accepted() -> None:
    """The intrinsic-property case (`area_of_rogowski_coil`) shares `coil`."""
    ok, reason = _is_attachment_consistent(
        "magnetics/rogowski_coil/area",
        "cross_sectional_area_of_rogowski_coil",
    )
    assert ok, reason


def test_spatial_locus_not_treated_as_hardware() -> None:
    """A spatial-feature locus (magnetic_axis) is not a hardware token — the
    zero-overlap rejection must NOT fire even though the path lacks the token."""
    ok, reason = _is_attachment_consistent(
        "core_profiles/profiles_1d/electrons/temperature",
        "electron_temperature_at_magnetic_axis",
    )
    assert ok, reason


# ---------------------------------------------------------------------------
# Distinct-vector guard — two vector fields of one device node
# ---------------------------------------------------------------------------


def test_distinct_vector_fields_of_one_device_rejected() -> None:
    """`camera/direction/z` may not attach to a name that already sources
    `camera/up/z` — line-of-sight and image-up are DIFFERENT vectors."""
    ok, reason = _is_attachment_consistent(
        "camera_ir/channel/camera/direction/z",
        "z_direction_unit_vector_of_camera",
        existing_sources=["camera_ir/channel/camera/up/z"],
    )
    assert not ok
    assert "vector" in reason.lower()


def test_same_vector_field_siblings_not_flagged() -> None:
    """Two axis leaves of the SAME vector field (direction/z + direction/x)
    are legitimate components — no conflict."""
    ok, reason = _is_attachment_consistent(
        "camera_ir/channel/camera/direction/z",
        "z_direction_unit_vector_of_camera",
        existing_sources=["camera_ir/channel/camera/direction/x"],
    )
    assert ok, reason


def test_distinct_vector_guard_requires_same_axis_leaf() -> None:
    """Different axis leaves (direction/z vs up/x) do not conflict — the guard
    fires only on the SAME leaf axis of a different vector field."""
    ok, reason = _is_attachment_consistent(
        "camera_ir/channel/camera/direction/z",
        "z_direction_unit_vector_of_camera",
        existing_sources=["camera_ir/channel/camera/up/x"],
    )
    assert ok, reason


def test_distinct_vector_guard_requires_common_device() -> None:
    """Same leaf/parent-name but different device grandparent → no conflict."""
    ok, reason = _is_attachment_consistent(
        "camera_ir/channel/camera/direction/z",
        "z_direction_unit_vector_of_camera",
        existing_sources=["ec_launchers/beam/launching_position/direction/z"],
    )
    assert ok, reason
