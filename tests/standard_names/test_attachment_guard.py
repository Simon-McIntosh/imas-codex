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
    ],
)
def test_consistent_pairs(source_id: str, sn_name: str) -> None:
    ok, reason = _is_attachment_consistent(source_id, sn_name)
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
        # The DD's ``d<quantity>_dt`` leaf is also a rate marker: the leading
        # ``d`` is the differential of the quantity, ``_dt`` the denominator.
        (
            "transport_solver_numerics/derivatives_1d/dpsi_dt",
            "time_derivative_of_poloidal_magnetic_flux",
        ),
        (
            "summary/global_quantities/denergy_thermal_dt/value",
            "time_derivative_of_thermal_stored_energy",
        ),
        (
            "runaway_electrons/profiles_1d/ddensity_dt_total",
            "time_derivative_of_runaway_electron_density",
        ),
    ],
)
def test_rate_marker_paths_accept_rate_names(source_id: str, sn_name: str) -> None:
    """A ``d_dt`` / ``d<quantity>_dt`` rate path matches a rate SN."""
    ok, reason = _is_attachment_consistent(source_id, sn_name)
    assert ok, reason


@pytest.mark.parametrize(
    "sn_name",
    [
        "time_derivative_of_electron_density",
        "tendency_of_electron_density",
        "change_in_electron_density",
        "volume_averaged_time_derivative_of_electron_density",
        "volume_integrated_time_derivative_of_electron_density",
        "flux_surface_averaged_time_derivative_of_electron_density",
        "difference_of_time_derivative_of_electron_density_and_ion_density",
        "product_of_time_derivative_of_electron_density_and_volume",
        "ratio_of_time_derivative_of_electron_density_to_ion_density",
    ],
)
def test_recursive_time_change_shapes_match_a_derivative_path(sn_name: str) -> None:
    """Every supported time-change expression is found anywhere in public IR."""
    ok, reason = _is_attachment_consistent(
        "summary/volume_average/dn_e_dt/value", sn_name
    )
    assert ok, reason


def test_nested_time_derivative_still_rejects_a_plain_path() -> None:
    ok, reason = _is_attachment_consistent(
        "summary/volume_average/n_e/value",
        "volume_averaged_time_derivative_of_electron_density",
    )
    assert not ok
    assert "tense mismatch" in reason


def test_non_time_operator_does_not_claim_a_derivative() -> None:
    ok, reason = _is_attachment_consistent(
        "summary/volume_average/dn_e_dt/value",
        "volume_averaged_electron_density",
    )
    assert not ok
    assert "tense mismatch" in reason


def test_parse_failure_retains_conservative_lexical_fallback() -> None:
    ok, reason = _is_attachment_consistent(
        "summary/volume_average/dn_e_dt/value",
        "time_derivative_of_unregistered_quantity",
    )
    assert ok, reason


@pytest.mark.parametrize(
    "sn_name",
    [
        "time_derivative_of_electron_density",
        "gradient_of_time_derivative_of_electron_temperature",
        "difference_of_time_derivative_of_pressure_and_temperature",
        "difference_of_change_in_electron_density_and_ion_density",
        "change_in_electron_density",
    ],
)
def test_public_parser_ir_exposes_time_change_semantics(sn_name: str) -> None:
    """ISN semantics apply across direct, nested, binary, and qualifier IR."""
    from imas_standard_names import parse

    from imas_codex.standard_names.workers import _ir_denotes_time_change

    assert _ir_denotes_time_change(parse(sn_name, strict=True).ir)


@pytest.mark.parametrize(
    "sn_name",
    [
        "gradient_of_electron_temperature",
        "time_averaged_electron_density",
        "ratio_of_electron_density_to_ion_density",
    ],
)
def test_public_parser_ir_rejects_non_time_operator_semantics(sn_name: str) -> None:
    from imas_standard_names import parse

    from imas_codex.standard_names.workers import _ir_denotes_time_change

    assert not _ir_denotes_time_change(parse(sn_name, strict=True).ir)


# ---------------------------------------------------------------------------
# ``_dt`` is overloaded: time denominator vs deuterium-tritium species
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "source_id,sn_name",
    [
        # The DD uses ``_dt`` for the deuterium-tritium REACTION as well as for
        # a time denominator. These leaves carry no leading differential ``d``,
        # so they are plain base quantities of the D-T reaction.
        (
            "neutron_diagnostic/reconstructed_emissivity/emissivity_dt",
            "deuterium_tritium_emissivity_due_to_fusion",
        ),
        (
            "neutron_diagnostic/reconstructed_emissivity/fusion_power_dt",
            "deuterium_tritium_power_density_due_to_fusion",
        ),
    ],
)
def test_deuterium_tritium_suffix_is_not_a_time_derivative(
    source_id: str, sn_name: str
) -> None:
    """A ``<quantity>_dt`` species leaf is a base quantity, not a rate."""
    ok, reason = _is_attachment_consistent(source_id, sn_name)
    assert ok, reason


@pytest.mark.parametrize(
    "source_id",
    [
        # ``d<quantity>_dt`` — the differentiated quantity carries the leading d.
        "ntms/time_slice/mode/dphase_dt",
        "transport_solver_numerics/derivatives_1d/dpsi_dt",
        "summary/line_average/dn_e_dt/value",
        # ``d_dt`` container segment.
        "transport_solver_numerics/derivatives_1d/electrons/d_dt/pressure",
    ],
)
def test_time_derivative_paths_still_reject_base_names(source_id: str) -> None:
    """The rate marker must keep firing on the DD's genuine derivative form."""
    ok, reason = _is_attachment_consistent(source_id, "electron_pressure")
    assert not ok
    assert "tense mismatch" in reason


# ---------------------------------------------------------------------------
# ``derivatives_1d`` is a container, not a per-leaf rate marker
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "source_id,sn_name",
    [
        # The radial grid the derivatives are computed ON is plain geometry:
        # ``grid/area`` is documented "Cross-sectional area of the flux surface".
        (
            "transport_solver_numerics/derivatives_1d/grid/area",
            "area_of_flux_surface",
        ),
        (
            "transport_solver_numerics/derivatives_1d/grid/volume",
            "volume_of_flux_surface",
        ),
        # Species metadata inside the container is likewise not a rate.
        (
            "transport_solver_numerics/derivatives_1d/ion/z_ion",
            "ion_charge_number",
        ),
    ],
)
def test_container_leaf_without_a_rate_marker_is_a_base_quantity(
    source_id: str, sn_name: str
) -> None:
    """A leaf under ``derivatives_1d`` is only a rate when it says so itself."""
    ok, reason = _is_attachment_consistent(source_id, sn_name)
    assert ok, reason


# ---------------------------------------------------------------------------
# Rate-ness reads from the name's BASE as well as from a leading prefix
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "source_id,sn_name",
    [
        (
            "runaway_electrons/profiles_1d/ddensity_dt_total",
            "fast_electron_source_rate",
        ),
        (
            "runaway_electrons/ggd_fluid/ddensity_dt_compton/values",
            "runaway_electron_source_rate",
        ),
        (
            "runaway_electrons/profiles_1d/ddensity_dt_dreicer",
            "fast_electron_source_rate_due_to_dreicer",
        ),
        (
            "ntms/time_slice/mode/dphase_dt",
            "rotation_frequency_of_neoclassical_tearing_mode",
        ),
        (
            "ntms/time_slice/mode/detailed_evolution/dwidth_dt",
            "growth_rate_of_neoclassical_tearing_mode_width",
        ),
    ],
)
def test_rate_natured_base_absorbs_a_derivative_path(
    source_id: str, sn_name: str
) -> None:
    """A ``…_source_rate`` / ``…_frequency`` name IS a rate quantity.

    The name expresses rate-ness through its base token, not only through a
    ``time_derivative_of_`` prefix, so a genuine derivative path is a
    consistent source for it.
    """
    ok, reason = _is_attachment_consistent(source_id, sn_name)
    assert ok, reason


def test_rate_natured_base_does_not_demand_a_derivative_path() -> None:
    """A rate-natured base is not a CLAIM that the source differentiates.

    A frequency measured directly is a base quantity of its own — the
    rate-natured base must not force a derivative path on the source side.
    """
    ok, reason = _is_attachment_consistent(
        "ec_launchers/beam/frequency", "frequency_of_electron_cyclotron_beam"
    )
    assert ok, reason


def test_rate_word_inside_another_token_is_not_a_rate_base() -> None:
    """``substrate`` contains ``rate`` — token boundaries must be respected."""
    ok, reason = _is_attachment_consistent(
        "core_instant_changes/change/profiles_1d/electrons/temperature",
        "substrate_temperature",
    )
    assert not ok
    assert "tense mismatch" in reason


def test_derivative_claiming_name_on_a_plain_path_still_rejected() -> None:
    """A name claiming a time derivative of a plain intensity is still invalid."""
    ok, reason = _is_attachment_consistent(
        "bremsstrahlung_visible/channel/intensity",
        "time_derivative_of_bremsstrahlung_count_at_detector_pixel",
    )
    assert not ok
    assert "tense mismatch" in reason


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
# State resolution — a state-resolved path needs a state-resolved name
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "source_id,sn_name",
    [
        # The grammar carries state resolution in a dedicated ``state`` segment
        # whose tokens name the KIND of state, so an ``internal_state`` name is
        # state-resolved and its ``/state/`` source is a consistent one.
        (
            "core_profiles/profiles_1d/neutral/state/density",
            "neutral_internal_state_density",
        ),
        (
            "core_profiles/profiles_1d/neutral/state/temperature",
            "neutral_internal_state_temperature",
        ),
        (
            "core_profiles/profiles_1d/neutral/state/density_fast",
            "fast_neutral_internal_state_number_density",
        ),
        (
            "core_transport/model/profiles_1d/neutral/state/energy/d",
            "effective_neutral_internal_state_energy_diffusivity",
        ),
        (
            "edge_profiles/ggd/neutral/state/velocity_diamagnetic/diamagnetic",
            "effective_neutral_internal_state_velocity_due_to_diamagnetic_drift",
        ),
        (
            "edge_transport/model/ggd/neutral/state/energy/flux/values",
            "neutral_internal_state_energy_flux",
        ),
        (
            "wall/description_ggd/ggd/energy_fluxes/kinetic/neutral/state/emitted/values",
            "neutral_internal_state_particle_flux_at_wall",
        ),
        # The subject tokens that fold the state into the species keep working.
        (
            "core_profiles/profiles_1d/ion/state/z_min",
            "ion_state_minimum_charge_number",
        ),
        (
            "core_transport/model/profiles_1d/ion/state/particles/d",
            "radial_ion_charge_state_diffusion_coefficient",
        ),
    ],
)
def test_state_resolved_name_accepts_a_state_resolved_path(
    source_id: str, sn_name: str
) -> None:
    """Any grammar token naming a state marks the name as state-resolved."""
    ok, reason = _is_attachment_consistent(source_id, sn_name)
    assert ok, reason


@pytest.mark.parametrize(
    "source_id,sn_name",
    [
        # The name claims a state; the path stops at the element/species level.
        (
            "edge_transport/model/ggd/ion/element/atoms_n",
            "atomic_count_of_ion_state",
        ),
        (
            "core_transport/model/profiles_1d/ion/particles/d",
            "radial_ion_charge_state_diffusion_coefficient",
        ),
        (
            "core_profiles/profiles_1d/neutral/density",
            "neutral_internal_state_density",
        ),
    ],
)
def test_state_resolved_name_rejects_a_species_level_path(
    source_id: str, sn_name: str
) -> None:
    """A state-resolved name may not source a path that resolves no state."""
    ok, reason = _is_attachment_consistent(source_id, sn_name)
    assert not ok
    assert "state-resolution mismatch" in reason


@pytest.mark.parametrize(
    "source_id,sn_name",
    [
        # ``z_max`` is the upper bound of the charge-state BUNDLE the state entry
        # represents: the same species carries a different value in every state
        # entry, so a name with no state qualifier cannot say which it means.
        ("core_profiles/profiles_1d/ion/state/z_max", "ion_upper_bound_charge_number"),
        ("core_profiles/profiles_1d/neutral/state/density", "neutral_density"),
        ("edge_profiles/ggd/ion/state/energy_density_kinetic", "energy_density"),
    ],
)
def test_state_resolved_path_rejects_a_species_level_name(
    source_id: str, sn_name: str
) -> None:
    """A ``/state/`` path describes ONE state, not the species it sits under."""
    ok, reason = _is_attachment_consistent(source_id, sn_name)
    assert not ok
    assert "state-resolution mismatch" in reason


def test_steady_state_is_not_state_resolution() -> None:
    """``steady_state`` is a regime, not a species state.

    The state tokens are matched as whole grammar tokens, so a word that merely
    ends in ``state`` never gates the rule.
    """
    ok, reason = _is_attachment_consistent(
        "equilibrium/time_slice/global_quantities/ip", "steady_state_plasma_current"
    )
    assert ok, reason


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


def test_geometry_primitive_alternatives_still_conflict() -> None:
    """Alternative geometry primitives of one object are DIFFERENT quantities.

    ``rectangle/r`` is the centre of the rectangle while ``oblique/r`` is a
    reference corner — not two samples of one coordinate, so they must not
    collapse onto one name.
    """
    ok, reason = _is_attachment_consistent(
        "ferritic/object/axisymmetric/rectangle/r",
        "radial_coordinate_of_ferritic_object",
        existing_sources=["ferritic/object/axisymmetric/oblique/r"],
    )
    assert not ok
    assert "vector" in reason.lower()


# ---------------------------------------------------------------------------
# Ordinal samples of ONE object share a name — ordinality never enters it
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "source_id,existing,sn_name",
    [
        # The DD samples one line of sight at successive points.
        (
            "bolometer/channel/line_of_sight/second_point/r",
            "bolometer/channel/line_of_sight/first_point/r",
            "radial_coordinate_of_line_of_sight",
        ),
        (
            "ece/channel/line_of_sight/third_point/z",
            "ece/channel/line_of_sight/first_point/z",
            "vertical_coordinate_of_line_of_sight",
        ),
        # One conductor path sampled at its element start/end/centre positions.
        (
            "coils_non_axisymmetric/coil/conductor/elements/end_points/z",
            "coils_non_axisymmetric/coil/conductor/elements/start_points/z",
            "vertical_coordinate_of_conductor_element",
        ),
        (
            "coils_non_axisymmetric/coil/conductor/elements/centres/phi",
            "coils_non_axisymmetric/coil/conductor/elements/intermediate_points/phi",
            "toroidal_angle_of_conductor_element",
        ),
        # A thick line is delimited by two points of the same object.
        (
            "ferritic/object/axisymmetric/thick_line/second_point/r",
            "ferritic/object/axisymmetric/thick_line/first_point/r",
            "radial_coordinate_of_ferritic_object",
        ),
    ],
)
def test_ordinal_point_samples_of_one_object_share_a_name(
    source_id: str, existing: str, sn_name: str
) -> None:
    """Successive samples of one geometric object are one quantity.

    The point index is a position along the sampled object, not a different
    physical quantity, so it never enters the standard name and the guard must
    let every sample attach to the same name.
    """
    ok, reason = _is_attachment_consistent(
        source_id, sn_name, existing_sources=[existing]
    )
    assert ok, reason


def test_point_sample_versus_vector_field_still_conflicts() -> None:
    """A sampled position and a direction vector are different quantities."""
    ok, reason = _is_attachment_consistent(
        "camera_ir/channel/camera/direction/z",
        "z_direction_unit_vector_of_camera",
        existing_sources=["camera_ir/channel/camera/first_point/z"],
    )
    assert not ok
    assert "vector" in reason.lower()
