"""Dimensionality rule and ISN-derived shape bases in the attachment guard.

The lexical rules of ``_is_attachment_consistent`` (tense, state resolution,
shape-parameter surface, distinct-vector, locus/device) are covered in
``test_attachment_guard.py``; this module covers the unit-dimensionality rule
and the ISN derivation of the shape-parameter base set.
"""

import pytest

from imas_codex.standard_names.workers import (
    _is_attachment_consistent,
    _shape_parameter_bases,
)

# ---------------------------------------------------------------------------
# Dimensionality rule — a source may not attach to a dimensionally different name
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "source_id,sn_name,dd_unit,sn_unit",
    [
        # A flux-surface metric declared dimensionless in the DD may not source
        # a name that declares an inverse-area metric.
        (
            "equilibrium/time_slice/profiles_1d/gm3",
            "radial_flux_surface_averaged_metric",
            "1",
            "m^-2",
        ),
        # T^-2 is not T — a squared-reciprocal-field metric is not a field.
        (
            "equilibrium/time_slice/profiles_1d/gm4",
            "flux_surface_averaged_magnetic_field",
            "T^-2",
            "T",
        ),
        # An area-per-flux derivative is not a length.
        (
            "equilibrium/time_slice/profiles_1d/darea_dpsi",
            "derivative_with_respect_to_toroidal_flux_radius_of_area_of_flux_surface",
            "Wb^-1.m^2",
            "m",
        ),
    ],
)
def test_dimensional_disagreement_rejected(
    source_id: str, sn_name: str, dd_unit: str, sn_unit: str
) -> None:
    ok, reason = _is_attachment_consistent(
        source_id, sn_name, dd_unit=dd_unit, sn_unit=sn_unit
    )
    assert not ok
    assert "unit" in reason.lower()
    assert dd_unit in reason and sn_unit in reason


@pytest.mark.parametrize(
    "source_id,sn_name,dd_unit,sn_unit",
    [
        # Identical units.
        (
            "core_profiles/profiles_1d/electrons/density",
            "electron_density",
            "m^-3",
            "m^-3",
        ),
        # Recorded physical equivalence — spelling the canonical formatter keeps
        # distinct, not a real disagreement.
        (
            "ec_launchers/beam/frequency",
            "frequency_of_electron_cyclotron_beam",
            "Hz",
            "s^-1",
        ),
    ],
)
def test_dimensionally_compatible_accepted(
    source_id: str, sn_name: str, dd_unit: str, sn_unit: str
) -> None:
    ok, reason = _is_attachment_consistent(
        source_id, sn_name, dd_unit=dd_unit, sn_unit=sn_unit
    )
    assert ok, reason


@pytest.mark.parametrize(
    "source_id,sn_name,dd_unit,sn_unit",
    [
        # Direction unit-vector components declared metre by the DD: a filed
        # DD defect, the standard name correctly carries dimensionless.
        (
            "camera_ir/channel/camera/direction/x",
            "x_direction_unit_vector_of_camera",
            "m",
            "1",
        ),
        (
            "camera_ir/channel/camera/up/z",
            "z_image_up_unit_vector_of_camera",
            "m",
            "1",
        ),
        # Charge NUMBER fields declared with the elementary-charge unit.
        ("core_profiles/profiles_1d/ion/z_ion", "ion_charge_number", "e", "1"),
        (
            "core_profiles/profiles_1d/ion/state/z_min",
            "ion_state_minimum_charge_number",
            "e",
            "1",
        ),
        (
            "core_profiles/profiles_1d/ion/state/z_max",
            "ion_state_maximum_charge_number",
            "e",
            "1",
        ),
    ],
)
def test_registry_excepted_dd_defect_not_rejected(
    source_id: str, sn_name: str, dd_unit: str, sn_unit: str
) -> None:
    """A recorded DD-side unit bug is the DD's fault, not the name's.

    The exception route is the existing ``dd_unit_exceptions.yaml`` registry
    consulted through ``units_agree`` — the guard adds no second list.
    """
    ok, reason = _is_attachment_consistent(
        source_id, sn_name, dd_unit=dd_unit, sn_unit=sn_unit
    )
    assert ok, reason


@pytest.mark.parametrize(
    "dd_unit,sn_unit",
    [
        (None, "m^-2"),  # DD declares no unit — a DD-completeness gap
        ("1", None),  # name carries no unit yet
        (None, None),
        ("mixed", "m"),  # DD sentinel the canonical parser cannot resolve
        ("as parent", "m"),
    ],
)
def test_unknown_unit_never_rejects(dd_unit: str | None, sn_unit: str | None) -> None:
    """With nothing comparable on one side there is no disagreement to act on.

    An unresolvable or absent unit is a DD-completeness problem surfaced
    elsewhere; treating it as an attachment defect would detach sound edges.
    """
    ok, reason = _is_attachment_consistent(
        "equilibrium/time_slice/profiles_1d/gm3",
        "radial_flux_surface_averaged_metric",
        dd_unit=dd_unit,
        sn_unit=sn_unit,
    )
    assert ok, reason


def test_units_default_to_unsupplied() -> None:
    """Callers with no unit context keep the pre-existing lexical behaviour."""
    ok, _ = _is_attachment_consistent(
        "equilibrium/time_slice/profiles_1d/gm3", "radial_flux_surface_averaged_metric"
    )
    assert ok


def test_lexical_rejection_precedes_unit_agreement() -> None:
    """A semantic mis-attachment is still rejected when its units agree.

    Unit disagreement is only a detector for a subset of the defect; the
    strike-point/camera case has agreeing dimensionless units.
    """
    ok, reason = _is_attachment_consistent(
        "summary/boundary/strike_point_inner_z/value",
        "z_image_up_unit_vector_of_camera",
        dd_unit="m",
        sn_unit="m",
    )
    assert not ok
    assert "locus" in reason.lower()


# ---------------------------------------------------------------------------
# Shape-parameter bases are derived from the ISN vocabulary, not hardcoded
# ---------------------------------------------------------------------------


def test_shape_parameter_bases_are_isn_physical_bases() -> None:
    """Every shape-parameter base must exist in the ISN physical_base vocabulary.

    The set is a codex POLICY (which bases require a surface locus) keyed on
    ISN tokens; ISN owns the token universe. If ISN renames or retires one of
    these, this fails loudly rather than silently dropping the surface rule.
    """
    from imas_standard_names.grammar.constants import SEGMENT_TOKEN_MAP

    isn_bases = frozenset(SEGMENT_TOKEN_MAP.get("physical_base") or ())
    assert isn_bases, "ISN physical_base vocabulary is empty — grammar not loaded"
    bases = _shape_parameter_bases()
    assert bases, "no shape-parameter bases resolved from the ISN vocabulary"
    missing = bases - isn_bases
    assert not missing, (
        f"shape-parameter policy names {sorted(missing)}, absent from the ISN "
        "physical_base vocabulary — ISN drifted; update the policy or ask ISN"
    )


def test_shape_parameter_surface_rule_still_fires() -> None:
    """The ISN-derived set drives the same surface rule as before."""
    ok, reason = _is_attachment_consistent(
        "equilibrium/time_slice/profiles_1d/triangularity_upper",
        "upper_triangularity_of_plasma_boundary",
    )
    assert not ok
    assert "shape-parameter surface" in reason
