"""Precision regressions for structure-aware post-generation audits."""

from imas_codex.standard_names.audits import (
    cumulative_prefix_check,
    implicit_field_check,
    repeated_token_check,
    structural_dim_tag_check,
)


def test_line_integrated_quantities_are_not_cumulative_prefixes() -> None:
    legitimate_names = (
        "line_integrated_electron_density",
        "line_integrated_electron_number_density",
        "toroidal_line_integrated_impurity_ion_velocity",
    )

    for name in legitimate_names:
        assert cumulative_prefix_check({"id": name}) == []

    assert cumulative_prefix_check({"id": "integrated_electron_density"})


def test_governed_description_omits_storage_rank() -> None:
    candidate = {
        "id": "straight_field_line_angle",
        "description": (
            "The poloidal angular coordinate of a straight-field-line magnetic "
            "coordinate system, in radians."
        ),
    }

    assert structural_dim_tag_check(candidate) == []
    assert structural_dim_tag_check(
        {
            "id": candidate["id"],
            "description": (
                "Defined on a 3D equilibrium domain and stored as a 2D array "
                "of angle values."
            ),
        }
    )


def test_registered_field_compounds_are_not_bare_fields() -> None:
    legitimate_names = (
        "straight_field_line_angle",
        "poloidal_straight_field_line_angle",
        "magnetic_field_magnitude_at_pedestal_top_low_field_side",
    )

    for name in legitimate_names:
        assert implicit_field_check({"id": name}) == []

    assert implicit_field_check({"id": "vacuum_toroidal_field"})


def test_registered_field_locus_is_not_token_repetition() -> None:
    legitimate_name = "magnetic_field_magnitude_at_pedestal_top_low_field_side"

    assert repeated_token_check({"id": legitimate_name}) == []
    assert repeated_token_check({"id": "magnetic_magnetic_field_strength"})
