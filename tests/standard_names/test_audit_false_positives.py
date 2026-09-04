"""Regression census for audit findings that rejected valid identities."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from imas_codex.standard_names.audits import (
    implicit_field_check,
    multi_subject_check,
    name_description_consistency_check,
    name_unit_consistency_check,
)
from imas_codex.standard_names.workers import validate_name_candidate

_SPECTRAL_IDENTITIES = (
    "perturbed_electrostatic_potential_imaginary_part",
    "perturbed_electrostatic_potential_real_part",
    "perturbed_plasma_mass_density",
    "perturbed_plasma_mass_density_imaginary_part",
    "perturbed_plasma_pressure",
    "perturbed_plasma_pressure_imaginary_part",
    "perturbed_plasma_pressure_real_part",
    "perturbed_plasma_velocity",
    "poloidal_perturbed_plasma_magnetic_field_imaginary_part",
    "poloidal_perturbed_plasma_magnetic_field_real_part",
    "poloidal_perturbed_plasma_velocity",
    "poloidal_perturbed_vacuum_magnetic_field",
    "poloidal_perturbed_vacuum_magnetic_field_imaginary_part",
    "radial_perturbed_plasma_velocity",
    "radial_perturbed_plasma_velocity_real_part",
    "radial_perturbed_vacuum_magnetic_field_imaginary_part",
    "toroidal_perturbed_plasma_magnetic_field_imaginary_part",
    "toroidal_perturbed_plasma_velocity",
    "toroidal_perturbed_vacuum_magnetic_field",
    "toroidal_perturbed_vacuum_magnetic_field_real_part",
)

_SPECTRAL_PARENTS = frozenset(
    {
        "perturbed_plasma_mass_density",
        "perturbed_plasma_pressure",
        "perturbed_plasma_velocity",
        "poloidal_perturbed_plasma_velocity",
        "poloidal_perturbed_vacuum_magnetic_field",
        "radial_perturbed_plasma_velocity",
        "toroidal_perturbed_plasma_velocity",
        "toroidal_perturbed_vacuum_magnetic_field",
    }
)

_UNIT_FALSE_POSITIVES = (
    ("energy_confinement_enhancement_factor", "1"),
    ("rotation_frequency_time_derivative_of_neoclassical_tearing_mode", "s^-2"),
    ("tendency_of_total_thermal_plasma_internal_energy", "W"),
)

_FIELD_COMPOUNDS = (
    "ion_field_line_average_temperature_over_scrape_off_layer",
    "magnetic_field_at_pedestal_top_low_field_side",
    "poloidal_electron_beta_at_pedestal_top_high_field_side",
    "poloidal_electron_beta_at_pedestal_top_low_field_side",
)


@pytest.mark.parametrize("name", _SPECTRAL_IDENTITIES)
def test_spectral_family_uses_grammar_and_parent_structure(name: str) -> None:
    candidate = {
        "id": name,
        "description": "A Fourier coefficient of a complex eigenfunction.",
        "children": [f"{name}_real_part"] if name in _SPECTRAL_PARENTS else [],
    }
    assert name_description_consistency_check(candidate) == []


@pytest.mark.parametrize(("name", "unit"), _UNIT_FALSE_POSITIVES)
def test_unit_audit_uses_physical_base_and_operator_dimensions(
    name: str, unit: str
) -> None:
    assert name_unit_consistency_check({"id": name, "unit": unit}) == []


@pytest.mark.parametrize("name", _FIELD_COMPOUNDS)
def test_registered_field_compounds_are_not_bare_fields(name: str) -> None:
    assert implicit_field_check({"id": name}) == []


def test_locus_owner_is_not_a_second_subject() -> None:
    assert multi_subject_check({"id": "electron_density_at_pellet_path"}) == []


def _admission_status(candidate: dict[str, str]) -> tuple[str, list[str]]:
    with patch(
        "imas_codex.standard_names.workers._validate_via_isn",
        return_value=([], {"pydantic": {"passed": True}}),
    ):
        issues, _summary, status = validate_name_candidate(candidate)
    return status, issues


def test_spectral_claim_without_structured_decomposition_still_quarantines() -> None:
    status, issues = _admission_status(
        {
            "id": "normal_magnetic_field",
            "unit": "T",
            "description": "Fourier coefficients of the normal magnetic field.",
        }
    )
    assert status == "quarantined"
    assert any("name_description_consistency_check" in issue for issue in issues)


def test_wrong_unit_for_dimensional_head_still_quarantines() -> None:
    status, issues = _admission_status(
        {
            "id": "particle_energy",
            "unit": "A",
            "description": "Energy carried by particles.",
        }
    )
    assert status == "quarantined"
    assert any("name_unit_consistency_check" in issue for issue in issues)


def test_two_primary_subjects_still_quarantine() -> None:
    candidate = {
        "id": "electron_deuterium_density",
        "unit": "m^-3",
        "description": "Density attributed simultaneously to two species.",
    }
    audit_issues = multi_subject_check(candidate)
    assert audit_issues and "multi_subject_check" in audit_issues[0]

    status, issues = _admission_status(candidate)
    assert status == "quarantined"
    assert any(issue.startswith("parse_error:") for issue in issues)
