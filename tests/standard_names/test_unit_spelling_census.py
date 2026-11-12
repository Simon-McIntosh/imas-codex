"""The unit-spelling census and the recorded spelling each quantity takes."""

from __future__ import annotations

import pytest


def _census(rows):
    from imas_codex.standard_names.audits import unit_spelling_census

    return unit_spelling_census(rows)


def test_census_counts_a_quantity_planted_in_two_equal_spellings() -> None:
    census = _census(
        [
            {"id": "ion_torque", "unit": "kg.m^2.s^-2"},
            {"id": "toroidal_torque", "unit": "N.m"},
        ]
    )

    assert [entry["quantity"] for entry in census["duplicate_quantities"]] == ["torque"]
    entry = census["duplicate_quantities"][0]
    assert entry["equal_dimension"] is True
    assert len(entry["spellings"]) == 2
    assert census["clean"] is False


def test_census_resolves_a_quantity_the_record_spells_and_leaves_one_it_does_not() -> (
    None
):
    census = _census(
        [
            {"id": "ion_torque", "unit": "kg.m^2.s^-2"},
            {"id": "toroidal_torque", "unit": "N.m"},
            {"id": "synthetic_gap", "unit": "Hz"},
            {"id": "synthetic_gap_at_wall", "unit": "1/s"},
        ]
    )

    by_quantity = {entry["quantity"]: entry for entry in census["duplicate_quantities"]}
    assert by_quantity["torque"]["resolved"] is True
    assert by_quantity["torque"]["canonical"] == "N.m"
    assert by_quantity["synthetic_gap"]["resolved"] is False
    assert census["unresolved_duplicate_count"] == 1


def test_census_counts_a_name_planted_with_a_mismatched_dimension() -> None:
    census = _census(
        [
            {"id": "ion_torque_density", "unit": "s^-1"},
        ]
    )

    assert [row["name"] for row in census["dimension_mismatches"]] == [
        "ion_torque_density"
    ]
    assert census["dimension_mismatches"][0]["quantity"] == "torque_density"
    assert census["clean"] is False


def test_census_is_silent_on_a_planted_clean_pair() -> None:
    census = _census(
        [
            {"id": "wave_frequency", "unit": "Hz"},
            {"id": "edge_localized_mode_frequency", "unit": "Hz"},
        ]
    )

    assert census["duplicate_quantities"] == []
    assert census["divergences"] == []
    assert census["clean"] is True


def test_census_records_a_meaningful_difference_rather_than_unifying_it() -> None:
    census = _census(
        [
            {"id": "temperature_of_coil_conductor", "unit": "K"},
            {"id": "electron_temperature", "unit": "eV"},
        ]
    )

    assert census["duplicate_quantities"] == []
    by_quantity = {entry["quantity"]: entry for entry in census["divergences"]}
    assert by_quantity["temperature"]["recorded_exception"] is True
    assert by_quantity["temperature"]["reason"]
    assert census["unrecorded_divergence_count"] == 0


def test_census_reports_an_unrecorded_divergence() -> None:
    census = _census(
        [
            {"id": "synthetic_length", "unit": "m"},
            {"id": "synthetic_length_at_wall", "unit": "s"},
        ]
    )

    assert census["unrecorded_divergence_count"] == 1


@pytest.mark.parametrize(
    ("name", "quantity"),
    [
        ("ion_torque", "torque"),
        ("explicit_ion_torque", "torque"),
        ("toroidal_ion_torque_density", "torque_density"),
        ("neutron_source_rate_due_to_thermal_fusion", "source_rate"),
        ("incident_neutral_particle_flux_at_wall", "particle_flux"),
        ("alfven_frequency", "frequency"),
        ("electron_density", "electron_density"),
    ],
)
def test_quantity_of_name_reads_the_head_the_qualifiers_modify(name, quantity) -> None:
    from imas_codex.standard_names.unit_audit import quantity_of_name

    assert quantity_of_name(name) == quantity


def test_extraction_rewrites_a_dictionary_unit_to_the_recorded_spelling() -> None:
    from imas_codex.standard_names.sources.dd import _apply_unit_overrides

    rows = [
        {"path": "core_profiles/profiles_1d/ion/torque_tor", "unit": "kg.m^2.s^-2"},
        {
            "path": "core_profiles/profiles_1d/ion/torque_density_tor",
            "unit": "kg.m^-1.s^-2",
        },
        {"path": "core_profiles/profiles_1d/ion/velocity_tor", "unit": "m.s^-1"},
    ]

    kept = _apply_unit_overrides(rows, write_skipped=False)

    assert kept[0]["unit"] == "N.m"
    assert kept[0]["_unit_override"]["spelling_quantity"] == "torque"
    assert kept[1]["unit"] == "N.m^-2"
    assert kept[2]["unit"] == "m.s^-1"
    assert "_unit_override" not in kept[2]


def test_extraction_leaves_a_dimensionally_different_spelling_alone() -> None:
    from imas_codex.standard_names.unit_overrides import resolve_quantity_spelling

    # A frequency path carrying an angular frequency is a different quantity
    # of meaning, so the spelling record does not rewrite it.
    unit, meta = resolve_quantity_spelling(
        "core_profiles/profiles_1d/rotation_frequency", "rad.s^-1"
    )

    assert unit == "rad.s^-1"
    assert meta is None
