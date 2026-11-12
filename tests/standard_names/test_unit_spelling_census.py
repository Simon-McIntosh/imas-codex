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


@pytest.mark.graph
def test_catalog_write_takes_the_recorded_spelling_on_a_disposable_graph(
    disposable_neo4j_empty: str,
    seed_dd_resolutions,
) -> None:
    """A name written for a recorded quantity takes its spelling from the DD one.

    The dictionary leaf records ``kg.m^-1.s^-2`` for a torque density. That
    recorded spelling is the test input: it is read back from the graph, handed
    to the extraction path, and the name it produces is written through the
    catalog. The second name is a quantity with no recorded spelling, and it
    keeps the spelling the dictionary carries — so the write is shown to store
    what the pipeline decided rather than to rewrite it.
    """
    from imas_codex.graph.client import GraphClient
    from imas_codex.standard_names import graph_store  # noqa: F401 (binds services)
    from imas_codex.standard_names.graph_store.graph_catalog import (
        write_standard_names,
    )
    from imas_codex.standard_names.sources.dd import _apply_unit_overrides

    torque_path = "core_profiles/profiles_1d/ion/torque_density_tor"
    temperature_path = "core_profiles/profiles_1d/electron_temperature"
    seed_dd_resolutions(
        [
            {"path": torque_path, "effective": "kg.m^-1.s^-2"},
            {"path": temperature_path, "effective": "eV"},
        ]
    )

    with GraphClient() as gc:
        # The dictionary's own spelling, as the extraction stage receives it.
        leaves = list(
            gc.query(
                "MATCH (dd:IMASNode)-[:HAS_UNIT]->(u:Unit) "
                "WHERE dd.id IN $paths RETURN dd.id AS path, u.id AS unit",
                paths=[torque_path, temperature_path],
            )
        )
        dd_units = {row["path"]: str(row["unit"]) for row in leaves}
        assert dd_units == {
            torque_path: "kg.m^-1.s^-2",
            temperature_path: "eV",
        }

        rows = [
            {"path": torque_path, "unit": dd_units[torque_path]},
            {"path": temperature_path, "unit": dd_units[temperature_path]},
        ]
        kept = _apply_unit_overrides(rows, write_skipped=False)
        by_path = {row["path"]: row["unit"] for row in kept}
        assert by_path[torque_path] == "N.m^-2"
        assert by_path[temperature_path] == "eV"

        write_standard_names(
            [
                {
                    "id": "ion_torque_density",
                    "unit": by_path[torque_path],
                    "source_types": ["dd"],
                    "source_id": torque_path,
                    "description": "torque density from the data dictionary",
                    "name_stage": "drafted",
                },
                {
                    "id": "electron_temperature",
                    "unit": by_path[temperature_path],
                    "source_types": ["dd"],
                    "source_id": temperature_path,
                    "description": "electron temperature from the data dictionary",
                    "name_stage": "drafted",
                },
            ],
            gc=gc,
        )
        stored = {
            str(row["id"]): str(row["unit"])
            for row in gc.query(
                "MATCH (sn:StandardName) WHERE sn.id IN $ids "
                "RETURN sn.id AS id, sn.unit AS unit",
                ids=["ion_torque_density", "electron_temperature"],
            )
        }
        leaves_after = {
            str(row["path"]): str(row["unit"])
            for row in gc.query(
                "MATCH (dd:IMASNode)-[:HAS_UNIT]->(u:Unit) "
                "WHERE dd.id IN $paths RETURN dd.id AS path, u.id AS unit",
                paths=[torque_path, temperature_path],
            )
        }

    assert leaves_after[torque_path] == "kg.m^-1.s^-2"
    assert stored["ion_torque_density"] == "N.m^-2"
    assert stored["electron_temperature"] == "eV"
    assert disposable_neo4j_empty.startswith("bolt://")
