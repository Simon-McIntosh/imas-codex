"""Meaning preservation is narrow enough to protect inherited documentation."""

from __future__ import annotations

import inspect

import pytest
from imas_standard_names.grammar import parser as isn_parser

from imas_codex.standard_names.graph_store.graph_refinement import (
    persist_refined_name,
    rename_preserves_meaning,
)

LOCUS_ONLY_RENAME = (
    "normalized_poloidal_flux_coordinate_at_measurement_position",
    "normalized_poloidal_flux_coordinate_at_reflectometer_cutoff_position",
)


def test_a_locus_token_substitution_preserves_meaning() -> None:
    """The locus identity may move while every locus role stays unchanged."""
    old_ir = isn_parser.parse(LOCUS_ONLY_RENAME[0]).ir
    new_ir = isn_parser.parse(LOCUS_ONLY_RENAME[1]).ir

    assert old_ir != new_ir
    assert old_ir.locus is not None
    assert new_ir.locus is not None
    assert old_ir.locus.token == "measurement_position"
    assert new_ir.locus.token == "reflectometer_cutoff_position"
    assert old_ir.locus.relation == new_ir.locus.relation
    assert old_ir.locus.type == new_ir.locus.type
    assert old_ir.locus.qualifiers == new_ir.locus.qualifiers
    assert old_ir.locus.value == new_ir.locus.value
    assert old_ir.model_copy(
        update={"locus": old_ir.locus.model_copy(update={"token": None})}
    ) == new_ir.model_copy(
        update={"locus": new_ir.locus.model_copy(update={"token": None})}
    )

    assert rename_preserves_meaning(*LOCUS_ONLY_RENAME) is True


@pytest.mark.parametrize(
    ("old_name", "new_name"),
    [
        (
            "temperature_at_magnetic_axis",
            "density_at_plasma_boundary",
        ),
        (
            "radial_magnetic_field_at_magnetic_axis",
            "toroidal_magnetic_field_at_plasma_boundary",
        ),
        (
            "square_of_temperature_at_magnetic_axis",
            "inverse_of_temperature_at_plasma_boundary",
        ),
        (
            "electron_temperature_at_magnetic_axis",
            "ion_temperature_at_plasma_boundary",
        ),
        (
            "power_at_magnetic_axis_due_to_ohmic_dissipation",
            "power_at_plasma_boundary_due_to_ion_cyclotron_heating",
        ),
        (
            "temperature_at_magnetic_axis",
            "temperature_of_plasma_boundary",
        ),
        (
            "temperature_of_magnetic_axis",
            "temperature_of_camera",
        ),
    ],
    ids=[
        "base-changed",
        "projection-changed",
        "operator-changed",
        "outer-qualifier-changed",
        "mechanism-changed",
        "locus-relation-changed",
        "locus-type-changed",
    ],
)
def test_every_other_meaning_change_is_refused(
    old_name: str,
    new_name: str,
) -> None:
    assert rename_preserves_meaning(old_name, new_name) is False


def test_persistence_carries_the_complete_documentation_axis() -> None:
    """The predicate controls every field needed for a non-pending document."""
    source = " ".join(inspect.getsource(persist_refined_name).split())

    assert "meaning_preserved = rename_preserves_meaning(old_name, new_name)" in source
    assert (
        "new.docs_stage = CASE WHEN $meaning_preserved "
        "AND coalesce(old.documentation, '') <> '' "
        "THEN coalesce(old.docs_stage, 'pending') ELSE 'pending' END"
    ) in source
    assert (
        "new.documentation = CASE WHEN $meaning_preserved "
        "THEN old.documentation ELSE null END"
    ) in source
    assert (
        "new.docs_chain_length = CASE WHEN $meaning_preserved "
        "THEN coalesce(old.docs_chain_length, 0) ELSE 0 END"
    ) in source
    assert (
        "new.docs_model = CASE WHEN $meaning_preserved "
        "THEN old.docs_model ELSE null END"
    ) in source
    assert (
        "new.docs_generated_at = CASE WHEN $meaning_preserved "
        "THEN old.docs_generated_at ELSE null END"
    ) in source
