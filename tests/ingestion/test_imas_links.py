"""Code references resolve to existing DD paths and IDS roots."""

import pytest

from imas_codex.ingestion.extractors.ids import extract_imas_path_references


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (
            "equilibrium.time_slice[0].profiles_2d[0].psi",
            "equilibrium/time_slice/profiles_2d/psi",
        ),
        (
            'ids_factory.new("equilibrium"); ids%time_slice(1)%profiles_2d(1)%psi',
            "equilibrium/time_slice/profiles_2d/psi",
        ),
        (
            'path = "equilibrium/time_slice/profiles_2d/psi"',
            "equilibrium/time_slice/profiles_2d/psi",
        ),
    ],
)
def test_extracts_dd_path_notation(source, expected):
    assert expected in extract_imas_path_references(source)


def test_generic_fortran_variable_needs_one_ids_name():
    source = 'new("equilibrium"); new("core_profiles"); ids%time_slice(1)%psi'
    assert "equilibrium/time_slice/psi" not in extract_imas_path_references(source)
