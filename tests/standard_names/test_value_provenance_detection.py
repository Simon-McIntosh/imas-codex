"""Value-provenance facet detection (collapse provenance, de-conflate physics)."""

from __future__ import annotations

import pytest

from imas_codex.standard_names.provenance import detect_value_provenance


@pytest.mark.parametrize(
    "path,term,base",
    [
        (
            "equilibrium/time_slice/constraints/ip/measured",
            "measured",
            "equilibrium/time_slice/constraints/ip",
        ),
        (
            "equilibrium/time_slice/constraints/diamagnetic_flux/reconstructed",
            "reconstructed",
            "equilibrium/time_slice/constraints/diamagnetic_flux",
        ),
        (
            "pulse_schedule/position_control/radial_field/reference",
            "reference",
            "pulse_schedule/position_control/radial_field",
        ),
    ],
)
def test_detects_and_strips_provenance_facet(path, term, base):
    got_term, got_base = detect_value_provenance(path)
    assert got_term == term
    assert got_base == base


@pytest.mark.parametrize(
    "path",
    [
        "core_profiles/profiles_1d/electrons/temperature",
        "equilibrium/time_slice/constraints/ip/weight",
        "equilibrium/time_slice/constraints/ip/chi_squared",
        "equilibrium/time_slice/constraints/ip/time_measurement",
        "equilibrium/time_slice/constraints/ip/exact",
        "equilibrium/time_slice/global_quantities/ip",
    ],
)
def test_non_provenance_paths_unchanged(path):
    term, base = detect_value_provenance(path)
    assert term is None
    assert base == path


def test_measured_and_reconstructed_share_base():
    # The collapse invariant: two estimator facets of one quantity strip to the
    # same base path -> same grounding -> same name.
    _, b1 = detect_value_provenance("equilibrium/time_slice/constraints/ip/measured")
    _, b2 = detect_value_provenance(
        "equilibrium/time_slice/constraints/ip/reconstructed"
    )
    assert b1 == b2


def test_edge_cases():
    assert detect_value_provenance(None) == (None, "")
    assert detect_value_provenance("") == (None, "")
    assert detect_value_provenance("measured") == (None, "measured")  # no parent


class _FakeCand:
    """Minimal compose-candidate stand-in for canonicalizer tests."""

    def __init__(self, source_id: str, name: str):
        self.source_id = source_id
        self._name = name

    def compose_name(self) -> str:
        return self._name


def test_canonicalizer_collapses_drifted_estimators():
    from imas_codex.standard_names.workers import provenance_canonical_names

    cands = [
        _FakeCand(
            "equilibrium/time_slice/constraints/pressure/measured", "plasma_pressure"
        ),
        _FakeCand(
            "equilibrium/time_slice/constraints/pressure/reconstructed",
            "total_plasma_pressure",
        ),
    ]
    canon = provenance_canonical_names(cands)
    base = "equilibrium/time_slice/constraints/pressure"
    assert base in canon
    # shortest wins on the tie -> drops the spurious 'total_' qualifier
    assert canon[base] == "plasma_pressure"


def test_canonicalizer_noop_when_estimators_agree():
    from imas_codex.standard_names.workers import provenance_canonical_names

    cands = [
        _FakeCand("equilibrium/time_slice/constraints/ip/measured", "plasma_current"),
        _FakeCand(
            "equilibrium/time_slice/constraints/ip/reconstructed", "plasma_current"
        ),
    ]
    # Agreement -> no entry (nothing to rewrite).
    assert provenance_canonical_names(cands) == {}


def test_canonicalizer_ignores_non_provenance_candidates():
    from imas_codex.standard_names.workers import provenance_canonical_names

    cands = [
        _FakeCand(
            "core_profiles/profiles_1d/electrons/temperature", "electron_temperature"
        )
    ]
    assert provenance_canonical_names(cands) == {}
