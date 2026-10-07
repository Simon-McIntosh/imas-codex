"""Free-text unit spellings resolve through the alias file.

The EDDB catalogue (and MDSplus trees generally) publish units as human text
rather than SI symbols: letter case varies (``KA``, ``Mpa``, ``DegC``), digit
exponents are glued to the symbol (``m2``, ``m3``), products are glued
(``Pam3``), and neutron yields count events over a period (``n/day``). These
tests pin the alias definitions in ``data_dictionary_unit_aliases.txt`` so a
spelling either normalises to the canonical symbol of its unit or is reported
as unable to resolve — never left to a guess.
"""

import pytest

from imas_codex.ids.tools import analyze_units
from imas_codex.units import normalize_unit_symbol

# Each added spelling and the canonical symbol it must collapse to.
CASE_AND_EXPONENT_SPELLINGS = {
    "KA": "kA",
    "KV": "kV",
    "Mpa": "MPa",
    "DEG": "deg",
    "DegC": "degC",
    "oC": "degC",
    "KeV": "keV",
    "m2": "m^2",
    "m3": "m^3",
}

COMPOSITE_SPELLINGS = {
    "Pa*m3/s": "m^3.Pa.s^-1",
    "Pam3/s": "m^3.Pa.s^-1",
    "Pam3/sec": "m^3.Pa.s^-1",
    "m3/s": "m^3.s^-1",
    "T/m2": "T.m^2^-1",
    "KeV/m": "keV.m^-1",
    "n/day": "count.d^-1",
    "n/s": "count.s^-1",
    "n/shot": "count.jig^-1",
    "n/week": "count.week^-1",
    "n/year": "count.a^-1",
}


@pytest.mark.parametrize(
    ("raw", "canonical"), sorted(CASE_AND_EXPONENT_SPELLINGS.items())
)
def test_case_and_exponent_spellings_normalise(raw, canonical):
    assert normalize_unit_symbol(raw) == canonical


@pytest.mark.parametrize(("raw", "canonical"), sorted(COMPOSITE_SPELLINGS.items()))
def test_composite_spellings_normalise(raw, canonical):
    assert normalize_unit_symbol(raw) == canonical


def test_spelling_merges_with_native_symbol():
    # A glued spelling and the native spelling must land on one canonical form,
    # so the graph keeps a single Unit node for the unit.
    assert normalize_unit_symbol("m2") == normalize_unit_symbol("m^2")
    assert normalize_unit_symbol("m3") == normalize_unit_symbol("m^3")
    assert normalize_unit_symbol("KA") == normalize_unit_symbol("kA")
    assert normalize_unit_symbol("Mpa") == normalize_unit_symbol("MPa")


def test_glued_product_spellings_agree():
    forms = ["Pa*m3/s", "Pam3/s", "Pam3/sec"]
    assert len({normalize_unit_symbol(f) for f in forms}) == 1


def test_microstrain_is_scaled_dimensionless():
    # Strain gauges report microstrain. Strain is dimensionless, so the micro
    # prefix is the unit's entire scale and the factor survives normalisation
    # (the map stage compares normalised strings, so the scale must live in
    # the symbol's definition).
    assert normalize_unit_symbol("uST") == "uST"
    result = analyze_units("uST", "1")
    assert result["compatible"] is True
    assert result["conversion_factor"] == pytest.approx(1e-6)


def test_kiloampere_is_a_thousand_ampere():
    result = analyze_units(normalize_unit_symbol("KA"), normalize_unit_symbol("A"))
    assert result["compatible"] is True
    assert result["conversion_factor"] == pytest.approx(1000.0)


@pytest.mark.parametrize("raw", ["MPaG", "Unit", "ph/srsm2", "m-2", "m-3"])
def test_unresolved_spellings_are_reported_not_guessed(raw):
    # Gauge pressure is not an absolute pressure; "Unit" is a count of devices,
    # not a physical unit; the remaining composites name no unit pint can
    # represent. Each must resolve to None rather than a guessed unit.
    assert normalize_unit_symbol(raw) is None
