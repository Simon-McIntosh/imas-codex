"""Free-text unit spellings resolve through the alias file.

The EDDB catalogue (and MDSplus trees generally) publish units as human text
rather than SI symbols: letter case varies (``KA``, ``Mpa``, ``DegC``), digit
exponents are glued to the symbol (``m2``, ``m3``), and products are glued
(``Pam3``). Each added spelling either normalises to the registry's canonical
symbol for its unit or is reported as unable to resolve — never left to a guess.

Two facts are tested here that a normalise-only test cannot see. A case variant
of an existing unit goes through ``@alias``, which keeps the original unit
object, so the offset of ``degC`` survives; a glued form is a new unit carrying
the canonical symbol, so a spelling and its native form collapse to one string.
Both are checked against a registry parsed fresh from the alias file, so the
assertion is about the file's own grammar rather than about import-time state.
"""

from pathlib import Path

import pint
import pytest

import imas_codex.units as units
from imas_codex.ids.tools import analyze_units
from imas_codex.units import normalize_unit_symbol

HERE = Path(__file__).parent
ALIAS_FILE = Path(units.__file__).with_name("data_dictionary_unit_aliases.txt")

# Each added spelling and the canonical symbol it must collapse to. The Celsius
# spellings are covered by the conversion test instead: their canonical string
# comes from a second registry the project uses for standard names, which does
# not load this file, so pinning one spelling here would pin the wrong layer.
CASE_AND_EXPONENT_SPELLINGS = {
    "KA": "kA",
    "KV": "kV",
    "Mpa": "MPa",
    "DEG": "deg",
    "KeV": "keV",
    "m2": "m^2",
    "m3": "m^3",
    "Pam3": "m^3.Pa",
}

# Spellings that must stay unresolved: no definition resolves them without
# claiming a name the nano prefix or another unit already owns.
UNRESOLVED_SPELLINGS = [
    "n/day",
    "n/s",
    "n/shot",
    "n/week",
    "n/year",
    "MPaG",
    "Unit",
    "ph/srsm2",
    "m-2",
    "m-3",
]


@pytest.fixture
def fresh_registry():
    """A registry parsed from the alias file, with no import-time state."""
    registry = pint.UnitRegistry()
    registry.load_definitions(str(ALIAS_FILE))
    return registry


@pytest.mark.parametrize("raw", ["degC", "DegC", "oC"])
def test_celsius_spellings_convert_to_kelvin(fresh_registry, raw):
    # The alias keeps the original offset unit, so a converted temperature is
    # still a temperature: 20 degC is 293.15 K.
    kelvin = fresh_registry.Quantity(20, raw).to("K").magnitude
    assert kelvin == pytest.approx(293.15)


def test_count_parses_dimensionless(fresh_registry):
    assert fresh_registry.Quantity(1, "count").dimensionless


@pytest.mark.parametrize(
    ("spelling", "base"),
    [("ns", "s"), ("nA", "A"), ("nm", "m"), ("nT", "T")],
)
def test_nano_prefix_is_not_shadowed(fresh_registry, spelling, base):
    # A bare "n" unit or alias would claim the nano prefix's parse space.
    # Each of these must still mean 1e-9 of its base unit.
    value = fresh_registry.Quantity(1, spelling).to(base).magnitude
    assert value == pytest.approx(1e-9)


@pytest.mark.parametrize(
    ("raw", "canonical"), sorted(CASE_AND_EXPONENT_SPELLINGS.items())
)
def test_added_spellings_normalise(raw, canonical):
    assert normalize_unit_symbol(raw) == canonical


def test_glued_product_spellings_agree():
    forms = ["Pa*m3/s", "Pam3/s", "Pam3/sec"]
    assert len({normalize_unit_symbol(f) for f in forms}) == 1


def test_microstrain_is_scaled_dimensionless():
    # Strain gauges report microstrain. Strain is dimensionless, so the micro
    # prefix is the unit's entire scale and the factor survives normalisation.
    assert normalize_unit_symbol("uST") == "uST"
    result = analyze_units("uST", "1")
    assert result["compatible"] is True
    assert result["conversion_factor"] == pytest.approx(1e-6)


def test_kiloampere_is_a_thousand_ampere():
    result = analyze_units(normalize_unit_symbol("KA"), normalize_unit_symbol("A"))
    assert result["compatible"] is True
    assert result["conversion_factor"] == pytest.approx(1000.0)


@pytest.mark.parametrize("raw", UNRESOLVED_SPELLINGS)
def test_unresolved_spellings_are_reported_not_guessed(raw):
    # Gauge pressure is not an absolute pressure; "Unit" counts devices; the
    # neutron-yield spellings need a bare "n" that cannot be named; the rest
    # name no unit pint can represent. Each must be None rather than a guess.
    assert normalize_unit_symbol(raw) is None


def _read_base_snapshot():
    path = HERE / "dd_unit_strings_base.txt"
    rows = []
    for line in path.read_text().splitlines():
        if not line:
            continue
        raw, _, canonical = line.partition("\t")
        rows.append((raw, canonical))
    return rows


def test_dd_unit_normalisation_is_not_regressed():
    """Every DD unit string that resolved at the base revision still resolves
    to the same symbol. Strings that were unresolvable may gain a resolution
    (that is the point of the alias file); a string that already resolved may
    not change, which is what protects the rest of the catalogue from a new
    definition colliding with an existing unit."""
    changed = []
    for raw, base_canonical in _read_base_snapshot():
        if base_canonical == "":
            continue
        head = normalize_unit_symbol(raw)
        if head != base_canonical:
            changed.append((raw, base_canonical, head))
    assert changed == [], f"normalisation regressed for {changed}"
