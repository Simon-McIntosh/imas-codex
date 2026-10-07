"""Free-text unit spellings resolve through the alias file.

The EDDB catalogue (and MDSplus trees generally) publish units as human text
rather than SI symbols: letter case varies (``KA``, ``Mpa``, ``DegC``), digit
exponents are glued to the symbol (``m2``, ``m3``) and products are glued
(``Pam3``). Each added spelling either denotes the unit its author meant or is
reported as unable to resolve — never left to a guess.

Two facts are tested that a spell-to-symbol equality cannot see, because two
equivalent spellings need not share a spelling, only a unit:

* a case variant of an existing unit goes through ``@alias``, which keeps the
  original unit object, so the offset of ``degC`` survives and ``DegC``,
  ``oC`` and ``degC`` all mean the same temperature;
* a glued form is a unit in its own right, defined without a symbol, so it
  re-parses to exactly the unit its author wrote — the round-trip invariant.

Equivalence is decided by pint (dimensionality equal, conversion factor
exactly 1), the same reading ``analyze_units`` and ``build_dd._units_changed``
make, so the assertions measure the property downstream code depends on rather
than a cosmetic spelling. The alias file's grammar is parsed fresh from the
file in one test, so the assertion is about that file rather than import-time
state.
"""

from pathlib import Path

import pint
import pytest

import imas_codex.units as units
from imas_codex.ids.tools import analyze_units
from imas_codex.units import (
    normalize_unit_symbol,
    unit_registry,
    units_are_equivalent,
)

HERE = Path(__file__).parent
ALIAS_FILE = Path(units.__file__).with_name("data_dictionary_unit_aliases.txt")
JT60SA_FILE = HERE / "jt60sa_unit_strings.txt"
DD_BASE_FILE = HERE / "dd_unit_strings_base.txt"

# Spelling pairs a catalogue uses that must denote one unit. Every pair is
# compared with pint, so a spelling change is only a change if the unit moved.
EQUIVALENT_SPELLINGS = [
    ("DegC", "degC"),
    ("oC", "degC"),
    ("m2", "m^2"),
    ("m3", "m^3"),
    ("Pam3", "Pa.m^3"),
    ("1/m2", "m^-2"),
    ("1/m3", "m^-3"),
    ("T/m2", "T.m^-2"),
    ("Pa*m3/s", "Pa.m^3.s^-1"),
    ("Pam3/s", "Pa.m^3.s^-1"),
    ("Pam3/sec", "Pa.m^3.s^-1"),
    ("KeV/m", "keV.m^-1"),
    ("KA", "kA"),
]

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


def _parses(raw: str) -> bool:
    """Whether pint can read *raw* as a unit (a sentinel such as ``-`` cannot)."""
    try:
        unit_registry.Quantity(1.0, raw)
        return True
    except Exception:
        return False


def _read_strings(path: Path) -> list[str]:
    return [line for line in path.read_text().splitlines() if line]


def _normalising_strings(path: Path) -> list[str]:
    """Strings that parse as a unit, so the round-trip invariant applies."""
    return [raw for raw in _read_strings(path) if _parses(raw)]


def _base_snapshot() -> list[tuple[str, str]]:
    rows = []
    for line in _read_strings(DD_BASE_FILE):
        raw, _, canonical = line.partition("\t")
        rows.append((raw, canonical))
    return rows


ROUND_TRIP_STRINGS = sorted(
    set(_normalising_strings(JT60SA_FILE))
    | {raw for raw, base in _base_snapshot() if base and _parses(raw)}
)


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


@pytest.mark.parametrize(("first", "second"), EQUIVALENT_SPELLINGS)
def test_equivalent_spellings_denote_one_unit(first, second):
    assert units_are_equivalent(first, second)
    assert units_are_equivalent(
        normalize_unit_symbol(first), normalize_unit_symbol(second)
    )


def test_case_variants_normalise_to_equivalent_units():
    # The three Celsius spellings may render differently; what matters is that
    # each denotes the offset unit degC.
    forms = [normalize_unit_symbol(s) for s in ("DegC", "oC", "degC")]
    assert all(units_are_equivalent(a, b) for a in forms for b in forms)


def test_kiloampere_is_compatible_with_ampere_not_equal():
    result = analyze_units(normalize_unit_symbol("KA"), normalize_unit_symbol("A"))
    assert result["compatible"] is True
    assert result["conversion_factor"] == pytest.approx(1000.0)
    assert not units_are_equivalent("KA", "A")


def test_microstrain_is_scaled_dimensionless():
    # Strain gauges report microstrain. Strain is dimensionless, so the micro
    # prefix is the unit's entire scale and the factor survives normalisation.
    assert normalize_unit_symbol("uST") == "uST"
    result = analyze_units("uST", "1")
    assert result["compatible"] is True
    assert result["conversion_factor"] == pytest.approx(1e-6)


@pytest.mark.parametrize("raw", UNRESOLVED_SPELLINGS)
def test_unresolved_spellings_are_reported_not_guessed(raw):
    # Gauge pressure is not an absolute pressure; "Unit" counts devices; the
    # neutron-yield spellings need a bare "n" that cannot be named; the rest
    # name no unit pint can represent. Each must be None rather than a guess.
    assert normalize_unit_symbol(raw) is None


@pytest.mark.parametrize("raw", ROUND_TRIP_STRINGS)
def test_normalisation_round_trips(raw):
    """A normalised spelling denotes exactly the unit its source wrote.

    This is the invariant that keeps pint comparison sound: an output that
    re-parses to a different unit (a glued exponent under a negative power, for
    instance) would compare unequal to the source and silently split one unit
    across two spellings.
    """
    normalised = normalize_unit_symbol(raw)
    assert normalised is not None, f"{raw!r} normalises to nothing"
    assert units_are_equivalent(raw, normalised), (
        f"{raw!r} -> {normalised!r} is not the same unit"
    )


@pytest.mark.parametrize("raw", ROUND_TRIP_STRINGS)
def test_no_factor_carries_two_exponents(raw):
    """No factor of a normalised unit holds two ``^``.

    pint reads ``^`` as the right-associative ``**``, so ``m^2^-1`` means
    m^(2^-1) = m^0.5 rather than m^-2. A factor with two exponents is a
    malformed symbol, whatever it re-parses to.
    """
    normalised = normalize_unit_symbol(raw)
    assert normalised is not None  # noqa: F632 — explicit None check
    for factor in normalised.split("."):
        assert factor.count("^") <= 1, f"{raw!r} -> {normalised!r}: {factor!r}"


def test_dd_unit_normalisation_is_not_regressed():
    """Every DD unit string that resolved at the base revision still resolves
    to an equivalent unit. Comparison is pint's, not the string's: a spelling
    may change where the unit does not (that is a cosmetic change), but a
    string that resolved may not move to a different unit. Strings that were
    unresolvable may gain a resolution — that is the point of the alias file."""
    changed = []
    checked = 0
    for raw, base_canonical in _base_snapshot():
        if not base_canonical:
            continue
        checked += 1
        head = normalize_unit_symbol(raw)
        if not units_are_equivalent(base_canonical, head):
            changed.append((raw, base_canonical, head))
    assert checked > 0, "base snapshot carried no resolved strings"
    assert changed == [], (
        f"{len(changed)} of {checked} resolved DD unit strings moved: {changed}"
    )
