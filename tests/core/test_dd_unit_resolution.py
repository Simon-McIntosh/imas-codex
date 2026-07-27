"""DD unit-extraction integrity tests.

Guards two classes of unit corruption that previously reached IMASNode.unit
and corrupted the standard-name reviewer:

1. ``as_parent`` placeholder non-resolution — the DD emits
   ``as_parent`` / ``as_parent_level_2`` / ``as parent`` meaning "inherit the
   parent node's unit". The literal placeholder must never be stored; it must
   resolve to the ancestor's concrete unit.
2. Multi-letter SI unit truncation (e.g. ``Wb`` → ``W``, ``m^-2`` → ``m``) —
   the pint normalizer must round-trip multi-character symbols intact.
"""

from imas_codex.graph.build_dd import (
    _is_unit_parent_placeholder,
    _resolve_unit_placeholder,
)
from imas_codex.units import normalize_unit_symbol


class TestPlaceholderDetection:
    def test_recognises_placeholder_forms(self):
        for raw in (
            "as_parent",
            "as_parent_level_2",
            "as parent",
            "as_parent for a local measurement, as_parent.m for a line integrated measurement",
        ):
            assert _is_unit_parent_placeholder(raw), raw

    def test_concrete_units_are_not_placeholders(self):
        for raw in ("m^-3", "Wb", "T.m", "s^-1", "1", "-", "", None):
            assert not _is_unit_parent_placeholder(raw), raw


class TestPlaceholderResolution:
    def test_resolves_to_immediate_parent_unit(self):
        paths = {
            "summary/local/divertor_plate/n_i": {
                "units": "m^-3",
                "parent_path": "summary/local/divertor_plate",
            },
        }
        out = _resolve_unit_placeholder(
            "as_parent_level_2", "summary/local/divertor_plate/n_i", paths
        )
        assert out == "m^-3"

    def test_walks_past_structural_ancestors_with_empty_unit(self):
        # value -> argon(struct, '') -> n_i(m^-3)
        paths = {
            "summary/local/divertor_plate/n_i": {
                "units": "m^-3",
                "parent_path": "summary/local/divertor_plate",
            },
            "summary/local/divertor_plate/n_i/argon": {
                "units": "",
                "parent_path": "summary/local/divertor_plate/n_i",
            },
        }
        out = _resolve_unit_placeholder(
            "as_parent_level_2",
            "summary/local/divertor_plate/n_i/argon",
            paths,
        )
        assert out == "m^-3"

    def test_concrete_unit_passes_through_unchanged(self):
        assert _resolve_unit_placeholder("Wb", "x/y", {}) == "Wb"
        assert _resolve_unit_placeholder("", "x/y", {}) == ""

    def test_no_concrete_ancestor_yields_empty_not_placeholder(self):
        paths = {
            "a/b": {"units": "", "parent_path": "a"},
            "a": {"units": "", "parent_path": None},
        }
        out = _resolve_unit_placeholder("as parent", "a/b", paths)
        assert out == ""
        assert not _is_unit_parent_placeholder(out)

    def test_resolution_is_cycle_safe(self):
        # Pathological self/mutual parent references must not loop forever.
        paths = {
            "a": {"units": "", "parent_path": "b"},
            "b": {"units": "", "parent_path": "a"},
        }
        out = _resolve_unit_placeholder("as_parent", "a", paths)
        assert out == ""


class TestUnitNormalizationSurvival:
    """Multi-letter SI units must not be truncated by normalization."""

    def test_multiletter_si_units_round_trip(self):
        # (raw, expected normalized) — the previously-corrupted cases plus
        # other multi-character SI symbols that share a leading character.
        cases = {
            "Wb": "Wb",  # weber (poloidal flux) — was truncated to 'W'
            "W": "W",  # watt — must stay distinct from Wb
            "m^-2": "m^-2",  # was truncated to 'm'
            "Hz": "Hz",
            "Pa": "Pa",
            "kg": "kg",
            "rad": "rad",
            "sr": "sr",
        }
        for raw, expected in cases.items():
            assert normalize_unit_symbol(raw) == expected, raw

    def test_weber_and_watt_are_distinct(self):
        assert normalize_unit_symbol("Wb") != normalize_unit_symbol("W")


class TestDimensionlessIsTheUnitOne:
    """Dimensionless is the unit ``1``, not the absence of a unit.

    The IMAS DD marks dimensionless quantities with ``-``; imas-python returns
    that string. Mapping it (and the equivalent ``1`` / ``dimensionless``) to
    None dropped every dimensionless quantity's HAS_UNIT edge, desyncing the DD
    side (no unit) from the standard-name side (canonical ``1``) so the SN↔DD
    edge reconcile discarded the pair on a false unit disagreement.
    """

    def test_dimensionless_markers_normalize_to_one(self):
        for raw in ("-", "1", "dimensionless"):
            assert normalize_unit_symbol(raw) == "1", raw

    def test_genuine_non_units_stay_none(self):
        for raw in ("mixed", "as parent", "as_parent", "Toroidal angle", "", None):
            assert normalize_unit_symbol(raw) is None, raw

    def test_real_units_are_unaffected(self):
        assert normalize_unit_symbol("m.s^-1") == "m.s^-1"
        assert normalize_unit_symbol("m^2.sr") == "m^2.sr"


class TestDdCountPseudoUnits:
    """DD count pseudo-units are dimensionless counts, not unparseable junk.

    The DD expresses "a number of things" with a plural noun unit — ``electrons``
    (summary/gas_injection_*), ``atoms`` (isotope element counts). pint cannot
    parse these, so they were dropped and the quantity lost its unit entirely.
    A count is the dimensionless unit ``1``. This also recovers the
    ``as_parent`` chain: gas_injection ``value`` fields inherit ``electrons``
    from their grandparent, so dropping it dropped the resolved inheritance too.
    """

    def test_count_pseudo_units_are_dimensionless(self):
        for raw in ("electrons", "atoms"):
            assert normalize_unit_symbol(raw) == "1", raw

    def test_ambiguous_charge_spelling_is_not_guessed(self):
        """'Elementary Charge Unit' is dimensionally ambiguous — never guess it.

        The DD writes it both on charge numbers (z_ion / z_min / z_max / z_n,
        genuinely a charge) and on ionisation potentials, which are ENERGIES the
        DD carries as 'eV' elsewhere. A context-free normaliser that picked
        either would type one family wrongly, so it must resolve to None and
        leave the choice to the quantity-aware DD layer.
        """
        assert normalize_unit_symbol("Elementary Charge Unit") is None

    def test_parametric_and_runtime_units_stay_unresolved(self):
        # These carry no fixed dimensionality — better None than a wrong unit.
        for raw in (
            "m^dimension",
            "units given by process(i1)/results_units",
        ):
            assert normalize_unit_symbol(raw) is None, raw
