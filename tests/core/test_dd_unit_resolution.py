"""DD unit-extraction integrity tests.

Guards two classes of unit corruption that reach IMASNode.unit silently and
then mislead the standard-name reviewer:

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
        # (raw, expected normalized) — multi-character SI symbols that share
        # a leading character with a shorter unit, so a truncating normalizer
        # silently maps them onto the wrong quantity.
        cases = {
            "Wb": "Wb",  # weber (poloidal flux) — must not collapse to 'W'
            "W": "W",  # watt — must stay distinct from Wb
            "m^-2": "m^-2",  # must not collapse to 'm'
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


class TestCanonicalUnitOrdering:
    """One canonical authority: symbol ORDER is physics, not cosmetics.

    ``W.m^-3`` is the conventional rendering of a power density; ``m^-3.W`` is
    not. The pint formatter orders factors by its own internal sequence, so it
    emitted ``m^-3.W`` while the standard-name side canonicalised the same unit
    to ``W.m^-3`` — two spellings of one unit in one graph. The DD side must
    therefore canonicalise through the SAME authority the standard-name side
    uses (``imas_standard_names.canonical_unit``).
    """

    def test_leading_named_unit_is_preserved(self):
        for raw, want in {
            "W.m^-3": "W.m^-3",
            "m^-3.W": "W.m^-3",
            "W/m^3": "W.m^-3",
            "N.m^-2": "N.m^-2",
            "m^-2.N": "N.m^-2",
            "V.m^-1": "V.m^-1",
            "T.m": "T.m",
            "W.m^-2.sr^-1": "W.m^-2.sr^-1",
        }.items():
            assert normalize_unit_symbol(raw) == want, raw

    def test_agrees_with_the_standard_name_authority(self):
        from imas_standard_names import canonical_unit

        for raw in ("W.m^-3", "N.m^-2", "V.m^-1", "T.m", "m^3.Pa.s^-1", "A/m^2"):
            assert normalize_unit_symbol(raw) == canonical_unit(raw), raw

    def test_dimensionless_and_no_unit_stay_distinct(self):
        """Dimensionless (dimensions cancel -> '1') is NOT the same as no unit."""
        assert normalize_unit_symbol("-") == "1"
        assert normalize_unit_symbol("1") == "1"
        assert normalize_unit_symbol("dimensionless") == "1"
        # No unit / not a unit -> None, never '1'.
        assert normalize_unit_symbol("") is None
        assert normalize_unit_symbol(None) is None
        assert normalize_unit_symbol("mixed") is None


class TestDdUnitDefectCuration:
    """A DD-declared unit defect has exactly one effective correction authority.

    DD 4.1.1 declares ``ionisation_potential`` as ``eV`` under ``profiles_1d``
    but as ``e`` under ``ggd`` — the same quantity, one an energy and one a
    charge. The legacy graph correction remains the fallback for affected paths
    without exact active resolution authority. An exact active resolution makes
    that fallback inert so the raw declaration remains available to the active
    resolver, which applies the effective unit with provenance. Charge numbers
    legitimately carry ``e`` and must be untouched.
    """

    def test_active_resolution_retires_legacy_graph_correction(self):
        from imas_codex.units import resolve_dd_unit

        for path in (
            "edge_profiles/ggd/ion/state/ionisation_potential",
            "plasma_profiles/ggd/ion/state/ionisation_potential",
        ):
            assert resolve_dd_unit(path, "e") == "e", path

    def test_unresolved_descendant_keeps_legacy_graph_correction(self):
        from imas_codex.units import resolve_dd_unit

        path = "plasma_profiles/ggd/ion/state/ionisation_potential/values"
        assert resolve_dd_unit(path, "e") == "eV"

    def test_scalar_sibling_declaration_is_unchanged(self):
        from imas_codex.units import resolve_dd_unit

        path = "edge_profiles/profiles_1d/ion/state/ionisation_potential"
        assert resolve_dd_unit(path, "eV") == "eV"

    def test_charge_numbers_keep_the_charge_unit(self):
        from imas_codex.units import resolve_dd_unit

        for leaf in ("z_ion", "z_min", "z_max", "z_n", "z_average", "z_square_average"):
            path = f"core_profiles/profiles_1d/ion/state/{leaf}"
            assert resolve_dd_unit(path, "e") == "e", leaf

    def test_uncurated_paths_fall_through_to_normalisation(self):
        from imas_codex.units import resolve_dd_unit

        assert resolve_dd_unit("equilibrium/time_slice/profiles_1d/psi", "Wb") == "Wb"
        assert resolve_dd_unit("equilibrium/time_slice/boundary/elongation", "-") == "1"
        assert resolve_dd_unit("some/path", "mixed") is None


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
