"""Unit tests for the SN↔DD unit-mismatch exception loader (no graph)."""

from imas_codex.units.dd_unit_exceptions import (
    canonical_or_none,
    dd_unit_bug_globs,
    load_exceptions,
    units_agree,
)


class TestCanonicalOrNone:
    def test_orders_collapse(self):
        assert canonical_or_none("m^-2.W") == canonical_or_none("W.m^-2")

    def test_sentinels_and_garbage_are_none(self):
        assert canonical_or_none("") is None
        assert canonical_or_none(None) is None
        assert canonical_or_none("not_a_unit_xyzzy") is None

    def test_dimensionless_sentinel(self):
        assert canonical_or_none("1") == "1"


class TestUnitsAgree:
    def test_identical_after_ordering(self):
        assert units_agree("W.m^-2", "m^-2.W", "any/path")
        assert units_agree("V.m^-1", "m^-1.V", "any/path")

    def test_equivalences(self):
        # frequency and torque spelling equivalences
        assert units_agree("s^-1", "Hz", "any/path")
        assert units_agree("Hz", "s^-1", "any/path")
        assert units_agree("N.m", "kg.m^2.s^-2", "any/path")

    def test_energy_not_equated_with_torque_spelling(self):
        # J (canonical 'J') is NOT a member of the N.m/kg.m^2.s^-2 set, so an
        # energy SN is never silently equated with a torque path.
        assert not units_agree("J", "N.m", "any/path")

    def test_dd_side_bug_charge_number(self):
        # charge NUMBER: SN dimensionless, DD tags elementary charge
        assert units_agree("1", "e", "core_profiles/profiles_1d/ion/z_ion")
        assert units_agree("1", "e", "nbi/unit/species/z_n")

    def test_dd_side_bug_unit_vector(self):
        assert units_agree("1", "m", "camera_ir/channel/camera/direction/x")
        assert units_agree("1", "m", "spi/injector/shatter_cone/unit_vector_major/z")

    def test_dd_side_bug_requires_matching_units(self):
        # A matching path glob does NOT suppress an unrelated unit pair: the
        # dd_unit and correct_unit must both canonicalise as recorded.
        assert not units_agree("m^-3", "e", "core_profiles/profiles_1d/ion/z_ion")

    def test_glob_does_not_overmatch_positions(self):
        # A z-position path (dimensioned metre, SN should be metre) is NOT a
        # unit-vector-component bug — the direction globs must not match it.
        assert not units_agree("1", "m", "thomson_scattering/channel/position/z")

    def test_genuine_mismatch_fails(self):
        assert not units_agree("m", "m^-3", "pellets/time_slice/pellet/path_profiles/n_e")

    def test_unparseable_dd_never_agrees(self):
        assert not units_agree("1", "unit_error", "some/path")


class TestExceptionFileShape:
    def test_entries_have_required_keys(self):
        for entry in load_exceptions()["dd_unit_bugs"]:
            assert {"path", "dd_unit", "correct_unit", "reason"} <= set(entry)
            # every declared unit must canonicalise
            assert canonical_or_none(str(entry["dd_unit"])) is not None
            assert canonical_or_none(str(entry["correct_unit"])) is not None

    def test_globs_exposed(self):
        assert dd_unit_bug_globs() == [
            str(e["path"]) for e in load_exceptions()["dd_unit_bugs"]
        ]
