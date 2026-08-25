"""Tests for the SN↔DD unit-mismatch exception loader."""

import pytest

from imas_codex.standard_names.dd_resolutions import load_dd_resolution_manifest
from imas_codex.units.dd_unit_exceptions import (
    canonical_or_none,
    dd_unit_bug_globs,
    graph_unit_correction,
    load_exceptions,
    units_agree,
)


def _without_active_path(path: str):
    manifest = load_dd_resolution_manifest()
    return manifest.model_copy(
        update={
            "resolutions": tuple(
                record for record in manifest.resolutions if record.path != path
            )
        }
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

    @pytest.mark.graph
    def test_dd_side_bug_charge_number(self):
        # charge NUMBER: SN dimensionless, DD tags elementary charge
        assert units_agree("1", "e", "core_profiles/profiles_1d/ion/z_ion")
        assert units_agree("1", "e", "nbi/unit/species/z_n")

    @pytest.mark.graph
    def test_dd_side_bug_unit_vector(self):
        camera_path = "camera_ir/channel/camera/direction/x"
        assert units_agree(
            "1", "m", camera_path, manifest=_without_active_path(camera_path)
        )
        vector_path = "spi/injector/shatter_cone/unit_vector_major/z"
        assert units_agree(
            "1", "m", vector_path, manifest=_without_active_path(vector_path)
        )

    def test_dd_side_bug_requires_matching_units(self):
        # A matching path glob does NOT suppress an unrelated unit pair: the
        # dd_unit and correct_unit must both canonicalise as recorded.
        assert not units_agree("m^-3", "e", "core_profiles/profiles_1d/ion/z_ion")

    def test_glob_does_not_overmatch_positions(self):
        # A z-position path (dimensioned metre, SN should be metre) is NOT a
        # unit-vector-component bug — the direction globs must not match it.
        assert not units_agree("1", "m", "thomson_scattering/channel/position/z")

    @pytest.mark.graph
    def test_dd_side_bug_charge_state_bundle_bounds(self):
        # The bundle bounds are charge NUMBERS; the DD tags them `e`.
        assert units_agree("1", "e", "core_profiles/profiles_1d/ion/state/z_max")
        assert units_agree("1", "e", "waves/coherent_wave/profiles_2d/ion/state/z_min")

    @pytest.mark.graph
    def test_dd_side_bug_ggd_value_copies(self):
        # The ggd `/values` copies carry the same defect as their scalar twins.
        assert units_agree("1", "e", "edge_profiles/ggd/ion/state/z_average/values")
        assert units_agree(
            "1", "e", "plasma_profiles/ggd/ion/state/z_square_average/values"
        )

    @pytest.mark.graph
    def test_dd_side_bug_wave_vector_tagged_as_electric_field(self):
        assert units_agree(
            "m^-1", "V.m^-1", "waves/coherent_wave/profiles_1d/k_perpendicular"
        )
        assert units_agree(
            "m^-1", "V.m^-1", "waves/coherent_wave/full_wave/k_perpendicular/values"
        )

    @pytest.mark.graph
    def test_dd_side_bug_gas_flow_rate(self):
        assert units_agree(
            "Pa.m^3.s^-1", "s^-1", "spi/injector/fragmentation_gas/flow_rate"
        )
        # the correctly-tagged copies elsewhere never match this entry
        assert units_agree(
            "Pa.m^3.s^-1", "Pa.m^3.s^-1", "gas_injection/valve/flow_rate"
        )

    def test_torque_density_spelling_equivalence(self):
        assert units_agree("N.m^-2", "kg.m^-1.s^-2", "any/path")

    def test_pressure_not_equated_with_torque_density_spelling(self):
        # Pa shares the dimensionality but is a different physical quantity, so
        # it must not be swept into the torque-density equivalence set.
        assert not units_agree("Pa", "kg.m^-1.s^-2", "any/path")

    def test_genuine_mismatch_fails(self):
        assert not units_agree(
            "m", "m^-3", "pellets/time_slice/pellet/path_profiles/n_e"
        )

    def test_unparseable_dd_never_agrees(self):
        assert not units_agree("1", "unit_error", "some/path")


class TestGraphUnitCorrection:
    """Only self-contradicting DD declarations are rewritten at build time."""

    @pytest.mark.graph
    def test_legacy_reconstructed_constraint_sentinels_are_rewritten(self):
        # Each constraint carries `measured` and `reconstructed` copies of ONE
        # quantity; the reconstructed twins declare a dimensionless sentinel.
        cases = (
            ("equilibrium/time_slice/constraints/pressure/reconstructed", "Pa"),
            ("equilibrium/time_slice/constraints/n_e/reconstructed", "m^-3"),
            ("equilibrium/time_slice/constraints/j_phi/reconstructed", "A.m^-2"),
        )
        for path, expected in cases:
            assert (
                graph_unit_correction(path, "1", manifest=_without_active_path(path))
                == expected
            )

    def test_measured_twin_is_left_alone(self):
        # The correctly-declared side never matches — its unit is not the
        # sentinel the entry records.
        assert (
            graph_unit_correction(
                "equilibrium/time_slice/constraints/pressure/measured", "Pa"
            )
            is None
        )

    @pytest.mark.graph
    def test_poloidal_angle_sentinel_is_rewritten(self):
        assert (
            graph_unit_correction(
                "gyrokinetics_local/linear/wavevector/eigenmode/angle_pol", "1"
            )
            == "rad"
        )

    @pytest.mark.graph
    def test_phase_space_source_dimensionality_is_rewritten(self):
        assert (
            graph_unit_correction(
                "distribution_sources/source/ggd/particles/values", "m^-6.s^2"
            )
            == "m^-3.s^-1"
        )

    def test_suppression_only_entries_are_not_rewritten(self):
        # A DD-side bug the SN simply overrides is suppressed on the mismatch
        # axis, never rewritten in the graph.
        assert (
            graph_unit_correction("core_profiles/profiles_1d/ion/state/z_max", "e")
            is None
        )
        assert (
            graph_unit_correction("camera_ir/channel/camera/direction/x", "m") is None
        )

    @pytest.mark.graph
    def test_active_manifest_retires_matching_graph_and_comparator_authority(self):
        path = "equilibrium/time_slice/constraints/pressure/reconstructed"
        active = load_dd_resolution_manifest()
        inactive = _without_active_path(path)

        assert (
            graph_unit_correction(path, "1", dd_version="4.1.1", manifest=inactive)
            == "Pa"
        )
        assert units_agree("Pa", "1", path, dd_version="4.1.1", manifest=inactive)
        assert (
            graph_unit_correction(path, "1", dd_version="4.1.1", manifest=active)
            is None
        )
        assert not units_agree("Pa", "1", path, dd_version="4.1.1", manifest=active)

    @pytest.mark.graph
    def test_nonresolved_legacy_rule_keeps_exact_behavior(self):
        manifest = load_dd_resolution_manifest()
        path = "gyrokinetics_local/linear/wavevector/eigenmode/angle_pol"

        assert (
            graph_unit_correction(path, "1", dd_version="4.1.1", manifest=manifest)
            == "rad"
        )


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
