"""Tests for DD source qualifier.

The DD qualifier consolidates all DD-specific qualification logic into
``qualify_dd()``: structural Python predicates (S0-S11) and unit eligibility.
No YAML deny rules — all semantic quality judgments are delegated to the
LLM at compose time. These tests cover every structural check and verify
that formerly-denied paths (geometry, constraints, forces) are now eligible.
"""

from __future__ import annotations

import pytest

from imas_codex.standard_names.sources.base import (
    QualificationStatus,
    SourceCandidate,
)
from imas_codex.standard_names.sources.dd_qualifier import qualify_dd

# ============================================================================
# Helpers
# ============================================================================


def _candidate(path: str, **overrides: object) -> SourceCandidate:
    """Build a minimal SourceCandidate for testing."""
    row = {
        "path": path,
        "data_type": overrides.pop("data_type", "FLT_1D"),
        "unit": overrides.pop("unit", "m"),
        "description": overrides.pop("description", "Test quantity"),
        "documentation": overrides.pop("documentation", ""),
        **overrides,
    }
    return SourceCandidate.from_dd_row(row)


# ============================================================================
# Gold set — parametrised testing
# ============================================================================

# Format: (path, expected_eligible, expected_reason_code_prefix)
GOLD_SET: list[tuple[str, bool, str]] = [
    # -------------------------------------------------------------------
    # Eligible — normal physics paths
    # -------------------------------------------------------------------
    ("core_profiles/profiles_1d/electrons/temperature", True, ""),
    ("equilibrium/time_slice/profiles_1d/psi", True, ""),
    ("equilibrium/time_slice/global_quantities/ip", True, ""),
    ("core_profiles/profiles_1d/electrons/density", True, ""),
    ("magnetics/flux_loop/flux/data", True, ""),
    ("equilibrium/time_slice/profiles_1d/phi", True, ""),
    ("barometry/gauge/pressure", True, ""),
    # -------------------------------------------------------------------
    # Eligible — formerly denied by YAML (now LLM-decided)
    # -------------------------------------------------------------------
    # Geometry (formerly generic_cross_section_geometry deny rule)
    ("pf_active/coil/element/geometry/oblique/alpha", True, ""),
    ("pf_active/coil/element/geometry/rectangle/height", True, ""),
    ("pf_active/coil/element/geometry/annulus/radius_inner", True, ""),
    ("pf_passive/loop/element/geometry/oblique/alpha", True, ""),
    ("ferritic/element/geometry/thick_line/first_point/r", True, ""),
    # Constraints (formerly boolean_constraint_selector deny rule)
    ("equilibrium/time_slice/constraints/flux_loop/exact", True, ""),
    ("equilibrium/time_slice/constraints/q/exact", True, ""),
    # Forces (formerly control_system_parameter deny rule)
    ("pf_active/coil/force_self_per_unit_length", True, ""),
    ("pf_active/coil/force_other_per_unit_length", True, ""),
    # Boundary geometry — valuable physics
    ("equilibrium/time_slice/boundary/outline/r", True, ""),
    # -------------------------------------------------------------------
    # Eligible — top-level IDS time (S7 exclusion: depth < 3)
    # -------------------------------------------------------------------
    ("magnetics/time", True, ""),
    ("equilibrium/time", True, ""),
    ("barometry/time", True, ""),
    # -------------------------------------------------------------------
    # Ineligible — S1: core_instant_changes
    # -------------------------------------------------------------------
    (
        "core_instant_changes/change/profiles_1d/electrons/density",
        False,
        "duplicate_ids",
    ),
    ("core_instant_changes/vacuum_toroidal_field/b0", False, "duplicate_ids"),
    # -------------------------------------------------------------------
    # Ineligible — S2: error companion fields
    # -------------------------------------------------------------------
    (
        "core_profiles/profiles_1d/grid/rho_tor_norm_error_upper",
        False,
        "error_companion_field",
    ),
    (
        "equilibrium/time_slice/profiles_1d/psi_error_lower",
        False,
        "error_companion_field",
    ),
    (
        "equilibrium/time_slice/profiles_1d/psi_error_index",
        False,
        "error_companion_field",
    ),
    # -------------------------------------------------------------------
    # Ineligible — S3: placeholder containers
    # -------------------------------------------------------------------
    (
        "summary/local/parameter/value/constant_float_value",
        False,
        "placeholder_container",
    ),
    (
        "summary/local/parameter/value/constant_integer_value",
        False,
        "placeholder_container",
    ),
    # -------------------------------------------------------------------
    # Ineligible — S4: configurable meaning (/process/)
    # -------------------------------------------------------------------
    (
        "edge_transport/model/ggd/process/density",
        False,
        "configurable_meaning",
    ),
    # -------------------------------------------------------------------
    # Ineligible — S8: local coordinate frame unit vectors
    # -------------------------------------------------------------------
    (
        "bolometer/channel/line_of_sight/x1_unit_vector/r",
        False,
        "local_coordinate_frame",
    ),
    (
        "camera_visible/channel/detector/x3_unit_vector/phi",
        False,
        "local_coordinate_frame",
    ),
    (
        "interferometer/channel/line_of_sight/x2_unit_vector/z",
        False,
        "local_coordinate_frame",
    ),
    # -------------------------------------------------------------------
    # Ineligible — S9: GGD structural metadata
    # -------------------------------------------------------------------
    (
        "edge_profiles/grid_ggd/grid_subset/dimension",
        False,
        "ggd_structural_metadata",
    ),
    (
        "edge_profiles/grid_ggd/identifier/index",
        False,
        "ggd_structural_metadata",
    ),
    (
        "edge_profiles/grid_ggd/path",
        False,
        "ggd_structural_metadata",
    ),
    # -------------------------------------------------------------------
    # Ineligible — S10: GGD grid back-reference indices
    # -------------------------------------------------------------------
    (
        "edge_profiles/ggd/a_field/grid_index",
        False,
        "ggd_structural_metadata",
    ),
    (
        "edge_profiles/ggd/j_total/grid_subset_index",
        False,
        "ggd_structural_metadata",
    ),
]


@pytest.mark.parametrize(
    "path,expected_eligible,expected_code_prefix",
    GOLD_SET,
    ids=[g[0].rsplit("/", 1)[-1] for g in GOLD_SET],
)
def test_qualify_dd_gold_set(
    path: str,
    expected_eligible: bool,
    expected_code_prefix: str,
) -> None:
    """Gold-set parametrised test."""
    q = qualify_dd(_candidate(path))
    assert q.eligible == expected_eligible
    if expected_code_prefix:
        assert q.reason_code.startswith(expected_code_prefix)


# ============================================================================
# S0: String-typed leaves
# ============================================================================


class TestS0StringTypes:
    """S0: STR_* data types → skip."""

    def test_str_0d(self) -> None:
        q = qualify_dd(
            _candidate("core_profiles/profiles_1d/electrons/label", data_type="STR_0D")
        )
        assert not q.eligible
        assert q.reason_code == "string_data_type"

    def test_str_1d(self) -> None:
        q = qualify_dd(
            _candidate("core_profiles/profiles_1d/ion/label", data_type="STR_1D")
        )
        assert not q.eligible
        assert q.reason_code == "string_data_type"

    def test_flt_passes(self) -> None:
        q = qualify_dd(
            _candidate(
                "core_profiles/profiles_1d/electrons/temperature", data_type="FLT_1D"
            )
        )
        assert q.eligible

    def test_empty_data_type_passes(self) -> None:
        q = qualify_dd(
            _candidate("core_profiles/profiles_1d/electrons/temperature", data_type="")
        )
        assert q.eligible


# ============================================================================
# S5: Mixed units
# ============================================================================


class TestS5MixedUnits:
    """S5: mixed units → ineligible."""

    def test_mixed_unit_rejected(self) -> None:
        q = qualify_dd(_candidate("some/path/value", unit="mixed"))
        assert not q.eligible
        assert q.reason_code == "dd_unit_mixed_non_standard"

    def test_normal_unit_passes(self) -> None:
        q = qualify_dd(_candidate("some/path/value", unit="Pa"))
        assert q.eligible


# ============================================================================
# S6: Unparseable units
# ============================================================================


class TestS6UnparseableUnits:
    """S6: units that can't be parsed as valid SI → ineligible."""

    def test_unit_with_whitespace(self) -> None:
        q = qualify_dd(_candidate("some/path/value", unit="kg m"))
        assert not q.eligible
        assert q.reason_code == "dd_unit_unresolvable"

    def test_dimensionless_passes(self) -> None:
        q = qualify_dd(_candidate("some/path/value", unit="-"))
        assert q.eligible

    def test_empty_unit_passes(self) -> None:
        """Empty unit is valid (dimensionless quantity)."""
        q = qualify_dd(_candidate("some/path/value", unit=""))
        assert q.eligible


# ============================================================================
# S7: Temporal coordinate arrays
# ============================================================================


class TestS7TemporalCoordinates:
    """S7: nested time coordinate arrays → skip.

    Top-level <ids>/time paths (depth 2) are ELIGIBLE — they represent
    the IDS-level time array and may warrant a standard name.
    Nested time paths (depth >= 3) with node_category=coordinate are
    dimension axes for time-varying data — not physics quantities.
    """

    @pytest.mark.parametrize(
        "path",
        [
            "equilibrium/time_slice/time",
            "magnetics/ip/time",
            "magnetics/bpol_probe/field/time",
            "pf_active/circuit/current/time",
            "bolometer/camera/channel/power/time",
        ],
    )
    def test_nested_time_coordinate_skipped(self, path: str) -> None:
        """Deeply nested time coordinate arrays are skipped."""
        c = _candidate(path, data_type="FLT_1D", unit="s")
        # Simulate node_category=coordinate via the raw row
        c.metadata["node_category"] = "coordinate"
        q = qualify_dd(c)
        assert not q.eligible
        assert q.reason_code == "temporal_coordinate"

    @pytest.mark.parametrize(
        "path",
        [
            "magnetics/time",
            "equilibrium/time",
            "barometry/time",
            "pf_active/time",
        ],
    )
    def test_top_level_time_eligible(self, path: str) -> None:
        """Top-level <ids>/time (depth 2) is eligible."""
        c = _candidate(path, data_type="FLT_1D", unit="s")
        c.metadata["node_category"] = "coordinate"
        q = qualify_dd(c)
        assert q.eligible, f"{path} should be eligible (depth < 3)"

    def test_nested_time_without_coordinate_category_eligible(self) -> None:
        """Nested time that is NOT categorized as a coordinate passes.

        E.g., summary/disruption/time is a physics quantity (disruption time).
        """
        c = _candidate(
            "summary/disruption/time",
            data_type="FLT_0D",
            unit="s",
        )
        c.metadata["node_category"] = "quantity"
        q = qualify_dd(c)
        assert q.eligible


# ============================================================================
# S8: Local coordinate frame unit vectors
# ============================================================================


class TestS8UnitVectors:
    """S8: x1/x2/x3_unit_vector paths → skip."""

    def test_unit_vector_component(self) -> None:
        q = qualify_dd(
            _candidate(
                "bolometer/channel/line_of_sight/x1_unit_vector/r",
                unit="-",
                data_type="FLT_0D",
            )
        )
        assert not q.eligible
        assert q.reason_code == "local_coordinate_frame"

    def test_non_unit_vector_geometry_eligible(self) -> None:
        """Normal line_of_sight geometry is eligible."""
        q = qualify_dd(
            _candidate(
                "bolometer/channel/line_of_sight/first_point/r",
                unit="m",
                data_type="FLT_0D",
            )
        )
        assert q.eligible


# ============================================================================
# S9: GGD structural metadata (grid_ggd subtree)
# ============================================================================


class TestS9GGDMetadata:
    """S9: grid_ggd subtree → skip."""

    def test_grid_ggd_subtree_skipped(self) -> None:
        q = qualify_dd(
            _candidate(
                "edge_profiles/grid_ggd/grid_subset/dimension",
                unit="-",
                data_type="INT_0D",
            )
        )
        assert not q.eligible
        assert q.reason_code == "ggd_structural_metadata"

    def test_ggd_physics_value_eligible(self) -> None:
        """Physics values inside ggd/* (not grid_ggd/) are eligible."""
        q = qualify_dd(
            _candidate(
                "edge_profiles/ggd/electrons/temperature",
                unit="eV",
                data_type="FLT_1D",
            )
        )
        assert q.eligible


# ============================================================================
# S10: GGD grid back-reference indices
# ============================================================================


class TestS10GGDBackReferences:
    """S10: grid_index/grid_subset_index inside ggd paths → skip."""

    def test_grid_index_skipped(self) -> None:
        q = qualify_dd(
            _candidate(
                "edge_profiles/ggd/a_field/grid_index",
                unit="-",
                data_type="INT_0D",
            )
        )
        assert not q.eligible
        assert q.reason_code == "ggd_structural_metadata"

    def test_grid_subset_index_skipped(self) -> None:
        q = qualify_dd(
            _candidate(
                "edge_profiles/ggd/j_total/grid_subset_index",
                unit="-",
                data_type="INT_0D",
            )
        )
        assert not q.eligible
        assert q.reason_code == "ggd_structural_metadata"


# ============================================================================
# S11: Configuration flags
# ============================================================================


class TestS11ConfigurationFlags:
    """S11: boolean configuration flags → not_physical."""

    def test_flag_with_documentation(self) -> None:
        q = qualify_dd(
            _candidate(
                "gyrokinetics/wavevector/eigenmode/initial_value_run",
                unit="-",
                data_type="INT_0D",
                documentation="Flag = 1 if initial-value run; 0 if eigenvalue run",
            )
        )
        assert not q.eligible
        assert q.reason_code == "configuration_flag"
        assert q.status == QualificationStatus.not_physical_quantity

    def test_flag_zero_one_documentation(self) -> None:
        q = qualify_dd(
            _candidate(
                "some_ids/some_path/use_exact_boundary",
                unit="",
                data_type="INT_0D",
                documentation="1 if exact boundary is used, 0 if not",
            )
        )
        assert not q.eligible
        assert q.reason_code == "configuration_flag"

    def test_int_with_units_eligible(self) -> None:
        """INT_0D with real units is not a config flag."""
        q = qualify_dd(
            _candidate(
                "magnetics/bpol_probe/turns",
                unit="-",
                data_type="INT_0D",
                documentation="Number of turns in the coil",
            )
        )
        assert q.eligible

    def test_int_with_no_flag_docs_eligible(self) -> None:
        """INT_0D without flag-style docs is eligible."""
        q = qualify_dd(
            _candidate(
                "equilibrium/time_slice/boundary/type",
                unit="-",
                data_type="INT_0D",
                documentation="Index for the type of plasma boundary shape",
            )
        )
        assert q.eligible


# ============================================================================
# Formerly YAML-denied paths now eligible (semantic delegation to LLM)
# ============================================================================


class TestFormerlyDeniedNowEligible:
    """Paths that were denied by YAML rules are now eligible.

    The LLM compose step decides at runtime whether to name or skip
    these paths based on enriched context. The qualifier no longer
    blocks them.
    """

    @pytest.mark.parametrize(
        "path",
        [
            # Generic cross-section geometry (was 140 paths denied)
            "pf_active/coil/element/geometry/oblique/alpha",
            "pf_active/coil/element/geometry/rectangle/height",
            "pf_active/coil/element/geometry/annulus/radius_inner",
            "pf_passive/loop/element/geometry/oblique/alpha",
            "ic_antennas/antenna/module/strap/geometry/oblique/alpha",
            # Boolean constraint selectors (was deny rule)
            "equilibrium/time_slice/constraints/flux_loop/exact",
            "equilibrium/time_slice/constraints/bpol_probe/exact",
            # Control system parameters (was deny rule)
            "pf_active/coil/force_self_per_unit_length",
            "pf_active/coil/force_other_per_unit_length",
        ],
    )
    def test_formerly_denied_now_eligible(self, path: str) -> None:
        q = qualify_dd(_candidate(path))
        assert q.eligible, (
            f"{path} should be eligible — semantic quality judgment "
            "is delegated to LLM compose, not the qualifier."
        )


# ============================================================================
# Qualification result invariants
# ============================================================================


class TestQualificationInvariants:
    """Invariants that all qualification results must satisfy."""

    def test_eligible_has_no_reason(self) -> None:
        q = qualify_dd(_candidate("equilibrium/time_slice/profiles_1d/psi"))
        assert q.eligible
        assert q.reason_code == ""
        assert q.reason_detail == ""

    def test_ineligible_has_reason(self) -> None:
        q = qualify_dd(_candidate("core_instant_changes/vacuum_toroidal_field/b0"))
        assert not q.eligible
        assert q.reason_code != ""
        assert q.reason_detail != ""

    def test_status_is_enum(self) -> None:
        q = qualify_dd(_candidate("equilibrium/time_slice/profiles_1d/psi"))
        assert isinstance(q.status, QualificationStatus)


# ============================================================================
# SourceCandidate factory
# ============================================================================


class TestSourceCandidateFactory:
    """Test SourceCandidate.from_dd_row()."""

    def test_from_dd_row_basic(self) -> None:
        row = {
            "path": "equilibrium/time_slice/profiles_1d/psi",
            "data_type": "FLT_1D",
            "unit": "Wb",
            "description": "Poloidal flux",
            "documentation": "Full poloidal magnetic flux.",
        }
        c = SourceCandidate.from_dd_row(row)
        assert c.source_id == "equilibrium/time_slice/profiles_1d/psi"
        assert c.source_kind == "dd"
        assert c.unit == "Wb"
        assert c.hierarchy == ("equilibrium", "time_slice", "profiles_1d", "psi")
        assert c.metadata["ids_name"] == "equilibrium"
        assert c.raw is row

    def test_from_dd_row_missing_fields(self) -> None:
        """Missing fields default to empty strings."""
        c = SourceCandidate.from_dd_row({"path": "some/path"})
        assert c.unit == ""
        assert c.description == ""
        assert c.value_type == ""

    def test_from_signal_row(self) -> None:
        row = {
            "signal_id": "ip/measured",
            "description": "Plasma current",
            "units": "A",
            "physics_domain": "magnetics",
            "diagnostic": "magnetics",
            "facility": "tcv",
        }
        c = SourceCandidate.from_signal_row(row)
        assert c.source_id == "ip/measured"
        assert c.source_kind == "signals"
        assert c.unit == "A"
        assert c.metadata["facility"] == "tcv"
