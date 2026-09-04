"""Tests for DD source qualifier.

The DD qualifier consolidates all DD-specific qualification logic into
``qualify_dd()``: structural Python predicates (S0-S11) and unit eligibility.
There are no YAML deny rules: all semantic quality judgments are delegated
to the LLM at compose time. These tests cover every structural check and
verify that semantically-questionable-but-structurally-sound paths (geometry,
constraints, forces) reach compose rather than being dropped here.
"""

from __future__ import annotations

import pytest

from imas_codex.standard_names.sources.base import (
    ExtractionBatch,
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
        "node_category": overrides.pop("node_category", "quantity"),
        **overrides,
    }
    return SourceCandidate.from_dd_row(row)


_CONSTRAINT_WEIGHT_PATH = "equilibrium/time_slice/constraints/faraday_angle/weight"


class _ExtractionGraph:
    """Return one ineligible node only when an explicit path bypasses filtering."""

    def __init__(self) -> None:
        self.queries: list[str] = []

    def __enter__(self) -> _ExtractionGraph:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def query(self, query: str, **_params: object) -> list[dict]:
        self.queries.append(query)
        if "MATCH (dv:DDVersion" in query:
            return [
                {
                    "dd_version": "4.1.1",
                    "cocos_version": None,
                    "cocos_params": None,
                }
            ]
        if "n.id IN $explicit_paths" in query:
            return [
                {
                    "path": _CONSTRAINT_WEIGHT_PATH,
                    "description": "Weight of the constraint",
                    "documentation": "Weight used by the equilibrium fit.",
                    "unit": "1",
                    "unit_from_rel": "1",
                    "unit_relationships": ["1"],
                    "data_type": "FLT_0D",
                    "node_category": "fit_artifact",
                    "ids_name": "equilibrium",
                    "cluster_id": None,
                    "error_node_ids": [],
                }
            ]
        return []


def _extract_with_graph(
    monkeypatch: pytest.MonkeyPatch,
    graph: _ExtractionGraph,
    **kwargs: object,
) -> list[ExtractionBatch]:
    from imas_codex.standard_names.sources import dd

    monkeypatch.setattr("imas_codex.graph.client.GraphClient", lambda: graph)
    monkeypatch.setattr(dd, "_apply_typed_dd_resolutions", lambda rows, _version: rows)
    monkeypatch.setattr(dd, "_apply_unit_overrides", lambda rows, **_kwargs: rows)
    monkeypatch.setattr(
        "imas_codex.standard_names.enrichment.enrich_paths", lambda rows: rows
    )
    monkeypatch.setattr(
        "imas_codex.standard_names.enrichment.group_by_concept_and_unit",
        lambda *_args, **_kwargs: [
            ExtractionBatch(
                source="dd",
                group_key="counterfactual",
                items=[],
                context="",
            )
        ],
    )
    return dd.extract_dd_candidates(write_skipped=False, **kwargs)


def test_normal_extraction_excludes_fit_artifact_by_node_category(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The normal graph query excludes an ineligible category."""
    graph = _ExtractionGraph()

    batches = _extract_with_graph(monkeypatch, graph, force=True)

    assert batches == []
    assert any("n.node_category IN $sn_categories" in query for query in graph.queries)


def test_explicit_extraction_enforces_the_same_node_category_guard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fixed-path extraction cannot bypass the shared category authority."""
    graph = _ExtractionGraph()

    batches = _extract_with_graph(
        monkeypatch,
        graph,
        explicit_paths=[_CONSTRAINT_WEIGHT_PATH],
    )

    assert batches == []
    assert any("n.id IN $explicit_paths" in query for query in graph.queries)


def test_category_authority_preserves_measurements_and_excludes_bookkeeping() -> None:
    """Node category alone separates measurements from fit bookkeeping."""
    humidity = _candidate(
        "camera_x_rays/detector_humidity",
        data_type="FLT_0D",
        unit="1",
        documentation=("Fraction of humidity (0-1) measured at the detector level"),
        node_category="quantity",
    )
    assert qualify_dd(humidity).eligible

    ineligible_paths = (
        _CONSTRAINT_WEIGHT_PATH,
        "equilibrium/time_slice/convergence/iterations_n",
        "transport_solver_numerics/solver_1d/equation/convergence/iterations_n",
    )
    for path in ineligible_paths:
        result = qualify_dd(
            _candidate(
                path,
                data_type="INT_0D" if path.endswith("iterations_n") else "FLT_0D",
                unit="1",
                node_category="fit_artifact",
            )
        )
        assert not result.eligible
        assert result.reason_code == "node_category_ineligible"
        assert result.status == QualificationStatus.not_physical_quantity


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
    # Eligible — semantic judgment deferred to compose, not decided here
    # -------------------------------------------------------------------
    # Generic cross-section geometry
    ("pf_active/coil/element/geometry/oblique/alpha", True, ""),
    ("pf_active/coil/element/geometry/rectangle/height", True, ""),
    ("pf_active/coil/element/geometry/annulus/radius_inner", True, ""),
    ("pf_passive/loop/element/geometry/oblique/alpha", True, ""),
    ("ferritic/element/geometry/thick_line/first_point/r", True, ""),
    # Boolean constraint selectors
    ("equilibrium/time_slice/constraints/flux_loop/exact", True, ""),
    ("equilibrium/time_slice/constraints/q/exact", True, ""),
    # Control-system force parameters
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

    def test_struct_array_not_skipped_as_string(self) -> None:
        """STRUCT_ARRAY/STRUCTURE begin 'STR' but are signal containers, not
        string leaves — S0 matches 'STR_' only, so they pass the qualifier."""
        for dt in ("STRUCT_ARRAY", "STRUCTURE"):
            q = qualify_dd(_candidate("magnetics/ip", data_type=dt))
            assert q.eligible, f"{dt} wrongly skipped: {q.reason_code}"

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
# Semantic judgment is delegated to compose, not made by the qualifier
# ============================================================================


class TestSemanticJudgmentDeferredToCompose:
    """Structurally sound paths reach compose even if their value is arguable.

    Whether a generic cross-section dimension or a boolean constraint selector
    deserves a standard name is a semantic call that needs enriched context, so
    the compose step makes it at runtime. A structural qualifier that guessed
    here would drop the path before any context existed.
    """

    @pytest.mark.parametrize(
        "path",
        [
            # Generic cross-section geometry
            "pf_active/coil/element/geometry/oblique/alpha",
            "pf_active/coil/element/geometry/rectangle/height",
            "pf_active/coil/element/geometry/annulus/radius_inner",
            "pf_passive/loop/element/geometry/oblique/alpha",
            "ic_antennas/antenna/module/strap/geometry/oblique/alpha",
            # Boolean constraint selectors
            "equilibrium/time_slice/constraints/flux_loop/exact",
            "equilibrium/time_slice/constraints/bpol_probe/exact",
            # Control system parameters
            "pf_active/coil/force_self_per_unit_length",
            "pf_active/coil/force_other_per_unit_length",
        ],
    )
    def test_structurally_sound_path_is_eligible(self, path: str) -> None:
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
