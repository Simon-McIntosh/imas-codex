"""Unit tests for the derived-parent admission gate.

Pure-logic tests for ``is_admissible_parent_name`` and
``recompute_parent_kind``.  No live graph.  Clause B uses a stub
topology probe.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from imas_codex.standard_names.parents import (
    AdmissionResult,
    is_admissible_parent_name,
    is_single_child_shadow,
    recompute_parent_kind,
)


@dataclass
class _StubGraph:
    """Tiny ``query``-compatible stub for clause-B tests."""

    rows: list[dict]

    def query(self, cypher: str, **params):  # noqa: ARG002 - signature match
        return self.rows


# ---------------------------------------------------------------------------
# Clause A — structural specificity (no graph access)
# ---------------------------------------------------------------------------


class TestClauseA_StructuralSpecificity:
    @pytest.mark.parametrize(
        "name,expected_admit",
        [
            # Qualifier-bearing — admit
            ("electron_pressure", True),
            ("ion_temperature", True),
            ("upper_elongation_of_plasma_boundary", True),
            # Locus-bearing — admit
            ("elongation_of_plasma_boundary", True),
            ("radius_of_poloidal_field_coil", True),
            ("pressure_of_flux_surface", True),
            # Projection-bearing — admit
            ("radial_magnetic_field", True),
            # Bare base — reject
            ("pressure", False),
            ("density", False),
            ("temperature", False),
            ("volume", False),
            ("current", False),
            ("area", False),
            # Bare SHAPE PARAMETER — admit as a coherent family head
            # (gathers only triangularity/elongation/squareness quantities),
            # unlike the generic bare bases above.
            ("triangularity", True),
            ("elongation", True),
            ("squareness", True),
        ],
    )
    def test_admission_per_name(self, name: str, expected_admit: bool) -> None:
        result = is_admissible_parent_name(name, gc=None)
        assert isinstance(result, AdmissionResult)
        assert result.admit == expected_admit, (
            f"{name}: expected admit={expected_admit}, got {result.admit}; "
            f"reason: {result.reason}"
        )

    def test_admitted_clause_is_A(self) -> None:
        r = is_admissible_parent_name("electron_pressure", gc=None)
        assert r.admit is True
        assert r.clause == "A"

    def test_rejected_clause_is_None(self) -> None:
        r = is_admissible_parent_name("pressure", gc=None)
        assert r.admit is False
        assert r.clause is None


# ---------------------------------------------------------------------------
# Clause B — vector-like topology (graph probe required)
# ---------------------------------------------------------------------------


class TestClauseB_VectorTopology:
    def test_multi_axis_projections_admit(self) -> None:
        """A bare base with ≥2 distinct projection axes is admitted as vector."""
        gc = _StubGraph(rows=[{"axes": ["radial", "toroidal", "poloidal"]}])
        result = is_admissible_parent_name("magnetic_field", gc=gc)
        assert result.admit is True
        assert result.clause == "B"
        assert "vector-like" in result.reason

    def test_single_axis_rejects(self) -> None:
        """A bare base with only one projection axis is not vector-like."""
        gc = _StubGraph(rows=[{"axes": ["radial"]}])
        result = is_admissible_parent_name("some_bare_base", gc=gc)
        assert result.admit is False
        assert result.clause is None

    def test_no_projection_children_rejects(self) -> None:
        gc = _StubGraph(rows=[{"axes": []}])
        result = is_admissible_parent_name("density", gc=gc)
        assert result.admit is False
        assert result.clause is None

    def test_topology_query_returns_empty(self) -> None:
        gc = _StubGraph(rows=[])
        result = is_admissible_parent_name("density", gc=gc)
        assert result.admit is False
        assert result.clause is None

    def test_clause_A_takes_precedence(self) -> None:
        """Specificity-bearing names admit on A even if B would also pass."""
        gc = _StubGraph(rows=[{"axes": ["radial", "toroidal"]}])
        result = is_admissible_parent_name(
            "elongation_of_plasma_boundary",  # has locus → A admits
            gc=gc,
        )
        assert result.admit is True
        assert result.clause == "A"


# ---------------------------------------------------------------------------
# Topology-driven kind (Phase 2)
# ---------------------------------------------------------------------------


class TestRecomputeParentKind:
    def test_two_axes_is_vector(self) -> None:
        gc = _StubGraph(rows=[{"n": 2}])
        assert recompute_parent_kind("magnetic_field", gc) == "vector"

    def test_three_axes_is_vector(self) -> None:
        gc = _StubGraph(rows=[{"n": 3}])
        assert recompute_parent_kind("magnetic_field", gc) == "vector"

    def test_one_axis_is_scalar(self) -> None:
        gc = _StubGraph(rows=[{"n": 1}])
        assert recompute_parent_kind("foo", gc) == "scalar"

    def test_zero_axes_is_scalar(self) -> None:
        gc = _StubGraph(rows=[{"n": 0}])
        assert recompute_parent_kind("foo", gc) == "scalar"

    def test_tensor_pattern(self) -> None:
        gc = _StubGraph(rows=[{"n": 0}])
        assert recompute_parent_kind("metric_tensor", gc) == "tensor"

    def test_spectrum_pattern(self) -> None:
        gc = _StubGraph(rows=[{"n": 0}])
        assert recompute_parent_kind("density_spectrum", gc) == "spectrum"

    def test_eigenfunction_pattern(self) -> None:
        gc = _StubGraph(rows=[{"n": 0}])
        assert recompute_parent_kind("mhd_eigenfunction", gc) == "eigenfunction"

    def test_complex_real_part(self) -> None:
        gc = _StubGraph(rows=[{"n": 0}])
        assert recompute_parent_kind("real_part_of_foo", gc) == "complex"

    def test_topology_beats_pattern(self) -> None:
        """≥2 projection axes wins even if the name has a tensor/spectrum
        token — the topology signal is stronger."""
        gc = _StubGraph(rows=[{"n": 2}])
        # Hypothetical name carrying both signals
        assert recompute_parent_kind("foo_tensor", gc) == "vector"


# ---------------------------------------------------------------------------
# Integration: gate filters edges at write time
# ---------------------------------------------------------------------------


class TestWriteEdgesAdmissionIntegration:
    """The Phase 1 _filter_admissible_parents drops bare-base parents.

    These tests verify the integration with ``_write_standard_name_edges``
    via the same mock fixture used by ``test_graph_edge_writers``.
    """

    def test_bare_base_parent_edge_dropped(self) -> None:
        """A child whose only parent is a bare-base name produces no write."""
        from unittest.mock import MagicMock, patch

        from imas_codex.standard_names.graph_ops import write_standard_names

        gc = MagicMock()
        gc.query = MagicMock(return_value=[])

        with patch("imas_codex.standard_names.graph_ops.GraphClient") as MockGC:
            MockGC.return_value.__enter__ = MagicMock(return_value=gc)
            MockGC.return_value.__exit__ = MagicMock(return_value=False)
            with (
                patch(
                    "imas_codex.standard_names.protection.filter_protected",
                    side_effect=lambda n, **kw: (n, []),
                ),
                patch(
                    "imas_codex.standard_names.graph_ops._write_grammar_decomposition",
                    return_value=[],
                ),
            ):
                # electron_pressure peels to qualifier 'electron' → parent 'pressure'.
                # 'pressure' is a bare base — admission gate should drop the edge.
                write_standard_names([{"id": "electron_pressure", "unit": "Pa"}])

        # Find HAS_PARENT write cyphers (skip probe queries)
        write_calls = [
            c
            for c in gc.query.call_args_list
            if "HAS_PARENT" in c[0][0]
            and "MERGE" in c[0][0]
            and "batch" in (c[1] or {})
        ]
        # If a write was emitted, it MUST NOT contain pressure as target
        for call_args in write_calls:
            batch = call_args[1].get("batch") or []
            for row in batch:
                assert row.get("to_name") != "pressure", (
                    f"Bare-base parent 'pressure' should be dropped by gate, "
                    f"but found edge: {row}"
                )

    def test_admissible_parent_edge_kept(self) -> None:
        """A child whose parent passes Clause A produces a write."""
        from unittest.mock import MagicMock, patch

        from imas_codex.standard_names.graph_ops import write_standard_names

        gc = MagicMock()
        gc.query = MagicMock(return_value=[])

        with patch("imas_codex.standard_names.graph_ops.GraphClient") as MockGC:
            MockGC.return_value.__enter__ = MagicMock(return_value=gc)
            MockGC.return_value.__exit__ = MagicMock(return_value=False)
            with (
                patch(
                    "imas_codex.standard_names.protection.filter_protected",
                    side_effect=lambda n, **kw: (n, []),
                ),
                patch(
                    "imas_codex.standard_names.graph_ops._write_grammar_decomposition",
                    return_value=[],
                ),
            ):
                # maximum_of_electron_temperature → parent electron_temperature (Clause A)
                write_standard_names(
                    [{"id": "maximum_of_electron_temperature", "unit": "eV"}]
                )

        write_calls = [
            c
            for c in gc.query.call_args_list
            if "HAS_PARENT" in c[0][0]
            and "MERGE" in c[0][0]
            and "batch" in (c[1] or {})
        ]
        # Find at least one edge → electron_temperature
        found = False
        for call_args in write_calls:
            batch = call_args[1].get("batch") or []
            for row in batch:
                if row.get("to_name") == "electron_temperature":
                    found = True
                    break
        assert found, "Admissible parent edge should be written"


# ---------------------------------------------------------------------------
# Class-B suppression veto — single-child shadow of an accepted sibling
# ---------------------------------------------------------------------------


class TestSingleChildShadowVeto:
    """A derived parent that is merely a less-specific spelling of one
    accepted child sourced from the same DD path must NOT be admitted —
    materialising it produces a second accepted name competing for the
    source (the Class-B duplicate). Genuine shared parents (≥2 children)
    survive untouched.
    """

    def _shadow_row(
        self,
        *,
        child_id: str,
        child_sources: list[str],
        parent_sources: list[str] | None = None,
        child_stage: str = "accepted",
        child_origin: str = "pipeline",
    ) -> dict:
        return {
            "child_id": child_id,
            "child_sources": child_sources,
            "parent_sources": parent_sources or [],
            # is_single_child_shadow only reads child_id/sources; the gate's
            # batch path also reads stage/origin, so include them for parity.
            "child_stage": child_stage,
            "child_origin": child_origin,
        }

    def test_single_child_shadow_suppressed(self):
        """``radius_of_divertor_target`` shadows its lone accepted child
        ``major_radius_of_divertor_target`` from the same DD source."""
        gc = _StubGraph(
            rows=[
                self._shadow_row(
                    child_id="major_radius_of_divertor_target",
                    child_sources=["divertors/divertor/target/tile/surface_outline/r"],
                )
            ]
        )
        suppress, reason = is_single_child_shadow("radius_of_divertor_target", gc)
        assert suppress is True
        assert "major_radius_of_divertor_target" in reason

    def test_admission_vetoes_single_child_shadow(self):
        """The veto wins even though the name passes Clause A (has a locus)."""
        gc = _StubGraph(
            rows=[
                self._shadow_row(
                    child_id="major_radius_of_divertor_target",
                    child_sources=["divertors/divertor/target/tile/surface_outline/r"],
                )
            ]
        )
        result = is_admissible_parent_name("radius_of_divertor_target", gc=gc)
        assert result.admit is False
        assert result.clause is None
        assert "suppressed" in result.reason

    def test_no_children_not_suppressed(self):
        """A candidate with no single child is not a shadow."""
        gc = _StubGraph(rows=[])
        suppress, _ = is_single_child_shadow("radius_of_divertor_target", gc)
        assert suppress is False

    def test_child_without_source_not_suppressed(self):
        """A lone child with no DD source cannot claim source-equivalence."""
        gc = _StubGraph(
            rows=[self._shadow_row(child_id="some_child", child_sources=[])]
        )
        suppress, reason = is_single_child_shadow("some_parent", gc)
        assert suppress is False
        assert "no DD source" in reason

    def test_independently_sourced_parent_not_suppressed(self):
        """A parent that owns a DD source the child does not is a real
        sourced name, not a pure shadow — keep it."""
        gc = _StubGraph(
            rows=[
                self._shadow_row(
                    child_id="major_radius_of_divertor_target",
                    child_sources=["divertors/divertor/target/tile/surface_outline/r"],
                    parent_sources=["some/other/dd/path/radius"],
                )
            ]
        )
        suppress, reason = is_single_child_shadow("radius_of_divertor_target", gc)
        assert suppress is False
        assert "independently sourced" in reason

    def test_genuine_shared_parent_survives(self):
        """A parent with ≥2 distinct children is never a single-child shadow.

        The probe Cypher filters to ``size(children) = 1`` so a genuine shared
        parent (``temperature`` over ``electron_temperature`` AND
        ``ion_temperature``) returns no shadow row.
        """
        gc = _StubGraph(rows=[])  # multi-child parent yields no shadow row
        suppress, _ = is_single_child_shadow("temperature", gc)
        assert suppress is False
        # And it still admits via Clause B if it has vector topology, or stays
        # subject to the normal bare-base rejection — the veto does not fire.
        result = is_admissible_parent_name("temperature", gc=gc)
        assert result.admit is False  # bare base, no projection children
        assert "suppressed" not in result.reason


class TestFilterAdmissibleParentsShadowVeto:
    """Integration: ``_filter_admissible_parents`` drops a single-child-shadow
    HAS_PARENT edge but keeps a genuine shared-parent edge.
    """

    def _probe_rows(self, rows: list[dict]):
        """Build a GraphClient stub returning *rows* from the bulk probe."""
        from unittest.mock import MagicMock

        gc = MagicMock()
        gc.query = MagicMock(return_value=rows)
        return gc

    def test_shadow_edge_dropped(self):
        from imas_codex.standard_names.graph_ops import _filter_admissible_parents

        # The bulk probe reports: parent has exactly one child
        # (major_radius_of_divertor_target), sourced from the divertor path,
        # parent owns no independent source.
        gc = self._probe_rows(
            [
                {
                    "name": "radius_of_divertor_target",
                    "axes": [],
                    "child_ids": ["major_radius_of_divertor_target"],
                    "lone_child_id": "major_radius_of_divertor_target",
                    "lone_child_stage": "accepted",
                    "lone_child_origin": "pipeline",
                    "parent_sources": [],
                    "lone_child_sources": [
                        "divertors/divertor/target/tile/surface_outline/r"
                    ],
                    "origin": "derived",
                    "pipeline_status": None,
                }
            ]
        )
        co_batch = [
            {
                "from_name": "major_radius_of_divertor_target",
                "to_name": "radius_of_divertor_target",
                "operator_kind": "qualifier",
                "axis": None,
            }
        ]
        kept = _filter_admissible_parents(co_batch, gc)
        assert kept == [], "single-child shadow parent edge must be dropped"

    def test_shared_parent_edge_kept(self):
        from imas_codex.standard_names.graph_ops import _filter_admissible_parents

        # 'pressure' is a bare base, but here it groups TWO children
        # (electron_pressure + ion_pressure). The probe reports >1 child so the
        # shadow veto cannot fire; the edge survives (admitted as a real
        # multi-child grouping target through the normal gate).
        gc = self._probe_rows(
            [
                {
                    "name": "electron_pressure",
                    "axes": [],
                    "child_ids": [],
                    "lone_child_id": None,
                    "lone_child_stage": None,
                    "lone_child_origin": None,
                    "parent_sources": [],
                    "lone_child_sources": [],
                    "origin": None,
                    "pipeline_status": None,
                }
            ]
        )
        # electron_pressure passes Clause A (qualifier 'electron'); two batch
        # children both point at it, so even if it were bare it would not be a
        # single-child shadow.
        co_batch = [
            {
                "from_name": "maximum_of_electron_pressure",
                "to_name": "electron_pressure",
                "operator_kind": "unary_prefix",
                "axis": None,
            },
            {
                "from_name": "minimum_of_electron_pressure",
                "to_name": "electron_pressure",
                "operator_kind": "unary_prefix",
                "axis": None,
            },
        ]
        kept = _filter_admissible_parents(co_batch, gc)
        kept_targets = {e["to_name"] for e in kept}
        assert "electron_pressure" in kept_targets
