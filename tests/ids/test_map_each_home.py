"""Tests for the per-IDS map stage.

The map stage no longer reads in-memory assignments the assign stage filled.
Each IDS pass claims the sources that selected a home in that IDS and are not
yet bound there, builds one ``TargetAssignment`` per selected section from
their selected ``MAPPING_CANDIDATE`` edges, guards each chosen ``target_id``
against the source's selected sections, and refreshes the per-source
``mapping_status`` summary.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from imas_codex.ids.mapping import (
    _target_type_from_section,
    build_target_assignments,
    validate_mappings,
)
from imas_codex.ids.models import (
    SignalMappingBatch,
    SignalMappingEntry,
    TargetAssignment,
    TargetAssignmentBatch,
)
from imas_codex.ids.workers import (
    _mapping_status_for,
    claim_sources_for_mapping,
    map_worker,
    record_mapping_verdict,
)

_NO_BINDING = "NOT EXISTS { (n)-[:MAPS_TO_IMAS]->(:IMASNode {ids: $ids_name}) }"


def _edge(
    path,
    ids,
    rank=1,
    route=True,
    data_type="STRUCT_ARRAY",
    timebasepath=None,
    ndim=None,
):
    return {
        "path": path,
        "ids": ids,
        "rank": rank,
        "route": route,
        "section": "/".join(path.split("/")[:2]),
        "data_type": data_type,
        "timebasepath": timebasepath,
        "ndim": ndim,
        "choice_probability": 0.7,
        "p_same_quantity": 0.6,
    }


class TestMapClaimPredicate:
    def test_claims_selected_edge_without_binding(self):
        with patch("imas_codex.ids.workers.claim_batch", return_value=[]) as mock:
            claim_sources_for_mapping("jet", "magnetics")
        predicate = mock.call_args.kwargs["status_predicate"]
        assert "MAPPING_CANDIDATE" in predicate
        assert "c.route = true" in predicate
        assert "c.ids = $ids_name" in predicate
        assert _NO_BINDING in predicate
        assert mock.call_args.kwargs["status_params"] == {"ids_name": "magnetics"}

    def test_bound_source_is_skipped_by_the_no_binding_clause(self):
        # The no-binding clause is what makes a second pass over the same IDS
        # skip a source already bound into it.
        with patch("imas_codex.ids.workers.claim_batch", return_value=[]) as mock:
            claim_sources_for_mapping("jet", "magnetics")
        assert _NO_BINDING in mock.call_args.kwargs["status_predicate"]


class TestTargetTypeFromSection:
    def test_time_indexed_array_is_time_slice(self):
        edge = _edge(
            "equilibrium/time_slice/psi",
            "equilibrium",
            timebasepath="equilibrium/time",
        )
        assert _target_type_from_section(edge) == "time_slice"

    def test_other_array_is_struct_array(self):
        assert _target_type_from_section(_edge("pf_active/coil/data", "pf_active")) == (
            "struct_array"
        )

    def test_zero_dim_is_scalar(self):
        edge = _edge(
            "operational_instrumentation/latency",
            "operational_instrumentation",
            data_type="FLT_0D",
            ndim=0,
        )
        assert _target_type_from_section(edge) == "scalar"


class TestBuildTargetAssignments:
    def test_section_is_top_level_ancestor_and_type_from_section(self):
        edges = {"jet:s1": [_edge("magnetics/flux_loop/flux/data", "magnetics")]}
        batch = build_target_assignments("magnetics", ["jet:s1"], edges)
        assert len(batch.assignments) == 1
        assert batch.assignments[0].imas_target_path == "magnetics/flux_loop"
        assert batch.assignments[0].target_type.value == "struct_array"

    def test_two_sections_in_one_ids_give_two_assignments(self):
        edges = {
            "jet:s1": [
                _edge("magnetics/flux_loop/flux", "magnetics"),
                _edge("magnetics/b_field_pol_probe/field", "magnetics"),
            ]
        }
        batch = build_target_assignments("magnetics", ["jet:s1"], edges)
        assert {a.imas_target_path for a in batch.assignments} == {
            "magnetics/flux_loop",
            "magnetics/b_field_pol_probe",
        }

    def test_each_ids_pass_builds_only_its_own_sections(self):
        edges = {
            "jet:s1": [
                _edge("magnetics/flux_loop/flux", "magnetics"),
                _edge("summary/global_quantities/ip", "summary"),
            ]
        }
        magnetics = build_target_assignments("magnetics", ["jet:s1"], edges)
        summary = build_target_assignments("summary", ["jet:s1"], edges)
        assert [a.imas_target_path for a in magnetics.assignments] == [
            "magnetics/flux_loop"
        ]
        assert [a.imas_target_path for a in summary.assignments] == [
            "summary/global_quantities"
        ]

    def test_unselected_and_other_ids_edges_are_ignored(self):
        edges = {
            "jet:s1": [
                _edge("magnetics/flux_loop/flux", "magnetics", route=False),
                _edge("summary/ip", "summary"),
            ]
        }
        assert (
            build_target_assignments("magnetics", ["jet:s1"], edges).assignments == []
        )


class TestMappingStatusSummary:
    def test_assigned_until_every_ids_is_bound(self):
        assert (
            _mapping_status_for(["magnetics", "summary"], {"magnetics"}) == "assigned"
        )

    def test_mapped_once_every_ids_is_bound(self):
        assert (
            _mapping_status_for(["magnetics", "summary"], {"magnetics", "summary"})
            == "mapped"
        )

    def test_no_selected_edge_leaves_status_untouched(self):
        assert _mapping_status_for([], set()) is None


class TestRecordMappingVerdict:
    def test_records_disposition_and_evidence_without_touching_status(self):
        gc = MagicMock()
        with patch("imas_codex.ids.workers.GraphClient") as gc_cls:
            gc_cls.return_value.__enter__.return_value = gc
            record_mapping_verdict("jet:coil:1", "no_imas_equivalent", "not listed")
        query = gc.query.call_args.args[0]
        assert "mapping_disposition" in query
        assert "mapping_evidence" in query
        assert "mapping_status" not in query


class TestSectionGuard:
    def _sections(self):
        return TargetAssignmentBatch(
            ids_name="magnetics",
            assignments=[
                TargetAssignment(
                    source_id="jet:s1",
                    imas_target_path="magnetics/flux_loop",
                    confidence=0.9,
                    reasoning="selected",
                )
            ],
        )

    def _batch(self, target_id):
        return SignalMappingBatch(
            ids_name="magnetics",
            target_path="magnetics/flux_loop",
            mappings=[
                SignalMappingEntry(
                    source_id="jet:s1",
                    source_property="value",
                    target_id=target_id,
                    transform_expression="value",
                    confidence=0.9,
                    reasoning="r",
                )
            ],
        )

    @staticmethod
    def _report():
        report = MagicMock()
        report.escalations = []
        report.duplicate_targets = []
        report.all_passed = True
        report.binding_checks = []
        return report

    def test_outside_every_selected_section_is_escalated(self):
        gc = MagicMock()
        with (
            patch(
                "imas_codex.ids.validation.validate_mapping",
                return_value=self._report(),
            ),
            patch(
                "imas_codex.ids.validation.check_coverage_threshold",
                return_value=[],
            ),
            patch("imas_codex.ids.tools.get_sign_flip_paths", return_value=set()),
        ):
            result = validate_mappings(
                "jet",
                "magnetics",
                "4.1.1",
                self._sections(),
                [self._batch("summary/ip")],
                gc=gc,
            )
        assert result.bindings == []
        assert any(
            "outside every selected section" in e.reason for e in result.escalations
        )

    def test_inside_section_outside_selected_node_is_kept(self):
        gc = MagicMock()
        with (
            patch(
                "imas_codex.ids.validation.validate_mapping",
                return_value=self._report(),
            ),
            patch(
                "imas_codex.ids.validation.check_coverage_threshold",
                return_value=[],
            ),
            patch("imas_codex.ids.tools.get_sign_flip_paths", return_value=set()),
        ):
            result = validate_mappings(
                "jet",
                "magnetics",
                "4.1.1",
                self._sections(),
                [self._batch("magnetics/flux_loop/sibling/data")],
                gc=gc,
            )
        assert len(result.bindings) == 1


class TestMapWorkerMapsFromSelectedEdges:
    def _run(self):
        from imas_codex.ids.workers import MappingDiscoveryState

        state = MappingDiscoveryState(facility="jet", target_ids_list=["magnetics"])
        state.assign_phase = MagicMock()
        state.assign_phase.done = True
        state.contexts["magnetics"] = {"groups": []}

        source = {"id": "jet:s1", "physics_domain": "magnetics"}
        edges = {"jet:s1": [_edge("magnetics/flux_loop/flux", "magnetics")]}
        calls = {"n": 0}

        def _claim(*_a, **_k):
            calls["n"] += 1
            return [source] if calls["n"] == 1 else []

        prepared = []

        def _prep(_facility, ids_name, assignment, context, **_kw):
            prepared.append(assignment)
            return {"prompt": "p"}

        with (
            patch(
                "imas_codex.ids.workers.claim_sources_for_mapping", side_effect=_claim
            ),
            patch("imas_codex.ids.workers.GraphClient") as gc_cls,
            patch("imas_codex.ids.graph_ops.read_candidates", return_value=edges),
            patch("imas_codex.ids.mapping._prepare_section_context", side_effect=_prep),
            patch("imas_codex.ids.mapping._build_messages", return_value=[]),
            patch(
                "imas_codex.ids.mapping._acall_llm",
                new=AsyncMock(
                    return_value=SignalMappingBatch(
                        ids_name="magnetics",
                        target_path="magnetics/flux_loop",
                        mappings=[],
                    )
                ),
            ),
            patch("imas_codex.ids.workers.refresh_mapping_status"),
        ):
            gc_cls.return_value.__enter__.return_value = MagicMock()
            asyncio.run(map_worker(state))
        return state, prepared

    def test_selected_source_is_mapped_from_its_selected_edges(self):
        state, prepared = self._run()
        assert len(prepared) == 1
        assert prepared[0].imas_target_path == "magnetics/flux_loop"
        assert prepared[0].source_id == "jet:s1"
        assert state.sources_mapped == 1


class TestAssemblerSectionFilter:
    def test_sibling_section_sharing_a_name_prefix_is_not_matched(self):
        from imas_codex.ids.assembler import IDSAssembler

        inst = object.__new__(IDSAssembler)
        inst.facility = "jet"
        inst.ids_name = "pf_active"

        class _Binding:
            def __init__(self, target):
                self.target_id = target

        mapping = MagicMock()
        mapping.bindings = [
            _Binding("pf_active/coil/element/r"),
            _Binding("pf_active/coil_extra/element/r"),
        ]
        section = {
            "root_path": "pf_active/coil",
            "structure": "array_per_node",
            "init_arrays": "{}",
            "elements_config": "{}",
        }
        seen = []
        inst._apply_mappings = lambda *a, **_k: seen.append(a[2])

        with (
            patch(
                "imas_codex.ids.assembler.select_nodes",
                return_value=[{"path": "pf_active/coil/element/coil:1"}],
            ),
        ):
            inst._build_graph_section(
                MagicMock(), section, "epoch", mapping, MagicMock()
            )

        assert len(seen) == 1
        assert [m.target_id for m in seen[0]] == ["pf_active/coil/element/r"]


class TestProgressAssignCost:
    def test_sections_row_reports_the_choose_targets_cost(self):
        from rich.console import Console

        from imas_codex.ids.progress import MappingProgressDisplay

        display = MappingProgressDisplay(
            "jet",
            ["magnetics"],
            cost_limit=10.0,
            console=Console(force_terminal=False, width=200),
        )
        display.state.cost.steps["choose_targets"] = 0.25
        text = display._build_pipeline_section()
        assert "$0.25" in text.plain
