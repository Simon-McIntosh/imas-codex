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
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

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
    claim_sources_for_mapping,
    map_worker,
    record_mapping_verdict,
    refresh_mapping_status,
    release_mapping_claim,
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


class TestMapWorkerMapsEveryIdsOfOneSource:
    """One source selected in two IDSs is mapped once by each IDS's pass."""

    def _run(self):
        from imas_codex.ids.workers import MappingDiscoveryState

        state = MappingDiscoveryState(
            facility="jet", target_ids_list=["magnetics", "summary"]
        )
        state.assign_phase = MagicMock()
        state.assign_phase.done = True

        source = {"id": "jet:s1", "physics_domain": "magnetics"}
        edges = {
            "jet:s1": [
                _edge("magnetics/flux_loop/flux", "magnetics"),
                _edge("summary/global_quantities/ip", "summary"),
            ]
        }
        seen: set[str] = set()

        def _claim(_facility, ids_name, **_kw):
            if ids_name in seen:
                return []
            seen.add(ids_name)
            return [source]

        prepared: list[tuple[str, object]] = []

        def _prep(_facility, ids_name, assignment, context, **_kw):
            prepared.append((ids_name, assignment))
            return {"prompt": "p"}

        with (
            patch(
                "imas_codex.ids.workers.claim_sources_for_mapping", side_effect=_claim
            ),
            patch("imas_codex.ids.workers.release_mapping_claims_batch") as released,
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
            patch("imas_codex.ids.workers.refresh_mapping_status") as refresh,
        ):
            gc_cls.return_value.__enter__.return_value = MagicMock()
            asyncio.run(map_worker(state))
        return state, prepared, refresh, released

    def test_each_ids_pass_prepares_its_own_section(self):
        _state, prepared, _refresh, _released = self._run()
        assert {(ids, a.imas_target_path) for ids, a in prepared} == {
            ("magnetics", "magnetics/flux_loop"),
            ("summary", "summary/global_quantities"),
        }

    def test_each_pass_releases_the_source_it_finished(self):
        _state, _prepared, refresh, released = self._run()
        # The single owner clears the claim on every finished pass, so the
        # second IDS's pass is not blocked by the first IDS's claim.
        assert {(c.args[0], c.args[1]) for c in refresh.call_args_list} == {
            ("jet:s1", "magnetics"),
            ("jet:s1", "summary"),
        }
        released.assert_not_called()


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


@pytest.mark.graph
def test_mapping_status_rule_against_the_live_graph():
    """The status rule runs as Cypher; prove the statement, not a Python copy.

    Creates one source with a selected ``MAPPING_CANDIDATE`` edge into
    ``magnetics`` and one into ``summary`` under a unique facility, asserts
    what ``refresh_mapping_status`` writes across the three binding states,
    that a source bound only into ``magnetics`` is still claimable by the
    ``summary`` pass, and that ``has_pending_validation_work`` selects a
    source the map stage marked ``mapped``. Removes every trace in a
    ``finally`` block, so the source and its edges leave no residue.
    """
    from imas_codex.graph.client import GraphClient
    from imas_codex.ids.workers import has_pending_validation_work

    marker = uuid.uuid4().hex[:12]
    facility = f"pytest-map-status-{marker}"
    source_id = f"{facility}:fixture:source"

    with GraphClient() as gc:
        rows = gc.query(
            """
            MATCH (m:IMASNode {ids: 'magnetics'})
            MATCH (s:IMASNode {ids: 'summary'})
            RETURN m.id AS m_id, s.id AS s_id LIMIT 1
            """
        )
        assert rows, "live graph needs an IMASNode in magnetics and one in summary"
        m_id, s_id = rows[0]["m_id"], rows[0]["s_id"]

        gc.query(
            """
            MERGE (sg:SignalSource {id: $source_id})
            SET sg.facility_id = $facility, sg.status = 'enriched'
            WITH sg
            MATCH (m:IMASNode {id: $m_id}), (s:IMASNode {id: $s_id})
            MERGE (sg)-[:MAPPING_CANDIDATE {route: true, ids: 'magnetics', rank: 1}]->(m)
            MERGE (sg)-[:MAPPING_CANDIDATE {route: true, ids: 'summary', rank: 1}]->(s)
            """,
            source_id=source_id,
            facility=facility,
            m_id=m_id,
            s_id=s_id,
        )

        try:
            # No binding yet: the map pass that handled magnetics marks the
            # source 'mapped', which is what the validate stage selects on.
            assert refresh_mapping_status(source_id, "magnetics") == "mapped"
            assert has_pending_validation_work(facility) is True

            # A source bound only into magnetics is still claimable by the
            # summary pass: the claim predicate is per IDS.
            claimed = claim_sources_for_mapping(facility, "summary")
            assert [c["id"] for c in claimed] == [source_id]
            # Release the claim just taken, so the magnetics pass's skip below
            # can only come from the no-binding clause, not a held claim.
            release_mapping_claim(source_id)

            # Only magnetics is bound: the pass that validated magnetics cannot
            # call the source validated while summary remains unbound.
            gc.query(
                """
                MATCH (sg:SignalSource {id: $source_id}), (m:IMASNode {id: $m_id})
                MERGE (sg)-[:MAPS_TO_IMAS]->(m)
                """,
                source_id=source_id,
                m_id=m_id,
            )
            assert refresh_mapping_status(source_id, "magnetics") == "assigned"
            assert has_pending_validation_work(facility) is False

            # The magnetics pass skips the source it is already bound into, so
            # its own second pass cannot re-map it.
            assert claim_sources_for_mapping(facility, "magnetics") == []

            # Both IDSs bound: the source is validated.
            gc.query(
                """
                MATCH (sg:SignalSource {id: $source_id}), (s:IMASNode {id: $s_id})
                MERGE (sg)-[:MAPS_TO_IMAS]->(s)
                """,
                source_id=source_id,
                s_id=s_id,
            )
            assert refresh_mapping_status(source_id, "summary") == "validated"
        finally:
            release_mapping_claim(source_id)
            gc.query(
                "MATCH (sg:SignalSource {id: $source_id}) DETACH DELETE sg",
                source_id=source_id,
            )


class TestValidateStageUsesTheSingleStatusOwner:
    """The validate stage writes status through the one owner, not a literal."""

    def _run(self, ids_name="magnetics"):
        from imas_codex.ids.models import ValidatedMappingResult
        from imas_codex.ids.workers import MappingDiscoveryState, validate_worker

        state = MappingDiscoveryState(facility="jet", target_ids_list=[ids_name])
        state.persist = False
        state.skip_errors = True
        state.dd_version = "4.1.1"

        assignment = TargetAssignment(
            source_id="jet:s1",
            imas_target_path=f"{ids_name}/flux_loop",
            confidence=0.9,
            reasoning="selected",
        )
        state.assignments[ids_name] = TargetAssignmentBatch(
            ids_name=ids_name, assignments=[assignment]
        )
        state.mapping_batches[ids_name] = [
            (
                assignment,
                SignalMappingBatch(
                    ids_name=ids_name,
                    target_path=f"{ids_name}/flux_loop",
                    mappings=[],
                ),
            )
        ]

        async def _no_assembly(*_a, **_kw):
            return
            yield  # pragma: no cover

        validated = ValidatedMappingResult(
            facility="jet",
            ids_name=ids_name,
            dd_version="4.1.1",
            sections=[],
            bindings=[],
            escalations=[],
        )

        with (
            patch("imas_codex.ids.mapping.adiscover_assembly", _no_assembly),
            patch("imas_codex.ids.mapping.validate_mappings", return_value=validated),
            patch("imas_codex.ids.workers.GraphClient") as gc_cls,
            patch("imas_codex.ids.workers.refresh_mapping_status") as refresh,
        ):
            gc_cls.return_value = MagicMock()
            asyncio.run(validate_worker(state))
        return refresh

    def test_status_written_by_the_single_owner(self):
        refresh = self._run()
        refresh.assert_called_once_with("jet:s1", "magnetics")

    def test_no_stage_writes_a_status_literal(self):
        from imas_codex.ids import workers

        assert not hasattr(workers, "set_mapping_status")
