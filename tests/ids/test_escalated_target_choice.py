"""The escalated source's target choice: shortlist, refusal, selection, bridge.

The assign stage no longer assigns every source in one all-sources call. A
source the candidate stage selected already carries its marked edges; an
escalated source is asked, from only its own shortlist, which listed paths
hold its values, and each picked path becomes a selected edge. A path outside
that shortlist is refused. A no_candidate source is skipped, and the semantic
bridge is computed only for the escalated sources.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from imas_codex.ids.graph_ops import CandidateWriteError
from imas_codex.ids.mapping import (
    PipelineCost,
    _format_shortlist,
    _render_choice_prompt,
    choose_targets,
    escalated_shortlist,
)
from imas_codex.ids.models import TargetChoice, TargetChoiceBatch


def _edge(path, rank, ids="magnetics", arms=None):
    return {
        "path": path,
        "rank": rank,
        "ids": ids,
        "arms": arms or [],
        "p_same_quantity": 0.5,
        "documentation": f"doc for {path}",
    }


def _batch(source_id, paths, disposition=None):
    return TargetChoiceBatch(
        choices=[
            TargetChoice(
                source_id=source_id,
                paths=paths,
                disposition=disposition,
                confidence=0.8,
                reasoning="test",
            )
        ]
    )


class TestEscalatedShortlist:
    def test_takes_top_n_in_jev_order(self):
        edges = [_edge(f"magnetics/p{i}", i) for i in range(1, 9)]
        short = escalated_shortlist(edges, 5)
        assert [e["path"] for e in short] == [f"magnetics/p{i}" for i in range(1, 6)]

    def test_carries_every_cross_ids_sibling_regardless_of_rank(self):
        edges = [_edge(f"magnetics/p{i}", i) for i in range(1, 9)] + [
            _edge("summary/ip", 20, ids="summary", arms=["cluster"])
        ]
        short = escalated_shortlist(edges, 5)
        paths = [e["path"] for e in short]
        assert paths[:5] == [f"magnetics/p{i}" for i in (1, 2, 3, 4, 5)]
        assert "summary/ip" in paths

    def test_sibling_already_in_top_n_is_not_duplicated(self):
        edges = [
            _edge("magnetics/p1", 1),
            _edge("summary/ip", 2, ids="summary", arms=["cluster"]),
        ]
        short = escalated_shortlist(edges, 5)
        assert [e["path"] for e in short].count("summary/ip") == 1


class TestShortlistPrompt:
    def test_prompt_lists_only_shortlist_paths(self):
        shortlist = [
            _edge("magnetics/ip", 1),
            _edge("summary/ip", 2, ids="summary", arms=["cluster"]),
        ]
        rendered = _render_choice_prompt(
            "jet",
            {"id": "jet:coil:1", "physics_domain": "magnetics"},
            shortlist,
            None,
        )
        assert "magnetics/ip" in rendered
        assert "summary/ip" in rendered
        assert "[cross-IDS sibling]" in rendered
        assert "core_profiles/psi" not in rendered

    def test_format_shortlist_marks_siblings_only(self):
        text = _format_shortlist(
            [
                _edge("magnetics/ip", 1),
                _edge("summary/ip", 2, ids="summary", arms=["cluster"]),
            ]
        )
        lines = text.split("\n")
        assert "[cross-IDS sibling]" not in lines[0]
        assert "[cross-IDS sibling]" in lines[1]


class TestChoiceRefusal:
    @patch("imas_codex.ids.mapping._call_llm")
    def test_choice_refuses_path_outside_shortlist(self, mock_call_llm):
        shortlist = [_edge("magnetics/ip", 1), _edge("magnetics/coil", 2)]
        mock_call_llm.return_value = _batch(
            "jet:coil:1", ["magnetics/ip", "core_profiles/profiles_1d/electrons"]
        )
        with pytest.raises(CandidateWriteError):
            choose_targets("jet", {"id": "jet:coil:1"}, shortlist, cost=PipelineCost())

    @patch("imas_codex.ids.mapping._call_llm")
    def test_choice_accepts_only_listed_paths(self, mock_call_llm):
        shortlist = [_edge("magnetics/ip", 1), _edge("magnetics/coil", 2)]
        mock_call_llm.return_value = _batch("jet:coil:1", ["magnetics/ip"])
        choice = choose_targets(
            "jet", {"id": "jet:coil:1"}, shortlist, cost=PipelineCost()
        )
        assert choice.paths == ["magnetics/ip"]

    @patch("imas_codex.ids.mapping._call_llm")
    def test_empty_choice_requires_a_disposition(self, mock_call_llm):
        shortlist = [_edge("magnetics/ip", 1)]
        mock_call_llm.return_value = _batch("jet:coil:1", [])
        with pytest.raises(CandidateWriteError):
            choose_targets("jet", {"id": "jet:coil:1"}, shortlist, cost=PipelineCost())

    @patch("imas_codex.ids.mapping._call_llm")
    def test_paths_with_a_disposition_are_refused(self, mock_call_llm):
        from imas_codex.ids.models import MappingDisposition

        shortlist = [_edge("magnetics/ip", 1)]
        mock_call_llm.return_value = _batch(
            "jet:coil:1",
            ["magnetics/ip"],
            disposition=MappingDisposition.NO_IMAS_EQUIVALENT,
        )
        with pytest.raises(CandidateWriteError):
            choose_targets("jet", {"id": "jet:coil:1"}, shortlist, cost=PipelineCost())


class TestSemanticBridgeNarrowed:
    @patch("imas_codex.ids.mapping.fetch_code_context", return_value=[])
    @patch("imas_codex.ids.mapping.fetch_wiki_context", return_value=[])
    @patch("imas_codex.ids.mapping.compute_semantic_matches", return_value={})
    @patch("imas_codex.ids.mapping.query_signal_sources")
    @patch(
        "imas_codex.ids.mapping.query_ids_physics_domains", return_value=["magnetics"]
    )
    @patch("imas_codex.embeddings.encoder.Encoder")
    def test_bridge_runs_only_for_escalated_sources(
        self,
        mock_encoder,
        mock_domains,
        mock_sources,
        mock_matches,
        mock_wiki,
        mock_code,
    ):
        from imas_codex.ids.mapping import gather_shared_context

        mock_sources.return_value = [
            {"id": "jet:s1", "description": "d1", "rep_description": "d1"},
            {"id": "jet:s2", "description": "d2", "rep_description": "d2"},
        ]
        mock_encoder.return_value.embed_texts.return_value = [[0.1], [0.2]]

        gc = MagicMock()

        def _query(query, **kwargs):
            if "candidate_route = 'escalated'" in query:
                return [{"id": "jet:s1"}]
            return []

        gc.query.side_effect = _query

        shared = gather_shared_context("jet", ["magnetics"], gc=gc)

        assert mock_matches.call_count == 1
        escalated_descs = mock_matches.call_args[0][0]
        assert [sid for sid, _ in escalated_descs] == ["jet:s1"]
        assert shared["semantic_match_matrix"] == {}


class TestAssignWorkerSelects:
    def _state(self):
        from imas_codex.ids.workers import MappingDiscoveryState

        state = MappingDiscoveryState(facility="jet", target_ids_list=["magnetics"])
        state.assign_phase = MagicMock(done=True)
        return state

    def _run(self, choice_paths):
        import asyncio

        from imas_codex.ids.workers import assign_worker

        source = {"id": "jet:coil:1", "physics_domain": "magnetics"}
        edges = {
            "jet:coil:1": [
                _edge("magnetics/ip", 1),
                _edge("equilibrium/time_slice/constraints/ip", 2, ids="equilibrium"),
            ]
        }
        gc = MagicMock()
        with (
            patch(
                "imas_codex.ids.workers.claim_sources_for_escalated",
                return_value=[source],
            ),
            patch("imas_codex.ids.workers.GraphClient") as mock_gc_cls,
            patch("imas_codex.ids.graph_ops.read_candidates", return_value=edges),
            patch("imas_codex.ids.graph_ops.select_candidates") as mock_select,
            patch(
                "imas_codex.ids.mapping.achoose_targets",
                return_value=_batch("jet:coil:1", choice_paths).choices[0],
            ),
        ):
            mock_gc_cls.return_value.__enter__.return_value = gc
            asyncio.run(assign_worker(self._state()))
        return mock_select

    def test_escalated_pick_marks_selected(self):
        mock_select = self._run(["magnetics/ip"])
        mock_select.assert_called_once()
        assert mock_select.call_args[0][0] == "jet:coil:1"
        assert mock_select.call_args[0][1] == ["magnetics/ip"]

    def test_two_paths_in_two_idss_marked_both(self):
        mock_select = self._run(
            ["magnetics/ip", "equilibrium/time_slice/constraints/ip"]
        )
        assert mock_select.call_args[0][1] == [
            "magnetics/ip",
            "equilibrium/time_slice/constraints/ip",
        ]


class TestNoCandidateLogged:
    def _state(self):
        from imas_codex.ids.workers import CandidateDiscoveryState

        state = CandidateDiscoveryState(facility="jet")
        state.candidate_phase = MagicMock(done=True)
        return state

    @pytest.fixture(autouse=True)
    def _decisions_key(self, monkeypatch):
        """Set a decisions key so the worker reaches the judgement it replaces.

        The worker ends the stage before its first claim when the key is
        absent, so an unset key would test the skip instead of the routing.
        """
        monkeypatch.setenv("OPENROUTER_API_KEY_IMAS_CODEX", "test-key")

    def test_candidate_worker_skips_and_logs_no_candidate(self, caplog):
        import asyncio
        import logging

        from imas_codex.ids.candidates import PairJudgment, Route
        from imas_codex.ids.workers import candidate_worker

        source = {"id": "jet:s1", "description": "coil current"}
        judgments = [
            PairJudgment(
                path="magnetics/ip", p_same_quantity=0.41, model="m", judged_at="t"
            ),
            PairJudgment(
                path="magnetics/other", p_same_quantity=0.02, model="m", judged_at="t"
            ),
        ]
        decision = Route(
            decision="no_candidate", shortlist=tuple(judgments), selected=frozenset()
        )
        gc = MagicMock()
        with (
            patch(
                "imas_codex.ids.workers.claim_sources_for_candidates",
                side_effect=[[source], []],
            ),
            patch("imas_codex.ids.workers.GraphClient") as mock_gc_cls,
            patch("imas_codex.ids.workers._facility_block", return_value={}),
            patch("imas_codex.ids.workers.route_ids", return_value=["magnetics"]),
            patch(
                "imas_codex.ids.workers.retrieve_candidates",
                return_value={"jet:s1": [MagicMock()]},
            ),
            patch("imas_codex.ids.workers.judge_candidates", return_value=judgments),
            patch(
                "imas_codex.ids.workers.expand_cluster_siblings",
                return_value=([], []),
            ),
            patch("imas_codex.ids.workers.route", return_value=decision),
            patch("imas_codex.ids.workers._candidate_records", return_value=[]),
            patch("imas_codex.ids.workers.write_candidates", return_value=0),
        ):
            mock_gc_cls.return_value.__enter__.return_value = gc
            with caplog.at_level(logging.INFO):
                asyncio.run(candidate_worker(self._state()))

        assert "No candidate for jet:s1: best path magnetics/ip" in caplog.text
        assert "0.410" in caplog.text
        assert "magnetics/other" not in caplog.text


class TestNoPathVerdictPersisted:
    """A 'none' choice is paid once: its verdict is persisted, not released."""

    def _state(self):
        from imas_codex.ids.workers import MappingDiscoveryState

        state = MappingDiscoveryState(facility="jet", target_ids_list=["magnetics"])
        state.assign_phase = MagicMock(done=True)
        return state

    def _run(self, disposition):
        import asyncio

        from imas_codex.ids.models import TargetChoice
        from imas_codex.ids.workers import assign_worker

        source = {"id": "jet:coil:1", "physics_domain": "magnetics"}
        edges = {"jet:coil:1": [_edge("magnetics/ip", 1)]}
        gc = MagicMock()
        choice = TargetChoice(
            source_id="jet:coil:1",
            paths=[],
            disposition=disposition,
            confidence=0.3,
            reasoning="no IMAS node carries this value",
        )
        with (
            patch(
                "imas_codex.ids.workers.claim_sources_for_escalated",
                return_value=[source],
            ),
            patch("imas_codex.ids.workers.GraphClient") as mock_gc_cls,
            patch("imas_codex.ids.graph_ops.read_candidates", return_value=edges),
            patch("imas_codex.ids.graph_ops.select_candidates") as mock_select,
            patch("imas_codex.ids.mapping.achoose_targets", return_value=choice),
            patch("imas_codex.ids.workers.refresh_mapping_status") as mock_status,
            patch("imas_codex.ids.workers.record_mapping_verdict") as mock_verdict,
            patch("imas_codex.ids.workers.release_mapping_claim") as mock_release,
        ):
            mock_gc_cls.return_value.__enter__.return_value = gc
            asyncio.run(assign_worker(self._state()))
        return mock_status, mock_verdict, mock_select, mock_release

    def test_none_choice_persists_disposition_and_evidence(self):
        from imas_codex.ids.models import MappingDisposition

        mock_status, mock_verdict, mock_select, mock_release = self._run(
            MappingDisposition.NO_IMAS_EQUIVALENT
        )
        mock_verdict.assert_called_once_with(
            "jet:coil:1", "no_imas_equivalent", "no IMAS node carries this value"
        )
        mock_status.assert_not_called()
        mock_select.assert_not_called()
        mock_release.assert_not_called()


class TestDecidedSourceNotReclaimed:
    def test_claim_predicate_excludes_decided_sources(self):
        from imas_codex.ids.workers import claim_sources_for_escalated

        with patch("imas_codex.ids.workers.claim_batch", return_value=[]) as mock_claim:
            claim_sources_for_escalated("jet", ["magnetics"])
        predicate = mock_claim.call_args.kwargs["status_predicate"]
        assert "n.candidate_route = 'escalated'" in predicate
        assert "n.mapping_disposition IS NULL" in predicate


class TestResetClearsVerdict:
    def test_reset_nulls_disposition_and_evidence(self):
        from imas_codex.ids.workers import reset_mapping_state

        gc = MagicMock()
        gc.query.return_value = [{"cleared": 0}]
        with patch("imas_codex.ids.workers.GraphClient") as mock_cls:
            mock_cls.return_value.__enter__.return_value = gc
            reset_mapping_state("jet")
        query = gc.query.call_args[0][0]
        assert "sg.mapping_disposition = null" in query
        assert "sg.mapping_evidence = null" in query
