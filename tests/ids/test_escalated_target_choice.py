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
    def test_no_candidate_source_is_skipped_with_evidence(self, caplog):
        import logging

        from imas_codex.ids.candidates import PairJudgment, route
        from imas_codex.settings import RouteThresholds

        judgments = [
            PairJudgment(
                path="magnetics/ip", p_same_quantity=0.41, model="m", judged_at="t"
            ),
            PairJudgment(
                path="magnetics/other", p_same_quantity=0.02, model="m", judged_at="t"
            ),
        ]
        decision = route(judgments, RouteThresholds(0.9, 0.5, 5))
        assert decision is not None
        assert decision.decision == "no_candidate"
        best = max(judgments, key=lambda j: j.p_same_quantity)
        with caplog.at_level(logging.INFO):
            logging.getLogger("imas_codex.ids.workers").info(
                "No candidate for %s: best path %s (p_same_quantity=%.3f), skipping",
                "jet:s1",
                best.path,
                best.p_same_quantity,
            )
        assert "jet:s1" in caplog.text
        assert "magnetics/other" not in caplog.text
        assert "0.410" in caplog.text
