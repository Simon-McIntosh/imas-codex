"""Acceptance boundaries for exact-identity rescores with descendants."""

from __future__ import annotations

from typing import Any

from imas_codex.standard_names.graph_ops import (
    persist_reviewed_name,
    stage_name_for_rescore,
)
from tests.standard_names.test_edit_engine import FakeGraph, _patched_graph


class _RescoreGraph(FakeGraph):
    """Extend the edit graph with the exact rescore staging query."""

    def query(self, cypher: str, **params: Any) -> list[dict[str, Any]]:
        if "WHERE sn.name_stage IN ['exhausted', 'reviewed']" in cypher:
            node = self.nodes.get(params["id"])
            if node is None or node.get("name_stage") not in ("exhausted", "reviewed"):
                return []
            prior_stage = node["name_stage"]
            if not params["dry_run"]:
                node.update(
                    {
                        "name_stage": "drafted",
                        "reviewer_score_name": None,
                        "review_resubmit_count": 0,
                        "claim_token": None,
                        "claimed_at": None,
                        "run_id": params["run_id"] or node.get("run_id"),
                    }
                )
            return [{"prior_stage": prior_stage}]

        rows = super().query(cypher, **params)
        if "sn.run_id AS scope_run_id" in cypher and rows:
            rows[0]["scope_run_id"] = self.nodes[params["id"]].get("run_id")
        return rows


def _accept_reviewed_name(sn_id: str, *, ledger_run_id: str) -> str:
    return persist_reviewed_name(
        sn_id=sn_id,
        claim_token="",
        score=0.97,
        model="reviewer/x",
        min_score=0.75,
        rotation_cap=3,
        run_id=ledger_run_id,
        resolution_method="quorum_consensus",
        reviewer_chain_size=3,
    )


def test_rescore_accepts_parent_without_touching_accepted_child() -> None:
    graph = _RescoreGraph()
    graph.add_node(
        "temperature",
        name_stage="reviewed",
        edit_status="open",
        edit_scope="subtree",
        edit_include_accepted=False,
        edit_override_edits=False,
    )
    graph.add_node("ion_temperature", name_stage="accepted")
    graph.add_edge("ion_temperature", "temperature", "ion", "qualifier")
    run_id = "sn-rescore-20260902T151340Z"

    with _patched_graph(graph):
        staged = stage_name_for_rescore("temperature", run_id=run_id)
        stage = _accept_reviewed_name("temperature", ledger_run_id="run-ledger-uuid")

    assert staged == {
        "ok": True,
        "sn_id": "temperature",
        "prior_stage": "reviewed",
        "run_id": run_id,
        "dry_run": False,
    }
    assert stage == "accepted"
    assert graph.nodes["temperature"]["name_stage"] == "accepted"
    assert graph.nodes["ion_temperature"]["name_stage"] == "accepted"
    assert graph.edges_by_child["ion_temperature"][0]["parent_id"] == "temperature"
    assert set(graph.nodes) == {"temperature", "ion_temperature"}


def test_rename_acceptance_still_refuses_conflicting_descendant() -> None:
    graph = _RescoreGraph()
    graph.add_node(
        "temperature_of_plasma_boundary",
        name_stage="drafted",
        edit_status="open",
        edit_scope="subtree",
        edit_include_accepted=False,
        edit_override_edits=False,
        claim_token="tok",
        run_id="sn-edit-rename",
    )
    graph.add_node("ion_temperature", name_stage="accepted")
    graph.add_edge(
        "ion_temperature",
        "temperature_of_plasma_boundary",
        "ion",
        "qualifier",
    )

    with _patched_graph(graph):
        stage = persist_reviewed_name(
            sn_id="temperature_of_plasma_boundary",
            claim_token="tok",
            score=0.97,
            model="reviewer/x",
            min_score=0.75,
            rotation_cap=3,
            run_id="sn-edit-rename",
            resolution_method="quorum_consensus",
            reviewer_chain_size=3,
        )

    assert stage == "reviewed"
    assert graph.nodes["temperature_of_plasma_boundary"]["name_stage"] == "reviewed"
    assert graph.nodes["ion_temperature"]["name_stage"] == "accepted"
    assert "ion_temperature_of_plasma_boundary" not in graph.nodes
    assert any(
        "edit_cascade" in issue
        for issue in graph.nodes["temperature_of_plasma_boundary"]["validation_issues"]
    )
