"""Documentation acceptance obeys the source contradiction decision."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from imas_codex.standard_names import workers
from imas_codex.standard_names.review import jev_review


@pytest.mark.parametrize(
    ("answer", "expected_stage"),
    [
        ("contradicts", "reviewed"),
        ("supports", "accepted"),
        (None, "reviewed"),
        ("missing", "reviewed"),
        ("error", "reviewed"),
    ],
)
def test_claim_verdict_controls_acceptance(answer, expected_stage):
    item = {
        "id": "magnetic_field",
        "claim_token": "claim",
        "claim_seq": 1,
        "description": "Magnetic flux density.",
        "documentation": "The field points inward.",
        "dd_source_docs": [
            {"path": "dd:magnetic_field", "documentation": "The field points outward."}
        ],
    }
    request = {
        "response_model": MagicMock(),
        "user_prompt": "review",
        "system_prompt": "review",
        "models": ["test/model"],
        "disagreement_threshold": 0.2,
        "rubric_dims": ["documentation_quality"],
        "reasoning_effort": None,
        "escalation_reasoning_effort": None,
        "veto_item": item,
    }
    quorum = {
        "records": [],
        "winning_score": 1.0,
        "winning_scores": {"documentation_quality": 20},
        "winning_comments": "Clear.",
        "winning_comments_per_dim": {},
        "canonical_model": "test/model",
        "total_cost": 0.0,
        "total_tokens_in": 0,
        "total_tokens_out": 0,
        "resolution_method": "quorum_consensus",
        "dd_gaps": [],
    }
    persisted = {}

    async def prepare(_item):
        return request

    async def review(**_kwargs):
        return quorum

    async def decide(states, questions, **kwargs):
        assert states[0]["documentation_claims"] == ["The field points inward."]
        assert questions["check:claim_contradicts_source:0"]["type"] == "choice"
        if answer == "error":
            raise RuntimeError("Jev unavailable")
        if answer is None:
            return [None], 0.0
        answers = (
            {}
            if answer == "missing"
            else {"check:claim_contradicts_source:0": {"choice": answer}}
        )
        return [(answers, 0.0001)], 0.0001

    def persist(**kwargs):
        persisted.update(kwargs)
        return "accepted" if kwargs["score"] >= kwargs["min_score"] else "reviewed"

    import imas_codex.standard_names.graph_ops as graph_ops

    manager = MagicMock()
    manager.run_id = "run"
    with (
        patch.object(workers, "prepare_docs_review_request", prepare),
        patch.object(workers, "_run_rd_quorum_cycles", review),
        patch.object(workers, "_review_dd_source_bindings", return_value={}),
        patch.object(workers, "_persist_dd_gap_evidence"),
        patch.object(jev_review, "decide_batch", decide),
        patch.object(graph_ops, "write_reviews"),
        patch.object(graph_ops, "update_review_aggregates"),
        patch.object(graph_ops, "persist_reviewed_docs", persist),
    ):
        assert (
            asyncio.run(
                workers.process_review_docs_batch([item], manager, asyncio.Event())
            )
            == 1
        )

    stage = "accepted" if persisted["score"] >= persisted["min_score"] else "reviewed"
    assert stage == expected_stage
    if answer == "contradicts":
        assert (
            "The field points inward."
            in persisted["comments_per_dim"]["contradicts_source"]
        )
        assert (
            "The field points outward."
            in persisted["comments_per_dim"]["contradicts_source"]
        )
        assert "The field points inward." in persisted["comments"]
    elif answer == "supports":
        assert persisted["score"] == 1.0
        assert "contradicts_source" not in persisted["comments_per_dim"]
    manager.reserve.assert_called_once_with(0.001, phase="review_docs")
    if answer not in (None, "error"):
        assert manager.reserve.return_value.charge_event.call_args.args[0] == 0.0001


def test_second_bound_dd_source_can_hold_documentation():
    """Preparation must carry every bound source into the claim decision."""
    import imas_codex.standard_names.graph_ops as graph_ops

    first = "equilibrium/time_slice/profiles_1d/psi"
    second = "equilibrium/time_slice/profiles_1d/phi"
    item = {
        "id": "magnetic_field",
        "claim_token": "claim",
        "claim_seq": 1,
        "description": "Magnetic flux density.",
        "documentation": "The field points inward.",
        "source_paths": [first, second],
        "source_bindings": [
            {"source_type": "dd", "dd_path": first},
            {"source_type": "dd", "dd_path": second},
        ],
    }
    source_docs = {
        first: "The field points inward.",
        second: "The field points outward.",
    }
    observed = []
    persisted = {}
    graph = MagicMock()

    def query(cypher, **kwargs):
        if "UNWIND $paths AS pid" in cypher and "n.documentation" in cypher:
            assert set(kwargs["paths"]) == set(source_docs)
            return [
                {"id": path, "documentation": text, "description": text, "unit": "T"}
                for path, text in source_docs.items()
            ]
        if "RETURN n.keywords AS keywords" in cypher:
            assert kwargs["path"] == first
            return [{"dd_documentation": source_docs[first]}]
        return []

    graph.query.side_effect = query

    async def decide(states, questions, **kwargs):
        assert questions["check:claim_contradicts_source:0"]["type"] == "choice"
        results = []
        for state in states:
            docs = state["dd_source_docs"]["value"]
            assert len(docs) == 1
            observed.extend(docs)
            choice = (
                "contradicts"
                if docs[0].get("documentation") == source_docs[second]
                else "supports"
            )
            results.append(
                ({"check:claim_contradicts_source:0": {"choice": choice}}, 0.0001)
            )
        return results, 0.0001 * len(results)

    async def review(**_kwargs):
        return {
            "records": [],
            "winning_score": 1.0,
            "winning_scores": {"documentation_quality": 20},
            "winning_comments": "Clear.",
            "winning_comments_per_dim": {},
            "canonical_model": "test/model",
            "total_cost": 0.0,
            "total_tokens_in": 0,
            "total_tokens_out": 0,
            "resolution_method": "quorum_consensus",
            "dd_gaps": [],
        }

    def persist(**kwargs):
        persisted.update(kwargs)
        return "accepted" if kwargs["score"] >= kwargs["min_score"] else "reviewed"

    graph_context = MagicMock()
    graph_context.__enter__.return_value = graph
    manager = MagicMock()
    manager.run_id = "run"
    with (
        patch("imas_codex.graph.client.GraphClient", return_value=graph_context),
        patch(
            "imas_codex.standard_names.context.fetch_review_neighbours", return_value={}
        ),
        patch(
            "imas_codex.standard_names.context.conventions_context_for", return_value={}
        ),
        patch("imas_codex.llm.prompt_loader.render_prompt", return_value="review"),
        patch(
            "imas_codex.settings.get_sn_review_docs_models", return_value=["test/model"]
        ),
        patch.object(workers, "model_provider_exposure", return_value=0.001),
        patch.object(workers, "_load_docs_review_examples", return_value=[]),
        patch.object(workers, "_run_rd_quorum_cycles", review),
        patch.object(workers, "_review_dd_source_bindings", return_value={}),
        patch.object(workers, "_persist_dd_gap_evidence"),
        patch("imas_codex.standard_names.workers._enrich_batch_items"),
        patch(
            "imas_codex.standard_names.workers._hybrid_search_neighbours_batch",
            return_value=[],
        ),
        patch(
            "imas_codex.standard_names.workers._related_path_neighbours",
            return_value=[],
        ),
        patch.object(jev_review, "decide_batch", decide),
        patch.object(graph_ops, "write_reviews"),
        patch.object(graph_ops, "update_review_aggregates"),
        patch.object(graph_ops, "persist_reviewed_docs", persist),
    ):
        assert (
            asyncio.run(
                workers.process_review_docs_batch([item], manager, asyncio.Event())
            )
            == 1
        )

    assert [source["id"] for source in observed] == [first, second]
    assert persisted["score"] < persisted["min_score"]
    assert second in persisted["comments_per_dim"]["contradicts_source"]
    assert source_docs[second] in persisted["comments_per_dim"]["contradicts_source"]
