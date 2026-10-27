"""Documentation acceptance obeys the source contradiction decision."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from imas_codex.standard_names import workers
from imas_codex.standard_names.graph_store import (
    graph_catalog as _owning_graph_catalog,
    graph_review_persistence as _owning_graph_review_persistence,
)
from imas_codex.standard_names.review import cascade, holds, jev_review
from imas_codex.standard_names.worker_tasks import (
    worker_common as _owning_worker_common,
    worker_review_docs as _owning_worker_review_docs,
    worker_review_names as _owning_worker_review_names,
)


class _HoldRecorder:
    """Record the worker's hold writes without reaching a live graph."""

    def __init__(self) -> None:
        self.rows: dict[str, dict] = {}

    def raise_hold(self, name_id, axis, kind, finding, evidence, raised_by):
        key = holds.finding_key(kind, finding)
        row = {
            "id": holds.hold_id(name_id, axis, kind, finding),
            "axis": axis,
            "kind": kind,
            "finding": finding,
            "evidence": evidence,
            "raised_by": raised_by,
            "finding_key": key,
            "status": "open",
        }
        self.rows[key] = row
        return row

    def clear_holds_not_raised(self, name_id, axis, checked_kinds, raised_keys):
        cleared = 0
        for row in self.rows.values():
            if (
                row["id"].startswith(f"{name_id}:{axis}:")
                and row["kind"] in checked_kinds
                and row["finding_key"] not in raised_keys
                and row["status"] == "open"
            ):
                row["status"] = "cleared"
                cleared += 1
        return cleared

    def open_holds(self):
        return [row for row in self.rows.values() if row["status"] == "open"]

    def apply_review(self, name_id, review_holds, checked_kinds):
        raised_keys = {
            self.raise_hold(name_id, "docs", **hold)["finding_key"]
            for hold in review_holds
        }
        self.clear_holds_not_raised(name_id, "docs", checked_kinds, raised_keys)


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
    persisted_stages: list[str] = []
    hold_records = _HoldRecorder()

    async def prepare(_item):
        return request

    async def review(*_args, **_kwargs):
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
        hold_records.apply_review(
            kwargs["sn_id"], kwargs["review_holds"], kwargs["checked_hold_kinds"]
        )
        stage = (
            "accepted"
            if kwargs["score"] >= kwargs["min_score"] and not hold_records.open_holds()
            else "reviewed"
        )
        persisted_stages.append(stage)
        return stage

    import imas_codex.standard_names.graph_ops as graph_ops

    manager = MagicMock()
    manager.run_id = "run"
    with (
        patch.object(
            _owning_worker_review_docs, "prepare_docs_review_request", prepare
        ),
        patch.object(_owning_worker_review_names, "_run_rd_quorum_cycles", review),
        patch.object(cascade, "adjudicate_review", review),
        patch.object(holds, "raise_hold", hold_records.raise_hold),
        patch.object(
            holds, "clear_holds_not_raised", hold_records.clear_holds_not_raised
        ),
        patch.object(
            _owning_worker_common, "_review_dd_source_bindings", return_value={}
        ),
        patch.object(_owning_worker_common, "_persist_dd_gap_evidence"),
        patch.object(jev_review, "decide_batch", decide),
        patch.object(_owning_graph_catalog, "write_reviews"),
        patch.object(_owning_graph_catalog, "update_review_aggregates"),
        patch.object(
            _owning_graph_review_persistence, "persist_reviewed_docs", persist
        ),
    ):
        assert (
            asyncio.run(
                _owning_worker_review_docs.process_review_docs_batch(
                    [item], manager, asyncio.Event()
                )
            )
            == 1
        )

    assert persisted_stages == [expected_stage]
    assert persisted["score"] == 1.0
    assert bool(hold_records.open_holds()) == (expected_stage == "reviewed")
    if expected_stage == "reviewed":
        hold = hold_records.open_holds()[0]
        assert hold["kind"] == "documentation_contradiction"
        assert "The field points inward." in hold["evidence"]
        assert "The field points outward." in hold["evidence"]
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
    persisted_stages: list[str] = []
    hold_records = _HoldRecorder()
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

    async def review(*_args, **_kwargs):
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
        hold_records.apply_review(
            kwargs["sn_id"], kwargs["review_holds"], kwargs["checked_hold_kinds"]
        )
        stage = (
            "accepted"
            if kwargs["score"] >= kwargs["min_score"] and not hold_records.open_holds()
            else "reviewed"
        )
        persisted_stages.append(stage)
        return stage

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
        patch.object(
            _owning_worker_review_docs, "_load_docs_review_examples", return_value=[]
        ),
        patch.object(_owning_worker_review_names, "_run_rd_quorum_cycles", review),
        patch.object(cascade, "adjudicate_review", review),
        patch.object(holds, "raise_hold", hold_records.raise_hold),
        patch.object(
            holds, "clear_holds_not_raised", hold_records.clear_holds_not_raised
        ),
        patch.object(
            _owning_worker_common, "_review_dd_source_bindings", return_value={}
        ),
        patch.object(_owning_worker_common, "_persist_dd_gap_evidence"),
        patch(
            "imas_codex.standard_names.worker_tasks.worker_extraction_and_enrichment._enrich_batch_items"
        ),
        patch(
            "imas_codex.standard_names.worker_tasks.worker_extraction_and_enrichment._hybrid_search_neighbours_batch",
            return_value=[],
        ),
        patch(
            "imas_codex.standard_names.worker_tasks.worker_extraction_and_enrichment._related_path_neighbours",
            return_value=[],
        ),
        patch.object(jev_review, "decide_batch", decide),
        patch.object(_owning_graph_catalog, "write_reviews"),
        patch.object(_owning_graph_catalog, "update_review_aggregates"),
        patch.object(
            _owning_graph_review_persistence, "persist_reviewed_docs", persist
        ),
    ):
        assert (
            asyncio.run(
                _owning_worker_review_docs.process_review_docs_batch(
                    [item], manager, asyncio.Event()
                )
            )
            == 1
        )

    assert [source["id"] for source in observed] == [first, second]
    assert persisted_stages == ["reviewed"]
    assert persisted["score"] == 1.0
    assert len(hold_records.open_holds()) == 1
    assert hold_records.open_holds()[0]["kind"] == "documentation_contradiction"
    assert second in hold_records.open_holds()[0]["evidence"]
    assert source_docs[second] in hold_records.open_holds()[0]["evidence"]
    assert second in persisted["comments_per_dim"]["contradicts_source"]
    assert source_docs[second] in persisted["comments_per_dim"]["contradicts_source"]


def test_stale_docs_claim_cannot_change_winning_holds() -> None:
    """A no-op persistence must leave the graph owner's hold state untouched."""
    import imas_codex.standard_names.graph_ops as graph_ops

    item = {
        "id": "magnetic_field",
        "claim_token": "stale",
        "claim_seq": 1,
        "description": "Magnetic flux density.",
        "documentation": "The field points inward.",
    }
    request = {
        "models": ["test/model"],
        "veto_item": item,
        "response_model": MagicMock(),
        "user_prompt": "review",
        "system_prompt": "review",
        "disagreement_threshold": 0.2,
        "rubric_dims": ["documentation_quality"],
        "reasoning_effort": None,
        "escalation_reasoning_effort": None,
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
    hold_records = _HoldRecorder()

    async def prepare(_item):
        return request

    async def review(*_args, **_kwargs):
        return quorum

    async def prose_note(*_args):
        return None

    async def contradiction_note(*_args):
        return "claim contradicts source"

    manager = MagicMock()
    manager.run_id = "run"
    persist = MagicMock(return_value="")
    with (
        patch.object(
            _owning_worker_review_docs, "prepare_docs_review_request", prepare
        ),
        patch.object(cascade, "adjudicate_review", review),
        patch.object(
            _owning_worker_review_docs,
            "_documentation_conformance_note",
            return_value=None,
        ),
        patch.object(
            _owning_worker_review_docs, "_documentation_banned_prose_note", prose_note
        ),
        patch.object(
            _owning_worker_review_docs,
            "_documentation_contradiction_note",
            contradiction_note,
        ),
        patch.object(holds, "raise_hold", hold_records.raise_hold),
        patch.object(
            holds, "clear_holds_not_raised", hold_records.clear_holds_not_raised
        ),
        patch.object(
            _owning_graph_review_persistence, "persist_reviewed_docs", persist
        ),
        patch.object(_owning_graph_catalog, "update_review_aggregates"),
    ):
        processed = asyncio.run(
            _owning_worker_review_docs.process_review_docs_batch(
                [item], manager, asyncio.Event()
            )
        )

    assert processed == 0
    persist.assert_called_once()
    assert persist.call_args.kwargs["review_holds"][0]["finding"] == (
        "claim contradicts source"
    )
    assert hold_records.open_holds() == []
