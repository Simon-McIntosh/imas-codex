"""Documentation with banned prose is held before review can accept it."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

from imas_codex.standard_names import workers
from imas_codex.standard_names.campaign import measure_batch
from imas_codex.standard_names.prose_policy import banned_prose_findings


def _review(documentation: str) -> tuple[str, dict]:
    import imas_codex.standard_names.graph_ops as graph_ops

    item = {
        "id": "magnetic_field",
        "claim_token": "claim",
        "claim_seq": 1,
        "description": "Magnetic field strength.",
        "documentation": documentation,
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
    }
    quorum = {
        "records": [],
        "winning_score": 1.0,
        "winning_scores": {"documentation_quality": 20},
        "winning_comments": "Clear.",
        "winning_comments_per_dim": {"documentation_quality": "Clear."},
        "canonical_model": "test/model",
        "total_cost": 0.0,
        "total_tokens_in": 0,
        "total_tokens_out": 0,
        "resolution_method": "quorum_consensus",
        "dd_gaps": [],
    }
    persisted = {}
    persisted_stage = []

    async def prepare(_item):
        return request

    async def review(**_kwargs):
        return quorum

    def persist(**kwargs):
        persisted.update(kwargs)
        stage = "accepted" if kwargs["score"] >= kwargs["min_score"] else "reviewed"
        persisted_stage.append(stage)
        return stage

    manager = MagicMock()
    manager.run_id = "run"
    with (
        patch.object(workers, "prepare_docs_review_request", prepare),
        patch.object(workers, "_run_rd_quorum_cycles", review),
        patch.object(workers, "_review_dd_source_bindings", return_value={}),
        patch.object(workers, "_persist_dd_gap_evidence"),
        patch.object(graph_ops, "write_reviews"),
        patch.object(graph_ops, "update_review_aggregates"),
        patch.object(graph_ops, "persist_reviewed_docs", persist),
    ):
        processed = asyncio.run(
            workers.process_review_docs_batch([item], manager, asyncio.Event())
        )
    assert processed == 1
    return persisted_stage[0], persisted


def test_estimator_recipe_is_held_and_sent_to_refinement():
    documentation = (
        "The magnetic field describes magnetic flux density. "
        "It is obtained from a weighted mean of local measurements."
    )
    assert banned_prose_findings(documentation)["estimator_recipe"] == 1

    stage, persisted = _review(documentation)

    assert stage == "reviewed"
    assert "is obtained from" in persisted["comments"].lower()
    assert "is obtained from" in persisted["comments_per_dim"]["banned_prose"].lower()


def test_clean_documentation_is_accepted():
    documentation = "The magnetic field describes magnetic flux density."
    assert all(count == 0 for count in banned_prose_findings(documentation).values())

    stage, persisted = _review(documentation)

    assert stage == "accepted"
    assert persisted["score"] == 1.0


def test_batch_reports_clean_acceptance_and_prose_quarantine_apart():
    rows = [
        {
            "id": "clean",
            "name_stage": "accepted",
            "docs_stage": "accepted",
            "description": "Magnetic field strength.",
            "documentation": "A clean definition.",
        },
        {
            "id": "recipe",
            "name_stage": "accepted",
            "docs_stage": "accepted",
            "description": "Magnetic field strength.",
            "documentation": "It is obtained from measured field values.",
        },
    ]

    outcome = measure_batch(rows, ["clean", "recipe"], batch_index=0)

    assert outcome.accepted == 2
    assert outcome.accepted_clean == 1
    assert outcome.quarantined == 1
    assert outcome.reintroduced_ids == ["recipe"]
