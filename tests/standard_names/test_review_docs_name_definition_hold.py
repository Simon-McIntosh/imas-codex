"""The documentation review batch persists a name-versus-definition hold.

The batch resolves each of its note builders through the worker dependency
lookup, so a builder with no declared entry stops the whole batch before any
item is reviewed. This drives the batch itself, not the pure note helper: an
item whose definition states a qualifier its name omits must come back held,
and the hold must carry the seat's own reason.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

from imas_codex.standard_names.graph_store import (
    graph_catalog as _owning_graph_catalog,
    graph_review_persistence as _owning_graph_review_persistence,
)
from imas_codex.standard_names.review import cascade, name_definition_seat
from imas_codex.standard_names.worker_tasks import (
    worker_common as _owning_worker_common,
    worker_review_docs as _owning_worker_review_docs,
)

QUALIFIER = "cross-section average"


def _item() -> dict:
    return {
        "id": "magnetic_field",
        "claim_token": "claim",
        "claim_seq": 1,
        "description": "Magnetic flux density.",
        "documentation": (
            "The magnetic flux density averaged over the plasma cross-section."
        ),
    }


def _request(item: dict) -> dict:
    return {
        "response_model": MagicMock(),
        "user_prompt": "review",
        "system_prompt": "review",
        "models": ["test/model"],
        "disagreement_threshold": 0.2,
        "rubric_dims": ["documentation_quality"],
        "reasoning_effort": False,
        "escalation_reasoning_effort": False,
        "veto_item": item,
    }


def _quorum() -> dict:
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


def test_the_batch_persists_the_holds_its_seat_raises(caplog) -> None:
    """A definition that adds a qualifier leaves the item held, not accepted."""
    item = _item()
    request = _request(item)
    persisted: dict = {}
    stages: list[str] = []

    async def prepare(_item):
        return request

    async def review(*_args, **_kwargs):
        return _quorum()

    def clean_conformance(*_args, **_kwargs):
        return None

    async def clean_note(*_args, **_kwargs):
        return None

    async def judge(items, **_kwargs):
        assert [entry["id"] for entry in items] == ["magnetic_field"]
        return (
            [
                {
                    "id": "magnetic_field",
                    "verdict": "flag",
                    "qualifier": QUALIFIER,
                    "reasoning": "the definition averages over the cross-section",
                }
            ],
            0.0,
        )

    def persist(**kwargs):
        persisted.update(kwargs)
        stage = (
            "accepted"
            if kwargs["score"] >= kwargs["min_score"] and not kwargs["review_holds"]
            else "reviewed"
        )
        stages.append(stage)
        return stage

    manager = MagicMock()
    manager.run_id = "run"
    with (
        patch.object(
            _owning_worker_review_docs, "prepare_docs_review_request", prepare
        ),
        patch.object(cascade, "adjudicate_review", review),
        patch.object(
            _owning_worker_review_docs,
            "_documentation_conformance_note",
            clean_conformance,
        ),
        patch.object(
            _owning_worker_review_docs,
            "_documentation_banned_prose_note",
            clean_note,
        ),
        patch.object(
            _owning_worker_review_docs,
            "_documentation_contradiction_note",
            clean_note,
        ),
        patch.object(name_definition_seat, "judge_name_definitions", judge),
        patch.object(
            _owning_worker_common, "_review_dd_source_bindings", return_value={}
        ),
        patch.object(_owning_worker_common, "_persist_dd_gap_evidence"),
        patch.object(_owning_graph_catalog, "update_review_aggregates"),
        patch.object(
            _owning_graph_review_persistence, "persist_reviewed_docs", persist
        ),
    ):
        processed = asyncio.run(
            _owning_worker_review_docs.process_review_docs_batch(
                [item], manager, asyncio.Event()
            )
        )

    errors = "\n".join(
        record.getMessage() for record in caplog.records if record.levelname == "ERROR"
    )
    assert errors == "", errors
    assert processed == 1
    assert stages == ["reviewed"]
    holds = persisted["review_holds"]
    assert [hold["kind"] for hold in holds] == ["name_definition"]
    assert QUALIFIER in holds[0]["finding"]
