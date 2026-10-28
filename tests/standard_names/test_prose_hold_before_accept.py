"""Documentation with banned prose is held before review can accept it."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

from imas_codex.standard_names import prose_adjudicator, workers
from imas_codex.standard_names.campaign import measure_batch
from imas_codex.standard_names.graph_store import (
    graph_catalog as _owning_graph_catalog,
    graph_review_persistence as _owning_graph_review_persistence,
)
from imas_codex.standard_names.prose_policy import banned_prose_findings
from imas_codex.standard_names.worker_tasks import (
    worker_common as _owning_worker_common,
    worker_review_docs as _owning_worker_review_docs,
    worker_review_names as _owning_worker_review_names,
)


def _review(
    documentation: str, *, genuine_banned: bool = True
) -> tuple[str, dict, list[dict]]:
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
    adjudications = []

    async def prepare(_item):
        return request

    async def review(**_kwargs):
        return quorum

    def persist(**kwargs):
        """Mirror the persistence contract: a passing score plus no open hold."""
        persisted.update(kwargs)
        passing = kwargs["score"] >= kwargs["min_score"]
        held = bool(kwargs["review_holds"])
        stage = "accepted" if passing and not held else "reviewed"
        persisted_stage.append(stage)
        return stage

    async def adjudicate(model, name, text, findings, **kwargs):
        adjudications.append(
            {
                "model": model,
                "name": name,
                "text": text,
                "findings": findings,
                "budget_manager": kwargs["budget_manager"],
            }
        )
        return genuine_banned

    manager = MagicMock()
    manager.run_id = "run"
    with (
        patch.object(
            _owning_worker_review_docs, "prepare_docs_review_request", prepare
        ),
        patch.object(_owning_worker_review_names, "_run_rd_quorum_cycles", review),
        patch.object(
            _owning_worker_common, "_review_dd_source_bindings", return_value={}
        ),
        patch.object(_owning_worker_common, "_persist_dd_gap_evidence"),
        patch.object(_owning_graph_catalog, "write_reviews"),
        patch.object(_owning_graph_catalog, "update_review_aggregates"),
        patch.object(
            _owning_graph_review_persistence, "persist_reviewed_docs", persist
        ),
        patch.object(prose_adjudicator, "_adjudicate_one", adjudicate),
        patch("imas_codex.settings.get_model", return_value="test/model"),
        patch("imas_codex.settings.get_reasoning_effort", return_value="low"),
    ):
        processed = asyncio.run(
            _owning_worker_review_docs.process_review_docs_batch(
                [item], manager, asyncio.Event()
            )
        )
    assert processed == 1
    return persisted_stage[0], persisted, adjudications


def test_estimator_recipe_is_held_and_sent_to_refinement():
    documentation = (
        "The magnetic field describes magnetic flux density. "
        "It is obtained from a weighted mean of local measurements."
    )
    assert banned_prose_findings(documentation)["estimator_recipe"] == 1

    stage, persisted, adjudications = _review(documentation)

    assert stage == "reviewed"
    assert [hold["kind"] for hold in persisted["review_holds"]] == ["banned_prose"]
    assert "is obtained from" in persisted["comments"].lower()
    assert "is obtained from" in persisted["comments_per_dim"]["banned_prose"].lower()
    assert adjudications[0]["findings"]["estimator_recipe"] == 1
    assert adjudications[0]["budget_manager"].run_id == "run"


def test_definitional_relation_is_accepted_after_adjudication():
    documentation = (
        "The associated [ion particle flux](name:ion_particle_flux) is obtained "
        "by omitting the kinetic-energy factor from the integrand."
    )
    assert banned_prose_findings(documentation)["estimator_recipe"] == 1

    stage, persisted, adjudications = _review(documentation, genuine_banned=False)

    assert stage == "accepted"
    assert persisted["score"] == 1.0
    assert "banned_prose" not in persisted["comments_per_dim"]
    assert adjudications[0]["name"] == "magnetic_field"


def test_clean_documentation_is_accepted():
    documentation = "The magnetic field describes magnetic flux density."
    assert all(count == 0 for count in banned_prose_findings(documentation).values())

    stage, persisted, adjudications = _review(documentation)

    assert stage == "accepted"
    assert persisted["score"] == 1.0
    assert adjudications == []


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
