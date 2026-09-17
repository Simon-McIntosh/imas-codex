"""A refine that resubmits the name unchanged must not spend a rotation.

The refine pool charges a rotation when it CLAIMS a name, before any model call,
so the charge buys an improvement attempt. The pinned-rename step neither
improves nor rewrites the name: ``edit_mode = 'rename'`` carries an
operator-chosen string that is a decision rather than a draft, so the pool
routes the same name back to a fresh review quorum and produces no candidate.

The rotation cap is also the eligibility gate, so a pinned name charged on every
re-review walks to 3 of 3 on resubmission alone and then stops being claimed —
its quantity silently drops from export. These tests pin the counter across both
outcomes: an unchanged resubmission leaves it where the claim found it, while a
refine that produces a different spelling keeps its charge, because that
rotation bought a rewrite.

Graph interaction is mocked (no live Neo4j) and the rewritten case also mocks
the model call and persistence (no live LLM).
"""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

GC_PATH = "imas_codex.graph.client.GraphClient"
LLM_PATH = "imas_codex.discovery.base.llm.acall_llm_structured"
PERSIST_PATH = "imas_codex.standard_names.graph_ops.persist_refined_name"
RESUBMIT_PATH = (
    "imas_codex.standard_names.graph_ops.resubmit_pinned_rename_for_review"
)
PROMPT_PATH = "imas_codex.llm.prompt_loader.render_prompt"
NEIGHBOURS_PATH = "imas_codex.standard_names.workers._hybrid_search_neighbours"
MODEL_PATH = "imas_codex.settings.get_model"


class RotationCounter:
    """A StandardName reduced to its rotation counter and its claim fence.

    Models the one write this change adds: returning a rotation an unchanged
    resubmission did not spend. A write this fake does not model fails the test
    rather than passing as an unnoticed no-op, so a green run proves the return
    was issued and not merely that nothing raised.
    """

    def __init__(self, *, charged: int, token: str = "tok-abc-123") -> None:
        self.charged = charged
        self.refine_attempts = charged
        self.token = token
        self.name_stage = "refining"
        self.rotation_returns = 0

    def __enter__(self) -> RotationCounter:
        return self

    def __exit__(self, *_a: Any) -> None:
        return None

    def query(self, cypher: str, **params: Any) -> list[dict[str, Any]]:
        text = " ".join(cypher.split())
        if "SET sn.refine_attempts" in text:
            if params.get("token") != self.token:
                return []
            if self.name_stage != "refining":
                return []
            self.rotation_returns += 1
            if self.refine_attempts > 0:
                self.refine_attempts -= 1
            return [{"refine_attempts": self.refine_attempts}]
        if any(word in text for word in ("SET ", "MERGE ", "CREATE ", "DELETE ")):
            raise AssertionError("unmodelled write reached the counter: " + text)
        return []


def claimed_item(*, charged: int, **overrides: Any) -> dict[str, Any]:
    """A claimed refine item, carrying the rotation the claim just charged."""
    item: dict[str, Any] = {
        "id": "test_name",
        "description": "A test quantity",
        "documentation": None,
        "kind": "scalar",
        "unit": "eV",
        "physics_domain": ["core_profiles"],
        "validation_status": "valid",
        "reviewer_score_name": 0.6,
        "reviewer_comments_per_dim_name": None,
        "chain_length": 0,
        "refine_attempts": charged,
        "name_stage": "refining",
        "source_paths": ["core_profiles/profiles_1d/electrons/temperature"],
        "claim_token": "tok-abc-123",
        "chain_history": [],
    }
    item.update(overrides)
    return item


def budget_manager() -> MagicMock:
    mgr = MagicMock()
    mgr.reserve = MagicMock(return_value=MagicMock())
    return mgr


@contextmanager
def pinned(graph: RotationCounter, outcome: str):
    """Patch the graph, the pinned resubmit, and the (unused) model call."""
    with (
        patch(GC_PATH, return_value=graph),
        patch(RESUBMIT_PATH, return_value=outcome) as resubmit,
        patch(LLM_PATH) as llm,
    ):
        yield resubmit, llm


@contextmanager
def rewritten(graph: RotationCounter, refined: Any):
    """Patch the graph, the model call, the persistence and the prompt."""
    with (
        patch(GC_PATH, return_value=graph),
        patch(LLM_PATH, return_value=(refined, 0.05, {"input_tokens": 100})),
        patch(PROMPT_PATH, return_value="prompt text"),
        patch(PERSIST_PATH, return_value={"new_name": "x", "old_name": "y"}) as persist,
        patch(NEIGHBOURS_PATH, return_value=[]),
        patch(MODEL_PATH, return_value="default-model"),
    ):
        yield persist


class TestUnchangedResubmissionReturnsTheRotation:
    """A pinned rename is resubmitted as-is: the rotation buys nothing."""

    @pytest.mark.asyncio
    async def test_counter_is_left_where_the_claim_found_it(self) -> None:
        from imas_codex.standard_names.workers import process_refine_name_batch

        graph = RotationCounter(charged=1)  # the claim charged 0 -> 1
        events: list[dict[str, Any]] = []
        item = claimed_item(charged=1, edit_mode="rename", name_hint="test_name")

        with pinned(graph, "resubmitted") as (resubmit, llm):
            await process_refine_name_batch(
                [item], budget_manager(), asyncio.Event(), on_event=events.append
            )

        resubmit.assert_called_once()
        llm.assert_not_called()
        assert graph.rotation_returns == 1
        assert graph.refine_attempts == graph.charged - 1 == 0

        assert [e["outcome"] for e in events] == ["pinned_rename_resubmitted"]
        assert events[0]["cost"] == 0.0
        assert events[0]["refine_attempts"] == 0

    @pytest.mark.asyncio
    async def test_a_claim_this_pool_no_longer_holds_is_not_rewritten(self) -> None:
        """The return is fenced on the claim it was charged under."""
        from imas_codex.standard_names.workers import process_refine_name_batch

        graph = RotationCounter(charged=1)
        graph.name_stage = "reviewed"  # a sweep already closed this claim
        item = claimed_item(charged=1, edit_mode="rename", name_hint="test_name")

        with pinned(graph, "") as (resubmit, llm):
            await process_refine_name_batch([item], budget_manager(), asyncio.Event())

        assert graph.rotation_returns == 0
        assert graph.refine_attempts == graph.charged == 1


class TestRewrittenRefineKeepsItsCharge:
    """A refine that produces a different spelling spent its rotation."""

    @pytest.mark.asyncio
    async def test_a_different_spelling_keeps_the_charged_rotation(self) -> None:
        from imas_codex.standard_names.models import RefinedName
        from imas_codex.standard_names.workers import process_refine_name_batch

        graph = RotationCounter(charged=1)
        item = claimed_item(charged=1)  # an ordinary rewrite, no pinned edit
        refined = RefinedName(
            base_token="temperature",
            base_kind="quantity",
            qualifiers=["electron"],
            description="Electron temperature at the plasma core",
            kind="scalar",
            reason="Better specificity",
        )

        with rewritten(graph, refined) as persist:
            await process_refine_name_batch([item], budget_manager(), asyncio.Event())

        assert persist.call_args.kwargs["new_name"] != item["id"]
        assert graph.rotation_returns == 0
        assert graph.refine_attempts == graph.charged == 1
