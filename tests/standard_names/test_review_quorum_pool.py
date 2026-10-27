"""Tests for what the pool review workers write for one batch item.

A batch item reaches the cascade, which asks Jev to decide it; these tests pin
the pool side of that path: a quarantined name spends no LLM call, a batch
enforces its own retry/lease behaviour on a failed cycle, and an item that
reaches one seat still records one review. The reviewer chain's cycle 0/1/2
behaviour belongs to the audit panel, which runs it without persisting, and is
covered by ``test_rd_quorum_integrity.py``.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_budget_manager() -> MagicMock:
    mgr = MagicMock()
    lease = MagicMock()
    lease.charge_event = MagicMock(return_value=SimpleNamespace(overspend=0.0))
    lease.release_unused = MagicMock(return_value=0.0)
    mgr.reserve = MagicMock(return_value=lease)
    mgr.pool_admit = MagicMock(return_value=True)
    mgr.exhausted = MagicMock(return_value=False)
    mgr.run_id = "test-run"
    return mgr, lease


def _names_result(scores: dict[str, int], reasoning: str = "ok") -> Any:
    """Build a fake LLM result for the names rubric.

    ``scores`` is the int 0-20 dict returned by ``model_dump()``. The
    normalised ``score`` property is computed as the mean.
    """
    norm = sum(scores.values()) / (20.0 * len(scores)) if scores else 0.0
    return SimpleNamespace(
        scores=SimpleNamespace(score=norm, model_dump=lambda: scores),
        comments=None,
        reasoning=reasoning,
    )


def _llm_returns(values: list[Any]) -> AsyncMock:
    """AsyncMock that returns successive ``(result, cost, tokens)`` tuples."""
    side_effect = [(v, 0.01, 100) for v in values]
    return AsyncMock(side_effect=side_effect)


def _patch_common(
    *,
    captured: dict[str, Any],
    llm_mock: AsyncMock,
    models: list[str],
    review_axis: str,
):
    """Yield a single ``patch.multiple``-style context for review-pool tests."""

    persist_target = (
        "imas_codex.standard_names.graph_store.graph_review_persistence.persist_reviewed_name"
        if review_axis == "name"
        else "imas_codex.standard_names.graph_store.graph_review_persistence.persist_reviewed_docs"
    )
    release_target = (
        "imas_codex.standard_names.graph_store.graph_claim_release.release_review_names_failed_claims"
        if review_axis == "name"
        else "imas_codex.standard_names.graph_store.graph_claim_release.release_review_docs_failed_claims"
    )
    models_getter = (
        "imas_codex.settings.get_sn_review_names_models"
        if review_axis == "name"
        else "imas_codex.settings.get_sn_review_docs_models"
    )

    def _capture_persist(**kwargs):
        captured["persist"] = kwargs
        return "accepted"

    def _capture_write_reviews(records, **_):
        captured.setdefault("write_reviews_calls", []).append(list(records))
        return len(records)

    return [
        patch("imas_codex.discovery.base.llm.acall_llm_structured", new=llm_mock),
        patch(models_getter, return_value=models),
        # The docs path sizes its escalation prompt against the third reviewer's
        # published input allowance. These placeholder model names appear in no
        # catalog, so that allowance has to be supplied alongside them.
        patch(
            "imas_codex.discovery.base.llm.get_catalog_model_info",
            return_value={"max_input_tokens": 128_000},
        ),
        patch(
            "imas_codex.settings.get_sn_review_disagreement_threshold",
            return_value=0.20,
        ),
        patch(
            "imas_codex.llm.prompt_loader.render_prompt",
            return_value="mock prompt",
        ),
        patch(
            "imas_codex.standard_names.context.fetch_review_neighbours",
            return_value={
                "vector_neighbours": [],
                "same_base_neighbours": [],
                "same_path_neighbours": [],
            },
        ),
        patch(
            "imas_codex.standard_names.context._build_enum_lists",
            return_value={},
        ),
        patch(persist_target, side_effect=_capture_persist),
        patch(release_target),
        patch(
            "imas_codex.standard_names.graph_store.graph_catalog.write_reviews",
            side_effect=_capture_write_reviews,
        ),
        patch(
            "imas_codex.standard_names.graph_store.graph_catalog.update_review_aggregates",
        ),
    ]


# ---------------------------------------------------------------------------
# Names axis
# ---------------------------------------------------------------------------


def _names_item() -> dict[str, Any]:
    return {
        "id": "test_field",
        "description": "Test description.",
        "claim_token": "tok",
        "name_stage": "claimed_review_name",
    }


@pytest.mark.asyncio
async def test_names_batch_response_is_unwrapped() -> None:
    """LLM returning ``{"reviews": [item]}`` (Batch shape) must be unwrapped.

    Regression test for the bug where ``response_model`` was the singular
    ``StandardNameQualityReviewNameOnly`` but the prompt yields a batch —
    Pydantic validation failed with ``source_id Field required`` on every
    cycle-1 (gpt-5.4) call, leaving ZERO secondary reviews in the graph.
    """
    from imas_codex.standard_names.worker_tasks.worker_review_names import (
        process_review_name_batch,
    )

    captured: dict[str, Any] = {}
    mgr, _ = _mock_budget_manager()
    scores = {"grammar": 18, "semantic": 18, "convention": 18, "completeness": 18}
    inner = _names_result(scores)
    # Wrap as a batch: top-level object with a ``reviews`` list. The helper
    # must unwrap to ``reviews[0]`` before extracting ``.scores`` etc.
    batch = SimpleNamespace(reviews=[inner])
    llm = _llm_returns([batch])

    patches = _patch_common(
        captured=captured,
        llm_mock=llm,
        models=["only-model"],
        review_axis="name",
    )
    for p in patches:
        p.start()
    try:
        await process_review_name_batch(
            [_names_item()], mgr, asyncio.Event(), on_event=None
        )
    finally:
        for p in patches:
            p.stop()

    assert llm.await_count == 1
    records = captured["write_reviews_calls"][0]
    assert len(records) == 1
    # If the unwrap failed, .scores would raise AttributeError and the cycle
    # would log+skip, leaving zero records.
    assert records[0]["score"] > 0.0
    assert records[0]["cycle_index"] == 0


@pytest.mark.asyncio
async def test_names_empty_reviews_batch_is_a_failed_cycle() -> None:
    """LLM returning ``{"reviews": []}`` (empty batch) must be a FAILED cycle.

    Regression: an empty ``reviews`` list is falsy, so the wrapper was left
    in place and ``.scores`` raised AttributeError → swallowed to score=0.0
    → a spurious 0.0/'poor' canonical review persisted, driving a good name
    into refine/exhausted. The cycle must instead return None so the claim is
    released and the name stays 'drafted' (retryable), with NO review written.
    """
    from imas_codex.standard_names.worker_tasks.worker_review_names import (
        process_review_name_batch,
    )

    captured: dict[str, Any] = {}
    mgr, _ = _mock_budget_manager()
    empty_batch = SimpleNamespace(reviews=[])
    llm = _llm_returns([empty_batch])

    patches = _patch_common(
        captured=captured,
        llm_mock=llm,
        models=["only-model"],
        review_axis="name",
    )
    for p in patches:
        p.start()
    try:
        await process_review_name_batch(
            [_names_item()], mgr, asyncio.Event(), on_event=None
        )
    finally:
        for p in patches:
            p.stop()

    assert llm.await_count == 1
    # No review record and no stage transition — the empty response is a
    # failed cycle, not a 0.0 canonical review.
    assert "write_reviews_calls" not in captured
    assert "persist" not in captured


@pytest.mark.asyncio
async def test_names_single_model_chain_uses_single_review() -> None:
    """1-model chain → 1 cycle, ``single_review`` resolution method."""
    from imas_codex.standard_names.worker_tasks.worker_review_names import (
        process_review_name_batch,
    )

    captured: dict[str, Any] = {}
    mgr, _ = _mock_budget_manager()
    scores = {"grammar": 17, "semantic": 17, "convention": 17, "completeness": 17}
    llm = _llm_returns([_names_result(scores)])

    patches = _patch_common(
        captured=captured,
        llm_mock=llm,
        models=["only-model"],
        review_axis="name",
    )
    for p in patches:
        p.start()
    try:
        await process_review_name_batch(
            [_names_item()], mgr, asyncio.Event(), on_event=None
        )
    finally:
        for p in patches:
            p.stop()

    assert llm.await_count == 1
    records = captured["write_reviews_calls"][0]
    assert len(records) == 1
    assert records[0]["cycle_index"] == 0
    assert records[0]["resolution_role"] == "primary"
    assert records[0]["resolution_method"] == "single_review"


@pytest.mark.asyncio
async def test_names_quarantined_item_skips_llm() -> None:
    """Quarantined name → no LLM calls, persist via fast-path with score=0."""
    from imas_codex.standard_names.worker_tasks.worker_review_names import (
        process_review_name_batch,
    )

    captured: dict[str, Any] = {}
    mgr, _ = _mock_budget_manager()
    llm = AsyncMock()  # should never be awaited

    item = _names_item()
    item["validation_status"] = "quarantined"

    patches = _patch_common(
        captured=captured,
        llm_mock=llm,
        models=["model-a", "model-b", "model-c"],
        review_axis="name",
    )
    for p in patches:
        p.start()
    try:
        await process_review_name_batch([item], mgr, asyncio.Event(), on_event=None)
    finally:
        for p in patches:
            p.stop()

    assert llm.await_count == 0
    # No write_reviews invocation in the quarantine fast-path.
    assert "write_reviews_calls" not in captured
    # Persist called with score=0 and the quarantine model marker.
    assert captured["persist"]["score"] == 0.0
    assert "quarantined" in captured["persist"]["model"]


# ---------------------------------------------------------------------------
# Docs axis (mirrors names)
# ---------------------------------------------------------------------------
