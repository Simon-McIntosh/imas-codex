"""Bounded failed-call recovery, literal source fallback and review coalescing."""

import asyncio
import json
from unittest.mock import AsyncMock

import pytest

from imas_codex.discovery.signals.enrichment import (
    DescriptionReview,
    EnrichmentOptions,
    enrich_signal_batch,
)
from imas_codex.discovery.signals.enrichment_review import ReviewBatcher
from tests.discovery.test_signal_enrichment_grounding import draft, messages


async def enrich(options, call):
    return await enrich_signal_batch(
        model="local/test", messages=messages(), options=options, call=call
    )


def review(index=1, statements=None):
    return DescriptionReview(
        results=[{"signal_index": index, "statements": statements or []}]
    )


@pytest.mark.asyncio
async def test_all_unsupported_statements_receive_marked_source_fallback():
    call = AsyncMock(side_effect=[(draft(), 0, 1), (review(), 0, 1)])
    batch, _, _ = await enrich(
        EnrichmentOptions(review=True, source_fallback=True), call
    )
    assert batch.results[0].description == "Cooling temperature"
    assert batch.fallback_indices == [1]
    assert batch.fallback_context_items == {1: ["context-1"]}
    assert batch.results[0].context_quality.value == "low"
    assert "Unsupported circuit" not in batch.results[0].description


@pytest.mark.asyncio
@pytest.mark.parametrize("failed_stage", ["draft", "review"])
async def test_bad_coverage_retries_only_failed_call(failed_stage):
    bad = draft() if failed_stage == "draft" else review(3)
    if failed_stage == "draft":
        bad.results[0].signal_index = 3
        replies = [(bad, 0.1, 10), (draft(), 0.2, 20), (review(), 0.3, 30)]
    else:
        replies = [(draft(), 0.1, 10), (bad, 0.2, 20), (review(), 0.3, 30)]
    call = AsyncMock(side_effect=replies)
    batch, cost, tokens = await enrich(
        EnrichmentOptions(review=True, recovery_attempts=1, source_fallback=True), call
    )
    assert batch.results[0].description
    assert getattr(batch, failed_stage + "_retries") == 1
    assert cost == pytest.approx(0.6)
    assert tokens == 60
    assert call.await_count == 3
    assert call.call_args_list[-1].kwargs["response_model"] is DescriptionReview


@pytest.mark.asyncio
async def test_failed_call_retry_is_bounded_and_no_unchecked_draft_escapes():
    call = AsyncMock(
        side_effect=[
            (draft(), 0, 1),
            ValueError("bad review"),
            ValueError("bad review"),
        ]
    )
    with pytest.raises(ValueError, match="bad review"):
        await enrich(
            EnrichmentOptions(review=True, recovery_attempts=1, source_fallback=True),
            call,
        )
    assert call.await_count == 3


@pytest.mark.asyncio
async def test_literal_source_copy_skips_review_but_related_guess_does_not():
    value = draft()
    value.results[0].description = "Cooling temperature."
    call = AsyncMock(return_value=(value, 0, 1))
    batch, _, _ = await enrich(
        EnrichmentOptions(review=True, review_if_unsupported=True), call
    )
    assert batch.review_skips == 1
    assert call.await_count == 1
    call = AsyncMock(side_effect=[(draft(), 0, 1), (review(), 0, 1)])
    batch, _, _ = await enrich(
        EnrichmentOptions(
            review=True, review_if_unsupported=True, source_fallback=True
        ),
        call,
    )
    assert call.await_count == 2
    assert batch.review_skips == 0
    assert batch.fallback_indices == [1]


@pytest.mark.asyncio
async def test_fallback_cannot_use_another_signals_source_label():
    text = messages()
    text[1]["content"] = (
        "### Signal 1\naccessor: EMPTY\n### Signal 2\nsource_description: Unrelated current"
    )
    value = draft()
    value.results[0].description = ""
    with pytest.raises(ValueError, match="no authoritative"):
        await enrich_signal_batch(
            model="local/test",
            messages=text,
            options=EnrichmentOptions(source_fallback=True),
            call=AsyncMock(return_value=(value, 0, 1)),
        )


@pytest.mark.asyncio
async def test_two_reviews_share_one_call_with_isolated_context_and_accounting():
    batcher = ReviewBatcher(2)

    async def call(**kwargs):
        data = json.loads(kwargs["messages"][1]["content"])
        assert len(data["drafts"]) == 2
        assert data["allowed_context_items"]["1"] == ["request-0-context-1"]
        return (
            DescriptionReview(
                results=[
                    {
                        "signal_index": i,
                        "statements": [
                            {
                                "draft_index": 1,
                                "text": "Temperature.",
                                "context_items": [f"request-{i - 1}-context-1"],
                            }
                        ],
                    }
                    for i in (1, 2)
                ]
            ),
            0.2,
            20,
        )

    call = AsyncMock(side_effect=call)
    kwargs = {
        "messages": [
            {"role": "system", "content": "Review"},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "context_items": {"context-1": "Temperature"},
                        "drafts": {1: ["Temperature."]},
                    }
                ),
            },
        ]
    }
    outputs = await asyncio.gather(
        batcher.submit(call, kwargs), batcher.submit(call, kwargs)
    )
    assert call.await_count == 1
    for batch, cost, tokens in outputs:
        assert batch.results[0].signal_index == 1
        assert batch.results[0].statements[0].context_items == ["context-1"]
        assert cost == 0.1
        assert tokens == 10


@pytest.mark.asyncio
async def test_review_tail_flushes_and_duplicate_coverage_refuses_all_waiters():
    batcher = ReviewBatcher(2)
    call = AsyncMock(return_value=(review(), 0, 1))
    kwargs = {
        "messages": [
            {"role": "system", "content": "Review"},
            {
                "role": "user",
                "content": json.dumps(
                    {"context_items": {}, "drafts": {1: ["Temperature."]}}
                ),
            },
        ]
    }
    await asyncio.wait_for(batcher.submit(call, kwargs), 1)
    outputs = await asyncio.gather(
        batcher.submit(call, kwargs),
        batcher.submit(call, kwargs),
        return_exceptions=True,
    )
    assert all(isinstance(value, ValueError) for value in outputs)


@pytest.mark.parametrize(
    "settings",
    [
        {"recovery-attempts": -1},
        {"recovery-attempts": 3},
        {"review-batch-size": 9},
        {"review-if-unsupported": True},
        {"review-batch-size": 2},
    ],
)
def test_invalid_recovery_or_review_controls_are_refused(settings):
    with pytest.raises(ValueError):
        EnrichmentOptions.model_validate(settings)


@pytest.mark.asyncio
async def test_failed_pooled_call_preserves_consumed_tokens_once():
    batcher = ReviewBatcher(2)
    call = AsyncMock(return_value=(review(), 0.3, 11))
    kwargs = {
        "messages": [
            {"role": "system", "content": "Review"},
            {
                "role": "user",
                "content": json.dumps(
                    {"context_items": {}, "drafts": {1: ["Temperature."]}}
                ),
            },
        ]
    }
    errors = await asyncio.gather(
        batcher.submit(call, kwargs),
        batcher.submit(call, kwargs),
        return_exceptions=True,
    )
    assert sum(error.tokens for error in errors) == 11
    assert sum(error.cost for error in errors) == pytest.approx(0.3)


@pytest.mark.asyncio
async def test_budget_refusal_is_never_retried_or_replaced_with_fallback():
    from imas_codex.discovery.base.llm import ProviderBudgetExhausted

    call = AsyncMock(side_effect=ProviderBudgetExhausted("budget exhausted"))
    with pytest.raises(ProviderBudgetExhausted):
        await enrich(EnrichmentOptions(recovery_attempts=2, source_fallback=True), call)
    assert call.await_count == 1


@pytest.mark.asyncio
async def test_full_cohort_delivery_requires_bounded_retry():
    delivered = 0
    for index in range(200):
        value = draft()
        if index in (39, 157):
            value.results[0].signal_index = 3
            replies = [(value, 0, 1), (draft(), 0, 1), (review(), 0, 1)]
        else:
            replies = [(value, 0, 1), (review(), 0, 1)]
        try:
            batch, _, _ = await enrich(
                EnrichmentOptions(
                    review=True, recovery_attempts=1, source_fallback=True
                ),
                AsyncMock(side_effect=replies),
            )
        except ValueError:
            continue
        delivered += bool(batch.results[0].description.strip())
    print(f"Full cohort delivered: {delivered}/200")
    assert delivered == 200
