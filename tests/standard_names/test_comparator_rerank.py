"""The compose switch that reranks the nearby-name comparator block.

With the switch off the comparator block reaches the composed prompt exactly
as the embedding retrieval produced it. With it on, each batch item's judged
relevance reorders the block.
"""

from __future__ import annotations

import pytest

from imas_codex.discovery.base import judgment
from imas_codex.standard_names.workers import _rerank_comparators_for_compose

_SWITCH = "IMAS_CODEX_SN_COMPOSE_RERANK_COMPARATORS"

_ITEMS = [
    {
        "description": "electron temperature",
        "path": "core_profiles/profiles_1d/electrons/temperature",
    }
]

_POOL = [
    {"id": "first_name", "description": "nearest by embedding"},
    {"id": "second_name", "description": "next by embedding"},
    {"id": "third_name", "description": "last by embedding"},
]


def _stub_rerank(monkeypatch, order, *, cost=0.25):
    """Install a rerank stub that returns *order*, recording what it was asked.

    Each call records the query and the state each candidate is judged in, built
    through the same ``state_for`` the real pool is handed, so a test can pin the
    text the judgement is made on rather than only that a judgement happened.
    """
    seen: list[tuple[str, list[dict]]] = []

    async def fake_rerank_pool(
        query,
        candidates,
        *,
        state_for,
        levels,
        instructions,
        model,
        service,
        question_name="relevance_grade",
        concurrency=8,
        budget_seconds=None,
    ):
        seen.append((query, [state_for(query, candidate) for candidate in candidates]))
        by_id = {candidate["id"]: candidate for candidate in candidates}
        return [by_id[name] for name in order], None, cost

    monkeypatch.setattr(judgment, "rerank_pool", fake_rerank_pool)
    return seen


@pytest.mark.asyncio
async def test_switch_on_reorders_the_comparator_block_by_score(monkeypatch) -> None:
    """With the switch on the block follows the judged score, not the embedding."""
    monkeypatch.setenv(_SWITCH, "1")
    seen = _stub_rerank(monkeypatch, ["third_name", "second_name", "first_name"])

    ordered, cost = await _rerank_comparators_for_compose(_ITEMS, list(_POOL))

    assert [candidate["id"] for candidate in ordered] == [
        "third_name",
        "second_name",
        "first_name",
    ]
    assert cost == 0.25
    assert len(seen) == 1, "the whole block is judged once against the item"

    query, states = seen[0]
    assert query == (
        "electron temperature (core_profiles/profiles_1d/electrons/temperature)"
    ), "the query is the item's description and its DD path"
    assert states == [
        {
            "query": query,
            "candidate": {"locator": "first_name", "text": "nearest by embedding"},
        },
        {
            "query": query,
            "candidate": {"locator": "second_name", "text": "next by embedding"},
        },
        {
            "query": query,
            "candidate": {"locator": "third_name", "text": "last by embedding"},
        },
    ], "each comparator is judged in a state of its own name and description"


@pytest.mark.asyncio
async def test_switch_off_leaves_the_comparator_block_unchanged(monkeypatch) -> None:
    """With the switch off the block is the retrieval order and no call is made."""
    monkeypatch.delenv(_SWITCH, raising=False)
    seen = _stub_rerank(monkeypatch, ["third_name", "second_name", "first_name"])

    pool = list(_POOL)
    ordered, cost = await _rerank_comparators_for_compose(_ITEMS, pool)

    assert ordered is pool
    assert cost == 0.0
    assert seen == [], "the switch off must make no Jev call"


@pytest.mark.asyncio
async def test_comparator_rerank_merges_item_orders_by_mean_rank(monkeypatch) -> None:
    """Two items' orders merge by each comparator's mean rank across them."""
    monkeypatch.setenv(_SWITCH, "1")

    orders = iter(
        [
            ["third_name", "second_name", "first_name"],
            ["second_name", "third_name", "first_name"],
        ]
    )
    seen: list[str] = []

    async def fake_rerank_pool(
        query,
        candidates,
        *,
        state_for,
        levels,
        instructions,
        model,
        service,
        question_name="relevance_grade",
        concurrency=8,
        budget_seconds=None,
    ):
        seen.append(query)
        order = next(orders)
        by_id = {candidate["id"]: candidate for candidate in candidates}
        return [by_id[name] for name in order], None, 0.0

    monkeypatch.setattr(judgment, "rerank_pool", fake_rerank_pool)

    items = [
        {"description": "electron temperature", "path": "a/b"},
        {"description": "ion temperature", "path": "a/c"},
    ]
    ordered, cost = await _rerank_comparators_for_compose(items, list(_POOL))

    # third: (0+1)/2 = 0.5; second: (1+0)/2 = 0.5 tie broken by embedding index
    assert [candidate["id"] for candidate in ordered] == [
        "second_name",
        "third_name",
        "first_name",
    ]
    assert cost == 0.0
    assert len(seen) == 2
