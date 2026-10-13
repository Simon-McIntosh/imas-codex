"""The compose switch that reranks the nearby-name comparator block.

With the switch off the comparator block reaches the composed prompt exactly
as the embedding retrieval produced it. With it on, each batch item's judged
relevance reorders the block, the spend is booked under the seat that billed
it, and a rerank that cannot run leaves the embedding order in place.
"""

from __future__ import annotations

import pytest

from imas_codex.discovery.base import judgment
from imas_codex.standard_names import budget as budget_module
from imas_codex.standard_names.budget import BudgetManager
from imas_codex.standard_names.workers import (
    _rerank_comparators_for_compose,
    _reranked_nearby_comparators,
)

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

    ordered, cost, seat = await _rerank_comparators_for_compose(_ITEMS, list(_POOL))

    assert [candidate["id"] for candidate in ordered] == [
        "third_name",
        "second_name",
        "first_name",
    ]
    assert cost == 0.25
    assert seat, "the caller needs the seat that billed the call"
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
    ordered, cost, seat = await _rerank_comparators_for_compose(_ITEMS, pool)

    assert ordered is pool
    assert cost == 0.0
    assert seat == ""
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
    ordered, cost, _seat = await _rerank_comparators_for_compose(items, list(_POOL))

    # third: (0+1)/2 = 0.5; second: (1+0)/2 = 0.5 tie broken by embedding index
    assert [candidate["id"] for candidate in ordered] == [
        "second_name",
        "third_name",
        "first_name",
    ]
    assert cost == 0.0
    assert len(seen) == 2


def _record_cost_events(monkeypatch) -> list:
    """Capture every charge's LLMCostEvent, returning the real instances."""
    events: list = []
    real = budget_module.LLMCostEvent

    def recording_event(**kwargs):
        event = real(**kwargs)
        events.append(event)
        return event

    monkeypatch.setattr(budget_module, "LLMCostEvent", recording_event)
    return events


@pytest.mark.asyncio
async def test_rerank_spend_is_booked_under_the_seat_that_billed_it(
    monkeypatch,
) -> None:
    """The rerank's cost is charged under the judge seat, not the compose model.

    The lease is opened before the call for the floor amount, so the charge is
    recorded even though that reservation cannot fund it — the provider has
    already billed the spend, and the charge draws the shortfall from the pool
    and beyond it.
    """
    from imas_codex.settings import get_model

    monkeypatch.setenv(_SWITCH, "1")
    _stub_rerank(monkeypatch, ["third_name", "second_name", "first_name"], cost=0.25)
    events = _record_cost_events(monkeypatch)

    mgr = BudgetManager(total_budget=0.01)
    ordered = await _reranked_nearby_comparators(
        mgr, _ITEMS, list(_POOL), phase="generate_name"
    )

    assert [candidate["id"] for candidate in ordered] == [
        "third_name",
        "second_name",
        "first_name",
    ]
    assert mgr.spent == pytest.approx(0.25), "the paid charge is recorded"
    assert mgr.phase_spent["generate_name"] == pytest.approx(0.25)
    assert mgr.overspend == pytest.approx(0.24), (
        "the charge is recorded although neither the lease nor the pool could fund it"
    )
    assert len(events) == 1
    assert events[0].model == get_model("sn-judge"), (
        "the event names the seat the rerank was billed under"
    )
    assert events[0].tokens_in == 0
    assert events[0].tokens_out == 0


@pytest.mark.asyncio
async def test_a_rerank_without_a_lease_is_skipped_and_costs_nothing(
    monkeypatch,
) -> None:
    """An exhausted pool runs the batch in the embedding order, spending nothing.

    A paid charge is never dropped for want of a reservation: with no lease the
    rerank does not run, so there is no spend to lose.
    """
    monkeypatch.setenv(_SWITCH, "1")
    seen = _stub_rerank(monkeypatch, ["third_name", "second_name", "first_name"])
    events = _record_cost_events(monkeypatch)

    mgr = BudgetManager(total_budget=0.0)
    pool = list(_POOL)
    ordered = await _reranked_nearby_comparators(
        mgr, _ITEMS, pool, phase="generate_name"
    )

    assert ordered is pool, "the block is the retrieval order"
    assert seen == [], "no lease means no judgement call"
    assert events == [], "nothing was paid, so nothing is charged"
    assert mgr.spent == 0.0


@pytest.mark.asyncio
async def test_a_rerank_that_raises_leaves_the_embedding_order(monkeypatch) -> None:
    """A failing rerank is not a gate: the block keeps the embedding order."""
    monkeypatch.setenv(_SWITCH, "1")

    async def failing_rerank_pool(*_args, **_kwargs):
        raise RuntimeError("no decision service")

    monkeypatch.setattr(judgment, "rerank_pool", failing_rerank_pool)
    events = _record_cost_events(monkeypatch)

    mgr = BudgetManager(total_budget=1.0)
    pool = list(_POOL)
    ordered = await _reranked_nearby_comparators(
        mgr, _ITEMS, pool, phase="generate_name"
    )

    assert ordered is pool
    assert events == [], "a failed call is never charged"
    assert mgr.remaining == pytest.approx(1.0), "the unused lease is released"


@pytest.mark.asyncio
async def test_a_rerank_whose_seat_cannot_be_resolved_keeps_the_order(
    monkeypatch,
) -> None:
    """A raise out of the seat lookup keeps the order instead of aborting."""
    monkeypatch.setenv(_SWITCH, "1")

    def failing_get_model(*_args, **_kwargs):
        raise KeyError("sn-judge")

    monkeypatch.setattr("imas_codex.settings.get_model", failing_get_model)

    pool = list(_POOL)
    ordered, cost, seat = await _rerank_comparators_for_compose(_ITEMS, pool)

    assert ordered == pool
    assert cost == 0.0
    assert seat == ""
