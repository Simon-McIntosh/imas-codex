"""The compose switch that reranks the nearby-name comparator block.

With the switch off the comparator block reaches the composed prompt exactly
as the embedding retrieval produced it. With it on, each batch item's judged
relevance reorders the block, the spend is booked under the seat that billed
it, and a rerank that cannot run leaves the embedding order in place.
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest

from imas_codex.discovery.base import judgment
from imas_codex.standard_names import budget as budget_module
from imas_codex.standard_names.budget import BudgetManager
from imas_codex.standard_names.worker_tasks.worker_compose_and_validate import (
    compose_worker,
)
from imas_codex.standard_names.worker_tasks.worker_extraction_and_enrichment import (
    _rerank_comparators_for_compose,
    _rerank_comparators_for_worker,
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
async def test_comparator_rerank_fuses_item_orders_by_reciprocal_rank(
    monkeypatch,
) -> None:
    """Two items' orders fuse by reciprocal rank, ties keeping the retrieval order."""
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

    # Reciprocal rank fusion: third and second each take a first and a second
    # place across the two orders, so both carry 1/61 + 1/62; first is twice
    # second with 2/63. The tie breaks by the retrieval order, which ranks
    # second_name above third_name.
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
    ordered, cost = await _reranked_nearby_comparators(
        mgr, _ITEMS, list(_POOL), phase="generate_name"
    )

    assert [candidate["id"] for candidate in ordered] == [
        "third_name",
        "second_name",
        "first_name",
    ]
    assert cost == pytest.approx(0.25)
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
    ordered, cost = await _reranked_nearby_comparators(
        mgr, _ITEMS, pool, phase="generate_name"
    )

    assert ordered is pool, "the block is the retrieval order"
    assert cost == 0.0
    assert seen == [], "no lease means no judgement call"
    assert events == [], "nothing was paid, so nothing is charged"
    assert mgr.spent == 0.0


@pytest.mark.asyncio
async def test_the_off_switch_books_nothing_against_the_budget(monkeypatch) -> None:
    """With the switch off the block is reranked by nobody and reserves nothing.

    A batch run without the rerank must touch the budget exactly as it did
    before the switch existed, so the off path never opens a lease.
    """
    monkeypatch.delenv(_SWITCH, raising=False)
    seen = _stub_rerank(monkeypatch, ["third_name", "second_name", "first_name"])
    events = _record_cost_events(monkeypatch)

    mgr = BudgetManager(total_budget=1.0)
    reservations: list[float] = []
    real_reserve = mgr.reserve

    def counting_reserve(amount, **kwargs):
        reservations.append(amount)
        return real_reserve(amount, **kwargs)

    monkeypatch.setattr(mgr, "reserve", counting_reserve)
    pool = list(_POOL)
    ordered, cost = await _reranked_nearby_comparators(
        mgr, _ITEMS, pool, phase="generate_name"
    )

    assert ordered is pool
    assert cost == 0.0
    assert seen == [], "the switch off must make no judgement call"
    assert events == [], "the switch off must book no charge"
    assert reservations == [], "the switch off must open no lease"
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
    ordered, cost = await _reranked_nearby_comparators(
        mgr, _ITEMS, pool, phase="generate_name"
    )

    assert ordered is pool
    assert cost == 0.0
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


@pytest.mark.asyncio
async def test_the_worker_path_charges_the_rerank_to_a_lease(monkeypatch) -> None:
    """A compose worker's rerank is charged to a lease, not only reported.

    The worker books the same way the batch path does: a lease opened before
    the call, the spend charged under the seat that billed it, and the charge
    recorded even where the pool cannot fund it. The reported compose cost is
    telemetry on top of that, never a substitute for it.
    """
    from imas_codex.settings import get_model

    monkeypatch.setenv(_SWITCH, "1")
    _stub_rerank(monkeypatch, ["third_name", "second_name", "first_name"], cost=0.25)
    events = _record_cost_events(monkeypatch)

    mgr = BudgetManager(total_budget=0.01)
    state = SimpleNamespace(budget_manager=mgr, compose_stats=SimpleNamespace(cost=0.0))

    ordered = await _rerank_comparators_for_worker(state, _ITEMS, list(_POOL))

    assert [candidate["id"] for candidate in ordered] == [
        "third_name",
        "second_name",
        "first_name",
    ]
    assert mgr.spent == pytest.approx(0.25), "the worker path charges a lease"
    assert mgr.overspend == pytest.approx(0.24), (
        "the charge is recorded although the lease and the pool cannot fund it"
    )
    assert len(events) == 1
    assert events[0].model == get_model("sn-judge"), (
        "the worker's charge names the seat that billed the call"
    )
    assert state.compose_stats.cost == pytest.approx(0.25), (
        "the reported compose cost carries the rerank's spend too"
    )


@pytest.mark.asyncio
async def test_the_worker_path_books_nothing_when_the_switch_is_off(
    monkeypatch,
) -> None:
    """With the switch off the worker's rerank opens no lease and reports no cost."""
    monkeypatch.delenv(_SWITCH, raising=False)
    seen = _stub_rerank(monkeypatch, ["third_name", "second_name", "first_name"])
    events = _record_cost_events(monkeypatch)

    mgr = BudgetManager(total_budget=1.0)
    reservations: list[float] = []
    real_reserve = mgr.reserve

    def counting_reserve(amount, **kwargs):
        reservations.append(amount)
        return real_reserve(amount, **kwargs)

    monkeypatch.setattr(mgr, "reserve", counting_reserve)
    state = SimpleNamespace(budget_manager=mgr, compose_stats=SimpleNamespace(cost=0.0))
    pool = list(_POOL)

    ordered = await _rerank_comparators_for_worker(state, _ITEMS, pool)

    assert ordered is pool
    assert seen == [], "the switch off must make no judgement call"
    assert reservations == [], "the switch off must open no lease"
    assert events == [], "the switch off must book no charge"
    assert state.compose_stats.cost == 0.0


def test_compose_worker_books_through_the_charging_helper() -> None:
    """The worker body's rerank goes through the helper that books a lease.

    Without this the worker would keep a rerank whose spend reaches only the
    reported cost, which is the defect the helper removes; a body that stops
    calling the helper fails here.
    """
    source = inspect.getsource(compose_worker)
    assert "_rerank_comparators_for_worker(" in source
    assert "_rerank_comparators_for_compose(" not in source, (
        "the worker body must not reach the unbooked helper directly"
    )
