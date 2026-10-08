"""Tests for the batch decision core in imas_codex.discovery.base.judgment.

``decide_batch`` is driven through the call layer it imports
(``acall_decisions``), so no test here opens the live decisions endpoint.
The autouse guard in ``tests/conftest.py`` fails any test that does.

The decision core skips its calls when the decisions key is absent, so each
test here sets its own key: the judgement it replaces would otherwise never be
consulted and the assertion measures the skip rather than the ordering.
"""

from __future__ import annotations

import asyncio

import pytest

from imas_codex.discovery.base import judgment


@pytest.fixture(autouse=True)
def _decisions_key(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY_IMAS_CODEX", "test-key")


QUESTIONS = {
    "relevance_grade": {
        "type": "score",
        "instructions": "How relevant is this candidate?",
        "criteria": [
            "unrelated",
            "mentions the topic but does not help",
            "useful background",
            "answers a substantial part",
            "directly answers",
        ],
    }
}

MODEL = "typesafe/jev-1.13"
SERVICE = "facility-discovery"


def _answers(score: int) -> dict:
    return {
        "relevance_grade": {
            "type": "score",
            "score": score,
            "probabilities": {
                str(level): (1.0 if level == score else 0.0) for level in range(1, 6)
            },
            "confidence": 0.9,
        }
    }


def _states(n: int) -> list[dict]:
    return [
        {"query": "q", "candidate": {"path": f"f{i}", "text": "t"}} for i in range(n)
    ]


async def test_empty_batch_returns_no_results_and_no_cost():
    results, cost = await judgment.decide_batch(
        [], QUESTIONS, model=MODEL, service=SERVICE
    )
    assert results == []
    assert cost == 0.0


async def test_results_are_positional_and_cost_is_totalled(monkeypatch):
    calls: list[dict] = []

    async def fake(model, state, questions, *, service, **_kwargs):
        calls.append(state)
        index = int(state["candidate"]["path"][1:])
        return _answers(index + 1), 0.01 * (index + 1)

    monkeypatch.setattr(judgment, "acall_decisions", fake)

    results, cost = await judgment.decide_batch(
        _states(4), QUESTIONS, model=MODEL, service=SERVICE
    )

    assert [state["candidate"]["path"] for state in calls] == ["f0", "f1", "f2", "f3"]
    assert [r[0]["relevance_grade"]["score"] for r in results] == [1, 2, 3, 4]
    assert cost == pytest.approx(0.01 + 0.02 + 0.03 + 0.04)


async def test_concurrency_bound_is_never_exceeded(monkeypatch):
    in_flight = 0
    peak = 0

    async def fake(model, state, questions, *, service, **_kwargs):
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        await asyncio.sleep(0.01)
        in_flight -= 1
        return _answers(3), 0.0

    monkeypatch.setattr(judgment, "acall_decisions", fake)

    results, _cost = await judgment.decide_batch(
        _states(12), QUESTIONS, model=MODEL, service=SERVICE, concurrency=3
    )

    assert all(result is not None for result in results)
    assert peak <= 3


async def test_a_failed_call_becomes_none_and_does_not_sink_the_batch(monkeypatch):
    async def fake(model, state, questions, *, service, **_kwargs):
        if state["candidate"]["path"] == "f1":
            raise RuntimeError("the endpoint refused every attempt")
        return _answers(4), 0.02

    monkeypatch.setattr(judgment, "acall_decisions", fake)

    results, cost = await judgment.decide_batch(
        _states(3), QUESTIONS, model=MODEL, service=SERVICE
    )

    assert results[0] is not None
    assert results[1] is None
    assert results[2] is not None
    assert cost == pytest.approx(0.04)


async def test_states_unfinished_at_the_budget_are_none_and_the_batch_returns(
    monkeypatch,
):
    async def fake(model, state, questions, *, service, **_kwargs):
        if state["candidate"]["path"] == "slow":
            await asyncio.sleep(30)
        return _answers(2), 0.03

    monkeypatch.setattr(judgment, "acall_decisions", fake)

    states = [
        {"query": "q", "candidate": {"path": "fast", "text": "t"}},
        {"query": "q", "candidate": {"path": "slow", "text": "t"}},
    ]

    results, cost = await judgment.decide_batch(
        states, QUESTIONS, model=MODEL, service=SERVICE, budget_seconds=0.05
    )

    assert results[0] is not None
    assert results[1] is None
    assert cost == pytest.approx(0.03)


async def test_no_budget_waits_for_a_slow_call(monkeypatch):
    async def fake(model, state, questions, *, service, **_kwargs):
        await asyncio.sleep(0.05)
        return _answers(5), 0.07

    monkeypatch.setattr(judgment, "acall_decisions", fake)

    results, cost = await judgment.decide_batch(
        _states(2), QUESTIONS, model=MODEL, service=SERVICE
    )

    assert all(result is not None for result in results)
    assert cost == pytest.approx(0.14)


def _pool(n: int) -> list[dict]:
    return [{"path": f"f{i}", "text": f"t{i}"} for i in range(n)]


def _state_for(query: str, candidate: dict) -> dict:
    return {
        "query": query,
        "candidate": {
            "locator": {"path": candidate["path"]},
            "text": candidate["text"],
        },
    }


async def test_rerank_pool_orders_by_descending_score(monkeypatch):
    async def fake(model, state, questions, *, service, **_kwargs):
        score = {"f0": 3, "f1": 1, "f2": 5}[state["candidate"]["locator"]["path"]]
        return _answers(score), 0.05

    monkeypatch.setattr(judgment, "acall_decisions", fake)

    ordered, note, cost = await judgment.rerank_pool(
        "q",
        _pool(3),
        state_for=_state_for,
        levels=QUESTIONS["relevance_grade"]["criteria"],
        instructions=QUESTIONS["relevance_grade"]["instructions"],
        model=MODEL,
        service=SERVICE,
    )

    assert [candidate["path"] for candidate in ordered] == ["f2", "f0", "f1"]
    assert note is None
    # the batch cost is the summed per-candidate cost the fake returned
    assert cost == pytest.approx(0.15)


async def test_rerank_pool_keeps_the_pool_order_when_every_judgement_fails(
    monkeypatch,
):
    async def fake(model, state, questions, *, service, **_kwargs):
        raise RuntimeError("endpoint down")

    monkeypatch.setattr(judgment, "acall_decisions", fake)

    ordered, note, cost = await judgment.rerank_pool(
        "q",
        _pool(3),
        state_for=_state_for,
        levels=QUESTIONS["relevance_grade"]["criteria"],
        instructions=QUESTIONS["relevance_grade"]["instructions"],
        model=MODEL,
        service=SERVICE,
        budget_seconds=5.0,
    )

    assert [candidate["path"] for candidate in ordered] == ["f0", "f1", "f2"]
    assert note is not None and "kept their embedding position" in note
    # a wholly failed batch costs nothing
    assert cost == 0.0


async def test_rerank_pool_of_nothing_returns_nothing():
    ordered, note, cost = await judgment.rerank_pool(
        "q",
        [],
        state_for=_state_for,
        levels=QUESTIONS["relevance_grade"]["criteria"],
        instructions=QUESTIONS["relevance_grade"]["instructions"],
        model=MODEL,
        service=SERVICE,
    )

    assert ordered == []
    assert note is None
    assert cost == 0.0


def test_judgment_imports_nothing_from_the_llm_package():
    """The discovery core must not depend on the search-side llm package."""
    import ast
    import pathlib

    tree = ast.parse(pathlib.Path(judgment.__file__).read_text())
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            modules.append(node.module or "")
    assert not [
        module
        for module in modules
        if module == "imas_codex.llm" or module.startswith("imas_codex.llm.")
    ], modules


async def test_judge_rows_applies_only_answers_and_returns_reported_cost(monkeypatch):
    rows = [{"id": "answered"}, {"id": "failed"}]
    applied = []

    async def fake_batch(states, questions, **kwargs):
        assert [state["id"] for state in states] == ["answered", "failed"]
        return [({"admit": {"noul": 0.8}}, 0.012), None], 0.012

    monkeypatch.setattr(judgment, "decide_batch", fake_batch)

    def apply(answered, cost):
        applied.extend(answered)
        assert cost == 0.012
        return {"written": len(answered)}

    result, cost, failed = await judgment.judge_rows(
        rows,
        lambda row: row,
        lambda: {"admit": {"type": "noul"}},
        apply,
        model=MODEL,
        service=SERVICE,
    )
    assert result == {"written": 1}
    assert cost == 0.012
    assert [row["id"] for row in failed] == ["failed"]
    assert [row[0]["id"] for row in applied] == ["answered"]


def test_rerank_pool_is_larger_than_the_returned_search_window():
    from imas_codex.llm.search_tools import RERANK_POOL

    assert RERANK_POOL > 30
