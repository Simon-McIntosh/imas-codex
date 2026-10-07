"""Batch decision core shared by the discovery and search decision loops.

``decide_batch`` runs a batch of decision states through the decisions
endpoint under one concurrency bound and one wall-time budget. It is the
single loop the judge-and-apply callers build on, so the retry policy, the
concurrency limit and the budget live in one place rather than being repeated
per feature.

A decision state is the ``state`` object the decisions endpoint takes (the
judgement context). Results come back positionally, so a caller can re-attach
them to the items it built the states from. An item whose call failed, or that
was still unscored when the budget elapsed, is returned as ``None`` — a caller
that ranks keeps its own fallback rather than reading a missing judgement as a
zero.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from imas_codex.discovery.base.llm import acall_decisions

logger = logging.getLogger(__name__)

# One decision result: the validated answers keyed by question name, and the
# USD cost the endpoint reported for that call.
DecisionResult = tuple[dict[str, Any], float]

# The decisions endpoint is asked for at most this many judgements at once,
# matching the triage worker's own bound.
_DEFAULT_CONCURRENCY = 8


async def decide_batch(
    states: Sequence[Mapping[str, Any]],
    questions: Mapping[str, Any],
    *,
    model: str,
    service: str,
    concurrency: int = _DEFAULT_CONCURRENCY,
    budget_seconds: float | None = None,
) -> tuple[list[DecisionResult | None], float]:
    """Judge a batch of states under a concurrency bound and a wall-time budget.

    Every state is scored with :func:`acall_decisions`; at most ``concurrency``
    calls run at once. When all states finish before ``budget_seconds`` the
    results are complete; when the budget elapses first, the calls still
    running are cancelled and their states come back as ``None``. A failed
    call is retried inside the call layer and, if it still fails, is reported
    here as ``None`` rather than raised, so one bad state cannot sink the batch.

    Args:
        states: Decision states, one per item, in caller order.
        questions: Question definitions keyed by question name.
        model: Decisions model id for every call in the batch.
        service: Service tag for the API key and the ``X-Title`` header.
        concurrency: Maximum calls in flight at once.
        budget_seconds: Wall-time budget for the batch; ``None`` waits for
            every call.

    Returns:
        ``(results, total_cost)`` — ``results[i]`` is ``(answers, cost)`` for
        the state at index ``i``, or ``None`` when that state failed or was
        unscored at the budget; ``total_cost`` is the summed cost of the
        results that returned.
    """
    if not states:
        return [], 0.0

    semaphore = asyncio.Semaphore(max(1, concurrency))

    async def _judge(state: Mapping[str, Any]) -> DecisionResult:
        async with semaphore:
            return await acall_decisions(model, state, questions, service=service)

    tasks = [asyncio.ensure_future(_judge(state)) for state in states]
    results: list[DecisionResult | None] = [None] * len(tasks)
    total_cost = 0.0

    done, pending = await asyncio.wait(tasks, timeout=budget_seconds)

    for index, task in enumerate(tasks):
        if task not in done:
            continue
        try:
            answers, cost = task.result()
        except Exception as exc:  # noqa: BLE001 - a failed judgement is data
            logger.warning("decision batch item %d failed: %s", index, exc)
            continue
        results[index] = (answers, cost)
        total_cost += cost

    for task in pending:
        task.cancel()
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)

    return results, total_cost


async def rerank_pool(
    query: str,
    candidates: Sequence[Mapping[str, Any]],
    *,
    state_for: Callable[[str, Mapping[str, Any]], Mapping[str, Any]],
    levels: Sequence[str],
    instructions: str,
    model: str,
    service: str,
    question_name: str = "relevance_grade",
    concurrency: int = _DEFAULT_CONCURRENCY,
    budget_seconds: float | None = None,
) -> tuple[list[Mapping[str, Any]], str | None]:
    """Reorder a candidate pool by a graded judgement, keeping order on failure.

    The pool is a retrieval's own order (nearest embedding first). Each
    candidate is scored on the query through :func:`decide_batch`, with a
    single ``score`` question whose ``levels`` run from unrelated to directly
    answers; ``state_for`` builds the one judgement state per candidate, so
    the caller owns what the judge is shown (its text and its locator).
    Scored candidates take the slots in embedding order by descending score and
    a candidate that was unscored keeps its own embedding position, so the
    result is never worse than the order the retrieval produced.

    Args:
        query: The user's query, passed to ``state_for``.
        candidates: The retrieved pool, in retrieval order.
        state_for: Builds one candidate's judgement state from the query.
        levels: The score levels, lowest first.
        instructions: What the judge is asked to weigh.
        model: Decisions model id.
        service: Service tag for the API key.
        question_name: Key the answers come back under.
        concurrency: Maximum judgements in flight at once.
        budget_seconds: Wall-time budget for the whole pool.

    Returns:
        ``(ordered, note)``. ``note`` is ``None`` when every candidate was
        scored, otherwise it names how many kept their retrieval position.
        A failed batch returns the pool in its original order and says so —
        ranking is not a gate, so it never raises.
    """
    items = list(candidates)
    if not items:
        return [], None

    question = {
        question_name: {
            "type": "score",
            "instructions": instructions,
            "criteria": list(levels),
        }
    }
    states = [state_for(query, item) for item in items]
    try:
        results, _cost = await decide_batch(
            states,
            question,
            model=model,
            service=service,
            concurrency=concurrency,
            budget_seconds=budget_seconds,
        )
    except Exception as exc:  # noqa: BLE001 - ranking is not a gate
        logger.warning("rerank failed for %r: %s", query, exc)
        return items, f"rerank unavailable ({exc}); embedding order returned"

    scores: list[float | None] = []
    for result in results:
        if result is None:
            scores.append(None)
        else:
            answers, _call_cost = result
            grade = answers.get(question_name) or {}
            scores.append(grade.get("score"))

    # Scored candidates fill the slots in embedding order by descending score,
    # and an unscored candidate keeps its own embedding position rather than
    # being read as a zero.
    scored_order = sorted(
        (index for index, score in enumerate(scores) if score is not None),
        key=lambda index: (-scores[index], index),
    )
    ordered: list[Mapping[str, Any]] = []
    pointer = 0
    for index, candidate in enumerate(items):
        if scores[index] is None:
            ordered.append(candidate)
        else:
            ordered.append(items[scored_order[pointer]])
            pointer += 1

    unscored = sum(1 for score in scores if score is None)
    if not unscored:
        return ordered, None
    reason = (
        "judgement failed"
        if budget_seconds is None
        else f"unscored at the {budget_seconds}s budget"
    )
    return (
        ordered,
        f"rerank incomplete: {unscored} of {len(items)} candidates kept "
        f"their embedding position ({reason})",
    )
