"""Batch decision core shared by the discovery and search decision loops.

``decide_batch`` runs a batch of decision states through the decisions
endpoint under one concurrency bound and one wall-time budget. It is the
single loop the judge-and-apply callers build on, so the retry policy, the
concurrency limit and the budget live in one place rather than being repeated
per feature. :func:`decide_one` is the synchronous single-state counterpart for
a caller that judges one item and needs its answer now.

A decision state is the ``state`` object the decisions endpoint takes (the
judgement context). Results come back positionally, so a caller can re-attach
them to the items it built the states from. An item whose call failed, or that
was still unscored when the budget elapsed, is returned as ``None`` — a caller
that ranks keeps its own fallback rather than reading a missing judgement as a
zero.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import Callable, Mapping, Sequence
from enum import Enum
from typing import Any, NamedTuple

from imas_codex.discovery.base.llm import (
    DecisionsValidationError,
    acall_decisions,
    call_decisions,
    fault_text,
    get_api_key,
)

logger = logging.getLogger(__name__)

# One decision result: the validated answers keyed by question name, and the
# USD cost the endpoint reported for that call.
DecisionResult = tuple[dict[str, Any], float]
JudgedRow = tuple[Mapping[str, Any], dict[str, Any], float]

# The decisions endpoint is asked for at most this many judgements at once,
# matching the triage worker's own bound.
_DEFAULT_CONCURRENCY = 8

# Pipeline-cost step a decision books to when the caller names no other.
DEFAULT_COST_STEP = "decision"


def _book_cost(cost: Any | None, step: str, spent: float) -> None:
    """Add one call's reported USD cost to a caller's tracker.

    Every decisions call this module makes reports its own cost; this is the
    single place that turns one into a tracker entry, so a caller cannot
    silently drop the spend and the two entry points below cannot drift in how
    they book it. A ``None`` tracker is a caller that does not accumulate cost.
    """
    if cost is not None:
        cost.add(step, spent, 0)


class CallStatus(Enum):
    """How one decisions call ended, before an entry point applies its policy.

    A *malformed answer* is deterministic — the endpoint returned a body that
    violates the question contract — so it is never retried or read as an
    absent judgement. A *transport fault* is any other failure of the call.
    """

    OK = "ok"
    MALFORMED = "malformed"
    TRANSPORT = "transport"


class CallOutcome(NamedTuple):
    """One decisions call, settled: its status and, when OK, its result.

    ``fault`` carries the exception a non-OK call raised, so a caller that
    rejects a malformed answer re-raises the error the call layer produced
    rather than a freshly built one. ``fault_text`` is that fault's own
    description, always non-empty, so a log line or a recorded reason built
    from an outcome never reads as a missing value.
    """

    status: CallStatus
    answers: dict[str, Any] | None
    spent: float
    fault: BaseException | None
    fault_text: str


def _settle_call(
    call: Callable[[], tuple[dict[str, Any], float]],
    *,
    cost: Any | None,
    step: str,
) -> CallOutcome:
    """Make one decisions call, classify its fault, and book its cost.

    The one per-call unit both entry points rest on. ``call`` performs the
    transport — ``call_decisions`` for the synchronous caller, an awaited
    ``acall_decisions`` for the batch — and returns ``(answers, spent)``. This
    unit is the only place a fault is classified, so the two entry points
    cannot disagree on whether a failure is a malformed answer or a transport
    fault, and the only place a reported cost is booked, so neither can drop
    the spend. A successful call's cost is added to ``cost`` under ``step``; a
    failed call books nothing. Each entry point applies its own policy to the
    returned status: the batch maps both faults to ``None``, the single-state
    caller raises the malformed answer and maps the transport fault to ``None``.
    A fault's text is recorded with the same care as the fault itself, so an
    exception raised with a bare message still leaves a reason a reader can
    act on.
    """
    try:
        answers, spent = call()
    except DecisionsValidationError as exc:
        return CallOutcome(CallStatus.MALFORMED, None, 0.0, exc, fault_text(exc))
    except Exception as exc:  # noqa: BLE001 - a failed judgement is data
        return CallOutcome(CallStatus.TRANSPORT, None, 0.0, exc, fault_text(exc))
    _book_cost(cost, step, spent)
    return CallOutcome(CallStatus.OK, answers, spent, None, "")


def decide_one(
    model: str,
    state: Mapping[str, Any],
    questions: Mapping[str, Any],
    *,
    service: str,
    cost: Any | None = None,
    step: str = DEFAULT_COST_STEP,
) -> dict[str, Any] | None:
    """Judge a single state, returning ``None`` when the call itself fails.

    The synchronous single-state counterpart of :func:`decide_batch` for a
    caller that judges one item and needs its answer now. The call, its fault
    classification and its cost booking are the shared unit's (:func:`_settle_call`);
    this entry point only applies the single-state policy on top of that
    classification. It differs from the batch in one deliberate way: a
    malformed answer raises :class:`DecisionsValidationError` rather than being
    swallowed, because a contract violation is deterministic and must be
    rejected, never mistaken for a retryable transport fault. Any other failure
    returns ``None``, so a caller that ranks rather than gates leaves the item
    unjudged for a retry.

    Args:
        model: Decisions model id.
        state: The one judgement context.
        questions: Question definitions keyed by question name.
        service: Service tag for the API key and the ``X-Title`` header.
        cost: Optional run cost tracker; the call's reported cost is booked to
            it under ``step``.
        step: Pipeline-cost step name the reported cost is booked to.

    Returns:
        The validated answers keyed by question name, or ``None`` when the
        transport failed.
    """
    outcome = _settle_call(
        lambda: call_decisions(model, state, questions, service=service),
        cost=cost,
        step=step,
    )
    if outcome.status is CallStatus.MALFORMED:
        assert outcome.fault is not None
        raise outcome.fault
    if outcome.status is CallStatus.TRANSPORT:
        logger.warning("decision call failed: %s", outcome.fault_text)
        return None
    return outcome.answers


def decisions_key_present() -> bool:
    """Whether the decisions endpoint has a configured API key.

    The credential is resolved through :func:`get_api_key`, so its presence is
    reported in one place instead of by a second reading of the environment
    here. A caller that pays per call checks this first and skips cleanly when
    it is false, so an unconfigured key costs no request rather than a failed
    one.
    """
    try:
        get_api_key()
    except ValueError:
        return False
    return True


def decisions_key_absent_reason() -> str | None:
    """Why the decisions key is unusable, or ``None`` when it is present.

    Returns :func:`get_api_key`'s own message, which names the missing
    variable, so a skip note carries the credential's name without spelling it
    a second time.
    """
    try:
        get_api_key()
    except ValueError as exc:
        return str(exc)
    return None


async def _await_call(
    call: Any,
) -> tuple[dict[str, Any] | None, float, BaseException | None]:
    """Await one decisions call, returning its result or its fault as data.

    The transport half of the batch's per-call unit. It awaits and captures
    and classifies nothing, so the fault reaches :func:`_settle_call` — the one
    place a fault is classified — rather than being read here. A cancellation
    is a ``BaseException`` and is left to propagate, so it is never mistaken
    for a transport fault.
    """
    try:
        answers, spent = await call
    except Exception as exc:  # noqa: BLE001 - the fault is classified downstream
        return None, 0.0, exc
    return answers, spent, None


def _resume(
    answers: dict[str, Any] | None,
    spent: float,
    fault: BaseException | None,
) -> tuple[dict[str, Any], float]:
    """Re-present an awaited call's outcome to :func:`_settle_call` as a thunk.

    A captured fault is re-raised here so the shared unit — and not this
    adapter — decides whether it is a malformed answer or a transport fault.
    """
    if fault is not None:
        raise fault
    assert answers is not None
    return answers, spent


async def decide_batch(
    states: Sequence[Mapping[str, Any]],
    questions: Mapping[str, Any],
    *,
    model: str,
    service: str,
    concurrency: int = _DEFAULT_CONCURRENCY,
    budget_seconds: float | None = None,
    cost: Any | None = None,
    step: str = DEFAULT_COST_STEP,
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
        cost: Optional run cost tracker; the batch's summed reported cost is
            booked to it under ``step``.
        step: Pipeline-cost step name the reported cost is booked to.

    Returns:
        ``(results, total_cost)`` — ``results[i]`` is ``(answers, cost)`` for
        the state at index ``i``, or ``None`` when that state failed or was
        unscored at the budget; ``total_cost`` is the summed cost of the
        results that returned.
    """
    if not states:
        return [], 0.0

    semaphore = asyncio.Semaphore(max(1, concurrency))

    async def _judge(state: Mapping[str, Any]) -> CallOutcome:
        async with semaphore:
            answers, spent, fault = await _await_call(
                acall_decisions(model, state, questions, service=service)
            )
            return _settle_call(
                lambda: _resume(answers, spent, fault), cost=cost, step=step
            )

    tasks = [asyncio.ensure_future(_judge(state)) for state in states]
    results: list[DecisionResult | None] = [None] * len(tasks)
    total_cost = 0.0

    done, pending = await asyncio.wait(tasks, timeout=budget_seconds)

    for index, task in enumerate(tasks):
        if task not in done:
            continue
        outcome = task.result()
        if outcome.status is not CallStatus.OK:
            logger.warning(
                "decision batch item %d failed: %s", index, outcome.fault_text
            )
            continue
        assert outcome.answers is not None
        results[index] = (outcome.answers, outcome.spent)
        total_cost += outcome.spent

    for task in pending:
        task.cancel()
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)

    return results, total_cost


async def judge_rows(
    rows: Sequence[Mapping[str, Any]],
    state_for: Callable[[Mapping[str, Any]], Mapping[str, Any]],
    questions_for: Callable[[], Mapping[str, Any]],
    apply: Callable[[list[JudgedRow], float], Any],
    *,
    model: str,
    service: str,
    concurrency: int = _DEFAULT_CONCURRENCY,
    budget_seconds: float | None = None,
) -> tuple[Any, float, list[Mapping[str, Any]]]:
    """Judge claimed rows, apply only successful answers, and return real spend.

    The caller owns its claim and releases failed rows. An unanswered row never
    reaches ``apply`` and retains its prior judgment and durable status.
    ``apply`` may be synchronous or asynchronous so a content arm can describe
    admitted files before persisting the judgment.
    """
    if not rows:
        return None, 0.0, []
    states = [state_for(row) for row in rows]
    results, cost = await decide_batch(
        states,
        questions_for(),
        model=model,
        service=service,
        concurrency=concurrency,
        budget_seconds=budget_seconds,
    )
    answered = [
        (row, result[0], result[1])
        for row, result in zip(rows, results, strict=True)
        if result is not None
    ]
    failed = [row for row, result in zip(rows, results, strict=True) if result is None]
    if not answered:
        return None, cost, failed
    applied = apply(answered, cost)
    if inspect.isawaitable(applied):
        applied = await applied
    return applied, cost, failed


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
) -> tuple[list[Mapping[str, Any]], str | None, float]:
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
        ``(ordered, note, cost)``. ``note`` is ``None`` when every candidate
        was scored, otherwise it names how many kept their retrieval position.
        ``cost`` is the batch's summed decision cost, ``0.0`` when the whole
        batch failed or when the decisions key is absent. A failed batch
        returns the pool in its original order and says so — ranking is not a
        gate, so it never raises. When the decisions key is not set the pool is
        returned in retrieval order with a note naming the missing variable and
        no call is made.
    """
    items = list(candidates)
    if not items:
        return [], None, 0.0

    if not decisions_key_present():
        reason = decisions_key_absent_reason() or "decisions key not set"
        return items, f"rerank skipped: {reason}", 0.0

    question = {
        question_name: {
            "type": "score",
            "instructions": instructions,
            "criteria": list(levels),
        }
    }
    states = [state_for(query, item) for item in items]
    try:
        results, cost = await decide_batch(
            states,
            question,
            model=model,
            service=service,
            concurrency=concurrency,
            budget_seconds=budget_seconds,
        )
    except Exception as exc:  # noqa: BLE001 - ranking is not a gate
        logger.warning("rerank failed for %r: %s", query, exc)
        return items, f"rerank unavailable ({exc}); embedding order returned", 0.0

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
    scored_values = [
        (index, score) for index, score in enumerate(scores) if score is not None
    ]
    scored_order = [
        index
        for index, _ in sorted(scored_values, key=lambda item: (-item[1], item[0]))
    ]
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
        return ordered, None, cost
    reason = (
        "judgement failed"
        if budget_seconds is None
        else f"unscored at the {budget_seconds}s budget"
    )
    return (
        ordered,
        f"rerank incomplete: {unscored} of {len(items)} candidates kept "
        f"their embedding position ({reason})",
        cost,
    )
