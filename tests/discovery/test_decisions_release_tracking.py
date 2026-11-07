"""The dated build that answered a decisions call reaches its cost record.

A seat names a pinned model id (``typesafe/jev-1.14``), and the endpoint may
re-point that id at a newer build without the id changing. The response's own
``model`` field is the dated build that served the call, and these tests hold
the two places that must carry it: the call layer, through ``record_release``
sink, and the cost record, through :func:`judgment._book_cost`.

The endpoint is driven through the module's HTTP seams with a fake
transport, so no test here spends anything.
"""

from __future__ import annotations

import ast
import asyncio
import json
import pathlib

import pytest

from imas_codex.discovery.base import judgment, llm
from imas_codex.ids.mapping import PipelineCost

QUESTIONS = {
    "role": {
        "type": "choice",
        "instructions": "Which role best describes the file?",
        "criteria": {
            "diagnostic_data_access": "reads shot data",
            "infrastructure_or_utility": "build files and utilities",
        },
    },
}

SERVICE = "facility-discovery"
PINNED_MODEL = "typesafe/jev-1.14"
MODEL_RELEASE = "typesafe/jev-1.14-20261014"
LATER_RELEASE = "typesafe/jev-1.14-20261201"
COST = 1.4658e-05


class _FakeResponse:
    def __init__(self, payload: dict):
        self._payload = payload
        self.status_code = 200
        self.text = json.dumps(payload)

    def json(self):
        return self._payload


def _payload(release: str) -> dict:
    return {
        "answers": {
            "role": {
                "type": "choice",
                "choice": "diagnostic_data_access",
                "probabilities": {
                    "diagnostic_data_access": 0.97,
                    "infrastructure_or_utility": 0.03,
                },
                "confidence": 0.95,
            }
        },
        "usage": {"cost": COST},
        "model": release,
    }


@pytest.fixture(autouse=True)
def _api_key(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY_IMAS_CODEX", "test-key")


def test_sync_call_reports_the_dated_build(monkeypatch):
    monkeypatch.setattr(
        llm, "_post_decisions", lambda h, b, t: _FakeResponse(_payload(MODEL_RELEASE))
    )
    seen: list[str] = []
    answers, cost = llm.call_decisions(
        PINNED_MODEL,
        {"file": {"path": "geteddb.c"}},
        QUESTIONS,
        service=SERVICE,
        record_release=seen.append,
    )
    assert cost == COST
    assert answers["role"]["choice"] == "diagnostic_data_access"
    assert seen == [MODEL_RELEASE]


def test_async_call_reports_the_dated_build(monkeypatch):
    async def fake_post(headers, body, timeout):
        return _FakeResponse(_payload(MODEL_RELEASE))

    monkeypatch.setattr(llm, "_apost_decisions", fake_post)
    seen: list[str] = []
    asyncio.run(
        llm.acall_decisions(
            PINNED_MODEL, {}, QUESTIONS, service=SERVICE, record_release=seen.append
        )
    )
    assert seen == [MODEL_RELEASE]


def test_a_caller_that_ignores_the_sink_still_gets_answers_and_cost(monkeypatch):
    monkeypatch.setattr(
        llm, "_post_decisions", lambda h, b, t: _FakeResponse(_payload(MODEL_RELEASE))
    )
    answers, cost = llm.call_decisions(PINNED_MODEL, {}, QUESTIONS, service=SERVICE)
    assert answers["role"]["choice"] == "diagnostic_data_access"
    assert cost == COST


def test_decide_one_records_the_release_beside_the_spend(monkeypatch):
    monkeypatch.setattr(
        llm, "_post_decisions", lambda h, b, t: _FakeResponse(_payload(MODEL_RELEASE))
    )
    tracker = PipelineCost()
    judgment.decide_one(
        PINNED_MODEL,
        {"file": {"path": "geteddb.c"}},
        QUESTIONS,
        service=SERVICE,
        cost=tracker,
    )
    assert tracker.steps["decision"] == COST
    assert tracker.releases["decision"] == MODEL_RELEASE


def test_decide_batch_records_the_release_for_every_state(monkeypatch):
    async def fake_post(headers, body, timeout):
        return _FakeResponse(_payload(MODEL_RELEASE))

    monkeypatch.setattr(llm, "_apost_decisions", fake_post)
    tracker = PipelineCost()
    asyncio.run(
        judgment.decide_batch(
            [{"file": {"path": "a.c"}}, {"file": {"path": "b.c"}}],
            QUESTIONS,
            model=PINNED_MODEL,
            service=SERVICE,
            cost=tracker,
        )
    )
    assert tracker.releases["decision"] == MODEL_RELEASE


def test_a_repointed_build_shows_while_the_pinned_id_is_unchanged(monkeypatch):
    """The spend line is identical; only the release records the change."""
    tracker = PipelineCost()
    for release in (MODEL_RELEASE, LATER_RELEASE):
        monkeypatch.setattr(
            llm,
            "_post_decisions",
            lambda h, b, t, r=release: _FakeResponse(_payload(r)),
        )
        judgment.decide_one(PINNED_MODEL, {}, QUESTIONS, service=SERVICE, cost=tracker)
        assert tracker.releases["decision"] == release
    assert tracker.steps["decision"] == pytest.approx(2 * COST)


def test_a_tracker_without_release_storage_still_books_the_spend(monkeypatch):
    class SpendOnly:
        def __init__(self) -> None:
            self.steps: dict[str, float] = {}

        def add(self, step: str, cost: float, tokens: int) -> None:
            self.steps[step] = self.steps.get(step, 0) + cost

    monkeypatch.setattr(
        llm, "_post_decisions", lambda h, b, t: _FakeResponse(_payload(MODEL_RELEASE))
    )
    tracker = SpendOnly()
    judgment.decide_one(PINNED_MODEL, {}, QUESTIONS, service=SERVICE, cost=tracker)
    assert tracker.steps["decision"] == COST


def test_a_stored_llm_cost_carries_the_release():
    """The graph row, not just the tracker, holds the build that answered.

    ``record_llm_cost`` is the only writer of a ``:LLMCost`` node, so the
    property is asserted where the row is built: the release travels through
    the event and lands as its own property beside ``llm_model``.
    """
    import os
    import uuid
    from datetime import datetime

    from imas_codex.standard_names.graph_store.graph_cost_and_claims import (
        record_llm_cost,
    )

    captured: dict = {}

    class _Graph:
        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

        def query(self, cypher, **params):
            captured["cypher"] = cypher
            captured["params"] = params

    class _Services:
        GraphClient = _Graph

        @staticmethod
        def _normalize_pool(phase):
            return phase

    _Services.datetime = datetime
    _Services.uuid = uuid
    _Services.os = os

    record_llm_cost(
        run_id="run-1",
        phase="sn_judge",
        model=PINNED_MODEL,
        model_release=MODEL_RELEASE,
        cost=COST,
        tokens_in=0,
        tokens_out=0,
        services=_Services,
    )
    assert captured["params"]["llm_model"] == PINNED_MODEL
    assert captured["params"]["llm_model_release"] == MODEL_RELEASE
    assert "llm_model_release" in captured["cypher"]


def test_each_call_charges_its_own_record_with_its_own_release(monkeypatch):
    """A batch is many records, not one: the releases never collapse."""
    charged: list[tuple[str, str | None, str | None]] = []
    tracker = PipelineCost(
        charge=lambda step, spent, model, release: charged.append(
            (step, model, release)
        )
    )
    for release in (MODEL_RELEASE, LATER_RELEASE):
        monkeypatch.setattr(
            llm,
            "_post_decisions",
            lambda h, b, t, r=release: _FakeResponse(_payload(r)),
        )
        judgment.decide_one(
            PINNED_MODEL, {}, QUESTIONS, service=SERVICE, cost=tracker, step="sn_judge"
        )
    assert [record[2] for record in charged] == [MODEL_RELEASE, LATER_RELEASE]
    assert {record[0] for record in charged} == {"sn_judge"}
    assert {record[1] for record in charged} == {PINNED_MODEL}


def _payload_with_cost(release: str, cost: float) -> dict:
    payload = _payload(release)
    payload["usage"] = {"cost": cost}
    return payload


def test_a_zero_cost_decision_still_records_its_release(monkeypatch):
    """A free answer is still an answer, and its build is still the record.

    The provider admits a zero ``usage.cost``. Pricing the reservation for
    that spend is meaningless and the pool refuses an amount of zero, so the
    record has to be written without one rather than the call being dropped or
    the stage crashing.
    """
    charged: list[tuple[float, str | None]] = []
    monkeypatch.setattr(
        llm,
        "_post_decisions",
        lambda h, b, t: _FakeResponse(_payload_with_cost(MODEL_RELEASE, 0.0)),
    )
    tracker = PipelineCost(
        charge=lambda step, spent, model, release: charged.append((spent, release))
    )
    judgment.decide_one(
        PINNED_MODEL, {}, QUESTIONS, service=SERVICE, cost=tracker, step="sn_judge"
    )
    assert tracker.steps["sn_judge"] == 0.0
    assert tracker.releases["sn_judge"] == MODEL_RELEASE
    assert charged == [(0.0, MODEL_RELEASE)]


class _RecordingLease:
    def __init__(self) -> None:
        self.events: list[tuple[float, object]] = []

    def charge_event(self, cost, event):
        self.events.append((cost, event))
        return None


class _RecordingManager:
    """A budget manager that refuses nothing and records what it was asked
    for, so a zero-cost call can be shown to reserve nothing at all."""

    def __init__(self) -> None:
        self.reserved: list[float] = []
        self.written: list[tuple[float, object]] = []
        self.lease = _RecordingLease()

    def reserve(self, amount, phase=""):
        self.reserved.append(amount)
        return self.lease

    def record_llm_cost_event(self, cost, event) -> None:
        self.written.append((cost, event))


class _SinkState:
    def __init__(self, manager) -> None:
        self.budget_manager = manager


def _sink(manager):
    from imas_codex.standard_names.review.pipeline import _judge_cost_sink

    return _judge_cost_sink(_SinkState(manager))


def test_the_judge_sink_writes_a_zero_cost_row_without_reserving():
    manager = _RecordingManager()
    _sink(manager)("sn_judge", 0.0, PINNED_MODEL, MODEL_RELEASE)
    assert manager.reserved == []
    assert len(manager.written) == 1
    cost, event = manager.written[0]
    assert cost == 0.0
    assert event.model_release == MODEL_RELEASE
    assert event.model == PINNED_MODEL


def test_the_judge_sink_reserves_and_charges_a_paid_call():
    manager = _RecordingManager()
    _sink(manager)("sn_judge", COST, PINNED_MODEL, MODEL_RELEASE)
    assert manager.reserved == [COST]
    assert manager.lease.events[0][0] == COST
    assert manager.lease.events[0][1].model_release == MODEL_RELEASE


def test_the_judge_sink_rejects_a_negative_cost():
    manager = _RecordingManager()
    with pytest.raises(ValueError):
        _sink(manager)("sn_judge", -COST, PINNED_MODEL, MODEL_RELEASE)


def _function(tree, name):
    for node in ast.walk(tree):
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == name
        ):
            return node
    raise AssertionError(f"{name} is missing from the module")


def test_the_review_enrich_worker_binds_the_judge_cost_sink():
    """Reverting the wiring to a bare tracker must turn this red.

    A behavioural test of the enrich worker needs a full review state; the
    wiring itself is what can silently regress, so it is read off the module's
    own syntax tree: the judge's tracker must be constructed with a ``charge``
    sink.
    """
    from imas_codex.standard_names.review import pipeline

    tree = ast.parse(pathlib.Path(pipeline.__file__).read_text())
    worker = _function(tree, "enrich_review_worker")
    charges = [
        keyword
        for node in ast.walk(worker)
        if isinstance(node, ast.Call)
        and getattr(node.func, "id", getattr(node.func, "attr", "")) == "PipelineCost"
        for keyword in node.keywords
        if keyword.arg == "charge"
    ]
    assert charges, "enrich_review_worker no longer binds a charge sink to judge_cost"


def test_the_docs_contradiction_call_stamps_the_release_it_was_served():
    """Reverting the stamp leaves the Jev charge without its build."""
    from imas_codex.standard_names.worker_tasks import worker_review_docs

    tree = ast.parse(pathlib.Path(worker_review_docs.__file__).read_text())
    note = _function(tree, "_documentation_contradiction_note")
    events = [
        node
        for node in ast.walk(note)
        if isinstance(node, ast.Call)
        and getattr(node.func, "id", getattr(node.func, "attr", "")) == "LLMCostEvent"
        for keyword in node.keywords
        if keyword.arg == "model_release"
    ]
    assert events, "the review_docs Jev charge no longer stamps model_release"
