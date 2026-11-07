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

import asyncio
import json

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
