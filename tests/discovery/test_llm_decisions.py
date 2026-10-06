"""Tests for the decisions call layer in imas_codex.discovery.base.llm.

The decisions endpoint is driven through the module's HTTP seam
(``_post_decisions`` / ``_apost_decisions``) with a fake transport, so no
test here opens the live endpoint; an autouse guard in ``tests/conftest.py``
fails any test that does, and one test below drives the real seam to prove
that guard fires.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from imas_codex.discovery.base import llm

QUESTIONS = {
    "loads_diagnostic_data": {
        "type": "noul",
        "instructions": "Does this read measured shot data?",
        "criteria": {
            "true": "reads shot data",
            "false": "does not",
        },
    },
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
MODEL = "typesafe/jev-1.13"
COST = 8.169e-05


def _good_answers() -> dict:
    return {
        "loads_diagnostic_data": {"type": "noul", "noul": 0.82},
        "role": {
            "type": "choice",
            "choice": "diagnostic_data_access",
            "probabilities": {
                "diagnostic_data_access": 0.91,
                "infrastructure_or_utility": 0.09,
            },
            "confidence": 0.85,
        },
    }


class _FakeResponse:
    """Minimal stand-in for an httpx response the call layer consumes."""

    def __init__(self, payload, status_code: int = 200):
        self._payload = payload
        self.status_code = status_code
        self.text = payload if isinstance(payload, str) else json.dumps(payload)

    def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


def _payload(answers: dict | None = None, cost: float = COST) -> dict:
    return {
        "answers": answers if answers is not None else _good_answers(),
        "usage": {"cost": cost},
        "model": MODEL,
    }


@pytest.fixture(autouse=True)
def _api_key(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY_IMAS_CODEX", "test-key")


def test_success_returns_answers_and_cost(monkeypatch):
    """A valid response yields its answers and the usage cost, unchanged."""
    captured: dict = {}

    def fake_post(headers, body, timeout):
        captured["headers"] = headers
        captured["body"] = body
        return _FakeResponse(_payload())

    monkeypatch.setattr(llm, "_post_decisions", fake_post)
    answers, cost = llm.call_decisions(
        MODEL, {"file": {"path": "geteddb.c"}}, QUESTIONS, service=SERVICE
    )

    assert answers == _good_answers()
    assert cost == COST
    # Request contract: the key rides as a bearer token and the service tag as
    # X-Title, the model and questions are forwarded verbatim.
    assert captured["headers"]["Authorization"] == "Bearer test-key"
    assert captured["headers"]["X-Title"] == f"imas-codex:{SERVICE}"
    assert captured["body"]["model"] == MODEL
    assert captured["body"]["questions"] == QUESTIONS
    assert captured["body"]["state"] == {"file": {"path": "geteddb.c"}}


def test_async_success_returns_answers_and_cost(monkeypatch):
    """The async seam delivers the same answers and cost as the sync seam."""

    async def fake_post(headers, body, timeout):
        return _FakeResponse(_payload())

    monkeypatch.setattr(llm, "_apost_decisions", fake_post)
    answers, cost = asyncio.run(
        llm.acall_decisions(MODEL, {}, QUESTIONS, service=SERVICE)
    )
    assert answers == _good_answers()
    assert cost == COST


@pytest.mark.parametrize("status", [429, 500, 503])
def test_retryable_status_then_success(monkeypatch, status):
    """A 429 or 5xx response is retried; the next good response succeeds."""
    calls = {"n": 0}

    def fake_post(headers, body, timeout):
        calls["n"] += 1
        if calls["n"] == 1:
            return _FakeResponse({"error": "busy"}, status)
        return _FakeResponse(_payload())

    monkeypatch.setattr(llm, "_post_decisions", fake_post)
    answers, _ = llm.call_decisions(
        MODEL, {}, QUESTIONS, service=SERVICE, max_retries=3, retry_base_delay=0.0
    )
    assert calls["n"] == 2
    assert answers == _good_answers()


def test_noul_outside_unit_interval_is_refused(monkeypatch):
    """A noul answer above 1 is refused."""
    bad = _good_answers()
    bad["loads_diagnostic_data"] = {"type": "noul", "noul": 1.5}
    monkeypatch.setattr(
        llm, "_post_decisions", lambda h, b, t: _FakeResponse(_payload(bad))
    )
    with pytest.raises(llm.DecisionsValidationError):
        llm.call_decisions(MODEL, {}, QUESTIONS, service=SERVICE, max_retries=1)


def test_choice_outside_criteria_is_refused(monkeypatch):
    """A choice that names a criterion the question never offered is refused."""
    bad = _good_answers()
    bad["role"] = {
        "type": "choice",
        "choice": "simulation_or_solver",
        "probabilities": {
            "diagnostic_data_access": 0.9,
            "infrastructure_or_utility": 0.1,
        },
        "confidence": 0.7,
    }
    monkeypatch.setattr(
        llm, "_post_decisions", lambda h, b, t: _FakeResponse(_payload(bad))
    )
    # The distribution is valid over the offered criteria and sums to 1, so the
    # membership of ``choice`` is the only property left for the validator to
    # refuse; removing that check lets this answer through.
    with pytest.raises(llm.DecisionsValidationError, match="not one of the offered"):
        llm.call_decisions(MODEL, {}, QUESTIONS, service=SERVICE, max_retries=1)


def test_probabilities_not_summing_to_one_is_refused(monkeypatch):
    """A choice distribution that does not sum to 1 is refused."""
    bad = _good_answers()
    bad["role"] = {
        "type": "choice",
        "choice": "diagnostic_data_access",
        "probabilities": {
            "diagnostic_data_access": 0.5,
            "infrastructure_or_utility": 0.2,
        },
    }
    monkeypatch.setattr(
        llm, "_post_decisions", lambda h, b, t: _FakeResponse(_payload(bad))
    )
    with pytest.raises(llm.DecisionsValidationError, match="sum to"):
        llm.call_decisions(MODEL, {}, QUESTIONS, service=SERVICE, max_retries=1)


def test_confidence_outside_unit_interval_is_refused(monkeypatch):
    """A choice confidence outside [0, 1] is refused."""
    bad = _good_answers()
    bad["role"]["confidence"] = 1.2
    monkeypatch.setattr(
        llm, "_post_decisions", lambda h, b, t: _FakeResponse(_payload(bad))
    )
    with pytest.raises(llm.DecisionsValidationError, match="confidence"):
        llm.call_decisions(MODEL, {}, QUESTIONS, service=SERVICE, max_retries=1)


def test_unknown_criterion_probability_is_refused(monkeypatch):
    """A probability assigned to a criterion the question never offered is refused."""
    bad = _good_answers()
    bad["role"]["probabilities"] = {
        "diagnostic_data_access": 0.5,
        "infrastructure_or_utility": 0.4,
        "visualization": 0.1,
    }
    monkeypatch.setattr(
        llm, "_post_decisions", lambda h, b, t: _FakeResponse(_payload(bad))
    )
    with pytest.raises(llm.DecisionsValidationError, match="unknown criterion"):
        llm.call_decisions(MODEL, {}, QUESTIONS, service=SERVICE, max_retries=1)


def test_validation_refusal_is_not_retried(monkeypatch):
    """A contract violation raises on the first response without a retry."""
    calls = {"n": 0}

    def fake_post(headers, body, timeout):
        calls["n"] += 1
        bad = _good_answers()
        bad["loads_diagnostic_data"] = {"type": "noul", "noul": -0.2}
        return _FakeResponse(_payload(bad))

    monkeypatch.setattr(llm, "_post_decisions", fake_post)
    with pytest.raises(llm.DecisionsValidationError):
        llm.call_decisions(
            MODEL, {}, QUESTIONS, service=SERVICE, max_retries=5, retry_base_delay=0.0
        )
    assert calls["n"] == 1


def test_missing_cost_is_refused(monkeypatch):
    """A response without a usable usage.cost is refused, not read as zero."""
    payload = _good_answers()
    response = _FakeResponse({"answers": payload, "usage": {}})
    monkeypatch.setattr(llm, "_post_decisions", lambda h, b, t: response)
    with pytest.raises(llm.DecisionsValidationError, match="cost"):
        llm.call_decisions(MODEL, {}, QUESTIONS, service=SERVICE, max_retries=1)


def test_live_decisions_request_is_refused():
    """The conftest guard trips a real request to the live decisions endpoint.

    This drives the un-replaced HTTP seam, so it exercises the guard that
    protects every other test rather than the call layer's retry logic.
    """
    with pytest.raises(RuntimeError, match="refusing live request"):
        llm._post_decisions({"Authorization": "Bearer x"}, {"model": MODEL}, 5.0)
