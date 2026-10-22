"""What a failed decision call reports, and which faults it retries.

Two properties of the decision call layer are pinned here. A fault whose
message is empty must still be identifiable wherever it is written down, and
a transport fault must be retried because of what it is rather than because
of what its message says — a timeout raised with a bare message is the case a
substring policy alone reads as permanent and gives one attempt.

The HTTP seam is replaced in every test, so no test here reaches the live
decisions endpoint; the autouse guard in ``tests/conftest.py`` fails any test
that does. Each test that guards a behaviour names the mutation it must fail
against, so the control run recorded beside this node's evidence shows the
guard firing rather than merely passing.
"""

from __future__ import annotations

import logging

import httpx
import pytest

from imas_codex.discovery.base import judgment, llm

MODEL = "typesafe/jev-1.13"
SERVICE = "facility-discovery"
COST = 8.169e-05

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


def _state() -> dict:
    return {"candidate": {"locator": {"path": "geteddb.c"}, "text": "reads shot data"}}


def _answers() -> dict:
    return {
        "relevance_grade": {
            "type": "score",
            "score": 3,
            "probabilities": dict.fromkeys((str(i) for i in range(5)), 0.2),
            "confidence": 0.8,
        }
    }


class _FakeResponse:
    """Minimal stand-in for the httpx response the call layer consumes."""

    def __init__(self, payload: dict, status_code: int = 200):
        self._payload = payload
        self.status_code = status_code
        self.text = str(payload)

    def json(self):
        return self._payload


def _payload() -> dict:
    return {"answers": _answers(), "usage": {"cost": COST}, "model": MODEL}


@pytest.fixture(autouse=True)
def _decisions_key(monkeypatch):
    """Every test here resolves the service credential on its way to the seam."""
    monkeypatch.setenv("OPENROUTER_API_KEY_IMAS_CODEX", "test-key")


async def _no_backoff(model, state, questions, *, service, **_kwargs):
    """The real async call layer, driven without its backoff waits.

    ``decide_batch`` takes no retry-delay argument, and the call layer's
    default is bound when it is defined, so the wait is removed by calling it
    here rather than by patching a module constant. The retry loop under test
    is the call layer's own.
    """
    return await llm.acall_decisions(
        model, state, questions, service=service, max_retries=3, retry_base_delay=0.0
    )


def _recorded(caplog) -> str:
    return "\n".join(record.getMessage() for record in caplog.records)


def test_empty_message_transport_fault_is_retried_then_succeeds(monkeypatch):
    """A transport fault raised with an empty message is retried, not raised.

    The fault of record is an HTTP read timeout carrying an empty message, so
    a policy that classifies by substring alone finds no pattern to match and
    gives the call a single attempt. Reverting the classification to its
    message-only form turns this test red at ``calls["n"] == 2``.
    """
    calls = {"n": 0}

    def fake_post(headers, body, timeout):
        calls["n"] += 1
        if calls["n"] == 1:
            raise httpx.ReadTimeout("")
        return _FakeResponse(_payload())

    monkeypatch.setattr(llm, "_post_decisions", fake_post)
    answers, cost = llm.call_decisions(
        MODEL,
        _state(),
        QUESTIONS,
        service=SERVICE,
        max_retries=3,
        retry_base_delay=0.0,
    )

    assert calls["n"] == 2
    assert answers == _answers()
    assert cost == COST


async def test_empty_message_transport_fault_is_retried_through_the_batch(
    monkeypatch,
):
    """The same fault is retried on the async path the batch entry point uses."""
    calls = {"n": 0}

    async def fake_post(headers, body, timeout):
        calls["n"] += 1
        if calls["n"] == 1:
            raise httpx.RemoteProtocolError("")
        return _FakeResponse(_payload())

    monkeypatch.setattr(llm, "_apost_decisions", fake_post)
    monkeypatch.setattr(judgment, "acall_decisions", _no_backoff)

    results, total_cost = await judgment.decide_batch(
        [_state()], QUESTIONS, model=MODEL, service=SERVICE
    )

    assert calls["n"] == 2
    assert results == [(_answers(), COST)]
    assert total_cost == COST


async def test_exhausted_transport_fault_is_reported_with_its_type_name(
    monkeypatch, caplog
):
    """A message-less fault that outlasts its retries names its type when a
    decision call gives up on it.

    This is the whole path the run log was missing: the call layer records
    each attempt, exhausts its retries, and the batch reports the item as
    unjudged with a reason that identifies the failure. Reverting the fault's
    description to bare ``str(error)`` turns the assertion on the batch
    warning red, and the reported line reads ``failed: `` with nothing after
    it.
    """
    calls = {"n": 0}

    async def fake_post(headers, body, timeout):
        calls["n"] += 1
        raise httpx.ReadTimeout("")

    monkeypatch.setattr(llm, "_apost_decisions", fake_post)
    monkeypatch.setattr(judgment, "acall_decisions", _no_backoff)

    with caplog.at_level(logging.DEBUG):
        results, total_cost = await judgment.decide_batch(
            [_state()], QUESTIONS, model=MODEL, service=SERVICE
        )

    assert results == [None]
    assert total_cost == 0.0
    assert calls["n"] == 3, "every retry is attempted before the item is given up"
    text = _recorded(caplog)
    assert "decision batch item 0 failed: ReadTimeout: ReadTimeout('')" in text
    assert "decisions error (attempt 1/3)" in text


def test_empty_message_fault_is_described_by_type_and_repr(monkeypatch, caplog):
    """The single-state caller writes a description, never an empty field.

    The message carries no text at all, so a description taken from
    ``str(error)`` alone would leave the warning's fault field blank and the
    line would not say which failure it reported.
    """
    raised = httpx.ReadTimeout("")

    def raising_call(model, state, questions, *, service, **_kwargs):
        raise raised

    monkeypatch.setattr(judgment, "call_decisions", raising_call)

    with caplog.at_level(logging.WARNING, logger=judgment.__name__):
        assert judgment.decide_one(MODEL, _state(), QUESTIONS, service=SERVICE) is None

    assert _recorded(caplog).strip() == (
        "decision call failed: ReadTimeout: ReadTimeout('')"
    )


def test_validation_error_is_not_retried(monkeypatch):
    """A contract violation raises on the first answer without a second call."""
    calls = {"n": 0}

    def fake_post(headers, body, timeout):
        calls["n"] += 1
        return _FakeResponse({"answers": {}, "usage": {"cost": COST}})

    monkeypatch.setattr(llm, "_post_decisions", fake_post)
    with pytest.raises(llm.DecisionsValidationError):
        llm.call_decisions(
            MODEL,
            _state(),
            QUESTIONS,
            service=SERVICE,
            max_retries=4,
            retry_base_delay=0.0,
        )
    assert calls["n"] == 1


@pytest.mark.parametrize(
    "fault",
    [
        httpx.ReadTimeout(""),
        httpx.ConnectTimeout(""),
        httpx.ConnectError(""),
        httpx.ReadError(""),
        httpx.RemoteProtocolError(""),
        TimeoutError(),
        ConnectionError(),
    ],
)
def test_a_transport_fault_is_retryable_however_bare_its_message(fault):
    """The transport types are retried on their type, whatever they say."""
    assert llm._decisions_retryable(fault) is True


def test_a_validation_error_is_never_retryable_whatever_its_message_says():
    """A deterministic refusal stays non-retryable even when its words match a
    retryable pattern, so the decided property is the type and not the text.
    """
    error = llm.DecisionsValidationError("timeout: the answer was refused")
    assert llm._is_retryable(str(error)) is True, "the message alone would retry"
    assert llm._decisions_retryable(error) is False


def test_an_empty_message_alone_does_not_make_a_fault_retryable(monkeypatch):
    """An empty message is not what earns a retry; the fault's type is.

    A deterministic failure raised with no text keeps its single attempt, so
    the retry above cannot be read as "retry anything that says nothing".
    """
    calls = {"n": 0}

    class _Refused(RuntimeError):
        """A refusal carrying no message of its own."""

    def fake_post(headers, body, timeout):
        calls["n"] += 1
        raise _Refused("")

    monkeypatch.setattr(llm, "_post_decisions", fake_post)
    with pytest.raises(_Refused):
        llm.call_decisions(
            MODEL,
            _state(),
            QUESTIONS,
            service=SERVICE,
            max_retries=4,
            retry_base_delay=0.0,
        )
    assert calls["n"] == 1
