"""An engine-aborted reply is a transport failure, not content.

The local endpoint streams a partial prefix and sets ``finish_reason='abort'``
when it stops generation. That prefix must be rejected as an empty reply — the
retry loop then re-requests — rather than handed to the JSON parser or accepted
as an answer. Only ``finish_reason='length'`` grows the token budget.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import BaseModel

from imas_codex.discovery.base import llm

LOCAL_MODEL = "local/deepseek-v4.1-flash"
LOCAL_ENDPOINT = "http://local.example.test/v1"


class Answer(BaseModel):
    value: str


def _response(content: str | None, finish_reason: str | None) -> Any:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content),
                finish_reason=finish_reason,
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=5,
            completion_tokens=3,
            prompt_tokens_details=None,
        ),
    )


def _register_local_endpoint(monkeypatch) -> None:
    from imas_codex import settings

    monkeypatch.setattr(
        settings,
        "_MODEL_ENDPOINTS",
        {
            LOCAL_MODEL: {
                "api_base": LOCAL_ENDPOINT,
                "api_key_env": "TEST_LOCAL_API_KEY",
                "endpoint_class": "local-free",
            }
        },
    )
    monkeypatch.setenv("TEST_LOCAL_API_KEY", "local-key")


def _install_client(monkeypatch, responses: list[Any]) -> list[dict[str, Any]]:
    """Return the list of request kwargs the local client receives."""
    received: list[dict[str, Any]] = []
    queue = list(responses)

    class Completions:
        def create(self, **kwargs):
            received.append(kwargs)
            return queue.pop(0)

    class Client:
        class Chat:
            completions = Completions()

        chat = Chat()

    monkeypatch.setattr(
        llm, "_get_local_sync_client", lambda api_base, api_key: Client()
    )
    return received


def test_aborted_reply_with_partial_content_raises_empty_response_error(monkeypatch):
    """A truncated aborted prefix is rejected as an empty reply, not parsed."""
    _register_local_endpoint(monkeypatch)
    _install_client(monkeypatch, [_response('{"value": "partial', "abort")])

    with pytest.raises(Exception) as excinfo:
        llm.call_llm_structured(
            model=LOCAL_MODEL,
            messages=[{"role": "user", "content": "hi"}],
            response_model=Answer,
            max_tokens=16,
            max_retries=1,
            retry_base_delay=0.0,
        )

    cause = excinfo.value.__cause__
    assert isinstance(cause, llm.EmptyResponseError)
    assert cause.finish_reason == "abort"
    assert "empty response content" in str(cause)


def test_aborted_reply_is_retried_without_growing_the_token_budget(monkeypatch):
    """The abort reaches the retry loop; the budget is unchanged across attempts."""
    _register_local_endpoint(monkeypatch)
    received = _install_client(
        monkeypatch,
        [
            _response('{"value": "partial', "abort"),
            _response('{"value": "ok"}', "stop"),
        ],
    )

    result = llm.call_llm_structured(
        model=LOCAL_MODEL,
        messages=[{"role": "user", "content": "hi"}],
        response_model=Answer,
        max_tokens=16,
        max_retries=2,
        retry_base_delay=0.0,
    )

    assert result.parsed == Answer(value="ok")
    assert len(received) == 2
    assert received[0]["max_tokens"] == received[1]["max_tokens"] == 16


def test_invalid_local_reply_is_still_rejected(monkeypatch):
    """A non-aborted reply that is not valid JSON still fails the parse."""
    _register_local_endpoint(monkeypatch)
    _install_client(monkeypatch, [_response("not json at all", "stop")])

    with pytest.raises(Exception) as excinfo:
        llm.call_llm_structured(
            model=LOCAL_MODEL,
            messages=[{"role": "user", "content": "hi"}],
            response_model=Answer,
            max_tokens=16,
            max_retries=1,
            retry_base_delay=0.0,
        )

    assert not isinstance(excinfo.value, llm.EmptyResponseError)
