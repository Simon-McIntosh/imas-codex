"""Routing coverage for models served by the dedicated local OpenAI client.

A ``local/`` or ``hosted_vllm/`` model whose registered endpoint is local must
reach the dedicated local client on both the synchronous and asynchronous
paths; ``ollama/`` and ``openai/localhost`` keep litellm routing.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import BaseModel

from imas_codex.discovery.base import llm

LOCAL_MODEL = "local/deepseek-v4.1-flash"
LOCAL_ENDPOINT = "http://local.example.test/v1"


class Answer(BaseModel):
    value: str


def _fake_response(content: str) -> Any:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage=SimpleNamespace(
            prompt_tokens=5,
            completion_tokens=3,
            prompt_tokens_details=None,
        ),
    )


def _register_local_endpoint(monkeypatch, model: str = LOCAL_MODEL) -> None:
    from imas_codex import settings

    monkeypatch.setattr(
        settings,
        "_MODEL_ENDPOINTS",
        {
            model: {
                "api_base": LOCAL_ENDPOINT,
                "api_key_env": "TEST_LOCAL_API_KEY",
                "endpoint_class": "local-free",
            }
        },
    )
    monkeypatch.setenv("TEST_LOCAL_API_KEY", "local-key")
    monkeypatch.setenv("OPENROUTER_API_KEY_IMAS_CODEX", "test-or-key")


def test_local_client_prefixes_subset_of_local_model_prefixes():
    """The client set is drawn from the wider local-model set and stays narrower."""
    assert llm._LOCAL_CLIENT_MODEL_PREFIXES
    assert set(llm._LOCAL_CLIENT_MODEL_PREFIXES) <= set(llm._LOCAL_MODEL_PREFIXES)
    assert "ollama/" not in llm._LOCAL_CLIENT_MODEL_PREFIXES
    assert "openai/localhost" not in llm._LOCAL_CLIENT_MODEL_PREFIXES


@pytest.mark.parametrize("model", ["local/deepseek-v4.1-flash", "hosted_vllm/x"])
def test_predicate_true_for_local_client_models(model):
    assert llm._is_local_client_model(model)


@pytest.mark.parametrize(
    "model",
    [
        "ollama/llama3",
        "openai/localhost/v1",
        "openrouter/openai/gpt-5.4",
        "anthropic/claude-sonnet-4.6",
    ],
)
def test_predicate_false_for_litellm_routed_models(model):
    assert not llm._is_local_client_model(model)


def test_local_model_reaches_synchronous_local_client(monkeypatch):
    """call_llm_structured sends a local model to the sync local client."""
    _register_local_endpoint(monkeypatch)
    monkeypatch.setenv("LITELLM_PROXY_URL", "http://proxy.example.test/v1")

    clients: list[tuple[str, str | None]] = []
    received: dict[str, Any] = {}

    class Completions:
        def create(self, **kwargs):
            received.update(kwargs)
            return _fake_response('{"value": "ok"}')

    class Client:
        class Chat:
            completions = Completions()

        chat = Chat()

    def local_sync_client(api_base: str, api_key: str | None):
        clients.append((api_base, api_key))
        return Client()

    monkeypatch.setattr(llm, "_get_local_sync_client", local_sync_client)

    def litellm_must_not_run(**kwargs):  # pragma: no cover - guard
        raise AssertionError("litellm.completion used for a local model")

    monkeypatch.setattr("litellm.completion", litellm_must_not_run)

    result = llm.call_llm_structured(
        model=LOCAL_MODEL,
        messages=[{"role": "user", "content": "hi"}],
        response_model=Answer,
        max_tokens=16,
    )

    assert result.parsed == Answer(value="ok")
    assert clients == [(LOCAL_ENDPOINT, "local-key")]
    assert received["model"] == "deepseek-v4.1-flash"


def test_openrouter_model_reaches_litellm(monkeypatch):
    """An OpenRouter model keeps litellm routing on the synchronous path."""
    from imas_codex import settings

    monkeypatch.setattr(settings, "_MODEL_ENDPOINTS", {})
    monkeypatch.setenv("OPENROUTER_API_KEY_IMAS_CODEX", "test-or-key")

    called: dict[str, Any] = {}

    def fake_completion(**kwargs):
        called.update(kwargs)
        return _fake_response('{"value": "ok"}')

    monkeypatch.setattr("litellm.completion", fake_completion)

    def local_must_not_run(**kwargs):  # pragma: no cover - guard
        raise AssertionError("local client used for an OpenRouter model")

    monkeypatch.setattr(llm, "_completion_local", local_must_not_run)

    result = llm.call_llm_structured(
        model="anthropic/claude-sonnet-4.6",
        messages=[{"role": "user", "content": "hi"}],
        response_model=Answer,
        max_tokens=16,
    )

    assert result.parsed == Answer(value="ok")
    assert called["model"] == "openrouter/anthropic/claude-sonnet-4.6"


def test_local_client_request_carries_no_response_format(monkeypatch):
    """The local client is not asked to constrain the grammar; the schema is in the prompt."""
    _register_local_endpoint(monkeypatch)

    received: dict[str, Any] = {}

    class Completions:
        def create(self, **kwargs):
            received.update(kwargs)
            return _fake_response('{"value": "ok"}')

    class Client:
        class Chat:
            completions = Completions()

        chat = Chat()

    monkeypatch.setattr(
        llm, "_get_local_sync_client", lambda api_base, api_key: Client()
    )

    result = llm.call_llm_structured(
        model=LOCAL_MODEL,
        messages=[{"role": "user", "content": "hi"}],
        response_model=Answer,
        max_tokens=16,
    )

    assert result.parsed == Answer(value="ok")
    assert "response_format" not in received


def test_local_call_kwargs_drops_response_format():
    """The local request builder never forwards response_format."""
    out = llm._local_call_kwargs(
        {
            "model": LOCAL_MODEL,
            "messages": [{"role": "user", "content": "hi"}],
            "response_format": Answer,
            "max_tokens": 8,
        }
    )

    assert "response_format" not in out
    assert out["model"] == "deepseek-v4.1-flash"
    assert out["max_tokens"] == 8


def test_async_path_routes_local_model_to_async_client(monkeypatch):
    """The async path still sends a local model to the dedicated async client."""
    _register_local_endpoint(monkeypatch)

    clients: list[tuple[str, str | None]] = []
    received: dict[str, Any] = {}

    class Completions:
        async def create(self, **kwargs):
            received.update(kwargs)
            return _fake_response('{"value": "ok"}')

    class Client:
        class Chat:
            completions = Completions()

        chat = Chat()

    def local_client(api_base: str, api_key: str | None):
        clients.append((api_base, api_key))
        return Client()

    monkeypatch.setattr(llm, "_get_local_client", local_client)

    async def litellm_must_not_run(**kwargs):  # pragma: no cover - guard
        raise AssertionError("litellm.acompletion used for a local model")

    monkeypatch.setattr("litellm.acompletion", litellm_must_not_run)

    result = asyncio.run(
        llm.acall_llm_structured(
            model=LOCAL_MODEL,
            messages=[{"role": "user", "content": "hi"}],
            response_model=Answer,
            max_tokens=16,
        )
    )

    assert result.parsed == Answer(value="ok")
    assert clients == [(LOCAL_ENDPOINT, "local-key")]
    assert received["model"] == "deepseek-v4.1-flash"
