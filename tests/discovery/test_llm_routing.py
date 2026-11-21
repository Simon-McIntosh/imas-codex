"""Regression coverage for endpoint-aware LLM request routing."""

from __future__ import annotations

import asyncio
from typing import Any


def test_registered_local_endpoint_uses_bare_served_model(monkeypatch):
    """A configured local endpoint keeps its served model identifier intact."""
    from iter_codex import settings
    from iter_codex.discovery.base import llm

    model = "local/deepseek-v4-flash"
    served_model = "deepseek-v4-flash"
    endpoint = "http://local.example.test/v1"
    monkeypatch.setattr(
        settings,
        "_MODEL_ENDPOINTS",
        {
            model: {
                "api_base": endpoint,
                "api_key_env": "TEST_LOCAL_API_KEY",
                "endpoint_class": "local-free",
            }
        },
    )
    monkeypatch.setenv("TEST_LOCAL_API_KEY", "local-key")

    request = llm._build_kwargs(
        model=model,
        api_key="fallback-key",
        messages=[{"role": "user", "content": "Reply with one word."}],
        response_format=None,
        max_tokens=1,
        temperature=None,
        timeout=10,
    )

    assert request["api_base"] == endpoint
    assert request["model"] == model
    assert request["api_key"] == "local-key"

    received: dict[str, Any] = {}
    clients: list[tuple[str, str | None]] = []

    class Completions:
        async def create(self, **kwargs):
            received.update(kwargs)
            return object()

    class Client:
        class Chat:
            completions = Completions()

        chat = Chat()

    def local_client(api_base: str, api_key: str | None):
        clients.append((api_base, api_key))
        return Client()

    monkeypatch.setattr(llm, "_get_local_client", local_client)
    asyncio.run(llm._acompletion_local(request))

    assert clients == [(endpoint, "local-key")]
    assert received["model"] == served_model


def test_unregistered_openrouter_model_routes_directly(monkeypatch):
    """Provider routing does not depend on a custom endpoint registration."""
    from iter_codex import settings
    from iter_codex.discovery.base import llm

    monkeypatch.setattr(settings, "_MODEL_ENDPOINTS", {})
    monkeypatch.delenv("OPENROUTER_API_KEY_ITER_CODEX", raising=False)
    request = llm._build_kwargs(
        model="anthropic/claude-sonnet-4.6",
        api_key="explicit-key",
        messages=[{"role": "user", "content": "Reply with one word."}],
        response_format=None,
        max_tokens=1,
        temperature=None,
        timeout=10,
    )
    assert "api_base" not in request
    assert request["api_key"] == "explicit-key"
    assert request["model"] == "openrouter/anthropic/claude-sonnet-4.6"
