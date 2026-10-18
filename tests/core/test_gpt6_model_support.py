"""Checked-in rates and production request shape for current OpenAI models."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from imas_codex import settings
from imas_codex.discovery.base import llm


class Answer(BaseModel):
    value: str


@pytest.mark.parametrize(
    ("model", "prompt", "completion", "cache_read", "cache_write", "long_completion"),
    [
        ("openrouter/openai/gpt-6.1-sol", 2.0, 10.0, 0.10, 2.50, 15.0),
        ("openrouter/openai/gpt-6-luna", 0.10, 0.50, 0.01, 0.125, 0.75),
    ],
)
def test_checked_in_price_resolves_with_long_context(
    model, prompt, completion, cache_read, cache_write, long_completion
):
    pricing = settings.get_openrouter_pricing(model)
    assert pricing["prompt"] == prompt
    assert pricing["completion"] == completion
    assert pricing["cache_read"] == cache_read
    assert pricing["cache_write"] == cache_write
    assert (
        pricing["source"]
        == f"https://openrouter.ai/api/v1/model/{model.removeprefix('openrouter/')}"
    )
    assert pricing["verified_at"]
    assert pricing["overrides"][0]["cache_read"] > cache_read
    assert pricing["overrides"][0]["cache_write"] > cache_write
    assert (
        llm.get_openrouter_expected_price(model, input_tokens=1000)["prompt"] == prompt
    )
    assert (
        llm.get_openrouter_expected_price(model, input_tokens=272000)["completion"]
        == long_completion
    )


@pytest.mark.parametrize(
    "model", ["openrouter/openai/gpt-6.1-sol", "openrouter/openai/gpt-6-luna"]
)
def test_reasoning_model_uses_direct_structured_request(model, monkeypatch):
    monkeypatch.setattr(settings, "get_llm_location", lambda: "remote")
    monkeypatch.setattr(settings, "get_llm_proxy_url", lambda: "http://proxy.test/v1")
    monkeypatch.setattr(settings, "get_model_endpoint", lambda _model: None)
    monkeypatch.setenv("OPENROUTER_API_KEY_IMAS_CODEX", "test-direct-key")
    request = llm._build_kwargs(
        model=model,
        api_key="fallback-key",
        messages=[{"role": "user", "content": "Return JSON with value yes."}],
        response_format=Answer,
        max_tokens=None,
        temperature=0.0,
        timeout=None,
        reasoning_effort="low",
    )
    assert request["model"] == model
    assert request["api_key"] == "test-direct-key"
    assert "api_base" not in request
    assert "temperature" not in request
    assert request["max_tokens"] == llm.MODEL_TOKEN_LIMITS["default"]["max_tokens"]
    assert request["response_format"]["type"] == "json_schema"
    assert request["extra_body"]["reasoning"] == {"effort": "low"}


def test_current_models_are_available_to_benchmark_defaults():
    compose = settings._SN_BENCHMARK_DEFAULTS["compose-models"]
    candidates = settings._SN_BENCHMARK_DEFAULTS["candidate-models"]
    for bare in ("openai/gpt-6.1-sol", "openai/gpt-6-luna"):
        assert bare in compose
        assert f"openrouter/{bare}" in candidates
