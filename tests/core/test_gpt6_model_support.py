"""Checked-in rates and production request shape for current OpenAI models."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from iter_codex import settings
from iter_codex.discovery.base import llm


class Answer(BaseModel):
    value: str


@pytest.mark.parametrize(
    (
        "model",
        "prompt",
        "completion",
        "cache_read",
        "cache_write",
        "long_completion",
        "long_cache_read",
        "long_cache_write",
    ),
    [
        ("openrouter/openai/gpt-6.1-sol", 2.0, 10.0, 0.10, 2.50, 15.0, 0.20, 5.0),
        ("openrouter/openai/gpt-6-luna", 0.10, 0.50, 0.01, 0.125, 0.75, 0.02, 0.25),
    ],
)
def test_checked_in_price_resolves_with_long_context(
    model,
    prompt,
    completion,
    cache_read,
    cache_write,
    long_completion,
    long_cache_read,
    long_cache_write,
):
    expected_price = llm.get_openrouter_expected_price(model, input_tokens=1000)
    long_price = llm.get_openrouter_expected_price(model, input_tokens=272000)
    assert expected_price["prompt"] == prompt
    assert long_price["completion"] == long_completion

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
    assert pricing["overrides"][0]["cache_read"] == long_cache_read
    assert pricing["overrides"][0]["cache_write"] == long_cache_write


@pytest.mark.parametrize(
    "model", ["openrouter/openai/gpt-6.1-sol", "openrouter/openai/gpt-6-luna"]
)
def test_unpriced_model_is_refused_by_call_layer(model, monkeypatch):
    monkeypatch.setattr(settings, "get_openrouter_pricing", lambda _model: {})
    monkeypatch.setattr(llm, "_lookup_catalog", lambda *_args, **_kwargs: ({}, False))
    with pytest.raises(
        llm.ProviderPricingUnbounded, match="expected price unavailable"
    ):
        llm.get_openrouter_expected_price(model, input_tokens=1000)


@pytest.mark.parametrize(
    "model", ["openrouter/openai/gpt-6.1-sol", "openrouter/openai/gpt-6-luna"]
)
def test_reasoning_model_uses_direct_structured_request(model, monkeypatch):
    monkeypatch.setattr(settings, "get_model_endpoint", lambda _model: None)
    monkeypatch.setenv("OPENROUTER_API_KEY_ITER_CODEX", "test-direct-key")
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


def test_benchmark_slates_match_their_configured_models_and_prices():
    config = settings._get_section("sn-benchmark")
    for slate in ("compose-models", "candidate-models"):
        defaults = settings._SN_BENCHMARK_DEFAULTS[slate]
        assert defaults == config[slate], (
            f"{slate} fallback differs from its configuration"
        )
        assert all(settings.get_openrouter_pricing(model) for model in defaults), slate
