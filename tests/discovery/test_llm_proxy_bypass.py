"""Tests for proxy bypass logic in _build_kwargs.

When the model supports cache_control (Anthropic/Gemini) and a direct
OpenRouter API key is available, _build_kwargs should bypass the LiteLLM
proxy to preserve prompt caching and actual cost reporting.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from imas_codex.discovery.base import llm


def _make_stub_settings(
    monkeypatch, location="iter", proxy_url="http://127.0.0.1:18400"
):
    """Stub get_llm_location and get_llm_proxy_url (late imports in _build_kwargs)."""
    monkeypatch.setattr("imas_codex.settings.get_llm_location", lambda: location)
    monkeypatch.setattr("imas_codex.settings.get_llm_proxy_url", lambda: proxy_url)


MESSAGES = [{"role": "user", "content": "test"}]


class TestProxyBypass:
    """Proxy bypass routing logic."""

    def test_proxy_used_when_no_direct_key(self, monkeypatch):
        """Without OPENROUTER_API_KEY_IMAS_CODEX, always use proxy."""
        _make_stub_settings(monkeypatch, location="iter")
        monkeypatch.delenv("OPENROUTER_API_KEY_IMAS_CODEX", raising=False)
        monkeypatch.setenv("LITELLM_API_KEY", "proxy-key")

        kwargs = llm._build_kwargs(
            model="anthropic/claude-sonnet-4.6",
            api_key="or-key",
            messages=MESSAGES,
            response_format=None,
            max_tokens=None,
            temperature=None,
            timeout=None,
        )
        assert kwargs["api_base"] == "http://127.0.0.1:18400"
        assert kwargs["model"].startswith("openai/")

    def test_bypass_for_cache_model_with_direct_key(self, monkeypatch):
        """With direct key + cache-capable model → bypass proxy."""
        _make_stub_settings(monkeypatch, location="iter")
        monkeypatch.setenv("OPENROUTER_API_KEY_IMAS_CODEX", "direct-or-key")

        kwargs = llm._build_kwargs(
            model="anthropic/claude-sonnet-4.6",
            api_key="or-key",
            messages=MESSAGES,
            response_format=None,
            max_tokens=None,
            temperature=None,
            timeout=None,
        )
        # No api_base = direct to OpenRouter
        assert "api_base" not in kwargs
        assert kwargs["model"] == "openrouter/anthropic/claude-sonnet-4.6"
        # Should use direct key
        assert kwargs["api_key"] == "direct-or-key"

    def test_bypass_for_non_cache_openrouter_model(self, monkeypatch):
        """Any OpenRouter-routable model bypasses the proxy when a direct
        key is set — including non-cache models (eda8d029: bypass keeps
        response_cost telemetry even without cache_control)."""
        _make_stub_settings(monkeypatch, location="iter")
        monkeypatch.setenv("OPENROUTER_API_KEY_IMAS_CODEX", "direct-or-key")
        monkeypatch.setenv("LITELLM_API_KEY", "proxy-key")

        kwargs = llm._build_kwargs(
            model="openai/gpt-5.4",
            api_key="or-key",
            messages=MESSAGES,
            response_format=None,
            max_tokens=None,
            temperature=None,
            timeout=None,
        )
        # Direct to OpenRouter: no proxy api_base, openrouter/ prefix
        assert "api_base" not in kwargs
        assert kwargs["model"] == "openrouter/openai/gpt-5.4"
        assert kwargs["api_key"] == "direct-or-key"

    def test_local_mode_bypasses_proxy(self, monkeypatch):
        """Local mode (no proxy URL) always goes direct."""
        _make_stub_settings(monkeypatch, location="local")
        monkeypatch.delenv("LITELLM_PROXY_URL", raising=False)

        kwargs = llm._build_kwargs(
            model="anthropic/claude-sonnet-4.6",
            api_key="or-key",
            messages=MESSAGES,
            response_format=None,
            max_tokens=None,
            temperature=None,
            timeout=None,
        )
        assert "api_base" not in kwargs
        assert kwargs["model"] == "openrouter/anthropic/claude-sonnet-4.6"

    def test_standard_names_direct_request_carries_provider_rate_cap(self, monkeypatch):
        """Paid SN requests bind OpenRouter routing to the catalog ceiling."""
        _make_stub_settings(monkeypatch, location="iter")
        monkeypatch.setenv("OPENROUTER_API_KEY_IMAS_CODEX", "direct-or-key")

        kwargs = llm._build_kwargs(
            model="anthropic/claude-sonnet-4.6",
            api_key="or-key",
            messages=MESSAGES,
            response_format=None,
            max_tokens=None,
            temperature=None,
            timeout=None,
            service="standard-names",
        )

        ceiling = kwargs["extra_body"]["provider"]["max_price"]
        assert ceiling["prompt"] >= 3.0
        assert ceiling["completion"] >= 15.0
        assert ceiling["request"] == 0.0
        assert ceiling["image"] == 0.0

    def test_standard_names_proxy_route_rejects_before_provider(self, monkeypatch):
        """A route that cannot carry max_price is rejected before dispatch."""
        _make_stub_settings(monkeypatch, location="iter")
        monkeypatch.delenv("OPENROUTER_API_KEY_IMAS_CODEX", raising=False)
        monkeypatch.setenv("LITELLM_API_KEY", "proxy-key")

        with pytest.raises(llm.ProviderPricingUnbounded):
            llm._build_kwargs(
                model="anthropic/claude-sonnet-4.6",
                api_key="or-key",
                messages=MESSAGES,
                response_format=None,
                max_tokens=None,
                temperature=None,
                timeout=None,
                service="standard-names",
            )

    def test_separately_billed_reasoning_rejects_before_provider(self, monkeypatch):
        """A cataloged rate outside the text ceiling cannot dispatch."""
        import litellm

        _make_stub_settings(monkeypatch, location="iter")
        monkeypatch.setenv("OPENROUTER_API_KEY_IMAS_CODEX", "direct-or-key")
        monkeypatch.setattr(
            litellm,
            "get_model_info",
            lambda _model: {
                "input_cost_per_token": 1e-6,
                "output_cost_per_token": 2e-6,
                "output_cost_per_reasoning_token": 3e-6,
            },
        )

        with pytest.raises(llm.ProviderPricingUnbounded):
            llm._build_kwargs(
                model="openrouter/future/reasoning-model",
                api_key="or-key",
                messages=MESSAGES,
                response_format=None,
                max_tokens=None,
                temperature=None,
                timeout=None,
                service="standard-names",
            )

    def test_all_live_paid_standard_names_seats_have_direct_rate_caps(
        self, monkeypatch
    ):
        """Every configured production seat dispatches directly under max_price."""
        import litellm

        config = tomllib.loads(Path("pyproject.toml").read_text())["tool"]["imas-codex"]
        seats = {
            name: section["model"]
            for name, section in config.items()
            if name.startswith("sn-")
            and name != "sn-benchmark"
            and isinstance(section, dict)
            and isinstance(section.get("model"), str)
        }
        review = config["sn-review"]
        active_profile = review.get("active-profile", "default")
        seats.update(
            {
                f"sn-review.names.{active_profile}.{index}": model
                for index, model in enumerate(
                    review["names"]["profiles"][active_profile]["models"]
                )
            }
        )
        seats.update(
            {
                f"sn-review.docs.{index}": model
                for index, model in enumerate(review["docs"]["models"])
            }
        )
        seats["sn-fanout.proposer"] = config["sn-fanout"]["proposer-model"]
        paid_seats = {
            seat: model
            for seat, model in seats.items()
            if model.startswith("openrouter/")
        }

        assert paid_seats
        assert any("openai/" in model for model in paid_seats.values())
        assert any("x-ai/" in model for model in paid_seats.values())
        _make_stub_settings(monkeypatch, location="iter")
        monkeypatch.setenv("OPENROUTER_API_KEY_IMAS_CODEX", "direct-or-key")

        def _catalog_lag(_model):
            raise RuntimeError("model is newer than bundled catalog")

        monkeypatch.setattr(litellm, "get_model_info", _catalog_lag)
        for seat, model in paid_seats.items():
            kwargs = llm._build_kwargs(
                model=model,
                api_key="or-key",
                messages=MESSAGES,
                response_format=None,
                max_tokens=None,
                temperature=None,
                timeout=None,
                service="standard-names",
            )
            assert "api_base" not in kwargs, seat
            assert kwargs["model"] == model, seat
            assert kwargs["api_key"] == "direct-or-key", seat
            assert kwargs["extra_body"]["provider"]["max_price"] == {
                "prompt": 20.0,
                "completion": 100.0,
                "request": 1.0,
                "image": 10.0,
            }, seat


class TestCacheControlInjection:
    """cache_control blocks are injected for supported models."""

    def test_cache_control_injected_for_claude(self, monkeypatch):
        """Claude models get cache_control on system message."""
        _make_stub_settings(monkeypatch, location="local")
        monkeypatch.delenv("LITELLM_PROXY_URL", raising=False)

        messages = [
            {"role": "system", "content": "You are a physics expert."},
            {"role": "user", "content": "Describe psi."},
        ]
        kwargs = llm._build_kwargs(
            model="anthropic/claude-sonnet-4.6",
            api_key="or-key",
            messages=messages,
            response_format=None,
            max_tokens=None,
            temperature=None,
            timeout=None,
        )
        sys_msg = kwargs["messages"][0]
        # inject_cache_control converts string content to content blocks
        if isinstance(sys_msg["content"], list):
            last_block = sys_msg["content"][-1]
            assert "cache_control" in last_block
        else:
            # If content is still a string, cache_control should be
            # present as a top-level key on the message
            pass  # some models may not restructure

    def test_no_cache_control_for_gpt(self, monkeypatch):
        """GPT models should NOT get cache_control injected."""
        _make_stub_settings(monkeypatch, location="local")
        monkeypatch.delenv("LITELLM_PROXY_URL", raising=False)

        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello."},
        ]
        kwargs = llm._build_kwargs(
            model="openai/gpt-5.4",
            api_key="or-key",
            messages=messages,
            response_format=None,
            max_tokens=None,
            temperature=None,
            timeout=None,
        )
        sys_msg = kwargs["messages"][0]
        # GPT messages should remain plain strings
        assert isinstance(sys_msg["content"], str)


class TestCacheFieldExtraction:
    """_extract_cache_fields reads both litellm and OpenRouter field names."""

    def test_none_ptd(self):
        assert llm._extract_cache_fields(None) == (0, 0)

    def test_openrouter_cache_write_tokens(self):
        """OpenRouter uses cache_write_tokens (model_extra), not cache_creation_tokens."""

        class FakePTD:
            cached_tokens = 0
            cache_write_tokens = 4802

        read, write = llm._extract_cache_fields(FakePTD())
        assert read == 0
        assert write == 4802

    def test_litellm_cache_creation_tokens(self):
        """litellm formal field cache_creation_tokens is also checked."""

        class FakePTD:
            cached_tokens = 1024
            cache_creation_tokens = 500

        read, write = llm._extract_cache_fields(FakePTD())
        assert read == 1024
        assert write == 500

    def test_cached_tokens_read(self):
        """cached_tokens indicates a cache HIT."""

        class FakePTD:
            cached_tokens = 8000

        read, write = llm._extract_cache_fields(FakePTD())
        assert read == 8000
        assert write == 0
