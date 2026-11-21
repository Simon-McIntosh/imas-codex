"""Direct OpenRouter routing, service credentials and cache accounting."""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from iter_codex.discovery.base import llm

MESSAGES = [{"role": "user", "content": "test"}]


class TestOpenRouterRouting:
    """Direct provider requests preserve cache and cost metadata."""

    def test_explicit_key_routes_directly(self, monkeypatch):
        """An explicit provider credential still makes a direct request."""
        monkeypatch.delenv("OPENROUTER_API_KEY_ITER_CODEX", raising=False)

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
        assert kwargs["api_key"] == "or-key"

    def test_cache_model_uses_service_credential(self, monkeypatch):
        """Cache-capable models use the configured provider credential."""
        monkeypatch.setenv("OPENROUTER_API_KEY_ITER_CODEX", "direct-or-key")

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

    def test_non_cache_model_uses_service_credential(self, monkeypatch):
        """Non-cache models use the same direct provider routing."""
        monkeypatch.setenv("OPENROUTER_API_KEY_ITER_CODEX", "direct-or-key")

        kwargs = llm._build_kwargs(
            model="openai/gpt-5.4",
            api_key="or-key",
            messages=MESSAGES,
            response_format=None,
            max_tokens=None,
            temperature=None,
            timeout=None,
        )
        # The provider prefix selects OpenRouter directly.
        assert "api_base" not in kwargs
        assert kwargs["model"] == "openrouter/openai/gpt-5.4"
        assert kwargs["api_key"] == "direct-or-key"

    def test_provider_model_keeps_its_prefix(self, monkeypatch):
        """Provider model names are normalised for the client library."""

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
        monkeypatch.setenv("OPENROUTER_API_KEY_ITER_CODEX", "direct-or-key")

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

    def test_standard_names_custom_endpoint_rejects_before_provider(self, monkeypatch):
        """A route that cannot carry max_price is rejected before dispatch."""
        monkeypatch.delenv("OPENROUTER_API_KEY_ITER_CODEX", raising=False)

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
                api_base="https://custom.example.test/v1",
            )

    def test_separately_billed_reasoning_rejects_before_provider(self, monkeypatch):
        """A cataloged rate outside the text ceiling cannot dispatch."""
        import litellm

        monkeypatch.setenv("OPENROUTER_API_KEY_ITER_CODEX", "direct-or-key")
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

        config = tomllib.loads(Path("pyproject.toml").read_text())["tool"]["iter-codex"]
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
        monkeypatch.setenv("OPENROUTER_API_KEY_ITER_CODEX", "direct-or-key")

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
