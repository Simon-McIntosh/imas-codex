"""Tests for settings.py module."""

import hashlib
import json
from datetime import UTC, datetime

import pytest

from imas_codex import settings
from imas_codex.settings import _parse_bool


class TestSettingsFunctions:
    """Tests for settings module functions."""

    def test_get_embedding_model_env_override(self, monkeypatch):
        """Environment variable overrides embedding model setting."""
        settings._load_pyproject_settings.cache_clear()

        monkeypatch.setenv("IMAS_CODEX_EMBEDDING_MODEL", "test-model")
        result = settings.get_embedding_model()

        assert result == "test-model"

    def test_get_model_language_env_override(self, monkeypatch):
        """Environment variable overrides language model setting."""
        settings._load_pyproject_settings.cache_clear()

        monkeypatch.setenv("IMAS_CODEX_LANGUAGE_MODEL", "test-llm")
        result = settings.get_model("language")

        assert result == "test-llm"

    def test_get_model_vision_env_override(self, monkeypatch):
        """Environment variable overrides vision model setting."""
        settings._load_pyproject_settings.cache_clear()

        monkeypatch.setenv("IMAS_CODEX_VISION_MODEL", "test-vlm")
        result = settings.get_model("vision")

        assert result == "test-vlm"

    def test_get_labeling_batch_size_env_override(self, monkeypatch):
        """Environment variable overrides labeling batch size."""
        settings._load_pyproject_settings.cache_clear()

        monkeypatch.setenv("IMAS_CODEX_LABELING_BATCH_SIZE", "100")
        result = settings.get_labeling_batch_size()

        assert result == 100

    def test_get_include_ggd_env_override(self, monkeypatch):
        """Environment variable overrides include_ggd setting."""
        settings._load_pyproject_settings.cache_clear()

        monkeypatch.setenv("IMAS_CODEX_INCLUDE_GGD", "false")
        result = settings.get_include_ggd()

        assert result is False

    def test_get_include_error_fields_env_override(self, monkeypatch):
        """Environment variable overrides include_error_fields setting."""
        settings._load_pyproject_settings.cache_clear()

        monkeypatch.setenv("IMAS_CODEX_INCLUDE_ERROR_FIELDS", "true")
        result = settings.get_include_error_fields()

        assert result is True

    def test_get_dd_version_env_override(self, monkeypatch):
        """Environment variable overrides DD version."""
        settings._load_pyproject_settings.cache_clear()

        monkeypatch.setenv("IMAS_DD_VERSION", "3.99.0")
        result = settings.get_dd_version()

        assert result == "3.99.0"

    def test_get_embedding_model_default(self, monkeypatch):
        """get_embedding_model returns default when env not set."""
        settings._load_pyproject_settings.cache_clear()

        monkeypatch.delenv("IMAS_CODEX_EMBEDDING_MODEL", raising=False)
        result = settings.get_embedding_model()

        assert isinstance(result, str)
        assert len(result) > 0

    def test_get_model_language_default(self, monkeypatch):
        """get_model('language') returns default when env not set."""
        settings._load_pyproject_settings.cache_clear()

        monkeypatch.delenv("IMAS_CODEX_LANGUAGE_MODEL", raising=False)
        result = settings.get_model("language")

        assert isinstance(result, str)
        assert len(result) > 0

    # Sections that intentionally use a LOCAL model (free, served on a
    # dedicated client) and are therefore EXEMPT from the openrouter/ prefix
    # guard: sn-compose (hosted_vllm DeepSeek-V4) and embedding (local Qwen).
    _LOCAL_MODEL_SECTIONS = frozenset({"sn-compose", "embedding"})

    @pytest.mark.parametrize(
        "section",
        sorted(set(settings._MODEL_ENV_VARS) - _LOCAL_MODEL_SECTIONS),
    )
    def test_openrouter_prefix_present(self, monkeypatch, section):
        """OpenRouter-billed sections must carry the 'openrouter/' prefix.

        Without it, calls silently route through the LiteLLM proxy, which
        strips cache_control breakpoints (~80% cache discount lost) and
        zeroes response_cost (cost telemetry broken). Regression guard.

        Derived from ``_MODEL_ENV_VARS`` (minus the local-model sections) so
        the guard auto-covers new sections and cannot rot — the previous
        static list silently referenced a non-existent ``sn-enrich`` section.
        """
        settings._load_pyproject_settings.cache_clear()
        env_var = settings._MODEL_ENV_VARS[section]
        monkeypatch.delenv(env_var, raising=False)

        result = settings.get_model(section)
        assert result.startswith("openrouter/"), (
            f"[{section}] model='{result}' missing openrouter/ prefix — "
            "this re-enables proxy routing which strips cache_control."
        )

    def test_get_labeling_batch_size_default(self, monkeypatch):
        """get_labeling_batch_size returns default when env not set."""
        settings._load_pyproject_settings.cache_clear()

        monkeypatch.delenv("IMAS_CODEX_LABELING_BATCH_SIZE", raising=False)
        result = settings.get_labeling_batch_size()

        assert isinstance(result, int)
        assert result > 0

    def test_get_include_ggd_default(self, monkeypatch):
        """get_include_ggd returns default when env not set."""
        settings._load_pyproject_settings.cache_clear()

        monkeypatch.delenv("IMAS_CODEX_INCLUDE_GGD", raising=False)
        result = settings.get_include_ggd()

        assert isinstance(result, bool)

    def test_get_include_error_fields_default(self, monkeypatch):
        """get_include_error_fields returns default when env not set."""
        settings._load_pyproject_settings.cache_clear()

        monkeypatch.delenv("IMAS_CODEX_INCLUDE_ERROR_FIELDS", raising=False)
        result = settings.get_include_error_fields()

        assert isinstance(result, bool)


class TestGetModel:
    """Tests for unified get_model(section) function."""

    def test_language_section_returns_model(self):
        """Language section returns a model string."""
        model = settings.get_model("language")
        assert isinstance(model, str)
        assert "/" in model

    def test_vision_section_returns_model(self):
        """Vision section returns a model string."""
        model = settings.get_model("vision")
        assert isinstance(model, str)
        assert "/" in model

    def test_embedding_section_returns_model(self):
        """Embedding section returns a model string."""
        model = settings.get_model("embedding")
        assert isinstance(model, str)
        assert len(model) > 0

    def test_unknown_section_raises(self):
        """Unknown section raises ValueError."""
        with pytest.raises(ValueError, match="Unknown model section"):
            settings.get_model("nonexistent_section")


class TestParseBool:
    """Tests for the _parse_bool helper function."""

    def test_true_string_values(self):
        """True string values are parsed correctly."""
        assert _parse_bool("true") is True
        assert _parse_bool("True") is True
        assert _parse_bool("TRUE") is True
        assert _parse_bool("1") is True
        assert _parse_bool("yes") is True

    def test_false_string_values(self):
        """False string values are parsed correctly."""
        assert _parse_bool("false") is False
        assert _parse_bool("0") is False
        assert _parse_bool("no") is False

    def test_bool_values_pass_through(self):
        """Boolean values pass through unchanged."""
        assert _parse_bool(True) is True
        assert _parse_bool(False) is False


class TestModuleLevelConstants:
    """Tests for module-level constants."""

    def test_module_constants_exist(self):
        """Module-level constants are defined."""
        assert hasattr(settings, "LABELING_BATCH_SIZE")
        assert hasattr(settings, "INCLUDE_GGD")
        assert hasattr(settings, "INCLUDE_ERROR_FIELDS")
        assert hasattr(settings, "EMBEDDING_DIMENSION")

    def test_module_constants_have_correct_types(self):
        """Module-level constants have correct types."""
        assert isinstance(settings.LABELING_BATCH_SIZE, int)
        assert isinstance(settings.INCLUDE_GGD, bool)
        assert isinstance(settings.INCLUDE_ERROR_FIELDS, bool)
        assert isinstance(settings.EMBEDDING_DIMENSION, int)


def test_free_local_endpoint_requires_explicit_trusted_classification():
    assert settings.is_explicit_free_local_endpoint("hosted_vllm/deepseek-v4-flash")
    assert not settings.is_explicit_free_local_endpoint(
        "openrouter/openai/gpt-5.6-luna"
    )


def test_checked_in_pricing_preserves_missing_dimensions_and_stays_inactive():
    model = "openrouter/openai/gpt-5.6-luna"

    pricing = settings.get_openrouter_pricing(model)

    assert pricing["request"] is None
    assert pricing["image"] is None
    with pytest.raises(settings.PricingAuthorityError):
        settings.get_typed_openrouter_pricing(model)


def _typed_pricing_authority(*, require_image: bool) -> dict[str, object]:
    model = "openrouter/openai/example-alias"
    canonical = "openai/example-canonical"
    model_pricing = {
        "prompt": "0.0000001",
        "completion": "0.0000006",
        "request": "0.01",
        "image": "0.02",
    }
    architecture = {
        "input_modalities": ["text", "image"],
        "output_modalities": ["text"],
    }
    model_payload = json.dumps(
        {
            "data": {
                "id": canonical,
                "architecture": architecture,
                "pricing": model_pricing,
            }
        },
        separators=(",", ":"),
    )
    endpoint_pricing = dict(model_pricing)
    endpoints_payload = json.dumps(
        {
            "data": {
                "id": canonical,
                "endpoints": [
                    {
                        "name": "OpenAI/standard",
                        "provider_name": "OpenAI",
                        "pricing": endpoint_pricing,
                    }
                ],
            }
        },
        separators=(",", ":"),
    )
    raw: dict[str, object] = {
        "prompt": 0.1,
        "completion": 0.6,
        "request": 0.01,
        "image": 0.02 if require_image else None,
        "cache_read": None,
        "cache_write": None,
        "cache_write_ttl": None,
        "image_unit": "per-image" if require_image else None,
        "canonical_slug": canonical,
        "provider": "OpenAI",
        "provider_selector": "OpenAI/standard",
        "source": "https://openrouter.ai/api/v1/model/openai/example-alias",
        "endpoints_source": (
            "https://openrouter.ai/api/v1/models/openai/example-canonical/endpoints"
        ),
        "retrieved_at": "2026-08-10T00:00:00Z",
        "model_payload_json": model_payload,
        "endpoints_payload_json": endpoints_payload,
        "model_payload_sha256": hashlib.sha256(model_payload.encode()).hexdigest(),
        "endpoints_payload_sha256": hashlib.sha256(
            endpoints_payload.encode()
        ).hexdigest(),
        "other_charged_dimensions": [],
        "overrides": [],
    }
    required = (
        ["completion", "image", "prompt", "request"]
        if require_image
        else [
            "completion",
            "prompt",
            "request",
        ]
    )
    projection = {
        "configured_alias": model,
        "canonical_slug": canonical,
        "canonical_wire_model": f"openrouter/{canonical}",
        "provider": "OpenAI",
        "provider_selector": "OpenAI/standard",
        "source": raw["source"],
        "endpoints_source": raw["endpoints_source"],
        "retrieved_at": "2026-08-10T00:00:00+00:00",
        "model_payload_sha256": raw["model_payload_sha256"],
        "endpoints_payload_sha256": raw["endpoints_payload_sha256"],
        "architecture": architecture,
        "model_pricing": model_pricing,
        "provider_endpoint": {
            "name": "OpenAI/standard",
            "provider_name": "OpenAI",
            "pricing": endpoint_pricing,
        },
        "completion": 0.6,
        **({"image": 0.02} if require_image else {}),
        "prompt": 0.1,
        "request": 0.01,
        "required_dimensions": required,
        "image_unit": "per-image" if require_image else None,
        "cache_control": "disabled",
        "other_charged_dimensions": [],
        "overrides": [],
    }
    raw["canonical_projection_sha256"] = hashlib.sha256(
        settings._canonical_payload_bytes(projection)
    ).hexdigest()
    return raw


def test_typed_pricing_recomputes_payloads_projection_and_exact_selector(monkeypatch):
    authority = _typed_pricing_authority(require_image=False)
    monkeypatch.setattr(settings, "get_openrouter_pricing", lambda model: authority)

    pricing = settings.get_typed_openrouter_pricing(
        "openrouter/openai/example-alias",
        now=datetime(2026, 8, 10, 1, tzinfo=UTC),
    )

    assert pricing["canonical_wire_model"] == "openrouter/openai/example-canonical"
    assert pricing["provider_selector"] == "OpenAI/standard"
    assert pricing["required_dimensions"] == ["completion", "prompt", "request"]
    assert "image" not in pricing

    authority["provider"] = "unverified-provider"
    with pytest.raises(settings.PricingAuthorityError, match="provider identity"):
        settings.get_typed_openrouter_pricing(
            "openrouter/openai/example-alias",
            now=datetime(2026, 8, 10, 1, tzinfo=UTC),
        )


def test_typed_pricing_rejects_payload_tampering_and_requires_image_by_modality(
    monkeypatch,
):
    authority = _typed_pricing_authority(require_image=True)
    monkeypatch.setattr(settings, "get_openrouter_pricing", lambda model: authority)

    pricing = settings.get_typed_openrouter_pricing(
        "openrouter/openai/example-alias",
        require_image=True,
        now=datetime(2026, 8, 10, 1, tzinfo=UTC),
    )
    assert pricing["image"] == pytest.approx(0.02)

    authority["endpoints_payload_json"] = str(
        authority["endpoints_payload_json"]
    ).replace("0.02", "0.03")
    with pytest.raises(settings.PricingAuthorityError, match="payload_json receipt"):
        settings.get_typed_openrouter_pricing(
            "openrouter/openai/example-alias",
            require_image=True,
            now=datetime(2026, 8, 10, 1, tzinfo=UTC),
        )


def test_model_sources_separate_route_seats_from_candidate_selection():
    fixed = settings.resolve_model_source("section:sn-compose")
    assert fixed.source_id == "section:sn-compose"
    assert fixed.model == settings.get_model("sn-compose")
    assert fixed.endpoint_class == "local-free"

    review_models = settings.get_model_source_models("sn-review:names")
    assert "hosted_vllm/deepseek-v4-flash" in review_models
    assert any(model.startswith("openrouter/") for model in review_models)
    with pytest.raises(ValueError, match="requires an explicit"):
        settings.resolve_model_source("sn-review:names")
    with pytest.raises(ValueError, match="outside source"):
        settings.resolve_model_source(
            "sn-review:names", candidate_model="openrouter/unregistered/model"
        )


def test_local_reviewer_source_binds_its_own_endpoint_contract():
    resolved = settings.resolve_model_source(
        "sn-review:names", candidate_model="hosted_vllm/deepseek-v4-flash"
    )

    assert resolved.api_key_env == "AMBIX_API_KEY"
    assert resolved.api_base
    assert resolved.endpoint_class == "local-free"


class TestGraphSettings:
    """Tests for graph (Neo4j) settings accessors."""

    def test_get_graph_uri_default(self, monkeypatch):
        """get_graph_uri returns pyproject value or default."""
        settings._load_pyproject_settings.cache_clear()
        monkeypatch.delenv("NEO4J_URI", raising=False)
        result = settings.get_graph_uri()
        assert isinstance(result, str)
        assert result.startswith("bolt://")

    def test_get_graph_uri_env_override(self, monkeypatch):
        """NEO4J_URI env var overrides pyproject.toml."""
        settings._load_pyproject_settings.cache_clear()
        monkeypatch.setenv("NEO4J_URI", "bolt://remote-host:7687")
        result = settings.get_graph_uri()
        assert result == "bolt://remote-host:7687"

    def test_get_graph_username_default(self, monkeypatch):
        """get_graph_username returns pyproject value or default."""
        settings._load_pyproject_settings.cache_clear()
        monkeypatch.delenv("NEO4J_USERNAME", raising=False)
        result = settings.get_graph_username()
        assert isinstance(result, str)
        assert result == "neo4j"

    def test_get_graph_username_env_override(self, monkeypatch):
        """NEO4J_USERNAME env var overrides pyproject.toml."""
        settings._load_pyproject_settings.cache_clear()
        monkeypatch.setenv("NEO4J_USERNAME", "custom-user")
        result = settings.get_graph_username()
        assert result == "custom-user"

    def test_get_graph_password_default(self, monkeypatch):
        """get_graph_password returns pyproject value or default."""
        settings._load_pyproject_settings.cache_clear()
        monkeypatch.delenv("NEO4J_PASSWORD", raising=False)
        result = settings.get_graph_password()
        assert isinstance(result, str)
        assert result == "imas-codex"

    def test_get_graph_password_env_override(self, monkeypatch):
        """NEO4J_PASSWORD env var overrides pyproject.toml."""
        settings._load_pyproject_settings.cache_clear()
        monkeypatch.setenv("NEO4J_PASSWORD", "secret-pw")
        result = settings.get_graph_password()
        assert result == "secret-pw"

    def test_graph_settings_from_pyproject(self, monkeypatch):
        """Graph settings are read from pyproject.toml [tool.imas-codex.graph]."""
        settings._load_pyproject_settings.cache_clear()
        monkeypatch.delenv("NEO4J_URI", raising=False)
        monkeypatch.delenv("NEO4J_USERNAME", raising=False)
        monkeypatch.delenv("NEO4J_PASSWORD", raising=False)

        # These should resolve from pyproject.toml which has the graph section
        uri = settings.get_graph_uri()
        username = settings.get_graph_username()
        password = settings.get_graph_password()

        assert "bolt://" in uri
        assert username == "neo4j"
        assert password == "imas-codex"

    def test_get_graph_name_default(self, monkeypatch):
        """get_graph_name returns active graph name."""
        settings._load_pyproject_settings.cache_clear()
        monkeypatch.delenv("IMAS_CODEX_GRAPH", raising=False)
        name = settings.get_graph_name()
        # In CI/local environments without an initialized graph symlink,
        # get_active_graph_name() returns "uninitialized".
        assert name in {"codex", "uninitialized"}

    def test_get_graph_profile_returns_profile(self, monkeypatch):
        """get_graph_profile returns a Neo4jProfile object."""
        settings._load_pyproject_settings.cache_clear()
        monkeypatch.delenv("IMAS_CODEX_GRAPH", raising=False)
        monkeypatch.delenv("NEO4J_URI", raising=False)
        monkeypatch.delenv("NEO4J_PASSWORD", raising=False)
        profile = settings.get_graph_profile()
        assert profile.name in {"codex", "uninitialized"}
        assert profile.bolt_port == 7687
