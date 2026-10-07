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

    # Sections that intentionally use a LOCAL model (free, served on a
    # dedicated client) and are therefore EXEMPT from the openrouter/ prefix
    # guard: the locally routed compose and parent-enrichment seats, the
    # discovery function seats (text/vision, cluster labelling and IDS
    # mapping), plus the local embedding model.
    _LOCAL_MODEL_SECTIONS = frozenset(
        {
            "sn-compose",
            "sn-parent-enrich",
            "embedding",
            "discovery-triage",
            "discovery-score",
            "discovery-describe",
            "discovery-vision",
            "cluster-labels",
            "ids-mapping",
        }
    )

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
    assert settings.is_explicit_free_local_endpoint("local/deepseek-v4.1-flash")
    assert not settings.is_explicit_free_local_endpoint(
        "openrouter/openai/gpt-5.6-luna"
    )


def test_checked_in_pricing_zeroes_uncharged_dimensions_and_stays_inactive():
    """An undeclared per-request or per-image charge prices at zero.

    The arithmetic consumers multiply and sum these dimensions directly, so a
    route that charges neither must yield a number rather than ``None``.
    """
    model = "openrouter/openai/gpt-5.6-luna"

    pricing = settings.get_openrouter_pricing(model)

    assert pricing["request"] == 0.0
    assert pricing["image"] == 0.0


def test_model_sources_separate_route_seats_from_candidate_selection():
    fixed = settings.resolve_model_source("section:sn-compose")
    assert fixed.source_id == "section:sn-compose"
    assert fixed.model == settings.get_model("sn-compose")
    assert fixed.endpoint_class == "local-free"

    review_models = settings.get_model_source_models("sn-review:names")
    assert "local/deepseek-v4.1-flash" in review_models
    assert any(model.startswith("openrouter/") for model in review_models)
    with pytest.raises(ValueError, match="requires an explicit"):
        settings.resolve_model_source("sn-review:names")
    with pytest.raises(ValueError, match="outside source"):
        settings.resolve_model_source(
            "sn-review:names", candidate_model="openrouter/unregistered/model"
        )


def test_local_reviewer_source_binds_its_own_endpoint_contract():
    resolved = settings.resolve_model_source(
        "sn-review:names", candidate_model="local/deepseek-v4.1-flash"
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


class TestDiscoveryFunctionSeats:
    """Discovery reads one model seat per function.

    The discovery function seats run on the local lane through the ambix
    router: the four text/vision seats, cluster labelling and IDS mapping.
    The two Jev seats (discovery-relevance, mapping-candidates) stay on the
    OpenRouter decisions endpoint.
    """

    DISCOVERY_SEATS = (
        "discovery-triage",
        "discovery-score",
        "discovery-describe",
        "discovery-vision",
        "cluster-labels",
        "ids-mapping",
    )

    def test_seats_resolve_configured_models(self, monkeypatch):
        """Each seat resolves the model configured in pyproject.toml."""
        settings._load_pyproject_settings.cache_clear()

        for seat in self.DISCOVERY_SEATS:
            monkeypatch.delenv(settings._MODEL_ENV_VARS[seat], raising=False)
            assert settings.get_model(seat) == "local/deepseek-v4.1-flash"

    def test_seats_honour_environment_override(self, monkeypatch):
        """A seat's model is overridable through its environment variable."""
        settings._load_pyproject_settings.cache_clear()

        monkeypatch.setenv("IMAS_CODEX_DISCOVERY_SCORE_MODEL", "test-score-model")
        assert settings.get_model("discovery-score") == "test-score-model"

    def test_discovery_seats_register_local_endpoint(self):
        """The discovery seats bind their model to the ambix-local route."""
        settings._load_pyproject_settings.cache_clear()
        settings.register_model_endpoints()

        route_api_base = settings._get_section("model-routes")["ambix-local"][
            "api-base"
        ]
        for seat in self.DISCOVERY_SEATS:
            config = settings.get_model_config(seat)
            assert config["api_base"] == route_api_base, (
                f"{seat} is not routed to ambix-local"
            )
            assert config["api_key_env"] == "AMBIX_API_KEY"

            model = settings.get_model(seat)
            endpoint = settings.get_model_endpoint(model)
            assert endpoint is not None, f"{seat} registered no endpoint"
            assert endpoint["api_base"] == route_api_base
            assert endpoint["endpoint_class"] == "local-free"
            assert settings.is_explicit_free_local_endpoint(model)

    def test_ids_mapping_carries_high_reasoning_effort(self):
        """IDS mapping raises reasoning effort for the escalated choice."""
        settings._load_pyproject_settings.cache_clear()
        assert settings.get_reasoning_effort("ids-mapping") == "high"


class TestMapRunModelSeat:
    """The map run command's start-up model check probes the seat it calls.

    ``imas map run``'s LLM calls go through the ``ids-mapping`` seat, so its
    service monitor must probe that seat, not the unrelated ``reasoning``
    seat.
    """

    def test_map_run_model_check_names_ids_mapping(self, monkeypatch):
        from unittest.mock import MagicMock

        import imas_codex.cli.discover.common as common
        import imas_codex.cli.map as map_mod
        import imas_codex.discovery.base.facility as facility_mod
        import imas_codex.ids.progress as progress_mod

        captured: dict[str, object] = {}

        monkeypatch.setattr(
            facility_mod,
            "get_facility",
            lambda name: {"id": name, "wiki_sites": []},
        )
        monkeypatch.setattr(progress_mod, "MappingProgressDisplay", MagicMock())

        def _fake_run_discovery(config, async_main):
            captured["model_section"] = config.model_section
            captured["check_model"] = config.check_model
            return {}

        monkeypatch.setattr(common, "run_discovery", _fake_run_discovery)

        results = map_mod._run_rich_mode(
            facility="test",
            ids_names=["equilibrium"],
            targets=[{"ids_name": "equilibrium"}],
            model=None,
            dd_version=None,
            cost_limit=0.0,
            dry_run=True,
            no_activate=True,
            clear=False,
            deadline=None,
            verbose=False,
            console=None,
            log_print=lambda *a, **k: None,
        )

        assert captured["check_model"] is True
        assert captured["model_section"] == "ids-mapping"
        assert results == []
