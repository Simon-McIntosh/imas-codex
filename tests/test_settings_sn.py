"""Tests for SN-related settings accessors in imas_codex.settings.

Verifies default values from pyproject.toml and environment variable overrides
for the five example-injection and retry tunables added in Wave 2.
"""

from __future__ import annotations

import importlib

import pytest


@pytest.fixture(autouse=True)
def _clear_settings_cache():
    """Clear cached pyproject.toml settings between tests."""
    import imas_codex.settings as mod

    mod._load_pyproject_settings.cache_clear()
    yield
    mod._load_pyproject_settings.cache_clear()


# ── Default values (from pyproject.toml) ────────────────────────────────────


def test_example_target_scores_default():
    from imas_codex.settings import get_sn_example_target_scores

    result = get_sn_example_target_scores()
    assert isinstance(result, tuple)
    assert result == (1.0, 0.8, 0.65, 0.4)


def test_example_tolerance_default():
    from imas_codex.settings import get_sn_example_tolerance

    assert get_sn_example_tolerance() == pytest.approx(0.05)


def test_example_per_bucket_default():
    from imas_codex.settings import get_sn_example_per_bucket

    assert get_sn_example_per_bucket() == 1


def test_retry_attempts_default():
    from imas_codex.settings import get_sn_retry_attempts

    assert get_sn_retry_attempts() == 1


def test_retry_k_expansion_default():
    from imas_codex.settings import get_sn_retry_k_expansion

    assert get_sn_retry_k_expansion() == 12


# ── Environment variable overrides ──────────────────────────────────────────


def test_example_target_scores_env(monkeypatch):
    monkeypatch.setenv("IMAS_CODEX_SN_EXAMPLE_TARGET_SCORES", "0.9,0.7,0.5")
    import imas_codex.settings as mod

    importlib.reload(mod)
    result = mod.get_sn_example_target_scores()
    assert result == (0.9, 0.7, 0.5)


def test_example_tolerance_env(monkeypatch):
    monkeypatch.setenv("IMAS_CODEX_SN_EXAMPLE_TOLERANCE", "0.1")
    import imas_codex.settings as mod

    importlib.reload(mod)
    assert mod.get_sn_example_tolerance() == pytest.approx(0.1)


def test_example_per_bucket_env(monkeypatch):
    monkeypatch.setenv("IMAS_CODEX_SN_EXAMPLE_PER_BUCKET", "3")
    import imas_codex.settings as mod

    importlib.reload(mod)
    assert mod.get_sn_example_per_bucket() == 3


def test_retry_attempts_env(monkeypatch):
    monkeypatch.setenv("IMAS_CODEX_SN_RETRY_ATTEMPTS", "5")
    import imas_codex.settings as mod

    importlib.reload(mod)
    assert mod.get_sn_retry_attempts() == 5


def test_retry_k_expansion_env(monkeypatch):
    monkeypatch.setenv("IMAS_CODEX_SN_RETRY_K_EXPANSION", "20")
    import imas_codex.settings as mod

    importlib.reload(mod)
    assert mod.get_sn_retry_k_expansion() == 20


# ── Per-pool replica settings ([tool.imas-codex.sn-pools]) ──────────────────


def test_pool_replicas_default_from_pyproject():
    """Per-pool replica counts come from ``[tool.imas-codex.sn-pools]``.

    Generate pools are sized for the 2-GPU local vLLM profile (32); review
    pools hit OpenRouter (64) and refine pools carry a small backlog (32).
    """
    from imas_codex.settings import get_pool_replicas

    assert get_pool_replicas("generate_name") == 32
    assert get_pool_replicas("review_name") == 64
    assert get_pool_replicas("refine_name") == 32
    assert get_pool_replicas("generate_docs") == 32
    assert get_pool_replicas("review_docs") == 64
    assert get_pool_replicas("refine_docs") == 32


def test_pool_replicas_env_override(monkeypatch):
    """``IMAS_CODEX_SN_POOLS_<NAME>_REPLICAS`` overrides pyproject.toml.

    Operators raise the generate count to the 4-GPU profile (e.g. 96) without
    a code change via this env var.
    """
    monkeypatch.setenv("IMAS_CODEX_SN_POOLS_GENERATE_NAME_REPLICAS", "96")
    import imas_codex.settings as mod

    importlib.reload(mod)
    assert mod.get_pool_replicas("generate_name") == 96
    # Sibling pools unaffected.
    assert mod.get_pool_replicas("review_name") == 64


def test_pool_replicas_unknown_pool_raises():
    """Calling with an unknown pool name surfaces a ``ValueError``."""
    from imas_codex.settings import get_pool_replicas

    with pytest.raises(ValueError, match="Unknown SN pool name"):
        get_pool_replicas("nonexistent_pool")


def test_pool_replicas_fallback_to_compose_concurrency(monkeypatch):
    """When the ``[sn-pools]`` section is absent, replicas derive from
    ``[sn-compose].max-concurrency``: ``generate_*`` matches compose
    concurrency, ``review_*`` is half, ``refine_*`` is a quarter (each
    with a small floor for tiny configurations)."""
    import imas_codex.settings as mod

    # Force-empty sn-pools by stubbing the section accessor.
    monkeypatch.setattr(
        mod,
        "_get_section",
        lambda name: {"max-concurrency": 80} if name == "sn-compose" else {},
    )
    assert mod.get_pool_replicas("generate_name") == 80
    assert mod.get_pool_replicas("review_name") == 40
    assert mod.get_pool_replicas("refine_name") == 20
    # Tiny configurations clamp to the per-pool floor.
    monkeypatch.setattr(
        mod,
        "_get_section",
        lambda name: {"max-concurrency": 8} if name == "sn-compose" else {},
    )
    assert mod.get_pool_replicas("generate_name") == 8
    assert mod.get_pool_replicas("review_name") == 16  # floor
    assert mod.get_pool_replicas("refine_name") == 8  # floor
