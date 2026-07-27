"""Fan-out settings defaults and the proposer-prompt catalog-version hash."""

from __future__ import annotations

import shutil

from imas_codex.standard_names.fanout import config as fanout_config
from imas_codex.standard_names.fanout.config import (
    FanoutSettings,
    load_settings,
    render_proposer_system_prompt,
)


class TestFanoutSettings:
    def test_defaults(self) -> None:
        s = FanoutSettings()
        assert s.enabled is False
        assert s.max_fan_degree == 3
        assert s.function_timeout_s == 5.0
        assert s.total_timeout_s == 12.0
        assert s.result_hit_cap == 8
        assert s.evidence_token_cap_baseline == 2000
        assert s.evidence_token_cap_escalation == 800
        # Tier helpers.
        assert (
            s.cap_for_charge(escalate=False) == s.fanout_max_charge_per_cycle_baseline
        )
        assert (
            s.cap_for_charge(escalate=True) == s.fanout_max_charge_per_cycle_escalation
        )
        assert s.evidence_token_cap_for(escalate=False) == 2000
        assert s.evidence_token_cap_for(escalate=True) == 800
        assert s.cost_estimate_for(escalate=False) == s.fanout_cost_estimate_baseline
        assert s.cost_estimate_for(escalate=True) == s.fanout_cost_estimate_escalation

    def test_load_settings_from_pyproject(self) -> None:
        # The shipped pyproject.toml has the section in place — load
        # it and confirm the shipped configuration (enabled, refine_name on).
        s = load_settings()
        assert isinstance(s, FanoutSettings)
        assert s.enabled is True
        assert s.sites == {"refine_name": True}


class TestCatalogVersion:
    def test_first_line_contains_hash(self) -> None:
        prompt = render_proposer_system_prompt()
        first_line = prompt.split("\n", 1)[0]
        assert first_line.startswith("catalog_version=")
        assert len(first_line) == len("catalog_version=") + 64  # sha256 hex

    def test_hash_covers_body(self, tmp_path, monkeypatch) -> None:
        """Mutating the prompt body flips the hash.

        The mutation is applied to a COPY of the prompt tree in *tmp_path*, with
        the loader's ``PROMPTS_DIR`` redirected at it. Never write inside the
        source tree: this repo is shared by concurrent agents, so mutating the
        committed prompt and relying on a ``finally`` to restore it is unsafe
        two ways — two pytest runs race the mutate/restore window and one can
        restore the *other's* mutated text as "original", and a run killed
        mid-test leaves the live prompt corrupted with no cleanup at all.
        """
        from imas_codex.llm import prompt_loader

        original_hash = fanout_config._compute_catalog_version()

        # Copy the whole prompts tree so includes and siblings still resolve.
        prompts_copy = tmp_path / "prompts"
        shutil.copytree(prompt_loader.PROMPTS_DIR, prompts_copy)
        monkeypatch.setattr(prompt_loader, "PROMPTS_DIR", prompts_copy)
        # The config reads PROMPTS_DIR through the loader at call time, so the
        # redirect must be what the module under test actually resolves.
        assert fanout_config._prompt_path().is_relative_to(prompts_copy)

        copied = fanout_config._prompt_path()
        original_text = copied.read_text(encoding="utf-8")
        # Mutate help text below the frontmatter so it lands in the body.
        mutated = original_text.replace(
            "AT MOST 3", "AT MOST 3 (mutated for catalog-version test)"
        )
        assert mutated != original_text
        copied.write_text(mutated, encoding="utf-8")

        fanout_config._reset_catalog_version_cache()
        try:
            assert fanout_config._compute_catalog_version() != original_hash
        finally:
            # monkeypatch restores PROMPTS_DIR; the cache is module state and
            # must be cleared so later tests see the real prompt again.
            monkeypatch.undo()
            fanout_config._reset_catalog_version_cache()
        assert fanout_config._compute_catalog_version() == original_hash

    def test_hash_stable_across_calls(self) -> None:
        h1 = fanout_config._compute_catalog_version()
        h2 = fanout_config._compute_catalog_version()
        assert h1 == h2
