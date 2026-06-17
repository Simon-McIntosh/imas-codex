"""Compose-prompt guard tests for the field-choice rule set.

Tests that generate_name_system.md contains the required guard SIGNAL — the
distinct semantic rules that gate name composition — regardless of section
heading.  The prompt was consolidated into goal-first ``Field-Choice Rules``
and ``Output-Discipline Rules`` sections (the compose model emits IR segment
fields, not a string, so surface-syntax-only checks — adjacent-duplicate
tokens, character-length / ``_of_`` nesting limits — were dropped because the
model never emits the string they validated; the composer assembles it).
These are structural/content tests — they do not call the LLM.

A companion file ``test_compose_regex_guards.py`` provides pure-regex
validators for each anti-pattern.
"""

from __future__ import annotations

import pytest

from imas_codex.llm.prompt_loader import PROMPTS_DIR

# =====================================================================
# Helpers
# =====================================================================


def _load_compose_system_raw() -> str:
    """Return the fully rendered generate_name_system.md content.

    Uses ``render_prompt`` with the compose context so that ``{% include %}``
    directives are resolved and Jinja2 template variables (e.g.
    ``composition_rules`` for NC rules) are substituted.  This gives the
    same content that reaches the LLM at runtime.
    """
    from imas_codex.llm.prompt_loader import render_prompt
    from imas_codex.standard_names.context import build_compose_context

    return render_prompt("sn/generate_name_system", build_compose_context())


# =====================================================================
# Tests — prompt content assertions
# =====================================================================


class TestFieldChoiceGuardSignal:
    """Verify every distinct semantic guard SIGNAL survives the consolidation.

    The prompt no longer numbers ten ``HARD PRE-EMIT CHECKS``; the rules were
    folded into the field-choice / output-discipline sections.  Each test below
    asserts the surviving signal, not the old heading.
    """

    @pytest.fixture(autouse=True)
    def _load(self) -> None:
        self.raw = _load_compose_system_raw()

    def test_locus_preposition_semantic(self) -> None:
        # of/at/over is taught as the locus_relation field choice.
        assert "locus_relation" in self.raw
        assert "defining attribute" in self.raw
        assert "magnetic_axis" in self.raw

    def test_hardware_tokens_postfix_locus(self) -> None:
        assert "postfix" in self.raw and "locus" in self.raw
        assert "probe" in self.raw
        assert "sensor" in self.raw

    def test_provenance_prefixes_banned(self) -> None:
        assert "initial_" in self.raw
        assert "launched_" in self.raw
        assert "post_crash_" in self.raw

    def test_invented_bases_to_vocab_gap(self) -> None:
        assert "base_token" in self.raw and "registered" in self.raw

    def test_abbreviations_rejected(self) -> None:
        assert "No abbreviations, acronyms, alphanumerics" in self.raw
        assert "3db" in self.raw

    def test_one_subject_rule(self) -> None:
        assert "Exactly one subject" in self.raw
        assert "hydrogen_ion" in self.raw

    def test_us_spelling_rule(self) -> None:
        assert "US spelling only" in self.raw
        assert "ionisation" in self.raw

    def test_structural_leakage_banned(self) -> None:
        # Structural / data-model leakage tokens are forbidden anywhere.
        assert "obtained_from" in self.raw
        assert "stored_in" in self.raw
        assert "derived_from" in self.raw


class TestRejectListExpansion:
    """Verify the REJECT list was expanded with Report 7 patterns."""

    @pytest.fixture(autouse=True)
    def _load(self) -> None:
        self.raw = _load_compose_system_raw()

    def test_bandwidth_3db_rejected(self) -> None:
        assert "bandwidth_3db" in self.raw

    def test_turn_count_rejected(self) -> None:
        assert "turn_count" in self.raw

    def test_nuclear_charge_number_rejected(self) -> None:
        assert "nuclear_charge_number" in self.raw

    def test_azimuth_angle_rejected(self) -> None:
        assert "azimuth_angle" in self.raw

    def test_distance_between_rejected(self) -> None:
        assert "distance_between_" in self.raw


class TestChannelGuidance:
    """Verify NC-32 channel guidance block is present."""

    @pytest.fixture(autouse=True)
    def _load(self) -> None:
        self.raw = _load_compose_system_raw()

    def test_nc32_heading_present(self) -> None:
        assert "NC-32" in self.raw

    def test_channel_path_pattern_mentioned(self) -> None:
        assert "*/channel/*" in self.raw

    def test_observable_examples(self) -> None:
        assert "faraday_rotation_angle" in self.raw
        assert "line_integrated_electron_density" in self.raw

    def test_diagnostic_examples(self) -> None:
        assert "polarimeter" in self.raw
        assert "interferometer" in self.raw
        assert "thomson_scattering" in self.raw
        assert "refractometer" in self.raw

    def test_anti_pattern_examples(self) -> None:
        assert "polarimeter_channel_angle" in self.raw
        assert "interferometer_channel_density" in self.raw


class TestRulePlacement:
    """Field-choice rules must appear after the prelude includes and before the
    curated-examples / dynamic blocks, so the static cacheable prefix is large."""

    @pytest.fixture(autouse=True)
    def _load(self) -> None:
        self.raw = _load_compose_system_raw()

    def test_rules_before_curated_examples(self) -> None:
        rules_pos = self.raw.index("## Field-Choice Rules")
        examples_pos = self.raw.index("## Curated Examples")
        assert rules_pos < examples_pos

    def test_rules_after_includes(self) -> None:
        # Field-choice rules must appear after the prelude includes (grammar,
        # exemplars). _load_compose_system_raw returns rendered content, so we
        # anchor on a distinctive phrase from the end of _exemplars_name_only.md.
        prelude_anchor = "Does the name describe a physical quantity"
        prelude_end = self.raw.index(prelude_anchor) + len(prelude_anchor)
        rules_pos = self.raw.index("## Field-Choice Rules")
        assert rules_pos > prelude_end


class TestInverseProblemRoleGuard:
    """Inverse-problem role wrappers (formerly 'CONSTRAINT ROLE ABSTRACTION')
    must still be routed to skip with the base physical quantity kept."""

    @pytest.fixture(autouse=True)
    def _load(self) -> None:
        self.raw = _load_compose_system_raw()

    def test_constraint_role_signal_present(self) -> None:
        assert "_constraint_weight" in self.raw
        assert "_constraint_measured_value" in self.raw
        assert "inverse_problem_role" in self.raw


class TestNonNameableAndMissingBaseGuidance:
    """Compose prompts steer non-nameable coordinates to skip and true
    missing-base concepts to vocab_gap — the two beta-rotation exhaustion
    classes (``time``/``delay`` and the phase/angle/extent base gaps)."""

    @pytest.fixture(autouse=True)
    def _load(self) -> None:
        self.system = _load_compose_system_raw()
        self.user = (PROMPTS_DIR / "sn" / "generate_name_dd.md").read_text(
            encoding="utf-8"
        )

    def test_system_has_non_nameable_skip_section(self) -> None:
        # Renamed heading: "When NOT to name — route to `skipped`".
        assert "route to `skipped`" in self.system
        # The infra classes that exhausted in beta rotation are called out.
        assert "latency" in self.system
        assert "timestamp" in self.system or "time_stamp" in self.system

    def test_system_routes_missing_base_to_vocab_gap(self) -> None:
        # Renamed heading: "When the base is missing — emit a clean `vocab_gap`".
        assert "base is missing" in self.system
        # Phase / angle base-gap concepts are explicitly named.
        assert "phase shift" in self.system or "phase_shift" in self.system
        assert "vocab_gap" in self.system

    def test_user_prompt_has_non_nameable_skip_table(self) -> None:
        assert "Non-Nameable Paths" in self.user
        assert "real_time_data/topic/time_stamp" in self.user
        assert "bremsstrahlung_visible/latency" in self.user

    def test_user_prompt_guides_missing_base_vocab_gap(self) -> None:
        # The user prompt's Vocabulary Gaps section must distinguish a genuine
        # missing base from the common false positive.
        assert "missing base IS a real gap" in self.user
