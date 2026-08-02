"""Closed-vocabulary completeness for the SN compose system prompt.

Verifies that the rendered ``sn/generate_name_system`` prompt — and the
shared ``_grammar_reference.md`` partial it embeds — contains EVERY token
from the canonical closed vocabulary segments declared by
``imas_standard_names.grammar.constants.SEGMENT_TOKEN_MAP``.

The dominant LLM failure mode is closed-vocabulary tokens (toroidal, parallel,
thermal, e_cross_b_drift, normalized, fast_ion, …) being absorbed into
``physical_base`` instead of placed in their correct grammar segment slot.
Injecting EVERY closed token verbatim into the system prompt makes that error
correctable by prompting alone; these tests guard the injection from silent
regressions.
"""

from __future__ import annotations

import re

import pytest

from imas_codex.llm.prompt_loader import PROMPTS_DIR, render_prompt
from imas_codex.standard_names.context import build_compose_context

# Aliased segments are duplicated in SEGMENT_TOKEN_MAP under multiple keys
# (component=coordinate, device=object, geometry=position).  We emit only the
# canonical names to keep the prompt cacheable; the alias names are noted in
# parentheses next to the canonical heading.
_ALIASES = {"coordinate", "object", "position"}

# Tokens shorter than this are skipped because single-letter coordinate
# tokens (x, y, z) are guaranteed to clash with arbitrary substrings.
_MIN_TOKEN_LEN = 3


def _closed_segments() -> dict[str, list[str]]:
    """Canonical closed segments (alias-deduped, open ones excluded)."""
    from imas_standard_names.grammar.constants import SEGMENT_TOKEN_MAP

    out: dict[str, list[str]] = {}
    for seg, toks in SEGMENT_TOKEN_MAP.items():
        if seg in _ALIASES:
            continue
        if not toks:  # skip open segments (physical_base)
            continue
        out[seg] = sorted(toks)
    return out


@pytest.fixture(scope="module")
def rendered_system_prompt() -> str:
    """Render the SN compose system prompt with the production context."""
    ctx = build_compose_context()
    return render_prompt("sn/generate_name_system", context=ctx)


@pytest.fixture(scope="module")
def closed_segments() -> dict[str, list[str]]:
    return _closed_segments()


class TestClosedVocabFull:
    """Exercise the ``_load_closed_vocab_full`` context builder directly."""

    def test_returns_nonempty_list(self):
        from imas_codex.standard_names.context import _load_closed_vocab_full

        data = _load_closed_vocab_full()
        assert isinstance(data, list)
        assert len(data) >= 6, "expected ≥6 canonical closed segments"

    def test_alias_segments_omitted_at_top_level(self):
        from imas_codex.standard_names.context import _load_closed_vocab_full

        seg_names = {entry["segment"] for entry in _load_closed_vocab_full()}
        assert _ALIASES.isdisjoint(seg_names), (
            f"alias segments must not appear as top-level entries; got {seg_names & _ALIASES}"
        )

    def test_aliases_attached_to_canonicals(self):
        from imas_codex.standard_names.context import _load_closed_vocab_full

        by_seg = {e["segment"]: e for e in _load_closed_vocab_full()}
        # component is canonical for coordinate; device for object; geometry for position
        assert "coordinate" in by_seg.get("component", {}).get("aliases", [])
        assert "object" in by_seg.get("device", {}).get("aliases", [])
        assert "position" in by_seg.get("geometry", {}).get("aliases", [])

    def test_physical_base_excluded(self):
        from imas_codex.standard_names.context import _load_closed_vocab_full

        seg_names = {entry["segment"] for entry in _load_closed_vocab_full()}
        assert "physical_base" in seg_names

    def test_tokens_sorted_alphabetically(self):
        from imas_codex.standard_names.context import _load_closed_vocab_full

        for entry in _load_closed_vocab_full():
            toks = entry["tokens"]
            assert toks == sorted(toks), (
                f"tokens for segment {entry['segment']!r} not sorted"
            )


class TestSystemPromptContainsAllClosedTokens:
    """The rendered prompt must contain every token from every closed segment."""

    def test_all_tokens_appear_in_rendered_prompt(
        self, rendered_system_prompt, closed_segments
    ):
        missing: list[str] = []
        for seg, toks in closed_segments.items():
            for tok in toks:
                if len(tok) < _MIN_TOKEN_LEN:
                    continue
                if tok not in rendered_system_prompt:
                    missing.append(f"{seg}:{tok}")
        assert not missing, (
            f"closed-vocab tokens missing from rendered system prompt "
            f"({len(missing)} missing): {missing[:25]}"
        )

    @pytest.mark.parametrize(
        "token",
        [
            "toroidal",
            "parallel",
            "thermal_electron",
            "e_cross_b_drift",
            "fast_ion",
            "volume_averaged",
            "pfirsch_schlueter",
            "scrape_off_layer",
            "edge_region",
            "diamagnetic_drift",
        ],
    )
    def test_high_signal_tokens_present(self, rendered_system_prompt, token):
        """Tokens cited in mid-tier reviewer comments must appear verbatim."""
        assert token in rendered_system_prompt, (
            f"high-signal closed-vocab token {token!r} missing from prompt — "
            "the reviewer corpus singles this token out as a recurring "
            "decomposition-failure absorber."
        )

    def test_decomposition_checklist_present(self, rendered_system_prompt):
        """The numbered checklist that drives self-correction must render."""
        assert "Decomposition Checklist" in rendered_system_prompt
        # Look for the action-verb cues so the checklist is recognisable
        assert "Tokenise the candidate" in rendered_system_prompt
        assert "physical_base" in rendered_system_prompt

    def test_decomposition_anti_pattern_gallery_renders(self, rendered_system_prompt):
        """Anti-pattern gallery section must render with at least 6 entries.

        The gallery heading is ``DECOMPOSITION-FAILURE GALLERY`` and each
        entry is prefixed ``D{n} —`` (a positional index within the gallery).
        """
        assert "DECOMPOSITION-FAILURE GALLERY" in rendered_system_prompt
        # Each entry uses the D{n} prefix
        marker_count = sum(
            1 for n in range(1, 16) if f"D{n} —" in rendered_system_prompt
        )
        assert marker_count >= 6, (
            f"expected ≥6 decomposition anti-pattern entries, found {marker_count}"
        )


class TestStaleGuidanceStripped:
    """The legacy 'physical_base is OPEN — no decomposition' wording must be gone."""

    def test_review_names_no_stale_open_section(self):
        """``review_names.md`` must not contain stale 'OPEN vocabulary' claims."""
        from imas_codex.llm.prompt_loader import PROMPTS_DIR

        text = (PROMPTS_DIR / "sn" / "review_names.md").read_text(encoding="utf-8")
        assert "## `physical_base` is OPEN vocabulary" not in text
        assert "SINGLE open grammar segment" not in text
        # New token vocabulary section must be present
        assert "Token vocabulary" in text

    def test_retry_prompt_no_open_vocab_phrasing(self):
        """The grammar-retry helper does not claim physical_base is open."""
        import inspect

        from imas_codex.standard_names.workers import _grammar_retry

        src = inspect.getsource(_grammar_retry)
        assert "open vocabulary" not in src, (
            "stale 'physical_base is open vocabulary' wording must be removed "
            "from the grammar retry helper"
        )


class TestDDGapEvidencePromptContract:
    """Every production response seat keeps DD-defect evidence flag-only."""

    @pytest.mark.parametrize(
        "seat",
        [
            "sn/generate_name_dd",
            "sn/review_names",
            "sn/review_docs",
            "sn/review_names_system",
            "sn/review_docs_system",
        ],
    )
    def test_prompt_preserves_exact_path_and_behavior_independence(self, seat):
        text = (PROMPTS_DIR / f"{seat}.md").read_text(encoding="utf-8")
        normalized = " ".join(text.replace("*", "").split())

        assert "exact claimed" in normalized or "exact source-binding" in normalized
        assert "Lexical name or attachment disagreement alone" in normalized
        assert "must not change" in normalized
        assert "score" in normalized
        assert "status" in normalized
        assert "enforcement" in normalized

    @pytest.mark.parametrize(
        "seat",
        [
            "sn/review_names_user",
            "sn/review_docs_user",
        ],
    )
    def test_live_review_user_prompt_requests_typed_evidence_field(self, seat):
        text = (PROMPTS_DIR / f"{seat}.md").read_text(encoding="utf-8")

        assert '"dd_gaps"' in text
        assert '"path"' in text
        assert '"kind"' in text
        assert '"reason"' in text

    def test_dead_source_fidelity_anchor_is_removed(self):
        text = (PROMPTS_DIR / "sn" / "review_names.md").read_text(encoding="utf-8")

        assert re.search(r"\[[A-Z]\d+\.\d+[a-z]?\]", text) is None
        assert "Source-fidelity check" in text


class TestClosedVocabFullSegments:
    """Verify closed_vocab_full includes ALL expected segments with tokens."""

    def test_all_expected_segments_present(self):
        from imas_codex.standard_names.context import _load_closed_vocab_full

        data = _load_closed_vocab_full()
        seg_names = {entry["segment"] for entry in data}
        # Must include at least these canonical segments
        for expected in (
            "physical_base",
            "qualifier",
            "component",
            "subject",
            "device",
            "geometry",
            "region",
            "process",
        ):
            assert expected in seg_names, (
                f"expected segment {expected!r} missing from closed_vocab_full"
            )

    def test_all_segments_have_nonzero_tokens(self):
        from imas_codex.standard_names.context import _load_closed_vocab_full

        for entry in _load_closed_vocab_full():
            assert len(entry["tokens"]) > 0, (
                f"segment {entry['segment']!r} has zero tokens — "
                "every closed segment must have at least one token"
            )


class TestRenderedPromptDecomposition:
    """Verify rendered prompt contains decomposition instructions and no stale counts."""

    def test_decomposition_rule_section_present(self, rendered_system_prompt):
        """The DECOMPOSITION RULE section must be present."""
        assert "DECOMPOSITION RULE" in rendered_system_prompt

    def test_anti_pattern_examples_present(self, rendered_system_prompt):
        """Key decomposition anti-patterns must appear in the prompt."""
        for pattern in (
            "momentum_diffusivity",
            "convection_velocity",
            "thermal_pressure",
        ):
            assert pattern in rendered_system_prompt, (
                f"decomposition anti-pattern {pattern!r} missing from prompt"
            )

    def test_no_stale_250_token_count(self, rendered_system_prompt):
        """No stale ~250 token counts should remain in the rendered prompt."""
        assert "~250" not in rendered_system_prompt, (
            "stale '~250' token count found in rendered prompt — "
            "physical_base has ~80 tokens, not ~250"
        )

    def test_no_stale_79_base_count(self, rendered_system_prompt):
        """The old 79-token count should not appear (now 80)."""
        assert "79 irreducible" not in rendered_system_prompt
        assert "79 listed" not in rendered_system_prompt
        assert "79 registered" not in rendered_system_prompt


def _endorsed_table_examples() -> list[str]:
    """Extract every corrected example from the generator's endorsed tables."""
    path = PROMPTS_DIR / "sn" / "generate_name_dd.md"
    examples: list[str] = []
    in_endorsed_table = False
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("| ❌") and "✅" in line:
            in_endorsed_table = True
            continue
        if in_endorsed_table and not line.startswith("|"):
            in_endorsed_table = False
            continue
        if not in_endorsed_table:
            continue
        columns = line.split("|")
        if len(columns) < 4:
            continue
        examples.extend(re.findall(r"`([a-z][a-z0-9_]{2,})`", columns[2]))
    return examples


class TestEndorsedExamplesParse:
    """Prompt recommendations must be accepted by the installed ISN oracle."""

    def test_corrected_table_examples_parse(self):
        from imas_standard_names.grammar import parse_standard_name

        examples = _endorsed_table_examples()
        assert len(examples) >= 9
        failures: dict[str, str] = {}
        for name in examples:
            try:
                parse_standard_name(name)
            except Exception as exc:
                failures[name] = str(exc)
        assert not failures

    def test_operator_precedence_is_visible_and_ordered(self):
        from imas_codex.standard_names.context import _load_operators_full

        operators = _load_operators_full()
        assert operators is not None
        for entries in operators.values():
            precedences = [int(entry["precedence"] or 0) for entry in entries]
            assert precedences == sorted(precedences, reverse=True)
        rendered = render_prompt(
            "sn/generate_name_system", context=build_compose_context()
        )
        assert "ordered by registry precedence" in rendered
        assert "precedence 35" in rendered


class TestGrammarOwnedAdvisoryAliases:
    """Alias guidance must render directly from the installed ISN contract."""

    @staticmethod
    def _aliases() -> list[tuple[str, str]]:
        from imas_standard_names import get_grammar_context

        return [
            (alias, details["canonical"])
            for aliases in get_grammar_context()["grammar"]["advisory_aliases"].values()
            for alias, details in aliases.items()
        ]

    def test_compose_prompt_renders_every_published_alias(self, rendered_system_prompt):
        for alias, canonical in self._aliases():
            assert f"`{alias}`" in rendered_system_prompt
            assert f"`{canonical}`" in rendered_system_prompt

    @pytest.mark.parametrize(
        "seat",
        ["sn/review_names_system", "sn/refine_name_system"],
    )
    def test_production_review_and_refine_prompts_render_every_alias(self, seat):
        rendered = render_prompt(seat, context=build_compose_context())
        assert "{{" not in rendered
        assert "{%" not in rendered
        for alias, canonical in self._aliases():
            assert f"`{alias}`" in rendered
            assert f"`{canonical}`" in rendered

    def test_assigned_policy_files_do_not_collapse_registered_loci(self):
        paths = [
            PROMPTS_DIR / "sn" / "generate_name_system.md",
            PROMPTS_DIR / "sn" / "generate_name_dd.md",
            PROMPTS_DIR / "sn" / "generate_name_dd_names.md",
            PROMPTS_DIR / "sn" / "review_names.md",
            PROMPTS_DIR / "sn" / "review_names_system.md",
            PROMPTS_DIR / "sn" / "refine_name_system.md",
            PROMPTS_DIR / "sn" / "refine_name_user.md",
            PROMPTS_DIR.parent / "config" / "sn_composition_rules.yaml",
            PROMPTS_DIR.parent / "config" / "sn_review_criteria.yaml",
        ]
        stale_claims = (
            "plasma_boundary replaces separatrix",
            "`separatrix` is a deprecated synonym",
            "separatrix` / `last_closed_flux_surface` / `lcfs`",
            "| plasma_boundary | separatrix",
        )
        for path in paths:
            text = path.read_text(encoding="utf-8")
            assert not any(claim in text for claim in stale_claims), path


class TestStructuredOperatorContract:
    """Prompt and response schema expose one outer-to-inner operator contract."""

    def test_prompt_describes_structured_chain(self, rendered_system_prompt):
        assert "outer-to-inner" in rendered_system_prompt
        assert "secondary_operand" in rendered_system_prompt
        assert '"token":"ratio"' in rendered_system_prompt
        for removed in (
            "operator_token",
            "operator_kind",
            "operator_coordinate",
            "secondary_base",
            "projection_shape",
        ):
            assert removed not in rendered_system_prompt

    def test_response_schema_has_ordered_operator_items(self):
        from imas_codex.standard_names.models import GrammarSegments

        schema = GrammarSegments.model_json_schema()
        properties = schema["properties"]
        assert "operators" in properties
        for removed in (
            "operator_token",
            "operator_kind",
            "operator_coordinate",
            "secondary_base",
        ):
            assert removed not in properties
        operator_properties = schema["$defs"]["GrammarOperator"]["properties"]
        assert set(operator_properties) == {
            "token",
            "coordinate",
            "secondary_operand",
        }
