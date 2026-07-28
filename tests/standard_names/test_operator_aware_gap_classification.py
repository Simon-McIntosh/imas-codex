"""Operator awareness across every consumer of the grammar token vocabulary.

Operators (``square``, ``inverse``, ``flux_surface_averaged``,
``derivative_with_respect_to`` …) are a grammar *mechanism*: they compose
through ``operator_token`` rather than occupying a ``SEGMENT_TOKEN_MAP`` slot.
Any consumer that reaches for ``SEGMENT_TOKEN_MAP`` alone therefore cannot see
them, and treats a perfectly composable name as missing vocabulary.

These tests pin the shared accessor that carries operators alongside the
segment vocabularies, and the behaviour every consumer must inherit from it:
a token expressible as registered operators applied to a registered base is
never a genuine vocabulary deficiency.
"""

from __future__ import annotations

import pytest

from imas_codex.standard_names.segments import (
    OPERATOR_SEGMENT,
    classify_gap,
    describe_gap,
    grammar_token_index,
    grammar_tokens_by_segment,
    operator_composition,
)

pytestmark = pytest.mark.usefixtures()


def _isn_available() -> bool:
    try:
        import imas_standard_names.grammar.constants  # noqa: F401
    except ImportError:
        return False
    return True


requires_isn = pytest.mark.skipif(
    not _isn_available(), reason="imas-standard-names not installed"
)


# ---------------------------------------------------------------------------
# The shared accessor
# ---------------------------------------------------------------------------


@requires_isn
class TestGrammarTokensBySegment:
    """The single accessor every consumer reads its vocabulary from."""

    def test_carries_the_operator_class(self):
        by_seg = grammar_tokens_by_segment()
        assert OPERATOR_SEGMENT in by_seg
        assert by_seg[OPERATOR_SEGMENT], "operator class must not be empty"

    def test_carries_every_isn_operator(self):
        from imas_standard_names import get_grammar_context

        operators = set(
            get_grammar_context()["grammar"]["vocabularies"]["operators"].keys()
        )
        exposed = set(grammar_tokens_by_segment()[OPERATOR_SEGMENT])
        assert operators == exposed, (
            f"accessor drifted from ISN operator registry; "
            f"missing={sorted(operators - exposed)} extra={sorted(exposed - operators)}"
        )

    def test_carries_every_segment_vocabulary(self):
        from imas_standard_names.grammar.constants import SEGMENT_TOKEN_MAP

        by_seg = grammar_tokens_by_segment()
        for segment, tokens in SEGMENT_TOKEN_MAP.items():
            if not tokens:
                continue
            assert set(tokens) <= set(by_seg.get(segment, ())), (
                f"accessor under-reports segment {segment}"
            )

    def test_reverse_index_resolves_operators(self):
        index = grammar_token_index()
        assert OPERATOR_SEGMENT in index.get("square", ())
        assert OPERATOR_SEGMENT in index.get("inverse", ())
        assert OPERATOR_SEGMENT in index.get("flux_surface_averaged", ())

    def test_degrades_to_empty_without_isn(self, monkeypatch):
        """Absent ISN the accessor yields nothing — the rule turns off, never stale."""
        import imas_codex.standard_names.segments as seg_mod

        monkeypatch.setattr(seg_mod, "_load_segment_token_map", lambda: None)
        monkeypatch.setattr(seg_mod, "_operator_tokens", lambda: frozenset())
        grammar_tokens_by_segment.cache_clear()
        grammar_token_index.cache_clear()
        try:
            assert grammar_tokens_by_segment() == {}
            assert grammar_token_index() == {}
        finally:
            grammar_tokens_by_segment.cache_clear()
            grammar_token_index.cache_clear()


# ---------------------------------------------------------------------------
# Operator composition
# ---------------------------------------------------------------------------

# Compounds the composer proposed as missing vocabulary that are in fact
# expressible from registered operators applied to a registered base.
OPERATOR_EXPRESSIBLE = [
    ("qualifier", "inverse_square"),
    ("physical_base", "flux_surface_averaged_square_magnetic_field"),
    ("physical_base", "inverse_square_of_magnetic_field"),
    ("physical_base", "line_integrated_density"),
    ("physical_base", "magnetic_field_magnitude_squared"),
    ("qualifier", "average_square"),
    ("qualifier", "mean_square"),
    ("physical_base", "volume_derivative_with_respect_to_toroidal_flux"),
]


@requires_isn
class TestOperatorComposition:
    """``operator_composition`` reports the operators and the residual base."""

    @pytest.mark.parametrize(("segment", "token"), OPERATOR_EXPRESSIBLE)
    def test_recognises_expressible_compounds(self, segment, token):
        comp = operator_composition(token)
        assert comp is not None, f"{token} should be operator-expressible"
        assert comp.operators, f"{token} yielded no operators"

    def test_reports_the_operators_it_found(self):
        comp = operator_composition("flux_surface_averaged_square_magnetic_field")
        assert comp is not None
        assert "flux_surface_averaged" in comp.operators
        assert "square" in comp.operators
        assert "magnetic_field" in comp.bases

    def test_single_operator_morphological_variant(self):
        """``squared`` is the operator ``square`` in participle form."""
        comp = operator_composition("squared")
        assert comp is not None
        assert comp.operators == ("square",)

    def test_requires_at_least_one_operator(self):
        """A compound of ordinary segment tokens is not an operator composition."""
        assert operator_composition("thermal_density") is None

    def test_containing_an_operator_word_is_not_enough(self):
        """A token that merely *contains* operator letters is not composable.

        ``substrate`` ends in the registered base ``rate`` and ``product`` is a
        registered operator, but neither is reachable at a token boundary.
        """
        assert operator_composition("substrate") is None
        assert operator_composition("byproduct") is None

    def test_unknown_residual_is_not_composable(self):
        assert operator_composition("square_of_zzz_unknown_xyzzy") is None

    def test_atomic_compounds_are_not_decomposed(self):
        from imas_codex.standard_names.segments import ATOMIC_COMPOUNDS

        for compound in ("magnetic_field", "poloidal_magnetic_flux", "safety_factor"):
            assert compound in ATOMIC_COMPOUNDS
            assert operator_composition(compound) is None


# ---------------------------------------------------------------------------
# classify_gap must never call an operator-expressible token absent
# ---------------------------------------------------------------------------


@requires_isn
class TestClassifyGapOperatorAware:
    """``absent`` is reserved for tokens the grammar genuinely cannot express."""

    @pytest.mark.parametrize(("segment", "token"), OPERATOR_EXPRESSIBLE)
    def test_never_absent_for_operator_expressible(self, segment, token):
        category, actual = classify_gap(segment, token)
        assert category != "absent", (
            f"{segment}/{token} is expressible from registered operators "
            f"but was classified absent"
        )
        assert actual, "an actionable verdict must name the segments involved"

    @pytest.mark.parametrize(("segment", "token"), OPERATOR_EXPRESSIBLE)
    def test_not_actionable_for_operator_expressible(self, segment, token):
        from imas_codex.standard_names.segments import is_actionable_gap

        assert not is_actionable_gap(segment, token), (
            f"{segment}/{token} must not strand its source as a vocabulary gap"
        )

    def test_operator_slot_is_named_in_the_verdict(self):
        _, actual = classify_gap("qualifier", "inverse_square")
        assert OPERATOR_SEGMENT in actual

    def test_single_operator_still_wrong_slot(self):
        """The single-operator verdict is unchanged."""
        assert classify_gap("qualifier", "square") == (
            "wrong_slot_placement",
            [OPERATOR_SEGMENT],
        )
        assert classify_gap("qualifier", "inverse") == (
            "wrong_slot_placement",
            [OPERATOR_SEGMENT],
        )

    def test_genuinely_absent_token_still_absent(self):
        assert classify_gap("position", "zzz_nonexistent_token_xyzzy") == ("absent", [])
        assert classify_gap("qualifier", "zzz_truly_unique_xyzzy") == ("absent", [])


# ---------------------------------------------------------------------------
# The verdict has to be usable as retry feedback
# ---------------------------------------------------------------------------


@requires_isn
class TestDescribeGap:
    """``describe_gap`` renders a verdict a model can act on."""

    def test_names_the_operators_and_the_slot(self):
        verdict = describe_gap("qualifier", "inverse_square")
        assert verdict.category != "absent"
        assert verdict.operators == ("inverse", "square")
        assert "operator_token" in verdict.guidance
        assert "inverse" in verdict.guidance
        assert "square" in verdict.guidance

    def test_single_operator_guidance_names_the_registered_class(self):
        verdict = describe_gap("qualifier", "square")
        assert "square" in verdict.guidance
        assert "operator" in verdict.guidance
        assert "qualifier" in verdict.guidance

    def test_absent_token_guidance_states_the_deficiency(self):
        verdict = describe_gap("position", "zzz_nonexistent_token_xyzzy")
        assert verdict.category == "absent"
        assert verdict.operators == ()
        assert verdict.guidance

    def test_guidance_is_a_single_nonempty_line(self):
        for segment, token in OPERATOR_EXPRESSIBLE:
            guidance = describe_gap(segment, token).guidance
            assert guidance.strip()
            assert "\n" not in guidance

    def test_category_agrees_with_classify_gap(self):
        for segment, token in [*OPERATOR_EXPRESSIBLE, ("qualifier", "square")]:
            assert (
                describe_gap(segment, token).category == classify_gap(segment, token)[0]
            )
