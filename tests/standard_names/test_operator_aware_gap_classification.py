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
# expressible from registered operators applied to a registered base.  Every
# token here was taken verbatim from a recorded ``VocabGap``, so this is the real
# proposal distribution rather than a set of constructed cases.
OPERATOR_EXPRESSIBLE = [
    ("qualifier", "inverse_square"),
    ("physical_base", "flux_surface_averaged_square_magnetic_field"),
    ("physical_base", "inverse_square_of_magnetic_field"),
    ("physical_base", "line_integrated_density"),
    ("physical_base", "magnetic_field_magnitude_squared"),
    ("qualifier", "average_square"),
    ("qualifier", "mean_square"),
    ("qualifier", "squared"),
    ("physical_base", "volume_derivative_with_respect_to_toroidal_flux"),
    ("physical_base", "z_square_average"),
    ("qualifier", "perpendicular_square"),
    ("qualifier", "inverse_magnetic_field_squared"),
]

# Ratio spellings: `over` is the composer's English for the binary `ratio`
# operator.  Both operands must cover independently, so these are composable only
# when every word on each side is registered.
RATIO_EXPRESSIBLE = [
    ("qualifier", "gradient_squared_over_magnetic_field_squared"),
    # Symbol shorthand in both operands, read as the registered tokens it
    # abbreviates (`rho` -> toroidal_flux_radius, `B` -> magnetic_field).
    ("physical_base", "gradient_rho_squared_over_B_squared"),
]

# Infix operator spellings: a registered multi-word operator whose words straddle
# the operand it applies to.
INFIX_EXPRESSIBLE = [
    ("physical_base", "derivative_of_area_with_respect_to_poloidal_flux"),
]

# Recorded proposals that stay `absent`, with the reason each needs something
# other than operator awareness.  A token moving out of this list into an
# expressible list is an improvement; the reverse is a regression.  The rule is
# NOT stretched to reach these — an over-eager matcher that called novel physics
# composable would suppress the vocabulary requests ISN actually needs.
KNOWN_UNRESOLVED = {
    # `field` alone is not a registered base — only `magnetic_field` is, and no
    # symbol expansion reads a bare `field` as one: the composer may mean the
    # electric field just as well.
    "radial_gradient_squared_over_field_squared": "unregistered ratio operand",
    # `variation` is a registered operator and `length` a registered base, but
    # `path` is registered nowhere, so a token is genuinely needed.
    "path_length_variation": "unregistered residual token",
    # A synonym of the registered base `opacity` — a reuse question for the
    # token-similarity check, not a composition one.
    "optical_depth": "synonym of a registered base",
}

# Compounds that LOOK like the composable cases but are genuinely novel physics.
# These guard the widened matcher: each contains registered-looking material yet
# must keep asking ISN for a token.  A failure here means the rule was loosened
# until it stopped discriminating, which would suppress the vocabulary requests
# ISN actually needs — a worse failure than the fabricated gaps being removed.
#
# Every token here must be unregistered in EVERY class and unresolvable by the
# ISN parser.  A token that is merely in a different slot (``suprathermal`` is a
# registered population) or that the parser resolves as a lexical compound is a
# different verdict entirely, so it does not test this boundary.
MUST_STAY_ABSENT = [
    # Substrings of registered tokens, unreachable at a token boundary:
    # `substrate` ends in the base `rate`, `byproduct` in the operator `product`.
    ("physical_base", "substrate"),
    ("physical_base", "byproduct"),
    ("physical_base", "wibble_frobnicator"),
    ("qualifier", "zzz_not_a_token"),
    # An operator word reachable only inside a longer unregistered word.
    ("physical_base", "squaring_kernel"),
    ("physical_base", "gradiental_wibble"),
    # A ratio spelling whose operands are not registered — the ratio rule must
    # not fire just because `over` is present.
    ("physical_base", "zzz_alpha_over_zzz_beta"),
    ("physical_base", "wibble_over_frobnicator"),
    # A registered operator applied to an unregistered residue: the operator half
    # must not carry the whole compound.
    ("physical_base", "gradient_of_wibble_frobnicator"),
    ("physical_base", "line_integrated_wibble"),
    # An infix operator spelling whose operand and residue are unregistered.
    ("physical_base", "derivative_of_wibble_with_respect_to_frobnicator"),
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

    @pytest.mark.parametrize("token", sorted(KNOWN_UNRESOLVED))
    def test_documented_boundary_of_the_rule(self, token):
        """The unresolved cases stay unresolved until something else handles them.

        Pinned so widening the cover walk shows up here as a deliberate change
        rather than as an unnoticed side effect.
        """
        assert operator_composition(token) is None, (
            f"{token} now resolves ({KNOWN_UNRESOLVED[token]}) — move it into "
            f"an expressible list"
        )

    @pytest.mark.parametrize(("segment", "token"), MUST_STAY_ABSENT)
    def test_novel_physics_still_asks_for_a_token(self, segment, token):
        """The matcher must keep discriminating, not match everything.

        A compound that only resembles the composable cases has to keep
        classifying ``absent``: suppressing a real vocabulary request is a worse
        failure than the fabricated gaps this rule removes.
        """
        assert operator_composition(token) is None, (
            f"{token} is novel physics but was called operator-composable"
        )
        assert classify_gap(segment, token) == ("absent", []), (
            f"{segment}/{token} must stay absent — a real gap was suppressed"
        )


@requires_isn
class TestRatioSpelling:
    """``over`` spells the binary ratio operator, and both operands must cover."""

    @pytest.mark.parametrize(("segment", "token"), RATIO_EXPRESSIBLE)
    def test_ratio_is_composable(self, segment, token):
        comp = operator_composition(token)
        assert comp is not None
        assert comp.binary_operator == "ratio"
        assert classify_gap(segment, token)[0] != "absent"

    def test_guidance_names_the_binary_operator_and_both_operands(self):
        verdict = describe_gap(
            "qualifier", "gradient_squared_over_magnetic_field_squared"
        )
        assert "ratio" in verdict.guidance
        assert "secondary_base" in verdict.guidance

    def test_a_ratio_with_an_unregistered_operand_stays_absent(self):
        """One uncovered operand is enough — the rule does not guess.

        Symbol expansion narrows what counts as uncovered but does not soften
        this: a bare `field` has no unambiguous reading, so the request for a
        token stands.
        """
        assert (
            operator_composition("radial_gradient_squared_over_field_squared") is None
        )
        assert operator_composition("zzz_alpha_over_zzz_beta") is None


@requires_isn
class TestInfixOperatorSpelling:
    """A registered multi-word operator split around the operand it applies to."""

    @pytest.mark.parametrize(("segment", "token"), INFIX_EXPRESSIBLE)
    def test_infix_operator_is_composable(self, segment, token):
        comp = operator_composition(token)
        assert comp is not None
        assert classify_gap(segment, token)[0] != "absent"

    def test_reports_the_whole_registered_operator(self):
        comp = operator_composition("derivative_of_area_with_respect_to_poloidal_flux")
        assert comp is not None
        assert "derivative_with_respect_to" in comp.operators
        assert "area" in comp.bases

    def test_a_split_operator_needs_both_halves_present(self):
        """Only the head of a multi-word operator is not an infix spelling."""
        assert operator_composition("derivative_of_zzz_unknown_xyzzy") is None

    def test_the_residue_must_still_be_registered(self):
        assert (
            operator_composition("derivative_of_wibble_with_respect_to_frobnicator")
            is None
        )


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
