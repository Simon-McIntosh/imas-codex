"""Operator tokens are known vocabulary, not absent gaps.

ISN operators (flux_surface_averaged, line_integrated, normalized, square,
derivative_with_respect_to, …) are a grammar mechanism separate from
SEGMENT_TOKEN_MAP: they compose into names as ``<op>_of_<base>`` / postfix.
A composer that reports one as a "missing" segment token has mis-slotted a
known operator — the gap classifier must see it as known (wrong-slot, hence
non-actionable), never fabricate an ``absent`` VocabGap for it.
"""

from __future__ import annotations

from imas_codex.standard_names.segments import (
    classify_gap,
    is_actionable_gap,
    is_known_token,
)


def test_operator_tokens_are_known():
    # These live in the ISN operators vocabulary, not SEGMENT_TOKEN_MAP.
    for op in ("flux_surface_averaged", "line_integrated", "normalized", "square"):
        segs = is_known_token(op)
        assert segs, f"{op!r} must be recognized as a known operator token"


def test_operator_reported_as_wrong_slot_not_absent():
    # Composer mis-slots the operator as a qualifier/physical_base.
    cat, _ = classify_gap("qualifier", "normalized")
    assert cat != "absent", "a known operator must never classify as an absent gap"
    cat2, _ = classify_gap("physical_base", "flux_surface_averaged")
    assert cat2 != "absent"


def test_operator_gap_is_not_actionable():
    # Non-actionable ⇒ no VocabGap node, source stays retryable (not stranded).
    assert is_actionable_gap("qualifier", "normalized") is False
    assert is_actionable_gap("physical_base", "flux_surface_averaged") is False


def test_genuinely_absent_token_still_absent():
    # A made-up token that is neither a segment token nor an operator must
    # still surface as a genuine, actionable gap.
    cat, _ = classify_gap("physical_base", "zzz_not_a_real_token_qqq")
    assert cat == "absent"
    assert is_actionable_gap("physical_base", "zzz_not_a_real_token_qqq") is True
