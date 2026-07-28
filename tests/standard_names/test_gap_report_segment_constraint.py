"""Response-model constraints on vocabulary-gap reports and qualifier slots.

Two invariants keep a composer's correct physics from being lost on the way into
the graph.

A gap report must name a grammar class that exists: a class nothing reads makes
the gap invisible to the reconcile that would resolve or retire it, so the field
is constrained, and a report that misses is repaired at batch level rather than
failing every well-formed candidate beside it.

A registered operator offered in ``qualifiers`` names the right physics in the
wrong slot. That is repairable by re-slotting it into ``operator_token``, which
the operator registry renders unambiguously — so it is repaired, not rejected.
Rejection has no path back: the candidate becomes a vocabulary gap and strands
its source.
"""

from __future__ import annotations

import pytest

from imas_codex.standard_names.models import (
    GrammarSegments,
    StandardNameComposeBatch,
    StandardNameVocabGap,
)
from imas_codex.standard_names.segments import OPERATOR_SEGMENT, reportable_segments


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
# The gap-report segment field
# ---------------------------------------------------------------------------


@requires_isn
class TestVocabGapSegmentConstraint:
    """``segment`` may only name a grammar class that exists."""

    def test_real_segment_accepted(self):
        gap = StandardNameVocabGap(
            source_id="a/b/c", segment="physical_base", token="foo", reason="r"
        )
        assert gap.segment == "physical_base"

    def test_operator_class_accepted(self):
        gap = StandardNameVocabGap(
            source_id="a/b/c", segment=OPERATOR_SEGMENT, token="square", reason="r"
        )
        assert gap.segment == OPERATOR_SEGMENT

    def test_pseudo_segment_accepted(self):
        gap = StandardNameVocabGap(
            source_id="a/b/c", segment="grammar_ambiguity", token="foo", reason="r"
        )
        assert gap.segment == "grammar_ambiguity"

    def test_invented_segment_rejected(self):
        with pytest.raises(ValueError, match="not a grammar segment class"):
            StandardNameVocabGap(
                source_id="a/b/c",
                segment="vibes_segment_xyzzy",
                token="foo",
                reason="r",
            )

    def test_reportable_segments_covers_the_real_classes(self):
        classes = reportable_segments()
        for expected in ("physical_base", "qualifier", "component", OPERATOR_SEGMENT):
            assert expected in classes


@requires_isn
class TestBatchNormalisesGapSegments:
    """A mis-named segment is repaired at batch level, never fails the batch."""

    def _batch(self, gaps: list[dict]) -> StandardNameComposeBatch:
        return StandardNameComposeBatch.model_validate(
            {"candidates": [], "vocab_gaps": gaps}
        )

    def test_operator_token_spelling_normalised_to_the_class(self):
        batch = self._batch(
            [
                {
                    "source_id": "a/b",
                    "segment": "operator_token",
                    "token": "square",
                    "reason": "r",
                }
            ]
        )
        assert [g.segment for g in batch.vocab_gaps] == [OPERATOR_SEGMENT]

    def test_unmappable_segment_inferred_from_the_token(self):
        """An unrecognised class whose token is registered is re-filed, not dropped."""
        batch = self._batch(
            [
                {
                    "source_id": "a/b",
                    "segment": "nonsense_class_xyzzy",
                    "token": "square",
                    "reason": "r",
                }
            ]
        )
        assert [g.segment for g in batch.vocab_gaps] == [OPERATOR_SEGMENT]

    def test_unmappable_and_unknown_token_is_dropped_not_fatal(self):
        batch = self._batch(
            [
                {
                    "source_id": "a/b",
                    "segment": "nonsense_class_xyzzy",
                    "token": "zzz_unknown_xyzzy",
                    "reason": "r",
                },
                {
                    "source_id": "a/c",
                    "segment": "physical_base",
                    "token": "zzz_other_xyzzy",
                    "reason": "r",
                },
            ]
        )
        # The undiagnosable gap carries no actionable content and goes; the
        # well-formed sibling in the same batch survives.
        assert [g.source_id for g in batch.vocab_gaps] == ["a/c"]

    def test_well_formed_gaps_pass_through_untouched(self):
        batch = self._batch(
            [
                {
                    "source_id": "a/b",
                    "segment": "physical_base",
                    "token": "zzz_novel_xyzzy",
                    "reason": "r",
                }
            ]
        )
        assert len(batch.vocab_gaps) == 1
        assert batch.vocab_gaps[0].segment == "physical_base"


# ---------------------------------------------------------------------------
# Operator-valued qualifiers
# ---------------------------------------------------------------------------


def _segments(**kwargs) -> GrammarSegments:
    base = {"base_token": "temperature", "base_kind": "quantity"}
    base.update(kwargs)
    return GrammarSegments.model_validate(base)


@requires_isn
class TestQualifierOperatorPromotion:
    """A registered operator in ``qualifiers`` moves to the operator slot."""

    def test_operator_qualifier_is_promoted(self):
        seg = _segments(qualifiers=["square"])
        assert seg.operator_token == "square"
        assert "square" not in seg.qualifiers

    def test_promotion_preserves_ordinary_qualifiers(self):
        seg = _segments(qualifiers=["electron", "square"])
        assert seg.operator_token == "square"
        assert seg.qualifiers == ["electron"]

    def test_promoted_operator_renders_into_the_name(self):
        """The promotion has to reach the composed name, not just the field."""
        seg = _segments(qualifiers=["square"])
        rendered = seg._to_model_dict()
        assert "square" in (
            rendered.get("transformation") or rendered.get("decomposition") or ""
        )

    def test_occupied_operator_slot_is_not_overwritten(self):
        """With the slot taken the operator qualifier is an error, not a silent drop."""
        with pytest.raises(ValueError, match="registered OPERATOR"):
            _segments(qualifiers=["square"], operator_token="magnitude")

    def test_unregistered_qualifier_still_rejected(self):
        with pytest.raises(ValueError, match="not a registered grammar token"):
            _segments(qualifiers=["zzz_not_a_token_xyzzy"])

    def test_ordinary_qualifier_untouched(self):
        seg = _segments(qualifiers=["electron"])
        assert seg.qualifiers == ["electron"]
        assert seg.operator_token is None

    def test_operator_qualifier_survives_the_batch_as_a_candidate(self):
        """Promotion happens early enough that the candidate is never rescued.

        A registered operator offered as a qualifier is repairable, so the
        candidate must reach the batch intact rather than being converted into a
        vocabulary gap that strands its source.
        """
        batch = StandardNameComposeBatch.model_validate(
            {
                "candidates": [
                    {
                        "source_id": "p/a",
                        "segments": {
                            "base_token": "torque",
                            "base_kind": "quantity",
                            "qualifiers": ["cumulative"],
                        },
                        "description": "Cumulative torque",
                        "reason": "test",
                    }
                ]
            }
        )
        assert len(batch.candidates) == 1
        assert batch.candidates[0].segments.operator_token == "cumulative"
        assert batch.vocab_gaps == []


# ---------------------------------------------------------------------------
# Token-reuse comparison set
# ---------------------------------------------------------------------------


@requires_isn
class TestDedupSeesOperators:
    """The reuse check must be able to compare a proposal against operators."""

    def test_operator_class_is_a_comparison_segment(self):
        from imas_codex.standard_names.vocab_semantic_dedup import (
            _existing_tokens_by_segment,
        )

        by_seg = _existing_tokens_by_segment()
        assert OPERATOR_SEGMENT in by_seg
        assert "square" in by_seg[OPERATOR_SEGMENT]
        assert "flux_surface_averaged" in by_seg[OPERATOR_SEGMENT]

    def test_segment_vocabularies_still_present(self):
        from imas_codex.standard_names.vocab_semantic_dedup import (
            _existing_tokens_by_segment,
        )

        by_seg = _existing_tokens_by_segment()
        assert "physical_base" in by_seg
        assert "qualifier" in by_seg
