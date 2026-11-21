"""Repairable defects in optional reviewer-response fields repair in place.

A paid review must survive a repairable defect in an OPTIONAL field: half of a
reference-evidence pair, or a suggested name the grammar cannot parse, carries
no usable information, so the field is cleared and the repair recorded rather
than the whole document being discarded. The one invariant that must NOT move
is the sibling rule that a wildcard DD path is not repairable — it still
rejects.
"""

from __future__ import annotations

import logging

import pytest
from pydantic import ValidationError

from iter_codex.standard_names.models import (
    DDGapEvidence,
    StandardNameQualityReview,
    StandardNameQualityReviewNameOnly,
)

PATH = "summary/global_quantities/ip/value"
EVIDENCE = {
    "path": PATH,
    "kind": "unit_defect",
    "reason": "The DD declares 1 while the twin path uses ampere.",
    "observed_dd_version": "4.1.0",
    "observed_value": "1",
    "expected_value": "A",
    "evidence_rule": "declared unit must match the defined physical quantity",
    "reference_path": "summary/global_quantities/ip/value_current",
    "reference_value": "A",
}

# 4- and 6-dimensional score payloads for the two review models under test.
_MODELS_AND_SCORES = [
    (
        StandardNameQualityReviewNameOnly,
        {
            "grammar": 18,
            "semantic": 17,
            "convention": 16,
            "completeness": 15,
        },
    ),
    (
        StandardNameQualityReview,
        {
            "grammar": 18,
            "semantic": 17,
            "documentation": 15,
            "convention": 16,
            "completeness": 15,
            "compliance": 17,
        },
    ),
]


# ---------------------------------------------------------------------------
# Repair one: half a reference-evidence pair is cleared, never raised
# ---------------------------------------------------------------------------


def test_half_reference_pair_repairs_and_names_the_missing_field() -> None:
    payload = dict(EVIDENCE)
    payload.pop("reference_value")

    evidence = DDGapEvidence.model_validate(payload)

    assert evidence.reference_path is None
    assert evidence.reference_value is None
    assert evidence.reference_evidence_repaired is True
    assert evidence.reference_field_missing == "reference_value"


def test_complete_reference_pair_is_kept_unchanged() -> None:
    evidence = DDGapEvidence.model_validate(EVIDENCE)

    assert evidence.reference_path == EVIDENCE["reference_path"]
    assert evidence.reference_value == EVIDENCE["reference_value"]
    assert evidence.reference_evidence_repaired is False
    assert evidence.reference_field_missing is None


def test_absent_reference_pair_is_unchanged_and_unrepaired() -> None:
    payload = {
        k: v
        for k, v in EVIDENCE.items()
        if k not in ("reference_path", "reference_value")
    }

    evidence = DDGapEvidence.model_validate(payload)

    assert evidence.reference_path is None
    assert evidence.reference_value is None
    assert evidence.reference_evidence_repaired is False
    assert evidence.reference_field_missing is None


def test_wildcard_reference_path_still_raises() -> None:
    """A pattern in a DD path is not repairable; the sibling rule is intact."""
    payload = dict(EVIDENCE)
    payload["reference_path"] = "summary/*/ip/value"

    with pytest.raises(ValidationError, match="exact DD path"):
        DDGapEvidence.model_validate(payload)


# ---------------------------------------------------------------------------
# Repair two: an unparseable suggested name is cleared, never a score-loser
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model_cls,scores", _MODELS_AND_SCORES)
def test_unparseable_suggested_name_cleared_and_recorded(
    model_cls, scores, caplog
) -> None:
    payload = {
        "source_id": PATH,
        "standard_name": "plasma_current",
        "scores": scores,
        "reasoning": "The name is sound independently of the DD declaration defect.",
        "suggested_name": "electron temperature",
        "suggestion_justification": "Cluster siblings use a different qualifier order.",
    }

    with caplog.at_level(logging.WARNING, logger="iter_codex.standard_names.models"):
        review = model_cls.model_validate(payload)

    assert review.suggested_name is None
    assert review.suggestion_justification is None
    # every numeric score survives byte-identical
    for dimension, value in scores.items():
        assert getattr(review.scores, dimension) == value
    assert "Cleared reviewer suggested_name" in caplog.text
    assert "not a valid grammar token" in caplog.text


@pytest.mark.parametrize("model_cls,scores", _MODELS_AND_SCORES)
def test_parseable_suggested_name_is_kept_unchanged(model_cls, scores) -> None:
    payload = {
        "source_id": PATH,
        "standard_name": "plasma_current",
        "scores": scores,
        "reasoning": "The name is sound independently of the DD declaration defect.",
        "suggested_name": "ion_temperature",
        "suggestion_justification": "Cluster siblings use the ion carrier.",
    }

    review = model_cls.model_validate(payload)

    assert review.suggested_name == "ion_temperature"
    assert review.suggestion_justification == ("Cluster siblings use the ion carrier.")


# ---------------------------------------------------------------------------
# The unwired helper: suggestion identity, not suggestion quality
# ---------------------------------------------------------------------------


def test_suggestion_identity_helper_distinguishes_collapse_parse_and_emittable() -> (
    None
):
    from iter_codex.standard_names.models import suggestion_is_semantically_distinct

    # a different spelling of the same reviewed identity: no objection to record
    assert (
        suggestion_is_semantically_distinct("atomic_number", "nuclear_charge_number")
        is False
    )
    # a proposal the grammar cannot parse is not a proposal
    assert (
        suggestion_is_semantically_distinct("plasma_current", "electron temperature")
        is False
    )
    # a different, emittable name is a real objection
    assert (
        suggestion_is_semantically_distinct("plasma_current", "ion_temperature") is True
    )
