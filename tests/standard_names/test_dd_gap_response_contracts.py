"""Typed, evidence-only DD-gap response contracts for composition and review."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from imas_codex.standard_names.models import (
    DDGapEvidence,
    StandardNameComposeBatch,
    StandardNameQualityReviewDocs,
    StandardNameQualityReviewNameOnly,
)

PATH = "equilibrium/time_slice/profiles_1d/pressure"
EVIDENCE = {
    "path": PATH,
    "kind": "unit_defect",
    "reason": "The DD declares 1 while the pressure definition and twin path use Pa.",
    "observed_dd_version": "4.1.0",
    "observed_value": "1",
    "expected_value": "Pa",
    "evidence_rule": "declared unit must match the defined physical quantity",
    "reference_path": "equilibrium/time_slice/profiles_1d/pressure_thermal",
    "reference_value": "Pa",
}


def _name_review(**extra: object) -> StandardNameQualityReviewNameOnly:
    return StandardNameQualityReviewNameOnly.model_validate(
        {
            "source_id": PATH,
            "standard_name": "pressure",
            "scores": {
                "grammar": 20,
                "semantic": 19,
                "convention": 20,
                "completeness": 19,
            },
            "reasoning": "The name is sound independently of the DD declaration defect.",
            **extra,
        }
    )


def _docs_review(**extra: object) -> StandardNameQualityReviewDocs:
    return StandardNameQualityReviewDocs.model_validate(
        {
            "source_id": PATH,
            "standard_name": "pressure",
            "scores": {
                "description_quality": 19,
                "documentation_quality": 18,
                "completeness": 19,
                "physics_accuracy": 20,
            },
            "reasoning": "The documentation accurately defines pressure.",
            **extra,
        }
    )


def test_compose_batch_accepts_structured_optional_dd_gap_evidence() -> None:
    batch = StandardNameComposeBatch.model_validate(
        {"candidates": [], "dd_gaps": [EVIDENCE]}
    )

    assert batch.dd_gaps == [DDGapEvidence.model_validate(EVIDENCE)]
    assert batch.dd_gaps[0].path == PATH
    assert batch.dd_gaps[0].observed_value == "1"
    assert batch.dd_gaps[0].expected_value == "Pa"


def test_name_and_docs_reviews_accept_evidence_without_changing_scores() -> None:
    name_without = _name_review()
    name_with = _name_review(dd_gaps=[EVIDENCE])
    docs_without = _docs_review()
    docs_with = _docs_review(dd_gaps=[EVIDENCE])

    assert name_with.scores.score == name_without.scores.score
    assert docs_with.scores.score == docs_without.scores.score
    assert name_with.dd_gaps[0].kind == "unit_defect"
    assert docs_with.dd_gaps[0].path == PATH


def test_dd_gap_evidence_is_optional_and_defaults_empty() -> None:
    assert StandardNameComposeBatch(candidates=[]).dd_gaps == []
    assert _name_review().dd_gaps == []
    assert _docs_review().dd_gaps == []


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("kind", "name_disagreement", "Input should be"),
        ("path", "equilibrium/*/pressure", "exact DD path"),
        ("path", "", "at least 1 character"),
        ("reason", "too vague", "at least 12 characters"),
    ],
)
def test_dd_gap_evidence_rejects_invalid_contract_values(
    field: str, value: str, message: str
) -> None:
    payload = {**EVIDENCE, field: value}
    with pytest.raises(ValidationError, match=message):
        DDGapEvidence.model_validate(payload)


@pytest.mark.parametrize("forbidden", ["status", "enforcement", "disposition"])
def test_dd_gap_evidence_rejects_lifecycle_or_behavior_fields(forbidden: str) -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        DDGapEvidence.model_validate({**EVIDENCE, forbidden: "accepted"})


def test_reference_evidence_requires_path_and_value_together() -> None:
    payload = {**EVIDENCE}
    payload.pop("reference_value")
    with pytest.raises(ValidationError, match="reference_path and reference_value"):
        DDGapEvidence.model_validate(payload)


def test_schema_exposes_only_schema_owned_dd_gap_kinds() -> None:
    from imas_codex.graph.models import DDGapKind

    schema = DDGapEvidence.model_json_schema()
    kinds = schema["properties"]["kind"]["enum"]

    assert kinds == [
        "unit_defect",
        "self_contradiction",
        "doc_mismatch",
        "type_wiring",
        "missing_declaration",
        "rename_inconsistency",
    ]
    assert kinds == [item.value for item in DDGapKind]
    assert "status" not in schema["properties"]
