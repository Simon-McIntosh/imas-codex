"""Tests for the check that a name and its definition say the same thing.

The check reads the definition and the name against each other and reports the
qualifier family the definition states and the name does not. These tests hold
three things: that a planted qualifier is found, that a name which states the
qualifier is not flagged for stating it once more, and that a name the check
flags is not accepted by review without a recorded reason.
"""

from __future__ import annotations

import pytest

from imas_codex.standard_names.audits import name_definition_agreement_check
from imas_codex.standard_names.review.checks import load_review_checks
from imas_codex.standard_names.review.jev_review import (
    aggregate_docs_review,
    build_docs_audit_report,
    build_docs_state,
    claim_binding_key,
    dimension_question_name,
    docs_question_table,
)

CHECK_ID = "definition_states_qualifier_name_lacks"

# A quantity whose definition normalises it while the name does not say so.
NORMALISED_CANDIDATE = {
    "id": "momentum_flux_due_to_e_cross_b_drift",
    "unit": "1",
    "description": (
        "Momentum flux due to E cross B drift is the scalar transport of a "
        "selected component of linear plasma momentum across an oriented "
        "surface by the electric-field-cross-magnetic-field drift, expressed in "
        "dimensionless form."
    ),
    "documentation": (
        "Momentum flux due to E cross B drift is the scalar transport of a "
        "selected component of linear plasma momentum across an oriented "
        "surface, expressed in dimensionless form."
    ),
    "source_paths": ["core_profiles/profiles_1d/momentum_flux_norm"],
}

# The same quantity with the normalisation stated in the name.
NAMED_CANDIDATE = {
    **NORMALISED_CANDIDATE,
    "id": "normalized_momentum_flux_due_to_e_cross_b_drift",
}


def test_the_planted_normalisation_is_found() -> None:
    findings = name_definition_agreement_check(NORMALISED_CANDIDATE)
    assert findings, "the planted normalised quantity was not flagged"
    assert any("unstated-normalisation" in finding for finding in findings)


def test_a_name_that_states_the_normalisation_is_not_flagged() -> None:
    assert name_definition_agreement_check(NAMED_CANDIDATE) == []


def test_an_equivalent_electron_convention_is_found() -> None:
    findings = name_definition_agreement_check(
        {
            "id": "accumulated_ammonia_count",
            "description": (
                "Cumulative inventory of ammonia molecules, expressed as an "
                "equivalent-electron count with ten count units per molecule."
            ),
            "documentation": "",
        }
    )
    assert any("equivalent-electron-count" in finding for finding in findings)


def test_a_name_with_no_definition_raises_nothing() -> None:
    assert name_definition_agreement_check({"id": "plasma_current"}) == []


def _conforming_answers(state, checks, claims, bindings):
    """Every question answered in the conforming form.

    The review decision this produces is the one a reviewer gives when nothing
    is wrong: every rubric sits in its accepted category and every claim is
    supported. Anything that still rejects is rejecting for the reason under
    test.
    """

    table = docs_question_table(checks, claims, bindings)
    answers: dict[str, dict] = {}
    from imas_codex.standard_names.review.jev_review import _DOCS_DIMENSION_CRITERIA

    for dimension, criteria in _DOCS_DIMENSION_CRITERIA.items():
        key = dimension_question_name(dimension)
        if key in table:
            answers[key] = {"choice": next(iter(criteria))}
    for key in table:
        if key.startswith("claim:"):
            answers[key] = {"choice": "supports"}
    return answers


@pytest.fixture(scope="module")
def checks():
    return load_review_checks()[0]


def _review(candidate, checks):
    state = build_docs_state(candidate, checks=checks)
    report = build_docs_audit_report(
        {**candidate, "id": candidate["id"]}, catalog_names=[], known_ids=set()
    )
    claims = state["documentation_claims"]
    bindings = state["binding_ids"]
    answers = _conforming_answers(state, checks, claims, bindings)
    for _ in claims:
        answers[claim_binding_key(0, 0)] = {"choice": "supports"}
    return aggregate_docs_review(state, checks, answers, audit_report=report)


def test_a_flagged_name_is_refused_without_a_recorded_reason(checks) -> None:
    """A normalised quantity planted under an unnormalised name is refused.

    The negative control is the same candidate whose name states the
    normalisation: identical definition, identical unit, identical answers.
    Only the name's qualifier differs, so the refusal is the check's and not
    the scaffolding's.
    """

    refused = _review(NORMALISED_CANDIDATE, checks)
    assert refused["checks"][CHECK_ID]["audit_finding"], (
        "the check recorded no finding for the planted name"
    )
    assert refused["checks"][CHECK_ID]["failed"]
    assert refused["decision"] == "reject"

    accepted = _review(NAMED_CANDIDATE, checks)
    assert accepted["checks"][CHECK_ID]["audit_finding"] is None
    assert not accepted["checks"][CHECK_ID]["failed"]
    refused_failed = {
        check for check, record in refused["checks"].items() if record["failed"]
    }
    accepted_failed = {
        check for check, record in accepted["checks"].items() if record["failed"]
    }
    assert refused_failed == accepted_failed | {CHECK_ID}, (
        "the only check that stops failing when the name states the "
        f"normalisation must be this one: refused={refused_failed} "
        f"accepted={accepted_failed}"
    )
