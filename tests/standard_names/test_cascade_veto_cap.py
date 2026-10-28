"""A documentation contradiction never leaves an acceptable decision score.

The veto decides the route on its own, but the decision score is what a
score-reading consumer compares against its own threshold. A vetoed document
whose score stays at the unvetoed mapping reads as acceptable everywhere except
in the route field, so the cap is part of the veto rather than a display detail.
"""

from __future__ import annotations

import math
from typing import Any

from imas_codex.standard_names.defaults import DEFAULT_MIN_SCORE
from imas_codex.standard_names.review import cascade, jev_review
from imas_codex.standard_names.review.audits import AuditReport
from imas_codex.standard_names.review.checks import load_review_checks

PLACEMENT = "equilibrium/time_slice/profiles_1d/poloidal_mode_number"
CLAIM = "The quantity is the poloidal mode number of the flux surface."

# The rubric is not the subject here, so the fit names one dimension and one
# check: the claim the veto reads. The mapping under test is the real piecewise
# map against the real consumer threshold.
MINIMAL_THRESHOLDS = {
    "score_kind": "logistic",
    "accept_threshold": 0.5,
    "rubric": {
        "offset": 0.0,
        "dimensions": {
            "physics_accuracy": {
                "intercept": 0.0,
                "weights": {"claim_contradicts_source": 1.0},
            }
        },
    },
}


def _item() -> dict[str, Any]:
    return {
        "id": "poloidal_mode_number",
        "documentation": CLAIM,
        "description": "Poloidal mode number of the flux surface.",
        "dd_source_docs": [
            {"id": PLACEMENT, "documentation": "Poloidal mode number."},
        ],
    }


def _vetoed_verdict() -> dict[str, Any]:
    checks = [check for check in load_review_checks()[0] if check.axis == "docs"]
    state = jev_review.build_docs_state(_item(), checks=checks)
    binding_ids = state["binding_ids"]
    keys = (
        [
            jev_review.claim_binding_key(0, position)
            for position in range(len(binding_ids))
        ]
        if len(binding_ids) > 1
        else [jev_review.check_question_name("claim_contradicts_source:0")]
    )
    answers = {
        key: {"choice": "contradicts", "probabilities": {"contradicts": 1.0}}
        for key in keys
    }
    verdict = jev_review.aggregate_docs_review(
        state, checks, answers, audit_report=AuditReport()
    )
    assert verdict["contradiction_veto"] is True
    return verdict


def test_a_vetoed_document_decision_score_stays_below_the_accept_threshold():
    route = cascade._route(
        "docs", _vetoed_verdict(), 0.99, thresholds=MINIMAL_THRESHOLDS
    )
    unvetoed = cascade.map_decision_score(
        0.99, MINIMAL_THRESHOLDS["accept_threshold"], DEFAULT_MIN_SCORE
    )
    assert unvetoed > DEFAULT_MIN_SCORE
    assert route.route == "refine"
    assert route.decision_score < DEFAULT_MIN_SCORE
    assert route.decision_score == math.nextafter(DEFAULT_MIN_SCORE, 0.0)


def test_a_vetoed_document_already_below_the_threshold_is_not_raised():
    route = cascade._route(
        "docs", _vetoed_verdict(), 0.1, thresholds=MINIMAL_THRESHOLDS
    )
    assert route.route == "refine"
    assert route.decision_score < DEFAULT_MIN_SCORE


def test_an_unvetoed_document_at_the_same_score_is_acceptable():
    judgment = {**_vetoed_verdict(), "contradiction_veto": False}
    route = cascade._route("docs", judgment, 0.99, thresholds=MINIMAL_THRESHOLDS)
    assert route.route == "accept"
    assert route.decision_score >= DEFAULT_MIN_SCORE
