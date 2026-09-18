"""The audit walk and the validation worker answer different questions.

Two surfaces are read as "is this name quarantined?".  The audit-set walk
reports which deterministic checks fire for the fields a caller supplies and
holds no verdict state; the validation worker writes the stored verdict
(``validation_status`` plus its ``validated_at`` observation time).  The worker
reads strictly more, so a clean audit report is not evidence of a live name.

These tests hold the pair to the measure: the walk answers its own question and
refuses the other, and the refusal is not decoration — the signals the worker
quarantines on are not fields the walk ever reads.
"""

from __future__ import annotations

import pytest

from imas_codex.standard_names.review.audits import (
    AUDIT_WALK_QUESTION,
    QUARANTINE_AUTHORITY,
    QUARANTINE_QUESTION,
    QuarantineVerdictNotAnswered,
    answer_quarantine_question,
    run_all_audits,
)
from imas_codex.standard_names.workers import _is_quarantined


def test_each_single_question_is_answered_once() -> None:
    """The walk names its own question and declares the one it refuses."""
    report = run_all_audits([])

    assert report.answers_question == AUDIT_WALK_QUESTION
    assert report.refuses_question == QUARANTINE_QUESTION
    assert report.answers_question != report.refuses_question
    assert "quarantine" in report.refuses_question.lower()
    assert "quarantine" not in report.answers_question.lower()


def test_the_walk_refuses_to_settle_a_quarantine_verdict() -> None:
    """Asked the other instrument's question, the walk refuses and names it."""
    name_id = "radial_outline_of_plasma_boundary"

    with pytest.raises(QuarantineVerdictNotAnswered) as excinfo:
        answer_quarantine_question(name_id)

    message = str(excinfo.value)
    assert name_id in message
    assert QUARANTINE_AUTHORITY in message
    assert QUARANTINE_QUESTION in message


def test_the_verdict_predicate_reads_signals_the_walk_cannot_see() -> None:
    """A clean audit report is not a live verdict, shown for one name.

    The worker quarantines on an ISN ERROR-level semantic issue.  The walk
    never reads an issue string of that shape — that signal arrives on the
    worker's own layer-summary path — so the walk cannot return the same
    verdict, and declaring its question is the alternative the measure allows.
    """
    issues = ["[semantic] ERROR - 'outline' does not name the represented entity"]

    assert _is_quarantined(issues, {}) is True

    report = run_all_audits([])
    assert report.refuses_question == QUARANTINE_QUESTION
    assert not hasattr(report, "validation_status")
    assert not hasattr(report, "validated_at")
