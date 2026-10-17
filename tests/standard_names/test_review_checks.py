"""Tests for the named review check taxonomy.

The taxonomy is data (``imas_codex/llm/prompts/sn/review_checks.yaml``); the
loader validates its shape. These tests hold the three invariants the taxonomy
is accepted against: every check is decided exactly once (a Jev question or an
audit finding class, never both or neither), every check's evidence fields are
actually carried by the review state, and the human edit reasons map to the
checks at a share of at least four fifths.
"""

from __future__ import annotations

import pytest

from imas_codex.standard_names.review.checks import (
    AUDIT_FINDING_CLASSES,
    DOCS_DIMENSIONS,
    NAME_DIMENSIONS,
    JevQuestion,
    JevQuestionKind,
    ReviewCheck,
    load_review_checks,
    review_state_fields,
)

MIN_MAPPED_SHARE = 0.80


def fields_not_carried(checks: list[ReviewCheck]) -> list[str]:
    """Evidence fields named by ``checks`` that the review state does not carry."""

    carried = review_state_fields()
    return sorted(
        {
            field
            for check in checks
            for field in check.evidence_fields
            if field not in carried
        }
    )


@pytest.fixture(scope="module")
def loaded() -> tuple[list[ReviewCheck], list]:
    return load_review_checks()


def test_every_check_is_decided_exactly_once(loaded) -> None:
    checks, _ = loaded
    assert checks, "the taxonomy names no checks"
    for check in checks:
        has_question = check.question is not None
        has_audit = check.audit_finding_class is not None
        assert has_question != has_audit, (
            f"{check.id} must carry exactly one of a Jev question or an audit "
            "finding class"
        )


def test_repeated_segment_uses_audit_class(loaded) -> None:
    checks, _ = loaded
    check = next(check for check in checks if check.id == "redundant_repeated_segment")
    assert check.question is None
    assert check.audit_finding_class == "convention_violation"


def test_every_check_reads_evidence_and_declares_retrieval(loaded) -> None:
    checks, _ = loaded
    for check in checks:
        assert check.evidence_fields, f"{check.id} reads no evidence field"
        assert isinstance(check.retrieval, bool), f"{check.id} retrieval is not a flag"


def test_evidence_fields_are_carried_by_the_review_state(loaded) -> None:
    checks, _ = loaded
    missing = fields_not_carried(checks)
    assert missing == [], (
        f"checks name evidence fields the review state does not carry: {missing}"
    )


def test_negative_control_unplanted_field_is_rejected() -> None:
    """Plant a check naming a field the review state lacks; the test must fail."""

    planted = ReviewCheck(
        id="unplanted_evidence_field",
        axis="name",
        dimension="grammar",
        evidence_fields=["field_absent_from_review_state"],
        retrieval=False,
        question=JevQuestion(
            kind=JevQuestionKind.NOUL, text="Does this planted field exist?"
        ),
    )
    assert fields_not_carried([planted]) == ["field_absent_from_review_state"]


def _write_taxonomy(tmp_path, question_block: str):
    fixture = tmp_path / "checks.yaml"
    fixture.write_text(
        "checks:\n"
        "- id: planted_check\n"
        "  axis: name\n"
        "  dimension: grammar\n"
        "  evidence_fields:\n"
        "  - physical_base\n"
        "  retrieval: false\n"
        "  question:\n" + question_block,
        encoding="utf-8",
    )
    return fixture


def test_loader_refuses_a_noul_without_the_polarity_field(tmp_path) -> None:
    """A Noul that omits ``true_case`` is refused, not read by default.

    The reader fails a Noul on a high probability, so a Noul that does not
    declare its true case is the defect would be read backwards silently. The
    loader is the boundary that refuses it.
    """

    fixture = _write_taxonomy(
        tmp_path,
        "    kind: noul\n    text: Does the planted check carry the polarity field?\n",
    )
    with pytest.raises(ValueError, match="true_case"):
        load_review_checks(fixture)


def test_loader_refuses_a_noul_with_any_polarity_but_the_defect(tmp_path) -> None:
    fixture = _write_taxonomy(
        tmp_path,
        "    kind: noul\n    true_case: acceptable\n"
        "    text: Does the planted check carry the polarity field?\n",
    )
    with pytest.raises(ValueError, match="true_case"):
        load_review_checks(fixture)


def test_loader_accepts_a_noul_that_declares_the_defect_polarity(tmp_path) -> None:
    fixture = _write_taxonomy(
        tmp_path,
        "    kind: noul\n    true_case: defect\n"
        "    text: Does the planted check carry the polarity field?\n",
    )
    checks, _ = load_review_checks(fixture)
    assert checks[0].question.true_case == "defect"


def test_every_shipped_noul_declares_the_defect_polarity(loaded) -> None:
    checks, _ = loaded
    nouls = [
        check
        for check in checks
        if check.question is not None and check.question.kind is JevQuestionKind.NOUL
    ]
    assert nouls, "the taxonomy ships no Noul questions"
    for check in nouls:
        assert check.question.true_case == "defect", (
            f"{check.id} Noul true_case is {check.question.true_case!r}"
        )


def test_check_ids_are_unique(loaded) -> None:
    checks, _ = loaded
    ids = [check.id for check in checks]
    assert len(ids) == len(set(ids))


def test_axis_and_dimension_are_consistent(loaded) -> None:
    checks, _ = loaded
    for check in checks:
        assert check.axis in {"name", "docs"}, f"{check.id} has axis {check.axis!r}"
        valid = NAME_DIMENSIONS if check.axis == "name" else DOCS_DIMENSIONS
        assert check.dimension in valid, (
            f"{check.id} dimension {check.dimension!r} is not a {check.axis} dimension"
        )


def test_audit_finding_classes_are_emitted(loaded) -> None:
    checks, _ = loaded
    for check in checks:
        if check.audit_finding_class is not None:
            assert check.audit_finding_class in AUDIT_FINDING_CLASSES


def test_human_reasons_map_at_least_four_fifths(loaded) -> None:
    checks, reasons = loaded
    assert reasons, "no human edit reasons were mapped"
    known_ids = {check.id for check in checks}
    mapped = [entry for entry in reasons if entry.check_id is not None]
    share = len(mapped) / len(reasons)
    assert share >= MIN_MAPPED_SHARE, (
        f"only {share:.1%} of {len(reasons)} human reasons map to a check"
    )
    for entry in reasons:
        if entry.check_id is not None:
            assert entry.check_id in known_ids, (
                f"reason {entry.reason!r} maps to unknown check {entry.check_id!r}"
            )
        else:
            assert entry.unmapped_reason, (
                f"reason {entry.reason!r} is unmapped with no stated reason"
            )
