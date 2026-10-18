"""Documentation claim questions and their deterministic review decision."""

from __future__ import annotations

import pytest

from imas_codex.standard_names.context import conventions_context_for
from imas_codex.standard_names.docs_gates import HouseStyleFindingClass
from imas_codex.standard_names.review import jev_review
from imas_codex.standard_names.review.audits import AuditReport, LintFinding
from imas_codex.standard_names.review.checks import load_review_checks


@pytest.mark.parametrize(
    ("documentation", "expected"),
    [
        (
            "The current is positive. The field is toroidal.",
            ["The current is positive.", "The field is toroidal."],
        ),
        (
            "The current is positive and the field is toroidal.",
            ["The current is positive and the field is toroidal."],
        ),
        (
            "The current and field are measured together.",
            ["The current and field are measured together."],
        ),
    ],
)
def test_claim_splitter_keeps_each_sentence_as_one_claim(documentation, expected):
    assert jev_review.split_documentation_claims(documentation) == expected


@pytest.mark.parametrize(
    ("documentation", "expected"),
    [
        (
            "The density is negative [related](name:argon_density).",
            ["The density is negative related."],
        ),
        ("The density is negative $n<0$.", ["The density is negative [equation]."]),
    ],
)
def test_inline_markup_keeps_surrounding_claim(documentation, expected):
    assert jev_review.split_documentation_claims(documentation) == expected


def test_declarative_sentence_with_unlisted_verb_is_a_claim():
    assert jev_review.split_documentation_claims("The pressure peaks at the axis.") == [
        "The pressure peaks at the axis."
    ]


def test_coordinated_noun_phrase_stays_one_claim():
    assert jev_review.split_documentation_claims(
        "The density of hydrogen and the density of deuterium are measured together."
    ) == ["The density of hydrogen and the density of deuterium are measured together."]


def test_inline_math_keeps_sentence_with_equation_placeholder():
    assert jev_review.split_documentation_claims("The relation is $n = N/V$.") == [
        "The relation is [equation]."
    ]
    assert jev_review.split_documentation_claims("The density is $n<0$.") == [
        "The density is [equation]."
    ]


def test_claim_splitter_excludes_registry_text_and_equations_but_keeps_prose():
    opening = "Number density of argon"
    note = "This form distinguishes the cold population."
    locus = "The plasma boundary is the last closed flux surface."
    documentation = (
        f"{opening}\n\n{note}\n\n"
        "The density is measured locally and the profile is radial. "
        "The relation is $n = N/V$.\n\n"
        "$$n = N/V$$\n\n"
        "The related quantity is [argon density](name:argon_density).\n\n"
        f"{locus}\n\n"
        "The density increases at the edge."
    )
    assert jev_review.split_documentation_claims(
        documentation, registry_sentences=[opening, note, locus]
    ) == [
        "The density is measured locally and the profile is radial.",
        "The relation is [equation].",
        "The related quantity is argon density.",
        "The density increases at the edge.",
    ]


def test_docs_state_carries_source_and_description_without_review_comments():
    state = jev_review.build_docs_state(
        {
            "id": "argon_density",
            "documentation": "The argon density is positive.",
            "description": "Number density of argon.",
            "dd_source_docs": [{"path": "dd:argon", "documentation": "Argon density."}],
            "source_paths": ["dd:argon"],
            "reviewer_comments": "The incumbent disagreed.",
            "reviewer_comments_per_dim": {"physics_accuracy": "not grounded"},
            "conventions": {},
        }
    )
    assert state["documentation_claims"] == ["The argon density is positive."]
    assert state["description"] == "Number density of argon."
    assert state["dd_source_docs"]["value"][0]["documentation"] == "Argon density."
    assert "reviewer_comments" not in str(state)
    assert "incumbent" not in str(state)


def test_docs_state_carries_every_taxonomy_evidence_field():
    checks, _ = load_review_checks()
    state = jev_review.build_docs_state(
        {"id": "argon_density", "conventions": {}}, checks=checks
    )
    required = {
        field
        for check in checks
        if check.axis == "docs"
        for field in check.evidence_fields
    }
    assert required <= (set(state) | {"id"})
    assert state["dd_source_docs"] == {"observed": False, "value": None}


def test_docs_state_excludes_selected_registry_opening():
    state = jev_review.build_docs_state(
        {
            "id": "flux_surface_averaged_argon_density_at_plasma_boundary",
            "documentation": "Number density of argon\n\nThe density is positive.",
        }
    )
    assert state["documentation_claims"] == ["The density is positive."]


def test_docs_state_excludes_rendered_template_note():
    conventions = conventions_context_for(
        "flux_surface_averaged_argon_density_at_plasma_boundary"
    )
    conventions["templates"][0]["note"] = "The {subject} population is distinct."
    state = jev_review.build_docs_state(
        {
            "id": "flux_surface_averaged_argon_density_at_plasma_boundary",
            "documentation": (
                "Number density of argon\n\n"
                "The argon population is distinct.\n\n"
                "The density is positive."
            ),
            "conventions": conventions,
        }
    )
    assert state["documentation_claims"] == ["The density is positive."]


def test_question_table_asks_three_categories_and_one_choice_per_claim():
    checks, _ = load_review_checks()
    table = jev_review.docs_question_table(checks, ["one", "two"])
    for dimension in (
        "description_quality",
        "documentation_quality",
        "completeness",
    ):
        question = table[jev_review.dimension_question_name(dimension)]
        assert question["type"] == "choice"
        assert len(question["criteria"]) == 3
        assert all(len(text.split()) > 3 for text in question["criteria"].values())
    claim_keys = [
        key for key in table if key.startswith("check:claim_contradicts_source:")
    ]
    assert claim_keys == [
        "check:claim_contradicts_source:0",
        "check:claim_contradicts_source:1",
    ]
    assert all(
        set(table[key]["criteria"]) == {"supports", "contradicts", "says nothing"}
        for key in claim_keys
    )
    assert "check:claim_contradicts_source" not in table


def test_claim_choice_options_follow_taxonomy_order():
    checks, _ = load_review_checks()
    reordered = [
        check.model_copy(
            update={
                "question": check.question.model_copy(
                    update={"options": ["says nothing", "contradicts", "supports"]}
                )
            }
        )
        if check.id == "claim_contradicts_source"
        else check
        for check in checks
    ]
    question = jev_review.docs_question_table(reordered, ["The density is positive."])[
        "check:claim_contradicts_source:0"
    ]
    assert list(question["criteria"]) == ["says nothing", "contradicts", "supports"]
    assert "Any part" in question["criteria"]["contradicts"]


def _top_answers(table):
    acceptable = {
        "dimension:description_quality": "precise_definition",
        "dimension:documentation_quality": "normative_definition",
        "dimension:completeness": "complete",
        "check:description_false_of_some_binding": "true of all",
    }
    answers = {}
    for key, question in table.items():
        if question["type"] == "choice":
            choice = (
                "supports"
                if key.startswith("check:claim_contradicts_source:")
                else acceptable[key]
            )
            assert choice in question["criteria"]
            answers[key] = {"choice": choice}
        else:
            answers[key] = {"noul": 0.0}
    return answers


def test_one_contradiction_vetoes_top_scores_and_clean_checks():
    checks, _ = load_review_checks()
    state = jev_review.build_docs_state(
        {
            "id": "argon_density",
            "documentation": "The argon density is negative.",
            "description": "Number density of argon.",
            "dd_source_docs": [{"documentation": "Number density is nonnegative."}],
            "conventions": {},
        }
    )
    assert len(state["documentation_claims"]) == 1
    answers = _top_answers(
        jev_review.docs_question_table(checks, state["documentation_claims"])
    )
    answers["check:claim_contradicts_source:0"] = {"choice": "contradicts"}
    result = jev_review.aggregate_docs_review(
        state, checks, answers, audit_report=AuditReport()
    )
    assert set(result["dimensions"].values()) == {"accept"}
    assert result["checks"]["claim_contradicts_source"]["failed"]
    assert not any(
        record["failed"]
        for check_id, record in result["checks"].items()
        if check_id != "claim_contradicts_source"
    )
    assert result["contradiction_veto"] is True
    assert result["decision"] == "reject"
    answers["check:claim_contradicts_source:0"] = {"choice": "supports"}
    assert (
        jev_review.aggregate_docs_review(
            state, checks, answers, audit_report=AuditReport()
        )["decision"]
        == "accept"
    )


def test_material_contradiction_probability_vetoes_ambiguous_support():
    checks, _ = load_review_checks()
    state = jev_review.build_docs_state(
        {
            "id": "argon_density",
            "documentation": "The argon density is negative.",
            "description": "Number density of argon.",
            "conventions": {},
        }
    )
    answers = _top_answers(
        jev_review.docs_question_table(checks, state["documentation_claims"])
    )
    claim = "check:claim_contradicts_source:0"
    answers[claim] = {
        "choice": "supports",
        "probabilities": {"supports": 0.44, "contradicts": 0.32, "says nothing": 0.24},
    }
    report = AuditReport()
    verdict = jev_review.aggregate_docs_review(
        state, checks, answers, audit_report=report
    )
    assert verdict["contradiction_veto"]
    assert state["documentation_claims"][0] in verdict["contradiction_reason"]
    assert verdict["raw_claim_answers"][0]["choice"] == "supports"
    assert verdict["claim_answers"][0]["choice"] == "contradicts"
    assert (
        jev_review.score_docs_review(state, checks, answers, audit_report=report) == 0.0
    )
    answers[claim]["probabilities"]["contradicts"] = 0.04
    assert not jev_review.aggregate_docs_review(
        state, checks, answers, audit_report=report
    )["contradiction_veto"]


def test_docs_score_requires_audit_evidence_before_scoring_or_veto():
    checks, _ = load_review_checks()
    state = jev_review.build_docs_state(
        {
            "id": "argon_density",
            "documentation": "The argon density is positive.",
            "conventions": {},
        }
    )
    answers = _top_answers(
        jev_review.docs_question_table(checks, state["documentation_claims"])
    )
    assert (
        jev_review.score_docs_review(state, checks, answers, audit_report=None) is None
    )
    answers["check:claim_contradicts_source:0"] = {"choice": "contradicts"}
    assert (
        jev_review.score_docs_review(state, checks, answers, audit_report=None) is None
    )


def test_unanswered_claim_is_unknown_and_unsupported_claim_is_rejected():
    checks, _ = load_review_checks()
    state = jev_review.build_docs_state(
        {
            "id": "argon_density",
            "documentation": "The density is positive.",
            "conventions": {},
        }
    )
    answers = _top_answers(
        jev_review.docs_question_table(checks, state["documentation_claims"])
    )
    key = "check:claim_contradicts_source:0"
    answers.pop(key)
    assert (
        jev_review.aggregate_docs_review(
            state, checks, answers, audit_report=AuditReport()
        )["decision"]
        == "unknown"
    )
    answers[key] = {"choice": "says nothing"}
    assert (
        jev_review.aggregate_docs_review(
            state, checks, answers, audit_report=AuditReport()
        )["checks"]["claim_contradicts_source"]["failure_probability"]
        == 1.0
    )
    assert (
        jev_review.aggregate_docs_review(
            state, checks, answers, audit_report=AuditReport()
        )["decision"]
        == "reject"
    )


def test_every_house_style_finding_has_a_named_check_and_is_read_from_audit():
    checks, _ = load_review_checks()
    for finding_class in HouseStyleFindingClass:
        matching = [
            check
            for check in checks
            if check.axis == "docs" and check.audit_finding_class == finding_class.value
        ]
        assert len(matching) == 1, finding_class.value
        finding = LintFinding(
            name_id="argon_density",
            finding_type="convention_violation",
            detail=f"{finding_class.value}: registry wording differs [concept:argon]",
        )
        report = AuditReport(lint_findings=[finding])
        assert jev_review.audit_finding_for(matching[0], "argon_density", report)
        assert (
            jev_review.audit_finding_for(matching[0], "other_density", report) is None
        )


def test_house_style_finding_lowers_score_with_identical_judgments():
    checks, _ = load_review_checks()
    name_id = "flux_surface_averaged_argon_density_at_plasma_boundary"
    clean = {
        "id": name_id,
        "description": "Number density of argon.",
        "documentation": "Number density of argon\n\nThe density is positive.",
        "conventions": conventions_context_for(name_id),
    }
    flawed = {**clean, "documentation": "Wrong opening. The density is positive."}
    clean_report = jev_review.build_docs_audit_report(
        clean, catalog_names=[clean], known_ids={name_id}
    )
    flawed_report = jev_review.build_docs_audit_report(
        flawed, catalog_names=[flawed], known_ids={name_id}
    )
    target = "description_template_opening_mismatch"
    check = next(check for check in checks if check.id == target)
    assert jev_review.audit_finding_for(check, name_id, clean_report) is None
    assert jev_review.audit_finding_for(check, name_id, flawed_report)
    state = jev_review.build_docs_state(clean, checks=checks)
    answers = _top_answers(
        jev_review.docs_question_table(checks, state["documentation_claims"])
    )
    clean_score = jev_review.score_docs_review(
        state, checks, answers, audit_report=clean_report
    )
    flawed_score = jev_review.score_docs_review(
        state, checks, answers, audit_report=flawed_report
    )
    assert flawed_score < clean_score
