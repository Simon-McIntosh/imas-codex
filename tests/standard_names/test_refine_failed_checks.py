"""Checks that drive retrieval and refinement instructions."""

from __future__ import annotations

import json

import pytest
import yaml

from imas_codex.llm.prompt_loader import render_prompt
from imas_codex.standard_names import benchmark_roles
from imas_codex.standard_names.fanout.config import FanoutSettings
from imas_codex.standard_names.fanout.trigger import (
    should_trigger_fanout,
    should_trigger_fanout_from_checks,
)
from imas_codex.standard_names.review.checks import (
    JevQuestion,
    ReviewCheck,
    load_review_checks,
)
from imas_codex.standard_names.review.jev_review import check_failure_probability


def _question_check(*, retrieval: bool) -> ReviewCheck:
    return ReviewCheck(
        id="source_binding",
        axis="name",
        dimension="semantic",
        evidence_fields=["source_paths"],
        retrieval=retrieval,
        question=JevQuestion(
            kind="choice",
            text="Which source object does this name denote?",
            options=["different", "bound", "uncertain"],
            conforming=["bound"],
        ),
    )


@pytest.mark.parametrize("conforming", [None, [], ["absent"]])
def test_loader_refuses_choice_without_valid_conforming_options(tmp_path, conforming):
    question = {
        "kind": "choice",
        "text": "Which source object does this name denote?",
        "options": ["bound", "different"],
    }
    if conforming is not None:
        question["conforming"] = conforming
    fixture = tmp_path / "checks.yaml"
    fixture.write_text(
        yaml.safe_dump(
            {
                "checks": [
                    {
                        "id": "source_binding",
                        "axis": "name",
                        "dimension": "semantic",
                        "evidence_fields": ["source_paths"],
                        "retrieval": True,
                        "question": question,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="conforming"):
        load_review_checks(fixture)


def test_loader_refuses_conforming_list_on_noul(tmp_path):
    fixture = tmp_path / "checks.yaml"
    fixture.write_text(
        yaml.safe_dump(
            {
                "checks": [
                    {
                        "id": "wrong_source",
                        "axis": "name",
                        "dimension": "semantic",
                        "evidence_fields": ["source_paths"],
                        "retrieval": True,
                        "question": {
                            "kind": "noul",
                            "true_case": "defect",
                            "text": "Is the source wrong?",
                            "conforming": ["no"],
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Noul.*conforming"):
        load_review_checks(fixture)


def test_failure_probability_reads_each_check_kind():
    choice = _question_check(retrieval=True)
    assert check_failure_probability(
        choice,
        {
            "choice": "bound",
            "probabilities": {
                "different": 0.2,
                "bound": 0.5,
                "uncertain": 0.3,
            },
        },
    ) == pytest.approx(0.5)
    noul = ReviewCheck(
        id="wrong_source",
        axis="name",
        dimension="semantic",
        evidence_fields=["source_paths"],
        retrieval=True,
        question=JevQuestion(
            kind="noul", text="Is the source wrong?", true_case="defect"
        ),
    )
    assert check_failure_probability(noul, {"noul": 0.73}) == pytest.approx(0.73)
    audit = ReviewCheck(
        id="invalid_render",
        axis="name",
        dimension="grammar",
        evidence_fields=["id"],
        retrieval=False,
        audit_finding_class="round_trip_failure",
    )
    assert check_failure_probability(audit, audit_finding="invalid") == 1.0
    assert check_failure_probability(audit, audit_finding=None) == 0.0


def test_unverifiable_documentation_claim_is_a_failure():
    checks, _ = load_review_checks()
    claim = next(check for check in checks if check.id == "claim_contradicts_source")
    assert claim.question.conforming == ["supports"]
    answer = {
        "choice": "says nothing",
        "probabilities": {"supports": 0.1, "contradicts": 0.1, "says nothing": 0.8},
    }
    assert check_failure_probability(claim, answer) == pytest.approx(0.9)


def test_shipped_choice_conforming_forms_are_explicit():
    checks, _ = load_review_checks()
    choices = {
        check.id: check.question.conforming
        for check in checks
        if check.question is not None and check.question.kind == "choice"
    }
    assert choices == {
        "bound_to_wrong_object": ["the bound source object"],
        "collapsed_two_quantities": ["one quantity"],
        "description_false_of_some_binding": ["true of all"],
        "claim_contradicts_source": ["supports"],
    }


def test_check_trigger_reads_only_retrieval_failures():
    relevant = _question_check(retrieval=True)
    irrelevant = _question_check(retrieval=False).model_copy(
        update={"id": "grammar_only"}
    )
    results = {
        relevant.id: {"answered": True, "failure_probability": 0.81},
        irrelevant.id: {"answered": True, "failure_probability": 0.99},
    }
    assert should_trigger_fanout_from_checks(
        [relevant, irrelevant], results, threshold=0.8
    )
    results[relevant.id]["failure_probability"] = 0.79
    assert not should_trigger_fanout_from_checks(
        [relevant, irrelevant], results, threshold=0.8
    )
    assert not should_trigger_fanout_from_checks([relevant], {}, threshold=0.0)


def test_check_trigger_reads_only_check_results():
    check = _question_check(retrieval=True)
    fire, excerpt = should_trigger_fanout(
        checks=[check],
        check_results={check.id: {"answered": True, "failure_probability": 0.9}},
        check_threshold=0.8,
    )
    assert fire is True
    assert excerpt == ""


def test_check_trigger_threshold_default_remains():
    settings = FanoutSettings()
    assert settings.refine_trigger_check_threshold == 0.5


@pytest.mark.parametrize(
    ("prompt_name", "context"),
    [
        (
            "sn/refine_name_user",
            {
                "item": {
                    "id": "electron_temperature",
                    "path": "core_profiles/profiles_1d/electrons/temperature",
                    "reviewer_comments_per_dim_name": json.dumps(
                        {"semantic": "CURRENT_COMMENT"}
                    ),
                },
                "chain_history": [
                    {
                        "name": "electron_temperature",
                        "model": "reviewer",
                        "reviewer_score": 0.4,
                        "reviewer_comments_per_dim": {"semantic": "ANCESTOR_COMMENT"},
                    }
                ],
                "chain_length": 1,
            },
        ),
        (
            "sn/refine_docs_user",
            {
                "sn_name": "electron_temperature",
                "reviewer_comments_per_dim_docs": json.dumps(
                    {"physics": "CURRENT_COMMENT"}
                ),
                "docs_chain_history": [
                    {
                        "documentation": "Old documentation",
                        "model": "reviewer",
                        "reviewer_score": 0.4,
                        "reviewer_comments_per_dim": {"physics": "ANCESTOR_COMMENT"},
                    }
                ],
                "docs_chain_length": 1,
            },
        ),
    ],
)
def test_failed_check_instruction_replaces_current_comments(prompt_name, context):
    failed_check = {
        "id": "source_binding",
        "criterion": "Which source object does this name denote?",
        "evidence": "core_profiles/profiles_1d/electrons/temperature",
        "segment": "subject: electron",
    }
    rendered = render_prompt(prompt_name, {**context, "failed_checks": [failed_check]})
    for value in failed_check.values():
        assert value in rendered
    assert "CURRENT_COMMENT" not in rendered
    assert "ANCESTOR_COMMENT" in rendered


def test_checks_bench_case_without_checks_is_refused():
    with pytest.raises(ValueError, match="requires failed checks"):
        benchmark_roles.build_refine_prompt_context(
            {"failed_checks": []}, {}, [], instruction_source="checks"
        )


def test_checks_bench_case_carries_its_instruction_list():
    case = {
        "prior_name": "electron_temperature",
        "prior_score": 0.4,
        "critique": {"semantic": "old comment"},
        "path": "core_profiles/profiles_1d/electrons/temperature",
        "description": "Electron temperature",
        "unit": "eV",
        "data_type": "FLT_0D",
        "physics_domain": "plasma",
        "failed_checks": [
            {
                "id": "bound_to_wrong_object",
                "criterion": "Which object does the name denote, read against its bound source path?",
                "evidence": "core_profiles/profiles_1d/electrons/temperature",
                "segment": "subject: electron",
            }
        ],
    }
    context = benchmark_roles.build_refine_prompt_context(
        case, {}, [], instruction_source="checks"
    )
    assert context["failed_checks"] == case["failed_checks"]
    assert "failed_checks" not in benchmark_roles.build_refine_prompt_context(
        case, {}, []
    )
