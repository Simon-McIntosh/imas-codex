"""Tests for the name-axis Jev review builders.

The judgement call is driven through the module's own seam: ``decide_batch``
and the sibling reranker are replaced, so no test opens the autouse-guarded
live decisions endpoint. The autouse guard in ``tests/conftest.py`` fails any
test that does.
"""

from __future__ import annotations

import textwrap

import pytest

from imas_codex.standard_names.review import jev_review
from imas_codex.standard_names.review.audits import AuditReport, LintFinding
from imas_codex.standard_names.review.checks import (
    JevQuestion,
    JevQuestionKind,
    ReviewCheck,
    load_review_checks,
)


def _noul(
    check_id: str,
    dimension: str,
    fields: list[str],
    text: str,
    true_case: str | None = None,
) -> ReviewCheck:
    return ReviewCheck(
        id=check_id,
        axis="name",
        dimension=dimension,
        evidence_fields=fields,
        retrieval=False,
        question=JevQuestion(kind=JevQuestionKind.NOUL, text=text, true_case=true_case),
    )


def _choice(
    check_id: str, dimension: str, fields: list[str], text: str, options: list[str]
) -> ReviewCheck:
    return ReviewCheck(
        id=check_id,
        axis="name",
        dimension=dimension,
        evidence_fields=fields,
        retrieval=False,
        question=JevQuestion(kind=JevQuestionKind.CHOICE, text=text, options=options),
    )


def _audit(check_id: str, dimension: str, fields: list[str], cls: str) -> ReviewCheck:
    return ReviewCheck(
        id=check_id,
        axis="name",
        dimension=dimension,
        evidence_fields=fields,
        retrieval=False,
        audit_finding_class=cls,
    )


def _small_checks() -> list[ReviewCheck]:
    """A census spanning the taxonomy's shapes: Noul, Choice, and audit-decided.

    The Choice lists the acceptable form first, which is the convention the
    taxonomy authors and the rule the code derives its defect from.
    """
    return [
        _noul(
            "malformed_physical_base",
            "grammar",
            ["physical_base", "source_paths"],
            "Is the physical base unregistered, malformed, or more than one token?",
            true_case="defect",
        ),
        _choice(
            "bound_to_wrong_object",
            "semantic",
            ["source_paths"],
            "Which object does the name denote?",
            ["the bound source object", "a different object"],
        ),
        _audit(
            "grammar_round_trip_failure",
            "grammar",
            ["id"],
            "round_trip_failure",
        ),
    ]


# ---------------------------------------------------------------------------
# The state carries every named field, and unknown is not empty
# ---------------------------------------------------------------------------


def test_state_carries_name_segments_kind_unit_description():
    item = {
        "id": "electron_temperature",
        "kind": "scalar",
        "unit": "eV",
        "description": "The electron temperature.",
        "documentation": "Measured by Thomson scattering.",
        "physical_base": "temperature",
        "subject": "electron",
        "coordinate": "radial",
    }
    state = jev_review.build_name_state(item)
    name = state["name"]
    assert name["id"] == "electron_temperature"
    assert name["kind"] == "scalar"
    assert name["unit"] == "eV"
    assert name["description"] == "The electron temperature."
    assert name["documentation"] == "Measured by Thomson scattering."
    assert name["segments"]["physical_base"] == "temperature"
    assert name["segments"]["subject"] == "electron"
    assert set(name["segments"]) == set(jev_review.SEGMENT_FIELDS)


def test_unobserved_neighbourhood_field_is_marked_unobserved_not_empty():
    state = jev_review.build_name_state({"id": "x"})
    assert state["neighbourhood"]["source_paths"] == {"observed": False, "value": None}


def test_observed_but_empty_neighbourhood_field_is_observed_and_empty():
    state = jev_review.build_name_state({"id": "x", "source_paths": []})
    assert state["neighbourhood"]["source_paths"] == {
        "observed": True,
        "value": [],
    }


def test_state_carries_source_docs_and_siblings_when_observed():
    item = {
        "id": "x",
        "source_paths": ["equilibrium/time_slice/profiles_1d/psi"],
        "dd_source_docs": ["Poloidal flux."],
        "sibling_names": ["poloidal_flux", "flux_surface_area"],
    }
    state = jev_review.build_name_state(item)
    nb = state["neighbourhood"]
    assert nb["source_paths"]["value"] == ["equilibrium/time_slice/profiles_1d/psi"]
    assert nb["dd_source_docs"]["observed"] is True
    assert nb["sibling_names"]["value"] == ["poloidal_flux", "flux_surface_area"]


def test_cocos_source_path_supplies_label_from_dd_reader():
    item = {
        "id": "poloidal_magnetic_flux",
        "source_paths": ["dd:equilibrium/time_slice/profiles_1d/psi"],
        "dd_source_docs": [
            {
                "path": "equilibrium/time_slice/profiles_1d/psi",
                "cocos_label": "psi_like",
            }
        ],
    }
    state = jev_review.build_name_state(item)
    assert state["neighbourhood"]["cocos_label"] == {
        "observed": True,
        "value": "psi_like",
    }


def test_state_carries_closed_base_roster_full_parse_and_parent_owner():
    state = jev_review.build_name_state(
        {
            "id": "poloidal_magnetic_flux",
            "parent_path": "equilibrium/time_slice/boundary/outline",
        }
    )
    assert "magnetic_flux" in state["neighbourhood"]["physical_base_roster"]["value"]
    assert state["name"]["decomposition"] is not None
    assert state["neighbourhood"]["owner_segment"]["value"] == "boundary"
    qualified = jev_review.build_name_state({"id": "total_plasma_heating_power"})
    assert {
        qualifier["token"]
        for qualifier in qualified["name"]["decomposition"]["qualifiers"]
    } >= {"total", "plasma"}


def test_state_carries_every_taxonomy_neighbourhood_field():
    checks, _ = load_review_checks()
    fields = jev_review.neighbourhood_fields(checks)
    state = jev_review.build_name_state({"id": "x"})
    assert set(fields) <= set(state["neighbourhood"])


def test_repeated_segment_is_decided_by_the_existing_audit():
    check = next(
        check
        for check in load_review_checks()[0]
        if check.id == "redundant_repeated_segment"
    )
    assert check.question is None
    assert jev_review.audit_finding_for(
        check, "electron_electron_temperature", AuditReport()
    )
    assert (
        jev_review.audit_finding_for(check, "electron_temperature", AuditReport())
        is None
    )


# ---------------------------------------------------------------------------
# The question table follows the taxonomy
# ---------------------------------------------------------------------------


def test_question_table_is_generated_from_the_checks_it_is_given(tmp_path):
    fixture = tmp_path / "checks.yaml"
    fixture.write_text(
        textwrap.dedent(
            """
            checks:
            - id: malformed_physical_base
              axis: name
              dimension: grammar
              evidence_fields: [physical_base]
              retrieval: false
              question:
                kind: noul
                true_case: defect
                text: Is the physical base unregistered, malformed, or more than one token?
            """
        ),
        encoding="utf-8",
    )
    checks_one, _ = load_review_checks(fixture)
    table_one = jev_review.name_question_table(checks_one)

    checks_two = [
        *checks_one,
        _noul("extra_check", "grammar", ["physical_base"], "A newly named check?"),
    ]
    table_two = jev_review.name_question_table(checks_two)

    assert jev_review.check_question_name("malformed_physical_base") in table_one
    assert jev_review.check_question_name("extra_check") in table_two
    assert jev_review.check_question_name("extra_check") not in table_one
    assert table_two[jev_review.check_question_name("extra_check")] == {
        "type": "noul",
        "instructions": "A newly named check?",
    }


def test_question_table_asks_one_category_per_dimension_present():
    table = jev_review.name_question_table(_small_checks())
    assert jev_review.dimension_question_name("grammar") in table
    assert jev_review.dimension_question_name("semantic") in table
    assert jev_review.dimension_question_name("convention") not in table
    grammar = table[jev_review.dimension_question_name("grammar")]
    assert grammar["type"] == "choice"
    assert set(grammar["criteria"]) == set(jev_review._DIMENSION_LEVELS["grammar"])


def test_audit_decided_check_carries_no_question():
    table = jev_review.name_question_table(_small_checks())
    assert jev_review.check_question_name("grammar_round_trip_failure") not in table


def test_choice_question_options_propagate_from_the_taxonomy():
    table = jev_review.name_question_table(_small_checks())
    choice = table[jev_review.check_question_name("bound_to_wrong_object")]
    assert choice["type"] == "choice"
    assert set(choice["criteria"]) == {"the bound source object", "a different object"}


def test_choice_without_an_identifiable_acceptable_form_is_refused():
    # A Choice authors its acceptable form first, so a blank first option names
    # no acceptable form and the check cannot be split into acceptable against
    # defect. Table construction refuses it rather than reading a blank
    # acceptable form and mis-deciding the check.
    check = _choice(
        "unsplittable",
        "semantic",
        ["source_paths"],
        "Which form is it?",
        ["", "a defective form"],
    )
    with pytest.raises(ValueError, match="acceptable form"):
        jev_review.name_question_table([check])


def test_choice_acceptable_form_reads_the_authored_first_option():
    question = JevQuestion(
        kind=JevQuestionKind.CHOICE, text="q?", options=["ok", "defect"]
    )
    assert jev_review.choice_acceptable_form(question) == "ok"
    question.options = ["only one"]
    with pytest.raises(ValueError, match="acceptable form"):
        jev_review.choice_acceptable_form(question)


# ---------------------------------------------------------------------------
# A dimension's decision is derived in code from its category answer
# ---------------------------------------------------------------------------


def test_derive_dimension_decision_maps_category_to_decision():
    assert jev_review.derive_dimension_decision("grammar", "canonical") == "accept"
    assert jev_review.derive_dimension_decision("grammar", "malformed") == "reject"
    assert jev_review.derive_dimension_decision("semantic", "ambiguous") == "reject"


def test_derive_dimension_decision_never_reads_an_unknown_as_accept():
    assert jev_review.derive_dimension_decision("grammar", None) == "unknown"
    assert jev_review.derive_dimension_decision("grammar", "nonsense") == "unknown"


def test_derive_decision_not_a_constant_of_the_dimension_alone():
    # Two categories of one dimension decide differently, so the decision is a
    # function of the category answer rather than of the dimension.
    accept = jev_review.derive_dimension_decision("completeness", "complete")
    reject = jev_review.derive_dimension_decision("completeness", "incomplete")
    assert accept != reject


# ---------------------------------------------------------------------------
# Check failure is derived in code from the answer
# ---------------------------------------------------------------------------


def test_noul_check_fails_above_the_probability_midpoint():
    check = _noul("c", "grammar", ["physical_base"], "q?")
    assert jev_review.check_failed(check, {"noul": 0.9}) is True
    assert jev_review.check_failed(check, {"noul": 0.1}) is False


def test_defect_worded_noul_high_probability_is_read_as_a_failure():
    """A defect-worded Noul's high probability is the defect holding.

    The taxonomy declares every Noul's true case as the defect, and the reader
    fails it at or above the probability midpoint. A high answer to a
    defect-worded question — the shipped convention — must therefore read as a
    failure, and a low answer as no failure.
    """
    check = _noul(
        "malformed_physical_base",
        "grammar",
        ["physical_base"],
        "Is the physical base unregistered, malformed, or more than one token?",
        true_case="defect",
    )
    assert check.question.true_case == "defect"
    assert jev_review.check_failed(check, {"noul": 0.9}) is True
    assert jev_review.check_failed(check, {"noul": 0.1}) is False
    assert jev_review.check_failed(check, None) is False


def test_shipped_defect_worded_nouls_read_high_probability_as_failure():
    """The shipped taxonomy's Nouls are defect-worded, so the reader holds."""
    checks, _ = load_review_checks()
    nouls = [
        check
        for check in checks
        if check.question is not None and check.question.kind is JevQuestionKind.NOUL
    ]
    assert nouls
    for check in nouls:
        assert check.question.true_case == "defect"
        assert jev_review.check_failed(check, {"noul": 1.0}) is True


def test_choice_check_fails_when_a_defective_form_is_chosen():
    check = _choice("c", "semantic", ["source_paths"], "q?", ["ok", "defect"])
    assert jev_review.check_failed(check, {"choice": "ok"}) is False
    assert jev_review.check_failed(check, {"choice": "defect"}) is True


def test_check_with_no_answer_does_not_fail():
    check = _noul("c", "grammar", ["physical_base"], "q?")
    assert jev_review.check_failed(check, None) is False


# ---------------------------------------------------------------------------
# The batch entry point
# ---------------------------------------------------------------------------


def _fake_decide_batch(answers, cost=0.004):
    async def _fake(states, questions, **kwargs):
        results = [(dict(answers), cost) for _state in states]
        return results, cost * len(states)

    return _fake


def _item(name_id="electron_temperature"):
    return {
        "id": name_id,
        "kind": "scalar",
        "unit": "eV",
        "description": "d",
        "physical_base": "temperature",
        "source_paths": ["p"],
    }


async def test_batch_returns_categories_decisions_check_answers_and_cost(monkeypatch):
    answers = {
        jev_review.dimension_question_name("grammar"): {"choice": "canonical"},
        jev_review.dimension_question_name("semantic"): {"choice": "ambiguous"},
        jev_review.check_question_name("malformed_physical_base"): {"noul": 0.9},
        jev_review.check_question_name("bound_to_wrong_object"): {
            "choice": "a different object"
        },
    }
    monkeypatch.setattr(
        jev_review, "decide_batch", _fake_decide_batch(answers, cost=0.004)
    )
    monkeypatch.setattr(jev_review, "rerank_name_comparators", _no_rerank)

    review = await jev_review.review_names([_item()], checks=_small_checks())

    assert review.cost == 0.004
    record = review.items[0]
    assert record["answered"] is True
    assert record["decisions"]["grammar"] == "accept"
    assert record["decisions"]["semantic"] == "reject"
    assert record["checks"]["malformed_physical_base"]["failed"] is True
    assert record["checks"]["bound_to_wrong_object"]["failed"] is True


async def test_batch_marks_a_failed_call_unknown_rather_than_accepted(monkeypatch):
    async def _fake(states, questions, **kwargs):
        return [None for _state in states], 0.0

    monkeypatch.setattr(jev_review, "rerank_name_comparators", _no_rerank)
    monkeypatch.setattr(jev_review, "decide_batch", _fake)

    review = await jev_review.review_names([_item()], checks=_small_checks())
    record = review.items[0]
    assert record["answered"] is False
    assert record["decisions"]["grammar"] == "unknown"
    assert review.cost == 0.0


async def test_audit_decided_check_contributes_its_finding_with_no_question(
    monkeypatch,
):
    monkeypatch.setattr(jev_review, "rerank_name_comparators", _no_rerank)
    monkeypatch.setattr(jev_review, "decide_batch", _fake_decide_batch({}, cost=0.001))
    report = AuditReport(
        lint_findings=[
            LintFinding(
                name_id="electron_temperature",
                finding_type="round_trip_failure",
                detail="Round-trip mismatch",
            )
        ]
    )
    review = await jev_review.review_names(
        [_item()], checks=_small_checks(), audit_report=report
    )
    record = review.items[0]["checks"]["grammar_round_trip_failure"]
    assert record["failed"] is True
    assert record["question"] is None
    assert "round_trip_failure" in record["audit_finding"]


async def test_audit_check_without_a_report_is_unanswered_not_passed(monkeypatch):
    # An audit that was never run must not read as answered-and-not-failed: the
    # check is unknown, so its derived decision is neither accept nor reject.
    monkeypatch.setattr(jev_review, "rerank_name_comparators", _no_rerank)
    monkeypatch.setattr(jev_review, "decide_batch", _fake_decide_batch({}, cost=0.0))
    review = await jev_review.review_names([_item()], checks=_small_checks())
    record = review.items[0]["checks"]["grammar_round_trip_failure"]
    assert record["answered"] is False
    assert record["failed"] is False
    assert record["decision"] == "unknown"


async def test_repeated_segment_is_decided_without_an_audit_report(monkeypatch):
    monkeypatch.setattr(jev_review, "rerank_name_comparators", _no_rerank)
    monkeypatch.setattr(jev_review, "decide_batch", _fake_decide_batch({}, cost=0.0))
    items = [_item("electron_electron_temperature"), _item("electron_temperature")]

    review = await jev_review.review_names(items)

    repeated = review.items[0]["checks"]["redundant_repeated_segment"]
    clean = review.items[1]["checks"]["redundant_repeated_segment"]
    assert repeated["answered"] is True
    assert repeated["failed"] is True
    assert repeated["decision"] == "reject"
    assert "repeated_token" in repeated["audit_finding"]
    assert repeated["question"] is None
    assert clean["answered"] is True
    assert clean["failed"] is False
    assert clean["decision"] == "accept"


async def test_audit_check_with_a_report_is_answered_from_it(monkeypatch):
    # A report that carries no finding for this name still means the audit ran,
    # so the check is answered and accepted rather than left unknown.
    monkeypatch.setattr(jev_review, "rerank_name_comparators", _no_rerank)
    monkeypatch.setattr(jev_review, "decide_batch", _fake_decide_batch({}, cost=0.0))
    report = AuditReport(lint_findings=[])
    review = await jev_review.review_names(
        [_item()], checks=_small_checks(), audit_report=report
    )
    record = review.items[0]["checks"]["grammar_round_trip_failure"]
    assert record["answered"] is True
    assert record["failed"] is False
    assert record["decision"] == "accept"


# ---------------------------------------------------------------------------
# Sibling reranking, and keeping the embedding order on failure
# ---------------------------------------------------------------------------


async def _no_rerank(items, comparators, **kwargs):
    return list(comparators), 0.0, ""


async def _raise_rerank(items, comparators, **kwargs):
    raise RuntimeError("rerank service down")


async def test_sibling_rerank_failure_keeps_embedding_order(monkeypatch):
    monkeypatch.setattr(jev_review, "rerank_name_comparators", _raise_rerank)
    monkeypatch.setattr(jev_review, "decide_batch", _fake_decide_batch({}, cost=0.0))
    item = _item()
    item["sibling_names"] = ["b_first", "a_second"]
    review = await jev_review.review_names([item], checks=_small_checks())

    record = review.items[0]
    assert record["state"]["neighbourhood"]["sibling_names"]["value"] == [
        "b_first",
        "a_second",
    ]
    assert "embedding order" in record["sibling_note"]


async def test_sibling_rerank_success_uses_the_judged_order(monkeypatch):
    async def _ordered(items, comparators, **kwargs):
        return list(reversed(comparators)), 0.002, "seat"

    monkeypatch.setattr(jev_review, "rerank_name_comparators", _ordered)
    monkeypatch.setattr(jev_review, "decide_batch", _fake_decide_batch({}, cost=0.0))
    item = _item()
    item["sibling_names"] = ["first", "second"]
    review = await jev_review.review_names([item], checks=_small_checks())

    assert review.items[0]["state"]["neighbourhood"]["sibling_names"]["value"] == [
        "second",
        "first",
    ]
    assert review.cost == 0.002


# ---------------------------------------------------------------------------
# The taxonomy the module reads is the installed one
# ---------------------------------------------------------------------------


def test_installed_taxonomy_has_name_checks_with_questions_and_audit_classes():
    checks, _ = load_review_checks()
    name_checks = jev_review._name_checks(checks)
    assert name_checks
    table = jev_review.name_question_table(checks)
    for check in name_checks:
        if check.audit_finding_class is not None:
            assert jev_review.check_question_name(check.id) not in table
        else:
            assert jev_review.check_question_name(check.id) in table
