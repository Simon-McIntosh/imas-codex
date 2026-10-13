"""Name-axis review judged by Jev, built from the named-check taxonomy.

Review is a judgment, and judgment is what the decisions endpoint does. This
module turns one candidate name into a judgment state and one taxonomy into a
question table, both read from
:mod:`imas_codex.standard_names.review.checks` rather than from a second
hand-written list, so a check added to
``imas_codex/llm/prompts/sn/review_checks.yaml`` becomes a question the judge is
asked without a matching code change.

Two kinds of question are asked of every candidate, in one call.

* A **dimension category** question per review dimension: the judge places the
  candidate in one of the dimension's ordered categories, and the decision is
  derived in code from the category (:func:`derive_dimension_decision`) rather
  than asked of the model. This is the categorise-first shape the discovery
  decision loops use: the model does the reading, the code decides.
* A **failed-check** question per named check that carries one — a Noul whose
  question is the check's own criterion, or a Choice whose options are the
  forms the check must tell apart. A check the audit already decides
  deterministically contributes its ``AuditReport`` finding and is not asked.

The state carries the candidate name, its grammar segments, kind, unit,
description, the source DD paths with their documentation, and its sibling
names. Every neighbourhood field records whether it was observed, because
unknown is not the same as empty and a check that reads an unobserved field is
not reading a field it was shown to be absent.

The module writes no graph state and nothing in the pipeline calls it yet.
Every call goes through the shared decisions owner
(:func:`imas_codex.discovery.base.judgment.decide_batch`) under the ``sn-judge``
seat, so there is no third decisions owner.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

from imas_codex.discovery.base.judgment import decide_batch
from imas_codex.standard_names.context import rerank_name_comparators
from imas_codex.standard_names.review.checks import (
    JevQuestion,
    JevQuestionKind,
    ReviewCheck,
    load_review_checks,
)

logger = logging.getLogger(__name__)

# Service tag the decisions call layer uses for the API key and request title.
# Every standard-names stage shares it.
SERVICE = "standard-names"

# Pipeline-cost step the calls book to, matching the other judge stages.
COST_STEP = "sn_judge"

# The axis this module builds for. The documentation axis is built beside it.
NAME_AXIS = "name"

# Grammar-segment fields the candidate name decomposes into. A name's segments
# are the closed-vocabulary slots of the grammar, so they are carried flat.
SEGMENT_FIELDS: tuple[str, ...] = (
    "physical_base",
    "subject",
    "component",
    "coordinate",
    "position",
    "process",
)

# The named fields carried directly, each a scalar the checks read as text.
# A field the taxonomy reads but that is not one of these is a neighbourhood
# field and is carried as an observed/unobserved pair.
_CORE_FIELDS: frozenset[str] = frozenset(
    {"id", "kind", "unit", "description", "documentation", *SEGMENT_FIELDS}
)

# The sibling names the state carries beside the candidate. They are not an
# evidence field in the taxonomy — they are the neighbourhood a name is read
# against — so they are named here rather than derived from the checks.
_SIBLING_FIELD = "sibling_names"

# The defect a Choice question names, keyed by check id. A Choice check's
# options list one acceptable form and the defective forms beside it; the
# accepted form is the first option, which the taxonomy authors first, and the
# defect set is the rest. It is stated in code because the taxonomy holds the
# question, not the verdict, and a check whose options do not carry an
# acceptable form first would be mis-decided silently — so a Choice check whose
# options the rule cannot split is refused at table construction rather than
# guessed.
# Noul questions need no entry: the check's true case is the defect.

# The probability above which a Noul's true case is read as holding. The
# cascade's calibrated per-check threshold replaces this when the cascade of
# the plan lands; until then a check's fail is derived here at the question's
# own midpoint, never asked of the model.
NOUL_FAIL_PROBABILITY = 0.5

# Ordered categories per review dimension, best first. The judge places a
# candidate in one; the decision is derived in code and the categories are not
# sent as verdicts.
_DIMENSION_LEVELS: dict[str, tuple[str, ...]] = {
    "grammar": ("canonical", "minor_deviation", "malformed"),
    "semantic": ("unambiguous", "ambiguous", "wrong_object"),
    "convention": ("conforms", "minor_deviation", "violates"),
    "completeness": ("complete", "minor_gap", "incomplete"),
}

# The categories a dimension accepts. Every other category rejects.
_DIMENSION_PASSING: dict[str, frozenset[str]] = {
    "grammar": frozenset({"canonical"}),
    "semantic": frozenset({"unambiguous"}),
    "convention": frozenset({"conforms"}),
    "completeness": frozenset({"complete"}),
}


def derive_dimension_decision(dimension: str, category: str | None) -> str:
    """Derive one dimension's decision from the category the judge chose.

    The judge places the candidate in a category; the decision is a code
    function of that category, so the model never answers a verdict directly.
    An unknown category — a missing answer, or one the dimension does not offer
    — yields ``"unknown"`` rather than a decision, so a failed or absent call
    is never read as an acceptance.
    """
    levels = _DIMENSION_LEVELS.get(dimension)
    if category is None or levels is None or category not in levels:
        return "unknown"
    return "accept" if category in _DIMENSION_PASSING[dimension] else "reject"


def _name_checks(checks: Sequence[ReviewCheck]) -> list[ReviewCheck]:
    return [check for check in checks if check.axis == NAME_AXIS]


@lru_cache(maxsize=1)
def _default_checks() -> tuple[ReviewCheck, ...]:
    """The installed taxonomy, loaded once for callers that name no checks."""
    return tuple(load_review_checks()[0])


def neighbourhood_fields(checks: Sequence[ReviewCheck]) -> tuple[str, ...]:
    """The neighbourhood fields the name-axis checks read, plus siblings.

    Read from the taxonomy: every evidence field the name-axis checks name
    that is not a core scalar or a grammar segment is a neighbourhood field,
    and the sibling names the candidate is read against join them. A check
    that names a field the state does not carry would be reading nothing, so
    the state carries exactly this set.
    """
    evidence: set[str] = set()
    for check in _name_checks(checks):
        evidence.update(check.evidence_fields)
    derived = sorted(evidence - _CORE_FIELDS)
    return tuple(sorted({*derived, _SIBLING_FIELD}))


def _observed(value: Any) -> dict[str, Any]:
    """Wrap one neighbourhood value so unknown is distinct from empty.

    A value present but empty (an empty list, an empty string) is observed and
    empty; a value the item does not carry is unobserved. The two are different
    facts and a check that reads the field must be able to tell them apart.
    """
    if value is None:
        return {"observed": False, "value": None}
    return {"observed": True, "value": value}


def build_name_state(
    item: Mapping[str, Any],
    *,
    sibling_names: Sequence[Any] | None = None,
    sibling_note: str | None = None,
    neighbourhood: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Build the one judgement state for a candidate name.

    The state carries the candidate's identity and written text, its grammar
    segments, and a neighbourhood block in which every field records whether it
    was observed. ``sibling_names`` overrides the item's own sibling list when
    the caller has reranked it; a ``None`` override falls back to the item, and
    an explicitly empty list is observed-and-empty rather than unobserved.

    Args:
        item: The candidate, carrying at least ``id`` and ``description`` and
            whatever neighbourhood fields it was enriched with.
        sibling_names: The reranked sibling order, when the caller has one.
        sibling_note: A note about the sibling ordering (for example that the
            rerank was unavailable and the embedding order was kept), recorded
            on the state so a reader knows how the siblings were ordered.
        neighbourhood: The neighbourhood field names to carry; defaults to the
            set the name-axis checks read.

    Returns:
        The judgement-state mapping handed to the decisions endpoint.
    """
    name_id = item.get("id") or item.get("standard_name") or ""
    segments = {field_name: item.get(field_name) for field_name in SEGMENT_FIELDS}
    name_block: dict[str, Any] = {
        "id": name_id,
        "kind": item.get("kind"),
        "unit": item.get("unit"),
        "description": item.get("description"),
        "documentation": item.get("documentation"),
        "segments": segments,
    }

    sibling_value = (
        sibling_names if sibling_names is not None else item.get(_SIBLING_FIELD)
    )
    if sibling_value is not None:
        sibling_value = list(sibling_value)
    fields = (
        tuple(neighbourhood)
        if neighbourhood is not None
        else neighbourhood_fields(_default_checks())
    )
    neighbourhood_block: dict[str, Any] = {
        field_name: _observed(
            sibling_value if field_name == _SIBLING_FIELD else item.get(field_name)
        )
        for field_name in fields
    }
    if _SIBLING_FIELD not in neighbourhood_block:
        neighbourhood_block[_SIBLING_FIELD] = _observed(sibling_value)

    state: dict[str, Any] = {"name": name_block, "neighbourhood": neighbourhood_block}
    if sibling_note:
        state["sibling_note"] = sibling_note
    return state


def name_question_table(checks: Sequence[ReviewCheck]) -> dict[str, dict[str, Any]]:
    """Build the questions the name axis asks, read from the taxonomy.

    One category Choice per review dimension the taxonomy carries, and one
    question per name-axis check that carries one. A check decided by an
    ``AuditReport`` finding class carries no question and contributes no entry,
    because the audit already decides it and asking would be asking twice.

    Args:
        checks: The taxonomy's checks.

    Returns:
        Question definitions keyed by question name: ``dimension:<dimension>``
        for a dimension category, ``check:<check id>`` for a named check.
    """
    table: dict[str, dict[str, Any]] = {}

    dimensions = {check.dimension for check in _name_checks(checks)}
    for dimension in sorted(dimensions):
        table[dimension_question_name(dimension)] = _dimension_question_def(dimension)

    for check in _name_checks(checks):
        if check.question is None:
            continue
        table[check_question_name(check.id)] = _question_def(check.question)

    return table


def dimension_question_name(dimension: str) -> str:
    return f"dimension:{dimension}"


def check_question_name(check_id: str) -> str:
    return f"check:{check_id}"


def _dimension_question_def(dimension: str) -> dict[str, Any]:
    levels = _DIMENSION_LEVELS[dimension]
    return {
        "type": "choice",
        "instructions": (
            f"Categorise the candidate name's {dimension.replace('_', ' ')}."
        ),
        "criteria": {level: level for level in levels},
    }


def _question_def(question: JevQuestion) -> dict[str, Any]:
    """Render one taxonomy question as the decisions-endpoint definition."""
    if question.kind is JevQuestionKind.NOUL:
        return {"type": "noul", "instructions": question.text}
    return {
        "type": "choice",
        "instructions": question.text,
        "criteria": {option: option for option in question.options},
    }


def check_failed(check: ReviewCheck, answer: Mapping[str, Any] | None) -> bool:
    """Whether one answered check names the defect it asks about.

    A Noul's true case is the defect, so it fails when its probability is at or
    above :data:`NOUL_FAIL_PROBABILITY`. A Choice fails when its selected option
    is one of the defective forms — every option but the first, which the
    taxonomy authors as the acceptable form. An unanswered check does not fail;
    a missing answer is unknown, not a pass, and is reported as unanswered by
    the caller rather than as a failing check.
    """
    if not answer:
        return False
    if check.question is None:
        return False
    if check.question.kind is JevQuestionKind.NOUL:
        value = answer.get("noul")
        return (
            isinstance(value, int | float)
            and not isinstance(value, bool)
            and float(value) >= NOUL_FAIL_PROBABILITY
        )
    choice = answer.get("choice")
    if choice is None:
        return False
    return str(choice) != str(check.question.options[0])


def audit_finding_for(
    check: ReviewCheck, name_id: str, audit_report: Any | None
) -> str | None:
    """The audit finding that decides one check for one name, or ``None``.

    A check decided by an ``AuditReport`` finding class reads the report for a finding
    of that class on this name. The near-duplicate check reads the component
    the name belongs to instead, since the finding is a group rather than a
    per-name entry.
    """
    if audit_report is None or check.audit_finding_class is None:
        return None
    finding_class = check.audit_finding_class
    if finding_class == "duplicate_component":
        for component in getattr(audit_report, "duplicate_components", []):
            if name_id in component.names:
                return (
                    f"[duplicate_component] {name_id}: in a near-duplicate group "
                    f"of {len(component.names)} names"
                )
        return None
    for finding in getattr(audit_report, "lint_findings", []):
        if finding.name_id == name_id and finding.finding_type == finding_class:
            return f"[{finding_class}] {name_id}: {finding.detail}"
    for finding in getattr(audit_report, "link_findings", []):
        if finding.name_id == name_id and finding.finding_type == finding_class:
            return f"[{finding_class}] {name_id}: {finding.detail}"
    return None


@dataclass
class NameAxisReview:
    """The result of judging a batch of names on the name axis.

    ``items`` holds one record per input candidate, in input order: the state
    judged, the category chosen per dimension with its derived decision, and
    per-check answers. ``cost`` is the call layer's summed reported USD spend,
    both the judgement batch and any sibling rerank.
    """

    items: list[dict[str, Any]] = field(default_factory=list)
    cost: float = 0.0


async def _ordered_siblings(
    item: Mapping[str, Any],
    siblings: Sequence[Mapping[str, Any]],
    *,
    model: str | None,
    budget_seconds: float | None,
) -> tuple[list[Mapping[str, Any]], float, str | None]:
    """Order a candidate's siblings by judged relevance, keeping embedding order on failure.

    Ranking is not a gate: a rerank that raises or fails keeps the sibling
    names in the embedding order the retrieval produced, and says so in the
    returned note, rather than failing the judgement of the candidate whose
    context it was gathering.
    """
    pool = list(siblings)
    if not pool:
        return pool, 0.0, None
    try:
        ordered, cost, _seat = await rerank_name_comparators(
            [item], pool, model=model, budget_seconds=budget_seconds
        )
    except Exception as exc:  # noqa: BLE001 - ranking is not a gate
        logger.warning("sibling rerank unavailable for %r: %s", item.get("id"), exc)
        return pool, 0.0, f"rerank unavailable ({exc}); embedding order returned"
    return list(ordered), cost, None


async def review_names(
    items: Sequence[Mapping[str, Any]],
    *,
    checks: Sequence[ReviewCheck] | None = None,
    audit_report: Any | None = None,
    model: str | None = None,
    rerank_model: str | None = None,
    cost: Any | None = None,
    concurrency: int = 8,
    budget_seconds: float | None = None,
    rerank_budget_seconds: float | None = None,
) -> NameAxisReview:
    """Judge a batch of candidate names on the name axis.

    Each candidate's state is built with its reranked siblings, then every
    candidate is judged in one batch through
    :func:`~imas_codex.discovery.base.judgment.decide_batch` under the
    ``sn-judge`` seat. The dimension category questions and the failed-check
    questions are asked together in that one call, so a candidate costs one
    judgement rather than one per question.
    """
    checks = list(checks) if checks is not None else load_review_checks()[0]
    name_checks = _name_checks(checks)

    if model is None:
        from imas_codex.settings import get_model

        model = get_model("sn-judge")

    states: list[dict[str, Any]] = []
    sibling_notes: list[str | None] = []
    rerank_cost = 0.0
    for item in items:
        siblings = item.get(_SIBLING_FIELD) or []
        ordered, item_rerank_cost, note = await _ordered_siblings(
            item,
            siblings,
            model=rerank_model,
            budget_seconds=rerank_budget_seconds,
        )
        rerank_cost += item_rerank_cost
        sibling_notes.append(note)
        states.append(build_name_state(item, sibling_names=ordered, sibling_note=note))

    questions = name_question_table(name_checks)
    results, batch_cost = await decide_batch(
        states,
        questions,
        model=model,
        service=SERVICE,
        concurrency=concurrency,
        budget_seconds=budget_seconds,
        cost=cost,
        step=COST_STEP,
    )
    # The sibling rerank spends under the same seat but cannot book to the
    # caller's tracker itself, so it is added here rather than dropped.
    if rerank_cost and cost is not None:
        cost.add(COST_STEP, rerank_cost, 0)

    records: list[dict[str, Any]] = []
    for item, state, note, result in zip(
        items, states, sibling_notes, results, strict=True
    ):
        records.append(
            _item_record(
                item, state, note, name_checks, questions, audit_report, result
            )
        )

    return NameAxisReview(items=records, cost=batch_cost + rerank_cost)


def _item_record(
    item: Mapping[str, Any],
    state: Mapping[str, Any],
    sibling_note: str | None,
    name_checks: Sequence[ReviewCheck],
    questions: Mapping[str, Any],
    audit_report: Any | None,
    result: tuple[dict[str, Any], float] | None,
) -> dict[str, Any]:
    """Assemble one candidate's record: categories, decisions, check answers."""
    answers = result[0] if result is not None else {}
    call_cost = result[1] if result is not None else 0.0
    name_id = item.get("id") or item.get("standard_name") or ""

    categories: dict[str, Any] = {}
    decisions: dict[str, str] = {}
    for check in name_checks:
        dimension = check.dimension
        qname = dimension_question_name(dimension)
        answer = answers.get(qname)
        categories[dimension] = answer
        if dimension not in decisions:
            decisions[dimension] = derive_dimension_decision(
                dimension, (answer or {}).get("choice")
            )

    check_records: dict[str, Any] = {}
    for check in name_checks:
        if check.audit_finding_class is not None:
            finding = audit_finding_for(check, name_id, audit_report)
            check_records[check.id] = {
                "answered": True,
                "failed": finding is not None,
                "audit_finding": finding,
                "question": None,
            }
            continue
        question_name = check_question_name(check.id)
        answer = answers.get(question_name)
        check_records[check.id] = {
            "answered": answer is not None,
            "failed": check_failed(check, answer),
            "answer": answer,
            "question": questions.get(question_name),
            "audit_finding": None,
        }

    return {
        "name_id": name_id,
        "state": state,
        "sibling_note": sibling_note,
        "answered": result is not None,
        "categories": categories,
        "decisions": decisions,
        "checks": check_records,
        "cost_usd": call_cost,
    }
