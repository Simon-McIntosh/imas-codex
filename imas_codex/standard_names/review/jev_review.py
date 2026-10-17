"""Name and documentation review built from the named-check taxonomy.

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

The documentation builder splits model-written prose into claims, excluding
registry text and notation. Its question table asks one source-comparison Choice
per claim and categorises three rubric dimensions. Code gives any contradictory
claim a veto; a high dimension score cannot offset it.

The module writes no graph state. The documentation acceptance path calls its
claim comparison and veto while leaving the holistic score in shadow.
Every call goes through the shared decisions owner
(:func:`imas_codex.discovery.base.judgment.decide_batch`) under the ``sn-judge``
seat, so there is no third decisions owner.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

from imas_codex.discovery.base.judgment import decide_batch
from imas_codex.standard_names.context import rerank_name_comparators
from imas_codex.standard_names.grammar_adapter import parse_canonical_name
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

# The two axes share question and audit helpers.
NAME_AXIS = "name"
DOCS_AXIS = "docs"

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

# The probability at or above which a Noul's true case is read as the defect
# holding. The model returns a probability rather than a verdict, and the fail
# is derived here rather than asked of it, so this midpoint is the classifier.
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


def _docs_checks(checks: Sequence[ReviewCheck]) -> list[ReviewCheck]:
    return [check for check in checks if check.axis == DOCS_AXIS]


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


@lru_cache(maxsize=1)
def _physical_base_roster() -> tuple[str, ...]:
    """Read the closed base registry also supplied to the incumbent prompt."""
    from imas_standard_names.grammar.vocab_loaders import load_physical_bases

    return tuple(sorted(load_physical_bases().bases))


def _source_cocos_label(item: Mapping[str, Any]) -> str | None:
    """Take the label the DD source reader obtained from its IMASNode."""
    if item.get("cocos_label"):
        return str(item["cocos_label"])
    for source in item.get("dd_source_docs") or ():
        if isinstance(source, Mapping) and source.get("cocos_label"):
            return str(source["cocos_label"])
    return None


def _owner_segment(item: Mapping[str, Any]) -> str | None:
    """Read the owner from the DD reader's parent IMASNode path."""
    parent_path = item.get("parent_path")
    if not parent_path:
        return item.get("owner_segment")
    parts = str(parent_path).removeprefix("dd:").split("/")
    if not parts:
        return None
    if parts[-1] in {"outline", "value"} and len(parts) > 1:
        return parts[-2]
    return parts[-1]


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
    try:
        name_block["decomposition"] = parse_canonical_name(str(name_id)).ir.model_dump(
            mode="json"
        )
    except (TypeError, ValueError):
        name_block["decomposition"] = None

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
    neighbourhood_block["cocos_label"] = _observed(_source_cocos_label(item))
    neighbourhood_block["physical_base_roster"] = _observed(
        list(_physical_base_roster())
    )
    neighbourhood_block["owner_segment"] = _observed(_owner_segment(item))

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


def choice_acceptable_form(question: JevQuestion) -> str:
    """The acceptable form of a Choice question, or a refusal when it is unreadable.

    The taxonomy authors a Choice question's options with the acceptable form
    first and the defective forms after it, so the split the check reads is the
    first option against the rest. A question whose options carry no readable
    acceptable form — fewer than two options, so there is no defect set to
    split off, or a blank first option — is refused rather than read from
    whichever option happens to be first, because a mis-split decides the check
    silently in the wrong direction.
    """
    options = question.options
    if len(options) < 2 or not options[0].strip():
        raise ValueError(
            f"Choice question {question.text!r} carries no identifiable "
            "acceptable form: its options must list the acceptable form first, "
            "with at least one defective form beside it"
        )
    return options[0]


def _question_def(question: JevQuestion) -> dict[str, Any]:
    """Render one taxonomy question as the decisions-endpoint definition."""
    if question.kind is JevQuestionKind.NOUL:
        return {"type": "noul", "instructions": question.text}
    choice_acceptable_form(question)
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
    return str(choice) != str(choice_acceptable_form(check.question))


def _check_decision(answered: bool, failed: bool) -> str:
    """Derive one check's decision from whether it was answered and what it found.

    An answered check accepts when it found, and rejects when it found the
    defect. A check that was never answered — no answer to its question, or an
    audit that was not run — is ``unknown``: an absent call is not an
    acceptance, so an audit that never ran is unknown rather than a pass.
    """
    if not answered:
        return "unknown"
    return "reject" if failed else "accept"


def audit_finding_for(
    check: ReviewCheck, name_id: str, audit_report: Any | None
) -> str | None:
    """The audit finding that decides one check for one name, or ``None``.

    A check decided by an ``AuditReport`` finding class reads the report for a finding
    of that class on this name. The near-duplicate check reads the component
    the name belongs to instead, since the finding is a group rather than a
    per-name entry.
    """
    if check.audit_finding_class is None:
        return None
    if check.id == "redundant_repeated_segment":
        from imas_codex.standard_names.audits import repeated_token_check

        findings = repeated_token_check({"id": name_id})
        return findings[0] if findings else None
    if audit_report is None:
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
        if (
            finding.name_id == name_id
            and finding.finding_type == "convention_violation"
            and finding.detail.startswith(f"{finding_class}:")
        ):
            return f"[{finding_class}] {name_id}: {finding.detail}"
    for finding in getattr(audit_report, "link_findings", []):
        if finding.name_id == name_id and finding.finding_type == finding_class:
            return f"[{finding_class}] {name_id}: {finding.detail}"
    return None


_DOCS_DIMENSION_CRITERIA: dict[str, dict[str, str]] = {
    "description_quality": {
        "precise_definition": "One concise physical definition of the bound quantity.",
        "imprecise_definition": "A definition with ambiguous wording or an omitted physical distinction.",
        "not_a_definition": "Multiple lines, editorial filler, or text that does not define the quantity.",
    },
    "documentation_quality": {
        "normative_definition": "A physical definition with readable notation and no recipe or padding.",
        "partial_definition": "A physical definition whose explanation or notation is incomplete.",
        "non_normative": "A diagnostic list, recipe, padding, or broken notation instead of a definition.",
    },
    "completeness": {
        "complete": "The text covers the bound source meaning and needed distinctions and relationships.",
        "minor_gap": "The meaning is mostly present but a relevant distinction or relationship is missing.",
        "incomplete": "The text omits a necessary part of the bound quantity's meaning.",
    },
}

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")
_DISPLAY_MATH = re.compile(r"\$\$.*?\$\$", re.DOTALL)
_INLINE_MATH = re.compile(r"(?<!\$)\$(?!\$)[^$\n]*\$(?!\$)")
_MARKDOWN_LINK = re.compile(r"\[([^\]]+)\]\([^)]+\)")


def _registry_texts(item: Mapping[str, Any]) -> tuple[str, ...]:
    """Find exact registry text inserted into this documentation entry."""
    from imas_standard_names.grammar.terms import AppliedConventions

    from imas_codex.standard_names.context import conventions_context_for
    from imas_codex.standard_names.docs_gates import (
        _template_opening_pattern,
        locus_definition_sentence,
    )

    texts = [str(value) for value in item.get("registry_sentences") or ()]
    selected = item.get("conventions")
    if selected is None and item.get("id"):
        selected = conventions_context_for(str(item["id"]))
    if not selected:
        return tuple(texts)
    conventions = AppliedConventions.model_validate(selected)
    documentation = str(item.get("documentation") or "")
    opening = next(
        (line.strip() for line in documentation.splitlines() if line.strip()), ""
    )
    for template in conventions.templates:
        pattern = _template_opening_pattern(template.documentation)
        if pattern is not None and pattern.fullmatch(opening):
            texts.append(opening)
        note = getattr(template, "note", None)
        if note:
            note_pattern = _template_opening_pattern(str(note))
            if note_pattern is not None:
                texts.extend(
                    line.strip()
                    for line in documentation.splitlines()
                    if note_pattern.fullmatch(line.strip())
                )
    for concept in conventions.concepts:
        sentence = locus_definition_sentence(concept)
        if sentence and sentence in documentation:
            texts.append(sentence)
    return tuple(texts)


def split_documentation_claims(
    documentation: str, *, registry_sentences: Sequence[str] = ()
) -> list[str]:
    """Keep each authored sentence intact for source comparison."""
    registry = {
        " ".join(sentence.split()) for sentence in registry_sentences if sentence
    }
    claims: list[str] = []
    for paragraph in re.split(r"\n\s*\n", documentation):
        paragraph = paragraph.strip()
        if not paragraph or " ".join(paragraph.split()) in registry:
            continue
        if _DISPLAY_MATH.fullmatch(paragraph):
            continue
        lines = [
            line.strip()
            for line in paragraph.splitlines()
            if line.strip() and not _DISPLAY_MATH.fullmatch(line.strip())
        ]
        prose = " ".join(lines)
        prose = _DISPLAY_MATH.sub("[equation]", prose)
        prose = _INLINE_MATH.sub("[equation]", prose)
        prose = _MARKDOWN_LINK.sub(r"\1", prose)
        for sentence in _SENTENCE_BOUNDARY.split(prose):
            claim = " ".join(sentence.split())
            if claim and claim not in registry:
                claims.append(claim)
    return claims


def build_docs_state(
    item: Mapping[str, Any], *, checks: Sequence[ReviewCheck] | None = None
) -> dict[str, Any]:
    """Carry one entry's claims and DD evidence without incumbent comments."""
    documentation = str(item.get("documentation") or "")
    state = {
        "name_id": item.get("id") or item.get("standard_name"),
        "documentation": documentation,
        "documentation_claims": split_documentation_claims(
            documentation, registry_sentences=_registry_texts(item)
        ),
        "description": item.get("description"),
    }
    evidence_fields = {
        evidence_field
        for check in _docs_checks(checks or _default_checks())
        for evidence_field in check.evidence_fields
    }
    for evidence_field in sorted(
        evidence_fields - {"id", "documentation", "description"}
    ):
        state[evidence_field] = _observed(item.get(evidence_field))
    return state


def docs_question_table(
    checks: Sequence[ReviewCheck], claims: Sequence[str]
) -> dict[str, dict[str, Any]]:
    """Build three rubric categories and one source comparison per claim."""
    docs_checks = _docs_checks(checks)
    dimensions = {check.dimension for check in docs_checks}
    table: dict[str, dict[str, Any]] = {}
    for dimension, criteria in _DOCS_DIMENSION_CRITERIA.items():
        if dimension in dimensions:
            table[dimension_question_name(dimension)] = {
                "type": "choice",
                "instructions": f"Which {dimension.replace('_', ' ')} level describes this entry?",
                "criteria": criteria,
            }
    claim_check = next(
        (check for check in docs_checks if check.id == "claim_contradicts_source"),
        None,
    )
    if claim_check is None or claim_check.question is None:
        raise ValueError("the claim support check is missing from the docs taxonomy")
    if set(claim_check.question.options) != {"supports", "contradicts", "says nothing"}:
        raise ValueError("the claim support check has unexpected Choice options")
    for index, _claim in enumerate(claims):
        table[check_question_name(f"{claim_check.id}:{index}")] = {
            "type": "choice",
            "instructions": (
                f"{claim_check.question.text} Compare only "
                f"`documentation_claims[{index}]` with `dd_source_docs.value` and "
                "`description`. If DD documentation was not observed, choose "
                "'says nothing'."
            ),
            "criteria": {
                "supports": "The DD evidence directly supports this specific claim.",
                "contradicts": "The DD evidence says the opposite of this specific claim.",
                "says nothing": "The DD evidence neither supports nor contradicts this claim.",
            },
        }
    for check in docs_checks:
        if check.id != claim_check.id and check.question is not None:
            table[check_question_name(check.id)] = _question_def(check.question)
    return table


def aggregate_docs_review(
    state: Mapping[str, Any],
    checks: Sequence[ReviewCheck],
    answers: Mapping[str, Mapping[str, Any]],
    *,
    audit_report: Any | None = None,
) -> dict[str, Any]:
    """Derive the docs decision, with any contradictory claim as a veto."""
    docs_checks = _docs_checks(checks)
    dimensions: dict[str, str] = {}
    for dimension, criteria in _DOCS_DIMENSION_CRITERIA.items():
        if any(check.dimension == dimension for check in docs_checks):
            choice = (answers.get(dimension_question_name(dimension)) or {}).get(
                "choice"
            )
            dimensions[dimension] = (
                "accept"
                if choice == next(iter(criteria))
                else "reject"
                if choice in criteria
                else "unknown"
            )
    claim_check = next(
        check for check in docs_checks if check.id == "claim_contradicts_source"
    )
    claim_answers = [
        answers.get(check_question_name(f"{claim_check.id}:{index}"))
        for index, _claim in enumerate(state["documentation_claims"])
    ]
    contradiction_veto = any(
        answer is not None and answer.get("choice") == "contradicts"
        for answer in claim_answers
    )
    check_records: dict[str, dict[str, Any]] = {}
    for check in docs_checks:
        if check.id == claim_check.id:
            continue
        finding = audit_finding_for(check, str(state["name_id"]), audit_report)
        answer = answers.get(check_question_name(check.id))
        answered = (
            audit_report is not None
            if check.audit_finding_class
            else answer is not None
        )
        failed = (
            finding is not None
            if check.audit_finding_class
            else check_failed(check, answer)
        )
        check_records[check.id] = {
            "answered": answered,
            "failed": failed,
            "decision": _check_decision(answered, failed),
            "audit_finding": finding,
            "answer": answer,
        }
    if (
        contradiction_veto
        or "reject" in dimensions.values()
        or any(record["failed"] for record in check_records.values())
    ):
        decision = "reject"
    elif (
        "unknown" in dimensions.values()
        or not claim_answers
        or any(
            answer is None or answer.get("choice") not in {"supports", "says nothing"}
            for answer in claim_answers
        )
        or any(not record["answered"] for record in check_records.values())
    ):
        decision = "unknown"
    elif any(answer.get("choice") == "says nothing" for answer in claim_answers):
        decision = "reject"
    else:
        decision = "accept"
    return {
        "decision": decision,
        "contradiction_veto": contradiction_veto,
        "dimensions": dimensions,
        "claim_answers": claim_answers,
        "checks": check_records,
    }


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
            answered = (
                audit_report is not None or check.id == "redundant_repeated_segment"
            )
            check_records[check.id] = {
                "answered": answered,
                "failed": answered and finding is not None,
                "decision": _check_decision(answered, finding is not None),
                "audit_finding": finding,
                "question": None,
            }
            continue
        question_name = check_question_name(check.id)
        answer = answers.get(question_name)
        answered = answer is not None
        failed = check_failed(check, answer)
        check_records[check.id] = {
            "answered": answered,
            "failed": failed,
            "decision": _check_decision(answered, failed),
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
