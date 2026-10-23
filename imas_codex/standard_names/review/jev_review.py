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
claim a veto; a high dimension score cannot offset it. Only a source that denies
the claim is a contradiction: a source that names a different quantity, an
opposite sign or direction, an incompatible unit or dimension, or a different
definition. One kind of text is therefore removed before a claim is held
against anything: a source whose wording carries no definition at all, only a
field label, has that wording emptied out of the evidence. The source stays,
with its path, unit and links, because the path names the field and can deny a
claim about a locus or an axis where the label alone could not. The decision is
a property of the text, so code makes it and the judge is not shown the label.

One kind of claim is settled the same way. A sentence that states a
non-requirement — the quantity does not prescribe, impose, require or mandate a
locus, an averaging or an aggregation — carries nothing the evidence can deny,
so code answers it "says nothing" and the judge is not asked. The settlement is
narrow on purpose. It is not applied to a claim about a name that already
carries a locus or an average or a sum, and it is not applied to a sentence
that asserts the quantity's own combination — that it is not averaged, that it
holds at one place only, or that it excludes what the source states. Those
sentences disagree with a source and the judge reads them.

The module writes no graph state. The documentation acceptance path calls its
claim comparison and veto while leaving the holistic score in shadow.
Every call goes through the shared decisions owner
(:func:`imas_codex.discovery.base.judgment.decide_batch`) under the ``sn-judge``
seat, so there is no third decisions owner.
"""

from __future__ import annotations

import logging
import math
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

# The probability at or above which a check is read as failed.
CHECK_FAIL_PROBABILITY = 0.5

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


def _question_def(question: JevQuestion) -> dict[str, Any]:
    """Render one taxonomy question as the decisions-endpoint definition."""
    if question.kind is JevQuestionKind.NOUL:
        return {"type": "noul", "instructions": question.text}
    return {
        "type": "choice",
        "instructions": question.text,
        "criteria": {option: option for option in question.options},
    }


def check_failure_probability(
    check: ReviewCheck,
    answer: Mapping[str, Any] | None = None,
    *,
    audit_finding: str | None = None,
) -> float:
    """Return the probability that this check found its named defect."""
    question = check.question
    if question is None:
        return float(audit_finding is not None)
    if not answer:
        return 0.0
    if question.kind is JevQuestionKind.NOUL:
        value = answer.get("noul")
        return (
            float(value)
            if isinstance(value, int | float) and not isinstance(value, bool)
            else 0.0
        )
    probabilities = answer.get("probabilities")
    if isinstance(probabilities, Mapping):
        return sum(
            float(probabilities.get(option, 0.0))
            for option in question.options
            if option not in question.conforming
        )
    choice = answer.get("choice")
    return float(choice is not None and choice not in question.conforming)


def check_failed(
    check: ReviewCheck,
    answer: Mapping[str, Any] | None = None,
    *,
    audit_finding: str | None = None,
) -> bool:
    """Read the check's failure probability against the review threshold."""
    if check.question is not None and not answer:
        return False
    return (
        check_failure_probability(check, answer, audit_finding=audit_finding)
        >= CHECK_FAIL_PROBABILITY
    )


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

# Fitted on coordinator-labelled documentation calibration entries after
# excluding flat or inverted inputs. The contradiction veto remains separate.
DOCS_SCORE_WEIGHTS: dict[str, float] = {
    "audit:defining_equation_missing": 1.4281029956523246,
    "audit:description_template_opening_mismatch": 3.670467788582173,
    "audit:non_canonical_equation": 0.00988088041115091,
    "audit:relationship_link_missing": 0.0958949018903927,
    "audit:sign_convention_missing": 2.293782934762642,
    "audit:symbols_undefined": 0.8544845928211299,
    "check:description_false_of_some_binding": 1.3798742789993892,
    "check:equation_physically_incorrect": 1.0913023861739364,
    "check:false_equivalence_with_related": 1.0811825425429835,
    "check:incompatible_physics_qualifier": 1.0280113111152924,
    "check:units_in_prose": 1.10749182431996,
    "dimension:completeness": 1.9758975397558163,
    "dimension:documentation_quality": 0.15368999988603704,
}
DOCS_SCORE_INTERCEPT = -15.142945942036562
CONTRADICTION_VETO_PROBABILITY = 0.30

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")
_DISPLAY_MATH = re.compile(r"\$\$.*?\$\$", re.DOTALL)
_INLINE_MATH = re.compile(r"(?<!\$)\$(?!\$)[^$\n]*\$(?!\$)")
_MARKDOWN_LINK = re.compile(r"\[([^\]]+)\]\([^)]+\)")


# Vocabulary a DD source uses to describe a field's storage shape or its place
# in a vector, and nothing else. A text whose every word is drawn from this set
# names no quantity, so no claim can be compared against it.
_FIELD_PLACEMENT_WORDS = frozenset(
    """
    a an the
    value values scalar component components axis axes
    along one each per element elements grid subset
    provided given is are be
    of in on at for to
    x y z
    """.split()
)

_LABEL_ONLY_WORD = re.compile(r"[A-Za-z]+")


def label_only_source(text: Any) -> bool:
    """True when a DD source names no quantity, only the field's placement.

    The DD catalogue holds many terse field texts — "Value", "Component along Y
    axis", "One scalar value is provided per element in the grid subset." — that
    label where the number lives rather than defining the quantity. Such a text
    has no definitional content, so it cannot deny a claim and is not evidence.

    The test is deterministic and reads the words, not a list of known strings:
    a text is label-only when every word it carries belongs to the vocabulary of
    field placement. A definition of any quantity keeps at least one content
    word, so a terse but real definition ("Height", "Toroidal angle") is spared.
    """
    if text is None:
        return True
    normalised = " ".join(str(text).split()).strip(" .,;:").lower()
    if not normalised:
        return True
    return all(
        word in _FIELD_PLACEMENT_WORDS for word in _LABEL_ONLY_WORD.findall(normalised)
    )


def source_text(source: Mapping[str, Any]) -> str:
    """Read a DD source's own words, however its reader filled the field."""
    return str(source.get("documentation") or source.get("description") or "")


_SOURCE_TEXT_FIELDS = ("documentation", "description")


def comparison_sources(
    sources: Sequence[Mapping[str, Any]] | None,
) -> list[Mapping[str, Any]]:
    """Take the field labels out of the DD sources the judge is given.

    A label names where the number lives rather than defining the DD quantity,
    so its words are not evidence and are removed from the source. Everything
    else the source carries stays: its path names the field, and a field path
    is exactly what a claim about a locus, an axis or a component can be
    denied by — ``.../unit_vector_major/y`` denies a claim that the major-axis
    component is along X. Dropping the whole source would delete that denial
    along with the label, so only the source's own words are emptied.
    """
    kept: list[Mapping[str, Any]] = []
    for source in sources or ():
        empty = {
            field
            for field in _SOURCE_TEXT_FIELDS
            if field in source and label_only_source(source[field])
        }
        if not empty:
            kept.append(source)
            continue
        kept.append({key: value for key, value in source.items() if key not in empty})
    return kept


# A documented name that carries a locus or an aggregation qualifier of its own
# already fixes where the quantity lives or how it is combined, so a sentence
# about it is not a statement about a generic quantity and the judge reads it.
_NAME_LATTICE_QUALIFIER = re.compile(
    r"(?:^|_)(?:averag\w*|mean|sum|summation|aggregat\w*|integrat\w*|"
    r"flux_surface|midplane|outboard|inboard|profile|total|median|"
    r"over|at|in|on|along|per)(?:_|$)"
)

# What a non-requirement sentence is about: a locus, a placement, or a way of
# combining values — every term a source could otherwise be read against.
_AGGREGATION_TERM = re.compile(
    r"\b(?:locus|loci|location|placement|position|aggregat\w*|averag\w*|"
    r"spatial\w*|sum|sums|summation|summed|spectral|wavenumber|"
    r"flux[- ]surface|midplane|temporal\w*|time[-_ ]average)\b"
)

# The verbs by which a sentence says a quantity imposes something. A sentence
# that negates one of these, or gives one a negative object, states a
# non-requirement: the quantity does not mandate the locus or the combination.
_REQUIREMENT_VERB = r"(?:prescrib\w*|impos\w*|requir\w*|mandat\w*|demand\w*|specif\w*)"
_NEGATED_REQUIREMENT = re.compile(
    rf"\b(?:does|do|did|is|are|was|were|need|needs)\s+not\s+"
    rf"(?:\w+\s+){{0,2}}{_REQUIREMENT_VERB}"
)
_REQUIREMENT_WITH_NO_OBJECT = re.compile(
    rf"\b{_REQUIREMENT_VERB}\s+(?:no|nothing|neither)\b"
)
_NO_MANDATORY_TERM = re.compile(
    r"\bno\s+(?:mandatory|specific|particular|intrinsic|required|fixed|"
    r"prescribed|implied|preferred)\s+\w*\s*"
)

# Sentences that assert the quantity's own combination rather than denying that
# one is required: it is not averaged, it holds at one place only, or it
# excludes what the source states. Those disagree with a source and are the
# judge's to read, however they are phrased.
_ASSERTED_COMBINATION = re.compile(
    r"\b(?:is|are|was|were)\s+(?:not|never)\s+(?:averag|aggregat|summ|integrat)\w*"
    r"|\bnot\s+an\s+average\b"
    r"|\bwithout\s+(?:\w+\s+){0,2}(?:averag|aggregat|summation)\w*"
    r"|\bpointwise\b"
    r"|\bexcludes?\b|\bomits?\b|\bonly\s+(?:the|a|an|one|at|upstream|downstream)\b"
)


def states_non_requirement(sentence: str) -> bool:
    """True when a sentence denies that its quantity requires a locus or a combination.

    "The quantity has no mandatory locus or aggregation." and "It does not
    prescribe a wavenumber sum or time or flux-surface average." state what the
    quantity does not impose, so nothing in the evidence can deny it — a source
    that names one locus of a generic quantity agrees with it.

    A sentence that instead asserts the combination itself — that the quantity
    is not averaged, holds at one place only, or excludes what the source
    states — is a claim about the evidence and is left to the judge.
    """
    text = " ".join(str(sentence or "").split()).lower()
    if not text or _ASSERTED_COMBINATION.search(text):
        return False
    if not _AGGREGATION_TERM.search(text):
        return False
    if _NEGATED_REQUIREMENT.search(text) or _REQUIREMENT_WITH_NO_OBJECT.search(text):
        return True
    return bool(_NO_MANDATORY_TERM.search(text))


def name_fixes_locus_or_aggregation(name: str) -> bool:
    """True when a documented name carries a locus or aggregation qualifier itself.

    ``electron_field_line_average_temperature_over_scrape_off_layer`` already
    fixes both the averaging and the region, so a sentence about it is not a
    statement about a generic quantity and is left to the judge.
    """
    return bool(_NAME_LATTICE_QUALIFIER.search(str(name or "").lower()))


def claim_settlement(claim: str, name: str) -> str | None:
    """The category code settles for one claim without asking the judge.

    A claim that states a non-requirement of a quantity whose name carries no
    locus or aggregation qualifier of its own carries nothing the evidence can
    deny, so it is answered "says nothing" in code. Every other claim goes to
    the judge unchanged.
    """
    if name_fixes_locus_or_aggregation(name):
        return None
    if states_non_requirement(claim):
        return "says nothing"
    return None


def deterministic_claim_determinations(
    claims: Sequence[str], name: str
) -> dict[int, str]:
    """The claims code decides itself, keyed by their index in the claim list."""
    return {
        index: outcome
        for index, claim in enumerate(claims)
        if (outcome := claim_settlement(claim, name)) is not None
    }


def _registry_texts(item: Mapping[str, Any]) -> tuple[str, ...]:
    """Find exact registry text inserted into this documentation entry."""
    from imas_standard_names.grammar.terms import AppliedConventions

    from imas_codex.standard_names.context import conventions_context_for
    from imas_codex.standard_names.docs_gates import (
        locus_definition_sentence,
        registry_template_texts,
    )

    texts = [str(value) for value in item.get("registry_sentences") or ()]
    selected = item.get("conventions")
    if selected is None and item.get("id"):
        selected = conventions_context_for(str(item["id"]))
    if not selected:
        return tuple(texts)
    conventions = AppliedConventions.model_validate(selected)
    documentation = str(item.get("documentation") or "")
    texts.extend(
        registry_template_texts(conventions.templates, documentation).rendered_texts
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
        "unit": _observed(item.get("unit")),
        "links": _observed(item.get("links")),
        "source_paths": _observed(item.get("source_paths")),
        "sibling_names": _observed(item.get("sibling_names")),
        "related_neighbours": _observed(item.get("related_neighbours")),
    }
    evidence_fields = {
        evidence_field
        for check in _docs_checks(checks or _default_checks())
        for evidence_field in check.evidence_fields
    }
    for evidence_field in sorted(
        evidence_fields - {"id", "documentation", "description"}
    ):
        if evidence_field == "dd_source_docs":
            state[evidence_field] = _observed(
                comparison_sources(item.get("dd_source_docs")) or None
            )
        else:
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
    options = claim_check.question.options
    for index, _claim in enumerate(claims):
        table[check_question_name(f"{claim_check.id}:{index}")] = {
            "type": "choice",
            "instructions": (
                f"{claim_check.question.text} Compare only "
                f"`documentation_claims[{index}]` with `dd_source_docs.value`, "
                "`description`, `unit.value`, `links.value`, "
                "`source_paths.value`, `sibling_names.value` and "
                "`related_neighbours.value`. A contradiction in any part of "
                "the sentence is a contradiction even when another part is "
                "supported. If no observed source bears on the claim, choose "
                "'says nothing'."
            ),
            "criteria": {
                option: {
                    "supports": "The DD evidence directly supports this specific claim.",
                    "contradicts": "Any part of this claim conflicts with an observed source that denies it: a different quantity, an opposite sign or direction, an incompatible unit or dimension, or a different definition.",
                    "says nothing": "The observed evidence neither supports nor contradicts this claim.",
                }[option]
                for option in options
            },
        }
    for check in docs_checks:
        if check.id != claim_check.id and check.question is not None:
            table[check_question_name(check.id)] = _question_def(check.question)
    return table


def build_docs_audit_report(
    item: Mapping[str, Any],
    *,
    catalog_names: Sequence[Mapping[str, Any]],
    known_ids: set[str],
) -> Any:
    """Run the owning deterministic checks against one documented entry."""
    from imas_codex.standard_names.review.audits import (
        AuditReport,
        run_documentation_gate_checks,
        run_house_style_checks,
        run_link_integrity,
    )

    candidate = dict(item)
    name_id = str(candidate["id"])
    return AuditReport(
        lint_findings=[
            *run_house_style_checks([candidate]),
            *run_documentation_gate_checks([candidate]),
        ],
        link_findings=[
            finding
            for finding in run_link_integrity(
                [dict(name) for name in catalog_names], known_ids=known_ids
            )
            if finding.name_id == name_id
        ],
    )


def docs_score_components(
    state: Mapping[str, Any],
    checks: Sequence[ReviewCheck],
    answers: Mapping[str, Mapping[str, Any]],
    *,
    audit_report: Any | None,
) -> dict[str, float] | None:
    """Expose calibrated score inputs while keeping absent audits unknown."""
    if audit_report is None:
        return None
    values: dict[str, float] = {}
    for dimension, criteria in _DOCS_DIMENSION_CRITERIA.items():
        answer = answers.get(dimension_question_name(dimension))
        if answer is None:
            return None
        values[f"dimension:{dimension}"] = float(
            answer.get("probabilities", {}).get(next(iter(criteria)), 0.0)
        )
    for check in _docs_checks(checks):
        if check.id == "claim_contradicts_source":
            continue
        if check.audit_finding_class is not None:
            if audit_report is not None:
                values[f"audit:{check.id}"] = float(
                    audit_finding_for(check, str(state["name_id"]), audit_report)
                    is None
                )
            continue
        answer = answers.get(check_question_name(check.id))
        if answer is None:
            return None
        if check.question.kind is JevQuestionKind.NOUL:
            values[f"check:{check.id}"] = 1 - float(answer["noul"])
        else:
            values[f"check:{check.id}"] = sum(
                float(answer.get("probabilities", {}).get(option, 0.0))
                for option in check.question.conforming
            )
    settled = settled_claim_answers(state)
    claims = [
        answers.get(check_question_name(f"claim_contradicts_source:{index}"))
        for index, _claim in enumerate(state["documentation_claims"])
        if index not in settled
    ]
    if not claims or any(answer is None for answer in claims):
        return None
    values["claim:support"] = sum(
        float(answer.get("probabilities", {}).get("supports", 0.0)) for answer in claims
    ) / len(claims)
    return values


def score_docs_review(
    state: Mapping[str, Any],
    checks: Sequence[ReviewCheck],
    answers: Mapping[str, Mapping[str, Any]],
    *,
    audit_report: Any | None,
    weights: Mapping[str, float] | None = None,
    intercept: float | None = None,
    veto_claims: bool = True,
) -> float | None:
    """Apply calibration weights after the hard contradiction veto."""
    values = docs_score_components(state, checks, answers, audit_report=audit_report)
    if values is None:
        return None
    selected_weights = DOCS_SCORE_WEIGHTS if weights is None else weights
    selected_intercept = DOCS_SCORE_INTERCEPT if intercept is None else intercept
    if any(name not in values for name in selected_weights):
        return None
    if veto_claims and any(
        claim_contradicts(answer)
        for key, answer in answers.items()
        if key.startswith("check:claim_contradicts_source:")
    ):
        return 0.0
    linear = selected_intercept + sum(
        weight * values[name] for name, weight in selected_weights.items()
    )
    return 1.0 / (1.0 + math.exp(-max(-50.0, min(50.0, linear))))


def claim_contradicts(answer: Mapping[str, Any]) -> bool:
    """Veto a selected contradiction or a substantial conflicting probability.

    A low-confidence Choice can select support while still assigning material
    probability to contradiction. A hard safety veto should retain that signal.
    """
    return (
        answer.get("choice") == "contradicts"
        or float(answer.get("probabilities", {}).get("contradicts", 0.0))
        >= CONTRADICTION_VETO_PROBABILITY
    )


def settled_claim_answers(state: Mapping[str, Any]) -> dict[int, dict[str, Any]]:
    """The answers code settles itself for one state, keyed by claim index.

    The determination reads a claim and the documented name, and the question
    table is built without either, so it is applied here and in the score,
    where the state and the answers meet.
    """
    return {
        index: {
            "choice": outcome,
            "probabilities": {outcome: 1.0},
            "settled_by": "non_requirement_sentence",
        }
        for index, outcome in deterministic_claim_determinations(
            state["documentation_claims"], str(state.get("name_id") or "")
        ).items()
    }


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
    raw_claim_answers = [
        answers.get(check_question_name(f"{claim_check.id}:{index}"))
        for index, _claim in enumerate(state["documentation_claims"])
    ]
    claim_answers = []
    contradicted_claims = []
    settled = settled_claim_answers(state)
    for index, (claim, answer) in enumerate(
        zip(state["documentation_claims"], raw_claim_answers, strict=True)
    ):
        if index in settled:
            answer = settled[index]
        elif answer is not None and claim_contradicts(answer):
            contradicted_claims.append(claim)
            if answer.get("choice") != "contradicts":
                answer = {
                    **answer,
                    "raw_choice": answer.get("choice"),
                    "choice": "contradicts",
                    "veto_source": "probability",
                }
        claim_answers.append(answer)
    contradiction_veto = bool(contradicted_claims)
    check_records: dict[str, dict[str, Any]] = {}
    claims_answered = bool(claim_answers) and all(
        answer is not None for answer in claim_answers
    )
    claim_failed = any(check_failed(claim_check, answer) for answer in claim_answers)
    check_records[claim_check.id] = {
        "answered": claims_answered,
        "failed": claim_failed,
        "failure_probability": max(
            (
                check_failure_probability(claim_check, answer)
                for answer in claim_answers
            ),
            default=0.0,
        ),
        "decision": _check_decision(claims_answered, claim_failed),
        "audit_finding": None,
        "answer": claim_answers,
    }
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
        failure_probability = check_failure_probability(
            check, answer, audit_finding=finding
        )
        failed = answered and check_failed(check, answer, audit_finding=finding)
        check_records[check.id] = {
            "answered": answered,
            "failed": failed,
            "failure_probability": failure_probability,
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
    else:
        decision = "accept"
    return {
        "decision": decision,
        "contradiction_veto": contradiction_veto,
        "contradiction_reason": (
            "contradicts source: " + "; ".join(contradicted_claims)
            if contradicted_claims
            else None
        ),
        "dimensions": dimensions,
        "raw_claim_answers": raw_claim_answers,
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
            failed = answered and check_failed(check, audit_finding=finding)
            check_records[check.id] = {
                "answered": answered,
                "failed": failed,
                "failure_probability": check_failure_probability(
                    check, audit_finding=finding
                ),
                "decision": _check_decision(answered, failed),
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
            "failure_probability": check_failure_probability(check, answer),
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
