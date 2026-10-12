"""Report-only Jev-backed judge stage for the ``sn review`` pipeline.

The deterministic conformance checks in
:mod:`imas_codex.standard_names.docs_gates` decide what a pattern can: a
forbidden variant written exactly as its registered pattern, a symbol spelled
as a registered forbidden spelling, an opening matching a selected template. A
pattern cannot decide the cases the registry states only as equivalence classes
— an equation a name writes in one of several equivalent ways, a gloss that
equates two loci, an opening that paraphrases a template. Those ask the judge
one categorising Choice per judged entry through the shared decisions owner
(:func:`imas_codex.discovery.base.judgment.decide_batch`), which books the call
under the ``sn-judge`` seat.

The Choice offers the forms a check must tell apart as words a reader would
use, never as the registry's patterns. The registry states a forbidden variant
as a regular expression, so handing that expression to the model asks it to
reason about a pattern rather than about physics; a name written correctly then
reads as forbidden because the pattern for a different form still matches its
prose. Each judged concept therefore carries an authored form vocabulary
(:data:`_CONCEPT_FORMS`): a Choice whose options describe the forms in words,
and a mapping from the chosen form to the verdict the registry's pattern
states — canonical, forbidden or absent. A concept that carries forbidden
variants but has no authored form vocabulary is not judged, because judging it
would mean sending a pattern.

A judged entry is a selected concept that carries forbidden variants. A concept
with no forbidden variant has nothing a pattern cannot decide, and a name whose
selected entries carry none makes no call.

Every result is report-only. The finding joins the same per-name list the audit
findings use, so the reviewer sees it as information, but it is never booked
into the error-lint findings the score cap reads, so it never caps a score.
Lifting a check out of report-only is a separate measurement made against a
coordinator-labelled sample.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Any

from imas_codex.discovery.base.judgment import decide_batch

logger = logging.getLogger(__name__)

# Service tag the decisions call layer uses for the API key and the request
# title. The standard-names stages share it.
SERVICE = "standard-names"

# Pipeline-cost step the judge stage books its calls to.
COST_STEP = "sn_judge"

# Question name the Choice comes back under. One question is asked per judged
# entry, so the name is fixed rather than derived from the concept.
QUESTION_NAME = "concept_form"

# A finding is report-only: it carries this marker in the per-name findings the
# reviewer receives, and it is never booked as an error-lint finding, so nothing
# downstream caps a score on it.
REPORT_ONLY_MARKER = "report-only:judge:concept_form"

# The verdict a chosen form maps to. A form the registry's canonical equation
# writes, a form the registry states as a forbidden pattern, or a concept the
# entry does not write at all.
CANONICAL = "canonical"
FORBIDDEN = "forbidden"
ABSENT = "absent"

# The authored form vocabulary, keyed by the grammar token of the judged
# concept. Each entry holds the question asked of that concept, the words its
# options are described in, and the verdict each option maps to. The option
# text is a reader's description of the form — the canonical equation written in
# words, or the forbidden form described — so the registry's regular expressions
# for the forbidden variants never reach the model.
_CONCEPT_FORMS: dict[str, dict[str, Any]] = {
    "flux_surface_averaged": {
        "instructions": (
            "The entry's documentation may write an equation defining the "
            "flux-surface average. Which form does that equation take?"
        ),
        "criteria": {
            "surface": (
                "A surface integral weighted by dS/|grad psi|: the integral of "
                "f dS/|grad psi| divided by the integral of dS/|grad psi|."
            ),
            "line_poloidal": (
                "A poloidal line integral weighted by dl/B_p, using the "
                "poloidal magnetic field B_p (also written B_pol or B_theta)."
            ),
            "line_total_field": (
                "A line integral weighted by dl/B where B is the total magnetic "
                "field magnitude, not the poloidal field."
            ),
            "none": "No equation defining the flux-surface average appears.",
        },
        "verdicts": {
            "surface": CANONICAL,
            "line_poloidal": FORBIDDEN,
            "line_total_field": FORBIDDEN,
            "none": ABSENT,
        },
    },
    "plasma_boundary": {
        "instructions": (
            "How does the entry's description together with its documentation "
            "describe the plasma boundary?"
        ),
        "criteria": {
            "last_closed_flux_surface": (
                "It describes the plasma boundary as the last closed flux "
                "surface — the outermost closed magnetic flux surface of the "
                "confined plasma. It may also give the diverted-configuration "
                "name for that same surface as a synonym."
            ),
            "diverted_name_alone": (
                "It names the plasma boundary only by the term 'separatrix' — "
                "the surface bounding the confined plasma in a diverted "
                "configuration — without describing it as the last closed flux "
                "surface."
            ),
            "not_described": "It does not say what the plasma boundary is.",
        },
        "verdicts": {
            "last_closed_flux_surface": CANONICAL,
            "diverted_name_alone": FORBIDDEN,
            "not_described": ABSENT,
        },
    },
}


def _form_question(token: str) -> dict[str, Any]:
    """The categorising Choice asked once per judged entry of one concept.

    The options are the forms :data:`_CONCEPT_FORMS` authors for ``token``,
    each described in words. The entry's concept is carried in the state, so
    one question serves every judged entry of that concept.
    """
    spec = _CONCEPT_FORMS[token]
    return {
        QUESTION_NAME: {
            "type": "choice",
            "instructions": spec["instructions"],
            "criteria": dict(spec["criteria"]),
        }
    }


def _verdict(token: str, choice: str | None) -> str:
    """The verdict the registry's pattern states for a chosen form."""
    return _CONCEPT_FORMS[token]["verdicts"].get(choice, ABSENT)


def _judged_concepts(conventions: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    """The selected concept entries that need judgment.

    A concept needs judgment when it carries forbidden variants — the registry
    states those as forms a pattern cannot reliably tell apart from the
    canonical one — and the stage carries an authored form vocabulary for it.
    A concept with forbidden variants but no vocabulary is left unjudged rather
    than judged by handing the model the registry's pattern. The notation and
    template entries are decided by the deterministic conformance checks, so
    they are not judged here.
    """
    if not conventions:
        return []
    return [
        dict(concept)
        for concept in conventions.get("concepts", [])
        if concept.get("forbidden_variants") and concept.get("token") in _CONCEPT_FORMS
    ]


def _judge_state(name: Mapping[str, Any], concept: Mapping[str, Any]) -> dict[str, Any]:
    """Build the one judgement state for a (name, concept) judged entry.

    The state carries the entry's written text and the concept's token and
    phrase; it never carries the registry's forbidden-variant patterns, so no
    pattern reaches the model.
    """
    return {
        "entry": {
            "name": name.get("id") or name.get("standard_name") or "",
            "description": name.get("description") or "",
            "documentation": name.get("documentation") or "",
        },
        "concept": {
            "token": concept.get("token") or "",
            "phrase": concept.get("phrase") or "",
            "definition": concept.get("definition") or "",
        },
    }


def _finding_line(name_id: str, concept: Mapping[str, Any]) -> str:
    """Render one report-only judge finding as a single reviewer line."""
    phrase = concept.get("phrase") or concept.get("token") or ""
    return (
        f"[{REPORT_ONLY_MARKER}] {name_id}: '{phrase}' is written in a "
        f"forbidden form (report-only; does not cap the score)"
    )


def judged_entries_for_names(
    names: Sequence[Mapping[str, Any]],
) -> list[tuple[str, Mapping[str, Any], dict[str, Any]]]:
    """Pair each name with each of its selected concepts that need judgment.

    Returns ``(name_id, name, concept)`` triples in name order, concept order.
    An empty list means no judged entry, which is exactly the case that makes no
    call.
    """
    from imas_codex.standard_names.context import conventions_context_for

    entries: list[tuple[str, Mapping[str, Any], dict[str, Any]]] = []
    for name in names:
        name_id = name.get("id") or name.get("standard_name") or ""
        if not name_id:
            continue
        conventions = conventions_context_for(name_id)
        for concept in _judged_concepts(conventions):
            entries.append((name_id, name, concept))
    return entries


async def judge_entries(
    names: Sequence[Mapping[str, Any]],
    *,
    model: str | None = None,
    cost: Any | None = None,
    concurrency: int = 8,
    budget_seconds: float | None = None,
) -> list[dict[str, Any]]:
    """Ask one Choice per judged entry and return one record per entry.

    Entries are grouped by concept so each concept's question offers only its
    own forms. Every group is judged through :func:`decide_batch`, so the retry
    policy, concurrency bound and cost booking live in one place. A record
    carries the name, the concept, the question's options, the answer with its
    probabilities, the verdict the answer maps to, and the call's booked cost.

    Args:
        names: The batch's name dicts, each carrying ``id``, ``description``
            and ``documentation``.
        model: Decisions model id; defaults to the ``sn-judge`` seat.
        cost: Optional run cost tracker; every call's reported cost is booked
            to it under ``sn_judge``.
        concurrency: Maximum judgements in flight at once within a group.
        budget_seconds: Wall-time budget per group; ``None`` waits.

    Returns:
        One record per judged entry, in name order, concept order. Empty when
        no name carries a judged entry, which is exactly the case that makes no
        call.
    """
    if model is None:
        from imas_codex.settings import get_model

        model = get_model("sn-judge")

    entries = judged_entries_for_names(names)
    if not entries:
        return []

    groups: dict[str, list[tuple[str, Mapping[str, Any], dict[str, Any]]]] = {}
    for entry in entries:
        groups.setdefault(entry[2]["token"], []).append(entry)

    records: list[dict[str, Any]] = []
    for token, group in groups.items():
        question = _form_question(token)
        states = [_judge_state(name, concept) for _name_id, name, concept in group]
        results, _spent = await decide_batch(
            states,
            question,
            model=model,
            service=SERVICE,
            concurrency=concurrency,
            budget_seconds=budget_seconds,
            cost=cost,
            step=COST_STEP,
        )
        for (name_id, _name, concept), result in zip(group, results, strict=True):
            answer = (result[0].get(QUESTION_NAME) or {}) if result is not None else {}
            records.append(
                {
                    "name_id": name_id,
                    "concept_token": token,
                    "concept_phrase": concept.get("phrase") or "",
                    "question": dict(question[QUESTION_NAME]),
                    "answer": answer,
                    "verdict": _verdict(token, answer.get("choice")),
                    "cost_usd": result[1] if result is not None else 0.0,
                }
            )
    return records


async def judge_review_stage(
    names: Sequence[Mapping[str, Any]],
    *,
    model: str | None = None,
    cost: Any | None = None,
    concurrency: int = 8,
    budget_seconds: float | None = None,
) -> dict[str, list[str]]:
    """Ask one Choice per judged entry and return report-only findings by name.

    For every name whose selected concept needs judgment, one Choice is asked
    about that entry through :func:`judge_entries`. The results are attached to
    the name they judged, each rendered report-only. A name with no judged
    entry contributes nothing and, when no name has one, no call is made.

    Args:
        names: The batch's name dicts, each carrying ``id``, ``description``
            and ``documentation``.
        model: Decisions model id; defaults to the ``sn-judge`` seat.
        cost: Optional run cost tracker; the batch's summed reported cost is
            booked to it under ``sn_judge``.
        concurrency: Maximum judgements in flight at once.
        budget_seconds: Wall-time budget for each group; ``None`` waits.

    Returns:
        ``{name_id: [finding_line, ...]}`` for every name the judge classified
        as writing a forbidden form. A name with no judged entry, and a judged
        entry the call failed on, are absent.
    """
    records = await judge_entries(
        names,
        model=model,
        cost=cost,
        concurrency=concurrency,
        budget_seconds=budget_seconds,
    )

    findings: dict[str, list[str]] = {}
    for record in records:
        if record["verdict"] != FORBIDDEN:
            continue
        concept = {
            "phrase": record["concept_phrase"],
            "token": record["concept_token"],
        }
        findings.setdefault(record["name_id"], []).append(
            _finding_line(record["name_id"], concept)
        )
    return findings
