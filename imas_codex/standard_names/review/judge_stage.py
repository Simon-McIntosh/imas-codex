"""Report-only Jev-backed judge stage for the ``sn review`` pipeline.

The deterministic conformance checks in :mod:`imas_codex.standard_names.docs_gates`
decide what a pattern can: a forbidden variant written exactly as its registered
pattern, a symbol spelled as a registered forbidden spelling, an opening matching
a selected template. A pattern cannot decide the cases the registry states only as
equivalence classes — an equation a name writes in one of several equivalent ways,
a gloss that equates two loci, an opening that paraphrases a template. Those ask
the judge one categorising Choice per judged entry through the shared decisions
owner (:func:`imas_codex.discovery.base.judgment.decide_batch`), which books the
call under the ``sn-judge`` seat.

A judged entry is a selected concept that carries forbidden variants. A concept
with no forbidden variant has nothing a pattern cannot decide, and a name whose
selected entries carry none makes no call.

Every result is report-only. The finding joins the same per-name list the audit
findings use, so the reviewer sees it as information, but it is never booked into
the error-lint findings the score cap reads, so it never caps a score. Lifting a
check out of report-only is a separate measurement made against a
coordinator-labelled sample.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Any

from imas_codex.discovery.base.judgment import decide_batch
from imas_codex.discovery.base.llm import DecisionsValidationError

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


def _judge_question() -> dict[str, Any]:
    """The categorising Choice asked once per judged entry.

    The forms a check must tell apart are the ones the registry states: the
    concept's canonical form, a form the registry forbids, and the concept
    absent. The entry's concept is carried in the state, so one question serves
    every judged entry and the judge reads the concrete canonical and forbidden
    forms from ``concept``.
    """
    return {
        QUESTION_NAME: {
            "type": "choice",
            "instructions": (
                "The entry's description and documentation may write or gloss the "
                "shared concept in `concept`. Decide which one of the following the "
                "entry does for that concept, judging the written content against "
                "the concept's canonical form and its forbidden variants."
            ),
            "criteria": {
                "canonical": (
                    "Writes the concept in its canonical form `concept.canonical`, "
                    "or names the concept without writing any form listed in "
                    "`concept.forbidden`."
                ),
                "forbidden": (
                    "Writes the concept in one of the forms listed in "
                    "`concept.forbidden` — a variant the registry forbids as a "
                    "written form."
                ),
                "absent": "Does not write or gloss the concept at all.",
            },
        }
    }


def _judged_concepts(conventions: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    """The selected concept entries that need judgment.

    A concept needs judgment when it carries forbidden variants: the registry
    states those as forms a pattern cannot reliably tell apart from the
    canonical one. The notation and template entries are decided by the
    deterministic conformance checks, so they are not judged here.
    """
    if not conventions:
        return []
    return [
        dict(concept)
        for concept in conventions.get("concepts", [])
        if concept.get("forbidden_variants")
    ]


def _judge_state(name: Mapping[str, Any], concept: Mapping[str, Any]) -> dict[str, Any]:
    """Build the one judgement state for a (name, concept) judged entry."""
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
            "canonical": concept.get("equation") or "",
            "forbidden": list(concept.get("forbidden_variants", [])),
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
    about that entry through :func:`decide_batch`. The results are attached to
    the name they judged, each rendered report-only. A name with no judged entry
    contributes nothing and, when no name has one, no call is made.

    Args:
        names: The batch's name dicts, each carrying ``id``, ``description`` and
            ``documentation``.
        model: Decisions model id; defaults to the ``sn-judge`` seat.
        cost: Optional run cost tracker; the batch's summed reported cost is
            booked to it under ``sn_judge``.
        concurrency: Maximum judgements in flight at once.
        budget_seconds: Wall-time budget for the batch; ``None`` waits.

    Returns:
        ``{name_id: [finding_line, ...]}`` for every name the judge classified
        as writing a forbidden form. A name with no judged entry, and a judged
        entry the call failed on, are absent.
    """
    if model is None:
        from imas_codex.settings import get_model

        model = get_model("sn-judge")

    entries = judged_entries_for_names(names)
    if not entries:
        return {}

    states = [_judge_state(name, concept) for _name_id, name, concept in entries]
    try:
        results, _spent = await decide_batch(
            states,
            _judge_question(),
            model=model,
            service=SERVICE,
            concurrency=concurrency,
            budget_seconds=budget_seconds,
            cost=cost,
            step=COST_STEP,
        )
    except DecisionsValidationError as exc:
        # A malformed answer is deterministic; the stage reports only, so it
        # leaves the names unjudged rather than aborting the batch.
        logger.warning("judge stage rejected a malformed answer: %s", exc)
        return {}

    findings: dict[str, list[str]] = {}
    for (name_id, _name, concept), result in zip(entries, results, strict=True):
        if result is None:
            continue
        answer = result[0].get(QUESTION_NAME) or {}
        if answer.get("choice") != "forbidden":
            continue
        findings.setdefault(name_id, []).append(_finding_line(name_id, concept))
    return findings
