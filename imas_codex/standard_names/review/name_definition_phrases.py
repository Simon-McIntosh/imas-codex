"""Name-against-definition check as open narrowing phrases, decided in code.

A closed slot vocabulary cannot name what a definition adds, so this check lets
the text say it. One extraction call reads the description and documentation
with the standard name withheld and returns every phrase that narrows,
normalises, aggregates, weights, restricts, re-types or re-references the
quantity, quoted verbatim and drawn from no fixed list. Each phrase is then put
to the judge as one closed question — given the name's words, what the name
already implies and the recorded conventions, does the name state or imply this
phrase? — and code decides: a name is flagged when any phrase's probability
that it is narrower, different or additional clears the threshold.

The phrase vocabulary, the dispositions and the conventions are data
(:data:`PHRASE_KINDS`, :data:`DISPOSITION_BY_KIND`, :data:`CONVENTIONS`), so a
wording change is a data edit.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field

from imas_codex.standard_names.review.name_definition_probes import (
    SPECIES_TOKENS,
    name_closure,
    name_tokens,
)

logger = logging.getLogger(__name__)

SERVICE = "standard-names"

# Every way a definition can say something about the quantity that the name's
# own words may not. The list is a reading aid for the extractor, never a
# closed vocabulary: a phrase outside these kinds is kept under `other`.
PHRASE_KINDS: tuple[str, ...] = (
    "narrows",
    "normalises",
    "aggregates",
    "weights",
    "restricts",
    "re-types",
    "re-references",
    "other",
)

# The disposition class a flagged phrase implies, read by a reviewer choosing
# between renaming the quantity and rewriting its definition.
DISPOSITION_BY_KIND: dict[str, str] = {
    "narrows": "rename",
    "restricts": "rename",
    "normalises": "rewrite-the-definition-or-rename",
    "aggregates": "rewrite-the-definition-or-rename",
    "weights": "rewrite-the-definition-or-rename",
    "re-types": "rewrite-the-definition-or-rename",
    "re-references": "rewrite-the-definition-or-rename",
    "other": "rewrite-the-definition-or-rename",
}

# The judge's option that flags. The other two say the name already carries it.
FLAGGING_OPTION = "narrower_different_or_additional"

PHRASE_RELATION_QUESTION = "definition_phrase_relation"

PHRASE_RELATION_CRITERIA: dict[str, str] = {
    "stated_by_name": (
        "the name's own words state the phrase, so a reader of the name already "
        "knows it"
    ),
    "implied_by_meaning_or_convention": (
        "the phrase follows from the standard physical meaning of the name's own "
        "quantity or from one of the recorded conventions listed"
    ),
    FLAGGING_OPTION: (
        "the phrase narrows the quantity to less than the name states, or states "
        "a different form, a normalisation, a weighting, an aggregation, a "
        "restriction or an additional qualifier the name does not carry"
    ),
}

REVERSE_QUESTION = "name_qualifier_in_definition"

REVERSE_CRITERIA: dict[str, str] = {
    "stated_by_definition": "the definition states or supports the qualifier",
    "implied_by_convention": (
        "the definition leaves it unstated and a recorded convention supplies it"
    ),
    "unsupported_by_definition": (
        "the definition describes the quantity without this qualifier, so the "
        "name states more than its definition"
    ),
}


# ---------------------------------------------------------------------------
# Recorded conventions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PhraseConvention:
    """A recorded reason a phrase is not an extra qualifier.

    A convention reads the quoted phrase and the name, and suppresses the flag
    when both match. Only a convention a recorded lead ruling states belongs
    here; the ruling is named so a reader can check it.
    """

    phrase_tokens: tuple[str, ...]
    name_tokens: tuple[str, ...]
    reason: str
    # Name tokens any one of which must be present; empty means any name.
    name_subject_tokens: tuple[str, ...] = ()


CONVENTIONS: tuple[PhraseConvention, ...] = (
    PhraseConvention(
        phrase_tokens=("charge state", "charge states", "all charge"),
        name_tokens=(),
        name_subject_tokens=SPECIES_TOKENS,
        reason=(
            "a bare species name counts that species' ions summed over all charge "
            "states with neutral atoms excluded (lead ruling 2026-11-12)"
        ),
    ),
    PhraseConvention(
        phrase_tokens=("mean kinetic energy", "thermal energy", "kinetic energy"),
        name_tokens=("temperature", "temp"),
        reason="a temperature is a mean kinetic energy per particle",
    ),
    PhraseConvention(
        phrase_tokens=("per unit area", "divided by the area", "per area"),
        name_tokens=("flux",),
        reason="a flux is per unit area",
    ),
    PhraseConvention(
        phrase_tokens=("per unit volume", "divided by the volume", "per volume"),
        name_tokens=("density",),
        reason="a density is per unit volume",
    ),
    PhraseConvention(
        phrase_tokens=("cocos", "sign convention", "positive when"),
        name_tokens=(),
        name_subject_tokens=("phase", "angle", "velocity", "field", "current"),
        reason="the catalog uses COCOS 17 for sign and reference conventions",
    ),
)


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


class ExtractedPhrase(BaseModel):
    """One phrase the definition adds, quoted from the text."""

    kind: str = Field(
        description=(
            "One of: narrows, normalises, aggregates, weights, restricts, "
            "re-types, re-references, other."
        )
    )
    phrase: str = Field(
        description="The wording, quoted verbatim from the definition text."
    )
    adds: str = Field(
        description="What the phrase adds to the quantity, in a few words.",
        default="",
    )


class DefinitionPhrases(BaseModel):
    """One definition's narrowing phrases."""

    phrases: list[ExtractedPhrase] = Field(default_factory=list)


def _definition_text(item: dict[str, Any]) -> str:
    parts = [item.get("description") or "", item.get("documentation") or ""]
    return "\n\n".join(part for part in parts if part).strip()


def _item_ref(item: dict[str, Any]) -> str:
    return str(item.get("id") or item.get("name") or "")


def _phrase_kind_lines() -> str:
    return "\n".join(f"- {kind}" for kind in PHRASE_KINDS)


async def _extract_one(
    item: dict[str, Any],
    *,
    model: str,
    reasoning_effort: str | None,
) -> tuple[list[dict[str, Any]], float]:
    """One chat call's phrases for one name, with the name withheld."""

    from imas_codex.llm.prompt_loader import render_prompt

    messages = [
        {
            "role": "system",
            "content": render_prompt("sn/review_name_definition_phrases_system", {}),
        },
        {
            "role": "user",
            "content": render_prompt(
                "sn/review_name_definition_phrases_user",
                {
                    "definition": _definition_text(item),
                    "unit": item.get("unit") or "",
                    "phrase_kinds": _phrase_kind_lines(),
                },
            ),
        },
    ]
    try:
        result, cost, _ = await _call_structured(
            model=model,
            messages=messages,
            response_model=DefinitionPhrases,
            reasoning_effort=reasoning_effort,
        )
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.warning("phrase extraction failed on one definition", exc_info=True)
        return [], 0.0

    phrases: list[dict[str, Any]] = []
    for phrase in result.phrases:
        text = " ".join((phrase.phrase or "").split())
        if not text:
            continue
        kind = phrase.kind.strip().lower()
        phrases.append(
            {
                "kind": kind if kind in PHRASE_KINDS else "other",
                "phrase": text,
                "adds": phrase.adds,
                "arm": "chat",
            }
        )
    return phrases, cost


async def _call_structured(*, model, messages, response_model, reasoning_effort):
    from imas_codex.discovery.base.llm import acall_llm_structured

    return await acall_llm_structured(
        model=model,
        messages=messages,
        response_model=response_model,
        service=SERVICE,
        max_retries=2,
        reasoning_effort=reasoning_effort,
    )


async def extract_phrases(
    items: list[dict[str, Any]],
    *,
    model: str,
    concurrency: int = 8,
    reasoning_effort: str | None = None,
) -> tuple[dict[str, list[dict[str, Any]]], float]:
    """Every item's narrowing phrases from the chat seat, and the run's cost."""

    limit = asyncio.Semaphore(max(1, concurrency))

    async def one(item: dict[str, Any]):
        async with limit:
            return _item_ref(item), await _extract_one(
                item, model=model, reasoning_effort=reasoning_effort
            )

    pending = [asyncio.create_task(one(item)) for item in items]
    extracted: dict[str, list[dict[str, Any]]] = {_item_ref(item): [] for item in items}
    total_cost = 0.0
    for finished in asyncio.as_completed(pending):
        ref, (phrases, cost) = await finished
        extracted[ref] = phrases
        total_cost += cost
    return extracted, total_cost


async def extract_phrases_jev(
    items: list[dict[str, Any]],
    *,
    model: str,
    concurrency: int = 8,
) -> tuple[dict[str, list[dict[str, Any]]], float]:
    """The same extraction on the judge's own question kinds.

    The judge answers one closed ``choice`` question about the definition text
    alone: which kind of thing the text adds, if any. Its option probabilities
    are recorded as that kind's weight, so the arm produces a phrase record per
    kind the definition was read as adding rather than a verbatim quote.
    """

    from imas_codex.discovery.base.judgment import decide_batch

    criteria = {
        "none": "the text states nothing beyond the quantity",
        **{f"k{index}": kind for index, kind in enumerate(PHRASE_KINDS)},
    }
    option_of = dict(criteria)
    states = [
        {"unit": item.get("unit") or "", "definition": _definition_text(item)}
        for item in items
    ]
    results, cost = await decide_batch(
        states,
        {
            "definition_adds": {
                "type": "choice",
                "text": (
                    "Which of these, if any, does the definition text add to the "
                    "quantity it defines? Answer none when it adds nothing."
                ),
                "criteria": criteria,
            }
        },
        model=model,
        service=SERVICE,
        concurrency=concurrency,
        step="sn_judge",
    )
    extracted: dict[str, list[dict[str, Any]]] = {}
    for item, outcome in zip(items, results, strict=False):
        ref = _item_ref(item)
        answers = (outcome[0] if isinstance(outcome, tuple) else outcome) or {}
        bucket = answers.get("definition_adds") or {}
        probabilities = {
            option_of.get(str(key), str(key)): float(value)
            for key, value in (bucket.get("probabilities") or {}).items()
        }
        extracted[ref] = [
            {
                "kind": kind,
                "phrase": kind,
                "adds": "",
                "arm": "jev",
                "probability": round(probability, 4),
            }
            for kind, probability in probabilities.items()
            if kind in PHRASE_KINDS and probability >= 0.2
        ]
    return extracted, cost


# ---------------------------------------------------------------------------
# The closed question per phrase
# ---------------------------------------------------------------------------


def _conventions_text(name: str = "") -> str:
    """The conventions that can apply to this name, one per line.

    A convention offered to a name it cannot apply to reads as permission to
    excuse the phrase, so only the ones whose own tokens the name carries are
    shown; a convention with no name tokens applies to every name.
    """

    tokens = set(name_tokens(name)) if name else set()
    applicable = [
        convention
        for convention in CONVENTIONS
        if not convention.name_tokens
        and not convention.name_subject_tokens
        or (convention.name_tokens and tokens & set(convention.name_tokens))
        or (
            not convention.name_tokens
            and convention.name_subject_tokens
            and tokens & set(convention.name_subject_tokens)
        )
    ]
    if not applicable:
        return "none that apply to this name"
    return "\n".join(f"- {convention.reason}" for convention in applicable)


def relation_state(item: dict[str, Any], phrase: dict[str, Any]) -> dict[str, Any]:
    """What the judge is shown for one phrase: the name, its implications, the phrase."""

    name = _item_ref(item)
    return {
        "standard_name": name,
        "name_words": " ".join(name_tokens(name)),
        "definition_phrase": phrase.get("phrase") or "",
        "phrase_kind": phrase.get("kind") or "other",
        "name_already_states": " ".join(
            implication.reason
            for implication in name_closure(name, str(item.get("unit") or ""))
        )
        or "nothing beyond its words",
        "recorded_conventions": _conventions_text(name),
    }


async def judge_phrases(
    rows: list[dict[str, Any]],
    extracted: dict[str, list[dict[str, Any]]],
    *,
    model: str,
    concurrency: int = 8,
) -> tuple[dict[str, dict[str, Any]], float]:
    """One closed question per phrase, returning the graded record.

    The judgment never decides: it returns the option probabilities, and
    :func:`phrase_check` compares the flagging option's probability with the
    threshold tuned on the development names.
    """

    from imas_codex.discovery.base.judgment import decide_batch

    by_name = {_item_ref(row): row for row in rows}
    states: list[dict[str, Any]] = []
    index: list[tuple[str, int]] = []
    for row in rows:
        name = _item_ref(row)
        for position, phrase in enumerate(extracted.get(name) or []):
            states.append(relation_state(by_name[name], phrase))
            index.append((name, position))
    if not states:
        return {}, 0.0
    results, cost = await decide_batch(
        states,
        {
            PHRASE_RELATION_QUESTION: {
                "type": "choice",
                "text": (
                    "The definition of a physics quantity contains a phrase. The "
                    "standard name may already state it. Decide what the phrase is "
                    "relative to the name's own words, the implications listed and "
                    "the recorded conventions."
                ),
                "criteria": PHRASE_RELATION_CRITERIA,
            }
        },
        model=model,
        service=SERVICE,
        concurrency=concurrency,
        step="sn_judge",
    )
    judgments: dict[str, dict[str, Any]] = {}
    for (name, position), outcome in zip(index, results, strict=False):
        answers = (outcome[0] if isinstance(outcome, tuple) else outcome) or {}
        answer = answers.get(PHRASE_RELATION_QUESTION) or {}
        probabilities = {
            str(key): float(value)
            for key, value in (answer.get("probabilities") or {}).items()
        }
        judgments[f"{name}#{position}"] = {
            "relation": str(answer.get("choice") or ""),
            "p_flag": round(probabilities.get(FLAGGING_OPTION, 0.0), 4),
            "answered": bool(answer),
        }
    return judgments, cost


# ---------------------------------------------------------------------------
# Code decides
# ---------------------------------------------------------------------------

# The probability at or above which a phrase is read as adding something the
# name lacks. The lowest value whose development false-flag rate stayed at or
# under the ceiling while the arm still found a development mismatch; frozen before
# any scored name is read.
THRESHOLD = 0.95


def _convention_suppresses(phrase: str, name: str) -> tuple[bool, str]:
    """Whether a recorded convention explains the phrase under this name."""

    text = phrase.lower()
    tokens = set(name_tokens(name))
    for convention in CONVENTIONS:
        if not any(word in text for word in convention.phrase_tokens):
            continue
        if convention.name_tokens and not (tokens & set(convention.name_tokens)):
            continue
        if convention.name_subject_tokens and not (
            tokens & set(convention.name_subject_tokens)
        ):
            continue
        return True, convention.reason
    return False, ""


def phrase_check(
    item: dict[str, Any],
    phrases: list[dict[str, Any]],
    judgments: dict[str, dict[str, Any]],
    *,
    threshold: float = THRESHOLD,
    apply_conventions: bool = True,
    apply_judgment: bool = True,
) -> list[dict[str, Any]]:
    """The named review check's findings for one name.

    A finding carries the phrase, the name's words and the disposition class
    the phrase's kind implies. ``apply_judgment=False`` removes the closed
    question: nothing is flag-worthy, which is how the planted tests show what
    the question contributes.
    """

    if not apply_judgment:
        return []
    name = _item_ref(item)
    findings: list[dict[str, Any]] = []
    for position, phrase in enumerate(phrases):
        judgment = judgments.get(f"{name}#{position}") or {}
        if judgment.get("p_flag") is None or judgment["p_flag"] < threshold:
            continue
        kind = phrase.get("kind") or "other"
        if apply_conventions:
            suppressed, reason = _convention_suppresses(
                phrase.get("phrase") or "", name
            )
            if suppressed:
                continue
        else:
            reason = ""
        findings.append(
            {
                "finding_type": "name_definition_phrase_disagreement",
                "pattern": kind,
                "phrase": phrase.get("phrase") or "",
                "phrase_adds": phrase.get("adds") or "",
                "name_words": name_tokens(name),
                "p_flag": judgment["p_flag"],
                "relation": judgment.get("relation") or "",
                "disposition": DISPOSITION_BY_KIND.get(kind, "rewrite-the-definition"),
                "convention": reason,
            }
        )
    return findings


def phrases_hold(findings: list[dict[str, Any]]) -> str | None:
    """The review hold the findings raise, or None when the name passes."""

    if not findings:
        return None
    first = findings[0]
    return (
        f"definition adds what the name does not state ({first['pattern']}): "
        f"'{first['phrase']}'"
        + (f" (+{len(findings) - 1} more)" if len(findings) > 1 else "")
    )


# ---------------------------------------------------------------------------
# The reverse direction
# ---------------------------------------------------------------------------
# A qualifier the name states that the definition never supports is the same
# question with the roles swapped. The name side is read from a closing table
# because a name is snake_case words, not prose.

NAME_QUALIFIERS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("effective", ("effective", "apparent")),
    ("normalised", ("normalised", "normalized", "relative", "fraction")),
    ("complex", ("complex", "real_part", "imaginary_part")),
    ("averaged", ("averaged", "average", "mean")),
    ("weighted", ("weighted", "weight")),
    ("restricted", ("at_wall", "at_target", "in_divertor", "on_axis", "surface")),
    ("extensive total", ("total", "integrated", "accumulated")),
)


def name_qualifiers(name: str) -> list[str]:
    """The qualifiers a name's own words state, from the closing table."""

    tokens = set(name_tokens(name))
    return [qualifier for qualifier, words in NAME_QUALIFIERS if tokens & set(words)]


async def judge_reverse(
    rows: list[dict[str, Any]],
    *,
    model: str,
    concurrency: int = 8,
) -> tuple[dict[str, dict[str, Any]], float]:
    """One closed question per qualifier a name states, about its definition."""

    from imas_codex.discovery.base.judgment import decide_batch

    states: list[dict[str, Any]] = []
    index: list[tuple[str, str]] = []
    for row in rows:
        name = _item_ref(row)
        for qualifier in name_qualifiers(name):
            states.append(
                {
                    "standard_name": name,
                    "name_qualifier": qualifier,
                    "definition": _definition_text(row),
                    "recorded_conventions": _conventions_text(name),
                }
            )
            index.append((name, qualifier))
    if not states:
        return {}, 0.0
    results, cost = await decide_batch(
        states,
        {
            REVERSE_QUESTION: {
                "type": "choice",
                "text": (
                    "The standard name states a qualifier. Does the definition "
                    "state or support it, or does the definition describe the "
                    "quantity without it?"
                ),
                "criteria": REVERSE_CRITERIA,
            }
        },
        model=model,
        service=SERVICE,
        concurrency=concurrency,
        step="sn_judge",
    )
    judgments: dict[str, dict[str, Any]] = {}
    for (name, qualifier), outcome in zip(index, results, strict=False):
        answers = (outcome[0] if isinstance(outcome, tuple) else outcome) or {}
        answer = answers.get(REVERSE_QUESTION) or {}
        probabilities = {
            str(key): float(value)
            for key, value in (answer.get("probabilities") or {}).items()
        }
        judgments[f"{name}@{qualifier}"] = {
            "relation": str(answer.get("choice") or ""),
            "p_flag": round(probabilities.get("unsupported_by_definition", 0.0), 4),
            "answered": bool(answer),
        }
    return judgments, cost


def reverse_check(
    item: dict[str, Any],
    judgments: dict[str, dict[str, Any]],
    *,
    threshold: float = THRESHOLD,
    apply_judgment: bool = True,
) -> list[dict[str, Any]]:
    """Findings where the name states a qualifier its definition does not support."""

    if not apply_judgment:
        return []
    name = _item_ref(item)
    findings: list[dict[str, Any]] = []
    for qualifier in name_qualifiers(name):
        judgment = judgments.get(f"{name}@{qualifier}") or {}
        if judgment.get("p_flag") is None or judgment["p_flag"] < threshold:
            continue
        findings.append(
            {
                "finding_type": "name_definition_reverse_disagreement",
                "pattern": qualifier,
                "qualifier": qualifier,
                "name_words": name_tokens(name),
                "p_flag": judgment["p_flag"],
                "relation": judgment.get("relation") or "",
                "disposition": "rewrite-the-definition",
            }
        )
    return findings
