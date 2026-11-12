"""Name-against-definition check as narrow probes compared in code.

A judgment model is strongest on one narrow question about one text. So this
check never asks it to compare: code reads the name into a closed vocabulary of
*slots*, the model is asked one closed question per slot about the definition
text alone, and code compares the two answers slot by slot in both directions.
A candidate disagreement is then put to the model once, as one closed question,
to separate a real extra qualifier from a convention of the name's own quantity.

The vocabulary, the token table and the conventions are data (:data:`SLOTS`,
:func:`name_slots`, :data:`CONVENTIONS`) so a wording change is a data edit.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

SERVICE = "standard-names"

# The verdict a definition's slot answer carries when it states nothing about it.
NONE_STATED = "none stated"

# The disposition class each slot implies when the definition states more than
# the name or the name states more than the definition says.
SLOT_DISPOSITIONS: dict[str, str] = {
    "aggregation": "rewrite-the-definition-or-rename",
    "normalisation": "rewrite-the-definition-or-rename",
    "weighting": "rewrite-the-definition-or-rename",
    "counting_unit": "rewrite-the-definition-or-rename",
    "restriction": "rename",
    "locus": "rewrite-the-definition",
    "value_kind": "rewrite-the-definition",
}


@dataclass(frozen=True)
class Slot:
    """One closed question about the definition, and the name tokens it reads."""

    key: str
    # The question asked about the definition text alone. The name is withheld,
    # so the model reads one text and answers one narrow question.
    question: str
    options: tuple[str, ...]
    # Name tokens, as they appear in a snake_case name, that state this slot.
    # The first matching token group wins, so the table is ordered narrowest
    # first. A token that marks a default form of the quantity belongs in the
    # conventions rather than here.
    tokens: tuple[tuple[str, tuple[str, ...]], ...]
    # The probability at or above which the definition's answer is read as
    # asserted. Set on the tuning split only.
    threshold: float = 0.6
    # Options that state the absence of the property rather than a value of it.
    # An answer naming one of these is not an assertion, so it can neither add
    # a qualifier nor contradict the name.
    negative_options: tuple[str, ...] = (NONE_STATED,)
    # The option an unstatted name already means. A definition that asserts
    # this option adds nothing the name lacks, so it never raises a candidate.
    default_option: str | None = None


SLOTS: tuple[Slot, ...] = (
    Slot(
        key="aggregation",
        question=(
            "Over what is the quantity aggregated? Answer with the option that "
            "matches the definition, or 'none stated' when the definition does "
            "not say."
        ),
        options=(
            "local value at a point",
            "flux-surface average",
            "volume average",
            "line average",
            "line or surface integral",
            "volume integral",
            "time-accumulated total",
            "extremum over a set",
            NONE_STATED,
        ),
        default_option="local value at a point",
        tokens=(
            ("time-accumulated total", ("accumulated", "total", "cumulative")),
            ("extremum over a set", ("maximum", "minimum", "max", "min", "peak")),
            (
                "flux-surface average",
                ("flux_surface_average", "flux_surface_averaged", "surface_average"),
            ),
            (
                "volume average",
                ("volume_average", "volume_averaged", "averaged_over_volume"),
            ),
            ("line average", ("line_average", "line_averaged", "averaged_over_line")),
            ("line or surface integral", ("line_integral", "surface_integral")),
            ("volume integral", ("volume_integral", "integrated_over_volume")),
            ("local value at a point", ("local",)),
        ),
    ),
    Slot(
        key="normalisation",
        question=(
            "Is the quantity divided by, or expressed relative to, a scale such "
            "as a volume, an area, a time interval, a reference value or a "
            "machine radius? Answer with the option the definition states, or "
            "'none stated'."
        ),
        options=(
            "per unit volume",
            "per unit area",
            "per unit time",
            "per unit length",
            "per particle or ion",
            "relative to a reference value",
            "logarithm taken",
            NONE_STATED,
        ),
        tokens=(
            ("per unit volume", ("density", "density_of", "per_volume")),
            ("per unit time", ("rate", "frequency", "per_second")),
            ("relative to a reference value", ("normalised", "normalized")),
            ("logarithm taken", ("logarithm", "log")),
        ),
    ),
    Slot(
        key="counting_unit",
        question=(
            "What is counted? Answer 'not a count' when the quantity is not a "
            "count of discrete items."
        ),
        options=(
            "particles or ions",
            "equivalent electrons",
            "atoms or molecules",
            "not a count",
            NONE_STATED,
        ),
        negative_options=("not a count", NONE_STATED),
        tokens=(
            ("equivalent electrons", ("equivalent_electron", "equivalent_electrons")),
            ("atoms or molecules", ("atom", "atoms", "molecule", "molecules")),
            ("particles or ions", ("count", "number_of", "population", "particle")),
        ),
    ),
    Slot(
        key="restriction",
        question=(
            "Does the definition explicitly limit the quantity to one part, "
            "layer, channel, spectral band, energy channel, reaction branch or "
            "particle population, so that the other members are excluded? A "
            "phrase that merely names what the quantity belongs to, without "
            "excluding anything, is 'none stated'."
        ),
        options=(
            "one device part or component",
            "one spatial layer or region boundary",
            "one spectral band or energy channel",
            "one reaction branch",
            "one particle population",
            NONE_STATED,
        ),
        tokens=(
            (
                "one device part or component",
                (
                    "strap",
                    "limiter",
                    "divertor",
                    "target",
                    "wall",
                    "coil",
                    "antenna",
                    "reflector",
                    "injector",
                    "detector",
                    "element",
                ),
            ),
            ("one spectral band or energy channel", ("spectral", "band", "channel")),
            ("one reaction branch", ("branch", "reaction")),
            (
                "one particle population",
                ("trapped", "passing", "co_passing", "counter_passing", "runaway"),
            ),
            ("one spatial layer or region boundary", ("pedestal", "boundary", "edge")),
        ),
    ),
    Slot(
        key="weighting",
        question=(
            "Does the definition weight the quantity by a velocity, an energy or "
            "another distribution, rather than leaving it unweighted? Answer "
            "with the option stated, or 'none stated'."
        ),
        options=("weighted by velocity", "weighted by energy", "unweighted", NONE_STATED),
        tokens=(("weighted by energy", ("weighted", "weight", "moment")),),
    ),
)

SLOT_BY_KEY: dict[str, Slot] = {slot.key: slot for slot in SLOTS}


@dataclass(frozen=True)
class Convention:
    """A recorded reason an assertion in the definition is not an extra qualifier.

    A convention reads the name's own slots and the definition's answer, and
    suppresses the candidate when both match. Only conventions a recorded lead
    ruling states belong here.
    """

    slot: str
    definition_option: str
    name_tokens: tuple[str, ...]
    reason: str
    subject_tokens: tuple[str, ...] = ()


CONVENTIONS: tuple[Convention, ...] = (
    Convention(
        slot="aggregation",
        definition_option="local value at a point",
        name_tokens=(),
        reason="a density, pressure or temperature is a local fluid quantity by default",
        subject_tokens=("density", "pressure", "temperature"),
    ),
    Convention(
        slot="normalisation",
        definition_option="per unit area",
        name_tokens=("flux",),
        reason="a flux is per unit area by definition",
    ),
    Convention(
        slot="normalisation",
        definition_option="relative to a reference value",
        name_tokens=(),
        reason="the catalog uses one sign and reference convention (COCOS 17)",
        subject_tokens=("phase", "angle", "sign", "velocity", "field"),
    ),
    Convention(
        slot="aggregation",
        definition_option="flux-surface average",
        name_tokens=(),
        reason="a temperature is a mean kinetic energy over the distribution",
        subject_tokens=("temperature",),
    ),
)


def name_tokens(name: str) -> list[str]:
    """The name's words, in order, as the token table reads them."""

    return [token for token in name.lower().split("_") if token]


def name_slots(name: str) -> dict[str, str | None]:
    """The slot values a name states, read from a closed token table."""

    tokens = name_tokens(name)
    joined = "_".join(tokens)
    stated: dict[str, str | None] = {}
    for slot in SLOTS:
        value: str | None = None
        for option, words in slot.tokens:
            if any(word in joined.split("_") or word in joined for word in words):
                value = option
                break
        stated[slot.key] = value
    return stated


class SlotAnswer(BaseModel):
    """One definition's answer to one slot's closed question."""

    ref: str = Field(description="The item reference echoed back unchanged.")
    option: str = Field(description="One of the question's options, verbatim.")
    phrase: str = Field(
        description="The definition wording the answer rests on, or empty."
    )
    probability: float = Field(
        description="How strongly the definition states this option, 0 to 1.",
        default=0.0,
    )


class SlotAnswers(BaseModel):
    """One call's answers, in the order the items were given."""

    answers: list[SlotAnswer]


PROBE_SYSTEM = (
    "You read one physics definition at a time and answer one narrow closed "
    "question about it. The standard name is withheld: do not guess it and do "
    "not use it. Answer only from the definition text you are given. Choose "
    "exactly one of the question's options, verbatim, using 'none stated' when "
    "the definition does not state the property. Quote the wording your answer "
    "rests on."
)


def _definition_text(item: dict[str, Any]) -> str:
    parts = [item.get("description") or "", item.get("documentation") or ""]
    return "\n\n".join(part for part in parts if part).strip()


def _item_ref(item: dict[str, Any]) -> str:
    return str(item.get("id") or item.get("name") or "")


def _probe_user_prompt(slot: Slot, batch: list[dict[str, Any]]) -> str:
    options = "\n".join(f"- {option}" for option in slot.options)
    items = "\n\n".join(
        json.dumps(
            {
                "ref": _item_ref(item),
                "unit": item.get("unit") or "",
                "definition": _definition_text(item),
            },
            sort_keys=True,
        )
        for item in batch
    )
    return (
        f"QUESTION: {slot.question}\n\nOPTIONS:\n{options}\n\n"
        f"ITEMS (answer each one, echoing its ref):\n{items}"
    )


async def _probe_batch(
    slot: Slot,
    batch: list[dict[str, Any]],
    *,
    model: str,
    reasoning_effort: str | None = None,
) -> tuple[list[dict[str, Any]], float]:
    """One call's answers to one slot's question for one batch."""

    from imas_codex.discovery.base.llm import acall_llm_structured

    messages = [
        {"role": "system", "content": PROBE_SYSTEM},
        {"role": "user", "content": _probe_user_prompt(slot, batch)},
    ]

    def unanswered(reason: str) -> list[dict[str, Any]]:
        return [
            {
                "ref": _item_ref(item),
                "option": NONE_STATED,
                "phrase": reason,
                "probability": 0.0,
                "answered": False,
            }
            for item in batch
        ]

    try:
        result, cost, _ = await acall_llm_structured(
            model=model,
            messages=messages,
            response_model=SlotAnswers,
            service=SERVICE,
            max_retries=2,
            reasoning_effort=reasoning_effort,
        )
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.warning("definition slot probe failed on a batch", exc_info=True)
        return unanswered("the probe returned no answer"), 0.0

    returned = {answer.ref: answer for answer in result.answers}
    answers: list[dict[str, Any]] = []
    for item in batch:
        answer = returned.get(_item_ref(item))
        if answer is None:
            answers.extend(unanswered("the probe returned no answer for this item"))
            continue
        answers.append(
            {
                "ref": _item_ref(item),
                "option": answer.option if answer.option in slot.options else NONE_STATED,
                "phrase": answer.phrase,
                "probability": float(answer.probability or 0.0),
                "answered": True,
            }
        )
    return answers, cost


async def probe_definitions(
    items: list[dict[str, Any]],
    *,
    slots: tuple[Slot, ...] = SLOTS,
    model: str,
    batch_size: int = 10,
    concurrency_limit: int = 6,
    reasoning_effort: str | None = None,
) -> tuple[dict[str, dict[str, dict[str, Any]]], float]:
    """Every slot's answers for every item, plus the run's total cost.

    The result is keyed by item ref and then by slot key. A model that answers
    nothing yields ``answered=False`` rather than a silent pass.
    """

    batches: list[tuple[Slot, list[dict[str, Any]]]] = [
        (slot, items[index : index + batch_size])
        for slot in slots
        for index in range(0, len(items), batch_size)
    ]
    limit = asyncio.Semaphore(max(1, concurrency_limit))

    async def one(slot: Slot, batch: list[dict[str, Any]]):
        async with limit:
            return slot, await _probe_batch(
                slot, batch, model=model, reasoning_effort=reasoning_effort
            )

    pending = [asyncio.create_task(one(slot, batch)) for slot, batch in batches]
    answers: dict[str, dict[str, dict[str, Any]]] = {
        _item_ref(item): {} for item in items
    }
    total_cost = 0.0
    for finished in asyncio.as_completed(pending):
        slot, (batch_answers, cost) = await finished
        total_cost += cost
        for answer in batch_answers:
            answers.setdefault(answer["ref"], {})[slot.key] = answer
    return answers, total_cost


def _convention_suppresses(
    convention: Convention,
    *,
    name: str,
    definition_option: str,
) -> bool:
    """Whether a recorded convention explains the definition's assertion."""

    if definition_option != convention.definition_option:
        return False
    tokens = name_tokens(name)
    if convention.name_tokens and not any(
        token in tokens for token in convention.name_tokens
    ):
        return False
    if convention.subject_tokens and not any(
        token in tokens for token in convention.subject_tokens
    ):
        return False
    return True


def compare_slots(
    name: str,
    name_slots_stated: dict[str, str | None],
    definition_answers: dict[str, dict[str, Any]],
    *,
    slots: tuple[Slot, ...] = SLOTS,
    apply_conventions: bool = True,
    unit: str = "",
) -> list[dict[str, Any]]:
    """Candidates where the two texts disagree, in both directions.

    A candidate carries the slot, the direction, the definition's phrase and
    the name tokens that state the slot. Conventions are applied here as data,
    so a candidate they suppress never reaches the confirming question.
    """

    candidates: list[dict[str, Any]] = []
    closure = name_closure(name, unit)
    for slot in slots:
        answer = definition_answers.get(slot.key) or {}
        if not answer.get("answered", False):
            continue
        option = answer.get("option") or NONE_STATED
        probability = float(answer.get("probability") or 0.0)
        stated = name_slots_stated.get(slot.key)
        asserted = (
            option not in slot.negative_options and probability >= slot.threshold
        )

        if asserted and option == slot.default_option:
            continue
        if asserted and stated is None:
            direction = "definition states more"
        elif asserted and stated != option:
            direction = "definition contradicts the name"
        elif stated is not None and option in slot.negative_options:
            direction = "name states more"
        else:
            continue

        if any(
            implication.slot == slot.key and option in implication.options
            for implication in closure
        ):
            continue

        if apply_conventions and direction == "definition states more":
            if any(
                _convention_suppresses(
                    convention, name=name, definition_option=option
                )
                and convention.slot == slot.key
                for convention in CONVENTIONS
            ):
                continue

        candidates.append(
            {
                "slot": slot.key,
                "direction": direction,
                "definition_phrase": answer.get("phrase") or "",
                "definition_option": option,
                "name_tokens": [w for w in name_tokens(name) if w in name],
                "disposition": SLOT_DISPOSITIONS.get(slot.key, "hold"),
            }
        )
    return candidates


def probe_check(
    item: dict[str, Any],
    definition_answers: dict[str, dict[str, Any]],
    *,
    slots: tuple[Slot, ...] = SLOTS,
    apply_conventions: bool = True,
) -> list[dict[str, Any]]:
    """The named review check's findings for one name.

    A finding is the check's ``name_definition_disagreement``: the slot, the
    direction, the definition phrase and the name tokens, with the disposition
    class the slot implies.
    """

    if not definition_answers:
        return []
    name = _item_ref(item)
    found = compare_slots(
        name,
        name_slots(name),
        definition_answers,
        slots=slots,
        apply_conventions=apply_conventions,
        unit=str(item.get("unit") or ""),
    )
    for finding in found:
        finding["finding_type"] = "name_definition_probe_disagreement"
        finding["pattern"] = finding["slot"]
    return found

# ---------------------------------------------------------------------------
# What a name already implies
# ---------------------------------------------------------------------------
# A name states more than its slot tokens spell out. A density is per unit volume
# and a local value; a power is the integral over its natural extent; a species
# name counts that species' ions summed over charge states; a name at a locus is
# local there; an object name restricts to that object as a whole. Each of these
# is data, so a slot value a name already implies is never a candidate.

SPECIES_TOKENS: tuple[str, ...] = (
    "hydrogen", "deuterium", "tritium", "helium", "helium_4", "helium_3",
    "beryllium", "boron", "carbon", "nitrogen", "oxygen", "neon", "argon",
    "krypton", "xenon", "iron", "nickel", "silicon", "lithium", "tungsten",
    "electron", "ion", "ion_charge_state", "alpha", "impurity", "neutral",
    "ammonia", "deuterated",
)

# Objects a name can restrict to as a whole: the name already says the quantity
# belongs to that object, so a definition restating the same object adds nothing.
OBJECT_TOKENS: tuple[str, ...] = (
    "coil", "limiter", "divertor", "antenna", "strap", "detector", "element",
    "lens", "mirror", "gauge", "camera", "reflector", "injector", "target",
    "wall", "loop", "row", "segment",
)

# Locus prepositions: a name that places its quantity at a locus states a local
# value there.
LOCUS_PREPOSITIONS: tuple[str, ...] = ("at", "in", "near")

# Bases whose natural extent is integrated: a definition that says the quantity
# is integrated over a volume or a surface adds nothing to such a name.
EXTENSIVE_BASES: tuple[str, ...] = (
    "power", "current", "energy", "count", "force", "torque", "rate", "mass",
    "charge", "voltage", "momentum", "fluence", "fluence_rate",
)

# Unit prefixes that carry an extensive quantity, read on the unit string.
EXTENSIVE_UNIT_PREFIXES: tuple[str, ...] = (
    "W", "A", "J", "C", "N", "kg", "V", "N.m", "A.m", "W.m",
)

INTEGRAL_AGGREGATIONS: tuple[str, ...] = (
    "volume integral",
    "line or surface integral",
)


@dataclass(frozen=True)
class Implication:
    """One thing a name states, in words a confirming question can show."""

    slot: str
    options: tuple[str, ...]
    reason: str


def unit_is_per_volume(unit: str) -> bool:
    return "m^-3" in unit.replace(" ", "") or "/m^3" in unit.replace(" ", "")


def unit_is_extensive(unit: str) -> bool:
    text = unit.strip()
    return any(text.startswith(prefix) for prefix in EXTENSIVE_UNIT_PREFIXES)


def name_closure(name: str, unit: str = "") -> list[Implication]:
    """What this name already implies, as data the comparison can read.

    A slot value in this closure is never a candidate: the name states it, so a
    definition that asserts it says nothing the name lacks.
    """

    tokens = name_tokens(name)
    token_set = set(tokens)
    implied: list[Implication] = []

    intensive = "density" in token_set or unit_is_per_volume(unit)
    if intensive:
        implied.append(
            Implication(
                "normalisation",
                ("per unit volume",),
                "a density or an m^-3 unit is per unit volume",
            )
        )
        implied.append(
            Implication(
                "aggregation",
                ("local value at a point",),
                "an intensive quantity is a local value",
            )
        )

    if "flux" in token_set and "flux_surface" not in tokens:
        implied.append(
            Implication(
                "normalisation",
                ("per unit area",),
                "a flux is per unit area",
            )
        )

    extensive = bool(token_set & set(EXTENSIVE_BASES)) or unit_is_extensive(unit)
    if extensive:
        implied.append(
            Implication(
                "aggregation",
                INTEGRAL_AGGREGATIONS,
                "an extensive quantity is the integral over its natural extent",
            )
        )

    species = token_set & set(SPECIES_TOKENS)
    if species:
        implied.append(
            Implication(
                "counting_unit",
                ("particles or ions",),
                "a species name counts that species' ions summed over charge "
                "states, with neutrals excluded",
            )
        )
        implied.append(
            Implication(
                "restriction",
                ("one particle population",),
                "a species name is already restricted to that species",
            )
        )

    objects = token_set & set(OBJECT_TOKENS)
    if objects:
        implied.append(
            Implication(
                "restriction",
                ("one device part or component",),
                "an object name is already restricted to that object as a whole",
            )
        )

    if token_set & set(LOCUS_PREPOSITIONS) and len(tokens) > 1:
        implied.append(
            Implication(
                "aggregation",
                ("local value at a point",),
                "a name that places its quantity at a locus is a local value there",
            )
        )

    return implied


def closure_words(closure: list[Implication]) -> str:
    """The closure as one phrase a confirming question shows the judge."""

    if not closure:
        return "nothing beyond its stated tokens"
    return "; ".join(f"{imp.slot}: {imp.reason}" for imp in closure)


# ---------------------------------------------------------------------------
# The confirming question
# ---------------------------------------------------------------------------
# The probes say what the definition states and the closure says what the name
# already means. One closed question per surviving candidate separates the two
# cases that flag from the two that do not.

CONFIRM_QUESTION_NAME = "definition_phrase_relation"

CONFIRM_CRITERIA: dict[str, str] = {
    "already_implied": (
        "the phrase states something the name already means, as the name's own "
        "tokens or the listed implications state it"
    ),
    "narrower_part": (
        "the phrase restricts the quantity to a part, layer, channel, band or "
        "branch narrower than anything the name states"
    ),
    "different_form": (
        "the phrase states a different aggregation, normalisation, weighting or "
        "counting unit than the name's own"
    ),
    "excludes_or_default": (
        "the phrase states what the quantity excludes, or a convention of its "
        "own quantity rather than an extra qualifier"
    ),
}

FLAGGING_RELATIONS = ("narrower_part", "different_form")


def confirm_states(
    rows: list[dict[str, Any]], findings: dict[str, list[dict[str, Any]]]
) -> tuple[list[dict[str, Any]], list[tuple[str, int]]]:
    """One decision state per candidate, with the row and index it came from."""

    states: list[dict[str, Any]] = []
    index: list[tuple[str, int]] = []
    by_name = {str(row.get("id") or row.get("name") or ""): row for row in rows}
    for name, found in findings.items():
        row = by_name.get(name, {})
        closure = closure_words(name_closure(name, str(row.get("unit") or "")))
        for position, finding in enumerate(found):
            states.append(
                {
                    "standard_name": name,
                    "slot": finding["slot"],
                    "definition_phrase": finding["definition_phrase"],
                    "name_tokens": " ".join(name_tokens(name)),
                    "name_already_states": closure,
                }
            )
            index.append((name, position))
    return states, index


async def confirm_candidates(
    rows: list[dict[str, Any]],
    findings: dict[str, list[dict[str, Any]]],
    *,
    model: str,
) -> tuple[dict[str, list[dict[str, Any]]], float, dict[str, int]]:
    """Keep only the candidates the confirming question flags.

    Returns the surviving findings, the run's cost, and the count of each
    relation the judge chose, so a relation that fires on everything is visible.
    """

    from imas_codex.discovery.base.judgment import decide_batch

    states, index = confirm_states(rows, findings)
    if not states:
        return {}, 0.0, {}
    results, cost = await decide_batch(
        states,
        {CONFIRM_QUESTION_NAME: {
            "type": "choice",
            "text": (
                "The definition of a physics quantity contains a phrase. The "
                "standard name withholds some of what it already means. Decide "
                "what the phrase is relative to the name and the implications "
                "listed."
            ),
            "criteria": CONFIRM_CRITERIA,
        }},
        model=model,
        service=SERVICE,
        concurrency=6,
        step="sn_judge",
    )
    relation_counts: dict[str, int] = {}
    kept: dict[str, list[dict[str, Any]]] = {}
    for (name, position), outcome in zip(index, results, strict=False):
        finding = findings[name][position]
        if outcome is None:
            finding["confirmation"] = "unanswered"
            kept.setdefault(name, []).append(finding)
            continue
        answers = outcome[0] if isinstance(outcome, tuple) else outcome
        relation = str((answers or {}).get(CONFIRM_QUESTION_NAME, {}).get("choice") or "")
        finding["confirmation"] = relation or "unanswered"
        relation_counts[relation] = relation_counts.get(relation, 0) + 1
        if relation in FLAGGING_RELATIONS or not relation:
            kept.setdefault(name, []).append(finding)
    return kept, cost, relation_counts


def pair_of(finding: dict[str, Any]) -> str:
    """The (slot, option) pair a finding rests on."""

    return f"{finding['slot']}:{finding['definition_option']}"
