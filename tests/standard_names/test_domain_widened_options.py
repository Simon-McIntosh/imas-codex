"""A family's domain options are not bounded by where its sources live.

A name whose sources all sit in one part of the Data Dictionary is offered only
that part's domains when the options are drawn from the sources, so a concept
that belongs elsewhere cannot receive its domain. The widened decision offers
every domain of the band the concept is judged to belong to, so the sources'
stored domains are evidence rather than a bound.

The model is stubbed here: the test is about which options the decision is
offered and which it can therefore choose, not about the model's judgment.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from imas_codex.standard_names import domain_decision as dd

#: A concept in the heating band whose only Data Dictionary source sits in
#: transport: the family's sources cannot reach the domain the concept needs.
CONCEPT = "auxiliary_heating"
SOURCE_DOMAIN = "transport"
PLANTED = [
    {
        "id": "neutral_beam_shine_through_fraction",
        "description": "Fraction of injected neutral beam power that passes through the plasma.",
        "physics_domain": SOURCE_DOMAIN,
        "domain_correction": None,
        "sources": {
            "dd:equilibrium/time_slice/profiles_1d/neutral_beam": SOURCE_DOMAIN
        },
    }
]

#: The same concept in a family of two names whose sources disagree, so the
#: source-bounded set offers two domains and can therefore answer below the
#: floor: a family with one option is auto-decided and never reaches a second
#: judgment.
PLANTED_PAIR = [
    PLANTED[0],
    {
        "id": "neutral_beam_power_absorbed",
        "description": "Fraction of the injected neutral beam power absorbed by the plasma.",
        "physics_domain": "equilibrium",
        "domain_correction": None,
        "sources": {"dd:equilibrium/time_slice/profiles_1d/beam_power": "equilibrium"},
    },
]


def _stub(concept: str, *, domain_probability: float = 0.9, fallback: str = "") -> Any:
    """A stub that picks *concept* when it is offered and *fallback* otherwise.

    Which options the decision is offered is the subject of these tests, so the
    stub plays a model that would choose the concept's domain whenever it can.
    """

    async def decide(states, questions, **_kwargs):  # noqa: ANN001
        options = list(questions[dd.QUESTION]["criteria"])
        if concept in options:
            choice = concept
        elif fallback in options:
            choice = fallback
        else:
            choice = options[0]
        return [
            (
                {
                    dd.QUESTION: {
                        "choice": choice,
                        "probabilities": {choice: domain_probability},
                    }
                },
                0.0,
            )
        ], 0.0

    return decide


def test_the_widened_option_set_reaches_past_the_source_band() -> None:
    """A source in transport no longer pins the family to transport's band alone."""
    options = dd.widened_candidate_domains(PLANTED, {})
    assert {"equilibrium", "core_plasma_physics"} <= set(options), (
        "every domain of the source's band is offered"
    )
    assert {"auxiliary_heating", "edge_plasma_physics"} <= set(options), (
        "every band umbrella is offered, so another band is reachable"
    )
    assert "current_drive" not in options, (
        "a specific domain in an unrelated band is reached through that band's "
        "umbrella, not offered directly"
    )


def test_a_single_source_does_not_bound_the_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The concept's domain is offered though no source lives in it."""
    monkeypatch.setattr(dd, "decide_batch", _stub(CONCEPT))

    bounded = dd.family_candidate_domains(PLANTED, {})
    assert CONCEPT not in bounded, (
        "the source-bounded option set cannot reach the concept's domain; if this "
        "fails the comparison arm is not the narrow set it claims to be"
    )

    decision = asyncio.run(
        dd.decide_family_domain(
            "planted-family", PLANTED, {}, model="stub", option_set="widened"
        )
    )
    assert decision.decided, decision.undecided_reason
    assert decision.choice == CONCEPT, (
        "a name sourced only from transport must still receive its concept's "
        "domain under the widened options"
    )
    assert CONCEPT in decision.candidates
    assert set(decision.candidates) == set(dd.widened_candidate_domains(PLANTED, {}))
    assert decision.probability == pytest.approx(0.9)


def test_the_same_name_offered_only_its_sources_does_not(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The comparison arm keeps the source domain, which is the defect."""
    monkeypatch.setattr(dd, "decide_batch", _stub(CONCEPT, fallback=SOURCE_DOMAIN))

    decision = asyncio.run(
        dd.decide_family_domain(
            "planted-family", PLANTED, {}, model="stub", option_set="bounded"
        )
    )
    assert decision.decided, decision.undecided_reason
    assert decision.choice == SOURCE_DOMAIN
    assert CONCEPT not in decision.candidates


def test_a_band_below_the_floor_leaves_the_family_undecided(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A family none of whose options clears the floor writes nothing."""
    monkeypatch.setattr(dd, "decide_batch", _stub(CONCEPT, domain_probability=0.4))
    decision = asyncio.run(
        dd.decide_family_domain(
            "planted-family", PLANTED, {}, model="stub", option_set="widened"
        )
    )
    assert not decision.decided
    assert decision.choice is None
    assert "below the floor" in decision.undecided_reason


def test_the_fallback_set_widens_only_what_the_bounded_answer_cannot_place(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A family the narrow options leave undecided gets a second, wider Choice.

    The first Choice is offered only the sources' domains and answers
    ``transport`` below the floor; the second is offered the widened set and
    reaches the concept's domain. Both judgments are paid for and the record
    carries their total.
    """
    asked: list[list[str]] = []

    async def decide(states, questions, **_kwargs):  # noqa: ANN001
        options = list(questions[dd.QUESTION]["criteria"])
        asked.append(options)
        if CONCEPT in options:
            return [
                (
                    {dd.QUESTION: {"choice": CONCEPT, "probabilities": {CONCEPT: 0.9}}},
                    0.5,
                )
            ], 1.5
        return [
            (
                {
                    dd.QUESTION: {
                        "choice": SOURCE_DOMAIN,
                        "probabilities": {SOURCE_DOMAIN: 0.4},
                    }
                },
                0.5,
            )
        ], 1.5

    monkeypatch.setattr(dd, "decide_batch", decide)

    decision = asyncio.run(
        dd.decide_family_domain(
            "planted-family", PLANTED_PAIR, {}, model="stub", option_set="fallback"
        )
    )
    assert len(asked) == 2, "the second judgment runs only after the first"
    assert CONCEPT not in asked[0] and CONCEPT in asked[1]
    assert decision.decided, decision.undecided_reason
    assert decision.choice == CONCEPT
    assert decision.cost == pytest.approx(3.0)


def test_the_fallback_set_does_not_re_litigate_a_placed_family(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A family the narrow options place costs one judgment and keeps its answer."""
    asked: list[list[str]] = []

    async def decide(states, questions, **_kwargs):  # noqa: ANN001
        options = list(questions[dd.QUESTION]["criteria"])
        asked.append(options)
        return [
            (
                {
                    dd.QUESTION: {
                        "choice": SOURCE_DOMAIN,
                        "probabilities": {SOURCE_DOMAIN: 0.8},
                    }
                },
                0.5,
            )
        ], 1.5

    monkeypatch.setattr(dd, "decide_batch", decide)

    decision = asyncio.run(
        dd.decide_family_domain(
            "planted-family", PLANTED_PAIR, {}, model="stub", option_set="fallback"
        )
    )
    assert len(asked) == 1, "a decided family is not offered the wider options"
    assert decision.choice == SOURCE_DOMAIN
    assert decision.cost == pytest.approx(1.5)
