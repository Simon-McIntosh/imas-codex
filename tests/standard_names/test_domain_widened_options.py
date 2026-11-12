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
CONCEPT_BAND = "auxiliary_heating"
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


def _stub(
    concept: str, band: str, *, domain_probability: float = 0.9, fallback: str = ""
) -> Any:
    """A stub that answers the band choice and the domain choice as the model would.

    The domain stage picks *concept* when it is offered and otherwise falls back
    to the first option, which is what makes the two option sets differ here.
    """

    async def decide(states, questions, **_kwargs):  # noqa: ANN001
        if dd.BAND_QUESTION in questions:
            return [
                (
                    {dd.BAND_QUESTION: {"choice": band, "probabilities": {band: 0.95}}},
                    0.0,
                )
            ], 0.0
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


def test_the_widened_option_set_is_every_domain_of_the_band() -> None:
    """All thirty-two domains are reachable, grouped into the seven bands."""
    everything = dd.all_domains()
    bands = dd.domain_bands()
    assert len(everything) == 32, "the guidance defines thirty-two domains"
    assert len(bands) == 7, "the guidance groups them into seven bands"
    reachable = {domain for band in bands for domain in dd.band_domains(band)}
    assert reachable == set(everything), "every domain sits in exactly one band"
    assert all(len(dd.band_domains(band)) <= 8 for band in bands)


def test_a_single_source_does_not_bound_the_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The concept's domain is offered though no source lives in it."""
    monkeypatch.setattr(dd, "decide_batch", _stub(CONCEPT, CONCEPT_BAND))

    bounded = dd.family_candidate_domains(PLANTED, {})
    assert CONCEPT not in bounded, (
        "the source-bounded option set cannot reach the concept's domain; if this "
        "fails the comparison arm is not the narrow set it claims to be"
    )

    decision = asyncio.run(
        dd.decide_family_domain("planted-family", PLANTED, {}, model="stub")
    )
    assert decision.decided, decision.undecided_reason
    assert decision.choice == CONCEPT, (
        "a name sourced only from transport must still receive its concept's "
        "domain under the widened options"
    )
    assert CONCEPT in decision.candidates
    assert set(decision.candidates) == set(dd.band_domains(CONCEPT_BAND))
    assert decision.probability == pytest.approx(0.95 * 0.9)


def test_the_same_name_offered_only_its_sources_does_not(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The comparison arm keeps the source domain, which is the defect."""
    monkeypatch.setattr(
        dd, "decide_batch", _stub(CONCEPT, CONCEPT_BAND, fallback=SOURCE_DOMAIN)
    )

    decision = asyncio.run(
        dd.decide_family_domain(
            "planted-family", PLANTED, {}, model="stub", bounded_by_sources=True
        )
    )
    assert decision.decided, decision.undecided_reason
    assert decision.choice == SOURCE_DOMAIN
    assert CONCEPT not in decision.candidates


def test_a_band_below_the_floor_leaves_the_family_undecided(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A family none of whose options clears the floor writes nothing."""
    monkeypatch.setattr(
        dd, "decide_batch", _stub(CONCEPT, CONCEPT_BAND, domain_probability=0.4)
    )
    decision = asyncio.run(
        dd.decide_family_domain("planted-family", PLANTED, {}, model="stub")
    )
    assert not decision.decided
    assert decision.choice is None
    assert "below the floor" in decision.undecided_reason
