"""The contradiction veto fires only on a real contradiction.

Three failure modes are covered here. A DD source that carries a field label
instead of a definition has that label emptied from the evidence before the
claim is compared, while the source and its path stay, so the claim answers
"says nothing" about a label and can still be denied by the field the path
names. A source that denies the claim still vetoes it, and a source that
denies nothing leaves the veto clear.

Jev is mocked: these tests measure the state handed to the judge, the question
it is asked and the decision code derives, not the judge's own answer.

Two more cases cover the settlement. A sentence that states a non-requirement of
a quantity whose name fixes no locus or aggregation of its own is answered "says
nothing" in code, so a source that names one locus of a generic quantity never
vetoes it. A sentence that instead asserts the combination itself — that the
quantity is not averaged, holds at one place only, or excludes what the source
states — is left to the judge and vetoes on a denial.
"""

from __future__ import annotations

from typing import Any

import pytest

from imas_codex.standard_names.review import jev_review
from imas_codex.standard_names.review.audits import AuditReport
from imas_codex.standard_names.review.checks import load_review_checks

CLAIM_KEY = "check:claim_contradicts_source:0"

# The conforming option of every non-claim choice question, so a reader that
# needs the whole table answered is given a clean sheet rather than a failing
# one. The claim questions are answered by the test that reads them.
CONFORMING_CHOICES = {
    "dimension:description_quality": "precise_definition",
    "dimension:documentation_quality": "normative_definition",
    "dimension:completeness": "complete",
    "check:description_false_of_some_binding": "true of all",
}


def _answers(checks: Any, claims: list[str]) -> dict[str, Any]:
    """A clean answer set for every question, the claim questions included.

    A claim is answered "supports" with full confidence, so a test that means to
    read a different answer for one of them replaces that key rather than
    composing the whole table.
    """
    answers: dict[str, Any] = {}
    for key, question in jev_review.docs_question_table(checks, claims).items():
        if question["type"] != "choice":
            answers[key] = {"noul": 0.0}
            continue
        if key.startswith("check:claim_contradicts_source:"):
            answers[key] = {"choice": "supports", "probabilities": {"supports": 1.0}}
            continue
        options = list(question["criteria"])
        choice = CONFORMING_CHOICES.get(key, options[0])
        assert choice in options, (key, choice)
        answers[key] = {"choice": choice}
    return answers


# The terse DD texts the catalogue files for fields whose number lives in a
# place rather than a physical quantity.
LABEL_ONLY_TEXTS = [
    "Value",
    "Component along Y axis",
    "One scalar value is provided per element in the grid subset.",
]

# Terse but real definitions: each names a quantity, so each stays evidence.
DEFINITION_TEXTS = [
    "Electrostatic potential, averaged on the magnetic flux surface",
    "Height",
    "Toroidal angle",
    "Distance to conducting wall or other conductor behind the antenna strap",
]


@pytest.mark.parametrize("text", LABEL_ONLY_TEXTS)
def test_field_label_is_not_evidence(text: str) -> None:
    assert jev_review.label_only_source(text) is True


@pytest.mark.parametrize("text", DEFINITION_TEXTS)
def test_a_definition_stays_evidence(text: str) -> None:
    assert jev_review.label_only_source(text) is False


def test_a_field_label_is_emptied_from_the_evidence() -> None:
    """The label text goes; the source and its path stay."""
    item = {
        "id": "gap_at_outboard_midplane",
        "documentation": "The quantity has no mandatory locus or aggregation.",
        "dd_source_docs": [
            {
                "id": "summary/boundary/distance_inner_outer_separatrices/value",
                "documentation": "Value",
            }
        ],
    }

    state = jev_review.build_docs_state(item)

    assert state["dd_source_docs"] == {
        "observed": True,
        "value": [{"id": "summary/boundary/distance_inner_outer_separatrices/value"}],
    }


def test_a_definition_beside_a_label_is_kept() -> None:
    item = {
        "id": "radial_coordinate_at_outboard_midplane",
        "documentation": "The quantity has no mandatory locus.",
        "dd_source_docs": [
            {"id": "a/value", "documentation": "Value"},
            {
                "id": "b/r_outboard",
                "documentation": "Radial coordinate (major radius) on the outboard side",
            },
        ],
    }

    state = jev_review.build_docs_state(item)

    kept = state["dd_source_docs"]
    assert kept["observed"] is True
    assert [source["id"] for source in kept["value"]] == ["a/value", "b/r_outboard"]
    assert [source.get("documentation") for source in kept["value"]] == [
        None,
        "Radial coordinate (major radius) on the outboard side",
    ]


def test_a_label_source_keeps_the_field_path_that_can_deny_a_claim() -> None:
    """The path names the field, so a claim about the axis is still compared.

    Every word of this source's text is a field label, but its path ends in the
    axis coordinate the label repeats, and a claim that fixes a direction is
    denied by a source whose path fixes another one.
    """
    item = {
        "id": "spi_injector_shatter_cone_unit_vector_major_component_x",
        "documentation": "The shatter-cone major-axis component is along X.",
        "dd_source_docs": [
            {
                "id": "spi/injector/shatter_cone/unit_vector_major/y",
                "documentation": "Component along Y axis",
            }
        ],
    }

    state = jev_review.build_docs_state(item)

    # The judge is mocked, but not on a fixed answer: it denies the claim only
    # when the evidence it is handed still carries the field path. A regression
    # that drops the path hands it nothing to deny, so the veto below depends on
    # what the code sent rather than on what the mock was told to say.
    verdict = jev_review.aggregate_docs_review(
        state, jev_review._default_checks(), _answer_from_evidence(state)
    )

    assert state["dd_source_docs"] == {
        "observed": True,
        "value": [{"id": "spi/injector/shatter_cone/unit_vector_major/y"}],
    }
    assert verdict["contradiction_veto"] is True


def _answer_from_evidence(state: dict[str, Any]) -> dict[str, Any]:
    """Answer the claim from the evidence the state carries, as a reader would."""
    paths = [
        str(source.get("id") or "")
        for source in (state["dd_source_docs"]["value"] or [])
    ]
    choice = (
        "contradicts"
        if any(path.endswith("/y") for path in paths)
        else ("says nothing")
    )
    return {CLAIM_KEY: {"choice": choice, "probabilities": {choice: 1.0}}}


def _claim_only(choice: str) -> dict[str, Any]:
    return {CLAIM_KEY: {"choice": choice, "probabilities": {choice: 1.0}}}


def test_a_settled_claim_is_never_vetoed_or_failed_wherever_the_answers_are_read() -> (
    None
):
    """All three readers take the settled answer, not the judge's for that claim.

    The judge's answer for the settled claim is a contradiction, which is what
    an unsettled reader would veto on and score to zero. One claim is settled
    and one is not, so the mean the score reads separates the two readings: it
    is the unsettled claim's support alone, not the pair's mean. An auditor that
    scored or vetoed the judge's answer instead fails here.
    """
    state = jev_review.build_docs_state(
        {
            "id": "electrostatic_potential",
            "documentation": (
                "The quantity has no mandatory locus or aggregation. "
                "The potential is measured at the boundary."
            ),
            "dd_source_docs": [
                {
                    "id": "edge_profiles/profiles_1d/phi_potential",
                    "documentation": (
                        "Electrostatic potential, averaged on the magnetic flux surface"
                    ),
                }
            ],
        }
    )
    claims = state["documentation_claims"]
    assert len(claims) == 2
    assert jev_review.settled_claim_answers(state) == {
        0: {
            "choice": "says nothing",
            "probabilities": {"says nothing": 1.0},
            "settled_by": "non_requirement_sentence",
        }
    }

    checks, _ = load_review_checks()
    answers = _answers(checks, claims)
    answers[CLAIM_KEY] = {
        "choice": "contradicts",
        "probabilities": {"contradicts": 1.0},
    }
    report = AuditReport()

    verdict = jev_review.aggregate_docs_review(
        state, checks, answers, audit_report=report
    )
    assert verdict["contradiction_veto"] is False
    assert verdict["claim_answers"][0]["settled_by"] == "non_requirement_sentence"

    components = jev_review.docs_score_components(
        state, checks, answers, audit_report=report
    )
    assert components is not None
    assert components["claim:support"] == 1.0

    score = jev_review.score_docs_review(state, checks, answers, audit_report=report)
    assert score is not None
    assert score > 0.0


def test_a_claim_that_fixes_its_locus_still_vetoes_on_a_denial() -> None:
    state = jev_review.build_docs_state(
        {
            "id": "distance_to_wall",
            "documentation": "The quantity is averaged on the magnetic flux surface.",
            "dd_source_docs": [
                {
                    "id": "a/y",
                    "documentation": "Distance to the wall in the major radius direction",
                }
            ],
        }
    )

    verdict = jev_review.aggregate_docs_review(
        state, jev_review._default_checks(), _claim_only("contradicts")
    )

    assert verdict["contradiction_veto"] is True


def test_the_judge_never_sees_a_field_label(monkeypatch: pytest.MonkeyPatch) -> None:
    """Drive the same sequence the acceptance path drives, with Jev mocked.

    The label's words are gone from the state the judge is handed; the source's
    path stays, which is the field name and the only coordinate it carries.
    """
    seen: dict[str, Any] = {}

    async def fake_decide_batch(
        states: list[dict[str, Any]],
        questions: dict[str, Any],
        **_kwargs: Any,
    ) -> tuple[list[Any], float]:
        seen["states"] = states
        seen["questions"] = questions
        answers = {
            key: {"choice": "says nothing", "probabilities": {"says nothing": 1.0}}
            for key in questions
            if key.startswith("check:claim_contradicts_source:")
        }
        return [(answers, 0.0) for _ in states], 0.0

    monkeypatch.setattr(jev_review, "decide_batch", fake_decide_batch)

    item = {
        "id": "x_coordinate_of_diagnostic_aperture",
        "documentation": "The coordinate is the component along the aperture axis.",
        "dd_source_docs": [
            {
                "id": "camera_ir/channel/camera/up/x",
                "documentation": "Component along X axis",
            }
        ],
    }
    state = jev_review.build_docs_state(item)
    questions = {
        key: question
        for key, question in jev_review.docs_question_table(
            jev_review._default_checks(), state["documentation_claims"]
        ).items()
        if key.startswith("check:claim_contradicts_source:")
    }

    async def run() -> None:
        results, _ = await jev_review.decide_batch(
            [state], questions, model="sn-judge", service=jev_review.SERVICE
        )
        assert results[0] is not None
        answers, _spent = results[0]
        verdict = jev_review.aggregate_docs_review(
            state, jev_review._default_checks(), answers
        )
        assert verdict["contradiction_veto"] is False

    import anyio

    anyio.run(run)

    assert seen["states"][0]["dd_source_docs"] == {
        "observed": True,
        "value": [{"id": "camera_ir/channel/camera/up/x"}],
    }


# Sentences that state what the quantity does not require. Each is generic: the
# quantity holds at any locus and under any combination, so a source that names
# one locus of it agrees rather than denies.
NON_REQUIREMENT_SENTENCES = [
    "The quantity has no mandatory locus or aggregation.",
    (
        "The unqualified physical quantity imposes no intrinsic spatial "
        "averaging; the applicable locus and aggregation convention are "
        "specified by a more specific realization."
    ),
    "It does not prescribe a wavenumber sum or time or flux-surface average.",
    (
        "The channel name does not prescribe spectral summation, temporal or "
        "flux-surface averaging, or species aggregation; these belong to the "
        "particular momentum-flux quantity being decomposed."
    ),
]

# Sentences that assert the quantity's own combination instead. Each disagrees
# with a source that states the other combination, so the judge reads it.
ASSERTED_COMBINATION_SENTENCES = [
    "The temperature is only the upstream value without downstream averaging.",
    (
        "The quantity applies pointwise to one measurement position, its "
        "assigned magnetic flux surface, and one selected magnetic separatrix, "
        "with no flux-surface, spatial, species, or time aggregation."
    ),
    "The offset vanishes on the separatrix and is negative inside it.",
    "It excludes the parallel velocity-space component.",
]


@pytest.mark.parametrize("sentence", NON_REQUIREMENT_SENTENCES)
def test_a_non_requirement_sentence_is_settled(sentence: str) -> None:
    assert jev_review.states_non_requirement(sentence) is True


@pytest.mark.parametrize("sentence", ASSERTED_COMBINATION_SENTENCES)
def test_a_sentence_that_asserts_the_combination_goes_to_the_judge(
    sentence: str,
) -> None:
    assert jev_review.states_non_requirement(sentence) is False


@pytest.mark.parametrize(
    "name",
    [
        "electron_field_line_average_temperature_over_scrape_off_layer",
        "gap_at_outboard_midplane",
        "radial_separatrix_distance_at_outboard_midplane",
        "temperature_over_scrape_off_layer",
    ],
)
def test_a_name_that_fixes_a_locus_or_an_aggregation_is_not_settled(name: str) -> None:
    assert jev_review.name_fixes_locus_or_aggregation(name) is True


@pytest.mark.parametrize(
    "name",
    [
        "electrostatic_potential",
        "perpendicular_momentum_flux_normalized_due_to_e_cross_b_drift",
        "toroidal_angle_of_control_surface",
    ],
)
def test_a_generic_name_carries_no_qualifier_of_its_own(name: str) -> None:
    assert jev_review.name_fixes_locus_or_aggregation(name) is False


# The source-comparison criterion the judge is given. The precision and recall
# bar was measured on this wording, so a criterion retuned here without a fresh
# measurement makes the evidence describe a question the product does not ask.
MEASURED_CRITERIA = {
    "supports": "The DD evidence directly supports this specific claim.",
    "contradicts": (
        "Any part of this claim conflicts with an observed source, unit, link "
        "target, sibling meaning or source path."
    ),
    "says nothing": (
        "The observed evidence neither supports nor contradicts this claim."
    ),
}


def test_the_shipped_comparison_criterion_is_the_measured_wording() -> None:
    """The question the product asks is the question the counts were measured on."""
    assert jev_review._CLAIM_COMPARISON_CRITERIA == MEASURED_CRITERIA

    table = jev_review.docs_question_table(jev_review._default_checks(), ["a claim"])
    key = next(
        name for name in table if name.startswith("check:claim_contradicts_source")
    )
    assert table[key]["criteria"] == MEASURED_CRITERIA


def test_a_generic_claim_about_a_locus_is_settled_before_the_judge() -> None:
    """A source naming one locus of a generic quantity cannot deny the claim."""
    state = jev_review.build_docs_state(
        {
            "id": "electrostatic_potential",
            "documentation": "The quantity has no mandatory locus or aggregation.",
            "dd_source_docs": [
                {
                    "id": "edge_profiles/profiles_1d/phi_potential",
                    "documentation": "Electrostatic potential, averaged on the magnetic flux surface",
                }
            ],
        }
    )

    verdict = jev_review.aggregate_docs_review(
        state, jev_review._default_checks(), _claim_only("contradicts")
    )

    assert verdict["contradiction_veto"] is False
    assert verdict["claim_answers"][0]["settled_by"] == "non_requirement_sentence"


def test_an_asserted_combination_still_vetoes_on_its_denial() -> None:
    """The sentence denies the source's averaging, so the judge reads it.

    The name is the one case that settles nothing twice over: it carries the
    average itself, and the sentence asserts the quantity's own combination
    rather than a non-requirement. The second check holds the name generic, so
    the assertion guard alone is what keeps it from the settlement.
    """
    sentence = (
        "The temperature is only the upstream value without downstream averaging."
    )

    assert (
        jev_review.claim_settlement(
            sentence, "electron_field_line_average_temperature_over_scrape_off_layer"
        )
        is None
    )
    assert jev_review.claim_settlement(sentence, "temperature") is None

    state = jev_review.build_docs_state(
        {
            "id": "electron_field_line_average_temperature_over_scrape_off_layer",
            "documentation": (
                "This quantity is the mean electron temperature along one open "
                "field line.\n\n" + sentence
            ),
            "dd_source_docs": [
                {
                    "id": "langmuir_probes/reciprocating/plunge/t_e_average",
                    "documentation": "Electron temperature (upstream to downstream average)",
                }
            ],
        }
    )

    assert jev_review.settled_claim_answers(state) == {}

    verdict = jev_review.aggregate_docs_review(
        state, jev_review._default_checks(), _claim_only("contradicts")
    )

    assert verdict["contradiction_veto"] is True


# A judge that answers "says nothing" and spreads the remainder evenly has told
# us the evidence does not bear on the claim. Reading one point of that spread as
# a denial turns an undecided distribution into a veto.
UNDECIDED_ANSWERS = [
    {
        "choice": "says nothing",
        "probabilities": {"contradicts": 0.32, "says nothing": 0.35, "supports": 0.33},
    },
    {
        "choice": "says nothing",
        "probabilities": {"contradicts": 0.30, "says nothing": 0.47, "supports": 0.23},
    },
    {
        "choice": "says nothing",
        "probabilities": {"contradicts": 0.36, "says nothing": 0.48, "supports": 0.16},
    },
]


@pytest.mark.parametrize("answer", UNDECIDED_ANSWERS)
def test_an_undecided_distribution_does_not_veto(answer: dict[str, Any]) -> None:
    assert jev_review.claim_contradicts(answer) is False


def test_a_contradiction_mass_that_leads_silence_still_vetoes() -> None:
    """The probability channel keeps the denial signal a spread cannot explain."""
    answer = {
        "choice": "supports",
        "probabilities": {"contradicts": 0.40, "says nothing": 0.25, "supports": 0.35},
    }

    assert jev_review.claim_contradicts(answer) is True


def test_a_selected_contradiction_vetoes_whatever_the_probabilities_say() -> None:
    assert (
        jev_review.claim_contradicts({"choice": "contradicts", "probabilities": {}})
        is True
    )


# A sentence that disclaims a population restriction, beside a name whose
# qualifier fixes a locus or an aggregation. The qualifier is a different axis
# from the population, so it cannot turn the disclaimer into a denial.
SCOPE_DISCLAIMERS = [
    "The quantity does not impose a species, charge-state, or thermal-population restriction.",
    (
        "The quantity imposes no species, charge-state, or thermal-population "
        "restriction."
    ),
]

POPULATION_RESTRICTIONS = [
    "The quantity applies to thermal electrons only.",
    "Only the deuterium species contributes to this quantity.",
]


@pytest.mark.parametrize("sentence", SCOPE_DISCLAIMERS)
def test_a_population_scope_disclaimer_is_settled(sentence: str) -> None:
    assert jev_review.states_scope_non_requirement(sentence) is True


@pytest.mark.parametrize("sentence", POPULATION_RESTRICTIONS)
def test_a_population_restriction_is_not_settled(sentence: str) -> None:
    assert jev_review.states_scope_non_requirement(sentence) is False


@pytest.mark.parametrize("sentence", SCOPE_DISCLAIMERS)
def test_a_scope_disclaimer_settles_under_a_name_that_fixes_an_aggregation(
    sentence: str,
) -> None:
    name = "volume_integrated_particle_source_rate"

    assert jev_review.name_fixes_locus_or_aggregation(name) is True
    assert jev_review.claim_settlement(sentence, name) == "says nothing"


def test_a_direction_word_label_is_not_evidence() -> None:
    """A component name places the value; the source path still names it."""
    text = "Toroidal component, one scalar value is provided per element in the grid subset."

    assert jev_review.label_only_source(text) is True
    assert (
        jev_review.label_only_source(
            "Radial coordinate (major radius) on the outboard side"
        )
        is False
    )


# The judge's own readings of one planted contradiction, sampled twice through
# the shipped question. Both answer "says nothing" with the silence mass
# leading, so neither is a denial; the former 0.30 probability cutoff fired on
# the one whose contradiction mass reached it, which is why both are pinned
# rather than described: they are the evidence that the fixed rule's narrower
# reading decides this row. The row is the veto bar's named recall gap.
SAMPLED_PLANTED_READINGS = [
    {"supports": 0.12, "contradicts": 0.38, "says nothing": 0.50},
    {"supports": 0.33, "contradicts": 0.25, "says nothing": 0.42},
]


@pytest.mark.parametrize("probabilities", SAMPLED_PLANTED_READINGS)
def test_a_sampled_reading_whose_silence_mass_leads_does_not_veto(
    probabilities: dict[str, float],
) -> None:
    answer = {"choice": "says nothing", "probabilities": probabilities}

    assert jev_review.claim_contradicts(answer) is False
