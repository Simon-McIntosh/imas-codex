"""The contradiction question fires only on a real contradiction.

Two failure modes are covered here. A DD source that carries a field label
instead of a definition is not evidence, so it is dropped before the claim is
compared and the claim answers "says nothing". A source that merely specialises
the claim's quantity — one locus, average or component instance of it — is
compatible with the claim, which the question text and its criteria must say.

Jev is mocked: these tests measure the state handed to the judge and the
question it is asked, not the judge's answer.
"""

from __future__ import annotations

from typing import Any

import pytest

from imas_codex.standard_names.review import jev_review

CLAIM_KEY = "check:claim_contradicts_source:0"

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


def test_label_only_source_is_dropped_from_the_evidence() -> None:
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

    assert state["dd_source_docs"] == {"observed": False, "value": None}


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
    assert [source["id"] for source in kept["value"]] == ["b/r_outboard"]


def test_question_names_the_specialisation_rule() -> None:
    table = jev_review.docs_question_table(jev_review._default_checks(), ["a claim"])
    question = table[CLAIM_KEY]

    instructions = question["instructions"].lower()
    assert "specialises" in instructions or "specialised" in instructions
    assert "locus-specialised, averaged or component instance" in instructions
    assert "is not evidence" in instructions

    contradicts = question["criteria"]["contradicts"].lower()
    assert "opposite sign or direction" in contradicts
    assert "does not deny it" in contradicts
    assert "specialises the quantity" in contradicts
    assert "neither supports nor contradicts" in question["criteria"]["says nothing"]


def test_the_judge_never_sees_a_field_label(monkeypatch: pytest.MonkeyPatch) -> None:
    """Drive the same sequence the acceptance path drives, with Jev mocked."""
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

    assert seen["states"][0]["dd_source_docs"] == {"observed": False, "value": None}
