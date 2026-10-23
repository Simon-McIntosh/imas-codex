"""The contradiction veto fires only on a real contradiction.

Three failure modes are covered here. A DD source that carries a field label
instead of a definition has that label emptied from the evidence before the
claim is compared, while the source and its path stay, so the claim answers
"says nothing" about a label and can still be denied by the field the path
names. A source that denies the claim still vetoes it, and a source that
denies nothing leaves the veto clear.

Jev is mocked: these tests measure the state handed to the judge, the question
it is asked and the decision code derives, not the judge's own answer.
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

    assert state["dd_source_docs"] == {
        "observed": True,
        "value": [{"id": "spi/injector/shatter_cone/unit_vector_major/y"}],
    }

    verdict = jev_review.aggregate_docs_review(
        state, jev_review._default_checks(), _claim_only("contradicts")
    )

    assert verdict["contradiction_veto"] is True


def _claim_only(choice: str) -> dict[str, Any]:
    return {CLAIM_KEY: {"choice": choice, "probabilities": {choice: 1.0}}}


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
