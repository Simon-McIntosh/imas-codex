"""Jev scores each eligible family member as the family's documentation template.

The decisions layer and the house-style template accessor are stubbed, so every
test here exercises the module's own logic — eligibility, the question, the
state, the choice and the tie-break — against answers the test owns. The
``template`` sample below stands in for the openings the grammar renders; the
real accessor is exercised by :func:`test_template_accessor_reuses_the_docs_context`.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from imas_codex.standard_names import families
from imas_codex.standard_names.families import AnchorScore, score_family_anchor

QUESTION = families.TEMPLATE_FITNESS_QUESTION
TEMPLATE_MARKER = "The <base quantity> at <locus>, in SI units."


def _member(member_id: str, **extra: Any) -> dict[str, Any]:
    member: dict[str, Any] = {
        "id": member_id,
        "description": f"{member_id} described as the shared family quantity",
        "documentation": f"# {member_id}\n\nOpening sentence for this member.",
        "has_dd_source": True,
    }
    member.update(extra)
    return member


def _stub(score_for, *, cost: float = 0.0, record: list | None = None):
    """A ``decide_batch`` stand-in returning ``score_for``'s score per state.

    ``score_for(state)`` returns an expected score, or ``None`` for a member
    whose call the test wants to fail. The stub shapes each result as the
    endpoint would: an answer under the question name carrying the expected
    ``score`` and a per-level ``probabilities`` distribution. When ``record``
    is given, each call appends the ``(states, questions)`` it was handed, so a
    test can assert what the judge was shown and asked.
    """

    async def fake_decide_batch(states, questions, **kwargs):
        if record is not None:
            record.append((list(states), questions))
        results = []
        for state in states:
            score = score_for(state)
            if score is None:
                results.append(None)
                continue
            results.append(
                (
                    {
                        QUESTION: {
                            "type": "score",
                            "score": score,
                            "probabilities": {"0": 0.0, "4": 1.0},
                        }
                    },
                    cost,
                )
            )
        return results, cost

    return fake_decide_batch


@pytest.fixture
def _stub_template(monkeypatch):
    """Serve the house-style template without loading the grammar registry."""
    monkeypatch.setattr(
        families, "_house_style_template", lambda pattern_name: TEMPLATE_MARKER
    )


def _patch_decisions(monkeypatch, fake):
    monkeypatch.setattr("imas_codex.discovery.base.judgment.decide_batch", fake)


# ---------------------------------------------------------------------------
# The choice
# ---------------------------------------------------------------------------


def test_highest_expected_score_wins(monkeypatch, _stub_template):
    members = [_member("alpha"), _member("beta")]

    def score_for(state):
        return 0.4 if state["member"]["name"] == "alpha" else 0.9

    _patch_decisions(monkeypatch, _stub(score_for))

    result = asyncio.run(score_family_anchor(None, None, None, members, pattern="p"))

    assert isinstance(result, AnchorScore)
    assert result.choice == "beta"
    assert result.expected == {"alpha": 0.4, "beta": 0.9}
    assert result.asked == ("alpha", "beta")


def test_ties_fall_to_select_anchor_order(monkeypatch, _stub_template):
    # Equal judged scores; member "beta" already meets the docs standard better
    # (higher reviewer_score_docs, both accepted), so select_anchor ranks it
    # first and the tie falls to that order, not to bare id order.
    members = [
        _member("alpha", docs_stage="accepted", reviewer_score_docs=0.5),
        _member("beta", docs_stage="accepted", reviewer_score_docs=0.8),
    ]
    _patch_decisions(monkeypatch, _stub(lambda state: 0.7))

    result = asyncio.run(score_family_anchor(None, None, None, members, pattern="p"))

    assert result.expected == {"alpha": 0.7, "beta": 0.7}
    assert result.choice == "beta"


# ---------------------------------------------------------------------------
# Eligibility
# ---------------------------------------------------------------------------


def test_ineligible_member_is_never_asked(monkeypatch, _stub_template):
    members = [
        _member("good"),
        _member("placeholder", description=""),
        _member("no_docs", documentation=""),
    ]
    _patch_decisions(monkeypatch, _stub(lambda state: 0.5))

    result = asyncio.run(score_family_anchor(None, None, None, members, pattern="p"))

    assert result.asked == ("good",)
    assert set(result.expected) == {"good"}
    assert result.choice == "good"


def test_no_eligible_member_makes_no_call(monkeypatch, _stub_template):
    calls: list = []
    _patch_decisions(monkeypatch, _stub(lambda state: 0.5, record=calls))

    result = asyncio.run(
        score_family_anchor(
            None, None, None, [_member("blank", description="")], pattern="p"
        )
    )

    assert calls == []
    assert result.choice is None
    assert result.asked == ()
    assert result.expected == {}


# ---------------------------------------------------------------------------
# Failure
# ---------------------------------------------------------------------------


def test_failed_call_leaves_no_choice_when_no_member_scored(
    monkeypatch, _stub_template
):
    _patch_decisions(monkeypatch, _stub(lambda state: None))

    result = asyncio.run(
        score_family_anchor(None, None, None, [_member("only")], pattern="p")
    )

    assert result.choice is None
    assert result.expected == {}
    assert result.asked == ("only",)


def test_a_failed_member_does_not_block_the_scored_choice(monkeypatch, _stub_template):
    members = [_member("alpha"), _member("beta")]

    def score_for(state):
        return None if state["member"]["name"] == "alpha" else 0.6

    _patch_decisions(monkeypatch, _stub(score_for))

    result = asyncio.run(score_family_anchor(None, None, None, members, pattern="p"))

    assert result.expected == {"beta": 0.6}
    assert result.choice == "beta"


# ---------------------------------------------------------------------------
# The state and the question
# ---------------------------------------------------------------------------


def test_state_carries_the_house_style_template(monkeypatch, _stub_template):
    members = [_member("alpha", description="the shared quantity", documentation="D")]
    calls: list = []
    _patch_decisions(monkeypatch, _stub(lambda state: 0.5, record=calls))

    asyncio.run(
        score_family_anchor(None, None, None, members, pattern="electron|temperature")
    )

    states, _questions = calls[0]
    state = states[0]
    assert state["house_style_template"] == TEMPLATE_MARKER
    assert state["family_pattern"] == "electron|temperature"
    assert state["member"] == {
        "name": "alpha",
        "description": "the shared quantity",
        "documentation": "D",
    }


def test_question_offers_distinct_rungs_of_an_ordinal_scale(
    monkeypatch, _stub_template
):
    """The level text is the scale, so each rung must name its own situation."""
    calls: list = []
    _patch_decisions(monkeypatch, _stub(lambda state: 0.5, record=calls))

    asyncio.run(score_family_anchor(None, None, None, [_member("alpha")], pattern="p"))

    _states, questions = calls[0]
    question = questions[QUESTION]
    assert question["type"] == "score"
    criteria = question["criteria"]
    assert len(criteria) >= 2
    assert all(isinstance(rung, str) and rung.strip() for rung in criteria)
    assert len(set(criteria)) == len(criteria)
    assert "family" in question["instructions"]


def test_a_higher_level_beats_a_lower_one(monkeypatch, _stub_template):
    """The scale's top rung wins over its bottom, its extent set by the levels."""
    top = len(families.TEMPLATE_FITNESS_LEVELS) - 1
    members = [_member("alpha"), _member("beta")]

    def score_for(state):
        return 0.0 if state["member"]["name"] == "alpha" else float(top)

    _patch_decisions(monkeypatch, _stub(score_for))

    result = asyncio.run(score_family_anchor(None, None, None, members, pattern="p"))

    assert result.expected == {"alpha": 0.0, "beta": float(top)}
    assert result.choice == "beta"


def test_template_resolved_from_a_member_when_the_parent_id_does_not_parse(
    monkeypatch,
):
    """A parent id the grammar cannot parse must not empty the family's state.

    The parent leads the candidate rows, so reading the template from that row
    alone leaves it blank for a legacy or unresolvable parent id. The family's
    pattern is shared, so a member that parses yields the family's template.
    """
    calls: list = []

    def fake_template(member_id: str) -> str:
        return "" if member_id == "legacy_parent" else TEMPLATE_MARKER

    monkeypatch.setattr(families, "_house_style_template", fake_template)
    _patch_decisions(monkeypatch, _stub(lambda state: 0.5, record=calls))

    asyncio.run(
        score_family_anchor(
            "legacy_parent",
            "accepted",
            "the family's real base description",
            [_member("scorable")],
            parent_documentation="the parent's documentation",
            parent_has_dd_source=True,
            pattern="p",
        )
    )

    states, _questions = calls[0]
    assert len(states) == 2
    assert all(state["house_style_template"] == TEMPLATE_MARKER for state in states)


def test_a_family_with_no_template_is_refused_without_asking(monkeypatch):
    """No member yielding a template refuses the family rather than asking blind."""
    calls: list = []
    monkeypatch.setattr(families, "_house_style_template", lambda member_id: "")
    _patch_decisions(monkeypatch, _stub(lambda state: 0.5, record=calls))

    result = asyncio.run(
        score_family_anchor(None, None, None, [_member("alpha")], pattern="p")
    )

    assert calls == []
    assert result.choice is None
    assert result.asked == ()
    assert result.expected == {}
    assert result.reason


def test_an_accepted_parent_tied_at_the_top_wins_the_tie_break(
    monkeypatch, _stub_template
):
    """The heuristic's accepted-parent rule must fire in the tie-break too."""
    _patch_decisions(monkeypatch, _stub(lambda state: 0.7))

    result = asyncio.run(
        score_family_anchor(
            "base_quantity",
            "accepted",
            "the base quantity's real description",
            [_member("child", docs_stage="accepted", reviewer_score_docs=0.99)],
            parent_documentation="the base quantity's documentation",
            parent_has_dd_source=True,
            pattern="p",
        )
    )

    assert result.expected == {"base_quantity": 0.7, "child": 0.7}
    # select_anchor would rank the accepted parent above the child, so the tie
    # falls to the parent rather than to the higher reviewer_score_docs.
    assert result.choice == "base_quantity"


def test_parent_node_is_offered_as_a_template(monkeypatch, _stub_template):
    _patch_decisions(monkeypatch, _stub(lambda state: 0.5))

    result = asyncio.run(
        score_family_anchor(
            "base_quantity",
            "accepted",
            "the base quantity's real description",
            [_member("child")],
            parent_documentation="the base quantity's documentation",
            parent_has_dd_source=True,
            pattern="p",
        )
    )

    assert result.asked == ("base_quantity", "child")
    assert set(result.expected) == {"base_quantity", "child"}


# ---------------------------------------------------------------------------
# The template accessor reuses the docs context (no copy)
# ---------------------------------------------------------------------------


def test_template_accessor_reuses_the_docs_context(monkeypatch):
    seen: list[str] = []

    def fake_context(name: str):
        seen.append(name)
        return {
            "templates": [
                {"documentation": "Opening.", "note": "A shared note."},
                {"documentation": "Second opening.", "note": "A shared note."},
            ]
        }

    monkeypatch.setattr(
        "imas_codex.standard_names.context.conventions_context_for", fake_context
    )

    text = families._house_style_template("electron_temperature")

    assert seen == ["electron_temperature"]
    assert text == "Opening.\nA shared note.\nSecond opening."
