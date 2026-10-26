"""The fan-out proposer is told which named checks opened its search.

The refine fan-out fires on a retrieval check's failure, so the proposer is
asked to plan retrieval for a reason.  That reason is carried in the
proposer's own prompt; an empty slot renders the heading with ``(none)``
beneath it, which tells the proposer nothing and is indistinguishable from a
candidate that failed no check.

Five properties are locked here:

- the prompt names the checks that fired, with the reason each was decided
  against and the evidence the deciding route already held;
- a check-free candidate still takes the unchanged path — no trigger, empty
  excerpt, ``(none)`` in the prompt;
- a failure that does not request retrieval never reaches the excerpt, so a
  grammar-only failure cannot manufacture retrieval intent;
- a check the route answered and passed is never listed as a failure and never
  fires the trigger, even though the persisted failed-check list carries it;
- the excerpt stops at its line and character caps.

Mutations that must turn a test here red: make ``should_trigger_fanout``
return an always-empty excerpt again (``test_prompt_names_the_checks_that_fired``),
or drop the veto filter from ``check_results_from_failed_checks``
(``test_a_passing_check_never_reaches_the_excerpt_or_the_decision``).
"""

from __future__ import annotations

from typing import Any

from imas_codex.standard_names.fanout.dispatcher import _build_proposer_user_prompt
from imas_codex.standard_names.fanout.schemas import CandidateContext, FanoutScope
from imas_codex.standard_names.fanout.trigger import (
    _EXCERPT_MAX_CHARS,
    _EXCERPT_MAX_CHECKS,
    check_results_from_failed_checks,
    should_trigger_fanout,
)
from imas_codex.standard_names.review.checks import JevQuestion, ReviewCheck

THRESHOLD = 0.5

HEADING = "Named checks this candidate failed"


def _check(*, check_id: str, retrieval: bool) -> ReviewCheck:
    return ReviewCheck(
        id=check_id,
        axis="name",
        dimension="semantic",
        evidence_fields=["source_paths"],
        retrieval=retrieval,
        question=JevQuestion(
            kind="choice",
            text="Which source object does this name denote?",
            options=["different", "bound", "uncertain"],
            conforming=["bound"],
        ),
    )


def _failed_check(**overrides: Any) -> dict[str, Any]:
    """One entry in the shape a deciding route persists for a failed check."""
    entry: dict[str, Any] = {
        "id": "bound_to_wrong_object",
        "criterion": "Which source object does this name denote?",
        "evidence": "source_paths='equilibrium/time_slice/profiles_1d/zeff'",
        "probability": 0.93,
        "veto": True,
        "judgment_evidence": "The name denotes the profile, not the bound scalar.",
    }
    entry.update(overrides)
    return entry


def _candidate() -> CandidateContext:
    return CandidateContext(
        sn_id="zeff_profile",
        name="zeff_profile",
        path="equilibrium/time_slice/profiles_1d/zeff",
        description="Effective charge of the plasma profile.",
        physics_domain="equilibrium",
        chain_length=1,
    )


def _trigger(failed_checks: list[dict[str, Any]], checks: list[ReviewCheck]):
    check_trigger: dict[str, Any] = {}
    if failed_checks:
        check_trigger = {
            "checks": checks,
            "check_results": check_results_from_failed_checks(failed_checks),
            "check_threshold": THRESHOLD,
        }
    return should_trigger_fanout(**check_trigger)


def test_prompt_names_the_checks_that_fired() -> None:
    retrieval = _check(check_id="bound_to_wrong_object", retrieval=True)
    fires, excerpt = _trigger([_failed_check()], [retrieval])
    assert fires is True
    assert "bound_to_wrong_object" in excerpt
    assert "denotes the profile, not the bound scalar" in excerpt
    assert "equilibrium/time_slice/profiles_1d/zeff" in excerpt

    prompt = _build_proposer_user_prompt(_candidate(), excerpt, FanoutScope())

    assert HEADING in prompt, (
        "the proposer prompt must carry the section the excerpt is rendered "
        f"into; it read:\n{prompt}"
    )
    assert "bound_to_wrong_object" in prompt
    assert "denotes the profile, not the bound scalar" in prompt
    assert "(none)" not in prompt, (
        "a fired trigger must not leave the slot reading (none); the prompt "
        f"read:\n{prompt}"
    )


def test_a_check_free_candidate_keeps_the_empty_slot() -> None:
    fires, excerpt = _trigger(
        [], [_check(check_id="bound_to_wrong_object", retrieval=True)]
    )

    assert fires is False
    assert excerpt == ""
    prompt = _build_proposer_user_prompt(_candidate(), excerpt, FanoutScope())
    assert HEADING in prompt
    assert "(none)" in prompt, (
        "a check-free candidate must report no failure rather than invent "
        f"one; the prompt read:\n{prompt}"
    )


def test_a_non_retrieval_failure_never_reaches_the_excerpt() -> None:
    grammar = _check(check_id="grammar_only", retrieval=False)
    fires, excerpt = _trigger(
        [_failed_check(id="grammar_only", judgment_evidence="")], [grammar]
    )

    assert fires is False
    assert excerpt == ""
    prompt = _build_proposer_user_prompt(_candidate(), excerpt, FanoutScope())
    assert "grammar_only" not in prompt, (
        "a check that does not request retrieval must not be presented to the "
        f"proposer as a reason to search; the prompt read:\n{prompt}"
    )


def test_a_passing_check_never_reaches_the_excerpt_or_the_decision() -> None:
    """The persisted list carries passing checks; the trigger must not.

    ``failed_check_mirror`` style entries are the union of the route's
    vetoed checks and the checks it holds details for, so an
    answered-and-passing check sits in the same list.  Its ``veto`` flag is
    the only thing that separates it from a failure, and ignoring it both
    reports a passing check as a failure and lets it fire fan-out.
    """
    failed = _check(check_id="bound_to_wrong_object", retrieval=True)
    passed = _check(check_id="collapsed_two_quantities", retrieval=True)
    persisted = [
        _failed_check(),
        _failed_check(
            id="collapsed_two_quantities",
            veto=False,
            probability=0.05,
            judgment_evidence="One quantity, carried intact.",
        ),
    ]

    fires, excerpt = _trigger(persisted, [failed, passed])

    assert fires is True, "the vetoed check still fires the trigger"
    assert "bound_to_wrong_object" in excerpt
    assert "collapsed_two_quantities" not in excerpt, (
        "a check that was answered and passed must not be reported as a "
        f"failure; the excerpt read:\n{excerpt}"
    )
    prompt = _build_proposer_user_prompt(_candidate(), excerpt, FanoutScope())
    assert "collapsed_two_quantities" not in prompt, (
        "a passing check reached the proposer's failure section; the prompt "
        f"read:\n{prompt}"
    )


def test_a_lone_passing_check_cannot_fire_the_trigger() -> None:
    """With every persisted check passing, there is no failure to fan out on."""
    passed = _check(check_id="collapsed_two_quantities", retrieval=True)
    persisted = [
        _failed_check(id="collapsed_two_quantities", veto=False, probability=0.05)
    ]

    for threshold in (0.5, 0.0):
        fires, excerpt = should_trigger_fanout(
            checks=[passed],
            check_results=check_results_from_failed_checks(persisted),
            check_threshold=threshold,
        )
        assert fires is False, (
            "a passing check fired the trigger at threshold "
            f"{threshold}; the decision must rest on failures only"
        )
        assert excerpt == ""

    prompt = _build_proposer_user_prompt(_candidate(), "", FanoutScope())
    assert "(none)" in prompt


def test_the_excerpt_honours_its_line_and_length_caps() -> None:
    """A long failure list cannot grow the proposer's prompt without bound."""
    checks = [_check(check_id=f"check_{i}", retrieval=True) for i in range(8)]
    persisted = [
        _failed_check(
            id=f"check_{i}",
            judgment_evidence="a reason long enough to overrun the per-line cap " * 8,
        )
        for i in range(8)
    ]

    fires, excerpt = _trigger(persisted, checks)

    assert fires is True
    lines = excerpt.splitlines()
    assert len(lines) == _EXCERPT_MAX_CHECKS, (
        f"expected the line cap {_EXCERPT_MAX_CHECKS}, got {len(lines)} lines"
    )
    assert all(len(line) <= _EXCERPT_MAX_CHARS for line in lines), (
        "a line overran the character cap: "
        f"{[len(line) for line in lines]} > {_EXCERPT_MAX_CHARS}"
    )
