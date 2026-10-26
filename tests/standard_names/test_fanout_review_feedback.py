"""The fan-out proposer is told which named checks opened its search.

The refine fan-out fires on a retrieval check's failure, so the proposer is
asked to plan retrieval for a reason.  That reason is carried in the
proposer's own prompt; an empty slot renders the heading with ``(none)``
beneath it, which tells the proposer nothing and is indistinguishable from a
candidate that failed no check.

Three properties are locked here:

- the prompt names the checks that fired, with the reason each was decided
  against and the evidence the deciding route already held;
- a check-free candidate still takes the unchanged path — no trigger, empty
  excerpt, ``(none)`` in the prompt;
- a failure that does not request retrieval never reaches the excerpt, so a
  grammar-only failure cannot manufacture retrieval intent.

Mutation that must turn ``test_prompt_names_the_checks_that_fired`` red:
make ``should_trigger_fanout`` return an always-empty excerpt again.
"""

from __future__ import annotations

from typing import Any

from imas_codex.standard_names.fanout.dispatcher import _build_proposer_user_prompt
from imas_codex.standard_names.fanout.schemas import CandidateContext, FanoutScope
from imas_codex.standard_names.fanout.trigger import (
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
