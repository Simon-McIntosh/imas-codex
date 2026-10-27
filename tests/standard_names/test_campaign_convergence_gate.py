"""The convergence gate halts on what the campaign changed, not on what it found.

A docs-only campaign never writes the name axis. A member that was already
outside ``accepted`` when the batch was planned cannot be claimed by the
documentation pools, so its stage is the same after the drain as before it.
Reading the absolute post-drain stage reported that member as drift and as an
acceptance failure, halting a family on state the campaign did not cause.

The fixtures carry the shape of the electrostatic-potential family re-measure:
two members whose name stage is ``reviewed`` and never moves, and two whose
documentation the drain rewrote — one accepted at the stage, one held for a
refinement round at a score above the acceptance threshold.
"""

from __future__ import annotations

from imas_codex.standard_names.campaign import (
    ConvergenceThresholds,
    evaluate_convergence,
    measure_batch,
)

ACCEPT_SCORE = 0.85


def _row(sid: str, *, name_stage: str, docs_stage: str, score: float | None = None):
    return {
        "id": sid,
        "name_stage": name_stage,
        "docs_stage": docs_stage,
        "reviewer_score_docs": score,
        "description": f"{sid} description",
        "documentation": f"{sid} documentation",
    }


# The family the member batch inherited: two names accepted, two not.
_INHERITED = [
    _row(
        "electrostatic_potential",
        name_stage="accepted",
        docs_stage="reviewed",
        score=0.849,
    ),
    _row(
        "floating_electrostatic_potential",
        name_stage="accepted",
        docs_stage="accepted",
        score=0.9375,
    ),
    _row(
        "electrostatic_potential_at_outboard_midplane",
        name_stage="reviewed",
        docs_stage="pending",
    ),
    _row(
        "electrostatic_potential_at_wall",
        name_stage="reviewed",
        docs_stage="accepted",
        score=0.8625,
    ),
]

# The same members after the drain wrote documentation for the two it could
# claim: the general name is held at `reviewed` by a contradiction veto on a
# judged score of 0.9521; the sibling is accepted at 0.9590.
_AFTER = [
    _row(
        "electrostatic_potential",
        name_stage="accepted",
        docs_stage="reviewed",
        score=0.9521,
    ),
    _row(
        "floating_electrostatic_potential",
        name_stage="accepted",
        docs_stage="accepted",
        score=0.9590,
    ),
    _row(
        "electrostatic_potential_at_outboard_midplane",
        name_stage="reviewed",
        docs_stage="pending",
    ),
    _row(
        "electrostatic_potential_at_wall",
        name_stage="reviewed",
        docs_stage="accepted",
        score=0.8625,
    ),
]

_MEMBERS = [row["id"] for row in _INHERITED]


def test_member_outside_accepted_before_the_drain_is_skipped_not_drift():
    """Two names the campaign never wrote are skipped, not reported as drift."""
    from imas_codex.standard_names.harmonize import _split_name_accepted

    eligible, skipped = _split_name_accepted(
        _MEMBERS, {row["id"]: row["name_stage"] for row in _INHERITED}
    )
    assert eligible == [
        "electrostatic_potential",
        "floating_electrostatic_potential",
    ]
    assert [entry["id"] for entry in skipped] == [
        "electrostatic_potential_at_outboard_midplane",
        "electrostatic_potential_at_wall",
    ]
    assert {entry["name_stage"] for entry in skipped} == {"reviewed"}

    outcome = measure_batch(
        _AFTER,
        eligible,
        batch_index=1,
        prior_rows=_INHERITED,
        accept_score=ACCEPT_SCORE,
    )
    assert outcome.name_drift == []
    assert outcome.touched == 2
    assert outcome.accepted == 2


def test_member_the_drain_moves_out_of_accepted_still_halts():
    """A name the batch itself moves off `accepted` is drift and halts."""
    inherited = [
        _row("moved_away", name_stage="accepted", docs_stage="accepted", score=0.94)
    ]
    after = [_row("moved_away", name_stage="reviewed", docs_stage="drafted")]

    outcome = measure_batch(
        after,
        ["moved_away"],
        batch_index=0,
        prior_rows=inherited,
        accept_score=ACCEPT_SCORE,
    )
    assert outcome.name_drift == ["moved_away"]
    assert outcome.touched == 0
    assert outcome.accepted == 0

    converged, reasons = evaluate_convergence(outcome, ConvergenceThresholds())
    assert not converged
    assert any("drift" in reason for reason in reasons)


def test_acceptance_count_follows_the_docs_the_drain_wrote():
    """The count reads the stages and scores the drain wrote, not absolutes."""
    members = ["electrostatic_potential", "floating_electrostatic_potential"]

    outcome = measure_batch(
        _AFTER,
        members,
        batch_index=1,
        prior_rows=_INHERITED,
        accept_score=ACCEPT_SCORE,
    )
    assert outcome.touched == 2
    assert outcome.accepted == 2
    assert outcome.score_accepted == ["electrostatic_potential"]
    converged, reasons = evaluate_convergence(outcome, ConvergenceThresholds())
    assert converged, reasons

    # A doc the drain left below the threshold is still an acceptance failure.
    below = [
        _row(
            "electrostatic_potential",
            name_stage="accepted",
            docs_stage="reviewed",
            score=0.7421,
        ),
        _row(
            "floating_electrostatic_potential",
            name_stage="accepted",
            docs_stage="accepted",
            score=0.9590,
        ),
    ]
    failed = measure_batch(
        below,
        members,
        batch_index=1,
        prior_rows=_INHERITED,
        accept_score=ACCEPT_SCORE,
    )
    assert failed.touched == 2
    assert failed.accepted == 1
    halted, halt_reasons = evaluate_convergence(failed, ConvergenceThresholds())
    assert not halted
    assert any("docs acceptance" in reason for reason in halt_reasons)
