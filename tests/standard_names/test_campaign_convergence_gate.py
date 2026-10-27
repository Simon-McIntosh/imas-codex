"""The convergence gate halts on what the campaign changed, not on what it found.

A docs-only campaign never writes the name axis. A member that was already
outside ``accepted`` when the batch was planned cannot be claimed by the
documentation pools, so its stage is the same after the drain as before it.
Reading the absolute post-documentation stage reported that member as drift,
halting a family on state the campaign did not cause. Its documentation was
never touched either, so it does not enter the acceptance rate at all.

The third clause below is the opposite guarantee: a document the cascade holds
at ``reviewed`` is not accepted, whatever the review scored it. A pinned
assertion keeps that true, so a later change cannot reintroduce score-based
acceptance and let the gate pass a document the cascade refused.

The fixtures carry the shape of the electrostatic-potential family re-measure:
two members whose name stage is ``reviewed`` and never moves, and two whose
documentation the drain rewrote — one accepted at the stage, one held for a
refinement round at a score above the acceptance threshold.
"""

from __future__ import annotations

import logging

from imas_codex.standard_names.campaign import (
    ConvergenceThresholds,
    evaluate_convergence,
    measure_batch,
)
from imas_codex.standard_names.defaults import DEFAULT_MIN_SCORE
from imas_codex.standard_names.harmonize import apply_families


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
    )
    assert outcome.name_drift == []
    assert outcome.touched == 2


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
    )
    assert outcome.name_drift == ["moved_away"]
    assert outcome.touched == 0
    assert outcome.accepted == 0

    converged, reasons = evaluate_convergence(outcome, ConvergenceThresholds())
    assert not converged
    assert any("drift" in reason for reason in reasons)


def test_a_held_review_is_not_accepted_however_it_scored():
    """A held document is not accepted, whatever the review scored it.

    The contradiction veto holds a document at ``reviewed`` for a refinement
    round on purpose. Counting a held review at or above the acceptance score
    would let the gate pass a document the cascade refused, which lowers the
    acceptance bar in effect.
    """
    members = ["electrostatic_potential", "floating_electrostatic_potential"]
    held = next(row for row in _AFTER if row["id"] == "electrostatic_potential")
    assert held["docs_stage"] == "reviewed"
    assert held["reviewer_score_docs"] >= DEFAULT_MIN_SCORE

    outcome = measure_batch(_AFTER, members, batch_index=1, prior_rows=_INHERITED)
    assert outcome.touched == 2
    assert outcome.accepted == 1
    converged, reasons = evaluate_convergence(outcome, ConvergenceThresholds())
    assert not converged
    assert "docs acceptance 50.00% < 90.00% (1/2 touched)" in reasons


def test_unchanged_non_accepted_stage_is_not_drift():
    """A stage the batch did not move is not drift, accepted or not.

    This is the behavioural guard on the inherited-versus-post-drain
    comparison: the same non-accepted stage in both reads is skipped, while an
    absolute check of the post-drain stage reports drift instead.
    """
    inherited = [_row("untouched", name_stage="reviewed", docs_stage="pending")]
    after = [_row("untouched", name_stage="reviewed", docs_stage="pending")]

    outcome = measure_batch(
        after,
        ["untouched"],
        batch_index=1,
        prior_rows=inherited,
    )
    assert outcome.name_drift == []
    assert outcome.touched == 0
    assert outcome.accepted == 0


def test_member_with_no_pre_drain_row_is_drift():
    """A member the campaign never read before the drain is its own drift.

    The gate excuses a non-accepted stage only when the pre-drain read shows
    the batch inherited it. With no pre-drain row there is no inherited state
    to excuse, so the stage the drain left is the batch's own.
    """
    after = [_row("unread_before", name_stage="reviewed", docs_stage="pending")]

    outcome = measure_batch(
        after,
        ["unread_before"],
        batch_index=1,
        prior_rows=[],
    )
    assert outcome.name_drift == ["unread_before"]
    assert outcome.touched == 0


class _FamilyGraph:
    """A client stub that answers the campaign's member read from fixtures."""

    def __init__(self, rows: list[dict]) -> None:
        self.rows = rows

    def query(self, cypher: str, **params: object) -> list[dict]:
        ids = set(params.get("ids") or [])
        return [dict(row) for row in self.rows if row["id"] in ids]


def _family_member(sid: str, *, name_stage: str, docs_stage: str) -> dict:
    return {
        "id": sid,
        "physical_base": "electrostatic_potential",
        "physics_domain": "equilibrium",
        "description": f"{sid} description",
        "documentation": f"{sid} documentation",
        "docs_stage": docs_stage,
        "reviewer_score_docs": 0.9,
        "name_stage": name_stage,
        "harmonized_group_signature": None,
        "harmonized_anchor": None,
        "links": [],
        "has_dd_source": True,
    }


def test_planner_skip_reaches_the_apply_summary(monkeypatch, caplog):
    """A non-accepted member the planner skips is named by the apply.

    The skip is produced by the planner reading the name stage from the graph,
    not by a hand-written plan, and the apply's log line and returned summary
    both name the member with the stage it carried.
    """
    from types import SimpleNamespace

    from imas_codex.standard_names import harmonize
    from imas_codex.standard_names.families import PATTERN_INVARIANT_SEGMENTS

    anchor = "electrostatic_potential"
    member = "electrostatic_potential_at_wall"
    rows = [
        _family_member(anchor, name_stage="accepted", docs_stage="accepted"),
        _family_member(member, name_stage="reviewed", docs_stage="pending"),
    ]
    graph = _FamilyGraph(rows)
    monkeypatch.setattr(
        harmonize,
        "fetch_pattern_families",
        lambda gc, **_kw: [
            {
                "physical_base": "electrostatic_potential",
                "invariant": ["" for _ in PATTERN_INVARIANT_SEGMENTS],
                "n": 2,
                "members": rows,
            }
        ],
    )

    plans = harmonize.plan_family_applies(graph, min_size=1)

    assert plans[0]["decision"] == "apply"
    assert plans[0]["docs_eligible"] == [anchor]
    assert plans[0]["skipped"] == [{"id": member, "name_stage": "reviewed"}]

    class _HaltedRunner:
        def run(self, **_kwargs: object):
            return SimpleNamespace(halted=True)

    monkeypatch.setattr(harmonize, "_record_family_anchor", lambda *a, **k: None)
    monkeypatch.setattr(harmonize, "_family_docs_scores", lambda gc, ids: {})
    monkeypatch.setattr(
        harmonize, "refuse_recommended_members", lambda gc, targets, **k: {}
    )

    logger_name = "imas_codex.standard_names.harmonize"
    with caplog.at_level(logging.INFO, logger=logger_name):
        summary = harmonize.apply_families(
            graph,
            plan=plans,
            drain_fn=lambda *a, **k: None,
            runner_factory=lambda spec: _HaltedRunner(),
        )

    assert summary["skipped"] == [
        {
            "physical_base": "electrostatic_potential",
            "id": member,
            "name_stage": "reviewed",
        }
    ]
    assert summary["applied"][0]["skipped"] == [
        {"id": member, "name_stage": "reviewed"}
    ]
    assert f"member {member} skipped (name_stage=reviewed)" in caplog.text
