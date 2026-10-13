"""The fold ambiguity guard reads the change ledger, not the edge alone.

A ``REFINED_FROM`` edge records every succession a rename ever made, including
the ones a later rename revoked. The guard's ambiguity decision -- does some
identity other than the fold target currently claim the source's meaning? --
must therefore come from the ordering of ``StandardNameChange`` rows, not from
the existence of the edge. The current successor of a name is the spelling the
most recent rename or supersede event brought the meaning to, provided that
event was not itself reverted; an event back onto the source means the rename
was reverted and the edge is history, not a live claim.

The canonical case is the normalized beta family. A ``regenerate`` renamed the
normalized identity to ``beta`` at 2026-07-22T13:07:04 and a ``human_edit``
renamed it back three minutes later at 13:10:05, leaving both edges but only
one live meaning. Under the ledger rule ``normalized_toroidal_plasma_beta``
has no current successor other than the fold target, so folding it onto its
canonical spelling is unambiguous -- where the edge-existence reading refused
forever. Every own test here would fail under that older rule, pinning the
decision to the ledger.
"""

from __future__ import annotations

from tests.standard_names.test_tombstone_supersede import (
    _Graph,
    _node,
    _run,
    _State,
)

_OLD = "normalized_toroidal_plasma_beta"
_TARGET = "normalized_toroidal_beta"
_BETA = "beta"

#: The two ledger events of the reverted beta rename, by name and time.
_REGENERATE_ID = "sn-change:regenerate:nbeta-to-beta"
_REVERT_ID = "sn-change:human-edit:beta-to-nbeta"


def _reverted_beta_state() -> _State:
    """The pre-fold beta family: source accepted, canonical target a free tombstone.

    The pair that blocked the fold was ``beta``'s ``REFINED_FROM`` edge onto
    the source -- a reverted rename that the ledger records with ``regenerate``
    (source -> beta, 13:07:04) then ``human_edit`` (beta -> source, 13:10:05).
    The canonical target is a tombstone whose only lineage closes back onto the
    source (the straight refinement chain the tombstone path admits), and it
    carries no sources, parent, child or successor of its own.
    """
    state = _State(
        nodes={
            _OLD: _node(_OLD, stage="accepted", validation="valid"),
            # The canonical target is a free tombstone: no sources, parent,
            # child or successor of its own, revived by the fold itself.
            _TARGET: _node(
                _TARGET,
                stage="superseded",
                validation="valid",
                superseded_from_stage="accepted",
            ),
            _BETA: _node(_BETA, stage="accepted"),
        },
    )
    # The reverted rename residue: each spelling records the other as its
    # successor. The edge onto the source is what made the fold look ambiguous.
    state.refined_from.append((_BETA, _OLD))
    state.refined_from.append((_OLD, _BETA))
    # The canonical target's only lineage closes back onto the source.
    state.refined_from.append((_OLD, _TARGET))
    state.changes[_REGENERATE_ID] = {
        "id": _REGENERATE_ID,
        "from_name": _OLD,
        "to_name": _BETA,
        "operation": "regenerate",
        "changed_at": "2026-07-22T13:07:04Z",
        "internal": True,
    }
    state.changes[_REVERT_ID] = {
        "id": _REVERT_ID,
        "from_name": _BETA,
        "to_name": _OLD,
        "operation": "human_edit",
        "changed_at": "2026-07-22T13:10:05Z",
        "internal": True,
    }
    # The create event is owned by the successor, the revert by the source:
    # the fold's own snapshot only carries the latter, which is why the guard
    # must re-read the pair's ledger rather than trust the fold snapshot alone.
    state.change_links.append((_BETA, _REGENERATE_ID))
    state.change_links.append((_OLD, _REVERT_ID))
    return state


def test_reverted_beta_rename_clears_the_tombstone_fold_dry_run() -> None:
    """The canonical case: a reverted rename is not a current successor.

    Folding the source onto the free tombstone is unambiguous under the ledger
    rule, because ``beta``'s succession was revoked three minutes after it was
    written. The edge-existence reading of the same state refuses with "another
    successor lineage", so this test fails if the guard is reverted to it.
    """
    graph = _Graph(_reverted_beta_state())
    result = _run(graph, old=_OLD, target=_TARGET, dry_run=True)

    assert result["ok"] is True
    assert result["old_id"] == _OLD
    assert result["into_id"] == _TARGET
    assert result["dry_run"] is True
    assert graph.commits == 0
    assert graph.rollbacks == 1


def test_reverted_rename_clears_an_accepted_target_fold() -> None:
    """The same meaning rule holds when the target is a live accepted name."""
    state = _reverted_beta_state()
    state.nodes[_TARGET]["name_stage"] = "accepted"
    state.nodes[_TARGET].pop("superseded_from_stage", None)
    state.refined_from.remove((_OLD, _TARGET))
    graph = _Graph(state)

    result = _run(graph, old=_OLD, target=_TARGET, dry_run=True)

    assert result["ok"] is True
    assert graph.commits == 0
    assert graph.rollbacks == 1


def test_live_rename_is_still_a_current_successor() -> None:
    """A succession the ledger has NOT reverted still blocks the fold.

    Without the revert event the only rename between the source and ``beta`` is
    source -> beta, which makes ``beta`` the current successor and the fold
    genuinely ambiguous. This is the no-regression half of the rule: verdicts
    outside the reverted-rename set must not move.
    """
    state = _reverted_beta_state()
    state.changes.pop(_REVERT_ID, None)
    state.change_links.remove((_OLD, _REVERT_ID))
    graph = _Graph(state)

    result = _run(graph, old=_OLD, target=_TARGET, dry_run=True)

    assert result["ok"] is False
    assert "another successor lineage" in result["reason"]
    assert graph.commits == 0
    assert graph.rollbacks == 1


def test_edge_with_no_ledger_keeps_the_edge_existence_reading() -> None:
    """A successor edge the ledger predates still refuses, conservatively.

    An edge with no recorded lineage event cannot be vouched for by the ledger,
    so the guard keeps the edge-existence reading rather than silently clearing
    a fold it cannot confirm. The verification is the same shape as the beta
    case minus the ledger rows entirely.
    """
    state = _reverted_beta_state()
    state.changes.clear()
    state.change_links.clear()
    graph = _Graph(state)

    result = _run(graph, old=_OLD, target=_TARGET, dry_run=True)

    assert result["ok"] is False
    assert "another successor lineage" in result["reason"]
    assert graph.commits == 0


def test_a_reclaim_after_revert_is_current_again() -> None:
    """The ordering is what decides: the most recent event not yet reverted.

    A rename onto ``beta``, a revert back onto the source, and then a second
    rename onto ``beta`` leave ``beta`` current -- the latest event is the
    second claim, not the revert. This pins the ledger-ordering semantics
    against a naive "any revert kills the claim" reading.
    """
    state = _reverted_beta_state()
    state.nodes[_TARGET]["name_stage"] = "accepted"
    state.nodes[_TARGET].pop("superseded_from_stage", None)
    state.refined_from.remove((_OLD, _TARGET))
    reclaim_id = "sn-change:human-edit:nbeta-to-beta-again"
    state.changes[reclaim_id] = {
        "id": reclaim_id,
        "from_name": _OLD,
        "to_name": _BETA,
        "operation": "human_edit",
        "changed_at": "2026-07-23T09:12:00Z",
        "internal": True,
    }
    state.change_links.append((_BETA, reclaim_id))
    graph = _Graph(state)

    result = _run(graph, old=_OLD, target=_TARGET, dry_run=True)

    assert result["ok"] is False
    assert "another successor lineage" in result["reason"]
    assert graph.commits == 0
