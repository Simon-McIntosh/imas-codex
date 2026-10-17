"""Folding into a superseded tombstone restores the target's catalog status.

Folding a live identity onto a tombstoned spelling revives the tombstone: the
target takes back a live ``name_stage``. The catalog ``status`` is a separate
axis, and 'superseded' is the terminal value; export reads it to exclude a
spelling, so a revived identity that kept it would be live by stage and
invisible to export at once. Revival restores ``status='draft'`` in the same
fold statement, and the fold receipt records both sides of that transition.
"""

from __future__ import annotations

import json
from typing import Any

from tests.standard_names.test_supersede_into_tombstone import (
    _RevivingGraph,
    _RevivingTransaction,
)
from tests.standard_names.test_tombstone_supersede import (
    _run,
    _Session,
    _state,
)

_TARGET = "electron_density"


class _StatusTransaction(_RevivingTransaction):
    """Mirror the production fold statement's own declared status write.

    The base mock applies the ``name_stage`` revival; this subclass applies the
    catalog ``status`` mutation exactly when the fold statement declares it, so
    a statement with the status assignment removed leaves the target
    tombstoned and the postflight proof refuses the fold.
    """

    def run(self, cypher: str, **params: Any) -> list[dict[str, Any]]:
        rows = super().run(cypher, **params)
        if (
            "ATOMIC_FOLD_MUTATE_NAMES" in cypher
            and "target.status" in cypher
            and params.get("target_revived_status") is not None
        ):
            self.state.nodes[params["into_id"]]["status"] = params[
                "target_revived_status"
            ]
        return rows


class _StatusSession(_Session):
    def begin_transaction(self) -> _StatusTransaction:
        transaction = _StatusTransaction(self.graph)
        self.graph.transactions.append(transaction)
        return transaction


class _StatusGraph(_RevivingGraph):
    def session(self) -> _StatusSession:
        return _StatusSession(self)


def _revived_tombstone_state() -> Any:
    """A fold target tombstoned on both axes, as the real row is."""
    state = _state()
    target = state.nodes[_TARGET]
    target["name_stage"] = "superseded"
    target["superseded_from_stage"] = "accepted"
    target["status"] = "superseded"
    return state


def _fold_receipt(graph: _StatusGraph) -> dict[str, Any]:
    change = next(iter(graph.state.changes.values()))
    return json.loads(change["reason"])


def test_fold_into_a_superseded_tombstone_restores_draft_status() -> None:
    graph = _StatusGraph(_revived_tombstone_state())
    applied = _run(graph)
    assert applied["ok"] is True
    assert graph.commits == 1
    assert graph.state.nodes[_TARGET]["status"] == "draft"
    assert graph.state.nodes[_TARGET]["name_stage"] != "superseded"


def test_fold_receipt_records_the_restored_status_on_both_sides() -> None:
    graph = _StatusGraph(_revived_tombstone_state())
    applied = _run(graph)
    assert applied["ok"] is True
    receipt = _fold_receipt(graph)
    assert receipt["before"]["target_properties"]["status"] == "superseded"
    assert receipt["expected_after"]["names"]["target"]["status"] == "draft"
