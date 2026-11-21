"""A fold's dry run reaches the same verdict as its live transaction.

``supersede_into`` answers twice. The dry run evaluates the guard and the
idempotency read, then rolls back and reports a mutation plan. The live path
evaluates the same guard, applies the declared mutations, and proves the
complete post-state equals a receipt computed before any write. A preflight is
only worth running if those two answers agree, so every scalar the mutation
writes must be a scalar the receipt can name.

The modification stamp is the one that cannot be guessed. The name mutation
takes it from ``$changed_at`` -- the single instant the receipt already
declares for the fold's ledger event -- rather than from the transaction clock,
so the expected post-state carries the same value and the proof pins the stamp
exactly instead of comparing an unpredictable one. The stamp reaches five
places in the compared state: each participant's own properties, plus the three
embedded copies of the target's properties carried by its scalar target, its
source binding, and its backing projection.

The harness mirrors the production write here rather than assuming it: the
negative control stamps from an instant the receipt never declared and the
proof must still refuse, because a postflight that passes by ignoring a field
reports a protection it does not provide.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any
from unittest.mock import patch

import pytest
from neo4j.time import DateTime

from iter_codex.standard_names import edit
from tests.standard_names.test_tombstone_supersede import (
    _Graph,
    _run,
    _state,
    _Transaction,
)

_OLD = "invalid_duplicate"
_TARGET = "electron_density"

#: Distinct pre-fold modification times, one per participant, so a stamp that
#: is merely copied from the snapshot cannot satisfy the comparison.
_OLD_STAMP = DateTime.from_iso_format("2026-09-07T05:48:33.280000000+00:00")
_TARGET_STAMP = DateTime.from_iso_format("2026-09-05T15:21:04.031000000+00:00")

#: An instant the receipt never declares, standing in for the transaction clock.
_UNDECLARED_STAMP = DateTime.from_iso_format("2026-09-08T07:45:11.137000000+00:00")

#: Every place the compared state carries a participant's modification time.
_STAMP_PATHS = (
    ("names", "old"),
    ("names", "target"),
)


def _stamped_state() -> Any:
    state = _state()
    state.nodes[_OLD]["updated_at"] = _OLD_STAMP
    state.nodes[_TARGET]["updated_at"] = _TARGET_STAMP
    return state


@contextmanager
def _mutation_stamp(stamp: DateTime | None = None) -> Iterator[list[str]]:
    """Mirror the production name mutation's modification stamp in the harness.

    ``stamp`` of ``None`` takes the declared instant from the query parameters,
    which is what the production query's ``datetime($changed_at)`` does. Any
    other value stands in for a clock the receipt cannot name.
    """
    original = _Transaction.run
    declared: list[str] = []

    def run(self: Any, cypher: str, **params: Any) -> list[dict[str, Any]]:
        rows = original(self, cypher, **params)
        if "ATOMIC_FOLD_MUTATE_NAMES" in cypher and rows:
            declared.append(params["changed_at"])
            written = stamp or DateTime.from_iso_format(params["changed_at"])
            for key in ("old_id", "into_id"):
                self.state.nodes[params[key]]["updated_at"] = written
        return rows

    with patch.object(_Transaction, "run", run):
        yield declared


def test_name_mutation_stamps_the_instant_the_receipt_declares() -> None:
    """The mutation reads its clock from the parameter, not from the server."""
    query = edit._FOLD_NAME_MUTATION_QUERY
    assert "old.updated_at = datetime($changed_at)" in query
    assert "target.updated_at = datetime($changed_at)" in query
    assert "updated_at = datetime()" not in query


def test_dry_run_and_live_transaction_return_the_same_verdict() -> None:
    """Both answers are ok for one identity, and the live one commits."""
    dry = _run(_Graph(_stamped_state()), dry_run=True)
    graph = _Graph(_stamped_state())
    with _mutation_stamp() as declared:
        live = _run(graph)
    assert dry["ok"] is True
    assert dry["dry_run"] is True
    assert live["ok"] is True
    assert graph.commits == 1
    assert graph.rollbacks == 0
    assert len(declared) == 1
    assert dry["sources_carried"] == live["sources_carried"]
    assert dry["projections_carried"] == live["projections_carried"]
    assert dry["old_prior_stage"] == live["old_prior_stage"]


def test_committed_stamp_is_the_ledger_event_instant() -> None:
    """Both participants land on the instant the fold's ledger row records."""
    graph = _Graph(_stamped_state())
    with _mutation_stamp() as declared:
        result = _run(graph)
    changed_at = declared[0]
    event = graph.state.changes[result["change_id"]]
    assert event["changed_at"] == changed_at
    for name in (_OLD, _TARGET):
        assert graph.state.nodes[name]["updated_at"] == DateTime.from_iso_format(
            changed_at
        )


def test_receipt_names_the_stamp_on_every_copy_of_a_participant() -> None:
    """The expectation carries the stamp wherever the compared state does."""
    graph = _Graph(_stamped_state())
    with _mutation_stamp() as declared:
        result = _run(graph)
    expected_after = json.loads(graph.state.changes[result["change_id"]]["reason"])[
        "expected_after"
    ]
    stamp = DateTime.from_iso_format(declared[0]).iso_format()
    for section, participant in _STAMP_PATHS:
        assert expected_after[section][participant]["updated_at"] == stamp
    source = expected_after["sources"][0]
    assert source["scalar_target"]["properties"]["updated_at"] == stamp
    binding = next(item for item in source["bindings"] if item["target_id"] == _TARGET)
    assert binding["target_properties"]["updated_at"] == stamp
    projection = next(
        item
        for item in expected_after["backings"][0]["projections"]
        if item["target_id"] == _TARGET
    )
    assert projection["target_properties"]["updated_at"] == stamp


def test_postflight_refuses_a_stamp_the_receipt_never_declared() -> None:
    """The proof still bites: an undeclared clock rolls the fold back."""
    graph = _Graph(_stamped_state())
    with (
        _mutation_stamp(_UNDECLARED_STAMP),
        pytest.raises(RuntimeError, match="postflight exact-state proof did not hold"),
    ):
        _run(graph)
    assert graph.commits == 0
    assert graph.state.nodes[_OLD]["updated_at"] == _OLD_STAMP
    assert graph.state.nodes[_TARGET]["updated_at"] == _TARGET_STAMP
    assert graph.state.nodes[_OLD]["name_stage"] == "accepted"
