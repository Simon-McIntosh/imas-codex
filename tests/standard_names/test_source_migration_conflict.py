"""A source already bound to a live identity is an adjudicable conflict.

The migration's compare-and-set distinguishes its refusals by what a retry can
change. A source that already binds a live identity other than the expected
predecessor is a state an operator must settle, so it is reported as a named
conflict carrying the holding identity and its stage. A source bound to the
live predecessor and a successor that has itself been retired is one live claim
beside a settled one: neither a retry nor the dual-live branch applies, so it is
reported under its own named outcome naming the live holder and the retired
holder with its stage. Any other ineligible state — a concurrent claim, a stale
row, a lost binding — is a transient fault and keeps raising the pre-existing
fault.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from imas_codex.standard_names.provenance_lifecycle import (
    SourceBindingConflictError,
    SourceRetiredSuccessorConflictError,
    retarget_standard_name_sources,
)

_SOURCE_ID = "dd:example/predicted_path"


def _binding(name: str, stage: str) -> dict[str, object]:
    return {"id": name, "name_stage": stage}


def _migration_row(
    bindings: list[dict[str, object]],
    *,
    source_id: str = _SOURCE_ID,
    status: str = "attached",
    scalar: str | None = "old_name",
    claimed: bool = False,
) -> dict[str, object]:
    return {
        "source_id": source_id,
        "source_exists": True,
        "source_status": status,
        "scalar_binding": scalar,
        "actively_claimed": claimed,
        "current_bindings": [entry["id"] for entry in bindings],
        "binding_state": bindings,
        "manifest_recorded": False,
    }


def _retarget(graph: object, row: dict[str, object]) -> object:
    return retarget_standard_name_sources(
        graph,
        "old_name",
        "replacement_name",
        source_ids=[row["source_id"]],
        expected_current_bindings={row["source_id"]: "old_name"},
        record_change=False,
        enforce_consistency=False,
    )


class _BudgetTrackingGraph:
    """A source holding a retry budget, recording every write it receives."""

    def __init__(self, row: dict[str, object], *, attempt_count: int) -> None:
        self._row = row
        self.attempt_count = attempt_count
        self.writes: list[str] = []

    def query(self, cypher: str, **params: object) -> list[dict[str, object]]:
        if "manifest_recorded" in cypher:
            return [self._row]
        self.writes.append(cypher)
        if "attempt_count" in cypher:
            raise AssertionError("the migration wrote the source retry budget")
        return []

    def charge_attempt(self) -> None:
        """The charge the claim path applies, so an unmoved reading is measured
        against an instrument that provably moves."""
        self.attempt_count += 1


def test_live_binding_conflict_names_the_holding_identity_and_stage() -> None:
    row = _migration_row(
        [_binding("old_name", "drafted"), _binding("live_holder", "reviewed")]
    )
    gc = MagicMock()
    gc.query.return_value = [row]

    with pytest.raises(SourceBindingConflictError) as raised:
        _retarget(gc, row)

    assert [
        (item.holding_identity, item.holding_stage) for item in raised.value.conflicts
    ] == [("live_holder", "reviewed")]
    assert "live_holder" in str(raised.value)
    assert "reviewed" in str(raised.value)
    # Existing callers catch RuntimeError; the named type narrows the signal
    # without taking that away.
    assert isinstance(raised.value, RuntimeError)
    # The conflict is decided before any mutation is attempted.
    assert gc.query.call_count == 1


def test_conflict_branch_leaves_the_retry_budget_unmoved() -> None:
    row = _migration_row(
        [_binding("live_holder", "reviewed"), _binding("old_name", "drafted")]
    )
    graph = _BudgetTrackingGraph(row, attempt_count=3)

    before = graph.attempt_count
    with pytest.raises(SourceBindingConflictError):
        _retarget(graph, row)
    after = graph.attempt_count

    assert (before, after) == (3, 3)
    assert graph.writes == []


def test_the_budget_instrument_moves_when_an_attempt_is_charged() -> None:
    row = _migration_row([_binding("old_name", "drafted")])
    graph = _BudgetTrackingGraph(row, attempt_count=5)

    graph.charge_attempt()

    assert graph.attempt_count == 6


@pytest.mark.parametrize(
    "row",
    [
        _migration_row([_binding("old_name", "drafted")], claimed=True),
        _migration_row([_binding("old_name", "drafted")], status="stale"),
        _migration_row([]),
        _migration_row([_binding("old_name", "drafted")], scalar="other_name"),
    ],
    ids=["concurrent-claim", "stale-source", "no-binding", "scalar-mismatch"],
)
def test_transient_fault_still_raises_an_anonymous_fault(
    row: dict[str, object],
) -> None:
    gc = MagicMock()
    gc.query.return_value = [row]

    with pytest.raises(RuntimeError) as raised:
        _retarget(gc, row)

    assert not isinstance(raised.value, SourceBindingConflictError)
    assert gc.query.call_count == 1


def test_dual_bound_conflict_names_the_successor_as_the_holder() -> None:
    """The conflict names the identity that holds the binding, not the one
    being retired.

    A source bound to the live predecessor and its live successor is held by
    the successor: the migration is retiring the predecessor, so naming it as
    the holder would point the operator at the identity being removed.
    """
    row = _migration_row(
        [_binding("old_name", "drafted"), _binding("replacement_name", "drafted")]
    )
    gc = MagicMock()
    gc.query.return_value = [row]

    with pytest.raises(SourceBindingConflictError) as raised:
        _retarget(gc, row)

    assert [item.holding_identity for item in raised.value.conflicts] == [
        "replacement_name"
    ]


def test_retired_successor_is_not_a_dual_bound_conflict() -> None:
    """A retired successor is a named outcome, not a second live claim.

    A source bound to a live predecessor and a retired successor holds one live
    identity, so it is not the dual-live conflict. Repeating the migration
    cannot settle it either, so it reaches the caller as its own named outcome
    naming the live holder and the retired holder with its stage — never the
    anonymous transient fault that a retry loop would spend a budget on.
    """
    row = _migration_row(
        [
            _binding("old_name", "drafted"),
            _binding("replacement_name", "superseded"),
        ]
    )
    gc = MagicMock()
    gc.query.return_value = [row]

    with pytest.raises(SourceRetiredSuccessorConflictError) as raised:
        _retarget(gc, row)

    (item,) = raised.value.conflicts
    assert item.source_id == _SOURCE_ID
    assert (item.holding_identity, item.holding_stage) == ("old_name", "drafted")
    assert (item.retired_identity, item.retired_stage) == (
        "replacement_name",
        "superseded",
    )
    # One live claim beside a settled one, so the two named identities differ
    # and the row does not take the dual-live conflict branch.
    assert item.holding_identity != item.retired_identity
    assert not isinstance(raised.value, SourceBindingConflictError)
    assert gc.query.call_count == 1


def test_single_live_binding_is_not_a_dual_bound_conflict() -> None:
    """A source carrying one live identity is not a dual-bound conflict.

    The dual-bound classification fires only when a source holds more than one
    live identity. A source whose single live binding is already the successor,
    with the manifest not yet recorded, is an in-flight state rather than a
    decision for an operator, so it keeps raising the anonymous fault.
    """
    row = _migration_row(
        [_binding("replacement_name", "drafted")],
        scalar="replacement_name",
    )
    gc = MagicMock()
    gc.query.return_value = [row]

    with pytest.raises(RuntimeError) as raised:
        _retarget(gc, row)

    assert not isinstance(raised.value, SourceBindingConflictError)
