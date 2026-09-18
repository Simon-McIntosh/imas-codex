"""A source already bound to a live identity is an adjudicable conflict.

The migration's compare-and-set distinguishes two refusals. A source that
already binds a live identity other than the expected predecessor is a state
an operator must settle, and repeating the migration cannot change it, so it
is reported as a named conflict carrying the holding identity and its stage.
Any other ineligible state — a concurrent claim, a stale row, a lost binding —
is a transient fault and keeps raising the pre-existing fault.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from imas_codex.standard_names.provenance_lifecycle import (
    SourceBindingConflictError,
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
