"""A rename moves an identity; the sources that produced it must move with it.

The quantity does not change when its spelling does, so every
``StandardNameSource`` bound to the old name produces the new one. When that
carriage does not happen the failure hides in the one place nobody looks: the
export excludes an unsourced name at the population boundary, so a batch
closes at zero residue with the quantity absent from it and both spellings
holding nothing. Measured on 2026-09-08, a rename of
``power_due_to_ion_cyclotron_heating`` ended with two data-dictionary paths —
``ic_antennas/antenna/power_launched`` and
``summary/heating_current_drive/ic/power/value`` — producing no standard name
at all, and a re-cut of 224 candidates exported 206 with neither spelling in
it.

These tests pin the postcondition rather than the mechanism: count the
``PRODUCED_NAME`` producers the predecessor holds before the rename, count
what the successor holds after, and refuse a successor holding fewer. The
predecessor must claim none of them afterwards either — one source feeding two
live names is its own defect in the other direction.
"""

from __future__ import annotations

from typing import Any

import pytest

from iter_codex.standard_names.edit import apply_edit
from tests.standard_names.test_edit_engine import (
    FakeGraph,
    _admitting_pairing_guard,
    _patched_graph,
)

OLD = "power_due_to_ion_cyclotron_heating"
NEW = "net_power_due_to_ion_cyclotron_heating"
DD_SOURCES = (
    "dd:ic_antennas/antenna/power_launched",
    "dd:summary/heating_current_drive/ic/power/value",
)
REASON = (
    "the quantity is net boundary power, forward minus reflected at the "
    "antenna boundary"
)


class ProducerGraph(FakeGraph):
    """``FakeGraph`` plus the marked read of a name's producing cohort.

    ``drop_carriage`` emulates the defect under test: the migration reports a
    move that never reached the successor's bindings, which is exactly the
    observable state a rename that loses its sources leaves behind.
    """

    drop_carriage: bool = False

    def _producers_of(self, name: str) -> list[str]:
        return sorted(
            source_id
            for source_id, target in self.produced_name.items()
            if target == name
        )

    def query(self, cypher: str, **params: Any) -> list[dict[str, Any]]:
        if "// EDIT_FETCH_BOUND_SOURCE_IDS" in cypher:
            return [
                {"source_id": source_id}
                for source_id in self._producers_of(params["id"])
            ]
        rows = super().query(cypher, **params)
        if "// EDIT_FETCH_TARGET" in cypher and rows:
            rows[0]["producing_source_ids"] = self._producers_of(params["id"])
        return rows

    def _tx_run(self, cypher: str, **params: Any) -> list[dict[str, Any]]:
        rows = super()._tx_run(cypher, **params)
        if self.drop_carriage and "RETURN size(moved) AS moved" in cypher:
            for source_id in params.get("source_ids") or ():
                self.produced_name.pop(source_id, None)
                self.sources[source_id]["produced_sn_id"] = None
        return rows


def _sourced_graph(*, drop_carriage: bool = False) -> ProducerGraph:
    graph = ProducerGraph()
    graph.drop_carriage = drop_carriage
    graph.add_node(
        OLD,
        name_stage="accepted",
        docs_stage="accepted",
        unit="W",
        validation_status="valid",
        physics_domain="auxiliary_heating",
    )
    for source_id in DD_SOURCES:
        graph.add_source(source_id, sn_id=OLD, status="composed")
    return graph


def _rename(graph: ProducerGraph) -> Any:
    with _patched_graph(graph), _admitting_pairing_guard():
        return apply_edit(
            target=OLD,
            rename=NEW,
            reason=REASON,
            scope="only_self",
            gc=graph,
        )


def _producers(graph: ProducerGraph, name: str) -> list[str]:
    return sorted(
        source_id for source_id, target in graph.produced_name.items() if target == name
    )


class TestTheSuccessorHoldsWhatThePredecessorProduced:
    def test_the_producer_count_does_not_drop(self) -> None:
        graph = _sourced_graph()
        before = len(_producers(graph, OLD))
        assert before == len(DD_SOURCES)

        plan = _rename(graph)

        assert plan.blocked is None
        assert plan.applied is True
        after = len(_producers(graph, plan.successor))
        assert after >= before, (
            f"{plan.successor!r} holds {after} producer(s) where {OLD!r} held {before}"
        )
        assert _producers(graph, plan.successor) == sorted(DD_SOURCES)

    def test_the_predecessor_no_longer_claims_them(self) -> None:
        """One source bound to two live names is its own defect."""
        graph = _sourced_graph()
        plan = _rename(graph)

        assert plan.applied is True
        assert _producers(graph, OLD) == []
        assert graph.nodes[OLD]["source_paths"] == []

    def test_the_scalar_mirror_follows_the_edge(self) -> None:
        graph = _sourced_graph()
        plan = _rename(graph)

        for source_id in DD_SOURCES:
            assert graph.sources[source_id]["produced_sn_id"] == plan.successor

    def test_the_receipt_states_the_carriage(self) -> None:
        graph = _sourced_graph()
        plan = _rename(graph)

        assert any(
            f"{len(DD_SOURCES)} producing source(s)" in action
            for action in plan.actions
        ), plan.actions

    def test_a_dry_run_names_the_cohort_and_moves_nothing(self) -> None:
        graph = _sourced_graph()
        with _patched_graph(graph), _admitting_pairing_guard():
            plan = apply_edit(
                target=OLD,
                rename=NEW,
                reason=REASON,
                scope="only_self",
                dry_run=True,
                gc=graph,
            )

        assert plan.applied is False
        assert any(
            f"[dry-run] would carry {len(DD_SOURCES)} producing" in action
            for action in plan.actions
        ), plan.actions
        assert _producers(graph, OLD) == sorted(DD_SOURCES)


class TestALostBindingRefusesInsteadOfReportingSuccess:
    """The gate fails when the successor ends up holding fewer producers.

    Without the postcondition this state returns ``applied=True`` with a clean
    receipt, which is how a rename can retire a published quantity in silence.
    """

    def test_a_dropped_carriage_raises(self) -> None:
        graph = _sourced_graph(drop_carriage=True)
        with pytest.raises(RuntimeError, match="producing source"):
            _rename(graph)

    def test_the_refusal_names_the_stranded_sources(self) -> None:
        graph = _sourced_graph(drop_carriage=True)
        with pytest.raises(RuntimeError) as excinfo:
            _rename(graph)

        message = str(excinfo.value)
        assert NEW in message
        for source_id in DD_SOURCES:
            assert source_id in message
        assert "produce no standard name" in message


class TestAnUnsourcedRenameIsStillAdmitted:
    """A name nothing produced has nothing to carry, and renames as before.

    The postcondition asserts carriage, not the existence of sources: a
    derived or structural identity legitimately has no producing cohort, and
    the stranded-name repair route depends on that rename staying open.
    """

    def test_a_producerless_rename_applies(self) -> None:
        graph = ProducerGraph()
        graph.add_node(
            OLD,
            name_stage="accepted",
            docs_stage="accepted",
            unit="W",
            validation_status="valid",
        )

        plan = _rename(graph)

        assert plan.blocked is None
        assert plan.applied is True
        assert any("nothing to carry" in action for action in plan.actions)
