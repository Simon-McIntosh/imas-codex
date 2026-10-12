"""A fold onto an occupied identity keeps whichever side carries the richer documentation.

Folding one spelling into an identity that already exists merges two prose
bodies under one surviving spelling. The fold must keep the side a reader
would have preferred had the two never been split: the richer documentation,
and never the source's sparser text merely because it happened to arrive as
the fold's source. The rule is stated in ``_fold_carries_source_documentation``
— a documented side beats an undocumented or placeholder side; when both hold
prose the accepted docs stage beats a lower one; the incumbent target wins
every tie. Prose is carried whole, never averaged.

The integration tests drive the real ``supersede_into`` transaction through the
stateful graph harness. The harness mirrors the production name mutation by
hand, so the stateful mirror here also applies the documentation and docs_stage
the mutation query writes; the mirror must reproduce the write the receipt's
expected post-state names or the exact postflight proof refuses.
"""

from __future__ import annotations

from typing import Any

from imas_codex.standard_names import edit
from tests.standard_names.test_tombstone_supersede import (
    _Graph as _BaseGraph,
    _run,
    _Session as _BaseSession,
    _state,
    _Transaction as _BaseTransaction,
)

_OLD = "invalid_duplicate"
_TARGET = "electron_density"

_RICH_TARGET = (
    "Energy deposited in the plasma-facing component by the coolant loop, "
    "excluding the contribution from the structural support."
)
_RICH_SOURCE = (
    "Total energy absorbed by the plasma-facing component across the shot, "
    "integrated over the coolant enthalpy rise."
)


class _DocsTransaction(_BaseTransaction):
    """Mirror of the fold name mutation that also writes the documentation pair.

    The base harness mirrors ``source_paths``/``name_stage``/``updated_at``; the
    fold additionally writes ``documentation`` and ``docs_stage`` (a None value
    removes the slot, matching ``SET x = null``), so the mirror applies the two
    parameters the mutation query now carries before delegating.
    """

    def run(self, cypher: str, **params: Any) -> list[dict[str, Any]]:
        if "ATOMIC_FOLD_MUTATE_NAMES" in cypher:
            target = self.state.nodes[params["into_id"]]
            for field, param in (
                ("documentation", "target_documentation"),
                ("docs_stage", "target_docs_stage"),
            ):
                value = params.get(param)
                if value is None:
                    target.pop(field, None)
                else:
                    target[field] = value
        return super().run(cypher, **params)


class _DocsSession(_BaseSession):
    def begin_transaction(self) -> _DocsTransaction:
        transaction = _DocsTransaction(self.graph)
        self.graph.transactions.append(transaction)
        return transaction


class _DocsGraph(_BaseGraph):
    def session(self) -> _DocsSession:
        return _DocsSession(self)


def _documented_state(
    *,
    source_documentation: str,
    source_stage: str | None,
    target_documentation: str,
    target_stage: str | None,
) -> Any:
    state = _state()
    source = state.nodes[_OLD]
    source["documentation"] = source_documentation
    source["docs_stage"] = source_stage
    target = state.nodes[_TARGET]
    target["documentation"] = target_documentation
    target["docs_stage"] = target_stage
    return state


def test_sparse_source_onto_rich_target_keeps_target_documentation() -> None:
    """A fold of a sparse source onto a richly documented target keeps the target's."""
    state = _documented_state(
        source_documentation="",
        source_stage="exhausted",
        target_documentation=_RICH_TARGET,
        target_stage="accepted",
    )
    graph = _DocsGraph(state)
    result = _run(graph)
    assert result["ok"] is True
    assert graph.state.nodes[_TARGET]["documentation"] == _RICH_TARGET
    assert graph.state.nodes[_TARGET]["docs_stage"] == "accepted"


def test_rich_source_onto_sparse_target_carries_source_documentation() -> None:
    """A fold of a rich source onto a sparse target carries the source's documentation."""
    state = _documented_state(
        source_documentation=_RICH_SOURCE,
        source_stage="accepted",
        target_documentation="",
        target_stage="pending",
    )
    graph = _DocsGraph(state)
    result = _run(graph)
    assert result["ok"] is True
    assert graph.state.nodes[_TARGET]["documentation"] == _RICH_SOURCE
    assert graph.state.nodes[_TARGET]["docs_stage"] == "accepted"


def test_documented_side_beats_a_placeholder() -> None:
    """Whitespace-only prose is a placeholder, so a documented side wins over it."""
    assert edit._fold_holds_documentation({"documentation": "   \n  "}) is False
    assert (
        edit._fold_carries_source_documentation(
            {"documentation": "   ", "docs_stage": "accepted"},
            {"documentation": _RICH_SOURCE, "docs_stage": "reviewed"},
        )
        is True
    )


def test_accepted_stage_beats_a_lower_one_when_both_document() -> None:
    """With both sides documented, the accepted docs stage beats a lower one."""
    carry = edit._fold_carries_source_documentation(
        {"documentation": _RICH_TARGET, "docs_stage": "reviewed"},
        {"documentation": _RICH_SOURCE, "docs_stage": "accepted"},
    )
    assert carry is True


def test_incumbent_target_wins_a_stage_tie() -> None:
    """A tie keeps the target's documentation; the source's prose is not silently preferred."""
    carry = edit._fold_carries_source_documentation(
        {"documentation": _RICH_TARGET, "docs_stage": "accepted"},
        {"documentation": _RICH_SOURCE, "docs_stage": "accepted"},
    )
    assert carry is False
