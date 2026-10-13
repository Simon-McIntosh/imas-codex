"""The tombstoned fold target's successor is resolved by walking lineage.

A superseded spelling that something inherited is not a free identity, and the
relation that says so is carried by the ``REFINED_FROM`` edges: on the live
graph they answer for most superseded names. The refusal walks the edges to
the live tip of the chain and names that tip as the identity to fold into.
"""

from __future__ import annotations

from typing import Any

from tests.standard_names.test_tombstone_supersede import (
    _Graph,
    _node,
    _run,
    _state,
)

_OLD = "invalid_duplicate"
_TARGET = "electron_density"
_TIP = "electron_number_density"
_INTERMEDIATE = "electron_density_alias"


def _tombstoned(**kwargs: Any) -> Any:
    state = _state(**kwargs)
    state.nodes[_TARGET]["name_stage"] = "superseded"
    state.nodes[_TARGET]["superseded_from_stage"] = "accepted"
    return state


def test_lineage_tip_refuses_the_fold() -> None:
    """The walk names the live identity that inherited the spelling."""
    graph = _Graph(_tombstoned())
    graph.state.nodes[_TIP] = _node(_TIP, stage="accepted")
    graph.state.refined_from.append((_TIP, _TARGET))

    result = _run(graph)

    assert result["ok"] is False
    assert f"successor lineage: {_TIP}" in result["reason"]
    assert "fold into the successor instead" in result["reason"]
    assert graph.commits == 0


def test_lineage_walk_names_the_tip_beyond_a_tombstoned_intermediate() -> None:
    """A chain through a dead intermediate still resolves the live inheritor.

    The edges carry target -> intermediate -> tip with only the tip
    still live, so the refusal names the tip and tells the caller to fold
    there.
    """
    graph = _Graph(_tombstoned())
    graph.state.nodes[_INTERMEDIATE] = _node(_INTERMEDIATE, stage="superseded")
    graph.state.nodes[_TIP] = _node(_TIP, stage="accepted")
    graph.state.refined_from.append((_INTERMEDIATE, _TARGET))
    graph.state.refined_from.append((_TIP, _INTERMEDIATE))

    result = _run(graph)

    assert result["ok"] is False
    assert f"successor lineage: {_TIP}" in result["reason"]
    assert "fold into the successor instead" in result["reason"]
    assert graph.commits == 0
    assert graph.state.nodes[_OLD]["name_stage"] == "accepted"
