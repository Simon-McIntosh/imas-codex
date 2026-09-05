"""The tombstoned fold target's successor is resolved by walking lineage.

A superseded spelling that something inherited is not a free identity, and the
relation that says so is carried by the ``REFINED_FROM`` edges: on the live
graph they answer for 1588 of 2102 superseded names, while the
``superseded_by`` scalar summarising the same fact is written on 18. So the
refusal walks the edges to the live tip of the chain and names that tip as the
identity to fold into, and consults the scalar only afterwards — where a
recorded successor no lineage carries is the graph disagreeing with itself,
which a guard settles by refusing rather than by admitting.
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
_STALE_SCALAR = "some_other_identity"


def _tombstoned(**kwargs: Any) -> Any:
    state = _state(**kwargs)
    state.nodes[_TARGET]["name_stage"] = "superseded"
    state.nodes[_TARGET]["superseded_from_stage"] = "accepted"
    state.nodes[_TARGET]["superseded_by"] = None
    return state


def test_lineage_tip_outranks_a_disagreeing_recorded_successor() -> None:
    """The walk decides who inherited the spelling; the scalar does not.

    The target carries both a live ``REFINED_FROM`` successor and a scalar
    naming a completely different name. Reading the scalar first answers with
    a name nothing in the lineage supports, so the refusal must name the tip
    the edges resolve to and must not name the scalar's.
    """
    graph = _Graph(_tombstoned())
    graph.state.nodes[_TARGET]["superseded_by"] = _STALE_SCALAR
    graph.state.nodes[_TIP] = _node(_TIP, stage="accepted")
    graph.state.refined_from.append((_TIP, _TARGET))

    result = _run(graph)

    assert result["ok"] is False
    assert f"successor lineage: {_TIP}" in result["reason"]
    assert "fold into the successor instead" in result["reason"]
    assert _STALE_SCALAR not in result["reason"]
    assert graph.commits == 0


def test_lineage_walk_names_the_tip_beyond_a_tombstoned_intermediate() -> None:
    """A chain through a dead intermediate still resolves the live inheritor.

    Nothing records a successor here at all, so a scalar reader sees a free
    identity. The edges carry target -> intermediate -> tip with only the tip
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


def test_recorded_successor_no_lineage_carries_refuses_as_a_disagreement() -> None:
    """A scalar the edges do not support refuses, and says which it is.

    This is the population the walk cannot reach — measured at 4 of the 18
    names the scalar answers for, 3 of which have no successor edge at all.
    The guard keeps refusing them, but the refusal states that the lineage is
    missing rather than reporting a successor the graph can be walked to.
    """
    graph = _Graph(_tombstoned())
    graph.state.nodes[_TARGET]["superseded_by"] = _STALE_SCALAR

    result = _run(graph)

    assert result["ok"] is False
    assert f"records successor {_STALE_SCALAR!r}" in result["reason"]
    assert "no successor lineage carries" in result["reason"]
    assert graph.commits == 0
