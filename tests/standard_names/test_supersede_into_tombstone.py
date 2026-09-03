"""Folding an identity onto a tombstoned spelling that nothing else reads.

A superseded name with no successor, no sources and no parent or child edge is
a free identity: the spelling is dead and the fold may re-occupy it. Anything
that still reads the spelling — a recorded successor, a successor lineage edge,
a bound source, a parent or a child — keeps the fold refused, and a target in a
live stage other than ``accepted`` stays refused as before.
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
_BACKING = "core_profiles/profiles_1d/electrons/density"


def _tombstone_state(**kwargs: Any) -> Any:
    state = _state(**kwargs)
    state.nodes[_TARGET]["name_stage"] = "superseded"
    state.nodes[_TARGET]["superseded_from_stage"] = "accepted"
    state.nodes[_TARGET]["superseded_by"] = None
    return state


def _has_parent(child: str, parent: str) -> dict[str, Any]:
    return {
        "element_id": f"rel:HAS_PARENT:{child}:{parent}",
        "type": "HAS_PARENT",
        "start_element_id": f"name:{child}",
        "end_element_id": f"name:{parent}",
        "start_id": child,
        "end_id": parent,
        "start_labels": ["StandardName"],
        "end_labels": ["StandardName"],
        "properties": {},
    }


def test_free_tombstone_target_is_occupied_by_the_fold() -> None:
    graph = _Graph(_tombstone_state())
    preview = _run(graph, dry_run=True)
    assert preview["ok"] is True
    assert graph.commits == 0
    assert graph.state.nodes[_OLD]["name_stage"] == "accepted"

    applied = _run(graph)
    assert applied["ok"] is True
    assert applied["already_superseded"] is False
    assert applied["old_prior_stage"] == "accepted"
    assert applied["sources_carried"] == 1
    assert graph.commits == 1

    source = graph.state.sources["dd:" + _BACKING]
    assert source["properties"]["produced_sn_id"] == _TARGET
    assert source["bindings"] == [_TARGET]
    assert graph.state.backings[_BACKING]["projections"] == [_TARGET]
    assert graph.state.nodes[_OLD]["name_stage"] == "superseded"
    assert graph.state.nodes[_OLD]["source_paths"] == []
    assert graph.state.nodes[_TARGET]["source_paths"] == ["dd:" + _BACKING]
    assert graph.state.refined_from == [(_TARGET, _OLD)]


def test_tombstone_target_holding_a_successor_is_refused() -> None:
    recorded = _Graph(_tombstone_state())
    recorded.state.nodes[_TARGET]["superseded_by"] = "electron_number_density"
    result = _run(recorded)
    assert result["ok"] is False
    assert "records successor 'electron_number_density'" in result["reason"]
    assert recorded.commits == 0
    assert recorded.state.nodes[_OLD]["name_stage"] == "accepted"

    lineage = _Graph(_tombstone_state())
    lineage.state.nodes["electron_number_density"] = _node(
        "electron_number_density", stage="accepted"
    )
    lineage.state.refined_from.append(("electron_number_density", _TARGET))
    result = _run(lineage)
    assert result["ok"] is False
    assert "successor lineage: electron_number_density" in result["reason"]
    assert lineage.commits == 0


def test_tombstone_target_whose_only_successor_is_the_folded_name_is_admitted() -> None:
    """A straight-line refinement chain (target -> old) closes back onto target.

    ``old`` already carries a REFINED_FROM edge onto ``target``, so ``target``'s
    only live descendant is the very name now being folded into it. That is
    the chain collapsing on itself, not a third-party identity being seized,
    so the fold is admitted despite the target being tombstoned.
    """
    graph = _Graph(_tombstone_state())
    graph.state.refined_from.append((_OLD, _TARGET))
    result = _run(graph)
    assert result["ok"] is True
    assert graph.commits == 1
    assert graph.state.nodes[_OLD]["name_stage"] == "superseded"


def test_tombstone_target_with_a_different_successor_stays_refused_even_when_old_is_also_one() -> (
    None
):
    graph = _Graph(_tombstone_state())
    graph.state.nodes["electron_number_density"] = _node(
        "electron_number_density", stage="accepted"
    )
    graph.state.refined_from.append((_OLD, _TARGET))
    graph.state.refined_from.append(("electron_number_density", _TARGET))
    result = _run(graph)
    assert result["ok"] is False
    assert "successor lineage: electron_number_density" in result["reason"]
    assert graph.commits == 0


def test_live_target_that_is_not_accepted_is_still_refused() -> None:
    for stage in ("pending", "drafted", "reviewed", "approved", "refining"):
        graph = _Graph(_state())
        graph.state.nodes[_TARGET]["name_stage"] = stage
        result = _run(graph)
        assert result["ok"] is False
        assert f"name_stage={stage!r}, not 'accepted'" in result["reason"]
        assert graph.commits == 0


def test_tombstone_target_that_still_carries_a_source_is_refused() -> None:
    graph = _Graph(_tombstone_state())
    graph.state.sources["dd:equilibrium/electron_density"] = {
        "properties": {
            "id": "dd:equilibrium/electron_density",
            "source_type": "dd",
            "source_id": "equilibrium/electron_density",
            "status": "composed",
            "produced_sn_id": _TARGET,
            "claim_token": None,
            "claimed_at": None,
        },
        "bindings": [_TARGET],
        "backings": [],
    }
    result = _run(graph)
    assert result["ok"] is False
    assert "still carries 1 source(s)" in result["reason"]
    assert graph.commits == 0


def test_tombstone_target_with_a_parent_or_a_child_is_refused() -> None:
    parented = _Graph(_tombstone_state())
    parented.state.other_relationships.append(_has_parent(_TARGET, "density"))
    result = _run(parented)
    assert result["ok"] is False
    assert "still has parent 'density'" in result["reason"]
    assert parented.commits == 0

    childed = _Graph(_tombstone_state())
    childed.state.other_relationships.append(
        _has_parent("electron_density_at_boundary", _TARGET)
    )
    result = _run(childed)
    assert result["ok"] is False
    assert "still has child 'electron_density_at_boundary'" in result["reason"]
    assert childed.commits == 0
