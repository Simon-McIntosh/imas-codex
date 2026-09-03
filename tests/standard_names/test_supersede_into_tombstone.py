"""Folding an identity onto a tombstoned spelling that nothing else reads.

A superseded name with no successor, no sources and no parent or child edge is
a free identity: the spelling is dead and the fold may re-occupy it. Anything
that still reads the spelling — a recorded successor, a successor lineage edge,
a bound source, a parent or a child — keeps the fold refused, and a target in a
live stage other than ``accepted`` stays refused as before.
"""

from __future__ import annotations

from typing import Any

import pytest

from tests.standard_names.test_tombstone_supersede import (
    _Graph,
    _node,
    _run,
    _Session,
    _state,
    _Transaction,
)

_OLD = "invalid_duplicate"
_TARGET = "electron_density"
_BACKING = "core_profiles/profiles_1d/electrons/density"


def _tombstone_state(
    *, target_predecessor_stage: str = "accepted", **kwargs: Any
) -> Any:
    state = _state(**kwargs)
    state.nodes[_TARGET]["name_stage"] = "superseded"
    state.nodes[_TARGET]["superseded_from_stage"] = target_predecessor_stage
    state.nodes[_TARGET]["superseded_by"] = None
    return state


class _RevivingTransaction(_Transaction):
    """Mirrors the production ``ATOMIC_FOLD_MUTATE_NAMES`` write exactly,
    including the ``target_revived_stage`` parameter the shared stateful
    mock in :mod:`test_tombstone_supersede` predates and does not apply."""

    def run(self, cypher: str, **params: Any) -> list[dict[str, Any]]:
        if "ATOMIC_FOLD_MUTATE_NAMES" in cypher:
            self.write_markers.append("names")
            old = self.state.nodes[params["old_id"]]
            target = self.state.nodes[params["into_id"]]
            if (
                self._element_id("name", params["old_id"]) != params["old_element_id"]
                or self._element_id("name", params["into_id"])
                != params["target_element_id"]
            ):
                return []
            old["superseded_from_stage"] = params["predecessor_stage"]
            old["name_stage"] = "superseded"
            old.pop("claim_token", None)
            old.pop("claimed_at", None)
            old["source_paths"] = []
            if old.get("edit_status") == "open":
                old["edit_status"] = "applied"
            target["source_paths"] = list(params["target_paths"])
            if params.get("target_revived_stage") is not None:
                target["name_stage"] = params["target_revived_stage"]
            lineage = (params["into_id"], params["old_id"])
            if lineage not in self.state.refined_from:
                self.state.refined_from.append(lineage)
            return [
                {
                    "old_stage": "superseded",
                    "predecessor_stage": params["predecessor_stage"],
                }
            ]
        return super().run(cypher, **params)


class _RevivingSession(_Session):
    def begin_transaction(self) -> _RevivingTransaction:
        transaction = _RevivingTransaction(self.graph)
        self.graph.transactions.append(transaction)
        return transaction


class _RevivingGraph(_Graph):
    """A fold graph whose mutation mock matches the revival-aware query."""

    def session(self) -> _RevivingSession:
        return _RevivingSession(self)


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
    graph = _RevivingGraph(_tombstone_state())
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
    graph = _RevivingGraph(_tombstone_state())
    graph.state.refined_from.append((_OLD, _TARGET))
    result = _run(graph)
    assert result["ok"] is True
    assert graph.commits == 1
    assert graph.state.nodes[_OLD]["name_stage"] == "superseded"


def test_tombstone_target_whose_two_hop_chain_closes_onto_its_root_is_admitted() -> (
    None
):
    """target -> intermediate -> old, intermediate itself tombstoned, closes.

    ``target``'s only direct successor is ``electron_number_density``, which
    is itself superseded and whose only successor is ``old``. Every live
    descendant of ``target``, at the end of the chain, is the name now being
    folded into it, so the fold is admitted despite the walk needing two hops
    rather than one.
    """
    graph = _RevivingGraph(_tombstone_state())
    graph.state.nodes["electron_number_density"] = _node(
        "electron_number_density", stage="superseded"
    )
    graph.state.refined_from.append(("electron_number_density", _TARGET))
    graph.state.refined_from.append((_OLD, "electron_number_density"))
    result = _run(graph)
    assert result["ok"] is True
    assert graph.commits == 1
    assert graph.state.nodes[_OLD]["name_stage"] == "superseded"


def test_tombstone_target_whose_two_hop_chain_branches_stays_refused() -> None:
    """A live descendant off the chain keeps the spelling load-bearing.

    ``electron_number_density`` sits between ``target`` and ``old`` and is
    itself superseded, but it also carries a second, still-live successor
    (``ion_density_alias``) of its own. That branch is a live descendant of
    ``target`` that is not the name being folded, so the fold refuses even
    though the direct path to ``old`` is otherwise a clean chain.
    """
    graph = _Graph(_tombstone_state())
    graph.state.nodes["electron_number_density"] = _node(
        "electron_number_density", stage="superseded"
    )
    graph.state.nodes["ion_density_alias"] = _node(
        "ion_density_alias", stage="accepted"
    )
    graph.state.refined_from.append(("electron_number_density", _TARGET))
    graph.state.refined_from.append((_OLD, "electron_number_density"))
    graph.state.refined_from.append(("ion_density_alias", "electron_number_density"))
    result = _run(graph)
    assert result["ok"] is False
    assert "successor lineage: ion_density_alias" in result["reason"]
    assert graph.commits == 0


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


@pytest.mark.parametrize(
    "predecessor_stage,revived_stage",
    [
        ("accepted", "reviewed"),
        ("reviewed", "reviewed"),
        ("drafted", "drafted"),
        ("exhausted", "exhausted"),
    ],
)
def test_fold_revives_a_tombstoned_target_at_its_predecessor_stage(
    predecessor_stage: str, revived_stage: str
) -> None:
    """The target's own pre-tombstone stage is restored, capped short of
    acceptance: revival re-enters the review pipeline, it never grants
    acceptance for free. It never stays 'superseded' either — a fold that
    carries a source onto a still-tombstoned target would leave the
    data-dictionary path with no live standard name at all."""
    graph = _RevivingGraph(_tombstone_state(target_predecessor_stage=predecessor_stage))
    applied = _run(graph)
    assert applied["ok"] is True
    assert graph.commits == 1
    assert graph.state.nodes[_TARGET]["name_stage"] == revived_stage
    assert graph.state.nodes[_TARGET]["name_stage"] != "superseded"
    assert graph.state.nodes[_TARGET]["name_stage"] != "accepted"


def test_fold_revival_defaults_to_drafted_without_a_recorded_predecessor_stage() -> (
    None
):
    state = _tombstone_state()
    del state.nodes[_TARGET]["superseded_from_stage"]
    graph = _RevivingGraph(state)
    applied = _run(graph)
    assert applied["ok"] is True
    assert graph.state.nodes[_TARGET]["name_stage"] == "drafted"


def test_fold_onto_a_tombstone_leaves_every_carried_source_on_a_live_name() -> None:
    """Every source (and its backing projection) the fold carries onto the
    revived target resolves to a name that is no longer tombstoned — the
    provenance the fold exists to preserve is reachable again, not stranded
    on a dead spelling."""
    graph = _RevivingGraph(_tombstone_state())
    applied = _run(graph)
    assert applied["ok"] is True
    assert applied["sources_carried"] == 1

    revived_stage = graph.state.nodes[_TARGET]["name_stage"]
    assert revived_stage not in ("superseded", "accepted")

    source = graph.state.sources["dd:" + _BACKING]
    assert source["properties"]["produced_sn_id"] == _TARGET
    assert source["bindings"] == [_TARGET]
    assert graph.state.backings[_BACKING]["projections"] == [_TARGET]
