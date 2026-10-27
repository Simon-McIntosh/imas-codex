"""Admission of a structural parent that no source produces.

A parent whose producing source moved to another identity keeps its children
and loses its ``PRODUCED_NAME`` edge. Its name can never earn a review — there
is no source to review it against — so it strands on the name axis unless the
structural-accept path admits it from the topology instead of from the
``origin`` and ``source_path`` scalars, which by then describe a source it no
longer has.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import imas_codex.standard_names.graph_ops as graph_ops
from imas_codex.standard_names.graph_store import (
    graph_parent_lifecycle as _owning_graph_parent_lifecycle,
)

_SELECT_MARKER = "live_child_count"
_SNAPSHOT_MARKER = "parent_element_id"
_PERSIST_MARKER = "CREATE (parent)-[:HAS_STRUCTURAL_AUTHORITY]->(authority)"


def _candidate(
    parent_id: str,
    *,
    origin: str | None,
    producer_count: int,
    live_child_count: int,
) -> dict[str, Any]:
    return {
        "id": parent_id,
        "origin": origin,
        "producer_count": producer_count,
        "live_child_count": live_child_count,
    }


def _snapshot(parent_id: str, *child_ids: str, origin: str | None) -> dict[str, Any]:
    return {
        "parent_id": parent_id,
        "parent_element_id": f"element-{parent_id}",
        "name_stage": "reviewed",
        "origin": origin,
        "claim_token": None,
        "reviewed_name_at": None,
        "docs_stage": "pending",
        "validation_status": "valid",
        "embedded_at": None,
        "chain_length": 0,
        "authority_ids": [],
        "children": [
            {
                "id": child_id,
                "element_id": f"child-element-{index}",
                "name_stage": "accepted",
                "reviewer_score_name": 0.9,
                "reviewer_model_name": "reviewer",
            }
            for index, child_id in enumerate(child_ids)
        ],
    }


class _RecordingGraph:
    """Answers the three statements the structural accept issues, in order."""

    def __init__(
        self,
        candidates: list[dict[str, Any]],
        snapshots: dict[str, dict[str, Any]],
    ) -> None:
        self._candidates = candidates
        self._snapshots = snapshots
        self.snapshot_reads: list[str] = []
        self.persists: list[dict[str, Any]] = []

    def query(self, cypher: str, **params: Any) -> list[dict[str, Any]]:
        if _SELECT_MARKER in cypher:
            self.selection_cypher = cypher
            return list(self._candidates)
        if _SNAPSHOT_MARKER in cypher:
            parent_id = str(params["id"])
            self.snapshot_reads.append(parent_id)
            snapshot = self._snapshots.get(parent_id)
            return [snapshot] if snapshot else []
        if _PERSIST_MARKER in cypher:
            self.persists.append(dict(params))
            self.persist_cypher = cypher
            return [
                {
                    "name_stage": "accepted",
                    "authority_id": params["authority_properties"]["id"],
                }
            ]
        raise AssertionError(f"unexpected statement: {cypher[:80]}")


# --- the route decision ----------------------------------------------------


def test_source_free_structural_parent_takes_the_source_free_route() -> None:
    assert (
        _owning_graph_parent_lifecycle._structural_accept_route(
            origin=None, producer_count=0, live_child_count=3
        )
        == "source_free"
    )


def test_a_live_producing_source_leaves_the_name_reviewable() -> None:
    assert (
        _owning_graph_parent_lifecycle._structural_accept_route(
            origin=None, producer_count=1, live_child_count=3
        )
        is None
    )


def test_a_stamped_derived_parent_keeps_its_existing_route() -> None:
    assert (
        _owning_graph_parent_lifecycle._structural_accept_route(
            origin="derived", producer_count=1, live_child_count=2
        )
        == "derived"
    )


def test_a_childless_name_has_no_route_by_either_rule() -> None:
    assert (
        _owning_graph_parent_lifecycle._structural_accept_route(
            origin=None, producer_count=0, live_child_count=0
        )
        is None
    )
    assert (
        _owning_graph_parent_lifecycle._structural_accept_route(
            origin="derived", producer_count=0, live_child_count=0
        )
        is None
    )


def test_an_authored_name_is_never_admitted_structurally() -> None:
    for origin in ("pipeline", "catalog_edit", "deterministic"):
        assert (
            _owning_graph_parent_lifecycle._structural_accept_route(
                origin=origin, producer_count=0, live_child_count=4
            )
            is None
        )


# --- the promotion ---------------------------------------------------------


def test_source_free_parent_with_children_reaches_accepted() -> None:
    gc = _RecordingGraph(
        [_candidate("beta", origin=None, producer_count=0, live_child_count=3)],
        {"beta": _snapshot("beta", "poloidal_beta", "toroidal_beta", origin=None)},
    )

    promoted = _owning_graph_parent_lifecycle.structural_accept_derived_parents(gc)

    assert promoted == 1
    assert len(gc.persists) == 1
    updates = gc.persists[0]["parent_updates"]
    assert updates["name_stage"] == "accepted"
    assert updates["reviewer_model_name"] == "structural-inheritance"
    assert updates["origin"] == "derived"
    assert gc.persists[0]["require_derived"] is False
    assert gc.persists[0]["require_source_free"] is True


def test_the_same_parent_with_a_producing_source_is_refused_unwritten() -> None:
    gc = _RecordingGraph(
        [_candidate("beta", origin=None, producer_count=1, live_child_count=3)],
        {"beta": _snapshot("beta", "poloidal_beta", "toroidal_beta", origin=None)},
    )

    promoted = _owning_graph_parent_lifecycle.structural_accept_derived_parents(gc)

    assert promoted == 0
    assert gc.persists == []
    assert gc.snapshot_reads == []


def test_a_name_that_is_no_parent_at_all_is_refused_unwritten() -> None:
    gc = _RecordingGraph(
        [
            _candidate(
                "beta_inclination", origin=None, producer_count=0, live_child_count=0
            ),
            _candidate(
                "toroidal_beta", origin="pipeline", producer_count=0, live_child_count=2
            ),
        ],
        {},
    )

    promoted = _owning_graph_parent_lifecycle.structural_accept_derived_parents(gc)

    assert promoted == 0
    assert gc.persists == []
    assert gc.snapshot_reads == []


def test_a_source_path_no_edge_backs_is_cleared_by_the_promotion() -> None:
    gc = _RecordingGraph(
        [_candidate("beta", origin=None, producer_count=0, live_child_count=3)],
        {"beta": _snapshot("beta", "poloidal_beta", origin=None)},
    )

    _owning_graph_parent_lifecycle.structural_accept_derived_parents(gc)

    updates = gc.persists[0]["parent_updates"]
    for scalar in _owning_graph_parent_lifecycle._UNBACKED_SOURCE_PATH_SCALARS:
        assert scalar in updates
        assert updates[scalar] is None


def test_a_stamped_derived_parent_keeps_its_source_scalars() -> None:
    gc = _RecordingGraph(
        [
            _candidate(
                "plasma_beta", origin="derived", producer_count=1, live_child_count=1
            )
        ],
        {
            "plasma_beta": _snapshot(
                "plasma_beta", "normalized_toroidal_plasma_beta", origin="derived"
            )
        },
    )

    promoted = _owning_graph_parent_lifecycle.structural_accept_derived_parents(gc)

    assert promoted == 1
    updates = gc.persists[0]["parent_updates"]
    assert "source_path" not in updates
    assert "origin" not in updates
    assert gc.persists[0]["require_derived"] is True
    assert gc.persists[0]["require_source_free"] is False


# --- the guard that holds the admission through the write ------------------


def test_the_write_rechecks_that_no_source_produces_the_parent() -> None:
    record = _owning_graph_parent_lifecycle._structural_authority_record(
        _snapshot("beta", "poloidal_beta", "toroidal_beta", origin=None),
        accepting=True,
    )
    gc = MagicMock()
    gc.query.return_value = [{"name_stage": "accepted", "authority_id": record["id"]}]

    assert _owning_graph_parent_lifecycle._persist_structural_authority(
        gc,
        record,
        parent_updates={"name_stage": "accepted"},
        require_derived=False,
        require_source_free=True,
    )

    cypher = gc.query.call_args.args[0]
    params = gc.query.call_args.kwargs
    assert "$require_source_free = false OR NOT EXISTS" in cypher
    assert "(:StandardNameSource)-[:PRODUCED_NAME]->(parent)" in cypher
    assert params["require_source_free"] is True


def test_a_source_attached_mid_flight_costs_the_promotion() -> None:
    class _RaceGraph(_RecordingGraph):
        def query(self, cypher: str, **params: Any) -> list[dict[str, Any]]:
            if _PERSIST_MARKER in cypher:
                self.persists.append(dict(params))
                return []
            return super().query(cypher, **params)

    gc = _RaceGraph(
        [_candidate("beta", origin=None, producer_count=0, live_child_count=3)],
        {"beta": _snapshot("beta", "poloidal_beta", origin=None)},
    )

    raised = False
    try:
        _owning_graph_parent_lifecycle.structural_accept_derived_parents(gc)
    except _owning_graph_parent_lifecycle.StructuralAuthorityConflict:
        raised = True
    assert raised


# --- the selection that reaches the route ----------------------------------


def test_the_selection_admits_an_absent_origin_and_requires_a_child() -> None:
    gc = _RecordingGraph([], {})

    _owning_graph_parent_lifecycle.structural_accept_derived_parents(gc)

    cypher = gc.selection_cypher
    assert "sn.origin = 'derived' OR sn.origin IS NULL" in cypher
    assert "EXISTS { MATCH (:StandardName)-[:HAS_PARENT]->(sn) }" in cypher
    assert "(:StandardNameSource)-[:PRODUCED_NAME]->(sn)" in cypher
