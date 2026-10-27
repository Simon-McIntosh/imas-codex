"""Retired Standard Names cannot receive derived parent edges."""

from unittest.mock import patch

import pytest

from imas_codex.standard_names.derivation import DerivedEdge
from imas_codex.standard_names.graph_store.graph_lineage import (
    _filter_admissible_parents,
)
from imas_codex.standard_names.parents import superseded_parent_names
from imas_codex.standard_names.signed_manifest import _apply_structural_edge_reconcile

RETIRED = "area_of_flux_surface"
CURRENT = "poloidal_plane_cross_sectional_area_of_flux_surface"
CHILD = "surface_area_of_flux_surface"


class _Graph:
    def __init__(self):
        self.parents = {CHILD: {RETIRED}}
        self.queries = []

    def query(self, cypher, **params):
        self.queries.append((cypher, params))
        if "RETURN requested AS id" in cypher:
            return [{"id": name, "exists": True} for name in params["ids"]]
        if "WHERE sn.name_stage = 'superseded'" in cypher:
            return [{"id": RETIRED}] if RETIRED in params["ids"] else []
        if "RETURN nm AS name" in cypher:
            return [
                {
                    "name": name,
                    "axes": [],
                    "child_ids": [],
                    "origin": "catalog_edit" if name == RETIRED else "pipeline",
                    "name_stage": "superseded" if name == RETIRED else "accepted",
                }
                for name in params["names"]
            ]
        if "STANDARD_NAME_EDGE_IDENTITY_DISCOVERY" in cypher:
            return [
                {
                    "id": name,
                    "name_stage": "superseded" if name == RETIRED else "accepted",
                }
                for name in params["ids"]
            ]
        if "DELETE r" in cypher and "HAS_PARENT" in cypher:
            for row in params["recon"]:
                self.parents[row["child"]].intersection_update(row["keep"])
        if "MERGE (src)-[r:HAS_PARENT]" in cypher:
            for row in params["batch"]:
                self.parents.setdefault(row["from_name"], set()).add(row["to_name"])
        return []


def test_superseded_parent_is_vetoed_before_legacy_protection():
    graph = _Graph()
    edges = [
        {"from_name": CHILD, "to_name": RETIRED},
        {"from_name": "other_child", "to_name": CURRENT},
    ]

    kept = _filter_admissible_parents(edges, graph)

    assert kept == [edges[1]]


def test_superseded_lookup_finds_only_the_retired_candidate():
    graph = _Graph()

    assert superseded_parent_names(graph, {RETIRED, CURRENT}) == {RETIRED}
    assert any(
        "WHERE sn.name_stage = 'superseded'" in cypher
        and set(params["ids"]) == {RETIRED, CURRENT}
        for cypher, params in graph.queries
    )


def test_failed_topology_probe_cannot_admit_unchecked_parent():
    class _BrokenGraph:
        def query(self, _cypher, **_params):
            raise RuntimeError("topology unavailable")

    with pytest.raises(RuntimeError, match="topology unavailable"):
        _filter_admissible_parents(
            [{"from_name": CHILD, "to_name": RETIRED}], _BrokenGraph()
        )


def test_reconcile_releases_existing_edge_to_superseded_parent():
    graph = _Graph()
    edge = DerivedEdge(
        edge_type="HAS_PARENT",
        from_name=CHILD,
        to_name=RETIRED,
        props={"operator_kind": "qualifier"},
    )
    with patch(
        "imas_codex.standard_names.derivation.derive_edges", return_value=[edge]
    ):
        assert _apply_structural_edge_reconcile(graph, [CHILD]) == 1

    assert graph.parents[CHILD] == set()
    assert any("DELETE r" in cypher for cypher, _ in graph.queries)
