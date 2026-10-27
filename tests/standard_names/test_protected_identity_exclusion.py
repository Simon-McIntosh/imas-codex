"""Candidate filtering and deletion backstop for protected identities."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, call, patch

import pytest

from imas_codex.standard_names import graph_ops, protection
from imas_codex.standard_names.graph_store import (
    graph_catalog as _owning_graph_catalog,
    graph_foundation as _owning_graph_foundation,
    graph_lineage as _owning_graph_lineage,
)


class _SkeletonCleanupGraph:
    def __init__(
        self,
        *,
        placeholder_ids: set[str],
        protected_ids: set[str],
    ) -> None:
        self.placeholder_ids = placeholder_ids
        self.protected_ids = protected_ids
        self.queries: list[tuple[str, dict[str, Any]]] = []

    def query(self, cypher: str, **params: Any) -> list[dict[str, Any]]:
        self.queries.append((cypher, params))
        if "STANDARD_NAME_SKELETON_PLACEHOLDER_SELECTION" in cypher:
            return [
                {"candidate_id": candidate_id}
                for candidate_id in params["candidate_ids"]
                if candidate_id in self.placeholder_ids
            ]
        if "MERGE (cost)-[:FOR_STANDARD_NAME]->(sn)" in cypher:
            return [{"linked": len(self.protected_ids)}]
        if "OPTIONAL MATCH (cost:LLMCost)-[:FOR_STANDARD_NAME]->(sn)" in cypher:
            return [
                {
                    "id": candidate_id,
                    "reasons": ["recorded_llm_spend_usd=0.25"],
                    "recorded_spend": 0.25,
                }
                for candidate_id in params["names"]
                if candidate_id in self.protected_ids
            ]
        if "DETACH DELETE sn" in cypher:
            return [{"swept": len(params["candidate_ids"])}]
        return []


def _composed_name(name_id: str) -> dict[str, Any]:
    return {
        "id": name_id,
        "description": "A composed physical quantity.",
        "kind": "scalar",
        "unit": None,
        "source_types": [],
        "source_id": None,
        "physics_domain": None,
    }


def _write_with_skeleton_candidates(
    graph: _SkeletonCleanupGraph,
    *,
    composed_id: str,
    candidate_ids: set[str],
) -> int:
    def filter_protected(
        names: list[dict[str, Any]],
        *,
        protected_names: set[str] | None = None,
        **_kwargs: Any,
    ) -> tuple[list[dict[str, Any]], list[str]]:
        if protected_names is None:
            return names, []
        protected_ids = [
            str(name["id"]) for name in names if name["id"] in protected_names
        ]
        return [
            name for name in names if name["id"] not in protected_names
        ], protected_ids

    with (
        patch.object(
            _owning_graph_catalog, "_write_grammar_decomposition", return_value=[]
        ),
        patch.object(
            _owning_graph_lineage,
            "_write_standard_name_edges",
            return_value=candidate_ids,
        ),
        patch.object(
            protection,
            "filter_protected",
            side_effect=filter_protected,
        ),
    ):
        result = _owning_graph_catalog.write_standard_names(
            [_composed_name(composed_id)],
            gc=graph,
        )
    assert isinstance(result, int)
    return result


def _graph_with_protected_parent() -> MagicMock:
    gc = MagicMock()

    def query(cypher: str, **params: Any) -> list[dict[str, Any]]:
        if "RETURN DISTINCT parent.id AS parent_id" in cypher:
            return [
                {"parent_id": "ion_power_density"},
                {"parent_id": "unspent_structural_parent"},
            ]
        if "MERGE (cost)-[:FOR_STANDARD_NAME]->(sn)" in cypher:
            return [{"linked": 1}]
        if "OPTIONAL MATCH (cost:LLMCost)-[:FOR_STANDARD_NAME]->(sn)" in cypher:
            assert params["names"] in (
                ["ion_power_density"],
                ["ion_power_density", "unspent_structural_parent"],
            )
            return [
                {
                    "id": "ion_power_density",
                    "reasons": ["recorded_llm_spend_usd=0.25"],
                    "recorded_spend": 0.25,
                }
            ]
        raise AssertionError(f"unexpected query: {cypher}")

    gc.query.side_effect = query
    return gc


def test_structural_candidate_selector_excludes_protected_identity() -> None:
    gc = _graph_with_protected_parent()

    assert _owning_graph_lineage._query_derived_parents_for_admission_cleanup(gc) == [
        "unspent_structural_parent"
    ]


def test_direct_structural_delete_still_refuses_protected_identity() -> None:
    gc = _graph_with_protected_parent()

    with pytest.raises(
        _owning_graph_foundation.DerivedParentCleanupRefusal,
        match=r"ion_power_density.*recorded_llm_spend_usd=0.25",
    ) as raised:
        _owning_graph_lineage._delete_derived_parent_nodes(gc, ["ion_power_density"])

    assert isinstance(raised.value.__cause__, protection.ProtectedDeletionError)
    assert not any(
        "DETACH DELETE sn" in call.args[0] for call in gc.query.call_args_list
    )


def test_childless_selector_excludes_protected_before_structural_delete() -> None:
    gc = MagicMock()

    def query(cypher: str, **params: Any) -> list[dict[str, Any]]:
        if "NOT EXISTS { MATCH (:StandardName)-[:HAS_PARENT]->(p) }" in cypher:
            return [
                {"id": "ion_power_density"},
                {"id": "unspent_structural_parent"},
            ]
        if "MERGE (cost)-[:FOR_STANDARD_NAME]->(sn)" in cypher:
            return [{"linked": 1}]
        if "OPTIONAL MATCH (cost:LLMCost)-[:FOR_STANDARD_NAME]->(sn)" in cypher:
            assert params["names"] in (
                ["ion_power_density"],
                ["ion_power_density", "unspent_structural_parent"],
            )
            return [
                {
                    "id": "ion_power_density",
                    "reasons": ["recorded_llm_spend_usd=0.25"],
                    "recorded_spend": 0.25,
                }
            ]
        if "MATCH (dr:DocsRevision)" in cypher:
            return [{"n": 0}]
        raise AssertionError(f"unexpected query: {cypher}")

    gc.query.side_effect = query
    with (
        patch.object(
            _owning_graph_lineage, "_query_seedable_derived_parents", return_value=[]
        ),
        patch.object(
            _owning_graph_lineage,
            "_query_legacy_repairable_derived_parents",
            return_value=[],
        ),
        patch.object(
            _owning_graph_lineage,
            "_query_derived_parents_for_admission_cleanup",
            return_value=[],
        ),
        patch.object(
            _owning_graph_lineage,
            "_delete_derived_parent_nodes",
            side_effect=[0, 1],
        ) as delete_nodes,
    ):
        changed = _owning_graph_lineage.normalize_derived_parent_lifecycle(gc)

    assert changed == 1
    assert delete_nodes.call_args_list == [
        call(gc, []),
        call(gc, ["unspent_structural_parent"]),
    ]


def test_paid_composed_identity_is_filtered_before_skeleton_refusal() -> None:
    composed_id = "ion_power_density"
    graph = _SkeletonCleanupGraph(
        placeholder_ids=set(),
        protected_ids={composed_id},
    )

    assert (
        _write_with_skeleton_candidates(
            graph,
            composed_id=composed_id,
            candidate_ids={composed_id},
        )
        == 1
    )

    selection, params = next(
        (cypher, params)
        for cypher, params in graph.queries
        if "STANDARD_NAME_SKELETON_PLACEHOLDER_SELECTION" in cypher
    )
    assert params["candidate_ids"] == [composed_id]
    assert "sn.created_at IS NULL" in selection
    assert not any("DETACH DELETE sn" in cypher for cypher, _ in graph.queries)


def test_paid_id_only_placeholder_is_still_refused() -> None:
    composed_id = "ion_power_density"
    placeholder_id = "density_at_pedestal_top"
    graph = _SkeletonCleanupGraph(
        placeholder_ids={placeholder_id},
        protected_ids={placeholder_id},
    )

    with pytest.raises(
        protection.ProtectedDeletionError,
        match=r"density_at_pedestal_top.*recorded_llm_spend_usd=0.25",
    ):
        _write_with_skeleton_candidates(
            graph,
            composed_id=composed_id,
            candidate_ids={composed_id, placeholder_id},
        )

    assert not any("DETACH DELETE sn" in cypher for cypher, _ in graph.queries)
