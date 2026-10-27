"""Deletion guards for automatic standard-name cleanup paths."""

from __future__ import annotations

import inspect
from unittest.mock import MagicMock

import pytest

from imas_codex.standard_names import graph_ops, protection, provenance_lifecycle
from imas_codex.standard_names.graph_store import (
    graph_foundation as _owning_graph_foundation,
    graph_lineage as _owning_graph_lineage,
)
from tests.standard_names.source_scan import graph_source_text


def test_positive_llm_spend_is_materialized_and_refuses_deletion() -> None:
    gc = MagicMock()
    gc.query.side_effect = [
        [{"linked": 1894}],
        [
            {
                "id": "etendue_of_spectrometer_channel",
                "reasons": ["recorded_llm_spend_usd=0.84"],
                "recorded_spend": 0.84,
            }
        ],
    ]

    with pytest.raises(
        protection.ProtectedDeletionError,
        match=r"etendue_of_spectrometer_channel.*recorded_llm_spend_usd=0.84",
    ):
        protection.refuse_protected_automatic_deletion(
            gc,
            ["etendue_of_spectrometer_channel"],
            operation="test cleanup",
        )

    materialize = gc.query.call_args_list[0].args[0]
    assert "coalesce(cost.llm_cost, 0.0) > 0.0" in materialize
    assert "MERGE (cost)-[:FOR_STANDARD_NAME]->(sn)" in materialize
    assert "UNWIND cost.standard_name_ids AS name_id" in materialize


def test_structural_cleanup_refuses_the_whole_oversized_batch() -> None:
    gc = MagicMock()
    candidates = [f"parent_{index}" for index in range(81)]

    with pytest.raises(
        _owning_graph_foundation.DerivedParentCleanupRefusal,
        match="81 candidate identities exceeds the 80-identity ceiling",
    ):
        _owning_graph_lineage._delete_derived_parent_nodes(gc, candidates)

    gc.query.assert_not_called()


def test_structural_delete_requires_placeholder_and_keeps_recovery_material() -> None:
    gc = MagicMock()
    gc.query.side_effect = [[{"linked": 0}], [], [{"deleted": 1}]]

    assert _owning_graph_lineage._delete_derived_parent_nodes(gc, ["parent_name"]) == 1

    statement = gc.query.call_args_list[-1].args[0]
    assert "WHERE sn.needs_composition = true" in statement
    assert "StandardNameDeletionSnapshot" in statement
    assert "StandardNameDeletedEdge" in statement
    assert "relationship_type: type(edge)" in statement


def test_every_automatic_deletion_route_calls_the_common_refusal() -> None:
    graph_source = graph_source_text()
    provenance_source = inspect.getsource(provenance_lifecycle)

    assert 'operation="structural derived-parent cleanup"' in graph_source
    assert 'operation="skeleton placeholder cleanup"' in graph_source
    assert 'operation="provenance-orphan retirement"' in provenance_source


def test_all_ledgered_deletions_snapshot_node_and_edges() -> None:
    clause = provenance_lifecycle.deletion_change_cypher("sn")

    assert "SET snapshot = properties(sn)" in clause
    assert "HAS_DELETION_SNAPSHOT" in clause
    assert "HAS_EDGE_SNAPSHOT" in clause
    assert "OPTIONAL MATCH (sn)-[edge]-(neighbor)" in clause
