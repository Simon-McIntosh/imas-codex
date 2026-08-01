"""Batch-local cleanup contracts for relationship-created placeholders."""

from __future__ import annotations

from unittest.mock import MagicMock, patch


def _name() -> dict:
    return {
        "id": "electron_pressure",
        "description": "Electron pressure.",
        "kind": "scalar",
        "unit": None,
        "source_types": [],
        "source_id": None,
        "physics_domain": None,
    }


def _write_with_candidates(candidate_ids: set[str]):
    from imas_codex.standard_names import graph_ops

    main = MagicMock()
    main.__enter__.return_value = main
    main.__exit__.return_value = None
    main.query.return_value = []
    sweep = MagicMock()
    sweep.__enter__.return_value = sweep
    sweep.__exit__.return_value = None
    sweep.query.return_value = [{"swept": len(candidate_ids)}]

    graph_clients = [main, sweep] if candidate_ids else [main]
    with (
        patch.object(graph_ops, "GraphClient", side_effect=graph_clients) as client,
        patch(
            "imas_codex.standard_names.protection.filter_protected",
            side_effect=lambda names, **_kwargs: (names, []),
        ),
        patch.object(graph_ops, "_write_grammar_decomposition", return_value=[]),
        patch.object(
            graph_ops,
            "_write_standard_name_edges",
            return_value=candidate_ids,
        ),
    ):
        assert graph_ops.write_standard_names([_name()]) == 1
    return client, main, sweep


def test_sweep_is_limited_to_current_write_endpoints() -> None:
    _, _, sweep = _write_with_candidates(
        {"touched_parent", "touched_predecessor", "touched_error"}
    )

    query = sweep.query.call_args.args[0]
    assert "sn.id IN $candidate_ids" in query
    assert sweep.query.call_args.kwargs["candidate_ids"] == [
        "touched_error",
        "touched_parent",
        "touched_predecessor",
    ]
    assert "CREATE (change:StandardNameChange" in query
    assert sweep.query.call_args.kwargs["deletion_operation"] == (
        "remove_skeleton_placeholder"
    )


def test_empty_endpoint_set_skips_placeholder_query() -> None:
    client, main, sweep = _write_with_candidates(set())

    assert client.call_count == 1
    assert main.query.called
    sweep.query.assert_not_called()


def test_durable_and_source_backed_names_are_excluded() -> None:
    _, _, sweep = _write_with_candidates({"touched_endpoint"})

    query = sweep.query.call_args.args[0]
    assert "coalesce(sn.name_stage, '') IN ['', 'pending']" in query
    assert "NOT EXISTS { ()-[:HAS_PARENT]->(sn) }" in query
    assert "NOT EXISTS { ()-[:HAS_ERROR]->(sn) }" in query
    assert "NOT EXISTS { ()-[:HAS_STANDARD_NAME]->(sn) }" in query
    assert "(:StandardNameSource)-[:PRODUCED_NAME]->(sn)" in query
    assert "source.produced_sn_id = sn.id" in query


def test_edge_writer_reports_every_standard_name_merge_endpoint() -> None:
    from imas_codex.standard_names.derivation import DerivedEdge
    from imas_codex.standard_names.graph_ops import _write_standard_name_edges

    edges = {
        "written_name": [
            DerivedEdge(
                edge_type="HAS_PARENT",
                from_name="written_name",
                to_name="closure_parent",
                props={"operator_kind": "binary"},
            ),
            DerivedEdge(
                edge_type="HAS_LOCUS",
                from_name="written_name",
                to_name="ignored_locus_node",
                props={"locus_token": "magnetic_axis"},
            ),
        ],
        "closure_parent": [
            DerivedEdge(
                edge_type="HAS_ERROR",
                from_name="error_source",
                to_name="error_target",
                props={"error_type": "upper"},
            )
        ],
        "error_source": [],
    }
    graph = MagicMock()

    with (
        patch(
            "imas_codex.standard_names.derivation.derive_edges",
            side_effect=lambda name: edges.get(name, []),
        ),
        patch(
            "imas_codex.standard_names.graph_ops._filter_admissible_parents",
            side_effect=lambda batch, _graph, **_kwargs: batch,
        ),
    ):
        candidates = _write_standard_name_edges(
            graph,
            [
                {
                    "id": "written_name",
                    "predecessor": "predecessor_placeholder",
                    "successor": "successor_placeholder",
                    "primary_cluster_id": "cluster",
                    "physics_domain": "transport",
                }
            ],
        )

    assert candidates == {
        "written_name",
        "closure_parent",
        "error_source",
        "error_target",
        "predecessor_placeholder",
        "successor_placeholder",
    }
    assert "ignored_locus_node" not in candidates
