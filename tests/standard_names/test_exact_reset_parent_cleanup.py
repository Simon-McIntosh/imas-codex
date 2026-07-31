"""Exact reset cleanup contracts for relationship-created parent scaffolds."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner


def _graph(
    *,
    candidate_count: int = 1,
    skeleton_count: int = 1,
    mutation_error: Exception | None = None,
) -> MagicMock:
    graph = MagicMock()
    graph.__enter__.return_value = graph
    graph.__exit__.return_value = None

    def query(cypher: str, **_params):
        if "WITH collect(sn) AS reset_candidates" in cypher:
            return [
                {
                    "n": candidate_count + skeleton_count,
                    "candidate_count": candidate_count,
                    "skeleton_count": skeleton_count,
                }
            ]
        if "RETURN count(sn) AS n" in cypher:
            return [{"n": candidate_count}]
        if mutation_error is not None and "DETACH DELETE parent" in cypher:
            raise mutation_error
        if "RETURN size(candidate_changes) + skeleton_count AS n" in cypher:
            return [
                {
                    "n": candidate_count + skeleton_count,
                    "candidate_count": candidate_count,
                    "skeleton_count": skeleton_count,
                }
            ]
        if "RETURN count(" in cypher:
            return [{"n": 0}]
        return []

    graph.query = MagicMock(side_effect=query)
    return graph


def _clear(graph: MagicMock, **kwargs) -> int:
    from imas_codex.standard_names import graph_ops

    with patch.object(graph_ops, "GraphClient", return_value=graph):
        return graph_ops.clear_standard_names(**kwargs)


def _parent_cleanup_call(graph: MagicMock):
    return next(
        call
        for call in graph.query.call_args_list
        if "DETACH DELETE parent" in call.args[0]
    )


def test_exact_reset_deletes_candidate_parent_and_source_in_one_statement() -> None:
    """The failed candidate and its two binary operands share one transaction."""
    graph = _graph()

    assert _clear(graph, path_allowlist=["spectrometer/channel/isotope_ratio"]) == 2

    query = _parent_cleanup_call(graph).args[0]
    assert "candidate_parent_edge:HAS_PARENT" in query
    assert "collect(DISTINCT candidate_parent.id)" in query
    assert "DETACH DELETE sn" in query
    assert "DETACH DELETE source" in query
    assert "DETACH DELETE parent" in query
    assert query.index("DETACH DELETE sn") < query.index("DETACH DELETE parent")


def test_cleanup_is_limited_to_parents_of_deleted_exact_candidates() -> None:
    """Unrelated skeletons never enter the mutation's candidate parent set."""
    graph = _graph()

    _clear(graph, path_allowlist=["equilibrium/exact/path"])

    query = _parent_cleanup_call(graph).args[0]
    assert "UNWIND candidate_parent_ids AS parent_id" in query
    assert "MATCH (parent:StandardName {id: parent_id})" in query
    assert "MATCH (parent:StandardName)" not in query
    assert "src.id IN $path_allowlist" in query
    assert _parent_cleanup_call(graph).kwargs["path_allowlist"] == [
        "equilibrium/exact/path"
    ]


def test_parent_requires_null_pending_lifecycle_and_no_other_relationship() -> None:
    """Accepted, materialized, catalog-owned, and childful parents survive."""
    graph = _graph()

    _clear(graph, path_allowlist=["equilibrium/exact/path"])

    call = _parent_cleanup_call(graph)
    query = call.args[0]
    assert "parent.origin = 'derived'" in query
    assert "parent.origin IS NULL" in query
    assert "parent.transformation IS NOT NULL" in query
    assert "coalesce(parent.name_stage, '') IN ['', 'pending']" in query
    assert "NOT (parent.id IN reset_candidate_ids)" in query
    assert "all(key IN keys(parent)" in query
    assert "type(parent_rel) = 'PRODUCED_NAME'" in query
    assert "other.id = 'derived:' + parent.id" in query
    assert call.kwargs["reset_skeleton_parent_keys"] == [
        "id",
        "origin",
        "name_stage",
        "needs_composition",
        "claimed_at",
        "claim_token",
        "physical_base",
        "aggregation",
        "orbit",
        "population",
        "subject",
        "state",
        "transformation",
        "component",
        "coordinate",
        "process",
        "position",
        "region",
        "device",
        "geometric_base",
        "object",
        "geometry",
    ]
    assert "docs_stage" not in call.kwargs["reset_skeleton_parent_keys"]
    assert "catalog_approved_at" not in call.kwargs["reset_skeleton_parent_keys"]


def test_parent_and_exact_derived_source_are_claim_and_identity_fenced() -> None:
    """A claimed or non-structural source cannot be erased as scaffolding."""
    graph = _graph()

    _clear(graph, path_allowlist=["equilibrium/exact/path"])

    call = _parent_cleanup_call(graph)
    query = call.args[0]
    assert "parent.claimed_at IS NULL" in query
    assert "parent.claim_token IS NULL" in query
    assert "derived_source.id = 'derived:' + parent.id" in query
    assert "derived_source.source_type = 'derived'" in query
    assert "derived_source.source_id = parent.id" in query
    assert "derived_source.status IS NULL" in query
    assert "derived_source.status = 'composed'" in query
    assert "derived_source.created_at IS NOT NULL" in query
    assert "derived_source.composed_at IS NOT NULL" in query
    assert "derived_source.claimed_at IS NULL" in query
    assert "derived_source.claim_token IS NULL" in query
    assert "derived_source.produced_sn_id = parent.id" in query
    assert "mirror_source.produced_sn_id = parent.id" in query
    assert "mirror_source.id <> 'derived:' + parent.id" in query
    assert "FROM_DD_PATH" not in call.kwargs["reset_skeleton_composed_source_keys"]
    assert "all(key IN keys(derived_source)" in query

    assert call.kwargs["reset_skeleton_composed_source_keys"] == [
        *call.kwargs["reset_skeleton_minimal_source_keys"],
        "status",
        "created_at",
        "composed_at",
    ]


def test_only_structural_parent_edges_enter_cleanup() -> None:
    """A generic HAS_PARENT relation is provenance, not reset scaffolding."""
    graph = _graph()

    _clear(graph, path_allowlist=["equilibrium/exact/path"])

    count_query = graph.query.call_args_list[0].args[0]
    mutation_query = _parent_cleanup_call(graph).args[0]
    for query in (count_query, mutation_query):
        assert "candidate_parent_edge:HAS_PARENT" in query
        assert "candidate_parent_edge.operator_kind IS NOT NULL" in query
    assert "parent_rel.operator_kind IS NOT NULL" in count_query
    assert "parent_rel.operator_kind IS NOT NULL" in mutation_query


def test_normalized_parent_allows_only_writer_owned_outgoing_edges() -> None:
    """Grammar decomposition may survive only with exact typed identities."""
    graph = _graph()

    _clear(graph, path_allowlist=["equilibrium/exact/path"])

    call = _parent_cleanup_call(graph)
    query = call.args[0]
    assert "startNode(parent_rel) = parent" in query
    assert "endNode(parent_rel) = other" in query
    assert "type(parent_rel) = 'HAS_LOCUS'" in query
    assert "parent_rel.locus_token = other.id" in query
    assert "type(parent_rel) = 'HAS_SEGMENT'" in query
    assert "parent[parent_rel.segment] = other.value" in query
    assert "type(parent_rel) IN $reset_skeleton_segment_edge_types" in query
    assert "toLower(substring(type(parent_rel), 4))" in query
    assert call.kwargs["reset_skeleton_segment_edge_types"] == [
        "HAS_PHYSICAL_BASE",
        "HAS_SUBJECT",
        "HAS_TRANSFORMATION",
        "HAS_COMPONENT",
        "HAS_COORDINATE",
        "HAS_PROCESS",
        "HAS_POSITION",
        "HAS_REGION",
        "HAS_DEVICE",
        "HAS_GEOMETRIC_BASE",
        "HAS_AGGREGATION",
        "HAS_ORBIT",
        "HAS_POPULATION",
    ]


def test_dry_run_and_mutation_use_the_same_exact_source_scope() -> None:
    """Preview and mutation select candidates with the same exact allowlist."""
    path = "equilibrium/exact/path"
    preview = _graph()
    mutation = _graph()

    assert _clear(preview, path_allowlist=[path], dry_run=True) == 2
    assert _clear(mutation, path_allowlist=[path]) == 2

    assert preview.query.call_count == 1
    preview_count = preview.query.call_args_list[0]
    mutation_count = mutation.query.call_args_list[0]
    mutation_write = _parent_cleanup_call(mutation)
    for call in (preview_count, mutation_count, mutation_write):
        assert "src.id IN $path_allowlist" in call.args[0]
        assert call.kwargs["path_allowlist"] == [path]


def test_shared_parent_and_multi_edge_cardinality_is_distinct() -> None:
    """Two selected candidates sharing one parent report three deletions."""
    path = "equilibrium/exact/path"
    preview = _graph(candidate_count=2, skeleton_count=1)
    mutation = _graph(candidate_count=2, skeleton_count=1)

    assert _clear(preview, path_allowlist=[path], dry_run=True) == 3
    assert _clear(mutation, path_allowlist=[path]) == 3

    count_query = preview.query.call_args_list[0].args[0]
    write_query = _parent_cleanup_call(mutation).args[0]
    assert "WITH DISTINCT sn" in count_query
    assert "collect(DISTINCT candidate_parent)" in count_query
    assert "count(DISTINCT parent) AS skeleton_count" in count_query
    assert "collect(DISTINCT reset_candidate_id)" in write_query
    assert "WITH DISTINCT reset_candidate_ids, parent_id" in write_query
    assert "count(DISTINCT parent) AS skeleton_count" in write_query


def test_reset_only_dry_run_reports_exact_cleanup_count() -> None:
    """The CLI executes the read-only preview instead of skipping reset logic."""
    from imas_codex.cli.sn import sn

    with patch(
        "imas_codex.standard_names.graph_ops.clear_standard_names",
        return_value=2,
    ) as clear:
        result = CliRunner().invoke(
            sn,
            [
                "run",
                "--focus",
                "equilibrium/exact/path",
                "--reset-to",
                "extracted",
                "--reset-only",
                "--dry-run",
            ],
        )

    assert result.exit_code == 0, result.output
    clear.assert_called_once()
    assert clear.call_args.kwargs["path_allowlist"] == ["equilibrium/exact/path"]
    assert clear.call_args.kwargs["dry_run"] is True
    assert "would clear 2 SN nodes" in result.output
    assert "preview complete" in result.output


def test_prefix_scoped_clear_does_not_run_exact_parent_cleanup() -> None:
    """Only an explicit path allowlist owns relationship-created scaffolds."""
    graph = _graph()

    _clear(graph, ids_filter="equilibrium")

    assert not any(
        "DETACH DELETE parent" in call.args[0] for call in graph.query.call_args_list
    )


def test_cleanup_failure_rolls_back_candidate_and_parent_statement() -> None:
    """No post-statement cleanup can commit after the atomic delete fails."""
    graph = _graph(mutation_error=RuntimeError("injected statement failure"))

    with pytest.raises(RuntimeError, match="injected statement failure"):
        _clear(graph, path_allowlist=["equilibrium/exact/path"])

    mutations = [
        call.args[0]
        for call in graph.query.call_args_list
        if "DETACH DELETE sn" in call.args[0]
        or "DETACH DELETE parent" in call.args[0]
        or "DETACH DELETE source" in call.args[0]
    ]
    assert len(mutations) == 1
    assert "DETACH DELETE sn" in mutations[0]
    assert "DETACH DELETE parent" in mutations[0]
    assert graph.query.call_count == 2


def test_parent_cleanup_records_a_separate_internal_deletion_event() -> None:
    """Scaffold retirement remains auditable without borrowing child identity."""
    graph = _graph()

    _clear(graph, path_allowlist=["equilibrium/exact/path"])

    call = _parent_cleanup_call(graph)
    query = call.args[0]
    assert query.count("CREATE (") >= 2
    assert "from_name: parent.id" in query
    assert call.kwargs["reset_skeleton_deletion_operation"] == (
        "remove_skeleton_placeholder"
    )
