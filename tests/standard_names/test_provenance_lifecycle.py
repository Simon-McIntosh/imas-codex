"""Semantic source retargeting and internal-history boundary tests."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from imas_codex.standard_names.merge import mark_catalog_name_approved, run_merge
from imas_codex.standard_names.provenance_lifecycle import (
    compact_unapproved_superseded,
    fetch_public_semantic_sources,
    find_semantic_source_invariant_violations,
    official_dd_documentation_url,
    refresh_renamed_source_mirrors,
    retarget_standard_name_sources,
    trace_standard_name_provenance,
)


def test_retarget_selector_includes_edge_migrated_sources_only() -> None:
    gc = MagicMock()
    gc.query.side_effect = [[{"moved": 2}], []]

    moved = retarget_standard_name_sources(gc, "old", "new", record_change=False)

    assert moved == 2
    cypher = gc.query.call_args_list[0].args[0]
    selector = cypher.split("WHERE", 1)[1].split("WITH new, old", 1)[0]
    assert "(sns)-[:PRODUCED_NAME]->(old)" in selector
    assert "sns.produced_sn_id = $old_name" in selector
    assert "(sns)-[:PRODUCED_NAME]->(new)" in selector
    assert "sns.produced_sn_id = $new_name" not in selector


def test_retarget_cache_uses_surviving_edge_bound_sources() -> None:
    gc = MagicMock()
    gc.query.return_value = [{"moved": 1}]

    retarget_standard_name_sources(gc, "old", "new", record_change=False)

    cypher = gc.query.call_args.args[0]
    selector = cypher.split("WHERE", 1)[1].split("WITH new, old", 1)[0]
    cache_projection = cypher.split("AS authoritative_paths", 1)[1].split(
        "RETURN size(moved)", 1
    )[0]

    sources = [
        {
            "id": "dd:accepted/path",
            "source_type": "dd",
            "source_id": "accepted/path",
            "edge_target": "new",
            "scalar_target": "new",
            "path": "dd:accepted/path",
        },
        {
            "id": "dd:rejected/path",
            "source_type": "dd",
            "source_id": "rejected/path",
            "edge_target": None,
            "scalar_target": "new",
            "path": "dd:rejected/path",
        },
        {
            "id": "derived:structural_parent",
            "source_type": "derived",
            "source_id": "derived:structural_parent",
            "edge_target": "new",
            "scalar_target": "new",
            "path": "derived:structural_parent",
        },
    ]
    selected = [
        source
        for source in sources
        if source["edge_target"] in {"old", "new"} or source["scalar_target"] == "old"
    ]

    assert [source["id"] for source in selected] == [
        "dd:accepted/path",
        "derived:structural_parent",
    ]
    assert [source["path"] for source in selected] == [
        "dd:accepted/path",
        "derived:structural_parent",
    ]
    assert "(sns)-[:PRODUCED_NAME]->(new)" in selector
    assert "sns.produced_sn_id = $new_name" not in selector
    assert "SET new.source_paths = []" in cypher
    assert "coalesce(old.source_paths" not in cache_projection
    assert "coalesce(new.source_paths" not in cache_projection
    assert "source.source_id STARTS WITH 'derived:'" in cypher
    assert "THEN source.source_id" in cypher
    assert "ELSE source.id" in cypher
    assert "[p IN authoritative_paths WHERE p IS NOT NULL] AS paths" in cache_projection


def test_retarget_query_repairs_all_source_mirrors() -> None:
    gc = MagicMock()
    gc.query.return_value = [{"moved": 1}]

    moved = retarget_standard_name_sources(gc, "old", "new", record_change=False)

    assert moved == 1
    cypher = gc.query.call_args.args[0]
    assert "DELETE prior" in cypher
    assert "MERGE (source)-[:PRODUCED_NAME]->(new)" in cypher
    assert "source.produced_sn_id = new.id" in cypher
    assert "OPTIONAL MATCH (source)-[:FROM_DD_PATH]->(dd:IMASNode)" in cypher
    assert "OPTIONAL MATCH (dd)-[dd_old:HAS_STANDARD_NAME]->(:StandardName)" in cypher
    assert "DELETE dd_old" in cypher
    assert "MERGE (dd)-[:HAS_STANDARD_NAME]->(new)" in cypher
    assert "MERGE (signal)-[:HAS_STANDARD_NAME]->(new)" in cypher
    assert "'dd:' + dd.id" in cypher
    assert "SET new.source_paths" in cypher
    assert (
        "FROM_DD_PATH" in cypher
        and "DELETE" not in cypher.split("FROM_DD_PATH")[1].split("FROM_SIGNAL")[0]
    )


def test_rename_mirror_refresh_separates_updates_from_matches() -> None:
    gc = MagicMock()
    gc.query.return_value = [{"refreshed": 0}]

    assert (
        refresh_renamed_source_mirrors(gc, [{"from": "old_name", "to": "new_name"}])
        == 0
    )

    cypher = gc.query.call_args.args[0]
    source_update = cypher.index("SET source.produced_sn_id = sn.id")
    review_match = cypher.index(
        "OPTIONAL MATCH (sn)-[:HAS_REVIEW]->(review:StandardNameReview)"
    )
    review_update = cypher.index("SET review.standard_name_id = sn.id")
    revision_match = cypher.index(
        "OPTIONAL MATCH (sn)-[:DOCS_REVISION_OF]->(revision:DocsRevision)"
    )

    assert "WITH sn, source" in cypher[source_update:review_match]
    assert "WITH sn, source" in cypher[review_update:revision_match]


@pytest.mark.graph
def test_rename_mirror_refresh_compiles_in_transaction(graph_client) -> None:
    from imas_codex.standard_names.cascade import _TransactionQueryAdapter

    with graph_client.session() as session:
        transaction = session.begin_transaction()
        try:
            adapter = _TransactionQueryAdapter(transaction)
            refreshed = refresh_renamed_source_mirrors(
                adapter,
                [
                    {
                        "from": "missing_predecessor_for_query_compilation",
                        "to": "missing_successor_for_query_compilation",
                    }
                ],
            )
            assert refreshed == 0
        finally:
            transaction.rollback()


def test_cleanup_manifest_is_unapproved_only() -> None:
    """The default (non-applying) call is a read-only unapproved-only manifest."""
    gc = MagicMock()
    gc.query.return_value = []
    assert compact_unapproved_superseded(gc) == []
    assert gc.query.call_count == 1
    cypher = gc.query.call_args.args[0]
    assert "old.catalog_approved_at IS NULL" in cypher
    assert "safe_to_compact" in cypher
    assert "DELETE" not in cypher


def test_cleanup_manifest_can_select_exact_names() -> None:
    """Targeted cleanup passes a deduplicated id allowlist to the graph."""
    gc = MagicMock()
    gc.query.return_value = []

    assert (
        compact_unapproved_superseded(
            gc,
            names=["obsolete_name", "obsolete_name", "other_name"],
        )
        == []
    )

    assert gc.query.call_args.kwargs["names"] == ["obsolete_name", "other_name"]
    assert "old.id IN $names" in gc.query.call_args.args[0]


def test_invariant_audit_checks_edge_scalar_and_backing_projection() -> None:
    gc = MagicMock()
    gc.query.return_value = []
    assert find_semantic_source_invariant_violations(gc) == []
    cypher = gc.query.call_args.args[0]
    assert "size(live_targets) <> 1" in cypher
    assert "source.produced_sn_id <> live_targets[0].id" in cypher
    assert "HAS_STANDARD_NAME" in cypher


def test_trace_separates_semantic_sources_and_internal_changes() -> None:
    gc = MagicMock()
    gc.query.side_effect = [
        [
            {
                "dd_path": "equilibrium/time_slice/global_quantities/ip",
                "dd_version": "4.1.0",
                "dd_snapshot_pinned": True,
                "signal_id": None,
                "semantic_facet": "measured",
                "coordinates": [],
            }
        ],
        [{"from_name": "ip", "to_name": "plasma_current", "operation": "human_edit"}],
    ]
    result = trace_standard_name_provenance(gc, "plasma_current")
    assert result["semantic_sources"][0]["semantic_facet"] == "measured"
    assert result["semantic_sources"][0]["dd_version"] == "4.1.0"
    assert result["internal_changes"][0]["from_name"] == "ip"
    assert "reviews" not in result


def test_official_dd_url_is_version_and_path_pinned() -> None:
    assert official_dd_documentation_url(
        "4.1.0", "equilibrium/time_slice/global_quantities/ip"
    ) == (
        "https://imas-data-dictionary.readthedocs.io/en/4.1.0/generated/ids/"
        "equilibrium.html#equilibrium-time_slice-global_quantities-ip"
    )


def test_public_dd_projection_never_falls_back_to_latest() -> None:
    gc = MagicMock()
    gc.query.return_value = [
        {
            "dd_path": "equilibrium/time_slice/global_quantities/ip",
            "dd_version": None,
            "signal_id": None,
        }
    ]
    with pytest.raises(ValueError, match="refusing to infer the latest"):
        fetch_public_semantic_sources(gc, "plasma_current")


def test_approval_requires_complete_merged_pr_metadata() -> None:
    gc = MagicMock()
    gc.query.return_value = [{"id": "plasma_current"}]
    assert mark_catalog_name_approved(
        "plasma_current",
        catalog_pr_number=42,
        catalog_pr_url="https://example.invalid/pull/42",
        catalog_merge_commit_sha="abc123",
        gc=gc,
    )
    cypher = gc.query.call_args.args[0]
    assert "sn.name_stage = 'approved'" in cypher
    assert "sn.docs_stage = 'accepted'" in cypher
    assert "catalog_approved_at" in cypher


def test_partial_approval_metadata_is_rejected_before_catalog_read(tmp_path) -> None:
    with pytest.raises(ValueError, match="PR number, PR URL, and merge commit"):
        run_merge(
            isnc_dir=tmp_path,
            base_ref="HEAD~1",
            catalog_pr_number=42,
        )
