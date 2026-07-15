"""Semantic source retargeting and internal-history boundary tests."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from imas_codex.standard_names.merge import mark_catalog_name_approved, run_merge
from imas_codex.standard_names.provenance_lifecycle import (
    build_unapproved_cleanup_manifest,
    fetch_public_semantic_sources,
    find_semantic_source_invariant_violations,
    official_dd_documentation_url,
    report_unpinned_dd_sources,
    retarget_standard_name_sources,
    trace_standard_name_provenance,
)


def test_retarget_query_enforces_all_source_mirrors() -> None:
    gc = MagicMock()
    gc.query.side_effect = [[{"moved": 2}], []]

    moved = retarget_standard_name_sources(gc, "old", "new", record_change=False)

    assert moved == 2
    cypher = gc.query.call_args_list[0].args[0]
    assert "DELETE prior" in cypher
    assert "MERGE (source)-[:PRODUCED_NAME]->(new)" in cypher
    assert "source.produced_sn_id = new.id" in cypher
    assert "MERGE (dd)-[:HAS_STANDARD_NAME]->(new)" in cypher
    assert "MERGE (signal)-[:HAS_STANDARD_NAME]->(new)" in cypher
    assert "new.source_paths" in cypher
    assert (
        "FROM_DD_PATH" in cypher
        and "DELETE" not in cypher.split("FROM_DD_PATH")[1].split("FROM_SIGNAL")[0]
    )


def test_cleanup_manifest_is_unapproved_only() -> None:
    gc = MagicMock()
    gc.query.return_value = []
    assert build_unapproved_cleanup_manifest(gc) == []
    cypher = gc.query.call_args.args[0]
    assert "old.catalog_approved_at IS NULL" in cypher
    assert "safe_to_compact" in cypher


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


def test_legacy_backfill_report_is_read_only_and_never_guesses() -> None:
    gc = MagicMock()
    gc.query.return_value = [
        {
            "source_id": "dd:equilibrium/time_slice/global_quantities/ip",
            "dd_path": "equilibrium/time_slice/global_quantities/ip",
            "dd_version": None,
            "missing_fields": ["dd_version", "dd_documentation"],
            "reason": "original_dd_version_unprovable",
            "safe_to_backfill": False,
        }
    ]
    rows = report_unpinned_dd_sources(gc)
    assert rows[0]["safe_to_backfill"] is False
    cypher = gc.query.call_args.args[0]
    assert "SET " not in cypher and "MERGE " not in cypher


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
