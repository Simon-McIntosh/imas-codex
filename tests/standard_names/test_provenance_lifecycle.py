"""Semantic source retargeting and internal-history boundary tests."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from imas_codex.standard_names.merge import mark_catalog_name_approved, run_merge
from imas_codex.standard_names.provenance_lifecycle import (
    bind_sources_exclusively,
    compact_unapproved_superseded,
    fetch_public_semantic_sources,
    find_semantic_source_invariant_violations,
    official_dd_documentation_url,
    refresh_renamed_source_mirrors,
    reset_standard_name_sources,
    retarget_standard_name_sources,
    trace_standard_name_provenance,
)


def _migration_row(
    source_id: str,
    *,
    binding: str = "old",
    scalar: str | None = "old",
    status: str = "composed",
    claimed: bool = False,
    recorded: bool = False,
) -> dict[str, object]:
    return {
        "source_id": source_id,
        "source_exists": True,
        "source_status": status,
        "scalar_binding": scalar,
        "actively_claimed": claimed,
        "current_bindings": [binding] if binding else [],
        "manifest_recorded": recorded,
    }


def test_retarget_rejects_a_partially_admitted_explicit_cohort() -> None:
    from imas_codex.standard_names.attachment_audit import (
        AttachmentPairingGuardResult,
        AttachmentVerdict,
    )

    gc = MagicMock()
    gc.query.side_effect = [
        [_migration_row("dd:valid"), _migration_row("dd:invalid")],
    ]
    verdict = AttachmentVerdict(
        "dd:invalid", "bad/path", "new", "drafted", "geometry mismatch"
    )
    with patch(
        "imas_codex.standard_names.attachment_audit.guard_source_pairings",
        return_value=AttachmentPairingGuardResult(("dd:valid",), (verdict,)),
    ) as guard:
        with pytest.raises(ValueError, match="attachment rejected"):
            retarget_standard_name_sources(
                gc,
                "old",
                "new",
                source_ids=["dd:valid", "dd:invalid"],
                expected_current_bindings={"dd:valid": "old", "dd:invalid": "old"},
                record_change=False,
            )

    guard.assert_called_once_with(gc, "new", ["dd:invalid", "dd:valid"])
    assert gc.query.call_count == 1


def test_retarget_requires_an_explicit_nonempty_manifest() -> None:
    gc = MagicMock()

    with pytest.raises(ValueError, match="non-empty explicit source_ids"):
        retarget_standard_name_sources(gc, "old", "new")

    gc.query.assert_not_called()


def test_retarget_rejects_heterogeneous_expected_predecessors() -> None:
    gc = MagicMock()

    with pytest.raises(ValueError, match="heterogeneous"):
        retarget_standard_name_sources(
            gc,
            "old",
            "new",
            source_ids=["dd:a", "dd:b"],
            expected_current_bindings={"dd:a": "old", "dd:b": "other"},
        )

    gc.query.assert_not_called()


def test_exclusive_bind_can_bypass_guard_only_for_recovery_replay() -> None:
    gc = MagicMock()
    gc.query.return_value = [{"bound": 1}]

    bound = bind_sources_exclusively(
        gc, "restored_name", ["dd:history"], enforce_consistency=False
    )

    assert bound == 1
    assert gc.query.call_args.kwargs["source_ids"] == ["dd:history"]


def test_exclusive_bind_mutates_only_sources_admitted_by_guard() -> None:
    from imas_codex.standard_names.attachment_audit import (
        AttachmentPairingGuardResult,
    )

    gc = MagicMock()
    gc.query.return_value = [{"bound": 1}]
    with patch(
        "imas_codex.standard_names.attachment_audit.guard_source_pairings",
        return_value=AttachmentPairingGuardResult(("dd:valid",), ()),
    ) as guard:
        bound = bind_sources_exclusively(gc, "target_name", ["dd:invalid", "dd:valid"])

    assert bound == 1
    guard.assert_called_once_with(gc, "target_name", ["dd:invalid", "dd:valid"])
    assert gc.query.call_args.kwargs["source_ids"] == ["dd:valid"]


def test_retarget_query_repairs_exact_source_mirrors_and_both_caches() -> None:
    gc = MagicMock()
    gc.query.side_effect = [[_migration_row("dd:one")], [{"moved": 1}]]

    moved = retarget_standard_name_sources(
        gc,
        "old",
        "new",
        source_ids=["dd:one"],
        expected_current_bindings={"dd:one": "old"},
        record_change=False,
        enforce_consistency=False,
    )

    assert moved == 1
    cypher = gc.query.call_args_list[1].args[0]
    assert "COUNT { (source)-[:PRODUCED_NAME]->(:StandardName) } = 1" in cypher
    assert "DELETE prior" in cypher
    assert "MERGE (source)-[:PRODUCED_NAME]->(new)" in cypher
    assert "source.produced_sn_id = new.id" in cypher
    assert "OPTIONAL MATCH (source)-[:FROM_DD_PATH]->(dd:IMASNode)" in cypher
    assert "OPTIONAL MATCH (dd)-[dd_old:HAS_STANDARD_NAME]->(:StandardName)" in cypher
    assert "DELETE dd_old" in cypher
    assert "MERGE (dd)-[:HAS_STANDARD_NAME]->(new)" in cypher
    assert "MERGE (signal)-[:HAS_STANDARD_NAME]->(new)" in cypher
    assert "SET old.source_paths =" in cypher
    assert "new.source_paths =" in cypher
    assert "source_migration_manifest" in cypher
    assert (
        "FROM_DD_PATH" in cypher
        and "DELETE" not in cypher.split("FROM_DD_PATH")[1].split("FROM_SIGNAL")[0]
    )


@pytest.mark.parametrize(
    ("row", "detail"),
    [
        (
            _migration_row("dd:fanout") | {"current_bindings": ["old", "other"]},
            "bindings",
        ),
        (_migration_row("dd:scalar", scalar="other"), "scalar"),
        (_migration_row("dd:claimed", claimed=True), "claimed"),
        (_migration_row("dd:stale", status="stale"), "status"),
    ],
)
def test_retarget_rejects_precondition_drift_before_mutation(row, detail) -> None:
    gc = MagicMock()
    gc.query.return_value = [row]

    with pytest.raises(RuntimeError, match="compare-and-set") as exc:
        retarget_standard_name_sources(
            gc,
            "old",
            "new",
            source_ids=[row["source_id"]],
            expected_current_bindings={row["source_id"]: "old"},
            record_change=False,
            enforce_consistency=False,
        )

    assert detail in str(exc.value)
    assert gc.query.call_count == 1


def test_retarget_same_completed_manifest_is_idempotent() -> None:
    gc = MagicMock()
    gc.query.return_value = [
        _migration_row("dd:one", binding="new", scalar="new", recorded=True)
    ]

    moved = retarget_standard_name_sources(
        gc,
        "old",
        "new",
        source_ids=["dd:one"],
        expected_current_bindings={"dd:one": "old"},
        record_change=False,
        enforce_consistency=False,
    )

    assert moved == 0
    assert gc.query.call_count == 1


def test_retarget_conflicting_manifest_cannot_claim_an_unrecorded_postcondition() -> (
    None
):
    gc = MagicMock()
    gc.query.return_value = [
        _migration_row("dd:one", binding="new", scalar="new", recorded=False)
    ]

    with pytest.raises(RuntimeError, match="compare-and-set"):
        retarget_standard_name_sources(
            gc,
            "old",
            "new",
            source_ids=["dd:one"],
            expected_current_bindings={"dd:one": "old"},
            record_change=False,
            enforce_consistency=False,
        )


@pytest.mark.graph
def test_retarget_query_compiles_in_transaction(graph_client) -> None:
    from imas_codex.standard_names.cascade import _TransactionQueryAdapter

    with graph_client.session() as session:
        transaction = session.begin_transaction()
        try:
            adapter = _TransactionQueryAdapter(transaction)
            with pytest.raises(RuntimeError, match="compare-and-set"):
                retarget_standard_name_sources(
                    adapter,
                    "missing_predecessor_for_query_compilation",
                    "missing_successor_for_query_compilation",
                    source_ids=["dd:missing_source_for_query_compilation"],
                    expected_current_bindings={
                        "dd:missing_source_for_query_compilation": "missing_predecessor_for_query_compilation"
                    },
                    record_change=False,
                    enforce_consistency=False,
                )
        finally:
            transaction.rollback()


def _reset_preflight(
    *,
    status: str = "composed",
    stage: str = "reviewed",
    origin: str = "pipeline",
    catalog_pr_number: int | None = None,
    bindings: list[str] | None = None,
    scalar: str | None = "wrong_name",
    claimed: bool = False,
    event_exists: bool = False,
    event_reason: str | None = None,
) -> dict[str, object]:
    targets = bindings if bindings is not None else ["wrong_name"]
    return {
        "source_id": "dd:example/path",
        "source_exists": True,
        "status": status,
        "scalar": scalar,
        "actively_claimed": claimed,
        "binding_state": [
            {
                "id": target,
                "name_stage": stage,
                "catalog_pr_number": catalog_pr_number,
                "origin": origin,
                "other_sources": 0,
            }
            for target in targets
        ],
        "event_exists": event_exists,
        "event_reason": event_reason,
    }


def _reset_manifest() -> list[dict[str, object]]:
    return [
        {
            "source_id": "dd:example/path",
            "expected_status": "composed",
            "expected_scalar": "wrong_name",
            "expected_bindings": ["wrong_name"],
        }
    ]


def test_source_reset_accepted_binding_requires_explicit_acknowledgement() -> None:
    gc = MagicMock()
    gc.query.return_value = [_reset_preflight(stage="accepted")]

    with pytest.raises(RuntimeError, match="include_accepted"):
        reset_standard_name_sources(
            gc,
            _reset_manifest(),
            manifest_id="repair-one",
            reason="wrong physical owner",
        )

    assert gc.query.call_count == 1


def test_source_reset_publication_requires_separate_authority() -> None:
    gc = MagicMock()
    gc.query.return_value = [_reset_preflight(stage="approved", catalog_pr_number=42)]

    with pytest.raises(RuntimeError, match="publication authority"):
        reset_standard_name_sources(
            gc,
            _reset_manifest(),
            manifest_id="repair-one",
            reason="wrong physical owner",
            include_accepted=True,
        )


def test_source_reset_catalog_origin_alone_is_not_publication_authority() -> None:
    gc = MagicMock()
    gc.query.return_value = [_reset_preflight(origin="catalog_edit")]

    result = reset_standard_name_sources(
        gc,
        _reset_manifest(),
        manifest_id="repair-one",
        reason="wrong physical owner",
        dry_run=True,
    )

    assert result["dry_run"] is True
    assert result["potential_name_orphans"] == ["wrong_name"]
    assert gc.query.call_count == 1


def test_source_reset_holds_stale_source() -> None:
    gc = MagicMock()
    gc.query.return_value = [_reset_preflight(status="stale")]

    with pytest.raises(RuntimeError, match="status='stale'"):
        reset_standard_name_sources(
            gc,
            _reset_manifest(),
            manifest_id="repair-one",
            reason="wrong physical owner",
        )


def test_source_reset_writes_only_source_state_and_non_lifecycle_name_cache() -> None:
    gc = MagicMock()
    gc.query.side_effect = [
        [_reset_preflight()],
        [{"applied": 1, "source_ids": ["dd:example/path"]}],
    ]

    result = reset_standard_name_sources(
        gc,
        _reset_manifest(),
        manifest_id="repair-one",
        reason="wrong physical owner",
    )

    assert result["applied"] == 1
    assert result["potential_name_orphans"] == ["wrong_name"]
    cypher = gc.query.call_args_list[1].args[0]
    assert "DELETE edge" in cypher
    assert "source.status = 'extracted'" in cypher
    assert "source.produced_sn_id = null" in cypher
    assert "source.attempt_count = 0" in cypher
    assert "source.composed_at = null" in cypher
    assert "StandardNameSourceRetry" in cypher
    assert "FROM_DD_PATH|FROM_SIGNAL" in cypher
    assert "DELETE backing" not in cypher
    for lifecycle_field in (
        "name.name_stage",
        "name.status",
        "name.origin",
        "name.docs_stage",
        "name.reviewer_score_name",
    ):
        assert lifecycle_field not in cypher


def test_source_reset_same_completed_manifest_is_idempotent() -> None:
    preview_gc = MagicMock()
    preview_gc.query.return_value = [_reset_preflight()]
    preview = reset_standard_name_sources(
        preview_gc,
        _reset_manifest(),
        manifest_id="repair-one",
        reason="wrong physical owner",
        dry_run=True,
    )
    event_reason = (
        f"wrong physical owner [source-reset-manifest {preview['manifest_hash']}]"
    )
    gc = MagicMock()
    gc.query.return_value = [
        _reset_preflight(
            status="extracted",
            bindings=[],
            scalar=None,
            event_exists=True,
            event_reason=event_reason,
        )
    ]

    result = reset_standard_name_sources(
        gc,
        _reset_manifest(),
        manifest_id="repair-one",
        reason="wrong physical owner",
    )

    assert result["already_applied"] is True
    assert result["applied"] == 0
    assert gc.query.call_count == 1


def test_source_reset_conflicting_manifest_fails_after_completion() -> None:
    gc = MagicMock()
    gc.query.return_value = [
        _reset_preflight(
            status="extracted",
            bindings=[],
            scalar=None,
            event_exists=True,
            event_reason="different manifest content",
        )
    ]

    with pytest.raises(RuntimeError, match="compare-and-set"):
        reset_standard_name_sources(
            gc,
            _reset_manifest(),
            manifest_id="repair-one",
            reason="wrong physical owner",
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
    revision_update = cypher.index("SET revision.standard_name_id = sn.id")

    assert "WITH sn, source" in cypher[source_update:review_match]
    assert "WITH sn, source" in cypher[review_update:revision_match]
    assert revision_update > revision_match


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
    assert "source.produced_sn_id IS NULL" in cypher
    assert "source.produced_sn_id <> live_targets[0].id" in cypher
    assert "source.source_type IN $projection_source_types" in cypher
    assert "HAS_STANDARD_NAME" in cypher
    assert gc.query.call_args.kwargs["projection_source_types"] == ["dd", "signals"]


@pytest.mark.parametrize(
    ("source_type", "live_targets", "scalar", "mapped_ids", "is_violation"),
    [
        ("dd", ["density"], "density", [], True),
        ("signals", ["density"], "density", [], True),
        ("catalog", ["density"], "density", [], False),
        ("manual", ["density"], "density", [], False),
        ("derived", ["density"], "density", [], False),
        ("catalog", ["density"], "temperature", [], True),
        ("derived", ["density"], None, [], True),
        ("manual", ["density", "temperature"], "density", [], True),
        ("manual", [], None, [], True),
        ("dd", ["density"], "density", ["density"], False),
    ],
)
def test_invariant_audit_applies_projection_only_to_carrier_sources(
    source_type, live_targets, scalar, mapped_ids, is_violation
) -> None:
    row = {
        "source_id": f"{source_type}:example",
        "source_type": source_type,
        "produced_targets": live_targets,
        "live_targets": live_targets,
        "produced_sn_id": scalar,
        "mapped_ids": mapped_ids,
    }
    gc = MagicMock()
    gc.query.return_value = [row]

    violations = find_semantic_source_invariant_violations(gc)

    assert bool(violations) is is_violation


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
