"""Every caller of the attachment consistency reconcile threads its run id.

``reconcile_attachment_consistency`` records ``run_id`` on every
``detach_inconsistent_attachment`` change it writes so a detachment can be
traced back to the run that performed it (see
:func:`imas_codex.standard_names.attachment_audit.count_unattributed_detachments`).
The record accepting a run id is useless if no caller supplies one — these
tests pin the two production call sites, the pool loop's global-maintenance
pass and the provenance rebuild's post-replay consistency check, to actually
pass their active run identifier through.
"""

from __future__ import annotations

import asyncio
from contextlib import ExitStack
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from imas_codex.standard_names.attachment_audit import AttachmentAuditResult

_GO = "imas_codex.standard_names.graph_ops"
_LOOP = "imas_codex.standard_names.loop"


def _graph_context() -> tuple[MagicMock, MagicMock]:
    graph = MagicMock()
    graph.query.return_value = [{"cnt": 1}]
    context = MagicMock()
    context.__enter__.return_value = graph
    context.__exit__.return_value = False
    return context, graph


def _maintenance_mocks(stack: ExitStack) -> None:
    """Patch the graph-wide maintenance call set used by ``run_sn_pools``."""
    specs: dict[str, object] = {
        "mark_orphaned_standard_name_runs_stale": 0,
        "reconcile_standard_name_sources": {},
        "reconcile_vocab_gaps": {},
        "revive_unit_skipped_sources": {},
        "retry_vocab_gap_sources_on_grammar_change": {},
        "reconcile_provenance": {},
        "reconcile_source_status_liveness": {},
        "retire_unreachable_hint_edits": 0,
        "reconcile_grammar_segments": {},
        "reconcile_catalog_status": {},
        "reconcile_reviewable_name_stage": {},
        "reconcile_standard_name_cocos_links": {},
        "reconcile_standard_name_unit_edges": {},
        "reconcile_standard_name_dd_edges": {},
        "reconcile_standard_name_source_paths": {},
        "promote_stranded_reviewed": {},
        "rederive_structural_edges": {},
        "seed_parent_sources": 0,
        "normalize_derived_parent_lifecycle": 0,
        "structural_accept_derived_parents": 0,
        "reconcile_orphan_parent_sources": 0,
        "release_all_orphan_claims": {},
        "resolve_doc_links": {},
    }
    for name, result in specs.items():
        stack.enter_context(patch(f"{_GO}.{name}", return_value=result))
    stack.enter_context(
        patch(
            "imas_codex.graph.dd_graph_ops.reconcile_dd_unit_corrections",
            return_value={},
        )
    )
    stack.enter_context(
        patch(
            "imas_codex.standard_names.source_refresh.refresh_drifted_sources",
            return_value={},
        )
    )
    stack.enter_context(
        patch(
            "imas_codex.standard_names.harmonize.restamp_harmonized_families",
            return_value={},
        )
    )
    stack.enter_context(
        patch(
            "imas_codex.standard_names.orphan_sweep.run_orphan_sweep_loop",
            new_callable=AsyncMock,
        )
    )
    stack.enter_context(
        patch(
            "imas_codex.discovery.base.embed_worker.embed_description_worker",
            new_callable=AsyncMock,
        )
    )


@pytest.mark.asyncio
async def test_pool_loop_passes_active_run_id_to_attachment_reconcile() -> None:
    """``run_sn_pools`` threads its own ``run_id`` into the attachment reconcile.

    Without this wiring a detachment written by the pool loop's global
    maintenance pass carries ``run_id IS NULL`` and cannot be traced back to
    the drain that performed it.
    """
    graph_context, _ = _graph_context()
    with ExitStack() as stack:
        _maintenance_mocks(stack)
        attach_mock = stack.enter_context(
            patch(
                "imas_codex.standard_names.attachment_audit."
                "reconcile_attachment_consistency",
                return_value=AttachmentAuditResult(),
            )
        )
        stack.enter_context(patch(f"{_GO}.create_sn_run_open"))
        stack.enter_context(patch(f"{_GO}.finalize_sn_run"))
        stack.enter_context(
            patch(
                f"{_GO}.scoped_terminal_residue",
                return_value={"total": 0, "names": [], "sources": []},
            )
        )
        stack.enter_context(
            patch(
                "imas_codex.standard_names.ledger.find_provenance_orphans",
                return_value=[],
            )
        )
        stack.enter_context(
            patch(
                "imas_codex.standard_names.audits."
                "find_flux_surface_reduction_violations",
                return_value=[],
            )
        )
        stack.enter_context(
            patch(
                "imas_codex.standard_names.audits.find_removed_dd_sources",
                return_value=[],
            )
        )
        stack.enter_context(patch(f"{_GO}.persist_outcome_snapshot", return_value={}))
        stack.enter_context(patch(f"{_GO}.reset_persist_outcomes"))
        stack.enter_context(patch(f"{_LOOP}._build_pool_specs", return_value=[]))
        stack.enter_context(
            patch(
                "imas_codex.standard_names.pools.run_pools",
                new_callable=AsyncMock,
                return_value={},
            )
        )
        stack.enter_context(
            patch(
                "imas_codex.standard_names.budget.BudgetManager.start",
                new_callable=AsyncMock,
            )
        )
        stack.enter_context(
            patch(
                "imas_codex.standard_names.budget.BudgetManager.drain_pending",
                new_callable=AsyncMock,
                return_value=True,
            )
        )
        stack.enter_context(
            patch(
                "imas_codex.standard_names.budget.BudgetManager."
                "_get_total_spent_sync",
                return_value=0.0,
            )
        )
        stack.enter_context(
            patch("imas_codex.graph.client.GraphClient", return_value=graph_context)
        )

        from imas_codex.standard_names.loop import run_sn_pools

        stop = asyncio.Event()
        stop.set()
        summary = await run_sn_pools(
            cost_limit=5.0,
            scope_run_id="bounded-run",
            stop_event=stop,
            skip_global_maintenance=False,
        )

    attach_mock.assert_called_once()
    assert attach_mock.call_args.kwargs["run_id"] == summary.run_id


def test_provenance_rebuild_passes_run_id_to_attachment_reconcile() -> None:
    """``rebuild_provenance`` threads a run id into the attachment reconcile.

    Without this wiring a detachment written by the drain's post-replay
    consistency check carries ``run_id IS NULL`` and cannot be traced back to
    the rebuild invocation that performed it.
    """
    import imas_codex.standard_names.provenance_rebuild as pr

    gc = MagicMock()
    with (
        patch.object(pr, "find_provenance_orphans", return_value=[]),
        patch.object(pr, "find_edge_scalar_desyncs", return_value=[]),
        patch.object(pr, "reattach_produced_name_edges", return_value=0),
        patch.object(pr, "_run_deterministic_fixpoints"),
        patch.object(pr, "find_orphan_parent_source_candidates", return_value=[]),
        patch.object(
            pr,
            "classify_orphan_parent_source_candidates",
            return_value={"repairable": [], "rejected_derived": []},
        ),
        patch.object(pr, "reconcile_orphan_parent_sources", return_value=0),
        patch.object(pr, "_fetch_dd_source_paths", return_value={}),
        patch.object(pr, "_fetch_change_history_sources", return_value={}),
        patch.object(pr, "_fetch_pending_source_names", return_value=set()),
        patch.object(
            pr,
            "reconcile_attachment_consistency",
            return_value=AttachmentAuditResult(),
        ) as attach_mock,
        patch.object(pr, "audit_attachments", return_value=AttachmentAuditResult()),
        patch.object(pr, "find_semantic_source_invariant_violations", return_value=[]),
    ):
        explicit_run_id = "provenance-rebuild:pinned-test-run"
        summary = pr.rebuild_provenance(
            gc=gc, recovery_map={}, run_id=explicit_run_id
        )

    attach_mock.assert_called_once()
    assert attach_mock.call_args.kwargs["run_id"] == explicit_run_id
    assert summary["consistency"]["attachment_reconcile"] == (
        AttachmentAuditResult().as_dict()
    )


def test_provenance_rebuild_synthesizes_a_run_id_when_none_is_given() -> None:
    """A live (non-dry-run) rebuild always attributes its own detachments.

    An operator invocation that forgets to pass ``run_id`` must not silently
    fall back to an untraceable ``run_id=None`` write.
    """
    import imas_codex.standard_names.provenance_rebuild as pr

    gc = MagicMock()
    with (
        patch.object(pr, "find_provenance_orphans", return_value=[]),
        patch.object(pr, "find_edge_scalar_desyncs", return_value=[]),
        patch.object(pr, "reattach_produced_name_edges", return_value=0),
        patch.object(pr, "_run_deterministic_fixpoints"),
        patch.object(pr, "find_orphan_parent_source_candidates", return_value=[]),
        patch.object(
            pr,
            "classify_orphan_parent_source_candidates",
            return_value={"repairable": [], "rejected_derived": []},
        ),
        patch.object(pr, "reconcile_orphan_parent_sources", return_value=0),
        patch.object(pr, "_fetch_dd_source_paths", return_value={}),
        patch.object(pr, "_fetch_change_history_sources", return_value={}),
        patch.object(pr, "_fetch_pending_source_names", return_value=set()),
        patch.object(
            pr,
            "reconcile_attachment_consistency",
            return_value=AttachmentAuditResult(),
        ) as attach_mock,
        patch.object(pr, "audit_attachments", return_value=AttachmentAuditResult()),
        patch.object(pr, "find_semantic_source_invariant_violations", return_value=[]),
    ):
        pr.rebuild_provenance(gc=gc, recovery_map={})

    attach_mock.assert_called_once()
    assert attach_mock.call_args.kwargs["run_id"]
