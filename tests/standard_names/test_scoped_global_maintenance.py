"""Scoped pool runs can bypass every graph-wide maintenance writer."""

from __future__ import annotations

import asyncio
from contextlib import ExitStack
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from click.testing import CliRunner

from imas_codex.cli.sn import sn
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


def _maintenance_mocks(stack: ExitStack) -> dict[str, MagicMock]:
    """Patch the complete graph-wide maintenance call set used by the loop."""
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
    mocks = {
        name: stack.enter_context(patch(f"{_GO}.{name}", return_value=result))
        for name, result in specs.items()
    }
    mocks["reconcile_dd_unit_corrections"] = stack.enter_context(
        patch(
            "imas_codex.graph.dd_graph_ops.reconcile_dd_unit_corrections",
            return_value={},
        )
    )
    mocks["reconcile_attachment_consistency"] = stack.enter_context(
        patch(
            "imas_codex.standard_names.attachment_audit."
            "reconcile_attachment_consistency",
            return_value=AttachmentAuditResult(),
        )
    )
    mocks["refresh_drifted_sources"] = stack.enter_context(
        patch(
            "imas_codex.standard_names.source_refresh.refresh_drifted_sources",
            return_value={},
        )
    )
    mocks["restamp_harmonized_families"] = stack.enter_context(
        patch(
            "imas_codex.standard_names.harmonize.restamp_harmonized_families",
            return_value={},
        )
    )
    mocks["run_orphan_sweep_loop"] = stack.enter_context(
        patch(
            "imas_codex.standard_names.orphan_sweep.run_orphan_sweep_loop",
            new_callable=AsyncMock,
        )
    )
    mocks["embed_description_worker"] = stack.enter_context(
        patch(
            "imas_codex.discovery.base.embed_worker.embed_description_worker",
            new_callable=AsyncMock,
        )
    )
    return mocks


async def _run_loop(*, skip_global_maintenance: bool):
    """Run the orchestrator with graph and worker boundaries mocked."""
    graph_context, _ = _graph_context()
    with ExitStack() as stack:
        maintenance = _maintenance_mocks(stack)
        create_run = stack.enter_context(patch(f"{_GO}.create_sn_run_open"))
        finalize_run = stack.enter_context(patch(f"{_GO}.finalize_sn_run"))
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
        build_specs = stack.enter_context(
            patch(f"{_LOOP}._build_pool_specs", return_value=[])
        )
        run_pools = stack.enter_context(
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
                "imas_codex.standard_names.budget.BudgetManager._get_total_spent_sync",
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
            skip_global_maintenance=skip_global_maintenance,
        )

    return summary, maintenance, create_run, finalize_run, build_specs, run_pools


@pytest.mark.asyncio
async def test_scoped_run_bypasses_complete_global_maintenance_set() -> None:
    result = await _run_loop(skip_global_maintenance=True)
    summary, maintenance, create_run, finalize_run, build_specs, run_pools = result

    for writer in maintenance.values():
        writer.assert_not_called()
    create_run.assert_called_once()
    finalize_run.assert_called_once()
    build_specs.assert_called_once()
    assert build_specs.call_args.kwargs["scope_run_id"] == "bounded-run"
    run_pools.assert_awaited_once()
    assert summary.run_id


@pytest.mark.asyncio
async def test_ordinary_scoped_run_keeps_global_maintenance() -> None:
    result = await _run_loop(skip_global_maintenance=False)
    _, maintenance, create_run, finalize_run, build_specs, run_pools = result

    for writer in maintenance.values():
        writer.assert_called()
    create_run.assert_called_once()
    finalize_run.assert_called_once()
    build_specs.assert_called_once()
    run_pools.assert_awaited_once()


@pytest.mark.asyncio
async def test_orchestrator_rejects_unbounded_maintenance_bypass() -> None:
    from imas_codex.standard_names.loop import run_sn_pools

    with pytest.raises(ValueError, match="requires scope_run_id"):
        await run_sn_pools(cost_limit=1.0, skip_global_maintenance=True)


def test_cli_help_documents_scoped_maintenance_bypass() -> None:
    result = CliRunner().invoke(sn, ["run", "--help"])
    normalized_help = " ".join(result.output.split())

    assert result.exit_code == 0
    assert "--skip-global-maintenance" in result.output
    assert "--focus/--batch or --scope-run-id" in normalized_help


@pytest.mark.parametrize(
    "args, message",
    [
        (["--skip-global-maintenance"], "requires --focus/--batch"),
        (
            ["--scope-run-id", "bounded", "--reseed", "--skip-global-maintenance"],
            "cannot be combined with --reseed",
        ),
        (
            [
                "--focus",
                "equilibrium/time_slice/profiles_1d/psi",
                "--reset-to",
                "drafted",
                "--skip-global-maintenance",
            ],
            "cannot be combined with --reset-to/--reset-only",
        ),
        (
            [
                "--scope-run-id",
                "bounded",
                "--only",
                "reconcile",
                "--skip-global-maintenance",
            ],
            "cannot be combined with --only reconcile",
        ),
        (
            [
                "--focus",
                "equilibrium/time_slice/profiles_1d/psi",
                "--revalidate",
                "--skip-global-maintenance",
            ],
            "cannot be combined with --revalidate",
        ),
        (
            [
                "--scope-run-id",
                "bounded",
                "--revalidate",
                "--skip-global-maintenance",
            ],
            "cannot be combined with --revalidate",
        ),
        (
            [
                "--source",
                "signals",
                "--scope-run-id",
                "bounded",
                "--skip-global-maintenance",
            ],
            "requires the DD pool orchestrator",
        ),
    ],
)
def test_cli_rejects_unsafe_maintenance_bypass_combinations(
    args: list[str], message: str
) -> None:
    result = CliRunner().invoke(sn, ["run", *args])

    assert result.exit_code == 2
    assert message in result.output


def test_cli_dry_run_previews_scope_without_graph_writes() -> None:
    with (
        patch(
            "imas_codex.graph.client.GraphClient",
            side_effect=AssertionError("graph opened"),
        ),
        patch("imas_codex.cli.sn._run_sn_cmd") as run_cmd,
    ):
        result = CliRunner().invoke(
            sn,
            [
                "run",
                "--focus",
                "equilibrium/time_slice/profiles_1d/psi",
                "--skip-global-maintenance",
                "--dry-run",
            ],
        )

    assert result.exit_code == 0
    assert "no graph writes performed" in result.output
    run_cmd.assert_not_called()


def test_cli_wires_bypass_to_existing_pool_orchestrator() -> None:
    with (
        patch("imas_codex.cli.sn._require_embed_ready"),
        patch("imas_codex.cli.sn._auto_sync_grammar") as grammar_sync,
        patch("imas_codex.cli.sn._note_pipeline_version_drift") as drift_note,
        patch("imas_codex.cli.sn._run_sn_cmd", return_value={}) as run_cmd,
    ):
        result = CliRunner().invoke(
            sn,
            [
                "run",
                "--scope-run-id",
                "bounded",
                "--skip-global-maintenance",
                "--quiet",
            ],
        )

    assert result.exit_code == 0
    grammar_sync.assert_not_called()
    drift_note.assert_not_called()
    run_cmd.assert_called_once()
    assert run_cmd.call_args.kwargs["scope_run_id"] == "bounded"
    assert run_cmd.call_args.kwargs["skip_global_maintenance"] is True


def test_focused_bypass_seeds_exact_scope_without_global_run_id_clear() -> None:
    graph_context, graph = _graph_context()
    focus_path = "equilibrium/time_slice/profiles_1d/psi"
    with (
        patch("imas_codex.cli.sn._require_embed_ready"),
        patch("imas_codex.cli.sn._auto_sync_grammar") as grammar_sync,
        patch("imas_codex.cli.sn._note_pipeline_version_drift") as drift_note,
        patch("imas_codex.cli.sn._run_sn_cmd", return_value={}) as run_cmd,
        patch("imas_codex.graph.client.GraphClient", return_value=graph_context),
        patch(
            f"{_GO}.partition_focus_by_accepted",
            return_value=([focus_path], []),
        ),
        patch(f"{_GO}.merge_standard_name_sources", return_value=1) as merge_sources,
        patch(f"{_GO}.scope_focus_names") as scope_names,
        patch("imas_codex.settings.get_dd_version", return_value="4.0.0"),
    ):
        result = CliRunner().invoke(
            sn,
            [
                "run",
                "--focus",
                focus_path,
                "--skip-global-maintenance",
                "--quiet",
            ],
        )

    assert result.exit_code == 0
    grammar_sync.assert_not_called()
    drift_note.assert_not_called()
    merge_sources.assert_called_once()
    scope_names.assert_called_once()
    run_cmd.assert_called_once()
    assert run_cmd.call_args.kwargs["skip_global_maintenance"] is True
    queries = [call.args[0] for call in graph.query.call_args_list if call.args]
    assert not any("WHERE sn.run_id IS NOT NULL" in query for query in queries)


def test_ordinary_cli_run_keeps_global_startup() -> None:
    with (
        patch("imas_codex.cli.sn._require_embed_ready"),
        patch("imas_codex.cli.sn._auto_sync_grammar") as grammar_sync,
        patch("imas_codex.cli.sn._note_pipeline_version_drift") as drift_note,
        patch("imas_codex.cli.sn._run_sn_cmd", return_value={}) as run_cmd,
    ):
        result = CliRunner().invoke(
            sn,
            ["run", "--scope-run-id", "bounded", "--quiet"],
        )

    assert result.exit_code == 0
    grammar_sync.assert_called_once()
    drift_note.assert_called_once()
    assert run_cmd.call_args.kwargs["skip_global_maintenance"] is False
