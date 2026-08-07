"""Regression contracts for explicit queue recovery and scoped pool plumbing."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml
from click.testing import CliRunner


def test_source_retry_schema_is_auditable() -> None:
    """The retry event and source relationship are declared in LinkML."""
    schema_path = (
        Path(__file__).parents[2] / "imas_codex" / "schemas" / "standard_name.yaml"
    )
    schema = yaml.safe_load(schema_path.read_text())
    source = schema["classes"]["StandardNameSource"]["attributes"]
    retry = schema["classes"]["StandardNameSourceRetry"]["attributes"]

    assert source["retry_events"]["range"] == "StandardNameSourceRetry"
    assert (
        source["retry_events"]["annotations"]["relationship_type"] == "HAS_RETRY_EVENT"
    )
    assert retry["previous_status"]["range"] == "StandardNameSourceStatus"
    assert retry["previous_attempt_count"]["range"] == "integer"
    assert retry["reason"]["required"] is True


def test_retry_failed_sources_records_event_before_reset() -> None:
    """Terminal and extracted-at-cap sources receive durable retry events."""
    from imas_codex.standard_names.graph_ops import retry_failed_sources

    gc = MagicMock()
    gc.query.side_effect = [
        [
            {
                "id": "dd:path/a",
                "source_path": "path/a",
                "status": "failed",
                "attempt_count": 5,
                "last_error": "grammar rejected candidate",
            },
            {
                "id": "dd:path/b",
                "source_path": "path/b",
                "status": "extracted",
                "attempt_count": 7,
                "last_error": "batch omitted source",
            },
        ],
        [
            {"source_id": "dd:path/a", "event_id": "source-retry:a"},
            {"source_id": "dd:path/b", "event_id": "source-retry:b"},
        ],
    ]

    result = retry_failed_sources(
        ["path/a", "dd:path/b"],
        reason="composer can now express both quantities",
        gc=gc,
    )

    assert result["eligible"] == 2
    assert result["retried"] == 2
    write_query = gc.query.call_args_list[1].args[0]
    assert "CREATE (event:StandardNameSourceRetry" in write_query
    assert "MERGE (sns)-[:HAS_RETRY_EVENT]->(event)" in write_query
    assert "sns.status = 'extracted'" in write_query
    assert "sns.attempt_count = 0" in write_query
    assert "sns.last_error = null" in write_query
    assert gc.query.call_args_list[1].kwargs["reason"].startswith("composer")


def test_retry_failed_sources_dry_run_never_mutates() -> None:
    """A dry run reports eligibility without writing an event or source state."""
    from imas_codex.standard_names.graph_ops import retry_failed_sources

    gc = MagicMock()
    gc.query.return_value = [
        {
            "id": "dd:path/a",
            "status": "failed",
            "attempt_count": 5,
            "last_error": "failure",
        }
    ]

    result = retry_failed_sources(
        ["path/a"], reason="inspect candidate", dry_run=True, gc=gc
    )

    assert result["eligible"] == 1
    assert result["retried"] == 0
    assert gc.query.call_count == 1


def test_retry_skipped_sources_is_fenced_and_context_neutral() -> None:
    """Skipped recovery is audited, claim-fenced, and source-context neutral."""
    from imas_codex.standard_names.graph_ops import retry_skipped_sources

    gc = MagicMock()
    gc.query.side_effect = [
        [
            {
                "id": "dd:path/a",
                "source_path": "path/a",
                "status": "skipped",
                "attempt_count": 2,
                "last_error": "prior composition failure",
                "skip_reason": "dd_unit_unresolvable",
                "skip_reason_detail": "source declaration was incomplete",
            },
            {
                "id": "dd:path/b",
                "source_path": "path/b",
                "status": "skipped",
                "attempt_count": 3,
                "last_error": None,
                "skip_reason": "dd_unit_context_dependent",
                "skip_reason_detail": "unit depends on source context",
            },
        ],
        [
            {"source_id": "dd:path/a", "event_id": "source-retry:a"},
            {"source_id": "dd:path/b", "event_id": "source-retry:b"},
        ],
    ]

    result = retry_skipped_sources(
        ["dd:path/a", "dd:path/b"],
        reason="the exact source is now nameable",
        gc=gc,
    )

    assert result["eligible"] == 2
    assert result["retried"] == 2
    assert result["refused"] == 0
    write_call = gc.query.call_args_list[1]
    write_query = write_call.args[0]
    item = write_call.kwargs["items"][0]
    assert item["previous_status"] == "skipped"
    assert item["previous_attempt_count"] == 2
    assert item["previous_error"] == (
        "prior composition failure; skip_reason=dd_unit_unresolvable; "
        "detail=source declaration was incomplete"
    )
    fallback_item = write_call.kwargs["items"][1]
    assert fallback_item["previous_error"] == (
        "skip_reason=dd_unit_context_dependent; detail=unit depends on source context"
    )
    assert "CREATE (event:StandardNameSourceRetry" in write_query
    assert "event.previous_status = item.previous_status" in write_query
    assert "event.reason = $reason" in write_query
    assert "sns.status = 'skipped'" in write_query
    assert "sns.claimed_at IS NULL" in write_query
    assert "sns.claim_token IS NULL" in write_query
    assert "sns.produced_sn_id IS NULL" in write_query
    assert "NOT (sns)-[:PRODUCED_NAME]->(:StandardName)" in write_query
    assert "sns.status = 'extracted'" in write_query
    assert "sns.attempt_count = 0" in write_query
    assert "sns.skip_reason = null" in write_query
    assert "sns.skip_reason_detail = null" in write_query
    assert "sns.last_error = null" in write_query
    assert "compose_hint" not in write_query
    assert "FROM_DD_PATH" not in write_query
    assert "DELETE" not in write_query


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ({"last_error": "bare failure;", "skip_reason": None}, "bare failure"),
        (
            {
                "last_error": "  ",
                "skip_reason": None,
                "skip_reason_detail": "orphan detail",
            },
            None,
        ),
    ],
)
def test_skipped_recovery_error_without_classification(
    source: dict[str, str | None], expected: str | None
) -> None:
    """Absent classifications preserve only substantive errors, without debris."""
    from imas_codex.standard_names.graph_ops import _skipped_recovery_error

    assert _skipped_recovery_error(source) == expected


def test_retry_skipped_sources_dry_run_has_exact_eligibility_cardinality() -> None:
    """Dry-run counts only exact, still-eligible skipped source paths."""
    from imas_codex.standard_names.graph_ops import retry_skipped_sources

    gc = MagicMock()
    gc.query.return_value = [
        {
            "id": "dd:path/a",
            "source_path": "path/a",
            "status": "skipped",
            "attempt_count": 0,
            "last_error": None,
            "skip_reason": "dd_unit_unresolvable",
            "skip_reason_detail": "",
        }
    ]

    result = retry_skipped_sources(
        ["path/a", "path/not-skipped"],
        reason="inspect exact recovery scope",
        dry_run=True,
        gc=gc,
    )

    assert result == {
        "requested": 2,
        "eligible": 1,
        "retried": 0,
        "refused": 1,
        "source_ids": ["dd:path/a"],
        "event_ids": [],
        "dry_run": True,
    }
    assert gc.query.call_count == 1
    selection_query = gc.query.call_args.args[0]
    assert "sns.id IN $requested OR sns.source_id IN $requested" in selection_query
    assert "sns.status = 'skipped'" in selection_query
    assert "sns.claimed_at IS NULL" in selection_query
    assert "sns.produced_sn_id IS NULL" in selection_query


def test_retry_skipped_sources_reports_compare_and_set_refusal() -> None:
    """A source that changes after selection is refused without collateral writes."""
    from imas_codex.standard_names.graph_ops import retry_skipped_sources

    gc = MagicMock()
    gc.query.side_effect = [
        [
            {
                "id": "dd:path/a",
                "source_path": "path/a",
                "status": "skipped",
                "attempt_count": 0,
                "last_error": None,
                "skip_reason": "dd_unit_unresolvable",
                "skip_reason_detail": "",
            }
        ],
        [],
    ]

    result = retry_skipped_sources(
        ["path/a"], reason="source declaration changed", gc=gc
    )

    assert result["eligible"] == 1
    assert result["retried"] == 0
    assert result["refused"] == 1
    assert result["source_ids"] == []
    assert result["event_ids"] == []


def test_retry_cli_reports_dry_run_eligibility() -> None:
    """The CLI exposes the release instrument without starting a pool run."""
    from imas_codex.cli.sn import sn

    result_payload = {
        "requested": 1,
        "eligible": 1,
        "retried": 0,
        "refused": 0,
        "source_ids": ["dd:path/a"],
        "event_ids": [],
        "dry_run": True,
    }
    with patch(
        "imas_codex.standard_names.graph_ops.retry_failed_sources",
        return_value=result_payload,
    ) as retry:
        result = CliRunner().invoke(
            sn,
            [
                "retry",
                "--failed",
                "--reason",
                "grammar now supports the quantity",
                "--dry-run",
                "path/a",
            ],
        )

    assert result.exit_code == 0
    assert "eligible: 1 of 1 requested source(s)" in result.output
    retry.assert_called_once_with(
        ["path/a"],
        reason="grammar now supports the quantity",
        dry_run=True,
    )


def test_retry_cli_dispatches_exact_skipped_dry_run() -> None:
    """The skipped mode dispatches only the explicitly supplied paths."""
    from imas_codex.cli.sn import sn

    result_payload = {
        "requested": 2,
        "eligible": 1,
        "retried": 0,
        "refused": 1,
        "source_ids": ["dd:path/a"],
        "event_ids": [],
        "dry_run": True,
    }
    with patch(
        "imas_codex.standard_names.graph_ops.retry_skipped_sources",
        return_value=result_payload,
    ) as retry:
        result = CliRunner().invoke(
            sn,
            [
                "retry",
                "--skipped",
                "--reason",
                "source is now nameable",
                "--dry-run",
                "dd:path/a",
                "dd:path/b",
            ],
        )

    assert result.exit_code == 0
    assert "eligible: 1 of 2 requested source(s)" in result.output
    assert "claimed, or already name-bound" in result.output
    retry.assert_called_once_with(
        ["dd:path/a", "dd:path/b"],
        reason="source is now nameable",
        dry_run=True,
    )


@pytest.mark.parametrize(
    "mode_args",
    [[], ["--failed", "--skipped"]],
)
def test_retry_cli_requires_exactly_one_recovery_mode(mode_args: list[str]) -> None:
    """Neither implicit recovery nor overlapping lifecycle modes are allowed."""
    from imas_codex.cli.sn import sn

    result = CliRunner().invoke(
        sn,
        [
            "retry",
            *mode_args,
            "--reason",
            "operator reviewed the source",
            "dd:path/a",
        ],
    )

    assert result.exit_code == 2
    assert "select exactly one retry mode: --failed or --skipped" in result.output


def test_run_retry_skipped_help_is_selection_only() -> None:
    """Run help cannot imply that its selector performs lifecycle recovery."""
    from imas_codex.cli.sn import sn

    result = CliRunner().invoke(sn, ["run", "--help"])

    assert result.exit_code == 0
    assert "Select skipped sources for run scoping only" in result.output
    assert "sn retry --skipped" in result.output


def test_source_status_reconcile_repairs_both_liveness_directions() -> None:
    """Live mappings advance stale mirrors and dead mappings re-enter compose."""
    from imas_codex.standard_names.graph_ops import reconcile_source_status_liveness

    gc = MagicMock()
    gc.query.side_effect = [
        [{"n": 4}],
        [{"n": 6, "edges": 5, "projections": 3, "source_paths": 4}],
        [{"n": 2, "projections": 1, "terminal_targets": 2}],
    ]

    result = reconcile_source_status_liveness(gc=gc, source_ids=["dd:scoped/path"])

    assert result == {
        "live_realigned": 4,
        "orphaned_reset": 6,
        "terminal_edges_dropped": 5,
        "terminal_projections_dropped": 3,
        "terminal_source_paths_dropped": 4,
        "projection_ghosts_reset": 2,
        "ghost_projections_dropped": 1,
        "ghost_source_paths_dropped": 2,
    }
    live_query = gc.query.call_args_list[0].args[0]
    orphan_query = gc.query.call_args_list[1].args[0]
    projection_query = gc.query.call_args_list[2].args[0]
    assert all(
        call.kwargs["source_ids"] == ["dd:scoped/path"]
        for call in gc.query.call_args_list
    )
    assert "$source_ids IS NULL OR sns.id IN $source_ids" in live_query
    assert "$source_ids IS NULL OR sns.id IN $source_ids" in orphan_query
    assert "$source_ids IS NULL OR sns.id IN $source_ids" in projection_query
    assert "sns.status = 'attached'" in live_query
    assert "sns.claim_token IS NULL" in live_query
    assert "hint.edit_mode, '') = 'hint'" in live_query
    assert "hint.edit_status, '') = 'open'" in live_query
    assert "hint.name_hint IS NOT NULL" in live_query
    assert "NOT EXISTS" in live_query
    assert "sns.status = 'extracted'" in orphan_query
    assert "sns.produced_sn_id = null" in orphan_query
    assert "sns.composed_at = null" in orphan_query
    assert "path <> sns.id" in orphan_query
    assert "size(dd_nodes) = 1" in orphan_query
    assert "other <> sns" in orphan_query
    assert "other_live.name_stage" in orphan_query
    assert "AND NOT backed_by_other" in orphan_query
    assert "terminal IS NULL OR backed_by_other" in orphan_query
    assert "DELETE edge" in orphan_query
    assert "DELETE terminal" not in orphan_query
    assert "HAS_STANDARD_NAME_VOCAB_GAP" not in orphan_query

    assert "sns.claim_token IS NULL" in projection_query
    assert "NOT (sns)-[:PRODUCED_NAME]->(:StandardName)" in projection_query
    assert "size(dd_nodes) = 1" in projection_query
    assert "dd_nodes[0].id = sns.source_id" in projection_query
    assert "terminal.name_stage" in projection_query
    assert "other <> sns" in projection_query
    assert "WHEN backed_by_other THEN []" in projection_query
    assert "WHEN NOT backed_by_other THEN terminal" in projection_query
    assert "path <> sns.id" in projection_query
    assert "sns.composed_at = null" in projection_query


def test_source_status_reconcile_is_idempotent_when_no_residue_matches() -> None:
    """A clean graph reports zero changes on every lifecycle repair path."""
    from imas_codex.standard_names.graph_ops import reconcile_source_status_liveness

    gc = MagicMock()
    gc.query.side_effect = [
        [{"n": 0}],
        [{"n": 0, "edges": 0, "projections": 0, "source_paths": 0}],
        [{"n": 0, "projections": 0, "terminal_targets": 0}],
    ]

    assert reconcile_source_status_liveness(gc=gc) == {
        "live_realigned": 0,
        "orphaned_reset": 0,
        "terminal_edges_dropped": 0,
        "terminal_projections_dropped": 0,
        "terminal_source_paths_dropped": 0,
        "projection_ghosts_reset": 0,
        "ghost_projections_dropped": 0,
        "ghost_source_paths_dropped": 0,
    }


def test_terminal_name_hints_are_retired() -> None:
    """Startup cleanup closes historical open edits no name pool can consume."""
    from imas_codex.standard_names.graph_ops import retire_unreachable_hint_edits

    gc = MagicMock()
    gc.query.return_value = [{"n": 9}]

    assert retire_unreachable_hint_edits(gc=gc) == 9
    query = gc.query.call_args.args[0]
    assert "sn.name_stage IN ['accepted', 'exhausted']" in query
    assert "sn.edit_status = 'rejected'" in query


@pytest.mark.parametrize("stage", ["accepted", "exhausted"])
def test_name_hint_refuses_terminal_target(stage: str) -> None:
    """A hint cannot be staged where no name-generation claim can consume it."""
    from imas_codex.standard_names.edit import _apply_hint

    gc = MagicMock()
    result = _apply_hint(
        gc,
        target="electron_density",
        target_row={"name_stage": stage, "has_successor": False},
        hint="prefer the canonical quantity spelling",
        axis="name",
        reason="clarify the quantity",
        origin="human",
        scope="only_self",
        dry_run=False,
    )

    assert result.applied is False
    assert result.blocked is not None
    assert "--rename" in result.blocked
    gc.query.assert_not_called()


def test_orphaned_run_sweep_falls_back_to_started_at() -> None:
    """A heartbeat-less run remains age-testable through its start timestamp."""
    from imas_codex.standard_names.graph_ops import (
        mark_orphaned_standard_name_runs_stale,
    )

    gc = MagicMock()
    gc.__enter__.return_value = gc
    gc.query.return_value = [{"id": "stale-run"}]

    with patch("imas_codex.standard_names.graph_ops.GraphClient", return_value=gc):
        assert mark_orphaned_standard_name_runs_stale(max_age_hours=1.0) == 1

    query = gc.query.call_args.args[0]
    assert "rr.created_at, rr.started_at, rr.stopped_at" in query


def test_pending_counts_mirror_review_and_edit_scope() -> None:
    """The aggregate counter applies claim eligibility and edits-only scope."""
    from imas_codex.standard_names.graph_ops import pool_pending_counts

    row = {
        "generate_name": 0,
        "review_name": 2,
        "refine_name": 1,
        "generate_docs": 3,
        "review_docs": 4,
        "refine_docs": 5,
    }
    gc = MagicMock()
    gc.__enter__.return_value = gc
    gc.query.return_value = [row]

    with patch("imas_codex.graph.client.GraphClient", return_value=gc):
        result = pool_pending_counts(
            scope_run_id="edit-scope",
            edits_only=True,
            domain="magnetics",
        )

    assert result == row
    query = gc.query.call_args.args[0]
    assert "sn.description <> $parent_desc_placeholder" in query
    assert "coalesce(sn.origin, '') <> 'derived'" in query
    assert "coalesce(sn.edit_status, '') = 'open'" in query
    assert "AND false" in query
    assert gc.query.call_args.kwargs["scope_run_id"] == "edit-scope"


def test_cli_pending_progress_uses_the_same_edit_scope() -> None:
    """The watchdog/display query cannot see global draft work under --edits."""
    from imas_codex.cli.sn import _compute_pool_progress

    gc = MagicMock()
    pools = (
        "generate_name",
        "enrich_parents",
        "review_name",
        "refine_name",
        "generate_docs",
        "review_docs",
        "refine_docs",
    )
    gc.query.return_value = [
        {key: 0 for pool in pools for key in (pool, f"{pool}_done")}
    ]

    _compute_pool_progress(
        gc,
        domains=["magnetics"],
        rotation_cap=3,
        min_score=0.75,
        edits_only=True,
    )

    query = gc.query.call_args.args[0]
    assert "sn.description <> $parent_desc_placeholder" in query
    assert "coalesce(sn.edit_status, '') = 'open'" in query
    assert "AND false" in query
    assert "coalesce(s.attempt_count, 0) < $max_compose_attempts" in query


@pytest.mark.asyncio
async def test_generate_processor_forwards_compose_model() -> None:
    """The pooled processor passes the configured override into compose."""
    from imas_codex.standard_names import workers

    compose = AsyncMock(return_value=1)
    with patch.object(workers, "compose_batch", new=compose):
        result = await workers.process_generate_name_batch(
            [{"id": "dd:path"}],
            MagicMock(),
            asyncio.Event(),
            compose_model="configured-compose-seat",
        )

    assert result == 1
    assert compose.await_args.kwargs["compose_model"] == "configured-compose-seat"


@pytest.mark.asyncio
async def test_pool_spec_carries_compose_model_to_processor() -> None:
    """The orchestrator adapter retains the CLI override in pooled mode."""
    from imas_codex.standard_names import loop, workers

    processor = AsyncMock(return_value=1)
    with (
        patch.object(workers, "process_generate_name_batch", new=processor),
        patch("imas_codex.settings.get_pool_replicas", return_value=1),
    ):
        specs = loop._build_pool_specs(
            MagicMock(),
            asyncio.Event(),
            compose_model="configured-compose-seat",
        )
        generate = next(spec for spec in specs if spec.name == "generate_name")
        await generate.process({"items": [{"id": "dd:path", "claim_token": "token"}]})

    assert processor.await_args.kwargs["compose_model"] == "configured-compose-seat"
