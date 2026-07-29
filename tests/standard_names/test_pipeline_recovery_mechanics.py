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


def test_source_status_reconcile_repairs_both_liveness_directions() -> None:
    """Live mappings advance stale mirrors and dead mappings re-enter compose."""
    from imas_codex.standard_names.graph_ops import reconcile_source_status_liveness

    gc = MagicMock()
    gc.query.side_effect = [[{"n": 4}], [{"n": 6, "edges": 5}]]

    result = reconcile_source_status_liveness(gc=gc)

    assert result == {
        "live_realigned": 4,
        "orphaned_reset": 6,
        "terminal_edges_dropped": 5,
    }
    live_query = gc.query.call_args_list[0].args[0]
    orphan_query = gc.query.call_args_list[1].args[0]
    assert "sns.status = 'attached'" in live_query
    assert "sns.status = 'extracted'" in orphan_query
    assert "sns.produced_sn_id = null" in orphan_query


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
    gc.query.return_value = [{}]

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
