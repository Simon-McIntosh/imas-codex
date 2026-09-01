"""CLI decisions and scheduling for verified offsite pushes."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from subprocess import CompletedProcess

import click
from click.testing import CliRunner

from imas_codex.cli.graph.offsite import graph_offsite_push
from imas_codex.graph.neo4j_ops import OffsiteCurrency
from imas_codex.graph.offsite import OffsitePushResult


def _stale_currency() -> OffsiteCurrency:
    return OffsiteCurrency(
        status="stale",
        offsite_ref="ghcr.io/example/imas-codex-graph:old",
        offsite_modified_at=datetime(2026, 7, 7, tzinfo=UTC),
        live_path=Path("/graph/data/store"),
        live_modified_at=datetime(2026, 9, 1, tzinfo=UTC),
        age_seconds=4_838_400.0,
    )


def _common(monkeypatch) -> None:
    monkeypatch.setattr(
        "imas_codex.cli.graph.offsite.get_git_info",
        lambda: {"is_fork": True, "remote_owner": "example"},
    )
    monkeypatch.setattr(
        "imas_codex.cli.graph.offsite.get_offsite_currency",
        lambda registry, token: _stale_currency(),
    )
    monkeypatch.setattr(
        "imas_codex.cli.graph.offsite.graph_archive_stamp", lambda: "dev-abc-stamp"
    )


def test_dry_run_prints_decision_and_self_resubmitting_script(monkeypatch):
    _common(monkeypatch)
    pushed = False

    def run_cycle(**kwargs):
        nonlocal pushed
        pushed = True
        raise AssertionError(kwargs)

    monkeypatch.setattr(
        "imas_codex.cli.graph.offsite.run_offsite_push_cycle", run_cycle
    )

    result = CliRunner().invoke(graph_offsite_push, ["--dry-run"])

    assert result.exit_code == 0
    assert "56.000 days stale" in result.output
    assert "#SBATCH --job-name=codex-offsite-push" in result.output
    assert 'sbatch --begin=now+7days "$0"' in result.output
    assert "No archive was exported or pushed" in result.output
    assert not pushed


def test_schedule_uses_service_submission_helper(monkeypatch):
    _common(monkeypatch)
    submissions: list[tuple] = []
    monkeypatch.setattr(
        "imas_codex.cli.services._submit_service_job",
        lambda *args, **kwargs: submissions.append((args, kwargs)),
    )

    result = CliRunner().invoke(graph_offsite_push, ["--schedule"])

    assert result.exit_code == 0
    assert len(submissions) == 1
    args, kwargs = submissions[0]
    assert args[0] == "codex-offsite-push"
    assert "uv run --no-sync imas-codex graph offsite-push" in args[1]
    assert 'sbatch --begin=now+7days "$0"' in args[1]
    assert kwargs == {"cpus": 2, "mem": "8G"}


def test_rejected_environment_token_falls_back_to_gh_credential(monkeypatch):
    _common(monkeypatch)
    calls: list[str | None] = []

    def read_currency(registry: str, token: str | None) -> OffsiteCurrency:
        calls.append(token)
        if token == "stale-token":
            raise RuntimeError("unexpected unwrapped error")
        return _stale_currency()

    monkeypatch.setattr(
        "imas_codex.cli.graph.offsite.get_offsite_currency",
        lambda registry, token: (
            (_ for _ in ()).throw(click.ClickException("HTTP 401"))
            if token == "stale-token"
            else read_currency(registry, token)
        ),
    )
    monkeypatch.setattr(
        "imas_codex.cli.graph.offsite.subprocess.run",
        lambda *args, **kwargs: CompletedProcess(args[0], 0, "gh-token\n", ""),
    )

    result = CliRunner().invoke(
        graph_offsite_push, ["--dry-run", "--token", "stale-token"]
    )

    assert result.exit_code == 0
    assert calls == ["gh-token"]


def test_cycle_reports_success_receipt(monkeypatch, tmp_path):
    _common(monkeypatch)
    receipt = tmp_path / "receipt.json"
    monkeypatch.setattr(
        "imas_codex.cli.graph.offsite.run_offsite_push_cycle",
        lambda **kwargs: OffsitePushResult(
            outcome="pushed",
            receipt_path=receipt,
            archive_ref="ghcr.io/example/imas-codex-graph:stamp",
            archive_bytes=2_434_869_050,
            wall_time_seconds=91.25,
        ),
    )

    result = CliRunner().invoke(graph_offsite_push)

    assert result.exit_code == 0
    assert "2434869050 bytes in 91.250 s" in result.output
    assert f"Receipt: {receipt}" in result.output
