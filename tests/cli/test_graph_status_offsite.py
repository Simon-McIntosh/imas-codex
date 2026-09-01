"""Graph status reporting for the full-scope registry recovery point."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

from click.testing import CliRunner

from imas_codex.cli.graph.server import graph_status
from imas_codex.graph.neo4j_ops import BackupCurrency, OffsiteCurrency


def test_graph_status_names_offsite_copy_and_lag(monkeypatch):
    local = BackupCurrency(
        status="current",
        backup_path=Path("/backups/neo4j.dump"),
        backup_modified_at=datetime(2026, 9, 1, tzinfo=UTC),
        live_path=Path("/data/store"),
        live_modified_at=datetime(2026, 9, 1, tzinfo=UTC),
        age_seconds=0.0,
    )
    offsite = OffsiteCurrency(
        status="stale",
        offsite_ref="ghcr.io/example/imas-codex-graph:v5.3.0-rc6",
        offsite_modified_at=datetime(2026, 7, 7, 12, 27, 23, tzinfo=UTC),
        live_path=Path("/data/store"),
        live_modified_at=datetime(2026, 9, 1, 18, 10, 6, tzinfo=UTC),
        age_seconds=4_859_563.0,
    )
    monkeypatch.setattr(
        "imas_codex.cli.graph.server.get_git_info",
        lambda: {
            "commit_short": "abc1234",
            "tag": None,
            "is_fork": True,
            "remote_owner": "example",
        },
    )
    monkeypatch.setattr(
        "imas_codex.cli.graph.server.get_local_graph_manifest", lambda: None
    )
    monkeypatch.setattr(
        "imas_codex.cli.graph.server.get_backup_currency", lambda: local
    )
    monkeypatch.setattr(
        "imas_codex.cli.graph.server.get_offsite_currency",
        lambda registry, token: offsite,
    )
    monkeypatch.setattr(
        "imas_codex.cli.graph.server.is_neo4j_running", lambda *args: False
    )
    monkeypatch.setattr(
        "imas_codex.graph.profiles.resolve_neo4j",
        lambda: SimpleNamespace(
            host=None,
            name="iter",
            location="iter",
            data_dir=Path("/data"),
            bolt_port=7687,
            http_port=7474,
        ),
    )

    result = CliRunner().invoke(graph_status, ["--registry", "ghcr.io/example"])

    assert result.exit_code == 0
    assert "imas-codex-graph:v5.3.0-rc6" in result.output
    assert "Offsite behind live data: 4859563 s (stale)" in result.output
