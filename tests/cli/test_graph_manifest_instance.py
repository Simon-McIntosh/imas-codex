"""Graph load bookkeeping follows the graph instance identity."""

from __future__ import annotations

from types import SimpleNamespace

from click.testing import CliRunner

from imas_codex.cli.graph import server
from imas_codex.graph import ghcr
from imas_codex.graph.neo4j_ops import save_graph_instance_manifest


def test_status_reports_load_source_only_for_loaded_instance(tmp_path, monkeypatch):
    manifest_path = tmp_path / "graph-manifest.json"
    monkeypatch.setattr(ghcr, "LOCAL_GRAPH_MANIFEST", manifest_path)
    ghcr.save_local_graph_manifest(
        {
            "version": "registry-copy",
            "pushed": True,
            "loaded_from": "legacy-global-load.tar.gz",
        }
    )
    save_graph_instance_manifest(
        "instance-a",
        {
            "version": "restored-copy",
            "pushed": False,
            "loaded_from": "instance-a-load.tar.gz",
        },
    )

    active_instance = {"name": "instance-b"}

    def resolve_profile(**_kwargs):
        return SimpleNamespace(
            name=active_instance["name"],
            host="localhost",
            location="iter",
            uri="bolt://localhost:7687",
            data_dir=tmp_path / active_instance["name"],
            bolt_port=7687,
            http_port=7474,
        )

    monkeypatch.setattr(
        server,
        "get_git_info",
        lambda: {
            "commit_short": "abcdef0",
            "tag": None,
            "is_fork": True,
        },
    )
    monkeypatch.setattr(server, "get_registry", lambda *_args: "ghcr.example")
    monkeypatch.setattr(
        server,
        "get_backup_currency",
        lambda: SimpleNamespace(
            backup_path=None,
            live_path=None,
            age_seconds=None,
        ),
    )
    monkeypatch.setattr(
        server,
        "get_offsite_currency",
        lambda *_args: SimpleNamespace(offsite_ref=None),
    )
    monkeypatch.setattr(server, "is_neo4j_running", lambda *_args: False)
    monkeypatch.setattr("imas_codex.graph.profiles.resolve_neo4j", resolve_profile)
    monkeypatch.setattr("imas_codex.remote.executor.is_local_host", lambda _host: True)
    monkeypatch.setattr("imas_codex.cli.services._get_neo4j_job", lambda: None)

    other_status = CliRunner().invoke(server.graph_status)

    assert other_status.exit_code == 0, other_status.output
    assert "Loaded from:" not in other_status.output
    assert "legacy-global-load.tar.gz" not in other_status.output

    active_instance["name"] = "instance-a"
    loaded_status = CliRunner().invoke(server.graph_status)

    assert loaded_status.exit_code == 0, loaded_status.output
    assert "Loaded from: instance-a-load.tar.gz" in loaded_status.output
    assert "Version: restored-copy" in loaded_status.output
