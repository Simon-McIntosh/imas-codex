"""Graph launch configuration remains independent from embedding services."""

from pathlib import Path
from types import SimpleNamespace

import click
import pytest

from imas_codex.cli.graph.data import graph_export
from imas_codex.cli.graph.server import graph_start


def test_export_restart_surfaces_compute_configuration_error(tmp_path, monkeypatch):
    """An automatic restart failure remains the export command's failure."""
    profile = SimpleNamespace(
        name="codex",
        host="localhost",
        data_dir=tmp_path / "neo4j",
    )
    profile.data_dir.mkdir()

    class RunningGraphOperation:
        def __init__(self, *_args, **_kwargs):
            self.was_running = False

        def __enter__(self):
            self.was_running = True
            return self

        def __exit__(self, *_args):
            return False

    def create_dump(_profile, dumps_dir: Path, *, verbose: bool) -> None:
        assert not verbose
        (dumps_dir / "neo4j.dump").write_bytes(b"graph dump")

    def fail_restart(**_kwargs) -> None:
        raise click.ClickException("No compute config in iter_private.yaml.")

    monkeypatch.setattr("imas_codex.graph.profiles.resolve_neo4j", lambda: profile)
    monkeypatch.setattr(
        "imas_codex.graph.remote.is_remote_location", lambda _host: False
    )
    monkeypatch.setattr("imas_codex.graph.ghcr.require_apptainer", lambda: None)
    monkeypatch.setattr(
        "imas_codex.cli.graph.data.get_git_info",
        lambda: {
            "commit": "0123456789abcdef0123456789abcdef01234567",
            "commit_short": "0123456",
            "tag": None,
        },
    )
    monkeypatch.setattr(
        "imas_codex.cli.graph.data._measure_live_graph_scope",
        lambda: {"nodes": 1, "relationships": 0, "labels": {"Graph": 1}},
    )
    monkeypatch.setattr(
        "imas_codex.cli.graph.data.Neo4jOperation", RunningGraphOperation
    )
    monkeypatch.setattr("imas_codex.cli.graph.data._run_neo4j_dump", create_dump)
    monkeypatch.setattr(
        "imas_codex.cli.graph.server.graph_start.callback", fail_restart
    )

    with pytest.raises(click.ClickException, match="No compute config"):
        graph_export.callback(
            output=str(tmp_path / "export.tar.gz"),
            no_restart=False,
            facilities=(),
            without_dd=False,
            dd_only=False,
            local=False,
            source_dump=None,
            version_label="test",
        )


def test_graph_start_does_not_read_embedding_compute_config(monkeypatch):
    """A graph-only SLURM launch resolves only graph-owned configuration."""
    profile = SimpleNamespace(
        name="codex",
        password="secret",
        host="localhost",
    )
    graph_location = SimpleNamespace(is_compute=True)
    job = {
        "job_id": "123",
        "node": "cpu-001",
        "state": "RUNNING",
        "cpus": "4",
        "time": "0:01",
    }
    submitted = {}

    def ensure_graph_job(job_name, command, **kwargs):
        submitted.update(job_name=job_name, command=command, **kwargs)
        return job

    monkeypatch.setattr("imas_codex.graph.profiles.resolve_neo4j", lambda: profile)
    monkeypatch.setattr("imas_codex.graph.profiles.get_graph_location", lambda: "cpu")
    monkeypatch.setattr(
        "imas_codex.remote.locations.resolve_location",
        lambda location: graph_location,
    )
    monkeypatch.setattr(
        "imas_codex.settings.get_embedding_location",
        lambda: pytest.fail("graph start read embedding configuration"),
    )
    monkeypatch.setattr("imas_codex.cli.services._get_neo4j_job", lambda: None)
    monkeypatch.setattr(
        "imas_codex.cli.services._neo4j_service_command", lambda: "neo4j console"
    )
    monkeypatch.setattr(
        "imas_codex.cli.services._neo4j_pre_launch", lambda: "prepare graph"
    )
    monkeypatch.setattr("imas_codex.cli.services._ensure_service_job", ensure_graph_job)
    monkeypatch.setattr("imas_codex.cli.services._graph_port", lambda: 7687)
    monkeypatch.setattr("imas_codex.cli.services._graph_http_port", lambda: 7474)
    monkeypatch.setattr(
        "imas_codex.cli.graph.server.is_neo4j_running", lambda _port: True
    )

    graph_start.callback(
        image=None,
        data_dir=None,
        password=None,
        foreground=False,
    )

    assert submitted == {
        "job_name": "codex-neo4j",
        "command": "neo4j console",
        "cpus": 4,
        "mem": "32G",
        "gpus": 0,
        "pre_launch": "prepare graph",
    }
