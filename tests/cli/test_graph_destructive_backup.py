"""Regression tests for recoverable graph replacement."""

from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import click
from click.testing import CliRunner

from imas_codex.cli.graph import data, registry


def _local_profile(tmp_path):
    return SimpleNamespace(
        host="localhost",
        name="test",
        data_dir=tmp_path / "neo4j",
    )


def test_load_aborts_before_archive_replacement_when_dump_fails(tmp_path, monkeypatch):
    profile = _local_profile(tmp_path)
    archive = tmp_path / "graph.tar.gz"
    archive.touch()
    dump = Mock(side_effect=click.ClickException("dump failed"))
    continued = Mock(side_effect=AssertionError("archive replacement continued"))
    operation = SimpleNamespace(was_running=False)

    monkeypatch.setattr("imas_codex.graph.profiles.resolve_neo4j", lambda: profile)
    monkeypatch.setattr("imas_codex.settings.get_graph_password", lambda: "secret")
    monkeypatch.setattr(
        "imas_codex.graph.remote.is_remote_location", lambda _host: False
    )
    monkeypatch.setattr("imas_codex.graph.ghcr.require_apptainer", lambda: None)
    monkeypatch.setattr(
        data, "Neo4jOperation", lambda *args, **kwargs: nullcontext(operation)
    )
    monkeypatch.setattr(data, "backup_graph_dump", dump, raising=False)
    monkeypatch.setattr(data.tarfile, "open", continued)

    result = CliRunner().invoke(data.graph_load, [str(archive), "--force"])

    assert result.exit_code != 0
    assert "dump failed" in result.output
    dump.assert_called_once_with()
    continued.assert_not_called()


def test_pull_aborts_before_registry_fetch_when_dump_fails(tmp_path, monkeypatch):
    profile = _local_profile(tmp_path)
    dump = Mock(side_effect=click.ClickException("dump failed"))
    continued = Mock(side_effect=AssertionError("registry fetch continued"))
    operation = SimpleNamespace(was_running=False)

    monkeypatch.setattr("imas_codex.graph.profiles.resolve_neo4j", lambda: profile)
    monkeypatch.setattr(
        "imas_codex.graph.remote.is_remote_location", lambda _host: False
    )
    monkeypatch.setattr(registry, "get_git_info", lambda: {})
    monkeypatch.setattr(registry, "get_registry", lambda *_args: "registry.example")
    monkeypatch.setattr(registry, "get_package_name", lambda *_args, **_kwargs: "graph")
    monkeypatch.setattr(registry, "require_oras", lambda: None)
    monkeypatch.setattr(
        registry,
        "Neo4jOperation",
        lambda *args, **kwargs: nullcontext(operation),
        raising=False,
    )
    monkeypatch.setattr(registry, "backup_graph_dump", dump, raising=False)
    monkeypatch.setattr(
        "imas_codex.cli.graph_progress.GraphProgress",
        continued,
    )

    result = CliRunner().invoke(
        registry.graph_pull,
        ["--version", "test", "--force"],
    )

    assert result.exit_code != 0
    assert "dump failed" in result.output
    dump.assert_called_once_with()
    continued.assert_not_called()
