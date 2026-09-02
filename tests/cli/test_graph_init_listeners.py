"""Listener configuration for graph initialization and restored graphs."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from click.testing import CliRunner

from imas_codex.cli import services
from imas_codex.cli.graph import data
from imas_codex.graph import dirs


def _listener_settings(config: str) -> dict[str, str]:
    return {
        key: value
        for line in config.splitlines()
        if line.startswith("server.") and "listen_address=" in line
        for key, value in [line.split("=", maxsplit=1)]
    }


def test_new_and_restored_graphs_use_compute_service_listeners(tmp_path, monkeypatch):
    start_after_switch = data._start_neo4j_after_switch
    profile = SimpleNamespace(
        name="scratch",
        host="localhost",
        data_dir=tmp_path / "active",
        bolt_port=17687,
        http_port=17474,
    )
    graph_store = tmp_path / "graphs"
    active_link = tmp_path / "active"

    monkeypatch.setattr(dirs, "GRAPH_STORE", graph_store)
    monkeypatch.setattr(dirs, "ACTIVE_LINK", active_link)
    monkeypatch.setattr(
        "imas_codex.graph.profiles.resolve_neo4j", lambda **_kwargs: profile
    )
    monkeypatch.setattr(
        "imas_codex.graph.remote.is_remote_location", lambda _host: False
    )
    monkeypatch.setattr(data, "is_neo4j_running", lambda _port: False)
    monkeypatch.setattr(data, "_start_neo4j_after_switch", lambda _profile: None)

    result = CliRunner().invoke(data.graph_init, ["scratch"])

    assert result.exit_code == 0, result.output
    initialized_config = (graph_store / "scratch" / "conf" / "neo4j.conf").read_text()

    restored_dir = tmp_path / "restored"
    restored_profile = SimpleNamespace(
        name="restored",
        data_dir=restored_dir,
        bolt_port=profile.bolt_port,
        http_port=profile.http_port,
    )
    image = tmp_path / "neo4j.sif"
    image.touch()
    monkeypatch.setattr(data, "NEO4J_IMAGE", image)
    monkeypatch.setattr(data, "secure_data_directory", lambda _path: None)
    monkeypatch.setattr(data, "is_neo4j_running", lambda _port: True)
    monkeypatch.setattr(
        data.subprocess,
        "Popen",
        lambda *_args, **_kwargs: SimpleNamespace(pid=12345),
    )

    start_after_switch(restored_profile)
    restored_config = (restored_dir / "conf" / "neo4j.conf").read_text()
    service_config = services._neo4j_pre_launch()

    expected = {
        "server.default_listen_address": "0.0.0.0",
        "server.bolt.listen_address": f":{profile.bolt_port}",
        "server.http.listen_address": f":{profile.http_port}",
    }
    assert _listener_settings(initialized_config) == expected
    assert _listener_settings(restored_config) == expected
    assert _listener_settings(service_config) == expected
    assert "127.0.0.1" not in initialized_config
    assert "127.0.0.1" not in restored_config
