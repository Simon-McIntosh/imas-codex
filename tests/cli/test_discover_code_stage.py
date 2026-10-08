"""Exercise the code discovery stage and its command adapter."""

from __future__ import annotations

import asyncio
import dataclasses
import importlib
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

from imas_codex.cli.discover.code import CodeStageOptions, code, run_code_stage

_CODE_MODULE = importlib.import_module("imas_codex.cli.discover.code")
FACILITY = "tcv"


@pytest.fixture
def stage_env(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "imas_codex.discovery.base.facility.get_facility",
        lambda name: {"ssh_host": "facility.example.org"},
    )
    monkeypatch.setattr(
        "imas_codex.cli.discover.common.ensure_remote_environment", lambda _: None
    )
    monkeypatch.setattr("imas_codex.cli.discover.common.use_rich_output", lambda: False)
    monkeypatch.setattr(
        "imas_codex.cli.discover.common.setup_logging", lambda *a, **k: None
    )
    monkeypatch.setattr(
        "imas_codex.cli.discover.common.make_log_print",
        lambda *a, **k: lambda *args: None,
    )
    monkeypatch.setattr(
        "imas_codex.cli.discover.common.DiscoveryConfig",
        lambda **kwargs: SimpleNamespace(**kwargs),
    )
    monkeypatch.setattr("imas_codex.settings.get_path_scan_threshold", lambda: 0.37)
    monkeypatch.setattr("imas_codex.settings.get_discovery_threshold", lambda: 0.9)

    async def fake_engine(**kwargs):
        calls.append(kwargs)
        return {}

    def fake_run_discovery(_config, async_main):
        return asyncio.run(async_main(asyncio.Event(), None))

    monkeypatch.setattr(
        "imas_codex.discovery.code.parallel.run_parallel_code_discovery",
        fake_engine,
    )
    monkeypatch.setattr(
        "imas_codex.cli.discover.common.run_discovery", fake_run_discovery
    )
    return calls


def test_options_are_frozen() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        CodeStageOptions().limit = 1  # type: ignore[misc]


def test_command_builds_options_and_calls_the_stage(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(
        _CODE_MODULE,
        "run_code_stage",
        lambda facility, options: calls.append((facility, options)),
    )

    result = CliRunner().invoke(
        code,
        [
            FACILITY,
            "--focus",
            "/analysis/src",
            "--topic",
            "equilibrium",
            "--limit",
            "3",
        ],
    )

    assert result.exit_code == 0, result.output
    assert calls == [
        (
            FACILITY,
            CodeStageOptions(focus=("/analysis/src",), topic="equilibrium", limit=3),
        )
    ]


def test_scan_only_runs_the_seeding_workers(stage_env) -> None:
    run_code_stage(FACILITY, CodeStageOptions(scan_only=True))

    assert len(stage_env) == 1
    call = stage_env[0]
    assert call["scan_only"] is True
    assert call["num_scan_workers"] == 2
    assert call["score_only"] is False


def test_flush_runs_only_the_draining_workers(stage_env) -> None:
    result = CliRunner().invoke(code, [FACILITY, "--flush"])

    assert result.exit_code == 0, result.output
    assert len(stage_env) == 1
    call = stage_env[0]
    assert call["num_scan_workers"] == 0
    assert call["num_triage_workers"] > 0
    assert call["num_enrich_workers"] > 0
    assert call["num_score_workers"] > 0
    assert call["num_code_workers"] > 0
    assert call["scan_only"] is False
    assert call["score_only"] is False


def test_scan_only_and_flush_are_refused_before_the_engine(stage_env) -> None:
    result = CliRunner().invoke(code, [FACILITY, "--scan-only", "--flush"])

    assert result.exit_code != 0
    assert stage_env == []


def test_topic_limit_focus_and_path_gate_reach_the_engine(stage_env) -> None:
    result = CliRunner().invoke(
        code,
        [
            FACILITY,
            "--focus",
            "/analysis/src/one",
            "--focus",
            "/analysis/src/two",
            "--topic",
            "equilibrium",
            "--limit",
            "4",
        ],
    )

    assert result.exit_code == 0, result.output
    assert len(stage_env) == 1
    call = stage_env[0]
    assert call["focus"] == "equilibrium"
    assert call["max_paths"] == 4
    assert call["path_prefixes"] == ["/analysis/src/one", "/analysis/src/two"]
    assert call["min_score"] == 0.37


def test_command_help_shows_the_settled_spellings() -> None:
    result = CliRunner().invoke(code, ["--help"])

    assert result.exit_code == 0, result.output
    for option in ("--scan-only", "--flush", "--topic", "--limit", "--focus"):
        assert option in result.output
    assert "--path-prefix" not in result.output
