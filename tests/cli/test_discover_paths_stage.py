"""The paths command and stage preserve seed, drain, and item scope."""

from __future__ import annotations

import asyncio
import importlib
from dataclasses import FrozenInstanceError
from unittest.mock import patch

import click
import pytest
from click.testing import CliRunner

from imas_codex.cli.discover.paths import PathsStageOptions, paths, run_paths_stage


@pytest.fixture
def engine(monkeypatch: pytest.MonkeyPatch) -> dict:
    captured: dict = {}

    async def fake_engine(**kwargs):
        captured.update(kwargs)
        return {
            "scanned": 2,
            "scored": 0,
            "cost": 0.0,
            "elapsed_seconds": 0.1,
        }

    def fake_run_discovery(config, async_main, *, on_complete=None):
        result = asyncio.run(async_main(asyncio.Event(), None))
        if on_complete:
            on_complete(result)
        return result

    monkeypatch.setattr(
        "imas_codex.discovery.paths.parallel.run_parallel_discovery", fake_engine
    )
    monkeypatch.setattr(
        "imas_codex.cli.discover.common.run_discovery", fake_run_discovery
    )
    monkeypatch.setattr("imas_codex.cli.discover.common.use_rich_output", lambda: False)
    monkeypatch.setattr(
        "imas_codex.cli.discover.common.setup_logging", lambda *a, **kw: None
    )
    monkeypatch.setattr(
        "imas_codex.cli.discover.common.make_log_print",
        lambda *a, **kw: lambda *a: None,
    )
    monkeypatch.setattr(
        "imas_codex.discovery.get_discovery_stats",
        lambda facility: {"total": 4, "scanned": 2, "scored": 1},
    )
    monkeypatch.setattr("imas_codex.discovery.seed_facility_roots", lambda *a, **kw: 0)
    monkeypatch.setattr(
        "imas_codex.discovery.base.facility.get_facility", lambda facility: {}
    )
    monkeypatch.setattr("imas_codex.settings.get_model", lambda section: "test-model")
    monkeypatch.setattr(
        importlib.import_module("imas_codex.cli.discover.paths"),
        "_print_discovery_summary",
        lambda *a, **kw: None,
    )
    return captured


def test_options_are_frozen() -> None:
    with pytest.raises(FrozenInstanceError):
        PathsStageOptions().flush = True  # type: ignore[misc]


def test_scan_only_runs_only_seeding_workers(engine: dict) -> None:
    run_paths_stage("jet", PathsStageOptions(scan_only=True))
    assert engine["num_scan_workers"] == 1
    assert engine["num_expand_workers"] == 1
    assert engine["num_triage_workers"] == 0
    assert engine["num_enrich_workers"] == 0
    assert engine["num_score_workers"] == 0


def test_flush_runs_only_draining_workers(engine: dict) -> None:
    run_paths_stage("jet", PathsStageOptions(flush=True))
    assert engine["num_scan_workers"] == 0
    assert engine["num_expand_workers"] == 0
    assert engine["num_triage_workers"] == 2
    assert engine["num_enrich_workers"] == 2
    assert engine["num_score_workers"] == 1


def test_topic_reaches_scorer_and_limit_caps_items(engine: dict) -> None:
    run_paths_stage("jet", PathsStageOptions(topic="equilibrium", limit=7))
    assert engine["focus"] == "equilibrium"
    assert engine["path_limit"] == 7


def test_focus_restricts_root_filter(engine: dict) -> None:
    run_paths_stage("jet", PathsStageOptions(focus=("/work/equilibrium",)))
    assert engine["root_filter"] == ["/work/equilibrium"]


def test_flush_with_focus_does_not_seed_roots(engine: dict, monkeypatch) -> None:
    def reject_seed(*args, **kwargs):
        pytest.fail("flush seeded a root")

    monkeypatch.setattr("imas_codex.discovery.seed_facility_roots", reject_seed)
    run_paths_stage("jet", PathsStageOptions(flush=True, focus=("/work/equilibrium",)))
    assert engine["root_filter"] == ["/work/equilibrium"]


def test_flush_refuses_add_roots() -> None:
    with pytest.raises(click.UsageError, match="mutually exclusive"):
        run_paths_stage("jet", PathsStageOptions(flush=True, add_roots=True))


def test_focus_manifest_restricts_root_filter(engine: dict, tmp_path) -> None:
    manifest = tmp_path / "paths.txt"
    manifest.write_text("# paths\n/work/equilibrium\n/work/magnetics\n")
    run_paths_stage("jet", PathsStageOptions(focus=(str(manifest),)))
    assert engine["root_filter"] == ["/work/equilibrium", "/work/magnetics"]


def test_default_threshold_is_the_calibrated_scan_gate(
    engine: dict, monkeypatch
) -> None:
    monkeypatch.setattr("imas_codex.settings.get_path_scan_threshold", lambda: 0.42)
    run_paths_stage("jet", PathsStageOptions())
    assert engine["threshold"] == 0.42


def test_command_builds_options_and_calls_stage() -> None:
    with patch("imas_codex.cli.discover.paths.run_paths_stage") as stage:
        result = CliRunner().invoke(
            paths,
            [
                "jet",
                "--flush",
                "--topic",
                "equilibrium",
                "--limit",
                "7",
                "--focus",
                "/work/equilibrium",
            ],
        )
    assert result.exit_code == 0, result.output
    facility, options = stage.call_args.args
    assert facility == "jet"
    assert isinstance(options, PathsStageOptions)
    assert options.flush is True
    assert options.topic == "equilibrium"
    assert options.limit == 7
    assert options.focus == ("/work/equilibrium",)


def test_scan_only_and_flush_are_mutually_exclusive() -> None:
    with pytest.raises(click.UsageError, match="mutually exclusive"):
        run_paths_stage("jet", PathsStageOptions(scan_only=True, flush=True))
