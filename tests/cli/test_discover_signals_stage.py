"""The signals discovery stage function and its thin click wrapper.

``run_signals_stage`` carries the discovery body behind a frozen
:class:`SignalsStageOptions`; the ``signals`` click command builds those
options and calls the stage. These tests measure the settled surface:

- ``--scan-only`` selects the seeding half: the engine gets ``discover_only``.
- ``--flush`` selects the draining half: the engine gets ``enrich_only``.
- ``--topic`` is the free-text steer the old free-text ``--focus`` reached.
- ``--limit`` caps items (the old ``--signal-limit``).
- ``--focus ITEMS`` is refused with a message stating the mechanism: the
  signals claim query takes no item filter.

The engine is replaced at its single entry point
(``run_parallel_data_discovery``) so the kwargs it receives are the subject of
the assertion. ``run_discovery`` is replaced with one that drives
``async_main`` directly, keeping the measurement off the rich/plain harness.
"""

from __future__ import annotations

import asyncio
from dataclasses import FrozenInstanceError
from unittest.mock import MagicMock, patch

import click
import pytest
from click.testing import CliRunner

from imas_codex.cli.discover.signals import (
    SignalsStageOptions,
    run_signals_stage,
    signals,
)

FACILITY = "jet"

FACILITY_CONFIG = {
    "ssh_host": "jet-host",
    "data_systems": {"ppf": {"reference_shot": 12345}},
}

_ENGINE_RESULT = {
    "scanned": 4,
    "discovered": 12,
    "enriched": 0,
    "checked": 0,
    "cost": 0.0,
    "elapsed_seconds": 0.5,
}


@pytest.fixture
def engine(monkeypatch: pytest.MonkeyPatch) -> dict:
    """Replace the engine entry point and record the kwargs it receives."""
    captured: dict = {}

    async def fake_engine(**kwargs):
        captured.update(kwargs)
        return dict(_ENGINE_RESULT)

    monkeypatch.setattr(
        "imas_codex.discovery.signals.parallel.run_parallel_data_discovery",
        fake_engine,
    )
    monkeypatch.setattr(
        "imas_codex.discovery.base.facility.get_facility",
        lambda facility: FACILITY_CONFIG,
    )
    monkeypatch.setattr(
        "imas_codex.discovery.signals.scanners.base.get_scanners_for_facility",
        lambda facility: [MagicMock(scanner_type="mdsplus")],
    )
    monkeypatch.setattr(
        "imas_codex.discovery.signals.scanners.base.list_scanners",
        lambda: ["mdsplus", "tdi"],
    )
    monkeypatch.setattr(
        "imas_codex.cli.discover.common.ensure_remote_environment",
        lambda config: None,
    )

    # Drive async_main directly, off the rich/plain harness.
    def fake_run_discovery(config, async_main, *, on_complete=None):
        result = asyncio.run(async_main(asyncio.Event(), None))
        if on_complete is not None:
            on_complete(result)
        return result

    monkeypatch.setattr(
        "imas_codex.cli.discover.common.run_discovery", fake_run_discovery
    )
    monkeypatch.setattr("imas_codex.cli.discover.common.use_rich_output", lambda: False)
    return captured


def test_stage_options_are_frozen() -> None:
    options = SignalsStageOptions()
    with pytest.raises(FrozenInstanceError):
        options.scan_only = True  # type: ignore[misc]


def test_scan_only_selects_the_seeding_half(engine) -> None:
    run_signals_stage(FACILITY, SignalsStageOptions(scan_only=True))
    assert engine["discover_only"] is True
    assert engine["enrich_only"] is False


def test_flush_selects_the_draining_half(engine) -> None:
    run_signals_stage(FACILITY, SignalsStageOptions(flush=True))
    assert engine["enrich_only"] is True
    assert engine["discover_only"] is False


def test_topic_reaches_the_enricher(engine) -> None:
    run_signals_stage(FACILITY, SignalsStageOptions(topic="equilibrium"))
    assert engine["focus"] == "equilibrium"


def test_limit_caps_items(engine) -> None:
    run_signals_stage(FACILITY, SignalsStageOptions(limit=7))
    assert engine["signal_limit"] == 7


def test_focus_items_are_refused_stating_the_mechanism(engine) -> None:
    with pytest.raises(click.UsageError) as excinfo:
        run_signals_stage(FACILITY, SignalsStageOptions(focus=("MAG/coil",)))
    message = str(excinfo.value)
    assert "claim query takes no item filter" in message
    assert "facility-discovery-sequence" not in message
    assert "section 7" not in message


def test_cli_focus_is_refused(engine) -> None:
    result = CliRunner().invoke(signals, [FACILITY, "--focus", "MAG/coil"])
    assert result.exit_code != 0
    assert "claim query takes no item filter" in result.output
    assert "facility-discovery-sequence" not in result.output
    assert "section 7" not in result.output


def test_click_command_is_a_thin_wrapper() -> None:
    with patch("imas_codex.cli.discover.signals.run_signals_stage") as mock_stage:
        result = CliRunner().invoke(
            signals,
            [
                FACILITY,
                "--scan-only",
                "--flush",
                "--topic",
                "eq",
                "--limit",
                "5",
                "-c",
                "2.5",
                "-s",
                "mdsplus",
                "--category",
                "MAG,PSRC",
                "--rescan",
                "--enrich-workers",
                "3",
                "--check-workers",
                "6",
                "--time",
                "7",
                "--reference-shot",
                "99",
            ],
        )
    assert result.exit_code == 0, result.output
    facility, options = mock_stage.call_args.args
    assert facility == FACILITY
    assert isinstance(options, SignalsStageOptions)
    assert options.scan_only is True
    assert options.flush is True
    assert options.topic == "eq"
    assert options.limit == 5
    assert options.cost_limit == 2.5
    assert options.scanners == "mdsplus"
    assert options.categories == "MAG,PSRC"
    assert options.rescan is True
    assert options.enrich_workers == 3
    assert options.check_workers == 6
    assert options.time_limit == 7
    assert options.reference_shot == 99


def test_enrich_only_is_a_deprecated_alias_for_flush() -> None:
    with patch("imas_codex.cli.discover.signals.run_signals_stage") as mock_stage:
        result = CliRunner().invoke(signals, [FACILITY, "--enrich-only"])
    assert result.exit_code == 0, result.output
    options = mock_stage.call_args.args[1]
    assert options.flush is True


def test_cli_scan_only_reaches_the_engine(engine) -> None:
    result = CliRunner().invoke(signals, [FACILITY, "--scan-only", "-s", "mdsplus"])
    assert result.exit_code == 0, result.output
    assert engine["discover_only"] is True
    assert engine["enrich_only"] is False


def test_cli_flush_reaches_the_engine(engine) -> None:
    result = CliRunner().invoke(signals, [FACILITY, "--flush", "-s", "mdsplus"])
    assert result.exit_code == 0, result.output
    assert engine["enrich_only"] is True
    assert engine["discover_only"] is False
