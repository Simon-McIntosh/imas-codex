"""The signals discovery stage function and bare command routing.

``run_signals_stage`` carries the discovery body behind a frozen
:class:`SignalsStageOptions`; the discovery sequence builds those
options and calls the stage. These tests measure the command surface:

- ``--scan-only`` selects the seeding half: the engine gets ``discover_only``.
- ``--flush`` selects the draining half: the engine gets ``enrich_only``.
- ``--topic`` is the free-text steer for enrichment.
- ``--limit`` caps items.
- ``--focus ITEMS`` is validated against the graph: an item naming nothing at
  the facility is refused with the item named, and a known item reaches the
  engine as ``focus_items``.

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

from imas_codex.cli.discover import discover
from imas_codex.cli.discover.signals import (
    SignalsStageOptions,
    _validate_focus,
    run_signals_stage,
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


class _FocusCatalogue:
    """Resolve focus items against a toy catalogue as ``_validate_focus`` does.

    The validator runs an id/accessor query and a ``data_source_path`` segment
    query. The segment query joins ``MEMBER_OF``, so a segment reaches the
    catalogue only through a source member; a source-less signal carrying the
    segment is not matched — mirroring the claim predicate. The fake models
    both shapes: a segment query that joins ``MEMBER_OF`` resolves through
    members only, while a segment query without the join resolves a segment on
    any node's own path.
    """

    def __init__(
        self,
        signals: list[dict] | None = None,
        sources: list[dict] | None = None,
    ) -> None:
        self.signals = signals or []
        self.sources = sources or []

    def __enter__(self) -> _FocusCatalogue:
        return self

    def __exit__(self, *_: object) -> bool:
        return False

    def query(self, cypher: str, **params: object) -> list[dict]:
        ids = set(params["ids"])
        if "data_source_path" in cypher:
            member_only = "MEMBER_OF" in cypher
            return [
                {
                    "id": signal["id"],
                    "accessor": signal.get("accessor"),
                    "data_source_path": signal["data_source_path"],
                }
                for signal in self.signals
                if signal.get("data_source_path")
                and (signal.get("source_id") or not member_only)
                and any(
                    segment in ids for segment in signal["data_source_path"].split("/")
                )
            ]
        return [
            {"id": node["id"], "accessor": node.get("accessor")}
            for node in [*self.signals, *self.sources]
            if node["id"] in ids or node.get("accessor") in ids
        ]


def _install_catalogue(monkeypatch, catalogue: _FocusCatalogue) -> None:
    monkeypatch.setattr(
        "imas_codex.graph.GraphClient", lambda *a, **k: catalogue, raising=True
    )


@pytest.fixture(autouse=True)
def bare_discover_stage(monkeypatch, tmp_path):
    from imas_codex.cli import logging as cli_logging
    from imas_codex.cli.discover import sequence

    monkeypatch.setattr(
        sequence,
        "evaluate_stage",
        lambda stage, facility, config: sequence.StageOutcome(
            stage.name, stage.domain, sequence.RUNNABLE, "ready"
        ),
    )
    monkeypatch.setattr(sequence, "_remaining_count", lambda *args: None)
    monkeypatch.setattr(
        "imas_codex.discovery.base.facility.get_facility", lambda facility: {}
    )
    monkeypatch.setattr(cli_logging, "configure_cli_logging", lambda *a, **k: None)
    monkeypatch.setattr(
        cli_logging, "get_log_file", lambda *a, **k: tmp_path / "discover.log"
    )


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


def test_flush_opens_no_ssh(engine, monkeypatch) -> None:
    """A draining run reads only the graph and the model, so it needs no SSH."""
    from imas_codex.cli.discover import common

    seen: dict = {}
    drive = common.run_discovery

    def record_config(config, async_main, *, on_complete=None):
        seen["check_ssh"] = config.check_ssh
        return drive(config, async_main, on_complete=on_complete)

    monkeypatch.setattr(common, "run_discovery", record_config)
    probes: list[dict] = []
    monkeypatch.setattr(common, "ensure_remote_environment", probes.append)

    run_signals_stage(FACILITY, SignalsStageOptions(flush=True))
    assert seen["check_ssh"] is False
    assert probes == []

    run_signals_stage(FACILITY, SignalsStageOptions())
    assert seen["check_ssh"] is True
    assert len(probes) == 1


def test_reset_takes_the_category_scope(engine, monkeypatch) -> None:
    """A category-scoped reset touches only the categories the run will claim."""
    calls: list[dict] = []

    def fake_reset(spec, facility, *, extra_filter="", extra_params=None, **kw):
        calls.append({"filter": extra_filter, "params": extra_params or {}})
        return 3

    monkeypatch.setattr("imas_codex.discovery.base.reset.reset_to_status", fake_reset)

    run_signals_stage(
        FACILITY,
        SignalsStageOptions(
            reset_to="discovered", categories="MMSYS, MDAC", flush=True
        ),
    )

    (call,) = calls
    assert call["params"]["categories"] == ["MMSYS", "MDAC"]
    assert "split(coalesce(n.data_source_path, n.name), '/')[0]" in call["filter"]
    assert call["filter"].lstrip().startswith("AND ")
    assert engine["categories"] == ["MMSYS", "MDAC"]


def test_topic_reaches_the_enricher(engine) -> None:
    run_signals_stage(FACILITY, SignalsStageOptions(topic="equilibrium"))
    assert engine["focus"] == "equilibrium"


def test_limit_caps_items(engine) -> None:
    run_signals_stage(FACILITY, SignalsStageOptions(limit=7))
    assert engine["signal_limit"] == 7


def test_unknown_focus_item_is_refused_naming_it(engine, monkeypatch) -> None:
    _install_catalogue(monkeypatch, _FocusCatalogue())
    with pytest.raises(click.UsageError) as excinfo:
        run_signals_stage(FACILITY, SignalsStageOptions(focus=("MAG/coil",)))
    message = str(excinfo.value)
    assert "MAG/coil" in message
    assert "unknown signal or source id" in message


def test_known_focus_item_reaches_the_engine_as_focus_items(
    engine, monkeypatch
) -> None:
    _install_catalogue(
        monkeypatch,
        _FocusCatalogue(
            signals=[
                {
                    "id": "jet:sig",
                    "accessor": "A1",
                    "source_id": "jet:src",
                    "data_source_path": "MDAC/magPbTC10",
                },
            ]
        ),
    )
    run_signals_stage(FACILITY, SignalsStageOptions(focus=("magPbTC10",)))
    assert engine["focus_items"] == ["magPbTC10"]


def test_validate_focus_accepts_a_segment_through_a_source_member(monkeypatch) -> None:
    _install_catalogue(
        monkeypatch,
        _FocusCatalogue(
            signals=[
                {
                    "id": "jet:sig",
                    "accessor": "A1",
                    "source_id": "jet:src",
                    "data_source_path": "MDAC/magPbTC10",
                },
            ]
        ),
    )
    _validate_focus(FACILITY, ["magPbTC10"])


def test_validate_focus_refuses_a_sourceless_signal_carrying_the_segment(
    monkeypatch,
) -> None:
    """A segment is reached only through a MEMBER_OF source, as the claim does.

    A source-less signal carrying the segment would pass this validation and
    then be selected by nothing, so it must be refused.
    """
    _install_catalogue(
        monkeypatch,
        _FocusCatalogue(
            signals=[
                {
                    "id": "jet:sig",
                    "accessor": "A1",
                    "source_id": None,
                    "data_source_path": "MDAC/magPbTC10",
                },
            ]
        ),
    )
    with pytest.raises(click.UsageError) as excinfo:
        _validate_focus(FACILITY, ["magPbTC10"])
    assert "magPbTC10" in str(excinfo.value)


def test_validate_focus_accepts_a_signal_accessor(monkeypatch) -> None:
    _install_catalogue(
        monkeypatch,
        _FocusCatalogue(signals=[{"id": "jet:sig", "accessor": "A1"}]),
    )
    _validate_focus(FACILITY, ["A1"])


def test_cli_unknown_focus_is_refused(engine, monkeypatch) -> None:
    _install_catalogue(monkeypatch, _FocusCatalogue())
    result = CliRunner().invoke(
        discover, [FACILITY, "--only", "signals", "--focus", "MAG/coil"]
    )
    assert result.exit_code != 0
    assert "MAG/coil" in result.output


def test_click_command_is_a_thin_wrapper() -> None:
    with patch("imas_codex.cli.discover.signals.run_signals_stage") as mock_stage:
        result = CliRunner().invoke(
            discover,
            [
                FACILITY,
                "--only",
                "signals",
                "--scan-only",
                "--topic",
                "eq",
                "--limit",
                "5",
                "-c",
                "2.5",
                "--scanners",
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
    assert options.flush is False
    assert options.topic == "eq"
    assert options.limit == 5
    assert options.cost_limit == 2.5
    assert options.scanners == "mdsplus"
    assert options.categories == "MAG,PSRC"
    assert options.rescan is True
    assert options.enrich_workers == 3
    assert options.check_workers == 6
    assert 6.9 < options.time_limit <= 7
    assert options.reference_shot == 99


def test_retired_enrich_alias_is_refused() -> None:
    result = CliRunner().invoke(
        discover, [FACILITY, "--only", "signals", "--enrich-only"]
    )
    assert result.exit_code != 0
    assert "No such option: --enrich-only" in result.output


def test_cli_scan_only_reaches_the_engine(engine) -> None:
    result = CliRunner().invoke(
        discover,
        [FACILITY, "--only", "signals", "--scan-only", "--scanners", "mdsplus"],
    )
    assert result.exit_code == 0, result.output
    assert engine["discover_only"] is True
    assert engine["enrich_only"] is False


def test_cli_flush_reaches_the_engine(engine) -> None:
    result = CliRunner().invoke(
        discover, [FACILITY, "--only", "signals", "--flush", "--scanners", "mdsplus"]
    )
    assert result.exit_code == 0, result.output
    assert engine["enrich_only"] is True
    assert engine["discover_only"] is False
