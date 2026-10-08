"""Scanner failures remain visible through the signals stage receipt."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from imas_codex.cli.discover.signals import SignalsStageOptions, run_signals_stage
from imas_codex.discovery.signals.scanners.base import ScanResult


@pytest.mark.parametrize("scan_only", [True, False])
def test_scanner_failures_reach_stage_receipt(
    monkeypatch: pytest.MonkeyPatch, scan_only: bool
) -> None:
    from imas_codex.cli.discover import common
    from imas_codex.discovery.signals import parallel

    config = {"ssh_host": "facility-host", "data_systems": {"edas": {}, "ppf": {}}}
    monkeypatch.setattr(
        "imas_codex.discovery.base.facility.get_facility", lambda facility: config
    )
    monkeypatch.setattr(common, "ensure_remote_environment", lambda config: None)
    monkeypatch.setattr(common, "use_rich_output", lambda: False)
    monkeypatch.setattr(common, "setup_logging", lambda *args: None)
    monkeypatch.setattr(common, "make_log_print", lambda *args: lambda message: None)
    monkeypatch.setattr(
        "imas_codex.discovery.signals.scanners.base.get_scanners_for_facility",
        lambda facility: [MagicMock(scanner_type=name) for name in ("edas", "ppf")],
    )
    monkeypatch.setattr(parallel, "reset_transient_signals", lambda facility: None)
    monkeypatch.setattr(parallel, "detect_signal_sources", lambda facility: (0, 0))
    monkeypatch.setattr(parallel, "get_data_discovery_stats", lambda *args: {})
    monkeypatch.setattr(
        "imas_codex.discovery.mdsplus.graph_ops.get_version_counts",
        lambda facility: {},
    )
    graph = MagicMock()
    graph.__enter__.return_value = graph
    monkeypatch.setattr("imas_codex.graph.GraphClient", lambda: graph)

    class Scanner:
        def __init__(self, failure: str):
            self.failure = failure

        async def scan(self, **kwargs):
            if self.failure == "raised":
                raise ValueError("remote catalog failed")
            return ScanResult(stats={"error": "remote scan failed"})

    scanners = {"edas": Scanner("returned"), "ppf": Scanner("raised")}
    monkeypatch.setattr(
        "imas_codex.discovery.signals.scanners.base.get_scanner",
        scanners.__getitem__,
    )

    async def drive_seed(state, workers, **kwargs):
        await parallel.seed_worker(state)

    monkeypatch.setattr(parallel, "run_discovery_engine", drive_seed)

    async def no_individualization(*args, **kwargs):
        return 0

    monkeypatch.setattr(
        parallel, "individualize_source_descriptions", no_individualization
    )

    def run_discovery(config, async_main, *, on_complete=None):
        return asyncio.run(async_main(asyncio.Event(), None))

    monkeypatch.setattr(common, "run_discovery", run_discovery)

    receipt = run_signals_stage("facility", SignalsStageOptions(scan_only=scan_only))
    assert receipt["errors"] == {
        "edas": "remote scan failed",
        "ppf": "remote catalog failed",
    }
