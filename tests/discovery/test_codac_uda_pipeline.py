"""Exercise archive discovery and checking through the pipeline workers."""

from unittest.mock import AsyncMock, MagicMock, Mock

import pytest

from iter_codex.discovery.signals import parallel
from iter_codex.discovery.signals.scanners import codac_uda as catalogue
from iter_codex.graph.models import FacilitySignal


@pytest.fixture
def archive_config(tmp_path):
    return {
        "archives": [
            {"role": "fast", "host": "fast.invalid"},
            {"role": "slow", "host": "slow.invalid"},
            {"role": "operator", "host": "operators.invalid", "excluded": True},
        ],
        "readable_roles": ["slow"],
        "catalogue_role": "slow",
        "variable_patterns": ["DEMO:*"],
        "_checkpoint_path": str(tmp_path / "catalogue.json"),
    }


@pytest.mark.asyncio
async def test_seed_reports_only_acknowledged_archive_signals(
    monkeypatch, archive_config
):
    state = parallel.DataDiscoveryState(
        facility="demo",
        ssh_host="local",
        scanner_types=["codac_uda"],
        facility_config={"data_systems": {"codac_uda": archive_config}},
    )
    stored = []

    def ingest(signals):
        stored.extend(signals)
        return len(signals)

    monkeypatch.setattr(parallel, "ingest_discovered_signals", ingest)
    monkeypatch.setattr(parallel, "persist_data_access", Mock())
    monkeypatch.setattr(catalogue, "_persist_archives", Mock())
    monkeypatch.setattr(catalogue, "GraphClient", MagicMock())
    parse = Mock()
    monkeypatch.setattr(catalogue, "persist_name_parses", parse)
    monkeypatch.setattr(
        catalogue,
        "request_archive",
        Mock(
            return_value={
                "entries": [
                    {
                        "name": "DEMO:TEMP",
                        "unit": {"value": "K"},
                        "archive_metadata": {"type": {}, "full": []},
                    }
                ],
            }
        ),
    )
    monkeypatch.setattr(
        "iter_codex.discovery.signals.scanners.base.get_scanner",
        lambda name: catalogue.CodacUDAScanner(),
    )
    await parallel.seed_worker(state)
    assert state.discover_stats.processed == len(stored) == 1
    assert stored[0]["accessor"] == "DEMO:TEMP"
    parse.assert_called_once()
    assert parse.call_args.args[1] == [
        {"id": stored[0]["id"], "facility": "demo", "selector": "DEMO:TEMP"}
    ]


@pytest.mark.asyncio
async def test_check_worker_routes_archive_signal_to_registered_scanner(
    monkeypatch, archive_config
):
    signal = FacilitySignal(
        id="demo:general/codac_uda/DEMO:TEMP",
        facility_id="demo",
        accessor="DEMO:TEMP",
        discovery_source="codac_uda",
    )
    state = parallel.DataDiscoveryState(
        facility="demo",
        ssh_host="local",
        scanner_types=["codac_uda"],
        facility_config={"data_systems": {"codac_uda": archive_config}},
    )
    scanner = Mock()
    scanner.check = AsyncMock(
        return_value=[
            {"signal_id": signal.id, "valid": True, "shape": [2], "dtype": "float64"}
        ]
    )

    def claim(*args, **kwargs):
        state.stop_requested = True
        return [signal.model_dump()]

    marked = Mock()
    monkeypatch.setattr(parallel, "claim_signals_for_check", claim)
    monkeypatch.setattr(parallel, "carry_forward_recorded_checks", lambda *args: 0)
    monkeypatch.setattr(parallel, "mark_signals_checked", marked)
    monkeypatch.setattr(
        "iter_codex.discovery.signals.scanners.base.get_scanner", lambda name: scanner
    )
    await parallel.check_worker(state)
    scanner.check.assert_awaited_once()
    assert scanner.check.call_args.kwargs["signals"][0].id == signal.id
    assert marked.call_args.args[0][0]["success"] is True
    assert state.check_stats.processed == 1
