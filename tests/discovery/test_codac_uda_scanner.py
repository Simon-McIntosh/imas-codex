"""Archive cohort progress and serving provenance with invented identities."""

import asyncio
import json
from unittest.mock import Mock

import pytest

from imas_codex.discovery.signals.scanners import codac_uda as catalogue
from imas_codex.remote import codac_uda as transport


@pytest.fixture
def config(tmp_path):
    return {
        "archives": [
            {"role": "fast", "host": "fast.invalid"},
            {"role": "slow", "host": "slow.invalid"},
            {"role": "operator", "host": "operators.invalid", "excluded": True},
        ],
        "readable_roles": ["fast", "slow"],
        "catalogue_role": "slow",
        "variable_patterns": ["DEMO:*"],
        "validation_start_ns": 1_000_000_000,
        "validation_end_ns": 2_000_000_000,
        "_checkpoint_path": str(tmp_path / "cohort.json"),
    }


def entries(count):
    return [
        {
            "name": f"DEMO:TEMP-{index}",
            "unit": {"value": "K"},
            "description": {"value": "Archive wording"},
        }
        for index in range(count)
    ]


@pytest.fixture
def scanner_io(monkeypatch):
    stored = {}

    def write(facility, cohort, signals):
        stored.update({signal.id: signal for signal in signals})
        return len(signals)

    monkeypatch.setattr(catalogue, "_persist_archives", Mock())
    monkeypatch.setattr(catalogue, "_persist_signals", write)
    reader = Mock(return_value={"entries": entries(3), "warnings": []})
    monkeypatch.setattr(catalogue, "request_archive", reader)
    return stored, reader


@pytest.mark.asyncio
async def test_scanner_is_registered_and_preserves_exact_metadata(config, scanner_io):
    from imas_codex.discovery.signals.parallel import get_signal_scanner_type
    from imas_codex.discovery.signals.scanners import get_scanner

    scanner = get_scanner("codac_uda")
    result = await scanner.scan("demo", "local", config)
    assert result.stats["complete"] is True
    assert result.stats["enumerated"] == result.stats["written"] == 3
    assert len(scanner_io[0]) == 3
    signal = result.signals[0]
    assert signal.accessor == signal.name == "DEMO:TEMP-0"
    assert signal.unit == "K" and signal.source_description == "Archive wording"
    assert signal.metadata_archive == "demo:codac_uda:slow"
    assert signal.data_archive is None and signal.sampling_mode == "unknown"
    assert get_signal_scanner_type(signal.model_dump()) == "codac_uda"
    state = json.loads(open(config["_checkpoint_path"]).read())
    assert state["complete"] is True and state["written"] == 3


@pytest.mark.asyncio
async def test_second_scan_reenumerates_but_writes_no_new_identity(config, scanner_io):
    scanner = catalogue.CodacUDAScanner()
    first = await scanner.scan("demo", "local", config)
    second = await scanner.scan("demo", "local", config)
    assert [item.id for item in first.signals] == [item.id for item in second.signals]
    assert len(scanner_io[0]) == 3 and scanner_io[1].call_count == 2


@pytest.mark.asyncio
async def test_interrupted_graph_batch_stays_incomplete_and_resumes(
    config, scanner_io, monkeypatch
):
    scanner_io[1].return_value = {"entries": entries(150)}
    calls = []

    def interrupted(facility, cohort, signals):
        calls.append(len(signals))
        if len(calls) == 2:
            raise asyncio.CancelledError()
        return len(signals)

    monkeypatch.setattr(catalogue, "_persist_signals", interrupted)
    scanner = catalogue.CodacUDAScanner()
    with pytest.raises(asyncio.CancelledError):
        await scanner.scan("demo", "local", config)
    state = json.loads(open(config["_checkpoint_path"]).read())
    assert state["complete"] is False
    assert state["written"] == 100 and state["enumerated"] == 150
    assert state["error"] == "CancelledError"
    result = await scanner.scan("demo", "local", config)
    assert result.stats["complete"] is True and result.stats["written"] == 150
    assert calls == [100, 50, 50]
    assert scanner_io[1].call_count == 1
    assert result.stats["skipped"]["already_written"] == 100


@pytest.mark.asyncio
async def test_unacknowledged_graph_write_cannot_finish(
    config, scanner_io, monkeypatch
):
    monkeypatch.setattr(catalogue, "_persist_signals", lambda *args: 0)
    with pytest.raises(RuntimeError, match="incomplete graph batch"):
        await catalogue.CodacUDAScanner().scan("demo", "local", config)
    state = json.loads(open(config["_checkpoint_path"]).read())
    assert state["complete"] is False and state["written"] == 0


@pytest.mark.asyncio
async def test_empty_reads_leave_serving_archive_unset(config, monkeypatch):
    reader = Mock(return_value={"time": [], "values": [], "empty": True})
    confirm = Mock()
    monkeypatch.setattr(catalogue, "request_archive", reader)
    monkeypatch.setattr(catalogue, "_confirm", confirm)
    signal = catalogue.CodacUDAScanner._signal(
        "demo", "demo:codac_uda:slow", "demo:codac_uda", entries(1)[0], "cohort"
    )
    result = await catalogue.CodacUDAScanner().check("demo", "local", [signal], config)
    assert result[0]["valid"] is False and result[0]["data_archive"] is None
    assert signal.data_archive is None
    assert reader.call_count == 2
    confirm.assert_not_called()


@pytest.mark.asyncio
async def test_serving_archive_requires_samples_and_can_differ_from_metadata(
    config, monkeypatch
):
    reader = Mock(
        side_effect=[
            {"time": [], "values": [], "empty": True},
            {"time": [1, 2], "values": [[3, 2, 4], [4, 3, 5]], "empty": False},
        ]
    )
    confirm = Mock()
    monkeypatch.setattr(catalogue, "request_archive", reader)
    monkeypatch.setattr(catalogue, "_confirm", confirm)
    signal = catalogue.CodacUDAScanner._signal(
        "demo", "demo:codac_uda:fast", "demo:codac_uda", entries(1)[0], "cohort"
    )
    results = await catalogue.CodacUDAScanner().check("demo", "local", [signal], config)
    assert results[0]["valid"] is True
    assert signal.metadata_archive == "demo:codac_uda:fast"
    assert signal.data_archive == "demo:codac_uda:slow"
    confirm.assert_called_once_with(signal.id, "demo:codac_uda:slow")


@pytest.mark.asyncio
async def test_operator_alias_is_refused_before_any_work(config, scanner_io):
    config["archives"][0]["host"] = "OPERATORS.other."
    with pytest.raises(ValueError, match="reserved or excluded"):
        await catalogue.CodacUDAScanner().scan("demo", "local", config)
    scanner_io[1].assert_not_called()
    catalogue._persist_archives.assert_not_called()


def test_private_block_must_redeclare_even_nonreadable_roles(config):
    config["readable_roles"] = ["slow"]
    config["archives"].pop(0)
    with pytest.raises(ValueError, match="redeclare every archive role"):
        transport.archive_config("demo", config)


def test_transport_uses_facility_runner_and_preserves_reader_error(config, monkeypatch):
    import subprocess

    runner = Mock(return_value=json.dumps({"ok": True, "result": {"entries": []}}))
    monkeypatch.setattr(transport, "run_python_script", runner)
    assert transport.request_archive(
        "demo", "slow", {"operation": "variables", "pattern": "DEMO:*"}, config=config
    ) == {"entries": []}
    assert runner.call_args.kwargs["facility"] == "demo"
    assert runner.call_args.kwargs["data_system"] == "codac_uda"
    assert runner.call_args.args[1]["connection"]["host"] == "slow.invalid"
    runner.side_effect = subprocess.CalledProcessError(
        1,
        "reader",
        json.dumps(
            {"ok": False, "error_type": "ReadError", "error": "bounded refusal"}
        ),
    )
    with pytest.raises(transport.ArchiveRequestError, match="bounded refusal") as error:
        transport.request_archive(
            "demo",
            "slow",
            {"operation": "variables", "pattern": "DEMO:*"},
            config=config,
        )
    assert error.value.error_type == "ReadError"


@pytest.mark.parametrize(
    "payload",
    [
        {"operation": "read", "start": 0, "end": 100_000_000_000},
        {"operation": "read", "start": None, "end": None},
        {"operation": "variables", "host": "other.invalid"},
    ],
)
def test_transport_refuses_unbounded_or_connection_override(
    config, monkeypatch, payload
):
    runner = Mock()
    monkeypatch.setattr(transport, "run_python_script", runner)
    with pytest.raises(ValueError):
        transport.request_archive("demo", "slow", payload, config=config)
    runner.assert_not_called()


def test_compute_host_resolves_facility_scripts_locally(monkeypatch):
    from imas_codex.remote import executor

    monkeypatch.setattr(executor.socket, "getfqdn", lambda: "98dci4-clu-9999.iter.org")
    monkeypatch.setattr(executor.socket, "gethostname", lambda: "98dci4-clu-9999")
    executor._get_facility_local_hosts.cache_clear()
    executor._get_configured_local_hosts.cache_clear()
    try:
        assert executor.is_local_host("iter")
    finally:
        executor._get_facility_local_hosts.cache_clear()
        executor._get_configured_local_hosts.cache_clear()
