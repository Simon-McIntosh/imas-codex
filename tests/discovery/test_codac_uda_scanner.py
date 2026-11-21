"""Archive cohort progress and serving provenance with invented identities."""

import asyncio
import json
from unittest.mock import MagicMock, Mock

import numpy as np
import pytest

from iter_codex.discovery.signals import parallel
from iter_codex.discovery.signals.scanners import codac_uda as catalogue
from iter_codex.remote import codac_uda as transport


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
            "archive_metadata": {"type": {"datatypes": []}, "full": []},
            "unit": {"value": "K"},
            "description": {"value": "Archive wording"},
        }
        for index in range(count)
    ]


@pytest.fixture
def scanner_io(monkeypatch):
    stored = {}

    def write(signals):
        stored.update({signal["id"]: signal for signal in signals})
        return len(signals)

    monkeypatch.setattr(catalogue, "_persist_archives", Mock())
    monkeypatch.setattr(catalogue, "GraphClient", MagicMock())
    monkeypatch.setattr(catalogue, "persist_name_parses", Mock())
    monkeypatch.setattr(parallel, "ingest_discovered_signals", write)
    reader = Mock(return_value={"entries": entries(3), "warnings": []})

    def respond(facility, role, payload, **kwargs):
        response = reader.return_value
        if payload["operation"] == "metadata":
            wanted = set(payload["names"])
            return {
                "entries": [
                    entry for entry in response["entries"] if entry["name"] in wanted
                ],
                "warnings": response.get("warnings", []),
            }
        return response

    reader.side_effect = respond
    monkeypatch.setattr(catalogue, "request_archive", reader)
    return stored, reader


@pytest.mark.asyncio
async def test_scanner_is_registered_and_preserves_exact_metadata(config, scanner_io):
    from iter_codex.discovery.signals.parallel import get_signal_scanner_type
    from iter_codex.discovery.signals.scanners import get_scanner

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
    assert len(scanner_io[0]) == 3
    assert (
        sum(
            call.args[2]["operation"] == "variables"
            for call in scanner_io[1].call_args_list
        )
        == 2
    )


@pytest.mark.asyncio
async def test_interrupted_graph_batch_stays_incomplete_and_resumes(
    config, scanner_io, monkeypatch
):
    scanner_io[1].return_value = {"entries": entries(150)}
    calls = []

    def interrupted(signals):
        calls.append(len(signals))
        if len(calls) == 2:
            raise asyncio.CancelledError()
        return len(signals)

    monkeypatch.setattr(parallel, "ingest_discovered_signals", interrupted)
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
    assert scanner_io[1].call_count == 3
    assert result.stats["skipped"]["already_written"] == 100


@pytest.mark.asyncio
async def test_unacknowledged_graph_write_cannot_finish(
    config, scanner_io, monkeypatch
):
    monkeypatch.setattr(parallel, "ingest_discovered_signals", lambda *args: 0)
    with pytest.raises(RuntimeError, match="incomplete graph batch"):
        await catalogue.CodacUDAScanner().scan("demo", "local", config)
    state = json.loads(open(config["_checkpoint_path"]).read())
    assert state["complete"] is False and state["written"] == 0


@pytest.mark.asyncio
async def test_metadata_calls_checkpoint_and_resume_independently(
    config, scanner_io, monkeypatch
):
    config["metadata_batch_size"] = 20
    records = entries(55)
    by_name = {entry["name"]: entry for entry in records}
    metadata_calls = []
    interrupted = False

    def reader(facility, role, payload, **kwargs):
        nonlocal interrupted
        if payload["operation"] == "variables":
            assert payload.get("metadata") is False
            return {"entries": [{"name": name} for name in by_name]}
        assert payload["operation"] == "metadata"
        assert 0 < len(payload["names"]) <= 20
        metadata_calls.append(list(payload["names"]))
        if len(metadata_calls) == 2 and not interrupted:
            interrupted = True
            state = json.loads(open(config["_checkpoint_path"]).read())
            assert state["complete"] is False
            assert state["metadata_completed"] == 20
            raise TimeoutError("metadata request interrupted")
        return {"entries": [by_name[name] for name in payload["names"]]}

    monkeypatch.setattr(catalogue, "request_archive", reader)
    scanner = catalogue.CodacUDAScanner()
    with pytest.raises(TimeoutError, match="metadata request interrupted"):
        await scanner.scan("demo", "local", config)
    state = json.loads(open(config["_checkpoint_path"]).read())
    assert state["complete"] is False
    assert state["enumerated"] == 55
    assert state["metadata_completed"] == 20
    assert state["written"] == 20
    result = await scanner.scan("demo", "local", config)
    assert result.stats["complete"] is True
    assert result.stats["metadata_completed"] == result.stats["written"] == 55
    assert [len(names) for names in metadata_calls] == [20, 20, 20, 15]
    assert set(metadata_calls[0]).isdisjoint(
        {name for batch in metadata_calls[2:] for name in batch}
    )


@pytest.mark.asyncio
async def test_metadata_is_durable_before_graph_acknowledgement(
    config, scanner_io, monkeypatch
):
    def write(signals):
        state = json.loads(open(config["_checkpoint_path"]).read())
        assert state["complete"] is False
        assert state["metadata_completed"] == len(signals) == 3
        assert state["written"] == 0
        raise TimeoutError("graph acknowledgement interrupted")

    monkeypatch.setattr(parallel, "ingest_discovered_signals", write)
    with pytest.raises(TimeoutError, match="graph acknowledgement interrupted"):
        await catalogue.CodacUDAScanner().scan("demo", "local", config)
    state = json.loads(open(config["_checkpoint_path"]).read())
    assert state["metadata_completed"] == 3 and state["complete"] is False


@pytest.mark.asyncio
async def test_empty_reads_leave_serving_archive_unset(config, monkeypatch):
    reader = Mock()
    reader.check.return_value = {
        "success": False,
        "error_classification": "empty",
        "error": "archive interval is empty",
    }
    confirm = Mock()
    monkeypatch.setattr(catalogue, "ArchiveReader", Mock(return_value=reader))
    monkeypatch.setattr(catalogue, "_confirm", confirm)
    signal = catalogue.CodacUDAScanner._signal(
        "demo", "demo:codac_uda:slow", "demo:codac_uda", entries(1)[0], "cohort"
    )
    result = await catalogue.CodacUDAScanner().check("demo", "local", [signal], config)
    assert result[0]["valid"] is False and result[0]["data_archive"] is None
    assert signal.data_archive is None
    assert reader.check.call_count == 2
    confirm.assert_not_called()


@pytest.mark.asyncio
async def test_serving_archive_requires_samples_and_can_differ_from_metadata(
    config, monkeypatch
):
    reader = Mock(archive="demo:codac_uda:slow")
    reader.check.side_effect = [
        {"success": False, "error_classification": "empty", "error": "empty"},
        {
            "success": True,
            "signal": transport.Signal(
                np.array([1, 2]), np.array([[3, 2, 4], [4, 3, 5]]), {}
            ),
        },
    ]
    confirm = Mock()
    monkeypatch.setattr(catalogue, "ArchiveReader", Mock(return_value=reader))
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
    assert runner.call_args.kwargs["input_data"]["operation"] == "variables"
    assert "CODAC_UDA_HOST=slow.invalid" in " ".join(
        runner.call_args.kwargs["setup_commands"]
    )
    runner.side_effect = subprocess.CalledProcessError(
        1,
        "reader",
        json.dumps(
            {"ok": False, "error_type": "ReadError", "error": "bounded refusal"}
        ),
    )
    with pytest.raises(transport.ArchiveReadError, match="bounded refusal") as error:
        transport.request_archive(
            "demo",
            "slow",
            {"operation": "variables", "pattern": "DEMO:*"},
            config=config,
        )
    assert error.value.category == "limit"


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
    from iter_codex.remote import executor

    monkeypatch.setattr(executor.socket, "getfqdn", lambda: "98dci4-clu-9999.iter.org")
    monkeypatch.setattr(executor.socket, "gethostname", lambda: "98dci4-clu-9999")
    executor._get_facility_local_hosts.cache_clear()
    executor._get_configured_local_hosts.cache_clear()
    try:
        assert executor.is_local_host("iter")
    finally:
        executor._get_facility_local_hosts.cache_clear()
        executor._get_configured_local_hosts.cache_clear()


@pytest.mark.asyncio
async def test_scanner_uses_loader_samples_and_classifies_the_same_refusal(
    config, monkeypatch, tmp_path
):
    config["readable_roles"] = ["slow"]
    original_reader = transport.ArchiveReader
    reader = original_reader(
        "demo",
        "slow",
        config={
            "data_systems": {
                "codac_uda": {
                    key: value
                    for key, value in config.items()
                    if not key.startswith("_")
                }
            }
        },
        cache_dir=tmp_path / "reads",
    )
    runner = Mock(
        return_value=json.dumps(
            {
                "ok": True,
                "result": {
                    "time": [1, 2],
                    "values": [3, 4],
                    "time_dtype": "int64",
                    "value_dtype": "float32",
                    "time_base": "epoch_ns",
                },
            }
        )
    )
    monkeypatch.setattr(transport, "run_python_script", runner)
    monkeypatch.setattr(catalogue, "ArchiveReader", Mock(return_value=reader))
    confirm = Mock()
    monkeypatch.setattr(catalogue, "_confirm", confirm)
    signal = catalogue.CodacUDAScanner._signal(
        "demo",
        reader.archive,
        catalogue._access("demo", "slow").id,
        entries(1)[0],
        "cohort",
    )
    graph = Mock()
    graph.query.return_value = [
        {
            "signal_id": signal.id,
            "facility_id": "demo",
            "selector": signal.accessor,
            "unit": "K",
            "description": "Archive wording",
            "archive_id": reader.archive,
            "archive_role": "slow",
            "method_type": "codac_uda",
        }
    ]
    loader = transport.ArchiveSignalLoader(graph, reader_factory=lambda *args: reader)
    parameters = {"start": 1, "end": 3, "mode": "raw", "bins": 2, "max_samples": 10}
    checked = (
        await catalogue.CodacUDAScanner().check(
            "demo", "local", [signal], config, read_parameters=parameters
        )
    )[0]
    loaded = loader.load(signal.id, **parameters)
    np.testing.assert_array_equal(checked["signal"].time, loaded.time)
    np.testing.assert_array_equal(checked["signal"].values, loaded.values)
    assert checked["signal"].values.dtype == loaded.values.dtype == np.float32
    confirm.assert_called_once_with(signal.id, reader.archive)
    refused = parameters | {"max_samples": config.get("max_samples", 100_000) + 1}
    failed = (
        await catalogue.CodacUDAScanner().check(
            "demo", "local", [signal], config, read_parameters=refused
        )
    )[0]
    with pytest.raises(transport.ArchiveReadError) as failure:
        loader.load(signal.id, **refused)
    assert failed["error_classification"] == failure.value.category == "limit"
    assert confirm.call_count == 1
    runner.assert_called_once()


@pytest.mark.asyncio
async def test_failed_metadata_is_catalogued_as_unavailable(config, scanner_io):
    stored, reader = scanner_io

    def respond(facility, role, payload, **kwargs):
        if payload["operation"] == "variables":
            return {"entries": [{"name": "DEMO:FAILED"}]}
        return {
            "entries": [],
            "failed": [
                {"name": "DEMO:FAILED", "error": "fixture refusal", "error_code": 9105}
            ],
        }

    reader.side_effect = respond
    result = await catalogue.CodacUDAScanner().scan("demo", "local", config)
    assert result.stats["failed"] == 1
    assert result.stats["errors"] == {"DEMO:FAILED": "fixture refusal"}
    assert result.stats["complete"] is True
    assert result.stats["written"] == 1
    assert len(result.signals) == len(stored) == 1
    signal = result.signals[0]
    assert signal.metadata_state == "unavailable" and signal.metadata_error_code == 9105
    assert signal.archive_metadata is signal.unit is signal.source_description is None


def test_signal_keeps_all_archive_fields_and_absent_times():
    document = {
        "datatypes": [
            {"name": "main", "fields": [{"name": "val", "type": "DOUBLE", "unit": "K"}]}
        ],
        "signature": "fixture",
        "additional": {"source": "archive"},
    }
    entry = entries(1)[0] | {
        "archive_metadata": document,
        "data_type": "DOUBLE",
        "structure_fields": document["datatypes"],
    }
    signal = catalogue.CodacUDAScanner._signal(
        "demo", "demo:codac_uda:slow", "access", entry, "cohort"
    )
    assert json.loads(signal.archive_metadata) == document
    assert json.loads(signal.structure_fields) == document["datatypes"]
    assert signal.data_type == "DOUBLE"
    assert signal.first_sample_ns is signal.last_sample_ns is None
    assert signal.listed_archives == ["demo:codac_uda:slow"]


def test_unavailable_signal_clears_archive_fields_and_available_requires_document():
    entry = entries(1)[0] | {
        "metadata_state": "unavailable",
        "metadata_error_code": 9105,
        "metadata_error": "fixture refusal",
        "archive_labels": ["untrusted"],
    }
    signal = catalogue.CodacUDAScanner._signal(
        "demo", "demo:codac_uda:slow", "access", entry, "cohort"
    )
    assert signal.metadata_state == "unavailable" and signal.metadata_error_code == 9105
    assert (
        signal.archive_metadata
        is signal.archive_labels
        is signal.unit
        is signal.structure_fields
        is None
    )
    with pytest.raises(ValueError, match="requires a successful document"):
        catalogue.CodacUDAScanner._signal(
            "demo",
            "demo:codac_uda:slow",
            "access",
            {"name": "DEMO:EMPTY", "metadata_state": "available"},
            "cohort",
        )


@pytest.mark.asyncio
async def test_private_catalogue_cannot_enter_pilot_graph_ingest(
    config, scanner_io, tmp_path
):
    import sqlite3

    path = tmp_path / "private.sqlite"
    with sqlite3.connect(path) as db:
        db.executescript(
            "CREATE TABLE census(role TEXT,name TEXT,document TEXT,error TEXT,acknowledged INTEGER NOT NULL DEFAULT 0,PRIMARY KEY(role,name)); CREATE TABLE receipts(key TEXT PRIMARY KEY,value TEXT);"
        )
        db.execute("INSERT INTO census(role,name) VALUES ('slow','DEMO:TEMP-0')")
        db.execute(
            "INSERT INTO receipts VALUES ('census',?)",
            (
                json.dumps(
                    {
                        "counts": {"slow": 1, "fast": 0},
                        "union": 1,
                        "overlap": 0,
                        "fingerprint": "fixture",
                        "timestamp": "fixture",
                    }
                ),
            ),
        )
    config["_catalogue_path"] = str(path)
    with pytest.raises(ValueError, match="private catalogue"):
        await catalogue.CodacUDAScanner().scan("demo", "local", config)
    assert scanner_io[0] == {}
    scanner_io[1].assert_not_called()


def test_pilot_name_parse_follows_owner_acknowledgement(monkeypatch):
    signal = catalogue.CodacUDAScanner._signal(
        "demo", "demo:codac_uda:slow", "demo:access", entries(1)[0], "cohort"
    )
    graph = MagicMock()
    parse = Mock()
    owner = Mock(return_value=0)
    monkeypatch.setattr(catalogue, "GraphClient", graph)
    monkeypatch.setattr(catalogue, "persist_name_parses", parse)
    monkeypatch.setattr(parallel, "ingest_discovered_signals", owner)
    assert catalogue._persist_signals("demo", "cohort", [signal]) == 0
    graph.assert_not_called()
    parse.assert_not_called()
    owner.return_value = 1
    assert catalogue._persist_signals("demo", "cohort", [signal]) == 1
    owner.assert_called_with([signal.model_dump(mode="json")])
    parse.assert_called_once_with(
        graph.return_value.__enter__.return_value,
        [{"id": signal.id, "facility": "demo", "selector": signal.accessor}],
    )
