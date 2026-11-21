"""Private metadata completion, disjoint readers and graph-free job receipts."""

import fcntl
import json
import os
import subprocess
import sys
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from iter_codex.remote.archive_catalogue import ArchiveCatalogue
from iter_codex.remote.catalogue_scan import MetadataReaders, main, run_scan
from iter_codex.remote.scripts import codac_uda as reader


def available(name):
    return {
        "name": name,
        "metadata_state": "available",
        "archive_metadata": {"type": {"datatypes": []}, "full": []},
        "unit": {"value": None, "source": None},
    }


def test_store_interruption_resume_repeat_and_all_archive_memberships(tmp_path):
    path = tmp_path / "catalogue.sqlite"
    store = ArchiveCatalogue(path)
    census = store.seed(
        {"fast": ["DEMO:ONE", "DEMO:TWO"], "slow": ["DEMO:TWO", "DEMO:THREE"]},
        timestamp="fixture",
    )
    assert census["union"] == 3 and census["overlap"] == 1
    requested = []

    def fetch(role, names, cached):
        requested.extend((role, name) for name in names)
        return {"entries": [available(name) for name in names]}

    paused = store.scan(fetch, batch_size=1, max_batches=1)
    assert paused["completed"] == 1 and not paused["complete"]
    store.close()
    store = ArchiveCatalogue(path)
    completed = store.scan(fetch, batch_size=1, workers=2)
    assert completed["complete"] and completed["variables"] == 3
    assert completed["metadata_states"] == {"available": 3}
    assert len(requested) == len(set(requested)) == 4
    record = store.record("DEMO:TWO")
    assert record["listed_archives"] == ["fast", "slow"]
    assert all(item["completed"] for item in record["archives"].values())
    assert "unit" not in record["archives"]["fast"]["metadata"]
    repeated = store.scan(fetch)
    assert repeated["metadata_requests"] == repeated["rows_added"] == 0
    assert len(requested) == 4
    store.close()


def test_store_metadata_refusal_is_unavailable_with_archive_code(tmp_path):
    class Refused:
        def getMetaTypeJSONCollapsed(self, name):
            return ""

        def getMetaJSON(self, name, when):
            return ""

        def getErrorCode(self):
            return 9105

        def getErrorMsg(self):
            return "fixture archive refusal"

    store = ArchiveCatalogue(tmp_path / "refused.sqlite")
    store.seed({"slow": ["DEMO:REFUSED"]}, timestamp="fixture")
    result = store.scan(
        lambda role, names, cached: reader.metadata(Refused(), {"names": names})
    )
    entry = store.record("DEMO:REFUSED")["archives"]["slow"]["metadata"]
    assert result["complete"] and result["metadata_states"] == {"unavailable": 1}
    assert result["unavailable_codes"] == {9105: 1}
    assert (
        entry["metadata_state"] == "unavailable"
        and entry["metadata_error_code"] == 9105
    )
    assert "fixture archive refusal" in entry["metadata_error"]
    assert "archive_metadata" not in entry and "unit" not in entry
    store.close()


def test_store_validates_receipt_before_committing_and_rejects_empty_available(
    tmp_path,
):
    store = ArchiveCatalogue(tmp_path / "invalid.sqlite")
    store.seed({"slow": ["DEMO:ONE", "DEMO:TWO"]}, timestamp="fixture")
    with pytest.raises(RuntimeError, match="requested selectors"):
        store.save_batch(
            "slow", ["DEMO:ONE", "DEMO:TWO"], {"entries": [available("DEMO:ONE")]}
        )
    with pytest.raises(ValueError, match="both interface documents"):
        store.save_batch(
            "slow",
            ["DEMO:ONE"],
            {
                "entries": [
                    {
                        "name": "DEMO:ONE",
                        "metadata_state": "available",
                        "archive_metadata": {},
                    }
                ]
            },
        )
    assert store.summary()["completed"] == 0
    store.close()


def test_store_reuses_type_checkpoint_without_changing_pilot_acknowledgements(tmp_path):
    store = ArchiveCatalogue(tmp_path / "pilot.sqlite")
    store.seed({"slow": ["DEMO:EXISTING"]}, timestamp="fixture")
    document = {"datatypes": []}
    with store.db:
        store.db.execute(
            "UPDATE census SET document=?,acknowledged=1",
            (json.dumps({"name": "DEMO:EXISTING", "archive_metadata": document}),),
        )
        store.db.execute("INSERT INTO written VALUES ('DEMO:EXISTING')")

    def fetch(role, names, cached):
        assert cached == {"DEMO:EXISTING": document}
        return {"entries": [available(names[0])]}

    result = store.scan(fetch)
    assert (
        result["complete"]
        and result["type_requests"] == 0
        and result["full_requests"] == 1
    )
    assert store.db.execute("SELECT acknowledged FROM census").fetchone() == (1,)
    assert store.db.execute("SELECT count(*) FROM written").fetchone() == (1,)
    store.close()


def test_concurrent_readers_receive_disjoint_ranges_and_respect_bound(tmp_path):
    store = ArchiveCatalogue(tmp_path / "ranges.sqlite")
    names = [f"DEMO:TEMP-{index}" for index in range(4)]
    store.seed({"slow": names}, timestamp="fixture")
    barrier = threading.Barrier(4)
    seen = []
    lock = threading.Lock()

    def fetch(role, selected, cached):
        with lock:
            seen.append(selected[0])
        barrier.wait(timeout=5)
        return {"entries": [available(selected[0])]}

    result = store.scan(fetch, workers=4, batch_size=1)
    assert result["complete"] and sorted(seen) == names
    for workers in (0, 9):
        with pytest.raises(ValueError, match="between 1 and 8"):
            store.scan(fetch, workers=workers)
    store.close()


def test_stopping_scan_drains_and_checkpoints_inflight_batches(tmp_path):
    store = ArchiveCatalogue(tmp_path / "stop.sqlite")
    store.seed(
        {"slow": [f"DEMO:TEMP-{index}" for index in range(10)]}, timestamp="fixture"
    )
    stop = threading.Event()

    def fetch(role, names, cached):
        stop.set()
        return {"entries": [available(name) for name in names]}

    result = store.scan(fetch, workers=2, batch_size=1, stop=stop)
    assert 1 <= result["completed"] <= 2 and not result["complete"]
    store.close()


def test_job_receipt_is_written_on_reader_start_failure(tmp_path):
    receipt = tmp_path / "failed.json"

    def failed(config):
        raise RuntimeError("fixture startup refusal")

    with pytest.raises(RuntimeError, match="startup refusal"):
        run_scan(
            store_path=tmp_path / "store.sqlite",
            config={},
            receipt_path=receipt,
            reader_factory=failed,
        )
    result = json.loads(receipt.read_text())
    assert result["status"] == "failed" and result["graph_operations"] == 0
    assert result["error"] == "fixture startup refusal"


def test_job_runs_with_graph_and_discovery_imports_refused(tmp_path):
    program = """
import importlib.abc, json, sys
from pathlib import Path
from types import SimpleNamespace
class NoGraph(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith(('iter_codex.graph', 'iter_codex.discovery', 'neo4j')):
            raise AssertionError('graph import refused by control')
sys.meta_path.insert(0, NoGraph())
try:
    import iter_codex.graph.client
except AssertionError as error:
    assert 'refused by control' in str(error)
else:
    raise AssertionError('import guard did not fire')
from iter_codex.remote.archive_catalogue import ArchiveCatalogue
from iter_codex.remote.catalogue_scan import MetadataReaders, run_scan
probe=MetadataReaders({'data_systems':{'codac_uda':{'archives':[{'role':'fast','host':'fast.invalid'},{'role':'slow','host':'slow.invalid'},{'role':'operator','host':'operators.invalid','excluded':True}],'readable_roles':['slow']}}})
probe.close()
root=Path(sys.argv[1])
store=ArchiveCatalogue(root/'store.sqlite')
store.seed({'slow':['DEMO:TEMP']},timestamp='fixture')
store.close()
class Readers:
    settings=SimpleNamespace(readable_roles=['slow'])
    def __init__(self, config): pass
    def fetch(self, role, names, cached):
        return {'entries':[{'name':name,'metadata_state':'available','archive_metadata':{'type':{},'full':[]}} for name in names]}
    def close(self): pass
result=run_scan(store_path=root/'store.sqlite',config={},receipt_path=root/'receipt.json',reader_factory=Readers)
assert result['status']=='complete' and result['repeat']['metadata_requests']==0
assert result['graph_operations']==result['data_reads']==0
assert not any(name.startswith(('iter_codex.graph', 'iter_codex.discovery', 'neo4j')) for name in sys.modules)
print('graph import control fired; store-only scan and repeat completed')
"""
    result = subprocess.run(
        [sys.executable, "-c", program, str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "graph import control fired" in result.stdout


def test_reader_policy_refuses_operator_alias_before_start(monkeypatch):
    import iter_codex.remote.catalogue_scan as scan

    started = []
    monkeypatch.setattr(
        scan.subprocess, "Popen", lambda *args, **kwargs: started.append(args)
    )
    config = {
        "data_systems": {
            "codac_uda": {
                "archives": [
                    {"role": "fast", "host": "fast.invalid"},
                    {"role": "slow", "host": "operators.invalid"},
                    {"role": "operator", "host": "operators.invalid", "excluded": True},
                ],
                "readable_roles": ["slow"],
            }
        }
    }
    with pytest.raises(ValueError, match="reserved or excluded"):
        MetadataReaders(config)
    assert started == []


def test_cli_configuration_failure_has_terminal_receipt(tmp_path):
    receipt = tmp_path / "receipt.json"
    assert (
        main(
            [
                "--config",
                str(tmp_path / "missing.yaml"),
                "--store",
                str(tmp_path / "store.sqlite"),
                "--receipt",
                str(receipt),
            ]
        )
        == 1
    )
    assert json.loads(receipt.read_text())["error_type"] == "FileNotFoundError"


def test_metadata_field_counter_distinguishes_present_zero_and_absent_unit(tmp_path):
    store = ArchiveCatalogue(tmp_path / "coverage.sqlite")
    store.seed({"slow": ["DEMO:KNOWN", "DEMO:ABSENT"]}, timestamp="fixture")
    known = available("DEMO:KNOWN") | {
        "unit": {"value": "K"},
        "archive_lower_display_limit": 0,
        "first_sample_ns": 123,
        "last_sample_ns": 456,
        "sampling_mode": "event",
        "sampling_evidence": "archive fixture",
        "structure_fields": [{"name": "val", "type": "DOUBLE"}],
        "data_type": "DOUBLE",
        "description": {"value": "Fixture archive description"},
        "archive_labels": ["Enabled"],
        "archive_format": "%.2f",
        "archive_upper_display_limit": 100,
        "archive_lower_alarm_limit": 1,
        "archive_upper_alarm_limit": 99,
        "archive_lower_warning_limit": 2,
        "archive_upper_warning_limit": 98,
    }
    store.save_batch(
        "slow",
        ["DEMO:KNOWN", "DEMO:ABSENT"],
        {"entries": [known, available("DEMO:ABSENT")]},
    )
    # A checkpoint produced by another caller may retain an absent-value wrapper.
    with store.db:
        store.db.execute(
            "UPDATE census SET document=json_set(document,'$.unit',json(?)) WHERE name='DEMO:ABSENT'",
            ('{"value":null,"source":null}',),
        )
    coverage = store.summary()["metadata_fields"]
    assert coverage["unit"] == coverage["archive_lower_display_limit"] == 1
    assert coverage["archive_metadata"] == 2
    assert all(
        value == 1
        for field, value in coverage.items()
        if field not in {"archive_metadata", "metadata_interfaces"}
    )
    store.close()


def test_partial_interface_success_is_retained_with_unavailable_state(tmp_path):
    class Partial:
        code = 0

        def getMetaTypeJSONCollapsed(self, name):
            self.code = 9105
            return ""

        def getMetaJSON(self, name, when):
            self.code = 0
            return '[{"name":"label","value":"Enabled"}]'

        def getErrorCode(self):
            return self.code

        def getErrorMsg(self):
            return "fixture type refusal"

    store = ArchiveCatalogue(tmp_path / "partial.sqlite")
    store.seed({"slow": ["DEMO:PARTIAL"]}, timestamp="fixture")
    result = store.scan(
        lambda role, names, cached: reader.metadata(Partial(), {"names": names})
    )
    entry = store.record("DEMO:PARTIAL")["archives"]["slow"]["metadata"]
    assert (
        entry["metadata_state"] == "unavailable"
        and entry["metadata_error_code"] == 9105
    )
    assert entry["metadata_interfaces"]["full"] == [
        {"name": "label", "value": "Enabled"}
    ]
    assert "archive_metadata" not in entry
    assert result["metadata_fields"]["metadata_interfaces"] == 1
    store.close()


def test_another_job_is_refused_before_any_reader_starts(tmp_path):
    path = tmp_path / "catalogue.sqlite"
    readers = MagicMock()
    with path.with_suffix(".sqlite.lock").open("a") as lease:
        fcntl.flock(lease.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="another catalogue job"):
            run_scan(
                store_path=path,
                config={},
                receipt_path=tmp_path / "refused.json",
                reader_factory=readers,
            )
    readers.assert_not_called()
    assert json.loads((tmp_path / "refused.json").read_text())["status"] == "failed"


def test_reader_framing_retains_second_receipt_and_skips_native_diagnostic():
    read_fd, write_fd = os.pipe()
    os.write(
        write_fd,
        b'native diagnostic\nCODAC_METADATA {"ready":true}\nCODAC_METADATA {"ok":true}\n',
    )
    os.close(write_fd)
    readers = MetadataReaders.__new__(MetadataReaders)
    readers.timeout = 1
    with os.fdopen(read_fd, "rb") as stream:
        session = {"process": MagicMock(stdout=stream), "buffer": bytearray()}
        assert readers._reply(session) == {"ready": True}
        assert readers._reply(session) == {"ok": True}


def test_separate_population_audit_reads_only_and_requires_positive_control(
    monkeypatch, tmp_path
):
    from iter_codex.graph import client
    from iter_codex.remote.catalogue_audit import measure_population

    graph = MagicMock()
    monkeypatch.setattr(client, "GraphClient", lambda: graph)
    graph.__enter__.return_value = graph
    graph.query.return_value = [{"signals": 0, "with_facility": 0, "identities": 0}]
    with pytest.raises(RuntimeError, match="nonempty facility-property control"):
        measure_population("demo", tmp_path / "before.json")
    assert not (tmp_path / "before.json").exists()
    graph.query.return_value = [{"signals": 3, "with_facility": 3, "identities": 3}]
    receipt = measure_population("demo", tmp_path / "before.json")
    assert (
        receipt["signals"] == 3
        and json.loads((tmp_path / "before.json").read_text()) == receipt
    )
    assert all(call.args[0].startswith("MATCH") for call in graph.query.call_args_list)


def test_cleanup_failure_cannot_publish_a_complete_job_receipt(tmp_path):
    from types import SimpleNamespace

    store_path = tmp_path / "cleanup.sqlite"
    store = ArchiveCatalogue(store_path)
    store.seed({"slow": ["DEMO:TEMP"]}, timestamp="fixture")
    store.close()

    class Readers:
        settings = SimpleNamespace(readable_roles=["slow"])

        def __init__(self, config):
            pass

        def fetch(self, role, names, cached):
            return {"entries": [available(name) for name in names]}

        def close(self):
            raise RuntimeError("fixture reader cleanup failed")

    receipt = tmp_path / "receipt.json"
    with pytest.raises(RuntimeError, match="reader cleanup failed"):
        run_scan(
            store_path=store_path,
            config={},
            receipt_path=receipt,
            reader_factory=Readers,
        )
    result = json.loads(receipt.read_text())
    assert result["status"] == "failed"
    assert result["error"] == "fixture reader cleanup failed"


def timeout_reader(monkeypatch, permanent=False):
    """Exercise the real fetch boundary with deterministic native batch receipts."""
    import iter_codex.remote.catalogue_scan as scan

    readers = MetadataReaders.__new__(MetadataReaders)
    readers.local = threading.local()
    readers.max_retries = 3
    readers.backoff = 0.01
    process = MagicMock()
    requests = []
    delays = []
    readers._open = lambda role: {"role": role, "process": process}

    def reply(session):
        request = json.loads(process.stdin.write.call_args.args[0])
        names = request["names"]
        requests.append(names)
        if (not permanent and len(requests) == 1) or (
            permanent and "DEMO:TEMP-0" in names
        ):
            raise TimeoutError("fixture native batch deadline")
        return {"ok": True, "result": {"entries": [available(name) for name in names]}}

    readers._reply = reply
    readers._restart = MagicMock(
        side_effect=lambda: setattr(readers.local, "session", None)
    )
    monkeypatch.setattr(scan.time, "sleep", delays.append)
    return readers, requests, delays


def test_timeout_restarts_reader_halves_range_and_recovers_with_backoff(monkeypatch):
    readers, requests, delays = timeout_reader(monkeypatch)
    names = [f"DEMO:TEMP-{index}" for index in range(4)]
    result = readers.fetch("slow", names, {})
    assert requests == [names, names[:2], names[2:]]
    assert [entry["name"] for entry in result["entries"]] == names
    assert result["range_failures"] == []
    readers._restart.assert_called_once()
    assert delays == [0.01]


def test_exhausted_timeout_range_is_durable_and_scan_moves_on(monkeypatch, tmp_path):
    readers, requests, delays = timeout_reader(monkeypatch, permanent=True)
    path = tmp_path / "timeouts.sqlite"
    store = ArchiveCatalogue(path)
    names = [f"DEMO:TEMP-{index}" for index in range(5)]
    store.seed({"slow": names}, timestamp="fixture")
    result = store.scan(readers.fetch, batch_size=4)
    assert result["completed"] == 1 and result["failed_variables"] == 4
    assert result["metadata_states"] == {"available": 1}
    assert len(requests) == 5 and requests[-1] == [names[-1]]
    assert delays == [0.01, 0.02, 0.04]
    assert readers._restart.call_count == 4
    failed = store.record(names[0])["archives"]["slow"]
    assert (
        failed["fetch_state"] == "failed"
        and "fixture native batch deadline" in failed["fetch_error"]
    )
    assert failed["completed"] is False
    store.close()
    store = ArchiveCatalogue(path)
    assert store.summary()["failed_variables"] == 4
    retried = []

    def healthy(role, pending, cached):
        retried.extend(pending)
        return {"entries": [available(name) for name in pending]}

    resumed = store.scan(healthy)
    assert resumed["complete"] and resumed["failed_variables"] == 0
    assert retried == names[:4]
    store.close()


def test_timeout_restart_reaps_only_its_reader_and_accepts_expected_exit():
    readers = MetadataReaders.__new__(MetadataReaders)
    readers.local = threading.local()
    readers.lock = threading.Lock()
    process = MagicMock(pid=123, returncode=-15)
    process.poll.return_value = None
    process.stdin.closed = False
    session = {"role": "slow", "process": process, "buffer": bytearray()}
    readers.local.session = session
    readers.sessions = [session]
    readers._restart()
    process.terminate.assert_called_once()
    process.wait.assert_called_once()
    assert readers.local.session is None and readers.sessions == []
    readers.close()


def test_unexpected_failure_receipt_reports_current_checkpoint_not_start(tmp_path):
    path = tmp_path / "checkpoint.sqlite"
    store = ArchiveCatalogue(path)
    store.seed({"slow": ["DEMO:ONE", "DEMO:TWO"]}, timestamp="fixture")
    store.close()

    class Readers:
        settings = SimpleNamespace(readable_roles=["slow"])
        calls = 0

        def __init__(self, config):
            pass

        def fetch(self, role, names, cached):
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("fixture unexpected failure")
            return {"entries": [available(name) for name in names]}

        def close(self):
            pass

    receipt = tmp_path / "receipt.json"
    with pytest.raises(RuntimeError, match="unexpected failure"):
        run_scan(
            store_path=path,
            config={},
            receipt_path=receipt,
            batch_size=1,
            reader_factory=Readers,
        )
    result = json.loads(receipt.read_text())
    assert result["status"] == "failed"
    assert result["before"]["completed"] == 0 and result["result"]["completed"] == 1
