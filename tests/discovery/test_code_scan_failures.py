"""A failed remote file scan leaves its paths available for another attempt."""

import asyncio
import json
import subprocess
from unittest.mock import patch

from imas_codex.discovery.base import reachability
from imas_codex.discovery.code import scanner, workers
from imas_codex.discovery.code.state import FileDiscoveryState


class ScanClaims:
    def __init__(self, state):
        self.state = state
        self.paths = [
            {"id": "first", "path": "/source/first", "score": 0.9},
            {"id": "second", "path": "/source/second", "score": 0.9},
        ]
        self.claimed = set()
        self.scanned = {}
        self.released = []

    def claim(self, *_args, **_kwargs):
        available = [
            path
            for path in self.paths
            if path["id"] not in self.claimed and path["id"] not in self.scanned
        ]
        self.claimed.update(path["id"] for path in available)
        return available

    def mark(self, path_id, file_count):
        self.scanned[path_id] = file_count

    def release(self, path_id):
        self.claimed.remove(path_id)
        self.released.append(path_id)
        if len(self.released) == len(self.paths):
            self.state.stop_requested = True


class FacilityGraph:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def ensure_facility(self, _facility):
        pass


def run_scan(monkeypatch, output):
    state = FileDiscoveryState(facility="sample")
    claims = ScanClaims(state)
    monkeypatch.setattr(
        "imas_codex.discovery.code.graph_ops.claim_paths_for_file_scan", claims.claim
    )
    monkeypatch.setattr(
        "imas_codex.discovery.code.graph_ops.mark_path_file_scanned", claims.mark
    )
    monkeypatch.setattr(
        "imas_codex.discovery.code.graph_ops.release_path_file_scan_claim",
        claims.release,
    )
    monkeypatch.setattr("imas_codex.graph.GraphClient", FacilityGraph)
    monkeypatch.setattr(scanner, "_get_pattern_categories", lambda _facility: {})
    with (
        patch("imas_codex.discovery.base.facility.get_facility", return_value={}),
        patch(
            "imas_codex.remote.executor.run_python_script",
            side_effect=output if isinstance(output, Exception) else None,
            return_value=output,
        ),
    ):
        asyncio.run(workers.scan_worker(state, batch_size=2))
    return claims, state


def assert_batch_claimable(claims):
    assert claims.scanned == {}
    assert claims.claimed == set()
    assert {path["id"] for path in claims.claim("sample")} == {"first", "second"}


def test_exit_255_releases_batch_without_scanning(monkeypatch):
    error = subprocess.CalledProcessError(255, "discover_files.py")
    claims, state = run_scan(monkeypatch, error)
    assert_batch_claimable(claims)
    assert state.scan_stats.errors == 2


def test_timeout_releases_batch_without_scanning(monkeypatch):
    error = subprocess.TimeoutExpired("discover_files.py", 300)
    claims, state = run_scan(monkeypatch, error)
    assert_batch_claimable(claims)
    assert state.scan_stats.errors == 2


def test_unparseable_output_releases_batch_without_scanning(monkeypatch):
    claims, state = run_scan(monkeypatch, "not json")
    assert_batch_claimable(claims)
    assert state.scan_stats.errors == 2


def test_reported_non_directory_is_recorded_as_empty(monkeypatch):
    output = json.dumps(
        [
            {"path": "/source/first", "error": "not_a_directory"},
            {"path": "/source/second", "files": []},
        ]
    )
    claims, state = run_scan(monkeypatch, output)
    assert claims.scanned == {"first": 0, "second": 0}
    assert claims.claimed == set()
    assert state.scan_stats.errors == 0


class ScanUntilScanned(ScanClaims):
    """Keep claims available across failures and stop once every path is scanned."""

    def release(self, path_id):
        self.claimed.remove(path_id)
        self.released.append(path_id)

    def mark(self, path_id, file_count):
        super().mark(path_id, file_count)
        if len(self.scanned) == len(self.paths):
            self.state.stop_requested = True


class SshMonitor:
    """Report SSH down for a fixed number of checks, then healthy."""

    def __init__(self, unhealthy_checks):
        self.unhealthy_checks = unhealthy_checks
        self.checks = 0

    def is_service_healthy(self, *names):
        assert names == ("ssh",)
        self.checks += 1
        return self.checks > self.unhealthy_checks


def run_retrying_scan(monkeypatch, outcomes, monitor=None):
    state = FileDiscoveryState(facility="sample")
    state.service_monitor = monitor
    claims = ScanUntilScanned(state)
    waits = []

    async def record_wait(_state, seconds):
        waits.append(seconds)

    monkeypatch.setattr(
        "imas_codex.discovery.code.graph_ops.claim_paths_for_file_scan", claims.claim
    )
    monkeypatch.setattr(
        "imas_codex.discovery.code.graph_ops.mark_path_file_scanned", claims.mark
    )
    monkeypatch.setattr(
        "imas_codex.discovery.code.graph_ops.release_path_file_scan_claim",
        claims.release,
    )
    monkeypatch.setattr("imas_codex.graph.GraphClient", FacilityGraph)
    monkeypatch.setattr(scanner, "_get_pattern_categories", lambda _facility: {})
    monkeypatch.setattr(reachability, "_sleep_unless_stopped", record_wait)
    with (
        patch("imas_codex.discovery.base.facility.get_facility", return_value={}),
        patch(
            "imas_codex.remote.executor.run_python_script", side_effect=outcomes
        ) as remote,
    ):
        asyncio.run(workers.scan_worker(state, batch_size=2))
    return claims, state, waits, remote


SCANNED = json.dumps(
    [
        {"path": "/source/first", "files": []},
        {"path": "/source/second", "files": []},
    ]
)


def test_unreachable_host_pauses_the_scan_instead_of_ending_it(monkeypatch):
    drop = subprocess.CalledProcessError(255, "discover_files.py")
    claims, state, waits, _remote = run_retrying_scan(
        monkeypatch, [drop] * 7 + [SCANNED]
    )
    assert claims.scanned == {"first": 0, "second": 0}
    assert not state.scan_phase.done
    assert waits == [2.0, 4.0, 8.0, 16.0, 32.0, 60.0, 60.0]


def test_scan_holds_while_the_monitor_reports_ssh_down(monkeypatch):
    drop = subprocess.TimeoutExpired("discover_files.py", 300)
    monitor = SshMonitor(unhealthy_checks=3)
    claims, _state, waits, remote = run_retrying_scan(
        monkeypatch, [drop, SCANNED], monitor=monitor
    )
    assert claims.scanned == {"first": 0, "second": 0}
    assert remote.call_count == 2
    assert waits == [2.0, 5.0, 5.0, 5.0]


def test_script_failure_still_stops_after_the_retry_limit(monkeypatch):
    broken = subprocess.CalledProcessError(1, "discover_files.py")
    claims, state, _waits, remote = run_retrying_scan(monkeypatch, [broken] * 5)
    assert claims.scanned == {}
    assert state.scan_phase.done
    assert remote.call_count == 5
