"""A failed remote file scan leaves its paths available for another attempt."""

import asyncio
import json
import subprocess
from unittest.mock import patch

import pytest

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
        patch("imas_codex.remote.executor.run_python_script", side_effect=output),
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
