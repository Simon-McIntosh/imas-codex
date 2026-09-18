"""Address resolution for the default graph client inside a SLURM step.

Locality in the profile layer is decided from the facility's login-node
hostname patterns, which a SLURM compute node does not match, so profile
resolution takes its remote branch there and returns a loopback tunnel
endpoint that nothing is listening on. These tests pin the client-side
resolution to the reachable service address, and pin the branches that
must not change.
"""

from __future__ import annotations

import socket
from dataclasses import dataclass

import pytest

from imas_codex.graph import client as client_module
from imas_codex.graph.client import _resolve_graph_uri, _slurm_service_uri

PROFILE_URI = "bolt://localhost:17687"
DIRECT_URI = "bolt://98dci4-gpu-0002:7687"
STEP_HOST = "98dci4-clu-3141"


@dataclass
class Info:
    scheduler: str
    service_job_name: str = "codex-neo4j"


def pin(monkeypatch, *, scheduler, node, host=STEP_HOST, port=7687):
    monkeypatch.setattr(
        client_module, "resolve_neo4j", lambda **_: type("P", (), {"bolt_port": port})()
    )
    monkeypatch.setattr("imas_codex.graph.profiles.get_graph_location", lambda: "iter")
    monkeypatch.setattr(
        "imas_codex.remote.locations.resolve_location", lambda _l: Info(scheduler)
    )
    monkeypatch.setattr(
        "imas_codex.remote.tunnel.discover_compute_node_local", lambda **_: node
    )
    monkeypatch.setattr(socket, "gethostname", lambda: host + ".iter.org")


def test_direct_address_outside_a_slurm_step(monkeypatch):
    """The profile URI stands, and no node discovery is attempted."""
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.setattr(client_module, "get_graph_uri", lambda: PROFILE_URI)
    monkeypatch.setattr(
        client_module, "_slurm_service_uri", lambda: pytest.fail("must not resolve")
    )
    assert _resolve_graph_uri() == PROFILE_URI


def test_peer_service_node_yields_its_direct_address(monkeypatch):
    pin(monkeypatch, scheduler="slurm", node="98dci4-gpu-0002")
    assert _slurm_service_uri() == DIRECT_URI


def test_service_node_on_this_node_yields_localhost(monkeypatch):
    pin(monkeypatch, scheduler="slurm", node=STEP_HOST)
    assert _slurm_service_uri() == "bolt://localhost:7687"


def test_no_service_node_returns_none(monkeypatch):
    """No invented address: the caller keeps whatever it had."""
    pin(monkeypatch, scheduler="slurm", node=None)
    monkeypatch.setattr(client_module, "_resolve_compute_host", None, raising=False)
    monkeypatch.setattr(
        "imas_codex.remote.locations._resolve_compute_host", lambda *_a, **_k: None
    )
    assert _slurm_service_uri() is None


def test_non_slurm_location_returns_none(monkeypatch):
    pin(monkeypatch, scheduler="none", node="98dci4-gpu-0002")
    assert _slurm_service_uri() is None
