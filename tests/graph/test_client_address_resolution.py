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
from imas_codex.graph.client import GraphClient, _resolve_graph_uri, _slurm_service_uri

PROFILE_URI = "bolt://localhost:17687"
DIRECT_URI = "bolt://98dci4-gpu-0002:7687"
STEP_HOST = "98dci4-clu-3141"
EXPLICIT_URI = "bolt://example.invalid:1"


@dataclass
class Info:
    scheduler: str
    service_job_name: str = "codex-neo4j"


def pin(monkeypatch, *, scheduler, node, host=STEP_HOST, port=7687, uri=PROFILE_URI):
    monkeypatch.setattr(
        client_module, "resolve_neo4j", lambda **_: type("P", (), {"bolt_port": port})()
    )
    monkeypatch.setattr(client_module, "get_graph_uri", lambda: uri)
    monkeypatch.setattr("imas_codex.graph.profiles.get_graph_location", lambda: "iter")
    monkeypatch.setattr(
        "imas_codex.remote.locations.resolve_location", lambda _l: Info(scheduler)
    )
    monkeypatch.setattr(
        "imas_codex.remote.tunnel.discover_compute_node_local", lambda **_: node
    )
    monkeypatch.setattr(socket, "gethostname", lambda: host + ".iter.org")


@pytest.fixture(autouse=True)
def resolver_environment_isolated(monkeypatch):
    """Start every case from neither environment input the resolver reads.

    ``_resolve_graph_uri`` consults ``NEO4J_URI`` and ``SLURM_JOB_ID`` from
    the process environment, so a shell that exports either -- CI, or an
    operator using the documented escape hatch -- would decide what these
    cases observe instead of the code under test. A case that exercises one
    of them sets it explicitly; the ambient value never reaches the resolver.
    """
    monkeypatch.delenv("NEO4J_URI", raising=False)
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)


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
    monkeypatch.setattr(
        "imas_codex.remote.locations._resolve_compute_host", lambda *_a, **_k: None
    )
    assert _slurm_service_uri() is None


def test_the_fallback_is_read_from_the_module_it_is_imported_from(monkeypatch):
    """A patch on the client module cannot reach the fallback.

    ``_resolve_compute_host`` is imported inside the function body from
    ``imas_codex.remote.locations``, so it is that module's attribute the
    function calls. Answering from there is what shows which one is live.
    """
    pin(monkeypatch, scheduler="slurm", node=None)
    monkeypatch.setattr(
        "imas_codex.remote.locations._resolve_compute_host",
        lambda *_a, **_k: "98dci4-gpu-0002",
    )
    assert _slurm_service_uri() == DIRECT_URI


def test_explicit_neo4j_uri_wins_inside_a_slurm_step(monkeypatch):
    """The documented escape hatch outranks the scheduled address.

    ``resolve_neo4j`` applies ``NEO4J_URI`` last, so the URI the client
    starts from is already the explicit one; the SLURM branch must leave it
    alone rather than replacing it with an address it discovered.
    """
    monkeypatch.setenv("SLURM_JOB_ID", "12345")
    monkeypatch.setenv("NEO4J_URI", EXPLICIT_URI)
    pin(monkeypatch, scheduler="slurm", node="98dci4-gpu-0002", uri=EXPLICIT_URI)
    assert _resolve_graph_uri() == EXPLICIT_URI


def test_an_unreadable_location_is_not_swallowed(monkeypatch):
    """A failed location read surfaces instead of restoring the tunnel.

    Swallowing it returns ``None``, which leaves the profile's loopback
    tunnel endpoint in place -- the address this module exists to avoid
    inside a SLURM step, put back silently.
    """
    monkeypatch.setenv("SLURM_JOB_ID", "12345")
    pin(monkeypatch, scheduler="slurm", node="98dci4-gpu-0002")

    def unreadable(_location):
        raise RuntimeError("location config unreadable")

    monkeypatch.setattr("imas_codex.remote.locations.resolve_location", unreadable)
    with pytest.raises(RuntimeError):
        _slurm_service_uri()


def test_the_client_default_uri_is_the_slurm_aware_resolver(monkeypatch):
    """The dataclass default, not the profile URI, is what a client gets.

    Pins ``field(default_factory=_resolve_graph_uri)``: with the profile URI
    as the factory the whole repair is inert for a default ``GraphClient()``,
    which is how the loopback endpoint reached the SLURM step at all.
    """
    pin(monkeypatch, scheduler="slurm", node="98dci4-gpu-0002")
    monkeypatch.setenv("SLURM_JOB_ID", "12345")
    factory = GraphClient.__dataclass_fields__["uri"].default_factory
    assert factory is _resolve_graph_uri
    assert factory() == DIRECT_URI


def test_non_slurm_location_returns_none(monkeypatch):
    pin(monkeypatch, scheduler="none", node="98dci4-gpu-0002")
    assert _slurm_service_uri() is None
