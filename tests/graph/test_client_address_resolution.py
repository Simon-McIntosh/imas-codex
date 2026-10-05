"""Address resolution for the default graph client.

The bolt address a default :class:`GraphClient` connects to is resolved by the
profile layer, which obtains SLURM service addresses from the shared owner in
:mod:`imas_codex.remote.locations` (``resolve_service_url`` /
``_service_url_for_slurm``). The client holds no discovery of its own.

Locality in the profile layer is decided from the facility's login-node
hostname patterns, which a SLURM compute node does not match, so the profile
layer would otherwise take its remote branch and return a loopback tunnel
endpoint that nothing is listening on. These tests pin that the client
inherits the shared owner's answer for a compute node rather than holding a
discovery copy of its own, and that an unreadable location surfaces instead of
being swallowed into a fallback address.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from imas_codex.graph import client as client_module, profiles as profiles_module
from imas_codex.remote import locations as loc
from imas_codex.remote.locations import LocationInfo

DIRECT_URI = "bolt://98dci4-gpu-0002:7687"
GPU_NODE = "98dci4-gpu-0002"
COMPUTE_CALLER = "98dci4-clu-2018"
EXPLICIT_URI = "bolt://example.invalid:1"


def _slurm_location() -> LocationInfo:
    return LocationInfo(
        name="titan",
        facility="iter",
        ssh_host="iter",
        scheduler="slurm",
        partition="titan",
        service_job_name="codex-neo4j",
        is_compute=True,
    )


def _plain_location() -> LocationInfo:
    return LocationInfo(name="iter", facility="iter", ssh_host="iter", scheduler="none")


@pytest.fixture(autouse=True)
def resolver_state_isolated(monkeypatch):
    """Start every case from un-cached resolution and a clean environment.

    The profile layer caches resolved URIs and the locations layer caches
    service URLs, so a prior case's address would decide what this one
    observes. ``NEO4J_URI`` is the documented escape hatch and is applied by
    the profile layer last, so an ambient value would outrank the address
    under test; it is cleared unless a case sets it explicitly.
    """
    profiles_module._resolved_uri_cache.clear()
    loc._service_url_cache.clear()
    monkeypatch.delenv("NEO4J_URI", raising=False)
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    yield
    profiles_module._resolved_uri_cache.clear()
    loc._service_url_cache.clear()


def _pin_slurm(monkeypatch) -> None:
    """Send the profile layer down its SLURM-scheduled branch for a compute node."""
    monkeypatch.setattr(profiles_module, "get_graph_location", lambda: "titan")
    monkeypatch.setattr(loc, "resolve_location", lambda _l: _slurm_location())
    monkeypatch.setattr("imas_codex.remote.executor.is_local_host", lambda _h: False)


def test_client_address_is_the_shared_owners_answer_inside_a_slurm_step(monkeypatch):
    """The client's bolt address is the shared owner's answer, never its own.

    A compute node matches no login-node pattern, so the profile layer's
    hostname test would leave it on a loopback endpoint. The address the client
    connects to is the one the profile layer obtains from the shared SLURM
    owner; the client contributes no discovery of its own.
    """
    _pin_slurm(monkeypatch)
    with patch.object(loc, "_service_url_for_slurm", return_value=DIRECT_URI) as owner:
        uri = client_module._resolve_graph_uri()

    assert uri == DIRECT_URI
    owner.assert_called_once()
    kwargs = owner.call_args.kwargs
    assert kwargs["protocol"] == "bolt"
    assert kwargs["service_job_name"] == "codex-neo4j"


def test_explicit_neo4j_uri_wins_inside_a_slurm_step(monkeypatch):
    """The documented escape hatch outranks the scheduled address.

    The profile layer applies ``NEO4J_URI`` last, so the operator who sets it
    has named an address that the shared owner's discovery is not entitled to
    replace.
    """
    monkeypatch.setenv("NEO4J_URI", EXPLICIT_URI)
    _pin_slurm(monkeypatch)
    with patch.object(loc, "_service_url_for_slurm", return_value=DIRECT_URI):
        assert client_module._resolve_graph_uri() == EXPLICIT_URI


def test_the_client_default_uri_factory_is_the_profile_resolver(monkeypatch):
    """The dataclass default, not a bare profile URI, is what a client gets.

    Pins ``field(default_factory=_resolve_graph_uri)``: with the profile URI as
    the factory the repair is inert for a default ``GraphClient()``, which is
    how the loopback endpoint reached the SLURM step at all.
    """
    _pin_slurm(monkeypatch)
    with patch.object(loc, "_service_url_for_slurm", return_value=DIRECT_URI):
        factory = client_module.GraphClient.__dataclass_fields__["uri"].default_factory
        assert factory is client_module._resolve_graph_uri
        assert factory() == DIRECT_URI


def test_an_unreadable_location_is_not_swallowed(monkeypatch):
    """A failed location read surfaces instead of restoring a fallback address.

    Swallowing it would silently restore the loopback tunnel endpoint this
    resolution exists to avoid inside a SLURM step.
    """
    monkeypatch.setattr(profiles_module, "get_graph_location", lambda: "titan")

    def unreadable(_location):
        raise RuntimeError("location config unreadable")

    monkeypatch.setattr(loc, "resolve_location", unreadable)
    with pytest.raises(RuntimeError):
        client_module._resolve_graph_uri()


def test_a_compute_step_reaches_the_discovered_service_node(monkeypatch):
    """The end-to-end client address is the discovered service node's.

    Exercises the real shared owner rather than a stubbed return: with squeue
    answering the service node and the caller on a different compute node, the
    client's address is that node's, not the loopback the profile layer's
    hostname test would choose.
    """
    _pin_slurm(monkeypatch)
    monkeypatch.setattr(loc, "_resolve_compute_host", lambda *a, **k: None)
    monkeypatch.setattr(loc.socket, "gethostname", lambda: COMPUTE_CALLER)
    monkeypatch.setattr(
        "imas_codex.remote.tunnel.discover_compute_node_local", lambda **_: GPU_NODE
    )

    expected = f"bolt://{GPU_NODE}:{profiles_module._convention_bolt_port('titan')}"
    assert client_module._resolve_graph_uri() == expected


def test_non_slurm_location_uses_the_direct_loopback(monkeypatch):
    """A local, non-scheduled location never consults the SLURM owner."""
    monkeypatch.setattr(profiles_module, "get_graph_location", lambda: "iter")
    monkeypatch.setattr(loc, "resolve_location", lambda _l: _plain_location())
    monkeypatch.setattr("imas_codex.remote.executor.is_local_host", lambda _h: True)

    with patch.object(loc, "_service_url_for_slurm") as owner:
        uri = client_module._resolve_graph_uri()

    assert uri.startswith("bolt://localhost:")
    owner.assert_not_called()
