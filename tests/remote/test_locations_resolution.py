"""Service-URL resolution for cluster and remote callers.

``resolve_service_url`` owns SLURM service-node discovery: it decides by
discovery, not by whether the caller's hostname counts as local.  A compute
node reaches the SLURM-hosted service directly rather than falling through
to a loopback address behind a tunnel it cannot open.
"""

from __future__ import annotations

import builtins
from unittest.mock import patch

import pytest

from imas_codex.remote import locations as loc
from imas_codex.remote.locations import LocationInfo

GPU_NODE = "98dci4-gpu-0002"
COMPUTE_CALLER = "98dci4-clu-2018"
PORT = 18765
BOLT_PORT = 7687


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
def _clear_url_cache():
    loc._service_url_cache.clear()
    yield
    loc._service_url_cache.clear()


def _resolve(location, info, *, local, hostname, discovered=None, compute_host=None):
    """Call the resolver with squeue and the hostname stubbed."""
    with (
        patch.object(loc, "resolve_location", return_value=info),
        patch.object(loc, "is_location_local", return_value=local),
        patch.object(loc, "_resolve_compute_host", return_value=compute_host),
        patch(
            "imas_codex.remote.tunnel.discover_compute_node_local",
            return_value=discovered,
        ) as discover,
        patch.object(loc.socket, "gethostname", return_value=hostname),
    ):
        url = loc.resolve_service_url(location, PORT, service_job_name="codex-embed")
    return url, discover


def test_compute_node_caller_reaches_discovered_service_node():
    """A compute node is served the service node, not a loopback tunnel."""
    url, discover = _resolve(
        "titan",
        _slurm_location(),
        local=False,
        hostname=COMPUTE_CALLER,
        discovered=GPU_NODE,
    )
    assert url == f"http://{GPU_NODE}:{PORT}"
    discover.assert_called_once_with(service_job_name="codex-embed")


def test_service_node_caller_collapses_to_loopback():
    """The caller on the service node itself uses the loopback."""
    url, _ = _resolve(
        "titan",
        _slurm_location(),
        local=False,
        hostname=GPU_NODE,
        discovered=GPU_NODE,
    )
    assert url == f"http://localhost:{PORT}"


def test_login_node_caller_falls_back_to_loopback_when_undiscovered():
    """A local caller keeps the loopback when squeue finds nothing."""
    url, _ = _resolve(
        "titan",
        _slurm_location(),
        local=True,
        hostname="98dci4-srv-1006",
        discovered=None,
        compute_host=None,
    )
    assert url == f"http://localhost:{PORT}"


def test_remote_caller_uses_loopback_for_tunnel_when_undiscovered():
    """An off-cluster caller falls through to the tunnel branch."""
    url, _ = _resolve(
        "titan",
        _slurm_location(),
        local=False,
        hostname="some-workstation",
        discovered=None,
        compute_host=None,
    )
    assert url == f"http://localhost:{PORT}"


def test_slurm_discovery_uses_recorded_host_when_no_job_visible():
    """When squeue is empty the recorded compute host is used."""
    url, _ = _resolve(
        "titan",
        _slurm_location(),
        local=True,
        hostname="98dci4-srv-1006",
        discovered=None,
        compute_host=GPU_NODE,
    )
    assert url == f"http://{GPU_NODE}:{PORT}"


def test_non_slurm_local_location_is_unchanged():
    """A local, non-scheduled location resolves to localhost without discovery."""
    url, discover = _resolve(
        "iter",
        _plain_location(),
        local=True,
        hostname=COMPUTE_CALLER,
        discovered=None,
    )
    assert url == f"http://localhost:{PORT}"
    discover.assert_not_called()


def test_graph_and_embedding_jobs_may_sit_on_different_nodes():
    """Each service resolves through its own job name's node.

    The graph and embedding services are separate SLURM jobs and may run on
    different compute nodes. A caller must be sent to the node of the job it
    asked for, never redirected to the other service's node.
    """

    def node_for(*, service_job_name):
        return {
            "codex-neo4j": "98dci4-gpu-0001",
            "codex-embed": GPU_NODE,
        }[service_job_name]

    with (
        patch.object(loc, "resolve_location", return_value=_slurm_location()),
        patch.object(loc, "is_location_local", return_value=False),
        patch.object(loc, "_resolve_compute_host", return_value=None),
        patch.object(loc.socket, "gethostname", return_value=COMPUTE_CALLER),
        patch(
            "imas_codex.remote.tunnel.discover_compute_node_local",
            side_effect=node_for,
        ),
    ):
        graph_url = loc.resolve_service_url(
            "titan", BOLT_PORT, protocol="bolt", service_job_name="codex-neo4j"
        )
        embed_url = loc.resolve_service_url(
            "titan", PORT, protocol="http", service_job_name="codex-embed"
        )

    assert graph_url == f"bolt://98dci4-gpu-0001:{BOLT_PORT}"
    assert embed_url == f"http://{GPU_NODE}:{PORT}"


@pytest.mark.parametrize("resolver", [loc._find_compute_location, loc.resolve_location])
def test_facility_import_error_propagates(resolver):
    """A broken facility import must be visible to the caller."""
    original_import = builtins.__import__

    def fail_facility_import(name, *args, **kwargs):
        if name == "imas_codex.discovery.base.facility":
            raise ImportError("injected facility import failure")
        return original_import(name, *args, **kwargs)

    loc.resolve_location.cache_clear()
    with (
        patch("builtins.__import__", side_effect=fail_facility_import),
        pytest.raises(ImportError, match="injected facility import failure") as error,
    ):
        resolver("titan")
    assert "while resolving location 'titan'" in error.value.__notes__


def test_unknown_facility_keeps_direct_location_fallback():
    """A location without a facility config remains a direct SSH location."""
    loc.resolve_location.cache_clear()
    info = loc.resolve_location("missing-facility-for-location-test")

    assert info.name == "missing-facility-for-location-test"
    assert info.facility == info.ssh_host == info.name
    assert info.scheduler == "none"
    assert not info.is_compute


def test_compute_location_propagates_facility_read_error():
    """A present facility with a broken config read is not an unknown one."""
    with (
        patch(
            "imas_codex.discovery.base.facility.list_facilities", return_value=["iter"]
        ),
        patch(
            "imas_codex.discovery.base.facility.get_facility",
            side_effect=RuntimeError("facility read failed"),
        ),
        pytest.raises(RuntimeError, match="facility read failed") as error,
    ):
        loc._find_compute_location("titan")
    assert "while resolving location 'titan'" in error.value.__notes__


def test_direct_location_propagates_facility_read_error():
    """A broken direct facility config must reach the caller."""
    loc.resolve_location.cache_clear()
    with (
        patch.object(loc, "_find_compute_location", return_value=None),
        patch(
            "imas_codex.discovery.base.facility.get_facility",
            side_effect=RuntimeError("facility read failed"),
        ),
        pytest.raises(RuntimeError, match="facility read failed") as error,
    ):
        loc.resolve_location("iter")
    assert "while resolving location 'iter'" in error.value.__notes__
