"""Validate archive policy and resolve a readable role before connecting.

LinkML-derived models validate field types and the readable-role vocabulary.
These cross-endpoint checks also prevent an allowed role from aliasing an
excluded host. Public configuration can omit hosts; connection resolution
requires the private operator exclusion as well as the requested archive.
Archive lists replace public entries wholesale during facility configuration
merging, so private archive blocks must redeclare all roles and exclusions.
Configuration validation uses :func:`archive_host_key` for its host checks.
:func:`create_archive_pulse` derives and validates graph pulse identities from
the complete category-qualified archive name.
"""

from __future__ import annotations

import ipaddress
import re
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from iter_codex.config.models import ArchiveEndpointConfig, CodacUDAConfig
    from iter_codex.graph.models import Pulse


def create_archive_pulse(
    *,
    facility_id: str,
    archive: str,
    name: str,
    start_ns: int,
    end_ns: int | None,
    categories: list[str] | None = None,
    description: str | None = None,
    status: str | None = None,
    id: str | None = None,
) -> Pulse:
    """Construct a pulse using its complete archive-reported identity.

    The category-qualified name is preserved byte for byte in both the name
    and derived graph ID. A supplied ID is checked rather than overwritten,
    so importing an existing record cannot silently hide a namespace mismatch.
    Construct scanner pulse records here instead of using the structural model.
    """
    from iter_codex.graph.models import Pulse

    pulse = Pulse(
        id=f"{archive}:{name}",
        facility_id=facility_id,
        archive=archive,
        name=name,
        start_ns=start_ns,
        end_ns=end_ns,
        categories=categories,
        description=description or None,
        status=status or None,
    )
    if "/" not in pulse.name or any(not segment for segment in pulse.name.split("/")):
        raise ValueError("pulse name must include its archive category and pulse path")
    if id is not None and id != pulse.id:
        raise ValueError(
            "pulse id disagrees with archive and full archive-reported name"
        )
    return pulse


def archive_host_key(host: str) -> str:
    """Canonical archive policy key: DNS short aliases or exact IP addresses.

    Endpoint policy validation calls this function for each configured host.
    IP address normalization preserves all address components; DNS names use
    their short alias so case and a domain suffix cannot evade an exclusion.
    """
    hostname = host.strip().rstrip(".").lower()
    try:
        return str(ipaddress.ip_address(hostname))
    except ValueError:
        if not hostname or not re.fullmatch(r"[a-z0-9_-]+(?:\.[a-z0-9_-]+)*", hostname):
            raise ValueError("archive host must be a hostname or IP address") from None
        return hostname.split(".", 1)[0]


def validate_codac_uda_config(
    data: Mapping[str, Any] | CodacUDAConfig,
) -> CodacUDAConfig:
    """Validate structure and exclusions without importing a client or connecting."""
    from iter_codex.config.models import ArchiveRole, CodacUDAConfig

    config = CodacUDAConfig.model_validate(
        data.model_dump() if isinstance(data, CodacUDAConfig) else dict(data)
    )
    endpoints = config.archives or []
    roles = [endpoint.role for endpoint in endpoints]
    if len(roles) != len(set(roles)):
        raise ValueError("archive roles must be unique")
    if not any(
        endpoint.role == ArchiveRole.operator and endpoint.excluded
        for endpoint in endpoints
    ):
        raise ValueError(
            "operator archive must be explicitly excluded; archive lists replace "
            "public entries wholesale, so a private block must redeclare every "
            "role, including operator with excluded: true"
        )

    excluded_hosts = {
        archive_host_key(endpoint.host)
        for endpoint in endpoints
        if endpoint.excluded and endpoint.host is not None
    }
    for endpoint in endpoints:
        if endpoint.host is None:
            continue
        host_key = archive_host_key(endpoint.host)
        if not endpoint.excluded and host_key in excluded_hosts:
            raise ValueError("readable archive endpoint is reserved or excluded")
    for role in config.readable_roles or []:
        endpoint = next((item for item in endpoints if item.role == role), None)
        if endpoint is None:
            raise ValueError(
                "readable archive role must name a non-excluded endpoint; archive "
                "lists replace public entries wholesale, so a private block must "
                "redeclare every role selected by readable_roles"
            )
        if endpoint.excluded:
            raise ValueError("readable archive role must name a non-excluded endpoint")
    if set(roles) != set(ArchiveRole):
        raise ValueError("a private archive block must redeclare every archive role")
    return config


def resolve_archive_endpoint(
    data: Mapping[str, Any] | CodacUDAConfig, role: str
) -> ArchiveEndpointConfig:
    """Return a configured endpoint only after its complete read policy passes."""
    from iter_codex.config.models import ArchiveRole

    config = validate_codac_uda_config(data)
    if role not in (config.readable_roles or []):
        raise ValueError("archive role is not readable")
    operator = next(
        item for item in config.archives if item.role == ArchiveRole.operator
    )
    if not operator.host:
        raise ValueError("private operator endpoint exclusion must be configured")
    endpoint = next(item for item in config.archives if item.role == role)
    if not endpoint.host:
        raise ValueError("private archive endpoint must be configured")
    return endpoint
