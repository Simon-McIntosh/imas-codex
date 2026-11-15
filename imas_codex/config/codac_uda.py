"""Validate archive policy and resolve a readable role before connecting.

LinkML-derived models validate field types and the readable-role vocabulary.
These cross-endpoint checks also prevent an allowed role from aliasing an
excluded host. Public configuration can omit hosts; connection resolution
requires the private operator exclusion as well as the requested archive.
Archive lists replace public entries wholesale during facility configuration
merging, so private archive blocks must redeclare all roles and exclusions.
Production readers share the host comparison through :func:`archive_host_key`.
"""

from __future__ import annotations

import ipaddress
import re
from collections.abc import Mapping
from typing import Any

from imas_codex.config.models import (
    ArchiveEndpointConfig,
    ArchiveRole,
    CodacUDAConfig,
)


def archive_host_key(host: str) -> str:
    """Canonical archive policy key: DNS short aliases or exact IP addresses.

    Readers use this comparison rather than deriving exclusion keys themselves.
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
    return config


def resolve_archive_endpoint(
    data: Mapping[str, Any] | CodacUDAConfig, role: str
) -> ArchiveEndpointConfig:
    """Return a configured endpoint only after its complete read policy passes."""
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
