"""Authenticated, read-only access to a configured ICP integration instance."""

from iter_codex.discovery.icp.client import (
    ICPAuthenticationError,
    ICPClient,
    ICPConfigurationError,
    ICPRequestRefused,
)
from iter_codex.discovery.icp.records import (
    ICPNotFound,
    ICPPermissionDenied,
    ICPReadLimit,
    ICPRecords,
    ICPResponseError,
)

__all__ = [
    "ICPAuthenticationError",
    "ICPClient",
    "ICPConfigurationError",
    "ICPRequestRefused",
    "ICPNotFound",
    "ICPPermissionDenied",
    "ICPReadLimit",
    "ICPRecords",
    "ICPResponseError",
]
