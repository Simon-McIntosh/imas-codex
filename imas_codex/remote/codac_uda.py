"""Resolve private archive policy and call the packaged reader through facility tools."""

from __future__ import annotations

import json
import subprocess
from typing import Any

from imas_codex.config.codac_uda import (
    resolve_archive_endpoint,
    validate_codac_uda_config,
)
from imas_codex.discovery.base.facility import get_facility_with_infrastructure
from imas_codex.remote.tools import run_python_script


class ArchiveRequestError(RuntimeError):
    """A structured archive refusal, preserving the reader's error classification."""

    def __init__(self, error_type: str, message: str):
        self.error_type = error_type
        super().__init__(message)


def archive_config(facility: str, config: dict | None = None):
    """Load private endpoint settings explicitly and ignore discovery runtime keys."""
    if config is None:
        config = get_facility_with_infrastructure(facility)["data_systems"]["codac_uda"]
    return validate_codac_uda_config(
        {key: value for key, value in config.items() if not key.startswith("_")}
    )


def request_archive(
    facility: str, role: str, request: dict[str, Any], *, config: dict | None = None
) -> dict:
    """Run one bounded request after refusing excluded endpoints locally.

    A host or connection override is never accepted from a reader request.
    The packaged reader repeats operator policy before native construction.
    """
    settings = archive_config(facility, config)
    endpoint = resolve_archive_endpoint(settings, role)
    if "connection" in request or "host" in request or "endpoint" in request:
        raise ValueError("archive connections come only from private configuration")
    payload = dict(request)
    if payload.get("operation") == "read":
        start, end = payload.get("start"), payload.get("end")
        if (
            type(start) is not int
            or type(end) is not int
            or not 0 < end - start <= settings.max_window_ns
        ):
            raise ValueError(
                "archive read requires a bounded absolute window within configured limits"
            )
        if (
            not 0
            < payload.get("max_samples", settings.max_samples)
            <= settings.max_samples
        ):
            raise ValueError("archive read exceeds configured sample limit")
        payload.setdefault("max_samples", settings.max_samples)
    payload["connection"] = {
        "host": endpoint.host,
        "port": endpoint.port,
        "operator_hosts": [
            item.host for item in settings.archives if item.excluded and item.host
        ],
    }
    try:
        output = run_python_script(
            "codac_uda.py",
            payload,
            facility=facility,
            data_system="codac_uda",
            timeout=settings.timeout,
            setup_commands=settings.setup_commands,
            python_command=settings.python_command,
        )
    except subprocess.CalledProcessError as exc:
        try:
            receipt = json.loads(exc.stdout)
        except (ValueError, TypeError):
            raise ArchiveRequestError(
                "ReaderProcessError", "archive reader failed without a receipt"
            ) from exc
        raise ArchiveRequestError(
            receipt.get("error_type", "ReaderError"),
            receipt.get("error", "archive reader refused request"),
        ) from exc
    if len(output.encode()) > settings.max_bytes:
        raise ArchiveRequestError(
            "ReadError", "archive reader receipt exceeds configured byte limit"
        )
    receipt = json.loads(output)
    if receipt.get("ok") is not True:
        raise ArchiveRequestError(
            receipt.get("error_type", "ReaderError"),
            receipt.get("error", "archive reader refused request"),
        )
    return receipt["result"]
