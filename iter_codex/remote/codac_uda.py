"""Bounded archive reads shared by catalogue validity checks and signal loading.

The native client runs only through the facility-aware packaged-script executor.
Graph access resolves identities and metadata; private facility configuration
resolves endpoints and operator exclusions. No graph template is executed.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shlex
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from iter_codex.config.codac_uda import (
    archive_host_key,
    create_archive_pulse,
    resolve_archive_endpoint,
    validate_codac_uda_config,
)
from iter_codex.discovery.base.facility import get_facility_with_infrastructure
from iter_codex.remote.environment import resolve_remote_environment
from iter_codex.remote.tools import run_python_script


class ArchiveReadError(RuntimeError):
    """A classified refusal shared by scanner and loader callers."""

    def __init__(self, category: str, message: str):
        super().__init__(message)
        self.category = category


@dataclass(frozen=True)
class Signal:
    """Typed samples and their sampling, query and cache evidence."""

    time: np.ndarray
    values: np.ndarray
    metadata: dict[str, Any]


def _decode(result: dict) -> Signal:
    try:
        times = np.asarray(result["time"], dtype=result["time_dtype"])
        values = np.asarray(result["values"], dtype=result["value_dtype"])
        if times.ndim != 1 or values.ndim == 0 or len(times) != len(values):
            raise ValueError("timestamp and value dimensions disagree")
        expected = "iu" if result["time_base"] == "epoch_ns" else "f"
        if times.dtype.kind not in expected:
            raise ValueError("timestamp dtype disagrees with time base")
    except (KeyError, ValueError, TypeError) as exc:
        raise ArchiveReadError("invalid_receipt", str(exc)) from exc
    return Signal(
        times, values, {k: v for k, v in result.items() if k not in {"time", "values"}}
    )


def _classify(message: str, error_type: str = "") -> str:
    lower = message.lower()
    if "cadence" in lower or "timestamps" in lower or "spacing" in lower:
        return "sampling"
    if any(word in lower for word in ("bound", "limit", "exceed")):
        return "limit"
    if "empty" in lower:
        return "empty"
    if error_type in {"ValueError", "TypeError", "KeyError"}:
        return "invalid_request"
    return "reader"


class ArchiveReader:
    """Validate policy and budgets before running the packaged archive reader.

    Cache keys include the resolved endpoint, full selector, bounds and all
    sampling settings. Policy and request limits are checked on cache hits too.
    The scanner calls ``check`` on this same object; it has no separate fetch.
    """

    def __init__(
        self,
        facility: str,
        role: str,
        *,
        config: dict | None = None,
        cache_dir: Path | None = None,
    ):
        self.facility = facility
        self.facility_config = (
            config if config is not None else get_facility_with_infrastructure(facility)
        )
        data = self.facility_config.get("data_systems", {}).get("codac_uda", {})
        self.config = validate_codac_uda_config(data)
        self.endpoint = resolve_archive_endpoint(self.config, role)
        self.role = role
        self.archive = f"{facility}:codac_uda:{role}"
        self.cache_dir = (
            cache_dir or Path.home() / ".cache" / "iter-codex-codac-uda" / "signals"
        )

    def _request(self, operation: str, **parameters) -> dict:
        environment = resolve_remote_environment(self.facility_config, "codac_uda")
        excluded = [
            item.host for item in self.config.archives if item.excluded and item.host
        ]
        if archive_host_key(self.endpoint.host) in {
            archive_host_key(host) for host in excluded
        }:
            raise ArchiveReadError("policy", "archive endpoint is excluded")
        variables = {
            "CODAC_UDA_HOST": self.endpoint.host,
            "CODAC_UDA_PORT": str(self.endpoint.port),
            "CODAC_UDA_OPERATOR_HOSTS": json.dumps(excluded),
        }
        setup = list(environment.setup_commands) + [
            f"export {key}={shlex.quote(value)}" for key, value in variables.items()
        ]
        try:
            output = run_python_script(
                "codac_uda.py",
                input_data={"operation": operation, **parameters},
                facility=self.facility,
                data_system="codac_uda",
                ssh_host=self.facility_config.get("ssh_host"),
                python_command=environment.python_command,
                setup_commands=setup,
                timeout=max(1, math.ceil(self.config.timeout)),
            )
        except subprocess.CalledProcessError as exc:
            output = exc.stdout
            try:
                receipt = json.loads(output)
            except (ValueError, TypeError):
                raise ArchiveReadError(
                    "transport", "reader process failed without a JSON receipt"
                ) from exc
            message = receipt.get("error", "reader process failed")
            raise ArchiveReadError(
                _classify(message, receipt.get("error_type", "")), message
            ) from exc
        except (subprocess.TimeoutExpired, TimeoutError) as exc:
            raise ArchiveReadError("timeout", "archive reader timed out") from exc
        try:
            receipt = json.loads(output)
            if not receipt["ok"]:
                message = receipt.get("error", "archive read failed")
                raise ArchiveReadError(
                    _classify(message, receipt.get("error_type", "")), message
                )
            return receipt["result"]
        except (ValueError, KeyError, TypeError) as exc:
            raise ArchiveReadError(
                "invalid_receipt", "reader returned invalid JSON receipt"
            ) from exc

    def read(self, name: str, **parameters) -> Signal:
        """Read an absolute interval or a pulse with an explicit bounded budget."""
        request = {"name": name, **parameters}
        allowed = {
            "name",
            "field",
            "start",
            "end",
            "pulse",
            "time_format",
            "mode",
            "max_samples",
            "max_bytes",
            "page_samples",
            "max_pages",
            "bins",
            "expected_spacing",
            "semantics",
            "lookback_ns",
        }
        if set(request) - allowed:
            raise ArchiveReadError(
                "invalid_request", "unsupported archive read parameter"
            )
        maximum = request.setdefault("max_samples", self.config.max_samples)
        byte_limit = request.setdefault("max_bytes", self.config.max_bytes)
        for value, ceiling in (
            (maximum, self.config.max_samples),
            (byte_limit, self.config.max_bytes),
        ):
            if type(value) is not int or value < 1 or value > ceiling:
                raise ArchiveReadError(
                    "limit", "requested budget exceeds configured limit"
                )
        start, end = request.get("start"), request.get("end")
        if start is None or end is None:
            raise ArchiveReadError(
                "invalid_request",
                "resolved absolute or relative pulse bounds are required",
            )
        time_format = request.get("time_format", "absolute")
        if time_format not in {"absolute", "relative"}:
            raise ArchiveReadError("invalid_request", "unsupported timestamp format")
        if time_format == "absolute" and (
            type(start) is not int or type(end) is not int
        ):
            raise ArchiveReadError(
                "invalid_request", "absolute bounds must be integer epoch nanoseconds"
            )
        if (
            isinstance(start, bool)
            or isinstance(end, bool)
            or not isinstance(start, (int, float))
            or not isinstance(end, (int, float))
            or not math.isfinite(start)
            or not math.isfinite(end)
            or start >= end
        ):
            raise ArchiveReadError(
                "invalid_request", "time bounds must be finite and increasing"
            )
        factor = 1 if time_format == "absolute" else 1_000_000_000
        lookback = request.get("lookback_ns", 0)
        if (
            type(lookback) is not int
            or lookback < 0
            or (end - start) * factor + lookback > self.config.max_window_ns
        ):
            raise ArchiveReadError(
                "limit", "requested window and lookback exceed configured bound"
            )
        request["max_window_ns"] = self.config.max_window_ns
        digest = hashlib.sha256(
            json.dumps(
                {
                    "archive": self.archive,
                    "endpoint": [self.endpoint.host, self.endpoint.port],
                    "request": request,
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()
        path = self.cache_dir / f"{digest}.json"
        if path.exists():
            try:
                if path.stat().st_size <= byte_limit:
                    result = json.loads(path.read_text())
                    signal = _decode(result)
                    if len(signal.time) > maximum:
                        raise ValueError("cache exceeds sample bound")
                    signal.metadata.update(cache_hit=True, cache_key=digest)
                    return signal
            except (ValueError, OSError, ArchiveReadError):
                pass
        result = self._request("read", **request)
        encoded = json.dumps(result, allow_nan=False)
        if len(encoded.encode()) > byte_limit:
            raise ArchiveReadError("limit", "serialized reply exceeds byte bound")
        signal = _decode(result)
        if len(signal.time) > maximum:
            raise ArchiveReadError("limit", "reply exceeds sample bound")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w", dir=self.cache_dir, delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(encoded)
        try:
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        signal.metadata.update(cache_hit=False, cache_key=digest)
        return signal

    def check(self, name: str, **parameters) -> dict:
        """Return the same read or classified refusal for scanner validity checks."""
        try:
            signal = self.read(name, **parameters)
        except ArchiveReadError as exc:
            return {
                "success": False,
                "error_classification": exc.category,
                "error": str(exc),
            }
        if not len(signal.time):
            return {
                "success": False,
                "error_classification": "empty",
                "error": "archive interval is empty",
                "signal": signal,
            }
        return {"success": True, "signal": signal}

    def pulse_categories(self, *, limit: int = 10000) -> list[str]:
        """List all archive-declared categories, including empty categories."""
        return self._request("pulse_categories", limit=limit)["categories"]

    def pulses(self, pattern: str, *, limit: int = 1000):
        """Construct validated graph pulse identities from an archive listing."""
        result = self._request("pulses", pattern=pattern, limit=limit)
        return [
            create_archive_pulse(
                facility_id=self.facility,
                archive=self.archive,
                name=item["id"],
                start_ns=item["start_ns"],
                end_ns=item.get("end_ns"),
                description=item.get("description"),
                status=item.get("status"),
                categories=[
                    f"{self.archive}:{category}"
                    for category in result["categories"]
                    if item["id"].startswith(category + "/")
                ],
            )
            for item in result["pulses"]
        ]


SIGNAL_QUERY = """
MATCH (s:FacilitySignal {id: $signal_id})-[:DATA_ACCESS]->(access:DataAccess)
MATCH (source:DataSource)-[:HOSTED_BY_ARCHIVE]->(archive:ArchiveEndpoint)
WHERE source.facility_id = s.facility_id
  AND (source.id = access.data_source OR source.name = access.data_source)
RETURN s.id AS signal_id, s.facility_id AS facility_id, s.accessor AS selector,
       s.unit AS unit, s.description AS description, archive.id AS archive_id,
       archive.role AS archive_role, access.method_type AS method_type
LIMIT 2
"""
PULSE_QUERY = """
MATCH (pulse:Pulse {id: $pulse_id})
RETURN pulse.archive AS archive, pulse.start_ns AS start_ns,
       pulse.end_ns AS end_ns, pulse.name AS name
LIMIT 2
"""


class ArchiveSignalLoader:
    """Resolve a graph identity and use the same bounded reader as the scanner."""

    def __init__(self, graph=None, *, reader_factory=ArchiveReader):
        self.graph = graph
        self.reader_factory = reader_factory

    def _query(self, query: str, **parameters):
        if self.graph is not None:
            return self.graph.query(query, **parameters)
        from iter_codex.graph.client import GraphClient

        with GraphClient() as graph:
            return graph.query(query, **parameters)

    def load(
        self,
        signal_id: str,
        *,
        start: int | float | None = None,
        end: int | float | None = None,
        pulse_id: str | None = None,
        **parameters,
    ) -> Signal:
        rows = self._query(SIGNAL_QUERY, signal_id=signal_id)
        if len(rows) != 1 or rows[0]["method_type"] != "codac_uda":
            raise ArchiveReadError(
                "identity", "signal must resolve to exactly one CODAC archive binding"
            )
        binding = rows[0]
        if "pulse" in parameters:
            raise ArchiveReadError("invalid_request", "resolve pulses using pulse_id")
        reader = self.reader_factory(binding["facility_id"], binding["archive_role"])
        if reader.archive != binding["archive_id"]:
            raise ArchiveReadError(
                "identity", "graph archive identity disagrees with configured role"
            )
        if pulse_id is not None:
            pulses = self._query(PULSE_QUERY, pulse_id=pulse_id)
            if len(pulses) != 1 or pulses[0]["archive"] != reader.archive:
                raise ArchiveReadError(
                    "identity", "pulse must belong to the signal archive"
                )
            pulse = pulses[0]
            relative = parameters.get("time_format", "absolute") == "relative"
            if start is None and end is None:
                if pulse["end_ns"] is None:
                    raise ArchiveReadError(
                        "limit", "open-ended pulse requires an explicit bounded window"
                    )
                start, end = (
                    (0.0, (pulse["end_ns"] - pulse["start_ns"]) / 1e9)
                    if relative
                    else (pulse["start_ns"], pulse["end_ns"])
                )
            if relative:
                parameters["pulse"] = pulse["name"]
        signal = reader.read(binding["selector"], start=start, end=end, **parameters)
        signal.metadata.update(
            signal_id=signal_id,
            facility_id=binding["facility_id"],
            archive=binding["archive_id"],
        )
        signal.metadata["unit"] = signal.metadata.get("unit") or binding["unit"] or ""
        signal.metadata["description"] = (
            signal.metadata.get("description") or binding["description"] or ""
        )
        return signal


def archive_config(facility: str, config: dict | None = None):
    """Load private endpoint settings explicitly and ignore discovery runtime keys."""
    if config is None:
        config = get_facility_with_infrastructure(facility)["data_systems"]["codac_uda"]
    return validate_codac_uda_config(
        {key: value for key, value in config.items() if not key.startswith("_")}
    )


def request_archive(
    facility: str, role: str, request: dict, *, config: dict | None = None
) -> dict:
    """Fetch catalogue metadata through the shared reader transport.

    Sample reads belong to ArchiveReader.read/check; this entry point accepts
    only listing and metadata operations, with one bounded metadata batch.
    """
    settings = archive_config(facility, config)
    payload = dict(request)
    operation = payload.pop("operation", None)
    if operation not in {"variables", "metadata"}:
        raise ValueError("catalogue requests accept only variables or metadata")
    if {"connection", "host", "endpoint"} & payload.keys():
        raise ValueError("archive connections come only from private configuration")
    if operation == "metadata":
        maximum = payload.setdefault(
            "metadata_batch_size", settings.metadata_batch_size
        )
        names = payload.get("names")
        if (
            type(maximum) is not int
            or not 0 < maximum <= settings.metadata_batch_size
            or not isinstance(names, list)
            or not 0 < len(names) <= maximum
        ):
            raise ValueError("archive metadata request exceeds configured batch limit")
    reader = ArchiveReader(
        facility,
        role,
        config={"data_systems": {"codac_uda": settings.model_dump(mode="json")}},
    )
    return reader._request(operation, **payload)
