"""Standalone CODAC reader: one JSON request on stdin, one JSON receipt on stdout.

Run with the module-provided interpreter. No project imports are required.
Archive addresses and operator exclusions come from deployment configuration.
"""

from __future__ import annotations

import contextlib
import json
import os
import sys
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any

import numpy as np


class ReadError(RuntimeError):
    """A refused read, invalid sampling cadence, or exhausted read bound."""


def _host_key(host: str) -> str:
    return host.strip().rstrip(".").split(".", 1)[0].lower()


def connect(host: str, operator_hosts: list[str], port: int = 3090, factory=None):
    """Refuse operator endpoints before importing or constructing the client.

    A deployment must supply the exclusion list, even when injecting a client.
    Short labels are compared so case, a domain and a trailing dot cannot evade it.
    """
    if not operator_hosts or not all(_host_key(item) for item in operator_hosts):
        raise ValueError("operator host exclusions must be configured")
    if not host or any(char in host for char in "/:@ \t\n"):
        raise ValueError("host must be a DNS name or IPv4 address without a port")
    if _host_key(host) in {_host_key(item) for item in operator_hosts}:
        raise ValueError("archive endpoint is reserved for operators")
    if factory is None:
        from uda_client_reader.uda_client_reader_python import UdaClientReaderPython

        factory = UdaClientReaderPython
    return factory(host, port)


def _token(value: str) -> str:
    if not value or any(char in value for char in ",=\n\r"):
        raise ValueError("query tokens must be nonempty and contain no separators")
    return value


def _field(value: Any, source: str) -> dict:
    text = str(value).strip() if value is not None else ""
    return {"value": text or None, "source": source if text else None}


def variables(client, request: dict) -> dict:
    """Merge declared metadata field by field, retaining provenance and warnings."""
    pattern = _token(request["pattern"])
    names = list(client.getVariableList(pattern))
    limit = int(request.get("limit", 1000))
    if limit < 1 or len(names) > limit:
        raise ReadError("variable catalogue exceeds limit; narrow the pattern")
    metadata_url = request.get("metadata_url")
    rows = {}
    warnings = []
    if metadata_url:
        query = urllib.parse.urlencode(
            {
                "select": "name,unit,description",
                "name": f"like.{pattern.replace('*', '%')}",
            }
        )
        with urllib.request.urlopen(f"{metadata_url}?{query}", timeout=5) as response:
            rows = {str(row["name"]): row for row in json.load(response)}
    entries = []
    for item in names:
        name = str(item)
        unit, description = _field(None, "uda"), _field(None, "uda")
        try:
            document = json.loads(client.getMetaTypeJSONCollapsed(name))
            fields = document["datatypes"][0]["fields"]
            signal = next(
                (field for field in fields if field.get("name") == "val"), None
            )
            if signal is None and len(fields) == 1:
                signal = fields[0]
            if signal:
                unit = _field(signal.get("unit"), "uda")
                description = _field(signal.get("description"), "uda")
        except Exception as exc:
            warnings.append({"name": name, "source": "uda", "error": str(exc)})
        row = rows.get(name, {})
        if row.get("unit") and str(row["unit"]).strip():
            unit = _field(row["unit"], "controlsmetadata")
        if row.get("description") and str(row["description"]).strip():
            description = _field(row["description"], "controlsmetadata")
        entries.append({"name": name, "unit": unit, "description": description})
    return {"entries": entries, "warnings": warnings}


def pulses(client, request: dict) -> dict:
    """Serialize SWIG pulse objects explicitly rather than relying on repr()."""
    found = list(client.getPulses2(_token(request["pattern"])))
    limit = int(request.get("limit", 1000))
    if limit < 1 or len(found) > limit:
        raise ReadError("pulse catalogue exceeds limit; narrow the pattern")
    return {
        "pulses": [
            {
                "id": str(item.pulseID),
                "start_ns": int(item.timeFrom),
                "end_ns": int(item.timeTo),
            }
            for item in found
        ],
        "categories": [str(item) for item in client.getPulseCategories()],
    }


@dataclass
class Reply:
    time: np.ndarray
    values: np.ndarray
    unit: str = ""
    description: str = ""
    empty: bool = False
    error: str | None = None


def fetch(client, query: str) -> Reply:
    handle = client.fetchData(query)
    if handle < 0:
        error = str(client.getErrorMsg())
        empty = "requested interval is empty" in error.lower()
        return Reply(
            np.array([], dtype=np.int64),
            np.array([]),
            empty=empty,
            error=None if empty else error,
        )
    try:
        time_type = client.getTypeString(client.getFetchedTimeType(handle))
        times = np.array(
            client.getTimeStampsAsDouble(handle)
            if time_type.endswith(("FLOAT", "DOUBLE"))
            else client.getTimeStampsAsLong(handle),
            copy=True,
        )
        data_type = client.getTypeString(client.getFetchedType(handle))
        values = np.array(
            list(client.getDataAsStrings(handle))
            if data_type.endswith("STRING")
            else client.getDataNativeRank(handle),
            copy=True,
        )
        if times.ndim != 1 or values.ndim == 0 or len(times) != len(values):
            raise ReadError("timestamp and value dimensions disagree")
        return Reply(
            times,
            values,
            str(client.getUnitsY(handle)).strip(),
            str(client.getDescriptionY(handle)).strip(),
            empty=not len(times),
        )
    finally:
        client.releaseData(handle)


def check_spacing(times: np.ndarray, expected: float | None) -> None:
    if len(times) < 2:
        return
    if np.any(times[1:] <= times[:-1]):
        raise ReadError("raw timestamps must increase strictly")
    steps = np.diff(times)
    if not np.all(np.isfinite(steps)) or np.any(steps <= 0):
        raise ReadError("raw timestamps must increase strictly")
    if expected is not None and np.any(
        (steps < 0.75 * expected) | (steps > 1.25 * expected)
    ):
        raise ReadError("raw sample spacing differs from expected cadence")


def _limited(error: str | None) -> bool:
    message = (error or "").lower()
    return "samples in reply" in message and ("limit" in message or "exceed" in message)


def read(client, request: dict) -> dict:
    """Request raw samples explicitly, paging only after a reply-limit refusal.

    Absolute bounds are integer epoch nanoseconds; relative bounds are seconds.
    Paged raw reads require an independent cadence in the returned timestamp units.
    The server's isDecimated() flag is never used as sampling evidence.
    """
    variable = _token(request["name"])
    field = request.get("field")
    name = f"{variable}/{_token(field)}" if field else variable
    time_format = request.get("time_format", "absolute")
    mode = request.get("mode", "raw")
    if time_format not in {"absolute", "relative"} or mode not in {"raw", "envelope"}:
        raise ValueError("unsupported timestamp format or read mode")
    start, end = request.get("start"), request.get("end")
    pulse = request.get("pulse")
    if (start is None) != (end is None) or (start is None and not pulse):
        raise ValueError("supply a complete time window or a pulse")
    if time_format == "relative" and not pulse:
        raise ValueError("relative time requires a pulse")
    if start is not None:
        if isinstance(start, bool) or isinstance(end, bool):
            raise ValueError("time bounds must be numeric")
        if time_format == "absolute" and (
            type(start) is not int or type(end) is not int
        ):
            raise ValueError("absolute bounds must be integer epoch nanoseconds")
        if not np.isfinite(start) or not np.isfinite(end) or start >= end:
            raise ValueError("time bounds must be finite and increasing")
    maximum = int(request.get("max_samples", 100_000))
    page_size = int(request.get("page_samples", 10_000))
    max_pages = int(request.get("max_pages", 20))
    bins = int(request.get("bins", 1000))
    expected = request.get("expected_spacing")
    if min(maximum, page_size, max_pages, bins) < 1 or maximum > 100_000:
        raise ValueError(
            "sample and page bounds must be positive; at most 100000 samples"
        )
    if bins > maximum:
        raise ValueError("envelope bin count exceeds sample bound")
    if expected is not None and (not np.isfinite(expected) or expected <= 0):
        raise ValueError("expected_spacing must be finite and positive")
    parts = [f"variable={name}", f"tsFormat={time_format}"]
    if pulse:
        parts.append(f"pulse={_token(pulse)}")
    if start is not None:
        suffix = "S" if time_format == "relative" else ""
        parts.extend((f"startTime={start}{suffix}", f"endTime={end}{suffix}"))
    parts.append(f"decSamples={-1 if mode == 'raw' else bins}")
    if mode == "envelope":
        parts.append("decType=env")
    query = ",".join(parts)
    queries = [query]
    reply = fetch(client, query)
    if mode == "raw" and _limited(reply.error):
        if expected is None:
            raise ReadError("expected_spacing is required to verify raw pages")
        pages = []
        offset = 0
        for _ in range(max_pages):
            page_query = f"{query},ddContinueFrom={offset},ddSamples={page_size}"
            queries.append(page_query)
            page = fetch(client, page_query)
            if _limited(page.error) and page_size > 1:
                page_size = max(1, page_size // 2)
                continue
            if page.error:
                raise ReadError(page.error)
            if page.empty:
                break
            if len(page.time) > page_size:
                raise ReadError("raw page exceeds requested count")
            check_spacing(page.time, expected)
            if pages:
                check_spacing(np.array([pages[-1].time[-1], page.time[0]]), expected)
            pages.append(page)
            offset += len(page.time)
            if offset > maximum:
                raise ReadError("raw read exceeds sample bound")
            if len(page.time) < page_size:
                break
        else:
            raise ReadError("raw read exhausted page bound")
        reply = (
            Reply(
                np.concatenate([page.time for page in pages]),
                np.concatenate([page.values for page in pages]),
                pages[0].unit,
                pages[0].description,
            )
            if pages
            else Reply(np.array([], dtype=np.int64), np.array([]), empty=True)
        )
    if reply.error:
        raise ReadError(reply.error)
    if reply.empty:
        reply.time = np.array(
            [], dtype=np.int64 if time_format == "absolute" else np.float64
        )
    if len(reply.time) > maximum:
        raise ReadError("read exceeds sample bound")
    if mode == "raw":
        check_spacing(reply.time, expected)
    elif len(reply.time) and (reply.values.ndim != 2 or reply.values.shape[1] != 3):
        raise ReadError("envelope must contain mean, minimum and maximum columns")
    if reply.time.dtype.kind not in ("iu" if time_format == "absolute" else "f"):
        raise ReadError("returned timestamp dtype disagrees with requested time base")
    return {
        "name": name,
        "unit": reply.unit,
        "description": reply.description,
        "time_base": "epoch_ns" if time_format == "absolute" else "pulse_seconds",
        "decimated": True
        if mode == "envelope"
        else (False if expected is not None and len(reply.time) > 1 else None),
        "sampling_verification": "server_envelope"
        if mode == "envelope"
        else (
            "cadence_checked"
            if expected is not None and len(reply.time) > 1
            else "explicit_raw_request"
        ),
        "mode": mode,
        "query": query,
        "queries": queries,
        "empty": reply.empty,
        "time_dtype": str(reply.time.dtype),
        "value_dtype": str(reply.values.dtype),
        "time": reply.time.tolist(),
        "values": reply.values.tolist(),
    }


def dispatch(client, request: dict) -> dict:
    operation = request["operation"]
    if operation == "variables":
        return variables(client, request)
    if operation == "pulses":
        return pulses(client, request)
    if operation == "read":
        return read(client, request)
    raise ValueError("unknown operation")


def main() -> int:
    try:
        request = json.load(sys.stdin)
        # Native libraries can print diagnostics; reserve stdout for the receipt.
        with contextlib.redirect_stdout(sys.stderr):
            client = connect(
                os.environ.get("CODAC_UDA_HOST", ""),
                json.loads(os.environ.get("CODAC_UDA_OPERATOR_HOSTS", "[]")),
                int(os.environ.get("CODAC_UDA_PORT", "3090")),
            )
            result = dispatch(client, request)
        print(json.dumps({"ok": True, "result": result}))
        return 0
    except Exception as exc:
        print(
            json.dumps(
                {"ok": False, "error_type": type(exc).__name__, "error": str(exc)}
            )
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
