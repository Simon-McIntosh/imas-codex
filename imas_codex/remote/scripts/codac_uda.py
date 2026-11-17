"""CODAC UDA reader: one JSON request on stdin, one JSON receipt on stdout.

Python 3.12+; requires numpy and the module-provided UDA client.

Run with the module-provided interpreter. No project imports are required.
Archive addresses and operator exclusions come from deployment configuration.
"""

from __future__ import annotations

import contextlib
import json
import os
import sys
from dataclasses import dataclass
from typing import Any

import numpy as np

from imas_codex.config.codac_uda import archive_host_key


class ReadError(RuntimeError):
    """A refused read, invalid sampling cadence, or exhausted read bound."""


def connect(host: str, operator_hosts: list[str], port: int = 3090, factory=None):
    """Refuse operator endpoints before importing or constructing the client.

    A deployment must supply the exclusion list, even when injecting a client.
    Canonical keys preserve full IP addresses and compare DNS aliases consistently.
    """
    if not operator_hosts or not all(archive_host_key(item) for item in operator_hosts):
        raise ValueError("operator host exclusions must be configured")
    if archive_host_key(host) in {archive_host_key(item) for item in operator_hosts}:
        raise ValueError("archive endpoint is reserved for operators")
    if factory is None:
        from uda_client_reader.uda_client_reader_python import UdaClientReaderPython

        factory = UdaClientReaderPython
    return factory(host, port)


def _token(value: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or any(char in value for char in ",=\n\r")
    ):
        raise ValueError("query tokens must be nonempty and contain no separators")
    return value


def _field(value: Any, source: str) -> dict:
    text = str(value).strip() if value is not None else ""
    return {"value": text or None, "source": source if text else None}


def metadata_batch(
    client, names: list[str], type_documents: dict | None = None
) -> list[dict]:
    """Query type and full attributes, retaining each interface's archive error.

    Previously checkpointed type documents may be reused during attribute
    backfill. Both interfaces are attempted even when one refuses a variable.
    """
    results = []
    for name in names:
        documents = {}
        errors = []
        for interface, method in (
            ("type", client.getMetaTypeJSONCollapsed),
            ("full", client.getMetaJSON),
        ):
            if interface == "type" and name in (type_documents or {}):
                documents[interface] = type_documents[name]
                continue
            code = 0
            try:
                raw = method(name) if interface == "type" else method(name, "-1")
                try:
                    code = int(client.getErrorCode())
                except (ValueError, TypeError):
                    code = 0
                if code:
                    raise ReadError(str(client.getErrorMsg()))
                document = json.loads(raw)
                if not isinstance(document, dict if interface == "type" else list):
                    raise ValueError("archive metadata has an unexpected JSON shape")
                documents[interface] = document
            except Exception as exc:
                errors.append(
                    {"interface": interface, "error_code": code, "error": str(exc)}
                )
        if errors:
            results.append(
                {
                    "error": errors[0]["error"],
                    "error_code": errors[0]["error_code"],
                    "errors": errors,
                    "documents": documents,
                }
            )
        else:
            results.append(documents)
    return results


def variables(client, request: dict) -> dict:
    """List selectors independently of bounded metadata retrieval."""
    if "metadata_url" in request:
        raise ValueError("metadata endpoints are not accepted in reader requests")
    pattern = _token(request["pattern"])
    names = list(dict.fromkeys(str(item) for item in client.getVariableList(pattern)))
    if client.getErrorCode() != 0:
        raise ReadError("archive variable listing failed")
    limit = int(request.get("limit", 1000))
    if limit < 1 or len(names) > limit:
        raise ReadError("variable catalogue exceeds limit; narrow the pattern")
    if request.get("metadata") is not False and names:
        return metadata(client, {**request, "names": names})
    return {
        "entries": [
            {
                "name": name,
                "unit": _field(None, "uda"),
                "description": _field(None, "uda"),
            }
            for name in names
        ],
        "warnings": [],
    }


def metadata(client, request: dict) -> dict:
    """Return one bounded metadata batch; the caller checkpoints between calls."""
    names = request.get("names")
    maximum = request.get("metadata_batch_size", 100)
    if type(maximum) is not int or not 1 <= maximum <= 1000:
        raise ValueError("metadata batch size must be between 1 and 1000")
    if not isinstance(names, list) or not 0 < len(names) <= maximum:
        raise ReadError("metadata request must contain one bounded batch of names")
    if not all(isinstance(name, str) for name in names):
        raise ValueError("metadata names must be strings")
    names = [_token(name) for name in names]
    if len(set(names)) != len(names):
        raise ValueError("metadata names must be unique")
    warnings = []
    entries = []
    failed = []
    cached = request.get("type_documents")
    documents = (
        metadata_batch(client, names, cached)
        if cached
        else metadata_batch(client, names)
    )
    for name, document in zip(names, documents, strict=True):
        if "error" in document:
            entry = {
                "name": name,
                "metadata_state": "unavailable",
                "metadata_error_code": document.get("error_code", 0),
                "metadata_error": str(document["error"]),
                "metadata_errors": document.get("errors", []),
                **(
                    {"metadata_interfaces": document["documents"]}
                    if document.get("documents")
                    else {}
                ),
            }
            entries.append(entry)
            failed.append(
                {
                    "name": name,
                    "error": entry["metadata_error"],
                    "error_code": entry["metadata_error_code"],
                }
            )
            continue
        type_document = document.get("type", document)
        attributes = document.get("full", [])
        datatypes = type_document.get("datatypes", [])
        valid_structure = isinstance(datatypes, list) and all(
            isinstance(item, dict)
            and isinstance(item.get("fields", []), list)
            and all(isinstance(field, dict) for field in item.get("fields", []))
            for item in datatypes
        )
        if not valid_structure or not all(
            isinstance(item, dict) for item in attributes
        ):
            error = "archive metadata document has invalid fields"
            entries.append(
                {
                    "name": name,
                    "metadata_state": "unavailable",
                    "metadata_error_code": 0,
                    "metadata_error": error,
                    "metadata_interfaces": {"type": type_document, "full": attributes},
                }
            )
            failed.append({"name": name, "error": error, "error_code": 0})
            continue
        root = next(
            (item for item in datatypes if item.get("name") == "main"),
            datatypes[-1] if datatypes else {},
        )
        fields = root.get("fields", [])
        signal = next((field for field in fields if field.get("name") == "val"), None)
        if signal is None and len(fields) == 1:
            signal = fields[0]
        values = {}
        for attribute in attributes:
            value = attribute.get("value")
            if value is not None and str(value).strip():
                values.setdefault(attribute.get("name"), []).append(value)
        entry = {
            "name": name,
            "metadata_state": "available",
            "unit": _field(next(iter(values.get("Units", [])), None), "uda_full"),
            "description": _field(
                signal.get("description") if signal else None, "uda_type"
            ),
            "archive_metadata": {"type": type_document, "full": attributes},
            "data_type": signal.get("type") if signal else None,
            "structure_fields": datatypes,
            **{
                key: type_document[key]
                for key in (
                    "first_sample_ns",
                    "last_sample_ns",
                    "sampling_mode",
                    "sampling_evidence",
                )
                if type_document.get(key) is not None
            },
        }
        if values.get("label"):
            entry["archive_labels"] = [str(value) for value in values["label"]]
        if values.get("Format"):
            entry["archive_format"] = str(values["Format"][0])
        for source, target in (
            ("LowerDisplayLimit", "archive_lower_display_limit"),
            ("UpperDisplayLimit", "archive_upper_display_limit"),
            ("LowerAlarmLimit", "archive_lower_alarm_limit"),
            ("UpperAlarmLimit", "archive_upper_alarm_limit"),
            ("LowerWarningLimit", "archive_lower_warning_limit"),
            ("UpperWarningLimit", "archive_upper_warning_limit"),
        ):
            if values.get(source):
                try:
                    value = float(values[source][0])
                    if np.isfinite(value):
                        entry[target] = value
                except (ValueError, TypeError):
                    pass
        entries.append(entry)
    return {"entries": entries, "warnings": warnings, "failed": failed}


def pulse_categories(client, request: dict) -> dict:
    """List category identities without requiring any category to have pulses."""
    limit = int(request.get("limit", 10000))
    if limit < 1:
        raise ValueError("category limit must be positive")
    found = [str(item) for item in client.getPulseCategories()]
    if client.getErrorCode() != 0:
        raise ReadError("archive category listing failed: " + str(client.getErrorMsg()))
    if len(found) > limit:
        raise ReadError("category catalogue exceeds limit")
    if any(not name for name in found) or len(set(found)) != len(found):
        raise ReadError("category listing contains empty or duplicate identities")
    return {"categories": found}


def pulses(client, request: dict) -> dict:
    """Serialize SWIG pulse objects explicitly rather than relying on repr()."""
    pattern = _token(request["pattern"])
    if not pattern.endswith("/*"):
        pattern = pattern.rstrip("/") + "/*"
    limit = int(request.get("limit", 1000))
    if limit < 1:
        raise ValueError("pulse limit must be positive")
    found = list(client.getPulses2(pattern))
    if client.getErrorCode() != 0:
        raise ReadError("archive pulse listing failed: " + str(client.getErrorMsg()))
    if len(found) > limit:
        raise ReadError("pulse catalogue exceeds limit; narrow the pattern")
    return {
        "pulses": [
            {
                "id": str(item.pulseID),
                "start_ns": int(item.timeFrom),
                "end_ns": None if item.timeTo in (None, "") else int(item.timeTo),
                "description": str(item.description)
                if item.description is not None
                else None,
            }
            for item in found
        ],
        "categories": pulse_categories(client, request)["categories"],
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


def _held_window(reply: Reply, start: int, end: int) -> tuple[Reply, int | None]:
    """Clip event samples and expose the held boundary values explicitly."""
    check_spacing(reply.time, None)
    before = np.flatnonzero(reply.time <= start)
    since = int(reply.time[before[-1]]) if len(before) else None
    inside = (reply.time >= start) & (reply.time <= end)
    times, values = reply.time[inside], reply.values[inside]
    if len(before) and (not len(times) or times[0] != start):
        times = np.concatenate((np.array([start], dtype=reply.time.dtype), times))
        values = np.concatenate((reply.values[before[-1] : before[-1] + 1], values))
    if len(times) and times[-1] < end:
        times = np.concatenate((times, np.array([end], dtype=reply.time.dtype)))
        values = np.concatenate((values, values[-1:]))
    return Reply(
        times, values, reply.unit, reply.description, empty=not len(times)
    ), since


def _aggregate(
    reply: Reply, start, end, bins: int, *, held: bool, envelope: bool
) -> Reply:
    """Aggregate numeric samples; event values are weighted by their hold time."""
    if reply.values.dtype.kind not in "biuf" or reply.values.ndim != 1:
        raise ReadError("binning requires a scalar numeric field")
    centers, values = [], []
    for index in range(bins):
        integer_time = reply.time.dtype.kind in "iu"
        left_offset = (end - start) * index
        right_offset = (end - start) * (index + 1)
        left = start + (left_offset // bins if integer_time else left_offset / bins)
        right = start + (right_offset // bins if integer_time else right_offset / bins)
        if held:
            if not len(reply.time) or reply.time[0] > left:
                continue
            selected = (reply.time[:-1] < right) & (reply.time[1:] > left)
            samples = reply.values[:-1][selected]
            durations = np.minimum(reply.time[1:][selected], right) - np.maximum(
                reply.time[:-1][selected], left
            )
            if not len(samples):
                continue
            mean = float(np.average(samples, weights=durations))
        else:
            selected = (reply.time >= left) & (
                (reply.time <= right) if index == bins - 1 else (reply.time < right)
            )
            samples = reply.values[selected]
            if not len(samples):
                continue
            mean = float(np.mean(samples))
        centers.append(
            left + ((right - left) // 2 if integer_time else (right - left) / 2)
        )
        values.append(
            [mean, float(np.min(samples)), float(np.max(samples))] if envelope else mean
        )
    return Reply(
        np.asarray(centers, dtype=reply.time.dtype),
        np.asarray(values, dtype=float),
        reply.unit,
        reply.description,
        empty=not len(centers),
    )


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
    if time_format not in {"absolute", "relative"} or mode not in {
        "raw",
        "envelope",
        "binned",
    }:
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
    semantics = request.get("semantics", "samples")
    lookback = request.get("lookback_ns", 0)
    byte_limit = request.get("max_bytes", 8_388_608)
    window_limit = request.get("max_window_ns", 60_000_000_000)
    if semantics not in {"samples", "held"}:
        raise ValueError("unsupported signal semantics")
    held = semantics == "held"
    if type(lookback) is not int or lookback < 0 or (lookback and not held):
        raise ValueError("lookback requires held-value semantics")
    if held and (time_format != "absolute" or start is None or expected is not None):
        raise ValueError("held values require absolute bounds and no regular cadence")
    if type(byte_limit) is not int or not 1 <= byte_limit <= 8_388_608:
        raise ValueError("byte bound must be positive and at most 8388608")
    if type(window_limit) is not int or not 1 <= window_limit <= 60_000_000_000:
        raise ValueError("window bound must be positive and at most 60000000000 ns")
    if (
        start is not None
        and (end - start) * (1 if time_format == "absolute" else 1e9) + lookback
        > window_limit
    ):
        raise ReadError("requested window exceeds bound")
    if min(maximum, page_size, max_pages, bins) < 1 or maximum > 100_000:
        raise ValueError(
            "sample and page bounds must be positive; at most 100000 samples"
        )
    if bins > maximum:
        raise ValueError("envelope bin count exceeds sample bound")
    if expected is not None and (not np.isfinite(expected) or expected <= 0):
        raise ValueError("expected_spacing must be finite and positive")
    if (
        mode != "envelope"
        and expected is not None
        and start is not None
        and (end - start) / expected > maximum
    ):
        raise ReadError("expected sample count exceeds sample bound")
    parts = [f"variable={name}", f"tsFormat={time_format}"]
    if pulse:
        parts.append(f"pulse={_token(pulse)}")
    if start is not None:
        suffix = "S" if time_format == "relative" else ""
        parts.extend(
            (f"startTime={start - lookback}{suffix}", f"endTime={end}{suffix}")
        )
    raw_fetch = mode != "envelope" or held
    parts.append(f"decSamples={-1 if raw_fetch else bins}")
    if held:
        parts.append("extSamples=true")
    if not raw_fetch:
        parts.append("decType=env")
    query = ",".join(parts)
    queries = [query]
    reply = fetch(client, query)
    if raw_fetch and _limited(reply.error):
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
    if raw_fetch:
        check_spacing(reply.time, expected)
    elif len(reply.time) and (reply.values.ndim != 2 or reply.values.shape[1] != 3):
        raise ReadError("envelope must contain mean, minimum and maximum columns")
    if reply.time.dtype.kind not in ("iu" if time_format == "absolute" else "f"):
        raise ReadError("returned timestamp dtype disagrees with requested time base")
    held_since = None
    if held:
        reply, held_since = _held_window(reply, start, end)
    if mode == "binned" or (held and mode == "envelope"):
        reply = _aggregate(
            reply, start, end, bins, held=held, envelope=mode == "envelope"
        )
    if len(reply.time) > maximum:
        raise ReadError("held boundaries exceed sample bound")
    result = {
        "name": name,
        "unit": reply.unit,
        "description": reply.description,
        "time_base": "epoch_ns" if time_format == "absolute" else "pulse_seconds",
        "decimated": True
        if mode != "raw"
        else (False if expected is not None and len(reply.time) > 1 else None),
        "sampling_verification": "held_value_bins"
        if held and mode != "raw"
        else "local_sample_bins"
        if mode == "binned"
        else "server_envelope"
        if mode == "envelope"
        else (
            "cadence_checked"
            if expected is not None and len(reply.time) > 1
            else "explicit_raw_request"
        ),
        "mode": mode,
        "semantics": "zero_order_hold" if held else "archived_samples",
        "held_since_ns": held_since,
        "held_start_covered": held_since is not None if held else None,
        "boundary_values_synthesized": held,
        "query": query,
        "queries": queries,
        "empty": reply.empty,
        "time_dtype": str(reply.time.dtype),
        "value_dtype": str(reply.values.dtype),
        "time": reply.time.tolist(),
        "values": reply.values.tolist(),
    }
    try:
        size = len(json.dumps(result, allow_nan=False).encode())
    except ValueError as exc:
        raise ReadError(
            "nonfinite values cannot be represented in a JSON reply"
        ) from exc
    if size > byte_limit:
        raise ReadError("serialized reply exceeds byte bound")
    return result


def dispatch(client, request: dict) -> dict:
    operation = request["operation"]
    if operation == "variables":
        return variables(client, request)
    if operation == "metadata":
        return metadata(client, request)
    if operation == "pulses":
        return pulses(client, request)
    if operation == "pulse_categories":
        return pulse_categories(client, request)
    if operation == "read":
        return read(client, request)
    raise ValueError("unknown operation")


def metadata_stream(client, source, output) -> int:
    """Serve metadata-only batches on one persistent native client.

    The framed channel excludes data operations and survives native diagnostic
    lines on stdout. Each process handles one archive and one request at a time.
    """
    prefix = "CODAC_METADATA "
    print(prefix + json.dumps({"ready": True}), file=output, flush=True)
    for line in source:
        try:
            request = json.loads(line)
            if request.get("operation") != "metadata":
                raise ValueError("metadata stream refuses non-metadata operations")
            with contextlib.redirect_stdout(sys.stderr):
                result = metadata(client, request)
            receipt = {"ok": True, "result": result}
        except Exception as exc:
            receipt = {"ok": False, "error_type": type(exc).__name__, "error": str(exc)}
        print(prefix + json.dumps(receipt), file=output, flush=True)
    return 0


def main() -> int:
    try:
        streaming = os.environ.get("CODAC_UDA_METADATA_STREAM") == "1"
        request = None if streaming else json.load(sys.stdin)
        # Native libraries can print diagnostics; reserve stdout for the receipt.
        with contextlib.redirect_stdout(sys.stderr):
            client = connect(
                os.environ.get("CODAC_UDA_HOST", ""),
                json.loads(os.environ.get("CODAC_UDA_OPERATOR_HOSTS", "[]")),
                int(os.environ.get("CODAC_UDA_PORT", "3090")),
            )
            if not streaming:
                result = dispatch(client, request)
        if streaming:
            return metadata_stream(client, sys.stdin, sys.stdout)
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
