"""Focused fake-client checks; all archive identities here are invented."""

from __future__ import annotations

import json
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.codac_uda_prototype import loader, reader


class FakeClient:
    def __init__(self, *, paged=False, relative=False, strings=False, error=None):
        self.paged = paged
        self.relative = relative
        self.strings = strings
        self.error = error
        self.queries = []
        self.released = []
        self.times = np.arange(8, dtype=float if relative else np.int64)
        self.values = np.arange(8, dtype=np.float32)
        self.current_time = self.times
        self.current_values = self.values

    def getVariableList(self, pattern):
        assert pattern == "DEMO-*"
        return ["DEMO-PLANT:TEMP"]

    def getMetaTypeJSONCollapsed(self, name):
        assert name == "DEMO-PLANT:TEMP"
        return json.dumps(
            {
                "datatypes": [
                    {
                        "fields": [
                            {
                                "name": "val",
                                "unit": "K",
                                "description": "Demonstration temperature",
                            }
                        ]
                    }
                ]
            }
        )

    def getPulses2(self, pattern):
        assert pattern == "DEMO:*/*"
        return [SimpleNamespace(pulseID="DEMO:TRIAL/1", timeFrom=10, timeTo=20)]

    def getPulseCategories(self):
        return ["DEMO:TRIAL"]

    def fetchData(self, query):
        self.queries.append(query)
        parameters = dict(item.split("=", 1) for item in query.split(","))
        if self.error:
            return -1
        if self.paged and "ddContinueFrom" not in parameters:
            self.error = "samples in reply exceed limit"
            return -1
        self.current_time, self.current_values = self.times, self.values
        if "ddContinueFrom" in parameters:
            self.error = None
            offset, count = (
                int(parameters["ddContinueFrom"]),
                int(parameters["ddSamples"]),
            )
            stride = 1 if parameters.get("decSamples") == "-1" else 2
            self.current_time = self.times[::stride][offset : offset + count]
            self.current_values = self.values[::stride][offset : offset + count]
        if parameters.get("decType") == "env":
            self.current_values = np.column_stack(
                [self.values, self.values - 1, self.values + 1]
            )
        if self.strings:
            self.current_values = np.array(
                [f"record-{int(i)}" for i in self.current_time]
            )
        return 7

    def getErrorMsg(self):
        message = self.error
        # The initial refusal is transient: following requests may page it.
        if message == "samples in reply exceed limit":
            self.error = None
        return message

    def getFetchedTimeType(self, handle):
        return "DOUBLE" if self.relative else "LONG"

    def getFetchedType(self, handle):
        return "STRING" if self.strings else "FLOAT"

    def getTypeString(self, value):
        return value

    def getTimeStampsAsDouble(self, handle):
        assert self.relative
        return self.current_time

    def getTimeStampsAsLong(self, handle):
        assert not self.relative
        return self.current_time

    def getDataNativeRank(self, handle):
        return self.current_values

    def getDataAsStrings(self, handle):
        return list(self.current_values)

    def getUnitsY(self, handle):
        return "K"

    def getDescriptionY(self, handle):
        return "Demonstration temperature"

    def isDecimated(self, handle):
        return False

    def releaseData(self, handle):
        self.released.append(handle)


def window(**parameters):
    return {"name": "DEMO-PLANT:TEMP", "start": 0, "end": 8, **parameters}


def test_variables_metadata():
    result = reader.dispatch(
        FakeClient(), {"operation": "variables", "pattern": "DEMO-*"}
    )
    assert result["entries"] == [
        {
            "name": "DEMO-PLANT:TEMP",
            "unit": {"value": "K", "source": "uda"},
            "description": {"value": "Demonstration temperature", "source": "uda"},
        }
    ]
    assert result["warnings"] == []


def test_metadata_absence_and_external_endpoint_refusal(monkeypatch):
    with pytest.raises(ValueError, match="endpoints are not accepted"):
        reader.variables(
            FakeClient(),
            {"pattern": "DEMO-*", "metadata_url": "http://metadata.invalid/table"},
        )
    client = FakeClient()
    monkeypatch.setattr(client, "getMetaTypeJSONCollapsed", lambda name: "invalid JSON")
    result = reader.variables(client, {"pattern": "DEMO-*"})
    assert result["entries"][0]["unit"]["value"] is None
    assert result["warnings"][0]["error"]


def test_pulses_are_structured():
    result = reader.dispatch(
        FakeClient(), {"operation": "pulses", "pattern": "DEMO:*/*"}
    )
    assert result == {
        "pulses": [{"id": "DEMO:TRIAL/1", "start_ns": 10, "end_ns": 20}],
        "categories": ["DEMO:TRIAL"],
    }


def test_raw_integer_precision_and_release():
    client = FakeClient()
    client.times += 1_800_000_000_000_000_000
    result = reader.read(client, window(expected_spacing=1))
    assert result["time"][-1] == 1_800_000_000_000_000_007
    assert result["decimated"] is False
    assert result["sampling_verification"] == "cadence_checked"
    assert "decSamples=-1" in result["query"]
    assert client.released == [7]


def test_unverified_raw_does_not_claim_sampling_certainty():
    result = reader.read(FakeClient(), window())
    assert result["decimated"] is None
    assert result["sampling_verification"] == "explicit_raw_request"


def test_envelope_and_statistics():
    result = reader.read(
        FakeClient(relative=True),
        window(pulse="DEMO:TRIAL/1", time_format="relative", mode="envelope", bins=8),
    )
    assert result["decimated"] is True
    assert result["time_base"] == "pulse_seconds"
    assert "startTime=0S" in result["query"]
    assert "decType=env" in result["query"]
    signal = loader.Signal(
        np.asarray(result["time"]), np.asarray(result["values"]), result
    )
    assert loader.statistics(signal) == {
        "count": 8,
        "time_span_s": 7,
        "sample_interval_s": 1,
        "min": -1,
        "max": 8,
        "mean": 3.5,
        "statistics_kind": "unweighted_bin_means",
    }


def test_decimation_trap_is_observable():
    client = FakeClient()
    bad = reader.fetch(client, "variable=DEMO-PLANT:TEMP,ddContinueFrom=0,ddSamples=4")
    assert bad.time.tolist() == [0, 2, 4, 6]
    assert client.isDecimated(7) is False
    with pytest.raises(reader.ReadError, match="expected cadence"):
        reader.check_spacing(bad.time, 1)


def test_raw_pages_retain_explicit_no_decimation():
    client = FakeClient(paged=True)
    result = reader.read(client, window(expected_spacing=1, page_samples=3))
    assert result["time"] == list(range(8))
    pages = [query for query in client.queries if "ddContinueFrom" in query]
    assert len(pages) == 3
    assert all("decSamples=-1" in query for query in pages)
    assert len(client.released) == 3


def test_paged_raw_requires_independent_cadence():
    with pytest.raises(reader.ReadError, match="expected_spacing is required"):
        reader.read(FakeClient(paged=True), window())


def test_paging_bounds_and_boundary_gap():
    with pytest.raises(reader.ReadError, match="page bound"):
        reader.read(
            FakeClient(paged=True),
            window(expected_spacing=1, page_samples=3, max_pages=1),
        )
    client = FakeClient(paged=True)
    client.times[3:] += 1
    with pytest.raises(reader.ReadError, match="expected cadence"):
        reader.read(client, window(expected_spacing=1, page_samples=3))
    with pytest.raises(reader.ReadError, match="sample bound"):
        reader.read(FakeClient(), window(max_samples=4, bins=4))


def test_empty_interval_is_distinct_from_refusal():
    result = reader.read(FakeClient(error="requested interval is empty"), window())
    assert result["empty"] is True
    with pytest.raises(reader.ReadError, match="permission denied"):
        reader.read(FakeClient(error="permission denied"), window())


def test_handle_released_on_bad_dimensions():
    client = FakeClient()
    client.values = np.array([1])
    with pytest.raises(reader.ReadError, match="dimensions disagree"):
        reader.read(client, window())
    assert client.released == [7]


def test_structured_string_field():
    result = reader.read(FakeClient(strings=True), window(field="label"))
    assert "variable=DEMO-PLANT:TEMP/label" in result["query"]
    assert result["values"][0] == "record-0"


@pytest.mark.parametrize("host", ["operators.invalid", "OPERATORS.other.", "operators"])
def test_operator_refusal_precedes_client_construction(host):
    def forbidden(*args):
        pytest.fail("operator refusal must precede factory construction")

    with pytest.raises(ValueError, match="reserved for operators"):
        reader.connect(host, ["operators.invalid"], factory=forbidden)


def test_operator_policy_is_mandatory():
    with pytest.raises(ValueError, match="exclusions must be configured"):
        reader.connect("archive.invalid", [])
    assert reader.connect(
        "archive.invalid", ["operators.invalid"], factory=lambda *args: args
    ) == ("archive.invalid", 3090)


@pytest.mark.parametrize(
    "parameters",
    [
        {"start": None, "end": None},
        {"start": 2, "end": 1},
        {"start": 1.5},
        {"name": "DEMO,decSamples=2"},
        {"time_format": "relative"},
        {"mode": "unknown"},
        {"expected_spacing": 0},
    ],
)
def test_invalid_requests_never_fetch(parameters):
    client = FakeClient()
    with pytest.raises((ValueError, TypeError)):
        reader.read(client, window(**parameters))
    assert not client.queries


def test_loader_crosses_process_boundary_with_typed_receipt():
    result = reader.read(FakeClient(), window())
    source = (
        "import json,sys; request=json.load(sys.stdin); assert request['operation']=='read'; print("
        + repr(json.dumps({"ok": True, "result": result}))
        + ")"
    )
    signal = loader.Loader([sys.executable, "-c", source]).read(
        "DEMO-PLANT:TEMP", start=0, end=8
    )
    assert signal.time.dtype == np.int64
    assert signal.values.dtype == np.float32
    assert signal.metadata["unit"] == "K"
    assert loader.statistics(signal)["mean"] == 3.5


def test_loader_reads_failure_receipt_and_exit_status():
    command = [
        sys.executable,
        "-c",
        'print(\'{"ok":false,"error_type":"ReadError","error":"refused"}\'); raise SystemExit(1)',
    ]
    with pytest.raises(RuntimeError, match="ReadError: refused"):
        loader.Loader(command).variables("DEMO-*")


def test_loader_timeout():
    command = [sys.executable, "-c", "import time; time.sleep(1)"]
    with pytest.raises(subprocess.TimeoutExpired):
        loader.Loader(command, timeout=0.01).pulses("DEMO:*/*")


def test_unsigned_timestamps_cannot_wrap_backwards():
    client = FakeClient()
    client.times = np.array([9, 8, 7, 6, 5, 4, 3, 2], dtype=np.uint64)
    with pytest.raises(reader.ReadError, match="increase strictly"):
        reader.read(client, window())
    assert client.released == [7]


def test_empty_relative_interval_has_relative_dtype():
    result = reader.read(
        FakeClient(relative=True, error="requested interval is empty"),
        window(pulse="DEMO:TRIAL/1", time_format="relative"),
    )
    assert result["empty"] is True
    assert result["time_dtype"] == "float64"


def test_plot_saves_internal_svg_and_prints_statistics(capsys):
    from pathlib import Path

    signal = loader.Signal(
        np.arange(8),
        np.arange(8, dtype=float),
        {"time_base": "pulse_seconds", "unit": "K", "mode": "raw"},
    )
    path = Path.home() / ".cache/iter-codex-codac-uda/fake-signal.svg"
    result = loader.plot(signal, path)
    assert result["count"] == 8
    assert json.loads(capsys.readouterr().out)["mean"] == 3.5
    assert 'width="31.1cm" height="16cm"' in path.read_text()
    with pytest.raises(ValueError, match="internal cache"):
        loader.plot(signal, Path("public.svg"))
