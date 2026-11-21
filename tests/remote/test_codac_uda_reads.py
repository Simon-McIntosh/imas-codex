"""Sampling and handle-lifetime controls with invented archive identities."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from iter_codex.remote.scripts import codac_uda as reader


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

    def getErrorCode(self):
        return 1 if self.error else 0

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
        return [
            SimpleNamespace(
                pulseID="DEMO:TRIAL/1",
                timeFrom=10,
                timeTo=20,
                description="Demonstration pulse",
                status="finished",
            )
        ]

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


def test_raw_pages_keep_explicit_no_decimation():
    client = FakeClient(paged=True)
    result = reader.read(client, window(expected_spacing=1, page_samples=3))
    assert result["time"] == list(range(8))
    assert result["values"] == list(range(8))
    assert result["decimated"] is False
    assert result["sampling_verification"] == "cadence_checked"
    assert len(client.queries) == 4
    assert all("decSamples=-1" in query for query in client.queries)
    assert client.released == [7, 7, 7]


def test_decimated_pages_fail_cadence_control():
    client = FakeClient(paged=True)
    actual_fetch = client.fetchData
    client.fetchData = lambda query: actual_fetch(query.replace(",decSamples=-1", ""))
    with pytest.raises(
        reader.ReadError, match="sample spacing differs from expected cadence"
    ):
        reader.read(client, window(expected_spacing=1, page_samples=3))
    assert "ddContinueFrom" in client.queries[-1]
    assert client.released == [7]


def test_raw_paging_requires_independent_cadence():
    with pytest.raises(reader.ReadError, match="expected_spacing is required"):
        reader.read(FakeClient(paged=True), window())


def test_server_envelope_shape_and_evidence():
    result = reader.read(FakeClient(), window(mode="envelope", bins=8))
    assert np.asarray(result["values"]).shape == (8, 3)
    assert result["decimated"] is True
    assert result["sampling_verification"] == "server_envelope"
    assert "decType=env" in result["query"]


def test_unknown_raw_sampling_without_independent_cadence():
    result = reader.read(FakeClient(), window())
    assert result["decimated"] is None
    assert result["sampling_verification"] == "explicit_raw_request"


def test_relative_string_field_and_handle_release():
    client = FakeClient(relative=True, strings=True)
    result = reader.read(
        client, window(field="label", time_format="relative", pulse="DEMO:TRIAL/1")
    )
    assert result["time_base"] == "pulse_seconds"
    assert result["value_dtype"].startswith("<U")
    assert result["values"][0] == "record-0"
    assert "variable=DEMO-PLANT:TEMP/label" in result["query"]
    assert client.released == [7]


def test_handle_released_when_dimensions_disagree():
    client = FakeClient()
    client.values = np.array([1])
    with pytest.raises(reader.ReadError, match="dimensions disagree"):
        reader.fetch(client, "variable=DEMO-PLANT:TEMP")
    assert client.released == [7]


def test_empty_and_permission_refusal_are_distinct():
    assert (
        reader.read(FakeClient(error="requested interval is empty"), window())["empty"]
        is True
    )
    with pytest.raises(reader.ReadError, match="permission denied"):
        reader.read(FakeClient(error="permission denied"), window())


def test_pulse_listing_adds_explicit_pulse_segment():
    result = reader.pulses(FakeClient(), {"pattern": "DEMO:*"})
    assert result == {
        "pulses": [
            {
                "id": "DEMO:TRIAL/1",
                "start_ns": 10,
                "end_ns": 20,
                "description": "Demonstration pulse",
                "status": "finished",
            }
        ],
        "categories": ["DEMO:TRIAL"],
    }
    with pytest.raises(ValueError, match="pulse limit must be positive"):
        reader.pulses(FakeClient(), {"pattern": "DEMO:*/*", "limit": 0})


@pytest.mark.parametrize("times", [[1, 1], [2, 1], [0.0, float("nan")], [0, 2]])
def test_spacing_refusal(times):
    with pytest.raises(reader.ReadError):
        reader.check_spacing(np.array(times), 1)


@pytest.mark.parametrize(
    "parameters",
    [
        {"max_samples": 100001},
        {"max_bytes": 8388609},
        {"start": 0, "end": 60000000001},
        {"max_samples": 4, "bins": 4, "expected_spacing": 1},
    ],
)
def test_limit_refused_before_fetch(parameters):
    client = FakeClient()
    with pytest.raises((ValueError, reader.ReadError)):
        reader.read(client, window(**parameters))
    assert client.queries == []


def test_byte_limit_applies_before_serializing_transport():
    with pytest.raises(reader.ReadError, match="byte bound"):
        reader.read(FakeClient(), window(max_bytes=1))


def event_client():
    client = FakeClient()
    client.times = np.array([-10, 4, 7, 12], dtype=np.int64)
    client.values = np.array([2, 6, 10, 20], dtype=np.float32)
    return client


def test_held_raw_has_boundary_values_and_provenance():
    result = reader.read(
        event_client(), window(start=0, end=10, semantics="held", lookback_ns=20)
    )
    assert result["time"] == [0, 4, 7, 10]
    assert result["values"] == [2, 6, 10, 10]
    assert result["held_since_ns"] == -10
    assert result["semantics"] == "zero_order_hold"
    assert result["decimated"] is None
    assert "startTime=-20" in result["query"]
    assert "extSamples=true" in result["query"]


def test_held_envelope_uses_duration_weights():
    result = reader.read(
        event_client(),
        window(start=0, end=10, semantics="held", mode="envelope", bins=1),
    )
    assert result["time"] == [5]
    assert result["values"] == [[5.6, 2.0, 10.0]]
    assert result["sampling_verification"] == "held_value_bins"
    assert "decSamples=-1" in result["query"]
    assert "decType=env" not in result["query"]


def test_held_window_without_predecessor_reports_coverage_gap():
    client = event_client()
    client.times += 20
    result = reader.read(client, window(start=0, end=10, semantics="held"))
    assert result["held_start_covered"] is False
    assert result["held_since_ns"] is None


def test_sample_binning_retains_sampling_evidence():
    result = reader.read(FakeClient(), window(mode="binned", bins=2))
    assert result["time"] == [2, 6]
    assert result["values"] == [1.5, 5.5]
    assert result["sampling_verification"] == "local_sample_bins"
    assert result["decimated"] is True


def test_relative_bins_keep_fractional_seconds():
    client = FakeClient(relative=True)
    client.times = np.arange(8, dtype=float) / 10
    result = reader.read(
        client,
        window(
            start=0.0,
            end=0.8,
            time_format="relative",
            pulse="DEMO:TRIAL/1",
            mode="binned",
            bins=2,
        ),
    )
    np.testing.assert_allclose(result["time"], [0.2, 0.6])
    assert result["values"] == [1.5, 5.5]


def test_held_unsigned_epoch_timestamps_preserve_nanosecond_precision():
    client = event_client()
    origin = 1700000000000000000
    client.times = np.array(
        [origin - 10, origin + 4, origin + 7, origin + 12], dtype=np.uint64
    )
    result = reader.read(
        client, window(start=origin, end=origin + 10, semantics="held")
    )
    assert result["time_dtype"] == "uint64"
    assert result["time"] == [origin, origin + 4, origin + 7, origin + 10]
