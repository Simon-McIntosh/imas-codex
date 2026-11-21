"""Graph identity and bounded archive service controls with invented selectors."""

import json
import subprocess
from unittest.mock import Mock

import numpy as np
import pytest

from iter_codex.remote import codac_uda as service


@pytest.fixture
def facility_config():
    return {
        "data_systems": {
            "codac_uda": {
                "archives": [
                    {"role": "fast", "host": "fast.invalid"},
                    {"role": "slow", "host": "archive.invalid"},
                    {"role": "operator", "host": "operators.invalid", "excluded": True},
                ],
                "readable_roles": ["slow"],
                "max_samples": 100,
                "max_bytes": 10000,
                "max_window_ns": 60000000000,
                "setup_commands": ["export DEMONSTRATION_CLIENT=enabled"],
            }
        }
    }


@pytest.fixture
def result():
    return {
        "name": "DEMO-PLANT:TEMP",
        "time": [0, 1, 2],
        "values": [2, 4, 8],
        "time_dtype": "int64",
        "value_dtype": "float32",
        "time_base": "epoch_ns",
        "mode": "raw",
        "decimated": False,
        "sampling_verification": "cadence_checked",
        "unit": "K",
        "description": "Demonstration temperature",
        "query": "variable=DEMO-PLANT:TEMP,decSamples=-1",
    }


@pytest.fixture
def reader(facility_config, tmp_path):
    return service.ArchiveReader(
        "demo", "slow", config=facility_config, cache_dir=tmp_path / "signals"
    )


def test_reader_reuses_facility_script_and_cache(reader, monkeypatch, result):
    runner = Mock(return_value=json.dumps({"ok": True, "result": result}))
    monkeypatch.setattr(service, "run_python_script", runner)
    first = reader.read("DEMO-PLANT:TEMP", start=0, end=3, bins=3)
    second = reader.read("DEMO-PLANT:TEMP", start=0, end=3, bins=3)
    assert first.metadata["cache_hit"] is False
    assert second.metadata["cache_hit"] is True
    np.testing.assert_array_equal(first.time, second.time)
    np.testing.assert_array_equal(first.values, second.values)
    assert first.time.dtype == np.int64
    assert first.values.dtype == np.float32
    runner.assert_called_once()
    args = runner.call_args
    assert args.args == ("codac_uda.py",)
    assert args.kwargs["facility"] == "demo"
    assert args.kwargs["data_system"] == "codac_uda"
    assert args.kwargs["setup_commands"][0] == "export DEMONSTRATION_CLIENT=enabled"
    assert "CODAC_UDA_OPERATOR_HOSTS" in " ".join(args.kwargs["setup_commands"])


@pytest.mark.parametrize(
    "parameters",
    [
        {"max_samples": 101},
        {"max_bytes": 10001},
        {"end": 60000000001},
        {"end": 60000000000, "semantics": "held", "lookback_ns": 1},
        {"max_samples": True},
    ],
)
def test_out_of_budget_never_starts_reader(reader, parameters):
    reader._request = Mock(side_effect=AssertionError("fetch must not run"))
    with pytest.raises(service.ArchiveReadError) as error:
        reader.read("DEMO-PLANT:TEMP", **{"start": 0, "end": 3, **parameters})
    assert error.value.category == "limit"
    reader._request.assert_not_called()


def test_scanner_and_reader_share_samples_and_error_classification(reader, result):
    reader._request = Mock(return_value=result)
    signal = reader.read("DEMO-PLANT:TEMP", start=0, end=3)
    check = reader.check("DEMO-PLANT:TEMP", start=0, end=3)
    assert check["success"] is True
    np.testing.assert_array_equal(signal.time, check["signal"].time)
    np.testing.assert_array_equal(signal.values, check["signal"].values)
    with pytest.raises(service.ArchiveReadError) as failure:
        reader.read("DEMO-PLANT:TEMP", start=0, end=3, max_samples=101)
    check = reader.check("DEMO-PLANT:TEMP", start=0, end=3, max_samples=101)
    assert check["error_classification"] == failure.value.category == "limit"
    reader._request.assert_called_once()


def test_native_failure_receipt_classified_identically(reader, monkeypatch):
    runner = Mock(
        side_effect=subprocess.CalledProcessError(
            1,
            "reader",
            output=json.dumps(
                {
                    "ok": False,
                    "error_type": "ReadError",
                    "error": "raw sample spacing differs from expected cadence",
                }
            ),
        )
    )
    monkeypatch.setattr(service, "run_python_script", runner)
    with pytest.raises(service.ArchiveReadError) as failure:
        reader.read("DEMO-PLANT:TEMP", start=0, end=3)
    check = reader.check("DEMO-PLANT:TEMP", start=0, end=3)
    assert failure.value.category == check["error_classification"] == "sampling"


def test_corrupt_cache_is_refetched_and_sampling_settings_split_cache(reader, result):
    reader._request = Mock(return_value=result)
    signal = reader.read("DEMO-PLANT:TEMP", start=0, end=3)
    path = reader.cache_dir / (signal.metadata["cache_key"] + ".json")
    path.write_text('{"time": [1]}')
    assert reader.read("DEMO-PLANT:TEMP", start=0, end=3).metadata["cache_hit"] is False
    reader.read("DEMO-PLANT:TEMP", start=0, end=3, field="val")
    assert reader._request.call_count == 3


def test_excluded_host_alias_cannot_construct_service(facility_config, tmp_path):
    facility_config["data_systems"]["codac_uda"]["archives"][0]["host"] = (
        "OPERATORS.other."
    )
    with pytest.raises(ValueError, match="excluded"):
        service.ArchiveReader(
            "demo", "slow", config=facility_config, cache_dir=tmp_path
        )


def test_pulses_construct_category_qualified_graph_identity(reader):
    reader._request = Mock(
        return_value={
            "pulses": [{"id": "DEMO:TRIAL/nested/1", "start_ns": 0, "end_ns": 3}],
            "categories": ["DEMO:TRIAL"],
        }
    )
    (pulse,) = reader.pulses("DEMO:*")
    assert pulse.id == "demo:codac_uda:slow:DEMO:TRIAL/nested/1"
    assert pulse.name == "DEMO:TRIAL/nested/1"
    assert pulse.categories == ["demo:codac_uda:slow:DEMO:TRIAL"]


def binding():
    return {
        "signal_id": "demo:plant/temperature",
        "facility_id": "demo",
        "selector": "DEMO-PLANT:TEMP",
        "unit": "K",
        "description": "Demonstration temperature",
        "archive_id": "demo:codac_uda:slow",
        "archive_role": "slow",
        "method_type": "codac_uda",
    }


def test_loader_resolves_graph_identity_and_uses_same_validity_path(reader, result):
    reader._request = Mock(return_value=result)
    graph = Mock()
    graph.query.return_value = [binding()]
    factory = Mock(return_value=reader)
    loader = service.ArchiveSignalLoader(graph, reader_factory=factory)
    loaded = loader.load("demo:plant/temperature", start=0, end=3)
    check = reader.check("DEMO-PLANT:TEMP", start=0, end=3)
    np.testing.assert_array_equal(loaded.values, check["signal"].values)
    np.testing.assert_array_equal(loaded.time, check["signal"].time)
    assert loaded.metadata["signal_id"] == "demo:plant/temperature"
    factory.assert_called_once_with("demo", "slow")
    graph.query.assert_called_once_with(
        service.SIGNAL_QUERY, signal_id="demo:plant/temperature"
    )
    with pytest.raises(service.ArchiveReadError) as failure:
        loader.load("demo:plant/temperature", start=0, end=3, max_samples=101)
    assert (
        failure.value.category
        == reader.check("DEMO-PLANT:TEMP", start=0, end=3, max_samples=101)[
            "error_classification"
        ]
    )


def test_loader_resolves_pulse_bounds_and_rejects_another_archive(reader, result):
    reader._request = Mock(return_value=result)
    graph = Mock()
    pulse = {
        "archive": reader.archive,
        "start_ns": 0,
        "end_ns": 3,
        "name": "DEMO:TRIAL/1",
    }
    graph.query.side_effect = [[binding()], [pulse]]
    loader = service.ArchiveSignalLoader(graph, reader_factory=lambda *args: reader)
    loader.load("demo:plant/temperature", pulse_id="demo:codac_uda:slow:DEMO:TRIAL/1")
    assert reader._request.call_args.kwargs["start"] == 0
    pulse["archive"] = "demo:codac_uda:fast"
    graph.query.side_effect = [[binding()], [pulse]]
    with pytest.raises(service.ArchiveReadError, match="pulse must belong"):
        loader.load(
            "demo:plant/temperature", pulse_id="demo:codac_uda:fast:DEMO:TRIAL/1"
        )


@pytest.mark.parametrize("rows", [[], [binding(), binding()]])
def test_unresolved_or_ambiguous_identity_refused_before_fetch(rows):
    graph = Mock()
    graph.query.return_value = rows
    factory = Mock()
    with pytest.raises(service.ArchiveReadError, match="exactly one"):
        service.ArchiveSignalLoader(graph, reader_factory=factory).load(
            "demo:plant/temperature", start=0, end=3
        )
    factory.assert_not_called()


def test_default_reader_loads_explicit_private_infrastructure(
    monkeypatch, facility_config, tmp_path
):
    infrastructure = Mock(return_value=facility_config)
    monkeypatch.setattr(service, "get_facility_with_infrastructure", infrastructure)
    service.ArchiveReader("demo", "slow", cache_dir=tmp_path)
    infrastructure.assert_called_once_with("demo")


def test_relative_pulse_defaults_use_seconds(reader, result):
    reader._request = Mock(return_value=result)
    graph = Mock()
    graph.query.side_effect = [
        [binding()],
        [
            {
                "archive": reader.archive,
                "start_ns": 10000000000,
                "end_ns": 12000000000,
                "name": "DEMO:TRIAL/1",
            }
        ],
    ]
    loader = service.ArchiveSignalLoader(graph, reader_factory=lambda *args: reader)
    loader.load(
        "demo:plant/temperature",
        pulse_id="demo:codac_uda:slow:DEMO:TRIAL/1",
        time_format="relative",
    )
    assert reader._request.call_args.kwargs["start"] == 0.0
    assert reader._request.call_args.kwargs["end"] == 2.0
    assert reader._request.call_args.kwargs["pulse"] == "DEMO:TRIAL/1"


@pytest.mark.parametrize(
    ("bounds", "category", "reason"),
    [
        ({}, "invalid_request", "bounds are required"),
        ({"start": 0}, "invalid_request", "bounds are required"),
        (
            {"start": 0, "end": 60000000001},
            "limit",
            "exceed configured bound",
        ),
    ],
)
def test_shell_load_refuses_missing_or_excessive_window(
    reader, bounds, category, reason
):
    graph = Mock()
    graph.query.return_value = [binding()]
    reader._request = Mock(side_effect=AssertionError("fetch must not run"))
    loader = service.ArchiveSignalLoader(graph, reader_factory=lambda *args: reader)
    with pytest.raises(service.ArchiveReadError, match=reason) as error:
        loader.load("demo:plant/temperature", **bounds)
    assert error.value.category == category
    reader._request.assert_not_called()


def test_shell_load_returns_arrays_unit_and_time_base_over_three_pulses(reader, result):
    graph = Mock()
    graph.query.return_value = [binding()]
    reader._request = Mock(
        side_effect=lambda operation, **request: {
            **result,
            "time": [request["start"] + index for index in range(3)],
        }
    )
    loader = service.ArchiveSignalLoader(graph, reader_factory=lambda *args: reader)
    signals = [
        loader.load(
            "demo:plant/temperature",
            start=start,
            end=start + 3,
            mode="raw",
            max_samples=100,
            max_bytes=10000,
        )
        for start in [0, 10, 20]
    ]
    assert reader._request.call_count == 3
    assert [call.kwargs["start"] for call in reader._request.call_args_list] == [
        0,
        10,
        20,
    ]
    for start, signal in zip([0, 10, 20], signals, strict=True):
        np.testing.assert_array_equal(
            signal.time, np.array([start, start + 1, start + 2])
        )
        np.testing.assert_array_equal(signal.values, np.array([2, 4, 8]))
        assert signal.metadata["unit"] == "K"
        assert signal.metadata["time_base"] == "epoch_ns"
        assert signal.metadata["sampling_verification"] == "cadence_checked"
        assert signal.metadata["decimated"] is False


def test_open_pulse_requires_explicit_window_and_keeps_its_missing_end(reader, result):
    reader._request = Mock(return_value=result)
    pulse = {
        "archive": reader.archive,
        "start_ns": 0,
        "end_ns": None,
        "name": "DEMO:TRIAL/open",
    }
    graph = Mock()
    graph.query.side_effect = [[binding()], [pulse], [binding()], [pulse]]
    loader = service.ArchiveSignalLoader(graph, reader_factory=lambda *args: reader)
    with pytest.raises(service.ArchiveReadError, match="explicit bounded window"):
        loader.load(
            "demo:plant/temperature", pulse_id="demo:codac_uda:slow:DEMO:TRIAL/open"
        )
    reader._request.assert_not_called()
    signal = loader.load(
        "demo:plant/temperature",
        pulse_id="demo:codac_uda:slow:DEMO:TRIAL/open",
        start=0,
        end=3,
    )
    assert len(signal.time) == 3
    assert pulse["end_ns"] is None


def test_exact_native_name_can_load_without_an_imported_graph_identity(reader, result):
    native = {**result, "name": "DEMO-NATIVE:CURRENT", "unit": "A"}
    reader._request = Mock(return_value=native)
    signal = reader.read(
        "DEMO-NATIVE:CURRENT",
        start=0,
        end=3,
        mode="raw",
        max_samples=100,
        max_bytes=10000,
    )
    assert reader._request.call_args.kwargs["name"] == "DEMO-NATIVE:CURRENT"
    np.testing.assert_array_equal(signal.values, np.array([2, 4, 8]))
    assert signal.metadata["unit"] == "A"
    assert signal.metadata["time_base"] == "epoch_ns"
