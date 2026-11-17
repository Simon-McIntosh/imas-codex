"""Metadata-only pulse collection retains empty categories and missing ends."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from imas_codex.remote.codac_uda import (
    ArchiveReader,
    ArchiveReadError,
    ArchiveSignalLoader,
)
from imas_codex.remote.pulse_catalogue import collect_pulses, snapshot_models
from imas_codex.remote.scripts import codac_uda as native


def fake_reader():
    reader = object.__new__(ArchiveReader)
    reader.facility = "demo"
    reader.archive = "demo:codac_uda:slow"
    reader._request = Mock(
        side_effect=lambda operation, **kw: (
            {"categories": ["DEMO:EMPTY", "DEMO:ACTIVE"]}
            if operation == "pulse_categories"
            else {
                "categories": ["DEMO:EMPTY", "DEMO:ACTIVE"],
                "pulses": []
                if kw["pattern"] == "DEMO:EMPTY"
                else [
                    {
                        "id": "DEMO:ACTIVE/open",
                        "start_ns": 1731600000000000001,
                        "end_ns": None,
                        "description": "Demonstration pulse",
                    }
                ],
            }
        )
    )
    return reader


def test_collection_preserves_empty_categories_and_open_ends(tmp_path):
    reader = fake_reader()
    state = collect_pulses(reader, tmp_path / "pulses.json")
    categories, pulses = snapshot_models(state)
    assert [c.name for c in categories] == ["DEMO:EMPTY", "DEMO:ACTIVE"]
    assert state["listings"]["DEMO:EMPTY"] == []
    assert pulses[0].end_ns is None
    assert pulses[0].description == "Demonstration pulse"
    assert pulses[0].start_ns == 1731600000000000001
    assert [call.args[0] for call in reader._request.call_args_list] == [
        "pulse_categories",
        "pulses",
        "pulses",
    ]
    reader._request.reset_mock()
    assert collect_pulses(reader, tmp_path / "pulses.json") == state
    reader._request.assert_not_called()


def test_interrupted_listing_is_incomplete_and_resumes(tmp_path):
    reader = fake_reader()
    request = reader._request.side_effect
    reader._request.side_effect = lambda operation, **kw: (
        (_ for _ in ()).throw(RuntimeError("interrupted"))
        if kw.get("pattern") == "DEMO:ACTIVE"
        else request(operation, **kw)
    )
    path = tmp_path / "pulses.json"
    with pytest.raises(RuntimeError, match="interrupted"):
        collect_pulses(reader, path)
    import json

    state = json.loads(path.read_text())
    assert state["complete"] is False
    with pytest.raises(ValueError, match="incomplete"):
        snapshot_models(state)
    reader._request.reset_mock()
    reader._request.side_effect = request
    state = collect_pulses(reader, path)
    assert state["complete"] is True
    reader._request.assert_called_once_with(
        "pulses", pattern="DEMO:ACTIVE", limit=100000
    )


@pytest.mark.parametrize("end", [None, "", 1731600000000000010])
def test_native_pulse_listing_keeps_missing_end_and_description(end):
    client = Mock()
    client.getErrorCode.return_value = 0
    client.getPulseCategories.return_value = ["DEMO:ACTIVE", "DEMO:EMPTY"]
    client.getPulses2.return_value = [
        SimpleNamespace(
            pulseID="DEMO:ACTIVE/open",
            timeFrom=1731600000000000001,
            timeTo=end,
            description="Demonstration pulse",
        )
    ]
    result = native.dispatch(
        client, {"operation": "pulses", "pattern": "DEMO:ACTIVE", "limit": 1}
    )
    assert result["pulses"][0]["end_ns"] == (None if end in (None, "") else end)
    assert result["pulses"][0]["description"] == "Demonstration pulse"
    client.fetchData.assert_not_called()


@pytest.mark.parametrize("operation", ["pulses", "pulse_categories"])
def test_native_listing_error_is_not_reported_as_empty(operation):
    client = Mock()
    client.getErrorCode.return_value = 17
    client.getErrorMsg.return_value = "Demonstration refusal"
    client.getPulses2.return_value = []
    client.getPulseCategories.return_value = []
    with pytest.raises(native.ReadError, match="listing failed"):
        native.dispatch(client, {"operation": operation, "pattern": "DEMO:ACTIVE"})


def test_native_pulse_catalogue_overflow_is_refused():
    client = Mock()
    client.getErrorCode.return_value = 0
    client.getPulses2.return_value = [object(), object()]
    with pytest.raises(native.ReadError, match="pulse catalogue exceeds limit"):
        native.dispatch(
            client, {"operation": "pulses", "pattern": "DEMO:ACTIVE", "limit": 1}
        )
    client.getPulseCategories.assert_not_called()
    client.fetchData.assert_not_called()


@pytest.mark.parametrize("categories", [[""], ["DEMO:ACTIVE", "DEMO:ACTIVE"]])
def test_native_category_identity_refusal(categories):
    client = Mock()
    client.getErrorCode.return_value = 0
    client.getPulseCategories.return_value = categories
    with pytest.raises(native.ReadError, match="empty or duplicate identities"):
        native.dispatch(client, {"operation": "pulse_categories"})
    client.getPulses2.assert_not_called()
    client.fetchData.assert_not_called()


def test_open_pulse_requires_explicit_window_before_data_read():
    reader = Mock(archive="demo:codac_uda:slow")
    graph = Mock()
    graph.query.side_effect = [
        [
            {
                "method_type": "codac_uda",
                "facility_id": "demo",
                "archive_role": "slow",
                "archive_id": reader.archive,
                "selector": "DEMO:TEMP",
            }
        ],
        [
            {
                "archive": reader.archive,
                "start_ns": 10,
                "end_ns": None,
                "name": "DEMO:ACTIVE/open",
            }
        ],
    ]
    with pytest.raises(ArchiveReadError, match="explicit bounded window"):
        ArchiveSignalLoader(graph, reader_factory=lambda *args: reader).load(
            "demo:signal", pulse_id="demo:pulse"
        )
    reader.read.assert_not_called()
