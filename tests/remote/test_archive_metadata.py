"""Both archive metadata interfaces and explicit unavailability semantics."""

import json
from io import StringIO

from iter_codex.remote.scripts import codac_uda as reader


class Client:
    def __init__(self, refused=False):
        self.refused = refused
        self.code = 0
        self.calls = []

    def getMetaTypeJSONCollapsed(self, name):
        self.calls.append(("type", name))
        self.code = 9105 if self.refused else 0
        return (
            ""
            if self.refused
            else json.dumps(
                {
                    "datatypes": [
                        {
                            "name": "main",
                            "fields": [
                                {
                                    "name": "val",
                                    "type": "DOUBLE",
                                    "unit": "invented type unit",
                                    "description": "Signal value",
                                }
                            ],
                        }
                    ]
                }
            )
        )

    def getMetaJSON(self, name, when):
        self.calls.append(("full", name))
        assert when == "-1"
        self.code = 9105 if self.refused else 0
        return (
            ""
            if self.refused
            else json.dumps(
                [
                    {"name": "label", "value": "Enabled"},
                    {"name": "label", "value": "Disabled"},
                    {"name": "Units", "value": "K"},
                    {"name": "Format", "value": "%.2f"},
                    {"name": "LowerDisplayLimit", "value": "0"},
                    {"name": "UpperDisplayLimit", "value": "1000"},
                    {"name": "LowerAlarmLimit", "value": "10"},
                    {"name": "UpperAlarmLimit", "value": "990"},
                    {"name": "LowerWarningLimit", "value": "20"},
                    {"name": "UpperWarningLimit", "value": "980"},
                ]
            )
        )

    def getErrorCode(self):
        return self.code

    def getErrorMsg(self):
        return "fixture metadata unavailable" if self.code else ""


def test_both_metadata_interfaces_supply_attributes_without_data_reads():
    client = Client()
    result = reader.metadata(client, {"names": ["DEMO:TEMP"]})
    entry = result["entries"][0]
    assert client.calls == [("type", "DEMO:TEMP"), ("full", "DEMO:TEMP")]
    assert entry["metadata_state"] == "available"
    assert entry["unit"]["value"] == "K"
    assert entry["archive_labels"] == ["Enabled", "Disabled"]
    assert entry["archive_format"] == "%.2f"
    assert entry["archive_lower_display_limit"] == 0
    assert entry["archive_upper_display_limit"] == 1000
    assert entry["archive_lower_alarm_limit"] == 10
    assert entry["archive_upper_alarm_limit"] == 990
    assert entry["archive_lower_warning_limit"] == 20
    assert entry["archive_upper_warning_limit"] == 980
    assert set(entry["archive_metadata"]) == {"type", "full"}


def test_metadata_refusal_is_unavailable_with_code_and_no_metadata_fields():
    client = Client(refused=True)
    result = reader.metadata(client, {"names": ["DEMO:REFUSED"]})
    entry = result["entries"][0]
    assert client.calls == [("type", "DEMO:REFUSED"), ("full", "DEMO:REFUSED")]
    assert entry["metadata_state"] == "unavailable"
    assert entry["metadata_error_code"] == 9105
    assert "fixture metadata unavailable" in entry["metadata_error"]
    assert not (
        {
            "archive_metadata",
            "unit",
            "description",
            "data_type",
            "structure_fields",
            "archive_labels",
        }
        & entry.keys()
    )


def test_missing_full_metadata_unit_stays_absent_and_cached_type_is_reused():
    client = Client()
    document = json.loads(client.getMetaTypeJSONCollapsed("DEMO:TEMP"))
    client.calls.clear()
    client.getMetaJSON = lambda name, when: "[]"
    entry = reader.metadata(
        client, {"names": ["DEMO:TEMP"], "type_documents": {"DEMO:TEMP": document}}
    )["entries"][0]
    assert client.calls == []
    assert entry["unit"]["value"] is None
    assert entry["metadata_state"] == "available"


def test_invalid_structure_is_unavailable_without_aborting_batch():
    client = Client()
    client.getMetaTypeJSONCollapsed = lambda name: '{"datatypes": null}'
    entry = reader.metadata(client, {"names": ["DEMO:MALFORMED"]})["entries"][0]
    assert (
        entry["metadata_state"] == "unavailable" and entry["metadata_error_code"] == 0
    )
    assert "archive_metadata" not in entry


def test_metadata_stream_refuses_data_operation_and_keeps_serving_metadata():
    client = Client()
    source = StringIO(
        json.dumps({"operation": "read"})
        + "\n"
        + json.dumps({"operation": "metadata", "names": ["DEMO:TEMP"]})
        + "\n"
    )
    output = StringIO()
    assert reader.metadata_stream(client, source, output) == 0
    receipts = [
        json.loads(line.removeprefix("CODAC_METADATA "))
        for line in output.getvalue().splitlines()
    ]
    assert receipts[0] == {"ready": True}
    assert receipts[1]["ok"] is False and "refuses non-metadata" in receipts[1]["error"]
    assert receipts[2]["ok"] is True
    assert client.calls == [("type", "DEMO:TEMP"), ("full", "DEMO:TEMP")]
