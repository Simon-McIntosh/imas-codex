"""Packaged archive reader controls using invented endpoints and selectors."""

import json
import shlex
import subprocess
import sys
from unittest.mock import Mock

import pytest

from iter_codex.remote.scripts import codac_uda as reader


@pytest.fixture
def archive_environment(tmp_path, monkeypatch):
    """Make the native-client stand-in available only after the shell setup."""
    package = tmp_path / "uda_client_reader"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "uda_client_reader_python.py").write_text(
        "import json\n"
        "class UdaClientReaderPython:\n"
        "    def __init__(self, host, port):\n"
        "        assert host == 'archive.invalid' and port == 3090\n"
        "        print('client diagnostic')\n"
        "    def getVariableList(self, pattern):\n"
        "        assert pattern == 'DEMO-PLANT:TEMP'\n"
        "        return [pattern]\n"
        "    def getErrorCode(self): return 0\n"
        "    def getMetaTypeJSONCollapsed(self, name):\n"
        "        return json.dumps({'datatypes': [{'fields': [{'name': 'val', "
        "'unit': 'K', 'description': 'Demonstration temperature'}]}]})\n"
    )
    monkeypatch.delenv("PYTHONPATH", raising=False)
    monkeypatch.setenv("CODAC_UDA_HOST", "archive.invalid")
    monkeypatch.setenv("CODAC_UDA_PORT", "3090")
    monkeypatch.setenv("CODAC_UDA_OPERATOR_HOSTS", '["operators.invalid"]')
    return [f"export PYTHONPATH={shlex.quote(str(tmp_path))}"]


def test_packaged_reader_lists_variable(run_local, archive_environment):
    receipt = json.loads(
        run_local(
            "codac_uda.py",
            input_data={
                "operation": "variables",
                "pattern": "DEMO-PLANT:TEMP",
                "limit": 1,
                "metadata": False,
            },
            python_command=sys.executable,
            setup_commands=archive_environment,
        )
    )
    assert receipt == {
        "ok": True,
        "result": {
            "entries": [
                {
                    "name": "DEMO-PLANT:TEMP",
                    "unit": {"value": None, "source": None},
                    "description": {
                        "value": None,
                        "source": None,
                    },
                }
            ],
            "warnings": [],
        },
    }


def test_packaged_reader_without_setup_cannot_import_client(
    run_local, archive_environment
):
    with pytest.raises(subprocess.CalledProcessError) as error:
        run_local(
            "codac_uda.py",
            input_data={
                "operation": "variables",
                "pattern": "DEMO-PLANT:TEMP",
                "limit": 1,
            },
            python_command=sys.executable,
        )
    receipt = json.loads(error.value.stdout)
    assert receipt["ok"] is False
    assert receipt["error_type"] == "ModuleNotFoundError"
    assert "uda_client_reader" in receipt["error"]


@pytest.mark.parametrize("host", ["operators.invalid", "OPERATORS.other.", "operators"])
def test_operator_endpoint_refused_before_construction(host):
    factory = Mock()
    with pytest.raises(ValueError, match="reserved for operators"):
        reader.connect(host, ["operators.invalid"], factory=factory)
    factory.assert_not_called()


def test_operator_exclusions_required():
    factory = Mock()
    with pytest.raises(ValueError, match="exclusions must be configured"):
        reader.connect("archive.invalid", [], factory=factory)
    factory.assert_not_called()


def test_variables_bound_and_metadata_absence(monkeypatch):
    client = Mock()
    client.getErrorCode.return_value = 0
    client.getVariableList.return_value = ["DEMO-PLANT:TEMP", "DEMO-PLANT:PRESSURE"]
    with pytest.raises(reader.ReadError, match="catalogue exceeds limit"):
        reader.variables(client, {"pattern": "DEMO-*", "limit": 1})
    client.getMetaTypeJSONCollapsed.assert_not_called()
    client.getVariableList.return_value = ["DEMO-PLANT:TEMP"]
    monkeypatch.setattr(
        reader, "metadata_batch", lambda client, names: [{"error": "invalid JSON"}]
    )
    result = reader.variables(client, {"pattern": "DEMO-*", "limit": 1})
    assert result["entries"][0]["metadata_state"] == "unavailable"
    assert result["failed"] == [
        {"name": "DEMO-PLANT:TEMP", "error": "invalid JSON", "error_code": 0}
    ]


@pytest.mark.parametrize(
    "host,operator", [("192.0.2.12", "192.0.2.25"), ("2001:db8::1", "2001:db8::2")]
)
def test_distinct_ip_endpoints_are_not_operator_aliases(host, operator):
    factory = Mock()
    reader.connect(host, [operator], factory=factory)
    factory.assert_called_once_with(host, 3090)


def test_variable_metadata_is_batched_and_keeps_archive_wording(monkeypatch):
    client = Mock()
    client.getErrorCode.return_value = 0
    client.getVariableList.return_value = [f"DEMO:TEMP-{index}" for index in range(5)]
    metadata = Mock(
        side_effect=lambda client, names: [
            {
                "datatypes": [
                    {
                        "fields": [
                            {
                                "name": "val",
                                "unit": "K",
                                "description": "Archive wording",
                            }
                        ]
                    }
                ]
            }
            for _ in names
        ]
    )
    monkeypatch.setattr(reader, "metadata_batch", metadata)
    results = [
        reader.dispatch(
            client,
            {
                "operation": "metadata",
                "names": client.getVariableList.return_value[offset : offset + 2],
                "metadata_batch_size": 2,
            },
        )
        for offset in range(0, 5, 2)
    ]
    assert [len(call.args[1]) for call in metadata.call_args_list] == [2, 2, 1]
    assert sum(len(result["entries"]) for result in results) == 5
    assert results[0]["entries"][0]["description"]["value"] == "Archive wording"
    client.getMetaTypeJSONCollapsed.assert_not_called()


def test_variable_listing_never_loops_over_multiple_metadata_batches(monkeypatch):
    client = Mock()
    client.getVariableList.return_value = [f"DEMO:TEMP-{index}" for index in range(5)]
    client.getErrorCode.return_value = 0
    batch = Mock()
    monkeypatch.setattr(reader, "metadata_batch", batch)
    with pytest.raises(reader.ReadError, match="one bounded batch"):
        reader.variables(client, {"pattern": "DEMO:*", "metadata_batch_size": 2})
    batch.assert_not_called()
    listing = reader.variables(client, {"pattern": "DEMO:*", "metadata": False})
    assert len(listing["entries"]) == 5
    batch.assert_not_called()


@pytest.mark.parametrize("names", [[], ["DEMO:TEMP"] * 3])
def test_metadata_request_bound_is_enforced_before_native_call(monkeypatch, names):
    batch = Mock()
    monkeypatch.setattr(reader, "metadata_batch", batch)
    with pytest.raises(reader.ReadError, match="one bounded batch"):
        reader.metadata(Mock(), {"names": names, "metadata_batch_size": 2})
    batch.assert_not_called()


def test_listing_error_is_not_a_completed_empty_cohort():
    client = Mock()
    client.getVariableList.return_value = []
    client.getErrorCode.return_value = 9
    with pytest.raises(reader.ReadError, match="listing failed"):
        reader.variables(client, {"pattern": "DEMO:*"})


def test_metadata_requests_each_variable_and_isolates_refusal():
    client = Mock()
    client.getMetaTypeJSONCollapsed.side_effect = [
        '{"datatypes": []}',
        ValueError("fixture refusal"),
        '{"datatypes": []}',
    ]
    client.getMetaJSON.return_value = "[]"
    client.getErrorCode.return_value = 0
    documents = reader.metadata_batch(client, ["DEMO:ONE", "DEMO:TWO", "DEMO:THREE"])
    assert documents[0] == documents[2] == {"type": {"datatypes": []}, "full": []}
    assert (
        documents[1]["error"] == "fixture refusal" and documents[1]["error_code"] == 0
    )
    assert client.getMetaJSON.call_count == 3
    assert [
        call.args[0] for call in client.getMetaTypeJSONCollapsed.call_args_list
    ] == ["DEMO:ONE", "DEMO:TWO", "DEMO:THREE"]


def test_failed_metadata_is_not_an_empty_success(monkeypatch):
    monkeypatch.setattr(
        reader, "metadata_batch", lambda client, names: [{"error": "fixture refusal"}]
    )
    result = reader.metadata(Mock(), {"names": ["DEMO:FAILED"]})
    assert result["entries"][0]["metadata_state"] == "unavailable"
    assert result["entries"][0]["metadata_error_code"] == 0
    assert "archive_metadata" not in result["entries"][0]
    assert result["failed"] == [
        {"name": "DEMO:FAILED", "error": "fixture refusal", "error_code": 0}
    ]
