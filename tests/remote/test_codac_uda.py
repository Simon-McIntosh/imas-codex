"""Packaged archive reader controls using invented endpoints and selectors."""

import json
import shlex
import subprocess
import sys
from unittest.mock import Mock

import pytest

from imas_codex.remote.scripts import codac_uda as reader


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
    assert result["entries"][0]["unit"] == {"value": None, "source": None}
    assert len(result["warnings"]) == 1


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
    result = reader.variables(client, {"pattern": "DEMO:*", "metadata_batch_size": 2})
    assert [len(call.args[1]) for call in metadata.call_args_list] == [2, 2, 1]
    assert len(result["entries"]) == 5
    assert result["entries"][0]["description"]["value"] == "Archive wording"
    client.getMetaTypeJSONCollapsed.assert_not_called()


def test_listing_error_is_not_a_completed_empty_cohort():
    client = Mock()
    client.getVariableList.return_value = []
    client.getErrorCode.return_value = 9
    with pytest.raises(reader.ReadError, match="listing failed"):
        reader.variables(client, {"pattern": "DEMO:*"})


def test_batch_uses_corrected_names_and_releases_handles(monkeypatch):
    import ctypes

    library = Mock()
    buffer = ctypes.create_string_buffer(b'{"datatypes": []}')

    def batch(signals, sources, count, handles, host, port):
        assert count == 2 and host == b"archive.invalid" and port == 3090
        assert b"DEMO/CHANNEL/meta" in signals[0]
        handles[0], handles[1] = 10, 11
        return 0

    library.idamGetBatchAPIWithHost.side_effect = batch
    library.getIdamErrorCode.return_value = 0
    library.getIdamDataType.return_value = 17
    library.getIdamDataNum.return_value = len(buffer)
    library.getIdamData.return_value = ctypes.addressof(buffer)
    monkeypatch.setattr(reader.ctypes, "CDLL", lambda path: library)
    client = Mock()
    client.correctVariableName.return_value = "DEMO/CHANNEL"
    client.getHost.return_value = "archive.invalid"
    client.getPort.return_value = 3090
    assert (
        reader.metadata_batch(client, ["DEMO:CHANNEL", "DEMO:OTHER"])
        == [{"datatypes": []}] * 2
    )
    assert [call.args[0] for call in library.udaFree.call_args_list] == [10, 11]


def test_failed_batch_releases_every_allocated_handle(monkeypatch):
    library = Mock()

    def batch(signals, sources, count, handles, host, port):
        handles[0] = 10
        return -1

    library.idamGetBatchAPIWithHost.side_effect = batch
    monkeypatch.setattr(reader.ctypes, "CDLL", lambda path: library)
    client = Mock()
    client.correctVariableName.return_value = "DEMO/CHANNEL"
    client.getHost.return_value = "archive.invalid"
    client.getPort.return_value = 3090
    with pytest.raises(reader.ReadError, match="batch failed"):
        reader.metadata_batch(client, ["DEMO:CHANNEL"])
    library.udaFree.assert_called_once_with(10)
