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
                    "unit": {"value": "K", "source": "uda"},
                    "description": {
                        "value": "Demonstration temperature",
                        "source": "uda",
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


def test_variables_bound_and_metadata_absence():
    client = Mock()
    client.getVariableList.return_value = ["DEMO-PLANT:TEMP", "DEMO-PLANT:PRESSURE"]
    with pytest.raises(reader.ReadError, match="catalogue exceeds limit"):
        reader.variables(client, {"pattern": "DEMO-*", "limit": 1})
    client.getMetaTypeJSONCollapsed.assert_not_called()
    client.getVariableList.return_value = ["DEMO-PLANT:TEMP"]
    client.getMetaTypeJSONCollapsed.return_value = "invalid JSON"
    result = reader.variables(client, {"pattern": "DEMO-*", "limit": 1})
    assert result["entries"][0]["unit"] == {"value": None, "source": None}
    assert len(result["warnings"]) == 1
