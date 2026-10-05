"""Remote environment probe: interpreter floor and module importability.

Covers the extension of the facility probe: ``get_python_status``, handed the
already-resolved ``remote_environment`` probe, judges the block's interpreter
against the block's floor through ``PythonVersion.meets_minimum``, and
``check_all_tools`` reports per-module importability. The probe script itself is
exercised end to end so a module that fails to import is named rather than
raised.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from imas_codex.remote import python as remote_python, tools as remote_tools

SCRIPT = Path(remote_tools.__file__).parent / "scripts" / "probe_environment.py"

SETUP = ["module unload python/3.5.6", "module load python/3.12"]


def _declared(
    python_version: str,
    floor: str,
    modules: list[dict],
    fix: list[str],
    error: str | None = None,
) -> dict:
    return {
        "status": "declared",
        "declared": True,
        "floor": floor,
        "python_version": python_version,
        "meets_floor": None,
        "modules": modules,
        "fix": list(fix),
        "error": error,
        "python_command": "python",
        "setup_commands": list(fix),
    }


@pytest.fixture
def quiet_python(monkeypatch: pytest.MonkeyPatch) -> None:
    """Silence the SSH-backed checks around get_python_status."""
    monkeypatch.setattr(
        remote_python,
        "check_tool",
        lambda key, facility=None: {"available": False, "version": None},
    )
    monkeypatch.setattr(remote_python, "run", lambda *a, **k: "")


class TestProbeScript:
    def test_reports_version_and_names_unimportable_module(self) -> None:
        payload = json.dumps(
            {
                "modules": [
                    {"name": "sys", "path": None},
                    {"name": "imas_codex_missing_module_xyz", "path": None},
                ]
            }
        )
        proc = subprocess.run(
            [sys.executable, str(SCRIPT)],
            input=payload,
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert proc.returncode == 0, proc.stderr
        data = json.loads(proc.stdout)
        assert data["python_version"].split(".")[0] == str(sys.version_info[0])
        by_name = {m["name"]: m for m in data["modules"]}
        assert by_name["sys"]["importable"] is True
        assert by_name["imas_codex_missing_module_xyz"]["importable"] is False
        assert (
            "imas_codex_missing_module_xyz"
            in by_name["imas_codex_missing_module_xyz"]["error"]
        )

    def test_adds_module_path_before_import(self, tmp_path: Path) -> None:
        sidecar = tmp_path / "probe_sidecar_module.py"
        sidecar.write_text("MARKER = 'ok'\n")
        payload = json.dumps(
            {"modules": [{"name": "probe_sidecar_module", "path": str(tmp_path)}]}
        )
        proc = subprocess.run(
            [sys.executable, str(SCRIPT)],
            input=payload,
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert proc.returncode == 0, proc.stderr
        data = json.loads(proc.stdout)
        assert data["modules"][0]["importable"] is True


class TestProbeRemoteEnvironment:
    def test_uses_block_commands_and_computes_floor(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config = {
            "remote_environment": {
                "python_command": "python",
                "setup_commands": SETUP,
                "min_python_version": "3.12",
                "required_modules": ["numpy"],
            }
        }
        captured: dict = {}

        def fake_run(script_name, input_data=None, **kwargs):
            captured["script"] = script_name
            captured.update(kwargs)
            return json.dumps(
                {
                    "python_version": "3.12.9",
                    "modules": [{"name": "numpy", "importable": True, "error": None}],
                }
            )

        monkeypatch.setattr(remote_tools, "run_python_script", fake_run)
        env = remote_tools.probe_remote_environment("jt-60sa", facility_config=config)

        assert captured["script"] == "probe_environment.py"
        assert env["status"] == "declared"
        assert env["python_version"] == "3.12.9"
        assert env["meets_floor"] is True
        assert env["fix"] == SETUP

    def test_old_interpreter_below_floor(self, monkeypatch: pytest.MonkeyPatch) -> None:
        config = {
            "remote_environment": {
                "python_command": "python",
                "setup_commands": SETUP,
                "min_python_version": "3.12",
                "required_modules": [],
            }
        }
        monkeypatch.setattr(
            remote_tools,
            "run_python_script",
            lambda *a, **k: json.dumps({"python_version": "3.5.6", "modules": []}),
        )
        env = remote_tools.probe_remote_environment("jt-60sa", facility_config=config)
        assert env["meets_floor"] is False

    def test_no_block_probes_nothing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        called: list[int] = []
        monkeypatch.setattr(
            remote_tools,
            "run_python_script",
            lambda *a, **k: called.append(1) or "{}",
        )
        env = remote_tools.probe_remote_environment("tcv", facility_config={})
        assert env["status"] == "not_declared"
        assert called == []


class TestCheckAllToolsEnvironment:
    def test_declared_block_carries_module_verdicts(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            remote_tools,
            "check_tool",
            lambda key, facility=None: {
                "available": True,
                "version": "1.0",
                "required": False,
                "meets_min_version": True,
            },
        )
        monkeypatch.setattr(
            remote_tools,
            "probe_remote_environment",
            lambda *a, **k: _declared(
                "3.12.9",
                "3.12",
                [
                    {
                        "name": "eddb_pwrapper",
                        "importable": False,
                        "error": "ModuleNotFoundError: No module named 'eddb_pwrapper'",
                    }
                ],
                SETUP,
            ),
        )
        results = remote_tools.check_all_tools(facility="jt-60sa")
        assert results["environment"]["status"] == "declared"
        assert results["environment"]["modules"][0]["name"] == "eddb_pwrapper"
        assert results["environment"]["fix"] == SETUP

    def test_no_block_reports_not_declared(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            remote_tools,
            "check_tool",
            lambda key, facility=None: {
                "available": True,
                "version": "1.0",
                "required": False,
                "meets_min_version": True,
            },
        )
        monkeypatch.setattr(
            remote_tools,
            "probe_remote_environment",
            lambda *a, **k: {
                "status": "not_declared",
                "declared": False,
                "modules": [],
                "fix": [],
            },
        )
        results = remote_tools.check_all_tools(facility="tcv")
        assert results["environment"]["status"] == "not_declared"


class TestGetPythonStatusFloor:
    def test_block_interpreter_used_and_floor_named(self, quiet_python: None) -> None:
        status = remote_python.get_python_status(
            "jt-60sa", environment=_declared("3.5.6", "3.12", [], SETUP)
        )
        assert status.active_python is not None
        assert status.active_python.version_string == "3.5.6"
        assert status.active_python.source == "remote_environment"
        assert status.min_python_version == "3.12"
        assert status.meets_floor is False
        assert status.environment["fix"] == SETUP

    def test_floor_is_the_judged_against_the_block(self, quiet_python: None) -> None:
        # 3.10 clears MIN_PYTHON but not the block's 3.12 floor.
        status = remote_python.get_python_status(
            "jt-60sa", environment=_declared("3.10.6", "3.12", [], SETUP)
        )
        assert status.meets_floor is False

    def test_modern_reply_passes(self, quiet_python: None) -> None:
        status = remote_python.get_python_status(
            "jt-60sa",
            environment=_declared(
                "3.12.9",
                "3.12",
                [{"name": "numpy", "importable": True, "error": None}],
                SETUP,
            ),
        )
        assert status.meets_floor is True

    def test_no_block_leaves_floor_unset(self, quiet_python: None) -> None:
        status = remote_python.get_python_status(
            "tcv",
            environment={
                "status": "not_declared",
                "declared": False,
                "modules": [],
                "fix": [],
            },
        )
        assert status.environment["declared"] is False
        assert status.meets_floor is None
        assert status.active_python is None

    def test_absent_environment_probes_nothing(self, quiet_python: None) -> None:
        """Without an environment the local path judges no remote floor."""
        status = remote_python.get_python_status("tcv")
        assert status.environment is None
        assert status.meets_floor is None
        assert status.active_python is None


class TestSetupPythonEnvFloor:
    def test_declared_below_floor_interpreter_is_refused(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Setup refuses a facility whose declared interpreter is below its floor.

        The interpreter the block declares is the one scans run under; building
        a venv beneath it would report success while scans keep using the
        unmet interpreter, so setup stops and names the floor.
        """
        monkeypatch.setattr(
            remote_python,
            "check_tool",
            lambda key, facility=None: {
                "available": key == "uv",
                "version": "1.0" if key == "uv" else None,
            },
        )
        monkeypatch.setattr(remote_python, "run", lambda *a, **k: "")
        monkeypatch.setattr(
            remote_python,
            "probe_remote_environment",
            lambda facility, **kw: _declared("3.10.6", "3.12", [], SETUP),
        )
        result = remote_python.setup_python_env("jt-60sa")
        assert result["success"] is False
        assert "3.12" in result["error"]
        assert all(step["step"] != "create_venv" for step in result["steps"])
