"""Facility remote environment resolution.

Covers the merge the facility-aware layer performs before a remote script
runs: the facility ``remote_environment`` block supplies the default, a data
system's own ``python_command`` / ``setup_commands`` override it for that
system's scripts, and an explicit caller argument wins over both.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from imas_codex.discovery.signals.parallel import _facility_scanner_config
from imas_codex.remote import tools as remote_tools
from imas_codex.remote.environment import resolve_remote_environment

FACILITIES_DIR = (
    Path(__file__).parent.parent.parent / "imas_codex" / "config" / "facilities"
)


def _load_facility(name: str) -> dict:
    with open(FACILITIES_DIR / f"{name}.yaml") as handle:
        return yaml.safe_load(handle)


@pytest.fixture
def synthetic_config() -> dict:
    return {
        "remote_environment": {
            "python_command": "python",
            "setup_commands": ["module load python/3.12"],
            "min_python_version": "3.12",
            "required_modules": [
                "numpy",
                {"name": "eddb_pwrapper", "path": "/analysis/src/eddb"},
            ],
        },
        "data_systems": {
            "edas": {"api_path": "/analysis/src/eddb"},
            "mdsplus": {
                "python_command": "python3",
                "setup_commands": ["source /etc/profile.d/mdsplus.sh"],
            },
        },
    }


class TestResolver:
    def test_facility_block_provides_default(self, synthetic_config: dict) -> None:
        env = resolve_remote_environment(synthetic_config)
        assert env.python_command == "python"
        assert env.setup_commands == ("module load python/3.12",)
        assert env.min_python_version == "3.12"

    def test_data_system_without_keys_inherits_block(
        self, synthetic_config: dict
    ) -> None:
        env = resolve_remote_environment(synthetic_config, "edas")
        assert env.python_command == "python"
        assert env.setup_commands == ("module load python/3.12",)

    def test_data_system_override_wins(self, synthetic_config: dict) -> None:
        env = resolve_remote_environment(synthetic_config, "mdsplus")
        assert env.python_command == "python3"
        assert env.setup_commands == ("source /etc/profile.d/mdsplus.sh",)

    def test_required_modules_parse_path(self, synthetic_config: dict) -> None:
        env = resolve_remote_environment(synthetic_config)
        assert [(m.name, m.path) for m in env.required_modules] == [
            ("numpy", None),
            ("eddb_pwrapper", "/analysis/src/eddb"),
        ]

    def test_empty_config_defaults_to_python3(self) -> None:
        env = resolve_remote_environment({})
        assert env.python_command == "python3"
        assert env.setup_commands == ()
        assert env.min_python_version is None
        assert env.required_modules == ()

    def test_jt60sa_block_carries_the_recipe(self) -> None:
        env = resolve_remote_environment(_load_facility("jt-60sa"), "edas")
        assert env.python_command == "python"
        assert env.setup_commands == (
            "module unload python/3.5.6",
            "module load python/3.12",
        )
        assert env.min_python_version == "3.12"
        names = [m.name for m in env.required_modules]
        assert "numpy" in names and "eddb_pwrapper" in names


class TestToolsFacilityWrapper:
    def test_caller_args_win_over_resolved_env(
        self, monkeypatch: pytest.MonkeyPatch, synthetic_config: dict
    ) -> None:
        captured: dict = {}

        def fake_executor(script_name, input_data=None, **kwargs):
            captured.update(kwargs)
            return "ok"

        monkeypatch.setattr(
            "imas_codex.remote.tools._executor_run_python_script", fake_executor
        )
        monkeypatch.setattr(
            remote_tools, "_facility_config", lambda _f: synthetic_config
        )
        monkeypatch.setattr(remote_tools, "_resolve_ssh_host", lambda _f: "host")

        remote_tools.run_python_script(
            "x.py",
            facility="jt-60sa",
            data_system="edas",
            python_command="explicit",
            setup_commands=["explicit-setup"],
        )

        assert captured["python_command"] == "explicit"
        assert captured["setup_commands"] == ["explicit-setup"]

    def test_environment_passed_to_executor(
        self, monkeypatch: pytest.MonkeyPatch, synthetic_config: dict
    ) -> None:
        captured: dict = {}

        def fake_executor(script_name, input_data=None, **kwargs):
            captured.update(kwargs)
            return "ok"

        monkeypatch.setattr(
            "imas_codex.remote.tools._executor_run_python_script", fake_executor
        )
        monkeypatch.setattr(
            remote_tools, "_facility_config", lambda _f: synthetic_config
        )
        monkeypatch.setattr(remote_tools, "_resolve_ssh_host", lambda _f: "host")

        remote_tools.run_python_script("x.py", facility="jt-60sa", data_system="edas")

        assert captured["python_command"] == "python"
        assert captured["setup_commands"] == ["module load python/3.12"]


class TestScannerConfig:
    def test_block_supplies_commands_when_data_system_declares_none(
        self, synthetic_config: dict
    ) -> None:
        merged = _facility_scanner_config(synthetic_config, "edas")
        assert merged["python_command"] == "python"
        assert merged["setup_commands"] == ["module load python/3.12"]

    def test_data_system_commands_are_kept(self, synthetic_config: dict) -> None:
        merged = _facility_scanner_config(synthetic_config, "mdsplus")
        assert merged["python_command"] == "python3"
        assert merged["setup_commands"] == ["source /etc/profile.d/mdsplus.sh"]
