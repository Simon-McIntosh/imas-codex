"""``tools status`` environment exit codes and the ``discover signals`` refusal.

The probe is stubbed at both modules that own it: the tool summary
(``imas_codex.remote.tools``) and the Python status (``imas_codex.remote.python``).
"""

from __future__ import annotations

import pytest
from click.testing import CliRunner

from imas_codex.cli.discover.signals import signals
from imas_codex.cli.tools import tools
from imas_codex.remote import python as remote_python, tools as remote_tools

SETUP = ["module unload python/3.5.6", "module load python/3.12"]


def _declared(python_version: str, floor: str, modules: list[dict]) -> dict:
    return {
        "status": "declared",
        "declared": True,
        "floor": floor,
        "python_version": python_version,
        "meets_floor": None,
        "modules": modules,
        "fix": list(SETUP),
        "error": None,
        "python_command": "python",
        "setup_commands": list(SETUP),
    }


@pytest.fixture
def no_ssh(monkeypatch: pytest.MonkeyPatch) -> None:
    """Neutralise every SSH-backed check the two commands touch."""
    monkeypatch.setattr(
        remote_python,
        "check_tool",
        lambda key, facility=None: {"available": False, "version": None},
    )
    monkeypatch.setattr(remote_python, "run", lambda *a, **k: "")
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


def _stub_probe(monkeypatch: pytest.MonkeyPatch, payload: dict) -> None:
    monkeypatch.setattr(
        remote_tools, "probe_remote_environment", lambda *a, **k: payload
    )
    monkeypatch.setattr(
        remote_python, "probe_remote_environment", lambda *a, **k: payload
    )


def test_status_passes_on_met_environment(
    no_ssh: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_probe(
        monkeypatch,
        _declared(
            "3.12.9",
            "3.12",
            [
                {"name": "numpy", "importable": True, "error": None},
                {"name": "eddb_pwrapper", "importable": True, "error": None},
            ],
        ),
    )
    result = CliRunner().invoke(tools, ["status", "jt-60sa"])
    assert result.exit_code == 0, result.output
    assert "3.12.9" in result.output
    assert "numpy" in result.output


def test_status_fails_and_shows_fix_on_unmet_floor(
    no_ssh: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_probe(
        monkeypatch,
        _declared(
            "3.10.6",
            "3.12",
            [
                {
                    "name": "eddb_pwrapper",
                    "importable": False,
                    "error": "ModuleNotFoundError: No module named 'eddb_pwrapper'",
                }
            ],
        ),
    )
    result = CliRunner().invoke(
        tools, ["status", "jt-60sa", "--no-setup", "--python-command", "python3"]
    )
    assert result.exit_code != 0
    assert "3.12" in result.output
    assert "module load python/3.12" in result.output


def test_discover_signals_refuses_on_failing_probe(
    no_ssh: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_probe(
        monkeypatch,
        _declared(
            "3.10.6",
            "3.12",
            [{"name": "numpy", "importable": True, "error": None}],
        ),
    )
    result = CliRunner().invoke(signals, ["jt-60sa"])
    assert result.exit_code != 0
    assert "3.12" in result.output
    assert "module load python/3.12" in result.output


def test_discover_environment_skips_facility_without_block(
    no_ssh: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    from imas_codex.cli.discover import common

    # A facility declaring no block is left alone (nothing probed, no refusal).
    calls: list = []
    monkeypatch.setattr(
        remote_tools,
        "probe_remote_environment",
        lambda *a, **k: (
            calls.append(1) or {"status": "not_declared", "declared": False}
        ),
    )
    common.ensure_remote_environment({"id": "tcv", "ssh_host": "tcv"})
    assert calls == []
