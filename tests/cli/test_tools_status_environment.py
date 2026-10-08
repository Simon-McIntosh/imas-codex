"""``tools status`` environment exit codes and the signal discovery refusal.

The probe is stubbed at both probe sites: the tool summary
(``imas_codex.remote.tools``) and the ``discover`` preflight
(``imas_codex.remote.tools``), plus the python module
(``imas_codex.remote.python``), which binds the probe by direct import. Each
command is asserted to open the remote environment exactly once per
invocation, the install path included.
"""

from __future__ import annotations

import pytest
from click.testing import CliRunner

from imas_codex.cli.discover import discover, sequence
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


def _stub_probe(
    monkeypatch: pytest.MonkeyPatch,
    payload: dict,
    calls: list | None = None,
) -> None:
    """Stub every probe site, recording each call in one shared counter.

    ``remote_python`` binds ``probe_remote_environment`` by direct import, so a
    probe opened from the python module is only visible through its own name.
    Patching both sites lets one invocation be counted across the whole path.
    """

    def _probe(*_a, **_k):
        if calls is not None:
            calls.append(1)
        return payload

    monkeypatch.setattr(remote_tools, "probe_remote_environment", _probe)
    monkeypatch.setattr(remote_python, "probe_remote_environment", _probe)


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
    no_ssh: None, monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    from imas_codex.cli import logging as cli_logging

    monkeypatch.setattr(
        sequence,
        "evaluate_stage",
        lambda stage, facility, config: sequence.StageOutcome(
            stage.name, stage.domain, sequence.RUNNABLE, "ready"
        ),
    )
    monkeypatch.setattr(sequence, "_remaining_count", lambda *args: None)
    monkeypatch.setattr(cli_logging, "configure_cli_logging", lambda *a, **k: None)
    monkeypatch.setattr(
        cli_logging, "get_log_file", lambda *a, **k: tmp_path / "discover.log"
    )
    _stub_probe(
        monkeypatch,
        _declared(
            "3.10.6",
            "3.12",
            [{"name": "numpy", "importable": True, "error": None}],
        ),
    )
    result = CliRunner().invoke(
        discover, ["jt-60sa", "--only", "signals", "--scan-only"]
    )
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


def test_status_probes_the_environment_once(
    no_ssh: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One ``tools status`` invocation opens the remote environment once."""
    calls: list = []
    _stub_probe(
        monkeypatch,
        _declared(
            "3.12.9",
            "3.12",
            [{"name": "numpy", "importable": True, "error": None}],
        ),
        calls=calls,
    )
    result = CliRunner().invoke(tools, ["status", "jt-60sa"])
    assert result.exit_code == 0, result.output
    assert len(calls) == 1, f"expected one probe, saw {len(calls)}"


def test_discover_preflight_probes_the_environment_once(
    no_ssh: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One ``discover`` preflight opens the remote environment once."""
    from imas_codex.cli.discover import common

    calls: list = []
    payload = _declared("3.12.9", "3.12", [])
    payload["meets_floor"] = True
    _stub_probe(monkeypatch, payload, calls=calls)
    common.ensure_remote_environment(
        {
            "id": "jt-60sa",
            "ssh_host": "jt-60sa",
            "remote_environment": {
                "python_command": "python",
                "min_python_version": "3.12",
            },
        }
    )
    assert len(calls) == 1, f"expected one probe, saw {len(calls)}"


def test_install_probes_the_environment_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One ``tools install`` invocation opens the remote environment once.

    ``setup_python_env`` probes the declared environment to judge the floor and
    hands that probe to ``create_venv``; without the hand-off the venv step
    would probe a second time.
    """
    monkeypatch.setattr(
        remote_python,
        "check_tool",
        lambda key, facility=None: {
            "available": key == "uv",
            "version": "1.0" if key == "uv" else None,
        },
    )
    monkeypatch.setattr(
        remote_python,
        "run",
        lambda cmd, **k: "Python 3.12.9" if "--version" in cmd else "",
    )
    calls: list = []
    _stub_probe(monkeypatch, _declared("3.12.9", "3.12", []), calls=calls)
    result = remote_python.setup_python_env("jt-60sa")
    assert any(step["step"] == "create_venv" for step in result["steps"])
    assert len(calls) == 1, f"expected one probe, saw {len(calls)}"
