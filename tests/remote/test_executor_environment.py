"""Real subprocess controls for caller-supplied local script environments."""

import json
import shlex
import subprocess
import sys
from unittest.mock import patch

import pytest

from iter_codex.remote import executor


@pytest.fixture
def scripts(tmp_path):
    """Supply small scripts while retaining real interpreter processes."""
    with patch("importlib.resources.files", return_value=tmp_path):
        yield tmp_path


def test_local_setup_environment(run_local, scripts, monkeypatch):
    monkeypatch.delenv("ARCHIVE_EXECUTOR_CONTROL", raising=False)
    (scripts / "environment.py").write_text(
        "import json, os, sys\n"
        "print(json.dumps([os.environ.get('ARCHIVE_EXECUTOR_CONTROL'), "
        "json.load(sys.stdin)]))\n"
    )
    result = run_local(
        "environment.py",
        input_data={"selector": "demo with 'quotes' and $symbols"},
        python_command=sys.executable,
        setup_commands=["export ARCHIVE_EXECUTOR_CONTROL='module environment'"],
    )
    assert json.loads(result) == [
        "module environment",
        {"selector": "demo with 'quotes' and $symbols"},
    ]


def test_local_setup_makes_client_importable(run_local, scripts, monkeypatch):
    monkeypatch.delenv("PYTHONPATH", raising=False)
    client_dir = scripts / "client with spaces and 'quotes'"
    client_dir.mkdir()
    (client_dir / "archive_client.py").write_text("VALUE = 'client available'\n")
    (scripts / "client.py").write_text(
        "import archive_client\nprint(archive_client.VALUE)\n"
    )
    result = run_local(
        "client.py",
        python_command=sys.executable,
        setup_commands=[f"export PYTHONPATH={shlex.quote(str(client_dir))}"],
    )
    assert result == "client available"
    with pytest.raises(subprocess.CalledProcessError) as error:
        run_local("client.py", python_command=sys.executable)
    assert "ModuleNotFoundError: No module named 'archive_client'" in error.value.stderr


@pytest.mark.parametrize("setup_commands", [None, []], ids=["omitted", "empty"])
def test_local_without_setup_preserves_direct_launch(
    run_local, scripts, setup_commands
):
    (scripts / "direct.py").write_text(
        "import json, sys\n"
        "print(json.dumps(json.load(sys.stdin)))\n"
        "print('diagnostic', file=sys.stderr)\n"
    )
    with (
        patch.object(executor.subprocess, "run", wraps=subprocess.run) as sync_launch,
        patch.object(
            executor, "_communicate_async", wraps=executor._communicate_async
        ) as async_launch,
    ):
        result = run_local(
            "direct.py",
            input_data=["demo", 42],
            python_command=sys.executable,
            setup_commands=setup_commands,
        )
    launch = sync_launch if sync_launch.called else async_launch
    assert launch.call_args.args[0][:2] == [sys.executable, "-c"]
    assert json.loads(result) == ["demo", 42]


def test_local_setup_failure_prevents_script(run_local, scripts, tmp_path):
    marker = tmp_path / "executed"
    (scripts / "marker.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).touch()\n"
    )
    with pytest.raises(subprocess.CalledProcessError) as error:
        run_local("marker.py", python_command=sys.executable, setup_commands=["exit 7"])
    assert error.value.returncode == 7
    assert not marker.exists()


def test_local_setup_can_select_interpreter(run_local, scripts, tmp_path):
    interpreter_dir = tmp_path / "interpreter with spaces and 'quotes'"
    interpreter_dir.mkdir()
    (interpreter_dir / "archive-python").symlink_to(sys.executable)
    (scripts / "interpreter.py").write_text("print('selected interpreter')\n")
    result = run_local(
        "interpreter.py",
        python_command="archive-python",
        setup_commands=[f'export PATH={shlex.quote(str(interpreter_dir))}:"$PATH"'],
    )
    assert result == "selected interpreter"
