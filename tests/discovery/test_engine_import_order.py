"""Import order checks for discovery engines and CLI registration."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


def _run_fresh_interpreter(source: str) -> subprocess.CompletedProcess[str]:
    root = Path(__file__).resolve().parents[2]
    return subprocess.run(
        [sys.executable, "-c", source],
        cwd=root,
        env={**os.environ, "PYTHONPATH": str(root)},
        capture_output=True,
        text=True,
        check=False,
    )


def test_engine_import_does_not_load_cli() -> None:
    result = _run_fresh_interpreter(
        "import sys\n"
        "import imas_codex.discovery.base.engine as engine\n"
        "assert 'imas_codex.cli' not in sys.modules, "
        "sorted(name for name in sys.modules if name.startswith('imas_codex.cli'))\n"
        "print(engine.__file__)\n"
    )
    assert result.returncode == 0, result.stderr


def test_cli_then_discovery_engines_import() -> None:
    result = _run_fresh_interpreter(
        "import imas_codex.cli.discover\n"
        "import imas_codex.discovery.wiki.parallel\n"
        "import imas_codex.discovery.paths.parallel\n"
        "import imas_codex.discovery.code.parallel\n"
    )
    assert result.returncode == 0, result.stderr


_DEFERRED_MODULES = (
    "imas_codex.cli.compute",
    "imas_codex.cli.config_cli",
    "imas_codex.cli.credentials",
    "imas_codex.cli.discover",
    "imas_codex.cli.embed",
    "imas_codex.cli.facilities",
    "imas_codex.cli.graph",
    "imas_codex.cli.host",
    "imas_codex.cli.imas_dd",
    "imas_codex.cli.llm_cli",
    "imas_codex.cli.release",
    "imas_codex.cli.sn",
    "imas_codex.cli.tools",
    "imas_codex.cli.tunnel",
)


def _loaded_from(models: tuple[str, ...]) -> str:
    return (
        "loaded = sorted(m for m in sys.modules if m in " + repr(models) + ")\n"
        "assert not loaded, loaded\n"
        "print('clean')\n"
    )


def test_help_imports_no_command_module() -> None:
    """`imas-codex --help` answers from the command table alone."""
    result = _run_fresh_interpreter(
        "import sys\n"
        "from click.testing import CliRunner\n"
        "from imas_codex.cli import main\n"
        "outcome = CliRunner().invoke(main, ['--help'])\n"
        "assert outcome.exit_code == 0, outcome.output\n"
        "assert 'Commands:' in outcome.output, outcome.output\n"
        + _loaded_from(_DEFERRED_MODULES)
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.timeout(300)
def test_initialize_path_imports_only_the_serve_command() -> None:
    """Resolving and running `serve` loads no other command module.

    The server object that answers ``initialize`` is built exactly as the
    command builds it, after the group has resolved ``serve`` the way the
    command line does. Neither the other command modules nor the language-model
    client library may be on that path, whether or not the deferred warm-up has
    started.
    """
    result = _run_fresh_interpreter(
        "import sys\n"
        "import click\n"
        "from imas_codex.cli import main\n"
        "from imas_codex.llm.server import AgentsServer\n"
        "served = main.get_command(click.Context(main), 'serve')\n"
        "assert served is not None\n"
        "server = AgentsServer(dd_only=True)\n"
        "assert server is not None\n"
        "assert 'litellm' not in sys.modules, 'litellm imported'\n"
        + _loaded_from(_DEFERRED_MODULES)
    )
    assert result.returncode == 0, result.stderr
