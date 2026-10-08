"""Import order checks for discovery engines and CLI registration."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


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
