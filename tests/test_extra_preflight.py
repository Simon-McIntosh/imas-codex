"""Subprocess coverage for the test-extra dependency preflight."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

_DEPENDENT_TEST = (
    "tests/embeddings/test_prompt_name.py::test_embed_request_accepts_prompt_name"
)
_INSTALL_INSTRUCTION = "uv sync --extra test"


def _run_dependent_test(
    *, python_path: Path | None = None
) -> subprocess.CompletedProcess:
    environment = os.environ.copy()
    if python_path is not None:
        existing_path = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = os.pathsep.join(
            part for part in (str(python_path), existing_path) if part
        )
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-p",
            "no:cacheprovider",
            _DEPENDENT_TEST,
        ],
        cwd=Path(__file__).resolve().parent.parent,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )


def test_missing_torch_exits_once_with_install_instruction(tmp_path):
    """A missing test extra produces one actionable exit, not import traces."""
    (tmp_path / "torch.py").write_text(
        "raise ModuleNotFoundError(\"No module named 'torch'\")\n",
        encoding="utf-8",
    )

    result = _run_dependent_test(python_path=tmp_path)
    output = result.stdout + result.stderr

    assert result.returncode != 0
    assert output.count(_INSTALL_INSTRUCTION) == 1
    assert "ModuleNotFoundError" not in output


def test_available_torch_keeps_preflight_silent():
    """A provisioned test extra does not alter successful collection."""
    result = _run_dependent_test()
    output = result.stdout + result.stderr

    assert result.returncode == 0, output
    assert _INSTALL_INSTRUCTION not in output
