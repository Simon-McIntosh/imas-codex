#!/usr/bin/env python3
"""Measure the gap-rule and geometry test ids at the base and at this node's head.

Usage: sph-red-gap-rules-run.py {base|head} <log-path>

The ``base`` arm extracts the recorded base revision into a scratch tree, so
the two arms differ only in the change under test.  Each arm runs the whole
``tests/standard_names`` selection on whichever host runs it, and the log's
first lines name the revision, the tree, the command and the module actually
imported, so a reader can tell which tree answered.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

BASE_SHA = "5e28aa872002b981abfba731a0d27251fbfefeaf"
MAIN_CHECKOUT = Path("/home/ITER/mcintos/Code/imas-codex")
WORKTREE = Path(
    "/home/ITER/mcintos/Code/.reckon-worktrees/imas-codex-c994bf55fb01/"
    "4f5c460d-ed3a-4460-943c-233208992958/sph-red-gap-rules"
)
GENERATED = (
    "imas_codex/graph/models.py",
    "imas_codex/graph/dd_models.py",
    "imas_codex/config/models.py",
    "imas_codex/graph/schema_context_data.py",
    "agents/schema-reference.md",
)
SELECTION = "tests/standard_names"


def provision(tree: Path) -> None:
    """Give a scratch tree the shared environment and the gitignored inputs."""
    venv = tree / ".venv"
    if not venv.exists():
        venv.symlink_to(MAIN_CHECKOUT / ".venv")
    env = tree / ".env"
    if not env.exists():
        env.symlink_to(MAIN_CHECKOUT / ".env")
    for rel in GENERATED:
        dest = tree / rel
        if dest.exists():
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(MAIN_CHECKOUT / rel, dest)


def main() -> int:
    mode, log_path = sys.argv[1], Path(sys.argv[2])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if mode == "base":
        tree = Path(os.environ["TMPDIR"]) / "sph-red-gap-rules-base"
        if tree.exists():
            shutil.rmtree(tree)
        tree.mkdir(parents=True)
        archive = subprocess.run(
            ["git", "-C", str(MAIN_CHECKOUT), "archive", BASE_SHA],
            check=False,
            capture_output=True,
        )
        if archive.returncode != 0:
            sys.stderr.write(archive.stderr.decode())
            return archive.returncode
        subprocess.run(["tar", "-x", "-C", str(tree)], input=archive.stdout, check=True)
    else:
        tree = WORKTREE

    provision(tree)

    env = dict(os.environ)
    env.update(
        UV_PROJECT_ENVIRONMENT=str(MAIN_CHECKOUT / ".venv"),
        PYTHONPATH=str(tree),
        TMPDIR=os.environ["TMPDIR"],
        UV_NO_SYNC="1",
    )
    probe = subprocess.run(
        [
            str(MAIN_CHECKOUT / ".venv/bin/python"),
            "-c",
            "import os, imas_standard_names, imas_codex.standard_names.segments as m;"
            "print('isn', getattr(imas_standard_names, '__version__', '?'));"
            "print('module.__file__', m.__file__);print('cwd', os.getcwd())",
        ],
        cwd=str(tree),
        env=env,
        capture_output=True,
        text=True,
    )
    header = [
        f"# mode {mode}\n",
        f"# revision {BASE_SHA if mode == 'base' else 'head'}\n",
        f"# tree {tree}\n",
        f"# command python -m pytest {SELECTION}\n",
    ] + [f"# {line}\n" for line in probe.stdout.splitlines()]
    print("".join(header), end="")

    with log_path.open("w") as log:
        log.writelines(header)
        log.flush()
        result = subprocess.run(
            [
                str(MAIN_CHECKOUT / ".venv/bin/python"),
                "-m",
                "pytest",
                "-p",
                "no:cacheprovider",
                "-q",
                "--no-header",
                SELECTION,
            ],
            cwd=str(tree),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        log.write(f"EXIT={result.returncode}\n")
    print(f"# wrote {log_path} exit={result.returncode}")
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())