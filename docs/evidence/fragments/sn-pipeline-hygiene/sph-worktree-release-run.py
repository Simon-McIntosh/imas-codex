"""Measure whether a release pre-flight admits a detached worktree head.

Builds a scratch repository whose main branch is checked out nowhere, then a
detached worktree at the tip of main (admitted) and at a commit main has not
seen (refused). Prints one JSON document describing what each guard did, so
the record shows the guard admitting and refusing rather than only passing.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import subprocess
import tempfile
from pathlib import Path


def git(*args: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=False
    )


def build(tmp: Path) -> tuple[Path, Path]:
    repo = tmp / "isnc"
    repo.mkdir()
    git("init", "-q", "-b", "main", cwd=repo)
    git("config", "user.email", "probe@example.invalid", cwd=repo)
    git("config", "user.name", "probe", cwd=repo)
    (repo / "README.md").write_text("x\n", encoding="utf-8")
    git("add", "README.md", cwd=repo)
    git("commit", "-qm", "init", cwd=repo)
    return repo, tmp / "wt"


def guard_verdict(name: str, call) -> dict:
    try:
        call()
    except Exception as exc:  # the refusal is the datum, not a failure
        return {"guard": name, "verdict": "refused", "error": str(exc)}
    return {"guard": name, "verdict": "admitted", "error": None}


def main() -> int:
    from imas_codex.cli.release import _check_on_main as cli_guard
    from imas_codex.standard_names.catalog_release import (
        _check_on_main as catalog_guard,
    )

    def cli_at(path: Path):
        def call():
            previous = Path.cwd()
            os.chdir(path)
            try:
                # The guard reports its verdict on stdout; keep the JSON clean.
                with contextlib.redirect_stdout(io.StringIO()):
                    cli_guard()
            finally:
                os.chdir(previous)

        return call

    report: dict = {"detached_head": {}, "merged": [], "unmerged": []}

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        repo, wt = build(tmp)
        added = git("worktree", "add", "--detach", str(wt), "main", cwd=repo)
        report["detached_head"] = {
            "worktree_add_rc": added.returncode,
            "branch_show_current": git("branch", "--show-current", cwd=wt).stdout,
            "ancestor_of_main": git(
                "merge-base", "--is-ancestor", "HEAD", "main", cwd=wt
            ).returncode
            == 0,
        }
        report["merged"] = [
            guard_verdict("catalog_release._check_on_main", lambda: catalog_guard(wt)),
            guard_verdict("cli.release._check_on_main", cli_at(wt)),
        ]

        (wt / "unreviewed.txt").write_text("not on main\n", encoding="utf-8")
        git("add", "unreviewed.txt", cwd=wt)
        git("commit", "-qm", "unreviewed work", cwd=wt)
        report["unmerged"] = [
            guard_verdict("catalog_release._check_on_main", lambda: catalog_guard(wt)),
            guard_verdict("cli.release._check_on_main", cli_at(wt)),
        ]

    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
