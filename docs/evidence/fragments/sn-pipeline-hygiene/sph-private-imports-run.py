"""Negative controls for the private-name rule and the single-owner git helpers.

Two checks must be able to fail:

1. Restoring one closed private import in a copied package makes the
   private-name rule name that site.
2. The pre-repair revision, which held a second copy of the git helpers in
   ``release_notes.py``, makes the no-second-copy check fail.

The first control mutates a copy of the package in place; the second builds a
tree from the pre-repair revision with ``git archive`` and runs the check
against it. Both fail as expected when the guards work.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
PACKAGE = REPO / "imas_codex/standard_names"
RULE_TEST = (
    "tests/standard_names/test_package_import_boundaries.py"
    "::test_no_private_name_imports_between_top_level_modules"
)
REUSE_TEST = (
    "tests/standard_names/test_fallback_pr_body_links.py"
    "::test_release_notes_does_not_define_a_second_git_helper_copy"
)
CACHE = Path("~/.cache/imas-codex-s19").expanduser()
RULE_LOG = CACHE / "sph-private-imports-negative-control.log"
REUSE_LOG = CACHE / "sph-private-imports-reuse-control.log"
CLOSED = (
    "from imas_codex.standard_names.unit_overrides import load_rules as _load_rules"
)
RESTORED = "from imas_codex.standard_names.unit_overrides import _load_rules"
PRE_REPAIR = "611ba81d3197474abf0f11979f634d4dcc53a5e7"
GENERATED = (
    "imas_codex/graph/models.py",
    "imas_codex/graph/dd_models.py",
    "imas_codex/config/models.py",
    "imas_codex/graph/schema_context_data.py",
    "agents/schema-reference.md",
)


def _pytest(target: str, *, cwd: Path, env: dict[str, str], log: Path) -> int:
    command = [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", target]
    log.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        command, cwd=cwd, env=env, capture_output=True, text=True, timeout=600
    )
    output = proc.stdout + proc.stderr
    log.write_text(
        f"# command: {' '.join(command)}\n"
        f"# cwd: {cwd}\n"
        f"# PYTHONPATH: {env.get('PYTHONPATH', '')}\n\n"
        f"{output}\nEXIT={proc.returncode}\n",
        encoding="utf-8",
    )
    print(output)
    print(f"EXIT={proc.returncode}")
    return proc.returncode


def rule_control(scratch: Path) -> int:
    """Restore a closed private import; the rule must name the site."""
    target = scratch / "imas_codex/standard_names"
    target.mkdir(parents=True)
    for path in PACKAGE.glob("*.py"):
        shutil.copy2(path, target / path.name)
    victim = target / "legacy_authority.py"
    text = victim.read_text(encoding="utf-8")
    if CLOSED not in text:
        print(f"mutation anchor missing in {victim}")
        return 2
    victim.write_text(text.replace(CLOSED, RESTORED), encoding="utf-8")
    print(f"mutation: restored {RESTORED!r} in {victim.name}")
    env = dict(os.environ, SN_STANDARD_NAMES_ROOT=str(target))
    return 0 if _pytest(RULE_TEST, cwd=REPO, env=env, log=RULE_LOG) != 0 else 1


def reuse_control(scratch: Path) -> int:
    """Build the pre-repair tree; the no-second-copy check must fail there."""
    tree = scratch / "pre-repair"
    tree.mkdir(parents=True)
    archive = extract_pre_repair(tree)
    if archive != 0:
        return archive
    for relative in GENERATED:
        source = REPO / relative
        if source.is_file():
            destination = tree / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
    env = dict(os.environ, PYTHONPATH=str(tree))
    print(f"pre-repair tree: {tree} at {PRE_REPAIR}")
    return 0 if _pytest(REUSE_TEST, cwd=tree, env=env, log=REUSE_LOG) != 0 else 1


def extract_pre_repair(tree: Path) -> int:
    """Extract the pre-repair revision into *tree*."""
    proc = subprocess.run(
        ["git", "-C", str(REPO), "archive", PRE_REPAIR],
        capture_output=True,
        timeout=300,
    )
    if proc.returncode != 0:
        print(proc.stderr.decode(errors="replace"))
        return 2
    extract = subprocess.run(
        ["tar", "-x", "-C", str(tree)], input=proc.stdout, timeout=300
    )
    return extract.returncode


def main() -> int:
    scratch = Path(
        tempfile.mkdtemp(prefix="sph-private-imports-", dir=os.environ.get("TMPDIR"))
    )
    try:
        rule = rule_control(scratch)
        reuse = reuse_control(scratch)
        print(f"rule control EXIT={rule} reuse control EXIT={reuse}")
        return rule or reuse
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
