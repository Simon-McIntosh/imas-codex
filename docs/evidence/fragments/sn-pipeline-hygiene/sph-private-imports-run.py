"""Negative control for the private-name import rule.

Copies the package's top-level modules to a scratch directory, restores one
closed private import, and runs the rule against the mutated copy. The rule
must fail naming the restored site.
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
TEST = "tests/standard_names/test_package_import_boundaries.py::test_no_private_name_imports_between_top_level_modules"
LOG = Path(
    "~/.cache/imas-codex-s19/sph-private-imports-negative-control.log"
).expanduser()
CLOSED = (
    "from imas_codex.standard_names.unit_overrides import load_rules as _load_rules"
)
RESTORED = "from imas_codex.standard_names.unit_overrides import _load_rules"


def main() -> int:
    scratch = Path(
        tempfile.mkdtemp(prefix="sph-private-imports-", dir=os.environ.get("TMPDIR"))
    )
    try:
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

        env = dict(os.environ)
        env["SN_STANDARD_NAMES_ROOT"] = str(target)
        command = [
            sys.executable,
            "-m",
            "pytest",
            "-p",
            "no:cacheprovider",
            TEST,
        ]
        LOG.parent.mkdir(parents=True, exist_ok=True)
        proc = subprocess.run(
            command,
            cwd=REPO,
            env=env,
            capture_output=True,
            text=True,
            timeout=600,
        )
        output = proc.stdout + proc.stderr
        LOG.write_text(
            f"# command: {' '.join(command)}\n"
            f"# cwd: {REPO}\n"
            f"# SN_STANDARD_NAMES_ROOT: {target}\n\n"
            f"{output}\nEXIT={proc.returncode}\n",
            encoding="utf-8",
        )
        print(output)
        print(f"EXIT={proc.returncode}")
        return 0 if proc.returncode != 0 else 1
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
