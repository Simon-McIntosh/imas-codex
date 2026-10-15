#!/usr/bin/env python3
"""Run the DD build's classify phase in bounded passes and log each pass.

A single tool call cannot outlast ten minutes, so the re-classification runs as
one pass per IDS group. This driver invokes the CLI as a subprocess (the CLI
itself must not be piped or redirected from a shell), captures the combined
output, and writes a pass log whose first line is the literal command and whose
last line is ``EXIT=<code>``. The CLI's own DEBUG log under
``~/.local/share/imas-codex/logs/`` is cited beside each pass log.

Usage:
    python scg-dd-reclassify-run.py --ids a b c [--reset] [--dry] \
        [--timeout 540] --log <path>
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
VENV = Path("/home/ITER/mcintos/Code/imas-codex/.venv")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", nargs="*", default=None)
    ap.add_argument("--reset", action="store_true", help="pass --reset-to embedded")
    ap.add_argument("--dry", action="store_true", help="pass --dry-run")
    ap.add_argument("--timeout", type=int, default=540)
    ap.add_argument("--log", required=True)
    ap.add_argument("--enrich-workers", type=int, default=None)
    args = ap.parse_args()

    cmd = [str(VENV / "bin" / "imas-codex"), "imas", "dd", "build"]
    if args.reset:
        cmd += ["--reset-to", "embedded"]
    if args.dry:
        cmd += ["--dry-run"]
    if args.ids:
        cmd += ["--ids", " ".join(args.ids)]
    if args.enrich_workers is not None:
        cmd += ["--enrich-workers", str(args.enrich_workers)]

    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as fh:
        fh.write("command: " + " ".join(cmd) + "\n")
        fh.flush()
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(REPO),
                stdout=fh,
                stderr=subprocess.STDOUT,
                timeout=args.timeout,
            )
            code = proc.returncode
        except subprocess.TimeoutExpired:
            fh.write(f"\n[pass driver] timeout after {args.timeout}s\n")
            code = 124
        fh.write(f"\nEXIT={code}\n")
    print(f"wrote {log_path} EXIT={code}")
    return 0


if __name__ == "__main__":
    sys.exit(main())