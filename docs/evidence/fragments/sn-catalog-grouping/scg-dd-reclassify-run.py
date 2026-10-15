#!/usr/bin/env python3
"""Run the DD build's classify phase in bounded passes and log each pass.

A single tool call cannot outlast ten minutes, so the re-classification runs as
one pass per IDS group. This driver invokes the CLI as a subprocess (the CLI
itself must not be piped or redirected from a shell), captures the combined
output, and writes a pass log whose first line is the literal command and whose
last line is ``EXIT=<code>``. The CLI's own DEBUG log under
``~/.local/share/imas-codex/logs/`` is cited beside each pass log.

``run_pass`` is the single pass implementation; the sibling readback driver
imports it rather than carrying a second copy of the same subprocess handling.

Usage:
    python scg-dd-reclassify-run.py --ids a b c [--reset] [--dry] \
        [--timeout 540] [--stop-after-classify] --log <path>
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
VENV = Path("/home/ITER/mcintos/Code/imas-codex/.venv")


def run_pass(
    *,
    ids: list[str] | None,
    reset: bool,
    dry: bool,
    timeout: int,
    stop_after_classify: bool,
    log: str,
    enrich_workers: int | None = None,
) -> int:
    """Invoke one classify pass, logging the command first and ``EXIT=`` last.

    Without ``stop_after_classify`` the pass runs to its natural end under
    ``timeout`` and a timeout is reported as ``EXIT=124``. With it, the pass is
    terminated once its classify phase has written, skipping the graph-wide
    re-embed and cluster phases a reset triggers.
    """
    cmd = [str(VENV / "bin" / "imas-codex"), "imas", "dd", "build"]
    if reset:
        cmd += ["--reset-to", "embedded"]
    if dry:
        cmd += ["--dry-run"]
    if ids:
        cmd += ["--ids", " ".join(ids)]
    if enrich_workers is not None:
        cmd += ["--enrich-workers", str(enrich_workers)]

    log_path = Path(log)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    def _log_has(marker: str) -> bool:
        try:
            return marker in log_path.read_text(errors="ignore")
        except FileNotFoundError:
            return False

    with log_path.open("w") as fh:
        fh.write("command: " + " ".join(cmd) + "\n")
        fh.flush()
        if stop_after_classify:
            proc = subprocess.Popen(
                cmd, cwd=str(REPO), stdout=fh, stderr=subprocess.STDOUT
            )
            deadline = time.monotonic() + timeout
            while proc.poll() is None:
                if time.monotonic() > deadline:
                    fh.write(f"\n[pass driver] timeout after {timeout}s\n")
                    fh.flush()
                    proc.terminate()
                    break
                if _log_has("classify_worker: Classification complete"):
                    fh.write(
                        "\n[pass driver] classify complete; stopping before the "
                        "cluster phase\n"
                    )
                    fh.flush()
                    proc.terminate()
                    break
                time.sleep(2)
            try:
                code = proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                proc.kill()
                code = proc.wait()
        else:
            try:
                proc = subprocess.run(
                    cmd,
                    cwd=str(REPO),
                    stdout=fh,
                    stderr=subprocess.STDOUT,
                    timeout=timeout,
                )
                code = proc.returncode
            except subprocess.TimeoutExpired:
                fh.write(f"\n[pass driver] timeout after {timeout}s\n")
                code = 124
        fh.write(f"\nEXIT={code}\n")
    print(f"wrote {log_path} EXIT={code}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", nargs="*", default=None)
    ap.add_argument("--reset", action="store_true", help="pass --reset-to embedded")
    ap.add_argument("--dry", action="store_true", help="pass --dry-run")
    ap.add_argument("--timeout", type=int, default=540)
    ap.add_argument("--log", required=True)
    ap.add_argument("--enrich-workers", type=int, default=None)
    ap.add_argument(
        "--stop-after-classify",
        action="store_true",
        help="terminate once the classify phase has written, skipping the "
        "graph-wide re-embed and cluster phases a reset triggers",
    )
    args = ap.parse_args()
    return run_pass(
        ids=args.ids,
        reset=args.reset,
        dry=args.dry,
        timeout=args.timeout,
        stop_after_classify=args.stop_after_classify,
        log=args.log,
        enrich_workers=args.enrich_workers,
    )


if __name__ == "__main__":
    sys.exit(main())