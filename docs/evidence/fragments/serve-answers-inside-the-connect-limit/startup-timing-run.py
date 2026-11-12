#!/usr/bin/env python3
"""Time an MCP server from spawn to its answer to ``initialize``.

The spawn recipe is read from the committed ``.mcp.json`` rather than copied by
hand, so the measurement follows whatever the client configuration actually
launches. Each run is a fresh process; the answer is the first newline-delimited
JSON line carrying the ``initialize`` request id.

    startup-timing-run.py --label new --repeat 5 --json startup-timings.json

``--project`` rewrites the ``--project`` argument of the recipe, which is how the
base revision is timed from a separate tree with the same spawn recipe.
"""

from __future__ import annotations

import argparse
import json
import os
import selectors
import subprocess
import sys
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

INITIALIZE = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
        "protocolVersion": "2024-11-05",
        "capabilities": {},
        "clientInfo": {"name": "startup-timing", "version": "1"},
    },
}


def _expand(value: str) -> str:
    return os.path.expandvars(os.path.expanduser(value))


def spawn_recipe(mcp_config: Path, project: str | None = None) -> tuple[list[str], str]:
    config = json.loads(mcp_config.read_text())
    servers = config["mcpServers"]
    first = next(iter(servers))
    entry = servers[first]
    argv = [_expand(entry["command"])] + [_expand(arg) for arg in entry.get("args", [])]
    if project is not None:
        rewritten = []
        previous = None
        for part in argv:
            if previous == "--project":
                rewritten.append(project)
            elif part.startswith("--project="):
                rewritten.append(f"--project={project}")
            else:
                rewritten.append(part)
            previous = part
        argv = rewritten
    cwd = _expand(entry.get("cwd") or project or str(mcp_config.parent))
    return argv, cwd


def _nearest_config() -> Path:
    for parent in Path(__file__).resolve().parents:
        candidate = parent / ".mcp.json"
        if candidate.exists():
            return candidate
    raise SystemExit("no .mcp.json found above this script; pass --mcp-config")


def _loadavg() -> list[float]:
    with open("/proc/loadavg") as handle:
        return [float(part) for part in handle.read().split()[:3]]


def one_run(argv: list[str], cwd: str, timeout: float) -> dict:
    before = _loadavg()
    started = time.monotonic()
    process = subprocess.Popen(
        argv,
        cwd=cwd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    drain: list[bytes] = []

    def _swallow_stderr() -> None:
        # The server logs to stderr; an undrained pipe fills and blocks it.
        assert process.stderr is not None
        for chunk in process.stderr:
            drain.append(chunk)

    reader = threading.Thread(target=_swallow_stderr, daemon=True)
    reader.start()
    answer = None
    error = None
    buffered = b""
    try:
        assert process.stdin is not None and process.stdout is not None
        process.stdin.write(json.dumps(INITIALIZE).encode() + b"\n")
        process.stdin.flush()
        deadline = started + timeout
        selector = selectors.DefaultSelector()
        selector.register(process.stdout, selectors.EVENT_READ)
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not selector.select(remaining):
                break
            chunk = os.read(process.stdout.fileno(), 65536)
            if not chunk:
                break
            buffered += chunk
            while b"\n" in buffered:
                line, buffered = buffered.split(b"\n", 1)
                try:
                    message = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if message.get("id") == 1:
                    answer = message
                    break
            if answer is not None:
                break
    except (BrokenPipeError, OSError) as exc:
        error = f"{type(exc).__name__}: {exc}"
    elapsed = time.monotonic() - started
    after = _loadavg()
    if answer is None and error is None:
        error = "no answer before timeout"
    try:
        process.kill()
    except ProcessLookupError:
        pass
    process.wait(timeout=30)
    reader.join(timeout=5)
    return {
        "seconds_to_answer": round(elapsed, 3),
        "answered": answer is not None,
        "error": error,
        "loadavg_before": before,
        "loadavg_after": after,
        "protocol_version": ((answer or {}).get("result") or {}).get("protocolVersion"),
        "stderr_tail": b"".join(drain).decode("utf-8", "replace")[-400:],
    }


def _time_help(argv: list[str], cwd: str, timeout: float) -> dict:
    before = _loadavg()
    started = time.monotonic()
    result = subprocess.run(
        [*argv, "serve", "--help"],
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    elapsed = time.monotonic() - started
    return {
        "seconds_to_answer": round(elapsed, 3),
        "answered": result.returncode == 0,
        "error": None if result.returncode == 0 else f"exit {result.returncode}",
        "loadavg_before": before,
        "loadavg_after": _loadavg(),
        "protocol_version": None,
        "stderr_tail": result.stderr[-400:],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True, help="arm label, e.g. new or base")
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument(
        "--project", default=None, help="override --project in the recipe"
    )
    parser.add_argument(
        "--mode",
        choices=["initialize", "help"],
        default="initialize",
        help="time the answer to initialize, or the exit of `serve --help`",
    )
    parser.add_argument(
        "--mcp-config",
        default=None,
        help="client configuration to read the spawn recipe from (default: the nearest .mcp.json above this script)",
    )
    parser.add_argument("--json", default=None, help="append results to this JSON file")
    parser.add_argument("--note", default="")
    args = parser.parse_args()

    mcp_config = Path(args.mcp_config) if args.mcp_config else _nearest_config()
    argv, cwd = spawn_recipe(mcp_config, args.project)
    record = {
        "label": args.label,
        "note": args.note,
        "argv": argv,
        "cwd": cwd,
        "host": os.uname().nodename,
        "utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "runs": [],
    }
    for index in range(args.repeat):
        if args.mode == "help":
            run = _time_help(argv, cwd, args.timeout)
        else:
            run = one_run(argv, cwd, args.timeout)
        print(
            f"{args.label} run {index + 1}/{args.repeat}: "
            f"{run['seconds_to_answer']:.2f} s (load {run['loadavg_before'][0]:.1f})"
            + (f" ERROR {run['error']}" if run["error"] else ""),
            flush=True,
        )
        record["runs"].append(run)

    times = sorted(run["seconds_to_answer"] for run in record["runs"])
    record["median_seconds"] = times[len(times) // 2]
    record["max_seconds"] = times[-1]
    print(
        f"{args.label} median {record['median_seconds']:.2f} s over {len(times)} runs"
    )

    if args.json:
        path = Path(args.json)
        existing = json.loads(path.read_text()) if path.exists() else []
        existing.append(record)
        path.write_text(json.dumps(existing, indent=2) + "\n")
        print(f"appended to {path}")
    return 0 if all(run["answered"] for run in record["runs"]) else 1


if __name__ == "__main__":
    sys.exit(main())
