#!/usr/bin/env python3
"""Drive the DD path re-classification and read its result back from the graph.

Two subcommands:

``pass``
    Run one bounded re-classification pass over a set of IDS groups by invoking
    the DD build's classify phase as a subprocess. The CLI must not be piped or
    redirected from a shell, and a single tool call cannot outlast ten minutes,
    so the pass is invoked through this driver and each pass log carries the
    literal command on its first line and ``EXIT=<code>`` on its last. The
    classify phase writes its result once, after every batch completes, so a
    pass that is cut off before its write leaves the graph unchanged -- only
    its spend is lost.

``readback``
    Read the classified result back from the graph under one bounded
    connection: the ``domain_source`` distribution, the ``domain_source`` x
    ``physics_domain`` matrix for the LLM-tier paths, how many ``summary`` paths
    sit at ``data_management``, and the rule-labelled paths scored against the
    graph.

Usage:
    python scg-dd-reclassify-finish-run.py pass --ids a b c [--reset] [--dry] \
        [--timeout 540] --log <path>
    python scg-dd-reclassify-finish-run.py readback --labels <json> --out <json>
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

VENV = Path("/home/ITER/mcintos/Code/imas-codex/.venv")

# domain_source values the two-tier classifier writes for a path it judged
# itself, as opposed to one that inherited a parent's domain or was left null
# for infrastructure metadata.
LLM_TIER_SOURCES = ("jev_tree", "llm_classified")


def _find_repo(start: Path) -> Path:
    for parent in [start, *start.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    raise SystemExit("could not locate the repository root")


REPO = _find_repo(Path(__file__).resolve())


def _run_pass(args: argparse.Namespace) -> int:
    cmd = [str(VENV / "bin" / "imas-codex"), "imas", "dd", "build"]
    if args.reset:
        cmd += ["--reset-to", "embedded"]
    if args.dry:
        cmd += ["--dry-run"]
    if args.ids:
        cmd += ["--ids", " ".join(args.ids)]

    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    def _log_has(marker: str) -> bool:
        try:
            return marker in log_path.read_text(errors="ignore")
        except FileNotFoundError:
            return False

    with log_path.open("w") as fh:
        fh.write("command: " + " ".join(cmd) + "\n")
        fh.flush()
        proc = subprocess.Popen(
            cmd, cwd=str(REPO), stdout=fh, stderr=subprocess.STDOUT
        )
        deadline = time.monotonic() + args.timeout
        code: int
        while proc.poll() is None:
            if time.monotonic() > deadline:
                fh.write(f"\n[pass driver] timeout after {args.timeout}s\n")
                fh.flush()
                proc.terminate()
                break
            if args.stop_after_classify and _log_has(
                "classify_worker: Classification complete"
            ):
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
        fh.write(f"\nEXIT={code}\n")
    print(f"wrote {log_path} EXIT={code}")
    return 0


def _readback(args: argparse.Namespace) -> int:
    from imas_codex.graph.client import GraphClient

    labels = json.loads(Path(args.labels).read_text())["path_answers"]
    label_paths = sorted(labels)

    out: dict[str, object] = {"labelled_paths_total": len(label_paths)}

    with GraphClient() as gc:
        out["domain_source_distribution"] = {
            str(r["src"]): r["c"]
            for r in gc.query(
                "MATCH (n:IMASNode) RETURN n.domain_source AS src, "
                "count(*) AS c ORDER BY c DESC"
            )
        }

        out["llm_tier_matrix"] = [
            {
                "domain_source": r["src"],
                "physics_domain": r["dom"],
                "count": r["c"],
            }
            for r in gc.query(
                "MATCH (n:IMASNode) WHERE n.domain_source IN $srcs "
                "RETURN n.domain_source AS src, n.physics_domain AS dom, "
                "count(*) AS c ORDER BY src, c DESC",
                srcs=list(LLM_TIER_SOURCES),
            )
        ]

        out["summary_at_data_management"] = gc.query(
            "MATCH (n:IMASNode {ids: 'summary'}) "
            "WHERE n.physics_domain = 'data_management' "
            "RETURN count(*) AS c"
        )[0]["c"]
        out["summary_total"] = gc.query(
            "MATCH (n:IMASNode {ids: 'summary'}) RETURN count(*) AS c"
        )[0]["c"]

        graph_domains = {
            r["id"]: r["dom"]
            for r in gc.query(
                "MATCH (n:IMASNode) WHERE n.id IN $paths "
                "RETURN n.id AS id, n.physics_domain AS dom",
                paths=label_paths,
            )
        }

    matches = 0
    mismatches = []
    for path in label_paths:
        lab = labels[path]
        dom = graph_domains.get(path)
        if dom == lab:
            matches += 1
        else:
            mismatches.append({"path": path, "label": lab, "graph": dom})
    out["labelled_paths_scored"] = len(graph_domains)
    out["labelled_matches"] = matches
    out["labelled_accuracy"] = (
        round(matches / len(graph_domains), 4) if graph_domains else None
    )
    out["labelled_mismatches"] = mismatches

    Path(args.out).write_text(json.dumps(out, indent=2) + "\n")
    src = out["domain_source_distribution"]
    print(
        f"wrote {args.out}: {matches}/{len(graph_domains)} labelled match; "
        f"summary/data_management={out['summary_at_data_management']}; "
        f"jev_tree={src.get('jev_tree')} llm_classified={src.get('llm_classified')}"
    )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("pass")
    p.add_argument("--ids", nargs="*", default=None)
    p.add_argument("--reset", action="store_true")
    p.add_argument("--dry", action="store_true")
    p.add_argument("--timeout", type=int, default=540)
    p.add_argument(
        "--stop-after-classify",
        action="store_true",
        help="terminate the pass once classification has completed and written, "
        "skipping the graph-wide re-embed and cluster phases a reset triggers",
    )
    p.add_argument("--log", required=True)
    p.set_defaults(func=_run_pass)

    r = sub.add_parser("readback")
    r.add_argument("--labels", required=True)
    r.add_argument("--out", required=True)
    r.set_defaults(func=_readback)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())