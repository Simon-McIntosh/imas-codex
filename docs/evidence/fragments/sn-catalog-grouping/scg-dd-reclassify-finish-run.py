#!/usr/bin/env python3
"""Drive the DD path re-classification and read its result back from the graph.

Three subcommands:

``pass``
    Run one bounded re-classification pass over a set of IDS groups by invoking
    the DD build's classify phase as a subprocess. The CLI must not be piped or
    redirected from a shell, and a single tool call cannot outlast ten minutes,
    so the pass is invoked through this driver and each pass log carries the
    literal command on its first line and ``EXIT=<code>`` on its last. The
    classify phase writes its result once, after every batch completes, so a
    pass that is cut off before its write leaves the graph unchanged -- only
    its spend is lost. The invocation itself lives once, in the sibling
    ``scg-dd-reclassify-run.py``, which this driver imports.

``readback``
    Read the classified result back from the graph under one bounded
    connection: the ``domain_source`` distribution, the ``domain_source`` x
    ``physics_domain`` matrix for the LLM-tier paths, how many ``summary`` paths
    sit at ``data_management``, and the rule-labelled paths scored against the
    graph. The primary score is against the revised rule labels; the score
    against the earlier label set is reported beside it, naming that set. The
    score's denominator is the number of labelled paths, so a labelled path
    absent from the graph counts as a miss rather than dropping out.

``spend``
    Sum the classifier spend from the pass logs. A pass log carries two kinds
    of cost line: a per-call line (``cost=$X, model=...``) and a running total
    (``batch N/M done (cost=$Y)``, and the ``spend=$`` field of an
    ``LLM heartbeat`` line). Only the per-call line is summed, and each log's
    last heartbeat is reported as an independent cross-check of that log's
    total -- summing the running totals too would count the same spend twice.

Usage:
    python scg-dd-reclassify-finish-run.py pass --ids a b c [--reset] [--dry] \
        [--timeout 540] [--stop-after-classify] --log <path>
    python scg-dd-reclassify-finish-run.py readback --labels <json> \
        [--labels-prior <json>] [--instrument labels|bare-string] --out <json>
    python scg-dd-reclassify-finish-run.py spend --passes <dir> --out <json>
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path


# domain_source values the two-tier classifier writes for a path it judged
# itself, as opposed to one that inherited a parent's domain or was left null
# for infrastructure metadata.
LLM_TIER_SOURCES = ("jev_tree", "llm_classified")

# a per-call line carries both a cost and the model that served it; a running
# total carries a cost and a batch counter, and a heartbeat carries the log's
# cumulative spend. Only the first is a call.
PER_CALL = re.compile(r"cost=\$([0-9.]+),\s*model=")
HEARTBEAT_SPEND = re.compile(r"LLM heartbeat:.*spend=\$([0-9.]+)")


def _sibling_pass_module():
    """Load the sibling pass driver that sits beside this file.

    The pass invocation is implemented once, in ``scg-dd-reclassify-run.py``;
    this driver imports it rather than carrying a second copy of the same
    subprocess handling. The fragment directory carries no package identifier,
    so the module is loaded by path.
    """
    path = Path(__file__).with_name("scg-dd-reclassify-run.py")
    spec = importlib.util.spec_from_file_location("scg_dd_reclassify_run", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"could not load the sibling pass driver at {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_pass(args: argparse.Namespace) -> int:
    return _sibling_pass_module().run_pass(
        ids=args.ids,
        reset=args.reset,
        dry=args.dry,
        timeout=args.timeout,
        stop_after_classify=args.stop_after_classify,
        log=args.log,
    )


def _read_label(entry: object, instrument: str) -> object:
    """Read one label entry the way the named instrument does.

    ``labels`` is the corrected instrument: a label file maps each path to a
    same-named field of its entry. ``bare-string`` is the pre-repair
    instrument, which read the entry itself as the domain string and so scored
    zero against any label file whose entries are records.
    """
    if instrument == "bare-string":
        return entry
    if isinstance(entry, dict):
        return entry.get("label")
    return entry


def _score(
    labels: dict,
    graph: dict,
    source: str,
    out: dict,
    key: str,
    *,
    instrument: str = "labels",
) -> None:
    """Score the graph's domains against one label set under ``key``.

    The denominator is the number of labelled paths, never the number of rows
    the graph returned: the labelled set is the fixed quantity here, and a
    labelled path the graph does not hold is a miss, not a path that drops
    silently out of the denominator.
    """
    matches = 0
    absent = 0
    mismatches = []
    for path in sorted(labels):
        lab = _read_label(labels[path], instrument)
        row = graph.get(path)
        dom = row["dom"] if row else None
        if row is None:
            absent += 1
        if dom is not None and dom == lab:
            matches += 1
        else:
            mismatches.append(
                {
                    "path": path,
                    "label": lab if isinstance(lab, str) else repr(lab),
                    "graph": dom,
                    "domain_source": row["src"] if row else None,
                    "label_set": source,
                    "absent_from_graph": row is None,
                }
            )
    out[f"{key}_instrument"] = instrument
    out[f"{key}_label_set"] = source
    out[f"{key}_paths_scored"] = len(labels)
    out[f"{key}_graph_rows"] = len(graph)
    out[f"{key}_paths_absent_from_graph"] = absent
    out[f"{key}_matches"] = matches
    out[f"{key}_accuracy"] = round(matches / len(labels), 4) if labels else None
    out[f"{key}_mismatch_count"] = len(mismatches)
    out[f"{key}_mismatches"] = mismatches


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
            "MATCH (n:IMASNode) WHERE n.ids = 'summary' "
            "AND n.physics_domain = 'data_management' RETURN count(*) AS c"
        )[0]["c"]
        out["summary_total"] = gc.query(
            "MATCH (n:IMASNode) WHERE n.ids = 'summary' RETURN count(*) AS c"
        )[0]["c"]

        graph = {
            r["id"]: {"dom": r["dom"], "src": r["src"]}
            for r in gc.query(
                "MATCH (n:IMASNode) WHERE n.id IN $paths "
                "RETURN n.id AS id, n.physics_domain AS dom, "
                "n.domain_source AS src",
                paths=label_paths,
            )
        }

    # each label entry is a per-path record whose adjudicated rule label is its
    # "label" field; the revised label set is the primary score, and the label
    # set this node was originally scored against is kept beside it.
    _score(labels, graph, Path(args.labels).name, out, "revised", instrument=args.instrument)
    if args.labels_prior:
        prior = json.loads(Path(args.labels_prior).read_text())["path_answers"]
        _score(prior, graph, Path(args.labels_prior).name, out, "prerevision")

    Path(args.out).write_text(json.dumps(out, indent=2) + "\n")
    src = out["domain_source_distribution"]
    print(
        f"wrote {args.out}: revised {out['revised_matches']}/"
        f"{out['revised_paths_scored']} "
        f"({out['revised_accuracy']}), absent "
        f"{out['revised_paths_absent_from_graph']}, "
        f"instrument={out['revised_instrument']}; prerevision "
        f"{out.get('prerevision_matches')}/"
        f"{out.get('prerevision_paths_scored')} "
        f"({out.get('prerevision_accuracy')}); "
        f"summary/data_management={out['summary_at_data_management']}; "
        f"jev_tree={src.get('jev_tree')} llm={src.get('llm_classified')}"
    )
    return 0


def _spend(args: argparse.Namespace) -> int:
    """Sum per-call spend from the pass logs, cross-checked against heartbeats.

    A cost line that carries a model is one billed call; a line that carries a
    batch counter, or the ``spend=`` field of a heartbeat, is a running total of
    those call lines. Summing the running totals as well would count the same
    spend twice, so only the per-call lines are summed.
    """
    passes_dir = Path(args.passes)
    per_log = []
    for log in sorted(passes_dir.glob("pass-*.log")):
        text = log.read_text(errors="ignore")
        calls = [float(x) for x in PER_CALL.findall(text)]
        beats = [float(x) for x in HEARTBEAT_SPEND.findall(text)]
        per_log.append(
            {
                "log": log.name,
                "calls": len(calls),
                "call_spend_usd": round(sum(calls), 4),
                "last_heartbeat_spend_usd": beats[-1] if beats else None,
            }
        )

    total = round(sum(r["call_spend_usd"] for r in per_log), 4)
    calls = sum(r["calls"] for r in per_log)
    heart = round(
        sum(
            r["last_heartbeat_spend_usd"]
            for r in per_log
            if r["last_heartbeat_spend_usd"]
        ),
        4,
    )
    out = {
        "passes": per_log,
        "billed_calls": calls,
        "per_call_spend_usd": total,
        "heartbeat_sum_usd": heart,
        "heartbeat_delta_usd": round(total - heart, 4),
        "heartbeat_role": "lower bound: each log's last heartbeat is written "
        "when the pass is stopped after its classify phase, so it can lag the "
        "final calls and never exceeds the per-call sum",
        "basis": "sum of per-call lines (cost=$X, model=...); running "
        "totals (batch N/M done, and the spend=$ field of LLM heartbeat) "
        "excluded because they restate the per-call lines",
    }
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n")
    print(
        f"wrote {args.out}: {calls} calls, per-call ${total}, "
        f"heartbeat sum ${heart}, delta ${out['heartbeat_delta_usd']}"
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
    r.add_argument("--labels-prior", default=None)
    r.add_argument(
        "--instrument",
        choices=("labels", "bare-string"),
        default="labels",
        help="how a label entry is read: 'labels' takes its 'label' field, "
        "'bare-string' reads the whole entry as the domain (the pre-repair "
        "instrument, kept so its zero can be re-shown as a control)",
    )
    r.add_argument("--out", required=True)
    r.set_defaults(func=_readback)

    s = sub.add_parser("spend")
    s.add_argument("--passes", required=True)
    s.add_argument("--out", required=True)
    s.set_defaults(func=_spend)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())