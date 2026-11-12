#!/usr/bin/env python
"""Read the 143-source rerun cohort back, rerun the unproduced remainder, read again.

Reuses the cohort read-back and capped-rerun implementation written for the
second-pass cohort (``sib-cohort-rerun-isn-0-11-run.py``) by importing it and
repointing its cohort file and artifact directory at this node's own.  Only the
cohort path, the artifact home and the disposable-graph negative control are
new here.  The cohort itself is committed at
``docs/evidence/fragments/sn-full-dd-continuation/cohort-143.json`` so the
read-back is reproducible from the repository alone.

Phases leave one artifact each in the run directory:

    census      live state of all 143 cohort sources, names, export gate
    pin         immutable DD snapshots for the selected sources
    reset       sanctioned reset of terminal-failed sources at the claim cap
    manifest    emit the sn-sources manifest that scopes the rerun
    run         the scoped pipeline (--dry-run or live)
    readback    per-source outcome, names and scores, spend, export gate
    plant       the disposable-graph negative control
    fragment    the landed record
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import os
import shlex
import socket
import subprocess
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
PRIOR = ROOT / (
    "docs/evidence/fragments/sn-integrity-backlog/sib-cohort-rerun-isn-0-11-run.py"
)
COHORT_FILE = ROOT / "docs/evidence/fragments/sn-full-dd-continuation/cohort-143.json"
RUN = Path(
    "/home/ITER/mcintos/.config/reckon/crew/runs/"
    "r-20261112T014614393815-sfdc-cohort-floor"
)
CAP_FLOOR = 100
COST_LIMIT = 10.0
FRAGMENT = (
    ROOT / "docs/evidence/fragments/sn-full-dd-continuation/sfdc-cohort-floor.html"
)
PLANT_ABSENT = "dd:sfdc_negative_control/planted/attempted/not_produced"
PLANT_PRESENT = "dd:sfdc_negative_control/planted/produced"


def reuse():
    """Load the second-pass cohort implementation with this node's paths."""
    spec = importlib.util.spec_from_file_location("cohort_rerun_suite", PRIOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ROOT = ROOT
    module.RUN = RUN
    module.COHORT_FILE = COHORT_FILE
    module.COST_LIMIT = COST_LIMIT
    module.CAP_FLOOR = CAP_FLOOR
    return module


def save(name, payload):
    (RUN / f"{name}.json").write_text(
        json.dumps(payload, indent=2, default=str, sort_keys=True) + "\n"
    )


def load(name):
    return json.loads((RUN / f"{name}.json").read_text())


def _free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def pipeline(*, dry_run):
    """Run the scoped pipeline in this process's tree, bounded to the budget."""
    import imas_codex.standard_names.loop as loop

    print(f"# module: {loop.__file__}", flush=True)
    command = [
        "uv",
        "run",
        "--no-sync",
        "imas-codex",
        "sn",
        "run",
        "--focus",
        str(RUN / "cohort-sources.json"),
        "--skip-global-maintenance",
        "-c",
        str(COST_LIMIT),
    ]
    if dry_run:
        command.append("--dry-run")
    else:
        command.extend(["-t", "22"])
    print(f"# pipeline command: {shlex.join(command)}", flush=True)
    print(f"# SLURM_JOB_ID: {os.environ.get('SLURM_JOB_ID')}", flush=True)
    completed = subprocess.run(
        command,
        cwd=ROOT,
        stdout=sys.stdout,
        stderr=subprocess.STDOUT,
        env=dict(
            os.environ,
            UV_PROJECT_ENVIRONMENT="/home/ITER/mcintos/Code/imas-codex/.venv",
            PYTHONPATH=str(ROOT),
        ),
        timeout=660,
    )
    return completed.returncode


def plant():
    """Plant two sources in a disposable graph and read them back.

    One is attempted (counted, failed) with no produced name, the other is
    produced.  The cohort read-back must report the first as not produced and
    the second as produced, which proves the read-back can see a produced
    binding at all rather than reporting an empty result from an instrument
    that never worked.
    """
    import tempfile

    from imas_codex.graph.temp_neo4j import start_temp_neo4j, stop_temp_neo4j

    scratch = Path(
        tempfile.mkdtemp(prefix="sfdc-plant-", dir=os.environ.get("TMPDIR", "/tmp"))
    )
    bolt, http = _free_port(), _free_port()
    proc, log_path = start_temp_neo4j(scratch, bolt, http, load_dump=False)
    uri = f"bolt://localhost:{bolt}"
    saved = os.environ.get("NEO4J_URI")
    os.environ["NEO4J_URI"] = uri
    result = {"graph_uri": uri, "log": str(log_path)}
    try:
        from imas_codex.graph.client import GraphClient

        with GraphClient() as gc:
            gc.query(
                "CREATE (:StandardNameSource {id: $id, status: 'failed', "
                "attempt_count: 1, last_error: 'planted: attempted, nothing produced'})",
                id=PLANT_ABSENT,
            )
            gc.query(
                "CREATE (:StandardNameSource {id: $id, status: 'composed', "
                "attempt_count: 1})",
                id=PLANT_PRESENT,
            )
            gc.query(
                "MATCH (s:StandardNameSource {id: $sid}) "
                "CREATE (s)-[:PRODUCED_NAME]->(:StandardName {id: $nid, "
                "name_stage: 'draft', status: 'draft'})",
                sid=PLANT_PRESENT,
                nid="planted_produced_control",
            )
        # The read-back the cohort uses, pointed at the disposable endpoint.
        state = reuse().previous().source_state([PLANT_ABSENT, PLANT_PRESENT])
        produced = [i for i, r in state.items() if r["bindings"]]
        result.update(
            state={
                k: {
                    "status": v["status"],
                    "attempt_count": v["attempt_count"],
                    "bindings": v["bindings"],
                }
                for k, v in state.items()
            },
            produced=produced,
            absent_read_as_produced=PLANT_ABSENT in produced,
            present_read_as_produced=PLANT_PRESENT in produced,
            ok=PLANT_ABSENT not in produced and PLANT_PRESENT in produced,
        )
    finally:
        if saved is None:
            os.environ.pop("NEO4J_URI", None)
        else:
            os.environ["NEO4J_URI"] = saved
        stop_temp_neo4j(proc)
    save("plant", result)
    print(json.dumps(result, indent=2))
    return 0 if result.get("ok") else 1


def approval():
    """Read every run under the full-DD approval and the remaining balance."""
    from imas_codex.graph.client import GraphClient

    approval_cap = 350.0
    with GraphClient() as gc:
        runs = gc.query(
            """MATCH (r:SNRun)
            RETURN r.id AS id, r.cost_limit AS cost_limit,
                   r.cost_spent AS cost_spent, r.status AS status,
                   toString(r.started_at) AS started_at,
                   r.invocation AS invocation
            ORDER BY started_at"""
        )
    runs = [r for r in runs if (r["started_at"] or "")[:10] >= "2026-11-07"]
    total = sum(float(r["cost_spent"] or 0) for r in runs)
    payload = {
        "approval_cap": approval_cap,
        "spent_total": total,
        "remaining": approval_cap - total,
        "runs": runs,
    }
    save("approval", payload)
    print(
        json.dumps(
            {
                "spent_total": total,
                "remaining": approval_cap - total,
                "runs": len(runs),
            },
            indent=2,
        )
    )
    return 0


def _reason(outcome, pins, receipts):
    """The recorded reason, naming the missing token when it is a vocab gap."""
    import re

    source_id = outcome["source_id"]
    if source_id in pins:
        return f"refused before the run: {pins[source_id]}"
    text = outcome.get("after_error") or outcome.get("before_error")
    if outcome.get("after_status") == "vocab_gap":
        pass
    if outcome.get("after_status") == "vocab_gap":
        match = re.search(r"vocabulary token '([^']+)'", text or "") or re.search(
            r"vocabulary gap: '([^']+)'", text or ""
        )
        return (
            f"vocab_gap, missing token '{match.group(1)}'"
            if match
            else f"vocab_gap: {text or 'no token recorded'}"
        )
    if text:
        return str(text)
    receipt = receipts.get(source_id)
    cited = f"; reset receipt {receipt}" if receipt else "; no reset receipt"
    return (
        f"{outcome.get('after_status')} with no produced name "
        f"({outcome.get('after_attempt_count') or 0} attempts{cited})"
    )


def fragment():
    """Write the landed record from the read-back artifacts."""
    from html import escape

    payload = load("readback")
    summary = payload["summary"]
    census_payload = load("census")
    plant_payload = load("plant")
    approval_payload = load("approval")
    pin_payload = load("pin")
    pins = {r["source_id"]: r["reason"] for r in pin_payload["refused"]}
    receipts = {
        source_id: (rows["events"][0]["id"] if rows.get("events") else None)
        for source_id, rows in load("verify-reset")["receipts"].items()
    }
    outcomes = payload["outcomes"]
    names = {}
    for row in payload["names"]:
        names.setdefault(row["source_id"], []).append(row)
    produced_rows, unproduced_rows = [], []
    for source_id in sorted(outcomes):
        o = outcomes[source_id]
        o["source_id"] = source_id
        if o["produced"]:
            entries = names.get(source_id) or []
            detail = "<br>".join(
                f"<code>{escape(e['name'] or '')}</code> ({escape(e['stage'] or '')}"
                + (f", score {escape(str(e['score']))}" if e.get("score") else "")
                + ")"
                for e in entries
            )
            produced_rows.append(
                "<tr>"
                f"<td><code>{escape(source_id)}</code></td>"
                f"<td>{escape(str(o['before_status']))}</td>"
                f"<td>{escape(str(o['after_status']))}</td>"
                f"<td>{detail}</td></tr>"
            )
        else:
            reason = _reason(o, pins, receipts)
            unproduced_rows.append(
                "<tr>"
                f"<td><code>{escape(source_id)}</code></td>"
                f"<td>{escape(str(o['before_status']))}</td>"
                f"<td>{escape(str(o['after_status']))}</td>"
                f"<td>{escape(str(reason))}</td></tr>"
            )
    spend = summary["spend"]
    short = CAP_FLOOR - summary["produced"]
    floor_note = "met" if short <= 0 else f"short by {short}"
    cost_rows = "".join(
        f"<tr><td>{escape(str(c['phase']))}</td><td>{c['calls']}</td>"
        f"<td>{float(c['cost'] or 0):.6f}</td></tr>"
        for c in payload["costs"]
    )
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>The rerun cohort reaches its floor</title>
</head>
<body>
<section id="sfdc-cohort-floor-record">
<h2 id="sfdc-cohort-floor-heading">The 143-source cohort read back after the scoped rerun</h2>

<p id="sfdc-cohort-floor-cohort">The cohort is the 143 sources recorded by
<code>sib-second-pass</code>, committed as a manifest at
<code>docs/evidence/fragments/sn-full-dd-continuation/cohort-143.json</code>.
A live read before this run found {summary["produced_before"]} of them produced
and {len(census_payload["selected"])} without a produced name. Status counts
across the 143 were
{escape(json.dumps(census_payload["status_counts"], sort_keys=True))}.</p>

<h3 id="sfdc-cohort-floor-rerun">The scoped rerun</h3>

<p id="sfdc-cohort-floor-rerun-scope">The rerun is one
<code>sn run --focus cohort-sources.json -c {COST_LIMIT:.0f}</code> over
{len(load("pin")["eligible"])} sources, run in place on the <code>all_debug</code>
compute partition (SLURM job <code>{escape(str(os.environ.get("SFDC_PIPELINE_JOB", "unknown")))}</code>).
It recorded run <code>{escape(str(summary["run_id"]))}</code> with status
<code>{escape(str(summary["run_status"]))}</code>, spending
<strong>USD {spend:.6f}</strong> of its USD {COST_LIMIT:.0f} cap. Against the
USD {approval_payload["approval_cap"]:.0f} approval, the USD
{approval_payload["spent_total"]:.6f} spent by every run since 2026-11-07
leaves <strong>USD {approval_payload["remaining"]:.6f}</strong> remaining. The
pipeline was stopped by this node's 11-minute process bound rather than reaching
its own time limit, so the run record still reads <code>started</code> with no
end time; its cost is the sum of that run's own per-call receipts.</p>

<table id="sfdc-cohort-floor-summary">
<caption>Read-back of the cohort after the rerun</caption>
<thead><tr><th>Reading</th><th>Value</th></tr></thead>
<tbody>
<tr><td>Produced before the run</td><td>{summary["produced_before"]} of 143</td></tr>
<tr><td>Produced after the run</td><td><strong>{summary["produced"]} of 143</strong> against the 100-of-143 floor ({floor_note})</td></tr>
<tr><td>Selected sources produced in this run</td><td>{summary["selected_produced"]} of {summary["selected"]}</td></tr>
<tr><td>Export gate blocking set</td><td>{summary["export_blocking"]}</td></tr>
<tr><td>Sourceless live drafts, before then after</td><td>{summary["drafts_before"]} then {summary["drafts_after"]}</td></tr>
<tr><td>Still unproduced</td><td>{len(summary["still_unproduced"])}</td></tr>
</tbody>
</table>

<h3 id="sfdc-cohort-floor-negative">The disposable-graph negative control</h3>

<p id="sfdc-cohort-floor-negative-text">In a disposable graph two sources were
planted and read back by the same function the cohort read uses. The attempted
one (status <code>failed</code>, one attempt, no produced name) read back as
<strong>not produced</strong> ({escape(str(plant_payload["absent_read_as_produced"]))}),
and the produced control read back as <strong>produced</strong>
({escape(str(plant_payload["present_read_as_produced"]))}). The second is the
positive control: it shows the read-back sees a produced binding, so the
absent result is the instrument answering rather than an empty graph.</p>

<h3 id="sfdc-cohort-floor-produced">Sources produced</h3>

<table id="sfdc-cohort-floor-produced-table">
<caption>Every cohort source holding a produced name after the run</caption>
<thead><tr><th>Source</th><th>Status before</th><th>Status after</th><th>Produced name, stage and review score</th></tr></thead>
<tbody>
{chr(10).join(produced_rows)}
</tbody>
</table>

<h3 id="sfdc-cohort-floor-unproduced">Sources still unproduced, with a reason</h3>

<p id="sfdc-cohort-floor-receipts">{len(load("verify-reset")["verified"])} sources were
reset through <code>retry_failed_sources</code> with a written reason and a
<code>StandardNameSourceRetry</code> receipt each, all verified with the counters
cleared and no claim; {len(load("verify-reset")["bad"])} were refused by the
operation. The receipts are in the run directory's
<code>verify-reset.json</code>, and {len(pins)} sources were refused before the
run because they carry no provable immutable snapshot.</p>

<table id="sfdc-cohort-floor-unproduced-table">
<caption>Every cohort source with no produced name after the run</caption>
<thead><tr><th>Source</th><th>Status before</th><th>Status after</th><th>Reason</th></tr></thead>
<tbody>
{chr(10).join(unproduced_rows)}
</tbody>
</table>

<h3 id="sfdc-cohort-floor-cost">Spend by phase</h3>

<table id="sfdc-cohort-floor-cost-table">
<caption>Calls and cost recorded against run {escape(str(summary["run_id"]))}</caption>
<thead><tr><th>Phase</th><th>Calls</th><th>Cost [USD]</th></tr></thead>
<tbody>
{cost_rows}
<tr><td><strong>Total</strong></td><td>{sum(c["calls"] for c in payload["costs"])}</td><td><strong>{spend:.6f}</strong></td></tr>
</tbody>
</table>
</section>
</body>
</html>
"""
    FRAGMENT.write_text(html)
    print(json.dumps({"fragment": str(FRAGMENT), "bytes": len(html)}))
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = {
        "census": lambda: reuse().census(),
        "pin": lambda: reuse().pin(),
        "reset": lambda: reuse().reset(),
        "manifest": lambda: reuse().manifest(),
        "run-dry": lambda: pipeline(dry_run=True),
        "run": lambda: pipeline(dry_run=False),
        "readback": lambda: reuse().readback(),
        "plant": plant,
        "approval": approval,
        "fragment": fragment,
    }
    parser.add_argument("action", choices=list(actions))
    parser.add_argument("--log", default=None)
    args = parser.parse_args()
    revision = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
    ).strip()
    log_path = Path(args.log) if args.log else RUN / f"{args.action}.log"
    status = 1
    with (
        log_path.open("x") as log,
        contextlib.redirect_stdout(log),
        contextlib.redirect_stderr(log),
    ):
        print(
            f"# revision: {revision}; tree: {ROOT}; "
            f"command: {shlex.join(['uv', 'run', '--no-sync', 'python', *sys.argv])}",
            flush=True,
        )
        print(
            f"# cwd: {Path.cwd().resolve()}; PYTHONPATH: {os.environ.get('PYTHONPATH')}",
            flush=True,
        )
        try:
            status = actions[args.action]()
        except Exception:
            traceback.print_exc()
        finally:
            print(f"EXIT={status}", flush=True)
    print(f"{args.action}: EXIT={status}; log: {log_path}")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
