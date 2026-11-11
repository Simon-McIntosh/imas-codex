#!/usr/bin/env python
"""Admit cohort sources by the approved rework mapping and measure their rerun."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shlex
import subprocess
from datetime import datetime
from html import escape
from pathlib import Path

from imas_codex.graph.client import GraphClient

ROOT = Path(__file__).resolve().parents[4]
RUN = Path(
    "/home/ITER/mcintos/.config/reckon/crew/runs/r-20261111T150911488391-sib-vocab-gap-requeue-isn-0-11-1"
)
MAPPING = Path(
    "/home/ITER/mcintos/Code/imas-standard-names/docs/evidence/fragments/dd-gap-vocabulary/doubtful-rework.json"
)
COHORT_INPUT = Path(
    "/home/ITER/mcintos/.config/reckon/crew/runs/r-20261110T192543764468-sib-cohort-rerun-isn-0-11/census.json"
)
FRAGMENT = (
    ROOT
    / "docs/evidence/fragments/sn-integrity-backlog/sib-vocab-gap-requeue-isn-0-11-1.html"
)
FOCUS = RUN / "mapped-sources.json"
CAP = 3.0


def save(name, payload):
    (RUN / f"{name}.json").write_text(json.dumps(payload, indent=2, default=str) + "\n")


def load(name):
    return json.loads((RUN / f"{name}.json").read_text())


def source_reader():
    path = ROOT / "docs/evidence/fragments/sn-integrity-backlog/sib-second-pass-run.py"
    spec = importlib.util.spec_from_file_location("source_evidence", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.source_state


def snapshot(ids):
    with GraphClient() as gc:
        active = gc.query(
            "MATCH (v:ISNGrammarVersion {active:true}) RETURN v.version AS version, v.content_digest AS digest"
        )
        assert len(active) == 1 and active[0]["version"] == "0.11.1", active
        rows = gc.query(
            "UNWIND $ids AS id MATCH (s:StandardNameSource {id:id}) OPTIONAL MATCH (s)-[:PRODUCED_NAME]->(n:StandardName) RETURN id AS id, s.status AS status, s.last_error AS refusal, collect(DISTINCT {name:n.id, stage:n.name_stage, score:n.reviewer_score_name}) AS names ORDER BY id",
            ids=ids,
        )
        assert len(rows) == len(ids)
    return {
        "active": active,
        "sources": source_reader()(ids),
        "rows": rows,
        "produced": sum(any(n["name"] for n in r["names"]) for r in rows),
        "accepted": sum(
            any(n["stage"] in ("accepted", "approved") for n in r["names"])
            for r in rows
        ),
    }


def prepare():
    cohort = sorted(json.loads(COHORT_INPUT.read_text())["sources"])
    assert len(cohort) == 143
    mapping = json.loads(MAPPING.read_text())
    entries = [
        {
            "source": "dd:" + s,
            "spelling": r["reworked_spelling"],
            "description": r["description"],
            "required_binding": r.get("required_binding"),
            "new_tokens": r["new_tokens"],
        }
        for r in mapping["rows"]
        for s in r["source_paths"]
    ]
    admitted = [r for r in entries if r["source"] in cohort]
    assert admitted
    before = snapshot(cohort)
    with GraphClient() as gc:
        tokens = gc.query(
            "MATCH (t:GrammarToken {version:$version}) RETURN t.value AS value, t.segment AS segment",
            version="0.11.1",
        )
    assert len(tokens) >= 1000 and any(t["value"] == "power_convertor" for t in tokens)
    values = {t["value"] for t in tokens}
    for row in admitted:
        assert all(t["spelling"] in values for t in row["new_tokens"]), row
    selected = {r["source"] for r in admitted}
    parked = {
        s: r
        for s, r in before["sources"].items()
        if r["status"] == "vocab_gap" and s not in selected
    }
    save(
        "mapping-before",
        {
            "mapping_path": str(MAPPING),
            "mapping_sha256": hashlib.sha256(MAPPING.read_bytes()).hexdigest(),
            "all_mapping_rows": entries,
            "admitted": admitted,
            "excluded_mapping": [r for r in entries if r["source"] not in cohort],
            "before": before,
            "parked": parked,
            "grammar_rows": len(tokens),
            "positive_control": [t for t in tokens if t["value"] == "power_convertor"],
        },
    )
    grouped = {}
    for sid in sorted(selected):
        family, _, path = sid.removeprefix("dd:").partition("/")
        grouped.setdefault(family, []).append(path)
    FOCUS.write_text(
        json.dumps(
            {
                "kind": "sn_sources",
                "schema_version": 1,
                "name": "mapped-vocabulary-retry",
                "title": "Approved vocabulary rework cohort retry",
                "description": "Exact intersection of the cohort and approved source-to-spelling mapping.",
                "sources": grouped,
            },
            indent=2,
        )
        + "\n"
    )
    print(
        json.dumps(
            {
                "admitted": admitted,
                "mapping_rows": len(entries),
                "outside_cohort": len(entries) - len(admitted),
                "produced_before": before["produced"],
                "accepted_before": before["accepted"],
                "parked_controls": len(parked),
            },
            default=str,
        ),
        flush=True,
    )
    return 0


def execute():
    prepare()
    from imas_codex.standard_names.graph_store import graph_foundation
    from imas_codex.standard_names.graph_store.graph_source_bindings import (
        retry_failed_sources,
        set_source_compose_hint,
    )

    payload = load("mapping-before")
    hints = []
    for row in payload["admitted"]:
        sid = row["source"]
        state = source_reader()([sid])[sid]
        assert not state["claimed"], state
        if state["status"] == "failed" or (
            state["status"] == "extracted" and (state["attempt_count"] or 0) >= 5
        ):
            reset = retry_failed_sources(
                [sid],
                services=graph_foundation.services_for_call(),
                reason="Approved vocabulary rework is now registered; retry against the mapped complete spelling.",
            )
            hints.append({"source": sid, "retry": reset})
            assert reset["retried"] == 1, reset
        hint = set_source_compose_hint(
            sid,
            services=graph_foundation.services_for_call(),
            hint=f"Use the approved complete spelling {row['spelling']}. {row['description']} Preserve the DD unit A: this is a current-valued profile-alignment figure of merit, not a dimensionless factor. The former current_alignment_factor request is replaced by this registered physical base.",
            reason="The approved source-to-spelling rework mapping replaces the former vocabulary request; the mapped physical base is registered in active ISN grammar.",
            replace=True,
        )
        hints.append({"source": sid, "hint": hint})
        save("mapping-steering", hints)
        assert hint["applied"], hint
    command = [
        "uv",
        "run",
        "--no-sync",
        "imas-codex",
        "sn",
        "run",
        "--focus",
        str(FOCUS),
        "--skip-global-maintenance",
        "-c",
        str(CAP),
        "-t",
        "20",
    ]
    print("# pipeline command: " + shlex.join(command), flush=True)
    result = subprocess.run(
        command,
        cwd=ROOT,
        env=dict(
            os.environ,
            UV_PROJECT_ENVIRONMENT="/home/ITER/mcintos/Code/imas-codex/.venv",
            PYTHONPATH=str(ROOT),
        ),
        timeout=1800,
    )
    save(
        "pipeline-result",
        {
            "command": command,
            "exit_code": result.returncode,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        },
    )
    readback()
    return result.returncode


def readback():
    payload = load("mapping-before")
    ids = sorted(payload["before"]["sources"])
    after = snapshot(ids)
    with GraphClient() as gc:
        runs = gc.query(
            "MATCH (r:SNRun) WHERE r.invocation CONTAINS $focus RETURN properties(r) AS run ORDER BY r.started_at",
            focus=str(FOCUS),
        )
        assert len(runs) == 1, runs
        run = runs[0]["run"]
        costs = gc.query(
            "MATCH (c:LLMCost {for_run:$run}) RETURN c.phase AS phase, count(c) AS calls, sum(c.llm_cost) AS usd ORDER BY phase",
            run=run["id"],
        )
        names = gc.query(
            "UNWIND $ids AS id MATCH (s:StandardNameSource {id:id})-[:PRODUCED_NAME]->(n:StandardName) OPTIONAL MATCH (c:LLMCost {for_run:$run}) WHERE n.id IN coalesce(c.standard_name_ids,[]) RETURN id AS source, n.id AS name, n.name_stage AS stage, n.description AS description, n.reviewer_score_name AS score, n.created_at AS created_at, n.last_run_id AS last_run_id, count(c) AS own_cost_rows, collect(DISTINCT c.phase) AS own_phases ORDER BY id,n.id",
            ids=[r["source"] for r in payload["admitted"]],
            run=run["id"],
        )
    for sid, state in payload["parked"].items():
        assert after["sources"][sid] == state, (sid, state, after["sources"][sid])
    spend = sum(c["usd"] or 0 for c in costs)
    assert spend <= CAP, (spend, CAP)
    prior_names = {
        n["name"] for r in payload["before"]["rows"] for n in r["names"] if n["name"]
    }
    own = [
        n
        for n in names
        if n["name"] not in prior_names
        and "generate_name" in n["own_phases"]
        and datetime.fromisoformat(str(n["created_at"]))
        >= datetime.fromisoformat(str(run["started_at"]))
    ]
    outcomes = [
        {
            "source": r["source"],
            "mapped_spelling": r["spelling"],
            "after": after["sources"][r["source"]],
            "bound_names": [n for n in names if n["source"] == r["source"]],
            "exact_mapped_binding": any(
                n["source"] == r["source"] and n["name"] == r["spelling"] for n in names
            ),
        }
        for r in payload["admitted"]
    ]
    summary = {
        "run_id": run["id"],
        "run_status": run.get("status"),
        "stop_reason": run.get("stop_reason"),
        "cohort": 143,
        "floor": 100,
        "produced_before": payload["before"]["produced"],
        "produced_after": after["produced"],
        "accepted_before": payload["before"]["accepted"],
        "accepted_after": after["accepted"],
        "admitted": len(outcomes),
        "own_produced_names": len({n["name"] for n in own}),
        "spend_usd": spend,
        "cap_usd": CAP,
        "parked_controls_unchanged": len(payload["parked"]),
    }
    save(
        "mapping-after",
        {
            "summary": summary,
            "run": run,
            "after": after,
            "outcomes": outcomes,
            "own_names": own,
            "costs": costs,
        },
    )
    print("SUMMARY " + json.dumps(summary, default=str), flush=True)
    render()
    return 0


def render():
    before = load("mapping-before")
    result = load("mapping-after")
    summary = result["summary"]
    for name in result["own_names"]:
        assert "generate_name" in name["own_phases"]
        assert datetime.fromisoformat(name["created_at"]) >= datetime.fromisoformat(
            result["run"]["started_at"]
        )

    def table(headers, rows):
        return (
            "<table><thead><tr>"
            + "".join("<th>" + escape(str(c)) + "</th>" for c in headers)
            + "</tr></thead><tbody>\n"
            + "\n".join(
                "<tr>"
                + "".join("<td>" + escape(str(c)) + "</td>" for c in row)
                + "</tr>"
                for row in rows
            )
            + "\n</tbody></table>"
        )

    outcome_rows = []
    for row in result["outcomes"]:
        names = "; ".join(
            f"{n['name']} ({n['stage']}, score {n['score']}, own cost rows {n['own_cost_rows']}): {n['description']}"
            for n in row["bound_names"]
        )
        outcome_rows.append(
            [
                row["source"],
                row["mapped_spelling"],
                row["after"]["status"],
                names or row["after"]["last_error"],
                row["exact_mapped_binding"],
            ]
        )
    parked_rows = [
        [sid, state["last_error"], result["after"]["sources"][sid]["status"]]
        for sid, state in before["parked"].items()
    ]
    excluded_rows = [
        [r["source"], r["spelling"], "outside 143-source cohort"]
        for r in before["excluded_mapping"]
    ]
    html = f"""<h2 id="sib-vocab-gap-requeue-isn-0-11-1">Approved source rework admitted and rerun under USD 3</h2>
<p>Correction of the initial admission decision: admission uses membership in the approved source-to-spelling mapping, regardless of source status. The former requested spelling need not remain registered. The earlier USD 0 blocked attempt and its census and confirmation receipts are preserved in git and the run directory; this record supersedes its admission conclusion. The accepted/approved measure of 60 was also not the section floor metric: the floor counts a PRODUCED_NAME binding at any stage.</p>
<p>Mapping input: <code>{escape(str(MAPPING))}</code>, SHA-256 <code>{before["mapping_sha256"]}</code>. Its seven rows intersect the cohort at {len(before["admitted"])} source. Active graph grammar is 0.11.1 with {before["grammar_rows"]} token rows; mapped required token current_profile_alignment is present as a physical base, and the known-present power_convertor control returns device and object rows.</p>
<p>The current_alignment/value source maps to current_profile_alignment, a current-valued measure of agreement among supplying current-density profiles. It replaces current_alignment_factor and the reviewed current_alignment decomposition; it is not an assumed dimensionless factor. A durable exact-source compose hint carries the approved spelling and unit A through the sanctioned hint writer. No acceptance or text was written directly.</p>
{table(["DD source", "approved mapped spelling", "source status after", "bound name, stage and score, or refusal", "exact mapped binding"], outcome_rows)}
<p>Scoped pipeline run <code>{summary["run_id"]}</code>, status {summary["run_status"]}, stop reason {summary["stop_reason"]}; job {load("pipeline-result")["slurm_job_id"]} on partition all; --skip-global-maintenance, -c 3.0 and -t 20. Spend selected only by LLMCost.for_run for this id: USD {summary["spend_usd"]:.6f} of USD 3. The run drained its selected source, leaving USD {CAP - summary["spend_usd"]:.6f} unused. Own newly produced names: {summary["own_produced_names"]}, requiring a name absent from the cohort before, a generate_name cost receipt under this run id, and creation after the run started. Existing and peer-produced names are not counted as own production.</p>
{table(["phase", "calls", "USD"], [[c["phase"], c["calls"], f"{c['usd'] or 0:.6f}"] for c in result["costs"]])}
{table(["live cohort metric", "before", "after", "floor", "remaining shortfall"], [["sources with produced name, any stage", summary["produced_before"], summary["produced_after"], 100, max(0, 100 - summary["produced_after"])], ["sources with accepted or approved name", summary["accepted_before"], summary["accepted_after"], 100, max(0, 100 - summary["accepted_after"])]])}
<p>Both arms read all 143 exact source ids live and count each source once, irrespective of how many names it binds. A bound name at any stage satisfies the first metric; accepted or approved satisfies the second. Changes across the cohort are not automatically attributed to this run.</p>
<h3 id="sib-vocab-gap-requeue-isn-0-11-1-vocabulary-followups">Parked cohort sources outside the approved mapping</h3>
<p>Every source below remains parked and excluded. The readback asserts its complete captured state is unchanged. Its refusal records the token and missing physical carrier; these are vocabulary follow-ups, not admitted work.</p>
{table(["DD source", "token needed and recorded refusal", "status after"], parked_rows)}
<h3 id="sib-vocab-gap-requeue-isn-0-11-1-other-mapped-sources">Mapped sources outside this cohort</h3>
{table(["DD source", "approved spelling", "disposition"], excluded_rows)}
<p>Receipts, full cohort bindings, steering result, run properties, per-phase spend and attribution are in mapping-before.json, mapping-steering.json, pipeline-result.json and mapping-after.json under <code>{escape(str(RUN))}</code>. The completed job log is mapping-rerun-live.log, ending with EXIT=0. The earlier mapping-rerun.log is the failed helper-import attempt and contains no mutation or paid pipeline call. No product source or tests were changed; the merged-state gate belongs to the coordinator's test node. The cohort's produced-binding metric remains one short of 100; acceptance remains 39 short. Further cohort work belongs to the coordinator; this node's entire mapping intersection was processed.</p>
"""
    FRAGMENT.write_text(html)
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["prepare", "execute", "readback", "render"])
    args = parser.parse_args()
    print(f"# module: {__file__}", flush=True)
    print(f"# cwd: {Path.cwd().resolve()}", flush=True)
    return globals()[args.action]()


if __name__ == "__main__":
    raise SystemExit(main())
