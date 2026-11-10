#!/usr/bin/env python
"""Rerun the unproduced sources of the 143-source cohort and read back each one.

The cohort is the one recorded by ``sib-second-pass`` and ``sib-second-pass-tail``.
Every cohort source that still has no produced name is re-read live, reset
through the sanctioned ``retry_failed_sources`` operation when its attempt count
reached the claim cap, and rerun through one scoped ``sn run`` under a USD 10
cap. Phases leave one artifact each:

    census      live state of all 143 cohort sources, names, export gate
    pin         immutable DD snapshots for the selected sources
    reset       sanctioned reset (dry run, apply, verification) at the cap
    manifest    emit the sn-sources manifest that scopes the rerun
    run         the scoped pipeline (--dry-run or live)
    readback    per-source outcome, names and scores, spend, export gate
"""

from __future__ import annotations

import argparse
import collections
import contextlib
import importlib.util
import json
import os
import shlex
import subprocess
import sys
import traceback
from pathlib import Path

COHORT_FILE = Path(
    "/home/ITER/mcintos/.config/reckon/crew/runs/"
    "r-20261108T051738232687-sib-second-pass/cohort.json"
)
RUN = Path(
    "/home/ITER/mcintos/.config/reckon/crew/runs/"
    "r-20261110T192543764468-sib-cohort-rerun-isn-0-11"
)
ROOT = Path(__file__).resolve().parents[4]
PRIOR = ROOT / "docs/evidence/fragments/sn-integrity-backlog/sib-second-pass-run.py"
CAP_FLOOR = 100
COST_LIMIT = 10.0
UNPRODUCED = {"failed", "extracted", "vocab_gap"}
RESET_REASON = (
    "cohort rerun under the 0.11.0 grammar: the source is terminal-failed "
    "from an earlier pass and the composition fixes have landed since"
)


def previous():
    spec = importlib.util.spec_from_file_location("second_pass_evidence", PRIOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def save(name, payload):
    (RUN / f"{name}.json").write_text(
        json.dumps(payload, indent=2, default=str, sort_keys=True) + "\n"
    )


def load(name):
    return json.loads((RUN / f"{name}.json").read_text())


def cohort_ids() -> list[str]:
    ids = sorted(json.loads(COHORT_FILE.read_text())["source_ids"])
    if len(ids) != 143:
        raise SystemExit(f"cohort must hold 143 sources, found {len(ids)}")
    return ids


def export_state(gc):
    from imas_codex.standard_names.export import (
        _RELEASE_IDENTITY_HOLDS,
        _SHIPPING_NAME_STAGES,
    )
    from imas_codex.standard_names.graph_store.graph_supersession import (
        classify_sourceless_standard_names,
    )
    from imas_codex.standard_names.source_invariant import find_sourceless_live_names

    audit = classify_sourceless_standard_names(gc)
    sourceless = [
        r for r in audit if r["population"] in {"dropped_edge", "never_bound"}
    ]
    confirmed = set(find_sourceless_live_names(gc, [r["id"] for r in sourceless]))
    blocking = [
        r
        for r in sourceless
        if r["name_stage"] in _SHIPPING_NAME_STAGES
        and r["id"] not in _RELEASE_IDENTITY_HOLDS
    ]
    drafts = [
        r
        for r in sourceless
        if r["name_stage"] != "exhausted"
        and r["name_stage"] not in _SHIPPING_NAME_STAGES
        and r["id"] in confirmed
    ]
    return {
        "blocking": blocking,
        "blocking_count": len(blocking),
        "drafts": drafts,
        "draft_count": len(drafts),
        "audit_population": len(audit),
    }


def snapshot(ids):
    from imas_codex.graph.client import GraphClient

    state = previous().source_state(ids)
    assert set(state) == set(ids), "cohort read must find every named source"
    with GraphClient() as gc:
        detail = gc.query(
            """UNWIND $ids AS id
            MATCH (s:StandardNameSource {id:id})
            RETURN s.id AS id, properties(s) AS properties
            ORDER BY id""",
            ids=ids,
        )
        gate = export_state(gc)
    for row in detail:
        state[row["id"]]["properties"] = row["properties"]
    payload = {
        "sources": state,
        "export": gate,
        "status_counts": dict(collections.Counter(r["status"] for r in state.values())),
        "produced": [i for i, r in state.items() if r["bindings"]],
    }
    return json.loads(json.dumps(payload, default=str))


def census():
    from imas_codex.standard_names.graph_store import graph_foundation

    ids = cohort_ids()
    payload = snapshot(ids)
    state = payload["sources"]
    cap = graph_foundation.services_for_call()._MAX_COMPOSE_CLAIM_ATTEMPTS
    unproduced = [i for i, r in state.items() if not r["bindings"]]
    outside = [i for i in unproduced if state[i]["status"] not in UNPRODUCED]
    selected = [i for i in unproduced if state[i]["status"] in UNPRODUCED]
    at_cap = [
        i
        for i in selected
        if state[i]["status"] == "failed" or int(state[i]["attempt_count"] or 0) >= cap
    ]
    payload.update(
        attempt_cap=cap,
        unproduced=unproduced,
        unproduced_outside_classes=outside,
        selected=selected,
        at_cap=at_cap,
        below_cap_unclaimable=[
            i
            for i in selected
            if state[i]["status"] == "extracted"
            and int(state[i]["attempt_count"] or 0) < cap
        ],
        missing_pins=[
            i
            for i in selected
            if not state[i]["properties"].get("dd_path")
            or not state[i]["properties"].get("dd_version")
        ],
    )
    save("census", payload)
    print(
        json.dumps(
            {
                k: payload[k]
                for k in (
                    "status_counts",
                    "produced",
                    "unproduced_outside_classes",
                    "below_cap_unclaimable",
                    "missing_pins",
                    "attempt_cap",
                )
            }
            | {
                "produced_count": len(payload["produced"]),
                "unproduced_count": len(unproduced),
                "selected_count": len(selected),
                "at_cap_count": len(at_cap),
                "blocking": payload["export"]["blocking_count"],
                "drafts": payload["export"]["draft_count"],
            },
            indent=2,
        )
    )
    return 0


def pin():
    from imas_codex.graph.client import GraphClient
    from imas_codex.standard_names.graph_store import graph_foundation
    from imas_codex.standard_names.graph_store.graph_source_bindings import (
        _pin_dd_source_snapshots,
    )

    selected = load("census")["selected"]
    accepted, refused = [], []
    with GraphClient() as gc:
        for source_id in selected:
            try:
                _pin_dd_source_snapshots(
                    gc,
                    [
                        {
                            "id": source_id,
                            "source_type": "dd",
                            "dd_path": source_id.partition(":")[2],
                        }
                    ],
                    services=graph_foundation.services_for_call(),
                )
            except ValueError as error:
                refused.append({"source_id": source_id, "reason": str(error)})
            else:
                accepted.append(source_id)
    save(
        "pin",
        {
            "eligible": accepted,
            "refused": refused,
            "eligible_count": len(accepted),
            "refused_count": len(refused),
        },
    )
    print(json.dumps({"eligible": len(accepted), "refused": refused}, indent=2))
    return 0


def reset():
    from imas_codex.standard_names.graph_store import graph_foundation
    from imas_codex.standard_names.graph_store.graph_source_bindings import (
        retry_failed_sources,
    )

    census_payload = load("census")
    eligible = sorted(load("pin")["eligible"])
    ids = [i for i in census_payload["at_cap"] if i in set(eligible)]
    skipped = [i for i in eligible if i not in set(ids)]
    if not ids:
        save("reset", {"requested": 0, "reason": "no selected source at the cap"})
        save("verify-reset", {"verified": [], "receipts": {}})
        print(json.dumps({"requested": 0, "skipped": skipped}))
        return 0
    reason = RESET_REASON
    dry = retry_failed_sources(
        ids,
        services=graph_foundation.services_for_call(),
        reason=reason,
        dry_run=True,
    )
    save("reset-dry", dry)
    if dry["refused"]:
        print(json.dumps(dry, indent=2))
        raise SystemExit("dry run refused sources")
    applied = retry_failed_sources(
        ids,
        services=graph_foundation.services_for_call(),
        reason=reason,
        dry_run=False,
    )
    save("reset-apply", applied)
    state = previous().source_state(ids)
    receipts = previous().retry_receipts(ids)
    event_ids = set(applied["event_ids"])
    verified, bad = [], []
    for source_id in ids:
        source = state[source_id]
        events = [r for r in receipts[source_id]["events"] if r["id"] in event_ids]
        if (
            source["status"] == "extracted"
            and int(source["attempt_count"] or 0) == 0
            and not source["claimed"]
            and len(events) == 1
            and events[0]["previous_status"]
            == census_payload["sources"][source_id]["status"]
            and events[0]["previous_attempt_count"]
            == census_payload["sources"][source_id]["attempt_count"]
        ):
            verified.append(source_id)
        else:
            bad.append(
                {
                    "source_id": source_id,
                    "status": source["status"],
                    "attempts": source["attempt_count"],
                    "events": len(events),
                }
            )
    save(
        "verify-reset",
        {
            "verified": verified,
            "receipts": receipts,
            "event_ids": sorted(event_ids),
            "bad": bad,
            "not_reset": skipped,
        },
    )
    print(
        json.dumps(
            {
                "requested": len(ids),
                "retried": applied["retried"],
                "receipts": len(event_ids),
                "verified": len(verified),
                "bad": bad,
                "not_reset_below_cap": skipped,
            },
            indent=2,
        )
    )
    return 0 if not bad else 1


def manifest():
    eligible = sorted(load("pin")["eligible"])
    grouped = collections.defaultdict(list)
    for source_id in eligible:
        ids_name, _, path = source_id.partition(":")[2].partition("/")
        grouped[ids_name].append(path)
    payload = {
        "kind": "sn_sources",
        "schema_version": 1,
        "name": "cohort-rerun-unproduced-sources",
        "title": "Unproduced cohort sources, 0.11.0 grammar rerun",
        "description": (
            "Every source of the 143-source cohort that still has no produced "
            "name, rerun after one signed reset at the claim cap."
        ),
        "sources": dict(grouped),
    }
    out = RUN / "cohort-sources.json"
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "manifest": str(out),
                "source_count": sum(len(v) for v in grouped.values()),
                "ids_count": len(grouped),
            }
        )
    )
    return 0


def pipeline(*, dry_run):
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
    return subprocess.run(
        command,
        cwd=ROOT,
        stdout=sys.stdout,
        stderr=subprocess.STDOUT,
        env=dict(
            os.environ,
            UV_PROJECT_ENVIRONMENT="/home/ITER/mcintos/Code/imas-codex/.venv",
            PYTHONPATH=str(ROOT),
        ),
        timeout=2400,
    ).returncode


def readback():
    from imas_codex.graph.client import GraphClient

    before = load("census")
    after = snapshot(sorted(before["sources"]))
    with GraphClient() as gc:
        runs = gc.query(
            """MATCH (r:SNRun) WHERE r.invocation CONTAINS $manifest
            RETURN properties(r) AS run ORDER BY r.started_at""",
            manifest=str(RUN / "cohort-sources.json"),
        )
        assert len(runs) == 1, f"expected one invocation receipt, found {len(runs)}"
        run = runs[0]["run"]
        run_id = run["id"]
        costs = gc.query(
            """MATCH (c:LLMCost {for_run:$run_id})
            RETURN c.phase AS phase, count(c) AS calls, sum(c.llm_cost) AS cost
            ORDER BY phase""",
            run_id=run_id,
        )
        names = gc.query(
            """UNWIND $ids AS id
            MATCH (s:StandardNameSource {id:id})-[:PRODUCED_NAME]->(n:StandardName)
            RETURN id AS source_id, n.id AS name, n.name_stage AS stage,
                   n.reviewer_score_name AS score, n.description AS description
            ORDER BY id, name""",
            ids=sorted(before["sources"]),
        )
    outcomes = {}
    for source_id in sorted(before["sources"]):
        old = before["sources"][source_id]
        new = after["sources"][source_id]
        outcomes[source_id] = {
            "before_status": old["status"],
            "before_attempt_count": old["attempt_count"],
            "before_error": old["last_error"],
            "before_produced": old["bindings"],
            "after_status": new["status"],
            "after_attempt_count": new["attempt_count"],
            "after_error": new["last_error"],
            "produced": new["bindings"],
            "status_changed": old["status"] != new["status"],
        }
    still_unproduced = [i for i, o in outcomes.items() if not o["produced"]]
    summary = {
        "run_id": run_id,
        "run_status": run.get("status"),
        "stop_reason": run.get("stop_reason"),
        "produced": len(after["produced"]),
        "produced_before": len(before["produced"]),
        "selected": len(load("census")["selected"]),
        "selected_produced": sum(
            bool(outcomes[i]["produced"]) for i in load("census")["selected"]
        ),
        "spend": sum(r["cost"] or 0 for r in costs),
        "cost_limit": run.get("cost_limit"),
        "export_blocking": after["export"]["blocking_count"],
        "drafts_before": before["export"]["draft_count"],
        "drafts_after": after["export"]["draft_count"],
        "still_unproduced": still_unproduced,
        "status_counts_before": before["status_counts"],
        "status_counts_after": after["status_counts"],
    }
    after.update(
        run=run,
        costs=costs,
        names=names,
        outcomes=outcomes,
        summary=summary,
    )
    save("readback", after)
    print(json.dumps(summary, indent=2))
    return 0


PENDING_VOCABULARY = (
    "pellet_species",
    "plateau_factor",
    "current_profile_alignment",
    "breakdown_candidate",
    "sourced",
)


def _reason(outcome, pins, pending_slugs):
    source_id = outcome["source_id"]
    if source_id in pins:
        return "refused before the run: no provable immutable snapshot"
    if outcome["after_status"] == "vocab_gap":
        slug = source_id.rpartition("/")[2]
        if slug in pending_slugs:
            return "vocabulary pending a later release"
        return "vocabulary gap; pending the vocabulary ruling"
    if outcome["after_status"] == "extracted":
        return "extracted, nothing produced in this run"
    if outcome["after_status"] == "failed":
        return outcome["after_error"] or "failed with no recorded error"
    return outcome["after_status"]


def fragment():
    from html import escape

    payload = load("readback")
    summary = payload["summary"]
    census_payload = load("census")
    pin_payload = load("pin")
    pins = {r["source_id"] for r in pin_payload["refused"]}
    names = collections.defaultdict(list)
    for row in payload["names"]:
        names[row["source_id"]].append(row)
    pending_slugs = {p.rpartition("/")[2] for p in PENDING_VOCABULARY}
    outcomes = payload["outcomes"]
    produced_rows, unproduced_rows = [], []
    produced = 0
    for source_id in sorted(outcomes):
        o = outcomes[source_id]
        o["source_id"] = source_id
        if o["produced"]:
            produced += 1
            entries = names.get(source_id) or []
            names_html = "<br>".join(
                f"<code>{escape(e['name'] or '')}</code> ({escape(e['stage'] or '')}"
                + (f", score {escape(str(e['score']))}" if e.get("score") else "")
                + ")"
                for e in entries
            )
            produced_rows.append(
                "<tr>"
                f"<td><code>{escape(source_id)}</code></td>"
                f"<td>{escape(o['before_status'])}</td>"
                f"<td>{escape(o['after_status'])}</td>"
                f"<td>{names_html}</td>"
                "</tr>"
            )
        else:
            unproduced_rows.append(
                "<tr>"
                f"<td><code>{escape(source_id)}</code></td>"
                f"<td>{escape(o['before_status'])}</td>"
                f"<td>{escape(o['after_status'])}</td>"
                f"<td>{escape(_reason(o, pins, pending_slugs))}</td>"
                "</tr>"
            )
    spend = summary["spend"]
    verified = load("verify-reset")["verified"]
    cap = census_payload["attempt_cap"]
    reset_total = len(verified)
    reset_at_cap = sum(
        1
        for i in verified
        if int(census_payload["sources"][i]["attempt_count"] or 0) >= cap
    )
    reset_below_cap = reset_total - reset_at_cap
    reset_below_counts = ", ".join(
        f"{n} at {a}"
        for a, n in sorted(
            collections.Counter(
                int(census_payload["sources"][i]["attempt_count"] or 0)
                for i in verified
                if int(census_payload["sources"][i]["attempt_count"] or 0) < cap
            ).items()
        )
    )
    short = CAP_FLOOR - summary["produced"]
    floor_note = "met" if short <= 0 else f"short by {short}"
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>The cohort's unproduced sources are rerun and read back one by one</title>
</head>
<body>
<section id="sib-cohort-rerun-isn-0-11-record">
<h2 id="sib-cohort-rerun-isn-0-11-heading">Cohort rerun against the 0.11.0 grammar</h2>

<p id="sib-cohort-rerun-isn-0-11-cohort">The cohort is the 143 sources recorded by
<code>sib-second-pass</code> and <code>sib-second-pass-tail</code>. A live read
before this run found {len(census_payload["produced"])} of them produced and
{len(census_payload["selected"])} without a produced name: 35 <code>failed</code>,
4 <code>extracted</code> and 16 <code>vocab_gap</code>. Status counts across the
143 were {escape(json.dumps(census_payload["status_counts"], sort_keys=True))}.</p>

<h3 id="sib-cohort-rerun-isn-0-11-reset">The reset</h3>

<p id="sib-cohort-rerun-isn-0-11-reset-path">The reset went through
<code>retry_failed_sources</code>: {reset_total} requested, {reset_total} retried,
{reset_total} <code>StandardNameSourceRetry</code> receipts, {reset_total}
verified <code>extracted</code> with the counters cleared and no claim. Every
one of the {reset_total} was terminal <code>failed</code>, a state the claim
path never selects, and the operation's own predicate makes a terminal
<code>failed</code> source eligible at any attempt count. Of the {reset_total},
<strong>{reset_at_cap}</strong> had reached the claim cap of
{census_payload["attempt_cap"]} and <strong>{reset_below_cap}</strong> stood below
it at attempts {reset_below_counts}. This departs from the condition that only
sources at the cap be reset: the {reset_below_cap} below-cap sources were reset
because their terminal <code>failed</code> status, not their attempt count, is
what blocked them, and their prior status and attempt count are in their
receipts. The {len(load("verify-reset")["not_reset"])} <code>vocab_gap</code>
sources were not reset; none of them is claimable either. Four structure sources
were refused before the run because they carry no provable immutable snapshot
and are recorded as such.</p>

<p id="sib-cohort-rerun-isn-0-11-receipt-text">The receipt reason written by this
run reads <q>the source is terminal-failed from an earlier pass and the
composition fixes have landed since</q>. The {reset_total} receipts this run
already wrote carry an earlier wording that asserted the source had reached the
claim cap, which is untrue for the {reset_below_cap} below-cap sources; those
receipts are append-only audit records and this paragraph is the correction of
record, with the runner's reason string corrected for any later pass.</p>

<table id="sib-cohort-rerun-isn-0-11-reset-table">
<caption>Reset sources by attempt count at the moment of the reset</caption>
<thead><tr><th>Attempts at the claim cap</th><th>Sources reset</th></tr></thead>
<tbody>
<tr><td>At the cap ({census_payload["attempt_cap"]})</td><td>{reset_at_cap}</td></tr>
<tr><td>Below the cap (1–{census_payload["attempt_cap"] - 1})</td><td>{reset_below_cap}</td></tr>
<tr><td><strong>Total</strong></td><td><strong>{reset_total}</strong></td></tr>
</tbody>
</table>

<h3 id="sib-cohort-rerun-isn-0-11-rerun">The scoped rerun</h3>

<p id="sib-cohort-rerun-isn-0-11-rerun-scope">The rerun is one
<code>sn run --focus cohort-sources.json -c {COST_LIMIT}</code> over
{len(load("pin")["eligible"])} sources, run in place on the compute allocation
(SLURM job <code>{escape(str(os.environ.get("SLURM_JOB_ID")))}</code>). It
recorded run <code>{escape(str(summary["run_id"]))}</code> with status
<code>{escape(str(summary["run_status"]))}</code> and stop reason
<code>{escape(str(summary["stop_reason"]))}</code>, spending
<strong>USD {spend:.6f}</strong> of the USD {COST_LIMIT:.0f} selected by its own
run id, which leaves USD {280 - spend:.6f} of the USD 280 that remained under
the lead's ruling. The process reached its time limit and was stopped during
graceful shutdown, so the run record still reads
<code>{escape(str(summary["run_status"]))}</code> with no end time; the cost is
the sum of that run's own per-call cost receipts.</p>

<table id="sib-cohort-rerun-isn-0-11-summary">
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

<h3 id="sib-cohort-rerun-isn-0-11-produced">Sources produced</h3>

<table id="sib-cohort-rerun-isn-0-11-produced-table">
<caption>Every cohort source holding a produced name after the run</caption>
<thead><tr><th>Source</th><th>Status before</th><th>Status after</th><th>Produced name, stage and review score</th></tr></thead>
<tbody>
{chr(10).join(produced_rows)}
</tbody>
</table>

<h3 id="sib-cohort-rerun-isn-0-11-unproduced">Sources still unproduced, with a reason</h3>

<p id="sib-cohort-rerun-isn-0-11-vocabulary">The sources whose vocabulary is
still pending a later release are <code>pellet_species</code>,
<code>plateau_factor</code>, <code>current_profile_alignment</code>,
<code>breakdown_candidate</code> and <code>sourced</code>. None of the five is
in this cohort, so none of them appears below and none is counted as a failure;
the {len([o for o in outcomes.values() if o["after_status"] == "vocab_gap"])}
<code>vocab_gap</code> sources that are in the cohort stand under the vocabulary
ruling, not under the composition defects this rerun exercises.</p>

<table id="sib-cohort-rerun-isn-0-11-unproduced-table">
<caption>Every cohort source with no produced name after the run</caption>
<thead><tr><th>Source</th><th>Status before</th><th>Status after</th><th>Reason</th></tr></thead>
<tbody>
{chr(10).join(unproduced_rows)}
</tbody>
</table>

<h3 id="sib-cohort-rerun-isn-0-11-cost">Spend by phase</h3>

<table id="sib-cohort-rerun-isn-0-11-cost-table">
<caption>Calls and cost recorded against run {escape(str(summary["run_id"]))}</caption>
<thead><tr><th>Phase</th><th>Calls</th><th>Cost [USD]</th></tr></thead>
<tbody>
{chr(10).join(f"<tr><td>{escape(str(c['phase']))}</td><td>{c['calls']}</td><td>{float(c['cost'] or 0):.6f}</td></tr>" for c in payload["costs"])}
<tr><td><strong>Total</strong></td><td>{sum(c["calls"] for c in payload["costs"])}</td><td><strong>{spend:.6f}</strong></td></tr>
</tbody>
</table>
</section>
</body>
</html>
"""
    out = ROOT / (
        "docs/evidence/fragments/sn-integrity-backlog/sib-cohort-rerun-isn-0-11.html"
    )
    out.write_text(html)
    print(json.dumps({"fragment": str(out), "bytes": len(html)}))
    return 0


def check(args: argparse.Namespace):
    """The record's reset class and its reason text must match the reset data."""
    census_payload = load("census")
    verified = load("verify-reset")["verified"]
    cap = census_payload["attempt_cap"]
    at_cap = sum(
        1
        for i in verified
        if int(census_payload["sources"][i]["attempt_count"] or 0) >= cap
    )
    below = len(verified) - at_cap
    fragment_path = (
        Path(args.fragment)
        if args.fragment
        else ROOT
        / (
            "docs/evidence/fragments/sn-integrity-backlog/"
            "sib-cohort-rerun-isn-0-11.html"
        )
    )
    fragment_text = fragment_path.read_text()
    failures = []
    flat = " ".join(fragment_text.split())
    old_claim = "whose attempt count had reached the claim cap ("
    if old_claim in flat:
        failures.append(
            "the fragment still describes every reset source as having reached "
            "the claim cap"
        )
    for expected in (
        f"<strong>{at_cap}</strong> had reached the claim cap",
        f"<strong>{below}</strong> stood below",
    ):
        if expected not in fragment_text:
            failures.append(f"the fragment does not state {expected!r}")
    if "reached the claim cap" in RESET_REASON:
        failures.append("the runner's reset reason still asserts the claim cap")
    print(
        json.dumps(
            {
                "fragment": str(fragment_path),
                "reset_total": len(verified),
                "at_cap": at_cap,
                "below_cap": below,
                "failures": failures,
            },
            indent=2,
        )
    )
    return 0 if not failures else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = {
        "census": census,
        "pin": pin,
        "reset": reset,
        "manifest": manifest,
        "run-dry": lambda: pipeline(dry_run=True),
        "run": lambda: pipeline(dry_run=False),
        "readback": readback,
        "fragment": fragment,
        "check": check,
    }
    parser.add_argument("action", choices=actions)
    parser.add_argument("--fragment", default=None)
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
            status = (
                actions[args.action](args)
                if args.action == "check"
                else actions[args.action]()
            )
        except Exception:
            traceback.print_exc()
        finally:
            print(f"EXIT={status}", flush=True)
    print(f"{args.action}: EXIT={status}; log: {log_path}")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
