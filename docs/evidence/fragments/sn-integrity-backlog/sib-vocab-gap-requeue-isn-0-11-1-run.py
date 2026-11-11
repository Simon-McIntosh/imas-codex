#!/usr/bin/env python
"""Read source refusals and active vocabulary before admitting a scoped retry."""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
from html import escape
from pathlib import Path

from imas_codex.graph.client import GraphClient

ROOT = Path(__file__).resolve().parents[4]
RUN = Path(
    "/home/ITER/mcintos/.config/reckon/crew/runs/r-20261111T150911488391-sib-vocab-gap-requeue-isn-0-11-1"
)
PREDECESSOR = Path(
    "/home/ITER/mcintos/.config/reckon/crew/runs/r-20261110T210102261723-sib-vocab-gap-requeue"
)
COHORT_INPUT = Path(
    "/home/ITER/mcintos/.config/reckon/crew/runs/r-20261110T192543764468-sib-cohort-rerun-isn-0-11/census.json"
)
FRAGMENT = (
    ROOT
    / "docs/evidence/fragments/sn-integrity-backlog/sib-vocab-gap-requeue-isn-0-11-1.html"
)
VERSION = "0.11.1"
CONTROL_TOKEN = "power_convertor"
RELEASE_TOKENS = [
    "pellet_species",
    "plateau_factor",
    "current_profile_alignment",
    "breakdown_candidate",
    "sourced",
]


def source_reader():
    path = ROOT / "docs/evidence/fragments/sn-integrity-backlog/sib-second-pass-run.py"
    spec = importlib.util.spec_from_file_location("source_evidence", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.source_state


def census():
    prior = json.loads((PREDECESSOR / "readback.json").read_text())["outcomes"]
    targets = {}
    for sid, row in prior.items():
        if row["after_status"] == "vocab_gap":
            segment, token = re.search(
                r"missing (\w+) vocabulary token '([^']+)'", row["after_error"]
            ).groups()
            targets[sid] = {
                "segment": segment,
                "token": token,
                "recorded_refusal": row["after_error"],
            }
    sid = "dd:summary/global_quantities/current_alignment/value"
    targets[sid] = {
        "segment": "physical_base",
        "token": "current_alignment_factor",
        "decomposition_token": "current_alignment",
        "recorded_refusal": "Predecessor excluded this source because its reviewed decomposition still required current_alignment.",
    }
    assert len(targets) == 5, targets
    cohort = sorted(json.loads(COHORT_INPUT.read_text())["sources"])
    assert len(cohort) == 143
    read_sources = source_reader()
    state = read_sources(sorted(targets))
    with GraphClient() as gc:
        active = gc.query(
            "MATCH (v:ISNGrammarVersion {active:true}) RETURN v.version AS version, v.content_digest AS digest"
        )
        assert len(active) == 1 and active[0]["version"] == VERSION, active
        tokens = gc.query(
            "MATCH (t:GrammarToken {version:$version}) RETURN t.value AS value, t.segment AS segment, t.aliases AS aliases, t.description AS description",
            version=VERSION,
        )
        assert len(tokens) >= 1000 and all(t["value"] and t["segment"] for t in tokens)
        assert any(t["value"] == CONTROL_TOKEN for t in tokens)
        accepted = gc.query(
            "UNWIND $ids AS id MATCH (s:StandardNameSource {id:id}) OPTIONAL MATCH (s)-[:PRODUCED_NAME]->(n:StandardName) RETURN id AS id, s.status AS status, collect(DISTINCT {name:n.id, stage:n.name_stage, score:n.reviewer_score_name}) AS names ORDER BY id",
            ids=cohort,
        )
        assert len(accepted) == 143
    for sid, row in targets.items():
        row["live"] = state[sid]
        spellings = [row["token"]] + (
            [row["decomposition_token"]] if "decomposition_token" in row else []
        )
        row["token_rows"] = [
            t
            for t in tokens
            if t["value"] in spellings
            or any(a in spellings for a in t["aliases"] or [])
        ]
        row["admitted"] = bool(row["token_rows"])
    count = sum(
        any(n["stage"] in ("accepted", "approved") for n in r["names"])
        for r in accepted
    )
    payload = {
        "active": active,
        "grammar_rows": len(tokens),
        "positive_control": [t for t in tokens if t["value"] == CONTROL_TOKEN],
        "release_additions": [t for t in tokens if t["value"] in RELEASE_TOKENS],
        "sources": targets,
        "eligible": sorted(s for s, r in targets.items() if r["admitted"]),
        "cohort": accepted,
        "accepted_sources": count,
        "floor": 100,
        "size": 143,
    }
    (RUN / "census.json").write_text(json.dumps(payload, indent=2, default=str) + "\n")
    print(
        json.dumps(
            {
                k: payload[k]
                for k in (
                    "active",
                    "grammar_rows",
                    "eligible",
                    "accepted_sources",
                    "floor",
                    "size",
                )
            },
            indent=2,
        )
    )
    for sid, row in targets.items():
        print(json.dumps({"source": sid, **row}, default=str))
    return 0


def confirm():
    before = json.loads((RUN / "census.json").read_text())
    assert not before["eligible"], "Eligible sources require a scoped pipeline run"
    state = source_reader()(sorted(before["sources"]))
    cohort = source_reader()([row["id"] for row in before["cohort"]])
    with GraphClient() as gc:
        stages = gc.query(
            "UNWIND $ids AS id MATCH (s:StandardNameSource {id:id})-[:PRODUCED_NAME]->(n:StandardName) RETURN id AS id, collect(DISTINCT n.name_stage) AS stages",
            ids=sorted(cohort),
        )
    accepted = sum(
        any(s in ("accepted", "approved") for s in r["stages"]) for r in stages
    )
    controls = []
    for sid, row in before["sources"].items():
        assert not row["token_rows"]
        assert state[sid] == row["live"], (sid, row["live"], state[sid])
        if row["live"]["status"] == "vocab_gap":
            controls.append(sid)
    assert len(controls) == 4
    result = {
        "accepted_before": before["accepted_sources"],
        "accepted_after": accepted,
        "floor": 100,
        "cohort": 143,
        "eligible": 0,
        "scoped_run_launched": False,
        "run_id": None,
        "own_produced_names": 0,
        "spend_usd": 0,
        "cap_usd": 3,
        "absent_token_controls_still_parked": controls,
    }
    (RUN / "confirmation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    rows = []
    for sid, row in before["sources"].items():
        token = row["token"] + (
            "; decomposition: " + row["decomposition_token"]
            if "decomposition_token" in row
            else ""
        )
        cells = [
            sid.removeprefix("dd:"),
            row["segment"],
            token,
            "absent, including aliases",
            state[sid]["status"],
            state[sid]["last_error"],
        ]
        rows.append(
            "<tr>" + "".join("<td>" + escape(str(c)) + "</td>" for c in cells) + "</tr>"
        )
    additions = ", ".join(sorted({t["value"] for t in before["release_additions"]}))
    witnesses = []
    for row in before["cohort"]:
        for name in row["names"]:
            if name["stage"] in ("accepted", "approved"):
                witnesses.append(
                    f"{row['id']}: {name['name']} ({name['stage']}, score {name['score']})"
                )
        if len(witnesses) >= 2:
            break
    FRAGMENT.write_text(f"""<h2 id="sib-vocab-gap-requeue-isn-0-11-1">Retry blocked: the requested tokens remain absent from active ISN 0.11.1</h2>
<p>The active graph snapshot is 0.11.1, digest <code>{escape(before["active"][0]["digest"])}</code>, with {before["grammar_rows"]} GrammarToken rows. The same read sees <code>power_convertor</code> twice (device and object), and the release additions <code>{escape(additions)}</code>. All requested spellings and their aliases return no matching row.</p>
<p>The predecessor fragment records four sources that returned to vocab_gap and one additional source excluded for missing current_alignment. Each is listed below with its live refusal. None has a bound name, name stage or review score. This node changes no source state and launches no paid pipeline: the admission set is empty.</p>
<table><thead><tr><th>DD source</th><th>segment</th><th>waited-on spelling</th><th>0.11.1</th><th>live status, both reads</th><th>live refusal</th></tr></thead><tbody>
{chr(10).join(rows)}
</tbody></table>
<p>The recorded defect still reproduces: <q>missing geometry vocabulary token 'electron_cyclotron_power_deposition_peak'</q>; the other three re-parked sources likewise retain their original refusals. The negative control reads all four again and asserts their complete captured state unchanged, status vocab_gap, no binding, absent token and exclusion from the empty admission set. It does not mutate production data.</p>
<p><code>current_alignment_factor</code> is the source's requested figure-of-merit spelling; its reviewed decomposition requires <code>current_alignment</code>. The release instead registers <code>current_profile_alignment</code> as a physical base with no alias to either spelling. That spelling names profile alignment directly. Whether to replace the reviewed decomposition with this base is a semantic ruling still needed; this read does not establish equivalence or authorize that substitution.</p>
<p>The cohort acceptance measure counts a source once if a PRODUCED_NAME target is at accepted or approved. Before admission: {result["accepted_before"]} of 143; after the empty admission decision: {accepted} of 143; required floor: 100 (shortfall {100 - accepted}). The two measurements are live reads, not a pipeline before/after result. Known accepted witnesses establish the query can see accepted bindings: {escape("; ".join(witnesses))}.</p>
<p>Own run id: none; own produced names: 0; spend: USD 0 of USD 3. No cost reservation, capacity limit or scheduling delay withheld the spend: no source satisfied the token-presence admission condition. The requested scoped run on partition all remains unperformed.</p>
<p>Evidence: census job 1310643 on all_debug, census.json and census.log, followed by confirmation.json and confirmation.log, under <code>{escape(str(RUN))}</code>. The committed helper reproduces both reads. No source or tests were changed. The coordinator must settle an accepted vocabulary spelling or approve a specific decomposition before dispatching the scoped rerun.</p>
""")
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["census", "confirm"])
    args = parser.parse_args()
    print(f"# module: {__file__}", flush=True)
    print(f"# cwd: {Path.cwd().resolve()}", flush=True)
    return globals()[args.action]()


if __name__ == "__main__":
    raise SystemExit(main())
