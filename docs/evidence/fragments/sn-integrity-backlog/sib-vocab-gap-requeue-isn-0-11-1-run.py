#!/usr/bin/env python
"""Read source refusals and active vocabulary before admitting a scoped retry."""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["census"])
    args = parser.parse_args()
    print(f"# module: {__file__}", flush=True)
    print(f"# cwd: {Path.cwd().resolve()}", flush=True)
    return globals()[args.action]()


if __name__ == "__main__":
    raise SystemExit(main())
