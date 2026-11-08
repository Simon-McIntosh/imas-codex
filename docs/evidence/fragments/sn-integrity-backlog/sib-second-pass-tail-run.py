#!/usr/bin/env python
"""Measure and retry an exact source cohort through sanctioned operations."""

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

ROOT = Path("/home/ITER/mcintos/Code/imas-codex")
RUN = Path(
    "/home/ITER/mcintos/.config/reckon/crew/runs/r-20261108T141648847743-sib-second-pass-tail"
)
PRIOR = ROOT / "docs/evidence/fragments/sn-integrity-backlog/sib-second-pass-run.py"


def previous():
    spec = importlib.util.spec_from_file_location("cohort_evidence", PRIOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def save(name, payload):
    (RUN / f"{name}.json").write_text(
        json.dumps(payload, indent=2, default=str, sort_keys=True) + "\n"
    )


def load(name):
    return json.loads((RUN / f"{name}.json").read_text())


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
        "population_counts": dict(collections.Counter(r["population"] for r in audit)),
    }


def snapshot(ids):
    from imas_codex.graph.client import GraphClient

    state = previous().source_state(ids)
    assert set(state) == set(ids), "cohort read must find every named source"
    with GraphClient() as gc:
        detail = gc.query(
            """UNWIND $ids AS id
            MATCH (s:StandardNameSource {id:id})
            RETURN s.id AS id, properties(s) AS properties,
              [(s)-[:HAS_CLAIM_ATTEMPT]->(a:StandardNameSourceClaimAttempt) | properties(a)] AS claims
            ORDER BY id""",
            ids=ids,
        )
        gate = export_state(gc)
    for row in detail:
        state[row["id"]]["properties"] = row["properties"]
        state[row["id"]]["claim_receipts"] = row["claims"]
    return {
        "sources": state,
        "export": gate,
        "status_counts": dict(collections.Counter(r["status"] for r in state.values())),
        "produced": [i for i, r in state.items() if r["bindings"]],
    }


def census():
    import imas_codex.graph.client as client
    import imas_codex.graph.models as models

    assert hasattr(models, "StandardNameSourceClaimAttempt")
    ids = sorted(r["id"] for r in previous().cohort())
    assert len(ids) == 143
    payload = snapshot(ids)
    state = payload["sources"]
    payload["selected"] = [
        i
        for i, r in state.items()
        if r["status"] == "extracted" or (r["status"] == "failed" and not r["bindings"])
    ]
    payload["failed"] = [
        i for i in payload["selected"] if state[i]["status"] == "failed"
    ]
    payload["extracted"] = [
        i for i in payload["selected"] if state[i]["status"] == "extracted"
    ]
    payload["vocab_gap"] = [i for i, r in state.items() if r["status"] == "vocab_gap"]
    payload["bound_failed_excluded"] = [
        i for i, r in state.items() if r["status"] == "failed" and r["bindings"]
    ]
    payload["missing_pins"] = [
        i
        for i in payload["selected"]
        if not state[i]["properties"].get("dd_path")
        or not state[i]["properties"].get("dd_version")
    ]
    payload["module"] = client.__file__
    payload["models"] = models.__file__
    payload["prerequisite"] = {
        "coordinator_build_revision": "0efd4cf50",
        "coordinator_build_exit": 0,
        "receipt": str(RUN / "resume-1-advice.txt"),
        "model_present": True,
    }
    save("census", payload)
    print(
        json.dumps(
            {
                k: payload[k]
                for k in [
                    "status_counts",
                    "failed",
                    "extracted",
                    "vocab_gap",
                    "bound_failed_excluded",
                    "missing_pins",
                ]
            },
            indent=2,
        )
    )
    print(
        json.dumps(
            {
                "produced": len(payload["produced"]),
                "selected": len(payload["selected"]),
                "blocking": payload["export"]["blocking_count"],
                "drafts": payload["export"]["draft_count"],
                "audit_population": payload["export"]["audit_population"],
            }
        )
    )
    return 0


def preflight():
    from imas_codex.graph.client import GraphClient
    from imas_codex.standard_names.graph_store import graph_foundation
    from imas_codex.standard_names.graph_store.graph_source_bindings import (
        _pin_dd_source_snapshots,
    )

    before = load("census")
    accepted, refused = [], []
    with GraphClient() as gc:
        for source_id in before["selected"]:
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
    payload = {
        "eligible": accepted,
        "refused": refused,
        "eligible_count": len(accepted),
        "refused_count": len(refused),
        "graph_writes": 0,
    }
    save("seed-preflight", payload)
    print(json.dumps(payload, indent=2))
    return 1 if refused else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["census", "seed-preflight"])
    args = parser.parse_args()
    revision = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
    ).strip()
    log_path = RUN / f"{args.action}.log"
    status = 1
    with (
        log_path.open("x") as log,
        contextlib.redirect_stdout(log),
        contextlib.redirect_stderr(log),
    ):
        print(
            f"# revision: {revision}; tree: {ROOT}; command: {shlex.join(['uv', 'run', '--no-sync', 'python', *sys.argv])}",
            flush=True,
        )
        print(
            f"# cwd: {Path.cwd().resolve()}; PYTHONPATH: {os.environ.get('PYTHONPATH')}",
            flush=True,
        )
        try:
            status = {"census": census, "seed-preflight": preflight}[args.action]()
        except Exception:
            traceback.print_exc()
        finally:
            print(f"EXIT={status}", flush=True)
    print(f"{args.action}: EXIT={status}; log: {log_path}")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
