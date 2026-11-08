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
    payload = {
        "sources": state,
        "export": gate,
        "status_counts": dict(collections.Counter(r["status"] for r in state.values())),
        "produced": [i for i, r in state.items() if r["bindings"]],
    }
    return json.loads(json.dumps(payload, default=str))


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


def prepare():
    ids = sorted(load("seed-preflight")["eligible"])
    assert len(ids) == 41 and ids == sorted(load("census")["failed"])
    before = snapshot(sorted(load("census")["sources"]))
    for source_id in ids:
        source = before["sources"][source_id]
        assert source["status"] == "failed" and not source["bindings"]
        assert not source["claimed"]
    before["retry_receipts"] = previous().retry_receipts(ids)
    before["selected"] = ids
    save("before-reset", before)
    grouped = collections.defaultdict(list)
    for source_id in ids:
        ids_name, _, path = source_id.partition(":")[2].partition("/")
        grouped[ids_name].append(path)
    save(
        "cohort-sources",
        {
            "kind": "sn_sources",
            "schema_version": 1,
            "name": "eligible-source-retry",
            "title": "Eligible failed source retry",
            "description": "Exact authorized cohort; unpinned structure sources excluded.",
            "sources": dict(grouped),
        },
    )
    print(
        json.dumps(
            {
                "selected": len(ids),
                "produced_before": len(before["produced"]),
                "drafts_before": before["export"]["draft_count"],
                "blocking_before": before["export"]["blocking_count"],
            }
        )
    )
    return 0


def retry(*, dry_run):
    from imas_codex.standard_names.graph_store import graph_foundation
    from imas_codex.standard_names.graph_store.graph_source_bindings import (
        retry_failed_sources,
    )

    before = load("before-reset")
    ids = before["selected"]
    current = previous().source_state(ids)
    for source_id in ids:
        for field in ("status", "attempt_count", "last_error", "bindings", "claimed"):
            assert current[source_id][field] == before["sources"][source_id][field], (
                source_id,
                field,
            )
    result = retry_failed_sources(
        ids,
        services=graph_foundation.services_for_call(),
        reason="Retry after source-level persist refusal isolation and durable claim-outcome receipts were deployed; immutable-snapshot sources only.",
        dry_run=dry_run,
    )
    save("retry-dry" if dry_run else "retry-apply", result)
    print(json.dumps(result, indent=2))
    assert result["eligible"] == 41 and result["refused"] == 0
    if not dry_run:
        assert result["retried"] == 41 and len(result["event_ids"]) == 41
    return 0


def verify_reset():
    before = load("before-reset")
    state = previous().source_state(sorted(before["sources"]))
    receipts = previous().retry_receipts(before["selected"])
    event_ids = set(load("retry-apply")["event_ids"])
    verified = []
    for source_id in before["selected"]:
        source = state[source_id]
        assert source["status"] == "extracted" and source["attempt_count"] == 0
        assert not source["claimed"] and not source["bindings"]
        events = [r for r in receipts[source_id]["events"] if r["id"] in event_ids]
        assert len(events) == 1
        event = events[0]
        for receipt_field, field in (
            ("previous_status", "status"),
            ("previous_attempt_count", "attempt_count"),
            ("previous_error", "last_error"),
        ):
            assert event[receipt_field] == before["sources"][source_id][field]
        verified.append(source_id)
    changed_controls = [
        source_id
        for source_id in state
        if source_id not in before["selected"]
        and any(
            state[source_id][field] != before["sources"][source_id][field]
            for field in state[source_id]
        )
    ]
    payload = {
        "verified": verified,
        "receipt_count": len(event_ids),
        "receipts": receipts,
        "changed_controls": changed_controls,
        "control_count": len(state) - len(verified),
    }
    save("verify-reset", payload)
    print(
        json.dumps(
            {
                "verified_count": len(verified),
                "receipt_count": len(event_ids),
                "changed_controls": changed_controls,
                "control_count": payload["control_count"],
            }
        )
    )
    assert not changed_controls
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
        "48.51",
    ]
    if dry_run:
        command.append("--dry-run")
    else:
        assert len(load("verify-reset")["verified"]) == 41
        command.extend(["-t", "55"])
    print(f"# pipeline command: {shlex.join(command)}", flush=True)
    print(f"# SLURM_JOB_ID: {os.environ.get('SLURM_JOB_ID')}", flush=True)
    return subprocess.run(
        command,
        cwd=ROOT,
        stdout=sys.stdout,
        stderr=subprocess.STDOUT,
        env=dict(
            os.environ, UV_PROJECT_ENVIRONMENT=str(ROOT / ".venv"), PYTHONPATH=str(ROOT)
        ),
        timeout=3600,
    ).returncode


def readback():
    from imas_codex.graph.client import GraphClient

    before = load("before-reset")
    after = snapshot(sorted(before["sources"]))
    with GraphClient() as gc:
        runs = gc.query(
            """MATCH (r:SNRun) WHERE r.invocation CONTAINS $manifest
            RETURN properties(r) AS run ORDER BY r.started_at""",
            manifest=str(RUN / "cohort-sources.json"),
        )
        assert len(runs) == 1, (
            f"expected one exact invocation receipt, found {len(runs)}"
        )
        run = runs[0]["run"]
        run_id = run["id"]
        costs = gc.query(
            """MATCH (c:LLMCost {for_run:$run_id})
            RETURN c.phase AS phase, count(c) AS calls, sum(c.llm_cost) AS cost ORDER BY phase""",
            run_id=run_id,
        )
        prior_costs = gc.query(
            """MATCH (c:LLMCost {for_run:$run_id})
            RETURN count(c) AS calls, sum(c.llm_cost) AS cost""",
            run_id="9192044a-3fa0-4592-95a8-cb3cf0977823",
        )
        names = gc.query(
            """UNWIND $ids AS id MATCH (s:StandardNameSource {id:id})-[:PRODUCED_NAME]->(n:StandardName)
            RETURN id AS source_id, n.id AS name, n.description AS description,
                n.name_stage AS stage, n.reviewer_score_name AS score ORDER BY id, name""",
            ids=before["selected"],
        )
    assert prior_costs[0]["calls"] > 0
    outcomes = {}
    for source_id in before["selected"]:
        source = after["sources"][source_id]
        old_ids = {r["id"] for r in before["sources"][source_id]["claim_receipts"]}
        new_receipts = [r for r in source["claim_receipts"] if r["id"] not in old_ids]
        outcomes[source_id] = {
            "status": source["status"],
            "bindings": source["bindings"],
            "last_error": source["last_error"],
            "attempt_count": source["attempt_count"],
            "new_claim_receipts": new_receipts,
        }
    controls = load("census")["extracted"] + load("census")["vocab_gap"]
    control_changes = [
        i for i in controls if after["sources"][i] != before["sources"][i]
    ]
    spend = sum(r["cost"] for r in costs)
    scope_id = json.loads(run["invocation_scope"])["scope_run_id"]
    receipt_run_ids = sorted(
        {
            r.get("run_id") or "(null)"
            for o in outcomes.values()
            for r in o["new_claim_receipts"]
        }
    )
    missing_outcomes = [
        i
        for i, o in outcomes.items()
        if not o["bindings"]
        and (
            o["status"] not in {"failed", "vocab_gap"}
            or (
                o["status"] == "failed"
                and (not o["last_error"] or not o["new_claim_receipts"])
            )
        )
    ]
    summary = {
        "run_id": run_id,
        "scope_run_id": scope_id,
        "produced": len(after["produced"]),
        "selected_produced": sum(bool(o["bindings"]) for o in outcomes.values()),
        "selected": len(outcomes),
        "spend": spend,
        "prior_spend": prior_costs[0]["cost"],
        "cumulative_spend": spend + prior_costs[0]["cost"],
        "export_blocking": after["export"]["blocking_count"],
        "drafts_before": before["export"]["draft_count"],
        "drafts_after": after["export"]["draft_count"],
        "excluded_control_changes": control_changes,
        "missing_outcomes": missing_outcomes,
        "claim_receipt_run_ids": receipt_run_ids,
        "selected_status_counts": dict(
            collections.Counter(o["status"] for o in outcomes.values())
        ),
    }
    after.update(
        run=run,
        cost_by_phase=costs,
        prior_costs=prior_costs,
        selected_outcomes=outcomes,
        names=names,
        summary=summary,
    )
    save("readback", after)
    print(json.dumps(summary, indent=2))
    return (
        0
        if summary["produced"] >= 100
        and not missing_outcomes
        and not control_changes
        and spend <= 48.51
        and summary["cumulative_spend"] <= 50
        and summary["export_blocking"] == 0
        and summary["drafts_after"] <= summary["drafts_before"]
        else 1
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = {
        "census": census,
        "seed-preflight": preflight,
        "prepare": prepare,
        "retry-dry": lambda: retry(dry_run=True),
        "retry-apply": lambda: retry(dry_run=False),
        "verify-reset": verify_reset,
        "run-dry": lambda: pipeline(dry_run=True),
        "run": lambda: pipeline(dry_run=False),
        "readback": readback,
    }
    parser.add_argument("action", choices=actions)
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
            status = actions[args.action]()
        except Exception:
            traceback.print_exc()
        finally:
            print(f"EXIT={status}", flush=True)
    print(f"{args.action}: EXIT={status}; log: {log_path}")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
