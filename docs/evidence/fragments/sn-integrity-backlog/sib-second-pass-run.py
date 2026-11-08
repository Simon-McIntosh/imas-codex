#!/usr/bin/env python
"""Reset the recoverable failed DD sources and rerun them under a spend cap.

The cohort is read from the full-run failure diagnosis: every source whose
recovery class is isolated_retry, registered_token_retry,
binding_now_admissible or attachment_now_admissible. The reset goes through
``retry_failed_sources`` — the same operation the ``sn retry --failed`` CLI
exposes — which writes one StandardNameSourceRetry event per source carrying
its prior status, attempt count and error before the counters are cleared.

Phases are separate so the dry run, the mutation and the read-back each leave
their own artifact:

    census        pre-reset live state of the cohort and of every non-cohort
                  failed source (the control that must not move)
    retry-dry     sanctioned reset, dry run
    retry-apply   sanctioned reset, applied
    verify        post-reset state of the cohort and of the control
    manifest      emit the sn-sources manifest that scopes the rerun
    readback      per-source run outcome, names by stage, spend by run id
"""

from __future__ import annotations

import argparse
import collections
import json
import subprocess
import sys
from pathlib import Path

DIAGNOSIS = Path(
    "/home/ITER/mcintos/Code/imas-codex/docs/evidence/fragments/"
    "sn-integrity-backlog/sib-full-run-failure-diagnosis.json"
)
RUN_DIR = Path(
    "/home/ITER/mcintos/.config/reckon/crew/runs/r-20261108T051738232687-sib-second-pass"
)
RECOVERY_CLASSES = (
    "isolated_retry",
    "registered_token_retry",
    "binding_now_admissible",
    "attachment_now_admissible",
)
REASON = (
    "second pass over the recoverable failed cohort: the composition batch "
    "isolation and grammar fixes have landed since these sources exhausted "
    "their attempt cap"
)


def cohort() -> list[dict]:
    payload = json.loads(DIAGNOSIS.read_text())
    rows = [s for s in payload["sources"] if s.get("recovery") in RECOVERY_CLASSES]
    if not rows:
        raise SystemExit("cohort is empty: the diagnosis carries no recoverable source")
    return rows


def source_state(source_ids: list[str]) -> dict[str, dict]:
    from imas_codex.graph.client import GraphClient

    out: dict[str, dict] = {}
    with GraphClient() as gc:
        rows = list(
            gc.query(
                """
                UNWIND $ids AS id
                MATCH (s:StandardNameSource {id: id})
                OPTIONAL MATCH (s)-[:PRODUCED_NAME]->(n:StandardName)
                RETURN id AS source_id, s.status AS status,
                       s.attempt_count AS attempt_count,
                       s.last_error AS last_error,
                       s.claimed_at IS NOT NULL OR s.claim_token IS NOT NULL
                         AS claimed,
                       [r IN coalesce(s.retry_events, [])] AS retry_events,
                       collect(DISTINCT n.id) AS bindings
                ORDER BY source_id
                """,
                ids=sorted(source_ids),
            )
        )
        for row in rows:
            out[row["source_id"]] = {
                "status": row["status"],
                "attempt_count": row["attempt_count"],
                "last_error": row["last_error"],
                "claimed": bool(row["claimed"]),
                "retry_events": list(row["retry_events"] or []),
                "bindings": sorted(b for b in row["bindings"] if b),
            }
    return out


def failed_census(exclude: list[str]) -> dict:
    from imas_codex.graph.client import GraphClient

    with GraphClient() as gc:
        rows = list(
            gc.query(
                """
                MATCH (s:StandardNameSource)
                WHERE s.status = 'failed' AND NOT s.id IN $exclude
                RETURN s.id AS id ORDER BY s.id
                """,
                exclude=sorted(exclude),
            )
        )
    ids = [r["id"] for r in rows]
    return {"count": len(ids), "ids": ids}


def retry_receipts(source_ids: list[str]) -> dict[str, dict]:
    """Every retry event hanging off a cohort source, oldest first."""
    from imas_codex.graph.client import GraphClient

    out: dict[str, dict] = {}
    with GraphClient() as gc:
        rows = list(
            gc.query(
                """
                UNWIND $ids AS id
                MATCH (s:StandardNameSource {id: id})
                OPTIONAL MATCH (s)-[:HAS_RETRY_EVENT]->(e:StandardNameSourceRetry)
                RETURN id AS source_id,
                       collect(DISTINCT {
                         id: e.id,
                         previous_status: e.previous_status,
                         previous_attempt_count: e.previous_attempt_count,
                         previous_error: e.previous_error,
                         reason: e.reason,
                         retried_at: toString(e.retried_at)
                       }) AS events
                ORDER BY source_id
                """,
                ids=sorted(source_ids),
            )
        )
    for row in rows:
        events = [e for e in row["events"] if e.get("id")]
        events.sort(key=lambda e: e.get("retried_at") or "")
        out[row["source_id"]] = {"events": events, "count": len(events)}
    return out


def phase_cohort(_: argparse.Namespace) -> dict:
    rows = cohort()
    counts = collections.Counter(r["recovery"] for r in rows)
    return {
        "cohort_size": len(rows),
        "recovery_counts": dict(sorted(counts.items())),
        "source_ids": [r["id"] for r in rows],
        "classes": dict(sorted(collections.Counter(r.get("class") for r in rows).items())),
        "attempts": dict(
            sorted(collections.Counter(r.get("attempts") or 0 for r in rows).items())
        ),
    }


def phase_census(args: argparse.Namespace) -> dict:
    rows = cohort()
    ids = [r["id"] for r in rows]
    state = source_state(ids)
    missing = sorted(set(ids) - set(state))
    if missing:
        raise SystemExit(f"cohort sources absent from the graph: {missing[:5]}")
    control = failed_census(ids)
    payload = {
        "cohort_size": len(ids),
        "cohort_state": state,
        "cohort_status_counts": dict(
            sorted(collections.Counter(s["status"] for s in state.values()).items())
        ),
        "cohort_claimed": sum(1 for s in state.values() if s["claimed"]),
        "control_failed_count": control["count"],
        "control_failed_ids": control["ids"],
    }
    return payload


def _retry(rows: list[dict], *, dry_run: bool) -> dict:
    from imas_codex.standard_names.graph_store import graph_foundation
    from imas_codex.standard_names.graph_store.graph_source_bindings import (
        retry_failed_sources,
    )

    return retry_failed_sources(
        [r["id"] for r in rows],
        services=graph_foundation.services_for_call(),
        reason=REASON,
        dry_run=dry_run,
    )


def phase_retry_dry(args: argparse.Namespace) -> dict:
    return _retry(cohort(), dry_run=True)


def phase_retry_apply(args: argparse.Namespace) -> dict:
    return _retry(cohort(), dry_run=False)


def phase_verify(args: argparse.Namespace) -> dict:
    """Every cohort source is extracted, carries one receipt, and is unclaimed."""
    rows = cohort()
    ids = [r["id"] for r in rows]
    state = source_state(ids)
    receipts = retry_receipts(ids)
    before = json.loads((RUN_DIR / "census.json").read_text())
    control = failed_census(ids)

    expected = {r["id"]: r for r in rows}
    not_extracted: list[str] = []
    still_claimed: list[str] = []
    wrong_attempts: list[str] = []
    no_receipt: list[str] = []
    receipt_mismatch: list[str] = []
    for source_id in ids:
        live = state.get(source_id)
        if not live:
            not_extracted.append(source_id)
            continue
        if live["status"] != "extracted":
            not_extracted.append(source_id)
        if live["claimed"]:
            still_claimed.append(source_id)
        if live["attempt_count"] != 0:
            wrong_attempts.append(source_id)
        events = receipts.get(source_id, {}).get("events") or []
        if not events:
            no_receipt.append(source_id)
            continue
        newest = events[-1]
        want = expected[source_id]
        prior_status = before["cohort_state"][source_id]["status"]
        prior_attempts = before["cohort_state"][source_id]["attempt_count"]
        prior_error = before["cohort_state"][source_id]["last_error"]
        if (
            newest.get("previous_status") != prior_status
            or newest.get("previous_attempt_count") != prior_attempts
            or newest.get("previous_error") != prior_error
        ):
            receipt_mismatch.append(
                f"{source_id}: receipt={newest} census="
                f"({prior_status},{prior_attempts},{prior_error!r})"
            )
        if want.get("last_error") and prior_error != want["last_error"]:
            receipt_mismatch.append(
                f"{source_id}: census error {prior_error!r} does not match "
                f"diagnosis {want['last_error']!r}"
            )

    control_after = set(control["ids"])
    control_before = set(before["control_failed_ids"])
    control_lost = sorted(control_before - control_after)
    control_gained = sorted(control_after - control_before)

    reset_outside = [i for i in control_lost]
    return {
        "cohort_size": len(ids),
        "cohort_status_counts": dict(
            sorted(collections.Counter(s["status"] for s in state.values()).items())
        ),
        "not_extracted": not_extracted,
        "still_claimed": still_claimed,
        "attempts_not_cleared": wrong_attempts,
        "missing_receipt": no_receipt,
        "receipt_mismatch": receipt_mismatch,
        "cohort_all_extracted": not not_extracted,
        "cohort_all_receipted": not no_receipt,
        "receipt_preserves_prior_state": not receipt_mismatch,
        "control_failed_before": before["control_failed_count"],
        "control_failed_after": control["count"],
        "control_lost": control_lost,
        "control_gained": control_gained,
        "reset_outside_cohort": reset_outside,
    }


def phase_manifest(args: argparse.Namespace) -> dict:
    rows = cohort()
    grouped: dict[str, list[str]] = collections.defaultdict(list)
    excluded = set(args.exclude or ())
    rows = [r for r in rows if r["id"] not in excluded]
    for row in rows:
        ids_relative = row["id"].partition(":")[2]
        # The loader flattens sources.<ids>[] to '<ids>/<path>', so the value
        # carries the path within the IDS, never the IDS name again.
        ids_name, _, path_within_ids = ids_relative.partition("/")
        grouped[ids_name].append(path_within_ids)
    manifest = {
        "kind": "sn_sources",
        "schema_version": 1,
        "name": "sib-second-pass-recoverable-failed-cohort",
        "title": "Recoverable failed DD sources, second composition pass",
        "description": (
            "The 143 sources the full-DD run left failed whose recovery class "
            "is a retry or a now-admissible binding."
        ),
        "sources": {k: sorted(v) for k, v in sorted(grouped.items())},
    }
    out = RUN_DIR / (args.manifest_name or "cohort-sources.json")
    out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return {
        "manifest_path": str(out),
        "ids_count": len(grouped),
        "source_count": sum(len(v) for v in grouped.values()),
    }


def phase_readback(args: argparse.Namespace) -> dict:
    """Per-source outcome after the rerun, names by stage, spend by run id."""
    from imas_codex.graph.client import GraphClient

    rows = cohort()
    ids = [r["id"] for r in rows]
    run_id = args.run_id
    scope_run_id = args.scope_run_id
    state = source_state(ids)
    produced: list[str] = []
    still_failed: list[dict] = []
    untouched: list[str] = []
    for source_id in sorted(ids):
        live = state.get(source_id) or {}
        bindings = live.get("bindings") or []
        if bindings:
            produced.append(source_id)
        elif live.get("status") == "failed":
            still_failed.append(
                {
                    "source_id": source_id,
                    "attempt_count": live.get("attempt_count"),
                    "last_error": live.get("last_error"),
                }
            )
        else:
            untouched.append(source_id)

    with GraphClient() as gc:
        names_by_stage = list(
            gc.query(
                """
                MATCH (s:StandardName)
                WHERE s.run_id = $scope_run_id
                RETURN coalesce(s.name_stage, '(null)') AS stage, count(s) AS n
                ORDER BY stage
                """,
                scope_run_id=scope_run_id,
            )
        )
        run_rows = list(
            gc.query(
                """
                MATCH (r:SNRun {id: $run_id})
                RETURN r.id AS id, r.status AS status,
                       r.stop_reason AS stop_reason,
                       r.cost_spent AS cost_spent,
                       r.cost_limit AS cost_limit,
                       r.cost_total AS cost_total,
                       r.names_composed AS names_composed,
                       r.names_reviewed AS names_reviewed,
                       r.started_at AS started_at,
                       r.ended_at AS ended_at
                """,
                run_id=run_id,
            )
        )
        cost_by_phase = list(
            gc.query(
                """
                MATCH (c:LLMCost)-[:FOR_RUN]->(r:SNRun {id: $run_id})
                RETURN c.phase AS phase, sum(coalesce(c.llm_cost, 0.0)) AS cost,
                       count(c) AS calls
                ORDER BY phase
                """,
                run_id=run_id,
            )
        )
    return {
        "run_id": run_id,
        "run": run_rows,
        "names_by_stage": names_by_stage,
        "scope_run_id": scope_run_id,
        "cost_by_phase": cost_by_phase,
        "cohort_size": len(ids),
        "produced_count": len(produced),
        "still_failed_count": len(still_failed),
        "untouched_count": len(untouched),
        "produced": produced,
        "still_failed": still_failed,
        "untouched": untouched,
        "status_counts": dict(
            sorted(collections.Counter(s["status"] for s in state.values()).items())
        ),
        "produced_by_status": dict(
            sorted(
                collections.Counter(
                    s["status"] for s in state.values() if s.get("bindings")
                ).items()
            )
        ),
    }


def main() -> int:
    phases = {
        "cohort": phase_cohort,
        "census": phase_census,
        "retry-dry": phase_retry_dry,
        "retry-apply": phase_retry_apply,
        "verify": phase_verify,
        "manifest": phase_manifest,
        "readback": phase_readback,
    }
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=sorted(phases))
    parser.add_argument("--out", default=None)
    parser.add_argument("--manifest-name", default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--scope-run-id", default=None)
    parser.add_argument("--exclude", action="append", default=[])
    args = parser.parse_args()
    payload = phases[args.phase](args)
    text = json.dumps(payload, indent=2, sort_keys=True, default=str)
    if args.out:
        Path(args.out).write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())