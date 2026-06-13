"""Matched A/B measurement of the free local self-refine compose pass.

Runs two ``sn run --focus`` cohorts over the SAME reference DD paths — one
with self-refine OFF (current behaviour), one with it ON — then reads from
the graph, per arm:

* paid refine-cycle count (``refine_name`` + ``refine_docs`` LLMCost rows),
* final accept rate (StandardName.name_stage == 'accepted'),
* mean reviewer_score_name,
* total PAID cost (sum of LLMCost.llm_cost for the arm's SNRun).

Self-refine itself runs on the free local GPU model, so it contributes $0;
the comparison isolates *downstream paid* spend (review + refine quorum).

Per-arm cleanup: each arm resets its focus sources to ``extracted`` before
running (scoped to the focus paths only), so the comparison starts from the
same clean state and never pollutes the broad graph.

Cost attribution is by ``SNRun.id`` (LLMCost.run_id == SNRun.id). The focus
StandardName nodes carry the ``scope_run_id`` printed by the seed step, used
for accept-rate / score scoping.

Usage::

    uv run --no-sync python scripts/measure_self_refine.py \
        --cap 8 --cost-limit 2.0
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from datetime import UTC, datetime

from imas_codex.graph.client import GraphClient

# 12 reference DD paths: a mix of historically-refine-prone species-unresolved
# distributions densities, the dimensionless q-family, and the compound-unit
# power_density family across several physics domains.
REFERENCE_PATHS = [
    "distributions/distribution/profiles_1d/co_passing/density",
    "distributions/distribution/profiles_1d/counter_passing/density",
    "distributions/distribution/profiles_1d/trapped/density",
    "distributions/distribution/profiles_1d/density",
    "core_profiles/profiles_1d/q",
    "edge_profiles/profiles_1d/q",
    "equilibrium/time_slice/profiles_1d/q",
    "equilibrium/time_slice/global_quantities/q_95",
    "bolometer/power_density",
    "soft_x_rays/channel/power_density",
    "disruption/profiles_1d/power_density_conductive_losses",
    "disruption/profiles_1d/power_density_radiative_losses",
]

_SEED_RE = re.compile(r"run_id=([0-9a-f]{8})")


def _reset_focus_sources(paths: list[str]) -> None:
    """Hard-reset the focus paths' SN + source nodes to a clean 'extracted'.

    Scoped to exactly the focus paths — deletes their StandardName subgraph
    and resets their StandardNameSource so the next arm composes fresh.
    """
    sns_ids = [f"dd:{p}" for p in paths]
    with GraphClient() as gc:
        # Detach-delete the StandardName nodes produced for these sources
        # (chain history included — this is a measurement scratch set).
        gc.query(
            """
            UNWIND $ids AS sid
            MATCH (s:StandardNameSource {id: sid})
            OPTIONAL MATCH (s)-[:PRODUCED_NAME]->(sn:StandardName)
            OPTIONAL MATCH (sn)-[:REFINED_FROM*0..]->(anc:StandardName)
            DETACH DELETE sn, anc
            """,
            ids=sns_ids,
        )
        gc.query(
            """
            UNWIND $ids AS sid
            MATCH (s:StandardNameSource {id: sid})
            SET s.status = 'extracted',
                s.claimed_at = null,
                s.claim_token = null,
                s.run_id = null,
                s.skipped_at = null
            """,
            ids=sns_ids,
        )


def _run_arm(
    name: str, paths: list[str], cost_limit: float, time_limit: float, self_refine: bool
) -> str:
    """Run one focus arm; return the scope_run_id (8-char prefix) it seeded.

    The focus path set IS the cohort (one StandardNameSource per path); the
    six-pool loop drains it to terminal state. ``--cost-limit`` bounds paid
    spend and ``--time`` bounds wall-clock as a safety stop.
    """
    import os

    env = dict(os.environ)
    env["IMAS_CODEX_SN_COMPOSE_SELF_REFINE"] = "true" if self_refine else "false"

    cmd = [
        "uv",
        "run",
        "--no-sync",
        "imas-codex",
        "sn",
        "run",
        "--cost-limit",
        str(cost_limit),
        "--time",
        str(time_limit),
        "--quiet",
    ]
    for p in paths:
        cmd += ["--focus", p]

    print(f"\n=== ARM {name} (self-refine={self_refine}) ===")
    print(" ".join(cmd[:9]), "... +", len(paths), "focus paths")
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    out = proc.stdout + "\n" + proc.stderr
    m = _SEED_RE.search(out)
    scope = m.group(1) if m else ""
    print(f"   exit={proc.returncode} scope_run_id={scope or '?'}")
    if proc.returncode != 0:
        print("   STDERR tail:")
        print("\n".join(proc.stderr.splitlines()[-15:]))
    return scope


def _latest_snrun_since(t_iso: str) -> str | None:
    """The newest SNRun started at or after t_iso (the arm's BudgetManager run)."""
    with GraphClient() as gc:
        rows = gc.query(
            """
            MATCH (r:SNRun)
            WHERE r.started_at >= $t
            RETURN r.id AS id ORDER BY r.started_at DESC LIMIT 1
            """,
            t=t_iso,
        )
    return rows[0]["id"] if rows else None


def _measure(run_id: str | None, scope_prefix: str, paths: list[str]) -> dict:
    """Read per-arm metrics from the graph."""
    sns_ids = [f"dd:{p}" for p in paths]
    out: dict = {"run_id": run_id, "scope": scope_prefix}
    with GraphClient() as gc:
        if run_id:
            cost_rows = gc.query(
                """
                MATCH (c:LLMCost {run_id: $rid})
                RETURN coalesce(c.phase, c.pool, 'unknown') AS phase,
                       count(c) AS calls,
                       sum(c.llm_cost) AS usd
                """,
                rid=run_id,
            )
            phases = {r["phase"]: (r["calls"], r["usd"] or 0.0) for r in cost_rows}
            refine_calls = sum(
                phases.get(p, (0, 0.0))[0]
                for p in phases
                if p and p.startswith("refine_")
            )
            total_usd = sum(v[1] for v in phases.values())
            out["paid_refine_cycles"] = refine_calls
            out["paid_total_usd"] = round(total_usd, 4)
            out["phase_breakdown"] = {
                k: {"calls": v[0], "usd": round(v[1], 4)} for k, v in phases.items()
            }
        else:
            out["paid_refine_cycles"] = None
            out["paid_total_usd"] = None
            out["phase_breakdown"] = {}

        # Accept-rate + mean score over the focus SN nodes (scope_run_id is the
        # FULL uuid; we match SN nodes produced by these exact sources instead,
        # which is unambiguous for a focus run).
        sn_rows = gc.query(
            """
            UNWIND $ids AS sid
            MATCH (s:StandardNameSource {id: sid})-[:PRODUCED_NAME]->(sn:StandardName)
            RETURN sn.id AS id, sn.name_stage AS ns,
                   sn.reviewer_score_name AS rsn
            """,
            ids=sns_ids,
        )
        n = len(sn_rows)
        accepted = sum(1 for r in sn_rows if r["ns"] == "accepted")
        scores = [r["rsn"] for r in sn_rows if r["rsn"] is not None]
        out["sn_count"] = n
        out["accepted"] = accepted
        out["accept_rate"] = round(accepted / n, 3) if n else None
        out["mean_reviewer_score_name"] = (
            round(sum(scores) / len(scores), 4) if scores else None
        )
        out["names"] = [
            {"id": r["id"], "stage": r["ns"], "score": r["rsn"]} for r in sn_rows
        ]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--cost-limit", type=float, default=2.0, help="paid budget per arm (USD)"
    )
    ap.add_argument(
        "--time",
        type=float,
        default=15.0,
        help="wall-clock safety cap per arm (minutes)",
    )
    ap.add_argument("--out", default="self_refine_measurement.json")
    args = ap.parse_args()

    paths = REFERENCE_PATHS
    results: dict = {
        "paths": paths,
        "cost_limit": args.cost_limit,
        "time_limit": args.time,
        "arms": {},
    }

    for arm_name, self_refine in (("OFF", False), ("ON", True)):
        _reset_focus_sources(paths)
        time.sleep(1)
        t0 = datetime.now(UTC).isoformat()
        scope = _run_arm(arm_name, paths, args.cost_limit, args.time, self_refine)
        # allow async budget writer to flush
        time.sleep(8)
        run_id = _latest_snrun_since(t0)
        metrics = _measure(run_id, scope, paths)
        results["arms"][arm_name] = metrics
        print(
            f"   metrics: {json.dumps({k: v for k, v in metrics.items() if k not in ('names', 'phase_breakdown')})}"
        )

    # ── Verdict ────────────────────────────────────────────────────────
    off = results["arms"]["OFF"]
    on = results["arms"]["ON"]
    print("\n" + "=" * 64)
    print("MEASUREMENT SUMMARY (OFF vs ON)")
    print("=" * 64)
    hdr = f"{'metric':32} {'OFF':>14} {'ON':>14}"
    print(hdr)
    for key, label in [
        ("paid_refine_cycles", "paid refine cycles"),
        ("accept_rate", "accept rate"),
        ("mean_reviewer_score_name", "mean score (name)"),
        ("paid_total_usd", "paid total $"),
        ("sn_count", "SN count"),
    ]:
        print(f"{label:32} {str(off.get(key)):>14} {str(on.get(key)):>14}")

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
