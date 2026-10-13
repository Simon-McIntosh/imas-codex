"""Held-out batch acceptance of the comparator rerank, off (arm A) vs on (arm B).

Draws a held-out set of Data Dictionary sources the compose benchmark has not
seen (excluding every source on the comparator label sheet and every
boundary-density source it reserves), then composes and reviews that batch
twice through the benchmark harness with ``BenchmarkConfig.comparator_block``
on: arm A leaves the compose rerank switch
(``IMAS_CODEX_SN_COMPOSE_RERANK_COMPARATORS``) unset so the nearby-name
comparator block reaches the prompt in embedding order, arm B sets it so the
block is reranked. Compose and review run on the locally served seat in both
arms, so only arm B's rerank spends.

First-pass review acceptance is the share of composed names the reviewer scores
at or above the repository's acceptance threshold (``DEFAULT_MIN_SCORE`` = 0.85,
``imas_codex/standard_names/defaults.py``). The positive control renders the
harness compose prompt under both switch settings and requires at least one
batch to hash differently between arms; if none does, the two arms are the same
run and the driver stops. Acceptance rates, the paired difference and the
bootstrap interval (``imas_codex.discovery.base.evaluation.bootstrap_interval``)
are written to the results JSON. Read-only elsewhere: the harness composes and
scores without writing the graph.

Run::

    uv run python docs/evidence/fragments/sn-selection-over-generation/ssog-rerank-held-out-acceptance-run.py --out <results.json>
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Sequence

SEED = 20261014
N_ITEMS = 40
SWITCH_ENV = "IMAS_CODEX_SN_COMPOSE_RERANK_COMPARATORS"
COMPOSE_SEAT = "local/deepseek-v4.1-flash"
REVIEWER_SEAT = "local/deepseek-v4.1-flash"
LABEL_SHEET = Path(__file__).resolve().parent / "ssog-rerank-label-sheet.json"


# ---------------------------------------------------------------------------
# Held-out draw
# ---------------------------------------------------------------------------


def load_population(gc: Any) -> list[dict[str, Any]]:
    """Every DD source with a description and a domain, in path order."""
    rows = gc.query(
        """
        MATCH (s:StandardNameSource)
        WHERE s.source_id IS NOT NULL
        AND s.source_id CONTAINS '/'
        AND s.description IS NOT NULL AND s.description <> ''
        AND s.physics_domain IS NOT NULL
        RETURN DISTINCT s.source_id AS path,
               s.description AS description,
               s.physics_domain AS domain
        ORDER BY path
        """
    )
    return [dict(r) for r in rows]


def load_boundary_paths(gc: Any, suffix: str) -> list[str]:
    """DD sources that produced a boundary-density standard name."""
    rows = gc.query(
        """
        MATCH (s:StandardNameSource)-[:PRODUCED_NAME]->(n:StandardName)
        WHERE n.id ENDS WITH $suffix
        RETURN DISTINCT s.source_id AS path
        ORDER BY path
        """,
        suffix=suffix,
    )
    return [r["path"] for r in rows]


def draw_held_out(
    population: Sequence[dict[str, Any]],
    excluded: set[str],
    *,
    seed: int = SEED,
    count: int = N_ITEMS,
) -> list[dict[str, Any]]:
    """Draw *count* sources outside *excluded*, seeded, spread across domains."""
    eligible = [r for r in population if r["path"] not in excluded]
    by_domain: dict[str, list[dict[str, Any]]] = {}
    for row in eligible:
        by_domain.setdefault(row["domain"], []).append(row)
    rng = random.Random(seed)
    domains = sorted(by_domain)
    for domain in domains:
        rng.shuffle(by_domain[domain])
    pointers = {d: 0 for d in domains}
    chosen: list[dict[str, Any]] = []
    while len(chosen) < count:
        progressed = False
        for domain in domains:
            if len(chosen) >= count:
                break
            pool = by_domain[domain]
            if pointers[domain] >= len(pool):
                continue
            chosen.append(pool[pointers[domain]])
            pointers[domain] += 1
            progressed = True
        if not progressed:
            break
    return chosen


def label_sheet_paths() -> list[str]:
    sheet = json.loads(LABEL_SHEET.read_text())
    return [it["path"] for it in sheet["items"]]


# ---------------------------------------------------------------------------
# Positive control
# ---------------------------------------------------------------------------


def render_harness_messages(
    batch: dict[str, Any], context: dict, nearby: list | None
) -> list[dict]:
    from imas_codex.llm.prompt_loader import render_prompt

    items = batch.get("items", [])
    existing = set(batch.get("existing_names", []))
    user_context = {
        "items": items,
        "ids_name": batch.get("group_key", "unknown"),
        "existing_names": sorted(existing)[:200],
        "cluster_context": batch.get("context", ""),
    }
    if nearby is not None:
        user_context["nearby_existing_names"] = nearby
    user_prompt = render_prompt("sn/generate_name_dd", {**context, **user_context})
    system_prompt = render_prompt(
        "sn/generate_name_system", {**context, "compose_scored_examples": []}
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


async def positive_control(batches: list[dict], context: dict) -> dict[str, Any]:
    """Render each batch's harness prompt off and on; require a differing hash."""
    from imas_codex.standard_names.benchmark import _rendered_messages_hash
    from imas_codex.standard_names.workers import (
        _collect_nearby_name_comparators,
        _rerank_comparators_for_compose,
    )

    rows: list[dict[str, Any]] = []
    any_diff = False
    total_rerank_cost = 0.0
    for batch in batches:
        if not batch.get("items"):
            continue
        os.environ.pop(SWITCH_ENV, None)
        nearby_off = await asyncio.to_thread(
            _collect_nearby_name_comparators, batch["items"]
        )
        off_msgs = render_harness_messages(batch, context, nearby_off)
        off_hash = _rendered_messages_hash(off_msgs)

        os.environ[SWITCH_ENV] = "1"
        nearby_on, cost, seat = await _rerank_comparators_for_compose(
            batch["items"], nearby_off
        )
        total_rerank_cost += cost
        on_msgs = render_harness_messages(batch, context, nearby_on)
        on_hash = _rendered_messages_hash(on_msgs)
        os.environ.pop(SWITCH_ENV, None)

        differs = off_hash != on_hash
        any_diff = any_diff or differs
        rows.append(
            {
                "group_key": batch.get("group_key"),
                "n_items": len(batch["items"]),
                "n_comparators": len(nearby_off),
                "hash_off": off_hash,
                "hash_on": on_hash,
                "differs": differs,
                "rerank_cost_usd": round(cost, 6),
                "rerank_seat": seat,
            }
        )
    return {
        "batches": rows,
        "n_batches": len(rows),
        "switch_changes_harness_compose_prompt": any_diff,
        "rerank_cost_usd": round(total_rerank_cost, 6),
    }


# ---------------------------------------------------------------------------
# Arms
# ---------------------------------------------------------------------------


async def run_arm(
    batches: list[dict],
    context: dict,
    source_paths: list[str],
    switch: str | None,
    out_path: Path,
) -> dict[str, Any]:
    from imas_codex.standard_names.benchmark import BenchmarkConfig, run_benchmark

    if switch is None:
        os.environ.pop(SWITCH_ENV, None)
    else:
        os.environ[SWITCH_ENV] = switch

    config = BenchmarkConfig(
        models=[COMPOSE_SEAT],
        reviewer_models=[REVIEWER_SEAT],
        names_only=True,
        temperature=0.0,
        max_candidates=200,
        comparator_block=True,
        source_paths=list(source_paths),
        output_path=str(out_path),
    )
    report = await run_benchmark(config, extraction_batches=batches)
    os.environ.pop(SWITCH_ENV, None)
    return report


def _acceptance_rows(report: Any) -> dict[str, Any]:
    """Per-candidate reviewer score per arm.

    Returns ``{"records": [...], "cost": float}`` where each record carries the
    composed name, its reviewer score and the acceptance flag at the repository
    threshold.
    """
    from imas_codex.standard_names.defaults import DEFAULT_MIN_SCORE

    records: list[dict[str, Any]] = []
    cost = 0.0
    for res in report.results:
        review = next(
            (r for r in getattr(res, "name_reviewer_results", []) if not r.error),
            None,
        )
        if review is None:
            continue
        cost += float(getattr(res, "total_cost", 0.0)) + float(review.cost)
        by_name = {s.get("name"): s for s in review.scores}
        for cand in res.candidates:
            name = cand.get("standard_name") or cand.get("name")
            scored = by_name.get(name)
            score = float(scored.get("score", 0.0)) if scored else 0.0
            records.append(
                {
                    "source_id": cand.get("source_id", ""),
                    "name": name,
                    "reviewer_score": score,
                    "accepted": score >= DEFAULT_MIN_SCORE,
                }
            )
    return {"records": records, "cost": round(cost, 6), "threshold": DEFAULT_MIN_SCORE}


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def acceptance_interval(
    accept_a: Sequence[float], accept_b: Sequence[float]
) -> dict[str, Any] | None:
    """Paired bootstrap interval from the evaluation module, when defined.

    The module's ``bootstrap_interval`` resamples units and reports the paired
    difference of an AUC over the same labels. Both arms' per-item acceptance
    (1 accepted, 0 not) are the scores and item membership is the unit; a
    population carrying a single label class has no defined AUC and is reported
    as ``None`` rather than raising.
    """
    from imas_codex.discovery.base.evaluation import bootstrap_interval

    labels = [1.0 if a or b else 0.0 for a, b in zip(accept_a, accept_b, strict=True)]
    interval = bootstrap_interval(
        labels, [float(b) for b in accept_b], scores_b=[float(a) for a in accept_a], seed=123
    )
    return interval


def summarise(rows_a: dict[str, Any], rows_b: dict[str, Any]) -> dict[str, Any]:
    a = rows_a["records"]
    b = rows_b["records"]
    by_a = {r["source_id"]: r for r in a}
    by_b = {r["source_id"]: r for r in b}
    common = sorted(set(by_a) & set(by_b))
    acc_a = [by_a[s]["accepted"] for s in common]
    acc_b = [by_b[s]["accepted"] for s in common]
    rate_a = sum(acc_a) / len(common) if common else 0.0
    rate_b = sum(acc_b) / len(common) if common else 0.0

    interval = None
    err = ""
    try:
        interval = acceptance_interval(acc_a, acc_b)
    except ValueError as exc:
        err = str(exc)

    return {
        "n_items_common": len(common),
        "acceptance_rate_arm_a": rate_a,
        "acceptance_rate_arm_b": rate_b,
        "paired_difference_b_minus_a": rate_b - rate_a,
        "accepted_arm_a": sum(acc_a),
        "accepted_arm_b": sum(acc_b),
        "bootstrap_interval": interval,
        "bootstrap_error": err,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


async def _amain(args: argparse.Namespace) -> int:
    from imas_codex.graph.client import GraphClient
    from imas_codex.standard_names.context import build_compose_context

    out = Path(args.out)
    context = build_compose_context()

    # 1. Draw held-out sources.
    with GraphClient() as gc:
        population = load_population(gc)
        boundary = load_boundary_paths(gc, args.boundary_suffix)
    sheet = label_sheet_paths()
    excluded = set(sheet) | set(boundary)
    drawn = draw_held_out(population, excluded, seed=args.seed, count=args.items)
    source_paths = [r["path"] for r in drawn]
    print(f"population={len(population)} excluded={len(excluded)} drawn={len(drawn)}")

    if args.draw_only:
        print(json.dumps({"source_paths": source_paths}, indent=2))
        return 0

    # 2. Extract once; both arms and the control share the same batches.
    from imas_codex.standard_names.benchmark import BenchmarkConfig, _extract_candidates

    extract_cfg = BenchmarkConfig(
        models=[COMPOSE_SEAT],
        max_candidates=200,
        source_paths=list(source_paths),
    )
    batches = _extract_candidates(extract_cfg)
    n_items = sum(len(b.get("items", [])) for b in batches)

    # 3. Positive control.
    control = await positive_control(batches, context)
    if not control["switch_changes_harness_compose_prompt"]:
        result = {
            "status": "blocked",
            "reason": "positive control failed: switch leaves every batch hash identical",
            "source_paths": source_paths,
            "positive_control": control,
        }
        out.write_text(json.dumps(result, indent=2) + "\n")
        print("POSITIVE CONTROL FAILED — wrote", out)
        return 3

    # 4. Both arms.
    report_a = await run_arm(batches, context, source_paths, None, out.with_suffix(".armA.json"))
    report_b = await run_arm(batches, context, source_paths, "1", out.with_suffix(".armB.json"))
    rows_a = _acceptance_rows(report_a)
    rows_b = _acceptance_rows(report_b)
    summary = summarise(rows_a, rows_b)

    spend = round(control["rerank_cost_usd"] + rows_a["cost"] + rows_b["cost"], 6)
    result = {
        "status": "ok",
        "seed": args.seed,
        "n_source_paths": len(source_paths),
        "n_extracted_items": n_items,
        "source_paths": source_paths,
        "drawn": drawn,
        "compose_seat": COMPOSE_SEAT,
        "reviewer_seat": REVIEWER_SEAT,
        "switch_env": SWITCH_ENV,
        "positive_control": control,
        "arm_a": {"records": rows_a["records"]},
        "arm_b": {"records": rows_b["records"]},
        "summary": summary,
        "spend_usd": spend,
    }
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"summary": summary, "spend_usd": spend}, indent=2))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "ssog-rerank-held-out-acceptance-results.json"))
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--items", type=int, default=N_ITEMS)
    ap.add_argument("--boundary-suffix", default="_density_at_plasma_boundary")
    ap.add_argument("--draw-only", action="store_true")
    args = ap.parse_args(argv)
    return asyncio.run(_amain(args))


if __name__ == "__main__":
    sys.exit(main())