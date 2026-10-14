"""Held-out batch acceptance of the comparator rerank, off (arm A) vs on (arm B).

This is the merged held-out acceptance driver
(``ssog-rerank-held-out-acceptance-run.py``) with one addition: each unit's
result is written to the results JSON as soon as it completes, and a later run
skips units already recorded. The measure composes and reviews a held-out batch
twice through the harness with ``BenchmarkConfig.comparator_block`` on: arm A
leaves the compose rerank switch
(``IMAS_CODEX_SN_COMPOSE_RERANK_COMPARATORS``) unset so the nearby-name
comparator block reaches the prompt in embedding order, arm B sets it so the
block is reranked. Compose and review run on the locally served seat in both
arms, so only arm B's rerank spends.

A pass may take minutes per unit and a shell tool call cannot outlast a bounded
wall clock, so the run is repeated as repeated foreground passes under
``timeout 540``. The persistence is what makes that possible: every completed
control batch, arm-A source and arm-B source is durable in the results JSON, so
the next pass picks up at the first unrecorded unit and never recomposes a
recorded one.

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

    uv run python docs/evidence/fragments/sn-selection-over-generation/ssog-rerank-acceptance-rerun-run.py --out <results.json>
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
import tempfile
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


async def positive_control_batch(batch: dict, context: dict) -> dict[str, Any]:
    """Render one batch's harness prompt off and on; report whether hashes differ."""
    from imas_codex.standard_names.benchmark import _rendered_messages_hash
    from imas_codex.standard_names.workers import (
        _collect_nearby_name_comparators,
        _rerank_comparators_for_compose,
    )

    if not batch.get("items"):
        return {}
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
    on_msgs = render_harness_messages(batch, context, nearby_on)
    on_hash = _rendered_messages_hash(on_msgs)
    os.environ.pop(SWITCH_ENV, None)

    return {
        "group_key": batch.get("group_key"),
        "n_items": len(batch["items"]),
        "n_comparators": len(nearby_off),
        "hash_off": off_hash,
        "hash_on": on_hash,
        "differs": off_hash != on_hash,
        "rerank_cost_usd": round(cost, 6),
        "rerank_seat": seat,
    }


def control_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "batches": rows,
        "n_batches": len(rows),
        "switch_changes_harness_compose_prompt": any(r.get("differs") for r in rows),
        "rerank_cost_usd": round(sum(r.get("rerank_cost_usd", 0.0) for r in rows), 6),
    }


# ---------------------------------------------------------------------------
# Arms
# ---------------------------------------------------------------------------


def _set_switch(switch: str | None) -> None:
    if switch is None:
        os.environ.pop(SWITCH_ENV, None)
    else:
        os.environ[SWITCH_ENV] = switch


async def run_arm_batch(
    batch: dict, source_paths: list[str], switch: str | None, out_path: Path
) -> list[dict[str, Any]]:
    """Compose and review one batch through the harness under one switch setting."""
    from imas_codex.standard_names.benchmark import BenchmarkConfig, run_benchmark

    _set_switch(switch)
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
    try:
        report = await run_benchmark(config, extraction_batches=[batch])
    finally:
        _set_switch(None)
    rows = _acceptance_rows(report)
    return rows["records"], rows["cost"]


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
# Persistence
# ---------------------------------------------------------------------------


def _load_results(out: Path) -> dict[str, Any]:
    if out.exists():
        try:
            return json.loads(out.read_text())
        except json.JSONDecodeError:
            return {}
    return {}


def _write_results(out: Path, state: dict[str, Any]) -> None:
    """Atomically persist the full run state after every completed unit."""
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(json.dumps(state, indent=2) + "\n")
    tmp.replace(out)


def _recorded_sources(records: Sequence[dict[str, Any]]) -> set[str]:
    return {r["source_id"] for r in records if r.get("source_id")}


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

    # The harness writes an incremental report per arm; keep it in scratch so
    # the repository holds only the measure's own results JSON.
    scratch = Path(tempfile.mkdtemp(prefix="ssog-rerank-arm-"))
    state = _load_results(out)
    state.update(
        {
            "seed": args.seed,
            "n_source_paths": len(source_paths),
            "n_extracted_items": n_items,
            "n_batches": len(batches),
            "source_paths": source_paths,
            "drawn": drawn,
            "compose_seat": COMPOSE_SEAT,
            "reviewer_seat": REVIEWER_SEAT,
            "switch_env": SWITCH_ENV,
        }
    )
    state.setdefault("positive_control", {"batches": [],
                                          "switch_changes_harness_compose_prompt": False,
                                          "rerank_cost_usd": 0.0})
    state.setdefault("arm_a", {"records": [], "cost_usd": 0.0})
    state.setdefault("arm_b", {"records": [], "cost_usd": 0.0})
    state["status"] = "in-progress"
    _write_results(out, state)
    print(f"batches={len(batches)} items={n_items}")

    # 3. Positive control, per batch, persisted as it completes.
    done_keys = {r.get("group_key") for r in state["positive_control"]["batches"]}
    for batch in batches:
        if not batch.get("items") or batch.get("group_key") in done_keys:
            continue
        row = await positive_control_batch(batch, context)
        if not row:
            continue
        state["positive_control"]["batches"].append(row)
        state["positive_control"]["n_batches"] = len(state["positive_control"]["batches"])
        state["positive_control"]["switch_changes_harness_compose_prompt"] = any(
            r.get("differs") for r in state["positive_control"]["batches"]
        )
        state["positive_control"]["rerank_cost_usd"] = round(
            sum(r.get("rerank_cost_usd", 0.0)
                for r in state["positive_control"]["batches"]), 6)
        _write_results(out, state)
        print(f"control {row['group_key']} differs={row['differs']}")

    if not state["positive_control"]["switch_changes_harness_compose_prompt"]:
        state["status"] = "blocked"
        state["reason"] = (
            "positive control failed: switch leaves every batch hash identical"
        )
        _write_results(out, state)
        print("POSITIVE CONTROL FAILED — wrote", out)
        return 3

    # 4. Both arms, per batch, persisted as each batch completes; a batch whose
    #    sources are all recorded is skipped, so a later pass never recomposes
    #    a recorded source.
    for arm_key, switch in (("arm_a", None), ("arm_b", "1")):
        # A source is done once it has been composed in this arm, whether or not
        # it yielded a scoring candidate: a batch the model answers with only
        # vocabulary gaps leaves no record, and without the attempted set it
        # would be recomposed on every pass and never advance.
        done = _recorded_sources(state[arm_key]["records"]) | set(
            state[arm_key].get("sources_done", [])
        )
        for batch in batches:
            batch_sources = {
                it.get("source_id", it.get("path", "")) for it in batch.get("items", [])
            }
            if not batch_sources or batch_sources <= done:
                continue
            records, cost = await run_arm_batch(
                batch, source_paths, switch, scratch / f"{arm_key}.json"
            )
            state[arm_key]["records"].extend(records)
            state[arm_key]["cost_usd"] = round(
                state[arm_key]["cost_usd"] + cost, 6)
            done |= _recorded_sources(records) | batch_sources
            state[arm_key]["sources_done"] = sorted(done)
            _write_results(out, state)
            print(f"{arm_key} {batch.get('group_key')} records={len(records)}")

    state["summary"] = summarise(state["arm_a"], state["arm_b"])
    state["spend_usd"] = round(
        state["positive_control"]["rerank_cost_usd"]
        + state["arm_a"]["cost_usd"]
        + state["arm_b"]["cost_usd"],
        6,
    )
    state["status"] = "ok"
    _write_results(out, state)
    print(json.dumps({"summary": state["summary"],
                      "spend_usd": state["spend_usd"]}, indent=2))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "ssog-rerank-acceptance-rerun-results.json"))
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--items", type=int, default=N_ITEMS)
    ap.add_argument("--boundary-suffix", default="_density_at_plasma_boundary")
    ap.add_argument("--draw-only", action="store_true")
    args = ap.parse_args(argv)
    return asyncio.run(_amain(args))


if __name__ == "__main__":
    sys.exit(main())