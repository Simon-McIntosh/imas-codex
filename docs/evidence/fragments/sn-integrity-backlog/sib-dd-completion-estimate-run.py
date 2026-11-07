#!/usr/bin/env python3
"""Estimate the remaining Data Dictionary standard-name workload.

Read-only: graph reads only, no LLM call, no graph write. Reports:

1. the remaining population by source status and by name stage, plus a
   classification of the extracted DD sources into eligible leaves, plain
   structure containers, signal-STRUCTURE paths and error paths;
2. the ``sn run --dry-run`` planned-call figures parsed from its own log, and
   the population query those figures reconcile against;
3. measured per-phase cost and call rate from ``LLMCost`` records over the last
   30 days, with a 90-day window beside it, because the 30-day pipeline is a
   drain trickle whose per-name ratios are unstable;
4. a wall-time projection from recent ``SNRun`` throughput.

Estimate phases follow ``LLMCost.phase``: compose = ``generate_name``,
``generate_docs``, ``review_name``, ``review_docs``, refine =
``refine_name``/``refine_name+fanout``/``refine_docs``.

Usage::

    PYTHONPATH=. uv run --no-sync python <this-file> \
        --dry-run-log <log> --json <out.json>
"""

from __future__ import annotations

import argparse
import json
import re
import tomllib
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from imas_codex.core.node_categories import SN_SOURCE_CATEGORIES
from imas_codex.graph.client import GraphClient

PHASES = {
    "compose": ("generate_name",),
    "generate_docs": ("generate_docs",),
    "review_name": ("review_name",),
    "review_docs": ("review_docs",),
    "refine": ("refine_name", "refine_name+fanout", "refine_docs"),
}

SEAT_KEYS = {
    "compose": ("sn-compose",),
    "generate_docs": ("sn-docs",),
    "review_name": ("sn-review", "names", "profiles", "default"),
    "review_docs": ("sn-review", "docs"),
    "refine": ("sn-refine",),
}

ERROR_SUFFIXES = ("_error_upper", "_error_lower", "_error_index")

EXTRACTION_FILTER = """
MATCH (n:IMASNode)-[:IN_IDS]->(ids:IDS)
WHERE n.node_category IN $cats
  AND n.description IS NOT NULL AND n.description <> ''
  AND ids.id <> 'core_instant_changes'
  AND coalesce(n.lifecycle_status, '') <> 'removed'
  AND (
        NOT (n.data_type IN ['STRUCTURE', 'STRUCTURE_ARRAY', 'STRUCT_ARRAY'])
        OR (
             n.node_category = 'quantity'
             AND EXISTS { MATCH (d:IMASNode)-[:HAS_PARENT]->(n)
                          WHERE d.id = n.id + '/data' }
             AND NOT EXISTS { MATCH (q:IMASNode)-[:HAS_PARENT]->(n)
                              WHERE q.node_category = 'quantity' }
        )
      )
  AND NOT EXISTS {
        MATCH (s:StandardNameSource {source_id: n.id, source_type: 'dd'})
        WHERE NOT (s.status IN ['stale', 'failed', 'extracted'])
      }
RETURN count(n) AS c
"""


def rows(gc: GraphClient, cypher: str, **params: Any) -> list[dict]:
    return [dict(r) for r in gc.query(cypher, **params)]


def population(gc: GraphClient) -> dict[str, Any]:
    status_rows = rows(
        gc,
        "MATCH (s:StandardNameSource) "
        "RETURN s.source_type AS source_type, coalesce(s.status, 'none') AS status, "
        "count(*) AS c ORDER BY source_type, status",
    )
    name_rows = rows(
        gc,
        "MATCH (n:StandardName) "
        "RETURN coalesce(n.name_stage, 'none') AS stage, count(*) AS c "
        "ORDER BY c DESC",
    )
    paths = [
        r["path"]
        for r in rows(
            gc,
            "MATCH (s:StandardNameSource {status: 'extracted', source_type: 'dd'}) "
            "RETURN s.source_id AS path",
        )
    ]
    joins = {
        r["id"]: r
        for r in rows(
            gc,
            "MATCH (n:IMASNode) WHERE n.id IN $paths "
            "RETURN n.id AS id, n.data_type AS data_type",
            paths=paths,
        )
    }
    signal = {
        r["id"]
        for r in rows(
            gc,
            "MATCH (n:IMASNode) WHERE n.id IN $paths "
            "AND n.data_type IN ['STRUCTURE', 'STRUCT_ARRAY'] "
            "AND n.node_category = 'quantity' "
            "AND EXISTS { MATCH (d:IMASNode)-[:HAS_PARENT]->(n) "
            "             WHERE d.id = n.id + '/data' } "
            "AND NOT EXISTS { MATCH (q:IMASNode)-[:HAS_PARENT]->(n) "
            "                 WHERE q.node_category = 'quantity' } "
            "RETURN n.id AS id",
            paths=paths,
        )
    }
    buckets = {
        "eligible_leaf": 0,
        "signal_structure": 0,
        "structure": 0,
        "error_path": 0,
        "no_dd_node": 0,
    }
    for path in paths:
        node = joins.get(path)
        if node is None:
            buckets["no_dd_node"] += 1
        elif path.endswith(ERROR_SUFFIXES):
            buckets["error_path"] += 1
        elif path in signal:
            buckets["signal_structure"] += 1
        elif node["data_type"] in ("STRUCTURE", "STRUCT_ARRAY"):
            buckets["structure"] += 1
        else:
            buckets["eligible_leaf"] += 1
    return {
        "source_status_by_type": status_rows,
        "dd_sources_by_status": {
            r["status"]: r["c"] for r in status_rows if r["source_type"] == "dd"
        },
        "names_by_stage": {r["stage"]: r["c"] for r in name_rows},
        "extracted_dd_sources": {
            "total": len(paths),
            "classification": buckets,
        },
    }


def parse_dry_run(log_path: str | None) -> dict[str, Any] | None:
    if not log_path:
        return None
    path = Path(log_path)
    if not path.exists():
        return None
    text = path.read_text()
    match = re.search(r"Extraction candidates:\s*(\d+)", text)
    pools = {
        name: int(pending)
        for name, pending in re.findall(r"(\w+)\s*\(pending=(\d+)\)", text)
    }
    return {
        "log": str(path),
        "extraction_candidates": int(match.group(1)) if match else None,
        "pools_pending": pools,
    }


def extraction_cross_check(gc: GraphClient) -> int:
    out = rows(gc, EXTRACTION_FILTER, cats=list(SN_SOURCE_CATEGORIES))
    return int(out[0]["c"]) if out else 0


def cost_by_phase(gc: GraphClient, days: int) -> dict[str, Any]:
    per_phase: dict[str, Any] = {}
    for r in rows(
        gc,
        f"MATCH (c:LLMCost) WHERE c.llm_at >= datetime() - duration('P{days}D') "
        "RETURN c.phase AS phase, count(*) AS calls, sum(c.llm_cost) AS cost, "
        "sum(size(coalesce(c.standard_name_ids, []))) AS slots",
    ):
        slots = int(r["slots"] or 0)
        per_phase[r["phase"]] = {
            "calls": int(r["calls"]),
            "cost_usd": round(float(r["cost"] or 0.0), 6),
            "name_slots": slots,
            "cost_per_name_slot": (
                round(float(r["cost"] or 0.0) / slots, 6) if slots else None
            ),
            "distinct_names": None,
            "calls_per_name": None,
        }
    for r in rows(
        gc,
        f"MATCH (c:LLMCost) WHERE c.llm_at >= datetime() - duration('P{days}D') "
        "AND size(coalesce(c.standard_name_ids, [])) > 0 "
        "UNWIND c.standard_name_ids AS sid "
        "RETURN c.phase AS phase, count(DISTINCT sid) AS names",
    ):
        phase = per_phase.get(r["phase"])
        if phase:
            phase["distinct_names"] = int(r["names"])
            phase["calls_per_name"] = round(phase["calls"] / max(1, int(r["names"])), 3)
    return per_phase


def cost_by_model(gc: GraphClient, days: int) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for r in rows(
        gc,
        f"MATCH (c:LLMCost) WHERE c.llm_at >= datetime() - duration('P{days}D') "
        "RETURN c.phase AS phase, c.llm_model AS model, count(*) AS calls, "
        "sum(c.llm_cost) AS cost",
    ):
        calls = int(r["calls"])
        out.setdefault(r["phase"], {})[r["model"]] = {
            "calls": calls,
            "cost_usd": round(float(r["cost"] or 0.0), 6),
            "cost_per_call": round(float(r["cost"] or 0.0) / calls, 6),
        }
    return out


def throughput(gc: GraphClient, days: int) -> dict[str, Any]:
    cut = (datetime.now(UTC) - timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%S")
    runs = rows(
        gc,
        "MATCH (rr:SNRun) WHERE rr.started_at >= $cut "
        "RETURN rr.elapsed_s AS el, rr.names_composed AS nc, "
        "rr.names_reviewed AS nr",
        cut=cut,
    )
    elapsed_h = sum(float(r["el"] or 0.0) for r in runs) / 3600.0
    productive = [r for r in runs if (r["nc"] or 0) > 0 or (r["nr"] or 0) > 0]
    prod_h = sum(float(r["el"] or 0.0) for r in productive) / 3600.0
    created = rows(
        gc,
        f"MATCH (n:StandardName) WHERE n.created_at >= datetime() - duration('P{days}D') "
        "RETURN count(n) AS c",
    )[0]["c"]
    calls = rows(
        gc,
        f"MATCH (c:LLMCost) WHERE c.llm_at >= datetime() - duration('P{days}D') "
        "RETURN count(c) AS c",
    )[0]["c"]
    return {
        "days": days,
        "runs": len(runs),
        "productive_runs": len(productive),
        "elapsed_hours": round(elapsed_h, 3),
        "productive_hours": round(prod_h, 3),
        "llm_calls": int(calls),
        "calls_per_productive_hour": round(calls / prod_h, 1) if prod_h else None,
        "names_created": int(created),
        "names_created_per_productive_hour": (
            round(created / prod_h, 2) if prod_h else None
        ),
        "names_composed_counter": sum(int(r["nc"] or 0) for r in runs),
        "counter_note": (
            "SNRun.names_composed counts only counter bumps and under-reports the "
            "names created; created_at is the throughput basis"
        ),
    }


def seats_from_config(config: dict[str, Any]) -> dict[str, list[str]]:
    root = config["tool"]["imas-codex"]
    seats: dict[str, list[str]] = {}
    for phase, keys in SEAT_KEYS.items():
        section: Any = root
        for key in keys:
            section = section.get(key, {}) if isinstance(section, dict) else {}
        models: list[str] = []
        if isinstance(section, dict):
            if section.get("model"):
                models.append(section["model"])
            models.extend(section.get("models", []))
        seats[phase] = models
    return seats


def estimate(
    *,
    remaining_sources: int,
    batch_size: int,
    measured: dict[str, Any],
    model_costs: dict[str, dict[str, Any]],
    seats: dict[str, list[str]],
    yield_low: float,
    yield_high: float,
    throughputs: dict[str, Any],
) -> dict[str, Any]:
    def calls_for_names(phase: str, names: int) -> int:
        total = 0.0
        for span_phase in PHASES[phase]:
            span = measured.get(span_phase) or {}
            if span.get("calls_per_name"):
                total += span["calls_per_name"] * names
        return int(round(total))

    def spend(phase: str, calls: int, names: int) -> float:
        total = 0.0
        priced = False
        for span_phase in PHASES[phase]:
            for model in seats[phase]:
                row = (model_costs.get(span_phase) or {}).get(model)
                if row:
                    total += row["cost_per_call"] * calls
                    priced = True
        if not priced:
            for span_phase in PHASES[phase]:
                row = measured.get(span_phase) or {}
                if row.get("cost_per_name_slot") is not None:
                    total += row["cost_per_name_slot"] * names
                    priced = True
        return round(total, 4)

    arms: dict[str, Any] = {}
    for label, ratio in (("low_reuse", yield_low), ("high_yield", yield_high)):
        names = int(round(remaining_sources * ratio))
        compose_calls = (remaining_sources + batch_size - 1) // batch_size
        phases: dict[str, Any] = {
            "compose": {
                "calls": compose_calls,
                "basis": f"{remaining_sources} sources / batch size {batch_size}",
                "spend_usd": spend("compose", compose_calls, names),
            }
        }
        for phase in ("generate_docs", "review_name", "review_docs", "refine"):
            n_calls = calls_for_names(phase, names)
            phases[phase] = {
                "calls": n_calls,
                "basis": "measured 90d calls per distinct name",
                "spend_usd": spend(phase, n_calls, names),
            }
        arms[label] = {
            "names_per_source": round(ratio, 4),
            "expected_new_names": names,
            "planned_calls_total": sum(int(v["calls"] or 0) for v in phases.values()),
            "estimated_spend_usd": round(
                sum(v["spend_usd"] for v in phases.values()), 4
            ),
            "phases": phases,
        }
    wall: dict[str, Any] = {}
    for window, row in throughputs.items():
        rate = row.get("calls_per_productive_hour")
        if rate:
            wall[window] = {
                "basis": (
                    f"{rate} LLM calls per productive hour over "
                    f"{row['productive_hours']} h ({row['runs']} runs)"
                ),
                "hours_low_reuse": round(
                    arms["low_reuse"]["planned_calls_total"] / rate, 1
                ),
                "hours_high_yield": round(
                    arms["high_yield"]["planned_calls_total"] / rate, 1
                ),
            }
    return {"arms": arms, "wall_time": wall}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run-log", default=None)
    ap.add_argument("--json", default=None)
    ap.add_argument("--config", default="pyproject.toml")
    ap.add_argument("--window", type=int, default=30)
    args = ap.parse_args()

    with GraphClient() as gc:
        pop = population(gc)
        dry = parse_dry_run(args.dry_run_log)
        cross = extraction_cross_check(gc)
        measured = cost_by_phase(gc, args.window)
        measured90 = cost_by_phase(gc, 90)
        models90 = cost_by_model(gc, 90)
        t_recent = throughput(gc, args.window)
        t90 = throughput(gc, 90)
        bind = rows(
            gc,
            "MATCH (s:StandardNameSource {source_type: 'dd'})-[:PRODUCED_NAME]->"
            "(n:StandardName) WHERE s.status IN ['attached', 'composed'] "
            "RETURN count(DISTINCT n) AS names, count(DISTINCT s) AS sources",
        )[0]
        composed90 = rows(
            gc,
            "MATCH (s:StandardNameSource {source_type: 'dd'}) "
            "WHERE s.composed_at >= datetime() - duration('P90D') RETURN count(s) AS c",
        )[0]["c"]
        created90 = rows(
            gc,
            "MATCH (n:StandardName) "
            "WHERE n.created_at >= datetime() - duration('P90D') "
            "AND NOT coalesce(n.model, '') STARTS WITH 'deterministic:' "
            "AND coalesce(n.model, '') <> 'sn-edit' "
            "AND coalesce(n.origin, '') <> 'derived' "
            "RETURN count(n) AS c",
        )[0]["c"]

    yield_low = float(bind["names"]) / float(bind["sources"])
    yield_high = float(created90) / float(composed90) if composed90 else yield_low
    config = tomllib.loads(Path(args.config).read_text())
    seats = seats_from_config(config)
    batch = int(config["tool"]["imas-codex"]["sn-compose"].get("batch-size", 25))
    remaining = (
        dry["extraction_candidates"]
        if dry and dry.get("extraction_candidates")
        else cross
    )
    est = estimate(
        remaining_sources=remaining,
        batch_size=batch,
        measured=measured90,
        model_costs=models90,
        seats=seats,
        yield_low=yield_low,
        yield_high=yield_high,
        throughputs={f"{args.window}d": t_recent, "90d": t90},
    )
    record = {
        "generated_at": datetime.now(UTC).isoformat(),
        "dry_run": dry,
        "population": pop,
        "cross_check": {
            "extraction_filter_count": cross,
            "dry_run_candidates": dry.get("extraction_candidates") if dry else None,
            "delta": (
                dry["extraction_candidates"] - cross
                if dry and dry.get("extraction_candidates")
                else None
            ),
            "note": (
                "the dry run applies unit-override and qualification filters after "
                "the extraction query, so a small negative delta is expected"
            ),
        },
        f"measured_cost_{args.window}d": measured,
        "measured_cost_90d": measured90,
        "model_costs_90d": models90,
        "seats": seats,
        "throughput": {f"{args.window}d": t_recent, "90d": t90},
        "yield_basis": {
            "names_per_source_catalog": round(yield_low, 4),
            "names_per_source_90d": round(yield_high, 4),
            "catalog": (
                f"{bind['names']} names / {bind['sources']} "
                "attached-or-composed dd sources"
            ),
            "window_90d": (
                f"{created90} pipeline names created / {composed90} dd sources composed"
            ),
        },
        "estimate": est,
    }
    if args.json:
        Path(args.json).write_text(json.dumps(record, indent=2, default=str))
    print(json.dumps(record, indent=2, default=str))


if __name__ == "__main__":
    main()