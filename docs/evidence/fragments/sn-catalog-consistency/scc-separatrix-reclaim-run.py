"""Reclaim the four tombstoned boundary spellings, then re-read the census.

The reclaim runs through the signed rename writer only: this script calls
``edit.apply_edit``, never Cypher. Three modes:

``--probe``            the writer's dry run for each of the four names
``--apply --run-id``   the signed reclaim, then a read-back of every receipt
``(default)``          the separatrix census re-read after the applies

The population rule, the locus classifier and the conforming classifier are
the census instrument's own — imported from the sibling census script so both
records count the same population — while the four names are named here
explicitly, because a census cannot tell a lead-ruled reclaim from a collision.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from imas_codex.graph.client import GraphClient
from imas_codex.standard_names import edit

HERE = Path(__file__).resolve().parent
NODE = "scc-separatrix-reclaim"
JSON_PATH = HERE / f"{NODE}.json"

#: The four names the lead ruled on 2026-11-12: live value name, tombstoned
#: spelling it takes.
CASES = [
    (
        "normalized_toroidal_flux_coordinate_at_separatrix",
        "normalized_toroidal_flux_coordinate_at_plasma_boundary",
    ),
    (
        "toroidal_lithium_velocity_at_separatrix",
        "toroidal_lithium_velocity_at_plasma_boundary",
    ),
    (
        "toroidal_total_plasma_momentum_at_separatrix",
        "toroidal_total_plasma_momentum_at_plasma_boundary",
    ),
    (
        "tungsten_density_at_separatrix",
        "tungsten_density_at_plasma_boundary",
    ),
]

REASON = (
    "the boundary locus is spelled at_plasma_boundary under the lead's "
    "ruling; the spelling's former holder's history moves to an archived "
    "identity so it travels with the quantity it records"
)


def census_module():
    """The sibling census instrument, loaded by path (its name has no module form)."""
    path = HERE / "scc-separatrix-remainder-run.py"
    spec = importlib.util.spec_from_file_location("scc_separatrix_census", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def revision() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def readback(gc, name_id: str) -> dict:
    rows = gc.query(
        """
        MATCH (sn:StandardName {id: $id})
        OPTIONAL MATCH (sn)-[:HAS_REVIEW]->(review:StandardNameReview)
        OPTIONAL MATCH (source:StandardNameSource)-[:PRODUCED_NAME]->(sn)
        RETURN sn.name_stage AS name_stage, sn.status AS status,
               sn.unit AS unit, sn.docs_stage AS docs_stage,
               count(DISTINCT review) AS reviews,
               [s IN collect(DISTINCT source) WHERE s IS NOT NULL
                 | s.source_id] AS sources,
               EXISTS {
                 MATCH path = (sn)-[:REFINED_FROM*2..]->(sn) RETURN path LIMIT 1
               } AS closes_cycle
        """,
        id=name_id,
    )
    return dict(rows[0]) if rows else {}


def archive_readback(gc, tombstone_id: str) -> dict:
    rows = gc.query(
        """
        MATCH (archived:ArchivedStandardName {archived_from: $id})
        OPTIONAL MATCH (holder:StandardName)-[:HAS_ARCHIVED_HISTORY]->(archived)
        OPTIONAL MATCH (archived)-[:HAS_REVIEW]->(review:StandardNameReview)
        RETURN archived.id AS id, archived.name_stage AS name_stage,
               count(DISTINCT review) AS reviews,
               [h IN collect(DISTINCT holder) WHERE h IS NOT NULL | h.id] AS holders
        """,
        id=tombstone_id,
    )
    return dict(rows[0]) if rows else {}


def probe(gc) -> list[dict]:
    out = []
    for target, tombstoned in CASES:
        plan = edit.apply_edit(
            target=target,
            rename=tombstoned,
            reason=REASON,
            origin="agent",
            scope="only_self",
            dry_run=True,
            gc=gc,
        )
        out.append(
            {
                "id": target,
                "target": tombstoned,
                "applied": plan.applied,
                "blocked": plan.blocked,
                "actions": plan.actions,
            }
        )
    return out


def applied_already(gc, target: str, tombstoned: str) -> bool:
    """The reclaim already ran: the spelling is live and its holder retired."""
    retired = readback(gc, target)
    live = readback(gc, tombstoned)
    return retired.get("name_stage") == "superseded" and live.get("name_stage") not in (
        None,
        "superseded",
    )


def apply(gc, run_id: str) -> list[dict]:
    results = []
    for target, tombstoned in CASES:
        if applied_already(gc, target, tombstoned):
            record = {
                "id": target,
                "target": tombstoned,
                "applied": True,
                "blocked": None,
                "run_id": None,
                "actions": [],
                "already_applied": True,
                "live": readback(gc, tombstoned),
                "archive": archive_readback(gc, tombstoned),
                "retired": readback(gc, target),
            }
        else:
            try:
                plan = edit.apply_edit(
                    target=target,
                    rename=tombstoned,
                    reason=REASON,
                    origin="agent",
                    scope="only_self",
                    gc=gc,
                )
                record = {
                    "id": target,
                    "target": tombstoned,
                    "applied": plan.applied,
                    "blocked": plan.blocked,
                    "run_id": plan.run_id,
                    "actions": plan.actions,
                }
                if plan.applied:
                    record["live"] = readback(gc, tombstoned)
                    record["archive"] = archive_readback(gc, tombstoned)
                    record["retired"] = readback(gc, target)
            except Exception as error:  # one case's refusal must not hide another's
                record = {
                    "id": target,
                    "target": tombstoned,
                    "applied": False,
                    "blocked": f"{type(error).__name__}: {error}",
                    "actions": [],
                }
        results.append(record)
        print("CASE " + json.dumps(record, default=str), flush=True)
    return results


def census(gc) -> dict:
    module = census_module()
    names = gc.query(
        """
        MATCH (sn:StandardName)
        WHERE sn.status <> 'superseded'
        OPTIONAL MATCH (source:StandardNameSource)-[:PRODUCED_NAME]->(sn)
        OPTIONAL MATCH (imas:IMASNode)-[:HAS_STANDARD_NAME]->(sn)
        WITH sn, collect(DISTINCT source.dd_path) AS sp,
             collect(DISTINCT imas.id) AS ip
        RETURN sn.id AS id, sp + ip + coalesce(sn.source_paths, []) AS paths
        """
    )
    # Population membership is the census instrument's own rule: a name that
    # carries a boundary locator, or that a separatrix DD path realizes.
    population = 0
    in_population = set()
    for row in names:
        seen = set()
        for raw in row["paths"] or []:
            if isinstance(raw, str):
                seen.add(str(raw).removeprefix("dd:").strip("/"))
        named = row["id"].endswith(("_at_separatrix", "_at_plasma_boundary"))
        if named or any(module.is_separatrix_path(path) for path in seen):
            population += 1
            in_population.add(row["id"])
    staged = gc.query(
        """
        MATCH (sn:StandardName)
        WHERE coalesce(sn.name_stage, '') = 'superseded'
          AND coalesce(sn.status, '') <> 'superseded'
        RETURN count(sn) AS n
        """
    )[0]["n"]
    out_of_line = module.out_of_line(gc)
    out_ids = {row["id"] for row in out_of_line}
    conforming = len(in_population - out_ids)
    return {
        "live_names_in_graph": len(names),
        "superseded_stage_with_live_status": staged,
        "population": population,
        "out_of_line": out_of_line,
        "out_of_line_count": len(out_ids),
        "conforming_count": conforming,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(REASON)
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args(argv)
    if args.probe and args.apply:
        parser.error("choose one mode")
    run_id = args.run_id or "sn-edit-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    payload: dict = {
        "node": NODE,
        "revision": revision(),
        "observed_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "cases": CASES,
        "reason": REASON,
        "run_id": run_id,
    }
    with GraphClient() as gc:
        if args.apply:
            payload["applied"] = apply(gc, run_id)
        if args.probe:
            payload["probe"] = probe(gc)
        if not args.probe and not args.apply:
            payload["census"] = census(gc)
    if JSON_PATH.exists():
        previous = json.loads(JSON_PATH.read_text())
        payload = {**previous, **payload}
    JSON_PATH.write_text(json.dumps(payload, indent=1, default=str) + "\n")
    print("RESULT " + json.dumps(payload, default=str)[:6000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
