"""Apply the ruled separatrix name convention to the live catalog.

The lead ruled one suffix per data-dictionary locus: a value at the separatrix
is ``X_at_plasma_boundary`` and a separatrix average is
``flux_surface_averaged_X_at_plasma_boundary``. This runner derives that
mapping from the recorded census and applies it through the owning writers.

Locus comes from each name's DD source path, never from the census's locus
label. The census labels ``normalized_toroidal_flux_coordinate_at_plasma_boundary``
a diagnostic separatrix distance although its only path is
``summary/local/separatrix_average/position/rho_tor_norm`` -- a summary path,
and the average locus. Deriving from the path keeps one derivation for every
name and makes that row obey the same rule as its sibling.

Modes:

``map``
    Read-only. Derives the ruled target for every live name in the census,
    reads each target's live state, and writes the authority JSON.
``renames``
    Applies every ruled rename whose target spelling is free, through
    ``apply_edit`` (the sanctioned rename writer: it mints the successor, marks
    the predecessor superseded, carries the producing sources and raises if any
    source is left behind). Ordered so a spelling that must be vacated first
    is vacated before the name that needs it is renamed onto it.
``supersedes``
    Reports, per name whose ruled target is occupied, the live holder of that
    tombstone's lineage and the ruled outcome. Nothing is revived.
``readback``
    Read-only. Re-derives the population from the live graph and reports every
    name that still departs from the convention, with its reason.

The last line of the log is a bare ``EXIT=`` line.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, "/home/ITER/mcintos/Code/imas-codex")

from imas_codex.graph.client import GraphClient  # noqa: E402
from imas_codex.standard_names.edit import apply_edit  # noqa: E402

HERE = Path(__file__).resolve().parent
NODE = "sib-separatrix-suffix-apply"
CENSUS_PATH = HERE / "sib-separatrix-locus-census.json"
AUTHORITY_PATH = HERE / f"{NODE}-authority.json"

LOCATORS = (
    "_at_plasma_boundary",
    "_at_separatrix",
    "_at_last_closed_flux_surface",
    "_at_outboard_midplane",
)

REASON = (
    "the catalog spells a value at the separatrix with one suffix; the lead "
    "ruled X_at_plasma_boundary for summary/local/separatrix and "
    "summary/pedestal_fits/*/separatrix values, and "
    "flux_surface_averaged-prefixed X_at_plasma_boundary for "
    "summary/local/separatrix_average, matching the catalog majority on both "
    "axes. This name carries the other spelling for its own DD locus."
)


def locus_of(paths: list[str]) -> str:
    """Classify a name by the DD loci of its produced paths."""
    if not paths:
        return "unresolved"
    loci = set()
    for path in paths:
        if "separatrix_average" in path:
            loci.add("separatrix_average")
        elif "separatrix" in path and path.startswith("summary/pedestal_fits/"):
            loci.add("separatrix")
        elif "separatrix" in path and path.startswith("summary/local/"):
            loci.add("separatrix")
        elif "separatrix" in path:
            loci.add("diagnostic")
        elif "/boundary" in path:
            loci.add("equilibrium_boundary")
        else:
            loci.add("other")
    if len(loci) == 1:
        return loci.pop()
    return "mixed"


def strip_locator(name: str) -> tuple[str, str | None]:
    for locator in LOCATORS:
        if name.endswith(locator):
            return name[: -len(locator)], locator
    return name, None


def ruled_target(name: str, locus: str) -> str:
    """The spelling the ruling requires for ``name`` at ``locus``."""
    base, _ = strip_locator(name)
    if not base:
        return name
    if locus in ("separatrix", "equilibrium_boundary"):
        return base + "_at_plasma_boundary"
    if locus == "separatrix_average":
        stem = base.replace("_flux_surface_averaged", "")
        if stem.startswith("flux_surface_averaged_"):
            stem = stem[len("flux_surface_averaged_") :]
        return "flux_surface_averaged_" + stem + "_at_plasma_boundary"
    return name


def derive() -> list[dict[str, Any]]:
    census = json.loads(CENSUS_PATH.read_text())
    rows = []
    for row in census["names"]:
        paths = row["locus_paths"]
        locus = locus_of(paths)
        target = ruled_target(row["id"], locus)
        rows.append(
            {
                "id": row["id"],
                "unit": row["unit"],
                "paths": paths,
                "census_locus": row["locus"],
                "locus": locus,
                "census_target": row["proposed_target"],
                "target": target,
                "tombstones": row["tombstones"],
            }
        )
    return rows


def live_state(gc: GraphClient, ids: list[str]) -> dict[str, dict[str, Any]]:
    if not ids:
        return {}
    rows = gc.query(
        """
        UNWIND $ids AS id
        MATCH (sn:StandardName {id: id})
        OPTIONAL MATCH (succ:StandardName)-[:REFINED_FROM]->(sn)
        RETURN id, sn.name_stage AS stage, sn.status AS status,
               collect(DISTINCT succ.id) AS successors
        """,
        ids=sorted(set(ids)),
    )
    return {row["id"]: dict(row) for row in rows}


def build_map() -> dict[str, Any]:
    rows = derive()
    with GraphClient() as gc:
        state = live_state(gc, [r["target"] for r in rows])
    actions = Counter()
    for row in rows:
        target = row["target"]
        if target == row["id"]:
            row["action"] = "keep"
        elif target in state:
            occupant = state[target]
            row["action"] = "blocked_target_occupied"
            row["occupant"] = occupant
        else:
            row["action"] = "rename"
        row["target_state"] = state.get(target)
        actions[row["action"]] += 1
    order = {"rename": 0, "blocked_target_occupied": 1, "keep": 2}
    rows.sort(key=lambda r: (order[r["action"]], r["id"]))
    document = {
        "node": NODE,
        "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "convention": {
            "separatrix": "X_at_plasma_boundary",
            "separatrix_average": "flux_surface_averaged_X_at_plasma_boundary",
            "source": "lead ruling 2026-11-07, comment c-s49-lead-ruling-convention",
        },
        "totals": {
            "population": len(rows),
            "by_action": dict(actions),
            "by_locus": dict(Counter(r["locus"] for r in rows)),
        },
        "names": rows,
    }
    AUTHORITY_PATH.write_text(json.dumps(document, indent=1) + "\n")
    return document


def apply_renames() -> list[dict[str, Any]]:
    """Apply each rename whose target spelling is free, in a resolvable order."""
    receipts: list[dict[str, Any]] = []
    attempted: set[str] = set()
    with GraphClient() as gc:
        # A rename whose target is another renaming name's current spelling
        # becomes possible only once that spelling is vacated, so re-derive
        # the mapping after each batch rather than once for the whole run.
        while True:
            document = build_map()
            pending = [
                r
                for r in document["names"]
                if r["action"] == "rename" and r["id"] not in attempted
            ]
            if not pending:
                break
            current = live_state(gc, [r["id"] for r in pending])
            progressed = False
            for row in list(pending):
                if row["id"] not in current:
                    pending.remove(row)
                    receipts.append(
                        {"id": row["id"], "target": row["target"], "ok": False,
                         "detail": "source name no longer live"}
                    )
                    progressed = True
                    continue
                state = live_state(gc, [row["target"]])
                if row["target"] in state:
                    continue
                plan = apply_edit(
                    target=row["id"],
                    rename=row["target"],
                    reason=REASON,
                    origin="agent",
                    dry_run=False,
                    gc=gc,
                )
                receipts.append(
                    {
                        "id": row["id"],
                        "target": row["target"],
                        "ok": bool(plan.applied and not plan.blocked),
                        "detail": plan.blocked or "; ".join(plan.actions),
                        "successor": plan.successor,
                        "run_id": plan.run_id,
                    }
                )
                attempted.add(row["id"])
                pending.remove(row)
                progressed = True
            if not progressed:
                break
    return receipts


def readback() -> dict[str, Any]:
    rows = derive()
    with GraphClient() as gc:
        state = live_state(gc, [r["id"] for r in rows] + [r["target"] for r in rows])
        population = gc.query(
            """
            MATCH (sn:StandardName)
            WHERE sn.name_stage <> 'superseded'
              AND (sn.id CONTAINS 'separatrix' OR sn.id CONTAINS 'plasma_boundary')
            RETURN sn.id AS id, sn.name_stage AS stage
            ORDER BY id
            """
        )
    departures = []
    for row in rows:
        live = row["id"] in state and state[row["id"]]["stage"] != "superseded"
        target = row["target"]
        target_state = state.get(target)
        if not live:
            successor = None
            if target_state and target_state["stage"] != "superseded":
                successor = target
            departure = None
            if successor is None:
                departure = "renamed off the convention with its target occupied"
            row_out = {
                "id": row["id"],
                "locus": row["locus"],
                "target": target,
                "conforms": successor is not None,
                "departure": departure,
            }
            departures.append(row_out)
            continue
        conforms = row["id"] == target
        departures.append(
            {
                "id": row["id"],
                "locus": row["locus"],
                "target": target,
                "conforms": conforms,
                "departure": None if conforms else row.get("action"),
            }
        )
    by_locus = {}
    for row in departures:
        entry = by_locus.setdefault(row["locus"], {"live": 0, "conforming": 0})
        entry["live"] += 1
        entry["conforming"] += int(row["conforms"])
    return {
        "live_population_from_graph": len(population),
        "census_population": len(rows),
        "by_locus": by_locus,
        "non_conforming": [d for d in departures if not d["conforms"]],
        "conforming": sum(1 for d in departures if d["conforms"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["map", "renames", "readback"])
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    started = time.time()
    if args.mode == "map":
        result = build_map()
        result = {"totals": result["totals"], "authority": str(AUTHORITY_PATH)}
    elif args.mode == "renames":
        result = {"renames": apply_renames()}
    else:
        result = readback()
    result["elapsed_sha"] = hashlib.sha256(
        json.dumps(result, sort_keys=True).encode()
    ).hexdigest()[:16]
    text = json.dumps(result, indent=1)
    if args.out:
        Path(args.out).write_text(text + "\n")
    print(text)
    print(f"elapsed={time.time() - started:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())