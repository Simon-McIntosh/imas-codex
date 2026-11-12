"""Supersede the losing momentum-diffusivity tip and measure the fold.

Applies the lead's ruling of 2026-11-12 through the signed supersede path:
toroidal_momentum_diffusion_coefficient is folded into
toroidal_plasma_momentum_diffusivity after a preview, and the survivor is
read back against the receipt the writer emits.

The lineage census is the committed instrument under the integrity-backlog
evidence directory, imported here rather than re-derived. Its classify
function also drives a two-tip lineage planted in a disposable graph, so
the one-tip reading this node reports is shown to be able to fail.

Phases run in order inside one job: --pre, --preview, --apply, --post,
--census, --control.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from imas_codex.graph.client import GraphClient  # noqa: E402

STEM = Path(__file__).stem.removesuffix("-run")

CENSUS = (
    ROOT
    / "docs/evidence/fragments/sn-integrity-backlog/sib-lineage-cycle-census-run.py"
)
TIP = ROOT / "docs/evidence/fragments/sn-catalog-consistency/scc-momentum-tip-run.py"

SURVIVOR = "toroidal_plasma_momentum_diffusivity"
LOSER = "toroidal_momentum_diffusion_coefficient"
NAMES = [SURVIVOR, LOSER]
CARRIED_DD_PATH = "core_transport/model/profiles_1d/momentum_phi/d"
STALE = "toroidal_total_plasma_momentum_diffusivity"
UNBLOCK_REASON = (
    "this tombstone's own retirement row records that the census found no "
    "supersession event for it, and the edge points at a generalized-grid "
    "momentum diffusion coefficient while the tombstone's meaning, total "
    "plasma momentum diffusivity, is carried by "
    "toroidal_plasma_momentum_diffusivity; the unbacked claim blocks the fold "
    "of the coefficient into that survivor"
)


def load_script(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read(cypher: str, **params):
    with GraphClient() as graph:
        return [dict(row) for row in graph.query(cypher, **params)]


STATE = """
MATCH (n:StandardName) WHERE n.id IN $ids
RETURN n.id AS id, n.name_stage AS stage, n.status AS status,
  n.superseded_from_stage AS superseded_from_stage, n.unit AS unit,
  n.description AS description, n.documentation AS documentation,
  n.docs_stage AS docs_stage, n.source_paths AS source_paths,
  n.reviewer_score_name AS name_score, n.review_count AS review_count,
  n.review_mean_score AS review_mean_score, n.llm_cost AS llm_cost,
  n.updated_at AS updated_at
ORDER BY id
"""

SOURCES = """
MATCH (s:StandardNameSource)-[:PRODUCED_NAME]->(n:StandardName)
WHERE n.id IN $ids
RETURN n.id AS name, s.id AS source, s.source_id AS path,
  s.source_type AS source_type,
  [(s)-[:FROM_DD_PATH]->(dd:IMASNode) | dd.id] AS dd_paths
ORDER BY name, path
"""

REVIEWS = """
MATCH (n:StandardName)-[:HAS_REVIEW]->(r:StandardNameReview)
WHERE n.id IN $ids
RETURN n.id AS name, r.id AS id, r.review_axis AS axis, r.tier AS tier,
  r.model AS model, r.score AS score, r.reviewed_at AS reviewed_at
ORDER BY name, r.reviewed_at, r.id
"""

EDGES = """
MATCH (a:StandardName)-[e:REFINED_FROM]->(b:StandardName)
WHERE a.id IN $ids OR b.id IN $ids
RETURN a.id AS successor, b.id AS predecessor,
  properties(e) AS properties, elementId(e) AS edge_id
ORDER BY successor, predecessor
"""

CHANGES = """
MATCH (c:StandardNameChange)
WHERE c.from_name IN $ids OR c.to_name IN $ids
RETURN c.id AS id, c.operation AS operation, c.from_name AS from_name,
  c.to_name AS to_name, c.reason AS reason, c.changed_at AS changed_at,
  c.receipt AS receipt
ORDER BY c.changed_at, c.id
"""

LINEAGE = """
MATCH (n:StandardName) WHERE n.id IN $ids
OPTIONAL MATCH (up:StandardName)-[:REFINED_FROM*0..]->(n)
WITH collect(DISTINCT up) AS ups UNWIND ups AS u
OPTIONAL MATCH (u)-[:REFINED_FROM*0..]->(d:StandardName)
WITH collect(DISTINCT d) AS ds UNWIND ds AS d
RETURN DISTINCT d.id AS id ORDER BY id
"""


def snapshot() -> dict:
    state = read(STATE, ids=NAMES)
    if len(state) != 2:
        raise RuntimeError(f"expected both names, read {len(state)}")
    return {
        "observed_at": datetime.now(UTC).isoformat(),
        "nodes": state,
        "sources": read(SOURCES, ids=NAMES),
        "reviews": read(REVIEWS, ids=NAMES),
        "edges": read(EDGES, ids=NAMES),
        "changes": read(CHANGES, ids=NAMES),
        "lineage_members": [row["id"] for row in read(LINEAGE, ids=NAMES)],
    }


def supersede(dry_run: bool) -> dict:
    argv = ["sn", "supersede", LOSER, "--into", SURVIVOR]
    if dry_run:
        argv.append("--dry-run")
    return signed(argv)


def drop_stale_successor(dry_run: bool) -> dict:
    argv = ["sn", "remove-lineage", STALE, LOSER, "--reason", UNBLOCK_REASON]
    if dry_run:
        argv.append("--dry-run")
    return signed(argv)


def signed(argv: list[str]) -> dict:
    command = [str(Path(sys.executable).parent / "imas-codex"), *argv]
    done = subprocess.run(command, capture_output=True, text=True, cwd=ROOT)
    print(done.stdout, end="", flush=True)
    if done.stderr:
        print(done.stderr, end="", file=sys.stderr, flush=True)
    if done.returncode != 0:
        raise RuntimeError(f"{' '.join(argv)} exited {done.returncode}")
    return {
        "command": " ".join(command),
        "returncode": done.returncode,
        "stdout": done.stdout,
        "stderr": done.stderr,
    }


def checks(before: dict, after: dict) -> dict:
    nodes = {row["id"]: row for row in after["nodes"]}
    loser, survivor = nodes[LOSER], nodes[SURVIVOR]
    fold_edges = [
        edge
        for edge in after["edges"]
        if edge["successor"] == SURVIVOR and edge["predecessor"] == LOSER
    ]
    loser_edges = [edge for edge in after["edges"] if edge["predecessor"] == LOSER]
    loser_edges_before = [
        edge for edge in before["edges"] if edge["predecessor"] == LOSER
    ]
    survivor_sources = [s for s in after["sources"] if s["name"] == SURVIVOR]
    carried = [s for s in survivor_sources if CARRIED_DD_PATH in (s["dd_paths"] or [])]
    stale_after = [e for e in after["edges"] if e["successor"] == STALE]
    loser_reviews_before = {r["id"] for r in before["reviews"] if r["name"] == LOSER}
    loser_reviews_after = {r["id"] for r in after["reviews"] if r["name"] == LOSER}
    result = {
        "loser_stage": loser["stage"],
        "loser_status": loser["status"],
        "loser_superseded_from_stage": loser["superseded_from_stage"],
        "survivor_stage": survivor["stage"],
        "fold_edges": fold_edges,
        "loser_incoming_successors": loser_edges,
        "loser_incoming_successors_before": loser_edges_before,
        "survivor_dd_paths": sorted(
            path for s in survivor_sources for path in (s["dd_paths"] or [])
        ),
        "carried_source_rows": len(carried),
        "stale_edges_after": stale_after,
        "loser_reviews_before": len(loser_reviews_before),
        "loser_reviews_after": len(loser_reviews_after),
        "loser_reviews_kept": loser_reviews_before == loser_reviews_after,
        "loser_sources_after": [
            s["source"] for s in after["sources"] if s["name"] == LOSER
        ],
        "changes_after": len(after["changes"]),
        "changes_before": len(before["changes"]),
    }
    assert loser["stage"] == "superseded", result
    assert loser["superseded_from_stage"] == "accepted", result
    assert len(fold_edges) == 1, result
    assert fold_edges[0]["properties"].get("writer") == "edit.supersede_into", result
    assert len(loser_edges) == len(loser_edges_before) + 1, result
    assert stale_after == [], result
    assert carried, result
    assert result["loser_sources_after"] == [], result
    assert result["loser_reviews_kept"], result
    assert result["loser_reviews_after"] >= 3, result
    return result


def census(run_dir: Path, tag: str) -> dict:
    module = load_script(CENSUS, f"lineage_census_{tag}")
    tip = load_script(TIP, "momentum_tip")
    target = run_dir / f"census-{tag}"
    target.mkdir(parents=True, exist_ok=True)
    module.capture(target)
    reading = tip.census_readings(target / "graph-snapshot.json")
    reading["snapshot"] = str(target / "graph-snapshot.json")
    return reading


SEALED = (
    "receipt_type",
    "schema_version",
    "mechanism",
    "run_id",
    "change_id",
    "old_id",
    "into_id",
    "predecessor_stage",
    "source_count",
    "source_ids",
    "projection_count",
    "backing_ids",
)
PROFILE = (
    "stage",
    "unit",
    "name_score",
    "review_count",
    "review_mean_score",
    "llm_cost",
    "description",
)


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def node_of(state: dict, name: str) -> dict:
    return next(row for row in state["nodes"] if row["id"] == name)


def rows_of(state: dict, key: str, name: str) -> list:
    return [row for row in state[key] if row["name"] == name]


def write_arm(record: dict) -> dict:
    return {
        "command": record["command"],
        "returncode": record["returncode"],
        "stdout": record["stdout"].strip(),
    }


def census_arm(reading: dict) -> dict:
    lineage = reading["lineage"]
    return {
        "observed_at": reading["observed_at"],
        "revision": reading["revision"],
        "names": reading["inventory"]["names"],
        "edges": reading["inventory"]["edges"],
        "lineage_members": lineage["members"],
        "live_tips": lineage["live_tips"],
        "lineage_live_tip_count": reading["lineage_live_tip_count"],
        "cycle_components": reading["cycle_components"],
        "multi_tip_components": reading["multi_tip_components"],
        "tip_cross_check_agrees": reading["tip_cross_check_agrees"],
        "snapshot": reading["snapshot"],
    }


def assemble(run_dir: Path) -> dict:
    pre, post = load(run_dir / "pre-state.json"), load(run_dir / "post-state.json")
    checks = load(run_dir / "checks.json")
    control = load(run_dir / "disposable-control.json")
    fold = next(row for row in post["changes"] if row["operation"] == "fold_identity")
    removal = next(
        row
        for row in post["changes"]
        if row["operation"] == "remove_refined_from_relationship"
    )
    receipt = json.loads(fold["reason"])
    before_read = load(run_dir / "census-before.json")
    after_read = load(run_dir / "census-after.json")
    return {
        "node": STEM,
        "observed_at": after_read["observed_at"],
        "revision": after_read["revision"],
        "spend_usd": 0.0,
        "spend_note": (
            "no model call: the fold, the lineage removal and both census "
            "reads are deterministic graph operations"
        ),
        "survivor": SURVIVOR,
        "loser": LOSER,
        "before": {
            "observed_at": pre["observed_at"],
            "survivor": {k: node_of(pre, SURVIVOR)[k] for k in PROFILE},
            "loser": {k: node_of(pre, LOSER)[k] for k in PROFILE},
            "survivor_sources": rows_of(pre, "sources", SURVIVOR),
            "loser_sources": rows_of(pre, "sources", LOSER),
            "survivor_reviews": len(rows_of(pre, "reviews", SURVIVOR)),
            "loser_reviews": len(rows_of(pre, "reviews", LOSER)),
            "edges": pre["edges"],
            "lineage_members": pre["lineage_members"],
        },
        "signed_writes": {
            "lineage_removal_preview": write_arm(
                load(run_dir / "unblock-preview.json")
            ),
            "lineage_removal": dict(
                write_arm(load(run_dir / "unblock.json")),
                change_id=removal["id"],
                changed_at=removal["changed_at"],
                ledger_from=removal["from_name"],
                ledger_to=removal["to_name"],
                reason=removal["reason"],
            ),
            "supersede_preview": write_arm(load(run_dir / "preview.json")),
            "supersede": dict(
                write_arm(load(run_dir / "apply.json")),
                change_id=fold["id"],
                changed_at=fold["changed_at"],
            ),
        },
        "receipt": {
            "sealed": {key: receipt[key] for key in SEALED},
            "sha256_of_receipt_text": hashlib.sha256(
                fold["reason"].encode()
            ).hexdigest(),
        },
        "after": checks,
        "census": {
            "instrument": str(CENSUS.relative_to(ROOT)),
            "before": census_arm(before_read),
            "after": census_arm(after_read),
        },
        "disposable_control": {
            "fixture": control["fixture"],
            "two_tip": control["arms"]["two_tip"],
            "two_tip_plus_cycle": control["arms"]["two_tip_plus_cycle"],
            "one_tip": control["arms"]["one_tip"],
            "assertions": control["assertions"],
        },
    }


def dump(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    print(f"WROTE {path} bytes={path.stat().st_size}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--phase", action="append", required=True)
    args = parser.parse_args()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    for phase in args.phase:
        if phase == "pre":
            dump(args.run_dir / "pre-state.json", snapshot())
        elif phase == "preview":
            dump(args.run_dir / "preview.json", supersede(dry_run=True))
        elif phase == "unblock-preview":
            dump(
                args.run_dir / "unblock-preview.json",
                drop_stale_successor(dry_run=True),
            )
        elif phase == "unblock":
            dump(args.run_dir / "unblock.json", drop_stale_successor(dry_run=False))
        elif phase == "apply":
            dump(args.run_dir / "apply.json", supersede(dry_run=False))
        elif phase == "baseline":
            dump(args.run_dir / "baseline-state.json", snapshot())
        elif phase == "post":
            after = snapshot()
            baseline = args.run_dir / "baseline-state.json"
            if not baseline.exists():
                baseline = args.run_dir / "pre-state.json"
            dump(args.run_dir / "post-state.json", after)
            dump(
                args.run_dir / "checks.json",
                checks(json.loads(baseline.read_text()), after),
            )
        elif phase == "census-before":
            dump(args.run_dir / "census-before.json", census(args.run_dir, "before"))
        elif phase == "census-after":
            dump(args.run_dir / "census-after.json", census(args.run_dir, "after"))
        elif phase == "assemble":
            dump(
                ROOT
                / "docs/evidence/fragments/sn-catalog-consistency"
                / f"{STEM}.json",
                assemble(args.run_dir),
            )
        elif phase == "control":
            tip = load_script(TIP, "momentum_tip_control")
            module = load_script(CENSUS, "lineage_census_control")
            dump(
                args.run_dir / "disposable-control.json",
                tip.disposable_control(module.classify),
            )
        else:
            raise SystemExit(f"unknown phase {phase!r}")
    return 0


if __name__ == "__main__":
    print(
        "# revision: "
        + subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
        ).strip()
        + f" tree: {ROOT} command: {' '.join(sys.argv)}",
        flush=True,
    )
    print(f"# cwd: {Path.cwd().resolve()}", flush=True)
    try:
        code = main()
    except Exception:
        import traceback

        traceback.print_exc()
        code = 1
    print(f"EXIT={code}", flush=True)
