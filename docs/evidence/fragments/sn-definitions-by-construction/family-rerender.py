"""Re-render the live species-family members from their stored sentences.

Every live standard name the stored family sentences match is an edit target:
its documentation is reset and regenerated through the pipeline's own write
path, so the writer installs the family sentence with only the species
substituted. Each member's before and after state, its documentation score and
the spend of its own run are appended to a progress file, one line when the
edit is submitted and one when it settles, so an interrupted pass resumes from
the members that have not settled.

The live graph is read directly and written only through ``sn edit``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import traceback
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

PROGRESS = Path(
    os.path.expanduser(
        "~/.cache/imas-codex-live-rewrite/family-rerender-progress.jsonl"
    )  # noqa: E501
)
LOGDIR = Path(
    os.path.expanduser("~/.cache/imas-codex-live-rewrite/family-rerender-logs")
)

HINT = (
    "Open this definition with its stored species-family sentence, written "
    "verbatim: every member of the family states that one wording, with only "
    "the species changing. Do not restate, paraphrase or duplicate it, and do "
    "not write a second sentence about the population it names."
)
REASON = (
    "This standard name belongs to a species family whose shared sentence is "
    "stored once for the catalog, as the operator wordings and the species "
    "population statement are. Every member of the family must state that one "
    "sentence with only the species changing; the name's Data Dictionary source "
    "is the quantity the family sentence describes, and where the family counts "
    "a species number density the ruled population is ions summed over all "
    "charge states with neutral atoms excluded."
)

STATE_QUERY = """
MATCH (sn:StandardName {id: $id})
OPTIONAL MATCH (sn)-[:HAS_UNIT]->(u:Unit)
WITH sn, collect(DISTINCT u.id) AS unit_ids
WITH sn, unit_ids,
     [src IN [(src:StandardNameSource)-[:PRODUCED_NAME]->(sn) | src.id]
        WHERE src IS NOT NULL AND src <> ''] AS sources
RETURN sn.id AS id,
       coalesce(sn.description, '') AS description,
       coalesce(sn.documentation, '') AS documentation,
       coalesce(unit_ids[0], sn.unit, '') AS unit,
       coalesce(sn.name_stage, '') AS name_stage,
       coalesce(sn.docs_stage,'') AS _unused,
       coalesce(sn.docs_stage, '') AS docs_stage,
       sn.reviewer_score_docs AS score_docs,
       coalesce(sn.edit_status, '') AS edit_status,
       coalesce(sn.run_id, '') AS run_id,
       sources
"""

SPEND_QUERY = """
MATCH (c:LLMCost {run_id: $run_id})
RETURN coalesce(sum(c.llm_cost), 0.0) AS spend, count(c) AS events,
       collect(DISTINCT c.phase) AS phases
"""

SOURCELESS_QUERY = """
MATCH (sn:StandardName)
WHERE coalesce(sn.name_stage, '') <> 'superseded'
  AND coalesce(sn.unit, '') <> ''
WITH sn, size([(src:StandardNameSource)-[:PRODUCED_NAME]->(sn) | 1]) AS edges
WHERE edges = 0
RETURN count(sn) AS n
"""

RUN_ID_RE = re.compile(r"scope run_id=([A-Za-z0-9_\-]+)")


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def read_members() -> dict[str, list[dict]]:
    from imas_codex.graph.client import GraphClient
    from imas_codex.standard_names.docs_gates import species_family

    query = """
    MATCH (sn:StandardName)
    WHERE coalesce(sn.name_stage, '') <> 'superseded'
    OPTIONAL MATCH (sn)-[:HAS_UNIT]->(u:Unit)
    WITH sn, collect(DISTINCT u.id) AS unit_ids
    RETURN sn.id AS id, coalesce(unit_ids[0], sn.unit, '') AS unit
    ORDER BY sn.id
    """
    with GraphClient() as gc:
        rows = [dict(r) for r in gc.query(query)]
    families: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        found = species_family(row["id"])
        if found:
            families[found[0]].append(row)
    return dict(families)


def read_state(gc, name: str) -> dict:
    rows = gc.query(STATE_QUERY, id=name)
    row = dict(rows[0]) if rows else {}
    row.pop("_unused", None)
    return row


def read_spend(gc, run_id: str) -> dict:
    if not run_id:
        return {"spend": 0.0, "events": 0, "phases": []}
    rows = gc.query(SPEND_QUERY, run_id=run_id)
    row = dict(rows[0]) if rows else {}
    return {
        "spend": round(float(row.get("spend") or 0.0), 6),
        "events": int(row.get("events") or 0),
        "phases": sorted(row.get("phases") or []),
    }


def sourceless_count(gc) -> int:
    rows = gc.query(SOURCELESS_QUERY)
    return int(rows[0]["n"]) if rows else 0


def settled_names() -> set[str]:
    settled = set()
    if not PROGRESS.exists():
        return settled
    for line in PROGRESS.read_text().splitlines():
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if entry.get("stage") == "settled":
            settled.add(entry["name"])
    return settled


def append(entry: dict) -> None:
    PROGRESS.parent.mkdir(parents=True, exist_ok=True)
    with PROGRESS.open("a") as handle:
        handle.write(json.dumps(entry, sort_keys=True) + "\n")


def edit_command(name: str, cost_limit: float) -> list[str]:
    return [
        sys.executable,
        "-c",
        "from imas_codex.cli import main; main()",
        "sn",
        "edit",
        name,
        "--hint",
        HINT,
        "--reason",
        REASON,
        "--axis",
        "docs",
        "--cost-limit",
        f"{cost_limit:.2f}",
    ]


def run_member(gc, name: str, family: str, cost_limit: float) -> dict:
    LOGDIR.mkdir(parents=True, exist_ok=True)
    before = read_state(gc, name)
    submitted = {
        "stage": "submitted",
        "at": now(),
        "name": name,
        "family": family,
        "before": before,
    }
    append(submitted)
    log = LOGDIR / f"{name}.log"
    started = time.time()
    proc = subprocess.run(
        edit_command(name, cost_limit), capture_output=True, text=True
    )
    elapsed = round(time.time() - started, 1)
    log.write_text(
        "# command: " + " ".join(edit_command(name, cost_limit)) + "\n"
        "# exit: "
        + str(proc.returncode)
        + "\n\n"
        + proc.stdout
        + "\n--- stderr ---\n"
        + proc.stderr
    )
    run_id = ""
    match = RUN_ID_RE.search(proc.stdout)
    if match:
        run_id = match.group(1)
    after = read_state(gc, name)
    spend = read_spend(gc, run_id or after.get("run_id", ""))
    score_before = before.get("score_docs")
    score_after = after.get("score_docs")
    settled = {
        "stage": "settled",
        "at": now(),
        "name": name,
        "family": family,
        "run_id": run_id,
        "exit": proc.returncode,
        "elapsed_s": elapsed,
        "outcome": (
            "accepted"
            if after.get("docs_stage") == "accepted"
            else after.get("docs_stage") or "unknown"
        ),
        "edit_status": after.get("edit_status", ""),
        "before": before,
        "after": after,
        "score_before": score_before,
        "score_after": score_after,
        "spend": spend,
        "log": str(log),
    }
    append(settled)
    return settled


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="list families")
    parser.add_argument(
        "--family", action="append", default=[], help="family key to run"
    )
    parser.add_argument("--name", action="append", default=[], help="one member to run")
    parser.add_argument(
        "--limit", type=int, default=0, help="stop after this many members"
    )
    parser.add_argument("--cost-limit", type=float, default=0.25)
    parser.add_argument("--spend-ceiling", type=float, default=30.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--settled-spend", action="store_true", help="print the spend recorded so far"
    )
    args = parser.parse_args(argv)

    from imas_codex.graph.client import GraphClient

    families = read_members()
    if args.list:
        for key in sorted(families, key=lambda k: (-len(families[k]), k)):
            print(f"{len(families[key]):3d}  {key}")
        print(
            f"families {len(families)} members {sum(len(v) for v in families.values())}"
        )
        return 0

    if args.settled_spend:
        total = 0.0
        if PROGRESS.exists():
            for line in PROGRESS.read_text().splitlines():
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if entry.get("stage") == "settled":
                    total += float(entry.get("spend", {}).get("spend") or 0.0)
        print(f"settled spend ${total:.4f}")
        return 0

    targets: list[tuple[str, str]] = []
    if args.name:
        for name in args.name:
            for key, members in families.items():
                if any(member["id"] == name for member in members):
                    targets.append((key, name))
                    break
            else:
                raise SystemExit(f"not a stored family member: {name}")
    for key in args.family:
        if key not in families:
            raise SystemExit(f"no such family: {key}")
        targets.extend((key, member["id"]) for member in families[key])

    if not targets:
        raise SystemExit("no target selected")

    if args.dry_run:
        for _key, name in targets:
            proc = subprocess.run(
                edit_command(name, args.cost_limit) + ["--dry-run"],
                capture_output=True,
                text=True,
            )
            print(f"{name}: exit={proc.returncode}")
            print(proc.stdout[-2000:])
            if proc.stderr:
                print(proc.stderr[-1000:])
        return 0

    settled = settled_names()
    total_spend = 0.0
    if PROGRESS.exists():
        for line in PROGRESS.read_text().splitlines():
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if entry.get("stage") == "settled":
                total_spend += float(entry.get("spend", {}).get("spend") or 0.0)
    for _key, name in targets:
        if name in settled:
            continue
        with GraphClient() as gc:
            if total_spend >= args.spend_ceiling:
                print(f"spend ceiling reached: ${total_spend:.4f}")
                return 2
            try:
                result = run_member(gc, name, _key, args.cost_limit)
            except Exception:
                append(
                    {
                        "stage": "error",
                        "at": now(),
                        "name": name,
                        "traceback": traceback.format_exc(),
                    }
                )
                raise
            total_spend += float(result["spend"]["spend"] or 0.0)
            print(
                f"{name}  {result['outcome']}  "
                f"score {result['score_before']} -> {result['score_after']}  "
                f"${result['spend']['spend']:.4f}  {result['elapsed_s']}s",
                flush=True,
            )
            if args.limit and len(settled) >= args.limit:
                break
    print(f"spend so far ${total_spend:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
