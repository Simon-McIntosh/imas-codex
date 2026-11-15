"""Check the refine-pass evidence against the live graph, read-only.

The evidence fragment states what the pass reached; this gate re-derivable-facts
that statement so a drifted graph or an edited JSON fails loudly instead of
being quoted onward.

What it re-reads from the graph:

* every name's live identity, by the same lineage rule the report uses — the
  deepest identity accepted on both axes, else the deepest accepted one, else
  the refinement dead-end;
* that identity's stage and documentation score;
* the live-names-without-a-source count, from the repository's own instrument;
* that the recorded spend stays inside the ceiling.

It never writes: every call is a ``MATCH ... RETURN``, and ``--dry-run`` prints
the checks it would make.

``--inject-accepted`` is the red-arm control: it adds a count the graph does not
hold, so the gate must fail. A gate that only ever runs green proves nothing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from imas_codex.graph.client import GraphClient
from imas_codex.standard_names.source_reads import count_live_names_without_source

HERE = Path(__file__).resolve().parent
EVIDENCE = HERE / "held-names-refine-pass.json"

LINEAGE = """
MATCH path = (s:StandardName)-[:REFINED_FROM*0..6]->(o:StandardName {id:$n})
RETURN s.id AS id, s.name_stage AS ns, s.docs_stage AS ds,
       min(length(path)) AS depth
"""

SUCCESSORS = """
MATCH (t:StandardName)-[:REFINED_FROM]->(p:StandardName) WHERE p.id IN $ids
RETURN p.id AS id, count(t) AS n
"""

STAGE = """
MATCH (s:StandardName {id:$id})
RETURN s.name_stage AS ns, s.docs_stage AS ds,
       s.reviewer_score_name AS scn, s.reviewer_score_docs AS scd
"""


def pick_live(candidates: list[dict], root: str) -> dict:
    if not candidates:
        return {"id": root, "ns": None, "ds": None, "depth": 0, "out_deg": 0}

    def deepest(rows: list[dict]) -> dict:
        return max(rows, key=lambda r: (r["depth"], -r["out_deg"]))

    published = [
        c for c in candidates if c["ns"] == "accepted" and c["ds"] == "accepted"
    ]
    if published:
        return deepest(published)
    named = [c for c in candidates if c["ns"] == "accepted"]
    if named:
        return deepest(named)
    stopped = [c for c in candidates if c["out_deg"] == 0 and c["ns"] != "superseded"]
    if stopped:
        return deepest(stopped)
    return next((c for c in candidates if c["id"] == root), candidates[0])


def resolve_all(gc, names: list[str]) -> dict[str, str]:
    rows: dict[str, list[dict]] = {}
    for name in names:
        rows[name] = gc.query(LINEAGE, n=name)
    ids = sorted({r["id"] for rs in rows.values() for r in rs})
    degrees = {r["id"]: r["n"] for r in gc.query(SUCCESSORS, ids=ids)}
    for rs in rows.values():
        for r in rs:
            r["out_deg"] = degrees.get(r["id"], 0)
    return {name: pick_live(rs, name)["id"] for name, rs in rows.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, default=EVIDENCE)
    parser.add_argument(
        "--inject-accepted",
        type=int,
        default=0,
        help="add this many to the accepted count the gate expects, to prove the "
        "gate fails against a count the graph does not hold",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    doc = json.loads(args.evidence.read_text())
    names = [row["name"] for row in doc["names"]]
    # Two counts live in the artifact and they answer different questions: the
    # outcome column counts the names this pass settled (`accepted`), while a row
    # can read accepted on both axes without that outcome — a grouping parent
    # whose live identity is already published, or a name an earlier slice
    # settled. The graph check covers the second, the arithmetic check the first.
    expected_accepted = sum(
        1
        for row in doc["names"]
        if (row["after"] or {}).get("name_stage") == "accepted"
        and (row["after"] or {}).get("docs_stage") == "accepted"
    )
    expected_accepted += args.inject_accepted
    settled_outcome = sum(1 for row in doc["names"] if row["outcome"] == "accepted")

    if args.dry_run:
        print(f"{len(names)} names, {expected_accepted} accepted to check")
        return 0

    failures: list[str] = []
    with GraphClient() as gc:
        live = resolve_all(gc, names)
        for row in doc["names"]:
            want = row["live_id"]
            got = live[row["name"]]
            if got != want:
                failures.append(
                    f"{row['name']}: live identity {got}, evidence says {want}"
                )
        stages = {name: gc.query(STAGE, id=live[name])[0] for name in names}
        accepted = sum(
            1
            for name in names
            if stages[name]["ns"] == "accepted" and stages[name]["ds"] == "accepted"
        )
        if accepted != expected_accepted:
            failures.append(f"accepted: graph {accepted}, evidence {expected_accepted}")
        if doc["outcomes"]["counts"].get("accepted") != settled_outcome:
            failures.append(
                f"accepted outcomes: rows {settled_outcome}, "
                f"summary {doc['outcomes']['counts'].get('accepted')}"
            )
        for row in doc["names"]:
            after = row["after"] or {}
            stage = stages[row["name"]]
            if (after.get("name_stage"), after.get("docs_stage")) != (
                stage["ns"],
                stage["ds"],
            ):
                failures.append(
                    f"{row['name']}: stage {stage['ns']}/{stage['ds']}, "
                    f"evidence {after.get('name_stage')}/{after.get('docs_stage')}"
                )
            recorded = after.get("score_docs")
            actual = stage["scd"]
            if (
                recorded is not None
                and actual is not None
                and abs(recorded - actual) > 1e-6
            ):
                failures.append(
                    f"{row['name']}: docs score {actual:.6f}, evidence {recorded:.6f}"
                )
        sourceless = count_live_names_without_source(gc)["count"]
        if sourceless != doc["sourceless"]["now"]:
            failures.append(
                f"sourceless: graph {sourceless}, evidence {doc['sourceless']['now']}"
            )
        if doc["spend"]["total_usd"] > doc["spend"]["ceiling_usd"]:
            failures.append("spend exceeds the ceiling")

    for line in failures:
        print(f"FAIL {line}")
    print(f"checked {len(names)} names, {accepted} accepted, {sourceless} sourceless")
    code = 1 if failures else 0
    print(f"EXIT={code}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
