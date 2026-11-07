"""Preview, apply and audit the exact recorded separatrix naming cohort.

The authority fixes identities and source paths. Mutations use the owning
rename writer; every apply is preceded by a persisted dry-run and followed by
a receipt read keyed to the writer's returned run id.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from imas_codex.graph.client import GraphClient
from imas_codex.standard_names import edit

HERE = Path(__file__).resolve().parent
AUTHORITY = HERE / "sib-separatrix-reclaim-apply-authority.json"
PRIOR = HERE / "sib-separatrix-suffix-apply-authority.json"
RUN = Path(
    "/home/ITER/mcintos/.config/reckon/crew/runs/r-20261107T185046746594-sib-separatrix-reclaim-apply"
)
BARE = "toroidal_tritium_velocity"
REASON = (
    "Use X_at_plasma_boundary for a local separatrix value and "
    "flux_surface_averaged_X_at_plasma_boundary for its surface average. "
    "Reclaim only when superseded intermediates lead to this identity, "
    "or the holder has a different DD locus from the tombstone and this identity."
)


def write(path, data):
    path.write_text(json.dumps(data, indent=2, default=str) + "\n")


def snapshot(gc, ids):
    rows = gc.query(
        """
      UNWIND $ids AS id MATCH (n:StandardName {id:id})
      OPTIONAL MATCH (s:StandardNameSource)-[:PRODUCED_NAME]->(n)
      OPTIONAL MATCH (n)-[:HAS_REVIEW]->(v:StandardNameReview)
      RETURN id, n.name_stage AS stage, n.status AS status,
        n.source_paths AS paths, n.description AS description,
        n.reviewer_score_name AS score, n.unit AS unit, n.run_id AS run_id,
        collect(DISTINCT s.id) AS sources, collect(DISTINCT v.id) AS reviews
      ORDER BY id
    """,
        ids=sorted(set(ids)),
    )
    return {r["id"]: dict(r) for r in rows}


def lineage(gc, name):
    return gc.query(
        """
      MATCH (root:StandardName {id:$id})
      OPTIONAL MATCH p=(root)<-[:REFINED_FROM*1..16]-(tip:StandardName)
      WHERE tip.name_stage <> 'superseded'
        AND all(n IN nodes(p)[0..-1] WHERE n.name_stage='superseded')
      RETURN root.id AS root, root.source_paths AS root_paths,
        [n IN nodes(p) | {id:n.id,stage:n.name_stage,paths:n.source_paths}] AS chain
      LIMIT 100
    """,
        id=name,
    )


def prepare():
    original = json.loads(PRIOR.read_text())["names"]
    assert len(original) == 98
    with GraphClient() as gc:
        state = snapshot(
            gc,
            [n for r in original for n in (r["id"], r["target"])]
            + [BARE + "_at_pedestal_top"],
        )
        assert len(state) > 98 and "effective_charge_at_separatrix" in state
        rows = []
        for r in original:
            if r["id"] == r["target"] or state[r["id"]]["stage"] == "superseded":
                continue
            row = {k: r[k] for k in ("id", "target", "paths", "locus")}
            row["before"] = state[r["id"]]
            row["target_before"] = state.get(r["target"])
            row["lineage"] = lineage(gc, r["target"])
            row["preview"] = asdict(
                edit.apply_edit(
                    target=r["id"],
                    rename=r["target"],
                    reason=REASON,
                    origin="agent",
                    scope="only_self",
                    dry_run=True,
                    gc=gc,
                )
            )
            row["action"] = (
                "tritium_split"
                if r["id"] == BARE
                else ("report" if row["preview"]["blocked"] else "reclaim")
            )
            rows.append(row)
        result = {
            "run_id": RUN.name,
            "observed_at": datetime.now(UTC).isoformat(),
            "revision": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "authority": {
                "convention": REASON,
                "clarification": "Follow superseded intermediates to the live tip; electron-temperature closes on the renaming name. Generated models are expected ignored provisioning.",
                "origin": "explicit continuation instruction",
            },
            "census": original,
            "before": state,
            "operations": rows,
            "tritium": {
                "retire": BARE,
                "compose": [BARE + "_at_pedestal_top", BARE + "_at_plasma_boundary"],
            },
            "spend_ceiling_usd": 2.0,
        }
        write(RUN / "authority-full-preview.json", result)
        result["census"] = [
            {k: r[k] for k in ("id", "target", "paths", "locus")} for r in original
        ]
        for item in state.values():
            reviews = item.pop("reviews")
            item["review_count"] = len(reviews)
            item["review_ids_sha256"] = hashlib.sha256(
                json.dumps(sorted(reviews)).encode()
            ).hexdigest()
        write(AUTHORITY, result)
        print(
            json.dumps(
                {
                    "operations": [
                        {k: r[k] for k in ("id", "target", "action")} for r in rows
                    ]
                },
                indent=2,
            ),
            flush=True,
        )


def apply_one(name):
    a = json.loads(AUTHORITY.read_text())
    row = next(r for r in a["operations"] if r["id"] == name)
    assert row["action"] == "reclaim", row
    key = hashlib.sha256(name.encode()).hexdigest()[:16]
    receipt_path = RUN / f"reclaim-{key}.json"
    assert not receipt_path.exists(), "read an existing receipt rather than replaying"
    with GraphClient() as gc:
        before = snapshot(gc, [name, row["target"]])
        preview = asdict(
            edit.apply_edit(
                target=name,
                rename=row["target"],
                reason=REASON,
                origin="agent",
                scope="only_self",
                dry_run=True,
                gc=gc,
            )
        )
        evidence = {
            "authority_sha256": hashlib.sha256(AUTHORITY.read_bytes()).hexdigest(),
            "before": before,
            "preview": preview,
        }
        write(RUN / f"preview-{key}.json", evidence)
        assert not preview["blocked"], preview
        result = asdict(
            edit.apply_edit(
                target=name,
                rename=row["target"],
                reason=REASON,
                origin="agent",
                scope="only_self",
                dry_run=False,
                gc=gc,
            )
        )
        evidence["apply"] = result
        write(receipt_path, evidence)
        assert result["applied"] and not result["blocked"], result
        evidence["receipts"] = gc.query(
            "MATCH (c:StandardNameChange {run_id:$run}) RETURN properties(c) AS receipt",
            run=result["run_id"],
        )
        evidence["after"] = snapshot(gc, [name, row["target"]])
        write(receipt_path, evidence)
        receipts = [
            r["receipt"]
            for r in evidence["receipts"]
            if r["receipt"].get("operation") == "reclaim_spelling"
        ]
        assert len(receipts) == 1, evidence["receipts"]
        assert json.loads(receipts[0]["reason"])["case"] in (
            "closing_lineage",
            "different_locus",
        )
        after = evidence["after"][row["target"]]
        assert after["stage"] != "superseded" and after["run_id"] == result["run_id"]
        assert set(before[name]["sources"]) <= set(after["sources"])
        assert set(before[row["target"]]["reviews"]) <= set(after["reviews"])
        print(
            json.dumps(
                {
                    "from": name,
                    "to": row["target"],
                    "receipt": receipts[0]["id"],
                    "run_id": result["run_id"],
                    "case": json.loads(receipts[0]["reason"])["case"],
                    "sources": after["sources"],
                },
                default=str,
            ),
            flush=True,
        )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["prepare", "apply", "apply-all"])
    p.add_argument("--name")
    args = p.parse_args()
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    print(
        f"# revision: {revision}; tree: {Path.cwd()}; command: "
        + shlex.join(["uv", "run", "--no-sync", "python", *sys.argv]),
        flush=True,
    )
    print("module=" + edit.__file__, flush=True)
    print("cwd=" + str(Path.cwd().resolve()), flush=True)
    if args.mode == "prepare":
        prepare()
    elif args.mode == "apply":
        apply_one(args.name)
    else:
        authority = json.loads(AUTHORITY.read_text())
        for row in authority["operations"]:
            if row["action"] == "reclaim":
                apply_one(row["id"])


if __name__ == "__main__":
    main()
