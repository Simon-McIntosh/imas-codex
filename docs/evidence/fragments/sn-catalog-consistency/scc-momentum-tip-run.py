"""Gather the evidence for one live tip of the momentum-diffusivity lineage.

Reads the two-name lineage's records and its DD sources from the live graph,
loads the lineage census snapshot the committed census produced, and shows the
census separating a two-tip lineage from a one-tip one planted in a disposable
graph. The lineage reader is the committed census; this script imports its
``classify`` rather than re-deriving tip selection.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import socket
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from imas_codex.graph.client import GraphClient  # noqa: E402

HERE = Path(__file__).resolve().parent
STEM = Path(__file__).stem
CENSUS = (
    ROOT
    / "docs/evidence/fragments/sn-integrity-backlog/sib-lineage-cycle-census-run.py"
)
NAMES = [
    "toroidal_plasma_momentum_diffusivity",
    "toroidal_momentum_diffusion_coefficient",
]
NODE_FIELDS = """n.id AS id, n.name_stage AS stage, n.description AS description,
 n.documentation AS documentation, n.unit AS unit, n.status AS status,
 n.origin AS origin, n.physics_domain AS physics_domain,
 n.reviewer_score_name AS name_score, n.reviewer_score_docs AS docs_score,
 n.review_count AS review_count, n.review_mean_score AS review_mean,
 n.llm_cost AS llm_cost, n.llm_cost_generate_name AS cost_generate_name,
 n.llm_cost_refine_name AS cost_refine_name,
 n.llm_cost_review_name AS cost_review_name,
 n.llm_cost_review_docs AS cost_review_docs,
 n.created_at AS created_at, n.updated_at AS updated_at,
 n.source_paths AS source_paths, n.superseded_by AS superseded_by"""


def load_census():
    spec = importlib.util.spec_from_file_location("lineage_census", CENSUS)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def live_records() -> dict:
    out: dict = {"names": NAMES}
    with GraphClient() as gc:
        out["endpoint"] = gc.uri
        out["nodes"] = [
            dict(r)
            for r in gc.query(
                f"MATCH (n:StandardName) WHERE n.id IN $ids RETURN {NODE_FIELDS}",
                ids=NAMES,
            )
        ]
        out["sources"] = [
            dict(r)
            for r in gc.query(
                """MATCH (s:StandardNameSource)-[:PRODUCED_NAME]->(n:StandardName)
                   WHERE n.id IN $ids
                   RETURN n.id AS name, s.id AS source, s.source_id AS path,
                     s.status AS status, s.source_type AS source_type,
                     [(s)-[:FROM_DD_PATH]->(dd:IMASNode) | dd.id] AS dd_paths
                   ORDER BY name, source""",
                ids=NAMES,
            )
        ]
        out["dd_nodes"] = [
            dict(r)
            for r in gc.query(
                """MATCH (dd:IMASNode)-[:HAS_STANDARD_NAME]->(n:StandardName)
                   WHERE n.id IN $ids
                   RETURN n.id AS name, dd.id AS path,
                     dd.documentation AS dd_documentation, dd.unit AS dd_unit,
                     dd.data_type AS data_type, dd.physics_domain AS domain
                   ORDER BY name, path""",
                ids=NAMES,
            )
        ]
        out["units"] = [
            dict(r)
            for r in gc.query(
                """MATCH (n:StandardName)-[:HAS_UNIT]->(u:Unit)
                   WHERE n.id IN $ids RETURN n.id AS name, u.id AS unit""",
                ids=NAMES,
            )
        ]
        out["reviews"] = [
            dict(r)
            for r in gc.query(
                """MATCH (n:StandardName)-[:HAS_REVIEW]->(r:StandardNameReview)
                   WHERE n.id IN $ids
                   RETURN n.id AS name, r.id AS id, r.score AS score,
                     r.tier AS tier, r.model AS model, r.llm_cost AS llm_cost,
                     r.judge_kind AS judge_kind, r.is_canonical AS is_canonical,
                     r.resolution_method AS resolution_method,
                     r.reviewed_at AS reviewed_at
                   ORDER BY name, r.reviewed_at, r.id""",
                ids=NAMES,
            )
        ]
        out["changes"] = [
            dict(r)
            for r in gc.query(
                """MATCH (n:StandardName)-[:HAS_INTERNAL_CHANGE]->(c:StandardNameChange)
                   WHERE n.id IN $ids
                   RETURN n.id AS attached_to, c.id AS id, c.operation AS operation,
                     c.from_name AS from_name, c.to_name AS to_name,
                     c.reason AS reason, c.changed_at AS changed_at
                   ORDER BY changed_at, id""",
                ids=NAMES,
            )
        ]
        out["lineage_changes"] = [
            dict(r)
            for r in gc.query(
                """MATCH (c:StandardNameChange)
                   WHERE c.from_name IN $ids OR c.to_name IN $ids
                   RETURN c.id AS id, c.operation AS operation,
                     c.from_name AS from_name, c.to_name AS to_name,
                     c.reason AS reason, c.changed_at AS changed_at
                   ORDER BY changed_at, id""",
                ids=NAMES + ["toroidal_momentum_diffusivity"],
            )
        ]
        out["signed_successors"] = [
            dict(r)
            for r in gc.query(
                """MATCH (a:StandardName)-[e:HAS_SUCCESSOR]->(b:StandardName)
                   WHERE a.id IN $ids OR b.id IN $ids
                   RETURN a.id AS predecessor, b.id AS successor,
                     properties(e) AS properties""",
                ids=NAMES,
            )
        ]
        members = [
            r["id"]
            for r in gc.query(
                """MATCH (n:StandardName) WHERE n.id IN $ids
                   OPTIONAL MATCH (up:StandardName)-[:REFINED_FROM*0..]->(n)
                   WITH collect(DISTINCT up) AS ups UNWIND ups AS u
                   OPTIONAL MATCH (u)-[:REFINED_FROM*0..]->(d:StandardName)
                   WITH collect(DISTINCT d) AS ds UNWIND ds AS d
                   RETURN DISTINCT d.id AS id ORDER BY id""",
                ids=NAMES,
            )
        ]
        out["lineage"] = {
            "members": members,
            "nodes": [
                dict(r)
                for r in gc.query(
                    """MATCH (n:StandardName) WHERE n.id IN $ids
                       RETURN n.id AS id, n.name_stage AS stage,
                         n.name_stage_previous AS previous_stage,
                         n.unit AS unit, n.reviewer_score_name AS name_score,
                         n.description AS description
                       ORDER BY id""",
                    ids=members,
                )
            ],
            "edges": [
                dict(r)
                for r in gc.query(
                    """MATCH (a:StandardName)-[:REFINED_FROM]->(b:StandardName)
                       WHERE a.id IN $ids AND b.id IN $ids
                       RETURN a.id AS successor, b.id AS predecessor
                       ORDER BY successor, predecessor""",
                    ids=members,
                )
            ],
        }
    return out


def census_readings(snapshot_path: Path) -> dict:
    snapshot = json.loads(snapshot_path.read_text())
    records = {n["id"]: n for n in snapshot["nodes"]}
    edges = [
        {"successor": e["successor"], "predecessor": e["predecessor"]}
        for e in snapshot["edges"]
    ]
    target = [c for c in snapshot["components"] if set(NAMES) & set(c["members"])]
    assert len(target) == 1, target
    return {
        "snapshot": str(snapshot_path),
        "observed_at": snapshot["observed_at"],
        "revision": snapshot["revision"],
        "inventory": snapshot["inventory"],
        "cycle_components": len(snapshot["cycles"]),
        "cycle_names": sum(len(c) for c in snapshot["cycles"]),
        "components": len(snapshot["components"]),
        "multi_tip_components": sum(
            1 for c in snapshot["components"] if len(c["live_tips"]) > 1
        ),
        "lineage": target[0],
        "lineage_live_tip_count": len(target[0]["live_tips"]),
        "tip_cross_check_rows": len(snapshot["tip_cross_check"]),
        "cycle_cross_check_rows": len(snapshot["cycle_cross_check"]),
        "tip_cross_check_agrees": {r["id"] for r in snapshot["tip_cross_check"]}
        == {n for c in snapshot["components"] for n in c["live_tips"]},
        "records_for_names": {
            name: {k: records[name].get(k) for k in
                   ("stage", "unit", "name_score", "docs_score", "description")}
            for name in NAMES
        },
        "edges_checked": len(edges),
    }


ANCESTOR = "control_momentum_diffusivity"
TIP_A = "control_plasma_momentum_diffusivity"
TIP_B = "control_momentum_diffusion_coefficient"
LOOP = ["control_loop_a", "control_loop_b"]


def disposable_control(classify) -> dict:
    from imas_codex.graph import temp_neo4j

    def port():
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            return sock.getsockname()[1]

    def client_for(uri):
        return lambda: GraphClient(
            uri=uri, username="neo4j", password="", graph_name="lineage-control"
        )

    def read(gc):
        names = gc.query(
            "MATCH (n:StandardName) RETURN n.id AS id, n.name_stage AS stage"
        )
        edges = gc.query(
            """MATCH (a:StandardName)-[:REFINED_FROM]->(b:StandardName)
               RETURN a.id AS successor, b.id AS predecessor"""
        )
        cycles, components = classify({n["id"]: n for n in names}, edges)
        return {
            "nodes": sorted(n["id"] for n in names),
            "edges": sorted((e["successor"], e["predecessor"]) for e in edges),
            "cycles": cycles,
            "components": components,
            "live_tips": sorted(t for c in components for t in c["live_tips"]),
            "multi_tip": [
                {"members": c["members"], "live_tips": c["live_tips"]}
                for c in components
                if len(c["live_tips"]) > 1
            ],
        }

    result: dict = {"fixture": {}, "arms": {}}
    with tempfile.TemporaryDirectory(
        prefix="momentum-control-", dir=os.environ["TMPDIR"]
    ) as directory:
        root = Path(directory)
        bolt, http = port(), port()
        process, server_log = temp_neo4j.start_temp_neo4j(
            root, bolt, http, load_dump=False
        )
        uri = f"bolt://127.0.0.1:{bolt}"
        result["endpoint"] = uri
        client = client_for(uri)
        try:
            with client() as gc:
                empty = gc.query("MATCH (n) RETURN count(n) AS n")[0]["n"]
                assert empty == 0, empty
                result["empty_before_plant"] = empty
                result["fixture"] = {
                    "ancestor": ANCESTOR,
                    "tips": [TIP_A, TIP_B],
                    "cycle": LOOP,
                }
                # Two-tip arm: one superseded ancestor refined into two live
                # successors, the shape the census reports for this lineage.
                gc.query(
                    """CREATE (a:StandardName {id:$ancestor, name_stage:'superseded',
                        status:'superseded'})
                    CREATE (x:StandardName {id:$tip_a, name_stage:'accepted',
                        status:'draft'})
                    CREATE (y:StandardName {id:$tip_b, name_stage:'accepted',
                        status:'draft'})
                    CREATE (x)-[:REFINED_FROM]->(a)
                    CREATE (y)-[:REFINED_FROM]->(a)""",
                    ancestor=ANCESTOR,
                    tip_a=TIP_A,
                    tip_b=TIP_B,
                )
                result["arms"]["two_tip"] = read(gc)
                # Cycle arm: two live names refining each other.
                gc.query(
                    """CREATE (p:StandardName {id:$a, name_stage:'accepted',
                        status:'draft'})
                    CREATE (q:StandardName {id:$b, name_stage:'accepted',
                        status:'draft'})
                    CREATE (p)-[:REFINED_FROM]->(q)
                    CREATE (q)-[:REFINED_FROM]->(p)""",
                    a=LOOP[0],
                    b=LOOP[1],
                )
                result["arms"]["two_tip_plus_cycle"] = read(gc)
                # One-tip arm: fold the second successor into the first, so the
                # reader's tip count must fall to one on the same instrument.
                gc.query(
                    """MATCH (y:StandardName {id:$tip_b})
                    OPTIONAL MATCH (y)-[:REFINED_FROM]->(a:StandardName {id:$ancestor})
                    DETACH DELETE y, a""",
                    tip_b=TIP_B,
                    ancestor=ANCESTOR,
                )
                gc.query(
                    """MATCH (x:StandardName {id:$tip_a})
                    CREATE (y:StandardName {id:$tip_b, name_stage:'superseded',
                        status:'superseded'})
                    CREATE (y)-[:REFINED_FROM]->(x)""",
                    tip_a=TIP_A,
                    tip_b=TIP_B,
                )
                # The cycle pair leaves the graph with the arm that planted it,
                # so the one-tip reading sees only the folded lineage.
                gc.query(
                    """MATCH (n:StandardName) WHERE n.id IN $ids DETACH DELETE n""",
                    ids=LOOP,
                )
                result["arms"]["one_tip"] = read(gc)
                result["counts_after"] = gc.query(
                    "MATCH (n) RETURN count(n) AS n"
                )[0]["n"]
            assert result["arms"]["two_tip"]["live_tips"] == sorted([TIP_A, TIP_B])
            assert [
                {"members": c["members"], "live_tips": c["live_tips"]}
                for c in result["arms"]["two_tip"]["components"]
                if len(c["live_tips"]) > 1
            ] == [{"members": sorted([ANCESTOR, TIP_A, TIP_B]),
                   "live_tips": sorted([TIP_A, TIP_B])}]
            assert result["arms"]["two_tip"]["cycles"] == []
            assert result["arms"]["two_tip_plus_cycle"]["cycles"] == [sorted(LOOP)]
            assert result["arms"]["one_tip"]["live_tips"] == [TIP_A]
            assert result["arms"]["one_tip"]["multi_tip"] == []
            assert result["arms"]["one_tip"]["cycles"] == []
            result["assertions"] = "all passed"
            print(
                "CONTROL two_tip live_tips=%d cycles=%d | cycle_arm cycles=%d | "
                "one_tip live_tips=%d"
                % (
                    len(result["arms"]["two_tip"]["live_tips"]),
                    len(result["arms"]["two_tip"]["cycles"]),
                    len(result["arms"]["two_tip_plus_cycle"]["cycles"]),
                    len(result["arms"]["one_tip"]["live_tips"]),
                ),
                flush=True,
            )
        finally:
            with (root / "processes.txt").open("w") as handle:
                handle.write(f"{process.pid}\n")
            process.terminate()
            process.wait(timeout=60)
            print(f"DISPOSABLE removed scratch {root}", flush=True)
            result["server_log"] = str(server_log)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=HERE)
    parser.add_argument("--capture", action="store_true")
    parser.add_argument("--control", action="store_true")
    parser.add_argument("--assemble", action="store_true")
    args = parser.parse_args()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    census = load_census()
    if args.capture:
        records = live_records()
        records["spend_usd"] = 0.0
        records["spend_note"] = (
            "no LLM call in this read; the run's own model spend is recorded "
            "in the manifest"
        )
        (args.run_dir / "live-records.json").write_text(
            json.dumps(records, indent=2, default=str) + "\n"
        )
        print(
            "CAPTURE names=%d sources=%d reviews=%d changes=%d lineage_members=%d"
            % (
                len(records["nodes"]),
                len(records["sources"]),
                len(records["reviews"]),
                len(records["changes"]),
                len(records["lineage"]["members"]),
            ),
            flush=True,
        )
    if args.control:
        control = disposable_control(census.classify)
        (args.run_dir / "disposable-control.json").write_text(
            json.dumps(control, indent=2, default=str) + "\n"
        )
    if args.assemble:
        evidence = {
            "observed_at": datetime.now(UTC).isoformat(),
            "node": STEM,
            "revision": subprocess.check_output(
                ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
            ).strip(),
            "live": json.loads((args.run_dir / "live-records.json").read_text()),
            "census": census_readings(args.run_dir / "graph-snapshot.json"),
            "disposable_control": json.loads(
                (args.run_dir / "disposable-control.json").read_text()
            ),
            "spend_usd": 0.0,
            "reuse": {
                "lineage_reader": str(CENSUS.relative_to(ROOT)),
                "cycle_owner": "imas_codex.standard_names.catalog_ordering"
                "_find_cyclic_components",
            },
        }
        evidence["spend_note"] = (
            "this node made no LLM call; the catalog's stored review spend is "
            "in live.nodes[].llm_cost"
        )
        target = args.out / f"{STEM.removesuffix('-run')}.json"
        target.write_text(json.dumps(evidence, indent=2, default=str) + "\n")
        print(f"ASSEMBLE {target} bytes={target.stat().st_size}", flush=True)
        print(
            "READING live_tips=%d cycles=%d multi_tip_components=%d names=%d"
            % (
                evidence["census"]["lineage_live_tip_count"],
                evidence["census"]["cycle_components"],
                evidence["census"]["multi_tip_components"],
                evidence["census"]["inventory"]["names"],
            ),
            flush=True,
        )
    return 0


if __name__ == "__main__":
    print(
        f"# revision: {subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()}"
        f" tree: {ROOT} command: {' '.join(sys.argv)}",
        flush=True,
    )
    print(
        f"# module: {Path(__file__).resolve()}; cwd: {Path.cwd().resolve()}",
        flush=True,
    )
    print(
        f"# runner_sha256: {hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}",
        flush=True,
    )
    try:
        code = main()
    except Exception:
        import traceback

        traceback.print_exc()
        code = 1
    print(f"EXIT={code}", flush=True)
    raise SystemExit(code)