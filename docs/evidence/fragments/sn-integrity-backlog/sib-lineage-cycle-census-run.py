"""Census refinement cycles and branched live lineages without mutating the graph.

REFINED_FROM is directed successor to predecessor. A lineage is a weakly
connected component. A live tip is a non-superseded member with no other live
descendant; all live members are also reported to expose reclaimed ancestors.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from imas_codex.graph.client import GraphClient  # noqa: E402

RUN = Path(
    "/home/ITER/mcintos/.config/reckon/crew/runs/r-20261107T201632402746-sib-lineage-cycle-census"
)
HERE = Path(__file__).resolve().parent
STEM = Path(__file__).stem.removesuffix("-run")


def reachable(start, adjacency):
    seen, pending = set(), list(adjacency[start])
    while pending:
        node = pending.pop()
        if node not in seen:
            seen.add(node)
            pending.extend(adjacency[node] - seen)
    return seen


def classify(nodes, edges):
    parents, children, neighbours = (defaultdict(set) for _ in range(3))
    for edge in edges:
        a, b = edge["successor"], edge["predecessor"]
        parents[a].add(b)
        children[b].add(a)
        neighbours[a].add(b)
        neighbours[b].add(a)
    cycle_names = {node for node in nodes if node in reachable(node, parents)}
    cycles, pending_cycles = [], set(cycle_names)
    while pending_cycles:
        start = min(pending_cycles)
        component = (reachable(start, parents) & reachable(start, children)) | {start}
        cycles.append(sorted(component))
        pending_cycles -= component
    components, remaining = [], set(nodes)
    while remaining:
        start = min(remaining)
        component = reachable(start, neighbours) | {start}
        remaining -= component
        live = {n for n in component if nodes[n]["stage"] != "superseded"}
        tips = {n for n in live if not ((reachable(n, children) - {n}) & live)}
        components.append(
            {
                "members": sorted(component),
                "live_members": sorted(live),
                "live_tips": sorted(tips),
                "cycle_names": sorted(component & cycle_names),
            }
        )
    return cycles, components


def controls():
    nodes = {
        x: {"stage": "accepted" if x in {"left", "right", "solo"} else "superseded"}
        for x in ["ancestor", "left", "right", "loop_a", "loop_b", "self", "solo"]
    }
    edges = [
        {"successor": a, "predecessor": b}
        for a, b in [
            ("left", "ancestor"),
            ("right", "ancestor"),
            ("loop_a", "loop_b"),
            ("loop_b", "loop_a"),
            ("self", "self"),
        ]
    ]
    cycles, components = classify(nodes, edges)
    assert cycles == [["loop_a", "loop_b"], ["self"]]
    assert [c["live_tips"] for c in components if len(c["live_tips"]) > 1] == [
        ["left", "right"]
    ]
    print(
        "CONTROLS reciprocal cycle=2 names; self-loop=1; branching tips=2; isolated clean name=1"
    )


def capture():
    observations = []
    with GraphClient() as gc:
        print(
            f"endpoint={gc.uri}; graph-client={sys.modules[GraphClient.__module__].__file__}"
        )

        def query(label, cypher, **params):
            # Every statement is a read and every transaction is rolled back.
            assert cypher.lstrip().startswith(("MATCH", "CALL", "UNWIND"))
            began = time.monotonic()
            with gc.session() as session, session.begin_transaction(timeout=10) as tx:
                rows = [dict(r) for r in tx.run(cypher, **params)]
                tx.rollback()
            elapsed = time.monotonic() - began
            observations.append(
                {
                    "label": label,
                    "cypher": cypher,
                    "params": params,
                    "rows": len(rows),
                    "elapsed_seconds": elapsed,
                }
            )
            print(f"READ {label}: rows={len(rows)} seconds={elapsed:.3f}", flush=True)
            return rows

        inventory = query(
            "inventory",
            """CALL { MATCH (n:StandardName)
          RETURN count(n) AS names, count(n.id) AS ids, count(n.name_stage) AS stages }
          CALL { MATCH (:StandardName)-[e:REFINED_FROM]->(:StandardName) RETURN count(e) AS edges }
          CALL { MATCH (c:StandardNameChange) RETURN count(c) AS changes, count(c.operation) AS operations }
          RETURN names, ids, stages, edges, changes, operations""",
        )[0]
        assert inventory["names"] == inventory["ids"] == inventory["stages"] > 0
        assert inventory["edges"] > 0 and inventory["operations"] > 0
        nodes = query(
            "nodes",
            """MATCH (n:StandardName) RETURN n.id AS id,
          n.name_stage AS stage, n.status AS status, n.origin AS origin,
          n.source_paths AS recorded_paths, n.source_id AS source_id,
          n.refined_from AS refined_from, n.run_id AS run_id,
          n.created_at AS created_at, n.updated_at AS updated_at,
          n.reviewer_score_name AS name_score, n.reviewer_score_docs AS docs_score,
          n.description AS description, n.unit AS unit ORDER BY id LIMIT 20000""",
        )
        edges = query(
            "edges",
            """MATCH (a:StandardName)-[e:REFINED_FROM]->(b:StandardName)
          RETURN a.id AS successor, b.id AS predecessor, properties(e) AS properties,
          elementId(e) AS edge_id ORDER BY successor, predecessor LIMIT 30000""",
        )
        assert len(nodes) == inventory["names"] and len(edges) == inventory["edges"]
        by_id = {n["id"]: n for n in nodes}
        cycles, components = classify(by_id, edges)
        independent = query(
            "cycle-path-cross-check",
            """MATCH (n:StandardName)
          WHERE EXISTS { MATCH (n)-[:REFINED_FROM*1..]->(n) }
          RETURN n.id AS id ORDER BY id LIMIT 20000""",
        )
        assert {r["id"] for r in independent} == {n for cycle in cycles for n in cycle}
        independent_tips = query(
            "live-tip-cross-check",
            """MATCH (n:StandardName)
          WHERE n.name_stage <> 'superseded' AND NOT EXISTS {
            MATCH (other:StandardName)-[:REFINED_FROM*1..]->(n)
            WHERE other <> n AND other.name_stage <> 'superseded' }
          RETURN n.id AS id ORDER BY id LIMIT 20000""",
        )
        assert {r["id"] for r in independent_tips} == {
            n for c in components for n in c["live_tips"]
        }
        cases = [
            c
            for c in components
            if c["cycle_names"]
            or len(c["live_tips"]) > 1
            or any(
                n in by_id
                and (
                    "argon" in n
                    or "tungsten" in n
                    or "deuterium_tritium" in n
                    or "normalized_toroidal_flux_coordinate_at_plasma_boundary" == n
                )
                for n in c["members"]
            )
        ]
        selected = sorted({n for c in cases for n in c["members"]})
        changes = query(
            "selected-change-ledgers",
            """MATCH (c:StandardNameChange)
          WHERE c.from_name IN $ids OR c.to_name IN $ids
            OR EXISTS { MATCH (n:StandardName)-[:HAS_INTERNAL_CHANGE]->(c) WHERE n.id IN $ids }
          RETURN properties(c) AS properties,
          [(n:StandardName)-[:HAS_INTERNAL_CHANGE]->(c) | n.id] AS attached_to
          ORDER BY c.changed_at, c.id LIMIT 30000""",
            ids=selected,
        )
        sources = query(
            "selected-producing-sources",
            """MATCH (s:StandardNameSource)-[:PRODUCED_NAME]->(n:StandardName)
          WHERE n.id IN $ids RETURN n.id AS name, s.id AS source, s.source_id AS path,
          s.status AS status, s.source_type AS source_type,
          [(s)-[:FROM_DD_PATH]->(dd:IMASNode) | dd.id] AS dd_paths
          ORDER BY name, source LIMIT 30000""",
            ids=selected,
        )
        projections = query(
            "selected-projections",
            """MATCH (dd:IMASNode)-[:HAS_STANDARD_NAME]->(n:StandardName)
          WHERE n.id IN $ids RETURN n.id AS name, dd.id AS path ORDER BY name, path LIMIT 30000""",
            ids=selected,
        )
        successors = query(
            "selected-signed-successors",
            """MATCH (a:StandardName)-[e:HAS_SUCCESSOR]->(b:StandardName)
          WHERE a.id IN $ids OR b.id IN $ids RETURN a.id AS predecessor,
          b.id AS successor, properties(e) AS properties LIMIT 30000""",
            ids=selected,
        )
        counts_after = query(
            "inventory-readback",
            """CALL { MATCH (n:StandardName) RETURN count(n) AS names }
          CALL { MATCH (:StandardName)-[e:REFINED_FROM]->(:StandardName) RETURN count(e) AS edges }
          RETURN names, edges""",
        )[0]
        assert counts_after == {k: inventory[k] for k in ["names", "edges"]}
    result = {
        "observed_at": datetime.now(UTC).isoformat(),
        "revision": subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
        ).strip(),
        "cwd": str(ROOT),
        "inventory": inventory,
        "nodes": nodes,
        "edges": edges,
        "cycles": cycles,
        "components": components,
        "cases": cases,
        "changes": changes,
        "sources": sources,
        "projections": projections,
        "signed_successors": successors,
        "queries": observations,
        "cycle_cross_check": independent,
        "tip_cross_check": independent_tips,
        "spend_usd": 0,
    }
    (RUN / "graph-snapshot.json").write_text(
        json.dumps(result, default=str, indent=2) + "\n"
    )
    print(
        f"CAPTURE names={len(nodes)} edges={len(edges)} cycle_components={len(cycles)} "
        f"cycle_names={sum(map(len, cycles))} multi_tip_components={sum(len(c['live_tips']) > 1 for c in components)}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", action="store_true")
    args = parser.parse_args()
    controls()
    if args.capture:
        capture()
    return 0


if __name__ == "__main__":
    try:
        code = main()
    except Exception:
        import traceback

        traceback.print_exc()
        code = 1
    print(f"EXIT={code}", flush=True)
    raise SystemExit(code)
