"""Census refinement cycles and branched live lineages without mutating the graph.

REFINED_FROM is directed successor to predecessor. A lineage is a weakly
connected component. A live tip is a non-superseded member with no other live
descendant; all live members are also reported to expose reclaimed ancestors.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shlex
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from imas_codex.graph.client import GraphClient  # noqa: E402
from imas_codex.standard_names import catalog_ordering  # noqa: E402

HERE = Path(__file__).resolve().parent
STEM = Path(__file__).stem.removesuffix("-run")
SPECIAL = {
    "argon_density_at_plasma_boundary",
    "argon_density_at_separatrix",
    "tungsten_density_at_plasma_boundary",
    "tungsten_density_at_separatrix",
    "flux_surface_averaged_deuterium_tritium_density_at_plasma_boundary",
    "deuterium_tritium_density_flux_surface_averaged_at_plasma_boundary",
    "normalized_toroidal_flux_coordinate_at_plasma_boundary",
    "normalized_toroidal_flux_coordinate_at_separatrix",
}


def reachable(start, adjacency):
    seen, pending = set(), list(adjacency[start])
    while pending:
        node = pending.pop()
        if node not in seen:
            seen.add(node)
            pending.extend(adjacency[node] - seen)
    return seen


def classify(nodes, edges):
    children, neighbours = (defaultdict(set) for _ in range(2))
    for edge in edges:
        a, b = edge["successor"], edge["predecessor"]
        children[b].add(a)
        neighbours[a].add(b)
        neighbours[b].add(a)
    # Reversing every edge preserves strongly connected components. Pass the
    # predecessor-to-successor adjacency to the catalog's shared cycle owner.
    cycle_sets = catalog_ordering._find_cyclic_components(
        set(nodes), {node: sorted(children[node]) for node in nodes}
    )
    cycles = [sorted(component) for component in cycle_sets]
    cycle_names = set().union(*cycle_sets)
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


def controls(classifier=classify):
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
    cycles, components = classifier(nodes, edges)
    assert cycles == [["loop_a", "loop_b"], ["self"]]
    assert [c["live_tips"] for c in components if len(c["live_tips"]) > 1] == [
        ["left", "right"]
    ]
    print(
        "CONTROLS reciprocal cycle=2 names; self-loop=1; branching tips=2; isolated clean name=1"
    )


def verify(run_dir, candidate_path=None):
    """Check shared cycle ownership and topology against the retained snapshot."""
    classifier = classify
    if candidate_path is not None:
        spec = importlib.util.spec_from_file_location(
            "census_candidate", candidate_path
        )
        candidate = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(candidate)
        classifier = candidate.classify
    source = Path(classifier.__code__.co_filename).resolve()
    print(
        f"# candidate_module: {source}; sha256: {hashlib.sha256(source.read_bytes()).hexdigest()}"
    )
    nodes = {name: {"stage": "accepted"} for name in ["alpha", "beta", "child"]}
    edges = [
        {"successor": a, "predecessor": b}
        for a, b in [("alpha", "beta"), ("beta", "alpha"), ("child", "beta")]
    ]

    def owner_delegation():
        with patch.object(
            catalog_ordering,
            "_find_cyclic_components",
            wraps=catalog_ordering._find_cyclic_components,
        ) as owner:
            cycles, _ = classifier(nodes, edges)
        assert cycles == [["alpha", "beta"]], "acyclic descendant entered cycle"
        owner.assert_called_once_with(
            set(nodes), {"alpha": ["beta"], "beta": ["alpha", "child"], "child": []}
        )

    def owner_refusal():
        class OwnerRefusal(RuntimeError):
            pass

        with patch.object(
            catalog_ordering,
            "_find_cyclic_components",
            side_effect=OwnerRefusal("cycle owner refused the planted input"),
        ):
            try:
                classifier(nodes, edges)
            except OwnerRefusal as exc:
                print(f"REFUSAL propagated: {exc}")
                return
        raise AssertionError("census bypassed the cycle owner's refusal")

    def snapshot_equivalence():
        data = json.loads((run_dir / "graph-snapshot.json").read_text())
        cycles, components = classifier(
            {n["id"]: n for n in data["nodes"]}, data["edges"]
        )
        assert cycles == data["cycles"], "cycle groups changed"
        assert components == data["components"], "lineages or live tips changed"
        assert {n for cycle in cycles for n in cycle} == {
            row["id"] for row in data["cycle_cross_check"]
        }, "cycle membership differs from independent Cypher"
        assert {n for component in components for n in component["live_tips"]} == {
            row["id"] for row in data["tip_cross_check"]
        }, "live tips differ from independent Cypher"
        print(
            f"REPLAY names={len(data['nodes'])} edges={len(data['edges'])} "
            f"cycle_components={len(cycles)} cycle_names={sum(map(len, cycles))} "
            f"live_tips={len(data['tip_cross_check'])} "
            f"multi_tip_components={sum(len(c['live_tips']) > 1 for c in components)}"
        )

    checks = {
        "cycle_owner_delegation": owner_delegation,
        "cycle_owner_refusal": owner_refusal,
        "planted_topology": lambda: controls(classifier),
        "snapshot_equivalence": snapshot_equivalence,
    }
    failures = []
    for name, check in checks.items():
        try:
            check()
        except AssertionError as exc:
            failures.append(name)
            print(f"FAILED {name}: {exc}")
        else:
            print(f"PASSED {name}")
    print(f"VERIFY passed={len(checks) - len(failures)} failed={len(failures)}")
    return int(bool(failures))


def capture(run_dir):
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
            or SPECIAL.intersection(c["members"])
        ]
        selected = sorted({n for c in cases for n in c["members"]})
        assert SPECIAL <= set(selected)
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
        for result_rows in [changes, sources]:
            assert len(result_rows) < 30000, "supporting evidence reached its row cap"
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
    (run_dir / "graph-snapshot.json").write_text(
        json.dumps(result, default=str, indent=2) + "\n"
    )
    print(
        f"CAPTURE names={len(nodes)} edges={len(edges)} cycle_components={len(cycles)} "
        f"cycle_names={sum(map(len, cycles))} multi_tip_components={sum(len(c['live_tips']) > 1 for c in components)}"
    )


def reproduce(run_dir):
    """Exercise the actual writers against an empty, isolated local database."""
    from imas_codex.graph import temp_neo4j
    from imas_codex.standard_names import edit
    from imas_codex.standard_names.graph_store import graph_refinement

    def port():
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            return sock.getsockname()[1]

    results = []
    with tempfile.TemporaryDirectory(
        prefix="lineage-probe-", dir=os.environ["TMPDIR"]
    ) as directory:
        root = Path(directory)
        bolt, http = port(), port()
        process, server_log = temp_neo4j.start_temp_neo4j(
            root, bolt, http, load_dump=False
        )
        uri = f"bolt://127.0.0.1:{bolt}"

        def client():
            return GraphClient(
                uri=uri, username="neo4j", password="", graph_name="lineage-disposable"
            )

        try:
            with client() as gc:
                assert gc.query("MATCH (n) RETURN count(n) AS n")[0]["n"] == 0
            print(
                f"DISPOSABLE endpoint={uri} empty=confirmed module={edit.__file__}",
                flush=True,
            )
            services = SimpleNamespace(
                GraphClient=client, bump_sn_run_counter=lambda *_: None
            )

            def seed(target_stage="superseded", *, branch=False):
                with client() as gc:
                    previous = gc.query("MATCH (n) RETURN count(n) AS n")[0]["n"]
                    print(
                        f"DISPOSABLE reset {previous} fixture nodes; live graph excluded by explicit endpoint {uri}"
                    )
                    gc.query("MATCH (n) DETACH DELETE n")
                    gc.query(
                        """CREATE (old:StandardName {id:'electron_number_density',
                        name_stage:'accepted',status:'draft',origin:'pipeline',
                        unit:'m^-3',description:'Electron number density.',validation_status:'valid'})
                        CREATE (target:StandardName {id:'electron_density', name_stage:$stage,
                        status:'draft', origin:'pipeline', unit:'m^-3',
                        description:'Electron number density.',validation_status:'valid',
                        superseded_from_stage:'accepted'})
                        CREATE (old)-[:REFINED_FROM]->(target)
                        CREATE (dd:IMASNode {id:'core_profiles/profiles_1d/electrons/density',
                        unit:'m^-3',documentation:'Electron number density'})
                        CREATE (s:StandardNameSource {id:'dd:core_profiles/profiles_1d/electrons/density',
                        source_type:'dd',source_id:dd.id,status:'composed',produced_sn_id:old.id})
                        CREATE (s)-[:FROM_DD_PATH]->(dd)
                        CREATE (s)-[:PRODUCED_NAME]->(old)
                        CREATE (dd)-[:HAS_STANDARD_NAME]->(old)
                        CREATE (u:Unit {id:'m^-3',symbol:'m^-3'})
                        CREATE (old)-[:HAS_UNIT]->(u)
                        CREATE (target)-[:HAS_UNIT]->(u)
                        CREATE (dd)-[:HAS_UNIT]->(u)""",
                        stage=target_stage,
                    )
                    if branch:
                        gc.query("""MATCH (old:StandardName {id:'electron_number_density'})
                            CREATE (other:StandardName {id:'electron_density_at_plasma_boundary',
                            name_stage:'accepted',status:'draft',origin:'pipeline',unit:'m^-3'})
                            CREATE (other)-[:REFINED_FROM]->(old)""")

            def state():
                with client() as gc:
                    names = gc.query(
                        "MATCH (n:StandardName) RETURN n.id AS id,n.name_stage AS stage"
                    )
                    edges = gc.query("""MATCH (a:StandardName)-[:REFINED_FROM]->(b:StandardName)
                        RETURN a.id AS successor,b.id AS predecessor""")
                    cycles, components = classify({n["id"]: n for n in names}, edges)
                    receipts = gc.query("""MATCH (c:StandardNameChange) RETURN c.id AS id,
                        c.operation AS operation,c.from_name AS from_name,c.to_name AS to_name""")
                    return {
                        "cycles": cycles,
                        "components": components,
                        "edges": edges,
                        "receipts": receipts,
                    }

            with patch.object(edit, "GraphClient", client):
                seed("accepted")
                refused = edit.supersede_into(
                    "electron_number_density", "electron_density"
                )
                assert not refused["ok"] and "cycle" in refused["reason"]
                assert state()["cycles"] == []
                results.append(
                    {
                        "case": "fold_accepted_ancestor",
                        "result": refused,
                        "after": state(),
                    }
                )
                print("REFUSAL", json.dumps(results[-1]), flush=True)

                seed()
                folded = edit.supersede_into(
                    "electron_number_density", "electron_density"
                )
                after = state()
                results.append(
                    {
                        "case": "fold_tombstoned_ancestor",
                        "result": folded,
                        "after": after,
                    }
                )
                print("REPRODUCTION", json.dumps(results[-1]), flush=True)
                assert folded["ok"] and len(after["cycles"]) == 1

                seed()
                try:
                    graph_refinement.persist_refined_name(
                        old_name="electron_number_density",
                        new_name="electron_density",
                        description="Electron number density.",
                        unit="m^-3",
                        expected_old_stage="accepted",
                        services=services,
                    )
                except graph_refinement.RefinedNamePersistenceRefusal as exc:
                    refusal = str(exc)
                else:
                    raise AssertionError("superseded target was not refused")
                assert state()["cycles"] == []
                results.append(
                    {
                        "case": "refine_superseded_ancestor",
                        "refusal": refusal,
                        "after": state(),
                    }
                )
                print("REFUSAL", json.dumps(results[-1]), flush=True)

                seed("pending")
                refined = graph_refinement.persist_refined_name(
                    old_name="electron_number_density",
                    new_name="electron_density",
                    description="Electron number density.",
                    unit="m^-3",
                    expected_old_stage="accepted",
                    services=services,
                )
                after = state()
                results.append(
                    {
                        "case": "refine_pending_ancestor",
                        "result": refined,
                        "after": after,
                    }
                )
                print("REPRODUCTION", json.dumps(results[-1]), flush=True)
                assert len(after["cycles"]) == 1

                seed(branch=True)
                refined = graph_refinement.persist_refined_name(
                    old_name="electron_number_density",
                    new_name="electron_density_at_pedestal_top",
                    description="Electron number density.",
                    unit="m^-3",
                    expected_old_stage="accepted",
                    services=services,
                )
                after = state()
                results.append(
                    {
                        "case": "refine_existing_live_branch",
                        "result": refined,
                        "after": after,
                    }
                )
                print("REPRODUCTION", json.dumps(results[-1]), flush=True)
                assert any(len(c["live_tips"]) == 2 for c in after["components"])
            (run_dir / "writer-reproductions.json").write_text(
                json.dumps(results, indent=2) + "\n"
            )
        finally:
            # The process group was created by this helper; only that group is stopped.
            print(
                f"DISPOSABLE stop owned group {process.pid}; project service untouched",
                flush=True,
            )
            subprocess.run(
                ["ps", "-eo", "pid,ppid,pgid,etime,args"],
                stdout=(run_dir / "disposable-processes.txt").open("w"),
                check=True,
            )
            temp_neo4j.stop_temp_neo4j(process)
            shutil.copyfile(server_log, run_dir / "disposable-server.log")
            print(f"DISPOSABLE removed scratch {root}", flush=True)
    print(f"WRITERS scenarios={len(results)} confirmed_open=3 refused_controls=2")


WRITERS = {
    "fold_identity": {
        "file": "imas_codex/standard_names/edit.py",
        "line": 1553,
        "guard_line": 2323,
        "receipt_line": 1469,
        "status": "open: tombstoned-ancestor fold creates a reciprocal cycle; accepted ancestor is refused",
        "reproduction": "fold_tombstoned_ancestor",
        "history": "a4b21a4e6d190063f9ab8169723ef35b73ea162a: edit.py:1241 writes the edge; :1654 exempts closing_straight_chain from the cycle refusal",
    },
    "refinement": {
        "file": "imas_codex/standard_names/graph_store/graph_refinement.py",
        "line": 1181,
        "guard_line": 822,
        "receipt_line": 1350,
        "status": "partly open: superseded/accepted targets refuse; a pending ancestor can cycle; an existing live branch can gain another tip",
        "reproduction": "refine_pending_ancestor; refine_existing_live_branch",
        "history": "1f95b7565be13a6355d24d8299bf213d77b67315: imas_codex/standard_names/graph_ops.py:9526; atomic persistence added in b045fa23e; current owner extracted by ec4257ab3",
    },
    "regeneration": {
        "file": "imas_codex/standard_names/graph_store/graph_supersession.py",
        "line": 431,
        "guard_line": 355,
        "receipt_line": 265,
        "status": "cycle predicate present in transaction; not independently exercised in this census",
        "reproduction": None,
        "history": "1f95b7565be13a6355d24d8299bf213d77b67315: imas_codex/standard_names/graph_ops.py:9814",
    },
    "fold": {
        "file": "imas_codex/standard_names/graph_store/graph_supersession.py",
        "line": 636,
        "guard_line": 624,
        "receipt_line": 614,
        "status": "cycle predicate present in transaction; not independently exercised in this census",
        "reproduction": None,
    },
    "reclaim": {
        "file": "imas_codex/standard_names/edit.py",
        "line": 3333,
        "guard_line": 3434,
        "receipt_line": 3334,
        "status": "closing-lineage reclaim omits the reverse edge; different-locus cycle refuses before mutation; two existing disposable tests pass",
        "reproduction": "reclaim-controls.log",
    },
    "unresolved": {
        "file": None,
        "line": None,
        "status": "original creator cannot be identified from the surviving graph records; no stamped edge properties",
        "reproduction": None,
    },
}

# These are proposed semantic dispositions, never an authorization to write.
BRANCH_REPAIRS = {
    "breakdown_initial_time": "Keep breakdown_initial_time; propose signed supersession of exhausted plasma_breakdown_time onto it after checking the shared summary/time_breakdown/value binding. Preserve both ledgers.",
    "cumulative_inside_flux_surface_electron_particle_source_rate": "Keep the integrated particle source rate and the volumetric source rate as separate quantities. Split the shared historical ancestry into distinct identity histories; do not fold the tips together.",
    "peak_wave_current_of_antenna_strap_amplitude": "Adjudicate peak versus RMS against the strap-current DD definition; both currently bind the same path. Keep module-current amplitude distinct. Prefer the accepted RMS holder only if the DD confirms RMS, then supersede the exhausted peak candidate with a signed source disposition.",
    "derivative_of_volume_of_flux_surface_with_respect_to_toroidal_flux_coordinate": "Split volume derivative, inverse major radius and metric component histories by their actual DD fields. Keep accepted dvolume_drho_tor and gm9 holders; adjudicate the exhausted derived metric against its derivation authority before retiring it. Remove the two cyclic subgraphs only under their per-edge history receipt.",
    "energy_flux_normalized_due_to_e_cross_b_drift": "Keep electrostatic and magnetic-flutter energy-flux channels separate; their DD leaves name different mechanisms. Archive the shared generic ancestry as history rather than declaring either tip the other's successor.",
    "energy_flux_normalized_due_to_perturbed_parallel_vector_potential": "Compare the two parallel-vector-potential energy-flux names against all frame and integration dimensions shown in the binding table. If equivalent, propose the shorter accepted energy_flux_normalized_due_to_perturbed_parallel_vector_potential with a signed source union; otherwise split the histories. Do not infer equivalence from their shared ancestor.",
    "extent_of_soft_xray_detector": "The two tips bind camera_dimensions. Rule whether the stored quantity is a detector extent or camera dimension, then signed-supersede exhausted total_size_of_camera onto the confirmed holder. Its name score is 0.525 versus 0.975 on extent_of_soft_xray_detector, which supports review priority but does not settle the physics.",
    "first_measurement_direction_unit_vector_of_shatter_cone": "Keep direction/x and unit_vector_major/x separate: one is the measurement direction, the other the major-axis direction. Split the shared x-unit-vector ancestry, preserving reviews on the quantity they assessed.",
    "flux_surface_averaged_toroidal_current_density": "Keep equilibrium j_phi and source j_parallel distinct. Separate the generic current-density ancestry; review the exhausted parallel wave-driven source spelling against its source/channel before any rename.",
    "flux_surface_averaged_tungsten_density_at_plasma_boundary": "Keep local tungsten density and its flux-surface average as distinct holders. Archive the ambiguous tungsten_density_at_plasma_boundary tombstone with the average history, restore its lost recorded-path evidence from receipts if available, then reclaim its spelling for tungsten_density_at_separatrix with history kept apart. Do not fold the local value into the average.",
    "radial_ion_charge_state_momentum_flux": "Split ion-state transport flux, total-ion transport flux, ion source, neutral source and neutral-state source by species, charge-state aggregation and DD source/transport role. Keep all distinct bound quantities; adjudicate the total-ion flux holder's mixed source and transport bindings before any fold.",
    "length_of_poloidal_magnetic_field_probe": "Keep poloidal and toroidal magnetic probe lengths separate. Replace the generic predecessor's ambiguous successor history with separate component histories, retaining each current producer.",
    "particle_flux_normalized_due_to_e_cross_b_drift": "Keep electrostatic and magnetic-flutter particle-flux channels separate; split their generic normalized_particle_flux ancestry. The exhausted electrostatic candidate needs its own review, not supersession into the magnetic channel.",
    "poloidal_ion_charge_state_energy_convection_velocity": "Compare removed edge_transport and current plasma_transport v_pol paths via DD rename history. If they are the same quantity across DD versions, move sources and signed-supersede the reviewed older spelling onto accepted poloidal_ion_charge_state_energy_convection_velocity; preserve a distinct quantity if the DD definitions differ.",
    "spectral_calibration_factor_of_spectrometer_channel": "Separate optical etendue from instrument/calibration response. Audit the calibration holder's wavelength-gain and instrument-intensity paths individually, and preserve the derived etendue parent with its children. Do not fold those two quantities together.",
    "toroidal_angle_of_measurement_position": "Keep the general measurement-position toroidal angle and launcher-mirror geometry coordinate distinct until their full DD source semantics are adjudicated. Split misleading generic ancestry; do not transfer all 25 measurement bindings onto the mirror.",
    "vertical_coordinate_of_camera": "Keep camera-centre vertical coordinate and camera-direction unit-vector component separate (different physical dimensions). Archive the reciprocal direction-spelling history independently of the positional name.",
    "vertical_ion_charge_state_momentum_flux": "Keep transport momentum flux and volumetric momentum source separate; their DD branches are plasma_transport and plasma_sources. Split the shared state-momentum ancestry without moving either source set.",
}


def draw_topology(run_dir):
    """Show two measured components, with arrow direction matching the graph."""
    import re
    import textwrap

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch

    plt.style.use(Path.home() / ".config/matplotlib/stylelib/data-ink.mplstyle")
    observed = json.loads((run_dir / "graph-snapshot.json").read_text())
    observed_names = {n["id"]: n["stage"] for n in observed["nodes"]}
    fig = plt.figure(figsize=(14, 7.2), dpi=100)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set(xlim=(0, 1400), ylim=(0, 720))
    ax.axis("off")
    labels = []

    def label(x, y, identifier, stage, width=32):
        assert observed_names[identifier] == stage, identifier
        wrapped = "\n".join(textwrap.wrap(identifier.replace("_", " "), width=width))
        artist = ax.text(
            x,
            y,
            wrapped + "\n[" + stage + "]",
            ha="center",
            va="center",
            fontsize=14,
            color="#303030" if stage != "superseded" else "#666666",
        )
        labels.append(artist)

    def arrow(a, b, curve=0):
        ax.add_patch(
            FancyArrowPatch(
                a,
                b,
                arrowstyle="-|>",
                mutation_scale=17,
                linewidth=1.8,
                color="#929292",
                connectionstyle=f"arc3,rad={curve}",
                zorder=1,
            )
        )

    label(300, 560, "accumulated_carbon_count_due_to_gas_injection", "accepted")
    label(300, 175, "carbon_count_accumulated_due_to_gas_injection", "superseded")
    arrow((245, 490), (245, 248), 0.18)
    arrow((355, 248), (355, 490), 0.18)
    label(1010, 610, "tungsten_density_at_plasma_boundary", "superseded", 29)
    label(795, 365, "tungsten_density_at_separatrix", "accepted", 27)
    label(
        1190,
        365,
        "flux_surface_averaged_tungsten_density_at_plasma_boundary",
        "reviewed",
        29,
    )
    label(
        1190,
        105,
        "tungsten_density_flux_surface_averaged_at_plasma_boundary",
        "superseded",
        29,
    )
    arrow((800, 425), (955, 555))
    arrow((1165, 440), (1055, 555))
    arrow((1190, 184), (1190, 283))
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes = [(a, a.get_window_extent(renderer)) for a in labels]
    for a, box in boxes:
        if (
            box.x0 < 0
            or box.y0 < 0
            or box.x1 > fig.bbox.width
            or box.y1 > fig.bbox.height
        ):
            raise AssertionError("clipped figure label: " + a.get_text())
    for i, (a, box) in enumerate(boxes):
        for b, other in boxes[i + 1 :]:
            if box.overlaps(other):
                raise AssertionError(
                    "overlapping figure labels: " + a.get_text() + " / " + b.get_text()
                )
        for arrow_artist in ax.patches:
            if box.overlaps(arrow_artist.get_window_extent(renderer)):
                raise AssertionError("figure label overlaps an arrow: " + a.get_text())
    target = ROOT / "docs/figures/sn-integrity-backlog" / STEM / "topology.svg"
    target.parent.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({"svg.fonttype": "none"}):
        fig.savefig(target, format="svg")
    svg = target.read_text()
    svg = re.sub(
        r'width="[^"]+pt" height="[^"]+pt"',
        'width="31.1cm" height="16cm"',
        svg,
        count=1,
    )
    target.write_text(svg)
    fig.savefig(run_dir / "topology-preview.png", dpi=100)
    plt.close(fig)
    print(
        "FIGURE labels=6 clipping=0 overlap=0 arrows=5 preview="
        + str(run_dir / "topology-preview.png")
    )


def report(run_dir, out_dir=HERE):
    """Write a normalized, lossless case index and a readable landing fragment."""
    from collections import Counter
    from html import escape

    raw_path = run_dir / "graph-snapshot.json"
    raw_bytes = raw_path.read_bytes()
    data = json.loads(raw_bytes)
    reproductions = json.loads((run_dir / "writer-reproductions.json").read_text())
    assert len(reproductions) == 5
    assert (run_dir / "reproduce-scratch.log").read_text().rstrip().endswith("EXIT=0")
    reclaim_log = (run_dir / "reclaim-controls.log").read_text()
    assert "2 passed" in reclaim_log and reclaim_log.rstrip().endswith("EXIT=0")
    nodes = {n["id"]: n for n in data["nodes"]}
    cycles, components = classify(nodes, data["edges"])
    assert cycles == data["cycles"], "cycle membership differs from retained snapshot"
    assert components == data["components"], (
        "lineage membership differs from retained snapshot"
    )
    cases = data["cases"]
    selected = sorted({n for c in cases for n in c["members"]})
    name_index = {n: i for i, n in enumerate(selected)}
    changes = defaultdict(list)
    for i, row in enumerate(data["changes"]):
        c = row["properties"]
        changes[c.get("to_name"), c.get("from_name")].append((i, c))
    sources = defaultdict(list)
    for s in data["sources"]:
        sources[s["name"]].append(s)
    projections = defaultdict(list)
    for p in data["projections"]:
        projections[p["name"]].append(p["path"])
    receipts, receipt_index, edges = [], {}, []
    for e in data["edges"]:
        if e["successor"] not in name_index:
            continue
        relevant = changes[e["successor"], e["predecessor"]]
        refs = []
        for raw_index, c in relevant:
            if c["id"] not in receipt_index:
                receipt_index[c["id"]] = len(receipts)
                receipts.append(
                    [
                        c["id"],
                        c.get("operation"),
                        c.get("changed_at"),
                        c.get("run_id"),
                        c.get("manifest_sha256"),
                        raw_index,
                    ]
                )
            refs.append(receipt_index[c["id"]])
        producers = [
            (i, c)
            for i, c in relevant
            if c["operation"]
            in {"fold_identity", "human_edit", "refine", "regenerate", "fold"}
        ]
        if producers:
            _, creator = min(
                producers, key=lambda item: str(item[1].get("changed_at", ""))
            )
            operation = creator["operation"]
            writer = (
                "refinement"
                if operation in {"refine", "human_edit"}
                else ("regeneration" if operation == "regenerate" else operation)
            )
            evidence = "matching_operation_receipt"
            if operation == "fold_identity":
                detail = json.loads(creator["reason"])
                was_there = any(
                    r.get("type") == "REFINED_FROM"
                    and r.get("start_id") == e["successor"]
                    and r.get("end_id") == e["predecessor"]
                    for r in detail["before"]["relationships"]
                )
                if not was_there:
                    evidence = "before_after_proves_edge_created"
                else:
                    evidence = "receipt_reuses_preexisting_edge"
        elif any(c["operation"] == "backfill_refine" for _, c in relevant):
            writer = "unresolved"
            evidence = "backfill_records_direction_but_not_original_creator"
        else:
            writer = "unresolved"
            evidence = (
                "no_creation_receipt" if not relevant else "noncreating_change_only"
            )
        edges.append(
            [
                name_index[e["successor"]],
                name_index[e["predecessor"]],
                writer,
                evidence,
                refs,
                e["edge_id"],
            ]
        )
    names = []
    for n in selected:
        row = nodes[n]
        names.append(
            [
                n,
                row["stage"],
                row.get("unit"),
                row.get("name_score"),
                row.get("docs_score"),
                row.get("recorded_paths") or [],
                [
                    [s["source"], s["path"], s["status"], s["source_type"]]
                    for s in sources[n]
                ],
                projections[n],
                (
                    (row.get("description") or "")
                    if row["stage"] != "superseded" or n in SPECIAL
                    else None
                ),
            ]
        )
    repair_cases = []
    for c in cases:
        members = set(c["members"])
        case_edges = [i for i, e in enumerate(edges) if selected[e[0]] in members]
        cycle_groups = [group for group in data["cycles"] if set(group) <= members]
        proposals = []
        suggested_remove = []
        for group in cycle_groups:
            live = [n for n in group if nodes[n]["stage"] != "superseded"]
            if len(live) == 1:
                holder = live[0]
                stale = [
                    i
                    for i in case_edges
                    if selected[edges[i][1]] == holder
                    and selected[edges[i][0]] in group
                ]
                suggested_remove.extend(stale)
                proposals.append(
                    "Retain "
                    + holder
                    + " as the current identity. After ruling, archive the historical reverse edge(s) into a signed change receipt and remove the listed edge(s) pointing from a superseded member onto this live identity; keep sources, documents and paid reviews on their current meanings. Recheck the whole component for cycles and source loss."
                )
            elif not live:
                proposals.append(
                    "This cyclic group has no live member: "
                    + ", ".join(group)
                    + ". Preserve the full edge snapshot and reconstruct chronology from receipts/backups; propose archiving enough obsolete edges to make the ancestry acyclic, retaining a route to "
                    + (
                        ", ".join(c["live_tips"])
                        or "an explicitly adjudicated successor"
                    )
                    + ". Do not choose an edge from lexicographic order or a backfill timestamp."
                )
            else:
                proposals.append(
                    "Several live names lie on this cycle: "
                    + ", ".join(live)
                    + ". Rule each source/meaning first, split distinct identities and archive obsolete history edges; no automatic single-tip fold is justified."
                )
        for key, text in BRANCH_REPAIRS.items():
            if key in c["live_tips"]:
                proposals.append(text)
                break
        if "argon_density_at_plasma_boundary" in members:
            proposals.append(
                "Argon: the tombstone records separatrix_average and its live averaged holder has that same locus. Propose history-apart archival of this tombstone before reclaiming argon_density_at_plasma_boundary for the local argon_density_at_separatrix. Keep the accepted averaged holder and its review history separate. Resolve its reciprocal average-spelling cycle independently."
            )
        if "argon_density_at_separatrix" in members:
            proposals.append(
                "Argon local value: keep its separatrix/n_i/argon/value source and propose the ruled argon_density_at_plasma_boundary spelling only after the average tombstone history is archived separately; the local and averaged names must not be folded."
            )
        if (
            "flux_surface_averaged_deuterium_tritium_density_at_plasma_boundary"
            in members
        ):
            proposals.append(
                "Deuterium-tritium: both prefix ancestors are superseded, while the infix holder is exhausted (name score 0.70) and its DD source is extracted. Preserve the source and histories, break the old reciprocal ancestry under a signed edge disposition, then reclaim flux_surface_averaged_deuterium_tritium_density_at_plasma_boundary through the sanctioned route and rerun review. Do not promote it from the stale 0.96875 ancestor score."
            )
        if "normalized_toroidal_flux_coordinate_at_plasma_boundary" in members:
            proposals.append(
                "Normalized coordinate: the source-migration receipt and human_edit receipt from 2026-11-07 identify the rename to flux_surface_averaged_normalized_toroidal_flux_coordinate_at_plasma_boundary. The tombstone now has an empty source_paths cache, not absent provenance. Recover its source from that receipt and the successor's separatrix_average/position/rho_tor_norm binding. Rule whether a flux-surface label evaluated at a surface should carry an averaging prefix before the history-apart reclaim; preserve both recorded DD fields."
            )
        if "normalized_toroidal_flux_coordinate_at_separatrix" in members:
            proposals.append(
                "Normalized local coordinate: keep the separatrix/position/rho_tor_norm producer; propose the ruled normalized_toroidal_flux_coordinate_at_plasma_boundary spelling after adjudicating the occupied target's recorded source and whether the two DD location labels are semantically equivalent. Do not infer a different quantity solely from the average container."
            )
        assert proposals, c
        repair_cases.append(
            {
                "anchor": STEM
                + "-"
                + hashlib.sha256("\0".join(c["members"]).encode()).hexdigest()[:10],
                "members": [name_index[n] for n in c["members"]],
                "live_members": [name_index[n] for n in c["live_members"]],
                "live_tips": [name_index[n] for n in c["live_tips"]],
                "cycles": [[name_index[n] for n in group] for group in cycle_groups],
                "edges": case_edges,
                "proposed_archive_edges": sorted(set(suggested_remove)),
                "proposed_repair": proposals,
            }
        )
    counts = Counter(e[3] for e in edges)
    pairs = {(e["successor"], e["predecessor"]) for e in data["edges"]}
    summary = {
        **data["inventory"],
        "cycle_components": len(data["cycles"]),
        "cycle_names": sum(map(len, data["cycles"])),
        "reciprocal_pairs": sum((b, a) in pairs for a, b in pairs) // 2,
        "multi_tip_components": sum(
            len(c["live_tips"]) > 1 for c in data["components"]
        ),
        "reported_components": len(cases),
        "reported_names": len(names),
        "reported_edges": len(edges),
        "edge_attribution": dict(counts),
        "unresolved_creator_edges": sum(e[2] == "unresolved" for e in edges),
        "spend_usd": 0,
        "live_graph_writes": 0,
    }
    report_data = {
        "observed_at": data["observed_at"],
        "source_revision": data["revision"],
        "raw_snapshot": str(raw_path),
        "raw_snapshot_sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "summary": summary,
        "cycle_detector": {
            "owner": "imas_codex.standard_names.catalog_ordering._find_cyclic_components",
            "source_sha256": hashlib.sha256(
                Path(catalog_ordering.__file__).read_bytes()
            ).hexdigest(),
            "snapshot_replay_agrees": True,
        },
        "definitions": {
            "direction": "successor -> predecessor",
            "lineage": "weakly connected component of StandardName REFINED_FROM edges",
            "live": "name_stage != superseded, including exhausted; status is not substituted",
            "live_tip": "live member with no other live descendant in reverse REFINED_FROM traversal",
            "cycle": "strongly connected group; every member returns to itself on one or more edges",
            "provenance": "matching event receipts identify operation-level writers, not the exact deployed commit; a backfill is not an edge creation event",
        },
        "columns": {
            "names": [
                "id",
                "stage",
                "unit",
                "name_score",
                "docs_score",
                "recorded_paths",
                "producers(source_id,path,status,type)",
                "projections",
                "description",
            ],
            "edges": [
                "successor_name_index",
                "predecessor_name_index",
                "writer_key",
                "evidence",
                "receipt_indices",
                "element_id",
            ],
            "receipts": [
                "id",
                "operation",
                "changed_at",
                "run_id",
                "manifest_sha256",
                "raw_changes_index",
            ],
        },
        "names": names,
        "edges": edges,
        "receipts": receipts,
        "writers": WRITERS,
        "cases": repair_cases,
        "cycle_cross_check": {
            "query": next(
                q["cypher"]
                for q in data["queries"]
                if q["label"] == "cycle-path-cross-check"
            ),
            "count": len(data["cycle_cross_check"]),
            "agrees": True,
        },
        "tip_cross_check": {
            "query": next(
                q["cypher"]
                for q in data["queries"]
                if q["label"] == "live-tip-cross-check"
            ),
            "count": len(data["tip_cross_check"]),
            "agrees": True,
        },
        "reproduction_file": str(run_dir / "writer-reproductions.json"),
        "limitations": [
            f"{sum(not e['properties'] for e in data['edges'])} of {summary['edges']} REFINED_FROM edges have empty property maps with no writer id, receipt id or creation timestamp.",
            "Unresolved creators are explicit missing evidence, never inferred from a name, later maintenance event or backfill timestamp.",
            "This snapshot is time-bounded; graph writers may change the corpus after observed_at. Inventory before/after and independent traversals agreed during capture.",
            "Proposed repairs require lead adjudication. No live identity, source, edge, review or documentation was changed.",
            "Descriptions of superseded names outside the named special cases are retained in the raw snapshot, omitted from the compact name table.",
        ],
    }
    encoded = json.dumps(report_data, separators=(",", ":"), ensure_ascii=False) + "\n"
    assert len(encoded.encode()) <= 300000, (
        f"report JSON too large: {len(encoded.encode())}"
    )
    (out_dir / f"{STEM}.json").write_text(encoded)
    (run_dir / "census-expanded.json").write_text(
        json.dumps(report_data, indent=2, ensure_ascii=False) + "\n"
    )

    def code(s):
        return "<code>" + escape(str(s)) + "</code>"

    def name(i):
        return code(names[i][0])

    html = [
        f'<section id="{STEM}">',
        "<h2>Refinement cycles and live-tip lineage census</h2>",
        f"<p>Read-only graph snapshot at {escape(data['observed_at'])}: <strong>{summary['cycle_names']} names in {summary['cycle_components']} cyclic components</strong> ({summary['reciprocal_pairs']} reciprocal pairs), and <strong>{summary['multi_tip_components']} lineages with more than one live tip</strong>, across {summary['names']} StandardNames and {summary['edges']} REFINED_FROM edges. The plan recorded 17 reciprocal cases; this graph-wide instrument finds the larger population reported here. Independent Cypher finds the same {len(data['cycle_cross_check'])} cycle members and {len(data['tip_cross_check'])} live tips. Spend USD 0 of USD 0; live writes 0.</p>",
        f'<p>This record lists {len(cases)} components and {len(names)} names, including all cycle and multiple-tip cases plus the four specifically requested families. The <a href="/imas-codex/evidence/fragments/sn-integrity-backlog/{STEM}.json">machine-readable census</a> holds every name, chain edge, producing source, recorded source path, review score, receipt reference, writer location and proposed action. Numeric name and receipt indices refer to its declared column tables.</p>',
        f"<p><strong>Attribution limit:</strong> {summary['unresolved_creator_edges']} of the {len(edges)} reported edges have no original creation receipt that identifies their writer. {sum(not e['properties'] for e in data['edges'])} of {summary['edges']} graph edges have empty property maps. Backfilled refinement records establish a historical direction but do not identify who originally wrote an edge. Every such edge is marked unresolved below. A matching receipt maps an operation to today's writer; it does not prove which deployed source revision ran then.</p>",
        "<p>Arrows below follow the stored direction, successor → predecessor. A lineage is a weakly connected component. “Live” means name_stage is not superseded, so exhausted names are included. A live tip has no other live descendant; all live members are listed as well, because reclaimed ancestors and intermediate live identities need not be terminal tips. Cycles are strongly connected groups rather than an exponentially large listing of equivalent closed walks.</p>",
        f'<figure><img src="/imas-codex/figures/sn-integrity-backlog/{STEM}/topology.svg" alt="Reciprocal carbon spelling edges and the branched tungsten lineage"/><figcaption>Arrows are stored REFINED_FROM edges, from successor to predecessor. Left: the carbon spelling reversal retains both directions. Right: tungsten has separate local and averaged live holders; the average is currently reviewed after reclaim while its infix spelling is superseded. A graph leaf and a live tip differ here: the infix node is a superseded leaf. Names are spaced for readability; exact identifiers are in the tables.</figcaption></figure>',
        "<p>Cycle membership is computed by <code>catalog_ordering._find_cyclic_components</code>, the same owner used by catalog ordering. Replaying the retained snapshot through this owner reproduces every cycle and lineage. The census keeps its own live-tip predicate because it includes exhausted names. The replay gate also requires delegation and propagation of a planted owner refusal. The parent implementation passes topology and snapshot checks but fails both ownership checks (<code>reuse-negative.log</code>, <code>EXIT=1</code>); the delivered runner passes all four (<code>reuse-head.log</code>, <code>EXIT=0</code>).</p>",
        "<h3>Writers and the history they retain</h3>",
        '<p>The September fold design deliberately retained reversed rename edges as complete history, as documented in <a href="/imas-codex/evidence/sn-grammar-refinement/fold-guard-divergence.md">the fold guard evidence</a>. The current reclaim requires acyclic ancestry. Repair therefore needs to distinguish a current successor from historical succession: retain immutable change records and paid reviews, then archive only the obsolete lineage edges under a signed disposition. A blanket deletion of every newest edge would sometimes discard the current identity.</p>',
        "<table><thead><tr><th>Writer</th><th>Current code</th><th>Measured state</th></tr></thead><tbody>",
    ]
    for key, w in WRITERS.items():
        if key == "unresolved":
            continue
        html.append(
            "<tr><td>"
            + code(key)
            + "</td><td>"
            + code(f"{w['file']}:{w['line']}")
            + "<br>guard "
            + str(w["guard_line"])
            + "</td><td>"
            + escape(w["status"])
            + "</td></tr>"
        )
    html += [
        "</tbody></table>",
        "<p>Disposable production-writer probe: five scenarios, three open behaviors and two refusing controls, EXIT=0. The accepted-ancestor fold refused with “already descends … (REFINED_FROM cycle)”; the superseded-target refine refused with “reason=successor_lifecycle_collision”. Changing only the fold target to a tombstone admitted the write and committed both directions with a fold_identity receipt. A pending ancestor passed refinement and committed both directions; a predecessor with an existing live descendant gained a second live tip. No attachment or lineage writer was stubbed; only the unused run-counter callback was inert.</p>",
        "<p>The two focused existing reclaim tests also exercise successful closing-lineage reclaim and in-transaction refusal of a different-locus cycle. Their log is retained beside the writer probe. Regeneration and stub-fold cycle guards were inspected at their mutation predicates; no new live behavior is claimed for those paths.</p>",
        "<h3>Per-component chains, evidence and proposed repairs</h3>",
    ]
    for c in repair_cases:
        title = (
            "cycle"
            if c["cycles"]
            else "multiple tips"
            if len(c["live_tips"]) > 1
            else "requested family"
        )
        html.append(
            f'<details id="{c["anchor"]}"><summary>{escape(title)}: {escape(names[c["members"][0]][0])} ({len(c["members"])} names)</summary>'
        )
        html.append(
            "<p>Live tips: "
            + (", ".join(name(i) for i in c["live_tips"]) or "none")
            + ". All live members: "
            + (", ".join(name(i) for i in c["live_members"]) or "none")
            + ".</p>"
        )
        if c["cycles"]:
            html.append(
                "<p>Cycle members: "
                + "; ".join(", ".join(name(i) for i in group) for group in c["cycles"])
                + ".</p>"
            )
        html.append("<ul>")
        for i in c["members"]:
            n = names[i]
            paths = [p[1] for p in n[6]]
            shown = ", ".join(code(p) for p in paths[:3]) or "none"
            if len(paths) > 3:
                shown += f"; {len(paths) - 3} further bindings in JSON"
            html.append(
                "<li>"
                + name(i)
                + " — "
                + escape(n[1])
                + f"; name/docs scores {n[3]}/{n[4]}; producers: "
                + shown
                + ".</li>"
            )
        html.append(
            "</ul><table><thead><tr><th>Stored edge (successor → predecessor)</th><th>Writer / evidence</th><th>Change receipts</th></tr></thead><tbody>"
        )
        for ei in c["edges"]:
            e = edges[ei]
            refs = (
                ", ".join(f'<a href="#{STEM}-receipt-{r}">{r}</a>' for r in e[4])
                or "none"
            )
            html.append(
                "<tr><td>"
                + name(e[0])
                + " → "
                + name(e[1])
                + "</td><td>"
                + escape(e[2] + "; " + e[3].replace("_", " "))
                + "</td><td>"
                + refs
                + "</td></tr>"
            )
        html.append("</tbody></table>")
        for proposal in c["proposed_repair"]:
            html.append("<p><strong>Proposed:</strong> " + escape(proposal) + "</p>")
        if c["proposed_archive_edges"]:
            html.append(
                "<p>Candidate obsolete edges for signed archival: "
                + "; ".join(
                    name(edges[i][0]) + " → " + name(edges[i][1])
                    for i in c["proposed_archive_edges"]
                )
                + ". These are candidates, not an executable authority.</p>"
            )
        html.append("</details>")
    html += [
        f'<h3 id="{STEM}-receipts">Receipt index</h3>',
        "<p>Full reasons and before/after snapshots remain in "
        + code(raw_path)
        + ". Each JSON receipt records the exact index into that snapshot's changes array. The snapshot SHA-256 is "
        + code(report_data["raw_snapshot_sha256"])
        + ". Noncreating maintenance receipts are retained as context, never promoted to edge creators.</p>",
        "<ul>",
    ]
    for i, r in enumerate(receipts):
        html.append(
            f'<li id="{STEM}-receipt-{i}">{i}: '
            + code(r[0])
            + " "
            + escape(str(r[1]))
            + "; "
            + escape(str(r[2])[:10])
            + ".</li>"
        )
    html += [
        "</ul>",
        f'<h3 id="{STEM}-verification">Verification and durable artifacts</h3>',
        "<p>Inventory, cycle membership and live-tip membership are independently checked. Planted controls cover a two-node cycle, a self-loop, a two-tip branch and an isolated clean identity. Every graph read uses a ten-second transaction timeout and rolls back. Capture counters agree before and after; no source or lifecycle writes are present in capture mode.</p>",
        "<pre>" + escape(report_data["cycle_cross_check"]["query"]) + "</pre>",
        "<pre>" + escape(report_data["tip_cross_check"]["query"]) + "</pre>",
        "<p>Run record: "
        + code(run_dir)
        + ". Logs: "
        + code("capture-complete.log")
        + ", "
        + code("reproduce-scratch.log")
        + ", "
        + code("reclaim-controls.log")
        + " and "
        + code("reuse-head.log")
        + ". The invocation revision and imported source paths are recorded in the logs. Large snapshots and raw receipts are retained in the run directory, not duplicated into the repository.</p>",
        "</section>",
    ]
    html_text = "\n".join(html) + "\n"
    assert len(html_text.encode()) <= 300000, "fragment exceeds artifact size ceiling"
    (out_dir / f"{STEM}.html").write_text(html_text)
    assert all(c["anchor"].startswith(STEM) for c in repair_cases)
    assert {names[n][0] for c in repair_cases for g in c["cycles"] for n in g} == {
        r["id"] for r in data["cycle_cross_check"]
    }
    assert (
        sum(len(c["live_tips"]) > 1 for c in repair_cases)
        == summary["multi_tip_components"]
    )
    print("REPORT " + json.dumps(summary, sort_keys=True), flush=True)
    print(
        f"ARTIFACT json_bytes={len(encoded.encode())} html_bytes={len(html_text.encode())} complete_chains={len(cases)} receipts={len(receipts)}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Directory containing the snapshot, receipts and run logs",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=HERE,
        help="Destination directory for the HTML and JSON reports",
    )
    parser.add_argument("--capture", action="store_true")
    parser.add_argument("--reproduce", action="store_true")
    parser.add_argument("--report", action="store_true")
    parser.add_argument(
        "--figure",
        action="store_true",
        help="Regenerate the topology SVG and its run-directory preview",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Check shared owner use and replay the saved snapshot; no graph access",
    )
    parser.add_argument(
        "--verify-runner",
        type=Path,
        help="Verify another runner's classifier as a negative control",
    )
    args = parser.parse_args()
    if args.verify_runner and not args.verify:
        parser.error("--verify-runner requires --verify")
    if args.verify:
        result = verify(args.run_dir, args.verify_runner)
        if result:
            return result
    controls()
    if args.capture:
        capture(args.run_dir)
    if args.reproduce:
        reproduce(args.run_dir)
    if args.figure:
        draw_topology(args.run_dir)
    if args.report:
        args.out.mkdir(parents=True, exist_ok=True)
        report(args.run_dir, args.out)
    return 0


if __name__ == "__main__":
    print(
        f"# revision: {subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()} tree: {ROOT} command: {shlex.join([sys.executable, *sys.argv])}",
        flush=True,
    )
    print(f"# module: {__file__}; cwd: {Path.cwd().resolve()}", flush=True)
    print(
        f"# runner_sha256: {hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}",
        flush=True,
    )
    print(
        f"# cycle_owner_module: {catalog_ordering.__file__}; sha256: "
        f"{hashlib.sha256(Path(catalog_ordering.__file__).read_bytes()).hexdigest()}",
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
