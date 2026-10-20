"""Inspect one fixed-export family in an isolated Neo4j scratch instance.

This reproduces the runner's dump extraction and load without draining any
pool, so the evidence check makes no LLM calls and writes only to scratch.
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import sys
import tarfile
from pathlib import Path

from imas_codex.graph.client import GraphClient
from imas_codex.graph.temp_neo4j import start_temp_neo4j, stop_temp_neo4j
from imas_codex.standard_names import harmonize
from imas_codex.standard_names.campaign import (
    BatchOutcome,
    ConvergenceThresholds,
    evaluate_convergence,
)
from imas_codex.standard_names.defaults import DEFAULT_MIN_SCORE
from imas_codex.standard_names.families import fetch_pattern_families
from imas_codex.standard_names.harmonize import (
    _record_family_anchor,
    plan_family_applies,
)
from imas_codex.standard_names.review.pipeline import _cap_docs_scores_for_lint_errors

ROOT = Path(__file__).resolve().parents[4]
FRAGMENT = Path(__file__).resolve().parent
SELECTION_PATH = FRAGMENT / "ssog-anchor-ab" / "selection.json"
RUNNER_PATH = FRAGMENT / "ssog-anchor-ab-run.py"
FAMILY = "length_of_passive_loop_element"


def main() -> None:
    print(f"module.__file__={Path(harmonize.__file__).resolve()}", flush=True)
    print(f"resolved_cwd={Path.cwd().resolve()}", flush=True)
    if not os.environ.get("TMPDIR", "").startswith("/tmp/reckon-crew-scratch/"):
        raise RuntimeError("TMPDIR must be /tmp/reckon-crew-scratch/<run id>")
    scratch_parent = Path(os.environ["TMPDIR"])
    scratch_parent.mkdir(parents=True, exist_ok=True)
    selection = json.loads(SELECTION_PATH.read_text())
    export = Path(selection["source_export"])
    spec = importlib.util.spec_from_file_location("anchor_runner", RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load runner module from {RUNNER_PATH}")
    runner = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = runner
    spec.loader.exec_module(runner)
    reservation = runner.reserve_port_pair("diagnosis")
    scratch = scratch_parent / "scratch-instance"
    place = scratch / "instance"
    proc = None
    try:
        for name in ("data", "dumps", "logs", "run", "tmp"):
            (place / name).mkdir(parents=True, exist_ok=True)
        dump = place / "dumps" / "neo4j.dump"
        with tarfile.open(export, "r:gz") as archive:
            member = next(item for item in archive if item.name.endswith("/graph.dump"))
            with archive.extractfile(member) as source, dump.open("wb") as target:
                shutil.copyfileobj(source, target)
        print(f"export={export} dump_bytes={dump.stat().st_size}", flush=True)

        previous_bind = os.environ.get("APPTAINER_BIND")
        bind = f"{place / 'tmp'}:/tmp"
        os.environ["APPTAINER_BIND"] = (
            f"{previous_bind},{bind}" if previous_bind else bind
        )
        try:
            proc, neo4j_log = start_temp_neo4j(
                place, reservation.base, reservation.base + 1000
            )
        finally:
            if previous_bind is None:
                os.environ.pop("APPTAINER_BIND", None)
            else:
                os.environ["APPTAINER_BIND"] = previous_bind
        uri = f"bolt://127.0.0.1:{reservation.base}"
        print(f"scratch_uri={uri} neo4j_log={neo4j_log}", flush=True)

        with GraphClient(uri=uri) as gc:
            family = fetch_pattern_families(gc, seeds=[FAMILY])[0]
            ids = [member["id"] for member in family["members"]]
            _record_family_anchor(gc, ids, FAMILY)
            plans = plan_family_applies(gc, seeds=[FAMILY])
            if len(plans) != 1 or plans[0]["decision"] != "apply":
                raise RuntimeError(f"Expected one planned family apply: {plans}")
            plan = plans[0]
            print(
                "plan="
                + json.dumps(
                    {
                        "members": plan["members"],
                        "anchor": plan["anchor"],
                        "batches": plan["batches"],
                        "planned_calls": plan["planned_calls"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            members = family["members"]
            edges = gc.query(
                "MATCH (sn:StandardName) WHERE sn.id IN $ids "
                "OPTIONAL MATCH (sn)-[r]-(x) "
                "RETURN sn.id AS name, type(r) AS relationship, labels(x) AS labels, "
                "x.id AS target_id, x.path AS target_path, "
                "sn.source_paths AS source_paths, "
                "sn.source_bindings AS source_bindings, "
                "coalesce(x.documentation, '') AS documentation, "
                "coalesce(x.description, '') AS description "
                "ORDER BY name, relationship, target_id",
                ids=ids,
            )
            source_rows = [
                row
                for row in edges
                if row.get("labels") and "IMASNode" in row["labels"]
            ]
            print(
                "dd_source_probe="
                + json.dumps(
                    {
                        "family_members": ids,
                        "member_source_probe": [
                            {
                                "name": name,
                                "declared_source_paths": next(
                                    (
                                        row.get("source_paths")
                                        for row in edges
                                        if row["name"] == name
                                        and row.get("source_paths")
                                    ),
                                    None,
                                ),
                                "declared_source_bindings": next(
                                    (
                                        row.get("source_bindings")
                                        for row in edges
                                        if row["name"] == name
                                        and row.get("source_bindings")
                                    ),
                                    None,
                                ),
                                "linked_imas_nodes": [
                                    {
                                        "id": row["target_id"],
                                        "documentation_present": bool(
                                            row["documentation"].strip()
                                        ),
                                        "description_present": bool(
                                            row["description"].strip()
                                        ),
                                    }
                                    for row in source_rows
                                    if row["name"] == name
                                ],
                            }
                            for name in ids
                        ],
                        "direct_imasnode_edges": len(source_rows),
                        "with_documentation": sum(
                            bool(row["documentation"].strip()) for row in source_rows
                        ),
                        "with_description": sum(
                            bool(row["description"].strip()) for row in source_rows
                        ),
                        "sample": [
                            {
                                "name": row["name"],
                                "relationship": row["relationship"],
                                "target_id": row["target_id"],
                                "documentation_present": bool(
                                    row["documentation"].strip()
                                ),
                                "description_present": bool(row["description"].strip()),
                            }
                            for row in source_rows[:8]
                        ],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            scores = {row["id"]: row.get("reviewer_score_docs") for row in members}
            seed_score = scores.get(ids[0])
            scored = [{"id": ids[0], "reviewer_score": seed_score or 0.95}]
            _cap_docs_scores_for_lint_errors(
                scored,
                {ids[0]: "planted error-severity lint finding"},
                __import__("logging").LoggerAdapter(
                    __import__("logging").getLogger(__name__), {}
                ),
            )
            print(
                "lint_cap_probe="
                + json.dumps(
                    {
                        "threshold": DEFAULT_MIN_SCORE,
                        "score_before": seed_score or 0.95,
                        "score_after": scored[0]["reviewer_score"],
                        "cap_applied": scored[0].get("audit_cap_applied", False),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

        outcome = BatchOutcome(
            batch_index=0,
            requested=1,
            touched=1,
            accepted=0,
        )
        converged, reasons = evaluate_convergence(outcome, ConvergenceThresholds())
        print(
            "recorded_batch_one_gate="
            + json.dumps(
                {
                    "accept_rate": outcome.accept_rate,
                    "minimum": ConvergenceThresholds().min_docs_accept_rate,
                    "converged": converged,
                    "halt_reasons": reasons,
                    "next_batch": plan["batches"][1],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    finally:
        if proc is not None:
            stop_temp_neo4j(proc)
        if scratch.exists():
            shutil.rmtree(scratch)
        reservation.release()
        print(f"scratch_removed={scratch}", flush=True)


if __name__ == "__main__":
    main()
