"""Run one anchor-comparison arm of one family against a scratch graph copy.

The family selection, its anchors and the adoption rule are fixed in
``ssog-anchor-ab/selection.json`` and are never re-derived here. Each
invocation loads a fresh scratch copy of the archived export, runs a single
arm of a single family against it, records the arm, then stops the instance
and removes its tree. Exactly one output file is written, under
``ssog-anchor-ab/``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
import tarfile
import tempfile
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

from imas_codex.graph.client import GraphClient
from imas_codex.graph.temp_neo4j import start_temp_neo4j, stop_temp_neo4j
from imas_codex.standard_names.families import fetch_pattern_families
from imas_codex.standard_names.harmonize import (
    plan_family_applies,
    report_family_rows,
)

FRAGMENT_DIR = Path(__file__).resolve().parent
SELECTION = FRAGMENT_DIR / "ssog-anchor-ab" / "selection.json"
OUT_DIR = FRAGMENT_DIR / "ssog-anchor-ab"
PROJECT_VENV = "/home/ITER/mcintos/Code/imas-codex/.venv"
ARM_BUDGET_USD = 2.00
LOG_DIR = Path(
    os.environ.get(
        "SSOG_LOG_DIR",
        "/home/ITER/mcintos/.config/reckon/crew/runs/"
        "r-20261019T035327482346-ssog-anchor-family-runner",
    )
)


def now() -> str:
    return datetime.now(UTC).isoformat()


def apply_cap(planned_calls: int) -> float:
    """Per-apply cost cap from the family's planned call count."""
    if planned_calls == 3:
        return 1.98
    return round(planned_calls * 0.674, 2)


def free_port_pair() -> int:
    """A bolt port whose http port (bolt + 1000) is also free."""
    for base in range(28000, 40000, 2):
        if _free(base) and _free(base + 1000):
            return base
    raise RuntimeError("No free Neo4j port pair in range")


def _free(port: int) -> bool:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind(("127.0.0.1", port))
        return True
    except OSError:
        return False
    finally:
        sock.close()


def graph_count(gc: GraphClient) -> int:
    return int(gc.query("MATCH (n) RETURN count(n) AS total")[0]["total"])


def code_identity() -> dict[str, str]:
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    listing = subprocess.check_output(
        ["git", "ls-tree", "-r", "HEAD", "imas_codex"], text=True
    )
    digest = hashlib.sha256(listing.encode()).hexdigest()
    from imas_codex.standard_names import harmonize

    return {
        "git_head": head,
        "tree_digest_imas_codex": digest,
        "module_file": str(Path(harmonize.__file__).resolve()),
        "cwd": str(Path.cwd().resolve()),
    }


def prove_isolation(
    gc: GraphClient, live: GraphClient, scratch_uri: str, live_uri: str
) -> dict:
    """Show the scratch instance is a distinct, live-free copy."""
    scratch_nodes = graph_count(gc)
    live_nodes = graph_count(live)
    token = str(uuid.uuid4())
    gc.query("CREATE (:ScratchProbe {id: $token})", token=token)
    marked = graph_count(gc)
    scratch_probe = gc.query(
        "MATCH (p:ScratchProbe {id: $token}) RETURN count(p) AS n", token=token
    )[0]["n"]
    live_probe = live.query(
        "MATCH (p:ScratchProbe {id: $token}) RETURN count(p) AS n", token=token
    )[0]["n"]
    gc.query("MATCH (p:ScratchProbe {id: $token}) DELETE p", token=token)
    restored = graph_count(gc)
    proof = {
        "scratch_uri": scratch_uri,
        "live_uri": live_uri,
        "scratch_nodes": scratch_nodes,
        "live_nodes": live_nodes,
        "marked_scratch_nodes": marked,
        "scratch_probe": int(scratch_probe),
        "live_probe": int(live_probe),
        "restored_scratch_nodes": restored,
        "at": now(),
    }
    if (
        scratch_nodes <= 0
        or marked != scratch_nodes + 1
        or scratch_probe != 1
        or live_probe != 0
        or restored != scratch_nodes
    ):
        raise RuntimeError(f"Scratch-only marker contrast failed: {proof}")
    return proof


def run_apply(
    seed: str, scratch_uri: str, cost_limit: float, log_path: Path
) -> list[dict]:
    command = [
        "uv",
        "run",
        "imas-codex",
        "sn",
        "harmonize",
        "--family",
        seed,
        "--apply",
        "--cost-limit",
        f"{cost_limit:.2f}",
        "--json",
    ]
    env = dict(
        os.environ,
        UV_NO_SYNC="1",
        NEO4J_URI=scratch_uri,
        UV_PROJECT_ENVIRONMENT=PROJECT_VENV,
        PYTHONPATH=str(Path.cwd()),
    )
    from imas_codex.standard_names import harmonize

    with log_path.open("w") as stream:
        stream.write(
            f"revision={code_identity()['git_head']} tree={Path.cwd()} "
            f"command={' '.join(command)}\n"
        )
        stream.write(
            f"module.__file__={Path(harmonize.__file__).resolve()}\n"
            f"resolved_cwd={Path.cwd().resolve()}\n"
        )
        stream.flush()
        completed = subprocess.run(
            command, env=env, check=False, stdout=stream, stderr=subprocess.STDOUT
        )
        stream.write(f"EXIT={completed.returncode}\n")
    print(f"apply log: {log_path} EXIT={completed.returncode}", flush=True)
    if completed.returncode:
        raise RuntimeError(
            f"Harmonize apply failed for {seed}: {completed.returncode}; {log_path}"
        )
    text = log_path.read_text()
    for line in reversed(text.splitlines()):
        stripped = line.strip()
        if not stripped.startswith("["):
            continue
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, list) and parsed and "result" in parsed[0]:
            return parsed
    raise RuntimeError(f"Harmonize returned no JSON receipt for {seed}: {log_path}")


def family_scores(gc: GraphClient, seed: str) -> dict[str, float | None]:
    family = fetch_pattern_families(gc, seeds=[seed])
    if len(family) != 1:
        raise RuntimeError(f"Expected one family for {seed}, got {len(family)}")
    return {m["id"]: m.get("reviewer_score_docs") for m in family[0]["members"]}


def run_arm(seed: str, arm: str) -> dict:
    selection = json.loads(SELECTION.read_text())
    rows = [row for row in selection["selected"] if row["seed"] == seed]
    if len(rows) != 1:
        raise RuntimeError(f"Selection holds {len(rows)} rows for {seed}")
    row = rows[0]
    target_anchor = row["heuristic_anchor"] if arm == "heuristic" else row["jev_anchor"]
    if not target_anchor:
        raise RuntimeError(f"No {arm} anchor recorded for {seed}")

    live_uri = GraphClient().uri
    if live_uri.startswith(("bolt://127.0.0.1", "bolt://localhost")):
        raise RuntimeError(f"Live URI unexpectedly resolves to loopback: {live_uri}")

    data: dict = {
        "family": seed,
        "arm": arm,
        "kind": row.get("kind"),
        "members": row["members"],
        "heuristic_anchor": row["heuristic_anchor"],
        "jev_anchor": row["jev_anchor"],
        "anchor": target_anchor,
        "source_export": selection["source_export"],
        "adoption_rule": selection["adoption_rule"],
        "code": code_identity(),
        "budget_limit_usd": ARM_BUDGET_USD,
        "started_at": now(),
        "outcome_status": "running",
    }
    out_path = OUT_DIR / f"{seed}-{arm}.json"
    out_path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")

    scratch = Path(
        tempfile.mkdtemp(
            prefix=f"{os.environ.get('USER', 'user')}-"
            f"{os.environ.get('SLURM_JOB_ID', os.getpid())}-",
            dir=os.environ["TMPDIR"],
        )
    )
    data["scratch_tree"] = str(scratch)
    out_path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    port = free_port_pair()
    data["port"] = port
    instance = None
    try:
        place = scratch / "instance"
        for sub in ("data", "dumps", "logs", "run", "tmp"):
            (place / sub).mkdir(parents=True, exist_ok=True)
        dump = place / "dumps" / "neo4j.dump"
        with tarfile.open(selection["source_export"], "r:gz") as archive:
            member = next(m for m in archive if m.name.endswith("/graph.dump"))
            with archive.extractfile(member) as source, dump.open("wb") as target:
                shutil.copyfileobj(source, target)
        data["dump_bytes"] = dump.stat().st_size
        out_path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
        print(f"extracted dump: {data['dump_bytes']} bytes", flush=True)

        # Apptainer's default bind of the host temp dir over the container's
        # /tmp makes the image's /tmp read-only here (the host /tmp is mounted
        # read-only in this sandbox), so the runtime cannot unpack its native
        # libraries. Point the container's /tmp at the instance's own writable
        # directory for the duration of the load and start commands.
        previous_bind = os.environ.get("APPTAINER_BIND")
        tmp_bind = f"{place / 'tmp'}:/tmp"
        os.environ["APPTAINER_BIND"] = (
            f"{previous_bind},{tmp_bind}" if previous_bind else tmp_bind
        )
        try:
            proc, neo4j_log = start_temp_neo4j(place, port, port + 1000)
        finally:
            if previous_bind is None:
                os.environ.pop("APPTAINER_BIND", None)
            else:
                os.environ["APPTAINER_BIND"] = previous_bind
        instance = (place, proc)
        scratch_uri = f"bolt://127.0.0.1:{port}"
        data["scratch_uri"] = scratch_uri
        data["neo4j_log"] = str(neo4j_log)

        with GraphClient(uri=scratch_uri) as gc, GraphClient(uri=live_uri) as live:
            data["isolation"] = prove_isolation(gc, live, scratch_uri, live_uri)
            out_path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
            print(f"isolation: {data['isolation']}", flush=True)

            if arm == "jev":
                from imas_codex.standard_names.harmonize import _record_family_anchor

                _record_family_anchor(gc, row["members"], target_anchor)
            plan = plan_family_applies(gc, seeds=[seed])
            if len(plan) != 1 or plan[0]["decision"] != "apply":
                raise RuntimeError(f"Family {seed} does not plan one apply: {plan}")
            if plan[0]["anchor"] != target_anchor:
                raise RuntimeError(
                    f"Anchor mismatch for {seed}: planned {plan[0]['anchor']} "
                    f"!= target {target_anchor}"
                )
            data["planned_calls"] = plan[0]["planned_calls"]
            data["recorded_anchor_before"] = plan[0]["anchor"]

            before = report_family_rows(gc, seeds=[seed])[0]
            before_scores = family_scores(gc, seed)
            data["lint_before"] = before["lint_total"]
            data["lint_before_classes"] = before["lint"]
            data["scores_before"] = before_scores

            ceiling = min(apply_cap(plan[0]["planned_calls"]), ARM_BUDGET_USD)
            data["cost_limit_passed_usd"] = ceiling
            out_path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")

            started_at = now()
            started = time.monotonic()
            outcome = run_apply(
                seed, scratch_uri, ceiling, LOG_DIR / f"apply-{arm}-{seed}.log"
            )
            elapsed = time.monotonic() - started
            ended_at = now()

            after = report_family_rows(gc, seeds=[seed])[0]
            after_scores = family_scores(gc, seed)
            summary = outcome[0]["result"]
            run_ids = list(summary["run_ids"])
            ledger = gc.query(
                "MATCH (c:LLMCost) WHERE c.llm_at >= datetime($started) "
                "AND c.llm_at <= datetime($ended) "
                "RETURN count(c) AS calls, coalesce(sum(c.llm_cost), 0.0) AS spend",
                started=started_at,
                ended=now(),
            )[0]
            per_model = gc.query(
                "MATCH (c:LLMCost) WHERE c.llm_at >= datetime($started) "
                "AND c.llm_at <= datetime($ended) "
                "RETURN c.llm_model AS model, count(c) AS calls, "
                "sum(c.llm_cost) AS spend ORDER BY model",
                started=started_at,
                ended=now(),
            )
            # Containment is proved by this apply's own run ids: if the command
            # had resolved the live graph, its LLMCost rows would be there. The
            # live node count is NOT a containment signal, because the live
            # graph is a shared service that other sessions write to while this
            # arm runs; the delta is recorded as an observation, not a failure.
            live_runs = live.query(
                "UNWIND $ids AS rid MATCH (c:LLMCost {run_id: rid}) "
                "RETURN count(c) AS n",
                ids=run_ids,
            )[0]["n"]
            scratch_runs = gc.query(
                "UNWIND $ids AS rid MATCH (c:LLMCost {run_id: rid}) "
                "RETURN count(c) AS n",
                ids=run_ids,
            )[0]["n"]
            live_nodes_after = graph_count(live)
            data["isolation"]["live_nodes_after"] = live_nodes_after
            data["isolation"]["live_nodes_delta_observed"] = (
                live_nodes_after - data["isolation"]["live_nodes"]
            )
            data["isolation"]["live_scope_run_rows"] = int(live_runs)
            data["isolation"]["scratch_scope_run_rows"] = int(scratch_runs)
            if live_runs != 0 or scratch_runs == 0:
                raise RuntimeError(
                    "Apply resolved the live graph: "
                    f"live_rows={live_runs} scratch_rows={scratch_runs}"
                )

            calls = int(ledger["calls"])
            spend = float(ledger["spend"])
            data.update(
                ended_at=ended_at,
                ledger_window={"started": started_at, "ended": ended_at},
                harmonize_summary=summary,
                run_ids=run_ids,
                calls=calls,
                spend_usd=spend,
                spend_limit_usd=ARM_BUDGET_USD,
                per_model=[
                    {
                        "model": r["model"],
                        "calls": int(r["calls"]),
                        "spend_usd": float(r["spend"]),
                    }
                    for r in per_model
                ],
                ledger_calls=calls,
                ledger_spend_usd=spend,
                wall_seconds=elapsed,
                lint_after=after["lint_total"],
                lint_after_classes=after["lint"],
                scores_after=after_scores,
                stamped=bool(outcome[0].get("stamped")),
                applied=bool(outcome[0].get("applied")),
                decision=outcome[0].get("decision"),
                harmonize_receipts=outcome,
            )
            if spend >= ARM_BUDGET_USD:
                data["outcome_status"] = "failed"
                raise RuntimeError(f"Arm spend ${spend:.4f} reached its ceiling")
            data["outcome_status"] = "complete"
            out_path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
            print(
                f"{arm} {seed}: lint {data['lint_before']}->{data['lint_after']}, "
                f"{calls} calls, ${spend:.4f}, {elapsed:.1f}s, "
                f"stamped={data['stamped']}",
                flush=True,
            )
        return data
    finally:
        if instance is not None:
            place, proc = instance
            stop_temp_neo4j(proc)
            data["stopped_instance"] = {"tree": str(place), "at": now()}
        removed = None
        if scratch.exists():
            removed = sorted(p.name for p in scratch.iterdir())
            shutil.rmtree(scratch)
        data["scratch_removed"] = {
            "tree": str(scratch),
            "removed_entries": removed,
            "at": now(),
        }
        out_path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", required=True)
    parser.add_argument("--arm", required=True, choices=["heuristic", "jev"])
    args = parser.parse_args()
    if not SELECTION.exists():
        raise SystemExit(f"Missing fixed selection: {SELECTION}")
    result = run_arm(args.family, args.arm)
    print(json.dumps({k: result[k] for k in ("family", "arm", "outcome_status")}))


if __name__ == "__main__":
    sys.exit(main())
