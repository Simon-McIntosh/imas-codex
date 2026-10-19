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
import fcntl
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
from dataclasses import dataclass
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


# The ports are node-local, and each arm runs in its own run directory with its
# own TMPDIR, so the lock must live in a directory that every concurrent arm on
# the node shares and can write. /tmp is that directory on an ordinary host, but
# a worker sandbox mounts it read-only, so the default is the node-local shared
# memory filesystem; SSOG_PORT_LOCK_DIR overrides it for any other shared path.
PORT_LOCK_DIR = Path(os.environ.get("SSOG_PORT_LOCK_DIR", "/dev/shm"))
PORT_LOCK_PREFIX = "ssog-scratch-port-"


def _lock_path(base: int) -> Path:
    return PORT_LOCK_DIR / f"{PORT_LOCK_PREFIX}{base}.lock"


@dataclass
class PortReservation:
    """An exclusive claim on a scratch port pair, held for the arm's life.

    ``release`` is idempotent: the arm's cleanup runs on every exit path, and
    a path that released twice must not close the descriptor twice.
    """

    base: int
    lock_path: Path
    _fd: int | None

    def release(self) -> None:
        if self._fd is None:
            return
        try:
            fcntl.flock(self._fd, fcntl.LOCK_UN)
        finally:
            os.close(self._fd)
            self._fd = None


def reserve_port_pair(arm: str) -> PortReservation:
    """Reserve an exclusive bolt/http pair for this arm's whole lifetime.

    A scratch instance binds its bolt port only after its multi-gigabyte dump
    has loaded, minutes after the port is chosen, so probing for a port that is
    free right now cannot keep two concurrent arms apart: both see the same
    ports free and later race for them. Each candidate pair carries a lock file
    taken with an exclusive non-blocking ``flock``, held from reservation until
    the instance is stopped and its tree removed. A pair whose lock another arm
    holds, or whose ports are bound, is skipped. The lock file records the
    holder's pid and arm so a reader can see who has it.
    """
    for base in range(28000, 40000, 2):
        if not (_free(base) and _free(base + 1000)):
            continue
        lock_path = _lock_path(base)
        fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            os.close(fd)
            continue
        # Another arm may have bound the ports between the probe above and the
        # lock; re-probe under the lock before committing to the pair.
        if not (_free(base) and _free(base + 1000)):
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)
            continue
        os.ftruncate(fd, 0)
        os.write(fd, f"pid={os.getpid()} arm={arm}\n".encode())
        os.fsync(fd)
        return PortReservation(base=base, lock_path=lock_path, _fd=fd)
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


def family_window_rows(
    gc: GraphClient, member_ids: list[str], started: str, ended: str
) -> int:
    """Cost rows in one window that name any of this family's members.

    ``standard_name_ids`` make a row attributable to this apply rather than to
    whatever else shares the graph, which is what a leak test needs. On the
    scratch copy every such row is ours; on the live graph a row naming our
    members could only have come from this apply writing there.
    """
    return int(
        gc.query(
            "MATCH (c:LLMCost) WHERE c.llm_at >= datetime($started) "
            "AND c.llm_at <= datetime($ended) "
            "AND any(sid IN coalesce(c.standard_name_ids, []) "
            "        WHERE sid IN $members) "
            "RETURN count(c) AS n",
            members=list(member_ids),
            started=started,
            ended=ended,
        )[0]["n"]
    )


def evaluate_containment(measured: dict) -> dict:
    """Decide whether this apply's writes stayed inside the scratch copy.

    Two controls, and a check that measures neither is not a check:

    * positive -- the apply's own cost rows are present in the scratch
      instance, either as rows naming this family's members or as rows in the
      apply window on a graph only this run writes to. Without it an apply
      that wrote nowhere would pass a leak test by writing nothing.
    * negative -- none of those rows are present in the live graph, so the
      command did not write there.

    The live graph is shared and other sessions write to it while this arm
    runs, so its total node count and its bare window row count are recorded
    as observations and never judged. Only rows attributable to this apply
    can indict it.
    """
    scratch_signal = (
        int(measured.get("scratch_family_rows") or 0)
        + int(measured.get("scratch_ledger_rows") or 0)
        + int(measured.get("scratch_scope_run_rows") or 0)
    )
    live_signal = int(measured.get("live_family_rows") or 0) + int(
        int(measured.get("live_scope_run_rows") or 0)
    )
    reasons: list[str] = []
    if scratch_signal <= 0:
        reasons.append(
            "no scratch cost row names this family or falls in the apply "
            "window: the apply wrote nothing to the copy being measured"
        )
    if live_signal != 0:
        reasons.append(
            f"{live_signal} cost rows attributable to this apply are present "
            "in the live graph"
        )
    return {
        "holds": not reasons,
        "reasons": reasons,
        "scratch_signal": scratch_signal,
        "live_signal": live_signal,
        "evidence": dict(measured),
    }


def self_check_containment(receipt_path: Path) -> dict:
    """Exercise the containment check on planted inputs from a saved record.

    No apply is run and no graph is touched: the measurement numbers are
    planted, seeded from an earlier invocation's saved record. The point is to
    see the negative control fire on a fabricated live row rather than to
    assert that a passing means anything on its own.
    """
    receipt = json.loads(receipt_path.read_text())
    isolation = receipt.get("isolation", {})
    base = {
        "scratch_family_rows": int(receipt.get("calls") or 3),
        "scratch_ledger_rows": int(receipt.get("calls") or 3),
        "scratch_scope_run_rows": int(isolation.get("scratch_scope_run_rows") or 0),
        "live_family_rows": 0,
        "live_scope_run_rows": 0,
    }
    cases = {
        "scratch-only": ({}, True),
        "planted-live-family-row": ({"live_family_rows": 1}, False),
        "planted-live-run-id-row": ({"live_scope_run_rows": 1}, False),
        "wrote-nothing": (
            {
                "scratch": None,
                "scratch_family_rows": 0,
                "scratch_ledger_rows": 0,
                "scratch_scope_run_rows": 0,
            },
            False,
        ),
    }
    results = {}
    for name, (overrides, expected) in cases.items():
        measured = {**base, **{k: v for k, v in overrides.items() if k != "scratch"}}
        verdict = evaluate_containment(measured)
        results[name] = {
            "expected_holds": expected,
            "holds": verdict["holds"],
            "reasons": verdict["reasons"],
            "scratch_signal": verdict["scratch_signal"],
            "live_signal": verdict["live_signal"],
        }
        if verdict["holds"] is not expected:
            raise RuntimeError(
                f"Containment self-check disagreed with expectation for "
                f"{name}: {results[name]}"
            )
    report = {
        "receipt": str(receipt_path),
        "at": now(),
        "cases": results,
        "summary": {
            "cases": len(results),
            "failed_as_planted": sum(1 for r in results.values() if not r["holds"]),
            "passed_as_expected": sum(1 for r in results.values() if r["holds"]),
            "all_match": True,
        },
    }
    return report


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
    data["scratch_removed"] = None
    instance = None
    reservation = None
    try:
        reservation = reserve_port_pair(arm)
        port = reservation.base
        data["port"] = port
        data["port_lock"] = str(reservation.lock_path)
        out_path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
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
            member_ids = list(row["members"])
            window_end = now()
            ledger = gc.query(
                "MATCH (c:LLMCost) WHERE c.llm_at >= datetime($started) "
                "AND c.llm_at <= datetime($ended) "
                "RETURN count(c) AS calls, coalesce(sum(c.llm_cost), 0.0) AS spend",
                started=started_at,
                ended=window_end,
            )[0]
            per_model = gc.query(
                "MATCH (c:LLMCost) WHERE c.llm_at >= datetime($started) "
                "AND c.llm_at <= datetime($ended) "
                "RETURN c.llm_model AS model, count(c) AS calls, "
                "sum(c.llm_cost) AS spend ORDER BY model",
                started=started_at,
                ended=window_end,
            )
            scratch_family_rows = family_window_rows(
                gc, member_ids, started_at, window_end
            )
            scratch_scope_run_rows = gc.query(
                "UNWIND $ids AS rid MATCH (c:LLMCost {run_id: rid}) "
                "RETURN count(c) AS n",
                ids=run_ids,
            )[0]["n"]
            live_family_rows = family_window_rows(
                live, member_ids, started_at, window_end
            )
            live_scope_run_rows = live.query(
                "UNWIND $ids AS rid MATCH (c:LLMCost {run_id: rid}) "
                "RETURN count(c) AS n",
                ids=run_ids,
            )[0]["n"]
            live_window_rows = live.query(
                "MATCH (c:LLMCost) WHERE c.llm_at >= datetime($started) "
                "AND c.llm_at <= datetime($ended) RETURN count(c) AS n",
                started=started_at,
                ended=window_end,
            )[0]["n"]
            live_nodes_after = graph_count(live)

            calls = int(ledger["calls"])
            spend = float(ledger["spend"])

            # The complete arm record is written BEFORE any check that could
            # reject it. A check is a statement about a measurement, and a
            # defective one must not be able to destroy a paid measurement:
            # every field the arm exists to produce lands on disk first, and
            # only then is containment judged and its verdict recorded.
            data.update(
                ended_at=ended_at,
                ledger_window={"started": started_at, "ended": window_end},
                harmonize_summary=summary,
                run_ids=run_ids,
                calls=calls,
                spend_usd=spend,
                spend_limit_usd=ARM_BUDGET_USD,
                per_model=[
                    {
                        "model": r["model"],
                        "calls": int(r["calls"]),
                        "spend_usd": None if r["spend"] is None else float(r["spend"]),
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
            # The live node COUNT is not a containment signal: the live graph
            # is a shared service other sessions write to while this arm runs.
            # The delta and the bare window row count are recorded as
            # observations, never judged; only rows attributable to this apply
            # can indict it.
            data["isolation"].update(
                live_nodes_after=live_nodes_after,
                live_nodes_delta_observed=(
                    live_nodes_after - data["isolation"]["live_nodes"]
                ),
                scratch_family_rows=int(scratch_family_rows),
                scratch_scope_run_rows=int(scratch_scope_run_rows),
                scratch_ledger_rows=calls,
                live_family_rows=int(live_family_rows),
                live_scope_run_rows=int(live_scope_run_rows),
                live_window_rows_observed=int(live_window_rows),
            )
            data["containment"] = evaluate_containment(
                {
                    "scratch_family_rows": int(scratch_family_rows),
                    "scratch_scope_run_rows": int(scratch_scope_run_rows),
                    "scratch_ledger_rows": calls,
                    "live_family_rows": int(live_family_rows),
                    "live_scope_run_rows": int(live_scope_run_rows),
                }
            )
            data["containment"]["at"] = now()
            data["outcome_status"] = (
                "complete" if data["containment"]["holds"] else "containment"
            )
            out_path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")

            print(
                f"{arm} {seed}: lint {data['lint_before']}->{data['lint_after']}, "
                f"{calls} calls, ${spend:.4f}, {elapsed:.1f}s, "
                f"stamped={data['stamped']}, "
                f"containment={data['containment']['holds']}",
                flush=True,
            )
            if spend >= ARM_BUDGET_USD:
                data["outcome_status"] = "over-budget"
                out_path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
                raise RuntimeError(
                    f"Arm spend ${spend:.4f} reached its ceiling; the arm record "
                    f"is on disk at {out_path}"
                )
            if not data["containment"]["holds"]:
                raise RuntimeError(
                    f"Containment failed for {arm} {seed}: "
                    f"{'; '.join(data['containment']['reasons'])}. The arm "
                    f"record is on disk at {out_path}"
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
        # The pair stays reserved until the instance is stopped and its tree
        # removed, so no later arm can claim a port this arm is still using.
        if reservation is not None:
            reservation.release()
            data["port_lock_released"] = {
                "lock": str(reservation.lock_path),
                "at": now(),
            }
        out_path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family")
    parser.add_argument("--arm", choices=["heuristic", "jev"])
    parser.add_argument(
        "--self-check-containment",
        metavar="SAVED_RECORD",
        help="Exercise the containment check on planted inputs seeded from a "
        "saved arm record. Runs no apply and touches no graph.",
    )
    args = parser.parse_args()
    if args.self_check_containment:
        report = self_check_containment(Path(args.self_check_containment))
        log_path = LOG_DIR / "containment-self-check.log"
        log_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(json.dumps(report["summary"], sort_keys=True))
        print(f"self-check log: {log_path}")
        return
    if not args.family or not args.arm:
        raise SystemExit("--family and --arm are required unless self-checking")
    if not SELECTION.exists():
        raise SystemExit(f"Missing fixed selection: {SELECTION}")
    result = run_arm(args.family, args.arm)
    print(json.dumps({k: result[k] for k in ("family", "arm", "outcome_status")}))


if __name__ == "__main__":
    sys.exit(main())
