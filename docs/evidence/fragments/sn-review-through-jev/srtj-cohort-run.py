"""Run a fixed review cohort through one production pool in an isolated graph."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import socket
import tarfile
import tempfile
import time
import uuid
from collections import defaultdict
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from imas_codex.graph.client import GraphClient
from imas_codex.graph.temp_neo4j import start_temp_neo4j, stop_temp_neo4j
from imas_codex.settings import _load_pyproject_settings
from imas_codex.standard_names.loop import run_sn_pools

HERE = Path(__file__).resolve().parent
COHORT = HERE / "srtj-cohort-estimate.json"
ARCHIVE = (
    Path.home()
    / ".local/share/imas-codex/exports/imas-codex-graph-dev-af50a71-20261018T114748Z.tar.gz"
)
RESULTS = HERE / "srtj-cohort"
LIMIT_USD = 3.0


def stamp() -> str:
    return datetime.now(UTC).isoformat()


def save(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")


def verify_containment(record: dict) -> None:
    evidence = record["containment"]
    if not (
        evidence["scratch_run_rows"] == 1
        and evidence["live_run_rows"] == 0
        and evidence["scratch_cost_rows"] == record["receipt_count"]
        and evidence["live_cost_rows"] == 0
    ):
        raise RuntimeError(f"run receipt escaped scratch graph: {evidence}")


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def make_instance(root: Path, dump: Path) -> tuple[Path, object, Path, str]:
    """Use the same dump placement and private /tmp bind as the anchor runner."""
    place = root / "instance"
    for subdir in ("data", "dumps", "logs", "run", "tmp"):
        (place / subdir).mkdir(parents=True, exist_ok=True)
    os.link(dump, place / "dumps/neo4j.dump")
    previous_bind = os.environ.get("APPTAINER_BIND")
    tmp_bind = f"{place / 'tmp'}:/tmp"
    os.environ["APPTAINER_BIND"] = (
        f"{previous_bind},{tmp_bind}" if previous_bind else tmp_bind
    )
    bolt, http = free_port(), free_port()
    try:
        proc, log = start_temp_neo4j(place, bolt, http)
    finally:
        if previous_bind is None:
            os.environ.pop("APPTAINER_BIND", None)
        else:
            os.environ["APPTAINER_BIND"] = previous_bind
    return place, proc, log, f"bolt://127.0.0.1:{bolt}"


def selected_items(axis: str, half: int | None) -> list[dict]:
    cohort = json.loads(COHORT.read_text())["cohort"]
    if cohort["size"] != 464 or cohort["per_axis"] != 232:
        raise RuntimeError("fixed cohort cardinality changed")
    items = [item for item in cohort["items"] if item["axis"] == axis]
    if len(items) != 232 or len({item["id"] for item in items}) != 232:
        raise RuntimeError("fixed axis is incomplete or contains duplicate identities")
    return items[(half - 1) * 116 : half * 116] if half else items


def prepare_scope(gc: GraphClient, items: list[dict], scope_id: str, axis: str) -> None:
    ids = [item["id"] for item in items]
    rows = gc.query(
        "UNWIND $ids AS id MATCH (sn:StandardName {id:id}) "
        "RETURN sn.id AS id, sn.validation_status AS validation_status, "
        "sn.origin AS origin, sn.description AS description, "
        "sn.documentation AS documentation, sn.name_stage AS name_stage LIMIT 233",
        ids=ids,
    )
    if len(rows) != len(ids):
        missing = sorted(set(ids) - {row["id"] for row in rows})
        raise RuntimeError(
            f"scratch snapshot lacks {len(missing)} cohort names: {missing[:8]}"
        )
    ineligible = [
        row["id"]
        for row in rows
        if row["validation_status"] != "valid"
        or (axis == "name" and (row["origin"] == "derived" or not row["description"]))
        or (
            axis == "docs"
            and (row["name_stage"] != "accepted" or not row["documentation"])
        )
    ]
    if ineligible:
        raise RuntimeError(
            f"scratch review pool cannot claim {len(ineligible)} items: {ineligible[:8]}"
        )
    stage_field = "name_stage" if axis == "name" else "docs_stage"
    changed = gc.query(
        "UNWIND $ids AS id MATCH (sn:StandardName {id:id}) "
        f"SET sn.run_id=$scope_id, sn.{stage_field}='drafted' "
        "REMOVE sn.claimed_at, sn.claim_token "
        "RETURN count(sn) AS changed",
        ids=ids,
        scope_id=scope_id,
    )
    if changed[0]["changed"] != len(ids):
        raise RuntimeError("scratch scope restaging was incomplete")


def receipts(gc: GraphClient, run_id: str) -> list[dict]:
    rows = gc.query(
        "MATCH (c:LLMCost {run_id:$run_id}) "
        "RETURN c.id AS id, c.phase AS phase, c.llm_model AS model, "
        "c.llm_cost AS cost, c.standard_name_ids AS names LIMIT 5001",
        run_id=run_id,
    )
    if len(rows) == 5001:
        raise RuntimeError("receipt read reached its bound")
    return rows


def review_rows(gc: GraphClient, ids: list[str], started: str, axis: str) -> list[dict]:
    rows = gc.query(
        "UNWIND $ids AS id MATCH (r:StandardNameReview {standard_name_id:id}) "
        "WHERE r.review_axis=$axis AND r.reviewed_at >= datetime($started) "
        "RETURN r.standard_name_id AS id, r.judge_kind AS kind, "
        "r.comments AS comments, r.resolution_method AS method, "
        "r.model AS model, r.reviewed_at AS at LIMIT 3001",
        ids=ids,
        axis=axis,
        started=started,
    )
    if len(rows) == 3001:
        raise RuntimeError("review row read reached its bound")
    return rows


def count_run(gc: GraphClient, run_id: str) -> dict:
    run = gc.query("MATCH (r:SNRun {id:$run_id}) RETURN count(r) AS n", run_id=run_id)[
        0
    ]["n"]
    cost = gc.query(
        "MATCH (c:LLMCost {run_id:$run_id}) RETURN count(c) AS n", run_id=run_id
    )[0]["n"]
    return {"run": int(run), "cost": int(cost)}


async def measure(args: argparse.Namespace, record: dict, path: Path) -> None:
    items = selected_items(args.axis, args.half)
    ids = [item["id"] for item in items]
    record["items"] = [
        {
            **item,
            "route": "unjudged",
            "failure_reason": "review did not run",
            "chat_reviewer_called": False,
            "decision": None,
            "score": None,
            "calls": 0,
            "spend_usd": 0.0,
            "wall_time_s": None,
        }
        for item in items
    ]
    save(path, record)
    root = Path(tempfile.mkdtemp(prefix="review-cohort-", dir=os.environ["TMPDIR"]))
    record["scratch_tree"] = str(root)
    save(path, record)
    proc = None
    prior_uri = os.environ.get("NEO4J_URI")
    try:
        live_uri = GraphClient().uri
        if live_uri.startswith(("bolt://127.0.0.1", "bolt://localhost")):
            raise RuntimeError("live graph unexpectedly resolves to loopback")
        with tarfile.open(ARCHIVE, "r:gz") as archive:
            member = next(
                member for member in archive if member.name.endswith("/graph.dump")
            )
            dump = root / "neo4j.dump"
            with archive.extractfile(member) as source, dump.open("wb") as target:
                shutil.copyfileobj(source, target)
        place, proc, log, uri = make_instance(root, dump)
        record["scratch_instance"] = {"path": str(place), "uri": uri, "log": str(log)}
        os.environ["NEO4J_URI"] = uri
        scope_id = str(uuid.uuid4())
        with GraphClient(uri=uri) as gc:
            prepare_scope(gc, items, scope_id, args.axis)
        config = _load_pyproject_settings().setdefault("sn-review", {})
        config["cascade-deciding"] = args.arm == "cascade"
        record["switch"] = "in-process cached settings[sn-review].cascade-deciding"
        record["switch_value"] = config["cascade-deciding"]
        record["scope_id"] = scope_id
        save(path, record)
        events = []
        started = stamp()
        start_time = time.monotonic()
        summary = await run_sn_pools(
            cost_limit=(LIMIT_USD if args.axis == "name" else 5.0)
            if args.arm == "cascade"
            else 55.0,
            time_limit_s=2400,
            only_pool="review_name" if args.axis == "name" else "review_docs",
            names_only=args.axis == "name",
            docs_only=args.axis == "docs",
            scope_run_id=scope_id,
            scope_size_hint=len(ids),
            skip_global_maintenance=True,
            on_event=lambda event: events.append(
                {**event, "at": stamp(), "elapsed_s": time.monotonic() - start_time}
            ),
        )
        record["run_id"] = summary.run_id
        record["run_summary"] = {
            **asdict(summary),
            "started_at": str(summary.started_at),
            "stopped_at": str(summary.stopped_at),
            "domains_touched": sorted(summary.domains_touched),
        }
        record["finished_at"] = stamp()
        record["wall_time_s"] = time.monotonic() - start_time
        completed = {
            event.get("name"): event for event in events if event.get("name") in ids
        }
        with GraphClient(uri=uri) as gc:
            cost_rows = receipts(gc, summary.run_id)
            rows = review_rows(gc, ids, started, args.axis)
            stage_field = "name_stage" if args.axis == "name" else "docs_stage"
            score_field = (
                "reviewer_score_name" if args.axis == "name" else "reviewer_score_docs"
            )
            state = gc.query(
                "UNWIND $ids AS id MATCH (sn:StandardName {id:id}) "
                f"RETURN sn.id AS id, sn.{stage_field} AS stage, sn.{score_field} AS score LIMIT 233",
                ids=ids,
            )
            by_name = defaultdict(list)
            by_model = defaultdict(float)
            for cost in cost_rows:
                by_model[cost["model"]] += float(cost["cost"] or 0)
                for name in cost["names"] or []:
                    by_name[name].append(cost)
            routes = defaultdict(list)
            for row in rows:
                if row["kind"] == "jev":
                    routes[row["id"]].append(row)
            states = {row["id"]: row for row in state}
            for item in record["items"]:
                name = item["id"]
                route_rows = routes[name]
                route_data = (
                    json.loads(route_rows[-1]["comments"]) if route_rows else {}
                )
                route = route_data.get("route", "unjudged")
                if route == "chat_review":
                    route = "band"
                item.update(
                    route=route,
                    failure_reason=(route_data.get("failure_reason") or "no route row")
                    if route == "unjudged"
                    else None,
                    chat_reviewer_called=(
                        args.arm == "incumbent"
                        or route in {"band", "chat_below_reject", "unjudged"}
                    )
                    and bool(by_name[name]),
                    decision=states[name]["stage"],
                    score=states[name]["score"],
                    calls=len(by_name[name]),
                    spend_usd=round(
                        sum(
                            float(c["cost"] or 0) / max(1, len(c["names"] or []))
                            for c in by_name[name]
                        ),
                        6,
                    ),
                    wall_time_s=round(
                        completed[name]["elapsed_s"]
                        if name in completed
                        else record["wall_time_s"],
                        3,
                    ),
                )
            record["receipt_count"] = len(cost_rows)
            if not cost_rows:
                raise RuntimeError("review produced no cost receipts")
            record["per_model_spend_usd"] = {
                model: round(value, 6) for model, value in sorted(by_model.items())
            }
            record["spend_usd"] = round(sum(by_model.values()), 6)
            record["route_counts"] = {
                route: sum(item["route"] == route for item in record["items"])
                for route in (
                    "accept",
                    "refine",
                    "band",
                    "chat_below_reject",
                    "unjudged",
                )
            }
            record["scratch_evidence"] = count_run(gc, summary.run_id)
        # Preserve every paid item and receipt total before checking containment.
        save(path, record)
        with GraphClient(uri=uri) as scratch, GraphClient(uri=live_uri) as live:
            local = count_run(scratch, summary.run_id)
            remote = count_run(live, summary.run_id)
        record["containment"] = {
            "scratch_uri": uri,
            "live_uri": live_uri,
            "scratch_run_rows": local["run"],
            "scratch_cost_rows": local["cost"],
            "live_run_rows": remote["run"],
            "live_cost_rows": remote["cost"],
        }
        try:
            verify_containment(record)
            record["containment_verdict"] = "pass"
        except RuntimeError as exc:
            record["containment_verdict"] = f"fail: {exc}"
            raise
        finally:
            save(path, record)
        if len(record["items"]) != len(ids) or (
            record["spend_usd"] >= LIMIT_USD
            and args.arm == "cascade"
            and args.axis == "name"
        ):
            raise RuntimeError("piece cardinality or spend ceiling failed")
    finally:
        if prior_uri is None:
            os.environ.pop("NEO4J_URI", None)
        else:
            os.environ["NEO4J_URI"] = prior_uri
        if proc is not None:
            stop_temp_neo4j(proc)
            record["scratch_stopped"] = True
        shutil.rmtree(root)
        record["scratch_removed"] = {"path": str(root), "exists_after": root.exists()}
        save(path, record)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("incumbent", "cascade"), required=True)
    parser.add_argument("--axis", choices=("name", "docs"), required=True)
    parser.add_argument("--half", type=int, choices=(1, 2))
    parser.add_argument("--check-containment", type=Path)
    args = parser.parse_args()
    if args.check_containment:
        verify_containment(json.loads(args.check_containment.read_text()))
        return
    import subprocess

    record = {
        "arm": args.arm,
        "axis": args.axis,
        "half": args.half,
        "code_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "module_path": __file__,
        "working_directory": str(Path.cwd().resolve()),
        "created_at": stamp(),
        "status": "running",
    }
    suffix = f"-{args.half}" if args.half else ""
    path = RESULTS / f"{args.arm}-{args.axis}{suffix}.json"
    save(path, record)
    try:
        asyncio.run(measure(args, record, path))
        record["status"] = "complete"
    except Exception as exc:
        record["status"] = "failed"
        record["failure"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        save(path, record)


if __name__ == "__main__":
    main()
