"""Run a fixed review cohort through one production pool in an isolated graph."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
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
from decimal import Decimal
from pathlib import Path

from imas_codex.graph.client import GraphClient
from imas_codex.graph.models import DDResolutionField, DDResolutionValueKind
from imas_codex.graph.temp_neo4j import start_temp_neo4j, stop_temp_neo4j
from imas_codex.settings import _load_pyproject_settings
from imas_codex.standard_names.dd_resolutions import (
    DDResolutionError,
    DDResolutionValue,
    load_dd_resolution_manifest,
    read_graph_dd_field,
)
from imas_codex.standard_names.loop import run_sn_pools

HERE = Path(__file__).resolve().parent
COHORT = HERE / "srtj-cohort-estimate.json"
ARCHIVE = (
    Path.home()
    / ".local/share/imas-codex/exports/imas-codex-graph-dev-af50a71-20261018T114748Z.tar.gz"
)
RESULTS = HERE / "srtj-cohort"
EXCLUSIONS = RESULTS / "excluded.json"
PIECE_CEILING_USD = 3.5
COMPARISON_CEILING_USD = Decimal("60.00")
PRIOR_SPEND_USD = 0.000288
RUN_LIMIT_USD = 2.99
RECOVERY_LIMIT_USD = 0.53


def stamp() -> str:
    return datetime.now(UTC).isoformat()


def save(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")


def verify_containment(record: dict) -> None:
    checks = [(record["containment"], record["receipt_count"])]
    if recovery := record.get("recovery"):
        checks = [
            (
                record["containment"],
                record["receipt_count"] - recovery["receipt_count"],
            ),
            (recovery["containment"], recovery["receipt_count"]),
        ]
    for evidence, count in checks:
        if not (
            evidence["scratch_run_rows"] == 1
            and evidence["live_run_rows"] == 0
            and evidence["scratch_cost_rows"] == count
            and evidence["live_cost_rows"] == 0
        ):
            raise RuntimeError(f"run receipt escaped scratch graph: {evidence}")


def verify_completion(record: dict) -> None:
    """Refuse a paid piece that stopped with selected identities still pending."""
    if record.get("recovery") and (
        len(record["items"]) != 231
        or {item["id"] for item in record["items"]}
        != {item["id"] for item in selected_items("name", 1, 1)}
    ):
        raise RuntimeError("combined review differs from the selected cohort")
    pending = [
        item["id"]
        for item in record["items"]
        if item.get("decision") in {None, "drafted"} or item.get("wall_time_s") is None
    ]
    if pending:
        raise RuntimeError(
            f"review piece left {len(pending)} selected items pending: {pending[:8]}"
        )
    reason = record.get("run_summary", {}).get("stop_reason")
    if reason not in {"completed", "no_eligible_work"}:
        raise RuntimeError(f"review piece stopped without completion: {reason}")
    if record.get("receipt_count", 0) <= 0:
        raise RuntimeError("review piece has no cost receipts")


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


def fixed_items() -> list[dict]:
    cohort = json.loads(COHORT.read_text())["cohort"]
    if cohort["size"] != 464 or cohort["per_axis"] != 232:
        raise RuntimeError("fixed cohort cardinality changed")
    items = cohort["items"]
    if len(items) != 464 or len({item["id"] for item in items}) != 464:
        raise RuntimeError(
            "fixed cohort is incomplete or contains duplicate identities"
        )
    if any(
        sum(item["axis"] == axis for item in items) != 232 for axis in ("name", "docs")
    ):
        raise RuntimeError("fixed cohort axis cardinality changed")
    return items


def selected_items(axis: str, part: int, parts: int) -> list[dict]:
    items = [item for item in fixed_items() if item["axis"] == axis]
    excluded = json.loads(EXCLUSIONS.read_text())
    if (
        excluded["status"] != "accepted"
        or excluded["cohort_digest"] != hashlib.sha256(COHORT.read_bytes()).hexdigest()
        or excluded["excluded_count"] > 23
    ):
        raise RuntimeError("DD exclusion record is not valid for the fixed cohort")
    omitted = {row["id"] for row in excluded["excluded_items"]}
    if len(omitted) != excluded["excluded_count"]:
        raise RuntimeError("DD exclusion record contains duplicate identities")
    items = [item for item in items if item["id"] not in omitted]
    return items[(part - 1) * len(items) // parts : part * len(items) // parts]


def check_comparison_budget(ceiling_usd: float) -> Decimal:
    """Reserve this piece against completed and incomplete canonical pieces."""
    if not math.isfinite(ceiling_usd) or ceiling_usd <= 0:
        raise RuntimeError("piece spend ceiling must be positive and finite")
    spent = Decimal("0")
    for path in sorted(RESULTS.glob("*.json")):
        if path.name == "excluded.json" or path.name.endswith("-recovery.json"):
            continue
        piece = json.loads(path.read_text())
        if "piece_spend_total_usd" not in piece:
            raise RuntimeError(f"piece has no spend total: {path}")
        spent += Decimal(str(piece["piece_spend_total_usd"]))
    if spent + Decimal(str(ceiling_usd)) > COMPARISON_CEILING_USD:
        raise RuntimeError(
            f"comparison ceiling USD {COMPARISON_CEILING_USD} exceeded: "
            f"spent USD {spent} plus piece ceiling USD {ceiling_usd}"
        )
    return spent


def _resolution_value(field: DDResolutionField, raw: object) -> DDResolutionValue:
    if field == DDResolutionField.coordinates:
        return DDResolutionValue(
            kind=DDResolutionValueKind.string_list,
            value=tuple(raw or ()),
        )
    return DDResolutionValue(
        kind=DDResolutionValueKind.null
        if raw is None
        else DDResolutionValueKind.string,
        value=raw,
    )


def scan_dd_sources(gc: GraphClient) -> dict:
    """Check each exact source projection that the review worker can resolve."""
    items = fixed_items()
    manifest = load_dd_resolution_manifest()
    by_item = {item["id"]: item for item in items}
    errors: dict[str, dict[tuple[str, str, str], dict]] = defaultdict(dict)
    source_counts: dict[str, int] = defaultdict(int)
    source_rows = 0
    field_checks = 0
    for start in range(0, len(items), 25):
        ids = [item["id"] for item in items[start : start + 25]]
        rows = gc.query(
            "UNWIND $ids AS id MATCH (sn:StandardName {id:id}) "
            "OPTIONAL MATCH (source:StandardNameSource)-[:PRODUCED_NAME]->(sn) "
            "WHERE source.source_type='dd' "
            "RETURN sn.id AS item_id, sn.unit AS name_unit, "
            "properties(source) AS source LIMIT 5001",
            ids=ids,
        )
        if len(rows) == 5001:
            raise RuntimeError("DD source read reached its row bound")
        for row in rows:
            item_id = row["item_id"]
            source = row["source"]
            if source is None:
                continue
            source_counts[item_id] += 1
            source_rows += 1
            path = str(
                source.get("dd_path") or source.get("source_id") or ""
            ).removeprefix("dd:")
            version = source.get("dd_version") or ""
            if not path or not version:
                raise RuntimeError(
                    f"DD source lacks exact path or version: {source.get('id')}"
                )
            values = {
                "unit": source.get("dd_unit") or row["name_unit"] or None,
                "documentation": source.get("dd_documentation"),
                "data_type": source.get("dd_data_type"),
                "coordinates": source.get("dd_coordinates") or (),
                "lifecycle_status": source.get("dd_lifecycle_status"),
                "lifecycle_version": None,
            }
            contexts = [(path, values)]
            parent_path = source.get("dd_parent_path")
            if parent_path:
                contexts.append(
                    (
                        parent_path,
                        {"documentation": source.get("dd_parent_documentation") or ""},
                    )
                )
            for source_path, context in contexts:
                for field in DDResolutionField:
                    field_checks += 1
                    try:
                        read_graph_dd_field(
                            path=source_path,
                            dd_version=version,
                            field=field,
                            graph_value=_resolution_value(
                                field, context.get(field.value)
                            ),
                            manifest=manifest,
                        )
                    except DDResolutionError as exc:
                        key = (source_path, field.value, type(exc).__name__)
                        errors[item_id][key] = {
                            "path": source_path,
                            "field": field.value,
                            "error_type": type(exc).__name__,
                            "reason": str(exc),
                            "source_id": source.get("id"),
                        }
    if not source_rows or not any(source_counts.values()):
        raise RuntimeError("DD source scan found no known-present source")
    without_sources = sorted(set(by_item) - set(source_counts))
    excluded_items = [
        {
            "id": item["id"],
            "axis": item["axis"],
            "errors": list(errors[item["id"]].values()),
        }
        for item in items
        if errors[item["id"]]
    ]
    if not any(
        error["path"] == "camera_ir/channel/camera/direction/x"
        and error["field"] == "unit"
        for item in excluded_items
        for error in item["errors"]
    ):
        raise RuntimeError("DD source scan missed the known stale unit")
    return {
        "status": "accepted" if len(excluded_items) <= 23 else "over_limit",
        "cohort_digest": hashlib.sha256(COHORT.read_bytes()).hexdigest(),
        "cohort_count": 464,
        "max_exclusions": 23,
        "archive": str(ARCHIVE),
        "manifest_digest": manifest.digest,
        "resolution_count": len(manifest.resolutions),
        "source_rows": source_rows,
        "field_checks": field_checks,
        "items_without_dd_sources": without_sources,
        "excluded_count": len(excluded_items),
        "excluded_by_axis": {
            axis: sum(item["axis"] == axis for item in excluded_items)
            for axis in ("name", "docs")
        },
        "excluded_items": excluded_items,
        "scanned_at": stamp(),
    }


def preflight() -> None:
    """Scan all fixed identities in a fresh export copy without a review call."""
    import subprocess

    record = {
        "status": "scanning",
        "created_at": stamp(),
        "code_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "module_path": __file__,
        "working_directory": str(Path.cwd().resolve()),
    }
    save(EXCLUSIONS, record)
    root = Path(tempfile.mkdtemp(prefix="review-preflight-", dir=os.environ["TMPDIR"]))
    record["scratch_tree"] = str(root)
    save(EXCLUSIONS, record)
    proc = None
    prior_uri = os.environ.get("NEO4J_URI")
    try:
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
        with GraphClient(uri=uri) as gc:
            result = scan_dd_sources(gc)
        record.update(result)
        save(EXCLUSIONS, record)
        print(
            f"DD preflight: {result['excluded_count']} of 464 excluded; "
            f"{result['source_rows']} source rows and {result['field_checks']} field checks",
            flush=True,
        )
        if result["status"] != "accepted":
            raise RuntimeError(
                f"DD preflight exceeds 23-item exclusion limit: {result['excluded_count']}"
            )
    except Exception as exc:
        if record["status"] == "scanning":
            record["status"] = "failed"
        record["failure"] = f"{type(exc).__name__}: {exc}"
        raise
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
        save(EXCLUSIONS, record)


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


def combine_recovery(base: dict, recovery: dict) -> dict:
    """Preserve the paid first run and add only the completed retry identities."""
    pending = set(base["incomplete_ids"])
    recovered = {item["id"]: item for item in recovery["items"]}
    if not pending or pending != set(recovered):
        raise RuntimeError("recovery identities differ from the saved pending set")
    if len(base["items"]) != len(selected_items("name", 1, 1)):
        raise RuntimeError("saved piece differs from the selected cohort")
    combined = dict(base)
    combined["initial_attempt"] = {
        key: combined.pop(key)
        for key in (
            "attempt_log",
            "failure",
            "receipt_accounting_note",
            "run_status_at_stop",
            "unfunded_items",
        )
        if key in combined
    }
    combined["items"] = [recovered.get(item["id"], item) for item in base["items"]]
    combined["recovery"] = {
        "run_id": recovery["run_id"],
        "code_revision": recovery["code_revision"],
        "receipt_count": recovery["receipt_count"],
        "receipt_rows": recovery["receipt_rows"],
        "spend_usd": recovery["spend_usd"],
        "scratch_evidence": recovery["scratch_evidence"],
        "switch": recovery["switch"],
        "switch_value": recovery["switch_value"],
    }
    combined["receipt_count"] = base["receipt_count"] + recovery["receipt_count"]
    combined["spend_usd"] = round(base["spend_usd"] + recovery["spend_usd"], 6)
    combined["piece_spend_total_usd"] = round(
        base["piece_spend_total_usd"] + recovery["spend_usd"], 6
    )
    models = defaultdict(float, base["per_model_spend_usd"])
    for model, spend in recovery["per_model_spend_usd"].items():
        models[model] += spend
    combined["per_model_spend_usd"] = {
        model: round(spend, 6) for model, spend in sorted(models.items())
    }
    combined["run_summary"] = recovery["run_summary"]
    combined["route_counts"] = {
        route: sum(item["route"] == route for item in combined["items"])
        for route in ("accept", "refine", "band", "chat_below_reject", "unjudged")
    }
    combined["band_share"] = round(
        combined["route_counts"]["band"] / len(combined["items"]), 6
    )
    combined["stage_counts"] = {
        str(stage): sum(item["decision"] == stage for item in combined["items"])
        for stage in {item["decision"] for item in combined["items"]}
    }
    combined["incomplete_ids"] = [
        item["id"]
        for item in combined["items"]
        if item["decision"] in {None, "drafted"}
    ]
    combined["measurement_complete"] = False
    combined["comparison_eligible"] = False
    combined["status"] = "containment_pending"
    return combined


async def measure(args: argparse.Namespace, record: dict, path: Path) -> None:
    items = selected_items(args.axis, args.part, args.parts)
    base = None
    combined_path = RESULTS / "cascade-name.json" if args.retry_pending else None
    combined_ready = False
    if args.retry_pending:
        base = json.loads((RESULTS / "cascade-name.json").read_text())
        pending = set(base["incomplete_ids"])
        if (
            args.arm != "cascade"
            or args.axis != "name"
            or (args.part, args.parts) != (1, 1)
            or len(pending) != 3
            or pending
            != {item["id"] for item in base["items"] if item["decision"] == "drafted"}
        ):
            raise RuntimeError("saved pending set is invalid for name recovery")
        items = [item for item in items if item["id"] in pending]
        if len(items) != len(pending):
            raise RuntimeError("pending identities are absent from the selected cohort")
    ids = [item["id"] for item in items]
    record["exclusions"] = str(EXCLUSIONS)
    record["excluded_count"] = json.loads(EXCLUSIONS.read_text())["excluded_count"]
    record["items"] = [
        {
            **item,
            "route": "unjudged",
            "failure_reason": "review did not run",
            "chat_reviewer_called": False,
            "decision": None,
            "score": None,
            "rubric_total": None,
            "dimension_scores": {},
            "failed_checks": [],
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
            cost_limit=min(
                args.ceiling_usd,
                (
                    RECOVERY_LIMIT_USD
                    if args.retry_pending
                    else RUN_LIMIT_USD
                    if args.axis == "name"
                    else 5.0
                )
                if args.arm == "cascade"
                else 55.0,
            ),
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
                    rubric_total=route_data.get("rubric_total"),
                    dimension_scores=route_data.get("dimension_scores", {}),
                    failed_checks=route_data.get("failed_check_details", []),
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
            if args.retry_pending:
                record["receipt_rows"] = cost_rows
            if not cost_rows:
                raise RuntimeError("review produced no cost receipts")
            record["per_model_spend_usd"] = {
                model: round(value, 6) for model, value in sorted(by_model.items())
            }
            record["spend_usd"] = round(sum(by_model.values()), 6)
            if args.arm == "cascade" and args.axis == "name" and args.parts == 1:
                record["prior_attempt_spend_usd"] = (
                    base["piece_spend_total_usd"] if base else PRIOR_SPEND_USD
                )
            record["piece_spend_total_usd"] = round(
                record["spend_usd"] + record.get("prior_attempt_spend_usd", 0), 6
            )
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
            record["incomplete_ids"] = [
                item["id"] for item in record["items"] if item["decision"] == "drafted"
            ]
            record["scratch_evidence"] = count_run(gc, summary.run_id)
        # Preserve every paid item and receipt total before checking containment.
        save(path, record)
        if base:
            combined = combine_recovery(base, record)
            save(combined_path, combined)
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
        verify_completion(record)
        if base:
            combined["recovery"]["containment"] = record["containment"]
            combined["recovery"]["containment_verdict"] = record["containment_verdict"]
            verify_containment(base)
            verify_completion(combined)
            if combined["piece_spend_total_usd"] > args.ceiling_usd:
                raise RuntimeError("combined piece exceeds its spend ceiling")
            combined["containment_verdict"] = "pass"
            combined["code_revision"] = record["code_revision"]
            combined["finished_at"] = stamp()
            save(combined_path, combined)
            combined_ready = True
        if (
            len(record["items"]) != len(ids)
            or record["piece_spend_total_usd"] > args.ceiling_usd
        ):
            raise RuntimeError("piece cardinality or spend ceiling failed")
        record["selected_count"] = len(ids)
        record["measurement_complete"] = True
        record["comparison_eligible"] = not args.retry_pending
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
        if base and combined_path and combined_path.exists():
            combined = json.loads(combined_path.read_text())
            combined.setdefault("recovery", {})["scratch_stopped"] = record.get(
                "scratch_stopped", False
            )
            combined["recovery"]["scratch_removed"] = record["scratch_removed"]
            if combined_ready and record.get("scratch_stopped") and not root.exists():
                combined["measurement_complete"] = True
                combined["comparison_eligible"] = True
                combined["status"] = "complete"
            save(combined_path, combined)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("incumbent", "cascade"))
    parser.add_argument("--axis", choices=("name", "docs"))
    parser.add_argument("--part", type=int, default=1)
    parser.add_argument("--parts", type=int, default=1)
    parser.add_argument("--ceiling-usd", type=float, default=PIECE_CEILING_USD)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--retry-pending", action="store_true")
    parser.add_argument("--check-containment", type=Path)
    parser.add_argument("--check-record", type=Path)
    args = parser.parse_args()
    if args.check_containment:
        verify_containment(json.loads(args.check_containment.read_text()))
        return
    if args.check_record:
        verify_completion(json.loads(args.check_record.read_text()))
        return
    if args.preflight:
        preflight()
        return
    if args.arm is None or args.axis is None:
        parser.error("--arm and --axis are required for a review piece")
    if args.parts < 1 or not 1 <= args.part <= args.parts:
        parser.error("--part must be between 1 and --parts")
    spent_usd = check_comparison_budget(args.ceiling_usd)
    import subprocess

    record = {
        "arm": args.arm,
        "axis": args.axis,
        "part": args.part,
        "parts": args.parts,
        "ceiling_usd": args.ceiling_usd,
        "comparison_spent_before_usd": float(spent_usd),
        "code_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "module_path": __file__,
        "working_directory": str(Path.cwd().resolve()),
        "created_at": stamp(),
        "status": "running",
    }
    suffix = f"-part{args.part}of{args.parts}" if args.parts > 1 else ""
    path = RESULTS / (
        "cascade-name-recovery.json"
        if args.retry_pending
        else f"{args.arm}-{args.axis}{suffix}.json"
    )
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
