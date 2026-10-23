"""Measure the decision the documentation change would make over the 77 labelled rows.

Every labelled row is rebuilt the way the cascade expects it -- documentation,
name and unit from the sheet the labeller saw, the remainder from the archived
export loaded into a disposable scratch Neo4j -- and put through the product's
own documentation-deciding path: the Jev route from review/cascade.py, any hold
the route raises, and any hold the deterministic error-severity lint cap raises.
A row is accepted only when the route accepts and no hold is open. The Jev calls
run twice, because Jev varies between runs.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import socket
import subprocess
import tarfile
import tempfile
import traceback
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

from imas_codex.graph.client import GraphClient
from imas_codex.graph.temp_neo4j import start_temp_neo4j, stop_temp_neo4j
from imas_codex.settings import get_model
from imas_codex.standard_names.review import cascade as cascade_mod
from imas_codex.standard_names.review.audits import (
    run_documentation_gate_checks,
    run_house_style_checks,
)

HERE = Path(__file__).resolve().parent
SHEET = HERE / "srtj-docs-disagreement-sheet.json"
KEY = HERE / "srtj-docs-disagreement-key.json"
LABELS = HERE / "srtj-docs-disagreement-labels.json"
ARCHIVE = (
    Path.home()
    / ".local/share/imas-codex/exports/imas-codex-graph-dev-af50a71-20261018T114748Z.tar.gz"
)
OUT = HERE / "srtj-docs-switch-measure.json"
MAX_NEO4J_PROCESSES = 4
JEV_CONCURRENCY = 8
SPEND_CEILING_USD = 1.00


def log(message: str) -> None:
    print(f"[{datetime.now(UTC).strftime('%H:%M:%S')}] {message}", flush=True)


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def neo4j_process_count() -> int:
    result = subprocess.run(
        ["ps", "-eo", "args="], capture_output=True, text=True, check=False
    )
    return sum(
        1
        for line in result.stdout.splitlines()
        if "neo4j" in line and line.strip() != "ps -eo args="
    )


def make_instance(root: Path, dump: Path):
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
        proc, temp_log = start_temp_neo4j(place, bolt, http)
    finally:
        if previous_bind is not None:
            os.environ["APPTAINER_BIND"] = previous_bind
    return place, proc, temp_log, f"bolt://127.0.0.1:{bolt}"


def build_item(item_id, sheet, export):
    item = {
        "id": item_id,
        "name": sheet.get("name") or item_id,
        "unit": sheet.get("unit") if sheet.get("unit") is not None else "",
        "documentation": sheet.get("documentation") or "",
        "description": "",
        "links": [],
    }
    if export:
        item["description"] = export.get("description") or ""
        item["links"] = list(export.get("links") or [])
        item["cocos_transformation_type"] = export.get("cocos_transformation_type")
        for key in ("source_paths", "related_neighbours", "dd_source_docs"):
            if export.get(key):
                item[key] = list(export[key])
    if not item.get("source_paths") and sheet.get("source_path"):
        item["source_paths"] = [sheet["source_path"]]
    if not item.get("dd_source_docs") and sheet.get("source_path"):
        item["dd_source_docs"] = [
            {
                "id": sheet["source_path"],
                "unit": sheet.get("source_unit") or "",
                "documentation": sheet.get("source_description") or "",
            }
        ]
    return item


def fetch_export_rows(ids):
    from imas_codex.standard_names.workers import (
        _canonical_docs_review_item,
        _enrich_docs_review_dd_context,
        _load_docs_review_source_item,
        _related_path_neighbours,
        _set_docs_review_dd_source_paths,
    )

    rows = {}
    for item_id in ids:
        prepared = None
        try:
            source = _load_docs_review_source_item(item_id)
            prepared = _canonical_docs_review_item(source)
            _set_docs_review_dd_source_paths(prepared)
            try:
                _enrich_docs_review_dd_context(prepared)
            except Exception:
                pass
            paths = prepared.get("source_paths") or []
            if paths:
                try:
                    related = _related_path_neighbours(GraphClient(), paths[0])
                except Exception:
                    related = []
                if related:
                    prepared["related_neighbours"] = related
        except Exception:
            prepared = None
        rows[item_id] = prepared
    return rows


def documentation_error_lint(candidate):
    findings = [
        *run_house_style_checks([candidate]),
        *run_documentation_gate_checks([candidate]),
    ]
    return [
        {
            "finding_type": f.finding_type,
            "severity": f.severity,
            "detail": (f.detail or "")[:300],
        }
        for f in findings
        if f.severity == "error"
    ]


def lint_hold_line(errors):
    if not errors:
        return None
    error = errors[0]
    return (
        f"[{error['severity']}:lint:{error['finding_type']}] "
        f"{error['finding_type']}: {error['detail']}"
    )


async def jev_pass(items, model):
    semaphore = asyncio.Semaphore(JEV_CONCURRENCY)

    async def one(item):
        async with semaphore:
            try:
                route, _answers, cost = await cascade_mod._docs_judgment(item, model)
            except Exception as exc:
                return {
                    "route": "unjudged",
                    "failed_checks": [],
                    "holds": [],
                    "cost": 0.0,
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            return {
                "route": route.route,
                "failed_checks": list(route.failed_checks),
                "holds": [dict(h) for h in route.holds],
                "cost": float(cost or 0.0),
                "reason": None,
            }

    results = await asyncio.gather(*[one(item) for item in items])
    return list(results), sum(r["cost"] for r in results)


def accepted(res, lint_hold):
    reasons = [f"route hold {h.get('kind')}: {h.get('finding')}" for h in res["holds"]]
    if lint_hold:
        reasons.append(f"lint cap hold docs_lint_error: {lint_hold}")
    return (res["route"] == "accept" and not reasons), reasons


def main() -> int:
    parser = argparse.ArgumentParser(description="measure the docs switch decision")
    parser.add_argument("--limit", type=int, default=0, help="first N rows only")
    args = parser.parse_args()

    sheet_by_id = {r["sheet_id"]: r for r in json.loads(SHEET.read_text())["rows"]}
    label_by_id = {r["sheet_id"]: r for r in json.loads(LABELS.read_text())["labels"]}
    key_rows = list(json.loads(KEY.read_text())["key"])
    if args.limit:
        key_rows = key_rows[: args.limit]

    payload = {
        "schema": "srtj-docs-switch-measure",
        "node": "srtj-docs-switch-measure",
        "created_at": datetime.now(UTC).isoformat(),
        "code_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "module_path": str(Path(__file__).resolve()),
        "working_directory": os.getcwd(),
        "archive": str(ARCHIVE),
        "rows_measured": len(key_rows),
        "spend_ceiling_usd": SPEND_CEILING_USD,
        "decision_rule": (
            "accepted = the Jev docs route is accept AND no hold is open; a hold is "
            "open when the route raised one or the deterministic error-severity "
            "lint cap raised a docs_lint_error hold"
        ),
        "bar": (
            "all 19 disagreement rows labelled not acceptable (link defects) refined, "
            "and none of the 30 disagreement rows labelled acceptable refined, on "
            "both Jev passes"
        ),
    }

    scratch = Path(
        tempfile.mkdtemp(
            prefix="srtj-docs-switch-measure-", dir=os.environ.get("TMPDIR", "/tmp")
        )
    )
    proc = None
    prior_uri = os.environ.get("NEO4J_URI")
    try:
        processes = neo4j_process_count()
        log(f"neo4j processes before load: {processes}")
        if processes > MAX_NEO4J_PROCESSES:
            raise RuntimeError(
                f"{processes} neo4j processes running; "
                f"guard allows at most {MAX_NEO4J_PROCESSES}"
            )
        with tarfile.open(ARCHIVE, "r:gz") as archive:
            member = next(m for m in archive if m.name.endswith("/graph.dump"))
            dump = scratch / "neo4j.dump"
            with archive.extractfile(member) as source:
                with dump.open("wb") as target:
                    shutil.copyfileobj(source, target)
        place, proc, temp_log, uri = make_instance(scratch, dump)
        os.environ["NEO4J_URI"] = uri
        payload["scratch_instance"] = {
            "path": str(place),
            "uri": uri,
            "log": str(temp_log),
        }
        item_ids = [row["item_id"] for row in key_rows]
        export = fetch_export_rows(item_ids)
        missing = [i for i in item_ids if not export.get(i)]
        log(f"archive returned {len(item_ids) - len(missing)} of {len(item_ids)} items")
        if missing:
            raise RuntimeError(f"sheet items absent from the export: {missing[:8]}")
        rows = []
        doc_mismatch = []
        for key_row in key_rows:
            sid = key_row["sheet_id"]
            item_id = key_row["item_id"]
            sheet = sheet_by_id[sid]
            exp = export[item_id]
            if (exp.get("documentation") or "") != (sheet.get("documentation") or ""):
                doc_mismatch.append(sid)
            rows.append(
                {
                    "sheet_id": sid,
                    "item_id": item_id,
                    "class": key_row["class"],
                    "sheet_jev_route": key_row["jev_route"],
                    "panel_route": key_row["panel_route"],
                    "label_acceptable": label_by_id[sid]["label"] == 1,
                    "label_reason": label_by_id[sid]["reason"],
                    "item": build_item(item_id, sheet, exp),
                }
            )
        payload["documentation_match"] = {
            "compared": len(rows),
            "mismatches": doc_mismatch,
        }
        log(f"documentation sheet-vs-archive mismatches: {len(doc_mismatch)}")

        for row in rows:
            row["error_lint"] = documentation_error_lint(row["item"])
            row["lint_hold"] = lint_hold_line(row["error_lint"])

        defect_row = next(
            (
                r
                for r in rows
                if r["class"] == "disagreement" and not r["label_acceptable"]
            ),
            None,
        )
        controls = {}
        if defect_row is not None:
            controls["defect_row_real_error_lint"] = defect_row["error_lint"]
            injected = dict(defect_row["item"])
            injected["documentation"] = (
                defect_row["item"].get("documentation") or ""
            ) + "\n\nThe ratio follows [injected_bare_name] directly."
            controls["injected_bracket_error_lint"] = documentation_error_lint(injected)
            controls["instrument_fires_on_injected_bracket"] = any(
                f["finding_type"] == "link_hygiene"
                for f in controls["injected_bracket_error_lint"]
            )
        payload["positive_controls"] = controls
        if not controls.get("instrument_fires_on_injected_bracket"):
            raise RuntimeError(
                "the deterministic instrument did not fire on an injected "
                "snake_case bare bracket; the cap is unproven"
            )

        model = get_model("sn-judge")
        payload["jev_model"] = model
        items = [row["item"] for row in rows]
        spend_total = 0.0
        for pass_name in ("pass_a", "pass_b"):
            log(f"running Jev {pass_name} over {len(items)} rows through {model}")
            results, spend = asyncio.run(jev_pass(items, model))
            spend_total += spend
            for row, res in zip(rows, results, strict=True):
                row[pass_name] = res
            if spend_total > SPEND_CEILING_USD:
                raise RuntimeError(f"spend ceiling exceeded: {spend_total}")
            payload["spend_usd"] = round(spend_total, 6)
            payload["spend_within_ceiling"] = spend_total <= SPEND_CEILING_USD
            payload["pass_spend_usd"] = {
                k: round(sum(r[pass_name]["cost"] for r in rows), 6)
                for k in ("pass_a", "pass_b")
                if all(k in r for r in rows)
            }
            OUT.write_text(json.dumps(payload, indent=2) + "\n")
            log(f"{pass_name} complete; cumulative spend {round(spend_total, 6)} USD")

        report = {}
        wrong_by_pass = {}
        for pass_name in ("pass_a", "pass_b"):
            per_class = defaultdict(lambda: {"rows": 0, "accepted": 0, "refined": 0})
            wrong = []
            for row in rows:
                res = row[pass_name]
                ok, hold_reasons = accepted(res, row.get("lint_hold"))
                bucket = per_class[
                    "acceptable" if row["label_acceptable"] else "not-acceptable"
                ]
                bucket["rows"] += 1
                bucket["accepted" if ok else "refined"] += 1
                if row["class"] == "disagreement":
                    expected = "accept" if row["label_acceptable"] else "refine"
                    actual = "accept" if ok else "refine"
                    if expected != actual:
                        wrong.append(
                            {
                                "sheet_id": row["sheet_id"],
                                "item_id": row["item_id"],
                                "label_acceptable": row["label_acceptable"],
                                "label_reason": row["label_reason"],
                                "expected": expected,
                                "actual": actual,
                                "route": res["route"],
                                "failed_checks": res["failed_checks"],
                                "route_holds": res.get("holds", []),
                                "lint_hold": row.get("lint_hold"),
                                "hold_reasons": hold_reasons,
                                "route_failure_reason": res.get("reason"),
                            }
                        )
            link_defects = [
                r
                for r in rows
                if r["class"] == "disagreement" and not r["label_acceptable"]
            ]
            acceptable = [
                r
                for r in rows
                if r["class"] == "disagreement" and r["label_acceptable"]
            ]
            refined_defects = sum(
                1
                for r in link_defects
                if not accepted(r[pass_name], r.get("lint_hold"))[0]
            )
            refined_ok = sum(
                1
                for r in acceptable
                if not accepted(r[pass_name], r.get("lint_hold"))[0]
            )
            report[pass_name] = {
                "per_class": dict(sorted(per_class.items())),
                "link_defect_rows": len(link_defects),
                "link_defects_refined": refined_defects,
                "acceptable_rows": len(acceptable),
                "acceptable_rows_refined": refined_ok,
                "all_defects_refined": refined_defects == len(link_defects),
                "no_acceptable_refined": refined_ok == 0,
                "meets_bar": refined_defects == len(link_defects) and refined_ok == 0,
                "spend_usd": round(sum(r[pass_name]["cost"] for r in rows), 6),
            }
            wrong_by_pass[pass_name] = wrong

        payload["passes"] = report
        payload["wrong_rows"] = wrong_by_pass
        payload["per_row"] = [
            {
                "sheet_id": r["sheet_id"],
                "item_id": r["item_id"],
                "class": r["class"],
                "label_acceptable": r["label_acceptable"],
                "label_reason": r["label_reason"],
                "error_lint": r["error_lint"],
                "lint_hold": r.get("lint_hold"),
                "pass_a_route": r["pass_a"]["route"],
                "pass_a_failed_checks": r["pass_a"]["failed_checks"],
                "pass_a_holds": r["pass_a"]["holds"],
                "pass_b_route": r["pass_b"]["route"],
                "pass_b_failed_checks": r["pass_b"]["failed_checks"],
                "pass_b_holds": r["pass_b"]["holds"],
            }
            for r in rows
        ]
        payload["verdict"] = {
            "pass_a_meets_bar": report["pass_a"]["meets_bar"],
            "pass_b_meets_bar": report["pass_b"]["meets_bar"],
            "meets_bar_both_passes": report["pass_a"]["meets_bar"]
            and report["pass_b"]["meets_bar"],
        }
        payload["status"] = "accepted"
    except Exception as exc:
        payload["status"] = "failed"
        payload["failure"] = f"{type(exc).__name__}: {exc}"
        payload["traceback"] = traceback.format_exc()
        OUT.write_text(json.dumps(payload, indent=2) + "\n")
        raise
    finally:
        if prior_uri is None:
            os.environ.pop("NEO4J_URI", None)
        else:
            os.environ["NEO4J_URI"] = prior_uri
        if proc is not None:
            stop_temp_neo4j(proc)
            payload["scratch_stopped"] = True
        shutil.rmtree(scratch, ignore_errors=True)
        payload["scratch_removed"] = {
            "path": str(scratch),
            "exists_after": scratch.exists(),
        }

    payload["finished_at"] = datetime.now(UTC).isoformat()
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    log(f"wrote {OUT.name}")
    log("EXIT=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
