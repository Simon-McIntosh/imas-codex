"""Build a blind label sheet of the documentation items two judges route differently.

The paid panel and the in-process cascade each judged the same fixed cohort of
documentation items. This runner re-derives, from the cohort's piece files, the
items where the two judges' routes disagree, draws seeded agreement controls,
loads the archived graph export into a disposable Neo4j instance, and writes a
blind sheet whose rows carry only the item's name, unit, DD source path with its
description and unit, and the stored documentation text.

The mapping from a sheet row back to the item, its class and both judges' routes
is written to a separate key file that the labeller never sees.

Read-only with respect to the live graph: the archive is loaded into a scratch
instance and removed on exit. No graph write and no model call.
"""

from __future__ import annotations

import json
import os
import random
import shutil
import socket
import subprocess
import tarfile
import tempfile
import traceback
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from imas_codex.graph.client import GraphClient
from imas_codex.graph.temp_neo4j import start_temp_neo4j, stop_temp_neo4j

HERE = Path(__file__).resolve().parent
COHORT = HERE / "srtj-cohort"
JEV_PIECE = COHORT / "cascade-docs-jev-only.json"
INCUMBENT_PIECES = (
    COHORT / "incumbent-docs-part1of4.json",
    COHORT / "incumbent-docs-part2of4.json",
    COHORT / "incumbent-docs-part3of4.json",
)
ARCHIVE = (
    Path.home()
    / ".local/share/imas-codex/exports/imas-codex-graph-dev-af50a71-20261018T114748Z.tar.gz"
)

SHEET = HERE / "srtj-docs-disagreement-sheet.json"
KEY = HERE / "srtj-docs-disagreement-key.json"
HTML = HERE / "srtj-docs-disagreement-sheet.html"

SHUFFLE_SEED = 20261022
CONTROL_SEED = 20261022
TARGET_CONTROLS_PER_CLASS = 18

# The seven fields a sheet row may carry. Anything else is a leak.
ALLOWED_ROW_FIELDS = (
    "sheet_id",
    "name",
    "unit",
    "source_path",
    "source_description",
    "source_unit",
    "documentation",
)
# Values that would give the labeller the answer away.
FORBIDDEN_ROW_KEYS = (
    "route",
    "jev_route",
    "panel_route",
    "score",
    "rubric_total",
    "comment",
)

STAMP = datetime.now(UTC).isoformat()


def log(message: str) -> None:
    print(f"[{datetime.now(UTC).strftime('%H:%M:%S')}] {message}", flush=True)


# ---------------------------------------------------------------------------
# Derivation from the cohort pieces
# ---------------------------------------------------------------------------


def load_routes() -> tuple[dict[str, dict], dict[str, dict]]:
    jev = {item["id"]: item for item in json.loads(JEV_PIECE.read_text())["items"]}
    incumbent: dict[str, dict] = {}
    for path in INCUMBENT_PIECES:
        for item in json.loads(path.read_text())["items"]:
            if item["id"] in incumbent:
                raise RuntimeError(f"duplicate incumbent item id {item['id']}")
            incumbent[item["id"]] = item
    return jev, incumbent


def derive(jev: dict[str, dict], incumbent: dict[str, dict]) -> dict:
    """Split the shared items into disagreement and agreement classes.

    Both judges' recorded route fields are used unchanged; no threshold is
    invented here. `refine` is the non-accept route a reviewer sends to the
    refinement pool.
    """
    shared = sorted(set(jev) & set(incumbent))

    def route(item: dict) -> str:
        return item.get("route") or "unjudged"

    jev_accept_panel_refine = [
        i
        for i in shared
        if route(jev[i]) == "accept" and route(incumbent[i]) == "refine"
    ]
    panel_accept_jev_refine = [
        i
        for i in shared
        if route(incumbent[i]) == "accept" and route(jev[i]) == "refine"
    ]
    both_accept = [
        i
        for i in shared
        if route(jev[i]) == "accept" and route(incumbent[i]) == "accept"
    ]
    both_refine = [
        i
        for i in shared
        if route(jev[i]) == "refine" and route(incumbent[i]) == "refine"
    ]

    disagreement = sorted(jev_accept_panel_refine + panel_accept_jev_refine)

    return {
        "shared_ids": shared,
        "classes": {
            "disagreement_jev_accept_panel_refine": jev_accept_panel_refine,
            "disagreement_panel_accept_jev_refine": panel_accept_jev_refine,
            "control_both_accept": both_accept,
            "control_both_refine": both_refine,
        },
        "disagreement": disagreement,
    }


def score_counts(
    jev: dict[str, dict], incumbent: dict[str, dict], threshold: float = 0.85
) -> dict:
    """The same split read from each arm's stored score at the consumer cutoff.

    Reported beside the route split so a reader can see how much the choice of
    reading moves each count.
    """
    shared = sorted(set(jev) & set(incumbent))

    def acc(item: dict) -> bool:
        return (item.get("score") or 0.0) >= threshold

    return {
        "threshold": threshold,
        "jev_accept_panel_refine": sum(
            1 for i in shared if acc(jev[i]) and not acc(incumbent[i])
        ),
        "panel_accept_jev_refine": sum(
            1 for i in shared if acc(incumbent[i]) and not acc(jev[i])
        ),
        "both_accept": sum(1 for i in shared if acc(jev[i]) and acc(incumbent[i])),
        "both_refine": sum(
            1 for i in shared if not acc(jev[i]) and not acc(incumbent[i])
        ),
    }


# ---------------------------------------------------------------------------
# Conditions the sheet must never be written under
# ---------------------------------------------------------------------------


class SheetLeakError(ValueError):
    """A sheet row would hand the labeller a judge's route, score or comment."""


def assert_sheet_row(row: dict) -> None:
    extra = set(row) - set(ALLOWED_ROW_FIELDS)
    if extra:
        raise SheetLeakError(f"row carries fields outside the seven: {sorted(extra)}")
    present = [key for key in FORBIDDEN_ROW_KEYS if key in row]
    if present:
        raise SheetLeakError(
            f"row carries a judge's route, score or comment: {present}"
        )
    missing = [key for key in ALLOWED_ROW_FIELDS if key not in row]
    if missing:
        raise SheetLeakError(f"row is missing required fields: {missing}")


def refusal_demonstration(sample_row: dict) -> dict:
    """Show the guard refusing a planted row, so a green write is not mistaken for a working check."""
    checks: list[dict] = []
    planted = dict(sample_row)
    planted["route"] = "accept"
    try:
        assert_sheet_row(planted)
    except SheetLeakError as exc:
        checks.append({"planted": "judge route", "refused": True, "message": str(exc)})
    else:
        raise RuntimeError("planted row carrying a judge's route was accepted")

    extra_field = dict(sample_row)
    extra_field["jev_route"] = "refine"
    try:
        assert_sheet_row(extra_field)
    except SheetLeakError as exc:
        checks.append(
            {"planted": "field outside the seven", "refused": True, "message": str(exc)}
        )
    else:
        raise RuntimeError("planted row carrying an eighth field was accepted")

    return {"checks": checks, "all_refused": all(check["refused"] for check in checks)}


# ---------------------------------------------------------------------------
# Scratch graph
# ---------------------------------------------------------------------------


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def make_instance(root: Path, dump: Path) -> tuple[Path, object, Path, str]:
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
        if previous_bind is None:
            os.environ.pop("APPTAINER_BIND", None)
        else:
            os.environ["APPTAINER_BIND"] = previous_bind
    return place, proc, temp_log, f"bolt://127.0.0.1:{bolt}"


def neo4j_process_count() -> int:
    result = subprocess.run(["ps", "-eo", "args="], capture_output=True, text=True)
    return sum(
        1
        for line in result.stdout.splitlines()
        if "neo4j" in line and "ps -eo" not in line
    )


def fetch_items(gc: GraphClient, ids: list[str]) -> dict[str, dict]:
    """Read the sheet's fields for each item from the archived export."""
    rows: dict[str, dict] = {}
    for start in range(0, len(ids), 50):
        chunk = ids[start : start + 50]
        records = gc.query(
            "UNWIND $ids AS id "
            "MATCH (sn:StandardName {id:id}) "
            "OPTIONAL MATCH (src:StandardNameSource {source_type:'dd'})-[:PRODUCED_NAME]->(sn) "
            "RETURN id AS item_id, properties(sn) AS sn, properties(src) AS src",
            ids=chunk,
        )
        for record in records:
            item_id = record["item_id"]
            if item_id in rows:
                continue
            sn = record["sn"]
            src = record["src"] or {}
            path = str(src.get("dd_path") or src.get("source_id") or "").removeprefix(
                "dd:"
            )
            documentation = sn.get("documentation")
            if documentation is None:
                documentation = ""
            rows[item_id] = {
                "name": sn.get("name") or item_id,
                "unit": sn.get("unit") or src.get("dd_unit") or "",
                "source_path": path,
                "source_description": src.get("dd_documentation") or "",
                "source_unit": src.get("dd_unit") or sn.get("unit") or "",
                "documentation": documentation,
            }
    return rows


# ---------------------------------------------------------------------------
# Sheet assembly
# ---------------------------------------------------------------------------


def build_sheet(
    derived: dict, fields: dict[str, dict]
) -> tuple[list[dict], list[dict]]:
    rng = random.Random(CONTROL_SEED)

    disagreement_ids = list(
        derived["classes"]["disagreement_jev_accept_panel_refine"]
    ) + list(derived["classes"]["disagreement_panel_accept_jev_refine"])
    both_accept = list(derived["classes"]["control_both_accept"])
    both_refine = list(derived["classes"]["control_both_refine"])

    rng.shuffle(disagreement_ids)
    rng.shuffle(both_accept)
    rng.shuffle(both_refine)

    chosen_controls = (
        both_accept[:TARGET_CONTROLS_PER_CLASS]
        + both_refine[:TARGET_CONTROLS_PER_CLASS]
    )

    selected = [(item_id, "disagreement") for item_id in disagreement_ids] + [
        (item_id, "control") for item_id in chosen_controls
    ]

    shuffle = random.Random(SHUFFLE_SEED)
    shuffle.shuffle(selected)
    missing = [item_id for item_id, _ in selected if item_id not in fields]
    if missing:
        raise RuntimeError(f"sheet items absent from the export: {missing[:8]}")

    sheet_rows: list[dict] = []
    key_rows: list[dict] = []
    for index, (item_id, klass) in enumerate(selected, start=1):
        sheet_id = f"S{index:04d}"
        values = fields[item_id]
        row = {
            "sheet_id": sheet_id,
            "name": values["name"],
            "unit": values["unit"],
            "source_path": values["source_path"],
            "source_description": values["source_description"],
            "source_unit": values["source_unit"],
            "documentation": values["documentation"],
        }
        assert_sheet_row(row)
        sheet_rows.append(row)
        key_rows.append(
            {
                "sheet_id": sheet_id,
                "item_id": item_id,
                "class": klass,
                "jev_route": None,
                "panel_route": None,
            }
        )
    return sheet_rows, key_rows


def main() -> int:
    log(f"worktree {Path.cwd()}")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    log(f"revision {revision}")

    jev, incumbent = load_routes()
    derived = derive(jev, incumbent)
    counts = {name: len(ids) for name, ids in derived["classes"].items()}
    log(f"shared documentation items: {len(derived['shared_ids'])}")
    for name, count in counts.items():
        log(f"  {name}: {count}")

    scratch = Path(
        tempfile.mkdtemp(prefix="disagreement-sheet-", dir=os.environ["TMPDIR"])
    )
    log(f"scratch tree {scratch}")
    proc = None
    prior_uri = os.environ.get("NEO4J_URI")
    payload: dict = {
        "schema": "srtj-docs-disagreement-sheet",
        "created_at": STAMP,
        "code_revision": revision,
        "module_path": __file__,
        "working_directory": str(Path.cwd()),
        "source_pieces": [JEV_PIECE.name, *[p.name for p in INCUMBENT_PIECES]],
        "archive": str(ARCHIVE),
        "seeds": {"shuffle": SHUFFLE_SEED, "control": CONTROL_SEED},
        "derivation": counts,
        "score_reading": score_counts(jev, incumbent),
        "targets": {
            "disagreement": 72,
            "controls_per_class": TARGET_CONTROLS_PER_CLASS,
        },
    }
    try:
        processes = neo4j_process_count()
        log(f"neo4j processes before load: {processes}")
        if processes > 6:
            raise RuntimeError(
                f"{processes} neo4j processes running; the guard allows at most 6"
            )
        with tarfile.open(ARCHIVE, "r:gz") as archive:
            member = next(m for m in archive if m.name.endswith("/graph.dump"))
            dump = scratch / "neo4j.dump"
            with archive.extractfile(member) as source, dump.open("wb") as target:
                shutil.copyfileobj(source, target)
        place, proc, temp_log, uri = make_instance(scratch, dump)
        os.environ["NEO4J_URI"] = uri
        payload["scratch_instance"] = {
            "path": str(place),
            "uri": uri,
            "log": str(temp_log),
        }

        candidates = sorted(set(jev) & set(incumbent))
        with GraphClient(uri=uri) as gc:
            fields = fetch_items(gc, candidates)
        log(f"export returned {len(fields)} of {len(candidates)} candidate items")
        if not fields:
            raise RuntimeError(
                "the export returned no items; the load produced nothing"
            )

        sample_id = next(iter(fields))
        payload["export_probe"] = {"item_id": sample_id, **fields[sample_id]}

        sheet_rows, key_rows = build_sheet(derived, fields)
        payload["row_counts"] = {
            "total": len(sheet_rows),
            "disagreement": sum(
                1 for row in key_rows if row["class"] == "disagreement"
            ),
            "control": sum(1 for row in key_rows if row["class"] == "control"),
        }
        payload["refusal"] = refusal_demonstration(sheet_rows[0])
        payload["class_counts"] = dict(Counter(row["class"] for row in key_rows))
    except Exception as exc:
        payload["status"] = "failed"
        payload["failure"] = f"{type(exc).__name__}: {exc}"
        payload["traceback"] = traceback.format_exc()
        SHEET.write_text(json.dumps(payload, indent=2) + "\n")
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

    payload["status"] = "accepted"
    SHEET.write_text(
        json.dumps({"record": payload, "rows": sheet_rows}, indent=2) + "\n"
    )
    KEY.write_text(json.dumps({"record": payload, "key": key_rows}, indent=2) + "\n")
    (HERE / "srtj-docs-disagreement-sheet.html").write_text(render_html(payload))
    log(f"wrote {SHEET.name}, {KEY.name}, {HTML.name}")
    log("EXIT=0")
    return 0


def render_html(payload: dict) -> str:
    counts = payload["derivation"]
    rows = payload["row_counts"]
    refusal = payload["refusal"]
    return (
        "<!doctype html>\n<html><head><meta charset='utf-8'>"
        "<title>Documentation disagreement sheet</title></head><body>\n"
        "<h1>Documentation disagreement sheet</h1>\n"
        f"<p>Rows: {rows['total']} total, {rows['disagreement']} disagreement, "
        f"{rows['control']} control.</p>\n"
        f"<p>Disagreement classes: Jev accept / panel refine {counts['disagreement_jev_accept_panel_refine']}, "
        f"panel accept / Jev refine {counts['disagreement_panel_accept_jev_refine']}.</p>\n"
        f"<p>Agreement classes available: both accept {counts['control_both_accept']}, "
        f"both refine {counts['control_both_refine']}.</p>\n"
        f"<p>Seeds: shuffle {payload['seeds']['shuffle']}, control {payload['seeds']['control']}.</p>\n"
        f"<p>Refusal check: {refusal['all_refused']} on {len(refusal['checks'])} planted rows.</p>\n"
        "</body></html>\n"
    )


if __name__ == "__main__":
    raise SystemExit(main())
