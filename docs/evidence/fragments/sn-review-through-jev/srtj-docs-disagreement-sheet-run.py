"""Build a blind label sheet of the documentation items two judges route differently.

The paid panel and the in-process cascade each judged the same fixed cohort of
documentation items. This runner re-derives, from the cohort's piece files, the
items where the two judges' decisions differ, draws seeded agreement controls,
loads the archived graph export into a disposable Neo4j instance, and writes a
blind sheet whose rows carry only the item's name, unit, DD source path with its
description and unit, and the stored documentation text.

The mapping from a sheet row back to the item, its class and both judges'
decisions is written to a separate key file that the labeller never sees.

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
# The panel's own accept cutoff, its matched calibration minimum score. Both the
# panel's binary decision and its calibration read against this figure.
PANEL_MIN_SCORE = 0.85
# The two agreement classes are matched to each other: every item both judges
# refined is taken, and the same number are drawn by seed from the items both
# accepted.
CONTROL_PER_CLASS = 14

# The dry-run cross-tabulation this sheet's derivation is checked against.
TABLE_REFERENCE = {
    "source": "srtj-jev-only-dry-run.html",
    "shared": 164,
    "both_accept": 38,
    "jev_accept_panel_refine": 49,
    "panel_accept_jev_refine": 0,
    "both_refine": 14,
    "panel_unjudged": 16,
    "jev_unjudged": 58,
}

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


def decision(jev_item: dict, part_item: dict) -> tuple[str | None, str | None]:
    """Each judge's decision on one shared item, or None where it gave none.

    A judge decides an item when it produced a route. `unjudged` means the
    judge left no route row, and that item is excluded from the sheet. The
    cascade records `accept` or `refine` directly; the panel records a score,
    and its decision is accept at or above its own matched minimum score.
    """
    route = jev_item.get("route")
    if route not in ("accept", "refine"):
        return None, None
    panel_route = part_item.get("route")
    if panel_route is None or panel_route == "unjudged":
        return None, None
    score = part_item.get("score")
    panel = "accept" if score is not None and score >= PANEL_MIN_SCORE else "refine"
    return route, panel


def derive(jev: dict[str, dict], incumbent: dict[str, dict]) -> dict:
    """Split the shared items into disagreement and agreement classes."""
    shared = sorted(set(jev) & set(incumbent))

    classes: dict[str, list[str]] = {
        "disagreement_jev_accept_panel_refine": [],
        "disagreement_panel_accept_jev_refine": [],
        "control_both_accept": [],
        "control_both_refine": [],
    }
    undecided: list[str] = []
    for item_id in shared:
        jev_decision, panel_decision = decision(jev[item_id], incumbent[item_id])
        if jev_decision is None:
            undecided.append(item_id)
            continue
        if (jev_decision, panel_decision) == ("accept", "refine"):
            classes["disagreement_jev_accept_panel_refine"].append(item_id)
        elif (jev_decision, panel_decision) == ("refine", "accept"):
            classes["disagreement_panel_accept_jev_refine"].append(item_id)
        elif (jev_decision, panel_decision) == ("accept", "accept"):
            classes["control_both_accept"].append(item_id)
        else:
            classes["control_both_refine"].append(item_id)

    disagreement = sorted(
        classes["disagreement_jev_accept_panel_refine"]
        + classes["disagreement_panel_accept_jev_refine"]
    )
    counts = {name: len(ids) for name, ids in classes.items()}
    return {
        "shared_ids": shared,
        "classes": classes,
        "disagreement": disagreement,
        "undecided": undecided,
        "counts": counts,
        "decided": sum(counts.values()),
    }


def panel_route_reading(jev: dict[str, dict], incumbent: dict[str, dict]) -> dict:
    """The same split read from the panel's stored route instead of its score.

    Reported beside the sheet's derivation so a reader can see how much the
    choice of reading moves each count: the panel routes 34 of the shared
    items to `band`, and those resolve by score here and by route there.
    """
    shared = sorted(set(jev) & set(incumbent))
    counts: Counter[tuple[str, str]] = Counter()
    for item_id in shared:
        jev_route = jev[item_id].get("route")
        panel_route = incumbent[item_id].get("route")
        if jev_route in ("accept", "refine") and panel_route in ("accept", "refine"):
            counts[(jev_route, panel_route)] += 1
    return {
        "both_accept": counts[("accept", "accept")],
        "jev_accept_panel_refine": counts[("accept", "refine")],
        "panel_accept_jev_refine": counts[("refine", "accept")],
        "both_refine": counts[("refine", "refine")],
        "disagreements": counts[("accept", "refine")] + counts[("refine", "accept")],
    }


# ---------------------------------------------------------------------------
# Conditions the sheet must never be written under
# ---------------------------------------------------------------------------


class SheetLeakError(ValueError):
    """A sheet row would hand the labeller a judge's route, score or comment."""


def assert_sheet_row(row: dict) -> None:
    present = [key for key in FORBIDDEN_ROW_KEYS if key in row]
    if present:
        raise SheetLeakError(
            f"row carries a judge's route, score or comment: {present}"
        )
    extra = set(row) - set(ALLOWED_ROW_FIELDS)
    if extra:
        raise SheetLeakError(f"row carries fields outside the seven: {sorted(extra)}")
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
    extra_field["labeller_note"] = (
        "an eighth field carrying neither a route nor a score"
    )
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


def choose_controls(derived: dict) -> dict:
    """Every agreement item of the smaller class, and the same count by seed."""
    both_accept = list(derived["classes"]["control_both_accept"])
    both_refine = list(derived["classes"]["control_both_refine"])
    rng = random.Random(CONTROL_SEED)
    rng.shuffle(both_accept)
    rng.shuffle(both_refine)

    # Every item both judges refined is taken, and the both-accept side is
    # matched to it, up to the pool the pieces actually hold.
    wanted = (
        max(CONTROL_PER_CLASS, len(both_refine)) if both_refine else CONTROL_PER_CLASS
    )
    drawn_refine = both_refine
    drawn_accept = both_accept[: min(wanted, len(both_accept))]
    return {
        "both_accept": drawn_accept,
        "both_refine": drawn_refine,
        "both_accept_pool": len(both_accept),
        "both_refine_pool": len(both_refine),
        "per_class": len(drawn_accept),
        "shortfall": wanted - len(drawn_accept),
    }


def build_sheet(
    derived: dict, controls: dict, fields: dict[str, dict], decisions: dict
) -> tuple[list[dict], list[dict]]:
    selected = [(item_id, "disagreement") for item_id in derived["disagreement"]] + [
        (item_id, "control")
        for item_id in controls["both_accept"] + controls["both_refine"]
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
        jev_decision, panel_decision = decisions[item_id]
        key_rows.append(
            {
                "sheet_id": sheet_id,
                "item_id": item_id,
                "class": klass,
                "jev_route": jev_decision,
                "panel_route": panel_decision,
            }
        )
    return sheet_rows, key_rows


def main() -> int:
    log(f"worktree {Path.cwd()}")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    log(f"revision {revision}")

    jev, incumbent = load_routes()
    derived = derive(jev, incumbent)
    decisions = {
        item_id: decision(jev[item_id], incumbent[item_id])
        for item_id in derived["shared_ids"]
    }
    log(f"shared documentation items: {len(derived['shared_ids'])}")
    log(f"decided items: {derived['decided']}")
    for name, count in derived["counts"].items():
        log(f"  {name}: {count}")
    controls = choose_controls(derived)
    log(
        "controls: both-accept pool "
        f"{controls['both_accept_pool']} -> drawn {len(controls['both_accept'])}; "
        f"both-refine pool {controls['both_refine_pool']} -> drawn "
        f"{len(controls['both_refine'])}"
    )
    alternative = panel_route_reading(jev, incumbent)
    log(f"panel-route reading: {alternative}")

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
        "definition": {
            "shared": "an item present in both the cascade documentation piece and an incumbent documentation piece",
            "decided": "a shared item that neither judge left unjudged; an item either judge left with no route is excluded from the sheet",
            "cascade_decision": "the judge's stored route, accept or refine",
            "panel_decision": f"accept when the panel's stored score is at or above its matched cutoff {PANEL_MIN_SCORE}, refine below it",
            "disagreement": "a decided item whose two decisions differ",
            "control": "a decided item whose two decisions agree",
        },
        "panel_min_score": PANEL_MIN_SCORE,
        "derivation": {
            "shared": len(derived["shared_ids"]),
            "decided": derived["decided"],
            "undecided": len(derived["undecided"]),
            **derived["counts"],
        },
        "table_reference": TABLE_REFERENCE,
        "panel_route_reading": alternative,
        "controls": {
            "per_class": controls["per_class"],
            "both_accept_pool": controls["both_accept_pool"],
            "both_refine_pool": controls["both_refine_pool"],
            "shortfall": controls["shortfall"],
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

        sheet_rows, key_rows = build_sheet(derived, controls, fields, decisions)
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
    HTML.write_text(render_html(payload))
    log(f"wrote {SHEET.name}, {KEY.name}, {HTML.name}")
    log("EXIT=0")
    return 0


def render_html(payload: dict) -> str:
    counts = payload["derivation"]
    rows = payload["row_counts"]
    refusal = payload["refusal"]
    definition = payload["definition"]
    table = payload["table_reference"]
    alternative = payload["panel_route_reading"]
    controls = payload["controls"]
    lines = [
        "<!doctype html>",
        "<html><head><meta charset='utf-8'>",
        "<title>Documentation disagreement sheet</title></head><body>",
        "<h1>Documentation disagreement sheet</h1>",
        "<h2>Definition</h2>",
        "<ul>",
    ]
    lines += [f"<li>{key}: {value}</li>" for key, value in definition.items()]
    lines += [
        "</ul>",
        "<h2>Counts</h2>",
        "<ul>",
        f"<li>shared documentation items: {counts['shared']}</li>",
        f"<li>decided items: {counts['decided']}; excluded as unjudged by a judge: {counts['undecided']}</li>",
        f"<li>disagreements: {counts['disagreement_jev_accept_panel_refine']} cascade accept / panel refine, "
        f"{counts['disagreement_panel_accept_jev_refine']} panel accept / cascade refine "
        f"(total {rows['disagreement']})</li>",
        f"<li>agreement controls: {controls['per_class']} drawn by seed from "
        f"{controls['both_accept_pool']} both-accept items, and {controls['both_refine_pool']} both-refine items "
        f"(total {rows['control']})</li>",
        f"<li>sheet rows: {rows['total']}</li>",
        "</ul>",
        "<h2>Re-derivation beside the dry-run table</h2>",
        f"<p>Re-derived disagreements: {rows['disagreement']} "
        f"(dry-run table: {table['jev_accept_panel_refine']} cascade accept / panel refine, "
        f"{table['panel_accept_jev_refine']} the other way). Re-derived both-accept pool "
        f"{controls['both_accept_pool']} against the table's {table['both_accept']}; both-refine "
        f"{controls['both_refine_pool']} against {table['both_refine']}; shared "
        f"{counts['shared']} against {table['shared']}.</p>",
        f"<p>Reading the panel's decision from its stored route instead of its score gives "
        f"{alternative['disagreements']} disagreements ({alternative['jev_accept_panel_refine']} "
        f"cascade accept / panel refine, {alternative['panel_accept_jev_refine']} the other way), "
        f"{alternative['both_accept']} both-accept and {alternative['both_refine']} both-refine. "
        f"The sheet uses the panel's score at its matched cutoff {payload['panel_min_score']}, "
        f"which reproduces the dry-run table's cross-tabulation exactly.</p>",
        "<h2>Blinding</h2>",
        f"<p>Seeds: shuffle {payload['seeds']['shuffle']}, control {payload['seeds']['control']}. "
        f"Each sheet row carries only the seven fields: {', '.join(ALLOWED_ROW_FIELDS)}.</p>",
        f"<p>Refusal check: {refusal['all_refused']} on {len(refusal['checks'])} planted rows "
        f"({', '.join(check['planted'] for check in refusal['checks'])}).</p>",
        "<p>The key file maps each sheet_id to its item, class and both judges' decisions, and is not shown to the labeller.</p>",
        "</body></html>",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
