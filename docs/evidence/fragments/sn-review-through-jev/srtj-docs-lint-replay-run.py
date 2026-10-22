"""Replay the deterministic documentation checks over the 77 labelled disagreement rows.

The blind documentation sheet holds 49 items Jev accepts and the paid panel
refines, plus 28 seeded controls. A labeller marked 30 of the disagreements
acceptable and 19 not acceptable, every one of the 19 for a link defect. This
runner asks one question of the incumbent's deterministic lint: run the three
documentation checks the cascade's own audit path uses -- the registry
conformance checks, the documentation gates and the link-integrity check -- over
each labelled row and report what each class carries.

The catalog the judges saw is the set of ``StandardName`` ids in the cohort's
archived export, loaded into a disposable Neo4j instance and stopped and removed
on exit. Read-only with respect to the live graph: no graph write and no model
call.

Three positive controls prove the instrument can see a defect of each relevant
kind before any absence is reported:

* the documentation gates fire an error-severity ``link_hygiene`` finding on a
  snake_case bare bracket, which is the pattern the check is written to catch;
* the same gates pass on one of the 19 real defect rows' own documentation, and
  fail on that identical text with a snake_case bracket injected -- so the
  absence on the real rows is the check's pattern, not a dead instrument;
* the link-integrity check reports a ``dead_link`` for a structured link whose
  target is a real catalog name held out of the known set.
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import tarfile
import tempfile
import traceback
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

from imas_codex.graph.client import GraphClient
from imas_codex.graph.temp_neo4j import start_temp_neo4j, stop_temp_neo4j
from imas_codex.standard_names.review.audits import (
    run_documentation_gate_checks,
    run_house_style_checks,
    run_link_integrity,
)

HERE = Path(__file__).resolve().parent
SHEET = HERE / "srtj-docs-disagreement-sheet.json"
KEY = HERE / "srtj-docs-disagreement-key.json"
LABELS = HERE / "srtj-docs-disagreement-labels.json"
ARCHIVE = (
    Path.home()
    / ".local/share/imas-codex/exports/imas-codex-graph-dev-af50a71-20261018T114748Z.tar.gz"
)
OUT = HERE / "srtj-docs-lint-replay.json"

MAX_NEO4J_PROCESSES = 6
STAMP = datetime.now(UTC).isoformat()


def log(message: str) -> None:
    print(f"[{datetime.now(UTC).strftime('%H:%M:%S')}] {message}", flush=True)


# ---------------------------------------------------------------------------
# Scratch instance
# ---------------------------------------------------------------------------


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def neo4j_process_count() -> int:
    result = subprocess.run(["ps", "-eo", "args="], capture_output=True, text=True)
    return sum(
        1
        for line in result.stdout.splitlines()
        if "neo4j" in line and "ps -eo" not in line
    )


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


def catalog_ids(gc: GraphClient) -> list[str]:
    rows = gc.query("MATCH (sn:StandardName) RETURN sn.id AS id")
    return sorted({str(r["id"]) for r in rows if r.get("id")})


def fetch_items(gc: GraphClient, ids: list[str]) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    for start in range(0, len(ids), 50):
        chunk = ids[start : start + 50]
        for record in gc.query(
            "UNWIND $ids AS id "
            "MATCH (sn:StandardName {id:id}) "
            "RETURN id AS item_id, properties(sn) AS sn",
            ids=chunk,
        ):
            item_id = record["item_id"]
            if item_id in rows or record["sn"] is None:
                continue
            rows[item_id] = record["sn"]
    return rows


def _doc_link_targets(documentation: str) -> list[str]:
    """Return the ``(name:...)`` targets a documentation text links to."""
    import re

    return re.findall(r"\]\(name:([a-z][a-z0-9_]+)\)", documentation)


# ---------------------------------------------------------------------------
# The three deterministic checks
# ---------------------------------------------------------------------------


def run_lint_and_link_checks(
    candidate: dict, catalog: list[str], known_ids: set[str] | None = None
) -> tuple[list, list]:
    """Return (lint_findings, link_findings) for one candidate row.

    The cap's selector reads only ``lint_findings`` with ``severity == 'error'``,
    so the two lists, and the severities inside the first, are kept apart.
    """
    lint = [
        *run_house_style_checks([candidate]),
        *run_documentation_gate_checks([candidate]),
    ]
    link = run_link_integrity(
        [candidate],
        known_ids=set(catalog) if known_ids is None else known_ids,
    )
    return lint, link


def find_types(lint: list, severity: str | None = None) -> list[str]:
    return [f.finding_type for f in lint if severity is None or f.severity == severity]


def finding_summary(lint: list, link: list) -> dict:
    error_types = find_types(lint, "error")
    any_lint_types = find_types(lint)
    return {
        "error_lint": [
            {
                "finding_type": f.finding_type,
                "severity": f.severity,
                "detail": f.detail[:300],
            }
            for f in lint
            if f.severity == "error"
        ],
        "all_lint": [
            {
                "finding_type": f.finding_type,
                "severity": f.severity,
                "detail": f.detail[:300],
            }
            for f in lint
        ],
        "link_findings": [
            {
                "finding_type": f.finding_type,
                "target": f.target,
                "detail": f.detail[:300],
            }
            for f in link
        ],
        "error_lint_types": error_types,
        "any_lint_types": any_lint_types,
        "has_error_lint": bool(error_types),
        "has_any_finding": bool(any_lint_types) or bool(link),
    }


def positive_controls(catalog: list[str], sample_target: str, defect_doc: str) -> dict:
    """Prove the instrument fires on each defect kind before any absence is read."""
    catalog_set = set(catalog)

    bracket_doc = defect_doc + "\n\nSee [injected_bare_name] for the peer quantity."
    bracket_candidate = {
        "id": "control",
        "name": "control",
        "documentation": bracket_doc,
    }
    lint_a, _ = run_lint_and_link_checks(bracket_candidate, catalog)
    control_a = {
        "purpose": (
            "the documentation gates must fire an error-severity link_hygiene "
            "finding on a snake_case bare bracket"
        ),
        "documentation": bracket_doc,
        "lint_types": find_types(lint_a),
        "error_types": find_types(lint_a, "error"),
        "saw_link_hygiene_error": "link_hygiene" in find_types(lint_a, "error"),
    }

    real_candidate = {"id": "control", "name": "control", "documentation": defect_doc}
    lint_b, _ = run_lint_and_link_checks(real_candidate, catalog)
    control_b = {
        "purpose": (
            "the same real defect-row documentation passes link_hygiene, while the "
            "identical text with a snake_case bracket injected fails it -- so a "
            "failure is reachable on this text and the real rows are not seen"
        ),
        "documentation": defect_doc,
        "real_error_types": find_types(lint_b, "error"),
        "injected_error_types": find_types(lint_a, "error"),
        "real_saw_link_hygiene_error": "link_hygiene" in find_types(lint_b, "error"),
    }

    dead_candidate = {
        "id": "control",
        "name": "control",
        "documentation": "A quantity.",
        "links": [f"name:{sample_target}"],
    }
    _, link_c = run_lint_and_link_checks(dead_candidate, catalog, known_ids={"control"})
    control_c = {
        "purpose": (
            "the link-integrity check must report a dead_link for a structured "
            "link to a real catalog name held out of the known set"
        ),
        "target": sample_target,
        "target_in_catalog": sample_target in catalog_set,
        "link_types": [f.finding_type for f in link_c],
        "saw_dead_link": any(f.finding_type == "dead_link" for f in link_c),
    }

    return {
        "link_hygiene_bracket_fires": control_a,
        "defect_shape_not_seen": control_b,
        "out_of_scope_structured_link_dead": control_c,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    started = datetime.now(UTC).isoformat()
    key_rows = json.loads(KEY.read_text())["key"]
    label_rows = json.loads(LABELS.read_text())["labels"]
    sheet_rows = json.loads(SHEET.read_text())["rows"]
    sheet_by_id = {row["sheet_id"]: row for row in sheet_rows}
    label_by_id = {row["sheet_id"]: row for row in label_rows}

    payload: dict = {
        "schema": "srtj-docs-lint-replay",
        "created_at": str(STAMP),
        "started_at": started,
        "code_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "module_path": str(Path(__file__).resolve()),
        "working_directory": os.getcwd(),
        "archive": str(ARCHIVE),
    }

    scratch = Path(
        tempfile.mkdtemp(
            prefix="srtj-docs-lint-replay-", dir=os.environ.get("TMPDIR", "/tmp")
        )
    )
    proc = None
    prior_uri = os.environ.get("NEO4J_URI")
    try:
        processes = neo4j_process_count()
        log(f"neo4j processes before load: {processes}")
        if processes > MAX_NEO4J_PROCESSES:
            raise RuntimeError(
                f"{processes} neo4j processes running; the guard allows at most "
                f"{MAX_NEO4J_PROCESSES}"
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

        catalog = catalog_ids(GraphClient(uri=uri))
        log(f"catalog holds {len(catalog)} StandardName ids")
        if not catalog:
            raise RuntimeError("the export returned no ids; the load produced nothing")
        item_ids = [row["item_id"] for row in key_rows]
        fields = fetch_items(GraphClient(uri=uri), item_ids)
        missing = [i for i in item_ids if i not in fields]
        log(f"export returned {len(fields)} of {len(item_ids)} sheet items")
        if missing:
            raise RuntimeError(f"sheet items absent from the export: {missing[:8]}")

        defect_ids = [
            row["item_id"]
            for row in key_rows
            if row["class"] == "disagreement"
            and label_by_id[row["sheet_id"]]["label"] == 0
        ]
        sample_target = next((i for i in catalog if i != defect_ids[0]), catalog[0])
        defect_doc = fields[defect_ids[0]].get("documentation") or ""
        controls = positive_controls(catalog, sample_target, defect_doc)
        payload["positive_controls"] = controls
        log(
            "positive controls: link_hygiene_bracket_fires="
            f"{controls['link_hygiene_bracket_fires']['saw_link_hygiene_error']} "
            "defect_shape_not_seen="
            f"{controls['defect_shape_not_seen']['real_saw_link_hygiene_error']} "
            "out_of_scope_dead="
            f"{controls['out_of_scope_structured_link_dead']['saw_dead_link']}"
        )
        if not (
            controls["link_hygiene_bracket_fires"]["saw_link_hygiene_error"]
            and controls["out_of_scope_structured_link_dead"]["saw_dead_link"]
        ):
            raise RuntimeError(
                "a positive control did not fire; the instrument is unproven"
            )

        rows_out: list[dict] = []
        doc_mismatch: list[str] = []
        for key_row in key_rows:
            sid = key_row["sheet_id"]
            item_id = key_row["item_id"]
            sheet = sheet_by_id[sid]
            label = label_by_id[sid]
            sn = fields[item_id]
            documentation = sn.get("documentation") or ""
            if documentation != (sheet.get("documentation") or ""):
                doc_mismatch.append(sid)
            candidate = {
                "id": item_id,
                "name": sn.get("name") or item_id,
                "unit": sn.get("unit") or sheet.get("unit") or "",
                "description": sn.get("description") or "",
                "documentation": documentation,
                "cocos_transformation_type": sn.get("cocos_transformation_type"),
                "links": list(sn.get("links") or []),
            }
            lint, link = run_lint_and_link_checks(candidate, catalog)
            summary = finding_summary(lint, link)
            rows_out.append(
                {
                    "sheet_id": sid,
                    "item_id": item_id,
                    "class": key_row["class"],
                    "jev_route": key_row["jev_route"],
                    "panel_route": key_row["panel_route"],
                    "label_acceptable": label["label"] == 1,
                    "label_reason": label["reason"],
                    "documentation_chars": len(documentation),
                    "doc_link_targets": _doc_link_targets(documentation),
                    "stored_links": candidate["links"],
                    **summary,
                }
            )

        payload["documentation_match"] = {
            "compared": len(rows_out),
            "mismatches": doc_mismatch,
        }
        payload["catalog_id_count"] = len(catalog)
        payload["rows"] = rows_out

        def class_key(r: dict) -> tuple:
            return (
                r["class"],
                r["jev_route"],
                r["panel_route"],
                "acceptable" if r["label_acceptable"] else "not-acceptable",
            )

        groups: dict = defaultdict(list)
        for row in rows_out:
            groups[class_key(row)].append(row)
        per_class = []
        for gkey in sorted(groups):
            members = groups[gkey]
            per_class.append(
                {
                    "class": gkey[0],
                    "jev_route": gkey[1],
                    "panel_route": gkey[2],
                    "label": gkey[3],
                    "rows": len(members),
                    "rows_with_error_lint": sum(
                        1 for r in members if r["has_error_lint"]
                    ),
                    "rows_with_any_finding": sum(
                        1 for r in members if r["has_any_finding"]
                    ),
                    "error_lint_types": dict(
                        Counter(t for r in members for t in r["error_lint_types"])
                    ),
                    "any_finding_types": dict(
                        Counter(
                            t
                            for r in members
                            for t in (
                                r["any_lint_types"]
                                + [f["finding_type"] for f in r["link_findings"]]
                            )
                        )
                    ),
                }
            )
        payload["per_class"] = per_class

        defects = [
            r
            for r in rows_out
            if r["class"] == "disagreement" and not r["label_acceptable"]
        ]
        acceptable = [
            r
            for r in rows_out
            if r["class"] == "disagreement" and r["label_acceptable"]
        ]
        flagged_defects = [r for r in defects if r["has_error_lint"]]
        missed = [r for r in defects if not r["has_error_lint"]]
        false_flags = [r for r in acceptable if r["has_error_lint"]]
        payload["bar"] = {
            "defect_rows": len(defects),
            "defect_rows_flagged": len(flagged_defects),
            "acceptable_rows": len(acceptable),
            "acceptable_rows_flagged": len(false_flags),
            "meets_bar": len(flagged_defects) == len(defects) and not false_flags,
            "missed": [
                {
                    "sheet_id": r["sheet_id"],
                    "item_id": r["item_id"],
                    "defect": r["label_reason"],
                    "has_any_finding": r["has_any_finding"],
                    "any_lint_types": r["any_lint_types"],
                    "link_findings": r["link_findings"],
                    "stored_links": r["stored_links"],
                    "doc_link_targets": r["doc_link_targets"],
                    "miss_kind": (
                        "missing check: the link_hygiene gate's bare-bracket pattern "
                        "requires a snake_case identity and the documented defect is a "
                        "prose phrase in brackets"
                        if not r["doc_link_targets"]
                        else "missing check: the link target resolves, so no gate reads "
                        "the self-referential use"
                    ),
                }
                for r in missed
            ],
            "false_flags": [
                {
                    "sheet_id": r["sheet_id"],
                    "item_id": r["item_id"],
                    "finding": r["error_lint"],
                    "label_reason": r["label_reason"],
                }
                for r in false_flags
            ],
        }

        payload["cap_sites"] = {
            "cap_function": (
                "imas_codex/standard_names/review/pipeline.py:1963 "
                "_cap_docs_scores_for_lint_errors"
            ),
            "cap_selector": (
                "imas_codex/standard_names/review/pipeline.py:1817 "
                "_extract_error_lint_ids -- reads lint_findings with severity == "
                "'error' only"
            ),
            "cap_call_site": (
                "imas_codex/standard_names/review/pipeline.py:2184 (only when "
                "target == 'docs')"
            ),
            "incumbent_docs_path": (
                "imas_codex/standard_names/workers.py:11150 process_review_docs_batch "
                "-> prepare_docs_review_request (workers.py:10712). No audit report and "
                "no _cap_docs_scores_for_lint_errors reference anywhere in workers.py, "
                "so this path applies no error-severity lint cap. It applies its own "
                "clamp at workers.py:11242 from the conformance/banned-prose/"
                "contradiction notes."
            ),
            "jev_docs_audit": (
                "imas_codex/standard_names/review/cascade.py:519 _docs_judgment -> "
                "jev_review.build_docs_audit_report (jev_review.py:623) -> "
                "run_link_integrity with known_ids = {item} union siblings "
                "(cascade.py:535)."
            ),
            "link_integrity_reads": (
                "imas_codex/standard_names/review/audits.py:406 run_link_integrity "
                "reads name['links'], the structured link list, never the "
                "documentation text."
            ),
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
