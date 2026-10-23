#!/usr/bin/env python3
"""Measure the link-hygiene gate against the labelled documentation rows.

The gate under test is the deterministic documentation gate vector in
``imas_codex.standard_names.docs_gates``, whose ``link_hygiene`` gate now
reports a bracketed prose phrase standing where a name link belongs and a
``name:`` link that points back at the name it documents. Nothing here writes
to the graph and nothing calls a model.

Three arms are measured over the 77 labelled documentation rows:

* the changed gate alone, over every documentation gate in the vector;
* the two link rules this change adds, by their own reasons;
* the cap's selector, the union of the changed gate and the error-severity
  lint findings the incumbent already produced for each row (read from the
  replay record, so the union is measured against an unmodified instrument).

The gate is then run over the live catalog's accepted documentation, bounded
and read-only, as the false-positive estimate.
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

FRAGMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from imas_codex.standard_names.docs_gates import (  # noqa: E402
    DocumentationGateOutcome,
    score_documentation,
)

SHEET_PATH = FRAGMENT_DIR / "srtj-docs-disagreement-sheet.json"
LABELS_PATH = FRAGMENT_DIR / "srtj-docs-disagreement-labels.json"
KEY_PATH = FRAGMENT_DIR / "srtj-docs-disagreement-key.json"
REPLAY_PATH = FRAGMENT_DIR / "srtj-docs-lint-replay.json"
OUTPUT_PATH = FRAGMENT_DIR / "srtj-docs-prose-link-check.json"

PROSE_REASON = "bracketed phrase names no standard name:"
SELF_LINK_REASON = "targets the documented name itself:"
LIVE_DOCUMENT_LIMIT = 4000


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _new_rule_reason(reason: str) -> str | None:
    """Classify a hygiene failure as one this change adds, or as pre-existing."""

    if reason.startswith(PROSE_REASON):
        return "prose_bracket"
    if SELF_LINK_REASON in reason:
        return "self_link"
    return None


def _measure_rows() -> dict:
    sheet = {row["sheet_id"]: row for row in _load(SHEET_PATH)["rows"]}
    labels = {row["sheet_id"]: row["label"] for row in _load(LABELS_PATH)["labels"]}
    key = {row["sheet_id"]: row for row in _load(KEY_PATH)["key"]}
    replay = {row["sheet_id"]: row for row in _load(REPLAY_PATH)["rows"]}

    rows: list[dict] = []
    for sheet_id, row in sorted(sheet.items()):
        documentation = row["documentation"]
        name = row["name"]
        score = score_documentation(documentation, documenting_name=name)
        hygiene = score.gate_vector["link_hygiene"]
        failed_gates = [
            gate
            for gate, result in score.gate_vector.items()
            if result.outcome is DocumentationGateOutcome.FAIL
        ]
        new_rule = (
            _new_rule_reason(hygiene.reason)
            if hygiene.outcome is DocumentationGateOutcome.FAIL
            else None
        )
        incumbent_error = bool(replay[sheet_id]["error_lint"])
        rows.append(
            {
                "sheet_id": sheet_id,
                "name": name,
                "class": key[sheet_id]["class"],
                "acceptable": bool(labels[sheet_id]),
                "changed_gate_fails": bool(failed_gates),
                "changed_gate_failed_gates": failed_gates,
                "new_rule": new_rule,
                "hygiene_reason": hygiene.reason,
                "incumbent_error": incumbent_error,
                "cap_refines": bool(failed_gates) or incumbent_error,
            }
        )

    def counted(predicate) -> dict[str, int]:
        counter: Counter = Counter(
            f"{row['class']} / "
            f"{'acceptable' if row['acceptable'] else 'not acceptable'}"
            for row in rows
            if predicate(row)
        )
        return dict(sorted(counter.items()))

    classes = sorted({(row["class"], row["acceptable"]) for row in rows})
    per_class = [
        {
            "class": cls,
            "label": "acceptable" if acceptable else "not acceptable",
            "rows": sum(
                1
                for row in rows
                if (row["class"], row["acceptable"]) == (cls, acceptable)
            ),
            "changed_gate_fails": sum(
                1
                for row in rows
                if (row["class"], row["acceptable"]) == (cls, acceptable)
                and row["changed_gate_fails"]
            ),
            "new_rule_fails": sum(
                1
                for row in rows
                if (row["class"], row["acceptable"]) == (cls, acceptable)
                and row["new_rule"]
            ),
            "cap_refines": sum(
                1
                for row in rows
                if (row["class"], row["acceptable"]) == (cls, acceptable)
                and row["cap_refines"]
            ),
        }
        for cls, acceptable in classes
    ]

    disagreement_defects = [
        row for row in rows if row["class"] == "disagreement" and not row["acceptable"]
    ]
    disagreement_acceptable = [
        row for row in rows if row["class"] == "disagreement" and row["acceptable"]
    ]
    missed_ids = [entry["sheet_id"] for entry in _load(REPLAY_PATH)["bar"]["missed"]]
    missed_rows = [row for row in rows if row["sheet_id"] in set(missed_ids)]

    missed_new_rule = [
        row["sheet_id"] for row in missed_rows if row["new_rule"] is not None
    ]
    defect_cap_refined = [
        row["sheet_id"] for row in disagreement_defects if row["cap_refines"]
    ]
    acceptable_cap_refined = [
        row["sheet_id"] for row in disagreement_acceptable if row["cap_refines"]
    ]
    acceptable_changed_gate_fails = [
        row["sheet_id"] for row in disagreement_acceptable if row["changed_gate_fails"]
    ]

    bar = {
        "missed_defects": len(missed_ids),
        "missed_defects_failed_by_new_rule": len(missed_new_rule),
        "missed_defects_unrepaired": [
            i for i in missed_ids if i not in missed_new_rule
        ],
        "defect_rows": len(disagreement_defects),
        "defect_rows_cap_refines": len(defect_cap_refined),
        "defect_rows_unrefined": [
            row["sheet_id"] for row in disagreement_defects if not row["cap_refines"]
        ],
        "acceptable_rows": len(disagreement_acceptable),
        "acceptable_rows_changed_gate_fails": len(acceptable_changed_gate_fails),
        "acceptable_rows_changed_gate_fail_ids": acceptable_changed_gate_fails,
        "acceptable_rows_cap_refines": len(acceptable_cap_refined),
        "acceptable_rows_cap_refine_ids": acceptable_cap_refined,
        "meets_bar": (
            len(missed_new_rule) == len(missed_ids)
            and len(defect_cap_refined) == len(disagreement_defects)
            and not acceptable_changed_gate_fails
        ),
    }

    new_rule_failures = [row for row in rows if row["new_rule"]]
    return {
        "rows": rows,
        "per_class": per_class,
        "bar": bar,
        "fail_counts": {
            "changed_gate_any_class": counted(lambda row: row["changed_gate_fails"]),
            "new_rule_any_class": counted(lambda row: row["new_rule"]),
            "cap_refines_any_class": counted(lambda row: row["cap_refines"]),
        },
        "new_rule_failure_sample": [
            {
                "sheet_id": row["sheet_id"],
                "name": row["name"],
                "rule": row["new_rule"],
                "reason": row["hygiene_reason"],
            }
            for row in new_rule_failures[:6]
        ],
        "reproductions": {
            row["sheet_id"]: row["hygiene_reason"]
            for row in rows
            if row["sheet_id"] in {"S0027", "S0060", "S0071", "S0074"}
        },
    }


def _positive_controls() -> dict:
    """Show the instrument sees a known instance of each construct before any absence."""

    from imas_codex.standard_names.doc_links import find_prose_bracket_phrases

    prose = find_prose_bracket_phrases("differs from the [ion density at a limiter].")
    notation = find_prose_bracket_phrases("in [m^-3] over [0, 1] and $[a b]$")
    self_link = score_documentation(
        "See [lithium density at a limiter](name:lithium_density_at_limiter).",
        documenting_name="lithium_density_at_limiter",
    ).gate_vector["link_hygiene"]
    prose_gate = score_documentation(
        "differs from the [ion density at a limiter]."
    ).gate_vector["link_hygiene"]
    return {
        "prose_phrase_seen": {
            "purpose": "the scanner must return a prose bracket it is given",
            "input": "differs from the [ion density at a limiter].",
            "found": list(prose),
            "passed": prose == ("ion density at a limiter",),
        },
        "notation_brackets_pass": {
            "purpose": "units, a numeric interval and mathematics must raise no phrase",
            "input": "in [m^-3] over [0, 1] and $[a b]$",
            "found": list(notation),
            "passed": notation == (),
        },
        "prose_bracket_gate_fails": {
            "purpose": "the hygiene gate must fail a document holding a prose bracket",
            "outcome": prose_gate.outcome.name,
            "reason": prose_gate.reason,
            "passed": prose_gate.outcome is DocumentationGateOutcome.FAIL
            and prose_gate.reason.startswith(PROSE_REASON),
        },
        "self_link_gate_fails": {
            "purpose": "the hygiene gate must fail a link to the documented name",
            "outcome": self_link.outcome.name,
            "reason": self_link.reason,
            "passed": self_link.outcome is DocumentationGateOutcome.FAIL
            and SELF_LINK_REASON in self_link.reason,
        },
    }


def _measure_live_catalog() -> dict:
    """Run the gate over the live catalog's accepted documentation, bounded."""

    started = time.monotonic()
    try:
        from imas_codex.graph.client import GraphClient

        with GraphClient() as gc:
            records = gc.query(
                "MATCH (sn:StandardName) "
                "WHERE sn.name_stage IN ['accepted', 'approved'] "
                "AND sn.documentation IS NOT NULL "
                "RETURN sn.id AS id, sn.documentation AS docs "
                "ORDER BY sn.id LIMIT $limit",
                limit=LIVE_DOCUMENT_LIMIT,
            )
    except Exception as exc:  # noqa: BLE001 - reported, never mistaken for a clean read
        return {
            "available": False,
            "error": f"{type(exc).__name__}: {exc}",
            "elapsed_s": round(time.monotonic() - started, 1),
        }

    by_rule: Counter = Counter()
    also_failing = 0
    examples: list[dict] = []
    for record in records:
        documentation = (record.get("docs") or "").strip()
        if not documentation:
            continue
        score = score_documentation(
            documentation, documenting_name=record.get("id") or None
        )
        hygiene = score.gate_vector["link_hygiene"]
        if hygiene.outcome is not DocumentationGateOutcome.FAIL:
            continue
        rule = _new_rule_reason(hygiene.reason)
        if rule is None:
            continue
        by_rule[rule] += 1
        other = [
            gate
            for gate, result in score.gate_vector.items()
            if gate != "link_hygiene"
            and result.outcome is DocumentationGateOutcome.FAIL
        ]
        also_failing += bool(other)
        if len(examples) < 10:
            examples.append(
                {
                    "id": record.get("id"),
                    "rule": rule,
                    "reason": hygiene.reason,
                    "other_failed_gates": other,
                }
            )
    return {
        "available": True,
        "source": "live graph: accepted StandardName documentation",
        "documents_read": len(records),
        "limit": LIVE_DOCUMENT_LIMIT,
        "newly_failing_documents": sum(by_rule.values()),
        "newly_failing_by_rule": dict(by_rule),
        "newly_failing_also_failing_another_gate": also_failing,
        "examples": examples,
        "labelled": False,
        "elapsed_s": round(time.monotonic() - started, 1),
    }


def main() -> int:
    payload = {
        "schema": "srtj-docs-prose-link-check",
        "created_at": datetime.now(UTC).isoformat(),
        "code_revision": __import__("subprocess")
        .run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        .stdout.strip(),
        "gate": "imas_codex/standard_names/docs_gates.py::link_hygiene",
        "module_path": str(
            Path(sys.modules[score_documentation.__module__].__file__).resolve()
        ),
        "working_directory": str(Path.cwd()),
        "rows_measurement": _measure_rows(),
        "positive_controls": _positive_controls(),
        "live_catalog": _measure_live_catalog(),
    }
    payload["status"] = (
        "accepted"
        if payload["rows_measurement"]["bar"]["meets_bar"]
        and all(control["passed"] for control in payload["positive_controls"].values())
        else "rejected"
    )
    OUTPUT_PATH.write_text(
        json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload["rows_measurement"]["bar"], indent=2))
    for entry in payload["rows_measurement"]["per_class"]:
        print(entry)
    print("live_catalog:", json.dumps(payload["live_catalog"])[:1200])
    print("status:", payload["status"], "wrote", OUTPUT_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
