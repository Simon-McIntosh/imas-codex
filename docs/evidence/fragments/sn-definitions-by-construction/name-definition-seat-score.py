"""Score the name-versus-definition review seat on the two-reader labels.

The scored set is every name both readers labelled alike: the both-flag set and
the both-pass set. Disputed names are excluded because neither reading settles
them. The prompt was written from the qualifier classes before this file was
run, and was not changed after seeing the result.
"""

from __future__ import annotations

import asyncio
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from imas_codex.standard_names.review.name_definition_seat import (  # noqa: E402
    judge_name_definitions,
    review_model,
)

REPO = Path(__file__).resolve().parents[4]
STANDALONE_REPO = Path("/home/ITER/mcintos/Code/imas-codex")
CATALOG = Path("/home/ITER/mcintos/Code/imas-standard-names-catalog/standard_names")
OUT = Path(__file__).with_name("name-definition-seat-results.json")

RECALL_FLOOR = 0.80
FALSE_FLAG_CEILING = 0.10


def two_reader_labels() -> dict:
    here = REPO / "docs/evidence/fragments/sn-name-definition-second-reading/two-reader-labels.json"
    path = here if here.exists() else (
        STANDALONE_REPO
        / "docs/evidence/fragments/sn-name-definition-second-reading/two-reader-labels.json"
    )
    return json.loads(path.read_text()) | {"path": str(path)}


def published_names() -> dict[str, dict]:
    names: dict[str, dict] = {}
    for path in sorted(CATALOG.glob("*.y*ml")):
        import yaml

        payload = yaml.safe_load(path.read_text()) or []
        for entry in payload if isinstance(payload, list) else []:
            names[entry.get("name") or entry.get("id")] = entry | {"file": path.name}
    return names


def scored_rows(labels: dict) -> tuple[list[dict], list[dict]]:
    both_flag = [row for row in labels["rows"] if row["first"] != "agrees" and row["second"] != "agrees"]
    both_pass = [row for row in labels["rows"] if row["first"] == "agrees" and row["second"] == "agrees"]
    return both_flag, both_pass


def main() -> None:
    labels = two_reader_labels()
    catalog = published_names()
    both_flag, both_pass = scored_rows(labels)
    rows = [{"row": row, "label": "flag"} for row in both_flag] + [
        {"row": row, "label": "pass"} for row in both_pass
    ]
    items = []
    missing = []
    for entry in rows:
        name = entry["row"]["name"]
        record = catalog.get(name)
        if record is None:
            missing.append(name)
            continue
        items.append(
            {
                "id": name,
                "description": record.get("description") or "",
                "documentation": record.get("documentation") or "",
                "unit": record.get("unit") or "",
            }
        )

    partial = OUT.with_suffix(".partial.jsonl")
    recorded: dict[str, dict] = {}
    if partial.exists():
        for line in partial.read_text().splitlines():
            if line.strip():
                verdict = json.loads(line)
                recorded[verdict["name"]] = verdict
    remaining = [item for item in items if item["id"] not in recorded]
    print(f"recorded {len(recorded)}, remaining {len(remaining)}", flush=True)
    verdicts, cost = asyncio.run(
        judge_name_definitions(
            remaining,
            concurrency=8,
            progress_path=None if not remaining else partial,
        )
    )
    by_name = dict(recorded) | {verdict["name"]: verdict for verdict in verdicts}
    label_of = {entry["row"]["name"]: entry["label"] for entry in rows}

    flagged = [name for name, verdict in by_name.items() if verdict["verdict"] == "flag"]
    missed = [name for name in label_of if label_of[name] == "flag" and name not in flagged]
    false_flags = [
        name for name in label_of if label_of[name] == "pass" and name in flagged
    ]
    unanswered = [
        name for name, verdict in by_name.items() if verdict["verdict"] == "unanswered"
    ]

    flagged_total = sum(1 for label in label_of.values() if label == "flag")
    pass_total = sum(1 for label in label_of.values() if label == "pass")
    recall = (flagged_total - len(missed)) / flagged_total if flagged_total else 0.0
    false_flag_rate = len(false_flags) / pass_total if pass_total else 0.0

    by_qualifier = collections.Counter(
        verdict["qualifier"] for verdict in verdicts if verdict["verdict"] == "flag"
    )

    result = {
        "schema": "name-definition-seat",
        "labels": {
            "path": labels["path"],
            "produced_by": labels["note"],
            "both_flag": flagged_total,
            "both_pass": pass_total,
            "disputed_excluded": len(labels["rows"]) - flagged_total - pass_total,
        },
        "model": review_model(),
        "scored": {
            "items": len(items),
            "missing_from_catalog": missing,
            "unanswered": unanswered,
        },
        "arm": {
            "recall_floor": RECALL_FLOOR,
            "false_flag_ceiling": FALSE_FLAG_CEILING,
            "flagged_total": flagged_total,
            "flagged_found": flagged_total - len(missed),
            "recall": round(recall, 3),
            "pass_total": pass_total,
            "pass_flagged": len(false_flags),
            "false_flag_rate": round(false_flag_rate, 3),
            "met": recall >= RECALL_FLOOR and false_flag_rate <= FALSE_FLAG_CEILING,
        },
        "missed": missed,
        "false_flags": false_flags,
        "by_qualifier": dict(by_qualifier),
        "verdicts": by_name,
        "cost_usd": round(cost, 4),
    }
    OUT.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    print(json.dumps({k: result[k] for k in ("scored", "arm", "by_qualifier", "cost_usd")}, indent=1))
    print("missed:", len(missed), missed[:12])
    print("false flags:", len(false_flags), false_flags[:12])


if __name__ == "__main__":
    main()
