"""Run the name-versus-definition review seat over every published name.

Names, definitions and derived-parent state are read from the live graph
read-only. A name is published when it carries the catalog commit it was exported
in; the graph's ``status`` field marks drafts and superseded names, not the cut. Verdicts are appended to a JSONL file on GPFS one batch at a time, so
a run cut short keeps everything it paid for and a rerun judges only the names
not yet recorded. A per-name disposition is proposed; none is applied.
"""

from __future__ import annotations

import asyncio
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from imas_codex.standard_names.audits import (  # noqa: E402
    name_definition_agreement_check,
)
from imas_codex.standard_names.review.name_definition_seat import (  # noqa: E402
    judge_name_definitions,
    review_model,
)

HERE = Path(__file__).with_name("name-definition-seat-catalog")
OUT = HERE.with_suffix(".json")
PARTIAL = HERE.with_suffix(".jsonl")

PUBLISHED_QUERY = """
MATCH (n:StandardName)
WHERE n.catalog_commit_sha IS NOT NULL
OPTIONAL MATCH (child:StandardName)-[:HAS_PARENT]->(n)
RETURN n.id AS id,
       coalesce(n.description, '') AS description,
       coalesce(n.documentation, '') AS documentation,
       coalesce(n.unit, '') AS unit,
       count(child) > 0 AS derived,
       coalesce(collect(DISTINCT child.id), []) AS children
ORDER BY id
LIMIT 4000
"""


def published_names() -> list[dict]:
    from imas_codex.graph.client import GraphClient

    with GraphClient() as client:
        rows = [dict(row) for row in client.query(PUBLISHED_QUERY)]
    for row in rows:
        row["children"] = [child for child in row["children"] if child]
    return rows


def proposed_disposition(verdict: dict, lexical: list[str]) -> tuple[str, str]:
    """One proposed disposition, and the spelling it rests on where that applies."""

    qualifier = verdict.get("qualifier")
    if qualifier == "family-of-quantities":
        return "hold from the cut", ""
    if qualifier == "restriction-to-one-object":
        return "rename", verdict.get("proposed_name") or ""
    if qualifier is None and not lexical:
        return "stands", ""
    return "rewrite the definition", ""


def main() -> None:
    names = published_names()
    print(f"published names read from the graph: {len(names)}", flush=True)

    recorded: dict[str, dict] = {}
    if PARTIAL.exists():
        for line in PARTIAL.read_text().splitlines():
            if line.strip():
                verdict = json.loads(line)
                recorded[verdict["name"]] = verdict
    published_ids = {entry["id"] for entry in names}
    recorded = {
        name: verdict for name, verdict in recorded.items() if name in published_ids
    }
    remaining = [entry for entry in names if entry["id"] not in recorded]
    print(f"recorded {len(recorded)}, remaining {len(remaining)}", flush=True)

    cost = 0.0
    if remaining:
        verdicts, cost = asyncio.run(
            judge_name_definitions(
                [
                    {
                        "id": entry["id"],
                        "description": entry["description"],
                        "documentation": entry["documentation"],
                        "unit": entry["unit"],
                        "derived_children": entry["children"],
                    }
                    for entry in remaining
                ],
                concurrency=12,
                progress_path=PARTIAL,
            )
        )
        recorded.update({verdict["name"]: verdict for verdict in verdicts})

    by_name = recorded
    flagged = [
        name for name, verdict in by_name.items() if verdict["verdict"] == "flag"
    ]
    unanswered = [
        name for name, verdict in by_name.items() if verdict["verdict"] == "unanswered"
    ]
    by_qualifier = collections.Counter(by_name[name]["qualifier"] for name in flagged)

    findings = []
    for name in flagged:
        record = next((entry for entry in names if entry["id"] == name), {})
        lexical = name_definition_agreement_check(record) if record else []
        disposition, spelling = proposed_disposition(by_name[name], lexical)
        findings.append(
            {
                "name": name,
                "unit": record.get("unit") or "",
                "qualifier": by_name[name]["qualifier"],
                "definition_phrase": by_name[name]["reasoning"],
                "proposed_disposition": disposition,
                "proposed_spelling": spelling,
                "lexical_families": [f.split(":")[1].strip() for f in lexical],
            }
        )
    findings.sort(key=lambda finding: (finding["qualifier"], finding["name"]))

    parents = [entry for entry in names if entry["derived"]]
    parent_rows = []
    for entry in parents:
        verdict = by_name.get(entry["id"])
        if verdict is None:
            continue
        states_one_quantity = not (
            verdict["verdict"] == "flag"
            and verdict["qualifier"] == "family-of-quantities"
        )
        parent_rows.append(
            {
                "name": entry["id"],
                "verdict": verdict["verdict"],
                "qualifier": verdict["qualifier"],
                "states_one_quantity": states_one_quantity,
                "children": len(entry["children"]),
            }
        )

    dispositions = collections.Counter(
        finding["proposed_disposition"] for finding in findings
    )
    judged = len(
        [
            name
            for name, verdict in by_name.items()
            if verdict["verdict"] != "unanswered"
        ]
    )
    result = {
        "schema": "name-definition-seat-catalog",
        "model": review_model(),
        "published_names": len(names),
        "judged": judged,
        "unanswered": unanswered,
        "flagged": len(flagged),
        "by_qualifier": dict(sorted(by_qualifier.items())),
        "proposed_dispositions": dict(sorted(dispositions.items())),
        "findings": findings,
        "abstract_parents": {
            "published": len(parents),
            "judged": len(parent_rows),
            "states_one_quantity": sum(
                1 for row in parent_rows if row["states_one_quantity"]
            ),
            "does_not": sum(1 for row in parent_rows if not row["states_one_quantity"]),
            "rule": (
                "a parent's definition states one quantity unless the seat flags it "
                "as family-of-quantities"
            ),
        },
        "cost_usd": round(cost, 4),
        "cost_per_name_usd": round(cost / judged, 5) if judged else None,
    }
    OUT.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "published_names",
                    "judged",
                    "flagged",
                    "by_qualifier",
                    "proposed_dispositions",
                    "abstract_parents",
                    "cost_usd",
                    "cost_per_name_usd",
                )
            },
            indent=1,
        ),
        flush=True,
    )
    print("unanswered:", len(unanswered), flush=True)


if __name__ == "__main__":
    main()
