"""Run the name-versus-definition seat over every published name.

Names are read from the live graph read-only; the seat is the docs review primary
reviewer. Verdicts are appended to a JSONL file as each batch lands, so a run cut
short still yields counts.
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

OUT = Path(__file__).with_name("name-definition-seat-catalog.json")
PARTIAL = Path(__file__).with_name("name-definition-seat-catalog.jsonl")

PUBLISHED_QUERY = """
MATCH (n:StandardName)
WHERE n.lifecycle IS NULL OR n.lifecycle = 'published'
RETURN n.id AS id,
       coalesce(n.description, '') AS description,
       coalesce(n.documentation, '') AS documentation,
       coalesce(n.unit, '') AS unit
ORDER BY id
LIMIT 4000
"""


def published_names() -> list[dict]:
    from imas_codex.graph.client import GraphClient

    with GraphClient() as client:
        return [dict(row) for row in client.query(PUBLISHED_QUERY)]


def disposition(entry: dict, verdict: dict, lexical: list[str]) -> str:
    qualifier = verdict.get("qualifier")
    if qualifier == "family-of-quantities":
        return "hold-from-the-cut"
    if qualifier == "restriction-to-one-object":
        return "rename"
    if lexical:
        return "rewrite-the-definition"
    return "rewrite-the-definition"


def main() -> None:
    names = published_names()
    print(f"published names read from the graph: {len(names)}", flush=True)
    verdicts, cost = asyncio.run(judge_name_definitions(names))
    by_name = {verdict["name"]: verdict for verdict in verdicts}
    PARTIAL.write_text(
        "".join(json.dumps(verdict, sort_keys=True) + "\n" for verdict in verdicts)
    )

    flagged = [name for name, verdict in by_name.items() if verdict["verdict"] == "flag"]
    by_qualifier = collections.Counter(
        by_name[name]["qualifier"] for name in flagged
    )
    unanswered = [
        name for name, verdict in by_name.items() if verdict["verdict"] == "unanswered"
    ]
    findings = []
    for name in flagged:
        record = next((entry for entry in names if entry["id"] == name), {})
        lexical = name_definition_agreement_check(record)
        findings.append(
            {
                "name": name,
                "unit": record.get("unit") or "",
                "qualifier": by_name[name]["qualifier"],
                "reasoning": by_name[name]["reasoning"],
                "lexical_families": [f.split(":")[1].strip() for f in lexical],
                "disposition": disposition(record, by_name[name], lexical),
            }
        )

    result = {
        "schema": "name-definition-seat-catalog",
        "model": review_model(),
        "published_names": len(names),
        "flagged": len(flagged),
        "by_qualifier": dict(sorted(by_qualifier.items())),
        "unanswered": unanswered,
        "dispositions": dict(
            collections.Counter(finding["disposition"] for finding in findings)
        ),
        "findings": findings,
        "cost_usd": round(cost, 4),
    }
    OUT.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    print(json.dumps({k: result[k] for k in ("published_names", "flagged", "by_qualifier", "dispositions", "cost_usd")}, indent=1))
    print("unanswered:", len(unanswered))


if __name__ == "__main__":
    main()
