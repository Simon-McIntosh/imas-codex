"""Live run of the rebuilt judge stage over the four boundary fixtures.

Reads the four flux-surface-averaged boundary entries from the live graph and
drives the rebuilt judge stage over them with the live decisions seat. Each
judged entry is one categorising Choice, so four names x two concepts is eight
calls. Records every question, its options, the answer with its probabilities,
the verdict the answer maps to, and the booked cost to
``shs-judge-calibration-decisions.json`` beside this script.

Run from the worktree with the project environment:

    UV_PROJECT_ENVIRONMENT=/home/ITER/mcintos/Code/imas-codex/.venv \
      PYTHONPATH="$PWD" uv run --no-sync python \
      docs/evidence/fragments/sn-house-style/shs-judge-calibration-run.py
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from imas_codex.graph.client import GraphClient
from imas_codex.ids.mapping import PipelineCost
from imas_codex.standard_names.review.judge_stage import judge_entries

FIXTURE_IDS = [
    "flux_surface_averaged_carbon_density_at_plasma_boundary",
    "flux_surface_averaged_argon_density_at_plasma_boundary",
    "flux_surface_averaged_beryllium_density_at_plasma_boundary",
    "flux_surface_averaged_electron_density_at_plasma_boundary",
]


async def main() -> None:
    with GraphClient() as gc:
        rows = gc.query(
            "MATCH (sn:StandardName) WHERE sn.id IN $ids "
            "RETURN sn.id AS id, sn.description AS description, "
            "sn.documentation AS documentation",
            ids=FIXTURE_IDS,
        )
    by_id = {row["id"]: row for row in rows}
    names = [by_id[i] for i in FIXTURE_IDS if i in by_id]

    cost = PipelineCost()
    records = await judge_entries(names, cost=cost)

    out = {
        "fixtures": [row["id"] for row in names],
        "calls": len(records),
        "booked_cost_usd": cost.total_usd,
        "decisions": records,
    }
    path = Path(__file__).with_name("shs-judge-calibration-decisions.json")
    path.write_text(json.dumps(out, indent=2) + "\n")

    for rec in records:
        answer = rec["answer"]
        print(
            rec["name_id"],
            "|",
            rec["concept_token"],
            "|",
            answer.get("choice"),
            "| verdict:",
            rec["verdict"],
            "| p:",
            {k: round(v, 2) for k, v in (answer.get("probabilities") or {}).items()},
        )
    print("calls", len(records), "booked USD", round(cost.total_usd, 6))


if __name__ == "__main__":
    asyncio.run(main())
