"""Live run of the judge stage's boundary criterion over five boundary fixtures.

Four flux-surface-averaged boundary entries are read from the live graph, and a
fifth fixture is authored here: an entry that glosses the plasma boundary as the
last closed flux surface and also calls it the separatrix, the form the lead's
gloss ruling forbids. Each entry is judged for its flux-surface-average equation
form and its boundary gloss, so five names x two concepts is ten calls at most.
Every question, its options, the answer with its probabilities, the verdict the
answer maps to and the booked cost are recorded beside this script.

Run from the worktree with the project environment:

    UV_NO_SYNC=1 UV_PROJECT_ENVIRONMENT=/home/ITER/mcintos/Code/imas-codex/.venv \
      PYTHONPATH="$PWD" uv run python \
      docs/evidence/fragments/sn-house-style/shs-boundary-criterion-run.py
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

# The authored fixture: a parseable boundary id whose text glosses the boundary
# as the last closed flux surface and also equates it with the separatrix.
SYNONYM_FIXTURE = {
    "id": "flux_surface_averaged_neon_density_at_plasma_boundary",
    "description": (
        "Flux-surface-averaged number density of neon ions at the plasma "
        "boundary (separatrix)."
    ),
    "documentation": (
        "The flux-surface-averaged neon density at the plasma boundary is "
        "averaged over the last closed flux surface, the separatrix that bounds "
        "the confined plasma. The flux-surface average is defined by "
        "$$\\langle n_{Ne}\\rangle = \\oint n_{Ne}\\,dS/|\\nabla\\psi| \\,/\\, "
        "\\oint dS/|\\nabla\\psi|$$ evaluated on the separatrix, the outermost "
        "closed magnetic flux surface."
    ),
}


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
    names.append(SYNONYM_FIXTURE)

    cost = PipelineCost()
    records = await judge_entries(names, cost=cost)

    out = {
        "fixtures": [row["id"] for row in names],
        "calls": len(records),
        "booked_cost_usd": cost.total_usd,
        "decisions": records,
    }
    path = Path(__file__).with_name("shs-boundary-criterion-decisions.json")
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