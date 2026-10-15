"""Dry-run spend estimates for the completion apply and the family measure.

Read-only over the live graph. For the boundary-averaged density family and one
non-WEST family per pattern kind (species, component, locus) it plans the
anchor-gated apply, previews the recommended-name refusals the apply would
raise, reads the current lint findings by class and each member's documentation
score, and measures the per-call documentation cost from the recent ledger.
Nothing is written to the graph; only ``*-results.json`` is written, beside this
script.

Run from the worktree with the shared environment:

    UV_PROJECT_ENVIRONMENT=/home/ITER/mcintos/Code/imas-codex/.venv \
    PYTHONPATH="$PWD" uv run --no-sync python \
      docs/evidence/fragments/sn-harmonize-command/shc-apply-scope-estimates-run.py
"""

from __future__ import annotations

import json
from pathlib import Path

from imas_codex.graph.client import GraphClient
from imas_codex.standard_names.defaults import DEFAULT_MIN_SCORE
from imas_codex.standard_names.harmonize import (
    measure_doc_call_cost,
    plan_family_applies,
    preview_recommended_refusals,
    report_family_rows,
)

# One family per pattern kind. A pattern family is a cohort of live names that
# share their invariant grammar segments and differ only along the variation
# axes; the kind names the axis that distinguishes its members.
SCOPE: tuple[dict[str, str], ...] = (
    {
        "kind": "boundary",
        "seed": "flux_surface_averaged_argon_density_at_plasma_boundary",
        "why": "the 17-member boundary-averaged density family §3 completes on",
    },
    {
        "kind": "species",
        "seed": "electron_number_density",
        "why": "members differ by subject species (electron, ion, neutral …)",
    },
    {
        "kind": "component",
        "seed": "parallel_conductivity",
        "why": "members differ by vector component (parallel, poloidal, radial …)",
    },
    {
        "kind": "locus",
        "seed": "electrostatic_potential",
        "why": "members differ by the locus that places the quantity (wall, midplane …)",
    },
)

_OUT = Path(__file__).with_name("shc-apply-scope-estimates-results.json")


def _lint_by_kind(rows: list[dict], seed: str) -> dict[str, int]:
    for row in rows:
        if seed in row["members"]:
            return {k: int(v) for k, v in row["lint"].items()}
    return {}


def main() -> int:
    families: list[dict] = []
    total_calls = 0
    with GraphClient() as gc:
        estimate = measure_doc_call_cost(gc)
        per_call = estimate.mean_usd if estimate is not None else None
        for item in SCOPE:
            seed = item["seed"]
            plans = plan_family_applies(gc, seeds=[seed], min_size=1)
            if not plans:
                families.append({**item, "error": "family not found"})
                continue
            plan = plans[0]
            refusals = preview_recommended_refusals(gc, plans)[0]
            report_rows = report_family_rows(gc, seeds=[seed], min_size=1)
            # Member documentation scores, read from the same family query the
            # plan uses, so the estimate and the scores share one read.
            fam = next(
                (f for f in report_rows if seed in f["members"]),
                None,
            )
            calls = plan["planned_calls"]
            total_calls += calls
            entry = {
                "kind": item["kind"],
                "seed": seed,
                "why": item["why"],
                "physical_base": plan["physical_base"],
                "invariant": plan["invariant"],
                "members": plan["members"],
                "n": plan["n"],
                "decision": plan["decision"],
                "planned_calls": calls,
                "anchor": plan["anchor"],
                "estimate_usd": (
                    round(calls * per_call, 4) if per_call is not None else None
                ),
                "refusals": dict(sorted(refusals.items())),
                "lint": _lint_by_kind(report_rows, seed),
                "doc_scores": {
                    mid: score for mid, score in sorted(plan["member_scores"].items())
                },
                "docs_below_threshold": sorted(
                    mid
                    for mid, score in plan["member_scores"].items()
                    if score is not None and score < DEFAULT_MIN_SCORE
                ),
                "drift": fam["drift"] if fam else None,
            }
            families.append(entry)

    payload = {
        "scope": [{"kind": i["kind"], "seed": i["seed"], "why": i["why"]} for i in SCOPE],
        "acceptance_threshold": DEFAULT_MIN_SCORE,
        "estimate_basis": (
            None
            if estimate is None
            else {
                "runs": estimate.runs,
                "calls": estimate.calls,
                "mean_usd_per_call": round(estimate.mean_usd, 4),
                "total_usd": round(estimate.total_usd, 4),
            }
        ),
        "total_planned_calls": total_calls,
        "total_estimate_usd": (
            round(total_calls * estimate.mean_usd, 4) if estimate is not None else None
        ),
        "families": families,
        "wrote_graph": False,
    }
    _OUT.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")
    print(f"wrote {_OUT}")
    print(
        f"total calls={total_calls} "
        f"estimate=${payload['total_estimate_usd']} basis={payload['estimate_basis']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())