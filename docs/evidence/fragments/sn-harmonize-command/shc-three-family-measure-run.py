"""Capture read-only family measurements and a score-regression control.

The snapshot command reads one seeded family and updates the adjacent JSON
record. The control uses fixed fixture scores and never opens the graph.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from imas_codex.standard_names.defaults import DEFAULT_MIN_SCORE
from imas_codex.standard_names.harmonize import docs_score_regressions

RESULTS = Path(__file__).with_name("shc-three-family-measure-results.json")
BOUNDARY_SEED = "flux_surface_averaged_argon_density_at_plasma_boundary"


def read_results() -> dict:
    if RESULTS.exists():
        return json.loads(RESULTS.read_text())
    return {"acceptance_threshold": DEFAULT_MIN_SCORE, "families": {}}


def write_results(payload: dict) -> None:
    RESULTS.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def snapshot(seed: str, phase: str) -> None:
    from imas_codex.graph.client import GraphClient
    from imas_codex.standard_names.harmonize import (
        measure_doc_call_cost,
        plan_family_applies,
        preview_recommended_refusals,
        report_family_rows,
    )

    with GraphClient() as graph:
        plans = plan_family_applies(graph, seeds=[seed])
        reports = report_family_rows(graph, seeds=[seed])
        boundary = plan_family_applies(graph, seeds=[BOUNDARY_SEED])
        estimate = measure_doc_call_cost(graph)
        if not plans or not reports or not boundary:
            raise RuntimeError("a named family is absent from the graph")
        plan = next(p for p in plans if seed in p["members"])
        report = next(r for r in reports if seed in r["members"])
        overlap = sorted(set(plan["members"]) & set(boundary[0]["members"]))
        if overlap:
            raise RuntimeError(f"family overlaps the boundary apply: {overlap}")
        refusals = preview_recommended_refusals(graph, [plan])[0]
        states = list(
            graph.query(
                """
                MATCH (member:StandardName)
                WHERE member.id IN $ids
                RETURN member.id AS id,
                       member.name_stage AS name_stage,
                       member.docs_stage AS docs_stage,
                       member.validation_status AS validation_status
                ORDER BY id
                """,
                ids=plan["members"],
            )
            or []
        )
        if {row["id"] for row in states} != set(plan["members"]):
            raise RuntimeError("the status read missed a family member")
        accepted = sum(
            row.get("docs_stage") == "accepted"
            and row.get("validation_status") != "quarantined"
            for row in states
        )
        quarantined = sum(
            row.get("validation_status") == "quarantined" for row in states
        )
        data = {
            "seed": seed,
            "physical_base": plan["physical_base"],
            "invariant": plan["invariant"],
            "members": plan["members"],
            "anchor": plan["anchor"],
            "decision": plan["decision"],
            "planned_calls": plan["planned_calls"],
            "estimate_usd": (
                round(plan["planned_calls"] * estimate.mean_usd, 4)
                if estimate is not None
                else None
            ),
            "estimate_basis": (
                {
                    "runs": estimate.runs,
                    "calls": estimate.calls,
                    "total_usd": round(estimate.total_usd, 4),
                    "mean_usd_per_call": round(estimate.mean_usd, 4),
                }
                if estimate is not None
                else None
            ),
            "refusals": refusals,
            "lint": report["lint"],
            "lint_total": report["lint_total"],
            "documentation_scores": plan["member_scores"],
            "status_counts": {
                "accepted": accepted,
                "quarantined": quarantined,
                "other": len(states) - accepted - quarantined,
            },
            "member_status": {row["id"]: row for row in states},
            "boundary_overlap": overlap,
        }
    payload = read_results()
    payload["families"].setdefault(seed, {})[phase] = data
    write_results(payload)
    print(json.dumps(data, sort_keys=True))


def fixture() -> None:
    name = "fixture_low_scoring_member"
    before = {name: 0.95, "fixture_stable_member": 0.9}
    after = {name: round(DEFAULT_MIN_SCORE - 0.05, 2), "fixture_stable_member": 0.9}
    result = docs_score_regressions(before, after)
    named = [item["id"] for item in result["below_threshold"]]
    if named != [name]:
        raise AssertionError(f"below-threshold member not named: {named}")
    payload = read_results()
    payload["fixture"] = {
        "before": before,
        "after": after,
        "result": result,
        "threshold": DEFAULT_MIN_SCORE,
    }
    write_results(payload)
    print(json.dumps(payload["fixture"], sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    subcommands = parser.add_subparsers(dest="command", required=True)
    snap = subcommands.add_parser("snapshot")
    snap.add_argument("seed")
    snap.add_argument(
        "phase",
        choices=("before", "partial", "after_job", "current_before", "after"),
    )
    subcommands.add_parser("fixture")
    args = parser.parse_args()
    if args.command == "snapshot":
        snapshot(args.seed, args.phase)
    else:
        fixture()


if __name__ == "__main__":
    main()
