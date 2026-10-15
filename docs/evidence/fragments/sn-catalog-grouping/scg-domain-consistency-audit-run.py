"""Run the live-name domain audit once and record its census.

Read-only. Opens the live graph, scores every live standard name against the
one derivation in ``domain_ranking`` (see ``domain_audit``), and writes the
per-check counts, the names with the most source domains, and the twenty
lead-approved sheet names beside their current, derived and approved domains.

The 2026-10-13 census recorded beside this audit is embedded so each check's
count prints beside the figure it supersedes. Several census rows measure the
same properties by a different route, so a check is mapped to the nearest
census row and the mapping's looseness is stated in the output rather than
implied by adjacency.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SHEET = HERE / "scg-domain-label-sheet.json"
RESULTS = HERE / "scg-domain-consistency-audit-results.json"

CENSUS_DATE = "2026-10-13"

# The previously recorded census over the 2,675 live names of 2026-10-13, keyed
# by its own wording. Each of this audit's checks is reported beside the census
# row that measures the same property.
CENSUS_2026_10_13 = {
    "live_names": 2675,
    "physics_domain_unset": 212,
    "no_physics_domain_edge": 984,
    "edges_differ_from_source_domains": 777,
    "physics_domain_not_among_edges": 29,
    "physics_domain_not_in_source_domains": 26,
    "domain_none_of_dd_sources": 560,
    "source_domains_differ_from_dd_sources": 1147,
}

CHECK_TO_CENSUS = {
    "physics_domain_unset": ("physics_domain_unset", "exact"),
    "physics_domain_not_derived": (
        "domain_none_of_dd_sources",
        "approximate — the census splits this disagreement over three rows",
    ),
    "source_domains_not_expected": (
        "source_domains_differ_from_dd_sources",
        "near-exact — the census set is the name's DD sources' domains",
    ),
    "edges_not_source_domains": (
        "edges_differ_from_source_domains",
        "exact",
    ),
}


def _state_payload(state) -> dict:
    return {
        "sn_id": state.sn_id,
        "name_stage": state.name_stage,
        "physics_domain": state.physics_domain,
        "derived_domain": state.derived_domain,
        "derived_domain_error": state.derived_domain_error,
        "source_domains": sorted(state.source_domains),
        "expected": sorted(state.expected),
        "edge_domains": sorted(state.edge_domains),
        "dd_source_count": state.dd_source_count,
        "is_derived_parent": state.is_derived_parent,
    }


def main() -> int:
    sys.path.insert(0, str(ROOT))
    from imas_codex.graph.client import GraphClient
    from imas_codex.standard_names.domain_audit import audit_domains

    sheet = json.loads(SHEET.read_text())

    client = GraphClient()
    try:
        result = audit_domains(client)
    finally:
        client.close()

    counts = result.counts()
    checks = {}
    for key, check in result.checks.items():
        census_key, fidelity = CHECK_TO_CENSUS[key]
        checks[key] = {
            "description": check.description,
            "count": check.count,
            "names": list(check.names),
            "census_2026_10_13_row": census_key,
            "census_2026_10_13_count": CENSUS_2026_10_13[census_key],
            "comparison_fidelity": fidelity,
        }

    sample = [_state_payload(state) for state in result.largest_source_sets(10)]

    approved = []
    states_by_id = {state.sn_id: state for state in result.states}
    for row in sheet["reference"]:
        state = states_by_id.get(row["name"])
        approved.append(
            {
                "name": row["name"],
                "approved_domain": row["label"],
                "live": state is not None,
                "current_physics_domain": state.physics_domain if state else None,
                "derived_domain": state.derived_domain if state else None,
                "matches_approved": bool(
                    state and state.physics_domain == row["label"]
                ),
                "derivation_matches_approved": bool(
                    state and state.derived_domain == row["label"]
                ),
                "source_domains": sorted(state.source_domains) if state else None,
                "expected": sorted(state.expected) if state else None,
            }
        )

    payload = {
        "audit": "live-name domain consistency",
        "read_only": True,
        "census_date": CENSUS_DATE,
        "live_names": result.live_names,
        "census_2026_10_13": CENSUS_2026_10_13,
        "checks": checks,
        "counts": counts,
        "largest_source_sets": sample,
        "lead_approved_names": approved,
    }
    RESULTS.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")

    print(
        f"live names: {result.live_names} (census {CENSUS_DATE}: {CENSUS_2026_10_13['live_names']})"
    )
    for key, check in checks.items():
        print(
            f"{key}: {check['count']}"
            f"  [census {check['census_2026_10_13_row']}={check['census_2026_10_13_count']},"
            f" {check['comparison_fidelity']}]"
        )
    print("largest source sets:")
    for state in sample:
        print(f"  {state['sn_id']}: expected={state['expected']}")
    matched = sum(
        1 for row in approved if row["current_physics_domain"] == row["approved_domain"]
    )
    derived_match = sum(1 for row in approved if row["derivation_matches_approved"])
    print(
        f"lead-approved names: {matched}/{len(approved)} carry the approved domain; "
        f"{derived_match}/{len(approved)} have the derivation agree with the approval"
    )
    for row in approved:
        print(
            f"  {row['name']}: current={row['current_physics_domain']} "
            f"derived={row['derived_domain']} approved={row['approved_domain']} "
            f"live={row['live']}"
        )
    print(f"wrote {RESULTS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
