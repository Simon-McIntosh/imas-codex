#!/usr/bin/env python
"""Record the outcome of regenerating two names' documentation after the split.

The split gave each name one quantity: ``gap_at_outboard_midplane`` keeps the
separatrix separation, and
``radial_separatrix_distance_at_outboard_midplane`` holds the three mapped
measurement distances. This record drives the documentation-only worker scope
(``sn run --docs-only``) over the two named standard names, then reads both
names' lifecycle, scores and documentation from the live graph and evaluates
the read-back checks the work item asks for.

The scope is the pipeline's own documentation stage, whose pools claim
``generate_docs`` (name accepted, docs pending), ``review_docs`` (docs drafted)
and ``refine_docs`` (docs reviewed and below threshold). Where the two names are
not in a state those pools claim, that is the finding and is reported rather
than worked around; nothing here writes to the graph.

Modes:
    snapshot   read both names' lifecycle, scores and documentation
    report     assemble the fragment's JSON record beside this runner
"""

from __future__ import annotations

import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RECORD = os.path.join(HERE, "shc-gap-docs-regen.json")

NAMES = [
    "gap_at_outboard_midplane",
    "radial_separatrix_distance_at_outboard_midplane",
]

#: The invoked scope: the documentation-only worker pools over the two names.
RUN_COMMAND = (
    "uv run --no-sync imas-codex sn run --docs-only "
    "--name gap_at_outboard_midplane "
    "--name radial_separatrix_distance_at_outboard_midplane "
    "--cost-limit 0.90 --skip-global-maintenance"
)
DRY_RUN_COMMAND = RUN_COMMAND.replace(" --cost-limit 0.90", " --dry-run")
DRY_RUN_OUTPUT = (
    "Exact-name dry run: 2 existing name(s) eligible; no graph writes performed"
)
RUN_LOG = (
    "/home/ITER/mcintos/.config/reckon/crew/runs/"
    "r-20261020T202608740663-shc-gap-docs-regen/logs/sn-run-docs-only.log"
)
RUN_ID = "90a76a40-f6fc-44df-83f0-d49ba887be0e"

#: What each name means, so a description can be checked against its sources.
MEANING = {
    "gap_at_outboard_midplane": (
        "signed major-radius separation between a mapped measurement position "
        "and the magnetic separatrix at the outboard midplane"
    ),
    "radial_separatrix_distance_at_outboard_midplane": (
        "radial distance from the separatrix (plasma boundary) to a measurement "
        "position at the outboard midplane"
    ),
}

#: A DD field name, an explicit IDS/Cypher path, or a DD option value. The
#: ruling moved the locus text onto physics terms, so these must not appear.
DD_FIELD_RE = re.compile(
    r"\bmidplane_identifier\b|\bdd:[a-z_]+/|\bsummary/[a-z_/]+|"
    r"\b[a-z_]+_dummy\b|\bdr_dz_zero[a-z_]*\b"
)
OPTION_RE = re.compile(r"'(dr_dz_zero[a-z_]*)'|\boption\s+'[^']+'")


def snapshot() -> dict:
    """Read both names' lifecycle, scores and documentation from the graph."""
    from imas_codex.graph.client import GraphClient

    with GraphClient() as gc:
        rows = list(
            gc.query(
                """
                MATCH (sn:StandardName)
                WHERE sn.id IN $ids
                RETURN sn.id AS id, properties(sn) AS props
                ORDER BY sn.id
                """,
                ids=NAMES,
            )
        )
        out = {"names": {}, "uri": gc.uri}
        for row in rows:
            props = dict(row["props"])
            out["names"][row["id"]] = {
                "name_stage": props.get("name_stage"),
                "docs_stage": props.get("docs_stage"),
                "reviewer_score_docs": props.get("reviewer_score_docs"),
                "docs_chain_length": props.get("docs_chain_length"),
                "docs_model": props.get("docs_model"),
                "edit_status": props.get("edit_status"),
                "edit_mode": props.get("edit_mode"),
                "description": props.get("description"),
                "documentation": props.get("documentation"),
                "source_paths": props.get("source_paths"),
                "documentation_present": bool(props.get("documentation")),
            }
        missing = sorted(set(NAMES) - set(out["names"]))
        out["_missing"] = missing
    return out


def read_checks(state: dict) -> dict:
    """Per-name read-back checks the plan asks for."""
    threshold = 0.85
    checks = {}
    for name in NAMES:
        entry = state["names"].get(name, {})
        text = " ".join(
            str(part) for part in (entry.get("description"), entry.get("documentation"))
        )
        field_hits = sorted(set(DD_FIELD_RE.findall(text)))
        option_hits = sorted(set(OPTION_RE.findall(text)))
        score = entry.get("reviewer_score_docs")
        checks[name] = {
            "has_documentation": bool(entry.get("documentation")),
            "score": score,
            "score_at_or_above_threshold": (
                score is not None and float(score) >= threshold
            ),
            "names_dd_field_or_path": bool(field_hits),
            "dd_field_or_path_hits": field_hits,
            "names_option_value": bool(option_hits),
            "option_value_hits": option_hits,
        }
    return checks


def report() -> dict:
    """Assemble the fragment's JSON record."""
    state = snapshot()
    checks = read_checks(state)
    docs_pools = {}
    for pool in ("generate_docs", "review_docs", "refine_docs"):
        docs_pools[pool] = {"processed": 0, "spent_usd": 0.0}
    record = {
        "node": "shc-gap-docs-regen",
        "invocation": {"dry_run": DRY_RUN_COMMAND, "live": RUN_COMMAND},
        "dry_run_output": DRY_RUN_OUTPUT,
        "run_log": RUN_LOG,
        "run_id": RUN_ID,
        "run_outcome": {
            "exit": 0,
            "stop_reason": "no_eligible_work",
            "cost_spent_usd": 0.0,
            "cost_limit_usd": 0.9,
            "names_regenerated": 0,
            "docs_pools_processed": docs_pools,
        },
        "spend": {
            "authorized_usd": 1.0,
            "spent_usd": 0.0,
            "basis": "the run's own summary table in the log",
        },
        "state": state["names"],
        "before": state["names"],
        "after": state["names"],
        "before_after_identical": True,
        "before_after_note": (
            "the documentation stage claimed no work (every docs pool processed "
            "0), so the graph state is unchanged; before and after are the same "
            "single read and are recorded once rather than duplicated"
        ),
        "read_checks": checks,
        "blocker": {
            "summary": "neither name is in a state the documentation pools claim",
            "gap_at_outboard_midplane": (
                "name_stage=accepted but docs_stage=exhausted; generate_docs "
                "requires docs_stage=pending, so a prior docs reset is required"
            ),
            "radial_separatrix_distance_at_outboard_midplane": (
                "docs_stage=pending but name_stage=drafted; generate_docs "
                "requires name_stage=accepted, so name acceptance is required"
            ),
            "acceptance_threshold": 0.85,
        },
    }
    return record


def main() -> int:
    mode = sys.argv[1] if len(sys.argv) > 1 else "report"
    if mode == "snapshot":
        print(json.dumps(snapshot(), indent=2, default=str, sort_keys=True))
    elif mode == "report":
        record = report()
        with open(RECORD, "w") as handle:
            json.dump(record, handle, indent=2, sort_keys=True, default=str)
            handle.write("\n")
        print(json.dumps(record["run_outcome"], indent=2, default=str))
        print("wrote", RECORD)
    else:
        raise SystemExit(f"unknown mode: {mode}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
