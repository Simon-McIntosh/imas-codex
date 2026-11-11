"""Read a flush drain's live state and exact-run billing without graph writes."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from imas_codex.graph.client import GraphClient
from imas_codex.standard_names import pools


def read_state(gc):
    groups = pools.identity_state_groups(gc=gc)
    census = pools.classify_identity_state_groups(
        groups, pools.standard_name_pool_predicates()
    )
    coverage = gc.query(
        "MATCH (n:StandardName) RETURN count(n) AS candidates, "
        "count(n.id) AS with_id, count(n.name_stage) AS with_name_stage, "
        "count(n.docs_stage) AS with_docs_stage"
    )[0]
    assert coverage["candidates"] > 0
    assert coverage["candidates"] == coverage["with_id"]
    assert coverage["candidates"] == census["population_count"]
    docs = gc.query(
        "MATCH (n:StandardName) RETURN n.name_stage AS name_stage, "
        "n.docs_stage AS docs_stage, count(*) AS count "
        "ORDER BY name_stage, docs_stage"
    )
    accepted = gc.query(
        "MATCH (n:StandardName {name_stage: 'accepted'}) "
        "RETURN n.id AS id, n.docs_stage AS docs_stage ORDER BY id"
    )
    controls = pools.classify_identity_state_groups(
        [{"name_stage": "unrecognized", "identities": ["control"], "count": 1}],
        pools.standard_name_pool_predicates(),
    )
    assert controls["stranded_count"] == 1
    return {
        "observed_at": datetime.now(UTC).isoformat(),
        "coverage": dict(coverage),
        "stages": [dict(row) for row in docs],
        "accepted_names": [dict(row) for row in accepted],
        "census": census,
        "instrument_control": {"unknown_stage_stranded_count": 1},
    }


def read_receipt(gc, before):
    rows = gc.query(
        "MATCH (r:SNRun) WHERE r.started_at >= datetime($since) "
        "RETURN properties(r) AS receipt ORDER BY r.started_at LIMIT 100",
        since=before["observed_at"],
    )
    candidates = []
    for row in rows:
        receipt = dict(row["receipt"])
        flags = json.loads(receipt.get("invocation_flags") or "{}")
        if flags.get("flush") and flags.get("cost_limit") == 20:
            candidates.append(receipt)
    assert len(candidates) == 1, f"ambiguous drain receipt: {candidates}"
    receipt = candidates[0]
    run_id = receipt["id"]
    costs = gc.query(
        "MATCH (c:LLMCost {run_id: $run_id}) "
        "RETURN c.phase AS phase, count(*) AS calls, "
        "count(c.llm_cost) AS priced, sum(c.llm_cost) AS billed_usd, "
        "count(c.standard_name_ids) AS named_calls ORDER BY phase",
        run_id=run_id,
    )
    # Review costs bind existing names to this invocation without changing origin.
    touched = gc.query(
        "MATCH (c:LLMCost {run_id: $run_id}) "
        "WHERE c.phase = 'review_docs' "
        "UNWIND c.standard_name_ids AS id WITH DISTINCT id "
        "MATCH (n:StandardName {id: id}) "
        "RETURN n.id AS id, n.docs_stage AS docs_stage, "
        "n.reviewed_docs_at AS reviewed_at, "
        "n.reviewer_score_docs AS docs_score ORDER BY id",
        run_id=run_id,
    )
    before_stages = {row["id"]: row["docs_stage"] for row in before["accepted_names"]}
    newly_accepted = [
        dict(row)
        for row in touched
        if row["docs_stage"] == "accepted"
        and before_stages.get(row["id"]) != "accepted"
    ]
    samples = gc.query(
        "MATCH (n:StandardName) WHERE n.id IN $ids "
        "OPTIONAL MATCH (s:StandardNameSource)-[:PRODUCED_NAME]->(n) "
        "RETURN n.id AS id, n.description AS description, n.documentation AS documentation, "
        "n.reviewer_score_docs AS docs_score, n.reviewer_score_name AS name_score, "
        "collect(s.source_id) AS source_paths ORDER BY id",
        ids=[row["id"] for row in newly_accepted[:5]],
    )
    return {
        "run": receipt,
        "costs": [dict(row) for row in costs],
        "billed_usd": sum(float(row["billed_usd"] or 0) for row in costs),
        "docs_reviewed_under_run_id": [dict(row) for row in touched],
        "docs_newly_accepted_under_run_id": newly_accepted,
        "representatives": [dict(row) for row in samples],
    }


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("phase", choices=["before", "after", "progress"])
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    directory = args.directory
    with GraphClient() as gc:
        state = read_state(gc)
        if args.phase != "before":
            before = json.loads((directory / "before.json").read_text())
            state.update(read_receipt(gc, before))
    state["revision"] = (
        subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        if (Path.cwd() / ".git").exists()
        else os.environ["DRAIN_REVISION"]
    )
    state["module"] = pools.__file__
    state["cwd"] = str(Path.cwd().resolve())
    (directory / f"{args.phase}.json").write_text(
        json.dumps(state, indent=2, sort_keys=True, default=str) + "\n"
    )
    census = state["census"]
    summary = {
        "phase": args.phase,
        "population": census["population_count"],
        "stranded": census["stranded_count"],
        "held_for_run": census["terminal_for_run_count"],
        "claimable": census["claimable_count"],
        "accepted_names_by_docs_stage": {
            str(row["docs_stage"]): row["count"]
            for row in state["stages"]
            if row["name_stage"] == "accepted"
        },
    }
    if args.phase != "before":
        summary.update(
            run_id=state["run"]["id"],
            stop_reason=state["run"].get("stop_reason"),
            billed_usd=state["billed_usd"],
            newly_docs_accepted=len(state["docs_newly_accepted_under_run_id"]),
        )
    print(json.dumps(summary, sort_keys=True), flush=True)
    assert census["stranded_count"] == 0, "stranded identities remain"
    assert census["overlap_count"] == 0, "overlapping admission rules"
    if args.phase == "after":
        assert state["run"].get("stop_reason") in {
            "no_eligible_work",
            "budget_exhausted",
        }, "drain did not reach an eligible terminal condition"
        assert state["run"].get("cost_is_exact") is True, "billing is not exact"
        assert state["costs"] and all(
            row["calls"] == row["priced"] for row in state["costs"]
        )
        assert state["billed_usd"] <= 20, "billed spend exceeded authorization"


if __name__ == "__main__":
    main()
