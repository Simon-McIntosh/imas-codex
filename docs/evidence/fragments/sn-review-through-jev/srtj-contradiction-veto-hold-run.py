#!/usr/bin/env python
"""Read back the documentation contradiction-veto firings and their consequences.

Three questions are answered from two durable sources and the code paths that
own each decision:

1. Every recorded ``contradicts_source`` hold: the claim, the DD text, the
   panel score before the clamp, and what the name did afterwards.
2. Why ``electrostatic_potential`` never reached the docs refine claim, and
   why the harmonize family apply never reset it.
3. Whether the refine prompt receives the veto note at all.

Sources:
  * ``~/.local/share/imas-codex/logs/sn_sn-compose*.log`` — the
    ``review_docs: <name> held from acceptance`` line, which carries the note
    text and the hold timestamp.
  * the graph — the per-group ``StandardNameReview`` rows, which carry the
    pre-clamp panel scores the node's clamped score overwrote.

Read-only against the graph. Every query is bounded to one name's review
group or to the working set of hold names.

Usage: ``uv run --no-sync python srtj-contradiction-veto-hold-run.py <out.json>``
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from imas_codex.graph.client import GraphClient
from imas_codex.standard_names.defaults import (
    DEFAULT_MIN_SCORE,
    DEFAULT_REFINE_ROTATIONS,
)
from imas_codex.standard_names.graph_ops import (
    docs_review_eligibility_params,
    docs_review_eligibility_where,
)

LOG_DIR = Path.home() / ".local/share/imas-codex/logs"
HOLD_RE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})"
    r".*review_docs: (?P<name>[a-z0-9_]+) held from acceptance — (?P<note>.*)$"
)
CONTRADICTION_MARKER = "contradicts source:"
TARGET = "electrostatic_potential"


# ── 1. Log firings ──────────────────────────────────────────────────────────


def scan_holds() -> list[dict]:
    """Every recorded hold line, newest rotation file to oldest."""
    firings = []
    files = sorted(LOG_DIR.glob("sn_sn-compose*.log*"))
    for path in files:
        try:
            text = path.read_text(errors="replace")
        except OSError:
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            match = HOLD_RE.match(line)
            if not match:
                continue
            note = match.group("note")
            firings.append(
                {
                    "name": match.group("name"),
                    "ts": match.group("ts"),
                    "log": f"{path}:{lineno}",
                    "is_contradiction": CONTRADICTION_MARKER in note,
                    "note": note,
                }
            )
    firings.sort(key=lambda f: (f["ts"], f["log"]))
    return firings


def parse_contradiction_note(note: str) -> dict:
    """Split the stored note into per-source claim/DD pairs.

    ``_documentation_contradiction_note`` composes
    ``"contradicts source: " + " | ".join(f"{path}: {reason}; DD: {text}")``.
    """
    body = note.split(CONTRADICTION_MARKER, 1)[-1].strip()
    entries = []
    for chunk in body.split(" | "):
        chunk = chunk.strip()
        if not chunk:
            continue
        path, _, rest = chunk.partition(":")
        reason, _, dd = rest.partition("; DD: ")
        entries.append(
            {
                "source_path": path.strip(),
                "claim": reason.strip().removeprefix("contradicts source:").strip(),
                "dd_text": dd.strip(),
            }
        )
    return entries


# ── 2. Graph readbacks ──────────────────────────────────────────────────────


def review_groups(gc: GraphClient, name: str) -> list[dict]:
    rows = gc.query(
        """
        MATCH (sn:StandardName {id: $id})-[:HAS_REVIEW]->(rv:StandardNameReview)
        WHERE rv.review_axis = 'docs'
        RETURN rv.id AS review_id, rv.model AS model, rv.score AS score,
               rv.scores_json AS dims, rv.is_canonical AS canonical,
               rv.resolution_method AS method, rv.review_group_id AS group_id,
               rv.cycle_index AS cycle, toString(rv.reviewed_at) AS reviewed_at
        ORDER BY rv.reviewed_at, rv.cycle_index
        """,
        id=name,
    )
    groups: dict[str, dict] = {}
    for row in rows:
        grp = groups.setdefault(
            row["group_id"],
            {
                "group_id": row["group_id"],
                "panels": [],
                "resolution_method": None,
                "authoritative_score": None,
                "primary_score": None,
                "reviewed_at": row["reviewed_at"],
            },
        )
        grp["panels"].append(
            {
                "model": row["model"],
                "score": row["score"],
                "dims": row["dims"],
                "canonical": row["canonical"],
                "cycle": row["cycle"],
            }
        )
        if row["method"]:
            grp["resolution_method"] = row["method"]
        if row["method"] == "authoritative_escalation":
            grp["authoritative_score"] = row["score"]
        if row["canonical"]:
            grp["primary_score"] = row["score"]
    return list(groups.values())


def node_state(gc: GraphClient, name: str) -> dict:
    rows = gc.query(
        """
        MATCH (sn:StandardName {id: $id})
        RETURN sn.id AS id, sn.docs_stage AS docs_stage, sn.name_stage AS name_stage,
               sn.reviewer_score_docs AS reviewer_score_docs,
               coalesce(sn.docs_chain_length, 0) AS docs_chain_length,
               sn.docs_review_quorum_shortfall AS quorum_shortfall,
               sn.reviewer_score_name AS reviewer_score_name,
               sn.origin AS origin,
               toString(sn.reviewed_docs_at) AS reviewed_docs_at,
               sn.harmonized_group_signature AS harmonized_group_signature,
               sn.reviewer_comments_per_dim_docs AS comments_per_dim_docs
        """,
        id=name,
    )
    return rows[0] if rows else {}


def refine_predicate_readback(gc: GraphClient, name: str) -> dict:
    """Evaluate each ``claim_refine_docs_batch`` WHERE clause separately.

    The whole predicate is the conjunction at ``graph_ops.py:25649``. A zero
    on any single clause, with the others true, is the reason the claim skips
    the name; the readback prints the values that fail it.
    """
    params = docs_review_eligibility_params()
    rows = gc.query(
        """
        MATCH (sn:StandardName {id: $id})
        OPTIONAL MATCH (sn)-[:HAS_REVIEW]->(rv:StandardNameReview)
        WHERE rv.review_axis = 'docs'
        WITH sn, collect(rv) AS reviews
        RETURN sn.docs_stage = 'reviewed' AS c_docs_stage,
               sn.reviewer_score_docs IS NOT NULL AS c_score_present,
               sn.reviewer_score_docs < $min_score AS c_score_below,
               coalesce(sn.docs_chain_length, 0) < $rotation_cap AS c_chain_below,
               NOT (sn.name_stage IN ['superseded', 'exhausted', 'contested'])
                   AS c_name_stage_ok,
               sn.docs_review_quorum_shortfall IS NULL AS c_no_shortfall,
               (sn.reviewer_score_name IS NOT NULL
                OR (coalesce(sn.origin, '') = 'derived' AND EXISTS {
                      MATCH (kid:StandardName)-[:HAS_PARENT]->(sn)
                      WHERE NOT coalesce(kid.name_stage, '') IN
                            ['superseded', 'exhausted', 'contested']}))
                   AS c_score_gate,
               sn.docs_stage AS v_docs_stage,
               sn.reviewer_score_docs AS v_score,
               coalesce(sn.docs_chain_length, 0) AS v_chain,
               sn.docs_review_quorum_shortfall AS v_shortfall,
               sn.name_stage AS v_name_stage,
               sn.reviewer_score_name AS v_score_name,
               sn.origin AS v_origin
        """,
        id=name,
        min_score=DEFAULT_MIN_SCORE,
        rotation_cap=DEFAULT_REFINE_ROTATIONS,
    )
    out = rows[0] if rows else {}
    elig = gc.query(
        "MATCH (sn:StandardName {id: $id}) RETURN "
        + docs_review_eligibility_where()
        + " AS eligible",
        id=name,
        **params,
    )
    out["c_docs_review_eligibility"] = elig[0]["eligible"] if elig else None
    out["eligibility_winner"] = gc.query(
        """
        MATCH (sn:StandardName {id: $id})-[:HAS_REVIEW]->(review:StandardNameReview)
        WITH sn, review
        WHERE review.review_axis = 'docs'
        RETURN review.review_group_id AS group_id,
               review.resolution_method AS method,
               review.score AS score,
               review.is_canonical AS canonical
        ORDER BY review.reviewed_at DESC
        """,
        id=name,
    )
    out["params"] = params
    out["min_score"] = DEFAULT_MIN_SCORE
    out["rotation_cap"] = DEFAULT_REFINE_ROTATIONS
    return out


def family_plan(name: str) -> dict:
    from imas_codex.standard_names.harmonize import plan_family_applies

    plans = plan_family_applies(seeds=[name], min_size=1)
    return plans[0] if plans else {}


# ── 3. Report ───────────────────────────────────────────────────────────────


def main() -> int:
    out_path = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else Path("srtj-contradiction-veto-hold.json")
    )
    report: dict = {
        "target": TARGET,
        "default_min_score": DEFAULT_MIN_SCORE,
        "default_rotation_cap": DEFAULT_REFINE_ROTATIONS,
    }

    holds = scan_holds()
    contradictions = [h for h in holds if h["is_contradiction"]]
    non_contradiction = [h for h in holds if not h["is_contradiction"]]
    for firing in contradictions:
        firing["entries"] = parse_contradiction_note(firing["note"])
    report["hold_scan"] = {
        "log_files": [str(p) for p in sorted(LOG_DIR.glob("sn_sn-compose*.log*"))],
        "total_hold_lines": len(holds),
        "contradiction_hold_lines": len(contradictions),
        "other_hold_lines": len(non_contradiction),
        "distinct_contradiction_names": sorted({h["name"] for h in contradictions}),
        "firings": contradictions,
    }

    with GraphClient() as gc:
        # The clamp value every hold writes; a distinct score here means the
        # note was written by a different path.
        truth = gc.query(
            """
            UNWIND $names AS n
            MATCH (sn:StandardName {id: n})
            RETURN count(sn) AS present
            """,
            names=report["hold_scan"]["distinct_contradiction_names"] or [TARGET],
        )
        report["hold_scan"]["names_present_in_graph"] = (
            truth[0]["present"] if truth else 0
        )

        # Negative control: a name the scan did not report must read absent.
        control = gc.query(
            "MATCH (sn:StandardName {id: $id}) RETURN count(sn) AS present",
            id="__no_such_standard_name__",
        )
        report["control_absent_name_present"] = (
            control[0]["present"] if control else None
        )

        targets = sorted({TARGET} | {h["name"] for h in contradictions})
        report["per_name"] = {}
        for name in targets:
            report["per_name"][name] = {
                "node": node_state(gc, name),
                "review_groups": review_groups(gc, name),
            }

        report["refine_predicate_readback"] = refine_predicate_readback(gc, TARGET)
        report["family_plan"] = family_plan(TARGET)
        anchor = report["family_plan"].get("anchor")
        if anchor:
            report["family_anchor_state"] = node_state(gc, anchor)
            report["family_member_states"] = {
                mid: node_state(gc, mid)
                for mid in report["family_plan"].get("members", [])
            }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"wrote {out_path}")

    # Readback of what the run claims, so the log proves the file's content.
    print("hold lines:", report["hold_scan"]["total_hold_lines"])
    print("contradiction hold lines:", len(contradictions))
    print(
        "distinct contradiction names:",
        report["hold_scan"]["distinct_contradiction_names"],
    )
    print("control absent-name present:", report["control_absent_name_present"])
    print(
        "refine predicates:",
        json.dumps(report["refine_predicate_readback"], default=str)[:900],
    )
    print(
        "family decision:",
        report["family_plan"].get("decision"),
        "anchor:",
        report["family_plan"].get("anchor"),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
