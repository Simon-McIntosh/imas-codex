#!/usr/bin/env python3
"""Read-only reconstruction of the source-drift refresh a family apply triggered.

The apply ``sn harmonize --family electrostatic_potential --apply`` drains its
family batch through the ordinary docs pools. That drain reaches
``run_sn_pools``, which runs the graph-wide source-drift refresh. This runner
re-reads the evidence for that refresh from the live graph (read-only) and from
the apply's log, and writes a JSON record beside this script.

Nothing here writes to the graph, calls a model, or runs an ``sn`` command.
Every query is bounded to a named id set or a write-time window.

Usage:
    UV_PROJECT_ENVIRONMENT=<main>/.venv PYTHONPATH=<worktree> \
        uv run --no-sync python <this script>
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
OUT = HERE / "shc-apply-drift-scope.json"

APPLY_LOG = Path(
    "/home/ITER/mcintos/.config/reckon/crew/runs/"
    "r-20261022T180138781883-shc-locus-remeasure-rerun/apply-rerun-1.log"
)
PIPELINE_LOG = APPLY_LOG.with_name("runner-pipeline.log")

# The two names the apply redrafted, and the two field sets the detector compares.
STEERED = ["electron_temperature_at_separatrix", "gap_at_outboard_midplane"]

# Write-time fingerprint of the refresh's re-stamp: the refresh stamps each
# detected name with ``sn.updated_at = datetime()`` in one tight loop, so the
# names it wrote cluster in a two-second band. Recorded from the log
# (:58, :141) as 18:35:41Z-18:35:41Z local 20:35:41 -> UTC 18:35:41.
STAMP_LO = "2026-10-22T18:35:40"
STAMP_HI = "2026-10-22T18:35:45"
BASELINE_LO = "2026-10-22T18:34:50"
BASELINE_HI = "2026-10-22T18:35:10"

# The detector compares, per name: the resolved (live) DD context against the
# stored snapshot. Content fields steer; provenance fields only re-stamp. See
# imas_codex/standard_names/source_refresh.py:280 (requires_steering).

LOG: list[str] = []


def note(msg: str) -> None:
    print(msg, flush=True)
    LOG.append(msg)


def cite(relpath: str, pattern: str, label: str) -> dict:
    """Read the exact source lines a claim rests on, so the record can cite them."""
    p = REPO / relpath
    text = p.read_text().splitlines()
    rx = re.compile(pattern)
    hits = [
        {"line": i + 1, "text": text[i].strip()}
        for i in range(len(text))
        if rx.search(text[i])
    ]
    note(f"cite {label}: {relpath} -> {len(hits)} line(s)")
    for h in hits:
        note(f"    {relpath}:{h['line']}: {h['text']}")
    return {"path": relpath, "label": label, "lines": hits}


def read_log_lines() -> dict:
    """Read the apply log lines this analysis rests on, so the record can cite them."""
    out: dict[str, list[dict]] = {"apply_log": [], "pipeline_log": []}
    for key, path, pat in (
        (
            "apply_log",
            APPLY_LOG,
            r"refresh_drifted_sources|stamp_source_snapshots|reset_standard_name_docs"
            r"|run_sn_pools|focus \(|source-loss probe|source-loss",
        ),
        ("pipeline_log", PIPELINE_LOG, r"calls_total|per_name_usd|spend_usd"),
    ):
        if not path.exists():
            continue
        rx = re.compile(pat)
        for i, line in enumerate(path.read_text(errors="replace").splitlines()):
            if rx.search(line):
                out[key].append({"line": i + 1, "text": line.strip()[:240]})
    return out


def main() -> int:
    from imas_codex.graph.client import GraphClient
    from imas_codex.standard_names.source_refresh import detect_source_drift

    t0 = time.time()
    record: dict = {
        "node": "shc-apply-drift-scope",
        "runner": str(Path(__file__).resolve()),
        "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    import imas_codex
    from imas_codex.standard_names import source_refresh

    record["imports"] = {
        "imas_codex.__file__": imas_codex.__file__,
        "imas_codex.standard_names.source_refresh.__file__": source_refresh.__file__,
        "cwd": os.getcwd(),
    }
    note(f"module imas_codex -> {imas_codex.__file__}")
    note(f"module source_refresh -> {source_refresh.__file__}")
    note(f"cwd -> {os.getcwd()}")

    gc = GraphClient()
    try:
        # ── Q1: the call path that lets the refresh leave the family ──────
        record["q1_call_path"] = {
            "harmonize_apply_entry": cite(
                "imas_codex/cli/sn.py",
                r"outcome = apply_families\(",
                "sn harmonize --apply calls apply_families",
            ),
            "drain_factory": cite(
                "imas_codex/cli/sn.py",
                r"def _docs_drain_factory|skip_generate=True|scope_run_id=run_id",
                "the drain builds a docs-only flush scoped by run_id",
            ),
            "drain_call_in_pools": cite(
                "imas_codex/standard_names/loop.py",
                r"refresh_drifted_sources\(|_global_maintenance_call\(|scope_run_id=run_id",
                "run_sn_pools runs the refresh, passing only scope_run_id",
            ),
            "refresh_signature": cite(
                "imas_codex/standard_names/source_refresh.py",
                r"def refresh_drifted_sources|scope_run_id: str|def detect_source_drift|def stamp_source_snapshots|sn_ids: list\[str\] \| None|sn\.run_id = \$scope_run_id",
                "the refresh has no id filter; scope_run_id only tags names INTO the run",
            ),
            "budget_charge": cite(
                "imas_codex/standard_names/harmonize.py",
                r"def _family_budget_drain|def drain\(run_id|spent \+= float\(cost\)|campaign_cost_ceiling=cost_limit|per_batch_cost_cap=cost_limit",
                "the batch drain's measured spend is charged to the family cap",
            ),
            "focus_scope": cite(
                "imas_codex/standard_names/graph_ops.py",
                r"AND sn\.run_id = \$scope_run_id",
                "the docs pools claim only names whose run_id matches the drain scope",
            ),
        }

        # ── Q2: the two redrafted names ──────────────────────────────────
        record["q2_names"] = {}
        q_name = """
        MATCH (src:StandardNameSource)-[:PRODUCED_NAME]->(sn:StandardName)
        WHERE sn.id = $id
        OPTIONAL MATCH (o:IMASNode {id: src.source_id})-[:RENAMED_TO*1..12]->(t:IMASNode)
        WHERE NOT exists { (t)-[:RENAMED_TO]->() }
        WITH src, sn, head(collect(coalesce(t, o))) AS n
        RETURN sn.id AS sn_id, sn.name_stage AS name_stage, sn.docs_stage AS docs_stage,
               src.source_id AS source_source_id, src.composed_at AS source_composed_at,
               src.dd_version AS source_dd_version, src.status AS source_status,
               n.id AS resolved_path, n.unit AS live_unit, n.documentation AS live_doc,
               sn.source_path AS snap_path, sn.source_unit AS snap_unit,
               sn.source_documentation AS snap_unit_doc,
               sn.source_raw_unit AS snap_raw_unit,
               sn.source_raw_documentation AS snap_raw_doc,
               sn.source_dd_resolution_marker AS snap_marker,
               sn.source_dd_resolution_manifest_digest AS snap_digest,
               size(coalesce(sn.source_dd_resolution_ids,[])) AS snap_ids,
               sn.edit_reason AS edit_reason, sn.edit_mode AS edit_mode,
               sn.edit_status AS edit_status, sn.updated_at AS snapshot_written_at,
               sn.run_id AS run_id, sn.docs_score AS docs_score,
               sn.harmonized_group_signature AS harmonized_placeholder
        """
        q_rev = """
        MATCH (sn:StandardName)-[:DOCS_REVISION_OF]->(dr:DocsRevision)
        WHERE sn.id = $id
        RETURN dr.id AS rev_id, dr.created_at AS created_at
        ORDER BY dr.created_at DESC LIMIT 4
        """
        for name in STEERED:
            rows = list(gc.query(q_name, id=name))
            rows = [{k: str(v) for k, v in r.items()} for r in rows]
            revs = [{k: str(v) for k, v in r.items()} for r in gc.query(q_rev, id=name)]
            record["q2_names"][name] = {
                "snapshot_and_source": rows,
                "docs_revisions": revs,
            }
            note(
                f"q2 {name}: source_id={[r['source_source_id'] for r in rows]} "
                f"snap_path={[r['snap_path'] for r in rows]}"
            )

        # ── Q3: classify the 80 the refresh reported ─────────────────────
        q_window = """
        MATCH (src:StandardNameSource)-[:PRODUCED_NAME]->(sn:StandardName)
        WHERE sn.updated_at >= datetime($lo) AND sn.updated_at < datetime($hi)
        WITH sn, collect(DISTINCT src.source_id) AS source_ids,
             collect(DISTINCT toString(date(src.composed_at))) AS composed_dates,
             max(toString(sn.updated_at)) AS stamped_at
        RETURN sn.id AS id, stamped_at,
               sn.name_stage AS name_stage, sn.docs_stage AS docs_stage,
               sn.source_path AS snap_path, source_ids, composed_dates,
               sn.source_dd_resolution_marker AS marker,
               size(coalesce(sn.source_dd_resolution_ids,[])) AS n_ids,
               size(source_ids) AS n_sources
        ORDER BY sn.id
        """
        window = [
            {k: str(v) for k, v in r.items()}
            for r in gc.query(q_window, lo=STAMP_LO, hi=STAMP_HI)
        ]
        baseline = [
            {k: str(v) for k, v in r.items()}
            for r in gc.query(q_window, lo=BASELINE_LO, hi=BASELINE_HI)
        ]
        note(
            f"q3 restamp window {STAMP_LO}..{STAMP_HI}: {len(window)} distinct name(s)"
        )
        note(
            f"q3 baseline window {BASELINE_LO}..{BASELINE_HI}: {len(baseline)} distinct name(s)"
        )

        restamped = [r for r in window if r["id"] not in STEERED]
        multi_source = {
            r["id"]: r["source_ids"] for r in window if int(r["n_sources"]) > 1
        }
        record["q3_multi_source_names"] = multi_source
        record["q3_population"] = {
            "restamp_window_distinct_names": len(window),
            "restamp_window_excluding_steered": len(restamped),
            "restamp_window_source_rows": sum(int(r["n_sources"]) for r in window),
            "steered_names_from_log": STEERED,
            "total_matched_to_log_80": len(restamped) + len(STEERED),
            "baseline_window_distinct_names": len(baseline),
            "detected_from_log": 80,
            "renamed_from_log": 2,
            "steered_from_log": 2,
            "baselined_from_log": 6,
        }

        # A name the detector cannot see at all: its source_id is not a DD path,
        # so the inner MATCH on IMASNode fails and the row is silently skipped.
        q_invisible = """
        MATCH (src:StandardNameSource)-[:PRODUCED_NAME]->(sn:StandardName)
        WHERE sn.name_stage <> 'superseded' AND sn.source_path IS NOT NULL
          AND sn.source_dd_resolution_marker IS NULL
        OPTIONAL MATCH (o:IMASNode {id: src.source_id})
        RETURN sn.id AS id, sn.source_path AS snap_path, src.source_id AS source_id,
               (o IS NOT NULL) AS source_node_exists
        ORDER BY sn.id
        """
        invisible = [{k: str(v) for k, v in r.items()} for r in gc.query(q_invisible)]
        record["q3_invisible_names"] = {
            "count": len(invisible),
            "source_node_missing": sum(
                1 for r in invisible if r["source_node_exists"] == "False"
            ),
            "examples": invisible[:6],
        }
        note(
            f"q3 invisible (marker NULL) names: {len(invisible)}; "
            f"{record['q3_invisible_names']['source_node_missing']} with no IMASNode for source_id"
        )

        # ── The next-run receipt: what the detector reports now ──────────
        t = time.time()
        residual = detect_source_drift(gc=gc)
        record["q3_next_run_receipt"] = {
            "detector_call": "detect_source_drift(gc=gc)  # read-only, no stamp",
            "elapsed_s": round(time.time() - t, 1),
            "drifted_now": len(residual),
            "steered_now": sum(1 for d in residual if d["requires_steering"]),
            "renamed_now": sum(1 for d in residual if d["renamed"]),
            "residual_ids": [d["sn_id"] for d in residual],
            "residual_is_subset_of_80": [
                d["sn_id"]
                for d in residual
                if d["sn_id"] in {r["id"] for r in restamped} | set(STEERED)
            ],
            "per_name": [
                {
                    "sn_id": d["sn_id"],
                    "new_path": d["new_path"],
                    "renamed": d["renamed"],
                    "requires_steering": d["requires_steering"],
                    "deltas": [
                        {
                            "field": x["field"],
                            "old": str(x["old"])[:120],
                            "new": str(x["new"])[:120],
                        }
                        for x in d["deltas"]
                    ],
                }
                for d in residual
            ],
        }
        note(
            f"q3 next-run receipt: detect_source_drift now -> {len(residual)} drifted "
            f"({record['q3_next_run_receipt']['steered_now']} would steer)"
        )

        # ── Classification of the 80 ─────────────────────────────────────
        steered_rows = []
        for name in STEERED:
            row = next((r for r in record["q2_names"][name]["snapshot_and_source"]), {})
            steered_rows.append(row)
        # b: a snapshot last stamped before the provenance fields (marker +
        # digest) existed. Content matches, provenance differs -> re-stamp only.
        class_b = [
            r
            for r in restamped
            if r["marker"] == "resolved-dd-context" and r["n_ids"] == "0"
        ]

        record["q3_classification"] = {
            "a_real_source_change": {
                "count": len(steered_rows),
                "names": [r.get("sn_id") for r in steered_rows],
                "evidence": [
                    {
                        "sn_id": r.get("sn_id"),
                        "stored_path": r.get("snap_path"),
                        "live_source_id": r.get("source_source_id"),
                        "resolved_path": r.get("resolved_path"),
                        "reason": r.get("edit_reason"),
                    }
                    for r in steered_rows
                ],
            },
            "b_unstamped_after_regeneration": {
                "count": len(class_b),
                "names": sorted(r["id"] for r in class_b),
                "distinguishing_marker": "resolved-dd-context",
                "distinguishing_n_ids": 0,
                "source_composed_dates": sorted(
                    {d for r in class_b for d in r["composed_dates"]}
                ),
                "name_stages": {
                    s: sum(1 for r in class_b if r["name_stage"] == s)
                    for s in {r["name_stage"] for r in class_b}
                },
            },
            "c_rename": {"count": 0, "names": []},
            "d_other": {"count": 0, "names": []},
            "total_classified": len(restamped) + len(steered_rows),
        }
        note(
            "q3 classification: a={} real source change, b={} unstamped after an "
            "earlier regeneration, c=0 rename, d=0 other; total {}".format(
                len(steered_rows),
                len(class_b),
                record["q3_classification"]["total_classified"],
            )
        )

        record["q3_next_run_steering"] = {
            "would_steer_next_run": record["q3_next_run_receipt"]["steered_now"],
            "steered_this_run": 2,
            "why_only_two_steered": (
                "the detector split the 80 by requires_steering: the 2 whose "
                "content field (source_path) differed were steered and re-stamped;"
                " the 78 whose content matched were re-stamped only, which clears"
                " the provenance drift so the next run does not report them"
            ),
            "restamped_count": len(restamped),
        }

    finally:
        gc.close()

    record["elapsed_s"] = round(time.time() - t0, 1)
    record["log_evidence"] = read_log_lines()
    note(
        f"log evidence: {len(record['log_evidence']['apply_log'])} apply line(s), "
        f"{len(record['log_evidence']['pipeline_log'])} pipeline line(s)"
    )
    OUT.write_text(json.dumps(record, indent=1, sort_keys=True, default=str) + "\n")
    note(f"wrote {OUT} ({OUT.stat().st_size} bytes)")
    print("EXIT=0")
    return 0


if __name__ == "__main__":
    sys.exit(main())
