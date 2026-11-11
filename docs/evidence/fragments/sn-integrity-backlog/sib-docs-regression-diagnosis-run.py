#!/usr/bin/env python
"""Read-only attribution of the docs-stage regression in drain job 1310773.

For every live accepted name whose ``docs_stage`` left ``accepted`` between the
drain's before and after censuses, this reader joins three evidence sources and
assigns the name to the writer that moved it:

* the drain's own log, which names the writers it ran (``refresh_drifted_sources``,
  ``refresh_drifted_sources``' ``reset_standard_name_docs`` lines, the
  ``review_name`` pool failures and the startup reconciles);
* ``DocsRevision`` receipts, which ``reset_standard_name_docs`` creates one per
  reset name at the moment of the reset;
* ``StandardNameChange`` receipts written in the drain window.

It also reports the live names with no producing source and the parents of the
``review_name`` refusal, so each named cause has a count. Read-only: the only
write is the JSON report named by ``--out``.

Usage::

    python sib-docs-regression-diagnosis-run.py \
        --run-dir /path/to/r-...-sib-docs-drain-after-admission --out report.json
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

#: The drain log stamps in UTC+1 (Europe/Paris); the run itself started at
#: 17:24:01Z and its first log line reads 18:24:02. These windows are UTC.
WINDOW_START = "2026-11-11T17:20:00+00:00"
WINDOW_END = "2026-11-11T18:10:00+00:00"
RESET_START = "2026-11-11T17:27:30+00:00"
RESET_END = "2026-11-11T17:29:30+00:00"
DRAIN_RUN_ID = "9f56f177-cb52-4a19-9edd-4ebc797ec58e"


def _stage_map(path: Path) -> dict[str, str | None]:
    rows = json.loads(path.read_text())["accepted_names"]
    return {r["id"]: r.get("docs_stage") for r in rows}


def _transitions(run_dir: Path) -> dict:
    before = _stage_map(run_dir / "before.json")
    after = _stage_map(run_dir / "after.json")
    left = sorted(
        n for n, s in before.items() if s == "accepted" and after.get(n) != "accepted"
    )
    entered = sorted(
        n for n, s in before.items() if s != "accepted" and after.get(n) == "accepted"
    )
    left_detail = {
        n: {"before": before[n], "after": after.get(n), "in_after": n in after}
        for n in left
    }
    return {
        "population_before": len(before),
        "population_after": len(after),
        "left_accepted": left,
        "left_accepted_detail": left_detail,
        "left_to": dict(Counter(str(left_detail[n]["after"]) for n in left)),
        "entered_accepted": entered,
        "net": len(left) - len(entered),
    }


def _live_resets(gc) -> list[dict]:
    rows = gc.query(
        """
        MATCH (rev:DocsRevision)
        WHERE rev.created_at >= datetime($start)
          AND rev.created_at < datetime($end)
        RETURN rev.standard_name_id AS sn_id,
               rev.created_at AS created_at,
               rev.revision_number AS revision_number
        ORDER BY rev.created_at
        """,
        start=RESET_START,
        end=RESET_END,
    )
    return [
        {
            "sn_id": r["sn_id"],
            "created_at": str(r["created_at"]),
            "revision_number": r["revision_number"],
        }
        for r in (rows or [])
    ]


def _window_changes(gc) -> list[dict]:
    rows = gc.query(
        """
        MATCH (change:StandardNameChange)
        WHERE change.changed_at >= datetime($start)
          AND change.changed_at < datetime($end)
        RETURN change.id AS id, change.operation AS operation,
               change.origin AS origin, change.reason AS reason,
               change.run_id AS run_id, change.changed_at AS changed_at
        LIMIT 4000
        """,
        start=WINDOW_START,
        end=WINDOW_END,
    )
    return [
        {
            "id": r["id"],
            "operation": r["operation"],
            "origin": r["origin"],
            "reason": (r["reason"] or "")[:160],
            "run_id": r["run_id"],
            "changed_at": str(r["changed_at"]),
        }
        for r in (rows or [])
    ]


def _sourceless(gc) -> list[str]:
    rows = gc.query(
        """
        MATCH (sn:StandardName)
        WHERE sn.name_stage <> 'superseded'
          AND NOT EXISTS { (:StandardNameSource)-[:PRODUCED_NAME]->(sn) }
        RETURN sn.id AS id, sn.name_stage AS name_stage, sn.docs_stage AS docs_stage
        ORDER BY sn.id
        """
    )
    return [
        {"id": r["id"], "name_stage": r["name_stage"], "docs_stage": r["docs_stage"]}
        for r in (rows or [])
    ]


def _refused_review_names(gc) -> dict:
    """Names whose DD binding lacks an exact published version.

    The ``review_name`` refusal is raised while resolving a source binding's
    ``dd_version``; the bindings without one are the parents of the refusal.
    """
    summary = gc.query(
        """
        MATCH (src:StandardNameSource)-[:PRODUCED_NAME]->(sn:StandardName)
        WHERE sn.name_stage <> 'superseded'
        RETURN count(src) AS bindings,
               sum(CASE WHEN src.dd_version IS NULL
                          OR trim(src.dd_version) = '' THEN 1 ELSE 0 END) AS without,
               sum(CASE WHEN src.dd_version IS NULL
                          OR trim(src.dd_version) = '' THEN 0 ELSE 1 END) AS with_version,
               count(DISTINCT CASE WHEN src.dd_version IS NULL
                          OR trim(src.dd_version) = '' THEN sn.id END) AS names_without
        """
    )
    sample = gc.query(
        """
        MATCH (src:StandardNameSource)-[:PRODUCED_NAME]->(sn:StandardName)
        WHERE sn.name_stage <> 'superseded'
          AND (src.dd_version IS NULL OR trim(src.dd_version) = '')
        RETURN sn.id AS sn_id, src.id AS source_id, sn.docs_stage AS docs_stage
        LIMIT 25
        """
    )
    row = (summary or [{}])[0]
    return {
        "bindings": row.get("bindings"),
        "bindings_without_dd_version": row.get("without"),
        "bindings_with_dd_version": row.get("with_version"),
        "names_without_dd_version": row.get("names_without"),
        "sample": sample or [],
    }


def _drift_probe(gc, names: list[str]) -> list[dict]:
    """Stored source snapshot against the live DD node for a few reset names.

    Shows which snapshot field moved: a documentation/unit delta is content
    drift that steers, anything else is provenance-only drift that re-stamps.
    """
    rows = gc.query(
        """
        MATCH (src:StandardNameSource)-[:PRODUCED_NAME]->(sn:StandardName)
        WHERE sn.id IN $id
        OPTIONAL MATCH (o:IMASNode)
        WHERE o.id = src.source_id
           OR o.id = replace(src.source_id, 'dd:', '')
        OPTIONAL MATCH (o)-[:RENAMED_TO*1..12]->(t:IMASNode)
        WHERE NOT exists { (t)-[:RENAMED_TO]->() }
        WITH src, sn, o, coalesce(head(collect(t)), o) AS n
        RETURN src.source_id AS source_id, sn.id AS sn_id, sn.name_stage AS name_stage,
               sn.docs_stage AS docs_stage,
               sn.source_path AS old_path, n.id AS new_path,
               sn.source_unit AS old_unit, n.unit AS new_unit,
               sn.source_documentation AS old_doc, n.documentation AS new_doc,
               sn.source_dd_resolution_ids AS old_ids,
               sn.source_dd_resolution_marker AS old_marker
        LIMIT 600
        """,
        id=names,
    )
    out = []
    for r in rows or []:
        out.append(
            {
                "sn_id": r["sn_id"],
                "source_id": r["source_id"],
                "name_stage": r["name_stage"],
                "docs_stage": r["docs_stage"],
                "path_changed": (r["old_path"] or "") != (r["new_path"] or ""),
                "unit_changed": (r["old_unit"] or "") != (r["new_unit"] or ""),
                "doc_changed": (r["old_doc"] or "") != (r["new_doc"] or ""),
                "old_doc_len": len(r["old_doc"] or ""),
                "new_doc_len": len(r["new_doc"] or ""),
                "has_resolution_ids": bool(r["old_ids"]),
            }
        )
    return out


def _name_state(gc, names: list[str]) -> list[dict]:
    rows = gc.query(
        """
        UNWIND $names AS id
        MATCH (sn:StandardName {id: id})
        OPTIONAL MATCH (src:StandardNameSource)-[:PRODUCED_NAME]->(sn)
        RETURN sn.id AS id, sn.name_stage AS name_stage,
               sn.docs_stage AS docs_stage, sn.run_id AS run_id,
               sn.docs_chain_length AS chain,
               count(DISTINCT src) AS sources
        """,
        names=names,
    )
    return [dict(r) for r in (rows or [])]


def _docs_revision_diagnostics(gc) -> dict:
    total = gc.query("MATCH (rev:DocsRevision) RETURN count(rev) AS n")
    shape = gc.query(
        """
        MATCH (rev:DocsRevision)
        RETURN rev.id AS id, rev.created_at AS created_at LIMIT 2
        """
    )
    return {
        "total": (total or [{}])[0].get("n"),
        "sample": [
            {"id": r["id"], "created_at": str(r["created_at"])} for r in (shape or [])
        ],
    }


def _log_writers(log_path: Path) -> dict:
    text = log_path.read_text(errors="replace")
    resets = re.findall(r"reset_standard_name_docs: (\d+)/(\d+) reset to docs pending \(run_id=([^)]*)\)", text)
    per_run = Counter()
    reset_total = 0
    for reset, attempted, run_id in resets:
        per_run[run_id] += 1
        reset_total += int(reset)
    return {
        "log_path": str(log_path),
        "log_lines": text.count("\n"),
        "reset_lines": len(resets),
        "reset_names": reset_total,
        "reset_run_ids": len(per_run),
        "drifted_line": next(
            (ln for ln in text.splitlines() if "refresh_drifted_sources:" in ln and "drifted" in ln),
            None,
        ),
        "unchecked_line": next(
            (ln for ln in text.splitlines() if "refresh_drifted_sources:" in ln and "unchecked" in ln),
            None,
        ),
        "reset_context_line": next(
            (ln for ln in text.splitlines() if "stamp_source_snapshots: stamped" in ln),
            None,
        ),
        "review_name_failures": len(re.findall(r"pool\[review_name[^\]]*\] process failed", text)),
        "review_name_refusals": text.count(
            "DD resolution version must be one exact published version"
        ),
        "source_loss_line": next(
            (ln for ln in text.splitlines() if "source-loss probe" in ln),
            None,
        ),
        "stranded_line": next(
            (ln for ln in text.splitlines() if "stranded identities=" in ln),
            None,
        ),
        "startup_reconcile_lines": [
            ln
            for ln in text.splitlines()
            if "reconcile_vocab_gaps" in ln or "retry_vocab_gap_sources_on_grammar_change" in ln
        ][:6],
    }


def attribute(gc, run_dir: Path) -> dict:
    trans = _transitions(run_dir)
    left = set(trans["left_accepted"])
    resets = _live_resets(gc)
    reset_ids = {r["sn_id"] for r in resets}
    changes = _window_changes(gc)

    by_writer: Counter = Counter()
    unaccounted: list[str] = []
    per_name = {}
    for name in sorted(left):
        tags = []
        if name in reset_ids:
            tags.append("refresh_drifted_sources/apply_edit(docs)->reset_standard_name_docs")
        if not tags:
            unaccounted.append(name)
        else:
            by_writer[tags[0]] += 1
        per_name[name] = {
            **trans["left_accepted_detail"][name],
            "writers": tags,
        }

    # Which of the reset names were accepted before the drain?
    reset_on_left = sorted(left & reset_ids)
    reset_not_left = sorted(reset_ids - left)
    return {
        "transitions": {k: v for k, v in trans.items() if k != "left_accepted_detail"},
        "by_writer": dict(by_writer),
        "unaccounted": unaccounted,
        "per_name": per_name,
        "resets": {
            "count": len(resets),
            "on_left_accepted": len(reset_on_left),
            "not_left": len(reset_not_left),
            "not_left_ids": reset_not_left,
            "window": [RESET_START, RESET_END],
            "first": resets[:3],
            "last": resets[-3:],
            "_rows": resets,
        },
        "changes_in_window": {
            "total": len(changes),
            "by_operation": dict(Counter(str(c["operation"]) for c in changes)),
            "by_origin": dict(Counter(str(c["origin"]) for c in changes)),
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    run_dir = Path(args.run_dir)

    from imas_codex.graph.client import GraphClient

    with GraphClient() as gc:
        attribution = attribute(gc, run_dir)
        reset_names = [r["sn_id"] for r in attribution["resets"].get("_rows", [])]
        unaccounted = attribution["unaccounted"]
        report = {
            "instrument": {
                "drain_run_id": DRAIN_RUN_ID,
                "window": [WINDOW_START, WINDOW_END],
                "run_dir": str(run_dir),
            },
            "log": _log_writers(run_dir / "drain-1310773.log"),
            "attribution": attribution,
            "sourceless": _sourceless(gc),
            "drift_probe": _drift_probe(gc, reset_names + unaccounted),
            "unaccounted_state": _name_state(gc, unaccounted),
            "refused_review_names": _refused_review_names(gc),
            "docs_revisions": _docs_revision_diagnostics(gc),
        }
        attribution.pop("_rows", None)

    Path(args.out).write_text(json.dumps(report, indent=1, sort_keys=True))
    print(
        json.dumps(
            {
                "left_accepted": len(report["attribution"]["transitions"]["left_accepted"]),
                "by_writer": report["attribution"]["by_writer"],
                "unaccounted": len(report["attribution"]["unaccounted"]),
                "resets": report["attribution"]["resets"]["count"],
                "sourceless": len(report["sourceless"]),
                "bindings_without_dd_version": report["refused_review_names"][
                    "bindings_without_dd_version"
                ],
                "review_name_failures": report["log"]["review_name_failures"],
                "out": args.out,
            },
            indent=1,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())