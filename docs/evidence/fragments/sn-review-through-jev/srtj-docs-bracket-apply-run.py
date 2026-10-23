#!/usr/bin/env python3
"""Apply the bare-link normaliser to the live catalogue once, under control.

The normaliser (``_normalize_bare_doc_links``) already runs unscoped from the
pool's post-rotation reconcile, so the next pool run would rewrite these
documents anyway. This node applies it once, deliberately, against a fresh
read-only dry run, and reads the result back.

Three phases, in this order, and the order is the point:

1. **Restore** — turn the fresh dry run into a restore record holding every
   changed document's text before and after. It lands on disk before the graph
   is touched, so an apply that goes wrong has a record to restore from.
2. **Apply** — one unscoped call to the product function. No bespoke Cypher and
   no hand-edited document.
3. **Read back** — the updated count against the dry run's, every stored text
   against its dry-run after text, an independent rescan of the accepted
   documents for the prose-link rule, and a second call to prove idempotence.

Nothing here calls a model. Every graph read is bounded to a named row set.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from imas_codex.graph.client import GraphClient
from imas_codex.standard_names.docs_gates import (
    DocumentationGateOutcome,
    score_documentation,
)
from imas_codex.standard_names.graph_ops import _normalize_bare_doc_links

RESTORE_SCHEMA = "imas-codex/sn-review-through-jev/srtj-docs-bracket-apply-restore/1"
APPLY_SCHEMA = "imas-codex/sn-review-through-jev/srtj-docs-bracket-apply/1"
ACCEPTED_CYPHER = (
    "MATCH (sn:StandardName) "
    "WHERE sn.name_stage IN ['accepted', 'approved'] "
    "AND sn.documentation IS NOT NULL "
    "RETURN sn.id AS id, sn.documentation AS docs ORDER BY sn.id"
)
CHUNK = 200


def build_restore(dry_run: dict[str, Any], source: str) -> dict[str, Any]:
    """Turn a dry run into the restore record: text before and after, per id."""

    documents = dry_run["documents"]
    assert documents, "dry run changed no documents"
    missing = [row["id"] for row in documents if not {"before", "after"} <= set(row)]
    assert not missing, f"dry run rows lack before/after text: {missing[:5]}"
    return {
        "schema": RESTORE_SCHEMA,
        "created_at": dry_run["created_at"],
        "read_only_dry_run": True,
        "purpose": (
            "Restoring means writing each document's `before` text back to the "
            "StandardName.documentation property of its `id`."
        ),
        "source_dry_run": source,
        "live_name_control": dry_run["live_name_control"],
        "accepted_documents_read": dry_run["accepted_documents_read"],
        "documents_changed": dry_run["documents_changed"],
        "action_counts": dry_run["action_counts"],
        "documents": [
            {
                "id": row["id"],
                "before": row["before"],
                "after": row["after"],
                "actions": row["actions"],
            }
            for row in documents
        ],
    }


def write_json(path: Path, payload: dict[str, Any]) -> int:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"
    )
    return path.stat().st_size


def prose_rule_failures(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Accepted documents the prose-link rule still fails, with its own reason."""

    failing = []
    for row in rows:
        result = score_documentation(
            row["docs"], documenting_name=row["id"]
        ).gate_vector["link_hygiene"]
        if result.outcome is not DocumentationGateOutcome.PASS:
            failing.append(
                {
                    "document_id": row["id"],
                    "outcome": str(result.outcome),
                    "reason": result.reason,
                }
            )
    return failing


def read_docs(graph: Any, ids: list[str]) -> dict[str, str]:
    found: dict[str, str] = {}
    for start in range(0, len(ids), CHUNK):
        batch = ids[start : start + CHUNK]
        for row in graph.query(
            "MATCH (sn:StandardName) WHERE sn.id IN $ids "
            "RETURN sn.id AS id, sn.documentation AS docs",
            ids=batch,
        ):
            found[row["id"]] = row["docs"]
    return found


def choose_examples(dry_run: dict[str, Any], count: int = 10, linked_min: int = 3):
    """Pick quoted examples, guaranteeing the linked action is represented."""

    by_id = {row["id"]: row for row in dry_run["documents"]}
    linked = [
        row["id"]
        for row in dry_run["documents"]
        if any(a["action"] == "linked" for a in row["actions"])
    ]
    chosen = linked[:linked_min]
    for row in dry_run["documents"]:
        if len(chosen) >= count:
            break
        if row["id"] not in chosen:
            chosen.append(row["id"])
    return [by_id[document_id] for document_id in chosen[:count]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run-json", type=Path, required=True)
    parser.add_argument(
        "--restore-out",
        type=Path,
        help="write the restore record here, before anything is applied",
    )
    parser.add_argument("--out", type=Path)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="run the apply and readback phases; without it, only write the restore record",
    )
    args = parser.parse_args()

    dry_run = json.loads(args.dry_run_json.read_text())
    restore = build_restore(dry_run, str(args.dry_run_json))
    report: dict[str, Any] = {
        "schema": APPLY_SCHEMA,
        "created_at": datetime.now(UTC).isoformat(),
        "dry_run_json": str(args.dry_run_json),
    }
    if args.restore_out:
        report["restore_record_bytes"] = write_json(args.restore_out, restore)
        report["restore_record"] = str(args.restore_out)
        print(
            f"restore_written={args.restore_out} "
            f"documents={len(restore['documents'])} "
            f"bytes={report['restore_record_bytes']}"
        )
    if not args.apply:
        return

    planned = {row["id"]: row["after"] for row in dry_run["documents"]}
    planned_before = {row["id"]: row["before"] for row in dry_run["documents"]}

    with GraphClient() as graph:
        control = graph.query(
            "MATCH (sn:StandardName) RETURN count(sn) AS names, count(sn.id) AS ids"
        )[0]
        assert control["names"] > 0 and control["ids"] == control["names"], control
        accepted = graph.query(ACCEPTED_CYPHER)
        assert accepted, "no accepted documents read"

        # The dry run is only a plan if the text it planned against is still the
        # text in the graph. Any drift here means the plan is stale.
        pre = read_docs(graph, sorted(planned_before))
        drifted = sorted(
            document_id
            for document_id, before in planned_before.items()
            if pre.get(document_id) != before
        )
        report["dry_run_still_current"] = not drifted
        report["drifted_since_dry_run"] = drifted
        if drifted:
            report["stopped"] = "graph changed since the dry run; nothing applied"
            print(f"STOPPED drifted={len(drifted)}")
            if args.out:
                write_json(args.out, report)
            raise SystemExit(1)

        updated = _normalize_bare_doc_links(graph)
        report["updated_reported"] = updated
        report["dry_run_documents_changed"] = dry_run["documents_changed"]
        report["updated_matches_dry_run"] = updated == dry_run["documents_changed"]

        stored = read_docs(graph, sorted(planned))
        mismatched = sorted(
            document_id
            for document_id, after in planned.items()
            if stored.get(document_id) != after
        )
        report["stored_equals_dry_run_after"] = not mismatched
        report["stored_mismatches"] = mismatched

        # Independent of the dry run: re-read the accepted documents and put
        # each through the prose-link rule as the gate implements it.
        accepted_after = graph.query(ACCEPTED_CYPHER)
        failing = prose_rule_failures(accepted_after)
        report["accepted_documents_read_after"] = len(accepted_after)
        report["prose_rule_failing_after"] = len(failing)
        report["prose_rule_failing_documents"] = failing

        second = _normalize_bare_doc_links(graph)
        report["second_call_updated"] = second
        report["idempotent"] = second == 0

        examples = choose_examples(dry_run)
        report["examples"] = [
            {
                "document_id": row["id"],
                "actions": row["actions"],
                "before": row["before"],
                "after": row["after"],
            }
            for row in examples
        ]
        report["examples_linked"] = sum(
            1
            for row in examples
            if any(a["action"] == "linked" for a in row["actions"])
        )
        report["action_counts"] = dry_run["action_counts"]
        report["live_name_control"] = dry_run["live_name_control"]

    checks = {
        "updated_matches_dry_run": report["updated_matches_dry_run"],
        "stored_equals_dry_run_after": report["stored_equals_dry_run_after"],
        "prose_rule_zero_failing": report["prose_rule_failing_after"] == 0,
        "second_call_zero": report["idempotent"],
        "examples_at_least_ten": len(report["examples"]) >= 10,
        "examples_linked_at_least_three": report["examples_linked"] >= 3,
    }
    report["checks"] = checks
    report["all_checks_passed"] = all(checks.values())
    if args.out:
        report["out_bytes"] = write_json(args.out, report)
    print("checks=" + json.dumps(checks))
    print(
        f"updated={report['updated_reported']} planned={report['dry_run_documents_changed']} "
        f"mismatches={len(report['stored_mismatches'])} "
        f"prose_failing={report['prose_rule_failing_after']} "
        f"second={report['second_call_updated']} "
        f"examples={len(report['examples'])} linked={report['examples_linked']}"
    )
    print(f"out={args.out} bytes={report.get('out_bytes')}")
    counts = Counter(
        a["action"] for row in dry_run["documents"] for a in row["actions"]
    )
    print(f"actions={dict(counts)}")
    assert report["all_checks_passed"], checks


if __name__ == "__main__":
    main()
