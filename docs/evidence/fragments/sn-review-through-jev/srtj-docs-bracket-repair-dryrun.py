#!/usr/bin/env python3
"""Read the live documentation pool and preview exact-link normalization."""

from __future__ import annotations

import json
import os
import re
from argparse import ArgumentParser
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from imas_codex.graph.client import GraphClient
from imas_codex.standard_names.doc_links import (
    _is_prose_bracket,
    find_prose_bracket_phrases,
    find_prose_bracket_spans,
    find_self_links,
)
from imas_codex.standard_names.graph_ops import _normalize_bare_doc_links

OUTPUT = Path(__file__).with_suffix(".json")
FULL_OUTPUT = Path(os.environ["RECKON_RUN_DIR"]) / "dryrun-full.json"
LOOSE_UNIT_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9^\-\*\./]*")


def previously_flagged(text: str, name: str) -> bool:
    """Apply the prior loose unit exception to the same unmasked spans."""

    for phrase, _, _ in find_prose_bracket_spans(text):
        stripped = phrase.strip()
        tokens = stripped.split()
        if not (
            all(LOOSE_UNIT_TOKEN.fullmatch(token) for token in tokens)
            and (len(tokens) == 1 or re.search(r"[0-9^/*]", stripped))
        ):
            return True
    return bool(find_self_links(text, name))


def new_rule_flagged(text: str, name: str) -> bool:
    return bool(find_prose_bracket_phrases(text) or find_self_links(text, name))


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--compare-json", type=Path)
    parser.add_argument("--compare-full", type=Path)
    args = parser.parse_args()
    if bool(args.compare_json) != bool(args.compare_full):
        parser.error("--compare-json and --compare-full must be provided together")
    previous = json.loads(args.compare_json.read_text()) if args.compare_json else None
    previous_full = (
        json.loads(args.compare_full.read_text()) if args.compare_full else None
    )

    with GraphClient() as graph:
        controls = graph.query(
            "MATCH (s:StandardName) RETURN count(s) AS names, count(s.id) AS ids"
        )[0]
        assert controls["names"] > 0 and controls["ids"] == controls["names"]
        accepted = graph.query(
            "MATCH (s:StandardName) "
            "WHERE s.name_stage IN ['accepted', 'approved'] "
            "AND s.documentation IS NOT NULL "
            "RETURN s.id AS id, s.documentation AS docs ORDER BY s.id"
        )
        assert accepted and len({row["id"] for row in accepted}) == len(accepted)
        changes = _normalize_bare_doc_links(graph, dry_run=True)
        changed = {row["id"]: row for row in changes}
        assert len(changed) == len(changes)
        # A dry run must leave the source text intact, including its defects.
        reread = graph.query(
            "MATCH (s:StandardName) WHERE s.id IN $ids "
            "RETURN s.id AS id, s.documentation AS docs",
            ids=list(changed),
        )
        assert {row["id"]: row["docs"] for row in reread} == {
            row["id"]: row["before"] for row in changes
        }
        if previous_full:
            previous_before = {
                row["id"]: row["before"] for row in previous_full["documents"]
            }
            previous_reread = graph.query(
                "MATCH (s:StandardName) WHERE s.id IN $ids "
                "RETURN s.id AS id, s.documentation AS docs",
                ids=list(previous_before),
            )
            current_previous_text = {row["id"]: row["docs"] for row in previous_reread}
            assert current_previous_text == previous_before

    original_failing = {
        row["id"] for row in accepted if previously_flagged(row["docs"], row["id"])
    }
    accepted_ids = {row["id"] for row in accepted}
    new_failing = {
        row["id"] for row in accepted if new_rule_flagged(row["docs"], row["id"])
    }
    newly_flagged = sorted(new_failing - original_failing)
    left = [
        row["id"]
        for row in accepted
        if new_rule_flagged(
            changed.get(row["id"], {}).get("after", row["docs"]), row["id"]
        )
    ]
    counts = Counter(action["action"] for row in changes for action in row["actions"])
    accepted_counts = Counter(
        action["action"]
        for row in changes
        if row["id"] in accepted_ids
        for action in row["actions"]
    )
    linked = [
        {"document_id": row["id"], **action}
        for row in changes
        for action in row["actions"]
        if action["action"] == "linked"
    ]
    examples = {
        kind: [
            {"document_id": row["id"], **action}
            for row in changes
            for action in row["actions"]
            if action["action"] == kind
        ][:20]
        for kind in ("unbracketed", "self-link removed")
    }
    report = {
        "created_at": datetime.now(UTC).isoformat(),
        "read_only": True,
        "live_name_control": controls,
        "accepted_documents_read": len(accepted),
        "original_rule_failing": len(original_failing),
        "updated_rule_failing_before": len(new_failing),
        "original_rule_cleared": len(original_failing - set(left)),
        "original_rule_left": sorted(original_failing & set(left)),
        "updated_rule_left": left,
        "newly_flagged_by_tightened_rule": [
            {
                "document_id": row["id"],
                "phrases": find_prose_bracket_phrases(row["docs"]),
            }
            for row in accepted
            if row["id"] in newly_flagged
        ],
        "action_counts": dict(counts),
        "accepted_action_counts": dict(accepted_counts),
        "accepted_documents_changed": sum(row["id"] in accepted_ids for row in changes),
        "documents_changed": len(changes),
        "linked_phrases": linked,
        "examples": examples,
        "documents": changes,
    }
    if previous:

        def unbracketed(rows: list[dict]) -> Counter[tuple[str, str]]:
            return Counter(
                (row["id"], action["phrase"])
                for row in rows
                for action in row["actions"]
                if action["action"] == "unbracketed"
            )

        before_actions = unbracketed(previous["documents"])
        after_actions = unbracketed(changes)
        assert sum(before_actions.values()) == previous["action_counts"]["unbracketed"]
        removed = before_actions - after_actions
        added = after_actions - before_actions
        for (document_id, phrase), count in removed.items():
            assert current_previous_text[document_id].count(f"[{phrase}]") >= count
            assert not _is_prose_bracket(phrase)
        report["comparison_to_previous"] = {
            "previous_json_path": str(args.compare_json),
            "previous_full_path": str(args.compare_full),
            "previous_unbracketed_actions": sum(before_actions.values()),
            "current_unbracketed_actions": sum(after_actions.values()),
            "removed_unbracketed_occurrences": sum(removed.values()),
            "added_unbracketed_occurrences": sum(added.values()),
            "removed_unbracketed": [
                {"document_id": document_id, "phrase": phrase, "count": count}
                for (document_id, phrase), count in sorted(removed.items())
            ],
            "added_unbracketed": [
                {"document_id": document_id, "phrase": phrase, "count": count}
                for (document_id, phrase), count in sorted(added.items())
            ],
            "live_text_matches_previous_snapshot": True,
        }
    FULL_OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    report["full_preview_path"] = str(FULL_OUTPUT)
    report["documents"] = [
        {"id": row["id"], "actions": row["actions"]} for row in changes
    ]
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(
        f"accepted={len(accepted)} original={len(original_failing)} "
        f"updated_before={len(new_failing)} changed={len(changes)} "
        f"linked={counts['linked']} unbracketed={counts['unbracketed']} "
        f"self_links={counts['self-link removed']} left={len(left)}"
    )
    if previous:
        comparison = report["comparison_to_previous"]
        print(
            f"previous_unbracketed={comparison['previous_unbracketed_actions']} "
            f"removed={comparison['removed_unbracketed_occurrences']} "
            f"added={comparison['added_unbracketed_occurrences']}"
        )
    print(f"full_preview={FULL_OUTPUT} bytes={FULL_OUTPUT.stat().st_size}")
    print(f"wrote={OUTPUT} bytes={OUTPUT.stat().st_size}")


if __name__ == "__main__":
    main()
