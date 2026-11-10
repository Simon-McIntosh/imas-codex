#!/usr/bin/env python
"""Append corrective retry receipts for the cohort sources reset below the cap.

The cohort rerun reset 35 terminal-failed sources through
``retry_failed_sources``. Fifteen of them stood at the compose claim cap of 5
and twenty below it. Every receipt it wrote asserted that the source had
reached the claim cap, which is untrue for the twenty. Those receipts are
append-only audit records, so this script appends one correction record per
below-cap source instead of editing them: each correction states the true
attempt count at reset, that the reset stood below the cap, and the id of the
receipt it corrects, linked by a ``CORRECTS`` relationship. No other graph
write is made; the sources and the original receipts are untouched.

Phases leave one artifact each under the run directory: ``selection``,
``originals``, ``negative-control``, ``apply`` and ``readback``.
"""

from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path

COHORT_RUN = Path(
    "/home/ITER/mcintos/.config/reckon/crew/runs/"
    "r-20261110T192543764468-sib-cohort-rerun-isn-0-11"
)
RUN = Path(
    "/home/ITER/mcintos/.config/reckon/crew/runs/"
    "r-20261110T215919887569-sib-retry-receipt-correction"
)

COHORT_REASON_MARKER = "cohort rerun under the 0.11.0 grammar"

CORRECTION_REASON = (
    "correction of the cohort-rerun receipt: the source was reset at "
    "attempt_count {attempts} of the claim cap {cap}, below the cap. The reset "
    "was warranted because the source was terminal-failed, not because it had "
    "reached the cap; the receipt this correction names asserted the source "
    "had reached the claim cap, which was untrue for this source."
)

# The only write this node makes.
_CORRECTION_WRITE = """
UNWIND $items AS item
MATCH (original:StandardNameSourceRetry {id: item.original_receipt_id})
WHERE original.source_id = item.source_id
  AND toInteger(original.previous_attempt_count) = item.attempts_at_reset
  AND item.attempts_at_reset < $cap
  AND NOT EXISTS {
    MATCH (:StandardNameSourceRetry {correction_of: original.id})
          -[:CORRECTS]->(original)
  }
CREATE (correction:StandardNameSourceRetry {id: item.correction_id})
SET correction.source_id = item.source_id,
    correction.previous_status = original.previous_status,
    correction.previous_attempt_count = item.attempts_at_reset,
    correction.previous_error = original.previous_error,
    correction.reason = item.reason,
    correction.retried_at = CORRECTION_RETRIED_AT,
    correction.correction_of = original.id,
    correction.attempt_cap = $cap,
    correction.below_attempt_cap = true
CREATE (correction)-[:CORRECTS]->(original)
RETURN correction.id AS correction_id,
       original.id AS original_id,
       item.source_id AS source_id
ORDER BY source_id
""".replace("CORRECTION_RETRIED_AT", "datetime()")


def save(name: str, payload) -> None:
    RUN.mkdir(parents=True, exist_ok=True)
    (RUN / f"{name}.json").write_text(
        json.dumps(payload, indent=2, default=str, sort_keys=True) + "\n"
    )


def load(name: str):
    return json.loads((RUN / f"{name}.json").read_text())


def _cap() -> int:
    from imas_codex.standard_names.graph_store import graph_foundation

    return graph_foundation.services_for_call()._MAX_COMPOSE_CLAIM_ATTEMPTS


def _selection_rows() -> list[dict]:
    """The 35 reset sources with the receipt each cohort-rerun write created."""
    census = json.loads((COHORT_RUN / "census.json").read_text())
    applied = json.loads((COHORT_RUN / "reset-apply.json").read_text())
    verified = json.loads((COHORT_RUN / "verify-reset.json").read_text())
    reset = sorted(census["at_cap"])
    if len(reset) != 35 or applied["retried"] != 35:
        raise SystemExit(
            f"expected 35 reset sources, read {len(reset)} and "
            f"{applied['retried']} retried"
        )
    rows = []
    for source_id in reset:
        source = census["sources"][source_id]
        attempts = int(source["attempt_count"] or 0)
        matching = [
            event
            for event in verified["receipts"][source_id]["events"]
            if int(event["previous_attempt_count"]) == attempts
            and event["previous_status"] == source["status"]
            and COHORT_REASON_MARKER in (event.get("reason") or "")
        ]
        if len(matching) != 1:
            raise SystemExit(
                f"{source_id}: expected one cohort-rerun receipt carrying the "
                f"reset state, found {len(matching)}"
            )
        rows.append(
            {
                "source_id": source_id,
                "attempts_at_reset": attempts,
                "previous_status": source["status"],
                "original_receipt_id": matching[0]["id"],
                "original_reason": matching[0]["reason"],
            }
        )
    return rows


def select() -> int:
    cap = _cap()
    rows = _selection_rows()
    below = [row for row in rows if row["attempts_at_reset"] < cap]
    at_cap = [row for row in rows if row["attempts_at_reset"] >= cap]
    selection = {
        "attempt_cap": cap,
        "reset_total": len(rows),
        "below_cap": below,
        "below_cap_count": len(below),
        "at_cap_count": len(at_cap),
        "at_cap": [row["source_id"] for row in at_cap],
        "at_cap_receipts": {
            row["source_id"]: row["original_receipt_id"] for row in at_cap
        },
        "below_cap_attempt_histogram": {
            str(attempts): sum(
                1 for row in below if row["attempts_at_reset"] == attempts
            )
            for attempts in sorted({row["attempts_at_reset"] for row in below})
        },
    }
    save("selection", selection)
    print(
        json.dumps(
            {
                "reset_total": selection["reset_total"],
                "below_cap_count": selection["below_cap_count"],
                "at_cap_count": selection["at_cap_count"],
                "histogram": selection["below_cap_attempt_histogram"],
            },
            indent=2,
        )
    )
    return 0


def originals() -> int:
    from imas_codex.graph.client import GraphClient

    selection = load("selection")
    items = selection["below_cap"]
    with GraphClient() as gc:
        rows = list(
            gc.query(
                """
                UNWIND $items AS item
                MATCH (event:StandardNameSourceRetry {id: item.original_receipt_id})
                RETURN item.source_id AS source_id,
                       event.id AS id,
                       event.source_id AS event_source_id,
                       event.previous_status AS previous_status,
                       toInteger(event.previous_attempt_count) AS previous_attempt_count,
                       event.reason AS reason,
                       toString(event.retried_at) AS retried_at,
                       COUNT { (event)<-[:CORRECTS]-(:StandardNameSourceRetry) }
                           AS corrections
                ORDER BY item.source_id
                """,
                items=items,
            )
        )
    found = {row["source_id"]: row for row in rows}
    missing = [row["source_id"] for row in items if row["source_id"] not in found]
    if missing:
        raise SystemExit(f"original receipts not found for {missing}")
    mismatched = [
        row["source_id"]
        for row in rows
        if row["event_source_id"] != row["source_id"]
        or row["previous_attempt_count"]
        != next(
            item["attempts_at_reset"]
            for item in items
            if item["source_id"] == row["source_id"]
        )
    ]
    if mismatched:
        raise SystemExit(f"live receipts disagree with the census: {mismatched}")
    save("originals", {"receipts": rows})
    print(
        json.dumps(
            {
                "read": len(rows),
                "preexisting_corrections": sum(int(row["corrections"]) for row in rows),
                "reason_wording": sorted({row["reason"] for row in rows}),
            },
            indent=2,
        )
    )
    return 0


def _items(rows: list[dict], cap: int) -> list[dict]:
    return [
        {
            "source_id": row["source_id"],
            "attempts_at_reset": int(row["attempts_at_reset"]),
            "original_receipt_id": row["original_receipt_id"],
            "correction_id": f"source-retry-correction:{uuid.uuid4()}",
            "reason": CORRECTION_REASON.format(
                attempts=row["attempts_at_reset"], cap=cap
            ),
        }
        for row in rows
    ]


def negative() -> int:
    """Attempt the correction write for a source that was reset at the cap.

    The declared control is that such a source must receive no correction
    record, so the statement that writes twenty rows in ``apply`` must write
    none here. The apply phase's twenty rows are the positive control that the
    statement does write when its predicate admits the item.
    """
    from imas_codex.graph.client import GraphClient

    selection = load("selection")
    cap = int(selection["attempt_cap"])
    source_id = selection["at_cap"][0]
    rows = _selection_rows()
    row = next(row for row in rows if row["source_id"] == source_id)
    with GraphClient() as gc:
        written = list(gc.query(_CORRECTION_WRITE, items=_items([row], cap), cap=cap))
    saved = {
        "attempted_source": source_id,
        "attempts_at_reset": row["attempts_at_reset"],
        "attempt_cap": cap,
        "rows_written": len(written),
        "refused": not written,
    }
    save("negative-control", saved)
    print(json.dumps(saved, indent=2))
    return 0 if saved["refused"] else 1


def apply(*, dry_run: bool) -> int:
    from imas_codex.graph.client import GraphClient

    selection = load("selection")
    cap = int(selection["attempt_cap"])
    items = _items(selection["below_cap"], cap)
    if dry_run:
        save("apply-dry", {"requested": len(items), "dry_run": True})
        print(json.dumps({"requested": len(items), "dry_run": True}, indent=2))
        return 0
    with GraphClient() as gc:
        rows = list(gc.query(_CORRECTION_WRITE, items=items, cap=cap))
    save("apply", {"requested": len(items), "written": rows})
    print(json.dumps({"requested": len(items), "written": len(rows)}, indent=2))
    return 0 if len(rows) == len(items) else 1


def readback() -> int:
    from imas_codex.graph.client import GraphClient

    selection = load("selection")
    cap = int(selection["attempt_cap"])
    ids = [row["source_id"] for row in selection["below_cap"]]
    with GraphClient() as gc:
        rows = list(
            gc.query(
                """
                UNWIND $source_ids AS source_id
                MATCH (correction:StandardNameSourceRetry {source_id: source_id})
                WHERE correction.correction_of IS NOT NULL
                MATCH (correction)-[:CORRECTS]->(original:StandardNameSourceRetry)
                RETURN source_id,
                       correction.id AS correction_id,
                       toInteger(correction.previous_attempt_count) AS attempts,
                       original.id AS original_id,
                       toInteger(original.previous_attempt_count) AS original_attempts,
                       original.reason AS original_reason,
                       correction.reason AS correction_reason,
                       COUNT { (correction)-[:CORRECTS]->(:StandardNameSourceRetry) }
                           AS corrected_targets
                ORDER BY source_id
                """,
                source_ids=ids,
            )
        )
        total = int(
            list(
                gc.query(
                    """
                    MATCH (correction:StandardNameSourceRetry)
                    WHERE correction.correction_of IS NOT NULL
                    RETURN count(correction) AS total
                    """
                )
            )[0]["total"]
        )
    problems = []
    if len(rows) != 20:
        problems.append(f"read {len(rows)} correction records, expected 20")
    if total != 20:
        problems.append(f"graph holds {total} correction records, expected 20")
    for row in rows:
        if int(row["attempts"]) >= cap:
            problems.append(
                f"{row['source_id']}: attempt {row['attempts']} is not below the cap"
            )
        if row["corrected_targets"] != 1:
            problems.append(
                f"{row['source_id']}: correction is not linked to one receipt"
            )
        if row["original_attempts"] != row["attempts"]:
            problems.append(
                f"{row['source_id']}: correction states {row['attempts']}, "
                f"original states {row['original_attempts']}"
            )
    saved = {
        "correction_count": len(rows),
        "graph_correction_total": total,
        "attempt_cap": cap,
        "all_below_cap": all(int(row["attempts"]) < cap for row in rows),
        "linked": all(row["corrected_targets"] == 1 for row in rows),
        "records": rows,
        "problems": problems,
    }
    save("readback", saved)
    print(
        json.dumps(
            {
                key: saved[key]
                for key in (
                    "correction_count",
                    "graph_correction_total",
                    "all_below_cap",
                    "linked",
                    "problems",
                )
            },
            indent=2,
        )
    )
    return 0 if not problems else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "phase", choices=("select", "originals", "negative", "apply", "readback")
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.phase == "select":
        return select()
    if args.phase == "originals":
        return originals()
    if args.phase == "negative":
        return negative()
    if args.phase == "apply":
        return apply(dry_run=args.dry_run)
    return readback()


if __name__ == "__main__":
    raise SystemExit(main())
