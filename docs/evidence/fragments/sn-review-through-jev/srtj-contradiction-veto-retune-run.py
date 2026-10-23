#!/usr/bin/env python3
"""Replay two candidate contradiction questions against two sets, twice each.

The arms differ in one way: whether a claim that states a non-requirement of a
quantity whose name carries no locus or aggregation qualifier of its own is
settled in code before the judge is asked.

* Arm A is the base question text plus the label-only predicate, which empties a
  field label out of the evidence while keeping the source's path.
* Arm B2 is arm A plus the narrowed deterministic settlement: such a claim is
  answered "says nothing" in code and the judge is never asked about it. A
  sentence that asserts the quantity's own combination — that it is not
  averaged, holds at one place only, or excludes what the source states — is
  left to the judge.

Both arms carry the predicate and the same source-comparison criterion. No arm
is measured on the wording that states a specialisation rule to the judge, and
that wording is no longer shipped: it is not a question these counts cover.

Every arm is scored twice on the same rows in one process. The judge is not
deterministic run to run, so an arm's recall is the pair of counts, and the arm
is only kept when both passes meet the bar. Rows are matched between passes by
row id, never by position in a sorted list.

Set A is the recorded firings: every (name, claim, DD text) triple the
contradiction veto held on, read from the diagnosis
``srtj-contradiction-veto-hold.json``. Each state is rebuilt from the stored
claim and its stored DD source text, so the question is asked what the pipeline
asked it and the count that still fires is the precision of each arm.

Set B is the planted contradictions: the 60 rows of
``golden-documentation-planted.json`` labelled ``contradicts`` and the 20
labelled ``says_nothing``, joined to the golden documentation set by source
unit so each state carries the entry's own description. The golden
documentation set itself carries no contradiction-labelled row (its label is a
binary accept/reject, and its ``label_check`` values are subject and
house-style checks), so the planted rows are the whole recall set.

Every arm is asked through the same ``decide_batch`` the cascade uses under the
``sn-judge`` seat, so the arms are comparable call for call. Each arm's
questions come from the product's own ``docs_question_table``, so the wording
measured here is the wording the product ships; the source-comparison criterion
is pinned to the base revision's wording at c5b7f76ed, which is what these
counts were measured on.

Read-only: no graph write, no LLM other than the named seat.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from imas_codex.settings import get_model  # noqa: E402
from imas_codex.standard_names.review import jev_review  # noqa: E402

HOLD = HERE / "srtj-contradiction-veto-hold.json"
PLANTED = HERE / "golden-documentation-planted.json"
GOLDEN = HERE.parent / "sn-judgment-evaluation/golden-documentation-acceptable.json"
OUT = HERE / "srtj-contradiction-veto-retune.json"

CLAIM_KEY = "check:claim_contradicts_source:{index}"
CEILING_USD = 1.00
RECORDED_CEILING = 3
BASE_REVISION = "c5b7f76ed"
ARMS = ("a", "b2")
PASSES = (1, 2)
ARM_NOTES = {
    "a": "base question text plus the label-only predicate",
    "b2": (
        "arm A plus the narrowed settlement: a sentence stating a non-requirement "
        "of a quantity whose name carries no locus or aggregation qualifier of its "
        "own is answered 'says nothing' in code"
    ),
}


def claim_questions(
    checks: Any,
    claim_count: int,
    *,
    indices: tuple[int, ...] | None = None,
) -> dict[str, Any]:
    """The claim-comparison questions of one arm for the given claim indices.

    The questions come from the product's own table, so the arm is asked the
    wording the product ships rather than a copy of it. A claim question
    depends only on its index, not on the claim's words, so the table is built
    over ``claim_count`` placeholders and any state in the group can be asked
    from it.
    """
    if indices is None:
        indices = tuple(range(claim_count))
    table = jev_review.docs_question_table(checks, [""] * claim_count)
    return {
        CLAIM_KEY.format(index=index): table[CLAIM_KEY.format(index=index)]
        for index in indices
    }


async def judge(
    states: list[dict[str, Any]], *, arm: str
) -> tuple[list[dict[str, Any] | None], float]:
    """Ask one arm's question of every state through ``decide_batch``.

    For arm b2 the settled claims are removed from the question set before the
    call, so the judge is never asked about them; their answers are the settled
    ones. Arm A asks every claim.
    """
    checks = jev_review._default_checks()
    built = [jev_review.build_docs_state(state["item"]) for state in states]
    counts = [len(state["documentation_claims"]) for state in built]
    settled: list[dict[int, str]] = [{} for _ in built]
    if arm == "b2":
        for row, built_state in zip(settled, built, strict=True):
            row.update(
                jev_review.deterministic_claim_determinations(
                    built_state["documentation_claims"],
                    str(built_state.get("name_id") or ""),
                )
            )

    groups: dict[tuple[int, ...], list[int]] = {}
    for position, (row, count) in enumerate(zip(settled, counts, strict=True)):
        asked = tuple(index for index in range(count) if index not in row)
        groups.setdefault(asked, []).append(position)

    answers: list[dict[str, Any]] = [{} for _ in built]
    failed: set[int] = set()
    cost = 0.0
    for asked, positions in groups.items():
        if not asked:
            continue
        questions = claim_questions(
            checks,
            max(asked) + 1,
            indices=asked,
        )
        results, group_cost = await jev_review.decide_batch(
            [built[position] for position in positions],
            questions,
            model=get_model("sn-judge"),
            service=jev_review.SERVICE,
            concurrency=8,
        )
        cost += group_cost
        for position, result in zip(positions, results, strict=True):
            if result is None:
                failed.add(position)
            else:
                answers[position] = result[0]

    out: list[dict[str, Any] | None] = []
    for position, (state, built_state, count) in enumerate(
        zip(states, built, counts, strict=True)
    ):
        if position in failed:
            out.append(None)
            continue
        row = settled[position]
        choices = {
            CLAIM_KEY.format(index=index): (
                "says nothing"
                if index in row
                else (answers[position].get(CLAIM_KEY.format(index=index)) or {}).get(
                    "choice"
                )
            )
            for index in range(count)
        }
        fired = [
            index
            for index in range(count)
            if index not in row
            and jev_review.claim_contradicts(
                answers[position].get(CLAIM_KEY.format(index=index)) or {}
            )
        ]
        out.append(
            {
                **state,
                "claims": built_state["documentation_claims"],
                "choices": choices,
                "settled": sorted(row),
                "fired": bool(fired),
                "fired_indices": fired,
                "sources_seen": [
                    source.get("id")
                    for source in (built_state["dd_source_docs"]["value"] or [])
                ],
            }
        )
    return out, cost


def summarise(rows: list[dict[str, Any] | None]) -> dict[str, Any]:
    return {
        "rows": len(rows),
        "unscored": sum(1 for row in rows if row is None),
        "fired": sum(1 for row in rows if row and row["fired"]),
        "fired_ids": sorted(row["name"] for row in rows if row and row["fired"]),
    }


def recorded_states() -> list[dict[str, Any]]:
    """Every recorded firing, deduplicated to its (name, claim, DD text) triple."""
    payload = json.loads(HOLD.read_text())
    seen: set[tuple[str, str, str, str]] = set()
    states: list[dict[str, Any]] = []
    for firing in payload["hold_scan"]["firings"]:
        for entry in firing["entries"]:
            key = (
                firing["name"],
                entry["claim"],
                entry["source_path"],
                entry["dd_text"],
            )
            if key in seen:
                continue
            seen.add(key)
            name, claim, source_path, dd_text = key
            states.append(
                {
                    "name": name,
                    "claim": claim,
                    "source_path": source_path,
                    "dd_text": dd_text,
                    "item": {
                        "id": name,
                        "documentation": claim,
                        "dd_source_docs": [
                            {"id": source_path, "documentation": dd_text}
                        ],
                    },
                }
            )
    return states


def planted_states() -> list[dict[str, Any]]:
    """The planted rows, joined to the golden set for the entry's description."""
    copies = json.loads(PLANTED.read_text())["copies"]
    golden = {
        row["unit"]: row for row in json.loads(GOLDEN.read_text()) if row.get("unit")
    }
    states: list[dict[str, Any]] = []
    for row in copies:
        base = golden.get(row["source_unit"], {})
        states.append(
            {
                "name": row["id"],
                "claim_label": row["claim_label"],
                "rule": row["rule"],
                "planted_sentence": row["planted_sentence"],
                "item": {
                    "id": row["source_unit"],
                    "documentation": row["documentation"],
                    "description": base.get("description"),
                    "dd_source_docs": [
                        {
                            "id": row["dd_source"]["path"],
                            "documentation": row["dd_source"]["documentation"],
                        }
                    ],
                },
            }
        )
    return states


def arm_meets_bar(arm: dict[str, Any], recall_total: int, ceiling: int) -> bool:
    """True only when every pass holds both legs of the bar."""
    return all(
        pass_result["recorded_still_fire"] <= ceiling
        and pass_result["recall"] >= recall_total
        for pass_result in arm["passes"].values()
    )


def keep_arm(arms: dict[str, Any], recall_total: int, ceiling: int) -> tuple[str, str]:
    """The arm to keep: one that meets the bar, else no revision at all.

    An arm may stay only when it holds both legs on every pass. When none does,
    no revision is kept, whatever its numbers: an arm with better precision and
    lower recall than the base question is a recall regression, not a trade.
    """
    met = [arm for arm in ARMS if arms[arm]["meets_bar"]]
    if met:
        return (
            min(
                met,
                key=lambda arm: (
                    max(
                        pass_result["recorded_still_fire"]
                        for pass_result in arms[arm]["passes"].values()
                        if pass_result["recorded_still_fire"] is not None
                    ),
                    arm,
                ),
            ),
            "best arm that meets the bar on every pass",
        )
    return "", (
        "no arm meets the bar on every pass; the base question is restored with no change"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()

    recorded = recorded_states()
    planted = planted_states()
    controls = [row for row in planted if row["claim_label"] == "says_nothing"]
    contradictions = [row for row in planted if row["claim_label"] == "contradicts"]

    async def run() -> dict[str, Any]:
        spend = 0.0
        out: dict[str, Any] = {}
        for label, rows in (
            ("recorded", recorded),
            ("contradictions", contradictions),
            ("controls", controls),
        ):
            arms: dict[str, Any] = {}
            for arm in ARMS:
                passes: dict[str, Any] = {}
                for number in PASSES:
                    judged, cost = await judge(rows, arm=arm)
                    spend += cost
                    passes[str(number)] = {
                        "summary": summarise(judged),
                        "rows": list(judged),
                    }
                arms[arm] = passes
            out[label] = arms
        out["spend_usd"] = spend
        return out

    result = asyncio.run(run())

    arms_report: dict[str, Any] = {}
    for arm in ARMS:
        passes: dict[str, Any] = {}
        for number in PASSES:
            key = str(number)
            passes[key] = {
                "recorded_still_fire": result["recorded"][arm][key]["summary"]["fired"],
                "recall": result["contradictions"][arm][key]["summary"]["fired"],
                "false_alarms": result["controls"][arm][key]["summary"]["fired"],
                "unscored": sum(
                    result[label][arm][key]["summary"]["unscored"]
                    for label in ("recorded", "contradictions", "controls")
                ),
                "recall_ids": result["contradictions"][arm][key]["summary"][
                    "fired_ids"
                ],
                "recorded_ids": result["recorded"][arm][key]["summary"]["fired_ids"],
            }
        arms_report[arm] = {
            "note": ARM_NOTES[arm],
            "passes": passes,
            "meets_bar": False,
        }
        arms_report[arm]["meets_bar"] = arm_meets_bar(
            arms_report[arm], len(contradictions), RECORDED_CEILING
        )

    recall_total = len(contradictions)
    kept, because = keep_arm(arms_report, recall_total, RECORDED_CEILING)

    result["arms"] = arms_report
    result["kept_arm"] = kept
    result["kept_because"] = because
    result["bar"] = {
        "recorded_total": len(recorded),
        "recall_total": recall_total,
        "control_total": len(controls),
        "recorded_ceiling": RECORDED_CEILING,
        "passes": len(PASSES),
        "arms_meeting": [arm for arm in ARMS if arms_report[arm]["meets_bar"]],
    }
    result["spend_usd"] = round(result["spend_usd"], 6)
    result["spend_ceiling_usd"] = CEILING_USD
    result["base_revision"] = BASE_REVISION
    result["questions_from"] = "jev_review.docs_question_table"
    result["shipped_criterion"] = dict(jev_review._CLAIM_COMPARISON_CRITERIA)
    result["old_question_captured_from"] = (
        f"imas_codex/standard_names/review/jev_review.py at {BASE_REVISION}"
    )

    args.out.write_text(json.dumps(result, indent=1) + "\n")
    print(
        json.dumps(
            {
                "arms": arms_report,
                "kept_arm": kept,
                "kept_because": because,
                "bar": result["bar"],
                "spend_usd": result["spend_usd"],
            },
            indent=1,
        )
    )
    if result["spend_usd"] > CEILING_USD:
        raise RuntimeError(
            f"Jev spend USD {result['spend_usd']:.6f} exceeds USD {CEILING_USD:.2f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
