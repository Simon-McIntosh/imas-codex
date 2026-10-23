#!/usr/bin/env python3
"""Replay the retuned documentation contradiction question against two sets.

Set A is the recorded firings: every (name, claim, DD text) triple the
contradiction veto held on, read from the diagnosis
``srtj-contradiction-veto-hold.json``. Each state is rebuilt from the stored
claim and its stored DD source text, so the question is asked exactly what the
pipeline asked it and the count that still fires is the precision of the
retune.

Set B is the planted contradictions: the 60 rows of
``golden-documentation-planted.json`` labelled ``contradicts`` and the 20
labelled ``says_nothing``, joined to the golden documentation set by source
unit so each state carries the entry's own description. The golden
documentation set itself carries no contradiction-labelled row (its label is a
binary accept/reject, and its ``label_check`` values are subject and
house-style checks), so the planted rows are the whole recall set.

Both sets are asked the old question and the new one in one process through the
same ``decide_batch`` the cascade uses under the ``sn-judge`` seat. The old arm
restores the base revision's question text and its unfiltered evidence, both
captured from revision c5b7f76ed.

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
CLAIM_PREFIX = "check:claim_contradicts_source:"
CEILING_USD = 1.00
BASE_REVISION = "c5b7f76ed"

# The base revision's source-comparison question, verbatim from
# imas_codex/standard_names/review/jev_review.py at c5b7f76ed (lines 598-615).
OLD_INSTRUCTIONS = (
    "{question} Compare only `documentation_claims[{index}]` with "
    "`dd_source_docs.value`, `description`, `unit.value`, `links.value`, "
    "`source_paths.value`, `sibling_names.value` and "
    "`related_neighbours.value`. A contradiction in any part of the sentence "
    "is a contradiction even when another part is supported. If no observed "
    "source bears on the claim, choose 'says nothing'."
)
OLD_CRITERIA = {
    "supports": "The DD evidence directly supports this specific claim.",
    "contradicts": (
        "Any part of this claim conflicts with an observed source, unit, link "
        "target, sibling meaning or source path."
    ),
    "says nothing": (
        "The observed evidence neither supports nor contradicts this claim."
    ),
}


def claim_questions(
    checks: Any, claims: list[str], *, old: bool, index_floor: int
) -> dict[str, Any]:
    """The claim-comparison questions of one arm, for ``index_floor`` claims.

    The new arm is the module's own table, so the measured question is the
    shipped one. The old arm rebuilds the base revision's wording.
    """
    if not old:
        return {
            key: question
            for key, question in jev_review.docs_question_table(checks, claims).items()
            if key.startswith(CLAIM_PREFIX)
        }
    claim_check = next(
        check
        for check in jev_review._docs_checks(checks)
        if check.id == "claim_contradicts_source"
    )
    options = claim_check.question.options
    return {
        CLAIM_KEY.format(index=index): {
            "type": "choice",
            "instructions": OLD_INSTRUCTIONS.format(
                question=claim_check.question.text, index=index
            ),
            "criteria": {option: OLD_CRITERIA[option] for option in options},
        }
        for index in range(index_floor)
    }


def _state(item: dict[str, Any], *, old: bool) -> dict[str, Any]:
    """One claim's state, with the base revision's evidence on the old arm."""
    state = jev_review.build_docs_state(item)
    if old:
        state["dd_source_docs"] = jev_review._observed(item.get("dd_source_docs"))
    return state


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


async def judge(
    states: list[dict[str, Any]], *, old: bool
) -> tuple[list[dict[str, Any] | None], float]:
    """Ask one arm's question of every state through ``decide_batch``."""
    checks = jev_review._default_checks()
    built = [_state(state["item"], old=old) for state in states]
    index_floor = (
        max((len(state["documentation_claims"]) for state in built), default=1) or 1
    )
    questions = claim_questions(
        checks, built[0]["documentation_claims"], old=old, index_floor=index_floor
    )
    results, cost = await jev_review.decide_batch(
        built,
        questions,
        model=get_model("sn-judge"),
        service=jev_review.SERVICE,
        concurrency=8,
    )
    out: list[dict[str, Any] | None] = []
    for state, built_state, result in zip(states, built, results, strict=True):
        if result is None:
            out.append(None)
            continue
        answers, _spent = result
        claim_count = len(built_state["documentation_claims"])
        fired = [
            index
            for index in range(claim_count)
            if jev_review.claim_contradicts(
                answers.get(CLAIM_KEY.format(index=index)) or {}
            )
        ]
        out.append(
            {
                **state,
                "claims": built_state["documentation_claims"],
                "choices": {
                    CLAIM_KEY.format(index=index): (
                        answers.get(CLAIM_KEY.format(index=index)) or {}
                    ).get("choice")
                    for index in range(claim_count)
                },
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
    fired = sum(1 for row in rows if row and row["fired"])
    return {
        "rows": len(rows),
        "unscored": sum(1 for row in rows if row is None),
        "fired": fired,
    }


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
            for arm in ("old", "new"):
                judged, cost = await judge(rows, old=(arm == "old"))
                spend += cost
                arms[arm] = {
                    "summary": summarise(judged),
                    "rows": list(judged),
                }
            out[label] = arms
        out["spend_usd"] = spend
        return out

    result = asyncio.run(run())

    recorded_old = result["recorded"]["old"]["summary"]["fired"]
    recorded_new = result["recorded"]["new"]["summary"]["fired"]
    recall_old = result["contradictions"]["old"]["summary"]["fired"]
    recall_new = result["contradictions"]["new"]["summary"]["fired"]
    control_old = result["controls"]["old"]["summary"]["fired"]
    control_new = result["controls"]["new"]["summary"]["fired"]

    bar = {
        "recorded_total": len(recorded),
        "recorded_still_fire_old": recorded_old,
        "recorded_still_fire_new": recorded_new,
        "at_most_three_recorded_fire": recorded_new <= 3,
        "recall_total": len(contradictions),
        "recall_old": recall_old,
        "recall_new": recall_new,
        "recall_not_lower": recall_new >= recall_old,
        "met": recorded_new <= 3 and recall_new >= recall_old,
    }
    result["bar"] = bar
    result["spend_usd"] = round(result["spend_usd"], 6)
    result["base_revision"] = BASE_REVISION
    result["old_question_captured_from"] = (
        f"imas_codex/standard_names/review/jev_review.py at {BASE_REVISION}"
    )
    result["false_alarms"] = {"old": control_old, "new": control_new}
    result["spend_ceiling_usd"] = CEILING_USD

    args.out.write_text(json.dumps(result, indent=1) + "\n")
    print(
        json.dumps(
            {
                "bar": bar,
                "false_alarms": result["false_alarms"],
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
