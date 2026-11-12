"""Run the name-against-definition probes over the labelled names.

Two arms: the prototype (the names the chat seat missed and the names it
wrongly flagged, three slots) and the scored arm (every agreed name, every
slot). Labels come from the second reading's two-reader agreement; the split
is the one the earlier deterministic check froze, so probe wording and
thresholds are tuned on the development third only.

Usage: ``name-definition-probes-run.py prototype`` or ``... full``.
"""

from __future__ import annotations

import asyncio
import collections
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from imas_codex.standard_names.review.name_definition_probes import (  # noqa: E402
    SLOTS,
    probe_check,
    probe_definitions,
)
from imas_codex.standard_names.review.name_definition_seat import (  # noqa: E402
    review_model,
)

FRAGMENTS = Path(__file__).resolve().parent
LABELS = FRAGMENTS / "name-definition-labels.json"
CONSENSUS = (
    Path("/home/ITER/mcintos/Code/imas-codex")
    / "docs/evidence/fragments/sn-name-definition-second-reading/two-reader-labels.json"
)
SEAT = FRAGMENTS / "name-definition-seat-results.json"
CHECK = FRAGMENTS / "name-definition-check-results.json"

SPLIT_SEED = 7714
DEV_SHARE = 1 / 3

PROTOTYPE_SLOTS = ("aggregation", "restriction", "counting_unit")

RECALL_FLOOR = 0.80
FALSE_FLAG_CEILING = 0.10


def load_rows() -> list[dict]:
    labelled = json.loads(LABELS.read_text())["rows"]
    by_name = {row["name"]: row for row in labelled}
    consensus = json.loads(CONSENSUS.read_text())["rows"]
    agreed: list[dict] = []
    for entry in consensus:
        name = entry["name"]
        both_flag = entry["first"] != "agrees" and entry["second"] != "agrees"
        both_pass = entry["first"] == "agrees" and entry["second"] == "agrees"
        if not (both_flag or both_pass):
            continue
        row = dict(by_name[name])
        row["label"] = "mismatch" if both_flag else "agrees"
        row["ref"] = name
        agreed.append(row)
    return agreed


def split(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    groups: dict[str, list[dict]] = collections.defaultdict(list)
    for row in rows:
        groups[row["class_under_ruling"]].append(row)
    dev: list[dict] = []
    held: list[dict] = []
    for group in groups.values():
        ordered = sorted(
            group,
            key=lambda row: hashlib.sha256((str(SPLIT_SEED) + row["name"]).encode()).hexdigest(),
        )
        cut = max(1, round(len(ordered) * DEV_SHARE))
        if group[0]["class_under_ruling"] == "agrees":
            cut = round(len(ordered) * DEV_SHARE)
        dev.extend(ordered[:cut])
        held.extend(ordered[cut:])
    return dev, held


def score(rows: list[dict], flagged: set[str]) -> dict:
    mismatched = [row for row in rows if row["label"] == "mismatch"]
    agreeing = [row for row in rows if row["label"] == "agrees"]
    found = [row["name"] for row in mismatched if row["name"] in flagged]
    false_flags = [row["name"] for row in agreeing if row["name"] in flagged]
    return {
        "names": len(rows),
        "mismatched": len(mismatched),
        "mismatched_found": len(found),
        "recall": round(len(found) / len(mismatched), 3) if mismatched else None,
        "agreeing": len(agreeing),
        "agreeing_flagged": len(false_flags),
        "false_flag_rate": round(len(false_flags) / len(agreeing), 3) if agreeing else None,
        "mismatched_missed": [row["name"] for row in mismatched if row["name"] not in flagged],
        "false_flags": false_flags,
    }


async def run(rows: list[dict], slots, label: str) -> tuple[dict, float]:
    items = [{**row, "id": row["name"]} for row in rows]
    answers, cost = await probe_definitions(items, slots=slots, model=review_model())
    findings = {
        row["name"]: probe_check(row, answers.get(row["name"], {}), slots=slots)
        for row in rows
    }
    by_slot = collections.Counter(
        finding["slot"] for found in findings.values() for finding in found
    )
    out = {
        "arm": label,
        "slots": [slot.key for slot in slots],
        "cost_usd": round(cost, 4),
        "names": len(rows),
        "findings": {name: found for name, found in findings.items() if found},
        "slot_counts": dict(sorted(by_slot.items())),
        "answers": answers,
    }
    return out, cost


def main() -> None:
    which = sys.argv[1] if len(sys.argv) > 1 else "prototype"
    rows = load_rows()
    if which == "prototype":
        seat = json.loads(SEAT.read_text())
        wanted = set(seat["missed"]) | set(seat["false_flags"])
        proto = [row for row in rows if row["name"] in wanted]
        slot_set = tuple(slot for slot in SLOTS if slot.key in PROTOTYPE_SLOTS)
        result, _ = asyncio.run(run(proto, slot_set, "prototype"))
        result["prototype_names"] = {
            "missed_by_seat": sorted(wanted & set(seat["missed"])),
            "wrongly_flagged_by_seat": sorted(wanted & set(seat["false_flags"])),
        }
        result["flagged_by_probe"] = {
            name: [f["slot"] for f in found]
            for name, found in result["findings"].items()
        }
        (FRAGMENTS / "name-definition-probes-prototype.json").write_text(
            json.dumps(result, indent=1, sort_keys=True) + "\n"
        )
        print(json.dumps({k: v for k, v in result.items() if k != "answers"}, indent=1)[:4000])
        return

    dev, held = split(rows)
    proto_json = FRAGMENTS / "name-definition-probes-prototype.json"
    seen = set()
    if proto_json.exists():
        prototype = json.loads(proto_json.read_text())
        seen = set(prototype.get("answers", {}))
    items = [{**row, "id": row["name"]} for row in rows]
    answers, cost = asyncio.run(probe_definitions(items, slots=SLOTS, model=review_model()))

    def flagged_from(by_name_answers) -> set[str]:
        return {
            row["name"]
            for row in rows
            if probe_check(row, by_name_answers.get(row["name"], {}))
        }

    without_conventions = {
        row["name"]
        for row in rows
        if probe_check(
            row, answers.get(row["name"], {}), apply_conventions=False
        )
    }
    probe_flagged = flagged_from(answers)
    seat = json.loads(SEAT.read_text())["verdicts"]
    seat_flagged = {
        name for name, verdict in seat.items() if verdict.get("verdict") == "flag"
    }
    union = probe_flagged | seat_flagged

    held_names = {row["name"] for row in held}
    seen_held = held_names & seen
    held_scored = [row for row in held if row["name"] not in seen_held]
    dev_scored = [row for row in dev if row["name"] not in seen]

    by_slot: dict[str, dict] = {}
    for slot in SLOTS:
        def slot_flagged(source: dict[str, dict]) -> set[str]:
            return {
                row["name"]
                for row in rows
                if any(
                    finding["slot"] == slot.key
                    for finding in probe_check(row, source.get(row["name"], {}))
                )
            }

        flagged = slot_flagged(answers)
        by_slot[slot.key] = {
            "held_out": score(held_scored, flagged),
            "all_agreed": score(rows, flagged),
        }

    result = {
        "schema": "name-definition-probes",
        "check": {
            "id": "definition_qualifier_absent_from_name_probes",
            "implementation": "imas_codex.standard_names.review.name_definition_probes.probe_check",
            "reads": "description, documentation, name",
            "calls_a_model": True,
            "joined_finding": "name_definition_disagreement",
        },
        "labels": {
            "path": "docs/evidence/fragments/sn-name-definition-second-reading/two-reader-labels.json",
            "rule": "only names both readers agree on score: both-flag is a mismatch, both-pass agrees",
            "produced_by": "two independent readers; no probe, seat or pipeline judge produced a label",
            "agreed_names": len(rows),
        },
        "split": {
            "seed": SPLIT_SEED,
            "development_names": len(dev),
            "held_out_names": len(held),
            "prototype_names_seen_before_scoring": sorted(seen),
            "held_out_scored": len(held_scored),
            "development_scored": len(dev_scored),
        },
        "cost_usd": round(cost, 4),
        "arms": {
            "probes": {"held_out": score(held_scored, probe_flagged), "all_agreed": score(rows, probe_flagged)},
            "seat": {"held_out": score(held_scored, seat_flagged), "all_agreed": score(rows, seat_flagged)},
            "union": {"held_out": score(held_scored, union), "all_agreed": score(rows, union)},
            "probes_without_conventions": score(held_scored, without_conventions),
        },
        "by_slot": by_slot,
        "slot_counts": dict(sorted(collections.Counter(
            finding["slot"]
            for row in rows
            for finding in probe_check(row, answers.get(row["name"], {}))
        ).items())),
        "probe_flagged_names": sorted(probe_flagged),
        "answers": answers,
    }
    gate = result["arms"]["probes"]["held_out"]
    result["gate"] = {
        "recall_floor": RECALL_FLOOR,
        "false_flag_ceiling": FALSE_FLAG_CEILING,
        "met": bool(
            (gate["recall"] or 0.0) >= RECALL_FLOOR
            and (gate["false_flag_rate"] or 1.0) <= FALSE_FLAG_CEILING
        ),
    }
    (FRAGMENTS / "name-definition-probes-results.json").write_text(
        json.dumps(result, indent=1, sort_keys=True) + "\n"
    )
    print(json.dumps({k: result[k] for k in ("arms", "gate", "by_slot", "cost_usd")}, indent=1)[:5000])


if __name__ == "__main__":
    main()