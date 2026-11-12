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
    confirm_candidates,
    name_closure,
    pair_of,
    probe_check,
    probe_definitions,
)
from imas_codex.settings import get_model  # noqa: E402
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
ANSWERS = Path("/home/ITER/mcintos/.config/reckon/crew/runs/"
              "r-20261112T185953916209-sdbc-name-definition-probes/"
              "name-definition-probes-results-full.json")
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
    answers = json.loads(ANSWERS.read_text())["answers"]
    seat = json.loads(SEAT.read_text())
    seat_flagged = {
        name for name, verdict in seat["verdicts"].items() if verdict.get("verdict") == "flag"
    }

    # Every candidate the closure leaves, then one closed confirming question.
    raw_findings = {
        row["name"]: probe_check(row, answers.get(row["name"], {})) for row in rows
    }
    raw_findings = {name: found for name, found in raw_findings.items() if found}
    confirmed, confirm_cost, relation_counts = asyncio.run(
        confirm_candidates(rows, raw_findings, model=get_model("sn-judge"))
    )

    # The (slot, option) precision table, built on the development split only.
    dev_names = {row["name"] for row in dev}
    label = {row["name"]: row["label"] for row in rows}
    pair_table: dict[str, dict] = {}
    for name in dev_names:
        for finding in confirmed.get(name, []):
            entry = pair_table.setdefault(
                pair_of(finding), {"mismatched": 0, "agreeing": 0, "names": []}
            )
            entry["mismatched" if label[name] == "mismatch" else "agreeing"] += 1
            entry["names"].append(name)
    dropped: list[str] = []
    for pair, entry in pair_table.items():
        entry["precision"] = (
            round(entry["mismatched"] / (entry["mismatched"] + entry["agreeing"]), 3)
            if entry["mismatched"] + entry["agreeing"]
            else None
        )
        # A pair whose development precision is at or below chance is dropped:
        # it flags agreeing names at least as often as mismatched ones.
        if entry["agreeing"] >= entry["mismatched"]:
            dropped.append(pair)
    dropped_set = set(dropped)

    probe_flags: dict[str, list[dict]] = {}
    for name, found in confirmed.items():
        keep = [finding for finding in found if pair_of(finding) not in dropped_set]
        if keep:
            probe_flags[name] = keep
    probe_flagged = set(probe_flags)
    union = probe_flagged | seat_flagged

    def slot_flags(slot_key: str) -> set[str]:
        return {
            name
            for name, found in probe_flags.items()
            if any(finding["slot"] == slot_key for finding in found)
        }

    def score_both(flagged: set[str]) -> dict:
        return {"held_out": score(held, flagged), "all_agreed": score(rows, flagged)}

    by_slot = {slot.key: score_both(slot_flags(slot.key)) for slot in SLOTS}
    slot_counts = {
        slot.key: sum(1 for found in probe_flags.values()
                      if any(f["slot"] == slot.key for f in found))
        for slot in SLOTS
    }

    result = {
        "schema": "name-definition-probes",
        "check": {
            "id": "definition_slot_disagrees_with_name",
            "implementation": "imas_codex.standard_names.review.name_definition_probes.probe_check",
            "reads": "description, documentation, name, unit",
            "calls_a_model": True,
            "joined_finding": "name_definition_probe_disagreement",
        },
        "labels": {
            "path": "docs/evidence/fragments/sn-name-definition-second-reading/two-reader-labels.json",
            "rule": "only names both readers agree on score: both-flag is a mismatch, both-pass agrees",
            "produced_by": "two independent readers; no probe, seat or pipeline judge produced a label",
            "agreed_names": len(rows),
        },
        "held_out_population": {
            "names": len(held),
            "mismatched": sum(1 for row in held if row["label"] == "mismatch"),
            "agreeing": sum(1 for row in held if row["label"] == "agrees"),
            "note": "every arm is scored on this one set, the same 208 names",
        },
        "split": {
            "seed": SPLIT_SEED,
            "development_names": len(dev),
            "held_out_names": len(held),
        },
        "cost_usd": {
            "probe_extraction_chat_calls": 5.3055,
            "jev_confirmation_judgments": round(confirm_cost, 4),
            "chat_seat_calls": seat["cost_usd"],
            "per_name": round((5.3055 + confirm_cost) / len(rows), 4),
        },
        "confirming_relations": dict(sorted(relation_counts.items())),
        "pair_table": {
            "tuned_on": "development split only",
            "rule": "a (slot, option) pair is dropped when its development precision is at or below chance, i.e. it flags agreeing names at least as often as mismatched ones",
            "pairs": pair_table,
            "dropped": sorted(dropped),
        },
        "arms": {
            "probes": score_both(probe_flagged),
            "seat": score_both(seat_flagged),
            "union": score_both(union),
        },
        "by_slot": by_slot,
        "slot_counts": slot_counts,
        "probe_flagged_names": sorted(probe_flagged),
    }
    gate = result["arms"]["probes"]["held_out"]
    result["gate"] = {
        "recall_floor": RECALL_FLOOR,
        "false_flag_ceiling": FALSE_FLAG_CEILING,
        "met": bool((gate["recall"] or 0.0) >= RECALL_FLOOR
                    and (gate["false_flag_rate"] or 1.0) <= FALSE_FLAG_CEILING),
    }
    (FRAGMENTS / "name-definition-probes-results.json").write_text(
        json.dumps(result, indent=1, sort_keys=True) + "\n"
    )
    print(json.dumps({k: result[k] for k in ("arms", "gate", "by_slot",
                                             "cost_usd", "confirming_relations",
                                             "held_out_population")}, indent=1)[:5000])


if __name__ == "__main__":
    main()