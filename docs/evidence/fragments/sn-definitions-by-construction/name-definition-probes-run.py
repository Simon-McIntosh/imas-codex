"""Run the name-against-definition probes over the labelled names.

Two arms: the prototype (the names the chat seat missed and the names it
wrongly flagged, three slots) and the scored arm (every agreed name, every
slot). Labels come from the second reading's two-reader agreement; the split
is the one the earlier deterministic check froze, so probe wording and
thresholds are tuned on the development third only.

Usage: ``name-definition-probes-run.py prototype`` or ``... final``.
"""

from __future__ import annotations

import asyncio
import collections
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from imas_codex.settings import get_model  # noqa: E402
from imas_codex.standard_names.review.name_definition_probes import (  # noqa: E402
    SLOTS,
    confirm_candidates,
    pair_of,
    probe_check,
    probe_definitions,
    probe_definitions_jev,
)
from imas_codex.standard_names.review.name_definition_seat import (  # noqa: E402
    review_model,
)

FRAGMENTS = Path(__file__).resolve().parent
RUN_DIR = Path(
    "/home/ITER/mcintos/.config/reckon/crew/runs/"
    "r-20261112T185953916209-sdbc-name-definition-probes"
)
LABELS = FRAGMENTS / "name-definition-labels.json"
CONSENSUS = (
    Path("/home/ITER/mcintos/Code/imas-codex")
    / "docs/evidence/fragments/sn-name-definition-second-reading/two-reader-labels.json"
)
SEAT = FRAGMENTS / "name-definition-seat-results.json"
ANSWERS = RUN_DIR / "name-definition-probes-results-full.json"
JEV_ANSWERS = RUN_DIR / "name-definition-probes-jev-answers.json"
CONFIRMATIONS = RUN_DIR / "name-definition-probes-confirmations.json"

SPLIT_SEED = 7714
DEV_SHARE = 1 / 3

PROTOTYPE_SLOTS = ("aggregation", "restriction", "counting_unit")

RECALL_FLOOR = 0.80
FALSE_FLAG_CEILING = 0.10
THRESHOLD_GRID = tuple(round(0.05 * step, 2) for step in range(1, 20)) + (1.0, 1.01)


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
            key=lambda row: hashlib.sha256(
                (str(SPLIT_SEED) + row["name"]).encode()
            ).hexdigest(),
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
        "mismatched_missed": [
            row["name"] for row in mismatched if row["name"] not in flagged
        ],
        "false_flags": false_flags,
    }


def rate(scored: dict) -> float | None:
    if not scored["agreeing"]:
        return None
    return round(scored["agreeing_flagged"] / scored["agreeing"], 3)


def tune_slot(dev_findings: list[dict], dev_labels: dict[str, str]) -> dict:
    """The lowest threshold whose development false-flag rate is at or under the ceiling."""

    curve: list[dict] = []
    chosen: float | None = None
    for threshold in THRESHOLD_GRID:
        kept = [finding for finding in dev_findings if finding["p_flag"] >= threshold]
        found = {finding["name"] for finding in kept}
        mismatched = {name for name, label in dev_labels.items() if label == "mismatch"}
        agreeing = {name for name, label in dev_labels.items() if label == "agrees"}
        hits = found & mismatched
        false_flags = found & agreeing
        false_rate = len(false_flags) / len(agreeing) if agreeing else 0.0
        curve.append(
            {
                "threshold": threshold,
                "found": len(hits),
                "recall": round(len(hits) / len(mismatched), 3) if mismatched else None,
                "false_flags": len(false_flags),
                "false_flag_rate": round(false_rate, 3),
            }
        )
        if chosen is None and false_rate <= FALSE_FLAG_CEILING and hits:
            chosen = threshold
    if chosen is None:
        chosen = THRESHOLD_GRID[-1]
    return {"threshold": chosen, "curve": curve}


def tune_joint(
    dev_findings: list[dict],
    dev_labels: dict[str, str],
    slot_keys: list[str],
    base_flags: set[str] | None = None,
) -> dict:
    """Raise the slot thresholds together until the arm's development rate holds.

    A per-slot threshold that keeps its own slot under the ceiling is not
    enough: a name flagged by any one of five slots is a false flag for the
    arm, so five slots at six percent each clear a third of the agreeing
    names. The thresholds are therefore raised one grid step at a time,
    worst-contributing slot first, until the union is at or under the ceiling.
    """

    mismatched = {name for name, label in dev_labels.items() if label == "mismatch"}
    agreeing = {name for name, label in dev_labels.items() if label == "agrees"}
    base = set(base_flags or set()) & set(dev_labels)
    level = dict.fromkeys(slot_keys, 0)
    trace: list[dict] = []

    def union() -> tuple[set[str], set[str]]:
        found: set[str] = set(base)
        for finding in dev_findings:
            if finding["p_flag"] >= THRESHOLD_GRID[level[finding["slot"]]]:
                found.add(finding["name"])
        return found & mismatched, found & agreeing

    hits, false = union()
    while len(false) / len(agreeing) > FALSE_FLAG_CEILING:
        contributions = {
            key: len(
                {
                    finding["name"]
                    for finding in dev_findings
                    if finding["slot"] == key
                    and finding["p_flag"] >= THRESHOLD_GRID[level[key]]
                    and finding["name"] in false
                }
            )
            for key in slot_keys
        }
        movable = [key for key in slot_keys if level[key] < len(THRESHOLD_GRID) - 1]
        if not movable:
            break
        worst = max(movable, key=lambda key: (contributions[key], -level[key]))
        if contributions[worst] == 0:
            worst = max(movable, key=lambda key: -level[key])
        level[worst] += 1
        hits, false = union()
        trace.append(
            {
                "slot": worst,
                "threshold": THRESHOLD_GRID[level[worst]],
                "development_found": len(hits),
                "development_false_flags": len(false),
                "development_false_flag_rate": round(len(false) / len(agreeing), 3),
            }
        )
    return {
        "thresholds": {key: THRESHOLD_GRID[level[key]] for key in slot_keys},
        "development_found": len(hits),
        "development_recall": round(len(hits) / len(mismatched), 3),
        "development_false_flags": len(false),
        "development_false_flag_rate": round(len(false) / len(agreeing), 3),
        "trace": trace,
    }


def tune_pairs(
    dev_findings: list[dict],
    dev_labels: dict[str, str],
    thresholds: dict[str, float],
    base_flags: set[str] | None = None,
) -> tuple[list[str], list[str], list[dict]]:
    """Drop the (slot, option) pairs that flag agreeing names as often as mismatched ones.

    A pair whose development precision is at or below chance is dropped, even
    when its finding clears the slot threshold. A dropped pair is then
    re-admitted when it recovers a development mismatch that no other admitted
    finding recovers and admitting it keeps the development false-flag rate at
    or under the ceiling. Pairs are considered best precision first.
    """

    mismatched = {name for name, label in dev_labels.items() if label == "mismatch"}
    agreeing = {name for name, label in dev_labels.items() if label == "agrees"}
    dev_flags = set(dev_labels)
    table: dict[str, dict] = {}
    for finding in dev_findings:
        entry = table.setdefault(
            pair_of(finding), {"mismatched": set(), "agreeing": set()}
        )
        entry[
            "mismatched" if dev_labels[finding["name"]] == "mismatch" else "agreeing"
        ].add(finding["name"])
    for entry in table.values():
        hits = entry["mismatched"]
        entry["precision"] = (
            round(len(hits) / (len(hits) + len(entry["agreeing"])), 3)
            if (hits or entry["agreeing"])
            else None
        )
    ordered = sorted(
        table.items(), key=lambda item: (-(item[1]["precision"] or 0.0), item[0])
    )
    dropped = [
        pair
        for pair, entry in table.items()
        if len(entry["agreeing"]) >= len(entry["mismatched"])
    ]
    base = set(base_flags or set()) & set(dev_flags)
    admitted: list[str] = []
    admitted_found: set[str] = set(base & mismatched)
    admitted_false: set[str] = set(base & agreeing)
    for pair, entry in ordered:
        if pair not in dropped:
            continue
        if not (entry["mismatched"] - admitted_found):
            continue
        false = admitted_false | entry["agreeing"]
        if len(false) / len(agreeing) > FALSE_FLAG_CEILING:
            continue
        admitted.append(pair)
        admitted_found |= entry["mismatched"]
        admitted_false |= entry["agreeing"]
    rows = [
        {
            "pair": pair,
            "development_precision": entry["precision"],
            "dropped": pair in dropped,
            "readmitted": pair in admitted,
            "development_mismatched": sorted(entry["mismatched"]),
            "development_agreeing": sorted(entry["agreeing"]),
        }
        for pair, entry in ordered
    ]
    return dropped, admitted, rows


def keep(
    graded: dict[str, list[dict]],
    thresholds: dict[str, float],
    dropped: list[str],
    admitted: list[str],
) -> dict[str, list[dict]]:
    dropped_set = set(dropped)
    admitted_set = set(admitted)
    kept: dict[str, list[dict]] = {}
    for name, found in graded.items():
        survivors = [
            finding
            for finding in found
            if (
                finding["p_flag"] >= thresholds[finding["slot"]]
                and pair_of(finding) not in dropped_set
            )
            or pair_of(finding) in admitted_set
        ]
        if survivors:
            kept[name] = survivors
    return kept


async def graded_findings(
    rows: list[dict], answers: dict, label: str
) -> tuple[dict[str, list[dict]], dict[str, int], float]:
    cache: dict = {}
    if CONFIRMATIONS.exists():
        cache = json.loads(CONFIRMATIONS.read_text())
    if label in cache:
        return (
            cache[label]["graded"],
            cache[label]["relations"],
            cache[label]["cost_usd"],
        )
    raw = {row["name"]: probe_check(row, answers.get(row["name"], {})) for row in rows}
    raw = {name: found for name, found in raw.items() if found}
    graded, cost, relations = await confirm_candidates(
        rows, raw, model=get_model("sn-judge")
    )
    cache[label] = {
        "graded": graded,
        "relations": relations,
        "cost_usd": round(cost, 4),
        "candidates": sum(len(found) for found in raw.values()),
    }
    CONFIRMATIONS.write_text(json.dumps(cache, indent=1, sort_keys=True) + "\n")
    return graded, relations, round(cost, 4)


def flat(graded: dict[str, list[dict]], names: set[str]) -> list[dict]:
    return [
        {**finding, "name": name}
        for name, found in graded.items()
        if name in names
        for finding in found
    ]


async def final() -> None:
    rows = load_rows()
    dev, held = split(rows)
    dev_names = {row["name"] for row in dev}
    dev_labels = {row["name"]: row["label"] for row in rows if row["name"] in dev_names}
    items = [{**row, "id": row["name"]} for row in rows]

    chat_answers = json.loads(ANSWERS.read_text())["answers"]
    if JEV_ANSWERS.exists():
        jev_answers = json.loads(JEV_ANSWERS.read_text())["answers"]
        jev_extraction_cost = json.loads(JEV_ANSWERS.read_text()).get("cost_usd", 0.0)
    else:
        jev_answers, jev_extraction_cost = await probe_definitions_jev(
            items, model=get_model("sn-judge"), concurrency=10
        )
        JEV_ANSWERS.write_text(
            json.dumps(
                {"answers": jev_answers, "cost_usd": round(jev_extraction_cost, 4)},
                indent=1,
                sort_keys=True,
            )
            + "\n"
        )

    seat = json.loads(SEAT.read_text())
    seat_flagged = {
        name
        for name, verdict in seat["verdicts"].items()
        if verdict.get("verdict") == "flag"
    }

    arms: dict[str, dict] = {}
    for arm_label, answers in (("chat", chat_answers), ("jev", jev_answers)):
        graded, relations, confirm_cost = await graded_findings(
            rows, answers, arm_label
        )
        dev_graded = flat(graded, dev_names)
        thresholds = {
            slot.key: tune_slot(
                [finding for finding in dev_graded if finding["slot"] == slot.key],
                dev_labels,
            )
            for slot in SLOTS
        }
        thresholds["joint"] = tune_joint(
            dev_graded, dev_labels, [slot.key for slot in SLOTS]
        )
        thresholds_plain = dict(thresholds["joint"]["thresholds"])
        dropped, admitted, pair_rows = tune_pairs(
            dev_graded, dev_labels, thresholds_plain
        )
        kept = keep(graded, thresholds_plain, dropped, admitted)
        seat_dev = {name for name in seat_flagged if name in dev_labels}
        union_thresholds = tune_joint(
            dev_graded,
            dev_labels,
            [slot.key for slot in SLOTS],
            base_flags=seat_dev,
        )
        union_plain = dict(union_thresholds["thresholds"])
        union_dropped, union_admitted, union_rows = tune_pairs(
            dev_graded, dev_labels, union_plain, base_flags=seat_dev
        )
        union_kept = keep(graded, union_plain, union_dropped, union_admitted)
        arms[arm_label] = {
            "graded": kept,
            "union_graded": union_kept,
            "relations": relations,
            "confirm_cost": confirm_cost,
            "thresholds": thresholds,
            "dropped_pairs": dropped,
            "admitted_pairs": admitted,
            "pair_rows": pair_rows,
            "union_thresholds": union_thresholds,
            "union_dropped_pairs": union_dropped,
            "union_admitted_pairs": union_admitted,
            "union_pair_rows": union_rows,
            "cost_usd": round(
                (5.3055 if arm_label == "chat" else jev_extraction_cost) + confirm_cost,
                4,
            ),
        }

    def slot_flags(kept: dict[str, list[dict]], slot_key: str) -> set[str]:
        return {
            name
            for name, found in kept.items()
            if any(finding["slot"] == slot_key for finding in found)
        }

    def score_both(flagged: set[str]) -> dict:
        return {"held_out": score(held, flagged), "all_agreed": score(rows, flagged)}

    def per_slot(kept: dict[str, list[dict]]) -> dict:
        out: dict[str, dict] = {}
        for slot in SLOTS:
            flags = slot_flags(kept, slot.key)
            scored = score_both(flags)
            scored["held_out"]["false_flag_rate"] = rate(scored["held_out"])
            scored["all_agreed"]["false_flag_rate"] = rate(scored["all_agreed"])
            out[slot.key] = scored
        return out

    chat_flagged = set(arms["chat"]["graded"])
    jev_flagged = set(arms["jev"]["graded"])
    chat_union = set(arms["chat"]["union_graded"]) | seat_flagged
    jev_union = set(arms["jev"]["union_graded"]) | seat_flagged
    union_flags = {
        "probes_chat": chat_flagged,
        "probes_jev": jev_flagged,
        "probes_chat_or_jev": chat_flagged | jev_flagged,
        "seat": seat_flagged,
        "union_chat_seat": chat_union,
        "union_jev_seat": jev_union,
        "union_every_arm": chat_union | jev_union,
    }
    arm_scores = {key: score_both(flags) for key, flags in union_flags.items()}
    for scored in arm_scores.values():
        for split_name in ("held_out", "all_agreed"):
            scored[split_name]["false_flag_rate"] = rate(scored[split_name])

    every_arm = chat_union | jev_union
    missed_everywhere = [
        {
            "name": row["name"],
            "description": " ".join((row.get("description") or "").split())[:400],
            "documentation": " ".join((row.get("documentation") or "").split())[:400],
            "unit": row.get("unit") or "",
        }
        for row in held
        if row["label"] == "mismatch" and row["name"] not in every_arm
    ]

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
            "development_mismatched": sum(
                1 for row in dev if row["label"] == "mismatch"
            ),
            "development_agreeing": sum(1 for row in dev if row["label"] == "agrees"),
        },
        "held_out_population": {
            "names": len(held),
            "mismatched": sum(1 for row in held if row["label"] == "mismatch"),
            "agreeing": sum(1 for row in held if row["label"] == "agrees"),
            "note": "every arm is scored on this one set, the same names",
        },
        "split": {
            "seed": SPLIT_SEED,
            "development_names": len(dev),
            "held_out_names": len(held),
        },
        "scorings": {
            "count": 3,
            "note": (
                "the held-out names have been scored three times: no closure, then "
                "the full closure with a hard relation class and a dropped pair "
                "table, and now the narrowed closure with a graded judgment and "
                "per-slot thresholds"
            ),
            "changed_between": [
                {
                    "scoring": 1,
                    "change": "first reading: no closure, no confirming question, no pair table",
                },
                {
                    "scoring": 2,
                    "change": (
                        "added a name-implication closure covering species, object, "
                        "locus, extensive base and unit; one hard-class Jev confirming "
                        "question; (slot, option) pairs dropped below chance"
                    ),
                },
                {
                    "scoring": 3,
                    "change": (
                        "closure narrowed to exact implications only (per-unit-volume, "
                        "the species' own ions, the named locus, the named transformation, "
                        "the natural integral); an object token no longer suppresses a "
                        "restriction; the confirming judgment is graded and each slot's "
                        "threshold is tuned on development; a Jev extraction arm added",
                    ),
                },
            ],
        },
        "cost_usd": {
            "chat_probe_extraction": 5.3055,
            "jev_probe_extraction": round(jev_extraction_cost, 4),
            "jev_confirmation_chat_arm": arms["chat"]["confirm_cost"],
            "jev_confirmation_jev_arm": arms["jev"]["confirm_cost"],
            "chat_seat_calls": seat["cost_usd"],
            "per_name_all_arms": round(
                (
                    5.3055
                    + jev_extraction_cost
                    + arms["chat"]["confirm_cost"]
                    + arms["jev"]["confirm_cost"]
                )
                / len(rows),
                4,
            ),
            "per_name_jev_only": round(
                (jev_extraction_cost + arms["jev"]["confirm_cost"]) / len(rows), 4
            ),
        },
        "confirming_relations": {
            "chat_arm": dict(sorted(arms["chat"]["relations"].items())),
            "jev_arm": dict(sorted(arms["jev"]["relations"].items())),
        },
        "thresholds": {
            "rule": (
                "per slot, the lowest threshold on the judge's probability that the "
                "phrase is a narrower part or a different form, raised jointly one "
                "grid step at a time from the worst-contributing slot until the "
                "arm's own development false-flag rate is at or under the ceiling; "
                "each slot's own curve is reported beside the joint trace. The union arms "
                "are tuned the same way with the chat seat's own development flags "
                "counted in the instrument they are scored as"
            ),
            "chat_arm": arms["chat"]["thresholds"],
            "jev_arm": arms["jev"]["thresholds"],
            "union_chat_arm": arms["chat"]["union_thresholds"],
            "union_jev_arm": arms["jev"]["union_thresholds"],
        },
        "pairs": {
            "rule": (
                "a (slot, option) pair the threshold excludes is re-admitted when its "
                "development precision is above chance, it recovers a development "
                "mismatch and the development false-flag rate stays at or under the "
                "ceiling"
            ),
            "chat_arm": {
                "dropped": arms["chat"]["dropped_pairs"],
                "readmitted": arms["chat"]["admitted_pairs"],
                "rows": arms["chat"]["pair_rows"],
            },
            "jev_arm": {
                "dropped": arms["jev"]["dropped_pairs"],
                "readmitted": arms["jev"]["admitted_pairs"],
                "rows": arms["jev"]["pair_rows"],
            },
            "union_chat_arm": {
                "dropped": arms["chat"]["union_dropped_pairs"],
                "readmitted": arms["chat"]["union_admitted_pairs"],
                "rows": arms["chat"]["union_pair_rows"],
            },
            "union_jev_arm": {
                "dropped": arms["jev"]["union_dropped_pairs"],
                "readmitted": arms["jev"]["union_admitted_pairs"],
                "rows": arms["jev"]["union_pair_rows"],
            },
        },
        "arms": arm_scores,
        "by_slot": {
            "chat_arm": per_slot(arms["chat"]["graded"]),
            "jev_arm": per_slot(arms["jev"]["graded"]),
        },
        "missed_by_every_arm": {
            "names": [row["name"] for row in missed_everywhere],
            "rows": missed_everywhere,
        },
        "probe_flagged_names": {
            "chat_arm": sorted(chat_flagged),
            "jev_arm": sorted(jev_flagged),
        },
    }
    gate = result["arms"]["union_every_arm"]["held_out"]
    result["gate"] = {
        "recall_floor": RECALL_FLOOR,
        "false_flag_ceiling": FALSE_FLAG_CEILING,
        "arm_gates": {
            arm: {
                "recall_met": bool(
                    (values["held_out"]["recall"] or 0.0) >= RECALL_FLOOR
                ),
                "false_flag_ceiling_met": bool(
                    (values["held_out"]["false_flag_rate"] or 1.0) <= FALSE_FLAG_CEILING
                ),
                "met": bool(
                    (values["held_out"]["recall"] or 0.0) >= RECALL_FLOOR
                    and (values["held_out"]["false_flag_rate"] or 1.0)
                    <= FALSE_FLAG_CEILING
                ),
            }
            for arm, values in arm_scores.items()
        },
        "met": bool(
            (gate["recall"] or 0.0) >= RECALL_FLOOR
            and (gate["false_flag_rate"] or 1.0) <= FALSE_FLAG_CEILING
        ),
    }
    (FRAGMENTS / "name-definition-probes-results.json").write_text(
        json.dumps(result, indent=1, sort_keys=True) + "\n"
    )
    print(
        json.dumps(
            {
                "arms": result["arms"],
                "gate": result["gate"],
                "by_slot": result["by_slot"],
                "cost_usd": result["cost_usd"],
                "confirming_relations": result["confirming_relations"],
                "thresholds": {
                    "chat_arm": arms["chat"]["thresholds"]["joint"],
                    "jev_arm": arms["jev"]["thresholds"]["joint"],
                },
                "held_out_population": result["held_out_population"],
                "missed_by_every_arm": result["missed_by_every_arm"]["names"],
            },
            indent=1,
        )[:6000]
    )


def main() -> None:
    which = sys.argv[1] if len(sys.argv) > 1 else "prototype"
    rows = load_rows()
    if which == "prototype":
        seat = json.loads(SEAT.read_text())
        wanted = set(seat["missed"]) | set(seat["false_flags"])
        proto = [row for row in rows if row["name"] in wanted]
        slot_set = tuple(slot for slot in SLOTS if slot.key in PROTOTYPE_SLOTS)
        items = [{**row, "id": row["name"]} for row in proto]
        answers, cost = asyncio.run(
            probe_definitions(items, slots=slot_set, model=review_model())
        )
        findings = {
            row["name"]: probe_check(row, answers.get(row["name"], {}), slots=slot_set)
            for row in proto
        }
        by_slot = collections.Counter(
            finding["slot"] for found in findings.values() for finding in found
        )
        result = {
            "arm": "prototype",
            "slots": [slot.key for slot in slot_set],
            "cost_usd": round(cost, 4),
            "names": len(proto),
            "findings": {name: found for name, found in findings.items() if found},
            "slot_counts": dict(sorted(by_slot.items())),
            "answers": answers,
            "prototype_names": {
                "missed_by_seat": sorted(wanted & set(seat["missed"])),
                "wrongly_flagged_by_seat": sorted(wanted & set(seat["false_flags"])),
            },
            "flagged_by_probe": {
                name: [f["slot"] for f in found]
                for name, found in findings.items()
                if found
            },
        }
        (FRAGMENTS / "name-definition-probes-prototype.json").write_text(
            json.dumps(result, indent=1, sort_keys=True) + "\n"
        )
        print(
            json.dumps({k: v for k, v in result.items() if k != "answers"}, indent=1)[
                :4000
            ]
        )
        return
    asyncio.run(final())


if __name__ == "__main__":
    main()
