"""Measure the per-family domain decision against the blind readers' labels.

Two readings, both from the landed per-family decision
(:func:`imas_codex.standard_names.domain_decision.decide_family_domain`):

* the labelled comparison — for the families and single names the two blind
  readers agree on, the share of member names the per-family decision files
  under the reader's domain, beside the share the stored per-name domains
  reach;
* the calibration regression — accuracy on the names the domain decision was
  calibrated on, beside the stored domains' accuracy on the same names.

The stored domains are read from the live graph and nothing is written to it:
every decision is held in memory and the graph is opened read-only. Each family
is decided once and every member reads that decision, so the count of judgments
paid is the count of families decided.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from collections import Counter
from pathlib import Path
from typing import Any

from imas_codex.graph.client import GraphClient
from imas_codex.settings import get_model
from imas_codex.discovery.base.judgment import decide_batch
from imas_codex.graph.dd_domain_classifier import SERVICE_TAG
from imas_codex.standard_names.domain_decision import (
    CONFIDENCE_FLOOR,
    QUESTION,
    SEAT,
    build_family_question,
    build_family_state,
    decide_family_domain,
    family_candidate_domains,
    ids_domain_map,
)
from imas_codex.standard_names.families import fetch_domain_families

ROOT = Path(__file__).resolve().parents[4]
LABEL_DIR = ROOT / "docs/evidence/fragments/sn-domain-family-labels"
CALIBRATION_DIR = ROOT / "docs/evidence/fragments/sn-catalog-grouping"

#: The eight species families §1 measured split, by their sheet pattern.
SECTION_ONE_PATTERNS = (
    "<X>_density",
    "ratio_of_volume_averaged_<X>_density_to_volume_averaged_electron_density",
    "<X>_density_at_internal_transport_barrier",
    "<X>_density_at_pedestal_top",
    "line_averaged_<X>_density",
    "<X>_density_at_plasma_boundary",
    "flux_surface_averaged_<X>_density_at_plasma_boundary",
    "toroidal_<X>_velocity_at_magnetic_axis",
)

#: The three names §5 singles out; they are reported apart from the drawn draw.
NAMED = ("safety_factor", "greenwald_density", "voltage_of_valve")


def _load(path: Path) -> Any:
    with path.open() as handle:
        return json.load(handle)


def _agreed_labels() -> tuple[dict[str, str], list[dict[str, str]]]:
    """The items the two readers agree on, and the items they disagree on."""
    first = {row["id"]: row for row in _load(LABEL_DIR / "labels.json")}
    second = {row["id"]: row for row in _load(LABEL_DIR / "labels-second-reader.json")}
    agreed: dict[str, str] = {}
    disagreed: list[dict[str, str]] = []
    for item_id in sorted(set(first) & set(second)):
        one, two = first[item_id], second[item_id]
        if one.get("domain") and one.get("domain") == two.get("domain"):
            agreed[item_id] = str(one["domain"])
        else:
            disagreed.append(
                {
                    "id": item_id,
                    "reader_one": str(one.get("domain")),
                    "reader_two": str(two.get("domain")),
                }
            )
    return agreed, disagreed


def _reader_agreement() -> dict[str, Any]:
    """The agreement counts by sheet kind, for the record."""
    sheet = _load(LABEL_DIR / "blind-sheet.json")
    kind_of = {item["id"]: item["kind"] for item in sheet["items"]}
    agreed, disagreed = _agreed_labels()
    counted = Counter(kind_of[item_id] for item_id in agreed)
    split = Counter(kind_of[row["id"]] for row in disagreed)
    return {
        "agreed_by_kind": dict(counted),
        "disagreed_by_kind": dict(split),
        "disagreed": disagreed,
        "items_by_kind": dict(Counter(kind_of.values())),
    }


def _index_families(families: list[dict[str, Any]]) -> tuple[dict[str, dict], dict[str, dict]]:
    by_name: dict[str, dict] = {}
    by_key: dict[str, dict] = {}
    for family in families:
        by_key[family["key"]] = family
        for member in family["members"]:
            by_name[member["id"]] = family
    return by_name, by_key


def _sheet_targets(
    sheet: dict[str, Any], agreed: dict[str, str], by_name: dict[str, dict]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Map each agreed sheet item onto the live decision family it names.

    A sheet family is grouped by species or component token, and the decision
    groups by the invariant segments with the locus held fixed, so the mapping
    runs through the sheet family's example member names. A sheet family whose
    examples land in more than one decision family is reported rather than
    resolved here.
    """
    family_targets: list[dict[str, Any]] = []
    single_targets: list[dict[str, Any]] = []
    multi: list[dict[str, Any]] = []
    for item in sheet["items"]:
        label = agreed.get(item["id"])
        if label is None:
            continue
        if item["kind"] == "family":
            keys = []
            for example in item["examples"]:
                family = by_name.get(example["name"])
                if family is not None and family["key"] not in keys:
                    keys.append(family["key"])
            if len(keys) != 1:
                multi.append(
                    {
                        "sheet_id": item["id"],
                        "pattern": item["pattern"],
                        "decision_families": keys,
                        "reference": label,
                    }
                )
                continue
            family_targets.append(
                {
                    "sheet_id": item["id"],
                    "pattern": item["pattern"],
                    "sheet_members": item.get("member_count"),
                    "family_key": keys[0],
                    "reference": label,
                }
            )
        else:
            name = item.get("name")
            family = by_name.get(name) if name else None
            if family is None:
                continue
            single_targets.append(
                {
                    "sheet_id": item["id"],
                    "name": name,
                    "family_key": family["key"],
                    "family_size": len(family["members"]),
                    "reference": label,
                    "named": name in NAMED,
                }
            )
    return family_targets, single_targets, multi


def _calibration_rows() -> list[dict[str, str]]:
    """The labelled names the domain decision was calibrated on."""
    sheet = _load(CALIBRATION_DIR / "scg-domain-label-sheet.json")
    rows: list[dict[str, str]] = []
    for item in list(sheet["reference"]) + list(sheet["sample"]):
        label = item.get("label") or item.get("reference")
        if item.get("name") and label:
            rows.append(
                {
                    "name": str(item["name"]),
                    "label": str(label),
                    "split": str(item.get("split") or item.get("label_source") or ""),
                }
            )
    return rows


def _incumbent() -> dict[str, str]:
    raw = _load(CALIBRATION_DIR / "scg-domain-incumbent.json")
    return {
        str(name): str(row.get("physics_domain") or "")
        for name, row in raw.items()
    }


async def _decide_all(
    families: dict[str, dict], model: str
) -> dict[str, Any]:
    ids_map = ids_domain_map()
    decisions: dict[str, Any] = {}
    semaphore = asyncio.Semaphore(8)

    async def one(key: str) -> None:
        family = families[key]
        async with semaphore:
            decisions[key] = await decide_family_domain(
                key, family["members"], ids_map, model=model
            )

    await asyncio.gather(*(one(key) for key in sorted(families)))
    return decisions


async def _leading_domains(
    key: str, family: dict, model: str, ids_map: dict[str, str]
) -> list[dict[str, Any]]:
    """The answer's two leading domains, read again to keep its probabilities."""
    options = family_candidate_domains(family["members"], ids_map)
    state = build_family_state(key, family["members"], ids_map, options)
    results, _ = await decide_batch(
        [state],
        build_family_question(options, len(family["members"])),
        model=model,
        service=SERVICE_TAG,
        concurrency=1,
    )
    answer = None
    if results and results[0]:
        answers = results[0][0]
        if isinstance(answers, dict):
            answer = answers.get(QUESTION)
    if not isinstance(answer, dict):
        return []
    probabilities = {
        str(k): float(v) for k, v in (answer.get("probabilities") or {}).items()
    }
    ranked = sorted(probabilities.items(), key=lambda pair: pair[1], reverse=True)
    return [{"domain": domain, "probability": round(p, 4)} for domain, p in ranked[:2]]


def _filed_domain(decision: Any, member: dict) -> str:
    """The domain a member is filed under: the family's choice, else its stored."""
    if decision is not None and decision.decided:
        return str(decision.choice)
    return str(member.get("physics_domain") or "")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--only-families", action="store_true")
    args = parser.parse_args()

    seat = get_model(SEAT)
    agreement = _reader_agreement()
    agreed, _ = _agreed_labels()
    sheet = _load(LABEL_DIR / "blind-sheet.json")

    with GraphClient() as gc:
        families = fetch_domain_families(gc, min_size=1)
    by_name, by_key = _index_families(families)

    family_targets, single_targets, multi_mapped = _sheet_targets(
        sheet, agreed, by_name
    )

    calibration = _calibration_rows()
    calibration_family: dict[str, str] = {}
    for row in calibration:
        family = by_name.get(row["name"])
        if family is not None:
            calibration_family[row["name"]] = family["key"]

    # One judgment per family: the union of every family the three readings
    # touch, decided once each.
    needed: dict[str, dict] = {}
    for target in family_targets:
        needed[target["family_key"]] = by_key[target["family_key"]]
    for target in single_targets:
        needed[target["family_key"]] = by_key[target["family_key"]]
    for key in calibration_family.values():
        needed[key] = by_key[key]

    decisions = asyncio.run(_decide_all(needed, seat))
    ids_map = ids_domain_map()

    # ---- the labelled family comparison ---------------------------------
    family_rows: list[dict[str, Any]] = []
    counted_members = 0
    family_matches = 0
    stored_matches = 0
    for target in family_targets:
        family = by_key[target["family_key"]]
        decision = decisions[target["family_key"]]
        members = family["members"]
        row_matches = sum(
            1 for m in members if _filed_domain(decision, m) == target["reference"]
        )
        row_stored = sum(
            1 for m in members if str(m.get("physics_domain") or "") == target["reference"]
        )
        family_rows.append(
            {
                **{k: v for k, v in target.items() if k != "family_key"},
                "family_key": target["family_key"],
                "live_members": len(members),
                "choice": decision.choice,
                "probability": decision.probability,
                "decided": decision.decided,
                "undecided_reason": decision.undecided_reason,
                "family_matches": row_matches,
                "stored_matches": row_stored,
                "family_share": round(row_matches / len(members), 4) if members else 0.0,
                "stored_share": round(row_stored / len(members), 4) if members else 0.0,
                "section_one": target["pattern"] in SECTION_ONE_PATTERNS,
            }
        )
        if len(members) >= 3:
            counted_members += len(members)
            family_matches += row_matches
            stored_matches += row_stored

    families_of_three = [r for r in family_rows if r["live_members"] >= 3]
    comparison = {
        "families_measured": len(families_of_three),
        "section_one_families_measured": sum(
            1 for r in families_of_three if r["section_one"]
        ),
        "members_measured": counted_members,
        "family_match_share": round(family_matches / counted_members, 4)
        if counted_members
        else 0.0,
        "stored_match_share": round(stored_matches / counted_members, 4)
        if counted_members
        else 0.0,
        "families_where_family_share_higher": sum(
            1 for r in families_of_three if r["family_share"] > r["stored_share"]
        ),
        "families_where_stored_share_higher": sum(
            1 for r in families_of_three if r["stored_share"] > r["family_share"]
        ),
        "disagreements": [
            {
                "pattern": r["pattern"],
                "reference": r["reference"],
                "decision": r["choice"],
                "members": r["live_members"],
            }
            for r in families_of_three
            if r["decided"] and r["choice"] != r["reference"]
        ],
    }

    # ---- the agreed single names ----------------------------------------
    single_rows: list[dict[str, Any]] = []
    drawn_family = drawn_stored = drawn_count = 0
    named_rows: list[dict[str, Any]] = []
    for target in single_targets:
        family = by_key[target["family_key"]]
        decision = decisions[target["family_key"]]
        member = next((m for m in family["members"] if m["id"] == target["name"]), None)
        if member is None:
            continue
        filed = _filed_domain(decision, member)
        stored = str(member.get("physics_domain") or "")
        row = {
            **target,
            "filed": filed,
            "stored": stored,
            "family_choice": decision.choice,
            "probability": decision.probability,
            "decided": decision.decided,
            "undecided_reason": decision.undecided_reason,
            "family_match": filed == target["reference"],
            "stored_match": stored == target["reference"],
        }
        single_rows.append(row)
        if target["named"]:
            named_rows.append(row)
        else:
            drawn_count += 1
            drawn_family += int(row["family_match"])
            drawn_stored += int(row["stored_match"])

    singles = {
        "drawn_measured": drawn_count,
        "drawn_family_share": round(drawn_family / drawn_count, 4) if drawn_count else 0.0,
        "drawn_stored_share": round(drawn_stored / drawn_count, 4) if drawn_count else 0.0,
        "drawn_undecided": sum(1 for r in single_rows if not r["named"] and not r["decided"]),
        "named": named_rows,
        "disagreements": [
            {
                "name": r["name"],
                "reference": r["reference"],
                "decision": r["filed"],
                "stored": r["stored"],
            }
            for r in single_rows
            if not r["family_match"]
        ],
    }

    # ---- the calibration regression -------------------------------------
    incumbent = _incumbent()
    calibration_rows: list[dict[str, Any]] = []
    family_hits = stored_hits = 0
    undecided_calibration = 0
    for row in calibration:
        key = calibration_family.get(row["name"])
        if key is None:
            continue
        family = by_key[key]
        decision = decisions[key]
        member = next((m for m in family["members"] if m["id"] == row["name"]), None)
        if member is None:
            continue
        filed = _filed_domain(decision, member)
        stored = incumbent.get(row["name"]) or str(member.get("physics_domain") or "")
        calibration_rows.append(
            {
                "name": row["name"],
                "label": row["label"],
                "split": row["split"],
                "family_size": len(family["members"]),
                "filed": filed,
                "stored": stored,
                "family_match": filed == row["label"],
                "stored_match": stored == row["label"],
                "decided": decision.decided,
            }
        )
        family_hits += int(filed == row["label"])
        stored_hits += int(stored == row["label"])
        undecided_calibration += int(not decision.decided)

    total_cal = len(calibration_rows)
    calibration_reading = {
        "names_measured": total_cal,
        "single_member_families": sum(
            1 for r in calibration_rows if r["family_size"] == 1
        ),
        "family_accuracy": round(family_hits / total_cal, 4) if total_cal else 0.0,
        "stored_accuracy": round(stored_hits / total_cal, 4) if total_cal else 0.0,
        "undecided": undecided_calibration,
    }

    # ---- undecided families, with their two leading domains -------------
    undecided_rows: list[dict[str, Any]] = []
    for key in sorted(needed):
        decision = decisions[key]
        if decision.decided:
            continue
        family = by_key[key]
        leading = asyncio.run(_leading_domains(key, family, seat, ids_map))
        undecided_rows.append(
            {
                "family_key": key,
                "member_count": len(family["members"]),
                "members": [m["id"] for m in family["members"]],
                "candidates": list(decision.candidates),
                "reason": decision.undecided_reason,
                "leading": leading,
            }
        )

    judged = [d for d in decisions.values() if len(d.candidates) > 1]
    payload = {
        "seat": seat,
        "confidence_floor": CONFIDENCE_FLOOR,
        "reader_agreement": agreement,
        "sheet_families_mapping_to_more_than_one_decision_family": multi_mapped,
        "judgments": {
            "families_decided": len(decisions),
            "judgments_paid": len(judged),
            "no_call_families": len(decisions) - len(judged),
            "cost_usd": round(sum(float(d.cost or 0.0) for d in decisions.values()), 6),
            "live_families_scanned": len(families),
        },
        "family_comparison": comparison,
        "family_rows": family_rows,
        "single_names": singles,
        "single_rows": single_rows,
        "calibration": calibration_reading,
        "calibration_rows": calibration_rows,
        "undecided": undecided_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as handle:
        json.dump(payload, handle, indent=1)
        handle.write("\n")

    print(json.dumps(
        {
            "seat": seat,
            "agreement": agreement["agreed_by_kind"],
            "families_measured": comparison["families_measured"],
            "family_share": comparison["family_match_share"],
            "stored_share": comparison["stored_match_share"],
            "drawn_singles": (singles["drawn_measured"], singles["drawn_family_share"], singles["drawn_stored_share"]),
            "calibration": calibration_reading,
            "judgments": payload["judgments"],
            "undecided": len(undecided_rows),
            "multi_mapped": len(multi_mapped),
        },
        indent=1,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())