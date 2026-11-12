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
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from imas_codex.discovery.base.judgment import decide_batch
from imas_codex.graph.client import GraphClient
from imas_codex.graph.dd_domain_classifier import SERVICE_TAG
from imas_codex.settings import get_model
from imas_codex.standard_names import domain_decision as _dd
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


async def _counted_decide_batch(*args: Any, **kwargs: Any) -> Any:
    """Count the judgments an arm pays before delegating to ``decide_batch``."""
    _JUDGMENT_CALLS[0] += 1
    return await decide_batch(*args, **kwargs)


_dd.decide_batch = _counted_decide_batch

ROOT = Path(__file__).resolve().parents[4]
LABEL_DIR = ROOT / "docs/evidence/fragments/sn-domain-family-labels"
CALIBRATION_DIR = ROOT / "docs/evidence/fragments/sn-catalog-grouping"

#: The eight species families whose members the catalog files across more than
#: one domain, by their sheet pattern.
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

#: The three names whose sources pin them to a domain the concept does not
#: share; they are reported apart from the drawn draw.
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


def _index_families(
    families: list[dict[str, Any]],
) -> tuple[dict[str, dict], dict[str, dict]]:
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
        str(name): str(row.get("physics_domain") or "") for name, row in raw.items()
    }


#: The ``decide_batch`` calls paid by the arm in flight. The fallback set makes
#: a second call for a family the narrow options cannot place, so the count of
#: calls is what the arm is billed for and not the count of families.
_JUDGMENT_CALLS: list[int] = [0]


async def _decide_all(
    families: dict[str, dict],
    model: str,
    *,
    option_set: str,
    concurrency: int = 16,
) -> dict[str, Any]:
    """Decide every family once under one option set."""
    ids_map = ids_domain_map()
    decisions: dict[str, Any] = {}
    semaphore = asyncio.Semaphore(concurrency)

    async def one(key: str) -> None:
        family = families[key]
        async with semaphore:
            decisions[key] = await decide_family_domain(
                key,
                family["members"],
                ids_map,
                model=model,
                option_set=option_set,
            )

    _JUDGMENT_CALLS[0] = 0
    await asyncio.gather(*(one(key) for key in sorted(families)))
    return decisions


def _family_comparison(
    family_targets: list[dict[str, Any]],
    by_key: dict[str, dict],
    decisions: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """The member-share reading over the agreed sheet families, for one arm."""
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
            1
            for m in members
            if str(m.get("physics_domain") or "") == target["reference"]
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
                "family_share": round(row_matches / len(members), 4)
                if members
                else 0.0,
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
    return comparison, family_rows


def _single_comparison(
    single_targets: list[dict[str, Any]],
    by_key: dict[str, dict],
    decisions: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """The drawn and named single-name reading, for one arm."""
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
        "drawn_family_share": round(drawn_family / drawn_count, 4)
        if drawn_count
        else 0.0,
        "drawn_stored_share": round(drawn_stored / drawn_count, 4)
        if drawn_count
        else 0.0,
        "drawn_undecided": sum(
            1 for r in single_rows if not r["named"] and not r["decided"]
        ),
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
    return singles, single_rows


def _calibration_comparison(
    calibration: list[dict[str, str]],
    calibration_family: dict[str, str],
    by_key: dict[str, dict],
    decisions: dict[str, Any],
    incumbent: dict[str, str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """The calibration regression reading, for one arm."""
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
        "accuracy_basis": (
            "family_accuracy files an undecided name at its stored domain, which "
            "is what the landed decision leaves it holding; family_accuracy_strict "
            "counts every undecided name as wrong"
        ),
        **_strict_reading(calibration_rows, stored_hits),
    }
    return calibration_reading, calibration_rows


def _undecided_rows(
    needed: dict[str, dict],
    by_key: dict[str, dict],
    decisions: dict[str, Any],
    seat: str,
    ids_map: dict[str, str],
) -> list[dict[str, Any]]:
    """Every undecided family with its two leading domains, for one arm."""
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
    return undecided_rows


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


def _strict_reading(
    calibration_rows: list[dict[str, Any]], stored_hits: int
) -> dict[str, Any]:
    """The calibration accuracy that counts every undecided name as wrong.

    An undecided family writes nothing, so a name in one keeps its stored
    domain and can land on the label without the decision having earned it.
    The landed reading keeps that accident, because it describes what the
    measurement leaves the catalog holding; this reading removes it, so the
    two together say how much of the accuracy is the decision's own.
    """
    total = len(calibration_rows)
    strict_hits = sum(
        1 for row in calibration_rows if row["decided"] and row["family_match"]
    )
    carried = sum(
        1 for row in calibration_rows if not row["decided"] and row["family_match"]
    )
    return {
        "family_accuracy_strict": round(strict_hits / total, 4) if total else 0.0,
        "family_hits_strict": strict_hits,
        "stored_hits": stored_hits,
        "undecided_landing_on_label": carried,
    }


def _undecided_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """The undecided families by size, naming the largest with its members."""
    ordered = sorted(rows, key=lambda row: row["member_count"], reverse=True)
    largest = ordered[0] if ordered else None
    return {
        "families": len(ordered),
        "with_two_or_more_members": sum(
            1 for row in ordered if row["member_count"] >= 2
        ),
        "sizes_descending": [row["member_count"] for row in ordered],
        "largest": (
            {
                "family_key": largest["family_key"],
                "member_count": largest["member_count"],
                "members": largest["members"],
                "leading": largest["leading"],
            }
            if largest
            else None
        ),
    }


def _provenance(seat: str) -> dict[str, Any]:
    """What a later reader needs to re-run or audit this measurement."""
    import subprocess

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=ROOT, capture_output=True, text=True, check=False
        ).stdout.strip()

    return {
        "revision": git("rev-parse", "HEAD"),
        "tree": str(ROOT),
        "module": "imas_codex/standard_names/domain_decision.py",
        "seat": seat,
        "confidence_floor": CONFIDENCE_FLOOR,
        "generated_at": datetime.now(UTC).isoformat(),
        "command": (
            "uv run --no-sync python "
            "docs/evidence/fragments/sn-domain-per-family/"
            "labelled-comparison-run.py --output "
            "docs/evidence/fragments/sn-domain-per-family/widened-options.json"
        ),
        "reproduce": (
            "re-running spends one judgment per family again; --from-existing "
            "re-derives every reading from the committed rows without spending any"
        ),
        "writes_to_graph": False,
        "graph_access": "read-only",
    }


def _derive_from_existing(path: Path, revision: str = "") -> int:
    """Re-derive the strict reading and the undecided summary from committed rows.

    Used to repair an artifact whose rows are sound but whose derived fields
    or prose were wrong: it spends no judgments and changes no measurement.
    """
    with path.open() as handle:
        payload = json.load(handle)
    stored_hits = sum(1 for row in payload["calibration_rows"] if row["stored_match"])
    payload["calibration"].update(
        _strict_reading(payload["calibration_rows"], stored_hits)
    )
    payload["calibration"].setdefault(
        "accuracy_basis",
        "family_accuracy files an undecided name at its stored domain, which is "
        "what the landed decision leaves it holding; family_accuracy_strict "
        "counts every undecided name as wrong",
    )
    payload["undecided_summary"] = _undecided_summary(payload["undecided"])
    payload.setdefault("provenance", {})
    payload["provenance"].update(
        {
            "revised_at": datetime.now(UTC).isoformat(),
            "revision": revision or payload.get("provenance", {}).get("revision", ""),
            "module": "imas_codex/standard_names/domain_decision.py",
            "writes_to_graph": False,
            "graph_access": "read-only",
            "note": (
                "strict reading and undecided summary re-derived from the "
                "committed rows; no judgment was spent and no row changed"
            ),
        }
    )
    with path.open("w") as handle:
        json.dump(payload, handle, indent=1)
        handle.write("\n")
    print(json.dumps(payload["calibration"], indent=1))
    print(
        json.dumps(_undecided_summary(payload["undecided"])["largest"]["member_count"])
    )
    return 0


def _check_fragment(fragment: Path, artifact: Path) -> int:
    """Check the fragment's numeric claims against the measurement artifact.

    Two claims are read back: the size of the largest undecided family, and
    the calibration accuracy on the strict basis. A fragment that quotes a
    figure its own artifact does not hold fails here, which is what the
    reverted state of the artifact is for.
    """
    import re

    html = fragment.read_text()
    with artifact.open() as handle:
        data = json.load(handle)
    largest = (data.get("undecided_summary") or {}).get("largest") or {}
    calibration = data.get("calibration") or {}

    claimed = re.search(r"family of (?:<strong>)?(\d+)(?:</strong>)? members", html)
    claim = int(claimed.group(1)) if claimed else None
    actual = largest.get("member_count")
    size_ok = claim is not None and claim == actual
    print(f"check:undecided-largest claim={claim} artifact={actual} ok={size_ok}")

    strict = calibration.get("family_accuracy_strict")
    strict_text = (
        f"{strict * 100:.2f}" if isinstance(strict, (int, float)) else "absent"
    )
    contradicts = "counted as wrong" in html
    strict_ok = (
        isinstance(strict, (int, float)) and strict_text in html and not contradicts
    )
    print(
        f"check:calibration-strict claim={strict_text}% "
        f"artifact={strict_text}% contradicted_basis={contradicts} ok={strict_ok}"
    )
    return 0 if (size_ok and strict_ok) else 1


#: The bars the measurement is taken against. The stored baselines are read
#: from the same run; these are the fixed floors the section names.
CALIBRATION_FLOOR_STRICT = 0.597
UNDECIDED_MULTI_BOUND = 20

#: Option sets measured and refused in an earlier run of this harness, kept
#: here because their per-family rows were overwritten by the run that followed
#: and the readings are the record of why they were refused.
_REFUSED_OPTION_SETS = {
    "band-split two-stage": {
        "revision": "6d1bd724f",
        "family_match_share": 0.4780,
        "drawn_family_share": 0.6230,
        "calibration_strict": 0.4556,
        "undecided_with_two_or_more_members": 39,
        "judgments_paid": 217,
        "cost_usd": 0.061835,
        "note": (
            "a band step then a domain step, with the two probabilities "
            "multiplied; the product falls below the floor even when both "
            "steps are confident, so families the narrow options had placed "
            "become undecided"
        ),
    }
}


def _reading(arm: dict[str, Any]) -> dict[str, Any]:
    """The five readings a bar is taken on, for one arm."""
    return {
        "family_match_share": arm["family_comparison"]["family_match_share"],
        "stored_family_match_share": arm["family_comparison"]["stored_match_share"],
        "drawn_family_share": arm["single_names"]["drawn_family_share"],
        "drawn_stored_share": arm["single_names"]["drawn_stored_share"],
        "calibration_strict": arm["calibration"]["family_accuracy_strict"],
        "undecided_with_two_or_more_members": arm["undecided_summary"][
            "with_two_or_more_members"
        ],
        "judgments_paid": arm["judgments"]["judgments_paid"],
        "cost_usd": arm["judgments"]["cost_usd"],
        "cost_per_decision_usd": arm["judgments"]["cost_per_decision_usd"],
    }


def _bars_failed(reading: dict[str, Any], bounded: dict[str, Any]) -> list[str]:
    """The bars an arm does not clear, each named with what it read."""
    failed = []
    if reading["family_match_share"] <= reading["stored_family_match_share"]:
        failed.append(
            "family member share "
            f"{reading['family_match_share']:.4f} is not above the stored "
            f"{reading['stored_family_match_share']:.4f}"
        )
    if reading["family_match_share"] < bounded["family_match_share"]:
        failed.append(
            "family member share "
            f"{reading['family_match_share']:.4f} is below the source-bounded "
            f"{bounded['family_match_share']:.4f}"
        )
    if reading["drawn_family_share"] <= reading["drawn_stored_share"]:
        failed.append(
            "drawn single-name share "
            f"{reading['drawn_family_share']:.4f} is not above the stored "
            f"{reading['drawn_stored_share']:.4f}"
        )
    if reading["calibration_strict"] < CALIBRATION_FLOOR_STRICT:
        failed.append(
            f"calibration accuracy with undecided counted wrong "
            f"{reading['calibration_strict']:.4f} is below {CALIBRATION_FLOOR_STRICT}"
        )
    if reading["calibration_strict"] < bounded["calibration_strict"]:
        failed.append(
            "calibration accuracy with undecided counted wrong "
            f"{reading['calibration_strict']:.4f} is below the source-bounded "
            f"{bounded['calibration_strict']:.4f}"
        )
    if reading["undecided_with_two_or_more_members"] > UNDECIDED_MULTI_BOUND:
        failed.append(
            "multi-member families below the floor "
            f"{reading['undecided_with_two_or_more_members']} is above "
            f"{UNDECIDED_MULTI_BOUND}"
        )
    return failed


def _filed_domain(decision: Any, member: dict) -> str:
    """The domain a member is filed under: the family's choice, else its stored."""
    if decision is not None and decision.decided:
        return str(decision.choice)
    return str(member.get("physics_domain") or "")


def _all_live(output: Path, seat: str, option_set: str) -> int:
    """The transport and general share over every live family, never written.

    Every live family is decided under *option_set* in memory; a name
    in an undecided family keeps its stored domain, which is what a live pass
    would leave it holding. The twenty largest moves between domains are
    listed by family so a physicist can read what the widening would change.
    """
    from collections import Counter

    with GraphClient() as gc:
        families = fetch_domain_families(gc, min_size=1)
    indexed = {family["key"]: family for family in families}
    decisions = asyncio.run(
        _decide_all(indexed, seat, option_set=option_set, concurrency=32)
    )
    stored_counts: Counter[str] = Counter()
    filed_counts: Counter[str] = Counter()
    moves: Counter[tuple[str, str]] = Counter()
    family_moves: dict[tuple[str, str], list[str]] = {}
    total = 0
    for family in families:
        decision = decisions[family["key"]]
        for member in family["members"]:
            total += 1
            stored = str(member.get("physics_domain") or "unset")
            stored_counts[stored] += 1
            filed = str(decision.choice) if decision.decided else stored
            filed_counts[filed] += 1
            if filed != stored:
                moves[(stored, filed)] += 1
                family_moves.setdefault((stored, filed), []).append(family["key"])
    calls = _JUDGMENT_CALLS[0]
    cost = round(sum(float(d.cost or 0.0) for d in decisions.values()), 6)
    payload = {
        "seat": seat,
        "option_set": option_set,
        "provenance": _provenance(seat),
        "families": len(families),
        "names": total,
        "writes_to_graph": False,
        "graph_access": "read-only",
        "stored_domain_counts": dict(stored_counts.most_common()),
        "filed_domain_counts": dict(filed_counts.most_common()),
        "transport": {
            "stored": stored_counts.get("transport", 0),
            "widened": filed_counts.get("transport", 0),
        },
        "general": {
            "stored": stored_counts.get("general", 0),
            "widened": filed_counts.get("general", 0),
        },
        "judgments": {
            "families_decided": len(decisions),
            "judgments_paid": calls,
            "cost_usd": cost,
            "cost_per_decision_usd": round(cost / calls, 8) if calls else 0.0,
        },
        "largest_moves": [
            {
                "from": pair[0],
                "to": pair[1],
                "names": count,
                "families": family_moves[pair][:5],
            }
            for pair, count in moves.most_common(20)
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as handle:
        json.dump(payload, handle, indent=1)
        handle.write("\n")
    print(
        json.dumps(
            {
                "families": len(families),
                "names": total,
                "transport": payload["transport"],
                "general": payload["general"],
                "judgments": payload["judgments"],
            },
            indent=1,
        )
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--from-existing", action="store_true")
    parser.add_argument(
        "--all-live",
        action="store_true",
        help="decide every live family and record the transport and general share",
    )
    parser.add_argument(
        "--option-set",
        choices=("bounded", "widened", "fallback"),
        default="fallback",
        help="the option set the --all-live pass decides under",
    )
    parser.add_argument(
        "--check-fragment",
        type=Path,
        default=None,
        help="check a fragment's numeric claims against the artifact and exit",
    )
    parser.add_argument(
        "--revision",
        default="",
        help="the revision the measurement ran at, when deriving from existing rows",
    )
    args = parser.parse_args()

    if args.check_fragment is not None:
        return _check_fragment(args.check_fragment, args.output)

    if args.from_existing:
        return _derive_from_existing(args.output, revision=args.revision)

    if args.all_live:
        return _all_live(args.output, get_model(SEAT), args.option_set)

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

    decided: dict[str, tuple[dict[str, Any], int]] = {}
    for option_set in ("bounded", "widened", "fallback"):
        decisions = asyncio.run(_decide_all(needed, seat, option_set=option_set))
        decided[option_set] = (decisions, _JUDGMENT_CALLS[0])
    ids_map = ids_domain_map()
    incumbent = _incumbent()

    arms: dict[str, Any] = {}
    for arm, (decisions, calls) in decided.items():
        comparison, family_rows = _family_comparison(family_targets, by_key, decisions)
        singles, single_rows = _single_comparison(single_targets, by_key, decisions)
        calibration_reading, calibration_rows = _calibration_comparison(
            calibration, calibration_family, by_key, decisions, incumbent
        )
        undecided_rows = _undecided_rows(needed, by_key, decisions, seat, ids_map)
        cost = round(sum(float(d.cost or 0.0) for d in decisions.values()), 6)
        arms[arm] = {
            "option_set": arm,
            "judgments": {
                "families_decided": len(decisions),
                "judgments_paid": calls,
                "no_call_families": sum(
                    1 for decision in decisions.values() if not decision.cost
                ),
                "cost_usd": cost,
                "cost_per_decision_usd": round(cost / calls, 8) if calls else 0.0,
                "live_families_scanned": len(families),
            },
            "family_comparison": comparison,
            "family_rows": family_rows,
            "single_names": singles,
            "single_rows": single_rows,
            "calibration": calibration_reading,
            "calibration_rows": calibration_rows,
            "undecided": undecided_rows,
            "undecided_summary": _undecided_summary(undecided_rows),
        }

    readings = {name: _reading(arm) for name, arm in arms.items()}
    bounded_reading = readings["bounded"]
    for reading in readings.values():
        reading["bars_failed"] = _bars_failed(reading, bounded_reading)
        reading["clears_every_bar"] = not reading["bars_failed"]
    reported = next(
        (
            name
            for name in ("fallback", "widened")
            if readings[name]["clears_every_bar"]
        ),
        "widened",
    )
    arm = arms[reported]
    payload = {
        "seat": seat,
        "confidence_floor": CONFIDENCE_FLOOR,
        "provenance": _provenance(seat),
        "reader_agreement": agreement,
        "reported_arm": reported,
        "sheet_families_mapping_to_more_than_one_decision_family": multi_mapped,
        "judgments": arm["judgments"],
        "family_comparison": arm["family_comparison"],
        "family_rows": arm["family_rows"],
        "single_names": arm["single_names"],
        "single_rows": arm["single_rows"],
        "calibration": arm["calibration"],
        "calibration_rows": arm["calibration_rows"],
        "undecided": arm["undecided"],
        "undecided_summary": arm["undecided_summary"],
        "readings": readings,
        "arms": arms,
        "refused_option_sets": _REFUSED_OPTION_SETS,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as handle:
        json.dump(payload, handle, indent=1)
        handle.write("\n")

    print(
        json.dumps(
            {
                "seat": seat,
                "agreement": agreement["agreed_by_kind"],
                "reported_arm": reported,
                "readings": readings,
                "multi_mapped": len(multi_mapped),
            },
            indent=1,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
