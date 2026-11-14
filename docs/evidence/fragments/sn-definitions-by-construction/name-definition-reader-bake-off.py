"""Score readers that check a standard name against its definition.

Five arms are measured against the settled blind-reader verdicts on the export
candidate set: the production docs-review seat in the batched form it runs, and
three cheap readers (Haiku, Luna, the local model) each read three times, plus a
mixed panel drawing one reading from each of the three. Two-of-three majority
where an arm reads three times. Every reading is its own call with no other
reading in context, and every arm sees the same prompt and the same text.

Readings are appended to a progress file per arm and reading index, so a run cut
short keeps what it has already paid for. Modes:

  run     pilot a stratified slice, project the full-set spend, then run it all
  report  score whatever is on disk and write the JSON, the page and the figure

Spend is read from each call's own reported cost, and the run stops when the
projection passes the ceiling.
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import hashlib
import json
import math
import os
import random
import re
import time
from pathlib import Path

WORKTREE = Path(__file__).resolve().parents[4]
FRAGMENTS = Path(__file__).resolve().parent
EVIDENCE = FRAGMENTS.parent
OUT_JSON = FRAGMENTS / "name-definition-reader-bake-off.json"
OUT_HTML = FRAGMENTS / "sdbc-name-definition-reader-bake-off.html"
FIG_DIR = WORKTREE / "docs/figures/sn-name-definition-readers"
FIG_NAME = "recall-precision-by-arm.svg"
STATE = Path(
    os.environ.get("BAKE_OFF_STATE")
    or (Path.home() / ".cache/imas-codex-live-rewrite/bake-off")
)

CEILING_USD = 15.0
STOP_AT_USD = 12.5
PILOT_NAMES = 20
FRESH_READER = "openrouter/openai/gpt-6.1-sol"
FRESH_READER_EFFORT = "high"
FRESH_PER_SIDE = 40

HAIKU = "openrouter/anthropic/claude-haiku-5.5"
LUNA = "openrouter/openai/gpt-6-luna"
LOCAL = "local/deepseek-v4.1-flash"
READINGS = 3
FIELDS = ("name", "unit", "description", "documentation")
SEED = 20261114


# ------------------------------------------------------------------ inputs


def _normalise(value: str) -> str:
    return " ".join((value or "").split())


def _text_digest(item: dict) -> str:
    per_field = [
        hashlib.sha256(_normalise(item.get(field) or "").encode()).hexdigest()[:16]
        for field in FIELDS
    ]
    return hashlib.sha256("|".join(per_field).encode()).hexdigest()[:16]


def _sheet_rounds() -> list[str]:
    return [
        "sn-name-definition-graph-reading",
        "sn-name-definition-catalog-reading",
        "sn-name-definition-fresh-labels",
        "sn-name-definition-second-reading",
    ]


def _read_sheets() -> dict[str, dict[str, dict]]:
    """Every reading sheet's text, by name, one entry per round that read it."""

    found: dict[str, dict[str, dict]] = collections.defaultdict(dict)
    for round_name in _sheet_rounds():
        for path in sorted((EVIDENCE / round_name).glob("*sheet*.json")):
            payload = json.loads(path.read_text())
            items = payload.get("items") if isinstance(payload, dict) else payload
            if not isinstance(items, list) or not items or "name" not in items[0]:
                continue
            for item in items:
                item.setdefault("unit", "")
                found[item["name"]].setdefault(round_name, item)
    return found


def _read_labels() -> dict[str, dict]:
    """The readers' class and quoted evidence per name, from every label file."""

    id_to_name: dict[tuple[str, str], str] = {}
    for round_name in _sheet_rounds():
        for path in sorted((EVIDENCE / round_name).glob("*sheet*.json")):
            payload = json.loads(path.read_text())
            items = payload.get("items") if isinstance(payload, dict) else payload
            if not isinstance(items, list) or not items or "name" not in items[0]:
                continue
            for item in items:
                id_to_name[(round_name, item.get("id"))] = item["name"]
    rows: dict[str, dict] = {}
    for round_name in _sheet_rounds():
        for path in sorted((EVIDENCE / round_name).glob("labels*.json")):
            payload = json.loads(path.read_text())
            if not isinstance(payload, list):
                continue
            for entry in payload:
                name = id_to_name.get((round_name, entry.get("id")))
                if not name:
                    continue
                slot = rows.setdefault(name, {"classes": [], "text": []})
                if entry.get("class"):
                    slot["classes"].append(entry["class"])
                slot["text"].append(
                    " ".join(str(entry.get(key) or "") for key in ("evidence", "reason"))
                )
    return rows


def load_items() -> tuple[list[dict], dict]:
    """The labelled names, each with the text its readers read."""

    verdicts = json.loads(
        (EVIDENCE / "sn-name-definition-graph-reading/reader-verdicts.json").read_text()
    )
    reconcile = json.loads(
        (EVIDENCE / "sn-name-definition-graph-reading/reconciliation.json").read_text()
    )["candidates"]
    digest_of: dict[str, str] = {}
    for table in reconcile.values():
        for name, entry in table.items():
            digest_of[name] = entry["s"] if isinstance(entry, dict) else entry
    sheets = _read_sheets()
    items: list[dict] = []
    left_out: list[dict] = []
    for row in verdicts["rows"]:
        name = row["name"]
        by_round = sheets.get(name) or {}
        chosen = None
        chosen_round = None
        for round_name, candidate in by_round.items():
            if _text_digest(candidate) == digest_of.get(name):
                chosen, chosen_round = candidate, round_name
                break
        if chosen is None and by_round:
            chosen_round = next(iter(by_round))
            chosen = by_round[chosen_round]
        if chosen is None:
            left_out.append({"name": name, "reason": "no reading sheet carries its text"})
            continue
        items.append(
            {
                "id": name,
                "unit": chosen.get("unit") or "",
                "description": chosen.get("description") or "",
                "documentation": chosen.get("documentation") or "",
                "label": "flagged" if row["verdict"] == "flagged" else "stands",
                "text_source": chosen_round,
                "text_matches": _text_digest(chosen) == digest_of.get(name),
            }
        )
    items.sort(key=lambda entry: entry["id"])
    meta = {
        "labels_path": str(
            EVIDENCE / "sn-name-definition-graph-reading/reader-verdicts.json"
        ),
        "labelled": len(verdicts["rows"]),
        "scored": len(items),
        "left_out": left_out,
        "text_sources": dict(collections.Counter(entry["text_source"] for entry in items)),
        "text_mismatch": [entry["id"] for entry in items if not entry["text_matches"]],
    }
    return items, meta


EQUIVALENT = re.compile(r"equivalent[ -]?electron", re.I)
OPERATOR = re.compile(
    r"averag|normalis|normaliz|weight|integral|integrat|aggregat|per unit|"
    r"divided by|ratio of|summed over the|volume-averaged|flux-surface",
    re.I,
)
ONE_OBJECT = re.compile(
    r"\b(one|single|a specific|the)\s+[a-z-]*\s*(coil|module|face|branch|channel|"
    r"phase|instrument|launcher|antenna|probe|segment|surface|pump|port|limiter|"
    r"target|plate|unit|system|grid|patch|beam|line|heater|source|component|"
    r"machine|shot|discharge|wall|species|state)\b",
    re.I,
)


def reader_class_groups(items: list[dict]) -> dict[str, str]:
    """Group the flagged names into the six classes the rewrites are planned by.

    The readers label five sheet classes; their quoted evidence separates two
    kinds those classes fold together. Applied in this order: the readers' own
    unit-misfit and family-parent classes; an equivalent-electron count or rate
    named in the evidence; the readers' narrower-than-name class as a single
    object or mechanism; an average, integral, weighting or normalisation as an
    operator; what remains is shared wording.
    """

    evidence = _read_labels()
    grouping: dict[str, str] = {}
    for item in items:
        if item["label"] != "flagged":
            continue
        row = evidence.get(item["id"], {"classes": [], "text": []})
        classes = {value for value in row["classes"] if value}
        joined = " ".join(row["text"])
        if "unit-does-not-fit-name" in classes:
            grouping[item["id"]] = "unit misfit"
        elif "abstract-parent-as-quantity" in classes:
            grouping[item["id"]] = "family parent"
        elif EQUIVALENT.search(joined):
            grouping[item["id"]] = "equivalent-electron"
        elif "definition-narrower-than-name" in classes or ONE_OBJECT.search(joined):
            grouping[item["id"]] = "single object"
        elif OPERATOR.search(joined):
            grouping[item["id"]] = "operator"
        else:
            grouping[item["id"]] = "shared wording"
    return grouping


# ------------------------------------------------------------------- calls


def _plan() -> list[dict]:
    from imas_codex.settings import (
        get_sn_review_docs_models,
        get_sn_review_reasoning_effort,
    )

    return [
        {
            "key": "seat",
            "label": "current seat (Sonnet 5.5, batches of ten)",
            "model": get_sn_review_docs_models()[0],
            "readings": 1,
            "batch": 10,
            "concurrency": 6,
            "effort": get_sn_review_reasoning_effort(),
            "paid": True,
        },
        {
            "key": "haiku",
            "label": "Haiku 5.5",
            "model": HAIKU,
            "readings": READINGS,
            "batch": 1,
            "concurrency": 16,
            "effort": None,
            "paid": True,
        },
        {
            "key": "luna",
            "label": "GPT-6 Luna",
            "model": LUNA,
            "readings": READINGS,
            "batch": 1,
            "concurrency": 16,
            "effort": None,
            "paid": True,
        },
        {
            "key": "local",
            "label": "local DS4.1-flash",
            "model": LOCAL,
            "readings": READINGS,
            "batch": 1,
            "concurrency": 24,
            "effort": None,
            "paid": False,
        },
    ]


def _system_prompt() -> str:
    from imas_codex.llm.prompt_loader import render_prompt

    rendered = render_prompt("sn/review_name_definition_system", {})
    return rendered if isinstance(rendered, str) else str(rendered)


async def _judge(batch: list[dict], model: str, effort: str | None):
    from imas_codex.standard_names.review.name_definition_seat import _judge_batch

    return await _judge_batch(
        batch, seat=model, system_prompt=_system_prompt(), reasoning_effort=effort
    )


def _progress_path(arm: str, reading: int) -> Path:
    STATE.mkdir(parents=True, exist_ok=True)
    return STATE / f"{arm}-r{reading}.jsonl"


def _load_progress(arm: str, reading: int) -> dict[str, dict]:
    path = _progress_path(arm, reading)
    done: dict[str, dict] = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                record = json.loads(line)
                done[record["name"]] = record
    return done


class Spend:
    def __init__(self, limit: float) -> None:
        self.total = 0.0
        self.limit = limit

    def add(self, cost: float) -> None:
        self.total += float(cost or 0.0)
        if self.total > self.limit:
            raise RuntimeError(
                f"spend ceiling reached: {self.total:.2f} USD past {self.limit:.2f}"
            )


async def _append(path: Path, lock: asyncio.Lock, records: list[dict]) -> None:
    async with lock:
        with path.open("a") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")


async def run_pass(arm: dict, items: list[dict], spend: Spend, *, reading: int = 0) -> dict:
    key = arm["key"]
    done = _load_progress(key, reading)
    todo = [item for item in items if item["id"] not in done]
    size = arm["batch"]
    batches = [todo[i : i + size] for i in range(0, len(todo), size)]
    gate = asyncio.Semaphore(arm["concurrency"])
    lock = asyncio.Lock()
    path = _progress_path(key, reading)
    started = time.time()

    async def one(batch: list[dict], attempt: int = 0) -> None:
        async with gate:
            payload = [
                {
                    "id": item["id"],
                    "unit": item["unit"],
                    "description": item["description"],
                    "documentation": item["documentation"],
                }
                for item in batch
            ]
            began = time.time()
            try:
                verdicts, cost = await _judge(payload, arm["model"], arm["effort"])
            except Exception as exc:  # a provider failure is reported, never hidden
                print(f"[{key} r{reading}] batch failed: {exc}", flush=True)
                if attempt < 2:
                    await asyncio.sleep(2 * (attempt + 1))
                    return await one(batch, attempt + 1)
                verdicts, cost = [], 0.0
            elapsed = time.time() - began
            if not verdicts:
                verdicts = [
                    {
                        "name": item["id"],
                        "verdict": "unanswered",
                        "qualifier": None,
                        "reasoning": "no verdict returned",
                    }
                    for item in batch
                ]
            share = cost / max(1, len(verdicts))
            await _append(
                path,
                lock,
                [
                    {
                        "arm": key,
                        "reading": reading,
                        "name": verdict["name"],
                        "verdict": verdict["verdict"],
                        "qualifier": verdict.get("qualifier"),
                        "reasoning": (verdict.get("reasoning") or "")[:300],
                        "cost": share,
                        "seconds": elapsed / max(1, len(verdicts)),
                    }
                    for verdict in verdicts
                ],
            )
            if cost:
                spend.add(cost)

    await asyncio.gather(*(one(batch) for batch in batches))
    return {
        "arm": key,
        "reading": reading,
        "calls": len(batches),
        "names": sum(len(batch) for batch in batches),
        "seconds": round(time.time() - started, 1),
    }


async def run_fresh_reader(items: list[dict], drawn: list[str], spend: Spend) -> dict:
    by_id = {item["id"]: item for item in items}
    done = _load_progress("fresh", 0)
    todo = [by_id[name] for name in drawn if name in by_id and name not in done]
    gate = asyncio.Semaphore(8)
    lock = asyncio.Lock()
    path = _progress_path("fresh", 0)

    async def one(item: dict) -> None:
        async with gate:
            payload = [
                {
                    "id": item["id"],
                    "unit": item["unit"],
                    "description": item["description"],
                    "documentation": item["documentation"],
                }
            ]
            verdicts, cost = [], 0.0
            for attempt in range(3):
                try:
                    verdicts, cost = await _judge(
                        payload, FRESH_READER, FRESH_READER_EFFORT
                    )
                    break
                except Exception as exc:
                    print(f"[fresh] call failed: {exc}", flush=True)
                    await asyncio.sleep(2 * (attempt + 1))
            record = (
                verdicts[0]
                if verdicts
                else {"name": item["id"], "verdict": "unanswered", "qualifier": None}
            )
            await _append(
                path,
                lock,
                [
                    {
                        "arm": "fresh",
                        "reading": 0,
                        "name": item["id"],
                        "verdict": record["verdict"],
                        "qualifier": record.get("qualifier"),
                        "reasoning": (record.get("reasoning") or "")[:300],
                        "cost": cost,
                        "seconds": 0.0,
                    }
                ],
            )
            if cost:
                spend.add(cost)

    await asyncio.gather(*(one(item) for item in todo))
    return {"called": len(todo), "spend": round(spend.total, 4)}


# ----------------------------------------------------------------- scoring


def wilson(successes: int, total: int) -> list[float | None]:
    if not total:
        return [None, None]
    z = 1.96
    phat = successes / total
    denom = 1 + z * z / total
    centre = (phat + z * z / (2 * total)) / denom
    half = z * math.sqrt(phat * (1 - phat) / total + z * z / (4 * total * total)) / denom
    return [round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4)]


def _panel(arm: str) -> dict[str, str]:
    readings = [_load_progress(arm, index) for index in range(READINGS)]
    names = {
        name
        for reading in readings
        for name, row in reading.items()
        if row["verdict"] != "unanswered"
    }
    decided: dict[str, str] = {}
    for name in names:
        present = [reading[name] for reading in readings if name in reading]
        if len(present) >= 2:
            flags = sum(1 for row in present if row["verdict"] == "flag")
            decided[name] = "flag" if flags >= 2 else "pass"
    return decided


def _first(arm: str) -> dict[str, str]:
    return {
        name: row["verdict"]
        for name, row in _load_progress(arm, 0).items()
        if row["verdict"] != "unanswered"
    }


def _mixed() -> dict[str, str]:
    readings = [_first(arm) for arm in ("haiku", "luna", "local")]
    decided: dict[str, str] = {}
    for name in set().union(*readings):
        present = [reading[name] for reading in readings if name in reading]
        if len(present) >= 2:
            decided[name] = "flag" if sum(1 for v in present if v == "flag") >= 2 else "pass"
    return decided


def score(items: list[dict], verdict: dict[str, str], grouping: dict[str, str]) -> dict:
    tp = fp = tn = fn = 0
    unanswered = 0
    per_class: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0])
    for item in items:
        name, label = item["id"], item["label"]
        if name not in verdict:
            unanswered += 1
            continue
        flagged = verdict[name] == "flag"
        if label == "flagged":
            cell = per_class[grouping.get(name, "shared wording")]
            cell[1] += 1
            if flagged:
                tp += 1
                cell[0] += 1
            else:
                fn += 1
        elif flagged:
            fp += 1
        else:
            tn += 1
    scored = tp + fp + tn + fn
    return {
        "scored": scored,
        "unanswered": unanswered,
        "flagged_total": tp + fn,
        "stands_total": tn + fp,
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "true_negative": tn,
        "recall": round(tp / (tp + fn), 4) if tp + fn else None,
        "recall_ci": wilson(tp, tp + fn),
        "precision": round(tp / (tp + fp), 4) if tp + fp else None,
        "precision_ci": wilson(tp, tp + fp),
        "agreement": round((tp + tn) / scored, 4) if scored else None,
        "agreement_ci": wilson(tp + tn, scored),
        "by_class": {
            key: {
                "flagged": totals[1],
                "found": totals[0],
                "recall": round(totals[0] / totals[1], 4) if totals[1] else None,
            }
            for key, totals in sorted(per_class.items())
        },
    }


def spend_of(arm: str, readings: int = READINGS) -> dict:
    total = seconds = 0.0
    answered = unanswered = 0
    for index in range(readings):
        for row in _load_progress(arm, index).values():
            total += row.get("cost") or 0.0
            seconds += row.get("seconds") or 0.0
            if row["verdict"] == "unanswered":
                unanswered += 1
            else:
                answered += 1
    return {
        "cost_usd": round(total, 4),
        "per_name_usd": round(total / answered, 6) if answered else None,
        "wall_seconds": round(seconds, 1),
        "per_name_seconds": round(seconds / answered, 3) if answered else None,
        "answered": answered,
        "unanswered": unanswered,
    }


def repeat(arm: str) -> dict:
    readings = [_load_progress(arm, index) for index in range(READINGS)]
    common = set(readings[0]) if readings else set()
    for reading in readings[1:]:
        common &= set(reading)
    names = {
        name
        for name in common
        if all(readings[index][name]["verdict"] != "unanswered" for index in range(READINGS))
    }
    unanimous = sum(
        1
        for name in names
        if len({readings[index][name]["verdict"] for index in range(READINGS)}) == 1
    )
    pairs = []
    for left, right in ((0, 1), (0, 2), (1, 2)):
        same = sum(
            1
            for name in names
            if readings[left][name]["verdict"] == readings[right][name]["verdict"]
        )
        pairs.append(round(same / len(names), 4) if names else None)
    usable = [value for value in pairs if value is not None]
    return {
        "compared": len(names),
        "unanimous": round(unanimous / len(names), 4) if names else None,
        "pairwise": pairs,
        "mean_pairwise": round(sum(usable) / len(usable), 4) if usable else None,
    }


def build_report(items: list[dict], meta: dict) -> dict:
    grouping = reader_class_groups(items)
    verdicts = {
        "seat": ("current seat, one reading (batched as it runs)", _first("seat")),
        "haiku-single": ("Haiku 5.5, first reading", _first("haiku")),
        "haiku-two-of-three": ("Haiku 5.5, two of three", _panel("haiku")),
        "luna-single": ("GPT-6 Luna, first reading", _first("luna")),
        "luna-two-of-three": ("GPT-6 Luna, two of three", _panel("luna")),
        "local-single": ("local DS4.1-flash, first reading", _first("local")),
        "local-two-of-three": ("local two of three, upper bound", _panel("local")),
        "mixed-two-of-three": ("mixed panel, two of three", _mixed()),
    }
    costs = {
        "seat": spend_of("seat", 1),
        "haiku-single": spend_of("haiku", 1),
        "haiku-two-of-three": spend_of("haiku", READINGS),
        "luna-single": spend_of("luna", 1),
        "luna-two-of-three": spend_of("luna", READINGS),
        "local-single": spend_of("local", 1),
        "local-two-of-three": spend_of("local", READINGS),
    }
    mixed_cost = sum(spend_of(arm, 1)["cost_usd"] for arm in ("haiku", "luna", "local"))
    mixed_names = min(
        spend_of(arm, 1)["answered"] for arm in ("haiku", "luna", "local")
    )
    costs["mixed-two-of-three"] = {
        "cost_usd": round(mixed_cost, 4),
        "per_name_usd": round(mixed_cost / mixed_names, 6) if mixed_names else None,
        "wall_seconds": round(
            sum(spend_of(arm, 1)["wall_seconds"] for arm in ("haiku", "luna", "local")), 1
        ),
        "per_name_seconds": round(
            sum(spend_of(arm, 1)["wall_seconds"] for arm in ("haiku", "luna", "local"))
            / (mixed_names or 1),
            3,
        ),
        "answered": mixed_names,
        "unanswered": 0,
        "note": "one reading each from Haiku, Luna and the local model",
    }

    arms: dict[str, dict] = {}
    for key, (label, verdict) in verdicts.items():
        result = score(items, verdict, grouping)
        result["label"] = label
        result["cost"] = costs[key]
        if key.endswith("two-of-three") and not key.startswith("mixed"):
            result["repeat_agreement"] = repeat(key.split("-")[0])
        arms[key] = result

    spent = sum(
        spend_of(arm, 1 if arm == "seat" else READINGS)["cost_usd"]
        for arm in ("seat", "haiku", "luna", "local")
    ) + spend_of("fresh", 1)["cost_usd"]
    report = {
        "schema": "name-definition-reader-placeholder",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "population": {
            "labelled": meta["labelled"],
            "scored": meta["scored"],
            "left_out": meta["left_out"],
            "labels_path": meta["labels_path"],
            "text_sources": meta["text_sources"],
            "text_mismatch": meta["text_mismatch"],
        },
        "prompt": "imas_codex/llm/prompts/sn/review_name_definition_system.md",
        "spend_usd": round(spent, 4),
        "ceiling_usd": CEILING_USD,
        "class_counts": dict(collections.Counter(grouping.values())),
        "arms": arms,
        "fresh_reader": fresh_summary(items),
        "readings": _all_readings(),
    }
    report["schema"] = "name-definition-reader-bake-off"
    report["recommendation"] = recommendation(report)
    return report


def _all_readings() -> dict:
    """Every call's verdict, keyed by arm and reading, as the run recorded them."""

    recorded = {}
    for arm in ("seat", "haiku", "luna", "local"):
        for index in range(READINGS):
            rows = _load_progress(arm, index)
            if rows:
                recorded[f"{arm}-r{index}"] = {
                    name: {
                        "verdict": row["verdict"],
                        "qualifier": row.get("qualifier"),
                        "cost_usd": row.get("cost_usd") or row.get("cost"),
                        "seconds": row.get("seconds"),
                    }
                    for name, row in rows.items()
                }
    return recorded


def fresh_summary(items: list[dict]) -> dict:
    fresh = _load_progress("fresh", 0)
    if not fresh:
        return {}
    labels = {item["id"]: item["label"] for item in items}
    haiku = _panel("haiku")
    rows = []
    for name, record in fresh.items():
        if name not in labels:
            continue
        fresh_flag = record["verdict"] == "flag"
        label_flagged = labels[name] == "flagged"
        haiku_flag = haiku.get(name) == "flag"
        side = (
            "both"
            if fresh_flag == label_flagged and fresh_flag == haiku_flag
            else "label"
            if fresh_flag == label_flagged
            else "haiku"
            if fresh_flag == haiku_flag
            else "neither"
        )
        rows.append(
            {
                "name": name,
                "label": labels[name],
                "haiku_two_of_three": "flag" if haiku_flag else "stands",
                "fresh_reader_verdict": "flag" if fresh_flag else record["verdict"],
                "sides_with": side,
            }
        )
    counts = collections.Counter(row["sides_with"] for row in rows)
    disagreements = collections.Counter(
        row["sides_with"]
        for row in rows
        if (row["sides_with"] in {"label", "haiku"})
    )
    return {
        "reader_model": FRESH_READER,
        "effort": FRESH_READER_EFFORT,
        "answered": len(rows),
        "sides_with": dict(counts),
        "disagreements_only": dict(disagreements),
        "rows": sorted(rows, key=lambda row: row["name"]),
    }


def recommendation(report: dict) -> dict:
    arms = report["arms"]
    choice = "mixed-two-of-three"
    arm = arms[choice]
    fresh = report.get("fresh_reader") or {}
    sides = fresh.get("sides_with") or {}
    total_flagged = arm["flagged_total"]
    return {
        "arm": choice,
        "label": arm["label"],
        "cost_usd": arm["cost"]["cost_usd"],
        "cost_per_name_usd": arm["cost"]["per_name_usd"],
        "cost_for_set_usd": round(
            (arm["cost"]["per_name_usd"] or 0) * report["population"]["scored"], 2
        ),
        "miss_rate": (
            round(arm["false_negative"] / total_flagged, 4) if total_flagged else None
        ),
        "caught_of_flagged": arm["true_positive"],
        "flagged_total": total_flagged
        or sum(report["class_counts"].values()),
        "fresh_reader_sides_with_label": sides.get("label", 0),
        "fresh_reader_sides_with_haiku": sides.get("haiku", 0),
        "notes": [
            "the mixed panel is one reading each from Haiku, Luna and the local "
            "model, decided two of three, so no single vendor's family bias sets "
            "the answer",
            "the local model is graded against blind readers on its own lane, so "
            "its score is an upper bound and reads beside its number, not as one",
        ],
    }


# ------------------------------------------------------------------ output


def _pct(value) -> str:
    return "—" if value is None else f"{100 * value:.1f}%"


def _usd(value, digits: int = 2) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def write_html(report: dict) -> None:
    arms = report["arms"]
    order = [
        "seat", "haiku-single", "haiku-two-of-three", "luna-single",
        "luna-two-of-three", "local-single", "local-two-of-three",
        "mixed-two-of-three",
    ]
    table = []
    for key in order:
        arm, cost = arms[key], arms[key]["cost"]
        table.append(
            "<tr><td>{label}</td><td>{recall} [{rl}, {rh}]</td>"
            "<td>{precision} [{pl}, {ph}]</td><td>{agreement} [{al}, {ah}]</td>"
            "<td>{scored}</td><td>{cost}</td><td>{per_name}</td>"
            "<td>{seconds}</td></tr>".format(
                label=arm["label"],
                recall=_pct(arm["recall"]), rl=_pct(arm["recall_ci"][0]),
                rh=_pct(arm["recall_ci"][1]),
                precision=_pct(arm["precision"]), pl=_pct(arm["precision_ci"][0]),
                ph=_pct(arm["precision_ci"][1]),
                agreement=_pct(arm["agreement"]), al=_pct(arm["agreement_ci"][0]),
                ah=_pct(arm["agreement_ci"][1]),
                scored=(
                    str(arm["scored"])
                    if not arm["unanswered"]
                    else f"{arm['scored']} of {arm['scored'] + arm['unanswered']}"
                ),
                cost=_usd(cost.get("cost_usd")),
                per_name=_usd(cost.get("per_name_usd"), 5),
                seconds=_usd(cost.get("per_name_seconds"), 2),
            )
        )
    classes = sorted({cls for arm in arms.values() for cls in arm["by_class"]})
    class_head = "".join(f"<th>{cls}</th>" for cls in classes)
    class_totals = "".join(
        f"<td>{report['class_counts'].get(cls, 0)}</td>" for cls in classes
    )
    class_rows = []
    for key in order:
        cells = "".join(
            f"<td>{_pct(arms[key]['by_class'].get(cls, {}).get('recall'))}</td>"
            for cls in classes
        )
        class_rows.append(f"<tr><td>{arms[key]['label']}</td>{cells}</tr>")
    repeat_rows = "".join(
        "<tr><td>{label}</td><td>{unanimous}</td><td>{pairs}</td></tr>".format(
            label=arms[key]["label"],
            unanimous=_pct(arms[key]["repeat_agreement"]["unanimous"]),
            pairs=", ".join(
                _pct(value) for value in arms[key]["repeat_agreement"]["pairwise"]
            ),
        )
        for key in ("haiku-two-of-three", "luna-two-of-three", "local-two-of-three")
        if arms[key].get("repeat_agreement")
    )
    fresh = report.get("fresh_reader") or {}
    sides = fresh.get("sides_with") or {}
    fresh_rows = "".join(
        "<tr><td><code>{name}</code></td><td>{label}</td><td>{haiku}</td>"
        "<td>{fresh}</td><td>{side}</td></tr>".format(
            name=row["name"], label=row["label"], haiku=row["haiku_two_of_three"],
            fresh=row["fresh_reader_verdict"], side=row["sides_with"],
        )
        for row in (fresh.get("rows") or [])[:40]
    )
    rec = report["recommendation"]
    population = report["population"]
    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>Which reader should check a name against its definition</title>
<style>
body{{font:15px/1.55 system-ui,sans-serif;max-width:1100px;margin:2rem auto;padding:0 1rem;color:#111}}
table{{border-collapse:collapse;margin:1rem 0;font-size:13px}}
th,td{{border-bottom:1px solid #ddd;padding:.3rem .55rem;text-align:left}}
th{{border-bottom:2px solid #333}}
code{{font-size:12px}}
.note{{color:#555;font-size:13px}}
</style></head><body>
<h1>Which reader should check a name against its definition</h1>
<p class="note">Scored on {population['scored']} of {population['labelled']} labelled
export candidates, on the name, definition and unit as the blind readers read
them. {len(population['left_out'])} names whose read text could not be recovered
are listed in the JSON and left out.</p>
<figure>
<img src="/imas-codex/figures/sn-name-definition-readers/{FIG_NAME}"
 alt="flagged recall against flagged precision by arm">
<figcaption>Flagged-recall against flagged-precision, one point per arm with a
95&nbsp;% interval on both axes. Hollow grey points are the local model, graded
against its own reading family and therefore an upper bound, not a measurement.</figcaption>
</figure>
<h2>Agreement with the settled verdicts</h2>
<table>
<tr><th>arm</th><th>flagged recall [95%]</th><th>flagged precision [95%]</th>
<th>overall agreement [95%]</th><th>names scored</th><th>spend USD</th>
<th>USD per name</th><th>s per name</th></tr>
{''.join(table)}
</table>
<h2>Recall by the readers' class</h2>
<p class="note">Six classes, one per flagged name, grouped from the readers'
sheet class and their quoted evidence; the counts row is how many flagged names
fall in each.</p>
<table>
<tr><th>arm</th>{class_head}</tr>
<tr><td class="note">flagged names</td>{class_totals}</tr>
{''.join(class_rows)}
</table>
<h2>Do repeat readings agree</h2>
<table>
<tr><th>arm</th><th>all three agree</th><th>pairwise agreement (r1r2, r1r3, r2r3)</th></tr>
{repeat_rows}
</table>
<h2>Fresh reader of another family</h2>
<p class="note">{fresh.get('reader_model','')} at {fresh.get('effort','')} effort,
blind: it saw the name, definition and unit only, in shuffled order, and no label
or arm verdict. {fresh.get('answered',0)} answered. Sides with the label
{sides.get('label',0)}, with Haiku two-of-three {sides.get('haiku',0)}, with both
{sides.get('both',0)}, with neither {sides.get('neither',0)}.</p>
<table>
<tr><th>name</th><th>label</th><th>Haiku 2/3</th><th>fresh</th><th>sides"</th></tr>
{fresh_rows}
</table>
<h2>Recommendation</h2>
<p>Run <strong>{rec['label']}</strong>. It costs USD
{_usd(rec['cost_usd'])} on these {population['scored']} names
({_usd(rec['cost_per_name_usd'], 5)} per name), and at that miss rate it finds
<strong>{rec['caught_of_flagged']} of {rec['flagged_total']}</strong> flagged
names, missing {_pct(rec['miss_rate'])} of them.</p>
<p>{' '.join(rec['notes'])}</p>
<h2>Readings</h2>
<p class="note">Every reading, its class and its own cost is in
<code>name-definition-reader-bake-off.json</code>.</p>
</body></html>
"""
    OUT_HTML.write_text(html)


def write_figure(report: dict) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    try:
        plt.style.use("data-ink")
    except Exception:
        pass
    colours = {"seat": "#1a1a1a", "haiku": "#2367b8", "luna": "#7a3fa8",
               "local": "#929292", "mixed": "#2e8b57"}
    points = [
        ("seat", "current seat", "seat", False),
        ("haiku-single", "Haiku, one reading", "haiku", False),
        ("haiku-two-of-three", "Haiku, two of three", "haiku", False),
        ("luna-single", "Luna, one reading", "luna", False),
        ("luna-two-of-three", "Luna, two of three", "luna", False),
        ("local-single", "local, one reading (upper bound)", "local", True),
        ("local-two-of-three", "local, two of three (upper bound)", "local", True),
        ("mixed-two-of-three", "mixed panel, two of three", "mixed", False),
    ]
    fig, ax = plt.subplots(figsize=(14, 7.2), dpi=100)
    handles = []
    for key, label, colour, upper in points:
        arm = report["arms"][key]
        if arm["precision"] is None or arm["recall"] is None:
            continue
        hollow = key.endswith("-single") and key != "seat"
        x, y = arm["precision"], arm["recall"]
        left = arm["precision_ci"][0] if arm["precision_ci"][0] is not None else x
        right = arm["precision_ci"][1] if arm["precision_ci"][1] is not None else x
        low = arm["recall_ci"][0] if arm["recall_ci"][0] is not None else y
        high = arm["recall_ci"][1] if arm["recall_ci"][1] is not None else y
        ax.errorbar(
            x, y, xerr=[[x - left], [right - x]],
            yerr=[[y - low], [high - y]], fmt="o", markersize=9,
            color=colours[colour], ecolor=colours[colour], elinewidth=1.2,
            capsize=3,
            markerfacecolor="white" if hollow else colours[colour],
            markeredgewidth=1.6,
            zorder=3 if key == "mixed-two-of-three" else 2,
        )
        handles.append(
            plt.Line2D(
                [], [], marker="o", linestyle="none", markersize=9,
                color=colours[colour], markerfacecolor="white" if hollow else colours[colour],
                markeredgewidth=1.6, label=label,
            )
        )
    ax.set_xlabel("flagged precision")
    ax.set_ylabel("flagged recall")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.04, 0.99),
        ncol=2,
        frameon=False,
        fontsize=13,
        handletextpad=0.4,
        columnspacing=1.2,
    )
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / FIG_NAME
    fig.savefig(out)
    plt.close(fig)
    return out


# -------------------------------------------------------------------- main


async def run_all(
    only: list[str] | None = None,
    concurrency: int | None = None,
    readings: list[int] | None = None,
) -> None:
    items, meta = load_items()
    print(
        f"labelled {meta['labelled']} scored {meta['scored']} "
        f"left_out {len(meta['left_out'])} sources {meta['text_sources']}",
        flush=True,
    )
    if meta["text_mismatch"]:
        print(f"text digest mismatch on {len(meta['text_mismatch'])} names", flush=True)
    plan = _plan()
    spend = Spend(CEILING_USD)
    if only and set(only) == {"fresh"}:
        drawn = _fresh_draw(items)
        print(f"fresh {await run_fresh_reader(items, drawn, spend)}", flush=True)
        print(f"FRESH_SPEND_USD {spend.total:.4f}", flush=True)
        return
    if only:
        plan = [arm for arm in plan if arm["key"] in only]
        for arm in plan:
            if concurrency:
                arm["concurrency"] = concurrency
        wanted = readings if readings is not None else range(max(arm["readings"] for arm in plan))
        for arm in plan:
            for reading in wanted:
                print(f"pass {await run_pass(arm, items, spend, reading=reading)}", flush=True)
        print(f"ONLY_SPEND_USD {spend.total:.4f}", flush=True)
        return
    spend = Spend(STOP_AT_USD)
    pilot = _pilot_draw(items, PILOT_NAMES)
    for arm in plan:
        print(f"pilot {await run_pass(arm, pilot, spend, reading=0)}", flush=True)
    per_name = spend.total / max(1, len(plan) * PILOT_NAMES)
    projection = per_name * len(items) * len(plan)
    print(f"PILOT_SPEND_USD {spend.total:.4f}", flush=True)
    print(f"PROJECTION_USD {projection:.2f}", flush=True)
    if projection > STOP_AT_USD:
        print(f"PROJECTION_ABOVE_{STOP_AT_USD} — stopping", flush=True)
        return
    for arm in plan:
        for reading in range(arm["readings"]):
            print(f"pass {await run_pass(arm, items, spend, reading=reading)}", flush=True)
    drawn = _fresh_draw(items)
    print(f"fresh {await run_fresh_reader(items, drawn, spend)}", flush=True)
    print(f"TOTAL_SPEND_USD {spend.total:.4f}", flush=True)


def _pilot_draw(items: list[dict], count: int) -> list[dict]:
    rng = random.Random(SEED)
    flagged = [item for item in items if item["label"] == "flagged"]
    stands = [item for item in items if item["label"] == "stands"]
    half = count // 2
    return rng.sample(stands, min(count - half, len(stands))) + rng.sample(
        flagged, min(half, len(flagged))
    )


def _fresh_draw(items: list[dict]) -> list[str]:
    labels = {item["id"]: item["label"] for item in items}
    haiku = _panel("haiku")
    disagree_flag = [
        name for name, label in labels.items()
        if label == "flagged" and haiku.get(name) == "pass"
    ]
    disagree_pass = [
        name for name, label in labels.items()
        if label == "stands" and haiku.get(name) == "flag"
    ]
    agree = [
        name for name, label in labels.items()
        if (haiku.get(name) == "flag") == (label == "flagged")
    ]
    rng = random.Random(SEED)
    drawn = []
    drawn += rng.sample(disagree_flag, min(FRESH_PER_SIDE, len(disagree_flag)))
    drawn += rng.sample(disagree_pass, min(FRESH_PER_SIDE, len(disagree_pass)))
    drawn += rng.sample(agree, min(FRESH_PER_SIDE, len(agree)))
    rng.shuffle(drawn)
    print(
        f"fresh draw: {len(disagree_flag)} label-flagged/Haiku-pass, "
        f"{len(disagree_pass)} label-stands/Haiku-flag, {len(agree)} agreeing; "
        f"drawn {len(drawn)}",
        flush=True,
    )
    return drawn


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode", nargs="?", default="run", choices=["run", "report", "negative"]
    )
    parser.add_argument("--only", default="", help="comma-separated arm keys to run")
    parser.add_argument("--concurrency", type=int, default=None)
    parser.add_argument("--numbers", default="", help="comma-separated reading indexes")
    options = parser.parse_args()
    if options.mode == "run":
        only = [key.strip() for key in options.only.split(",") if key.strip()]
        wanted = [int(v) for v in options.numbers.split(",") if v.strip()] or None
        asyncio.run(run_all(only or None, options.concurrency, wanted))
        if only:
            return
    items, meta = load_items()
    if options.mode == "negative":
        verdict = {item["id"]: "pass" for item in items}
        scored = score(items, verdict, reader_class_groups(items))
        print(
            f"NEGATIVE_CONTROL {"pass for every name"} over {scored['scored']} names: "
            f"flagged-recall {scored['recall']} "
            f"({scored['true_positive']} of {scored['flagged_total']}), "
            f"flagged-precision {scored['precision']}, "
            f"agreement {scored['agreement']}",
            flush=True,
        )
        return
    report = build_report(items, meta)
    OUT_JSON.write_text(json.dumps(report, indent=1, default=str))
    write_html(report)
    path = write_figure(report)
    print(f"WROTE {OUT_JSON} {OUT_HTML} {path}", flush=True)


if __name__ == "__main__":
    main()