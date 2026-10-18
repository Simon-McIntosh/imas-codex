"""Compare the two refine instruction arms over a held-out cohort, blindly.

    uv run --no-sync python srtj-refine-ab-run.py                # the A/B run
    uv run --no-sync python srtj-refine-ab-run.py --check-sheet   # sheet leak check

Cohort: REFINED_FROM history, so each case is a name that was refined from an
ancestor and the ancestor's context and critique are the input.  Cases whose
name or ancestor appears in a golden fixture are excluded.  Each surviving
ancestor is judged on the name axis through ``review_names`` with its sibling
names, at that function's own seat; a case enters the cohort when its ancestor
failed at least one check and carries a non-empty critique.

Arms: each case is refined twice through the production prompt builder and the
production refine seat, once with the ancestor's comments driving the
instruction and once with its failed checks (``instruction_source``).  The
refine judge is deliberately not run — the outputs are labelled blind instead.

The blind sheet carries only the DD source context, the ancestor name and the
refined name and description.  The sheet-to-case-and-arm key is written to the
run directory, never into the repository.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
import time as _time
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
GOLDEN_DIR = HERE.parent / "sn-judgment-evaluation"
PLANTED_GOLDEN = HERE / "golden-documentation-planted.json"
RESULTS = HERE / "srtj-refine-ab-run-results.json"
SHEET = HERE / "srtj-refine-ab-label-sheet.json"
RUN_DIR = Path(
    os.environ.get(
        "RECKON_RUN_DIR",
        "/home/ITER/mcintos/.config/reckon/crew/runs/"
        "r-20261018T101616902704-srtj-refine-ab-run",
    )
)
KEY_PATH = RUN_DIR / "srtj-refine-ab-key.json"
RECEIPTS = RUN_DIR / "srtj-refine-receipts.jsonl"

SEED = 20261018
SHEET_SEED = 20261019
COHORT_SIZE = 120
JUDGE_POOL = 160
CEILING_USD = 33.0
HEADROOM_USD = 1.2
CONCURRENCY = 6
ARMS = ("comments", "checks")

SEGMENTS = (
    "physical_base",
    "subject",
    "component",
    "coordinate",
    "position",
    "process",
)

LEAK_FIELDS = frozenset(
    {
        "arm",
        "critique",
        "checks",
        "failed_checks",
        "reason",
        "score",
        "jev_score",
        "prior_score",
        "comments",
        "evidence",
        "criterion",
        "segment",
        "instruction_source",
        "model",
    }
)
ALLOWED_ROW_FIELDS = frozenset(
    {
        "sheet_id",
        "source_path",
        "source_description",
        "source_unit",
        "ancestor_name",
        "refined_name",
        "refined_description",
    }
)


def log(message: str) -> None:
    print(message, flush=True)


def save_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=1, default=str))
    tmp.replace(path)


def golden_units() -> set[str]:
    """Every string a golden fixture carries, a superset of its name units.

    The fixtures key a name unit under ``unit``, ``canonical_unit`` or
    ``copy_group`` and some pack a pattern and a name into one ``pattern::name``
    string, so every string and every ``::`` part is collected.
    """
    units: set[str] = set()

    def walk(node) -> None:
        if isinstance(node, str):
            units.add(node)
            units.update(part for part in node.split("::") if part)
        elif isinstance(node, dict):
            for key, value in node.items():
                units.add(str(key))
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    for path in sorted(GOLDEN_DIR.glob("golden-*.json")) + [PLANTED_GOLDEN]:
        if path.exists():
            walk(json.loads(path.read_text(encoding="utf-8")))
    return units


def ancestor_items(cases: list[dict], gc) -> list[dict]:
    """One review item per case, enriched the way production enriches them."""
    from imas_codex.standard_names.review.pipeline import (
        _fetch_review_dd_context,
        _fetch_review_derived_children,
    )

    rows = gc.query(
        """
        UNWIND $ids AS id
        OPTIONAL MATCH (sn:StandardName {id:id})
        RETURN id, sn.description AS description, sn.documentation AS documentation,
               sn.unit AS unit, sn.kind AS kind, sn.source_paths AS source_paths,
               sn.physics_domain AS physics_domain, sn.physical_base AS physical_base,
               sn.subject AS subject, sn.component AS component,
               sn.coordinate AS coordinate, sn.position AS position,
               sn.process AS process
        """,
        ids=[c["prior_name"] for c in cases],
    )
    by_id = {r["id"]: r for r in rows}

    roster = gc.query(
        "MATCH (sn:StandardName) WHERE sn.id IS NOT NULL "
        "RETURN sn.id AS id, sn.description AS description, "
        "sn.physical_base AS physical_base LIMIT 6000"
    )
    if not any(r["id"] == "ion_average_temperature" for r in roster):
        raise RuntimeError(
            "control name absent from the roster scan, so the scan is not the "
            "whole catalogue and the sibling names would be wrong"
        )
    by_base = defaultdict(list)
    for row in roster:
        if row["physical_base"]:
            by_base[row["physical_base"]].append(row)

    items = []
    for case in cases:
        row = by_id.get(case["prior_name"]) or {}
        item = {
            "id": case["prior_name"],
            "description": row.get("description") or case["description"],
            "documentation": row.get("documentation"),
            "unit": row.get("unit") or case["unit"],
            "kind": row.get("kind") or case["data_type"],
            "source_paths": row.get("source_paths") or case["source_paths"],
            "physics_domain": row.get("physics_domain") or case["physics_domain"],
        }
        for segment in SEGMENTS:
            item[segment] = row.get(segment)
        tokens = set(str(item["id"]).split("_"))
        peers = by_base.get(row.get("physical_base") or "", [])
        ranked = sorted(
            (p for p in peers if p["id"] != item["id"]),
            key=lambda p: (-len(tokens & set(p["id"].split("_"))), p["id"]),
        )[:5]
        item["sibling_names"] = [
            {"id": p["id"], "description": p["description"]} for p in ranked
        ]
        item["existing_names"] = [p["id"] for p in ranked]
        item["nearby_existing_names"] = [p["id"] for p in ranked]
        items.append(item)

    _fetch_review_dd_context(items)
    _fetch_review_derived_children(items)
    return items


def failed_check_records(record: dict) -> list[dict]:
    """Render one ancestor review's failed checks as refine instructions."""
    out = []
    name = (record.get("state") or {}).get("name") or {}
    for check_id, check in (record.get("checks") or {}).items():
        if not check.get("failed"):
            continue
        question = check.get("question") or {}
        answer = check.get("answer") or {}
        if "choice" in answer:
            evidence = f"chosen: {answer.get("choice")}"
        elif "noul" in answer:
            evidence = f"failure probability {float(answer['noul']):.2f}"
        elif check.get("audit_finding"):
            evidence = f"audit finding: {check['audit_finding']}"
        else:
            evidence = ""
        out.append(
            {
                "id": check_id,
                "criterion": question.get("instructions")
                or check.get("audit_finding")
                or check_id,
                "evidence": evidence,
                "segment": name.get("id") or "",
            }
        )
    return out


async def build_cohort(gc) -> dict:
    from imas_codex.settings import get_model
    from imas_codex.standard_names.benchmark_roles import load_refine_corpus
    from imas_codex.standard_names.review.checks import load_review_checks
    from imas_codex.standard_names.review.jev_review import NAME_AXIS, review_names

    population = load_refine_corpus(sample=100000, seed=SEED)
    excluded = golden_units()
    held_out = [
        c
        for c in population
        if c["sn_id"] not in excluded and c["prior_name"] not in excluded
    ]

    rng = random.Random(SEED)
    buckets = defaultdict(list)
    for case in held_out:
        buckets[case.get("physics_domain") or ""].append(case)
    for bucket in buckets.values():
        rng.shuffle(bucket)
    keys = sorted(buckets)
    drawn = []
    while len(drawn) < JUDGE_POOL and any(buckets[k] for k in keys):
        for key in keys:
            if buckets[key] and len(drawn) < JUDGE_POOL:
                drawn.append(buckets[key].pop())

    items = ancestor_items(drawn, gc)
    checks = [c for c in load_review_checks()[0] if c.axis == NAME_AXIS]
    review = await review_names(
        items, checks=checks, model=get_model("sn-judge"), concurrency=CONCURRENCY
    )

    accepted = []
    rejected = defaultdict(int)
    for case, item, record in zip(drawn, items, review.items, strict=True):
        if not record.get("answered"):
            rejected["ancestor review unanswered"] += 1
            continue
        if not case.get("critique"):
            rejected["empty critique"] += 1
            continue
        failed = failed_check_records(record)
        if not failed:
            rejected["ancestor passed every check"] += 1
            continue
        accepted.append(
            {
                **case,
                "source_description": item.get("description") or case["description"],
                "source_unit": item.get("unit") or case["unit"],
                "failed_checks": failed,
                "ancestor_checks": {
                    cid: {
                        "failed": c.get("failed"),
                        "failure_probability": c.get("failure_probability"),
                    }
                    for cid, c in (record.get("checks") or {}).items()
                },
            }
        )

    return {
        "population": len(population),
        "golden_units": len(excluded),
        "excluded_by_golden": len(population) - len(held_out),
        "candidates_judged": len(drawn),
        "accepted": accepted,
        "rejection_reasons": dict(rejected),
        "review_cost_usd": review.cost,
        "review_model": get_model("sn-judge"),
        "seed": SEED,
    }


def load_receipts() -> dict:
    if not RECEIPTS.exists():
        return {}
    out = {}
    for line in RECEIPTS.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            out[(row["case"], row["arm"])] = row
    return out


def append_receipt(row: dict) -> None:
    RECEIPTS.parent.mkdir(parents=True, exist_ok=True)
    with RECEIPTS.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


async def run_arms(cohort: list[dict], gc, already_spent: float) -> dict:
    from imas_codex.discovery.base.llm import acall_llm_structured, ensure_model_prefix
    from imas_codex.llm.prompt_loader import load_prompt_config, render_prompt
    from imas_codex.settings import get_model
    from imas_codex.standard_names.benchmark import _resolve_name
    from imas_codex.standard_names.benchmark_roles import build_refine_prompt_context
    from imas_codex.standard_names.context import build_compose_context
    from imas_codex.standard_names.example_loader import load_compose_examples
    from imas_codex.standard_names.grammar_adapter import parse_canonical_name
    from imas_codex.standard_names.models import RefinedName

    effort = None
    try:
        from imas_codex.settings import get_reasoning_effort

        effort = get_reasoning_effort("sn-refine")
    except Exception:
        effort = None
    model = ensure_model_prefix(get_model("sn-refine"))

    compose = build_compose_context()
    try:
        rules = load_prompt_config("sn_composition_rules").get("composition_rules", [])
    except Exception:
        rules = []
    examples = {}
    for domain in {c.get("physics_domain") or "" for c in cohort}:
        try:
            examples[domain] = load_compose_examples(
                gc, physics_domains=[domain], axis="name"
            )
        except Exception:
            examples[domain] = []

    receipts = load_receipts()
    booked = already_spent + sum(r["cost_usd"] for r in receipts.values())
    state = {"spent": booked, "calls": 0, "skipped": 0}
    log(f"resume: {len(receipts)} receipts, USD {booked:.4f} booked including Jev")
    sem = asyncio.Semaphore(CONCURRENCY)
    ceiling = CEILING_USD - HEADROOM_USD

    async def one(case: dict, arm: str) -> None:
        key = (case["sn_id"], arm)
        done = receipts.get(key)
        # A call that failed is retried once; only a produced output ends the work.
        if done is not None and done.get("outcome") == "ok":
            return
        async with sem:
            if state["spent"] >= ceiling:
                state["skipped"] += 1
                return
            context = build_refine_prompt_context(
                case,
                compose,
                rules,
                examples.get(case.get("physics_domain") or "", []),
                instruction_source=arm,
            )
            if arm == "comments":
                context.pop("failed_checks", None)
            user = render_prompt("sn/refine_name_user", context)
            try:
                system = render_prompt("sn/refine_name_system", context)
            except Exception:
                system = None
            messages = ([{"role": "system", "content": system}] if system else []) + [
                {"role": "user", "content": user}
            ]
            started = _time.time()
            row = {"case": case["sn_id"], "arm": arm, "model": model}
            kwargs = {
                "model": model,
                "messages": messages,
                "response_model": RefinedName,
                "service": "standard-names",
                "max_retries": 2,
            }
            if effort:
                kwargs["reasoning_effort"] = effort
            cost = 0.0
            try:
                refined, cost, _ = await acall_llm_structured(**kwargs)
                name = (
                    _resolve_name(
                        {
                            "segments": refined.segments.model_dump()
                            if hasattr(refined.segments, "model_dump")
                            else refined.segments
                        }
                    )
                    or ""
                )
                try:
                    parse_canonical_name(name)
                    parses = True
                except Exception:
                    parses = False
                row.update(
                    {
                        "outcome": "ok",
                        "refined_name": name,
                        "refined_description": refined.description or "",
                        "parses": parses,
                    }
                )
            except Exception as exc:  # an arm failing is itself an outcome
                row.update(
                    {
                        "outcome": cost and "ok" or f"error:{type(exc).__name__}",
                        "refined_name": "",
                        "refined_description": "",
                        "parses": False,
                    }
                )
            row["cost_usd"] = float(cost or 0.0)
            row["latency_s"] = round(_time.time() - started, 2)
            row["prompt_chars"] = len(user) + len(system or "")
            state["spent"] += row["cost_usd"]
            state["calls"] += 1
            receipts[key] = row
            append_receipt(row)
            log(
                f"  [{state['calls']:3d}] {case['sn_id'][:42]:42s} {arm:8s} "
                f"${row['cost_usd']:.4f} booked ${state["spent"]:.4f} {row["outcome"]}"
            )

    await asyncio.gather(*[one(c, arm) for c in cohort for arm in ARMS])
    return {
        "model": model,
        "reasoning_effort": effort,
        "spent_usd": state["spent"],
        "arm_calls": state["calls"],
        "skipped_for_ceiling": state["skipped"],
        "receipts": receipts,
    }


def build_sheet(cohort: list[dict], receipts: dict) -> tuple[list[dict], dict]:
    """One row per distinct refined output; identical outputs collapse."""
    rows = []
    key = {}
    by_output = {}
    identical_pairs = 0
    complete = []
    incomplete = {}

    for case in cohort:
        outputs = {}
        for arm in ARMS:
            row = receipts.get((case["sn_id"], arm))
            if not row or row.get("outcome") != "ok" or not row.get("refined_name"):
                incomplete[case["sn_id"]] = (row or {}).get("outcome", "not called")
                break
            outputs[arm] = (row["refined_name"], row["refined_description"])
        if len(outputs) != len(ARMS):
            continue
        complete.append(case["sn_id"])
        if outputs["comments"] == outputs["checks"]:
            identical_pairs += 1
        for arm in ARMS:
            name, description = outputs[arm]
            found = by_output.get((name, description))
            if found is not None:
                key[found]["arms"][arm].append(case["sn_id"])
                continue
            sheet_id = f"row-{len(rows) + 1:03d}"
            by_output[(name, description)] = sheet_id
            rows.append(
                {
                    "sheet_id": sheet_id,
                    "source_path": case["path"],
                    "source_description": case.get("source_description")
                    or case["description"],
                    "source_unit": case.get("source_unit") or case["unit"],
                    "ancestor_name": case["prior_name"],
                    "refined_name": name,
                    "refined_description": description,
                }
            )
            key[sheet_id] = {
                "case": case["sn_id"],
                "prior_name": case["prior_name"],
                "arms": {arm: [case["sn_id"]] for arm in ARMS},
            }

    random.Random(SHEET_SEED).shuffle(rows)
    return rows, {
        "rows": key,
        "identical_output_pairs": identical_pairs,
        "cases_in_sheet": complete,
        "cases_incomplete": incomplete,
        "sheet_seed": SHEET_SEED,
        "cohort_seed": SEED,
    }


def sheet_leaks(rows: list[dict]) -> list[str]:
    findings = []
    for row in rows:
        unexpected = set(row) - ALLOWED_ROW_FIELDS
        if unexpected:
            findings.append(f"{row.get('sheet_id')}: unexpected {sorted(unexpected)}")
        leaking = set(row) & LEAK_FIELDS
        if leaking:
            findings.append(f"{row.get('sheet_id')}: leaking {sorted(leaking)}")
    return findings


def check_sheet() -> int:
    """Fail if the sheet carries a hidden field, and prove the check can fail."""
    rows = json.loads(SHEET.read_text(encoding="utf-8"))
    findings = sheet_leaks(rows)
    if findings:
        log("FAIL: committed label sheet carries hidden fields")
        for finding in findings:
            log(f"  {finding}")
        return 1
    if not rows:
        log("FAIL: label sheet is empty, so the check proves nothing")
        return 1
    caught = sheet_leaks([{**rows[0], "arm": "checks"}])
    if not caught:
        log("FAIL: the planted leaking row was not caught")
        return 1
    log(f"OK: {len(rows)} rows clean; planted row refused: {caught[0]}")
    return 0


async def main() -> int:
    from imas_codex.graph.client import GraphClient

    gc = GraphClient()
    cohort_info = await build_cohort(gc)
    log(
        f"population {cohort_info['population']} | golden-excluded "
        f"{cohort_info['excluded_by_golden']} | judged "
        f"{cohort_info['candidates_judged']} | accepted "
        f"{len(cohort_info['accepted'])} | jev USD {cohort_info['review_cost_usd']:.4f}"
    )
    log(f"rejections: {cohort_info['rejection_reasons']}")
    cohort = cohort_info["accepted"][:COHORT_SIZE]
    log(f"cohort {len(cohort)} cases -> {len(cohort) * len(ARMS)} refine calls")

    arms = await run_arms(cohort, gc, cohort_info["review_cost_usd"])
    rows, key = build_sheet(cohort, arms["receipts"])
    save_json(SHEET, rows)
    save_json(KEY_PATH, key)

    per_arm = {}
    for arm in ARMS:
        arm_rows = [r for r in arms["receipts"].values() if r["arm"] == arm]
        ok = [r for r in arm_rows if r["outcome"] == "ok"]
        per_arm[arm] = {
            "calls": len(arm_rows),
            "errors": len(arm_rows) - len(ok),
            "spend_usd": round(sum(r["cost_usd"] for r in arm_rows), 6),
            "outputs": len(ok),
            "parsed": sum(1 for r in ok if r["parses"]),
            "parse_rate": round(sum(1 for r in ok if r["parses"]) / len(ok), 4)
            if ok
            else 0.0,
            "mean_cost_usd": round(
                sum(r["cost_usd"] for r in arm_rows) / len(arm_rows), 6
            )
            if arm_rows
            else 0.0,
        }

    results = {
        "cohort": {k: v for k, v in cohort_info.items() if k != "accepted"},
        "cohort_size": len(cohort),
        "arms": per_arm,
        "model": arms["model"],
        "reasoning_effort": arms["reasoning_effort"],
        "refine_spend_usd": round(
            sum(r["cost_usd"] for r in arms["receipts"].values()), 6
        ),
        "total_spend_usd": round(arms["spent_usd"], 6),
        "ceiling_usd": CEILING_USD,
        "skipped_for_ceiling": arms["skipped_for_ceiling"],
        "identical_output_pairs": key["identical_output_pairs"],
        "sheet_rows": len(rows),
        "sheet_cases": len(key["cases_in_sheet"]),
        "sheet_incomplete": key["cases_incomplete"],
        "cases": [
            {
                "sn_id": c["sn_id"],
                "prior_name": c["prior_name"],
                "path": c["path"],
                "description": c["description"],
                "unit": c["unit"],
                "critique": c["critique"],
                "failed_checks": c["failed_checks"],
                "ancestor_checks": c["ancestor_checks"],
                "comments_arm": arms["receipts"].get((c["sn_id"], "comments")),
                "checks_arm": arms["receipts"].get((c["sn_id"], "checks")),
            }
            for c in cohort
        ],
    }
    save_json(RESULTS, results)
    log(
        f"total USD {arms['spent_usd']:.4f} over {arms['arm_calls']} refine calls "
        f"(ceiling {CEILING_USD}); skipped {arms['skipped_for_ceiling']}"
    )
    log(f"wrote {SHEET} ({len(rows)} rows), {RESULTS}, key {KEY_PATH}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-sheet", action="store_true")
    args = parser.parse_args()
    if args.check_sheet:
        sys.exit(check_sheet())
    sys.exit(asyncio.run(main()))