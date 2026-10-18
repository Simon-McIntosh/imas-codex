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
from collections import Counter, defaultdict
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
SHEET_SEED = 20261020
COHORT_SIZE = 120
JUDGE_POOL = 160
CEILING_USD = 33.0
HEADROOM_USD = 0.5
MAX_ATTEMPTS = 2
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


DD_SOURCE_QUERY = """
UNWIND $ids AS nid
MATCH (sn:StandardName {id: nid})<-[:PRODUCED_NAME]-(src:StandardNameSource)
WHERE src.id STARTS WITH 'dd:'
WITH nid, src.id AS source_id
ORDER BY source_id
WITH nid, collect(source_id)[0] AS source_id
WITH nid, source_id, substring(source_id, 3) AS path
OPTIONAL MATCH (node:IMASNode {id: path})
OPTIONAL MATCH (node)-[:HAS_UNIT]->(unit:Unit)
RETURN nid, source_id, path, node.description AS description,
       node.documentation AS documentation, unit.id AS unit
"""

# The same binding seen from the DD node side (IMASNode -> HAS_STANDARD_NAME).
# The brief routes DD context through PRODUCED_NAME; this second count is
# measured so the rows it would reach that PRODUCED_NAME does not are reported.
DD_PROJECTION_QUERY = """
UNWIND $ids AS nid
MATCH (node:IMASNode)-[:HAS_STANDARD_NAME]->(sn:StandardName {id: nid})
OPTIONAL MATCH (node)-[:HAS_UNIT]->(unit:Unit)
RETURN nid, node.id AS source_id, node.description AS description,
       node.documentation AS documentation, unit.id AS unit
ORDER BY nid, source_id
"""


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
            evidence = f"chosen: {answer.get('choice')}"
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


def resolve_dd_sources(gc, names: list[str]) -> tuple[dict[str, dict], dict[str, dict]]:
    """The DD source each name is bound to, through ``PRODUCED_NAME``.

    Returns ``(by_name, by_projection)``.  ``by_name`` holds the ancestor's and
    the refined name's dd source — path, description and unit — keyed on the
    name id; a name with no such binding is absent, and the row it would have
    carried is dropped rather than shipped without DD context.  ``by_projection``
    is the same binding read from the DD node side, kept only to measure how
    many rows the two routes disagree about.
    """
    wanted = sorted({n for n in names if n})
    if not wanted:
        return {}, {}

    by_name: dict[str, dict] = {}
    for row in gc.query(DD_SOURCE_QUERY, ids=wanted):
        by_name.setdefault(
            row["nid"],
            {
                "source_path": row["source_id"],
                "source_description": row["description"] or row["documentation"] or "",
                "source_unit": row["unit"] or "",
            },
        )

    by_projection: dict[str, dict] = {}
    for row in gc.query(DD_PROJECTION_QUERY, ids=wanted):
        by_projection.setdefault(
            row["nid"],
            {
                "source_path": f"dd:{row['source_id']}",
                "source_description": row["description"] or row["documentation"] or "",
                "source_unit": row["unit"] or "",
            },
        )
    return by_name, by_projection


def case_dd_source(case: dict, by_name: dict[str, dict]) -> dict | None:
    """The DD context a case contributes, or None when it has no DD source."""
    if case.get("path"):
        return {
            "source_path": case["path"],
            "source_description": case.get("source_description")
            or case.get("description")
            or "",
            "source_unit": case.get("source_unit") or case.get("unit") or "",
        }
    for name in (case.get("prior_name"), case.get("sn_id")):
        found = by_name.get(name or "")
        if found and found.get("source_path"):
            return dict(found)
    return None


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


def rescore_arms(already_spent: float) -> dict:
    """Rebuild the arm results from the receipts without buying a call.

    Every call the run made is recorded line by line, so the results can be
    recomputed at any time from the receipts alone.  Nothing here reaches the
    provider: a key whose last receipt did not produce a name stays a failure,
    which is what it is, rather than being bought again.  The ceiling counters
    are read from the stored results because they record what the run did, not
    what the receipts contain.
    """
    from imas_codex.discovery.base.llm import ensure_model_prefix
    from imas_codex.settings import get_model

    effort = None
    try:
        from imas_codex.settings import get_reasoning_effort

        effort = get_reasoning_effort("sn-refine")
    except Exception:
        effort = None
    lines = receipt_lines()
    costs = [float(row.get("cost_usd") or 0.0) for row in lines]
    arm_costs = {
        arm: [float(r.get("cost_usd") or 0.0) for r in lines if r.get("arm") == arm]
        for arm in ARMS
    }
    return {
        "model": ensure_model_prefix(get_model("sn-refine")),
        "reasoning_effort": effort,
        "spent_usd": already_spent + sum(costs),
        "arm_calls": len(lines),
        "costs": costs,
        "arm_costs": arm_costs,
        "receipts": load_receipts(),
    }


def receipt_lines() -> list[dict]:
    """Every receipt line, one per call made — a retried call has two.

    Spend is summed over these rather than over the last receipt per key: a
    retry does not unpay for the attempt that failed before it.
    """
    if not RECEIPTS.exists():
        return []
    return [
        json.loads(line)
        for line in RECEIPTS.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_receipts() -> dict:
    """The last receipt per (case, arm), with empty outputs read as failures.

    A call the provider answered but whose structured response carries no name
    is not an output: it is recorded here as ``empty_name`` so that it is
    counted as a failure, retried like one, and never reaches the sheet.  The
    receipts written before token capture carries ``prompt_tokens: null`` and
    ``completion_tokens: null`` — absent evidence, never zero.
    """
    if not RECEIPTS.exists():
        return {}
    out = {}
    for row in receipt_lines():
        if row.get("outcome") == "ok" and not row.get("refined_name"):
            row = {**row, "outcome": "empty_name"}
        out[(row["case"], row["arm"])] = row
    return out


def append_receipt(row: dict) -> None:
    if row.get("outcome") in {"ok", "empty_name"}:
        missing = [
            field
            for field in ("prompt_tokens", "completion_tokens")
            if row.get(field) is None
        ]
        if missing:
            raise RuntimeError(
                f"receipt for {row.get('case')}/{row.get('arm')} lacks "
                f"{missing}; the provider answered, so its usage must be recorded"
            )
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
    booked = already_spent + sum(r["cost_usd"] for r in receipt_lines())
    state = {
        "spent": booked,
        "calls": 0,
        "skipped": 0,
        "exhausted": 0,
        "costs": [],
        "arm_costs": {arm: [] for arm in ARMS},
    }
    log(f"resume: {len(receipts)} receipts, USD {booked:.4f} booked including Jev")
    sem = asyncio.Semaphore(CONCURRENCY)
    ceiling = CEILING_USD - HEADROOM_USD

    async def one(case: dict, arm: str) -> None:
        key = (case["sn_id"], arm)
        done = receipts.get(key)
        # A call that failed, or answered without a name, is retried once; only
        # a response that actually produced a name ends the work.
        if done is not None and done.get("outcome") == "ok":
            return
        if done is not None and int(done.get("attempt") or 1) >= MAX_ATTEMPTS:
            state["exhausted"] += 1
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
            prompt_tokens = None
            completion_tokens = None
            try:
                result = await acall_llm_structured(**kwargs)
                refined, cost = result.parsed, result.cost
                prompt_tokens = result.input_tokens
                completion_tokens = result.output_tokens
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
                        # A response that carries no name is not an output, so it
                        # is recorded as a failure at the moment it is written and
                        # retried like any other.
                        "outcome": "ok" if name else "empty_name",
                        "refined_name": name,
                        "refined_description": refined.description or "",
                        "parses": parses,
                    }
                )
            except Exception as exc:  # an arm failing is itself an outcome
                prompt_tokens = getattr(exc, "input_tokens", None)
                completion_tokens = getattr(exc, "output_tokens", None)
                row.update(
                    {
                        "outcome": f"error:{type(exc).__name__}",
                        "refined_name": "",
                        "refined_description": "",
                        "parses": False,
                    }
                )
            row["cost_usd"] = float(cost or 0.0)
            row["prompt_tokens"] = prompt_tokens
            row["completion_tokens"] = completion_tokens
            row["attempt"] = 1 + (
                int(done.get("attempt") or 1) if done is not None else 0
            )
            row["latency_s"] = round(_time.time() - started, 2)
            row["prompt_chars"] = len(user) + len(system or "")
            state["spent"] += row["cost_usd"]
            state["costs"].append(row["cost_usd"])
            state["arm_costs"][arm].append(row["cost_usd"])
            state["calls"] += 1
            receipts[key] = row
            append_receipt(row)
            log(
                f"  [{state['calls']:3d}] {case['sn_id'][:42]:42s} {arm:8s} "
                f"${row['cost_usd']:.4f} booked ${state['spent']:.4f} {row['outcome']}"
            )

    await asyncio.gather(*[one(c, arm) for c in cohort for arm in ARMS])
    return {
        "model": model,
        "reasoning_effort": effort,
        "spent_usd": state["spent"],
        "arm_calls": state["calls"],
        "skipped_for_ceiling": state["skipped"],
        "exhausted_attempts": state["exhausted"],
        "costs": state["costs"],
        "arm_costs": state["arm_costs"],
        "receipts": receipts,
    }


def sheet_key_findings(
    key: dict, rows: list[dict], produced: dict[tuple[str, str], tuple[str, str]]
) -> list[str]:
    """Check the key against the outputs it claims to map.

    Three facts have to hold. Every output the run produced is credited to
    exactly one row. Every credit names a case and arm whose output is that
    row's output. Together those give the third: a row carries both arms only
    when the two arms produced the same output for that case, because otherwise
    the two outputs would be credited to one row while only one of them is that
    row's output.
    """
    findings: list[str] = []
    row_outputs = {
        row["sheet_id"]: (row["refined_name"], row["refined_description"])
        for row in rows
    }
    claims: dict[tuple[str, str], list[str]] = {}
    for sheet_id, entry in key.items():
        if sheet_id not in row_outputs:
            findings.append(f"{sheet_id}: named in the key but not on the sheet")
            continue
        for arm, case_ids in (entry.get("arms") or {}).items():
            for case_id in case_ids:
                claims.setdefault((case_id, arm), []).append(sheet_id)

    for pair, sheet_ids in sorted(claims.items()):
        if len(sheet_ids) > 1:
            findings.append(
                f"{pair[0]} {pair[1]}: credited to {len(sheet_ids)} rows "
                f"({', '.join(sorted(sheet_ids))}); each output belongs to one"
            )
    for pair in sorted(produced):
        if pair not in claims:
            findings.append(f"{pair[0]} {pair[1]}: produced an output no row credits")
    for pair, sheet_ids in sorted(claims.items()):
        if pair not in produced:
            findings.append(f"{pair[0]} {pair[1]}: credited by a row but not produced")
            continue
        for sheet_id in sheet_ids:
            if produced[pair] != row_outputs[sheet_id]:
                findings.append(
                    f"{sheet_id}: credits {pair[0]} {pair[1]}, whose output is not "
                    "that row's output"
                )
    return findings


def rows_crediting_both_arms(key: dict) -> int:
    """Rows that name every arm, which only an identical pair can produce."""
    return sum(1 for entry in key.values() if set(entry.get("arms") or {}) >= set(ARMS))


def sheet_population(
    cohort: list[dict], receipts: dict, dd_sources: dict[str, dict]
) -> tuple[
    list[tuple[dict, dict[str, tuple[str, str]]]], list[str], dict, list[str], int
]:
    """Which cases reach the sheet, and what each arm produced for them.

    A case contributes only when both arms returned a name and the case
    resolves a DD source; anything else is counted and dropped.  The sheet and
    the key are both built from this, and the key check re-derives it from the
    receipts so that the check cannot inherit the builder's mistakes.
    """
    contributing: list[tuple[dict, dict[str, tuple[str, str]]]] = []
    complete: list[str] = []
    incomplete: dict[str, str] = {}
    no_dd_source: list[str] = []
    identical_pairs = 0
    for case in cohort:
        outputs: dict[str, tuple[str, str]] = {}
        for arm in ARMS:
            row = receipts.get((case["sn_id"], arm))
            if not row or not row.get("refined_name"):
                incomplete[case["sn_id"]] = (row or {}).get("outcome", "not called")
                break
            outputs[arm] = (row["refined_name"], row["refined_description"] or "")
        if len(outputs) != len(ARMS):
            continue
        complete.append(case["sn_id"])
        source = case_dd_source(case, dd_sources)
        if not source or not source.get("source_path"):
            no_dd_source.append(case["sn_id"])
            continue
        if outputs["comments"] == outputs["checks"]:
            identical_pairs += 1
        contributing.append((case, outputs))
    return contributing, complete, incomplete, no_dd_source, identical_pairs


def build_sheet(
    cohort: list[dict], receipts: dict, dd_sources: dict[str, dict]
) -> tuple[list[dict], dict]:
    """One row per distinct refined output; identical outputs collapse.

    Every row carries the DD context of the case that produced it, read from the
    case's own source path or, where it has none, from the ancestor's or the
    refined name's bound DD source. A case with no resolvable DD source
    contributes no row: it is counted in ``cases_without_dd_source`` rather than
    shipped with a blank path.

    The key is built from the outputs, never from the arm loop: a row credits an
    arm only for the cases whose output for that arm is the row's output, so a
    row carries both arms exactly when the two arms produced the same output.
    """
    rows = []
    claimed: dict[str, dict[str, list[str]]] = {}
    produced: dict[tuple[str, str], tuple[str, str]] = {}
    by_output = {}

    contributing, complete, incomplete, no_dd_source, identical_pairs = (
        sheet_population(cohort, receipts, dd_sources)
    )
    prior_names = {case["sn_id"]: case["prior_name"] for case, _ in contributing}

    for case, outputs in contributing:
        source = case_dd_source(case, dd_sources)
        for arm in ARMS:
            name, description = outputs[arm]
            produced[(case["sn_id"], arm)] = outputs[arm]
            found = by_output.get((name, description))
            if found is None:
                found = f"row-{len(rows) + 1:03d}"
                by_output[(name, description)] = found
                claimed[found] = {}
                rows.append(
                    {
                        "sheet_id": found,
                        "source_path": source["source_path"],
                        "source_description": source["source_description"],
                        "source_unit": source["source_unit"],
                        "ancestor_name": case["prior_name"],
                        "refined_name": name,
                        "refined_description": description,
                    }
                )
            claimed[found].setdefault(arm, []).append(case["sn_id"])

    key = {
        sheet_id: {
            "cases": sorted({c for ids in arms.values() for c in ids}),
            "prior_names": sorted(
                {prior_names[c] for ids in arms.values() for c in ids}
            ),
            "arms": {arm: sorted(ids) for arm, ids in sorted(arms.items())},
        }
        for sheet_id, arms in claimed.items()
    }
    disagreements = sheet_key_findings(key, rows, produced)
    both_arms = rows_crediting_both_arms(key)
    if both_arms != identical_pairs:
        disagreements.append(
            f"{both_arms} rows carry both arms against {identical_pairs} "
            "identical output pairs"
        )
    if disagreements:
        raise RuntimeError(
            "the sheet key disagrees with the outputs: " + "; ".join(disagreements)
        )

    random.Random(SHEET_SEED).shuffle(rows)
    return rows, {
        "rows": key,
        "identical_output_pairs": identical_pairs,
        "cases_in_sheet": complete,
        "cases_incomplete": incomplete,
        "cases_without_dd_source": no_dd_source,
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


def sheet_blank_paths(rows: list[dict]) -> list[str]:
    """Rows shipped without DD source context, which must never happen."""
    return [
        f"{row.get('sheet_id')}: blank source_path"
        for row in rows
        if not str(row.get("source_path") or "").strip()
    ]


def sheet_blank_names(rows: list[dict]) -> list[str]:
    """Rows shipped without a refined name, which must never happen.

    An answer that carries no name is recorded as a failure at the moment the
    call returns, so a nameless answer can never reach the sheet.  A row without
    a name would be a labeller asked to judge nothing.
    """
    return [
        f"{row.get('sheet_id')}: blank refined_name"
        for row in rows
        if not str(row.get("refined_name") or "").strip()
    ]


def check_sheet() -> int:
    """Fail on a hidden field, a blank DD path or a blank name, and prove all three."""
    rows = json.loads(SHEET.read_text(encoding="utf-8"))
    findings = sheet_leaks(rows) + sheet_blank_paths(rows) + sheet_blank_names(rows)
    if findings:
        log("FAIL: committed label sheet carries hidden fields or blank DD paths")
        for finding in findings:
            log(f"  {finding}")
        return 1
    if not rows:
        log("FAIL: label sheet is empty, so the check proves nothing")
        return 1
    planted_arm = sheet_leaks([{**rows[0], "arm": "checks"}])
    planted_blank = sheet_blank_paths([{**rows[0], "source_path": ""}])
    planted_name = sheet_blank_names([{**rows[0], "refined_name": ""}])
    if not planted_arm:
        log("FAIL: the planted leaking row was not caught")
        return 1
    if not planted_blank:
        log("FAIL: the planted blank-path row was not caught")
        return 1
    if not planted_name:
        log("FAIL: the planted blank-name row was not caught")
        return 1
    log(
        f"OK: {len(rows)} rows clean; planted arm row refused: {planted_arm[0]}; "
        f"planted blank-path row refused: {planted_blank[0]}; "
        f"planted blank-name row refused: {planted_name[0]}"
    )
    return 0


def check_key() -> int:
    """Re-derive the key's claims from the receipts and refuse a wrong key.

    The key lives only in the run directory, so this check is the one that has
    to hold: the sheet and the key are both written by the run, and a key that
    credits an arm for a case whose output the row does not carry would let a
    later reader attribute an output to the arm that did not produce it.
    """
    from imas_codex.graph.client import GraphClient

    gc = GraphClient()
    cohort, _cohort_info = resume_cohort()
    names = sorted({c["prior_name"] for c in cohort} | {c["sn_id"] for c in cohort})
    dd_sources, _projections = resolve_dd_sources(gc, names)
    receipts = load_receipts()
    rows = json.loads(SHEET.read_text(encoding="utf-8"))
    bundle = json.loads(KEY_PATH.read_text(encoding="utf-8"))
    key = bundle.get("rows", bundle)
    contributing, _complete, _incomplete, _no_source, identical = sheet_population(
        cohort, receipts, dd_sources
    )
    produced = {
        (case["sn_id"], arm): output
        for case, outputs in contributing
        for arm, output in outputs.items()
    }
    findings = sheet_key_findings(key, rows, produced)
    both_arms = rows_crediting_both_arms(key)
    if bundle.get("identical_output_pairs") not in (None, identical):
        findings.append(
            f"the key file records {bundle['identical_output_pairs']} identical "
            f"pairs against {identical} recomputed from the receipts"
        )
    if both_arms != identical:
        findings.append(
            f"{both_arms} rows carry both arms against {identical} identical "
            "output pairs"
        )
    if findings:
        log("FAIL: the committed key disagrees with the receipts")
        for finding in findings:
            log(f"  {finding}")
        return 1
    planted = _both_arms_key_with_differing_outputs(key, produced)
    if planted is None:
        log(
            "FAIL: no case has two different outputs, so the planted key proves nothing"
        )
        return 1
    planted_key, case_id = planted
    planted_findings = sheet_key_findings(planted_key, rows, produced)
    if not planted_findings:
        log("FAIL: the planted both-arms key was not caught")
        return 1
    log(
        f"OK: {len(key)} rows, {len(produced)} produced outputs, "
        f"{identical} identical pairs, both-arms rows {both_arms}; planted key "
        f"crediting both arms for {case_id} refused: {planted_findings[0]}"
    )
    return 0


def _both_arms_key_with_differing_outputs(
    key: dict, produced: dict[tuple[str, str], tuple[str, str]]
) -> tuple[dict, str] | None:
    """Take a real key and credit both arms to a row whose outputs differ.

    The planted defect is the one the brief names: a row that names both arms
    when the two arms produced different outputs.  It is built on the committed
    key rather than on a fresh stub so the assertion is exercised by the shape
    the run actually writes.
    """
    for sheet_id, entry in sorted(key.items()):
        case_id = (entry.get("arms") or {}).get("comments", [None])[0]
        if not case_id:
            continue
        other = produced.get((case_id, "checks"))
        if other is None or other != produced.get((case_id, "comments")):
            planted = dict(key)
            planted[sheet_id] = {
                **entry,
                "arms": {
                    **entry["arms"],
                    "checks": sorted({*entry["arms"].get("checks", []), case_id}),
                },
            }
            return planted, case_id
    return None


def resume_cohort() -> tuple[list[dict], dict]:
    """Rebuild the stored cohort from the corpus, with its stored critique.

    The refine prompt is rendered from the case dict, so a retry must see the
    case the first call saw.  Every case is re-read from the same loader at the
    same seed and its critique compared against the stored run, which proves the
    retried prompt is the prompt the earlier call was given.
    """
    from imas_codex.standard_names.benchmark_roles import load_refine_corpus

    stored = json.loads(RESULTS.read_text(encoding="utf-8"))
    corpus = {c["sn_id"]: c for c in load_refine_corpus(sample=100000, seed=SEED)}
    cohort = []
    for entry in stored["cases"]:
        case = corpus.get(entry["sn_id"])
        if case is None:
            raise RuntimeError(f"{entry['sn_id']} is absent from the corpus")
        if case["critique"] != entry["critique"]:
            raise RuntimeError(
                f"{entry['sn_id']}: the corpus critique differs from the stored run"
            )
        cohort.append(
            {
                **case,
                "failed_checks": entry["failed_checks"],
                "ancestor_checks": entry["ancestor_checks"],
            }
        )
    log(f"resumed cohort of {len(cohort)} cases from {RESULTS.name}")
    return cohort, dict(stored["cohort"])


async def main(resume: bool = False, rescore: bool = False) -> int:
    from imas_codex.graph.client import GraphClient

    gc = GraphClient()
    if resume or rescore:
        cohort, cohort_info = resume_cohort()
    else:
        cohort_info = await build_cohort(gc)
        cohort = cohort_info["accepted"][:COHORT_SIZE]
    log(
        f"population {cohort_info['population']} | golden-excluded "
        f"{cohort_info['excluded_by_golden']} | judged "
        f"{cohort_info['candidates_judged']} | rejected "
        f"{cohort_info['rejection_reasons']} | jev USD "
        f"{cohort_info['review_cost_usd']:.4f}"
    )
    log(f"cohort {len(cohort)} cases -> {len(cohort) * len(ARMS)} refine calls")

    if rescore:
        stored = json.loads(RESULTS.read_text(encoding="utf-8"))
        arms = rescore_arms(cohort_info["review_cost_usd"])
        arms["skipped_for_ceiling"] = stored.get("skipped_for_ceiling", 0)
        arms["exhausted_attempts"] = stored.get("exhausted_attempts", 0)
        log(
            f"rescore: {len(receipt_lines())} receipt lines over "
            f"{len(arms['receipts'])} calls, no call bought"
        )
    else:
        arms = await run_arms(cohort, gc, cohort_info["review_cost_usd"])
    names = sorted({c["prior_name"] for c in cohort} | {c["sn_id"] for c in cohort})
    dd_sources, projections = resolve_dd_sources(gc, names)
    log(f"DD sources bound through PRODUCED_NAME: {len(dd_sources)} of {len(names)}")
    rows, key = build_sheet(cohort, arms["receipts"], dd_sources)
    save_json(SHEET, rows)
    save_json(KEY_PATH, key)

    projection_only = [
        c["sn_id"]
        for c in cohort
        if not case_dd_source(c, dd_sources)
        and any(
            projections.get(name, {}).get("source_path")
            for name in (c["prior_name"], c["sn_id"])
        )
    ]

    per_arm = {}
    for arm in ARMS:
        arm_rows = [r for r in arms["receipts"].values() if r["arm"] == arm]
        outputs = [r for r in arm_rows if r.get("refined_name")]
        empty = [r for r in arm_rows if r["outcome"] == "empty_name"]
        errored = [r for r in arm_rows if r["outcome"] not in ("ok", "empty_name")]
        arm_costs = arms["arm_costs"][arm]
        per_arm[arm] = {
            "calls": len(arm_rows),
            "calls_made": len(arm_costs),
            "retry_calls": len(arm_costs) - len(arm_rows),
            "spend_usd": round(sum(arm_costs), 6),
            "spend_note": (
                "every call made in this arm, cohort calls and retries alike"
            ),
            "outputs": len(outputs),
            "outputs_note": (
                "an answer counts as an output only when it carries a refined "
                "name; an ok status on a nameless answer is the transport's "
                "verdict on the call, not evidence of a name"
            ),
            "failures": {
                "empty_name": len(empty),
                "model_error": len(errored),
                "total": len(empty) + len(errored),
                "model_error_kinds": {
                    kind: sum(1 for r in errored if r["outcome"] == kind)
                    for kind in sorted({r["outcome"] for r in errored})
                },
            },
            "parsed": sum(1 for r in outputs if r["parses"]),
            "parse_rate": round(
                sum(1 for r in outputs if r["parses"]) / len(outputs), 4
            )
            if outputs
            else 0.0,
            "mean_cost_usd": round(sum(arm_costs) / len(arm_costs), 6)
            if arm_costs
            else 0.0,
            "with_tokens": sum(1 for r in arm_rows if r.get("prompt_tokens")),
        }

    lines = receipt_lines()
    counts = Counter((row["case"], row["arm"]) for row in lines)
    retried_keys = {key for key, n in counts.items() if n > 1}
    retry_calls = len(lines) - len(arms["receipts"])
    # A retry line is a line whose case-and-arm has an earlier line. The
    # receipt's attempt field cannot identify one: the retry receipts were
    # written by a launch that predates the counter, so every line reads
    # attempt 1.
    retry_lines = []
    preceding: set[tuple[str, str]] = set()
    for row in lines:
        pair = (row.get("case"), row.get("arm"))
        if pair in preceding:
            retry_lines.append(row)
        else:
            preceding.add(pair)
    if len(retry_lines) != retry_calls:
        raise RuntimeError(
            f"{len(retry_lines)} lines repeat a case-and-arm against "
            f"{retry_calls} lines beyond the {len(arms['receipts'])} calls"
        )
    retry_calls = len(retry_lines)

    results = {
        "cohort": {k: v for k, v in cohort_info.items() if k != "accepted"},
        "cohort_size": len(cohort),
        "arms": per_arm,
        "model": arms["model"],
        "reasoning_effort": arms["reasoning_effort"],
        "refine_spend_usd": round(sum(arms["costs"]), 6),
        "refine_spend_note": (
            "Every refine call made, retries included. Summing one cost per "
            "case-and-arm instead would drop the first attempt of each retried "
            "call and understate the pass."
        ),
        "total_spend_usd": round(arms["spent_usd"], 6),
        "ceiling_usd": CEILING_USD,
        "skipped_for_ceiling": arms["skipped_for_ceiling"],
        "exhausted_attempts": arms["exhausted_attempts"],
        "tokens": {
            "captured_from": "the retry pass; the cohort calls predate capture",
            "earlier_calls": None,
            "with_tokens": sum(1 for r in receipt_lines() if r.get("prompt_tokens")),
            "note": (
                "Calls made before tokens were captured carry null prompt and "
                "completion counts: the data was never written, so it is "
                "absent, not zero. The receipt writer now records prompt and "
                "completion tokens from the response usage for every call and "
                "refuses a receipt whose outcome is ok and whose token counts "
                "are null."
            ),
            "retry_pass_calls": len(retry_lines),
            "retry_pass_without_tokens": sum(
                1 for r in retry_lines if r.get("prompt_tokens") is None
            ),
            "retry_pass_without_tokens_note": (
                "retry-pass calls that raised model-side and so exposed no "
                "usage; the 240 cohort calls are not counted here, because "
                "their keys being retried does not give them tokens"
            ),
        },
        "retried_calls": retry_calls,
        "retried_keys": len(retried_keys),
        "retried_note": (
            "A failed or empty first call is retried once. Retry calls are "
            "counted from the receipt lines, not from the per-key receipts: "
            "the attempt field on a retry receipt counts from the stored "
            "attempt, and a key retried by an earlier launch carries a third "
            "call."
        ),
        "identical_output_pairs": key["identical_output_pairs"],
        "sheet_rows": len(rows),
        "sheet_cases": len(key["cases_in_sheet"]),
        "sheet_incomplete": key["cases_incomplete"],
        "sheet_cases_without_dd_source": key["cases_without_dd_source"],
        "dd_sources_bound": len(dd_sources),
        "dd_names_asked": len(names),
        "cases_dd_source_only_via_projection": projection_only,
        "sheet_seed": SHEET_SEED,
        "cases": [
            {
                "sn_id": c["sn_id"],
                "prior_name": c["prior_name"],
                "path": c["path"],
                "dd_source": case_dd_source(c, dd_sources),
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
    parser.add_argument("--check-key", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--rescore", action="store_true")
    args = parser.parse_args()
    if args.check_sheet:
        sys.exit(check_sheet())
    if args.check_key:
        sys.exit(check_key())
    sys.exit(asyncio.run(main(resume=args.resume, rescore=args.rescore)))
