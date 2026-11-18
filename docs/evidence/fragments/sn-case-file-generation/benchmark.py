"""Benchmark one-pass against staged record generation on the reference set.

Each arm is a structure crossed with a model. Every arm writes a record for
every member of the reference set from the member's code-built case file, and a blind
adjudicator that is told neither the arm nor the incumbent scores every record
against the committed answer key. Two controls run beside them: one returns
every incumbent unchanged, and the best arm is rerun with the
enclosing-structure text withheld from its case files.

Two rounds run on a development half. The second round's prompt answers the
failure classes the first round recorded. The final figures are taken once on
the held-out half. Spend is reported beside the authorized ceiling.

Every generated record and every blind verdict is appended to a JSON line file
under the run directory before the next call, so an interrupted run keeps what
it has already paid for.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import json
import os
import sys
import time
import zlib
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from imas_codex.standard_names import benchmark as bench

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[3]
DATA = ROOT / "benchmark.json"
PAGE = ROOT / "benchmark.html"
FRAGMENT = ROOT / "scfg-benchmark.html"
REFERENCE = ROOT / "reference-set.json"
FIGURE = (
    REPO / "docs/figures/sn-case-file-generation/scfg-benchmark/heldout-arm-right.svg"
)
RUN = Path(
    os.environ.get(
        "BENCHMARK_RUN",
        "/home/ITER/mcintos/.config/reckon/crew/runs/"
        "r-20261118T064807925154-scfg-benchmark",
    )
)
CASE_CACHE = RUN / "case-files.json"
RECORD_LOG = RUN / "records.jsonl"
VERDICT_LOG = RUN / "verdicts.jsonl"

LOCAL_MODEL = "local/deepseek-v4.1-flash"
FRONTIER_MODEL = "openrouter/openai/gpt-6.1-sol"
JUDGE_MODEL = "openrouter/google/gemini-3.8-flash"
KIMI_COMMAND = (
    "KIMI_BASE_URL=<B300-server>/v1 uv run --no-sync python "
    "docs/evidence/fragments/sn-case-file-generation/benchmark.py --stage all "
    "--models kimi"
)
SPEND_CEILING_USD = 25.0
SPLIT_SEED = 20261118
MAX_TOKENS = 3000
CONCURRENCY = {"local": 12, "frontier": 8, "judge": 10}
ARM_SPECS = (
    ("one-pass/local", "one-pass", LOCAL_MODEL),
    ("staged/local", "staged", LOCAL_MODEL),
    ("one-pass/frontier", "one-pass", FRONTIER_MODEL),
    ("staged/frontier", "staged", FRONTIER_MODEL),
)
FAILURE_CLASSES = (
    "qualifier-in-name",
    "definition-omits-convention",
    "name-not-supported-by-source",
    "unit-or-group-wrong",
    "other",
)


def digest_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def digest_json(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class RecordDraft(BaseModel):
    """The whole record one author writes in one pass."""

    name: str = Field(description="The standard name the quantity is given.")
    description: str = Field(
        description="The definition: what the quantity is and how it is obtained."
    )
    documentation: str = Field(
        description="One paragraph of documentation for the record."
    )
    unit: str = Field(description="The unit, in the source's spelling.")
    group: str = Field(description="The physics group the quantity belongs to.")
    blocker: str = Field(
        default="",
        description=(
            "Empty when a record was produced. Otherwise one of grammar-gap, "
            "registry-conflict, insufficient-source-text, collision."
        ),
    )


class NameStage(BaseModel):
    """The first stage's definition-first name and definition."""

    name: str
    description: str
    unit: str
    group: str
    blocker: str = ""


class DocStage(BaseModel):
    """The second stage's documentation."""

    documentation: str
    blocker: str = ""


class Judgment(BaseModel):
    verdict: str = Field(description="right, wrong or blocked")
    failure_class: str = Field(default="", description="Empty when right.")
    reason: str = Field(description="The text that decided the verdict.")


ONE_PASS_SYSTEM = """\
You write one standard-name record for the quantity a case file describes.
Return the name, the definition (the description) of the quantity, one paragraph
of documentation, the unit and the physics group. The name must be composed from
the vocabulary the case file supplies. The definition must state what the
quantity is and, where the source states it, how the source obtains it.
Use the family and neighbour blocks to see what separates this quantity from the
names nearest to it, and say that separation in the record.
Set blocker to a short type only when the case file does not determine a record;
otherwise leave it empty.
"""

STAGED_NAME_SYSTEM = """\
You are the first of two authors of one standard-name record.
First write the definition of the quantity the case file describes, then derive
the name from that definition, using the vocabulary the case file supplies. The
name states only what separates the quantity from its family and neighbours.
Return the name, the definition (the description), the unit and the physics
group. Set blocker to a short type only when the case file does not determine a
record; otherwise leave it empty.
"""

STAGED_DOCUMENTATION_SYSTEM = """\
You are the second of two authors of one standard-name record. You are given the
case file and the name and definition the first author wrote. Write the
documentation paragraph for that record: expand the definition, name the source
path it comes from, and state how this quantity differs from its family and
nearest neighbours.
Set blocker to a short type only when the case file does not determine the
record; otherwise leave it empty.
"""

IDENTITY_RULE = """\
The lead ruled that a name identifies and its definition specifies. The name
carries only what separates the quantity from its neighbours: the kind of
quantity, the physical object or locus it describes, and a relation or component
where the separation depends on it. Normalisations, weightings, averaging or
integration, conventions, coordinate frames and methods belong to the definition,
not to the name, unless omitting them would change which quantity is meant. The
definition states how the number is obtained and its unit.
"""

JUDGE_SYSTEM = """\
You are a blind adjudicator. You are given the committed reference answer for
one quantity and a submitted record. You are not told which structure or model
produced the record, and you are not shown the incumbent record. Judge only the
submitted record against the reference answer.
Return right when the submitted name is one of the right names and the
definition states every definition point (and the unit agrees where a unit is
given). Return blocked when the submitted record declares a blocker and the
reference answer cannot be derived from what the record states; otherwise
return wrong. For a wrong record name the failure class: qualifier-in-name,
definition-omits-convention, name-not-supported-by-source, unit-or-group-wrong,
or other. Cite the deciding text in the reason.
"""


# --------------------------------------------------------------------------- #
# Durable logs and cached inputs
# --------------------------------------------------------------------------- #


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
        handle.flush()


def load_reference() -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, str]]:
    data = json.loads(REFERENCE.read_text(encoding="utf-8"))
    members = json.loads(
        zlib.decompress(base64.b85decode(data["member_payload_zlib_base85"])).decode()
    )
    packets = json.loads(
        zlib.decompress(base64.b85decode(data["packet_payload_zlib_base85"])).decode()
    )
    return data, members, packets


def load_case_cache() -> dict[str, Any]:
    if CASE_CACHE.exists():
        return json.loads(CASE_CACHE.read_text(encoding="utf-8"))
    return {}


def save_case_cache(cache: dict[str, Any]) -> None:
    CASE_CACHE.parent.mkdir(parents=True, exist_ok=True)
    CASE_CACHE.write_text(
        json.dumps(cache, sort_keys=True, ensure_ascii=False), encoding="utf-8"
    )


def build_case_cache(
    members: list[dict[str, Any]], *, withhold: bool = False
) -> dict[str, Any]:
    from imas_codex.graph.client import GraphClient
    from imas_codex.standard_names.case_file import (
        build_name_case_file,
        render_case_file,
    )

    cache = load_case_cache()
    key = "withheld" if withhold else "full"
    bucket = cache.setdefault(key, {})
    missing = [member["id"] for member in members if member["id"] not in bucket]
    if not missing:
        return bucket
    with GraphClient() as client:
        for index, member_id in enumerate(missing, start=1):
            record = build_name_case_file(
                member_id,
                gc=client,
                withhold=("enclosing_structure_text",) if withhold else (),
            )
            bucket[member_id] = {
                "record": record.model_dump(),
                "text": render_case_file(record),
                "tokens": record.tokens,
                "completeness": record.completeness,
            }
            if index % 10 == 0:
                save_case_cache(cache)
                label = "withheld" if withhold else "full"
                print(
                    f"case files: {index}/{len(missing)} built ({label})",
                    file=sys.stderr,
                    flush=True,
                )
    save_case_cache(cache)
    return bucket


# --------------------------------------------------------------------------- #
# Prompts and model calls
# --------------------------------------------------------------------------- #


def failure_answer_text(answered: dict[str, int]) -> str:
    if not answered:
        return ""
    listed = ", ".join(
        f"{count} {failure}" for failure, count in sorted(answered.items()) if count
    )
    return (
        f"\nThis round answers the previous round's failures: {listed}. "
        "Do not repeat them: keep a normalisation, a weighting, an average, a "
        "convention or a method out of the name unless it changes which quantity "
        "is meant, and state each of them, with the unit, in the definition."
    )


def system_for(structure: str, version: str, answered: dict[str, int]) -> str:
    if structure == "one-pass":
        text = ONE_PASS_SYSTEM
    else:
        text = STAGED_NAME_SYSTEM
    if version == "v2":
        text = text + "\n" + IDENTITY_RULE + failure_answer_text(answered)
    return text


async def call_model(
    model: str,
    system: str,
    user: str,
    schema: type[BaseModel],
) -> tuple[BaseModel, dict[str, Any]]:
    from imas_codex.discovery.base.llm import acall_llm_structured

    started = time.time()
    out = await acall_llm_structured(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        response_model=schema,
        service="standard-names",
        max_tokens=MAX_TOKENS,
        timeout=900,
        max_retries=2,
        temperature=0.0,
        seed=0,
        local_schema_constraint=model.startswith("local/"),
    )
    return out.parsed, {
        "cost": float(getattr(out, "cost", 0.0) or 0.0),
        "input_tokens": int(getattr(out, "input_tokens", 0) or 0),
        "output_tokens": int(getattr(out, "output_tokens", 0) or 0),
        "seconds": time.time() - started,
    }


async def generate_one(
    structure: str,
    model: str,
    case_text: str,
    version: str,
    answered: dict[str, int],
    semaphores: dict[str, asyncio.Semaphore],
) -> dict[str, Any]:
    """Write one record under one structure, reporting spend and elapsed time."""
    lane = "local" if model.startswith("local/") else "frontier"
    system = system_for(structure, version, answered)
    started = time.time()
    cost = 0.0
    input_tokens = 0
    output_tokens = 0
    prompt_hash = digest_text(system)
    try:
        async with semaphores[lane]:
            if structure == "one-pass":
                parsed, meta = await call_model(model, system, case_text, RecordDraft)
                draft = parsed.model_dump()
            else:
                first, meta = await call_model(model, system, case_text, NameStage)
                cost += meta["cost"]
                input_tokens += meta["input_tokens"]
                output_tokens += meta["output_tokens"]
                stage_one_parts = [
                    f"name: {first.name}",
                    f"definition: {first.description}",
                    f"unit: {first.unit}",
                    f"group: {first.group}",
                ]
                second, meta_two = await call_model(
                    model,
                    STAGED_DOCUMENTATION_SYSTEM,
                    case_text
                    + "\nThe first author wrote:\n"
                    + "\n".join(stage_one_parts),
                    DocStage,
                )
                draft = {
                    "name": first.name,
                    "description": first.description,
                    "documentation": second.documentation,
                    "unit": first.unit,
                    "group": first.group,
                    "blocker": first.blocker or second.blocker,
                }
                meta = meta_two
        cost += meta["cost"]
        input_tokens += meta["input_tokens"]
        output_tokens += meta["output_tokens"]
    except Exception as exc:  # a failed call is a blocked member, never a right
        draft = {
            "name": "",
            "description": "",
            "documentation": "",
            "unit": "",
            "input_tokens": 0,
            "group": "",
            "blocker": f"generation-failure:{type(exc).__name__}",
        }
    return {
        "structure": structure,
        "model": model,
        "prompt_version": version,
        "prompt_hash": prompt_hash,
        "draft": draft,
        "cost": cost,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "seconds": time.time() - started,
    }


def judge_input(member: dict[str, Any], draft: dict[str, Any]) -> str:
    points = "\n".join(f"- {point}" for point in member["definition_points"])
    names = ", ".join(member["right_names"])
    return (
        f"Reference answer:\n"
        f"right names (any one is right): {names}\n"
        f"unit: {member['unit'] or '(none declared)'}\n"
        f"group: {member['group'] or '(none declared)'}\n"
        f"definition points:\n{points}\n\nSubmitted record:\n"
        f"name: {draft.get('name', '')}\n"
        f"definition: {draft.get('description', '')}\n"
        f"documentation: {draft.get('documentation', '')}\n"
        f"unit: {draft.get('unit', '')}\n"
        f"group: {draft.get('group', '')}\n"
        f"declared blocker: {draft.get('blocker') or 'none'}\n"
    )


async def judge_one(
    member: dict[str, Any],
    draft: dict[str, Any],
    semaphores: dict[str, asyncio.Semaphore],
) -> dict[str, Any]:
    started = time.time()
    async with semaphores["judge"]:
        parsed, meta = await call_model(
            JUDGE_MODEL, JUDGE_SYSTEM, judge_input(member, draft), Judgment
        )
    verdict = parsed.verdict.strip().lower()
    if verdict not in bench.RECORD_VERDICTS:
        verdict = "wrong"
    failure_class = parsed.failure_class.strip()
    if verdict == "wrong" and failure_class not in FAILURE_CLASSES:
        failure_class = "other"
    if verdict != "wrong":
        failure_class = ""
    return {
        "verdict": verdict,
        "failure_class": failure_class,
        "reason": parsed.reason,
        "cost": meta["cost"],
        "seconds": time.time() - started,
    }


# --------------------------------------------------------------------------- #
# Running the arms
# --------------------------------------------------------------------------- #


def key_of(round_name: str, split: str, arm: str, member: str) -> str:
    return f"{round_name}|{split}|{arm}|{member}"


def squash(text: Any) -> str:
    return " ".join(str(text or "").split()).lower()


def incumbent_draft(case: dict[str, Any]) -> dict[str, str] | None:
    content = case["record"]["content"]
    incumbent = content.get("incumbent")
    if not incumbent:
        return None
    return {
        "name": incumbent.get("name") or "",
        "description": incumbent.get("description") or "",
        "documentation": incumbent.get("documentation") or "",
        "unit": incumbent.get("unit") or "",
        "group": "",
        "blocker": "",
    }


def changed_from_incumbent(draft: dict[str, Any], case: dict[str, Any]) -> bool | None:
    incumbent = incumbent_draft(case)
    if incumbent is None:
        return None
    return any(
        squash(draft.get(field)) != squash(incumbent.get(field))
        and bool(squash(draft.get(field)))
        and bool(squash(incumbent.get(field)))
        for field in ("name", "description", "documentation", "unit")
    )


def arm_objects() -> dict[str, bench.RecordBenchmarkArm]:
    arms = {
        name: bench.RecordBenchmarkArm(name=name, structure=structure, model=model)
        for name, structure, model in ARM_SPECS
    }
    arms["incumbent"] = bench.RecordBenchmarkArm(
        name="incumbent", structure="", model="", incumbent_unchanged=True
    )
    return arms


async def run_arm(
    arm_name: str,
    structure: str,
    model: str,
    members: list[dict[str, Any]],
    split: str,
    round_name: str = "final",
    version: str = "v2",
    answered: dict[str, int] | None = None,
    bucket: dict[str, Any] | None = None,
) -> None:
    bucket = bucket or load_case_cache()["full"]
    seen = {
        key_of(row["round"], row["split"], row["arm"], row["member"])
        for row in read_jsonl(RECORD_LOG)
    }
    semaphores = {
        "local": asyncio.Semaphore(CONCURRENCY["local"]),
        "frontier": asyncio.Semaphore(CONCURRENCY["frontier"]),
    }

    async def one(member: dict[str, Any]) -> None:
        key = key_of(round_name, split, arm_name, member["id"])
        if key in seen:
            return
        case = bucket[member["id"]]
        out = await generate_one(
            structure, model, case["text"], version, answered or {}, semaphores
        )
        row = {
            "round": round_name,
            "split": split,
            "arm": arm_name,
            "member": member["id"],
            "case_tokens": case["tokens"],
        }
        row.update(out)
        append_jsonl(RECORD_LOG, row)

    await asyncio.gather(*(one(member) for member in members))


async def run_incumbent(
    members: list[dict[str, Any]],
    split: str,
    round_name: str,
    bucket: dict[str, Any],
) -> None:
    seen = {
        key_of(row["round"], row["split"], row["arm"], row["member"])
        for row in read_jsonl(RECORD_LOG)
    }
    for member in members:
        key = key_of(round_name, split, "incumbent", member["id"])
        if key in seen:
            continue
        case = bucket[member["id"]]
        draft = incumbent_draft(case)
        row = {
            "round": round_name,
            "split": split,
            "arm": "incumbent",
            "member": member["id"],
            "case_tokens": case["tokens"],
            "structure": "",
            "model": "",
            "prompt_version": "none",
            "prompt_hash": "",
            "draft": draft
            or {
                "name": "",
                "description": "",
                "documentation": "",
                "unit": "",
                "group": "",
                "blocker": "no-incumbent",
            },
            "cost": 0.0,
            "input_tokens": 0,
            "output_tokens": 0,
            "seconds": 0.0,
        }
        append_jsonl(RECORD_LOG, row)


async def score_split(
    round_name: str,
    split: str,
    bucket: dict[str, Any],
) -> None:
    verdicts_seen = {
        key_of(row["round"], row["split"], row["arm"], row["member"])
        for row in read_jsonl(VERDICT_LOG)
    }
    members_by_id = {member["id"]: member for member in load_reference()[1]}
    case_bucket = bucket
    semaphores = {
        "local": asyncio.Semaphore(CONCURRENCY["local"]),
        "frontier": asyncio.Semaphore(CONCURRENCY["frontier"]),
        "judge": asyncio.Semaphore(CONCURRENCY["judge"]),
    }
    records = [
        row
        for row in read_jsonl(RECORD_LOG)
        if row["round"] == round_name and row["split"] == split
    ]

    async def one(record: dict[str, Any]) -> None:
        arm = record["arm"]
        member_id = record["member"]
        key = key_of(round_name, split, arm, member_id)
        if key in verdicts_seen:
            return
        member = members_by_id[member_id]
        if member_id not in case_bucket:
            return
        try:
            verdict = await judge_one(member, record["draft"], semaphores)
        except Exception as exc:  # a failed judge must not read as a right
            verdict = {
                "verdict": "blocked",
                "failure_class": f"judge-failure:{type(exc).__name__}",
                "reason": f"the blind adjudicator did not answer: {exc}"[:300],
                "cost": 0.0,
                "seconds": 0.0,
            }
        append_jsonl(
            VERDICT_LOG,
            {
                "round": round_name,
                "split": split,
                "arm": arm,
                "member": member_id,
                **verdict,
            },
        )

    await asyncio.gather(*(one(record) for record in records))


def to_bench_rows(
    round_name: str,
    split: str,
    members: list[dict[str, Any]],
    bucket: dict[str, Any],
) -> list[bench.RecordBenchmarkRow]:
    member_index = {member["id"]: member for member in members}
    records = {
        (row["arm"], row["member"]): row
        for row in read_jsonl(RECORD_LOG)
        if row["round"] == round_name and row["split"] == split
    }
    verdicts = {
        (row["arm"], row["member"]): row
        for row in read_jsonl(VERDICT_LOG)
        if row["round"] == round_name and row["split"] == split
    }
    rows: list[bench.RecordBenchmarkRow] = []
    for (arm, member_id), record in sorted(records.items()):
        verdict_row = verdicts.get((arm, member_id))
        if verdict_row is None or member_id not in member_index:
            continue
        verdict = verdict_row["verdict"]
        blocker = None
        if verdict == "blocked":
            blocker = (
                record["draft"].get("blocker")
                or verdict_row.get("failure_class")
                or "adjudicator-blocked"
            )
        rows.append(
            bench.RecordBenchmarkRow(
                member=member_id,
                arm=arm,
                verdict=verdict,
                blocker=blocker,
                changed_incumbent=changed_from_incumbent(
                    record["draft"], bucket.get(member_id, {"record": {"content": {}}})
                ),
                incumbent_right=member_index[member_id]["verdict"] == "sound",
                cost_usd=float(record["cost"]),
                seconds=float(record["seconds"]),
                input_tokens=int(record["input_tokens"]),
                output_tokens=int(record["output_tokens"]),
            )
        )
    return rows


def failure_counts(round_name: str, split: str) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in read_jsonl(VERDICT_LOG):
        if (
            row["round"] == round_name
            and row["split"] == split
            and row["verdict"] == "wrong"
        ):
            counts[row.get("failure_class") or "other"] += 1
    return dict(sorted(counts.items()))


def report_arms(
    rows: list[bench.RecordBenchmarkRow],
    arms: dict[str, bench.RecordBenchmarkArm],
    split_size: int,
) -> list[dict[str, Any]]:
    present = {row.arm for row in rows}
    reports = []
    for name, arm in arms.items():
        if name not in present:
            continue
        report = bench.summarize_record_arm(rows, arm)
        payload = report.to_dict()
        payload["complete"] = report.total == split_size
        reports.append(payload)
    return sorted(reports, key=lambda item: (-item["share_right"], item["arm"]))


def dependent_members(
    members: list[dict[str, Any]], bucket: dict[str, Any]
) -> list[str]:
    dependent = []
    for member in members:
        case = bucket.get(member["id"])
        if not case:
            continue
        sources = case["record"]["content"].get("sources") or []
        own_text = squash(" ".join(source.get("text") or "" for source in sources))
        enclosing = squash(
            " ".join(
                statement.get("text") or ""
                for source in sources
                for statement in (source.get("enclosing") or {}).get("statements") or []
            )
        )
        quote = squash(member.get("evidence_quote") or "")
        if quote and quote not in own_text and quote in enclosing:
            dependent.append(member["id"])
    return dependent


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #


def pick_best(reports: list[dict[str, Any]]) -> dict[str, Any] | None:
    repair_names = {name for name, _, _ in ARM_SPECS}
    candidates = [
        report
        for report in reports
        if report["arm"] in repair_names and report.get("complete")
    ]
    if not candidates:
        candidates = [report for report in reports if report["arm"] in repair_names]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda report: (
            report["share_right"],
            -report["cost_per_name"],
            report["arm"],
        ),
    )


def spec_for(arm_name: str) -> tuple[str, str]:
    for name, structure, model in ARM_SPECS:
        if name == arm_name:
            return structure, model
    raise ValueError(f"unknown arm {arm_name!r}")


def build_document(
    reference: dict[str, Any],
    members: list[dict[str, Any]],
    split: dict[str, str],
    development: list[dict[str, Any]],
    held_out: list[dict[str, Any]],
    bucket: dict[str, Any],
    withheld_bucket: dict[str, Any],
    best_arm: str,
    revision: str,
) -> dict[str, Any]:
    arms = arm_objects()
    structure, model = spec_for(best_arm)
    withheld_name = f"withheld/{best_arm}"
    arms[withheld_name] = bench.RecordBenchmarkArm(
        name=withheld_name,
        structure=structure,
        model=model,
        withheld_enclosing_text=True,
    )

    development_1 = to_bench_rows("round-1", "development", development, bucket)
    development_2 = to_bench_rows("round-2", "development", development, bucket)
    held_out_rows = to_bench_rows("final", "held_out", held_out, bucket)
    reports_1 = report_arms(development_1, arms, len(development))
    reports_2 = report_arms(development_2, arms, len(development))
    final_reports = report_arms(held_out_rows, arms, len(held_out))

    dependent = dependent_members(held_out, bucket)
    final_objects = [
        bench.summarize_record_arm(held_out_rows, arm)
        for name, arm in arms.items()
        if any(row.arm == name for row in held_out_rows)
    ]
    controls = bench.evaluate_controls(
        final_objects,
        held_out_rows,
        incumbent_arm="incumbent",
        withheld_arm=withheld_name,
        withheld_compared_to=best_arm,
        dependent_members=dependent,
        seed=SPLIT_SEED,
    )
    whole_split = bench.compare_record_arms(
        held_out_rows, best_arm, withheld_name, seed=SPLIT_SEED
    )
    controls["withheld"]["whole_split"] = whole_split
    best_repair_share = max(
        (
            report["share_right"]
            for report in final_reports
            if report["arm"] in {name for name, _, _ in ARM_SPECS}
        ),
        default=0.0,
    )
    admission = (
        "the controls separated"
        if controls["measuring"]
        else (
            "the controls did not separate: "
            f"the incumbent-unchanged arm scored {percent(controls['incumbent'].get('share_right', 0.0))} "
            f"against {percent(best_repair_share)} for the best repairing arm; "
            f"the withheld-text arm scored {whole_split['right_b']}/{whole_split['members']} "
            f"against {whole_split['right_a']}/{whole_split['members']} "
            f"({percent(whole_split['right_a'] / whole_split['members'])}), "
            f"losing {whole_split['only_a_right']} members and gaining {whole_split['only_b_right']}. "
            "No arm difference is called a win on this run."
        )
    )
    repair_names = [
        report["arm"]
        for report in final_reports
        if report["arm"] in {name for name, _, _ in ARM_SPECS}
    ]
    comparisons = []
    for index, first in enumerate(repair_names):
        for second in repair_names[index + 1 :]:
            comparisons.append(
                bench.compare_record_arms(held_out_rows, first, second, seed=SPLIT_SEED)
            )

    round_changes = []
    answered: dict[str, int] = {}
    for row in read_jsonl(VERDICT_LOG):
        if (
            row["round"] == "round-1"
            and row["split"] == "development"
            and row["verdict"] == "wrong"
        ):
            answered[row.get("failure_class") or "other"] = (
                answered.get(row.get("failure_class") or "other", 0) + 1
            )
    round_changes.append(
        {
            "round": "round-1",
            "prompt_version": "v1",
            "split": "development",
            "answered_failures": {},
            "change": "baseline prompt: write the record from the case file with no identity rule stated",
            "arms": reports_1,
        }
    )
    round_changes.append(
        {
            "round": "round-2",
            "prompt_version": "v2",
            "split": "development",
            "answered_failures": answered,
            "change": (
                "the prompt states the lead's identity rule and the definition's"
                " duty to carry normalisation, convention, method and unit,"
                " answering every failure class the first round recorded"
            ),
            "arms": reports_2,
        }
    )

    generation_cost = sum(float(row.get("cost", 0.0)) for row in read_jsonl(RECORD_LOG))
    judging_cost = sum(float(row.get("cost", 0.0)) for row in read_jsonl(VERDICT_LOG))
    from imas_codex.standard_names.advisory_benchmark import summarize_costs

    spend = summarize_costs(generation_cost, judging_cost, SPEND_CEILING_USD)

    repair_final = [
        report
        for report in final_reports
        if report["arm"] in {name for name, _, _ in ARM_SPECS}
    ]
    options = []
    for report in repair_final:
        options.append(
            {
                "arm": report["arm"],
                "structure": report["structure"],
                "model": report["model"],
                "share_right": report["share_right"],
                "interval": report["interval"],
                "complete": report["complete"],
                "blocked_by_type": report["blocked_by_type"],
                "changed_right_incumbent_share": report[
                    "changed_right_incumbent_share"
                ],
                "cost_per_name": report["cost_per_name"],
                "seconds_per_name": report["seconds_per_name"],
                "tokens_read_per_name": report["tokens_read_per_name"],
            }
        )
    recommended = max(
        (option for option in options if option["complete"]),
        key=lambda option: (option["share_right"], -option["cost_per_name"]),
        default=None,
    )
    if recommended is None:
        recommended = max(
            options, key=lambda option: option["share_right"], default=None
        )

    return {
        "schema": bench.RECORD_BENCHMARK_SCHEMA,
        "built_at": datetime.now(UTC).isoformat(),
        "revision": revision,
        "reference": {
            "path": str(REFERENCE.relative_to(REPO)),
            "members": len(members),
            "defective": sum(1 for member in members if member["verdict"] != "sound"),
            "development": len(development),
            "held_out": len(held_out),
            "blind": reference.get("blind", {}),
            "rule": reference.get("rule", ""),
        },
        "judge": {
            "model": JUDGE_MODEL,
            "blind_to": "the arm and the incumbent; each verdict sees only the submitted record and the committed answer",
        },
        "split_seed": SPLIT_SEED,
        "split": {
            "development": len(development),
            "held_out": len(held_out),
            "assignment_digest": digest_json(split),
        },
        "rounds": round_changes,
        "final": {
            "split": "held_out",
            "prompt_version": "v2",
            "best_development_arm": best_arm,
            "arms": final_reports,
            "comparisons": comparisons,
            "controls": controls,
            "measuring": controls["measuring"],
            "dependent_members": dependent,
        },
        "decision": {
            "question": "which structure and model should generate the record",
            "options": options,
            "recommended": recommended,
            "ruling": "awaiting the lead's ruling",
            "admission": admission,
        },
        "not_run": [
            {
                "arm": "kimi/local",
                "reason": "the B300 server that will serve Kimi is not up",
                "command": KIMI_COMMAND,
            },
            {
                "arm": "subscription/worker-lane",
                "reason": "the subscription worker lane was not exercised by this run",
                "command": (
                    "uv run --no-sync python docs/evidence/fragments/"
                    "sn-case-file-generation/benchmark.py --stage measure --models subscription"
                ),
            },
        ],
        "spend": {
            "ceiling_usd": SPEND_CEILING_USD,
            "generation_usd": generation_cost,
            "judging_usd": judging_cost,
            "total_usd": spend.total_cost,
            "remaining_usd": spend.remaining_authority,
            "within_ceiling": spend.within_authorized_ceiling,
        },
        "rows": [
            {
                "member": row.member,
                "arm": row.arm,
                "verdict": row.verdict,
                "blocker": row.blocker,
                "changed_incumbent": row.changed_incumbent,
                "incumbent_right": row.incumbent_right,
                "cost_usd": row.cost_usd,
                "seconds": row.seconds,
                "input_tokens": row.input_tokens,
                "output_tokens": row.output_tokens,
            }
            for row in held_out_rows
        ],
        "case_files": {
            "members": len(bucket),
            "withheld_members": len(withheld_bucket),
            "incomplete": sum(
                1
                for case in bucket.values()
                if (case.get("completeness") or {}).get("verdict") != "complete"
            ),
        },
    }


def percent(value: float) -> str:
    return f"{100.0 * value:.1f} %"


def render_html(document: dict[str, Any]) -> str:
    from html import escape

    reference = document["reference"]
    spend = document["spend"]
    lines = [
        "<!doctype html>",
        '<html lang="en"><head><meta charset="utf-8">',
        "<title>Record benchmark</title>",
        "<style>body{font:14px/1.45 system-ui,sans-serif;max-width:1100px;margin:2rem auto;padding:0 1rem}"
        "table{border-collapse:collapse;margin:1rem 0;width:100%}"
        "th,td{border-bottom:1px solid #ddd;padding:.35rem .5rem;text-align:left}"
        "th{background:#f6f6f6}code{font-size:.9em}</style></head><body>",
        "<h1>Record generation benchmark</h1>",
        f"<p>Revision <code>{escape(document['revision'])}</code>. "
        f"Reference set {reference['members']} members, "
        f"{reference['defective']} defective: "
        f"{reference['development']} development, {reference['held_out']} held out. "
        f"Blind adjudicator: <code>{escape(document['judge']['model'])}</code>, "
        f"told neither the arm nor the incumbent.</p>",
        f"<p>Spend USD {spend['total_usd']:.2f} of a USD {spend['ceiling_usd']:.0f} ceiling "
        f"(generation {spend['generation_usd']:.2f}, judging {spend['judging_usd']:.2f}). "
        f"Within ceiling: {str(spend['within_ceiling']).lower()}.</p>",
        "<h2>Rounds</h2>",
    ]
    for round_row in document["rounds"]:
        lines.append(
            f"<h3>{escape(round_row['round'])} ({escape(round_row['prompt_version'])}, "
            f"{escape(round_row['split'])})</h3>"
        )
        if round_row.get("answered_failures"):
            lines.append(
                "<p>Answers the previous round's failures: "
                + escape(str(round_row["answered_failures"]))
                + f". Change: {escape(round_row['change'])}</p>"
            )
        lines.append(_arm_table(round_row["arms"]))

    lines.append("<h2>Final arms, held out</h2>")
    lines.append(_arm_table(document["final"]["arms"]))
    lines.append("<h2>Controls</h2>")
    lines.append(
        f"<p>Measuring: <strong>{str(document['final']['measuring']).lower()}</strong>. "
        f"Incumbent-unchanged separated: "
        f"{str(document['final']['controls']['incumbent']['separated']).lower()}; "
        f"withheld-text separated: "
        f"{str(document['final']['controls']['withheld']['separated']).lower()} "
        f"over {document['final']['controls']['withheld'].get('dependent_members', 0)} "
        "members</p>"
    )
    lines.append("<h2>Member-by-member comparisons</h2>")
    lines.append(
        "<table><tr><th>arms</th><th>members</th><th>only first right</th>"
        "<th>only second right</th><th>difference</th><th>95 % interval</th>"
        "<th>called a win</th></tr>"
    )
    for comparison in document["final"]["comparisons"]:
        low, high = comparison["interval"]
        lines.append(
            "<tr><td>"
            + escape(" > ".join(comparison["arms"]))
            + f"</td><td>{comparison['members']}</td>"
            f"<td>{comparison['only_a_right']}</td><td>{comparison['only_b_right']}</td>"
            f"<td>{percent(comparison['difference'])}</td>"
            f"<td>[{percent(low)}, {percent(high)}]</td>"
            f"<td>{str(comparison['called_win']).lower()}</td></tr>"
        )
    lines.append("</table>")

    lines.append("<h2>The choice put to the lead</h2>")
    lines.append(_decision_table(document["decision"]))
    lines.append(
        "<h2>Not run</h2><table><tr><th>arm</th><th>reason</th><th>command</th></tr>"
    )
    for row in document["not_run"]:
        lines.append(
            f"<tr><td>{escape(row['arm'])}</td><td>{escape(row['reason'])}</td>"
            f"<td><code>{escape(row['command'])}</code></td></tr>"
        )
    lines.append("</table>")
    lines.append("</body></html>")
    return "\n".join(lines)


def _arm_table(reports: list[dict[str, Any]]) -> str:
    from html import escape

    lines = [
        "<table><tr><th>arm</th><th>structure</th><th>model</th><th>right</th>"
        "<th>95 % interval</th><th>blocked</th><th>changed a right incumbent</th>"
        "<th>cost/name</th><th>s/name</th><th>tokens read/name</th></tr>"
    ]
    for report in reports:
        low, high = report["interval"]
        lines.append(
            f"<tr><td>{escape(report['arm'])}</td>"
            f"<td>{escape(report['structure'] or 'copy')}</td>"
            f"<td>{escape(report['model'] or 'none')}</td>"
            f"<td>{report['right']}/{report['total']} ({percent(report['share_right'])})</td>"
            f"<td>[{percent(low)}, {percent(high)}]</td>"
            f"<td>{report['blocked']} {escape(str(report['blocked_by_type']))}</td>"
            f"<td>{report['changed_right_incumbent']}/{report['right_incumbent_total']}</td>"
            f"<td>{report['cost_per_name']:.4f}</td>"
            f"<td>{report['seconds_per_name']:.1f}</td>"
            f"<td>{report['tokens_read_per_name']:.0f}</td></tr>"
        )
    lines.append("</table>")
    return "\n".join(lines)


def _decision_table(decision: dict[str, Any]) -> str:
    from html import escape

    lines = [
        "<table><tr><th>option</th><th>structure</th><th>model</th><th>right</th>"
        "<th>95 % interval</th><th>complete</th><th>cost/name</th></tr>"
    ]
    for option in decision["options"]:
        low, high = option["interval"]
        lines.append(
            f"<tr><td>{escape(option['arm'])}</td>"
            f"<td>{escape(option['structure'])}</td><td>{escape(option['model'])}</td>"
            f"<td>{percent(option['share_right'])}</td>"
            f"<td>[{percent(low)}, {percent(high)}]</td>"
            f"<td>{str(option['complete']).lower()}</td>"
            f"<td>{option['cost_per_name']:.4f}</td></tr>"
        )
    lines.append("</table>")
    recommended = decision.get("recommended") or {}
    if recommended:
        lines.append(
            f"<p>Recommended by the measured figures, pending the lead's ruling: "
            f"<strong>{escape(recommended['arm'])}</strong> at "
            f"{percent(recommended['share_right'])} right on the held-out half. "
            f"Ruling: {escape(decision.get('ruling') or 'awaiting')}.</p>"
        )
    return "\n".join(lines)


def render_fragment(document: dict[str, Any]) -> str:
    from html import escape

    final = document["final"]
    controls = final["controls"]
    spend = document["spend"]
    lines = [
        '<section data-reckon="evidence">',
        '<h3 id="scfg-benchmark-record-benchmark">A blind record benchmark</h3>',
        "<p>Every arm wrote one record per reference member from its code-built "
        "case file. The blind adjudicator saw neither the arm nor the incumbent. "
        f"The final figures are the held-out half ({document['reference']['held_out']} members), "
        "taken once.</p>",
        '<div id="scfg-benchmark-final">' + _arm_table(final["arms"]) + "</div>",
        '<h4 id="scfg-benchmark-controls">Controls</h4>',
        "<p>Incumbent-unchanged separated: "
        f"{str(controls['incumbent']['separated']).lower()}; withheld-text separated: "
        f"{str(controls['withheld']['separated']).lower()} on "
        f"{controls['withheld'].get('dependent_members', 0)} members whose right "
        f"answer rests on the enclosing text. Measuring: {str(final['measuring']).lower()}.</p>",
        f'<p id="scfg-benchmark-admission">{escape(document["decision"]["admission"])}</p>',
        '<h4 id="scfg-benchmark-choice">The choice</h4>',
        _decision_table(document["decision"]),
        f"<p>Spend USD {spend['total_usd']:.2f} against the USD "
        f"{spend['ceiling_usd']:.0f} ceiling.</p>",
        "</section>",
    ]
    return "\n".join(lines)


def render_figure(document: dict[str, Any]) -> None:
    from html import escape

    reports = document["final"]["arms"]
    width, height = 1100, 60 + 52 * max(1, len(reports))
    left, plot_width = 260, 720
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        'font-family="system-ui, sans-serif" font-size="13">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="20" y="28" font-size="15" font-weight="600">'
        "Share right on the held-out half, with a 95 % interval</text>",
        f'<line x1="{left}" y1="45" x2="{left}" y2="{height - 15}" stroke="#333"/>',
    ]
    for index, report in enumerate(reports):
        y = 60 + index * 52
        low, high = report["interval"]
        bar_width = report["share_right"] * plot_width
        color = (
            "#4b7bec"
            if "withheld" not in report["arm"] and "incumbent" not in report["arm"]
            else "#e07a5f"
        )
        lines.append(
            f'<text x="20" y="{y + 16}" font-size="12">{escape(report["arm"])}</text>'
        )
        lines.append(
            f'<rect x="{left}" y="{y}" width="{bar_width:.1f}" height="22" fill="{color}"/>'
        )
        lines.append(
            f'<line x1="{left + low * plot_width:.1f}" y1="{y - 4}" '
            f'x2="{left + high * plot_width:.1f}" y2="{y - 4}" stroke="#222"/>'
        )
        lines.append(
            f'<line x1="{left + low * plot_width:.1f}" y1="{y - 8}" '
            f'x2="{left + low * plot_width:.1f}" y2="{y + 8}" stroke="#222"/>'
        )
        lines.append(
            f'<line x1="{left + high * plot_width:.1f}" y1="{y - 8}" x2="{left + high * plot_width:.1f}" y2="{y + 8}" stroke="#222"/>'
        )
        lines.append(
            f'<text x="{left + plot_width + 8}" y="{y + 16}" font-size="12">'
            f"{percent(report['share_right'])} [{percent(low)}, {percent(high)}]</text>"
        )
    for tick in range(0, 101, 20):
        x = left + plot_width * tick / 100
        lines.append(
            f'<line x1="{x:.1f}" y1="45" x2="{x:.1f}" y2="{height - 15}" stroke="#eee"/>'
        )
        lines.append(
            f'<text x="{x - 10:.1f}" y="{height - 2}" font-size="11">{tick} %</text>'
        )
    lines.append("</svg>")
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    FIGURE.write_text("\n".join(lines), encoding="utf-8")


# --------------------------------------------------------------------------- #
# Entrypoint
# --------------------------------------------------------------------------- #


def split_inputs() -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    dict[str, str],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    reference, members, _ = load_reference()
    strata = {member["id"]: member["verdict"] for member in members}
    split = bench.assign_development_split(
        [member["id"] for member in members], seed=SPLIT_SEED, strata=strata
    )
    development = [member for member in members if split[member["id"]] == "development"]
    held_out = [member for member in members if split[member["id"]] == "held_out"]
    return reference, members, split, development, held_out


async def measure(
    development: list[dict[str, Any]],
    held_out: list[dict[str, Any]],
    members: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], str]:
    bucket = build_case_cache(members)
    for name, structure, model in ARM_SPECS:
        await run_arm(
            name,
            structure,
            model,
            development,
            "development",
            "round-1",
            "v1",
            {},
            bucket,
        )
    await score_split("round-1", "development", bucket)
    answered = failure_counts("round-1", "development")
    print(f"round-1 failure classes: {answered}", file=sys.stderr, flush=True)

    for name, structure, model in ARM_SPECS:
        await run_arm(
            name,
            structure,
            model,
            development,
            "development",
            "round-2",
            "v2",
            answered,
            bucket,
        )
    await score_split("round-2", "development", bucket)
    round_two_reports = report_arms(
        to_bench_rows("round-2", "development", development, bucket),
        arm_objects(),
        len(development),
    )
    best = pick_best(round_two_reports)
    if best is None:
        raise RuntimeError("no development arm produced a report")
    best_name = best["arm"]
    structure, model = spec_for(best_name)

    withheld_bucket = build_case_cache(held_out, withhold=True)
    for name, structure_name, model_name in ARM_SPECS:
        await run_arm(
            name,
            structure_name,
            model_name,
            held_out,
            "held_out",
            "final",
            "v2",
            answered,
            bucket,
        )
    await run_arm(
        f"withheld/{best_name}",
        structure,
        model,
        held_out,
        "held_out",
        "final",
        "v2",
        answered,
        withheld_bucket,
    )
    await run_incumbent(held_out, "held_out", "final", bucket)
    await score_split("final", "held_out", bucket)
    return bucket, withheld_bucket, best_name


def write_report(
    bucket: dict[str, Any],
    withheld_bucket: dict[str, Any],
    best_name: str,
    revision: str | None = None,
) -> dict[str, Any]:
    import subprocess

    reference, members, split, development, held_out = split_inputs()
    if not revision:
        revision = subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
        ).strip()
    document = build_document(
        reference,
        members,
        split,
        development,
        held_out,
        bucket,
        withheld_bucket,
        best_name,
        revision,
    )
    DATA.write_text(
        json.dumps(document, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    PAGE.write_text(render_html(document), encoding="utf-8")
    FRAGMENT.write_text(render_fragment(document), encoding="utf-8")
    render_figure(document)
    return document


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("build", "measure", "report", "all"),
        default="all",
        help="which part of the run to execute; every stage resumes its logs",
    )
    parser.add_argument(
        "--models",
        choices=("all", "local", "frontier", "kimi", "subscription"),
        default="all",
        help="which model lanes this run may use",
    )
    args = parser.parse_args()

    if args.models in {"kimi", "subscription"}:
        print(
            f"{args.models} is not run: the server or lane is not up. "
            f"The command that will run it: {KIMI_COMMAND if args.models == 'kimi' else 'see benchmark.json not_run'}",
            file=sys.stderr,
        )
        return 0

    _, members, _, development, held_out = split_inputs()
    bucket = load_case_cache().get("full", {})
    withheld_bucket = load_case_cache().get("withheld", {})
    best_name = ""

    if args.stage in {"build", "all"}:
        bucket = build_case_cache(members)
        withheld_bucket = build_case_cache(held_out, withhold=True)

    if args.stage in {"measure", "all"}:
        if not bucket:
            print("no case files: run --stage build first", file=sys.stderr)
            return 1
        bucket, withheld_bucket, best_name = asyncio.run(
            measure(development, held_out, members)
        )

    if args.stage in {"report", "all"}:
        if not best_name:
            reports = report_arms(
                to_bench_rows("round-2", "development", development, bucket),
                arm_objects(),
                len(development),
            )
            best = pick_best(reports)
            if best is None:
                print("no round-2 report to choose a best arm from", file=sys.stderr)
                return 1
            best_name = best["arm"]
        document = write_report(bucket, withheld_bucket, best_name)
        reference = document["reference"]
        print(
            json.dumps(
                {
                    "reference_members": reference["members"],
                    "held_out": reference["held_out"],
                    "measuring": document["final"]["measuring"],
                    "recommended": document["decision"].get("recommended"),
                    "spend": document["spend"],
                },
                indent=2,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
