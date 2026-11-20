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
import random
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
    'KIMI_BASE_URL="${KIMI_BASE_URL:?set KIMI_BASE_URL to the B300 server /v1 root}" '
    "uv run --no-sync python "
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
    members: list[dict[str, Any]],
    bucket: dict[str, Any],
    right_members: set[str],
) -> list[str]:
    """Members whose right answer rests only on the enclosing text, and that
    the full arm actually got right.

    Restricting to members the full arm answered correctly is what makes the
    withheld control a comparison: on this subset the full arm has something
    to lose. Without it the control could count a member the full arm already
    failed, and withholding a text no arm used would look like a failure of
    the text.
    """
    dependent = []
    for member in members:
        if member["id"] not in right_members:
            continue
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

    full_right = {
        row.member for row in held_out_rows if row.arm == best_arm and row.right
    }
    dependent = dependent_members(held_out, bucket, full_right)
    final_objects = [
        bench.summarize_record_arm(held_out_rows, arm)
        for name, arm in arms.items()
        if any(row.arm == name for row in held_out_rows)
    ]
    if dependent:
        controls = bench.evaluate_controls(
            final_objects,
            held_out_rows,
            incumbent_arm="incumbent",
            withheld_arm=withheld_name,
            withheld_compared_to=best_arm,
            dependent_members=dependent,
            seed=SPLIT_SEED,
        )
    else:
        controls = bench.evaluate_controls(
            final_objects,
            held_out_rows,
            incumbent_arm="incumbent",
            seed=SPLIT_SEED,
        )
        controls["withheld"] = {
            "arm": withheld_name,
            "separated": False,
            "dependent_members": 0,
            "unmeasurable": (
                "no held-out member has its right answer only in the enclosing "
                f"text and is right for {best_arm}"
            ),
        }
    whole_split = bench.compare_record_arms(
        held_out_rows, best_arm, withheld_name, seed=SPLIT_SEED
    )
    controls["withheld"]["whole_split"] = whole_split
    try:
        bench.assert_controls_separate(controls)
        controls["refusal"] = None
    except bench.BenchmarkControlDidNotSeparate as exc:
        controls["refusal"] = str(exc)
    best_repair_share = max(
        (
            report["share_right"]
            for report in final_reports
            if report["arm"] in {name for name, _, _ in ARM_SPECS}
        ),
        default=0.0,
    )
    if controls["measuring"]:
        admission = "the controls separated"
    elif controls["withheld"].get("unmeasurable"):
        admission = (
            "the controls did not separate: "
            f"the incumbent-unchanged arm scored {percent(controls['incumbent'].get('share_right', 0.0))} "
            f"against {percent(best_repair_share)} for the best repairing arm; "
            "the withheld-text control is unmeasurable because "
            f"{controls['withheld']['unmeasurable']}."
        )
    else:
        admission = (
            "the controls did not separate: "
            f"the incumbent-unchanged arm scored "
            f"{percent(controls['incumbent'].get('share_right', 0.0))} "
            f"against {percent(best_repair_share)} for the best repairing arm; "
            f"the withheld-text arm scored {whole_split['right_b']}/{whole_split['members']} "
            f"against {whole_split['right_a']}/{whole_split['members']} "
            f"({percent(whole_split['right_a'] / whole_split['members'])}), "
            f"losing {whole_split['only_a_right']} members and gaining {whole_split['only_b_right']}. "
            "No arm difference is called a win on this run."
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
    withheld_control = document["final"]["controls"]["withheld"]
    if withheld_control.get("unmeasurable"):
        withheld_html = "unmeasurable: " + withheld_control["unmeasurable"]
    else:
        withheld_html = (
            str(withheld_control["separated"]).lower()
            + " over "
            + str(withheld_control.get("dependent_members", 0))
            + " members"
        )
    lines.append(
        f"<p>Measuring: <strong>{str(document['final']['measuring']).lower()}</strong>. "
        f"Incumbent-unchanged separated: "
        f"{str(document['final']['controls']['incumbent']['separated']).lower()}; "
        f"withheld-text: {withheld_html}</p>"
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
    withheld = controls["withheld"]
    if withheld.get("unmeasurable"):
        withheld_sentence = "unmeasurable: " + withheld["unmeasurable"]
    else:
        count = withheld.get("dependent_members", 0)
        withheld_sentence = (
            f"separated: {str(withheld['separated']).lower()}"
            f" on {count} members whose right answer rests on the enclosing text"
        )
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
        f"{str(controls['incumbent']['separated']).lower()}; withheld-text "
        f"{withheld_sentence}. "
        f"Measuring: {str(final['measuring']).lower()}.</p>",
        f'<p id="scfg-benchmark-admission">{escape(document["decision"]["admission"])}</p>',
        '<h4 id="scfg-benchmark-choice">The choice</h4>',
        _decision_table(document["decision"]),
        f"<p>Spend USD {spend['total_usd']:.2f} against the USD "
        f"{spend['ceiling_usd']:.0f} ceiling.</p>",
        "</section>",
    ]
    return "\n".join(lines)


def render_figure(document: dict[str, Any]) -> None:
    """Draw the held-out arms with text stored as paths.

    Hand-written SVG text depends on the renderer's fonts; a rasterizer without the declared family can drop every label while it keeps the bars and whiskers. Matplotlib's SVG backend with ``svg.fonttype = "path"`` stores glyph outlines, so the title, arm names, values and ticks need no font lookup.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    reports = document["final"]["arms"]
    names = [report["arm"] for report in reports]
    shares = [report["share_right"] * 100 for report in reports]
    lowers = [report["interval"][0] * 100 for report in reports]
    uppers = [report["interval"][1] * 100 for report in reports]
    colors = [
        "#e07a5f" if ("withheld" in name or "incumbent" in name) else "#4b7bec"
        for name in names
    ]
    positions = list(range(len(reports)))
    with matplotlib.rc_context(
        {
            "svg.fonttype": "path",
            "font.family": "DejaVu Sans",
            "font.size": 10,
        }
    ):
        figure, axes = plt.subplots(figsize=(11, 0.62 * len(reports) + 1.3), dpi=100)
        figure.patch.set_facecolor("white")
        axes.set_facecolor("white")
        axes.barh(positions, shares, height=0.5, color=colors)
        for position, low, high in zip(positions, lowers, uppers, strict=True):
            axes.plot([low, high], [position, position], color="#222")
            axes.plot([low, low], [position - 0.12, position + 0.12], color="#222")
            axes.plot([high, high], [position - 0.12, position + 0.12], color="#222")
        for position, report, high in zip(positions, reports, uppers, strict=True):
            axes.text(
                high + 1.5,
                position,
                f"{percent(report['share_right'])} "
                f"[{percent(report['interval'][0])}, "
                f"{percent(report['interval'][1])}]",
                va="center",
                fontsize=9,
            )
        axes.set_yticks(positions, names)
        axes.invert_yaxis()
        axes.set_xlim(0, 118)
        axes.set_xticks([0, 20, 40, 60, 80, 100])
        axes.set_xlabel("share right (%)")
        axes.set_title(
            "Share right on the held-out half, with a 95 % interval", loc="left"
        )
        axes.grid(axis="x", color="#eeeeee", linewidth=0.8)
        axes.set_axisbelow(True)
        for side in ("top", "right"):
            axes.spines[side].set_visible(False)
        figure.tight_layout()
        FIGURE.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(
            FIGURE, format="svg", bbox_inches="tight", metadata={"Date": None}
        )
        figure.clear()


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


# --------------------------------------------------------------------------- #
# The second round: keep-or-repair arms, a perturbed control, a blind packet
# --------------------------------------------------------------------------- #

SECOND_ROUND_SCHEMA = "imas-codex.standard-name-record-benchmark/second-round"
SECOND_RECORDS = ROOT / "benchmark-second-round-records.jsonl"
SECOND_PACKET = ROOT / "benchmark-second-round-packet.jsonl"
SECOND_KEY = ROOT / "benchmark-second-round-key.json"
SECOND_DATA = ROOT / "benchmark-second-round.json"
SECOND_PAGE = ROOT / "benchmark-second-round.html"
SECOND_RUN = Path(
    os.environ.get(
        "BENCHMARK_SECOND_ROUND_RUN",
        "/home/ITER/mcintos/.config/reckon/crew/runs/"
        "r-20261120T074941732698-scfg-benchmark-second-round",
    )
)
SECOND_WORK = SECOND_RUN / "development-iteration.jsonl"
ARCHIVED_CASE_FILES = Path(
    "/home/ITER/mcintos/.config/reckon/crew/archive/runs/"
    "r-20261118T064806928346-scfg-case-file/case-files"
)
SECOND_SPEND_CEILING_USD = 30.0
SECOND_SEED = 20261120
ESTIMATED_CALL_USD = 0.06
SECOND_ARMS = (
    ("keep-or-repair/frontier", "keep-or-repair", FRONTIER_MODEL, "full"),
    ("keep-or-repair/local", "keep-or-repair", LOCAL_MODEL, "full"),
    ("one-pass/frontier", "one-pass", FRONTIER_MODEL, "full"),
    ("incumbent", "", "", "full"),
    ("perturbed/frontier", "keep-or-repair", FRONTIER_MODEL, "perturbed"),
)
SECOND_MODELS = {"frontier": FRONTIER_MODEL, "local": LOCAL_MODEL}
KIMI_SECOND_COMMAND = (
    'KIMI_BASE_URL="${KIMI_BASE_URL:?set KIMI_BASE_URL to the B300 server /v1 root}" '
    "uv run --no-sync python "
    "docs/evidence/fragments/sn-case-file-generation/benchmark.py "
    "--stage second-round"
)
PACKET_FIELDS = (
    "packet_id",
    "source_descriptions",
    "enclosing_statements",
    "name",
    "definition",
    "documentation",
    "unit",
)
# An arm label or the incumbent's identity in a packet value would break the
# blind read. These are the strings that could carry one; a physics word such
# as "local" is deliberately not in the list because definitions use it.
PACKET_LEAK_TOKENS = (
    "keep-or-repair",
    "one-pass",
    "perturbed",
    "incumbent",
    "frontier",
    "kimi",
)

KEEP_OR_REPAIR_SYSTEM = (
    """\
You are given one standard-name case file. It opens with the incumbent record:
the name, definition, documentation and unit the catalog already holds. Decide
whether to keep it, repair it, or report that the case file cannot decide.

Return decision "keep" when the case file neither contradicts the incumbent nor
under-identifies the quantity. A keep copies the incumbent's name, definition
and documentation unchanged, with its unit and its physics group.
Return decision "repair" when the case file contradicts the incumbent or states
a detail the incumbent omits. A repair returns the corrected record, and the
reason quotes the words of the case file that forced the repair.
Return decision "blocked" only when the case file cannot determine a record;
then blocker names one of grammar-gap, registry-conflict,
insufficient-source-text or collision, and the reason quotes the source words
that stop it. Otherwise leave blocker and reason empty.
"""
    + IDENTITY_RULE
)

KEEP_OR_REPAIR_CHANGE_ONE = """
Two failures the previous round counted are answered here.
First, every detail the case file states that affects the number must appear in
the definition: a time dependence, a coordinate frame, a normalisation, a
weighting, an averaging convention or a dimension the source states is not to
be summarised away. State each one.
Second, derive a name the case file determines. When the case file supplies the
locus, the vocabulary and a registry entry, compose the record; a collision, a
grammar gap or an insufficient-source-text blocker applies only when the term
the answer needs is genuinely absent from what the case file supplies.
"""

KEEP_OR_REPAIR_CHANGE_TWO = """
A repair changes only what the case file contradicts or under-identifies: keep
every other word of the incumbent's name, definition and documentation, so a
reader comparing the two records sees the repaired clause and nothing else. A
keep returns the incumbent's text verbatim; do not reword a record you judge
sound.
"""

SECOND_PROMPTS = {
    "v1": KEEP_OR_REPAIR_SYSTEM,
    "v2": KEEP_OR_REPAIR_SYSTEM + KEEP_OR_REPAIR_CHANGE_ONE,
    "v3": KEEP_OR_REPAIR_SYSTEM + KEEP_OR_REPAIR_CHANGE_ONE + KEEP_OR_REPAIR_CHANGE_TWO,
}
DEVELOPMENT_ARMS = ("keep-or-repair/frontier", "keep-or-repair/local")


class KeepOrRepairDraft(BaseModel):
    """One author's decision on an incumbent, with the words that forced it."""

    decision: str = Field(description="keep, repair or blocked.")
    name: str = Field(description="The record's name; the incumbent's on a keep.")
    description: str = Field(description="The definition.")
    documentation: str = Field(description="One paragraph of documentation.")
    unit: str = Field(description="The unit.")
    group: str = Field(default="", description="The physics group.")
    blocker: str = Field(
        default="",
        description=(
            "Empty unless decision is blocked: grammar-gap, registry-conflict, "
            "insufficient-source-text or collision."
        ),
    )
    reason: str = Field(
        default="",
        description="The words of the case file that forced a repair or a block.",
    )


def second_prompt(version: str) -> str:
    return SECOND_PROMPTS[version]


def one_pass_prompt() -> str:
    """Round one's best arm, unchanged: the v2 one-pass system prompt.

    The answered-failure sentence is rebuilt from the committed round-one
    document so the comparison arm reads exactly what it read then.
    """
    answered: dict[str, int] = {}
    if DATA.exists():
        try:
            document = json.loads(DATA.read_text(encoding="utf-8"))
            for round_row in document.get("rounds", []):
                if round_row.get("prompt_version") == "v2":
                    answered = dict(round_row.get("answered_failures") or {})
        except (json.JSONDecodeError, OSError):
            answered = {}
    return system_for("one-pass", "v2", answered)


# --------------------------------------------------------------------------- #
# Archived case files, the perturbed draw and the source-text block
# --------------------------------------------------------------------------- #


def load_second_cases(members: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Load each member's archived case file; rebuild from the graph if missing.

    The archive holds the case files the reference set and the failure read were
    built from, so the round reads no graph when it serves. A missing member is
    rebuilt through the same code that wrote the archive and its digest is
    recorded, so the reader can see which source each case file came from.
    """
    cases: dict[str, dict[str, Any]] = {}
    missing: list[str] = []
    for member in members:
        path = ARCHIVED_CASE_FILES / f"{member['id']}.json"
        if path.exists():
            cases[member["id"]] = json.loads(path.read_text(encoding="utf-8"))
        else:
            missing.append(member["id"])
    if missing:
        from imas_codex.graph.client import GraphClient
        from imas_codex.standard_names.case_file import build_name_case_file

        with GraphClient() as client:
            for member_id in missing:
                cases[member_id] = build_name_case_file(
                    member_id, gc=client
                ).model_dump()
    return cases


def second_case_text(case: dict[str, Any]) -> str:
    from imas_codex.standard_names.case_file import render_case_file

    return render_case_file(case)


def incumbent_record(content: dict[str, Any]) -> dict[str, str] | None:
    incumbent = content.get("incumbent")
    if not incumbent:
        return None
    return {
        "name": incumbent.get("name") or "",
        "description": incumbent.get("description") or "",
        "documentation": incumbent.get("documentation") or "",
        "unit": incumbent.get("unit") or "",
        "group": "",
    }


def record_changed_from_incumbent(
    record: dict[str, Any], content: dict[str, Any]
) -> bool | None:
    incumbent = incumbent_record(content)
    if incumbent is None:
        return None
    return any(
        squash(record.get(field)) != squash(incumbent.get(field))
        and bool(squash(record.get(field)))
        and bool(squash(incumbent.get(field)))
        for field in ("name", "description", "documentation", "unit")
    )


def perturb_content(
    case: dict[str, Any], donor: dict[str, Any]
) -> dict[str, Any] | None:
    """Replace the source descriptions and enclosing text with a donor's.

    Every channel except these two is left as it was, so the only thing that
    moves is the text a case file gives about the data dictionary -- which is
    what the control is for: if the answer does not follow this text, the
    control moves nothing and the round says so.
    """
    import copy

    donor_sources = [
        source
        for source in (donor.get("content", {}).get("sources") or [])
        if source.get("text")
    ]
    if not donor_sources:
        return None
    content = copy.deepcopy(case["content"])
    for index, source in enumerate(content.get("sources") or []):
        replacement = donor_sources[index % len(donor_sources)]
        source["text"] = replacement.get("text") or ""
        enclosing = source.get("enclosing")
        if isinstance(enclosing, dict):
            enclosing["statements"] = [
                dict(statement)
                for statement in (
                    (replacement.get("enclosing") or {}).get("statements") or []
                )
            ]
    return content


def donor_for(
    member: dict[str, Any],
    members: list[dict[str, Any]],
    cases: dict[str, dict[str, Any]],
) -> tuple[str, dict[str, Any]] | None:
    candidates = sorted(
        other["id"]
        for other in members
        if other["id"] != member["id"] and other.get("group") != member.get("group")
    )
    if not candidates:
        return None
    pick = random.Random(f"{SECOND_SEED}:{member['id']}  Perturbation draw")
    donor_id = pick.choice(candidates)
    return donor_id, cases[donor_id]


def source_blocks(case: dict[str, Any]) -> dict[str, list[str]]:
    """The data-dictionary description and enclosing-structure text, deduped."""
    descriptions: list[str] = []
    statements: list[str] = []
    seen_descriptions: set[str] = set()
    seen_statements: set[str] = set()
    for source in case.get("content", {}).get("sources") or []:
        text = (source.get("text") or "").strip()
        if text and text not in seen_descriptions:
            seen_descriptions.add(text)
            descriptions.append(text)
        for statement in (source.get("enclosing") or {}).get("statements") or []:
            line = (statement.get("text") or "").strip()
            if line and line not in seen_statements:
                seen_statements.add(line)
                statements.append(line)
    return {"source_descriptions": descriptions, "enclosing_statements": statements}


# --------------------------------------------------------------------------- #
# Generation and the arms
# --------------------------------------------------------------------------- #


class SpendGuard:
    """Stop before a call that would take the round past its ceiling."""

    def __init__(self, ceiling: float, spent: float = 0.0) -> None:
        self.ceiling = ceiling
        self.spent = spent

    def reserve(self) -> None:
        if self.spent + ESTIMATED_CALL_USD > self.ceiling:
            raise RuntimeError(
                f"the next call would take spend past USD {self.ceiling:.2f} "
                f"(spent USD {self.spent:.2f}); the round stops here"
            )

    def add(self, cost: float) -> None:
        self.spent += float(cost)


def normalise_decision(raw: str, blocker: str) -> str:
    decision = squash(raw)
    if decision in {"keep", "repair", "blocked"}:
        return decision
    return "blocked" if blocker else "repair"


def second_row(
    phase: str,
    arm: str,
    member: dict[str, Any],
    case: dict[str, Any],
    version: str,
    structure: str,
    model: str,
    body: dict[str, Any],
) -> dict[str, Any]:
    record = {
        "name": body.get("name") or "",
        "description": body.get("description") or "",
        "documentation": body.get("documentation") or "",
        "unit": body.get("unit") or "",
        "group": body.get("group") or "",
    }
    decision = body["decision"]
    return {
        "phase": phase,
        "arm": arm,
        "member": member["id"],
        "decision": decision,
        "record": record,
        "blocker": (body.get("blocker") or "") if decision == "blocked" else "",
        "reason": body.get("reason") or "",
        "structure": structure,
        "model": model,
        "prompt_version": version,
        "prompt_hash": body.get("prompt_hash") or "",
        "cost_usd": float(body.get("cost_usd") or 0.0),
        "input_tokens": int(body.get("input_tokens") or 0),
        "output_tokens": int(body.get("output_tokens") or 0),
        "seconds": float(body.get("seconds") or 0.0),
        "changed_incumbent": record_changed_from_incumbent(
            record, case.get("content", {})
        ),
        "incumbent_right": member.get("verdict") == "sound",
        "case_digest": case.get("digest") or "",
        "target_group": member.get("group") or "",
    }


async def generate_second_one(
    structure: str,
    model: str,
    case_text: str,
    version: str,
    semaphores: dict[str, asyncio.Semaphore],
    guard: SpendGuard,
) -> dict[str, Any]:
    lane = "local" if model.startswith("local/") else "frontier"
    if structure == "one-pass":
        system = one_pass_prompt()
        schema: type[BaseModel] = RecordDraft
    else:
        system = second_prompt(version)
        schema = KeepOrRepairDraft
    started = time.time()
    cost = 0.0
    input_tokens = 0
    output_tokens = 0
    try:
        async with semaphores[lane]:
            guard.reserve()
            parsed, meta = await call_model(model, system, case_text, schema)
        cost = float(meta["cost"])
        input_tokens = int(meta["input_tokens"])
        output_tokens = int(meta["output_tokens"])
        raw = parsed.model_dump()
        blocker = str(raw.get("blocker") or "").strip()
        if structure == "one-pass":
            decision = "blocked" if blocker else "repair"
            reason = ""
        else:
            decision = normalise_decision(str(raw.get("decision") or ""), blocker)
            reason = str(raw.get("reason") or "").strip()
        body = {
            "decision": decision,
            "name": raw.get("name") or "",
            "description": raw.get("description") or "",
            "documentation": raw.get("documentation") or "",
            "unit": raw.get("unit") or "",
            "group": str(raw.get("group") or ""),
            "blocker": blocker,
            "reason": reason,
            "prompt_hash": digest_text(system),
            "cost_usd": cost,
            "seconds": time.time() - started,
        }
    except RuntimeError:
        raise
    except Exception as exc:  # a failed call is a blocked member, never a right
        body = {
            "decision": "blocked",
            "blocker": "generation-failure:" + type(exc).__name__,
            "reason": str(exc)[:300],
            "prompt_hash": digest_text(system),
            "cost_usd": cost,
            "seconds": time.time() - started,
        }
    body["input_tokens"] = input_tokens
    body["output_tokens"] = output_tokens
    guard.add(cost)
    return body


def incumbent_body(case: dict[str, Any]) -> dict[str, Any]:
    record = incumbent_record(case.get("content", {}))
    if record is None:
        return {
            "decision": "blocked",
            "blocker": "no-incumbent",
            "reason": "the case file carries no incumbent record",
        }
    return {
        "decision": "keep",
        "name": record["name"],
        "description": record["description"],
        "documentation": record["documentation"],
        "unit": record["unit"],
        "group": "",
        "blocker": "",
        "reason": "",
        "prompt_hash": "",
    }


async def run_second_arm(
    phase: str,
    arm: str,
    structure: str,
    model: str,
    members: list[dict[str, Any]],
    cases: dict[str, dict[str, Any]],
    version: str,
    log_path: Path,
    guard: SpendGuard,
) -> None:
    seen = {
        (row.get("phase"), row.get("arm"), row.get("prompt_version"), row.get("member"))
        for row in read_jsonl(log_path)
    }
    semaphores = {
        "local": asyncio.Semaphore(CONCURRENCY["local"]),
        "frontier": asyncio.Semaphore(CONCURRENCY["frontier"]),
    }

    async def one(member: dict[str, Any]) -> None:
        key = (phase, arm, version, member["id"])
        if key in seen:
            return
        case = cases[member["id"]]
        if structure == "incumbent":
            body = incumbent_body(case)
        else:
            body = await generate_second_one(
                structure, model, second_case_text(case), version, semaphores, guard
            )
        append_jsonl(
            log_path,
            second_row(phase, arm, member, case, version, structure, model, body),
        )

    await asyncio.gather(*(one(member) for member in members))


# --------------------------------------------------------------------------- #
# The development iteration: a mechanical proxy, labelled as one
# --------------------------------------------------------------------------- #

PROXY_DEFINITION = (
    "A proxy, not a verdict: the share of a member's records whose name is one of "
    "the reference's right names and whose unit agrees. It is mechanical, it is "
    "read only on the development half, and it decides nothing about the "
    "held-out half."
)

DEVELOPMENT_CHANGES = {
    "v1": (
        "the keep-or-repair prompt as first written: decide keep, repair or "
        "blocked, and quote the case-file words that forced a repair"
    ),
    "v2": (
        "answers a stated detail being dropped (8 of 50 in the failure read) by "
        "requiring every stated detail to appear in the definition, and answers "
        "blocking on a derivable name (7 of 50) by naming when a blocker is allowed"
    ),
    "v3": (
        "answers a repair that rewrites a sound clause: a repair changes only "
        "what the case file contradicts, and a keep copies the incumbent verbatim"
    ),
}

DEVELOPMENT_ANSWERS = {
    "v1": {},
    "v2": {
        "detail-the-case-file-states-dropped": 8,
        "blocked-on-a-name-the-generator-can-derive": 7,
    },
    "v3": {
        "detail-the-case-file-states-dropped": 8,
        "blocked-on-a-name-the-generator-can-derive": 7,
    },
}


def proxy_figures(
    rows: list[dict[str, Any]], members: list[dict[str, Any]]
) -> dict[str, Any]:
    by_id = {member["id"]: member for member in members}
    right = 0
    total = 0
    blocked = 0
    name_hits = 0
    unit_hits = 0
    for row in rows:
        total += 1
        if row["decision"] == "blocked":
            blocked += 1
            continue
        member = by_id[row["member"]]
        right_names = {squash(name) for name in member.get("right_names") or []}
        hit_name = squash(row["record"].get("name")) in right_names
        hit_unit = not member.get("unit") or squash(
            row["record"].get("unit")
        ) == squash(member.get("unit"))
        name_hits += 1 if hit_name else 0
        unit_hits += 1 if hit_unit else 0
        right += 1 if (hit_name and hit_unit) else 0
    return {
        "members": total,
        "right": right,
        "share": (right / total) if total else 0.0,
        "interval": list(bench.wilson_interval(right, total)),
        "blocked": blocked,
        "name_in_reference": name_hits,
        "unit_agrees": unit_hits,
    }


def failure_counts_from_rows(
    rows: list[dict[str, Any]], members: list[dict[str, Any]]
) -> dict[str, int]:
    by_id = {member["id"]: member for member in members}
    counts: Counter[str] = Counter()
    for row in rows:
        if row["decision"] == "blocked":
            counts["blocked"] += 1
            continue
        member = by_id[row["member"]]
        right_names = {squash(name) for name in member.get("right_names") or []}
        if squash(row["record"].get("name")) not in right_names:
            counts["name-not-in-reference"] += 1
        if member.get("unit") and squash(row["record"].get("unit")) != squash(
            member.get("unit")
        ):
            counts["unit-mismatch"] += 1
    return dict(sorted(counts.items()))


def development_model(arm: str) -> str:
    return FRONTIER_MODEL if arm.endswith("/frontier") else LOCAL_MODEL


async def run_development(
    development: list[dict[str, Any]],
    cases: dict[str, dict[str, Any]],
    guard: SpendGuard,
) -> list[dict[str, Any]]:
    rounds: list[dict[str, Any]] = []
    for version in ("v1", "v2", "v3"):
        for arm in DEVELOPMENT_ARMS:
            await run_second_arm(
                "development",
                arm,
                "keep-or-repair",
                development_model(arm),
                development,
                cases,
                version,
                SECOND_WORK,
                guard,
            )
        arms: list[dict[str, Any]] = []
        failures: dict[str, int] = {}
        for arm in DEVELOPMENT_ARMS:
            rows = [
                row
                for row in read_jsonl(SECOND_WORK)
                if row["arm"] == arm and row["prompt_version"] == version
            ]
            arms.append({"arm": arm, **proxy_figures(rows, development)})
            failures.update(failure_counts_from_rows(rows, development))
        rounds.append(
            {
                "prompt_version": version,
                "prompt_hash": digest_text(second_prompt(version)),
                "change": DEVELOPMENT_CHANGES[version],
                "answers": DEVELOPMENT_ANSWERS[version],
                "failures_counted_on_this_half": failures,
                "arms": arms,
                "measure": PROXY_DEFINITION,
            }
        )
    return rounds


# --------------------------------------------------------------------------- #
# The blind packet, its key, and the records digest
# --------------------------------------------------------------------------- #


def second_record_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    record = row["record"]
    return (
        row["member"],
        (record.get("name") or "").strip(),
        (record.get("description") or "").strip(),
        (record.get("unit") or "").strip(),
    )


def packet_id_for(key: tuple[str, str, str, str]) -> str:
    raw = "|".join(key)
    return "pkt-" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def records_digest(rows: list[dict[str, Any]]) -> str:
    raw = "\n".join(
        json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        for row in rows
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def build_packet_and_key(
    records: list[dict[str, Any]],
    members: list[dict[str, Any]],
    cases: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    members_by_id = {member["id"]: member for member in members}
    by_pair = {(row["member"], row["arm"]): row for row in records}
    groups: dict[tuple[str, str, str, str], list[list[str]]] = {}
    for row in sorted(records, key=lambda item: (item["member"], item["arm"])):
        if row["decision"] == "blocked":
            continue
        if not (row["record"].get("name") or "").strip():
            continue
        key = second_record_key(row)
        groups.setdefault(key, []).append([row["member"], row["arm"]])
    lines: list[dict[str, Any]] = []
    packets: dict[str, Any] = {}
    for key, pairs in groups.items():
        member_id, name, definition, unit = key
        blocks = source_blocks(cases[member_id])
        packet_id = packet_id_for(key)
        first = by_pair[(pairs[0][0], pairs[0][1])]
        lines.append(
            {
                "packet_id": packet_id,
                "source_descriptions": blocks["source_descriptions"],
                "enclosing_statements": blocks["enclosing_statements"],
                "name": name,
                "definition": definition,
                "documentation": first["record"]["documentation"],
                "unit": unit,
            }
        )
        packets[packet_id] = {"member": member_id, "arms": pairs}
    random.Random(SECOND_SEED).shuffle(lines)
    key_document: dict[str, Any] = {
        "rule": (
            "The adjudicator does not open this file until its verdicts are "
            "committed. It maps a packet id back to the member and arms it "
            "stands for, and carries the reference's right names and definition "
            "points."
        ),
        "seed": SECOND_SEED,
        "records_digest": records_digest(records),
        "packets": packets,
        "members": {
            member_id: {
                "right_names": members_by_id[member_id].get("right_names") or [],
                "definition_points": members_by_id[member_id].get("definition_points")
                or [],
                "unit": members_by_id[member_id].get("unit") or "",
                "verdict": members_by_id[member_id].get("verdict") or "",
            }
            for member_id in sorted({packet["member"] for packet in packets.values()})
        },
    }
    return (
        lines,
        key_document,
    )  # --------------------------------------------------------------------------- #


# Scoring a committed verdicts file
# --------------------------------------------------------------------------- #

VERDICT_VALUES = ("right", "wrong", "undecidable")


def load_verdicts(path: Path) -> dict[str, dict[str, str]]:
    """Read the adjudicator's file: one line per packet id, name and definition."""
    verdicts: dict[str, dict[str, str]] = {}
    for row in read_jsonl(path):
        packet_id = str(row.get("packet_id") or "").strip()
        if not packet_id:
            continue
        verdicts[packet_id] = {
            "name": squash(row.get("name_verdict") or row.get("name") or ""),
            "definition": squash(
                row.get("definition_verdict") or row.get("definition") or ""
            ),
        }
    return verdicts


def score_rows(
    records: list[dict[str, Any]],
    packet: list[dict[str, Any]],
    key: dict[str, Any],
    verdicts: dict[str, dict[str, str]],
) -> dict[str, Any]:
    """Turn a blind verdicts file into the per-arm figures the round reports."""
    pair_to_packet: dict[tuple[str, str], str] = {}
    for packet_id, entry in key["packets"].items():
        for member, arm in entry["arms"]:
            pair_to_packet[(member, arm)] = packet_id
    arms = sorted({row["arm"] for row in records})
    compare_rows: list[bench.RecordBenchmarkRow] = []
    reports: dict[str, Any] = {}
    for arm in arms:
        arm_records = [row for row in records if row["arm"] == arm]
        total = len(arm_records)
        name_right = 0
        definition_right = 0
        both_right = 0
        blocked = 0
        blocked_types: Counter[str] = Counter()
        changed_right = 0
        right_incumbent_total = 0
        cost = 0.0
        seconds = 0.0
        for row in arm_records:
            cost += float(row.get("cost_usd") or 0.0)
            seconds += float(row.get("seconds") or 0.0)
            incumbent_right = bool(row.get("incumbent_right"))
            if incumbent_right:
                right_incumbent_total += 1
                if row["decision"] != "keep" or row.get("changed_incumbent"):
                    changed_right += 1
            if row["decision"] == "blocked":
                blocked += 1
                blocked_types[bench.blocker_bucket(row.get("blocker") or "other")] += 1
                compare_rows.append(
                    bench.RecordBenchmarkRow(
                        member=row["member"],
                        arm=arm,
                        verdict="blocked",
                        blocker=row.get("blocker") or "other",
                        changed_incumbent=row.get("changed_incumbent"),
                        incumbent_right=row.get("incumbent_right"),
                        cost_usd=float(row.get("cost_usd") or 0.0),
                        seconds=float(row.get("seconds") or 0.0),
                        input_tokens=int(row.get("input_tokens") or 0),
                        output_tokens=int(row.get("output_tokens") or 0),
                    )
                )
                continue
            packet_id = pair_to_packet.get((row["member"], arm))
            verdict = verdicts.get(packet_id) if packet_id else None
            name_ok = bool(verdict and verdict.get("name") == "right")
            definition_ok = bool(verdict and verdict.get("definition") == "right")
            both_ok = name_ok and definition_ok
            name_right += 1 if name_ok else 0
            definition_right += 1 if definition_ok else 0
            both_right += 1 if both_ok else 0
            compare_rows.append(
                bench.RecordBenchmarkRow(
                    member=row["member"],
                    arm=arm,
                    verdict="right" if both_ok else "wrong",
                    changed_incumbent=row.get("changed_incumbent"),
                    incumbent_right=row.get("incumbent_right"),
                    cost_usd=float(row.get("cost_usd") or 0.0),
                    seconds=float(row.get("seconds") or 0.0),
                    input_tokens=int(row.get("input_tokens") or 0),
                    output_tokens=int(row.get("output_tokens") or 0),
                )
            )
        reports[arm] = {
            "members": total,
            "name_right": name_right,
            "name_share": (name_right / total) if total else 0.0,
            "name_interval": list(bench.wilson_interval(name_right, total)),
            "definition_right": definition_right,
            "definition_share": (definition_right / total) if total else 0.0,
            "definition_interval": list(bench.wilson_interval(definition_right, total)),
            "both_right": both_right,
            "both_share": (both_right / total) if total else 0.0,
            "both_interval": list(bench.wilson_interval(both_right, total)),
            "blocked": blocked,
            "blocked_by_type": dict(sorted(blocked_types.items())),
            "changed_right_incumbent": changed_right,
            "right_incumbent_total": right_incumbent_total,
            "changed_right_incumbent_share": (
                changed_right / right_incumbent_total if right_incumbent_total else 0.0
            ),
            "cost_usd": cost,
            "seconds": seconds,
            "cost_per_name": (cost / total) if total else 0.0,
            "seconds_per_name": (seconds / total) if total else 0.0,
        }
    comparisons: list[dict[str, Any]] = []
    for index, first in enumerate(arms):
        for second in arms[index + 1 :]:
            try:
                comparisons.append(
                    bench.compare_record_arms(
                        compare_rows, first, second, seed=SECOND_SEED
                    )
                )
            except ValueError:
                continue
    undecided = sum(
        1
        for packet_id in key["packets"]
        if "name" not in verdicts.get(packet_id, {})
        or "definition" not in verdicts.get(packet_id, {})
    )
    return {
        "schema": SECOND_ROUND_SCHEMA + "/scores",
        "packets": len(packet),
        "verdicts": len(verdicts),
        "undecided_packet_lines": undecided,
        "rule": "A name is right when it denotes the same quantity; a definition "
        "is right when it states every reference definition point. Both right "
        "means both. Group is not scored.",
        "arms": reports,
        "comparisons": comparisons,
    }


def score_files(
    verdicts_path: Path,
    records_path: Path = SECOND_RECORDS,
    packet_path: Path = SECOND_PACKET,
    key_path: Path = SECOND_KEY,
) -> dict[str, Any]:
    return score_rows(
        read_jsonl(records_path),
        read_jsonl(packet_path),
        json.loads(key_path.read_text(encoding="utf-8")),
        load_verdicts(verdicts_path),
    )


# --------------------------------------------------------------------------- #
# The check: a failure for an altered record, a leaking packet, a broken key
# --------------------------------------------------------------------------- #


def score_self_test() -> list[str]:
    """Prove the score subcommand on a synthetic verdicts file, never on real ones."""
    problems: list[str] = []
    cases = {
        member: {"content": {"sources": []}, "digest": "digest-" + member}
        for member in ("m1", "m2")
    }
    members = [
        {
            "id": "m1",
            "group": "g",
            "unit": "W",
            "right_names": ["n1"],
            "definition_points": ["p"],
            "verdict": "sound",
        },
        {
            "id": "m2",
            "group": "g",
            "unit": "W",
            "right_names": ["n2"],
            "definition_points": ["p"],
            "verdict": "definition_defect",
        },
    ]

    def row(
        member: str, arm: str, name: str, decision: str = "repair", blocker: str = ""
    ):
        return {
            "phase": "held_out",
            "arm": arm,
            "member": member,
            "decision": decision,
            "record": {
                "name": name,
                "description": "definition",
                "documentation": "documentation",
                "unit": "W",
                "group": "g",
            },
            "blocker": blocker,
            "reason": "",
            "structure": "keep-or-repair",
            "model": "synthetic",
            "prompt_version": "v3",
            "prompt_hash": "hash",
            "cost_usd": 0.5,
            "input_tokens": 100,
            "output_tokens": 10,
            "seconds": 2.0,
            "changed_incumbent": True,
            "incumbent_right": True,
            "case_digest": "digest-" + member,
            "target_group": "g",
        }

    records = [
        row("m1", "A", "n1"),
        row("m2", "A", "n2"),
        row("m1", "B", "", decision="blocked", blocker="collision"),
        row("m2", "B", "n3"),
    ]
    packet, key = build_packet_and_key(records, members, cases)
    lookup: dict[tuple[str, str], str] = {}
    for packet_id, entry in key["packets"].items():
        for member, arm in entry["arms"]:
            lookup[(member, arm)] = packet_id
    verdicts = {
        lookup[("m1", "A")]: {"name": "right", "definition": "right"},
        lookup[("m2", "A")]: {"name": "right", "definition": "wrong"},
        lookup[("m2", "B")]: {"name": "wrong", "definition": "wrong"},
    }
    scored = score_rows(records, packet, key, verdicts)
    figures = scored["arms"]
    if figures["A"]["name_right"] != 2:
        problems.append("score self-test: A name_right is not 2")
    if figures["A"]["definition_right"] != 1:
        problems.append("score self-test: A definition_right is not 1")
    if figures["A"]["both_right"] != 1:
        problems.append("score self-test: A both_right is not 1")
    if abs(figures["A"]["both_share"] - 0.5) > 1e-9:
        problems.append("score self-test: A both_share is not 0.5")
    if figures["B"]["blocked"] != 1:
        problems.append("score self-test: B blocked is not 1")
    if figures["B"]["blocked_by_type"].get("collision") != 1:
        problems.append("score self-test: B blocker is not counted as collision")
    if figures["B"]["both_right"] != 0:
        problems.append("score self-test: B both_right is not 0")
    if not scored["comparisons"]:
        problems.append("score self-test: no arm comparison was produced")
    low, high = figures["A"]["both_interval"]
    if not (0.0 <= low <= 0.5 <= high <= 1.0):
        problems.append("score self-test: the interval does not bracket the share")
    return problems


def check_second_round(
    records_path: Path = SECOND_RECORDS,
    packet_path: Path = SECOND_PACKET,
    key_path: Path = SECOND_KEY,
    data_path: Path = SECOND_DATA,
) -> list[str]:
    """Return every problem in the committed round, empty when the round holds."""
    problems: list[str] = []
    for path in (records_path, packet_path, key_path, data_path):
        if not path.exists():
            problems.append(f"missing {path.name}")
    if problems:
        return problems
    records = read_jsonl(records_path)
    packet = read_jsonl(packet_path)
    key = json.loads(key_path.read_text(encoding="utf-8"))
    document = json.loads(data_path.read_text(encoding="utf-8"))

    if not records:
        problems.append("the records file is empty")
    else:
        digest = records_digest(records)
        if key.get("records_digest") != digest:
            problems.append(
                "the records file does not match the key's records digest: a line "
                "was altered"
            )
        if document.get("records_digest") != digest:
            problems.append("the round document does not carry the records digest")

    for line in packet:
        fields = set(line)
        if fields != set(PACKET_FIELDS):
            extra = sorted(fields - set(PACKET_FIELDS))
            missing = sorted(set(PACKET_FIELDS) - fields)
            problems.append(
                f"packet line {line.get('packet_id', '?')} fields differ "
                f"(extra={extra}, missing={missing})"
            )
        for field, value in line.items():
            if not isinstance(value, str):
                continue
            lowered = value.lower()
            for token in PACKET_LEAK_TOKENS:
                if token in lowered:
                    problems.append(
                        f"packet line {line.get('packet_id', '?')} reveals "
                        f"{token!r} in {field!r}"
                    )

    packet_ids = [line.get("packet_id") for line in packet]
    if len(set(packet_ids)) != len(packet_ids):
        problems.append("a packet id appears more than once")
    key_packets = key.get("packets") or {}
    if set(packet_ids) != set(key_packets):
        only_packet = sorted(set(packet_ids) - set(key_packets))
        only_key = sorted(set(key_packets) - set(packet_ids))
        problems.append(
            "the key and the packet disagree on packet ids "
            f"(packet only={only_packet}, key only={only_key})"
        )

    arm_names = {name for name, _, _, _ in SECOND_ARMS}
    pair_records = {(row["member"], row["arm"]): row for row in records}
    packet_by_id = {line["packet_id"]: line for line in packet}
    for packet_id, entry in key_packets.items():
        line = packet_by_id.get(packet_id)
        if line is None:
            continue
        for member, arm in entry["arms"]:
            if arm not in arm_names:
                problems.append(f"{packet_id}: unknown arm {arm!r}")
            row = pair_records.get((member, arm))
            if row is None:
                problems.append(f"{packet_id}: names a (member, arm) with no record")
                continue
            expected = second_record_key(row)
            actual = (
                packet_id.split()[0],
                line["name"],
                line["definition"],
                line["unit"],
            )
            if expected[1:] != actual[1:]:
                problems.append(
                    f"{packet_id}: the packet record differs from the record line "
                    f"for {member} / {arm}"
                )

    covered: set[tuple[str, str]] = set()
    for entry in key_packets.values():
        for member, arm in entry["arms"]:
            covered.add((member, arm))
    for row in records:
        if row["decision"] == "blocked":
            continue
        if (row["member"], row["arm"]) not in covered:
            problems.append(
                f"{row['member']} / {row['arm']}: a non-blocked record has no "
                "packet line"
            )

    members_in_key = key.get("members") or {}
    for member in {pair[0] for pair in covered}:
        entry = members_in_key.get(member)
        if not entry:
            problems.append(f"key members: {member} missing")
            continue
        if not entry.get("right_names"):
            problems.append(f"key members: {member} carries no right name")
        if not entry.get("definition_points"):
            problems.append(f"key members: {member} carries no definition point")

    if document.get("packet", {}).get("lines") != len(packet):
        problems.append("the round document's packet line count is wrong")
    if document.get("records", {}).get("lines") != len(records):
        problems.append("the round document's records line count is wrong")
    ceiling = float(document.get("spend", {}).get("ceiling_usd") or 0.0)
    if ceiling != SECOND_SPEND_CEILING_USD:
        problems.append("the round document's ceiling is not the authorized one")
    if document.get("spend", {}).get("within_ceiling") is not True:
        problems.append("the round's spend is not within its ceiling")

    problems.extend(score_self_test())
    return problems


def run_check() -> int:
    problems = check_second_round()
    if problems:
        for problem in problems:
            print(f"FAIL: {problem}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "check": "passed",
                "records": SECOND_RECORDS.name,
                "packet": SECOND_PACKET.name,
                "key": SECOND_KEY.name,
                "note": (
                    "passes on the committed files; fails on an altered records "
                    "line, a packet line that reveals an arm, and a key whose "
                    "packet id is absent from the packet"
                ),
            },
            indent=2,
        )
    )
    return 0  # --------------------------------------------------------------------------- #


# The perturbed control
# --------------------------------------------------------------------------- #


def perturbed_cases(
    members: list[dict[str, Any]], cases: dict[str, dict[str, Any]]
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    """Return each member's perturbed case and the donor it was drawn from."""
    perturbed: dict[str, dict[str, Any]] = {}
    donors: dict[str, str] = {}
    for member in members:
        choice = donor_for(member, members, cases)
        if choice is None:
            continue
        donor_id, donor = choice
        content = perturb_content(cases[member["id"]], donor)
        if content is None:
            continue
        perturbed[member["id"]] = {
            "content": content,
            "digest": digest_json(content),
        }
        donors[member["id"]] = donor_id
    return perturbed, donors


def perturbed_change(
    records: list[dict[str, Any]],
    donors: dict[str, str],
) -> dict[str, Any]:
    """The share of members whose perturbed answer moved from the plain one."""
    plain: dict[str, dict[str, Any]] = {}
    moved: dict[str, dict[str, Any]] = {}
    for row in records:
        if row.get("phase") != "held_out":
            continue
        if row["arm"] == "keep-or-repair/frontier":
            plain[row["member"]] = row
        elif row["arm"] == "perturbed/frontier":
            moved[row["member"]] = row
    compared = 0
    changed = 0
    changed_decision = 0
    by_member: dict[str, bool] = {}
    for member_id, plain_row in sorted(plain.items()):
        moved_row = moved.get(member_id)
        if moved_row is None:
            continue
        compared += 1
        plain_record = plain_row.get("record") or {}
        moved_record = moved_row.get("record") or {}
        differs = (
            plain_row.get("decision") != moved_row.get("decision")
            or squash(plain_record.get("name")) != squash(moved_record.get("name"))
            or squash(plain_record.get("description"))
            != squash(moved_record.get("description"))
            or squash(plain_record.get("unit")) != squash(moved_record.get("unit"))
        )
        changed += 1 if differs else 0
        if plain_row.get("decision") != moved_row.get("decision"):
            changed_decision += 1
        by_member[member_id] = differs
    share = (changed / compared) if compared else 0.0
    if compared == 0:
        reading = (
            "No member carries both a plain and a perturbed answer, so the "
            "control measured nothing."
        )
    elif share >= 0.5:
        reading = (
            "At least half of the answers follow the source text the control replaced."
        )
    else:
        reading = (
            f"Fewer than half of the answers ({changed} of {compared}) follow "
            "the source text the control replaced: over half are unchanged by "
            "it, and the round says so plainly rather than reading the control "
            "as a pass."
        )
    return {
        "compared_members": compared,
        "changed_members": changed,
        "changed_share": share,
        "decision_changed_members": changed_decision,
        "donors": donors,
        "by_member": dict(sorted(by_member.items())),
        "reading": reading,
    }


# --------------------------------------------------------------------------- #
# The round document and its page
# --------------------------------------------------------------------------- #


def arm_decision_counts(records: list[dict[str, Any]]) -> dict[str, Any]:
    counts: dict[str, Any] = {}
    for row in records:
        arm = row["arm"]
        entry = counts.setdefault(
            arm,
            {"members": 0, "keep": 0, "repair": 0, "blocked": 0, "blocked_by_type": {}},
        )
        entry["members"] += 1
        decision = row["decision"]
        if decision in {"keep", "repair", "blocked"}:
            entry[decision] += 1
        if decision == "blocked":
            bucket = bench.blocker_bucket(row.get("blocker") or "other")
            entry["blocked_by_type"][bucket] = (
                entry["blocked_by_type"].get(bucket, 0) + 1
            )
    for entry in counts.values():
        entry["blocked_by_type"] = dict(sorted(entry["blocked_by_type"].items()))
    return dict(sorted(counts.items()))


def spend_report(records: list[dict[str, Any]]) -> dict[str, Any]:
    development = read_jsonl(SECOND_WORK) if SECOND_WORK.exists() else []
    spent = sum(float(row.get("cost_usd") or 0.0) for row in development)
    spent += sum(float(row.get("cost_usd") or 0.0) for row in records)
    return {
        "spent_usd": round(spent, 4),
        "ceiling_usd": SECOND_SPEND_CEILING_USD,
        "within_ceiling": spent <= SECOND_SPEND_CEILING_USD,
        "development_calls": len(development),
        "held_out_generation_calls": sum(
            1 for row in records if row["arm"] != "incumbent"
        ),
    }


def save_development(rounds: list[dict[str, Any]]) -> Path:
    path = SECOND_RUN / "development-iteration.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rounds, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def load_development() -> list[dict[str, Any]]:
    path = SECOND_RUN / "development-iteration.json"
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def build_second_document(
    records: list[dict[str, Any]],
    packet: list[dict[str, Any]],
    development: list[dict[str, Any]],
    donors: dict[str, str],
    members: list[dict[str, Any]],
) -> dict[str, Any]:
    development_ids: list[str] = []
    if SECOND_WORK.exists():
        development_ids = sorted({row["member"] for row in read_jsonl(SECOND_WORK)})
    return {
        "schema": SECOND_ROUND_SCHEMA,
        "generated": datetime.now(UTC).isoformat(timespec="seconds"),
        "seat": "Standard-name record generation, second round",
        "models": {
            "frontier": FRONTIER_MODEL,
            "local": LOCAL_MODEL,
            "kimi": {"run": False, "command": KIMI_SECOND_COMMAND},
        },
        "split": {
            "seed": SPLIT_SEED,
            "second_seed": SECOND_SEED,
            "members": len(members),
            "development": len(development_ids),
            "held_out": len({row["member"] for row in records}),
        },
        "records": {
            "file": SECOND_RECORDS.name,
            "lines": len(records),
            "digest": records_digest(records),
        },
        "records_digest": records_digest(records),
        "packet": {
            "file": SECOND_PACKET.name,
            "lines": len(packet),
            "fields": list(PACKET_FIELDS),
            "key": SECOND_KEY.name,
            "shuffled_by_seed": SECOND_SEED,
        },
        "arms": [
            {"arm": name, "structure": structure, "model": model, "case": case}
            for name, structure, model, case in SECOND_ARMS
        ],
        "prompts": {
            "versions": {
                version: {"hash": digest_text(prompt), "text": prompt}
                for version, prompt in SECOND_PROMPTS.items()
            },
            "one_pass": {
                "hash": digest_text(one_pass_prompt()),
                "note": "round one's v2 one-pass prompt, unchanged",
            },
            "final_version": "v3",
        },
        "development": {
            "half": "development",
            "measure": PROXY_DEFINITION,
            "rounds": development,
        },
        "decisions": arm_decision_counts(records),
        "perturbed": perturbed_change(records, donors),
        "spend": spend_report(records),
        "adjudication": {
            "state": "pending",
            "statement": (
                "The held-out names and definitions are in the blind packet and "
                "have not been scored. No held-out score appears here; the round "
                "reports what was generated and waits for the adjudicator's "
                "verdicts."
            ),
            "adjudicator": "the coordinating session, adjudicating blind",
        },
        "not_run": [
            {
                "arm": "keep-or-repair/kimi",
                "reason": "the Kimi endpoint did not answer",
                "command": KIMI_SECOND_COMMAND,
            }
        ],
    }


def render_second_svg(development: list[dict[str, Any]]) -> str:
    """Proxy share by prompt version, one bar per arm, drawn by hand."""
    width = 1400
    height = 520
    left = 90
    right = 40
    top = 70
    bottom = 80
    plot_w = width - left - right
    plot_h = height - top - bottom
    colours = {
        "keep-or-repair/frontier": "#2b6cb0",
        "keep-or-repair/local": "#2e8b57",
    }
    versions = [row["prompt_version"] for row in development]
    if not versions:
        return '<svg viewBox="0 0 1400 120" width="100%"></svg>'
    group_w = plot_w / max(len(versions), 1)
    bar_w = group_w * 0.22
    parts: list[str] = [
        f'<svg viewBox="0 0 {width} {height}" width="100%" '
        'font-family="system-ui, sans-serif" role="img" '
        'aria-label="proxy share by prompt version and arm">'
    ]
    for tick in (0, 25, 50, 75, 100):
        y = top + plot_h * (1 - tick / 100.0)
        parts.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" '
            'stroke="#d9d9d9" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{left - 12}" y="{y + 5:.1f}" font-size="14" '
            f'text-anchor="end" fill="#444">{tick}</text>'
        )
    parts.append(
        f'<text x="{left - 12}" y="{top - 28}" font-size="16" '
        f'text-anchor="end" fill="#1a1a1a">proxy share [%]</text>'
    )
    legend_x = left
    for arm, colour in colours.items():
        parts.append(
            f'<rect x="{legend_x}" y="26" width="22" height="6" fill="{colour}"/>'
        )
        parts.append(
            f'<text x="{legend_x + 30}" y="35" font-size="14" fill="#1a1a1a">'
            f"{arm}</text>"
        )
        legend_x += 320
    for group_index, round_row in enumerate(development):
        centre = left + group_w * (group_index + 0.5)
        parts.append(
            f'<text x="{centre:.1f}" y="{top + plot_h + 34}" font-size="16" '
            f'text-anchor="middle" fill="#1a1a1a">{round_row["prompt_version"]}</text>'
        )
        for arm_index, arm in enumerate(round_row["arms"]):
            share = 100.0 * float(arm["share"])
            bar_h = plot_h * share / 100.0
            x = centre - bar_w + 6 + arm_index * (bar_w + 12)
            y = top + plot_h - bar_h
            colour = colours.get(arm["arm"], "#929292")
            parts.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" '
                f'height="{bar_h:.1f}" fill="{colour}"/>'
            )
            parts.append(
                f'<text x="{x + bar_w / 2:.1f}" y="{y - 8:.1f}" font-size="14" '
                f'text-anchor="middle" fill="#1a1a1a">{share:.1f}</text>'
            )
    parts.append(
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" '
        f'y2="{top + plot_h}" stroke="#1a1a1a" stroke-width="1.2"/>'
    )
    parts.append("</svg>")
    return "".join(parts)


def render_second_html(document: dict[str, Any]) -> str:
    development = document["development"]["rounds"]
    tables: list[str] = []

    arm_rows: list[str] = []
    for arm, entry in document["decisions"].items():
        arm_rows.append(
            "<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>".format(
                arm, entry["members"], entry["keep"], entry["repair"], entry["blocked"]
            )
        )
    tables.append(
        "<table><tr><th>Arm</th><th>Members</th><th>Keep</th><th>Repair</th>"
        "<th>Blocked</th></tr>" + "".join(arm_rows) + "</table>"
    )

    development_rows: list[str] = []
    for round_row in development:
        for arm in round_row["arms"]:
            development_rows.append(
                "<tr><td>{}</td><td>{}</td><td>{} / {}</td>"
                "<td>{:.1f}% [{:.1f}, {:.1f}]</td><td>{}</td></tr>".format(
                    round_row["prompt_version"],
                    arm["arm"],
                    arm["right"],
                    arm["members"],
                    100.0 * arm["share"],
                    100.0 * arm["interval"][0],
                    100.0 * arm["interval"][1],
                    arm["blocked"],
                )
            )

    change_rows: list[str] = []
    for round_row in development:
        answers = round_row["answers"]
        change_rows.append(
            "<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>".format(
                round_row["prompt_version"],
                answers.get("detail-the-case-file-states-dropped", 0),
                answers.get("blocked-on-a-name-the-generator-can-derive", 0),
                round_row["change"],
            )
        )

    svg = render_second_svg(development)
    perturbed = document["perturbed"]
    spend = document["spend"]
    models = document["models"]
    arm_table = tables[0]

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Standard-name record generation, second round</title>
<style>
body {{ font-family: system-ui, sans-serif; max-width: 62rem; margin: 2rem auto;
  padding: 0 1rem; line-height: 1.5; color: #1a1a1a; }}
table {{ border-collapse: collapse; margin: 1rem 0; font-size: 0.9rem;
  max-width: 100%; }}
th, td {{ padding: 0.25rem 0.75rem; text-align: left;
  border-bottom: 1px solid #e2e2e2; vertical-align: top; }}
code {{ background: #f4f4f4; padding: 0.1rem 0.3rem; }}
.pending {{ padding: 0.75rem 1rem; border-left: 3px solid #2b6cb0;
  background: #f0f6fc; margin: 1rem 0; }}
figure {{ margin: 1.5rem 0; }}
figcaption {{ font-size: 0.85rem; color: #555; }}
</style>
</head>
<body>
<h1>Standard-name record generation, second round</h1>
<p>{document["adjudication"]["statement"]}</p>
<div class="pending"><strong>Held-out scores are pending adjudication.</strong>
The packet holds {document["packet"]["lines"]} distinct records. The key is not
opened until the adjudicator's verdicts are committed.</div>

<h2>Arms and models</h2>
<table>
<tr><th>Arm</th><th>Structure</th><th>Model</th><th>Case file</th></tr>
{"".join(f"<tr><td>{a['arm']}</td><td>{a['structure']}</td><td><code>{a['model']}</code></td><td>{a['case']}</td></tr>" for a in document["arms"])}
</table>
<p>Frontier model: <code>{models["frontier"]}</code>. Local model:
<code>{models["local"]}</code>.</p>
<p>Kimi: not run. The command that will run it is
<code>{models["kimi"]["command"]}</code>.</p>

<h2>Development iteration (proxy, development half only)</h2>
<p>{document["development"]["measure"]}</p>
<figure>{svg}<figcaption>Mechanical proxy by prompt version and arm on the
development half. It is not a verdict and it says nothing about the held-out
half.</figcaption></figure>
<table>
<tr><th>Version</th><th>Arm</th><th>Proxy right</th><th>Share [95%]</th>
<th>Blocked</th></tr>
{"".join(development_rows)}
</table>
<table>
<tr><th>Version</th><th>Stated detail dropped</th>
<th>Blocked on a derivable name</th><th>Change</th></tr>
{"".join(change_rows)}
</table>

<h2>Generated decisions, held-out half</h2>
{arm_table}
<p>Blocked types by arm: {json.dumps({a: e["blocked_by_type"] for a, e in document["decisions"].items()})}</p>

<h2>Perturbed control</h2>
<p>{perturbed["changed_members"]} of {perturbed["compared_members"]} answers
moved when the source description and enclosing-structure text were replaced
({100.0 * perturbed["changed_share"]:.1f}%). {perturbed["reading"]}</p>

<h2>Spend</h2>
<p>USD {spend["spent_usd"]:.2f} spent beside a ceiling of
USD {spend["ceiling_usd"]:.2f}. Within the ceiling:
<strong>{spend["within_ceiling"]}</strong>.</p>
<p>Records file <code>{document["records"]["file"]}</code>
({document["records"]["lines"]} lines). Packet
<code>{document["packet"]["file"]}</code> ({document["packet"]["lines"]} lines),
fields: {", ".join(document["packet"]["fields"])}.</p>
</body>
</html>
"""  # --------------------------------------------------------------------------- #


# The held-out run, the report, and the command line
# --------------------------------------------------------------------------- #


def existing_spend() -> float:
    spent = 0.0
    for path in (SECOND_WORK, SECOND_RECORDS):
        if path.exists():
            spent += sum(float(row.get("cost_usd") or 0.0) for row in read_jsonl(path))
    return spent


async def run_held_out(
    held_out: list[dict[str, Any]],
    cases: dict[str, dict[str, Any]],
    guard: SpendGuard,
) -> None:
    perturbed, _donors = perturbed_cases(held_out, cases)
    for name, structure, model, case_kind in SECOND_ARMS:
        arm_cases = perturbed if case_kind == "perturbed" else cases
        await run_second_arm(
            "held_out",
            name,
            structure,
            model,
            held_out,
            arm_cases,
            "v3",
            SECOND_RECORDS,
            guard,
        )


def write_second_round(
    members: list[dict[str, Any]],
    held_out: list[dict[str, Any]],
    cases: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    records = read_jsonl(SECOND_RECORDS)
    packet, key = build_packet_and_key(records, members, cases)
    SECOND_PACKET.parent.mkdir(parents=True, exist_ok=True)
    SECOND_PACKET.write_text("", encoding="utf-8")
    for line in packet:
        append_jsonl(SECOND_PACKET, line)
    SECOND_KEY.write_text(
        json.dumps(key, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _perturbed, donors = perturbed_cases(held_out, cases)
    document = build_second_document(
        records, packet, load_development(), donors, members
    )
    SECOND_DATA.write_text(
        json.dumps(document, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    SECOND_PAGE.write_text(render_second_html(document), encoding="utf-8")
    return document


def run_score(verdicts_path: Path, out_path: Path | None = None) -> int:
    scores = score_files(verdicts_path)
    text = json.dumps(scores, indent=2, ensure_ascii=False)
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text + "\n", encoding="utf-8")
        print(f"wrote {out_path}")
    else:
        print(text)
    return 0


def second_stage_summary(document: dict[str, Any]) -> str:
    return json.dumps(
        {
            "packet_lines": document["packet"]["lines"],
            "records_lines": document["records"]["lines"],
            "decisions": {
                arm: {
                    "keep": entry["keep"],
                    "repair": entry["repair"],
                    "blocked": entry["blocked"],
                }
                for arm, entry in document["decisions"].items()
            },
            "perturbed_changed_share": round(document["perturbed"]["changed_share"], 4),
            "spend": document["spend"],
            "adjudication": document["adjudication"]["state"],
        },
        indent=2,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=(
            "build",
            "measure",
            "report",
            "all",
            "second-development",
            "second-heldout",
            "second-report",
            "second-round",
        ),
        default="all",
        help="which part of the run to execute; every stage resumes its logs",
    )
    parser.add_argument(
        "--models",
        choices=("all", "local", "frontier", "kimi", "subscription"),
        default="all",
        help="which model lanes this run may use",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify the committed second-round files without generating anything",
    )
    parser.add_argument(
        "--score",
        type=Path,
        metavar="VERDICTS",
        help="score a committed verdicts file against the packet and key",
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="write the score report here instead of standard output",
    )
    args = parser.parse_args()

    if args.check:
        return run_check()
    if args.score is not None:
        return run_score(args.score, args.out)

    if args.models in {"kimi", "subscription"}:
        print(
            f"{args.models} is not run: the endpoint or lane is not up. "
            f"The command that will run it: "
            f"{KIMI_SECOND_COMMAND if args.models == 'kimi' else 'see benchmark.json not_run'}",
            file=sys.stderr,
        )
        return 0

    _, members, _, development, held_out = split_inputs()

    if args.stage in {
        "second-development",
        "second-heldout",
        "second-report",
        "second-round",
    }:
        return run_second_stages(args.stage, members, development, held_out)

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


def run_second_stages(
    stage: str,
    members: list[dict[str, Any]],
    development: list[dict[str, Any]],
    held_out: list[dict[str, Any]],
) -> int:
    if stage in {"second-development", "second-round"}:
        cases = load_second_cases(development)
        guard = SpendGuard(SECOND_SPEND_CEILING_USD, existing_spend())
        rounds = asyncio.run(run_development(development, cases, guard))
        path = save_development(rounds)
        print(
            json.dumps(
                {
                    "development_members": len(development),
                    "saved": str(path),
                    "rounds": [
                        {
                            "prompt_version": round_row["prompt_version"],
                            "arms": [
                                {
                                    "arm": arm["arm"],
                                    "proxy_share": round(arm["share"], 4),
                                    "blocked": arm["blocked"],
                                }
                                for arm in round_row["arms"]
                            ],
                        }
                        for round_row in rounds
                    ],
                    "spend": round(guard.spent, 4),
                },
                indent=2,
            )
        )

    if stage in {"second-heldout", "second-round"}:
        cases = load_second_cases(held_out)
        guard = SpendGuard(SECOND_SPEND_CEILING_USD, existing_spend())
        asyncio.run(run_held_out(held_out, cases, guard))
        print(
            json.dumps(
                {
                    "held_out_members": len(held_out),
                    "records": len(read_jsonl(SECOND_RECORDS)),
                    "spend": round(guard.spent, 4),
                    "ceiling_usd": SECOND_SPEND_CEILING_USD,
                },
                indent=2,
            )
        )

    if stage in {"second-report", "second-round"}:
        cases = load_second_cases(held_out)
        document = write_second_round(members, held_out, cases)
        print(second_stage_summary(document))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
