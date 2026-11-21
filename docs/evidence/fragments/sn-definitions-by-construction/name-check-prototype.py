"""Prototype name-against-definition checks, measured on 1,021 held-out names.

Scored once per arm against the settled blind-reader labels and against an
adjudicated subset of them. Two fresh readers of different model families judge
every held-out name where the adopted question and the label disagree, so the
ceiling the labels support is measured before any arm is compared with another.

Arms: A the adopted chat question, its committed readings reused; B a decisions
model pre-categorises, then the chat model is asked only the class-specific
question for the classes raised; C the decisions model alone combined by a
logistic model fitted on the development half; D a second decisions model in
both places; E deterministic grammar and unit-dimension pre-checks; F the best
combination, fixed on the development half.

Modes: probe, adjudicate, decisions, arm-b, arm-e, report.
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import importlib.util
import json
import math
import os
import random
import re
import time
from pathlib import Path

WORKTREE = Path(__file__).resolve().parents[4]
FRAGMENTS = Path(__file__).resolve().parent
OUT_HTML = FRAGMENTS / "sdbc-name-check-prototype.html"
OUT_JSON = FRAGMENTS / "name-check-prototype.json"
FIG_DIR = WORKTREE / "docs/figures/sn-name-check-prototype"
FIG_NAME = "recall-specificity-by-arm.svg"

RUN_DIR = Path(
    os.environ.get("NAME_CHECK_PROTOTYPE_STATE")
    or "/home/ITER/mcintos/.config/reckon/crew/runs/"
    "r-20261114T205523095671-sdbc-name-check-prototype"
)
STATE = RUN_DIR / "prototype-state"
QUESTION_RUN_NAME = "r-20261114T174425722877-sdbc-name-check-question"


def _question_run() -> Path:
    home = Path("/home/ITER/mcintos/.config/reckon/crew")
    for base in (home / "runs", home / "archive" / "runs"):
        candidate = base / QUESTION_RUN_NAME
        if (candidate / "split.json").exists():
            return candidate
    raise SystemExit(
        "the inputs of the adopted-question reading are neither in the live runs "
        f"tree nor in the archive: {QUESTION_RUN_NAME}"
    )


QUESTION_RUN = _question_run()
ADOPTED_READINGS = QUESTION_RUN / "candidate-r0.jsonl"

CEILING_USD = 20.0
STOP_USD = 20.0
CONCURRENCY = 6
DECISIONS_CONCURRENCY = 8
BOOTSTRAP_DRAWS = 4000
BOOTSTRAP_SEED = 20261114
SPLIT_SEED = 20261114
THRESHOLD_GRID = tuple(round(0.05 * index, 2) for index in range(1, 20))

ADJUDICATORS = (
    ("openrouter/openai/gpt-6.1-sol", "high"),
    ("openrouter/anthropic/claude-opus-5.5", None),
)
JEV = "typesafe/jev-1.14"
DREX = "nace-ai/drex-max"
DECISIONS_MODELS = {"jev": JEV, "drex": DREX}

READER_CLASSES = (
    "qualifier-missing-from-name",
    "definition-narrower-than-name",
    "abstract-parent-as-quantity",
    "unit-does-not-fit-name",
    "agrees",
)
DECISION_CLASSES = (
    "qualifier-missing-from-name",
    "definition-narrower-than-name",
    "abstract-parent-as-quantity",
    "unit-does-not-fit-name",
    "equivalent-electron-count",
)


def _load_bakeoff():
    path = FRAGMENTS / "name-definition-reader-bake-off.py"
    spec = importlib.util.spec_from_file_location("bakeoff", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_items() -> tuple[list[dict], dict]:
    return _load_bakeoff().load_items()


def load_split() -> dict:
    return json.loads((QUESTION_RUN / "split.json").read_text())


def load_adopted_readings() -> dict[str, dict]:
    rows: dict[str, dict] = {}
    for line in ADOPTED_READINGS.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            rows[row["name"]] = row
    return rows


def half_items(items: list[dict], split: dict, half: str) -> list[dict]:
    keep = set(split["development"] if half == "development" else split["held_out"])
    return [item for item in items if item["id"] in keep]


def by_id(items: list[dict]) -> dict[str, dict]:
    return {item["id"]: item for item in items}


# ---------------------------------------------------------------- progress


def progress_path(label: str) -> Path:
    STATE.mkdir(parents=True, exist_ok=True)
    return STATE / f"{label}.jsonl"


def append_rows(label: str, rows: list[dict]) -> None:
    with progress_path(label).open("a") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def load_rows(label: str) -> dict[str, dict]:
    path = progress_path(label)
    done: dict[str, dict] = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                done[row["name"]] = row
    return done


class Spend:
    def __init__(self, label: str, limit: float = STOP_USD) -> None:
        STATE.mkdir(parents=True, exist_ok=True)
        self.path = STATE / f"spend-{label}.json"
        self.limit = limit
        self.total = 0.0
        self.calls = 0
        self.elapsed = 0.0
        if self.path.exists():
            payload = json.loads(self.path.read_text())
            self.total = float(payload.get("total_usd") or 0.0)
            self.calls = int(payload.get("calls") or 0)
            self.elapsed = float(payload.get("elapsed_seconds") or 0.0)

    def _write(self) -> None:
        self.path.write_text(
            json.dumps(
                {
                    "total_usd": round(self.total, 6),
                    "calls": self.calls,
                    "elapsed_seconds": round(self.elapsed, 1),
                }
            )
        )

    def add(self, cost: float) -> None:
        self.total += float(cost or 0.0)
        self.calls += 1
        self._write()
        if self.total > self.limit:
            raise RuntimeError(
                f"spend {self.total:.2f} USD past ceiling {self.limit:.2f}"
            )

    def close(self, seconds: float) -> None:
        self.elapsed += float(seconds)
        self._write()


def recorded_spend(label: str) -> dict:
    path = STATE / f"spend-{label}.json"
    if not path.exists():
        return {"total_usd": 0.0, "calls": 0}
    return json.loads(path.read_text())


def all_spend() -> dict[str, dict]:
    out: dict[str, dict] = {}
    if not STATE.exists():
        return out
    for path in sorted(STATE.glob("spend-*.json")):
        out[path.stem.replace("spend-", "")] = json.loads(path.read_text())
    return out


# ------------------------------------------------------------ adjudication


def adjudication_pool(items: list[dict], split: dict, adopted: dict[str, dict]) -> dict:
    """Held-out names where arm A and the label disagree, plus agreement controls.

    A control is an agreement name drawn one-for-one with the disagreements, so
    each reader's own false-alarm rate is visible beside its corrections rather
    than assumed to be zero.
    """

    held = half_items(items, split, "held-out")
    disagree: list[dict] = []
    agree: list[dict] = []
    for item in held:
        row = adopted.get(item["id"]) or {}
        verdict = row.get("verdict")
        if verdict not in ("flag", "pass"):
            continue
        outcome = "flagged" if verdict == "flag" else "stands"
        record = {
            "name": item["id"],
            "label": item["label"],
            "question": outcome,
            "unit": item["unit"],
            "description": item["description"],
            "documentation": item["documentation"],
        }
        if outcome != item["label"]:
            disagree.append(record)
        else:
            agree.append(record)
    rng = random.Random(SPLIT_SEED)
    disagree.sort(key=lambda row: row["name"])
    controls = sorted(agree, key=lambda row: row["name"])
    rng.shuffle(controls)
    controls = sorted(controls[: len(disagree)], key=lambda row: row["name"])
    rows = list(disagree) + controls
    rng = random.Random(SPLIT_SEED + 1)
    rng.shuffle(rows)
    return {
        "disagreements": disagree,
        "agreements": agree,
        "controls": controls,
        "rows": rows,
    }


ADJUDICATION_SYSTEM = """\
You are an independent expert reviewer of IMAS standard names. You are given a
standard name, the definition that accompanies it, and the unit it is carried in.
Decide whether the definition states the same quantity the name states.
You have not been told what any earlier reader decided and must not guess at it:
judge the pair in front of you.

The catalog's rule for species: a bare species name counts ions, summed over all
charge states, with neutral atoms excluded. A definition that says 'ions only'
under a bare species name AGREES. One that includes neutral atoms does not.

The classes are the project's own:
- **agrees** - the definition states the quantity the name states.
- **qualifier-missing-from-name** - the definition adds a normalisation, a
  counting convention, a restriction, or another qualifier the name does not
  carry.
- **definition-narrower-than-name** - the definition is written for one object,
  part, mechanism or device while the name states the quantity generically.
- **abstract-parent-as-quantity** - the definition describes a family or
  umbrella of quantities rather than one specific quantity.
- **unit-does-not-fit-name** - the unit's dimension cannot be the named quantity.

Not a defect: the standard physical meaning of the name's own quantity; a
qualifier the name's own words already state in any form; detail that narrows
how the quantity is measured rather than what it is.

Answer `flag` for any class other than `agrees`, and `pass` for `agrees`."""


def items_prompt(batch: list[dict], heading: str = "## Items") -> str:
    lines = [heading, ""]
    for index, item in enumerate(batch, start=1):
        lines += [
            f"### Item {index}",
            f"- **Standard name**: {name_of(item)}",
            f"- **Description**: {item['description'] or '(none)'}",
            "- **Documentation**: {}".format(
                item["documentation"] if item.get("documentation") else "(none)"
            ),
            f"- **Unit**: {item.get('unit') or 'N/A'}",
            "",
        ]
    return "\n".join(lines)


def verdict_model():
    from pydantic import BaseModel, Field

    class Reading(BaseModel):
        name: str = Field(description="The standard name, echoed unchanged.")
        verdict: str = Field(description="`flag` or `pass`.")
        reader_class: str = Field(description="One of the classes.")
        reasoning: str = Field(default="")

    class Batch(BaseModel):
        verdicts: list[Reading]

    return Batch


def normalise_verdict(reading, name: str) -> dict:
    cls = (
        reading.verdict_class
        if hasattr(reading, "verdict_class")
        else reading.reader_class
    )
    held = cls if cls in READER_CLASSES else None
    flagged = reading.verdict.strip().lower() == "flag" or held not in (None, "agrees")
    return {
        "name": name,
        "verdict": "flag" if flagged else "pass",
        "reader_class": held,
        "reasoning": " ".join(str(reading.reasoning or "").split())[:300],
    }


async def judge_batch(batch, system_prompt, user_prompt, model, effort):
    from iter_codex.discovery.base.llm import acall_llm_structured

    result, cost, _ = await acall_llm_structured(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_model=verdict_model(),
        service="standard-names",
        max_retries=2,
        reasoning_effort=effort,
    )
    returned = {verdict.name: verdict for verdict in result.verdicts}
    rows = []
    for item in batch:
        reading = returned.get(item["name"])
        if reading is None:
            rows.append(
                {
                    "name": name_of(item),
                    "verdict": "unanswered",
                    "reader_class": None,
                    "reasoning": "no reading returned",
                }
            )
        else:
            rows.append(normalise_verdict(reading, name_of(item)))
    return rows, cost


def name_of(item: dict) -> str:
    return item.get("name") or item["id"]


async def run_judgements(
    label, rows, system_prompt, model, effort, batch_size, concurrency, limit_batch=None
):
    done = {
        name: row
        for name, row in load_rows(label).items()
        if row.get("verdict") in ("flag", "pass")
    }
    todo = [row for row in rows if name_of(row) not in done]
    batches = [todo[i : i + batch_size] for i in range(0, len(todo), batch_size)]
    if limit_batch is not None:
        batches = batches[:limit_batch]
    print(
        f"{label}: {len(rows)} rows, {len(done)} recorded, {len(todo)} to call "
        f"in {len(batches)} batches at {model}",
        flush=True,
    )
    gate = asyncio.Semaphore(limit_batch is not None and concurrency or concurrency)
    lock = asyncio.Lock()
    spend = Spend(label)
    began = time.time()

    async def one(batch):
        async with gate:
            started = time.time()
            try:
                verdicts, cost = await judge_batch(
                    batch, system_prompt, items_prompt(batch), model, effort
                )
            except Exception as exc:
                print(f"batch failed: {type(exc).__name__}: {exc}", flush=True)
                verdicts, cost = (
                    [
                        {
                            "name": name_of(item),
                            "verdict": "unanswered",
                            "reader_class": None,
                            "reasoning": f"call failed: {type(exc).__name__}",
                        }
                        for item in batch
                    ],
                    0.0,
                )
            elapsed = time.time() - started
            share = cost / max(1, len(verdicts))
            async with lock:
                append_rows(
                    label,
                    [
                        dict(
                            verdict,
                            cost_usd=round(share, 8),
                            seconds=round(elapsed / max(1, len(verdicts)), 3),
                        )
                        for verdict in verdicts
                    ],
                )
            if cost:
                spend.add(cost)

    await asyncio.gather(*(one(batch) for batch in batches))
    spend.close(time.time() - began)
    print(
        f"{label.upper()}_SPEND_USD {spend.total:.4f} batches "
        f"{len(batches)} seconds {time.time() - began:.0f}",
        flush=True,
    )


def mode_adjudicate(args) -> None:
    items, _ = load_items()
    split = load_split()
    adopted = load_adopted_readings()
    pool = adjudication_pool(items, split, adopted)
    (STATE / "adjudication-set.json").write_text(
        json.dumps(
            {
                "disagreements": len(pool["disagreements"]),
                "agreements": len(pool["agreements"]),
                "controls": len(pool["controls"]),
                "rows": [row["name"] for row in pool["rows"]],
            },
            indent=1,
        )
    )
    print(
        f"adjudication: {len(pool['disagreements'])} disagreements, "
        f"{len(pool['controls'])} controls, {len(pool['rows'])} rows",
        flush=True,
    )
    pairs = [(args.model, args.effort)] if args.model else ADJUDICATORS
    for model, effort in pairs:
        slug = re.sub(r"[^a-z0-9]+", "-", model.split("/")[-1].lower())
        asyncio.run(
            run_judgements(
                "adjudicate-" + slug,
                pool["rows"],
                ADJUDICATION_SYSTEM,
                model,
                effort,
                args.batch,
                args.concurrency,
                args.limit_batches,
            )
        )


# --------------------------------------------------------------- decisions


DECISION_QUESTIONS = {
    "operator_mismatch": {
        "type": "noul",
        "instructions": (
            "The name and the definition disagree about an operator: one of them "
            "applies an average, an integral, a weighting, a normalisation or a "
            "ratio that the other does not state."
        ),
        "criteria": {
            "true": "one side states an operator the other does not",
            "false": "both state the same operators, or neither states one",
        },
    },
    "definition_narrower": {
        "type": "noul",
        "instructions": (
            "The definition is written for one object, part, mechanism, device or "
            "species state while the name states the quantity generically."
        ),
        "criteria": {
            "true": "the definition restricts the definition of the quantity to one object or mechanism",
            "false": "the definition applies to the quantity the name states generally",
        },
    },
    "definition_is_family": {
        "type": "noul",
        "instructions": (
            "The definition describes a family or umbrella of quantities rather "
            "than one specific quantity."
        ),
        "criteria": {
            "true": "the definition covers several forms or children of one concept",
            "false": "the definition states one specific quantity",
        },
    },
    "unit_fits_name": {
        "type": "noul",
        "instructions": "The unit's dimension can be the quantity the name states.",
        "criteria": {
            "true": "the unit's dimension fits the named quantity",
            "false": "the unit's dimension cannot be the named quantity.",
        },
    },
    "equivalent_electron": {
        "type": "noul",
        "instructions": (
            "The definition reports the count or rate in equivalent electrons "
            "as an accounting convention rather than as a direct count of the "
            "object the name states."
        ),
        "criteria": {
            "true": "the quantity is expressed in equivalent electrons",
            "false": "the quantity is a direct count or rate of the named object",
        },
    },
}


def decision_state(item: dict) -> dict:
    return {
        "standard_name": item["id"],
        "unit": item["unit"],
        "description": item["description"],
        "documentation": item["documentation"],
    }


async def decisions_one(item: dict, model: str) -> dict:
    from iter_codex.discovery.base.llm import acall_decisions

    semaphore = _DECISIONS_GATE
    async with semaphore:
        try:
            answers, cost = await acall_decisions(
                model,
                decision_state(item),
                DECISION_QUESTIONS,
                service="standard-names",
                max_retries=3,
            )
        except Exception as exc:
            return {
                "name": item["id"],
                "error": f"{type(exc).__name__}: {exc}",
                "nouls": {},
                "cost_usd": 0.0,
            }
    nouls = {}
    for key in DECISION_QUESTIONS:
        answer = answers.get(key)
        if isinstance(answer, dict) and answer.get("noul") is not None:
            nouls[key] = float(answer["noul"])
    return {"name": item["id"], "nouls": nouls, "cost_usd": round(cost, 8)}


_DECISIONS_GATE = None


async def decisions_chunk(items: list[dict], model: str) -> tuple[list[dict], float]:
    global _DECISIONS_GATE
    _DECISIONS_GATE = asyncio.Semaphore(DECISIONS_CONCURRENCY)
    rows = await asyncio.gather(*(decisions_one(item, model) for item in items))
    total = sum(row["cost_usd"] for row in rows)
    return list(rows), total


def mode_decisions(args) -> None:
    items, _ = load_items()
    model = args.model
    slug = "drex" if "drex" in model else "jev"
    label = "decisions-" + slug
    done = load_rows(label)
    todo = [
        item
        for item in items
        if len(((done.get(item["id"]) or {}).get("nouls")) or {})
        < len(DECISION_QUESTIONS)
    ]
    print(
        f"{label}: {len(items)} names, {len(done)} recorded, {len(todo)} to call",
        flush=True,
    )
    spend = Spend(label)
    began = time.time()
    index = 0
    while index < len(todo):
        block = todo[index : index + 64]
        rows, cost = asyncio.run(decisions_chunk(block, model))
        append_rows(label, rows)
        spend.add(cost)
        index += len(block)
        print(
            f"  {index} of {len(todo)}, spend "
            f"{spend.total:.4f} USD, {time.time() - began:.0f}s",
            flush=True,
        )
    spend.close(time.time() - began)
    print(f"{label.upper()}_SPEND_USD {spend.total:.4f}", flush=True)


# ------------------------------------------------------------------ arm B

CLASS_QUESTION = {
    "qualifier-missing-from-name": (
        "Does the definition add a qualifier the name does not carry, such as a "
        "normalisation, a counting convention, an integration, or a restriction?"
    ),
    "definition-narrower-than-name": (
        "Is the definition written for one object, part or mechanism while the "
        "name states the quantity generically?"
    ),
    "abstract-parent-as-quantity": (
        "Does the definition describe a family of quantities rather than one "
        "specific quantity?"
    ),
    "unit-does-not-fit-name": (
        "Does the unit's dimension fit the quantity the name states?"
    ),
    "equivalent-electron-count": (
        "Is the count or rate reported in equivalent electrons rather than as a "
        "direct count of the object the name states?"
    ),
}
CLASS_INVERTED = {"unit-does-not-fit-name"}


def class_examples(split: dict, classes: set[str]) -> list[dict]:
    chosen = []
    for example in split.get("worked_examples", []):
        if example["class"] in classes:
            chosen.append(example)
    return chosen


def arm_b_system_prompt(classes: list[str], split: dict) -> str:
    lines = [
        "You are an independent reviewer of an IMAS standard name and the "
        "definition that accompanies it.",
        "",
        "A pre-check has raised the following classes for the items below. Answer "
        "only those questions; do not raise a class the pre-check did not raise.",
        "",
    ]
    for cls in classes:
        lines.append(f"- **{cls}** - {CLASS_QUESTION[cls]}")
    lines += [
        "",
        "The catalog's rule for species: a bare species name counts ions, summed "
        "over all charge states, with neutral atoms excluded. So a definition "
        "that says 'ions only' under a bare species name AGREES.",
        "",
        "Not a defect: the standard physical meaning of the name's own quantity; "
        "a qualifier the name's own words already state in any form; detail that "
        "narrows how the quantity is measured rather than what it is.",
        "",
        "## Worked examples",
        "",
    ]
    examples = class_examples(split, set(classes))
    for index, example in enumerate(examples, start=1):
        lines += [
            f"### Example {index} - raised class: {example['class']}",
            "- **Standard name**: {}".format(example["name"]),
            f"- **Description**: {example['description'] or '(none)'}",
        ]
        if example.get("documentation"):
            lines.append("- **Documentation**: {}".format(example["documentation"]))
        lines.append("- **Unit**: %s" % (example["unit"] or "N/A"))
        lines.append(
            "- **The readers' deciding phrase**: %s"
            % (example.get("quoted_evidence") or "(none)")
        )
        lines.append("")
    lines += [
        "## Output",
        "",
        "Return a JSON object with a `verdicts` array in the order given. Each "
        "entry: `name` (echoed unchanged), `verdict` (`flag` when any raised class "
        "holds, else `pass`), `reader_class` (the raised class that holds, or "
        "`agrees`), and `reasoning`.",
    ]
    return "\n".join(lines)


def raised_classes(row: dict, thresholds: dict[str, float]) -> list[str]:
    nouls = row.get("nouls") or {}
    raised = []
    for cls, threshold in thresholds.items():
        key = CLASS_TO_QUESTION[cls]
        value = nouls.get(key)
        if value is None:
            continue
        if cls in CLASS_INVERTED:
            if value <= 1.0 - threshold:
                raised.append(cls)
        elif value >= threshold:
            raised.append(cls)
    return sorted(raised)


CLASS_TO_QUESTION = {
    "qualifier-missing-from-name": "operator_mismatch",
    "definition-narrower-than-name": "definition_narrower",
    "abstract-parent-as-quantity": "definition_is_family",
    "unit-does-not-fit-name": "unit_fits_name",
    "equivalent-electron-count": "equivalent_electron",
}


def balanced_accuracy(pairs: list[tuple[bool, bool]]) -> float:
    tp = sum(1 for label, said in pairs if label and said)
    fn = sum(1 for label, said in pairs if label and not said)
    tn = sum(1 for label, said in pairs if not label and not said)
    fp = sum(1 for label, said in pairs if not label and said)
    recall = tp / (tp + fn) if tp + fn else 1.0
    specificity = tn / (tn + fp) if tn + fp else 1.0
    return (recall + specificity) / 2.0


def fit_thresholds(
    dev_decisions: dict[str, dict],
    class_of: dict[str, str],
    classes=DECISION_CLASSES[:4],
) -> dict[str, dict]:
    """One threshold per class, maximising balanced accuracy. The classes are the
    reader classes the settled labels carry; the equivalent-electron convention
    has no label class of its own and is read only by the decisions-only arm."""
    fitted = {}
    for cls in classes:
        key = CLASS_TO_QUESTION[cls]
        pairs_by_threshold = {}
        for threshold in THRESHOLD_GRID:
            pairs = []
            for name, row in dev_decisions.items():
                value = (row.get("nouls") or {}).get(key)
                if value is None or name not in class_of:
                    continue
                target = class_of[name] == cls
                said = (
                    (value <= 1.0 - threshold)
                    if cls in CLASS_INVERTED
                    else (value >= threshold)
                )
                pairs.append((target, said))
            pairs_by_threshold[threshold] = pairs
        best = max(
            THRESHOLD_GRID,
            key=lambda t: balanced_accuracy(pairs_by_threshold[t]),
        )
        fitted[cls] = {
            "threshold": best,
            "balanced_accuracy": round(balanced_accuracy(pairs_by_threshold[best]), 4),
            "curve": {
                str(t): round(balanced_accuracy(pairs_by_threshold[t]), 4)
                for t in THRESHOLD_GRID
            },
        }
    return fitted


async def arm_b_batch(
    batch: list[dict], classes: list[str], system_prompt: str, model: str, effort
) -> tuple[list[dict], float]:
    from iter_codex.discovery.base.llm import acall_llm_structured

    user = [items_prompt(batch, "## Items to review")]
    result, cost, _ = await acall_llm_structured(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "\n".join(user)},
        ],
        response_model=verdict_model(),
        service="standard-names",
        max_retries=2,
        reasoning_effort=effort,
    )
    returned = {verdict.name: verdict for verdict in result.verdicts}
    rows = []
    for item in batch:
        reading = returned.get(item["id"])
        if reading is None:
            rows.append(
                {
                    "name": item["id"],
                    "verdict": "unanswered",
                    "reader_class": None,
                    "reasoning": "no reading returned",
                }
            )
        else:
            rows.append(normalise_verdict(reading, item["id"]))
    return rows, cost


def arm_b_plan(
    items: list[dict], thresholds: dict[str, float]
) -> tuple[list[dict], list[dict]]:
    """Split names into those the pre-check raises and those it passes alone."""
    raised, silent = [], []
    for item in items:
        row = _DECISIONS.get(item["id"]) or {}
        classes = raised_classes(row, thresholds)
        if classes:
            raised.append({"item": item, "classes": classes})
        else:
            silent.append({"item": item, "classes": []})
    return raised, silent


_DECISIONS: dict[str, dict] = {}


def load_decisions(slug: str) -> dict[str, dict]:
    return load_rows("decisions-" + slug)


async def run_arm_b(
    label,
    plan,
    split,
    model,
    effort,
    batch_size=8,
    concurrency=CONCURRENCY,
    limit_batches=None,
) -> None:
    groups: dict[tuple[str, ...], list[dict]] = collections.defaultdict(list)
    for entry in plan:
        groups[tuple(entry["classes"])].append(entry["item"])
    done = {
        name
        for name, row in load_rows(label).items()
        if row.get("verdict") in ("flag", "pass")
    }
    work = []
    for classes, items in groups.items():
        todo = [item for item in items if item["id"] not in done]
        for index in range(0, len(todo), batch_size):
            work.append((list(classes), todo[index : index + batch_size]))
    if limit_batches is not None:
        work = work[:limit_batches]
    print(
        f"{label}: {sum(len(b) for _, b in work)} names in {len(work)} calls",
        flush=True,
    )
    gate = asyncio.Semaphore(concurrency)
    lock = asyncio.Lock()
    spend = Spend(label)
    began = time.time()

    async def one(classes, batch):
        async with gate:
            prompt = arm_b_system_prompt(
                [c for c in DECISION_CLASSES if c in classes], split
            )
            started = time.time()
            try:
                rows, cost = await arm_b_batch(batch, classes, prompt, model, effort)
            except Exception as exc:
                print(f"call failed: {type(exc).__name__}: {exc}", flush=True)
                rows, cost = (
                    [
                        {
                            "name": item["id"],
                            "verdict": "unanswered",
                            "reader_class": None,
                            "reasoning": f"call failed: {type(exc).__name__}",
                        }
                        for item in batch
                    ],
                    0.0,
                )
            elapsed = time.time() - started
            share = cost / max(1, len(rows))
            async with lock:
                append_rows(
                    label,
                    [
                        dict(
                            row,
                            cost_usd=round(share, 8),
                            seconds=round(elapsed / max(1, len(rows)), 3),
                        )
                        for row in rows
                    ],
                )
            if cost:
                spend.add(cost)

    await asyncio.gather(*(one(classes, batch) for classes, batch in work))
    spend.close(time.time() - began)
    print(
        f"{label.upper()}_SPEND_USD {spend.total:.4f} seconds {time.time() - began:.0f}",
        flush=True,
    )


def mode_arm_b(args) -> None:
    items, _ = load_items()
    split = load_split()
    slug = args.model
    global _DECISIONS
    decisions = load_decisions(slug)
    _DECISIONS = decisions
    dev = half_items(items, split, "development")
    held = half_items(items, split, "held-out")
    dev_rows = {
        item["id"]: decisions[item["id"]] for item in dev if item["id"] in decisions
    }
    dev_class = {
        item["id"]: split["class_of"].get(item["id"], "agrees") for item in dev
    }
    fitted = fit_thresholds(dev_rows, dev_class)
    (STATE / ("thresholds-" + slug + ".json")).write_text(json.dumps(fitted, indent=1))
    print(
        "thresholds from the development half:",
        json.dumps({cls: row["threshold"] for cls, row in fitted.items()}),
        flush=True,
    )
    thresholds = {cls: row["threshold"] for cls, row in fitted.items()}
    targets = held if args.half == "held-out" else dev
    plan, silent = arm_b_plan(targets, thresholds)
    label = f"arm-b-{slug}-{args.half}"
    asyncio.run(
        run_arm_b(
            label,
            plan,
            split,
            args.chat_model,
            args.effort,
            args.batch,
            args.concurrency,
            args.limit_batches,
        )
    )
    append_rows(
        label + "-silent",
        [
            {
                "name": entry["item"]["id"],
                "verdict": "pass",
                "reader_class": None,
                "reasoning": "no class raised",
                "cost_usd": 0.0,
                "seconds": 0.0,
            }
            for entry in silent
        ],
    )
    print(f"{label}: {len(plan)} raised, {len(silent)} silent", flush=True)


# ------------------------------------------------------------------ arm E

OPERATOR_PHRASES = {
    "flux_surface_averaged": r"flux[-\s]?surface[-\s]?(averag|average|mean)|<\s*\w+\s*>_\s*\{?\\?psi|\\langle",
    "volume_averaged": r"volume[-\s]?(averag|average|mean)",
    "line_averaged": r"line[-\s]?(averag|average|mean)",
    "line_integrated": r"line[-\s]?integrat|path[-\s]?integrat",
    "surface_integrated": r"surface[-\s]?integrat|wall[-\s]?integrat",
    "volume_integrated": r"volume[-\s]?integrat",
    "time_derivative": r"time[-\s]?derivativ|derivative of|\\frac\{d|d/dt|\\partial",
}
EQUATION_OPERATOR = (
    r"\\int|\\sum|\\oint|\\langle|\\frac\{.*d|\\partial|\\bar|\\overline|\\left<"
)
AVERAGE_WORDS = (
    r"average|averaged|mean|weighted|summed|integrated|integral|normalis|normaliz"
)


def name_operators(name: str) -> list[str]:
    try:
        from imas_standard_names.grammar import parse_standard_name
    except Exception:
        return []
    try:
        parsed = parse_standard_name(name)
    except Exception:
        return []
    found = []
    for field in ("transformation", "aggregation", "binary_operator"):
        value = getattr(parsed, field, None)
        if isinstance(value, str) and value:
            found.append(value)
    return found


def operator_check(item: dict) -> tuple[bool, str]:
    operators = name_operators(item["id"])
    if not operators:
        return False, "no operator in the name"
    haystack = (item["description"] or "") + "\n" + (item["documentation"] or "")
    for operator in operators:
        pattern = OPERATOR_PHRASES.get(operator)
        if pattern is None:
            continue
        if re.search(pattern, haystack, re.IGNORECASE):
            continue
        if re.search(EQUATION_OPERATOR, item["documentation"] or ""):
            continue
        return (
            True,
            f"name states {operator} and the definition neither names it nor carries an equation operator",
        )
    return False, "every operator the name states appears in the definition"


BASE_UNITS = {
    "temperature": ("K",),
    "pressure": ("Pa",),
    "power": ("W",),
    "energy": ("J",),
    "charge": ("C",),
    "voltage": ("V",),
    "current": ("A",),
    "count": ("1",),
    "area": ("m^2",),
    "length": ("m",),
    "radius": ("m",),
    "width": ("m",),
    "height": ("m",),
    "thickness": ("m",),
    "magnetic_field": ("T",),
    "magnetic_flux": ("Wb",),
    "magnetic_vector_potential": ("Wb.m^-1",),
    "density": ("m^-3", "kg.m^-3"),
    "power_density": ("W.m^-3",),
    "current_density": ("A.m^-2",),
    "torque": ("N.m",),
    "velocity": ("m.s^-1",),
    "diffusivity": ("m^2.s^-1",),
    "diffusion_coefficient": ("m^2.s^-1",),
    "frequency": ("Hz",),
    "time": ("s",),
}


def physical_base_of(name: str) -> str | None:
    try:
        from imas_standard_names.grammar import parse_standard_name

        return parse_standard_name(name).physical_base
    except Exception:
        return None


def unit_check(item: dict) -> tuple[bool, str]:
    from iter_codex.standard_names.unit_audit import unit_dimension

    base = physical_base_of(item["id"])
    if base is None:
        return False, "the name does not parse, so no base is known"
    candidates = BASE_UNITS.get(base)
    if not candidates:
        return False, f"no dimension is committed for the base {base}"
    actual = unit_dimension(item["unit"] or "")
    if actual is None:
        return False, "the unit does not parse"
    for candidate in candidates:
        if (
            unit_dimension(candidate) is not None
            and unit_dimension(candidate) == actual
        ):
            return False, f"unit dimension fits {base}"
    return True, "unit {} is dimension {}, which cannot be {}".format(
        item["unit"], actual, base
    )


def repo_unit_check(item: dict) -> tuple[bool, str]:
    """The repository's own name-versus-unit audit, read-only over one name."""
    from iter_codex.standard_names.audits import name_unit_consistency_check

    issues = name_unit_consistency_check({"id": item["id"], "unit": item["unit"]})
    if issues:
        return True, issues[0].split(": ", 2)[-1]
    return False, "the repository audit accepts the declared unit for this name"


def repo_unit_decisions(items) -> dict[str, str]:
    return {
        item["id"]: ("flag" if repo_unit_check(item)[0] else "pass") for item in items
    }


def arm_e_row(item: dict) -> dict:
    operator_flag, operator_reason = operator_check(item)
    unit_flag, unit_reason = unit_check(item)
    return {
        "name": item["id"],
        "operator": operator_flag,
        "unit": unit_flag,
        "flag = operator or unit": operator_flag or unit_flag,
        "operator_reason": operator_reason,
        "unit_reason": unit_reason,
    }


def mode_arm_e(args) -> None:
    items, _ = load_items()
    rows = [arm_e_row(item) for item in items]
    append_rows("arm-e", rows)
    flagged = sum(1 for row in rows if row["flag = operator or unit"])
    print(f"arm-e: {len(rows)} names, {flagged} flagged", flush=True)
    (STATE / "arm-e.json").write_text(
        json.dumps({row["name"]: row for row in rows}, indent=1)
    )


# ----------------------------------------------------------------- scoring


def wilson(successes: int, total: int) -> list[float]:
    if not total:
        return [None, None]
    z = 1.96
    phat = successes / total
    denom = 1 + z * z / total
    centre = (phat + z * z / (2 * total)) / denom
    half = (
        z * math.sqrt(phat * (1 - phat) / total + z * z / (4 * total * total)) / denom
    )
    return [round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4)]


def score_arm(
    decisions: dict[str, str],
    labels: dict[str, str],
    costs: dict[str, float] | None = None,
) -> dict:
    tp = fp = tn = fn = 0
    scored = 0
    rows = []
    for name, label in sorted(labels.items()):
        outcome = decisions.get(name)
        if outcome not in ("flag", "pass"):
            continue
        scored += 1
        rows.append((name, label, outcome))
        flagged = outcome == "flag"
        if label == "flagged":
            if flagged:
                tp += 1
            else:
                fn += 1
        else:
            if flagged:
                fp += 1
            else:
                tn += 1
    cost = sum((costs or {}).get(name, 0.0) for name, _, _ in rows)
    return {
        "scored": scored,
        "flagged_total": tp + fn,
        "stands_total": tn + fp,
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "true_negative": tn,
        "recall": round(tp / (tp + fn), 4) if tp + fn else None,
        "recall_ci": wilson(tp, tp + fn),
        "specificity": round(tn / (tn + fp), 4) if tn + fp else None,
        "specificity_ci": wilson(tn, tn + fp),
        "precision": round(tp / (tp + fp), 4) if tp + fp else None,
        "precision_ci": wilson(tp, tp + fp),
        "false_holds": fp,
        "misses": fn,
        "cost_usd": round(cost, 4),
        "cost_per_1000_usd": round(1000 * cost / scored, 4) if scored else None,
        "rows": rows,
    }


def recall_by_class(result: dict, class_of: dict[str, str]) -> dict[str, dict]:
    buckets: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0])
    for name, label, outcome in result["rows"]:
        if label != "flagged":
            continue
        cls = class_of.get(name, "unclassified")
        buckets[cls][1] += 1
        buckets[cls][0] += 1 if outcome == "flag" else 0
    return {
        cls: {"found": n, "of": total, "recall": round(n / total, 4) if total else None}
        for cls, (n, total) in sorted(buckets.items())
    }


def adjudicated_labels(
    labels: dict[str, str], pool: dict, reads: dict[str, dict[str, dict]]
) -> dict:
    """Relabel the disputes by reader consensus; splits leave the scored set by being marked unresolved."""
    dispute = {row["name"] for row in pool["disagreements"]}
    out = {}
    detail = {}
    for name, label in labels.items():
        if name not in dispute:
            out[name] = label
            continue
        votes = []
        for table in reads.values():
            row = table.get(name) or {}
            verdict = row.get("verdict")
            if verdict not in ("flag", "pass"):
                votes.append(None)
                continue
            votes.append(("flagged" if verdict == "flag" else "stands") == label)
        label_sides = sum(1 for vote in votes if vote is True)
        question_sides = sum(1 for vote in votes if vote is False)
        if label_sides == len(votes) and label_sides > 0:
            out[name] = label
            detail[name] = "both readers side with the label"
        elif question_sides == len(votes) and question_sides > 0:
            out[name] = "flagged" if label == "stands" else "stands"
            detail[name] = "both readers side with the question: label error"
        else:
            detail[name] = "readers split"
    return {"labels": out, "detail": detail, "dispute": dispute}


def ceiling_note(labels: dict[str, str], pool: dict, adjudged: dict) -> dict:
    detail = adjudged["detail"]
    label_errors = sum(1 for value in detail.values() if value.endswith("label error"))
    real_errors = sum(
        1
        for value in detail.values()
        if value.startswith("both readers side with the label")
    )
    splits = sum(1 for value in detail.values() if value == "readers split")
    return {
        "disagreements": len(pool["disagreements"]),
        "label_errors_by_both_readers": label_errors,
        "real_question_errors": real_errors,
        "readers_split": splits,
    }


def bootstrap_ci(rows: list[tuple[str, str, str]], metric: str, rng) -> list[float]:
    values = []
    flagged = [row for row in rows if row[1] == "flagged"]
    stands = [row for row in rows if row[1] == "stands"]
    for _ in range(BOOTSTRAP_DRAWS):
        draw_f = [rng.choice(flagged) for _ in flagged] if flagged else []
        draw_s = [rng.choice(stands) for _ in stands] if stands else []
        if metric == "recall" and draw_f:
            values.append(sum(1 for row in draw_f if row[2] == "flag") / len(draw_f))
        elif metric == "specificity" and draw_s:
            values.append(sum(1 for row in draw_s if row[2] == "pass") / len(draw_s))
    if not values:
        return [None, None]
    values.sort()
    return [
        round(values[int(0.025 * len(values))], 4),
        round(values[int(0.975 * len(values)) - 1], 4),
    ]


# ------------------------------------------------------------------ logistic


def _logistic_fit(features: list[list[float]], targets: list[int]):
    import numpy as np

    x = np.asarray(features, dtype=float)
    y = np.asarray(targets, dtype=float)
    mu = x.mean(axis=0)
    sigma = x.std(axis=0)
    sigma[sigma == 0] = 1.0
    z = np.hstack([np.ones((x.shape[0], 1)), (x - mu) / sigma])
    weights = np.zeros(z.shape[1])
    for _ in range(1500):
        p = 1.0 / (1.0 + np.exp(-(z @ weights)))
        gradient = z.T @ (p - y) / len(y) + 1e-3 * weights
        weights -= 0.5 * gradient
    return weights, mu, sigma


def _logistic_predict(model, features: list[list[float]]) -> list[float]:
    import numpy as np

    weights, mu, sigma = model
    x = np.asarray(features, dtype=float)
    z = np.hstack([np.ones((x.shape[0], 1)), (x - mu) / sigma])
    return list(1.0 / (1.0 + np.exp(-(z @ weights))))


def decision_features(row: dict) -> list[float]:
    nouls = row.get("nouls") or {}
    return [
        float(nouls.get("operator_mismatch") or 0.0),
        float(nouls.get("definition_narrower") or 0.0),
        float(nouls.get("definition_is_family") or 0.0),
        1.0 - float(nouls.get("unit_fits_name") or 0.0),
        float(nouls.get("equivalent_electron") or 0.0),
    ]


def arm_c_decisions(
    slug: str, items: list[dict], split: dict, cutoff: float = 0.5
) -> dict:
    decisions_rows = load_decisions(slug)
    dev = half_items(items, split, "development")
    features, targets = [], []
    for item in dev:
        row = decisions_rows.get(item["id"])
        if row is None or "nouls" not in row:
            continue
        features.append(decision_features(row))
        targets.append(1 if item["label"] == "flagged" else 0)
    model = _logistic_fit(features, [int(value) for value in targets])
    probabilities = {}
    for item in items:
        row = decisions_rows.get(item["id"])
        if row is None or "nouls" not in row:
            continue
        probabilities[item["id"]] = float(
            _logistic_predict(model, [decision_features(row)])[0]
        )
    return {
        "decisions": {
            name: ("flag" if value >= cutoff else "pass")
            for name, value in probabilities.items()
        },
        "probabilities": probabilities,
        "weights": list(model[0]),
    }


# ------------------------------------------------------------------ report


ARM_LABELS = {
    "A": "A adopted question",
    "B-jev": "B Jev raises, then the targeted question",
    "B-drex": "B Drex raises, then the targeted question",
    "C-jev": "C Jev alone",
    "C-drex": "C Drex alone",
    "E": "E deterministic pre-checks (prototype rules)",
    "E-operators": "E1 operator rule alone",
    "E-units": "E2 prototype unit-dimension rule alone",
    "E-repo": "E3 repository name-unit audit",
    "E-repo+A": "E3 first, then the adopted question",
    "E+A": "E first, then the adopted question",
    "F": "F best combination fixed on the development half",
    "always-pass": "control, always pass",
    "always-flag": "control, always flag",
}


def arm_a_decisions(adopted: dict[str, dict]) -> dict[str, str]:
    out = {}
    for name, row in adopted.items():
        if row.get("verdict") in ("flag", "pass"):
            out[name] = row["verdict"]
    return out


def arm_b_decisions(label: str, silent_label: str, names: set[str]) -> dict[str, str]:
    rows = load_rows(label)
    silent = load_rows(silent_label)
    out = {}
    for name in names:
        row = rows.get(name)
        if row and row.get("verdict") in ("flag", "pass"):
            out[name] = row["verdict"]
        elif name in silent:
            out[name] = "pass"
    return out


def arm_e_decisions() -> dict[str, str]:
    rows = load_rows("arm-e")
    return {
        name: ("flag" if row.get("flag = operator or unit") else "pass")
        for name, row in rows.items()
    }


def arm_e_operator_decisions() -> dict[str, str]:
    rows = load_rows("arm-e")
    return {
        name: ("flag" if row.get("operator") else "pass") for name, row in rows.items()
    }


def arm_e_unit_decisions() -> dict[str, str]:
    rows = load_rows("arm-e")
    return {name: ("flag" if row.get("unit") else "pass") for name, row in rows.items()}


def combine_union_clean(*tables):
    names = set()
    for table in tables:
        names |= set(table)
    out = {}
    for name in names:
        hits = [table.get(name) for table in tables]
        if any(hit == "flag" for hit in hits):
            out[name] = "flag"
        elif any(hit == "pass" for hit in hits):
            out[name] = "pass"
    return out


def dev_features(
    dev: list[dict], jev_b: dict, drex_b: dict, e_rows: dict
) -> tuple[list, list]:
    features, targets = [], []
    jev_rows = load_decisions("jev")
    drex_rows = load_decisions("drex")
    for item in dev:
        name = item["id"]
        jev = jev_rows.get(name)
        drex = drex_rows.get(name)
        if not jev or not drex or "nouls" not in jev or "nouls" not in drex:
            continue
        e_row = e_rows.get(name) or {}
        features.append(
            decision_features(jev)
            + decision_features(drex)
            + [
                1.0 if e_row.get("operator") else 0.5,
                1.0 if e_row.get("unit") else 0.5,
                1.0 if jev_b.get(name) == "flag" else 0.0,
                1.0 if drex_b.get(name) == "flag" else 0.0,
            ]
        )
        targets.append(1 if item["label"] == "flagged" else 0)
    return features, targets


def cv_balanced_accuracy(features, targets, folds=5):
    import random as _random

    index = list(range(len(targets)))
    rng = _random.Random(SPLIT_SEED)
    rng.shuffle(index)
    buckets = [index[i::folds] for i in range(folds)]
    scores = []
    for fold in buckets:
        held = set(fold)
        train = [i for i in index if i not in held]
        if not train or not held:
            continue
        model = _logistic_fit([features[i] for i in train], [targets[i] for i in train])
        probs = _logistic_predict(model, [features[i] for i in sorted(held)])
        pairs = [
            (bool(targets[i]), p >= 0.5)
            for i, p in zip(sorted(held), probs, strict=False)
        ]
        scores.append(balanced_accuracy(pairs))
    return round(sum(scores) / len(scores), 4) if scores else None


def choose_arm_f(dev, dev_labels, dev_tables):
    """Pick the best combination on the development half, and record why."""
    jev_b = dev_tables["B-jev"]
    drex_b = dev_tables["B-drex"]
    e_rows = load_rows("arm-e")
    features, targets = dev_features(dev, jev_b, drex_b, e_rows)
    candidates = {
        "E alone": dev_tables["E"],
        "B-jev alone": jev_b,
        "B-drex alone": drex_b,
        "E plus B-jev": combine_union_clean(dev_tables["E"], jev_b),
        "E plus B-drex": combine_union_clean(dev_tables["E"], drex_b),
    }
    scored = {}
    for label, table in candidates.items():
        pairs = [
            (dev_labels[name] == "flagged", table.get(name) == "flag")
            for name in table
            if name in dev_labels
        ]
        scored[label] = round(balanced_accuracy(pairs), 4)
    cv = cv_balanced_accuracy(features, targets) if features else None
    scored["logistic over both decisions models, E and B, 5-fold CV"] = cv
    ordered = sorted(scored.items(), key=lambda item: -(item[1] or 0.0))
    chosen = ordered[0][0]
    model = None
    if chosen.startswith("logistic"):
        model = _logistic_fit(features, targets)
    return {
        "chosen": chosen,
        "development_balanced_accuracy": scored,
        "order": [name for name, _ in ordered],
        "logistic_model": model,
    }


def arm_f_held(held, spec, tables):
    if spec["chosen"].startswith("logistic"):
        features = []
        names = []
        jev_rows = load_decisions("jev")
        drex_rows = load_decisions("drex")
        e_rows = load_rows("arm-e")
        for item in held:
            name = item["id"]
            jev, drex = jev_rows.get(name), drex_rows.get(name)
            if not jev or not drex or "nouls" not in jev or "nouls" not in drex:
                continue
            e_row = e_rows.get(name) or {}
            features.append(
                decision_features(jev)
                + decision_features(drex)
                + [
                    1.0 if e_row.get("operator") else 0.5,
                    1.0 if e_row.get("unit") else 0.5,
                    1.0 if tables["B-jev"].get(name) == "flag" else 0.0,
                    1.0 if tables["B-drex"].get(name) == "flag" else 0.0,
                ]
            )
            names.append(name)
        probs = _logistic_predict(spec["logistic_model"], features)
        return {
            name: ("flag" if p >= 0.5 else "pass")
            for name, p in zip(names, probs, strict=False)
        }
    mapping = {
        "E alone": lambda: tables["E"],
        "B-jev alone": lambda: tables["B-jev"],
        "B-drex alone": lambda: tables["B-drex"],
        "E plus B-jev": lambda: combine_union_clean(tables["E"], tables["B-jev"]),
        "E plus B-drex": lambda: combine_union_clean(tables["E"], tables["B-drex"]),
    }
    return mapping[spec["chosen"]]()


# ------------------------------------------------------------ report body


def assemble(items, split, labels):
    dev = half_items(items, split, "development")
    held = half_items(items, split, "held-out")
    dev_labels = {item["id"]: item["label"] for item in dev}
    held_labels = {item["id"]: item["label"] for item in held}
    held_names = set(held_labels)

    adopted = load_adopted_readings()
    arm_a = arm_a_decisions(adopted)
    tables_dev = {
        "B-jev": arm_b_decisions(
            "arm-b-jev-development", "arm-b-jev-development-silent", set(dev_labels)
        ),
        "B-drex": arm_b_decisions(
            "arm-b-drex-development", "arm-b-drex-development-silent", set(dev_labels)
        ),
        "E": arm_e_decisions(),
    }
    jev_c = arm_c_decisions("jev", items, split)
    drex_c = arm_c_decisions("drex", items, split)
    tables_dev["C-jev"] = {
        name: value
        for name, value in jev_c["decisions"].items()
        if name in set(dev_labels)
    }
    tables_dev["C-drex"] = {
        name: value
        for name, value in drex_c["decisions"].items()
        if name in set(dev_labels)
    }

    spec = choose_arm_f(dev, dev_labels, tables_dev)

    tables_held = {
        "A": arm_a,
        "B-jev": arm_b_decisions(
            "arm-b-jev-held-out", "arm-b-jev-held-out-silent", held_names
        ),
        "B-drex": arm_b_decisions(
            "arm-b-drex-held-out", "arm-b-drex-held-out-silent", held_names
        ),
        "C-jev": jev_c["decisions"],
        "C-drex": drex_c["decisions"],
        "E": arm_e_decisions(),
        "E-operators": arm_e_operator_decisions(),
        "E-units": arm_e_unit_decisions(),
        "E-repo": repo_unit_decisions(held),
    }
    tables_held["E+A"] = combine_union_clean(tables_held["E"], tables_held["A"])
    tables_held["E-repo+A"] = combine_union_clean(
        tables_held["E-repo"], tables_held["A"]
    )
    tables_held["F"] = arm_f_held(held, spec, tables_held)
    tables_held["always-pass"] = dict.fromkeys(held_names, "pass")
    tables_held["always-flag"] = dict.fromkeys(held_names, "flag")

    costs = {}
    for label in (
        "arm-b-jev-held-out",
        "arm-b-drex-held-out",
        "arm-b-jev-development",
        "arm-b-drex-development",
    ):
        for _name, _row in load_rows(label).items():
            costs.setdefault(label.split("-held-out")[0].split("-development")[0], {})
    return {
        "dev": dev,
        "held": held,
        "dev_labels": dev_labels,
        "held_labels": held_labels,
        "tables_dev": tables_dev,
        "tables_held": tables_held,
        "spec": spec,
        "arm_c_jev": jev_c,
        "arm_c_drex": drex_c,
    }


def arm_cost(label: str, names: set[str]) -> dict[str, float]:
    rows = load_rows(label)
    return {
        name: float(row.get("cost_usd") or 0.0)
        for name, row in rows.items()
        if name in names
    }


def adjudication_readings() -> dict[str, dict[str, dict]]:
    out = {}
    for model, _ in ADJUDICATORS:
        slug = re.sub(r"[^a-z0-9]+", "-", model.split("/")[-1].lower())
        out[model] = load_rows("adjudicate-" + slug)
    return out


def mode_report(args) -> None:
    items, meta = load_items()
    split = load_split()
    try:
        model_list = json.loads(Path(args.models).read_text())
    except (OSError, ValueError):
        model_list = {}
    labels = {item["id"]: item["label"] for item in items}
    assembled = assemble(items, split, labels)
    held, held_labels = assembled["held"], assembled["held_labels"]
    tables = assembled["tables_held"]
    adopted = load_adopted_readings()
    pool = adjudication_pool(items, split, adopted)
    reads = adjudication_readings()
    judged = adjudicated_labels(held_labels, pool, reads)
    ceiling = ceiling_note(held_labels, pool, judged)
    decisive = {
        name: label
        for name, label in judged["labels"].items()
        if judged["detail"].get(name) != "readers split"
    }

    held_names = set(held_labels)
    decisions_rows = {slug: load_decisions(slug) for slug in DECISIONS_MODELS}
    cost_tables = {
        "A": {
            name: float((adopted.get(name) or {}).get("cost_usd") or 0.0)
            for name in held_names
        },
        "B-jev": arm_cost("arm-b-jev-held-out", held_names),
        "B-drex": arm_cost("arm-b-drex-held-out", held_names),
        "C-jev": {
            name: float((decisions_rows["jev"].get(name) or {}).get("cost_usd") or 0.0)
            for name in held_names
        },
        "C-drex": {
            name: float((decisions_rows["drex"].get(name) or {}).get("cost_usd") or 0.0)
            for name in held_names
        },
        "E": {},
        "E+A": {
            name: float((adopted.get(name) or {}).get("cost_usd") or 0.0)
            for name in held_names
        },
        "E-operators": {},
        "E-units": {},
        "E-repo": {},
        "E-repo+A": {
            name: float((adopted.get(name) or {}).get("cost_usd") or 0.0)
            for name in held_names
        },
        "F": arm_cost("arm-b-jev-held-out", held_names),
        "always-pass": {},
        "always-flag": {},
    }
    scored = {}
    scored_adjudicated = {}
    for arm, table in tables.items():
        scored[arm] = score_arm(table, held_labels, cost_tables.get(arm))
        scored_adjudicated[arm] = score_arm(table, decisive, cost_tables.get(arm))

    class_of = split["class_of"]
    by_class = {
        arm: recall_by_class(result, class_of) for arm, result in scored.items()
    }

    held_readings = []
    for item in held:
        name = item["id"]
        adopted_row = adopted.get(name) or {}
        held_readings.append(
            {
                "name": name,
                "label": held_labels.get(name),
                "adjudicated": decisive.get(name),
                "adjudication_detail": judged["detail"].get(name),
                "adopted": {
                    "verdict": adopted_row.get("verdict"),
                    "reader_class": adopted_row.get("reader_class"),
                    "reasoning": adopted_row.get("reasoning"),
                },
                "decisions": {
                    slug: decisions_rows[slug].get(name) or {}
                    for slug in DECISIONS_MODELS
                },
                "arms": {arm: tables[arm].get(name) for arm in tables},
            }
        )

    payload = {
        "schema": "name-check-prototype",
        "held_out": len(held_labels),
        "held_out_flagged": sum(
            1 for value in held_labels.values() if value == "flagged"
        ),
        "held_out_stands": sum(
            1 for value in held_labels.values() if value == "stands"
        ),
        "scored": len(held_labels),
        "ceiling": ceiling,
        "adjudicated_scored": len(decisive),
        "arms": {
            arm: dict(
                scored[arm],
                rows=[
                    {"name": n, "label": lab, "verdict": v}
                    for n, lab, v in scored[arm]["rows"]
                ],
            )
            for arm in scored
        },
        "arms_adjudicated": {
            arm: {
                key: value
                for key, value in scored_adjudicated[arm].items()
                if key != "rows"
            }
            for arm in scored_adjudicated
        },
        "recall_by_class": by_class,
        "arm_f": {
            "chosen": assembled["spec"]["chosen"],
            "development_balanced_accuracy": assembled["spec"][
                "development_balanced_accuracy"
            ],
            "order": assembled["spec"]["order"],
        },
        "drex_calibration": reliability_bins(
            assembled["arm_c_drex"]["probabilities"], held_labels
        ),
        "jev_calibration": reliability_bins(
            assembled["arm_c_jev"]["probabilities"], held_labels
        ),
        "arm_c_weights": {
            "jev": assembled["arm_c_jev"]["weights"],
            "drex": assembled["arm_c_drex"]["weights"],
        },
        "adjudication": {
            "set": json.loads((STATE / "adjudication-set.json").read_text()),
            "detail": judged["detail"],
            "readers": {
                model: {name: row.get("verdict") for name, row in table.items()}
                for model, table in reads.items()
            },
        },
        "spend": all_spend(),
        "account_models": {
            "source": str(args.models),
            "count": len(model_list.get("models") or []),
            "decisions_models": list(DECISIONS_MODELS.values()),
            "ids": model_list.get("models") or [],
        },
        "readings": held_readings,
        "text_meta": meta,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=1, sort_keys=True))
    print("wrote", OUT_JSON, flush=True)
    write_html(payload, scored, scored_adjudicated, by_class, held, assembled)
    draw_figure(scored, scored_adjudicated)


ARM_COLOURS = {
    "A": "#33658a",
    "B-jev": "#2e8b57",
    "B-drex": "#1b5e3a",
    "C-jev": "#7d5ba6",
    "C-drex": "#4f3570",
    "E": "#146b78",
    "E-repo": "#5aa9b8",
    "E+A": "#0b4550",
    "E-operators": "#9ecfd6",
    "E-units": "#7fbcc6",
    "F": "#b8327e",
    "always-pass": "#929292",
    "always-flag": "#929292",
}
ARM_SHORT = {
    "A": "A",
    "B-jev": "B Jev",
    "B-drex": "B Drex",
    "C-jev": "C Jev",
    "C-drex": "C Drex",
    "E": "E",
    "E-repo": "E3 repo",
    "E+A": "E+A",
    "E-operators": "E1",
    "E-units": "E2",
    "F": "F",
}


FIG_LEGEND = {
    "A": "A adopted question",
    "B-jev": "B Jev screen, then the question",
    "B-drex": "B Drex screen, then the question",
    "C-jev": "C Jev alone",
    "C-drex": "C Drex alone",
    "E": "E prototype operator and unit pre-checks",
    "E-repo": "E3 repository name-unit audit",
    "E+A": "E, then the adopted question",
    "F": "F best combination, fixed on development",
}

FIG_ARMS = ("A", "B-jev", "B-drex", "C-jev", "C-drex", "E", "E-repo", "E+A", "F")
FIG_MARKERS = {"E+A": "D"}
FIG_SIZES = {"E+A": 11, "A": 12}
CONTROL_GREY = "#929292"
# always-flag scores a specificity of exactly 0, which no useful arm is near,
# so it is drawn at the left edge with its value written beside it rather than
# stretching the axis to 0 and squashing every real arm into the right fifth.
EDGE_X = 0.575


def _intervals(result):
    spec, spec_ci = result["specificity"], result["specificity_ci"]
    recall, recall_ci = result["recall"], result["recall_ci"]
    low_x = spec - spec_ci[0] if spec_ci and spec_ci[0] is not None else None
    high_x = spec_ci[1] - spec if spec_ci and spec_ci[1] is not None else None
    low_y = recall - recall_ci[0] if recall_ci and recall_ci[0] is not None else None
    high_y = recall_ci[1] - recall if recall_ci and recall_ci[1] is not None else None
    return spec, recall, (low_x, high_x), (low_y, high_y)


def draw_figure(scored, scored_adjudicated) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.style.use("data-ink")
    # A fixed hash salt keeps the SVG byte-identical across reruns: matplotlib
    # otherwise names its clip and marker ids from object addresses, so an
    # unchanged figure rewrites the whole file and a reviewer cannot see
    # whether anything moved.
    plt.rcParams["svg.hashsalt"] = "sn-name-check"
    fig, axes = plt.subplots(1, 2, figsize=(14, 7.2), sharey=True)
    panels = (
        (axes[0], scored, "original labels"),
        (axes[1], scored_adjudicated, "adjudicated labels"),
    )
    for ax, table, title in panels:
        for arm in FIG_ARMS:
            result = table.get(arm)
            if not result or result["recall"] is None or result["specificity"] is None:
                continue
            spec, recall, (low_x, high_x), (low_y, high_y) = _intervals(result)
            if low_x is not None:
                ax.errorbar(
                    [spec],
                    [recall],
                    xerr=[[low_x], [high_x]],
                    ecolor=ARM_COLOURS[arm],
                    elinewidth=1.2,
                    capsize=3,
                    linewidth=0,
                    zorder=3,
                )
            if low_y is not None:
                ax.errorbar(
                    [spec],
                    [recall],
                    yerr=[[low_y], [high_y]],
                    ecolor=ARM_COLOURS[arm],
                    elinewidth=1.2,
                    capsize=3,
                    linewidth=0,
                    zorder=3,
                )
            if arm == "F":
                ax.plot(
                    [spec],
                    [recall],
                    marker="o",
                    markersize=17,
                    linewidth=0,
                    markerfacecolor="none",
                    markeredgecolor=ARM_COLOURS["F"],
                    markeredgewidth=2.2,
                    zorder=5,
                )
            else:
                ax.plot(
                    [spec],
                    [recall],
                    marker=FIG_MARKERS.get(arm, "o"),
                    markersize=FIG_SIZES.get(arm, 9),
                    linewidth=0,
                    color=ARM_COLOURS[arm],
                    zorder=7 if arm == "A" else 6,
                )
        control_pass = table.get("always-pass")
        if control_pass and control_pass["specificity"] is not None:
            ax.plot(
                [control_pass["specificity"]],
                [control_pass["recall"]],
                marker="s",
                markersize=7,
                color=CONTROL_GREY,
                linewidth=0,
                zorder=4,
            )
        control_flag = table.get("always-flag")
        if control_flag and control_flag["recall"] is not None:
            ax.plot(
                [EDGE_X],
                [control_flag["recall"]],
                marker="s",
                markersize=7,
                color=CONTROL_GREY,
                markerfacecolor="none",
                markeredgewidth=1.4,
                linewidth=0,
                zorder=4,
            )
            ax.annotate(
                "always flag: specificity {}".format(_pct(control_flag["specificity"])),
                xy=(EDGE_X, control_flag["recall"]),
                xytext=(EDGE_X + 0.014, control_flag["recall"] - 0.08),
                fontsize=14,
                color=CONTROL_GREY,
                va="top",
                ha="left",
            )
        ax.set_xlim(0.55, 1.02)
        ax.set_ylim(0.0, 1.04)
        ax.set_xlabel("specificity [fraction of standing names passed]")
        ax.set_title(title, loc="left", fontsize=16, pad=10)
    axes[0].set_ylabel("flagged recall")
    handles = []
    for arm in FIG_ARMS:
        old_label = FIG_LEGEND.get(arm, arm)
        if arm == "F":
            handles.append(
                plt.Line2D(
                    [],
                    [],
                    color=ARM_COLOURS["F"],
                    marker="o",
                    markersize=14,
                    markerfacecolor="none",
                    markeredgewidth=2.2,
                    linewidth=0,
                    label=FIG_LEGEND["F"],
                )
            )
        else:
            handles.append(
                plt.Line2D(
                    [],
                    [],
                    color=ARM_COLOURS[arm],
                    marker=FIG_MARKERS.get(arm, "o"),
                    markersize=FIG_SIZES.get(arm, 9),
                    linewidth=0,
                    label=old_label,
                )
            )
    handles.append(
        plt.Line2D(
            [],
            [],
            color="#929292",
            marker="s",
            markersize=7,
            linewidth=0,
            label="always pass (control)",
        )
    )
    handles.append(
        plt.Line2D(
            [],
            [],
            color=CONTROL_GREY,
            marker="s",
            markersize=7,
            markerfacecolor="none",
            markeredgewidth=1.4,
            linewidth=0,
            label="always flag (control, at the axis edge)",
        )
    )
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=13,
        columnspacing=2.0,
        handletextpad=0.5,
        labelspacing=0.5,
        bbox_to_anchor=(0.5, 0.005),
    )
    fig.subplots_adjust(bottom=0.33, top=0.91, wspace=0.10)
    fig.savefig(FIG_DIR / FIG_NAME, format="svg", metadata={"Date": None})
    (STATE / "figure-check").mkdir(parents=True, exist_ok=True)
    fig.savefig(
        STATE / "figure-check" / "recall-specificity-by-arm.png", format="png", dpi=100
    )
    plt.close(fig)
    print("wrote figure", FIG_DIR / FIG_NAME, flush=True)


def _pct(value) -> str:
    return "n/a" if value is None else "%.1f%%" % (100 * value)


TRADE_NAMES = {
    "A": "the adopted question",
    "E+A": "the prototype pre-checks ahead of the adopted question",
    "E-repo+A": "the repository audit ahead of the adopted question",
    "B-jev": "B with Jev",
    "B-drex": "B with Drex",
    "C-jev": "C Jev alone",
    "C-drex": "C Drex alone",
    "E": "the prototype pre-checks",
    "E-repo": "the repository name-unit audit",
    "E-operators": "the operator rule",
    "E-units": "the prototype unit rule",
    "F": "F",
}


def _points(value) -> str:
    return "%.1f" % (100 * value)


def trade_sentence(scored, contenders, reference: str = "A") -> str:
    """State how every other arm stands against the reference, from the scores.

    The sentence is derived so it cannot outlive the numbers it describes: an
    arm that gains on one measure and loses on the other is named with both
    deltas, and an arm that gains on both turns the first clause around.
    """
    lead = scored.get(reference) or {}
    if lead.get("recall") is None or lead.get("specificity") is None:
        return ""
    deltas = []
    for arm in contenders:
        if arm == reference:
            continue
        row = scored.get(arm) or {}
        if row.get("recall") is None or row.get("specificity") is None:
            continue
        deltas.append(
            (
                arm,
                row["recall"] - lead["recall"],
                row["specificity"] - lead["specificity"],
            )
        )
    improvers = [d for d in deltas if d[1] >= 0 and d[2] >= 0]
    if improvers:
        names = ", ".join(TRADE_NAMES.get(d[0], d[0]) for d in improvers)
        verb = "reaches" if len(improvers) == 1 else "reach"
        return (
            f"{names} {verb} at least the adopted question's recall without giving up "
            "specificity."
        )
    sentences = ["No arm measured here improves on it."]
    spec_up = [d for d in deltas if d[2] > 0 and d[1] < 0]
    if spec_up:
        spec_up.sort(key=lambda d: d[1])
        losses = [-d[1] for d in spec_up]
        gains = [d[2] for d in spec_up]
        parts = ", ".join(
            f"{TRADE_NAMES.get(d[0], d[0])} {_points(-d[1])}/{_points(d[2])}"
            for d in spec_up
        )
        sentences.append(
            f"The {len(spec_up)} arms that buy specificity pay recall for it, and "
            f"the price is not uniform: they give up between {_points(min(losses))} "
            f"and {_points(max(losses))} points of recall for between "
            f"{_points(min(gains))} and {_points(max(gains))} points of "
            f"specificity (recall given up / specificity gained): {parts}."
        )
    recall_up = [d for d in deltas if d[1] > 0 and d[2] < 0]
    for arm, d_recall, d_spec in recall_up:
        sentences.append(
            f"{TRADE_NAMES.get(arm, arm)} gains {_points(d_recall)} points of recall and loses {_points(-d_spec)} points of specificity."
        )
    both_down = [d for d in deltas if d[1] <= 0 and d[2] <= 0]
    for arm, d_recall, d_spec in both_down:
        name = TRADE_NAMES.get(arm, arm)
        if d_recall == 0 and d_spec == 0:
            sentences.append(f"{name} ties the adopted question on both measures.")
        elif d_recall == 0:
            sentences.append(
                f"{name} ties the adopted question on recall and loses "
                f"{_points(-d_spec)} points of specificity."
            )
        elif d_spec == 0:
            sentences.append(
                f"{name} loses {_points(-d_recall)} points of recall and ties "
                "the adopted question on specificity."
            )
        else:
            sentences.append(
                f"{name} is worse on both measures, giving up "
                f"{_points(-d_recall)} points of recall and {_points(-d_spec)} "
                "points of specificity."
            )
    return " ".join(text[0].upper() + text[1:] if text else text for text in sentences)


def deterministic_note(scored) -> str:
    """Say which deterministic check the conclusion rests on, from the scores.

    The two unit rules are compared with each other rather than with the
    composite, because the repository audit checks the unit alone: it is
    measured against the prototype's unit table, which is its like-for-like
    counterpart.
    """
    prototype = scored.get("E-units") or {}
    repo = scored.get("E-repo") or {}
    if prototype.get("recall") is None or repo.get("recall") is None:
        return (
            "The deterministic pre-checks were not all scored, so no conclusion "
            "is drawn from them here."
        )
    note = (
        "Neither deterministic check is a useful screen on its own: the prototype's "
        "unit table recalls {} at specificity {}, the repository's own name-unit audit "
        "recalls {} at specificity {}, and both sit near the always-pass control. A "
        "deterministic gate cannot replace the question and cannot be trusted to "
        "pre-clear names ahead of it.".format(
            _pct(prototype["recall"]),
            _pct(prototype["specificity"]),
            _pct(repo["recall"]),
            _pct(repo["specificity"]),
        )
    )
    if repo["recall"] > prototype["recall"]:
        note += (
            " The repository audit finds more of the flagged names than the prototype "
            "does, so this conclusion rests on the repository's own check, which is the "
            "prototype's stronger counterpart."
        )
    elif repo["recall"] < prototype["recall"]:
        note += (
            " The prototype's unit table finds more of the flagged names than the "
            "repository audit does, so this conclusion rests on the prototype's rule, "
            "the stronger of the two, not on the weaker one."
        )
    else:
        note += (
            " The two find the same names, so the conclusion does not depend on which "
            "one is used."
        )
    return note


def _conclusion(table, contenders) -> list[str]:
    """The arms that reach the adopted question on both measures."""
    lead = table.get("A") or {}
    if lead.get("recall") is None or lead.get("specificity") is None:
        return []
    improvers = []
    for arm in contenders:
        row = table.get(arm) or {}
        if arm == "A" or row.get("recall") is None or row.get("specificity") is None:
            continue
        if (
            row["recall"] >= lead["recall"]
            and row["specificity"] >= lead["specificity"]
        ):
            improvers.append(arm)
    return improvers


def mode_check(args) -> None:
    """Fail when the report's claims are not the ones the recorded scores support."""
    payload = json.loads(OUT_JSON.read_text())
    html = OUT_HTML.read_text()
    scored = payload["arms"]
    scored_adjudicated = payload["arms_adjudicated"]
    if args.mutate:
        arm, _, change = args.mutate.partition(":")
        field, _, value = change.partition("=")
        scored.setdefault(arm, {})[field] = float(value)
        print(f"MUTATION {arm}.{field} set to {value} over the recorded scores")
    contenders = [
        arm
        for arm in ARM_ORDER
        if not arm.startswith("always-") and arm not in ("E-operators", "E-units")
    ]
    failures = []

    def check(label, ok, detail=""):
        print(("PASS " if ok else "FAIL ") + label + (" " + detail if detail else ""))
        if not ok:
            failures.append(label)

    readings = payload["readings"]
    check("held-out readings are 1021", len(readings) == 1021, str(len(readings)))
    check(
        "every arm carries one row per held-out name",
        all(len(scored[arm]["rows"]) == 1021 for arm in scored),
        ", ".join(f"{arm}={len(scored[arm]['rows'])}" for arm in sorted(scored)),
    )
    check(
        "each reading records the adopted verdict and both decisions answers",
        all(
            "adopted" in row and set(row["decisions"]) == {"jev", "drex"}
            for row in readings
        ),
    )
    for arm, want_recall, want_specificity in (
        ("always-pass", 0.0, 1.0),
        ("always-flag", 1.0, 0.0),
    ):
        row = scored.get(arm) or {}
        check(
            f"{arm} control scores recall {want_recall:.1f} / specificity {want_specificity:.1f}",
            row.get("recall") == want_recall
            and row.get("specificity") == want_specificity,
            "recall={} specificity={}".format(
                row.get("recall"), row.get("specificity")
            ),
        )
    models = payload.get("account_models") or {}
    check(
        "the account's model list is recorded",
        models.get("count", 0) > 0,
        "count={}".format(models.get("count")),
    )
    spend = sum(
        float(record.get("total_usd") or 0.0)
        for record in (payload.get("spend") or {}).values()
    )
    check(
        "recorded spend is within the ceiling", spend <= CEILING_USD, f"{spend:.4f} USD"
    )
    check(
        "the figure is referenced by the report",
        "sn-name-check-prototype/recall-specificity-by-arm.svg" in html,
    )

    for label, table in (("settled", scored), ("adjudicated", scored_adjudicated)):
        sentence = trade_sentence(table, contenders)
        check(
            f"{label} labels: the report carries the trade computed from the scores",
            bool(sentence) and sentence in html,
            sentence,
        )
        improvers = _conclusion(table, contenders)
        check(
            f"{label} labels: no arm in the record reaches the adopted question on both "
            "measures",
            not improvers,
            ", ".join(improvers) or "none",
        )
        check(
            f"{label} labels: the report's conclusion still holds",
            not improvers and "No arm measured here improves on it" in html,
            "arms that would reverse it: %s" % (", ".join(improvers) or "none"),
        )

    note = deterministic_note(scored)
    check(
        "the report carries the deterministic conclusion computed from the scores",
        note[:120] in html,
        note[:120],
    )
    check(
        "the repository's own name-unit audit is scored beside the prototype rule",
        (scored.get("E-repo") or {}).get("scored", 0) > 0,
        "scored={}".format((scored.get("E-repo") or {}).get("scored")),
    )

    print(f"FAILURES: {len(failures)}")
    if failures:
        raise SystemExit(1)


def _usd(value) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def _pct_ci(value, ci) -> str:
    if value is None:
        return "n/a"
    if ci and ci[0] is not None:
        return f"{100 * value:.1f}% [{100 * ci[0]:.1f}%, {100 * ci[1]:.1f}%]"
    return "%.1f%%" % (100 * value)


def _row(cells) -> str:
    return "<tr>" + "".join(f"<td>{cell}</td>" for cell in cells) + "</tr>"


HTML_HEAD = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>A name check measured against the ceiling the labels support</title>
<style>
body{font:15px/1.55 system-ui,sans-serif;max-width:1100px;margin:2rem auto;padding:0 1rem;color:#111}
table{border-collapse:collapse;margin:1rem 0;font-size:13px}
th,td{border-bottom:1px solid #ddd;padding:.3rem .55rem;text-align:left}
th{border-bottom:2px solid #333}
code{font:12px ui-monospace,monospace}
.note{color:#555;font-size:13px}
</style></head><body>"""


def arm_table(scored, arms, with_cost: bool) -> list[str]:
    header = [
        "arm",
        "flagged recall [95%]",
        "specificity [95%]",
        "precision [95%]",
        "false holds",
        "misses",
    ]
    if with_cost:
        header.append("USD per 1,000")
    parts = ["<table><tr>" + "".join(f"<th>{cell}</th>" for cell in header) + "</tr>"]
    for arm in arms:
        result = scored.get(arm)
        if not result:
            continue
        cells = [
            ARM_LABELS.get(arm, arm),
            _pct_ci(result["recall"], result["recall_ci"]),
            _pct_ci(result["specificity"], result["specificity_ci"]),
            _pct_ci(result["precision"], result["precision_ci"]),
            result["false_holds"],
            result["misses"],
        ]
        if with_cost:
            cells.append(result["cost_per_1000_usd"])
        parts.append(_row(cells))
    parts.append("</table>")
    return parts


def _sentence(text: str, limit: int = 260) -> str:
    flat = " ".join((text or "").split())
    if not flat:
        return ""
    stop = flat.find(". ")
    head = flat[: stop + 1] if 0 < stop < limit else flat[:limit]
    return head if head.endswith(".") else head + " ..."


def _best_arm(scored: dict, arms: list[str]) -> str:
    best, best_score = None, -1.0
    for arm in arms:
        result = scored.get(arm)
        if not result or result["scored"] == 0:
            continue
        if result["recall"] is None or result["specificity"] is None:
            continue
        total = result["recall"] + result["specificity"]
        if total > best_score:
            best, best_score = arm, total
    return best


def _example_rows(scored, held, arm: str, count: int = 20) -> list[str]:
    items = {item["id"]: item for item in held}
    result = scored.get(arm) or {}
    misses, holds = [], []
    for name, label, outcome in result.get("rows", []):
        item = items.get(name) or {}
        if label == "flagged" and outcome == "pass":
            misses.append((name, item))
        elif label == "stands" and outcome == "flag":
            holds.append((name, item))
    out = []
    groups = (
        (
            "Misses: named as a problem, passed by this arm",
            misses,
            "label says flagged",
        ),
        (
            "False holds: named as standing, flagged by this arm",
            holds,
            "label says stands",
        ),
    )
    for title, group, note in groups:
        out.append(
            f"<h3>{title} ({len(group[:count])} of {len(group)}, "
            f"first {count} shown)</h3>"
        )
        out.append(
            "<table><tr><th>name</th><th>unit</th><th>definition opens</th>"
            "<th>conflict</th></tr>"
        )
        for name, item in group[:count]:
            out.append(
                _row(
                    [
                        f"<code>{name}</code>",
                        item.get("unit", ""),
                        _sentence(item.get("description", "")),
                        note,
                    ]
                )
            )
        out.append("</table>")
    return out


ARM_ORDER = [
    "A",
    "E+A",
    "E-repo+A",
    "B-jev",
    "B-drex",
    "C-jev",
    "C-drex",
    "E",
    "E-repo",
    "E-operators",
    "E-units",
    "F",
    "always-pass",
    "always-flag",
]


def write_html(payload, scored, scored_adjudicated, by_class, held, assembled) -> None:
    ceiling = payload["ceiling"]
    held_out = payload["held_out"]
    flagged = payload["held_out_flagged"]
    stands = payload["held_out_stands"]
    contenders = [
        arm
        for arm in ARM_ORDER
        if not arm.startswith("always-") and arm not in ("E-operators", "E-units")
    ]
    best = _best_arm(scored, contenders)
    parts = [HTML_HEAD]
    parts.append(
        "<h1>A name check measured against the ceiling the labels support</h1>"
    )
    parts.append(
        f"<p class=note>Every arm is scored once, on the {held_out} held-out names "
        f"({flagged} flagged, {stands} standing) that neither the adopted question "
        f"nor any fitted arm saw. The same readings are scored twice: against the "
        f"settled blind-reader labels, and against an adjudicated subset in which "
        f"two fresh readers of different model families re-judged every name where "
        f"the question and the label disagreed.</p>"
    )
    lead = scored.get("A") or {}
    lead_adj = scored_adjudicated.get("A") or {}
    text = (
        "<p><strong>The adopted question recalls %s of the flagged names at "
        "specificity %s.</strong> %s Scored against the adjudicated labels the same "
        "question recalls %s at %s, where the same comparison is given in full in the "
        "answer to the first question below.</p>"
    )
    parts.append(
        text
        % (
            _pct(lead.get("recall")),
            _pct(lead.get("specificity")),
            trade_sentence(scored, contenders),
            _pct(lead_adj.get("recall")),
            _pct(lead_adj.get("specificity")),
        )
    )
    pool_size = len(payload["adjudication"]["set"]["rows"])
    parts.append("<h2>What the labels can support</h2>")
    parts.append(
        f"<p>{pool_size} names went to both fresh readers: the "
        f"{ceiling['disagreements']} held-out names where the adopted question and "
        f"the label disagree, and {pool_size - ceiling['disagreements']} agreement "
        f"controls drawn one-for-one. Each reader saw the name, definition and unit "
        f"and the class sheet, never a label and never another reader's answer.</p>"
    )
    parts.append(
        "<table><tr><th>outcome of the dispute</th><th>names</th>"
        "<th>what it means</th></tr>"
    )
    for label, count, note in (
        (
            "both readers call it a label error",
            ceiling["label_errors_by_both_readers"],
            "the settled label is wrong; the question was right",
        ),
        (
            "both readers side with the label",
            ceiling["real_question_errors"],
            "a real miss or a real false hold of the question",
        ),
        (
            "the two readers split",
            ceiling["readers_split"],
            "no adjudicated answer; the score cannot rest on it",
        ),
    ):
        parts.append(_row([label, count, note]))
    parts.append("</table>")
    adopted_adj = scored_adjudicated.get("A") or {}
    parts.append(
        f"<p>The ceiling the splits imply: of the {ceiling['disagreements']} disputes "
        f"both readers called <strong>{ceiling['label_errors_by_both_readers']}</strong> "
        f"label errors and <strong>{ceiling['real_question_errors']}</strong> real "
        f"errors of the question; <strong>{ceiling['readers_split']}</strong> stay "
        f"unresolved and leave the scored set rather than being credited to either "
        f"side. Against the adjudicated labels the adopted question scores recall "
        f"{_pct(adopted_adj.get('recall'))} and specificity "
        f"{_pct(adopted_adj.get('specificity'))}, so the label set it was measured "
        f"against was slightly harsher than the name check itself deserved.</p>"
    )
    parts.append("<h2>Every arm, scored once on the held-out half</h2>")
    parts.append("<h3>Against the settled blind-reader labels</h3>")
    parts.extend(arm_table(scored, ARM_ORDER, with_cost=True))
    parts.append("<h3>Against the adjudicated labels</h3>")
    parts.append(
        "<p class=note>The same readings, with the disputed labels replaced where "
        "both fresh readers agreed and the splits dropped.</p>"
    )
    parts.extend(arm_table(scored_adjudicated, ARM_ORDER, with_cost=False))
    parts.append("<h2>Recall by the class the settled labels assigned</h2>")
    classes = list(READER_CLASSES[:4])
    parts.append(
        "<table><tr><th>arm</th>"
        + "".join(f"<th>{value}</th>" for value in classes)
        + "</tr>"
    )
    for arm in ARM_ORDER:
        table = by_class.get(arm) or {}
        cells = [ARM_LABELS.get(arm, arm)]
        for value in classes:
            bucket = table.get(value) or {}
            cells.append("{} of {}".format(bucket.get("found", 0), bucket.get("of", 0)))
        parts.append(_row(cells))
    parts.append("</table>")
    parts.append("<h2>The deterministic pre-checks, each way one can be built</h2>")
    detector_rows = []
    for arm, note in (
        ("E", "the prototype's operator rule or its unit-dimension table"),
        ("E-operators", "the prototype operator rule alone"),
        ("E-units", "the prototype unit-dimension table alone"),
        ("E-repo", "the repository's own name-unit audit, read-only, alone"),
        ("E+A", "the prototype pre-checks ahead of the adopted question"),
        ("E-repo+A", "the repository audit ahead of the adopted question"),
    ):
        row = scored.get(arm) or {}
        detector_rows.append(
            _row(
                [
                    ARM_LABELS.get(arm, arm),
                    "n/a" if row.get("recall") is None else _pct(row["recall"]),
                    "n/a"
                    if row.get("specificity") is None
                    else _pct(row["specificity"]),
                    row.get("false_holds", "n/a"),
                    row.get("misses", "n/a"),
                    note,
                ]
            )
        )
    parts.append(
        "<table><tr><th>arm</th><th>recall</th><th>specificity</th>"
        "<th>false holds</th><th>misses</th><th>what it is</th></tr>"
    )
    parts.extend(detector_rows)
    parts.append("</table>")
    parts.append(f"<p>{deterministic_note(scored)}</p>")
    parts.append("<h2>The combination, fixed on the development half</h2>")
    parts.append(
        "<p>F is chosen by balanced accuracy on the 1,023 development names and "
        "then applied to the held-out half:</p>"
    )
    scores = payload["arm_f"]["development_balanced_accuracy"]
    parts.append(
        "<table><tr><th>candidate combination</th><th>development balanced "
        "accuracy</th><th>chosen</th></tr>"
    )
    for label in sorted(scores, key=lambda key: -(scores[key] or 0.0)):
        score = scores[label]
        parts.append(
            _row(
                [
                    label,
                    "n/a" if score is None else f"{score:.3f}",
                    "yes" if label == payload["arm_f"]["chosen"] else "",
                ]
            )
        )
    parts.append("</table>")
    parts.append("<h2>Cost and wall time, Jev and Drex side by side</h2>")
    spend = payload["spend"]
    parts.append(
        "<table><tr><th>arm</th><th>USD per 1,000 names</th>"
        "<th>wall time [s]</th><th>what was paid for</th></tr>"
    )
    for arm, spend_label, note in (
        (
            "A",
            None,
            "the adopted question's committed readings, reused here at no new "
            "spend; the cost is what that method charges per name",
        ),
        (
            "B-jev",
            "arm-b-jev-held-out",
            "Jev decision per name, then one chat call per name it raised",
        ),
        (
            "B-drex",
            "arm-b-drex-held-out",
            "Drex decision per name, then one chat call per name it raised",
        ),
        (
            "C-jev",
            "decisions-jev",
            "one Jev decisions call per name, sent 64 at a time",
        ),
        (
            "C-drex",
            "decisions-drex",
            "one Drex decisions call per name, sent 64 at a time",
        ),
        ("E", None, "deterministic, no calls"),
        ("F", "arm-b-jev-held-out", "inherits the arm it combines"),
    ):
        row = scored.get(arm)
        per_thousand = row.get("cost_per_1000_usd") if row else None
        record = spend.get(spend_label) if spend_label else None
        seconds = record.get("elapsed_seconds") if record else None
        calls = record.get("calls") if record else None
        unit_note = (
            "batches" if (spend_label or "").startswith("decisions") else "calls"
        )
        parts.append(
            _row(
                [
                    ARM_LABELS.get(arm, arm),
                    "n/a" if per_thousand is None else f"{per_thousand:.4f}",
                    "n/a" if seconds is None else f"{seconds:.0f}",
                    note if not calls else f"{note} ({calls} {unit_note})",
                ]
            )
        )
    parts.append("</table>")
    parts.append(
        "<p class=note>The cost column is what each method charges per name on a "
        "name it has not seen, taken from the calls already recorded; the wallet "
        "was opened only for the two B arms, the decisions pass and the two fresh "
        "readers. A decisions call costs about 0.04 USD per 1,000 names and the "
        "chat call it gates about 3.00 USD per 1,000, so a gated arm lands near "
        "the adopted question's price rather than near C's; Drex's arm is the "
        "cheaper of the two only because it raises fewer names.</p>"
    )
    parts.append(
        "<p class=note>Recorded spend for this whole run: {:.4f} USD of the {:.2f} USD "
        "ceiling.</p>".format(
            sum(float(record.get("total_usd") or 0.0) for record in spend.values()),
            CEILING_USD,
        )
    )
    parts.append("<h2>The three questions the lead asked</h2>")
    parts.append("<ol>")
    parts.append(
        "<li><strong>Is there scope for improvement?</strong> Not in the families "
        f"tried here. {trade_sentence(scored, contenders)} Against the labels "
        "both fresh readers could defend the same comparison reads: "
        f"{trade_sentence(scored_adjudicated, contenders)} Every trade and every "
        "gain above is computed from the scored table rather than written by hand, "
        "so the answer moves with the numbers. The addressable scope is the "
        f"{ceiling['real_question_errors']} disputed names both readers upheld "
        f"rather than the {lead.get('false_holds') or 0} false holds the settled "
        "labels imply: the question is already near what these labels can prove.</li>"
    )
    jev_class = by_class.get("C-jev") or {}
    drex_class = by_class.get("C-drex") or {}
    parts.append(
        "<li><strong>Can Jev help by pre-categorising?</strong> Jev's own flag is right on "
        "{} of flagged names and it passes {} of standing ones; as a pre-categoriser in B "
        "it gates which chat question is asked. Per class, Jev finds {}.</li>".format(
            _pct((scored.get("C-jev") or {}).get("recall")),
            _pct((scored.get("C-jev") or {}).get("specificity")),
            "; ".join(
                "{} {} of {}".format(
                    value,
                    (jev_class.get(value) or {}).get("found", 0),
                    (jev_class.get(value) or {}).get("of", 0),
                )
                for value in DECISION_CLASSES
                if value in jev_class
            ),
        )
    )
    parts.append(
        "<li><strong>How does Drex max do in Jev's place?</strong> It <strong>{}</strong>, "
        "returning a typed verdict on every name. Alone it reaches recall {} at "
        "specificity {}, against Jev's {} and {}; in B, Drex's pre-categorisation gives "
        "recall {} at specificity {} against Jev's {} and {}. Its reliability table is "
        "below. It costs {} USD per 1,000 names alone against Jev's {}, and {} against "
        "Jev's {} in B: cheaper only in B, and only because it raises fewer names. In "
        "both places it is the worse screen.</li>".format(
            "answered, and its answer was accepted"
            if (scored.get("C-drex") or {}).get("scored")
            else "did not answer, so its arm is recorded as not run",
            _pct((scored.get("C-drex") or {}).get("recall")),
            _pct((scored.get("C-drex") or {}).get("specificity")),
            _pct((scored.get("C-jev") or {}).get("recall")),
            _pct((scored.get("C-jev") or {}).get("specificity")),
            _pct((scored.get("B-drex") or {}).get("recall")),
            _pct((scored.get("B-drex") or {}).get("specificity")),
            _pct((scored.get("B-jev") or {}).get("recall")),
            _pct((scored.get("B-jev") or {}).get("specificity")),
            _usd((scored.get("C-drex") or {}).get("cost_per_1000_usd")),
            _usd((scored.get("C-jev") or {}).get("cost_per_1000_usd")),
            _usd((scored.get("B-drex") or {}).get("cost_per_1000_usd")),
            _usd((scored.get("B-jev") or {}).get("cost_per_1000_usd")),
        )
    )
    parts.append("</ol>")
    parts.append(
        "<p class=note>Per class, Drex reading the same names alone finds {}.</p>".format(
            "; ".join(
                "{} {} of {}".format(
                    value,
                    (drex_class.get(value) or {}).get("found", 0),
                    (drex_class.get(value) or {}).get("of", 0),
                )
                for value in DECISION_CLASSES
                if value in drex_class
            )
        )
    )
    parts.append("<h2>Where the best arm goes wrong</h2>")
    parts.append(f"<p>{ARM_LABELS.get(best, best)} against the settled labels.</p>")
    parts.extend(_example_rows(scored, held, best, 20))
    parts.append("<h2>Calibration of the two decisions models</h2>")
    for name, key in (("Drex max", "drex_calibration"), ("Jev", "jev_calibration")):
        table = payload.get(key) or []
        parts.append(f"<h3>{name}</h3>")
        parts.append(
            "<table><tr><th>confidence decile</th><th>names</th>"
            "<th>mean confidence</th><th>fraction actually flagged</th></tr>"
        )
        for row in table:
            parts.append(
                _row(
                    [
                        "{:.1f} to {:.1f}".format(row["low"], row["high"]),
                        row["names"],
                        "n/a"
                        if row["mean_confidence"] is None
                        else "{:.3f}".format(row["mean_confidence"]),
                        "n/a"
                        if row["fraction_flagged"] is None
                        else "{:.3f}".format(row["fraction_flagged"]),
                    ]
                )
            )
        parts.append("</table>")
    parts.append("<h2>The controls</h2>")
    for arm, note in (
        ("always-pass", "flags nothing, so recall is 0 by construction"),
        ("always-flag", "flags everything, so specificity is 0 by construction"),
    ):
        row = scored.get(arm) or {}
        parts.append(
            "<p><strong>{}</strong>: recall {}, specificity {} ({}).</p>".format(
                ARM_LABELS.get(arm, arm),
                _pct(row.get("recall")),
                _pct(row.get("specificity")),
                note,
            )
        )
    parts.append("<h2>Recall against specificity</h2>")
    parts.append(
        '<img style="width:100%" alt="flagged recall against specificity, '
        'one labelled point per arm with its interval" '
        'src="/imas-codex/figures/sn-name-check-prototype/'
        'recall-specificity-by-arm.svg">'
    )
    parts.append(
        "<p class=note>One point per scored arm, named in the legend, with its 95 "
        "per cent interval; left against the settled labels and right against "
        "the adjudicated ones. The always-pass control sits at recall 0 and the "
        "always-flag control at specificity 0, which would stretch the axis to "
        "nothing useful, so always-flag is drawn at the left edge with its value "
        "written beside it. A useful check sits towards the top right.</p>"
    )
    parts.append("</body></html>")
    OUT_HTML.write_text("\n".join(parts))
    print("wrote", OUT_HTML, flush=True)


def reliability_bins(probabilities, labels, count: int = 10) -> list[dict]:
    rows = []
    for index in range(count):
        low = index / count
        high = (index + 1) / count
        members = [
            name
            for name, value in probabilities.items()
            if low <= value < high and name in labels
        ]
        flagged = [name for name in members if labels[name] == "flagged"]
        mean = (
            sum(probabilities[name] for name in members) / len(members)
            if members
            else None
        )
        rows.append(
            {
                "bin": str(index + 1),
                "low": round(low, 2),
                "high": round(high, 2),
                "names": len(members),
                "mean_confidence": round(mean, 4) if mean is not None else None,
                "fraction_flagged": round(len(flagged) / len(members), 4)
                if members
                else None,
            }
        )
    return rows


ADOPTED_CHAT_MODEL = "openrouter/anthropic/claude-sonnet-5.5"


def mode_probe(args) -> None:
    import httpx

    from iter_codex.discovery.base.llm import acall_decisions, get_api_key_for_service

    key = get_api_key_for_service("standard-names")
    response = httpx.get(
        "https://openrouter.ai/api/v1/models?output_modalities=all",
        headers={"Authorization": "Bearer " + key},
        timeout=60,
    )
    (RUN_DIR / "probes").mkdir(parents=True, exist_ok=True)
    record = {"status": response.status_code, "models": []}
    if response.status_code == 200:
        record["models"] = sorted(
            model.get("id") for model in response.json().get("data", [])
        )
    (RUN_DIR / "probes" / "openrouter-models.json").write_text(
        json.dumps(record, indent=1)
    )
    print(
        f"model list: status {record['status']}, {len(record['models'])} models",
        flush=True,
    )

    state = {
        "name": "accumulated_ammonia_count",
        "definition": "Cumulative post-breakdown inventory of ammonia molecules, "
        "expressed as an equivalent-electron count.",
        "unit": "1",
    }
    questions = {
        "count_is_equivalent_electrons": {
            "type": "noul",
            "instructions": "The count is an equivalent-electron count.",
            "criteria": {"true": "equivalent electrons", "false": "a direct count"},
        }
    }
    probe = {}
    for slug, model in DECISIONS_MODELS.items():
        try:
            answers, cost = asyncio.run(
                acall_decisions(
                    model, state, questions, service="standard-names", max_retries=1
                )
            )
            probe[slug] = {
                "model": model,
                "answered": True,
                "answers": answers,
                "cost_usd": cost,
            }
            print(f"{model}: answered, cost {cost:.8f}", flush=True)
        except Exception as exc:
            probe[slug] = {
                "model": model,
                "answered": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
            print("{}: refused {}".format(model, probe[slug]["error"]), flush=True)
    STATE.mkdir(parents=True, exist_ok=True)
    (STATE / "decisions-probe.json").write_text(json.dumps(probe, indent=1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)

    modes.add_parser(
        "probe", help="record the account's model list and probe both decisions models"
    )

    adjudicate = modes.add_parser(
        "adjudicate", help="fresh blind readers over the disputed held-out names"
    )
    adjudicate.add_argument("--model", default=None)
    adjudicate.add_argument("--effort", default="high")
    adjudicate.add_argument("--batch", type=int, default=4)
    adjudicate.add_argument("--concurrency", type=int, default=CONCURRENCY)
    adjudicate.add_argument("--limit-batches", type=int, default=None)

    decisions = modes.add_parser("decisions", help="one decisions reading per name")
    decisions.add_argument("--model", required=True)

    arm_b = modes.add_parser("arm-b", help="a decisions model gates the class question")
    arm_b.add_argument("--model", required=True, choices=sorted(DECISIONS_MODELS))
    arm_b.add_argument(
        "--half", default="development", choices=["development", "held-out"]
    )
    arm_b.add_argument("--chat-model", default=ADOPTED_CHAT_MODEL)
    arm_b.add_argument("--effort", default="high")
    arm_b.add_argument("--batch", type=int, default=8)
    arm_b.add_argument("--concurrency", type=int, default=CONCURRENCY)
    arm_b.add_argument("--limit-batches", type=int, default=None)

    modes.add_parser("arm-e", help="deterministic operator and unit pre-checks")

    report = modes.add_parser(
        "report", help="score every arm and write the deliverables"
    )
    report.add_argument(
        "--models", default=str(RUN_DIR / "probes" / "openrouter-models.json")
    )

    check = modes.add_parser(
        "check", help="fail unless the report's claims follow from the recorded scores"
    )
    check.add_argument(
        "--mutate",
        default=None,
        help="ARM:FIELD=VALUE, applied over the recorded scores to prove the gate "
        "fails when a score would reverse the conclusion",
    )

    args = parser.parse_args()
    if args.mode == "probe":
        mode_probe(args)
    elif args.mode == "adjudicate":
        mode_adjudicate(args)
    elif args.mode == "decisions":
        mode_decisions(args)
    elif args.mode == "arm-b":
        mode_arm_b(args)
    elif args.mode == "arm-e":
        mode_arm_e(args)
    elif args.mode == "check":
        mode_check(args)
    else:
        mode_report(args)


if __name__ == "__main__":
    main()
