"""Adjudicate the settled name readings with a third blind reader.

Both fresh readers rejected 53 labels and upheld 68 of them; on 40 names they
split. This script takes the 40 undecided names to a third reader of a different
model family, drawn blind among 40 names the two fresh readers agreed on, and
writes the whole 2,044-name set with its original and its adjudicated verdict.

The third reading is the only new model call. Everything else reads the
committed readings of the two fresh readers and the adopted question.

Modes:
  third-reader  draw the 80-name reading set and call the third reader
  build         write reader-verdicts-adjudicated.json
  report        write sdbc-adjudicated-labels.html
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import hashlib
import html
import importlib.util
import json
import os
import re
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORKTREE = HERE.parents[3]
GRAPHLABELS = WORKTREE / "docs/evidence/fragments/sn-name-definition-graph-reading"
PROTOTYPE_JSON = HERE / "name-check-prototype.json"
FRESH_CLASSES = GRAPHLABELS / "fresh-reader-classes.json"
QUESTION_READINGS = GRAPHLABELS / "adopted-question-readings.json"
QUESTION_SPLIT = GRAPHLABELS / "question-split.json"
THIRD_READINGS = GRAPHLABELS / "third-reader-readings.json"
LABELS = GRAPHLABELS / "reader-verdicts.json"
OUT_JSON = GRAPHLABELS / "reader-verdicts-adjudicated.json"
OUT_HTML = HERE / "sdbc-adjudicated-labels.html"

STATE = (
    Path(os.environ["NAME_CHECK_ADJUDICATION_STATE"])
    if os.environ.get("NAME_CHECK_ADJUDICATION_STATE")
    else None
)

THIRD_MODEL = "openrouter/x-ai/grok-4.7"
THIRD_EFFORT = "high"
THIRD_LABEL = "third-reader"
CEILING_USD = 5.0
BATCH_SIZE = 8
CONCURRENCY = 4

REACHED_UNCHANGED = "unchanged"
REACHED_CORRECTED = "corrected: both fresh readers rejected the label"
REACHED_SETTLED = "settled by third reader"
REACHED_CONTESTED = "contested: no majority"

DISPUTE_LABEL_ERROR = "label error"
DISPUTE_UPHELD = "upholds the label"
DISPUTE_SPLIT = "readers split"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def prototype():
    return _load_module("name_check_prototype", HERE / "name-check-prototype.py")


def outcome(verdict: str) -> str:
    return "flagged" if verdict == "flag" else "stands"


def fresh_readings() -> tuple:
    """The two fresh readers' verdicts and classes, and the dispute outcomes."""

    payload = json.loads(PROTOTYPE_JSON.read_text())
    adjudication = payload["adjudication"]
    detail = adjudication["detail"]
    readers = adjudication["readers"]
    models = sorted(readers)
    classes = json.loads(FRESH_CLASSES.read_text())["classes"]
    return models, readers, detail, classes


def load_question_readings() -> dict:
    verdicts = json.loads(QUESTION_READINGS.read_text())["verdicts"]
    return {name: {"verdict": verdict} for name, verdict in verdicts.items()}


def reading_rows(items: list, names: list) -> list:
    by_id = {item["id"]: item for item in items}
    rows = []
    for name in names:
        item = by_id[name]
        rows.append(
            {
                "name": name,
                "id": name,
                "unit": item["unit"],
                "description": item["description"],
                "documentation": item["documentation"],
            }
        )
    return rows


def reading_packet() -> dict:
    return json.loads(THIRD_READINGS.read_text())


def reading_plan(items: list) -> dict:
    """The blind set as it was issued, joined to the text the reader saw."""

    packet = reading_packet()
    order = packet["reading_order"]
    return {
        "split_names": packet["split_names"],
        "control_flag": packet["control_flag"],
        "reading_order": order,
        "rows": reading_rows(items, order),
    }


def third_reading() -> dict:
    return {row["name"]: row for row in reading_packet()["rows"]}


def spend_report() -> dict:
    return reading_packet()["spend"]


def run_batches(
    proto,
    label,
    rows,
    system_prompt,
    model,
    effort,
    batch_size,
    concurrency,
    limit_batch,
):
    """Call the reader batch by batch, recording each name as it answers.

    The prototype's own runner reads its spend ceiling from a default argument,
    so it cannot be tightened from here; this one holds the node's own ceiling.
    """

    done = {
        name: row
        for name, row in proto.load_rows(label).items()
        if row.get("verdict") in ("flag", "pass")
    }
    todo = [row for row in rows if proto.name_of(row) not in done]
    batches = [
        todo[index : index + batch_size] for index in range(0, len(todo), batch_size)
    ]
    if limit_batch is not None:
        batches = batches[:limit_batch]
    print(
        f"{label}: {len(rows)} rows, {len(done)} recorded, {len(todo)} to call "
        f"in {len(batches)} batches at {model}",
        flush=True,
    )
    gate = asyncio.Semaphore(concurrency)
    lock = asyncio.Lock()
    spend = proto.Spend(label, CEILING_USD)
    began = time.time()

    async def one(batch):
        async with gate:
            started = time.time()
            try:
                verdicts, cost = await proto.judge_batch(
                    batch, system_prompt, proto.items_prompt(batch), model, effort
                )
            except Exception as exc:
                print(f"batch failed: {type(exc).__name__}: {exc}", flush=True)
                verdicts, cost = (
                    [
                        {
                            "name": proto.name_of(item),
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
                proto.append_rows(
                    label,
                    [
                        {
                            **verdict,
                            "cost_usd": round(share, 8),
                            "seconds": round(elapsed / max(1, len(verdicts)), 3),
                        }
                        for verdict in verdicts
                    ],
                )
            if cost:
                spend.add(cost)

    async def drive():
        await asyncio.gather(*(one(batch) for batch in batches))

    asyncio.run(drive())
    spend.close(time.time() - began)
    print(
        f"{label.upper()}_SPEND_USD {spend.total:.4f} batches {len(batches)} "
        f"seconds {time.time() - began:.0f}",
        flush=True,
    )


def mode_third_reader(args) -> None:
    if STATE is None:
        raise SystemExit("set NAME_CHECK_ADJUDICATION_STATE to a run state directory")
    proto = prototype()
    proto.STATE = STATE
    STATE.mkdir(parents=True, exist_ok=True)

    items, _ = proto.load_items()
    plan = reading_plan(items)
    print(
        f"reading set: {len(plan['split_names'])} split names, "
        f"{len(plan['control_flag'])} controls, {len(plan['reading_order'])} rows",
        flush=True,
    )
    run_batches(
        proto,
        THIRD_LABEL,
        plan["rows"],
        proto.ADJUDICATION_SYSTEM,
        THIRD_MODEL,
        THIRD_EFFORT,
        BATCH_SIZE,
        CONCURRENCY,
        args.limit_batch,
    )
    rows = third_reading()
    answered = [
        name for name, row in rows.items() if row["verdict"] in ("flag", "pass")
    ]
    print(
        f"third reader: {len(answered)} of {len(plan['reading_order'])} answered",
        flush=True,
    )
    print(f"spend: {spend_report()}", flush=True)


# ---------------------------------------------------------------------- build


def dispute_kind(payload: dict) -> dict:
    kind: dict = {}
    for name, text in payload["adjudication"]["detail"].items():
        if text == DISPUTE_SPLIT:
            kind[name] = DISPUTE_SPLIT
        elif text.startswith("both readers side with the question"):
            kind[name] = DISPUTE_LABEL_ERROR
        else:
            kind[name] = DISPUTE_UPHELD
    return kind


def build_rows() -> dict:
    models, readers, _, classes = fresh_readings()
    payload = json.loads(PROTOTYPE_JSON.read_text())
    kind_of = dispute_kind(payload)
    questions = load_question_readings()
    third = third_reading()
    verdicts = json.loads(LABELS.read_text())

    rows = []
    for row in verdicts["rows"]:
        name = row["name"]
        original = row["verdict"]
        readings: dict = {}
        for model in models:
            if name in readers[model]:
                readings[model] = {
                    "verdict": readers[model][name],
                    "class": classes.get(model, {}).get(name),
                }
        if name in third and third[name]["verdict"] in ("flag", "pass"):
            readings[THIRD_MODEL] = {
                "verdict": third[name]["verdict"],
                "class": third[name].get("reader_class"),
            }

        kind = kind_of.get(name)
        reached = REACHED_UNCHANGED
        adjudicated = original
        if kind == DISPUTE_LABEL_ERROR:
            agreeing = {
                readings[model]["verdict"] for model in models if model in readings
            }
            if len(agreeing) == 1:
                adjudicated = outcome(next(iter(agreeing)))
                reached = REACHED_CORRECTED
            else:
                adjudicated = None
                reached = REACHED_CONTESTED
        elif kind == DISPUTE_SPLIT:
            votes = collections.Counter(
                outcome(entry["verdict"])
                for entry in readings.values()
                if entry["verdict"] in ("flag", "pass")
            )
            if votes and votes.most_common(1)[0][1] >= 2:
                adjudicated = votes.most_common(1)[0][0]
                reached = REACHED_SETTLED
            else:
                adjudicated = None
                reached = REACHED_CONTESTED

        record = {
            "name": name,
            "original_verdict": original,
            "adjudicated_verdict": adjudicated,
            "reached": reached,
            "dispute": kind,
            "readings": readings,
        }
        if name in questions:
            record["question_verdict"] = outcome(questions[name]["verdict"])
        rows.append(record)
    return {
        "models": models,
        "rows": rows,
        "kind_of": kind_of,
        "questions": questions,
        "verdicts": verdicts,
    }


# ------------------------------------------------------------------ summaries


def moved(rows: list) -> tuple:
    flagged_to_stands = sorted(
        row["name"]
        for row in rows
        if row["adjudicated_verdict"] == "stands"
        and row["original_verdict"] == "flagged"
    )
    stands_to_flagged = sorted(
        row["name"]
        for row in rows
        if row["adjudicated_verdict"] == "flagged"
        and row["original_verdict"] == "stands"
    )
    return flagged_to_stands, stands_to_flagged


def deciding_class(row: dict, models: list):
    settled = row["adjudicated_verdict"]
    if settled is None:
        return None
    if row["dispute"] == DISPUTE_SPLIT:
        entry = row["readings"].get(THIRD_MODEL) or {}
        if entry.get("verdict") and outcome(entry["verdict"]) == settled:
            return entry.get("class")
    pool = []
    for model in models:
        entry = row["readings"].get(model) or {}
        if entry.get("verdict") and outcome(entry["verdict"]) == settled:
            if entry.get("class"):
                pool.append(entry["class"])
    if not pool:
        return None
    return collections.Counter(pool).most_common(1)[0][0]


def control_agreement(models: list, readers: dict) -> dict:
    plan = reading_packet()
    third = third_reading()
    read = 0
    agreed = 0
    disagreed = []
    for name in plan["control_flag"]:
        row = third.get(name) or {}
        verdict = row.get("verdict")
        if verdict not in ("flag", "pass"):
            continue
        read += 1
        if verdict == readers[models[0]][name]:
            agreed += 1
        else:
            disagreed.append(name)
    return {
        "names": len(plan["control_flag"]),
        "read": read,
        "agreed": agreed,
        "disagreed": sorted(disagreed),
    }


def opening_sentence(text: str) -> str:
    text = re.sub(r"\s+", " ", (text or "")).strip()
    if not text:
        return ""
    parts = re.split(r"(?<=[.!?])\s+", text, maxsplit=1)
    return parts[0]


RULE_TEXT = (
    "A name both fresh readers rejected takes their verdict. A pair they split on "
    "takes the third reader's verdict when that verdict forms a two-of-three "
    "majority, and stays contested otherwise. Every other name keeps its original "
    "verdict, including the 68 both readers upheld."
)


def mode_check(args) -> None:
    """Assert the written set against the rule and the inputs it was built from."""

    path = Path(args.document) if args.document else OUT_JSON
    document = json.loads(path.read_text())
    rows = document["rows"]
    problems: list = []

    def want(condition: bool, message: str) -> None:
        if not condition:
            problems.append(message)

    names = [row["name"] for row in rows]
    want(len(rows) == 2044, f"row count {len(rows)}")
    want(len(set(names)) == len(names), "duplicate names")
    want(all(row["adjudicated_verdict"] for row in rows), "a null adjudicated verdict")

    disputed = [row for row in rows if row["dispute"]]
    want(len(disputed) == 161, f"disputed names {len(disputed)}")
    models, readers, _, _ = fresh_readings()
    for row in disputed:
        name = row["name"]
        # Every stored reading must be the committed fresh reading for that
        # model, so a flip inside the artefact cannot agree with itself.
        for model in models:
            if name not in readers[model]:
                continue
            want(
                row["readings"].get(model, {}).get("verdict") == readers[model][name],
                f"{name}: stored reading departs from {model}'s committed reading",
            )
        votes = collections.Counter(
            outcome(entry["verdict"])
            for entry in row["readings"].values()
            if entry.get("verdict") in ("flag", "pass")
        )
        if row["dispute"] == DISPUTE_LABEL_ERROR:
            both = {
                outcome(readers[model][name])
                for model in models
                if name in readers[model]
            }
            want(
                len(both) == 1
                and row["reached"] == REACHED_CORRECTED
                and row["adjudicated_verdict"] == next(iter(both)),
                f"{name}: label-error dispute",
            )
        elif row["dispute"] == DISPUTE_SPLIT:
            want(
                votes
                and votes.most_common(1)[0][1] >= 2
                and row["reached"] == REACHED_SETTLED
                and row["adjudicated_verdict"] == votes.most_common(1)[0][0],
                f"{row['name']}: split dispute",
            )
        else:
            want(
                row["adjudicated_verdict"] == row["original_verdict"]
                and row["reached"] == REACHED_UNCHANGED,
                f"{row['name']}: upheld dispute moved",
            )

    recomputed = hashlib.sha256(LABELS.read_bytes()).hexdigest()
    want(document["labels_sha256"] == recomputed, "input label file changed")

    rebuilt = build_rows()["rows"]
    want(
        json.dumps(rebuilt, sort_keys=True) == json.dumps(rows, sort_keys=True),
        "the written set does not match a fresh build from its inputs",
    )
    expected_adjudicated, expected_controls = adjudicated_summary(rows, models)
    want(
        document.get("adjudicated") == expected_adjudicated,
        "the adjudicated summary does not match the rows it summarises",
    )
    want(
        document.get("controls") == expected_controls,
        "the controls block does not match the committed fresh readings",
    )
    spend = document.get("spend_usd")
    if STATE is not None:
        want(spend == spend_report().get("total_usd"), "the spend entry drifted")
    else:
        # The ledger lives in the run state, which a clean checkout does not
        # carry; the offline reading is a bound, and the build records the exact
        # figure from the ledger when it has it.
        want(
            isinstance(spend, int | float) and 0 < spend <= CEILING_USD,
            f"the spend entry is not a positive figure inside the ceiling: {spend}",
        )

    for problem in problems:
        print("FAIL", problem, flush=True)
    if problems:
        raise SystemExit(f"{len(problems)} problems in {path}")
    print(
        f"checked {path}: {len(rows)} rows, {len(disputed)} disputed, rule holds",
        flush=True,
    )


def adjudicated_summary(rows: list, models: list) -> tuple[dict, dict]:
    """The document's adjudicated and controls blocks, from the rows alone.

    One builder for the write and for the check, so a summary block cannot be
    regenerated from a rule the check does not apply.
    """
    reached = collections.Counter(row["reached"] for row in rows)
    flagged_to_stands, stands_to_flagged = moved(rows)
    settled_moved = sum(
        1
        for row in rows
        if row["reached"] == REACHED_SETTLED
        and row["adjudicated_verdict"] != row["original_verdict"]
    )
    read = [row for row in rows if row["readings"]]
    adjudicated = {
        "names": len(rows),
        "disputes": sum(1 for row in rows if row["dispute"]),
        "reached": {
            key: reached.get(key, 0)
            for key in (
                REACHED_UNCHANGED,
                REACHED_CORRECTED,
                REACHED_SETTLED,
                REACHED_CONTESTED,
            )
        },
        "flagged_to_stands": flagged_to_stands,
        "stands_to_flagged": stands_to_flagged,
        "contested": sorted(
            row["name"] for row in rows if row["reached"] == REACHED_CONTESTED
        ),
        "read_again": len(read),
        "never_read_again": len(rows) - len(read),
        "dispute_outcomes": {
            "corrected": reached.get(REACHED_CORRECTED, 0),
            "settled_by_third_reader": reached.get(REACHED_SETTLED, 0),
            "settled_moved": settled_moved,
            "upheld": sum(1 for row in rows if row["dispute"] == DISPUTE_UPHELD),
        },
        "fresh_reader_agreements": sum(
            1 for row in rows if row["readings"] and not row["dispute"]
        ),
    }
    payload = json.loads(PROTOTYPE_JSON.read_text())
    controls = control_agreement(models, payload["adjudication"]["readers"])
    return adjudicated, controls


def mode_build(args) -> None:
    built = build_rows()
    rows = built["rows"]
    models = built["models"]
    adjudicated, controls = adjudicated_summary(rows, models)
    document = {
        "schema": "reader-verdicts-adjudicated",
        "rule": RULE_TEXT,
        "labels_path": LABELS.relative_to(WORKTREE).as_posix(),
        "labels_sha256": hashlib.sha256(LABELS.read_bytes()).hexdigest(),
        "readers": {
            "fresh": models,
            "third": THIRD_MODEL,
            "third_effort": THIRD_EFFORT,
            "third_role": (
                "read the 40 split names blind among 40 agreement controls; only "
                "its reading of the split names enters a verdict"
            ),
        },
        "adjudicated": adjudicated,
        "controls": controls,
        "spend_usd": spend_report().get("total_usd"),
        "rows": rows,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(document, indent=1) + "\n")
    print(f"wrote {OUT_JSON}", flush=True)
    print(f"adjudicated: {adjudicated}", flush=True)
    print(f"controls: {controls}", flush=True)


# --------------------------------------------------------------------- report


def pct(value) -> str:
    return "n/a" if value is None else f"{100 * value:.1f}%"


def interval(values) -> str:
    return f"[{100 * values[0]:.1f}%, {100 * values[1]:.1f}%]"


def direction_counts(rows: list, models: list) -> dict:
    counts: dict = {}
    for title, original, final in (
        ("flagged to stands", "flagged", "stands"),
        ("stands to flagged", "stands", "flagged"),
    ):
        bucket: collections.Counter = collections.Counter()
        for row in rows:
            if (
                row["original_verdict"] == original
                and row["adjudicated_verdict"] == final
            ):
                bucket[deciding_class(row, models)] += 1
        counts[title] = bucket
    return counts


def arm_scores(rows: list) -> tuple:
    """The adopted question scored against both label sets over the names that carry both."""

    proto = prototype()
    decisions: dict = {}
    settled: dict = {}
    adjudicated: dict = {}
    for row in rows:
        question = row.get("question_verdict")
        if not question or not row["adjudicated_verdict"]:
            continue
        decisions[row["name"]] = "flag" if question == "flagged" else "pass"
        settled[row["name"]] = row["original_verdict"]
        adjudicated[row["name"]] = row["adjudicated_verdict"]
    return proto.score_arm(decisions, settled), proto.score_arm(decisions, adjudicated)


def score_row(title: str, measured: dict) -> str:
    return (
        "<tr>"
        f"<td>{title}</td><td>{measured['scored']}</td>"
        f"<td>{pct(measured['recall'])} {interval(measured['recall_ci'])}</td>"
        f"<td>{pct(measured['specificity'])} {interval(measured['specificity_ci'])}</td>"
        f"<td>{pct(measured['precision'])}</td>"
        f"<td>{measured['false_holds']}</td>"
        f"<td>{measured['misses']}</td>"
        "</tr>"
    )


def mode_report(args) -> None:
    document = json.loads(OUT_JSON.read_text())
    rows = document["rows"]
    models = document["readers"]["fresh"]
    adjudicated = document["adjudicated"]
    controls = document["controls"]
    items, _ = prototype().load_items()
    text_of = {
        item["id"]: item["documentation"] or item["description"] for item in items
    }

    counts = direction_counts(rows, models)

    def agreement(row: dict) -> tuple:
        entries = [
            entry
            for entry in row["readings"].values()
            if entry.get("verdict") in ("flag", "pass")
        ]
        matched = sum(
            1
            for entry in entries
            if outcome(entry["verdict"]) == row["adjudicated_verdict"]
        )
        return matched, len(entries)

    corrected = [
        row
        for row in rows
        if row["adjudicated_verdict"]
        and row["adjudicated_verdict"] != row["original_verdict"]
    ]
    ranked = sorted(
        corrected,
        key=lambda row: (
            -(agreement(row)[0] / max(1, agreement(row)[1])),
            -agreement(row)[1],
            row["name"],
        ),
    )
    spotlight = ranked[:25]

    on_settled, on_adjudicated = arm_scores(rows)

    split = json.loads(QUESTION_SPLIT.read_text())

    moved_rows = []
    for title, bucket in counts.items():
        total = sum(bucket.values())
        if not bucket:
            moved_rows.append(
                f"<tr><td>{title}</td><td>none</td><td>0</td><td></td></tr>"
            )
            continue
        for name, count in sorted(
            bucket.items(), key=lambda pair: (-pair[1], str(pair[0]))
        ):
            moved_rows.append(
                f"<tr><td>{title}</td><td><code>{html.escape(str(name))}</code></td>"
                f"<td>{count}</td><td>{100 * count / total:.0f}%</td></tr>"
            )

    spotlight_rows = []
    for row in spotlight:
        matched, taken = agreement(row)
        if row["original_verdict"] == "flagged":
            moved_to = "flagged to stands"
        else:
            moved_to = "stands to flagged"
        name_class = deciding_class(row, models)
        opening = opening_sentence(text_of.get(row["name"], ""))
        spotlight_rows.append(
            "<tr>"
            f"<td><code>{html.escape(row['name'])}</code></td>"
            f"<td>{moved_to}</td>"
            f"<td><code>{html.escape(str(name_class))}</code></td>"
            f"<td>{matched}/{taken}</td>"
            f"<td>{html.escape(opening)}</td>"
            "</tr>"
        )

    reached = adjudicated["reached"]
    disputes = adjudicated["disputes"]
    never_read = adjudicated["never_read_again"]
    outcomes = adjudicated["dispute_outcomes"]
    total = adjudicated["names"]
    flagged_to_stands = len(adjudicated["flagged_to_stands"])
    stands_to_flagged = len(adjudicated["stands_to_flagged"])
    contested = adjudicated["contested"]

    sections = []
    if contested:
        sections.append(
            "<p>Names with no two-of-three majority, excluded from scoring: "
            + ", ".join(f"<code>{html.escape(name)}</code>" for name in contested)
            + ".</p>"
        )
    sections.append(
        "<p class=note>each disputed name was read again by two fresh readers of "
        "different model families, and, where those two split, by a third reader of a "
        "third family. Names the fresh readers agreed on keep their settled label. The "
        "original labels are kept unchanged beside this set.</p>"
    )
    by_reached: dict = {}
    for row in rows:
        if row["original_verdict"] != row["adjudicated_verdict"]:
            key = (row["original_verdict"], row["adjudicated_verdict"], row["reached"])
            by_reached[key] = by_reached.get(key, 0) + 1

    def moved_from(reached_value: str) -> tuple:
        to_stands = by_reached.get(("flagged", "stands", reached_value), 0)
        to_flagged = by_reached.get(("stands", "flagged", reached_value), 0)
        return to_stands, to_flagged

    corrected_to_stands, corrected_to_flagged = moved_from(REACHED_CORRECTED)
    settled_to_stands, settled_to_flagged = moved_from(REACHED_SETTLED)
    settled_confirmed = (
        reached[REACHED_SETTLED] - settled_to_stands - settled_to_flagged
    )

    sections.append(
        f"<p><strong>{flagged_to_stands} labels moved from flagged to stands and "
        f"{stands_to_flagged} from stands to flagged.</strong> Of the "
        f"{reached[REACHED_CORRECTED]} names both fresh readers rejected the label "
        f"on, all moved: {corrected_to_stands} to stands and "
        f"{corrected_to_flagged} to flagged. Of the {reached[REACHED_SETTLED]} names "
        f"the fresh readers split on, the third reader moved "
        f"{settled_to_stands + settled_to_flagged} "
        f"({settled_to_stands} to stands, {settled_to_flagged} to flagged) and "
        f"confirmed the settled label on the other {settled_confirmed}. "
        + (
            "No name stays contested.</p>"
            if not reached[REACHED_CONTESTED]
            else f"{reached[REACHED_CONTESTED]} names stay contested.</p>"
        )
    )
    sections.append(
        f"<p>Of the {disputes} disputed names, {outcomes['corrected']} are corrected on "
        f"the fresh readers' consensus, {outcomes['settled_by_third_reader']} are "
        f"settled by the third reader ({outcomes['settled_moved']} of them moving and "
        f"{outcomes['settled_by_third_reader'] - outcomes['settled_moved']} confirming "
        f"the settled label), and {outcomes['upheld']} keep their original verdict "
        f"because both fresh readers upheld it.</p>"
    )
    sections.append("<h2>Moved, by the class the deciding reading gave</h2>")
    sections.append(
        "<table><tr><th>direction</th><th>class</th><th>names</th><th>share</th></tr>"
        + "".join(moved_rows)
        + "</table>"
    )
    sections.append(
        "<p>Both directions are counted on the adjudicated set: a move to stands "
        "means the readers found no defect where a label had found one, and is "
        "classed <code>agrees</code>; a move to flagged carries the defect class "
        "the deciding reading named.</p>"
    )
    sections.append("<h2>Twenty-five corrected labels to spot-check</h2>")
    sections.append(
        "<p class=note>ranked by the share of the readers who read the name whose "
        "verdict matches the adjudicated verdict, then by how many readers read it; "
        "ties are ordered by name. The definition is the opening sentence of the "
        "text the readers saw.</p>"
    )
    sections.append(
        "<table><tr><th>name</th><th>moved</th><th>class</th><th>readers agreeing</th>"
        "<th>definition, opening sentence</th></tr>"
        + "".join(spotlight_rows)
        + "</table>"
    )
    sections.append("<h2>The adopted question re-scored</h2>")
    sections.append(
        f"<p>The adopted name-against-definition question read {on_adjudicated['scored']} "
        f"of the {total} names: the held-out half of the committed split "
        f"({len(split['held_out'])} names). It has no reading on the remaining "
        f"{len(split['development'])} development-half names, which were used to fit and "
        f"validate the arms and carry no committed reading of the adopted question "
        f"itself. The question's readings are unchanged; only the labels they are "
        f"scored against moved.</p>"
    )
    sections.append(
        "<table><tr><th>scored against</th><th>names</th><th>flagged recall [95%]</th>"
        "<th>specificity [95%]</th><th>precision</th><th>false holds</th><th>misses</th></tr>"
        + score_row("the settled labels", on_settled)
        + score_row("the adjudicated labels", on_adjudicated)
        + "</table>"
    )
    sections.append(
        f"<p>Scoring against the adjudicated set instead of the settled labels changes "
        f"recall by {100 * (on_adjudicated['recall'] - on_settled['recall']):.1f} points and "
        f"specificity by {100 * (on_adjudicated['specificity'] - on_settled['specificity']):.1f}. "
        f"{on_adjudicated['scored']} names carry a reading on both sides and are scored; "
        f"the {total - on_adjudicated['scored']} that do not are the development half "
        + ("." if not contested else "plus any contested name.")
        + "</p>"
    )
    sections.append("<h2>What this set is and is not</h2>")
    sections.append(
        f"<p>Only the {disputes} names the fresh readers disagreed over were re-read "
        f"for a verdict. The {controls['names']} agreement controls were read blind to "
        f"measure the third reader's own agreement where no dispute existed, and "
        f"leave no verdict in the set. <strong>{never_read} names carry no further "
        f"reading at all</strong> (of {total}), so a label that is wrong where the "
        f"fresh readers agreed with it is still wrong here. This set corrects the "
        f"settled labels; it does not replace them with a re-reading of every name.</p>"
    )
    sections.append(
        f"<p>The third reader is <code>{THIRD_MODEL}</code> at {THIRD_EFFORT} reasoning "
        f"effort, a family distinct from both fresh readers (Anthropic and OpenAI) and "
        f"from the label readers (DeepSeek and OpenAI). It read {controls['read']} of "
        f"its {controls['names']} control names and agreed with the earlier readers on "
        f"{controls['agreed']} of them"
        + (
            "; the names it split on are "
            + ", ".join(
                f"<code>{html.escape(name)}</code>" for name in controls["disagreed"]
            )
            + "."
            if controls["disagreed"]
            else "."
        )
        + "</p>"
    )
    sections.append(
        f"<p class=note>recorded spend on the third reading: "
        f"USD {document['spend_usd'] or 0.0:.4f}. The original label file's sha256 is "
        f"<code>{document['labels_sha256'][:16]}</code>.</p>"
    )

    body = "\n".join(sections)
    page = (
        "<!doctype html>\n"
        '<html lang="en"><head><meta charset="utf-8">\n'
        "<title>The name readings after adjudication</title>\n"
        "<style>\n"
        "body{font:15px/1.55 system-ui,sans-serif;max-width:1100px;margin:2rem auto;"
        "padding:0 1rem;color:#111}\n"
        "table{border-collapse:collapse;margin:1rem 0;font-size:13px}\n"
        "th,td{border-bottom:1px solid #ddd;padding:.3rem .55rem;text-align:left;"
        "vertical-align:top}\n"
        "th{border-bottom:2px solid #333}\n"
        "code{font:12px ui-monospace,monospace}\n"
        ".note{color:#555;font-size:13px}\n"
        "</style></head><body>\n"
        "<h1>The name readings after adjudication</h1>\n"
        f"{body}\n"
        "</body></html>\n"
    )
    OUT_HTML.write_text(page)
    print(f"wrote {OUT_HTML}", flush=True)
    print(
        f"moved: flagged to stands {flagged_to_stands}, stands to flagged {stands_to_flagged}",
        flush=True,
    )
    print(
        f"scored: settled {on_settled['scored']}, adjudicated {on_adjudicated['scored']}",
        flush=True,
    )
    print(
        f"question recall {on_adjudicated['recall']}, specificity {on_adjudicated['specificity']}",
        flush=True,
    )


# ----------------------------------------------------------------------- main


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    third = subparsers.add_parser("third-reader")
    third.add_argument("--limit-batch", type=int, default=None)
    third.set_defaults(func=mode_third_reader)
    build = subparsers.add_parser("build")
    build.set_defaults(func=mode_build)
    check = subparsers.add_parser("check")
    check.add_argument("--document", default=None)
    check.set_defaults(func=mode_check)
    report = subparsers.add_parser("report")
    report.set_defaults(func=mode_report)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
