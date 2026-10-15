"""Read-only name-judgment measurement with resumable, per-name receipts."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import time
from collections import defaultdict
from pathlib import Path

from imas_codex.discovery.base.evaluation import (
    bootstrap_interval,
    choose_operating_point,
    cost_latency_summary,
    frozen_state_variation,
    load_labelled_fixture,
    option_order_stability,
    paired_auc,
)
from imas_codex.discovery.base.judgment import decide_batch
from imas_codex.graph.client import GraphClient
from imas_codex.settings import get_model
from imas_codex.standard_names.review.checks import load_review_checks
from imas_codex.standard_names.review.jev_review import (
    NAME_AXIS,
    SERVICE,
    name_question_table,
    review_names,
)

HERE = Path(__file__).resolve().parent
FIXTURE = HERE.parent / "sn-judgment-evaluation" / "golden-name-acceptable.json"
RESULTS = HERE / "srtj-name-review-measure-results.json"
RUN = Path(os.environ["RECKON_MEASURE_RUN"])
CONTEXT = RUN / "name-context.json"
RECEIPTS = RUN / "name-decisions.jsonl"
COSTS = RUN / "incumbent-costs.json"
MODEL = get_model("sn-judge")
SPEND_CEILING = 4.0
OPTION_FLIP_LIMIT = 0.05
FROZEN_CHANGE_LIMIT = 0.05
INCUMBENTS = (
    "openrouter/x-ai/grok-4.5",
    "openrouter/openai/gpt-5.6-luna",
    "openrouter/anthropic/claude-sonnet-5",
    "openrouter/minimax/minimax-m3",
    "openrouter/qwen/qwen3.7-max",
    "openrouter/openai/gpt-5.5",
)


def _batches(values, size):
    for start in range(0, len(values), size):
        yield values[start : start + size]


def _save_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)
    )
    temporary.replace(path)


def _load_receipts():
    if not RECEIPTS.exists():
        return {}
    records = {}
    for line in RECEIPTS.read_text().splitlines():
        row = json.loads(line)
        records[(row["mode"], row["name"])] = row
    return records


def _append_receipt(row):
    with RECEIPTS.open("a") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _graph_context():
    fixture = load_labelled_fixture(FIXTURE)
    names = sorted({row["canonical_unit"] for row in fixture})
    graph = GraphClient()
    entries = {}
    query = """
    UNWIND $names AS name
    OPTIONAL MATCH (sn:StandardName {id:name})
    OPTIONAL MATCH (sn)-[:HAS_REVIEW]->(rv:StandardNameReview {review_axis:'name'})
    RETURN name, sn.description AS description, sn.documentation AS documentation,
      sn.kind AS kind, sn.unit AS unit, sn.source_paths AS source_paths,
      sn.physical_base AS physical_base, sn.subject AS subject,
      sn.component AS component, sn.coordinate AS coordinate,
      sn.position AS position, sn.process AS process,
      sn.physics_domain AS physics_domain, sn.links AS links,
      sn.reviewer_score_name AS quorum,
      collect({model:rv.model, score:rv.score, reviewed_at:rv.reviewed_at,
        judge_kind:rv.judge_kind, group:rv.review_group_id}) AS reviews
    """
    for group in _batches(names, 80):
        for row in graph.query(query, names=group):
            entries[row["name"]] = row
    if len(entries) != len(names):
        raise RuntimeError(f"graph returned {len(entries)} of {len(names)} names")

    source_paths = sorted(
        {
            source[3:]
            for row in entries.values()
            for source in row["source_paths"] or []
            if source.startswith("dd:")
        }
    )
    dd = {}
    for group in _batches(source_paths, 80):
        for row in graph.query(
            "MATCH (dd:IMASNode) WHERE dd.id IN $paths "
            "RETURN dd.id AS path, dd.documentation AS documentation, "
            "dd.data_type AS data_type, dd.node_type AS node_type, "
            "dd.cocos_transformation_type AS cocos_label",
            paths=group,
        ):
            dd[row["path"]] = row

    all_names = graph.query(
        "MATCH (sn:StandardName) RETURN sn.id AS id, sn.description AS description, "
        "sn.physical_base AS physical_base LIMIT 6000"
    )
    if not any(row["id"] == "ion_average_temperature" for row in all_names):
        raise RuntimeError("sibling scan missed a known graph name")
    by_base = defaultdict(list)
    for row in all_names:
        if row["physical_base"]:
            by_base[row["physical_base"]].append(row)

    for name, row in entries.items():
        paths = [
            source[3:]
            for source in row["source_paths"] or []
            if source.startswith("dd:")
        ]
        source_docs = [dd[path] for path in paths if path in dd]
        candidates = (
            by_base.get(row["physical_base"], []) if row["physical_base"] else []
        )
        tokens = set(name.split("_"))
        ranked = sorted(
            (candidate for candidate in candidates if candidate["id"] != name),
            key=lambda candidate: (
                -len(tokens & set(candidate["id"].split("_"))),
                candidate["id"],
            ),
        )[:5]
        row["dd_source_docs"] = source_docs[:12]
        row["sibling_names"] = [
            {"id": candidate["id"], "description": candidate["description"]}
            for candidate in ranked
        ]
        row["nearby_existing_names"] = [candidate["id"] for candidate in ranked]
        row["existing_names"] = [candidate["id"] for candidate in ranked]
        row["path"] = paths[0] if paths else None
        row["cocos_label"] = next(
            (source["cocos_label"] for source in source_docs if source["cocos_label"]),
            None,
        )
        row["reviews"] = [
            review
            for review in row["reviews"]
            if review["model"] in INCUMBENTS
            and review["score"] is not None
            and review["judge_kind"] in (None, "incumbent")
        ]
    result = {
        "fixture_rows": len(fixture),
        "canonical_names": len(names),
        "entries": entries,
        "coverage": {
            "description": sum(bool(row["description"]) for row in entries.values()),
            "dd_source_docs": sum(
                bool(row["dd_source_docs"]) for row in entries.values()
            ),
            "siblings": sum(bool(row["sibling_names"]) for row in entries.values()),
            "quorum": sum(row["quorum"] is not None for row in entries.values()),
        },
    }
    _save_json(CONTEXT, result)
    print(json.dumps(result["coverage"], sort_keys=True), flush=True)


def _read_incumbent_costs():
    names = sorted({row["canonical_unit"] for row in load_labelled_fixture(FIXTURE)})
    graph = GraphClient()
    reviews = []
    query = """
    UNWIND $names AS name
    MATCH (sn:StandardName {id:name})-[:HAS_REVIEW]->(rv:StandardNameReview {review_axis:'name'})
    WHERE rv.llm_cost IS NOT NULL
    RETURN name, rv.model AS model, rv.review_group_id AS review_group,
      rv.reviewed_at AS reviewed_at, rv.llm_cost AS cost,
      rv.judge_kind AS judge_kind
    """
    for group in _batches(names, 80):
        reviews.extend(graph.query(query, names=group))
    reviews = [
        {**row, "reviewed_at": str(row["reviewed_at"] or "")}
        for row in reviews
        if row["model"] in INCUMBENTS and row["judge_kind"] in (None, "incumbent")
    ]
    if not reviews:
        raise RuntimeError("incumbent cost query found no known reviewer model")
    _save_json(COSTS, reviews)
    print(f"cost receipts {len(reviews)}", flush=True)


def _item(row):
    return {
        "id": row["name"],
        **{
            key: value for key, value in row.items() if key not in {"reviews", "quorum"}
        },
    }


async def _one(name, mode, context, checks, questions, receipts):
    entry = context["entries"][name]
    start = time.monotonic()
    if mode == "base":
        result = await review_names(
            [_item(entry)], checks=checks, model=MODEL, concurrency=1
        )
        record = result.items[0]
        answers = (
            {
                **{
                    f"dimension:{key}": value
                    for key, value in record["categories"].items()
                },
                **{
                    f"check:{key}": value["answer"]
                    for key, value in record["checks"].items()
                    if "answer" in value
                },
            }
            if record["answered"]
            else None
        )
        state = record["state"]
        cost = result.cost
    else:
        base = receipts[("base", name)]
        state = base["state"]
        table = questions
        if mode == "reverse":
            table = {
                key: {
                    **question,
                    "criteria": dict(reversed(list(question["criteria"].items()))),
                }
                if question["type"] == "choice"
                else question
                for key, question in questions.items()
            }
        outcomes, cost = await decide_batch(
            [state], table, model=MODEL, service=SERVICE, concurrency=1
        )
        answers = outcomes[0][0] if outcomes[0] is not None else None
    return {
        "mode": mode,
        "name": name,
        "answered": answers is not None,
        "answers": answers,
        "state": state if mode == "base" else None,
        "cost_usd": cost,
        "latency_s": time.monotonic() - start,
    }


async def _run(mode, limit, concurrency):
    context = json.loads(CONTEXT.read_text())
    receipts = _load_receipts()
    fixture = load_labelled_fixture(FIXTURE)
    names = sorted({row["canonical_unit"] for row in fixture})
    checks = [check for check in load_review_checks()[0] if check.axis == NAME_AXIS]
    questions = name_question_table(checks)
    if mode != "base":
        fixed = [
            name
            for name in names
            if ("base", name) in receipts
            and receipts[("base", name)]["answered"]
            and context["entries"][name]["description"]
        ]
        names = fixed[: max(100, limit)]
    pending = [name for name in names if (mode, name) not in receipts]
    if limit:
        pending = pending[:limit]
    booked = sum(row["cost_usd"] for row in receipts.values())
    if booked >= SPEND_CEILING:
        raise RuntimeError(f"booked USD {booked:.6f} reaches ceiling {SPEND_CEILING}")
    if receipts and pending:
        average = booked / max(1, len(receipts))
        projection = booked + average * len(pending)
        print(
            f"booked USD {booked:.6f}; projected pass total USD {projection:.6f}",
            flush=True,
        )
        if projection > SPEND_CEILING:
            raise RuntimeError("projected spend crosses USD 4 ceiling")
    semaphore = asyncio.Semaphore(concurrency)

    async def bounded(name):
        async with semaphore:
            return await _one(name, mode, context, checks, questions, receipts)

    tasks = [asyncio.create_task(bounded(name)) for name in pending]
    completed = 0
    for task in asyncio.as_completed(tasks):
        row = await task
        _append_receipt(row)
        booked += row["cost_usd"]
        completed += 1
        print(
            f"{mode} {completed}/{len(pending)} {row['name']} answered={row['answered']} booked_usd={booked:.6f}",
            flush=True,
        )
        if booked >= SPEND_CEILING:
            for remaining in tasks:
                remaining.cancel()
            raise RuntimeError("booked spend reached USD 4 ceiling")


def _score(answers, checks):
    if not answers:
        return None
    dimensions = []
    for dimension in ("grammar", "semantic", "convention", "completeness"):
        answer = answers.get(f"dimension:{dimension}")
        if not answer:
            return None
        first = name_question_table(checks)[f"dimension:{dimension}"]["criteria"]
        accepted = next(iter(first))
        chosen = answer.get("choice")
        probabilities = answer.get("probabilities") or {}
        dimensions.append(
            0.5 * float(chosen == accepted)
            + 0.5 * float(probabilities.get(accepted, 0))
        )
    checks_scored = []
    for check in checks:
        if check.question is None:
            continue
        answer = answers.get(f"check:{check.id}")
        if not answer:
            return None
        if check.question.kind.value == "noul":
            failure = float(answer["noul"])
            passes = failure < 0.5
            checks_scored.append(0.5 * passes + 0.5 * (1 - failure))
        else:
            acceptable = check.question.options[0]
            passes = answer.get("choice") == acceptable
            probability = float((answer.get("probabilities") or {}).get(acceptable, 0))
            checks_scored.append(0.5 * passes + 0.5 * probability)
    return 0.5 * sum(dimensions) / len(dimensions) + 0.5 * sum(checks_scored) / len(
        checks_scored
    )


def _incumbent_scores(entry):
    latest = {}
    for row in entry["reviews"]:
        model = row["model"]
        if model not in latest or str(row["reviewed_at"] or "") > str(
            latest[model]["reviewed_at"] or ""
        ):
            latest[model] = row
    return {model: float(row["score"]) for model, row in latest.items()}


def _rescore():
    fixture = load_labelled_fixture(FIXTURE)
    context = json.loads(CONTEXT.read_text())
    receipts = _load_receipts()
    checks = [check for check in load_review_checks()[0] if check.axis == NAME_AXIS]
    canonical = {}
    for row in fixture:
        name = row["canonical_unit"]
        if name in canonical:
            if (
                canonical[name]["label"] != row["label"]
                or canonical[name]["split"] != row["split"]
            ):
                raise ValueError(f"canonical unit {name} disagrees across fixture rows")
        else:
            canonical[name] = row
    scored = []
    for name, row in canonical.items():
        if not context["entries"][name]["description"]:
            continue
        receipt = receipts.get(("base", name))
        score = _score(receipt["answers"], checks) if receipt else None
        if score is not None:
            scored.append({**row, "name": name, "score": score})
    result = {
        "method": {
            "score": "Half the mean of four dimension acceptance values plus half the mean of answered Jev checks. Each value averages its code-derived accept/reject indicator and acceptance probability; Noul pass probability is 1 minus defect probability, Choice pass probability is the authored acceptable option's probability. Three audit-only checks were not run and are excluded. A missing Jev answer excludes the name.",
            "copy_group": "Canonical units are unique; paired bootstrap resamples their copy_group together.",
            "split": "Golden A's committed calibration/validation assignment is unchanged.",
        },
        "fixture": {"rows": len(fixture), "canonical_names": len(canonical)},
        "coverage": {**context["coverage"], "jev_answered": len(scored)},
        "spend": {
            "booked_usd": sum(row["cost_usd"] for row in receipts.values()),
            "ceiling_usd": SPEND_CEILING,
        },
        "splits": {},
    }
    for split in ("calibration", "validation"):
        split_rows = [row for row in scored if row["split"] == split]
        labels = [row["label"] for row in split_rows]
        scores = [row["score"] for row in split_rows]
        groups = [row["copy_group"] for row in split_rows]
        if len(set(labels)) < 2:
            continue
        result["splits"][split] = {
            "jev": {
                "n": len(split_rows),
                "auc": bootstrap_interval(labels, scores, copy_groups=groups),
            },
            "comparisons": {},
        }
        for model in (*INCUMBENTS, "quorum"):
            paired = []
            for row in split_rows:
                entry = context["entries"][row["name"]]
                value = (
                    entry["quorum"]
                    if model == "quorum"
                    else _incumbent_scores(entry).get(model)
                )
                if value is not None:
                    paired.append((row, float(value)))
            if len({row["label"] for row, _ in paired}) < 2:
                continue
            y = [row["label"] for row, _ in paired]
            a = [row["score"] for row, _ in paired]
            b = [value for _, value in paired]
            copy = [row["copy_group"] for row, _ in paired]
            result["splits"][split]["comparisons"][model] = {
                "paired": paired_auc(y, a, b),
                "difference_interval": bootstrap_interval(
                    y, a, scores_b=b, copy_groups=copy
                ),
                "jev_interval": bootstrap_interval(y, a, copy_groups=copy),
                "incumbent_interval": bootstrap_interval(y, b, copy_groups=copy),
                "n": len(paired),
            }
    if scored:
        operating = choose_operating_point(
            scored,
            cost_false_positive=1,
            cost_false_negative=1,
            report_split="validation",
        )
        result["operating_point"] = operating
    base_calls = [
        {"cost": row["cost_usd"], "latency_s": row["latency_s"]}
        for (mode, _), row in receipts.items()
        if mode == "base" and row["answered"]
    ]
    result["cost_latency_per_name"] = cost_latency_summary(base_calls)
    if COSTS.exists():
        cost_rows = json.loads(COSTS.read_text())
        by_model = defaultdict(list)
        by_group = defaultdict(list)
        for row in cost_rows:
            by_model[row["model"]].append(float(row["cost"]))
            by_group[(row["name"], row["review_group"])].append(row)
        quorum_costs = []
        by_name = defaultdict(list)
        for (name, group), rows in by_group.items():
            if group is not None and len(rows) >= 2:
                by_name[name].append(
                    (
                        max(row["reviewed_at"] for row in rows),
                        sum(float(row["cost"]) for row in rows),
                    )
                )
        for groups in by_name.values():
            quorum_costs.append(max(groups)[1])
        result["incumbent_cost_per_name"] = {
            **{
                model: {
                    "n": len(by_model[model]),
                    "mean_usd": sum(by_model[model]) / len(by_model[model]),
                }
                for model in INCUMBENTS
                if by_model[model]
            },
            "quorum": {
                "n": len(quorum_costs),
                "mean_usd": sum(quorum_costs) / len(quorum_costs),
            }
            if quorum_costs
            else None,
        }
    fixed = sorted(
        {name for mode, name in receipts if mode == "repeat"}
        & {name for mode, name in receipts if mode == "reverse"}
    )
    original_choices = []
    reversed_choices = []
    repeated = []

    def decisions(answers):
        return {
            key: value.get("choice")
            if value["type"] == "choice"
            else float(value["noul"]) >= 0.5
            for key, value in answers.items()
            if value is not None
        }

    for name in fixed:
        base = receipts.get(("base", name))
        repeat = receipts.get(("repeat", name))
        reverse = receipts.get(("reverse", name))
        if (
            not base
            or not repeat
            or not reverse
            or not all(r["answered"] for r in (base, repeat, reverse))
        ):
            continue
        digest = hashlib.sha256(
            json.dumps(base["state"], sort_keys=True).encode()
        ).hexdigest()
        repeated.extend(
            [
                {"digest": digest, "fields": decisions(base["answers"])},
                {"digest": digest, "fields": decisions(repeat["answers"])},
            ]
        )
        for key, question in name_question_table(checks).items():
            if question["type"] == "choice":
                original_choices.append(base["answers"][key]["choice"])
                reversed_choices.append(reverse["answers"][key]["choice"])
    result["stability"] = {
        "names": len(repeated) // 2,
        "option_order": option_order_stability(original_choices, reversed_choices),
        "frozen_state": frozen_state_variation(repeated),
    }
    choice_rate = result["stability"]["option_order"]["flip_rate"]
    frozen_fields = result["stability"]["frozen_state"]["fields"]
    frozen_n = sum(field["states"] for field in frozen_fields.values())
    frozen_changed = sum(field["states_changed"] for field in frozen_fields.values())
    result["stability"]["frozen_decision_change_rate"] = (
        frozen_changed / frozen_n if frozen_n else None
    )
    result["stability"]["bounds"] = {
        "option_flip_rate_max": OPTION_FLIP_LIMIT,
        "frozen_change_rate_max": FROZEN_CHANGE_LIMIT,
    }
    validation = result["splits"].get("validation", {})
    comparisons = validation.get("comparisons", {})
    auc_pass = len(comparisons) == 7 and all(
        arm["difference_interval"]["lower"] > -0.02 for arm in comparisons.values()
    )
    stability_pass = (
        result["stability"]["names"] >= 100
        and choice_rate <= OPTION_FLIP_LIMIT
        and result["stability"]["frozen_decision_change_rate"] is not None
        and result["stability"]["frozen_decision_change_rate"] <= FROZEN_CHANGE_LIMIT
    )
    costs = result.get("incumbent_cost_per_name", {})
    jev_cost = result["cost_latency_per_name"]["mean_cost"]
    cost_pass = len(costs) == 7 and all(
        value is not None and jev_cost < value["mean_usd"] for value in costs.values()
    )
    coverage_pass = len(scored) == len(canonical)
    result["verdict"] = {
        "rule": "A Jev judge replaces an incumbent decision only when all four hold: validation lower 95% bound of paired AUC difference exceeds -0.02 against each model and quorum; operating point chosen on calibration; option-order flip and frozen-state variation reported within prototype bounds; cost per decision lower.",
        "auc_pass": auc_pass,
        "calibration_only_cut": "operating_point" in result,
        "stability_pass": stability_pass,
        "cost_pass": cost_pass,
        "full_golden_coverage": coverage_pass,
        "status": "replace"
        if auc_pass and stability_pass and cost_pass and coverage_pass
        else "shadow",
    }
    _save_json(RESULTS, result)
    print(
        json.dumps(
            {
                "coverage": result["coverage"],
                "spend": result["spend"],
                "stability": result["stability"]["names"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


def _plot():
    import matplotlib.pyplot as plt

    result = json.loads(RESULTS.read_text())
    comparisons = result["splits"]["validation"]["comparisons"]
    names = (*INCUMBENTS, "quorum")
    labels = ("Grok", "Luna", "Sonnet", "MiniMax", "Qwen", "GPT", "Quorum")
    try:
        plt.style.use("data-ink")
    except OSError:
        pass
    figure, axis = plt.subplots(figsize=(14, 9))
    for position, model in enumerate(names):
        arm = comparisons[model]
        incumbent = arm["incumbent_interval"]
        jev = arm["jev_interval"]
        difference = arm["difference_interval"]
        axis.errorbar(
            incumbent["point"],
            position + 0.15,
            xerr=[
                [incumbent["point"] - incumbent["lower"]],
                [incumbent["upper"] - incumbent["point"]],
            ],
            fmt="o",
            color="#666666",
            markersize=7,
            linewidth=2.6,
            capsize=0,
            label="incumbent" if position == 0 else None,
        )
        axis.errorbar(
            jev["point"],
            position - 0.15,
            xerr=[[jev["point"] - jev["lower"]], [jev["upper"] - jev["point"]]],
            fmt="o",
            color="#246c78",
            markersize=8,
            linewidth=3.0,
            capsize=0,
            label="Jev on the same names" if position == 0 else None,
        )
        axis.text(
            0.995,
            position,
            f"Δ {difference['point']:+.3f} [{difference['lower']:+.3f}, {difference['upper']:+.3f}]",
            ha="right",
            va="center",
            fontsize=15,
            color="#333333",
        )
    axis.set_yticks(range(len(labels)), labels)
    axis.invert_yaxis()
    axis.set_xlim(0, 1.02)
    axis.set_xlabel("AUC on validation")
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(False)
    axis.legend(frameon=False, loc="lower left")
    figure.tight_layout()
    target = (
        HERE.parents[2]
        / "figures"
        / "sn-review-through-jev"
        / "srtj-name-review-measure"
        / "paired-auc.svg"
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target, format="svg")
    plt.close(figure)
    print(target, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action",
        choices=("prepare", "costs", "base", "repeat", "reverse", "rescore", "plot"),
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--concurrency", type=int, default=8)
    args = parser.parse_args()
    RUN.mkdir(parents=True, exist_ok=True)
    if args.action == "prepare":
        _graph_context()
    elif args.action == "costs":
        _read_incumbent_costs()
    elif args.action == "rescore":
        _rescore()
    elif args.action == "plot":
        _plot()
    else:
        asyncio.run(_run(args.action, args.limit, args.concurrency))


if __name__ == "__main__":
    main()
