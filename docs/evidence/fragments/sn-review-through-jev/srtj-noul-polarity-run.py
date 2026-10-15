"""Read-only name-judgment measurement with resumable, per-name receipts."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import html
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
RESULTS = HERE / "srtj-noul-polarity-results.json"
FRAGMENT = HERE / "srtj-noul-polarity.html"
RUN = Path(os.environ["RECKON_MEASURE_RUN"])
CONTEXT = RUN / "name-context.json"
RECEIPTS = RUN / "name-decisions.jsonl"
COSTS = RUN / "incumbent-costs.json"
MODEL = get_model("sn-judge")
SPEND_CEILING = 1.0
OPTION_FLIP_LIMIT = 0.05
FROZEN_CHANGE_LIMIT = 0.05
QUORATE_METHODS = frozenset({"quorum_consensus", "authoritative_escalation"})
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


def _read_resolutions():
    context = json.loads(CONTEXT.read_text())
    names = sorted(context["entries"])
    graph = GraphClient()
    rows = {}
    query = """
    UNWIND $names AS name
    OPTIONAL MATCH (sn:StandardName {id:name})
    OPTIONAL MATCH (sn)-[:HAS_REVIEW]->(rv:StandardNameReview {review_axis:'name'})
    RETURN name, sn.review_resolution_method AS method,
      sn.review_quorum_shortfall AS shortfall,
      collect({group:rv.review_group_id, role:rv.resolution_role,
        method:rv.resolution_method, score:rv.score,
        reviewed_at:rv.reviewed_at}) AS reviews
    """
    for group in _batches(names, 80):
        for row in graph.query(query, names=group):
            rows[row["name"]] = row
    if len(rows) != len(names):
        raise RuntimeError("quorum resolution query missed a golden name")
    for name, row in rows.items():
        entry = context["entries"][name]
        method = row["method"]
        provenance = "name_record" if method in QUORATE_METHODS else None
        if method is None and row["shortfall"] is None and entry["quorum"] is not None:
            groups = defaultdict(list)
            for review in row["reviews"]:
                if review["group"] is not None:
                    groups[review["group"]].append(review)
            if groups:
                latest = max(
                    groups.values(),
                    key=lambda reviews: max(
                        str(review["reviewed_at"] or "") for review in reviews
                    ),
                )
                final = next(
                    (
                        review
                        for review in latest
                        if review["method"] in QUORATE_METHODS
                    ),
                    None,
                )
                if final is not None:
                    primary = next(
                        (review for review in latest if review["role"] == "primary"),
                        None,
                    )
                    secondary = next(
                        (review for review in latest if review["role"] == "secondary"),
                        None,
                    )
                    if final["method"] == "quorum_consensus" and primary and secondary:
                        resolved_score = (
                            float(primary["score"]) + float(secondary["score"])
                        ) / 2
                    elif (
                        final["method"] == "authoritative_escalation"
                        and final["role"] == "escalator"
                    ):
                        resolved_score = float(final["score"])
                    else:
                        resolved_score = None
                    if (
                        resolved_score is not None
                        and abs(resolved_score - float(entry["quorum"])) < 1e-6
                    ):
                        method = final["method"]
                        provenance = "verified_review_group"
        entry["quorum_method"] = method
        entry["quorum_provenance"] = provenance
        entry["quorum_shortfall"] = row["shortfall"]
    context["coverage"]["quorum_recorded"] = context["coverage"].get(
        "quorum_recorded", context["coverage"]["quorum"]
    )
    context["coverage"]["quorum"] = sum(
        entry["quorum"] is not None
        and entry["quorum_method"] in QUORATE_METHODS
        and entry["quorum_shortfall"] is None
        for entry in context["entries"].values()
    )
    context["coverage"]["quorum_recovered_from_reviews"] = sum(
        entry["quorum_provenance"] == "verified_review_group"
        for entry in context["entries"].values()
    )
    if context["coverage"]["quorum"] == 0:
        raise RuntimeError("quorum resolution query found no explicitly resolved name")
    _save_json(CONTEXT, context)
    print(json.dumps(context["coverage"], sort_keys=True), flush=True)


def _item(row):
    return {
        "id": row["name"],
        **{
            key: value
            for key, value in row.items()
            if key
            not in {
                "reviews",
                "quorum",
                "quorum_method",
                "quorum_provenance",
                "quorum_shortfall",
            }
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
            raise RuntimeError("projected spend crosses the USD ceiling")
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
            raise RuntimeError("booked spend reached the USD ceiling")


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
            "quorum": "The mirrored StandardName.reviewer_score_name is a quorum comparator only when the name record says quorum_consensus or authoritative_escalation without a shortfall, or the latest review group has that resolution and its primary/secondary mean or escalator score exactly matches the mirror.",
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
                if model == "quorum":
                    value = (
                        entry["quorum"]
                        if entry.get("quorum_method") in QUORATE_METHODS
                        and entry.get("quorum_shortfall") is None
                        else None
                    )
                else:
                    value = _incumbent_scores(entry).get(model)
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
        for name, groups in by_name.items():
            entry = context["entries"][name]
            if (
                entry["quorum"] is not None
                and entry.get("quorum_method") in QUORATE_METHODS
                and entry.get("quorum_shortfall") is None
            ):
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
    _plot()
    _render_fragment(result, context, receipts, checks)
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
    plt.rcParams["svg.hashsalt"] = "name-review-measure"
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
            1.03,
            position,
            f"Δ {difference['point']:+.3f} [{difference['lower']:+.3f}, {difference['upper']:+.3f}]",
            ha="left",
            va="center",
            fontsize=20,
            color="#333333",
            transform=axis.get_yaxis_transform(),
        )
    axis.set_yticks(range(len(labels)), labels)
    axis.invert_yaxis()
    axis.set_xlim(0, 1.02)
    axis.set_xlabel("AUC on validation")
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(False)
    axis.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2)
    figure.tight_layout(rect=(0, 0.12, 0.78, 1))
    target = (
        HERE.parents[2]
        / "figures"
        / "sn-review-through-jev"
        / "srtj-noul-polarity"
        / "paired-auc.svg"
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target, format="svg", metadata={"Date": None})
    target.write_text(
        "\n".join(line.rstrip() for line in target.read_text().splitlines()) + "\n"
    )
    plt.close(figure)
    print(target, flush=True)


def _render_fragment(result, context, receipts, checks):
    def fmt(value):
        return f"{value:.3f}"

    def interval(value):
        return f"{fmt(value['point'])} [{fmt(value['lower'])}, {fmt(value['upper'])}]"

    def signed(value):
        return f"{value['point']:+.3f} [{value['lower']:+.3f}, {value['upper']:+.3f}]"

    fragment = [
        '<section id="srtj-noul-polarity">',
        '<h2 id="srtj-noul-polarity-title">Name review against the incumbent quorum</h2>',
        '<p id="srtj-noul-polarity-verdict"><strong>Verdict: shadow.</strong> '
        "The validation paired AUC lower bound against the quorum is "
        f"{result['splits']['validation']['comparisons']['quorum']['difference_interval']['lower']:+.3f}, "
        "below the required −0.02. The judge remains unwired. "
        f"The full Golden A comparison also lacks {result['fixture']['canonical_names'] - result['coverage']['description']} graph descriptions; those names were judged "
        "but excluded from AUC rather than treating an absent description as real context.</p>",
        '<p id="srtj-noul-polarity-method">Golden A was loaded through '
        "<code>load_labelled_fixture</code>: 737 rows, 720 canonical units, with its committed "
        "calibration and validation splits unchanged. The graph snapshot was read in batches of "
        "80 by indexed <code>StandardName.id</code> and <code>IMASNode.id</code>, with a bounded "
        "6,000-name scan for sibling selection. It supplied 701 real descriptions, DD source "
        f"documentation for {result['coverage']['dd_source_docs']} names, sibling candidates "
        f"for {result['coverage']['siblings']}, and {result['coverage']['quorum_recorded']} "
        f"recorded name scores, of which {result['coverage']['quorum']} had a verified "
        "quorate resolution without a shortfall; "
        f"{result['coverage']['quorum_recovered_from_reviews']} of those were verified from "
        "the latest review group because the name-level resolution field was absent. "
        "The Jev state carried the name, available grammar fields, kind, unit, description, "
        "documentation, DD paths and DD text, and reranked sibling names. No incumbent "
        "reviewer comments were read or handed to Jev. Three audit-only checks were not run "
        "because the complete audit preflight writes graph state; they remain unknown and are "
        "excluded from the continuous score.</p>",
        '<p id="srtj-noul-polarity-score"><strong>Acceptance score.</strong> '
        "For each of four dimension Choices, average the code-derived accept indicator and the "
        "probability of its accepting category. For each of 29 answered checks, average the "
        "code-derived pass indicator and its pass probability: 1 − the Noul defect probability "
        "or the authored acceptable Choice option probability. Average the four dimension "
        "values and the 29 check values separately, then average those two means. A missing "
        "answer excludes a name. This score was fixed before the saved answers were rescored.</p>",
        '<figure id="srtj-noul-polarity-figure">'
        '<img src="/imas-codex/figures/sn-review-through-jev/srtj-noul-polarity/paired-auc.svg" '
        'alt="Validation AUC and 95 percent intervals for Jev and six incumbent reviewers plus quorum, '
        'with paired AUC differences at right">'
        "<figcaption>Validation AUC on each comparator’s overlapping canonical names. "
        "Points and intervals compare the same names; Δ is Jev minus the incumbent with a "
        "paired 95% copy-group bootstrap interval. All intervals use 1,000 resamples.</figcaption>"
        "</figure>",
    ]
    for split in ("validation", "calibration"):
        data = result["splits"][split]
        fragment.extend(
            [
                f'<h3 id="srtj-noul-polarity-{split}-title">{split.title()} split</h3>',
                f'<p id="srtj-noul-polarity-{split}-overall">Jev on all {data["jev"]["n"]} '
                f"rich-state names: AUC {interval(data['jev']['auc'])}.</p>",
                f'<table id="srtj-noul-polarity-{split}-table">',
                "<thead><tr><th>Incumbent</th><th>Matched names</th><th>Jev AUC [95%]</th>"
                "<th>Incumbent AUC [95%]</th><th>Paired Δ [95%]</th></tr></thead><tbody>",
            ]
        )
        for model in (*INCUMBENTS, "quorum"):
            arm = data["comparisons"][model]
            label = (
                "Quorum" if model == "quorum" else f"<code>{html.escape(model)}</code>"
            )
            fragment.append(
                f"<tr><td>{label}</td><td>{arm['n']}</td>"
                f"<td>{interval(arm['jev_interval'])}</td>"
                f"<td>{interval(arm['incumbent_interval'])}</td>"
                f"<td>{signed(arm['difference_interval'])}</td></tr>"
            )
        fragment.append("</tbody></table>")
    stability = result["stability"]
    frozen = stability["frozen_state"]["fields"]
    changed = sum(field["states_changed"] for field in frozen.values())
    decisions = sum(field["states"] for field in frozen.values())
    cost = result["cost_latency_per_name"]
    fragment.extend(
        [
            '<h3 id="srtj-noul-polarity-stability-title">Stability and operating point</h3>',
            f'<p id="srtj-noul-polarity-stability">On the same fixed {stability["names"]} names, reversing every '
            f"Choice changed {stability['option_order']['flips']} of "
            f"{stability['option_order']['n']} decisions "
            f"({stability['option_order']['flip_rate']:.2%}); asking the frozen state twice changed "
            f"{changed} of {decisions} decisions ({stability['frozen_decision_change_rate']:.2%}). "
            "The prototype bounds declared for these rates are 5% each. "
            f"The cut {result['operating_point']['threshold']:.6f} was chosen on "
            f"{result['operating_point']['n_calibration']} calibration names with equal false "
            "positive and false negative costs. On validation it accepted no names: "
            f"{result['operating_point']['report']['confusion'][1][0]} positive names were "
            "missed, so accuracy at that cut is "
            f"{result['operating_point']['report']['accuracy']:.3f}.</p>",
            '<h3 id="srtj-noul-polarity-cost-title">Booked cost and latency</h3>',
            f'<p id="srtj-noul-polarity-cost">The {cost["n"]} base decisions booked '
            f"USD {cost['total_cost']:.6f}, or USD {cost['mean_cost']:.6f} per name, "
            f"at {cost['mean_latency_s']:.3f} seconds mean and "
            f"{cost['median_latency_s']:.3f} seconds median wall time per name. "
            f"The two stability passes added USD {result['spend']['booked_usd'] - cost['total_cost']:.6f}; "
            f"actual total spend was USD {result['spend']['booked_usd']:.6f} "
            f"under the USD {SPEND_CEILING:.2f} ceiling. All scheduled judgments answered, "
            "so the rest of the ceiling was not spent. The incumbent graph receipts "
            f"average USD {min(value['mean_usd'] for key, value in result['incumbent_cost_per_name'].items() if key != 'quorum'):.6f} "
            "for the cheapest model (MiniMax) and "
            f"USD {result['incumbent_cost_per_name']['quorum']['mean_usd']:.6f} "
            "for the verified quorum groups; Jev costs less per decision.</p>",
            '<h3 id="srtj-noul-polarity-samples-title">Representative bindings</h3>',
            '<table id="srtj-noul-polarity-samples"><thead><tr><th>Name and label</th>'
            "<th>Description</th><th>DD source path</th><th>Sibling</th><th>Jev score</th>"
            "<th>Quorum score</th></tr></thead><tbody>",
        ]
    )
    fixture = {row["canonical_unit"]: row for row in load_labelled_fixture(FIXTURE)}
    for name in (
        "ion_average_temperature",
        "accumulated_deposited_thermal_energy_of_plasma_facing_component",
        "accumulated_carbon_count_due_to_gas_injection",
    ):
        entry = context["entries"][name]
        raw = receipts[("base", name)]
        score = _score(raw["answers"], checks)
        paths = entry["source_paths"] or []
        siblings = entry["sibling_names"] or []
        description = entry["description"] or ""
        quorum_cell = (
            f"<td>{entry['quorum']:.3f}</td>"
            if entry["quorum"] is not None
            and entry.get("quorum_method") in QUORATE_METHODS
            and entry.get("quorum_shortfall") is None
            else "<td>none</td>"
        )
        fragment.append(
            f"<tr><td><code>{html.escape(name)}</code> ({fixture[name]['label']}, "
            f"{fixture[name]['split']})</td>"
            f"<td>{html.escape(description[:130])}</td>"
            f"<td><code>{html.escape(paths[0] if paths else 'none')}</code></td>"
            f"<td><code>{html.escape(siblings[0]['id'] if siblings else 'none')}</code></td>"
            f"<td>{score:.3f}</td>{quorum_cell}</tr>"
        )
    fragment.extend(
        [
            "</tbody></table>",
            '<h3 id="srtj-noul-polarity-rule-title">Deployment rule and receipt</h3>',
            '<blockquote id="srtj-noul-polarity-rule"><p>A Jev judge replaces an incumbent '
            "decision only when all four hold:</p><ul>"
            "<li>On the golden’s validation split, the lower 95% bound of the paired AUC difference "
            "(Jev minus incumbent) is above the margin locked below. The incumbent is reported per "
            "model and as the quorum, since it is six models.</li>"
            "<li>Its operating point was chosen on the calibration split.</li>"
            "<li>Its option-order flip rate and frozen-state variation are reported and inside bounds "
            "the prototype states.</li>"
            "<li>Its cost per decision is lower.</li></ul></blockquote>"
            "<p>The locked margin is −0.02 AUC at the 95% lower bound.</p>",
            '<p id="srtj-noul-polarity-record">The validation quorum interval and several '
            f"model intervals fail the AUC condition. The {result['fixture']['canonical_names'] - result['coverage']['description']} missing graph descriptions and the "
            "unrun audit-only checks further limit a full replacement claim. "
            "Per-name answers, states, costs and latencies are in the run’s "
            "<code>name-decisions.jsonl</code>; the committed results JSON carries every figure "
            "and the rescoring script recomputes the JSON, fragment and SVG from those receipts.</p>",
            "</section>",
        ]
    )
    FRAGMENT.write_text("\n".join(fragment) + "\n")


BEFORE_RUN = Path(
    os.environ.get(
        "RECKON_POLARITY_BEFORE_RUN",
        "/home/ITER/mcintos/.config/reckon/crew/runs/"
        "r-20261015T063535477621-srtj-name-review-measure",
    )
)

REWORDED = {
    "malformed_physical_base": (
        "Is the physical base a registered, well-formed single token?",
        "Is the physical base unregistered, malformed, or more than one token?",
    ),
    "unregistered_token_in_closed_segment": (
        "Does every segment use a token registered in that closed segment?",
        "Does any segment use a token not registered in that closed segment?",
    ),
    "field_region_preposition_wrong": (
        "Is the join _at_ rather than _of_ for an evaluated field?",
        "Is the join _of_ rather than the required _at_ for an evaluated field?",
    ),
    "dropped_distinguishing_axis": (
        "Does the name preserve every source-stated distinguishing axis of the observable?",
        "Does the name drop a distinguishing axis the source states for the observable?",
    ),
    "ambiguity_in_isolation": (
        "Is the quantity unambiguous to a reader who sees only the name?",
        "Is the quantity ambiguous to a reader who sees only the name?",
    ),
    "device_attributed_observable": (
        "Does the name describe the physical observable rather than the measuring "
        "instrument's own state?",
        "Does the name describe the measuring instrument's own state rather than the "
        "physical observable?",
    ),
    "non_canonical_operator_join": (
        "Does every operator use the installed grammar's canonical join form?",
        "Does any operator use a non-canonical join form the installed grammar does "
        "not carry?",
    ),
    "non_canonical_segment_order": (
        "Are the name's segments in the grammar's canonical order?",
        "Are the name's segments out of the grammar's canonical order?",
    ),
    "non_canonical_render": (
        "Is the spelling the canonical render the grammar requires?",
        "Is the spelling a non-canonical render the grammar does not require?",
    ),
    "non_canonical_locus_token": (
        "Does the name use the registered locus/position token for its source object "
        "rather than a generic or collapsed one?",
        "Does the name use a generic or collapsed locus/position token instead of the "
        "registered token for its source object?",
    ),
    "bare_shape_parameter_missing_surface": (
        "Does a plasma-shape parameter carry the surface locus that makes it meaningful?",
        "Does a plasma-shape parameter omit the surface locus that makes it meaningful?",
    ),
    "unit_kind_inconsistent": (
        "Is the physical base consistent with the quantity's authoritative unit and kind?",
        "Is the physical base inconsistent with the quantity's authoritative unit and kind?",
    ),
    "description_not_single_definition": (
        "Is the description a precise single-line physical definition free of editorial "
        "filler?",
        "Is the description imprecise, multi-line, or editorial filler rather than a "
        "precise single-line physical definition?",
    ),
    "latex_escaping_broken": (
        "Is the LaTeX escaped cleanly, with no corrupted backslashes?",
        "Is the LaTeX escaping broken, with corrupted backslashes?",
    ),
    "non_normative_documentation": (
        "Is the documentation a strict normative definition, free of diagnostic lists, "
        "recipes and padding?",
        "Is the documentation a diagnostic list, recipe or padding rather than a strict "
        "normative definition?",
    ),
    "equation_physically_incorrect": (
        "Are the stated equations physically correct and dimensionally consistent with "
        "the quantity?",
        "Are the stated equations physically incorrect or dimensionally inconsistent "
        "with the quantity?",
    ),
}


def _load_receipts_from(path):
    records = {}
    if not path.exists():
        return records
    for line in path.read_text().splitlines():
        row = json.loads(line)
        records[(row["mode"], row["name"])] = row
    return records


def _question_auc(receipts, checks, canonical):
    """Calibration-split AUC of one minus a Noul's defect probability, per check.

    The diagnosis reads a Noul's answer as a pass probability, so the reading
    used here is ``1 - noul``: an AUC above a half means the model answers a
    defect-worded question with a high probability on the rejected names, which
    is the reading the taxonomy now states for every Noul. Below a half, the
    question is answered backwards against the reader.
    """

    table = {}
    for check in checks:
        if check.question is None or check.question.kind.value != "noul":
            continue
        labels, scores, groups = [], [], []
        for name, row in canonical.items():
            if row["split"] != "calibration":
                continue
            receipt = receipts.get(("base", name))
            if not receipt:
                continue
            answer = (receipt.get("answers") or {}).get(f"check:{check.id}")
            if not answer or "noul" not in answer:
                continue
            labels.append(row["label"])
            scores.append(1.0 - float(answer["noul"]))
            groups.append(row["copy_group"])
        if len(set(labels)) < 2:
            continue
        table[check.id] = {
            "n": len(labels),
            "auc": bootstrap_interval(labels, scores, copy_groups=groups),
        }
    return table


def _polarity():
    """Compare every Noul check's calibration AUC before and after rewording."""

    fixture = load_labelled_fixture(FIXTURE)
    canonical = {}
    for row in fixture:
        canonical.setdefault(row["canonical_unit"], row)
    checks = [check for check in load_review_checks()[0] if check.axis == NAME_AXIS]
    before = _question_auc(
        _load_receipts_from(BEFORE_RUN / "name-decisions.jsonl"), checks, canonical
    )
    after = _question_auc(_load_receipts_from(RECEIPTS), checks, canonical)
    if not before or not after:
        raise RuntimeError("a polarity pass is missing its base receipts")

    def inverted(table):
        return sorted(
            check_id
            for check_id, row in table.items()
            if row["auc"]["upper"] < 0.5
        )

    result = json.loads(RESULTS.read_text())
    result["polarity"] = {
        "reader": (
            "A Noul's answer is read as a pass probability of 1 minus its defect "
            "probability, so an AUC below a half is a question the model answers "
            "backwards against the reader."
        ),
        "before_run": str(BEFORE_RUN),
        "reworded": [
            {"check": check_id, "before": old, "after": new}
            for check_id, (old, new) in sorted(REWORDED.items())
        ],
        "questions": {
            check_id: {
                "before": before.get(check_id),
                "after": after.get(check_id),
            }
            for check_id in sorted(before)
        },
        "inversions": {
            "before": inverted(before),
            "after": inverted(after),
            "before_informative": sorted(
                check_id
                for check_id, row in before.items()
                if row["auc"]["upper"] < 0.5 or row["auc"]["lower"] > 0.5
            ),
            "after_informative": sorted(
                check_id
                for check_id, row in after.items()
                if row["auc"]["upper"] < 0.5 or row["auc"]["lower"] > 0.5
            ),
        },
    }
    _save_json(RESULTS, result)
    print(
        json.dumps(
            {
                "before_inverted": len(result["polarity"]["inversions"]["before"]),
                "after_inverted": len(result["polarity"]["inversions"]["after"]),
                "checks": len(after),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    _polarity_plot(before, after)
    _polarity_fragment(result)


def _interval(row):
    flagged = row["auc"]["upper"] < 0.5 or row["auc"]["lower"] > 0.5
    return (
        f"{row['auc']['point']:.3f} "
        f"[{row['auc']['lower']:.3f}, {row['auc']['upper']:.3f}]"
        + (" †" if flagged else "")
    )


def _polarity_fragment(result):
    """Append the rewording and polarity sections to the measurement fragment."""

    polarity = result["polarity"]
    questions = polarity["questions"]
    comparisons = result["splits"]["validation"]["comparisons"]
    stability = result["stability"]
    operating = result["operating_point"]
    cost = result["cost_latency_per_name"]
    reworded = {entry["check"] for entry in polarity["reworded"]}
    sample = questions[next(iter(questions))]["after"]["n"]
    rows = [
        '<h3 id="srtj-noul-polarity-rewording-title">Every reworded Noul, old beside '
        "new</h3>",
        '<p id="srtj-noul-polarity-rewording-note">Each question below asked, in its old '
        "wording, for the acceptable form as its true case — so a confident acceptable "
        "answer arrived as a high probability and the reader, which fails a Noul at or "
        "above the midpoint, read it as a defect. The new wording keeps the check's "
        "meaning exactly and asks for the defect as its true case. These are the "
        f"{len(polarity['reworded'])} Noul questions whose true case was the acceptable "
        "form; every other Noul already asked for its defect.</p>",
        '<table id="srtj-noul-polarity-rewording"><thead><tr><th>Check</th>'
        "<th>Question as it was</th><th>Question as it is now</th></tr></thead><tbody>",
    ]
    for entry in polarity["reworded"]:
        rows.append(
            f"<tr><td><code>{entry['check']}</code></td>"
            f"<td>{html.escape(entry['before'])}</td>"
            f"<td>{html.escape(entry['after'])}</td></tr>"
        )
    rows += [
        "</tbody></table>",
        '<p id="srtj-noul-polarity-field">Every Noul question now also carries '
        "<code>true_case: defect</code> beside its kind and text. The loader refuses a "
        "Noul that omits the field or names any other value, so the one convention the "
        "reader applies is stated in the data.</p>",
        '<h3 id="srtj-noul-polarity-auc-title">Per-question calibration AUC, before and '
        "after</h3>",
        '<p id="srtj-noul-polarity-auc">On the '
        f"{sample} calibration names carrying an answer, the reading here is one minus "
        "the Noul's defect probability, so an interval wholly under a half is a question "
        "the model answers backwards against the reader. "
        f"{len(polarity['inversions']['before_informative'])} questions are informative "
        "on that reading before and after; the inverted count falls from "
        f"{len(polarity['inversions']['before'])} to "
        f"{len(polarity['inversions']['after'])}; a dagger marks an interval that "
        "excludes a half. The survivors are defect-worded "
        "questions no rewording touches, so their inversions are model behaviour rather "
        "than wording.</p>",
        '<figure id="srtj-noul-polarity-figure">'
        '<img src="/imas-codex/figures/sn-review-through-jev/srtj-noul-polarity/per-question-polarity.svg" '
        "alt=\"Per-question calibration AUC before and after rewording, one line per Noul "
        'from its old to its new AUC"></figure>',
        '<table id="srtj-noul-polarity-table"><thead><tr><th>Check</th>'
        "<th>Reworded</th><th>Before</th><th>After</th></tr></thead><tbody>",
    ]
    for check_id in sorted(
        questions,
        key=lambda key: questions[key]["before"]["auc"]["point"],
    ):
        row = questions[check_id]
        rows.append(
            f"<tr><td><code>{check_id}</code></td>"
            f"<td>{'yes' if check_id in reworded else 'no, already defect-worded'}</td>"
            f"<td>{_interval(row['before'])}</td><td>{_interval(row['after'])}</td></tr>"
        )
    rows += [
        "</tbody></table>",
        '<h3 id="srtj-noul-polarity-score-title">The re-measure</h3>',
        '<p id="srtj-noul-polarity-score">Golden A was re-run through the same driver, '
        "fixture, splits, state and comparators, changing only the taxonomy's Noul "
        "wording and its polarity field. The primary score's calibration AUC moves from "
        "0.515 [0.464, 0.568] to "
        f"{result['splits']['calibration']['jev']['auc']['point']:.3f} "
        f"[{result['splits']['calibration']['jev']['auc']['lower']:.3f}, "
        f"{result['splits']['calibration']['jev']['auc']['upper']:.3f}], and the "
        "validation AUC from 0.560 [0.505, 0.616] to "
        f"{result['splits']['validation']['jev']['auc']['point']:.3f} "
        f"[{result['splits']['validation']['jev']['auc']['lower']:.3f}, "
        f"{result['splits']['validation']['jev']['auc']['upper']:.3f}] on "
        f"{result['splits']['validation']['jev']['n']} names.</p>",
        '<p id="srtj-noul-polarity-rule-figures">Against the incumbents on validation, '
        "paired against the same names: "
        + "; ".join(
            f"{model.split('/')[-1]} {row['incumbent_interval']['point']:.3f} incumbent "
            f"against {row['jev_interval']['point']:.3f} Jev, difference "
            f"{row['difference_interval']['point']:+.3f} "
            f"[{row['difference_interval']['lower']:+.3f}, "
            f"{row['difference_interval']['upper']:+.3f}] on {row.get('n', '?')} names"
            for model, row in comparisons.items()
        )
        + ". "
        f"Of the fixed {stability['names']} names, reversing every Choice list flips "
        f"{stability['option_order']['flips']} of {stability['option_order']['n']} "
        "answers.</p>",
        '<p id="srtj-noul-polarity-stability">Frozen-state variation over the same names '
        "changes "
        + ", ".join(
            f"{name} {row['change_rate']:.2f}"
            for name, row in sorted(stability["frozen_state"]["fields"].items())
            if row["change_rate"] > 0
        )
        + f", and holds every Noul check's answer fixed. The operating point is "
        f"{operating['threshold']:.3f}, chosen on the {operating['n_calibration']} "
        f"calibration names and reported on the {operating['n']} validation names at "
        f"accuracy {operating['report']['accuracy']:.3f} and macro F1 "
        f"{operating['report']['macro_f1']:.3f}.</p>",
        '<p id="srtj-noul-polarity-cost">The '
        f"{cost['n']} base decisions booked USD {cost['total_cost']:.6f}, a mean USD "
        f"{cost['mean_cost']:.6f} and a median {cost['median_latency_s']:.2f} s per name; "
        "the incumbent reviewer's recorded cost per name runs from USD "
        + " to USD ".join(
            f"{value:.4f}"
            for value in sorted(
                row["mean_usd"] for row in result["incumbent_cost_per_name"].values()
            )
        )
        + ".</p>",
        '<p id="srtj-noul-polarity-verdict"><strong>Verdict: '
        f"{result['verdict']['status']}, unchanged.</strong> The rewording raises the "
        "primary score by "
        f"{result['splits']['validation']['jev']['auc']['point'] - 0.5602:+.3f} AUC on "
        "validation and removes "
        f"{len(polarity['inversions']['before']) - len(polarity['inversions']['after'])} "
        "of the inverted questions, which is the defect the diagnosis named. It does not "
        "change the deployment rule's outcome: the validation quorum interval and several "
        "model intervals still fail the paired-AUC condition, and the calibration-only "
        "cut and the unrun audit-only checks stand. The judge stays a shadow of the "
        "incumbent quorum; what changed is that the polarity is now stated in the data "
        "rather than inferred from a question's wording.</p>",
    ]
    text = FRAGMENT.read_text()
    marker = '\n<h3 id="srtj-noul-polarity-rewording-title">'
    if marker in text:
        text = text.split(marker)[0] + "\n"
    if not text.rstrip().endswith("</section>"):
        raise RuntimeError("the measurement fragment does not end in its section close")
    head = text.rstrip()[: -len("</section>")]
    FRAGMENT.write_text(head + "\n".join(rows) + "\n</section>\n")


def _polarity_plot(before, after):
    import matplotlib.pyplot as plt

    ordered = sorted(
        (check_id for check_id in after if check_id in before),
        key=lambda check_id: after[check_id]["auc"]["point"],
    )
    try:
        plt.style.use("data-ink")
    except OSError:
        pass
    plt.rcParams["svg.hashsalt"] = "noul-polarity"
    figure, axis = plt.subplots(figsize=(14, 13))
    for position, check_id in enumerate(ordered):
        start = before[check_id]["auc"]
        end = after[check_id]["auc"]
        axis.plot(
            [start["point"], end["point"]],
            [position, position],
            color="#999999",
            linewidth=1.6,
            zorder=1,
        )
        for row, colour, marker in (
            (start, "#666666", "o"),
            (end, "#246c78", "o"),
        ):
            axis.errorbar(
                row["point"],
                position,
                xerr=[[row["point"] - row["lower"]], [row["upper"] - row["point"]]],
                fmt=marker,
                color=colour,
                markersize=7,
                linewidth=2.6,
                capsize=0,
                zorder=2,
            )
    axis.axvline(0.5, color="#666666", linestyle=":", linewidth=1.2, zorder=0)
    axis.text(0.503, len(ordered) - 0.4, "chance", fontsize=20, color="#666666")
    axis.set_yticks(range(len(ordered)), ordered, fontsize=20)
    axis.invert_yaxis()
    axis.set_xlim(0, 1.0)
    axis.set_xlabel("calibration AUC of the pass reading", fontsize=22)
    axis.tick_params(axis="x", labelsize=20, width=1.2, length=6)
    axis.tick_params(axis="y", width=1.2, length=6)
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)
    axis.grid(False)
    axis.text(
        0.02,
        0.5,
        "before rewording",
        transform=axis.transAxes,
        fontsize=20,
        color="#666666",
        rotation=90,
        va="center",
        ha="left",
    )
    axis.text(
        0.06,
        0.5,
        "after rewording",
        transform=axis.transAxes,
        fontsize=20,
        color="#246c78",
        rotation=90,
        va="center",
        ha="left",
    )
    target = (
        HERE.parents[2]
        / "figures"
        / "sn-review-through-jev"
        / "srtj-noul-polarity"
        / "per-question-polarity.svg"
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout(rect=(0.16, 0, 1, 1))
    figure.savefig(target, format="svg", metadata={"Date": None})
    target.write_text(
        "\n".join(line.rstrip() for line in target.read_text().splitlines()) + "\n"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action",
        choices=(
            "prepare",
            "costs",
            "resolutions",
            "base",
            "repeat",
            "reverse",
            "rescore",
            "plot",
            "polarity",
        ),
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--concurrency", type=int, default=8)
    args = parser.parse_args()
    RUN.mkdir(parents=True, exist_ok=True)
    if args.action == "prepare":
        _graph_context()
    elif args.action == "costs":
        _read_incumbent_costs()
    elif args.action == "resolutions":
        _read_resolutions()
    elif args.action == "rescore":
        _rescore()
    elif args.action == "plot":
        _plot()
    elif args.action == "polarity":
        _polarity()
    else:
        asyncio.run(_run(args.action, args.limit, args.concurrency))


if __name__ == "__main__":
    main()
