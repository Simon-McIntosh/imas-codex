"""Run one production seat benchmark while retaining matched item outcomes."""

from __future__ import annotations

import asyncio
import html
import json
import random
import sys
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

from imas_codex.cli.sn import sn_bench
from imas_codex.discovery.base import llm
from imas_codex.graph import dd_domain_classifier
from imas_codex.settings import (
    get_model,
    get_sn_review_docs_models,
    get_sn_review_names_models,
)
from imas_codex.standard_names import benchmark, benchmark_roles

ROOT = Path(__file__).resolve().parents[4]
RESULTS = Path(__file__).with_name("llm-gpt6-seat-bench-results.json")
RUN = Path(
    "/home/ITER/mcintos/.config/reckon/crew/runs/"
    "r-20261018T180610585160-llm-gpt6-seat-bench"
)
EVENTS = RUN / "call-events.jsonl"
MODEL_SOL = "openrouter/openai/gpt-6.1-sol"
MODEL_LUNA = "openrouter/openai/gpt-6-luna"


def record(event: dict) -> None:
    with EVENTS.open("a") as stream:
        stream.write(json.dumps(event, default=str) + "\n")


def measured_spend() -> float:
    if not EVENTS.exists():
        return 0.0
    return sum(
        json.loads(line).get("cost_usd", 0.0)
        for line in EVENTS.read_text().splitlines()
    )


def main(role: str) -> None:
    results = json.loads(RESULTS.read_text())
    stop_before_call = results.get("execution_controls", {}).get(
        "budget_stop_before_next_call_usd",
        results["pre_registration"]["budget_stop_before_next_call_usd"],
    )
    if role in results["raw_reports"]:
        raise RuntimeError(f"Role already measured: {role}")
    if role not in results["pre_registration"]["samples"]:
        raise ValueError(role)
    if measured_spend() >= stop_before_call:
        raise RuntimeError("Spend stop reached before role start")

    outcomes: dict[str, dict] = defaultdict(dict)
    corpus: list[dict] = []
    candidate_by_task: dict[asyncio.Task, str] = {}
    budget_sem = asyncio.Semaphore(1)
    original_call = llm.acall_llm_structured
    original_review = benchmark.score_with_reviewer
    original_classify = dd_domain_classifier._classify_batch

    async def _metered_call_inner(*args, **kwargs):
        before = measured_spend()
        if before >= stop_before_call:
            raise RuntimeError(f"Spend stop reached at USD {before:.4f}")
        model = kwargs.get("model", "")
        response_type = kwargs.get("response_model")
        started = time.perf_counter()
        try:
            response, cost, usage = await original_call(*args, **kwargs)
        except Exception as exc:
            record(
                {
                    "role": role,
                    "model": model,
                    "error": str(exc)[:300],
                    "latency_seconds": time.perf_counter() - started,
                    "cost_usd": 0.0,
                }
            )
            raise
        duration = time.perf_counter() - started
        record(
            {
                "role": role,
                "model": model,
                "response_type": getattr(response_type, "__name__", ""),
                "latency_seconds": duration,
                "cost_usd": cost or 0.0,
                "timestamp_utc": datetime.now(UTC).isoformat(),
            }
        )
        task = asyncio.current_task()
        if role == "refine" and task is not None:
            if getattr(response_type, "__name__", "") == "RefinedName":
                candidate_by_task[task] = model
            elif getattr(response_type, "__name__", "") == "_RefineJudgement":
                prompt = kwargs["messages"][-1]["content"]
                prior_name = prompt.split("## Previous (rejected) name — score", 1)[
                    1
                ].split("\n", 2)[1]
                matches = [row for row in corpus if row["prior_name"] == prior_name]
                if len(matches) != 1:
                    raise RuntimeError(
                        f"Refine item key is ambiguous for {prior_name!r}: {len(matches)} matches"
                    )
                candidate = candidate_by_task.get(task)
                if candidate:
                    outcomes[candidate][matches[0]["sn_id"]] = float(
                        response.defect_resolution
                    )
        return response, cost, usage

    async def metered_call(*args, **kwargs):
        async with budget_sem:
            return await _metered_call_inner(*args, **kwargs)

    async def traced_review(candidates, reviewer_model, *args, **kwargs):
        reviews, cost = await original_review(
            candidates, reviewer_model, *args, **kwargs
        )
        measured_model = models[len(outcomes)] if role == "docs" else reviewer_model
        outcomes[measured_model] = {
            r["name"]: float(r.get("score") or 0.0) for r in reviews
        }
        if role == "breaker-names":
            (RUN / "partial-breaker-names-outcomes.json").write_text(
                json.dumps(dict(outcomes), indent=2) + "\n"
            )
        return reviews, cost

    async def traced_classify(batch, *args, **kwargs):
        rows, cost = await original_classify(batch, *args, **kwargs)
        model = kwargs.get("model", "")
        outcomes[model].update({r["id"]: r["physics_domain"] for r in rows})
        return rows, cost

    def wrap_loader(name: str) -> None:
        original = getattr(benchmark_roles, name)

        def capture(*args, **kwargs):
            rows = original(*args, **kwargs)
            corpus.extend(rows)
            return rows

        setattr(benchmark_roles, name, capture)

    loaders = {
        "refine": "load_refine_corpus",
        "docs": "load_docs_sample",
        "breaker-names": "load_breaker_corpus",
        "breaker-docs": "load_breaker_corpus",
        "classifier": "load_classifier_gold",
    }
    wrap_loader(loaders[role])
    llm.acall_llm_structured = metered_call
    benchmark.score_with_reviewer = traced_review
    dd_domain_classifier._classify_batch = traced_classify

    sample = (
        results.get("execution_controls", {})
        .get("sample_overrides", {})
        .get(role, results["pre_registration"]["samples"][role]["n"])
    )
    raw_path = RUN / f"raw-{role}.json"
    incumbent = {
        "refine": lambda: get_model("sn-refine"),
        "docs": lambda: get_model("sn-docs"),
        "breaker-names": lambda: get_sn_review_names_models()[-1],
        "breaker-docs": lambda: get_sn_review_docs_models()[-1],
        "classifier": lambda: get_model("sn-classifier"),
    }[role]()
    models = list(dict.fromkeys([incumbent, MODEL_SOL, MODEL_LUNA]))
    if role == "breaker-names":
        models = list(dict.fromkeys([MODEL_SOL, MODEL_LUNA, incumbent]))
    command = [
        "sn",
        "bench",
        "--role",
        role,
        "--models",
        ",".join(models),
        "--max-candidates",
        str(sample),
        "--seed",
        "1729",
        "--output",
        str(raw_path),
    ]
    print("command: imas-codex " + " ".join(command), flush=True)
    print(
        f"revision: {__import__('subprocess').check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()}",
        flush=True,
    )
    print(f"tree: {ROOT}", flush=True)
    print(f"module: {benchmark_roles.__file__}", flush=True)
    print(f"cwd: {Path.cwd().resolve()}", flush=True)
    sn_bench.main(args=command[2:], standalone_mode=False)
    report = json.loads(raw_path.read_text())
    if role == "refine":
        fixture = [{"id": row["sn_id"]} for row in corpus]
    elif role == "docs":
        fixture = [{"id": row["name"]} for row in corpus]
    elif role.startswith("breaker"):
        fixture = [
            {
                "id": row["name"],
                "pair_mean": (
                    row["pair_scores"]["qwen"] + row["pair_scores"]["minimax"]
                )
                / 2,
                "final_accepted": bool(row["final_accepted"]),
            }
            for row in corpus
        ]
    else:
        fixture = [
            {"id": row["path"], "expected": row["expected_domain"]}
            for row in corpus[:sample]
        ]
    results["raw_reports"][role] = report
    results["item_outcomes"][role] = {"fixture": fixture, "by_model": dict(outcomes)}
    if role == "refine":
        for row in report["results"]:
            if len(outcomes.get(row["model"], {})) != row["n"]:
                raise RuntimeError(
                    f"Refine outcomes for {row['model']} do not match judged count"
                )
    if role == "docs":
        for row in report["results"]:
            if len(outcomes.get(row["model"], {})) != len(fixture):
                raise RuntimeError(
                    f"Docs outcomes for {row['model']} do not match fixture count"
                )
    if role.startswith("breaker") or role == "classifier":
        for row in report["results"]:
            if not outcomes.get(row["model"]):
                raise RuntimeError(f"No item outcomes captured for {row['model']}")
    results["total_spend_usd"] = round(measured_spend(), 6)
    RESULTS.write_text(json.dumps(results, indent=2) + "\n")
    print(
        f"role={role} fixture={len(fixture)} spend_usd={results['total_spend_usd']:.6f}",
        flush=True,
    )


def _primary(role: str, fixture: list[dict], values: dict, indices: list[int]) -> float:
    rows = [fixture[index] for index in indices]
    if role in ("refine", "docs"):
        return sum(values[row["id"]] for row in rows) / len(rows)
    if role == "classifier":
        return sum(values[row["id"]] == row["expected"] for row in rows) / len(rows)
    scores = [values[row["id"]] for row in rows]
    pair_means = [row["pair_mean"] for row in rows]
    accepted = [row["final_accepted"] for row in rows]
    return benchmark_roles.verdict_flip_quality(scores, pair_means, accepted)[0]


def _interval(values: list[float]) -> list[float]:
    ordered = sorted(values)
    return [
        ordered[int(0.025 * (len(ordered) - 1))],
        ordered[int(0.975 * (len(ordered) - 1))],
    ]


def analyze() -> None:
    results = json.loads(RESULTS.read_text())
    events = [json.loads(line) for line in EVENTS.read_text().splitlines()]
    spent = sum(event.get("cost_usd", 0.0) for event in events)
    if spent >= results["pre_registration"]["budget_ceiling_usd"]:
        raise RuntimeError(f"Spend ceiling exceeded: USD {spent:.6f}")
    results["total_spend_usd"] = round(spent, 6)
    analysis: dict[str, dict] = {}
    for role, report in results["raw_reports"].items():
        captured = results["item_outcomes"][role]
        fixture = captured["fixture"]
        by_model = captured["by_model"]
        ids = [row["id"] for row in fixture]
        for model in (MODEL_SOL, MODEL_LUNA, report["incumbent"]):
            if model not in by_model or not set(ids) <= set(by_model[model]):
                raise RuntimeError(f"Incomplete matched outcomes: {role} {model}")
        rng = random.Random(1729)
        draws = [[rng.randrange(len(fixture)) for _ in fixture] for _ in range(10000)]
        full = list(range(len(fixture)))
        model_stats: dict[str, dict] = {}
        cutoff = max(
            (
                run["report"]["timestamp"]
                for run in results.get("superseded_runs", [])
                if run["role"] == role
            ),
            default="",
        )
        if role == "breaker-names":
            cutoff = results["execution_controls"]["sample_override_timestamp_utc"]
        for row in report["results"]:
            model = row["model"]
            vals = [_primary(role, fixture, by_model[model], draw) for draw in draws]
            calls = [
                event
                for event in events
                if event["role"] == role
                and event.get("model") == model
                and event.get("timestamp_utc", "") > cutoff
            ]
            model_stats[model] = {
                "primary": _primary(role, fixture, by_model[model], full),
                "interval_95": _interval(vals),
                "candidate_call_count": len(calls),
                "candidate_cost_per_call_usd": sum(event["cost_usd"] for event in calls)
                / len(calls)
                if calls
                else None,
                "candidate_latency_per_call_seconds": sum(
                    event["latency_seconds"] for event in calls
                )
                / len(calls)
                if calls
                else None,
                "effective_cost_per_item_usd": row["cost"] / len(fixture),
            }
        differences = [
            _primary(role, fixture, by_model[MODEL_LUNA], draw)
            - _primary(role, fixture, by_model[MODEL_SOL], draw)
            for draw in draws
        ]
        diff_ci = _interval(differences)
        shuffled = dict(by_model[MODEL_SOL])
        permuted = [shuffled[item_id] for item_id in ids]
        rng.shuffle(permuted)
        shuffled = dict(zip(ids, permuted, strict=True))
        control_ci = _interval(
            [
                _primary(role, fixture, by_model[MODEL_LUNA], draw)
                - _primary(role, fixture, shuffled, draw)
                for draw in draws
            ]
        )
        analysis[role] = {
            "sample_size": len(fixture),
            "primary_metric": {
                "refine": "defect_resolution",
                "docs": "rubric_score",
                "breaker-names": "verdict_flip_quality",
                "breaker-docs": "verdict_flip_quality",
                "classifier": "accuracy",
            }[role],
            "models": model_stats,
            "luna_minus_sol": _primary(role, fixture, by_model[MODEL_LUNA], full)
            - _primary(role, fixture, by_model[MODEL_SOL], full),
            "paired_interval_95": diff_ci,
            "capable_under_rule": diff_ci[0] >= -0.05,
            "shuffled_control_interval_95": control_ci,
            "shuffled_control_contains_zero": control_ci[0] <= 0 <= control_ci[1],
        }
    results["analysis"] = analysis
    RESULTS.write_text(json.dumps(results, indent=2) + "\n")
    print(
        json.dumps(
            {
                role: {
                    "difference": row["luna_minus_sol"],
                    "interval": row["paired_interval_95"],
                    "capable": row["capable_under_rule"],
                }
                for role, row in analysis.items()
            },
            indent=2,
        )
    )
    print(f"total_spend_usd={spent:.6f}")


def render_figure() -> None:
    import matplotlib.pyplot as plt
    from matplotlib.text import Text

    results = json.loads(RESULTS.read_text())
    analysis = results["analysis"]
    ordered = ["refine", "docs", "breaker-names", "breaker-docs", "classifier"]
    labels = ["Refine", "Docs", "Names breaker", "Docs breaker", "Classifier"]
    plt.style.use("data-ink")
    fig, axis = plt.subplots(figsize=(14, 6), dpi=100)
    for index, role in enumerate(ordered):
        y = len(ordered) - 1 - index
        if role not in analysis:
            axis.text(0.06, y, "pending", va="center", color="#777777", fontsize=14)
            continue
        row = analysis[role]
        point = row["luna_minus_sol"]
        lower, upper = row["paired_interval_95"]
        axis.errorbar(
            point,
            y,
            xerr=[[point - lower], [upper - point]],
            fmt="o",
            color="#3d7392",
            ecolor="#3d7392",
            linewidth=2.6,
            markersize=9,
            capsize=5,
            zorder=3,
        )
    axis.axvline(-0.05, color="#777777", linestyle=":", linewidth=1.2, zorder=1)
    axis.text(-0.05, 4.35, "−0.05 limit", color="#666666", ha="center", fontsize=14)
    axis.set_yticks(list(reversed(range(len(ordered)))), labels)
    axis.set_ylim(-0.5, len(ordered) - 0.35)
    axis.set_xlim(-0.8, 0.85)
    axis.set_xticks([-0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8])
    axis.set_xlabel("Luna − Sol primary metric")
    fig.tight_layout()
    fig.canvas.draw()
    canvas = fig.bbox
    for artist in fig.findobj(Text):
        if not artist.get_visible() or not artist.get_text():
            continue
        box = artist.get_window_extent(fig.canvas.get_renderer())
        if (
            box.x0 < canvas.x0
            or box.y0 < canvas.y0
            or box.x1 > canvas.x1
            or box.y1 > canvas.y1
        ):
            raise RuntimeError(f"Figure text outside canvas: {artist.get_text()!r}")
    destination = ROOT / "docs/figures/llm-seats/llm-gpt6-seat-bench"
    destination.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination / "paired-differences.svg")
    fig.savefig(RUN / "figure-preview.png", dpi=100)
    plt.close(fig)
    print(destination / "paired-differences.svg")


def render_fragment() -> None:
    results = json.loads(RESULTS.read_text())
    analysis = results["analysis"]
    labels = {
        "refine": "Refine",
        "docs": "Docs",
        "breaker-names": "Names breaker",
        "breaker-docs": "Docs breaker",
        "classifier": "Classifier",
    }
    classifier_outcomes = results["item_outcomes"]["classifier"]
    classifier_fixture = classifier_outcomes["fixture"]
    classifier_predictions = classifier_outcomes["by_model"]
    classifier_luna = classifier_predictions[MODEL_LUNA]
    classifier_sol = classifier_predictions[MODEL_SOL]
    classifier_luna_correct = sum(
        classifier_luna[row["id"]] == row["expected"] for row in classifier_fixture
    )
    classifier_sol_correct = sum(
        classifier_sol[row["id"]] == row["expected"] for row in classifier_fixture
    )
    classifier_agreement = sum(
        classifier_luna[row["id"]] == classifier_sol[row["id"]]
        for row in classifier_fixture
    )
    recommendations = {
        "refine": (
            "Luna",
            "Its paired lower bound clears the rule; the sampled defect-resolution score is higher at much lower direct call cost.",
        ),
        "docs": (
            "Sol",
            "Luna misses the bound and Sol has the highest sampled rubric score. Confirm on a larger set before moving the production seat.",
        ),
        "breaker-docs": (
            "Keep incumbent",
            "Only two Luna and three Sol verdict flips inform the conditional quality metric; its passing bound is too fragile for a seat move.",
        ),
        "classifier": (
            "Luna",
            f"Luna matched {classifier_luna_correct} and Sol matched {classifier_sol_correct} of {len(classifier_fixture)} gold labels; their predictions agreed on {classifier_agreement} paths. Luna has lower direct cost. Broaden the gold cohort beyond its two domains before changing production.",
        ),
    }
    rows: list[str] = []
    for role, label in labels.items():
        if role not in analysis:
            rows.append(
                f'<tr><th>{label}</th><td>—</td><td colspan="5">Pending: the production reviewer batch has not returned a complete report.</td><td>Keep incumbent while unmeasured.</td></tr>'
            )
            continue
        item = analysis[role]
        report = results["raw_reports"][role]
        if role == "breaker-names":
            recommendation = (
                "Luna"
                if item["capable_under_rule"]
                and next(r for r in report["results"] if r["model"] == MODEL_LUNA)[
                    "metrics"
                ].get("n_flips", 0)
                >= 2
                else "Keep incumbent",
                "Use the paired bound with the observed flip count; a sparse flip set warrants retaining the current reviewer.",
            )
        else:
            recommendation = recommendations[role]
        difference = item["luna_minus_sol"]
        low, high = item["paired_interval_95"]
        verdict = "capable" if item["capable_under_rule"] else "not established"
        for number, model_row in enumerate(report["results"]):
            model = model_row["model"]
            stats = item["models"][model]
            metric_low, metric_high = stats["interval_95"]
            role_cell = (
                f'<th rowspan="3">{label}<br><small>n={item["sample_size"]}</small></th>'
                if number == 0
                else ""
            )
            comparison = (
                f'<td rowspan="3">{difference:+.3f} [{low:+.3f}, {high:+.3f}]<br>{verdict}</td>'
                if number == 0
                else ""
            )
            decision = (
                f'<td rowspan="3"><strong>{recommendation[0]}</strong>. {html.escape(recommendation[1])}</td>'
                if number == 0
                else ""
            )
            rows.append(
                f"<tr>{role_cell}<td><code>{html.escape(model.split('/')[-1])}</code>{' (incumbent)' if model == report['incumbent'] else ''}</td>"
                f"<td>{stats['primary']:.3f} [{metric_low:.3f}, {metric_high:.3f}]</td>{comparison}"
                f"<td>${stats['candidate_cost_per_call_usd']:.4f}</td><td>{stats['candidate_latency_per_call_seconds']:.1f} s</td>"
                f"<td>${stats['effective_cost_per_item_usd']:.4f}</td>{decision}</tr>"
            )
    samples = results["pre_registration"]["samples"]
    sample_text = "; ".join(
        f"{labels[role]} {results.get('execution_controls', {}).get('sample_overrides', {}).get(role, spec['n'])}: {spec['reason'].rstrip('.')}"
        for role, spec in samples.items()
    )
    control_failures = [
        labels[role]
        for role, item in analysis.items()
        if not item["shuffled_control_contains_zero"]
    ]
    control_text = (
        "The shuffled-output interval excludes zero for "
        + ", ".join(control_failures)
        + "; that control does not remove a marginal score shift, so its failure limits the pairing check rather than overturning the measured paired interval."
        if control_failures
        else "Every shuffled-output interval contained zero."
    )
    figure_scope = (
        "All five roles have measured paired intervals."
        if len(analysis) == len(labels)
        else "An unmeasured role is shown as pending."
    )
    body = f"""<section id="gpt6-seat-benchmark">
  <h3>GPT seat comparison: Luna against Sol and the live incumbent</h3>
  <p>The decision rule was registered at <code>{results["pre_registration"]["timestamp_utc"]}</code> before paid calls: Luna is capable when the 95% lower bound of its paired primary-metric difference from Sol is at least −0.05. Matched items were resampled 10,000 times with seed 1729. No production seat or graph state changed.</p>
  <p><strong>Sample sizes.</strong> {html.escape(sample_text)}. The names-breaker sample was reduced from its registered 12 after the incumbent reviewer timed out on the first batch; the retry uses eight common seeded cases. The classifier fixture's first 24 paths cover two domains.</p>
  <figure><img src="/imas-codex/figures/llm-seats/llm-gpt6-seat-bench/paired-differences.svg" alt="Paired Luna minus Sol primary metric for five seats, with confidence intervals and the minus 0.05 decision limit"><figcaption>Points are paired primary-metric differences, bars are 95% bootstrap intervals, and the dotted line is the registered −0.05 limit. {figure_scope}</figcaption></figure>
  <table><thead><tr><th>Role</th><th>Model</th><th>Primary metric [95% interval]</th><th>Luna − Sol [95% interval], rule</th><th>Model cost/call</th><th>Latency/call</th><th>Effective cost/item</th><th>Recommendation</th></tr></thead><tbody>{"".join(rows)}</tbody></table>
  <p>Model cost and latency are means of successful direct candidate calls. Effective cost per item includes the held-out <code>claude-opus-4.8</code> judge for refine and docs. The raw per-role reports, item outcomes, intervals and receipts are in <code>llm-gpt6-seat-bench-results.json</code>; call-level receipts are in the run record. <strong>Total measured spend: USD {results["total_spend_usd"]:.4f} of the USD 10 ceiling</strong>, including superseded instrument runs and the interrupted names-breaker call.</p>
  <p><strong>Control and limits.</strong> {html.escape(control_text)} Breaker quality conditions on verdict flips, which are sparse in the docs sample. The classifier's equal scores are supported by 24 returned predictions per model and 16 exact matches each; the sampled gold paths cover only two domains. These bounds describe the sampled fixtures, not all pipeline traffic.</p>
  <p><code>sn-prose-adjudicator</code> and <code>cluster-labels</code> have no <code>sn bench --role</code> mode. Assess prose adjudication on a blinded, human-labeled set of true and false banned-prose flags, and assess cluster labels on held-out expert-labeled clusters with label fidelity and repeatability checks before considering a model change.</p>
</section>
"""
    destination = RESULTS.with_name("llm-gpt6-seat-bench.html")
    destination.write_text(body)
    print(destination)


if __name__ == "__main__":
    if sys.argv[1] == "analyze":
        analyze()
    elif sys.argv[1] == "figure":
        render_figure()
    elif sys.argv[1] == "fragment":
        render_fragment()
    else:
        main(sys.argv[1])
