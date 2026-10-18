"""Run one production seat benchmark while retaining matched item outcomes."""

from __future__ import annotations

import asyncio
import json
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
    if role in results["raw_reports"]:
        raise RuntimeError(f"Role already measured: {role}")
    if role not in results["pre_registration"]["samples"]:
        raise ValueError(role)
    if (
        measured_spend()
        >= results["pre_registration"]["budget_stop_before_next_call_usd"]
    ):
        raise RuntimeError("Spend stop reached before role start")

    outcomes: dict[str, dict] = defaultdict(dict)
    corpus: list[dict] = []
    candidate_by_task: dict[asyncio.Task, str] = {}
    original_call = llm.acall_llm_structured
    original_review = benchmark.score_with_reviewer
    original_classify = dd_domain_classifier._classify_batch

    async def metered_call(*args, **kwargs):
        before = measured_spend()
        if before >= results["pre_registration"]["budget_stop_before_next_call_usd"]:
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

    async def traced_review(candidates, reviewer_model, *args, **kwargs):
        reviews, cost = await original_review(
            candidates, reviewer_model, *args, **kwargs
        )
        measured_model = models[len(outcomes)] if role == "docs" else reviewer_model
        outcomes[measured_model] = {
            r["name"]: float(r.get("score") or 0.0) for r in reviews
        }
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

    sample = results["pre_registration"]["samples"][role]["n"]
    raw_path = RUN / f"raw-{role}.json"
    incumbent = {
        "refine": lambda: get_model("sn-refine"),
        "docs": lambda: get_model("sn-docs"),
        "breaker-names": lambda: get_sn_review_names_models()[-1],
        "breaker-docs": lambda: get_sn_review_docs_models()[-1],
        "classifier": lambda: get_model("sn-classifier"),
    }[role]()
    models = list(dict.fromkeys([incumbent, MODEL_SOL, MODEL_LUNA]))
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


if __name__ == "__main__":
    main(sys.argv[1])
