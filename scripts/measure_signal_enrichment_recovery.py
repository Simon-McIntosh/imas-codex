"""Measure local grounded enrichment with sealed context and private receipts.

Requests use the production stage and structured call layer. The trial never
retrieves new context or writes to the graph. Each completed signal is durable
before another request is started, and elapsed arm time includes queueing.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import time
from contextvars import ContextVar
from pathlib import Path
from unittest.mock import patch

from imas_codex.discovery.base import llm
from imas_codex.discovery.signals.enrichment import (
    EnrichmentOptions,
    enrich_signal_batch,
    fit_enrichment_prompt,
)
from imas_codex.graph.client import GraphClient
from imas_codex.settings import get_model
from scripts import benchmark_signal_enrichment as bench
from scripts.refine_signal_enrichment import (
    frozen_signal_order,
    normalized_context,
    subset_messages,
    supplied_target_context,
)

CALL_RECORD = ContextVar("enrichment_call_record")


def check_admission(root, arm, concurrency, lane_document):
    """Bind the trial's concurrency to a fresh published generation ceiling."""
    from datetime import UTC, datetime

    lane = json.loads(lane_document.read_text())
    observed = datetime.fromisoformat(lane["observed_at"].replace("Z", "+00:00"))
    if not 0 <= (datetime.now(UTC) - observed).total_seconds() <= 120:
        raise ValueError("local concurrency publication is stale")
    gate = lane["router_generation_gate"]
    if gate["paused"] or concurrency > gate["effective_width"]:
        raise ValueError(
            "requested concurrency exceeds the published local limit or the gate is paused"
        )
    bench.save(
        root / f"admission-{arm}.json",
        {"requested_concurrency": concurrency, "lane": lane},
    )


def metrics(records, wall_seconds):
    responses = [response for row in records for response in row["responses"]]
    outputs = [
        value
        for row in records
        for value in row["outputs"]
        if value.get("description", "").strip()
    ]
    recoveries = {
        key: sum(row.get("grounding_counts", {}).get(key, 0) for row in records)
        for key in (
            "draft_retries",
            "review_retries",
            "fallback_descriptions",
            "review_skips",
        )
    }
    replacement_calls = {"draft": 0, "review": 0}
    for row in records:
        for request in row.get("messages", []):
            if not request or request[-1].get("role") != "user":
                continue
            feedback = request[-1].get("content", "")
            if not isinstance(feedback, str) or not feedback.startswith(
                "The response failed validation:\n"
            ):
                continue
            try:
                payload = json.loads(request[1]["content"])
            except (ValueError, TypeError):
                payload = None
            stage = (
                "review"
                if isinstance(payload, dict)
                and "drafts" in payload
                and "context_items" in payload
                else "draft"
            )
            replacement_calls[stage] += 1
    for stage, schema in (
        ("draft", "SignalEnrichmentBatch"),
        ("review", "DescriptionReview"),
    ):
        directed = sum(
            bool(row.get("validation_reask"))
            for row in responses
            if row.get("response_schema") == schema
        )
        # A pooled replacement can serve several signal retries in one call.
        recoveries[stage + "_replacement_calls"] = replacement_calls[stage]
        recoveries[stage + "_validation_reasks"] = max(
            0, directed - replacement_calls[stage]
        )
    return {
        "signals": sum(len(row["ids"]) for row in records),
        "delivered": len(outputs),
        "raw_parses": sum(
            row["responses"][0]["score"]["parse_success"]
            for row in records
            if row["responses"]
        ),
        "calls": len(responses),
        "calls_per_signal": len(responses) / len(records),
        "wall_seconds": wall_seconds,
        "seconds_per_thousand": wall_seconds * 1000 / len(records),
        "input_tokens": sum(row.get("input_tokens", 0) for row in responses),
        "output_tokens": sum(row.get("output_tokens", 0) for row in responses),
        "reasoning_tokens": sum(
            row.get("reasoning_tokens", 0) or 0 for row in responses
        ),
        "recoveries": recoveries,
        "errors": [row["error"] for row in records if row.get("error")],
    }


async def run(root, arm, options, concurrency, lane_document):
    check_admission(root, arm, concurrency, lane_document)
    model = get_model("discovery-describe")
    if not model.startswith("local/"):
        raise ValueError("the experiment requires the local description seat")
    cohort = json.loads((root / "cohort.json").read_text())
    if (
        cohort["facility"] == "iter"
        or bench.digest(cohort["reference"]) != cohort["cohort_digest"]
    ):
        raise ValueError("sealed non-ITER cohort required")
    records = bench.load_records(root)
    if any(row["arm"] == arm for row in records):
        raise ValueError("use a fresh arm identity; completed arms are immutable")
    refs = {row["id"]: row for row in cohort["reference"]}
    contexts = [row for row in records if row["arm"] == "context"]
    if len(contexts) != 20:
        raise ValueError("twenty frozen retrieval batches required")
    before = bench.snapshot(cohort["order"])
    bench.assert_unchanged(cohort["reference"], before)
    original_transport = llm._acompletion_local
    completed = []
    semaphore = asyncio.Semaphore(concurrency)
    db = bench.open_store(root / "trial.sqlite")

    async def observe(kwargs):
        captured, schema = CALL_RECORD.get()
        started = time.monotonic()
        try:
            response = await original_transport(kwargs)
        except Exception as exc:
            captured.append(
                {
                    "raw": None,
                    "score": {"parse_success": False},
                    "response_schema": schema.__name__,
                    "seconds": time.monotonic() - started,
                    "error": repr(exc),
                }
            )
            raise
        raw = response.choices[0].message.content
        captured.append(
            {
                "raw": raw,
                "score": bench.score_response(raw, 1, response_model=schema),
                "response_schema": schema.__name__,
                "served_model": response.model,
                "finish_reason": response.choices[0].finish_reason,
                "seconds": time.monotonic() - started,
                "input_tokens": response.usage.prompt_tokens or 0,
                "output_tokens": response.usage.completion_tokens or 0,
                "reasoning_tokens": getattr(response.usage, "reasoning_tokens", 0),
                "validation_reask": kwargs["messages"][-1]["content"].startswith(
                    "The response failed validation:\n"
                ),
            }
        )
        return response

    async def one(id, messages):
        async with semaphore:
            captured, requests = [], []
            started = time.monotonic()

            async def call(**kwargs):
                requests.append(copy.deepcopy(kwargs["messages"]))
                token = CALL_RECORD.set((captured, kwargs["response_model"]))
                try:
                    return await llm.acall_llm_structured(
                        **kwargs, max_retries=1, timeout=180
                    )
                finally:
                    CALL_RECORD.reset(token)

            output, counts, error = [], {}, None
            try:
                result, cost, tokens = await enrich_signal_batch(
                    model=model,
                    messages=messages,
                    options=options,
                    call=call,
                    temperature=0.3,
                    service="facility-discovery",
                    **options.call_kwargs(),
                )
                output = [
                    {
                        "id": id,
                        **row.model_dump(mode="json"),
                        "description_fallback": row.signal_index
                        in result.fallback_indices,
                    }
                    for row in result.results
                    if row.signal_index == 1
                ]
                counts = {
                    key: getattr(result, key)
                    for key in (
                        "draft_retries",
                        "review_retries",
                        "review_skips",
                        "evidence_drops",
                        "review_drops",
                        "review_revisions",
                        "length_drops",
                    )
                }
                counts["fallback_descriptions"] = len(result.fallback_indices)
            except Exception as exc:
                error = repr(exc)
            record = {
                "arm": arm,
                "ids": [id],
                "options": options.model_dump(),
                "responses": captured,
                "messages": requests,
                "outputs": output,
                "grounding_counts": counts,
                "seconds": time.monotonic() - started,
                "error": error,
            }
            with db:
                db.execute(
                    "INSERT INTO batches(document) VALUES (?)", (bench.encoded(record),)
                )
            completed.append(record)
            print(
                json.dumps(
                    {
                        "arm": arm,
                        "completed": len(completed),
                        "delivered": sum(bool(row["outputs"]) for row in completed),
                    }
                ),
                flush=True,
            )

    tasks = []
    for context in contexts:
        original = context["messages"][0]
        order = frozen_signal_order(
            original, [bench.input_signal(refs[id]) for id in context["ids"]]
        )
        for index, id in enumerate(order, 1):
            messages = subset_messages(original, [index])
            accessor = refs[id]["properties"]["accessor"]
            if normalized_context(
                supplied_target_context(messages, accessor)
            ) != normalized_context(supplied_target_context(original, accessor)):
                raise ValueError("target context changed")
            if options.fitted_prompt:
                messages[0]["content"] = fit_enrichment_prompt(messages[0]["content"])
            tasks.append((id, messages))
    started = time.monotonic()
    with (
        patch.object(llm, "_acompletion_local", observe),
        patch.object(
            GraphClient,
            "query",
            side_effect=AssertionError("generation must never access the graph"),
        ),
    ):
        await asyncio.gather(*(one(id, messages) for id, messages in tasks))
    elapsed = time.monotonic() - started
    db.close()
    after = bench.snapshot(cohort["order"])
    bench.assert_unchanged(before, after)
    result = {
        "arm": arm,
        "options": options.model_dump(),
        "concurrency": concurrency,
        **metrics(completed, elapsed),
        "before_digest": bench.digest(before),
        "after_digest": bench.digest(after),
        "identical": True,
    }
    bench.save(root / f"metrics-{arm}.json", result)
    print(bench.encoded(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--options", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, choices=range(1, 9), default=1)
    parser.add_argument(
        "--lane-document",
        type=Path,
        default=Path.home() / "public/imas-ambix/lane.json",
    )
    args = parser.parse_args()
    options = EnrichmentOptions.model_validate(json.loads(args.options.read_text()))
    asyncio.run(
        run(args.store, args.arm, options, args.concurrency, args.lane_document)
    )


if __name__ == "__main__":
    main()
