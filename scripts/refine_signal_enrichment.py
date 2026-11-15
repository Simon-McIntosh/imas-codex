"""Replay local signal-enrichment variants against a sealed non-ITER cohort.

Frozen group retrieval remains byte-identical. Smaller batches remove only
other signals' input blocks and renumber the selected signals. No new context
is retrieved and no paid description is generated.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import re
from collections import Counter
from pathlib import Path

from imas_codex.discovery.signals.enrichment import (
    EnrichmentOptions,
    fit_enrichment_prompt,
)
from imas_codex.settings import get_model
from scripts import benchmark_signal_enrichment as bench

GROUP = re.compile(
    r"^## (?:EDAS Category:|PPF DDA:|MDSplus Tree:|JPF Subsystem:|TDI Function:|JET Device Description:)",
    re.M,
)
SIGNAL = re.compile(r"^### Signal (\d+)[ \t]*$", re.M)


def normalized_context(value: str) -> str:
    value = re.sub(
        r"Classify these \d+ signals\.", "Classify the selected signals.", value
    )
    return SIGNAL.sub("### Signal", value)


def subset_messages(messages: list[dict], indices: list[int]) -> list[dict]:
    """Change only the target signal blocks and their output indices."""
    result = copy.deepcopy(messages)
    user = next(row for row in result if row["role"] == "user")
    text = user["content"]
    headers = list(SIGNAL.finditer(text))
    positions = [h.start() for h in headers] + [g.start() for g in GROUP.finditer(text)]
    if [int(h[1]) for h in headers] != list(range(1, len(headers) + 1)):
        raise ValueError("frozen signal indices are not unique and sequential")
    if (
        not indices
        or len(set(indices)) != len(indices)
        or not set(indices) <= set(range(1, len(headers) + 1))
    ):
        raise ValueError("invalid frozen subset indices")
    # Work backwards so every boundary still addresses the original string.
    for header in reversed(headers):
        index = int(header[1])
        end = min((p for p in positions if p > header.start()), default=len(text))
        block = text[header.start() : end]
        if index in indices:
            block = SIGNAL.sub(f"### Signal {indices.index(index) + 1}", block, count=1)
        else:
            block = ""
        text = text[: header.start()] + block + text[end:]
    user["content"] = re.sub(
        r"Classify these \d+ signals\.",
        f"Classify these {len(indices)} signals.",
        text,
        count=1,
    )
    return result


def failure_class(response: dict, *, terminal: bool = False) -> str:
    """Classify recorded failures without treating repaired text as raw JSON."""
    if response.get("finish_reason") == "abort":
        return "provider_abort"
    if response.get("raw") is None:
        return "transport"
    error = (response.get("error") if terminal else None) or response.get(
        "score", {}
    ).get("error", "")
    if response.get("score", {}).get("empty"):
        return "empty"
    if "json_invalid" in error or "Invalid JSON" in error:
        return "invalid_json"
    if "enum" in error or "physics_domain" in error:
        return "enumeration"
    if "list_type" in error:
        return "array_type"
    return "schema_or_coverage"


def summarize(records: list[dict], cohort: dict) -> dict:
    summary = {}
    for arm in dict.fromkeys(r["arm"] for r in records):
        if arm in ("context", "paid"):
            continue
        batches = [r for r in records if r["arm"] == arm]
        firsts = [r["responses"][0] for r in batches]
        responses = [response for r in batches for response in r["responses"]]
        seconds = sum(r["seconds"] for r in batches)
        outputs = {o["id"]: o for r in batches for o in r["outputs"]}
        failed = Counter()
        raw_failed = Counter()
        for r in batches:
            if not r["responses"][0]["score"]["parse_success"]:
                raw_failed[failure_class(r["responses"][0])] += len(r["ids"])
            absent = set(r["ids"]) - {o["id"] for o in r["outputs"]}
            if absent:
                last = r["responses"][-1]
                cause = (
                    failure_class(last, terminal=True)
                    if last.get("error")
                    else "coverage"
                )
                failed[cause] += len(absent)
        summary[arm] = {
            "signals": sum(len(r["ids"]) for r in batches),
            "initial_calls": len(batches),
            "total_calls": len(responses),
            "raw_parses": sum(r["score"]["parse_success"] for r in firsts),
            "raw_parse_rate": sum(r["score"]["parse_success"] for r in firsts)
            / len(firsts),
            "raw_failed_signals_by_class": dict(raw_failed),
            "delivered_first": sum(
                len(r["outputs"]) for r in batches if len(r["responses"]) == 1
            ),
            "delivered_after_reask": sum(
                len(r["outputs"]) for r in batches if len(r["responses"]) == 2
            ),
            "delivered": len(outputs),
            "delivery_rate": len(outputs) / len(cohort["order"]),
            "failed_signals_by_class": dict(failed),
            "seconds": seconds,
            "calls_per_minute": len(responses) * 60 / seconds,
            "seconds_per_thousand": seconds * 1000 / len(cohort["order"]),
            "input_tokens": sum(r["input_tokens"] for r in batches),
            "output_tokens": sum(r["output_tokens"] for r in batches),
            "reasoning_tokens": sum(r.get("reasoning_tokens") or 0 for r in responses),
            "cost_usd": sum(r["cost_usd"] for r in batches),
            "options": batches[0].get("options", {}),
        }
    return summary


async def run_variant(root: Path, arm: str, options: EnrichmentOptions) -> None:
    """Checkpoint each production replay to the existing private SQLite sink."""
    model = get_model("discovery-describe")
    if not model.startswith("local/"):
        raise ValueError("refinement requires the configured local description model")
    cohort = json.loads((root / "cohort.json").read_text())
    if (
        cohort["facility"] == "iter"
        or bench.digest(cohort["reference"]) != cohort["cohort_digest"]
    ):
        raise ValueError("sealed non-ITER reference cohort required")
    records = bench.load_records(root)
    contexts = [r for r in records if r["arm"] == "context"]
    if len(contexts) != 20 or sum(len(r["ids"]) for r in contexts) != 200:
        raise ValueError("need the original twenty frozen context batches")
    refs = {r["id"]: r for r in cohort["reference"]}
    done = [r for r in records if r["arm"] == arm]
    if any(
        EnrichmentOptions.model_validate(r.get("options", {})).model_dump()
        != options.model_dump()
        for r in done
    ):
        raise ValueError("variant name already contains a different configuration")
    db = bench.open_store(root / "trial.sqlite")
    bench.assert_unchanged(cohort["reference"], bench.snapshot(cohort["order"]))
    for context in contexts:
        original = context["messages"][0]
        for offset in range(0, len(context["ids"]), options.batch_size):
            ids = context["ids"][offset : offset + options.batch_size]
            if any(r["ids"] == ids for r in done):
                continue
            messages = (
                original
                if len(ids) == len(context["ids"])
                else subset_messages(
                    original, list(range(offset + 1, offset + 1 + len(ids)))
                )
            )
            for id in ids:
                accessor = refs[id]["properties"]["accessor"]
                if normalized_context(
                    bench.target_context(messages, accessor)
                ) != normalized_context(bench.target_context(original, accessor)):
                    raise ValueError(
                        "subset changed the target's frozen retrieval context"
                    )
            messages = copy.deepcopy(messages)
            if options.fitted_prompt:
                system = next(m for m in messages if m["role"] == "system")
                system["content"] = fit_enrichment_prompt(system["content"])
            record = await bench.replay_batch(
                [bench.input_signal(refs[id]) for id in ids],
                cohort["facility"],
                db,
                messages=messages,
                arm=arm,
                enrichment_options=options,
                call_options=options.call_kwargs(),
            )
            done.append(record)
            all_records = bench.load_records(root)
            metrics = summarize(all_records, cohort)
            bench.save(root / "refinement-summary.json", metrics)
            print(bench.encoded({"arm": arm, **metrics[arm]}), flush=True)
    after = bench.snapshot(cohort["order"])
    bench.assert_unchanged(cohort["reference"], after)
    bench.save(
        root / f"integrity-{arm}.json",
        {
            "before": cohort["cohort_digest"],
            "after": bench.digest(after),
            "identical": True,
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--schema-constraint", action="store_true")
    parser.add_argument("--json-mode", action="store_true")
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--fitted-prompt", action="store_true")
    parser.add_argument("--validation-reask", action="store_true")
    args = parser.parse_args()
    options = EnrichmentOptions(
        batch_size=args.batch_size,
        schema_constraint=args.schema_constraint,
        json_mode=args.json_mode,
        thinking=True if args.thinking else None,
        max_tokens=args.max_tokens,
        fitted_prompt=args.fitted_prompt,
        validation_reask=args.validation_reask,
    )
    asyncio.run(run_variant(args.store, args.variant, options))


if __name__ == "__main__":
    main()
