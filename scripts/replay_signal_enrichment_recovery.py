"""Exercise archived failures through the production sink without graph writes."""

from __future__ import annotations

import argparse
import copy
import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

from imas_codex.discovery.base import llm
from imas_codex.discovery.signals.enrichment import EnrichmentOptions
from imas_codex.settings import get_model
from scripts import benchmark_signal_enrichment as bench
from scripts.refine_signal_enrichment import frozen_signal_order, subset_messages


async def replay(
    root: Path,
    archive: Path,
    arm: str,
    *,
    recover: bool,
    local_retry: bool,
    source_fallback: bool,
):
    import asyncio

    cohort = json.loads((root / "cohort.json").read_text())
    if cohort["facility"] == "iter":
        raise ValueError("non-ITER cohort required")
    with sqlite3.connect(f"file:{archive / 'trial.sqlite'}?mode=ro", uri=True) as db:
        records = [
            json.loads(row[0]) for row in db.execute("SELECT document FROM batches")
        ]
    archived = {
        row["ids"][0]: row for row in records if row["arm"] == "grounded-concise-review"
    }
    contexts = {
        id: row for row in records if row["arm"] == "context" for id in row["ids"]
    }
    refs = {row["id"]: row for row in cohort["reference"]}
    options = EnrichmentOptions.model_validate(next(iter(archived.values()))["options"])
    options = options.model_copy(
        update={
            "recovery_attempts": int(recover),
            "source_fallback": recover or source_fallback,
        }
    )
    sink = bench.open_store(root / f"{arm}.sqlite")
    all_records, new_calls = [], 0
    before = bench.snapshot(cohort["order"])
    bench.assert_unchanged(cohort["reference"], before)
    original = llm._acompletion_local
    for id in cohort["order"]:
        saved = copy.deepcopy(archived[id]["responses"])

        async def transport(kwargs, saved=saved):
            nonlocal new_calls
            schema = kwargs["response_format"]["json_schema"]["name"]
            matching = next(
                (
                    index
                    for index, row in enumerate(saved)
                    if row.get("response_schema", "SignalEnrichmentBatch") == schema
                ),
                None,
            )
            if matching is not None:
                response = saved.pop(matching)
                return SimpleNamespace(
                    model=response.get("served_model", get_model("discovery-describe")),
                    choices=[
                        SimpleNamespace(
                            message=SimpleNamespace(content=response["raw"]),
                            finish_reason=response.get("finish_reason", "stop"),
                        )
                    ],
                    usage=SimpleNamespace(
                        prompt_tokens=response.get("input_tokens", 0),
                        completion_tokens=response.get("output_tokens", 0),
                    ),
                )
            if local_retry:
                new_calls += 1
                return await original(kwargs)
            raise ValueError(
                "archive has no further response; retry transport was disabled"
            )

        context = contexts[id]
        original_messages = context["messages"][0]
        order = frozen_signal_order(
            original_messages,
            [bench.input_signal(refs[target]) for target in context["ids"]],
        )
        messages = subset_messages(original_messages, [order.index(id) + 1])
        # The frozen draft already includes its fitted task and grounding rules.
        # The stage itself applies those controls to the same baseline task.
        from imas_codex.discovery.signals.enrichment import fit_enrichment_prompt

        if options.fitted_prompt:
            messages[0]["content"] = fit_enrichment_prompt(messages[0]["content"])
        record = await bench.replay_batch(
            [bench.input_signal(refs[id])],
            cohort["facility"],
            sink,
            transport=transport,
            messages=messages,
            arm=arm,
            enrichment_options=options,
            call_options=options.call_kwargs(),
        )
        all_records.append(record)
        if len(all_records) % 25 == 0:
            print(json.dumps({"arm": arm, "completed": len(all_records)}), flush=True)
        await asyncio.sleep(0)
    sink.close()
    after = bench.snapshot(cohort["order"])
    bench.assert_unchanged(before, after)
    summary = {
        "arm": arm,
        "signals": len(all_records),
        "delivered": sum(
            bool(row["outputs"] and row["outputs"][0].get("description", "").strip())
            for row in all_records
        ),
        "new_local_retry_calls": new_calls,
        "before": bench.digest(before),
        "after": bench.digest(after),
        "identical": True,
    }
    bench.save(root / f"{arm}.json", summary)
    print(bench.encoded(summary), flush=True)
    if recover:
        assert summary["delivered"] == 200, summary
    else:
        assert summary["delivered"] < 200, summary


def main():
    import asyncio

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--recover", action="store_true")
    parser.add_argument("--local-retry", action="store_true")
    parser.add_argument("--source-fallback", action="store_true")
    args = parser.parse_args()
    asyncio.run(
        replay(
            args.store,
            args.archive,
            args.arm,
            recover=args.recover,
            local_retry=args.local_retry,
            source_fallback=args.source_fallback,
        )
    )


if __name__ == "__main__":
    main()
