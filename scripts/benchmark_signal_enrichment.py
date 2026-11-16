"""Compare signal descriptions with frozen production prompts and private sinks.

The context pass runs the production worker's normal graph and retrieval reads.
Both model arms replay its exact final messages through the same worker's call
and result-matching path. Raw parsing is scored before the production parser's
normal sanitization, so recovered output never improves the strict parse rate.
Units are authoritative input metadata, not a field the model predicts.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import re
import sqlite3
import time
from collections import Counter
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import AsyncMock, patch

from imas_codex.discovery.base import llm
from imas_codex.discovery.base.facility import get_facility
from imas_codex.discovery.signals import parallel
from imas_codex.discovery.signals.models import SignalEnrichmentBatch
from imas_codex.graph.client import GraphClient
from imas_codex.settings import get_model

MODEL_OVERRIDE = "IMAS_CODEX_DISCOVERY_DESCRIBE_MODEL"


def encoded(value) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)


def digest(value) -> str:
    return hashlib.sha256(encoded(value).encode()).hexdigest()


def save(path: Path, value) -> None:
    path.write_text(encoded(value) + "\n")
    path.chmod(0o600)


def assert_unchanged(before, after) -> None:
    if encoded(before) != encoded(after):
        raise ValueError("stored enrichment changed during the private trial")


def snapshot(ids: list[str]) -> list[dict]:
    with GraphClient() as gc:
        rows = gc.query(
            "UNWIND $ids AS id MATCH (s:FacilitySignal {id: id}) "
            "RETURN s.id AS id, properties(s) AS properties ORDER BY id LIMIT 200",
            ids=ids,
        )
    if len(rows) != len(ids) or {row["id"] for row in rows} != set(ids):
        raise ValueError("snapshot did not return exactly the named cohort")
    return rows


def prepare(root: Path, facility: str) -> None:
    if facility == "iter":
        raise ValueError("the comparison accepts a non-ITER reference facility only")
    if (root / "cohort.json").exists():
        raise FileExistsError("fixed cohort already exists")
    condition = (
        "s.enriched_at IS NOT NULL AND s.description IS NOT NULL "
        "AND s.physics_domain IS NOT NULL AND s.source_description IS NOT NULL "
        "AND trim(s.source_description) <> ''"
    )
    with GraphClient() as gc:
        domains = gc.query(
            "MATCH (s:FacilitySignal {facility_id: $facility}) WHERE "
            + condition
            + " RETURN s.physics_domain AS domain, count(*) AS count ORDER BY domain LIMIT 40",
            facility=facility,
        )
        buckets = []
        for domain in domains:
            rows = gc.query(
                "MATCH (s:FacilitySignal {facility_id: $facility}) WHERE "
                + condition
                + " AND s.physics_domain = $domain RETURN s.id AS id ORDER BY id LIMIT 200",
                facility=facility,
                domain=domain["domain"],
            )
            buckets.append([row["id"] for row in rows])
    selected = []
    while len(selected) < 200 and any(buckets):
        for bucket in buckets:
            if bucket and len(selected) < 200:
                selected.append(bucket.pop(0))
    if len(selected) != 200 or len(domains) < 3:
        raise ValueError(
            "need 200 enriched signals with source context across several domains"
        )
    reference = snapshot(selected)
    document = {
        "facility": facility,
        "reference": reference,
        "order": selected,
        "hand_ids": selected[::5],
        "domains": domains,
        "selection": "sorted ids round-robin across domains, nonempty catalogue description required; every fifth fixed for hand reading",
        "cohort_digest": digest(reference),
        "sealed_at": time.time(),
    }
    save(root / "cohort.json", document)
    print(
        encoded(
            {
                "cohort_digest": document["cohort_digest"],
                "size": len(reference),
                "domains": len(domains),
            }
        ),
        flush=True,
    )


def input_signal(row: dict) -> dict:
    """Remove stored model answers while retaining authoritative source context."""
    p = row["properties"]
    fields = (
        "id",
        "facility_id",
        "accessor",
        "data_source_name",
        "data_source_path",
        "discovery_source",
        "source_description",
        "unit",
        "data_class",
        "is_static",
        "tdi_quantity",
    )
    signal = {key: p.get(key) for key in fields}
    signal["name"] = p.get("data_source_path") or p.get("accessor") or row["id"]
    signal["description"] = ""
    return signal


def score_response(content: str | None, expected: int) -> dict:
    """A malformed response is a strict failure even if production recovers it."""
    result = {
        "parse_success": False,
        "empty": not bool(content and content.strip()),
        "matched": 0,
    }
    try:
        batch = SignalEnrichmentBatch.model_validate_json(content or "")
    except ValueError as exc:
        result["error"] = str(exc)
        return result
    indices = [row.signal_index for row in batch.results]
    result.update(
        parse_success=True,
        matched=len(set(indices) & set(range(1, expected + 1))),
        complete=sorted(indices) == list(range(1, expected + 1)),
    )
    return result


def open_store(path: Path) -> sqlite3.Connection:
    db = sqlite3.connect(path)
    path.chmod(0o600)
    db.execute("CREATE TABLE IF NOT EXISTS batches(document TEXT NOT NULL)")
    return db


async def replay_batch(
    signals: list[dict],
    facility: str,
    db: sqlite3.Connection,
    transport=None,
    *,
    messages=None,
    capture_context=False,
    arm="test",
    call_options=None,
    enrichment_options=None,
) -> dict:
    """Exercise production prompting or replay one exact captured request."""
    import litellm

    from imas_codex.discovery.signals import enrichment
    from imas_codex.discovery.signals.enrichment import EnrichmentOptions

    if enrichment_options is None:
        enrichment_options = EnrichmentOptions()
    state = parallel.DataDiscoveryState(facility=facility, scanner_types=[])
    state.facility_config = get_facility(facility)
    state.cost_limit = float("inf")
    pending = [signals]
    captured, requests, outputs = [], [], []
    accounting = {"cost_usd": 0.0, "input_tokens": 0, "output_tokens": 0}
    original_call = llm.acall_llm_structured
    original_local = transport or llm._acompletion_local
    original_paid = litellm.acompletion
    original_query = GraphClient.query
    original_enrich = enrichment.enrich_signal_batch
    grounding_counts = {}
    current_response_model = SignalEnrichmentBatch

    async def replay_enrichment(**kwargs):
        if messages is not None:
            kwargs["messages"] = messages
        result, cost, tokens = await original_enrich(**kwargs)
        grounding_counts.update(
            {
                key: getattr(result, key, 0)
                for key in (
                    "evidence_drops",
                    "review_drops",
                    "review_revisions",
                    "length_drops",
                )
            }
        )
        return result, cost, tokens

    def claim(*args, **kwargs):
        if pending:
            return pending.pop()
        state.stop_requested = True
        return []

    async def observe(awaitable, request_messages):
        started = time.monotonic()
        response = await awaitable
        raw = response.choices[0].message.content
        score = score_response(raw, len(signals))
        try:
            current_response_model.model_validate_json(raw)
            score["parse_success"] = True
        except (ValueError, TypeError):
            score["parse_success"] = False
        captured.append(
            {
                "raw": raw,
                "score": score,
                "response_schema": current_response_model.__name__,
                "served_model": response.model,
                "seconds": time.monotonic() - started,
                "finish_reason": response.choices[0].finish_reason,
                "validation_reask": bool(
                    request_messages
                    and request_messages[-1].get("role") == "user"
                    and isinstance(request_messages[-1].get("content"), str)
                    and request_messages[-1]["content"].startswith(
                        "The response failed validation:\n"
                    )
                ),
                "input_tokens": response.usage.prompt_tokens or 0,
                "output_tokens": response.usage.completion_tokens or 0,
                "reasoning_tokens": getattr(response.usage, "reasoning_tokens", 0),
                "reasoning": getattr(
                    response.choices[0].message, "reasoning_content", None
                ),
            }
        )
        accounting["input_tokens"] += response.usage.prompt_tokens or 0
        accounting["output_tokens"] += response.usage.completion_tokens or 0
        accounting["cost_usd"] += llm.extract_cost(
            response, model=get_model("discovery-describe")
        )
        return response

    async def local_call(kwargs):
        return await observe(original_local(kwargs), kwargs.get("messages", []))

    async def paid_call(**kwargs):
        return await observe(original_paid(**kwargs), kwargs.get("messages", []))

    async def call_once(**kwargs):
        nonlocal current_response_model
        current_response_model = kwargs["response_model"]
        requests.append(kwargs["messages"])
        if capture_context:
            return SignalEnrichmentBatch(results=[]), 0.0, 0
        kwargs.update(max_retries=1, timeout=180, max_tokens=8192)
        kwargs.update(call_options or {})
        try:
            return await original_call(**kwargs)
        except Exception as exc:
            if not captured:
                captured.append(
                    {
                        "raw": None,
                        "score": score_response(None, len(signals)),
                        "error": repr(exc),
                        "seconds": 0,
                    }
                )
            else:
                captured[-1]["error"] = repr(exc)
            raise

    def write(entries, *args, **kwargs):
        outputs.extend(entries)
        return len(entries)

    def guarded_query(client, cypher, **params):
        if not capture_context or re.search(
            r"\b(?:SET|MERGE|CREATE|DELETE|REMOVE|DROP|FOREACH)\b", cypher
        ):
            raise AssertionError("private trial attempted a live graph operation")
        return original_query(client, cypher, **params)

    started = time.monotonic()
    with ExitStack() as stack:
        stack.enter_context(
            patch(
                "imas_codex.discovery.signals.enrichment.get_enrichment_options",
                return_value=enrichment_options,
            )
        )
        replacements = {
            "claim_signals_for_enrichment": claim,
            "prepare_signal_sources": lambda *a, **k: (0, 0, 0),
            "propagate_units_from_signal_nodes": lambda *a, **k: 0,
            "mark_signals_enriched": write,
            "mark_signals_underspecified": write,
            "propagate_source_enrichment": lambda *a, **k: 0,
            "release_signal_claim": lambda *a, **k: None,
            "judge_and_store_signal_roles": AsyncMock(return_value=(0, 0)),
        }
        if not capture_context:
            replacements.update(
                fetch_tree_context=lambda *a, **k: {},
                fetch_epoch_context=lambda *a, **k: {},
                fetch_signal_code_refs=lambda *a, **k: {},
                _fetch_code_chunks=lambda *a, **k: [],
            )
            stack.enter_context(
                patch(
                    "imas_codex.discovery.signals.scanners.wiki.fetch_semantic_wiki_context",
                    return_value=[],
                )
            )
        for name, replacement in replacements.items():
            stack.enter_context(patch.object(parallel, name, replacement))
        stack.enter_context(patch.object(GraphClient, "query", guarded_query))
        stack.enter_context(patch.object(llm, "acall_llm_structured", call_once))
        stack.enter_context(
            patch.object(enrichment, "enrich_signal_batch", replay_enrichment)
        )
        stack.enter_context(patch.object(llm, "_acompletion_local", local_call))
        stack.enter_context(patch.object(litellm, "acompletion", paid_call))
        await parallel.enrich_worker(state)
    by_id = {signal["id"]: signal for signal in signals}
    for output in outputs:
        output["unit"] = by_id[output["id"]].get("unit")
    record = {
        "arm": arm,
        "options": enrichment_options.model_dump(),
        "facility": facility,
        "ids": list(by_id),
        "responses": captured,
        "messages": requests,
        "prompt_digest": digest(requests),
        "outputs": outputs,
        "seconds": time.monotonic() - started,
        "grounding_counts": grounding_counts,
        "context_cost_usd": state.enrich_stats.cost if capture_context else 0.0,
        **accounting,
    }
    with db:
        db.execute("INSERT INTO batches(document) VALUES (?)", (encoded(record),))
    return record


def load_records(root: Path) -> list[dict]:
    with open_store(root / "trial.sqlite") as db:
        return [
            json.loads(row[0]) for row in db.execute("SELECT document FROM batches")
        ]


async def capture(root: Path) -> None:
    cohort = json.loads((root / "cohort.json").read_text())
    if digest(cohort["reference"]) != cohort["cohort_digest"]:
        raise ValueError("sealed cohort digest mismatch")
    assert_unchanged(cohort["reference"], snapshot(cohort["order"]))
    if load_records(root):
        raise ValueError("store is not empty")
    refs = {row["id"]: row for row in cohort["reference"]}
    db = open_store(root / "trial.sqlite")
    for offset in range(0, 200, 10):
        ids = cohort["order"][offset : offset + 10]
        record = await replay_batch(
            [input_signal(refs[id]) for id in ids],
            cohort["facility"],
            db,
            capture_context=True,
            arm="context",
        )
        if len(record["messages"]) != 1:
            raise ValueError(
                "production context pass did not produce exactly one request"
            )
        print(
            encoded(
                {
                    "context_signals": offset + 10,
                    "prompt_digest": record["prompt_digest"],
                    "context_cost_usd": record["context_cost_usd"],
                }
            ),
            flush=True,
        )
    assert_unchanged(cohort["reference"], snapshot(cohort["order"]))


def budget_guard(pilot_cost: float, actual: float = 0.0) -> float:
    if not math.isfinite(pilot_cost) or pilot_cost <= 0:
        raise ValueError("paid pilot needs a positive measured cost before projection")
    estimate = pilot_cost * 20
    if estimate > 20 or actual + 2 * pilot_cost > 20:
        raise ValueError("paid comparison exceeds the twenty dollar ceiling")
    return estimate


async def model_arm(root: Path, arm: str, *, pilot=False) -> None:
    cohort = json.loads((root / "cohort.json").read_text())
    records = load_records(root)
    contexts = [record for record in records if record["arm"] == "context"]
    if len(contexts) != 20:
        raise ValueError("need all 200 frozen full-context inputs before either arm")
    refs = {row["id"]: row for row in cohort["reference"]}
    db = open_store(root / "trial.sqlite")
    done = [record for record in records if record["arm"] == arm]
    local = get_model("discovery-describe")
    paid = get_model("sn-docs")
    if not local.startswith("local/") or not paid.startswith("openrouter/"):
        raise ValueError(
            "configured local describe and paid documentation seats required"
        )
    model = local if arm == "local" else paid
    if pilot and done:
        raise ValueError("paid pilot already exists")
    if not pilot and arm == "paid":
        estimate = json.loads((root / "estimate.json").read_text())
        budget_guard(estimate["pilot_cost_usd"], sum(row["cost_usd"] for row in done))
    assert_unchanged(cohort["reference"], snapshot(cohort["order"]))
    with patch.dict(os.environ, {MODEL_OVERRIDE: model}):
        for context in contexts[:1] if pilot else contexts:
            if any(row["ids"] == context["ids"] for row in done):
                continue
            if arm == "paid" and not pilot:
                budget_guard(
                    estimate["pilot_cost_usd"], sum(row["cost_usd"] for row in done)
                )
            record = await replay_batch(
                [input_signal(refs[id]) for id in context["ids"]],
                cohort["facility"],
                db,
                messages=context["messages"][0],
                arm=arm,
            )
            if record["prompt_digest"] != context["prompt_digest"]:
                raise ValueError(
                    "model arm did not use exactly the frozen production messages"
                )
            done.append(record)
            print(
                encoded(
                    {
                        "arm": arm,
                        "signals_attempted": sum(len(row["ids"]) for row in done),
                        "strict_parses": sum(
                            response["score"]["parse_success"]
                            for row in done
                            for response in row["responses"]
                        ),
                        "pipeline_outputs": sum(len(row["outputs"]) for row in done),
                        "cost_usd": sum(row["cost_usd"] for row in done),
                    }
                ),
                flush=True,
            )
    after = snapshot(cohort["order"])
    assert_unchanged(cohort["reference"], after)
    save(
        root / f"integrity-{arm}.json",
        {"before": cohort["cohort_digest"], "after": digest(after), "identical": True},
    )
    if pilot:
        cost = done[0]["cost_usd"]
        projected = budget_guard(cost)
        save(
            root / "estimate.json",
            {
                "pilot_signals": 10,
                "pilot_cost_usd": cost,
                "projected_200_usd": projected,
                "doubled_projection_usd": projected * 2,
                "ceiling_usd": 20,
                "model": paid,
                "reason": "Existing sn-docs seat writes project descriptions through OpenRouter; only the discovery-describe environment override changes.",
                "input_tokens": done[0]["input_tokens"],
                "output_tokens": done[0]["output_tokens"],
            },
        )
        print(encoded(json.loads((root / "estimate.json").read_text())), flush=True)


def pair_rows(root: Path) -> list[dict]:
    cohort = json.loads((root / "cohort.json").read_text())
    records = load_records(root)
    outputs = {
        arm: {
            output["id"]: output
            for record in records
            if record["arm"] == arm
            for output in record["outputs"]
        }
        for arm in ("local", "paid")
    }
    contexts = {
        id: record["messages"][0]
        for record in records
        if record["arm"] == "context"
        for id in record["ids"]
    }
    signals = {row["id"]: input_signal(row) for row in cohort["reference"]}
    rows = []
    for id in cohort["order"]:
        first_arm = (
            "local" if int(hashlib.sha256(id.encode()).hexdigest(), 16) % 2 else "paid"
        )
        second_arm = "paid" if first_arm == "local" else "local"
        rows.append(
            {
                "id": id,
                "first_arm": first_arm,
                "context": contexts[id],
                "signal": signals[id],
                "first": outputs[first_arm].get(id, {}).get("description", ""),
                "second": outputs[second_arm].get(id, {}).get("description", ""),
            }
        )
    return rows


def target_context(messages: list[dict], accessor: str) -> str:
    """Retain shared retrieval context and exactly the named signal block."""
    text = next(row["content"] for row in messages if row["role"] == "user")
    match = re.search(r"^accessor: " + re.escape(accessor) + r"$", text, re.M)
    if match is None:
        raise ValueError("target accessor absent from production context")
    signal_headers = list(re.finditer(r"^### Signal \d+", text, re.M))
    preceding = [row.start() for row in signal_headers if row.start() < match.start()]
    if not preceding:
        raise ValueError("target has no signal block")
    start = preceding[-1]
    groups = list(
        re.finditer(
            r"^## (?:EDAS Category:|PPF DDA:|MDSplus Tree:|JPF Subsystem:|TDI Function:|JET Device Description:)",
            text,
            re.M,
        )
    )
    ends = [row.start() for row in signal_headers + groups if row.start() > start]
    end = min(ends) if ends else len(text)
    before = [row.start() for row in groups if row.start() < start]
    group_start = before[-1] if before else 0
    group_first = next(
        row.start() for row in signal_headers if row.start() >= group_start
    )
    global_prefix = text[: groups[0].start()] if groups else ""
    return global_prefix + text[group_start:group_first] + text[start:end]


def blind_state(row: dict, reverse: bool) -> dict:
    return {
        "signal": row["signal"],
        "context": target_context(row["context"], row["signal"]["accessor"]),
        "first": row["second"] if reverse else row["first"],
        "second": row["first"] if reverse else row["second"],
    }


def judge(root: Path) -> None:
    rows = pair_rows(root)
    cohort = json.loads((root / "cohort.json").read_text())
    total, judged, reversed_rows, releases = 0.0, [], [], []
    preference = {
        "first": "First is materially more accurate or useful with support in context.",
        "second": "Second is materially more accurate or useful with support in context.",
        "equivalent": "No material difference in supported accuracy and usefulness.",
    }
    wrong = {
        "neither": "Both are consistent with the context, even if vague.",
        "first": "Only first contradicts context or asserts unsupported factual specificity; an empty description is wrong.",
        "second": "Only second contradicts context or asserts unsupported factual specificity; an empty description is wrong.",
        "both": "Both contradict context or assert unsupported factual specificity, or both are empty.",
    }
    for reverse in (False, True):
        chosen = (
            [row for row in rows if row["id"] in cohort["hand_ids"]]
            if reverse
            else rows
        )
        for row in chosen:
            questions = {
                "preference": {
                    "type": "choice",
                    "instructions": "Compare first and second descriptions for the one target signal in state.signal against state.context. An empty description loses to a supported description. Prefer correct physical quantity and component role; unsupported factual specificity is worse than brevity. Ignore ordering and writing style.",
                    "criteria": preference,
                },
                "wrong": {
                    "type": "choice",
                    "instructions": "Independently check BOTH first and second descriptions for the target signal against state.signal and state.context. An empty description is wrong. Identify every empty description or unsupported factual claim, even when the other description is also wrong. Vague but supported descriptions are not wrong.",
                    "criteria": wrong,
                },
            }
            answers, cost = llm.call_decisions(
                model=get_model("discovery-relevance"),
                state=blind_state(row, reverse),
                questions=questions,
                service="facility-discovery",
                max_retries=1,
                record_release=releases.append,
            )
            total += cost
            first_arm = (
                ("paid" if row["first_arm"] == "local" else "local")
                if reverse
                else row["first_arm"]
            )
            second_arm = "paid" if first_arm == "local" else "local"
            choice = answers["preference"]["choice"]
            bad = answers["wrong"]["choice"]
            entry = {
                "id": row["id"],
                "preferred": {
                    "first": first_arm,
                    "second": second_arm,
                    "equivalent": "equivalent",
                }[choice],
                "blind_choice": choice,
                "wrong_arms": [
                    arm
                    for label, arm in (("first", first_arm), ("second", second_arm))
                    if bad in (label, "both")
                ],
                "answers": answers,
            }
            (reversed_rows if reverse else judged).append(entry)
            save(
                root / "judge.json",
                {
                    "model": get_model("discovery-relevance"),
                    "served_releases": sorted(set(releases)),
                    "cost_usd": total,
                    "rows": judged,
                    "reversed_rows": reversed_rows,
                },
            )
            print(
                encoded(
                    {
                        "judged": len(judged),
                        "reversed": len(reversed_rows),
                        "cost_usd": total,
                    }
                ),
                flush=True,
            )


def summarize(root: Path) -> None:
    cohort = json.loads((root / "cohort.json").read_text())
    records = load_records(root)
    refs = {row["id"]: row["properties"] for row in cohort["reference"]}
    summary = {
        "cohort_digest": cohort["cohort_digest"],
        "domains": dict(Counter(row["physics_domain"] for row in refs.values())),
        "shared_context_cost_usd": sum(
            row["context_cost_usd"] for row in records if row["arm"] == "context"
        ),
    }
    for arm in ("local", "paid"):
        batches = [record for record in records if record["arm"] == arm]
        responses = [response for record in batches for response in record["responses"]]
        outputs = {
            output["id"]: output for record in batches for output in record["outputs"]
        }
        seconds = sum(record["seconds"] for record in batches)
        cost = sum(record["cost_usd"] for record in batches)
        summary[arm] = {
            "selected": 200,
            "attempted": sum(len(record["ids"]) for record in batches),
            "responses": len(responses),
            "parse_successes": sum(
                response["score"]["parse_success"] for response in responses
            ),
            "failed_responses": sum(
                not response["score"]["parse_success"] or "error" in response
                for response in responses
            ),
            "empty_responses": sum(
                response["score"]["empty"] for response in responses
            ),
            "output_signals": len(outputs),
            "seconds": seconds,
            "calls_per_minute": len(responses) * 60 / seconds if seconds else 0,
            "cost_usd": cost,
            "cost_per_signal_usd": cost / 200,
            "input_tokens": sum(row["input_tokens"] for row in batches),
            "output_tokens": sum(row["output_tokens"] for row in batches),
            "domain_agreement": sum(
                output["physics_domain"] == refs[id].get("physics_domain")
                for id, output in outputs.items()
            ),
            "unit_agreement": sum(
                output.get("unit") == refs[id].get("unit")
                for id, output in outputs.items()
            ),
            "known_unit_count": sum(bool(row.get("unit")) for row in refs.values()),
            "served_models": sorted(
                {response.get("served_model", "") for response in responses}
            ),
        }
    if (root / "judge.json").exists():
        j = json.loads((root / "judge.json").read_text())
        original = {row["id"]: row for row in j["rows"]}
        summary["judge"] = {
            "preferences": dict(Counter(row["preferred"] for row in j["rows"])),
            "either_wrong": sum(bool(row["wrong_arms"]) for row in j["rows"]),
            "wrong_per_arm": dict(
                Counter(arm for row in j["rows"] for arm in row["wrong_arms"])
            ),
            "reverse_disagreements": sum(
                row["preferred"] != original[row["id"]]["preferred"]
                for row in j["reversed_rows"]
            ),
            "reverse_count": len(j["reversed_rows"]),
            "cost_usd": j["cost_usd"],
            "model": j["model"],
            "served_releases": j["served_releases"],
        }
    save(root / "summary.json", summary)
    print(encoded(summary), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=[
            "prepare",
            "capture",
            "estimate",
            "local",
            "paid",
            "judge",
            "summarize",
        ],
    )
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--facility", default="jt-60sa")
    args = parser.parse_args()
    args.store.mkdir(parents=True, exist_ok=True, mode=0o700)
    if args.action == "prepare":
        prepare(args.store, args.facility)
    elif args.action == "capture":
        asyncio.run(capture(args.store))
    elif args.action == "estimate":
        asyncio.run(model_arm(args.store, "paid", pilot=True))
    elif args.action in ("local", "paid"):
        asyncio.run(model_arm(args.store, args.action))
    elif args.action == "judge":
        judge(args.store)
    else:
        summarize(args.store)


if __name__ == "__main__":
    main()
