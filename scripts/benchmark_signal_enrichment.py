"""Replay fixed private cohorts through the signal worker without graph writes.

Raw completions are validated before any sanitizer or wrapper coercion. The
production prompt, schema, grouping and result matching are exercised; graph
claims, context reads and result writes are replaced by private trial boundaries.
The worker does not predict units: its incoming authoritative unit is retained.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
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


def encoded(value) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)


def digest(value) -> str:
    return hashlib.sha256(encoded(value).encode()).hexdigest()


def save(path: Path, value) -> None:
    path.write_text(encoded(value) + "\n")
    path.chmod(0o600)


def snapshot(ids: list[str]) -> list[dict]:
    """Export every stored property, bounded to one row per named signal."""
    with GraphClient() as gc:
        rows = gc.query(
            "UNWIND $ids AS id MATCH (s:FacilitySignal {id: id}) "
            "RETURN s.id AS id, properties(s) AS properties ORDER BY id LIMIT 200",
            ids=ids,
        )
    if len(rows) != len(ids) or {row["id"] for row in rows} != set(ids):
        raise ValueError("snapshot did not return exactly the named cohort")
    return rows


def prepare(root: Path, pilot_path: Path, facility: str) -> None:
    if (root / "cohort.json").exists():
        raise FileExistsError(
            "fixed cohort already exists; use another private directory"
        )
    with GraphClient() as gc:
        domains = gc.query(
            "MATCH (s:FacilitySignal {facility_id: $facility}) "
            "WHERE s.enriched_at IS NOT NULL AND s.description IS NOT NULL "
            "AND s.physics_domain IS NOT NULL "
            "RETURN s.physics_domain AS domain, count(*) AS count ORDER BY domain LIMIT 40",
            facility=facility,
        )
        buckets = []
        for domain in domains:
            rows = gc.query(
                "MATCH (s:FacilitySignal {facility_id: $facility}) "
                "WHERE s.enriched_at IS NOT NULL AND s.description IS NOT NULL "
                "AND s.physics_domain = $domain "
                "RETURN s.id AS id ORDER BY id LIMIT 200",
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
        raise ValueError("need 200 enriched signals across several domains")
    pilot = json.loads(pilot_path.read_text())
    pilot_ids = pilot["written_ids"]
    if len(pilot_ids) != 200 or len(set(pilot_ids)) != 200:
        raise ValueError("pilot checkpoint must name exactly 200 distinct signals")
    reference = snapshot(selected)
    iter_rows = snapshot(pilot_ids)
    cohorts = {facility: reference, "iter": iter_rows}
    # Even spacing across the round-robin domain ordering fixes the hand sample.
    hand_ids = {facility: selected[::5], "iter": sorted(pilot_ids)[::5]}
    document = {
        "cohorts": cohorts,
        "order": {facility: selected, "iter": sorted(pilot_ids)},
        "hand_ids": hand_ids,
        "selection": "sorted ids round-robin across all stored domains; every fifth for hand reading",
        "domains": domains,
        "pilot_checkpoint_digest": digest(pilot),
        "cohort_digest": digest(cohorts),
        "reference_digest": digest(reference),
        "sealed_at": time.time(),
    }
    save(root / "cohort.json", document)
    print(
        encoded(
            {
                "cohort_digest": document["cohort_digest"],
                "sizes": {k: len(v) for k, v in cohorts.items()},
            }
        ),
        flush=True,
    )


def input_signal(row: dict) -> dict:
    """Remove stored model answers while preserving catalogue metadata."""
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
    """A malformed response remains a failure even if repair could recover it."""
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


def assert_unchanged(before, after) -> None:
    if encoded(before) != encoded(after):
        raise ValueError("stored enrichment changed during the private trial")


async def replay_batch(
    signals: list[dict], facility: str, db: sqlite3.Connection, transport=None
) -> dict:
    """Use the real worker with private claims and sinks and a strict parser."""
    state = parallel.DataDiscoveryState(facility=facility, scanner_types=[])
    state.facility_config = get_facility(facility)
    state.cost_limit = float("inf")
    state.wiki_context = {}
    pending = [signals]
    captured = []
    requests = []
    outputs = []
    original_call = llm.acall_llm_structured
    original_transport = transport or llm._acompletion_local

    def claim(*args, **kwargs):
        if pending:
            return pending.pop()
        state.stop_requested = True
        return []

    async def capture_transport(kwargs):
        started = time.monotonic()
        response = await original_transport(kwargs)
        raw = response.choices[0].message.content
        captured.append(
            {
                "raw": raw,
                "score": score_response(raw, len(signals)),
                "served_model": response.model,
                "seconds": time.monotonic() - started,
                "finish_reason": response.choices[0].finish_reason,
            }
        )
        return response

    async def call_once(**kwargs):
        requests.append(kwargs["messages"])
        kwargs.update(max_retries=1, timeout=180)
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

    def refuse_graph(*args, **kwargs):
        raise AssertionError("private trial attempted a live graph operation")

    def strict_parse(content, model, model_id):
        return model.model_validate_json(content)

    started = time.monotonic()
    with ExitStack() as stack:
        substitutions = {
            "claim_signals_for_enrichment": claim,
            "prepare_signal_sources": lambda *a, **k: (0, 0, 0),
            "propagate_units_from_signal_nodes": lambda *a, **k: 0,
            "fetch_tree_context": lambda *a, **k: {},
            "fetch_epoch_context": lambda *a, **k: {},
            "fetch_signal_code_refs": lambda *a, **k: {},
            "_fetch_code_chunks": lambda *a, **k: [],
            "mark_signals_enriched": write,
            "mark_signals_underspecified": write,
            "propagate_source_enrichment": lambda *a, **k: 0,
            "release_signal_claim": lambda *a, **k: None,
            "judge_and_store_signal_roles": AsyncMock(return_value=(0, 0)),
            "GraphClient": refuse_graph,
        }
        for name, replacement in substitutions.items():
            stack.enter_context(patch.object(parallel, name, replacement))
        stack.enter_context(
            patch(
                "imas_codex.discovery.signals.scanners.wiki.fetch_semantic_wiki_context",
                return_value=[],
            )
        )
        stack.enter_context(patch.object(llm, "acall_llm_structured", call_once))
        stack.enter_context(patch.object(llm, "_acompletion_local", capture_transport))
        stack.enter_context(patch.object(llm, "_sanitize_content", lambda value: value))
        stack.enter_context(
            patch.object(llm, "_parse_structured_content", strict_parse)
        )
        await parallel.enrich_worker(state)
    by_id = {signal["id"]: signal for signal in signals}
    for output in outputs:
        output["unit"] = by_id[output["id"]].get("unit")
    record = {
        "facility": facility,
        "ids": list(by_id),
        "responses": captured,
        "messages": requests,
        "outputs": outputs,
        "seconds": time.monotonic() - started,
        "cost_usd": state.enrich_stats.cost,
    }
    with db:
        db.execute("INSERT INTO batches(document) VALUES (?)", (encoded(record),))
    return record


def open_store(path: Path) -> sqlite3.Connection:
    db = sqlite3.connect(path)
    path.chmod(0o600)
    db.execute("CREATE TABLE IF NOT EXISTS batches(document TEXT NOT NULL)")
    return db


async def run(root: Path) -> None:
    cohort = json.loads((root / "cohort.json").read_text())
    if digest(cohort["cohorts"]) != cohort["cohort_digest"]:
        raise ValueError("sealed cohort digest mismatch")
    if not get_model("discovery-describe").startswith("local/"):
        raise ValueError(
            "benchmark requires the locally served discovery-describe seat"
        )
    db = open_store(root / "trial.sqlite")
    if db.execute("SELECT count(*) FROM batches").fetchone()[0]:
        raise ValueError("trial already contains responses; refusing a mixed rerun")
    before = {
        facility: snapshot(cohort["order"][facility]) for facility in cohort["cohorts"]
    }
    for facility in before:
        assert_unchanged(cohort["cohorts"][facility], before[facility])
    save(root / "before.json", before)
    for facility, ids in cohort["order"].items():
        rows = {row["id"]: row for row in cohort["cohorts"][facility]}
        for offset in range(0, len(ids), 20):
            signals = [input_signal(rows[id]) for id in ids[offset : offset + 20]]
            record = await replay_batch(signals, facility, db)
            print(
                encoded(
                    {
                        "facility": facility,
                        "offset": offset,
                        "outputs": len(record["outputs"]),
                        "responses": [
                            response["score"] for response in record["responses"]
                        ],
                        "seconds": record["seconds"],
                    }
                ),
                flush=True,
            )
    after = {
        facility: snapshot(cohort["order"][facility]) for facility in cohort["cohorts"]
    }
    save(root / "after.json", after)
    for facility in before:
        assert_unchanged(before[facility], after[facility])
    save(
        root / "integrity.json",
        {
            facility: {
                "before": digest(before[facility]),
                "after": digest(after[facility]),
                "identical": True,
            }
            for facility in before
        },
    )
    db.close()


def judge(root: Path) -> None:
    cohort = json.loads((root / "cohort.json").read_text())
    db = open_store(root / "trial.sqlite")
    records = [json.loads(row[0]) for row in db.execute("SELECT document FROM batches")]
    rows = [
        output
        for record in records
        if record["facility"] != "iter"
        for output in record["outputs"]
    ]
    reference = {
        row["id"]: row["properties"]
        for facility, rows in cohort["cohorts"].items()
        if facility != "iter"
        for row in rows
    }
    judged = []
    total = 0.0
    releases = []
    for offset in range(0, len(rows), 20):
        batch = rows[offset : offset + 20]
        examples = {
            f"signal_{i}": {
                "source": input_signal(
                    {"id": row["id"], "properties": reference[row["id"]]}
                ),
                "reference": reference[row["id"]]["description"],
                "candidate": row["description"],
            }
            for i, row in enumerate(batch)
        }
        criteria = {
            "equivalent": "Preserves the reference's physical quantity, role, important qualifiers and useful information, or improves them with support in source metadata.",
            "weaker": "Consistent with the reference but loses meaningful specificity or leaves the physical quantity or role vague.",
            "wrong": "Changes or contradicts quantity, role, units, geometry or physical interpretation, or adds unsupported factual specificity.",
        }
        questions = {
            key: {
                "type": "choice",
                "instructions": f"Compare candidate with reference for {key}. Use source only as evidence; do not treat model style as quality. References are an incumbent comparison, not independent truth.",
                "criteria": criteria,
            }
            for key in examples
        }
        answers, cost = llm.call_decisions(
            model=get_model("discovery-relevance"),
            state=examples,
            questions=questions,
            service="facility-discovery",
            max_retries=1,
            record_release=releases.append,
        )
        total += cost
        for i, row in enumerate(batch):
            judged.append(
                {
                    "id": row["id"],
                    "category": answers[f"signal_{i}"]["choice"],
                    "answer": answers[f"signal_{i}"],
                }
            )
        save(
            root / "judge.json",
            {
                "model": get_model("discovery-relevance"),
                "served_releases": releases,
                "cost_usd": total,
                "rows": judged,
            },
        )
        print(encoded({"judged": len(judged), "cost_usd": total}), flush=True)


def summarize(root: Path) -> None:
    cohort = json.loads((root / "cohort.json").read_text())
    db = open_store(root / "trial.sqlite")
    records = [json.loads(row[0]) for row in db.execute("SELECT document FROM batches")]
    summary = {}
    for facility, originals in cohort["cohorts"].items():
        batches = [record for record in records if record["facility"] == facility]
        responses = [response for record in batches for response in record["responses"]]
        outputs = {
            output["id"]: output for record in batches for output in record["outputs"]
        }
        refs = {row["id"]: row["properties"] for row in originals}
        seconds = sum(record["seconds"] for record in batches)
        summary[facility] = {
            "selected": len(originals),
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
            "cost_usd": sum(record["cost_usd"] for record in batches),
            "domain_agreement": sum(
                output["physics_domain"] == refs[id].get("physics_domain")
                for id, output in outputs.items()
            ),
            "unit_agreement": sum(
                output.get("unit") == refs[id].get("unit")
                for id, output in outputs.items()
            ),
            "known_unit_count": sum(bool(row.get("unit")) for row in refs.values()),
            "stored_domains": dict(
                Counter(row.get("physics_domain") for row in refs.values())
            ),
            "model": get_model("discovery-describe"),
            "served_models": sorted(
                {response.get("served_model", "") for response in responses}
            ),
        }
    judge_path = root / "judge.json"
    if judge_path.exists():
        judgments = json.loads(judge_path.read_text())
        summary["judge"] = {
            "categories": dict(Counter(row["category"] for row in judgments["rows"])),
            "cost_usd": judgments["cost_usd"],
            "model": judgments["model"],
            "served_releases": judgments["served_releases"],
        }
    save(root / "summary.json", summary)
    print(encoded(summary), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "run", "judge", "summarize"])
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--facility", default="jt-60sa")
    parser.add_argument("--pilot", type=Path)
    args = parser.parse_args()
    args.store.mkdir(parents=True, exist_ok=True, mode=0o700)
    if args.action == "prepare":
        if args.pilot is None:
            parser.error("prepare requires --pilot")
        prepare(args.store, args.pilot, args.facility)
    elif args.action == "run":
        asyncio.run(run(args.store))
    elif args.action == "judge":
        judge(args.store)
    else:
        summarize(args.store)


if __name__ == "__main__":
    main()
