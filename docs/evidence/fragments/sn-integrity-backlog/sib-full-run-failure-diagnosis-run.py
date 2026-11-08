#!/usr/bin/env python3
"""Capture and diagnose terminal DD composition sources without graph writes."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import hashlib
import html
import importlib
import importlib.util
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import traceback
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from imas_codex.graph.client import GraphClient

RECORD_ID = Path(__file__).stem.removesuffix("-run")
TREE = Path(__file__).resolve().parents[4]

STATUS_QUERY = """
MATCH (s:StandardNameSource {source_type: 'dd'})
RETURN s.status AS status,
       EXISTS { MATCH (s)-[:PRODUCED_NAME]->(:StandardName) } AS produced,
       count(*) AS count ORDER BY status, produced
"""
SOURCE_QUERY = """
MATCH (s:StandardNameSource {source_type: 'dd'})
WHERE s.status IN ['failed', 'vocab_gap']
OPTIONAL MATCH (s)-[:PRODUCED_NAME]->(n:StandardName)
WITH s, collect(n { .id, .name_stage, .docs_stage, .unit, .description,
     .reviewer_score_name, .reviewer_score_docs, .successor,
     .created_at, .last_run_id }) AS names
OPTIONAL MATCH (s)-[:HAS_STANDARD_NAME_VOCAB_GAP]->(g:VocabGap)
RETURN s.id AS id, s.source_id AS path, s.status AS status,
       s.attempt_count AS attempts, s.claim_seq AS claim_seq,
       s.last_error AS last_error, s.failed_at AS failed_at,
       s.composed_at AS composed_at, s.batch_key AS batch_key,
       s.description AS description, s.dd_unit AS unit,
       s.produced_sn_id AS produced_sn_id, s.run_id AS run_id,
       s.parked_disposition AS parked_disposition, names,
       collect(g { .id, .segment, .token, .category, .editorial_disposition,
         .editorial_target, .editorial_reason }) AS gaps
ORDER BY id LIMIT 1000
"""
KEY_QUERY = """
MATCH (s:StandardNameSource {source_type: 'dd'})
RETURN count(*) AS candidates, count(s.id) AS with_id,
       count(s.source_id) AS with_source_id, count(s.status) AS with_status,
       count(s.last_error) AS with_last_error, count(s.attempt_count) AS with_attempts
"""
RUN_QUERY = """
MATCH (r:SNRun) WHERE r.id STARTS WITH '58a799f1'
RETURN r.id AS id, r.started_at AS started_at, r.ended_at AS ended_at,
       r.stop_reason AS stop_reason, r.status AS status,
       r.cost_spent AS cost_spent, r.cost_limit AS cost_limit,
       r.cost_total AS cost_total, r.cost_is_exact AS cost_is_exact,
       r.names_composed AS names_composed, r.names_enriched AS names_enriched,
       r.names_reviewed AS names_reviewed LIMIT 2
"""
EDGE_COUNT_QUERY = """
MATCH (s:StandardNameSource {source_type: 'dd'})
WHERE s.status IN ['failed', 'vocab_gap']
OPTIONAL MATCH (s)-[:PRODUCED_NAME]->(n:StandardName)
RETURN s.status AS status, n IS NOT NULL AS produced,
       count(*) AS expanded_rows, count(DISTINCT s) AS sources
ORDER BY status, produced
"""
PENDING_QUERY = """
MATCH (s:StandardNameSource {status: 'extracted'})
WHERE coalesce(s.attempt_count, 0) < 5
RETURN s.source_type AS source_type,
       EXISTS { MATCH (s)-[:PRODUCED_NAME]->(n:StandardName)
                WHERE n.name_stage IN ['approved', 'contested'] }
         AS frozen_binding,
       count(*) AS count, collect(s.id)[..5] AS examples
ORDER BY source_type, frozen_binding
"""
TARGET_QUERY = """
UNWIND $ids AS id
OPTIONAL MATCH (n:StandardName {id: id})
OPTIONAL MATCH (s:StandardNameSource)-[:PRODUCED_NAME]->(n)
WITH id, n, collect(s.source_id) AS sources
OPTIONAL MATCH (replacement:StandardName)-[:REFINED_FROM*1..8]->(n)
WHERE NOT (replacement.name_stage IN ['superseded', 'exhausted', 'contested'])
WITH id, n, sources, collect(DISTINCT replacement { .id, .name_stage, .unit,
     .reviewer_score_name }) AS refinements
OPTIONAL MATCH chain=(n)-[:HAS_SUCCESSOR*1..8]->(next:StandardName)
RETURN id, n.name_stage AS stage, n.unit AS unit, n.description AS description,
       n.reviewer_score_name AS review_score, sources, refinements,
       collect([x IN nodes(chain) | x { .id, .name_stage, .unit }]) AS chains
"""
AUTHORITY_QUERY = """
UNWIND $ids AS id
MATCH (s:StandardNameSource {id: id})
OPTIONAL MATCH (s)-[:FROM_DD_PATH]->(dd:IMASNode)
OPTIONAL MATCH (dd)-[:HAS_UNIT]->(u:Unit)
RETURN id, s.dd_unit AS pinned_unit, s.dd_documentation AS pinned_documentation,
       dd.id AS dd_path, coalesce(u.id, dd.unit) AS current_unit,
       dd.documentation AS current_documentation, dd.lifecycle_status AS lifecycle
ORDER BY id
"""
CLAIMABILITY_QUERY = """
MATCH (s:StandardNameSource)
RETURN count(*) AS sources,
       count(CASE WHEN EXISTS { MATCH (s)-[:FROM_DD_PATH]->(:IMASNode) }
                  THEN 1 END) AS dd_link_control,
       count(CASE WHEN s.status = 'extracted' AND coalesce(s.attempt_count, 0) < 5
                  THEN 1 END) AS pending,
       count(CASE WHEN s.status = 'extracted' AND coalesce(s.attempt_count, 0) < 5
                  AND EXISTS { MATCH (s)-[:FROM_DD_PATH]->(:IMASNode) }
                  AND NOT EXISTS { MATCH (s)-[:PRODUCED_NAME]->(n:StandardName)
                                   WHERE n.name_stage IN ['approved', 'contested'] }
                  THEN 1 END) AS seed_candidates
"""


class ReadOnlyQueries:
    """Reject mutating statements before delegating to the project client."""

    def __init__(self, client):
        self.client = client
        self.queries = []

    def query(self, cypher: str, **params):
        assert cypher.lstrip().startswith(("MATCH ", "UNWIND "))
        assert not re.search(r"\b(SET|MERGE|CREATE|DELETE|REMOVE|DROP|LOAD)\b", cypher)
        result = self.client.query(cypher, **params)
        self.queries.append({"cypher": cypher, "params": params, "rows": result})
        return result


def revision(tree: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(tree), "rev-parse", "HEAD"], text=True
    ).strip()


def capture(destination: Path, log_path: Path) -> None:
    import imas_codex.graph.client as client

    tree = Path(__file__).resolve().parents[4]
    snapshot = {
        "observed_at": datetime.now(UTC).isoformat(),
        "tree": str(tree),
        "revision": revision(tree),
        "measurement_module": client.__file__,
        "measurement_cwd": str(Path.cwd().resolve()),
        "client_revision": revision(Path(client.__file__).resolve().parents[2]),
        "log_path": str(log_path),
        "log_sha256": hashlib.sha256(log_path.read_bytes()).hexdigest(),
        "queries": {},
    }
    print(json.dumps({k: v for k, v in snapshot.items() if k != "queries"}))
    with GraphClient() as client_handle:
        gc = ReadOnlyQueries(client_handle)
        for name, query in (
            ("keys", KEY_QUERY),
            ("status_before", STATUS_QUERY),
            ("sources", SOURCE_QUERY),
            ("run", RUN_QUERY),
            ("status_after", STATUS_QUERY),
        ):
            assert query.lstrip().startswith("MATCH ")
            snapshot["queries"][name] = {"cypher": query, "rows": gc.query(query)}
            print(f"captured {name}: {len(snapshot['queries'][name]['rows'])} rows")
    destination.write_text(json.dumps(snapshot, indent=2, default=str) + "\n")
    keys = snapshot["queries"]["keys"]["rows"][0]
    assert keys["candidates"] == keys["with_id"] == keys["with_source_id"]
    assert keys["with_last_error"] > 0 and keys["with_attempts"] > 0
    before = snapshot["queries"]["status_before"]["rows"]
    assert before == snapshot["queries"]["status_after"]["rows"]
    sources = snapshot["queries"]["sources"]["rows"]
    counts = Counter((s["status"], bool(s["names"])) for s in sources)
    expected = {
        (r["status"], r["produced"]): r["count"]
        for r in before
        if r["status"] in ("failed", "vocab_gap")
    }
    assert dict(counts) == expected
    assert len(sources) < 1000 and len(snapshot["queries"]["run"]["rows"]) == 1
    print("COHORT " + json.dumps({str(k): v for k, v in counts.items()}))
    print("RUN " + json.dumps(snapshot["queries"]["run"]["rows"], default=str))


def supplement(snapshot_path: Path, destination: Path) -> None:
    from imas_codex.standard_names.attachment_audit import guard_source_pairings

    snapshot = json.loads(snapshot_path.read_text())
    sources = snapshot["queries"]["sources"]["rows"]
    targets = {
        match.group(1)
        for source in sources
        if (match := re.search(r'candidate "([^"]+)"', source["last_error"] or ""))
    }
    result = {"observed_at": datetime.now(UTC).isoformat(), "guards": {}}
    with GraphClient() as client:
        gc = ReadOnlyQueries(client)
        result["expanded_counts"] = gc.query(EDGE_COUNT_QUERY)
        result["pending"] = gc.query(PENDING_QUERY)
        result["claimability"] = gc.query(CLAIMABILITY_QUERY)
        result["authority"] = gc.query(AUTHORITY_QUERY, ids=[s["id"] for s in sources])
        result["targets"] = gc.query(TARGET_QUERY, ids=sorted(targets))
        for source in sources:
            error = source["last_error"] or ""
            if not any(
                part in error for part in ("attachment rejected", "binding collision")
            ):
                continue
            target = re.search(r'candidate "([^"]+)"', error).group(1)
            verdict = guard_source_pairings(gc, target, [source["id"]])
            result["guards"][source["id"]] = {
                "target": target,
                "accepted": list(verdict.accepted_source_ids),
                "rejected": [v.reason for v in verdict.rejected],
            }
        result["queries"] = gc.queries
    destination.write_text(json.dumps(result, indent=2, default=str) + "\n")
    print("EXPANDED " + json.dumps(result["expanded_counts"]))
    print("PENDING " + json.dumps(result["pending"]))
    print("CLAIMABILITY " + json.dumps(result["claimability"]))
    print(
        "GUARDS "
        + json.dumps(Counter(bool(v["rejected"]) for v in result["guards"].values()))
    )


def classify_error(source: dict) -> str:
    error = source["last_error"] or ""
    if "binding collision" in error:
        return "binding_collision"
    if "attachment rejected" in error:
        return "attachment_rejected"
    if "vocabulary" in error or source["status"] == "vocab_gap":
        return "vocabulary_gap"
    if "attempt cap" in error:
        return "attempt_cap"
    if not error:
        return "blank"
    raise AssertionError(f"Unclassified error for {source['id']}: {error}")


def parse_log(log_path: Path, sources: list[dict]) -> dict:
    lines = log_path.read_text().splitlines()
    stamp = re.compile(r"^\d\d:\d\d:\d\d - ")
    exception = re.compile(r"^[\w.]+(?:Error|Exception):")
    path_pattern = re.compile(
        r"(?<![a-zA-Z0-9_/])(?:"
        + "|".join(
            re.escape(s["path"]) for s in sorted(sources, key=lambda s: -len(s["path"]))
        )
        + r")(?![a-zA-Z0-9_/])"
    )
    mentions = {s["path"]: [] for s in sources}
    quotes = {}
    batches = []
    for index, line in enumerate(lines):
        number = index + 1
        hits = {m.group() for m in path_pattern.finditer(line)}
        for path in hits:
            mentions[path].append(number)
        if hits:
            quotes[str(number)] = line
        if "Pool generate_name: compose batch failed" not in line:
            continue
        end = next(
            (j for j in range(index + 1, len(lines)) if stamp.match(lines[j])),
            len(lines),
        )
        block = lines[index:end]
        errors = [
            (index + j + 1, text)
            for j, text in enumerate(block)
            if exception.match(text)
        ]
        assert errors, f"No exception found after batch failure at {number}"
        last_line, final_error = errors[-1]
        if "c.compose_name()" in "\n".join(block):
            kind = "candidate_grammar_aborted_batch"
        elif any(
            "Connection error" in error or "httpx." in error for _, error in errors
        ):
            kind = "transport_aborted_batch"
        elif "LLMStructuredCallError" in final_error:
            kind = "structured_response_exhausted"
        else:
            kind = "rich_writer_refusal"
        batches.append(
            {
                "line": number,
                "exception_line": last_line,
                "kind": kind,
                "error": final_error,
            }
        )
    assert len(batches) == sum(
        " - ERROR - Pool generate_name: compose batch failed" in line for line in lines
    )
    assert batches and any(mentions.values())
    stalls = [
        {"line": i + 1, "text": line}
        for i, line in enumerate(lines)
        if "no forward progress" in line
    ]
    assert (
        len(stalls) == 1 and "600s" in stalls[0]["text"] and "104" in stalls[0]["text"]
    )
    return {
        "line_count": len(lines),
        "batches": batches,
        "batch_counts": dict(Counter(b["kind"] for b in batches)),
        "mentions": mentions,
        "quotes": quotes,
        "stall": stalls[0],
    }


RECOVERY = {
    "binding_now_admissible": "Likely to bind the same existing identity after an explicit retry reset: its stage is now stable and the live pairing guard accepts it. This adds coverage, not a new identity; review still applies.",
    "attachment_now_admissible": "Likely to bind after an explicit retry reset: the recorded missing-source refusal no longer reproduces. Keep a regression test for absent source versus absent target; no blanket guard bypass.",
    "registered_token_retry": "Plausible after an explicit retry with the recorded registered token/slot/decomposition supplied. The refusal itself says the vocabulary exists; no vocabulary extension is justified by this message.",
    "isolated_retry": "Possible after an explicit retry reset. Candidate isolation can preserve valid siblings, but the cap retained no source-to-batch history: a source-specific success probability is not identifiable.",
    "historical_cap_retry": "Uncertain: the terminal timestamp predates this run. Reset only a selected cohort with source-specific attempt receipts; this run's batch failure cannot be assigned retrospectively.",
    "legacy_unknown": "Unknown: blank last_error and a July terminal timestamp provide no reason for the failed attempts. A bounded, explicitly reset probe with durable per-source receipts is needed.",
    "binding_identity_review": "Unlikely to bind the same spelling on an unchanged retry. Preserve terminal identity history, inspect the recorded live replacement/holder, and supply the intended current identity to compose; split meanings when the source differs.",
    "binding_guard_review": "The stage now permits reuse but the current pairing guard refuses it. Settle the recorded semantic/unit refusal before retrying; candidate isolation alone does not change it.",
    "quantity_guard_fix": "Unlikely until the vector guard distinguishes coordinate components from toroidal flux. The DD says Toroidal flux [Wb], not an angle; profiles_1d and profiles_2d are representations of the quantity.",
    "semantic_split": "Unlikely with the same candidate. Preserve the guard and distinguish the owning centre, curvature centre, endpoint, or orientation vector in the name; review the existing source binding and use signed repair where it carries the wrong locus.",
    "unit_split": "Unlikely with the same candidate. Preserve the dimensionality refusal and separate an areal particle flux [m^-2.s^-1] from an integrated particle rate [s^-1]; repair the mislabeled identity/source through a signed route.",
    "vocabulary_review": "Unlikely from candidate isolation alone. Review the stated missing concept against the registered grammar and preserve its distinguishing meaning; genuine gaps need vocabulary decisions before source retry.",
    "malformed_gap_request": "Unlikely without correcting the request: a placeholder or comma-joined token is not evidence for a new physical concept. Recompose from the DD meaning and valid grammar slots before considering vocabulary changes.",
}


def recovery(source: dict, context: dict, targets: dict) -> str:
    error = source["last_error"] or ""
    kind = classify_error(source)
    if kind == "binding_collision":
        target = targets[re.search(r'candidate "([^"]+)"', error).group(1)]
        if target["stage"] in {"drafted", "reviewed", "accepted", "approved"}:
            return (
                "binding_guard_review"
                if context["guards"][source["id"]]["rejected"]
                else "binding_now_admissible"
            )
        return "binding_identity_review"
    if kind == "attachment_rejected":
        if not context["guards"][source["id"]]["rejected"]:
            return "attachment_now_admissible"
        if source["path"] == "equilibrium/time_slice/profiles_2d/phi":
            return "quantity_guard_fix"
        return "unit_split" if "dimensionality mismatch" in error else "semantic_split"
    if kind == "vocabulary_gap":
        if any(
            part in error
            for part in (
                "already a registered",
                "decomposes into the registered",
                "is a registered geometric_base",
            )
        ):
            return "registered_token_retry"
        if "placeholder" in error or "'neutral, momentum'" in error:
            return "malformed_gap_request"
        return "vocabulary_review"
    if kind == "blank":
        return "legacy_unknown"
    return (
        "isolated_retry"
        if (source["failed_at"] or "") >= "2026-11-07T20:36:28"
        else "historical_cap_retry"
    )


def cite(relative: str, needle: str) -> str:
    lines = (TREE / relative).read_text().splitlines()
    number = next((i + 1 for i, line in enumerate(lines) if needle in line), None)
    assert number is not None, (relative, needle)
    return f"{relative}:{number}"


def mechanism_references() -> dict:
    store = "imas_codex/standard_names/graph_store/graph_source_bindings.py"
    worker = (
        "imas_codex/standard_names/worker_tasks/worker_extraction_and_enrichment.py"
    )
    return {
        "binding_writer": cite(store, "THEN 'standard-name binding collision:"),
        "attachment_writer": cite(store, "THEN 'standard-name attachment rejected:"),
        "binding_guard": cite(store, "END AS lifecycle_bindable"),
        "binding_entry": cite(store, "def _lock_claimed_name_bindings("),
        "attachment_entry": cite(
            "imas_codex/standard_names/attachment_audit.py",
            "def guard_source_pairings(",
        ),
        "vector_guard": cite(worker, "def _vector_fields_conflict("),
        "vector_refusal": cite(worker, 'f"distinct-vector conflict:'),
        "unit_refusal": cite(worker, 'f"unit dimensionality mismatch:'),
        "missing_source_refusal": cite(
            "imas_codex/standard_names/attachment_audit.py",
            'reason="missing semantic source:',
        ),
        "cap_writer": cite(
            "imas_codex/standard_names/orphan_sweep.py",
            "THEN 'compose claim-attempt cap reached'",
        ),
        "claim_seed": cite(
            "imas_codex/standard_names/graph_store/graph_cost_and_claims.py",
            "MATCH (sns)-[:FROM_DD_PATH]->(imas0:IMASNode)",
        ),
        "pending_count": cite(
            "imas_codex/standard_names/graph_store/graph_parent_lifecycle.py",
            "RETURN count(s) AS generate_name",
        ),
        "stall": cite(
            "imas_codex/standard_names/pools.py", '"run_pools: no forward progress'
        ),
        "exit": cite(
            "imas_codex/cli/sn.py",
            'raise SystemExit(4 if stop_reason == "provider_budget_exhausted" else 1)',
        ),
        "candidate_isolation": cite(
            "imas_codex/standard_names/worker_tasks/worker_compose_batches.py",
            "except Exception as compose_exc:",
        ),
        "transport_retry": cite(
            "imas_codex/standard_names/worker_tasks/worker_compose_batches.py",
            "async def _acall_retrying_transport_faults(",
        ),
        "explicit_retry": cite(store, "def retry_failed_sources("),
    }


async def transport_controls() -> dict:
    import httpx
    import openai

    from imas_codex.standard_names.worker_tasks.worker_compose_batches import (
        _acall_retrying_transport_faults,
    )

    outcomes = {}
    for kind in ("bare_network_error", "wrapped_api_connection_error"):
        calls = 0

        async def call(error_kind=kind):
            nonlocal calls
            calls += 1
            if calls == 1:
                cause = httpx.ReadError("read interrupted")
                if error_kind == "wrapped_api_connection_error":
                    raise openai.APIConnectionError(
                        request=httpx.Request("POST", "https://invalid.example")
                    ) from cause
                raise cause
            return "response"

        try:
            answer = await _acall_retrying_transport_faults(
                call, attempts=2, base_delay=0
            )
            outcomes[kind] = {"calls": calls, "result": answer}
        except openai.APIConnectionError as exc:
            outcomes[kind] = {"calls": calls, "error": type(exc).__name__}
    assert outcomes["bare_network_error"] == {"calls": 2, "result": "response"}
    assert outcomes["wrapped_api_connection_error"] == {
        "calls": 1,
        "error": "APIConnectionError",
    }
    return outcomes


def attempt_history(rows: list[dict], log: dict) -> dict:
    """Count numbered outcomes separately from unnumbered grammar observations."""
    original_evidence = Counter(row["attempt_evidence"] for row in rows)
    for row in rows:
        expected = set(range(1, row["attempts"] + 1))
        accounted = set(row["refusal_attempts"])
        assert accounted <= expected, f"Claim number outside stored count: {row['id']}"
        row["unexplained_attempts"] = sorted(expected - accounted)
        grammar_lines = []
        for number in row["log_lines"]:
            match = re.search(
                r"composition retry \d+/\d+: \d+ grammar failures \((.*?)\)",
                log["quotes"][str(number)],
            )
            if match and row["id"].removeprefix("dd:") in match[1].split(", "):
                grammar_lines.append(number)
        if grammar_lines:
            row["grammar_lines"] = grammar_lines
        del row["attempt_evidence"]
    incomplete = [row for row in rows if row["unexplained_attempts"]]
    return {
        "complete": not incomplete,
        "source_count": len(rows),
        "complete_numbered_histories": len(rows) - len(incomplete),
        "incomplete_numbered_histories": len(incomplete),
        "stored_claims": sum(row["attempts"] for row in rows),
        "numbered_refusal_outcomes": sum(len(set(r["refusal_attempts"])) for r in rows),
        "unexplained_claims": sum(len(r["unexplained_attempts"]) for r in rows),
        "grammar_observed_sources": sum(bool(r.get("grammar_lines")) for r in rows),
        "sources_without_direct_attempt_event": sum(
            not r["refusal_lines"] and not r.get("grammar_lines") for r in rows
        ),
        "cap_sources_with_grammar_observation": sum(
            r["class"] == "attempt_cap" and bool(r.get("grammar_lines")) for r in rows
        ),
        "original_evidence_counts": dict(original_evidence),
        "grammar_limit": "The sanitized response names these claimed sources, but the retry message has no claim number. It proves at least one grammar failure, not the cause of every claim or the final batch outcome.",
        "numbered_limit": "Numbered refusal coverage is an upper bound on reconstructed history: the source stores no run/claim ledger with which to exclude counter resets. Unnumbered messages never fill numbered holes.",
    }


def retained_history_inputs(directory: Path, log: dict) -> dict:
    """Read the bounded receipt audit, requiring positive controls for absences."""
    receipt_path = directory / "attribution-receipts.json"
    application_path = directory / "application-log-attribution.json"
    receipts = json.loads(receipt_path.read_text())["queries"]
    application = json.loads(application_path.read_text())
    costs = receipts["compose_costs"]["rows"]
    cost_control = next(
        row
        for row in receipts["cost_presence"]["rows"]
        if row["phase"] == "generate_name"
    )
    assert 0 < len(costs) == cost_control["rows"] < 2000
    retries = receipts["cohort_retries"]["rows"]
    assert receipts["retry_presence"]["rows"][0]["with_error"] > 0
    assert 0 < len(retries) == len(receipts["retry_scalar_match"]["rows"]) < 1500
    assert {r["retry"]["id"] for r in retries} == {
        r["retry"]["id"] for r in receipts["retry_scalar_match"]["rows"]
    }
    messages = [
        event for item in application["logs"] for event in item["source_events"]
    ]
    assert messages
    payloads = {text.split(" - ", 2)[-1] for text in log["quotes"].values()}
    assert all(event["text"].split(": ", 1)[-1] in payloads for event in messages)
    return {
        "inputs": [
            {"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
            for p in (receipt_path, application_path)
        ],
        "compose_cost_rows": len(costs),
        "distinct_reused_group_labels": len({row["batch_id"] for row in costs}),
        "cost_property_sets": cost_control["property_sets"],
        "cohort_retry_receipts": len(retries),
        "latest_cohort_retry": max(row["retry"]["retried_at"] for row in retries),
        "application_source_messages": len(messages),
        "application_source_messages_absent_from_job_log": 0,
        "application_logs": [
            {
                k: v
                for k, v in item.items()
                if k not in {"source_events", "unique_events"}
            }
            for item in application["logs"]
        ],
        "limitation": "Application rotations can contain concurrent processes; timestamps and reused group labels are not a source-to-claim join. Cost receipts have candidate names, not source IDs or claim tokens. Retry receipts all predate the full run.",
    }


def require_complete_attempt_history(result: dict) -> None:
    """Refuse completion when any source lacks a numbered attempt outcome."""
    missing = [
        set(range(1, row["attempts"] + 1)) - set(row["refusal_attempts"])
        for row in result["sources"]
    ]
    missing = [numbers for numbers in missing if numbers]
    if missing:
        count = sum(map(len, missing))
        raise RuntimeError(
            f"Attempt history incomplete: {len(missing)}/{len(result['sources'])} "
            f"sources have {count} unexplained claims; terminal reasons and "
            "unassigned batch exceptions do not satisfy per-source attribution"
        )


def report(snapshot_path: Path, context_path: Path, log_path: Path) -> None:
    snapshot = json.loads(snapshot_path.read_text())
    context = json.loads(context_path.read_text())
    assert hashlib.sha256(log_path.read_bytes()).hexdigest() == snapshot["log_sha256"]
    sources = snapshot["queries"]["sources"]["rows"]
    assert len({s["id"] for s in sources}) == len(sources)
    assert len(sources) == 371
    assert (
        snapshot["queries"]["status_before"]["rows"]
        == snapshot["queries"]["status_after"]["rows"]
    )
    targets = {t["id"]: t for t in context["targets"]}
    authority = {}
    for row in context["authority"]:
        authority.setdefault(row["id"], []).append(row)
    assert set(authority) == {s["id"] for s in sources}
    log = parse_log(log_path, sources)
    rows = []
    for source in sources:
        refs = log["mentions"][source["path"]]
        refusals = [
            n
            for n in refs
            if f"Released source {source['id']} " in log["quotes"][str(n)]
        ]
        kind = classify_error(source)
        error = source["last_error"] or ""
        quote_attempts = [
            int(
                re.search(r"\(attempt (\d+) retained\)", log["quotes"][str(n)]).group(1)
            )
            for n in refusals
        ]
        row = {
            "id": source["id"],
            "status": source["status"],
            "class": kind,
            "last_error": source["last_error"],
            "attempts": source["attempts"],
            "failed_at": source["failed_at"],
            "description": (
                authority[source["id"]][0]["pinned_documentation"]
                or source["description"]
                or ""
            )[:240],
            "unit": authority[source["id"]][0]["pinned_unit"],
            "names": [
                {k: n[k] for k in ("id", "name_stage", "reviewer_score_name")}
                for n in source["names"]
            ],
            "gaps": [g["id"] for g in source["gaps"]],
            "log_lines": refs,
            "refusal_lines": refusals,
            "refusal_attempts": quote_attempts,
            "attempt_evidence": "direct_logged_refusals"
            if refusals
            else "terminal_reason_only"
            if error
            else "legacy_error_not_recorded",
            "recovery": recovery(source, context, targets),
        }
        if kind in {"binding_collision", "attachment_rejected"}:
            row["candidate"] = re.search(r'candidate "([^"]+)"', error).group(1)
            row["writer"] = (
                "binding_writer" if kind == "binding_collision" else "attachment_writer"
            )
        rows.append(row)
    history = attempt_history(rows, log)
    history_inputs = retained_history_inputs(snapshot_path.parent, log)
    class_counts = dict(sorted(Counter(r["class"] for r in rows).items()))
    recovery_counts = dict(sorted(Counter(r["recovery"] for r in rows).items()))
    assert sum(class_counts.values()) == 371
    assert class_counts == {
        "attachment_rejected": 11,
        "attempt_cap": 149,
        "binding_collision": 70,
        "blank": 58,
        "vocabulary_gap": 83,
    }
    assert sum(r["expanded_rows"] for r in context["expanded_counts"]) == 373
    assert context["claimability"][0]["pending"] == 104
    assert context["claimability"][0]["dd_link_control"] > 0
    assert context["claimability"][0]["seed_candidates"] == 0
    from imas_codex.cli.sn import _require_terminal_drain

    _require_terminal_drain("no_eligible_work")
    try:
        _require_terminal_drain("stalled")
    except SystemExit as exc:
        assert exc.code == 1
    else:
        raise AssertionError("stalled run did not refuse clean exit")
    transport = asyncio.run(transport_controls())
    try:
        ReadOnlyQueries(None).query("MATCH (s) SET s.status = 'changed' RETURN s")
    except AssertionError:
        mutation_refused = True
    else:
        raise AssertionError("Diagnostic query wrapper admitted a mutation")
    from imas_codex.standard_names.segments import classify_gap

    gap_records = {g["id"]: g for s in sources for g in s["gaps"]}
    gap_classes = {
        key: classify_gap(g["segment"], g["token"])[0] for key, g in gap_records.items()
    }
    gap_control = classify_gap("physical_base", "temperature")[0]
    assert gap_control == "false_positive"
    auto_revivable = [
        s["id"]
        for s in sources
        if s["status"] == "vocab_gap"
        and all(
            gap_classes[g["id"]]
            in {"false_positive", "invalid_segment", "open_segment"}
            for g in s["gaps"]
        )
    ]
    assert not auto_revivable
    accounting_command = [
        "sacct",
        "-j",
        "1308077",
        "--noheader",
        "--parsable2",
        "--format=JobID,State,ExitCode,Elapsed,Start,End",
    ]
    accounting = subprocess.check_output(accounting_command, text=True, timeout=20)
    assert any(
        line.startswith("1308077|FAILED|1:0|") for line in accounting.splitlines()
    )
    names = {
        n["id"]: {
            key: n.get(key)
            for key in (
                "id",
                "name_stage",
                "unit",
                "description",
                "reviewer_score_name",
            )
        }
        for s in sources
        for n in s["names"]
    }
    gaps = {
        g["id"]: {
            key: g.get(key)
            for key in (
                "id",
                "segment",
                "token",
                "category",
                "editorial_disposition",
                "editorial_target",
            )
        }
        for s in sources
        for g in s["gaps"]
    }
    imported_modules = {}
    for module_name in (
        "imas_codex.graph.client",
        "imas_codex.standard_names.attachment_audit",
        "imas_codex.standard_names.worker_tasks.worker_compose_batches",
        "imas_codex.standard_names.segments",
        "imas_codex.cli.sn",
    ):
        module = importlib.import_module(module_name)
        module_path = Path(module.__file__).resolve()
        relative = Path(*module_name.split(".")).with_suffix(".py")
        digest = hashlib.sha256(module_path.read_bytes()).hexdigest()
        assert digest == hashlib.sha256((TREE / relative).read_bytes()).hexdigest()
        imported_modules[module_name] = {
            "path": str(module_path),
            "sha256": digest,
            "matches_assigned_tree": True,
        }
    result = {
        "observed_at": snapshot["observed_at"],
        "product_revision": snapshot["client_revision"],
        "analysis_base_revision": revision(TREE),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "measurement_modules": imported_modules,
        "source_log": {
            "path": str(log_path),
            "sha256": snapshot["log_sha256"],
            "lines": log["line_count"],
        },
        "snapshot": {
            "path": str(snapshot_path),
            "sha256": hashlib.sha256(snapshot_path.read_bytes()).hexdigest(),
        },
        "context": {
            "path": str(context_path),
            "sha256": hashlib.sha256(context_path.read_bytes()).hexdigest(),
        },
        "module": snapshot["measurement_module"],
        "module_revision": snapshot["client_revision"],
        "class_counts": class_counts,
        "recovery_counts": recovery_counts,
        "status_query": STATUS_QUERY,
        "status_counts": snapshot["queries"]["status_after"]["rows"],
        "expanded_query": EDGE_COUNT_QUERY,
        "expanded_counts": context["expanded_counts"],
        "key_control": snapshot["queries"]["keys"]["rows"],
        "run": snapshot["queries"]["run"]["rows"][0],
        "pending_query": PENDING_QUERY,
        "pending": context["pending"],
        "claimability_query": CLAIMABILITY_QUERY,
        "claimability": context["claimability"],
        "stall": log["stall"],
        "cli_stalled_exit_control": 1,
        "batch_counts": log["batch_counts"],
        "batch_failures": log["batches"],
        "transport_controls": transport,
        "mutation_query_refused_before_client_call": mutation_refused,
        "vocabulary_reconcile_preview": {
            "gap_count": len(gap_classes),
            "categories": dict(Counter(gap_classes.values())),
            "registered_temperature_control": gap_control,
            "would_requeue_source_ids": auto_revivable,
        },
        "scheduler_receipt": {
            "command": shlex.join(accounting_command),
            "stdout": accounting,
        },
        "mechanisms": mechanism_references(),
        "recovery_legend": RECOVERY,
        "targets": context["targets"],
        "attachment_rechecks": {
            s["id"]: context["guards"][s["id"]]
            for s in sources
            if classify_error(s) == "attachment_rejected"
        },
        "gaps": gaps,
        "sources": rows,
        "attempt_history": history,
        "retained_history_inputs": history_inputs,
        "batch_attribution_for_every_source": "No failed-batch record contains source membership or claim token; source-to-batch attribution cannot be recovered. Source-specific refusal lines, where present, are independent evidence.",
        "spend_usd": 0,
    }
    likely = [
        r
        for r in rows
        if r["recovery"] in {"binding_now_admissible", "attachment_now_admissible"}
    ]
    plausible = [
        r for r in rows if r["recovery"] in {"isolated_retry", "registered_token_retry"}
    ]
    result["yield"] = {
        "unchanged_run_expected_sources": 0,
        "reason": "Every cohort source is failed or vocab_gap; claims select extracted. Fixed code alone does not reset terminal source state.",
        "likely_existing_binding_opportunities_after_explicit_reset": len(likely),
        "likely_unproduced": sum(not r["names"] for r in likely),
        "plausible_source_retry_opportunities": len(plausible),
        "plausible_unproduced": sum(not r["names"] for r in plausible),
        "accepted_new_names_guaranteed": 0,
        "new_identity_ceiling_in_plausible_unproduced_cohort": sum(
            not r["names"] for r in plausible
        ),
        "limitation": "These are source opportunities, not expected accepted-name counts. No calibrated source-level success rate or failed-batch membership was persisted; a narrower numeric expectation would be invented. Existing identity reuse and review failures lower new-name yield.",
    }
    evidence_dir = Path(__file__).parent
    output_json = evidence_dir / f"{RECORD_ID}.json"
    data = json.dumps(result, ensure_ascii=False, separators=(",", ":")) + "\n"
    assert len(data.encode()) <= 300000, "Repository JSON exceeds size limit"
    output_json.write_text(data)
    full = {"log_evidence": log, "analysis": result, "produced_names": names}
    (snapshot_path.parent / "full-analysis.json").write_text(
        json.dumps(full, indent=2, ensure_ascii=False) + "\n"
    )
    write_html(result, evidence_dir / f"{RECORD_ID}.html")
    persisted = json.loads(output_json.read_text())
    assert {r["id"] for r in persisted["sources"]} == {s["id"] for s in sources}
    rendered = (evidence_dir / f"{RECORD_ID}.html").read_text()
    rendered_ids = re.findall(r'<tr data-source="([^"]+)">', rendered)
    assert rendered_ids == [r["id"] for r in rows]
    ids = re.findall(r'\bid="([^"]+)"', rendered)
    assert len(ids) == len(set(ids)) and all(i.startswith(RECORD_ID) for i in ids)
    assert any(
        "Released source dd:equilibrium/time_slice/profiles_2d/phi " in quote
        for quote in log["quotes"].values()
    )
    print(
        "SUMMARY "
        + json.dumps(
            {
                "sources": len(rows),
                "attempt_history": history,
                "edge_expanded_rows": 373,
                "classes": class_counts,
                "recoveries": recovery_counts,
                "batch_failures": log["batch_counts"],
                "yield": result["yield"],
                "usd": 0,
            }
        )
    )
    require_complete_attempt_history(persisted)


def write_html(result: dict, destination: Path) -> None:
    def esc(value):
        return html.escape(str(value))

    parts = [
        f'<section id="{RECORD_ID}-record">',
        f'<h3 id="{RECORD_ID}-title">Full DD run: incomplete attempt diagnosis</h3>',
    ]
    parts.append(
        "<p><strong>Incomplete: the last-error census covers all 371 distinct sources, but their consumed-attempt causes are not fully recoverable from the retained inputs.</strong> There are 319 failed sources and 52 vocabulary gaps. The supplied 373 is an edge-expanded count: two failed sources each link to two names, yielding 30 produced rows for 28 sources. The independent before/after source counts agree. This diagnosis made no graph writes or paid calls (USD 0).</p>"
    )
    parts.append(
        f'<p>Observed {esc(result["observed_at"])}; product revision <code>{result["product_revision"]}</code>; evidence base <code>{result["analysis_base_revision"]}</code>; runner SHA-256 <code>{result["runner_sha256"]}</code>. Imported product modules match the assigned tree byte-for-byte; the JSON records their absolute paths and hashes. The JSON contains every source, complete last_error, attempt count, source-specific log lines, current binding evidence, recovery classification and mechanism references. <a href="/imas-codex/evidence/fragments/sn-integrity-backlog/{RECORD_ID}.json">Machine-readable classification</a>.</p>'
    )
    parts.append(
        f'<h4 id="{RECORD_ID}-counts">Cause and recoverability</h4><table><thead><tr><th>Last-error class</th><th>Sources</th><th>Unproduced</th><th>Produced</th></tr></thead><tbody>'
    )
    for kind, count in result["class_counts"].items():
        unproduced = sum(
            r["class"] == kind and not r["names"] for r in result["sources"]
        )
        parts.append(
            f"<tr><td>{esc(kind)}</td><td>{count}</td><td>{unproduced}</td><td>{count - unproduced}</td></tr>"
        )
    parts.append(
        "</tbody></table><p>The vocabulary class includes 52 sources in vocab_gap and 31 failed sources whose last error reports a vocabulary problem. Blank errors belong to 57 terminal records dated 28 July and one dated 27 July; they are not evidence of failures caused by this run. Of 149 bare attempt-cap errors, 121 were stamped during this run and 28 predate it.</p>"
    )
    parts.append(f'<h4 id="{RECORD_ID}-attempts">What consumed attempts</h4>')
    history = result["attempt_history"]
    parts.append(
        f"<p><strong>The completion gate refuses: {history['incomplete_numbered_histories']} of 371 sources have {history['unexplained_claims']} unexplained claims.</strong> The stored counters total {history['stored_claims']} claims; exact collision/attachment refusals account for {history['numbered_refusal_outcomes']} distinct source/attempt pairs across 60 sources. Only {history['complete_numbered_histories']} sources have every stored attempt number represented, and even these lack a durable run/claim ledger to exclude counter resets. The same parser finds the known toroidal-flux refusal at log line 4821 before reporting missing evidence elsewhere.</p>"
    )
    parts.append(
        f"<p>Source-specific grammar-retry messages add partial evidence for {history['grammar_observed_sources']} sources, including {history['cap_sources_with_grammar_observation']} of the 149 bare caps. These messages follow claimed-source sanitization, so they show a grammar failure for the named source; they contain no claim number and cannot fill numbered holes. The other 121 capped sources have no such grammar observation. In total, {history['sources_without_direct_attempt_event']} sources have neither a numbered refusal nor a source-specific grammar retry. The 58 blank historical errors remain unexplained.</p>"
    )
    parts.append(
        "<p>Each claim increments attempt_count before composition; the sweep parks the source at five claims. The final job log has 73 whole-batch exceptions. Their categories below are taken from complete exception blocks, not keyword matches elsewhere in the log. The earlier 58-entry note described an incomplete log.</p><ul>"
    )
    for key, value in result["batch_counts"].items():
        parts.append(f"<li>{value}: {esc(key)}</li>")
    parts.append(
        "</ul><p>Candidate isolation in merge <code>0198b31b3</code> catches compose_name failures per source. It cannot prove that every capped source was a valid sibling: batch exceptions do not carry source IDs or claim tokens. The source table therefore distinguishes direct logged refusals from retained terminal reasons and explicitly leaves batch membership unresolved. Source mentions such as out-of-batch proposals are not counted as consumed attempts.</p>"
    )
    retained = result["retained_history_inputs"]
    parts.append(
        f"<p>The bounded retained-input audit found {retained['compose_cost_rows']} compose-cost receipts using {retained['distinct_reused_group_labels']} reused group labels, with candidate names but no source IDs or claim tokens. The {retained['cohort_retry_receipts']} retry receipts matched by both relationship and scalar source ID all predate this run (latest {esc(retained['latest_cohort_retry'])}). Three application-log rotations contain {retained['application_source_messages']} exact cohort source messages; all already occur in the job log. These nonempty controls establish that the readers see real records. Their hashes, queries and matched messages are retained beside the manifest. Reused group labels or adjacent timestamps cannot assign concurrent batch failures to sources.</p>"
    )
    parts.append(
        "<p>A deterministic, network-free replay confirms bare httpx.ReadError is retried (two calls then success), while an OpenAI APIConnectionError wrapping that error escapes after one call. The final log has 29 APIConnectionError terminal exceptions and one structured-call wrapper with a connection error. The transport recovery assumption is therefore incomplete; the report does not treat those batches as guaranteed recoveries.</p>"
    )
    parts.append(
        f'<h4 id="{RECORD_ID}-yield">Second-pass yield</h4><p><strong>An ordinary second run over unchanged source state is expected to recover 0 of these sources.</strong> Claims select extracted sources; all 371 here are terminal failed or vocab_gap. An explicit retry/reset or vocabulary reconciliation is required first.</p><table><thead><tr><th>After a scoped explicit reset</th><th>Source opportunities</th></tr></thead><tbody>'
    )
    y = result["yield"]
    parts.append(
        f"<tr><td>Current same-identity binding passes (coverage, not new identities)</td><td>{y['likely_existing_binding_opportunities_after_explicit_reset']} ({y['likely_unproduced']} unproduced)</td></tr>"
    )
    parts.append(
        f"<tr><td>Run-time bare caps plus recorded reusable vocabulary: plausible, uncalibrated retry cohort</td><td>{y['plausible_source_retry_opportunities']} ({y['new_identity_ceiling_in_plausible_unproduced_cohort']} unproduced)</td></tr>"
    )
    parts.append(
        "</tbody></table><p>Accepted new-name yield has an evidence-based lower bound of 0. The unproduced plausible cohort gives only a ceiling on new identities, because multiple sources may reuse one name and review may refuse a candidate. The log contains no calibrated source-level recovery probability; a narrower point estimate would be invented. Historical caps, blank errors, identity collisions and unresolved vocabulary remain separate cohorts.</p>"
    )
    parts.append(
        "<p>The startup re-seeder preserves failed/vocab_gap states unless forced. A pure preview of vocabulary reconciliation checks all 72 linked gap identities and would requeue none of the 52 vocab_gap sources; the registered temperature token is the positive false-positive control. Thus the zero unchanged-run prediction includes the startup reconciliation path, not only the claim predicate.</p>"
    )
    parts.append(
        f'<h4 id="{RECORD_ID}-refusals">Refusing writers and proposed repairs</h4><p>The binding writer is <code>{esc(result["mechanisms"]["binding_writer"])}</code> in <code>_lock_claimed_name_bindings</code>; the attachment writer is <code>{esc(result["mechanisms"]["attachment_writer"])}</code>. Both retain attempt_count while releasing the claim; the cap writer is <code>{esc(result["mechanisms"]["cap_writer"])}</code>. A proposal and its current target/lineage are carried per source in the JSON.</p>'
    )
    parts.append(
        "<p>Binding collisions preserve the terminal identity guard. Feed the prior refused candidate, its reason and any semantically matching current holder back into composition; use signed identity repair where required rather than reopening tombstones indiscriminately. Twelve candidates now have drafted targets, so the last-error stage is historical, not a current refusal. Their pairing checks were rerun.</p>"
    )
    parts.append(
        "<p>For example, <code>turbulence/profiles_2d/neutral/element/a</code> means “Mass of atom” [u]. Its refused <code>neutral_atomic_mass</code> is superseded (last name score 0.825); the refinement lineage reaches accepted <code>neutral_species_atomic_mass</code> (score 0.875), which explicitly names the species. Confirm that species meaning before routing the source to this current identity. Likewise <code>poloidal_ion_state_momentum_diffusion_coefficient</code> has a recorded successor, accepted <code>poloidal_ion_charge_state_momentum_diffusion_coefficient</code>: charge_state makes the ion-state axis explicit. These alternatives are evidence-backed review targets, not authority to move bindings.</p>"
    )
    parts.append(
        "<table><thead><tr><th>Attachment case</th><th>Recheck and proposed correction</th></tr></thead><tbody>"
    )
    for r in result["sources"]:
        if r["class"] != "attachment_rejected":
            continue
        check = result["attachment_rechecks"][r["id"]]
        parts.append(
            f"<tr><td><code>{esc(r['id'])}</code><br>Candidate: <code>{esc(r['candidate'])}</code><br>{esc(r['description'])} [{esc(r['unit'])}]</td><td>{'Refusal reproduced' if check['rejected'] else 'Pairing now accepted'}. {esc(RECOVERY[r['recovery']])}</td></tr>"
        )
    parts.append("</tbody></table>")
    parts.append(
        f"<p>The vector rule is <code>{esc(result['mechanisms']['vector_guard'])}</code>, with refusal at <code>{esc(result['mechanisms']['vector_refusal'])}</code>; it compares same-axis leaves and their parent paths. It misreads <code>equilibrium/time_slice/profiles_2d/phi</code> as a vector component even though the pinned DD says “Toroidal flux”, unit Wb. The dimension rule at <code>{esc(result['mechanisms']['unit_refusal'])}</code> correctly refuses the wall flux/rate conflation. The missing-source branch at <code>{esc(result['mechanisms']['missing_source_refusal'])}</code> is historical: the current source/target exists and the pairing passes.</p>"
    )
    parts.append(
        f'<h4 id="{RECORD_ID}-stop">Why the run stalled and exited 1</h4><blockquote><p>Log line {result["stall"]["line"]}: {esc(result["stall"]["text"])}</p></blockquote>'
    )
    parts.append(
        f"<p>The pending counter (<code>{esc(result['mechanisms']['pending_count'])}</code>) counts extracted sources below five attempts without requiring a DD path. The claim seed (<code>{esc(result['mechanisms']['claim_seed'])}</code>) requires <code>FROM_DD_PATH</code>. A read-only reconstruction finds exactly 104 pending rows: 101 derived and 3 manual, with zero DD-linked seed candidates. The positive control finds 9,078 sources with DD links. Thus the counter stays positive while no batch can start; the 600-second watchdog fires at <code>{esc(result['mechanisms']['stall'])}</code>. This reconstruction matches the logged residual; it is a post-run read, not a stored snapshot of each row at shutdown.</p>"
    )
    parts.append(
        f"<p>The CLI maps stalled to exit 1 at <code>{esc(result['mechanisms']['exit'])}</code>; the diagnostic invokes that pure mapping and observes SystemExit(1), with no_eligible_work as its passing control. Scheduler accounting confirms FAILED, 1:0. The run spent USD 69.41506 of USD 350, leaving USD 280.58494: it stopped on the no-progress guard, not on a budget ceiling.</p>"
    )
    parts.append(f'<h4 id="{RECORD_ID}-recovery-key">Recovery classifications</h4><dl>')
    for key, count in result["recovery_counts"].items():
        parts.append(
            f"<dt><code>{key}</code> ({count})</dt><dd>{esc(RECOVERY[key])}</dd>"
        )
    parts.append("</dl>")
    parts.append(
        f'<h4 id="{RECORD_ID}-sources">Every affected source</h4><p>Log references point to the numbered full job log. Each row lists claim numbers without a recorded outcome, plus any source-specific grammar-retry lines. Grammar observations are partial evidence and do not identify the numbered claim. A terminal cap is a symptom, not a recovered cause.</p><table><thead><tr><th>Source / binding</th><th>Class, attempts, last error</th><th>Evidence and retry</th></tr></thead><tbody>'
    )
    for r in result["sources"]:
        bindings = (
            "; ".join(
                f"{n['id']} ({n['name_stage']}; score {n['reviewer_score_name']})"
                for n in r["names"]
            )
            or "unproduced"
        )
        refs = (
            ", ".join(map(str, r["refusal_lines"] or r["log_lines"][:6]))
            or "no exact source-path mention"
        )
        parts.append(
            f'<tr data-source="{esc(r["id"])}"><td><code>{esc(r["id"])}</code><br>{esc(bindings)}</td><td>{r["class"]}; {r["attempts"]} claims<br>{esc(r["last_error"] or "(blank)")}</td><td>{esc(r["recovery"])}<br>Log: {esc(refs)}<br>Unexplained claim numbers: {esc(r["unexplained_attempts"])}<br>Grammar-retry lines: {esc(r.get("grammar_lines", []))}</td></tr>'
        )
    parts.append("</tbody></table>")
    parts.append(
        f'<h4 id="{RECORD_ID}-reproducibility">Reproduction and limits</h4><p>Job log: <code>{esc(result["source_log"]["path"])}</code>; SHA-256 <code>{result["source_log"]["sha256"]}</code>. Raw source/context snapshots and complete matched log quotes live beside this run’s manifest. The JSON embeds the independent source and edge-expanded count queries, positive controls, 73 exception references and current writer locations. The runner persists both artifacts before enforcing attempt-history completeness. This cohort must end with EXIT=1: the census and current guard checks do not satisfy the missing historical attribution. A new authorized run could record fresh source/claim outcomes but could not recreate these missing historical events.</p>'
    )
    parts.append("</section>\n")
    content = "\n".join(parts)
    assert len(content.encode()) <= 300000, "Repository HTML exceeds size limit"
    destination.write_text(content)


def verify_report_refusal(
    candidate_revision: str, snapshot_path: Path, context_path: Path, log_path: Path
) -> None:
    """Run a revision's producer in disposable output paths and demand refusal."""
    relative = Path(__file__).resolve().relative_to(TREE)
    source = subprocess.check_output(
        ["git", "-C", str(TREE), "show", f"{candidate_revision}:{relative}"],
    )
    with tempfile.TemporaryDirectory(
        prefix="attempt-history-", dir=os.environ["TMPDIR"]
    ) as temporary:
        scratch = Path(temporary)
        candidate = scratch / relative
        candidate.parent.mkdir(parents=True)
        candidate.write_bytes(source)
        spec = importlib.util.spec_from_file_location("history_candidate", candidate)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.TREE = TREE
        inputs = scratch / "inputs"
        inputs.mkdir()
        for path in (
            snapshot_path,
            context_path,
            snapshot_path.parent / "attribution-receipts.json",
            snapshot_path.parent / "application-log-attribution.json",
        ):
            shutil.copy2(path, inputs / path.name)
        print(f"# candidate_revision={candidate_revision} module={module.__file__}")
        print(f"# measurement_cwd={Path.cwd().resolve()}")
        failure_id = "test_incomplete_attempt_history_blocks_completion"
        try:
            module.report(
                inputs / snapshot_path.name, inputs / context_path.name, log_path
            )
        except RuntimeError as error:
            assert str(error).startswith("Attempt history incomplete:"), str(error)
            print(f"REFUSAL {error}")
        else:
            print(
                f"FAILED {failure_id}: producer returned successfully for missing history"
            )
            raise AssertionError(failure_id)
        result = json.loads(candidate.with_name(f"{RECORD_ID}.json").read_text())
        complete_rows = [
            row
            for row in result["sources"]
            if row["attempts"] > 0 and not row["unexplained_attempts"]
        ]
        assert complete_rows
        module.require_complete_attempt_history({"sources": complete_rows})
        corrupted = {**complete_rows[0], "refusal_attempts": []}
        try:
            module.require_complete_attempt_history({"sources": [corrupted]})
        except RuntimeError as error:
            assert "Attempt history incomplete:" in str(error)
            print(f"CONTROL refused removed numbered outcomes: {error}")
        else:
            raise AssertionError(
                "Completeness trusted a stale unexplained_attempts field"
            )
        print(
            f"PASSED {failure_id}; complete-history positive control={len(complete_rows)}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--run-log", type=Path, required=True)
    parser.add_argument("--supplement", type=Path)
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--gate-log", type=Path)
    parser.add_argument("--verify-report-refusal-at", metavar="REVISION")
    args = parser.parse_args()

    def execute():
        if args.verify_report_refusal_at:
            assert args.supplement
            verify_report_refusal(
                args.verify_report_refusal_at,
                args.snapshot,
                args.supplement,
                args.run_log,
            )
            return
        if args.refresh or not args.supplement:
            capture(args.snapshot, args.run_log)
        if args.supplement and (args.refresh or not args.report):
            supplement(args.snapshot, args.supplement)
        if args.report:
            assert args.supplement
            report(args.snapshot, args.supplement, args.run_log)

    if not args.gate_log:
        execute()
        return
    code = 0
    with (
        args.gate_log.open("w") as gate,
        contextlib.redirect_stdout(gate),
        contextlib.redirect_stderr(gate),
    ):
        print(
            f"# revision={revision(TREE)} tree={TREE} command={shlex.join(sys.argv)}",
            flush=True,
        )
        print(
            f"# cwd={Path.cwd().resolve()} module={GraphClient.__module__}", flush=True
        )
        print(
            f"# runner_sha256={hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}",
            flush=True,
        )
        print(
            "# command: "
            + " ".join(
                f"{key}={shlex.quote(os.environ[key])}"
                for key in (
                    "TMPDIR",
                    "PYTHONDONTWRITEBYTECODE",
                    "PYTHONPATH",
                    "UV_PROJECT_ENVIRONMENT",
                )
                if key in os.environ
            )
            + " uv run --no-sync python "
            + shlex.join(sys.argv),
            flush=True,
        )
        try:
            execute()
        except Exception:
            traceback.print_exc()
            code = 1
        finally:
            print(f"EXIT={code}", flush=True)
    print(f"Gate log: {args.gate_log}; EXIT={code}")
    raise SystemExit(code)


if __name__ == "__main__":
    main()
