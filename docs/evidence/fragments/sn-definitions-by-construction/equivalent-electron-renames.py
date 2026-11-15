"""Render the rename record from the population and per-name progress ledger."""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from imas_codex.graph.client import GraphClient
from imas_codex.standard_names.grammar_adapter import parse_canonical_name
from imas_codex.standard_names.source_reads import (
    count_live_names_without_source,
    read_source_paths_by_name,
)

HERE = Path(__file__).resolve().parent
POPULATION = HERE / "live-rewrite-bulk.json"
PRIOR_RECORD = HERE / "held-names-refine-pass.json"
PRIOR_PROGRESS = (
    Path.home() / ".cache/imas-codex-live-rewrite/held-names-progress.jsonl"
)
PROGRESS = (
    Path.home() / ".cache/imas-codex-live-rewrite/equivalent-electron-progress.jsonl"
)
JSON_OUTPUT = HERE / "equivalent-electron-renames.json"
HTML_OUTPUT = HERE / "equivalent-electron-renames.html"
SPEND_CEILING = 8.0
PROBE_AVERAGE_CEILING = 0.20
DERIVED_RECEIPT = Path(
    "/home/ITER/mcintos/.config/reckon/crew/runs/"
    "r-20261115T212654636291-sdbc-equivalent-electron-renames/"
    "derived-children-1318712.jsonl"
)

READ_COST = """
MATCH (run:SNRun)
WHERE run.invocation_scope CONTAINS $scope_id
OPTIONAL MATCH (cost:LLMCost)-[:FOR_RUN]->(run)
WHERE $name IN coalesce(cost.standard_name_ids, [])
RETURN run.id AS run_id,
       run.cost_is_exact AS cost_is_exact,
       count(cost) AS calls,
       coalesce(sum(cost.llm_cost), 0.0) AS spend_usd
"""

READ_IDENTITIES = """
UNWIND $ids AS wanted
MATCH (sn:StandardName {id: wanted})
RETURN sn.id AS id, sn.description AS description, sn.unit AS unit,
       sn.name_stage AS name_stage, sn.docs_stage AS docs_stage,
       sn.reviewer_score_name AS score_name,
       sn.reviewer_score_docs AS score_docs
"""

READ_BATCH_IDENTITIES = """
UNWIND $ids AS wanted
OPTIONAL MATCH (sn:StandardName {id: wanted})
RETURN wanted AS requested_id, sn.id AS id, sn.description AS description,
       sn.unit AS unit, sn.status AS status, sn.name_stage AS name_stage,
       sn.docs_stage AS docs_stage, sn.edit_status AS edit_status,
       sn.reviewer_score_name AS score_name,
       sn.reviewer_score_docs AS score_docs, sn.run_id AS run_id
"""

READ_DD_DOCUMENTATION = """
UNWIND $ids AS wanted
MATCH (node:IMASNode {id: wanted})
RETURN node.id AS id, node.documentation AS documentation
"""

EQUIVALENT_BASIS = re.compile(r"equivalent[- ]+electrons?", re.IGNORECASE)


def read_json_lines(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    for number, line in enumerate(path.read_text().splitlines(), start=1):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{number}: invalid JSON: {error}") from error
    return rows


def identity(name: str, record: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": name,
        "description": record.get("description"),
        "unit": record.get("unit"),
        "review_scores": {
            "name": record.get("score_name"),
            "docs": record.get("score_docs"),
        },
        "source_bindings": sorted(record.get("source_bindings", [])),
    }


def cost_receipt(
    graph: GraphClient, scope_id: str | None, live_name: str
) -> dict[str, Any]:
    if not scope_id:
        return {"run_id": None, "calls": 0, "spend_usd": 0.0}
    rows = graph.query(READ_COST, scope_id=scope_id, name=live_name) or []
    if len(rows) != 1:
        raise ValueError(
            f"{scope_id}: expected one scoped SNRun cost receipt, found {len(rows)}"
        )
    row = rows[0]
    if row.get("cost_is_exact") is False:
        raise ValueError(f"{scope_id}: SNRun marks its LLMCost total as inexact")
    return {
        "run_id": row["run_id"],
        "calls": int(row["calls"]),
        "spend_usd": round(float(row["spend_usd"] or 0.0), 6),
    }


def read_cost_receipt(scope_id: str | None, live_name: str) -> dict[str, Any]:
    with GraphClient() as graph:
        return cost_receipt(graph, scope_id, live_name)


def progress_by_name(
    events: list[dict[str, Any]], expected: set[str]
) -> dict[str, dict[str, dict[str, Any]]]:
    grouped: dict[str, dict[str, dict[str, Any]]] = {}
    for event in events:
        name = event.get("name")
        if name not in expected:
            continue
        grouped.setdefault(name, {})[event["event"]] = event
    missing = sorted(
        name
        for name in expected
        if name not in grouped or "settled" not in grouped[name]
    )
    if missing:
        raise ValueError("unsettled names in progress ledger: " + ", ".join(missing))
    return grouped


def prior_renames(
    population: list[dict[str, Any]], progress: list[dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    names = {row["name"] for row in population}
    settled = {
        row["name"]: row
        for row in progress
        if row.get("event") == "settled"
        and row.get("name") in names
        and row.get("rename_to")
    }
    return settled


def make_record(
    population_row: dict[str, Any],
    settled: dict[str, Any],
    *,
    graph: GraphClient,
    prior: bool,
) -> dict[str, Any]:
    old_id = population_row["name"]
    before = settled.get("before") or {}
    after = settled.get("after") or before
    outcome = settled.get("outcome", "left")
    target = settled.get("rename_to")
    live_name = after.get("name") if target else old_id
    live_name = live_name or target or old_id

    old = identity(old_id, before)
    after_identity = identity(live_name, after)
    renamed = bool(target and live_name != old_id)
    reason = settled.get("reason") or settled.get("left_reason")
    if not renamed and not reason:
        reason = settled.get("distinction") or (
            "Already renamed through the signed path before this pass."
            if prior
            else "The signed path settled the candidate without changing its identity."
        )

    spend = cost_receipt(graph, settled.get("run_id"), live_name)
    return {
        "old_identity": old,
        "new_identity": after_identity if renamed else None,
        "live_identity": {"id": live_name},
        "left_reason": None if renamed else reason,
        "outcome": "renamed" if renamed else outcome,
        "basis": population_row.get("basis", ""),
        "after": {
            "description": after.get("description"),
            "unit": after.get("unit"),
            "score_name": after.get("score_name"),
            "score_docs": after.get("score_docs"),
            "name_stage": after.get("name_stage"),
            "docs_stage": after.get("docs_stage"),
            "source_bindings": sorted(after.get("source_bindings", [])),
        },
        "edit_scope_id": settled.get("run_id"),
        "cost": spend,
    }


def render_html(document: dict[str, Any]) -> str:
    summary = document["summary"]
    rows = []
    for record in document["names"]:
        old = record["old_identity"]
        new = record["new_identity"]
        after = record["after"]
        scores_before = old["review_scores"]
        if new:
            destination = f"<code>{html.escape(new['id'])}</code>"
        else:
            destination = html.escape(record["left_reason"] or "left unchanged")
        sources_before = ", ".join(old["source_bindings"]) or "none"
        sources_after = ", ".join(after["source_bindings"]) or "none"
        score_text = (
            f"{scores_before['name']} / {scores_before['docs']} → "
            f"{after['score_name']} / {after['score_docs']}"
        )
        rows.append(
            "<tr>"
            f"<td><code>{html.escape(old['id'])}</code></td>"
            f"<td>{destination}</td>"
            f"<td>{html.escape(record['outcome'])}</td>"
            f"<td>{html.escape(score_text)}</td>"
            f"<td>{html.escape(sources_before)} → {html.escape(sources_after)}</td>"
            f"<td>{record['cost']['spend_usd']:.6f}</td>"
            "</tr>"
        )
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Equivalent-electron name updates</title>
</head>
<body>
<article>
<h2>Equivalent-electron name updates</h2>
<p>{summary["renamed"]} of {summary["population"]} names now state the
equivalent-electron basis in their identity; {summary["left"]} remain with a
recorded reason. Recorded spend is USD {summary["spend_usd"]:.6f} against the
USD {document["spend_ceiling_usd"]:.2f} ceiling. The live sourceless count is
{document["sourceless"]["count"]}.</p>
<table>
<caption>Each original identity, its settled identity or reason left, review
scores before and after, source bindings before and after, and unique-run spend</caption>
<thead><tr><th>Old identity</th><th>New identity or reason left</th>
<th>Outcome</th><th>Name / docs review scores before → after</th>
<th>Source bindings before → after</th><th>Spend (USD)</th></tr></thead>
<tbody>{"".join(rows)}</tbody>
</table>
<p>Spend is summed from LLMCost rows connected through each scoped SNRun once.
The signed edit scope is recorded per row in the JSON receipt. Generated at
{html.escape(document["generated_at_utc"])}.</p>
</article>
</body>
</html>
"""


def candidate_name(name: str) -> str:
    if name == "electron_particle_flux_at_wall_due_to_pumping":
        return "particle_equivalent_electron_flux_at_wall_due_to_pumping"
    if name.startswith("source_rate_"):
        return "equivalent_electron_" + name
    if "_source_rate" in name:
        return name.replace("_source_rate", "_equivalent_electron_source_rate", 1)
    if "_count" in name:
        return name.replace("_count", "_equivalent_electron_count", 1)
    raise ValueError(f"no qualifier position for {name}")


def append_progress(path: Path, event: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def read_embedded_json(path: Path) -> dict[str, Any]:
    for line in path.read_text().splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError(f"no JSON object found in {path}")


def dictionary_context(
    graph: GraphClient, sources: list[str]
) -> tuple[bool, list[dict[str, Any]]]:
    prefixes: set[str] = set()
    source_paths: dict[str, list[str]] = {}
    for source in sources:
        if source.startswith("dd:"):
            parts = source[3:].split("/")
            ancestors = ["/".join(parts[:index]) for index in range(1, len(parts) + 1)]
            source_paths[source] = ancestors
            prefixes.update(ancestors)

    documents = {
        row["id"]: row.get("documentation")
        for row in (graph.query(READ_DD_DOCUMENTATION, ids=sorted(prefixes)) or [])
    }
    evidence: list[dict[str, Any]] = []
    supported = bool(sources)
    for source in sources:
        if source.startswith("dd:"):
            rows = [
                {"id": path, "documentation": documents.get(path)}
                for path in source_paths[source]
                if path in documents
            ]
            matching = [
                row
                for row in rows
                if EQUIVALENT_BASIS.search(row.get("documentation") or "")
            ]
            evidence.append(
                {
                    "source": source,
                    "ancestors": rows,
                    "equivalent_electron_basis": bool(matching),
                }
            )
            supported &= bool(matching)
            continue

        if source == "derived:source_rate_due_to_gas_injection":
            receipt = read_embedded_json(DERIVED_RECEIPT)
            children = [row["id"] for row in receipt.get("children", [])]
            child_rows = graph.query(READ_BATCH_IDENTITIES, ids=children) or []
            child_by_id = {row["requested_id"]: row for row in child_rows}
            child_sources = read_source_paths_by_name(graph, children)
            child_all_sources = sorted(
                {
                    source_id
                    for bindings in child_sources.values()
                    for source_id in bindings
                }
            )
            child_supported, child_context = dictionary_context(
                graph, child_all_sources
            )
            children_supported = len(children) == 11 and all(
                child_by_id.get(child, {}).get("id")
                and EQUIVALENT_BASIS.search(child_by_id[child].get("description") or "")
                for child in children
            )
            supported &= children_supported and child_supported
            evidence.append(
                {
                    "source": source,
                    "children": [
                        {
                            "id": child,
                            "description": child_by_id.get(child, {}).get(
                                "description"
                            ),
                            "source_bindings": child_sources.get(child, []),
                        }
                        for child in children
                    ],
                    "data_dictionary": child_context,
                    "equivalent_electron_basis": children_supported and child_supported,
                }
            )
            continue

        evidence.append(
            {
                "source": source,
                "equivalent_electron_basis": False,
                "reason": "source is not a Data Dictionary path with a verified basis",
            }
        )
        supported = False
    return bool(supported), evidence


def is_live(row: dict[str, Any] | None) -> bool:
    return bool(
        row
        and row.get("id")
        and row.get("status") != "superseded"
        and row.get("name_stage") != "superseded"
    )


def live_rows(graph: GraphClient, names: list[str]) -> dict[str, dict[str, Any]]:
    rows = graph.query(READ_BATCH_IDENTITIES, ids=names) or []
    return {row["requested_id"]: row for row in rows}


def scheduler_snapshot() -> dict[str, tuple[str, str]]:
    user = os.environ.get("USER")
    if not user:
        raise RuntimeError("USER is unavailable for the single-writer check")
    result = subprocess.run(
        ["squeue", "-u", user, "-h", "-o", "%i|%j|%T"],
        check=True,
        capture_output=True,
        text=True,
    )
    jobs: dict[str, tuple[str, str]] = {}
    for line in result.stdout.splitlines():
        parts = line.split("|", 2)
        if len(parts) == 3:
            jobs[parts[0]] = (parts[1], parts[2])
    return jobs


def assert_writer_free(
    baseline_jobs: dict[str, tuple[str, str]],
    current_job_id: str | None,
    allowed_wrap_jobs: set[str],
) -> None:
    current_jobs = scheduler_snapshot()
    for job_id, (name, state) in current_jobs.items():
        lowered = name.casefold()
        if re.search(r"\bsn[-_ ]?(run|edit)\b|\bsigned[-_ ]apply\b", lowered):
            raise RuntimeError(f"graph writer job is active or queued: {job_id} {name}")
        if name == "wrap" and state == "RUNNING":
            baseline = baseline_jobs.get(job_id)
            if job_id != current_job_id and (
                job_id not in allowed_wrap_jobs
                or baseline is None
                or baseline[1] != "RUNNING"
            ):
                raise RuntimeError(
                    f"new or newly running wrapper job needs inspection: {job_id}"
                )
        if "importer" in lowered and state == "RUNNING":
            baseline = baseline_jobs.get(job_id)
            if baseline is None or baseline[1] != "RUNNING":
                raise RuntimeError(f"new importer job needs inspection: {job_id}")

    processes = subprocess.run(
        ["ps", "-eo", "pid=,comm=,args="],
        check=True,
        capture_output=True,
        text=True,
    )
    for line in processes.stdout.splitlines():
        if re.search(r"imas-codex\s+sn\s+(run|edit)\b", line):
            raise RuntimeError(f"graph writer process is active: {line.strip()}")


def event_snapshot(
    name: str,
    row: dict[str, Any],
    sources: list[str],
    source_evidence: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "name": name,
        "description": row.get("description"),
        "unit": row.get("unit"),
        "score_name": row.get("score_name"),
        "score_docs": row.get("score_docs"),
        "name_stage": row.get("name_stage"),
        "docs_stage": row.get("docs_stage"),
        "status": row.get("status"),
        "run_id": row.get("run_id"),
        "source_bindings": sorted(sources),
        "source_evidence": source_evidence,
    }


def run_batch(args: argparse.Namespace) -> int:
    population = json.loads(args.population.read_text())["waiting_equivalent_electron"]
    prior_events = read_json_lines(args.prior_progress)
    prior = prior_renames(population, prior_events)
    allowed_wrap_jobs = set(args.allow_wrap_job)
    baseline_jobs = scheduler_snapshot()
    current_job_id = os.environ.get("SLURM_JOB_ID")
    unexpected_wrappers = sorted(
        job_id
        for job_id, (name, state) in baseline_jobs.items()
        if name == "wrap" and state == "RUNNING" and job_id not in allowed_wrap_jobs
    )
    if unexpected_wrappers:
        raise RuntimeError(
            "inspect active wrapper jobs before graph writes: "
            + ", ".join(unexpected_wrappers)
        )
    candidates = [row for row in population if row["name"] not in prior]
    probe_names = [row["name"] for row in candidates[:5]]
    events = read_json_lines(args.progress)
    grouped: dict[str, dict[str, dict[str, Any]]] = {}
    for event in events:
        if event.get("name") in {row["name"] for row in candidates}:
            grouped.setdefault(event["name"], {})[event["event"]] = event

    if args.probe:
        if args.run_limit != 5:
            raise ValueError("the five-name probe requires --run-limit 5")
        selected_rows = [
            row
            for row in candidates
            if row["name"] in probe_names
            and "settled" not in grouped.get(row["name"], {})
        ]
    else:
        pending = [
            row
            for row in candidates
            if "submitted" in grouped.get(row["name"], {})
            and "settled" not in grouped.get(row["name"], {})
        ]
        remaining = [row for row in candidates if row["name"] not in grouped]
        selected_rows = (pending + remaining)[: args.run_limit]

    with GraphClient() as graph:
        historical_spend = sum(
            cost_receipt(graph, row.get("run_id"), row["after"]["name"])["spend_usd"]
            for row in prior.values()
        )
        current_ids = [
            identifier
            for row in selected_rows
            for identifier in (row["name"], candidate_name(row["name"]))
        ]
        states = live_rows(graph, current_ids)
        source_map = read_source_paths_by_name(graph, current_ids)
        source_support = {
            row["name"]: dictionary_context(graph, source_map.get(row["name"], []))
            for row in selected_rows
        }

    assert_writer_free(baseline_jobs, current_job_id, allowed_wrap_jobs)
    completed: dict[str, dict[str, Any]] = {}
    for population_row in selected_rows:
        old_id = population_row["name"]
        target = candidate_name(old_id)
        before_row = states.get(old_id, {})
        before_sources = source_map.get(old_id, [])
        existing = grouped.get(old_id, {})
        submitted = existing.get("submitted")
        basis_supported, source_evidence = source_support[old_id]
        before = event_snapshot(old_id, before_row, before_sources, source_evidence)
        if submitted and submitted.get("before"):
            before = submitted["before"]
            before_sources = before.get("source_bindings", [])
            source_evidence = before.get("source_evidence", source_evidence)

        if not submitted:
            submitted = {
                "event": "submitted",
                "at": datetime.now(UTC).isoformat(),
                "name": old_id,
                "proposed_target": target,
                "basis": population_row.get("basis", ""),
                "before": before,
            }
            append_progress(args.progress, submitted)

        if not is_live(before_row):
            target_row = states.get(target, {})
            if is_live(target_row):
                live_id = target
                outcome = "already_renamed"
                reason = "The live graph already carries the proposed identity."
            else:
                live_id = old_id
                outcome = "no_longer_live"
                reason = "The old identity is no longer live and its target is absent."
            after_row = target_row if live_id == target else before_row
            after_sources = source_map.get(live_id, [])
            scope_id = (after_row or {}).get("run_id")
            spend = (
                read_cost_receipt(scope_id, live_id)
                if outcome == "already_renamed" and scope_id
                else {"run_id": None, "calls": 0, "spend_usd": 0.0}
            )
        elif is_live(states.get(target)):
            live_id = old_id
            outcome = "collision"
            reason = f"The proposed live identity {target} already exists."
            after_row = before_row
            after_sources = before_sources
            scope_id = None
            spend = {"run_id": None, "calls": 0, "spend_usd": 0.0}
        elif not basis_supported:
            live_id = old_id
            outcome = "declined_source"
            evidence_text = (
                ", ".join(item["source"] for item in source_evidence)
                or "no producing source"
            )
            reason = (
                "The Data Dictionary source does not state an equivalent-electron "
                f"basis for {evidence_text}."
            )
            after_row = before_row
            after_sources = before_sources
            scope_id = None
            spend = {"run_id": None, "calls": 0, "spend_usd": 0.0}
        else:
            assert_writer_free(baseline_jobs, current_job_id, allowed_wrap_jobs)
            target = candidate_name(old_id)
            parse_canonical_name(target)
            reason = (
                "The Data Dictionary binding and definition state this value in "
                "equivalent electrons; the qualifier names that counting basis "
                "while preserving the supported species and delivery scope."
            )
            total_spend = historical_spend + sum(
                float(row.get("spend_usd", 0.0))
                for row in read_json_lines(args.progress)
                if row.get("event") == "settled"
            )
            remaining = SPEND_CEILING - total_spend
            if remaining < 0.01:
                raise RuntimeError(f"spend ceiling reached: USD {total_spend:.6f}")
            cost_limit = min(1.0, remaining)
            command = [
                "imas-codex",
                "sn",
                "edit",
                old_id,
                "--rename",
                target,
                "--reason",
                reason,
                "--scope",
                "self",
                "--cost-limit",
                f"{cost_limit:.6f}",
            ]
            result = subprocess.run(command, check=False)

            with GraphClient() as graph:
                states_after = live_rows(graph, [old_id, target])
                after_sources_by_name = read_source_paths_by_name(
                    graph,
                    [
                        name
                        for name in (old_id, target)
                        if states_after.get(name, {}).get("id")
                    ],
                )
                sourceless = count_live_names_without_source(graph)["count"]

            old_after = states_after.get(old_id, {})
            target_after = states_after.get(target, {})
            if is_live(target_after) and not is_live(old_after):
                live_id = target
                after_row = target_after
                after_sources = after_sources_by_name.get(target, [])
                outcome = (
                    "accepted"
                    if target_after.get("name_stage") == "accepted"
                    else "review_not_accepted"
                )
                scope_id = target_after.get("run_id")
            else:
                live_id = old_id
                after_row = old_after
                after_sources = after_sources_by_name.get(old_id, [])
                outcome = "unchanged"
                updated_scope = old_after.get("run_id")
                scope_id = (
                    updated_scope
                    if updated_scope and updated_scope != before_row.get("run_id")
                    else None
                )

            if result.returncode == 2:
                outcome = "refused_drift"
            elif result.returncode not in (0, 3):
                outcome = "review_error"
            elif result.returncode == 3:
                outcome = "review_not_accepted"

            cost = (
                read_cost_receipt(scope_id, live_id)
                if scope_id
                else {"run_id": None, "calls": 0, "spend_usd": 0.0}
            )
            if live_id == target and not scope_id:
                raise ValueError(
                    f"{old_id}: renamed identity has no edit run scope for cost lookup"
                )
            scope_id = scope_id or cost.get("run_id")
            if outcome in {"accepted", "review_not_accepted"} and (
                sorted(before_sources) != sorted(after_sources)
            ):
                outcome = "source_binding_mismatch"
                reason = "The signed rename did not carry the original producing source bindings."
            if after_sources:
                with GraphClient() as graph:
                    after_source_evidence = dictionary_context(graph, after_sources)[1]
            else:
                after_source_evidence = []
            after = event_snapshot(
                live_id, after_row, after_sources, after_source_evidence
            )
            settled = {
                "event": "settled",
                "at": datetime.now(UTC).isoformat(),
                "name": old_id,
                "proposed_target": target,
                "rename_to": target if live_id == target else None,
                "run_id": scope_id,
                "sn_run_id": cost.get("run_id"),
                "outcome": outcome,
                "reason": reason if outcome != "accepted" else None,
                "before": before,
                "after": after,
                "calls": cost["calls"],
                "spend_usd": cost["spend_usd"],
                "cost_is_exact": True,
            }
            append_progress(args.progress, settled)
            completed[old_id] = settled
            total_spend += cost["spend_usd"]
            print(
                f"settled {old_id} -> {live_id}: {outcome}, "
                f"USD {cost['spend_usd']:.6f}; total USD {total_spend:.6f}",
                flush=True,
            )
            if result.returncode not in (0, 3):
                return 2
            if outcome == "source_binding_mismatch":
                return 2
            if sourceless > 10:
                print(f"stop: live names without a source rose to {sourceless}")
                return 2
            if total_spend > SPEND_CEILING:
                print(f"stop: spend reached USD {total_spend:.6f}")
                return 2
            continue

        if after_sources:
            with GraphClient() as graph:
                after_source_evidence = dictionary_context(graph, after_sources)[1]
        else:
            after_source_evidence = []
        after = event_snapshot(
            live_id, after_row or {}, after_sources, after_source_evidence
        )
        settled = {
            "event": "settled",
            "at": datetime.now(UTC).isoformat(),
            "name": old_id,
            "proposed_target": target,
            "rename_to": target if live_id == target else None,
            "run_id": scope_id,
            "outcome": outcome,
            "reason": reason,
            "before": before,
            "after": after,
            "calls": spend["calls"],
            "spend_usd": spend["spend_usd"],
            "cost_is_exact": True,
        }
        append_progress(args.progress, settled)
        completed[old_id] = settled
        print(
            f"settled {old_id}: {outcome}; reason: {reason}; "
            f"USD {spend['spend_usd']:.6f}",
            flush=True,
        )

    total_spend = historical_spend + sum(
        float(row.get("spend_usd", 0.0))
        for row in read_json_lines(args.progress)
        if row.get("event") == "settled"
    )
    if args.probe:
        probe_rows = {
            event["name"]: event
            for event in read_json_lines(args.progress)
            if event.get("event") == "settled" and event.get("name") in probe_names
        }
        probe_spend = sum(
            float(row.get("spend_usd", 0.0)) for row in probe_rows.values()
        )
        landed = sum(
            1 for row in probe_rows.values() if row.get("outcome") == "accepted"
        )
        average = probe_spend / 5
        print(
            f"probe: {landed}/5 accepted, USD {probe_spend:.6f} total, "
            f"USD {average:.6f}/name; cumulative USD {total_spend:.6f}",
            flush=True,
        )
        if average > PROBE_AVERAGE_CEILING or landed < 3:
            print("stop: the five-name probe did not meet its cost and landing floor")
            return 3
    if total_spend > SPEND_CEILING:
        return 2
    print(f"batch complete: {len(completed)} names; cumulative USD {total_spend:.6f}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--population", type=Path, default=POPULATION)
    parser.add_argument("--prior-progress", type=Path, default=PRIOR_PROGRESS)
    parser.add_argument("--progress", type=Path, default=PROGRESS)
    parser.add_argument("--json-output", type=Path, default=JSON_OUTPUT)
    parser.add_argument("--html-output", type=Path, default=HTML_OUTPUT)
    parser.add_argument("--sourceless-count", type=int)
    parser.add_argument("--sourceless-allowance", type=int, default=10)
    parser.add_argument("--spend-ceiling", type=float, default=8.0)
    parser.add_argument("--run-limit", type=int)
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--allow-wrap-job", action="append", default=[])
    args = parser.parse_args()

    if args.run_limit is not None:
        if args.run_limit < 1:
            parser.error("--run-limit must be positive")
        return run_batch(args)
    if args.probe or args.allow_wrap_job:
        parser.error("--probe and --allow-wrap-job require --run-limit")
    if args.sourceless_count is None:
        parser.error("--sourceless-count is required when rendering the report")

    population_document = json.loads(args.population.read_text())
    population = population_document["waiting_equivalent_electron"]
    expected = {row["name"] for row in population}
    if len(population) != 56 or len(expected) != 56:
        raise ValueError(f"expected 56 distinct population names, got {len(expected)}")

    previous = prior_renames(population, read_json_lines(args.prior_progress))
    current_names = expected - previous.keys()
    current = progress_by_name(read_json_lines(args.progress), current_names)

    with GraphClient() as graph:
        records = []
        for population_row in population:
            old_id = population_row["name"]
            if old_id in previous:
                settled = previous[old_id]
                is_prior = True
            else:
                settled = current[old_id]["settled"]
                is_prior = False
            records.append(
                make_record(population_row, settled, graph=graph, prior=is_prior)
            )
        live_ids = [row["live_identity"]["id"] for row in records]
        live_rows = graph.query(READ_IDENTITIES, ids=live_ids) or []
        live = {row["id"]: row for row in live_rows}
        live_sources = read_source_paths_by_name(graph, live_ids)
        sourceless_now = count_live_names_without_source(graph)["count"]

    for row in records:
        live_id = row["live_identity"]["id"]
        snapshot = live.get(live_id)
        if not snapshot:
            raise ValueError(f"live identity disappeared while rendering: {live_id}")
        row["settled_after"] = row["after"]
        row["after"] = {
            "description": snapshot.get("description"),
            "unit": snapshot.get("unit"),
            "score_name": snapshot.get("score_name"),
            "score_docs": snapshot.get("score_docs"),
            "name_stage": snapshot.get("name_stage"),
            "docs_stage": snapshot.get("docs_stage"),
            "source_bindings": sorted(live_sources.get(live_id, [])),
        }
        if row["new_identity"] is not None:
            row["new_identity"] = {
                **row["new_identity"],
                "description": snapshot.get("description"),
                "unit": snapshot.get("unit"),
                "review_scores": {
                    "name": snapshot.get("score_name"),
                    "docs": snapshot.get("score_docs"),
                },
                "source_bindings": sorted(live_sources.get(live_id, [])),
            }

    if sourceless_now != args.sourceless_count:
        raise ValueError(
            f"sourceless count changed: graph says {sourceless_now}, "
            f"input says {args.sourceless_count}"
        )
    spend_total = round(sum(row["cost"]["spend_usd"] for row in records), 6)
    renamed = sum(1 for row in records if row["new_identity"] is not None)
    left = len(records) - renamed
    document = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "population": {
            "source": args.population.name,
            "count": len(records),
        },
        "spend_ceiling_usd": args.spend_ceiling,
        "spend_source": "LLMCost joined once through each scoped SNRun",
        "spend_total_usd": spend_total,
        "sourceless": {
            "count": sourceless_now,
            "allowance": args.sourceless_allowance,
        },
        "summary": {
            "population": len(records),
            "renamed": renamed,
            "left": left,
            "spend_usd": spend_total,
        },
        "names": records,
    }
    args.json_output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    args.html_output.write_text(render_html(document))
    print(
        f"wrote {len(records)} rows, {renamed} renamed, {left} left; "
        f"USD {spend_total:.6f}; sourceless {sourceless_now}"
    )
    return int(
        spend_total > args.spend_ceiling or sourceless_now > args.sourceless_allowance
    )


if __name__ == "__main__":
    raise SystemExit(main())
