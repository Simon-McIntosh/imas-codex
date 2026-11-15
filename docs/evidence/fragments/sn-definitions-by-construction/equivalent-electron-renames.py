"""Render the rename record from the population and per-name progress ledger."""

from __future__ import annotations

import argparse
import html
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from imas_codex.graph.client import GraphClient
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--population", type=Path, default=POPULATION)
    parser.add_argument("--prior-progress", type=Path, default=PRIOR_PROGRESS)
    parser.add_argument("--progress", type=Path, default=PROGRESS)
    parser.add_argument("--json-output", type=Path, default=JSON_OUTPUT)
    parser.add_argument("--html-output", type=Path, default=HTML_OUTPUT)
    parser.add_argument("--sourceless-count", type=int, required=True)
    parser.add_argument("--sourceless-allowance", type=int, default=10)
    parser.add_argument("--spend-ceiling", type=float, default=8.0)
    args = parser.parse_args()

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
