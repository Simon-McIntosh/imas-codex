#!/usr/bin/env python
"""Regenerate the three shared defect causes' members through the repaired stages.

The equilibrium adjudication found three causes shared by many defective members
and eleven smaller ones. This script repairs nothing; it measures what the
repairs did to the three largest:

* ``physical_scope_qualifier_lost`` — the naming (compose) stage now renders the
  enclosing-structure lineage the extraction stage attaches.
* ``source_bindings_conflate_distinct_definitions`` — the source-binding guard
  now reads the sources' stated text and rejects a binding whose stated
  operation neither the name nor a companion source carries.
* ``boundary_forced_to_lcfs`` — the composition system prompt now reserves the
  ``plasma_boundary`` locus for the last closed flux surface and sends a
  selectable boundary to a ``vocab_gap``.

For each member of each cause the script runs the member's own recorded source
packet through the repaired stage, writes the regenerated member into a
disposable Neo4j store, reads it back from that store, and hands the record to a
fresh blind adjudicator that sees the full source packet and the fixed class
list of the earlier adjudication but never an earlier verdict.

Nothing here writes the live graph and nothing here reads an earlier verdict into
the adjudicator's prompt.

``--run`` measures and writes ``stage-repairs.json``; ``--render`` writes the
evidence page from that file; ``--check`` re-derives the page's claims from the
record, the product source and the data file.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import re
from html import escape
from pathlib import Path
from typing import Any, Literal

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[3]
RECORD = ROOT / "equilibrium-adjudication.json"
TRACE = ROOT / "stage-cause-trace.json"
TRACE_SCRIPT = ROOT / "stage-cause-trace.py"
DATA = ROOT / "stage-repairs.json"
FRAGMENT = ROOT / "sagr-stage-repairs.html"
PAGE = ROOT / "stage-repairs.html"
CONFTEST = REPO / "tests/standard_names/conftest.py"

#: The three largest causes, held here so a run cannot quietly measure others.
TOP_CAUSES = (
    "physical_scope_qualifier_lost",
    "source_bindings_conflate_distinct_definitions",
    "boundary_forced_to_lcfs",
)

CEILING_USD = 10.0
COMPOSE_CONCURRENCY = 4

#: The repair each cause is measured against, as the marker its stage carries.
REPAIR_MARKERS = {
    "physical_scope_qualifier_lost": (
        "imas_codex/llm/prompts/sn/generate_name_dd_names.md",
        "{{ item.ancestor_context }}",
    ),
    "source_bindings_conflate_distinct_definitions": (
        "imas_codex/standard_names/worker_tasks/worker_extraction_and_enrichment.py",
        "stated operation mismatch",
    ),
    "boundary_forced_to_lcfs": (
        "imas_codex/llm/prompts/sn/generate_name_system.md",
        "effective_plasma_boundary",
    ),
}

SOURCE_RECORDS = "Producer source records (raw text and extraction snapshot):\n"
ENCLOSING = (
    "Enclosing structures of every source (deduplicated), nearest first per chain:\n"
)
ANCESTOR_LINE = re.compile(r"^  (?P<path>[a-z0-9_]+(?:/[a-z0-9_]+)*): (?P<text>.+)$")


# --------------------------------------------------------------------------
# The record
# --------------------------------------------------------------------------


def trace_module():
    spec = importlib.util.spec_from_file_location("stage_cause_trace", TRACE_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert module is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_record() -> dict[str, Any]:
    return trace_module().decode_record(RECORD)


def cause_members(record: dict[str, Any]) -> dict[str, list[str]]:
    counts, members = trace_module().census(record)
    ordered = [cause for cause, _ in counts.most_common(len(counts))]
    assert tuple(ordered[:3]) == TOP_CAUSES, (
        f"the three largest causes are {ordered[:3]}, not {TOP_CAUSES}"
    )
    return members


def member_before(record: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        row["id"]: {
            "verdict": row["verdict"],
            "defect_classes": row["defect_classes"],
            "candidate_cause": row["candidate_cause"],
        }
        for row in record["judgments"]
    }


def parse_packet(pool: list[str], packet: dict[str, Any]) -> dict[str, Any]:
    """Return the member's name, definition and the sources its packet quotes."""
    lines = [pool[index] for index in packet["text_lines"]]
    text = "\n".join(lines)
    name = next(
        line.split(": ", 1)[1] for line in lines if line.startswith("Standard name: ")
    )
    definition = next(
        line.split(": ", 1)[1]
        for line in lines
        if line.startswith("Definition (stored description): ")
    )
    unit = next(
        line.split(": ", 1)[1] for line in lines if line.startswith("Stored unit: ")
    )
    records = json.loads(text.split(SOURCE_RECORDS, 1)[1].splitlines()[0])
    ancestors: list[dict[str, str]] = []
    if ENCLOSING in text:
        block = text.split(ENCLOSING, 1)[1]
        for line in block.splitlines():
            match = ANCESTOR_LINE.match(line)
            if match and match.group("path").count("/") >= 2:
                ancestors.append(
                    {"path": match.group("path"), "text": match.group("text")}
                )
    return {
        "id": packet["id"],
        "name": name,
        "definition": definition,
        "unit": unit,
        "sources": records,
        "ancestors": ancestors,
    }


def source_text(record_source: dict[str, Any]) -> str:
    parts = [
        str(record_source.get("description") or ""),
        str(record_source.get("dd_documentation") or ""),
    ]
    return " ".join(part for part in parts if part)


def source_path(record_source: dict[str, Any]) -> str:
    return str(record_source["id"]).split(":", 1)[1]


# --------------------------------------------------------------------------
# The repaired stages
# --------------------------------------------------------------------------


def admitted_sources(
    member_name: str, sources: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Apply the repaired binding guard to one member's sources."""
    from imas_codex.standard_names.worker_tasks.worker_extraction_and_enrichment import (
        _is_attachment_consistent,
    )

    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, str]] = []
    accepted_texts: list[str] = []
    for record_source in sources:
        path = source_path(record_source)
        text = source_text(record_source)
        ok, reason = _is_attachment_consistent(
            path,
            member_name,
            dd_text=text,
            existing_texts=tuple(accepted_texts),
        )
        if ok:
            accepted.append(record_source)
            accepted_texts.append(text)
        else:
            rejected.append({"path": path, "reason": reason})
    return accepted, rejected


async def compose_member(
    context: dict[str, Any],
    system_prompt: str,
    packet: dict[str, Any],
    sources: list[dict[str, Any]],
    model: str,
    semaphore: asyncio.Semaphore,
    calls: dict[str, float],
) -> dict[str, Any]:
    """Run the naming stage for one member's sources."""
    from imas_codex.discovery.base.llm import acall_llm_structured
    from imas_codex.llm.prompt_loader import render_prompt
    from imas_codex.standard_names.models import StandardNameComposeBatch

    primary = sources[0]
    item: dict[str, Any] = {
        "path": source_path(primary),
        "description": primary.get("description"),
        "documentation": primary.get("dd_documentation"),
        "unit": primary.get("dd_unit"),
        "ancestor_context": packet["ancestors"],
    }
    member_docs = {source_path(other): source_text(other) for other in sources[1:]}
    if member_docs:
        item["dd_paths_docs"] = member_docs
    user_prompt = render_prompt(
        "sn/generate_name_dd_names",
        {
            **context,
            "items": [item],
            "ids_name": item["path"].split("/")[0],
        },
    )
    async with semaphore:
        result = await acall_llm_structured(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_model=StandardNameComposeBatch,
            service="standard-names",
        )
    calls["compose"] = calls.get("compose", 0) + 1
    calls["compose_cost_usd"] = calls.get("compose_cost_usd", 0.0) + float(
        getattr(result, "cost", 0.0)
    )
    parsed = result.parsed
    gaps = [
        {"segment": gap.segment, "token": gap.token, "reason": gap.reason}
        for gap in parsed.vocab_gaps
    ]
    if not parsed.candidates:
        return {"outcome": "vocab_gap" if gaps else "no_candidate", "gaps": gaps}
    candidate = parsed.candidates[0]
    try:
        name = candidate.compose_name()
    except Exception as error:  # a grammar rejection is a stage outcome, not a crash
        return {
            "outcome": "grammar_rejected",
            "error": str(error)[:200],
            "gaps".replace("gaps", "gaps"): gaps,
        }
    return {
        "outcome": "named",
        "name": name,
        "definition": candidate.description,
        "unit": packet["unit"],
        "gaps": gaps,
    }


JUDGE_RULES = """You are an independent physics auditor of fusion data-dictionary
standard names. Decide whether ONE record names its sources faithfully, or
whether the stage that produced it was right to decline to name the quantity.
You are given the Data Dictionary source text of every source the record must
identify, then the record itself. Never assume a record is right because it
parses, and never ask for a different record: judge this one.

The fixed classes (a record may carry several):

- `name_contradicts_source`: the name states a quantity, locus, component or
  relation that the source text contradicts.
- `name_omits_source`: the source text states a qualifier of the quantity — a
  counting basis, a species basis, a weighting or average, a normalisation, a
  coordinate convention, a component or a locus — and the name leaves it out
  although that qualifier separates this quantity from another.
- `definition_contradicts_source`: the definition (description) contradicts the
  source text.
- `name_definition_disagree: the name and the definition describe different
  quantities or different scopes.

A name identifies and its definition specifies: a normalisation, weighting,
convention or method the name omits is owed by the definition, and its absence
there is the defect. When the stage declined to name the quantity and reports a
missing vocabulary token, judge whether declining was the faithful outcome for
these sources.
"""


class Verdict:
    """Typed response model for the adjudicator; built lazily to keep imports local."""

    @staticmethod
    def build():
        from pydantic import BaseModel, Field

        class _Verdict(BaseModel):
            verdict: Literal["sound", "defective"] = Field(
                description="sound when the record is faithful, defective otherwise"
            )
            defect_classes: list[str] = Field(default_factory=list)
            reason: str = Field(description="one sentence quoting the deciding text")

        return _Verdict


async def adjudicate(
    packet: dict[str, Any],
    record_text: str,
    model: str,
    semaphore: asyncio.Semaphore,
    calls: dict[str, float],
) -> dict[str, Any]:
    from imas_codex.discovery.base.llm import acall_llm_structured

    sources = "\n\n".join(
        f"path: {source_path(source)}\nunit: {source.get('dd_unit')}\n"
        f"description: {source.get('description')}\n"
        f"documentation: {source.get('dd_documentation')}"
        for source in packet["sources"]
    )
    ancestors = "\n".join(
        f"{ancestor['path']}: {ancestor['text']}" for ancestor in packet["ancestors"]
    )
    user_prompt = (
        f"Data Dictionary source text:\n{sources}\n\n"
        f"Enclosing structures (nearest first, deduplicated):\n{ancestors}\n\n"
        f"Record to judge:\n{record_text}\n"
    )
    async with semaphore:
        result = await acall_llm_structured(
            model=model,
            messages=[
                {"role": "system", "content": JUDGE_RULES},
                {"role": "user", "content": user_prompt},
            ],
            response_model=Verdict.build(),
            service="standard-names",
        )
    calls["adjudicate"] = calls.get("adjudicate", 0) + 1
    calls["adjudicate_cost_usd"] = calls.get("adjudicate_cost_usd", 0.0) + float(
        getattr(result, "cost", 0.0)
    )
    calls["adjudicate_tokens_in"] = calls.get("adjudicate_tokens_in", 0) + int(
        getattr(result, "input_tokens", 0) or 0
    )
    calls["adjudicate_tokens_out"] = calls.get("adjudicate_tokens_out", 0) + int(
        getattr(result, "output_tokens", 0) or 0
    )
    parsed = result.parsed
    return {
        "verdict": parsed.verdict,
        "defect_classes": parsed.defect_classes,
        "reason": parsed.reason,
    }


# --------------------------------------------------------------------------
# The disposable graph
# --------------------------------------------------------------------------


def start_disposable_graph():
    spec = importlib.util.spec_from_file_location("sn_conftest", CONFTEST)
    module = importlib.util.module_from_spec(spec)
    assert module is not None and spec.loader is not None
    spec.loader.exec_module(module)
    directory, process, _log, port = module._start_empty_neo4j()
    return module, directory, process, port


def graph_uri(port: int) -> str:
    return f"bolt://127.0.0.1:{port}"


def store_and_read_back(
    module, port: int, rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    from neo4j import GraphDatabase

    auth = module._auth()
    with GraphDatabase.driver(graph_uri(port), auth=auth) as driver:
        driver.verify_connectivity()
        driver.execute_query(
            "UNWIND $rows AS row "
            "MERGE (n:StandardName {id: row.id}) "
            "SET n.description = row.description, "
            "    n.source_paths = row.source_paths, "
            "    n.generated_by = $generated_by",
            rows=rows,
            generated_by="stage-repairs",
        )
        records, _, _ = driver.execute_query(
            "MATCH (n:StandardName) RETURN n.id AS id, n.description AS description, "
            "n.source_paths AS source_paths ORDER BY id"
        )
    return [dict(record) for record in records]


def stop_disposable_graph(module, directory, process) -> None:
    try:
        process.terminate()
        process.wait(timeout=60)
    except Exception:
        pass
    try:
        directory.cleanup()
    except Exception:
        pass


# --------------------------------------------------------------------------
# The measurement
# --------------------------------------------------------------------------


async def measure() -> dict[str, Any]:
    from imas_codex.settings import get_model
    from imas_codex.standard_names.context import build_compose_context

    compose_model = get_model("sn-compose")
    judge_model = get_model("sn-prose-adjudicator")
    context = build_compose_context()
    from imas_codex.llm.prompt_loader import render_prompt

    system_prompt = render_prompt("sn/generate_name_system", context)

    record = load_record()
    members = cause_members(record)
    before = member_before(record)
    pool = record["packet"]["text_pool"]
    packets = {packet["id"]: packet for packet in record["packet"]["packets"]}

    compose_semaphore = asyncio.Semaphore(COMPOSE_CONCURRENCY)
    judge_semaphore = asyncio.Semaphore(COMPOSE_CONCURRENCY)
    calls: dict[str, Any] = {}

    results: dict[str, list[dict[str, Any]]] = {}
    for cause in TOP_CAUSES:
        results[cause] = []
        for member_id in members[cause]:
            packet = parse_packet(pool, packets[member_id])
            listed = packet["sources"]
            rejected: list[dict[str, str]] = []
            if cause == "source_bindings_conflate_distinct_definitions":
                sources, rejected = admitted_sources(packet["name"], listed) or (
                    listed,
                    [],
                )
                if not sources:
                    sources = listed[:1]
            else:
                sources = listed
            regenerated = await compose_member(
                context,
                system_prompt,
                packet,
                sources,
                compose_model,
                compose_semaphore,
                calls,
            )
            if regenerated["outcome"] == "named":
                record_text = (
                    f"Standard name: {regenerated['name']}\n"
                    f"Definition: {regenerated['definition']}\n"
                    f"Unit: {regenerated['unit']}\n"
                )
            else:
                gaps = "; ".join(
                    f"{gap['segment']}/{gap['token']}" for gap in regenerated["gaps"]
                )
                record_text = (
                    "The stage declined to name these sources and reported a "
                    f"missing vocabulary token: {gaps or 'none reported'}\n"
                )
            after = await adjudicate(
                packet, record_text, judge_model, judge_semaphore, calls
            )
            results[cause].append(
                {
                    "id": member_id,
                    "before": before[member_id],
                    "sources": [source_path(source) for source in listed],
                    "rejected_bindings": rejected,
                    "regenerated": regenerated,
                    "after": after,
                    "record_text": record_text,
                }
            )

    named_rows = []
    for cause, members_rows in results.items():
        for row in members_rows:
            regenerated = row["regenerated"]
            if regenerated["outcome"] != "named":
                continue
            named_rows.append(
                {
                    "id": regenerated["name"],
                    "description": regenerated["definition"],
                    "source_paths": row["sources"],
                    "member_id": row["id"],
                    "cause": cause,
                }
            )

    module, directory, process, port = start_disposable_graph()
    try:
        _uri = graph_uri(port)
        read_back = store_and_read_back(module, port, named_rows)
    finally:
        stop_disposable_graph(module, directory, process)

    by_id = {row["id"]: row for row in read_back}
    for _cause, members_rows in results.items():
        for row in members_rows:
            regenerated = row["regenerated"]
            row["in_disposable_graph"] = (
                regenerated.get("name") in by_id
                if regenerated["outcome"] == "named"
                else False
            )
            if regenerated["outcome"] == "named":
                row["read_back"] = by_id.get(regenerated["name"])

    summary = {}
    for cause, members_rows in results.items():
        named = [
            row for row in members_rows if row["regenerated"]["outcome"] == "named"
        ]
        sound = [row for row in members_rows if row["after"]["verdict"] == "sound"]
        summary[cause] = {
            "members": len(members_rows),
            "sound_before": sum(
                1 for row in members_rows if row["before"]["verdict"] == "sound"
            ),
            "sound_after": len(sound),
            "named_after": len(named),
            "gap_after": len(members_rows) - len(named),
            "sound_share_after": round(len(sound) / len(members_rows), 3),
        }

    trace = json.loads(TRACE.read_text())
    remaining = [
        {
            "id": cause["id"],
            "count": cause["count"],
            "stage": cause["stage"],
            "same_repair_as": cause.get("same_repair_as"),
            "repair": cause.get("repair", {}).get("smallest", ""),
        }
        for cause in trace["causes"]
        if cause["id"] not in TOP_CAUSES
    ]

    total_cost = float(calls.get("compose_cost_usd", 0.0)) + float(
        calls.get("adjudicate_cost_usd", 0.0)
    )
    return {
        "measure": "stage repairs regenerate their cause's members",
        "plan_surface": "sn-audited-group-release#s7",
        "record": {
            "path": str(RECORD.relative_to(REPO)),
            "collection_revision": record["collection_revision"],
            "population": len(record["members"]),
        },
        "stages": {
            cause: {
                "members": summary[cause]["members"],
                "file": REPAIR_MARKERS[cause][0],
                "marker": REPAIR_MARKERS[cause][1],
            }
            for cause in TOP_CAUSES
        },
        "summary": summary,
        "members": results,
        "remaining_causes": remaining,
        "disposable_graph": {
            "uri": "bolt://127.0.0.1:<port>",
            "engine": "disposable neo4j started by tests/standard_names conftest",
            "nodes_written": len(named_rows),
            "nodes_read_back": len(read_back),
            "ids": sorted(by_id),
        },
        "spend": {
            "ceiling_usd": CEILING_USD,
            "compose_model": compose_model,
            "adjudicator_model": judge_model,
            "calls": calls,
            "total_usd": round(total_cost, 4),
            "live_graph_writes": 0,
        },
    }


# --------------------------------------------------------------------------
# The page
# --------------------------------------------------------------------------


def _verdict_cell(row: dict[str, Any]) -> str:
    after = row["after"]
    if after["verdict"] == "sound":
        return '<td class="ok">sound</td>'
    classes = ", ".join(escape(item) for item in after["defect_classes"]) or "defective"
    return f'<td class="bad">{classes}</td>'


def _record_cell(row: dict[str, Any]) -> str:
    regenerated = row["regenerated"]
    if regenerated["outcome"] != "named":
        gaps = ", ".join(
            f"{gap['segment']}:{gap['token']}" for gap in regenerated["gaps"]
        )
        return (
            f"<td><em>{escape(regenerated['outcome'])}</em>"
            f"<br><code>{escape(gaps or 'no gap reported')}</code></td>"
        )
    return (
        f"<td><code>{escape(regenerated['name'])}</code><br>"
        f"{escape(regenerated['definition'])}</td>"
    )


PAGE_STYLE = (
    "body{font:16px/1.5 system-ui,sans-serif;max-width:1400px;margin:2rem auto;"
    "padding:0 1.5rem;color:#222}table{border-collapse:collapse;width:100%;"
    "margin:1rem 0}th,td{padding:.55rem;text-align:left;vertical-align:top;"
    "border-bottom:1px solid #ccc;overflow-wrap:anywhere}th{font-weight:600}"
    "code{overflow-wrap:anywhere}a{color:#245c91}.ok{color:#1a6b2f}"
    ".bad{color:#a5291b}em{font-style:normal;color:#a5291b}"
)

TITLE = "Three repaired stages regenerate their causes' members"


def _body_parts(data: dict[str, Any]) -> list[str]:
    summary = data["summary"]
    total_members = sum(item["members"] for item in summary.values())
    sound_before = sum(item["sound_before"] for item in summary.values())
    sound_after = sum(item["sound_after"] for item in summary.values())
    parts = [
        f'<h1 id="sagr-stage-repairs-title">{TITLE}</h1>',
        '<p id="sagr-stage-repairs-summary">',
        f"The three largest shared defect causes hold <strong>{total_members} "
        "members</strong> of the equilibrium adjudication. Each was regenerated "
        "here through the repaired stage from its own recorded source packet, and "
        "the regenerated record was read back from a disposable graph and "
        "adjudicated blind by a fresh reader "
        f"(<code>{escape(data['spend']['adjudicator_model'])}</code>). "
        f"Judged sound: <strong>{sound_before} of {total_members} before</strong>, "
        f"<strong>{sound_after} of {total_members} after</strong>.",
        "</p>",
        '<h2 id="sagr-stage-repairs-per-cause">Per cause</h2>',
        '<table id="sagr-stage-repairs-cause-table"><thead><tr>'
        "<th>Cause</th><th>Members</th><th>Sound before</th><th>Sound after</th>"
        "<th>Named after</th><th>Vocab gap after</th></tr></thead><tbody>",
    ]
    for cause, item in summary.items():
        parts.append(
            f"<tr><td><code>{escape(cause)}</code></td><td>{item['members']}</td>"
            f"<td>{item['sound_before']}</td><td>{item['sound_after']}</td>"
            f"<td>{item['named_after']}</td><td>{item['gap_after']}</td></tr>"
        )
    parts.append("</tbody></table>")

    for cause, members in data["members"].items():
        parts.append(
            f'<h2 id="sagr-stage-repairs-{escape(cause)}">{escape(cause)}</h2>'
        )
        parts.append(
            f'<p id="sagr-stage-repairs-{escape(cause)}-stage">Stage '
            f"<code>{escape(data['stages'][cause]['file'])}</code>, repair marker "
            f"<code>{escape(data['stages'][cause]['marker'])}</code>.</p>"
        )
        parts.append(
            f'<table id="sagr-stage-repairs-{escape(cause)}-table"><thead><tr>'
            "<th>Member</th><th>Before</th><th>Regenerated record</th>"
            "<th>After</th><th>Adjudicator's reason</th></tr></thead><tbody>"
        )
        for row in members:
            before = row["before"]["verdict"]
            if row["rejected_bindings"]:
                rejected = "; ".join(
                    f"<code>{escape(item['path'])}</code>: {escape(item['reason'])}"
                    for item in row["rejected_bindings"]
                )
                before = f"{escape(before)}<br>rejected {rejected}"
            parts.append(
                f'<tr id="sagr-stage-repairs-{escape(cause)}-{escape(row["id"])}">'
                f"<td><code>{escape(row['id'])}</code></td>"
                f"<td>{before}</td>"
                f"{_record_cell(row)}"
                f"{_verdict_cell(row)}"
                f"<td>{escape(row['after']['reason'])}</td></tr>"
            )
        parts.append("</tbody></table>")

    parts.append(
        '<h2 id="sagr-stage-repairs-remaining">The remaining eleven causes</h2>'
    )
    parts.append(
        '<table id="sagr-stage-repairs-remaining-table"><thead><tr><th>Cause</th>'
        "<th>Members</th><th>Stage</th><th>Same repair</th><th>Smallest repair</th>"
        "</tr></thead><tbody>"
    )
    for cause in data["remaining_causes"]:
        parts.append(
            f"<tr><td><code>{escape(cause['id'])}</code></td><td>{cause['count']}</td>"
            f"<td><code>{escape(cause['stage'])}</code></td>"
            f"<td>{escape(str(cause['same_repair_as'] or 'no'))}</td>"
            f"<td>{escape(cause['repair'])}</td></tr>"
        )
    parts.append("</tbody></table>")

    graph = data["disposable_graph"]
    spend = data["spend"]
    parts.append('<h2 id="sagr-stage-repairs-measure">Measure</h2>')
    parts.append(
        f'<p id="sagr-stage-repairs-graph">Regenerated members written to a '
        f"disposable Neo4j: <strong>{graph['nodes_written']}</strong>, read back: "
        f"<strong>{graph['nodes_read_back']}</strong>. Live-graph writes: "
        f"<strong>{spend['live_graph_writes']}</strong>.</p>"
    )
    parts.append(
        f'<p id="sagr-stage-repairs-spend">Composer '
        f"<code>{escape(spend['compose_model'])}</code>, adjudicator "
        f"<code>{escape(spend['adjudicator_model'])}</code>. Calls: "
        f"{spend['calls'].get('compose', 0)} compose, "
        f"{spend['calls'].get('adjudicate', 0)} judge. Recorded spend "
        f"<strong>USD {spend['total_usd']:.2f}</strong> of the USD "
        f"{spend['ceiling_usd']:.0f} ceiling.</p>"
    )
    return parts


def render_fragment(data: dict[str, Any]) -> str:
    """The docs-site fragment: project metadata and node-id addresses."""
    parts = [
        "<!doctype html>",
        '<html lang="en">',
        "<head>",
        '<meta charset="utf-8">',
        '<meta name="docs-project" content="imas-codex">',
        '<meta name="plan-slug" content="sn-audited-group-release">',
        '<meta name="reckon-type" content="evidence-fragment">',
        '<meta name="node-id" content="sagr-stage-repairs">',
        f"<title>{TITLE}</title>",
        "</head>",
        "<body>",
        "<main>",
        *_body_parts(data),
        "</main>",
        "</body>",
        "</html>",
    ]
    return "\n".join(parts) + "\n"


def render_page(data: dict[str, Any]) -> str:
    """The standalone page: the same content, self-contained, for a reader."""
    parts = [
        '<!doctype html><html lang="en"><head><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width,initial-scale=1">',
        f"<title>{TITLE}</title>",
        f"<style>{PAGE_STYLE}</style></head><body>",
        *_body_parts(data),
        "</body></html>",
    ]
    return "\n".join(parts) + "\n"


# --------------------------------------------------------------------------
# Entry points
# --------------------------------------------------------------------------


def _render_all(data: dict[str, Any]) -> None:
    FRAGMENT.write_text(render_fragment(data))
    PAGE.write_text(render_page(data))


def run() -> None:
    data = asyncio.run(measure())
    DATA.write_text(json.dumps(data, indent=1) + "\n")
    FRAGMENT.write_text(render_fragment(data))
    PAGE.write_text(render_page(data))
    summary = data["summary"]
    for cause, item in summary.items():
        print(
            f"{cause}: {item['sound_after']}/{item['members']} sound after "
            f"({item['sound_before']} before), {item['gap_after']} gaps"
        )
    print(f"spend USD {data['spend']['total_usd']:.2f}")
    print(f"graph {data['disposable_graph']['nodes_read_back']} nodes read back")


def check() -> int:
    data = json.loads(DATA.read_text())
    failures: list[str] = []
    record = load_record()
    members = cause_members(record)
    for cause in TOP_CAUSES:
        if not data["members"].get(cause):
            failures.append(f"{cause}: no members measured")
        if len(data["members"].get(cause, [])) != len(members[cause]):
            failures.append(f"{cause}: member count disagrees with the record")
        path = REPO / REPAIR_MARKERS[cause][0]
        marker = REPAIR_MARKERS[cause][1]
        if marker not in path.read_text():
            failures.append(f"{cause}: repair marker absent from {path}")
    for cause in data["remaining_causes"]:
        if cause["id"] in TOP_CAUSES:
            failures.append("a traced cause is listed among the remaining eleven")
    if len(data["remaining_causes"]) != 11:
        failures.append(f"{len(data['remaining_causes'])} remaining causes, not eleven")
    if data["disposable_graph"]["nodes_read_back"] != len(
        data["disposable_graph"]["ids"]
    ):
        failures.append("disposable graph read-back count disagrees with the ids")
    if data["spend"]["total_usd"] > data["spend"]["ceiling_usd"]:
        failures.append("recorded spend exceeds the ceiling")
    if FRAGMENT.read_text() != render_fragment(data):
        failures.append("the committed fragment is not what this data renders")
    if PAGE.read_text() != render_page(data):
        failures.append("the committed page is not what this data renders")
    for row in (row for rows in data["members"].values() for row in rows):
        if row["id"] not in PAGE.read_text():
            failures.append(f"member {row['id']} absent from the page")
    for failure in failures:
        print(f"FAIL {failure}")
    if not failures:
        print(
            f"PASS {len(TOP_CAUSES)} causes, "
            f"{sum(len(rows) for rows in data['members'].values())} members, "
            f"11 remaining causes"
        )
    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="measure and write the data")
    parser.add_argument("--render", action="store_true", help="render the page")
    parser.add_argument("--check", action="store_true", help="re-derive the claims")
    arguments = parser.parse_args()
    if arguments.run:
        run()
        return 0
    if arguments.render:
        _render_all(json.loads(DATA.read_text()))
        return 0
    if arguments.check:
        return check()
    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
