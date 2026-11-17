#!/usr/bin/env python
"""Trace the shared causes of the equilibrium adjudication to the stage that makes them.

The record this reads is ``equilibrium-adjudication.json``: 208 members judged
blind, each with its complete source packet, its stored name and definition, the
per-source text and the enclosing-structure statements.  The three causes with
the most members are traced here to the product stage that produces them; the
eleven further causes shared by three or more members are traced to their stage
too and marked with whether the same repair reaches them.

``--check`` re-derives every claim from the record and from the product source
in this worktree.  It prints one line per traced member and fails on any
disagreement, including a traced member's recorded stage differing from the
stage the cause's stage table names.  ``--render`` writes the evidence page.

Nothing here writes the graph or changes product source.
"""

from __future__ import annotations

import argparse
import ast
import base64
import json
import re
import sys
import zlib
from collections import Counter
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[3]
RECORD = ROOT / "equilibrium-adjudication.json"
TRACE = ROOT / "stage-cause-trace.json"
PAGE = ROOT / "stage-cause-trace.html"

MIN_TRACED_MEMBERS = 9
TRACED_CAUSES = 3

#: The stage each cause is produced by.  Held in code, not read from the trace
#: file, so a recorded stage that disagrees with this table fails the check.
STAGE_OF_CAUSE = {
    "physical_scope_qualifier_lost": "compose_name",
    "source_bindings_conflate_distinct_definitions": "attach_guard",
    "boundary_forced_to_lcfs": "locus_binding",
    "normalization_or_weighting_qualifier_lost": "compose_name",
    "coordinate_reference_qualifier_lost": "compose_name",
    "vacuum_source_qualifier_lost": "compose_name",
    "pressure_pedestal_qualifier_lost": "compose_name",
    "transformation_equation_replaced_by_base_quantity": "definition_construction",
    "averaging_qualifier_lost": "compose_name",
    "absolute_extremum_substituted_for_signed_extremum": "locus_binding",
    "population_basis_qualifier_lost": "compose_name",
    "physical_scope_qualifier_replaced": "definition_construction",
    "locus_template_overrides_source": "locus_binding",
    "consumer_context_used_as_quantity_domain": "domain_assignment",
}

NAMES_PROMPT = REPO / "imas_codex/llm/prompts/sn/generate_name_dd_names.md"
FULL_PROMPT = REPO / "imas_codex/llm/prompts/sn/generate_name_dd.md"
NAME_SYSTEM = REPO / "imas_codex/llm/prompts/sn/generate_name_system.md"


def decode_record(path: Path = RECORD) -> dict:
    """Return the adjudication record with its compressed payloads expanded."""
    value = json.loads(path.read_text())
    for key in ("judgments", "report", "coverage", "packet"):
        payload = value.pop(key + "_payload_zlib_base85", None)
        if payload is not None:
            value[key] = json.loads(zlib.decompress(base64.b85decode(payload)))
    return value


def census(record: dict) -> tuple[Counter, dict[str, list[str]]]:
    """Count each cause over the judgments."""
    counts: Counter = Counter()
    members: dict[str, list[str]] = {}
    for row in record["judgments"]:
        causes = [c for c in row["candidate_cause"].split("|") if c]
        if "wrong_physics_domain" in row["defect_classes"]:
            causes.append("consumer_context_used_as_quantity_domain")
        for cause in causes:
            counts[cause] += 1
            members.setdefault(cause, []).append(row["id"])
    return counts, members


def _function_node(tree: ast.AST, name: str):
    """Return the first function definition called *name*, nested definitions included."""
    for node in ast.walk(tree):
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == name
        ):
            return node
    return None


def function_body(path: Path, name: str) -> str | None:
    """Return the source of the function *name* in *path*."""
    try:
        source = path.read_text()
        tree = ast.parse(source)
    except (OSError, SyntaxError):
        return None
    node = _function_node(tree, name)
    if node is None:
        return None
    lines = source.splitlines()
    return "\n".join(lines[node.lineno - 1 : node.end_lineno])


def function_params(path: Path, name: str) -> list[str] | None:
    try:
        tree = ast.parse(path.read_text())
    except (OSError, SyntaxError):
        return None
    node = _function_node(tree, name)
    if node is None:
        return None
    args = node.args
    return [a.arg for a in args.posonlyargs + args.args + args.kwonlyargs]


def locus_registry_definition(token: str) -> str:
    """Return the ISN locus-registry definition of *token*."""
    from imas_standard_names.grammar import vocab_loaders

    entry = vocab_loaders.load_locus_registry().loci.get(token)
    return (getattr(entry, "definition", "") or "").strip()


class Failure(Exception):
    """A claim in the trace no longer holds."""


def check_stage_files(stages: dict) -> list[str]:
    """Each stage names a file and function that exist in this worktree."""
    notes = []
    for key, stage in stages.items():
        path = REPO / stage["file"]
        if not path.exists():
            raise Failure(f"stage {key}: {stage['file']} does not exist")
        body = function_body(path, stage["function"])
        if body is None:
            raise Failure(
                f"stage {key}: {stage['file']}::{stage['function']} not found"
            )
        for extra in stage.get("files_secondary", []):
            if not (REPO / extra).exists():
                raise Failure(f"stage {key}: secondary file {extra} does not exist")
        notes.append(f"  stage {key:24s} {stage['file']}::{stage['function']}")
    return notes


def check_stage_mechanisms(stages: dict) -> list[str]:
    """Each stage's claimed input drop still holds in product source."""
    notes = []

    names_text = NAMES_PROMPT.read_text()
    full_text = FULL_PROMPT.read_text()
    per_item = names_text.split("### {{ item.path }}", 1)
    if len(per_item) != 2:
        raise Failure(f"{NAMES_PROMPT.name}: per-item block marker not found")
    item_block = per_item[1].split("{% endfor %}\n{% endif %}", 1)[0]
    if "ancestor_context" in item_block:
        raise Failure(
            "compose_name stage now renders item.ancestor_context in names batching "
            "mode; the traced drop no longer exists and the trace needs re-measuring"
        )
    if "ancestor_context" not in full_text:
        raise Failure(
            f"{FULL_PROMPT.name} no longer renders the lineage block, so the "
            "names-mode omission is no longer a difference between the two paths"
        )
    compose_body = function_body(
        REPO / stages["compose_name"]["file"], stages["compose_name"]["function"]
    )
    if '"sn/generate_name_dd_names"' not in compose_body:
        raise Failure("_compose_batch_body no longer selects the names-mode template")
    notes.append(
        "  compose_name    names-mode per-item block omits ancestor_context; "
        "full mode renders it; _compose_batch_body selects the names template"
    )

    guard_path = REPO / stages["attach_guard"]["file"]
    guard = stages["attach_guard"]["function"]
    params = function_params(guard_path, guard)
    for param in params or []:
        if re.search(r"descri|documentation|text", param):
            raise Failure(
                f"{guard} now takes a text parameter {param!r}; the binding gate "
                "may read the sources' stated operations, so the trace is stale"
            )
    body = function_body(guard_path, guard)
    for word in ("description", "documentation"):
        if word in body:
            raise Failure(
                f"{guard} now reads {word!r}; the traced input drop no longer holds"
            )
    notes.append(
        f"  attach_guard    {guard} takes no source-text parameter and reads "
        "neither description nor documentation"
    )

    locus_body = function_body(
        REPO / stages["locus_binding"]["file"], stages["locus_binding"]["function"]
    )
    if "load_locus_registry" not in (locus_body or ""):
        raise Failure(
            "locus_context_for no longer reads the locus registry; the traced "
            "injection point has moved"
        )
    if "_of_plasma_boundary" not in NAME_SYSTEM.read_text():
        raise Failure(
            f"{NAME_SYSTEM.name} no longer binds _of_plasma_boundary to the LCFS "
            "contour; the naming half of the locus stage has changed"
        )
    definition = locus_registry_definition("plasma_boundary")
    if "last closed" not in definition:
        raise Failure(
            "the locus registry no longer defines plasma_boundary as the last "
            f"closed flux surface (read: {definition[:120]!r}"
        )
    notes.append(
        "  locus_binding   locus_context_for injects the registry entry for "
        f"plasma_boundary: {definition[:72]}..."
    )
    return notes


def check_trace(trace: dict, record: dict) -> None:
    """Every recorded member re-derives from the record it quotes."""
    counts, _members = census(record)
    shared = sorted(
        (c for c, n in counts.items() if n >= 3), key=lambda c: (-counts[c], c)
    )
    missing = [c for c in shared if c not in STAGE_OF_CAUSE]
    if missing:
        raise Failure(
            f"causes shared by three or more members with no stage: {missing}"
        )
    traced_causes = [c for c in trace["causes"] if c["traced"]]
    if len(traced_causes) != TRACED_CAUSES:
        raise Failure(
            f"{len(traced_causes)} causes carry traced members, expected {TRACED_CAUSES}"
        )
    listed = [c["id"] for c in trace["causes"]]
    if sorted(listed) != sorted(shared):
        raise Failure(
            f"the trace lists {sorted(listed)} but the record has {sorted(shared)}"
        )
    total = sum(counts[c] for c in shared)
    if trace["source_record"]["cause_instances"] != total:
        raise Failure(
            f"recorded cause instances {trace['source_record']['cause_instances']} "
            f"but the judgments give {total}"
        )
    pool = record["packet"]["text_pool"]
    packets = {p["id"]: p for p in record["packet"]["packets"]}
    for cause in trace["causes"]:
        kind = cause["id"]
        if counts[kind] != cause["count"]:
            raise Failure(f"{kind}: recorded count {cause['count']} != {counts[kind]}")
        if cause["stage"] != STAGE_OF_CAUSE[kind]:
            raise Failure(
                f"{kind}: recorded stage {cause['stage']!r} but the stage table "
                f"says {STAGE_OF_CAUSE[kind]!r}"
            )
        repair = cause.get("repair") or {}
        if not repair.get("smallest"):
            raise Failure(f"{kind}: no smallest repair recorded")
        if not repair.get("files"):
            raise Failure(f"{kind}: the repair names no files")
        for path in repair["files"]:
            if not (REPO / path).exists():
                raise Failure(
                    f"{kind}: repair names a file that does not exist: {path}"
                )
        if len(cause["traced"]) not in (0, 0) and len(cause["traced"]) < 3:
            raise Failure(f"{kind}: {len(cause['traced'])} traced members, need 3")
        for member in cause["traced"]:
            where = f"{kind}/{member['id']}"
            if member.get("stage") != STAGE_OF_CAUSE[kind]:
                raise Failure(
                    f"{where}: recorded member stage {member.get('stage')!r} but "
                    f"the stage table says {STAGE_OF_CAUSE[kind]!r}"
                )
            packet = packets.get(member["id"])
            if packet is None:
                raise Failure(f"{where}: no recorded packet")
            line = member["evidence_line"]
            if line not in packet["text_lines"]:
                raise Failure(f"{where}: line {line} is not in the recorded packet")
            if member["evidence_excerpt"] not in pool[line]:
                raise Failure(f"{where}: excerpt absent from recorded line {line}")
            recorded_name = pool[packet["text_lines"][0]].split("Standard name: ", 1)[1]
            if recorded_name.strip() != member["name"]:
                raise Failure(
                    f"{where}: the record's name is {recorded_name.strip()!r}"
                )
            for token in member.get("absent_from_name", []):
                if token.lower() in member["name"].lower():
                    raise Failure(f"{where}: name now carries {token!r}")
            if "absent_from_definition" in member:
                definition_line = member["definition_line"]
                if definition_line not in packet["text_lines"]:
                    raise Failure(
                        f"{where}: definition line {definition_line} not in packet"
                    )
                definition = pool[definition_line]
                for token in member["absent_from_definition"]:
                    if token.lower() in definition.lower():
                        raise Failure(
                            f"{where}: the definition now states {token!r}, so the "
                            "binding is not conflated with a dropped operation"
                        )
            if "name_locus_token" in member:
                if member["name_locus_token"] not in member["name"]:
                    raise Failure(
                        f"{where}: name does not carry {member['name_locus_token']!r}"
                    )


def trace_lines(trace: dict) -> list[str]:
    """One line per traced member, as the check prints them."""
    lines = []
    for cause in trace["causes"]:
        for member in cause["traced"]:
            lines.append(
                f"traced member {member['id']} | cause {cause['id']} | "
                f"stage {member['stage']} | line {member['evidence_line']} | "
                f"drop {member['drop_site']} | excerpt {member['evidence_excerpt']!r}"
            )
    return lines


def render(trace: dict, record: dict) -> str:
    """Render the evidence page from the trace and the record it quotes."""
    counts, _members = census(record)
    stages = trace["stages"]
    parts: list[str] = []
    add = parts.append
    add(
        '<!doctype html><html lang="en"><head><meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width,initial-scale=1">\n'
        "<title>The stages that produce the shared catalog defects</title>\n"
        "<style>body{font:16px/1.5 system-ui,sans-serif;max-width:1400px;margin:2rem auto;"
        "padding:0 1.5rem;color:#222}table{border-collapse:collapse;width:100%;margin:1rem 0}"
        "th,td{padding:.55rem;text-align:left;vertical-align:top;border-bottom:1px solid #ccc;"
        "overflow-wrap:anywhere}th{font-weight:600}pre{white-space:pre-wrap;"
        "overflow-wrap:anywhere;font:14px/1.45 ui-monospace,monospace;background:#f6f6f6;"
        "padding:.6rem}code{overflow-wrap:anywhere}a{color:#245c91}small{font-size:.85em}"
        ".num{text-align:right}</style></head><body>\n"
    )
    shared = [c for c in trace["causes"] if c["count"] >= 3]
    traced = sum(len(c["traced"]) for c in trace["causes"])
    add("<h1>The stages that produce the shared catalog defects</h1>\n")
    add(
        f"<p>The blind adjudication of the equilibrium group judged "
        f"<strong>{record['report']['population']} members</strong> against their complete "
        f"source packets. Reading its judgments gives <strong>{len(shared)} causes shared "
        f"by three or more defective members, {sum(c['count'] for c in shared)} defect "
        f"instances in all</strong>; that census reproduces the plan's figures exactly. "
        f"The three largest are traced here to the stage that produces them, with "
        f"<strong>{traced} traced members</strong> quoted from the record, and the "
        f"remaining {len(shared) - TRACED_CAUSES} are traced to their stage with whether "
        "the same repair reaches them. Nothing on this page changed product source or "
        "wrote the graph.</p>\n"
    )
    add(
        '<p>Record: <a href="equilibrium-adjudication.json">'
        "equilibrium-adjudication.json</a> at collection revision "
        f"<code>{escape(str(trace['source_record']['collection_revision']))}</code>, "
        f"verdicts completed {escape(str(trace['source_record']['verdict_completed_at']))}. "
        'Trace data: <a href="stage-cause-trace.json">stage-cause-trace.json</a>. '
        "Reproduce: <code>uv run python docs/evidence/fragments/"
        "sn-audited-group-release/stage-cause-trace.py --check</code>.</p>\n"
    )

    add("<h2>The five stages</h2>\n")
    add(
        "<p>A stage is a place in the pipeline that receives the source text and writes "
        "an artifact the catalog keeps. The last column is the part of its input it does "
        "not carry into the artifact.</p>\n"
    )
    add(
        "<table><thead><tr><th>Stage</th><th>File and function</th><th>Given</th>"
        "<th>Drops</th></tr></thead><tbody>\n"
    )
    order = [
        "compose_name",
        "attach_guard",
        "locus_binding",
        "definition_construction",
        "domain_assignment",
    ]
    for key in order:
        stage = stages[key]
        add(
            f"<tr><td><code>{key}</code><br>{escape(stage['label'])}</td>"
            f"<td><code>{escape(stage['file'])}</code><br><code>{escape(stage['function'])}</code></td>"
            f"<td>{escape(stage['given'])}</td><td>{escape(stage['drops'])}</td></tr>\n"
        )
    add("</tbody></table>\n")

    add("<h2>The three largest causes, traced member by member</h2>\n")
    for cause in shared[:TRACED_CAUSES]:
        stage = stages[cause["stage"]]
        add(
            f"<h3><code>{escape(cause['id'])}</code> — {cause['count']} members</h3>\n"
            f"<p>Stage: <code>{escape(cause['stage'])}</code>, "
            f"<code>{escape(stage['file'])}::{escape(stage['function'])}</code>.</p>\n"
            f"<p><strong>Given.</strong> {escape(stage['given'])}</p>\n"
            f"<p><strong>Drops.</strong> {escape(stage['drops'])}</p>\n"
            f"<p><strong>Smallest repair.</strong> {escape(cause['repair']['smallest'])}</p>\n"
            "<p>Files it would touch: "
            + ", ".join(f"<code>{escape(f)}</code>" for f in cause["repair"]["files"])
            + f"</p>\n<p><small>{escape(cause['repair'].get('note', ''))}</small></p>\n"
        )
        add(
            "<table><thead><tr><th>Member</th><th>Recorded input it is given vs what is dropped"
            "</th><th>Recorded evidence line</th><th>What the stored name or definition keeps</th>"
            "</tr></thead><tbody>\n"
        )
        for member in cause["traced"]:
            if "absent_from_name" in member:
                kept = (
                    f"name <code>{escape(member['name'])}</code> does not carry "
                    + ", ".join(
                        f"<code>{escape(t)}</code>" for t in member["absent_from_name"]
                    )
                )
            elif "absent_from_definition" in member:
                kept = (
                    f"definition line {member['definition_line']} does not carry "
                    + ", ".join(
                        f"<code>{escape(t)}</code>"
                        for t in member["absent_from_definition"]
                    )
                )
            else:
                kept = (
                    f"name carries <code>{escape(member['name_locus_token'])}</code>, "
                    "which the locus registry defines as the last closed flux surface"
                )
            add(
                f"<tr><td><code>{escape(member['id'])}</code><br>"
                f"<small>{escape(member['name'])}<br>drop site: {escape(member['drop_site'])}</small></td>"
                f"<td>{escape(member['given_segment'])}</td>"
                f"<td><small>line {member['evidence_line']}</small><br>"
                f"<code>{escape(member['evidence_excerpt'])}</code></td>"
                f"<td>{escape(member['drops_segment'])}.<br>{kept}</td></tr>\n"
            )
        add("</tbody></table>\n")

    add("<h2>The remaining eleven causes</h2>\n")
    add(
        "<p>Each is shared by three or more defective members. The last two columns say "
        "whether the repairs above reach it: <em>covered</em> when the same change fixes "
        "it, <em>partial</em> when the stage is reached but a second rule is needed, "
        "<em>not covered</em> when it needs its own repair.</p>\n"
    )
    add(
        '<table><thead><tr><th>Cause</th><th class="num">Members</th><th>Stage</th>'
        "<th>Smallest repair and files</th><th>Same repair</th><th>Why</th>"
        "</tr></thead><tbody>\n"
    )
    for cause in shared[TRACED_CAUSES:]:
        same = (
            f"{escape(cause['same_repair_verdict'])} of "
            f"<code>{escape(cause['same_repair_as'])}</code>"
            if cause.get("same_repair_as")
            else escape(cause["same_repair_verdict"].replace("_", " "))
        )
        add(
            f'<tr><td><code>{escape(cause["id"])}</code></td><td class="num">{cause["count"]}</td>'
            f"<td><code>{escape(cause['stage'])}</code></td>"
            f"<td>{escape(cause['repair']['smallest'])}<br>"
            + ", ".join(f"<code>{escape(f)}</code>" for f in cause["repair"]["files"])
            + f"</td><td>{same}</td>"
            f"<td>{escape(cause['same_repair_why'])}</td></tr>\n"
        )
    add("</tbody></table>\n")

    add("<h2>Why this is not a naming-style complaint</h2>\n")
    add(
        "<p>Each stage above is fed the text that states the missing qualifier and then "
        "writes an artifact without it, so the loss repeats for every group the stage "
        "processes — the defects are properties of the pipeline, not of the 208 names. "
        "The check that accompanies this page re-reads both the record and the product "
        "source: it fails if the recorded stage of any traced member moves, if a quoted "
        "line leaves the packet, if the names-mode prompt starts rendering the lineage, "
        "if the binding guard starts reading source text, or if the locus registry "
        "redefines <code>plasma_boundary</code>.</p>\n"
        "<pre>" + escape("\n".join(trace_lines(trace))) + "</pre>\n</body></html>\n"
    )
    return "".join(parts)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify every trace claim")
    parser.add_argument("--render", action="store_true", help="write the evidence page")
    parser.add_argument("--census", action="store_true", help="print the cause census")
    args = parser.parse_args(argv)

    record = decode_record()
    trace = json.loads(TRACE.read_text())
    if args.census:
        counts, _ = census(record)
        for cause, n in counts.most_common():
            print(f"{n:3d} {cause}")
        print(
            f"shared {sum(1 for n in counts.values() if n >= 3)} "
            f"instances {sum(n for n in counts.values() if n >= 3)}"
        )
        return 0

    if args.render:
        PAGE.write_text(render(trace, record))
        print(f"wrote {PAGE.relative_to(REPO)}")
        return 0

    if not args.check:
        parser.print_parser_help = None
        parser.print_help()
        return 2

    try:
        notes = check_stage_files(trace["stages"])
        notes += check_stage_mechanisms(trace["stages"])
        check_trace(trace, record)
    except Failure as error:
        print(f"FAIL {error}")
        return 1
    lines = trace_lines(trace)
    if len(lines) < MIN_TRACED_MEMBERS:
        print(f"FAIL {len(lines)} traced members printed, need {MIN_TRACED_MEMBERS}")
        return 1
    print("\n".join(notes))
    print("\n".join(lines))
    per_cause = {c["id"]: len(c["traced"]) for c in trace["causes"] if c["traced"]}
    print(
        f"traced members: {len(lines)} across {len(per_cause)} causes "
        + ", ".join(f"{k}={v}" for k, v in per_cause.items())
    )
    print(
        f"census: {trace['source_record']['shared_causes']} shared causes, "
        f"{trace['source_record']['cause_instances']} instances"
    )
    print(
        "stage mechanisms: names-mode lineage omission, text-free binding guard, "
        "registry-injected LCFS locus"
    )
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
