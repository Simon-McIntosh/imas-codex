"""Render discovery counts, mapping lifecycle and provenance-labelled JSON.

Run from the repository with the shared project environment and --no-sync.
Each read is facility- or identity-indexed, bounded and printed with its result.
The report is written after all SVG and slide PNG figures have been rendered.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import subprocess
import textwrap
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml
from matplotlib.font_manager import FontProperties
from neo4j import Query
from pygments import lex
from pygments.lexers import JsonLexer
from pygments.styles import get_style_by_name

from imas_codex.graph.client import GraphClient

ROOT = Path(__file__).resolve().parents[1]
MAIN = Path("/home/ITER/mcintos/Code/imas-codex")
LIVE_HANDOFF = Path(
    "/work/projects/imas_gpu/jt60sa/handoff/jt-60sa-mapping-handoff.json"
)
FIXTURE = MAIN / "tests/fixtures/mapping_handoff_example.json"
FIGURES = ROOT / "docs/figures/discovery-to-mapping"
REPORT = ROOT / "docs/research/discovery-to-mapping-pipeline.html"
PRESENTATION = Path("/work/projects/imas_gpu/jt60sa/presentation")
PYGMENTS_STYLE = "friendly"
FACILITY = "jt-60sa"
INK = "#242424"
REFERENCE = "#666666"
SCHEMA_FILES = ("facility.yaml", "imas_dd.yaml", "common.yaml")
SOURCE_FIELDS = ("id", "group_key", "status", "description")
TARGET_FIELDS = ("id", "data_type", "unit", "cocos_transformation_type")
EDGE_FIELDS = (
    "source_property",
    "transform_expression",
    "source_units",
    "target_units",
    "cocos_label",
    "confidence",
    "evidence",
    "mapping_type",
    "error_type",
    "derived_from",
)


def dumps(value: Any) -> str:
    return json.dumps(value, indent=2, ensure_ascii=False, default=str)


def check_schema() -> dict[str, Any]:
    """Verify every node property and relationship used by these reads."""
    classes: dict[str, Any] = {}
    receipts = {}
    for name in SCHEMA_FILES:
        path = ROOT / "imas_codex/schemas" / name
        raw = path.read_bytes()
        classes.update(yaml.safe_load(raw).get("classes", {}))
        receipts[name] = hashlib.sha256(raw).hexdigest()
    required = {
        "Facility": {"id"},
        "FacilityPath": {"id", "facility_id", "status"},
        "CodeFile": {"id", "facility_id", "status"},
        "Document": {"id", "facility_id", "status"},
        "WikiPage": {"id", "facility_id", "status"},
        "FacilitySignal": {"id", "facility_id", "status"},
        "SignalSource": {
            "facility_id",
            "candidate_route",
            "maps_to_imas",
            *SOURCE_FIELDS,
        },
        "IMASMapping": {"id", "facility_id", "status", "ids_name"},
        "IMASNode": set(TARGET_FIELDS),
    }
    for label, fields in required.items():
        if classes[label].get("deprecated"):
            raise ValueError(f"Deprecated inventory class: {label}")
        missing = fields - classes[label].get("attributes", {}).keys()
        if missing:
            raise ValueError(f"Undeclared {label} properties: {sorted(missing)}")
    rel = classes["SignalSource"]["attributes"]["maps_to_imas"]["annotations"]
    assert rel["relationship_type"] == "MAPS_TO_IMAS"
    print(
        "SCHEMA_CHECK " + json.dumps({k: sorted(v) for k, v in required.items()}),
        flush=True,
    )
    return receipts


class Reader:
    """Collect auditable read receipts with a server-side transaction bound."""

    def __init__(self, client: GraphClient):
        self.client = client
        self.receipts: list[dict[str, Any]] = []

    def read(self, name: str, cypher: str, **params: Any) -> list[dict[str, Any]]:
        query = " ".join(cypher.split())
        print(f"QUERY {name}: {query} PARAMS {json.dumps(params)}", flush=True)
        started = time.monotonic()
        with self.client.session() as session:
            result = session.run(Query(query, timeout=9), **params)
            rows = result.data()
            result.consume()
        elapsed = time.monotonic() - started
        receipt = {
            "name": name,
            "query": query,
            "parameters": params,
            "seconds": round(elapsed, 4),
            "rows": rows,
        }
        self.receipts.append(receipt)
        print(
            f"RESULT {name} ({elapsed:.4f}s): {json.dumps(rows, default=str)}",
            flush=True,
        )
        if elapsed >= 10:
            raise RuntimeError(f"Query {name} exceeded the ten-second bound")
        return rows


def facility_match(label: str, alias: str = "n") -> str:
    return (
        f"MATCH ({alias}:{label} {{facility_id: $facility}}) "
        f"USING INDEX {alias}:{label}(facility_id) "
    )


def read_counts(reader: Reader) -> dict[str, Any]:
    labels = [
        "FacilityPath",
        "CodeFile",
        "Document",
        "WikiPage",
        "FacilitySignal",
        "SignalSource",
        "IMASMapping",
    ]
    indexes = reader.read(
        "index_inventory",
        """SHOW INDEXES YIELD labelsOrTypes, properties, state, type
        WHERE any(label IN labelsOrTypes WHERE label IN $labels)
          AND type IN ['RANGE', 'TEXT']
        RETURN labelsOrTypes, properties, state, type LIMIT 100""",
        labels=[*labels, "Facility", "IMASNode"],
    )
    for label, prop in [(label, "facility_id") for label in labels] + [
        ("SignalSource", "id"),
        ("IMASNode", "id"),
        ("Facility", "id"),
    ]:
        if not any(
            row["labelsOrTypes"] == [label]
            and row["properties"] == [prop]
            and row["state"] == "ONLINE"
            and row["type"] == "RANGE"
            for row in indexes
        ):
            raise RuntimeError(f"Missing online range index: {label}.{prop}")
    control = reader.read(
        "facility_control",
        "MATCH (f:Facility {id: $facility}) USING INDEX f:Facility(id) RETURN f.id AS id LIMIT 1",
        facility=FACILITY,
    )
    if control != [{"id": FACILITY}]:
        raise RuntimeError("JT-60SA facility positive control did not fire")
    counts: dict[str, Any] = {}
    for label in labels:
        rows = reader.read(
            label,
            facility_match(label)
            + "RETURN count(n) AS total, count(n.facility_id) AS with_facility_id, count(n.status) AS with_status",
            facility=FACILITY,
        )
        row = rows[0]
        counts[label] = row["total"]
        if row["total"] != row["with_facility_id"]:
            raise RuntimeError(f"Facility property coverage mismatch: {label}")
        if row["total"] == 0:
            witness = reader.read(
                f"{label}_identity_control",
                f"MATCH (n:{label}) USING INDEX n:{label}(id) WHERE n.id IS NOT NULL RETURN n.id AS id, n.facility_id AS facility_id LIMIT 1",
            )
            counts[f"{label}_zero_control"] = witness
    if not counts["FacilitySignal"] or not counts["SignalSource"]:
        raise RuntimeError(
            "Known-present JT-60SA signal/source positive control failed"
        )
    counts["signal_statuses"] = reader.read(
        "signal_statuses",
        facility_match("FacilitySignal")
        + "RETURN n.status AS status, count(n) AS count ORDER BY status LIMIT 30",
        facility=FACILITY,
    )
    counts["candidate_routes"] = reader.read(
        "candidate_routes",
        facility_match("SignalSource")
        + "RETURN n.candidate_route AS route, count(n) AS count ORDER BY route LIMIT 20",
        facility=FACILITY,
    )
    counts["routed_sources"] = sum(
        row["count"] for row in counts["candidate_routes"] if row["route"] is not None
    )
    counts["bindings"] = reader.read(
        "bindings",
        facility_match("SignalSource", "s")
        + "MATCH (s)-[r:MAPS_TO_IMAS]->(n:IMASNode) RETURN count(r) AS count",
        facility=FACILITY,
    )[0]["count"]
    if counts["bindings"] == 0:
        witnesses = counts.get("IMASMapping_zero_control", [])
        if witnesses and witnesses[0]["facility_id"]:
            counts["binding_control"] = reader.read(
                "binding_control",
                facility_match("SignalSource", "s")
                + "MATCH (s)-[:MAPS_TO_IMAS]->(n:IMASNode) "
                "RETURN s.id AS source_id, n.id AS target_path LIMIT 1",
                facility=witnesses[0]["facility_id"],
            )
    counts["mapping_statuses"] = reader.read(
        "mapping_statuses",
        facility_match("IMASMapping")
        + "RETURN n.ids_name AS ids, n.status AS status, count(n) AS count ORDER BY ids, status LIMIT 30",
        facility=FACILITY,
    )
    return counts


def read_record(reader: Reader, row: dict[str, Any], kind: str) -> dict[str, Any]:
    source = reader.read(
        f"{kind}_source",
        "MATCH (s:SignalSource {id: $source, facility_id: $facility}) USING INDEX s:SignalSource(id) RETURN s {"
        + ", ".join("." + field for field in SOURCE_FIELDS)
        + "} AS record LIMIT 1",
        source=row["source_id"],
        facility=FACILITY,
    )
    edges = reader.read(
        f"{kind}_edge",
        "MATCH (s:SignalSource {id: $source, facility_id: $facility}) USING INDEX s:SignalSource(id) MATCH (s)-[r:MAPS_TO_IMAS]->(n:IMASNode {id: $target}) RETURN r {"
        + ", ".join("." + field for field in EDGE_FIELDS)
        + "} AS record LIMIT 2",
        source=row["source_id"],
        target=row["target_path"],
        facility=FACILITY,
    )
    target = reader.read(
        f"{kind}_target",
        "MATCH (n:IMASNode {id: $target}) USING INDEX n:IMASNode(id) RETURN n {"
        + ", ".join("." + field for field in TARGET_FIELDS)
        + "} AS record LIMIT 1",
        target=row["target_path"],
    )
    if len(edges) > 1:
        raise ValueError("Multiple graph bindings for the same source and target")
    return {
        "SignalSource": source[0]["record"] if source else None,
        "MAPS_TO_IMAS": edges[0]["record"] if edges else None,
        "IMASNode": target[0]["record"] if target else None,
    }


def load_handoff(path: Path | None) -> tuple[Path, dict[str, Any], str]:
    source = path or (LIVE_HANDOFF if LIVE_HANDOFF.exists() else FIXTURE)
    document = json.loads(source.read_text())
    if (
        document.get("format") != "imas-codex-mapping-handoff"
        or document.get("facility") != FACILITY
    ):
        raise ValueError(f"Unexpected hand-off format/facility: {source}")
    if document.get("format_version") != 1:
        raise ValueError("Unsupported hand-off format version")
    kind = "EXPLICIT INPUT"
    if source.resolve() == LIVE_HANDOFF.resolve():
        kind = "LIVE EXPORT"
    elif source.resolve() == FIXTURE.resolve():
        kind = "CONTRACT FIXTURE"
    return source, document, kind


def figure(height: float) -> tuple[Any, Any]:
    fig, ax = plt.subplots(figsize=(14, height), dpi=100)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.set(xlim=(0, 14), ylim=(0, height))
    ax.axis("off")
    return fig, ax


def save(fig: Any, stem: str, presentation: Path) -> dict[str, Any]:
    svg = FIGURES / f"{stem}.svg"
    png = presentation / f"{stem}.png"
    fig.savefig(svg, format="svg", metadata={"Date": None}, facecolor="white")
    svg.write_text(
        "\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n"
    )
    fig.savefig(png, dpi=200, facecolor="white")
    width, height = fig.get_size_inches()
    plt.close(fig)
    result = {
        "svg": str(svg.relative_to(ROOT)),
        "png": str(png),
        "svg_bytes": svg.stat().st_size,
        "png_bytes": png.stat().st_size,
        "png_width": round(width * 200),
        "png_height": round(height * 200),
    }
    print("ARTIFACT " + json.dumps(result), flush=True)
    return result


def arrow(ax: Any, start: tuple, end: tuple, *, context: bool = False) -> None:
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops={
            "arrowstyle": "->",
            "color": REFERENCE,
            "linewidth": 1.2,
            "linestyle": "--" if context else "-",
        },
    )


def pipeline(
    counts: dict[str, Any], doc: dict[str, Any], source_kind: str, presentation: Path
) -> dict[str, Any]:
    fig, ax = figure(10)

    def stage(x: float, y: float, name: str, measure: str) -> None:
        ax.text(x, y, name, ha="center", va="center", fontsize=22, color=INK)
        ax.text(x, y - 0.40, measure, ha="center", va="top", fontsize=20, color=INK)

    stage(2.3, 8.8, "Paths", f"{counts['FacilityPath']:,} paths")
    stage(7.1, 8.8, "Code", f"{counts['CodeFile']:,} files")
    stage(11.6, 8.8, "Documents", f"{counts['Document']:,} documents")
    arrow(ax, (3.5, 8.8), (5.5, 8.8))
    ax.plot([2.3, 2.3, 11.6], [9.2, 9.7, 9.7], color=REFERENCE, linewidth=1.2)
    arrow(ax, (11.6, 9.7), (11.6, 9.2))
    stage(2.3, 6.3, "Wiki", f"{counts['WikiPage']:,} pages")
    stage(2.3, 3.9, "Signals scan", f"{counts['FacilitySignal']:,} signals")
    stage(
        7.1,
        5.0,
        "Signals enrich + check",
        "\n".join(
            f"{r['status'] or 'unset'}: {r['count']:,}"
            for r in counts["signal_statuses"]
        ),
    )
    arrow(ax, (3.6, 3.9), (5.1, 4.7))
    arrow(ax, (3.4, 6.2), (5.1, 5.3), context=True)
    arrow(ax, (7.1, 7.8), (7.1, 5.5), context=True)
    stage(
        11.6,
        5.0,
        "Candidates",
        f"{counts['routed_sources']:,} / {counts['SignalSource']:,}\nsources judged",
    )
    arrow(ax, (9.1, 5.0), (10.0, 5.0))
    stage(
        11.6,
        2.5,
        "Mapping",
        f"{counts['IMASMapping']:,} mappings\n{counts['bindings']:,} bindings",
    )
    arrow(ax, (11.6, 3.7), (11.6, 3.0))
    expanded = sum(len(entry.get("signals", [])) for entry in doc["ids"])
    unexpanded = sum(len(entry.get("unexpanded", [])) for entry in doc["ids"])
    stage(
        6.3,
        2.5,
        "Hand-off export",
        f"{expanded:,} rows; {unexpanded:,} unexpanded\n{source_kind}",
    )
    arrow(ax, (10.0, 2.5), (8.6, 2.5))
    ax.text(
        0.45,
        0.6,
        "Solid arrows: required input   ·   Dashed arrows: enrichment context",
        fontsize=20,
        color=REFERENCE,
    )
    return save(fig, "pipeline", presentation)


def lifecycle(presentation: Path) -> dict[str, Any]:
    fig, ax = figure(5.3)
    for x, label, detail in [
        (2.1, "generated", "Draft field bindings"),
        (7, "validated", "Validation passed"),
        (11.7, "active", "Explicit promotion"),
    ]:
        ax.text(x, 3.4, label, ha="center", fontsize=24, color=INK)
        ax.text(x, 2.95, detail, ha="center", fontsize=20, color=INK)
    arrow(ax, (3.5, 3.55), (5.5, 3.55))
    arrow(ax, (8.5, 3.55), (10.3, 3.55))
    ax.text(4.5, 4.2, "map validate", ha="center", fontsize=20, color=INK)
    ax.text(9.4, 4.2, "map activate", ha="center", fontsize=20, color=INK)
    ax.text(2.1, 4.5, "map run", ha="center", fontsize=20, color=INK)
    ax.plot(
        [2.1, 2.1, 7.0], [2.5, 1.5, 1.5], color=REFERENCE, linewidth=1.2, linestyle=":"
    )
    ax.plot([7.0, 7.0], [1.25, 1.75], color=INK, linewidth=2.6)
    ax.text(
        7.4,
        1.5,
        "activate without validation: REFUSE",
        va="center",
        fontsize=20,
        color=INK,
    )
    ax.text(
        0.45,
        0.45,
        "Required lifecycle policy; this diagram is not an activation-test result.",
        fontsize=20,
        color=REFERENCE,
    )
    return save(fig, "mapping-lifecycle", presentation)


def styled_lines(value: Any, columns: int = 52) -> list[list[tuple[str, str]]]:
    style = get_style_by_name(PYGMENTS_STYLE)
    lines: list[list[tuple[str, str]]] = [[]]
    used = 0
    for token, text in lex(dumps(value), JsonLexer()):
        colour = style.style_for_token(token)["color"] or "242424"
        for char in text:
            if char == "\n":
                lines.append([])
                used = 0
            else:
                if used == columns:
                    lines.append([])
                    used = 0
                if lines[-1] and lines[-1][-1][1] == colour:
                    previous, _ = lines[-1][-1]
                    lines[-1][-1] = (previous + char, colour)
                else:
                    lines[-1].append((char, colour))
                used += 1
    return lines


def json_figure(
    example: dict[str, Any], source: Path, source_kind: str, presentation: Path
) -> dict[str, Any]:
    left = styled_lines(example["row"])
    right = styled_lines(example["graph"])
    height = max(7.0, 2.8 + max(len(left), len(right)) * 0.235)
    fig, ax = figure(height)
    ax.text(0.35, height - 0.45, source_kind, fontsize=20, color=INK)
    for index, part in enumerate(textwrap.wrap(str(source), width=108)):
        ax.text(0.35, height - 0.82 - index * 0.27, part, fontsize=12, color=REFERENCE)
    for x, label in [(0.35, "Hand-off signal row"), (7.3, "Live graph projection")]:
        ax.text(x, height - 1.65, label, fontsize=20, color=INK)
    font = FontProperties(family="DejaVu Sans Mono", size=14)
    for x, lines in [(0.35, left), (7.3, right)]:
        for index, segments in enumerate(lines):
            offset = 0
            for text, colour in segments:
                ax.text(
                    x + offset * (14 / 72 * 0.602),
                    height - 2.12 - index * 0.235,
                    text,
                    fontproperties=font,
                    va="baseline",
                    color="#" + colour,
                )
                offset += len(text)
    ax.text(0.35, 0.65, example["note"], fontsize=15, color=INK, wrap=True)
    ax.text(
        0.35,
        0.28,
        f"JSON style: {PYGMENTS_STYLE}; long strings wrap visually; null means no returned record or missing property.",
        fontsize=12,
        color=REFERENCE,
    )
    return save(fig, example["stem"], presentation)


def select_examples(reader: Reader, document: dict[str, Any]) -> list[dict[str, Any]]:
    wanted = [
        ("pickup-probe", "Pickup probe", "magnetics/b_field_pol_probe/field/data"),
        ("pf-coil-current", "PF coil current", "pf_active/coil/current/data"),
    ]
    examples = []
    for stem, title, target in wanted:
        candidates = [
            (entry, row)
            for entry in document["ids"]
            for row in entry.get("signals", [])
            if row.get("target_path") == target
        ]
        if not candidates:
            raise ValueError(f"Hand-off document has no {title} row")
        entry, row = candidates[0]
        graph = read_record(reader, row, stem)
        missing = [name for name, value in graph.items() if value is None]
        note = (
            "No matching live " + ", ".join(missing) + "."
            if missing
            else "Source, binding and DD target found in the live graph."
        )
        examples.append(
            {
                "stem": stem,
                "title": title,
                "row": row,
                "graph": graph,
                "mapping_id": entry["mapping_id"],
                "status": entry["status"],
                "note": note,
            }
        )
    return examples


def e(value: Any) -> str:
    return html.escape(str(value))


def report_html(receipt: dict[str, Any]) -> str:
    counts = receipt["counts"]
    source = receipt["source"]
    figures = {Path(item["svg"]).stem: item for item in receipt["artifacts"]}

    def embed(stem: str, caption: str) -> str:
        return f'<figure><img src="/imas-codex/figures/discovery-to-mapping/{stem}.svg" alt="{e(caption)}" style="width:100%;height:auto"><figcaption>{e(caption)}</figcaption></figure>'

    parts = [
        """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="docs-project" content="imas-codex">
<meta name="reckon-type" content="research">
<meta name="plan-slug" content="discovery-to-mapping-pipeline">
<meta name="plan-title" content="Discovery to mapping: the JT-60SA hand-off">
<meta name="plan-summary" content="Live discovery counts, draft mapping provenance and the hand-off boundary between imas-codex and imas-ambix.">
<meta name="plan-status" content="reference">
<meta name="plan-tags" content="jt-60sa">
<meta name="plan-evidence-for" content="generated-map-handoff">
<meta name="plan-informs" content="imas-ambix:jt60sa-tokamap-from-generated-maps">
<meta name="plan-source-quality" content="primary">
<title>Discovery to mapping | imas-codex</title>
<link rel="stylesheet" href="/_shared/foundation.css">
<link rel="stylesheet" href="/_shared/dashboard.css">
</head><body><main class="plan-doc">
<header><p class="eyebrow">JT-60SA · imas-codex chapter</p>
<h1>Discovery to mapping</h1>
<p class="lead">Discovery turns facility files, documentation and signal inventories into graph-backed mapping proposals. A generated mapping is a draft: imas-ambix still has to resolve machine indices, validate the rules against data and write the tokamap directory and IDS.</p></header>"""
    ]
    parts.append(
        f'<section id="snapshot"><h2>What this snapshot establishes</h2><p>Graph reads ran from <strong>{e(receipt["observed_started_at"])}</strong> to <strong>{e(receipt["observed_finished_at"])}</strong>. They found <strong>{counts["FacilitySignal"]:,} signals</strong>, <strong>{counts["SignalSource"]:,} sources</strong>, <strong>{counts["routed_sources"]:,} sources with a candidate route</strong>, <strong>{counts["IMASMapping"]:,} mapping records</strong> and <strong>{counts["bindings"]:,} field bindings</strong> for JT-60SA. These are sequential live reads, not an atomic snapshot; concurrent discovery can change the totals.</p>'
    )
    parts.append(
        f"<p><strong>Example provenance: {e(source['kind'])}.</strong> The JSON renders use <code>{e(source['path'])}</code>, SHA-256 <code>{e(source['sha256'])}</code>. Its export timestamp is <code>{e(source['exported_at'])}</code>; it is separate from the graph observation time. Fixture values are illustrative contract data, including confidence and units, and are not measurements of live mapping quality.</p>"
    )
    parts.append(
        "<p>No third system has a generated mapping in the hand-off available for this report. The examples cover pickup probes and PF coil currents only. Missing graph records are labelled explicitly; no record, transform, convention or machine index is invented.</p></section>"
    )
    parts.append(
        '<section id="discovery"><h2>From facility context to candidate targets</h2><p>The discovery sequence starts with paths. Code discovery reads those paths to recover source files and data-access logic; document discovery reads the same path inventory. Wiki discovery is an independent source of descriptions and operating context. Signal scanning inventories the facility data systems and creates signal identities. Enrichment adds physical meaning using code and wiki context, and access checks establish whether the signal can be read.</p><p>The diagram separates required stage inputs from context. Counts are inventories at the observation time, not an attrition funnel: files, sources, signals and bindings are different entities. Signal status counts are current states, not cumulative counts of completed transitions. In particular, checked means access was attempted; CHECKED_WITH.success holds the outcome, so checked is not a success count. Documents are part of the evidence inventory; the sequence explicitly lists code and wiki as enrichment context.</p>'
    )
    parts.append(
        embed(
            "pipeline",
            "Discovery dependencies and measured JT-60SA inventories. Export counts belong to the labelled input document; all upstream counts come from the live graph.",
        )
    )
    parts.append(
        "<p>A SignalSource groups signals that should map to the same DD field. The candidates stage retrieves and judges Data Dictionary targets, recording a candidate route and ranked MAPPING_CANDIDATE edges. A candidate is a proposal. It becomes a binding only when a MAPS_TO_IMAS edge carries the chosen target and transformation. A non-null candidate route includes selected, escalated and no-candidate outcomes; it does not mean that every judged source is ready to map.</p><table><thead><tr><th>Candidate route</th><th>Sources</th></tr></thead><tbody>"
    )
    parts.extend(
        f"<tr><td>{e(r['route'] or 'not yet judged')}</td><td>{r['count']:,}</td></tr>"
        for r in counts["candidate_routes"]
    )
    parts.append(
        '</tbody></table></section><section id="lifecycle"><h2>A draft needs validation before activation</h2><p>IMASMapping records orchestrate an IDS and link to SignalSource groups; each source-to-DD edge holds a field binding. The required lifecycle is generated → validated → active. Generation records a draft. Validation checks the mapping against signals, transforms and shapes; activation is an explicit promotion after that validation succeeds.</p>'
    )
    parts.append(
        embed(
            "mapping-lifecycle",
            "Required lifecycle and refusal of activation without validation. Commands use the intended top-level map group. This is the specified policy, not evidence that the activation guard passed a test.",
        )
    )
    parts.append(
        "<p>This report reads the graph and does not execute generation, validation or activation. It therefore makes no runtime claim about the activation guard. The command migration and guard implementation have their own tests; the figure records the hand-off contract they must enforce. Exporting a generated mapping is allowed and leaves its status visible in the document.</p><table><thead><tr><th>Live IDS</th><th>Mapping status</th><th>Records</th></tr></thead><tbody>"
    )
    if counts["mapping_statuses"]:
        parts.extend(
            f"<tr><td>{e(r['ids'])}</td><td>{e(r['status'])}</td><td>{r['count']}</td></tr>"
            for r in counts["mapping_statuses"]
        )
    else:
        parts.append(
            '<tr><td colspan="3">No JT-60SA IMASMapping record returned by the indexed live query.</td></tr>'
        )
    parts.append(
        '</tbody></table></section><section id="handoff"><h2>What crosses the repository boundary</h2><p>The hand-off has one document per facility. Its header identifies the format and version, facility, DD version and UTC export time. Each IDS entry records mapping identity, lifecycle status, expanded signal rows and unexpanded bindings. Each signal row carries the facility signal and source identities, data system, source group and array, member identifier, index-free DD path, transformation, units, COCOS label, confidence and evidence. Missing graph values are null.</p><p>The document assigns no array index. imas-ambix owns the machine description built from the SELENE deck and resolves a signal to a concrete probe or coil. It checks the rules against actual data and writes the machine and signal mappings into the tokamap directory and then the IDS. A member identifier is a source identity, not permission to use that number as an IMAS array index.</p><table><thead><tr><th>Input IDS</th><th>Status</th><th>Expanded rows</th><th>Unexpanded</th></tr></thead><tbody>'
    )
    for entry in receipt["handoff"]["ids"]:
        parts.append(
            f"<tr><td>{e(entry['ids_name'])}</td><td>{e(entry['status'])}</td><td>{len(entry.get('signals', []))}</td><td>{len(entry.get('unexpanded', []))}</td></tr>"
        )
    parts.append("</tbody></table>")
    for entry in receipt["handoff"]["ids"]:
        for row in entry.get("unexpanded", []):
            parts.append(
                f"<p>Unexpanded <code>{e(row['source_id'])}</code> → <code>{e(row['target_path'])}</code>: {e(row['reason'])}</p>"
            )
    parts.append(
        '</section><section id="examples"><h2>Two signal examples and their provenance</h2>'
    )
    for example in receipt["examples"]:
        parts.append(
            f"<h3>{e(example['title'])}</h3><p>Input mapping <code>{e(example['mapping_id'])}</code>; status <code>{e(example['status'])}</code>. {e(example['note'])}</p>"
        )
        parts.append(
            embed(
                example["stem"],
                f"{example['title']}: {source['kind']} signal row beside a bounded live projection of its source, binding and DD target. {example['note']}",
            )
        )
        parts.append(
            f"<details open><summary>Selectable hand-off row</summary><pre>{e(dumps(example['row']))}</pre></details><details open><summary>Selectable live graph projection</summary><pre>{e(dumps(example['graph']))}</pre></details>"
        )
    parts.append(
        f"<p>JSON syntax uses Pygments <code>{PYGMENTS_STYLE}</code> on white. The style is not yet matched to imas-ambix. Node projections show selected schema-declared fields; the edge projection shows the persisted transform, units, convention, confidence and evidence fields. Null records mean the exact source or edge was not found, not that another source with a similar name is equivalent. The same excerpt appears as selectable text above.</p></section>"
    )
    parts.append(
        '<section id="method"><h2>Reproduce and inspect the evidence</h2><p>Run the single script below. It prefers the live hand-off when present and otherwise uses the contract fixture in the main checkout. It regenerates all four SVGs, all four slide PNGs, the evidence receipt and this report. It never accesses the facility over SSH or writes the graph.</p><pre>UV_NO_SYNC=1 UV_PROJECT_ENVIRONMENT=/home/ITER/mcintos/Code/imas-codex/.venv PYTHONPATH="$PWD" uv run --no-sync python scripts/figures_discovery_to_mapping.py</pre>'
    )
    parts.append(
        f'<p>Repository base revision <code>{e(receipt["revision"])}</code>; script SHA-256 <code>{e(receipt["renderer_sha256"])}</code>. Schema property checks are recorded with schema hashes in <a href="/imas-codex/figures/discovery-to-mapping/evidence.json">the machine-readable evidence receipt</a>. Each query uses a nine-second server timeout and is rejected if measured wall time reaches ten seconds. Online range indexes are checked first; facility and identity hints enforce indexed starts. The known JT-60SA Facility and nonzero signals/sources act as positive controls. A zero inventory triggers an identity-index witness lookup for that label, and the schema check prevents a misspelled facility or status property from masquerading as an empty result.</p>'
    )
    parts.append(
        "<p>Every query, parameters, result and elapsed time is printed by the script and preserved below. A zero with no identity witness means that label has no indexed identity to inspect; it is explicitly weaker evidence than a populated positive control.</p>"
    )
    for query in receipt["queries"]:
        parts.append(
            f"<details><summary>{e(query['name'])} · {query['seconds']:.4f} s</summary><pre>{e(textwrap.fill(query['query'], width=100, break_long_words=False, break_on_hyphens=False))}\nPARAMS {e(dumps(query['parameters']))}\nRESULT {e(dumps(query['rows']))}</pre></details>"
        )
    parts.append(
        "<h3>Slide-ready PNG set</h3><p>SVGs use a 14-inch canvas at 100 dpi. PNGs have twice the pixel dimensions and live outside the repository.</p><table><thead><tr><th>Figure</th><th>PNG path</th><th>Pixels</th><th>Bytes</th></tr></thead><tbody>"
    )
    for stem, artifact in figures.items():
        parts.append(
            f"<tr><td>{e(stem)}</td><td><code>{e(artifact['png'])}</code></td><td>{artifact['png_width']} × {artifact['png_height']}</td><td>{artifact['png_bytes']:,}</td></tr>"
        )
    parts.append(
        '</tbody></table><p>Related records: <a href="/imas-codex/generated-map-handoff.html">generated mapping hand-off contract</a>; <a href="/imas-ambix/jt60sa-tokamap-from-generated-maps.html">imas-ambix tokamap work</a>.</p></section></main></body></html>'
    )
    return "\n".join(parts) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--handoff",
        type=Path,
        help="Explicit source document; default prefers the live export",
    )
    parser.add_argument("--presentation-dir", type=Path, default=PRESENTATION)
    args = parser.parse_args()
    plt.style.use("data-ink")
    plt.rcParams.update(
        {
            "svg.fonttype": "none",
            "svg.hashsalt": "discovery-to-mapping",
            "font.family": "DejaVu Sans",
        }
    )
    FIGURES.mkdir(parents=True, exist_ok=True)
    args.presentation_dir.mkdir(parents=True, exist_ok=True)
    source, document, source_kind = load_handoff(args.handoff)
    receipt = {
        "revision": subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
        ).strip(),
        "observed_started_at": datetime.now(UTC).isoformat(),
        "schema_sha256": check_schema(),
        "renderer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source": {
            "path": str(source),
            "kind": source_kind,
            "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "exported_at": document.get("exported_at"),
        },
        "handoff": document,
    }
    print("SOURCE " + json.dumps(receipt["source"]), flush=True)
    with GraphClient() as client:
        reader = Reader(client)
        receipt["counts"] = read_counts(reader)
        receipt["examples"] = select_examples(reader, document)
        receipt["queries"] = reader.receipts
    receipt["observed_finished_at"] = datetime.now(UTC).isoformat()
    receipt["artifacts"] = [
        pipeline(receipt["counts"], document, source_kind, args.presentation_dir),
        lifecycle(args.presentation_dir),
    ]
    receipt["artifacts"].extend(
        json_figure(example, source, source_kind, args.presentation_dir)
        for example in receipt["examples"]
    )
    (FIGURES / "evidence.json").write_text(dumps(receipt) + "\n")
    REPORT.write_text(report_html(receipt))
    print(
        f"SUMMARY figures={len(receipt['artifacts'])} queries={len(receipt['queries'])} report={REPORT} source={source}",
        flush=True,
    )


if __name__ == "__main__":
    main()
