"""Read-only census of live names without a producing source."""

from __future__ import annotations

import json
import sys
import time
from collections import Counter, defaultdict, deque
from html import escape
from pathlib import Path

from imas_codex.graph.client import GraphClient

ROOT = Path(__file__).resolve().parents[4]
OUTPUT = Path(__file__).with_name("sib-source-loss-census-results.json")
REPORT = Path(__file__).with_name("sib-source-loss-census.html")
SHEET = (
    ROOT
    / "docs/evidence/fragments/sn-review-through-jev/srtj-refine-ab-run-results.json"
)
TERMINAL = {"deprecated", "superseded"}


def classify(item: dict) -> str:
    if item.get("origin") == "derived" and item.get("children"):
        return "derived_parent"
    if item.get("stage") == "exhausted" and item.get("source_holding_successors"):
        return "orphan_awaiting_retirement"
    if item.get("history_sources"):
        return "dropped_edge"
    return "never_bound"


def fixture_control() -> None:
    fixtures = {
        "derived_parent": {"origin": "derived", "children": ["child"]},
        "orphan_awaiting_retirement": {
            "stage": "exhausted",
            "source_holding_successors": ["successor"],
        },
        "dropped_edge": {"history_sources": ["dd:path"]},
        "never_bound": {},
    }
    for expected, item in fixtures.items():
        got = classify(item)
        print(f"fixture {expected}: {got}")
        assert got == expected
    print("EXIT=0")


def query(gc: GraphClient, label: str, cypher: str, **params: object) -> list[dict]:
    start = time.monotonic()
    rows = gc.query(cypher, **params)
    elapsed = time.monotonic() - start
    print(f"query {label}: {len(rows)} rows, {elapsed:.3f}s", file=sys.stderr)
    if elapsed >= 10:
        raise RuntimeError(f"{label} exceeded ten seconds")
    return rows


def main() -> None:
    fixture_control()
    with GraphClient() as gc:
        counts = query(
            gc,
            "properties",
            """
            MATCH (n:StandardName)
            RETURN count(n) AS total, count(n.id) AS id,
                   count(n.status) AS status, count(n.name_stage) AS stage,
                   count(n.origin) AS origin, count(n.source_paths) AS paths,
                   count(n.source_path) AS path,
                   count(n.source_dd_resolution_converged_ids) AS resolutions
        """,
        )[0]
        assert counts["total"] == counts["id"] == counts["status"] == counts["stage"]
        names = query(
            gc,
            "names",
            """
            MATCH (n:StandardName)
            RETURN n.id AS id, n.name_stage AS stage, n.status AS status,
                   n.origin AS origin, n.source_paths AS paths,
                   n.source_path AS source_path,
                   n.source_dd_resolution_converged_ids AS resolutions,
                   n.source_types AS source_types, n.edit_origin AS edit_origin,
                   n.refine_reason AS refine_reason, n.edit_reason AS edit_reason,
                   n.generated_at IS NOT NULL AS generated,
                   n.imported_at IS NOT NULL AS imported
            ORDER BY id LIMIT 6000
        """,
        )
        assert len(names) == counts["total"]
        edges = query(
            gc,
            "source_edges",
            """
            MATCH (s:StandardNameSource)-[:PRODUCED_NAME]->(n:StandardName)
            RETURN s.id AS source, s.source_id AS source_id,
                   s.produced_sn_id AS mirror, s.source_type AS type,
                   n.id AS name ORDER BY source LIMIT 12000
        """,
        )
        relations = query(
            gc,
            "lineage",
            """
            MATCH (a:StandardName)-[r:REFINED_FROM|HAS_SUCCESSOR]->(b:StandardName)
            RETURN a.id AS a, type(r) AS type, b.id AS b LIMIT 12000
        """,
        )
        children = query(
            gc,
            "children",
            """
            MATCH (c:StandardName)-[:HAS_PARENT]->(p:StandardName)
            RETURN c.id AS child, p.id AS parent LIMIT 12000
        """,
        )
        source_nodes = query(
            gc,
            "source_nodes",
            """
            MATCH (s:StandardNameSource)
            RETURN s.id AS source, s.source_id AS source_id,
                   s.produced_sn_id AS mirror, s.source_type AS type
            ORDER BY source LIMIT 12000
        """,
        )
        source_properties = query(
            gc,
            "source_properties",
            """
            MATCH (s:StandardNameSource)
            RETURN count(s) AS total, count(s.id) AS id,
                   count(s.produced_sn_id) AS mirror,
                   count(s.source_id) AS source_id
        """,
        )[0]
        assert len(source_nodes) == source_properties["total"]
        assert source_properties["id"] == source_properties["total"]
        projections = query(
            gc,
            "dd_projections",
            """
            MATCH (dd:IMASNode)-[:HAS_STANDARD_NAME]->(n:StandardName)
            RETURN dd.id AS path, n.id AS name LIMIT 12000
        """,
        )
        change_count = query(
            gc,
            "change_count",
            """
            MATCH (c:StandardNameChange) RETURN count(c) AS total
        """,
        )[0]["total"]
        changes = query(
            gc,
            "changes",
            """
            MATCH (c:StandardNameChange)
            RETURN c.from_name AS predecessor, c.to_name AS successor,
                   c.operation AS operation, c.origin AS origin
            LIMIT 20000
        """,
        )
        assert len(changes) == change_count

    by_id = {n["id"]: n for n in names}
    holdings = defaultdict(list)
    by_source = defaultdict(list)
    source_by_id = {s["source"]: s for s in source_nodes}
    for edge in edges:
        holdings[edge["name"]].append(edge["source"])
        by_source[edge["source"]].append(edge["name"])
    assert edges and source_nodes and relations and children
    successors = defaultdict(set)
    predecessors = defaultdict(set)
    for rel in relations:
        old, new = (
            (rel["b"], rel["a"])
            if rel["type"] == "REFINED_FROM"
            else (rel["a"], rel["b"])
        )
        successors[old].add(new)
        predecessors[new].add(old)
    operations = defaultdict(list)
    for change in changes:
        old, new = change["predecessor"], change["successor"]
        if old in by_id and new in by_id and old != new:
            successors[old].add(new)
            predecessors[new].add(old)
        for name in (old, new):
            if name in by_id:
                operations[name].append(change["operation"])
    child_map = defaultdict(list)
    for edge in children:
        child_map[edge["parent"]].append(edge["child"])
    projections_by_name = defaultdict(list)
    for edge in projections:
        projections_by_name[edge["name"]].append("dd:" + edge["path"])
    detachment_paths = defaultdict(list)
    source_receipts = defaultdict(list)
    for change in changes:
        name = change["successor"]
        if name not in by_id:
            continue
        if change["operation"] in {
            "detach_inconsistent_attachment",
            "detach_stale_source_binding",
        }:
            path = change["predecessor"]
            if isinstance(path, str) and "/" in path:
                detachment_paths[name].append("dd:" + path.removeprefix("dd:"))
        if change["operation"] in {
            "source_migration_manifest",
            "reattach_lapsed_detachment",
            "repair_semantic_source_binding",
            "recover_terminal_source_binding",
        }:
            source_receipts[name].append(change["operation"])

    def source_locations(paths: set[str]) -> dict[str, dict]:
        found = {}
        for path in sorted(paths):
            source_id = path if path.startswith(("dd:", "derived:")) else "dd:" + path
            source = source_by_id.get(source_id)
            if source is None:
                continue
            found[source_id] = {
                "mirror": source["mirror"],
                "holders": sorted(by_source[source_id]),
                "live_holders": sorted(
                    h
                    for h in by_source[source_id]
                    if h in by_id and by_id[h]["status"] not in TERMINAL
                ),
            }
        return found

    def writer_for(item: dict) -> tuple[str, str]:
        operations = set(item["operations"])
        if {
            "detach_inconsistent_attachment",
            "detach_stale_source_binding",
        } <= operations:
            return (
                "unknown: multiple detachment writers",
                "attachment_audit.py:240 and signed_manifest.py:4904 both record "
                "detachment; the unordered change rows do not establish which "
                "writer removed the final producing edge",
            )
        if "detach_inconsistent_attachment" in operations:
            return (
                "attachment_audit.py:240",
                "recorded detachment removed PRODUCED_NAME",
            )
        if "detach_stale_source_binding" in operations:
            return "signed_manifest.py:4904", "recorded stale-source detachment"
        if item["source_receipts"]:
            return (
                "unknown after prior binding",
                "binding receipt proves an edge existed; no later deletion receipt identifies its remover",
            )
        if (
            item["recorded_paths"]
            or item["recorded_resolution_ids"]
            or item["lineage_paths"]
            or item["historical_edges"]
        ):
            return (
                "unknown after path or lineage evidence",
                "DD projection, path or relative's edge survives; no deletion receipt identifies its remover",
            )
        if "human_edit" in operations:
            return (
                "edit.py:3393",
                "rename allowed a zero-source predecessor; no producing source was available to carry",
            )
        if "refine" in operations:
            return (
                "unknown after refine",
                "refine event exists, but no binding receipt or retained DD path proves an edge was written",
            )
        return (
            "unknown initial writer",
            "searched paths, source mirrors, DD projections, lineage and changes; none records a binding",
        )

    def descend(name: str) -> set[str]:
        seen, queue = set(), deque(successors[name])
        while queue:
            nxt = queue.popleft()
            if nxt in seen:
                continue
            seen.add(nxt)
            queue.extend(successors[nxt] - seen)
        return seen

    def ascend(name: str) -> set[str]:
        seen, queue = set(), deque(predecessors[name])
        while queue:
            previous = queue.popleft()
            if previous in seen:
                continue
            seen.add(previous)
            queue.extend(predecessors[previous] - seen)
        return seen

    rows = []
    for n in names:
        name = n["id"]
        if n["status"] in TERMINAL or holdings[name]:
            continue
        descendants = descend(name)
        live_holders = sorted(
            d
            for d in descendants
            if d in by_id and by_id[d]["status"] not in TERMINAL and holdings[d]
        )
        ancestors = ascend(name)
        resolution_ids = set(n["resolutions"] or [])
        source_hints = set(n["paths"] or [])
        if n["source_path"]:
            source_hints.add(n["source_path"])
        source_hints.update(projections_by_name[name])
        source_hints.update(detachment_paths[name])
        source_hints.update(s["source"] for s in source_nodes if s["mirror"] == name)
        linked = ancestors | descendants
        historical_edges = sorted(
            {source for relative in linked for source in holdings[relative]}
        )
        lineage_paths = {
            path
            for relative in linked
            if relative in by_id
            for path in (
                list(by_id[relative]["paths"] or [])
                + (
                    [by_id[relative]["source_path"]]
                    if by_id[relative]["source_path"]
                    else []
                )
                + projections_by_name[relative]
                + detachment_paths[relative]
            )
        }
        lineage_receipts = {
            receipt for relative in linked for receipt in source_receipts[relative]
        }
        history_sources = sorted(
            source_hints
            | resolution_ids
            | lineage_paths
            | set(historical_edges)
            | set(source_receipts[name])
            | lineage_receipts
        )
        item = {
            "name": name,
            "stage": n["stage"],
            "status": n["status"],
            "origin": n["origin"],
            "children": sorted(child_map[name]),
            "successors": sorted(descendants),
            "source_holding_successors": live_holders,
            "history_sources": history_sources,
            "recorded_paths": sorted(source_hints),
            "recorded_resolution_ids": sorted(resolution_ids),
            "lineage_paths": sorted(lineage_paths),
            "historical_edges": historical_edges,
            "source_receipts": sorted(set(source_receipts[name]) | lineage_receipts),
            "source_locations": source_locations(
                source_hints | lineage_paths | set(historical_edges)
            ),
            "operations": sorted(set(operations[name])),
            "imported": n["imported"],
            "generated": n["generated"],
            "edit_origin": n["edit_origin"],
        }
        item["population"] = classify(item)
        if item["population"] in {"dropped_edge", "never_bound"}:
            item["responsible_writer"], item["writer_evidence"] = writer_for(item)
        item["repair_source"] = (
            "successor_edge"
            if live_holders
            else "recorded_path"
            if source_hints or lineage_paths
            else "recorded_resolution_id"
            if resolution_ids
            else "ancestor_or_successor_edge"
            if historical_edges
            else "binding_receipt_without_path"
            if item["source_receipts"]
            else "none"
        )
        rows.append(item)

    sheet = json.loads(SHEET.read_text())["sheet_cases_without_dd_source"]
    assert len(sheet) == 62
    row_map = {r["name"]: r for r in rows}
    assert len(row_map) == len(rows)
    assert len(set(sheet)) == len(sheet)
    sheet_rows = []
    for name in sheet:
        n = by_id.get(name)
        row = row_map.get(name)
        successors_for_case = sorted(descend(name)) if n else []
        live_successors = sorted(
            d
            for d in successors_for_case
            if d in by_id and by_id[d]["status"] not in TERMINAL and holdings[d]
        )
        case_paths = set(n["paths"] or []) if n else set()
        case_paths.update(projections_by_name[name])
        case_paths.update(detachment_paths[name])
        if n and n["source_path"]:
            case_paths.add(n["source_path"])
        for relative in ascend(name) | set(successors_for_case):
            related = by_id.get(relative)
            if related:
                case_paths.update(related["paths"] or [])
                if related["source_path"]:
                    case_paths.add(related["source_path"])
            case_paths.update(projections_by_name[relative])
            case_paths.update(detachment_paths[relative])
        case_sources = source_locations(case_paths)
        sheet_rows.append(
            {
                "name": name,
                "population": row["population"]
                if row
                else "outside_live_sourceless_cohort",
                "outside_reason": None
                if row
                else (
                    "terminal_predecessor"
                    if n and n["status"] in TERMINAL
                    else "directly_sourced"
                    if holdings[name]
                    else "missing_name"
                ),
                "current_stage": n["stage"] if n else None,
                "current_status": n["status"] if n else None,
                "direct_sources": sorted(holdings[name]),
                "source_holding_successors": live_successors,
                "successor_sources": {d: sorted(holdings[d]) for d in live_successors},
                "successors": successors_for_case,
                "recorded_paths": sorted(case_paths),
                "source_locations": case_sources,
                "repair_source": row["repair_source"]
                if row
                else (
                    "live_successor_edge"
                    if live_successors
                    else "direct_edge"
                    if holdings[name]
                    else "recorded_path"
                    if case_paths
                    else "none_recorded"
                ),
            }
        )
    output = {
        "property_counts": counts,
        "source_property_counts": source_properties,
        "total_live": sum(n["status"] not in TERMINAL for n in names),
        "population_counts": dict(Counter(r["population"] for r in rows)),
        "stage_counts": {
            stage: dict(Counter(r["population"] for r in rows if r["stage"] == stage))
            for stage in sorted({r["stage"] for r in rows})
        },
        "exhausted_dropped_edges": sum(
            r["stage"] == "exhausted" and r["population"] == "dropped_edge"
            for r in rows
        ),
        "names": rows,
        "refine_cases": sheet_rows,
    }
    assert len(rows) == 264
    assert sum(output["population_counts"].values()) == len(rows)
    assert all(
        output["population_counts"].get(population, 0) > 0
        for population in (
            "derived_parent",
            "orphan_awaiting_retirement",
            "dropped_edge",
            "never_bound",
        )
    )
    for name, population in {
        "beta": "derived_parent",
        "normalized_toroidal_thermal_plasma_beta": "orphan_awaiting_retirement",
        "line_averaged_neon_density": "dropped_edge",
        "electron_energy_flux_limiter_coefficient": "never_bound",
    }.items():
        assert row_map[name]["population"] == population
        print(f"witness {population}: {name}")
    write_report(output)
    for row in rows:
        row["unrelated_live_holders"] = sorted(
            {
                holder
                for source in row["source_locations"].values()
                for holder in source["live_holders"]
            }
            - set(row["source_holding_successors"])
        )
        row.pop("source_locations")
        row.pop("history_sources")
    serialized = json.dumps(output, separators=(",", ":"), sort_keys=True) + "\n"
    assert len(serialized.encode()) <= 300_000
    OUTPUT.write_text(serialized)
    print(
        json.dumps(
            {k: v for k, v in output.items() if k not in ("names", "refine_cases")},
            indent=2,
        )
    )
    print("EXIT=0")


def write_report(output: dict) -> None:
    names = output["names"]
    cases = output["refine_cases"]
    counts = output["population_counts"]
    by_writer = Counter(
        r.get("responsible_writer", "")
        for r in names
        if r["population"] in {"dropped_edge", "never_bound"}
    )
    cases_by_location = Counter(r["repair_source"] for r in cases)
    unrelated_holders = sum(
        bool(
            {
                holder
                for source in r["source_locations"].values()
                for holder in source["live_holders"]
            }
            - set(r["source_holding_successors"])
        )
        for r in names
        if r["population"] == "dropped_edge"
    )
    stage_rows = "\n".join(
        f"<tr><th>{escape(stage)}</th>"
        + "".join(
            f"<td>{by_pop.get(population, 0)}</td>"
            for population in (
                "derived_parent",
                "orphan_awaiting_retirement",
                "dropped_edge",
                "never_bound",
            )
        )
        + "</tr>"
        for stage, by_pop in output["stage_counts"].items()
    )
    writer_rows = "\n".join(
        f"<tr><td>{escape(writer)}</td><td>{count}</td></tr>"
        for writer, count in sorted(
            by_writer.items(), key=lambda item: (-item[1], item[0])
        )
    )
    name_rows = "\n".join(
        "<tr>"
        + "".join(
            f"<td>{escape(str(value))}</td>"
            for value in (
                r["name"],
                r["stage"],
                r["population"],
                ", ".join(r["source_holding_successors"]) or "—",
                r.get("responsible_writer", "—"),
                r["repair_source"],
                ", ".join(r["recorded_paths"][:3]) or "—",
            )
        )
        + "</tr>"
        for r in names
    )
    case_rows = "\n".join(
        "<tr>"
        + "".join(
            f"<td>{escape(str(value))}</td>"
            for value in (
                r["name"],
                r["population"],
                r["current_status"],
                ", ".join(r["source_holding_successors"]) or "—",
                r["repair_source"],
                ", ".join(r["recorded_paths"][:3]) or "—",
            )
        )
        + "</tr>"
        for r in cases
    )
    report = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Source loss census</title>
<style>body{{max-width:1000px;margin:2rem auto;font:16px/1.5 sans-serif;padding:0 1rem}}table{{border-collapse:collapse;width:100%;font-size:.85rem}}th,td{{border-bottom:1px solid #ccc;padding:.35rem;text-align:left;vertical-align:top;overflow-wrap:anywhere}}code{{font-size:.9em}}.scroll{{overflow-x:auto}}</style></head><body>
<article id="sib-source-loss-census-record"><h1>Where the DD sources went</h1>
<p>Read-only graph census at revision <code>993ddf413d290b09d1c8ea7518e2fdc4a0209b2f</code>. The control query found {output["property_counts"]["total"]} StandardName nodes, with {output["property_counts"]["status"]} statuses and {output["property_counts"]["stage"]} stages. There are {output["total_live"]} live names; {len(names)} have no incoming <code>PRODUCED_NAME</code> from a <code>StandardNameSource</code>. The source label has {output["source_property_counts"]["total"]} nodes and {output["source_property_counts"]["mirror"]} populated produced-name mirrors. Every result query returned fewer than its caller limit and took under ten seconds; the run log records each duration and ends <code>EXIT=0</code>.</p>
<h2 id="sib-source-loss-census-populations">Populations and controls</h2>
<p>Exactly one population is assigned per name: {counts["derived_parent"]} derived parents, {counts["orphan_awaiting_retirement"]} orphan awaiting retirement, {counts["dropped_edge"]} dropped edges, and {counts["never_bound"]} never bound. The exhausted dropped-edge count is <strong>{output["exhausted_dropped_edges"]}</strong>; the current source-only orphan selector would retire those names without checking that a successor holds their quantity.</p>
<div class="scroll"><table><thead><tr><th>Stage</th><th>Derived parent</th><th>Orphan</th><th>Dropped edge</th><th>Never bound</th></tr></thead><tbody>{stage_rows}</tbody></table></div>
<p>Observed witnesses: <code>beta</code> (derived parent), <code>normalized_toroidal_thermal_plasma_beta</code> (orphan; live source-holding successor <code>normalized_toroidal_beta</code>), <code>line_averaged_neon_density</code> (dropped edge; DD path remains recorded), and <code>electron_energy_flux_limiter_coefficient</code> (never bound; no edge, path, mirror, projection or binding receipt). Four planted in-memory fixtures, one for each population, each classified to its own expected population; the run log records all four and <code>EXIT=0</code>.</p>
<p>For a dropped-edge row, a recorded DD projection, path, source mirror, relative's edge or source-migration receipt establishes historical source evidence. A live holder without a <code>REFINED_FROM</code>, <code>HAS_SUCCESSOR</code> or recorded change-chain successor is not treated as an authorized successor. {unrelated_holders} dropped-edge names have a source node currently bound to another live name without that successor chain; those mappings need semantic review before any repair. A binding receipt without a retained path proves earlier binding but does not identify the lost DD path.</p>
<h2 id="sib-source-loss-census-writers">Identity-changing writers</h2>
<div class="scroll"><table><thead><tr><th>Writer and line</th><th>Edge verdict</th></tr></thead><tbody>
<tr><td><code>edit.py:891, 3393-3449</code>; <code>graph_ops.py:19026-19055, 19276-19307</code></td><td>Field edits keep identity. Rename delegates to refined-name persistence, which moves the expected producer set transactionally and verifies the count. The human-edit branch permits an empty source cohort, so a rename of an already sourceless predecessor can create a never-bound successor.</td></tr>
<tr><td><code>edit.py:1479-1495, 3057</code></td><td><code>sn supersede</code> fold deletes the old producing binding, creates the target binding and updates the source mirror inside the fold transaction.</td></tr>
<tr><td><code>workers.py:7533-7557</code>; <code>graph_ops.py:19276-19307</code></td><td>The refine worker delegates to <code>persist_refined_name</code>; its final transaction retargets every candidate source and refuses a short move.</td></tr>
<tr><td><code>graph_ops.py:19857-19955, 20190-20199, 23381-23425</code></td><td>Generated supersede and ancestor fold retarget sources. Ancestor dedup removes a redundant old edge only when the same source also produces the ancestor.</td></tr>
<tr><td><code>graph_ops.py:22627-22640</code></td><td><code>supersede_exhausted_standard_name_orphans</code> selects exhausted, non-derived, sourceless names without a successor condition. It does not move or copy sources.</td></tr>
<tr><td><code>harmonize.py:703-760</code>; <code>graph_ops.py:25746-25797</code></td><td>Harmonize stamps family signatures and documentation state; it does not move or copy <code>PRODUCED_NAME</code>.</td></tr>
<tr><td><code>catalog_import.py:51-101</code>; <code>graph_ops.py:5547-5561, 5810-5835, 7698</code></td><td>Catalog import provenance stamps existing names only. <code>write_standard_names</code> can create a name and DD projection without a producer edge; the composed-source finalizer separately creates <code>PRODUCED_NAME</code>. A name written outside that finalizer can therefore remain unbound.</td></tr>
<tr><td><code>attachment_audit.py:240-271, 1170-1185, 2758-2767</code>; <code>signed_manifest.py:4904</code></td><td>Consistency reconciliation and signed stale-source detachment delete producer edges. When both operations occur on one name, the unordered change records do not establish which removed its final edge; attribution remains unknown.</td></tr>
</tbody></table></div>
<p>Per-name attribution is in the results JSON. Where only a historical path or source-migration receipt survives, the remover is recorded as <code>unknown</code>: the census searched source mirrors, DD projections, lineage and change operations, and found no later deletion receipt. The earlier move is proved, but blaming that move for the later absence would exceed the evidence.</p>
<div class="scroll"><table><thead><tr><th>Attribution</th><th>Names</th></tr></thead><tbody>{writer_rows}</tbody></table></div>
<h2 id="sib-source-loss-census-names">All sourceless live names</h2>
<p>The JSON contains every name, its stage, population, source evidence, unrelated live source holders, repair source, writer attribution and uncertainty. <code>recorded_path</code> and <code>ancestor_or_successor_edge</code> can support deterministic history repair; <code>binding_receipt_without_path</code> and <code>none</code> need DD search.</p>
<div class="scroll"><table><thead><tr><th>Name</th><th>Stage</th><th>Population</th><th>Live successor</th><th>Writer</th><th>Repair source</th><th>Recorded path sample</th></tr></thead><tbody>{name_rows}</tbody></table></div>
<h2 id="sib-source-loss-census-refine-cases">Refine comparison cases</h2>
<p>The sheet lists {len(cases)} cases. Current location counts: {escape(", ".join(f"{count} {kind}" for kind, count in sorted(cases_by_location.items())))}. Fifty-one are now superseded predecessors, one is directly sourced and live, and ten remain live dropped-edge names. The <code>outside_reason</code>, <code>source_holding_successors</code> and <code>successor_sources</code> fields in JSON identify their current status, holders and source IDs.</p>
<div class="scroll"><table><thead><tr><th>Sheet case</th><th>Current population</th><th>Status</th><th>Live source holder</th><th>Source location</th><th>Recorded path sample</th></tr></thead><tbody>{case_rows}</tbody></table></div>
</article></body></html>
"""
    REPORT.write_text(report)


if __name__ == "__main__":
    main()
