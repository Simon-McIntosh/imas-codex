"""Read-only census of stale DD sources using the live extraction selector."""

from __future__ import annotations

import ast
import inspect
import json
import time
from collections import Counter
from datetime import UTC, datetime
from html import escape
from pathlib import Path

from imas_codex.core.node_categories import SN_SOURCE_CATEGORIES
from imas_codex.graph.client import GraphClient
from imas_codex.standard_names.sources import dd
from imas_codex.standard_names.sources.base import SourceCandidate
from imas_codex.standard_names.sources.dd_qualifier import qualify_dd
from imas_codex.standard_names.unit_overrides import resolve_unit

HERE = Path(__file__).resolve().parent
DATA = HERE / "sib-stale-source-census.json"
PAGE = HERE / "sib-stale-source-census.html"
CHUNK = 50
SELECTOR_REASONS = (
    "node_category_ineligible",
    "description_missing",
    "description_empty",
    "container_without_signal_signature",
    "duplicate_ids",
    "removed_from_current_dd",
)


def read(gc: GraphClient, query: str, *, cap: int, **params: object) -> list[dict]:
    started = time.monotonic()
    rows = gc.query(query, **params)
    elapsed = time.monotonic() - started
    if elapsed >= 10:
        raise RuntimeError(f"graph query exceeded ten seconds: {elapsed:.2f}s")
    if len(rows) > cap:
        raise RuntimeError(f"graph result exceeded caller bound: {len(rows)} > {cap}")
    print(f"read rows={len(rows)} seconds={elapsed:.3f}", flush=True)
    return rows


def extraction_clauses() -> list[str]:
    """Read the selector literals from the actual extraction function's syntax tree."""
    tree = ast.parse(inspect.getsource(dd.extract_dd_candidates))
    assignments = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "where_parts"
            for target in node.targets
        )
    ]
    if len(assignments) != 1:
        raise RuntimeError("could not identify the production DD selector")
    clauses = ast.literal_eval(assignments[0].value)
    if len(clauses) != len(SELECTOR_REASONS):
        raise RuntimeError("production DD selector changed; review reason mapping")
    return clauses


def chunks(values: list[str]) -> list[list[str]]:
    return [values[index : index + CHUNK] for index in range(0, len(values), CHUNK)]


def sources(gc: GraphClient) -> list[dict]:
    records: list[dict] = []
    after = ""
    while True:
        page = read(
            gc,
            """
            MATCH (s:StandardNameSource {source_type: 'dd', status: 'stale'})
            WHERE s.id > $after
            RETURN s.id AS id, s.source_id AS path,
                   s.produced_sn_id AS recorded_name, s.created_at AS created_at,
                   s.skip_reason AS prior_skip_reason
            ORDER BY s.id LIMIT $limit
            """,
            cap=100,
            after=after,
            limit=100,
        )
        if not page:
            break
        records.extend(page)
        after = page[-1]["id"]
    if len({row["id"] for row in records}) != len(records):
        raise RuntimeError("source pagination repeated an id")
    return records


def population(gc: GraphClient, stale: list[dict]) -> list[dict]:
    records: list[dict] = []
    for batch in chunks([row["id"] for row in stale]):
        rows = read(
            gc,
            """
            UNWIND $ids AS sid
            MATCH (s:StandardNameSource {id: sid})
            OPTIONAL MATCH (n:IMASNode {id: s.source_id})
            RETURN s.id AS id, n.id AS dd_path,
                   n.lifecycle_status AS lifecycle_status,
                   [(s)-[:PRODUCED_NAME]->(name:StandardName) | name.id]
                     AS produced_names
            """,
            cap=CHUNK,
            ids=batch,
        )
        by_id = {row["id"]: row for row in rows}
        for source in stale:
            if source["id"] in by_id:
                source.update(by_id[source["id"]])
    for batch in chunks(sorted({row["path"] for row in stale})):
        rows = read(
            gc,
            """
            MATCH (other:StandardNameSource)
            WHERE other.source_id IN $paths AND other.status <> 'stale'
            RETURN other.source_id AS path, collect(other.id) AS ids
            LIMIT 51
            """,
            cap=51,
            paths=batch,
        )
        active = {row["path"]: row["ids"] for row in rows}
        for source in stale:
            if source["path"] in active:
                source["other_live_sources"] = active[source["path"]]
    for source in stale:
        if (
            source["dd_path"] is not None
            and source["lifecycle_status"] != "removed"
            and not source["produced_names"]
            and not source.get("other_live_sources")
        ):
            records.append(source)
    return records


def selector_rows(
    gc: GraphClient, paths: list[str], clauses: list[str]
) -> dict[str, dict]:
    flags = ", ".join(
        f"CASE WHEN ({clause}) THEN true ELSE false END AS flag_{index}"
        for index, clause in enumerate(clauses)
    )
    found: dict[str, dict] = {}
    for batch in chunks(paths):
        rows = read(
            gc,
            "MATCH (n:IMASNode)-[:IN_IDS]->(ids:IDS) "
            "WHERE n.id IN $paths "
            f"RETURN n.id AS path, ids.id AS ids, n.node_category AS category, {flags} "
            "LIMIT 101",
            cap=100,
            paths=batch,
            sn_categories=list(SN_SOURCE_CATEGORIES),
        )
        for row in rows:
            if row["path"] in found:
                raise RuntimeError(f"multiple IDS bindings: {row['path']}")
            found[row["path"]] = row
    return found


def extracted_rows(
    gc: GraphClient, paths: list[str], clauses: list[str]
) -> dict[str, dict]:
    found: dict[str, dict] = {}
    where = "n.id IN $paths AND " + " AND ".join(clauses)
    query = dd._ENRICHED_QUERY_TPL.format(where_clause=where, limit_clause="")
    for batch in chunks(paths):
        rows = read(
            gc,
            query + "\nLIMIT 501",
            cap=500,
            paths=batch,
            sn_categories=list(SN_SOURCE_CATEGORIES),
        )
        for row in rows:
            found.setdefault(row["path"], row)
    return found


def qualification(row: dict, dd_version: str | None) -> tuple[bool, str]:
    row = dict(row)
    dd._resolve_authoritative_unit_edges([row])
    row["unit"] = (
        None
        if len(row.get("unit_relationships") or []) > 1
        else row.get("unit_from_rel") or row.get("unit") or None
    )
    resolved = dd._apply_typed_dd_resolutions([row], dd_version)
    if not resolved:
        return False, "typed_dd_resolution_refused"
    row = resolved[0]
    effective_unit, meta = resolve_unit(row["path"], row.get("unit"))
    if meta and meta.get("rule") == "skip":
        return False, str(meta["skip_reason"])
    if meta and meta.get("rule") == "override":
        row["unit"] = effective_unit
    unit_kept = dd._apply_unit_overrides([row], write_skipped=False)
    if not unit_kept:
        return False, "dd_unit_unresolvable"
    verdict = qualify_dd(SourceCandidate.from_dd_row(unit_kept[0]))
    return verdict.eligible, verdict.reason_code or "eligible"


def accepted_holders(gc: GraphClient, paths: list[str]) -> dict[str, list[dict]]:
    holders: dict[str, list[dict]] = {}
    for batch in chunks(paths):
        rows = read(
            gc,
            """
            UNWIND $paths AS path
            MATCH (n:IMASNode {id: path})-[:HAS_STANDARD_NAME]->(name:StandardName)
            WHERE name.name_stage = 'accepted'
            OPTIONAL MATCH (producer:StandardNameSource)-[:PRODUCED_NAME]->(name)
            WHERE producer.status <> 'stale'
            RETURN path, name.id AS name, collect(DISTINCT producer.id) AS producers
            LIMIT 501
            """,
            cap=500,
            paths=batch,
        )
        for row in rows:
            if row["producers"]:
                holders.setdefault(row["path"], []).append(row)
    return holders


def recorded_holders(gc: GraphClient, names: list[str]) -> dict[str, list[str]]:
    holders: dict[str, list[str]] = {}
    for batch in chunks(names):
        rows = read(
            gc,
            """
            MATCH (name:StandardName)
            WHERE name.id IN $names AND name.name_stage = 'accepted'
            MATCH (source:StandardNameSource)-[:PRODUCED_NAME]->(name)
            WHERE source.status <> 'stale'
            RETURN name.id AS name, collect(DISTINCT source.id) AS producers
            LIMIT 51
            """,
            cap=51,
            names=batch,
        )
        holders.update({row["name"]: row["producers"] for row in rows})
    return holders


def main() -> None:
    clauses = extraction_clauses()
    with GraphClient() as gc:
        stale = sources(gc)
        cohort = population(gc, stale)
        print(f"stale_dd={len(stale)} population={len(cohort)}", flush=True)
        if not cohort:
            raise RuntimeError("empty cohort; source instrument may be blind")
        paths = [row["path"] for row in cohort]
        controls = read(
            gc,
            """
            MATCH (source:StandardNameSource {source_type: 'dd', status: 'composed'})
                  -[:PRODUCED_NAME]->(name:StandardName {name_stage: 'accepted'})
            MATCH (node:IMASNode {id: source.source_id})
            RETURN DISTINCT source.source_id AS path, name.id AS name
            ORDER BY path LIMIT 100
            """,
            cap=100,
        )
        all_paths = sorted(set(paths + [row["path"] for row in controls]))
        flags = selector_rows(gc, all_paths, clauses)
        selected = [
            path
            for path in all_paths
            if path in flags
            and all(flags[path][f"flag_{index}"] for index in range(len(clauses)))
        ]
        details = extracted_rows(gc, selected, clauses)
        version = read(
            gc,
            "MATCH (version:DDVersion {is_current: true}) "
            "RETURN version.id AS id LIMIT 2",
            cap=2,
        )
        dd_version = version[0]["id"] if version else None
        qualifications = {
            path: qualification(row, dd_version) for path, row in details.items()
        }
        positive = [
            row
            for row in controls
            if row["path"] in flags
            and all(
                flags[row["path"]][f"flag_{index}"] for index in range(len(clauses))
            )
            and qualifications.get(row["path"], (False, ""))[0]
        ][:20]
        if len(positive) != 20:
            raise RuntimeError(f"only {len(positive)} positive controls passed")
        accepted = accepted_holders(gc, paths)
        prior_names = sorted(
            {str(row["recorded_name"]) for row in cohort if row.get("recorded_name")}
        )
        prior_holders = recorded_holders(gc, prior_names)

    rows: list[list] = []
    for source in sorted(cohort, key=lambda item: item["id"]):
        path = source["path"]
        item = flags.get(path)
        if item is None:
            eligible, reason = False, "no_current_IDS_binding"
            ids, category = path.split("/", 1)[0], None
        else:
            ids, category = item["ids"], item["category"]
            failed = next(
                (index for index in range(len(clauses)) if not item[f"flag_{index}"]),
                None,
            )
            if failed is not None:
                eligible, reason = False, SELECTOR_REASONS[failed]
            else:
                eligible, reason = qualifications.get(
                    path, (False, "selected_path_missing_from_enriched_query")
                )
        holder = None
        basis = None
        direct = accepted.get(path) or []
        if direct:
            holder = direct[0]["name"]
            basis = "current DD path points to accepted name with live producer"
        elif source.get("recorded_name") in prior_holders:
            holder = source["recorded_name"]
            basis = "stale source records accepted name with another live producer"
        if holder:
            label = "covered"
        elif not eligible:
            label = "ineligible"
        else:
            label = "dropped"
        evidence = basis or reason
        rows.append([path, ids, category, label, eligible, evidence, holder])

    class_counts = Counter(row[3] for row in rows)
    by_ids: dict[str, Counter] = {}
    by_category: dict[str, Counter] = {}
    for row in rows:
        by_ids.setdefault(row[1], Counter())[row[3]] += 1
        by_category.setdefault(row[2] or "null", Counter())[row[3]] += 1
    assert sum(class_counts.values()) == len(cohort)
    assert sum(sum(bucket.values()) for bucket in by_ids.values()) == len(cohort)
    assert sum(sum(bucket.values()) for bucket in by_category.values()) == len(cohort)
    result = {
        "observed_at": datetime.now(UTC).isoformat(),
        "population": len(cohort),
        "stale_dd_total": len(stale),
        "class_counts": dict(sorted(class_counts.items())),
        "by_ids": {key: dict(value) for key, value in sorted(by_ids.items())},
        "by_category": {key: dict(value) for key, value in sorted(by_category.items())},
        "positive_control": {
            "count": len(positive),
            "mode": "production selector with force=True, followed by unit resolution and qualify_dd",
            "rows": positive,
        },
        "selector_owner": "imas_codex.standard_names.sources.dd.extract_dd_candidates",
        "stale_writer": "imas_codex.standard_names.graph_ops.mark_sources_stale",
        "columns": [
            "source_id",
            "ids",
            "node_category",
            "class",
            "eligible",
            "evidence",
            "accepted_holder",
        ],
        "rows": rows,
    }
    DATA.write_text(
        json.dumps(result, ensure_ascii=False, separators=(",", ":")) + "\n"
    )
    dropped = Counter(row[1] for row in rows if row[3] == "dropped")
    summary = ", ".join(
        f"{key}: {value}" for key, value in sorted(class_counts.items())
    )
    spread = ", ".join(f"{key}: {value}" for key, value in dropped.most_common())
    PAGE.write_text(
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="reckon-type" content="evidence-fragment">'
        "<title>Stale DD source census</title></head><body><main>"
        "<h1>Stale DD sources without a produced name</h1>"
        f"<p>Live population: {len(cohort)} of {len(stale)} stale DD sources. "
        f"Classes: {escape(summary)}. The 20 composed accepted-source controls "
        "all passed the production selector and qualifier with force enabled.</p>"
        f"<p>Dropped paths by IDS: {escape(spread or 'none')}.</p>"
        "<p>The selector clauses were read from extract_dd_candidates itself and run "
        "against each current DD path; unit resolution and qualify_dd were called "
        "without graph writes. mark_sources_stale is the status writer. Its reconciliation "
        "treats a missing FROM_DD_PATH edge or removed upstream DD node as stale; "
        "that historical status does not prove the current DD path is absent.</p>"
        "<table><thead><tr><th>Class</th><th>Count</th><th>Recommendation</th></tr></thead><tbody>"
        f"<tr><td>Dropped</td><td>{class_counts['dropped']}</td><td>Re-extract and compose; audit why the source was left stale.</td></tr>"
        f"<tr><td>Covered</td><td>{class_counts['covered']}</td><td>Verify the recorded accepted holder and retire the stale duplicate source.</td></tr>"
        f"<tr><td>Ineligible</td><td>{class_counts['ineligible']}</td><td>Keep out of compose; reconcile status to the selector or qualifier reason.</td></tr>"
        f"<tr><td>Other</td><td>{class_counts['other']}</td><td>Review the row evidence before changing provenance.</td></tr>"
        "</tbody></table><p>The JSON gives one row per source and counts by IDS and node category. "
        "Covered means the DD path points to an accepted name with a live producer, "
        "or the stale source records such a name. No semantic match was guessed.</p>"
        "</main></body></html>\n"
    )
    print(f"population={len(cohort)} classes={summary} controls={len(positive)}")
    print(f"json_bytes={DATA.stat().st_size}")
    if DATA.stat().st_size > 300_000:
        raise RuntimeError("census JSON exceeds repository data-file size limit")


if __name__ == "__main__":
    main()
