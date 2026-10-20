"""Rebuild a blind comparison sheet from recorded outputs and live DD context."""

from __future__ import annotations

import argparse
import hashlib
import html
import importlib.util
import json
from pathlib import Path

from neo4j import READ_ACCESS, unit_of_work

from imas_codex.graph.client import GraphClient

HERE = Path(__file__).resolve().parent
BUILDER = HERE / "srtj-refine-ab-run.py"
RESULTS = HERE / "srtj-refine-ab-run-results.json"
SHEET = HERE / "srtj-refine-ab-label-sheet.json"
KEY = HERE / "srtj-refine-ab-arm-key.json"
FRAGMENT = HERE / "srtj-refine-ab-sheet-rebuild.html"
ANCHOR = "srtj-refine-ab-sheet-rebuild"

PATH_QUERY = """
UNWIND $ids AS path
OPTIONAL MATCH (node:IMASNode {id: path})
OPTIONAL MATCH (node)-[:HAS_UNIT]->(unit:Unit)
RETURN path, node.id AS node_id, node.description AS description,
       node.documentation AS documentation, unit.id AS unit
"""
IDENTITY_QUERY = """
UNWIND $ids AS nid
OPTIONAL MATCH (sn:StandardName {id: nid})
OPTIONAL MATCH (sn)<-[:PRODUCED_NAME]-(src:StandardNameSource)
RETURN nid, sn.id AS name_id, sn.status AS status,
       collect(src.id) AS sources
"""


class ReadOnlySources:
    """Only the enumerated source reads can reach a read transaction."""

    def __init__(self, client, builder):
        self.client = client
        self.allowed = {
            builder.DD_SOURCE_QUERY,
            builder.DD_PROJECTION_QUERY,
            PATH_QUERY,
            IDENTITY_QUERY,
        }
        self.receipts = []

    def query(self, cypher, **params):
        if cypher not in self.allowed:
            raise ValueError("query is not an allowlisted source read")

        @unit_of_work(timeout=10)
        def read(tx):
            result = tx.run(cypher, **params)
            rows = [dict(row) for row in result]
            summary = result.consume()
            assert not summary.counters.contains_updates
            self.receipts.append(
                {
                    "query_sha256": hashlib.sha256(cypher.encode()).hexdigest(),
                    "requested": len(params["ids"]),
                    "returned": len(rows),
                    "contains_updates": summary.counters.contains_updates,
                }
            )
            return rows

        with self.client._driver.session(default_access_mode=READ_ACCESS) as session:
            return session.execute_read(read)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("recorded_builder", BUILDER)
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    print(f"measurement_module: {builder.__file__}", flush=True)
    print(f"measurement_cwd: {Path.cwd().resolve()}", flush=True)

    inputs = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (BUILDER, RESULTS)
    }
    recorded = json.loads(RESULTS.read_text())
    cohort = recorded["cases"]
    receipts = {
        (case["sn_id"], arm): case[f"{arm}_arm"]
        for case in cohort
        for arm in builder.ARMS
        if case.get(f"{arm}_arm")
    }
    assert len(cohort) == 120
    assert sum(bool(row.get("refined_name")) for row in receipts.values()) == 214
    stored_cohort = [
        {
            **case,
            "path": (case.get("dd_source") or {}).get("source_path", ""),
            "source_description": (case.get("dd_source") or {}).get(
                "source_description", ""
            ),
            "source_unit": (case.get("dd_source") or {}).get("source_unit", ""),
        }
        for case in cohort
    ]
    baseline, baseline_key = builder.build_sheet(stored_cohort, receipts, {})
    assert len(baseline) == recorded["sheet_rows"] == 82
    assert len(baseline_key["cases_in_sheet"]) == recorded["sheet_cases"] == 104
    assert (
        baseline_key["cases_without_dd_source"]
        == recorded["sheet_cases_without_dd_source"]
    )
    assert len(baseline_key["cases_without_dd_source"]) == 62
    baseline_cases = {
        c for entry in baseline_key["rows"].values() for c in entry["cases"]
    }
    assert len(baseline_cases) == 42
    delivered = json.loads(SHEET.read_text())
    print(
        "REPRODUCTION: recorded context builds 82 rows, 42 contributing cases, "
        "104 complete pairs, 62 complete cases without DD source.",
        flush=True,
    )

    names = sorted({c[field] for c in cohort for field in ("sn_id", "prior_name")})
    with GraphClient() as client:
        reader = ReadOnlySources(client, builder)
        try:
            reader.query("CREATE (:ForbiddenProbe)", ids=[])
        except ValueError as error:
            print(
                f"REFUSAL: write query rejected before execution: {error}", flush=True
            )
        else:
            raise AssertionError("write query passed the allowlist")
        sources, projections = builder.resolve_dd_sources(reader, names)
        control_name, control_source = next(
            (case[field], sources[case[field]])
            for case in cohort
            for field in ("sn_id", "prior_name")
            if case.get("dd_source")
            and sources.get(case[field], {}).get("source_path")
            == case["dd_source"]["source_path"]
        )
        print(
            f"POSITIVE CONTROL: {control_name} -> {control_source['source_path']}",
            flush=True,
        )
        paths = sorted(
            {
                source["source_path"].removeprefix("dd:")
                for c in cohort
                if (source := builder.case_dd_source(c, sources))
            }
        )
        path_rows = reader.query(PATH_QUERY, ids=paths)
        by_path = {row["path"]: row for row in path_rows}
        assert all(row["node_id"] for row in path_rows), "bound DD path has no IMASNode"
        assert all(row["description"] or row["documentation"] for row in path_rows)
        live_cohort = []
        for case in cohort:
            source = builder.case_dd_source(case, sources)
            if source:
                node = by_path[source["source_path"].removeprefix("dd:")]
                case = {
                    **case,
                    "path": source["source_path"],
                    "source_description": node["description"] or node["documentation"],
                    "source_unit": node["unit"] or "",
                }
            live_cohort.append(case)
        missing = [c for c in live_cohort if not builder.case_dd_source(c, sources)]
        missing_names = sorted({c[f] for c in missing for f in ("sn_id", "prior_name")})
        identities = {
            row["nid"]: row for row in reader.query(IDENTITY_QUERY, ids=missing_names)
        }

    rows, key = builder.build_sheet(live_cohort, receipts, sources)
    contributing, complete, incomplete, no_source, identical = builder.sheet_population(
        live_cohort, receipts, sources
    )
    produced = {
        (case["sn_id"], arm): value
        for case, outputs in contributing
        for arm, value in outputs.items()
    }
    assert not builder.sheet_key_findings(key["rows"], rows, produced)
    assert not builder.sheet_leaks(rows)
    assert not builder.sheet_blank_paths(rows)
    assert not builder.sheet_blank_names(rows)
    assert all(set(row) == builder.ALLOWED_ROW_FIELDS for row in rows)
    assert set(complete) == set(baseline_key["cases_in_sheet"])
    assert baseline_cases <= {c["sn_id"] for c, _ in contributing}
    planted = builder._both_arms_key_with_differing_outputs(key["rows"], produced)
    assert planted is not None
    findings = builder.sheet_key_findings(planted[0], rows, produced)
    assert findings
    print(f"REFUSAL: false arm credit: {findings[0]}", flush=True)
    for detector, mutation in (
        (builder.sheet_leaks, {"arm": "checks"}),
        (builder.sheet_blank_paths, {"source_path": ""}),
        (builder.sheet_blank_names, {"refined_name": ""}),
    ):
        findings = detector([{**rows[0], **mutation}])
        assert findings
        print(f"REFUSAL: {findings[0]}", flush=True)

    missing_report = []
    for case in missing:
        bindings = [identities[case[f]] for f in ("sn_id", "prior_name")]
        assert all(not any(s.startswith("dd:") for s in r["sources"]) for r in bindings)
        reason = "No DD PRODUCED_NAME provenance for either recorded identity"
        projection = builder.case_dd_source(case, projections)
        if projection:
            reason += f"; projection-only {projection['source_path']} is not provenance"
        else:
            reason += "; neither identity has a DD projection"
        missing_report.append(
            {
                "case": case["sn_id"],
                "ancestor": case["prior_name"],
                "complete_pair": case["sn_id"] in complete,
                "reason": reason,
                "identities": bindings,
            }
        )

    # The builder's cases_in_sheet field counts complete pairs before source exclusion.
    key["contributing_cases"] = sorted(c["sn_id"] for c, _ in contributing)
    key["source_lookup_missing_cases"] = missing_report
    key["input_sha256"] = inputs
    summary = {
        "baseline_rows": 82,
        "baseline_complete_cases": 104,
        "baseline_contributing_cases": 42,
        "baseline_sourceless_complete_cases": 62,
        "rows": len(rows),
        "complete_cases": len(complete),
        "contributing_cases": len(contributing),
        "incomplete_cases": len(incomplete),
        "sourceless_complete_cases": len(no_source),
        "sourceless_all_cases": len(missing),
        "identical_output_pairs": identical,
        "recorded_named_outputs": 214,
        "credited_outputs": len(produced),
        "llm_calls": 0,
        "spend_usd": 0,
        "graph_updates": 0,
        "dd_paths_verified": len(paths),
    }
    # Re-execution accepts only an untouched baseline or this exact rebuilt sheet.
    assert delivered == baseline or delivered == rows
    for path, value in ((SHEET, rows), (KEY, key)):
        payload = json.dumps(value, ensure_ascii=False, indent=1) + "\n"
        assert len(payload.encode()) <= 300_000
        path.write_text(payload)
        assert json.loads(path.read_text()) == value
    assert hashlib.sha256(RESULTS.read_bytes()).hexdigest() == inputs[RESULTS.name]
    assert hashlib.sha256(BUILDER.read_bytes()).hexdigest() == inputs[BUILDER.name]
    args.receipt.write_text(
        json.dumps(
            {
                "summary": summary,
                "graph_read_receipts": reader.receipts,
                "missing_cases": missing_report,
                "input_sha256": inputs,
                "output_sha256": {
                    p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in (SHEET, KEY)
                },
            },
            indent=2,
        )
        + "\n"
    )
    missing_html = "\n".join(
        f"<tr><td><code>{html.escape(r['case'])}</code></td>"
        f"<td><code>{html.escape(r['ancestor'])}</code></td>"
        f"<td>{'complete' if r['complete_pair'] else 'incomplete'}</td>"
        f"<td>{html.escape(r['reason'])}</td></tr>"
        for r in missing_report
    )
    FRAGMENT.write_text(
        f'<h2 id="{ANCHOR}">Blind refine sheet rebuilt from recorded outputs</h2>\n'
        "<p>The original builder reproduced 82 rows from 42 contributing cases, "
        "with 104 complete pairs and 62 complete cases excluded for missing DD sources. "
        "The 104 count is the paired population, not the number represented on the sheet.</p>\n"
        f"<p>The rebuilt sheet has <strong>{len(rows)} rows</strong> "
        f"({len(rows) - 82:+d}), representing <strong>{len(contributing)} cases</strong> "
        f"({len(contributing) - 42:+d}) of the same 104 complete pairs. "
        f"{len(no_source)} complete cases remain sourceless. "
        f"The original 16 incomplete cases remain excluded; {identical} identical "
        f"output pairs collapse. {len(produced)} recorded outputs are credited.</p>\n"
        "<p>Only recorded case outputs were used. The original source resolver and "
        "sheet builder were imported unchanged. DD paths were verified against live "
        f"IMASNode descriptions and units ({len(paths)} distinct paths). "
        "PRODUCED_NAME remains the provenance route; projection-only matches are "
        "diagnostics. All graph receipts report no updates. LLM calls: 0; spend: USD 0.</p>\n"
        "<p>The sheet contains only its seven allowed fields. The separate arm key "
        "must not be given to the blind labeller. The driver checks every output credit, "
        "refuses a planted false arm credit, and refuses planted arm leakage, blank "
        "source path and blank name. Its read adapter refuses a write query before "
        "execution. A known bound case serves as the live lookup positive control.</p>\n"
        f'<h3 id="{ANCHOR}-missing">Every remaining sourceless case</h3>\n'
        "<table><thead><tr><th>Case</th><th>Ancestor</th><th>Pair</th><th>Reason</th>"
        f"</tr></thead><tbody>{missing_html}</tbody></table>\n"
        "<p>Missing cases include incomplete pairs so the source census covers all 120 "
        "recorded cases. The arm key records identity existence, lifecycle status, and "
        "all source ids for each missing case. It also distinguishes contributing_cases "
        "from the builder’s historical cases_in_sheet field, which counts complete pairs.</p>\n"
        f"<p>Driver: <code>{html.escape(Path(__file__).name)}</code>; "
        f"receipt: <code>{html.escape(str(args.receipt))}</code>. "
        "The driver log is alongside the receipt as <code>rebuild.log</code> and ends "
        "<code>EXIT=0</code>. Blind labelling and scoring remain separate work.</p>\n"
    )
    print("SUMMARY: " + json.dumps(summary, sort_keys=True), flush=True)
    print(f"WROTE: {SHEET}\nWROTE: {KEY}\nWROTE: {FRAGMENT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
