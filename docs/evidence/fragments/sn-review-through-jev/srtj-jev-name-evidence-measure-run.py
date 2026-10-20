"""Measure name discrimination with live source and family evidence."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import html
import json
import math
import os
import re
import subprocess
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

from neo4j import READ_ACCESS, unit_of_work

from imas_codex.discovery.base import evaluation, judgment
from imas_codex.graph.client import GraphClient
from imas_codex.settings import get_model
from imas_codex.standard_names.context import fetch_sibling_family
from imas_codex.standard_names.dd_resolutions import resolve_dd_row
from imas_codex.standard_names.review import (
    cascade,
    cascade_calibration as calibration,
    jev_review as jev,
)
from imas_codex.standard_names.workers import (
    _DD_CONTEXT_QUERY,
    _DOCS_GEN_PARENT_CHILDREN_QUERY,
)

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
STEM = Path(__file__).name.removesuffix("-run.py")
FIELDS = (
    "dd_source_docs",
    "parent_path",
    "sibling_names",
    "sibling_fields",
    "derived_children",
    "source_paths",
    "owner_segment",
)
SPLITS = ("calibration", "validation")
CEILING = 2.0
RESERVE = 0.10
SOURCE_QUERY = """
UNWIND $ids AS id
OPTIONAL MATCH (sn:StandardName {id: id})
OPTIONAL MATCH (source:StandardNameSource)-[:PRODUCED_NAME]->(sn)
WITH id, sn, source ORDER BY source.id
RETURN id, sn.id AS found, sn.origin AS origin,
       sn.description AS description,
       collect(CASE WHEN source IS NULL THEN null ELSE {
         id: source.id, source_type: source.source_type,
         source_id: source.source_id, dd_path: source.dd_path,
         dd_version: source.dd_version,
         dd_documentation: source.dd_documentation,
         dd_parent_path: source.dd_parent_path,
         dd_parent_documentation: source.dd_parent_documentation,
         dd_unit: source.dd_unit,
         dd_snapshot_pinned: source.dd_snapshot_pinned
       } END) AS sources
ORDER BY id
"""


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    with path.open("w") as stream:
        json.dump(value, stream, separators=(",", ":"), allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def receipt_map(path):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    return {r["name"]: r for r in rows if r.get("mode") == "base" and r.get("answered")}


def unflatten(state):
    item = {
        k: v for k, v in state["name"].items() if k not in {"segments", "decomposition"}
    }
    item.update(state["name"]["segments"])
    item.update(
        {k: v["value"] for k, v in state["neighbourhood"].items() if v["observed"]}
    )
    return item


def presence(state, field):
    if state is None:
        return "missing"
    cell = state["neighbourhood"].get(field)
    if cell is None or not cell["observed"]:
        return "unobserved"
    return "nonempty" if cell["value"] else "empty"


def refuse_write(cypher):
    if re.search(
        r"\b(CREATE|MERGE|SET|DELETE|REMOVE|DROP|LOAD|FOREACH)\b", cypher, re.I
    ):
        raise ValueError("read-only evidence reader refuses mutation")


class ReadOnlyGraph:
    """Adapt production graph readers to an explicitly read-only transaction."""

    def __init__(self):
        self.client = GraphClient()
        self.calls = 0
        self.errors = []

    def query(self, cypher, **params):
        refuse_write(cypher)
        try:
            with self.client._driver.session(
                default_access_mode=READ_ACCESS
            ) as session:
                rows = session.execute_read(
                    unit_of_work(timeout=30)(lambda tx: tx.run(cypher, **params).data())
                )
            self.calls += 1
            return rows
        except Exception as exc:
            self.errors.append(str(exc))
            raise

    def close(self):
        self.client.close()


def require_budget(spent):
    if spent + RESERVE > CEILING:
        raise ValueError("cost admission refuses less than USD 0.10 remaining")


def controls():
    for operation in (
        lambda: refuse_write("CREATE (n)"),
        lambda: require_budget(CEILING),
    ):
        try:
            operation()
        except ValueError as exc:
            print("guard refusal:", exc, flush=True)
        else:
            raise AssertionError("guard accepted prohibited operation")
    assert evaluation.roc_auc([0, 1], [0, 1]) == 1
    assert evaluation.roc_auc([0, 1], [1, 0]) == 0
    assert evaluation.roc_auc([0, 1], [0.5, 0.5]) == 0.5


def prepare(run):
    """Mirror claim_review_names, source enrichment, and the DD/family readers."""
    frozen = receipt_map(calibration.DEFAULT_NAME_RECEIPTS)
    labels = calibration._labels(calibration.GOLDEN_NAME)
    matched = sorted(set(labels) & set(frozen))
    assert len(matched) == 701
    before_counts = {
        field: dict(Counter(presence(frozen[name]["state"], field) for name in matched))
        for field in FIELDS
    }
    assert before_counts["dd_source_docs"]["empty"] == 406
    assert before_counts["parent_path"]["unobserved"] == 411
    assert before_counts["derived_children"]["unobserved"] == 701
    assert before_counts["sibling_names"]["empty"] == 701
    print("frozen reproduction:", json.dumps(before_counts), flush=True)
    graph = ReadOnlyGraph()
    try:
        live = {row["id"]: row for row in graph.query(SOURCE_QUERY, ids=sorted(labels))}
        assert all(live[name]["found"] for name in matched)
        # The independent aggregate checks the indexed cohort and producing bindings.
        count = graph.query(
            """UNWIND $ids AS id
            MATCH (sn:StandardName {id:id})
            OPTIONAL MATCH (src:StandardNameSource)-[:PRODUCED_NAME]->(sn)
            RETURN count(DISTINCT sn) AS names, count(src) AS sources,
                   count(src.dd_documentation) AS documented""",
            ids=sorted(labels),
        )[0]
        assert count["names"] == sum(bool(row["found"]) for row in live.values())
        assert count["sources"] == sum(len(row["sources"]) for row in live.values())
        assert count["documented"] > 0
        paths = sorted(
            {
                (source.get("dd_path") or source["source_id"]).removeprefix("dd:")
                for row in live.values()
                for source in row["sources"]
                if source["source_type"] == "dd"
            }
        )
        dd = {}
        for path in paths:
            rows = graph.query(_DD_CONTEXT_QUERY, path=path)
            assert len(rows) == 1, (path, len(rows))
            dd[path] = rows[0]
        assert any(row["node_documentation"] for row in dd.values())
        items, metadata = {}, {}
        for name in matched:
            item = unflatten(frozen[name]["state"])
            row = live[name]
            source_docs, parent_paths, siblings = [], [], {}
            for source in row["sources"]:
                if source["source_type"] != "dd":
                    continue
                path = (source.get("dd_path") or source["source_id"]).removeprefix(
                    "dd:"
                )
                context = dd[path]
                pinned = source["dd_snapshot_pinned"] is True
                raw_doc = (
                    source["dd_documentation"]
                    if pinned
                    else context["node_documentation"]
                )
                resolved = resolve_dd_row(
                    {
                        "path": path,
                        "unit": source["dd_unit"] if pinned else context["raw_unit"],
                        "documentation": raw_doc,
                    },
                    dd_version=source["dd_version"] or context["dd_version"],
                )
                parent = source["dd_parent_path"] or context["parent_path"] or ""
                source_docs.append(
                    {
                        "id": path,
                        "documentation": resolved.documentation or "",
                        "unit": resolved.unit,
                        "parent_path": parent,
                        "parent_documentation": source["dd_parent_documentation"] or "",
                        "source_binding": source["id"],
                        "snapshot_pinned": pinned,
                    }
                )
                if parent:
                    parent_paths.append(parent)
                for sibling in context["sibling_fields"]:
                    if sibling.get("path"):
                        siblings[sibling["path"]] = sibling
            family = fetch_sibling_family(name, gc=graph)
            assert not graph.errors, graph.errors
            children = (
                graph.query(_DOCS_GEN_PARENT_CHILDREN_QUERY, sn_id=name)
                if row["origin"] == "derived"
                else []
            )
            item.update(
                dd_source_docs=source_docs,
                source_paths=sorted({doc["id"] for doc in source_docs}),
                parent_path=next(iter(dict.fromkeys(parent_paths)), ""),
                sibling_fields=[siblings[key] for key in sorted(siblings)],
                sibling_names=[
                    {"id": s["name"], "description": s["description"]}
                    for s in (family or {}).get("siblings", [])
                ],
                derived_children=children,
            )
            state = jev.build_name_state(item)
            assert state["name"] == frozen[name]["state"]["name"], name
            allowed = set(FIELDS)
            unexpected = {
                key
                for key in set(state["neighbourhood"])
                | set(frozen[name]["state"]["neighbourhood"])
                if key not in allowed
                and state["neighbourhood"].get(key)
                != frozen[name]["state"]["neighbourhood"].get(key)
            }
            assert not unexpected, (name, unexpected)
            items[name] = item
            metadata[name] = {
                "origin": row["origin"],
                "parent_paths": sorted(set(parent_paths)),
                "source_bindings": [s["id"] for s in row["sources"]],
            }
        snapshot = {
            "observed_at": datetime.now(UTC).isoformat(),
            "items": items,
            "metadata": metadata,
            "source_rows": live,
            "dd_context": dd,
            "controls": count,
            "query_count": graph.calls,
            "before_counts": before_counts,
            "excluded": {
                name: {
                    **labels[name],
                    "live": bool(live[name]["found"]),
                    "description": bool(live[name]["description"]),
                    "reason": "no frozen receipt; absent live identity or description",
                }
                for name in labels
                if name not in items
            },
        }
        write(run / "live-evidence.json", snapshot)
        print(
            f"prepared {len(items)} names; DD paths {len(dd)}; reads {graph.calls}; "
            f"excluded {len(snapshot['excluded'])}",
            flush=True,
        )
    finally:
        graph.close()


async def preserve_order(item, siblings, **kwargs):
    return list(siblings), 0.0, None


async def collect(run):
    snapshot = read(run / "live-evidence.json")
    receipts = run / "supplied-reviews.jsonl"
    saved = (
        [json.loads(line) for line in receipts.read_text().splitlines()]
        if receipts.exists()
        else []
    )
    seen = {row["name_id"] for row in saved}
    assert len(seen) == len(saved), "duplicate review receipt"
    spent = sum(row["cost_usd"] for row in saved)
    original_call = judgment.acall_decisions
    calls = 0

    async def once(*args, **kwargs):
        nonlocal calls
        calls += 1
        return await original_call(*args, **kwargs, max_retries=1)

    with (
        patch.object(jev, "_ordered_siblings", preserve_order),
        patch.object(judgment, "acall_decisions", once),
        receipts.open("a") as stream,
    ):
        pending = [name for name in sorted(snapshot["items"]) if name not in seen]
        for offset in range(0, len(pending), 8):
            require_budget(spent)
            names = pending[offset : offset + 8]
            result = await jev.review_names(
                [snapshot["items"][name] for name in names],
                model=get_model("sn-judge"),
                concurrency=8,
                budget_seconds=90,
            )
            for row in result.items:
                stream.write(
                    json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n"
                )
                stream.flush()
                os.fsync(stream.fileno())
            spent += result.cost
            assert spent <= CEILING
            assert math.isclose(
                result.cost, sum(row["cost_usd"] for row in result.items), abs_tol=1e-12
            )
            assert all(row["answered"] for row in result.items), names
            if (offset // 8) % 5 == 0 or offset + 8 >= len(pending):
                print(
                    f"reviews {len(seen) + offset + len(names)}/{len(snapshot['items'])}; "
                    f"spend USD {spent:.8f}",
                    flush=True,
                )
    assert calls == len(pending), (calls, len(pending))


def comparison(rows, key, resamples):
    eligible = [
        row
        for row in rows
        if row["before"].get(key) is not None and row["after"].get(key) is not None
    ]
    labels = [row["label"] for row in eligible]
    output = {
        "n": len(labels),
        "acceptable": sum(labels),
        "defective": len(labels) - sum(labels),
    }
    if len(set(labels)) != 2:
        return {**output, "before": None, "after": None, "delta": None}
    before = [row["before"][key] for row in eligible]
    after = [row["after"][key] for row in eligible]
    output.update(
        before=evaluation.roc_auc(labels, before),
        after=evaluation.roc_auc(labels, after),
        delta=evaluation.bootstrap_interval(
            labels,
            after,
            scores_b=before,
            copy_groups=[row["copy_group"] for row in eligible],
            n_resamples=resamples,
            seed=123,
        ),
    )
    return output


def scores(record, checks):
    return {
        "logistic": cascade.score_name_review(record),
        "rubric": sum(
            cascade.rubric_scores(
                "name", record, cascade._calibration()["name"]["rubric"]
            ).values()
        )
        / 80,
        **{
            check.id: 1 - record["checks"][check.id]["failure_probability"]
            if record["checks"][check.id]["answered"]
            else None
            for check in checks
        },
    }


def summarize(run, resamples):
    snapshot = read(run / "live-evidence.json")
    frozen = receipt_map(calibration.DEFAULT_NAME_RECEIPTS)
    labels = calibration._labels(calibration.GOLDEN_NAME)
    raw = [
        json.loads(line)
        for line in (run / "supplied-reviews.jsonl").read_text().splitlines()
    ]
    after = {row["name_id"]: row for row in raw}
    assert len(raw) == len(after) == len(snapshot["items"]) == 701
    assert all(row["answered"] for row in raw)
    checks = [
        c
        for c in jev._default_checks()
        if c.axis == "name" and c.question and set(c.evidence_fields) & set(FIELDS)
    ]
    all_checks = jev._name_checks(jev._default_checks())
    questions = jev.name_question_table(all_checks)
    rows = []
    for name in sorted(after):
        old = frozen[name]
        baseline = jev._item_record(
            unflatten(old["state"]),
            old["state"],
            None,
            all_checks,
            questions,
            None,
            (old["answers"], old["cost_usd"]),
        )
        current = after[name]
        assert current["state"] == jev.build_name_state(snapshot["items"][name])
        changed = [
            key
            for key in FIELDS
            if old["state"]["neighbourhood"].get(key)
            != current["state"]["neighbourhood"].get(key)
        ]
        added = [
            key
            for key in FIELDS
            if presence(current["state"], key) == "nonempty"
            and presence(old["state"], key) != "nonempty"
        ]
        rows.append(
            {
                "id": name,
                **labels[name],
                "changed": changed,
                "added_nonempty": added,
                "before": scores(baseline, checks),
                "after": scores(current, checks),
            }
        )
    write(run / "paired-scores.json", rows)
    groups = {
        "all": rows,
        "evidence_changed": [row for row in rows if row["changed"]],
        "nonempty_evidence_added": [row for row in rows if row["added_nonempty"]],
    }
    tables, presence_counts = {}, {}
    for group, members in groups.items():
        tables[group] = {}
        presence_counts[group] = {}
        for split in SPLITS:
            selected = [row for row in members if row["split"] == split]
            tables[group][split] = {
                key: comparison(selected, key, resamples)
                for key in ("logistic", "rubric", *(c.id for c in checks))
            }
            presence_counts[group][split] = {
                field: {
                    arm: dict(
                        Counter(
                            presence(
                                frozen[row["id"]]["state"]
                                if arm == "before"
                                else after[row["id"]]["state"],
                                field,
                            )
                            for row in selected
                        )
                    )
                    for arm in ("before", "after")
                }
                for field in FIELDS
            }
    assert math.isclose(
        tables["all"]["validation"]["rubric"]["before"],
        0.7223345318957778,
        abs_tol=1e-7,
    )
    # Independently reconstruct counts from golden identities and the persisted receipts.
    for group in tables:
        for split in SPLITS:
            names = {
                name
                for name in after
                if labels[name]["split"] == split
                and (
                    group == "all"
                    or any(
                        (
                            frozen[name]["state"]["neighbourhood"].get(field)
                            != after[name]["state"]["neighbourhood"].get(field)
                        )
                        if group == "evidence_changed"
                        else (
                            presence(after[name]["state"], field) == "nonempty"
                            and presence(frozen[name]["state"], field) != "nonempty"
                        )
                        for field in FIELDS
                    )
                )
            }
            for result in tables[group][split].values():
                assert result["n"] == len(names)
                assert result["acceptable"] == sum(
                    labels[name]["label"] for name in names
                )
            for field in FIELDS:
                for arm, records in (("before", frozen), ("after", after)):
                    independent = {
                        category: sum(
                            presence(records[name]["state"], field) == category
                            for name in names
                        )
                        for category in ("nonempty", "empty", "unobserved", "missing")
                    }
                    assert {
                        k: v for k, v in independent.items() if v
                    } == presence_counts[group][split][field][arm]
    report = {
        "schema": "name-evidence-discrimination",
        "revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "model": get_model("sn-judge"),
        "spend_usd": sum(row["cost_usd"] for row in raw),
        "ceiling_usd": CEILING,
        "review_calls": len(raw),
        "rerank_calls": 0,
        "graph_writes": 0,
        "graph_reads": snapshot["query_count"],
        "observed_at": snapshot["observed_at"],
        "exclusions": snapshot["excluded"],
        "controls": snapshot["controls"],
        "frozen_reproduction": snapshot["before_counts"],
        "tables": tables,
        "evidence_presence": presence_counts,
        "presence_fields": FIELDS,
        "item_columns": [
            "id",
            "split",
            "label",
            "before_presence",
            "after_presence",
            "changed_fields",
            "added_nonempty_fields",
        ],
        "items": [
            [
                row["id"],
                row["split"],
                row["label"],
                [presence(frozen[row["id"]]["state"], f) for f in FIELDS],
                [presence(after[row["id"]]["state"], f) for f in FIELDS],
                row["changed"],
                row["added_nonempty"],
            ]
            for row in rows
        ],
        "examples": [
            {
                "id": row["id"],
                "label": row["label"],
                "description": after[row["id"]]["state"]["name"]["description"],
                "sources": snapshot["items"][row["id"]]["dd_source_docs"],
                "scores": {
                    arm: {key: row[arm][key] for key in ("logistic", "rubric")}
                    for arm in ("before", "after")
                },
            }
            for row in sorted(
                rows, key=lambda r: -abs(r["after"]["rubric"] - r["before"]["rubric"])
            )[:6]
        ],
        "inputs": [
            {"path": str(path), "sha256": digest(path)}
            for path in (
                calibration.GOLDEN_NAME,
                calibration.DEFAULT_NAME_RECEIPTS,
                Path(cascade.__file__).with_name("cascade_thresholds.json"),
                Path(cascade.__file__),
                Path(jev.__file__),
                Path(evaluation.__file__),
                run / "live-evidence.json",
                run / "supplied-reviews.jsonl",
                run / "paired-scores.json",
            )
        ],
    }
    assert 0 < report["spend_usd"] <= CEILING
    write(HERE / (STEM + ".json"), report)
    render(report)
    print(
        "SUMMARY "
        + json.dumps(
            {
                "paired": len(rows),
                "spent_usd": report["spend_usd"],
                "validation": tables["all"]["validation"],
            }
        ),
        flush=True,
    )


def render(report):
    parts = [
        f'<section id="{STEM}"><h2>Name review with live source and family evidence</h2>',
        "<p>Candidate identities, descriptions, questions and fitted mappings are fixed. "
        "The supplied arm reads exact PRODUCED_NAME bindings, pinned DD documentation "
        "(live DD text only for unpinned sources), parent paths, DD sibling fields, "
        "name families and derived children. The driver mirrors claim_review_names "
        "and _enrich_name_review_items, reuses _DD_CONTEXT_QUERY, fetch_sibling_family "
        "and _DOCS_GEN_PARENT_CHILDREN_QUERY. All transactions are read-only.</p>",
        f"<p>{report['review_calls']} single review calls through review_names under sn-judge "
        f"({html.escape(report['model'])}); USD {report['spend_usd']:.8f} of USD 2.00. "
        "Sibling order is retained without the optional paid rerank. No product code "
        "or graph state changed; the cascade remains in shadow.</p>",
        "<p>AUC uses the landed logistic and raw dimension rubric; no fitting occurs. "
        "Intervals are paired after-minus-before AUC differences, resampling golden "
        "copy groups with replacement (95%, seed 123). Every AUC and bootstrap calls "
        "the repository evaluation owner. Per-check AUC uses compliance against "
        "whole-name golden labels, not check-specific truth.</p>",
    ]
    rendered_rows = []
    for group, splits in report["tables"].items():
        parts.append(f"<h3>{html.escape(group.replace('_', ' '))}</h3>")
        parts.append(
            "<table><thead><tr><th>Split</th><th>Score or check</th><th>n</th>"
            "<th>Acceptable</th><th>Before AUC</th><th>After AUC</th>"
            "<th>Difference [95% interval]</th></tr></thead><tbody>"
        )
        for split, metrics in splits.items():
            for key, result in metrics.items():
                delta = result["delta"]
                cells = [
                    split,
                    key,
                    str(result["n"]),
                    str(result["acceptable"]),
                    f"{result['before']:.6f}" if delta else "undefined",
                    f"{result['after']:.6f}" if delta else "undefined",
                    f"{delta['point']:+.6f} [{delta['lower']:+.6f}, {delta['upper']:+.6f}]"
                    if delta
                    else "undefined",
                ]
                rendered_rows.append(cells)
                parts.append(
                    "<tr>"
                    + "".join(f"<td>{html.escape(cell)}</td>" for cell in cells)
                    + "</tr>"
                )
        parts.append("</tbody></table>")
    parts.append(
        "<h3>Evidence present in paired items</h3><table><thead><tr>"
        "<th>Split</th><th>Field</th><th>Before nonempty / empty / unobserved</th>"
        "<th>After nonempty / empty / unobserved</th></tr></thead><tbody>"
    )
    for split, fields in report["evidence_presence"]["all"].items():
        for field, arms in fields.items():
            cells = [split, field] + [
                " / ".join(
                    str(arms[arm].get(state, 0))
                    for state in ("nonempty", "empty", "unobserved")
                )
                for arm in ("before", "after")
            ]
            rendered_rows.append(cells)
            parts.append(
                "<tr>"
                + "".join(f"<td>{html.escape(cell)}</td>" for cell in cells)
                + "</tr>"
            )
    parts.append(
        "</tbody></table><p>Changed includes observed-empty replacing unobserved, "
        "source refreshes, and changed content. Nonempty evidence added requires "
        "at least one field to move from empty or unknown to nonempty. "
        "The JSON records presence for every paired item and explicit exclusions.</p>"
    )
    parts.append("<h3>Representative score movements</h3>")
    for example in report["examples"]:
        parts.append(
            f"<p><code>{html.escape(example['id'])}</code> — golden label {example['label']}. "
            f"{html.escape(example['description'] or '')} "
            f"Logistic {example['scores']['before']['logistic']:.4f} → {example['scores']['after']['logistic']:.4f}; "
            f"rubric {example['scores']['before']['rubric']:.4f} → {example['scores']['after']['rubric']:.4f}. "
            f"DD bindings: {html.escape(', '.join(s['id'] for s in example['sources']) or 'none')}.</p>"
        )
    parts.append(
        "<h3>Coverage and limits</h3><p>720 unique golden identities; 701 paired, "
        "363 calibration and 338 validation. Nineteen have no frozen receipt: "
        "18 are absent from the live graph and one has no live description. "
        "They receive no invented score. Exact exclusions and evidence presence "
        "are in the JSON. Golden labels remain frozen and independent of this arm. "
        "The frozen arm was collected earlier; model service drift and stochastic "
        "variation are not separated from evidence effects by this one-call design. "
        "Current graph family context may itself reflect later catalog corrections. "
        "Repeated inspection of validation requires a fresh holdout before deployment. "
        "No source binding is inferred from a name spelling or a golden label.</p>"
    )
    parts.append(
        "<p>Guards refused a mutation string and exhausted budget before any external "
        "call. Positive controls observed real producing sources and nonempty DD "
        "documentation. Frozen missing-evidence counts and validation rubric AUC "
        "were reproduced. Tables were checked against separately recomputed counts "
        "from golden identities and persisted receipts.</p></section>"
    )
    text = "\n".join(parts)
    parsed_rows = [
        [html.unescape(cell) for cell in re.findall(r"<td>(.*?)</td>", row)]
        for row in re.findall(r"<tr>(.*?)</tr>", text)
        if "<td>" in row
    ]
    assert parsed_rows == rendered_rows
    (HERE / (STEM + ".html")).write_text(text + "\n")
    assert (HERE / (STEM + ".json")).stat().st_size <= 300_000


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "collect", "report", "all"))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--resamples", type=int, default=1000)
    args = parser.parse_args()
    print("measurement_module=" + cascade.__file__, flush=True)
    print("measurement_cwd=" + str(Path.cwd().resolve()), flush=True)
    controls()
    if args.action in {"prepare", "all"}:
        prepare(args.run_dir)
    if args.action in {"collect", "all"}:
        asyncio.run(collect(args.run_dir))
    if args.action in {"report", "all"}:
        summarize(args.run_dir, args.resamples)


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        import traceback

        traceback.print_exc()
        print("EXIT=1", flush=True)
        sys.exit(1)
    print("EXIT=0", flush=True)
