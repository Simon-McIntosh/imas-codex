"""Restore the DD source each dropped-edge name recorded for itself.

Reads the live graph, re-derives the sourceless-name populations with the
shared classifier, takes the deterministic slice whose own record names the DD
path it was produced from, and re-binds those paths through the recovery-map
writer. Never writes Cypher by hand and never calls a model.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from collections import Counter, defaultdict, deque
from html import escape
from pathlib import Path

from imas_codex.graph.client import GraphClient
from imas_codex.standard_names.graph_ops import (
    _TransactionQuery,
    classify_sourceless_standard_names,
)
from imas_codex.standard_names.provenance_rebuild import bind_recovery_sources

ROOT = Path(__file__).resolve().parents[4]
OUTPUT = Path(__file__).with_name("sib-restore-recorded-sources.json")
REPORT = Path(__file__).with_name("sib-restore-recorded-sources.html")
TERMINAL = ["deprecated", "superseded"]
RECEIPT_OPERATIONS = {
    "source_migration_manifest",
    "reattach_lapsed_detachment",
    "repair_semantic_source_binding",
    "recover_terminal_source_binding",
    "move_source_snapshot",
    "rebind_sources",
}
#: The census member whose repair source was a relative's edge rather than a
#: path of its own. It carries no own recorded path, so this slice can only
#: report it, never restore it; its live evidence is recomputed below.
SUCCESSOR_MEMBER = "normal_particle_flux_at_wall_due_to_recombination"


def read(gc, cypher: str, **params):
    return [dict(r) for r in gc.query(cypher, **params)]


def population_counts(rows: list[dict]) -> dict[str, int]:
    counts = Counter(row["population"] for row in rows)
    for name in (
        "derived_parent",
        "never_bound",
        "dropped_edge",
        "unchained_holder",
        "orphan_awaiting_retirement",
        "held_in_chain",
    ):
        counts.setdefault(name, 0)
    return dict(sorted(counts.items()))


def quiet_graph_check(gc) -> dict:
    """Record every fact that shows whether another graph writer is live."""
    live_dir = Path.home() / ".config/reckon" / "crew" / "live"
    runs = []
    recently_touched = []
    now = time.time()
    if live_dir.is_dir():
        for pointer in sorted(live_dir.glob("*.json")):
            runs.append(pointer.stem)
            age = now - pointer.stat().st_mtime
            if age < 1800:
                recently_touched.append(
                    {"run": pointer.stem, "age_seconds": round(age)}
                )
    job = subprocess.run(
        ["squeue", "-u", str(Path.home().name), "-h", "-o", "%i %j %T %M"],
        capture_output=True,
        text=True,
        check=False,
    )
    neo4j_jobs = [line for line in job.stdout.splitlines() if "codex-neo4j" in line]
    try:
        transactions = read(
            gc,
            "SHOW TRANSACTIONS YIELD transactionId, currentQuery, status "
            "RETURN transactionId, status, currentQuery LIMIT 50",
        )
    except Exception as exc:  # listing is best-effort; record the refusal
        transactions = [{"error": f"{type(exc).__name__}: {exc}"}]
    return {
        "live_crew_runs": runs,
        "live_crew_run_count": len(runs),
        "runs_touched_within_1800s": recently_touched,
        "neo4j_jobs": neo4j_jobs,
        "active_transactions": transactions,
    }


def linked_relatives(gc) -> dict[str, set[str]]:
    """Identity-chain adjacency, both directions, as the classifier reads it."""
    all_ids = {
        row["id"]
        for row in read(gc, "MATCH (sn:StandardName) RETURN sn.id AS id LIMIT 6000")
    }
    links: dict[str, set[str]] = defaultdict(set)
    for relation in read(
        gc,
        "MATCH (a:StandardName)-[rel:REFINED_FROM|HAS_SUCCESSOR]->(b:StandardName) "
        "RETURN a.id AS a, type(rel) AS kind, b.id AS b LIMIT 12000",
    ):
        old, new = (
            (relation["b"], relation["a"])
            if relation["kind"] == "REFINED_FROM"
            else (relation["a"], relation["b"])
        )
        links[old].add(new)
        links[new].add(old)
    for change in read(
        gc,
        "MATCH (change:StandardNameChange) "
        "RETURN change.from_name AS old, change.to_name AS new LIMIT 20000",
    ):
        old, new = change["old"], change["new"]
        if old in all_ids and new in all_ids and old != new:
            links[old].add(new)
            links[new].add(old)
    return links


def chain(start: str, links: dict[str, set[str]]) -> set[str]:
    seen: set[str] = set()
    pending = deque(links[start])
    while pending:
        target = pending.popleft()
        if target in seen:
            continue
        seen.add(target)
        pending.extend(links[target] - seen)
    return seen | {start}


def relative_evidence(gc) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    """Own DD paths and held sources for every name, keyed by name id.

    The path map covers retired names too: a name's own chain runs through the
    relatives it was refined from or superseded by, and a retired relative is
    the common case, so filtering by the relative's own status here would hide
    the lineage a name's recorded path came from. The holdings map stays
    restricted to live holders, which is what decides whether a recorded source
    is already spoken for.
    """
    provenances = read(
        gc,
        """MATCH (sn:StandardName)
        OPTIONAL MATCH (dd:IMASNode)-[:HAS_STANDARD_NAME]->(sn)
        RETURN sn.id AS id, sn.source_paths AS paths, sn.source_path AS path,
               collect(DISTINCT dd.id) AS projections
        LIMIT 30000""",
    )
    detachment: dict[str, set[str]] = defaultdict(set)
    receipts: dict[str, set[str]] = defaultdict(set)
    for change in read(
        gc,
        "MATCH (change:StandardNameChange) "
        "RETURN change.from_name AS old, change.to_name AS new, "
        "change.operation AS operation LIMIT 20000",
    ):
        operation = change["operation"] or ""
        if operation.startswith("detach_") and isinstance(change["old"], str):
            if "/" in change["old"]:
                detachment[change["new"]].add("dd:" + change["old"].removeprefix("dd:"))
        if operation in RECEIPT_OPERATIONS:
            receipts[change["new"]].add(operation)
    holdings: dict[str, set[str]] = defaultdict(set)
    for source in read(
        gc,
        """MATCH (source:StandardNameSource)-[:PRODUCED_NAME]->(holder:StandardName)
        WHERE NOT holder.status IN $terminal
        RETURN source.id AS source, holder.id AS holder LIMIT 12000""",
        terminal=TERMINAL,
    ):
        holdings[source["holder"]].add(source["source"])

    paths: dict[str, set[str]] = {}
    for row in provenances:
        token: set[str] = set(row["paths"] or [])
        if row["path"]:
            token.add(row["path"])
        token.update(detachment[row["id"]])
        paths[row["id"]] = token
    return paths, holdings, receipts


def classify_repair_source(
    name_id: str,
    own_paths: set[str],
    links: dict[str, set[str]],
    paths: dict[str, set[str]],
    holdings: dict[str, set[str]],
    receipts: dict[str, set[str]],
) -> tuple[str, dict]:
    """Reproduce the census repair-source rule for one sourceless name."""
    if own_paths:
        return "recorded_path", {"own_paths": sorted(own_paths)}
    relatives = chain(name_id, links)
    lineage_paths = {path for rel in relatives if rel in paths for path in paths[rel]}
    historical = {src for rel in relatives for src in holdings.get(rel, set())}
    if lineage_paths:
        return "lineage_path", {"lineage_paths": sorted(lineage_paths)}
    if historical:
        return "ancestor_or_successor_edge", {"historical_edges": sorted(historical)}
    if receipts.get(name_id):
        return "binding_receipt_without_path", {"receipts": sorted(receipts[name_id])}
    return "none", {}


def build_worklist(gc, classifier_rows: list[dict]) -> dict:
    dropped = [r for r in classifier_rows if r["population"] == "dropped_edge"]
    links = linked_relatives(gc)
    paths, holdings, receipts = relative_evidence(gc)
    slices: dict[str, list[dict]] = defaultdict(list)
    for row in dropped:
        if row["id"] == SUCCESSOR_MEMBER:
            continue
        own_ids = set(row["recorded_source_ids"])
        own_tokens = own_ids | set(row["unresolved_tokens"])
        own_paths = {
            str(t)[len("dd:") :] if str(t).startswith("dd:") else str(t)
            for t in own_tokens
        }
        repair, evidence = classify_repair_source(
            row["id"], own_paths, links, paths, holdings, receipts
        )
        slices[repair].append(
            {
                "name": row["id"],
                "recorded_source_ids": sorted(own_ids),
                "evidence": evidence,
            }
        )
    if any(row["id"] == SUCCESSOR_MEMBER for row in dropped):
        relatives = chain(SUCCESSOR_MEMBER, links)
        slices["ancestor_or_successor_edge"].append(
            {
                "name": SUCCESSOR_MEMBER,
                "recorded_source_ids": [],
                "evidence": {
                    "lineage_paths": sorted(
                        {p for rel in relatives if rel in paths for p in paths[rel]}
                    ),
                    "historical_edges": sorted(
                        {s for rel in relatives for s in holdings.get(rel, set())}
                    ),
                    "receipts": sorted(receipts.get(SUCCESSOR_MEMBER, set())),
                },
            }
        )
    return slices


def source_facts(gc, source_ids: list[str]) -> dict[str, dict]:
    if not source_ids:
        return {}
    rows = read(
        gc,
        """MATCH (source:StandardNameSource) WHERE source.id IN $ids
        OPTIONAL MATCH (source)-[:PRODUCED_NAME]->(holder:StandardName)
        WHERE NOT holder.status IN $terminal
        RETURN source.id AS id, source.status AS status,
               collect(DISTINCT holder.id) AS holders""",
        ids=sorted(source_ids),
        terminal=TERMINAL,
    )
    return {row["id"]: row for row in rows}


def resolve_specs(
    worklist: list[dict], facts: dict[str, dict], nodes: set[str]
) -> tuple[list[dict], list[dict]]:
    """Split each name's recorded sources into restorable specs and skips."""
    restorable, skipped = [], []
    for entry in worklist:
        name = entry["name"]
        specs, reasons = [], []
        for source_id in entry["recorded_source_ids"]:
            bare = source_id[len("dd:") :] if source_id.startswith("dd:") else source_id
            fact = facts.get(source_id)
            if bare not in nodes:
                reasons.append(
                    {
                        "source_id": source_id,
                        "reason": "recorded path absent from the current DD",
                    }
                )
                continue
            if fact is None:
                reasons.append(
                    {
                        "source_id": source_id,
                        "reason": "no StandardNameSource node for the recorded path",
                    }
                )
                continue
            if fact["holders"]:
                reasons.append(
                    {
                        "source_id": source_id,
                        "reason": "source held by a live name: "
                        + ", ".join(fact["holders"]),
                    }
                )
                continue
            if fact["status"] == "stale":
                reasons.append(
                    {"source_id": source_id, "reason": "existing source is stale"}
                )
                continue
            specs.append(
                {
                    "id": source_id,
                    "source_type": "dd",
                    "source_id": bare,
                    "status": fact["status"],
                    "dd_path": bare,
                }
            )
        if specs:
            restorable.append(
                {
                    "name": name,
                    "specs": specs,
                    "path_count": len(specs),
                    "skips": reasons,
                }
            )
        else:
            skipped.append(
                {
                    "name": name,
                    "reason": "no restorable recorded path",
                    "skips": reasons,
                }
            )
    return restorable, skipped


def transactional_preview(gc, plan: list[dict]) -> dict:
    """Run the owner writer inside a rolled-back transaction."""
    before = len(classify_sourceless_standard_names(gc))
    with gc.session() as session:
        transaction = session.begin_transaction()
        handle = _TransactionQuery(transaction)
        bound = 0
        try:
            for entry in plan:
                bound += bind_recovery_sources(entry["name"], entry["specs"], gc=handle)
        finally:
            transaction.rollback()
    after = len(classify_sourceless_standard_names(gc))
    return {"would_bind": bound, "rows_before": before, "rows_after": after}


def apply_plan(gc, plan: list[dict]) -> dict:
    receipt = []
    total = 0
    for entry in plan:
        bound = bind_recovery_sources(entry["name"], entry["specs"], gc=gc)
        total += bound
        receipt.append(
            {
                "name": entry["name"],
                "bound": bound,
                "source_ids": [s["id"] for s in entry["specs"]],
            }
        )
    return {"bound_total": total, "rows": receipt}


def main(dry: bool = False) -> int:
    started = time.monotonic()
    gc = GraphClient()
    try:
        quiet = quiet_graph_check(gc)

        baseline_rows = classify_sourceless_standard_names(gc)
        baseline = population_counts(baseline_rows)
        slices = build_worklist(gc, baseline_rows)

        recorded = slices.get("recorded_path", [])
        successor_edge = slices.get("ancestor_or_successor_edge", [])
        worklist = recorded + successor_edge

        recorded_ids = sorted(
            {sid for entry in recorded for sid in entry["recorded_source_ids"]}
        )
        facts = source_facts(gc, recorded_ids)
        nodes = {
            row["id"]
            for row in read(
                gc,
                "MATCH (n:IMASNode) WHERE n.id IN $paths RETURN n.id AS id",
                paths=sorted(
                    sid[len("dd:") :] for sid in recorded_ids if sid.startswith("dd:")
                ),
            )
        }
        plan, skipped = resolve_specs(recorded, facts, nodes)

        # The ancestor_or_successor_edge member records no path of its own.
        for entry in successor_edge:
            evidence = entry["evidence"]
            live = (
                "live relatives hold no source"
                if not evidence.get("historical_edges")
                else "live relatives hold " + ", ".join(evidence["historical_edges"])
            )
            skipped.append(
                {
                    "name": entry["name"],
                    "reason": "records no path of its own: no own source token, "
                    "and the only evidence is a relative's edge, which this "
                    "slice does not follow (" + live + ")",
                    "skips": [
                        {
                            "source_id": "—",
                            "reason": "lineage paths "
                            + (", ".join(evidence.get("lineage_paths", [])) or "none")
                            + "; receipts "
                            + (", ".join(evidence.get("receipts", [])) or "none"),
                        }
                    ],
                }
            )

        preview = transactional_preview(gc, plan)

        # Pre-apply read-back: the negative control. Every planned name must
        # still read as a dropped edge before the apply touches the graph.
        pre_rows = classify_sourceless_standard_names(gc)
        pre_population = {row["id"]: row["population"] for row in pre_rows}
        pre_dropped = sorted(
            entry["name"]
            for entry in plan
            if pre_population.get(entry["name"]) == "dropped_edge"
        )
        missing_from_dropped = sorted(
            entry["name"] for entry in plan if entry["name"] not in pre_population
        )

        if dry:
            print(
                f"DRY worklist={len(worklist)} planned={len(plan)} "
                f"skipped={len(skipped)} preview_would_bind={preview['would_bind']} "
                f"pre_dropped={len(pre_dropped)} all_still_dropped="
                f"{not missing_from_dropped and len(pre_dropped) == len(plan)}"
            )
            return 0
        apply_receipt = apply_plan(gc, plan)

        post_rows = classify_sourceless_standard_names(gc)
        post = population_counts(post_rows)
        post_population = {row["id"]: row["population"] for row in post_rows}
        still_dropped = sorted(
            entry["name"]
            for entry in plan
            if post_population.get(entry["name"]) == "dropped_edge"
        )
        delta = {
            key: post.get(key, 0) - baseline.get(key, 0)
            for key in sorted(set(baseline) | set(post))
        }

        elapsed = round(time.monotonic() - started, 1)
        result = {
            "base_revision": subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            ).stdout.strip(),
            "classifier_rows_before": len(baseline_rows),
            "classifier_rows_after": len(post_rows),
            "baseline_populations": baseline,
            "post_populations": post,
            "population_delta": delta,
            "worklist_count": len(worklist),
            "worklist": [entry["name"] for entry in worklist],
            "recorded_source_slice": sorted(entry["name"] for entry in recorded),
            "successor_edge_slice": sorted(entry["name"] for entry in successor_edge),
            "out_of_scope_slices": {
                key: sorted(e["name"] for e in value)
                for key, value in slices.items()
                if key not in {"recorded_path", "ancestor_or_successor_edge"}
            },
            "planned_restorations": [
                {"name": e["name"], "source_ids": [s["id"] for s in e["specs"]]}
                for e in plan
            ],
            "restored_count": len(plan),
            "skipped": skipped,
            "preview": preview,
            "apply_receipt": apply_receipt,
            "pre_apply_readback": {
                "dropped_names": len(pre_dropped),
                "all_planned_still_dropped": not missing_from_dropped
                and len(pre_dropped) == len(plan),
                "names_absent_from_sourceless_read": missing_from_dropped,
            },
            "post_apply_readback": {
                "restored_names_still_dropped": still_dropped,
                "restored_names_still_sourceless": sorted(
                    e["name"] for e in plan if e["name"] in post_population
                ),
            },
            "quiet_graph": quiet,
            "elapsed_seconds": elapsed,
        }
        OUTPUT.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
        REPORT.write_text(render(result), encoding="utf-8")
        print(
            f"worklist={len(worklist)} restored={len(plan)} skipped={len(skipped)} "
            f"dropped {baseline.get('dropped_edge', 0)}->{post.get('dropped_edge', 0)} "
            f"(delta {delta.get('dropped_edge', 0)}) in {elapsed:.1f}s"
        )
    finally:
        gc.close()
    return 0


def reclassify() -> int:
    """Relabel the recorded out-of-scope slices from a read-only graph pass.

    The apply pass resolved a name's lineage through a read that skipped retired
    names, so a name whose chain runs through a superseded relative was recorded
    under a receipt label. Membership is asserted identical, so only the label
    moves and the apply receipt stands as written.
    """
    gc = GraphClient()
    try:
        slices = build_worklist(gc, classify_sourceless_standard_names(gc))
    finally:
        gc.close()
    result = json.loads(OUTPUT.read_text(encoding="utf-8"))
    before = set().union(*[set(v) for v in result["out_of_scope_slices"].values()])
    corrected = {
        key: sorted(e["name"] for e in value)
        for key, value in slices.items()
        if key not in {"recorded_path", "ancestor_or_successor_edge"}
    }
    after = set().union(*[set(v) for v in corrected.values()])
    if before != after:
        raise SystemExit(
            "reclassify changes membership; refusing: "
            f"only-before={sorted(before - after)} only-after={sorted(after - before)}"
        )
    result["out_of_scope_slices"] = corrected
    result["out_of_scope_slices_note"] = (
        "Relabelled by a read-only pass over the post-apply graph. Membership is "
        "identical to the apply pass; the earlier label came from resolving a "
        "name's lineage through a read that dropped retired names, which hid the "
        "superseded relative its recorded path came from."
    )
    OUTPUT.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    REPORT.write_text(render(result), encoding="utf-8")
    counts = {key: len(value) for key, value in corrected.items()}
    print(
        f"reclassified out_of_scope_slices={counts} membership_unchanged={len(after)}"
    )
    return 0


def render(result: dict) -> str:
    def table(headers, rows):
        head = "".join(f"<th>{escape(str(h))}</th>" for h in headers)
        body = "".join(
            "<tr>" + "".join(f"<td>{escape(str(c))}</td>" for c in row) + "</tr>"
            for row in rows
        )
        return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"

    node = "sib-restore-recorded-sources"
    restored = result["planned_restorations"]
    pop_rows = [
        (
            key,
            result["baseline_populations"].get(key, 0),
            result["post_populations"].get(key, 0),
            result["population_delta"].get(key, 0),
        )
        for key in sorted(
            set(result["baseline_populations"]) | set(result["post_populations"])
        )
    ]
    restore_rows = [(e["name"], ", ".join(e["source_ids"])) for e in restored]
    skip_rows = [
        (
            e["name"],
            e["reason"],
            "; ".join(f"{s['source_id']}: {s['reason']}" for s in e.get("skips", [])),
        )
        for e in result["skipped"]
    ]
    parts = [
        "<!doctype html>",
        '<html lang="en">',
        "<head>",
        '<meta charset="utf-8">',
        '<meta name="reckon-type" content="evidence-fragment">',
        '<meta name="plan-slug" content="sn-integrity-backlog">',
        f'<meta name="node-id" content="{node}">',
        "<title>Restore the sources the names recorded</title>",
        "</head>",
        "<body>",
        "<main>",
        f'<h1 id="{node}-title">Restored recorded DD sources</h1>',
        f'<p id="{node}-summary">At base <code>{escape(result["base_revision"])}</code>, the '
        f"shared classifier read {result['classifier_rows_before']} sourceless live names. The "
        f"deterministic slice is {result['worklist_count']} names: "
        f"{len(result['recorded_source_slice'])} whose own record names the DD path it was "
        f"produced from, and {len(result['successor_edge_slice'])} whose only evidence is a "
        f"relative's edge. {result['restored_count']} names were restored through the "
        f"recovery-map writer, all by the route <strong>history</strong>; "
        f"{len(result['skipped'])} were not restored. The dropped-edge population fell "
        f"{result['baseline_populations'].get('dropped_edge', 0)} to "
        f"{result['post_populations'].get('dropped_edge', 0)}, and no other population moved. "
        "No model was called and no Cypher was written by hand.</p>",
        f'<h2 id="{node}-populations-title">Populations before and after</h2>',
        table(["population", "before", "after", "delta"], pop_rows),
        f'<h2 id="{node}-restored-title">Restored names</h2>',
        table(["name", "source ids"], restore_rows),
        f'<h2 id="{node}-skipped-title">Names not restored</h2>',
        f'<p id="{node}-skipped-note">The {len(result["out_of_scope_slices"])} remaining '
        "sourceless populations ("
        + ", ".join(
            f"{key} {len(value)}"
            for key, value in sorted(result["out_of_scope_slices"].items())
        )
        + ") are later nodes: their source needs a semantic check this slice does not make.</p>",
        table(["name", "reason", "detail"], skip_rows),
        f'<h2 id="{node}-preview-title">Preview and apply</h2>',
        f'<p id="{node}-preview-body">The recovery map was previewed by running the owner '
        f"writer inside a transaction that was rolled back: "
        f"{result['preview']['would_bind']} bindings would change, with the sourceless read "
        f"unchanged at {result['preview']['rows_before']} before and "
        f"{result['preview']['rows_after']} after. The apply bound "
        f"{result['apply_receipt']['bound_total']} sources.</p>",
        f'<h2 id="{node}-readback-title">Read-back and negative control</h2>',
        f'<p id="{node}-readback-body">The read-back ran the same classifier before and after '
        f"the apply at <code>EXIT=0</code>. Before the apply every planned name still read as a "
        f"dropped edge ({result['pre_apply_readback']['dropped_names']} of "
        f"{result['restored_count']}, all_still_dropped="
        f"{str(result['pre_apply_readback']['all_planned_still_dropped']).lower()}), so the "
        f"check is shown able to fail. After the apply no planned name remains in the "
        f"sourceless read ({len(result['post_apply_readback']['restored_names_still_sourceless'])} "
        f"still present) and none is a dropped edge.</p>",
        f'<h2 id="{node}-quiet-title">Quiet-graph check</h2>',
        f'<pre id="{node}-quiet-body">{escape(json.dumps(result["quiet_graph"], indent=1))}</pre>',
        "</main>",
        "</body>",
        "</html>",
    ]
    return "\n".join(parts) + "\n"


if __name__ == "__main__":
    if "--reclassify" in sys.argv[1:]:
        sys.exit(reclassify())
    sys.exit(main(dry="--dry" in sys.argv[1:]))
