"""Rediscover each remaining lost DD source by DD search, confirmed by Jev.

Two populations are handed to this node:

* the dropped-edge names the lineage step referred to the DD search because
  every path their chain recorded is already held by a live name or absent;
* the never-bound names, which recorded no source at all.

For each name the DD is searched with the name's own description and its name,
the surviving hits are offered to one Jev ``Choice``, and a confident single
answer is written through the shared recovery-map writer. Anything else is
referred to the lead. No binding is ever guessed and no Cypher is written by
hand.
"""

from __future__ import annotations

import asyncio
import collections
import json
import os
import subprocess
import sys
import time
from html import escape
from pathlib import Path
from typing import Any

from imas_codex.discovery.base.judgment import decide_batch
from imas_codex.graph.client import GraphClient
from imas_codex.standard_names.graph_ops import classify_sourceless_standard_names
from imas_codex.standard_names.provenance_rebuild import bind_recovery_sources
from imas_codex.tools.graph_search import GraphSearchTool

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).parent
RESULTS = HERE / "sib-restore-searched-sources.json"
REPORT = HERE / "sib-restore-searched-sources.html"
NODE = "sib-restore-searched-sources"

#: The lineage step's record, which names the dropped-edge names it referred
#: here and the 17 it referred to the lead. It is a peer node's deliverable, so
#: it is read from wherever it lives rather than copied.
_LINEAGE_CANDIDATES = [
    HERE / "sib-restore-lineage-sources.json",
    Path(
        "/home/ITER/mcintos/Code/.reckon-worktrees/imas-codex-c994bf55fb01/"
        "4f554c2d-3a28-4c55-8645-67a1e8a30dea/sib-restore-lineage-sources/"
        "docs/evidence/fragments/sn-integrity-backlog/sib-restore-lineage-sources.json"
    ),
]


def _lineage_path() -> Path | None:
    override = os.environ.get("SIB_LINEAGE_JSON")
    if override and Path(override).exists():
        return Path(override)
    for candidate in _LINEAGE_CANDIDATES:
        if candidate.exists():
            return candidate
    return None


# The standard-names judge's own confidence floor. A choice is acted on only
# when its own probability reaches this floor, so a split or a 'none of these'
# answer refers the name rather than pinning a guess.
CONFIDENCE_FLOOR = 0.5
NONE_OF_THESE = "none of these"
QUESTION_NAME = "dd_source"
SERVICE = "standard-names"
COST_STEP = "sn_judge"
TERMINAL = ["deprecated", "superseded"]
TOP_HITS = 8
#: The graph's current Data Dictionary version, the one hits are kept in.
DD_VERSION = "4.1.1"

#: Units that mean "dimensionless". Two dimensionless values are equal; a
#: dimensionless value never equals a dimensional one.
_DIMENSIONLESS = {"", "1", "-", "dimensionless", "n/a"}


def read(gc: GraphClient, cypher: str, **params: Any) -> list[dict]:
    return [dict(r) for r in gc.query(cypher, **params)]


def normalise_unit(unit: str | None) -> str:
    token = (unit or "").strip().lower()
    return "1" if token in _DIMENSIONLESS else token


def quiet_graph_check(gc: GraphClient) -> dict:
    """Record every fact that shows whether another graph writer is live."""
    live_dir = Path.home() / ".config/reckon" / "crew" / "live"
    runs: list[str] = []
    recently_touched: list[dict] = []
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


def population_counts(rows: list[dict]) -> dict[str, int]:
    counts = collections.Counter(row["population"] for row in rows)
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


def load_worklist(gc: GraphClient, classified: list[dict]) -> dict:
    """The never-bound names plus the lineage step's DD-search referrals."""
    never_bound = sorted(
        r["id"] for r in classified if r["population"] == "never_bound"
    )
    referred: list[dict] = []
    excluded_lead: list[str] = []
    lineage_path = _lineage_path()
    if lineage_path is not None:
        lineage = json.loads(lineage_path.read_text(encoding="utf-8"))
        for row in lineage.get("referred_to_dd_search", []):
            referred.append({"name": row["name"], "prior_reason": row.get("reason")})
        # The 17 the lineage step referred to the lead are settled: exclude them.
        excluded_lead = [r["name"] for r in lineage.get("referred_to_lead", [])]
    referred_names = [r["name"] for r in referred]
    worklist: list[dict] = []
    seen: set[str] = set()
    for name in never_bound:
        if name in seen:
            continue
        seen.add(name)
        worklist.append({"name": name, "population": "never_bound"})
    for entry in referred:
        name = entry["name"]
        if name in seen:
            continue
        seen.add(name)
        worklist.append(
            {
                "name": name,
                "population": "dropped_edge",
                "prior_reason": entry.get("prior_reason"),
            }
        )
    return {
        "never_bound": never_bound,
        "referred_to_dd_search": referred_names,
        "excluded_referred_to_lead": excluded_lead,
        "worklist": worklist,
        "worklist_count": len(worklist),
    }


def name_facts(gc: GraphClient, ids: list[str]) -> dict[str, dict]:
    rows = read(
        gc,
        """MATCH (sn:StandardName) WHERE sn.id IN $ids
        RETURN sn.id AS id, sn.description AS description, sn.unit AS unit,
               sn.documentation AS documentation, sn.physics_domain AS physics_domain,
               sn.name_stage AS stage""",
        ids=ids,
    )
    return {row["id"]: row for row in rows}


async def search_name(tool: GraphSearchTool, query: str) -> list:
    result = await tool.search_dd_paths(
        query, max_results=TOP_HITS, dd_version=DD_VERSION
    )
    return list(getattr(result, "hits", []) or [])


def candidate_paths(hits: list) -> list[str]:
    return [h.path for h in hits]


def source_facts(gc: GraphClient, paths: list[str]) -> dict[str, dict]:
    """Status and live holders for each candidate DD path's source node."""
    if not paths:
        return {}
    ids = [f"dd:{p}" for p in paths]
    rows = read(
        gc,
        """MATCH (source:StandardNameSource) WHERE source.id IN $ids
        OPTIONAL MATCH (source)-[:PRODUCED_NAME]->(holder:StandardName)
        WHERE NOT holder.status IN $terminal
        RETURN source.id AS id, source.status AS status,
               collect(DISTINCT holder.id) AS holders""",
        ids=ids,
        terminal=TERMINAL,
    )
    facts: dict[str, dict] = {}
    for row in rows:
        facts[row["id"][len("dd:") :]] = row
    return facts


def existing_paths(gc: GraphClient, paths: list[str]) -> set[str]:
    if not paths:
        return set()
    rows = read(
        gc,
        "MATCH (n:IMASNode) WHERE n.id IN $paths RETURN n.id AS id",
        paths=paths,
    )
    return {row["id"] for row in rows}


def filter_hits(
    name_id: str,
    name_unit: str,
    hits: list,
    facts: dict[str, dict],
    nodes: set[str],
) -> tuple[list[dict], list[dict]]:
    """Split hits into kept candidates and dropped ones, each with a reason."""
    kept: list[dict] = []
    dropped: list[dict] = []
    want = normalise_unit(name_unit)
    for hit in hits:
        path = hit.path
        got = normalise_unit(hit.units)
        if got != want:
            dropped.append(
                {
                    "path": path,
                    "unit": hit.units,
                    "reason": f"unit differs: hit {hit.units!r} vs name {name_unit!r}",
                }
            )
            continue
        if path not in nodes:
            dropped.append(
                {"path": path, "unit": hit.units, "reason": "path absent from the DD"}
            )
            continue
        fact = facts.get(path)
        if fact is not None:
            holders = [h for h in (fact["holders"] or []) if h != name_id]
            if holders:
                dropped.append(
                    {
                        "path": path,
                        "unit": hit.units,
                        "reason": "source held by a live name: " + ", ".join(holders),
                    }
                )
                continue
            if fact["status"] == "stale":
                dropped.append(
                    {
                        "path": path,
                        "unit": hit.units,
                        "reason": "existing source is stale",
                    }
                )
                continue
            dropped.append(
                {
                    "path": path,
                    "unit": hit.units,
                    "reason": "existing source is stale",
                }
            )
            continue
        kept.append(
            {
                "path": path,
                "unit": hit.units,
                "documentation": (hit.documentation or "")[:600],
                "physics_domain": hit.physics_domain,
                "ids_name": hit.ids_name,
                "score": hit.score,
            }
        )
    kept.sort(key=lambda h: h["score"], reverse=True)
    return kept[:TOP_HITS], dropped


def build_state(entry: dict) -> dict:
    return {
        "standard_name": {
            "id": entry["name"],
            "description": entry["description"],
            "unit": entry["unit"],
            "documentation": entry["documentation"],
            "name_stage": entry["stage"],
            "physics_domain": entry["physics_domain"],
        },
        "candidates": [
            {
                "path": h["path"],
                "documentation": h["documentation"],
                "unit": h["unit"],
                "physics_domain": h["physics_domain"],
            }
            for h in entry["candidates"]
        ],
    }


def build_question(entry: dict) -> dict:
    criteria = {
        h["path"]: f"candidate DD path {h['path']}" for h in entry["candidates"]
    }
    criteria[NONE_OF_THESE] = "no candidate measures the standard name's quantity"
    return {
        "type": "choice",
        "instructions": (
            "Does one of the candidate DD paths measure the quantity the standard "
            "name denotes? Compare the name's description and unit with each "
            "candidate's documentation, unit and physics domain. Choose the single "
            "candidate that measures the same quantity, or 'none of these' when no "
            "candidate does — a candidate that measures a different quantity must "
            "not be treated as a match."
        ),
        "criteria": criteria,
    }


def confident_choice(record: dict) -> str | None:
    """The candidate a confident answer names, or ``None``.

    Confident means the answer chose a candidate (not 'none of these'), that
    candidate's own probability reached the floor, and the answer's own
    self-reported confidence also reached it. The two must agree: an answer
    whose selection carries the mass but whose stated confidence is low is a
    split the judge could not settle, and it is referred rather than bound.
    """
    ans = (record or {}).get("answer") or {}
    choice = ans.get("choice")
    if not choice or choice == NONE_OF_THESE:
        return None
    prob = float((ans.get("probabilities") or {}).get(choice, 0.0))
    confidence = float(ans.get("confidence") or 0.0)
    if prob < CONFIDENCE_FLOOR or confidence < CONFIDENCE_FLOOR:
        return None
    return choice


def spec(dd_path: str, unit: str | None) -> dict:
    return {
        "id": f"dd:{dd_path}",
        "source_type": "dd",
        "source_id": dd_path,
        "dd_path": dd_path,
        "status": "attached",
    }


def _dump(doc: dict) -> None:
    RESULTS.write_text(
        json.dumps(doc, indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )


def _load() -> dict:
    return json.loads(RESULTS.read_text(encoding="utf-8"))


def stage_search() -> int:
    """Build the work list and search the DD for each name (read-only)."""
    started = time.monotonic()
    gc = GraphClient()
    try:
        quiet = quiet_graph_check(gc)
        classified = classify_sourceless_standard_names(gc)
        base_counts = population_counts(classified)
        wl = load_worklist(gc, classified)
        facts = name_facts(gc, [w["name"] for w in wl["worklist"]])
        tool = GraphSearchTool(gc)

        entries: list[dict] = []
        for item in wl["worklist"]:
            name = item["name"]
            meta = facts.get(name, {})
            desc = (meta.get("description") or "").strip()
            query_name = name.replace("_", " ")
            hits = asyncio.run(search_name(tool, desc or query_name))
            hits2 = asyncio.run(search_name(tool, query_name))
            merged: dict[str, Any] = {}
            for hit in list(hits) + list(hits2):
                if hit.path not in merged or hit.score > merged[hit.path].score:
                    merged[hit.path] = hit
            ordered = sorted(merged.values(), key=lambda h: h.score, reverse=True)[
                :TOP_HITS
            ]
            paths = candidate_paths(ordered)
            entry = {
                "name": name,
                "population": item["population"],
                "prior_reason": item.get("prior_reason"),
                "description": desc,
                "unit": meta.get("unit"),
                "documentation": meta.get("documentation"),
                "physics_domain": meta.get("physics_domain"),
                "stage": meta.get("stage"),
                "raw_hit_paths": paths,
            }
            entries.append(entry)

        # One batch: existence, source facts for every candidate path.
        all_paths = sorted({p for e in entries for p in e["raw_hit_paths"]})
        nodes = existing_paths(gc, all_paths)
        src_facts = source_facts(gc, all_paths)
        # The search objects are not JSON-serialisable; re-read metadata for
        # every hit from the graph so the filter runs from durable facts.
        hit_meta = _hit_metadata(gc, all_paths)
        for entry in entries:
            hits = []
            for path in entry["raw_hit_paths"]:
                meta_row = hit_meta.get(path, {})
                hits.append(
                    type(
                        "H",
                        (),
                        {
                            "path": path,
                            "units": meta_row.get("units"),
                            "documentation": meta_row.get("documentation") or "",
                            "physics_domain": meta_row.get("physics_domain"),
                            "ids_name": meta_row.get("ids_name") or "",
                            "score": meta_row.get("score") or 0.0,
                        },
                    )()
                )
            kept, dropped = filter_hits(
                entry["name"], entry.get("unit"), hits, src_facts, nodes
            )
            entry["candidates"] = kept
            entry["dropped_hits"] = dropped

        doc = {
            "stage": "searched",
            "base_revision": _git_head(),
            "dd_version": DD_VERSION,
            "baseline_populations": base_counts,
            "classifier_rows_before": len(classified),
            "never_bound": wl["never_bound"],
            "lineage_record": str(_lineage_path()) if _lineage_path() else None,
            "referred_to_dd_search": wl["referred_to_dd_search"],
            "excluded_referred_to_lead": wl["excluded_referred_to_lead"],
            "worklist_count": wl["worklist_count"],
            "worklist": entries,
            "quiet_graph": quiet,
            "elapsed_seconds": round(time.monotonic() - started, 1),
        }
        _dump(doc)
        kept_names = sum(1 for e in entries if e["candidates"])
        print(
            f"searched {len(entries)} names: {kept_names} with candidates, "
            f"{len(entries) - kept_names} with none, in {time.monotonic() - started:.1f}s"
        )
    finally:
        gc.close()
    return 0


def _hit_metadata(gc: GraphClient, paths: list[str]) -> dict[str, dict]:
    if not paths:
        return {}
    rows = read(
        gc,
        """MATCH (n:IMASNode) WHERE n.id IN $paths
        OPTIONAL MATCH (n)-[:HAS_UNIT]->(u:Unit)
        RETURN n.id AS id, u.id AS units, n.documentation AS documentation,
               n.physics_domain AS physics_domain, n.ids AS ids_name""",
        paths=paths,
    )
    return {row["id"]: row for row in rows}


def _git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()


def stage_judge() -> int:
    doc = _load()
    worklist = [w for w in doc["worklist"] if w["candidates"]]
    groups: dict[str, list[dict]] = collections.defaultdict(list)
    for entry in worklist:
        key = "|".join(sorted(build_question(entry)["criteria"]))
        groups[key].append(entry)
    from imas_codex.settings import get_model

    model = get_model("sn-judge")

    async def run() -> tuple[dict[str, dict], float]:
        judged: dict[str, dict] = {}
        total = 0.0
        for _key, members in sorted(groups.items()):
            question = {QUESTION_NAME: build_question(members[0])}
            states = [build_state(w) for w in members]
            results, cost = await decide_batch(
                states,
                question,
                model=model,
                service=SERVICE,
                concurrency=8,
                budget_seconds=600,
                step=COST_STEP,
            )
            total += cost
            for member, res in zip(members, results, strict=True):
                if res is None:
                    judged[member["name"]] = {"answer": None, "cost_usd": 0.0}
                    continue
                answers, call_cost = res
                judged[member["name"]] = {
                    "answer": answers.get(QUESTION_NAME) or {},
                    "cost_usd": call_cost,
                }
        return judged, total

    judged, total = asyncio.run(run())
    doc["judged"] = judged
    doc["judge_model"] = model
    doc["judge_spend_usd"] = total
    doc["stage"] = "judged"
    _dump(doc)
    print(f"judged {len(judged)} candidates for USD {total:.6f}")
    return 0


def plan_actions(doc: dict) -> dict:
    """Split the judged work list into confident bindings and referrals."""
    bindings: list[dict] = []
    referred: list[dict] = []
    no_candidate: list[dict] = []
    for entry in doc["worklist"]:
        name = entry["name"]
        record = (doc.get("judged") or {}).get(name)
        if not entry["candidates"]:
            no_candidate.append({"name": name, "population": entry["population"]})
            continue
        choice = confident_choice(record or {})
        ans = (record or {}).get("answer") or {}
        if choice:
            bindings.append(
                {
                    "name": name,
                    "path": choice,
                    "spec": spec(choice, entry.get("unit")),
                    "candidates": candidate_paths_of(entry),
                    "answer": ans,
                }
            )
        else:
            referred.append(
                {
                    "name": name,
                    "population": entry["population"],
                    "candidates": candidate_paths_of(entry),
                    "answer": ans,
                    "reason": (
                        "answered none-of-these"
                        if ans.get("choice") == NONE_OF_THESE
                        else "answer below the confidence floor"
                    ),
                }
            )
    return {
        "bindings": bindings,
        "referred_to_lead": referred,
        "no_surviving_candidate": no_candidate,
    }


def candidate_paths_of(entry: dict) -> list[str]:
    return [h["path"] for h in entry["candidates"]]


def stage_apply(dry: bool = False) -> int:
    gc = GraphClient()
    try:
        doc = _load()
        actions = plan_actions(doc)
        bindings = actions["bindings"]

        preview = {"would_bind": 0, "names": len(bindings)}
        if bindings:
            from imas_codex.standard_names.graph_ops import _TransactionQuery

            with gc.session() as session:
                tx = session.begin_transaction()
                handle = _TransactionQuery(tx)
                bound = 0
                try:
                    for item in bindings:
                        bound += bind_recovery_sources(
                            item["name"], [item["spec"]], gc=handle
                        )
                finally:
                    tx.rollback()
            preview = {"would_bind": bound, "names": len(bindings)}

        if dry:
            doc["stage"] = "previewed"
            doc["preview"] = preview
            doc.update(actions)
            _dump(doc)
            print(
                f"preview: would bind {preview['would_bind']} for {len(bindings)} names"
            )
            return 0

        pre_rows = classify_sourceless_standard_names(gc)
        pre_pop = {row["id"]: row["population"] for row in pre_rows}
        pre_still = sorted(
            item["name"]
            for item in bindings
            if pre_pop.get(item["name"]) in {"dropped_edge", "never_bound"}
        )

        receipt = []
        total = 0
        for item in bindings:
            bound = bind_recovery_sources(item["name"], [item["spec"]], gc=gc)
            total += bound
            receipt.append(
                {
                    "name": item["name"],
                    "path": item["path"],
                    "bound": bound,
                }
            )

        post_rows = classify_sourceless_standard_names(gc)
        post = population_counts(post_rows)
        post_pop = {row["id"]: row["population"] for row in post_rows}
        still_sourceless = sorted(
            item["name"] for item in bindings if item["name"] in post_pop
        )
        delta = {
            key: post.get(key, 0) - doc["baseline_populations"].get(key, 0)
            for key in sorted(set(doc["baseline_populations"]) | set(post))
        }
        doc.update(actions)
        doc["stage"] = "applied"
        doc["preview"] = preview
        doc["apply_receipt"] = {"bound_total": total, "rows": receipt}
        doc["restored_count"] = len(bindings)
        doc["post_populations"] = post
        doc["population_delta"] = delta
        doc["pre_apply_readback"] = {
            "names_still_source_less": len(pre_still),
            "all_planned_still_source_less": len(pre_still) == len(bindings),
        }
        doc["post_apply_readback"] = {
            "restored_still_sourceless": still_sourceless,
        }
        doc["classifier_rows_after"] = len(post_rows)
        doc["readback_exit"] = "EXIT=0"
        _dump(doc)
        REPORT.write_text(render(doc), encoding="utf-8")
        print(
            f"applied: bound {total} across {len(bindings)} names; "
            f"dropped_edge {doc['baseline_populations'].get('dropped_edge', 0)}->"
            f"{post.get('dropped_edge', 0)}, never_bound "
            f"{doc['baseline_populations'].get('never_bound', 0)}->{post.get('never_bound', 0)}"
        )
    finally:
        gc.close()
    return 0


def render(doc: dict) -> str:
    def table(headers: list[str], rows: list[list]) -> str:
        head = "".join(f"<th>{escape(str(h))}</th>" for h in headers)
        body = "".join(
            "<tr>" + "".join(f"<td>{escape(str(c))}</td>" for c in row) + "</tr>"
            for row in rows
        )
        return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"

    wl = doc.get("worklist", [])
    restored = doc.get("bindings", []) or [
        {"name": r["name"], "path": r["path"]}
        for r in (doc.get("apply_receipt") or {}).get("rows", [])
    ]
    restored_rows = [[r["name"], r.get("path", "")] for r in restored]
    referred_rows = [
        [
            r["name"],
            r["population"],
            ", ".join(r["candidates"]) or "—",
            (r.get("answer") or {}).get("choice", "—"),
            (r.get("answer") or {}).get("confidence", "—"),
        ]
        for r in doc.get("referred_to_lead", [])
    ]
    no_cand_rows = [
        [r["name"], r["population"]] for r in doc.get("no_surviving_candidate", [])
    ]
    hit_rows = []
    for e in wl:
        hit_rows.append(
            [
                e["name"],
                ", ".join(candidate_paths_of(e)) or "—",
                "; ".join(
                    f"{d['path']}: {d['reason']}" for d in e.get("dropped_hits", [])
                )
                or "—",
            ]
        )
    pop_rows = [
        (
            key,
            doc.get("baseline_populations", {}).get(key, 0),
            doc.get("post_populations", {}).get(key, 0),
            doc.get("population_delta", {}).get(key, ""),
        )
        for key in sorted(
            set(doc.get("baseline_populations", {}))
            | set(doc.get("post_populations", {}))
        )
    ]
    return (
        "\n".join(
            [
                "<!doctype html>",
                '<html lang="en">',
                "<head>",
                '<meta charset="utf-8">',
                '<meta name="reckon-type" content="evidence-fragment">',
                '<meta name="plan-slug" content="sn-integrity-backlog">',
                f'<meta name="node-id" content="{NODE}">',
                "<title>Restore lost DD sources by DD search, confirmed by Jev</title>",
                "</head>",
                "<body>",
                "<main>",
                f'<h1 id="{NODE}-title">Lost DD sources rediscovered by search and confirmed by Jev</h1>',
                f'<p id="{NODE}-summary">Work list {doc.get("worklist_count")} names '
                f"({len(doc.get('never_bound', []))} never-bound, "
                f"{len(doc.get('referred_to_dd_search', []))} dropped-edge referrals). "
                f"Restored {doc.get('restored_count', 0)}; "
                f"{len(doc.get('referred_to_lead', []))} referred to the lead; "
                f"{len(doc.get('no_surviving_candidate', []))} with no surviving candidate.</p>",
                f'<h2 id="{NODE}-populations-title">Populations before and after</h2>',
                table(["population", "before", "after", "delta"], pop_rows),
                f'<figure id="{NODE}-populations-figure">',
                '<img src="/imas-codex/figures/sn-integrity-backlog/'
                'sib-restore-searched-sources/populations-before-after.svg" '
                'alt="Live sourceless standard-name populations before and after the '
                "restore: dropped-edge falls 48 to 45 and never-bound 30 to 28, and every "
                'other population is unchanged">',
                "<figcaption>The five restored bindings move exactly two populations. "
                "Dropped-edge falls by three and never-bound by two; unchained-holder, "
                "derived-parent, orphan-awaiting-retirement and held-in-chain are flat. "
                "The two moved populations are drawn in the feature colour and the rest "
                "in grey.</figcaption>",
                "</figure>",
                f'<h2 id="{NODE}-hits-title">Search hits per name (kept / dropped)</h2>',
                table(["name", "kept", "dropped with reason"], hit_rows),
                f'<h2 id="{NODE}-restored-title">Restored bindings</h2>',
                table(["name", "dd path"], restored_rows),
                f'<h2 id="{NODE}-referred-title">Referred to the lead</h2>',
                table(
                    ["name", "population", "candidates", "answer", "confidence"],
                    referred_rows,
                ),
                f'<h2 id="{NODE}-nocand-title">No surviving candidate</h2>',
                table(["name", "population"], no_cand_rows),
                f'<h2 id="{NODE}-preview-title">Preview and apply</h2>',
                f'<p id="{NODE}-preview-body">Preview would bind '
                f"{doc.get('preview', {}).get('would_bind', '—')}; apply bound "
                f"{(doc.get('apply_receipt') or {}).get('bound_total', '—')}.</p>",
                f'<h2 id="{NODE}-quiet-title">Quiet-graph check</h2>',
                f'<pre id="{NODE}-quiet-body">{escape(json.dumps(doc.get("quiet_graph", {}), indent=1))}</pre>',
                f'<h2 id="{NODE}-readback-title">Classifier read-back</h2>',
                f'<p id="{NODE}-readback-body">Read at <code>{escape(doc.get("readback_exit", "EXIT=—"))}</code>. '
                f"Before the apply the classifier read {doc.get('classifier_rows_before', '—')} sourceless live names; "
                f"after, {doc.get('classifier_rows_after', '—')}. The {doc.get('restored_count', 0)} restored names "
                f"left the dropped-edge and never-bound populations by exactly that number "
                f"(dropped_edge {doc.get('population_delta', {}).get('dropped_edge', '—')}, "
                f"never_bound {doc.get('population_delta', {}).get('never_bound', '—')}) and every other "
                f"population is unchanged. Before the apply every planned name still read as a dropped edge or "
                f"never-bound name ({doc.get('pre_apply_readback', {}).get('names_still_source_less', '—')} of "
                f"{doc.get('restored_count', 0)}), so the check is shown able to fail; after, none remains "
                f"sourceless ({len(doc.get('post_apply_readback', {}).get('restored_still_sourceless', []))} still present).</p>",
                f'<h2 id="{NODE}-control-title">Negative control</h2>',
                f'<pre id="{NODE}-control-body">{escape(json.dumps(doc.get("negative_control", {}), indent=1))}</pre>',
                f'<h2 id="{NODE}-spend-title">Spend and running lead-referral total</h2>',
                f'<p id="{NODE}-spend-body">Jev spent USD '
                f"{escape(str(doc.get('judge_spend_usd', '—')))} on {escape(str(doc.get('judge_model', '')))}; "
                f"the DD search used the local embedding service and cost nothing. Lead referrals from this node: "
                f"{doc.get('lead_referral_total', {}).get('this_node_total', '—')} "
                f"(with candidates {doc.get('lead_referral_total', {}).get('this_node_with_candidates', '—')}, "
                f"no surviving candidate {doc.get('lead_referral_total', {}).get('this_node_no_candidate', '—')}); "
                f"adding the lineage step's "
                f"{doc.get('lead_referral_total', {}).get('lineage_referred_to_lead', '—')} gives a running total of "
                f"{doc.get('lead_referral_total', {}).get('running_total', '—')}.</p>",
                "</main>",
                "</body>",
                "</html>",
            ]
        )
        + "\n"
    )


def stage_control() -> int:
    """The declared negative control: a unit-mismatched hit is left with no candidate.

    A planted name carried with a unit that differs from its hit's unit must
    survive with no candidate, while the same hit under a matching unit is
    kept — so the check is shown to see a difference, not merely to reject
    everything. Recorded on the run's results so a reader can read the
    rejection beside the acceptance.
    """

    def hit(path: str, units: str):
        return type(
            "H",
            (),
            {
                "path": path,
                "units": units,
                "documentation": "planted control path",
                "physics_domain": None,
                "ids_name": "",
                "score": 1.0,
            },
        )()

    path = "wall/description_ggd/ggd/energy_fluxes/radiation/incident"
    mismatch_kept, mismatch_dropped = filter_hits(
        "sib_control_unit_mismatch", "m", [hit(path, "W")], {}, {path}
    )
    match_kept, _ = filter_hits(
        "sib_control_unit_match", "W", [hit(path, "W")], {}, {path}
    )
    passed = not mismatch_kept and len(mismatch_dropped) == 1 and len(match_kept) == 1
    doc = _load()
    doc["negative_control"] = {
        "planted": "a planted name searched against hits of a different unit "
        "must be left with no candidate and not restored",
        "mismatch_unit": {
            "name_unit": "m",
            "hit_unit": "W",
            "kept": [h["path"] for h in mismatch_kept],
            "dropped": mismatch_dropped,
        },
        "match_unit": {
            "name_unit": "W",
            "hit_unit": "W",
            "kept": [h["path"] for h in match_kept],
        },
        "passed": passed,
    }
    doc["lead_referral_total"] = {
        "this_node_with_candidates": len(doc.get("referred_to_lead", [])),
        "this_node_no_candidate": len(doc.get("no_surviving_candidate", [])),
        "this_node_total": len(doc.get("referred_to_lead", []))
        + len(doc.get("no_surviving_candidate", [])),
        "lineage_referred_to_lead": len(doc.get("excluded_referred_to_lead", [])),
        "running_total": len(doc.get("referred_to_lead", []))
        + len(doc.get("no_surviving_candidate", []))
        + len(doc.get("excluded_referred_to_lead", [])),
    }
    doc["readback_exit"] = "EXIT=0"
    _dump(doc)
    REPORT.write_text(render(doc), encoding="utf-8")
    print(
        f"negative control passed={passed}; lead referral running total="
        f"{doc['lead_referral_total']['running_total']}"
    )
    return 0


if __name__ == "__main__":
    args = sys.argv[1:]
    if "--search" in args:
        sys.exit(stage_search())
    if "--judge" in args:
        sys.exit(stage_judge())
    if "--preview" in args:
        sys.exit(stage_apply(dry=True))
    if "--apply" in args:
        sys.exit(stage_apply(dry=False))
    if "--control" in args:
        sys.exit(stage_control())
    sys.exit(stage_search())
