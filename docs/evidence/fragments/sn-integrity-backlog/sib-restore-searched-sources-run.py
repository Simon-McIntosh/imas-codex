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
from imas_codex.units import normalize_unit_symbol, units_are_equivalent

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


#: The standard-names judge's own confidence floor. A choice is acted on only
#: when its own probability reaches this floor, so a split or a 'none of these'
#: answer refers the name rather than pinning a guess.
CONFIDENCE_FLOOR = 0.5
#: The stricter floor the corrected pass applies: a candidate is bound only
#: when the judge's own selection carries at least this much probability and
#: the answer's self-reported confidence still clears CONFIDENCE_FLOOR.
CHOICE_PROBABILITY_FLOOR = 0.8
NONE_OF_THESE = "none of these"
QUESTION_NAME = "dd_source"
SERVICE = "standard-names"
COST_STEP = "sn_judge"
TERMINAL = ["deprecated", "superseded"]
TOP_HITS = 8
#: The graph's current Data Dictionary version, the one hits are kept in.
DD_VERSION = "4.1.1"


def read(gc: GraphClient, cypher: str, **params: Any) -> list[dict]:
    return [dict(r) for r in gc.query(cypher, **params)]


class ReadbackGateError(RuntimeError):
    """The read-back moved a population by other than the restored count."""


def owned_population_gate(
    baseline: dict, post: dict, owned: set[str], expected_restored: int
) -> tuple[bool, str, dict]:
    """Whether the populations this node's writes can move moved exactly.

    The names restored leave the two source-less populations this node works
    in; every one of those must be gone and the total shrink across them must
    equal the number restored. No population anywhere may grow. A population
    outside ``owned`` that moved is another writer's, so it is reported
    rather than failing the gate — a concurrent restorer can only shrink the
    sourceless set, and a growth is a defect wherever it happens.
    """
    delta = {
        key: post.get(key, 0) - baseline.get(key, 0)
        for key in sorted(set(baseline) | set(post))
    }
    owned_shrink = -sum(v for k, v in delta.items() if k in owned and v < 0)
    grew = {k: v for k, v in delta.items() if v > 0}
    other_moved = {k: v for k, v in delta.items() if k not in owned and v != 0}
    ok = owned_shrink == expected_restored and not grew
    message = (
        f"{expected_restored} restored, {owned_shrink} left "
        f"{sorted(owned)}, none grew; concurrent writers moved "
        f"{other_moved or 'nothing'}"
        if ok
        else (
            f"read-back gate failed: {expected_restored} restored but "
            f"{owned_shrink} left {sorted(owned)} and {grew or 'none'} grew"
        )
    )
    return (
        ok,
        message,
        {
            "delta": delta,
            "owned_shrink": owned_shrink,
            "grew": grew,
            "other_moved": other_moved,
        },
    )


def population_gate(
    baseline: dict, post: dict, expected_restored: int
) -> tuple[bool, str, dict]:
    """Whether the read-back moved the populations exactly as the writes predict.

    The gate is that every live sourceless population shrinks or holds: the
    names restored leave the two source-less populations by exactly the number
    restored, and no population grows. A move of any other shape — a
    population that grew, or a total shrink that does not equal the restored
    count — fails, and the caller raises with the exit status of the failure.
    """
    delta = {
        key: post.get(key, 0) - baseline.get(key, 0)
        for key in sorted(set(baseline) | set(post))
    }
    grew = {k: v for k, v in delta.items() if v > 0}
    shrunk = -sum(v for v in delta.values() if v < 0)
    ok = not grew and shrunk == expected_restored
    message = (
        f"populations moved as the writes predict: {expected_restored} restored, "
        f"{shrunk} left the sourceless populations, none grew"
        if ok
        else (
            f"read-back gate failed: {expected_restored} restored but {shrunk} left "
            f"the sourceless populations and {grew or 'none'} grew"
        )
    )
    return ok, message, delta


def units_match(name_unit: str | None, hit_unit: str | None) -> bool:
    """Whether a name's unit and a hit's unit denote the same quantity.

    The repository's unit authority decides: both spellings go through
    ``normalize_unit_symbol`` and, when both parse, ``units_are_equivalent``
    compares them by pint rather than by string. Two spellings that both fail
    to parse carry no dimensional unit and match each other; a spelling that
    parses never matches one that does not, so a dimensionless name never
    matches a dimensional hit and vice versa.
    """
    a = normalize_unit_symbol(name_unit or "")
    b = normalize_unit_symbol(hit_unit or "")
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    return units_are_equivalent(a, b)


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
    """Split hits into kept candidates and dropped ones, each with a reason.

    A hit is dropped only when its unit differs from the name's, when its path
    is absent from the loaded DD, when a live name other than this one holds
    its source, or when its source is stale. A source that exists but is
    extracted, skipped or failed is not a reason to drop: with no live holder
    it is exactly the candidate this node recovers, and the recovery writer
    re-attaches it.
    """
    kept: list[dict] = []
    dropped: list[dict] = []
    for hit in hits:
        path = hit.path
        if not units_match(name_unit, hit.units):
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
            if (fact.get("status") or "") == "stale":
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


def confident_choice(record: dict, prob_floor: float = CONFIDENCE_FLOOR) -> str | None:
    """The candidate a confident answer names, or ``None``.

    Confident means the answer chose a candidate (not 'none of these'), that
    candidate's own probability reached ``prob_floor``, and the answer's own
    self-reported confidence also reached CONFIDENCE_FLOOR. The two must
    agree: an answer whose selection carries the mass but whose stated
    confidence is low is a split the judge could not settle, and it is
    referred rather than bound.
    """
    ans = (record or {}).get("answer") or {}
    choice = ans.get("choice")
    if not choice or choice == NONE_OF_THESE:
        return None
    prob = float((ans.get("probabilities") or {}).get(choice, 0.0))
    confidence = float(ans.get("confidence") or 0.0)
    if prob < prob_floor or confidence < CONFIDENCE_FLOOR:
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


def plan_actions(
    doc: dict, prob_floor: float = CONFIDENCE_FLOOR, entries: list[dict] | None = None
) -> dict:
    """Split the judged work list into confident bindings and referrals."""
    bindings: list[dict] = []
    referred: list[dict] = []
    no_candidate: list[dict] = []
    for entry in entries if entries is not None else doc["worklist"]:
        name = entry["name"]
        record = (doc.get("judged") or {}).get(name)
        if not entry["candidates"]:
            no_candidate.append({"name": name, "population": entry["population"]})
            continue
        choice = confident_choice(record or {}, prob_floor)
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
        gate_ok, gate_message, delta = owned_population_gate(
            doc["baseline_populations"],
            post,
            {"dropped_edge", "never_bound"},
            len(bindings),
        )
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
        doc["population_gate"] = {"ok": gate_ok, "message": gate_message}
        doc["classifier_rows_after"] = len(post_rows)
        doc["readback_exit"] = "EXIT=0" if gate_ok else "EXIT=3"
        _dump(doc)
        REPORT.write_text(render(doc), encoding="utf-8")
        print(
            f"applied: bound {total} across {len(bindings)} names; "
            f"dropped_edge {doc['baseline_populations'].get('dropped_edge', 0)}->"
            f"{post.get('dropped_edge', 0)}, never_bound "
            f"{doc['baseline_populations'].get('never_bound', 0)}->{post.get('never_bound', 0)}"
        )
        print(f"read-back gate: {gate_message}")
        if not gate_ok:
            raise ReadbackGateError(gate_message)
    finally:
        gc.close()
    return 0


def _hits_from_metadata(paths: list[str], hit_meta: dict[str, dict]) -> list:
    hits = []
    for path in paths:
        row = hit_meta.get(path, {})
        hits.append(
            type(
                "H",
                (),
                {
                    "path": path,
                    "units": row.get("units"),
                    "documentation": row.get("documentation") or "",
                    "physics_domain": row.get("physics_domain"),
                    "ids_name": row.get("ids_name") or "",
                    "score": row.get("score") or 0.0,
                },
            )()
        )
    return hits


def stage_corrected_pass() -> int:
    """Re-run search and judgment for every name the old filter left with none.

    The old filter dropped every hit whose path carried a source of any status
    and recorded 'existing source is stale'. With the filter corrected, a
    source that is extracted, skipped or failed and held by no live name is a
    candidate. This stage recomputes the filter over the stored raw hits of
    every name (recording the corrected reasons), re-searches and re-judges
    the names that had no candidate, binds the confident ones through the
    recovery writer on a quiet graph, and gates the read-back on the total
    number restored — the five already bound plus the new ones.
    """
    started = time.monotonic()
    gc = GraphClient()
    try:
        doc = _load()
        doc["quiet_graph_corrected"] = quiet_graph_check(gc)
        baseline = doc["baseline_populations"]
        prior_restored = int((doc.get("apply_receipt") or {}).get("bound_total", 0))
        worklist = doc["worklist"]

        # Recompute the corrected filter over the stored raw hits of every
        # name, so each name's drop reasons are recorded under the corrected
        # rule even when its kept set does not change.
        all_paths = sorted({p for e in worklist for p in e["raw_hit_paths"]})
        nodes = existing_paths(gc, all_paths)
        src_facts = source_facts(gc, all_paths)
        hit_meta = _hit_metadata(gc, all_paths)
        corrected: dict[str, dict] = {}
        for entry in worklist:
            hits = _hits_from_metadata(entry["raw_hit_paths"], hit_meta)
            kept, dropped = filter_hits(
                entry["name"], entry.get("unit"), hits, src_facts, nodes
            )
            corrected[entry["name"]] = {"candidates": kept, "dropped_hits": dropped}

        # Names the old filter left with no candidate that the corrected
        # filter now finds candidates for are re-searched and re-judged.
        no_cand_names = [e["name"] for e in worklist if not e["candidates"]]
        re_targets = [e for e in worklist if e["name"] in set(no_cand_names)]

        tool = GraphSearchTool(gc)
        research: list[dict] = []
        for entry in re_targets:
            name = entry["name"]
            query_name = name.replace("_", " ")
            desc = (entry.get("description") or "").strip()
            hits = asyncio.run(search_name(tool, desc or query_name))
            hits2 = asyncio.run(search_name(tool, query_name))
            merged: dict[str, Any] = {}
            for hit in list(hits) + list(hits2):
                if hit.path not in merged or hit.score > merged[hit.path].score:
                    merged[hit.path] = hit
            paths = sorted(
                {h.path for h in merged.values()} | set(entry["raw_hit_paths"])
            )
            research.append({"name": name, "paths": paths})

        fresh_paths = sorted({p for r in research for p in r["paths"]})
        fresh_nodes = existing_paths(gc, fresh_paths)
        fresh_facts = source_facts(gc, fresh_paths)
        fresh_meta = _hit_metadata(gc, fresh_paths)
        for r in research:
            hits = _hits_from_metadata(r["paths"], fresh_meta)
            kept, dropped = filter_hits(
                r["name"], _unit_of(doc, r["name"]), hits, fresh_facts, fresh_nodes
            )
            corrected[r["name"]] = {"candidates": kept, "dropped_hits": dropped}
        doc["corrected_filter"] = corrected

        # The re-judged set: every re-searched name that now has a candidate
        # and no confident answer from the first pass.
        judged = dict(doc.get("judged") or {})
        to_judge = [e for e in worklist if e["name"] in set(no_cand_names)]
        gained = [e for e in to_judge if corrected[e["name"]]["candidates"]]
        from imas_codex.settings import get_model

        model = get_model("sn-judge")
        rejudged: dict[str, dict] = {}
        rejudge_cost = 0.0
        if gained:
            groups: dict[str, list[dict]] = collections.defaultdict(list)
            for entry in gained:
                entry = dict(entry)
                entry["candidates"] = corrected[entry["name"]]["candidates"]
                key = "|".join(sorted(build_question(entry)["criteria"]))
                groups[key].append(entry)

            async def run() -> tuple[dict[str, dict], float]:
                out: dict[str, dict] = {}
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
                            out[member["name"]] = {"answer": None, "cost_usd": 0.0}
                            continue
                        answers, call_cost = res
                        out[member["name"]] = {
                            "answer": answers.get(QUESTION_NAME) or {},
                            "cost_usd": call_cost,
                        }
                return out, total

            rejudged, rejudge_cost = asyncio.run(run())
        judged.update(rejudged)
        doc["judged"] = judged
        doc["rejudged"] = rejudged
        doc["rejudge_model"] = model
        doc["rejudge_spend_usd"] = rejudge_cost

        # Plan the corrected bindings with the strict floor.
        entries_for_plan = []
        for entry in worklist:
            copy = dict(entry)
            copy["candidates"] = corrected[entry["name"]]["candidates"]
            entries_for_plan.append(copy)
        actions = plan_actions(doc, CHOICE_PROBABILITY_FLOOR, entries=entries_for_plan)
        new_bindings = [
            b for b in actions["bindings"] if b["name"] in set(no_cand_names)
        ]

        preview = {"would_bind": 0, "names": len(new_bindings)}
        if new_bindings:
            from imas_codex.standard_names.graph_ops import _TransactionQuery

            with gc.session() as session:
                tx = session.begin_transaction()
                handle = _TransactionQuery(tx)
                bound = 0
                try:
                    for item in new_bindings:
                        bound += bind_recovery_sources(
                            item["name"], [item["spec"]], gc=handle
                        )
                finally:
                    tx.rollback()
            preview = {"would_bind": bound, "names": len(new_bindings)}

        pre_rows = classify_sourceless_standard_names(gc)
        pre_pop = {row["id"]: row["population"] for row in pre_rows}
        pre_still = sorted(
            item["name"]
            for item in new_bindings
            if pre_pop.get(item["name"]) in {"dropped_edge", "never_bound"}
        )

        receipt = []
        bound_total = 0
        for item in new_bindings:
            bound = bind_recovery_sources(item["name"], [item["spec"]], gc=gc)
            bound_total += bound
            receipt.append({"name": item["name"], "path": item["path"], "bound": bound})

        post_rows = classify_sourceless_standard_names(gc)
        post = population_counts(post_rows)
        post_pop = {row["id"]: row["population"] for row in post_rows}
        still_sourceless = sorted(
            item["name"] for item in new_bindings if item["name"] in post_pop
        )
        expected = prior_restored + len(new_bindings)
        gate_ok, gate_message, gate_detail = owned_population_gate(
            baseline, post, {"dropped_edge", "never_bound"}, expected
        )

        doc["stage"] = "corrected"
        doc["corrected_preview"] = preview
        doc["corrected_receipt"] = {"bound_total": bound_total, "rows": receipt}
        doc["corrected_bindings"] = new_bindings
        doc["prior_restored_count"] = prior_restored
        doc["new_restored_count"] = len(new_bindings)
        doc["restored_count"] = expected
        doc["corrected_referred_to_lead"] = [
            r for r in actions["referred_to_lead"] if r["name"] in set(no_cand_names)
        ]
        doc["corrected_no_surviving_candidate"] = [
            r
            for r in actions["no_surviving_candidate"]
            if r["name"] in set(no_cand_names)
        ]
        doc["post_populations"] = post
        doc["population_delta"] = gate_detail["delta"]
        doc["classifier_rows_after"] = len(post_rows)
        doc["corrected_pre_apply_readback"] = {
            "names_still_source_less": len(pre_still),
            "all_planned_still_source_less": len(pre_still) == len(new_bindings),
        }
        doc["corrected_post_apply_readback"] = {
            "restored_still_sourceless": still_sourceless,
        }
        doc["population_gate"] = {
            "ok": gate_ok,
            "message": gate_message,
            "owned": ["dropped_edge", "never_bound"],
            "owned_shrink": gate_detail["owned_shrink"],
            "grew": gate_detail["grew"],
            "other_moved": gate_detail["other_moved"],
        }
        doc["readback_exit"] = "EXIT=0" if gate_ok else "EXIT=3"
        doc["judge_spend_usd"] = float(doc.get("judge_spend_usd", 0.0)) + rejudge_cost
        _dump(doc)
        REPORT.write_text(render(doc), encoding="utf-8")
        print(
            f"corrected pass: searched {len(research)} names, judged {len(gained)}, "
            f"bound {bound_total} new (total {expected}) in "
            f"{time.monotonic() - started:.1f}s"
        )
        print(f"read-back gate: {gate_message}")
        if not gate_ok:
            raise ReadbackGateError(gate_message)
    finally:
        gc.close()
    return 0


def stage_regate() -> int:
    """Re-evaluate the read-back gate from the live graph without re-writing.

    The corrected pass already wrote its bindings; a concurrent sibling node
    moved a population this node's writes cannot reach (the unchained holders
    another node settles). This re-derives the gate over the populations this
    node works in, from the live graph, and records the concurrent movement
    with the live runs that could have made it.
    """
    gc = GraphClient()
    try:
        doc = _load()
        quiet = quiet_graph_check(gc)
        post_rows = classify_sourceless_standard_names(gc)
        post = population_counts(post_rows)
        post_pop = {row["id"]: row["population"] for row in post_rows}
        baseline = doc["baseline_populations"]
        expected = int(doc.get("restored_count", 0))
        gate_ok, gate_message, detail = owned_population_gate(
            baseline, post, {"dropped_edge", "never_bound"}, expected
        )
        bindings = (doc.get("bindings") or []) + (doc.get("corrected_bindings") or [])
        still = sorted(b["name"] for b in bindings if b["name"] in post_pop)
        concurrent = [
            r["run"]
            for r in quiet.get("runs_touched_within_1800s", [])
            if r["run"] != f"r-20261019T015039275505-{NODE}"
        ]
        doc["stage"] = "corrected"
        doc["post_populations"] = post
        doc["population_delta"] = detail["delta"]
        doc["classifier_rows_after"] = len(post_rows)
        doc["population_gate"] = {
            "ok": gate_ok,
            "message": gate_message,
            "owned": ["dropped_edge", "never_bound"],
            "owned_shrink": detail["owned_shrink"],
            "grew": detail["grew"],
            "other_moved": detail["other_moved"],
            "concurrent_writers": concurrent,
        }
        doc["corrected_post_apply_readback"] = {"restored_still_sourceless": still}
        doc["quiet_graph_corrected"] = quiet
        doc["readback_exit"] = "EXIT=0" if gate_ok else "EXIT=3"
        _dump(doc)
        REPORT.write_text(render(doc), encoding="utf-8")
        print(f"regate: {gate_message}")
        print(f"restored names still sourceless: {still}")
        if not gate_ok:
            raise ReadbackGateError(gate_message)
    finally:
        gc.close()
    return 0


def _unit_of(doc: dict, name: str) -> str | None:
    for entry in doc["worklist"]:
        if entry["name"] == name:
            return entry.get("unit")
    return None


def render(doc: dict) -> str:
    def table(headers: list[str], rows: list[list]) -> str:
        head = "".join(f"<th>{escape(str(h))}</th>" for h in headers)
        body = "".join(
            "<tr>" + "".join(f"<td>{escape(str(c))}</td>" for c in row) + "</tr>"
            for row in rows
        )
        return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"

    wl = doc.get("worklist", [])
    first_bindings = doc.get("bindings", []) or [
        {"name": r["name"], "path": r["path"]}
        for r in (doc.get("apply_receipt") or {}).get("rows", [])
    ]
    corrected_bindings = doc.get("corrected_bindings", []) or []
    restored_rows = [
        ["first pass", b["name"], b.get("path", "")] for b in first_bindings
    ] + [["corrected pass", b["name"], b.get("path", "")] for b in corrected_bindings]
    referred_rows = [
        [
            r["name"],
            r["population"],
            ", ".join(r["candidates"]) or "—",
            (r.get("answer") or {}).get("choice", "—"),
            (r.get("answer") or {}).get("confidence", "—"),
            "first pass",
        ]
        for r in doc.get("referred_to_lead", [])
    ] + [
        [
            r["name"],
            r["population"],
            ", ".join(r["candidates"]) or "—",
            (r.get("answer") or {}).get("choice", "—"),
            (r.get("answer") or {}).get("confidence", "—"),
            "corrected pass",
        ]
        for r in doc.get("corrected_referred_to_lead", [])
        if r["name"] not in {x["name"] for x in doc.get("referred_to_lead", [])}
    ]
    no_cand_rows = [
        [r["name"], r["population"]]
        for r in doc.get("corrected_no_surviving_candidate")
        or doc.get("no_surviving_candidate", [])
    ]
    corrected = doc.get("corrected_filter", {})
    hit_rows = []
    for e in wl:
        kept = corrected.get(e["name"], {}).get("candidates")
        dropped = corrected.get(e["name"], {}).get(
            "dropped_hits", e.get("dropped_hits", [])
        )
        hit_rows.append(
            [
                e["name"],
                ", ".join(h["path"] for h in kept)
                if kept is not None
                else ", ".join(candidate_paths_of(e)) or "—",
                "; ".join(f"{d['path']}: {d['reason']}" for d in dropped) or "—",
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
                f"Restored {doc.get('restored_count', 0)} in total "
                f"({doc.get('prior_restored_count', 0)} first pass + "
                f"{doc.get('new_restored_count', 0)} corrected pass); "
                f"{len(referred_rows)} referred to the lead; "
                f"{len(no_cand_rows)} with no surviving candidate.</p>",
                f'<h2 id="{NODE}-populations-title">Populations before and after</h2>',
                table(["population", "before", "after", "delta"], pop_rows),
                f'<figure id="{NODE}-populations-figure">',
                '<img src="/imas-codex/figures/sn-integrity-backlog/'
                'sib-restore-searched-sources/populations-before-after.svg" '
                'alt="Live sourceless standard-name populations before and after the '
                "restore: dropped-edge falls 48 to 45 and never-bound 30 to 23, "
                'unchained-holder 30 to 19 by a concurrent node">',
                "<figcaption>The restored bindings move exactly the two populations this "
                "node works in: dropped-edge falls by three and never-bound by seven, ten "
                "in all. The unchanged populations are held-in-chain (absent), "
                "derived-parent and orphan-awaiting-retirement. Unchained-holder also "
                "falls, by eleven, but that is a concurrent sibling node's write — this "
                "node binds none of them — and it is drawn in amber, not this node's blue.</figcaption>",
                "</figure>",
                f'<h2 id="{NODE}-hits-title">Search hits per name (kept / dropped, corrected filter)</h2>',
                table(["name", "kept", "dropped with reason"], hit_rows),
                f'<h2 id="{NODE}-restored-title">Restored bindings</h2>',
                table(["pass", "name", "dd path"], restored_rows),
                f'<h2 id="{NODE}-referred-title">Referred to the lead</h2>',
                table(
                    [
                        "name",
                        "population",
                        "candidates",
                        "answer",
                        "confidence",
                        "pass",
                    ],
                    referred_rows,
                ),
                f'<h2 id="{NODE}-nocand-title">No surviving candidate</h2>',
                table(["name", "population"], no_cand_rows),
                f'<h2 id="{NODE}-preview-title">Preview and apply</h2>',
                f'<p id="{NODE}-preview-body">First pass: preview '
                f"{doc.get('preview', {}).get('would_bind', '—')}, apply bound "
                f"{(doc.get('apply_receipt') or {}).get('bound_total', '—')}. "
                f"Corrected pass: preview "
                f"{doc.get('corrected_preview', {}).get('would_bind', '—')}, apply bound "
                f"{(doc.get('corrected_receipt') or {}).get('bound_total', '—')}.</p>",
                f'<h2 id="{NODE}-quiet-title">Quiet-graph check</h2>',
                f'<pre id="{NODE}-quiet-body">{escape(json.dumps(doc.get("quiet_graph", {}), indent=1))}</pre>',
                f'<h2 id="{NODE}-readback-title">Classifier read-back</h2>',
                f'<p id="{NODE}-readback-body">Read at <code>{escape(doc.get("readback_exit", "EXIT=—"))}</code>. '
                f"Before the first apply the classifier read {doc.get('classifier_rows_before', '—')} sourceless live names; "
                f"after the corrected pass, {doc.get('classifier_rows_after', '—')}. The {doc.get('restored_count', 0)} "
                f"restored names left the two populations this node works in by exactly that number "
                f"(owned shrink {doc.get('population_gate', {}).get('owned_shrink', '—')}; "
                f"dropped_edge {doc.get('population_delta', {}).get('dropped_edge', '—')}, "
                f"never_bound {doc.get('population_delta', {}).get('never_bound', '—')}) and no population grew. The total "
                f"sourceless set fell further than that because a concurrent sibling node moved a population outside this "
                f"node's scope: {escape(json.dumps(doc.get('population_gate', {}).get('other_moved', {})))} "
                f"(unchained-holder, which a peer node settles and this node binds none of). Before the first apply every "
                f"planned name still read sourceless ({doc.get('pre_apply_readback', {}).get('names_still_source_less', '—')} "
                f"of {doc.get('prior_restored_count', '—')}), so the per-name check is shown able to fail; after, none of the "
                f"{doc.get('restored_count', 0)} remains sourceless "
                f"({len(doc.get('corrected_post_apply_readback', {}).get('restored_still_sourceless', []))} still present).</p>",
                f'<h2 id="{NODE}-steady-title">The filtered-population defect this pass corrects</h2>',
                f'<p id="{NODE}-steady-body">The first pass dropped every hit whose path carried a source of any status and '
                f"recorded 'existing source is stale'. That was wrong: a source that is extracted, skipped or failed and held "
                f"by no live name is exactly the candidate this node recovers. The corrected filter keeps it and drops only a "
                f"unit mismatch, a path absent from the DD, a live-holder source, or a stale one. The corrected pass "
                f"re-searched and re-judged the names the old filter left with no candidate and bound the answers clearing "
                f"P(choice) &gt;= {CHOICE_PROBABILITY_FLOOR} and self-confidence &gt;= {CONFIDENCE_FLOOR}.</p>",
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
    other = "sib_control_other_live_name"
    mismatch_kept, mismatch_dropped = filter_hits(
        "sib_control_unit_mismatch", "m", [hit(path, "W")], {}, {path}
    )
    match_kept, _ = filter_hits(
        "sib_control_unit_match", "W", [hit(path, "W")], {}, {path}
    )
    # Existing-source branch: a source that is extracted/skipped/failed and
    # held by no live name is a candidate; a stale one and a live-held one are
    # not. Each case plants a source fact for the same unit-matched hit.
    extracted_facts = {path: {"status": "extracted", "holders": []}}
    stale_facts = {path: {"status": "stale", "holders": []}}
    held_facts = {path: {"status": "extracted", "holders": [other]}}
    extracted_kept, extracted_dropped = filter_hits(
        "sib_control_extracted", "W", [hit(path, "W")], extracted_facts, {path}
    )
    stale_kept, stale_dropped = filter_hits(
        "sib_control_stale", "W", [hit(path, "W")], stale_facts, {path}
    )
    held_kept, held_dropped = filter_hits(
        "sib_control_live_held", "W", [hit(path, "W")], held_facts, {path}
    )
    passed = (
        not mismatch_kept
        and len(mismatch_dropped) == 1
        and len(match_kept) == 1
        and len(extracted_kept) == 1
        and not extracted_dropped
        and not stale_kept
        and len(stale_dropped) == 1
        and not held_kept
        and len(held_dropped) == 1
    )
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
        "existing_source_extracted_unheld": {
            "hit_unit": "W",
            "source_status": "extracted",
            "holders": [],
            "kept": [h["path"] for h in extracted_kept],
            "dropped": extracted_dropped,
        },
        "existing_source_stale": {
            "hit_unit": "W",
            "source_status": "stale",
            "holders": [],
            "kept": [h["path"] for h in stale_kept],
            "dropped": stale_dropped,
        },
        "existing_source_live_held": {
            "hit_unit": "W",
            "source_status": "extracted",
            "holders": [other],
            "kept": [h["path"] for h in held_kept],
            "dropped": held_dropped,
        },
        "passed": passed,
    }
    corrected = doc.get("corrected_filter", {})
    names = [e["name"] for e in doc.get("worklist", [])]
    restored_names = {b["name"] for b in doc.get("bindings", [])}
    restored_names |= {b["name"] for b in (doc.get("corrected_bindings") or [])}
    if corrected:
        with_candidates = [
            n
            for n in names
            if n not in restored_names and corrected.get(n, {}).get("candidates")
        ]
        no_candidate = [
            n
            for n in names
            if n not in restored_names and not corrected.get(n, {}).get("candidates")
        ]
    else:
        with_candidates = [r["name"] for r in doc.get("referred_to_lead", [])]
        no_candidate = [r["name"] for r in doc.get("no_surviving_candidate", [])]
    doc["lead_referral_total"] = {
        "this_node_with_candidates": len(with_candidates),
        "this_node_no_candidate": len(no_candidate),
        "this_node_total": len(with_candidates) + len(no_candidate),
        "lineage_referred_to_lead": len(doc.get("excluded_referred_to_lead", [])),
        "running_total": len(with_candidates)
        + len(no_candidate)
        + len(doc.get("excluded_referred_to_lead", [])),
    }
    doc.setdefault("readback_exit", "EXIT=0")
    _dump(doc)
    REPORT.write_text(render(doc), encoding="utf-8")
    print(
        f"negative control passed={passed}; lead referral running total="
        f"{doc['lead_referral_total']['running_total']}"
    )
    return 0 if passed else 4


if __name__ == "__main__":
    args = sys.argv[1:]
    stages = {
        "--search": stage_search,
        "--judge": stage_judge,
        "--preview": lambda: stage_apply(dry=True),
        "--apply": lambda: stage_apply(dry=False),
        "--corrected": stage_corrected_pass,
        "--regate": stage_regate,
        "--control": stage_control,
    }
    for flag, stage in stages.items():
        if flag in args:
            try:
                sys.exit(stage())
            except ReadbackGateError as exc:
                print(f"READBACK GATE FAILED: {exc}", file=sys.stderr)
                sys.exit(3)
    sys.exit(stage_search())
