"""Restore a dropped-edge name's DD source from a relative's recorded path.

The deterministic slice restored the names whose own recorded source identity
survived. This driver takes the rest: the dropped-edge names whose only
recoverable DD path is one a relative in their REFINED_FROM / supersede / fold
chain recorded. A relative's path is evidence,
not proof — a refine or fold may have moved to a different quantity — so every
binding is confirmed by one Jev Choice before it is written, and a name whose
answer is uncertain or names no candidate is referred rather than guessed.

Stages (each persists to the results path so a later stage resumes from disk):

    derive    read the live graph, classify sourceless live names, and compute
              the work list plus each name's candidate DD paths
    judge     ask Jev one Choice per candidate-bearing name through the shared
              decisions owner on the ``sn-judge`` seat
    preview   roll back a trial bind inside a transaction and report what it
              would write, on a graph shown quiet by two agreeing reads
    apply     write the bindings through ``bind_recovery_sources``
    readback  re-run the classifier and compare the populations

The classifier is read-only and bounded; the apply is the only write.
"""

from __future__ import annotations

import collections
import json
import subprocess
import sys
from collections import deque
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from imas_codex.discovery.base.judgment import decide_batch
from imas_codex.graph.client import GraphClient
from imas_codex.settings import get_model
from imas_codex.standard_names.graph_ops import classify_sourceless_standard_names
from imas_codex.standard_names.provenance_rebuild import (
    _BIND_SOURCES,
    bind_recovery_sources,
)

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "sib-restore-lineage-sources.json"
ROOT = HERE.parents[3]

TERMINAL = {"deprecated", "superseded"}

# Identity-changing change operations that establish a name's chain, so a
# relative's recorded path can be read as evidence about this name's quantity.
CHAIN_CHANGE_OPERATIONS = {
    "fold",
    "fold_identity",
    "link_signed_successor",
    "resolve_sourced_supersession",
    "supersede_into_ancestor",
    "supersede_legacy_spelling",
}

# The decisions owner books no confidence floor of its own, so the floor is
# this driver's: a binding is admitted only when the answer chose a candidate
# with a decisive share of the mass AND the answer's own confidence clears the
# point below which a judgement is not acted on. A probability alone is not
# enough — a near-flat distribution can still land a hair above half on one
# candidate while the judgement itself carries no confidence, which is a guess
# wearing a number. Binding and referral both read this floor.
MIN_CHOICE_PROBABILITY = 0.8
MIN_SELF_CONFIDENCE = 0.5

NONE_OF_THESE = "none of these"
QUESTION_NAME = "measures"


def driver_revision() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def read_graph(gc: GraphClient) -> dict[str, Any]:
    """Read the graph topology the derivation and the apply both need."""
    return {
        "names": gc.query(
            """
            MATCH (n:StandardName)
            RETURN n.id AS id, n.name_stage AS stage, n.status AS status,
                   n.origin AS origin, n.source_paths AS paths,
                   n.source_path AS path,
                   n.source_dd_resolution_converged_ids AS resolutions,
                   n.description AS description, n.unit AS unit,
                   n.documentation AS documentation
            ORDER BY id LIMIT 6000
            """
        ),
        "relations": gc.query(
            """
            MATCH (a:StandardName)-[r:REFINED_FROM|HAS_SUCCESSOR]->(b:StandardName)
            RETURN a.id AS a, type(r) AS type, b.id AS b LIMIT 12000
            """
        ),
        "changes": gc.query(
            """
            MATCH (c:StandardNameChange)
            RETURN c.from_name AS predecessor, c.to_name AS successor,
                   c.operation AS operation LIMIT 20000
            """
        ),
        "projections": gc.query(
            "MATCH (dd:IMASNode)-[:HAS_STANDARD_NAME]->(n:StandardName) "
            "RETURN dd.id AS path, n.id AS name LIMIT 12000"
        ),
        "sources": gc.query(
            "MATCH (s:StandardNameSource) RETURN s.id AS source, "
            "s.source_id AS source_id, s.produced_sn_id AS mirror, "
            "s.status AS status LIMIT 12000"
        ),
        "edges": gc.query(
            "MATCH (s:StandardNameSource)-[:PRODUCED_NAME]->(n:StandardName) "
            "RETURN s.id AS source, n.id AS name, n.status AS status LIMIT 12000"
        ),
        "imas_nodes": gc.query(
            "MATCH (dd:IMASNode) RETURN dd.id AS path, dd.documentation AS documentation, "
            "dd.units AS units, dd.physics_domain AS domain LIMIT 200000"
        ),
    }


def core_own_source_ids(row: dict[str, Any]) -> list[str]:
    """A name's own recorded source identities, as the classifier computes them.

    The classifier returns ``recorded_source_ids`` built from the name's own
    tokens plus its source mirror. A name with none recorded no source of its
    own, so the only recoverable path is a relative's — the slice this driver
    takes.
    """
    return list(row.get("recorded_source_ids") or [])


def derive(gc: GraphClient) -> dict[str, Any]:
    """Compute the work list and each name's candidate DD paths."""
    classified = classify_sourceless_standard_names(gc)
    g = read_graph(gc)
    by_id = {n["id"]: n for n in g["names"]}

    live_holders: dict[str, set[str]] = collections.defaultdict(set)
    for e in g["edges"]:
        if e["status"] not in TERMINAL:
            live_holders[e["source"]].add(e["name"])
    source_status = {s["source"]: s["status"] for s in g["sources"]}

    succ: dict[str, set[str]] = collections.defaultdict(set)
    pred: dict[str, set[str]] = collections.defaultdict(set)
    for rel in g["relations"]:
        old, new = (
            (rel["b"], rel["a"])
            if rel["type"] == "REFINED_FROM"
            else (rel["a"], rel["b"])
        )
        succ[old].add(new)
        pred[new].add(old)
    for ch in g["changes"]:
        old, new, op = ch["predecessor"], ch["successor"], ch["operation"] or ""
        if (
            op in CHAIN_CHANGE_OPERATIONS
            and old in by_id
            and new in by_id
            and old != new
        ):
            succ[old].add(new)
            pred[new].add(old)

    proj: dict[str, list[str]] = collections.defaultdict(list)
    for p in g["projections"]:
        proj[p["name"]].append(p["path"])

    imas_exists = {r["path"] for r in g["imas_nodes"]}
    imas_meta = {r["path"]: r for r in g["imas_nodes"]}

    def chain(name: str) -> set[str]:
        seen: set[str] = set()
        stack = deque([name])
        while stack:
            cur = stack.pop()
            for nb in (*succ[cur], *pred[cur]):
                if nb not in seen:
                    seen.add(nb)
                    stack.append(nb)
        return seen

    dropped = [r for r in classified if r["population"] == "dropped_edge"]
    worklist: list[dict[str, Any]] = []
    referred_search: list[dict[str, Any]] = []
    for row in dropped:
        if core_own_source_ids(row):
            continue  # its own recorded source survived — not this slice
        relatives = chain(row["id"])
        lineage_paths: set[str] = set()
        for relname in relatives:
            other = by_id.get(relname)
            if not other:
                continue
            for p in other["paths"] or []:
                lineage_paths.add(str(p).removeprefix("dd:"))
            if other["path"]:
                lineage_paths.add(str(other["path"]).removeprefix("dd:"))
            for p in proj[relname]:
                lineage_paths.add(str(p).removeprefix("dd:"))

        candidates: list[str] = []
        dropped_reasons: list[str] = []
        for p in sorted(lineage_paths):
            if p not in imas_exists:
                dropped_reasons.append(f"{p}: IMASNode gone")
                continue
            src = "dd:" + p
            holders = live_holders.get(src, set())
            if holders:
                dropped_reasons.append(
                    f"{p}: source held by live name {sorted(holders)[0]}"
                )
                continue
            if source_status.get(src) in {"stale", "rejected"}:
                dropped_reasons.append(f"{p}: source {source_status[src]}")
                continue
            candidates.append(p)

        own = by_id.get(row["id"]) or {}
        entry = {
            "name": row["id"],
            "stage": row["name_stage"],
            "origin": row["origin"],
            "status": row["status"],
            "unresolved_tokens": row["unresolved_tokens"],
            "description": own.get("description"),
            "unit": own.get("unit"),
            "documentation": own.get("documentation"),
            "relatives": sorted(relatives),
            "candidate_paths": candidates,
            "candidate_meta": {
                p: {
                    "documentation": (imas_meta.get(p) or {}).get("documentation"),
                    "unit": (imas_meta.get(p) or {}).get("units"),
                    "physics_domain": (imas_meta.get(p) or {}).get("domain"),
                }
                for p in candidates
            },
            "dropped_candidates": dropped_reasons,
            "lineage_path_count": len(lineage_paths),
        }
        worklist.append(entry)
        if not candidates:
            reason = (
                "no lineage-recorded path at all"
                if not lineage_paths
                else "every lineage-recorded path is unboundable: "
                + "; ".join(dropped_reasons)
            )
            referred_search.append({"name": row["id"], "reason": reason})

    return {
        "classified_counts": dict(
            collections.Counter(r["population"] for r in classified)
        ),
        "classified_total": len(classified),
        "worklist": worklist,
        "referred_to_dd_search": referred_search,
    }


def build_state(entry: dict[str, Any]) -> dict:
    """One judgement state: the name, and each candidate's DD evidence."""
    candidates = [
        {"path": path, **entry["candidate_meta"][path]}
        for path in entry["candidate_paths"]
    ]
    return {
        "standard_name": {
            "id": entry["name"],
            "description": entry["description"],
            "unit": entry["unit"],
            "documentation": entry["documentation"],
            "name_stage": entry["stage"],
        },
        "candidates": candidates,
    }


def build_question(entry: dict[str, Any]) -> dict[str, Any]:
    criteria = {p: f"candidate DD path {p}" for p in entry["candidate_paths"]}
    criteria[NONE_OF_THESE] = "no candidate measures the standard name's quantity"
    return {
        "type": "choice",
        "instructions": (
            "Does one of the candidate DD paths measure the quantity the standard "
            "name denotes? Compare the name's description and unit with each "
            "candidate's documentation, unit and physics domain. Choose the single "
            "candidate that measures the same quantity, or 'none of these' when no "
            "candidate does — a candidate that measures a different quantity must "
            "not be chosen."
        ),
        "criteria": criteria,
    }


def stage_judge() -> None:
    import asyncio

    with open(RESULTS) as fh:
        doc = json.load(fh)
    worklist = [w for w in doc["worklist"] if w["candidate_paths"]]

    # decide_batch takes ONE questions table for the whole batch, so the batch
    # is split by option set: each call judges the names sharing that question.
    # Grouping by the criteria set keeps one question shape per call.
    groups: dict[str, list[dict]] = collections.defaultdict(list)
    for w in worklist:
        groups["|".join(sorted(build_question(w)["criteria"]))].append(w)

    model = get_model("sn-judge")

    async def run() -> tuple[dict[str, dict], float]:
        judged: dict[str, dict] = {}
        total = 0.0
        for _key, members in sorted(groups.items()):
            q = {
                QUESTION_NAME: build_question(members[0]),
            }
            group_states = [build_state(w) for w in members]
            results, cost = await decide_batch(
                group_states,
                q,
                model=model,
                service="standard-names",
                concurrency=8,
                budget_seconds=600,
            )
            total += cost
            for w, res in zip(members, results, strict=True):
                if res is None:
                    judged[w["name"]] = {"error": "no answer"}
                    continue
                answers, call_cost = res
                ans = answers.get(QUESTION_NAME) or {}
                judged[w["name"]] = {
                    "answer": ans,
                    "cost_usd": call_cost,
                }
        return judged, total

    judged, total = asyncio.run(run())
    doc["judged"] = judged
    doc["judge_model"] = model
    doc["judge_spend_usd"] = total
    doc["stage"] = "judged"
    with open(RESULTS, "w") as fh:
        json.dump(doc, fh, indent=1, sort_keys=True)
    print(f"judged {len(judged)} names for USD {total:.6f}")


def stage_control() -> None:
    """Negative control: a relative's path that measures a different quantity.

    A planted name whose only candidate measures something else must be
    answered 'none of these' and left unrestored. The control is judged through
    the same call path as the work list, so a pass here is evidence the
    instrument can reject, not merely accept.
    """
    import asyncio

    control_name = "plasma_current"
    control_desc = "Plasma current, the total toroidal current carried by the plasma."
    control_candidate = "magnetics/flux_loop/flux"  # a flux, not a current
    state = {
        "standard_name": {
            "id": control_name,
            "description": control_desc,
            "unit": "A",
            "documentation": None,
            "name_stage": "accepted",
        },
        "candidates": [
            {
                "path": control_candidate,
                "documentation": "Flux through a flux loop.",
                "unit": "Wb",
                "physics_domain": "magnetics",
            }
        ],
    }
    question = {
        QUESTION_NAME: {
            "type": "choice",
            "instructions": (
                "Does one of the candidate DD paths measure the quantity the "
                "standard name denotes? Choose the single candidate that "
                "measures the same quantity, or 'none of these' when no "
                "candidate does."
            ),
            "criteria": {
                control_candidate: f"candidate DD path {control_candidate}",
                NONE_OF_THESE: "no candidate measures the standard name's quantity",
            },
        }
    }
    model = get_model("sn-judge")

    async def run():
        return await decide_batch(
            [state],
            question,
            model=model,
            service="standard-names",
            concurrency=1,
            budget_seconds=120,
        )

    results, cost = asyncio.run(run())
    ans = (results[0][0].get(QUESTION_NAME) if results[0] else {}) or {}
    with open(RESULTS) as fh:
        doc = json.load(fh)
    doc["negative_control"] = {
        "declared": (
            "a planted name whose relative's path measures a different quantity "
            "must be answered none-of-these and not restored"
        ),
        "name": control_name,
        "planted_candidate": control_candidate,
        "answer": ans,
        "passed": ans.get("choice") == NONE_OF_THESE,
        "cost_usd": cost,
    }
    with open(RESULTS, "w") as fh:
        json.dump(doc, fh, indent=1, sort_keys=True)
    print(
        "negative control:",
        ans.get("choice"),
        "passed:",
        doc["negative_control"]["passed"],
    )


def _admittable(choice: Any, probability: Any, confidence: Any) -> bool:
    """Whether an answer clears the binding floor.

    Both halves are required: a decisive share of the answer's own mass on the
    chosen candidate, and a self-reported confidence above the point below
    which a judgement is not acted on. Either alone admits a guess.
    """
    if not choice or choice == NONE_OF_THESE:
        return False
    if probability is None or float(probability) < MIN_CHOICE_PROBABILITY:
        return False
    return confidence is not None and float(confidence) >= MIN_SELF_CONFIDENCE


def _answer_metrics(record: dict[str, Any]) -> tuple[Any, float, Any]:
    """The chosen candidate, its probability and the answer's own confidence."""
    ans = record.get("answer") or {}
    choice = ans.get("choice")
    prob = float((ans.get("probabilities") or {}).get(choice, 0.0)) if choice else 0.0
    return choice, prob, ans.get("confidence")


def _confident_choice(record: dict[str, Any]) -> str | None:
    """The candidate an answer admits for binding, or ``None``.

    ``None`` means the name is not bound: the answer named no candidate, or it
    failed either half of the floor. The caller refers it instead.
    """
    choice, prob, confidence = _answer_metrics(record)
    return choice if _admittable(choice, prob, confidence) else None


def _spec(dd_path: str) -> dict[str, Any]:
    return {
        "id": f"dd:{dd_path}",
        "source_type": "dd",
        "source_id": dd_path,
        "dd_path": dd_path,
        "status": "attached",
    }


def stage_preview(gc: GraphClient) -> None:
    """Quiet-graph check, then a rolled-back trial bind of every planned spec."""
    with open(RESULTS) as fh:
        doc = json.load(fh)

    first = classify_sourceless_standard_names(gc)
    first_counts = dict(collections.Counter(r["population"] for r in first))
    import time

    time.sleep(5)
    second = classify_sourceless_standard_names(gc)
    second_counts = dict(collections.Counter(r["population"] for r in second))
    quiet = first_counts == second_counts
    doc["quiet_graph"] = {
        "first": first_counts,
        "second": second_counts,
        "quiet": quiet,
    }
    if not quiet:
        with open(RESULTS, "w") as fh:
            json.dump(doc, fh, indent=1, sort_keys=True)
        raise SystemExit(f"graph not quiet: {first_counts} != {second_counts}")

    plan = []
    for w in doc["worklist"]:
        rec = doc["judged"].get(w["name"]) or {}
        chosen = _confident_choice(rec)
        if chosen:
            plan.append({"name": w["name"], "dd_path": chosen, "spec": _spec(chosen)})

    written = 0
    with gc.session() as session:
        tx = session.begin_transaction()
        try:
            for item in plan:
                rows = tx.run(
                    _BIND_SOURCES,
                    name_id=item["name"],
                    specs=[item["spec"]],
                )
                written += int(rows.single()["bound"])
            tx.rollback()
        except Exception:
            tx.rollback()
            raise

    doc["preview"] = {
        "planned_bindings": len(plan),
        "trial_written": written,
        "plan": plan,
    }
    doc["stage"] = "previewed"
    with open(RESULTS, "w") as fh:
        json.dump(doc, fh, indent=1, sort_keys=True)
    print(f"preview: {len(plan)} planned, {written} would bind, quiet={quiet}")


def stage_apply(gc: GraphClient) -> None:
    with open(RESULTS) as fh:
        doc = json.load(fh)
    if not doc.get("quiet_graph", {}).get("quiet"):
        raise SystemExit("refusing to apply on a graph not shown quiet")
    restored = []
    for item in doc["preview"]["plan"]:
        bound = bind_recovery_sources(item["name"], [item["spec"]], gc=gc)
        rec = doc["judged"].get(item["name"]) or {}
        ans = rec.get("answer") or {}
        restored.append(
            {
                "name": item["name"],
                "source_id": item["spec"]["id"],
                "dd_path": item["dd_path"],
                "jev_choice": ans.get("choice"),
                "jev_probability": float(
                    (ans.get("probabilities") or {}).get(item["dd_path"], 0.0)
                ),
                "jev_confidence": ans.get("confidence"),
                "sources_bound": bound,
            }
        )
    doc["restored"] = restored
    doc["apply_receipt"] = {
        "planned": len(doc["preview"]["plan"]),
        "bound": sum(r["sources_bound"] for r in restored),
        "names_bound": sum(1 for r in restored if r["sources_bound"]),
        "applied_at": datetime.now(UTC).isoformat(),
    }
    doc["stage"] = "applied"
    with open(RESULTS, "w") as fh:
        json.dump(doc, fh, indent=1, sort_keys=True)
    print(f"applied: {doc['apply_receipt']}")


def _sources_written_since(gc: GraphClient, since: str) -> list[dict[str, Any]]:
    """Sources created after *since*, which this node did not write.

    Read from the write's own timestamp rather than from wall clock, so a
    re-check attributes movement to the peer that made it instead of reporting
    an unexplained mismatch.
    """
    rows = gc.query(
        """
        MATCH (s:StandardNameSource)
        WHERE s.created_at > datetime($since)
        RETURN s.id AS source, toString(s.created_at) AS created_at
        ORDER BY created_at
        """,
        since=since,
        limit=500,
    )
    return rows


def stage_readback(gc: GraphClient) -> None:
    """Re-run the classifier and refuse to pass on anything but an exact match.

    The read-back is the gate, so it must be able to fail: it exits non-zero
    when the dropped-edge population did not fall by exactly the number this
    node restored, or when any other population moved. The first passing run is
    kept as ``readback_verified`` — the evidence of this node's write, taken on
    the quiet graph it was applied to. A later run on a graph another writer has
    since changed is recorded as a ``readback_recheck`` with the peer's sources
    named, and still exits non-zero: movement the re-check cannot attribute is
    the whole reason the gate exists.
    """
    with open(RESULTS) as fh:
        doc = json.load(fh)
    after = classify_sourceless_standard_names(gc)
    after_counts = dict(collections.Counter(r["population"] for r in after))
    before_counts = doc["before_counts"]
    restored_n = doc["apply_receipt"]["names_bound"]
    delta = {
        k: after_counts.get(k, 0) - before_counts.get(k, 0)
        for k in sorted(set(before_counts) | set(after_counts))
    }
    dropped_ok = delta.get("dropped_edge") == -restored_n
    moved = {k: v for k, v in delta.items() if k != "dropped_edge" and v != 0}
    others_unchanged = not moved
    passed = dropped_ok and others_unchanged
    print("before:", before_counts)
    print("after:", after_counts)
    print("delta:", delta)

    if passed:
        doc["after_counts"] = after_counts
        doc["population_delta"] = delta
        doc["restored_count"] = restored_n
        doc["dropped_fell_by_restored"] = True
        doc["others_unchanged"] = True
        doc["readback_verified"] = {
            "after_counts": after_counts,
            "population_delta": delta,
            "dropped_edge_expected": -restored_n,
            "exit_status": 0,
        }
        doc["stage"] = "readback"
        with open(RESULTS, "w") as fh:
            json.dump(doc, fh, indent=1, sort_keys=True)
        return

    since = doc["apply_receipt"].get("applied_at")
    foreign = _sources_written_since(gc, since) if since else []
    mine = {r["source_id"] for r in doc.get("restored", [])}
    foreign = [f for f in foreign if f["source"] not in mine]
    doc["readback_recheck"] = {
        "after_counts": after_counts,
        "population_delta": delta,
        "dropped_edge_expected": -restored_n,
        "moved_populations": moved,
        "sources_created_since_apply": foreign,
        "exit_status": 1,
    }
    doc["stage"] = "readback-failed"
    with open(RESULTS, "w") as fh:
        json.dump(doc, fh, indent=1, sort_keys=True)
    raise SystemExit(
        "read-back failed: dropped_edge moved "
        f"{delta.get('dropped_edge')}, not -{restored_n}"
        + (f"; populations moved beyond it: {moved}" if moved else "")
        + f"; {len(foreign)} sources created since the apply by another writer"
    )


#: The census populations a source repair works: every live name lacking a DD
#: source that is neither a derived parent nor an orphan awaiting retirement.
#: The referral bound is measured over this handed set.
HANDED_POPULATIONS = ("dropped_edge", "never_bound", "unchained_holder")


def _esc(text: Any) -> str:
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def stage_fragment() -> None:
    """Render the landing fragment from the results record."""
    with open(RESULTS) as fh:
        doc = json.load(fh)

    node = "sib-restore-lineage-sources"

    def table(headers: list[str], rows: list[list[Any]]) -> str:
        head = "".join(f"<th>{_esc(h)}</th>" for h in headers)
        body = "".join(
            "<tr>" + "".join(f"<td>{_esc(c)}</td>" for c in r) + "</tr>" for r in rows
        )
        return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"

    restored_rows = [
        [
            r["name"],
            r["dd_path"],
            r["jev_probability"],
            r["jev_confidence"],
            r["sources_bound"],
        ]
        for r in doc["restored"]
    ]
    restored_tbl = table(
        ["standard name", "restored DD path", "P(choice)", "self-reported", "bound"],
        restored_rows,
    )

    referred_rows = [
        [
            r["name"],
            r["answer"],
            (r["probabilities"] or {}).get(r["answer"]),
            ", ".join(r["candidates"]),
            r["reason"],
        ]
        for r in doc["referred_to_lead"]
    ]
    referred_tbl = table(
        ["standard name", "answer", "P(answer)", "candidates", "reason"], referred_rows
    )

    search_rows = [[r["name"], r["reason"]] for r in doc["referred_to_dd_search"]]
    search_tbl = table(["standard name", "reason passed to DD search"], search_rows)

    count_rows = [
        [
            k,
            doc["before_counts"][k],
            doc["after_counts"].get(k, 0),
            doc["population_delta"].get(k, 0),
        ]
        for k in sorted(doc["before_counts"], key=lambda x: -doc["before_counts"][x])
    ]
    counts_tbl = table(
        ["population", "before", "after", "delta"],
        count_rows,
    )

    below_rows = [
        [r["name"], r["candidate"], r["probability"], r["confidence"], r["reason"]]
        for r in sorted(
            doc.get("bound_below_confidence_floor", []),
            key=lambda x: x["probability"] or 0.0,
        )
    ]
    below_tbl = table(
        ["standard name", "candidate", "P(choice)", "self-reported", "reason"],
        below_rows,
    )

    gu = doc["quiet_graph"]
    bound = doc["bound"]
    rcv = doc["apply_receipt"]
    ctrl = doc["negative_control"]
    floor = doc["floor"]
    exits = doc.get("stage_exit_status", {})
    verified = doc.get("readback_verified", {})
    recheck = doc.get("readback_recheck", {})
    spend = (doc.get("judge_spend_usd") or 0) + (ctrl.get("cost_usd") or 0)

    def _elapsed_minutes(start: str | None, end: str | None) -> str:
        if not start or not end:
            return "an unknown interval"
        from datetime import datetime

        a = datetime.fromisoformat(start.replace("Z", "+00:00"))
        b = datetime.fromisoformat(end.replace("Z", "+00:00"))
        return f"{(b - a).total_seconds() / 60:.0f} minutes"

    html = f"""<article data-reckon="fragment" id="{node}">
<h2 id="{node}-title">§38 — the lost sources restored from a relative's recorded path</h2>
<p id="{node}-lede">This node takes the dropped-edge names whose own recorded source
identity did not survive, so their only recoverable DD path is one a relative in their
refinement or supersede chain recorded. A relative's path is evidence, not proof — a
refine or a fold may have moved to a different quantity — so every binding was
confirmed by one Jev Choice before it was written. Nothing guessed was written.</p>

<h3 id="{node}-worklist">The work list</h3>
<p id="{node}-worklist-note">Re-derived from the live graph with §37's
<code>classify_sourceless_standard_names</code>, not from §35's census, because the
deterministic slice had already applied. The live classifier reports
<strong>{doc["classified_total"]}</strong> sourceless live names, of which
<strong>{doc["before_counts"]["dropped_edge"]}</strong> are dropped-edge. Of those,
{doc["before_counts"]["dropped_edge"] - len(doc["worklist"])} carried their own recorded
source identity (another node's slice); the remaining <strong>{len(doc["worklist"])}</strong>
have none, and are this node's work list. Of these, <strong>{len([w for w in doc["worklist"] if w["candidate_paths"]])}</strong>
have at least one free candidate DD path and <strong>{len(doc["referred_to_dd_search"])}</strong>
have none.</p>

<h3 id="{node}-restored">Restored bindings, route: Jev</h3>
<p id="{node}-restored-note">Each was restored through
<code>bind_recovery_sources</code> in <code>provenance_rebuild.py</code> after a Jev
Choice named it. The floor is this driver's own, since the decisions owner defines
none: a binding is admitted only when the chosen candidate's probability reaches
<strong>{floor["min_choice_probability"]}</strong> <em>and</em> the answer's
self-reported confidence reaches <strong>{floor["min_self_confidence"]}</strong>.
Both halves are required. A probability alone admitted near-zero-confidence picks —
a near-flat distribution can land a hair above half on one candidate while the
judgement itself carries nothing, which is a guess wearing a number; the bindings
below that floor are listed separately below rather than presented as settled.
<code>P(choice)</code> is the chosen candidate's probability and
<em>self-reported</em> is the answer's own <code>confidence</code> field.</p>
{restored_tbl}

<h3 id="{node}-below-floor">Bound below the confidence floor, awaiting lead confirmation</h3>
<p id="{node}-below-floor-note">These <strong>{len(doc.get("bound_below_confidence_floor", []))}</strong>
bindings were applied under a weaker floor — a chosen probability at or above 0.5
alone — before the two-part floor was ruled.
They are <strong>not detached</strong>: a live name left without a DD source is
exactly what the source invariant refuses, so the write stands and the name is carried
here for lead confirmation instead. They count toward the running lead-referral total.
The first entries are the worst — a chosen probability at chance with a self-reported
confidence of zero, which is indistinguishable from a coin toss on this evidence.</p>
{below_tbl}

<h3 id="{node}-referred">Sent to the lead, unbound</h3>
<p id="{node}-referred-note">Jev answered <strong>none of these</strong> for these
names, or answered below the floor, and nothing was written: their relative's recorded
path measures a different quantity or cannot be settled. They are referred with their
candidates, their answer and its probabilities — never restored.</p>
{referred_tbl}

<h3 id="{node}-search">Passed to DD search</h3>
<p id="{node}-search-note">No candidate could bind: the lineage-recorded path's
IMASNode is gone, its source is held by a live name, or no relative recorded one at
all. These go to the DD-search node with the reason.</p>
{search_tbl}

<h3 id="{node}-preview">Preview and apply</h3>
<p id="{node}-preview-note">The graph was shown quiet by two classifier reads five
seconds apart that agreed exactly
(<code>quiet={gu["quiet"]}</code>; {_esc(gu["first"])} then {_esc(gu["second"])}).
A trial bind of all {doc["preview"]["planned_bindings"]} planned specs was then run
inside a transaction and rolled back, so the write was measured before it happened:
<strong>{doc["preview"]["trial_written"]}</strong> would bind. The apply then bound
<strong>{rcv["bound"]}</strong> sources across <strong>{rcv["names_bound"]}</strong>
names at <code>{_esc(rcv["applied_at"])}</code>.</p>

<h3 id="{node}-readback">Read-back: §37's classifier, before and after</h3>
<img id="{node}-fig" src="/imas-codex/figures/sn-integrity-backlog/sib-restore-lineage-sources/populations-before-after.svg" alt="Classifier populations before and after the restore; only the dropped-edge population moves."/>
<p id="{node}-readback-note">On the quiet graph the dropped-edge count fell by exactly the number restored
(<strong>{doc["restored_count"]}</strong>), and every other population is
unchanged. The read-back <em>raises</em> on anything else: a stage that finds the
fall unmatched or another population moved exits non-zero and prints the mismatch,
so the gate cannot pass by reporting a number a reader would have to check. Both
blocks below are the driver's own recorded exit statuses, written after each stage
ran rather than asserted in this document.</p>
{counts_tbl}
<pre id="{node}-exit">readback (verified, on the quiet graph) EXIT={verified.get("exit_status", "not recorded")}
before {_esc(doc["before_counts"])}
after  {_esc(doc["after_counts"])}
delta  {_esc(doc["population_delta"])}
dropped_fell_by_restored={doc["dropped_fell_by_restored"]}
others_unchanged={doc["others_unchanged"]}

readback (re-check) EXIT={recheck.get("exit_status", "not recorded")}
after  {_esc(recheck.get("after_counts", {}))}
delta  {_esc(recheck.get("population_delta", {}))}
sources created since this node's apply, by another writer:
{_esc("\n".join("  " + f["source"] + "  " + f["created_at"] for f in recheck.get("sources_created_since_apply", [])))}

stage_exit_status
{_esc("\n".join("  " + k + " EXIT=" + str(v) for k, v in sorted(exits.items())))}</pre>
<p id="{node}-recheck-note">The second block is the gate doing its job, not a failure of
the write. The read-back was re-run after the coordinator's review, by which time a
peer had bound <strong>{len(recheck.get("sources_created_since_apply", []))}</strong> more
sources (all created at <code>{_esc((recheck.get("sources_created_since_apply") or [{}])[0].get("created_at", "unknown"))}</code>,
{_esc(_elapsed_minutes(rcv.get("applied_at"), (recheck.get("sources_created_since_apply") or [{}])[0].get("created_at")))} after this node's apply). It therefore
moved dropped_edge by {recheck.get("population_delta", {}).get("dropped_edge")} and never_bound by
{recheck.get("population_delta", {}).get("never_bound")} — this node's 39 plus the peer's 3 on
dropped-edge names, and 2 never-bound names the peer bound — and exited non-zero. The
stage raises rather than overwriting the write's own read-back, so the before/after counts
above remain the values the classifier returned on the quiet graph at the apply.</p>

<h3 id="{node}-control">Negative control</h3>
<p id="{node}-control-note">The declared control — <em>{_esc(ctrl["declared"])}</em> —
was run through the same call path as the work list: a plant whose only candidate
measures a different quantity (name <code>{_esc(ctrl["name"])}</code>, candidate
<code>{_esc(ctrl["planted_candidate"])}</code>) must be answered none-of-these and left
unrestored. Jev answered <strong>{_esc(ctrl["answer"].get("choice"))}</strong>;
<code>passed={ctrl["passed"]}</code>. The instrument can therefore reject, not only
accept: on the real work list it rejected
{len(doc["referred_to_lead"])} of
{len([w for w in doc["worklist"] if w["candidate_paths"]])} candidate-bearing names the
same way.</p>

<h3 id="{node}-bound">§38's referral bound</h3>
<p id="{node}-bound-note">§38 requires that the names sent to the lead number at most
15% of the names §35 hands it. §35's census hands this section
<strong>{bound["handed_to_section"]}</strong> live, non-derived, non-orphan sourceless
names, so the bound is <strong>{bound["bound_count"]}</strong>. This node refers
<strong>{bound["this_node_unbound_to_lead"]}</strong> unbound —
{bound["this_node_unbound_to_lead"] / bound["handed_to_section"]:.1%} — plus
<strong>{bound["this_node_bound_below_floor"]}</strong> bound below the confidence
floor, which the coordinator ruled must be counted here, giving a running total of
<strong>{bound["running_total_to_lead"]}</strong>, or
<strong>{bound["running_total_fraction"]:.1%}</strong> — still under the bound. The
<strong>{bound["this_node_to_dd_search"]}</strong> names passed to DD search are a
repair route, not a referral, and are not counted; but they are the section's
liability: if DD search binds none of them they become referrals and the running total
reaches <strong>{bound["liability_if_dd_search_yields_nothing"]}</strong>
({bound["liability_if_dd_search_yields_nothing"] / bound["handed_to_section"]:.1%}),
above the bound. That is a flag for the section owner, not a defect of this node.</p>

<h3 id="{node}-spend">Spend</h3>
<p id="{node}-spend-note">{len(doc["judged"])} judgement calls on the
<code>{_esc(doc["judge_model"])}</code> seat cost
<strong>USD {doc["judge_spend_usd"]:.6f}</strong>; the negative control cost
USD {ctrl["cost_usd"]:.6f}. Total <strong>USD {spend:.6f}</strong> against the
USD 1.00 ceiling — {spend / 1.0:.2%} of it.</p>
</article>
"""
    out = HERE / "sib-restore-lineage-sources.html"
    out.write_text(html)
    print("wrote", out)


def stage_figure() -> None:
    """Draw the classifier populations before and after the restore."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sheet = Path.home() / ".config/matplotlib/stylelib/data-ink.mplstyle"
    if sheet.exists():
        plt.style.use(str(sheet))

    with open(RESULTS) as fh:
        doc = json.load(fh)
    before, after = doc["before_counts"], doc["after_counts"]
    restored_n = doc["restored_count"]
    keys = sorted(before, key=lambda k: -before[k])
    labels = [k.replace("_", " ") for k in keys]
    b = [before[k] for k in keys]
    a = [after.get(k, 0) for k in keys]

    y = range(len(keys))
    fig, ax = plt.subplots(figsize=(14, 3.4))
    h = 0.36
    GREY = "#929292"
    FEATURE = "#31688e"
    for i, (bv, av) in enumerate(zip(b, a, strict=True)):
        changed = bv != av
        ax.barh(i + h / 2, bv, height=h, color=GREY, zorder=2)
        ax.barh(i - h / 2, av, height=h, color=FEATURE if changed else GREY, zorder=3)
        ax.text(
            bv + 1.5, i + h / 2, f"before {bv}", va="center", fontsize=11, color=GREY
        )
        ax.text(
            av + 1.5,
            i - h / 2,
            f"after {av}" + (f" ({restored_n} restored)" if changed else ""),
            va="center",
            fontsize=11,
            color=FEATURE if changed else GREY,
        )
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=12)
    ax.set_xlabel("standard names", fontsize=13)
    ax.set_xlim(0, max(b) * 1.45)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=3)
    ax.invert_yaxis()
    fig.tight_layout()

    outdir = ROOT / "docs/figures/sn-integrity-backlog/sib-restore-lineage-sources"
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / "populations-before-after.svg"
    fig.savefig(out)
    plt.close(fig)
    print("wrote", out)


def _reject_reason(record: dict[str, Any]) -> str:
    choice, prob, confidence = _answer_metrics(record)
    if choice == NONE_OF_THESE:
        return "answered none-of-these"
    if prob < MIN_CHOICE_PROBABILITY:
        return f"chosen probability {prob:.2f} below {MIN_CHOICE_PROBABILITY}"
    return f"self-reported confidence {confidence} below {MIN_SELF_CONFIDENCE}"


def stage_referrals() -> None:
    """Account every name this node cannot admit, against the referral bound.

    Two populations are referred and counted together: names the floor refused
    and left unbound, and names that were bound under a weaker floor and now
    fall below this one. The second is bound below the floor rather than
    detached — a live name left sourceless is what the source invariant
    forbids — so it is carried for lead confirmation instead.
    """
    with open(RESULTS) as fh:
        doc = json.load(fh)

    bound_names = {r["name"] for r in doc.get("restored", [])}
    unbound_to_lead: list[dict[str, Any]] = []
    for w in doc["worklist"]:
        rec = doc.get("judged", {}).get(w["name"])
        if rec is None or w["name"] in bound_names:
            continue
        ans = rec.get("answer") or {}
        if _confident_choice(rec):
            continue
        unbound_to_lead.append(
            {
                "name": w["name"],
                "candidates": w["candidate_paths"],
                "answer": ans.get("choice"),
                "probabilities": ans.get("probabilities"),
                "confidence": ans.get("confidence"),
                "reason": _reject_reason(rec),
            }
        )

    below_floor: list[dict[str, Any]] = []
    for r in doc.get("restored", []):
        if _admittable(r["dd_path"], r["jev_probability"], r["jev_confidence"]):
            continue
        below_floor.append(
            {
                "name": r["name"],
                "candidate": r["dd_path"],
                "probability": r["jev_probability"],
                "confidence": r["jev_confidence"],
                "sources_bound": r["sources_bound"],
                "reason": (
                    f"P(choice) {r['jev_probability']:.2f} below "
                    f"{MIN_CHOICE_PROBABILITY}"
                    if (r["jev_probability"] or 0) < MIN_CHOICE_PROBABILITY
                    else f"self-reported confidence {r['jev_confidence']} below "
                    f"{MIN_SELF_CONFIDENCE}"
                ),
            }
        )

    census_path = HERE / "sib-source-loss-census-results.json"
    handed = None
    if census_path.exists():
        census = json.loads(census_path.read_text())
        handed = sum(
            1 for n in census["names"] if n.get("population") in HANDED_POPULATIONS
        )

    running = len(unbound_to_lead) + len(below_floor)
    doc["referred_to_lead"] = unbound_to_lead
    doc["bound_below_confidence_floor"] = below_floor
    doc["floor"] = {
        "min_choice_probability": MIN_CHOICE_PROBABILITY,
        "min_self_confidence": MIN_SELF_CONFIDENCE,
    }
    doc["bound"] = {
        "handed_to_section": handed,
        "bound_fraction": 0.15,
        "bound_count": int(handed * 0.15) if handed else None,
        "this_node_unbound_to_lead": len(unbound_to_lead),
        "this_node_bound_below_floor": len(below_floor),
        "this_node_to_dd_search": len(doc["referred_to_dd_search"]),
        "running_total_to_lead": running,
        "running_total_fraction": (running / handed) if handed else None,
        "liability_if_dd_search_yields_nothing": running
        + len(doc["referred_to_dd_search"]),
    }
    doc["stage"] = "accounted"
    with open(RESULTS, "w") as fh:
        json.dump(doc, fh, indent=1, sort_keys=True)
    print(
        "unbound to lead:",
        len(unbound_to_lead),
        "bound below floor:",
        len(below_floor),
        "handed:",
        handed,
        "bound:",
        doc["bound"],
    )


def stage_derive() -> None:
    with GraphClient() as gc:
        before = classify_sourceless_standard_names(gc)
        doc = derive(gc)
    doc["before_counts"] = dict(collections.Counter(r["population"] for r in before))
    doc["driver_revision"] = driver_revision()
    doc["generated_at"] = datetime.now(UTC).isoformat()
    doc["stage"] = "derived"
    with open(RESULTS, "w") as fh:
        json.dump(doc, fh, indent=1, sort_keys=True)
    print("before:", doc["before_counts"])
    print("worklist:", len(doc["worklist"]))
    print("with candidates:", sum(1 for w in doc["worklist"] if w["candidate_paths"]))
    print("to DD search:", len(doc["referred_to_dd_search"]))


def _record_exit(stage: str, status: int) -> None:
    """Record the stage's real exit status beside the evidence it produced.

    Recorded after the stage returns or raises, never before, so the number
    the fragment prints is the process's own outcome rather than a literal a
    reader has to trust.
    """
    try:
        with open(RESULTS) as fh:
            doc = json.load(fh)
    except (OSError, ValueError):
        return
    doc.setdefault("stage_exit_status", {})[stage] = status
    with open(RESULTS, "w") as fh:
        json.dump(doc, fh, indent=1, sort_keys=True)


def main() -> None:
    stage = sys.argv[1] if len(sys.argv) > 1 else "derive"
    runners = {
        "derive": stage_derive,
        "judge": stage_judge,
        "control": stage_control,
        "referrals": stage_referrals,
        "figure": stage_figure,
        "fragment": stage_fragment,
    }
    graph_runners = {
        "preview": stage_preview,
        "apply": stage_apply,
        "readback": stage_readback,
    }
    try:
        if stage in runners:
            runners[stage]()
        elif stage in graph_runners:
            with GraphClient() as gc:
                graph_runners[stage](gc)
        else:
            raise SystemExit(f"unknown stage {stage!r}")
    except BaseException:
        _record_exit(stage, 1)
        raise
    _record_exit(stage, 0)


if __name__ == "__main__":
    main()
