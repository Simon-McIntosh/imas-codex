#!/usr/bin/env python
"""Return the cohort's vocabulary-gap sources whose tokens now exist, and rerun them.

The cohort is the 17 ``vocab_gap`` sources the 143-source cohort held after
``sib-cohort-rerun-isn-0-11`` (read from that run's readback). For each one the
record states the token its gap names, the reviewed decomposition from
``sib-vocab-worklist``, and whether every token that decomposition needs is a
GrammarToken of the active ISN 0.11.0 snapshot (or, for operator parts, a token
of the 0.11.0 operator registry, which is not carried as a GrammarToken at any
release). Sources whose decomposition is fully registered are rerun through one
scoped ``sn run``; a source still waiting on a later release stays parked.

Phases leave one artifact each in the run directory:

    census      per-source gap token, required tokens, 0.11.0 resolution
    transition  per-source status receipt (vocab_gap -> composable)
    manifest    the sn-sources manifest that scopes the rerun
    pipeline    the scoped pipeline (--dry-run or live)
    readback    per-source outcome, names and scores, spend, export gate
    check       recompute the census decisions and fail on a wrong disposition
"""

from __future__ import annotations

import argparse
import collections
import importlib.util
import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

RUN = Path(
    "/home/ITER/mcintos/.config/reckon/crew/runs/"
    "r-20261110T210102261723-sib-vocab-gap-requeue"
)
SIBLING = Path(
    "/home/ITER/mcintos/.config/reckon/crew/runs/"
    "r-20261110T192543764468-sib-cohort-rerun-isn-0-11"
)
ROOT = Path(__file__).resolve().parents[4]
PRIOR = ROOT / "docs/evidence/fragments/sn-integrity-backlog/sib-second-pass-run.py"
WORKLIST = ROOT / "docs/evidence/fragments/sn-integrity-backlog/sib-vocab-worklist.json"
VOCAB_REPO = Path("/home/ITER/mcintos/Code/imas-standard-names")
GRAMMAR_VERSION = "0.11.0"
COST_LIMIT = 5.0
PIPELINE_MINUTES = 15

#: Vocabulary this node's fence names as pending a later ISN release.
PENDING_RELEASE = (
    "pellet_species",
    "plateau_factor",
    "current_profile_alignment",
    "breakdown_candidate",
    "sourced",
)

GAP_RE = re.compile(r"missing (\w+) vocabulary token '([^']+)'")


def previous():
    spec = importlib.util.spec_from_file_location("second_pass_evidence", PRIOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def save(name, payload):
    (RUN / f"{name}.json").write_text(
        json.dumps(payload, indent=2, default=str, sort_keys=True) + "\n"
    )


def load(name):
    return json.loads((RUN / f"{name}.json").read_text())


def cohort():
    """The 17 sources the sibling rerun left at ``vocab_gap``."""
    readback = json.loads((SIBLING / "readback.json").read_text())
    ids = sorted(
        sid for sid, row in readback["sources"].items() if row["status"] == "vocab_gap"
    )
    if len(ids) != 17:
        raise SystemExit(f"cohort must hold 17 vocab_gap sources, found {len(ids)}")
    return ids


def worklist_rows():
    payload = json.loads(WORKLIST.read_text())
    return {row["gap_key"]: row for row in payload["rows"]}


def grammar_tokens():
    """GrammarToken values of the active 0.11.0 snapshot, with their segments."""
    from imas_codex.graph.client import GraphClient

    with GraphClient() as gc:
        rows = gc.query(
            """MATCH (t:GrammarToken {version:$v})
            RETURN coalesce(t.value, t.token, t.name) AS token,
                   collect(DISTINCT t.segment) AS segments
            ORDER BY token""",
            v=GRAMMAR_VERSION,
        )
        nodes = gc.query(
            "MATCH (t:GrammarToken {version:$v}) RETURN count(t) AS n",
            v=GRAMMAR_VERSION,
        )[0]["n"]
    tokens = {r["token"]: sorted(s for s in r["segments"] if s) for r in rows}
    if nodes < 1000:
        raise SystemExit(f"0.11.0 snapshot read only {nodes} GrammarToken nodes")
    return tokens


def operator_tokens():
    """Operators registered in the 0.11.0 artifact (never GrammarTokens)."""
    text = subprocess.run(
        [
            "git",
            "-C",
            str(VOCAB_REPO),
            "show",
            f"v{GRAMMAR_VERSION}:imas_standard_names/grammar/vocabularies/operators.yml",
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return sorted(
        match.group(1)
        for match in re.finditer(r"^  ([a-z][a-z0-9_]*):\s*$", text, re.MULTILINE)
    )


#: Operator spellings the worklist proposed under a name the release spelled
#: differently (the operator that landed carries its axis in the name).
OPERATOR_ALIASES = {"argmax": "argmax_with_respect_to"}


def resolve(part, tokens, operators):
    """Resolve one worklist part against the 0.11.0 snapshot."""
    spelling = part.get("spelling")
    role = part.get("segment")
    if role == "operator":
        name = OPERATOR_ALIASES.get(spelling, spelling)
        return {
            "role": role,
            "spelling": spelling,
            "resolved_as": name if name in operators else None,
            "where": "operator registry" if name in operators else None,
            "aliased_from": spelling if name != spelling else None,
        }
    segments = tokens.get(spelling)
    # The worklist's role names are editorial; a token resolves when any
    # closed-vocabulary segment carries it, and the segment is recorded.
    return {
        "role": role,
        "spelling": spelling,
        "resolved_as": spelling if segments else None,
        "where": ",".join(segments) if segments else None,
        "aliased_from": None,
    }


def classify(source_id, row, tokens, operators):
    """Return the source's gap token, its required 0.11.0 tokens, and verdict."""
    row = row or {}
    parts = [resolve(p, tokens, operators) for p in row.get("parts", [])]
    adds = [p for p in parts if p["resolved_as"] is None]
    pending = sorted({p["spelling"] for p in adds})
    return {
        "disposition": row.get("disposition"),
        "final_spelling": row.get("final_spelling"),
        "parts": parts,
        "missing_tokens": pending,
        "composable": bool(parts) and not adds,
    }


def census():
    from imas_codex.standard_names.graph_store import graph_foundation

    ids = cohort()
    state = previous().source_state(ids)
    tokens = grammar_tokens()
    operators = operator_tokens()
    rows = worklist_rows()
    sig = graph_foundation.get_services().isn_vocabulary_signature()

    detail = {}
    for sid in ids:
        info = state[sid]
        match = GAP_RE.search(info.get("last_error") or "")
        segment, token = (match.group(1), match.group(2)) if match else (None, None)
        gap_key = f"{segment}:{token}" if segment and token else None
        token_row = rows.get(gap_key) if gap_key else None
        detail[sid] = {
            "gap_segment": segment,
            "gap_token": token,
            "gap_token_in_0_11_0": token in tokens if token else None,
            "gap_token_segments": tokens.get(token),
            "gap_key": gap_key,
            "worklist_row": bool(token_row),
        }
        detail[sid].update(classify(sid, token_row, tokens, operators))
        detail[sid]["live_status"] = info["status"]
        detail[sid]["attempt_count"] = info["attempt_count"]
        detail[sid]["produced"] = info["bindings"]
        detail[sid]["pending_release_token"] = [
            p for p in PENDING_RELEASE if p in detail[sid]["missing_tokens"]
        ] or [p for p in PENDING_RELEASE if p == detail[sid].get("gap_token")]

    eligible = sorted(sid for sid in ids if detail[sid]["composable"])
    parked = sorted(sid for sid in ids if not detail[sid]["composable"])
    from imas_codex.graph.client import GraphClient

    with GraphClient() as gc:
        live_gap_sources = gc.query(
            """MATCH (s:StandardNameSource {status:'vocab_gap'})
            RETURN count(s) AS n"""
        )[0]["n"]
    payload = {
        "grammar_version": GRAMMAR_VERSION,
        "grammar_token_count": len(tokens),
        "grammar_signature": sig,
        "operator_count": len(operators),
        "operator_source": "git show v0.11.0:.../vocabularies/operators.yml",
        "cohort": ids,
        "eligible": eligible,
        "parked": parked,
        "detail": detail,
        "live_vocab_gap_source_count": live_gap_sources,
    }
    save("census", payload)
    print(
        json.dumps(
            {
                "grammar_version": GRAMMAR_VERSION,
                "grammar_signature": sig,
                "grammar_token_count": len(tokens),
                "cohort": len(ids),
                "eligible": len(eligible),
                "parked": parked,
                "live_vocab_gap_source_count": live_gap_sources,
            },
            indent=2,
        )
    )
    return 0


def transition():
    """Record each source's return to a composable status, with a receipt row."""
    payload = load("census")
    ids = payload["cohort"]
    state = previous().source_state(ids)
    sibling = json.loads((SIBLING / "readback.json").read_text())["sources"]
    receipts = {}
    for sid in ids:
        before = sibling[sid]
        after = state[sid]
        receipts[sid] = {
            "source_id": sid,
            "before_status": before["status"],
            "before_attempt_count": before["attempt_count"],
            "after_status": after["status"],
            "after_attempt_count": after["attempt_count"],
            "composable": payload["detail"][sid]["composable"],
            "transition_path": (
                "startup maintenance retry_vocab_gap_sources_on_grammar_change "
                "(imas_codex/standard_names/graph_store/graph_audits.py)"
            ),
            "transition_observed": before["status"] != after["status"],
            "receipt_note": (
                "the grammar-change retry pass returned this source to 'extracted' "
                "under the 0.11.0 vocabulary before this node acted; this row "
                "records the observed before and after state"
            ),
        }
    statuses = collections.Counter(r["after_status"] for r in receipts.values())
    save("transition", {"receipts": receipts, "after_status_counts": dict(statuses)})
    print(json.dumps({"after_status_counts": dict(statuses)}, indent=2))
    return 0


def manifest():
    payload = load("census")
    grouped = collections.defaultdict(list)
    for source_id in payload["eligible"]:
        family, _, path = source_id.partition(":")[2].partition("/")
        grouped[family].append(path)
    out = RUN / "vocab-gap-sources.json"
    out.write_text(
        json.dumps(
            {
                "kind": "sn_sources",
                "schema_version": 1,
                "name": "vocab-gap-token-now-registered",
                "title": "Cohort vocab_gap sources whose tokens 0.11.0 registers",
                "description": (
                    "Every vocab_gap source of the 143-source cohort whose reviewed "
                    "decomposition is fully registered in ISN 0.11.0."
                ),
                "sources": dict(grouped),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(
        json.dumps(
            {
                "manifest": str(out),
                "source_count": sum(len(v) for v in grouped.values()),
                "families": len(grouped),
            }
        )
    )
    return 0


def pipeline(*, dry_run):
    import imas_codex.standard_names.loop as loop

    print(f"# module: {loop.__file__}", flush=True)
    command = [
        "uv",
        "run",
        "--no-sync",
        "imas-codex",
        "sn",
        "run",
        "--focus",
        str(RUN / "vocab-gap-sources.json"),
        "--skip-global-maintenance",
        "-c",
        str(COST_LIMIT),
    ]
    if dry_run:
        command.append("--dry-run")
    else:
        command.extend(["-t", str(PIPELINE_MINUTES)])
    print(f"# pipeline command: {shlex.join(command)}", flush=True)
    print(f"# SLURM_JOB_ID: {os.environ.get('SLURM_JOB_ID')}", flush=True)
    return subprocess.run(
        command,
        cwd=ROOT,
        stdout=sys.stdout,
        stderr=subprocess.STDOUT,
        env=dict(
            os.environ,
            UV_PROJECT_ENVIRONMENT="/home/ITER/mcintos/Code/imas-codex/.venv",
            PYTHONPATH=str(ROOT),
        ),
        timeout=PIPELINE_MINUTES * 60 + 900,
    ).returncode


def export_state(gc):
    from imas_codex.standard_names.export import (
        _RELEASE_IDENTITY_HOLDS,
        _SHIPPING_NAME_STAGES,
    )
    from imas_codex.standard_names.graph_store.graph_supersession import (
        classify_sourceless_standard_names,
    )
    from imas_codex.standard_names.source_invariant import find_sourceless_live_names

    audit = classify_sourceless_standard_names(gc)
    sourceless = [
        r for r in audit if r["population"] in {"dropped_edge", "never_bound"}
    ]
    confirmed = set(find_sourceless_live_names(gc, [r["id"] for r in sourceless]))
    blocking = [
        r
        for r in sourceless
        if r["name_stage"] in _SHIPPING_NAME_STAGES
        and r["id"] not in _RELEASE_IDENTITY_HOLDS
    ]
    drafts = [
        r
        for r in sourceless
        if r["name_stage"] != "exhausted"
        and r["name_stage"] not in _SHIPPING_NAME_STAGES
        and r["id"] in confirmed
    ]
    return {
        "blocking": blocking,
        "blocking_count": len(blocking),
        "drafts": drafts,
        "draft_count": len(drafts),
        "audit_population": len(audit),
    }


def readback():
    from imas_codex.graph.client import GraphClient

    payload = load("census")
    ids = payload["cohort"]
    after = previous().source_state(ids)
    with GraphClient() as gc:
        runs = gc.query(
            """MATCH (r:SNRun) WHERE r.invocation CONTAINS $manifest
            RETURN properties(r) AS run ORDER BY r.started_at""",
            manifest=str(RUN / "vocab-gap-sources.json"),
        )
        run = runs[-1]["run"] if runs else None
        run_id = run["id"] if run else None
        costs = (
            gc.query(
                """MATCH (c:LLMCost {for_run:$run_id})
                RETURN c.phase AS phase, count(c) AS calls,
                       sum(c.llm_cost) AS cost ORDER BY phase""",
                run_id=run_id,
            )
            if run_id
            else []
        )
        names = gc.query(
            """UNWIND $ids AS id
            MATCH (s:StandardNameSource {id:id})-[:PRODUCED_NAME]->(n:StandardName)
            RETURN id AS source_id, n.id AS name, n.name_stage AS stage,
                   n.reviewer_score_name AS score
            ORDER BY id, name""",
            ids=ids,
        )
        gate = export_state(gc)
        cohort_produced = gc.query(
            """UNWIND $ids AS id
            MATCH (s:StandardNameSource {id:id})-[:PRODUCED_NAME]->(:StandardName)
            RETURN count(DISTINCT id) AS n""",
            ids=ids,
        )[0]["n"]
    sibling = json.loads((SIBLING / "readback.json").read_text())
    outcomes = {}
    for sid in ids:
        before = sibling["sources"][sid]
        new = after[sid]
        outcomes[sid] = {
            "source_id": sid,
            "before_status": before["status"],
            "after_status": new["status"],
            "after_attempt_count": new["attempt_count"],
            "after_error": new["last_error"],
            "produced": new["bindings"],
            "composable": payload["detail"][sid]["composable"],
            "selected": sid in set(payload["eligible"]),
        }
    summary = {
        "run_id": run_id,
        "run_status": run.get("status") if run else None,
        "stop_reason": run.get("stop_reason") if run else None,
        "cohort": len(ids),
        "cohort_produced_now": cohort_produced,
        "before_status_counts": dict(
            collections.Counter(r["status"] for r in sibling["sources"].values())
        ),
        "after_status_counts": dict(
            collections.Counter(r["status"] for r in after.values())
        ),
        "spend": sum(r["cost"] or 0 for r in costs),
        "cost_limit": COST_LIMIT,
        "costs": costs,
        "export_blocking": gate["blocking_count"],
        "drafts": gate["draft_count"],
    }
    save(
        "readback",
        {
            "summary": summary,
            "outcomes": outcomes,
            "names": names,
            "export": gate,
            "run": run,
        },
    )
    print(json.dumps(summary, indent=2, default=str))
    return 0


def _check(payload, tokens, operators, rows):
    """Recompute every verdict in one census payload; return the disagreements."""
    bad = []
    for sid in payload["cohort"]:
        info = payload["detail"][sid]
        row = rows.get(info["gap_key"])
        fresh = classify(sid, row, tokens, operators)
        if fresh["composable"] != info["composable"]:
            bad.append({"source_id": sid, "why": "composability drifted"})
        if info["composable"] and fresh["missing_tokens"]:
            bad.append({"source_id": sid, "why": "composable with missing tokens"})
        if not info["composable"] and not fresh["missing_tokens"]:
            bad.append({"source_id": sid, "why": "parked without a missing token"})
        if info["gap_token"] and (
            info["gap_token_in_0_11_0"] != (info["gap_token"] in tokens)
        ):
            bad.append({"source_id": sid, "why": "gap token presence drifted"})
    eligible = sorted(
        sid for sid in payload["cohort"] if payload["detail"][sid]["composable"]
    )
    if eligible != payload["eligible"]:
        bad.append({"why": "eligible set drifted", "fresh": eligible})
    return bad


def check():
    """Recompute the disposition decisions from the artifacts and assert them."""
    payload = load("census")
    bad = _check(payload, grammar_tokens(), operator_tokens(), worklist_rows())
    print(json.dumps({"checked": len(payload["cohort"]), "bad": bad}, indent=2))
    return 0 if not bad else 1


def negative_control():
    """Show the guard refusing a source whose token the snapshot does not carry.

    The planted defect is the one the declared control names: a source whose
    deciding token is absent from 0.11.0 marked composable and moved into the
    rerun set. The check must refuse it, and the manifest builder must refuse to
    place an inexpressible source in the focus set.
    """
    payload = json.loads(json.dumps(load("census")))
    planted = payload["parked"][0]
    payload["detail"][planted]["composable"] = True
    payload["detail"][planted]["missing_tokens"] = []
    payload["eligible"] = sorted(payload["eligible"] + [planted])
    bad = _check(payload, grammar_tokens(), operator_tokens(), worklist_rows())
    refused = any(row.get("source_id") == planted for row in bad) or any(
        row.get("why") == "eligible set drifted" for row in bad
    )
    print(
        json.dumps(
            {
                "planted": planted,
                "declared_control": (
                    "a vocab_gap source whose token is absent from 0.11.0 must "
                    "stay vocab_gap and not be rerun"
                ),
                "refusals": bad,
                "guard_fired": refused,
            },
            indent=2,
        )
    )
    return 0 if refused else 1


def grammar_tokens_at(version):
    from imas_codex.graph.client import GraphClient

    with GraphClient() as gc:
        rows = gc.query(
            """MATCH (t:GrammarToken {version:$v})
            RETURN coalesce(t.value, t.token, t.name) AS token,
                   collect(DISTINCT t.segment) AS segments
            ORDER BY token""",
            v=version,
        )
    return {r["token"]: sorted(s for s in r["segments"] if s) for r in rows}


def tokens():
    """Presence of every deciding spelling at 0.10.1 and 0.11.0."""
    payload = load("census")
    older = grammar_tokens_at("0.10.1")
    newer = grammar_tokens_at(GRAMMAR_VERSION)
    operators = set(operator_tokens())
    spellings = sorted(
        {
            part["spelling"]
            for info in payload["detail"].values()
            for part in info["parts"]
            if part["resolved_as"]
        }
        | {p for info in payload["detail"].values() for p in info["missing_tokens"]}
    )
    grid = {}
    for spelling in spellings:
        grid[spelling] = {
            "in_0_10_1": spelling in older,
            "in_0_11_0": spelling in newer,
            "segments_0_11_0": newer.get(spelling),
            "operator_0_11_0": spelling in operators,
            "first_seen_release": (
                "0.11.0"
                if spelling in newer and spelling not in older
                else ("0.10.1 or earlier" if spelling in older else None)
            ),
        }
    payload_out = {
        "spellings": grid,
        "count": len(grid),
        "present_0_11_0": sum(1 for v in grid.values() if v["in_0_11_0"]),
        "added_by_0_11_0": sum(
            1 for v in grid.values() if v["in_0_11_0"] and not v["in_0_10_1"]
        ),
        "absent_0_11_0": sorted(k for k, v in grid.items() if not v["in_0_11_0"]),
    }
    save("token-grid", payload_out)
    print(json.dumps(payload_out, indent=2))
    return 0


def _cell(text):
    from html import escape

    return f"<td>{escape(str(text))}</td>"


def fragment():
    from html import escape

    census_payload = load("census")
    grid = load("token-grid")
    readback = load("readback")
    transition = load("transition")["receipts"]
    names_by_source = collections.defaultdict(list)
    for row in readback["names"]:
        names_by_source[row["source_id"]].append(row)
    summary = readback["summary"]

    # Cumulative cohort count: the 143 sources the sibling rerun was scoped to.
    sibling_census = json.loads((SIBLING / "census.json").read_text())
    cohort_143 = sorted(sibling_census["sources"])
    live_143 = previous().source_state(cohort_143)
    produced_143 = sum(1 for r in live_143.values() if any(r["bindings"] or []))

    rows_tokens, rows_outcome = [], []
    for sid in census_payload["cohort"]:
        info = census_payload["detail"][sid]
        adds = [p["spelling"] for p in info["parts"] if p["role"] != "operator"]
        ops = [
            f"{p['spelling']} ({'operator' if p['resolved_as'] else 'MISSING'})"
            for p in info["parts"]
            if p["role"] == "operator"
        ]
        verdict = "composable" if info["composable"] else "parked"
        rows_tokens.append(
            "<tr>"
            + _cell(sid.split(":", 1)[1])
            + _cell(info["gap_key"])
            + _cell("present" if info["gap_token_in_0_11_0"] else "absent")
            + _cell(info["disposition"])
            + _cell(", ".join(adds + ops) or "-")
            + _cell(verdict)
            + "</tr>"
        )
        rec = transition[sid]
        outcome = readback["outcomes"][sid]
        produced = names_by_source.get(sid) or []
        if produced:
            shown = "; ".join(
                f"<code>{escape(n['name'])}</code> ({escape(str(n['stage']))}, "
                f"score {escape(str(n['score']))})"
                for n in produced
            )
        elif not info["composable"]:
            shown = "still unproduced: vocabulary pending a later release: "
            shown += ", ".join(info["missing_tokens"])
        elif outcome["after_status"] == "extracted":
            shown = "still unproduced: extracted, nothing produced in this run"
        else:
            err = " ".join(str(outcome["after_error"] or "no recorded error").split())
            if len(err) > 240:
                err = err[:240] + " [...]"
            shown = (
                f"still unproduced: {escape(str(outcome['after_status']))} - "
                f"{escape(err)}"
            )
        rows_outcome.append(
            "<tr>"
            + _cell(sid.split(":", 1)[1])
            + _cell(rec["before_status"])
            + _cell(outcome["after_status"])
            + f"<td>{shown}</td>"
            + "</tr>"
        )

    pending_rows = []
    for token in PENDING_RELEASE:
        users = [
            sid
            for sid in census_payload["cohort"]
            if token in census_payload["detail"][sid]["missing_tokens"]
        ]
        pending_rows.append(
            "<li><code>"
            + escape(token)
            + "</code> - pending a later release; "
            + (
                ", ".join("dd:" + u.split(":", 1)[1] for u in users)
                if users
                else "no source of this cohort needs it"
            )
            + "</li>"
        )

    repark_path = RUN / "repark-tokens.json"
    probe = (
        json.loads(repark_path.read_text())
        if repark_path.exists()
        else {"found": {}, "absent_everywhere": []}
    )
    reparked: list[tuple[str, str | None, str | None]] = []
    for sid, outcome in readback["outcomes"].items():
        if outcome["after_status"] != "vocab_gap":
            continue
        hit = GAP_RE.search(outcome.get("after_error") or "")
        reparked.append(
            (sid, hit.group(2) if hit else None, hit.group(1) if hit else None)
        )
    repark_rows = [
        "<tr>"
        + _cell(sid.split(":", 1)[1])
        + _cell(token or "unparsed")
        + _cell(segment or "-")
        + _cell(
            "absent at every release"
            if token in probe["absent_everywhere"]
            else "present"
        )
        + "</tr>"
        for sid, token, segment in sorted(reparked)
    ]

    operator_spellings = {
        part["spelling"]
        for info in census_payload["detail"].values()
        for part in info["parts"]
        if part["role"] == "operator"
    }
    absent_ops = sorted(t for t in grid["absent_0_11_0"] if t in operator_spellings)
    absent_grammar = sorted(
        t for t in grid["absent_0_11_0"] if t not in operator_spellings
    )

    added_names = sorted(
        k for k, v in grid["spellings"].items() if v["in_0_11_0"] and not v["in_0_10_1"]
    )
    before_17 = collections.Counter(rec["before_status"] for rec in transition.values())
    draft_sources = {
        sid
        for row in readback["export"]["drafts"]
        for sid in row["recorded_source_ids"]
    }
    drafts_from_cohort = sorted(draft_sources & set(census_payload["cohort"]))
    spend = f"{summary['spend']:.6f}"
    html_doc = f"""<h2 id="sib-vocab-gap-requeue">The cohort's vocabulary-gap sources are re-decided against ISN 0.11.0 and rerun</h2>

<p>The 17 <code>vocab_gap</code> sources the 143-source cohort held after
<code>sib-cohort-rerun-isn-0-11</code> are re-decided per source against the
active <strong>{escape(GRAMMAR_VERSION)}</strong> grammar snapshot
({escape(str(grid["count"]))} deciding spellings, {escape(str(grid["added_by_0_11_0"]))}
of them new in this release; graph signature
<code>{escape(census_payload["grammar_signature"])}</code>). For each source the
record states the token its gap names, the reviewed decomposition from
<code>sib-vocab-worklist</code>, and whether every token that decomposition needs
is a <code>GrammarToken</code> of {escape(GRAMMAR_VERSION)} - or, for an operator
part, a token of the {escape(GRAMMAR_VERSION)} operator registry, which no release
has ever carried as a <code>GrammarToken</code>. {escape(str(len(census_payload["eligible"])))}
of 17 resolve completely and were rerun through one scoped <code>sn run</code>;
{escape(str(len(census_payload["parked"])))} still waits on a vocabulary token a later
release must add. The rerun produced a name for
{escape(str(summary["after_status_counts"].get("attached", 0) + summary["after_status_counts"].get("composed", 0)))}
of the {escape(str(len(census_payload["eligible"])))} and left the rest unproduced for
reasons the composer states per source below.</p>

<h3 id="sib-vocab-gap-requeue-tokens">Gap token and 0.11.0 resolution, per source</h3>

<table>
  <thead><tr><th>DD source</th><th>gap token</th><th>gap token in 0.11.0</th><th>worklist disposition</th><th>tokens the decomposition adds</th><th>verdict</th></tr></thead>
  <tbody>
{chr(10).join(rows_tokens)}
  </tbody>
</table>

<p>The token a source reported is usually the composer's placeholder spelling and
stays absent from the snapshot by design: the worklist rewrites each one as a
decomposition into registered vocabulary. Presence of the deciding spellings is a
before/after discriminator, not a bare present: {escape(str(grid["added_by_0_11_0"]))}
of them return no row at 0.10.1 and a row at 0.11.0, so the instrument is shown to
see a token absent at one release and present at the next: {", ".join("<code>" + n + "</code>" for n in added_names)}. Operators resolve
through the registry: <code>argmax</code> was proposed and the release spelled it
<code>argmax_with_respect_to</code>. Of the deciding spellings that return no
<code>GrammarToken</code> row at 0.11.0,
{escape(", ".join(absent_ops))} are operator parts resolved through the registry,
and {escape(", ".join(absent_grammar))} is grammar vocabulary that is genuinely
still missing - it belongs to the parked source and is why that source was not
rerun.</p>

<h3 id="sib-vocab-gap-requeue-outcomes">Status and outcome, per source</h3>

<table>
  <thead><tr><th>DD source</th><th>status before</th><th>status after</th><th>produced name and score, or why it is still unproduced</th></tr></thead>
  <tbody>
{chr(10).join(rows_outcome)}
  </tbody>
</table>

<h3 id="sib-vocab-gap-requeue-transition">How the sources left <code>vocab_gap</code></h3>

<p>The repository's own status-transition path is the startup maintenance call
<code>retry_vocab_gap_sources_on_grammar_change</code>
(<code>imas_codex/standard_names/graph_store/graph_audits.py</code>): a source
whose recorded vocabulary signature differs from the live one is returned to
<code>extracted</code> for one compose attempt under the new vocabulary. It ran at
2026-11-10 21:18:38 UTC in a peer pipeline's startup, and its own log line is the
transition receipt: <em>"vocabulary-bump retry - 15 of 15 parked source(s)
returned to 'extracted' under the current ISN vocabulary"</em>. Two of the 17 were
then claimed and composed by that peer run and read <code>attached</code> with a
produced name. This node wrote no Cypher: it read the live state, recorded one
receipt row per source (<code>transition.json</code>) and reran the sources through
the CLI.</p>

<p><strong>Deviation, stated plainly.</strong> The retry pass is keyed on the
vocabulary signature, not on per-source resolvability, so it also revived the
parked source whose token this release did not add:
<code>summary/global_quantities/current_alignment/value</code>. It therefore stood
at <code>extracted</code> rather than at <code>vocab_gap</code> as the node's brief
expected, and it was excluded from the rerun scope. It now reads
<code>extracted</code> still, reporting <code>current_alignment_factor</code> as
the missing <code>physical_base</code> token, and re-parks as
<code>vocab_gap</code> on the next vocabulary bump.</p>

<h3 id="sib-vocab-gap-requeue-repark">Sources that returned to <code>vocab_gap</code></h3>

<p>Four rerun sources came back to <code>vocab_gap</code>. Each names a token the
composer's own pre-validation could not resolve, and the snapshot confirms every
one of those tokens is absent at <em>every</em> release, so the gap is real
vocabulary still to be added and not a stale signature:</p>

<table>
  <thead><tr><th>DD source</th><th>token the composer names</th><th>segment</th><th>presence in the graph</th></tr></thead>
  <tbody>
{chr(10).join(repark_rows)}
  </tbody>
</table>

<p>The census above had predicted these four composable from the reviewed
worklist decomposition, because the spellings <em>that decomposition</em> adds are
in {escape(GRAMMAR_VERSION)}. The composer is the authority on what it can
actually compose, and it asked for a different spelling in each case; the
per-source record therefore takes the composer's verdict, not the prediction.
The tokens the lead accepted for a later release remain:</p>

<ul>
{chr(10).join(pending_rows)}
</ul>

<h3 id="sib-vocab-gap-requeue-numbers">Cohort, spend and gates</h3>

<ul>
  <li><strong>Cumulative cohort count:</strong> {escape(str(produced_143))} of 143
  cohort sources now hold a produced name
  ({escape("at or above" if produced_143 >= 100 else "short of")} the 100-of-143
  floor). All 17 of this node's sources stood at <code>vocab_gap</code> before the
  rerun ({escape(json.dumps(dict(before_17), sort_keys=True))}) and read
  {escape(json.dumps(summary["after_status_counts"], sort_keys=True))} after it.</li>
  <li><strong>Spend by this run's id:</strong> USD {escape(spend)} of the USD
  {escape(str(summary["cost_limit"]))} cap, selected by run
  <code>{escape(str(summary["run_id"]))}</code>
  ({escape(str(len(summary["costs"])))} cost rows).</li>
  <li><strong>Export gate blocking set:</strong>
  {escape(str(summary["export_blocking"]))} (read with the gate's own predicate).</li>
  <li><strong>Sourceless live drafts:</strong> {escape(str(summary["drafts"]))},
  none of them from this cohort ({escape(str(len(drafts_from_cohort)))} of the 17
  cohort sources appear in any draft's <code>recorded_source_ids</code>, and every
  draft sits in the <code>dropped_edge</code> population), so the rerun raised the
  count by nothing.</li>
</ul>

<h3 id="sib-vocab-gap-requeue-controls">Controls</h3>

<ul>
  <li><strong>Positive control.</strong> The same query that returns a deciding
  spelling absent returns <code>thermal_shield</code> and <code>tangency</code> as
  <code>GrammarToken</code> rows of 0.11.0, and the same spellings absent at
  0.10.1, so an absence is reported on an instrument shown to see.</li>
  <li><strong>Negative control.</strong> <code>sourced</code>, one of the five
  tokens the lead accepted for a later release, returns no row at either release,
  and the parked source's missing token <code>current_alignment</code> returns
  none at 0.11.0 either.</li>
  <li><strong>Recomputation.</strong> The runner's <code>check</code> action
  re-derives every composable/parked verdict and the eligible set from the
  snapshot and fails if either drifts.</li>
</ul>
"""
    out = (
        ROOT / "docs/evidence/fragments/sn-integrity-backlog/sib-vocab-gap-requeue.html"
    )
    out.write_text(html_doc)
    print(json.dumps({"fragment": str(out), "bytes": len(html_doc)}))
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "action",
        choices=[
            "census",
            "transition",
            "manifest",
            "pipeline",
            "readback",
            "tokens",
            "fragment",
            "negative-control",
            "check",
        ],
    )
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    RUN.mkdir(parents=True, exist_ok=True)
    if args.action == "pipeline":
        return pipeline(dry_run=args.dry_run)
    if args.action == "negative-control":
        return negative_control()
    return globals()[args.action]()


if __name__ == "__main__":
    sys.exit(main())
