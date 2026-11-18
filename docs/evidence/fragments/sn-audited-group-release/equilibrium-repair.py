"""Record source-grounded proposals and verify their accepted graph identities.

Only the public CLI mutates the graph. The source invariant's canonical reader
supplies every command's before/after count. Full receipts remain in the run
folder; the compact record is independently inspectable with --show.
"""

from __future__ import annotations

import argparse
import base64
import importlib.util
import json
import os
import subprocess
import sys
import zlib
from collections import Counter
from datetime import UTC, datetime
from html import escape
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "equilibrium-repair.json"
PAGE = HERE / "equilibrium-repair.html"
MAIN = Path("/home/ITER/mcintos/Code/imas-codex")
RUN = Path(
    "/home/ITER/mcintos/.config/reckon/crew/runs/"
    "r-20261117T225452416830-sagr-equilibrium-agent-repair"
)
CEILING = 30.0
EXPECTED_SOURCELESS = 10
IDENTITY_FIELDS = (
    "id",
    "description",
    "documentation",
    "name_stage",
    "docs_stage",
    "edit_status",
    "reviewer_score_name",
    "reviewer_score_docs",
    "successor",
)


def now():
    return datetime.now(UTC).isoformat()


def revision(tree=MAIN):
    return subprocess.check_output(
        ["git", "-C", str(tree), "rev-parse", "HEAD"], text=True
    ).strip()


def audit():
    spec = importlib.util.spec_from_file_location(
        "equilibrium_adjudication", HERE / "equilibrium-adjudication.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load(path=DATA):
    raw = json.loads(Path(path).read_text())
    payload = raw.pop("payload_zlib_base85", None)
    return raw | (
        json.loads(zlib.decompress(base64.b85decode(payload))) if payload else {}
    )


def save(record, path=DATA):
    raw = {k: v for k, v in record.items() if k not in ("members", "commands")}
    raw["counts"] = counts(record)
    raw["payload_zlib_base85"] = base64.b85encode(
        zlib.compress(
            json.dumps(
                {k: record[k] for k in ("members", "commands")}, ensure_ascii=False
            ).encode(),
            9,
        )
    ).decode()
    content = json.dumps(raw, ensure_ascii=False, indent=2) + "\n"
    assert len(content.encode()) <= 300_000, "Move oversized evidence to run directory"
    Path(path).write_text(content)


def client():
    from imas_codex.graph.client import GraphClient

    return GraphClient()


def sourceless(gc):
    from imas_codex.standard_names.source_invariant import (
        find_sourceless_live_name_rows,
    )

    rows = find_sourceless_live_name_rows(gc, derived_parents_exempt=False)
    control = list(
        gc.query(
            "MATCH (sn:StandardName) WHERE sn.id IN $ids "
            "RETURN count(sn) AS candidates, count(sn.id) AS keyed",
            ids=["electron_temperature", "elongation_of_plasma_boundary"],
        )
    )[0]
    assert control["candidates"] == control["keyed"] == 2, control
    return {"at": now(), "count": len(rows), "rows": rows, "known_present": control}


def identities(gc, ids):
    rows = list(
        gc.query(
            "MATCH (sn:StandardName) WHERE sn.id IN $ids "
            "OPTIONAL MATCH (s:StandardNameSource)-[:PRODUCED_NAME]->(sn) "
            "WHERE coalesce(s.status, '') <> 'stale' "
            "WITH sn, collect(s.id) AS sources "
            "OPTIONAL MATCH (sn)-[:HAS_UNIT]->(u:Unit) "
            "WITH sn, sources, collect(u.id) AS units "
            "OPTIONAL MATCH (sn)-[:HAS_PHYSICS_DOMAIN]->(p:PhysicsDomain) "
            "RETURN properties(sn) AS props, sources, units, collect(p.id) AS domains",
            ids=sorted(set(ids)),
        )
    )
    return {
        row["props"]["id"]: {
            **{k: row["props"].get(k) for k in IDENTITY_FIELDS},
            "sources": sorted(row["sources"]),
            "units": sorted(row["units"]),
            "domains": sorted(row["domains"]),
        }
        for row in rows
    }


def source_bindings(gc, source_ids):
    rows = list(
        gc.query(
            "UNWIND $ids AS sid MATCH (s:StandardNameSource {id: sid}) "
            "OPTIONAL MATCH (s)-[:PRODUCED_NAME]->(sn:StandardName) "
            "WHERE NOT coalesce(sn.name_stage, '') IN ['superseded', 'exhausted'] "
            "RETURN sid AS source, collect(sn.id) AS names",
            ids=source_ids,
        )
    )
    return {r["source"]: sorted(r["names"]) for r in rows}


def initialize():
    assert not DATA.exists(), "Record already exists; do not replace it"
    module = audit()
    original = module.load()
    judgments = {r["id"]: r for r in original["judgments"] if r["verdict"] != "sound"}
    causes = original["report"]["candidate_pipeline_defects"]
    order = list(
        dict.fromkeys(
            [name for c in causes for name in c["members"] if name in judgments]
            + [name for name, j in judgments.items() if j["verdict"] == "defective"]
            + [name for name, j in judgments.items() if j["verdict"] == "undecidable"]
        )
    )
    assert len(order) == 126
    with client() as gc:
        count = sourceless(gc)
        initial = identities(gc, order)
    assert count["count"] == EXPECTED_SOURCELESS, count
    record = {
        "created_at": now(),
        "base_revision": revision(),
        "ceiling_usd": CEILING,
        "spent_usd": 0.0,
        "baseline_sourceless": count,
        "commands": [],
        "causes": causes,
        "class_list": original["class_list"],
        "pilot": {},
        "pending_excluded": [p["id"] for p in original["pending"]],
        "members": [
            {
                "id": name,
                "judgment": judgments[name],
                "causes": [c["cause"] for c in causes if name in c["members"]],
                "before": initial.get(name),
                "after": initial.get(name),
                "proposals": [],
                "outcome": "pending",
                "cost_usd": 0.0,
                "lead_question": "",
            }
            for name in order
        ],
    }
    save(record)
    render(record)
    print(json.dumps({"members": len(order), "sourceless": count["count"]}))


def reviews(gc, ids, since):
    return list(
        gc.query(
            "MATCH (sn:StandardName)-[:HAS_REVIEW]->(r:StandardNameReview) "
            "WHERE sn.id IN $ids AND toString(r.reviewed_at) >= $since "
            "RETURN sn.id AS name, r.id AS id, r.review_axis AS axis, "
            "r.score AS score, r.comments AS comments, r.model AS model, "
            "r.review_group_id AS review_group ORDER BY r.reviewed_at",
            ids=ids,
            since=since,
        )
    )


def command(member, arguments):
    record = load()
    row = next(r for r in record["members"] if r["id"] == member)
    assert record["spent_usd"] < CEILING, "Spend ceiling reached"
    stamp = now()
    receipt_path = RUN / f"command-{len(record['commands']):04d}.json"
    argv = [
        "uv",
        "run",
        "--no-sync",
        "--directory",
        str(MAIN),
        "imas-codex",
        "sn",
        *arguments,
    ]
    with client() as gc:
        before = sourceless(gc)
    assert before["count"] == EXPECTED_SOURCELESS, before
    receipt = {
        "member": member,
        "argv": argv,
        "started_at": stamp,
        "main_revision": revision(),
        "sourceless_before": before,
        "state": "started",
    }
    receipt_path.write_text(json.dumps(receipt, indent=2, default=str))
    record["commands"].append(str(receipt_path))
    save(record)
    env = os.environ | {
        "PYTHONPATH": str(MAIN),
        "UV_PROJECT_ENVIRONMENT": str(MAIN / ".venv"),
    }
    result = subprocess.run(argv, cwd=MAIN, env=env, timeout=1200, check=False)
    with client() as gc:
        after = sourceless(gc)
        ids = [row["id"]]
        if "--rename" in arguments:
            ids.append(arguments[arguments.index("--rename") + 1])
        bindings = source_bindings(gc, (row.get("before") or {}).get("sources", []))
        ids = sorted(set(ids) | {n for names in bindings.values() for n in names})
        state = identities(gc, ids)
        review_rows = reviews(gc, ids, stamp)
        costs = list(
            gc.query(
                "MATCH (c:LLMCost) WHERE toString(c.llm_at) >= $since "
                "RETURN c.id AS id, c.run_id AS run_id, c.llm_cost AS cost, "
                "c.standard_name_ids AS names",
                since=stamp,
            )
        )
    spent = sum(float(c["cost"] or 0.0) for c in costs)
    receipt.update(
        {
            "state": "finished",
            "finished_at": now(),
            "exit_status": result.returncode,
            "sourceless_after": after,
            "identities": state,
            "source_bindings": bindings,
            "reviews": review_rows,
            "cost_rows": costs,
            "cost_usd": spent,
        }
    )
    receipt_path.write_text(json.dumps(receipt, indent=2, default=str))
    row["cost_usd"] += spent
    record["spent_usd"] += spent
    row["proposals"].append(
        {
            "receipt": str(receipt_path),
            "arguments": arguments,
            "exit_status": result.returncode,
            "cost_usd": spent,
            "reviews": review_rows,
        }
    )
    current = state.get((row.get("after") or {}).get("id")) or state.get(row["id"])
    if "--rename" in arguments and "--dry-run" not in arguments:
        candidate = state.get(arguments[arguments.index("--rename") + 1])
        if current and current["name_stage"] == "superseded" and candidate:
            current = candidate
    row["after"] = current
    row["source_bindings"] = bindings
    row["related_identities"] = {
        n: state[n] for names in bindings.values() for n in names if n in state
    }
    save(record)
    render(record)
    assert after["count"] == EXPECTED_SOURCELESS, after
    assert record["spent_usd"] <= CEILING, "Spend ceiling exceeded"
    print(
        json.dumps(
            {
                "receipt": str(receipt_path),
                "exit_status": result.returncode,
                "cost_usd": spent,
                "total_usd": record["spent_usd"],
                "sourceless": after["count"],
            }
        )
    )
    return result.returncode


def counts(record):
    def tally(rows):
        outcomes = Counter(r["outcome"] for r in rows)
        return {
            "repaired": outcomes["accepted_repair"],
            "refused": outcomes["refused"],
            "left_for_lead": outcomes["lead"],
            "pending": outcomes["pending"],
        }

    by_class = {
        c: tally(r for r in record["members"] if c in r["judgment"]["defect_classes"])
        for c, _ in record["class_list"]
    }
    by_cause = {
        c["cause"]: tally(r for r in record["members"] if r["id"] in c["members"])
        for c in record["causes"]
    }
    return {
        "outcomes": dict(Counter(r["outcome"] for r in record["members"])),
        "classes": by_class,
        "causes": by_cause,
    }


def render(record):
    def e(x):
        return escape(str(x))

    summary_tables = []
    for label, groups in (
        ("Defect class", counts(record)["classes"]),
        ("Shared cause", counts(record)["causes"]),
    ):
        table = [
            f"<table><thead><tr><th>{label}</th><th>Repaired</th>"
            "<th>Refused</th><th>For the lead</th><th>Pending</th></tr></thead><tbody>"
        ]
        for group, values in groups.items():
            table.append(
                f"<tr><td>{e(group)}</td>"
                + "".join(
                    f"<td>{values[key]}</td>"
                    for key in ("repaired", "refused", "left_for_lead", "pending")
                )
                + "</tr>"
            )
        table.append("</tbody></table>")
        summary_tables.extend(table)
    content = [
        "<!doctype html><html lang='en'><meta charset='utf-8'>",
        "<meta name='reckon-type' content='evidence'><title>Equilibrium repair record</title>",
        "<link rel='stylesheet' href='/_shared/foundation.css'><main>",
        "<style>table{table-layout:fixed;width:100%}td,th{vertical-align:top;"
        "overflow-wrap:anywhere}pre{white-space:pre-wrap;overflow-wrap:anywhere}"
        "code{overflow-wrap:anywhere}</style>",
        "<h1>Equilibrium repair record</h1>",
        f"<p><strong>{e(record.get('state', 'in progress'))}</strong>: "
        f"{e(record.get('checkpoint', 'Repair work is incomplete.'))}</p>",
        f"<p>Recorded spend USD {record['spent_usd']:.6f} / USD {CEILING:.2f}. "
        f"Initial live names without a source: {record['baseline_sourceless']['count']}. "
        "Pipeline acceptance is recorded here; correctness awaits a fresh blind adjudication.</p>",
        "<p>Scores describe the stored identity. A held member with no submitted command "
        "has no fresh review claimed. Refused counts mean two scored name-review refusals; "
        "deterministic vetoes and source questions are included among members left for the lead.</p>",
        f"<p>{e(record.get('pilot', {}).get('result', 'Pilot not completed.'))}</p>",
        f"<p>Member outcomes: {e(counts(record)['outcomes'])}. "
        "Class and cause counts may overlap.</p>",
        *summary_tables,
        "<p>The two pending adjudications remain excluded. Full command receipts, source packets, "
        f"and guard measurements are preserved in <code>{e(RUN)}</code>. "
        "Receipt basenames below resolve in that directory; --show decodes the full JSON record.</p>",
        "<table><thead><tr><th>Before → after</th><th>Source and adjudication</th>"
        "<th>Proposal and review</th><th>Outcome / cost / question for the lead</th></tr></thead><tbody>",
    ]
    for r in record["members"]:
        after = r.get("after") or {}
        summaries = []
        for proposal in r["proposals"]:
            args = proposal["arguments"]
            text = " ".join(args[:2])
            if "--dry-run" in args:
                text += " (dry run)"
            elif "--rename" in args:
                text += " → " + args[args.index("--rename") + 1]
            text += "\nReceipt: " + Path(proposal["receipt"]).name
            scores = [
                (v["axis"], v["score"])
                for v in proposal["reviews"]
                if v.get("score") is not None
            ]
            text += f"\nreview scores: {scores}; exit: {proposal['exit_status']}; USD {proposal['cost_usd']:.6f}"
            summaries.append(text)
        proposals = "\n\n".join(summaries)
        if r.get("proposal_reason"):
            proposals += "\nProposal reason: " + r["proposal_reason"]
        if r.get("intended_proposal"):
            proposals += "\nProposed correction: " + r["intended_proposal"]
        if r.get("parked_gate"):
            proposals += "\nRefusing gate: " + r["parked_gate"]
        if r.get("source_words"):
            proposals += "\nSource words: " + r["source_words"]
        if r.get("parked_state"):
            proposals += "\nParked live state: " + r["parked_state"]
        if r.get("refused_documentation_attempts"):
            proposals += "\nDocumentation refusal attempts: " + json.dumps(
                r["refused_documentation_attempts"], ensure_ascii=False
            )
        detail = ""
        if r["outcome"] == "accepted_repair":
            detail = (
                "<details><summary>Stored documentation and source bindings</summary>"
                f"<pre>{e(after.get('documentation'))}\n\n"
                f"{e(json.dumps(r.get('source_bindings', {}), indent=2))}</pre></details>"
            )
        if r["outcome"] == "accepted_repair":
            for name, related in r.get("related_identities", {}).items():
                if name != after.get("id"):
                    detail += (
                        f"<p>Related accepted identity: <code>{e(name)}</code>. "
                        f"{e(related['description'])} Name/docs scores: "
                        f"{e(related['reviewer_score_name'])} / "
                        f"{e(related['reviewer_score_docs'])}. "
                        f"Sources: {e(related['sources'])}.</p>"
                    )
        content.append(
            f"<tr><td><code>{e(r['id'])}</code> → <code>{e(after.get('id'))}</code>"
            f"<p>{e((r.get('before') or {}).get('description'))}</p><p>{e(after.get('description'))}</p>"
            f"<p>Units: {e(after.get('units'))}; domains: {e(after.get('domains'))}. "
            f"Name score: {e(after.get('reviewer_score_name'))}; "
            f"documentation score: {e(after.get('reviewer_score_docs'))}.</p>"
            f"{detail}"
            f"</td><td>{e(r['judgment']['evidence_path'])}<p>{e(r['judgment']['reason'])}</p>"
            f"</td><td><pre>{e(proposals)}</pre></td><td>{e(r['outcome'])}; USD {r['cost_usd']:.6f}"
            f"<p>{e(r['lead_question'])}</p></td></tr>"
        )
    content.append("</tbody></table></main></html>\n")
    PAGE.write_text("\n".join(content))


def check(path=DATA, *, checkpoint=False):
    print(
        json.dumps(
            {
                "module": __file__,
                "cwd": str(Path.cwd().resolve()),
                "main_revision": revision(),
            }
        )
    )
    from imas_codex.standard_names import source_invariant

    print(json.dumps({"source_reader_module": source_invariant.__file__}))
    record = load(path)
    module = audit()
    expected = {j["id"] for j in module.load()["judgments"] if j["verdict"] != "sound"}
    rows = record["members"]
    assert {r["id"] for r in rows} == expected and len(rows) == 126, "Cohort mismatch"
    assert record["ceiling_usd"] == CEILING and record["spent_usd"] <= CEILING, (
        "Spend ceiling"
    )
    assert record["counts"] == counts(record), "Outcome totals differ from member rows"
    assert abs(sum(r["cost_usd"] for r in rows) - record["spent_usd"]) < 1e-8
    repaired = [r for r in rows if r["outcome"] == "accepted_repair"]
    with client() as gc:
        guard = sourceless(gc)
        assert guard["count"] == EXPECTED_SOURCELESS, guard
        live = identities(
            gc,
            [r["after"]["id"] for r in repaired]
            + [n for r in repaired for n in r.get("related_identities", {})],
        )
        bindings = source_bindings(
            gc,
            sorted(
                {
                    p
                    for r in repaired
                    for p in (r.get("before") or {}).get("sources", [])
                }
            ),
        )
    for r in repaired:
        target = live.get(r["after"]["id"])
        assert target == r["after"], (
            r["id"],
            "Stored identity differs from repair record",
        )
        assert target["name_stage"] == "accepted", (r["id"], "Name not accepted")
        assert target["sources"], (r["id"], "Missing producing source")
        assert r["proposals"], (r["id"], "No signed proposal receipt")
        assert target["docs_stage"] == "accepted", (r["id"], "Docs not accepted")
        for source, names in r.get("source_bindings", {}).items():
            assert names and bindings.get(source) == names, (source, "Binding differs")
        for name, identity in r.get("related_identities", {}).items():
            assert live.get(name) == identity, (name, "Related identity differs")
            assert (
                identity["name_stage"] == identity["docs_stage"] == "accepted"
                and identity["sources"]
            ), name
    cost_ids = set()
    receipt_spend = 0.0
    positive_cost_rows = zero_cost_rows = 0
    for path in record["commands"]:
        receipt = json.loads(Path(path).read_text())
        assert receipt["state"] == "finished", (path, "Unfinished command")
        for k in ("sourceless_before", "sourceless_after"):
            assert receipt[k]["count"] == EXPECTED_SOURCELESS, (path, k)
        for cost in receipt["cost_rows"]:
            assert cost["cost"] is not None, (path, "Unknown cost is not zero")
            assert cost["id"] not in cost_ids, (path, "Duplicate cost receipt")
            cost_ids.add(cost["id"])
            positive_cost_rows += cost["cost"] > 0
            zero_cost_rows += cost["cost"] == 0
        observed = sum(float(c["cost"]) for c in receipt["cost_rows"])
        assert abs(observed - receipt["cost_usd"]) < 1e-8, (path, "Cost differs")
        receipt_spend += observed
    assert abs(receipt_spend - record["spent_usd"]) < 1e-8, "Unreconciled spend"
    assert positive_cost_rows, "Cost reader must see known paid reviews"
    for r in rows:
        assert r["outcome"] in ("accepted_repair", "lead", "refused", "pending"), (
            r["id"],
            r["outcome"],
        )
        if r["outcome"] in ("lead", "refused"):
            assert r["lead_question"].strip(), (r["id"], "No precise lead question")
            assert r.get("intended_proposal"), (r["id"], "No proposed correction")
            assert r.get("source_words"), (r["id"], "No deciding source words")
            assert r.get("parked_gate"), (r["id"], "No reason proposal cannot proceed")
    print(
        json.dumps(
            {
                "checked": len(rows),
                "complete": not any(r["outcome"] == "pending" for r in rows),
                "repaired": len(repaired),
                **counts(record),
                "sourceless": guard["count"],
                "spent_usd": record["spent_usd"],
                "ceiling_usd": CEILING,
                "cost_receipts": len(cost_ids),
                "positive_cost_rows": positive_cost_rows,
                "zero_cost_rows": zero_cost_rows,
            }
        )
    )
    if not checkpoint:
        pending = [r["id"] for r in rows if r["outcome"] == "pending"]
        assert not pending, f"Incomplete repair cohort: {len(pending)} pending members"
        pilot = record["pilot"]
        assert pilot["completed"] and pilot["accepted"] == len(pilot["members"]) == 6
        assert len(set(pilot["members"])) == 6
        assert all(
            r["outcome"] == "accepted_repair"
            for r in rows
            if r["id"] in pilot["members"]
        ), "Pilot member is no longer an accepted repair"
        assert pilot["projected_all_126_usd"] <= CEILING, "Pilot exceeds ceiling"


def main():
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument("--init", action="store_true")
    p.add_argument("--check", action="store_true")
    p.add_argument(
        "--checkpoint",
        action="store_true",
        help="Verify recorded completed work while reporting pending members",
    )
    p.add_argument("--show", action="store_true")
    p.add_argument("--render", action="store_true")
    p.add_argument("--data", type=Path, default=DATA)
    p.add_argument("--member")
    p.add_argument("arguments", nargs=argparse.REMAINDER)
    args = p.parse_args()
    if args.init:
        initialize()
    elif args.check:
        check(args.data, checkpoint=args.checkpoint)
    elif args.show:
        print(json.dumps(load(args.data), indent=2, ensure_ascii=False))
    elif args.render:
        render(load(args.data))
    elif args.member:
        arguments = (
            args.arguments[1:] if args.arguments[:1] == ["--"] else args.arguments
        )
        return command(args.member, arguments)
    else:
        p.error("Choose --init, --check, --show, --render or --member")
    return 0


if __name__ == "__main__":
    sys.exit(main())
