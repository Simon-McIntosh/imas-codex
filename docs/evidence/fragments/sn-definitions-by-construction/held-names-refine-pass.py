"""Plan, run and report the second refine pass over the held names.

The first rewrite slice settled 349 of the 485 names it took; this pass takes
the 136 it left unaccepted (119 reviewed, 13 exhausted, 4 drafted, one
superseded by an earlier slice's rename) and gives each one a defined outcome:
an edit that settles it, a rename with its successor's definition, or a
recorded reason it cannot be settled here.

Stored text changes only through ``imas-codex sn edit``; this script never
writes the graph. It is re-runnable and every figure is read from the graph,
the pass's own progress file, and the per-call cost ledger.

Phases:

``status``  read the progress file and say what has settled and what is left.
``report``  assemble the delivered JSON and HTML: population and hold counts,
            the before/after table with scores, every name still held with its
            reason, the renames with their semantic distinction, the names
            waiting on grammar with the word each needs, the names needing a
            rebind, the live-names-without-a-source figure, and the spend.
"""

from __future__ import annotations

import argparse
import collections
import html
import json
from datetime import UTC, datetime
from pathlib import Path

from imas_codex.graph.client import GraphClient
from imas_codex.standard_names.source_reads import count_live_names_without_source

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
RUNDIR = Path(
    "/home/ITER/mcintos/.config/reckon/crew/runs/"
    "r-20261115T165306964846-sdbc-held-names-refine-pass"
)
PROGRESS = Path(
    "/home/ITER/mcintos/.cache/imas-codex-live-rewrite/held-names-progress.jsonl"
)
OUT = HERE / "held-names-refine-pass.json"
HTML_OUT = HERE / "held-names-refine-pass.html"

CEILING_USD = 20.0
NODE_OPENED = datetime(2026, 11, 15, 16, 53, 6, tzinfo=UTC)
SOURCELESS_GATE = 10
EXCERPT = 320


def load_json(path: Path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text())


def load_progress() -> list[dict]:
    records = []
    if not PROGRESS.exists():
        return records
    for line in PROGRESS.read_text().splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def settled(records: list[dict]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for rec in records:
        if rec.get("event") == "settled" and rec.get("step_id"):
            out[rec["step_id"]] = rec
    return out


def live_rows(gc, names: list[str]) -> dict[str, dict]:
    if not names:
        return {}
    rows = gc.query(
        "MATCH (s:StandardName) WHERE s.id IN $n "
        "RETURN s.id AS id, s.unit AS unit, s.description AS description, "
        "s.documentation AS documentation, s.name_stage AS name_stage, "
        "s.docs_stage AS docs_stage, s.status AS status, "
        "s.edit_status AS edit_status, s.reviewer_score_docs AS score_docs, "
        "s.reviewer_score_name AS score_name, s.superseded_by AS superseded_by",
        n=names,
    )
    return {r["id"]: r for r in rows}


def open_holds(gc, names: list[str]) -> dict[str, list[dict]]:
    if not names:
        return {}
    rows = gc.query(
        "MATCH (h:ReviewHold) WHERE h.status = 'open' AND h.id IN $n "
        "RETURN h.id AS id, h.axis AS axis, h.kind AS kind, h.finding AS finding",
        n=names,
    )
    out: dict[str, list[dict]] = collections.defaultdict(list)
    for r in rows:
        out[r["id"]].append(
            {"axis": r["axis"], "kind": r["kind"], "finding": r["finding"]}
        )
    return out


LINEAGE = """
MATCH path = (s:StandardName)-[:REFINED_FROM*0..6]->(o:StandardName {id:$n})
RETURN s.id AS id, s.name_stage AS ns, s.docs_stage AS ds,
       min(length(path)) AS depth
"""

SUCCESSORS = """
MATCH (t:StandardName)-[:REFINED_FROM]->(p:StandardName) WHERE p.id IN $ids
RETURN p.id AS id, count(t) AS n
"""


def lineage(gc, names: list[str]) -> dict[str, list[dict]]:
    """Every identity a name's refinement reached, with depth and out-degree.

    Lineage branches, and the branch the reviewer abandoned is not the identity
    that carries the name: `beta` produced both `toroidal_plasma_beta`, which
    died undecided, and `normalized_toroidal_beta`, which landed accepted. A
    walk that stops at the first hop follows whichever edge the database returns
    first, so this report gathers the whole descendant set and picks the live
    identity from all of it — every figure stated about a name is read from the
    node that pick returns. Depth is the shortest path from the name, and a
    node's out-degree is counted in its own query because one row per path would
    otherwise multiply it by the number of routes that reach the node.
    """
    out: dict[str, list[dict]] = {name: gc.query(LINEAGE, n=name) for name in names}
    ids = sorted({row["id"] for rows in out.values() for row in rows})
    degrees = {r["id"]: r["n"] for r in gc.query(SUCCESSORS, ids=ids)}
    for rows in out.values():
        for row in rows:
            row["out_deg"] = degrees.get(row["id"], 0)
    return out


def pick_live(candidates: list[dict], root: str) -> dict:
    """Choose the identity that carries the name, by the order the graph settles.

    An identity accepted on both axes is what the catalog publishes; failing
    that, one whose name is accepted is the name a reader will find; failing
    that, the refinement dead-end the pipeline stopped on.
    """
    if not candidates:
        return {"id": root, "ns": None, "ds": None, "depth": 0, "out_deg": 0}

    def deepest(rows: list[dict]) -> dict:
        return max(rows, key=lambda r: (r["depth"], -r["out_deg"]))

    published = [
        c for c in candidates if c["ns"] == "accepted" and c["ds"] == "accepted"
    ]
    if published:
        return deepest(published)
    named = [c for c in candidates if c["ns"] == "accepted"]
    if named:
        return deepest(named)
    stopped = [c for c in candidates if c["out_deg"] == 0 and c["ns"] != "superseded"]
    if stopped:
        return deepest(stopped)
    return next((c for c in candidates if c["id"] == root), candidates[0])


def resolve_live(gc, names: list[str]) -> dict[str, str]:
    """Map each name to the identity that carries it forward."""
    return {
        name: pick_live(rows, name)["id"] for name, rows in lineage(gc, names).items()
    }


def spend_by_name(gc) -> dict[str, dict]:
    rows = gc.query(
        "MATCH (c:LLMCost) WHERE c.llm_at >= $lo "
        "RETURN c.phase AS phase, c.llm_cost AS cost, "
        "c.standard_name_ids AS ids",
        lo=NODE_OPENED,
    )
    out: dict[str, dict] = {}
    for r in rows:
        ids = r["ids"]
        if not isinstance(ids, list):
            try:
                ids = json.loads(ids or "[]")
            except json.JSONDecodeError:
                ids = []
        for name in ids:
            entry = out.setdefault(name, {"spend_usd": 0.0, "calls": 0, "phases": {}})
            entry["spend_usd"] = round(entry["spend_usd"] + (r["cost"] or 0.0), 6)
            entry["calls"] += 1
            phase = r["phase"] or ""
            entry["phases"][phase] = round(
                entry["phases"].get(phase, 0.0) + (r["cost"] or 0.0), 6
            )
    return out


def outcome_of(
    name: str,
    decisions: dict,
    row: dict | None,
    holds: dict,
    settled_step: bool,
) -> str:
    record = decisions.get(name, {})
    mode = record.get("mode")
    if mode == "settled-earlier":
        return "settled by an earlier slice"
    if mode == "hold":
        return "held: grouping parent"
    if mode == "wait-grammar":
        return "waiting on grammar"
    if mode == "rebind":
        return "needs a rebind or fold decision"
    if mode in ("docs", "rename") and not settled_step:
        return "queued (not yet run)"
    if not row:
        return "absent from the graph"
    if row["name_stage"] == "accepted" and row["docs_stage"] == "accepted":
        return "accepted"
    if holds.get(name):
        return "still held"
    return f"unsettled ({row['name_stage']}/{row['docs_stage']})"


def build() -> dict:
    inventory = load_json(RUNDIR / "inventory.json", [])
    decisions = load_json(RUNDIR / "decisions.json", {})
    queue = load_json(RUNDIR / "queue.json", [])
    records = load_progress()
    done = settled(records)

    population = [row["id"] for row in inventory]
    successors = [
        rec["rename_to"] for rec in decisions.values() if rec.get("rename_to")
    ]
    names = sorted(set(population) | set(successors))

    with GraphClient() as gc:
        resolved = resolve_live(gc, names)
        live = live_rows(gc, names)
        live.update(live_rows(gc, [v for v in set(resolved.values()) if v not in live]))
        holds = open_holds(gc, names)
        spend = spend_by_name(gc)
        sourceless = count_live_names_without_source(gc)

    before_by_stage = collections.Counter()
    for row in inventory:
        before_by_stage[f"{row['name_stage']}/{row['docs_stage']}"] += 1

    hold_before = collections.Counter()
    for name in population:
        for kind in {h["kind"] for h in holds.get(name, [])}:
            hold_before[kind] += 1

    def spend_for(name: str, record: dict, live_id: str) -> float:
        """The node's spend on a name: the edit's own target plus any lineage.

        A rename's cost rows carry the successor's id, so the predecessor's row
        is credited with the whole chain's spend rather than reading zero.
        """
        chain = {name, live_id}
        if record.get("rename_to"):
            chain.add(record["rename_to"])
        return round(sum(spend.get(n, {}).get("spend_usd", 0.0) for n in chain), 6)

    name_rows = []
    for name in names:
        record = decisions.get(name, {})
        step = next((q for q in queue if q["step_id"] == f"{name}:docs"), None)
        if step is None:
            step = next((q for q in queue if q["name"] == name), None)
        settled_rec = done.get(step["step_id"]) if step else None
        row = live.get(resolved.get(name, name), {})
        inv = next((r for r in inventory if r["id"] == name), {})
        name_rows.append(
            {
                "name": name,
                "live_id": resolved.get(name, name),
                "successor_of": next(
                    (
                        old
                        for old, rec in decisions.items()
                        if rec.get("rename_to") == name
                    ),
                    None,
                ),
                "hold_category": record.get("category"),
                "live_children": inv.get("live_children"),
                "source_kind": inv.get("source_kind"),
                "remedy": record.get("mode"),
                "outcome": outcome_of(name, decisions, row, holds, bool(settled_rec)),
                "before": {
                    "name_stage": inv.get("name_stage"),
                    "docs_stage": inv.get("docs_stage"),
                    "score_docs": inv.get("score_docs"),
                    "score_name": inv.get("score_name"),
                    "description_excerpt": (inv.get("description") or "")[:EXCERPT],
                },
                "after": {
                    "name_stage": row.get("name_stage"),
                    "docs_stage": row.get("docs_stage"),
                    "score_docs": row.get("score_docs"),
                    "score_name": row.get("score_name"),
                    "description": row.get("description"),
                },
                "open_holds": [
                    {"kind": h["kind"], "finding": h["finding"]}
                    for h in holds.get(name, [])
                ],
                "spend_usd": spend_for(name, record, resolved.get(name, name)),
                "settled_step": bool(settled_rec),
                "reason": record.get("reason"),
                "rename_to": record.get("rename_to"),
                "distinction": record.get("distinction"),
                "grammar_word": record.get("grammar_word"),
                "grammar_note": record.get("grammar_note"),
                "rebind_note": record.get("rebind_note"),
                "link_pairs": record.get("link_pairs"),
            }
        )

    totals = collections.Counter(r["outcome"] for r in name_rows)
    paid_steps = [q for q in queue if q["mode"] in ("hint-docs", "rename")]
    settled_paid = [q for q in paid_steps if q["step_id"] in done]

    total_spend = round(sum(v["spend_usd"] for v in spend.values()), 6)
    rate = round(total_spend / len(settled_paid), 4) if settled_paid else None

    by_hold_kind: dict[str, dict] = {}
    for row in name_rows:
        key = row["hold_category"] or "unclassified"
        slot = by_hold_kind.setdefault(
            key, {"names": 0, "accepted": 0, "spend_usd": 0.0}
        )
        slot["names"] += 1
        if row["outcome"] == "accepted":
            slot["accepted"] += 1
        slot["spend_usd"] = round(slot["spend_usd"] + row["spend_usd"], 6)
        slot["rate_per_name_usd"] = round(slot["spend_usd"] / slot["names"], 4)

    return {
        "schema": "held-names-refine-pass/1",
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "plan_section": "sn-definitions-by-construction#s7",
        "run": {
            "node": "sdbc-held-names-refine-pass",
            "run_id": "r-20261115T165306964846-sdbc-held-names-refine-pass",
            "opened_at": NODE_OPENED.isoformat(),
            "worktree": str(REPO),
            "base_sha": "c266404dfb0dc4d83bf7a1dfd4df6b0b19868fa6",
        },
        "population": {
            "count": len(population),
            "before_stage_counts": dict(before_by_stage),
            "note": "the 136 names the first rewrite slice left unaccepted, read "
            "from the live graph; succeeded names are listed as successors and "
            "are not part of the population count",
        },
        "holds_before": {
            "names_with_open_holds": sum(1 for n in population if holds.get(n)),
            "open_hold_count": sum(len(holds.get(n, [])) for n in population),
            "by_kind": dict(hold_before),
        },
        "outcomes": {
            "counts": dict(totals),
            "by_hold_kind": by_hold_kind,
        },
        "spend": {
            "total_usd": total_spend,
            "ceiling_usd": CEILING_USD,
            "remaining_usd": round(CEILING_USD - total_spend, 4),
            "paid_steps": len(paid_steps),
            "settled_paid_steps": len(settled_paid),
            "rate_per_settled_step_usd": rate,
        },
        "names": name_rows,
        "renames": [
            {
                "old": r["name"],
                "new": r["rename_to"],
                "live_id": r["live_id"],
                "landed_note": (
                    None
                    if r["live_id"] == r["rename_to"]
                    or (r["after"] or {}).get("name_stage") != "superseded"
                    else "the planned spelling did not survive review; the name "
                    f"reviewer route landed the identity on {r['live_id']}"
                ),
                "distinction": r["distinction"],
                "after": r["after"],
                "spend_usd": r["spend_usd"],
            }
            for r in name_rows
            if r["rename_to"]
        ],
        "still_held": [
            {
                "name": r["name"],
                "hold_category": r["hold_category"],
                "outcome": r["outcome"],
                "live_children": r["live_children"],
                "reason": r["reason"],
                "open_holds": r["open_holds"],
            }
            for r in name_rows
            if r["outcome"] not in ("accepted", "settled by an earlier slice")
        ],
        "grammar_waits": [
            {"name": r["name"], "word": r["grammar_word"], "note": r["grammar_note"]}
            for r in name_rows
            if r["grammar_word"]
        ],
        "rebinds": [
            {"name": r["name"], "note": r["rebind_note"], "reason": r["reason"]}
            for r in name_rows
            if r["rebind_note"]
        ],
        "sourceless": {
            "baseline": SOURCELESS_GATE,
            "now": sourceless["count"],
            "gate": SOURCELESS_GATE,
            "names": sourceless["names"],
        },
        "progress": {
            "records": len(records),
            "settled_steps": len(done),
        },
    }


def render(doc: dict) -> str:
    def esc(value) -> str:
        return html.escape("" if value is None else str(value))

    def score(value) -> str:
        return "—" if value is None else f"{value:.3f}"

    parts = [
        "<!DOCTYPE html>",
        '<html lang="en">',
        "<head>",
        '<meta charset="utf-8">',
        "<title>The held names: second refine pass</title>",
        "</head>",
        "<body>",
        "<article>",
        "<h2>The held names: second refine pass</h2>",
    ]
    spend = doc["spend"]
    outcomes = doc["outcomes"]["counts"]
    parts.append(
        f"<p>The first rewrite slice left <strong>{doc['population']['count']}</strong> "
        "names unaccepted. This pass took each of them to one of three ends: a "
        "definition the seat accepts, a rename to the identity its binding states, "
        "or a recorded reason it cannot be settled in this pass. "
        f"<strong>{outcomes.get('accepted', 0)} accepted</strong>, "
        f"<strong>{sum(v for k, v in outcomes.items() if k.startswith('waiting'))}</strong> "
        "waiting on grammar, "
        f"<strong>{sum(v for k, v in outcomes.items() if k.startswith('held'))}</strong> "
        "held as grouping parents, "
        f"<strong>{sum(v for k, v in outcomes.items() if 'rebind' in k)}</strong> "
        "needing a rebind or fold decision. Recorded spend "
        f"<strong>USD {spend['total_usd']:.4f}</strong> against the ceiling of "
        f"USD {spend['ceiling_usd']:.0f}, at USD "
        f"{spend['rate_per_settled_step_usd'] or 0:.4f} per settled step.</p>"
    )

    parts.append("<h3>Where the population started</h3>")
    parts.append("<table><caption>Names by stage as the pass opened</caption>")
    parts.append("<tr><th>name axis / docs axis</th><th>Names</th></tr>")
    for stage, count in sorted(doc["population"]["before_stage_counts"].items()):
        parts.append(f"<tr><td>{esc(stage)}</td><td>{count}</td></tr>")
    parts.append("</table>")

    parts.append("<h3>Open holds as the pass opened</h3>")
    parts.append("<table><caption>Open hold nodes over the population</caption>")
    parts.append("<tr><th>Hold kind</th><th>Names</th></tr>")
    for kind, count in sorted(doc["holds_before"]["by_kind"].items()):
        parts.append(f"<tr><td><code>{esc(kind)}</code></td><td>{count}</td></tr>")
    parts.append(
        f"<tr><td><strong>all</strong></td>"
        f"<td><strong>{doc['holds_before']['open_hold_count']}</strong></td></tr>"
    )
    parts.append("</p>")

    parts.append("<h3>Outcome by hold kind</h3>")
    parts.append(
        "<table><caption>Names, acceptances and spend per hold kind "
        "(hold kind as classified from the open hold nodes)</caption>"
    )
    parts.append(
        "<tr><th>Hold kind</th><th>Names</th><th>Accepted</th><th>Spend (USD)</th></tr>"
    )
    for kind, slot in sorted(
        doc["outcomes"]["by_hold_kind"].items(), key=lambda kv: -kv[1]["names"]
    ):
        parts.append(
            f"<tr><td>{esc(kind)}</td><td>{slot['names']}</td>"
            f"<td>{slot['accepted']}</td><td>{slot['spend_usd']:.4f}</td>"
            f"<td>{slot['rate_per_name_usd']:.4f}</td></tr>"
        )
    parts.append("</table>")

    parts.append("<h3>Renames: old identity, new identity, distinction</h3>")
    parts.append(
        "<table><caption>Every rename this pass made, with the semantic "
        "distinction the new name carries</caption>"
    )
    parts.append(
        "<tr><th>Old identity</th><th>New identity</th><th>Live identity</th>"
        "<th>Distinction</th><th>After</th></tr>"
    )
    for row in doc["renames"]:
        after = row["after"] or {}
        parts.append(
            f"<tr><td><code>{esc(row['old'])}</code></td>"
            f"<td><code>{esc(row['new'])}</code></td>"
            f"<td><code>{esc(row['live_id'])}</code></td>"
            f"<td>{esc(row['landed_note'] or row['distinction'])}</td>"
            f"<td>{esc(after.get('name_stage'))} / {esc(after.get('docs_stage'))}</td></tr>"
        )
    parts.append("</table>")

    parts.append("<h3>Before and after, with scores</h3>")
    parts.append(
        "<table><caption>Every name, its stage and documentation score before "
        "and after this pass</caption>"
    )
    parts.append(
        "<tr><th>Name</th><th>Before</th><th>Score</th><th>After</th>"
        "<th>Score</th><th>Outcome</th></tr>"
    )
    for row in sorted(doc["names"], key=lambda r: r["name"]):
        before = row["before"]
        after = row["after"]
        parts.append(
            f"<tr><td><code>{esc(row['name'])}</code></td>"
            f"<td>{esc(before['name_stage'])} / {esc(before['docs_stage'])}</td>"
            f"<td>{score(before['score_docs'])}</td>"
            f"<td>{esc(after.get('name_stage'))} / {esc(after.get('docs_stage'))}</td>"
            f"<td>{score(after.get('score_docs'))}</td>"
            f"<td>{esc(row['outcome'])}</td></tr>"
        )
    parts.append("</table>")

    parts.append("<h3>Still held, with the reason</h3>")
    parts.append(
        "<table><caption>Every name this pass leaves unsettled and why</caption>"
    )
    parts.append(
        "<tr><th>Name</th><th>Hold kind</th><th>Outcome</th>"
        "<th>Reason</th><th>Open hold at close</th></tr>"
    )
    for row in sorted(doc["still_held"], key=lambda r: r["name"]):
        open_text = " ".join(
            f"[{h['kind']}] {h['finding']}" for h in row["open_holds"][:2]
        )
        parts.append(
            f"<tr><td><code>{esc(row['name'])}</code></td>"
            f"<td>{esc(row['hold_category'])}</td>"
            f"<td>{esc(row['live_children'] if row['live_children'] is not None else '—')}</td>"
            f"<td>{esc(row['outcome'])}</td>"
            f"<td>{esc(row['reason'])}</td><td>{esc(open_text)}</td></tr>"
        )
    parts.append("</table>")

    parts.append("<h3>Waiting on grammar</h3>")
    parts.append(
        "<table><caption>Names whose definition cannot agree with the name "
        "until the grammar can say the qualifier the binding stores</caption>"
    )
    parts.append(
        "<tr><th>Name</th><th>Word it needs</th><th>Why no spelling "
        "parses today</th></tr>"
    )
    for row in sorted(doc["grammar_waits"], key=lambda r: r["name"]):
        parts.append(
            f"<tr><td><code>{esc(row['name'])}</code></td>"
            f"<td><code>{esc(row['word'])}</code></td>"
            f"<td>{esc(row['note'])}</td></tr>"
        )
    parts.append("</table>")

    parts.append("<h3>Needing a rebind or a fold decision</h3>")
    parts.append(
        "<table><caption>Names whose binding contradicts the name, or whose "
        "identity duplicates an accepted one</caption>"
    )
    parts.append("<tr><th>Name</th><th>Decision needed</th></tr>")
    for row in sorted(doc["rebinds"], key=lambda r: r["name"]):
        parts.append(
            f"<tr><td><code>{esc(row['name'])}</code></td>"
            f"<td>{esc(row['note'])}</td></tr>"
        )
    parts.append("</table>")

    sourceless = doc["sourceless"]
    parts.append("<h3>Live names without a source</h3>")
    parts.append(
        f"<p>The repository's own instrument (<code>sn sourceless</code>, "
        f"<code>count_live_names_without_source</code>) reports "
        f"<strong>{sourceless['now']}</strong> live names with no live producer, "
        f"against a gate of {sourceless['gate']}.</p>"
    )
    parts.append("<ul>")
    for name in sourceless["names"]:
        parts.append(f"<li><code>{esc(name)}</code></li>")
    parts.append("</ul>")

    parts.append("</article>")
    parts.append("</body>")
    parts.append("</html>")
    return "\n".join(parts) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("status", "report"))
    args = parser.parse_args()

    if args.phase == "status":
        records = load_progress()
        done = settled(records)
        queue = load_json(RUNDIR / "queue.json", [])
        paid = [q for q in queue if q["mode"] in ("hint-docs", "rename")]
        print(f"queue steps {len(queue)}; paid {len(paid)}")
        print(f"settled steps {len(done)}")
        outstanding = [q["step_id"] for q in paid if q["step_id"] not in done]
        print(f"paid steps left {len(outstanding)}")
        for step_id in outstanding[:40]:
            print("  ", step_id)
        return 0

    doc = build()
    OUT.write_text(json.dumps(doc, indent=1) + "\n")
    HTML_OUT.write_text(render(doc))
    print(f"wrote {OUT.relative_to(REPO)} and {HTML_OUT.relative_to(REPO)}")
    print(
        f"spend USD {doc['spend']['total_usd']:.4f} of {CEILING_USD:.0f}; "
        f"outcomes {doc['outcomes']['counts']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
