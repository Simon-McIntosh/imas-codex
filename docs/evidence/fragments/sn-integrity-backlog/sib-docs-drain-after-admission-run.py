"""Read a flush drain's live state and exact-run billing without graph writes."""

from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
from collections import Counter
from datetime import UTC, datetime
from decimal import Decimal
from html.parser import HTMLParser
from pathlib import Path


class EvidenceHTML(HTMLParser):
    """Read visible table cells and explicitly bound narrative values."""

    def __init__(self):
        super().__init__()
        self.tables = []
        self.metrics = {}
        self.table = self.row = self.cell = self.metric = None

    def handle_starttag(self, tag, attrs):
        if tag == "table":
            assert self.table is None, "nested evidence table"
            self.table = []
        elif tag == "tr" and self.table is not None:
            self.row = []
        elif tag in {"td", "th"} and self.row is not None:
            self.cell = []
        elif tag == "span" and "data-check" in dict(attrs):
            key = dict(attrs)["data-check"]
            assert key not in self.metrics and self.metric is None, "duplicate metric"
            self.metric = (key, [])

    def handle_data(self, data):
        if self.cell is not None:
            self.cell.append(data)
        if self.metric is not None:
            self.metric[1].append(data)

    def handle_endtag(self, tag):
        if tag in {"td", "th"} and self.cell is not None:
            self.row.append(" ".join("".join(self.cell).split()))
            self.cell = None
        elif tag == "tr" and self.row is not None:
            self.table.append(self.row)
            self.row = None
        elif tag == "table" and self.table is not None:
            self.tables.append(self.table)
            self.table = None
        elif tag == "span" and self.metric is not None:
            key, parts = self.metric
            self.metrics[key] = "".join(parts).strip()
            self.metric = None


def money(value):
    return Decimal(str(value)).quantize(Decimal("0.000001"))


def validate_receipt(state):
    """Reconcile phase-ledger observations against the independent run mirrors."""
    run, costs = state["run"], state["costs"]
    assert costs and all(row["calls"] == row["priced"] for row in costs)
    assert len({row["phase"] for row in costs}) == len(costs), "duplicate phase"
    billed = money(sum(Decimal(str(row["billed_usd"])) for row in costs))
    assert billed == money(state["billed_usd"]), "snapshot billing disagrees"
    assert billed == money(run["cost_spent"]), "cost_spent disagrees with ledger"
    assert billed == money(run["cost_total"]), "cost_total disagrees with ledger"
    assert sum(row["calls"] for row in costs) == run["events_total"], (
        "event count disagrees"
    )
    assert run["cost_is_exact"] is True, "billing is not exact"
    assert money(run["cost_limit"]) == 20, "unexpected authorization"
    assert 0 <= billed <= 20, "billed spend exceeded authorization"


def verify_report(before, after, fragment):
    """Check saved receipts and rendered report values without a live connection."""
    validate_receipt(after)
    counts = []
    for state in (before, after):
        census, coverage = state["census"], state["coverage"]
        assert (
            census["population_count"] == coverage["candidates"] == coverage["with_id"]
        )
        assert (
            sum(
                census[key]
                for key in (
                    "terminal_count",
                    "terminal_for_run_count",
                    "claimable_count",
                    "stranded_count",
                )
            )
            == census["population_count"]
        ), "census does not reconcile"
        assert state["instrument_control"]["unknown_stage_stranded_count"] == 1
        names = state["accepted_names"]
        assert len({row["id"] for row in names}) == len(names), (
            "duplicate accepted name"
        )
        count = Counter(row["docs_stage"] for row in names)
        assert count == {
            row["docs_stage"]: row["count"]
            for row in state["stages"]
            if row["name_stage"] == "accepted"
        }, "accepted-name enumeration disagrees with stage aggregation"
        counts.append(count)
    run = after["run"]
    started = datetime.fromisoformat(run["started_at"])
    stopped = datetime.fromisoformat(run["stopped_at"])
    accepted = {
        row["id"]
        for row in after["docs_reviewed_under_run_id"]
        if row["docs_stage"] == "accepted"
        and row["reviewed_at"] is not None
        and started <= datetime.fromisoformat(row["reviewed_at"]) <= stopped
    }
    prior = {row["id"]: row["docs_stage"] for row in before["accepted_names"]}
    newly = {name for name in accepted if prior.get(name) != "accepted"}
    for key, expected in (
        ("docs_accepted_under_run_id", accepted),
        ("docs_newly_accepted_under_run_id", newly),
    ):
        ids = [row["id"] for row in after[key]]
        assert len(ids) == len(set(ids)) and set(ids) == expected, key + " disagrees"
    parsed = EvidenceHTML()
    parsed.feed(fragment)
    stage_rows = [["Documentation stage", "Before", "After", "Change"]]
    for stage in ("accepted", "reviewed", "drafted", "pending", "exhausted"):
        left, right = (count[stage] for count in counts)
        delta = right - left
        change = f"+{delta:,}" if delta > 0 else f"−{-delta:,}" if delta < 0 else "0"
        stage_rows.append([stage, f"{left:,}", f"{right:,}", change])
    totals = [sum(count.values()) for count in counts]
    stage_rows.append(
        [
            "Total accepted names",
            *(f"{n:,}" for n in totals),
            f"{totals[1] - totals[0]:+,}",
        ]
    )
    census_rows = [["Classification", "Before", "After"]]
    for label, key in (
        ("Population", "population_count"),
        ("Stranded", "stranded_count"),
        ("Held for run", "terminal_for_run_count"),
        ("Claimable", "claimable_count"),
        ("Durably terminal", "terminal_count"),
        ("Overlapping pool ownership", "overlap_count"),
    ):
        census_rows.append([label, *(f"{s['census'][key]:,}" for s in (before, after))])
    costs = {row["phase"]: row for row in after["costs"]}
    phase_order = (
        "generate_docs",
        "review_docs",
        "refine_docs",
        "review_name",
        "refine_name",
        "prose_adjudication",
        "enrich_parents",
    )
    assert set(costs) == set(phase_order), "unreported ledger phase"
    spend_rows = [["Phase", "Ledger events", "USD"]] + [
        [
            phase,
            f"{costs[phase]['calls']:,}",
            f"{money(costs[phase]['billed_usd']):.6f}",
        ]
        for phase in phase_order
    ]
    spend_rows.append(
        ["Total", f"{run['events_total']:,}", f"{money(after['billed_usd']):.6f}"]
    )
    assert parsed.tables == [stage_rows, census_rows, spend_rows], (
        "visible report table drift"
    )
    expected_metrics = {
        "terminal_verdict": "met"
        if run["stop_reason"] in {"no_eligible_work", "budget_exhausted"}
        else "failed",
        "stop_reason": run["stop_reason"],
        "run_id": run["id"],
        "billed_usd": f"{money(after['billed_usd']):.6f}",
        "cost_limit": "20",
        "unused_usd": f"{money(20 - money(after['billed_usd'])):.6f}",
        "stranded_after": str(after["census"]["stranded_count"]),
        "accepted_under_run": str(len(accepted)),
        "newly_accepted": str(len(newly)),
        "previously_accepted": str(len(accepted - newly)),
        "accepted_docs_decline": str(counts[0]["accepted"] - counts[1]["accepted"]),
        "cost_spent": f"{money(run['cost_spent']):.6f}",
        "cost_total": f"{money(run['cost_total']):.6f}",
    }
    assert parsed.metrics == expected_metrics, "narrative report metric drift"
    return {
        "evidence_verified": True,
        "stop_reason": run["stop_reason"],
        "terminal_measure_met": run["stop_reason"]
        in {"no_eligible_work", "budget_exhausted"},
        "accepted_under_run": len(accepted),
        "newly_accepted": len(newly),
    }


def check_report_guards(before, after, fragment):
    """Refuse independent corruptions of visible text and saved billing mirrors."""
    verify_report(before, after, fragment)
    cases = {}
    for field in ("cost_spent", "cost_total", "events_total"):
        changed = copy.deepcopy(after)
        changed["run"][field] += 1
        cases[field] = (changed, fragment)
    changed = copy.deepcopy(after)
    text = fragment
    for field in ("cost_spent", "cost_total"):
        original = money(after["run"][field])
        changed["run"][field] += 1
        target = f'<span data-check="{field}">{original:.6f}</span>'
        replacement = f'<span data-check="{field}">{original + 1:.6f}</span>'
        assert text.count(target) == 1, "billing mirror mutation has no unique target"
        text = text.replace(target, replacement)
    cases["billing_mirrors_and_report"] = (changed, text)
    changed = copy.deepcopy(after)
    changed["run"]["events_total"] += 1
    target = f"<th>Total</th><td>{after['run']['events_total']:,}</td>"
    assert fragment.count(target) == 1, "event total mutation has no unique target"
    cases["event_total_and_report"] = (
        changed,
        fragment.replace(
            target, f"<th>Total</th><td>{changed['run']['events_total']:,}</td>"
        ),
    )
    visible = f"<td>{Counter(row['docs_stage'] for row in after['accepted_names'])['accepted']:,}</td>"
    assert fragment.count(visible) == 1, "visible table mutation has no unique target"
    cases["visible_stage_count"] = (after, fragment.replace(visible, "<td>0</td>"))
    marker = '<span data-check="accepted_under_run">'
    assert fragment.count(marker) == 1, "narrative mutation has no unique target"
    cases["visible_run_count"] = (after, fragment.replace(marker, marker + "0"))
    marker = '<span data-check="terminal_verdict">failed</span>'
    assert fragment.count(marker) == 1, "completion mutation has no unique target"
    cases["false_completion_claim"] = (
        after,
        fragment.replace(marker, marker.replace("failed", "completed")),
    )
    for name, (state, text) in cases.items():
        try:
            verify_report(before, state, text)
        except AssertionError as exc:
            print(f"REFUSED {name}: {exc}")
        else:
            raise AssertionError(f"guard accepted corruption: {name}")
    print(f"report_guard_checks={len(cases)} all_refused=true")


def read_state(gc):
    from imas_codex.standard_names import pools

    groups = pools.identity_state_groups(gc=gc)
    census = pools.classify_identity_state_groups(
        groups, pools.standard_name_pool_predicates()
    )
    coverage = gc.query(
        "MATCH (n:StandardName) RETURN count(n) AS candidates, "
        "count(n.id) AS with_id, count(n.name_stage) AS with_name_stage, "
        "count(n.docs_stage) AS with_docs_stage"
    )[0]
    assert coverage["candidates"] > 0
    assert coverage["candidates"] == coverage["with_id"]
    assert coverage["candidates"] == census["population_count"]
    docs = gc.query(
        "MATCH (n:StandardName) RETURN n.name_stage AS name_stage, "
        "n.docs_stage AS docs_stage, count(*) AS count "
        "ORDER BY name_stage, docs_stage"
    )
    accepted = gc.query(
        "MATCH (n:StandardName {name_stage: 'accepted'}) "
        "RETURN n.id AS id, n.docs_stage AS docs_stage ORDER BY id"
    )
    controls = pools.classify_identity_state_groups(
        [{"name_stage": "unrecognized", "identities": ["control"], "count": 1}],
        pools.standard_name_pool_predicates(),
    )
    assert controls["stranded_count"] == 1
    return {
        "observed_at": datetime.now(UTC).isoformat(),
        "coverage": dict(coverage),
        "stages": [dict(row) for row in docs],
        "accepted_names": [dict(row) for row in accepted],
        "census": census,
        "instrument_control": {"unknown_stage_stranded_count": 1},
    }


def read_receipt(gc, before, directory):
    rows = gc.query(
        "MATCH (r:SNRun) WHERE r.cost_limit = 20 "
        "RETURN properties(r) AS receipt ORDER BY r.created_at DESC LIMIT 100",
    )
    (directory / "receipt-candidates.json").write_text(
        json.dumps([dict(row["receipt"]) for row in rows], indent=2, default=str) + "\n"
    )
    candidates = []
    for row in rows:
        receipt = dict(row["receipt"])
        flags = json.loads(receipt.get("invocation_flags") or "{}")
        started = datetime.fromisoformat(str(receipt["started_at"]))
        if (
            started >= datetime.fromisoformat(before["observed_at"])
            and flags.get("flush")
            and flags.get("cost_limit") == 20
        ):
            candidates.append(receipt)
    assert len(candidates) == 1, f"ambiguous drain receipt: {candidates}"
    receipt = candidates[0]
    run_id = receipt["id"]
    costs = gc.query(
        "MATCH (c:LLMCost {run_id: $run_id}) "
        "RETURN c.phase AS phase, count(*) AS calls, "
        "count(c.llm_cost) AS priced, sum(c.llm_cost) AS billed_usd, "
        "count(c.standard_name_ids) AS named_calls ORDER BY phase",
        run_id=run_id,
    )
    # Review costs bind existing names to this invocation without changing origin.
    touched = gc.query(
        "MATCH (c:LLMCost {run_id: $run_id}) "
        "WHERE c.phase = 'review_docs' "
        "UNWIND c.standard_name_ids AS id WITH DISTINCT id "
        "MATCH (n:StandardName {id: id}) "
        "RETURN n.id AS id, n.docs_stage AS docs_stage, "
        "n.reviewed_docs_at AS reviewed_at, "
        "n.reviewer_score_docs AS docs_score ORDER BY id",
        run_id=run_id,
    )
    before_stages = {row["id"]: row["docs_stage"] for row in before["accepted_names"]}
    started = datetime.fromisoformat(str(receipt["started_at"]))
    stopped = (
        datetime.fromisoformat(str(receipt["stopped_at"]))
        if receipt.get("stopped_at")
        else datetime.now(UTC)
    )
    accepted_under_run = [
        dict(row)
        for row in touched
        if row["docs_stage"] == "accepted"
        and row["reviewed_at"] is not None
        and started <= datetime.fromisoformat(str(row["reviewed_at"])) <= stopped
    ]
    newly_accepted = [
        row for row in accepted_under_run if before_stages.get(row["id"]) != "accepted"
    ]
    samples = gc.query(
        "MATCH (n:StandardName) WHERE n.id IN $ids "
        "OPTIONAL MATCH (s:StandardNameSource)-[:PRODUCED_NAME]->(n) "
        "RETURN n.id AS id, n.description AS description, n.documentation AS documentation, "
        "n.reviewer_score_docs AS docs_score, n.reviewer_score_name AS name_score, "
        "collect(s.source_id) AS source_paths ORDER BY id",
        ids=[row["id"] for row in newly_accepted[:5]],
    )
    return {
        "run": receipt,
        "costs": [dict(row) for row in costs],
        "billed_usd": sum(float(row["billed_usd"] or 0) for row in costs),
        "docs_reviewed_under_run_id": [dict(row) for row in touched],
        "docs_accepted_under_run_id": accepted_under_run,
        "docs_newly_accepted_under_run_id": newly_accepted,
        "representatives": [dict(row) for row in samples],
    }


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument(
        "phase", choices=["before", "after", "progress", "verify", "self-check"]
    )
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument(
        "--fragment",
        type=Path,
        default=Path(__file__).with_name(
            Path(__file__).name.replace("-run.py", ".html")
        ),
    )
    args = parser.parse_args()
    directory = args.directory
    if args.phase in {"verify", "self-check"}:
        before = json.loads((directory / "before.json").read_text())
        after = json.loads((directory / "after.json").read_text())
        fragment = args.fragment.read_text()
        result = verify_report(before, after, fragment)
        if args.phase == "self-check":
            check_report_guards(before, after, fragment)
        print(json.dumps(result, sort_keys=True))
        return

    from imas_codex.graph.client import GraphClient
    from imas_codex.standard_names import pools

    with GraphClient() as gc:
        state = read_state(gc)
        if args.phase != "before":
            before = json.loads((directory / "before.json").read_text())
            state.update(read_receipt(gc, before, directory))
    state["revision"] = (
        subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        if (Path.cwd() / ".git").exists()
        else os.environ["DRAIN_REVISION"]
    )
    state["module"] = pools.__file__
    state["cwd"] = str(Path.cwd().resolve())
    (directory / f"{args.phase}.json").write_text(
        json.dumps(state, indent=2, sort_keys=True, default=str) + "\n"
    )
    census = state["census"]
    summary = {
        "phase": args.phase,
        "population": census["population_count"],
        "stranded": census["stranded_count"],
        "held_for_run": census["terminal_for_run_count"],
        "claimable": census["claimable_count"],
        "accepted_names_by_docs_stage": {
            str(row["docs_stage"]): row["count"]
            for row in state["stages"]
            if row["name_stage"] == "accepted"
        },
    }
    if args.phase != "before":
        summary.update(
            run_id=state["run"]["id"],
            stop_reason=state["run"].get("stop_reason"),
            billed_usd=state["billed_usd"],
            reached_docs_accepted=len(state["docs_accepted_under_run_id"]),
            newly_docs_accepted=len(state["docs_newly_accepted_under_run_id"]),
        )
    print(json.dumps(summary, sort_keys=True), flush=True)
    assert census["stranded_count"] == 0, "stranded identities remain"
    assert census["overlap_count"] == 0, "overlapping admission rules"
    if args.phase == "after":
        validate_receipt(state)
        assert state["run"].get("stop_reason") in {
            "no_eligible_work",
            "budget_exhausted",
        }, "drain did not reach an eligible terminal condition"


if __name__ == "__main__":
    main()
