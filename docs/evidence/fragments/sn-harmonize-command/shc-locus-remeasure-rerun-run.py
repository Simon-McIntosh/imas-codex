"""Re-measure one non-WEST locus family through ``sn harmonize``.

The read-only subcommands snapshot a family's plan, report and status.  The
``record-apply`` subcommand ingests a captured apply log.  ``record-cost``
reads the apply's LLM cost and call count from the graph-ops cost owner.
``render`` assembles the HTML fragment from the JSON record.  The ``fixture``
subcommand exercises the score-regression comparison on fixed scores, so an
unchanged column passes only because nothing fell.

Where every cost figure comes from: the runs the apply's own summary names are
selected by their exact run id, never by a prefix, and their spend and call
counts are read through the graph-ops module that owns the ``LLMCost``
aggregation, rather than through Cypher written here — so the record and that
module cannot drift apart.  ``aggregate_spend_per_phase`` gives the per-phase
dollars, ``aggregate_calls_per_phase`` the per-phase call count behind them,
and ``aggregate_spend_per_name`` the apportionment that says which names paid.
Nothing is taken from the campaign summary's own ``total_cost``.

The record answers section 4's questions for the locus kind: calls, spend and
wall time; lint findings by kind before and after; every member's
documentation score before and after; the stamped status; the second dry run's
planned calls; and every attachment reconciliation held.  Every member below
the acceptance threshold, or whose score fell, is named with both scores.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from imas_codex.standard_names.defaults import DEFAULT_MIN_SCORE
from imas_codex.standard_names.harmonize import docs_score_regressions

RESULTS = Path(__file__).with_name("shc-locus-remeasure-rerun.json")
HTML = Path(__file__).with_name("shc-locus-remeasure-rerun.html")
REPO_ROOT = Path(__file__).resolve().parents[4]
REVIEWS_DIR = REPO_ROOT / "imas_codex/standard_names/manifests/reviews"

SEED = "electrostatic_potential"
# A name the roster search is known to find, so a zero member count is shown
# to be a real absence rather than a search that reads nothing.
POSITIVE_CONTROL = "flux_surface_averaged_electron_density_at_plasma_boundary"
POSITIVE_CONTROL_EXPECTED = 10
COST_CAP_USD = 3.59
# One held attachment as the reconciliation pass reports it: the DD path, the
# name it would have stranded, and the pass's reason.
HELD_ATTACHMENT = re.compile(r"(\S+) → (\S+) \(([^)]*)\)")
# The human apply summary the CLI prints without ``--json``: one status line
# per family, then the scope totals.  The CLI wraps the family line across
# whichever lines fit the terminal, so both are matched against the log's
# lines joined into one whitespace-normalised string.
FAMILY_STATUS = re.compile(
    r"family (?P<base>\S+)\s+anchor=(?P<anchor>\S+)\s+batches=(?P<batches>\d+)\s+"
    r"accepted=(?P<accepted>\d+)/(?P<touched>\d+)\s+"
    r"(?P<status>converged|halted)(?:: (?P<reasons>.*?))?(?=\s+apply:|$)"
)
SCOPE_TOTALS = re.compile(
    r"apply: (?P<applied>\d+) family\(ies\) applied, (?P<stamped>\d+) stamped\."
)


def read_results() -> dict:
    if RESULTS.exists():
        return json.loads(RESULTS.read_text())
    return {"acceptance_threshold": DEFAULT_MIN_SCORE, "seed": SEED}


def write_results(payload: dict) -> None:
    RESULTS.write_text(json.dumps(payload, indent=0, sort_keys=True) + "\n")


def _fmt(value: object, digits: int = 4) -> str:
    if isinstance(value, (int, float)):
        return f"{value:.{digits}f}"
    return "—" if value is None else str(value)


def roster_check(members: list[str]) -> dict:
    """Count each member's appearances across the frozen WEST review rosters."""
    files = sorted(REVIEWS_DIR.glob("*.yaml"))
    if not files:
        raise RuntimeError(f"no review rosters under {REVIEWS_DIR}")
    counts = {}
    for name in members:
        counts[name] = sum(path.read_text().count(name) for path in files)
    control_hits = sum(p.read_text().count(POSITIVE_CONTROL) for p in files)
    return {
        "searched": str(REVIEWS_DIR.relative_to(REPO_ROOT)),
        "roster_files": len(files),
        "member_matches": counts,
        "members_in_west_rosters": sorted(n for n in counts if counts[n]),
        "positive_control": POSITIVE_CONTROL,
        "positive_control_match_count": control_hits,
        "positive_control_expected": POSITIVE_CONTROL_EXPECTED,
        "valid_non_west": not any(counts.values()),
    }


def snapshot(phase: str) -> None:
    from imas_codex.graph.client import GraphClient
    from imas_codex.standard_names.harmonize import (
        measure_doc_call_cost,
        plan_family_applies,
        preview_recommended_refusals,
        report_family_rows,
    )

    with GraphClient() as graph:
        plans = plan_family_applies(graph, seeds=[SEED])
        reports = report_family_rows(graph, seeds=[SEED])
        estimate = measure_doc_call_cost(graph)
        if not plans or not reports:
            raise RuntimeError("the named family is absent from the graph")
        plan = plans[0]
        report = reports[0]
        refusals = preview_recommended_refusals(graph, [plan])[0]
        states = list(
            graph.query(
                """
                MATCH (member:StandardName)
                WHERE member.id IN $ids
                RETURN member.id AS id,
                       member.name_stage AS name_stage,
                       member.docs_stage AS docs_stage,
                       member.validation_status AS validation_status
                ORDER BY id
                """,
                ids=plan["members"],
            )
            or []
        )
        if {row["id"] for row in states} != set(plan["members"]):
            raise RuntimeError("the status read missed a family member")
        accepted = sum(
            row.get("docs_stage") == "accepted"
            and row.get("validation_status") != "quarantined"
            for row in states
        )
        data = {
            "phase": phase,
            "seed": SEED,
            "physical_base": plan["physical_base"],
            "invariant": plan["invariant"],
            "members": plan["members"],
            "anchor": plan["anchor"],
            "decision": plan["decision"],
            "harmonized": plan["harmonized"],
            "planned_calls": plan["planned_calls"],
            "estimate_usd": (
                round(plan["planned_calls"] * estimate.mean_usd, 4)
                if estimate is not None
                else None
            ),
            "estimate_basis": (
                {
                    "runs": estimate.runs,
                    "calls": estimate.calls,
                    "total_usd": round(estimate.total_usd, 4),
                    "mean_usd_per_call": round(estimate.mean_usd, 4),
                }
                if estimate is not None
                else None
            ),
            "refusals": refusals,
            "lint": report["lint"],
            "lint_total": report["lint_total"],
            "drift": report["drift"],
            "documentation_scores": plan["member_scores"],
            "member_status": {row["id"]: row for row in states},
            "status_counts": {
                "accepted": accepted,
                "other": len(states) - accepted,
            },
        }
    payload = read_results()
    payload.setdefault("phases", {})[phase] = data
    write_results(payload)
    print(json.dumps(data, sort_keys=True))


def fixture() -> None:
    name = "fixture_low_scoring_member"
    before = {name: 0.95, "fixture_stable_member": 0.9}
    after = {
        name: round(DEFAULT_MIN_SCORE - 0.05, 2),
        "fixture_stable_member": 0.9,
    }
    result = docs_score_regressions(before, after)
    named = [item["id"] for item in result["below_threshold"]]
    if named != [name]:
        raise AssertionError(f"below-threshold member not named: {named}")
    payload = read_results()
    payload["fixture"] = {
        "before": before,
        "after": after,
        "result": result,
        "threshold": DEFAULT_MIN_SCORE,
        "named": named,
    }
    write_results(payload)
    print(json.dumps(payload["fixture"], sort_keys=True))


def held_attachments(lines: list[str]) -> list[dict]:
    """Every attachment reconciliation held, by name and DD path."""
    held: list[dict] = []
    for line in lines:
        if "would leave these live names with no producing source" not in line:
            continue
        tail = line.split("retire the name: ", 1)[-1]
        for match in HELD_ATTACHMENT.finditer(tail):
            held.append(
                {
                    "dd_path": match.group(1),
                    "sn_id": match.group(2),
                    "reason": match.group(3),
                }
            )
    return held


def record_apply(
    log_path: str,
    wall_seconds: float,
    cost_limit: float,
    dry_run_log: str | None = None,
    run_ids: list[str] | None = None,
) -> None:
    """Ingest one apply's captured log into the record.

    Two summary shapes are read.  With ``--json`` the log's last JSON array
    line carries per-family decision, anchor, applied and stamped flags, the
    campaign summary and the documentation-score regressions, and its
    ``result.run_ids`` are the exact ids ``record-cost`` selects the run's
    LLMCost rows on.  Without it the human summary is parsed instead -- the
    per-family status line and the ``apply: N family(ies) applied, M
    stamped.`` totals -- and the run id is supplied by the caller, because
    that summary prints none.  No dollar figure is parsed from the log either
    way; spend comes from the graph-ops cost owners in ``record-cost``.
    """
    log = Path(log_path)
    lines = log.read_text().splitlines()
    summary = None
    for line in reversed(lines):
        stripped = line.strip()
        if stripped.startswith("["):
            try:
                summary = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            break
    exit_line = next(
        (ln for ln in reversed(lines) if ln.startswith("EXIT=")), "EXIT=<absent>"
    )
    error_line = next(
        (
            ln
            for ln in reversed(lines)
            if "Error:" in ln
            or "Exception:" in ln
            or "Source:" in ln
            or "Source :" in ln
        ),
        None,
    )
    entry = summary[0] if summary else {}
    joined = " ".join(ln.strip() for ln in lines)
    family = next((m.groupdict() for m in map(FAMILY_STATUS.search, [joined]) if m), {})
    totals = next((m.groupdict() for m in map(SCOPE_TOTALS.search, [joined]) if m), {})
    chosen_ids = [rid for rid in (run_ids or []) if rid]
    if summary is not None:
        chosen_ids = (entry.get("result") or {}).get("run_ids") or chosen_ids
    payload = read_results()
    payload.setdefault("applies", []).append(
        {
            "log": str(log),
            "dry_run_log": dry_run_log,
            "wall_seconds": wall_seconds,
            "cost_limit_usd": cost_limit,
            "exit_line": exit_line,
            "produced_summary": summary is not None,
            "error_line": error_line,
            "physical_base": entry.get("physical_base") or family.get("base"),
            "anchor": entry.get("anchor") or family.get("anchor"),
            "decision": entry.get("decision"),
            "applied": entry.get("applied"),
            "stamped": entry.get("stamped")
            if summary is not None
            else (totals.get("stamped") == "1" if totals else None),
            "batches_run": int(family["batches"]) if family.get("batches") else None,
            "halted": family.get("status") == "halted",
            "halt_reasons": family.get("reasons"),
            "members_touched": int(family["touched"])
            if family.get("touched")
            else None,
            "members_accepted": int(family["accepted"])
            if family.get("accepted")
            else None,
            "families_applied": int(totals["applied"])
            if totals.get("applied")
            else None,
            "summary": entry.get("result"),
            "run_ids": chosen_ids,
            "run_ids_source": (
                "the apply's own --json summary"
                if summary is not None
                else "named by the operator from the run's own rows: the exact id, never a prefix"
            ),
            "held_attachments": held_attachments(lines),
            "score_regressions": entry.get("score_regressions"),
        }
    )
    write_results(payload)
    print(json.dumps(payload["applies"][-1], sort_keys=True))


def record_cost() -> None:
    """Record the apply's calls, per-phase spend and per-name cost.

    Every figure comes from the graph-ops cost owner, read against the exact
    run ids the apply's own summary reported.  The LLMCost rows are the
    authority for what a run spent and how many calls it made; the dry run's
    ``planned_calls`` is what it would do and ``estimate_basis`` describes
    prior runs, so neither answers what the apply itself did.
    """
    from imas_codex.standard_names.graph_ops import (
        aggregate_calls_per_phase,
        aggregate_spend_per_name,
        aggregate_spend_per_phase,
    )

    payload = read_results()
    if not payload.get("applies"):
        raise RuntimeError("record the apply before its cost")
    entry = payload["applies"][-1]
    run_ids = [r for r in (entry.get("run_ids") or []) if r]
    if not run_ids:
        raise RuntimeError("the apply summary named no run id to select on")
    phases: dict[str, dict[str, float]] = {}
    per_name: dict[str, float] = {}
    for run_id in run_ids:
        spend = aggregate_spend_per_phase(run_id)
        calls = aggregate_calls_per_phase(run_id)
        if set(spend) != set(calls):
            raise RuntimeError(f"spend and calls disagree on phases for {run_id}")
        for phase, usd in spend.items():
            slot = phases.setdefault(phase, {"usd": 0.0, "calls": 0})
            slot["usd"] += float(usd)
            slot["calls"] += int(calls[phase])
        for name, value in aggregate_spend_per_name(run_id).items():
            per_name[name] = per_name.get(name, 0.0) + float(value)
    entry["cost"] = {
        "run_ids": run_ids,
        "phases": {
            phase: {"calls": int(row["calls"]), "usd": round(float(row["usd"]), 6)}
            for phase, row in sorted(phases.items())
        },
        "calls_total": sum(int(row["calls"]) for row in phases.values()),
        "spend_usd": round(sum(float(row["usd"]) for row in phases.values()), 6),
        "per_name_usd": {
            name: round(value, 6) for name, value in sorted(per_name.items())
        },
    }
    entry["cost_limit_unspent_usd"] = round(
        float(entry["cost_limit_usd"]) - entry["cost"]["spend_usd"], 4
    )
    write_results(payload)
    print(json.dumps(entry["cost"], sort_keys=True))


def _score_rows(members: list[str], before: dict, after: dict) -> str:
    rows = []
    for member in members:
        rows.append(
            f"<tr><td><code>{member}</code></td>"
            f"<td>{_fmt(before.get(member))}</td><td>{_fmt(after.get(member))}</td></tr>"
        )
    return "".join(rows)


def render() -> None:
    payload = read_results()
    phases = payload.get("phases", {})
    initial = phases["before"]
    after = phases.get("after", {})
    check = payload["roster_check"]
    applies = payload.get("applies", [])
    apply_entry = applies[-1] if applies else {}
    summary = apply_entry.get("summary") or {}
    cost = apply_entry.get("cost") or {}

    members = initial["members"]
    scores_before = initial.get("documentation_scores") or {}
    scores_after = after.get("documentation_scores") or {}

    below = []
    fell = []
    for member in members:
        b = scores_before.get(member)
        a = scores_after.get(member)
        if a is None:
            continue
        if a < DEFAULT_MIN_SCORE:
            below.append((member, b, a))
        elif b is not None and a < b:
            fell.append((member, b, a))

    roster_rows = "".join(
        f"<tr><td><code>{name}</code></td><td>{count}</td></tr>"
        for name, count in sorted(check["member_matches"].items())
    )
    below_rows = "; ".join(
        f"<code>{m}: {_fmt(b)} &rarr; {_fmt(a)}</code>" for m, b, a in below
    )
    fell_rows = "; ".join(
        f"<code>{m}: {_fmt(b)} &rarr; {_fmt(a)}</code>" for m, b, a in fell
    )
    score_rows = _score_rows(members, scores_before, scores_after)

    # A member the drain reset re-entered the pipeline and is unscored again:
    # its documentation score went from a value to none.
    status_before = initial.get("member_status") or {}
    status_after = after.get("member_status") or {}
    reset_rows = []
    for member in members:
        b = scores_before.get(member)
        a = scores_after.get(member)
        sb = (status_before.get(member) or {}).get("docs_stage")
        sa = (status_after.get(member) or {}).get("docs_stage")
        if b is not None and a is None and sb != sa:
            reset_rows.append((member, b, sb, sa))
    reset_html = "; ".join(
        f"<code>{m}: {_fmt(b)} ({sb} &rarr; {sa})</code>" for m, b, sb, sa in reset_rows
    )

    spend = cost.get("spend_usd")
    calls_total = cost.get("calls_total")
    touched = summary.get("total_touched")
    stamped = apply_entry.get("stamped")
    # A run was raised only if the log carries an error line; running without
    # --json costs the machine summary but is not a crash.
    crash = bool(apply_entry.get("error_line"))
    verdict = "converged" if stamped else ("raised" if crash else "halted")
    halt_reasons = apply_entry.get("halt_reasons") or "; ".join(
        summary.get("halt_reasons") or []
    )
    halt_reasons = halt_reasons.split(" below threshold", 1)[0].strip()
    halt_clause = f" on {halt_reasons}" if halt_reasons else ""
    lint_before = json.dumps(initial.get("lint") or {}, sort_keys=True)
    lint_after = json.dumps((after or {}).get("lint") or {}, sort_keys=True)

    cost_rows = cost.get("phases") or {}
    calls_by_phase = ", ".join(
        f"{phase} {row['calls']}" for phase, row in sorted(cost_rows.items())
    )
    per_name = cost.get("per_name_usd") or {}
    family = set(members)
    run_ids_html = ", ".join(f"<code>{r}</code>" for r in cost.get("run_ids") or [])
    if per_name:
        per_name_clause = (
            ", apportioned by the run's own LLMCost rows to "
            + ", ".join(
                f"<code>{name}</code> (USD {_fmt(value)})"
                for name, value in sorted(per_name.items())
            )
            + (
                ", so the family itself paid nothing"
                if not (set(per_name) & family)
                else ""
            )
        )
    else:
        per_name_clause = ""

    held = apply_entry.get("held_attachments") or []
    held_rows = "".join(
        f"<tr><td><code>{h['sn_id']}</code></td>"
        f"<td><code>{h['dd_path']}</code></td><td>{h['reason']}</td></tr>"
        for h in held
    )
    held_names = ", ".join(
        f"<code>{h['sn_id']}</code> (path <code>{h['dd_path']}</code>)" for h in held
    )

    lines = ["<section>"]
    lines.append(
        '  <h2 id="shc-locus-remeasure-rerun">Locus family '
        + SEED
        + ": re-measured after the stranding hold</h2>"
    )
    lines.append('  <p id="shc-locus-remeasure-rerun-summary">')
    lines.append(
        f"    The non-WEST locus family <code>{SEED}</code> ran under "
        f"<code>sn harmonize --family {SEED} --apply --cost-limit "
        f"{float(apply_entry.get('cost_limit_usd') or 0):.2f}</code>. It "
        f"<strong>{verdict}</strong>{halt_clause}. Spend "
        f"<strong>USD {_fmt(spend)}</strong> over "
        f"<strong>{calls_total if calls_total is not None else '—'} LLMCost call(s)</strong>"
        + (f" ({calls_by_phase})" if calls_by_phase else "")
        + (f" over {_fmt(touched, 0)} touched name(s)" if touched is not None else "")
        + f"; wall {_fmt(apply_entry.get('wall_seconds'), 0)}s. Lint findings "
        f"{lint_before} before, {lint_after} after. {len(below)} member(s) are below "
        f"the {DEFAULT_MIN_SCORE} threshold after the apply and {len(fell)} fell. "
        f"Attachment reconciliation held {len(held)} stranding detach(es)."
    )
    if crash:
        lines.append(
            "    The apply raised <code>"
            + str(apply_entry.get("error_line"))
            + "</code>, so the family is unstamped."
        )
    elif not apply_entry.get("produced_summary", True):
        lines.append(
            "    The apply was run without <code>--json</code>, so no machine "
            "summary line exists to read a result object from: the family's "
            "status and every cost figure here come from the apply's own "
            "human summary and from the graph-ops cost owners instead."
        )
    lines.append("  </p>")
    lines.append(
        '  <h3 id="shc-locus-remeasure-rerun-choice">Family choice and roster check</h3>'
    )
    lines.append('  <p id="shc-locus-remeasure-rerun-choice-body">')
    lines.append(
        f"    <code>{SEED}</code> is a locus family not yet stamped harmonized whose "
        f"members appear in no frozen WEST review roster. The search reads "
        f"{check['roster_files']} files under <code>{check['searched']}</code>. "
        f"Positive control <code>{check['positive_control']}</code> matched "
        f"{check['positive_control_match_count']} times, so a zero member count is a "
        "real absence. Members: "
        + ", ".join("<code>" + m + "</code>" for m in members)
        + "."
    )
    lines.append("  </p>")
    lines.append("  <table>")
    lines.append("    <thead><tr><th>member</th><th>roster matches</th></tr></thead>")
    lines.append("    <tbody>" + roster_rows + "</tbody>")
    lines.append("  </table>")
    lines.append(
        '  <h3 id="shc-locus-remeasure-rerun-dry-run">Dry run and planned calls</h3>'
    )
    lines.append('  <p id="shc-locus-remeasure-rerun-dry-run-body">')
    lines.append(
        f"    The dry run planned {initial['planned_calls']} call(s), estimated USD "
        f"{_fmt(initial.get('estimate_usd'))} from "
        f"{initial.get('estimate_basis', {}).get('runs')} recent run(s). The second "
        f"dry run after the apply planned "
        f"{(after or {}).get('planned_calls', '—')} call(s); the family is "
        f"{'stamped' if stamped else 'unstamped'}. Refusals previewed before the "
        f"apply: {json.dumps(initial.get('refusals') or {}, sort_keys=True)}."
    )
    lines.append("  </p>")
    lines.append(
        '  <h3 id="shc-locus-remeasure-rerun-held">Stranding detaches held</h3>'
    )
    lines.append('  <p id="shc-locus-remeasure-rerun-held-body">')
    if held:
        lines.append(
            f"    Attachment reconciliation held "
            f"{len(held)} detach(es) that would have left a live name with no "
            "producing source, left each attachment exactly as found, and "
            "proceeded with the rest: " + held_names + "."
        )
    else:
        lines.append("    Attachment reconciliation held no detach.")
    lines.append("  </p>")
    lines.append("  <table>")
    lines.append(
        "    <thead><tr><th>name</th><th>DD path</th><th>reason</th></tr></thead>"
    )
    lines.append("    <tbody>" + held_rows + "</tbody>")
    lines.append("  </table>")
    lines.append(
        '  <h3 id="shc-locus-remeasure-rerun-apply">Apply cost and member '
        "documentation score before and after</h3>"
    )
    lines.append('  <p id="shc-locus-remeasure-rerun-apply-body">')
    lines.append(
        f"    Apply log <code>{apply_entry.get('log', '—')}</code> ending "
        f"<code>{apply_entry.get('exit_line', '—')}</code>. Selected by exact run id "
        f"{run_ids_html or '—'}, the run made "
        f"<strong>{calls_total if calls_total is not None else '—'} LLMCost call(s)</strong>"
        f" in {len(cost_rows)} phase(s) ({calls_by_phase or '—'}), spending "
        f"<strong>USD {_fmt(spend)}</strong> of the USD "
        f"{float(apply_entry.get('cost_limit_usd') or 0):.2f} cap"
        + per_name_clause
        + "."
    )
    lines.append("  </p>")
    lines.append("  <table>")
    lines.append(
        "    <thead><tr><th>member</th><th>before</th><th>after</th></tr></thead>"
    )
    lines.append("    <tbody>" + score_rows + "</tbody>")
    lines.append("  </table>")
    lines.append(
        "  <p>Members below the acceptance threshold (both scores): "
        + (below_rows or "none")
        + ". Members whose score fell: "
        + (fell_rows or "none")
        + ". Members reset back into the pipeline (score to none) with their "
        "doc stage before and after: " + (reset_html or "none") + ".</p>"
    )
    lines.append("</section>")
    HTML.write_text("\n".join(lines) + "\n")
    print(f"wrote {HTML}")


def main() -> None:
    parser = argparse.ArgumentParser()
    subcommands = parser.add_subparsers(dest="command", required=True)
    subcommands.add_parser("roster")
    snap = subcommands.add_parser("snapshot")
    snap.add_argument("phase")
    subcommands.add_parser("fixture")
    rec = subcommands.add_parser("record-apply")
    rec.add_argument("log")
    rec.add_argument("wall_seconds", type=float)
    rec.add_argument(
        "--cost-limit", dest="cost_limit", type=float, default=COST_CAP_USD
    )
    rec.add_argument("--dry-run-log", dest="dry_run_log", default=None)
    rec.add_argument(
        "--run-id",
        dest="run_ids",
        action="append",
        default=None,
        help="the exact run id whose LLMCost rows this apply wrote; repeatable",
    )
    subcommands.add_parser("record-cost")
    subcommands.add_parser("render")
    args = parser.parse_args()
    if args.command == "roster":
        from imas_codex.graph.client import GraphClient
        from imas_codex.standard_names.harmonize import plan_family_applies

        with GraphClient() as graph:
            plan = plan_family_applies(graph, seeds=[SEED])[0]
        check = roster_check(plan["members"])
        payload = read_results()
        payload["roster_check"] = check
        payload["family"] = {"seed": SEED, "members": plan["members"]}
        write_results(payload)
        print(json.dumps(check, sort_keys=True))
    elif args.command == "snapshot":
        snapshot(args.phase)
    elif args.command == "record-apply":
        record_apply(
            args.log,
            args.wall_seconds,
            args.cost_limit,
            args.dry_run_log,
            args.run_ids,
        )
    elif args.command == "record-cost":
        record_cost()
    elif args.command == "render":
        render()
    else:
        fixture()


if __name__ == "__main__":
    main()
