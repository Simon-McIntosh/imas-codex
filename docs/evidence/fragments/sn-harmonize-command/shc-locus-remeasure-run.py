"""Measure one non-WEST locus family through ``sn harmonize``.

The read-only subcommands snapshot a family's plan, report and status.  The
``record-apply`` subcommand ingests a captured apply log.  ``render`` assembles
the HTML fragment from the JSON record.  The ``fixture`` subcommand exercises
the score-regression comparison on fixed scores, so an unchanged column passes
only because nothing fell.

The record answers section 4's questions for the locus kind: calls, spend and
wall time; lint findings by kind before and after; every member's
documentation score before and after; the stamped status; and the second dry
run's planned calls.  Every member below the acceptance threshold, or whose
score fell, is named with both scores.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from imas_codex.standard_names.defaults import DEFAULT_MIN_SCORE
from imas_codex.standard_names.harmonize import docs_score_regressions

RESULTS = Path(__file__).with_name("shc-locus-remeasure.json")
HTML = Path(__file__).with_name("shc-locus-remeasure.html")
REPO_ROOT = Path(__file__).resolve().parents[4]
REVIEWS_DIR = REPO_ROOT / "imas_codex/standard_names/manifests/reviews"

SEED = "electrostatic_potential"
# A name the roster search is known to find, so a zero member count is shown
# to be a real absence rather than a search that reads nothing.
POSITIVE_CONTROL = "flux_surface_averaged_electron_density_at_plasma_boundary"
POSITIVE_CONTROL_EXPECTED = 10
COST_CAP_USD = 4.00


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


def record_apply(
    log_path: str,
    wall_seconds: float,
    cost_limit: float,
    spend_usd: float | None = None,
) -> None:
    """Ingest one apply's captured log into the record.

    The log's last JSON array line is the CLI's machine summary: per family
    its decision, anchor, applied and stamped flags, the campaign summary (run
    ids and total cost) and the documentation-score regressions.  The spend is
    the campaign summary's ``total_cost``, which apportions the run's LLMCost
    rows.
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
    payload = read_results()
    payload.setdefault("applies", []).append(
        {
            "log": str(log),
            "wall_seconds": wall_seconds,
            "cost_limit_usd": cost_limit,
            "cost_limit_unspent_usd": (
                round(cost_limit - spend_usd, 4) if spend_usd is not None else None
            ),
            "spend_usd": spend_usd,
            "exit_line": exit_line,
            "produced_summary": summary is not None,
            "error_line": error_line,
            "physical_base": entry.get("physical_base"),
            "anchor": entry.get("anchor"),
            "decision": entry.get("decision"),
            "applied": entry.get("applied"),
            "stamped": entry.get("stamped"),
            "summary": entry.get("result"),
            "score_regressions": entry.get("score_regressions"),
        }
    )
    write_results(payload)
    print(json.dumps(payload["applies"][-1], sort_keys=True))


def _score_rows(members: list[str], before: dict, after: dict) -> str:
    rows = []
    for member in members:
        rows.append(
            f"<tr><td><code>{member}</code></td><td>{_fmt(before.get(member))}</td><td>{_fmt(after.get(member))}</td></tr>"
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

    # A member the drain reset re-entered the pipeline and is
    # unscored again: its documentation score went from a value to none.
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

    spend = apply_entry.get("spend_usd")
    touched = summary.get("total_touched")
    stamped = apply_entry.get("stamped")
    crash = not apply_entry.get("produced_summary", True)
    verdict = "converged" if stamped else ("raised" if crash else "halted")
    halt_reasons = "; ".join(summary.get("halt_reasons") or [])
    halt_clause = f" on {halt_reasons}" if halt_reasons else ""
    lint_before = json.dumps(initial.get("lint") or {}, sort_keys=True)
    lint_after = json.dumps(after.get("lint") or {}, sort_keys=True)

    lines = ["<section>"]
    lines.append(
        f'  <h2 id="shc-locus-remeasure">Locus family {SEED}: convergence and cost</h2>'
    )
    lines.append('  <p id="shc-locus-remeasure-summary">')
    lines.append(
        f"    The non-WEST locus family <code>{SEED}</code> ran under "
        f"<code>sn harmonize --apply --cost-limit {COST_CAP_USD:.2f}</code>. It "
        f"<strong>{verdict}</strong>{halt_clause}. Spend "
        f"<strong>USD {_fmt(spend)}</strong>"
        + (f" over {_fmt(touched, 0)} touched name(s)" if touched is not None else "")
        + f"; wall {_fmt(apply_entry.get('wall_seconds'), 0)}s. Lint findings "
        f"{lint_before} before, {lint_after} after. {len(below)} member(s) are below "
        f"the {DEFAULT_MIN_SCORE} threshold after the apply and {len(fell)} fell."
    )
    if crash:
        lines.append(
            "    The apply reached no summary line: it raised <code>"
            + str(apply_entry.get("error_line"))
            + "</code>, so the family is unstamped."
        )
    lines.append("  </p>")
    lines.append(
        '  <h3 id="shc-locus-remeasure-choice">Family choice and roster check</h3>'
    )
    lines.append('  <p id="shc-locus-remeasure-choice-body">')
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
        '  <h3 id="shc-locus-remeasure-dry-run">Dry run and planned calls</h3>'
    )
    lines.append('  <p id="shc-locus-remeasure-dry-run-body">')
    lines.append(
        f"    The dry run planned {initial['planned_calls']} call(s), estimated USD "
        f"{_fmt(initial.get('estimate_usd'))} from "
        f"{initial.get('estimate_basis', {}).get('runs')} recent run(s). The second "
        f"dry run after the apply planned "
        f"{(after or {}).get('planned_calls', '—')} call(s); the family is "
        f"{'stamped' if stamped else 'unstamped'}."
    )
    lines.append("  </p>")
    lines.append(
        '  <h3 id="shc-locus-remeasure-apply">Member documentation score before and after</h3>'
    )
    lines.append('  <p id="shc-locus-remeasure-apply-body">')
    lines.append(
        f"    Apply log <code>{apply_entry.get('log', '—')}</code> ending "
        f"<code>{apply_entry.get('exit_line', '—')}</code>. Spend "
        f"<strong>USD {_fmt(spend)}</strong> of the USD {COST_CAP_USD:.2f} "
        f"cap, from the run's LLMCost documentation-phase rows read while the "
        f"graph was reachable; a later graph authentication outage prevented "
        f"re-reading the per-name apportionment, so the per-name split is not "
        f"recorded."
    )
    lines.append("  </p>")
    lines.append("  <table>")
    lines.append("    <thead><tr><th>member</th><th>before</th><th>after</th>")
    lines.append("    <th></th></tr></thead>")
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
    rec.add_argument("--spend", dest="spend", type=float, default=None)
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
        record_apply(args.log, args.wall_seconds, args.cost_limit, args.spend)
    elif args.command == "render":
        render()
    else:
        fixture()


if __name__ == "__main__":
    main()
