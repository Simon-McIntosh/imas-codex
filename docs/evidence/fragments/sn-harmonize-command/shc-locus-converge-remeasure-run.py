"""Re-measure one non-WEST locus family through ``sn harmonize``.

The read-only subcommands snapshot a family's plan, report, status and
per-member documentation score.  ``record-apply`` ingests a captured apply log,
``record-cost`` reads the run's calls, per-phase spend and per-name cost from
the graph owner of the ``LLMCost`` aggregation, and ``render`` assembles the
HTML fragment from the JSON record.

``select-runs`` reads the exact run ids that own this apply's ``LLMCost``
rows from the apply log's own start and end window; the campaign summary's
``result.run_ids`` are scope ids and own none.  ``aggregate_calls_per_phase``,
``aggregate_spend_per_phase`` and ``aggregate_spend_per_name`` are imported
from ``graph_store.graph_cost_and_claims``, the module that owns them, so this
runner does not depend on the ``graph_ops`` entrypoint forwarding them.

The record answers section 4's questions for the locus kind: dry-run planned
calls, calls, spend and wall time; lint findings by kind before and after;
every member's documentation score before and after; the stamped status; the
second dry run's planned calls; and every attachment reconciliation held.
Every member below the acceptance threshold, or whose score fell, is named with
both scores, and the cost rows' name ids are compared with the family roster.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from imas_codex.standard_names.defaults import DEFAULT_MIN_SCORE
from imas_codex.standard_names.harmonize import docs_score_regressions

NODE = "shc-locus-converge-remeasure"
RESULTS = Path(__file__).with_name(f"{NODE}.json")
HTML = Path(__file__).with_name(f"{NODE}.html")
REPO_ROOT = Path(__file__).resolve().parents[4]
REVIEWS_DIR = REPO_ROOT / "imas_codex/standard_names/manifests/reviews"

SEED = "electrostatic_potential"
# A name the roster search is known to find, so a zero member count is shown to
# be a real absence rather than a search that reads nothing.
POSITIVE_CONTROL = "flux_surface_averaged_electron_density_at_plasma_boundary"
POSITIVE_CONTROL_EXPECTED = 10
COST_CAP_USD = 4.00
# Where the run ids used for cost selection come from.  The campaign summary's
# ``result.run_ids`` are scope ids and own no cost rows; the rows hang on the
# drain runs the apply starts for each batch, so they are selected from the
# ``LLMCost`` rows written inside the apply's own start/end window.
RUN_IDS_SOURCE = (
    "selected from the exact LLMCost rows written between the apply log's own "
    "'# started' and '# ended' lines; the summary's scope run ids own none"
)
RUN_IDS_UNSELECTED = "unselected: run select-runs after the apply"
# One held attachment as the reconciliation pass reports it: the DD path, the
# name it would have stranded, and the pass's reason.
HELD_ATTACHMENT = re.compile(r"(\S+) → (\S+) \(([^)]*)\)")
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


class _CostServices:
    """The only collaborator the cost owner reads off its services object."""

    from imas_codex.graph.client import GraphClient


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
    if "members" not in payload.get("family", {}):
        payload["family"] = {"seed": SEED, "members": plan["members"]}
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
            item = {
                "dd_path": match.group(1),
                "sn_id": match.group(2),
                "reason": match.group(3),
            }
            if item not in held:
                held.append(item)
    return held


def record_apply(
    log_path: str,
    wall_seconds: float,
    cost_limit: float,
    dry_run_log: str | None = None,
    run_ids: list[str] | None = None,
) -> None:
    """Ingest one apply's captured log into the record."""
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
    # The summary names scope ids, which are not the ids that carry cost rows.
    # They are recorded for the comparison; the cost selection is set later by
    # ``select-runs`` from the graph, so nothing here claims to source it.
    summary_scope_ids = [
        rid for rid in (entry.get("result") or {}).get("run_ids") or [] if rid
    ] + [rid for rid in (run_ids or []) if rid]
    joined = " ".join(ln.strip() for ln in lines)
    family = next((m.groupdict() for m in map(FAMILY_STATUS.search, [joined]) if m), {})
    totals = next((m.groupdict() for m in map(SCOPE_TOTALS.search, [joined]) if m), {})
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
            "run_ids": [],
            "run_ids_source": RUN_IDS_UNSELECTED,
            "summary_scope_run_ids": summary_scope_ids,
            "held_attachments": held_attachments(lines),
            "score_regressions": entry.get("score_regressions"),
        }
    )
    write_results(payload)
    print(json.dumps(payload["applies"][-1], sort_keys=True))


def select_runs(log_path: str) -> None:
    """Select the run ids that own this apply's cost rows, from the graph.

    The summary's ``result.run_ids`` are scope ids and own no ``LLMCost`` rows,
    so selecting on them reads zero spend.  The rows hang on the drain runs the
    apply starts per batch.  They are selected by the apply log's own start and
    end, which bounds the row set to the rows this apply could have written.
    """
    from imas_codex.graph.client import GraphClient

    lines = Path(log_path).read_text().splitlines()
    started = next((ln.split(":", 1)[1].strip() for ln in lines
                    if ln.startswith("# started:")), None)
    ended = next((ln.split(":", 1)[1].strip() for ln in lines
                  if ln.startswith("# ended:")), None)
    if not started or not ended:
        raise RuntimeError("the apply log carries no '# started'/'# ended' lines")
    payload = read_results()
    roster = set(payload.get("family", {}).get("members") or [])
    if not roster:
        raise RuntimeError("the family roster is absent from the record")
    with GraphClient() as graph:
        rows = graph.query(
            """
            MATCH (c:LLMCost)
            WHERE c.llm_at >= datetime($started) AND c.llm_at <= datetime($ended)
            RETURN c.run_id AS run_id, count(c) AS calls,
                   sum(c.llm_cost) AS usd,
                   collect(DISTINCT c.standard_name_ids) AS name_lists
            ORDER BY run_id
            """,
            started=started,
            ended=ended,
        )
    pool_ids = [row["run_id"] for row in rows]
    claimed: dict[str, list[str]] = {}
    spend: dict[str, float] = {}
    for row in rows:
        names = sorted({name for group in row["name_lists"] or []
                        for name in (group or []) if name})
        claimed[row["run_id"]] = names
        spend[row["run_id"]] = round(float(row["usd"] or 0.0), 6)
    pool_rows: dict[str, dict] = {}
    if pool_ids:
        with GraphClient() as graph:
            run_rows = graph.query(
                """
                UNWIND $ids AS rid
                MATCH (r:SNRun {id: rid})
                RETURN r.id AS id, r.cost_limit AS cap,
                       r.cost_spent AS cost_spent, r.started_at AS started_at,
                       r.stop_reason AS stop_reason
                """,
                ids=pool_ids,
            )
        pool_rows = {row["id"]: row for row in run_rows}
    entry = (payload.get("applies") or [{}])[-1]
    entry["run_ids"] = pool_ids
    entry["run_ids_source"] = RUN_IDS_SOURCE
    entry["selection_window_utc"] = {"started": started, "ended": ended}
    entry["run_selection"] = {
        "selection": RUN_IDS_SOURCE,
        "pool_run_ids": pool_ids,
        "pool_run_ids_with_cost_rows": pool_ids,
        "claimed_names": claimed,
        "pool_run_rows": pool_rows,
        "pool_run_spend": spend,
        "all_claimed_names_in_roster": all(
            set(names) <= roster for names in claimed.values()
        ),
        "summary_scope_run_ids": entry.get("summary_scope_run_ids") or [],
    }
    write_results(payload)
    print(json.dumps(entry["run_selection"], sort_keys=True))


def record_cost() -> None:
    """Record the apply's calls, per-phase spend, per-name cost and its scope.

    Every figure is selected by the exact run ids ``select-runs`` recorded.
    The per-name map is the run's cost rows apportioned to the names each row
    carries, so its keys are the names the run actually paid for; comparing
    them with the family roster is the out-of-family check.
    """
    from imas_codex.standard_names.graph_store.graph_cost_and_claims import (
        aggregate_calls_per_phase,
        aggregate_spend_per_name,
        aggregate_spend_per_phase,
    )

    payload = read_results()
    if not payload.get("applies"):
        raise RuntimeError("record the apply before its cost")
    entry = payload["applies"][-1]
    if entry.get("run_ids_source") != RUN_IDS_SOURCE or not entry.get("run_selection"):
        raise RuntimeError(
            "run select-runs first: cost is selected from the LLMCost rows the "
            "apply window owns, not from the summary's scope ids"
        )
    run_ids = [r for r in (entry.get("run_ids") or []) if r]
    if not run_ids:
        raise RuntimeError("the apply window owns no LLMCost rows to select on")
    services = _CostServices()
    phases: dict[str, dict[str, float]] = {}
    per_name: dict[str, float] = {}
    for run_id in run_ids:
        spend = aggregate_spend_per_phase(run_id, services=services)
        calls = aggregate_calls_per_phase(run_id, services=services)
        if set(spend) != set(calls):
            raise RuntimeError(f"spend and calls disagree on phases for {run_id}")
        for phase, usd in spend.items():
            slot = phases.setdefault(phase, {"usd": 0.0, "calls": 0})
            slot["usd"] += float(usd)
            slot["calls"] += int(calls[phase])
        for name, value in aggregate_spend_per_name(run_id, services=services).items():
            per_name[name] = per_name.get(name, 0.0) + float(value)
    roster = set(payload.get("family", {}).get("members") or [])
    if not roster:
        raise RuntimeError("the family roster is absent from the record")
    touched = set(per_name)
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
    entry["scope_check"] = {
        "roster": sorted(roster),
        "touched_names": sorted(touched),
        "outside_roster": sorted(touched - roster),
        "roster_names_not_touched": sorted(roster - touched),
        "outside_count": len(touched - roster),
    }
    write_results(payload)
    print(json.dumps({"cost": entry["cost"], "scope_check": entry["scope_check"]},
                     sort_keys=True))


def _score_rows(members: list[str], before: dict, after: dict) -> str:
    rows = []
    for member in members:
        rows.append(
            f"<tr><td><code>{member}</code></td>"
            f"<td>{_fmt(before.get(member))}</td><td>{_fmt(after.get(member))}</td></tr>"
        )
    return "".join(rows)


def _id(suffix: str = "") -> str:
    return NODE + (f"-{suffix}" if suffix else "")


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
    scope = apply_entry.get("scope_check") or {}

    members = initial["members"]
    scores_before = initial.get("documentation_scores") or {}
    scores_after = after.get("documentation_scores") or {}

    below = []
    fell = []
    unscored = [m for m in members if scores_after.get(m) is None]
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

    spend = cost.get("spend_usd")
    calls_total = cost.get("calls_total")
    stamped = apply_entry.get("stamped")
    crash = bool(apply_entry.get("error_line"))
    converged = bool(stamped) and not below and not fell and not unscored
    verdict = "converged" if converged else ("raised" if crash else "halted")
    halt_reasons = apply_entry.get("halt_reasons") or "; ".join(
        summary.get("halt_reasons") or []
    )
    halt_reasons = (halt_reasons or "").split(" below threshold", 1)[0].strip()
    halt_clause = f" on {halt_reasons}" if halt_reasons else ""
    lint_before = json.dumps(initial.get("lint") or {}, sort_keys=True)
    lint_after = json.dumps((after or {}).get("lint") or {}, sort_keys=True)

    cost_rows = cost.get("phases") or {}
    calls_by_phase = ", ".join(
        f"{phase} {row['calls']}" for phase, row in sorted(cost_rows.items())
    )
    per_name = cost.get("per_name_usd") or {}
    run_ids_html = ", ".join(f"<code>{r}</code>" for r in cost.get("run_ids") or [])
    if per_name:
        per_name_clause = ", apportioned by the run's own cost rows to " + ", ".join(
            f"<code>{name}</code> (USD {_fmt(value)})"
            for name, value in sorted(per_name.items())
        )
    else:
        per_name_clause = ""

    per_name_rows = "".join(
        f"<tr><td><code>{name}</code></td><td>{_fmt(value)}</td></tr>"
        for name, value in sorted(per_name.items())
    )
    outside_rows = "".join(
        f"<tr><td><code>{name}</code></td><td>{_fmt(per_name.get(name))}</td></tr>"
        for name in scope.get("outside_roster") or []
    )

    held = apply_entry.get("held_attachments") or []
    held_rows = "".join(
        f"<tr><td><code>{h['sn_id']}</code></td>"
        f"<td><code>{h['dd_path']}</code></td><td>{h['reason']}</td></tr>"
        for h in held
    )
    held_names = ", ".join(
        f"<code>{h['sn_id']}</code> (path <code>{h['dd_path']}</code>)" for h in held
    )
    dry_basis = initial.get("estimate_basis") or {}

    lines = ["<section>"]
    lines.append(
        f'  <h2 id="{_id()}">Locus family {SEED}: re-measured once its batch is claimable</h2>'
    )
    lines.append(f'  <p id="{_id("summary")}">')
    lines.append(
        f"    The non-WEST locus family <code>{SEED}</code> ran under "
        f"<code>sn harmonize --family {SEED} --apply --cost-limit "
        f"{float(apply_entry.get('cost_limit_usd') or 0):.2f}</code>. It "
        f"<strong>{verdict}</strong>"
        f"{halt_clause}. Spend <strong>USD {_fmt(spend)}</strong> over "
        f"<strong>{calls_total if calls_total is not None else '—'} LLMCost call(s)</strong>"
        + (f" ({calls_by_phase})" if calls_by_phase else "")
        + f"; wall {_fmt(apply_entry.get('wall_seconds'), 0)}s. Lint findings "
        f"{lint_before} before, {lint_after} after. {len(below)} member(s) are below "
        f"the {DEFAULT_MIN_SCORE} threshold after the apply, {len(fell)} fell and "
        f"{len(unscored)} carry no score. "
        f"{scope.get('outside_count', '—')} name(s) outside the family roster were "
        f"paid for. Attachment reconciliation held {len(held)} stranding detach(es)."
    )
    lines.append("  </p>")
    lines.append(
        f'  <p id="{_id("convergence")}"><strong>Converged:</strong> '
        + (
            "yes — stamped with every member scored at or above "
            f"{DEFAULT_MIN_SCORE} and none fallen."
            if converged
            else "no — the family is unstamped, "
            f"{len(unscored)} of {len(members)} member(s) carry no documentation "
            "score ("
            + ", ".join(f"<code>{m}</code>" for m in unscored)
            + "), and the halt reasons above stand."
        )
        + "</p>"
    )
    lines.append(f'  <h3 id="{_id("choice")}">Family choice and roster check</h3>')
    lines.append(f'  <p id="{_id("choice-body")}">')
    lines.append(
        f"    <code>{SEED}</code> is the locus family whose members appear in no "
        f"frozen WEST review roster. The search reads {check['roster_files']} files "
        f"under <code>{check['searched']}</code>. Positive control "
        f"<code>{check['positive_control']}</code> matched "
        f"{check['positive_control_match_count']} times (expected "
        f"{check['positive_control_expected']}), so a zero member count is a real "
        "absence rather than a search that reads nothing. Members: "
        + ", ".join("<code>" + m + "</code>" for m in members)
        + "."
    )
    lines.append("  </p>")
    lines.append("  <table>")
    lines.append("    <thead><tr><th>member</th><th>roster matches</th></tr></thead>")
    lines.append("    <tbody>" + roster_rows + "</tbody>")
    lines.append("  </table>")
    lines.append(f'  <h3 id="{_id("dry-run")}">Dry runs and planned calls</h3>')
    lines.append(f'  <p id="{_id("dry-run-body")}">')
    lines.append(
        f"    The first dry run planned {initial['planned_calls']} call(s), estimated "
        f"USD {_fmt(initial.get('estimate_usd'))} from {dry_basis.get('runs')} recent "
        f"run(s) and {dry_basis.get('calls')} call(s) at USD "
        f"{_fmt(dry_basis.get('mean_usd_per_call'))} per call, under a USD "
        f"{COST_CAP_USD:.2f} cap. The second dry run after the apply planned "
        f"{(after or {}).get('planned_calls', '—')} call(s); the family is "
        f"{'stamped' if stamped else 'unstamped'}. Refusals previewed before the "
        f"apply: {json.dumps(initial.get('refusals') or {}, sort_keys=True)}."
    )
    lines.append("  </p>")
    lines.append(f'  <h3 id="{_id("held")}">Stranding detaches held</h3>')
    lines.append(f'  <p id="{_id("held-body")}">')
    if held:
        lines.append(
            f"    Attachment reconciliation held {len(held)} detach(es) that would "
            "have left a live name with no producing source, left each attachment "
            "exactly as found, and proceeded with the rest: " + held_names + "."
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
        f'  <h3 id="{_id("apply")}">Apply cost and member documentation score '
        "before and after</h3>"
    )
    lines.append(f'  <p id="{_id("apply-body")}">')
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
        "<p>Members below the acceptance threshold (both scores): "
        + (below_rows or "none")
        + ". Members whose score fell: "
        + (fell_rows or "none")
        + ".</p>"
    )
    lines.append(f'  <h3 id="{_id("outside")}">Out-of-family check</h3>')
    lines.append(f'  <p id="{_id("outside-body")}">')
    lines.append(
        f"    The run's cost rows apportion to {len(scope.get('touched_names') or [])} "
        f"name(s); {scope.get('outside_count', '—')} of them are outside the "
        f"{len(members)}-member family roster. Names the roster carries that the run "
        f"paid nothing for: "
        + (
            ", ".join(f"<code>{n}</code>" for n in scope.get("roster_names_not_touched"))
            or "none"
        )
        + "."
    )
    lines.append("  </p>")
    lines.append("  <table>")
    lines.append("    <thead><tr><th>name</th><th>USD paid</th></tr></thead>")
    lines.append("    <tbody>" + per_name_rows + "</tbody>")
    lines.append("  </table>")
    if outside_rows:
        lines.append("<p>Outside the roster:</p>")
        lines.append("  <table>")
        lines.append("    <thead><tr><th>name</th><th>USD paid</th></tr></thead>")
        lines.append("    <tbody>" + outside_rows + "</tbody>")
        lines.append("  </table>")
    lines.append("</section>")
    HTML.write_text("\n".join(lines) + "\n")
    print(f"wrote {HTML}")


def verify() -> int:
    """Assert the done-when conditions from the record alone."""
    payload = read_results()
    problems: list[str] = []
    below_ref: tuple[list[str], list[str], list[str]] = ([], [], [])
    applies = payload.get("applies") or []
    if not applies:
        problems.append("no apply recorded")
        apply_entry = {}
    else:
        apply_entry = applies[-1]
    before = (payload.get("phases") or {}).get("before") or {}
    after = (payload.get("phases") or {}).get("after") or {}
    cost = apply_entry.get("cost") or {}
    if not cost:
        problems.append("no cost recorded")
    elif float(cost.get("spend_usd") or 0) > COST_CAP_USD:
        problems.append(f"spend {cost.get('spend_usd')} exceeds {COST_CAP_USD}")
    scope = apply_entry.get("scope_check") or {}
    if scope.get("outside_roster"):
        problems.append(f"names outside the roster: {scope['outside_roster']}")
    # Provenance: the cost selection must name the LLMCost window it came from
    # and agree with it, so no figure rests on the campaign summary's scope ids.
    selection = apply_entry.get("run_selection") or {}
    if apply_entry.get("run_ids_source") != RUN_IDS_SOURCE:
        problems.append(
            "run_ids_source does not state the LLMCost-window selection: "
            f"{apply_entry.get('run_ids_source')!r}"
        )
    if not selection:
        problems.append("no run selection recorded")
    else:
        cost_ids = list(cost.get("run_ids") or [])
        pool_ids = list(selection.get("pool_run_ids") or [])
        if sorted(cost_ids) != sorted(pool_ids):
            problems.append(
                f"cost run ids {cost_ids} are not the selected window ids {pool_ids}"
            )
        if sorted(apply_entry.get("run_ids") or []) != sorted(pool_ids):
            problems.append(
                f"run_ids {apply_entry.get('run_ids')} disagree with the selection"
            )
        if not selection.get("all_claimed_names_in_roster"):
            problems.append("a selected run claims a name outside the roster")
        if set(pool_ids) & set(selection.get("summary_scope_run_ids") or []):
            problems.append(
                "a selected run is a summary scope id, which owns no cost rows"
            )
        window_spend = sum((selection.get("pool_run_spend") or {}).values())
        if cost and abs(window_spend - float(cost.get("spend_usd") or 0.0)) > 0.01:
            problems.append(
                "the selected run ids' spend disagrees with the recorded cost"
            )
    # Attachment holds: one entry per (name, path, reason); the fragment's prose
    # and its table must report the same number.
    held = apply_entry.get("held_attachments") or []
    seen = {(h.get("sn_id"), h.get("dd_path"), h.get("reason")) for h in held}
    if len(seen) != len(held):
        problems.append(
            f"held_attachments carries {len(held)} entries for {len(seen)} holds"
        )
    if not before or not after:
        problems.append("a before/after snapshot is missing")
    else:
        members = before.get("members") or []
        scores_b = before.get("documentation_scores") or {}
        scores_a = after.get("documentation_scores") or {}
        missing = [m for m in members if m not in scores_b or m not in scores_a]
        if missing:
            problems.append(f"members with no before/after score: {missing}")
        fell = [
            m
            for m in members
            if scores_a.get(m) is not None
            and scores_b.get(m) is not None
            and scores_a[m] < scores_b[m]
        ]
        below_ref = (
            [m for m in members if scores_a.get(m) is not None
             and scores_a[m] < DEFAULT_MIN_SCORE],
            fell,
            [m for m in members if scores_a.get(m) is None],
        )
        if fell:
            problems.append(f"members whose score fell: {fell}")
    if not HTML.exists():
        problems.append("the HTML fragment is absent")
    else:
        rendered = HTML.read_text()
        if f'id="{NODE}-convergence"' not in rendered:
            problems.append("the HTML fragment states no convergence verdict")
        prose = re.search(r"held (\d+) stranding detach", rendered)
        if not prose:
            problems.append("the HTML fragment states no held-detach count")
        elif int(prose.group(1)) != len(held):
            problems.append(
                f"the fragment reports {prose.group(1)} held detach(es) against "
                f"{len(held)} recorded"
            )
        held_section = rendered.split(f'id="{NODE}-held"', 1)
        if len(held_section) == 2 and held:
            table_rows = held_section[1].split("<h3", 1)[0].count("<tr><td><code>")
            if table_rows != len(held):
                problems.append(
                    f"the held table carries {table_rows} row(s) against "
                    f"{len(held)} recorded hold(s)"
                )
    if problems:
        for problem in problems:
            print(f"FAIL {problem}")
        return 1
    stamped = apply_entry.get("stamped")
    print(
        "converged: " + ("yes" if stamped else "no")
        + " | stamped: " + str(stamped)
        + " | below threshold: " + str(len(below_ref[0]))
        + " | fell: " + str(len(below_ref[1]))
        + " | unscored: " + str(len(below_ref[2]))
    )
    print("OK")
    return 0


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
    sel = subcommands.add_parser("select-runs")
    sel.add_argument("log", help="the apply log carrying its '# started'/'# ended' lines")
    subcommands.add_parser("record-cost")
    subcommands.add_parser("render")
    subcommands.add_parser("verify")
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
        )
    elif args.command == "select-runs":
        select_runs(args.log)
    elif args.command == "record-cost":
        record_cost()
    elif args.command == "render":
        render()
    elif args.command == "verify":
        sys.exit(verify())
    else:
        fixture()


if __name__ == "__main__":
    main()
