"""Re-apply the locus family now that the contradiction veto judges every binding.

The read-only subcommands snapshot a family's plan, report, status and per-member
documentation score.  ``record-dry-run`` captures a dry run's planned calls,
``record-apply`` ingests a captured apply log, ``record-vetoes`` reads every
contradiction veto the apply raised with the claim and each binding's own answer,
``record-cost`` reads the run's calls, per-phase spend and per-name cost from the
graph owner of the ``LLMCost`` aggregation, and ``render`` assembles the HTML
fragment from the JSON record.

``select-runs`` reads the exact run ids that own this apply's ``LLMCost`` rows
from the apply log's own start and end window; the campaign summary's
``result.run_ids`` are scope ids and own none.  ``aggregate_calls_per_phase``,
``aggregate_spend_per_phase`` and ``aggregate_spend_per_name`` are imported from
``graph_store.graph_cost_and_claims``, the module that owns them.

The veto record answers the case this node exists for: a claim stating a base
name's scope is affirmed by the binding that carries that scope, so a claim no
binding supports is the only one that may be held.  Each vetoed claim is
recorded with the answer every one of the name's DD bindings gave it, and the
fold is recomputed here from those answers rather than trusted from the log.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

from imas_codex.standard_names.defaults import DEFAULT_MIN_SCORE
from imas_codex.standard_names.harmonize import docs_score_regressions

NODE = "shc-locus-reapply-all-bindings"
RESULTS = Path(__file__).with_name(f"{NODE}.json")
HTML = Path(__file__).with_name(f"{NODE}.html")
REPO_ROOT = Path(__file__).resolve().parents[4]
REVIEWS_DIR = REPO_ROOT / "imas_codex/standard_names/manifests/reviews"

SEED = "electrostatic_potential"
# A name the roster search is known to find, so a zero member count is shown to
# be a real absence rather than a search that reads nothing.
POSITIVE = "flux_surface_averaged_electron_density_at_plasma_boundary"
POSITIVE_CONTROL = POSITIVE
POSITIVE_CONTROL_EXPECTED = 10
COST_CAP_USD = 4.00
MAX_COST_USD = COST_CAP_USD
RUN_IDS_SOURCE = (
    "selected from the exact LLMCost rows written between the apply log's own "
    "'# started' and '# ended' lines; the summary's scope run ids own none"
)
RUN_IDS_UNSELECTED = "unselected: run select-runs after the apply"
FAMILY_STATUS = re.compile(
    r"family (?P<base>\S+)\s+anchor=(?P<anchor>\S+)\s+batches=(?P<batches>\d+)\s+"
    r"accepted=(?P<accepted>\d+)/(?P<touched>\d+)\s+"
    r"(?P<status>converged|halted)(?:: (?P<reasons>.*?))?"
    r"(?=\s+(?:apply:|docs scores:|fell\b)|$)"
)
SCOPE_TOTALS = re.compile(
    r"apply: (?P<applied>\d+) family\(ies\) applied, (?P<stamped>\d+) stamped\."
)
CLAIM_KEY = re.compile(
    r"^check:claim_contradicts_source:(?P<index>\d+)(?::binding:(?P<binding>\d+))?$"
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
                       member.validation_status AS validation_status,
                       member.reviewer_score_docs AS reviewer_score_docs
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
            "documentation_scores": plan["member_scores"],
            "member_status": {row["id"]: row for row in states},
            "status_counts": {"accepted": accepted, "other": len(states) - accepted},
            "read_at_utc": datetime.now(UTC).isoformat(),
        }
        data["drift"] = report["drift"]
    payload = read_results()
    payload.setdefault("phases", {})[phase] = data
    payload["family"] = {"seed": SEED, "members": plan["members"]}
    write_results(payload)
    print(json.dumps(data, sort_keys=True))


def fixture() -> None:
    name = "fixture_low_scoring_member"
    before = {name: 0.95, "fixture_stable_member": 0.9}
    after = {name: round(DEFAULT_MIN_SCORE - 0.05, 2), "fixture_stable_member": 0.9}
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


def record_dry_run(log_path: str, label: str) -> None:
    """Record one dry run's planned calls and estimate, from its own log."""
    lines = Path(log_path).read_text().splitlines()
    text = "\n".join(lines)
    planned = re.findall(
        r"(\d+)\s+planned\s+call|planned[_ ]calls?[ =:]+(\d+)", text, re.IGNORECASE
    )
    planned = [int(pair[0] or pair[1]) for pair in planned]
    entry = {
        "label": label,
        "log": log_path,
        "exit_line": next(
            (ln for ln in reversed(lines) if ln.startswith("EXIT=")), "EXIT=<absent>"
        ),
        "planned_calls": [int(value) for value in planned],
        "lines": [ln for ln in lines if "planned" in ln.lower()][:20],
    }
    payload = read_results()
    kept = [
        prior
        for prior in payload.get("dry_runs") or []
        if Path(prior.get("log") or "").resolve() != Path(log_path).resolve()
    ]
    payload["dry_runs"] = [*kept, entry]
    write_results(payload)
    print(json.dumps(entry, sort_keys=True))


def held_attachments(lines: list[str]) -> list[dict]:
    held: list[dict] = []
    for line in lines:
        if "would leave these live names with no producing source" not in line:
            continue
        tail = line.split("retire the name: ", 1)[-1]
        for match in re.finditer(r"(\S+) → (\S+) \(([^)]*)\)", tail):
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
            if "Error:" in ln or "EXIT=" in ln and "EXIT=0" not in ln
        ),
        None,
    )
    entry = summary[0] if summary else {}
    summary_scope_ids = [
        rid for rid in (entry.get("result") or {}).get("run_ids") or [] if rid
    ]
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


def _window(log_path: str) -> tuple[str, str]:
    lines = Path(log_path).read_text().splitlines()
    started = next(
        (ln.split(":", 1)[1].strip() for ln in lines if ln.startswith("# started:")),
        None,
    )
    ended = next(
        (ln.split(":", 1)[1].strip() for ln in lines if ln.startswith("# ended:")),
        None,
    )
    if not started or not ended:
        raise RuntimeError("the apply log carries no '# started'/'# ended' lines")
    return started, ended


def select_runs(log_path: str) -> None:
    """Select the run ids that own this apply's cost rows, from the graph."""
    from imas_codex.graph.client import GraphClient

    started, ended = _window(log_path)
    payload = read_results()
    roster = set(payload.get("family", {}).get("members") or [])
    if not roster:
        raise RuntimeError("the apply grid: family roster is absent from the record")
    with GraphClient() as graph:
        rows = graph.query(
            """
            MATCH (c:LLMCost)
            WHERE c.llm_at >= datetime($started) AND c.llm_at <= datetime($ended)
            RETURN c.run_id AS run_id, count(c) AS calls,
                   collect(DISTINCT c.standard_name_ids) AS name_lists
            ORDER BY run_id
            """,
            started=started,
            ended=ended,
        )
        pool_ids = [row["run_id"] for row in rows]
        pool_rows: dict[str, dict] = {}
        if pool_ids:
            run_rows = graph.query(
                """
                UNWIND $ids AS rid
                MATCH (r:SNRun {id: rid})
                RETURN r.id AS id, r.cost_limit AS cap,
                       r.cost_spent AS cost_spent, r.stop_reason AS stop_reason
                """,
                ids=pool_ids,
            )
            pool_rows = {row["id"]: row for row in run_rows}
    claimed: dict[str, list[str]] = {}
    for row in rows:
        claimed[row["run_id"]] = sorted(
            {
                name
                for group in row["name_lists"] or []
                for name in (group or [])
                if name
            }
        )
    entry = (payload.get("applies") or [{}])[-1]
    entry["run_ids"] = pool_ids
    entry["run_ids_source"] = RUN_IDS_SOURCE
    entry["selection_window_utc"] = {"started": started, "ended": ended}
    entry["run_selection"] = {
        "selection": RUN_IDS_SOURCE,
        "pool_run_ids": pool_ids,
        "claimed_names": claimed,
        "pool_run_rows": pool_rows,
        "all_claimed_names_in_roster": all(
            set(names) <= roster for names in claimed.values()
        ),
        "summary_scope_run_ids": entry.get("summary_scope_run_ids") or [],
    }
    write_results(payload)
    print(json.dumps(entry["run_selection"], sort_keys=True))


def record_cost() -> None:
    """Record the apply's calls, per-phase spend, per-name cost and its scope."""
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
    print(
        json.dumps(
            {"cost": entry["cost"], "scope_check": entry["scope_check"]},
            sort_keys=True,
        )
    )


def _rehydrate(graph, names: list[str]) -> dict[str, dict]:
    """Rebuild the review item each name's claims and binding order were read from."""
    rows = graph.query(
        """
        UNWIND $ids AS nid
        MATCH (sn:StandardName {id: nid})
        RETURN sn.id AS id, sn.documentation AS documentation,
               sn.description AS description, sn.standard_name AS standard_name,
               sn.unit AS unit, sn.source_paths AS source_paths,
               sn.links AS links
        """,
        ids=names,
    )
    items = {row["id"]: dict(row) for row in rows or []}
    paths = sorted(
        {
            path.removeprefix("dd:")
            for item in items.values()
            for path in (item.get("source_paths") or [])
        }
    )
    docs: dict[str, dict] = {}
    if paths:
        for row in (
            graph.query(
                """
            UNWIND $paths AS pid
            MATCH (n:IMASNode {id: pid})
            RETURN n.id AS id, n.unit AS unit, n.description AS description,
                   n.documentation AS documentation
            """,
                paths=paths,
            )
            or []
        ):
            docs[row["id"]] = dict(row)
    for item_id, item in items.items():
        item["id"] = item.pop("id") or item_id
        source_paths = item.get("source_paths") or []
        if isinstance(source_paths, str):
            try:
                source_paths = json.loads(source_paths)
            except json.JSONDecodeError:
                source_paths = [source_paths]
        item["source_paths"] = [path.removeprefix("dd:") for path in source_paths]
        item["dd_source_docs"] = [docs[p] for p in item["source_paths"] if p in docs]
    return items


def record_vetoes(log_path: str) -> None:
    """Record every contradiction veto the apply raised, per binding.

    Each vetoed claim is recorded with the answer every one of the name's DD
    bindings gave that claim, and the claim-level fold is recomputed here from
    those answers.  A claim some binding supports is exactly the claim the
    all-bindings rule says must not be held, so a veto whose fold is
    ``supports`` is recorded for the coordinator rather than trusted.
    """
    from imas_codex.graph.client import GraphClient
    from imas_codex.standard_names.review.jev_review import (
        aggregate_binding_answers,
        build_docs_state,
        claim_contradicts,
        settled_claim_answers,
    )

    started, ended = _window(log_path)
    payload = read_results()
    roster = list(payload.get("family", {}).get("members") or [])
    if not roster:
        raise RuntimeError("the family roster is absent from the record")
    with GraphClient() as graph:
        reviews = (
            graph.query(
                """
            MATCH (sn:StandardName)-[:HAS_REVIEW]->(rv:StandardNameReview)
            WHERE sn.id IN $ids AND rv.review_axis = 'docs'
              AND rv.reviewed_at >= datetime($started)
              AND rv.reviewed_at <= datetime($ended)
            RETURN rv.id AS id, sn.id AS name_id, rv.model AS model,
                   toString(rv.reviewed_at) AS reviewed_at,
                   rv.judge_answers_json AS answers,
                   rv.judge_confidence_json AS confidences,
                   rv.comments AS comments
            ORDER BY reviewed_at, id
            """,
                ids=roster,
                started=started,
                ended=ended,
            )
            or []
        )
        holds = (
            graph.query(
                """
            MATCH (sn:StandardName)-[:HAS_HOLD]->(hold:ReviewHold)
            WHERE sn.id IN $ids AND hold.raised_at >= datetime($started)
              AND hold.raised_at <= datetime($ended)
            RETURN hold.id AS id, sn.id AS name_id, hold.kind AS kind,
                   hold.status AS status, hold.finding AS finding,
                   hold.evidence AS evidence, toString(hold.raised_at) AS raised_at
            ORDER BY raised_at, id
            """,
                ids=roster,
                started=started,
                ended=ended,
            )
            or []
        )
        items = _rehydrate(graph, roster)

    claims_by_name: dict[str, list[str]] = {}
    bindings_by_name: dict[str, list[str]] = {}
    settled_by_name: dict[str, dict[int, dict]] = {}
    for name in roster:
        state = build_docs_state(items.get(name) or {"id": name})
        claims_by_name[name] = list(state["documentation_claims"])
        bindings_by_name[name] = list(state.get("binding_ids") or [])
        settled_by_name[name] = settled_claim_answers(state)

    records: list[dict] = []
    for review in reviews:
        answers = json.loads(review.get("answers") or "{}")
        confidences = json.loads(review.get("confidences") or "{}")
        per_claim: dict[int, dict[int, dict]] = {}
        claim_level: dict[int, dict] = {}
        for key, value in answers.items():
            match = CLAIM_KEY.match(key)
            if not match:
                continue
            index = int(match.group("index"))
            if match.group("binding") is None:
                claim_level[index] = {
                    "choice": value,
                    "confidence": confidences.get(key),
                }
                continue
            per_claim.setdefault(index, {})[int(match.group("binding"))] = {
                "choice": value,
                "confidence": confidences.get(key),
            }
        claims = claims_by_name.get(review["name_id"]) or []
        labels = bindings_by_name.get(review["name_id"]) or []
        settled = settled_by_name.get(review["name_id"]) or {}
        claims_read: list[dict] = []
        for index in sorted(set(per_claim) | set(claim_level)):
            claim = claims[index] if index < len(claims) else None
            by_binding = per_claim.get(index) or {}
            if by_binding:
                ordered = [
                    {
                        "position": position,
                        "binding": labels[position] if position < len(labels) else None,
                        "answer": by_binding[position]["choice"],
                        "confidence": by_binding[position]["confidence"],
                    }
                    for position in sorted(by_binding)
                ]
                fold = aggregate_binding_answers(
                    [
                        {"choice": entry["answer"], "probabilities": {}}
                        for entry in ordered
                    ]
                )
                folded = (fold or {}).get("choice")
            else:
                ordered = []
                folded = (claim_level.get(index) or {}).get("choice")
            contradicts = [
                entry["binding"]
                for entry in ordered
                if entry["answer"] == "contradicts"
            ]
            supports = [
                entry["binding"] for entry in ordered if entry["answer"] == "supports"
            ]
            settled_answer = settled.get(index)
            claims_read.append(
                {
                    "index": index,
                    "claim": claim,
                    "bindings": ordered,
                    "claim_level_answer": (claim_level.get(index) or {}).get("choice"),
                    "folded": folded,
                    "settled_by": (settled_answer or {}).get("settled_by"),
                    # A claim the code settles is answered by the code, so no
                    # reader vetoes it whatever the judge returned for it.
                    "vetoed": settled_answer is None
                    and (
                        folded == "contradicts"
                        or (
                            folded is not None and claim_contradicts({"choice": folded})
                        )
                    ),
                    "bindings_contradicting": contradicts,
                    "bindings_supporting": supports,
                }
            )
        vetoed = [claim for claim in claims_read if claim["vetoed"]]
        records.append(
            {
                "review_id": review["id"],
                "name_id": review["name_id"],
                "model": review["model"],
                "reviewed_at": review["reviewed_at"],
                "binding_labels": labels,
                "binding_count": len(labels),
                "claims_read": len(claims_read),
                "claims_with_per_binding_answers": len(per_claim),
                "vetoed_claims": vetoed,
                "veto_count": len(vetoed),
                "claims_with_a_contradicting_binding": [
                    claim for claim in claims_read if claim["bindings_contradicting"]
                ],
                "held_by_a_binding_that_supports_it": [
                    claim for claim in vetoed if claim["bindings_supporting"]
                ],
            }
        )
    family_vetoes = [record for record in records if record["veto_count"]]
    payload["vetoes"] = {
        "window_utc": {"started": started, "ended": ended},
        "reviews_read": len(records),
        "reviews": records,
        "holds": holds,
        "veto_count": sum(record["veto_count"] for record in records),
        "held_claim_names_a_supporting_binding": [
            {
                "name_id": record["name_id"],
                "claim": claim["claim"],
                "bindings_supporting": claim["bindings_supporting"],
            }
            for record in records
            for claim in record["held_by_a_binding_that_supports_it"]
        ],
        "names_vetoed": sorted({record["name_id"] for record in family_vetoes}),
    }
    write_results(payload)
    print(
        json.dumps(
            {
                key: payload["vetoes"][key]
                for key in (
                    "reviews_read",
                    "veto_count",
                    "names_vetoed",
                    "held_claim_names_a_supporting_binding",
                )
            },
            sort_keys=True,
        )
    )


def _score_rows(members: list[str], before: dict, after: dict) -> str:
    return "".join(
        f"<tr><td><code>{member}</code></td>"
        f"<td>{_fmt(before.get(member))}</td><td>{_fmt(after.get(member))}</td></tr>"
        for member in members
    )


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
    vetoes = payload.get("vetoes") or {}
    dry_runs = payload.get("dry_runs") or []

    members = initial["members"]
    scores_before = initial.get("documentation_scores") or {}
    scores_after = (
        after.get("documentation_scores") and after["documentation_scores"] or {}
    )
    below, fell = [], []
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

    spend = cost.get("spend_usd")
    calls_total = cost.get("calls_total")
    stamped = apply_entry.get("stamped")
    converged = bool(stamped) and not below and not fell and not unscored
    verdict = "converged" if converged else "halted"
    halt_reasons = apply_entry.get("halt_reasons") or "; ".join(
        summary.get("halt_reasons") or []
    )
    lint_before = json.dumps(initial.get("lint") or {}, sort_keys=True)
    lint_after = json.dumps(after.get("lint") or {}, sort_keys=True)
    cost_rows = cost.get("phases") or {}
    calls_by_phase = ", ".join(
        f"{phase} {row['calls']}" for phase, row in sorted(cost_rows.items())
    )
    per_name = cost.get("per_name_usd") or {}
    run_ids_html = ", ".join(f"<code>{r}</code>" for r in cost.get("run_ids") or [])
    below_rows = "; ".join(
        f"<code>{m}: {_fmt(b)} &rarr; {_fmt(a)}</code>" for m, b, a in below
    )
    fell_rows = "; ".join(
        f"<code>{m}: {_fmt(b)} &rarr; {_fmt(a)}</code>" for m, b, a in fell
    )
    score_rows = _score_rows(members, scores_before, scores_after)
    roster_rows = "".join(
        f"<tr><td><code>{name}</code></td><td>{count}</td></tr>"
        for name, count in sorted(check["member_matches"].items())
    )
    per_name_rows = "".join(
        f"<tr><td><code>{name}</code></td><td>{_fmt(value)}</td></tr>"
        for name, value in sorted(per_name.items())
    )
    dry_basis = initial.get("estimate_basis") or {}
    dry_plan = dry_runs[0]["planned_calls"] if dry_runs else initial["planned_calls"]
    dry_plan = ", ".join(str(value) for value in dry_plan or [])
    second_dry = next(
        (run for run in dry_runs if run["label"].startswith("after")), None
    )
    second_dry_calls = second_dry["planned_calls"] if second_dry else None
    if isinstance(second_dry_calls, list):
        second_dry_calls = ", ".join(str(v) for v in second_dry_calls) or None

    # Veto section: every claim a veto rests on, with every binding's answer.
    veto_rows = []
    for record in vetoes.get("reviews") or []:
        for claim in record["vetoed_claims"]:
            answers = (
                "; ".join(
                    f"<code>{entry['binding']}</code>: {entry['answer']}"
                    for entry in claim["bindings"]
                )
                or "(claim-level answer only)"
            )
            veto_rows.append(
                f"<tr><td><code>{record['name_id']}</code></td>"
                f"<td>{claim['claim'] or '—'}</td><td>{answers}</td>"
                f"<td>{claim['folded']}</td></tr>"
            )
    reviewed_rows = []
    # Claims some binding contradicted but that were not held: the fold or the
    # code's own settlement is what spared them, and the table shows which.
    spared_rows = []
    for record in vetoes.get("reviews") or []:
        for claim in record.get("claims_with_a_contradicting_binding") or []:
            answers = (
                "; ".join(
                    f"<code>{entry['binding']}</code>: {entry['answer']}"
                    for entry in claim["bindings"]
                )
                or "(claim-level answer only)"
            )
            if claim["vetoed"]:
                continue
            spared_rows.append(
                f"<tr><td><code>{record['name_id']}</code></td>"
                f"<td>{claim['claim'] or '—'}</td><td>{answers}</td>"
                f"<td>{claim['folded']}</td>"
                f"<td>{claim.get('settled_by') or 'not settled — the fold spared it'}</td>"
                f"</tr>"
            )
        reviewed_rows.append(
            f"<tr><td><code>{record['name_id']}</code></td>"
            f"<td>{record['binding_count']}</td><td>{record['claims_read']}</td>"
            f"<td>{record['claims_with_per_binding_answers']}</td>"
            f"<td>{record['veto_count']}</td></tr>"
        )
    held_hold_rows = "".join(
        f"<tr><td><code>{hold['name_id']}</code></td><td>{hold['kind']}</td>"
        f"<td>{hold['status']}</td><td>{hold.get('evidence') or '—'}</td></tr>"
        for hold in (vetoes.get("holds") or [])
    )
    under_supported = vetoes.get("held_claim_names_a_supporting_binding") or []

    lines = ["<section>"]
    lines.append(
        f'  <h2 id="{_id()}">Locus family {SEED}: re-applied with the veto judging every binding</h2>'
    )
    lines.append(f'  <p id="{_id("summary")}">')
    lines.append(
        f"    The non-WEST locus family <code>{SEED}</code> ran under "
        f"<code>sn harmonize --family {SEED} --apply --cost-limit "
        f"{float(apply_entry.get('cost_limit_usd') or 0):.2f}</code>. Its own family "
        f"gate reported <strong>"
        f"{apply_entry.get('members_accepted', '—')}/"
        f"{apply_entry.get('members_touched', '—')} member(s) accepted, "
        f"{'halted' if apply_entry.get('halted') else 'converged'}</strong>, and it "
        f"<strong>{'stamped' if stamped else 'stamped nothing'}</strong>. By this "
        f"node's convergence definition (every member scored, none fallen, family "
        f"stamped) the family is <strong>{verdict}</strong>"
        + (f" on {halt_reasons}" if halt_reasons and not converged else "")
        + f". Spend <strong>USD {_fmt(spend)}</strong> over "
        f"<strong>{calls_total if calls_total is not None else '—'} LLMCost call(s)</strong>"
        + (f" ({calls_by_phase})" if calls_by_phase else "")
        + f"; wall {_fmt(apply_entry.get('wall_seconds'), 0)}s. Lint findings "
        f"{lint_before} before, {lint_after} after. {len(below)} member(s) are below "
        f"the {DEFAULT_MIN_SCORE} threshold after the apply, {len(fell)} fell and "
        f"{len(unscored)} carry no score. "
        f"{scope.get('outside_count', '—')} name(s) outside the family roster were "
        f"paid for. Contradiction vetoes raised: "
        f"<strong>{vetoes.get('veto_count', '—')}</strong>."
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
            + (", ".join(f"<code>{m}</code>" for m in unscored) or "none")
            + f"), {len(below)} fall below {DEFAULT_MIN_SCORE}, and "
            f"{len(fell)} fell on re-review ("
            + (
                ", ".join(
                    f"<code>{m}</code>: {_fmt(b)} &rarr; {_fmt(a)}" for m, b, a in fell
                )
                or "none"
            )
            + ")."
        )
        + "</p>"
    )
    lines.append(f'  <h3 id="{_id("dry-run")}">Dry runs and planned calls</h3>')
    lines.append(f'  <p id="{_id("dry-run-body")}">')
    lines.append(
        f"    The first dry run planned {dry_plan} call(s), estimated USD "
        f"{_fmt(initial.get('estimate_usd'))} from {dry_basis.get('runs')} recent "
        f"run(s) and {dry_basis.get('calls')} call(s) at USD "
        f"{_fmt(dry_basis.get('mean_usd_per_call'))} per call, under a USD "
        f"{COST_CAP_USD:.2f} cap. The second dry run after the apply planned "
        f"{second_dry_calls or (after or {}).get('planned_calls', '—')} call(s); the "
        f"family is {'stamped' if stamped else 'unstamped'}. Refusals previewed "
        f"before the apply: {json.dumps(initial.get('refusals') or {}, sort_keys=True)}."
    )
    lines.append("  </p>")
    lines.append(
        f'  <h3 id="{_id("veto")}">Contradiction vetoes: the claim and every binding</h3>'
    )
    lines.append(f'  <p id="{_id("veto-body")}">')
    lines.append(
        f"    The apply window holds {vetoes.get('reviews_read', '—')} docs review "
        f"row(s) for the family and raised "
        f"<strong>{vetoes.get('veto_count', '—')}</strong> contradiction veto(es) "
        f"across {len(vetoes.get('names_vetoed') or [])} name(s): "
        + (
            ", ".join(f"<code>{n}</code>" for n in vetoes.get("names_vetoed") or [])
            or "none"
        )
        + ". Each claim was read against every DD binding the name carries and the "
        "claim-level fold is recomputed from those answers. Claims held while at "
        "least one binding supports them: "
        f"<strong>{len(under_supported)}</strong>."
    )
    lines.append("  </p>")
    lines.append("  <table>")
    lines.append(
        "    <thead><tr><th>name</th><th>bindings</th><th>claims read</th>"
        "<th>claims per binding</th><th>vetoed</th></tr></thead>"
    )
    lines.append("    <tbody>" + "".join(reviewed_rows) + "</tbody>")
    lines.append("  </table>")
    if veto_rows:
        lines.append("  <table>")
        lines.append(
            "    <thead><tr><th>name</th><th>vetoed claim</th>"
            "<th>each binding's answer</th><th>fold</th></tr></thead>"
        )
        lines.append("    <tbody>" + "".join(veto_rows) + "</tbody>")
        lines.append("  </table>")
    if spared_rows:
        lines.append(
            f'  <h3 id="{_id("spared")}">Claims a single binding contradicted that '
            "were not held</h3>"
        )
        lines.append(
            "  <p>Under the earlier per-source reading each of these was held on one "
            "binding's answer. The fold and the code's own settlement now decide, and "
            "none was held.</p>"
        )
        lines.append("  <table>")
        lines.append(
            "    <thead><tr><th>name</th><th>claim</th>"
            "<th>each binding's answer</th><th>fold</th><th>why not held</th>"
            "</tr></thead>"
        )
        lines.append("    <tbody>" + "".join(spared_rows) + "</tbody>")
        lines.append("  </table>")
    if held_hold_rows:
        lines.append(
            "  <table><thead><tr><th>name</th><th>hold kind</th><th>status</th>"
            "<th>evidence</th></tr></thead><tbody>"
            + held_hold_rows
            + "</tbody></table>"
        )
    lines.append(
        f'  <h3 id="{_id("apply")}">Apply cost and member documentation score before and after</h3>'
    )
    lines.append(f'  <p id="{_id("apply-body")}">')
    lines.append(
        f"    Apply log <code>{apply_entry.get('log', '—')}</code> ending "
        f"<code>{apply_entry.get('exit_line', '—')}</code>. Selected by exact run id "
        f"{run_ids_html or '—'}, the run made "
        f"<strong>{calls_total if calls_total is not None else '—'} LLMCost call(s)</strong>"
        f" in {len(cost_rows)} phase(s) ({calls_by_phase or '—'}), spending "
        f"<strong>USD {_fmt(spend)}</strong> of the USD "
        f"{float(apply_entry.get('cost_limit_usd') or 0):.2f} cap."
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
            ", ".join(
                f"<code>{n}</code>" for n in scope.get("roster_names_not_touched")
            )
            or "none"
        )
        + "."
    )
    lines.append(
        "  </p>",
    )
    lines.append("  <table>")
    lines.append("    <thead><tr><th>name</th><th>USD paid</th></tr></thead>")
    lines.append("    <tbody>" + per_name_rows + "</tbody>")
    lines.append("  </table>")
    lines.append(
        f'  <h3 id="{_id("roster")}">Family roster and review-roster check</h3>'
    )
    lines.append(f'  <p id="{_id("roster-body")}">')
    lines.append(
        f"    The search reads {check['roster_files']} files under "
        f"<code>{check['searched']}</code>. Positive control "
        f"<code>{check['positive_control']}</code> matched "
        f"{check['positive_control_match_count']} times (expected "
        f"{check['positive_control_expected']}), so a zero member count is a real "
        "absence rather than a search that reads nothing. Members: "
        + ", ".join("<code>" + name + "</code>" for name in check["member_matches"])
        + "."
    )
    lines.append("  </p>")
    lines.append(
        "<table><thead><tr><th>member</th><th>matches</th></tr></thead><tbody>"
        + roster_rows
        + "</tbody></table>"
    )
    lines.append("</section>")
    HTML.write_text("\n".join(lines) + "\n")
    print(f"wrote {HTML}")


def verify() -> int:
    """Assert the done-when conditions from the record alone."""
    payload = read_results()
    problems: list[str] = []
    applies = payload.get("applies") or []
    apply_entry = applies[-1] if applies else {}
    before = (payload.get("phases") or {}).get("before") or {}
    after = (payload.get("phases") or {}).get("after") or {}
    cost = apply_entry.get("cost") or {}
    if not applies:
        problems.append("no apply recorded")
    if not cost:
        problems.append("no cost recorded")
    elif float(cost.get("spend_usd") or 0) > MAX_COST_USD:
        problems.append(f"spend {cost.get('spend_usd')} exceeds {MAX_COST_USD}")
    scope = apply_entry.get("scope_check") or {}
    if scope.get("outside_roster"):
        problems.append(f"names outside the roster: {scope['outside_roster']}")
    selection = apply_entry.get("run_selection") or {}
    if apply_entry.get("run_ids_source") != RUN_IDS_SOURCE:
        problems.append("run_ids_source does not state the LLMCost-window selection")
    if not selection:
        problems.append("no run selection recorded")
    else:
        pool_ids = list(selection.get("pool_run_ids") or [])
        if sorted(cost.get("run_ids") or []) != sorted(pool_ids):
            problems.append("cost run ids are not the selected window ids")
        if not selection.get("all_claimed_names_in_roster"):
            problems.append("a selected run claims a name outside the roster")
    vetoes = payload.get("vetoes")
    if not vetoes:
        problems.append("no veto record")
    else:
        if vetoes.get("reviews_read") in (None, 0):
            problems.append("no docs review row was read in the apply window")
        if vetoes.get("held_claim_names_a_supporting_binding"):
            problems.append(
                "a claim held while a binding supports it: "
                f"{vetoes['held_claim_names_a_supporting_binding']}"
            )
        for record in vetoes.get("reviews") or []:
            for claim in record["vetoed_claims"]:
                if not claim.get("bindings"):
                    problems.append(
                        f"vetoed claim on {record['name_id']} carries no "
                        "per-binding answers"
                    )
    if not payload.get("dry_runs"):
        problems.append("no dry run recorded")
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
        if fell:
            problems.append(f"members whose score fell: {fell}")
    if not HTML.exists():
        problems.append("the HTML fragment is absent")
    else:
        rendered = HTML.read_text()
        for anchor in ("summary", "convergence", "veto", "outside", "apply"):
            if f'id="{NODE}-{anchor}"' not in rendered:
                problems.append(f"the fragment states no {anchor} section")
    if problems:
        for problem in problems:
            print(f"FAIL {problem}")
        return 1
    members = before.get("members") or []
    scores_a = after.get("documentation_scores") or {}
    below = [
        m
        for m in members
        if scores_a.get(m) is not None and scores_a[m] < DEFAULT_MIN_SCORE
    ]
    print(
        "stamped: "
        + str(apply_entry.get("stamped"))
        + " | vetoes: "
        + str(vetoes.get("veto_count"))
        + " | below threshold: "
        + str(len(below))
        + " | unscored: "
        + str(sum(1 for m in members if scores_a.get(m) is None))
        + " | spend: "
        + str(cost.get("spend_usd"))
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
    dry = subcommands.add_parser("record-dry-run")
    dry.add_argument("log")
    dry.add_argument("label")
    rec = subcommands.add_parser("record-apply")
    rec.add_argument("log")
    rec.add_argument("wall_seconds", type=float)
    rec.add_argument(
        "--cost-limit", dest="cost_limit", type=float, default=COST_CAP_USD
    )
    rec.add_argument("--dry-run-log", dest="dry_run_log", default=None)
    sel = subcommands.add_parser("select-runs")
    sel.add_argument("log")
    subcommands.add_parser("record-cost")
    vet = subcommands.add_parser("record-vetoes")
    vet.add_argument("log")
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
    elif args.command == "record-dry-run":
        record_dry_run(args.log, args.label)
    elif args.command == "record-apply":
        record_apply(args.log, args.wall_seconds, args.cost_limit, args.dry_run_log)
    elif args.command == "select-runs":
        select_runs(args.log)
    elif args.command == "record-cost":
        record_cost()
    elif args.command == "record-vetoes":
        record_vetoes(args.log)
    elif args.command == "render":
        render()
    elif args.command == "verify":
        sys.exit(verify())
    else:
        fixture()


if __name__ == "__main__":
    main()
