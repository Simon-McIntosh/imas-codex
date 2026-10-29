"""Reground the wall member's documentation on its restored DD source and stamp the locus family.

The locus family's source move landed, so `electrostatic_potential_at_wall` owns
`wall/description_ggd/ggd/phi_potential/values` as its only producing source. Its
documentation still predates that source, so this node runs one capped
`sn harmonize --apply` over the family and records what it measured: the dry
runs' planned calls, the apply's calls, spend and wall time, every member's
documentation score before and after, every contradiction veto with each
binding's own answer, the stamped status, and the out-of-family check.

The one thing the apply cannot do by itself is regenerate the wall member's
documentation, because the documentation pools claim a name only at
`name_stage='accepted'` while this member rests at `'reviewed'`. Both surfaces
the brief names for making that member eligible — `sn edit --docs` and the
scoped reset the harmonize apply uses (`mark_members_for_regen`) — refuse on
that same precondition, and both refusals are reproduced and recorded here
rather than worked around. The family therefore cannot stamp, and the record
says which name holds it and why.

``run`` drives the whole measure and writes the fragment and its JSON. Logs live
beside the run's own output directory, never in the working tree.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

NODE = "shc-locus-final-apply"
HERE = Path(__file__).resolve().parent
RESULTS = HERE / f"{NODE}.json"
HTML = HERE / f"{NODE}.html"
PRIOR_RUNNER = HERE / "shc-locus-reapply-all-bindings-run.py"
SEED = "electrostatic_potential"
WALL = "electrostatic_potential_at_wall"
COST_CAP_USD = 2.00
#: The regrounding leg's own caps. Each command is given its own ceiling and
#: their sum is the authorised combined ceiling, so neither can borrow headroom
#: from the other at run time.
RESCORE_CAP_USD = 0.45
FAMILY_APPLY_CAP_USD = 0.30
DRAIN_CAP_USD = 0.50
COMBINED_CAP_USD = 1.00
#: Drains whose wrapper process was stopped while the CLI it started had
#: already finished in the graph. Their cost rows fall outside every window this
#: record selects on, so they are counted by run id instead - declared here, so
#: recomputing the score block reproduces the total rather than dropping what
#: the windows cannot see.
INTERRUPTED_DRAINS = ["8445f79a-679e-489a-aa04-dc8acf4cf0f9"]
ACCEPTANCE = 0.85
#: A member whose documentation was NOT regenerated may still move by rescoring
#: noise; at or below this the move is recorded as noise, never as a fall.
NOISE_BAND = 0.02
FAMILY_CMD = [
    "sn",
    "harmonize",
    "--family",
    SEED,
    "--apply",
]
RUN_DIR = Path(
    os.environ.get(
        "SHC_RUN_DIR",
        "/home/ITER/mcintos/.config/reckon/crew/runs/"
        "r-20261028T215047215974-shc-locus-final-apply",
    )
)
LOGS = RUN_DIR / "logs"
CONSOLE = "/home/ITER/mcintos/Code/imas-codex/.venv/bin/imas-codex"


def _prior():
    """The re-apply runner, whose graph readers this node reuses unchanged."""
    spec = importlib.util.spec_from_file_location("shc_locus_reapply", PRIOR_RUNNER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    module.NODE = NODE
    module.RESULTS = RESULTS
    module.SEED = SEED
    return module


def read_results() -> dict:
    if RESULTS.exists():
        return json.loads(RESULTS.read_text())
    return {"seed": SEED, "node": NODE}


def write_results(payload: dict) -> None:
    RESULTS.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")


def _env() -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(HERE.parents[3])
    env["UV_PROJECT_ENVIRONMENT"] = "/home/ITER/mcintos/Code/imas-codex/.venv"
    env.pop("VIRTUAL_ENV", None)
    return env


def _cli(args: list[str], log_name: str) -> tuple[Path, str, float]:
    """Run one CLI command in the foreground, capturing it to its own log."""
    LOGS.mkdir(parents=True, exist_ok=True)
    path = LOGS / log_name
    started = datetime.now(UTC).isoformat()
    mark = time.monotonic()
    proc = subprocess.run(
        [CONSOLE, *args],
        capture_output=True,
        text=True,
        env=_env(),
        cwd=str(HERE.parents[3]),
    )
    wall = time.monotonic() - mark
    ended = datetime.now(UTC).isoformat()
    body = [
        f"# command: {CONSOLE} {' '.join(args)}",
        f"# tree: {HERE.parents[3]}",
        f"# started: {started}",
        f"# ended: {ended}",
        f"# wall_seconds: {wall:.3f}",
        "--- stdout ---",
        proc.stdout.rstrip("\n"),
        "--- stderr ---",
        proc.stderr.rstrip("\n"),
        f"EXIT={proc.returncode}",
    ]
    path.write_text("\n".join(body) + "\n")
    return path, proc.stdout + proc.stderr, wall


def _member_state(graph, members: list[str]) -> dict[str, dict]:
    rows = graph.query(
        """
        MATCH (sn:StandardName)
        WHERE sn.id IN $ids
        RETURN sn.id AS id, sn.name_stage AS name_stage,
               sn.docs_stage AS docs_stage,
               sn.reviewer_score_docs AS reviewer_score_docs,
               sn.reviewer_score_name AS reviewer_score_name,
               sn.validation_status AS validation_status,
               sn.source_paths AS source_paths,
               sn.harmonized_group_signature AS signature
        ORDER BY id
        """,
        ids=members,
    )
    return {row["id"]: dict(row) for row in rows or []}


def preflight() -> dict:
    """Read the family, and reproduce the two named eligibility surfaces."""
    from imas_codex.graph.client import GraphClient
    from imas_codex.standard_names.harmonize import (
        mark_members_for_regen,
        plan_family_applies,
    )

    with GraphClient() as graph:
        plan = plan_family_applies(graph, seeds=[SEED])[0]
        states = _member_state(graph, plan["members"])
    if set(states) != set(plan["members"]):
        raise RuntimeError("the member read missed a family member")

    reset_preview = mark_members_for_regen([WALL], dry_run=True)
    edit_log, edit_text, _ = _cli(
        [
            "sn",
            "edit",
            WALL,
            "--docs",
            "placeholder",
            "--reason",
            "reproduce the docs-mode precondition for a name whose documentation "
            "predates its producing source",
            "--dry-run",
        ],
        "repro-sn-edit-docs.log",
    )
    blocked = [
        line.strip()
        for line in edit_text.splitlines()
        if "BLOCKED" in line or "name_stage" in line
    ]

    payload = read_results()
    payload["preflight"] = {
        "read_at_utc": datetime.now(UTC).isoformat(),
        "decision": plan["decision"],
        "anchor": plan["anchor"],
        "members": plan["members"],
        "docs_eligible": plan["docs_eligible"],
        "skipped": plan["skipped"],
        "planned_calls": plan["planned_calls"],
        "batches": plan["batches"],
        "member_scores": plan["member_scores"],
        "member_state": states,
        "scoped_reset_preview": reset_preview,
        "scoped_reset_surface": "imas_codex.standard_names.harmonize.mark_members_for_regen",
        "docs_edit_refusal": {
            "log": str(edit_log),
            "lines": blocked,
            "exit": int(
                edit_text and edit_log.read_text().rsplit("EXIT=", 1)[-1] or -1
            ),
        },
        "acceptance_threshold": ACCEPTANCE,
        "wall_name_stage": states[WALL]["name_stage"],
    }
    write_results(payload)
    print(json.dumps(payload["preflight"], sort_keys=True))
    return payload["preflight"]


def dry_run(label: str) -> dict:
    """Capture one dry run's planned calls and estimate."""
    log, text, _ = _cli(
        [*FAMILY_CMD, "--dry-run", "-c", f"{COST_CAP_USD:.2f}"],
        f"dry-run-{label}.log",
    )
    planned = re.search(r"(\d+)\s+planned\s+call", text)
    estimate = re.search(r"estimated cost \$([0-9]+(?:\.[0-9]+)?)", text)
    exit_line = log.read_text().splitlines()[-1]
    payload = read_results()
    entry = {
        "label": label,
        "log": str(log),
        "planned_calls": int(planned.group(1)) if planned else None,
        "estimated_usd": float(estimate.group(1)) if estimate else None,
        "exit_line": exit_line,
        "output": [ln for ln in text.splitlines() if ln.strip()],
    }
    kept = [d for d in payload.get("dry_runs") or [] if d.get("label") != label]
    payload["dry_runs"] = [*kept, entry]
    write_results(payload)
    print(json.dumps(entry, sort_keys=True))
    return entry


def apply() -> dict:
    """Run the capped family apply in the foreground and ingest its log."""
    prior = _prior()
    log, text, wall = _cli(
        [*FAMILY_CMD, "-c", f"{COST_CAP_USD:.2f}"],
        "apply.log",
    )
    payload = read_results()
    dry_log = (payload.get("dry_runs") or [{}])[0].get("log")
    prior.record_apply(str(log), wall, COST_CAP_USD, dry_log)
    payload = read_results()
    payload["applies"][-1]["wall_seconds"] = round(wall, 3)
    payload["applies"][-1]["output"] = [ln for ln in text.splitlines() if ln.strip()]
    write_results(payload)
    print(json.dumps(payload["applies"][-1], sort_keys=True))
    return payload["applies"][-1]


def _exit_line(log: Path) -> str:
    return log.read_text().splitlines()[-1]


def _name_state(graph, name: str) -> dict:
    """One name's two axes, read whole so no verdict is inferred from a count."""
    rows = graph.query(
        """
        MATCH (sn:StandardName {id: $name})
        RETURN sn.id AS id, sn.name_stage AS name_stage,
               sn.docs_stage AS docs_stage,
               sn.reviewer_score_name AS score_name,
               sn.reviewer_score_docs AS score_docs,
               sn.validation_status AS validation_status,
               sn.reviewer_comments_per_dim_name AS checks_name,
               sn.reviewer_comments_docs AS comments_docs,
               sn.reviewer_comments_per_dim_docs AS comments_per_dim_docs,
               sn.source_paths AS source_paths
        """,
        name=name,
    )
    return dict(rows[0]) if rows else {}


def rescore() -> dict:
    """Re-run the name quorum on the wall member against its restored source.

    The verdict this replaces was written when the name carried no producing
    source, so the failing check's own evidence was ``source_paths=[]``. The
    rescore runs the ordinary review pipeline over the bound source; nothing is
    edited on the name axis, so a second failure leaves the stored verdict and
    its checks untouched and is recorded verbatim.
    """
    from imas_codex.graph.client import GraphClient

    with GraphClient() as graph:
        before = _name_state(graph, WALL)
    log, text, wall = _cli(
        ["sn", "rescore", WALL, "-c", f"{RESCORE_CAP_USD:.2f}"],
        "rescore.log",
    )
    with GraphClient() as graph:
        after = _name_state(graph, WALL)
    accepted = after.get("name_stage") == "accepted"
    score_line = next(
        (ln.strip() for ln in text.splitlines() if "score" in ln.lower()),
        None,
    )
    payload = read_results()
    payload["rescore"] = {
        "command": f"sn rescore {WALL} -c {RESCORE_CAP_USD:.2f}",
        "log": str(log),
        "exit_line": _exit_line(log),
        "wall_seconds": round(wall, 3),
        "cost_limit_usd": RESCORE_CAP_USD,
        "before": before,
        "after": after,
        "accepted": accepted,
        "checks_name": after.get("checks_name"),
        "score_line": score_line,
        "output": [ln for ln in text.splitlines() if ln.strip()][-40:],
    }
    write_results(payload)
    print(
        json.dumps(
            {
                "accepted": accepted,
                "score_name_before": before.get("score_name"),
                "score_name_after": after.get("score_name"),
                "name_stage_after": after.get("name_stage"),
                "exit_line": _exit_line(log),
            },
            sort_keys=True,
        )
    )
    return payload["rescore"]


def family_apply(label: str, cap: float, log_name: str) -> dict:
    """One capped family apply, recorded like the first one."""
    prior = _prior()
    dry = next(
        (d for d in read_results().get("dry_runs") or [] if d.get("label") == label),
        None,
    )
    log, text, wall = _cli([*FAMILY_CMD, "-c", f"{cap:.2f}"], log_name)
    prior.record_apply(str(log), wall, cap, (dry or {}).get("log"))
    payload = read_results()
    entry = payload["applies"][-1]
    entry["label"] = label
    entry["wall_seconds"] = round(wall, 3)
    entry["output"] = [ln for ln in text.splitlines() if ln.strip()]
    write_results(payload)
    print(
        json.dumps(
            {
                "label": label,
                "decision": entry.get("decision"),
                "applied": entry.get("applied"),
                "stamped": entry.get("stamped"),
                "exit_line": entry.get("exit_line"),
            },
            sort_keys=True,
        )
    )
    return entry


def wall_drain() -> dict:
    """Regenerate the wall member's docs through the scoped reset and drain.

    The family is stamped and no member's text has changed since, so a family
    apply plans no batch and cannot claim this member. The surface that
    regenerates one member is the one the apply itself uses: the reset that
    snapshots the current docs and sets ``docs_stage='pending'`` under a fresh
    scope id, then a drain restricted to that id.
    """
    from imas_codex.standard_names.graph_store.graph_cost_and_claims import (
        aggregate_calls_per_phase,
        aggregate_spend_per_phase,
    )
    from imas_codex.standard_names.harmonize import mark_members_for_regen

    scope = mark_members_for_regen([WALL])
    payload = read_results()
    entry: dict = {
        "scoped_reset": scope,
        "docs_regenerated": False,
    }
    if scope.get("reset"):
        run_id = scope["run_id"]
        log, text, wall = _cli(
            [
                "sn",
                "run",
                "--scope-run-id",
                run_id,
                "--docs-only",
                "--flush",
                "-c",
                f"{DRAIN_CAP_USD:.2f}",
                "--time",
                "20",
            ],
            "wall-drain.log",
        )
        started, ended = _prior()._window(str(log))
        from imas_codex.graph.client import GraphClient

        with GraphClient() as graph:
            rows = graph.query(
                """
                MATCH (c:LLMCost)
                WHERE c.llm_at >= datetime($started) AND c.llm_at <= datetime($ended)
                  AND any(g IN coalesce(c.standard_name_ids, []) WHERE $name IN g)
                RETURN c.run_id AS run_id, count(c) AS calls
                ORDER BY run_id
                """,
                started=started,
                ended=ended,
                name=WALL,
            )
            run_ids = [row["run_id"] for row in rows or []]
            state = _name_state(graph, WALL)
            services = type("Svc", (), {"GraphClient": GraphClient})
            calls = 0
            spend = 0.0
            for rid in run_ids:
                per_phase = aggregate_calls_per_phase(rid, services=services)
                calls += sum(int(value) for value in per_phase.values())
                spend += sum(
                    float(value)
                    for value in aggregate_spend_per_phase(
                        rid, services=services
                    ).values()
                )
        entry.update(
            {
                "command": (
                    f"sn run --scope-run-id {run_id} --docs-only --flush "
                    f"-c {DRAIN_CAP_USD:.2f} --time 20"
                ),
                "log": str(log),
                "exit_line": _exit_line(log),
                "wall_seconds": round(wall, 3),
                "cost_limit_usd": DRAIN_CAP_USD,
                "run_ids": run_ids,
                "run_ids_source": "LLMCost rows in the drain's own window naming this member",
                "selection_window_utc": {"started": started, "ended": ended},
                "calls": calls,
                "spend_usd": round(spend, 6),
                "state_after": state,
                "docs_regenerated": state.get("docs_stage") == "accepted",
                "output": [ln for ln in text.splitlines() if ln.strip()][-40:],
            }
        )
    else:
        entry["skipped"] = "the scoped reset admitted no member"
    payload["wall_drain"] = entry
    write_results(payload)
    print(
        json.dumps(
            {
                "reset": scope.get("reset"),
                "run_ids": entry.get("run_ids"),
                "calls": entry.get("calls"),
                "spend_usd": entry.get("spend_usd"),
                "docs_regenerated": entry.get("docs_regenerated"),
                "exit_line": entry.get("exit_line"),
            },
            sort_keys=True,
        )
    )
    return entry


def finalize_family() -> dict:
    """Re-stamp the family on its new text, with no model calls."""
    log, text, wall = _cli(
        ["sn", "harmonize", "--family", SEED, "--finalize"],
        "finalize.log",
    )
    payload = read_results()
    entry = {
        "command": f"sn harmonize --family {SEED} --finalize",
        "log": str(log),
        "exit_line": _exit_line(log),
        "wall_seconds": round(wall, 3),
        "output": [ln for ln in text.splitlines() if ln.strip()][-20:],
    }
    payload["finalize"] = entry
    write_results(payload)
    print(json.dumps(entry, sort_keys=True))
    return entry


def rescore_cost() -> dict:
    """Read the rescore's own cost rows, selected by its window and the name."""
    from imas_codex.graph.client import GraphClient
    from imas_codex.standard_names.graph_store.graph_cost_and_claims import (
        aggregate_calls_per_phase,
        aggregate_spend_per_phase,
    )

    payload = read_results()
    block = payload.get("rescore") or {}
    if not block.get("log"):
        raise RuntimeError("record the rescore before its cost")
    started, ended = _prior()._window(block["log"])
    with GraphClient() as graph:
        rows = graph.query(
            """
            MATCH (c:LLMCost)
            WHERE c.llm_at >= datetime($started) AND c.llm_at <= datetime($ended)
              AND any(g IN coalesce(c.standard_name_ids, []) WHERE $name IN g)
            RETURN c.run_id AS run_id, count(c) AS calls
            ORDER BY run_id
            """,
            started=started,
            ended=ended,
            name=WALL,
        )
        run_ids = [row["run_id"] for row in rows or []]
        services = type("Svc", (), {"GraphClient": GraphClient})
        calls = 0
        spend = 0.0
        phases: dict[str, int] = {}
        for rid in run_ids:
            per_phase = aggregate_calls_per_phase(rid, services=services)
            for phase, value in per_phase.items():
                phases[phase] = phases.get(phase, 0) + int(value)
                calls += int(value)
            spend += sum(
                float(value)
                for value in aggregate_spend_per_phase(rid, services=services).values()
            )
    block["cost"] = {
        "run_ids": run_ids,
        "run_ids_source": "LLMCost rows in the rescore's own window naming this member",
        "selection_window_utc": {"started": started, "ended": ended},
        "phases": phases,
        "calls": calls,
        "spend_usd": round(spend, 6),
    }
    payload["rescore"] = block
    write_results(payload)
    print(json.dumps(block["cost"], sort_keys=True))
    return block["cost"]


def _run_cost(run_id: str) -> tuple[int, float]:
    """One run's calls and spend, read from its own cost rows."""
    from imas_codex.graph.client import GraphClient
    from imas_codex.standard_names.graph_store.graph_cost_and_claims import (
        aggregate_calls_per_phase,
        aggregate_spend_per_phase,
    )

    services = type("Svc", (), {"GraphClient": GraphClient})
    calls = sum(
        int(value)
        for value in aggregate_calls_per_phase(run_id, services=services).values()
    )
    spend = sum(
        float(value)
        for value in aggregate_spend_per_phase(run_id, services=services).values()
    )
    return calls, spend


def reground_scores() -> dict:
    """The wall member's documentation score across the regrounding, and the family's."""
    payload = read_results()
    before_phase = (payload.get("phases") or {}).get("before-reground") or {}
    after_phase = (payload.get("phases") or {}).get("after-finalize") or {}
    scores_before = before_phase.get("documentation_scores") or {}
    scores_after = after_phase.get("documentation_scores") or {}
    members = payload.get("preflight", {}).get("members") or []
    rows = []
    for member in members:
        b = scores_before.get(member)
        a = scores_after.get(member)
        rows.append(
            {
                "id": member,
                "before": b,
                "after": a,
                "delta": None if (b is None or a is None) else round(a - b, 6),
            }
        )
    wall = next((row for row in rows if row["id"] == WALL), {})
    drain = payload.get("wall_drain") or {}
    rescore_block = payload.get("rescore") or {}
    reground_apply = next(
        (a for a in payload.get("applies") or [] if a.get("label") == "reground"),
        {},
    )
    rescore_cost_block = rescore_block.get("cost") or {}
    interrupted = []
    for run_id in INTERRUPTED_DRAINS:
        calls, usd = _run_cost(run_id)
        interrupted.append(
            {
                "run_id": run_id,
                "calls": calls,
                "spend_usd": round(usd, 6),
                "note": (
                    "a drain attempt whose wrapper was stopped while its CLI had "
                    "already completed in the graph; counted by run id because its "
                    "rows sit outside the recorded drain window"
                ),
            }
        )
    spend = (
        float((reground_apply.get("cost") or {}).get("spend_usd") or 0.0)
        + float(drain.get("spend_usd") or 0.0)
        + float(rescore_cost_block.get("spend_usd") or 0.0)
        + sum(float(item["spend_usd"]) for item in interrupted)
    )
    block = {
        "threshold": ACCEPTANCE,
        "members": rows,
        "wall": wall,
        "wall_regenerated": bool(drain.get("docs_regenerated")),
        "wall_docs_before": wall.get("before"),
        "wall_docs_after": wall.get("after"),
        "wall_docs_delta": wall.get("delta"),
        "name_before": {
            "name_stage": (rescore_block.get("before") or {}).get("name_stage"),
            "score_name": (rescore_block.get("before") or {}).get("score_name"),
            "checks_name": (rescore_block.get("before") or {}).get("checks_name"),
        },
        "name_after": {
            "name_stage": (rescore_block.get("after") or {}).get("name_stage"),
            "score_name": (rescore_block.get("after") or {}).get("score_name"),
            "checks_name": (rescore_block.get("after") or {}).get("checks_name"),
        },
        "rescore_accepted": bool(rescore_block.get("accepted")),
        "calls": int((reground_apply.get("cost") or {}).get("calls_total") or 0)
        + int(drain.get("calls") or 0)
        + int(rescore_cost_block.get("calls") or 0)
        + sum(int(item["calls"]) for item in interrupted),
        "interrupted_drain": interrupted,
        "rescore_calls": int(rescore_cost_block.get("calls") or 0),
        "rescore_spend_usd": float(rescore_cost_block.get("spend_usd") or 0.0),
        "drain_calls": int(drain.get("calls") or 0),
        "drain_spend_usd": float(drain.get("spend_usd") or 0.0),
        "spend_usd": round(spend, 6),
        "combined_cap_usd": COMBINED_CAP_USD,
        "within_combined_cap": spend <= COMBINED_CAP_USD,
        "all_at_or_above": all(
            row["after"] is not None and row["after"] >= ACCEPTANCE for row in rows
        ),
    }
    payload["reground"] = block
    write_results(payload)
    print(json.dumps(block, sort_keys=True))
    return block


def _trim_apply_output(payload: dict) -> None:
    """Keep the apply's summary lines; the full transcript stays in its log.

    The apply prints a progress line per pool claim, so carrying the whole
    stream in the record would put a transcript in the repository. The log is
    named on the same entry, so nothing is lost.
    """
    for entry in payload.get("applies") or []:
        output = entry.get("output") or []
        kept = [
            line
            for line in output
            if line.startswith(("family ", "apply:", "docs scores:", "    "))
        ]
        entry["output"] = kept[:60]
        entry["output_note"] = f"{len(output)} line(s) captured; full text in the log"


def post() -> dict:
    """Snapshot the members after the apply and re-plan the family."""
    prior = _prior()
    prior.snapshot("after")
    dry_run("after-apply")
    payload = read_results()
    _trim_apply_output(payload)
    write_results(payload)
    before = payload.get("phases", {}).get("before", {})
    after = payload.get("phases", {}).get("after", {})
    scores_before = before.get("documentation_scores") or {}
    scores_after = after.get("documentation_scores") or {}
    rows = []
    for member in payload.get("preflight", {}).get("members") or []:
        b = scores_before.get(member)
        a = scores_after.get(member)
        # A member is regenerated when it entered one of the apply's batches:
        # that is the set the runner snapshots and re-drains, so it is a fact
        # about the plan rather than an inference from two equal stage reads.
        regenerated = member in (
            payload.get("preflight", {}).get("docs_eligible") or []
        )
        delta = None if (b is None or a is None) else round(a - b, 6)
        rows.append(
            {
                "id": member,
                "before": b,
                "after": a,
                "delta": delta,
                "regenerated": regenerated,
                "noise_only": bool(
                    delta is not None
                    and not regenerated
                    and -NOISE_BAND <= (delta or 0.0) <= NOISE_BAND
                ),
                "below_acceptance": bool(a is not None and a < ACCEPTANCE),
            }
        )
    payload["scores"] = {
        "threshold": ACCEPTANCE,
        "noise_band": NOISE_BAND,
        "members": rows,
        "scored_at_or_above": all(
            row["after"] is not None and row["after"] >= ACCEPTANCE for row in rows
        ),
        "unscored": [row["id"] for row in rows if row["after"] is None],
        "fell": [
            row["id"]
            for row in rows
            if row["delta"] is not None and row["delta"] < 0 and not row["noise_only"]
        ],
    }
    write_results(payload)
    print(json.dumps(payload["scores"], sort_keys=True))
    return payload["scores"]


def cost() -> dict:
    """Select the first apply's run ids and read its calls, spend and scope.

    The cost reader reads the record's last apply, so it runs while the first
    apply is still the last one. Once a later apply is appended the first apply
    already carries its cost block and there is nothing left to select.
    """
    prior = _prior()
    payload = read_results()
    first = next(
        (a for a in payload.get("applies") or [] if a.get("label") in (None, "first")),
        None,
    )
    if first is None or first.get("cost"):
        print("cost already recorded for the first apply; not re-selected")
        return first or {}
    log = first["log"]
    prior.select_runs(log)
    prior.record_cost()
    prior.record_vetoes(log)
    payload = read_results()
    entry = payload["applies"][-1]
    print(
        json.dumps(
            {
                "cost": entry.get("cost"),
                "scope_check": entry.get("scope_check"),
                "vetoes": payload.get("vetoes"),
            },
            sort_keys=True,
        )
    )
    return entry


def _refusal(entry: dict) -> str:
    """The refusal verbatim, with the CLI's box drawing flattened."""
    joined = " / ".join(entry.get("lines") or [])
    return " ".join(joined.replace("│", " ").split())


def _cell(value: object) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def render() -> None:
    payload = read_results()
    pre = payload.get("preflight", {})
    applies = payload.get("applies") or [{}]
    # The verdict belongs to the apply that stamped the family. A later apply
    # that planned no batch stamps nothing, so taking it as the family's state
    # reports a stamp it never made and a scope it never measured.
    entry = next((a for a in reversed(applies) if a.get("stamped")), applies[-1])
    cost_block = entry.get("cost") or {}
    scope = entry.get("scope_check") or {}
    scores = payload.get("scores") or {}
    vetoes = payload.get("vetoes") or {}
    dry = payload.get("dry_runs") or []
    first = next((d for d in dry if d["label"] == "before-apply"), {})
    second = next((d for d in dry if d["label"] == "after-apply"), {})

    regrounded = payload.get("reground") or {}
    converged = bool(
        scores.get("scored_at_or_above")
        and entry.get("stamped")
        and not scores.get("unscored")
        and (second.get("planned_calls") == 0)
        and (not regrounded or regrounded.get("all_at_or_above"))
    )
    if converged and regrounded.get("wall_regenerated"):
        verdict = "converged and stamped, wall member regrounded"
    elif converged:
        verdict = "converged and stamped"
    else:
        verdict = "not converged"

    def uid(slug: str) -> str:
        return f'id="{NODE}-{slug}"'

    lines = [
        "<!doctype html>",
        '<html lang="en">',
        "<head>",
        '<meta charset="utf-8">',
        '<meta name="reckon-type" content="evidence-fragment">',
        '<meta name="plan-slug" content="sn-harmonize-command">',
        f'<meta name="node-id" content="{NODE}">',
        "<title>One capped apply regrounds the wall member and stamps the locus family</title>",
        "</head>",
        "<body>",
        "<main>",
        f"<h2 {uid('verdict')}>Verdict: {verdict}</h2>",
        f"<p {uid('verdict-p')}>Family <code>{SEED}</code> ran one "
        f"<code>sn harmonize --family {SEED} --apply -c {COST_CAP_USD:.2f}</code> in "
        f"the foreground. The apply spent <strong>USD "
        f"{_cell(cost_block.get('spend_usd'))}</strong> across "
        f"<strong>{_cell(cost_block.get('calls_total'))}</strong> call(s) in "
        f"<strong>{_cell(entry.get('wall_seconds'))} s</strong>, and stamped "
        f"<strong>{_cell(entry.get('stamped'))}</strong>. Convergence is defined as "
        f"every member scored at or above {ACCEPTANCE} and the family stamped: "
        f"that is <strong>{verdict}</strong>.</p>",
        (
            f"<p {uid('verdict-regrounded')}>The wall member's documentation was "
            f"then regrounded on its restored source in a second, separately "
            f"capped leg: the name quorum re-ran and accepted it "
            f"({_cell((regrounded.get('name_before') or {}).get('score_name'))} → "
            f"{_cell((regrounded.get('name_after') or {}).get('score_name'))}), its "
            f"documentation score moved "
            f"{_cell(regrounded.get('wall_docs_before'))} → "
            f"{_cell(regrounded.get('wall_docs_after'))}, and the family was "
            f"re-stamped on the new text for "
            f"{_cell(regrounded.get('calls'))} call(s) and USD "
            f"{_cell(regrounded.get('spend_usd'))} of the USD "
            f"{COMBINED_CAP_USD:.2f} ceiling. That leg is recorded in full under "
            f"<em>The authorised regrounding</em> below.</p>"
            if regrounded
            else ""
        ),
        (
            f"<p {uid('verdict-note')}>At the time this apply ran the wall member "
            f"sat at <code>name_stage={_cell(pre.get('wall_name_stage'))}</code>, so "
            f"the documentation pools could not claim it and the apply covered "
            f"{len(pre.get('docs_eligible') or [])} of the family's "
            f"{len(pre.get('members') or [])} members. Both surfaces the brief "
            f"named for making it eligible were reproduced and refused; that "
            f"blockade is what the regrounding below resolves.</p>"
            if regrounded
            else f"<p {uid('verdict-blocker')}>The name that holds the family is "
            f"<code>{WALL}</code>, at <code>name_stage="
            f"{_cell(pre.get('wall_name_stage'))}</code>. The documentation pools "
            f"claim a name only at <code>name_stage='accepted'</code>, so neither "
            f"the apply nor either surface the brief names for making it eligible "
            f"can touch it. Both refusals are reproduced below.</p>"
        ),
    ]

    # --- dry runs
    lines += [
        f"<h2 {uid('dry')}>The two dry runs</h2>",
        f"<table {uid('dry-table')}>",
        "<caption>Planned calls and estimate, captured from each dry run's own log</caption>",
        "<thead><tr><th>dry run</th><th>planned calls</th><th>estimated USD</th>"
        "<th>exit</th></tr></thead>",
        "<tbody>",
    ]
    for label, item in (("before the apply", first), ("after the apply", second)):
        lines.append(
            f"<tr><td>{label}</td><td>{_cell(item.get('planned_calls'))}</td>"
            f"<td>{_cell(item.get('estimated_usd'))}</td>"
            f"<td><code>{_cell(item.get('exit_line'))}</code></td></tr>"
        )
    lines += ["</tbody>", "</table>"]
    lines.append(
        f"<p {uid('dry-p')}>The pre-apply dry run planned "
        f"{_cell(first.get('planned_calls'))} call(s) at an estimated USD "
        f"{_cell(first.get('estimated_usd'))}, under the USD {COST_CAP_USD:.2f} cap. "
        f"The second dry run planned {_cell(second.get('planned_calls'))} call(s). A "
        f"stamped family plans zero, because every member then carries its current "
        f"group signature; the figure measured here is the control that the family "
        f"did not stamp.</p>"
    )

    # --- eligibility
    reset = pre.get("scoped_reset_preview") or {}
    edit = pre.get("docs_edit_refusal") or {}
    lines += [
        f"<h2 {uid('eligibility')}>Why the first apply could not reach the wall "
        f"member</h2>",
        f"<p {uid('eligibility-p')}>This is the state the first apply met. The "
        f"brief names two sanctioned surfaces for making the wall member's "
        f"documentation eligible for regeneration. Both require "
        f"<code>name_stage='accepted'</code>; at that point the member rested at "
        f"<code>{_cell(pre.get('wall_name_stage'))}</code>. Both were reproduced "
        f"against the live graph, and both wrote nothing. The regrounding below "
        f"records what changed that precondition and what followed.</p>",
        f"<table {uid('eligibility-table')}>",
        "<caption>The two named surfaces, each run as a preview, with its own refusal</caption>",
        "<thead><tr><th>surface</th><th>arguments</th><th>outcome</th></tr></thead>",
        "<tbody>",
        f"<tr><td><code>sn edit --docs</code></td><td><code>--dry-run</code></td>"
        f"<td>{_cell(_refusal(edit) or 'no refusal captured')}</td></tr>",
        f"<tr><td><code>mark_members_for_regen</code></td>"
        f"<td><code>dry_run=True</code></td>"
        f"<td>eligible={_cell(reset.get('eligible'))}, "
        f"reset={_cell(reset.get('reset'))} — the reset admits only "
        f"<code>name_stage='accepted'</code></td></tr>",
        "</tbody>",
        "</table>",
        f"<p {uid('eligibility-plan')}>The family plan therefore lists "
        f"{len(pre.get('docs_eligible') or [])} documentation-eligible member(s) and "
        f"skips <code>{WALL}</code> with its stage: "
        f"<code>{_cell(json.dumps(pre.get('skipped')))}</code>. The apply's batch "
        f"sequence never contains it: "
        f"<code>{_cell(json.dumps(pre.get('batches')))}</code>.</p>",
    ]

    # --- apply
    lines += [
        f"<h2 {uid('apply')}>The capped apply</h2>",
        f"<table {uid('apply-table')}>",
        "<caption>The apply's own figures, selected by the exact run ids that own its cost rows</caption>",
        "<thead><tr><th>measure</th><th>value</th></tr></thead>",
        "<tbody>",
    ]
    for label, value in (
        ("outcome", "converged" if not entry.get("halted") else "halted"),
        ("batches run", entry.get("batches_run")),
        (
            "members accepted / touched",
            f"{_cell(entry.get('members_accepted'))} / {_cell(entry.get('members_touched'))}",
        ),
        ("families applied", entry.get("families_applied")),
        ("stamped", entry.get("stamped")),
        ("calls", cost_block.get("calls_total")),
        ("spend USD", cost_block.get("spend_usd")),
        ("cap USD", entry.get("cost_limit_usd")),
        ("unspent USD", entry.get("cost_limit_unspent_usd")),
        ("wall seconds", entry.get("wall_seconds")),
        ("run ids selected", len(entry.get("run_ids") or [])),
        ("exit line", entry.get("exit_line")),
    ):
        lines.append(f"<tr><td>{label}</td><td>{_cell(value)}</td></tr>")
    lines += ["</tbody>", "</table>"]
    phases = cost_block.get("phases") or {}
    if phases:
        lines += [
            f"<table {uid('phases-table')}>",
            "<caption>Calls and spend by phase, from the LLMCost aggregation</caption>",
            "<thead><tr><th>phase</th><th>calls</th><th>USD</th></tr></thead>",
            "<tbody>",
        ]
        for phase, row in sorted(phases.items()):
            lines.append(
                f"<tr><td><code>{phase}</code></td><td>{row.get('calls')}</td>"
                f"<td>{_cell(float(row.get('usd') or 0.0))}</td></tr>"
            )
        lines += ["</tbody>", "</table>"]

    # --- scores
    lines += [
        f"<h2 {uid('scores')}>Documentation scores, before and after</h2>",
        f"<table {uid('scores-table')}>",
        "<caption>Every member's documentation-axis score read before the apply and "
        "after it; a move of 0.02 or less on a member whose documentation was not "
        "regenerated is rescoring noise</caption>",
        "<thead><tr><th>member</th><th>before</th><th>after</th><th>delta</th>"
        "<th>regenerated</th><th>reading</th></tr></thead>",
        "<tbody>",
    ]
    for row in scores.get("members") or []:
        if row.get("after") is None:
            reading = "unscored"
        elif row.get("below_acceptance"):
            reading = "below acceptance"
        elif row.get("fell"):
            reading = "fell"
        elif not row.get("regenerated") and row.get("delta") == 0:
            reading = "unchanged, not regenerated"
        elif row.get("noise_only"):
            reading = "noise"
        else:
            reading = "held"
        lines.append(
            f"<tr><td><code>{row['id']}</code></td><td>{_cell(row.get('before'))}</td>"
            f"<td>{_cell(row.get('after'))}</td><td>{_cell(row.get('delta'))}</td>"
            f"<td>{'yes' if row.get('regenerated') else 'no'}</td>"
            f"<td>{reading}</td></tr>"
        )
    lines += ["</tbody>", "</table>"]
    lines.append(
        f"<p {uid('scores-p')}>Scored at or above {ACCEPTANCE}: "
        f"<strong>{_cell(scores.get('scored_at_or_above'))}</strong>; unscored: "
        f"<code>{_cell(json.dumps(scores.get('unscored')))}</code>; fell beyond the "
        f"noise band: <code>{_cell(json.dumps(scores.get('fell')))}</code>.</p>"
    )

    # --- vetoes
    lines += [
        f"<h2 {uid('vetoes')}>Contradiction vetoes</h2>",
        f"<p {uid('vetoes-p')}>The apply raised "
        f"<strong>{_cell(vetoes.get('veto_count', 0))}</strong> "
        f"contradiction veto(es) over "
        f"<code>{_cell(json.dumps(vetoes.get('names_vetoed') or []))}</code>, reading "
        f"{_cell(vetoes.get('reviews_read', 0))} documentation review(s). Each "
        f"vetoed claim is recorded with the answer every one of the name's DD "
        f"bindings gave it; the claim-level fold is recomputed from those answers "
        f"rather than trusted from the log.</p>",
    ]
    claims = [
        {**claim, "name_id": review.get("name_id")}
        for review in vetoes.get("reviews") or []
        for claim in review.get("vetoed_claims") or []
    ]
    if claims:
        lines += [
            f"<table {uid('vetoes-table')}>",
            "<caption>Each vetoed claim, its fold and each binding's own answer</caption>",
            "<thead><tr><th>name</th><th>claim</th><th>fold</th>"
            "<th>per-binding answers</th></tr></thead>",
            "<tbody>",
        ]
        for claim in claims:
            answers = claim.get("bindings") or claim.get("answers") or []
            rendered = "; ".join(
                f"{a.get('dd_path') or a.get('binding')}: {a.get('answer')}"
                for a in answers
            )
            lines.append(
                f"<tr><td><code>{_cell(claim.get('name_id'))}</code></td>"
                f"<td><code>{_cell(claim.get('claim'))}</code></td>"
                f"<td>{_cell(claim.get('folded'))}</td><td>{rendered or '—'}</td></tr>"
            )
        lines += ["</tbody>", "</table>"]
    else:
        lines.append(
            f"<p {uid('vetoes-none')}>No contradiction veto was raised, so there is "
            f"no per-binding answer to show.</p>"
        )

    # --- out of family
    lines += [
        f"<h2 {uid('outside')}>Out-of-family check</h2>",
        f"<p {uid('outside-p')}>The cost rows the apply window owns claim "
        f"<strong>{len(scope.get('touched_names') or [])}</strong> name(s). Names "
        f"outside the {len(scope.get('roster') or [])}-member roster: "
        f"<strong>{_cell(scope.get('outside_count'))}</strong> "
        f"(<code>{_cell(json.dumps(scope.get('outside_roster') or []))}</code>). "
        f"Names the roster carries that the run did not touch: "
        f"<code>{_cell(json.dumps(scope.get('roster_names_not_touched') or []))}</code>.</p>",
    ]

    # --- the authorised regrounding
    reground = payload.get("reground") or {}
    if reground:
        rescore_block = payload.get("rescore") or {}
        drain = payload.get("wall_drain") or {}
        finalize_block = payload.get("finalize") or {}
        reground_apply = next(
            (a for a in payload.get("applies") or [] if a.get("label") == "reground"),
            {},
        )
        # The apply's own stdout names the decision on an apply that runs no
        # batch; the summary line only exists when a family was applied.
        reground_decision = reground_apply.get("decision") or next(
            (
                line.split()[2]
                for line in reground_apply.get("output") or []
                if line.startswith("family ") and len(line.split()) > 2
            ),
            None,
        )
        after_finalize = next((d for d in dry if d["label"] == "after-finalize"), {})
        name_before = reground.get("name_before") or {}
        name_after = reground.get("name_after") or {}
        lines += [
            f"<h2 {uid('reground')}>The authorised regrounding</h2>",
            f"<p {uid('reground-p')}>The wall member's stale name verdict rested on "
            f"a missing producer — its failing check's own evidence was "
            f"<code>source_paths=[]</code>. With the source restored, the name "
            f"quorum was re-run through <code>sn rescore</code>, then the member's "
            f"documentation was regenerated against "
            f"<code>wall/description_ggd/ggd/phi_potential/values</code> and the "
            f"family re-stamped. Combined spend was <strong>USD "
            f"{_cell(reground.get('spend_usd'))}</strong> across "
            f"<strong>{_cell(reground.get('calls'))}</strong> call(s), inside the "
            f"USD {COMBINED_CAP_USD:.2f} ceiling: "
            f"<strong>{_cell(reground.get('within_combined_cap'))}</strong>.</p>",
            f"<h3 {uid('name-axis')}>The name axis</h3>",
            f"<table {uid('name-table')}>",
            "<caption>The wall member's name-axis verdict before and after the rescore</caption>",
            "<thead><tr><th>measure</th><th>before</th><th>after</th></tr></thead>",
            "<tbody>",
            f"<tr><td><code>name_stage</code></td>"
            f"<td><code>{_cell(name_before.get('name_stage'))}</code></td>"
            f"<td><code>{_cell(name_after.get('name_stage'))}</code></td></tr>",
            f"<tr><td><code>reviewer_score_name</code></td>"
            f"<td>{_cell(name_before.get('score_name'))}</td>"
            f"<td>{_cell(name_after.get('score_name'))}</td></tr>",
            "</tbody>",
            "</table>",
            f"<pre {uid('checks-before')}>{_cell(name_before.get('checks_name'))}</pre>",
            f"<pre {uid('checks-after')}>{_cell(name_after.get('checks_name'))}</pre>",
            f"<p {uid('rescore-p')}>The rescore ran <code>"
            f"{_cell(rescore_block.get('command'))}</code> in the foreground and "
            f"exited <code>{_cell(rescore_block.get('exit_line'))}</code> after "
            f"{_cell(rescore_block.get('wall_seconds'))} s. Accepted: "
            f"<strong>{_cell(reground.get('rescore_accepted'))}</strong>. The name "
            f"text was not edited.</p>",
            f"<h3 {uid('docs-axis')}>The documentation axis</h3>",
            f"<p {uid('apply-p2')}>The family apply re-planned after the rescore and "
            f"returned <code>decision={_cell(reground_decision)}</code> "
            f"with <code>applied={_cell(reground_apply.get('applied'))}</code> and "
            f"<code>calls={_cell((reground_apply.get('cost') or {}).get('calls_total') or 0)}</code>"
            f": a stamped family plans no batch, and no member's text had changed, "
            f"so that surface cannot claim the member. The member was therefore "
            f"regenerated through the surface the apply itself uses — the scoped "
            f"reset that snapshots the old text and sets "
            f"<code>docs_stage='pending'</code> under a fresh scope id, then a drain "
            f"restricted to that id:</p>",
            f"<table {uid('drain-table')}>",
            "<caption>The scoped reset and drain, with the drain's own cost rows</caption>",
            "<thead><tr><th>measure</th><th>value</th></tr></thead>",
            "<tbody>",
            f"<tr><td>scoped reset (eligible / reset)</td><td>{_cell((drain.get('scoped_reset') or {}).get('eligible'))} / "
            f"{_cell((drain.get('scoped_reset') or {}).get('reset'))}</td></tr>",
            f"<tr><td>scope run id</td><td><code>{_cell((drain.get('scoped_reset') or {}).get('run_id'))}</code></td></tr>",
            f"<tr><td>command</td><td><code>{_cell(drain.get('command'))}</code></td></tr>",
            f"<tr><td>calls / spend</td><td>{_cell(drain.get('calls'))} / USD "
            f"{_cell(drain.get('spend_usd'))}</td></tr>",
            f"<tr><td>wall time</td><td>{_cell(drain.get('wall_seconds'))} s</td></tr>",
            f"<tr><td>exit</td><td><code>{_cell(drain.get('exit_line'))}</code></td></tr>",
            f"<tr><td>docs regenerated and accepted</td>"
            f"<td><strong>{_cell(drain.get('docs_regenerated'))}</strong></td></tr>",
            *[
                f"<tr><td>interrupted attempt <code>{_cell(item['run_id'][:8])}</code>"
                f"</td><td>{_cell(item['calls'])} call(s) / USD "
                f"{_cell(item['spend_usd'])} — counted by run id, outside the "
                f"recorded window</td></tr>"
                for item in reground.get("interrupted_drain") or []
            ],
            "</tbody>",
            "</table>",
            f"<table {uid('wall-score-table')}>",
            "<caption>The wall member's documentation score before and after regeneration</caption>",
            "<thead><tr><th>member</th><th>docs score before</th><th>docs score after</th>"
            "<th>delta</th><th>regenerated</th></tr></thead>",
            "<tbody>",
            f"<tr><td><code>{WALL}</code></td>"
            f"<td>{_cell(reground.get('wall_docs_before'))}</td>"
            f"<td>{_cell(reground.get('wall_docs_after'))}</td>"
            f"<td>{_cell(reground.get('wall_docs_delta'))}</td>"
            f"<td>{_cell(reground.get('wall_regenerated'))}</td></tr>",
            "</tbody>",
            "</table>",
            f"<p {uid('finalize-p')}>The family was then re-stamped on its new text "
            f"with <code>{_cell(finalize_block.get('command'))}</code> "
            f"(<code>{_cell(finalize_block.get('exit_line'))}</code>), which makes no "
            f"model calls, and the follow-up dry run planned "
            f"<strong>{_cell(after_finalize.get('planned_calls'))}</strong> call(s). "
            f"Every member is at or above {ACCEPTANCE} with the family stamped: "
            f"<strong>{_cell(reground.get('all_at_or_above'))}</strong>.</p>",
        ]

    lines += ["</main>", "</body>", "</html>"]
    HTML.write_text("\n".join(lines) + "\n")
    print(f"rendered {HTML}")


def verify() -> int:
    payload = read_results()
    problems: list[str] = []
    pre = payload.get("preflight") or {}
    applies = payload.get("applies") or [{}]
    entry = next(
        (a for a in reversed(applies) if a.get("stamped")),
        applies[-1],
    )
    cost = entry.get("cost") or {}
    scores = payload.get("scores") or {}
    dry = payload.get("dry_runs") or []
    first = next((d for d in dry if d["label"] == "before-apply"), {})
    second = next((d for d in dry if d["label"] == "after-apply"), {})

    if not first or first.get("planned_calls") is None:
        problems.append("the pre-apply dry run recorded no planned calls")
    if entry.get("exit_line") != "EXIT=0":
        problems.append(f"the apply did not exit 0: {entry.get('exit_line')}")
    if float(cost.get("spend_usd") or 0.0) > COST_CAP_USD:
        problems.append(
            f"spend {cost.get('spend_usd')} exceeds the USD {COST_CAP_USD:.2f} cap"
        )
    if (entry.get("scope_check") or {}).get("outside_count"):
        problems.append("the apply claimed names outside the family roster")
    if set(pre.get("members") or []) != set(
        (entry.get("scope_check") or {}).get("roster") or []
    ):
        problems.append("the scope roster differs from the planned family roster")
    if not scores.get("members"):
        problems.append("no per-member score was recorded")
    if second.get("planned_calls") is None:
        problems.append("the second dry run recorded no planned calls")
    stamped = bool(entry.get("stamped"))
    converged = bool(
        scores.get("scored_at_or_above")
        and stamped
        and not scores.get("unscored")
        and second.get("planned_calls") == 0
    )
    reg = payload.get("reground") or {}
    drain = payload.get("wall_drain") or {}
    finalize_block = payload.get("finalize") or {}
    after_finalize = next((d for d in dry if d["label"] == "after-finalize"), {})
    if reg:
        if not reg.get("rescore_accepted"):
            problems.append("the name quorum did not accept the wall member")
        if not reg.get("within_combined_cap"):
            problems.append(
                f"the regrounding spent USD {reg.get('spend_usd')}, over the "
                f"USD {COMBINED_CAP_USD:.2f} combined ceiling"
            )
        if not reg.get("wall_regenerated"):
            problems.append("the wall member's documentation was not regenerated")
        if drain.get("exit_line") != "EXIT=0":
            problems.append(f"the drain did not exit 0: {drain.get('exit_line')}")
        if finalize_block.get("exit_line") != "EXIT=0":
            problems.append(
                f"the finalize did not exit 0: {finalize_block.get('exit_line')}"
            )
        if after_finalize.get("planned_calls") != 0:
            problems.append(
                "the family plans calls after finalize: "
                f"{after_finalize.get('planned_calls')}"
            )
        if not reg.get("all_at_or_above"):
            problems.append("a member's documentation score is below acceptance")

    print(
        json.dumps(
            {
                "problems": problems,
                "stamped": stamped,
                "converged": converged,
                "spend_usd": cost.get("spend_usd"),
                "calls": cost.get("calls_total"),
                "wall_seconds": entry.get("wall_seconds"),
                "scores_after": {
                    row["id"]: row["after"] for row in scores.get("members") or []
                },
                "second_dry_run_planned_calls": second.get("planned_calls"),
            },
            sort_keys=True,
        )
    )
    if problems:
        print("FAILED")
        return 1
    print("OK")
    return 0


def run() -> None:
    LOGS.mkdir(parents=True, exist_ok=True)
    started = datetime.now(UTC).isoformat()
    print(f"node {NODE} starting at {started}")
    print(f"tree {HERE.parents[3]}")
    print(
        f"revision {subprocess.run(['git', '-C', str(HERE.parents[3]), 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()}"
    )
    # Each step is skipped when the record already holds its result, so a
    # rerun after a later step failed never re-spends the apply's budget and
    # never overwrites the pre-apply read with a post-apply one.
    payload = read_results()
    if "preflight" not in payload:
        preflight()
    if not any(d.get("label") == "before-apply" for d in payload.get("dry_runs") or []):
        dry_run("before-apply")
    if "before" not in (payload.get("phases") or {}):
        _prior().snapshot("before")
    payload = read_results()
    if not payload.get("applies"):
        apply()
    else:
        print("apply already recorded; not re-run")
    post()
    cost()

    # --- the authorised regrounding, each step bounded and resumable
    payload = read_results()
    if "rescore" not in payload:
        rescore()
    payload = read_results()
    if not (payload.get("rescore") or {}).get("accepted"):
        print("the name quorum did not accept the wall member; stopping as directed")
        render()
        sys.exit(verify())
    if "before-reground" not in (payload.get("phases") or {}):
        _prior().snapshot("before-reground")
    if not any(a.get("label") == "reground" for a in payload.get("applies") or []):
        family_apply("reground", FAMILY_APPLY_CAP_USD, "apply-reground.log")
    payload = read_results()
    reground_entry = next(
        (a for a in payload.get("applies") or [] if a.get("label") == "reground"), {}
    )
    if reground_entry.get("applied") and "cost" not in reground_entry:
        prior = _prior()
        try:
            prior.select_runs(reground_entry["log"])
            prior.record_cost()
            prior.record_vetoes(reground_entry["log"])
        except Exception as exc:  # noqa: BLE001 - recorded, never swallowed silently
            payload = read_results()
            payload["applies"][-1]["cost_note"] = f"cost not recorded: {exc}"
            write_results(payload)
    if "wall_drain" not in read_results():
        wall_drain()
    if "finalize" not in read_results():
        finalize_family()
    if "after-finalize" not in (read_results().get("phases") or {}):
        _prior().snapshot("after-finalize")
    if not any(
        d.get("label") == "after-finalize" for d in read_results().get("dry_runs") or []
    ):
        dry_run("after-finalize")
    reground_scores()
    render()
    sys.exit(verify())


def main() -> None:
    parser = argparse.ArgumentParser()
    subs = parser.add_subparsers(dest="command", required=True)
    subs.add_parser("preflight")
    dry = subs.add_parser("dry-run")
    dry.add_argument("label")
    sub_snap = subs.add_parser("snapshot")
    sub_snap.add_argument("phase")
    subs.add_parser("apply")
    subs.add_parser("rescore")
    subs.add_parser("wall-drain")
    subs.add_parser("finalize")
    subs.add_parser("reground-scores")
    subs.add_parser("rescore-cost")
    subs.add_parser("post")
    subs.add_parser("cost")
    subs.add_parser("render")
    subs.add_parser("verify")
    subs.add_parser("run")
    args = parser.parse_args()
    if args.command == "preflight":
        preflight()
    elif args.command == "dry-run":
        dry_run(args.label)
    elif args.command == "snapshot":
        _prior().snapshot(args.phase)
    elif args.command == "apply":
        apply()
    elif args.command == "rescore":
        rescore()
    elif args.command == "wall-drain":
        wall_drain()
    elif args.command == "finalize":
        finalize_family()
    elif args.command == "reground-scores":
        reground_scores()
    elif args.command == "rescore-cost":
        rescore_cost()
    elif args.command == "post":
        post()
    elif args.command == "cost":
        cost()
    elif args.command == "render":
        render()
    elif args.command == "verify":
        sys.exit(verify())
    elif args.command == "run":
        run()


if __name__ == "__main__":
    main()
