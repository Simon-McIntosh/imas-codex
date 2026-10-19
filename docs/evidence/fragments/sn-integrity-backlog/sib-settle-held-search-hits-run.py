"""Settle each referred standard name whose unit-matching DD hit a live name holds.

The DD-search repair step dropped a candidate DD path whenever another live
name already held it, and referred the name to the step's lead. For such a name
the question is whether it denotes the same quantity as one of those holders: a
duplicate is retired in the holder's favour through the signed supersede with a
successor link, and any other answer leaves every name in place and refers the
name to the lead with its holders.

The work list is re-derived from the live graph. The candidates come from the
search step's corrected referrals and the lineage step's held lineage paths,
then are filtered to the classifier's own live sourceless population through
``classify_sourceless_standard_names`` (the section's read-back instrument).
Each name is put to the ``sn-judge`` seat as one Choice through ``decide_batch``
whose options are one ``same quantity as <holder>`` per distinct holder, plus a
different-quantity answer and a cannot-tell answer. A supersede is written only
when the chosen option's probability is at least ``PROBABILITY_FLOOR`` and the
self-reported confidence is at least ``CONFIDENCE_FLOOR``.

Every graph write goes through ``apply_signed_manifest``: the authority is
built by ``build_repair_authority``, previewed, and written under the preview's
digest. The apply path itself calls the repository's own graph-activity probe
inside ``imas_codex.graph.offsite`` immediately before the signed write and
refuses when the graph is not quiet, so a replayed write on a busy graph stops.

Modes: read | judge | authority | preview | apply | readback | negcontrol |
quiet | quiet-refuse | overlap | guard | verify | falsify | record.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import subprocess
import time
from collections import defaultdict
from pathlib import Path

from imas_codex.graph import GraphClient
from imas_codex.standard_names.graph_ops import classify_sourceless_standard_names
from imas_codex.standard_names.repair_authority import build_repair_authority
from imas_codex.standard_names.signed_manifest import (
    apply_signed_manifest,
    signed_payload_sha256,
)

HERE = Path(__file__).resolve().parent
# Derived data (the live cohort read, the Jev answers) belongs in the run
# directory, not the repository: the fragment directory carries only the
# signed authority, the run driver, and the composed record. The run directory
# and the run id are runtime facts, so they arrive as arguments rather than
# being baked into the source of a driver that a different run may replay.
AUTHORITY_PATH = HERE / "sib-settle-held-search-hits-authority.json"
# The referral sources this node settles against: the search step's dropped
# hits and the lineage step's held lineage paths. The accounts are read from
# the fragment directory, which travels with the code.
SEARCH_RECORD = HERE / "sib-restore-searched-sources.json"
LINEAGE_RECORD = HERE / "sib-restore-lineage-sources.json"
UNCHAINED_RECORD = HERE / "sib-settle-unchained-holders.json"
RECORDED_RECORD = HERE / "sib-restore-recorded-sources.json"

RUN_DIR: Path | None = None
COHORT_PATH: Path | None = None
ANSWERS_PATH: Path | None = None
RUN_ID: str | None = None


def _resolve_runtime(args: argparse.Namespace) -> None:
    """Bind the run directory and run id from the arguments or the environment."""
    global RUN_DIR, COHORT_PATH, ANSWERS_PATH, RUN_ID  # noqa: PLW0603
    run_dir = args.run_dir or os.environ.get("SIB_RUN_DIR")
    if not run_dir:
        raise SystemExit("--run-dir is required (or set SIB_RUN_DIR)")
    RUN_DIR = Path(run_dir)
    COHORT_PATH = RUN_DIR / "sib-settle-held-search-hits-cohort.json"
    ANSWERS_PATH = RUN_DIR / "sib-settle-held-search-hits-answers.json"
    run_id = args.run_id or os.environ.get("SIB_RUN_ID")
    if not run_id:
        raise SystemExit("--run-id is required (or set SIB_RUN_ID)")
    RUN_ID = run_id


# Coordinator ruling: the floors every binding and retirement under this
# section clears, against the chosen option's probability and the model's own
# reported confidence.
PROBABILITY_FLOOR = 0.8
CONFIDENCE_FLOOR = 0.5

# One "same quantity as <holder>" option per distinct holder, plus these two.
SAME_PREFIX = "same quantity as "
DIFFERENT = "a different quantity from all of these"
CANNOT_TELL = "cannot tell"

QUESTION_NAME = "quantity_verdict"
HELD_REASON = re.compile(r"^source held by a live name:\s*(.+)$")
LINEAGE_PREFIX = "every lineage-recorded path is unboundable:"
LINEAGE_MARKER = ": source held by live name"

SERVICE = "standard-names"
COST_STEP = "sn_judge"
JUDGE_CONCURRENCY = 8
JUDGE_SPEND_GUARD_USD = 1.0

SELECTION = {
    "id": "artifact-rows",
    "mode": "exact_complete_signed_cohort",
    "predicate": "artifact-rows",
}
RETIRE_OPERATION = "supersede_held_source_name"
RETIRE_OPERATION_ID = "retire-held-source-name-in-favour-of-its-holder"
REASON = (
    "the name denotes the same quantity as the live name that holds the DD "
    "path the search step could not bind; retire the duplicate in the holder's "
    "favour through the signed supersede with a successor link"
)
SUPERSEDE_OPERATION = "link_signed_successor"


def _question(holder_ids: list[str]) -> dict:
    """One question naming every holder as a ``same quantity as`` option."""
    criteria = {
        f"{SAME_PREFIX}`{holder_id}`": (
            "The name denotes the same physical quantity as this holder: same "
            "subject, same base quantity, same coordinate and region, so the "
            "name is a duplicate of that holder."
        )
        for holder_id in holder_ids
    }
    criteria[DIFFERENT] = (
        "The name denotes a different physical quantity from every listed holder."
    )
    criteria[CANNOT_TELL] = (
        "The descriptions, units and documentation do not settle whether the "
        "name denotes the same quantity as any listed holder."
    )
    return {
        QUESTION_NAME: {
            "type": "choice",
            "instructions": (
                "A live IMAS standard name is shown first: it was recorded "
                "against the DD path shown, but every path it was recorded "
                "against is now held by another live name, so the search step "
                "could not bind it and referred it to the lead. Each holder is "
                "listed beside the DD path it holds. Decide whether the name "
                "denotes the same physical quantity as exactly one holder (so "
                "it is a duplicate to retire in that holder's favour), a "
                "different quantity from all of them, or whether the evidence "
                "does not settle it."
            ),
            "criteria": criteria,
        }
    }


def _holder_ids(detail: dict) -> list[str]:
    return [h["id"] for h in detail["holders"]]


def _held_hits(reason: str, path: str = "") -> list[tuple[str, str]]:
    """Yield ``(path, holder)`` pairs named by one held-hit reason string.

    The search step's held-hit reason names only the holder; the DD path that
    holder already holds is carried on the dropped hit itself, so it arrives
    here as ``path``. The lineage step's reason names both, path first.
    """
    text = str(reason)
    match = HELD_REASON.match(text)
    if match:
        return [(path, t.strip()) for t in match.group(1).split(",") if t.strip()]
    if LINEAGE_MARKER not in text:
        return []
    out: list[tuple[str, str]] = []
    for clause in text.split("; "):
        clause = clause.strip()
        if LINEAGE_MARKER not in clause:
            continue
        head, _, rest = clause.partition(LINEAGE_MARKER)
        path = head.removeprefix(LINEAGE_PREFIX).strip()
        path = path.split(":")[0].strip() if ":" in path else path
        for holder in rest.split(","):
            holder = holder.strip()
            if holder:
                out.append((path, holder))
    return out


def _candidate_worklist() -> dict[str, dict]:
    """The names the search and lineage steps referred, with their holders."""
    search = _load(SEARCH_RECORD)
    lineage = _load(LINEAGE_RECORD)
    unchained = _load(UNCHAINED_RECORD)
    excluded = {row["id"] for row in unchained.get("referrals", [])}
    sources: dict[str, dict] = {}

    def entry(name: str) -> dict:
        return sources.setdefault(
            name, {"held": {}, "steps": set(), "excluded": name in excluded}
        )

    def add_hits(row: dict, reason: str, path: str = "") -> None:
        for hit_path, holder in _held_hits(reason, path):
            if holder and holder not in row["held"]:
                row["held"][holder] = hit_path

    for name, blob in search["corrected_filter"].items():
        row = entry(name)
        row["steps"].add("corrected_filter")
        for hit in blob.get("dropped_hits", []):
            add_hits(row, hit.get("reason", ""), hit.get("path", ""))
    for row in search["corrected_referred_to_lead"]:
        entry(row["name"])["steps"].add("corrected_referred_to_lead")
    for row in search["corrected_no_surviving_candidate"]:
        entry(row["name"])["steps"].add("corrected_no_surviving_candidate")
    for name in search.get("referred_to_dd_search", []):
        entry(name)["steps"].add("searched_referred_to_dd_search")
    for record in lineage.get("referred_to_dd_search", []):
        row = entry(record["name"])
        row["steps"].add("lineage_referred_to_dd_search")
        add_hits(row, record.get("reason", ""))
    return sources


def suspect_unit_binding(name_unit, held_path_unit) -> bool:
    """Whether a holder's declared unit differs from its held DD path's unit.

    Both sides must be present: a missing unit on either side is absence of
    evidence, not a mismatch, so only a pair the graph can actually compare is
    reported. One predicate serves the derivation and the cohort check's
    planted-mismatch control, so the control exercises the real comparison.
    """
    return bool(name_unit and held_path_unit and str(name_unit) != str(held_path_unit))


def read() -> dict:
    """Re-derive the sourceless population and enrich each held pair's text.

    The candidates come from the search step's corrected referrals and the
    lineage step's held lineage paths; the cohort is the subset still
    sourceless on the live graph. Each name carries every holder that currently
    holds a path its own record named, the DD documentation and unit of each
    held path, and a deterministic flag wherever a holder's unit differs from
    the unit the DD path declares.
    """
    sources = _candidate_worklist()
    with GraphClient() as gc:
        uri = gc.uri
        classified = classify_sourceless_standard_names(gc)
        counts: dict[str, int] = defaultdict(int)
        for row in classified:
            counts[row["population"]] += 1
        by_id = {row["id"]: row for row in classified}
        names = [
            name
            for name in sorted(sources)
            if name in by_id and not sources[name]["excluded"]
        ]
        not_sourceless = sorted(name for name in sources if name not in by_id)
        all_holders = sorted({h for name in names for h in sources[name]["held"]})
        holder_rows = gc.query(
            "UNWIND $ids AS id MATCH (sn:StandardName {id:id}) "
            "RETURN sn.id AS id, properties(sn) AS p",
            ids=all_holders,
        )
        holder_props = {row["id"]: row["p"] for row in holder_rows}
        all_paths = sorted(
            {p for name in names for p in sources[name]["held"].values() if p}
        )
        dd_rows = gc.query(
            "UNWIND $paths AS p MATCH (dd:IMASNode {id:p}) "
            "RETURN dd.id AS id, dd.documentation AS documentation, "
            "coalesce(dd.unit, dd.units) AS units",
            paths=all_paths,
        )
        dd_by_path: dict[str, dict] = {}
        for row in dd_rows:
            dd_by_path[str(row["id"]).removeprefix("dd:")] = {
                "documentation": row["documentation"],
                "units": row["units"],
            }
        node_rows = gc.query(
            "UNWIND $ids AS id MATCH (sn:StandardName {id:id}) "
            "RETURN sn.id AS id, properties(sn) AS p",
            ids=names,
        )
        node_props = {row["id"]: row["p"] for row in node_rows}
        details = []
        for name in names:
            row = by_id[name]
            props = node_props.get(name, {})
            holders = []
            for holder in sorted(sources[name]["held"]):
                p = holder_props.get(holder, {})
                path = sources[name]["held"][holder]
                dd = dd_by_path.get(path.removeprefix("dd:"), {})
                holders.append(
                    {
                        "id": holder,
                        "name_stage": p.get("name_stage"),
                        "status": p.get("status"),
                        "description": p.get("description"),
                        "documentation": p.get("documentation"),
                        "unit": p.get("unit"),
                        "physics_domain": p.get("physics_domain"),
                        "held_path": path,
                        "held_path_unit": dd.get("units"),
                        "held_path_documentation": dd.get("documentation"),
                        "suspect_unit_binding": suspect_unit_binding(
                            p.get("unit"), dd.get("units")
                        ),
                    }
                )
            details.append(
                {
                    "id": name,
                    "population": row["population"],
                    "name_stage": row["name_stage"],
                    "status": row["status"],
                    "origin": row["origin"],
                    "description": props.get("description"),
                    "documentation": props.get("documentation"),
                    "unit": props.get("unit"),
                    "physics_domain": props.get("physics_domain"),
                    "steps": sorted(sources[name]["steps"]),
                    "holders": holders,
                    "suspect_holders": [
                        h["id"] for h in holders if h["suspect_unit_binding"]
                    ],
                    "shared_paths": sorted(
                        {h["held_path"] for h in holders if h["held_path"]}
                    ),
                }
            )
        out = {
            "uri": uri,
            "counts": dict(counts),
            "sourceless_total": len(classified),
            "candidate_count": len(sources),
            "worklist_count": len(details),
            "not_sourceless": not_sourceless,
            "no_holder": [d["id"] for d in details if not d["holders"]],
            "cohort": details,
        }
    COHORT_PATH.write_text(json.dumps(out, indent=2, default=str) + "\n")
    return out


def _prior_baseline(prior_record: str | None) -> tuple[dict, dict]:
    """The population before this node's first write, plus the names it retired.

    A later round can only see the graph as it stands, so the population before
    anything was written survives only in the first round's record. Its counts
    are the read-back baseline and its retired names are added back, so the
    cumulative check still asserts this node's whole footprint rather than the
    last round's slice of it.
    """
    current = json.loads(COHORT_PATH.read_text())
    if not prior_record:
        return current, {}
    record = json.loads(Path(prior_record).read_text())
    retired = [
        row for row in record.get("per_name", []) if row.get("action") == "superseded"
    ]
    cohort = [
        {"id": row["id"], "population": row.get("population")}
        for row in record.get("per_name", [])
    ]
    known = {row["id"] for row in cohort}
    cohort += [
        {"id": d["id"], "population": d["population"]}
        for d in current["cohort"]
        if d["id"] not in known
    ]
    before = {
        "counts": record["before"],
        "sourceless_total": record["sourceless_before"],
        "cohort": cohort,
    }
    prior = {
        "targets": sorted(row["id"] for row in retired),
        "retired": len(retired),
        "authority_file_sha256": record["authority"]["file_sha256"],
    }
    return before, prior


def cohort_check(cohort_file: Path | None = None) -> dict:
    """Assert the derivation recorded every held pair the two steps named.

    Two defects can each empty the suspect-binding check without failing
    anything: a holder recorded without the DD path it holds, which leaves the
    comparison no unit to compare, and a lineage reason whose holder was never
    attached to its name, which judges the name with no holder at all. This
    reads the cohort back and asserts both, then re-runs the suspect comparison
    on a planted mismatch so the check is shown able to fire.
    """
    cohort = json.loads(Path(cohort_file or COHORT_PATH).read_text())
    details = {d["id"]: d for d in cohort["cohort"]}

    expected: dict[str, dict[str, str]] = {}
    for record in _load(LINEAGE_RECORD).get("referred_to_dd_search", []):
        for held_path, holder in _held_hits(record.get("reason", "")):
            expected.setdefault(record["name"], {})[holder] = held_path
    missing_lineage = [
        {"name": name, "holder": holder, "held_path": held_path}
        for name, pairs in expected.items()
        if name in details
        for holder, held_path in pairs.items()
        if holder not in {h["id"] for h in details[name]["holders"]}
    ]

    holders = [h for d in cohort["cohort"] for h in d["holders"]]
    without_path = [
        {"name": d["id"], "holder": h["id"]}
        for d in cohort["cohort"]
        for h in d["holders"]
        if not h.get("held_path")
    ]
    without_unit = [
        {"name": d["id"], "holder": h["id"], "held_path": h.get("held_path")}
        for d in cohort["cohort"]
        for h in d["holders"]
        if not h.get("held_path_unit")
    ]
    suspect = [
        {
            "name": d["id"],
            "holder": h["id"],
            "unit": h.get("unit"),
            "held_path_unit": h.get("held_path_unit"),
        }
        for d in cohort["cohort"]
        for h in d["holders"]
        if h.get("suspect_unit_binding")
    ]
    sample = next(
        (h for h in holders if h.get("held_path_unit") and h.get("unit")), None
    )
    planted = None
    if sample:
        planted = {
            "name": sample["id"],
            "holder_unit": sample["unit"],
            "held_path_unit": sample["held_path_unit"],
            "planted_held_path_unit": f"{sample['held_path_unit']} (planted-mismatch)",
        }
        planted["fires"] = suspect_unit_binding(
            planted["holder_unit"], planted["planted_held_path_unit"]
        )
        planted["holds_on_the_unmutated_pair"] = suspect_unit_binding(
            planted["holder_unit"], planted["held_path_unit"]
        )
    planted_fires = bool(planted and planted["fires"])
    out = {
        "cohort_file": str(cohort_file or COHORT_PATH),
        "names": len(cohort["cohort"]),
        "holders": len(holders),
        "holders_with_held_path": len(holders) - len(without_path),
        "holders_with_dd_unit": len(holders) - len(without_unit),
        "lineage_held_pairs_expected": sum(len(p) for p in expected.values()),
        "lineage_holders_missing": missing_lineage,
        "holders_without_path": without_path[:20],
        "holders_without_dd_unit": without_unit[:20],
        "suspect_holder_count": len(suspect),
        "suspect_holders": suspect,
        "planted_mismatch": planted,
        "planted_mismatch_fires": planted_fires,
    }
    out["checks"] = {
        "every_lineage_holder_recorded": not missing_lineage,
        "every_holder_carries_its_held_path": not without_path,
        "every_held_path_resolved_to_a_dd_unit": not without_unit,
        "suspect_detector_fires_on_a_planted_mismatch": planted_fires,
    }
    if not all(out["checks"].values()):
        raise SystemExit(f"cohort check failed: {out['checks']} :: {out}")
    return out


def _state(detail: dict) -> dict:
    return {
        "recorded_name": {
            "id": detail["id"],
            "description": detail["description"],
            "documentation": detail["documentation"],
            "unit": detail["unit"],
            "physics_domain": detail["physics_domain"],
        },
        "holders": [
            {
                "id": h["id"],
                "description": h.get("description"),
                "documentation": h.get("documentation"),
                "unit": h.get("unit"),
                "physics_domain": h.get("physics_domain"),
                "name_stage": h.get("name_stage"),
            }
            for h in detail["holders"]
        ],
        "held_dd_paths": [
            {
                "path": h.get("held_path"),
                "documentation": h.get("held_path_documentation"),
                "unit": h.get("held_path_unit"),
                "held_by": h["id"],
            }
            for h in detail["holders"]
        ],
    }


def judge() -> dict:
    """Put each name to the seat as one Choice whose options are its holders."""
    from imas_codex.discovery.base.judgment import decide_batch
    from imas_codex.settings import get_model

    cohort = json.loads(COHORT_PATH.read_text())["cohort"]
    model = get_model("sn-judge")
    holderless = [d for d in cohort if not _holder_ids(d)]
    work = [d for d in cohort if _holder_ids(d)]
    semaphore = asyncio.Semaphore(JUDGE_CONCURRENCY)
    spent_total = 0.0

    async def _one(detail: dict) -> tuple[dict, float]:
        holder_ids = _holder_ids(detail)
        async with semaphore:
            results, spent = await decide_batch(
                [_state(detail)],
                _question(holder_ids),
                model=model,
                service=SERVICE,
                concurrency=1,
                budget_seconds=180,
                step=COST_STEP,
            )
        answer = (
            (results[0][0].get(QUESTION_NAME) or {}) if results and results[0] else {}
        )
        choice = answer.get("choice")
        holder = None
        if isinstance(choice, str) and choice.startswith(SAME_PREFIX):
            holder = choice[len(SAME_PREFIX) :].strip().strip("`")
            if holder not in holder_ids:
                holder = None
        probabilities = answer.get("probabilities")
        return (
            {
                "id": detail["id"],
                "holders": holder_ids,
                "choice": choice,
                "holder": holder,
                "probability": (probabilities or {}).get(choice),
                "confidence": answer.get("confidence"),
                "probabilities": probabilities,
                "answered": bool(answer),
                "description": detail["description"],
                "shared_paths": detail["shared_paths"],
            },
            float(spent or 0),
        )

    async def _run_all() -> list[tuple[dict, float]]:
        return await asyncio.gather(*(_one(d) for d in work))

    outcomes = asyncio.run(_run_all())
    records = [
        {
            "id": d["id"],
            "holders": [],
            "choice": None,
            "holder": None,
            "probability": None,
            "confidence": None,
            "probabilities": None,
            "answered": False,
            "note": "no live holder was named by either step's record",
            "description": d["description"],
            "shared_paths": d["shared_paths"],
        }
        for d in holderless
    ]
    for record, spent in outcomes:
        records.append(record)
        spent_total += spent
    records.sort(key=lambda r: r["id"])
    out = {
        "model": model,
        "spend_usd_guard": JUDGE_SPEND_GUARD_USD,
        "spend_usd": round(spent_total, 8),
        "judged_count": sum(1 for r in records if r["answered"]),
        "records": records,
    }
    if out["spend_usd"] > JUDGE_SPEND_GUARD_USD:
        raise SystemExit(f"judge spend {out['spend_usd']} over the guard")
    ANSWERS_PATH.write_text(json.dumps(out, indent=2, default=str) + "\n")
    return out


def _admitted(records: list[dict]) -> list[dict]:
    """The records cleared for a supersede by the ruling's floors."""
    out = []
    for rec in records:
        choice = rec.get("choice")
        holder = rec.get("holder")
        if not isinstance(choice, str) or not choice.startswith(SAME_PREFIX):
            continue
        if not holder or holder not in (rec.get("holders") or []):
            continue
        prob = rec.get("probability")
        conf = rec.get("confidence")
        if prob is None or conf is None:
            continue
        if float(prob) >= PROBABILITY_FLOOR and float(conf) >= CONFIDENCE_FLOOR:
            out.append(rec)
    return out


def _supersede_row(name_id: str, holder_id: str) -> dict:
    row_id = f"{name_id}=>{holder_id}"
    return {
        "id": row_id,
        "identity": {"id": row_id, "kind": "standard_name", "target_id": name_id},
        "participants": [
            {"id": name_id, "kind": "node", "graph_label": "StandardName"},
            {"id": holder_id, "kind": "node", "graph_label": "StandardName"},
        ],
        "selection": dict(SELECTION),
        "mutations": [
            {
                "id": f"{name_id}:supersede",
                "order": 0,
                "kind": "supersede",
                "participant_id": name_id,
                "arguments": {"successor_id": holder_id},
            }
        ],
        "guards": [
            {
                "id": f"{row_id}:structural-legitimacy",
                "kind": "semantic_authority",
                "implementation": "structural-legitimacy",
                "participant_ids": [name_id],
            },
            {
                "id": f"{row_id}:out-of-allowlist-immutability",
                "kind": "collateral_immutability",
                "implementation": "out-of-allowlist-immutability",
                "participant_ids": [name_id, holder_id],
            },
        ],
        "orphan_policy": "refuse",
    }


def build_authority() -> dict:
    answers = json.loads(ANSWERS_PATH.read_text())
    admitted = _admitted(answers["records"])
    rows = [_supersede_row(r["id"], r["holder"]) for r in admitted]
    authority = {
        "schema": "imas-codex.repair-authority.v1",
        "operation_id": RETIRE_OPERATION_ID,
        "authority_mode": "external_reviewed",
        "rows": rows,
        "repair_rows": [row["id"] for row in rows],
        "selection": dict(SELECTION),
        "receipt_policy": {
            "id": "one-per-superseded-identity",
            "operation": SUPERSEDE_OPERATION,
            "cardinality": "per_target",
            "expected_count": "admitted_rows",
            "link_participant_kind": "standard_name",
            "replay_projection": ["manifest_sha256", "row_id"],
        },
        "orphan_policy": "refuse",
    }
    built = build_repair_authority(authority)
    AUTHORITY_PATH.write_bytes(built.content)
    return {
        "rows": len(rows),
        "admitted": [r["id"] for r in admitted],
        "file_sha256": hashlib.sha256(AUTHORITY_PATH.read_bytes()).hexdigest(),
        "payload_sha256": built.payload_sha256,
    }


def _quiet_gate(probe=None) -> dict:
    """Refuse the signed write when another graph client is working.

    ``live_graph_activity`` in ``imas_codex.graph.offsite`` owns this
    measurement: it runs every probe in one session, identifies its own
    transaction by the ``SHOW TRANSACTIONS`` query text, and subtracts the
    probes from the id gap. The apply path calls it here, immediately before
    the signed write, so a replayed apply on a busy graph stops instead of
    racing a foreign writer. The probe is injectable so a run can show the
    refusal without planting a live writer.
    """
    from imas_codex.graph.offsite import live_graph_activity

    observed = (probe or live_graph_activity)(window_seconds=6.0, interval_seconds=3.0)
    reading = {
        "window_seconds": observed.window_seconds,
        "foreign_transactions_started": observed.transactions_started,
        "foreign_open_queries": observed.open_queries,
        "busy": observed.busy,
    }
    if observed.busy:
        raise SystemExit(f"graph is not quiet; refusing the signed write: {reading}")
    return reading


def _signed_call(apply: bool, digest: str | None) -> dict:
    raw = AUTHORITY_PATH.read_bytes()
    authority = json.loads(raw)
    gate = _quiet_gate() if apply else None
    res = apply_signed_manifest(
        AUTHORITY_PATH,
        authority_file_sha256=hashlib.sha256(raw).hexdigest(),
        authority_payload_sha256=signed_payload_sha256(authority),
        reason=REASON,
        apply=apply,
        manifest_sha256=digest if apply else None,
        run_id=RUN_ID,
    )
    out = {k: res[k] for k in res if k not in {"manifest", "admitted_actions"}}
    out["quiet_gate"] = gate
    return out


def readback(
    before: dict | None = None,
    raise_on_mismatch: bool = False,
    prior: dict | None = None,
) -> dict:
    raw = AUTHORITY_PATH.read_bytes()
    file_sha = hashlib.sha256(raw).hexdigest()
    authority = json.loads(raw)
    targets = {row["identity"]["target_id"] for row in authority["rows"]}
    targets |= set((prior or {}).get("targets") or [])
    shas = [file_sha, *[s for s in [(prior or {}).get("authority_file_sha256")] if s]]
    with GraphClient() as gc:
        classified = classify_sourceless_standard_names(gc)
        counts: dict[str, int] = defaultdict(int)
        for row in classified:
            counts[row["population"]] += 1
        sourceless_ids = {row["id"] for row in classified}
        receipts = gc.query(
            "MATCH (c:StandardNameChange) WHERE c.authority_file_sha256 IN $fss "
            "RETURN c.row_id AS row_id, c.operation AS op, "
            "c.manifest_sha256 AS digest",
            fss=shas,
        )
        per: dict[str, int] = defaultdict(int)
        for c in receipts:
            per[c["row_id"]] += 1
        receipt_targets = {row_id.split("=>", 1)[0] for row_id in per}
        stages = gc.query(
            "UNWIND $ids AS id MATCH (sn:StandardName {id:id}) "
            "RETURN sn.id AS id, sn.name_stage AS stage, sn.status AS status",
            ids=sorted(targets),
        )
    out = {
        "counts": dict(counts),
        "sourceless_total": len(classified),
        "targets": sorted(targets),
        "receipt_names": len(per),
        "receipt_targets": sorted(receipt_targets),
        "targets_without_receipt": sorted(targets - receipt_targets),
        "receipt_operations": sorted({str(c["op"]) for c in receipts}),
        "digests": sorted({str(c["digest"]) for c in receipts}),
        "retired_stages": sorted({(r["stage"], r["status"]) for r in stages}),
    }
    if before is not None:
        # The baseline is recorded as counts, not ids, so the population before
        # the apply is reconstructed as the population now plus the retired
        # names. If any name had entered the population since the baseline, that
        # reconstruction would carry more names than the recorded counts and the
        # per-population comparison below would fail; if any retired name were
        # still sourceless, the reconstruction would be short by it.
        reconstructed = sourceless_ids | targets
        out["before"] = before["counts"]
        out["expected_sourceless_total"] = before["sourceless_total"] - len(targets)
        out["retired_still_sourceless"] = sorted(targets & sourceless_ids)
        out["reconstructed_baseline_total"] = len(reconstructed)
        out["reconstructed_baseline_counts"] = dict(counts)
        moved = {
            k: (before["counts"].get(k), counts.get(k))
            for k in set(before["counts"]) | set(counts)
            if before["counts"].get(k) != counts.get(k)
        }
        out["population_moves"] = moved
        out["shortfall"] = sorted(targets - receipt_targets)
    if raise_on_mismatch:
        # The gate asserts this node's own footprint, which stays exact however
        # much another writer moves afterwards: the sourceless population falls
        # by exactly the retired count, every retired name has left it, no name
        # entered it (the population-plus-retired reproduces the recorded
        # baseline exactly), and every target carries a receipt. A movement in
        # another population is a foreign write and is reported, not a failure.
        baseline_rebuilt = dict(counts)
        recorded = {
            row["id"]: row.get("population") for row in before.get("cohort", [])
        }
        for name in targets:
            population = recorded.get(name)
            if population:
                baseline_rebuilt[population] = baseline_rebuilt.get(population, 0) + 1
        baseline_rebuilt = {k: v for k, v in baseline_rebuilt.items() if v}
        checks = {
            "sourceless_fell_by_retired": out["sourceless_total"]
            == (before["sourceless_total"] - len(targets)),
            "no_retired_name_still_sourceless": not out["retired_still_sourceless"],
            "nothing_entered_the_population": baseline_rebuilt == before["counts"],
            "every_target_has_receipt": not (targets - receipt_targets),
            "no_receipt_outside_targets": not (receipt_targets - targets),
            "all_targets_superseded": out["retired_stages"]
            == [("superseded", "superseded")],
        }
        if not all(checks.values()):
            raise SystemExit(f"read-back mismatch: {checks} :: {out}")
        out["checks"] = checks
        out["population_moves_from_baseline"] = moved
        # The movement above is measured from the baseline this node recorded
        # before its first round, so it carries this node's own retirements too.
        # Split them out: only the remainder is another writer's.
        own: dict[str, int] = {}
        for name in targets:
            population = recorded.get(name)
            if population:
                own[population] = own.get(population, 0) + 1
        out["own_retirements_by_population"] = own
        out["foreign_moves"] = {
            population: counts.get(population, 0)
            - (before["counts"].get(population, 0) - own.get(population, 0))
            for population in moved
        }
    return out


def quiet_graph_check() -> dict:
    """Show whether another graph client is writing, through the owner probe.

    ``live_graph_activity`` is the repository's owner of this measurement: it
    runs every probe in one session, identifies its own transaction by the
    ``SHOW TRANSACTIONS`` query text, subtracts the probes from the id gap, and
    raises when its own probe is absent. It is called directly rather than
    re-implemented, so the check cannot pass through a local filter that never
    matches.
    """
    from imas_codex.graph.offsite import live_graph_activity

    live_dir = Path.home() / ".config" / "reckon" / "crew" / "live"
    runs = []
    recently_touched = []
    now = time.time()
    if live_dir.is_dir():
        for pointer in sorted(live_dir.glob("*.json")):
            runs.append(pointer.stem)
            age = now - pointer.stat().st_mtime
            if age < 1800:
                recently_touched.append(
                    {"run": pointer.stem, "age_seconds": round(age)}
                )
    try:
        job = subprocess.run(
            ["squeue", "-u", str(Path.home().name), "-h", "-o", "%i %j %T %M"],
            capture_output=True,
            text=True,
            check=False,
        )
        neo4j_jobs = [ln for ln in job.stdout.splitlines() if "codex-neo4j" in ln]
    except Exception as exc:  # listing is best-effort; record the refusal
        neo4j_jobs = [{"error": f"{type(exc).__name__}: {exc}"}]
    activity = live_graph_activity(window_seconds=6.0, interval_seconds=3.0)
    return {
        "live_crew_runs": runs,
        "live_crew_run_count": len(runs),
        "runs_touched_within_1800s": recently_touched,
        "neo4j_jobs": neo4j_jobs,
        "activity_window_seconds": activity.window_seconds,
        "foreign_transactions_started": activity.transactions_started,
        "foreign_open_queries": activity.open_queries,
        "quiet": not activity.busy,
    }


def overlap_check() -> dict:
    """Establish after the fact whether another writer touched the settled names.

    The quiet-graph probe was vacuous at apply time, so this reads the graph
    directly: every change record, source write and name update in the apply
    window (from just before this authority's first receipt to just after its
    last). The window comes from the receipts themselves, keyed by the
    authority's file digest, so it is reproducible after the fact and does not
    depend on a preview that now reports the manifest already applied.
    """
    from datetime import timedelta

    answers = json.loads(ANSWERS_PATH.read_text())
    admitted = _admitted(answers["records"])
    ids = sorted({r["id"] for r in admitted} | {r["holder"] for r in admitted})
    file_sha = hashlib.sha256(AUTHORITY_PATH.read_bytes()).hexdigest()
    with GraphClient() as gc:
        mine = gc.query(
            "MATCH (c:StandardNameChange {authority_file_sha256:$fs}) "
            "RETURN min(c.changed_at) AS lo, max(c.changed_at) AS hi, "
            "count(c) AS n, collect(DISTINCT c.manifest_sha256) AS digests",
            fs=file_sha,
        )[0]
        if mine["lo"] is None:
            raise SystemExit("no receipt carries this authority's file digest")
        lo = mine["lo"] - timedelta(seconds=180)
        hi = mine["hi"] + timedelta(seconds=180)
        changes = gc.query(
            "MATCH (c:StandardNameChange) WHERE c.changed_at >= $lo "
            "AND c.changed_at <= $hi "
            "RETURN c.run_id AS run_id, c.operation AS op, c.row_id AS row_id, "
            "c.from_name AS frm, c.to_name AS to, c.changed_at AS at ORDER BY at",
            lo=lo,
            hi=hi,
        )
        sources = gc.query(
            "MATCH (s:StandardNameSource)-[:PRODUCED_NAME]->(n:StandardName) "
            "WHERE s.created_at >= $lo AND s.created_at <= $hi "
            "RETURN s.id AS source, n.id AS name, s.created_at AS at ORDER BY at",
            lo=lo,
            hi=hi,
        )
        touched = gc.query(
            "UNWIND $ids AS id MATCH (n:StandardName {id:id}) "
            "WHERE n.updated_at >= $lo AND n.updated_at <= $hi "
            "RETURN n.id AS id, n.updated_at AS at ORDER BY at",
            ids=ids,
            lo=lo,
            hi=hi,
        )
        # A writer that links an already-existing source node to a name leaves
        # ``created_at`` untouched and stamps ``composed_at``; that is invisible
        # to a created_at window and must be read separately.
        rebound = gc.query(
            "MATCH (s:StandardNameSource)-[:PRODUCED_NAME]->(n:StandardName) "
            "WHERE s.composed_at > $hi "
            "RETURN s.id AS source, n.id AS name, s.created_at AS created, "
            "s.composed_at AS at ORDER BY at",
            hi=mine["hi"],
        )
    mine_run = RUN_ID
    names = set(ids)
    foreign_changes = [
        row for row in changes if str(row.get("run_id")) != str(mine_run)
    ]
    foreign_touching = [
        row
        for row in foreign_changes
        if str(row.get("row_id") or "") in names
        or str(row.get("frm") or "") in names
        or str(row.get("to") or "") in names
    ]
    foreign_sources = [row for row in sources if str(row.get("name")) in names]
    return {
        "window": {"from": str(lo), "to": str(hi)},
        "own_receipts": dict(mine),
        "changes_in_window": changes,
        "foreign_changes": foreign_changes,
        "foreign_changes_touching_settled_names": foreign_touching,
        "sources_in_window": sources,
        "foreign_sources_touching_settled_names": foreign_sources,
        "settled_name_updates": touched,
        "sources_rebound_after_last_receipt": rebound,
        "updated_at_read_is_determinative": bool(touched),
        "overlap": bool(foreign_changes or foreign_sources),
    }


def shared_source_path_guard(gc) -> dict:
    """Prove each admitted supersede leaves the survivor holding the source.

    The predecessor recorded the DD path but binds no ``StandardNameSource``
    (that is why it is sourceless); the successor holds the binding. Retiring
    the predecessor therefore drops no source, and the path this pair is about
    stays held by the successor.

    The pair's own path is the one the admitted holder holds, not every path
    any of the predecessor's holders holds: a predecessor may share different
    paths with different names, and only the path that pairs it with the
    admitted holder is what this supersede is retiring. The check raises if an
    admitted predecessor carries a live source binding, or if the admitted
    holder does not hold the path the pair was referred on.
    """
    answers = json.loads(ANSWERS_PATH.read_text())
    admitted = _admitted(answers["records"])
    cohort = json.loads(COHORT_PATH.read_text())["cohort"]
    pair_path = {
        (d["id"], h["id"]): h.get("held_path") for d in cohort for h in d["holders"]
    }
    ids = sorted({r["id"] for r in admitted} | {r["holder"] for r in admitted})
    rows = gc.query(
        "UNWIND $ids AS id MATCH (sn:StandardName {id:id}) "
        "RETURN sn.id AS id, sn.source_paths AS paths, "
        "COUNT { (:StandardNameSource)-[:PRODUCED_NAME]->(sn) "
        "        WHERE coalesce(sn.status,'') <> 'stale' } AS producers",
        ids=ids,
    )
    by_id = {row["id"]: row for row in rows}
    pairs = []
    for rec in admitted:
        pred = by_id.get(rec["id"], {})
        succ = by_id.get(rec["holder"], {})
        pair_held = str(pair_path.get((rec["id"], rec["holder"])) or "").removeprefix(
            "dd:"
        )
        if not pair_held:
            raise SystemExit(
                "no held path on the cohort for the admitted pair "
                f"{rec['id']} <- {rec['holder']}: the pair cannot be checked"
            )
        succ_paths = {str(p).removeprefix("dd:") for p in (succ.get("paths") or [])}
        missing = [] if pair_held in succ_paths else [pair_held]
        pairs.append(
            {
                "predecessor": rec["id"],
                "successor": rec["holder"],
                "predecessor_binds_source": int(pred.get("producers") or 0),
                "pair_held_path": pair_held,
                "also_recorded_on_predecessor": sorted(
                    str(p).removeprefix("dd:") for p in (rec.get("shared_paths") or [])
                ),
                "successor_holds_paths": sorted(succ_paths),
                "paths_successor_does_not_hold": missing,
            }
        )
    offenders = [
        p
        for p in pairs
        if p["predecessor_binds_source"] != 0 or p["paths_successor_does_not_hold"]
    ]
    if offenders:
        raise SystemExit(f"supersede would not leave the source held: {offenders}")
    return {"admitted": len(admitted), "pairs": pairs}


def negcontrol() -> dict:
    """Plant a different-quantity pair and show it is answered and not retired.

    A name plainly denoting a different quantity from its holder (a neutron
    rate against an antenna voltage amplitude) is put to the same seat, and the
    run shows the answer is not a ``same quantity as`` option and the pair is
    absent from the authority the floor builds.
    """
    from imas_codex.discovery.base.judgment import decide_batch
    from imas_codex.settings import get_model

    planted_detail = {
        "id": "PLANTED-neutron_rate",
        "description": "Total number of neutrons produced per second.",
        "documentation": "The neutron emission rate of the plasma.",
        "unit": "s-1",
        "physics_domain": "neutron",
        "holders": [
            {
                "id": "PLANTED-voltage_amplitude",
                "description": "Amplitude of the voltage applied to an antenna.",
                "documentation": "Peak radio-frequency voltage on the antenna.",
                "unit": "V",
                "physics_domain": "heating",
                "name_stage": "drafted",
                "held_path": "PLANTED/shared",
                "held_path_documentation": "Peak radio-frequency voltage.",
                "held_path_unit": "V",
            }
        ],
        "shared_paths": ["PLANTED/shared"],
    }

    async def _run() -> dict:
        results, spent = await decide_batch(
            [_state(planted_detail)],
            _question(["PLANTED-voltage_amplitude"]),
            model=get_model("sn-judge"),
            service=SERVICE,
            concurrency=1,
            budget_seconds=180,
            step=COST_STEP,
        )
        answer = (results[0][0].get(QUESTION_NAME) or {}) if results[0] else {}
        return {
            "choice": answer.get("choice"),
            "probabilities": answer.get("probabilities"),
            "confidence": answer.get("confidence"),
        }

    observed = asyncio.run(_run())
    choice = observed["choice"]
    holder = None
    if isinstance(choice, str) and choice.startswith(SAME_PREFIX):
        holder = choice[len(SAME_PREFIX) :].strip().strip("`")
    record = {
        "id": "PLANTED-neutron_rate",
        "holders": ["PLANTED-voltage_amplitude"],
        "holder": holder,
        "choice": choice,
        "probability": (observed["probabilities"] or {}).get(choice),
        "confidence": observed["confidence"],
        "answered": choice is not None,
    }
    admitted = _admitted([record])
    refused = holder is None and not admitted
    if not refused:
        raise SystemExit(f"planted different-quantity pair was not refused: {observed}")
    return {
        "planted": record,
        "admitted_by_floors": [r["id"] for r in admitted],
        "refused": refused,
    }


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def _gate_log(path: Path) -> dict:
    """Read a gate log carrying a command header, a JSON payload and an EXIT line."""
    text = path.read_text()
    lines = text.splitlines()
    header = [ln for ln in lines if ln.startswith("#")]
    exit_line = next((ln for ln in lines if ln.startswith("EXIT=")), None)
    body = "\n".join(
        ln for ln in lines if not ln.startswith("#") and not ln.startswith("EXIT=")
    )
    payload = json.loads(body.strip() or "{}")
    return {
        "header": header,
        "exit_line": exit_line,
        "exit_code": int(exit_line.split("=", 1)[1]) if exit_line else None,
        "payload": payload,
    }


def _stage_split(referrals: list[dict]) -> dict[str, int]:
    split: dict[str, int] = defaultdict(int)
    for row in referrals:
        split[str(row.get("name_stage"))] += 1
    return dict(split)


def _referral_totals(referrals: list[dict]) -> dict:
    """The §38 lead-referral total with this node's share added.

    The search step's own record carries the running total through its step;
    the unchained-holder step's record carries its referral count; this node
    adds its own referrals to the sum, split by whether the name is still live
    or already exhausted.
    """
    search = _load(SEARCH_RECORD)
    unchained = _load(UNCHAINED_RECORD)
    prior = search.get("lead_referral_total", {})
    prior_running = int(prior.get("running_total") or 0)
    unchained_referrals = int(unchained.get("referral_count") or 0)
    total = prior_running + unchained_referrals + len(referrals)
    return {
        "search_and_lineage_running_total": prior_running,
        "search_step_detail": prior,
        "unchained_holder_referrals": unchained_referrals,
        "this_node_referrals": len(referrals),
        "this_node_referrals_by_stage": _stage_split(referrals),
        "this_node_referrals_live": sum(
            v for k, v in _stage_split(referrals).items() if k != "exhausted"
        ),
        "this_node_referrals_exhausted": _stage_split(referrals).get("exhausted", 0),
        "running_total": total,
    }


def compose() -> dict:
    """Assemble the JSON record and HTML fragment from the run's artifacts."""
    from html import escape

    node = "sib-settle-held-search-hits"
    cohort = _load(COHORT_PATH)
    answers = _load(ANSWERS_PATH)
    auth_raw = AUTHORITY_PATH.read_bytes()
    authority_sha = hashlib.sha256(auth_raw).hexdigest()
    authority_payload_sha = signed_payload_sha256(json.loads(auth_raw))
    preview = _gate_log(RUN_DIR / "preview.log")["payload"]
    apply_receipt = _gate_log(RUN_DIR / "apply.log")["payload"]
    quiet = _gate_log(RUN_DIR / "quiet.log")["payload"]
    quiet_refuse = _gate_log(RUN_DIR / "quiet-refuse.log")
    overlap = _gate_log(RUN_DIR / "overlap.log")["payload"]
    receipts = overlap["own_receipts"]
    manifest_sha = next((d for d in receipts.get("digests") or [] if d), None)
    guard = _gate_log(RUN_DIR / "guard.log")["payload"]
    verify_log = _gate_log(RUN_DIR / "verify.log")
    falsify_log = _gate_log(RUN_DIR / "falsify.log")
    verify = verify_log["payload"]
    falsify = falsify_log["payload"]
    negcontrol = _gate_log(RUN_DIR / "negcontrol.log")["payload"]
    cohort_gate = _gate_log(RUN_DIR / "cohort-check.log")
    cohort_check_payload = (
        cohort_gate["payload"] if "payload" in cohort_gate else cohort_gate
    )
    prior_path = RUN_DIR / "round1-record.json"
    prior_record = json.loads(prior_path.read_text()) if prior_path.is_file() else None
    prior_retired = [
        row["id"]
        for row in (prior_record or {}).get("per_name", [])
        if row.get("action") == "superseded"
    ]

    details = {row["id"]: row for row in cohort["cohort"]}
    admitted_ids = {r["id"] for r in _admitted(answers["records"])}

    per_name = []
    for rec in answers["records"]:
        det = details.get(rec["id"], {})
        action = "superseded" if rec["id"] in admitted_ids else "referred to lead"
        if action == "referred to lead":
            if not rec.get("holders"):
                reason = "no holder named by either step's record"
            elif rec.get("choice") == CANNOT_TELL:
                reason = "Jev answered cannot tell"
            elif rec.get("choice") == DIFFERENT:
                reason = "Jev answered a different quantity"
            else:
                reason = "below the probability/confidence floor"
        else:
            reason = "same quantity as its holder, cleared both floors"
        per_name.append(
            {
                "id": rec["id"],
                "population": det.get("population"),
                "name_stage": det.get("name_stage"),
                "holders": [
                    {
                        "id": h["id"],
                        "unit": h.get("unit"),
                        "physics_domain": h.get("physics_domain"),
                        "held_path": h.get("held_path"),
                        "held_path_unit": h.get("held_path_unit"),
                        "suspect_unit_binding": h.get("suspect_unit_binding"),
                    }
                    for h in det.get("holders", [])
                ],
                "suspect_holders": det.get("suspect_holders", []),
                "shared_paths": det.get("shared_paths"),
                "choice": rec.get("choice"),
                "holder": rec.get("holder"),
                "probability": rec.get("probability"),
                "confidence": rec.get("confidence"),
                "answered": rec.get("answered"),
                "action": action,
                "reason": reason,
            }
        )

    referrals = [r for r in per_name if r["action"] == "referred to lead"]
    retired = [r for r in per_name if r["action"] == "superseded"]
    suspect = [r for r in per_name if r["suspect_holders"]]
    totals = _referral_totals(referrals)
    rounds = [
        {
            "round": 1,
            "what": "the first derivation, which dropped every search-step held path "
            "and never attached a lineage holder",
            "worklist": prior_record.get("worklist_count") if prior_record else None,
            "retired": prior_retired,
            "referrals": prior_record.get("referral_count") if prior_record else None,
            "sourceless_before": prior_record.get("sourceless_before")
            if prior_record
            else None,
            "sourceless_after": prior_record.get("sourceless_after")
            if prior_record
            else None,
        },
        {
            "round": 2,
            "what": "the corrected derivation: every search-step held path and every "
            "lineage holder recorded, every held path resolved to its DD unit",
            "worklist": cohort["worklist_count"],
            "retired": [r["id"] for r in retired],
            "referrals": len(referrals),
            "sourceless_before": cohort["sourceless_total"],
            "sourceless_after": verify["sourceless_total"],
        },
    ]
    source_counts = {
        step: sum(1 for d in cohort["cohort"] if step in d["steps"])
        for step in (
            "corrected_filter",
            "corrected_referred_to_lead",
            "corrected_no_surviving_candidate",
            "searched_referred_to_dd_search",
            "lineage_referred_to_dd_search",
            "lineage_referred_to_lead",
        )
    }

    record = {
        "node": node,
        "plan": "sn-integrity-backlog",
        "section": "§38",
        "population": "held_search_hits",
        "floors": {"probability": PROBABILITY_FLOOR, "confidence": CONFIDENCE_FLOOR},
        "before": cohort["counts"],
        "sourceless_before": cohort["sourceless_total"],
        "after": verify["counts"],
        "sourceless_after": verify["sourceless_total"],
        "candidate_count": cohort["candidate_count"],
        "worklist_count": cohort["worklist_count"],
        "not_sourceless": cohort["not_sourceless"],
        "no_holder": cohort["no_holder"],
        "source_counts": source_counts,
        "retired_count": len(retired),
        "referral_count": len(referrals),
        "suspect_holder_count": len(suspect),
        "spend_usd": answers["spend_usd"],
        "judge_model": answers["model"],
        "authority": {
            "path": f"docs/evidence/fragments/sn-integrity-backlog/{node}-authority.json",
            "file_sha256": authority_sha,
            "payload_sha256": authority_payload_sha,
            "manifest_sha256": manifest_sha,
            "operation_id": RETIRE_OPERATION_ID,
            "receipt_operation": SUPERSEDE_OPERATION,
        },
        "preview": preview,
        "receipt": apply_receipt,
        "quiet_graph": quiet,
        "quiet_graph_refusal": quiet_refuse,
        "population_moves_from_baseline": verify.get("population_moves_from_baseline")
        or {},
        "apply_overlap": overlap,
        "source_held_guard": guard,
        "verification": verify,
        "verification_log": verify_log,
        "falsification": falsify,
        "falsification_log": falsify_log,
        "negative_control": negcontrol,
        "cohort_check": cohort_check_payload,
        "rounds": rounds,
        "retired_total": len(prior_retired) + len(retired),
        "lead_referral_total": totals,
        "per_name": per_name,
        "referrals": referrals,
        "prior_round": (
            {
                "retired": prior_retired,
                "referral_count": prior_record.get("referral_count"),
                "sourceless_before": prior_record.get("sourceless_before"),
                "sourceless_after": prior_record.get("sourceless_after"),
                "authority_file_sha256": (prior_record.get("authority") or {}).get(
                    "file_sha256"
                ),
            }
            if prior_record
            else None
        ),
    }
    (HERE / f"{node}.json").write_text(json.dumps(record, indent=2, default=str) + "\n")

    def row_html(r: dict) -> str:
        return (
            "<tr><td>"
            + escape(str(r["id"]))
            + "</td><td>"
            + escape(", ".join(h["id"] for h in r["holders"]) or "—")
            + "</td><td>"
            + escape(str(r["choice"]))
            + "</td><td>"
            + str(r["probability"])
            + "</td><td>"
            + str(r["confidence"])
            + "</td><td>"
            + escape(r["action"])
            + "</td><td>"
            + escape(", ".join(r["shared_paths"] or []))
            + "</td></tr>"
        )

    foreign_moves = verify.get("population_moves_from_baseline") or {}
    own_moves = verify.get("own_retirements_by_population") or {}
    residual = verify.get("foreign_moves") or {}
    foreign_note = (
        "Measured from the population counts this node recorded before its first "
        "round, "
        + ", ".join(
            f"{name} {span[0]} to {span[1]}" for name, span in foreign_moves.items()
        )
        + ". This node's own "
        + str(sum(own_moves.values()))
        + " receipts account for "
        + ", ".join(f"{name} -{count}" for name, count in sorted(own_moves.items()))
        + (
            ", and the remainder is another writer's ("
            + ", ".join(
                f"{name} {value:+d}"
                for name, value in sorted(residual.items())
                if value
            )
            + "), which the overlap read below dates."
            if [v for v in residual.values() if v]
            else ", leaving nothing for another writer in either population."
        )
        if foreign_moves
        else "No other population moved."
    )
    body = [
        f"<h2 id='{node}'>Held search hits settled</h2>",
        f"<p id='{node}-summary'>The search and lineage steps referred "
        f"{cohort['candidate_count']} names; {cohort['worklist_count']} are still "
        f"sourceless on the live graph and form this node's work list "
        f"({source_counts}). Of those, {len(retired)} were retired in a holder's favour "
        f"through the signed supersede and {len(referrals)} remain referred to the lead. "
        f"The classifier read-back holds {verify['sourceless_total']} sourceless names now, "
        f"against {cohort['sourceless_total']} before this round's apply; cumulatively the "
        f"population has fallen from {prior_record.get('sourceless_before') if prior_record else cohort['sourceless_total']} "
        f"to {verify['sourceless_total']} across the "
        f"{len(prior_retired) + len(retired)} names this node has retired, with nothing "
        f"entering it. Spend USD {answers['spend_usd']:.6f}.</p>",
        f"<p id='{node}-suspect'>{len(suspect)} name(s) carry a suspect holder binding, "
        "where a holder's unit differs from the unit its held DD path declares. Every "
        f"holder in this cohort carries a held DD path and a resolved DD unit "
        f"({cohort_check_payload.get('holders_with_held_path')} of "
        f"{cohort_check_payload.get('holders')}), so this count is a result the comparison "
        "reached rather than one an empty field made unfalsifiable; the derivation check "
        "below also runs the comparison on a planted mismatch. A flagged pair changes "
        "nothing about the holder.</p>",
        f"<p id='{node}-foreign'>{escape(foreign_note)}</p>",
        f"<figure id='{node}-fig'>"
        f"<img src='/imas-codex/figures/sn-integrity-backlog/{node}/"
        f"{node}.svg' alt='Sourceless population before and after this node&apos;s signed "
        f"supersedes, and the work list split into settled, live and exhausted referrals.'>"
        f"<figcaption>Sourceless names by population before and after this node's "
        f"{len(retired)} signed supersedes (left), and the {cohort['worklist_count']}-name work "
        f"list split into {len(retired)} settled, "
        f"{totals['this_node_referrals_live']} still-live and "
        f"{totals['this_node_referrals_exhausted']} exhausted referrals (right). "
        f"Only the dropped-edge and never-bound populations moved; the other three are unchanged."
        f"</figcaption></figure>",
        f"<h3 id='{node}-floors'>Floors</h3>",
        f"<p>Coordinator's ruling for every retirement: the chosen option's probability at "
        f"least <code>{PROBABILITY_FLOOR}</code> and Jev's self-reported confidence at least "
        f"<code>{CONFIDENCE_FLOOR}</code>.</p>",
        f"<h3 id='{node}-retired'>Retired in the holder's favour ({len(retired)})</h3>",
        f"<table id='{node}-retired-t'><thead><tr><th>name</th><th>holder(s)</th>"
        "<th>Jev</th><th>p</th><th>conf</th><th>action</th><th>held DD path</th></tr></thead><tbody>",
        *[row_html(r) for r in retired],
        "</tbody></table>",
        f"<h3 id='{node}-referrals'>Referred to the lead ({len(referrals)})</h3>",
        f"<table id='{node}-ref-t'><thead><tr><th>name</th><th>holder(s)</th><th>Jev</th>"
        "<th>p</th><th>conf</th><th>action</th><th>held DD path</th></tr></thead><tbody>",
        *[row_html(r) for r in referrals],
        "</tbody></table>",
        f"<h3 id='{node}-rounds'>Rounds</h3>",
        f"<pre id='{node}-rounds-pre'>{escape(json.dumps(record['rounds'], indent=1))}</pre>",
        f"<h3 id='{node}-cohort'>Derivation check</h3>",
        f"<p id='{node}-cohort-note'>The derivation is read back and checked: every "
        "lineage-reason holder is attached to its name, every holder carries the DD path "
        "it holds, and every held path resolves to a DD unit — the two ways a held hit "
        "could go missing, each of which leaves the suspect-binding check below with "
        "nothing to compare. The same comparison is then run on a planted mismatch so it "
        "is shown able to fire.</p>",
        f"<pre id='{node}-cohort-pre'>{escape(json.dumps({'command': cohort_gate.get('header'), 'exit_code': cohort_gate.get('exit_code'), 'payload': cohort_check_payload}, indent=1))}</pre>",
        f"<h3 id='{node}-suspect-table'>Suspect holder bindings ({len(suspect)})</h3>",
        f"<pre id='{node}-suspect-pre'>{escape(json.dumps([{'name': r['id'], 'suspect_holders': r['suspect_holders'], 'holders': r['holders']} for r in suspect], indent=1))}</pre>",
        f"<h3 id='{node}-lead-total'>Running lead-referral total (§38)</h3>",
        f"<pre id='{node}-lead-pre'>{escape(json.dumps(totals, indent=1))}</pre>",
        f"<h3 id='{node}-authority'>Signed authority</h3>",
        f"<pre id='{node}-authority-pre'>{escape(json.dumps(record['authority'], indent=1))}</pre>",
        f"<h3 id='{node}-preview'>Preview and receipt</h3>",
        f"<pre id='{node}-preview-pre'>{escape(json.dumps({'preview': preview, 'receipt': apply_receipt, 'quiet_gate': apply_receipt.get('quiet_gate')}, indent=1))}</pre>",
        f"<h3 id='{node}-quiet'>Quiet-graph probe (owner instrument)</h3>",
        f"<p id='{node}-quiet-note'>The apply path calls the repository's own probe in "
        "<code>imas_codex.graph.offsite</code> immediately before the signed write and refuses "
        "when the graph is busy. The refusal is shown below against an injected busy reading, "
        "so the guard is demonstrated to fire rather than merely present.</p>",
        f"<pre id='{node}-quiet-pre'>{escape(json.dumps(quiet, indent=1))}</pre>",
        f"<pre id='{node}-quiet-refuse-pre'>{escape(json.dumps({'header': quiet_refuse['header'], 'exit_code': quiet_refuse['exit_code'], 'payload': quiet_refuse['payload']}, indent=1))}</pre>",
        f"<h3 id='{node}-overlap'>Apply-window overlap (after the fact)</h3>",
        f"<pre id='{node}-overlap-pre'>{escape(json.dumps(overlap, indent=1))}</pre>",
        f"<h3 id='{node}-guard'>Source-stays-held guard</h3>",
        f"<pre id='{node}-guard-pre'>{escape(json.dumps(guard, indent=1))}</pre>",
        f"<h3 id='{node}-verify'>Classifier read-back (raise on mismatch)</h3>",
        f"<pre id='{node}-verify-pre'>{escape(json.dumps({'command': verify_log['header'], 'exit_code': verify_log['exit_code'], 'payload': verify}, indent=1))}</pre>",
        f"<h3 id='{node}-falsify'>Read-back shown able to fail</h3>",
        f"<pre id='{node}-falsify-pre'>{escape(json.dumps({'command': falsify_log['header'], 'exit_code': falsify_log['exit_code'], 'payload': falsify}, indent=1))}</pre>",
        f"<h3 id='{node}-negcontrol'>Negative control (planted different-quantity pair)</h3>",
        f"<pre id='{node}-negcontrol-pre'>{escape(json.dumps(negcontrol, indent=1))}</pre>",
    ]
    (HERE / f"{node}.html").write_text("\n".join(body) + "\n")
    return {"retired": len(retired), "referrals": len(referrals)}


def _stubbed_busy_probe(**_):
    """A busy reading, so ``quiet-refuse`` can show the gate refusing."""
    from imas_codex.graph.offsite import GraphActivity

    return GraphActivity(
        window_seconds=6.0,
        transactions_started=3,
        open_queries=["MATCH (n) RETURN count(n)"],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        choices=[
            "read",
            "judge",
            "authority",
            "preview",
            "apply",
            "readback",
            "negcontrol",
            "quiet",
            "quiet-refuse",
            "overlap",
            "guard",
            "verify",
            "cohort-check",
            "falsify",
            "record",
        ],
    )
    parser.add_argument("--digest")
    parser.add_argument(
        "--prior-record",
        help="an earlier round's record, whose counts anchor the cumulative read-back",
    )
    parser.add_argument(
        "--cohort-file",
        help="a cohort file to check instead of this run's (the pre-repair cohort, for a control)",
    )
    parser.add_argument("--run-dir", help="run directory holding the derived artifacts")
    parser.add_argument("--run-id", help="the crew run id recorded on new rows")
    args = parser.parse_args()
    _resolve_runtime(args)
    if args.mode == "read":
        out = read()
        print(json.dumps({k: v for k, v in out.items() if k != "cohort"}, indent=2))
    elif args.mode == "judge":
        out = judge()
        print(
            json.dumps(
                {
                    "spend_usd": out["spend_usd"],
                    "judged_count": out["judged_count"],
                },
                indent=2,
            )
        )
    elif args.mode == "authority":
        print(json.dumps(build_authority(), indent=2))
    elif args.mode == "preview":
        print(json.dumps(_signed_call(False, None), indent=2, default=str)[:6000])
    elif args.mode == "apply":
        assert args.digest
        print(json.dumps(_signed_call(True, args.digest), indent=2, default=str)[:6000])
    elif args.mode == "readback":
        print(json.dumps(readback(), indent=2, default=str))
    elif args.mode == "verify":
        before, prior = _prior_baseline(args.prior_record)
        result = readback(before=before, raise_on_mismatch=True, prior=prior)
        print(json.dumps(result, indent=2, default=str))
    elif args.mode == "cohort-check":
        print(
            json.dumps(
                cohort_check(Path(args.cohort_file) if args.cohort_file else None),
                indent=2,
                default=str,
            )
        )
    elif args.mode == "falsify":
        before, prior = _prior_baseline(args.prior_record)
        before["sourceless_total"] += 1
        try:
            readback(before=before, raise_on_mismatch=True, prior=prior)
        except SystemExit as exc:
            print(
                json.dumps({"check_failed_as_expected": True, "detail": str(exc)[:300]})
            )
        else:
            print(json.dumps({"check_failed_as_expected": False}))
            raise SystemExit(1)
    elif args.mode == "record":
        print(json.dumps(compose(), indent=2))
    elif args.mode == "negcontrol":
        print(json.dumps(negcontrol(), indent=2, default=str))
    elif args.mode == "quiet":
        print(json.dumps(quiet_graph_check(), indent=2, default=str))
    elif args.mode == "quiet-refuse":
        try:
            _quiet_gate(probe=_stubbed_busy_probe)
        except SystemExit as exc:
            print(json.dumps({"refused_as_expected": True, "detail": str(exc)}))
        else:
            print(json.dumps({"refused_as_expected": False}))
            raise SystemExit(1)
    elif args.mode == "overlap":
        print(json.dumps(overlap_check(), indent=2, default=str))
    elif args.mode == "guard":
        with GraphClient() as gc:
            print(json.dumps(shared_source_path_guard(gc), indent=2, default=str))


if __name__ == "__main__":
    main()
