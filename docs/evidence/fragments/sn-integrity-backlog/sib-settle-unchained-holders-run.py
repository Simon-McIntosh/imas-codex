"""Settle each unchained-holder name against the holder of its own DD source.

A name in the ``unchained_holder`` population recorded a DD source that another
live name now holds, with no ``REFINED_FROM``, ``HAS_SUCCESSOR`` or change edge
linking the two. Per name the question is whether the holder carries the same
quantity. The same quantity makes this name a duplicate, retired in the
holder's favour through the signed supersede with a successor link; a
different quantity leaves both in place and refers the pair to the lead.

The population is re-derived from the live graph with
``classify_sourceless_standard_names`` (the section's own read-back
instrument). Each pair is put to the ``sn-judge`` seat as one Choice through
``decide_batch``, and a supersede is applied only when the chosen option's
probability is at least ``PROBABILITY_FLOOR`` and the self-reported confidence
is at least ``CONFIDENCE_FLOOR``.

Every graph write goes through ``apply_signed_manifest``: the authority is
built by ``build_repair_authority``, previewed, and applied under the preview's
digest on a quiet graph.

Modes: read | judge | authority | preview | apply | readback | negcontrol.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
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
# signed authority, the run driver, and the composed record.
RUN_DIR = Path(
    os.environ.get(
        "SIB_RUN_DIR",
        "/home/ITER/mcintos/.config/reckon/crew/runs/"
        "r-20261019T021901634768-sib-settle-unchained-holders",
    )
)
COHORT_PATH = RUN_DIR / "sib-settle-unchained-holders-cohort.json"
ANSWERS_PATH = RUN_DIR / "sib-settle-unchained-holders-answers.json"
AUTHORITY_PATH = HERE / "sib-settle-unchained-holders-authority.json"

# Coordinator ruling: the floors every binding and retirement under this
# section clears, against the chosen option's probability and the model's own
# reported confidence.
PROBABILITY_FLOOR = 0.8
CONFIDENCE_FLOOR = 0.5

SAME = "same quantity"
DIFFERENT = "different quantity"
CANNOT_TELL = "cannot tell"
OPTIONS = (SAME, DIFFERENT, CANNOT_TELL)

QUESTION_NAME = "quantity_verdict"

SERVICE = "standard-names"
COST_STEP = "sn_judge"

SELECTION = {
    "id": "artifact-rows",
    "mode": "exact_complete_signed_cohort",
    "predicate": "artifact-rows",
}
RETIRE_OPERATION = "supersede_unchained_holder"
RETIRE_OPERATION_ID = "retire-unchained-holder-in-favour-of-its-source-holder"
REASON = (
    "the name and the holder of its recorded DD source carry the same "
    "quantity; retire the duplicate in the holder's favour through the signed "
    "supersede with a successor link"
)
SUPERSEDE_OPERATION = "link_signed_successor"

RUN_ID = os.environ.get(
    "SIB_RUN_ID", "r-20261019T021901634768-sib-settle-unchained-holders"
)


def _question() -> dict:
    return {
        QUESTION_NAME: {
            "type": "choice",
            "instructions": (
                "Two IMAS standard names exist. The FIRST name recorded the DD "
                "path shown. The SECOND name is a live name that currently "
                "holds that same DD path. Decide whether the two names denote "
                "the same physical quantity, so that the FIRST is a duplicate "
                "of the SECOND, or two different quantities that a single DD "
                "path cannot both mean."
            ),
            "criteria": {
                SAME: (
                    "The two names denote the same physical quantity: same "
                    "subject, same base quantity, same coordinate and region. "
                    "The first is a duplicate of the second."
                ),
                DIFFERENT: (
                    "The two names denote different physical quantities — a "
                    "different subject, base, coordinate or region — so one "
                    "shared DD path cannot be the source of both."
                ),
                CANNOT_TELL: (
                    "The descriptions and documentation do not settle whether "
                    "the two names denote the same quantity."
                ),
            },
        }
    }


def _name_ids(cohort: list[dict]) -> list[str]:
    return [row["id"] for row in cohort]


def read() -> dict:
    """Re-derive the population and enrich each pair with both sides' text."""
    with GraphClient() as gc:
        uri = gc.uri
        classified = classify_sourceless_standard_names(gc)
        cohort = [row for row in classified if row["population"] == "unchained_holder"]
        counts: dict[str, int] = defaultdict(int)
        for row in classified:
            counts[row["population"]] += 1
        details = []
        for row in cohort:
            name_id = row["id"]
            holder_ids = row["unchained_holders"]
            sources = row["recorded_source_ids"]
            name_node = gc.query(
                "MATCH (sn:StandardName {id:$id}) RETURN properties(sn) AS p",
                id=name_id,
            )
            props = name_node[0]["p"] if name_node else {}
            holders = gc.query(
                "UNWIND $ids AS id MATCH (sn:StandardName {id:id}) "
                "RETURN sn.id AS id, properties(sn) AS p",
                ids=holder_ids,
            )
            shared = gc.query(
                "UNWIND $ids AS sid "
                "MATCH (src:StandardNameSource {id:sid}) "
                "OPTIONAL MATCH (src)-[:PRODUCED_NAME]->(h:StandardName) "
                "RETURN src.id AS id, src.source_id AS source_id, "
                "src.produced_sn_id AS mirror, "
                "[x IN collect(DISTINCT h.id) WHERE x IS NOT NULL] AS holders",
                ids=sources,
            )
            shared_paths = sorted(
                {str(s["source_id"]) for s in shared if s["source_id"]}
            )
            dd_docs = gc.query(
                "UNWIND $paths AS p "
                "MATCH (dd:IMASNode {id:p}) "
                "RETURN dd.id AS id, dd.documentation AS documentation, "
                "dd.units AS units",
                paths=shared_paths,
            )
            details.append(
                {
                    "id": name_id,
                    "name_stage": row["name_stage"],
                    "status": row["status"],
                    "origin": row["origin"],
                    "description": props.get("description"),
                    "documentation": props.get("documentation"),
                    "unit": props.get("unit"),
                    "physics_domain": props.get("physics_domain"),
                    "holders": [
                        {
                            "id": h["id"],
                            "name_stage": h["p"].get("name_stage"),
                            "status": h["p"].get("status"),
                            "description": h["p"].get("description"),
                            "documentation": h["p"].get("documentation"),
                            "unit": h["p"].get("unit"),
                            "physics_domain": h["p"].get("physics_domain"),
                        }
                        for h in holders
                    ],
                    "recorded_source_ids": sources,
                    "shared_paths": shared_paths,
                    "shared_path_docs": dd_docs,
                }
            )
        out = {
            "uri": uri,
            "counts": dict(counts),
            "unchained_holder_count": len(cohort),
            "cohort": details,
        }
    COHORT_PATH.write_text(json.dumps(out, indent=2, default=str) + "\n")
    return out


def _state(detail: dict) -> dict:
    holder = detail["holders"][0] if detail["holders"] else {}
    return {
        "recorded_name": {
            "id": detail["id"],
            "description": detail["description"],
            "documentation": detail["documentation"],
            "unit": detail["unit"],
            "physics_domain": detail["physics_domain"],
        },
        "holder_name": {
            "id": holder.get("id"),
            "description": holder.get("description"),
            "documentation": holder.get("documentation"),
            "unit": holder.get("unit"),
            "physics_domain": holder.get("physics_domain"),
            "name_stage": holder.get("name_stage"),
        },
        "shared_dd_path": {
            "paths": detail["shared_paths"],
            "docs": detail["shared_path_docs"],
        },
    }


def judge() -> dict:
    from imas_codex.discovery.base.judgment import decide_batch
    from imas_codex.settings import get_model

    cohort = json.loads(COHORT_PATH.read_text())["cohort"]
    model = get_model("sn-judge")

    async def _run() -> dict:
        results, spent = await decide_batch(
            [_state(d) for d in cohort],
            _question(),
            model=model,
            service=SERVICE,
            concurrency=4,
            budget_seconds=600,
            step=COST_STEP,
        )
        records = []
        for detail, result in zip(cohort, results, strict=True):
            answer = (result[0].get(QUESTION_NAME) or {}) if result else {}
            records.append(
                {
                    "id": detail["id"],
                    "holder": detail["holders"][0]["id"] if detail["holders"] else None,
                    "choice": answer.get("choice"),
                    "probabilities": answer.get("probabilities"),
                    "probability": (answer.get("probabilities") or {}).get(
                        answer.get("choice")
                    ),
                    "confidence": answer.get("confidence"),
                    "answered": result is not None,
                    "shared_paths": detail["shared_paths"],
                    "description": detail["description"],
                    "holder_description": detail["holders"][0].get("description")
                    if detail["holders"]
                    else None,
                }
            )
        return {"model": model, "spend_usd": spent, "records": records}

    out = asyncio.run(_run())
    out["questions"] = _question()
    ANSWERS_PATH.write_text(json.dumps(out, indent=2, default=str) + "\n")
    return out


def _admitted(records: list[dict]) -> list[dict]:
    """The records cleared for a supersede by the ruling's floors."""
    out = []
    for rec in records:
        if rec["choice"] != SAME or not rec["holder"]:
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


def _signed_call(apply: bool, digest: str | None) -> dict:
    raw = AUTHORITY_PATH.read_bytes()
    authority = json.loads(raw)
    res = apply_signed_manifest(
        AUTHORITY_PATH,
        authority_file_sha256=hashlib.sha256(raw).hexdigest(),
        authority_payload_sha256=signed_payload_sha256(authority),
        reason=REASON,
        apply=apply,
        manifest_sha256=digest if apply else None,
        run_id=RUN_ID,
    )
    return {k: res[k] for k in res if k not in {"manifest", "admitted_actions"}}


def readback(before: dict | None = None, raise_on_mismatch: bool = False) -> dict:
    raw = AUTHORITY_PATH.read_bytes()
    file_sha = hashlib.sha256(raw).hexdigest()
    authority = json.loads(raw)
    targets = {row["identity"]["target_id"] for row in authority["rows"]}
    with GraphClient() as gc:
        classified = classify_sourceless_standard_names(gc)
        counts: dict[str, int] = defaultdict(int)
        for row in classified:
            counts[row["population"]] += 1
        receipts = gc.query(
            "MATCH (c:StandardNameChange {authority_file_sha256:$fs}) "
            "RETURN c.row_id AS row_id, c.operation AS op, "
            "c.manifest_sha256 AS digest",
            fs=file_sha,
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
        "unchained_holder_count": counts["unchained_holder"],
        "targets": sorted(targets),
        "receipt_names": len(per),
        "receipt_targets": sorted(receipt_targets),
        "targets_without_receipt": sorted(targets - receipt_targets),
        "receipt_operations": sorted({str(c["op"]) for c in receipts}),
        "digests": sorted({str(c["digest"]) for c in receipts}),
        "retired_stages": sorted({(r["stage"], r["status"]) for r in stages}),
    }
    if before is not None:
        retired = len(targets)
        expected = before["unchained_holder_count"] - retired
        out["before"] = before["counts"]
        out["expected_unchained_holder"] = expected
        moved = {
            k: (before["counts"].get(k), counts.get(k))
            for k in set(before["counts"]) | set(counts)
            if before["counts"].get(k) != counts.get(k) and k != "unchained_holder"
        }
        out["moved_other_populations"] = moved
        out["shortfall"] = sorted(targets - receipt_targets)
    if raise_on_mismatch:
        checks = {
            "unchained_holder_fell_by_retired": out["unchained_holder_count"]
            == (before["unchained_holder_count"] - len(targets)),
            "every_target_has_receipt": not (targets - receipt_targets),
            "no_other_population_moved": not out["moved_other_populations"],
            "all_targets_superseded": out["retired_stages"]
            == [("superseded", "superseded")],
        }
        if not all(checks.values()):
            raise SystemExit(f"read-back mismatch: {checks} :: {out}")
        out["checks"] = checks
    return out


def quiet_graph_check(gc) -> dict:
    """Record every fact that shows whether another graph writer is live."""
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
    try:
        transactions = [
            dict(r)
            for r in gc.query(
                "SHOW TRANSACTIONS YIELD transactionId, currentQuery, status "
                "RETURN transactionId, status, currentQuery LIMIT 50"
            )
        ]
    except Exception as exc:
        transactions = [{"error": f"{type(exc).__name__}: {exc}"}]
    foreign = [
        t
        for t in transactions
        if isinstance(t, dict)
        and str(t.get("currentQuery") or "").strip()
        and str(t.get("status") or "") == "running"
    ]
    return {
        "live_crew_runs": runs,
        "live_crew_run_count": len(runs),
        "runs_touched_within_1800s": recently_touched,
        "neo4j_jobs": neo4j_jobs,
        "active_transactions": transactions,
        "foreign_running_transactions": foreign,
        "quiet": not foreign,
    }


def shared_source_path_guard(gc) -> dict:
    """Prove each admitted supersede leaves the survivor holding the source.

    The predecessor recorded the DD path but binds no ``StandardNameSource``
    (that is why it is sourceless); the successor holds the binding. Retiring
    the predecessor therefore drops no source, and the shared path stays held
    by the successor. The check raises if any admitted predecessor carries a
    live source binding, or if the successor does not hold every path the
    predecessor's own record names.
    """
    answers = json.loads(ANSWERS_PATH.read_text())
    admitted = _admitted(answers["records"])
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
        recorded = set(rec.get("shared_paths") or [])
        succ_paths = {str(p).removeprefix("dd:") for p in (succ.get("paths") or [])}
        missing = sorted(p for p in recorded if p.removeprefix("dd:") not in succ_paths)
        pairs.append(
            {
                "predecessor": rec["id"],
                "successor": rec["holder"],
                "predecessor_binds_source": int(pred.get("producers") or 0),
                "recorded_paths": sorted(recorded),
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

    Two names that plainly denote different quantities (a neutron rate and an
    antenna voltage amplitude) are put to the same seat, and the run shows the
    answer is not ``same quantity`` and the pair is absent from the authority
    the floor builds.
    """
    from imas_codex.discovery.base.judgment import decide_batch
    from imas_codex.settings import get_model

    planted = {
        "recorded_name": {
            "id": "PLANTED-neutron_rate",
            "description": "Total number of neutrons produced per second.",
            "documentation": "The neutron emission rate of the plasma.",
            "unit": "s-1",
            "physics_domain": "neutron",
        },
        "holder_name": {
            "id": "PLANTED-voltage_amplitude",
            "description": "Amplitude of the voltage applied to an antenna.",
            "documentation": "Peak radio-frequency voltage on the antenna.",
            "unit": "V",
            "physics_domain": "heating",
        },
        "shared_dd_path": {"paths": ["PLANTED/shared"], "docs": []},
    }

    async def _run() -> dict:
        results, spent = await decide_batch(
            [planted],
            _question(),
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
    record = {
        "id": "PLANTED-neutron_rate",
        "holder": "PLANTED-voltage_amplitude",
        "choice": observed["choice"],
        "probability": (observed["probabilities"] or {}).get(observed["choice"]),
        "confidence": observed["confidence"],
        "answered": observed["choice"] is not None,
    }
    admitted = _admitted([record])
    refused = observed["choice"] != SAME and not admitted
    if not refused:
        raise SystemExit(f"planted different-quantity pair was not refused: {observed}")
    return {
        "planted": record,
        "admitted_by_floors": [r["id"] for r in admitted],
        "refused": refused,
    }


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def compose() -> dict:
    """Assemble the JSON record and HTML fragment from the run's artifacts."""
    from html import escape

    node = "sib-settle-unchained-holders"
    cohort = _load(COHORT_PATH)
    answers = _load(ANSWERS_PATH)
    preview = _load(RUN_DIR / "preview.log")
    apply_receipt = _load(RUN_DIR / "apply.log")
    quiet = _load(RUN_DIR / "quiet-before-apply.log")
    guard = _load(RUN_DIR / "guard.log")
    verify = _load(RUN_DIR / "verify.log")
    falsify = _load(RUN_DIR / "falsify.log")
    negcontrol = _load(RUN_DIR / "negcontrol.log")

    details = {row["id"]: row for row in cohort["cohort"]}
    admitted_ids = {r["id"] for r in _admitted(answers["records"])}

    per_name = []
    for rec in answers["records"]:
        det = details.get(rec["id"], {})
        holder = rec["holder"]
        action = "superseded" if rec["id"] in admitted_ids else "referred to lead"
        if action == "referred to lead":
            reason = (
                "Jev answered different quantity"
                if rec["choice"] in (DIFFERENT, CANNOT_TELL)
                else "below the probability/confidence floor"
            )
        else:
            reason = "same quantity, cleared both floors"
        per_name.append(
            {
                "id": rec["id"],
                "holder": holder,
                "name_stage": det.get("name_stage"),
                "description": det.get("description"),
                "holder_description": rec.get("holder_description"),
                "shared_paths": det.get("shared_paths"),
                "choice": rec["choice"],
                "probability": rec["probability"],
                "confidence": rec["confidence"],
                "answered": rec["answered"],
                "action": action,
                "reason": reason,
            }
        )

    referrals = [r for r in per_name if r["action"] == "referred to lead"]
    retired = [r for r in per_name if r["action"] == "superseded"]

    record = {
        "node": node,
        "plan": "sn-integrity-backlog",
        "section": "§38",
        "population": "unchained_holder",
        "floors": {"probability": PROBABILITY_FLOOR, "confidence": CONFIDENCE_FLOOR},
        "before": cohort["counts"],
        "after": verify["counts"],
        "retired_count": len(retired),
        "referral_count": len(referrals),
        "spend_usd": answers["spend_usd"],
        "judge_model": answers["model"],
        "authority": {
            "path": f"docs/evidence/fragments/sn-integrity-backlog/{node}-authority.json",
            "file_sha256": preview["authority_file_sha256"],
            "payload_sha256": preview["authority_payload_sha256"],
            "manifest_sha256": preview["manifest_sha256"],
            "operation_id": RETIRE_OPERATION_ID,
            "receipt_operation": SUPERSEDE_OPERATION,
        },
        "preview": preview,
        "receipt": apply_receipt,
        "quiet_graph": quiet,
        "source_held_guard": guard,
        "verification": verify,
        "falsification": falsify,
        "negative_control": negcontrol,
        "per_name": per_name,
        "referrals": referrals,
    }
    (HERE / f"{node}.json").write_text(json.dumps(record, indent=2, default=str) + "\n")

    def row_html(r: dict, cls: str) -> str:
        return (
            f"<tr class='{cls}'>"
            f"<td>{escape(str(r['id']))}</td>"
            f"<td>{escape(str(r['holder']))}</td>"
            f"<td>{escape(str(r['choice']))}</td>"
            f"<td>{r['probability']}</td><td>{r['confidence']}</td>"
            f"<td>{escape(r['action'])}</td>"
            f"<td>{escape(','.join(r['shared_paths'] or []))}</td></tr>"
        )

    body = [
        f"<h2 id='{node}'>Unchained holders settled</h2>",
        f"<p id='{node}-summary'>The classifier read-back holds {cohort['counts']['unchained_holder']} "
        f"unchained-holder names before this node and {verify['counts']['unchained_holder']} after: "
        f"exactly {len(retired)} retired through the signed supersede and {len(referrals)} referred to "
        f"the lead. No other population moved. Spend USD {answers['spend_usd']:.6f}.</p>",
        f"<h3 id='{node}-floors'>Floors</h3>",
        f"<p>Coordinator's ruling for every binding and retirement: the chosen option's probability at "
        f"least <code>{PROBABILITY_FLOOR}</code> and Jev's self-reported confidence at least "
        f"<code>{CONFIDENCE_FLOOR}</code>.</p>",
        f"<h3 id='{node}-retired'>Retired in the holder's favour ({len(retired)})</h3>",
        f"<table id='{node}-retired-t'><thead><tr><th>predecessor</th><th>successor (holder)</th>"
        "<th>Jev</th><th>p</th><th>conf</th><th>action</th><th>shared DD path</th></tr></thead><tbody>",
        *[row_html(r, "retired") for r in retired],
        "</tbody></table>",
        f"<h3 id='{node}-referrals'>Referred to the lead ({len(referrals)})</h3>",
        f"<table id='{node}-ref-t'><thead><tr><th>name</th><th>holder</th><th>Jev</th>"
        "<th>p</th><th>conf</th><th>action</th><th>shared DD path</th></tr></thead><tbody>",
        *[row_html(r, "referred") for r in referrals],
        "</tbody></table>",
        f"<h3 id='{node}-authority'>Signed authority</h3>",
        f"<pre id='{node}-authority-pre'>{escape(json.dumps(record['authority'], indent=1))}</pre>",
        f"<h3 id='{node}-preview'>Preview and receipt</h3>",
        f"<pre id='{node}-preview-pre'>{escape(json.dumps({'preview': preview, 'receipt': apply_receipt}, indent=1))}</pre>",
        f"<h3 id='{node}-quiet'>Quiet-graph check (before apply)</h3>",
        f"<pre id='{node}-quiet-pre'>{escape(json.dumps({k: quiet[k] for k in ('quiet', 'foreign_running_transactions', 'neo4j_jobs')}, indent=1))}</pre>",
        f"<h3 id='{node}-guard'>Source-stays-held guard</h3>",
        f"<pre id='{node}-guard-pre'>{escape(json.dumps(guard, indent=1))}</pre>",
        f"<h3 id='{node}-verify'>Classifier read-back (raise on mismatch)</h3>",
        f"<pre id='{node}-verify-pre'>{escape(json.dumps(verify, indent=1))}</pre>",
        f"<h3 id='{node}-falsify'>Read-back shown able to fail</h3>",
        f"<pre id='{node}-falsify-pre'>{escape(json.dumps(falsify, indent=1))}</pre>",
        f"<h3 id='{node}-negcontrol'>Negative control (planted different-quantity pair)</h3>",
        f"<pre id='{node}-negcontrol-pre'>{escape(json.dumps(negcontrol, indent=1))}</pre>",
    ]
    (HERE / f"{node}.html").write_text("\n".join(body) + "\n")
    return {"retired": len(retired), "referrals": len(referrals)}


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
            "guard",
            "verify",
            "falsify",
            "record",
        ],
    )
    parser.add_argument("--digest")
    args = parser.parse_args()
    if args.mode == "read":
        out = read()
        print(json.dumps({k: v for k, v in out.items() if k != "cohort"}, indent=2))
    elif args.mode == "judge":
        out = judge()
        print(json.dumps({"spend_usd": out["spend_usd"]}, indent=2))
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
        before = json.loads(COHORT_PATH.read_text())
        result = readback(before=before, raise_on_mismatch=True)
        print(json.dumps(result, indent=2, default=str))
    elif args.mode == "falsify":
        before = json.loads(COHORT_PATH.read_text())
        before["unchained_holder_count"] += 1
        try:
            readback(before=before, raise_on_mismatch=True)
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
        with GraphClient() as gc:
            print(json.dumps(quiet_graph_check(gc), indent=2, default=str))
    elif args.mode == "guard":
        with GraphClient() as gc:
            print(json.dumps(shared_source_path_guard(gc), indent=2, default=str))


if __name__ == "__main__":
    main()
