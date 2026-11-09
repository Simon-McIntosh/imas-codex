#!/usr/bin/env python3
"""Classify every unaccepted live DD quantity source and every exhausted name.

Read-only: graph reads only, no LLM call, no graph write.

Two populations are measured against the live graph and each item is assigned
exactly one class, so the class sums reconcile to the live reads:

* the live DD ``StandardNameSource`` rows with no accepted name, and
* the ``StandardName`` rows at ``name_stage = 'exhausted'``.

Classes follow how the pipeline is blocked, each naming the ``sn run`` step
that advances it: awaiting review, scored below the acceptance floor,
documentation missing, refused by a writer, attempt-capped, never claimed,
vocabulary gap.

Usage::

    PYTHONPATH=. uv run --no-sync python <this-file> --json <out.json>
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import UTC, datetime
from html import escape
from pathlib import Path
from typing import Any

from imas_codex.graph.client import GraphClient
from imas_codex.standard_names.defaults import DEFAULT_MIN_SCORE
from imas_codex.standard_names.graph_store.graph_cost_and_claims import (
    _MAX_COMPOSE_CLAIM_ATTEMPTS as COMPOSE_CLAIM_CAP,
)

HERE = Path(__file__).resolve().parent

#: Statuses that mean the source is live work. ``stale``, ``skipped`` and
#: ``not_physical_quantity`` are excluded because the extractor withdrew them
#: from the nameable population.
LIVE_STATUSES = ("composed", "attached", "failed", "extracted", "vocab_gap")

CLASSES = (
    "awaiting review",
    "scored below the acceptance floor",
    "documentation missing",
    "refused by a writer",
    "attempt-capped",
    "never claimed",
    "vocabulary gap",
)

#: The step each class names, and the scope it runs over.
CLASS_STEPS = {
    "awaiting review": (
        "review_name",
        "the drafted cohort, scoped by source_id",
    ),
    "scored below the acceptance floor": (
        "refine_name (or sn rescore where the chain never reached a verdict)",
        "the below-floor cohort, scoped by source_id",
    ),
    "documentation missing": (
        "generate_docs then review_docs",
        "the name cohort whose name passed review without accepted docs",
    ),
    "refused by a writer": (
        "compose (the candidate was refused at write time)",
        "the refused cohort, scoped by source_id",
    ),
    "attempt-capped": (
        "one signed reset with receipts (sn reset), then compose",
        "the parked cohort, scoped by source_id",
    ),
    "never claimed": (
        "compose",
        "the extracted cohort with no claim, scoped by source_id",
    ),
    "vocabulary gap": (
        "vocabulary release, then compose",
        "the vocabulary-blocked cohort, scoped by source_id",
    ),
}

STAGE_RANK = {"reviewed": 0, "drafted": 1, "exhausted": 2, "superseded": 3}


def read(gc: GraphClient, query: str, *, cap: int, **params: Any) -> list[dict]:
    rows = [dict(row) for row in gc.query(query, **params)]
    if len(rows) > cap:
        raise RuntimeError(f"graph result exceeded caller bound: {len(rows)} > {cap}")
    return rows


def source_rows(gc: GraphClient) -> dict[str, dict]:
    """One record per live DD source, with every binding it carries."""
    rows = read(
        gc,
        """
        MATCH (s:StandardNameSource {source_type: 'dd'})
        WHERE s.status IN $live
        OPTIONAL MATCH (s)-[:PRODUCED_NAME]->(sn:StandardName)
        OPTIONAL MATCH (ns:StandardName {id: s.produced_sn_id})
        RETURN s.id AS sid, s.source_id AS path, s.status AS status,
               coalesce(s.attempt_count, 0) AS attempts,
               s.last_error AS last_error,
               trim(coalesce(s.description, '')) = '' AS no_source_description,
               s.produced_sn_id AS produced_id,
               sn.id AS bound_id, sn.name_stage AS bound_stage,
               sn.review_mean_score AS bound_score,
               ns.id AS produced_stage_id, ns.name_stage AS produced_stage,
               ns.review_mean_score AS produced_score
        """,
        cap=20000,
        live=list(LIVE_STATUSES),
    )
    by_source: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_source[row["sid"]].append(row)
    return {
        sid: records[0] | {"bindings": records} for sid, records in by_source.items()
    }


def pick_binding(records: list[dict]) -> dict | None:
    """The binding that decides the source's class.

    ``accepted`` bindings are handled by the caller. Among the rest, a
    reviewed name outranks a drafted one, which outranks an exhausted one;
    ties break on id so the pick is deterministic across runs.
    """
    candidates = [
        r for r in records if r["bound_id"] and r["bound_stage"] != "accepted"
    ]
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda r: (STAGE_RANK.get(r["bound_stage"], 9), r["bound_id"]),
    )


def reason_family(error: str | None) -> str:
    text = (error or "").strip()
    if not text:
        return "none"
    lowered = text.lower()
    if "vocabulary" in lowered:
        return "vocabulary"
    if "binding collision" in lowered:
        return "binding_collision"
    if "attachment rejected" in lowered:
        return "attachment_rejected"
    if "pre-validation failed" in lowered:
        return "pre_validation"
    if "refus" in lowered:
        return "refused"
    if "claim-attempt cap" in lowered:
        return "cap_claim"
    return "other"


def classify_source(record: dict) -> str:
    """Assign one class to a source with no accepted name.

    A binding at drafted or reviewed decides first, because the name exists
    and only needs its remaining step. An exhausted binding decides next by
    what its score records. A source with no binding is decided by its own
    recorded state, and a recorded vocabulary cause outranks an empty DD
    description because the vocabulary gap is the step that can be taken.
    """
    binding = pick_binding(record["bindings"])
    family = reason_family(record["last_error"])
    vocabulary = record["status"] == "vocab_gap" or family == "vocabulary"

    if binding is not None:
        stage = binding["bound_stage"]
        score = binding["bound_score"]
        if stage in ("drafted", "reviewed"):
            if score is None or score >= DEFAULT_MIN_SCORE:
                return "awaiting review"
            return "scored below the acceptance floor"
        if stage == "superseded":
            return "refused by a writer"
        if stage == "exhausted":
            if vocabulary:
                return "vocabulary gap"
            if score is None:
                # Rotations were spent without a review verdict being
                # recorded, so the outstanding operation is the review.
                return "awaiting review"
            if score >= DEFAULT_MIN_SCORE:
                return "refused by a writer"
            return "scored below the acceptance floor"
        raise RuntimeError(f"unclassified binding stage {stage!r} for {record['sid']}")

    if vocabulary:
        return "vocabulary gap"
    if record["no_source_description"]:
        return "documentation missing"
    if record["status"] == "extracted":
        return "never claimed"
    if record["status"] == "failed":
        if family in (
            "binding_collision",
            "attachment_rejected",
            "pre_validation",
            "refused",
        ):
            return "refused by a writer"
        # A terminal failed row with a spent compose budget: the step is the
        # signed reset, whether or not the recorded count reached the cap.
        return "attempt-capped"
    if record["attempts"] >= COMPOSE_CLAIM_CAP:
        return "attempt-capped"
    if family in (
        "binding_collision",
        "attachment_rejected",
        "pre_validation",
        "refused",
    ):
        return "refused by a writer"
    return "never claimed"


def classify_exhausted(name: dict) -> str:
    """Assign one class to an exhausted catalog name."""
    score = name["score"]
    if score is None:
        # Rotations were spent without any review verdict recorded. The
        # outstanding operation is the review that never ran.
        return "awaiting review"
    if score >= DEFAULT_MIN_SCORE:
        if name["docs_stage"] != "accepted":
            return "documentation missing"
        # The name cleared the floor, so a guard ended it rather than the
        # score: either the validation quarantine or the rotation counter.
        return "refused by a writer"
    return "scored below the acceptance floor"


#: The full run's own receipt: USD 69.41506 spent to create 284 accepted
#: names in the run window, 4.1 accepted per USD (sib-full-run-failure-diagnosis).
FULL_RUN_SPEND_USD = 69.41506
FULL_RUN_ACCEPTED_IN_WINDOW = 284
#: Coordinator's corrected estimate on the current seats for the remaining
#: 1,308 eligible sources (USD 80-160), used as the low/high band.
REMAINING_SOURCES_ESTIMATED = 1308
REMAINING_COST_BAND_USD = (80.0, 160.0)


def cost_basis() -> dict[str, Any]:
    per_accepted = FULL_RUN_SPEND_USD / FULL_RUN_ACCEPTED_IN_WINDOW
    low, high = REMAINING_COST_BAND_USD
    return {
        "full_run_spend_usd": FULL_RUN_SPEND_USD,
        "full_run_accepted_in_window": FULL_RUN_ACCEPTED_IN_WINDOW,
        "accepted_per_usd": FULL_RUN_ACCEPTED_IN_WINDOW / FULL_RUN_SPEND_USD,
        "per_accepted_name_usd": round(per_accepted, 4),
        "corrected_estimate_band_usd": [low, high],
        "per_source_band_usd": [
            round(low / REMAINING_SOURCES_ESTIMATED, 4),
            round(high / REMAINING_SOURCES_ESTIMATED, 4),
        ],
        "cohort_rerun_measured": None,
        "cohort_rerun_note": (
            "No cohort rerun has run, so no measured cohort figure exists; "
            "the estimate uses the full run's measured cost per accepted name."
        ),
    }


def check_class_invariants(
    unaccepted: dict, source_class_by_id: dict, names: list
) -> list[str]:
    """Re-derive each class's defining predicate and refuse a mismatch.

    This is the census's internal check: a class whose members do not satisfy
    the rule that named it is a classification defect, not a result.
    """
    failures: list[str] = []
    for sid, rec in unaccepted.items():
        cls = source_class_by_id[sid]
        binding = pick_binding(rec["bindings"])
        family = reason_family(rec["last_error"])
        vocabulary = rec["status"] == "vocab_gap" or family == "vocabulary"
        if cls == "awaiting review":
            if binding is None:
                ok = False
            elif binding["bound_stage"] in ("drafted", "reviewed"):
                ok = (
                    binding["bound_score"] is None
                    or binding["bound_score"] >= DEFAULT_MIN_SCORE
                )
            else:
                ok = (
                    binding["bound_stage"] == "exhausted"
                    and binding["bound_score"] is None
                    and not vocabulary
                )
        elif cls == "scored below the acceptance floor":
            ok = (
                binding is not None
                and binding["bound_score"] is not None
                and binding["bound_score"] < DEFAULT_MIN_SCORE
            )
        elif cls == "refused by a writer":
            if binding is not None:
                ok = binding["bound_stage"] == "superseded" or (
                    binding["bound_stage"] == "exhausted"
                    and binding["bound_score"] is not None
                    and binding["bound_score"] >= DEFAULT_MIN_SCORE
                    and not vocabulary
                )
            else:
                ok = family in (
                    "binding_collision",
                    "attachment_rejected",
                    "pre_validation",
                    "refused",
                )
        elif cls == "vocabulary gap":
            ok = vocabulary
        elif cls == "documentation missing":
            ok = binding is None and not vocabulary and rec["no_source_description"]
        elif cls == "never claimed":
            ok = (
                binding is None
                and not vocabulary
                and not rec["no_source_description"]
                and rec["status"] != "failed"
            )
        elif cls == "attempt-capped":
            ok = (
                binding is None
                and not vocabulary
                and not rec["no_source_description"]
                and rec["status"] == "failed"
                and family
                not in (
                    "binding_collision",
                    "attachment_rejected",
                    "pre_validation",
                    "refused",
                )
            )
        else:
            ok = False
        if not ok:
            failures.append(f"{sid}: {cls} does not satisfy its own rule")
    if not source_class_by_id:
        failures.append("no sources classified")
    if not names:
        failures.append("no exhausted names read")
    return failures


def write_html(document: dict, path: Path) -> None:
    node = "sib-unaccepted-census"
    cost = document["cost_basis"]
    source_rows: list[str] = []
    name_rows: list[str] = []
    for cls in CLASSES:
        rec = document["source_classes"][cls]
        sample = (
            "; ".join(
                f"<code>{escape(item['source'])}</code> ({escape(item['status'])})"
                for item in rec["representative"][:4]
            )
            or "none"
        )
        source_rows.append(
            f'<tr id="{node}-src-{cls.replace(" ", "-")}">'
            f"<td>{escape(cls)}</td><td>{rec['count']}</td>"
            f"<td>{escape(rec['step'])}</td>"
            f"<td>USD {rec['expected_cost_usd_at_full_run_rate']:.2f}</td>"
            f"<td>{sample}</td></tr>"
        )
        nrec = document["name_classes"][cls]
        nsample = (
            "; ".join(
                f"<code>{escape(item['id'])}</code>"
                for item in nrec["representative"][:4]
            )
            or "none"
        )
        name_rows.append(
            f'<tr id="{node}-name-{cls.replace(" ", "-")}">'
            f"<td>{escape(cls)}</td><td>{nrec['count']}</td>"
            f"<td>{escape(nrec['step'])}</td>"
            f"<td>USD {nrec['expected_cost_usd_at_full_run_rate']:.2f}</td>"
            f"<td>{nsample}</td></tr>"
        )

    html = (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="reckon-type" content="evidence-fragment">'
        "<title>Unaccepted and exhausted census</title></head><body><main>"
        f'<h1 id="{node}-title">Unaccepted live quantity sources and exhausted names</h1>'
        f'<p id="{node}-read">Live read at {escape(document["observed_at"])}: '
        f"{document['live_sources']} live DD quantity sources, of which "
        f"{document['live_sources_accepted']} carry an accepted name and "
        f"{document['live_sources_unaccepted']} do not; the catalog holds "
        f"{document['exhausted_names']} exhausted names. Each unaccepted source "
        "and each exhausted name is assigned exactly one class, and the class "
        "sums reconcile to those totals. The acceptance floor is "
        f"{document['acceptance_floor']} and the compose claim cap is "
        f"{document['compose_claim_cap']} attempts.</p>"
        f'<p id="{node}-cost">Cost basis: the full run spent '
        f"USD {cost['full_run_spend_usd']:.5f} and created "
        f"{cost['full_run_accepted_in_window']} accepted names in its window, "
        f"so one accepted name cost USD {cost['per_accepted_name_usd']:.4f} "
        f"({cost['accepted_per_usd']:.1f} accepted per USD). Expected costs "
        "below use that measured rate; the band applies the coordinator's "
        f"corrected estimate of USD {cost['corrected_estimate_band_usd'][0]:.0f}"
        f"&ndash;{cost['corrected_estimate_band_usd'][1]:.0f} over "
        f"{REMAINING_SOURCES_ESTIMATED} eligible sources "
        f"(USD {cost['per_source_band_usd'][0]:.4f}&ndash;"
        f"{cost['per_source_band_usd'][1]:.4f} per source). "
        f"{escape(cost['cohort_rerun_note'])}</p>"
        '<h2 id="{node}-sources">Unaccepted live sources</h2>'.replace("{node}", node)
        + "<table><thead><tr><th>Class</th><th>Count</th><th>Advancing step</th>"
        "<th>At full-run rate</th><th>Representative sources</th></tr></thead><tbody>"
        + "".join(source_rows)
        + "</tbody></table>"
        f'<h2 id="{node}-names">Exhausted names</h2>'
        "<table><thead><tr><th>Class</th><th>Count</th><th>Advancing step</th>"
        "<th>At full-run rate</th><th>Representative names</th></tr></thead><tbody>"
        + "".join(name_rows)
        + '</tbody></table><h2 id="{node}-method">How a class is decided</h2>'.replace(
            "{node}", node
        )
        + f'<p id="{node}-method-text">A source with an accepted binding is '
        "not in the population. Otherwise its class comes from its binding and "
        "its own recorded state, in this order: a bound name at drafted or "
        "reviewed is awaiting review when it has no score or a score at or above "
        "the floor, and scored below the floor otherwise; a bound exhausted name "
        "is scored below the floor when its score is below it; a bound "
        "superseded name, and a name that cleared the floor yet ended exhausted, "
        "is refused by a writer. With no binding, an empty DD description is "
        "documentation missing, a vocabulary reason is a vocabulary gap, an "
        "unclaimed extracted row is never claimed, a row at the compose claim "
        "cap is attempt-capped, and a recorded binding-collision, "
        "attachment-rejection or pre-validation refusal is refused by a writer. "
        "For exhausted names, a missing score is awaiting review, a score below "
        "the floor is scored below the floor, and a score at or above it is "
        "documentation missing when its docs are not accepted and otherwise "
        "refused by a writer.</p>"
        '<p id="{node}-spend">Graph reads only; spend on this census was USD 0.</p>'
        "</main></body></html>".replace("{node}", node)
    )
    path.write_text(html)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--json", type=Path, default=HERE / "sib-unaccepted-census.json"
    )
    parser.add_argument(
        "--html", type=Path, default=HERE / "sib-unaccepted-census.html"
    )
    args = parser.parse_args()

    with GraphClient() as gc:
        sources = source_rows(gc)
        names = read(
            gc,
            """
            MATCH (n:StandardName {name_stage: 'exhausted'})
            RETURN n.id AS id, n.review_mean_score AS score,
                   coalesce(n.docs_stage, 'none') AS docs_stage,
                   coalesce(n.validation_status, 'none') AS validation_status
            ORDER BY n.id
            """,
            cap=2000,
        )
        costs = cost_basis()

    accepted = [
        sid
        for sid, rec in sources.items()
        if any(
            b["bound_stage"] == "accepted" or b["produced_stage"] == "accepted"
            for b in rec["bindings"]
        )
    ]
    accepted_set = set(accepted)
    unaccepted = {sid: rec for sid, rec in sources.items() if sid not in accepted_set}

    source_classes = Counter()
    source_class_by_id: dict[str, str] = {}
    per_class_sources: dict[str, list[dict]] = defaultdict(list)
    for sid, rec in unaccepted.items():
        cls = classify_source(rec)
        source_classes[cls] += 1
        source_class_by_id[sid] = cls
        per_class_sources[cls].append(
            {
                "source": rec["path"],
                "status": rec["status"],
                "attempts": rec["attempts"],
                "last_error": (rec["last_error"] or "")[:200],
            }
        )

    name_classes = Counter()
    per_class_names: dict[str, list[dict]] = defaultdict(list)
    for name in names:
        cls = classify_exhausted(name)
        name_classes[cls] += 1
        per_class_names[cls].append(name)

    for cls in source_classes:
        if cls not in CLASSES:
            raise RuntimeError(f"source class outside the declared set: {cls!r}")
    for cls in name_classes:
        if cls not in CLASSES:
            raise RuntimeError(f"name class outside the declared set: {cls!r}")
    if sum(source_classes.values()) != len(unaccepted):
        raise RuntimeError("source class sums do not reconcile to the live read")
    if sum(name_classes.values()) != len(names):
        raise RuntimeError("name class sums do not reconcile to the live read")
    failures = check_class_invariants(unaccepted, source_class_by_id, names)
    if failures:
        raise RuntimeError(
            f"{len(failures)} classification invariant(s) failed; first: {failures[0]}"
        )

    per_name_cost = costs["per_accepted_name_usd"]
    low_rate, high_rate = costs["per_source_band_usd"]

    def class_record(cls: str, counts: Counter, samples: dict) -> dict:
        count = counts.get(cls, 0)
        return {
            "count": count,
            "step": CLASS_STEPS[cls][0],
            "scope": CLASS_STEPS[cls][1],
            "expected_cost_usd_at_full_run_rate": round(count * per_name_cost, 2),
            "expected_cost_usd_band": [
                round(count * low_rate, 2),
                round(count * high_rate, 2),
            ],
            "representative": samples.get(cls, [])[:12],
        }

    document = {
        "observed_at": datetime.now(UTC).isoformat(),
        "acceptance_floor": DEFAULT_MIN_SCORE,
        "compose_claim_cap": COMPOSE_CLAIM_CAP,
        "live_sources": len(sources),
        "live_sources_accepted": len(accepted),
        "live_sources_unaccepted": len(unaccepted),
        "exhausted_names": len(names),
        "cost_basis": costs,
        "reconciliation": {
            "sources": "accepted + unaccepted == live_sources",
            "names": "sum(name_classes) == exhausted_names",
            "source_class_sum": sum(source_classes.values()),
            "name_class_sum": sum(name_classes.values()),
        },
        "source_classes": {
            cls: class_record(cls, source_classes, per_class_sources) for cls in CLASSES
        },
        "name_classes": {
            cls: class_record(cls, name_classes, per_class_names) for cls in CLASSES
        },
    }
    args.json.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    write_html(document, args.html)
    print(
        json.dumps(
            {
                "live_sources": document["live_sources"],
                "accepted": document["live_sources_accepted"],
                "unaccepted": document["live_sources_unaccepted"],
                "exhausted_names": document["exhausted_names"],
                "source_classes": dict(source_classes),
                "name_classes": dict(name_classes),
                "per_name_cost": per_name_cost,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
