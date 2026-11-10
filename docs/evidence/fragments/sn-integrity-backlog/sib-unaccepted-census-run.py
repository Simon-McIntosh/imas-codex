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
from imas_codex.standard_names.graph_store.graph_source_bindings import (
    CAP_PARKED_DISPOSITIONS,
    classify_parked_source,
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
        OPTIONAL MATCH (n:IMASNode {id: s.source_id})
        RETURN s.id AS sid, s.source_id AS path, s.status AS status,
               coalesce(s.attempt_count, 0) AS attempts,
               s.last_error AS last_error,
               trim(coalesce(s.description, '')) = '' AS no_source_description,
               s.produced_sn_id AS produced_id,
               sn.id AS bound_id, sn.name_stage AS bound_stage,
               sn.review_mean_score AS bound_score,
               ns.id AS produced_stage_id, ns.name_stage AS produced_stage,
               ns.review_mean_score AS produced_score,
               n.lifecycle_status AS node_lifecycle,
               n.node_category AS node_category
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


#: The recorded causes that mean a writer refused the candidate rather than
#: the compose budget ran out.
REFUSAL_FAMILIES = frozenset(
    {"binding_collision", "attachment_rejected", "pre_validation", "refused"}
)


def source_rules(record: dict) -> dict[str, bool]:
    """Every class's defining predicate for one source, from its own state.

    One table decides both the assignment and whether the classes stay total
    and exclusive for the whole population. A drafted or reviewed binding is
    awaiting review unless its score is below the floor; an exhausted binding
    is decided by its score, except that a recorded vocabulary cause outranks
    it because the vocabulary gap is the step that can be taken.
    """
    binding = pick_binding(record["bindings"])
    family = reason_family(record["last_error"])
    stage = binding["bound_stage"] if binding else None
    score = binding["bound_score"] if binding else None
    vocabulary = record["status"] == "vocab_gap" or family == "vocabulary"
    status = record["status"]
    unbound = binding is None
    plain = not vocabulary and not record["no_source_description"]
    refused_reason = unbound and family in REFUSAL_FAMILIES
    below_floor = score is not None and score < DEFAULT_MIN_SCORE
    cleared_floor = score is not None and score >= DEFAULT_MIN_SCORE
    return {
        "awaiting review": (
            stage in ("drafted", "reviewed") and (score is None or cleared_floor)
        )
        # Rotations were spent without a review verdict recorded, so the
        # outstanding operation is the review that never ran.
        or (stage == "exhausted" and not vocabulary and score is None),
        "scored below the acceptance floor": below_floor
        and not (stage == "exhausted" and vocabulary),
        "documentation missing": unbound
        and not vocabulary
        and record["no_source_description"],
        "refused by a writer": stage == "superseded"
        or (stage == "exhausted" and not vocabulary and cleared_floor)
        or (refused_reason and plain),
        "attempt-capped": unbound
        and plain
        and (
            (status == "failed" and not refused_reason)
            or (
                status not in ("failed", "extracted")
                and record["attempts"] >= COMPOSE_CLAIM_CAP
            )
        ),
        "never claimed": unbound
        and plain
        and (
            status == "extracted"
            or (
                status not in ("failed", "extracted")
                and record["attempts"] < COMPOSE_CLAIM_CAP
                and not refused_reason
            )
        ),
        "vocabulary gap": vocabulary,
    }


def name_rules(name: dict) -> dict[str, bool]:
    """Every class's defining predicate for one exhausted catalog name."""
    score = name["score"]
    cleared = score is not None and score >= DEFAULT_MIN_SCORE
    return {
        # Rotations were spent without any review verdict recorded.
        "awaiting review": score is None,
        "scored below the acceptance floor": score is not None and not cleared,
        "documentation missing": cleared and name["docs_stage"] != "accepted",
        # The name cleared the floor, so a guard ended it rather than the
        # score: either the validation quarantine or the rotation counter.
        "refused by a writer": cleared and name["docs_stage"] == "accepted",
    }


def assign_class(rules: dict[str, bool], item: str) -> str:
    """The single class whose predicate holds, refusing any other answer.

    Totality and exclusivity are what make the class sums reconcile, so a rule
    change that leaves an item in no class or in two is a defect the census
    refuses rather than a distribution it reports.
    """
    hits = [cls for cls in CLASSES if rules.get(cls)]
    if len(hits) != 1:
        raise RuntimeError(
            f"{item}: {len(hits)} classes satisfied ({', '.join(hits) or 'none'})"
        )
    return hits[0]


def classify_source(record: dict) -> str:
    return assign_class(source_rules(record), f"source {record['path']}")


def classify_exhausted(name: dict) -> str:
    return assign_class(name_rules(name), f"exhausted name {name['id']}")


def parked_disposition(record: dict) -> str | None:
    """The product's own triage of a source that carries no binding.

    Reused from ``graph_source_bindings`` rather than re-derived, because that
    is the vocabulary the continuation run reads: it separates a spent compose
    budget from a structural blocker such as a coordinate path.
    """
    if pick_binding(record["bindings"]) is not None:
        return None
    disposition = classify_parked_source(
        {
            "produced": 1 if record["produced_stage_id"] else 0,
            "lifecycle_status": record["node_lifecycle"],
            "node_category": record["node_category"],
            "last_error": record["last_error"],
        },
        services=None,
    )
    if disposition not in CAP_PARKED_DISPOSITIONS:
        raise RuntimeError(f"{record['path']}: unknown disposition {disposition!r}")
    return disposition


#: The full run's own receipt: USD 69.41506 spent to create 284 accepted
#: names in the run window, 4.1 accepted per USD (sib-full-run-failure-diagnosis).
FULL_RUN_SPEND_USD = 69.41506
FULL_RUN_ACCEPTED_IN_WINDOW = 284
#: Coordinator's corrected estimate on the current seats for the remaining
#: 1,308 eligible sources (USD 80-160), used as the low/high band.
REMAINING_SOURCES_ESTIMATED = 1308
REMAINING_COST_BAND_USD = (80.0, 160.0)

#: The measured cohort rerun (sn-integrity-backlog section 51): a signed reset
#: of 143 sources then a scoped compose, which named 83 of the 143 for
#: USD 1.49 before a persist defect ended the batch.
COHORT_RERUN_SOURCES = 143
COHORT_RERUN_NAMED = 83
COHORT_RERUN_SPEND_USD = 1.49
COHORT_RERUN_JOB = 1308283


def cost_basis() -> dict[str, Any]:
    per_accepted = FULL_RUN_SPEND_USD / FULL_RUN_ACCEPTED_IN_WINDOW
    low, high = REMAINING_COST_BAND_USD
    per_cohort_source = COHORT_RERUN_SPEND_USD / COHORT_RERUN_SOURCES
    per_named_source = COHORT_RERUN_SPEND_USD / COHORT_RERUN_NAMED
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
        "cohort_rerun_measured": {
            "node": "sib-second-pass",
            "job": COHORT_RERUN_JOB,
            "sources_in_cohort": COHORT_RERUN_SOURCES,
            "sources_named": COHORT_RERUN_NAMED,
            "named_fraction": round(COHORT_RERUN_NAMED / COHORT_RERUN_SOURCES, 4),
            "spend_usd": COHORT_RERUN_SPEND_USD,
            "per_cohort_source_usd": round(per_cohort_source, 5),
            "per_named_source_usd": round(per_named_source, 5),
            "recorded_in": "docs/plans/sn-integrity-backlog.html section 51",
            "note": (
                "A scoped cohort rerun has run, so the per-source cost is measured "
                "rather than interpolated from the full run: USD 1.49 bought 83 "
                "named sources out of 143 attempted, i.e. USD 0.0104 per attempted "
                "source and USD 0.01795 per named source."
            ),
        },
    }


def check_document_assignments(
    document: dict,
    unaccepted: dict,
    names: list,
    source_class_by_id: dict,
    name_class_by_id: dict,
) -> None:
    """Refuse a census that does not record every item in exactly one class.

    Completeness is what the continuation run depends on: it scopes a cohort
    by the source paths this document records, so a sampled list would
    silently under-run the work and a double-assigned one would run it twice.
    """
    groups = document.get("source_assignments")
    if not isinstance(groups, dict) or set(groups) != set(CLASSES):
        raise RuntimeError("source_assignments is missing or not keyed by class")
    for cls in CLASSES:
        expected = Counter(
            rec["path"]
            for sid, rec in unaccepted.items()
            if source_class_by_id[sid] == cls
        )
        got = Counter(row["source"] for row in groups[cls])
        if got != expected:
            raise RuntimeError(
                f"{cls}: recorded paths do not match the class membership"
            )
        if sum(got.values()) != document["source_classes"][cls]["count"]:
            raise RuntimeError(f"{cls}: recorded paths do not match the class count")
    duplicated = {
        path for path, rows in document.get("duplicate_source_paths", {}).items()
    }
    seen = Counter(row["source"] for cls in CLASSES for row in groups[cls])
    if {path for path, count in seen.items() if count > 1} != duplicated:
        raise RuntimeError("a path carrying more than one live node is not disclosed")
    name_groups = document.get("name_assignments")
    if not isinstance(name_groups, dict) or set(name_groups) != set(CLASSES):
        raise RuntimeError("name_assignments is missing or not keyed by class")
    for cls in CLASSES:
        expected = {name["id"] for name in names if name_class_by_id[name["id"]] == cls}
        got = [row["id"] for row in name_groups[cls]]
        if len(got) != len(set(got)) or set(got) != expected:
            raise RuntimeError(f"{cls}: recorded names do not match the class")
        if len(got) != document["name_classes"][cls]["count"]:
            raise RuntimeError(f"{cls}: recorded names do not match the class count")
    if not unaccepted or not names:
        raise RuntimeError("the live read returned an empty population")


def check_cost_basis(document: dict) -> None:
    """Refuse a cost basis that drops or contradicts the measured cohort run."""
    measured = document["cost_basis"].get("cohort_rerun_measured")
    if not isinstance(measured, dict):
        raise RuntimeError("cost basis records no measured cohort rerun figure")
    recorded = (
        measured["spend_usd"],
        measured["sources_in_cohort"],
        measured["sources_named"],
    )
    if recorded != (COHORT_RERUN_SPEND_USD, COHORT_RERUN_SOURCES, COHORT_RERUN_NAMED):
        raise RuntimeError(f"measured cohort figures disagree with the run: {recorded}")
    expected = round(COHORT_RERUN_SPEND_USD / COHORT_RERUN_NAMED, 5)
    if measured.get("per_named_source_usd") != expected:
        raise RuntimeError(
            f"per-named-source cost {measured.get('per_named_source_usd')} != {expected}"
        )
    for cls in CLASSES:
        per_class = document["source_classes"][cls]["expected_cost_usd_measured_cohort"]
        if not isinstance(per_class, dict):
            raise RuntimeError(f"{cls}: no measured cohort cost recorded")


def class_table_rows(
    document: dict, key: str, node: str, prefix: str, cell: Any
) -> list[str]:
    """The body rows of one class table; both tables share this renderer."""
    rows: list[str] = []
    for cls in CLASSES:
        rec = document[key][cls]
        sample = "; ".join(cell(item) for item in rec["representative"][:4]) or "none"
        rows.append(
            f'<tr id="{node}-{prefix}-{cls.replace(" ", "-")}">'
            f"<td>{escape(cls)}</td><td>{rec['count']}</td>"
            f"<td>{escape(rec['step'])}</td>"
            f"<td>USD {rec['expected_cost_usd_at_full_run_rate']:.2f}</td>"
            f"<td>{sample}</td></tr>"
        )
    return rows


def write_html(document: dict, path: Path) -> None:
    node = "sib-unaccepted-census"
    cost = document["cost_basis"]
    cohort = cost["cohort_rerun_measured"]
    source_rows = class_table_rows(
        document,
        "source_classes",
        node,
        "src",
        lambda item: (
            f"<code>{escape(item['source'])}</code> ({escape(item['status'])})"
        ),
    )
    name_rows = class_table_rows(
        document,
        "name_classes",
        node,
        "name",
        lambda item: f"<code>{escape(item['id'])}</code>",
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
        f"{document['compose_claim_cap']} attempts. The counts are by "
        f"source node, so {document['live_sources_unaccepted']} rows cover "
        f"{document['assignment_coverage']['source_assignments_distinct']} distinct "
        "DD paths: "
        + (
            ", ".join(
                f"<code>{escape(path)}</code> carries "
                f"{len(rows)} live rows ({', '.join(row['class'] for row in rows)})"
                for path, rows in document["duplicate_source_paths"].items()
            )
            or "no path carries more than one live row"
        )
        + ".</p>"
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
        f"{cost['per_source_band_usd'][1]:.4f} per source). The measured cohort "
        f"rerun ({escape(cohort['node'])}, job {cohort['job']}) named "
        f"{cohort['sources_named']} of {cohort['sources_in_cohort']} sources for "
        f"USD {cohort['spend_usd']:.2f}, so USD {cohort['per_cohort_source_usd']:.5f} "
        "buys one source attempt and USD "
        f"{cohort['per_named_source_usd']:.5f} one named source. The JSON records "
        "every assignment, not a sample: each of the "
        f"{document['assignment_coverage']['source_assignments']} unaccepted sources "
        "and each of the "
        f"{document['assignment_coverage']['name_assignments']} exhausted names "
        "appears once, under its class, with the state the continuation run needs "
        "to scope that cohort.</p>"
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
    for sid, rec in sorted(unaccepted.items(), key=lambda item: item[1]["path"]):
        cls = classify_source(rec)
        source_classes[cls] += 1
        source_class_by_id[sid] = cls
        per_class_sources[cls].append(
            {
                "source": rec["path"],
                "status": rec["status"],
                "attempts": rec["attempts"],
                "reason": reason_family(rec["last_error"]),
                "parked": parked_disposition(rec),
            }
        )

    # Two live nodes can name one DD path; the count is by node, so the
    # collision is disclosed rather than collapsed, and a cohort scoped by
    # path must decide whether both rows are one work item.
    duplicated = {
        path: [
            {"class": cls, **row}
            for cls in CLASSES
            for row in per_class_sources.get(cls, [])
            if row["source"] == path
        ]
        for path, seen in Counter(
            row["source"] for cls in CLASSES for row in per_class_sources.get(cls, [])
        ).items()
        if seen > 1
    }

    name_classes = Counter()
    name_class_by_id: dict[str, str] = {}
    per_class_names: dict[str, list[dict]] = defaultdict(list)
    for name in names:
        cls = classify_exhausted(name)
        name_classes[cls] += 1
        name_class_by_id[name["id"]] = cls
        per_class_names[cls].append(
            {
                "id": name["id"],
                "score": name["score"],
                "docs_stage": name["docs_stage"],
                "validation_status": name["validation_status"],
            }
        )

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

    per_name_cost = costs["per_accepted_name_usd"]
    low_rate, high_rate = costs["per_source_band_usd"]
    cohort = costs.get("cohort_rerun_measured") or {}

    def class_record(
        cls: str, counts: Counter, samples: dict, *, population: str
    ) -> dict:
        count = counts.get(cls, 0)
        record = {
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
        if population == "source":
            record["expected_cost_usd_measured_cohort"] = {
                "per_source_attempted": round(
                    count * cohort.get("per_cohort_source_usd", 0.0), 2
                ),
                "per_named_source": round(
                    count * cohort.get("per_named_source_usd", 0.0), 2
                ),
                "basis": (
                    f"{COHORT_RERUN_NAMED} of {COHORT_RERUN_SOURCES} cohort sources "
                    f"named for USD {COHORT_RERUN_SPEND_USD:.2f}"
                ),
            }
        else:
            record["expected_cost_usd_measured_cohort"] = None
            record["measured_cohort_note"] = (
                "the measured cohort figure is a compose rate over sources; an "
                "exhausted name advances by review or refine instead"
            )
        return record

    assignments = {cls: per_class_sources.get(cls, []) for cls in CLASSES}
    name_assignments = {cls: per_class_names.get(cls, []) for cls in CLASSES}
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
            cls: class_record(
                cls, source_classes, per_class_sources, population="source"
            )
            for cls in CLASSES
        },
        "name_classes": {
            cls: class_record(cls, name_classes, per_class_names, population="name")
            for cls in CLASSES
        },
        # Every item's class, not a sample: the continuation run scopes its
        # cohorts by these source paths, so a truncated list cannot serve it.
        "source_assignments": assignments,
        "name_assignments": name_assignments,
        "duplicate_source_paths": duplicated,
    }
    document["assignment_coverage"] = {
        "source_assignments": sum(len(v) for v in assignments.values()),
        "source_assignments_distinct": len(
            {a["source"] for v in assignments.values() for a in v}
        ),
        "duplicated_source_paths": len(duplicated),
        "name_assignments": sum(len(v) for v in name_assignments.values()),
        "name_assignments_distinct": len(
            {a["id"] for v in name_assignments.values() for a in v}
        ),
    }
    check_document_assignments(
        document, unaccepted, names, source_class_by_id, name_class_by_id
    )
    check_cost_basis(document)
    payload = json.dumps(document, indent=2, sort_keys=True) + "\n"
    if len(payload) > 300_000:
        raise RuntimeError(
            f"emitted census is {len(payload)} bytes, above the 300000-byte "
            "single-file bound; the assignments belong in the run directory"
        )
    args.json.write_text(payload)
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
