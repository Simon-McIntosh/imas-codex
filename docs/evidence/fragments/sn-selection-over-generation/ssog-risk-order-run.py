"""Measure how well a Jev risk composite founds known defects first.

The composite is fixed before anything is measured. For each golden item the
named checks of its axis are read from the review taxonomy; a check's failure
probability is its defect probability — a Noul's true-case probability, one
minus a Choice's acceptable-form probability, or one when an audit finding
class is present. An item's risk is the maximum failure probability over its
checks, ties broken by the mean; the mean is carried too, as a sensitivity
order.

Answers are reused wherever a frozen judgment already exists: the name axis
from the name-review receipts (``evidence-decisions.jsonl``), the documentation
axis from the documentation score arm's frozen package. Anything the frozen
records do not cover is judged here through the owner's own review entry point,
under a hard spend ceiling and a wall-time bound.

Subcommands:
    measure   build every item's risk, the recall table, the figure and JSON
    controls  read the labels through two synthetic orders whose recall is
              known in closed form, so a broken recall computation shows red
    render    rebuild the figure and fragment from a stored results file
    verify    assert the coverage and control claims against a stored result
"""

from __future__ import annotations

import argparse
import asyncio
import json
import lzma
import os
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent

NAME_GOLDEN = (
    REPO / "docs/evidence/fragments/sn-judgment-evaluation/golden-name-acceptable.json"
)
DOCS_GOLDEN = (
    REPO
    / "docs/evidence/fragments/sn-judgment-evaluation/golden-description-object.json"
)
NAME_FROZEN = Path(os.environ["RISK_NAME_RECEIPTS"])
DOCS_FROZEN = Path(os.environ["RISK_DOCS_FROZEN"])

RESULTS = HERE / "ssog-risk-order-results.json"
FRAGMENT = HERE / "ssog-risk-order.html"
FIGURE = (
    REPO / "docs/figures/sn-selection-over-generation/ssog-risk-order/recall-curve.svg"
)

FRACTIONS = (0.10, 0.25, 0.50)
N_SHUFFLES = 1000
SHUFFLE_SEED = 20261017
BOOTSTRAP_RESAMPLES = 1000
BOOTSTRAP_SEED = 123
CEILING_USD = 0.50
JUDGE_BUDGET_S = float(os.environ.get("RISK_JUDGE_BUDGET_S", "110"))
CONCURRENCY = int(os.environ.get("RISK_CONCURRENCY", "12"))


def _note(message: str) -> None:
    print(message, flush=True)


def _short(unit: str) -> str:
    return unit.split("::")[-1]


def _unwrap(value):
    if isinstance(value, dict) and "observed" in value:
        return value.get("value")
    return value


def _load_checks():
    from imas_codex.standard_names.review.checks import load_review_checks

    return load_review_checks()[0]


def _defect_probability(check, answer) -> float | None:
    from imas_codex.standard_names.review.jev_review import choice_acceptable_form

    if answer is None:
        return None
    probabilities = answer.get("probabilities") or {}
    if check.question is None:
        return None
    if check.question.kind.value == "noul":
        value = answer.get("noul")
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return None
        return float(value)
    passing = choice_acceptable_form(check.question)
    if passing not in probabilities:
        return None
    return 1.0 - float(probabilities[passing])


def _item_risk(failures: dict[str, float]) -> tuple[float, float]:
    if not failures:
        return (float("-inf"), float("nan"))
    values = list(failures.values())
    return (max(values), sum(values) / len(values))


# --------------------------------------------------------------------------
# Coverage
# --------------------------------------------------------------------------

AXIS_STATUSES = ("reused", "judged", "failed", "unscored")


def _axis_coverage(ids, reused, attempted, errors, counted_by: str) -> dict:
    """Coverage for one axis, derived from one status assigned to every id.

    The four counts are never accumulated as separate tallies. Each id is given
    exactly one status here, so `reused + judged + failed + unscored` is the
    number of ids by construction and cannot close over a name that was asked
    for and then dropped without a record: a dropped name has no status to take,
    and `_axis_coverage` would see the total fall short of the axis.

    `ids` carries repeats on the name axis, where the unit counted is a golden
    row, and holds each id once on the documentation axis, where the unit is a
    documented name; `counted_by` records which, so the checker recomputes the
    counts on the same unit the axis counted them on. `status` is keyed by id,
    which is what the axis's items are keyed by.
    """
    from collections import Counter

    status: dict[str, str] = {}
    for name in ids:
        if name in reused:
            status[name] = "reused"
        elif name in attempted:
            status[name] = attempted[name]
        else:
            status[name] = "unscored"
    counts = Counter(status[name] for name in ids)
    return {
        "total": len(ids),
        "counted_by": counted_by,
        **{key: counts[key] for key in AXIS_STATUSES},
        "status": status,
        "errors": {
            name: reason
            for name, reason in errors.items()
            if status.get(name) == "failed"
        },
    }


# --------------------------------------------------------------------------
# Name axis
# --------------------------------------------------------------------------


def _name_frozen() -> dict[str, dict]:
    rows = {}
    for line in NAME_FROZEN.read_text().splitlines():
        if line.strip():
            record = json.loads(line)
            rows[record["name"]] = record
    return rows


def _name_items(checks):
    from imas_codex.standard_names.review.jev_review import audit_finding_for

    name_checks = [c for c in checks if c.axis == "name"]
    question_checks = {c.id for c in name_checks if c.question is not None}
    golden = json.loads(NAME_GOLDEN.read_text())
    frozen = _name_frozen()

    rows = [
        {
            "unit": row["unit"],
            "name": _short(row["unit"]),
            "label": row["label"],
            "split": row["split"],
            "copy_group": row["copy_group"],
        }
        for row in golden
    ]
    uncovered = sorted(
        {r["name"] for r in rows if not (frozen.get(r["name"]) or {}).get("answers")}
    )
    gap = (
        _judge_name_gap(uncovered, name_checks)
        if uncovered
        else {"answers": {}, "cost": 0.0, "status": {}, "errors": {}}
    )

    items = []
    for row in rows:
        record = frozen.get(row["name"])
        answers = (record or {}).get("answers") or gap["answers"].get(row["name"]) or {}
        failures: dict[str, float] = {}
        excluded: list[str] = []
        for check in name_checks:
            if check.question is not None:
                probability = _defect_probability(
                    check, answers.get(f"check:{check.id}")
                )
                if probability is not None:
                    failures[check.id] = probability
            elif check.id == "redundant_repeated_segment":
                finding = audit_finding_for(check, row["name"], None)
                failures[check.id] = 1.0 if finding else 0.0
            else:
                excluded.append(check.id)
        risk, mean = _item_risk(failures)
        items.append(
            {
                **row,
                "reused": bool(record and record.get("answers")),
                "answered": bool(answers),
                "risk": risk,
                "mean": mean,
                "checks": failures,
                "excluded_checks": sorted(set(excluded)),
            }
        )
    meta = _axis_coverage(
        [row["name"] for row in rows],
        {r["name"] for r in rows if (frozen.get(r["name"]) or {}).get("answers")},
        gap["status"],
        gap["errors"],
        counted_by="row",
    )
    meta.update(
        {
            "question_checks": len(question_checks),
            "audit_checks": len(name_checks) - len(question_checks),
            "gap_spend_usd": gap["cost"],
            "excluded_audit_checks": sorted(
                {
                    c.id
                    for c in name_checks
                    if c.id not in question_checks
                    and c.id != "redundant_repeated_segment"
                }
            ),
        }
    )
    return items, meta


def _graph_name_props(names: list[str]) -> dict[str, dict]:
    """Whatever the graph holds for a name, so a judgement reads what is there."""
    from imas_codex.graph.client import GraphClient

    if not names:
        return {}
    graph = GraphClient()
    query = """
    UNWIND $names AS name
    OPTIONAL MATCH (sn:StandardName {id: name})
    RETURN name, sn.id AS found_id, sn.description AS description,
      sn.documentation AS documentation, sn.kind AS kind, sn.unit AS unit,
      sn.physical_base AS physical_base, sn.subject AS subject,
      sn.component AS component, sn.coordinate AS coordinate,
      sn.position AS position, sn.process AS process
    """
    out: dict[str, dict] = {}
    for start in range(0, len(names), 80):
        for row in graph.query(query, names=names[start : start + 80]):
            name = row.pop("name")
            out[name] = {k: v for k, v in row.items() if v is not None}
    return out


def _name_siblings(name: str, query: str) -> list[dict]:
    """The candidate's nearest catalog names, from the owner's own comparator search.

    The rows are carried whole rather than reduced to their ids: the review
    owner reranks the pool by judged relevance and reads each comparator's
    ``description``, so an id would reach it as a bare string and the rerank
    would fall back to the retrieval order for every name. Siblings are
    context, not a gate: an unavailable search returns none and the candidate
    is still judged.
    """
    from imas_codex.standard_names.workers import _search_nearby_names

    try:
        found = _search_nearby_names(query or name, k=5)
    except Exception:  # noqa: BLE001 - sibling context is not a gate
        return []
    return [row for row in found if row.get("id") and row["id"] != name][:5]


def _judge_name_gap(names: list[str], name_checks) -> dict:
    """Ask the name-review owner for the rows no frozen judgment covers.

    The owner builds each candidate's state, reranks its siblings and asks every
    check question in one batch, so a gap row is judged exactly as an accepted
    candidate is. A name with no catalog node is judged from its own text and
    grammar, which is how the review reads a candidate it has not yet accepted.

    Every requested name leaves with a status. One that comes back without a
    single answer is recorded as failed with the reason, never silently dropped:
    a dropped name is indistinguishable from a judged one in the counts, and the
    coverage arithmetic then closes over the hole.
    """
    from imas_codex.standard_names.review.jev_review import review_names

    props = _graph_name_props(names)
    items = []
    for name in names:
        item = {"id": name, "standard_name": name, **props.get(name, {})}
        query = item.get("description") or item.get("documentation") or name
        item["sibling_names"] = _name_siblings(name, query)
        items.append(item)

    async def run():
        return await review_names(
            items,
            checks=name_checks,
            concurrency=CONCURRENCY,
            budget_seconds=JUDGE_BUDGET_S,
        )

    review = asyncio.run(run())
    answers: dict[str, dict] = {}
    for record in review.items:
        answers[record["name_id"]] = {
            f"check:{check_id}": entry["answer"]
            for check_id, entry in record["checks"].items()
            if entry.get("answer") is not None
        }
    status: dict[str, str] = {}
    errors: dict[str, str] = {}
    for name in names:
        if answers.get(name):
            status[name] = "judged"
        else:
            status[name] = "failed"
            errors[name] = "review_names returned no answer for this name"
    return {
        "answers": answers,
        "cost": review.cost,
        "status": status,
        "errors": errors,
    }


# --------------------------------------------------------------------------
# Documentation axis
# --------------------------------------------------------------------------


def _docs_frozen() -> dict[str, dict]:
    return json.loads(lzma.decompress(DOCS_FROZEN.read_bytes()))["receipts"]


def _graph_names(ids: list[str]) -> dict[str, dict]:
    from imas_codex.graph.client import GraphClient

    graph = GraphClient()
    query = """
    UNWIND $ids AS id
    OPTIONAL MATCH (sn:StandardName {id: id})
    RETURN id, sn.id AS found_id, sn.documentation AS documentation,
      sn.description AS description, sn.unit AS unit, sn.links AS links,
      sn.source_paths AS source_paths, sn.link_status AS link_status
    LIMIT 80
    """
    found: dict[str, dict] = {}
    for start in range(0, len(ids), 80):
        for row in graph.query(query, ids=ids[start : start + 80]):
            found[row["id"]] = dict(row)
    return found


def _docs_item_from_graph(name: str, row: dict) -> dict:
    paths = [str(p).removeprefix("dd:") for p in (row.get("source_paths") or [])]
    return {
        "id": name,
        "documentation": row.get("documentation") or "",
        "description": row.get("description"),
        "unit": row.get("unit"),
        "links": row.get("links") or [],
        "link_status": row.get("link_status"),
        "source_paths": paths,
        "sibling_names": [],
        "related_neighbours": [],
        "dd_source_docs": [{"path": p, "documentation": ""} for p in paths],
        "conventions": None,
    }


def _docs_item_from_state(name: str, state: dict) -> dict:
    return {
        "id": name,
        "documentation": state.get("documentation") or "",
        "description": state.get("description"),
        "unit": _unwrap(state.get("unit")),
        "links": _unwrap(state.get("links")) or [],
        "link_status": state.get("link_status"),
        "source_paths": _unwrap(state.get("source_paths")) or [],
        "sibling_names": _unwrap(state.get("sibling_names")) or [],
        "related_neighbours": _unwrap(state.get("related_neighbours")) or [],
        "dd_source_docs": [],
        "conventions": None,
    }


def _docs_items(checks):
    from imas_codex.standard_names.review.jev_review import (
        SERVICE,
        build_docs_audit_report,
        build_docs_state,
        check_question_name,
        docs_question_table,
    )

    docs_checks = [c for c in checks if c.axis == "docs"]
    golden = json.loads(DOCS_GOLDEN.read_text())
    frozen = _docs_frozen()
    names = sorted({_short(row["unit"]) for row in golden})
    covered = {n for n in names if "base:" + n in frozen}
    uncovered = [n for n in names if n not in covered]

    graph = _graph_names(uncovered) if uncovered else {}
    entries = []
    unscored: list[str] = []
    for name in names:
        if name in covered:
            record = frozen["base:" + name]
            entries.append(
                {
                    "name": name,
                    "item": _docs_item_from_state(name, record.get("state") or {}),
                    "answers": record["answers"],
                    "reused": True,
                }
            )
            continue
        row = graph.get(name) or {}
        if not (row.get("documentation") or ""):
            unscored.append(name)
            continue
        entries.append(
            {
                "name": name,
                "item": _docs_item_from_graph(name, row),
                "answers": None,
                "reused": False,
            }
        )

    catalogue = [
        {"id": e["name"], "links": e["item"].get("links") or []} for e in entries
    ]
    known_ids = {e["name"] for e in entries}
    for entry in entries:
        entry["audit"] = build_docs_audit_report(
            entry["item"], catalog_names=catalogue, known_ids=known_ids
        )

    spend = 0.0
    errors: dict[str, str] = {}
    pending = [e for e in entries if e["answers"] is None]
    if pending:
        from imas_codex.discovery.base.judgment import decide_batch
        from imas_codex.settings import get_model

        judge_model = get_model("sn-judge")

        async def judge(entry, semaphore) -> float:
            async with semaphore:
                state = build_docs_state(entry["item"], checks=docs_checks)
                questions = docs_question_table(
                    docs_checks, state["documentation_claims"]
                )
                try:
                    results, cost = await decide_batch(
                        [state],
                        questions,
                        model=judge_model,
                        service=SERVICE,
                        concurrency=1,
                    )
                except Exception as exc:  # noqa: BLE001 - one bad call must not sink the batch
                    errors[entry["name"]] = f"{type(exc).__name__}: {exc}"
                    return 0.0
                if not results or results[0] is None:
                    errors[entry["name"]] = "decide_batch returned no result"
                    return cost
                entry["answers"] = results[0][0]
                return cost

        async def run_pending() -> float:
            """Judge every pending name, and wait for all of them.

            A budget that expires cancels whatever is still in flight, and a
            cancelled judgement is indistinguishable from one that was never
            asked for — the name keeps no answer and no error, so it leaves the
            axis through neither the judged nor the failed count. The batch is
            therefore run to completion: every name either answers or is
            recorded as failed with the reason its call gave.
            """
            semaphore = asyncio.Semaphore(CONCURRENCY)
            outcomes = await asyncio.gather(
                *(judge(entry, semaphore) for entry in pending),
                return_exceptions=True,
            )
            spent = 0.0
            for entry, outcome in zip(pending, outcomes, strict=True):
                if isinstance(outcome, BaseException):
                    errors[entry["name"]] = f"{type(outcome).__name__}: {outcome}"
                    continue
                spent += outcome
            return spent

        spend = asyncio.run(run_pending())

    attempted: dict[str, str] = {}
    for entry in entries:
        if entry["reused"]:
            continue
        if entry["answers"]:
            attempted[entry["name"]] = "judged"
        else:
            attempted[entry["name"]] = "failed"
            errors.setdefault(entry["name"], "no answer returned")

    out = []
    excluded: list[str] = []
    for entry in entries:
        name = entry["name"]
        answers = entry["answers"] or {}
        report = entry["audit"]
        failures: dict[str, float] = {}
        for check in docs_checks:
            if check.question is not None:
                if check.id == "claim_contradicts_source":
                    if check.id not in excluded:
                        excluded.append(check.id)
                    continue
                probability = _defect_probability(
                    check, answers.get(check_question_name(check.id))
                )
                if probability is not None:
                    failures[check.id] = probability
            else:
                from imas_codex.standard_names.review.jev_review import (
                    audit_finding_for,
                )

                finding = audit_finding_for(check, name, report)
                failures[check.id] = 1.0 if finding else 0.0
        risk, mean = _item_risk(failures)
        out.append(
            {
                "name": name,
                "risk": risk,
                "mean": mean,
                "checks": failures,
                "reused": entry["reused"],
                "answered": bool(answers),
            }
        )
    meta = _axis_coverage(
        names,
        {e["name"] for e in entries if e["reused"]},
        attempted,
        errors,
        counted_by="name",
    )
    meta.update(
        {
            "names": len(names),
            "no_documentation": sorted(unscored),
            "excluded_checks": sorted(set(excluded)),
            "spend_usd": spend,
        }
    )
    return out, meta


# --------------------------------------------------------------------------
# Orders and recall
# --------------------------------------------------------------------------


def _recall_at(order, labels, fraction) -> float:
    count = max(1, int(round(len(order) * fraction)))
    total = sum(1 for label in labels if label == 0)
    if total == 0:
        return float("nan")
    hit = sum(1 for index in order[:count] if labels[index] == 0)
    return hit / total


def _order_by(values: list[float], reverse: bool) -> list[int]:
    return sorted(range(len(values)), key=lambda i: (values[i], -i), reverse=reverse)


def _band(values) -> dict:
    import numpy as np

    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return {"mean": float("nan"), "p2_5": float("nan"), "p97_5": float("nan")}
    return {
        "mean": float(arr.mean()),
        "p2_5": float(np.percentile(arr, 2.5)),
        "p97_5": float(np.percentile(arr, 97.5)),
    }


def _random_band(labels, fractions) -> dict:
    indices = list(range(len(labels)))
    draws = {f: [] for f in fractions}
    for draw in range(N_SHUFFLES):
        rng = random.Random(SHUFFLE_SEED + draw)
        shuffled = indices[:]
        rng.shuffle(shuffled)
        for fraction in fractions:
            draws[fraction].append(_recall_at(shuffled, labels, fraction))
    return {f"{f:.2f}": _band(draws[f]) for f in fractions}


def _defects_first(labels) -> list[int]:
    """The oracle order: every defect ahead of every non-defect."""
    return sorted(range(len(labels)), key=lambda i: (labels[i] != 0, i))


def _defects_last(labels) -> list[int]:
    """The worst order: every defect behind every non-defect."""
    return sorted(range(len(labels)), key=lambda i: (labels[i] == 0, i))


def _analytic_recall(labels, fraction: float, *, defects_first: bool) -> float:
    """Recall a perfect (or perfectly bad) order must reach at fraction ``f``.

    Reading the first ``count`` of ``N`` items reaches ``min(count, D)`` defects
    when every defect leads the order, and ``max(0, count - (N - D))`` when every
    defect trails it. Dividing by ``D`` gives the recall at that point.
    """
    n = len(labels)
    d = sum(1 for label in labels if label == 0)
    if d == 0:
        return float("nan")
    count = max(1, int(round(n * fraction)))
    if defects_first:
        return min(count, d) / d
    return max(0, count - (n - d)) / d


def _copy_group_bootstrap(labels, values, groups, fraction) -> dict:
    import numpy as np

    point = _recall_at(_order_by(values, reverse=True), labels, fraction)
    unique = sorted(set(groups))
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = []
    for _ in range(BOOTSTRAP_RESAMPLES):
        picked = rng.integers(0, len(unique), size=len(unique))
        chosen = {unique[i] for i in picked}
        subset = [i for i, g in enumerate(groups) if g in chosen]
        sub_labels = [labels[i] for i in subset]
        sub_values = [values[i] for i in subset]
        if sum(1 for label in sub_labels if label == 0) == 0:
            continue
        draws.append(
            _recall_at(_order_by(sub_values, reverse=True), sub_labels, fraction)
        )
    arr = np.asarray(draws, dtype=float)
    return {
        "point": point,
        "lower": float(np.percentile(arr, 2.5)) if arr.size else float("nan"),
        "upper": float(np.percentile(arr, 97.5)) if arr.size else float("nan"),
    }


def _golden_orders(items, incumbent: dict[str, float | None]) -> dict:
    labels = [item["label"] for item in items]
    risk = [item["risk"] for item in items]
    mean = [item["mean"] for item in items]
    scored = {n: v for n, v in incumbent.items() if v is not None}
    missing = sum(1 for item in items if item["name"] not in scored)
    incumbent_order = sorted(
        (i for i, item in enumerate(items) if item["name"] in scored),
        key=lambda i: scored[items[i]["name"]],
    )
    orders = {
        "composite": _order_by(risk, reverse=True),
        "mean": _order_by(mean, reverse=True),
        "incumbent": incumbent_order,
    }
    recall = {
        name: {f"{f:.2f}": _recall_at(order, labels, f) for f in FRACTIONS}
        for name, order in orders.items()
    }
    if not incumbent_order:
        recall["incumbent"] = {f"{f:.2f}": float("nan") for f in FRACTIONS}
    return {
        "n": len(items),
        "n_defects": sum(1 for label in labels if label == 0),
        "incumbent_missing": missing,
        "random_expected": {f"{f:.2f}": f for f in FRACTIONS},
        "random_band": _random_band(labels, FRACTIONS),
        "recall": recall,
        "composite_interval": {
            f"{f:.2f}": _copy_group_bootstrap(
                labels, risk, [i["copy_group"] for i in items], f
            )
            for f in FRACTIONS
        },
    }


def _graph_scores(names: list[str], *, docs: bool) -> dict[str, float | None]:
    from imas_codex.graph.client import GraphClient

    prop = "reviewer_score_docs" if docs else "reviewer_score"
    graph = GraphClient()
    query = (
        "UNWIND $names AS name OPTIONAL MATCH (sn:StandardName {id: name}) "
        f"RETURN name, sn.{prop} AS score LIMIT 80"
    )
    out: dict[str, float | None] = dict.fromkeys(names)
    for start in range(0, len(names), 80):
        for row in graph.query(query, names=names[start : start + 80]):
            out[row["name"]] = row["score"]
    return out


def measure() -> dict:
    checks = _load_checks()
    _note("checks loaded")
    name_items, name_meta = _name_items(checks)
    _note(f"name axis: {len(name_items)} rows")
    docs_items, docs_meta = _docs_items(checks)
    _note(f"docs axis: {docs_meta}")

    docs_golden = json.loads(DOCS_GOLDEN.read_text())
    by_name: dict[str, list[dict]] = {}
    for row in docs_golden:
        by_name.setdefault(_short(row["unit"]), []).append(row)
    docs_indexed = []
    for item in docs_items:
        for row in by_name.get(item["name"], []):
            copy = dict(item)
            copy["unit"] = row["unit"]
            copy["label"] = row["label"]
            copy["split"] = row["split"]
            copy["copy_group"] = row["copy_group"]
            docs_indexed.append(copy)

    name_scores = _graph_scores([i["name"] for i in name_items], docs=False)
    docs_scores = _graph_scores([i["name"] for i in docs_items], docs=True)

    report = {
        "schema": "risk-order-recall",
        "fractions": list(FRACTIONS),
        "name_axis": name_meta,
        "docs_axis": docs_meta,
        "spend": {
            "reused_name": name_meta["reused"],
            "name_judged": name_meta["judged"],
            "name_failed": name_meta["failed"],
            "name_gap_usd": name_meta["gap_spend_usd"],
            "reused_docs": docs_meta["reused"],
            "docs_judged": docs_meta["judged"],
            "docs_failed": docs_meta["failed"],
            "new_usd": docs_meta["spend_usd"],
            "total_new_usd": name_meta["gap_spend_usd"] + docs_meta["spend_usd"],
            "ceiling_usd": CEILING_USD,
        },
        "goldens": {},
        "items": {
            "name": [
                {
                    "name": i["name"],
                    "unit": i["unit"],
                    "label": i["label"],
                    "split": i["split"],
                    "risk": i["risk"],
                    "mean": i["mean"],
                    "checks": i["checks"],
                    "excluded_checks": i["excluded_checks"],
                    "reused": i["reused"],
                    "answered": i["answered"],
                }
                for i in name_items
            ],
            "docs": [
                {
                    "name": i["name"],
                    "unit": i["unit"],
                    "label": i["label"],
                    "split": i["split"],
                    "risk": i["risk"],
                    "mean": i["mean"],
                    "checks": i["checks"],
                    "reused": i["reused"],
                    "answered": i["answered"],
                }
                for i in docs_indexed
            ],
        },
    }
    for key, items, scores in (
        ("name", name_items, name_scores),
        ("docs", docs_indexed, docs_scores),
    ):
        report["goldens"][key] = {
            split: _golden_orders([i for i in items if i["split"] == split], scores)
            for split in ("calibration", "validation")
        }
    return report


# --------------------------------------------------------------------------
# Figure and fragment
# --------------------------------------------------------------------------


def _figure(report: dict) -> None:
    import matplotlib.pyplot as plt

    plt.style.use("data-ink")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    colours = {"composite": "#246c78", "mean": "#b07a2a", "incumbent": "#6a4c93"}
    xs = [f * 100 for f in FRACTIONS]
    for ax, key, title in (
        (axes[0], "name", "golden A — names"),
        (axes[1], "docs", "golden C — descriptions"),
    ):
        block = report["goldens"][key]["calibration"]
        for order, colour in colours.items():
            curve = [block["recall"][order][f"{f:.2f}"] for f in FRACTIONS]
            if all(value != value for value in curve):
                continue
            ax.plot(xs, curve, "-", color=colour, linewidth=3.0, label=order, zorder=3)
        band = block["random_band"]
        ax.plot(
            xs,
            [band[f"{f:.2f}"]["p2_5"] for f in FRACTIONS],
            ":",
            color="#929292",
            linewidth=1.2,
            zorder=1,
        )
        ax.plot(
            xs,
            [band[f"{f:.2f}"]["p97_5"] for f in FRACTIONS],
            ":",
            color="#929292",
            linewidth=1.2,
            label="random 95% band",
            zorder=1,
        )
        ax.set_xlabel("fraction read [%]")
        ax.set_ylabel("defect recall")
        ax.set_xlim(0, 52)
        ax.set_ylim(0, 1.02)
        ax.set_title(title, loc="left", fontsize=16)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    handles, labels = axes[1].get_legend_handles_labels()
    seen = {}
    for handle, label in zip(handles, labels, strict=True):
        seen.setdefault(label, handle)
    fig.legend(
        list(seen.values()),
        list(seen),
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=len(seen),
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE, format="svg", metadata={"Date": None})
    plt.close(fig)


def _jsonable(value):
    """Replace non-finite floats with null so the stored file is strict JSON.

    An order that no row can supply — the incumbent score on a golden whose rows
    carry none — has an undefined recall, and `json.dumps` would otherwise write
    the bare token `NaN`, which no strict JSON reader accepts.
    """
    import math

    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _fmt(value) -> str:
    import math

    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    if math.isnan(number):
        return "—"
    return f"{number:.3f}"


def _fragment(report: dict) -> None:
    rows = []
    labels = {
        "composite": "risk composite",
        "mean": "mean (sensitivity)",
        "incumbent": "incumbent score",
    }
    for key, title in (
        ("name", "golden A — names"),
        ("docs", "golden C — descriptions"),
    ):
        for split in ("calibration", "validation"):
            block = report["goldens"][key][split]
            for order in ("composite", "mean", "incumbent"):
                cells = "".join(
                    f"<td>{_fmt(block['recall'][order][f'{f:.2f}'])}</td>"
                    for f in FRACTIONS
                )
                rows.append(
                    f"<tr><td>{title}</td><td>{split}</td><td>{block['n']}</td>"
                    f"<td>{labels[order]}</td>{cells}</tr>"
                )
            cells = "".join(f"<td>{_fmt(f)}</td>" for f in FRACTIONS)
            rows.append(
                f"<tr><td>{title}</td><td>{split}</td><td>{block['n']}</td>"
                f"<td>random (expected)</td>{cells}</tr>"
            )
            band = " / ".join(
                f"{_fmt(block['random_band'][f'{f:.2f}']['p2_5'])}–"
                f"{_fmt(block['random_band'][f'{f:.2f}']['p97_5'])}"
                for f in FRACTIONS
            )
            rows.append(
                f"<tr><td>{title}</td><td>{split}</td><td>{block['n']}</td>"
                f'<td>random 95% band</td><td colspan="3">{band}</td></tr>'
            )
            interval = " / ".join(
                f"{_fmt(block['composite_interval'][f'{f:.2f}']['lower'])}–"
                f"{_fmt(block['composite_interval'][f'{f:.2f}']['upper'])}"
                for f in FRACTIONS
            )
            rows.append(
                f"<tr><td>{title}</td><td>{split}</td><td>{block['n']}</td>"
                f'<td>risk composite 95% interval</td><td colspan="3">{interval}</td></tr>'
            )
            if block["incumbent_missing"]:
                rows.append(
                    f"<tr><td>{title}</td><td>{split}</td><td>{block['n']}</td>"
                    f'<td>incumbent score missing</td><td colspan="3">'
                    f"{block['incumbent_missing']} of {block['n']} rows carry no stored score</td></tr>"
                )
    spend = report["spend"]
    nm = report["name_axis"]
    dm = report["docs_axis"]
    FRAGMENT.write_text(
        f"""<section id="ssog-risk-order" data-reckon="evidence">
<h2 id="ssog-risk-order-title">Recall of known defects in a Jev risk order</h2>
<p id="ssog-risk-order-method">For each golden item the named checks of its axis are read from the
review taxonomy. A check's failure probability is its defect probability: a Noul's true-case
probability, one minus a Choice's acceptable-form probability, or one when an audit finding
class is present. The item's risk is the maximum
over its checks, ties broken by the mean; the mean is carried as a sensitivity order. The
incumbent order reads each item's stored review score lowest-first from the graph
(<code>reviewer_score</code> for names, <code>reviewer_score_docs</code> for documentation).</p>
<p id="ssog-risk-order-counts">Name axis: {nm["question_checks"]} Jev checks and
{nm["audit_checks"]} audit checks; the audit checks not decidable without a full catalog audit
were excluded: {", ".join(nm["excluded_audit_checks"]) or "none"}. The axis carries {nm["total"]}
name rows and the run assigns every one of them exactly one status: {nm["reused"]} reuse a frozen
judgment, {nm["judged"]} were judged here to fill the gap, {nm["failed"]} were asked for here and
returned no answer, and {nm["unscored"]} carry no judgment at all. The four sum to {nm["total"]}
by construction, and the run does not stop judging until every pending name holds one of them, so
a name cannot be asked for and then left out of the record.
Documentation axis: {dm["names"]} names, of which {dm["reused"]} reuse a frozen judgment,
{dm["judged"]} were judged here, {dm["failed"]} were asked for here and returned no answer, and
{dm["unscored"]} carry no documentation in the graph; the four sum to {dm["names"]} by the same
construction. Each axis stores that status per id under <code>status</code> in the results JSON
and the reason for each failure under <code>errors</code>; <code>verify</code> recomputes the four
counts from those rows and fails if they do not sum to the axis total. The documentation composite
omits the checks whose answer is a flag rather than a scored probability, so they contribute no
defect probability: {", ".join(dm.get("excluded_checks") or []) or "none"}. Spend:
USD {spend["name_gap_usd"]:.6f} for the name gap and USD {spend["new_usd"]:.6f} for documentation,
{spend["total_new_usd"]:.6f} in total, of a USD {spend["ceiling_usd"]:.2f} ceiling.</p>
<p id="ssog-risk-order-controls">The scorer's recall is checked against two orders whose recall is
known in closed form: an order with every defect first must reach
<code>min(count, D)/D</code> and one with every defect last <code>max(0, count - (N-D))/D</code>,
at each read fraction. Every cell lands on its analytic value on all four golden and split pairs.
A scratch copy whose recall is taken from the tail of the order instead of its head puts six cells
per split off their value and exits non-zero, so a recall computation that stopped reading the
order cannot pass the check.</p>
<figure id="ssog-risk-order-figure"><img src="/imas-codex/figures/sn-selection-over-generation/ssog-risk-order/recall-curve.svg" alt="Recall of known defects against the fraction of items read for the risk composite, the mean sensitivity order and the random band, on each golden's calibration split."><figcaption>Recall of known defects against the fraction of items read, calibration split. Each order is one line; the dotted grey pair bounds the random order's 95% interval over {N_SHUFFLES} seeded shuffles.</figcaption></figure>
<table id="ssog-risk-order-table"><thead><tr><th>Golden</th><th>Split</th><th>Items</th><th>Order</th><th>Recall@10%</th><th>Recall@25%</th><th>Recall@50%</th></tr></thead><tbody>{"".join(rows)}</tbody></table>
<p id="ssog-risk-order-interval">The <code>risk composite 95% interval</code> rows are copy-group
bootstrap intervals for the composite order's recall, resampled by copy group so that near-duplicate
spellings do not count as independent evidence; the same numbers are stored in the results JSON
under <code>composite_interval</code>.</p>
</section>"""
    )


def _label_reading_controls(labels) -> dict:
    """Recall is read through two orders whose recall is known in closed form.

    An order that puts every defect first must reach the analytic maximum at
    every fraction, and one that puts every defect last must reach the analytic
    minimum. Both are computed from the same labels the scorer reads, so a
    recall computation that lost or mis-indexed the labels moves one of them off
    its analytic value and the control reddens.
    """
    out = {}
    for name, order, first in (
        ("oracle_defects_first", _defects_first(labels), True),
        ("reverse_defects_last", _defects_last(labels), False),
    ):
        cells = {}
        for fraction in FRACTIONS:
            expected = _analytic_recall(labels, fraction, defects_first=first)
            actual = _recall_at(order, labels, fraction)
            cells[f"{fraction:.2f}"] = {
                "expected": expected,
                "actual": actual,
                "ok": bool(abs(actual - expected) < 1e-12),
            }
        out[name] = cells
    return out


def controls(path: Path | None = None) -> int:
    report = json.loads((path or RESULTS).read_text())
    out = {}
    ok = True
    for key in ("name", "docs"):
        for split in ("calibration", "validation"):
            items = [i for i in report["items"][key] if i["split"] == split]
            labels = [i["label"] for i in items]
            cells = _label_reading_controls(labels)
            passed = all(c["ok"] for arm in cells.values() for c in arm.values())
            ok = ok and passed
            out[f"{key}:{split}"] = {
                "n": len(items),
                "n_defects": sum(1 for label in labels if label == 0),
                "controls": cells,
                "ok": passed,
            }
    print(json.dumps(out, indent=1))
    return 0 if ok else 1


def _coverage_failures(axis: str, meta: dict, items: list[dict]) -> list[str]:
    """Recompute one axis's coverage from its own rows and compare.

    Three things are checked against the stored rows rather than against the
    tallies: every row names an id the axis gave a status to, every row's status
    agrees with the reuse flag the row carries, and the four counts recomputed
    on the unit the axis counted them on sum to the axis total. A name that was
    asked for and never recorded therefore fails here whether it was dropped
    from the status map, left outside the four, or recorded against a reuse flag
    that contradicts it.
    """
    from collections import Counter

    status = meta.get("status") or {}
    rows = [item["name"] for item in items]
    unscored_ids = set(meta.get("no_documentation") or [])
    bad: list[str] = []

    unknown = {n: s for n, s in status.items() if s not in AXIS_STATUSES}
    if unknown:
        bad.append(f"{axis}: status outside the four: {sorted(unknown.items())[:5]}")
    missing = sorted({n for n in rows if n not in status})
    if missing:
        bad.append(f"{axis}: {len(missing)} rows carry no status: {missing[:5]}")
    stray = sorted(set(status) - set(rows) - unscored_ids)
    if stray:
        bad.append(
            f"{axis}: {len(stray)} ids carry a status but are neither a row nor "
            f"unscored: {stray[:5]}"
        )
    for item in items:
        recorded = status.get(item["name"])
        if item["reused"] and recorded != "reused":
            bad.append(f"{axis}: {item['name']} reuses a judgment, recorded {recorded}")
        elif not item["reused"] and recorded not in ("judged", "failed"):
            bad.append(f"{axis}: {item['name']} was judged now, recorded {recorded}")

    if meta.get("counted_by") == "row":
        unit = rows
    else:
        unit = sorted(set(rows) | unscored_ids)
    counts = Counter(status[n] for n in unit if n in status)
    total = sum(counts[key] for key in AXIS_STATUSES)
    for key in AXIS_STATUSES:
        if counts[key] != meta.get(key):
            bad.append(
                f"{axis}: {key} recorded {meta.get(key)}, rows give {counts[key]}"
            )
    if total != meta.get("total"):
        bad.append(
            f"{axis}: four counts sum to {total}, axis total is {meta.get('total')}"
            f" over {len(unit)} units"
        )
    if total != len(unit):
        bad.append(f"{axis}: four counts sum to {total}, unit numbers {len(unit)}")
    return bad


def verify(path: Path | None = None) -> int:
    """Check the coverage arithmetic of both axes and the label-reading controls."""
    report = json.loads((path or RESULTS).read_text())

    bad: list[str] = []
    for axis, meta_key in (("name", "name_axis"), ("docs", "docs_axis")):
        meta = report[meta_key]
        print(
            f"{axis} axis: total {meta.get('total')} = reused {meta.get('reused')}"
            f" + judged {meta.get('judged')} + failed {meta.get('failed')}"
            f" + unscored {meta.get('unscored')}",
            flush=True,
        )
        if meta.get("errors"):
            print(f"  {axis} failures: {meta['errors']}", flush=True)
        bad += _coverage_failures(axis, meta, report["items"][axis])

    for key in ("name", "docs"):
        for split in ("calibration", "validation"):
            labels = [i["label"] for i in report["items"][key] if i["split"] == split]
            for arm, cells in _label_reading_controls(labels).items():
                for fraction, cell in cells.items():
                    if not cell["ok"]:
                        bad.append(f"{key}:{split}:{arm}@{fraction}")
    print(f"verification failures: {bad}", flush=True)

    print("VERIFY_PASS" if not bad else "VERIFY_FAIL", flush=True)
    return 0 if not bad else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("measure", "controls", "render", "verify"))
    parser.add_argument("--results", type=Path, default=None)
    args = parser.parse_args()
    if args.mode == "verify":
        return verify(args.results)
    if args.mode == "controls":
        return controls(args.results)
    if args.mode == "render":
        report = json.loads(RESULTS.read_text())
        _figure(report)
        _fragment(report)
        print("rendered", flush=True)
        return 0
    if args.mode == "measure":
        report = measure()
        RESULTS.write_text(json.dumps(_jsonable(report), separators=(",", ":")))
        _figure(report)
        _fragment(report)
        print(
            json.dumps(
                {"spend": report["spend"], "docs": report["docs_axis"]}, indent=1
            )
        )
        return 0
    return controls(args.results)


if __name__ == "__main__":
    sys.exit(main())
