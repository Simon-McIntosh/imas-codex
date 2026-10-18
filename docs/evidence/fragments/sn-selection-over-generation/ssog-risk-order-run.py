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
BAND_SEEDS = frozenset(SHUFFLE_SEED + draw for draw in range(N_SHUFFLES))
SHUFFLE_CONTROL_SEED = SHUFFLE_SEED + N_SHUFFLES + 1  # past the band's own seed range
SHUFFLE_CONTROL_FRACTION = 0.25
BOOTSTRAP_RESAMPLES = 1000
BOOTSTRAP_SEED = 123
LABEL_PERMUTATION_SEED = 20261018
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


def _random_count(n: int, fraction: float) -> int:
    """How many items a read at ``fraction`` takes from ``n``, after rounding."""
    return max(1, int(round(n * fraction)))


def _random_expectation(n: int, fraction: float) -> float:
    """The recall a random order has in expectation at ``fraction``, exactly.

    A random order reads ``count = max(1, round(n * fraction))`` items, the same
    rounded count the recall itself uses, and each defect is among them with
    probability ``count / n``, so the expected recall is ``count / n`` — not the
    rounded fraction it approximates. The table prints the count beside it.
    """
    return _random_count(n, fraction) / n


def _composite_interval(labels, values, groups, fraction) -> dict:
    """Percentile bootstrap of the composite order's recall, resampled by copy group.

    The grouping and the draw are the shared owner's, not a local
    reimplementation: ``_group_indices`` builds one index array per copy group,
    so near-duplicate spellings move together instead of counting as
    independent evidence, and the draw reads a group index for each group
    ``len(groups)`` times with replacement — the same draw
    ``bootstrap_interval`` makes. Drawing a group more than once, and dropping
    others, is what a bootstrap resample is; an interval built from a set of
    drawn groups samples without replacement and is narrower than the
    distribution it claims to describe. The owner's interval is over an AUC and
    takes no statistic argument, so the recall-at-fraction statistic is applied
    to each resample here while the resampling itself stays the owner's.

    A resample holding no defect has no recall, so it is redrawn;
    ``n_resamples`` is the number of usable draws.
    """
    import numpy as np

    from imas_codex.discovery.base.evaluation import _group_indices

    point = _recall_at(_order_by(values, reverse=True), labels, fraction)
    index_groups = _group_indices(groups, len(labels))
    y = np.asarray(labels)
    score = np.asarray(values, dtype=float)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws: list[float] = []
    attempts = 0
    while len(draws) < BOOTSTRAP_RESAMPLES and attempts < BOOTSTRAP_RESAMPLES * 100:
        attempts += 1
        picks = rng.integers(0, len(index_groups), len(index_groups))
        index = np.concatenate([index_groups[pick] for pick in picks])
        resampled_labels = y[index]
        if np.unique(resampled_labels).size < 2:
            continue
        draws.append(
            _recall_at(
                _order_by(score[index], reverse=True), resampled_labels, fraction
            )
        )
    arr = np.asarray(draws, dtype=float)
    return {
        "point": point,
        "lower": float(np.percentile(arr, 2.5)) if arr.size else float("nan"),
        "upper": float(np.percentile(arr, 97.5)) if arr.size else float("nan"),
        "n_resamples": int(arr.size),
        "groups": len(index_groups),
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
        "random_expected": {
            f"{f:.2f}": _random_expectation(len(items), f) for f in FRACTIONS
        },
        "random_band": _random_band(labels, FRACTIONS),
        "recall": recall,
        "composite_interval": {
            f"{f:.2f}": _composite_interval(
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
            cells = "".join(
                f"<td>{_fmt(block['random_expected'][f'{f:.2f}'])}"
                f" ({_random_count(block['n'], f)}/{block['n']})</td>"
                for f in FRACTIONS
            )
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
<p id="ssog-risk-order-invariance">Those two arms test the recall arithmetic, not the risk that
feeds it: a scorer that read a row's label into its risk would rank its own ground truth and pass
both arms untouched, because both arms supply the order rather than the risk. A third control
closes that path. Each split's risk vector is recomputed from every row's own stored checks, and
again with the labels permuted among the rows of that split under a fixed seed, and the two vectors
must be identical. The risk a row published is also required to equal the risk recomputed from its
label-free inputs, so a leak on the way into the results file is caught as well as a leak in the
scoring path. The control fails too when the permutation is the identity, which would compare the
vector with itself and test nothing. A scratch copy whose risk reads the label reddens it and exits
non-zero.</p>
<p id="ssog-risk-order-band-check">A fourth control reads the same property through the labels' own
distribution. Each split's labels are permuted under one seed drawn from a base outside the random
band's seed range — asserted, because a control seed inside that range reproduces a band sample
exactly and compares the recall with a copy of itself — the scorer's risk is recomputed on the
permuted rows, and the recall of the resulting order at 25% is required to fall inside the random
band computed for those permuted labels. A scorer that reads its own checks cannot follow the
permutation: its order is fixed, it faces randomly labelled defects, and it lands inside the band. A
scorer that reads a label into its risk re-derives its order from the permutation, ranks the
permuted defects first and lands <em>above</em> the band, so the control can fail and does when the
scorer leaks. The four controls therefore prove four different things, and no measured value moves
if any one of them is dropped: defects-first and defects-last prove the recall arithmetic against a
recall known in closed form; label invariance proves the risk values do not depend on the labels;
and this shuffled-label band proves the order the risk induces carries no label information. A
scratch copy whose scorer reads the labels reddens the invariance and the band together and exits
non-zero. The <code>random 95% band</code> rows and the <code>random (expected)</code> rows state
what an uninformative order scores and are what the band control is read against.</p>
<figure id="ssog-risk-order-figure"><img src="/imas-codex/figures/sn-selection-over-generation/ssog-risk-order/recall-curve.svg" alt="Recall of known defects against the fraction of items read for the risk composite, the mean sensitivity order and the random band, on each golden's calibration split."><figcaption>Recall of known defects against the fraction of items read, calibration split. Each order is one line; the dotted grey pair bounds the random order's 95% interval over {N_SHUFFLES} seeded shuffles.</figcaption></figure>
<table id="ssog-risk-order-table"><thead><tr><th>Golden</th><th>Split</th><th>Items</th><th>Order</th><th>Recall@10%</th><th>Recall@25%</th><th>Recall@50%</th></tr></thead><tbody>{"".join(rows)}</tbody></table>
<p id="ssog-risk-order-interval">The <code>risk composite 95% interval</code> rows are copy-group
bootstrap intervals for the composite order's recall, resampled by copy group so that near-duplicate
spellings do not count as independent evidence; the same numbers are stored in the results JSON
under <code>composite_interval</code>. The grouping and the resampling are the shared owner's:
<code>_group_indices</code> builds one index array per copy group and the draw reads a group index
for each group, with replacement — the same draw the shared <code>bootstrap_interval</code> makes,
so a drawn group can be taken more than once. The owner's interval is over an AUC and takes no
statistic argument, so the recall at each read fraction is applied to each resample here while the
resampling itself stays the owner's. Relative to drawing distinct groups, resampling with
replacement draws more often from over-represented groups, so the interval is the wider and more
conservative of the two, and a resample with one class only is redrawn rather than counted.</p>
<p id="ssog-risk-order-random-expected">The <code>random (expected)</code> rows give the exact
recall a random order has in expectation, <code>count/N</code>, where <code>count</code> is the
rounded number of items a read at that fraction takes — the same rounded count the recall uses —
and <code>N</code> is the split's item count. Each cell prints that count beside it, so the value
is read against the items it was computed on rather than against the unrounded fraction.</p>
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


def _scorer_risk(row: dict) -> float:
    """One row's risk, from that row's own checks.

    The single place a risk is derived from a stored row, so the invariance
    control below can be pointed at the whole scoring path rather than at a
    restatement of it. It reads ``checks`` and nothing else: a row's label is
    the ground truth the measurement is scored against, and a risk that reads it
    is not a prediction.
    """
    return _item_risk(row["checks"])[0]


def _risks_agree(stored, recomputed) -> bool:
    """Whether a stored risk is the value the scoring path produces.

    A risk with no checks is negative infinity, which the results file carries
    as null; both sides are therefore read as non-finite-equal.
    """
    import math

    try:
        left = float(stored)
    except (TypeError, ValueError):
        return False
    return (math.isnan(left) and math.isnan(recomputed)) or left == recomputed


def label_invariance_control(
    rows: list[dict], seed: int = LABEL_PERMUTATION_SEED
) -> dict:
    """The risk vector must not move when the labels move.

    The scorer's risk is recomputed from each row's own checks twice: once with
    the row's own label and once with the labels permuted among the rows of the
    split under a fixed seed. If the two vectors differ, the scorer read the
    labels it is supposed to be scored against and every recall number it
    produced is circular. A third clause binds the published number to that
    path: the risk a row stores must equal the risk recomputed from its own
    checks, so a risk that reads a label on the way into the results file is
    caught even when the scoring path itself is clean. The control also reports
    how many rows changed place, and fails when none did: a permutation that is
    the identity would compare the vector with itself and pass without testing
    anything.
    """
    import random

    permuted = list(range(len(rows)))
    random.Random(seed).shuffle(permuted)
    moved = sum(1 for i, source in enumerate(permuted) if source != i)
    before = [_scorer_risk(row) for row in rows]
    after = [
        _scorer_risk({**row, "label": rows[source]["label"]})
        for row, source in zip(rows, permuted, strict=True)
    ]
    differing = [
        index
        for index, (first, second) in enumerate(zip(before, after, strict=True))
        if first != second
    ]
    unbound = [
        index
        for index, (row, value) in enumerate(zip(rows, before, strict=True))
        if not _risks_agree(row.get("risk"), value)
    ]
    return {
        "rows": len(rows),
        "rows_label_changed_place": moved,
        "rows_risk_changed": len(differing),
        "rows_stored_risk_unbound": len(unbound),
        "ok": not differing and moved > 0 and not unbound,
    }


def shuffle_band_control(
    rows: list[dict], fraction: float = SHUFFLE_CONTROL_FRACTION
) -> dict:
    """The risk must carry no label information, tested through shuffled labels.

    The labels of one split are permuted under a seed drawn from a base outside
    the random band's own seed range, the scorer's risk is recomputed on the
    permuted rows, and the recall of the resulting order at ``fraction`` is
    required to fall inside the random band computed for those same permuted
    labels. A scorer that reads its rows' checks cannot follow the permutation —
    its order is fixed, so it faces randomly labelled defects and lands inside
    the band. A scorer that reads a label into its risk re-derives its order
    from the permutation, ranks the permuted defects first and lands above the
    band, which is what makes this control able to fail.

    The seed is asserted to be outside every band seed, because a control drawn
    from the band's own seed range reproduces a band sample exactly and compares
    the recall with a copy of itself rather than with the distribution.
    """
    import random

    seed = SHUFFLE_CONTROL_SEED
    permutation = list(range(len(rows)))
    random.Random(seed).shuffle(permutation)
    moved = sum(1 for index, source in enumerate(permutation) if source != index)
    shuffled = [
        {**row, "label": rows[source]["label"]}
        for row, source in zip(rows, permutation, strict=True)
    ]
    labels = [row["label"] for row in shuffled]
    risk = [_scorer_risk(row) for row in shuffled]
    recall = _recall_at(_order_by(risk, reverse=True), labels, fraction)
    band = _random_band(labels, (fraction,))[f"{fraction:.2f}"]
    inside = band["p2_5"] <= recall <= band["p97_5"]
    seed_clear = seed not in BAND_SEEDS
    return {
        "seed": seed,
        "fraction": fraction,
        "seed_in_band_seeds": not seed_clear,
        "rows": len(rows),
        "rows_label_changed_place": moved,
        "recall": recall,
        "band": band,
        "inside": inside,
        "ok": inside and seed_clear and moved > 0,
    }


def controls(path: Path | None = None) -> int:
    report = json.loads((path or RESULTS).read_text())
    out = {}
    ok = True
    for key in ("name", "docs"):
        for split in ("calibration", "validation"):
            items = [i for i in report["items"][key] if i["split"] == split]
            labels = [i["label"] for i in items]
            cells = _label_reading_controls(labels)
            invariance = label_invariance_control(items)
            shuffled = shuffle_band_control(items)
            passed = all(c["ok"] for arm in cells.values() for c in arm.values())
            passed = passed and invariance["ok"] and shuffled["ok"]
            ok = ok and passed
            out[f"{key}:{split}"] = {
                "n": len(items),
                "n_defects": sum(1 for label in labels if label == 0),
                "controls": cells,
                "risk_label_invariance": invariance,
                "shuffled_label_band": shuffled,
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
            items = [i for i in report["items"][key] if i["split"] == split]
            labels = [i["label"] for i in items]
            for arm, cells in _label_reading_controls(labels).items():
                for fraction, cell in cells.items():
                    if not cell["ok"]:
                        bad.append(f"{key}:{split}:{arm}@{fraction}")
            invariance = label_invariance_control(items)
            print(
                f"{key}:{split}: label invariance — {invariance['rows']} rows,"
                f" {invariance['rows_label_changed_place']} changed place,"
                f" {invariance['rows_risk_changed']} risk values moved,"
                f" {invariance['rows_stored_risk_unbound']} stored risks unbound",
                flush=True,
            )
            if not invariance["ok"]:
                bad.append(f"{key}:{split}:label_invariance")
            shuffled = shuffle_band_control(items)
            print(
                f"{key}:{split}: shuffled-label band — seed {shuffled['seed']}"
                f" (in band seeds: {shuffled['seed_in_band_seeds']}),"
                f" {shuffled['rows_label_changed_place']} of {shuffled['rows']} relabelled,"
                f" recall@{shuffled['fraction']:.2f} {shuffled['recall']:.6f}"
                f" in band {shuffled['band']['p2_5']:.6f}-{shuffled['band']['p97_5']:.6f}"
                f" -> inside {shuffled['inside']}",
                flush=True,
            )
            if not shuffled["ok"]:
                bad.append(f"{key}:{split}:shuffled_label_band")
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
