#!/usr/bin/env python
"""Re-decide the IDS-to-domain map by a two-level Jev Choice, one call per IDS.

Each IDS is scored on a state carrying its name, its DD documentation, a fixed
seeded sample of its leaf paths (description and unit) and its stored
path-domain mix. The question is the DD path classifier's own two-level tree,
reused rather than re-authored: the category question over every category, then
the domain question for the categories the first level leaves in play. The
current mapped domain is deliberately absent from the state so the answer does
not anchor on it. Only the model's judgment is added; the row comparison and
the proposed map are derived in code.

Modes
  --probe   score one IDS, print its raw answer and the spend, then stop.
  (default) score every IDS, write the results JSON and the fragment.
  --rescore read the saved answers and reproduce every figure at zero cost.
"""

from __future__ import annotations

import argparse
import asyncio
import html
import json
import random
import sys
import time
from pathlib import Path

from imas_codex.discovery.base.judgment import decide_batch
from imas_codex.graph.client import GraphClient
from imas_codex.graph.dd_domain_classifier import (
    CATEGORY_QUESTION,
    DOMAIN_QUESTION,
    SERVICE_TAG,
    _answer_for,
    _beamed_categories,
    _ranked_probabilities,
    build_category_question,
    build_domain_question,
)

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
RESULTS = HERE / "scg-ids-map-jev-results.json"
FRAGMENT = HERE / "scg-ids-map-jev.html"
IDS_MAP_PATH = REPO / "imas_codex" / "definitions" / "physics" / "ids_domains.json"
DESIGN_RESULTS = HERE / "scg-ids-domain-design-results.json"

SAMPLE_PER_IDS = 30
SEAT = "sn-domain-tree"
CONCURRENCY = 8
SPEND_CEILING_USD = 0.50
LOW_CONFIDENCE = 0.5


def model_id() -> str:
    from imas_codex.settings import get_model

    return get_model(SEAT)


def load_ids_map() -> dict[str, str]:
    return json.loads(IDS_MAP_PATH.read_text())["ids_domain_mappings"]


def load_path_mix() -> dict[str, dict]:
    """Per-IDS stored domain counts, read from the design's results JSON."""
    payload = json.loads(DESIGN_RESULTS.read_text())
    return payload["ids_mix_summary"]["entries"]


def load_proposed() -> dict[str, str]:
    """The 23 rule-derived corrections the design proposed, IDS to domain."""
    payload = json.loads(DESIGN_RESULTS.read_text())
    return payload["ids_map_summary"]["proposed_changes"]


def load_descriptions(gc: GraphClient) -> dict[str, str]:
    rows = gc.query(
        "MATCH (n:IDS) WHERE n.description IS NOT NULL "
        "RETURN n.id AS id, n.description AS description"
    )
    return {r["id"]: r["description"] for r in rows}


def sample_leaves(gc: GraphClient, ids: str, k: int = SAMPLE_PER_IDS) -> list[dict]:
    """A fixed, seeded sample of *ids*'s leaf paths with description and unit.

    Ordered by path id then drawn by a seed keyed to the IDS name, so the same
    IDS yields the same sample on every run and the state is reproducible.
    """
    rows = gc.query(
        "MATCH (n:IMASNode) WHERE n.ids = $ids AND n.is_leaf = true "
        "RETURN n.id AS id, n.name AS name, n.description AS description, "
        "n.documentation AS documentation, n.unit AS unit, "
        "n.node_category AS node_category ORDER BY n.id",
        ids=ids,
    )
    if len(rows) <= k:
        return [
            {
                "path": r["id"],
                "name": r["name"],
                "description": r["description"] or "",
                "unit": r["unit"] or "",
                "node_category": r["node_category"] or "",
            }
            for r in rows
        ]
    rng = random.Random(f"ids-sample:{ids}")
    picks = sorted(rng.sample(range(len(rows)), k))
    return [
        {
            "path": rows[i]["id"],
            "name": rows[i]["name"],
            "description": rows[i]["description"] or "",
            "unit": rows[i]["unit"] or "",
            "node_category": rows[i]["node_category"] or "",
        }
        for i in picks
    ]


def top_mix(mix_entry: dict | None, n: int = 2) -> list[list]:
    """The *n* most common real domains among the IDS's paths, ignoring unclassified."""
    if not mix_entry:
        return []
    domains = mix_entry.get("domains") or {}
    real = [(d, c) for d, c in domains.items() if d and d != "unclassified"]
    real.sort(key=lambda kv: (-kv[1], kv[0]))
    return [[d, c] for d, c in real[:n]]


def build_state(ids: str, description: str, leaves: list[dict], mix_entry: dict | None) -> dict:
    return {
        "ids_name": ids,
        "documentation": description,
        "leaf_path_sample": [
            {
                "path": leaf["path"],
                "description": leaf["description"],
                "unit": leaf["unit"],
                "node_category": leaf["node_category"],
            }
            for leaf in leaves
        ],
        "stored_path_domain_mix": {
            d: c for d, c in (mix_entry or {}).get("domains", {}).items()
        },
    }


def _answer_record(answer, category: str | None = None, category_probability=None):
    if not answer:
        return None
    probabilities = answer.get("probabilities") or {}
    return {
        "choice": answer.get("choice"),
        "confidence": answer.get("confidence"),
        "probabilities": {str(k): float(v) for k, v in probabilities.items()},
        "category": category,
        "category_probability": category_probability,
    }


async def score_one(state: dict, model: str) -> dict:
    """Score one IDS through the reused two-level tree; return its raw answers."""
    first, cost = await decide_batch(
        [state],
        build_category_question(),
        model=model,
        service=SERVICE_TAG,
        concurrency=CONCURRENCY,
    )
    result = first[0] if first else None
    record = {
        "category_answer": _answer_record(_answer_for(result, CATEGORY_QUESTION)),
        "domain_answers": [],
    }
    domain_answers = record["domain_answers"]
    for category in _beamed_categories(result):
        question = build_domain_question(category)
        options = set(question[DOMAIN_QUESTION]["criteria"])
        results, domain_cost = await decide_batch(
            [state],
            question,
            model=model,
            service=SERVICE_TAG,
            concurrency=CONCURRENCY,
        )
        cost += domain_cost
        answer = _answer_for(results[0] if results else None, DOMAIN_QUESTION)
        if not answer:
            continue
        category_probability = 0.0
        for label, value in _ranked_probabilities(
            _answer_for(result, CATEGORY_QUESTION)
        ):
            if label == category:
                category_probability = value
                break
        entry = _answer_record(answer, category, category_probability)
        entry["offered"] = sorted(options)
        entry["domain_probability"] = float(entry["probabilities"].get(entry["choice"], 0.0))
        entry["combined"] = category_probability * entry["domain_probability"]
        domain_answers.append(entry)
    return {"answers": record, "cost_usd": cost}


def choose_domain(record: dict) -> dict | None:
    best = None
    for entry in record.get("domain_answers") or []:
        if best is None or entry["combined"] > best["combined"]:
            best = entry
    return best


def classify_row(current: str, rule: str | None, chosen: str | None, confidence) -> str:
    if chosen is None:
        return "no_answer"
    if confidence is not None and confidence < LOW_CONFIDENCE:
        return "low_confidence"
    if rule is not None and chosen == rule:
        return "agree"
    if chosen == current:
        return "confirms_current"
    return "third_domain"


def build_rows(ids_map: dict[str, str], proposed: dict[str, str], mix: dict, answers: dict) -> list[dict]:
    rows = []
    for ids in sorted(ids_map):
        record = answers.get(ids) or {}
        chosen = choose_domain(record)
        category = record.get("category_answer")
        row = {
            "ids": ids,
            "current_domain": ids_map[ids],
            "rule_correction": proposed.get(ids),
            "jev_domain": chosen["choice"] if chosen else None,
            "jev_domain_probability": chosen["domain_probability"] if chosen else None,
            "jev_domain_confidence": chosen["confidence"] if chosen else None,
            "jev_category": chosen["category"] if chosen else None,
            "jev_category_probability": category["category_probability"]
            if category
            else None,
            "jev_category_confidence": category["confidence"] if category else None,
            "category_answer": category,
            "domain_answers": record.get("domain_answers") or [],
            "top_path_domains": top_mix(mix.get(ids)),
        }
        row["classification"] = classify_row(
            row["current_domain"],
            row["rule_correction"],
            row["jev_domain"],
            row["jev_domain_confidence"],
        )
        rows.append(row)
    return rows


def summarise_rows(rows: list[dict]) -> dict:
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["classification"]] = counts.get(row["classification"], 0) + 1
    review = [r["ids"] for r in rows if r["classification"] != "agree"]
    return {"counts": counts, "review_ids": review}


def proposed_map(rows: list[dict]) -> dict[str, str]:
    """The map Jev's answer would land, keeping the current value where unanswered."""
    out = {}
    for row in rows:
        out[row["ids"]] = row["jev_domain"] or row["current_domain"]
    return out


async def run(ids_filter: list[str] | None, probe: bool) -> int:
    ids_map = load_ids_map()
    proposed = load_proposed()
    mix = load_path_mix()
    with GraphClient() as gc:
        descriptions = load_descriptions(gc)
        targets = ids_filter or sorted(ids_map)
        states: dict[str, dict] = {}
        for ids in targets:
            states[ids] = build_state(
                ids,
                descriptions.get(ids, ""),
                sample_leaves(gc, ids),
                mix.get(ids),
            )
    model = model_id()
    print(f"model={model} ids={len(targets)} ceiling_usd={SPEND_CEILING_USD}")

    spent = 0.0
    answers: dict[str, dict] = {}
    if probe:
        outcome = await score_one(states[targets[0]], model)
        spent += outcome["cost_usd"]
        print(f"  {targets[0]:32s} cost={outcome['cost_usd']:.6f}")
        print(json.dumps(outcome["answers"], indent=2)[:4000])
    else:
        gate = asyncio.Semaphore(CONCURRENCY)

        async def _bounded(ids: str) -> tuple[str, dict]:
            async with gate:
                return ids, await score_one(states[ids], model)

        outcomes = await asyncio.gather(*(_bounded(ids) for ids in targets))
        for ids, outcome in outcomes:
            spent += outcome["cost_usd"]
            answers[ids] = outcome["answers"]
            if spent > SPEND_CEILING_USD:
                print(f"SPEND CEILING EXCEEDED: {spent:.6f}", file=sys.stderr)
                return 2

    per_ids_cost = spent / max(len(answers), 1)
    print(f"spent_usd={spent:.6f} per_ids_usd={per_ids_cost:.6f} "
          f"estimate_all_usd={per_ids_cost * len(ids_map):.6f}")
    if probe:
        return 0

    rows = build_rows(ids_map, proposed, mix, answers)
    summary = summarise_rows(rows)
    results = {
        "seat": SEAT,
        "model": model,
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "ids_count": len(ids_map),
        "sample_per_ids": SAMPLE_PER_IDS,
        "spend_usd": round(spent, 6),
        "spend_ceiling_usd": SPEND_CEILING_USD,
        "low_confidence_threshold": LOW_CONFIDENCE,
        "inputs": {
            "ids_map": ids_map,
            "proposed_changes": proposed,
            "path_mix": mix,
        },
        "answers": answers,
        "rows": rows,
        "summary": summary,
        "proposed_ids_map": proposed_map(rows),
    }
    RESULTS.write_text(json.dumps(results, indent=1, sort_keys=True))
    write_fragment(results)
    print(f"wrote {RESULTS} and {FRAGMENT} rows={len(rows)} summary={summary['counts']}")
    return 0


def rescore() -> int:
    """Recompute every figure from the saved answers and inputs; no call, no spend."""
    payload = json.loads(RESULTS.read_text())
    rows = build_rows(
        payload["inputs"]["ids_map"],
        payload["inputs"]["proposed_changes"],
        payload["inputs"]["path_mix"],
        payload["answers"],
    )
    summary = summarise_rows(rows)
    same_rows = rows == payload["rows"]
    same_summary = summary == payload["summary"]
    same_proposed = proposed_map(rows) == payload["proposed_ids_map"]
    print(
        f"rows_match={same_rows} summary_match={same_summary} "
        f"proposed_match={same_proposed}"
    )
    print(f"counts={summary['counts']} review={len(summary['review_ids'])}")
    return 0 if (same_rows and same_summary and same_proposed) else 1


def _mix_text(top: list) -> str:
    return ", ".join(f"{d} {c}" for d, c in top) or "—"


def _fmt(value) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.2f}"
    return html.escape(str(value))


def _row_html(row: dict) -> str:
    conf = row["jev_domain_confidence"]
    mark = "" if conf is None or conf >= LOW_CONFIDENCE else " *"
    return (
        "<tr>"
        f"<td><code>{html.escape(row['ids'])}</code></td>"
        f"<td>{_fmt(row['current_domain'])}</td>"
        f"<td>{_fmt(row['rule_correction'])}</td>"
        f"<td>{_fmt(row['jev_domain'])}{mark}</td>"
        f"<td>{_fmt(row['jev_domain_probability'])}</td>"
        f"<td>{_fmt(row['jev_domain_confidence'])}</td>"
        f"<td>{_fmt(row['jev_category'])}</td>"
        f"<td>{_mix_text(row['top_path_domains'])}</td>"
        f"<td>{_fmt(row['classification'])}</td>"
        "</tr>"
    )


def _table(rows: list[dict]) -> str:
    head = (
        "<table><thead><tr><th>IDS</th><th>current</th><th>rule</th>"
        "<th>Jev domain</th><th>p</th><th>conf</th><th>Jev category</th>"
        "<th>top path mix</th><th>class</th></tr></thead><tbody>"
    )
    return head + "".join(_row_html(r) for r in rows) + "</tbody></table>"


def write_fragment(payload: dict) -> None:
    rows = payload["rows"]
    counts = payload["summary"]["counts"]
    review = [r for r in rows if r["classification"] != "agree"]
    agreed = [r for r in rows if r["classification"] and r["classification"] == "agree"]
    order = {"third_domain": 0, "low_confidence": 1, "confirms_current": 2}
    review.sort(key=lambda r: (order.get(r["classification"], 9), r["ids"]))
    counts_text = ", ".join(f"{k} {v}" for k, v in sorted(counts.items()))
    body = [
        f'<section id="scg-ids-map-jev">',
        "<h2>Jev re-decides the IDS-to-domain map</h2>",
        "<p>One two-level Choice per IDS, over the DD path classifier's own "
        f"category-then-domain question, on the <code>{html.escape(payload['seat'])}</code> "
        f"seat (<code>{html.escape(payload['model'])}</code>). The state carries the IDS "
        "name, its DD documentation, "
        f"{payload['sample_per_ids']} seeded leaf paths with description and unit, and the "
        "IDS's stored path-domain mix; the current mapped domain is left out. "
        f"87 IDSs scored for USD {payload['spend_usd']:.6f} against a ceiling of "
        f"USD {payload['spend_ceiling_usd']:.2f}. Rows below the {payload['low_confidence_threshold']} "
        "domain confidence carry an asterisk and the class <code>low_confidence</code>.</p>",
        f"<p>Classification counts: {html.escape(counts_text)}</p>",
        "<h3>The rows the lead reviews</h3>",
        "<p>Every IDS whose answer is not an agreement with the rule-derived correction: "
        "Jev proposing a third domain, Jev confirming the current map, or a low-confidence "
        "answer. The separating evidence is the Jev domain's probability and confidence, the "
        "rule-derived target where the design proposed one, and the IDS's two most common "
        "stored path domains.</p>",
        _table(review),
        "<h3>Agreements with the rule-derived corrections</h3>",
        _table(agreed),
        "<h3>Every IDS, in full</h3>",
        _table(rows),
        "<h3>Distributions and data</h3>",
        "<p>Each IDS's full category probability distribution and every offered domain "
        "answer with its full distribution are in "
        "<code>scg-ids-map-jev-results.json</code> under <code>answers</code> "
        "(<code>category_answer</code> and <code>domain_answers</code>), beside the row table "
        "under <code>rows</code>. The proposed map is the <code>proposed_ids_map</code> object "
        "in that JSON, carried as data only; the definitions file is not written.</p>",
        "</section>",
    ]
    FRAGMENT.write_text("\n".join(body) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--ids", action="append", default=None)
    parser.add_argument("--rescore", action="store_true")
    parser.add_argument("--fragment", action="store_true")
    args = parser.parse_args()
    if args.rescore:
        return rescore()
    if args.fragment:
        write_fragment(json.loads(RESULTS.read_text()))
        print(f"wrote {FRAGMENT}")
        return 0
    return asyncio.run(run(args.ids, args.probe))


if __name__ == "__main__":
    raise SystemExit(main())