"""Measure the typed-Choice domain classifier against reviewed standard-name labels.

Same path set, source query, context gathering, derivation and scoring as the
chat-classifier measurement; only the classifier call differs. The run writes
nothing to the graph. A second invocation without ``--fresh`` rescores the saved
answers and reproduces every figure with no paid call.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from imas_codex.graph.client import GraphClient
from imas_codex.graph.dd_domain_classifier import (
    build_domain_questions,
    classify_tier1_llm,
    gather_classification_context,
)
from imas_codex.settings import get_model
from imas_codex.standard_names.domain_ranking import select_primary_domain

HERE = Path(__file__).resolve().parent
LABELS = HERE / "scg-domain-label-sheet.json"
BASELINE = HERE / "scg-classifier-vocabulary-results.json"
RESULTS = HERE / "scg-jev-domain-benchmark-results.json"
SPEND_CEILING_USD = 1.00
CHUNK = 20
PROBABILITY_BANDS = (
    ("above 0.8", 0.8, 1.01),
    ("0.5 to 0.8", 0.5, 0.8),
    ("below 0.5", -0.01, 0.5),
)
SOURCE_QUERY = """
UNWIND $names AS name
MATCH (sn:StandardName {id: name})
OPTIONAL MATCH (sn)<-[:PRODUCED_NAME]-(source:StandardNameSource)
              -[:FROM_DD_PATH]->(dd:IMASNode)
RETURN sn.id AS name, sn.physics_domain AS name_domain,
       sn.source_domains AS name_source_domains,
       source.id AS source_id, dd.id AS path,
       dd.physics_domain AS stored_domain, dd.domain_source AS domain_source,
       dd.domain_model AS domain_model,
       toString(dd.domain_classified_at) AS domain_classified_at,
       dd.node_category AS node_category, dd.description AS description,
       dd.unit AS unit
ORDER BY name, path
LIMIT 10000
"""


def _read_sources(gc: GraphClient, names: list[str]) -> list[dict]:
    rows = gc.query(SOURCE_QUERY, names=names)
    if not rows or not any(row["path"] for row in rows):
        raise RuntimeError("source query failed its positive control")
    if len(rows) == 10000:
        raise RuntimeError("source query reached its result bound")
    return rows


def _snapshot(rows: list[dict]) -> dict:
    names = {}
    paths = {}
    for row in rows:
        names[row["name"]] = [row["name_domain"], row["name_source_domains"]]
        if row["path"]:
            paths[row["path"]] = [
                row["stored_domain"],
                row["domain_source"],
                row["domain_model"],
                row["domain_classified_at"],
            ]
    canonical = json.dumps([names, paths], sort_keys=True, default=str)
    return {
        "name_count": len(names),
        "path_count": len(paths),
        "name_domain_distribution": dict(
            sorted(Counter(v[0] or "<none>" for v in names.values()).items())
        ),
        "path_domain_distribution": dict(
            sorted(Counter(v[0] or "<none>" for v in paths.values()).items())
        ),
        "domain_classified_at_max": max(
            (v[3] or "" for v in paths.values()), default=""
        ),
        "property_digest": hashlib.sha256(canonical.encode()).hexdigest(),
    }


def _score(entries: list[dict], key: str, *, source_set: str | None = None) -> dict:
    correct = sum(
        row["label"] in row[source_set] if source_set else row[key] == row["label"]
        for row in entries
    )
    confusion = Counter(
        (row["label"], row[key] or "<none>")
        for row in entries
        if not (
            row["label"] in row[source_set] if source_set else row[key] == row["label"]
        )
    )
    return {
        "correct": correct,
        "total": len(entries),
        "accuracy": correct / len(entries) if entries else None,
        "confusion": [
            {"label": actual, "predicted": predicted, "count": count}
            for (actual, predicted), count in confusion.most_common()
        ],
    }


async def _fresh(
    gc: GraphClient, paths: dict[str, dict], cache_path: Path, limit: int | None
) -> tuple[dict, dict, float, str, list[str]]:
    model = get_model("sn-classifier")
    cache = (
        json.loads(cache_path.read_text())
        if cache_path.exists()
        else {"model": model, "spend_usd": 0.0, "domains": {}, "probabilities": {}}
    )
    if cache["model"] != model:
        raise RuntimeError("cached model differs from current classifier model")
    domains = cache["domains"]
    probabilities = cache["probabilities"]
    keys = sorted(paths)
    if limit is not None:
        # evenly strided sample so a pilot spans every IDS in the set
        keys = keys[:: max(1, len(keys) // limit)][:limit]
    pending = [paths[key] for key in keys if key not in domains]
    if pending:
        context = gather_classification_context(gc, [p["id"] for p in pending])
        context_ids = {p["id"] for p in context}
        if context_ids != {p["id"] for p in pending}:
            raise RuntimeError("classification context missed live source paths")
    for _attempt in range(2):
        retry = []
        for start in range(0, len(pending), CHUNK):
            if cache["spend_usd"] >= SPEND_CEILING_USD * 0.95:
                raise RuntimeError("spend reserve reached before all paths were done")
            chunk = pending[start : start + CHUNK]
            charges: list[float] = []
            result = await classify_tier1_llm(
                gc, chunk, model=model, on_cost=charges.append
            )
            got = {item["id"] for item in result}
            if len(got) != len(result) or not got <= {item["id"] for item in chunk}:
                raise RuntimeError("classifier duplicated or invented a source path")
            retry.extend(item for item in chunk if item["id"] not in got)
            cache["spend_usd"] += sum(charges)
            if cache["spend_usd"] > SPEND_CEILING_USD:
                raise RuntimeError(
                    f"USD {SPEND_CEILING_USD:.2f} spend ceiling exceeded: "
                    f"{cache['spend_usd']:.6f}"
                )
            for item in result:
                domains[item["id"]] = item["physics_domain"]
                probabilities[item["id"]] = item["domain_probability"]
            cache_path.write_text(json.dumps(cache, sort_keys=True, indent=2) + "\n")
            print(
                f"classified={len(domains)}/{len(paths)} "
                f"spend_usd={cache['spend_usd']:.6f} unanswered={len(retry)}",
                flush=True,
            )
        pending = retry
        if not pending:
            break
    failed = sorted(item["id"] for item in pending)
    return domains, probabilities, cache["spend_usd"], model, failed


def _derive(
    label_rows: list[dict],
    graph_rows: list[dict],
    fresh: dict,
    probabilities: dict | None = None,
) -> tuple[list[dict], dict]:
    probabilities = probabilities or {}
    by_name = defaultdict(list)
    for row in graph_rows:
        if row["path"]:
            by_name[row["name"]].append(row)
    result = []
    for label in label_rows:
        sources = by_name[label["name"]]
        stored = sorted(
            {row["stored_domain"] for row in sources if row["stored_domain"]}
        )
        current = sorted(
            {fresh[row["path"]] for row in sources if row["path"] in fresh}
        )
        result.append(
            {
                "name": label["name"],
                "label": label["label"],
                "label_source": label.get("label_source", "coordinator"),
                "split": label.get("split"),
                "sources": [
                    {
                        "path": row["path"],
                        "stored": row["stored_domain"],
                        "fresh": fresh.get(row["path"]),
                        "probability": probabilities.get(row["path"]),
                        "domain_source": row["domain_source"],
                    }
                    for row in sources
                ],
                "stored_domains": stored,
                "fresh_domains": current,
                "stored_primary": select_primary_domain(stored) if stored else None,
                "fresh_primary": select_primary_domain(current) if current else None,
                "derived_domain": select_primary_domain(current) if current else None,
            }
        )
    scores = {}
    for cohort, rows in (
        ("lead", result[:20]),
        ("coordinator", result[20:]),
    ):
        scores[cohort] = {
            "stored": _score(rows, "stored_primary", source_set="stored_domains"),
            "fresh": _score(rows, "fresh_primary", source_set="fresh_domains"),
            "derived": _score(rows, "derived_domain"),
            "with_live_sources": sum(bool(row["sources"]) for row in rows),
        }
    return result, scores


def _baseline_answers(baseline: dict) -> dict[str, str]:
    """The recorded chat-classifier answer for every source path."""
    answers: dict[str, str] = {}
    for row in baseline["rows"]:
        for source in row["sources"]:
            answers[source["path"]] = source["fresh"]
    return answers


def _per_domain(rows: list[dict], baseline_rows: dict[str, dict]) -> list[dict]:
    """Derived-domain accuracy per labelled domain, for both classifiers."""
    tally: dict[str, dict] = defaultdict(
        lambda: {"names": 0, "jev_correct": 0, "chat_correct": 0}
    )
    for row in rows:
        bucket = tally[row["label"]]
        bucket["names"] += 1
        bucket["jev_correct"] += row["derived_domain"] == row["label"]
        bucket["chat_correct"] += (
            baseline_rows[row["name"]]["derived_domain"] == row["label"]
        )
    return [
        {"label": label, **values}
        | {
            "jev_accuracy": values["jev_correct"] / values["names"],
            "chat_accuracy": values["chat_correct"] / values["names"],
        }
        for label, values in sorted(tally.items())
    ]


def _disagreements(
    paths: dict[str, dict],
    jev: dict[str, str],
    probabilities: dict[str, float],
    chat: dict[str, str],
    labels_by_path: dict[str, list[tuple[str, str]]],
) -> list[dict]:
    result = []
    for path in sorted(paths):
        if path not in jev or jev[path] == chat.get(path):
            continue
        named = labels_by_path.get(path, [])
        result.append(
            {
                "path": path,
                "chat": chat.get(path),
                "jev": jev[path],
                "jev_probability": probabilities.get(path),
                "labelled_names": [[name, label] for name, label in named],
            }
        )
    return result


def _supplying_probability(row: dict) -> float | None:
    """Highest probability among the sources that supplied the derived domain."""
    supplied = [
        s["probability"]
        for s in row["sources"]
        if s["fresh"] == row["derived_domain"] and s["probability"] is not None
    ]
    return max(supplied) if supplied else None


def _bands(rows: list[dict], baseline_rows: dict[str, dict]) -> list[dict]:
    """Name accuracy by the probability of the path that supplied the domain.

    The chat classifier is scored on the same names, so a band shows where each
    classifier is right on identical ground.
    """
    out = []
    for band, low, high in PROBABILITY_BANDS:
        members = [
            row
            for row in rows
            if (p := _supplying_probability(row)) is not None and low <= p < high
        ]
        correct = sum(r["derived_domain"] == r["label"] for r in members)
        chat_correct = sum(
            baseline_rows[r["name"]]["derived_domain"] == r["label"] for r in members
        )
        out.append(
            {
                "band": band,
                "names": len(members),
                "correct": correct,
                "accuracy": correct / len(members) if members else None,
                "chat_correct": chat_correct,
                "chat_accuracy": chat_correct / len(members) if members else None,
            }
        )
    return out


def _path_bands(
    paths: dict[str, dict],
    jev: dict[str, str],
    probabilities: dict[str, float],
    labels_by_path: dict[str, list[tuple[str, str]]],
) -> list[dict]:
    """Per source path: Jev domain against the label of each name it feeds."""
    out = []
    for band, low, high in PROBABILITY_BANDS:
        in_band = [
            p
            for p in sorted(paths)
            if p in jev
            and labels_by_path.get(p)
            and low <= probabilities.get(p, 0.0) < high
        ]
        pairs = [(p, label) for p in in_band for _, label in labels_by_path[p]]
        correct = sum(jev[p] == label for p, label in pairs)
        out.append(
            {
                "band": band,
                "paths": len(in_band),
                "pairs": len(pairs),
                "correct": correct,
                "accuracy": correct / len(pairs) if pairs else None,
            }
        )
    return out


def _figure(payload: dict, path: Path) -> None:
    """Per-domain derived accuracy of both classifiers, one row per label."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    jev_colour, chat_colour, link_colour = "#3a78a8", "#52514e", "#bdbab4"
    rows = sorted(payload["per_domain"], key=lambda r: (-r["names"], r["label"]))
    total = payload["scores"]["coordinator"]["derived"]
    chat_total = payload["baseline"]["derived_rederived_at_this_selector"][
        "coordinator"
    ]
    plt.style.use("data-ink")
    fig, ax = plt.subplots(figsize=(14, 0.52 * (len(rows) + 1) + 1.6))
    ys = list(range(len(rows) + 1))[::-1]
    entries = [
        (
            f"all labelled names ({total['total']})",
            chat_total["correct"] / chat_total["total"],
            total["accuracy"],
        )
    ] + [
        (
            f"{r['label']} ({r['names']})",
            r["chat_accuracy"],
            r["jev_accuracy"],
        )
        for r in rows
    ]
    for y, (_, chat_acc, jev_acc) in zip(ys, entries, strict=True):
        ax.plot([chat_acc, jev_acc], [y, y], color=link_colour, lw=2.4, zorder=1)
        ax.scatter([chat_acc], [y], s=140, color=chat_colour, zorder=3, linewidths=0)
        ax.scatter([jev_acc], [y], s=140, color=jev_colour, zorder=3, linewidths=0)
    ax.axvline(0.597, color=link_colour, lw=1.2, ls=":", zorder=0)
    ax.text(0.597, ys[0] + 0.9, "0.597", ha="center", color=chat_colour, fontsize=20)
    ax.set_yticks(ys)
    ax.set_yticklabels([label for label, _, _ in entries])
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.7, ys[0] + 1.5)
    ax.set_xlabel("derived name-domain accuracy")
    ax.text(
        1.02,
        ys[0] - 0.05,
        "chat classifier",
        color=chat_colour,
        ha="right",
        va="bottom",
        fontsize=20,
    )
    ax.text(
        1.02,
        ys[0] - 0.95,
        "typed Choice",
        color=jev_colour,
        ha="right",
        va="bottom",
        fontsize=20,
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="svg", metadata={"Date": None})


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="pilot sample size")
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=RESULTS)
    parser.add_argument("--rows-out", type=Path, default=None)
    parser.add_argument("--figure", type=Path, default=None)
    args = parser.parse_args()
    sheet = json.loads(LABELS.read_text())
    label_rows = sheet["reference"] + sheet["sample"]
    names = [row["name"] for row in label_rows]
    if len(names) != 170 or len(set(names)) != 170:
        raise RuntimeError("expected 170 distinct labelled names")
    baseline = json.loads(BASELINE.read_text())
    failed: list[str] = []
    with GraphClient() as gc:
        before_rows = _read_sources(gc, names)
        before = _snapshot(before_rows)
        paths = {
            row["path"]: {
                "id": row["path"],
                "node_category": row["node_category"],
                "description": row["description"],
                "units": row["unit"],
            }
            for row in before_rows
            if row["path"]
        }
        print("BEFORE " + json.dumps(before, sort_keys=True), flush=True)
        if args.fresh:
            fresh, probabilities, spend, model, failed = await _fresh(
                gc, paths, args.cache, args.limit
            )
            if args.limit is not None:
                print(
                    f"PILOT paths={len(fresh)} spend_usd={spend:.6f} "
                    f"estimate_all_usd={spend / max(1, len(fresh)) * len(paths):.4f}",
                    flush=True,
                )
                return
        else:
            cache = json.loads(args.cache.read_text())
            fresh, spend, model = cache["domains"], cache["spend_usd"], cache["model"]
            probabilities = cache["probabilities"]
            failed = sorted(set(paths) - set(fresh))
        probabilities = {key: round(value, 4) for key, value in probabilities.items()}
        if not set(fresh) <= set(paths):
            raise RuntimeError("cached classifications name paths outside the set")
        after_rows = _read_sources(gc, names)
        after = _snapshot(after_rows)
        print("AFTER " + json.dumps(after, sort_keys=True), flush=True)
        if before != after:
            raise RuntimeError(
                "graph source or name properties changed during read-only run"
            )
    rows, scores = _derive(label_rows, after_rows, fresh, probabilities)

    # control: the recorded chat answers, pushed through this same scorer. The
    # selector-independent measures (stored, source-set) must reproduce the
    # published counts exactly; the derived measure is re-derived under the
    # selector at this revision, which may rank domains differently from the
    # selector the published figure used, so the drift is counted by name.
    chat = _baseline_answers(baseline)
    control_rows, control_scores = _derive(label_rows, after_rows, chat)
    for cohort in ("lead", "coordinator"):
        for key in ("stored", "fresh"):
            want = baseline["scores"][cohort][key]
            got = control_scores[cohort][key]
            if (want["correct"], want["total"]) != (got["correct"], got["total"]):
                raise RuntimeError(f"scorer failed to reproduce {cohort}/{key}")
    recorded = {row["name"]: row["derived_domain"] for row in baseline["rows"]}
    baseline_rows = {row["name"]: row for row in control_rows}
    selector_drift = sorted(
        name
        for name, row in baseline_rows.items()
        if row["derived_domain"] != recorded[name]
    )

    labels_by_path: dict[str, list[tuple[str, str]]] = defaultdict(list)
    label_of = {row["name"]: row["label"] for row in label_rows}
    for row in after_rows:
        if row["path"]:
            labels_by_path[row["path"]].append((row["name"], label_of[row["name"]]))
    coordinator = rows[20:]
    accuracy = scores["coordinator"]["derived"]["accuracy"]
    payload = {
        "model": model,
        "question_hash": hashlib.sha256(
            json.dumps(build_domain_questions(), sort_keys=True).encode()
        ).hexdigest()[:16],
        "spend_usd": spend,
        "source_path_count": len(paths),
        "classified_path_count": len(fresh),
        "unclassified_paths": failed,
        "verdict": {
            "rule": "adopt when derived accuracy >= 0.597 and spend < USD 2.66",
            "accuracy": accuracy,
            "spend_usd": spend,
            "adopt": bool(
                accuracy is not None
                and accuracy >= 0.597
                and spend < baseline["spend_usd"]
                and not failed
            ),
        },
        "baseline": {
            "model": baseline["model"],
            "spend_usd": baseline["spend_usd"],
            "scores": {
                cohort: {k: v for k, v in s.items() if k != "stored"}
                for cohort, s in baseline["scores"].items()
            },
            "scorer_reproduction": {
                cohort: {
                    key: control_scores[cohort][key]["accuracy"]
                    for key in ("stored", "fresh", "derived")
                }
                for cohort in ("lead", "coordinator")
            },
            "derived_rederived_at_this_selector": {
                cohort: control_scores[cohort]["derived"]
                for cohort in ("lead", "coordinator")
            },
            "selector_drift_names": selector_drift,
        },
        "snapshot_before": before,
        "snapshot_after": after,
        "scores": scores,
        "per_domain": _per_domain(coordinator, baseline_rows),
        "name_probability_bands": _bands(coordinator, baseline_rows),
        "path_probability_bands": _path_bands(
            paths, fresh, probabilities, labels_by_path
        ),
        "path_disagreements": _disagreements(
            paths, fresh, probabilities, chat, labels_by_path
        ),
        "name_disagreements": [
            {
                "name": row["name"],
                "label": row["label"],
                "jev": row["derived_domain"],
                "chat": baseline_rows[row["name"]]["derived_domain"],
                "jev_probability": _supplying_probability(row),
            }
            for row in coordinator
            if row["derived_domain"] != baseline_rows[row["name"]]["derived_domain"]
        ],
        "domains": fresh,
        "probabilities": probabilities,
    }
    # per-name rows are re-derived by a scoring pass; they sit outside the
    # recorded file to keep it small
    if args.rows_out is not None:
        args.rows_out.write_text(json.dumps(rows, sort_keys=True, indent=1) + "\n")
    args.out.write_text(json.dumps(payload, sort_keys=True, indent=1) + "\n")
    if args.figure is not None:
        _figure(payload, args.figure)
    print("SCORES " + json.dumps(scores, sort_keys=True), flush=True)
    print("VERDICT " + json.dumps(payload["verdict"], sort_keys=True), flush=True)
    print(
        f"SPEND_USD={spend:.6f} QUESTION_HASH={payload['question_hash']} "
        f"RESULT_SHA256={hashlib.sha256(args.out.read_bytes()).hexdigest()}",
        flush=True,
    )


if __name__ == "__main__":
    asyncio.run(main())
