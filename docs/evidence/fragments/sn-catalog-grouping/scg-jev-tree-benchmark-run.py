"""Measure the two-level Jev domain tree against the chat classifier on held-out labels.

Same path set, source query, derivation through the domain selector and scorer as
the flat-Jev benchmark; only the classifier differs. The tree arm runs the
category-then-domain Choice and keeps each path's two-level probability. The
threshold is chosen on the calibration names and every figure reported on the
validation names. The chat arm is the recorded chat classifier's answers, scored
through the same derivation and scorer with no paid call, so a second invocation
without ``--fresh`` reproduces every figure at zero cost.

The tree's own entry point ``decide_domain_tree`` is called rather than
``_classify_batch_tree`` because the composed classifier's threshold must be
swept over the tree's per-path probability, which ``_classify_batch_tree``
discards for the paths it routes to chat.

Writes nothing to the graph.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from imas_codex.graph.client import GraphClient
from imas_codex.graph.dd_domain_classifier import (
    apply_path_domain_override,
    build_category_question,
    build_domain_question,
    decide_domain_tree,
    gather_classification_context,
)
from imas_codex.settings import get_model
from imas_codex.standard_names.domain_ranking import select_primary_domain

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
LABELS = HERE / "scg-domain-label-sheet.json"
BASELINE = HERE / "scg-classifier-vocabulary-results.json"
GOLD_SET = ROOT / "imas_codex/definitions/physics/domain_gold_set.json"
RESULTS = HERE / "scg-jev-tree-benchmark-results.json"
SPEND_CEILING_USD = 4.00
CHUNK = 20
CALIBRATION = "calibration"
VALIDATION = "validation"
PROBABILITY_BANDS = (
    ("above 0.8", 0.8, 1.01),
    ("0.5 to 0.8", 0.5, 0.8),
    ("below 0.5", -0.01, 0.5),
)
# Grid the calibration split is swept over; the chosen threshold maximises the
# calibration derived-domain accuracy, ties broken toward the higher threshold
# so a tie does not buy chat spend it cannot justify.
THRESHOLD_GRID = tuple(round(0.05 * i, 2) for i in range(0, 21))
SOURCE_QUERY = """
UNWIND $names AS name
MATCH (sn:StandardName {id: name})
OPTIONAL MATCH (sn)<-[:PRODUCED_NAME]-(source:StandardNameSource)
              -[:FROM_DD_PATH]->(dd:IMASNode)
RETURN sn.id AS name, sn.physics_domain AS name_domain,
       sn.source_domains AS name_source_domains,
       source.id AS source_id, dd.id AS path,
       dd.physics_domain AS stored_domain, dd.domain_source AS domain_source,
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
            paths[row["path"]] = [row["stored_domain"], row["domain_source"]]
    canonical = json.dumps([names, paths], sort_keys=True, default=str)
    return {
        "name_count": len(names),
        "path_count": len(paths),
        "property_digest": hashlib.sha256(canonical.encode()).hexdigest(),
        "path_domain_distribution": dict(
            sorted(Counter(v[0] or "<none>" for v in paths.values()).items())
        ),
    }


def _score(entries: list[dict], key: str) -> dict:
    correct = sum(row[key] == row["label"] for row in entries)
    confusion = Counter(
        (row["label"], row[key] or "<none>")
        for row in entries
        if row[key] != row["label"]
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


async def _run_tree(
    paths: dict[str, dict], cache_path: Path, limit: int | None
) -> dict:
    model = get_model("sn-domain-tree")
    cache = (
        json.loads(cache_path.read_text())
        if cache_path.exists()
        else {"model": model, "spend_usd": 0.0, "decisions": {}}
    )
    if cache["model"] != model:
        raise RuntimeError("cached tree model differs from the current seat")
    decisions = cache["decisions"]
    keys = sorted(paths)
    if limit is not None:
        keys = keys[:: max(1, len(keys) // limit)][:limit]
    pending = [key for key in keys if key not in decisions]
    with GraphClient() as gc:
        context = gather_classification_context(gc, pending)
    context_by_id = {c["id"]: c for c in context}
    if set(context_by_id) != set(pending):
        raise RuntimeError("classification context missed live source paths")
    for start in range(0, len(pending), CHUNK):
        if cache["spend_usd"] >= SPEND_CEILING_USD * 0.9:
            raise RuntimeError("spend reserve reached before all paths were done")
        ids = pending[start : start + CHUNK]
        batch = [context_by_id[i] for i in ids]
        results, cost = await decide_domain_tree(
            batch, model=model, service="standard-names"
        )
        cache["spend_usd"] += cost
        if cache["spend_usd"] > SPEND_CEILING_USD:
            raise RuntimeError(
                f"USD {SPEND_CEILING_USD:.2f} spend ceiling exceeded: "
                f"{cache['spend_usd']:.6f}"
            )
        for decision in results:
            if decision is None:
                continue
            decisions[decision["id"]] = {
                "domain": decision["physics_domain"],
                "category": decision["physics_category"],
                "probability": decision["probability"],
                "category_probability": decision["category_probability"],
                "domain_probability": decision["domain_probability"],
            }
        cache_path.write_text(json.dumps(cache, sort_keys=True, indent=1) + "\n")
        print(
            f"tree_classified={len(decisions)}/{len(paths)} "
            f"spend_usd={cache['spend_usd']:.6f}",
            flush=True,
        )
    cache["model"] = model
    return cache


def _baseline_answers(baseline: dict) -> dict[str, str]:
    """The recorded chat-classifier answer for every source path."""
    answers: dict[str, str] = {}
    for row in baseline["rows"]:
        for source in row["sources"]:
            answers[source["path"]] = source["fresh"]
    return answers


def _compose(
    tree: dict, chat: dict, paths: list[str], threshold: float
) -> dict[str, str]:
    """The composed classifier: the tree above the threshold, chat below it.

    A path the tree did not score, or scored below *threshold*, is answered by
    the recorded chat classifier. A confident tree answer gets the deterministic
    path-domain override applied, as the routing rule does in production.
    """
    answers: dict[str, str] = {}
    for path in paths:
        decision = tree.get(path)
        if decision is None or decision["probability"] < threshold:
            if path in chat:
                answers[path] = chat[path]
            continue
        domain = decision["domain"]
        override = apply_path_domain_override(path)
        answers[path] = override if override is not None else domain
    return answers


def _derive(
    label_rows: list[dict], answers: dict[str, str], rows_by_name: dict
) -> list[dict]:
    derived = []
    for label in label_rows:
        sources = [r for r in rows_by_name[label["name"]] if r["path"]]
        domains = sorted({answers[r["path"]] for r in sources if r["path"] in answers})
        derived.append(
            {
                "name": label["name"],
                "label": label["label"],
                "split": label.get("split"),
                "reference": bool(label.get("reference")),
                "domains": domains,
                "derived_domain": select_primary_domain(domains) if domains else None,
                "sources": [
                    {"path": r["path"], "present": r["path"] in answers}
                    for r in sources
                ],
            }
        )
    return derived


def _overrides_changed(tree: dict, paths: list[str], threshold: float) -> int:
    """Confident tree answers the deterministic override rewrote."""
    changed = 0
    for path in paths:
        decision = tree.get(path)
        if decision is None or decision["probability"] < threshold:
            continue
        override = apply_path_domain_override(path)
        if override is not None and override != decision["domain"]:
            changed += 1
    return changed


def _paired_accuracy_interval(tree_ok: Sequence[int], chat_ok: Sequence[int]) -> dict:
    """Percentile bootstrap of the paired accuracy difference, tree minus chat.

    ``bootstrap_interval`` in the evaluation module is AUC-typed (its statistic is
    ``roc_auc`` over binary labels); it cannot express a mean of per-unit 0/1
    outcomes, which is what a paired accuracy difference is. This reuses the same
    percentile-bootstrap policy (1000 resamples, seed 123, central 95%) over the
    paired difference instead of the AUC.
    """
    a = np.asarray(list(tree_ok), dtype=float)
    b = np.asarray(list(chat_ok), dtype=float)
    if a.size == 0:
        raise ValueError("paired accuracy interval needs at least one name")
    diff = a - b
    point = float(diff.mean())
    rng = np.random.default_rng(123)
    n = a.size
    draws = [float(diff[rng.integers(0, n, n)].mean()) for _ in range(1000)]
    lower, upper = np.percentile(draws, [2.5, 97.5])
    return {
        "point": point,
        "lower": float(lower),
        "upper": float(upper),
        "level": 0.95,
        "n_resamples": len(draws),
        "names": int(n),
        "excludes_zero": bool(lower > 0.0 or upper < 0.0),
    }


def _per_domain(rows: list[dict], chat_rows: dict[str, dict]) -> list[dict]:
    tally: dict[str, dict] = defaultdict(
        lambda: {"names": 0, "tree_correct": 0, "chat_correct": 0}
    )
    for row in rows:
        bucket = tally[row["label"]]
        bucket["names"] += 1
        bucket["tree_correct"] += row["derived_domain"] == row["label"]
        bucket["chat_correct"] += (
            chat_rows[row["name"]]["derived_domain"] == row["label"]
        )
    return [
        {"label": label, **values}
        | {
            "tree_accuracy": values["tree_correct"] / values["names"],
            "chat_accuracy": values["chat_correct"] / values["names"],
        }
        for label, values in sorted(tally.items())
    ]


def _bands(
    rows: list[dict], tree: dict, chat_rows: dict[str, dict], threshold: float
) -> list[dict]:
    out = []
    for band, low, high in PROBABILITY_BANDS:
        members = []
        for row in rows:
            probs = [
                tree[s["path"]]["probability"]
                for s in row["sources"]
                if s["path"] in tree
            ]
            if not probs:
                continue
            peak = max(probs)
            if low <= peak < high:
                members.append(row)
        tree_correct = sum(r["derived_domain"] == r["label"] for r in members)
        chat_correct = sum(
            chat_rows[r["name"]]["derived_domain"] == r["label"] for r in members
        )
        out.append(
            {
                "band": band,
                "names": len(members),
                "tree_correct": tree_correct,
                "tree_accuracy": tree_correct / len(members) if members else None,
                "chat_correct": chat_correct,
                "chat_accuracy": chat_correct / len(members) if members else None,
            }
        )
    return out


def _figure(payload: dict, path: Path) -> None:
    """Per-domain validation accuracy of both classifiers, one row per label."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    jev_colour, chat_colour, link_colour = "#3a78a8", "#52514e", "#bdbab4"
    rows = sorted(payload["per_domain"], key=lambda r: (-r["names"], r["label"]))
    total = payload["validation"]["tree"]["derived"]
    chat_total = payload["validation"]["chat"]["derived"]
    plt.style.use("data-ink")
    fig, ax = plt.subplots(figsize=(14, 0.52 * (len(rows) + 1) + 1.6))
    ys = list(range(len(rows) + 1))[::-1]
    entries = [
        (
            f"all validation names ({total['total']})",
            chat_total["accuracy"],
            total["accuracy"],
        )
    ] + [
        (f"{r['label']} ({r['names']})", r["chat_accuracy"], r["tree_accuracy"])
        for r in rows
    ]
    for y, (_, chat_acc, tree_acc) in zip(ys, entries, strict=True):
        ax.plot([chat_acc, tree_acc], [y, y], color=link_colour, lw=2.4, zorder=1)
        ax.scatter([chat_acc], [y], s=140, color=chat_colour, zorder=3, linewidths=0)
        ax.scatter([tree_acc], [y], s=140, color=jev_colour, zorder=3, linewidths=0)
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
        "two-level tree",
        color=jev_colour,
        ha="right",
        va="bottom",
        fontsize=20,
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="svg", metadata={"Date": None})


async def _run_gold() -> dict:
    """Exact-match accuracy of both classifiers on the 209-path gold set.

    The chat arm exercises the class the composer falls back to and the tree arm
    exercises the two-level Choice, so the pair measures the same routing the
    validation rule does. Both go through the benchmark's own entry point.
    """
    from imas_codex.graph.client import GraphClient
    from imas_codex.standard_names.benchmark_roles import run_classifier_bench

    gold = json.loads(GOLD_SET.read_text())
    out: dict[str, dict] = {}
    with GraphClient() as gc:
        for arm, model, use_tree in (
            ("chat", get_model("sn-classifier"), False),
            ("tree", get_model("sn-domain-tree"), True),
        ):
            report = await run_classifier_bench([model], gold, gc=gc, use_tree=use_tree)
            result = report.results[0]
            out[arm] = {
                "model": result.model,
                "accuracy": result.metrics.get("accuracy"),
                "correct": int(result.metrics.get("correct", 0)),
                "n": result.n,
                "spend_usd": result.cost,
            }
    return out


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="pilot sample size")
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=RESULTS)
    parser.add_argument("--figure", type=Path, default=None)
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="override the calibration-chosen threshold",
    )
    parser.add_argument(
        "--gold",
        action="store_true",
        help="run both classifiers on the gold set and merge the result",
    )
    args = parser.parse_args()

    if args.gold:
        payload = json.loads(args.out.read_text())
        payload["gold"] = await _run_gold()
        args.out.write_text(json.dumps(payload, sort_keys=True, indent=1) + "\n")
        print("GOLD " + json.dumps(payload["gold"], sort_keys=True), flush=True)
        return

    sheet = json.loads(LABELS.read_text())
    label_rows = sheet["reference"] + sheet["sample"]
    names = [row["name"] for row in label_rows]
    if len(names) != 170 or len(set(names)) != 170:
        raise RuntimeError("expected 170 distinct labelled names")
    baseline = json.loads(BASELINE.read_text())
    chat = _baseline_answers(baseline)
    chat_per_path = baseline["spend_usd"] / max(1, baseline["source_path_count"])

    with GraphClient() as gc:
        before_rows = _read_sources(gc, names)
        before = _snapshot(before_rows)
        print("BEFORE " + json.dumps(before, sort_keys=True), flush=True)

    if args.fresh:
        tree = await _run_tree(
            {row["path"]: {"id": row["path"]} for row in before_rows if row["path"]},
            args.cache,
            args.limit,
        )
        if args.limit is not None:
            print(
                f"PILOT paths={len(tree['decisions'])} "
                f"spend_usd={tree['spend_usd']:.6f} "
                f"estimate_all_usd="
                f"{tree['spend_usd'] / max(1, len(tree['decisions'])) * before['name_count']:.4f}",
                flush=True,
            )
            return
    else:
        tree = json.loads(args.cache.read_text())

    decisions = tree["decisions"]
    all_paths = sorted({row["path"] for row in before_rows if row["path"]})
    rows_by_name: dict[str, list[dict]] = defaultdict(list)
    for row in before_rows:
        rows_by_name[row["name"]].append(row)

    validation = [r for r in label_rows if r.get("split") == VALIDATION]
    calibration = [r for r in label_rows if r.get("split") == CALIBRATION]

    def _accuracy(rows: list[dict], threshold: float) -> float | None:
        derived = _derive(
            rows, _compose(decisions, chat, all_paths, threshold), rows_by_name
        )
        return _score(derived, "derived_domain")["accuracy"]

    sweep = []
    for t in THRESHOLD_GRID:
        acc = _accuracy(calibration, t)
        sweep.append({"threshold": t, "calibration_accuracy": acc})
    chosen = max(
        (row for row in sweep if row["calibration_accuracy"] is not None),
        key=lambda r: (r["calibration_accuracy"], r["threshold"]),
    )["threshold"]
    if args.threshold is not None:
        chosen = args.threshold

    tree_answers = _compose(decisions, chat, all_paths, chosen)
    chat_derived = _derive(validation, chat, rows_by_name)
    tree_derived = _derive(validation, tree_answers, rows_by_name)
    chat_rows = {row["name"]: row for row in chat_derived}
    tree_rows = {row["name"]: row for row in tree_derived}

    tree_score = _score(tree_derived, "derived_domain")
    chat_score = _score(chat_derived, "derived_domain")

    validation_names = {r["name"] for r in validation}
    val_paths = [
        row["path"]
        for row in before_rows
        if row["path"] and row["name"] in validation_names
    ]
    val_path_set = sorted(set(val_paths))
    fallback = [
        p
        for p in val_path_set
        if p not in decisions or decisions[p]["probability"] < chosen
    ]
    jev_per_path = tree["spend_usd"] / max(1, len(decisions))
    tree_val_spend = jev_per_path * len(val_path_set) + chat_per_path * len(fallback)
    chat_val_spend = chat_per_path * len(val_path_set)

    paired = _paired_accuracy_interval(
        [tree_rows[r["name"]]["derived_domain"] == r["label"] for r in validation],
        [chat_rows[r["name"]]["derived_domain"] == r["label"] for r in validation],
    )

    adopt = bool(
        tree_score["accuracy"] is not None
        and tree_score["accuracy"] >= chat_score["accuracy"] - 0.03
        and tree_val_spend < chat_val_spend
    )

    payload = {
        "model": tree["model"],
        "question_hash": hashlib.sha256(
            json.dumps(
                [build_category_question(), build_domain_question("general")],
                sort_keys=True,
            ).encode()
        ).hexdigest()[:16],
        "spend_usd": tree["spend_usd"],
        "source_path_count": len(all_paths),
        "classified_path_count": len(decisions),
        "unclassified_paths": sorted(set(all_paths) - set(decisions)),
        "chosen_threshold": chosen,
        "chat_share": len(fallback) / len(val_path_set) if val_path_set else None,
        "overrides_changed_after_confident_jev": _overrides_changed(
            decisions, all_paths, chosen
        ),
        "verdict": {
            "rule": (
                "adopt when validation derived accuracy >= chat accuracy - 0.03 "
                "and tree spend (jev + chat fallback) < chat spend on the same paths"
            ),
            "validation_tree_accuracy": tree_score["accuracy"],
            "validation_chat_accuracy": chat_score["accuracy"],
            "validation_tree_spend_usd": tree_val_spend,
            "validation_chat_spend_usd": chat_val_spend,
            "adopt": adopt,
        },
        "validation": {
            "tree": {"derived": tree_score},
            "chat": {"derived": chat_score},
        },
        "paired_accuracy_difference": paired,
        "calibration_sweep": sweep,
        "probability_bands": _bands(tree_derived, decisions, chat_rows, chosen),
        "per_domain": _per_domain(tree_derived, chat_rows),
        "spend": {
            "jev_spend_usd": tree["spend_usd"],
            "chat_cost_per_path_usd": chat_per_path,
            "validation_path_count": len(val_path_set),
            "chat_fallback_path_count": len(fallback),
        },
    }
    args.out.write_text(json.dumps(payload, sort_keys=True, indent=1) + "\n")
    if args.figure is not None:
        _figure(payload, args.figure)
    print("VERDICT " + json.dumps(payload["verdict"], sort_keys=True), flush=True)
    print("PAIRED " + json.dumps(paired, sort_keys=True), flush=True)
    print(
        f"SPEND_USD={tree['spend_usd']:.6f} "
        f"RESULT_SHA256={hashlib.sha256(args.out.read_bytes()).hexdigest()}",
        flush=True,
    )


if __name__ == "__main__":
    asyncio.run(main())
