"""Measure labelled-only nDCG@10 of the comparator block, embedding vs reranked order.

The comparator label sheet records, for each drawn DD source, the union of the
embedding-ordered top ten and the reranked top ten of the comparator block, every
candidate carrying its ``embedding_rank``, its ``rerank_rank`` over the pool and a
coordinator ``grade`` from 0 to 3. This driver scores each item twice with
scikit-learn's ``ndcg_score`` -- once against the embedding order and once against
the reranked order, both keyed on ``-rank`` so rank 1 is best -- over the item's
listed candidates. Every listed candidate is graded, so the metric sees only the
labelled union and never an ungraded name; the ideal ordering is therefore taken
over the whole graded candidate set. An item whose grades are all zero has no
defined nDCG and is excluded, with its count.

Measure:  mean nDCG@10 for each order, the mean per-item difference (rerank minus
embedding) with a 95 percent percentile bootstrap interval, win/tie/loss counts,
the same figures for the ``boundary_density`` family and for all other items, a
negative control that shuffles the grades within each item, and each item's two
scores. Read-only: the driver reads the sheet and writes only its results JSON.

Run::

    uv run python .../ssog-rerank-ndcg-run.py --sheet <sheet.json> --out <results.json>
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import ndcg_score

SEED = 123
N_RESAMPLES = 1000
TOP_K = 10
BOUNDARY_FAMILY = "boundary_density"
RUBRIC_COMMIT = "95231110a"


class InvalidCandidate(ValueError):
    """A candidate whose grade or ranks are not the integers the metric needs."""


def validate_candidate(item_path: str, candidate: dict[str, Any]) -> None:
    """Refuse a candidate whose grade is not an integer 0..3 or whose two ranks
    are not both positive integers.

    ``bool`` is rejected explicitly: ``isinstance(True, int)`` is true and a
    boolean grade would silently score as a 0 or 1.
    """
    grade = candidate.get("grade")
    if isinstance(grade, bool) or not isinstance(grade, int) or not 0 <= grade <= 3:
        raise InvalidCandidate(
            f"{item_path}: candidate {candidate.get('name')!r} has grade "
            f"{grade!r}, not an integer from 0 to 3"
        )
    for key in ("embedding_rank", "rerank_rank"):
        value = candidate.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise InvalidCandidate(
                f"{item_path}: candidate {candidate.get('name')!r} has {key} "
                f"{value!r}, not a positive integer"
            )


def load_items(path: Path) -> list[dict[str, Any]]:
    """Read the sheet and validate every candidate before any scoring."""
    sheet = json.loads(path.read_text())
    items = sheet["items"]
    for item in items:
        for candidate in item["candidates"]:
            validate_candidate(item.get("path", "?"), candidate)
    return items


def _ndcg(grades: Sequence[int], ranks: Sequence[int]) -> float:
    """nDCG@k for one item: relevance = grades, score = -rank (rank 1 is best)."""
    y_true = np.asarray([grades], dtype=float)
    y_score = -np.asarray([ranks], dtype=float)
    return float(ndcg_score(y_true, y_score, k=TOP_K))


def score_item(item: dict[str, Any]) -> dict[str, Any] | None:
    """Score one item in both orders; ``None`` when its grades are all zero."""
    candidates = item["candidates"]
    grades = [c["grade"] for c in candidates]
    if not any(g > 0 for g in grades):
        return None
    embedding = _ndcg(grades, [c["embedding_rank"] for c in candidates])
    reranked = _ndcg(grades, [c["rerank_rank"] for c in candidates])
    return {
        "path": item.get("path"),
        "family": item.get("family"),
        "ndcg_embedding": embedding,
        "ndcg_rerank": reranked,
        "diff": reranked - embedding,
    }


def bootstrap_ci(differences: Sequence[float]) -> tuple[float, float, float]:
    """95 percent percentile bootstrap interval for the mean difference.

    A plain unit bootstrap over items: item indices are resampled with
    replacement (``N_RESAMPLES`` resamples at ``SEED``) and the mean difference
    recomputed on each, so the interval is not copy-grouped across items.
    Returns ``(mean, ci_low, ci_high)``.
    """
    values = np.asarray(differences, dtype=float)
    rng = np.random.default_rng(SEED)
    n = len(values)
    draws = rng.integers(0, n, size=(N_RESAMPLES, n))
    means = values[draws].mean(axis=1)
    return (
        float(values.mean()),
        float(np.percentile(means, 2.5)),
        float(np.percentile(means, 97.5)),
    )


def summarise(scored: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Per-order means, the mean difference with its interval and win/tie/loss."""
    diffs = [row["diff"] for row in scored]
    mean, low, high = bootstrap_ci(diffs)
    wins = sum(1 for d in diffs if d > 1e-12)
    losses = sum(1 for d in diffs if d < -1e-12)
    return {
        "n": len(scored),
        "mean_ndcg_embedding": float(np.mean([r["ndcg_embedding"] for r in scored])),
        "mean_ndcg_rerank": float(np.mean([r["ndcg_rerank"] for r in scored])),
        "mean_diff": mean,
        "ci_low": low,
        "ci_high": high,
        "wins": wins,
        "ties": len(scored) - wins - losses,
        "losses": losses,
    }


def negative_control(items: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Shuffle grades within each item (seeded), then score the reranked order.

    The shuffled grades carry no relation to the reranked order, so the mean
    difference of the shuffled-rerank score against the unshuffled embedding
    score is what no signal looks like.
    """
    rng = random.Random(SEED)
    diffs: list[float] = []
    shuffled_scores: list[float] = []
    for item in items:
        candidates = item["candidates"]
        grades = [c["grade"] for c in candidates]
        if not any(g > 0 for g in grades):
            continue
        shuffled = list(grades)
        rng.shuffle(shuffled)
        embedding = _ndcg(grades, [c["embedding_rank"] for c in candidates])
        rerank_shuffled = _ndcg(shuffled, [c["rerank_rank"] for c in candidates])
        diffs.append(rerank_shuffled - embedding)
        shuffled_scores.append(rerank_shuffled)
    return {
        "description": "grades shuffled within each item, reranked order rescored",
        "seed": SEED,
        "n": len(diffs),
        "mean_ndcg_shuffled_rerank": float(np.mean(shuffled_scores)),
        "mean_diff_vs_embedding": float(np.mean(diffs)),
    }


def build_results(sheet_path: Path, items: list[dict[str, Any]]) -> dict[str, Any]:
    scored: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for item in items:
        row = score_item(item)
        if row is None:
            excluded.append(
                {
                    "path": item.get("path"),
                    "reason": "all grades zero; nDCG undefined",
                    "n_candidates": len(item["candidates"]),
                }
            )
        else:
            scored.append(row)

    scored.sort(key=lambda r: r["diff"])
    boundary = [r for r in scored if r["family"] == BOUNDARY_FAMILY]
    other = [r for r in scored if r["family"] != BOUNDARY_FAMILY]

    return {
        "sheet": str(sheet_path),
        "rubric_commit": RUBRIC_COMMIT,
        "k": TOP_K,
        "n_items": len(items),
        "n_scored": len(scored),
        "n_excluded": len(excluded),
        "excluded_items": excluded,
        "bootstrap": {
            "resamples": N_RESAMPLES,
            "seed": SEED,
            "grouped": False,
            "note": "plain unit bootstrap over items; not copy-grouped",
        },
        "overall": summarise(scored),
        "strata": {
            BOUNDARY_FAMILY: summarise(boundary),
            "other": summarise(other),
        },
        "negative_control": negative_control(items),
        "items": scored,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve().parent
    parser.add_argument("--sheet", default=str(here / "ssog-rerank-label-sheet.json"))
    parser.add_argument("--out", default=str(here / "ssog-rerank-ndcg-results.json"))
    args = parser.parse_args(argv)

    items = load_items(Path(args.sheet))
    results = build_results(Path(args.sheet), items)
    out = Path(args.out)
    out.write_text(json.dumps(results, indent=2) + "\n")

    overall = results["overall"]
    print(
        f"wrote {out} :: scored={results['n_scored']} excluded={results['n_excluded']} "
        f"mean_embedding={overall['mean_ndcg_embedding']:.6f} "
        f"mean_rerank={overall['mean_ndcg_rerank']:.6f} "
        f"mean_diff={overall['mean_diff']:.6f} "
        f"ci=[{overall['ci_low']:.6f},{overall['ci_high']:.6f}] "
        f"w/t/l={overall['wins']}/{overall['ties']}/{overall['losses']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
