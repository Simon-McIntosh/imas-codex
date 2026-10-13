"""Measure how well Jev's link judgment separates good name links from bad.

Driver for the ``ssog-link-auc`` node. It reads the coordinator-labelled link
sheet (``ssog-link-label-sheet.json``) and reports the rank AUC of Jev's
per-link probability against the coordinator's label, over three populations
(all links, non-self-links, self-links), with copy-grouped bootstrap intervals.
It also reports the deterministic self-link baseline (its score is
``1 - is_self_link``) and the paired difference ``AUC(Jev) - AUC(baseline)``,
and a shuffled-label negative control whose interval should contain 0.5.

The metrics are :func:`imas_codex.discovery.base.evaluation.roc_auc` and
:func:`imas_codex.discovery.base.evaluation.bootstrap_interval`; this driver
owns the population split, the label contract and the report, not the AUC.

Usage::

    python ssog-link-auc-run.py --out ssog-link-auc-results.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from imas_codex.discovery.base.evaluation import bootstrap_interval, roc_auc

SHEET = Path(__file__).with_name("ssog-link-label-sheet.json")
LABELS_COMMIT = "9867930cd"
COPY_GROUP_FIELD = "documenting_name"
N_RESAMPLES = 1000
SEED = 123
LEVEL = 0.95
DISAGREEMENTS = 10


def _load_rows(path: Path) -> list[dict[str, Any]]:
    """Read the sheet and enforce the label contract.

    A row whose ``label`` is not 0 or 1 is refused by name, because a label
    outside the contract is a labelling fault rather than a link to score.
    Rows whose ``jev_probability`` is null are excluded later and counted.
    """
    rows = json.loads(path.read_text())
    if not isinstance(rows, list):
        raise ValueError(f"sheet {path} must be a JSON list")
    for index, row in enumerate(rows):
        label = row.get("label")
        if label not in (0, 1):
            raise ValueError(
                f"sheet row {index} has label {label!r}, expected 0 or 1"
            )
    return rows


def _scored(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """Rows carrying a Jev probability, and the count excluded for a null one."""
    kept = [row for row in rows if row.get("jev_probability") is not None]
    return kept, len(rows) - len(kept)


def _auc_population(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Jev's AUC and copy-grouped interval over one population.

    A population carrying one label class has no defined AUC, so it is reported
    with its counts and ``defined: false`` rather than raising.
    """
    labels = [int(row["label"]) for row in rows]
    scores = [float(row["jev_probability"]) for row in rows]
    groups = [row[COPY_GROUP_FIELD] for row in rows]
    positives = sum(labels)
    negatives = len(labels) - positives
    entry: dict[str, Any] = {
        "n": len(rows),
        "positives": positives,
        "negatives": negatives,
    }
    if len({*labels}) < 2:
        entry["defined"] = False
        return entry
    entry["defined"] = True
    entry.update(
        bootstrap_interval(
            labels,
            scores,
            copy_groups=groups,
            n_resamples=N_RESAMPLES,
            seed=SEED,
            level=LEVEL,
        )
    )
    entry["auc"] = entry["point"]
    return entry


def _baseline(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """The deterministic self-link baseline ``1 - is_self_link`` over all rows."""
    labels = [int(row["label"]) for row in rows]
    scores = [1.0 - float(bool(row["is_self_link"])) for row in rows]
    groups = [row[COPY_GROUP_FIELD] for row in rows]
    entry = bootstrap_interval(
        labels,
        scores,
        copy_groups=groups,
        n_resamples=N_RESAMPLES,
        seed=SEED,
        level=LEVEL,
    )
    entry["auc"] = entry["point"]
    return entry


def _paired_difference(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """The paired ``AUC(Jev) - AUC(baseline)`` over all rows, per copy group."""
    labels = [int(row["label"]) for row in rows]
    jev = [float(row["jev_probability"]) for row in rows]
    baseline = [1.0 - float(bool(row["is_self_link"])) for row in rows]
    groups = [row[COPY_GROUP_FIELD] for row in rows]
    entry = bootstrap_interval(
        labels,
        jev,
        scores_b=baseline,
        copy_groups=groups,
        n_resamples=N_RESAMPLES,
        seed=SEED,
        level=LEVEL,
    )
    entry["difference"] = entry["point"]
    entry["auc_jev"] = roc_auc(labels, jev)
    entry["auc_baseline"] = roc_auc(labels, baseline)
    return entry


def _negative_control(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Jev's AUC over all rows with the label column shuffled (seed 123).

    The labels are permuted across rows while the scores and copy groups stay,
    so any separation is chance; the interval should contain 0.5.
    """
    labels = [int(row["label"]) for row in rows]
    scores = [float(row["jev_probability"]) for row in rows]
    groups = [row[COPY_GROUP_FIELD] for row in rows]
    shuffled = [int(v) for v in np.random.default_rng(SEED).permutation(labels)]
    entry = bootstrap_interval(
        shuffled,
        scores,
        copy_groups=groups,
        n_resamples=N_RESAMPLES,
        seed=SEED,
        level=LEVEL,
    )
    entry["auc"] = roc_auc(shuffled, scores)
    entry["contains_0_5"] = entry["lower"] <= 0.5 <= entry["upper"]
    return entry


def _disagreements(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """The rows Jev is most wrong about, each kind, highest- and lowest-ranked.

    A label-0 link Jev scores near 1 is a wrong link he would let through; a
    label-1 link he scores near 0 is a good link he would block.
    """

    def brief(row: dict[str, Any]) -> dict[str, Any]:
        return {
            "documenting_name": row[COPY_GROUP_FIELD],
            "link_text": row["link_text"],
            "target_name": row["target_name"],
            "jev_probability": float(row["jev_probability"]),
            "is_self_link": bool(row["is_self_link"]),
        }

    negatives = sorted(
        (r for r in rows if r["label"] == 0),
        key=lambda r: float(r["jev_probability"]),
        reverse=True,
    )
    positives = sorted(
        (r for r in rows if r["label"] == 1),
        key=lambda r: float(r["jev_probability"]),
    )
    return {
        "label_0_highest_probability": [brief(r) for r in negatives[:DISAGREEMENTS]],
        "label_1_lowest_probability": [brief(r) for r in positives[:DISAGREEMENTS]],
    }


def measure(sheet: Path) -> dict[str, Any]:
    rows = _load_rows(sheet)
    scored, excluded = _scored(rows)
    non_self = [r for r in scored if not r["is_self_link"]]
    self_links = [r for r in scored if r["is_self_link"]]

    pop = {
        "all": _auc_population(scored),
        "non_self_links": _auc_population(non_self),
        "self_links": _auc_population(self_links),
    }
    return {
        "sheet": sheet.name,
        "labels_commit": LABELS_COMMIT,
        "copy_group_field": COPY_GROUP_FIELD,
        "n_rows": len(rows),
        "n_scored": len(scored),
        "n_excluded_null_probability": excluded,
        "n_copy_groups": len({r[COPY_GROUP_FIELD] for r in scored}),
        "n_resamples": N_RESAMPLES,
        "seed": SEED,
        "level": LEVEL,
        "jev": pop,
        "baseline_self_link": _baseline(scored),
        "paired_difference_jev_minus_baseline": _paired_difference(scored),
        "disagreements": _disagreements(scored),
        "negative_control_shuffled_label": _negative_control(scored),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sheet", type=Path, default=SHEET)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    results = measure(args.sheet)
    args.out.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")

    print(
        f"rows {results['n_rows']}  scored {results['n_scored']}  "
        f"excluded {results['n_excluded_null_probability']}  "
        f"groups {results['n_copy_groups']}"
    )
    for name, entry in results["jev"].items():
        if entry["defined"]:
            print(
                f"  jev {name}: AUC {entry['auc']:.4f} "
                f"[{entry['lower']:.4f}, {entry['upper']:.4f}]  "
                f"n={entry['n']} pos={entry['positives']} neg={entry['negatives']}"
            )
        else:
            print(f"  jev {name}: AUC undefined  n={entry['n']}")
    base = results["baseline_self_link"]
    diff = results["paired_difference_jev_minus_baseline"]
    print(
        f"  baseline self-link: AUC {base['auc']:.4f} "
        f"[{base['lower']:.4f}, {base['upper']:.4f}]"
    )
    print(
        f"  paired jev-baseline: {diff['difference']:+.4f} "
        f"[{diff['lower']:.4f}, {diff['upper']:.4f}]"
    )
    nc = results["negative_control_shuffled_label"]
    print(
        f"  shuffled-label control: AUC {nc['auc']:.4f} "
        f"[{nc['lower']:.4f}, {nc['upper']:.4f}]  contains 0.5: {nc['contains_0_5']}"
    )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()