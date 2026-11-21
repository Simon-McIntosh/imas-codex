"""Evaluation primitives for comparing judge and classifier arms.

Metric primitives are scikit-learn calls (``roc_auc_score``, ``accuracy_score``,
``f1_score``, ``confusion_matrix``, ``cohen_kappa_score``).  What this module
owns is what the library lacks: copy-group-aware resampling, the paired
difference over canonical units, an operating point chosen on the calibration
split and refused on that split, and the stability figures a judge is judged
by.

It imports nothing from the graph, the LLM layer or the standard-names package,
so discovery and standard names share one copy.  Arrays are the interface: a
caller reduces its own records to labels and per-row scores before calling in,
which is what keeps the module dependency-free.

A labelled fixture row carries exactly these fields, and
:func:`load_labelled_fixture` refuses a row that omits any of them:

``unit``            the item the judgement was made on.
``label``           the ground-truth label (0 or 1).
``canonical_unit``  the representative item a family of copies reduces to.
``copy_group``      the cluster of near-duplicate copies that move together.
``label_source``    where the label came from, named so overlap is checkable.
``source_evidence`` the evidence the label was read from.
``split``           ``calibration`` or ``validation``.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

# The fixture row contract.  A loader caller writes rows against exactly these
# names; the loader names any that are missing, with the row it found them in.
FIXTURE_FIELDS = (
    "unit",
    "label",
    "canonical_unit",
    "copy_group",
    "label_source",
    "source_evidence",
    "split",
)

# The two splits a fixture row may belong to.  An operating point is chosen on
# one and reported on the other.
CALIBRATION = "calibration"
VALIDATION = "validation"
SPLITS = (CALIBRATION, VALIDATION)


def _floats(values: Sequence[float]) -> np.ndarray:
    return np.asarray(values, dtype=float)


def _ints(values: Sequence[int]) -> np.ndarray:
    return np.asarray(values, dtype=int)


def load_labelled_fixture(path: str | Path) -> list[dict[str, Any]]:
    """Load a labelled fixture and enforce its row contract.

    Reads a JSON list of objects and returns it.  A row missing any field of
    :data:`FIXTURE_FIELDS` is refused by name and row index, so a malformed
    fixture fails loudly at load rather than as a key error deep in a metric.
    """
    rows = json.loads(Path(path).read_text())
    if not isinstance(rows, list):
        raise ValueError(
            f"fixture {path} must be a JSON list, got {type(rows).__name__}"
        )
    loaded: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"fixture row {index} is not an object")
        for field_name in FIXTURE_FIELDS:
            if field_name not in row:
                raise ValueError(f"fixture row {index} is missing field {field_name!r}")
        if row["split"] not in SPLITS:
            raise ValueError(
                f"fixture row {index} has split {row['split']!r}, "
                f"expected one of {SPLITS}"
            )
        loaded.append(dict(row))
    return loaded


def roc_auc(labels: Sequence[float], scores: Sequence[float]) -> float:
    """Rank AUC, via scikit-learn's ``roc_auc_score``.

    The one AUC owner: a hand-rolled pairwise loop and this call agree on
    ties (each contributes 0.5), so folding onto it changes no value.
    """
    return float(roc_auc_score(_floats(labels), _floats(scores)))


def paired_auc(
    labels: Sequence[float],
    scores_a: Sequence[float],
    scores_b: Sequence[float],
) -> dict[str, float]:
    """AUC for two arms over the same units, and their difference.

    Both arms are scored on the same labels in the same order, so the
    difference is paired rather than two independent point estimates.
    """
    y = _floats(labels)
    auc_a = roc_auc(labels, scores_a)
    auc_b = roc_auc(labels, scores_b)
    return {
        "auc_a": auc_a,
        "auc_b": auc_b,
        "difference": auc_a - auc_b,
        "n": int(y.size),
    }


def _group_indices(copy_groups: Sequence[Any] | None, n: int) -> list[np.ndarray]:
    """Index sets to resample: one per copy group, or one per observation.

    Copy groups keep near-duplicate copies together — resampling them
    independently would count the same content many times and narrow the
    interval.  With no groups given, every observation is its own group, so the
    resample is a plain unit bootstrap.
    """
    if copy_groups is None:
        return [np.array([i]) for i in range(n)]
    buckets: dict[Any, list[int]] = {}
    for i, key in enumerate(copy_groups):
        buckets.setdefault(key, []).append(i)
    return [np.array(idx) for idx in buckets.values()]


def bootstrap_interval(
    labels: Sequence[float],
    scores: Sequence[float],
    *,
    scores_b: Sequence[float] | None = None,
    copy_groups: Sequence[Any] | None = None,
    n_resamples: int = 1000,
    seed: int = 123,
    level: float = 0.95,
    statistic: Callable[[np.ndarray, np.ndarray], float] | None = None,
) -> dict[str, Any]:
    """Percentile bootstrap interval over copy groups.

    Resamples copy groups with replacement and recomputes the statistic under
    ``labels``/``scores``.  Without ``scores_b`` the interval is for a single
    arm's AUC; with it, for the paired difference ``AUC(a) - AUC(b)`` over the
    same units. A custom ``statistic(labels, scores)`` replaces AUC in both
    arms. It may return NaN for an undefined ratio: undefined draws are
    resampled, and an undefined point returns null bounds with zero draws.

    A resample that draws only one label class has no defined AUC, so it is
    redrawn; ``n_resamples`` is the number of usable draws, which equals the
    requested count whenever both classes are representable.
    """
    y = _floats(labels)
    va = _floats(scores)
    vb = _floats(scores_b) if scores_b is not None else None
    if y.size == 0:
        raise ValueError("bootstrap_interval needs at least one observation")
    if va.shape != y.shape or (vb is not None and vb.shape != y.shape):
        raise ValueError("bootstrap_interval needs aligned labels and scores")
    if copy_groups is not None and len(copy_groups) != y.size:
        raise ValueError("bootstrap_interval needs one copy group per observation")
    if n_resamples < 2 or not 0 < level < 1:
        raise ValueError(
            "bootstrap_interval needs at least two draws and 0 < level < 1"
        )
    auc_default = statistic is None
    measure = roc_auc if statistic is None else statistic
    if auc_default and np.unique(y).size < 2:
        raise ValueError("bootstrap_interval needs both label classes present")

    def paired_statistic(
        labels_in: np.ndarray, va_in: np.ndarray, vb_in: np.ndarray | None
    ) -> float:
        if vb_in is None:
            return measure(labels_in, va_in)
        return measure(labels_in, va_in) - measure(labels_in, vb_in)

    point = paired_statistic(y, va, vb)
    if not np.isfinite(point):
        return {
            "point": None,
            "lower": None,
            "upper": None,
            "level": level,
            "n_resamples": 0,
            "seed": seed,
        }
    groups = _group_indices(copy_groups, y.size)
    rng = np.random.default_rng(seed)
    draws: list[float] = []
    attempts = 0
    while len(draws) < n_resamples and attempts < n_resamples * 100:
        attempts += 1
        picks = rng.integers(0, len(groups), len(groups))
        idx = np.concatenate([groups[p] for p in picks])
        y_bs = y[idx]
        if auc_default and np.unique(y_bs).size < 2:
            continue
        draw = paired_statistic(y_bs, va[idx], None if vb is None else vb[idx])
        if np.isfinite(draw):
            draws.append(draw)

    if len(draws) < 2:
        raise ValueError("bootstrap_interval could not draw defined resamples")
    tail = (1.0 - level) / 2.0
    lower, upper = np.percentile(draws, [tail * 100.0, (1.0 - tail) * 100.0])
    return {
        "point": point,
        "lower": float(lower),
        "upper": float(upper),
        "level": level,
        "n_resamples": len(draws),
        "seed": seed,
    }


def classification_metrics(
    labels: Sequence[int],
    predictions: Sequence[int],
) -> dict[str, Any]:
    """Accuracy, macro-F1 and a confusion table for a classified arm."""
    y = _ints(labels)
    p = _ints(predictions)
    return {
        "accuracy": round(float(accuracy_score(y, p)), 4),
        "macro_f1": round(float(f1_score(y, p, average="macro", zero_division=0)), 4),
        "confusion": confusion_matrix(y, p).tolist(),
        "n": int(y.size),
    }


def cohen_kappa(labels: Sequence[int], ratings: Sequence[int]) -> float:
    """Cohen's kappa between two label sets, for inter-rater agreement."""
    return float(cohen_kappa_score(_ints(labels), _ints(ratings)))


def choose_operating_point(
    rows: Sequence[Mapping[str, Any]],
    *,
    cost_false_positive: float,
    cost_false_negative: float,
    report_split: str = VALIDATION,
) -> dict[str, Any]:
    """Choose a score threshold on the calibration split.

    The threshold minimises ``cost_false_positive * FP + cost_false_negative *
    FN`` over the candidate thresholds present in the calibration rows (a
    positive is ``score >= threshold``; the lowest-cost threshold wins a tie).
    The chosen threshold is reported together with the metrics of the
    ``report_split`` rows only: asking to report on the calibration split the
    threshold was fitted on is refused, because those figures are the ones the
    choice already saw.
    """
    if report_split not in SPLITS:
        raise ValueError(f"report_split {report_split!r} is not one of {SPLITS}")
    if report_split == CALIBRATION:
        raise ValueError(
            "choose_operating_point refuses to report on the calibration split "
            "it chose on; report on the validation split instead"
        )
    calibration = [r for r in rows if r["split"] == CALIBRATION]
    if not calibration:
        raise ValueError("choose_operating_point needs calibration rows to choose on")
    report = [r for r in rows if r["split"] == report_split]

    labels = _ints([r["label"] for r in calibration])
    scores = _floats([r["score"] for r in calibration])
    thresholds = sorted({float(s) for s in scores})
    best_threshold = thresholds[0]
    best_cost = float("inf")
    for threshold in thresholds:
        predicted = scores >= threshold
        false_positive = int(np.sum((labels == 0) & predicted))
        false_negative = int(np.sum((labels == 1) & ~predicted))
        cost = (
            cost_false_positive * false_positive + cost_false_negative * false_negative
        )
        if cost < best_cost:
            best_cost = cost
            best_threshold = threshold

    result: dict[str, Any] = {
        "threshold": best_threshold,
        "calibration_cost": round(best_cost, 4),
        "split": report_split,
        "n_calibration": int(labels.size),
        "n": len(report),
    }
    if report:
        result["report"] = classification_metrics(
            [r["label"] for r in report],
            [1 if float(r["score"]) >= best_threshold else 0 for r in report],
        )
    return result


def frozen_state_variation(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Per-field change counts and ranges across repeats of a frozen state.

    ``records`` is a sequence of ``{"digest": str, "fields": {name: value}}``.
    Rows sharing a digest are repeats of one immutable state, so any field that
    differs between them is a change the model made without the input changing.
    Reports, per field, how many such state groups varied it, the total number
    of differing values, and the observed range of numeric values.
    """
    groups: dict[Any, list[Mapping[str, Any]]] = {}
    for record in records:
        groups.setdefault(record.get("digest"), []).append(record.get("fields", {}))

    repeats = [fields for fields in groups.values() if len(fields) > 1]
    fields_seen: list[str] = []
    for state_fields in repeats:
        for name in state_fields[0]:
            if name not in fields_seen:
                fields_seen.append(name)

    per_field: dict[str, Any] = {}
    for name in fields_seen:
        changed_states = 0
        differing = 0
        numeric: list[float] = []
        for state_fields in repeats:
            values = [fields.get(name) for fields in state_fields]
            distinct = {_hashable(v) for v in values}
            if len(distinct) > 1:
                changed_states += 1
                differing += len(distinct) - 1
            numeric.extend(
                v
                for v in values
                if isinstance(v, (int, float)) and not isinstance(v, bool)
            )
        entry: dict[str, Any] = {
            "states": len(repeats),
            "states_changed": changed_states,
            "change_rate": round(changed_states / len(repeats), 4) if repeats else 0.0,
            "differing_values": differing,
        }
        if numeric:
            entry["min"] = min(numeric)
            entry["max"] = max(numeric)
        per_field[name] = entry

    return {
        "digests": len(groups),
        "repeated_states": len(repeats),
        "fields": per_field,
    }


def option_order_stability(
    original: Sequence[Any],
    permuted: Sequence[Any],
) -> dict[str, Any]:
    """Flip rate when a Choice is re-asked with its options permuted.

    Each sequence holds the identity of the option chosen for every item, so a
    permutation that does not change the answer still matches and a different
    identity is a flip.  A judge that leans to the first option in a list, rather
    than to the option itself, shows up here as a high flip rate — the signal
    this figure exists to expose.
    """
    if len(original) != len(permuted):
        raise ValueError(
            f"option_order_stability needs equal lengths, got {len(original)} and "
            f"{len(permuted)}"
        )
    n = len(original)
    flips = sum(1 for a, b in zip(original, permuted, strict=True) if a != b)
    return {
        "n": n,
        "flips": flips,
        "flip_rate": round(flips / n, 4) if n else 0.0,
    }


def cost_latency_summary(calls: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Cost and wall-time summary from reported ``cost`` and per-call latency.

    ``calls`` is a sequence of ``{"cost": float, "latency_s": float}`` for one
    arm.  Cost comes from the endpoint's reported ``usage.cost``; latency is
    the caller's measured wall time.
    """
    costs = [float(c["cost"]) for c in calls]
    latencies = [float(c["latency_s"]) for c in calls]
    n = len(calls)
    return {
        "n": n,
        "total_cost": round(sum(costs), 6),
        "mean_cost": round(sum(costs) / n, 6) if n else 0.0,
        "total_latency_s": round(sum(latencies), 6),
        "mean_latency_s": round(sum(latencies) / n, 6) if n else 0.0,
        "median_latency_s": round(float(np.median(latencies)), 6) if n else 0.0,
    }


def _hashable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    return repr(value)
