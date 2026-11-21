"""Rate matching of the calibration evidence against the panel's accept rate."""

from __future__ import annotations

import json
import lzma
from pathlib import Path
from typing import Any

import numpy as np

from iter_codex.standard_names.defaults import DEFAULT_MIN_SCORE


class ReceiptUnavailable(RuntimeError):
    """A receipt named by ``source_receipts`` is missing or unreadable."""


def _read_json(path: Path) -> Any:
    if not path.exists():
        raise ReceiptUnavailable(f"calibration receipt not found: {path}")
    try:
        if path.suffix == ".xz":
            return json.loads(lzma.decompress(path.read_bytes()))
        return json.loads(path.read_text())
    except (OSError, ValueError, lzma.LZMAError) as exc:
        raise ReceiptUnavailable(
            f"calibration receipt not readable: {path}: {type(exc).__name__}: {exc}"
        ) from exc


def matched_cutoff(scores, panel_scores, *, min_score=DEFAULT_MIN_SCORE):
    """Match the inclusive score predicate to the panel, before any veto.

    An unresolved score tie is refused rather than silently changing the target
    rate or using golden labels to break it. Validation never enters this fit.
    """
    values = np.asarray(scores, dtype=float)
    panel = np.asarray(panel_scores, dtype=float)
    if values.ndim != 1 or values.size == 0 or values.shape != panel.shape:
        raise ValueError("matched cutoff requires nonempty paired scores")
    if not np.all(np.isfinite(values)) or np.any((values < 0) | (values > 1)):
        raise ValueError("Jev scores must be finite and in [0, 1]")
    if not np.all(np.isfinite(panel)) or np.any((panel < 0) | (panel > 1)):
        raise ValueError("panel scores must be finite and in [0, 1]")
    target = int(np.count_nonzero(panel >= min_score))
    cutoff = (
        float(np.sort(values)[-target])
        if target
        else float(np.nextafter(values.max(), np.inf))
    )
    actual = int(np.count_nonzero(values >= cutoff))
    if actual != target or not 0 <= cutoff <= 1:
        raise ValueError(
            f"no inclusive cutoff matches {target}/{len(values)} accepts: "
            f"boundary score {cutoff} yields {actual}; score tie or endpoint"
        )
    return {
        "cutoff": cutoff,
        "n": len(values),
        "panel_accepts": target,
        "pre_veto_accepts": actual,
        "panel_min_score": min_score,
    }


METRICS = (
    "accept_rate",
    "accept_precision",
    "accept_recall",
    "refine_precision",
    "refine_recall",
    "false_refine_rate",
)


def _metric_counts(labels, accepted):
    y = np.asarray(labels, dtype=bool)
    a = np.asarray(accepted, dtype=bool)
    tp, fp = int(np.sum(y & a)), int(np.sum(~y & a))
    fn, tn = int(np.sum(y & ~a)), int(np.sum(~y & ~a))
    return dict(
        zip(
            METRICS,
            (
                (tp + fp, len(y)),
                (tp, tp + fp),
                (tp, tp + fn),
                (tn, tn + fn),
                (tn, tn + fp),
                (fn, tp + fn),
            ),
            strict=True,
        )
    )


def _metric(labels, accepted, key):
    numerator, denominator = _metric_counts(labels, accepted)[key]
    return numerator / denominator if denominator else float("nan")
