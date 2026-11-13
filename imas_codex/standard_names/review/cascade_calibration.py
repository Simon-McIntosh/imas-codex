"""Rate matching and explicit-input analysis of review calibration evidence."""

from __future__ import annotations

import json
import lzma
import math
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np

from imas_codex.standard_names.defaults import DEFAULT_MIN_SCORE

THRESHOLDS_PATH = Path(__file__).with_name("cascade_thresholds.json")
_REPO = Path(__file__).resolve().parents[3]
_GOLDEN_DIR = _REPO / "docs/evidence/fragments/sn-judgment-evaluation"
GOLDEN_NAME = _GOLDEN_DIR / "golden-name-acceptable.json"
GOLDEN_DOCS = _GOLDEN_DIR / "golden-documentation-acceptable.json"


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


def _labels(path: Path) -> dict[str, dict[str, Any]]:
    """Read the frozen defensible split: label and copy group per canonical name."""
    rows = _read_json(path)
    if not isinstance(rows, list):
        raise ReceiptUnavailable(f"golden fixture is not a list: {path}")
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        name = row.get("canonical_unit")
        if name is None:
            continue
        out.setdefault(
            name,
            {
                "label": int(row["label"]),
                "split": row["split"],
                "copy_group": row.get("copy_group", name),
            },
        )
    return out


@contextmanager
def _cached_conventions():
    """Resolve the grammar registry once for a whole calibration pass.

    The house-style and documentation-gate audits each import and re-read the
    convention registry on every call, which dominates the pass at about a
    second per item. The registry is frozen for the duration of a pass, so one
    cached read answers every item identically.
    """
    from functools import lru_cache

    from imas_standard_names.grammar import terms

    original = terms.load_conventions
    terms.load_conventions = lru_cache(maxsize=1)(original)  # type: ignore[assignment]
    try:
        yield
    finally:
        terms.load_conventions = original  # type: ignore[assignment]


def _name_axis(
    receipts_path: Path, golden: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build calibration and validation items for the name axis."""
    from imas_codex.standard_names.review.checks import load_review_checks
    from imas_codex.standard_names.review.jev_review import (
        check_failure_probability,
        name_question_table,
    )

    records: dict[str, dict[str, Any]] = {}
    if not receipts_path.exists():
        raise ReceiptUnavailable(f"calibration receipt not found: {receipts_path}")
    for line in receipts_path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("mode") == "base" and record.get("answered"):
            records[record["name"]] = record

    checks = [check for check in load_review_checks()[0] if check.axis == "name"]
    table = name_question_table(checks)
    questions = list(table)
    dimensions = [
        key.split(":", 1)[1] for key in questions if key.startswith("dimension:")
    ]
    accepted = {
        dim: next(iter(table[f"dimension:{dim}"]["criteria"])) for dim in dimensions
    }

    fixture = _labels(golden)
    items: list[dict[str, Any]] = []
    for name, meta in fixture.items():
        record = records.get(name)
        if record is None:
            continue
        answers = record["answers"]
        values: dict[str, float] = {}
        complete = True
        for key in questions:
            if key.startswith("dimension:"):
                answer = answers.get(key)
                if not answer:
                    complete = False
                    break
                values[key] = float(
                    (answer.get("probabilities") or {}).get(
                        accepted[key.split(":", 1)[1]], 0.0
                    )
                )
            else:
                check = next(c for c in checks if c.id == key.split(":", 1)[1])
                if check.question is None:
                    continue
                answer = answers.get(key)
                if not answer:
                    complete = False
                    break
                if check.question.kind.value == "noul":
                    values[key] = 1.0 - float(answer["noul"])
                else:
                    values[key] = float(
                        (answer.get("probabilities") or {}).get(
                            check.question.options[0], 0.0
                        )
                    )
        if not complete:
            continue
        check_fail = {
            check.id: check_failure_probability(check, answers.get(f"check:{check.id}"))
            for check in checks
            if check.question is not None
        }
        items.append(
            {
                "id": name,
                "label": meta["label"],
                "split": meta["split"],
                "copy_group": meta["copy_group"],
                "values": values,
                "check_fail": check_fail,
                "contradiction": False,
                "judgment": {
                    "categories": {
                        dim: answers[f"dimension:{dim}"] for dim in dimensions
                    },
                    "checks": {
                        key: {"answered": True, "failure_probability": value}
                        for key, value in check_fail.items()
                    },
                },
            }
        )
    fit = _fit_name_logistic(items, questions)
    for item in items:
        item["score"] = _logistic(fit, item["values"], questions)
    context = {"questions": questions, "fit": fit}
    return items, context


def _logistic(fit: Mapping[str, Any], values: Mapping[str, float], questions) -> float:
    linear = fit["intercept"] + sum(
        fit["weights"][key] * values.get(key, 0.0) for key in questions
    )
    return 1.0 / (1.0 + math.exp(-_clamp(linear)))


def _clamp(value: float) -> float:
    return max(-50.0, min(50.0, value))


def _fit_name_logistic(items: Sequence[Mapping[str, Any]], questions) -> dict[str, Any]:
    """L2 logistic with the regularisation chosen by grouped CV on calibration."""
    from sklearn import set_config
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.model_selection import GroupKFold

    set_config(enable_metadata_routing=True)
    rows = [
        item
        for item in items
        if item["split"] == "calibration"
        and all(item["values"].get(key) is not None for key in questions)
    ]
    features = np.asarray(
        [[row["values"][key] for key in questions] for row in rows], dtype=float
    )
    labels = np.asarray([row["label"] for row in rows], dtype=int)
    groups = np.asarray([row["copy_group"] for row in rows])
    model = LogisticRegressionCV(
        Cs=[0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0],
        cv=GroupKFold(n_splits=5),
        l1_ratios=(0.0,),
        solver="lbfgs",
        scoring="roc_auc",
        max_iter=5000,
        random_state=85,
        use_legacy_attributes=True,
    )
    model.fit(features, labels, groups=groups)
    return {
        "intercept": float(model.intercept_[0]),
        "weights": {
            key: float(coef)
            for key, coef in zip(questions, model.coef_[0], strict=True)
        },
        "regularization": float(model.C_[0]),
    }


def _docs_axis(
    receipts_path: Path, golden: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build calibration and validation items for the documentation axis."""
    from imas_codex.standard_names.review import jev_review
    from imas_codex.standard_names.review.checks import load_review_checks

    frozen = _read_json(receipts_path)
    checks = [check for check in load_review_checks()[0] if check.axis == "docs"]
    fixture = _labels(golden)
    items: list[dict[str, Any]] = []
    with _cached_conventions():
        for split in ("calibration", "validation"):
            for name, item in frozen[split]["items"].items():
                meta = fixture.get(name)
                receipt = frozen["receipts"].get(f"base:{name}")
                if meta is None or receipt is None:
                    continue
                state = receipt["state"]
                answers = receipt["answers"]
                audit = jev_review.build_docs_audit_report(
                    item,
                    catalog_names=[item],
                    known_ids={str(item["id"])},
                )
                score = jev_review.score_docs_review(
                    state, checks, answers, audit_report=audit
                )
                if score is None:
                    continue
                # The live path folds a claim's per-binding readings by support:
                # a claim contradicts only when no binding supports it. The
                # replay reads the same fold the score and the route read, so a
                # row cannot be vetoed here by one narrow binding that the
                # deciding pass accepts.
                docs_review = jev_review.aggregate_docs_review(
                    state, checks, answers, audit_report=audit
                )
                contradiction = bool(docs_review["contradiction_veto"])
                check_fail: dict[str, float] = {}
                for check in checks:
                    if check.id == "claim_contradicts_source":
                        check_fail[check.id] = float(contradiction)
                    elif check.audit_finding_class is not None:
                        finding = jev_review.audit_finding_for(
                            check, str(state["name_id"]), audit
                        )
                        check_fail[check.id] = float(finding is not None)
                    else:
                        check_fail[check.id] = jev_review.check_failure_probability(
                            check, answers.get(f"check:{check.id}")
                        )
                items.append(
                    {
                        "id": name,
                        "label": meta["label"],
                        "split": split,
                        "copy_group": meta["copy_group"],
                        "values": {},
                        "check_fail": check_fail,
                        "contradiction": contradiction,
                        "score": float(score),
                        "judgment": {
                            **docs_review,
                            "categories": {
                                dimension: answers.get("dimension:" + dimension)
                                for dimension in jev_review._DOCS_DIMENSION_CRITERIA
                            },
                        },
                    }
                )
    return items, {"questions": [], "fit": {}}


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
