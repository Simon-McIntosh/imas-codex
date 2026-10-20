"""Fit panel-matched operating cutoffs for the landed Jev score models.

Choose logistic or rubric by AUC on all scored calibration rows. Match the
panel on the paired calibration population before the documentation veto;
validation rows only measure the frozen rule. All intervals use the shared
evaluation owner. Rubric dimensions remain independent of the mapped scalar.

Run ``uv run python -m imas_codex.standard_names.review.cascade_calibration``
to rebuild from frozen receipts and the score models in the threshold artifact.
No LLM calls are made.
"""

from __future__ import annotations

import hashlib
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

DEFAULT_NAME_RECEIPTS = Path(
    "/home/ITER/mcintos/.config/reckon/crew/runs/"
    "r-20261016T101231058904-srtj-evidence-supply/evidence-decisions.jsonl"
)
DEFAULT_DOCS_RECEIPTS = Path(
    "/home/ITER/mcintos/.config/reckon/crew/runs/"
    "r-20261017T200215613429-srtj-docs-score-arm/documentation-score-frozen.json.xz"
)

SCHEMA = "review-cascade-thresholds"


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
                contradiction = any(
                    jev_review.claim_contradicts(answer)
                    for key, answer in answers.items()
                    if key.startswith("check:claim_contradicts_source:")
                )
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
                            **jev_review.aggregate_docs_review(
                                state, checks, answers, audit_report=audit
                            ),
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
RESAMPLES = 10000
RESAMPLE_SEED = 20261020
PAIRED_EVIDENCE = (
    _REPO / "docs/evidence/fragments/sn-review-through-jev/"
    "srtj-jev-vs-panel-operating-point.json"
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


def _measure(labels, accepted, groups):
    from imas_codex.discovery.base.evaluation import bootstrap_interval

    return {
        key: {
            "numerator": numerator,
            "denominator": denominator,
            **bootstrap_interval(
                labels,
                accepted,
                copy_groups=groups,
                statistic=lambda y, a, key=key: _metric(y, a, key),
                n_resamples=RESAMPLES,
                seed=RESAMPLE_SEED,
            ),
        }
        for key, (numerator, denominator) in _metric_counts(labels, accepted).items()
    }


def _matched_report(rows, axis, block):
    from imas_codex.discovery.base.evaluation import bootstrap_interval
    from imas_codex.standard_names.review.cascade import _route

    labels = np.asarray([row["label"] for row in rows])
    groups = [row["copy_group"] for row in rows]
    decisions = [
        _route(axis, r["judgment"], r["score"], thresholds=block) for r in rows
    ]
    before = np.asarray(
        [d.selected_score >= block["accept_threshold"] for d in decisions]
    )
    after = np.asarray([d.route == "accept" for d in decisions])
    veto = np.asarray([axis == "docs" and r["contradiction"] for r in rows])
    panel = np.asarray([r["panel"] >= DEFAULT_MIN_SCORE for r in rows])
    assert np.array_equal(after, before & ~veto)
    assert all(
        (d.decision_score >= DEFAULT_MIN_SCORE) == (d.route == "accept")
        for d in decisions
    )
    return {
        "n": len(rows),
        "copy_groups": len(set(groups)),
        "panel": _measure(labels, panel, groups),
        "pre_veto": _measure(labels, before, groups),
        "jev": _measure(labels, after, groups),
        "paired_difference": {
            key: bootstrap_interval(
                labels,
                after,
                scores_b=panel,
                copy_groups=groups,
                statistic=lambda y, a, key=key: _metric(y, a, key),
                n_resamples=RESAMPLES,
                seed=RESAMPLE_SEED,
            )
            for key in METRICS
        },
        "veto": {
            "refine_count": int(veto.sum()),
            "incremental_refine_count": int(np.sum(before & veto)),
            "defects": int(np.sum(veto & (labels == 0))),
            "precision": bootstrap_interval(
                labels,
                ~veto,
                copy_groups=groups,
                statistic=lambda y, a: _metric(y, a, "refine_precision"),
                n_resamples=RESAMPLES,
                seed=RESAMPLE_SEED,
            ),
        },
        "route_counts": {"accept": int(after.sum()), "refine": int((~after).sum())},
    }


def build(
    *,
    name_receipts: Path = DEFAULT_NAME_RECEIPTS,
    docs_receipts: Path = DEFAULT_DOCS_RECEIPTS,
    golden_name: Path = GOLDEN_NAME,
    golden_docs: Path = GOLDEN_DOCS,
    paired_evidence: Path = PAIRED_EVIDENCE,
    score_models: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Fit operating cutoffs for the landed score models, then measure holdout.

    Rubric models are frozen inputs carried in the threshold artifact itself;
    this fit changes the decision policy, not the already fitted rubrics. AUC
    selection uses all scored calibration items, and rate matching uses the
    same-identity subset with verified panel scores. Both splits retain the
    exclusions established by the paired evidence receipt.
    """
    from imas_codex.discovery.base.evaluation import roc_auc
    from imas_codex.standard_names.review.cascade import rubric_scores

    models = _read_json(THRESHOLDS_PATH) if score_models is None else score_models
    paired = _read_json(paired_evidence)
    blocks = {}
    for axis, builder, receipts, golden in (
        ("name", _name_axis, name_receipts, golden_name),
        ("docs", _docs_axis, docs_receipts, golden_docs),
    ):
        items, context = builder(receipts, golden)
        rubric = {
            k: v
            for k, v in models[axis]["rubric"].items()
            if k in ("dimensions", "offset")
        }
        for item in items:
            scores = rubric_scores(axis, item["judgment"], rubric)
            item["rubric_score"] = sum(scores.values()) / (20 * len(scores))
        calibration = [r for r in items if r["split"] == "calibration"]
        aucs = {
            name: roc_auc(
                [r["label"] for r in calibration], [r[key] for r in calibration]
            )
            for name, key in (("logistic", "score"), ("rubric", "rubric_score"))
        }
        selected = max(aucs, key=aucs.get)
        by_id = {row["id"]: row for row in items}
        rows = []
        for identity, label, split, group, logistic, panel in paired["paired_items"][
            axis
        ]:
            row = by_id[identity]
            if (row["label"], row["split"], row["copy_group"]) != (label, split, group):
                raise ValueError(f"paired identity metadata mismatch: {identity}")
            if not math.isclose(row["score"], logistic, abs_tol=1e-10, rel_tol=0):
                raise ValueError(f"landed logistic mismatch: {identity}")
            rows.append({**row, "panel": panel})
        fit_rows = [r for r in rows if r["split"] == "calibration"]
        key = "score" if selected == "logistic" else "rubric_score"
        match = matched_cutoff(
            [r[key] for r in fit_rows], [r["panel"] for r in fit_rows]
        )
        block = {
            "score_kind": selected,
            "calibration_auc": aucs,
            "score_selection_n": len(calibration),
            "accept_threshold": match["cutoff"],
            "matched_calibration": match,
            "rubric": rubric,
            "source_receipts": str(receipts),
            "source_sha256": hashlib.sha256(receipts.read_bytes()).hexdigest(),
            "golden_sha256": hashlib.sha256(golden.read_bytes()).hexdigest(),
            "n": len(fit_rows),
            "unjudged_count": 0,
        }
        if axis == "name":
            block.update(
                score_intercept=context["fit"]["intercept"],
                score_weights=context["fit"]["weights"],
                regularization=context["fit"]["regularization"],
            )
        block["report"] = {
            split: _matched_report(
                [r for r in rows if r["split"] == split], axis, block
            )
            for split in ("calibration", "validation")
        }
        block["route_counts"] = block["report"]["calibration"]["route_counts"]
        block["route_shares"] = {
            k: v / len(fit_rows) for k, v in block["route_counts"].items()
        }
        blocks[axis] = block
    document = {
        "schema": SCHEMA,
        "source_split": "calibration",
        "paired_evidence_sha256": hashlib.sha256(
            paired_evidence.read_bytes()
        ).hexdigest(),
        "paired_exclusions": {
            a: paired["coverage"][a]["counts_by_reason"] for a in blocks
        },
        "policy": {
            "score_selection": "Higher AUC on all scored calibration items: landed logistic or unvetoed rubric mean. Ties prefer logistic. Score models remain frozen.",
            "accept_selection": "On paired calibration items, select an inclusive cutoff matching panel accepts at min_score 0.85 before veto; refuse an unresolved boundary tie. Validation never fits a cutoff.",
            "routing": "Accept at or above the cutoff; refine below. No chat band. Only the documentation contradiction veto overrides acceptance; failed checks remain recorded without other vetoes.",
            "decision_score": "For raw score s, cutoff c and consumer min_score m: m*s/c below c; m+(1-m)*(s-c)/(1-c) at or above c. At c store exactly m; protect the strict lower branch against rounding. Endpoint c=0 or c=1 uses only its defined branch.",
            "veto_score": "A documentation contradiction refines and caps the mapped score at the representable float immediately below min_score. Preserve all original 0-to-20 rubric dimensions and failed checks.",
            "intervals": "Shared evaluation.bootstrap_interval: 10000 seeded copy-group draws, policies held fixed. Paired differences are final Jev minus panel. Undefined ratios have null estimates; undefined draws are omitted and resampled. Intervals condition on frozen fits and cutoffs, not fitting uncertainty.",
            "unjudged_selection": "Unjudged failures retain the existing incumbent fallback and separate unjudged shadow record.",
        },
        **blocks,
    }
    document["content_sha256"] = _content_digest(document)
    return document


def _content_digest(document: Mapping[str, Any]) -> str:
    content = {key: value for key, value in document.items() if key != "content_sha256"}
    return hashlib.sha256(
        json.dumps(content, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def main() -> int:
    document = build()
    THRESHOLDS_PATH.write_text(
        json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {
                a: {
                    "score_kind": document[a]["score_kind"],
                    "match": document[a]["matched_calibration"],
                    "validation": document[a]["report"]["validation"],
                }
                for a in ("name", "docs")
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
