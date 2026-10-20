"""Rebuild the cascade routing thresholds from the frozen calibration receipts.

``cascade_thresholds.json`` contains monotone dimension fits, a joint veto
selected using grouped cross-validation, and reports against the logistic
reference. Only calibration rows fit parameters; validation rows measure the
result. All answers come from frozen receipts, with no LLM call.

The binary labels calibrate each dimension's evidence, not a dimension gold
score. Their mean decides at the same threshold the persistence consumers use.
A veto lowers that mean, and the raw dimension scores remain in the record.

Run ``uv run python -m imas_codex.standard_names.review.cascade_calibration``
to rebuild the route file, or import :func:`build` for an in-memory derivation.
"""

from __future__ import annotations

import hashlib
import json
import lzma
import math
from collections.abc import Iterable, Mapping, Sequence
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

# The frozen golden splits label a row 1 when the item is defensible and 0 when
# it is a planted defect, so every precision and recall below counts the two
# against this constant rather than against a bare 0 or 1.
ACCEPTABLE_LABEL = 1
VETO_BUDGET = 0.10
PER_CHECK_FPR = 0.10
MIN_SEPARATION = 0.02
MIN_DEFECTS = 2
ACCEPT_GROUP_MIN = 5
ACCEPT_PRECISION = 0.90
REJECT_PERCENTILE = 10.0
SCHEMA = "review-cascade-thresholds"

# Membership of the independent-veto reference artifact, used only for removal
# accounting. It is not an admissible candidate pool or an input to selection.
INDEPENDENT_VETO_REFERENCE_SHA256 = (
    "a797be9b51cceb5bb1691a98240c2f2be17066fecfbc46837f6664d3336fc9bb"
)
INDEPENDENT_VETO_CHECKS = {
    "name": (
        "advisory_alias_collapsed",
        "ambiguity_in_isolation",
        "bare_shape_parameter_missing_surface",
        "bound_to_wrong_object",
        "coupled_vs_launched_confusion",
        "device_attributed_observable",
        "dropped_distinguishing_axis",
        "malformed_physical_base",
        "missing_required_segment",
        "model_surname_suffix",
        "non_canonical_locus_token",
        "non_canonical_operator_join",
        "ordered_sample_positional_token",
        "over_qualified_modifier",
        "over_specialised_parent",
        "source_fidelity_unsupported_qualifier",
        "unit_kind_inconsistent",
        "unregistered_token_in_closed_segment",
        "value_provenance_in_name",
    ),
    "docs": (
        "defining_equation_missing",
        "description_false_of_some_binding",
        "description_not_single_definition",
        "description_template_opening_mismatch",
        "equation_physically_incorrect",
        "false_equivalence_with_related",
        "incompatible_physics_qualifier",
        "latex_escaping_broken",
        "non_normative_documentation",
        "relationship_link_missing",
        "sign_convention_missing",
        "units_in_prose",
    ),
}


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


def _threshold_key(cutoff: float) -> float:
    return round(float(cutoff), 12)


def _candidate_cutoffs(
    items: Sequence[Mapping[str, Any]], check_id: str
) -> list[float]:
    values = sorted(
        {_threshold_key(item["check_fail"].get(check_id, 0.0)) for item in items}
    )
    return [value for value in values if value > 0.0]


def _per_check_threshold(
    items,
    check_id: str,
    accepted_ids: set[str],
    defect_ids: set[str],
    *,
    max_fpr: float = PER_CHECK_FPR,
) -> tuple[float | None, dict[str, Any]]:
    """The cutoff for one check, maximising recall minus false positives."""
    best: tuple[float, float, dict[str, Any]] | None = None
    for cutoff in _candidate_cutoffs(items, check_id):
        flagged = {
            item["id"]
            for item in items
            if item["check_fail"].get(check_id, 0.0) >= cutoff
        }
        accepted_flagged = len(flagged & accepted_ids)
        defects_caught = len(flagged & defect_ids)
        if defect_ids and defects_caught < MIN_DEFECTS:
            continue
        fpr = accepted_flagged / len(accepted_ids) if accepted_ids else 1.0
        if fpr > max_fpr:
            continue
        defect_values = [
            item["check_fail"].get(check_id, 0.0)
            for item in items
            if item["id"] in defect_ids
        ]
        accepted_values = [
            item["check_fail"].get(check_id, 0.0)
            for item in items
            if item["id"] in accepted_ids
        ]
        separation = _mean(defect_values) - _mean(accepted_values)
        if separation < MIN_SEPARATION:
            continue
        recall = defects_caught / len(defect_ids) if defect_ids else 0.0
        score = recall - fpr
        stats = {
            "accepted_flagged": accepted_flagged,
            "defects_caught": defects_caught,
            "separation": round(separation, 4),
            "threshold": cutoff,
        }
        if (
            best is None
            or score > best[1]
            or (score == best[1] and defects_caught > best[2]["defects_caught"])
        ):
            best = (cutoff, score, stats)
    if best is None:
        return None, {
            "accepted_flagged": 0,
            "defects_caught": 0,
            "separation": 0.0,
            "threshold": None,
        }
    return best[0], best[2]


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return float(np.mean(values)) if values else 0.0


def _flagged_at(
    items: Sequence[Mapping[str, Any]], thresholds: Mapping[str, float]
) -> set[str]:
    flagged: set[str] = set()
    for item in items:
        for check_id, cutoff in thresholds.items():
            if item["check_fail"].get(check_id, 0.0) >= cutoff:
                flagged.add(item["id"])
                break
    return flagged


def select_vetoes(
    items: Sequence[Mapping[str, Any]],
    axis: str,
    *,
    budget: float = VETO_BUDGET,
    hard_veto: bool = False,
) -> tuple[dict[str, float], dict[str, dict[str, Any]]]:
    """Choose the veto cutoffs jointly, holding the union inside its budget.

    ``hard_veto`` adds the documentation contradiction veto, which is a fixed
    rule rather than a fitted cutoff; it is kept only when the union of the
    chosen checks plus the veto still flags at most ``budget`` of the accepted
    items.
    """
    accepted = [item for item in items if item["label"] == ACCEPTABLE_LABEL]
    defects = [item for item in items if item["label"] != ACCEPTABLE_LABEL]
    accepted_ids = {item["id"] for item in accepted}
    defect_ids = {item["id"] for item in defects}
    cap = budget * len(accepted_ids)

    candidates: dict[str, tuple[float, dict[str, Any]]] = {}
    for check_id in _check_ids(items):
        cutoff, stats = _per_check_threshold(
            items,
            check_id,
            accepted_ids,
            defect_ids,
            max_fpr=min(budget, PER_CHECK_FPR),
        )
        if cutoff is not None:
            candidates[check_id] = (cutoff, stats)

    hard_flags: set[str] = set()
    if hard_veto:
        hard_flags = {item["id"] for item in items if item["contradiction"]}

    union = set(hard_flags)
    if len(union & accepted_ids) > cap:
        raise ValueError("hard veto alone exceeds the joint false-veto budget")

    selected: dict[str, float] = {}
    while True:
        best: tuple[float, str] | None = None
        for check_id, (cutoff, _stats) in candidates.items():
            if check_id in selected:
                continue
            flagged = {
                item["id"]
                for item in items
                if item["check_fail"].get(check_id, 0.0) >= cutoff
            }
            if len((union | flagged) & accepted_ids) > cap:
                continue
            gain = len((union | flagged) & defect_ids) - len(union & defect_ids)
            if gain <= 0:
                continue
            if (
                best is None
                or gain > best[0]
                or (gain == best[0] and check_id < best[1])
            ):
                best = (gain, check_id)
        if best is None:
            break
        check_id = best[1]
        selected[check_id] = candidates[check_id][0]
        union |= {
            item["id"]
            for item in items
            if item["check_fail"].get(check_id, 0.0) >= selected[check_id]
        }

    if hard_veto:
        selected = {**selected, "claim_contradicts_source": 1.0}
    fitted = {
        key: value
        for key, value in selected.items()
        if not (hard_veto and key == "claim_contradicts_source")
    }
    union = _flagged_at(items, fitted)
    if hard_veto:
        union |= hard_flags

    diagnostics: dict[str, dict[str, Any]] = {}
    for check_id in _check_ids(items):
        cutoff = selected.get(check_id)
        if cutoff is None:
            diagnostics[check_id] = {
                "accepted_flagged": 0,
                "defects_caught": 0,
                "separation": round(
                    _mean(item["check_fail"].get(check_id, 0.0) for item in defects)
                    - _mean(item["check_fail"].get(check_id, 0.0) for item in accepted),
                    4,
                ),
                "threshold": None,
            }
            continue
        flagged = {
            item["id"]
            for item in items
            if item["check_fail"].get(check_id, 0.0) >= cutoff
        }
        diagnostics[check_id] = {
            "accepted_flagged": len(flagged & accepted_ids),
            "defects_caught": len(flagged & defect_ids),
            "separation": round(
                _mean(item["check_fail"].get(check_id, 0.0) for item in defects)
                - _mean(item["check_fail"].get(check_id, 0.0) for item in accepted),
                4,
            ),
            "threshold": cutoff,
        }
    diagnostics["__union__"] = {
        "accepted_flagged": len(union & accepted_ids),
        "accepted_n": len(accepted_ids),
        "false_veto_rate": (
            len(union & accepted_ids) / len(accepted_ids) if accepted_ids else 0.0
        ),
        "defects_caught": len(union & defect_ids),
        "defect_n": len(defect_ids),
        "selected": sorted(selected),
    }
    return selected, diagnostics


def _check_ids(items: Sequence[Mapping[str, Any]]) -> list[str]:
    ids: list[str] = []
    for item in items:
        for key in item["check_fail"]:
            if key not in ids:
                ids.append(key)
    return ids


def _accept_threshold(
    items: Sequence[Mapping[str, Any]], vetoed: set[str]
) -> tuple[float, str]:
    """Lowest check-clear score bin of at least five items at 0.90 precision."""
    clear = [item for item in items if item["id"] not in vetoed]
    best_fallback: tuple[float, float] | None = None
    for cutoff in sorted({_threshold_key(item["score"]) for item in clear}):
        group = [item for item in clear if item["score"] >= cutoff]
        if len(group) < ACCEPT_GROUP_MIN:
            continue
        precision = sum(1 for item in group if item["label"] == ACCEPTABLE_LABEL) / len(
            group
        )
        if precision >= ACCEPT_PRECISION:
            return cutoff, "precision"
        if best_fallback is None or precision > best_fallback[1]:
            best_fallback = (cutoff, precision)
    if best_fallback is None:
        raise ValueError("no check-clear score bin of at least five items")
    return best_fallback[0], "best-available-precision"


def _reject_threshold(items: Sequence[Mapping[str, Any]]) -> float:
    scores = [item["score"] for item in items]
    return float(np.percentile(scores, REJECT_PERCENTILE))


def _calibration_items(items, split: str) -> list[dict[str, Any]]:
    return [item for item in items if item["split"] == split]


def _axis_report(
    calibration: Sequence[Mapping[str, Any]],
    validation: Sequence[Mapping[str, Any]],
    thresholds: Mapping[str, Any],
    *,
    hard_veto: bool,
) -> dict[str, Any]:
    """Route both splits with one threshold set and report the decision quality."""

    def route(item) -> str:
        for check_id, cutoff in thresholds["check_thresholds"].items():
            if item["check_fail"].get(check_id, 0.0) >= cutoff:
                return "refine"
        if hard_veto and item["contradiction"]:
            return "refine"
        if item["score"] >= thresholds["accept_threshold"]:
            return "accept"
        if item["score"] < thresholds["reject_threshold"]:
            return "chat_below_reject"
        return "chat_review"

    def measure(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        routed = [
            (item, route(item))  # type: ignore[arg-type]
            for item in items
        ]
        counts: dict[str, int] = {}
        for _item, decision in routed:
            counts[decision] = counts.get(decision, 0) + 1
        total = len(routed) or 1
        accepted = [item for item in items if item["label"] == ACCEPTABLE_LABEL]
        refined = [item for item, decision in routed if decision == "refine"]
        accepts = [item for item, decision in routed if decision == "accept"]
        return {
            "n": len(routed),
            "route_shares": {k: round(v / total, 6) for k, v in sorted(counts.items())},
            "route_counts": dict(sorted(counts.items())),
            "false_veto_rate_on_accepted": (
                round(
                    sum(1 for item in accepted if route(item) == "refine")
                    / len(accepted),
                    6,
                )
                if accepted
                else 0.0
            ),
            "refine_precision": (
                round(
                    sum(1 for item in refined if item["label"] != ACCEPTABLE_LABEL)
                    / len(refined),
                    6,
                )
                if refined
                else None
            ),
            "accept_precision": (
                round(
                    sum(1 for item in accepts if item["label"] == ACCEPTABLE_LABEL)
                    / len(accepts),
                    6,
                )
                if accepts
                else None
            ),
            "accept_n": len(accepts),
            "accept_correct": sum(item["label"] for item in accepts),
            "accept_precision_interval": _wilson(
                sum(item["label"] for item in accepts), len(accepts)
            ),
            "auc": _auc(items),
        }

    return {"calibration": measure(calibration), "validation": measure(validation)}


def _auc(items: Sequence[Mapping[str, Any]]) -> float | None:
    pos = [item["score"] for item in items if item["label"] == ACCEPTABLE_LABEL]
    neg = [item["score"] for item in items if item["label"] != ACCEPTABLE_LABEL]
    if not pos or not neg:
        return None
    wins = sum(1.0 if p > n else 0.5 if p == n else 0.0 for p in pos for n in neg)
    return round(wins / (len(pos) * len(neg)), 6)


def _axis_thresholds(
    items: Sequence[Mapping[str, Any]],
    axis: str,
    *,
    hard_veto: bool,
    budget: float = VETO_BUDGET,
    fixed_only: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    calibration = _calibration_items(items, "calibration")
    selection_rows = (
        [
            {
                **row,
                "check_fail": {"claim_contradicts_source": float(row["contradiction"])},
            }
            for row in calibration
        ]
        if fixed_only
        else calibration
    )
    selected, diagnostics = select_vetoes(
        selection_rows,
        axis,
        hard_veto=hard_veto,
        budget=VETO_BUDGET if fixed_only else budget,
    )
    union = diagnostics["__union__"]
    accept, accept_basis = _accept_threshold(
        calibration,
        _flagged_at(calibration, selected)
        | (
            {item["id"] for item in calibration if item["contradiction"]}
            if hard_veto
            else set()
        ),
    )
    block = {
        "check_thresholds": selected,
        "check_diagnostics": {
            key: value for key, value in diagnostics.items() if key != "__union__"
        },
        "accept_threshold": accept,
        "reject_threshold": _reject_threshold(calibration),
        "n": len(calibration),
        "unjudged_count": 0,
        "joint_selection": union,
        "accept_basis": accept_basis,
    }
    return block, union


def _wilson(successes: int, total: int) -> list[float] | None:
    """Two-sided 95% Wilson interval, including the small accept cohorts."""
    if not total:
        return None
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    radius = (
        z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    )
    return [max(0.0, centre - radius), min(1.0, centre + radius)]


def _folds(items):
    from sklearn.model_selection import GroupKFold

    return list(
        GroupKFold(n_splits=5).split(
            np.zeros(len(items)), groups=[item["copy_group"] for item in items]
        )
    )


def cross_validated_veto_budget(items, axis: str, *, hard_veto: bool):
    """Select recall using held-out calibration groups, with an uncertainty bound."""
    rows = _calibration_items(items, "calibration")
    reports = []
    for budget in (0.0, 0.01, 0.025, 0.05, 0.075, 0.10):
        accepted_n = false_vetoes = defects_n = caught = 0
        feasible = True
        for train, test in _folds(rows):
            try:
                selected, _ = select_vetoes(
                    [rows[i] for i in train], axis, budget=budget, hard_veto=hard_veto
                )
            except ValueError:
                feasible = False
                break
            held = [rows[i] for i in test]
            flagged = _flagged_at(held, selected)
            for item in held:
                veto = item["id"] in flagged or (hard_veto and item["contradiction"])
                if item["label"] == ACCEPTABLE_LABEL:
                    accepted_n += 1
                    false_vetoes += int(veto)
                else:
                    defects_n += 1
                    caught += int(veto)
        interval = _wilson(false_vetoes, accepted_n) if feasible else None
        reports.append(
            {
                "budget": budget,
                "feasible": feasible,
                "accepted_n": accepted_n,
                "false_vetoes": false_vetoes,
                "false_veto_interval": interval,
                "defect_recall": caught / defects_n if defects_n else 0.0,
            }
        )
    admissible = [
        row
        for row in reports
        if row["feasible"] and row["false_veto_interval"][1] <= VETO_BUDGET
    ]
    if admissible:
        winner = max(admissible, key=lambda row: (row["defect_recall"], -row["budget"]))
        return winner["budget"], reports, False
    # Refuse every learned veto when CV cannot substantiate its budget.
    # A fixed contradiction rule has no fitted cutoff, so its held-out count
    # is exact across these folds; retain it only within the observed limit.
    accepted_n = sum(row["label"] for row in rows)
    false_vetoes = sum(
        row["label"] and hard_veto and row["contradiction"] for row in rows
    )
    if false_vetoes / accepted_n > VETO_BUDGET:
        raise ValueError(f"{axis}: fixed veto alone exceeds the held-out limit")
    reports.append(
        {
            "budget": 0.0,
            "fixed_only": True,
            "feasible": True,
            "accepted_n": accepted_n,
            "false_vetoes": false_vetoes,
            "false_veto_interval": _wilson(false_vetoes, accepted_n),
            "defect_recall": sum(
                not row["label"] and hard_veto and row["contradiction"] for row in rows
            )
            / sum(not row["label"] for row in rows),
        }
    )
    return 0.0, reports, True


def _fit_rubric(rows, axis: str, penalty: float) -> dict[str, Any]:
    """Fit monotone dimension logits against defensibility labels.

    No dimension labels exist in the receipts. Each dimension sees only its
    own category expectation and check compliance; the binary supervision
    calibrates each dimension's evidence and is not an incumbent dimension gold.
    """
    from scipy.optimize import minimize
    from scipy.special import expit

    from imas_codex.standard_names.review.cascade import rubric_features

    features = [rubric_features(axis, row["judgment"]) for row in rows]
    dimensions = sorted(features[0])
    keys = {
        dim: sorted(set.intersection(*(set(row[dim]) for row in features)))
        for dim in dimensions
    }
    if any(not values for values in keys.values()):
        raise ValueError("each rubric dimension needs observed calibration features")
    matrices = {
        dim: np.asarray([[row[dim][key] for key in keys[dim]] for row in features])
        for dim in dimensions
    }
    labels = np.asarray([row["label"] for row in rows])
    slices = {}
    cursor = len(dimensions)
    for dim in dimensions:
        slices[dim] = slice(cursor, cursor + len(keys[dim]))
        cursor += len(keys[dim])
    initial = np.zeros(cursor)
    for dim in dimensions:
        initial[slices[dim]] = 1 / len(keys[dim])

    def objective(parameters):
        predicted = np.column_stack(
            [
                expit(parameters[i] + matrices[dim] @ parameters[slices[dim]])
                for i, dim in enumerate(dimensions)
            ]
        )
        bounded = np.clip(predicted, 1e-12, 1 - 1e-12)
        loss = -np.mean(
            labels[:, None] * np.log(bounded)
            + (1 - labels[:, None]) * np.log(1 - bounded)
        )
        gradient = np.zeros_like(parameters)
        for i, dim in enumerate(dimensions):
            local = (predicted[:, i] - labels) / (len(rows) * len(dimensions))
            gradient[i] = local.sum()
            gradient[slices[dim]] = matrices[dim].T @ local
        weights = parameters[len(dimensions) :]
        loss += penalty * np.dot(weights, weights) / (2 * len(weights))
        gradient[len(dimensions) :] += penalty * weights / len(weights)
        return loss, gradient

    model = minimize(
        objective,
        initial,
        jac=True,
        method="L-BFGS-B",
        bounds=[(-30, 30)] * len(dimensions) + [(0, 30)] * (cursor - len(dimensions)),
        options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-8},
    )
    if not model.success:
        raise ValueError(f"rubric fit failed: {model.message}")
    return {
        "dimensions": {
            dim: {
                "intercept": float(model.x[i]),
                "weights": dict(
                    zip(keys[dim], map(float, model.x[slices[dim]]), strict=True)
                ),
            }
            for i, dim in enumerate(dimensions)
        },
        "offset": 0.0,
        "band_width": 0.05,
        "penalty": penalty,
    }


def fit_rubric(items, axis: str, vetoes) -> dict[str, Any]:
    from scipy.special import expit

    from imas_codex.standard_names.review.cascade import rubric_scores

    rows = _calibration_items(items, "calibration")
    trials = []
    for penalty in (0.001, 0.01, 0.1):
        predictions = []
        for train, test in _folds(rows):
            fit = _fit_rubric([rows[i] for i in train], axis, penalty)
            for index in test:
                scores = rubric_scores(axis, rows[index]["judgment"], fit)
                predictions.append(
                    {**rows[index], "score": sum(scores.values()) / (20 * len(scores))}
                )
        trials.append({"penalty": penalty, "auc": _auc(predictions)})
    winner = max(trials, key=lambda trial: (trial["auc"], trial["penalty"]))
    fit = _fit_rubric(rows, axis, winner["penalty"])
    # Translate every dimension logit equally. Select the largest check-clear
    # cohort at the policy precision; fallback reports the attainable precision.
    vetoed = _flagged_at(rows, vetoes)
    clear = [row for row in rows if row["id"] not in vetoed]
    base = (
        np.asarray(
            [list(rubric_scores(axis, row["judgment"], fit).values()) for row in clear]
        )
        / 20
    )
    logits = np.log(base / (1 - base))
    labels = np.asarray([row["label"] for row in clear])
    candidates = []
    for offset in np.linspace(-6, max(6, 2 - float(logits.min())), 481):
        selected = expit(logits + offset).mean(axis=1) >= DEFAULT_MIN_SCORE
        count = int(selected.sum())
        if count >= ACCEPT_GROUP_MIN:
            precision = float(labels[selected].mean())
            candidates.append((precision, count, float(offset)))
    if not candidates:
        raise ValueError("no rubric accept cohort for calibration")
    qualified = [row for row in candidates if row[0] >= ACCEPT_PRECISION]
    chosen = (
        max(qualified, key=lambda row: (row[1], row[0], -row[2]))
        if qualified
        else max(candidates)
    )
    fit.update(
        offset=chosen[2],
        cv=trials,
        calibration_accept_precision=chosen[0],
        calibration_accept_n=chosen[1],
        min_score=DEFAULT_MIN_SCORE,
        offset_basis="precision" if qualified else "best-available-precision",
    )
    return fit


def _rubric_report(items, axis, block):
    from imas_codex.standard_names.review.cascade import _route

    report = {}
    for split in ("calibration", "validation"):
        rows = _calibration_items(items, split)
        decisions = [
            _route(axis, row["judgment"], row["score"], thresholds=block)
            for row in rows
        ]
        counts = {
            route: sum(d.route == route for d in decisions)
            for route in ("accept", "chat_review", "refine")
        }
        accepts = [
            row for row, d in zip(rows, decisions, strict=True) if d.route == "accept"
        ]
        refines = [
            row for row, d in zip(rows, decisions, strict=True) if d.route == "refine"
        ]
        accepted_n = sum(row["label"] for row in rows)
        false_vetoes = sum(
            row["label"] == ACCEPTABLE_LABEL and bool(d.failed_checks)
            for row, d in zip(rows, decisions, strict=True)
        )
        correct = sum(row["label"] for row in accepts)
        report[split] = {
            "n": len(rows),
            "route_counts": counts,
            "route_shares": {key: value / len(rows) for key, value in counts.items()},
            "accept_n": len(accepts),
            "accept_correct": correct,
            "accept_precision": correct / len(accepts) if accepts else None,
            "accept_precision_interval": _wilson(correct, len(accepts)),
            "refine_precision": sum(1 - row["label"] for row in refines) / len(refines)
            if refines
            else None,
            "false_veto_rate_on_accepted": false_vetoes / accepted_n,
            "false_refine_rate_on_accepted": sum(row["label"] for row in refines)
            / accepted_n,
            "auc": _auc(
                [
                    {**row, "score": d.total}
                    for row, d in zip(rows, decisions, strict=True)
                ]
            ),
            "raw_rubric_auc": _auc(
                [
                    {
                        **row,
                        "score": sum(d.raw_scores.values()) / (20 * len(d.raw_scores)),
                    }
                    for row, d in zip(rows, decisions, strict=True)
                ]
            ),
            "logistic_auc": _auc(rows),
        }
    return report


def build(
    *,
    name_receipts: Path = DEFAULT_NAME_RECEIPTS,
    docs_receipts: Path = DEFAULT_DOCS_RECEIPTS,
    golden_name: Path = GOLDEN_NAME,
    golden_docs: Path = GOLDEN_DOCS,
) -> dict[str, Any]:
    """Derive the full threshold document from the frozen receipts."""
    name_items, name_context = _name_axis(name_receipts, golden_name)
    docs_items, _docs_context = _docs_axis(docs_receipts, golden_docs)

    name_block, _ = _axis_thresholds(name_items, "name", hard_veto=False)
    docs_block, _ = _axis_thresholds(docs_items, "docs", hard_veto=True)

    for axis, items, block in (
        ("name", name_items, name_block),
        ("docs", docs_items, docs_block),
    ):
        block["logistic_only_report"] = _axis_report(
            _calibration_items(items, "calibration"),
            _calibration_items(items, "validation"),
            {**block, "check_thresholds": {}},
            hard_veto=False,
        )
        block["joint_only_report"] = _axis_report(
            _calibration_items(items, "calibration"),
            _calibration_items(items, "validation"),
            block,
            hard_veto=axis == "docs",
        )
        block["prior_veto_checks"] = sorted(block["check_thresholds"])
        block["reference_veto_checks"] = sorted(INDEPENDENT_VETO_CHECKS[axis])
        block["reference_veto_sha256"] = INDEPENDENT_VETO_REFERENCE_SHA256
        block["joint_only_removed_checks"] = sorted(
            set(block["reference_veto_checks"]) - set(block["prior_veto_checks"])
        )
        budget, cv, fixed_only = cross_validated_veto_budget(
            items, axis, hard_veto=axis == "docs"
        )
        fitted, _ = _axis_thresholds(
            items, axis, hard_veto=axis == "docs", budget=budget, fixed_only=fixed_only
        )
        block.update(fitted)
        block["veto_budget"] = budget
        block["fixed_veto_only"] = fixed_only
        block["veto_budget_cv"] = cv
        block["rubric"] = fit_rubric(items, axis, block["check_thresholds"])
        block["report"] = _rubric_report(items, axis, block)
        block["dropped_checks"] = _dropped(items, block)
        block["removed_from_joint_veto"] = sorted(
            set(block["prior_veto_checks"]) - set(block["check_thresholds"])
        )

    name_block["score_intercept"] = name_context["fit"]["intercept"]
    name_block["score_weights"] = name_context["fit"]["weights"]
    name_block["regularization"] = name_context["fit"]["regularization"]

    name_block["source_receipts"] = str(name_receipts)
    docs_block["source_receipts"] = str(docs_receipts)
    name_block["source_sha256"] = hashlib.sha256(name_receipts.read_bytes()).hexdigest()
    docs_block["source_sha256"] = hashlib.sha256(docs_receipts.read_bytes()).hexdigest()
    for block in (name_block, docs_block):
        fitted = block["report"]["calibration"]
        block["route_counts"] = dict(sorted(fitted["route_counts"].items()))
        block["route_shares"] = dict(sorted(fitted["route_shares"].items()))

    document = {
        "schema": SCHEMA,
        "source_split": "calibration",
        "policy": {
            "check_selection": (
                "five-fold copy-group CV within calibration selects the budget "
                "with highest held-out defect recall whose pooled 95% Wilson "
                "upper false-veto bound is at most 0.10; ties use the smaller "
                "budget; check cutoffs and union are then refitted on calibration. "
                "If none qualifies, disable all learned vetoes and retain only "
                "the fixed contradiction rule, provided its observed CV rate "
                "is at most 0.10; its interval is reported without claiming a bound."
            ),
            "check_cutoff": (
                "one cutoff per check, maximising defect recall minus accepted "
                f"false positive rate with that rate at most {PER_CHECK_FPR:.2f}, "
                f"at least {MIN_DEFECTS} defects caught and separation at least "
                f"{MIN_SEPARATION:.2f}"
            ),
            "accept_selection": (
                f"accept at the consumer min_score (default {DEFAULT_MIN_SCORE}) after veto; "
                "a common dimension-logit offset maximises calibration accept "
                "count with at least five accepts and precision at least 0.90; "
                "fallback maximises attainable precision and is recorded"
            ),
            "reject_selection": (
                "the interval [min_score - 0.05, min_score) uses chat_review; "
                "below it refines. The policy width is one point of a 20-point rubric."
            ),
            "rubric": (
                "mean of four dimension logits normalised to 0..1, reported as "
                "0..20 dimension scores; nonnegative weights on category expectation "
                "(best=1, middle=0.5, worst=0) and own-check compliance (1-p_failure). "
                "Binary defensibility supervision fits dimension evidence, not dimension gold "
                "labels. L2 penalty is selected by five-fold copy-group CV AUC."
            ),
            "veto_score": (
                "a veto scales the dimensions together if needed so their normalised "
                "mean is at most min_score - 0.001; raw dimensions and failed-check "
                "probability, cutoff and evidence remain in the review record"
            ),
            "unjudged_selection": (
                "A refused lease or unusable Jev judgment writes a separate "
                "unjudged shadow row with its failure reason; unjudged rows are "
                "excluded from route shares"
            ),
        },
        "name": name_block,
        "docs": docs_block,
    }
    document["content_sha256"] = _content_digest(document)
    return document


def _content_digest(document: Mapping[str, Any]) -> str:
    """The digest of the artifact, over the serialised content minus the digest.

    The route rule reads these cutoffs, so an edit that changes a cutoff without
    re-deriving the document has to be visible: the digest is recomputed from
    the content it protects and written beside it.
    """
    content = {key: value for key, value in document.items() if key != "content_sha256"}
    encoded = json.dumps(content, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _dropped(items: Sequence[Mapping[str, Any]], block: Mapping[str, Any]) -> list[str]:
    selected = set(block["check_thresholds"])
    if "reference_veto_checks" in block:
        return sorted(set(block["reference_veto_checks"]) - selected)
    calibration = _calibration_items(items, "calibration")
    accepted = {row["id"] for row in calibration if row["label"] == ACCEPTABLE_LABEL}
    defects = {row["id"] for row in calibration if row["label"] != ACCEPTABLE_LABEL}
    per_check = {
        key
        for key in _check_ids(calibration)
        if _per_check_threshold(calibration, key, accepted, defects)[0] is not None
    }
    return sorted(per_check - selected)


def main() -> int:
    document = build()
    THRESHOLDS_PATH.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    print(json.dumps(_summary(document), indent=2))
    return 0


def _summary(document: Mapping[str, Any]) -> dict[str, Any]:
    out = {}
    for axis in ("name", "docs"):
        block = document[axis]
        out[axis] = {
            "accepted_veto_rate": block["joint_selection"]["false_veto_rate"],
            "selected": block["joint_selection"]["selected"],
            "dropped": block["dropped_checks"],
            "report": block["report"],
        }
    return out


if __name__ == "__main__":
    raise SystemExit(main())
