"""Rebuild the cascade routing thresholds from the frozen calibration receipts.

``cascade_thresholds.json`` is derived data: per-axis check cutoffs, a name
score fit, and accept/reject score thresholds. This module rebuilds it from the
frozen receipts its ``source_receipts`` fields name, with no LLM call.

The veto set is chosen **jointly**. A check's cutoff on its own was chosen to
wrongly flag at most ten per cent of accepted calibration items, so a union of
many such cutoffs flags far more than that. Here checks are added one at a time
in descending order of the defect recall they add, and the union of the chosen
cutoffs is held to flagging at most ``VETO_BUDGET`` of the accepted calibration
items on each axis. A check left out is simply not a veto; it still feeds the
axis score, through the name logistic and the docs component weights.

Run ``python -m imas_codex.standard_names.review.cascade_calibration`` to rebuild
the route file, or import :func:`build` to derive the same thresholds in memory.
"""

from __future__ import annotations

import json
import lzma
import math
from collections.abc import Iterable, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np

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
            key: float(coef) for key, coef in zip(questions, model.coef_[0], strict=True)
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
                    }
                )
    return items, {"questions": [], "fit": {}}


def _threshold_key(cutoff: float) -> float:
    return round(float(cutoff), 12)


def _candidate_cutoffs(items: Sequence[Mapping[str, Any]], check_id: str) -> list[float]:
    values = sorted({_threshold_key(item["check_fail"].get(check_id, 0.0)) for item in items})
    return [value for value in values if value > 0.0]


def _per_check_threshold(
    items, check_id: str, accepted_ids: set[str], defect_ids: set[str]
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
        if fpr > PER_CHECK_FPR:
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
        if best is None or score > best[1] or (
            score == best[1] and defects_caught > best[2]["defects_caught"]
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
            items, check_id, accepted_ids, defect_ids
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
            if best is None or gain > best[0] or (gain == best[0] and check_id < best[1]):
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
        precision = (
            sum(1 for item in group if item["label"] == ACCEPTABLE_LABEL) / len(group)
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
        routed = [(item, route(item))  # type: ignore[arg-type]
                  for item in items]
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
            "auc": _auc(items),
        }

    return {"calibration": measure(calibration), "validation": measure(validation)}


def _auc(items: Sequence[Mapping[str, Any]]) -> float | None:
    pos = [item["score"] for item in items if item["label"] == ACCEPTABLE_LABEL]
    neg = [item["score"] for item in items if item["label"] != ACCEPTABLE_LABEL]
    if not pos or not neg:
        return None
    wins = sum(
        1.0 if p > n else 0.5 if p == n else 0.0 for p in pos for n in neg
    )
    return round(wins / (len(pos) * len(neg)), 6)


def _axis_thresholds(
    items: Sequence[Mapping[str, Any]],
    axis: str,
    *,
    hard_veto: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    calibration = _calibration_items(items, "calibration")
    selected, diagnostics = select_vetoes(
        calibration, axis, hard_veto=hard_veto
    )
    union = diagnostics["__union__"]
    accept, accept_basis = _accept_threshold(
        calibration, _flagged_at(calibration, selected) | (
            {item["id"] for item in calibration if item["contradiction"]}
            if hard_veto
            else set()
        )
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

    name_block, name_union = _axis_thresholds(name_items, "name", hard_veto=False)
    docs_block, docs_union = _axis_thresholds(docs_items, "docs", hard_veto=True)

    name_block["score_intercept"] = name_context["fit"]["intercept"]
    name_block["score_weights"] = name_context["fit"]["weights"]
    name_block["regularization"] = name_context["fit"]["regularization"]

    name_block["report"] = _axis_report(
        _calibration_items(name_items, "calibration"),
        _calibration_items(name_items, "validation"),
        name_block,
        hard_veto=False,
    )
    docs_block["report"] = _axis_report(
        _calibration_items(docs_items, "calibration"),
        _calibration_items(docs_items, "validation"),
        docs_block,
        hard_veto=True,
    )
    name_block["dropped_checks"] = _dropped(name_items, name_block)
    docs_block["dropped_checks"] = _dropped(docs_items, docs_block)

    return {
        "schema": SCHEMA,
        "source_split": "calibration",
        "policy": {
            "check_selection": (
                "checks are added to the veto set one at a time by the defect "
                "recall they add, holding the union of cutoffs to at most "
                f"{VETO_BUDGET:.2f} of accepted calibration items on each axis; "
                "a check left out is not a veto and still feeds the score"
            ),
            "check_cutoff": (
                "one cutoff per check, maximising defect recall minus accepted "
                f"false positive rate with that rate at most {PER_CHECK_FPR:.2f}, "
                f"at least {MIN_DEFECTS} defects caught and separation at least "
                f"{MIN_SEPARATION:.2f}"
            ),
            "accept_selection": (
                "lowest check-clear calibration score bin of at least "
                f"{ACCEPT_GROUP_MIN} items at empirical precision at least "
                f"{ACCEPT_PRECISION:.2f}; where no bin reaches it, the bin of "
                "least five items with the highest available precision"
            ),
            "reject_selection": (
                f"calibration score {REJECT_PERCENTILE:.0f}th percentile; a "
                "check-clear low-tail case uses chat_below_reject, distinct from "
                "the uncertain band"
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


def _dropped(items: Sequence[Mapping[str, Any]], block: Mapping[str, Any]) -> list[str]:
    selected = set(block["check_thresholds"])
    per_check = {
        check_id
        for check_id, stats in block["check_diagnostics"].items()
        if stats["threshold"] is not None
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