"""Calibrated Jev review routes with a default-off deciding mode.

Cutoffs match the panel's calibration acceptance rate before the docs veto.
The decision score is piecewise linear through (0, 0), (cutoff, min_score)
and (1, 1); it is an operating-score mapping, not an acceptance probability.
A contradiction routes a document to refine while its judged scores stay
unchanged; the refusal travels as a documentation review hold, not as a lowered
score. The same mapping handles explicit consumer min_score values.
Thresholds and name-score weights are fitted on frozen calibration judgments.
Every Jev judgment is stored as a noncanonical review with no scalar score;
only the explicit deciding switch may replace the incumbent.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

from imas_codex.standard_names.defaults import DEFAULT_MIN_SCORE

logger = logging.getLogger(__name__)
_THRESHOLDS = Path(__file__).with_name("cascade_thresholds.json")

# The hold kinds this module's checks raise. A deciding pass reconciles exactly
# these on the name it judged, so a failure it no longer reports stops holding
# the document without this module asserting anything about another check's
# holds.
ROUTE_HOLD_KINDS = frozenset({"cascade_contradiction"})

# The axis those kinds sit on. The checks that raise them run on this axis
# alone: a pass deciding a different axis carries no verdict about them, so it
# neither raises nor clears one. Without that, a name-axis pass — whose route
# carries no holds and therefore nothing to re-raise — reads as a pass that
# found none and clears an open documentation contradiction.
ROUTE_HOLD_AXIS = "docs"


@dataclass(frozen=True)
class CascadeRoute:
    route: str
    probability: float | None
    failed_checks: tuple[str, ...] = ()
    scores: Mapping[str, float] = field(default_factory=dict)
    check_details: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    raw_scores: Mapping[str, float] = field(default_factory=dict)

    selected_score: float | None = None
    decision_score: float | None = None
    holds: tuple[Mapping[str, str], ...] = ()

    @property
    def total(self) -> float | None:
        return (
            sum(self.scores.values()) / (20 * len(self.scores)) if self.scores else None
        )


@lru_cache(maxsize=2)
def _rubric_check_dimensions(axis: str) -> dict[str, str]:
    from imas_codex.standard_names.review.checks import load_review_checks

    return {
        check.id: check.dimension
        for check in load_review_checks()[0]
        if check.axis == axis
    }


def rubric_features(
    axis: str, judgment: Mapping[str, Any]
) -> dict[str, dict[str, float]]:
    """Keep category quality and check compliance in the dimension they dock."""
    from imas_codex.standard_names.review.checks import (
        DOCS_DIMENSIONS,
        NAME_DIMENSIONS,
    )
    from imas_codex.standard_names.review.jev_review import (
        _DIMENSION_LEVELS,
        _DOCS_DIMENSION_CRITERIA,
    )

    dimensions = NAME_DIMENSIONS if axis == "name" else DOCS_DIMENSIONS
    levels = _DIMENSION_LEVELS if axis == "name" else _DOCS_DIMENSION_CRITERIA
    features = {dimension: {} for dimension in sorted(dimensions)}
    for dimension in features:
        answer = judgment.get("categories", {}).get(dimension)
        if answer is not None and dimension in levels:
            categories = list(levels[dimension])
            probabilities = answer.get("probabilities", {})
            features[dimension]["category"] = sum(
                float(probabilities.get(level, 0.0)) * (1 - i / (len(categories) - 1))
                for i, level in enumerate(categories)
            )
    for check_id, dimension in _rubric_check_dimensions(axis).items():
        record = judgment.get("checks", {}).get(check_id)
        if record and record.get("answered"):
            probability = _failure_probability(record)
            if probability is not None:
                features[dimension][check_id] = 1 - probability
    return features


def rubric_scores(
    axis: str, judgment: Mapping[str, Any], fit: Mapping[str, Any]
) -> dict[str, float]:
    """Evaluate the fitted monotone dimension models on the 0-to-20 scale."""
    features = rubric_features(axis, judgment)
    scores = {}
    for dimension, model in fit["dimensions"].items():
        values = features[dimension]
        if any(key not in values for key in model["weights"]):
            raise ValueError(f"incomplete rubric answers for {dimension}")
        linear = (
            model["intercept"]
            + fit["offset"]
            + sum(
                coefficient * values[key]
                for key, coefficient in model["weights"].items()
            )
        )
        scores[dimension] = 20 / (1 + math.exp(-max(-50.0, min(50.0, linear))))
    return scores


@lru_cache(maxsize=1)
def _calibration() -> dict[str, Any]:
    return json.loads(_THRESHOLDS.read_text())


DECIDING_AXES = ("name", "docs")
"""The review axes the deciding switch may name."""


def deciding_enabled(axis: str) -> bool:
    """Whether Jev decides *axis*; absent means shadow mode for both axes.

    ``[sn-review].cascade-deciding`` accepts ``true`` (both axes), ``false``
    (neither) or a list of axes drawn from ``name`` and ``docs`` (Jev decides
    those, the incumbent keeps the rest). Any other value is refused, so a
    typo cannot silently leave an axis deciding or shadowing.
    """
    from imas_codex.settings import _get_section

    value = _get_section("sn-review").get("cascade-deciding", False)
    if isinstance(value, bool):
        return value
    if isinstance(value, (list, tuple)) and all(
        isinstance(axis_name, str) for axis_name in value
    ):
        unknown = [name for name in value if name not in (DECIDING_AXES)]
        if unknown:
            raise ValueError(
                "[sn-review].cascade-deciding names unknown review "
                f"axes {unknown}; accepted values are true, false or a list "
                f"drawn from {list(DECIDING_AXES)}"
            )
        return axis in value
    raise ValueError(
        "[sn-review].cascade-deciding must be a boolean or a list of axes "
        f"drawn from {list(DECIDING_AXES)}"
    )


def _failure_probability(record: Mapping[str, Any]) -> float | None:
    """The check's failure probability, or ``None`` when the record carries none.

    A missing or null probability is not a zero: the check was not scored, so it
    names no failure and no threshold is compared against it.
    """
    value = record.get("failure_probability")
    return None if value is None else float(value)


def _route(
    axis: str,
    judgment: Mapping[str, Any],
    probability: float | None,
    *,
    min_score: float = DEFAULT_MIN_SCORE,
    thresholds: Mapping[str, Any] | None = None,
) -> CascadeRoute:
    thresholds = _calibration()[axis] if thresholds is None else thresholds
    if not 0 < min_score <= 1:
        raise ValueError("min_score must be in (0, 1]")
    scores = rubric_scores(axis, judgment, thresholds["rubric"])
    raw_scores = dict(scores)
    failed = []
    details = {}
    from imas_codex.standard_names.review.jev_review import CHECK_FAIL_PROBABILITY

    for check_id, record in judgment.get("checks", {}).items():
        probability_failed = _failure_probability(record)
        if (
            record.get("answered")
            and probability_failed is not None
            and probability_failed >= CHECK_FAIL_PROBABILITY
        ):
            details[check_id] = {
                "probability": probability_failed,
                "cutoff": CHECK_FAIL_PROBABILITY,
                "evidence": record.get("audit_finding") or record.get("answer"),
                "criterion": "nonconforming check answer",
            }
    if axis == "docs" and judgment.get("contradiction_veto"):
        failed = tuple(sorted({*failed, "claim_contradicts_source"}))
        from imas_codex.standard_names.review.jev_review import (
            CONTRADICTION_VETO_PROBABILITY,
        )

        record = judgment.get("checks", {}).get("claim_contradicts_source", {})
        details["claim_contradicts_source"] = {
            "probability": max(
                (
                    float(answer.get("probabilities", {}).get("contradicts", 0.0))
                    for answer in judgment.get("raw_claim_answers", [])
                    if answer is not None
                ),
                default=_failure_probability(record),
            ),
            "cutoff": CONTRADICTION_VETO_PROBABILITY,
            "criterion": "choice=contradicts or probability>=cutoff",
            "evidence": judgment.get("contradiction_reason") or record.get("answer"),
        }
    total = sum(scores.values()) / (20 * len(scores))
    selected = probability if thresholds["score_kind"] == "logistic" else total
    if selected is None:
        raise ValueError("a decided review requires the selected Jev score")
    mapped = map_decision_score(selected, thresholds["accept_threshold"], min_score)
    holds: tuple[Mapping[str, str], ...] = ()
    if "claim_contradicts_source" in failed:
        detail = details.get("claim_contradicts_source", {})
        holds = (
            {
                "kind": "cascade_contradiction",
                "finding": str(
                    detail.get("evidence") or "claim contradicts its DD source"
                ),
                "evidence": "Jev contradiction veto on claim_contradicts_source",
                "raised_by": "jev-cascade",
            },
        )
    route = "refine" if (failed or mapped < min_score) else "accept"
    return CascadeRoute(
        route,
        probability,
        tuple(failed),
        scores,
        details,
        raw_scores,
        selected_score=selected,
        decision_score=mapped,
        holds=holds,
    )


def map_decision_score(
    score: float, cutoff: float, min_score: float = DEFAULT_MIN_SCORE
) -> float:
    """Map [0, cutoff, 1] to [0, min_score, 1], retaining the strict boundary.

    At an endpoint cutoff only its nonempty branch is used. Roundoff just below
    the cutoff is clamped to the preceding float so consumers cannot accept it.
    """
    if not (0 <= score <= 1 and 0 <= cutoff <= 1 and 0 < min_score <= 1):
        raise ValueError("score and cutoff must be in [0, 1]; min_score in (0, 1]")
    if score < cutoff:
        return min(min_score * score / cutoff, math.nextafter(min_score, 0.0))
    if score == cutoff:
        return min_score
    return min_score + (1 - min_score) * (score - cutoff) / (1 - cutoff)


def score_name_review(judgment: Mapping[str, Any]) -> float | None:
    """Apply the calibration-fit logistic to Jev question probabilities."""
    from imas_codex.standard_names.review.jev_review import _DIMENSION_LEVELS

    fit = _calibration()["name"]
    values: dict[str, float] = {}
    for key in fit["score_weights"]:
        kind, identifier = key.split(":", 1)
        if kind == "dimension":
            answer = judgment.get("categories", {}).get(identifier)
            accepted = _DIMENSION_LEVELS[identifier][0]
            if answer is None:
                return None
            values[key] = float(answer.get("probabilities", {}).get(accepted, 0.0))
        else:
            record = judgment.get("checks", {}).get(identifier)
            answer = record.get("answer") if record else None
            if answer is None:
                return None
            if "noul" in answer:
                values[key] = 1.0 - float(answer["noul"])
            else:
                question = record.get("question") or {}
                first_option = next(iter(question.get("criteria", {})), None)
                if first_option is None:
                    return None
                values[key] = float(
                    answer.get("probabilities", {}).get(first_option, 0.0)
                )
    linear = fit["score_intercept"] + sum(
        coefficient * values[key] for key, coefficient in fit["score_weights"].items()
    )
    return 1.0 / (1.0 + math.exp(-max(-50.0, min(50.0, linear))))


def route_name(
    judgment: Mapping[str, Any],
    *,
    probability: float | None = None,
    min_score: float = DEFAULT_MIN_SCORE,
) -> CascadeRoute:
    return _route(
        "name",
        judgment,
        probability if probability is not None else score_name_review(judgment),
        min_score=min_score,
    )


def route_docs(
    judgment: Mapping[str, Any],
    *,
    probability: float | None = None,
    min_score: float = DEFAULT_MIN_SCORE,
) -> CascadeRoute:
    return _route("docs", judgment, probability, min_score=min_score)


def _judgment_payload(
    answers: Mapping[str, Any], route: CascadeRoute
) -> dict[str, dict]:
    payload = {
        key: {
            "answer": answer.get("choice", answer.get("noul")),
            "probs": answer.get("probabilities", {"true": answer.get("noul")}),
            "confidence": answer.get("confidence"),
        }
        for key, answer in answers.items()
        if isinstance(answer, Mapping)
    }
    payload["cascade:route"] = {
        "answer": route.route,
        "probs": {"acceptance": route.probability},
        "confidence": None,
    }
    return payload


def sync_route_holds(
    name_id: str, route: CascadeRoute, axis: str, *, graph: Any | None = None
) -> list[str]:
    """Reconcile the documentation holds a deciding route carries on its name.

    A veto reaches the graph as its own review hold rather than as a lowered
    score: the judged score is unchanged, so the refusal is visible as a
    distinct dimension a triager can act on and can dismiss.

    The reconciliation runs in both directions on every deciding pass, mirroring
    the per-pass recomputation the score mapping used to do. A check the route
    still fails raises its hold; a check it no longer fails clears the hold
    previously raised for it. Without the second direction a resolved
    contradiction would keep the document held for the life of the name — the
    hold would outlive the finding that produced it.

    ``axis`` is the axis this pass decided. Only a pass on the axis these checks
    run on has a verdict about their holds, so any other axis returns without
    touching them: its route carries none of these holds, and an empty set of
    raised holds is what a pass that found nothing looks like.
    """
    if axis != ROUTE_HOLD_AXIS:
        return []

    from imas_codex.standard_names.review.holds import sync_holds

    return sync_holds(
        name_id, ROUTE_HOLD_AXIS, ROUTE_HOLD_KINDS, route.holds, graph=graph
    )


async def _store_route(
    item: Mapping[str, Any],
    axis: str,
    model: str,
    route: CascadeRoute,
    answers: Mapping[str, Any],
    cost: float,
    *,
    reason: str | None = None,
) -> None:
    from imas_codex.standard_names.graph_store.graph_catalog import write_reviews
    from imas_codex.standard_names.review.pipeline import _build_review_record

    record = _build_review_record(
        dict(item),
        model=model,
        is_canonical=False,
        reviewed_at=datetime.now(UTC).isoformat(),
        judgment=_judgment_payload(answers, route),
        judgment_model=model,
        comments=json.dumps(
            {
                "route": route.route,
                "failed_checks": route.failed_checks,
                "rubric_total": route.total,
                "selected_score": route.selected_score,
                "decision_score": route.decision_score,
                "dimension_scores": dict(route.scores),
                "raw_dimension_scores": dict(route.raw_scores),
                "failed_check_details": _failed_check_instructions(route, item, axis),
                **({"failure_reason": reason} if reason is not None else {}),
            }
        ),
        cost_usd=cost,
    )
    record.update(
        review_axis=axis,
        cycle_index=-1,
        resolution_role="shadow",
        resolution_method="cascade_route",
    )
    await asyncio.to_thread(write_reviews, [record])


async def _store_unjudged(
    item: Mapping[str, Any], axis: str, model: str, reason: str
) -> None:
    try:
        await _store_route(
            item, axis, model, CascadeRoute("unjudged", None), {}, 0.0, reason=reason
        )
    except Exception:
        logger.exception(
            "Jev unjudged record could not be stored for %s", item.get("id")
        )


async def _incumbent_then_unjudged(
    item: Mapping[str, Any],
    axis: str,
    models: Sequence[str],
    incumbent: Callable[..., Awaitable[dict[str, Any] | None]],
    model: str,
    reason: str,
) -> dict[str, Any] | None:
    """Run the incumbent, then record the unjudged row beside what it wrote.

    A shadow row describes a candidate the incumbent has reviewed, so it is
    written only after the incumbent's own reviews are persisted; an incumbent
    that produced no reviews reports that with ``None`` and takes no row.
    """
    result = await incumbent(models=models)
    if result is None:
        return None
    await _store_unjudged(item, axis, model, reason)
    return result


def _failed_check_instructions(
    route: CascadeRoute, item: Mapping[str, Any], axis: str
) -> list[dict[str, Any]]:
    from imas_codex.standard_names.review.checks import load_review_checks

    taxonomy = {
        check.id: check for check in load_review_checks()[0] if check.axis == axis
    }
    instructions = []
    for check_id in sorted(set(route.failed_checks) | set(route.check_details)):
        check = taxonomy.get(check_id)
        if check is None:
            continue
        evidence = "; ".join(
            f"{field}={item[field]!r}"
            for field in check.evidence_fields
            if field in item and item[field] is not None
        )
        instructions.append(
            {
                "id": check_id,
                "criterion": check.question.text
                if check.question
                else check.audit_finding_class,
                "evidence": evidence
                or "No matching field was present in the review item",
                "probability": route.check_details.get(check_id, {}).get("probability"),
                "veto": check_id in route.failed_checks,
                "cutoff": route.check_details.get(check_id, {}).get("cutoff"),
                "judgment_evidence": route.check_details.get(check_id, {}).get(
                    "evidence"
                ),
                "veto_criterion": route.check_details.get(check_id, {}).get(
                    "criterion"
                ),
                "segment": str(item.get("id", "")) if axis == "name" else "",
                "sentence": str(item.get("documentation", ""))[:200]
                if axis == "docs"
                else "",
            }
        )
    return instructions


def failed_check_instructions(
    item: Mapping[str, Any], axis: str
) -> list[dict[str, Any]]:
    """Read a deciding route's named failures from the persisted review mirror."""
    raw = item.get(f"reviewer_comments_per_dim_{axis}")
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except ValueError:
            return []
    if not isinstance(raw, Mapping):
        return []
    checks = raw.get("failed_checks")
    return checks if isinstance(checks, list) else []


def _direct_result(
    route: CascadeRoute,
    model: str,
    cost: float,
    item: Mapping[str, Any],
    axis: str,
) -> dict[str, Any]:
    if route.total is None or route.decision_score is None:
        raise ValueError("a decided review requires dimension scores")
    return {
        "records": [],
        "winning_score": route.decision_score,
        "winning_scores": dict(route.scores),
        "winning_comments": "Calibrated Jev route: " + route.route,
        "winning_comments_per_dim": {
            "failed_checks": _failed_check_instructions(route, item, axis),
            "raw_dimension_scores": dict(route.raw_scores),
            "acceptance_probability": route.probability,
            "selected_score": route.selected_score,
        },
        "canonical_model": model,
        "total_cost": cost,
        "total_tokens_in": 0,
        "total_tokens_out": 0,
        "resolution_method": "jev_cascade_" + route.route,
        "dd_gaps": [],
    }


async def _name_judgment(
    item: Mapping[str, Any], model: str
) -> tuple[CascadeRoute, dict, float]:
    from imas_codex.standard_names.review.jev_review import review_names

    result = await review_names([item], model=model)
    if not result.items or not result.items[0].get("answered"):
        raise ValueError("no usable judgment for name review")
    judgment = result.items[0]
    route = route_name(judgment)
    answers = {
        **{f"dimension:{key}": value for key, value in judgment["categories"].items()},
        **{
            f"check:{key}": record.get("answer")
            for key, record in judgment["checks"].items()
        },
    }
    return route, answers, result.cost


async def _docs_judgment(
    item: Mapping[str, Any], model: str
) -> tuple[CascadeRoute, dict, float]:
    from imas_codex.standard_names.review import jev_review

    checks = [check for check in jev_review._default_checks() if check.axis == "docs"]
    state = jev_review.build_docs_state(item, checks=checks)
    questions = jev_review.docs_question_table(checks, state["documentation_claims"])
    results, batch_cost = await jev_review.decide_batch(
        [state], questions, model=model, service=jev_review.SERVICE
    )
    if not results or results[0] is None:
        raise ValueError("no usable judgment for documentation review")
    answers, _ = results[0]
    related = item.get("sibling_names") or []
    audit = jev_review.build_docs_audit_report(
        item, catalog_names=[item], known_ids={str(item["id"]), *map(str, related)}
    )
    judgment = jev_review.aggregate_docs_review(
        state, checks, answers, audit_report=audit
    )
    judgment["categories"] = {
        dimension: answers.get("dimension:" + dimension)
        for dimension in jev_review._DOCS_DIMENSION_CRITERIA
    }
    score = jev_review.score_docs_review(state, checks, answers, audit_report=audit)
    return route_docs(judgment, probability=score), answers, batch_cost


async def _audit_skipped(
    item: Mapping[str, Any],
    axis: str,
    jev_route: str,
    reason: str,
) -> None:
    from imas_codex.standard_names.review import panel_audit

    try:
        await panel_audit.store_panel_audit(
            item,
            axis,
            panel_result=None,
            jev_route=jev_route,
            skipped_reason=reason,
            min_score=DEFAULT_MIN_SCORE,
        )
    except Exception:
        logger.exception(
            "panel audit skip row could not be stored for %s", item.get("id")
        )


async def _run_panel_audit(
    axis: str,
    item: Mapping[str, Any],
    models: Sequence[str],
    incumbent: Callable[..., Awaitable[dict[str, Any] | None]],
    route: CascadeRoute,
    *,
    mgr: Any,
) -> None:
    """Run the paid panel on a sampled deciding item, beside Jev's route.

    The decision has already been stored and returned by the caller, so every
    failure here is recorded as a skipped audit rather than raised: an audit
    that cannot run must never withhold or change a decided item. The panel runs
    through the quorum's non-persisting path, so its own review rows are never
    written and the item's aggregate is untouched; the audit adds one
    non-deciding row carrying the verdict for comparison.
    """
    from imas_codex.standard_names.review import panel_audit

    if not panel_audit.audit_selected(
        axis,
        str(item.get("id", "")),
        str(getattr(mgr, "run_id", "")),
        panel_audit.panel_audit_rate(),
    ):
        return
    reason: str | None = None
    try:
        lease = mgr.reserve(0.001, phase="review_audit_" + axis)
    except Exception as exc:
        lease, reason = None, f"{type(exc).__name__}: {exc}"
    else:
        if lease is None:
            reason = "review audit lease refused"
        else:
            try:
                lease.release_unused()
            except Exception:
                logger.exception(
                    "panel audit lease release failed for %s", item.get("id")
                )
    if reason is not None:
        await _audit_skipped(item, axis, route.route, reason)
        return
    try:
        panel_result = await panel_audit.quorum_without_persisting(incumbent)(
            models=models
        )
    except Exception as exc:
        logger.exception("panel audit run failed for %s", item.get("id"))
        await _audit_skipped(item, axis, route.route, f"{type(exc).__name__}: {exc}")
        return
    if panel_result is None:
        await _audit_skipped(item, axis, route.route, "panel produced no verdict")
        return
    try:
        await panel_audit.store_panel_audit(
            item,
            axis,
            panel_result=panel_result,
            jev_route=route.route,
            min_score=DEFAULT_MIN_SCORE,
        )
    except Exception:
        logger.exception("panel audit row could not be stored for %s", item.get("id"))


async def adjudicate_review(
    axis: str,
    item: Mapping[str, Any],
    models: Sequence[str],
    incumbent: Callable[..., Awaitable[dict[str, Any] | None]],
    *,
    mgr: Any,
) -> dict[str, Any] | None:
    """Record a Jev route, then use the incumbent unless deciding is enabled.

    The incumbent is expected to persist its own review rows before it returns,
    so the route row lands after them and the incumbent's stored rows are
    unchanged. An incumbent that produced no reviews returns ``None`` and takes
    no route row: there is no reviewed candidate to route.
    """
    from imas_codex.discovery.base.judgment import (
        decisions_key_absent_reason,
        decisions_key_present,
    )
    from imas_codex.settings import get_model
    from imas_codex.standard_names.budget import LLMCostEvent

    model = "sn-judge:unavailable"
    if not decisions_key_present():
        return await _incumbent_then_unjudged(
            item,
            axis,
            models,
            incumbent,
            model,
            decisions_key_absent_reason() or "decisions key not set",
        )
    try:
        model = get_model("sn-judge")
        lease = mgr.reserve(0.001, phase="review_" + axis)
    except Exception as exc:
        logger.exception("Jev cascade reservation unavailable for %s", item.get("id"))
        reason = f"{type(exc).__name__}: {exc}"
    else:
        if lease is None:
            reason = "review lease refused"
        else:
            reason = None
    if reason is not None:
        return await _incumbent_then_unjudged(
            item, axis, models, incumbent, model, reason
        )
    try:
        if axis == "name":
            route, answers, spent = await _name_judgment(item, model)
        elif axis == "docs":
            route, answers, spent = await _docs_judgment(item, model)
        else:
            raise ValueError(f"unknown review axis {axis!r}")
        lease.charge_event(
            spent,
            LLMCostEvent(
                model=model,
                tokens_in=0,
                tokens_out=0,
                sn_ids=(str(item["id"]),),
                phase="review_" + axis,
            ),
        )
    except Exception as exc:
        logger.exception("Jev cascade judgment unavailable for %s", item.get("id"))
        return await _incumbent_then_unjudged(
            item, axis, models, incumbent, model, f"{type(exc).__name__}: {exc}"
        )
    finally:
        try:
            lease.release_unused()
        except Exception:
            logger.exception("Jev cascade lease release failed for %s", item.get("id"))
    if deciding_enabled(axis):
        try:
            await _store_route(item, axis, model, route, answers, spent)
        except Exception:
            logger.exception(
                "Jev cascade route could not be stored for %s", item.get("id")
            )
        try:
            # Run for every deciding pass, not only a vetoing one: a pass that
            # reports no contradiction is what clears the hold an earlier pass
            # raised. The pass's own axis decides whether it may: only the axis
            # these checks run on has a verdict about their holds.
            await asyncio.to_thread(
                sync_route_holds, str(item.get("id", "")), route, axis
            )
        except Exception:
            logger.exception(
                "Jev cascade holds could not be reconciled for %s", item.get("id")
            )
        try:
            await _run_panel_audit(axis, item, models, incumbent, route, mgr=mgr)
        except Exception:
            logger.exception("panel audit failed for %s", item.get("id"))
        return _direct_result(route, model, spent, item, axis)
    result = await incumbent(models=models)
    if result is None:
        return None
    try:
        await _store_route(item, axis, model, route, answers, spent)
    except Exception:
        logger.exception("Jev cascade route could not be stored for %s", item.get("id"))
    return result
