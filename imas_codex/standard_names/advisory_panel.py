"""Read-only reviewer-panel adjudication for benchmark artifacts.

The production review engine owns graph persistence and lifecycle transitions.
This module deliberately stays outside that engine: it calls the benchmark's
query-only reviewer scorer, preserves every seat's response, and derives panel
statistics entirely in memory.
"""

from __future__ import annotations

import asyncio
import copy
import math
import statistics
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Any

from imas_codex import settings
from imas_codex.standard_names import benchmark

ReviewerScorer = Callable[
    [list[dict[str, Any]], str, str, str | None],
    Awaitable[tuple[list[dict[str, Any]], float]],
]


@dataclass(frozen=True)
class PanelSeatJudgment:
    """One review seat's unmodified judgment for one benchmark row."""

    seat_index: int
    reviewer_model: str
    judgment: dict[str, Any]


@dataclass(frozen=True)
class PanelRowAdjudication:
    """All seat judgments and derived consensus statistics for one row."""

    row_index: int
    standard_name: str
    seat_judgments: tuple[PanelSeatJudgment, ...]
    median_score: float
    score_spread: float
    contested: bool


@dataclass(frozen=True)
class PanelSeatSummary:
    """Artifact-local accounting for one independent reviewer seat."""

    seat_index: int
    reviewer_model: str
    cost: float


@dataclass(frozen=True)
class PanelAdjudication:
    """A complete non-persisted result for one frozen candidate batch."""

    rows: tuple[PanelRowAdjudication, ...]
    seats: tuple[PanelSeatSummary, ...]
    disagreement_threshold: float
    total_cost: float


async def adjudicate_with_production_panel(
    candidates: Sequence[dict[str, Any]],
    *,
    reviewer_models: Sequence[str] | None = None,
    disagreement_threshold: float | None = None,
    target: str = "names",
    reasoning_effort: str | None = None,
    scorer: ReviewerScorer | None = None,
) -> PanelAdjudication:
    """Score a frozen batch independently with exactly three review seats.

    By default the adapter resolves the named ``default`` review profile
    explicitly.  This prevents a process-level profile override from silently
    replacing the hosted production reference panel with a local-only profile.
    The scorer is the benchmark's production-context, query-only scorer; no
    review record, lifecycle state, embedding, or cost node is written here.

    The returned row order matches ``candidates``.  Every seat must return
    exactly one uniquely named review for every input row.  Partial or
    ambiguous panels fail closed instead of publishing a plausible consensus
    over incomplete evidence.
    """
    models = tuple(
        reviewer_models
        if reviewer_models is not None
        else settings.get_sn_review_profile_models("default")
    )
    _validate_models(models)

    threshold = (
        disagreement_threshold
        if disagreement_threshold is not None
        else settings.get_sn_review_profile_threshold("default")
    )
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError(
            "disagreement_threshold must be a finite value between 0 and 1"
        )

    frozen_candidates = copy.deepcopy(list(candidates))
    row_names = tuple(benchmark._resolve_name(row) for row in frozen_candidates)
    if any(not name for name in row_names):
        raise ValueError("every panel row must resolve to a non-empty standard name")
    if len(set(row_names)) != len(row_names):
        raise ValueError("panel row standard names must be unique")

    if not frozen_candidates:
        return PanelAdjudication(
            rows=(),
            seats=tuple(
                PanelSeatSummary(index, model, 0.0)
                for index, model in enumerate(models)
            ),
            disagreement_threshold=threshold,
            total_cost=0.0,
        )

    score_rows = scorer or _score_rows
    seat_outputs = await asyncio.gather(
        *(
            score_rows(
                copy.deepcopy(frozen_candidates),
                model,
                target,
                reasoning_effort,
            )
            for model in models
        )
    )

    judgments_by_seat: list[dict[str, dict[str, Any]]] = []
    seat_summaries: list[PanelSeatSummary] = []
    for seat_index, (model, (judgments, cost)) in enumerate(
        zip(models, seat_outputs, strict=True)
    ):
        if not math.isfinite(cost) or cost < 0.0:
            raise ValueError(f"review seat {seat_index} returned invalid cost {cost!r}")
        by_name = _index_judgments(judgments, row_names, seat_index)
        judgments_by_seat.append(by_name)
        seat_summaries.append(PanelSeatSummary(seat_index, model, float(cost)))

    rows: list[PanelRowAdjudication] = []
    for row_index, name in enumerate(row_names):
        seat_judgments = tuple(
            PanelSeatJudgment(
                seat_index=seat_index,
                reviewer_model=model,
                judgment=copy.deepcopy(judgments_by_seat[seat_index][name]),
            )
            for seat_index, model in enumerate(models)
        )
        scores = tuple(float(item.judgment["score"]) for item in seat_judgments)
        median_score = float(statistics.median(scores))
        score_spread = max(scores) - min(scores)
        rows.append(
            PanelRowAdjudication(
                row_index=row_index,
                standard_name=name,
                seat_judgments=seat_judgments,
                median_score=median_score,
                score_spread=score_spread,
                contested=score_spread > threshold,
            )
        )

    total_cost = sum(seat.cost for seat in seat_summaries)
    return PanelAdjudication(
        rows=tuple(rows),
        seats=tuple(seat_summaries),
        disagreement_threshold=threshold,
        total_cost=total_cost,
    )


async def _score_rows(
    candidates: list[dict[str, Any]],
    reviewer_model: str,
    target: str,
    reasoning_effort: str | None,
) -> tuple[list[dict[str, Any]], float]:
    return await benchmark.score_with_reviewer(
        candidates,
        reviewer_model,
        target=target,
        reasoning_effort=reasoning_effort,
    )


def _validate_models(models: tuple[str, ...]) -> None:
    if len(models) != 3:
        raise ValueError(
            f"panel adjudication requires exactly three reviewer models; got {len(models)}"
        )
    if len(set(models)) != len(models):
        raise ValueError("panel reviewer models must be unique")
    if any(not isinstance(model, str) or not model.strip() for model in models):
        raise ValueError("panel reviewer models must be non-empty strings")


def _index_judgments(
    judgments: list[dict[str, Any]],
    expected_names: tuple[str, ...],
    seat_index: int,
) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for judgment in judgments:
        name = judgment.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError(
                f"review seat {seat_index} returned a judgment without a name"
            )
        if name in indexed:
            raise ValueError(
                f"review seat {seat_index} returned duplicate judgments for {name!r}"
            )
        score = judgment.get("score")
        if (
            isinstance(score, bool)
            or not isinstance(score, int | float)
            or not math.isfinite(score)
            or not 0.0 <= score <= 1.0
        ):
            raise ValueError(
                f"review seat {seat_index} returned invalid score for {name!r}: {score!r}"
            )
        indexed[name] = judgment

    expected = set(expected_names)
    actual = set(indexed)
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise ValueError(
            f"review seat {seat_index} did not cover the frozen batch exactly; "
            f"missing={missing}, unexpected={unexpected}"
        )
    return indexed
