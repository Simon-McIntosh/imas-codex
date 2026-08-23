"""Priced holdout evaluation for standard-name documentation arms."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from imas_codex.llm.prompt_loader import render_prompt
from imas_codex.standard_names.benchmark import generate_docs_for_candidates
from imas_codex.standard_names.benchmark_reference import (
    DocsHoldoutRow,
    load_docs_holdout,
)
from imas_codex.standard_names.budget import model_provider_exposure
from imas_codex.standard_names.context import build_compose_context
from imas_codex.standard_names.docs_gates import (
    DOCUMENTATION_GATE_NAMES,
    DocumentationGateScore,
    score_documentation,
)
from imas_codex.standard_names.models import GeneratedDocs

_GENERATION_ATTEMPTS = 2


class HoldoutCostCeilingExceeded(RuntimeError):
    """Raised before generation when a priced arm exceeds its authority."""


@dataclass(frozen=True)
class GateComparison:
    """Aggregate catalog and arm results for one deterministic gate."""

    gate: str
    arm_pass_count: int
    catalog_pass_count: int
    total_count: int
    arm_pass_rate: float
    catalog_pass_rate: float
    pass_rate_delta: float


@dataclass(frozen=True)
class HoldoutRowScore:
    """Gate vectors for one arm output and its catalog counterpart."""

    split_key: str
    dd_path: str
    catalog_name: str
    arm_gates: dict[str, bool]
    catalog_gates: dict[str, bool]


@dataclass(frozen=True)
class DocsHoldoutReport:
    """Evaluation receipt for a named documentation arm."""

    arm: str
    model: str | None
    dry_run: bool
    row_count: int
    projected_call_count: int
    projected_cost_usd: float
    actual_cost_usd: float
    per_gate_table: tuple[GateComparison, ...]
    row_scores: tuple[HoldoutRowScore, ...]
    overall_pass_rate: float | None
    catalog_overall_pass_rate: float | None
    overall_pass_rate_delta: float | None


def _candidate_for(row: DocsHoldoutRow) -> dict[str, Any]:
    return {
        "standard_name": row["catalog_name"],
        "source_id": row["dd_path"],
        "description": row["catalog_description"],
        "unit": "",
        "kind": "scalar",
        "physics_domain": "",
        "source_paths": [row["dd_path"]],
    }


def _generation_item(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": candidate["standard_name"],
        "unit": candidate.get("unit", ""),
        "kind": candidate.get("kind", "scalar"),
        "physics_domain": candidate.get("physics_domain", ""),
        "description": candidate.get("description", ""),
        "source_paths": candidate.get("source_paths", []),
        "reviewer_score_name": None,
        "reviewer_comments_name": "",
        "chain_history": [],
    }


def _project_request_cost(
    model: str,
    messages: list[dict[str, Any]],
) -> float:
    return model_provider_exposure(
        model,
        messages,
        response_model=GeneratedDocs,
        provider_attempts=_GENERATION_ATTEMPTS,
    )


def _price_generation(
    candidates: Sequence[dict[str, Any]],
    *,
    model: str,
    context: dict[str, Any],
) -> float:
    """Price every rendered benchmark request before any call can execute."""
    system_prompt = render_prompt("sn/generate_docs_system", context)
    projected_cost = 0.0
    for candidate in candidates:
        item = _generation_item(candidate)
        user_prompt = render_prompt(
            "sn/generate_docs_user",
            {**context, "item": item},
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        projected_cost += _project_request_cost(model, messages)
    return projected_cost


def _aggregate_scores(
    arm_scores: Sequence[DocumentationGateScore],
    catalog_scores: Sequence[DocumentationGateScore],
) -> tuple[tuple[GateComparison, ...], float, float, float]:
    total_rows = len(arm_scores)
    if total_rows == 0 or len(catalog_scores) != total_rows:
        raise ValueError("Holdout evaluation requires paired, non-empty scores")

    table: list[GateComparison] = []
    for gate in DOCUMENTATION_GATE_NAMES:
        arm_passes = sum(score.gate_vector[gate] for score in arm_scores)
        catalog_passes = sum(score.gate_vector[gate] for score in catalog_scores)
        arm_rate = arm_passes / total_rows
        catalog_rate = catalog_passes / total_rows
        table.append(
            GateComparison(
                gate=gate,
                arm_pass_count=arm_passes,
                catalog_pass_count=catalog_passes,
                total_count=total_rows,
                arm_pass_rate=arm_rate,
                catalog_pass_rate=catalog_rate,
                pass_rate_delta=arm_rate - catalog_rate,
            )
        )

    gate_instances = total_rows * len(DOCUMENTATION_GATE_NAMES)
    arm_overall = sum(score.passed_count for score in arm_scores) / gate_instances
    catalog_overall = (
        sum(score.passed_count for score in catalog_scores) / gate_instances
    )
    return tuple(table), arm_overall, catalog_overall, arm_overall - catalog_overall


async def evaluate_docs_holdout(
    arm: str,
    *,
    model: str | None = None,
    dry_run: bool = False,
    cost_ceiling: float | None = None,
    rows: Sequence[DocsHoldoutRow] | None = None,
) -> DocsHoldoutReport:
    """Evaluate a named generated arm against catalog documentation.

    ``model=None`` selects the catalog documentation itself as the arm. A
    generated arm renders and prices every request before it may call the
    existing benchmark documentation generator. ``dry_run`` returns that
    projection without model calls, and ``cost_ceiling`` refuses unauthorised
    exposure before generation starts.
    """
    if not arm.strip():
        raise ValueError("Documentation arm must have a non-empty name")
    if cost_ceiling is not None and (
        not math.isfinite(cost_ceiling) or cost_ceiling < 0
    ):
        raise ValueError("Cost ceiling must be finite and non-negative")

    holdout = list(rows if rows is not None else load_docs_holdout())
    if not holdout:
        raise ValueError("Documentation holdout must not be empty")

    candidates = [_candidate_for(row) for row in holdout]
    context: dict[str, Any] | None = None
    projected_calls = 0
    projected_cost = 0.0
    if model is not None:
        context = build_compose_context()
        context["compose_scored_examples"] = []
        projected_calls = len(candidates)
        projected_cost = _price_generation(candidates, model=model, context=context)

    if cost_ceiling is not None and projected_cost > cost_ceiling:
        raise HoldoutCostCeilingExceeded(
            f"arm {arm!r} projects ${projected_cost:.6f}, "
            f"above the ${cost_ceiling:.6f} ceiling"
        )

    if dry_run:
        return DocsHoldoutReport(
            arm=arm,
            model=model,
            dry_run=True,
            row_count=len(holdout),
            projected_call_count=projected_calls,
            projected_cost_usd=projected_cost,
            actual_cost_usd=0.0,
            per_gate_table=(),
            row_scores=(),
            overall_pass_rate=None,
            catalog_overall_pass_rate=None,
            overall_pass_rate_delta=None,
        )

    actual_cost = 0.0
    if model is None:
        arm_documents = [row["catalog_documentation"] for row in holdout]
    else:
        assert context is not None
        generated, actual_cost, _elapsed = await generate_docs_for_candidates(
            candidates,
            model,
            context,
        )
        arm_documents = [candidate.get("documentation", "") for candidate in generated]

    catalog_scores = [
        score_documentation(row["catalog_documentation"]) for row in holdout
    ]
    arm_scores = [score_documentation(documentation) for documentation in arm_documents]
    table, arm_overall, catalog_overall, overall_delta = _aggregate_scores(
        arm_scores,
        catalog_scores,
    )
    row_scores = tuple(
        HoldoutRowScore(
            split_key=row["split_key"],
            dd_path=row["dd_path"],
            catalog_name=row["catalog_name"],
            arm_gates=dict(arm_score.gate_vector),
            catalog_gates=dict(catalog_score.gate_vector),
        )
        for row, arm_score, catalog_score in zip(
            holdout,
            arm_scores,
            catalog_scores,
            strict=True,
        )
    )
    return DocsHoldoutReport(
        arm=arm,
        model=model,
        dry_run=False,
        row_count=len(holdout),
        projected_call_count=projected_calls,
        projected_cost_usd=projected_cost,
        actual_cost_usd=actual_cost,
        per_gate_table=table,
        row_scores=row_scores,
        overall_pass_rate=arm_overall,
        catalog_overall_pass_rate=catalog_overall,
        overall_pass_rate_delta=overall_delta,
    )


__all__ = [
    "DocsHoldoutReport",
    "GateComparison",
    "HoldoutCostCeilingExceeded",
    "HoldoutRowScore",
    "evaluate_docs_holdout",
]
