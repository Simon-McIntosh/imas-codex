"""Priced holdout evaluation for standard-name documentation arms."""

from __future__ import annotations

import json
import math
import time
from asyncio import Semaphore, gather
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from imas_codex.llm.prompt_loader import render_prompt
from imas_codex.standard_names.benchmark_reference import (
    DocsHoldoutRow,
    load_docs_holdout,
)
from imas_codex.standard_names.budget import model_provider_exposure
from imas_codex.standard_names.context import build_compose_context
from imas_codex.standard_names.docs_gates import (
    DOCUMENTATION_GATE_NAMES,
    DocumentationGateOutcome,
    DocumentationGateResult,
    DocumentationGateScore,
    DocumentationPhysicsContext,
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
    arm_contradiction_count: int
    arm_not_evaluable_count: int
    arm_evaluable_count: int
    catalog_pass_count: int
    catalog_contradiction_count: int
    catalog_not_evaluable_count: int
    catalog_evaluable_count: int
    total_count: int
    arm_pass_rate: float | None
    catalog_pass_rate: float | None
    pass_rate_delta: float | None


@dataclass(frozen=True)
class HoldoutRowScore:
    """Gate vectors for one arm output and its catalog counterpart."""

    split_key: str
    dd_path: str
    catalog_name: str
    physics_context: DocumentationPhysicsContext
    generated_documentation: str | None
    catalog_documentation: str
    arm_gates: dict[str, DocumentationGateResult]
    catalog_gates: dict[str, DocumentationGateResult]


@dataclass(frozen=True)
class HoldoutContextCount:
    """Production relationship-context candidates available for one row."""

    split_key: str
    dd_path: str
    candidate_count: int


@dataclass(frozen=True)
class _PreparedDocsRequest:
    """A production-enriched, fully rendered holdout request."""

    row_index: int
    messages: list[dict[str, Any]]


@dataclass(frozen=True)
class DocsHoldoutReport:
    """Evaluation receipt for a named documentation arm."""

    arm: str
    model: str | None
    dry_run: bool
    row_count: int
    projected_call_count: int
    projected_cost_usd: float
    actual_call_count: int
    actual_cost_usd: float
    scored_row_count: int
    zero_candidate_row_count: int
    context_counts: tuple[HoldoutContextCount, ...]
    per_gate_table: tuple[GateComparison, ...]
    row_scores: tuple[HoldoutRowScore, ...]
    overall_pass_rate: float | None
    catalog_overall_pass_rate: float | None
    overall_pass_rate_delta: float | None


def write_docs_holdout_receipt(
    report: DocsHoldoutReport,
    receipt_path: Path,
) -> None:
    """Atomically persist a complete holdout report, including scored prose."""

    path = receipt_path.expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    generated_documentation_row_count = sum(
        bool(row.generated_documentation and row.generated_documentation.strip())
        for row in report.row_scores
    )
    payload = {
        "schema_version": 1,
        "generated_documentation_row_count": generated_documentation_row_count,
        **asdict(report),
    }
    with NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as temporary:
        temporary.write(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
        temporary.write("\n")
        temporary_path = Path(temporary.name)
    temporary_path.replace(path)


def _persist_if_requested(
    report: DocsHoldoutReport,
    receipt_path: Path | None,
) -> DocsHoldoutReport:
    if receipt_path is not None:
        write_docs_holdout_receipt(report, receipt_path)
    return report


def _candidate_for(row: DocsHoldoutRow) -> dict[str, Any]:
    return {
        "split_key": row["split_key"],
        "standard_name": row["catalog_name"],
        "source_id": row["dd_path"],
        "description": row["catalog_description"],
        "unit": "",
        "kind": "scalar",
        "physics_domain": "",
        "source_paths": [row["dd_path"]],
    }


def _physics_context_for(row: DocsHoldoutRow) -> DocumentationPhysicsContext:
    """Retain the pinned authority fields supplied by a holdout row."""
    row_data: Mapping[str, Any] = row
    return DocumentationPhysicsContext(
        dd_path=row["dd_path"],
        declared_unit=row_data.get("declared_unit"),
        cocos_transformation_type=row_data.get("cocos_transformation_type"),
        cocos_params=row_data.get("cocos_params"),
    )


def _generation_item(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": candidate["standard_name"],
        "name": candidate["standard_name"],
        "standard_name": candidate["standard_name"],
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


def _production_enrich_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Run the same graph enrichment used by the production docs worker."""
    from imas_codex.graph.client import GraphClient
    from imas_codex.settings import get_dd_version
    from imas_codex.standard_names.workers import (
        _DOCS_GEN_COCOS_PARAMS_QUERY,
        _enrich_for_docs_gen,
        _nearby_names_for_docs_gen,
    )

    with GraphClient() as gc:
        cocos_rows = list(gc.query(_DOCS_GEN_COCOS_PARAMS_QUERY, ver=get_dd_version()))
        cocos_params = cocos_rows[0]["cocos_params"] if cocos_rows else None
        _enrich_for_docs_gen(gc, items, cocos_params=cocos_params)
        return _nearby_names_for_docs_gen(gc, items)


def _relationship_context_candidate_count(item: dict[str, Any]) -> int:
    """Count candidate-specific related, parent, component, and peer context."""
    return (
        len(item.get("related_neighbours") or [])
        + len(item.get("nearest_peers") or [])
        + len(item.get("child_components") or [])
        + int(bool(item.get("parent_sn")))
    )


def _prepare_generation_requests(
    candidates: Sequence[dict[str, Any]],
    *,
    context: dict[str, Any],
) -> tuple[list[_PreparedDocsRequest], tuple[HoldoutContextCount, ...]]:
    """Enrich and render eligible rows through the production docs path."""
    from imas_codex.standard_names.context import locus_context_for

    items = [_generation_item(candidate) for candidate in candidates]
    nearby_existing_names = _production_enrich_items(items)
    requests: list[_PreparedDocsRequest] = []
    counts: list[HoldoutContextCount] = []
    for row_index, (candidate, item) in enumerate(zip(candidates, items, strict=True)):
        candidate_count = _relationship_context_candidate_count(item)
        counts.append(
            HoldoutContextCount(
                split_key=candidate["split_key"],
                dd_path=candidate["source_id"],
                candidate_count=candidate_count,
            )
        )
        if candidate_count == 0:
            continue
        prompt_context = {
            **context,
            "item": item,
            "chain_history": item.get("chain_history") or [],
            "nearby_existing_names": nearby_existing_names,
            "locus_context": locus_context_for(item["id"]),
        }
        user_prompt = render_prompt(
            "sn/generate_docs_user",
            prompt_context,
        )
        system_prompt = render_prompt("sn/generate_docs_system", prompt_context)
        requests.append(
            _PreparedDocsRequest(
                row_index=row_index,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
        )
    return requests, tuple(counts)


def _price_generation(requests: Sequence[_PreparedDocsRequest], *, model: str) -> float:
    """Price every production-rendered request before any call can execute."""
    return sum(_project_request_cost(model, request.messages) for request in requests)


async def _call_docs_model(
    model: str, messages: list[dict[str, Any]]
) -> tuple[GeneratedDocs, float]:
    """Call the configured docs model with production generation settings."""
    from imas_codex.discovery.base.llm import acall_llm_structured
    from imas_codex.settings import get_reasoning_effort

    result, cost, _tokens = await acall_llm_structured(
        model=model,
        messages=messages,
        response_model=GeneratedDocs,
        temperature=None if "gpt-5" in model else 0.0,
        service="standard-names",
        reasoning_effort=get_reasoning_effort("sn-docs"),
        max_retries=_GENERATION_ATTEMPTS,
    )
    return result, cost


async def _generate_requests(
    requests: Sequence[_PreparedDocsRequest], model: str
) -> tuple[list[str], float, float]:
    """Generate enriched requests concurrently and retain their row order."""
    started = time.monotonic()
    semaphore = Semaphore(8)

    async def _one(request: _PreparedDocsRequest) -> tuple[str, float]:
        async with semaphore:
            result, cost = await _call_docs_model(model, request.messages)
        return result.documentation, cost

    outputs = await gather(*(_one(request) for request in requests))
    return (
        [documentation for documentation, _cost in outputs],
        sum(cost for _documentation, cost in outputs),
        time.monotonic() - started,
    )


def _aggregate_scores(
    arm_scores: Sequence[DocumentationGateScore],
    catalog_scores: Sequence[DocumentationGateScore],
) -> tuple[
    tuple[GateComparison, ...],
    float | None,
    float | None,
    float | None,
]:
    total_rows = len(arm_scores)
    if total_rows == 0 or len(catalog_scores) != total_rows:
        raise ValueError("Holdout evaluation requires paired, non-empty scores")

    table: list[GateComparison] = []
    for gate in DOCUMENTATION_GATE_NAMES:
        arm_outcomes = [score.gate_vector[gate].outcome for score in arm_scores]
        catalog_outcomes = [score.gate_vector[gate].outcome for score in catalog_scores]
        arm_passes = arm_outcomes.count(DocumentationGateOutcome.PASS)
        arm_failures = arm_outcomes.count(DocumentationGateOutcome.FAIL)
        arm_not_evaluable = arm_outcomes.count(DocumentationGateOutcome.NOT_EVALUABLE)
        catalog_passes = catalog_outcomes.count(DocumentationGateOutcome.PASS)
        catalog_failures = catalog_outcomes.count(DocumentationGateOutcome.FAIL)
        catalog_not_evaluable = catalog_outcomes.count(
            DocumentationGateOutcome.NOT_EVALUABLE
        )
        arm_evaluable = arm_passes + arm_failures
        catalog_evaluable = catalog_passes + catalog_failures
        arm_rate = arm_passes / arm_evaluable if arm_evaluable else None
        catalog_rate = catalog_passes / catalog_evaluable if catalog_evaluable else None
        rate_delta = (
            arm_rate - catalog_rate
            if arm_rate is not None and catalog_rate is not None
            else None
        )
        table.append(
            GateComparison(
                gate=gate,
                arm_pass_count=arm_passes,
                arm_contradiction_count=arm_failures,
                arm_not_evaluable_count=arm_not_evaluable,
                arm_evaluable_count=arm_evaluable,
                catalog_pass_count=catalog_passes,
                catalog_contradiction_count=catalog_failures,
                catalog_not_evaluable_count=catalog_not_evaluable,
                catalog_evaluable_count=catalog_evaluable,
                total_count=total_rows,
                arm_pass_rate=arm_rate,
                catalog_pass_rate=catalog_rate,
                pass_rate_delta=rate_delta,
            )
        )

    arm_evaluable = sum(score.evaluable_count for score in arm_scores)
    catalog_evaluable = sum(score.evaluable_count for score in catalog_scores)
    arm_overall = (
        sum(score.passed_count for score in arm_scores) / arm_evaluable
        if arm_evaluable
        else None
    )
    catalog_overall = (
        sum(score.passed_count for score in catalog_scores) / catalog_evaluable
        if catalog_evaluable
        else None
    )
    overall_delta = (
        arm_overall - catalog_overall
        if arm_overall is not None and catalog_overall is not None
        else None
    )
    return tuple(table), arm_overall, catalog_overall, overall_delta


async def evaluate_docs_holdout(
    arm: str,
    *,
    model: str | None = None,
    dry_run: bool = False,
    cost_ceiling: float | None = None,
    rows: Sequence[DocsHoldoutRow] | None = None,
    receipt_path: Path | None = None,
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
    requests: list[_PreparedDocsRequest] = []
    context_counts: tuple[HoldoutContextCount, ...] = ()
    projected_calls = 0
    projected_cost = 0.0
    if model is not None:
        context = build_compose_context()
        context["compose_scored_examples"] = []
        requests, context_counts = _prepare_generation_requests(
            candidates, context=context
        )
        projected_calls = len(requests)
        projected_cost = _price_generation(requests, model=model)

    zero_candidate_rows = sum(count.candidate_count == 0 for count in context_counts)

    if cost_ceiling is not None and projected_cost > cost_ceiling:
        raise HoldoutCostCeilingExceeded(
            f"arm {arm!r} projects ${projected_cost:.6f}, "
            f"above the ${cost_ceiling:.6f} ceiling"
        )

    if dry_run:
        return _persist_if_requested(
            DocsHoldoutReport(
                arm=arm,
                model=model,
                dry_run=True,
                row_count=len(holdout),
                projected_call_count=projected_calls,
                projected_cost_usd=projected_cost,
                actual_call_count=0,
                actual_cost_usd=0.0,
                scored_row_count=0,
                zero_candidate_row_count=zero_candidate_rows,
                context_counts=context_counts,
                per_gate_table=(),
                row_scores=(),
                overall_pass_rate=None,
                catalog_overall_pass_rate=None,
                overall_pass_rate_delta=None,
            ),
            receipt_path,
        )

    actual_cost = 0.0
    if model is None:
        arm_documents = [row["catalog_documentation"] for row in holdout]
        scored_rows = holdout
    else:
        assert context is not None
        arm_documents, actual_cost, _elapsed = await _generate_requests(requests, model)
        scored_rows = [holdout[request.row_index] for request in requests]

    if not scored_rows:
        return _persist_if_requested(
            DocsHoldoutReport(
                arm=arm,
                model=model,
                dry_run=False,
                row_count=len(holdout),
                projected_call_count=projected_calls,
                projected_cost_usd=projected_cost,
                actual_call_count=0,
                actual_cost_usd=actual_cost,
                scored_row_count=0,
                zero_candidate_row_count=zero_candidate_rows,
                context_counts=context_counts,
                per_gate_table=(),
                row_scores=(),
                overall_pass_rate=None,
                catalog_overall_pass_rate=None,
                overall_pass_rate_delta=None,
            ),
            receipt_path,
        )

    physics_contexts = [_physics_context_for(row) for row in scored_rows]
    catalog_scores = [
        score_documentation(
            row["catalog_documentation"],
            physics_context=physics_context,
        )
        for row, physics_context in zip(
            scored_rows,
            physics_contexts,
            strict=True,
        )
    ]
    arm_scores = [
        score_documentation(
            documentation,
            physics_context=physics_context,
        )
        for documentation, physics_context in zip(
            arm_documents,
            physics_contexts,
            strict=True,
        )
    ]
    table, arm_overall, catalog_overall, overall_delta = _aggregate_scores(
        arm_scores,
        catalog_scores,
    )
    row_scores = tuple(
        HoldoutRowScore(
            split_key=row["split_key"],
            dd_path=row["dd_path"],
            catalog_name=row["catalog_name"],
            physics_context=physics_context,
            generated_documentation=(documentation if model is not None else None),
            catalog_documentation=row["catalog_documentation"],
            arm_gates=dict(arm_score.gate_vector),
            catalog_gates=dict(catalog_score.gate_vector),
        )
        for row, documentation, arm_score, catalog_score, physics_context in zip(
            scored_rows,
            arm_documents,
            arm_scores,
            catalog_scores,
            physics_contexts,
            strict=True,
        )
    )
    return _persist_if_requested(
        DocsHoldoutReport(
            arm=arm,
            model=model,
            dry_run=False,
            row_count=len(holdout),
            projected_call_count=projected_calls,
            projected_cost_usd=projected_cost,
            actual_call_count=(len(arm_documents) if model is not None else 0),
            actual_cost_usd=actual_cost,
            scored_row_count=len(scored_rows),
            zero_candidate_row_count=zero_candidate_rows,
            context_counts=context_counts,
            per_gate_table=table,
            row_scores=row_scores,
            overall_pass_rate=arm_overall,
            catalog_overall_pass_rate=catalog_overall,
            overall_pass_rate_delta=overall_delta,
        ),
        receipt_path,
    )


__all__ = [
    "DocsHoldoutReport",
    "GateComparison",
    "HoldoutContextCount",
    "HoldoutCostCeilingExceeded",
    "HoldoutRowScore",
    "evaluate_docs_holdout",
    "write_docs_holdout_receipt",
]
