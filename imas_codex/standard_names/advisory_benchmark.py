"""Frozen, artifact-only benchmark execution for advisory name review.

The caller supplies a read-only population projection.  This module freezes a
domain-balanced sample, sends the same immutable rows to one advisory candidate
and the production review panel, and publishes the resulting evidence without
using graph persistence or budget-settlement infrastructure.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import math
import os
import tempfile
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from imas_codex.standard_names import benchmark, benchmark_roles
from imas_codex.standard_names.advisory_panel import (
    PanelAdjudication,
    adjudicate_with_production_panel,
)

ReviewerScorer = Callable[
    [list[dict[str, Any]], str, str, str | None],
    Awaitable[tuple[list[dict[str, Any]], float]],
]

ADVISORY_CANDIDATE_TEMPERATURE = 0.0


class CandidateScorer(Protocol):
    """Production-context scorer with an explicit decoding envelope."""

    def __call__(
        self,
        candidates: list[dict[str, Any]],
        reviewer_model: str,
        target: str,
        reasoning_effort: str | None,
        *,
        temperature: float,
        seed: int,
        rendered_message_hashes: list[str],
    ) -> Awaitable[tuple[list[dict[str, Any]], float]]: ...


@dataclass(frozen=True)
class ArtifactCostSummary:
    """Provider-returned costs recorded only in the published artifact."""

    candidate_cost: float
    judging_cost: float
    total_cost: float
    authorized_ceiling: float
    remaining_authority: float
    within_authorized_ceiling: bool


@dataclass(frozen=True)
class FrozenResultRow:
    """One frozen input paired with candidate and panel judgments."""

    row_index: int
    standard_name: str
    input_row: dict[str, Any]
    candidate_judgment: dict[str, Any]
    panel_median_score: float
    panel_score_spread: float
    panel_contested: bool
    panel_seat_judgments: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class FrozenBenchmarkReport:
    """Reproducible benchmark inputs, outputs, provenance, and local costs."""

    candidate_model: str
    candidate_temperature: float
    candidate_seed: int
    candidate_rendered_message_hashes: tuple[str, ...]
    seed: int
    sample_size: int
    population_size: int
    population_hash: str
    ordered_input_hash: str
    ordered_result_hash: str
    input_rows: tuple[dict[str, Any], ...]
    result_rows: tuple[FrozenResultRow, ...]
    reviewer_models: tuple[str, ...]
    disagreement_threshold: float
    costs: ArtifactCostSummary
    source_provenance: dict[str, Any]
    dictionary_provenance: dict[str, str]
    implementation_provenance: dict[str, str]
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        """Return the exact JSON-compatible artifact payload."""
        return json.loads(
            json.dumps(
                asdict(self),
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
        )

    def to_json(self) -> str:
        """Serialize the artifact with stable key ordering."""
        return json.dumps(
            self.to_dict(),
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )

    def publish_atomic(self, path: str | Path) -> None:
        """Crash-safely replace *path* with this complete report."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_path = tempfile.mkstemp(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(self.to_json())
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, destination)
            directory_descriptor = os.open(destination.parent, os.O_RDONLY)
            try:
                os.fsync(directory_descriptor)
            finally:
                os.close(directory_descriptor)
        except BaseException:
            try:
                os.unlink(temporary_path)
            except OSError:
                pass
            raise


async def run_frozen_advisory_benchmark(
    population: Sequence[Mapping[str, Any]],
    *,
    sample_size: int,
    seed: int,
    candidate_model: str,
    report_path: str | Path,
    authorized_cost_ceiling: float,
    source_provenance: Mapping[str, Any],
    candidate_reasoning_effort: str | None = None,
    reviewer_models: Sequence[str] | None = None,
    disagreement_threshold: float | None = None,
    candidate_scorer: CandidateScorer | None = None,
    panel_scorer: ReviewerScorer | None = None,
    captured_provenance: benchmark.BenchmarkProvenance | None = None,
    created_at: str | None = None,
) -> FrozenBenchmarkReport:
    """Run and atomically publish one domain-balanced advisory benchmark.

    The population and selected rows must be JSON-compatible, and each selected
    row must resolve to a unique Standard Name.  The candidate and panel receive
    independent deep copies of the same ordered batch.  Exact row coverage is
    checked before publication so partial provider output cannot become a
    plausible-looking receipt.

    Costs come directly from scorer return values.  They are summarized after
    execution and never reserved, recorded, or settled through graph state.
    """
    _validate_run_arguments(
        population=population,
        sample_size=sample_size,
        seed=seed,
        candidate_model=candidate_model,
        authorized_cost_ceiling=authorized_cost_ceiling,
        source_provenance=source_provenance,
    )

    frozen_population = copy.deepcopy([dict(row) for row in population])
    population_hash = _ordered_rows_hash(frozen_population)
    sampled_rows = benchmark_roles._stratified_sample(
        copy.deepcopy(frozen_population),
        sample_size,
        seed,
        key="physics_domain",
    )
    frozen_rows = tuple(copy.deepcopy(sampled_rows))
    row_names = tuple(benchmark._resolve_name(row) for row in frozen_rows)
    _validate_row_names(row_names)
    ordered_input_hash = _ordered_rows_hash(frozen_rows)

    candidate_message_hashes: list[str] = []
    scorer = candidate_scorer or _score_with_production_context
    candidate_task = scorer(
        copy.deepcopy(list(frozen_rows)),
        candidate_model,
        "names",
        candidate_reasoning_effort,
        temperature=ADVISORY_CANDIDATE_TEMPERATURE,
        seed=seed,
        rendered_message_hashes=candidate_message_hashes,
    )
    panel_task = adjudicate_with_production_panel(
        copy.deepcopy(frozen_rows),
        reviewer_models=reviewer_models,
        disagreement_threshold=disagreement_threshold,
        target="names",
        reasoning_effort=candidate_reasoning_effort,
        scorer=panel_scorer,
    )
    (candidate_judgments, candidate_cost), panel = await asyncio.gather(
        candidate_task,
        panel_task,
    )

    candidate_cost = _validate_cost(candidate_cost, "candidate")
    _validate_rendered_message_hashes(candidate_message_hashes, len(frozen_rows))
    candidate_by_name = _index_candidate_judgments(candidate_judgments, row_names)
    result_rows = _assemble_result_rows(
        frozen_rows,
        row_names,
        candidate_by_name,
        panel,
    )
    ordered_result_hash = _ordered_rows_hash([asdict(row) for row in result_rows])
    costs = _summarize_costs(
        candidate_cost,
        panel.total_cost,
        authorized_cost_ceiling,
    )

    provenance = captured_provenance or benchmark.BenchmarkProvenance.capture()
    report = FrozenBenchmarkReport(
        candidate_model=candidate_model,
        candidate_temperature=ADVISORY_CANDIDATE_TEMPERATURE,
        candidate_seed=seed,
        candidate_rendered_message_hashes=tuple(candidate_message_hashes),
        seed=seed,
        sample_size=len(frozen_rows),
        population_size=len(frozen_population),
        population_hash=population_hash,
        ordered_input_hash=ordered_input_hash,
        ordered_result_hash=ordered_result_hash,
        input_rows=frozen_rows,
        result_rows=result_rows,
        reviewer_models=tuple(seat.reviewer_model for seat in panel.seats),
        disagreement_threshold=panel.disagreement_threshold,
        costs=costs,
        source_provenance=copy.deepcopy(dict(source_provenance)),
        dictionary_provenance={
            "data_dictionary_version": provenance.dd_version,
            "standard_names_dictionary_version": provenance.isn_version,
        },
        implementation_provenance={
            "codex_version": provenance.codex_version,
            "codex_commit": provenance.codex_commit,
        },
        created_at=created_at or datetime.now(tz=UTC).isoformat(),
    )
    report.publish_atomic(report_path)
    return report


async def _score_with_production_context(
    candidates: list[dict[str, Any]],
    model: str,
    target: str,
    reasoning_effort: str | None,
    *,
    temperature: float,
    seed: int,
    rendered_message_hashes: list[str],
) -> tuple[list[dict[str, Any]], float]:
    """Use the benchmark scorer that renders the production review context."""
    return await benchmark.score_with_reviewer(
        candidates,
        model,
        target=target,
        reasoning_effort=reasoning_effort,
        temperature=temperature,
        seed=seed,
        rendered_message_hashes=rendered_message_hashes,
    )


def _validate_run_arguments(
    *,
    population: Sequence[Mapping[str, Any]],
    sample_size: int,
    seed: int,
    candidate_model: str,
    authorized_cost_ceiling: float,
    source_provenance: Mapping[str, Any],
) -> None:
    if isinstance(sample_size, bool) or not isinstance(sample_size, int):
        raise TypeError("sample_size must be an integer")
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    if sample_size > len(population):
        raise ValueError(
            "sample_size cannot exceed the supplied frozen population size"
        )
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    if not isinstance(candidate_model, str) or not candidate_model.strip():
        raise ValueError("candidate_model must be a non-empty string")
    _validate_cost(authorized_cost_ceiling, "authorized ceiling")
    if not source_provenance:
        raise ValueError("source_provenance must identify the supplied population")
    _canonical_json(source_provenance)
    _canonical_json(population)


def _validate_row_names(row_names: tuple[str, ...]) -> None:
    if any(not name for name in row_names):
        raise ValueError("every sampled row must resolve to a non-empty standard name")
    if len(set(row_names)) != len(row_names):
        raise ValueError("sampled row standard names must be unique")


def _validate_rendered_message_hashes(hashes: list[str], row_count: int) -> None:
    expected_batches = math.ceil(row_count / 10)
    if len(hashes) != expected_batches:
        raise ValueError(
            "candidate did not checkpoint every rendered message batch; "
            f"expected={expected_batches}, actual={len(hashes)}"
        )
    if any(
        len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest)
        for digest in hashes
    ):
        raise ValueError("candidate returned an invalid rendered-message SHA-256")


def _index_candidate_judgments(
    judgments: Sequence[Mapping[str, Any]],
    expected_names: tuple[str, ...],
) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for judgment_like in judgments:
        judgment = copy.deepcopy(dict(judgment_like))
        name = judgment.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError("candidate returned a judgment without a name")
        if name in indexed:
            raise ValueError(f"candidate returned duplicate judgments for {name!r}")
        score = judgment.get("score")
        if (
            isinstance(score, bool)
            or not isinstance(score, int | float)
            or not math.isfinite(score)
            or not 0.0 <= score <= 1.0
        ):
            raise ValueError(
                f"candidate returned invalid score for {name!r}: {score!r}"
            )
        indexed[name] = judgment

    expected = set(expected_names)
    actual = set(indexed)
    if actual != expected:
        raise ValueError(
            "candidate did not cover the frozen batch exactly; "
            f"missing={sorted(expected - actual)}, "
            f"unexpected={sorted(actual - expected)}"
        )
    return indexed


def _assemble_result_rows(
    frozen_rows: tuple[dict[str, Any], ...],
    row_names: tuple[str, ...],
    candidate_by_name: Mapping[str, dict[str, Any]],
    panel: PanelAdjudication,
) -> tuple[FrozenResultRow, ...]:
    if len(panel.rows) != len(frozen_rows):
        raise ValueError("panel row count does not match the frozen batch")

    assembled: list[FrozenResultRow] = []
    for row_index, (input_row, name, panel_row) in enumerate(
        zip(frozen_rows, row_names, panel.rows, strict=True)
    ):
        if panel_row.row_index != row_index or panel_row.standard_name != name:
            raise ValueError("panel row order does not match the frozen batch")
        assembled.append(
            FrozenResultRow(
                row_index=row_index,
                standard_name=name,
                input_row=copy.deepcopy(input_row),
                candidate_judgment=copy.deepcopy(candidate_by_name[name]),
                panel_median_score=panel_row.median_score,
                panel_score_spread=panel_row.score_spread,
                panel_contested=panel_row.contested,
                panel_seat_judgments=tuple(
                    {
                        "seat_index": seat.seat_index,
                        "reviewer_model": seat.reviewer_model,
                        "judgment": copy.deepcopy(seat.judgment),
                    }
                    for seat in panel_row.seat_judgments
                ),
            )
        )
    return tuple(assembled)


def _summarize_costs(
    candidate_cost: float,
    judging_cost: float,
    authorized_ceiling: float,
) -> ArtifactCostSummary:
    judging_cost = _validate_cost(judging_cost, "judging")
    total_cost = candidate_cost + judging_cost
    remaining = authorized_ceiling - total_cost
    return ArtifactCostSummary(
        candidate_cost=candidate_cost,
        judging_cost=judging_cost,
        total_cost=total_cost,
        authorized_ceiling=authorized_ceiling,
        remaining_authority=max(0.0, remaining),
        within_authorized_ceiling=remaining >= 0.0,
    )


def _validate_cost(value: float, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(value)
        or value < 0.0
    ):
        raise ValueError(f"{label} cost must be a finite non-negative value")
    return float(value)


def _ordered_rows_hash(rows: Any) -> str:
    return hashlib.sha256(_canonical_json(rows).encode("utf-8")).hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
