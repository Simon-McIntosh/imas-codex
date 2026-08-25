"""The frozen advisory runner publishes reproducible read-only evidence."""

from __future__ import annotations

import ast
import copy
import inspect
import json
from pathlib import Path
from typing import Any

import pytest

from imas_codex.standard_names import advisory_benchmark
from imas_codex.standard_names.advisory_benchmark import (
    run_frozen_advisory_benchmark,
)
from imas_codex.standard_names.benchmark import BenchmarkProvenance

REVIEWERS = ("hosted/alpha", "hosted/bravo", "hosted/charlie")
POPULATION = (
    {
        "standard_name": "electron_temperature",
        "physics_domain": "transport",
        "source_id": "dd:core_profiles/electrons/temperature",
        "description": "Electron temperature.",
    },
    {
        "standard_name": "plasma_current",
        "physics_domain": "magnetics",
        "source_id": "dd:magnetics/ip/data",
        "description": "Plasma current.",
    },
    {
        "standard_name": "poloidal_magnetic_flux",
        "physics_domain": "equilibrium",
        "source_id": "dd:equilibrium/profiles_1d/psi",
        "description": "Poloidal magnetic flux.",
    },
    {
        "standard_name": "ion_temperature",
        "physics_domain": "transport",
        "source_id": "dd:core_profiles/ions/temperature",
        "description": "Ion temperature.",
    },
)
PROVENANCE = BenchmarkProvenance(
    codex_version="0.9.0",
    codex_commit="abc123",
    isn_version="0.8.0",
    dd_version="4.1.1",
)


def _scorer(
    calls: list[tuple[str, tuple[dict[str, Any], ...]]],
    *,
    candidate_cost: float = 0.04,
):
    async def score(
        candidates: list[dict[str, Any]],
        reviewer_model: str,
        target: str,
        reasoning_effort: str | None,
        *,
        temperature: float | None = None,
        seed: int | None = None,
        rendered_message_hashes: list[str] | None = None,
    ) -> tuple[list[dict[str, Any]], float]:
        assert target == "names"
        assert reasoning_effort == "medium"
        if reviewer_model == "local/candidate":
            assert temperature == 0.0
            assert seed is not None
            assert rendered_message_hashes is not None
            rendered_messages = [
                {"role": "system", "content": "frozen review fixture"},
                {
                    "role": "user",
                    "content": json.dumps(candidates, sort_keys=True),
                },
            ]
            rendered_message_hashes.append(
                advisory_benchmark.benchmark._rendered_messages_hash(rendered_messages)
            )
        else:
            assert temperature is None
            assert seed is None
            assert rendered_message_hashes is None
        calls.append((reviewer_model, tuple(copy.deepcopy(candidates))))
        offset = (
            0.0
            if reviewer_model == "local/candidate"
            else 0.01 * (REVIEWERS.index(reviewer_model) + 1)
        )
        return (
            [
                {
                    "name": row["standard_name"],
                    "score": 0.7 + offset,
                    "reasoning": f"{reviewer_model}:{row['source_id']}",
                }
                for row in candidates
            ],
            candidate_cost if reviewer_model == "local/candidate" else offset,
        )

    return score


async def _run(
    path: Path,
    *,
    seed: int = 17,
    sample_size: int = 3,
    scorer=None,
    source_provenance: dict[str, Any] | None = None,
):
    calls: list[tuple[str, tuple[dict[str, Any], ...]]] = []
    selected_scorer = scorer or _scorer(calls)
    selected_panel_scorer = _scorer(calls) if scorer is not None else selected_scorer
    report = await run_frozen_advisory_benchmark(
        POPULATION,
        sample_size=sample_size,
        seed=seed,
        candidate_model="local/candidate",
        report_path=path,
        authorized_cost_ceiling=1.0,
        source_provenance=source_provenance
        or {"catalog": "accepted-name-projection", "revision": "frozen-a"},
        candidate_reasoning_effort="medium",
        reviewer_models=REVIEWERS,
        disagreement_threshold=0.02,
        candidate_scorer=selected_scorer,
        panel_scorer=selected_panel_scorer,
        captured_provenance=PROVENANCE,
        created_at="2026-08-25T00:00:00+00:00",
    )
    return report, calls


@pytest.mark.asyncio
async def test_fixed_seed_freezes_domain_balanced_order_and_distinct_hashes(
    tmp_path: Path,
) -> None:
    first, first_calls = await _run(tmp_path / "first.json")
    second, second_calls = await _run(tmp_path / "second.json")

    assert first.input_rows == second.input_rows
    assert first.result_rows == second.result_rows
    assert first.ordered_input_hash == second.ordered_input_hash
    assert first.ordered_result_hash == second.ordered_result_hash
    assert (
        first.candidate_rendered_message_hashes
        == second.candidate_rendered_message_hashes
    )
    assert first.ordered_input_hash != first.ordered_result_hash
    assert {row["physics_domain"] for row in first.input_rows} == {
        "equilibrium",
        "magnetics",
        "transport",
    }
    assert all(batch == first.input_rows for _, batch in first_calls)
    assert first_calls == second_calls


@pytest.mark.asyncio
async def test_candidate_request_pins_decoding_and_hashes_same_rendered_row(
    tmp_path: Path,
) -> None:
    first, _ = await _run(
        tmp_path / "candidate-first.json", seed=20260825, sample_size=1
    )
    second, _ = await _run(
        tmp_path / "candidate-second.json", seed=20260825, sample_size=1
    )

    assert first.candidate_temperature == 0.0
    assert first.candidate_seed == 20260825
    assert second.candidate_temperature == 0.0
    assert second.candidate_seed == 20260825
    assert len(first.candidate_rendered_message_hashes) == 1
    assert (
        first.candidate_rendered_message_hashes
        == second.candidate_rendered_message_hashes
    )
    assert json.loads((tmp_path / "candidate-first.json").read_text())[
        "candidate_rendered_message_hashes"
    ] == list(first.candidate_rendered_message_hashes)


@pytest.mark.asyncio
async def test_population_and_result_hashes_detect_independent_redefinition(
    tmp_path: Path,
) -> None:
    report, _ = await _run(tmp_path / "report.json")
    payload = report.to_dict()

    changed_input = copy.deepcopy(payload["input_rows"])
    changed_input[0]["source_id"] = "dd:changed/path"
    assert (
        advisory_benchmark._ordered_rows_hash(changed_input)
        != report.ordered_input_hash
    )

    changed_result = copy.deepcopy(payload["result_rows"])
    changed_result[0]["candidate_judgment"]["score"] = 0.1
    assert (
        advisory_benchmark._ordered_rows_hash(changed_result)
        != report.ordered_result_hash
    )
    assert report.population_hash != report.ordered_input_hash


@pytest.mark.asyncio
async def test_report_publishes_source_and_dictionary_provenance_atomically(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "nested" / "report.json"
    destination.parent.mkdir()
    destination.write_text("incomplete")

    report, _ = await _run(
        destination,
        source_provenance={
            "catalog": "accepted-name-projection",
            "query_hash": "source-hash",
        },
    )

    stored = json.loads(destination.read_text())
    assert stored == report.to_dict()
    assert stored["source_provenance"] == {
        "catalog": "accepted-name-projection",
        "query_hash": "source-hash",
    }
    assert stored["dictionary_provenance"] == {
        "data_dictionary_version": "4.1.1",
        "standard_names_dictionary_version": "0.8.0",
    }
    assert stored["implementation_provenance"]["codex_commit"] == "abc123"
    assert list(destination.parent.glob("*.tmp")) == []


@pytest.mark.asyncio
async def test_costs_are_aggregated_in_the_artifact_without_budget_settlement(
    tmp_path: Path,
) -> None:
    report, _ = await _run(tmp_path / "costs.json")

    assert report.costs.candidate_cost == pytest.approx(0.04)
    assert report.costs.judging_cost == pytest.approx(0.06)
    assert report.costs.total_cost == pytest.approx(0.10)
    assert report.costs.authorized_ceiling == pytest.approx(1.0)
    assert report.costs.remaining_authority == pytest.approx(0.90)
    assert report.costs.within_authorized_ceiling is True

    source = inspect.getsource(advisory_benchmark)
    assert "BudgetManager" not in source
    assert "record_llm_cost" not in source
    assert "settle_budget" not in source


@pytest.mark.asyncio
async def test_default_execution_composes_the_production_context_scorer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, tuple[dict[str, Any], ...]]] = []
    scorer = _scorer(calls)
    monkeypatch.setattr(advisory_benchmark.benchmark, "score_with_reviewer", scorer)

    report = await run_frozen_advisory_benchmark(
        POPULATION,
        sample_size=3,
        seed=17,
        candidate_model="local/candidate",
        report_path=tmp_path / "default.json",
        authorized_cost_ceiling=1.0,
        source_provenance={"catalog": "accepted-name-projection"},
        candidate_reasoning_effort="medium",
        reviewer_models=REVIEWERS,
        disagreement_threshold=0.02,
        captured_provenance=PROVENANCE,
        created_at="2026-08-25T00:00:00+00:00",
    )

    assert [model for model, _ in calls] == ["local/candidate", *REVIEWERS]
    assert report.reviewer_models == REVIEWERS
    assert all(len(row.panel_seat_judgments) == 3 for row in report.result_rows)


@pytest.mark.asyncio
async def test_partial_candidate_evidence_fails_before_publication(
    tmp_path: Path,
) -> None:
    async def partial(
        candidates: list[dict[str, Any]],
        reviewer_model: str,
        target: str,
        reasoning_effort: str | None,
        *,
        temperature: float,
        seed: int,
        rendered_message_hashes: list[str],
    ) -> tuple[list[dict[str, Any]], float]:
        del reviewer_model, target, reasoning_effort, temperature, seed
        rendered_message_hashes.append("0" * 64)
        return [{"name": candidates[0]["standard_name"], "score": 0.8}], 0.0

    destination = tmp_path / "partial.json"
    with pytest.raises(ValueError, match="cover the frozen batch exactly"):
        await _run(destination, scorer=partial)

    assert not destination.exists()


def test_runner_has_no_graph_write_or_persistence_dependency() -> None:
    tree = ast.parse(inspect.getsource(advisory_benchmark))
    imported_modules = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    imported_modules.update(
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    )

    assert not any(
        module.startswith(
            (
                "imas_codex.graph",
                "imas_codex.standard_names.graph_ops",
                "imas_codex.standard_names.budget",
                "imas_codex.standard_names.review.pipeline",
            )
        )
        for module in imported_modules
    )
    source = inspect.getsource(advisory_benchmark)
    assert "GraphClient(" not in source
    assert "write_reviews" not in source
    assert "add_to_graph" not in source
