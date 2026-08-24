"""The advisory benchmark panel preserves evidence without graph mutation."""

from __future__ import annotations

import ast
import inspect
from typing import Any

import pytest

from imas_codex.standard_names import advisory_panel
from imas_codex.standard_names.advisory_panel import (
    adjudicate_with_production_panel,
)

SEATS = ("hosted/alpha", "hosted/bravo", "hosted/charlie")
FROZEN_BATCH = (
    {"standard_name": "electron_temperature", "source_id": "dd:a"},
    {"standard_name": "plasma_current", "source_id": "dd:b"},
)


def _frozen_scorer(
    scores: dict[str, dict[str, float]],
    calls: list[tuple[str, tuple[dict[str, Any], ...]]],
):
    async def score(
        candidates: list[dict[str, Any]],
        reviewer_model: str,
        target: str,
        reasoning_effort: str | None,
    ) -> tuple[list[dict[str, Any]], float]:
        assert target == "names"
        del reasoning_effort
        calls.append((reviewer_model, tuple(candidates)))
        return (
            [
                {
                    "name": candidate["standard_name"],
                    "score": scores[reviewer_model][candidate["standard_name"]],
                    "reasoning": f"{reviewer_model}:{candidate['source_id']}",
                }
                for candidate in candidates
            ],
            0.01 * (SEATS.index(reviewer_model) + 1),
        )

    return score


@pytest.mark.asyncio
async def test_panel_keeps_every_seat_judgment_unmerged() -> None:
    calls: list[tuple[str, tuple[dict[str, Any], ...]]] = []
    scores = {
        SEATS[0]: {"electron_temperature": 0.9, "plasma_current": 0.8},
        SEATS[1]: {"electron_temperature": 0.4, "plasma_current": 0.75},
        SEATS[2]: {"electron_temperature": 0.7, "plasma_current": 0.7},
    }

    result = await adjudicate_with_production_panel(
        FROZEN_BATCH,
        reviewer_models=SEATS,
        disagreement_threshold=0.2,
        reasoning_effort="medium",
        scorer=_frozen_scorer(scores, calls),
    )

    assert [model for model, _ in calls] == list(SEATS)
    assert all(batch == FROZEN_BATCH for _, batch in calls)
    first = result.rows[0]
    assert [item.reviewer_model for item in first.seat_judgments] == list(SEATS)
    assert [item.judgment["score"] for item in first.seat_judgments] == [
        0.9,
        0.4,
        0.7,
    ]
    assert [item.judgment["reasoning"] for item in first.seat_judgments] == [
        "hosted/alpha:dd:a",
        "hosted/bravo:dd:a",
        "hosted/charlie:dd:a",
    ]


@pytest.mark.asyncio
async def test_panel_computes_median_spread_and_contested_status() -> None:
    scores = {
        SEATS[0]: {"electron_temperature": 0.9, "plasma_current": 0.8},
        SEATS[1]: {"electron_temperature": 0.4, "plasma_current": 0.75},
        SEATS[2]: {"electron_temperature": 0.7, "plasma_current": 0.7},
    }

    result = await adjudicate_with_production_panel(
        FROZEN_BATCH,
        reviewer_models=SEATS,
        disagreement_threshold=0.2,
        scorer=_frozen_scorer(scores, []),
    )

    contested, agreed = result.rows
    assert contested.median_score == pytest.approx(0.7)
    assert contested.score_spread == pytest.approx(0.5)
    assert contested.contested is True
    assert agreed.median_score == pytest.approx(0.75)
    assert agreed.score_spread == pytest.approx(0.1)
    assert agreed.contested is False
    assert [seat.cost for seat in result.seats] == pytest.approx([0.01, 0.02, 0.03])
    assert result.total_cost == pytest.approx(0.06)


@pytest.mark.asyncio
async def test_panel_resolves_the_default_production_profile(monkeypatch) -> None:
    seen_profiles: list[str] = []

    def models(profile: str) -> list[str]:
        seen_profiles.append(profile)
        return list(SEATS)

    monkeypatch.setattr(advisory_panel.settings, "get_sn_review_profile_models", models)
    monkeypatch.setattr(
        advisory_panel.settings,
        "get_sn_review_profile_threshold",
        lambda profile: 0.2,
    )
    scores = {
        seat: {"electron_temperature": 0.8, "plasma_current": 0.8} for seat in SEATS
    }

    await adjudicate_with_production_panel(
        FROZEN_BATCH,
        scorer=_frozen_scorer(scores, []),
    )

    assert seen_profiles == ["default"]


@pytest.mark.asyncio
async def test_panel_fails_closed_on_partial_or_non_three_seat_evidence() -> None:
    async def partial_scorer(
        candidates: list[dict[str, Any]],
        reviewer_model: str,
        target: str,
        reasoning_effort: str | None,
    ) -> tuple[list[dict[str, Any]], float]:
        del reviewer_model, target, reasoning_effort
        return [{"name": candidates[0]["standard_name"], "score": 0.8}], 0.0

    with pytest.raises(ValueError, match="exactly three"):
        await adjudicate_with_production_panel(
            FROZEN_BATCH,
            reviewer_models=SEATS[:2],
            scorer=partial_scorer,
        )

    with pytest.raises(ValueError, match="cover the frozen batch exactly"):
        await adjudicate_with_production_panel(
            FROZEN_BATCH,
            reviewer_models=SEATS,
            scorer=partial_scorer,
        )


def test_adapter_has_no_graph_write_dependency() -> None:
    tree = ast.parse(inspect.getsource(advisory_panel))
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
                "imas_codex.standard_names.review.pipeline",
            )
        )
        for module in imported_modules
    )
    assert "run_sn_review_engine" not in inspect.getsource(advisory_panel)
    assert "write_reviews" not in inspect.getsource(advisory_panel)
