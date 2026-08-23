"""Network-free evidence for the priced documentation holdout harness."""

from __future__ import annotations

import pytest

from imas_codex.standard_names import docs_holdout_eval
from imas_codex.standard_names.benchmark_reference import load_docs_holdout
from imas_codex.standard_names.docs_gates import DOCUMENTATION_GATE_NAMES
from imas_codex.standard_names.docs_holdout_eval import (
    HoldoutCostCeilingExceeded,
    evaluate_docs_holdout,
)


async def test_catalog_arm_reports_numeric_per_gate_baseline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _network_call(*args, **kwargs):
        raise AssertionError("catalog scoring must not call a model")

    monkeypatch.setattr(
        docs_holdout_eval,
        "_call_docs_model",
        _network_call,
    )

    report = await evaluate_docs_holdout("catalog")

    assert report.row_count == len(load_docs_holdout()) == 85
    assert tuple(row.gate for row in report.per_gate_table) == (
        DOCUMENTATION_GATE_NAMES
    )
    assert all(isinstance(row.arm_pass_rate, float) for row in report.per_gate_table)
    assert all(row.pass_rate_delta == 0.0 for row in report.per_gate_table)
    assert isinstance(report.overall_pass_rate, float)
    assert 0.0 <= report.overall_pass_rate <= 1.0
    assert report.overall_pass_rate == report.catalog_overall_pass_rate
    assert report.projected_call_count == 0
    assert report.actual_cost_usd == 0.0


async def test_priced_dry_run_projects_calls_and_cost_without_model_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_calls = 0

    async def _network_call(*args, **kwargs):
        nonlocal model_calls
        model_calls += 1
        raise AssertionError("dry run must not call a model")

    monkeypatch.setattr(
        docs_holdout_eval,
        "_call_docs_model",
        _network_call,
    )
    monkeypatch.setattr(
        docs_holdout_eval,
        "_production_enrich_items",
        _add_peer_context,
    )
    monkeypatch.setattr(
        docs_holdout_eval,
        "_project_request_cost",
        lambda model, messages: 0.25,
    )
    rows = load_docs_holdout()[:2]

    report = await evaluate_docs_holdout(
        "candidate",
        model="priced-model",
        dry_run=True,
        rows=rows,
    )

    assert report.projected_call_count == 2
    assert report.projected_cost_usd == pytest.approx(0.50)
    assert report.actual_cost_usd == 0.0
    assert model_calls == 0


async def test_projected_cost_above_ceiling_refuses_before_model_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_calls = 0

    async def _network_call(*args, **kwargs):
        nonlocal model_calls
        model_calls += 1
        raise AssertionError("an over-ceiling run must not call a model")

    monkeypatch.setattr(
        docs_holdout_eval,
        "_call_docs_model",
        _network_call,
    )
    monkeypatch.setattr(
        docs_holdout_eval,
        "_production_enrich_items",
        _add_peer_context,
    )
    monkeypatch.setattr(
        docs_holdout_eval,
        "_project_request_cost",
        lambda model, messages: 0.25,
    )

    with pytest.raises(HoldoutCostCeilingExceeded, match=r"above the \$0\.490000"):
        await evaluate_docs_holdout(
            "candidate",
            model="priced-model",
            cost_ceiling=0.49,
            rows=load_docs_holdout()[:2],
        )

    assert model_calls == 0


def _add_peer_context(items: list[dict]) -> list[dict]:
    for item in items:
        item["nearest_peers"] = [
            {
                "tag": "name:related_quantity",
                "unit": "1",
                "physics_domain": "equilibrium",
                "doc_short": "A related physical quantity.",
                "cocos_label": "",
            }
        ]
    return []


def test_rendered_evaluation_prompt_uses_production_relationship_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    row = load_docs_holdout()[0]

    def _add_related_context(items: list[dict]) -> list[dict]:
        items[0]["related_neighbours"] = [
            {
                "path": "equilibrium/time_slice/global_quantities/ip",
                "ids": "equilibrium",
                "relationship_type": "shared coordinate",
                "via": "time",
                "physics_domain": "equilibrium",
                "doc": "Plasma current related to this quantity.",
            }
        ]
        return []

    monkeypatch.setattr(
        docs_holdout_eval,
        "_production_enrich_items",
        _add_related_context,
    )
    context = docs_holdout_eval.build_compose_context()
    context["compose_scored_examples"] = []

    requests, counts = docs_holdout_eval._prepare_generation_requests(
        [docs_holdout_eval._candidate_for(row)],
        context=context,
    )

    assert counts[0].candidate_count == 1
    assert len(requests) == 1
    rendered_user_prompt = requests[0].messages[1]["content"]
    assert "## Related Physics Quantities" in rendered_user_prompt
    assert "equilibrium/time_slice/global_quantities/ip" in rendered_user_prompt


async def test_zero_context_candidates_are_reported_and_not_scored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_calls = 0

    def _no_context(items: list[dict]) -> list[dict]:
        return []

    async def _network_call(*args, **kwargs):
        nonlocal model_calls
        model_calls += 1
        raise AssertionError("a row without context candidates must not call a model")

    monkeypatch.setattr(
        docs_holdout_eval,
        "_production_enrich_items",
        _no_context,
    )
    monkeypatch.setattr(docs_holdout_eval, "_call_docs_model", _network_call)

    report = await evaluate_docs_holdout(
        "candidate",
        model="priced-model",
        rows=load_docs_holdout()[:1],
    )

    assert report.row_count == 1
    assert report.scored_row_count == 0
    assert report.zero_candidate_row_count == 1
    assert report.context_counts[0].candidate_count == 0
    assert report.row_scores == ()
    assert report.per_gate_table == ()
    assert model_calls == 0
