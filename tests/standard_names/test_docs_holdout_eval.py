"""Network-free evidence for the priced documentation holdout harness."""

from __future__ import annotations

import pytest

from imas_codex.standard_names import docs_holdout_eval
from imas_codex.standard_names.benchmark_reference import load_docs_holdout
from imas_codex.standard_names.docs_gates import (
    DOCUMENTATION_GATE_NAMES,
    DocumentationGateOutcome,
    DocumentationGateResult,
    DocumentationGateScore,
)
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
    assert all(
        row.arm_pass_count + row.arm_contradiction_count + row.arm_not_evaluable_count
        == 85
        for row in report.per_gate_table
    )
    assert all(
        row.catalog_pass_count
        + row.catalog_contradiction_count
        + row.catalog_not_evaluable_count
        == 85
        for row in report.per_gate_table
    )
    assert all(row.pass_rate_delta == 0.0 for row in report.per_gate_table)
    defining_equation = next(
        row for row in report.per_gate_table if row.gate == "defining_equation"
    )
    assert (
        defining_equation.arm_pass_count,
        defining_equation.arm_contradiction_count,
        defining_equation.arm_not_evaluable_count,
    ) == (47, 6, 32)
    assert (
        defining_equation.catalog_pass_count,
        defining_equation.catalog_contradiction_count,
        defining_equation.catalog_not_evaluable_count,
    ) == (47, 6, 32)
    assert isinstance(report.overall_pass_rate, float)
    assert 0.0 <= report.overall_pass_rate <= 1.0
    assert report.overall_pass_rate == report.catalog_overall_pass_rate
    assert report.projected_call_count == 0
    assert report.actual_cost_usd == 0.0
    assert report.row_scores[0].physics_context.dd_path == report.row_scores[0].dd_path


def _gate_results(
    defining_equation: DocumentationGateOutcome,
) -> dict[str, DocumentationGateResult]:
    results = {
        gate: DocumentationGateResult(
            outcome=DocumentationGateOutcome.PASS,
            reason="authoritative check passed",
        )
        for gate in DOCUMENTATION_GATE_NAMES
    }
    reason_by_outcome = {
        DocumentationGateOutcome.PASS: "equation dimensions match the declared unit",
        DocumentationGateOutcome.FAIL: (
            "equation dimensions contradict the declared unit"
        ),
        DocumentationGateOutcome.NOT_EVALUABLE: "declared unit is unavailable",
    }
    results["defining_equation"] = DocumentationGateResult(
        outcome=defining_equation,
        reason=reason_by_outcome[defining_equation],
    )
    return results


def test_not_evaluable_rows_are_excluded_from_aggregate_denominators() -> None:
    all_pass = _gate_results(DocumentationGateOutcome.PASS)
    missing_authority = _gate_results(DocumentationGateOutcome.NOT_EVALUABLE)
    contradiction = _gate_results(DocumentationGateOutcome.FAIL)
    catalog_scores = [
        DocumentationGateScore(gate_vector=dict(all_pass), word_count=50),
        DocumentationGateScore(gate_vector=dict(all_pass), word_count=50),
    ]

    not_evaluable_table, not_evaluable_overall, _, _ = (
        docs_holdout_eval._aggregate_scores(
            [
                DocumentationGateScore(gate_vector=dict(all_pass), word_count=50),
                DocumentationGateScore(
                    gate_vector=missing_authority,
                    word_count=50,
                ),
            ],
            catalog_scores,
        )
    )
    false_scored_table, false_scored_overall, _, _ = (
        docs_holdout_eval._aggregate_scores(
            [
                DocumentationGateScore(gate_vector=dict(all_pass), word_count=50),
                DocumentationGateScore(
                    gate_vector=contradiction,
                    word_count=50,
                ),
            ],
            catalog_scores,
        )
    )

    print(
        f"not_evaluable aggregate={not_evaluable_overall:.6f}; "
        f"scored_false aggregate={false_scored_overall:.6f}"
    )
    assert not_evaluable_overall == 1.0
    assert false_scored_overall == pytest.approx(11 / 12)
    assert not_evaluable_overall != false_scored_overall
    assert not_evaluable_table[0].arm_not_evaluable_count == 1
    assert not_evaluable_table[0].arm_contradiction_count == 0
    assert not_evaluable_table[0].arm_evaluable_count == 1
    assert false_scored_table[0].arm_not_evaluable_count == 0
    assert false_scored_table[0].arm_contradiction_count == 1
    assert false_scored_table[0].arm_evaluable_count == 2


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
