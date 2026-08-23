"""Zero-call evidence for names-review campaign admission pricing."""

from __future__ import annotations

from typing import Any

import pytest

from imas_codex.standard_names import campaign_pricing
from imas_codex.standard_names.campaign_pricing import (
    CampaignCostCeilingExceeded,
    CampaignPricingPolicy,
    project_name_review_campaign,
)

_MODEL_COSTS = {
    "review-primary": 0.01,
    "review-secondary": 0.02,
    "review-escalator": 0.03,
    "refine": 0.04,
    "refine-escalator": 0.05,
}


def _policy(cohort_size: int) -> CampaignPricingPolicy:
    return CampaignPricingPolicy(
        reviewer_models=(
            "review-primary",
            "review-secondary",
            "review-escalator",
        ),
        refine_model="refine",
        refine_escalation_model="refine-escalator",
        refinement_rotations=3,
        max_refinement_names=cohort_size,
        fanout_enabled=True,
        fanout_baseline_cost_cap_usd=0.01,
        fanout_escalation_cost_cap_usd=0.02,
    )


def _prepared_requests(
    cohort: tuple[dict[str, Any], ...] | list[dict[str, Any]],
    *,
    escalation_critique_chars: int,
) -> list[campaign_pricing._PreparedNameRequest]:
    del escalation_critique_chars
    return [
        campaign_pricing._PreparedNameRequest(
            item=dict(item),
            base_messages=[{"role": "user", "content": item["id"]}],
            escalation_messages=[{"role": "user", "content": f"escalate {item['id']}"}],
        )
        for item in cohort
    ]


def _priced_request(
    model: str,
    messages: list[dict[str, Any]],
    *,
    response_model: type[Any],
    provider_attempts: int,
) -> float:
    del messages, response_model
    return _MODEL_COSTS[model] * provider_attempts


def _install_projection_fakes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(campaign_pricing, "_prepare_name_requests", _prepared_requests)
    monkeypatch.setattr(
        campaign_pricing,
        "_refinement_messages",
        lambda item, **kwargs: [{"role": "user", "content": item["id"]}],
    )
    monkeypatch.setattr(campaign_pricing, "_price_request", _priced_request)


def _cohort(size: int) -> list[dict[str, str]]:
    return [{"id": f"candidate_{index}"} for index in range(size)]


def test_projection_issues_exactly_zero_model_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_calls = 0

    async def _guarded_model_call(*args: Any, **kwargs: Any) -> None:
        nonlocal model_calls
        model_calls += 1
        raise AssertionError("campaign projection must not call a model")

    from imas_codex.discovery.base import llm

    monkeypatch.setattr(llm, "acall_llm_structured", _guarded_model_call)
    _install_projection_fakes(monkeypatch)

    projection = project_name_review_campaign(
        _cohort(1_179),
        policy=_policy(1_179),
    )

    assert projection.projected_call_count == 10_611
    assert projection.projected_cost_usd == pytest.approx(271.17)
    assert projection.mandatory_call_count == 2_358
    assert projection.mandatory_cost_usd == pytest.approx(35.37)
    assert projection.conditional_call_count == 8_253
    assert projection.conditional_cost_usd == pytest.approx(235.80)
    assert model_calls == 0


def test_refusal_one_cent_below_projection_issues_exactly_zero_model_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_calls = 0

    async def _guarded_model_call(*args: Any, **kwargs: Any) -> None:
        nonlocal model_calls
        model_calls += 1
        raise AssertionError("campaign refusal must not call a model")

    from imas_codex.discovery.base import llm

    monkeypatch.setattr(llm, "acall_llm_structured", _guarded_model_call)
    _install_projection_fakes(monkeypatch)

    with pytest.raises(
        CampaignCostCeilingExceeded,
        match=r"\$33\.120000 is above the \$33\.110000 ceiling",
    ):
        project_name_review_campaign(
            _cohort(144),
            cost_ceiling_usd=33.11,
            policy=_policy(144),
        )

    assert model_calls == 0


def test_review_request_renders_production_source_and_neighbour_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from imas_codex.standard_names import workers

    def _production_enrichment(items: list[dict[str, Any]]) -> None:
        items[0]["dd_source_docs"] = [
            {
                "id": "equilibrium/time_slice/global_quantities/ip",
                "ids": ["equilibrium/time_slice/global_quantities/ip"],
                "unit": "A",
                "description": "Plasma current from the exact DD source.",
                "documentation": "Exact source documentation.",
                "dd_version": "4.1.1",
                "snapshot_pinned": True,
            }
        ]
        items[0]["dd_parent_contexts"] = [
            {
                "path": "equilibrium/time_slice/global_quantities",
                "paths": ["equilibrium/time_slice/global_quantities"],
                "documentation": "Canonical parent structure.",
                "dd_version": "4.1.1",
                "snapshot_pinned": True,
            }
        ]
        items[0]["semantic_comparators"] = [
            {
                "path": "equilibrium/time_slice/global_quantities/current_non_inductive",
                "basis": "semantic_cluster",
            }
        ]

    neighbours = {
        "vector_neighbours": [
            {
                "id": "toroidal_plasma_current",
                "score": 0.91,
                "description": "A related reviewed quantity.",
            }
        ],
        "same_base_neighbours": [],
        "same_path_neighbours": [],
    }
    monkeypatch.setattr(workers, "_enrich_name_review_items", _production_enrichment)
    monkeypatch.setattr(
        campaign_pricing,
        "_load_review_context",
        lambda item: (neighbours, []),
    )

    request = campaign_pricing._prepare_name_requests(
        [
            {
                "id": "plasma_current",
                "description": "Electric current carried by the plasma.",
                "unit": "A",
                "physics_domain": "magnetics",
                "validation_issues": [],
            }
        ],
        escalation_critique_chars=32,
    )[0]

    rendered = request.base_messages[1]["content"]
    assert "equilibrium/time_slice/global_quantities/ip" in rendered
    assert "Exact source documentation" in rendered
    assert "Canonical parent structure" in rendered
    assert "toroidal_plasma_current" in rendered
    assert "current_non_inductive" in rendered
    assert len(request.escalation_messages[1]["content"]) > len(rendered)
    refinement_messages = campaign_pricing._refinement_messages(
        request.item,
        feedback_chars=32,
        context_chars=64,
    )
    assert "Current node review" in refinement_messages[1]["content"]
    assert "x" * 64 in refinement_messages[1]["content"]


def test_unpriced_route_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_projection_fakes(monkeypatch)
    monkeypatch.setattr(
        campaign_pricing,
        "_price_request",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            campaign_pricing.CampaignPricingUnknown("missing price")
        ),
    )

    with pytest.raises(campaign_pricing.CampaignPricingUnknown, match="missing price"):
        project_name_review_campaign(_cohort(1), policy=_policy(1))
