"""Every configured model seat names a current, cataloged and priced model.

Two properties are asserted here, both read from the configuration rather than
from a copy of it:

* each seat resolves to a model its provider's published catalog still lists,
  at the newest release of that model's own line; and
* every OpenRouter identifier a seat or panel names carries a cataloged price,
  so a seat cannot start a run it is unable to price.

The escalation seat gets its own test because it is the one seat whose model is
chosen by the refine rotation at chain cap rather than by the caller, and
because its reason for existing is that it stays a different vendor from the
refine and compose seats.
"""

from __future__ import annotations

import pytest

from imas_codex.settings import (
    _get_section,
    get_model,
    get_openrouter_pricing,
    get_sn_benchmark_candidate_models,
    get_sn_benchmark_compose_models,
    get_sn_benchmark_reviewer_model,
    get_sn_benchmark_reviewer_models,
    get_sn_review_docs_models,
    get_sn_review_names_models,
    get_sn_review_profile_models,
)
from imas_codex.standard_names.defaults import DEFAULT_ESCALATION_MODEL

# Every single-model seat under [tool.imas-codex.*].
SINGLE_SEATS = [
    "sn-docs",
    "sn-refine",
    "sn-escalation",
    "sn-parent-enrich",
    "sn-classifier",
    "sn-domain-tree",
    "sn-prose-adjudicator",
    "sn-release-notes",
    "dd-enrichment",
    "reasoning",
    "cluster-labels",
    "ids-mapping",
    "discovery-triage",
    "discovery-score",
    "discovery-describe",
    "discovery-vision",
    "discovery-relevance",
    "mapping-candidates",
    "sn-judge",
    "sn-compose",
]

REVIEW_PROFILES = ("default", "local-only", "quality-cost-balanced", "opus-only")

# The newest release of each line a seat or panel names, read from the public
# model list at https://openrouter.ai/api/v1/models on 2026-11-14. A line keeps
# its tier: Sonnet stays Sonnet, a flash stays a flash, and a preview is listed
# only where the line has no generally available successor.
LATEST = {
    "openrouter/anthropic/claude-sonnet-5.5",
    "openrouter/anthropic/claude-haiku-5.5",
    "openrouter/anthropic/claude-opus-5.5",
    "openrouter/openai/gpt-6.1-sol",
    "openrouter/openai/gpt-6.1-terra",
    "openrouter/openai/gpt-6-luna",
    "openrouter/x-ai/grok-4.7",
    "openrouter/google/gemini-3.8-flash",
    "openrouter/google/gemini-3.5-flash-lite",
    "openrouter/google/gemini-3.1-pro-preview",
    "openrouter/qwen/qwen3.7-plus",
    "openrouter/xiaomi/mimo-v2.6-pro",
    "openrouter/moonshotai/kimi-k3.1-pro",
    "openrouter/deepseek/deepseek-v4.1-flash",
    "openrouter/meta-llama/llama-4-maverick",
}


def _panel_models() -> list[str]:
    models = list(get_sn_review_names_models())
    models += list(get_sn_review_docs_models())
    for profile in REVIEW_PROFILES:
        models += list(get_sn_review_profile_models(profile))
    models += list(get_sn_benchmark_compose_models())
    models += list(get_sn_benchmark_candidate_models())
    models += list(get_sn_benchmark_reviewer_models())
    models.append(get_sn_benchmark_reviewer_model())
    return models


def _assert_priced(model: str, owner: str) -> None:
    if not model.startswith("openrouter/"):
        # A local or decisions seat is priced by its own lane: the router's
        # endpoint class, or the decisions response's own usage cost.
        return
    pricing = get_openrouter_pricing(model)
    assert pricing, f"{owner} names {model!r} with no cataloged price"
    for key in ("prompt", "completion", "cache_read"):
        value = pricing[key]
        assert isinstance(value, (int, float)), (
            f"{owner} prices {model!r} with {key}={value!r}"
        )
    assert pricing["prompt"] + pricing["completion"] > 0
    assert pricing["max_input_tokens"], f"{model!r} carries no input limit"
    assert pricing["source"], f"{model!r} carries no source record"
    assert pricing["verified_at"], f"{model!r} carries no verification date"


@pytest.mark.parametrize("seat", SINGLE_SEATS)
def test_every_seat_resolves_and_is_priced(seat: str) -> None:
    model = get_model(seat)
    assert model, f"seat {seat!r} resolved to an empty model"
    _assert_priced(model, f"seat {seat!r}")


@pytest.mark.parametrize("model", sorted(set(_panel_models())))
def test_panel_identifiers_are_priced(model: str) -> None:
    _assert_priced(model, "a review panel or benchmark slate")


@pytest.mark.parametrize("seat", SINGLE_SEATS)
def test_openrouter_seat_is_on_the_latest_release_of_its_line(seat: str) -> None:
    model = get_model(seat)
    if not model.startswith("openrouter/"):
        return
    assert model in LATEST, (
        f"seat {seat!r} names {model!r}, which is not the newest release of "
        "any line a seat or panel names"
    )


@pytest.mark.parametrize("model", sorted(set(_panel_models())))
def test_openrouter_panel_model_is_on_the_latest_release_of_its_line(
    model: str,
) -> None:
    if not model.startswith("openrouter/"):
        return
    assert model in LATEST, (
        f"a panel names {model!r}, which is not the newest release of any "
        "line a seat or panel names"
    )


def test_escalation_seat_drives_the_final_refine_rotation() -> None:
    """The chain-cap rotation reads the escalation seat, and the seat is set.

    Deleting the seat's ``model`` key makes this test fail at the first
    assertion: the rotation would fall back to a built-in default instead of
    reading a configured seat.
    """
    seat = _get_section("sn-escalation")
    assert "model" in seat, "the escalation seat names no model"
    assert get_model("sn-escalation") == seat["model"]
    assert DEFAULT_ESCALATION_MODEL == seat["model"]
    assert get_openrouter_pricing(seat["model"]), "escalation seat is unpriced"


def test_escalation_stays_vendor_diverse_from_refine_and_compose() -> None:
    def vendor(model: str) -> str:
        parts = model.split("/")
        return parts[1] if model.startswith(("openrouter/", "local/")) else parts[0]

    escalation = vendor(get_model("sn-escalation"))
    assert escalation != vendor(get_model("sn-refine"))
    assert escalation != vendor(get_model("sn-compose"))