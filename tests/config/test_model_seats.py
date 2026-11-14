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
    _MODEL_DEFAULTS,
    _SN_BENCHMARK_DEFAULTS,
    _get_section,
    get_model,
    get_sn_benchmark_candidate_models,
    get_sn_benchmark_compose_models,
    get_sn_benchmark_reviewer_model,
    get_sn_benchmark_reviewer_models,
    get_sn_review_docs_models,
    get_sn_review_names_models,
    get_sn_review_profile_models,
)
from imas_codex.standard_names.defaults import DEFAULT_REFINE_ROTATIONS
from imas_codex.standard_names.worker_tasks.worker_refine_names import (
    refine_model_for_attempt,
)
from tests.standard_names.test_seat_exposure_calibration import _assert_proven_pricing

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
    pricing = _assert_proven_pricing(owner, model)
    cache_read = pricing["cache_read"]
    assert isinstance(cache_read, (int, float)), (
        f"{owner} prices {model!r} with cache_read={cache_read!r}"
    )
    # A route may publish no cache-write rate, in which case the admission
    # estimate prices those tokens at the base prompt rate rather than at an
    # invented premium.
    cache_write = pricing["cache_write"]
    assert cache_write is None or isinstance(cache_write, (int, float)), (
        f"{owner} prices {model!r} with cache_write={cache_write!r}"
    )


@pytest.mark.parametrize("seat", SINGLE_SEATS)
def test_every_seat_resolves_and_is_priced(seat: str) -> None:
    model = get_model(seat)
    assert model, f"seat {seat!r} resolved to an empty model"
    _assert_priced(model, f"seat {seat!r}")


@pytest.mark.parametrize("model", sorted(set(_panel_models())))
def test_panel_identifiers_are_priced(model: str) -> None:
    _assert_priced(model, "a review panel or benchmark slate")


def _python_default_identifiers() -> list[tuple[str, str]]:
    """Every OpenRouter identifier a Python default names, with its owner.

    Read from the default mappings themselves rather than through the
    accessors: the accessors return the project configuration whenever it
    exists, so a default that names an unpriceable route is never exercised by
    them.
    """
    rows = [
        (f"_MODEL_DEFAULTS[{section!r}]", model)
        for section, model in _MODEL_DEFAULTS.items()
        if isinstance(model, str)
    ]
    for key, value in _SN_BENCHMARK_DEFAULTS.items():
        values = value if isinstance(value, list) else [value]
        rows += [(f"_SN_BENCHMARK_DEFAULTS[{key!r}]", model) for model in values]
    return [(owner, model) for owner, model in rows if model.startswith("openrouter/")]


@pytest.mark.parametrize(
    "owner,model", _python_default_identifiers(), ids=lambda value: str(value)
)
def test_python_default_identifiers_are_priced(owner: str, model: str) -> None:
    _assert_priced(model, owner)


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


def test_escalation_seat_drives_the_final_refine_rotation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The refine worker's last rotation runs the escalation seat's model.

    The seat is pointed at another priced model through its environment
    override, so the rotation has to resolve the seat at call time. A rotation
    that ignores the seat, reading either a built-in constant or the refine
    tier, resolves a model the seat does not name and fails here.
    """
    escalated_model = "openrouter/x-ai/grok-4.7"
    monkeypatch.setenv("IMAS_CODEX_SN_ESCALATION_MODEL", escalated_model)

    seat = _get_section("sn-escalation")
    cap = DEFAULT_REFINE_ROTATIONS
    final_model, final_escalates = refine_model_for_attempt(cap, cap)
    assert final_escalates is True
    assert final_model == escalated_model, (
        "the final rotation resolved a model the escalation seat does not name"
    )
    assert final_model == get_model("sn-escalation")
    assert seat["model"] != escalated_model, (
        "the override did not move the seat, so this run proves nothing"
    )

    earlier_model, earlier_escalates = refine_model_for_attempt(1, cap)
    assert earlier_escalates is False
    assert earlier_model == get_model("sn-refine")


def test_escalation_stays_vendor_diverse_from_refine_and_compose() -> None:
    def vendor(model: str) -> str:
        parts = model.split("/")
        return parts[1] if model.startswith(("openrouter/", "local/")) else parts[0]

    escalation = vendor(get_model("sn-escalation"))
    assert escalation != vendor(get_model("sn-refine"))
    assert escalation != vendor(get_model("sn-compose"))
