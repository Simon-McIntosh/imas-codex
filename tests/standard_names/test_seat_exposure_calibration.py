"""Every Standard Names launch seat must reserve its actual route exposure.

The pipeline reserves what a request is typically billed before it reaches a
provider.  That estimate must use a proven route price rather than the separate
provider policy ceiling, or a run can fail to fund a single batch despite
having ample budget for its actual cost.  These tests price the configured
local compose seat and three default paid name-review seats through the real
formula.

The bounds below are a calibration gate, not a physical bound: they sit either
side of what the seats price today, so both an inflated term (which starves
concurrency) and a collapsed one (which admits work no money is held for) fail
here rather than in a paid run.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from imas_codex.settings import (
    get_model,
    get_openrouter_pricing,
    get_sn_review_names_models,
)
from imas_codex.standard_names.budget import EPSILON, model_provider_exposure

#: Upper bound for one representative attempt on every launch seat.  Set just
#: above what the seats price today so an inflated term is caught here rather
#: than in a paid run — the reservation a pool holds per in-flight request is
#: what its cost limit must multiply by the replica count.
MAX_ATTEMPT_EXPOSURE_USD = 0.10

#: Upper bound for the stronger model used only after the refine chain is spent.
ESCALATION_MAX_ATTEMPT_EXPOSURE_USD = 0.50

#: Lower bound for one attempt on a paid seat.  Admission targets the typical
#: bill rather than a tail bound, so the gate has to catch collapse as well as
#: inflation: a term silently dropped to zero would admit unlimited concurrency
#: and only surface as overspend after the money was gone.
MIN_ATTEMPT_EXPOSURE_USD = 0.001

#: A rendered Standard Names request is a few tens of kilobytes of prompt.
#: Oversized on purpose so the gate measures a realistic worst case.
_RENDERED_REQUEST = [
    {"role": "system", "content": "x" * 20_000},
    {"role": "user", "content": "y" * 8_000},
]


class _SeatResponse(BaseModel):
    answer: str


def _configured_paid_seats() -> list[tuple[str, str]]:
    """Return the paid reviewer seats used after local composition."""
    return [
        (f"sn-review.names[{index}]", model)
        for index, model in enumerate(get_sn_review_names_models())
    ]


def _assert_proven_pricing(seat: str, model: str) -> dict[str, object]:
    """Require finite text rates with official per-route provenance."""
    pricing = get_openrouter_pricing(model)
    assert pricing, f"{seat} ({model}) has no catalog entry"
    assert pricing["prompt"] > 0
    assert pricing["completion"] > 0
    assert pricing["request"] >= 0
    assert pricing["source"].startswith("https://openrouter.ai/api/v1/model/")
    assert pricing["verified_at"]
    return pricing


@pytest.mark.parametrize("seat,model", _configured_paid_seats(), ids=lambda v: str(v))
def test_launch_seat_has_proven_catalog_pricing(seat: str, model: str) -> None:
    """Every paid launch route has finite rates and official provenance."""
    _assert_proven_pricing(seat, model)


def test_escalation_seat_has_proven_catalog_pricing() -> None:
    """The paid final refinement route has its published text rates."""
    pricing = _assert_proven_pricing("sn-escalation", get_model("sn-escalation"))
    assert pricing["prompt"] == 10.0
    assert pricing["completion"] == 50.0
    assert pricing["request"] == 0.0


@pytest.mark.parametrize("seat,model", _configured_paid_seats(), ids=lambda v: str(v))
def test_launch_seat_reserves_an_affordable_attempt(seat: str, model: str) -> None:
    """One attempt on a configured seat must price finite and affordable."""
    exposure = model_provider_exposure(
        model,
        _RENDERED_REQUEST,
        response_model=_SeatResponse,
        provider_attempts=1,
    )
    assert exposure > 0, f"{seat} ({model}) priced a non-positive exposure"
    assert exposure <= MAX_ATTEMPT_EXPOSURE_USD, (
        f"{seat} ({model}) reserves ${exposure:.4f} for a single provider "
        f"attempt, above the ${MAX_ATTEMPT_EXPOSURE_USD:.2f} calibration "
        "ceiling — a run would exhaust its cost limit on a handful of "
        "concurrent requests and stall without spending it"
    )
    assert exposure >= MIN_ATTEMPT_EXPOSURE_USD, (
        f"{seat} ({model}) reserves only ${exposure:.6f} for a paid attempt, "
        f"below the ${MIN_ATTEMPT_EXPOSURE_USD:.4f} floor — a collapsed input or "
        "completion term admits work no reservation is holding money for"
    )


def test_escalation_seat_reserves_an_affordable_attempt() -> None:
    """The stronger final refinement route still admits one expected attempt."""
    model = get_model("sn-escalation")
    exposure = model_provider_exposure(
        model,
        _RENDERED_REQUEST,
        response_model=_SeatResponse,
        provider_attempts=1,
    )
    assert MIN_ATTEMPT_EXPOSURE_USD <= exposure <= ESCALATION_MAX_ATTEMPT_EXPOSURE_USD


def test_local_compose_reserves_no_provider_exposure() -> None:
    """The configured local route must not reserve an OpenRouter attempt."""
    exposure = model_provider_exposure(
        get_model("sn-compose"),
        _RENDERED_REQUEST,
        response_model=_SeatResponse,
        provider_attempts=1,
    )
    assert exposure == EPSILON


def test_seat_enumeration_is_not_silently_empty() -> None:
    """The paid gate covers the complete three-seat reviewer quorum."""
    seats = _configured_paid_seats()
    assert len(seats) == 3, f"expected three paid reviewers, got {seats}"
    assert len({model for _, model in seats}) == 3


def _configured_paid_routes() -> list[tuple[str, str]]:
    """Every paid route a Standard Names run can dispatch to, with its seat."""
    from imas_codex.settings import get_sn_review_docs_models

    routes: dict[str, str] = {}
    for seat in (
        "sn-docs",
        "sn-refine",
        "sn-escalation",
        "sn-parent-enrich",
        "sn-classifier",
        "sn-prose-adjudicator",
    ):
        routes.setdefault(get_model(seat), seat)
    for index, model in enumerate(get_sn_review_names_models()):
        routes.setdefault(model, f"sn-review.names[{index}]")
    for index, model in enumerate(get_sn_review_docs_models()):
        routes.setdefault(model, f"sn-review.docs[{index}]")
    return [(seat, model) for model, seat in routes.items() if "/" in model]


@pytest.mark.parametrize(
    "seat,model",
    [(s, m) for s, m in _configured_paid_routes() if not m.startswith("hosted_vllm/")],
    ids=lambda v: str(v),
)
def test_every_paid_seat_is_priced_from_the_project_catalog(
    seat: str, model: str
) -> None:
    """A seated paid route must carry its own catalog entry, cache rates included.

    Without an entry the route can only be priced from LiteLLM's bundled data,
    which does not carry newer routes at all, and its cache tokens collapse to
    the base prompt rate — the input term then overstates the bill by the whole
    cache discount.  Seating a new model without pricing it must fail here.
    """
    pricing = get_openrouter_pricing(model)
    assert pricing, (
        f"{seat} ({model}) has no [tool.imas-codex.llm.openrouter-pricing] entry"
    )
    assert pricing["prompt"] and pricing["completion"], f"{seat} lacks token rates"
    assert pricing["source"].startswith("https://openrouter.ai/api/v1/model/")
    assert pricing["verified_at"]
    assert pricing["cache_read"] is not None, (
        f"{seat} ({model}) declares no cache-read rate, so the blended input "
        "term prices its cached prompt tokens as if they were fresh"
    )


def test_exposure_scales_with_the_rendered_request_not_the_context_window() -> None:
    """A short prompt must reserve far less than a long one on one route.

    Pricing the route's whole context window instead would make these equal
    and over-reserve every small request by the ratio of window to prompt.
    """
    model = get_sn_review_names_models()[0]
    short = model_provider_exposure(
        model,
        [{"role": "user", "content": "short"}],
        response_model=_SeatResponse,
        provider_attempts=1,
    )
    long = model_provider_exposure(
        model,
        [{"role": "user", "content": "y" * 400_000}],
        response_model=_SeatResponse,
        provider_attempts=1,
    )
    assert long > short, "exposure must track the rendered request size"
