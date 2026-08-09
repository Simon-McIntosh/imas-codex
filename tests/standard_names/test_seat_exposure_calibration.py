"""Every paid Standard Names launch seat must be affordable to reserve.

The pipeline reserves a request's expected billable cost before it reaches a
provider.  That estimate must use a proven route price rather than the separate
provider policy ceiling, or a run can fail to fund a single batch despite
having ample budget for its actual cost.  These tests price the configured
compose seat and three default name-review seats through the real formula.

The ceiling below is a calibration gate, not a physical bound: it is set well
under a realistic per-run cost limit so a pricing change that inflates
reservations by orders of magnitude fails here rather than in a paid run.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from imas_codex.settings import (
    get_model,
    get_openrouter_pricing,
    get_sn_review_names_models,
)
from imas_codex.standard_names.budget import model_provider_exposure

#: Upper bound for one representative attempt on every launch seat.
MAX_ATTEMPT_EXPOSURE_USD = 0.50

#: A rendered Standard Names request is a few tens of kilobytes of prompt.
#: Oversized on purpose so the gate measures a realistic worst case.
_RENDERED_REQUEST = [
    {"role": "system", "content": "x" * 20_000},
    {"role": "user", "content": "y" * 8_000},
]


class _SeatResponse(BaseModel):
    answer: str


def _configured_seats() -> list[tuple[str, str]]:
    """Return the paid seats used to compose and review a name batch."""
    return [("sn-compose", get_model("sn-compose"))] + [
        (f"sn-review.names[{index}]", model)
        for index, model in enumerate(get_sn_review_names_models())
    ]


@pytest.mark.parametrize("seat,model", _configured_seats(), ids=lambda v: str(v))
def test_launch_seat_has_proven_catalog_pricing(seat: str, model: str) -> None:
    """Every paid launch route has finite rates and official provenance."""
    pricing = get_openrouter_pricing(model)
    assert pricing, f"{seat} ({model}) has no catalog entry"
    assert pricing["prompt"] > 0
    assert pricing["completion"] > 0
    assert pricing["request"] >= 0
    assert pricing["source"].startswith("https://openrouter.ai/api/v1/model/")
    assert pricing["verified_at"]


@pytest.mark.parametrize("seat,model", _configured_seats(), ids=lambda v: str(v))
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
        f"{seat} ({model}) reserves ${exposure:.2f} for a single provider "
        f"attempt, above the ${MAX_ATTEMPT_EXPOSURE_USD:.2f} calibration "
        "ceiling — a run would exhaust its cost limit on a handful of "
        "concurrent requests and stall without spending it"
    )


def test_seat_enumeration_is_not_silently_empty() -> None:
    """The gate covers one compose route and the complete three-seat quorum."""
    seats = _configured_seats()
    assert len(seats) == 4, f"expected compose plus three reviewers, got {seats}"
    assert len({model for _, model in seats}) == 4


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
