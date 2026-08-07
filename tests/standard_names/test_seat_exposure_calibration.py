"""Every configured Standard Names seat must be affordable to reserve.

The pipeline reserves a request's maximum billable exposure before it reaches
a provider, so a seat whose priced exposure approaches the run's whole cost
limit cannot fund a single batch — ``reserve()`` returns ``None`` for every
pool and the paid stages stop without spending anything.  That failure is
invisible in a unit test that stubs the pricing, so these tests price the real
seats from ``pyproject.toml`` through the real formula.

The ceiling below is a calibration gate, not a physical bound: it is set well
under a realistic per-run cost limit so a pricing change that inflates
reservations by orders of magnitude fails here rather than in a paid run.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from imas_codex.settings import (
    MODEL_SECTIONS,
    get_model,
    get_sn_review_docs_models,
    get_sn_review_names_models,
)
from imas_codex.standard_names.budget import model_provider_exposure

#: Upper bound for one priced provider attempt on any configured seat.
MAX_ATTEMPT_EXPOSURE_USD = 10.0

#: A rendered Standard Names request is a few tens of kilobytes of prompt.
#: Oversized on purpose so the gate measures a realistic worst case.
_RENDERED_REQUEST = [
    {"role": "system", "content": "x" * 20_000},
    {"role": "user", "content": "y" * 8_000},
]


class _SeatResponse(BaseModel):
    answer: str


def _configured_seats() -> list[tuple[str, str]]:
    """Return ``(seat, model)`` for every Standard Names seat in config."""
    seats = [
        (section, get_model(section))
        for section in sorted(MODEL_SECTIONS)
        if section.startswith("sn-")
    ]
    seats += [("sn-review.names", m) for m in get_sn_review_names_models()]
    seats += [("sn-review.docs", m) for m in get_sn_review_docs_models()]
    return seats


@pytest.mark.parametrize("seat,model", _configured_seats(), ids=lambda v: str(v))
def test_configured_seat_reserves_an_affordable_attempt(seat: str, model: str) -> None:
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
    """The gate is worthless if it parametrizes over nothing."""
    seats = _configured_seats()
    assert len(seats) >= 5, f"expected the configured seat set, got {seats}"


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
