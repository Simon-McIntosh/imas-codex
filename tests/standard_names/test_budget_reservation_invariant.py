"""A paid review request must hold a reservation before it reaches a provider.

The review path prices the rendered request, reserves that maximum exposure,
and only then contacts the provider.  When the remaining pool cannot cover the
priced exposure the request raises ``BudgetExceeded`` instead of launching, so
spend can never precede the reservation that bounds it.

These tests drive ``_review_single_batch`` with a mocked provider and assert on
the dispatch itself: an unfundable request must leave the provider untouched,
and a fundable one must both launch and charge its completed cost.
"""

from __future__ import annotations

import asyncio
import logging
from unittest.mock import patch

import pytest

from imas_codex.standard_names.budget import BudgetExceeded, BudgetManager
from imas_codex.standard_names.models import (
    StandardNameQualityReview,
    StandardNameQualityReviewBatch,
    StandardNameQualityScore,
)

_NAMES = [{"id": "electron_temperature", "source_id": "core/te"}]


def _review_response() -> StandardNameQualityReviewBatch:
    return StandardNameQualityReviewBatch(
        reviews=[
            StandardNameQualityReview(
                source_id="core/te",
                standard_name="electron_temperature",
                scores=StandardNameQualityScore(
                    grammar=18,
                    semantic=18,
                    documentation=18,
                    convention=18,
                    completeness=18,
                    compliance=18,
                ),
                reasoning="Good.",
            )
        ]
    )


def _run_review(budget_manager, provider):
    from imas_codex.standard_names.review.pipeline import _review_single_batch

    with (
        patch(
            "imas_codex.llm.prompt_loader.render_prompt",
            return_value="mocked prompt",
        ),
        patch(
            "imas_codex.discovery.base.llm.acall_llm_structured",
            side_effect=provider,
        ),
    ):
        return asyncio.run(
            _review_single_batch(
                names=_NAMES,
                model="test-model",
                grammar_enums={},
                compose_ctx={},
                batch_context="test",
                neighborhood=[],
                audit_findings=[],
                wlog=logging.getLogger("test"),
                budget_manager=budget_manager,
                budget_phase="review_names",
                budget_batch_id="equilibrium",
            )
        )


class TestReviewReservationGate:
    """The reservation is a precondition of dispatch, not a running total."""

    def test_unfundable_request_never_reaches_the_provider(self) -> None:
        """A pool too small for the priced exposure must stop before launch."""
        dispatches: list[dict] = []

        async def _provider(**kwargs):
            dispatches.append(kwargs)
            raise AssertionError("provider was contacted without a reservation")

        # The priced exposure for a mocked route is $1.00; this pool cannot
        # cover it, so reserve() returns None and the request must fail closed.
        mgr = BudgetManager(total_budget=0.001)

        with pytest.raises(BudgetExceeded):
            _run_review(mgr, _provider)

        assert dispatches == []
        assert mgr.spent == 0.0
        assert mgr.remaining == pytest.approx(0.001)

    def test_fundable_request_launches_and_charges_its_cost(self) -> None:
        """A covered request dispatches once and settles its completed spend."""
        dispatches: list[dict] = []

        async def _provider(**kwargs):
            dispatches.append(kwargs)
            return _review_response(), 0.01, 100

        mgr = BudgetManager(total_budget=5.0)
        result = _run_review(mgr, _provider)

        assert len(dispatches) == 1
        assert len(result["_items"]) == 1
        assert mgr.spent == pytest.approx(0.01)
        # The unused remainder of the reservation returns to the pool, so the
        # pool is short only by what was actually charged.
        assert mgr.remaining == pytest.approx(5.0 - 0.01)
        assert mgr.check_invariant()


class TestBudgetManagerReserve:
    """Verify BudgetManager.reserve() returns None when pool is insufficient."""

    def test_reserve_returns_none_when_insufficient(self) -> None:
        """reserve() must return None when amount > remaining pool."""
        bm = BudgetManager(total_budget=3.0)
        assert bm.reserve(3.375) is None, (
            "reserve() must return None when amount > pool"
        )

    def test_reserve_succeeds_when_sufficient(self) -> None:
        """reserve() must return a BudgetLease when amount <= pool."""
        bm = BudgetManager(total_budget=3.0)
        assert bm.reserve(1.0) is not None, (
            "reserve() must succeed when pool is sufficient"
        )
