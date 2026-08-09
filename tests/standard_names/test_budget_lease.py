"""Tests for the lease-style BudgetManager API.

Covers: reserve, release, context manager, invariant,
concurrency, and edge cases.
"""

from __future__ import annotations

import asyncio
import threading

import pytest
from pydantic import BaseModel

from imas_codex.standard_names.budget import (
    BudgetExposureUnknown,
    BudgetLease,
    BudgetManager,
    LLMCostEvent,
    charge_billable_exception,
    model_provider_exposure,
)

# ── Test helper ───────────────────────────────────────────────────────


def _ce(lease: BudgetLease, amount: float, phase: str = "test") -> None:
    """Simulate an LLM spend via charge_event (replaces legacy charge())."""
    lease.charge_event(
        amount,
        LLMCostEvent(model="test-model", tokens_in=0, tokens_out=0, phase=phase),
    )


class _BoundedResponse(BaseModel):
    answer: str


def test_paid_exposure_covers_every_wrapper_attempt():
    """Pricing N attempts exceeds N times the first, as allowances escalate."""
    messages = [{"role": "user", "content": "short rendered prompt"}]
    one_attempt = model_provider_exposure(
        "openrouter/anthropic/claude-sonnet-4.6",
        messages,
        response_model=_BoundedResponse,
        provider_attempts=1,
    )
    all_attempts = model_provider_exposure(
        "openrouter/anthropic/claude-sonnet-4.6",
        messages,
        response_model=_BoundedResponse,
        provider_attempts=5,
    )
    assert one_attempt > 0
    assert all_attempts > one_attempt * 5


def test_paid_exposure_multiplies_the_complete_retry_budget_per_call():
    """The calls multiplier preserves the whole wrapper-attempt estimate."""
    messages = [{"role": "user", "content": "short rendered prompt"}]
    one_call = model_provider_exposure(
        "openrouter/anthropic/claude-sonnet-4.6",
        messages,
        response_model=_BoundedResponse,
        provider_attempts=3,
    )
    four_calls = model_provider_exposure(
        "openrouter/anthropic/claude-sonnet-4.6",
        messages,
        response_model=_BoundedResponse,
        provider_attempts=3,
        calls=4,
    )
    assert four_calls == pytest.approx(one_call * 4)


def test_paid_exposure_tracks_the_prompt_not_the_context_window():
    """A short request must not reserve what a context-window-sized one would.

    Pricing the route's whole input context would make every request cost the
    same regardless of size, over-reserving small ones by orders of magnitude.
    """
    route = "openrouter/anthropic/claude-sonnet-4.6"
    short = model_provider_exposure(
        route,
        [{"role": "user", "content": "short rendered prompt"}],
        response_model=_BoundedResponse,
        provider_attempts=1,
    )
    long = model_provider_exposure(
        route,
        [{"role": "user", "content": "y" * 500_000}],
        response_model=_BoundedResponse,
        provider_attempts=1,
    )
    assert short < long, "a smaller request must reserve less"


def test_catalog_lag_fails_closed_without_proven_expected_price():
    """An unpriced paid route cannot reach a provider through admission."""
    with pytest.raises(BudgetExposureUnknown):
        model_provider_exposure(
            "openrouter/future/model",
            [{"role": "user", "content": "prompt"}],
            response_model=_BoundedResponse,
            provider_attempts=5,
        )


def test_catalog_entry_without_official_provenance_fails_closed(monkeypatch):
    """Numeric rates alone are not price authority for paid admission."""
    from imas_codex import settings

    monkeypatch.setattr(
        settings,
        "get_openrouter_pricing",
        lambda model: {
            "prompt": 1.0,
            "completion": 1.0,
            "request": 0.0,
            "source": "",
            "verified_at": "",
            "overrides": [],
        },
    )
    with pytest.raises(BudgetExposureUnknown):
        model_provider_exposure(
            "openrouter/future/model",
            [{"role": "user", "content": "prompt"}],
            response_model=_BoundedResponse,
            provider_attempts=1,
        )


def test_expected_reservation_does_not_use_the_provider_policy_ceiling(monkeypatch):
    """Admission prices the catalog mean while the provider cap stays separate."""
    from imas_codex.discovery.base import llm

    def _policy_ceiling_must_not_price_admission(model: str) -> dict[str, float]:
        raise AssertionError(f"provider policy ceiling priced admission for {model}")

    monkeypatch.setattr(
        llm, "get_openrouter_max_price", _policy_ceiling_must_not_price_admission
    )

    exposure = model_provider_exposure(
        "openrouter/x-ai/grok-4.5",
        [{"role": "user", "content": "short rendered prompt"}],
        response_model=_BoundedResponse,
        provider_attempts=1,
    )

    assert 0 < exposure < 0.25


def test_non_text_input_rejects_before_reservation():
    """An input with unpriced media cannot reach a provider through the gate."""
    with pytest.raises(BudgetExposureUnknown):
        model_provider_exposure(
            "openrouter/anthropic/claude-sonnet-4.6",
            [{"role": "user", "content": [{"type": "image_url"}]}],
            response_model=_BoundedResponse,
            provider_attempts=5,
        )


@pytest.mark.parametrize("error_type", ["structured", "provider_budget"])
def test_billable_terminal_exception_is_charged_from_aggregate_telemetry(error_type):
    """Both terminal exception types preserve all completed retry spend."""
    from imas_codex.discovery.base.llm import (
        LLMStructuredCallError,
        ProviderBudgetExhausted,
    )

    error_class = (
        LLMStructuredCallError
        if error_type == "structured"
        else ProviderBudgetExhausted
    )
    exc = error_class(
        "terminal",
        cost=0.2,
        input_tokens=80,
        output_tokens=20,
        cache_read_tokens=10,
        cache_creation_tokens=5,
        response_count=3,
    )
    mgr = BudgetManager(total_budget=1.0)
    lease = mgr.reserve(0.5)
    assert lease is not None

    charged = charge_billable_exception(
        lease,
        exc,
        model="openrouter/anthropic/claude-sonnet-4.6",
        phase="review_name",
    )

    assert charged
    assert lease.charged == pytest.approx(0.2)
    assert mgr.spent == pytest.approx(0.2)


# =====================================================================
# Basic reserve / pool deduction
# =====================================================================


def test_reserve_deducts_from_pool():
    """Reserving deducts from the available pool."""
    mgr = BudgetManager(total_budget=1.0)
    lease = mgr.reserve(0.3)
    assert lease is not None
    assert abs(mgr.remaining - 0.7) < 1e-9
    assert mgr.check_invariant()


def test_reserve_returns_none_if_insufficient():
    """Two reserves that exceed total budget — second returns None."""
    mgr = BudgetManager(total_budget=1.0)

    lease1 = mgr.reserve(0.6)
    assert lease1 is not None

    lease2 = mgr.reserve(0.6)
    assert lease2 is None

    # Pool only reduced by first reservation
    assert abs(mgr.remaining - 0.4) < 1e-9
    assert mgr.check_invariant()


def test_reserve_exact_amount():
    """Reserving exactly the remaining pool succeeds."""
    mgr = BudgetManager(total_budget=0.5)
    lease = mgr.reserve(0.5)
    assert lease is not None
    assert mgr.remaining < 1e-9
    assert mgr.exhausted()
    assert mgr.check_invariant()


# =====================================================================
# Release
# =====================================================================


def test_release_unused_returns_to_pool():
    """Unused portion is returned to the pool on release."""
    mgr = BudgetManager(total_budget=1.0)
    lease = mgr.reserve(0.5)
    assert lease is not None

    _ce(lease, 0.2)
    unused = lease.release_unused()

    assert abs(unused - 0.3) < 1e-9
    assert abs(mgr.remaining - 0.8) < 1e-9  # 0.5 pool + 0.3 released
    assert abs(mgr.spent - 0.2) < 1e-9
    assert mgr.check_invariant()


def test_release_is_idempotent():
    """Calling release_unused twice doesn't double-count."""
    mgr = BudgetManager(total_budget=1.0)
    lease = mgr.reserve(0.5)
    assert lease is not None

    _ce(lease, 0.2)
    first = lease.release_unused()
    second = lease.release_unused()

    assert abs(first - 0.3) < 1e-9
    assert abs(second - 0.0) < 1e-9
    assert abs(mgr.remaining - 0.8) < 1e-9
    assert mgr.check_invariant()


def test_release_no_charge():
    """Releasing without any charges returns the full reservation."""
    mgr = BudgetManager(total_budget=1.0)
    lease = mgr.reserve(0.4)
    assert lease is not None

    unused = lease.release_unused()
    assert abs(unused - 0.4) < 1e-9
    assert abs(mgr.remaining - 1.0) < 1e-9
    assert abs(mgr.spent - 0.0) < 1e-9
    assert mgr.check_invariant()


# =====================================================================
# Context manager
# =====================================================================


def test_context_manager_auto_release():
    """Using `with lease:` auto-releases on exit."""
    mgr = BudgetManager(total_budget=1.0)
    lease = mgr.reserve(0.5)
    assert lease is not None

    with lease:
        _ce(lease, 0.1)

    # Remaining 0.4 auto-released
    assert abs(mgr.remaining - 0.9) < 1e-9
    assert abs(mgr.spent - 0.1) < 1e-9
    assert mgr.check_invariant()


def test_context_manager_on_exception():
    """Context manager releases even when an exception occurs."""
    mgr = BudgetManager(total_budget=1.0)
    lease = mgr.reserve(0.5)
    assert lease is not None

    with pytest.raises(RuntimeError):
        with lease:
            _ce(lease, 0.2)
            raise RuntimeError("boom")

    # 0.3 unused returned
    assert abs(mgr.remaining - 0.8) < 1e-9
    assert abs(mgr.spent - 0.2) < 1e-9
    assert mgr.check_invariant()


# =====================================================================
# Invariant
# =====================================================================


def test_invariant_pool_plus_reserved_plus_spent_equals_total():
    """Property-style: random operations maintain the invariant."""
    import random

    mgr = BudgetManager(total_budget=10.0)
    leases = []
    rng = random.Random(42)

    for _ in range(50):
        op = rng.choice(["reserve", "charge", "release"])

        if op == "reserve":
            amount = rng.uniform(0.01, 2.0)
            lease = mgr.reserve(amount)
            if lease is not None:
                leases.append(lease)

        elif op == "charge" and leases:
            lease = rng.choice(leases)
            if not lease._released:
                amount = rng.uniform(0.001, 0.1)
                if lease.charged + amount <= lease.reserved:
                    _ce(lease, amount)

        elif op == "release" and leases:
            lease = rng.choice(leases)
            lease.release_unused()

        assert mgr.check_invariant(), f"Invariant violated after op={op}"

    # Clean up all remaining leases
    for lease in leases:
        lease.release_unused()
    assert mgr.check_invariant()


# =====================================================================
# Concurrency
# =====================================================================


def test_concurrent_reserves():
    """Many threads racing for reserve() don't over-commit."""
    mgr = BudgetManager(total_budget=1.0)
    results: list[BudgetLease | None] = []
    lock = threading.Lock()

    def _reserve():
        lease = mgr.reserve(0.1)
        with lock:
            results.append(lease)

    threads = [threading.Thread(target=_reserve) for _ in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    successful = [r for r in results if r is not None]
    assert len(successful) == 10  # 1.0 / 0.1 = 10 max
    assert all(r is None for r in results if r not in successful)
    assert mgr.check_invariant()


def test_concurrent_reserves_async():
    """Async tasks racing for reserve() don't over-commit."""

    async def _run():
        mgr = BudgetManager(total_budget=1.0)

        async def _reserve():
            return mgr.reserve(0.1)

        tasks = [_reserve() for _ in range(20)]
        results = await asyncio.gather(*tasks)

        successful = [r for r in results if r is not None]
        assert len(successful) == 10
        assert mgr.check_invariant()

    asyncio.run(_run())


# =====================================================================
# Multiple leases
# =====================================================================


def test_multiple_leases_independent():
    """Multiple leases from the same manager are independent."""
    mgr = BudgetManager(total_budget=1.0)

    lease1 = mgr.reserve(0.3)
    lease2 = mgr.reserve(0.3)
    assert lease1 is not None
    assert lease2 is not None
    assert abs(mgr.remaining - 0.4) < 1e-9

    _ce(lease1, 0.1)
    _ce(lease2, 0.2)

    assert abs(mgr.spent - 0.3) < 1e-9
    assert mgr.check_invariant()

    lease1.release_unused()  # returns 0.2
    assert abs(mgr.remaining - 0.6) < 1e-9

    lease2.release_unused()  # returns 0.1
    assert abs(mgr.remaining - 0.7) < 1e-9
    assert abs(mgr.spent - 0.3) < 1e-9
    assert mgr.check_invariant()


# =====================================================================
# Summary / exhausted
# =====================================================================


def test_exhausted_when_pool_drained():
    """Manager reports exhausted when pool is zero."""
    mgr = BudgetManager(total_budget=0.5)
    assert not mgr.exhausted()

    lease = mgr.reserve(0.5)
    assert lease is not None
    assert mgr.exhausted()

    lease.release_unused()
    assert not mgr.exhausted()


def test_summary_reflects_state():
    """Summary dict reflects current state."""
    mgr = BudgetManager(total_budget=2.0)
    lease = mgr.reserve(0.5)
    assert lease is not None
    _ce(lease, 0.3)

    s = mgr.summary
    assert s["total_budget"] == 2.0
    assert abs(s["remaining"] - 1.5) < 1e-9
    assert abs(s["total_spent"] - 0.3) < 1e-9
    assert s["active_reservations"] == 1
    assert s["batch_count"] == 1


def test_repr():
    """BudgetLease has a useful repr."""
    mgr = BudgetManager(total_budget=1.0)
    lease = mgr.reserve(0.5)
    assert lease is not None
    r = repr(lease)
    assert "BudgetLease" in r
    assert "0.5000" in r


# =====================================================================
# _extend_reservation internal helper
# =====================================================================


def test_extend_reservation_returns_zero_when_pool_empty():
    """_extend_reservation returns 0 when pool is empty."""
    mgr = BudgetManager(total_budget=0.5)
    lease = mgr.reserve(0.5)
    assert lease is not None

    extended = mgr._extend_reservation(lease.lease_id, 0.2)
    assert extended == 0.0
    assert mgr.check_invariant()


# =====================================================================
# Backward compat: review/budget.py re-export
# =====================================================================


def test_review_budget_reexport():
    """ReviewBudgetManager is re-exported as an alias for BudgetManager."""
    from imas_codex.standard_names.review.budget import ReviewBudgetManager

    mgr = ReviewBudgetManager(total_budget=1.0)
    assert isinstance(mgr, BudgetManager)
    lease = mgr.reserve(0.5)
    assert lease is not None


# =====================================================================
# Regression: the summary key is total_spent, not total_actual
# =====================================================================


def test_summary_key_is_total_spent_not_total_actual():
    """BudgetManager.summary uses 'total_spent', not the incorrect 'total_actual'.

    A caller reading 'total_actual' raises KeyError; this pins the key name.
    """
    mgr = BudgetManager(total_budget=2.0)
    lease = mgr.reserve(0.5)
    assert lease is not None
    _ce(lease, 0.3)
    lease.release_unused()

    s = mgr.summary
    # The canonical key is 'total_spent'
    assert "total_spent" in s, "summary must contain 'total_spent'"
    assert "total_actual" not in s, "'total_actual' must NOT appear in summary"
    assert abs(s["total_spent"] - 0.3) < 1e-9
