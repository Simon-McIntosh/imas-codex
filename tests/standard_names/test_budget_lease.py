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
    _CHAT_FRAMING_TOKEN_ALLOWANCE,
    _MEAN_CACHE_WRITE_INPUT_FRACTION,
    _MEAN_CACHED_READ_INPUT_FRACTION,
    _REQUEST_BYTES_PER_TOKEN,
    _TYPICAL_COMPLETION_TOKENS,
    BudgetExposureUnknown,
    BudgetLease,
    BudgetManager,
    LLMCostEvent,
    bind_attempt_exposure,
    blended_input_rate,
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


def test_request_bytes_are_converted_to_tokens_before_pricing():
    """The input term prices tokens, not the request's byte length.

    Feeding bytes to a per-token rate over-prices the input several-fold, since
    prose and JSON both encode multiple bytes per token.  The exposure is
    reconstructed here from the priced rate to pin the conversion itself.
    """
    from imas_codex.settings import get_openrouter_pricing

    route = "openrouter/anthropic/claude-sonnet-5"
    price = get_openrouter_pricing(route)
    payload = "y" * 60_000
    messages = [{"role": "user", "content": payload}]

    exposure = model_provider_exposure(
        route,
        messages,
        response_model=_BoundedResponse,
        provider_attempts=1,
    )

    completion_term = _TYPICAL_COMPLETION_TOKENS * price["completion"] / 1e6
    input_rate = blended_input_rate(route, price["prompt"])
    priced_input_tokens = (exposure - completion_term) / (input_rate / 1e6)
    # The serialized request adds the JSON envelope and the response schema on
    # top of the payload, so the byte length is a lower bound on what is priced.
    lower = len(payload) / _REQUEST_BYTES_PER_TOKEN + _CHAT_FRAMING_TOKEN_ALLOWANCE
    assert lower <= priced_input_tokens < len(payload), (
        "the input term must price a token bound derived from the request bytes, "
        f"not the {len(payload)} bytes themselves"
    )


def test_admission_prices_the_typical_completion_not_the_published_maximum():
    """Admission must not reserve an output ceiling a typical response never reaches.

    Pricing the published maximum makes the reservation nearly request-size
    independent and several times the bill it settles, so a pool starves on
    reservation headroom while its cost limit stays unspent.  The tail is owned
    by the run cap and the reconciling charge path, not by admission.
    """
    from imas_codex.discovery.base.llm import get_model_limits

    route = "openrouter/anthropic/claude-sonnet-5"
    messages = [{"role": "user", "content": "y" * 38_000}]
    published = get_model_limits(route)["max_tokens"]
    assert published > _TYPICAL_COMPLETION_TOKENS

    admission = model_provider_exposure(
        route, messages, response_model=_BoundedResponse, provider_attempts=1
    )
    at_published = model_provider_exposure(
        route,
        messages,
        response_model=_BoundedResponse,
        provider_attempts=1,
        max_tokens=published,
    )
    at_typical = model_provider_exposure(
        route,
        messages,
        response_model=_BoundedResponse,
        provider_attempts=1,
        max_tokens=_TYPICAL_COMPLETION_TOKENS,
    )

    assert admission == pytest.approx(at_typical)
    assert admission < at_published


def test_input_term_blends_the_published_cache_rates():
    """Input is priced at the mix of rates a request actually pays.

    Static-first prompts mean most tokens are served from cache at a fraction of
    the base rate, while a cache write costs a premium on the routes that charge
    one.  Pricing every token as fresh overstates the bill by the cache
    discount; ignoring writes understates it.
    """
    from imas_codex.settings import get_openrouter_pricing

    route = "openrouter/anthropic/claude-sonnet-5"
    catalog = get_openrouter_pricing(route)
    assert catalog["cache_read"] and catalog["cache_write"], (
        "this route must declare both cache rates for the blend to be exercised"
    )

    prompt_rate = float(catalog["prompt"])
    blended = blended_input_rate(route, prompt_rate)
    fresh = 1.0 - _MEAN_CACHED_READ_INPUT_FRACTION - _MEAN_CACHE_WRITE_INPUT_FRACTION
    expected = (
        fresh * prompt_rate
        + _MEAN_CACHED_READ_INPUT_FRACTION * float(catalog["cache_read"])
        + _MEAN_CACHE_WRITE_INPUT_FRACTION * float(catalog["cache_write"])
    )

    assert blended == pytest.approx(expected)
    # Cache reads are the cheapest of the three rates and hold the largest
    # share, so the blend must land below the base rate.
    assert blended < prompt_rate


def test_uncatalogued_cache_rates_fall_back_to_the_base_prompt_rate():
    """A route publishing no cache rate is priced as if every token were fresh.

    Such providers bill cache tokens at the ordinary prompt rate, so collapsing
    the blend to that rate is the truthful estimate rather than a safety margin.
    """
    # A route absent from the project catalog declares no cache rates at all.
    assert blended_input_rate("openrouter/absent/route", 4.0) == pytest.approx(4.0)
    # A cataloged route does use them, so the fallback is not silently universal.
    assert blended_input_rate("openrouter/anthropic/claude-sonnet-5", 2.0) < 2.0


@pytest.mark.parametrize("bad_rate", [0.0, -1.0, float("nan"), float("inf")])
def test_blended_rate_fails_closed_without_a_base_rate(bad_rate: float):
    """No usable base prompt rate means no admission, not a silent zero."""
    with pytest.raises(BudgetExposureUnknown):
        blended_input_rate("openrouter/anthropic/claude-sonnet-5", bad_rate)


def test_admission_never_prices_above_a_routes_published_allowance(monkeypatch):
    """A route whose published output is smaller than the estimate uses its own."""
    from imas_codex.discovery.base import llm

    route = "openrouter/anthropic/claude-sonnet-5"
    messages = [{"role": "user", "content": "prompt"}]
    small = 2_048
    monkeypatch.setattr(
        llm, "get_model_limits", lambda model: {"max_tokens": small, "timeout": 60}
    )

    clamped = model_provider_exposure(
        route, messages, response_model=_BoundedResponse, provider_attempts=1
    )
    explicit = model_provider_exposure(
        route,
        messages,
        response_model=_BoundedResponse,
        provider_attempts=1,
        max_tokens=small,
    )

    assert clamped == pytest.approx(explicit)


class _RecordingLease:
    """Captures the amounts an attempt binding requires."""

    def __init__(self) -> None:
        self.required: list[float] = []

    def require_attempt(self, amount: float) -> None:
        self.required.append(amount)


def test_attempt_binding_funds_the_admission_estimate_until_the_allowance_grows():
    """Each attempt binds the size it will reach, escalation included.

    An unbumped retry (parse or transport failure) carries the opening
    allowance, so binding the published maximum on every attempt would restore
    the over-reservation admission just shed.  A length-exhausted attempt has
    proven it will use everything it is given, so that one binds its full
    escalated allowance.
    """
    route = "openrouter/anthropic/claude-sonnet-5"
    messages = [{"role": "user", "content": "y" * 20_000}]
    lease = _RecordingLease()
    bind = bind_attempt_exposure(
        lease,  # type: ignore[arg-type]
        route,
        messages,
        response_model=_BoundedResponse,
    )
    assert bind is not None

    bind(0, 32_000)  # opening allowance
    bind(1, 32_000)  # retry after a parse failure — same allowance
    bind(2, 64_000)  # retry after a length exhaustion — escalated

    admission = model_provider_exposure(
        route, messages, response_model=_BoundedResponse, provider_attempts=1
    )
    escalated = model_provider_exposure(
        route,
        messages,
        response_model=_BoundedResponse,
        provider_attempts=1,
        max_tokens=64_000,
    )
    assert lease.required[0] == pytest.approx(admission)
    assert lease.required[1] == pytest.approx(admission)
    assert lease.required[2] == pytest.approx(escalated)
    assert lease.required[2] > lease.required[0]


def test_binding_without_a_lease_is_a_noop():
    """An unbudgeted call has nothing to bind and must not build a hook."""
    assert (
        bind_attempt_exposure(
            None,
            "openrouter/anthropic/claude-sonnet-5",
            [{"role": "user", "content": "prompt"}],
            response_model=_BoundedResponse,
        )
        is None
    )


def test_oversized_request_is_refused_against_the_token_limit(monkeypatch):
    """The input-limit refusal compares tokens with tokens.

    Comparing the byte length against a token limit refuses a legal request
    whose token count fits comfortably inside the route's context window.
    """
    from imas_codex.discovery.base import llm

    route = "openrouter/anthropic/claude-sonnet-5"
    limit = 40_000
    monkeypatch.setattr(
        llm, "get_catalog_model_info", lambda model: {"max_input_tokens": limit}
    )

    # Bytes exceed the limit, tokens do not: this must price, not refuse.
    fits = model_provider_exposure(
        route,
        [{"role": "user", "content": "y" * 60_000}],
        response_model=_BoundedResponse,
        provider_attempts=1,
    )
    assert fits > 0

    with pytest.raises(BudgetExposureUnknown, match="exceeds the provider input"):
        model_provider_exposure(
            route,
            [{"role": "user", "content": "y" * 200_000}],
            response_model=_BoundedResponse,
            provider_attempts=1,
        )


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
