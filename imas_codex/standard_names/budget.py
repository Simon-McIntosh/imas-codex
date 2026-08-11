"""Lease-style budget manager for LLM batch pipelines.

Provides a shared ``BudgetManager`` that tracks reserved, spent, and
available budget.  Callers acquire a ``BudgetLease`` via ``reserve()``,
charge actual costs against it via ``charge_event()``, and release
unused headroom on completion.

**Three pricing stages, three different jobs**

A paid provider call passes three gates, and each one prices a different
quantity.  Conflating them is what makes a cost limit unspendable:

- ``reserve()`` **admits** work against the *typical* bill.
  ``model_provider_exposure`` converts the rendered request into a token bound,
  prices its input at the blend of fresh, cache-read and cache-write rates a
  request actually pays, and prices one completion at the size a typical
  response reaches rather than the route's published output maximum.  Pricing
  the maximum, or charging every input token at the base rate, makes each
  reservation several times the bill it settles, so a pool starves on
  reservation headroom while most of its cost limit sits unspent.  Admission
  does not attempt to bound the tail: the two gates below do that, and an
  estimate that covered the tail would reserve concurrency the pool cannot use.
- ``BudgetLease.require_attempt`` **binds** each real provider attempt, drawing
  any shortfall from the pool and refusing before the request leaves the
  process when the pool cannot fund it.  A retry whose output allowance
  escalated after a length exhaustion is re-priced at that larger allowance,
  so it is funded at the size it will actually reach.
- ``charge_event`` **settles** the actual bill.  Because admission is an
  estimate, a bill above the lease remainder tops the lease up from the pool;
  whatever the pool cannot fund is still recorded and reported as overspend.
  Spend the provider has already billed is never dropped from the ledger —
  dropping it understates the total and delays the hard stop that ends the run.

Invariant: ``pool + sum(active_reserved) + spent == total + overspend``

Thread-safe: ``threading.Lock`` protects all mutations.  The lock
critical sections are pure arithmetic (no I/O), so blocking is
negligible even in async contexts.

**Graph-backed cost tracking:**

``BudgetManager`` delegates spend recording to an async write queue
that persists ``LLMCost`` rows in Neo4j via ``record_llm_cost``.
In-memory ``_spent`` / ``_phase_spent`` are maintained as a local
cache for low-latency lease decisions.  The graph is the source of
truth; the in-memory counters are a write-ahead shadow.

The ``LLMCostEvent`` dataclass carries per-call metadata (model,
tokens, sn_ids, phase, etc.).  ``BudgetLease.charge_event()`` is the
single typed entry point that enqueues a graph write and updates the
local cache.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

logger = logging.getLogger(__name__)

EPSILON = 1e-9

# Writer retry parameters (matches retry_on_deadlock defaults)
_WRITER_MAX_RETRIES = 5
_WRITER_BASE_DELAY = 0.1
_WRITER_MAX_DELAY = 2.0

# Per-attempt timeout for record_llm_cost (run in a thread).
# Chosen to sit just above the worst-case retry_on_deadlock window
# (5 attempts × ~3s = 15s) so we don't cut into natural retries.
_WRITER_CALL_TIMEOUT = 20.0

# Heartbeat interval: how long the writer waits on an empty queue
# before emitting a health log line.
_WRITER_HEARTBEAT_SEC = 60.0

# Hard input-token ceiling used when a newly configured OpenRouter model is not
# yet present in LiteLLM's bundled catalog, so no published context limit is
# available to bound the request against.
_UNCATALOGED_INPUT_TOKEN_CEILING = 2_000_000

# Token allowance added to the request's own token bound to cover provider
# chat-template framing (role markers, turn delimiters, control tokens), which
# the serialized request does not contain.
_CHAT_FRAMING_TOKEN_ALLOWANCE = 4_096

# Bytes per token used to convert a serialized request's UTF-8 length into a
# token bound.  A byte length is not a token count: prose and JSON both encode
# several bytes per BPE token, so feeding bytes to a per-token rate over-prices
# the input term roughly fourfold and can refuse a legal request against the
# route's published input limit.
#
# The divisor sits below the smallest ratio measured over the rendered compose
# and review prompts and the payloads they carry, so dividing by it keeps the
# estimate on the high side of the real token count under every tokenizer
# tried.  A block of nothing but long underscored standard names is the floor —
# identifiers split into far more pieces than prose or JSON — and it lands near
# 2.75, which is why a plausible-looking 3.5 or 4 would under-price the very
# payloads this pipeline sends.  Raising it above the measured floor
# under-prices a dense request; lowering it far below reintroduces the
# fourfold over-reservation that pricing raw bytes caused.
_REQUEST_BYTES_PER_TOKEN = 2.5

# Completion size priced for one admission when the caller pins no explicit
# ``max_tokens``, clamped to the route's published allowance.  This is the
# TYPICAL completion, not a tail bound: measured across the paid Standard Names
# call history the mean structured completion is just under this figure and the
# median is smaller still.  The mean is what predicts aggregate spend, which is
# what a cost limit governs.
#
# Admission deliberately does not cover the tail.  The request still carries the
# route's published allowance so nothing is truncated; an overrun settles
# against the pool when the charge lands, a length-escalated retry is re-priced
# at its real allowance when the attempt binds, and the run-cap stop reads
# actual spend.  Pricing the tail here would only buy back concurrency the pool
# cannot then use.
_TYPICAL_COMPLETION_TOKENS = 3_000

# Mean share of a request's prompt tokens that a provider serves from its cache,
# and the mean share it bills as a cache write, measured over the paid Standard
# Names call history (cache reads and writes are both reported as subsets of
# ``prompt_tokens``).  Static-first prompt ordering means most of a request is a
# repeated system prefix, so a large fraction is served at the cache-read rate —
# an order of magnitude below the base prompt rate on every route that publishes
# one.  Pricing every input token at the base rate therefore overstates the bill
# by roughly the cache discount, while ignoring cache WRITES understates it on
# routes that charge a write premium.  Blending both against their published
# rates is what makes an admission estimate track the bill it settles.
_MEAN_CACHED_READ_INPUT_FRACTION = 0.46
_MEAN_CACHE_WRITE_INPUT_FRACTION = 0.18


# =====================================================================
# LLMCostEvent — typed metadata for a single LLM call
# =====================================================================


@dataclass(frozen=True, slots=True)
class LLMCostEvent:
    """Per-call metadata for an LLM invocation.

    Carried alongside the dollar ``cost`` through ``charge_event`` to
    the async writer queue, which persists it as an ``LLMCost`` node.
    """

    model: str
    tokens_in: int
    tokens_out: int
    tokens_cached_read: int = 0
    tokens_cached_write: int = 0
    sn_ids: tuple[str, ...] = ()
    batch_id: str | None = None
    """Generic correlation id linking related ``LLMCost`` rows.

    Two callers stamp this field today:

    - **Grammar retry** (``workers.py``): writes
      ``f"{group_key}-grammar-retry"`` so the original-vs-retry pair
      can be joined in analytics.
    - **Structured fan-out** (``fanout/dispatcher.py``):
      writes the ``fanout_run_id`` (uuid4) onto the proposer charge,
      and call-sites stamp the same id onto their synthesizer charge,
      enabling the ``Fanout`` ↔ ``LLMCost`` join.

    The field is intentionally unstructured — callers pick the
    encoding that suits their analytics query.
    """
    cycle: str | None = None  # e.g. "c0", "c1", "c2"
    phase: str = ""  # generate|regen|enrich|review_names|review_docs
    service: str = "standard-names"
    llm_at: datetime | None = None  # defaults to now(UTC) at write time


# =====================================================================
# ChargeResult — returned by charge_event
# =====================================================================


@dataclass(frozen=True, slots=True)
class ChargeResult:
    """Result of a ``charge_event`` call."""

    overspend: float = 0.0
    """Amount charged beyond reserved+pool (0.0 if within budget)."""
    hard_stop: bool = False
    """True when recorded spend has reached the run cap after this charge."""


# =====================================================================
# _PendingWrite — internal queue item
# =====================================================================


@dataclass(frozen=True, slots=True)
class _PendingWrite:
    """Enqueued graph write for the async writer."""

    cost: float
    event: LLMCostEvent
    overspend: float
    run_id: str
    llm_at: datetime = field(default_factory=lambda: datetime.now(UTC))


class BudgetExceeded(RuntimeError):
    """Raised when a lease charge exceeds the reserved amount."""


class BudgetExposureUnknown(RuntimeError):
    """Raised when a paid call has no finite pre-launch exposure bound."""


def provider_exposure(
    max_cost_per_attempt: float | None,
    *,
    provider_attempts: int,
    calls: int = 1,
) -> float:
    """Return the hard reservation needed for all possible paid attempts.

    Callers supply a provider-price ceiling for one attempt plus the exact
    number of wrapper attempts and calls that may execute under the lease.
    Missing, non-finite, or non-positive inputs fail closed before launch.
    """
    if (
        max_cost_per_attempt is None
        or not math.isfinite(max_cost_per_attempt)
        or max_cost_per_attempt <= 0
    ):
        raise BudgetExposureUnknown(
            "paid provider call has no finite positive per-attempt cost ceiling"
        )
    if provider_attempts < 1 or calls < 1:
        raise BudgetExposureUnknown(
            "paid provider call has no positive attempt and call bound"
        )
    exposure = max_cost_per_attempt * provider_attempts * calls
    if not math.isfinite(exposure) or exposure <= 0:
        raise BudgetExposureUnknown("paid provider exposure is not finite")
    return exposure


def blended_input_rate(model: str, prompt_rate: float) -> float:
    """Return the per-million input rate a typical request actually pays.

    A request's prompt tokens are billed in three parts — served from cache,
    written to cache, and charged fresh — at three different published rates.
    This blends them at the measured mean shares, so the input term tracks the
    bill instead of pricing every token as if it were fresh.

    A route that publishes no cache rate has its cache tokens priced at the base
    prompt rate, which is what such providers charge for them.  Only the base
    rate tier carries cache rates in the project catalog: the long-input
    override tiers restate prompt and completion, so a request past one of those
    thresholds blends an override prompt rate with base-tier cache rates. No
    request this pipeline sends approaches those thresholds.
    """
    from imas_codex.settings import get_openrouter_pricing

    if not math.isfinite(prompt_rate) or prompt_rate <= 0:
        raise BudgetExposureUnknown(
            f"blended input rate has no positive base prompt rate for {model}"
        )
    catalog = get_openrouter_pricing(model) or {}

    def _declared(field: str) -> float:
        value = catalog.get(field)
        if isinstance(value, bool) or not isinstance(value, int | float):
            return prompt_rate
        if not math.isfinite(value) or value < 0:
            return prompt_rate
        return float(value)

    fresh_fraction = (
        1.0 - _MEAN_CACHED_READ_INPUT_FRACTION - _MEAN_CACHE_WRITE_INPUT_FRACTION
    )
    rate = (
        fresh_fraction * prompt_rate
        + _MEAN_CACHED_READ_INPUT_FRACTION * _declared("cache_read")
        + _MEAN_CACHE_WRITE_INPUT_FRACTION * _declared("cache_write")
    )
    if not math.isfinite(rate) or rate <= 0:
        raise BudgetExposureUnknown(
            f"blended input rate is not finite and positive for {model}"
        )
    return rate


def model_provider_exposure(
    model: str,
    messages: list[dict[str, Any]],
    *,
    response_model: type[Any],
    provider_attempts: int,
    calls: int = 1,
    max_tokens: int | None = None,
) -> float:
    """Estimate the typical bill for a rendered structured request.

    The input term is sized from the rendered request itself — its serialized
    UTF-8 length converted to tokens plus a framing allowance, capped by the
    route's published context limit — and priced at the cache-blended rate a
    typical request pays rather than at the base prompt rate for every token.
    The output term prices *max_tokens* when the caller pins one, and otherwise
    the typical completion size rather than the route's published maximum.  Both
    choices point the same way: admission tracks the bill it will settle, and
    the run cap plus the reconciling charge path own the tail.  Cataloged route
    rates estimate admission cost; the separate provider ``max_price`` policy
    remains enforced at dispatch.  Every wrapper retry is represented and priced
    with the output allowance it can reach after a length exhaustion.  Routes
    without proven expected rates, and requests with unpriced non-text
    dimensions, fail closed.
    """
    from imas_codex.discovery.base.llm import (
        _LENGTH_RETRY_TOKEN_CAP,
        _LENGTH_RETRY_TOKEN_MULTIPLIER,
        _is_local_model,
        get_catalog_model_info,
        get_model_limits,
        get_openrouter_expected_price,
    )

    if _is_local_model(model):
        return EPSILON
    if provider_attempts < 1 or calls < 1:
        raise BudgetExposureUnknown("paid provider call has no bounded call count")

    for message in messages:
        if not isinstance(message.get("content"), str):
            raise BudgetExposureUnknown(
                "non-text provider input has no token-only bound"
            )

    schema = response_model.model_json_schema()
    serialized_request = json.dumps(
        {"messages": messages, "response_schema": schema},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    request_byte_bound = len(serialized_request.encode("utf-8"))
    if request_byte_bound <= 0:
        raise BudgetExposureUnknown("rendered provider request is empty")
    request_token_bound = math.ceil(request_byte_bound / _REQUEST_BYTES_PER_TOKEN)

    catalog_input_limit = get_catalog_model_info(model).get("max_input_tokens")
    input_limit = (
        catalog_input_limit
        if isinstance(catalog_input_limit, int) and catalog_input_limit > 0
        else _UNCATALOGED_INPUT_TOKEN_CEILING
    )
    if request_token_bound > input_limit:
        raise BudgetExposureUnknown("rendered request exceeds the provider input bound")
    input_bound = min(input_limit, request_token_bound + _CHAT_FRAMING_TOKEN_ALLOWANCE)

    try:
        expected_price = get_openrouter_expected_price(
            model,
            input_tokens=input_bound,
        )
    except Exception as exc:
        raise BudgetExposureUnknown(
            f"proven OpenRouter expected price unavailable for {model}"
        ) from exc

    published_output_limit = get_model_limits(model)["max_tokens"]
    if not isinstance(published_output_limit, int) or published_output_limit <= 0:
        raise BudgetExposureUnknown(
            "provider output allowance is not positively bounded"
        )
    output_limit = max_tokens or min(_TYPICAL_COMPLETION_TOKENS, published_output_limit)
    if not isinstance(output_limit, int) or output_limit <= 0:
        raise BudgetExposureUnknown(
            "provider output allowance is not positively bounded"
        )
    input_rate = blended_input_rate(model, expected_price["prompt"])

    total = 0.0
    for _ in range(provider_attempts):
        attempt_cost = (
            input_bound * input_rate / 1_000_000
            + output_limit * expected_price["completion"] / 1_000_000
            + expected_price["request"]
        )
        if not math.isfinite(attempt_cost) or attempt_cost <= 0:
            raise BudgetExposureUnknown(
                f"authoritative token prices are not finite and positive for {model}"
            )
        total += attempt_cost
        output_limit = min(
            output_limit * _LENGTH_RETRY_TOKEN_MULTIPLIER,
            _LENGTH_RETRY_TOKEN_CAP,
        )

    exposure = total * calls
    if not math.isfinite(exposure) or exposure <= 0:
        raise BudgetExposureUnknown("paid provider exposure is not finite")
    return exposure


def bind_attempt_exposure(
    lease: BudgetLease | None,
    model: str,
    messages: list[dict[str, Any]],
    *,
    response_model: type[Any],
) -> Callable[[int, int], None] | None:
    """Return a per-attempt hook binding each provider request to *lease*.

    Passed to ``acall_llm_structured(before_attempt=...)``.  Each attempt is
    funded at the completion size it is expected to reach: the admission
    estimate while the request still carries its opening allowance, and the
    full allowance once a length exhaustion has escalated it — an attempt that
    exhausted its budget mid-response has proven it will use whatever it is
    given.  The escalation is detected from the allowance itself rather than
    the attempt index, because a retry after a parse or transport error carries
    the opening allowance unchanged.  Returns ``None`` when there is no lease
    to bind.
    """
    if lease is None:
        return None

    opening_allowance: int | None = None

    def _bind(attempt: int, max_output_tokens: int) -> None:
        nonlocal opening_allowance
        if opening_allowance is None and max_output_tokens > 0:
            opening_allowance = max_output_tokens
        escalated = (
            opening_allowance is not None and max_output_tokens > opening_allowance
        )
        lease.require_attempt(
            model_provider_exposure(
                model,
                messages,
                response_model=response_model,
                provider_attempts=1,
                max_tokens=max_output_tokens if escalated else None,
            )
        )

    return _bind


def charge_billable_exception(
    lease: BudgetLease | None,
    exc: BaseException,
    *,
    model: str,
    sn_ids: tuple[str, ...] = (),
    batch_id: str | None = None,
    phase: str,
    service: str = "standard-names",
) -> bool:
    """Charge aggregate retry telemetry carried by a terminal exception once."""
    from imas_codex.discovery.base.llm import (
        LLMStructuredCallError,
        ProviderBudgetExhausted,
    )

    if lease is None or not isinstance(
        exc, LLMStructuredCallError | ProviderBudgetExhausted
    ):
        return False
    if exc.response_count <= 0:
        return False
    lease.charge_event(
        float(exc.cost),
        LLMCostEvent(
            model=model,
            tokens_in=int(exc.input_tokens),
            tokens_out=int(exc.output_tokens),
            tokens_cached_read=int(exc.cache_read_tokens),
            tokens_cached_write=int(exc.cache_creation_tokens),
            sn_ids=sn_ids,
            batch_id=batch_id,
            phase=phase,
            service=service,
        ),
    )
    return True


class BudgetLease:
    """A bounded spending grant from a :class:`BudgetManager`.

    Tracks charges against a reserved amount.  Use as a context manager
    for automatic release of unused budget::

        with mgr.reserve(0.5) as lease:
            result = lease.charge_event(0.3, LLMCostEvent(...))
        # Remaining 0.2 released to pool
    """

    __slots__ = ("_mgr", "_reserved", "_lease_id", "_charged", "_released", "_phase")

    def __init__(
        self,
        manager: BudgetManager,
        reserved: float,
        lease_id: str,
        phase: str = "",
    ) -> None:
        self._mgr = manager
        self._reserved = reserved
        self._lease_id = lease_id
        self._charged = 0.0
        self._released = False
        self._phase = phase

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def reserved(self) -> float:
        """Original reservation amount."""
        return self._reserved

    @property
    def charged(self) -> float:
        """Cumulative spend charged so far."""
        return self._charged

    @property
    def remaining(self) -> float:
        """Unspent portion of the reservation."""
        return self._reserved - self._charged

    @property
    def lease_id(self) -> str:
        return self._lease_id

    @property
    def phase(self) -> str:
        """Phase tag this lease is attributed to (empty string if untagged)."""
        return self._phase

    def require_attempt(self, amount: float) -> None:
        """Hold *amount* of unspent reservation before one provider attempt.

        Draws the shortfall from the manager pool so a lease seeded for the
        first attempt can cover a retry whose output allowance has escalated,
        and raises when the pool cannot fund it — denying the attempt before
        it reaches the provider rather than after.
        """
        if not math.isfinite(amount) or amount <= 0:
            raise BudgetExposureUnknown(
                "provider attempt exposure must be finite and positive"
            )
        if self._released:
            raise BudgetExceeded("cannot bind an attempt to a released lease")
        shortfall = amount - self.remaining
        if shortfall <= EPSILON:
            return
        self._reserved += self._mgr._extend_reservation(self._lease_id, shortfall)
        if self.remaining + EPSILON < amount:
            raise BudgetExceeded(
                f"provider attempt exposure ${amount:.6f} exceeds the fundable "
                f"remainder ${max(self.remaining, 0.0):.6f}"
            )

    def require_exposure(self, amount: float) -> None:
        """Fail before launch unless *amount* remains reserved on this lease."""
        if not math.isfinite(amount) or amount <= 0:
            raise BudgetExposureUnknown(
                "paid provider exposure must be finite and positive"
            )
        if self._released or self.remaining + EPSILON < amount:
            raise BudgetExceeded(
                f"provider exposure ${amount:.6f} exceeds lease remainder "
                f"${max(self.remaining, 0.0):.6f}"
            )

    # ------------------------------------------------------------------
    # Typed charge API
    # ------------------------------------------------------------------

    def charge_event(self, cost: float, event: LLMCostEvent) -> ChargeResult:
        """Settle one provider call: record spend + enqueue an ``LLMCost`` write.

        Reservations price an expected bill, so a charge above the remaining
        reservation draws its shortfall from the manager pool.  Whatever the
        pool cannot fund is still recorded and returned as ``overspend``,
        stamped on the ``LLMCost`` row: the provider has already billed it, and
        refusing the charge would leave real spend out of both the local
        counters and the graph while the run kept working against an
        understated total.
        """
        if cost < 0:
            raise ValueError("charge must be non-negative")
        if not math.isfinite(cost):
            raise ValueError("charge must be finite")
        overspend = self._mgr._record_spend(self._lease_id, cost)
        self._charged += cost
        # Enqueue async graph write
        self._mgr._enqueue_write(cost, event, overspend)
        return ChargeResult(overspend=overspend, hard_stop=self._mgr.hard_exhausted())

    def release_unused(self) -> float:
        """Return unspent portion to manager pool.  Idempotent."""
        if self._released:
            return 0.0
        unused = max(self._reserved - self._charged, 0.0)
        self._mgr._release(self._lease_id, unused)
        self._released = True
        return unused

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> BudgetLease:
        return self

    def __exit__(self, *exc: object) -> None:
        self.release_unused()

    def __repr__(self) -> str:
        return (
            f"BudgetLease(id={self._lease_id!r}, phase={self._phase!r}, "
            f"reserved={self._reserved:.4f}, "
            f"charged={self._charged:.4f}, released={self._released})"
        )


class BudgetManager:
    """Concurrent-safe budget manager with lease-based tracking.

    Usage::

        mgr = BudgetManager(total_budget=5.0, run_id="run-001")

        lease = mgr.reserve(0.50)
        if lease is None:
            return  # budget exhausted

        with lease:
            cost = await call_llm(...)
            result = lease.charge_event(cost, LLMCostEvent(...))
        # Unused portion auto-released

    Invariant: ``pool + sum(active_reserved) + spent == total + overspend``

    When ``run_id`` is set, ``charge_event`` enqueues async graph writes
    via :func:`record_llm_cost`.  Call :meth:`start` before first use
    and :meth:`drain_pending` at shutdown.
    """

    def __init__(
        self,
        total_budget: float,
        phase_caps: dict[str, float] | None = None,
        *,
        run_id: str | None = None,
    ) -> None:
        self._total = total_budget
        self._pool = total_budget
        self._reserved: dict[str, float] = {}  # lease_id → remaining reservation
        self._spent = 0.0
        # Spend that neither a lease reservation nor the pool could fund at
        # charge time.  Recorded rather than refused, so the ledger matches what
        # the provider billed; it is the arithmetic slack in the invariant.
        self._overspend = 0.0
        self._batch_count = 0
        self._lock = threading.Lock()
        # Per-phase hard caps.  Keys are phase names; values are the cap in
        # dollars.  Reservations that would push a phase's total committed
        # budget beyond cap × 1.5 are rejected.
        self._phase_caps: dict[str, float] = phase_caps or {}
        # Running total of amounts reserved for each tagged phase (cumulative;
        # not decremented on release so over-reservation is prevented even
        # after partial refunds).
        self._phase_committed: dict[str, float] = {}
        # Per-lease phase tag (for spend attribution).
        self._lease_phases: dict[str, str] = {}
        # Actual spend per phase tag (in-memory shadow — graph is SoT).
        self._phase_spent: dict[str, float] = {}

        # ── Graph-backed cost tracking ────────────────────────────────
        self.run_id: str | None = run_id
        self._write_queue: asyncio.Queue[_PendingWrite | None] = asyncio.Queue()
        self._pending_cost: float = 0.0  # in-flight cost not yet flushed
        self._pending_lock = threading.Lock()
        self._writer_task: asyncio.Task[None] | None = None
        self._write_failed: bool = False
        self._write_dropped: int = 0
        self._started: bool = False
        # Cached graph total (refreshed at most once per second)
        self._graph_total_cache: float = 0.0
        self._graph_total_ts: float = 0.0  # monotonic timestamp of last fetch
        self._graph_cache_ttl: float = 1.0  # seconds

        # ── Budget-saturation tracking ───────────────────────────────
        # Per-pool consecutive reserve-failure counter.  Incremented each
        # time ``reserve()`` returns ``None`` for a given phase, reset to 0
        # on success.  When ALL tracked pools exceed
        # ``SATURATION_THRESHOLD`` simultaneously, the budget is too small
        # to fund any batch.
        self._consecutive_reserve_failures: dict[str, int] = {}
        self.SATURATION_THRESHOLD: int = 10

    # ------------------------------------------------------------------
    # Async lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the background writer task.

        Called from harness setup.  Safe to call multiple times (idempotent).
        """
        if self._started:
            return
        self._started = True
        self._writer_task = asyncio.create_task(self._writer_loop())

    async def drain_pending(self, *, raise_on_failure: bool = False) -> bool:
        """Wait for the write queue to drain and the writer to finish.

        Returns ``True`` if everything was written, ``False`` if any write
        failed terminally.  Called from harness shutdown.

        When ``raise_on_failure`` is True, raises ``RuntimeError`` on
        terminal write failures instead of returning False.
        """
        # Send sentinel to tell writer loop to exit
        await self._write_queue.put(None)
        if self._writer_task is not None:
            try:
                await self._writer_task
            except Exception:  # noqa: BLE001
                logger.exception("Writer task raised during drain")
                self._write_failed = True

        # Surface dropped-write count so operators know telemetry is incomplete.
        if self._write_dropped > 0:
            logger.error(
                "drain_pending: %d LLMCost write(s) dropped after retry exhaustion",
                self._write_dropped,
            )

        if self._write_failed:
            logger.error(
                "drain_pending: _write_failed=True — one or more LLMCost "
                "writes failed terminally; cost_is_exact should be False"
            )

        success = not self._write_failed
        if not success and raise_on_failure:
            raise RuntimeError(
                "One or more LLMCost writes failed terminally; "
                "cost_is_exact should be set to False"
            )
        return success

    async def _writer_loop(self) -> None:
        """Pull events from the queue and call ``record_llm_cost`` for each.

        On transient errors, retries with exponential backoff.  On terminal
        failure (after retries exhausted), marks ``_write_failed = True`` and
        continues processing the queue (best-effort for remaining writes).

        A heartbeat fires every ``_WRITER_HEARTBEAT_SEC`` of idle time so
        operators can confirm the writer is alive.  INFO when there is
        pending work; DEBUG when idle and drained.
        """
        while True:
            try:
                item = await asyncio.wait_for(
                    self._write_queue.get(),
                    timeout=_WRITER_HEARTBEAT_SEC,
                )
            except TimeoutError:
                # Heartbeat — no items arrived within the window
                with self._pending_lock:
                    pc = self._pending_cost
                qs = self._write_queue.qsize()
                if pc > 0 or qs > 0:
                    logger.info(
                        "writer_loop heartbeat: pending=$%.4f qsize=%d",
                        pc,
                        qs,
                    )
                else:
                    logger.debug(
                        "writer_loop heartbeat: pending=$%.4f qsize=%d",
                        pc,
                        qs,
                    )
                continue

            if item is None:
                # Sentinel — drain complete
                self._write_queue.task_done()
                break
            try:
                await self._write_single(item)
            except Exception:  # noqa: BLE001
                logger.exception(
                    "Terminal write failure for LLMCost (run=%s, phase=%s, cost=%.6f)",
                    item.run_id,
                    item.event.phase,
                    item.cost,
                )
                self._write_failed = True
                self._write_dropped += 1
            finally:
                with self._pending_lock:
                    self._pending_cost -= item.cost
                self._write_queue.task_done()

    async def _write_single(self, item: _PendingWrite) -> None:
        """Write a single ``LLMCost`` to the graph with retry.

        Each attempt runs ``record_llm_cost`` in a thread via
        ``asyncio.to_thread`` with a per-attempt timeout so a wedged
        Neo4j connection cannot block the writer loop indefinitely.
        """
        from imas_codex.standard_names.graph_ops import record_llm_cost

        last_exc: Exception | None = None
        for attempt in range(_WRITER_MAX_RETRIES):
            try:
                await asyncio.wait_for(
                    asyncio.to_thread(
                        record_llm_cost,
                        run_id=item.run_id,
                        phase=item.event.phase,
                        cycle=item.event.cycle,
                        sn_ids=list(item.event.sn_ids) if item.event.sn_ids else None,
                        model=item.event.model,
                        cost=item.cost,
                        tokens_in=item.event.tokens_in,
                        tokens_out=item.event.tokens_out,
                        tokens_cached_read=item.event.tokens_cached_read,
                        tokens_cached_write=item.event.tokens_cached_write,
                        service=item.event.service,
                        batch_id=item.event.batch_id,
                        overspend=item.overspend,
                        llm_at=item.llm_at,
                    ),
                    timeout=_WRITER_CALL_TIMEOUT,
                )
                return  # success
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if attempt < _WRITER_MAX_RETRIES - 1:
                    delay = min(_WRITER_BASE_DELAY * (2**attempt), _WRITER_MAX_DELAY)
                    logger.warning(
                        "Writer retry %d/%d for run=%s: %s",
                        attempt + 1,
                        _WRITER_MAX_RETRIES,
                        item.run_id,
                        exc,
                    )
                    await asyncio.sleep(delay)
        # All retries exhausted — propagate to caller
        raise last_exc  # type: ignore[misc]

    # ------------------------------------------------------------------
    # Internal: enqueue a graph write
    # ------------------------------------------------------------------

    def _enqueue_write(
        self, cost: float, event: LLMCostEvent, overspend: float
    ) -> None:
        """Enqueue an ``LLMCost`` write (called from ``charge_event``).

        If no ``run_id`` is configured, the write is silently skipped
        (useful for tests without a graph).

        Detects a crashed/cancelled writer task and recreates it under
        ``_pending_lock`` to prevent TOCTOU double-recreation.
        """
        if self.run_id is None:
            return

        # Check writer health under lock (TOCTOU-safe).
        with self._pending_lock:
            self._pending_cost += cost
            if self._writer_task is not None and self._writer_task.done():
                try:
                    exc = self._writer_task.exception()
                except asyncio.CancelledError:
                    exc = "cancelled"
                logger.error("writer_task died unexpectedly (%s) — recreating", exc)
                self._writer_task = asyncio.create_task(self._writer_loop())

        pw = _PendingWrite(
            cost=cost,
            event=event,
            overspend=overspend,
            run_id=self.run_id,
            llm_at=event.llm_at or datetime.now(UTC),
        )
        try:
            qsize = self._write_queue.qsize()
            if qsize > 256:
                logger.warning(
                    "Write queue backpressure: qsize=%d — persist workers may be "
                    "falling behind LLM throughput",
                    qsize,
                )
            self._write_queue.put_nowait(pw)
        except asyncio.QueueFull:  # pragma: no cover — unbounded queue
            logger.error("Write queue full — dropping LLMCost event")
            with self._pending_lock:
                self._pending_cost -= cost
            self._write_failed = True

    # ------------------------------------------------------------------
    # Reserve
    # ------------------------------------------------------------------

    def reserve(self, amount: float, phase: str = "") -> BudgetLease | None:
        """Atomically reserve *amount* from the pool.

        Returns a :class:`BudgetLease` on success, ``None`` if the pool
        has insufficient funds or the named *phase* would exceed its hard
        cap (``phase_caps[phase] × 1.5``).

        Also tracks consecutive reserve failures per *phase* for the
        ``budget_saturated`` shutdown signal.

        Args:
            amount: Amount to reserve.
            phase: Optional phase tag (e.g. ``"compose"``, ``"review_names"``).
                When a cap is configured for this phase, the reservation is
                rejected if it would push the phase's cumulative committed
                spend beyond ``cap × 1.5``.
        """
        if not math.isfinite(amount) or amount <= 0:
            raise ValueError("reserved provider exposure must be finite and positive")
        with self._lock:
            # ── Per-phase cap check ────────────────────────────────────────
            if phase and phase in self._phase_caps:
                cap = self._phase_caps[phase]
                committed = self._phase_committed.get(phase, 0.0)
                if committed + amount > cap * 1.5 + EPSILON:
                    logger.debug(
                        "Phase %r cap exceeded: committed=%.4f + amount=%.4f"
                        " > cap*1.5=%.4f — reservation rejected",
                        phase,
                        committed,
                        amount,
                        cap * 1.5,
                    )
                    if phase:
                        self._consecutive_reserve_failures[phase] = (
                            self._consecutive_reserve_failures.get(phase, 0) + 1
                        )
                    return None
            # ── Global pool check ──────────────────────────────────────────
            if self._pool < amount - EPSILON:
                if phase:
                    self._consecutive_reserve_failures[phase] = (
                        self._consecutive_reserve_failures.get(phase, 0) + 1
                    )
                return None
            lease_id = str(uuid.uuid4())
            self._pool -= amount
            self._reserved[lease_id] = amount
            self._batch_count += 1
            if phase:
                self._phase_committed[phase] = (
                    self._phase_committed.get(phase, 0.0) + amount
                )
                # Reset failure counter on successful reservation.
                self._consecutive_reserve_failures[phase] = 0
            self._lease_phases[lease_id] = phase
            return BudgetLease(self, amount, lease_id, phase=phase)

    # ------------------------------------------------------------------
    # Internal helpers (called by BudgetLease)
    # ------------------------------------------------------------------

    def _record_spend(self, lease_id: str, amount: float) -> float:
        """Record actual spend from a lease, returning the unfunded overspend.

        Decrements the reservation's remaining balance and increments the
        manager-wide spent counter, tracking per-phase spend for diagnostic
        attribution.  A bill above the lease remainder draws its shortfall from
        the pool, since the reservation priced an expected cost rather than a
        ceiling.  Whatever neither the lease nor the pool can fund is recorded
        as overspend and returned: the spend has already happened, and leaving
        it out of the ledger would understate the total the hard stop reads.
        """
        with self._lock:
            remaining = self._reserved.get(lease_id)
            if remaining is None:
                raise BudgetExceeded("cannot charge a released or unknown lease")
            if amount - remaining > EPSILON:
                self._extend_reservation_locked(lease_id, amount - remaining)
                remaining = self._reserved.get(lease_id, 0.0)
            overspend = max(amount - remaining, 0.0)
            self._spent += amount
            self._overspend += overspend
            phase = self._lease_phases.get(lease_id, "")
            if phase:
                self._phase_spent[phase] = self._phase_spent.get(phase, 0.0) + amount
            self._reserved[lease_id] = max(remaining - amount, 0.0)
            spent_snapshot = self._spent
        if overspend > EPSILON:
            logger.warning(
                "budget: charge $%.6f on lease %s (phase %r) exceeded both its "
                "reservation and the pool by $%.6f — recorded as overspend "
                "(spend $%.4f of cap $%.4f)",
                amount,
                lease_id,
                phase,
                overspend,
                spent_snapshot,
                self._total,
            )
        return overspend

    def _extend_reservation(self, lease_id: str, amount: float) -> float:
        """Atomically extend an active reservation by drawing from the pool.

        Returns the amount actually extended, which may be less than
        *amount* when the pool is insufficient or the lease's phase would
        exceed its hard cap (``phase_caps[phase] × 1.5``).  The caller is
        responsible for checking whether the extension was sufficient.
        """
        with self._lock:
            return self._extend_reservation_locked(lease_id, amount)

    def _extend_reservation_locked(self, lease_id: str, amount: float) -> float:
        """Extend a reservation from the pool with ``_lock`` already held.

        Phase-cap enforcement on extension prevents a compose batch from
        draining the global pool past its allocated share via in-flight
        overshoot, which would starve downstream phases (review, regen).
        """
        extended = min(amount, self._pool)
        # ── Per-phase cap check on extension ───────────────────────────
        phase = self._lease_phases.get(lease_id, "")
        if extended > 0 and phase and phase in self._phase_caps:
            cap = self._phase_caps[phase]
            committed = self._phase_committed.get(phase, 0.0)
            room = cap * 1.5 - committed
            if room < extended:
                if room <= EPSILON:
                    logger.debug(
                        "Phase %r cap exhausted on extension: "
                        "committed=%.4f cap*1.5=%.4f — extension refused",
                        phase,
                        committed,
                        cap * 1.5,
                    )
                    return 0.0
                extended = room
        if extended > 0:
            self._pool -= extended
            if lease_id in self._reserved:
                self._reserved[lease_id] += extended
            if phase:
                self._phase_committed[phase] = (
                    self._phase_committed.get(phase, 0.0) + extended
                )
            logger.info(
                "budget: extended lease %s by $%.4f "
                "(requested $%.4f, reservation now $%.4f, pool $%.4f)",
                lease_id,
                extended,
                amount,
                self._reserved.get(lease_id, 0.0),
                self._pool,
            )
        return extended

    def _release(self, lease_id: str, unused: float) -> None:
        """Return unused reservation back to the pool."""
        with self._lock:
            self._pool += unused
            self._reserved.pop(lease_id, None)
            self._lease_phases.pop(lease_id, None)

    # ------------------------------------------------------------------
    # Read-only access
    # ------------------------------------------------------------------

    @property
    def remaining(self) -> float:
        """Available pool (excludes active reservations)."""
        with self._lock:
            return self._pool

    @property
    def spent(self) -> float:
        """Total spend recorded across all leases (in-memory shadow)."""
        with self._lock:
            return self._spent

    @property
    def overspend(self) -> float:
        """Recorded spend that no reservation or pool headroom could fund."""
        with self._lock:
            return self._overspend

    @property
    def phase_spent(self) -> dict[str, float]:
        """Per-phase spend snapshot.  Keys are phase tags; values are USD."""
        with self._lock:
            return dict(self._phase_spent)

    @property
    def total_budget(self) -> float:
        """Original total budget."""
        return self._total

    def pool_spent_total(self, pool: str) -> float:
        """Return cumulative spend attributed to *pool* (via phase tag).

        Reads directly from ``_phase_spent`` — the same dict that
        :meth:`pool_admit` consults for fairness decisions.  Returns 0.0
        when the pool has never been charged.
        """
        with self._lock:
            return self._phase_spent.get(pool, 0.0)

    # ------------------------------------------------------------------
    # Pool admission control
    # ------------------------------------------------------------------

    def pool_admit(
        self,
        pool: str,
        weights: dict[str, float],
        active_pools: set[str],
        free_pools: set[str] | None = None,
    ) -> bool:
        """Soft-fairness admission gate for a pool requesting a new batch.

        Implements the weighted-share rule:

            share = pool_spent[p] / sum(pool_spent.values() or epsilon)
            effective_weight = weights[p] / sum(weights[q] for q in active_pools)
            admit iff share < effective_weight  OR  no other pool is active

        Idle pools (queue empty → not in ``active_pools``) immediately
        forfeit their weight share so active pools can borrow it.

        ``pool`` here is a logical pool name matching the keys in
        ``weights`` (e.g. ``"generate_name"``, ``"review_name"``,
        ``"refine_name"``, ``"generate_docs"``, ``"review_docs"``,
        ``"refine_docs"``).  These map 1:1 to ``_phase_spent`` keys.

        ``free_pools`` names the pools whose configured model routes to a
        local / zero-cost endpoint (e.g. ``generate_name`` on a local vLLM
        GPU).  Free pools must NOT be rationed against the *dollar* budget's
        spend-fairness rule — they cost nothing, so their only real limit is
        GPU concurrency (their replica count).  Two consequences:

        * A free pool is admitted whenever it has pending work, bypassing the
          spend-share arithmetic entirely (still subject to the hard
          budget-exhausted gate below — see note).
        * Free pools are excluded from the ``active_weight_sum`` denominator
          when computing paid pools' effective weights, so a free pool's
          weight (e.g. ``generate_name``'s 0.15) does not inflate the
          denominator and unfairly shrink the paid pools' shares.

        Returns True if the pool is permitted to claim its next batch.
        """
        if pool not in weights:
            return False
        free = free_pools or set()
        # Hard gate: never admit when budget is exhausted, regardless of
        # active_pools state.  This prevents headless mode from bypassing the
        # cost cap (the bug that caused the 10.5× live smoke overshoot).
        #
        # This gate applies to free pools too.  Free *generation* costs
        # nothing, but once the shared dollar budget is drained the paid
        # review/docs/refine pools that consume free output can no longer run,
        # so continuing to generate would only grow an unbounded backlog of
        # never-reviewed drafts.  In practice ``_budget_watchdog`` sets
        # ``stop_event`` on ``hard_exhausted()`` and tears down ALL pools
        # cooperatively, so keeping the gate here merely stops a free pool from
        # claiming fresh work in the brief window before that shutdown
        # propagates — the safe, non-wasteful choice.
        if self.exhausted():
            return False
        # ── Free-pool fast path ───────────────────────────────────────────
        # A pool whose model is a local/zero-cost endpoint is admitted as soon
        # as it has pending work — its throughput is bounded by GPU concurrency
        # (replica count), never by how paid pools are spending the cost cap.
        # (When active_pools is empty — headless/startup — the generic bypass
        # below already admits it; this branch covers the steady state.)
        if pool in free and (not active_pools or pool in active_pools):
            return True
        # When active_pools is empty (e.g. headless/non-TTY mode where the
        # Rich display never updates pending_count, or at startup before the
        # first display refresh), admit all known pools unconditionally so
        # they can discover their own work via claim() and self-regulate via
        # backoff when claim() returns None.
        #
        # With ``pending_fn`` wired in ``run_pools`` this path is only hit
        # transiently: the ``_pending_count_watchdog`` updates
        # pending counts immediately on its first poll, so ``active_pools_fn``
        # returns a non-empty set before the pools have issued any claims.
        # Without ``pending_fn`` the bypass persists indefinitely, letting all
        # pools compete without fairness weighting.
        if not active_pools:
            return True
        if pool not in active_pools:
            # This pool has no pending work — forfeit its share.
            return False
        # The paid active set excludes free pools (they bypass fairness and
        # must not occupy weight share in the denominator).
        paid_active = active_pools - free
        # Sole active paid pool always admitted — the "no other pool is
        # active" branch of the weighted-share rule.  (A lone paid pool competing
        # only against free pools rations against nobody.)
        if len(paid_active) <= 1:
            return True
        with self._lock:
            spent = dict(self._phase_spent)
        # Spend share is computed over the paid pools only — free pools incur
        # no dollar spend, so they never appear meaningfully in ``_phase_spent``
        # and must not dilute the paid pools' shares.
        total_spent = sum(spent.get(q, 0.0) for q in paid_active)
        if total_spent < EPSILON:
            return True  # nothing spent yet; everyone gets a turn
        share = spent.get(pool, 0.0) / total_spent
        active_weight_sum = sum(
            weights.get(q, 0.0) for q in paid_active if q in weights
        )
        if active_weight_sum < EPSILON:
            return True
        effective = weights[pool] / active_weight_sum
        return share < effective

    @property
    def write_failed(self) -> bool:
        """True if any graph write failed terminally (cost_is_exact → False)."""
        return self._write_failed

    @property
    def pending_cost(self) -> float:
        """Cost enqueued but not yet flushed to the graph."""
        with self._pending_lock:
            return self._pending_cost

    @property
    def batch_count(self) -> int:
        """Number of leases issued (proxy for events_total)."""
        with self._lock:
            return self._batch_count

    def exhausted(self) -> bool:
        """Return ``True`` when the pool is non-positive."""
        with self._lock:
            return self._pool <= EPSILON

    def hard_exhausted(self) -> bool:
        """Return ``True`` when committed spend has reached the cost limit.

        Unlike :meth:`exhausted`, which triggers when the pool is drained
        (including by active reservations that may later be partially
        refunded), this predicate fires only when *actual spend* has
        consumed the budget.  Use for global-shutdown decisions (watchdog,
        final stop-reason) where a transient reservation spike should NOT
        terminate the run.
        """
        with self._lock:
            return self._spent >= self._total - EPSILON

    def pool_budget_saturated(self, phase: str) -> bool:
        """Return ``True`` when *phase* has failed to reserve budget
        ``SATURATION_THRESHOLD`` times in a row.

        A saturated pool cannot fund another batch from the remaining
        budget.  The counter is only advanced by an *attempted* reserve
        that fails, and reset to 0 by any success — so a pool that has
        stopped calling :meth:`reserve` (because it ran out of *work*,
        not *budget*) holds its counter frozen below threshold and is
        therefore reported as **not** saturated.  Callers that want a
        "can no longer progress" signal must combine this with the
        pool's pending-work count (see
        ``_budget_saturation_watchdog``).
        """
        with self._lock:
            return (
                self._consecutive_reserve_failures.get(phase, 0)
                >= self.SATURATION_THRESHOLD
            )

    def all_pools_budget_saturated(
        self,
        pool_names: tuple[str, ...] = (
            "generate_name",
            "review_name",
            "refine_name",
            "generate_docs",
            "review_docs",
            "refine_docs",
        ),
    ) -> bool:
        """Return ``True`` when **every** pool in *pool_names* is
        budget-saturated.

        WARNING — over an unfiltered pool list this is almost never
        true in practice, because a pool that runs out of *work* stops
        attempting reservations and so its failure counter never reaches
        the threshold (see :meth:`pool_budget_saturated`).  A single
        drained pool (e.g. a free local-GPU generate pool that exhausted
        its sources) then vetoes the whole conjunction forever while the
        paid pools sit budget-blocked — the 0-token indefinite spin.
        The shutdown gate therefore evaluates saturation only over pools
        that still have pending work; see ``_budget_saturation_watchdog``.
        This method is retained for diagnostics and direct unit tests.
        """
        return all(self.pool_budget_saturated(p) for p in pool_names)

    # ------------------------------------------------------------------
    # Graph-aware reads
    # ------------------------------------------------------------------

    async def get_total_spent(self, *, force_refresh: bool = False) -> float:
        """Return total spend for this run from graph + in-flight pending.

        Cached for ``_graph_cache_ttl`` seconds to avoid hammering Neo4j.
        Falls back to in-memory ``_spent`` when no ``run_id`` is set.
        """
        return self._get_total_spent_sync(force_refresh=force_refresh)

    def _get_total_spent_sync(self, *, force_refresh: bool = False) -> float:
        """Synchronous implementation of :meth:`get_total_spent`.

        Separated so callers in async shutdown paths can wrap this in
        ``asyncio.to_thread`` + ``wait_for`` without nesting coroutines.
        """
        if self.run_id is None:
            with self._lock:
                return self._spent

        now = time.monotonic()
        if force_refresh or (now - self._graph_total_ts) > self._graph_cache_ttl:
            try:
                from imas_codex.standard_names.graph_ops import (
                    aggregate_spend_for_run,
                )

                self._graph_total_cache = aggregate_spend_for_run(self.run_id)
                self._graph_total_ts = now
            except Exception:  # noqa: BLE001
                logger.debug("graph aggregate failed, using in-memory fallback")
                with self._lock:
                    return self._spent

        with self._pending_lock:
            return self._graph_total_cache + self._pending_cost

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @property
    def summary(self) -> dict[str, Any]:
        """Snapshot of budget state for logging / display."""
        with self._lock:
            return {
                "total_budget": self._total,
                "remaining": self._pool,
                "total_spent": self._spent,
                "overspend": self._overspend,
                "active_reservations": len(self._reserved),
                "total_reserved": sum(self._reserved.values()),
                "batch_count": self._batch_count,
                "phase_committed": dict(self._phase_committed),
                "phase_spent": dict(self._phase_spent),
                "run_id": self.run_id,
                "write_failed": self._write_failed,
                "pending_writes": self._write_queue.qsize(),
            }

    def check_invariant(self) -> bool:
        """Verify pool + sum(active_reserved) + spent == total + overspend."""
        with self._lock:
            expected = self._total + self._overspend
            actual = self._pool + sum(self._reserved.values()) + self._spent
            return abs(expected - actual) < EPSILON * 100
