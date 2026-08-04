"""Tests for graph-backed BudgetManager.

All tests are pure unit tests with mocked ``record_llm_cost``.
No live Neo4j required.

Covers:
- charge_event enqueues to async writer
- Lease decision uses pending cache
- charge_event hard lease enforcement
- drain_pending returns True on success
- drain_pending returns False on writer failure
- Writer retries on transient error
- get_total_spent combines graph + pending
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pytest

from imas_codex.standard_names.budget import (
    BudgetExceeded,
    BudgetExposureUnknown,
    BudgetManager,
    ChargeResult,
    LLMCostEvent,
    provider_exposure,
)

# ── Helpers ──────────────────────────────────────────────────────────────


def _make_event(**overrides) -> LLMCostEvent:
    """Create a test LLMCostEvent with sensible defaults."""
    defaults = {
        "model": "test-model",
        "tokens_in": 100,
        "tokens_out": 50,
        "phase": "generate",
    }
    defaults.update(overrides)
    return LLMCostEvent(**defaults)


# =====================================================================
# charge_event enqueues to writer
# =====================================================================


class TestChargeEventEnqueuesToWriter:
    """charge_event should enqueue a graph write and update spend."""

    @pytest.mark.asyncio
    async def test_charge_event_enqueues_to_writer(self):
        """Mock record_llm_cost, call charge_event, await drain, verify mock."""
        mock_record = MagicMock()
        mgr = BudgetManager(total_budget=10.0, run_id="run-test-001")

        await mgr.start()

        lease = mgr.reserve(5.0, phase="generate")
        assert lease is not None

        event = _make_event(
            sn_ids=("sn_a", "sn_b"),
            batch_id="batch-001",
            phase="generate",
        )
        result = lease.charge_event(2.0, event)
        assert isinstance(result, ChargeResult)
        assert result.overspend == 0.0

        # In-memory spend recorded
        assert abs(mgr.spent - 2.0) < 1e-9

        with patch(
            "imas_codex.standard_names.graph_ops.record_llm_cost",
            mock_record,
        ):
            success = await mgr.drain_pending()

        assert success is True
        assert mock_record.call_count == 1

        # Verify call args
        call_kwargs = mock_record.call_args[1]
        assert call_kwargs["run_id"] == "run-test-001"
        assert call_kwargs["phase"] == "generate"
        assert call_kwargs["model"] == "test-model"
        assert call_kwargs["cost"] == 2.0
        assert call_kwargs["tokens_in"] == 100
        assert call_kwargs["tokens_out"] == 50
        assert call_kwargs["sn_ids"] == ["sn_a", "sn_b"]
        assert call_kwargs["batch_id"] == "batch-001"

    @pytest.mark.asyncio
    async def test_charge_event_no_run_id_skips_enqueue(self):
        """Without run_id, charge_event should not enqueue any writes."""
        mgr = BudgetManager(total_budget=10.0)  # no run_id
        lease = mgr.reserve(5.0)
        assert lease is not None

        event = _make_event()
        result = lease.charge_event(1.0, event)
        assert result.overspend == 0.0
        assert abs(mgr.spent - 1.0) < 1e-9
        # No pending cost since run_id is None
        assert abs(mgr.pending_cost - 0.0) < 1e-9


# =====================================================================
# Lease decision uses pending cache
# =====================================================================


class TestLeaseDecisionUsesPendingCache:
    """In-memory pending counter must affect lease budgeting."""

    @pytest.mark.asyncio
    async def test_charge_event_lease_decision_uses_pending_cache(self):
        """Charge $4 in-flight (writer not draining), assert remaining reflects spend."""
        mgr = BudgetManager(total_budget=5.0, run_id="run-pending-test")
        # Don't start the writer — events will pile up in the queue
        lease = mgr.reserve(5.0, phase="generate")
        assert lease is not None

        event = _make_event()
        lease.charge_event(4.0, event)

        # In-memory spent reflects the charge
        assert abs(mgr.spent - 4.0) < 1e-9
        # Remaining reservation on lease
        assert abs(lease.remaining - 1.0) < 1e-9
        # Pending cost is $4 (not yet flushed to graph)
        assert abs(mgr.pending_cost - 4.0) < 1e-9

    @pytest.mark.asyncio
    async def test_multiple_charge_events_accumulate(self):
        """Multiple charge_events must accumulate correctly."""
        mgr = BudgetManager(total_budget=10.0, run_id="run-multi")
        lease = mgr.reserve(10.0, phase="generate")
        assert lease is not None

        event = _make_event()
        lease.charge_event(1.0, event)
        lease.charge_event(2.0, event)
        lease.charge_event(3.0, event)

        assert abs(mgr.spent - 6.0) < 1e-9
        assert abs(lease.remaining - 4.0) < 1e-9
        assert abs(mgr.pending_cost - 6.0) < 1e-9


# =====================================================================
# charge_event hard lease enforcement
# =====================================================================


class TestChargeEventHardLease:
    """A charge cannot exceed the exposure reserved before launch."""

    @pytest.mark.asyncio
    async def test_over_lease_charge_is_rejected_without_accounting_drift(self):
        mgr = BudgetManager(total_budget=0.3)
        lease = mgr.reserve(0.3)
        assert lease is not None

        event = _make_event()
        with pytest.raises(BudgetExceeded, match="reserved exposure"):
            lease.charge_event(0.5, event)

        assert mgr.spent == 0.0
        assert lease.charged == 0.0
        assert mgr.check_invariant()

    @pytest.mark.parametrize("amount", [0.0, -0.1, float("nan"), float("inf")])
    def test_unproven_exposure_is_rejected_before_reservation(self, amount: float):
        mgr = BudgetManager(total_budget=1.0)

        with pytest.raises(ValueError, match="finite and positive"):
            mgr.reserve(amount)

        assert mgr.spent == 0.0
        assert mgr.remaining == 1.0

    def test_concurrent_leases_cannot_expose_more_than_run_cap(self):
        mgr = BudgetManager(total_budget=1.0)
        first = mgr.reserve(0.7, phase="review_name")
        second = mgr.reserve(0.4, phase="review_docs")

        assert first is not None
        assert second is None
        first.charge_event(0.7, _make_event(phase="review_name"))
        assert mgr.spent == pytest.approx(0.7)
        assert mgr.check_invariant()

    def test_provider_exposure_multiplies_retries_and_calls(self):
        assert provider_exposure(
            0.1,
            provider_attempts=5,
            calls=3,
        ) == pytest.approx(1.5)

    def test_unknown_provider_exposure_fails_closed(self):
        with pytest.raises(BudgetExposureUnknown, match="no finite positive"):
            provider_exposure(None, provider_attempts=5)

    def test_lease_checks_subcall_exposure_before_launch(self):
        mgr = BudgetManager(total_budget=1.0)
        lease = mgr.reserve(0.4)
        assert lease is not None

        lease.require_exposure(0.4)
        with pytest.raises(BudgetExceeded, match="exceeds lease remainder"):
            lease.require_exposure(0.5)


# =====================================================================
# drain_pending returns True on success
# =====================================================================


class TestDrainPendingSuccess:
    """drain_pending must return True when all writes succeed."""

    @pytest.mark.asyncio
    async def test_drain_pending_returns_true_on_success(self):
        """Successful drain returns True and clears pending."""
        mock_record = MagicMock()
        mgr = BudgetManager(total_budget=10.0, run_id="run-drain-ok")

        await mgr.start()

        lease = mgr.reserve(5.0)
        assert lease is not None
        lease.charge_event(1.0, _make_event())
        lease.charge_event(2.0, _make_event())

        with patch(
            "imas_codex.standard_names.graph_ops.record_llm_cost",
            mock_record,
        ):
            success = await mgr.drain_pending()

        assert success is True
        assert mock_record.call_count == 2
        assert abs(mgr.pending_cost - 0.0) < 1e-9
        assert not mgr.write_failed


# =====================================================================
# drain_pending returns False on writer failure
# =====================================================================


class TestDrainPendingFailure:
    """drain_pending must return False when writes fail terminally."""

    @pytest.mark.asyncio
    async def test_drain_pending_returns_false_on_writer_failure(self):
        """Mock record_llm_cost to raise, verify _write_failed=True."""
        mgr = BudgetManager(total_budget=10.0, run_id="run-drain-fail")

        await mgr.start()

        lease = mgr.reserve(5.0)
        assert lease is not None
        lease.charge_event(1.0, _make_event())

        with patch(
            "imas_codex.standard_names.graph_ops.record_llm_cost",
            side_effect=RuntimeError("Neo4j unreachable"),
        ):
            success = await mgr.drain_pending()

        assert success is False
        assert mgr.write_failed is True

    @pytest.mark.asyncio
    async def test_drain_pending_raise_on_failure(self):
        """drain_pending with raise_on_failure=True raises RuntimeError."""
        mgr = BudgetManager(total_budget=10.0, run_id="run-drain-raise")

        await mgr.start()

        lease = mgr.reserve(5.0)
        assert lease is not None
        lease.charge_event(1.0, _make_event())

        with patch(
            "imas_codex.standard_names.graph_ops.record_llm_cost",
            side_effect=RuntimeError("Neo4j unreachable"),
        ):
            with pytest.raises(RuntimeError, match="cost_is_exact"):
                await mgr.drain_pending(raise_on_failure=True)

    @pytest.mark.asyncio
    async def test_writer_continues_after_failure(self):
        """Writer must not halt on terminal failure — continues best-effort."""
        from imas_codex.standard_names.budget import _WRITER_MAX_RETRIES

        call_count = 0

        def mock_record(**kwargs):
            nonlocal call_count
            call_count += 1
            # Fail on ALL retry attempts for the first event
            if call_count <= _WRITER_MAX_RETRIES:
                raise RuntimeError("persistent DB error")
            # Subsequent calls (second event) succeed

        mgr = BudgetManager(total_budget=10.0, run_id="run-continue")

        await mgr.start()

        lease = mgr.reserve(5.0)
        assert lease is not None
        lease.charge_event(1.0, _make_event())
        lease.charge_event(2.0, _make_event(phase="enrich"))

        with patch(
            "imas_codex.standard_names.graph_ops.record_llm_cost",
            side_effect=mock_record,
        ):
            success = await mgr.drain_pending()

        assert success is False  # first write failed terminally
        assert mgr.write_failed is True
        # Both events were attempted: first exhausted retries, second succeeded
        assert call_count == _WRITER_MAX_RETRIES + 1


# =====================================================================
# Writer retries on transient error
# =====================================================================


class TestWriterRetries:
    """Writer must retry on transient errors with exponential backoff."""

    @pytest.mark.asyncio
    async def test_writer_retries_on_transient_error(self):
        """Mock raises once then succeeds, verify retry happened."""
        call_count = 0

        def mock_record(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ConnectionError("transient Neo4j error")
            # Second call succeeds

        mgr = BudgetManager(total_budget=10.0, run_id="run-retry")

        await mgr.start()

        lease = mgr.reserve(5.0)
        assert lease is not None
        lease.charge_event(1.0, _make_event())

        with patch(
            "imas_codex.standard_names.graph_ops.record_llm_cost",
            side_effect=mock_record,
        ):
            success = await mgr.drain_pending()

        assert success is True
        assert call_count == 2  # first failed, second succeeded
        assert not mgr.write_failed


# =====================================================================
# get_total_spent combines graph + pending
# =====================================================================


class TestGetTotalSpent:
    """get_total_spent must combine graph aggregate with pending cache."""

    @pytest.mark.asyncio
    async def test_get_total_spent_combines_graph_plus_pending(self):
        """Graph total + pending cost = total reported by get_total_spent."""
        mgr = BudgetManager(total_budget=10.0, run_id="run-total")
        # Don't start writer — events stay in queue as pending
        lease = mgr.reserve(5.0)
        assert lease is not None
        lease.charge_event(2.0, _make_event())

        # Mock the graph aggregate to return $3.00 (previously flushed)
        with patch(
            "imas_codex.standard_names.graph_ops.aggregate_spend_for_run",
            return_value=3.0,
        ):
            total = await mgr.get_total_spent(force_refresh=True)

        # Graph says $3.00, pending has $2.00
        assert abs(total - 5.0) < 1e-9

    @pytest.mark.asyncio
    async def test_get_total_spent_no_run_id_uses_memory(self):
        """Without run_id, get_total_spent returns in-memory spent."""
        mgr = BudgetManager(total_budget=10.0)  # no run_id
        lease = mgr.reserve(5.0)
        assert lease is not None
        lease.charge_event(1.5, _make_event())

        total = await mgr.get_total_spent()
        assert abs(total - 1.5) < 1e-9

    @pytest.mark.asyncio
    async def test_get_total_spent_caches(self):
        """get_total_spent should cache the graph result for TTL period."""
        call_count = 0

        def mock_aggregate(run_id):
            nonlocal call_count
            call_count += 1
            return 1.0

        mgr = BudgetManager(total_budget=10.0, run_id="run-cache")

        with patch(
            "imas_codex.standard_names.graph_ops.aggregate_spend_for_run",
            side_effect=mock_aggregate,
        ):
            # First call fetches
            await mgr.get_total_spent(force_refresh=True)
            assert call_count == 1

            # Second call within TTL should use cache
            await mgr.get_total_spent()
            assert call_count == 1  # still 1 — cached

            # Force refresh should fetch again
            await mgr.get_total_spent(force_refresh=True)
            assert call_count == 2


# =====================================================================
# LLMCostEvent dataclass
# =====================================================================


class TestLLMCostEvent:
    """LLMCostEvent is frozen, has sensible defaults, and round-trips."""

    def test_event_is_frozen(self):
        event = _make_event()
        with pytest.raises(AttributeError):
            event.model = "other"  # type: ignore[misc]

    def test_event_defaults(self):
        event = LLMCostEvent(model="m", tokens_in=10, tokens_out=5)
        assert event.tokens_cached_read == 0
        assert event.tokens_cached_write == 0
        assert event.sn_ids == ()
        assert event.batch_id is None
        assert event.cycle is None
        assert event.phase == ""
        assert event.service == "standard-names"
        assert event.llm_at is None

    def test_event_with_all_fields(self):
        now = datetime.now(UTC)
        event = LLMCostEvent(
            model="gpt-4o",
            tokens_in=500,
            tokens_out=200,
            tokens_cached_read=100,
            tokens_cached_write=50,
            sn_ids=("sn_1", "sn_2"),
            batch_id="b-001",
            cycle="c0",
            phase="review_names",
            service="openrouter",
            llm_at=now,
        )
        assert event.model == "gpt-4o"
        assert event.sn_ids == ("sn_1", "sn_2")
        assert event.llm_at == now


# =====================================================================
# ChargeResult dataclass
# =====================================================================


class TestChargeResult:
    """ChargeResult has correct defaults."""

    def test_default_values(self):
        r = ChargeResult()
        assert r.overspend == 0.0
        assert r.hard_stop is False

    def test_overspend_set(self):
        r = ChargeResult(overspend=0.5)
        assert r.overspend == 0.5


# =====================================================================
# BudgetManager new properties
# =====================================================================


class TestBudgetManagerNewProperties:
    """New properties on BudgetManager are accessible and correct."""

    def test_run_id_stored(self):
        mgr = BudgetManager(total_budget=5.0, run_id="run-123")
        assert mgr.run_id == "run-123"

    def test_run_id_none_by_default(self):
        mgr = BudgetManager(total_budget=5.0)
        assert mgr.run_id is None

    def test_write_failed_false_initially(self):
        mgr = BudgetManager(total_budget=5.0)
        assert mgr.write_failed is False

    def test_pending_cost_zero_initially(self):
        mgr = BudgetManager(total_budget=5.0)
        assert mgr.pending_cost == 0.0

    def test_summary_includes_new_fields(self):
        mgr = BudgetManager(total_budget=5.0, run_id="run-s")
        s = mgr.summary
        assert "run_id" in s
        assert s["run_id"] == "run-s"
        assert "write_failed" in s
        assert s["write_failed"] is False

    @pytest.mark.asyncio
    async def test_start_is_idempotent(self):
        """Calling start() twice should not create two writer tasks."""
        mgr = BudgetManager(total_budget=5.0, run_id="run-idem")
        await mgr.start()
        first_task = mgr._writer_task
        await mgr.start()
        assert mgr._writer_task is first_task
        # Clean up
        await mgr._write_queue.put(None)
        await first_task
