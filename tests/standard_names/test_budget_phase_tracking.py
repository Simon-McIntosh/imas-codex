"""Tests for per-phase spend tracking in BudgetManager.

Covers:
- phase_spent property populated by _record_spend
- phase tag stored per lease and released on _release
- summary dict includes phase_spent key
- concurrent leases from different phases stay isolated
"""

from __future__ import annotations

import threading
from unittest.mock import patch

import pytest

from imas_codex.standard_names.budget import BudgetManager, LLMCostEvent

# ── Test helper ───────────────────────────────────────────────────────


def _ce(lease, amount, phase=None):
    """Simulate LLM spend via charge_event (legacy charge_soft replacement)."""
    evt_phase = phase or lease.phase or "test"
    return lease.charge_event(
        amount,
        LLMCostEvent(model="test-model", tokens_in=0, tokens_out=0, phase=evt_phase),
    )


class TestPhaseSpentTracking:
    """Unit tests for the phase_spent property on BudgetManager."""

    def test_phase_spent_empty_at_start(self):
        mgr = BudgetManager(10.0)
        assert mgr.phase_spent == {}

    def test_phase_spent_populated_on_charge(self):
        mgr = BudgetManager(10.0)
        lease = mgr.reserve(2.0, phase="generate")
        assert lease is not None
        _ce(lease, 1.5)
        lease.__exit__(None, None, None)
        assert mgr.phase_spent.get("generate", 0.0) == pytest.approx(1.5)

    def test_phase_spent_accumulates_across_batches(self):
        mgr = BudgetManager(10.0)
        for _ in range(3):
            lease = mgr.reserve(1.0, phase="review_names")
            assert lease is not None
            _ce(lease, 0.4)
            lease.__exit__(None, None, None)
        assert mgr.phase_spent.get("review_names", 0.0) == pytest.approx(1.2)

    def test_multiple_phases_tracked_independently(self):
        mgr = BudgetManager(10.0)

        g = mgr.reserve(3.0, phase="generate")
        assert g is not None
        _ce(g, 2.0)
        g.__exit__(None, None, None)

        r = mgr.reserve(3.0, phase="review_names")
        assert r is not None
        _ce(r, 1.5)
        r.__exit__(None, None, None)

        spent = mgr.phase_spent
        assert spent["generate"] == pytest.approx(2.0)
        assert spent["review_names"] == pytest.approx(1.5)

    def test_summary_includes_phase_spent(self):
        mgr = BudgetManager(5.0)
        lease = mgr.reserve(1.0, phase="regen")
        assert lease is not None
        _ce(lease, 0.7)
        lease.__exit__(None, None, None)
        s = mgr.summary
        assert "phase_spent" in s
        assert s["phase_spent"].get("regen", 0.0) == pytest.approx(0.7)

    def test_lease_phases_cleaned_up_on_release(self):
        """After release, the lease_id should not remain in _lease_phases."""
        mgr = BudgetManager(5.0)
        lease = mgr.reserve(1.0, phase="generate")
        assert lease is not None
        lease_id = lease._lease_id
        lease.__exit__(None, None, None)
        # Internal cleanup: no key left for this lease
        with mgr._lock:
            assert lease_id not in mgr._lease_phases

    def test_phase_spent_thread_safe(self):
        """Concurrent charges from multiple threads must not corrupt totals."""
        mgr = BudgetManager(100.0)
        errors: list[Exception] = []

        def worker(phase: str, amount: float, n: int) -> None:
            try:
                for _ in range(n):
                    lease = mgr.reserve(amount, phase=phase)
                    if lease is None:
                        return
                    _ce(lease, amount * 0.5)
                    lease.__exit__(None, None, None)
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [
            threading.Thread(target=worker, args=("generate", 0.1, 10)),
            threading.Thread(target=worker, args=("review_names", 0.1, 10)),
            threading.Thread(target=worker, args=("review_docs", 0.1, 10)),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        spent = mgr.phase_spent
        # Each thread charged 10 × 0.05 = 0.50
        for phase in ("generate", "review_names", "review_docs"):
            assert spent[phase] == pytest.approx(0.50, abs=1e-9)
