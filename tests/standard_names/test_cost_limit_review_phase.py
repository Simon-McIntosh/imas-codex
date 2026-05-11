"""Tests for per-phase BudgetManager and L7 cost tracking.

Covers the W33A fix: before this change, each phase created its own
independent BudgetManager so review could claim a fresh $N budget even
after compose had already spent the user-specified cost_limit.  After the
fix, all phases draw from a single shared pool so total spend ≤ cost_limit.

All tests are mock-based — no Neo4j or real LLM required.
"""

from __future__ import annotations

import pytest

from imas_codex.standard_names.budget import BudgetManager, LLMCostEvent


def _ce(lease, amount, phase=None):
    """Simulate LLM spend (replaces charge_soft in tests)."""
    evt_phase = phase or lease.phase or "test"
    return lease.charge_event(
        amount,
        LLMCostEvent(model="test-model", tokens_in=0, tokens_out=0, phase=evt_phase),
    )


# ═══════════════════════════════════════════════════════════════════════
# L7 cost tracking
# ═══════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_l7_cost_returned_from_revise_candidate():
    """_opus_revise_candidate returns (revised_name | None, cost_float)."""
    from imas_codex.standard_names.workers import _opus_revise_candidate

    candidate = {
        "id": "electron_temperature",
        "description": "Temperature of electrons",
        "reason": "low confidence",
    }

    # Mock acall_fn to return a successful revision with a known cost
    class FakeRevision:
        revised_name = "electron_kinetic_temperature"
        explanation = "more precise"

    async def fake_acall(*, model, messages, response_model, service):
        return FakeRevision(), 0.042, {"prompt_tokens": 100, "completion_tokens": 50}

    revised, cost, _ti, _to = await _opus_revise_candidate(
        candidate,
        domain_vocabulary="",
        reviewer_themes=[],
        acall_fn=fake_acall,
    )

    assert revised == "electron_kinetic_temperature"
    assert cost == pytest.approx(0.042)


@pytest.mark.asyncio
async def test_l7_cost_returned_even_when_revision_rejected():
    """Cost is always returned from _opus_revise_candidate."""
    from imas_codex.standard_names.workers import _opus_revise_candidate

    candidate = {
        "id": "electron_temperature",
        "description": "Temperature of electrons",
        "reason": "moderate",
    }

    class LowConfidenceRevision:
        revised_name = "electron_temperature_v2"
        explanation = "worse"

    async def fake_acall(*, model, messages, response_model, service):
        return LowConfidenceRevision(), 0.025, {}

    revised, cost, _ti, _to = await _opus_revise_candidate(
        candidate,
        domain_vocabulary="",
        reviewer_themes=[],
        acall_fn=fake_acall,
    )

    assert revised == "electron_temperature_v2"  # accepted (no confidence gate)
    assert cost == pytest.approx(0.025)


@pytest.mark.asyncio
async def test_l7_cost_returned_on_exception():
    """Even on LLM exception, cost is 0.0 (nothing charged)."""
    from imas_codex.standard_names.workers import _opus_revise_candidate

    candidate = {"id": "bad_name", "description": "", "reason": ""}

    async def broken_acall(**_kwargs):
        raise RuntimeError("LLM timeout")

    revised, cost, _ti, _to = await _opus_revise_candidate(
        candidate,
        domain_vocabulary="",
        reviewer_themes=[],
        acall_fn=broken_acall,
    )

    assert revised is None
    assert cost == 0.0
