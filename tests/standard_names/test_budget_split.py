"""Tests for budget-split and per-phase hard caps.

Covers:
  - TURN_SPLIT constant values
  - BudgetManager per-phase cap rejects over-budget reservations
  - Per-phase caps are independent (compose cap does not block review)
  - Split tuple sums exactly to 1.0
"""

from __future__ import annotations

import pytest

from imas_codex.standard_names.budget import BudgetManager
from imas_codex.standard_names.turn import (
    TURN_SPLIT,
    TurnConfig,
)

# ── Split constants ──────────────────────────────────────────────────────────


def test_split_sum_invariant():
    """Split must sum to exactly 1.0."""
    assert abs(sum(TURN_SPLIT) - 1.0) < 1e-9, (
        f"TURN_SPLIT sums to {sum(TURN_SPLIT)}, expected 1.0"
    )


# ── TurnConfig default ──────────────────────────────────────────────────────


def test_default_split():
    """TurnConfig default uses the 30/25/15/15/15 split."""
    cfg = TurnConfig(domain="magnetics")
    assert cfg.split == TURN_SPLIT
    assert cfg.split == (0.30, 0.25, 0.15, 0.15, 0.15)


# ── BudgetManager per-phase caps ─────────────────────────────────────────────


def test_phase_cap_enforces_at_1_5x():
    """BudgetManager with cap=1.0 accepts 1.4 but rejects 1.6 (limit = 1.5)."""
    # 1.4 ≤ 1.0 * 1.5 = 1.5 → should succeed
    mgr_ok = BudgetManager(total_budget=10.0, phase_caps={"compose": 1.0})
    lease = mgr_ok.reserve(1.4, phase="compose")
    assert lease is not None, "reservation of 1.4 should succeed (< cap*1.5=1.5)"
    lease.release_unused()

    # 1.6 > 1.0 * 1.5 = 1.5 → should be rejected
    mgr_bad = BudgetManager(total_budget=10.0, phase_caps={"compose": 1.0})
    lease2 = mgr_bad.reserve(1.6, phase="compose")
    assert lease2 is None, "reservation of 1.6 should be rejected (> cap*1.5=1.5)"


def test_phase_cap_per_phase():
    """Compose phase at-cap does not block an independent review reservation."""
    mgr = BudgetManager(
        total_budget=10.0,
        phase_caps={"compose": 1.0, "review_names": 3.0},
    )

    # Fill compose to its hard cap (1.0 * 1.5 = 1.5)
    lease_compose = mgr.reserve(1.5, phase="compose")
    assert lease_compose is not None, "initial compose reservation should succeed"

    # Compose is now at-cap — next compose reservation must be rejected
    lease_compose2 = mgr.reserve(0.1, phase="compose")
    assert lease_compose2 is None, "compose should be rejected once at-cap"

    # review_names has its own cap — must not be blocked by compose
    lease_review = mgr.reserve(1.0, phase="review_names")
    assert lease_review is not None, (
        "review_names reservation must succeed independently"
    )
