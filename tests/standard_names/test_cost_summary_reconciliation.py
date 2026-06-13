"""Cost-display reconciliation against the authoritative budget ledger.

Regression guard for a display-fidelity bug: the rich summary computed
``TOTAL COST`` as ``sum(pool.cost)`` over emitted ``on_event`` payloads,
but several charge paths bill the ``BudgetManager`` ledger *without*
emitting a matching display event — fanout sub-charges, grammar-retry
charges, and retry-accumulated cost inside ``acall_llm_structured``.
The budget ledger sees every charge; the event sum did not.  Result: a
live run showed ``TOTAL COST $2.42`` while the ledger (which correctly
enforced the cap) had ``$5.43`` — ~$3 of real spend invisible, and in
the dangerous direction (hiding spend).

These tests verify that once the run wires its ledger into the display
(:meth:`SN6PoolDisplay.set_budget_ledger`), both the mid-run COST gauge
and ``print_summary`` reconcile to the ledger total, while completed
counts still come from the live per-pool counters.  With no ledger
wired (tests / dry-run), the display falls back to the event sum.
"""

from __future__ import annotations

import pytest
from rich.console import Console

from imas_codex.standard_names.display import POOL_LABELS, SN6PoolDisplay


def _feed_event(display: SN6PoolDisplay, pool: str, *, name: str, cost: float) -> None:
    """Push one ``on_event`` payload (carrying only the surfaced cost)."""
    display.on_event({"pool": pool, "name": name, "cost": cost, "score": 0.9})


def _summary_text(display: SN6PoolDisplay) -> str:
    """Capture the rich ``print_summary`` output as plain text."""
    console = Console(record=True, width=120, force_terminal=True)
    display.console = console
    display.print_summary()
    return console.export_text()


class TestPrintSummaryReconcilesToLedger:
    """``print_summary`` TOTAL COST must equal the budget total, not events."""

    def test_total_cost_equals_ledger_not_event_sum(self):
        display = SN6PoolDisplay(cost_limit=5.0)

        # Events surface only part of the real spend (the visible cost of
        # one emitted event per pool).  Event sum = 0.40 + 0.80 = $1.20.
        _feed_event(display, "generate_name", name="a", cost=0.40)
        _feed_event(display, "refine_name", name="b", cost=0.80)
        event_sum = sum(p.cost for p in display.pools.values())
        assert event_sum == pytest.approx(1.20)

        # The ledger saw every charge — including fanout / retry
        # sub-charges that emitted no display event.  refine_name folds
        # its fanout sub-charges into its own phase tag automatically.
        display.set_budget_ledger(
            phase_spent={"generate_name": 0.40, "refine_name": 4.00},
            total=5.43,
        )

        text = _summary_text(display)

        # TOTAL COST reconciles to the ledger, NOT the $1.20 event sum.
        assert "TOTAL COST: $5.43" in text
        assert "$1.20" not in text

    def test_per_pool_cost_lines_use_ledger(self):
        display = SN6PoolDisplay(cost_limit=10.0)
        _feed_event(display, "generate_name", name="a", cost=0.40)
        _feed_event(display, "refine_name", name="b", cost=0.80)

        display.set_budget_ledger(
            phase_spent={"generate_name": 0.40, "refine_name": 4.00},
            total=4.40,
        )
        text = _summary_text(display)

        # The refine_name row shows the ledger figure ($4.00), which folds
        # in the invisible fanout sub-charges, not the $0.80 event cost.
        assert f"{POOL_LABELS['refine_name']}:" in text
        assert "$4.00" in text
        assert "$0.80" not in text

    def test_completed_counts_stay_from_live_counters(self):
        """Counts come from on_event; only COST switches to the ledger."""
        display = SN6PoolDisplay(cost_limit=10.0)
        for i in range(3):
            _feed_event(display, "generate_name", name=f"n{i}", cost=0.10)
        _feed_event(display, "refine_name", name="r", cost=0.20)

        display.set_budget_ledger(
            phase_spent={"generate_name": 0.30, "refine_name": 2.50},
            total=2.80,
        )
        text = _summary_text(display)

        # generate_name completed = 3 (live counter), refine_name = 1.
        assert f"{POOL_LABELS['generate_name']}: 3" in text
        assert f"{POOL_LABELS['refine_name']}: 1" in text

    def test_total_defaults_to_phase_sum_when_total_omitted(self):
        display = SN6PoolDisplay(cost_limit=10.0)
        _feed_event(display, "generate_name", name="a", cost=0.10)

        display.set_budget_ledger(
            phase_spent={"generate_name": 1.00, "refine_name": 2.50},
        )
        text = _summary_text(display)
        assert "TOTAL COST: $3.50" in text


class TestFallbackWhenNoLedger:
    """With no ledger wired (tests / dry-run), fall back to the event sum."""

    def test_total_cost_falls_back_to_event_sum(self):
        display = SN6PoolDisplay(cost_limit=5.0)
        _feed_event(display, "generate_name", name="a", cost=0.40)
        _feed_event(display, "review_name", name="b", cost=0.30)

        text = _summary_text(display)
        # No set_budget_ledger call — event sum ($0.70) is the source.
        assert "TOTAL COST: $0.70" in text

    def test_no_crash_with_empty_display(self):
        display = SN6PoolDisplay(cost_limit=5.0)
        # No events, no ledger — must not raise and prints nothing notable.
        _summary_text(display)  # exercises the total_items == 0 path


class TestResourceGaugeReconcilesToLedger:
    """The mid-run COST gauge must reflect the ledger total when wired."""

    def test_cost_gauge_uses_ledger_total(self):
        display = SN6PoolDisplay(cost_limit=5.0)
        _feed_event(display, "generate_name", name="a", cost=0.40)
        _feed_event(display, "refine_name", name="b", cost=0.80)

        display.set_budget_ledger(
            phase_spent={"generate_name": 0.40, "refine_name": 4.00},
            total=4.40,
        )
        section = display._build_resources_section()
        text = section.plain

        assert "COST" in text
        # Ledger total ($4.40) is shown, not the $1.20 event sum.
        assert "$4.40" in text
        assert "$1.20" not in text

    def test_cost_gauge_falls_back_to_event_sum(self):
        display = SN6PoolDisplay(cost_limit=5.0)
        _feed_event(display, "generate_name", name="a", cost=0.40)
        _feed_event(display, "review_name", name="b", cost=0.30)

        section = display._build_resources_section()
        text = section.plain
        assert "COST" in text
        assert "$0.70" in text
