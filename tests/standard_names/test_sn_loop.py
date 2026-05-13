"""Tests for the SN loop utilities.

Covers:
- ``RunSummary`` / ``summary_table`` shape.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from imas_codex.standard_names.loop import (
    RunSummary,
    summary_table,
)

# ═══════════════════════════════════════════════════════════════════════
# RunSummary / summary_table
# ═══════════════════════════════════════════════════════════════════════


class TestRunSummary:
    def test_summary_table_shape(self):
        s = RunSummary(
            run_id="abc123",
            turn_number=3,
            started_at=datetime(2025, 1, 1, tzinfo=UTC),
            stopped_at=datetime(2025, 1, 1, 0, 5, tzinfo=UTC),
            cost_spent=1.2345,
            cost_limit=5.0,
            names_composed=10,
            names_enriched=8,
            names_reviewed=8,
            names_regenerated=2,
            domains_touched={"equilibrium", "magnetics"},
            stop_reason="completed",
        )
        row = summary_table(s)
        assert row["run_id"] == "abc123"
        assert row["turn_number"] == 3
        assert row["cost_spent"] == 1.2345
        assert row["stop_reason"] == "completed"
        assert row["domains_touched"] == ["equilibrium", "magnetics"]
        assert row["elapsed_s"] == 300.0
