"""Tests for the DD completion rotator and skip-by-design filter.

Covers plan 32 Phase 4 deliverables:

- ``RunSummary`` / ``summary_table`` shape.
- ``_apply_skip_by_design`` filters ``/process/`` paths and writes
  ``configurable_meaning`` skip sources.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import patch

import pytest

from imas_codex.standard_names.loop import (
    RunSummary,
    summary_table,
)
from imas_codex.standard_names.sources.dd import _apply_skip_by_design

# ═══════════════════════════════════════════════════════════════════════
# Skip-by-design filter
# ═══════════════════════════════════════════════════════════════════════


class TestApplySkipByDesign:
    """``_apply_skip_by_design`` drops /process/ paths and records them."""

    def test_keeps_normal_paths(self):
        rows = [
            {"path": "equilibrium/time_slice/profiles_1d/psi"},
            {"path": "core_profiles/profiles_1d/electrons/temperature"},
        ]
        kept = _apply_skip_by_design(rows, write_skipped=False)
        assert len(kept) == 2

    def test_drops_process_paths(self):
        rows = [
            {"path": "equilibrium/time_slice/profiles_1d/psi"},
            {"path": "edge_sources/source/process/reactions/rate"},
        ]
        kept = _apply_skip_by_design(rows, write_skipped=False)
        assert len(kept) == 1
        assert kept[0]["path"] == "equilibrium/time_slice/profiles_1d/psi"

    def test_writes_skip_records(self):
        """Verify skip records are passed to write_skipped_sources with the
        expected ``configurable_meaning`` reason."""
        captured: list[list[dict]] = []

        def fake_write(records):
            captured.append(records)
            return len(records)

        with patch(
            "imas_codex.standard_names.graph_ops.write_skipped_sources",
            side_effect=fake_write,
        ):
            rows = [
                {
                    "path": "edge_sources/source/process/reactions/rate",
                    "description": "Reaction rate coefficient",
                }
            ]
            kept = _apply_skip_by_design(rows, write_skipped=True)

        assert kept == []
        assert len(captured) == 1
        records = captured[0]
        assert len(records) == 1
        assert records[0]["source_type"] == "dd"
        assert records[0]["skip_reason"] == "configurable_meaning"
        assert "identifier.index" in records[0]["skip_reason_detail"]
        assert records[0]["description"] == "Reaction rate coefficient"

    def test_graph_write_failure_is_non_fatal(self):
        """A Neo4j failure during skip-write must not abort extraction."""
        rows = [{"path": "edge_sources/source/process/x"}]
        with patch(
            "imas_codex.standard_names.graph_ops.write_skipped_sources",
            side_effect=RuntimeError("neo4j down"),
        ):
            # Should not raise; returns empty kept list regardless.
            kept = _apply_skip_by_design(rows, write_skipped=True)
        assert kept == []


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
