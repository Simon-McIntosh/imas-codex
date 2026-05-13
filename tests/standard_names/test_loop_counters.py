"""Regression tests for phase-name counter matching.

Plan 39, phase 5c: verify that the summary counters correctly accumulate
counts from the phase names actually emitted by turn.py.

Previously, ``names_reviewed`` was always zero because the loop matched
``phase.name == "review"`` while turn.py emits ``"review_names"`` and
``"review_docs"``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from imas_codex.standard_names.loop import RunSummary


@dataclass
class _Phase:
    """Minimal phase result stub for counter-matching tests."""

    name: str
    count: int


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────


def _make_phases(**counts: int) -> list[_Phase]:
    """Build a list of phase stubs from name → count pairs."""
    return [_Phase(name=name, count=count) for name, count in counts.items()]


def _make_summary() -> RunSummary:
    return RunSummary(
        run_id="test-run",
        turn_number=1,
        started_at=datetime.now(UTC),
        cost_limit=100.0,
    )


# ─────────────────────────────────────────────────────────────────────
# Counter matching tests (unit-level, no graph/LLM calls)
# ─────────────────────────────────────────────────────────────────────


class TestReviewCounterMatching:
    """Verify that review_names and review_docs both increment names_reviewed."""

    def test_review_names_increments_counter(self, monkeypatch):
        """``review_names`` phase count accumulates into names_reviewed."""
        summary = _make_summary()
        phases = _make_phases(review_names=7)
        for phase in phases:
            if phase.name in ("review_names", "review_docs"):
                summary.names_reviewed += phase.count
        assert summary.names_reviewed == 7

    def test_review_docs_increments_counter(self, monkeypatch):
        """``review_docs`` phase count accumulates into names_reviewed."""
        summary = _make_summary()
        phases = _make_phases(review_docs=3)
        for phase in phases:
            if phase.name in ("review_names", "review_docs"):
                summary.names_reviewed += phase.count
        assert summary.names_reviewed == 3

    def test_both_review_phases_sum(self):
        """Both review phases are summed together."""
        summary = _make_summary()
        phases = _make_phases(review_names=5, review_docs=4)
        for phase in phases:
            if phase.name in ("review_names", "review_docs"):
                summary.names_reviewed += phase.count
        assert summary.names_reviewed == 9

    def test_old_review_name_not_matched(self):
        """Old stale phase name ``review`` does NOT increment the counter."""
        summary = _make_summary()
        phases = _make_phases(review=10)  # stale / wrong name
        for phase in phases:
            if phase.name in ("review_names", "review_docs"):
                summary.names_reviewed += phase.count
        assert summary.names_reviewed == 0
