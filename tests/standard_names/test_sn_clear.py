"""Tests for LLMCost preservation across Standard Name clear operations.

Verifies that both the full-wipe path (`sn clear --force` → `clear_sn_subsystem`)
and the partial-reset path (`sn run --reset-to extracted` → `clear_standard_names`)
leave the all-time cost ledger unchanged while resetting their owned state.
"""

from __future__ import annotations

from unittest.mock import MagicMock, call, patch

import pytest

from imas_codex.standard_names.graph_store import (
    graph_catalog_reset as _owning_graph_catalog_reset,
    graph_foundation as _owning_graph_foundation,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fake_gc(
    counts: dict[str, int] | None = None,
    *,
    cost_rows: list[float] | None = None,
) -> MagicMock:
    """Return a mock GraphClient context manager.

    ``query`` returns a count row for MATCH…RETURN count(n) calls, and a
    no-op empty list for DETACH DELETE / DELETE calls.
    """
    fake_gc = MagicMock()
    fake_gc.__enter__.return_value = fake_gc
    fake_gc.__exit__.return_value = None

    default_count = counts or {}
    ledger_rows = list(cost_rows or [])

    def _query(cypher: str, **_kwargs):
        if "MATCH (c:LLMCost) DETACH DELETE c" in cypher:
            ledger_rows.clear()
            return []
        # Count queries: MATCH (n:Label) RETURN count(n) AS n
        for label, n in default_count.items():
            if f":{label}" in cypher and "count(n)" in cypher:
                return [{"n": n}]
        if "count(n)" in cypher or "count(r)" in cypher or "count(sn)" in cypher:
            return [{"n": 0}]
        # Delete / other queries return empty
        return []

    fake_gc.query = MagicMock(side_effect=_query)
    fake_gc.ledger_rows = ledger_rows
    return fake_gc


def _ledger_snapshot(fake_gc: MagicMock) -> tuple[int, float]:
    rows = fake_gc.ledger_rows
    return len(rows), sum(rows)


# ---------------------------------------------------------------------------
# clear_sn_subsystem (sn clear --force path)
# ---------------------------------------------------------------------------


class TestClearSnSubsystemPreservesLLMCost:
    """``clear_sn_subsystem`` must not include LLMCost in its deletion sweep."""

    def test_llmcost_not_in_returned_deletion_labels(self):
        """Dry-run result keys describe only state the clear may delete."""
        from imas_codex.standard_names import graph_ops

        fake_gc = _make_fake_gc({"LLMCost": 3})

        with patch.object(
            _owning_graph_foundation, "GraphClient", return_value=fake_gc
        ):
            result = _owning_graph_catalog_reset.clear_sn_subsystem(dry_run=True)

        assert "LLMCost" not in result

    def test_llmcost_count_and_spend_unchanged_in_dry_run(self):
        """Dry-run must preserve both LLMCost count and all-time spend."""
        from imas_codex.standard_names import graph_ops

        fake_gc = _make_fake_gc(cost_rows=[0.5, 1.25, 2.0])
        before = _ledger_snapshot(fake_gc)

        with patch.object(
            _owning_graph_foundation, "GraphClient", return_value=fake_gc
        ):
            _owning_graph_catalog_reset.clear_sn_subsystem(dry_run=True)

        assert _ledger_snapshot(fake_gc) == before

    def test_llmcost_count_and_spend_unchanged_on_wipe(self):
        """Full pipeline-state wipe must preserve the all-time cost ledger."""
        from imas_codex.standard_names import graph_ops

        fake_gc = _make_fake_gc(
            {
                "StandardName": 5,
                "StandardNameReview": 2,
                "StandardNameSource": 3,
                "VocabGap": 1,
                "SNRun": 1,
            },
            cost_rows=[0.25, 0.75, 3.0, 4.5],
        )
        before = _ledger_snapshot(fake_gc)

        with patch.object(
            _owning_graph_foundation, "GraphClient", return_value=fake_gc
        ):
            _owning_graph_catalog_reset.clear_sn_subsystem(dry_run=False)

        queries = [c.args[0] for c in fake_gc.query.call_args_list]
        delete_queries = [q for q in queries if "DETACH DELETE" in q]
        assert _ledger_snapshot(fake_gc) == before
        assert not any("LLMCost" in q for q in delete_queries), (
            "clear_sn_subsystem must preserve LLMCost rows"
        )

    def test_all_six_pipeline_labels_deleted(self):
        """All six pipeline-output labels must have a DETACH DELETE query."""
        from imas_codex.standard_names import graph_ops

        expected = {
            "StandardName",
            "StandardNameReview",
            "StandardNameSource",
            "DocsRevision",
            "VocabGap",
            "SNRun",
        }
        fake_gc = _make_fake_gc(dict.fromkeys(expected, 1))

        with patch.object(
            _owning_graph_foundation, "GraphClient", return_value=fake_gc
        ):
            _owning_graph_catalog_reset.clear_sn_subsystem(dry_run=False)

        queries = [c.args[0] for c in fake_gc.query.call_args_list]
        delete_queries = [q for q in queries if "DETACH DELETE" in q]
        for label in expected:
            assert any(label in q for q in delete_queries), (
                f"Missing DETACH DELETE for label: {label}"
            )

    def test_dry_run_no_detach_delete_for_llmcost(self):
        """Dry-run must never issue a DETACH DELETE for LLMCost."""
        from imas_codex.standard_names import graph_ops

        fake_gc = _make_fake_gc({"LLMCost": 10})

        with patch.object(
            _owning_graph_foundation, "GraphClient", return_value=fake_gc
        ):
            _owning_graph_catalog_reset.clear_sn_subsystem(dry_run=True)

        queries = [c.args[0] for c in fake_gc.query.call_args_list]
        assert not any("DETACH DELETE" in q for q in queries), (
            "dry_run=True must not issue any DETACH DELETE"
        )


# ---------------------------------------------------------------------------
# clear_standard_names (sn run --reset-to extracted path)
# ---------------------------------------------------------------------------


class TestClearStandardNamesPreservesLLMCost:
    """``clear_standard_names`` must preserve LLMCost rows on every path."""

    def _make_gc_for_clear_standard_names(self, sn_count: int = 5) -> MagicMock:
        """Build a fake GC that reports sn_count matching StandardName nodes."""
        fake_gc = MagicMock()
        fake_gc.__enter__.return_value = fake_gc
        fake_gc.__exit__.return_value = None

        ledger_rows = [0.4, 1.1, 2.5]

        def _query(cypher: str, **_kwargs):
            if "MATCH (c:LLMCost) DETACH DELETE c" in cypher:
                ledger_rows.clear()
                return []
            if "count(DISTINCT sn)" in cypher or "count(sn)" in cypher:
                return [{"n": sn_count}]
            if "count(r)" in cypher:
                return [{"n": 0}]
            return []

        fake_gc.query = MagicMock(side_effect=_query)
        fake_gc.ledger_rows = ledger_rows
        return fake_gc

    def test_llmcost_count_and_spend_unchanged_on_full_clear(self):
        """An unscoped clear must preserve LLMCost count and all-time spend."""
        from imas_codex.standard_names import graph_ops

        fake_gc = self._make_gc_for_clear_standard_names(sn_count=3)
        before = _ledger_snapshot(fake_gc)

        with patch.object(
            _owning_graph_foundation, "GraphClient", return_value=fake_gc
        ):
            _owning_graph_catalog_reset.clear_standard_names()

        queries = [c.args[0] for c in fake_gc.query.call_args_list]
        assert _ledger_snapshot(fake_gc) == before
        assert not any("LLMCost" in q and "DETACH DELETE" in q for q in queries), (
            "clear_standard_names must preserve LLMCost rows"
        )

    def test_llmcost_not_deleted_with_source_filter(self):
        """A SCOPED clear (source_filter) must NOT wipe the global LLMCost
        ledger — it removes only a slice of names, and wiping the whole
        ledger would erase cost history for the names left intact."""
        from imas_codex.standard_names import graph_ops

        fake_gc = self._make_gc_for_clear_standard_names(sn_count=2)

        with patch.object(
            _owning_graph_foundation, "GraphClient", return_value=fake_gc
        ):
            _owning_graph_catalog_reset.clear_standard_names(source_filter="dd")

        queries = [c.args[0] for c in fake_gc.query.call_args_list]
        assert not any("LLMCost" in q and "DETACH DELETE" in q for q in queries), (
            "scoped clear_standard_names must not wipe the whole LLMCost ledger"
        )

    def test_llmcost_not_deleted_with_stage_filter(self):
        """A stage-scoped clear must also spare the global LLMCost ledger."""
        from imas_codex.standard_names import graph_ops

        fake_gc = self._make_gc_for_clear_standard_names(sn_count=2)

        with patch.object(
            _owning_graph_foundation, "GraphClient", return_value=fake_gc
        ):
            _owning_graph_catalog_reset.clear_standard_names(stage_filter=["drafted"])

        queries = [c.args[0] for c in fake_gc.query.call_args_list]
        assert not any("LLMCost" in q and "DETACH DELETE" in q for q in queries), (
            "stage-scoped clear_standard_names must not wipe the LLMCost ledger"
        )

    def test_llmcost_not_deleted_in_dry_run(self):
        """Dry-run must not issue DETACH DELETE for LLMCost."""
        from imas_codex.standard_names import graph_ops

        fake_gc = self._make_gc_for_clear_standard_names(sn_count=5)

        with patch.object(
            _owning_graph_foundation, "GraphClient", return_value=fake_gc
        ):
            result = _owning_graph_catalog_reset.clear_standard_names(dry_run=True)

        queries = [c.args[0] for c in fake_gc.query.call_args_list]
        assert not any("LLMCost" in q and "DETACH DELETE" in q for q in queries), (
            "dry_run=True must not delete LLMCost"
        )
        # dry_run returns the count of SNs that would be deleted, not 0
        assert isinstance(result, int)

    def test_llmcost_not_deleted_when_no_matching_sn(self):
        """When count == 0, no deletions including LLMCost should occur."""
        from imas_codex.standard_names import graph_ops

        fake_gc = self._make_gc_for_clear_standard_names(sn_count=0)

        with patch.object(
            _owning_graph_foundation, "GraphClient", return_value=fake_gc
        ):
            result = _owning_graph_catalog_reset.clear_standard_names()

        queries = [c.args[0] for c in fake_gc.query.call_args_list]
        assert not any("DETACH DELETE" in q for q in queries), (
            "When no matching SNs, no DETACH DELETE should be issued"
        )
        assert result == 0


class TestClearStandardNamesResetsOrphanedSources:
    """Clearing names must reset orphaned composed/attached sources.

    Deleting a StandardName strands its StandardNameSource at
    'composed'/'attached' — statuses the generate pool never claims — so
    the clear path must revert sources with no remaining PRODUCED_NAME
    edge to 'extracted'.
    """

    def _make_gc(self, sn_count: int = 5) -> MagicMock:
        fake_gc = MagicMock()
        fake_gc.__enter__.return_value = fake_gc
        fake_gc.__exit__.return_value = None

        def _query(cypher: str, **_kwargs):
            if "count(DISTINCT sn)" in cypher or "count(sn)" in cypher:
                return [{"n": sn_count}]
            if "count(r)" in cypher:
                return [{"n": 0}]
            if "count(sns)" in cypher:
                return [{"n": 7}]
            return []

        fake_gc.query = MagicMock(side_effect=_query)
        return fake_gc

    def test_orphaned_sources_reset_on_clear(self):
        from imas_codex.standard_names import graph_ops

        fake_gc = self._make_gc(sn_count=3)
        with patch.object(
            _owning_graph_foundation, "GraphClient", return_value=fake_gc
        ):
            _owning_graph_catalog_reset.clear_standard_names()

        queries = [c.args[0] for c in fake_gc.query.call_args_list]
        reset_q = [
            q
            for q in queries
            if "PRODUCED_NAME" in q and "SET sns.status = 'extracted'" in q
        ]
        assert reset_q, (
            "clear_standard_names must reset orphaned composed/attached "
            "sources to 'extracted'"
        )
        # Orphan guard and claim-field clearing present
        assert "NOT (sns)-[:PRODUCED_NAME]->(:StandardName)" in reset_q[0]
        assert "sns.claimed_at = null" in reset_q[0]
        assert "'composed'" in reset_q[0] and "'attached'" in reset_q[0]

    def test_no_source_reset_in_dry_run(self):
        from imas_codex.standard_names import graph_ops

        fake_gc = self._make_gc(sn_count=5)
        with patch.object(
            _owning_graph_foundation, "GraphClient", return_value=fake_gc
        ):
            _owning_graph_catalog_reset.clear_standard_names(dry_run=True)

        queries = [c.args[0] for c in fake_gc.query.call_args_list]
        assert not any("SET sns.status = 'extracted'" in q for q in queries), (
            "dry_run must not reset source statuses"
        )

    def test_no_source_reset_when_no_matching_sn(self):
        from imas_codex.standard_names import graph_ops

        fake_gc = self._make_gc(sn_count=0)
        with patch.object(
            _owning_graph_foundation, "GraphClient", return_value=fake_gc
        ):
            _owning_graph_catalog_reset.clear_standard_names()

        queries = [c.args[0] for c in fake_gc.query.call_args_list]
        assert not any("SET sns.status = 'extracted'" in q for q in queries)
