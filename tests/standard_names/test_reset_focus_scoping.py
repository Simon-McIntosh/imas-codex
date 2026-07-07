"""Tests for dd-path scoping and the accepted-wipe guard on reset/clear.

Two behaviours are covered:

1. SCOPE — ``reset_standard_names`` / ``clear_standard_names`` accept a
   ``path_allowlist`` that restricts the operation to StandardName nodes
   attached (via ``HAS_STANDARD_NAME``) to an IMASNode whose ``src.id`` is
   *exactly* in the allowlist. This is an exact-path match (``src.id IN
   $path_allowlist``), NOT a prefix (``STARTS WITH``) — so names on other
   paths, including accepted catalog names, are never touched.

2. GUARD — ``sn run --reset-to … --include-accepted`` with no scope is the
   unscoped graph-wide accepted wipe that destroyed 1863 committed catalog
   names in a prior incident. It must hard-error (``click.UsageError``). A
   focused (``--focus``) or otherwise narrowed reset is allowed.

All graph interaction is mocked (no live Neo4j), mirroring the established
pattern in ``test_graph_ops.py`` / ``test_sn_clear.py``.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import click
import pytest
from click.testing import CliRunner

# ---------------------------------------------------------------------------
# Helpers — mock GraphClient that returns count/collect rows by query shape
# ---------------------------------------------------------------------------


def _make_gc(count: int = 5) -> MagicMock:
    """A mock GraphClient context manager for reset/clear.

    Count queries return ``[{"n": count}]``; the reset "collect ids" query
    (``RETURN DISTINCT sn.id AS sn_id``) returns two synthetic ids; all
    other (mutation) queries return an empty list.
    """
    gc = MagicMock()
    gc.__enter__.return_value = gc
    gc.__exit__.return_value = None

    def _query(cypher: str, **_kwargs):
        if "AS sn_id" in cypher:
            return [{"sn_id": "sn_alpha"}, {"sn_id": "sn_beta"}]
        if "count(" in cypher:
            return [{"n": count}]
        return []

    gc.query = MagicMock(side_effect=_query)
    return gc


def _all_cypher(gc: MagicMock) -> str:
    return " ".join(c.args[0] for c in gc.query.call_args_list)


def _all_kwargs(gc: MagicMock) -> list[dict]:
    return [c.kwargs for c in gc.query.call_args_list]


# ---------------------------------------------------------------------------
# SCOPE — path_allowlist on reset_standard_names
# ---------------------------------------------------------------------------


class TestResetStandardNamesPathAllowlist:
    def _call(self, gc: MagicMock, **kwargs):
        from imas_codex.standard_names import graph_ops

        with patch.object(graph_ops, "GraphClient", return_value=gc):
            return graph_ops.reset_standard_names(**kwargs)

    def test_joins_imasnode_with_exact_path_match(self) -> None:
        gc = _make_gc()
        self._call(gc, path_allowlist=["eq/a", "eq/b"])

        cypher = _all_cypher(gc)
        # Selection goes through the IMASNode HAS_STANDARD_NAME join …
        assert "(src:IMASNode)-[:HAS_STANDARD_NAME]->(sn:StandardName)" in cypher
        # … restricted by an EXACT path-list membership test …
        assert "src.id IN $path_allowlist" in cypher
        # … NOT a prefix match (that is ids_filter semantics, kept separate).
        assert "STARTS WITH" not in cypher

    def test_path_allowlist_param_passed_verbatim(self) -> None:
        gc = _make_gc()
        self._call(gc, path_allowlist=["eq/a", "eq/b"])

        lists = [
            kw["path_allowlist"] for kw in _all_kwargs(gc) if "path_allowlist" in kw
        ]
        assert lists, "path_allowlist param was never passed to any query"
        assert all(pl == ["eq/a", "eq/b"] for pl in lists)

    def test_mutations_scoped_to_collected_ids_only(self) -> None:
        """Field-clear / edge-delete queries operate on the collected sn ids,
        never on an unscoped ``MATCH (sn:StandardName) WHERE <stage>``."""
        gc = _make_gc()
        self._call(gc, path_allowlist=["eq/a"])

        mutating = [
            c.args[0]
            for c in gc.query.call_args_list
            if ("SET " in c.args[0] or "DELETE " in c.args[0])
            and "AS sn_id" not in c.args[0]
        ]
        assert mutating, "expected at least one mutation query"
        for q in mutating:
            assert "sn.id IN $sn_ids" in q, (
                f"mutation not scoped to collected ids:\n{q}"
            )

    def test_no_path_allowlist_leaves_behaviour_unchanged(self) -> None:
        gc = _make_gc()
        self._call(gc, from_stage="drafted")
        cypher = _all_cypher(gc)
        assert "$path_allowlist" not in cypher

    def test_empty_allowlist_fails_closed_matches_nothing(self) -> None:
        """An empty allowlist selects the SCOPED join branch (matching
        nothing via ``src.id IN []``) — it must NEVER fall through to the
        unscoped whole-graph reset."""
        gc = _make_gc(count=0)
        self._call(gc, path_allowlist=[])

        cypher = _all_cypher(gc)
        # Scoped join branch chosen (count(DISTINCT sn) + IN membership) …
        assert "count(DISTINCT sn)" in cypher
        assert "src.id IN $path_allowlist" in cypher
        # … and the empty list is passed verbatim so IN [] matches nothing.
        lists = [
            kw["path_allowlist"] for kw in _all_kwargs(gc) if "path_allowlist" in kw
        ]
        assert lists and all(pl == [] for pl in lists)


# ---------------------------------------------------------------------------
# SCOPE — path_allowlist on clear_standard_names
# ---------------------------------------------------------------------------


class TestClearStandardNamesPathAllowlist:
    def _call(self, gc: MagicMock, **kwargs):
        from imas_codex.standard_names import graph_ops

        with patch.object(graph_ops, "GraphClient", return_value=gc):
            return graph_ops.clear_standard_names(**kwargs)

    def test_joins_imasnode_with_exact_path_match(self) -> None:
        gc = _make_gc()
        self._call(gc, path_allowlist=["eq/a", "eq/b"])

        cypher = _all_cypher(gc)
        assert "(src:IMASNode)-[:HAS_STANDARD_NAME]->(sn:StandardName)" in cypher
        assert "src.id IN $path_allowlist" in cypher
        assert "STARTS WITH" not in cypher

    def test_path_allowlist_param_passed_verbatim(self) -> None:
        gc = _make_gc()
        self._call(gc, path_allowlist=["eq/a", "eq/b"])
        lists = [
            kw["path_allowlist"] for kw in _all_kwargs(gc) if "path_allowlist" in kw
        ]
        assert lists
        assert all(pl == ["eq/a", "eq/b"] for pl in lists)

    def test_deletion_is_relationship_first_not_blanket(self) -> None:
        """A scoped clear must remove the HAS_STANDARD_NAME edge for the
        allowlisted paths and only delete nodes that become orphans — never
        the unscoped ``MATCH (sn) WHERE <stage> DETACH DELETE sn`` that would
        wipe every matching name graph-wide."""
        gc = _make_gc()
        self._call(gc, path_allowlist=["eq/a"])

        detach_deletes = [
            c.args[0]
            for c in gc.query.call_args_list
            if "DETACH DELETE sn" in c.args[0]
        ]
        assert detach_deletes, "expected an orphan DETACH DELETE sn query"
        for q in detach_deletes:
            # The StandardName delete must be gated by the orphan guard, i.e.
            # it only removes names with no remaining HAS_STANDARD_NAME edge.
            assert "NOT EXISTS { MATCH ()-[:HAS_STANDARD_NAME]->(sn) }" in q, (
                f"scoped clear used an unscoped blanket delete:\n{q}"
            )

    def test_no_path_allowlist_uses_blanket_detach(self) -> None:
        """Without a scope, the default path is still the blanket delete
        (backward-compat) — the allowlist is purely additive."""
        gc = _make_gc()
        self._call(gc)
        cypher = _all_cypher(gc)
        assert "$path_allowlist" not in cypher

    def test_empty_allowlist_fails_closed_matches_nothing(self) -> None:
        """An empty allowlist selects the SCOPED join branch (matching
        nothing via ``src.id IN []``) — it must NEVER fall through to the
        unscoped blanket delete across the whole graph."""
        gc = _make_gc(count=0)
        self._call(gc, path_allowlist=[])

        cypher = _all_cypher(gc)
        assert "(src:IMASNode)-[:HAS_STANDARD_NAME]->(sn:StandardName)" in cypher
        assert "src.id IN $path_allowlist" in cypher
        # No unscoped blanket delete was issued (count==0 returns early, but
        # the scoped count query must be the one that ran).
        assert "count(DISTINCT sn)" in cypher
        lists = [
            kw["path_allowlist"] for kw in _all_kwargs(gc) if "path_allowlist" in kw
        ]
        assert lists and all(pl == [] for pl in lists)


# ---------------------------------------------------------------------------
# GUARD — unscoped accepted wipe
# ---------------------------------------------------------------------------


class TestAcceptedWipeGuardPredicate:
    """Direct unit tests of the guard predicate."""

    def _guard(self, **kwargs):
        from imas_codex.cli.sn import _reject_unscoped_accepted_reset

        defaults = {
            "reset_to": "extracted",
            "include_accepted": True,
            "flat_focus": [],
            "dry_run": False,
            "retry_quarantined": False,
            "below_score": None,
            "since": None,
            "before": None,
            "tier": None,
        }
        defaults.update(kwargs)
        return _reject_unscoped_accepted_reset(**defaults)

    def test_unscoped_accepted_reset_raises(self) -> None:
        with pytest.raises(click.UsageError):
            self._guard()

    def test_focus_scope_is_allowed(self) -> None:
        # Must not raise.
        self._guard(flat_focus=["eq/a"])

    def test_narrowing_filter_is_allowed(self) -> None:
        # The documented --retry-quarantined migration stays legal.
        self._guard(retry_quarantined=True)
        self._guard(below_score=0.6)
        self._guard(since="2026-04-19T10:00")
        self._guard(before="2026-05-01")
        self._guard(tier="poor")

    def test_dry_run_preview_is_allowed(self) -> None:
        # A dry-run touches nothing, so previewing the (unscoped) scope is ok.
        self._guard(dry_run=True)

    def test_no_include_accepted_is_allowed(self) -> None:
        self._guard(include_accepted=False)

    def test_no_reset_to_is_allowed(self) -> None:
        self._guard(reset_to=None)


class TestAcceptedWipeGuardCli:
    """End-to-end proof through the Click command (no graph reached)."""

    def test_reset_to_include_accepted_no_focus_raises(self) -> None:
        from imas_codex.cli.sn import sn

        runner = CliRunner()
        # --skip-clear-gate keeps the pipeline-version gate (which touches the
        # graph) from running; the guard fires before any graph access.
        result = runner.invoke(
            sn,
            [
                "run",
                "--reset-to",
                "extracted",
                "--include-accepted",
                "--skip-clear-gate",
            ],
        )
        assert result.exit_code != 0
        assert "include-accepted" in result.output.lower()
