"""``clear_standard_names`` deletes reviews + nodes in a SINGLE statement.

The delete used to run as separate transactions (delete reviews, delete
producer edges, delete orphaned nodes). A failure between them could strip a
name's producer edge without deleting the node — stranding the name out of
every future regeneration — or delete a survivor's review while the node
lived on. This suite pins the atomic shape: one statement that removes the
in-scope producer edges, deletes only the names left with no remaining
producer (orphan-guard), and deletes each deleted name's review in the same
statement (survivor-review handling).

All graph interaction is mocked (no live Neo4j), mirroring the established
mock-GraphClient pattern in ``test_reset_focus_scoping``.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch


def _make_gc(count: int = 5) -> MagicMock:
    gc = MagicMock()
    gc.__enter__.return_value = gc
    gc.__exit__.return_value = None

    def _query(cypher: str, **_kwargs):
        if "count(" in cypher:
            return [{"n": count}]
        return []

    gc.query = MagicMock(side_effect=_query)
    return gc


def _call(gc: MagicMock, **kwargs):
    from imas_codex.standard_names import graph_ops

    with patch.object(graph_ops, "GraphClient", return_value=gc):
        return graph_ops.clear_standard_names(**kwargs)


def _delete_sn_queries(gc: MagicMock) -> list[str]:
    """Every query that ends up deleting a StandardName node."""
    return [c.args[0] for c in gc.query.call_args_list if "DETACH DELETE sn" in c.args[0]]


class TestScopedClearIsAtomic:
    """A scoped (path-allowlist) clear deletes edge + review + node in one query."""

    def test_single_statement_deletes_edge_review_and_node(self) -> None:
        gc = _make_gc()
        _call(gc, path_allowlist=["eq/a"])

        deletes = _delete_sn_queries(gc)
        assert len(deletes) == 1, (
            "the scoped node delete must be a single atomic statement, not "
            f"a multi-query sequence:\n{deletes}"
        )
        q = deletes[0]
        # producer-edge removal, review removal, and node removal all in ONE query
        assert "DELETE rel" in q
        assert "-[:HAS_REVIEW]->(r:StandardNameReview)" in q
        assert "DETACH DELETE r" in q
        assert "DETACH DELETE sn" in q

    def test_orphan_guard_excludes_protected_nodes(self) -> None:
        """The delete stays gated by the orphan-guard so a name still attached
        to an out-of-scope producer path (a live producer) is never wiped."""
        gc = _make_gc()
        _call(gc, path_allowlist=["eq/a"])

        q = _delete_sn_queries(gc)[0]
        assert "NOT EXISTS { MATCH ()-[:HAS_STANDARD_NAME]->(sn) }" in q, (
            f"scoped clear dropped the orphan-guard:\n{q}"
        )

    def test_no_standalone_review_delete_before_node_delete(self) -> None:
        """Survivor-review handling: reviews are NOT pre-deleted in a separate
        pass keyed on the in-scope SN (that would delete a survivor's review).
        The only review delete is inside the node-delete statement."""
        gc = _make_gc()
        _call(gc, path_allowlist=["eq/a"])

        review_deletes = [
            c.args[0]
            for c in gc.query.call_args_list
            if "StandardNameReview" in c.args[0] and "DETACH DELETE r" in c.args[0]
        ]
        # Sweep-orphans (Step C) still runs; distinguish it — it matches on
        # StandardNameReview with a NOT EXISTS parent guard, not on HAS_REVIEW.
        keyed_on_sn = [
            q for q in review_deletes if "-[:HAS_REVIEW]->(r:StandardNameReview)" in q
        ]
        assert len(keyed_on_sn) == 1, (
            "review deletion keyed on the SN must happen only inside the "
            f"atomic node-delete statement:\n{keyed_on_sn}"
        )
        assert "DETACH DELETE sn" in keyed_on_sn[0]


class TestUnscopedClearIsAtomic:
    """An unscoped clear also deletes review + node in one statement."""

    def test_single_statement_deletes_review_and_node(self) -> None:
        gc = _make_gc()
        _call(gc)

        deletes = _delete_sn_queries(gc)
        assert len(deletes) == 1, (
            f"unscoped node delete must be a single statement:\n{deletes}"
        )
        q = deletes[0]
        assert "-[:HAS_REVIEW]->(r:StandardNameReview)" in q
        assert "DETACH DELETE r" in q
        assert "DETACH DELETE sn" in q

    def test_unscoped_has_no_orphan_guard(self) -> None:
        """Without a scope the delete is a blanket delete of every matching
        stage — the orphan-guard only applies to the relationship-first
        scoped path."""
        gc = _make_gc()
        _call(gc)
        q = _delete_sn_queries(gc)[0]
        assert "NOT EXISTS { MATCH ()-[:HAS_STANDARD_NAME]->(sn) }" not in q
