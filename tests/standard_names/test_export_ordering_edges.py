"""Regression: ordering-edge queries treat physics_domain as a scalar.

``physics_domain`` is a scalar string on StandardName nodes (the full list
lives on ``source_domains``).  The ordering-edge queries used
``$domain IN s.physics_domain``, which in Cypher expects a list on the
right-hand side — against a scalar string it yields null, so every edge was
silently dropped and each domain file fell back to alphabetic order instead
of the intended Kahn topological order.  The queries must use scalar
equality (``s.physics_domain = $domain``).
"""

from __future__ import annotations

from unittest.mock import MagicMock

from imas_codex.standard_names.export import _fetch_ordering_edges_for_domain


def _capturing_gc(rows_by_call: list[list[dict]]) -> tuple[MagicMock, list[str]]:
    """Return a mock gc whose query() records each cypher and returns queued rows."""
    captured: list[str] = []
    calls = {"n": 0}

    def _query(cypher: str, **kwargs) -> list[dict]:
        captured.append(cypher)
        i = calls["n"]
        calls["n"] += 1
        return rows_by_call[i] if i < len(rows_by_call) else []

    gc = MagicMock()
    gc.query = _query
    return gc, captured


class TestOrderingEdgeQueryContract:
    def test_queries_use_scalar_equality_not_list_membership(self) -> None:
        gc, captured = _capturing_gc([[], [], [], []])
        _fetch_ordering_edges_for_domain(gc, "equilibrium", {"a", "b"})

        assert len(captured) == 4, "expected 4 ordering-edge queries"
        joined = "\n".join(captured)
        assert "IN s.physics_domain" not in joined, (
            "list-membership on scalar physics_domain still present:\n" + joined
        )
        assert "IN t.physics_domain" not in joined, (
            "list-membership on scalar physics_domain still present:\n" + joined
        )
        assert "s.physics_domain = $domain" in joined
        assert "t.physics_domain = $domain" in joined


class TestOrderingEdgeRoundTrip:
    def test_in_domain_edges_returned(self) -> None:
        """A mock returning in-domain rows yields HAS_PARENT/HAS_ERROR edges."""
        rows = [
            [{"src": "child", "tgt": "parent"}],  # HAS_PARENT in-domain
            [{"src": "err", "tgt": "base"}],  # HAS_ERROR in-domain
            [],  # cross-domain parent
            [],  # cross-domain error
        ]
        gc, _ = _capturing_gc(rows)
        edges, cross = _fetch_ordering_edges_for_domain(
            gc, "equilibrium", {"child", "parent", "err", "base"}
        )
        assert ("child", "parent", "HAS_PARENT") in edges
        assert ("err", "base", "HAS_ERROR") in edges
        assert cross == set()

    def test_cross_domain_parent_collected(self) -> None:
        rows = [
            [],  # HAS_PARENT in-domain
            [],  # HAS_ERROR in-domain
            [{"name": "orphan"}],  # cross-domain HAS_PARENT source
            [],  # cross-domain HAS_ERROR target
        ]
        gc, _ = _capturing_gc(rows)
        edges, cross = _fetch_ordering_edges_for_domain(
            gc, "equilibrium", {"orphan"}
        )
        assert edges == []
        assert cross == {"orphan"}

    def test_edges_outside_entry_set_are_dropped(self) -> None:
        """Edges whose endpoints aren't in the export set are filtered out."""
        rows = [
            [{"src": "child", "tgt": "not_exported"}],
            [],
            [],
            [],
        ]
        gc, _ = _capturing_gc(rows)
        edges, cross = _fetch_ordering_edges_for_domain(gc, "equilibrium", {"child"})
        assert edges == []
