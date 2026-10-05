"""Tests for the shared embed worker's pending-row predicate.

The worker, its own count helper and the code phase's progress query all
decide what the embed worker will fetch, so they must embed one rendered
Cypher fragment.  A row whose ``embed_failed_at`` is older than the retry
cutoff is fetched again; a fresher failure is left alone.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

from imas_codex.discovery.base.embed_worker import (
    DEFAULT_EMBED_RETRY_HOURS,
    _count_unembedded,
    _fetch_unembedded,
    _mark_embed_failed,
    embed_retry_cutoff_time,
    pending_embed_predicate,
)


def _row_is_pending(
    fragment: str,
    alias: str,
    embedding,
    embed_failed_at: datetime | None,
    cutoff: datetime,
) -> bool:
    """Evaluate the rendered pending fragment against one row.

    Mirrors the fragment's own clauses; derived from the fragment text so
    that a fragment missing the cutoff admission changes the verdict.
    """
    assert f"{alias}.embedding IS NULL" in fragment
    if embedding is not None:
        return False
    cutoff_clause = re.search(
        rf"{re.escape(alias)}\.embed_failed_at < datetime\(\$(\w+)\)", fragment
    )
    if embed_failed_at is None:
        return f"{alias}.embed_failed_at IS NULL" in fragment
    if cutoff_clause is None:
        return False
    return embed_failed_at < cutoff


def _fake_gc(rows, alias: str, fragment: str):
    """Build a GraphClient stand-in that filters rows by the fragment."""

    def query(sql, **params):
        assert fragment in sql, "fetch query does not embed the shared fragment"
        cutoff = datetime.fromisoformat(params["embed_retry_cutoff"])
        return [
            {"id": r["id"], "text": r.get("text", "text")}
            for r in rows
            if _row_is_pending(
                fragment, alias, r["embedding"], r["embed_failed_at"], cutoff
            )
        ]

    gc = MagicMock()
    gc.query.side_effect = query
    ctx = MagicMock()
    ctx.__enter__ = MagicMock(return_value=gc)
    ctx.__exit__ = MagicMock(return_value=False)
    return gc, ctx


def _rows() -> list[dict]:
    now = datetime.now(UTC)
    return [
        {
            "id": "stale",
            "text": "stale text",
            "embedding": None,
            "embed_failed_at": now - timedelta(hours=DEFAULT_EMBED_RETRY_HOURS + 24),
        },
        {
            "id": "fresh",
            "text": "fresh text",
            "embedding": None,
            "embed_failed_at": now - timedelta(hours=1),
        },
        {
            "id": "never",
            "text": "never failed",
            "embedding": None,
            "embed_failed_at": None,
        },
        {
            "id": "done",
            "text": "embedded",
            "embedding": [0.1, 0.2],
            "embed_failed_at": None,
        },
    ]


class TestPendingPredicate:
    def test_fragment_admits_never_failed(self):
        frag = pending_embed_predicate("n")
        assert "n.embedding IS NULL" in frag
        assert "n.embed_failed_at IS NULL" in frag
        assert "n.embed_failed_at < datetime($embed_retry_cutoff)" in frag

    def test_fragment_honours_custom_cutoff_param(self):
        frag = pending_embed_predicate("cc", "my_cutoff")
        assert "datetime($my_cutoff)" in frag
        assert "cc.embedding IS NULL" in frag


class TestFetchUnembedded:
    def test_stale_failed_row_is_fetched_and_fresh_row_is_not(self):
        """A row with an old embed_failed_at is fetched; a fresh one is not."""
        frag = pending_embed_predicate("n")
        _gc, ctx = _fake_gc(_rows(), "n", frag)
        with patch("imas_codex.graph.GraphClient", return_value=ctx):
            fetched = _fetch_unembedded("FakeLabel", None, 10)
        got = {r["id"] for r in fetched}
        assert "stale" in got
        assert "fresh" not in got
        assert "never" in got
        assert "done" not in got

    def test_fetch_query_embeds_shared_fragment(self):
        captured = {}

        def query(sql, **params):
            captured["sql"] = sql
            captured["params"] = params
            return []

        gc = MagicMock()
        gc.query.side_effect = query
        ctx = MagicMock()
        ctx.__enter__ = MagicMock(return_value=gc)
        ctx.__exit__ = MagicMock(return_value=False)
        with patch("imas_codex.graph.GraphClient", return_value=ctx):
            _fetch_unembedded("FakeLabel", None, 10)
        assert pending_embed_predicate("n") in captured["sql"]
        assert "embed_retry_cutoff" in captured["params"]


class TestCountUnembedded:
    def test_count_query_embeds_shared_fragment(self):
        def query(sql, **params):
            assert pending_embed_predicate("n") in sql
            assert "embed_retry_cutoff" in params
            return [{"total": 7}]

        gc = MagicMock()
        gc.query.side_effect = query
        ctx = MagicMock()
        ctx.__enter__ = MagicMock(return_value=gc)
        ctx.__exit__ = MagicMock(return_value=False)
        with patch("imas_codex.graph.GraphClient", return_value=ctx):
            total = _count_unembedded("FakeLabel", None)
        assert total == 7


class TestMarkEmbedFailed:
    def test_docstring_no_longer_instructs_manual_clear(self):
        doc = _mark_embed_failed.__doc__ or ""
        assert "manually" not in doc
        assert "retry" in doc.lower()

    def test_marks_property(self):
        mock_gc = MagicMock()
        ctx = MagicMock()
        ctx.__enter__ = MagicMock(return_value=mock_gc)
        ctx.__exit__ = MagicMock(return_value=False)
        with patch("imas_codex.graph.GraphClient", return_value=ctx):
            _mark_embed_failed("FakeLabel", ["a", "b"])
        sql = mock_gc.query.call_args[0][0]
        assert "embed_failed_at = datetime()" in sql


def test_cutoff_is_in_the_past():
    cutoff = datetime.fromisoformat(embed_retry_cutoff_time())
    assert cutoff < datetime.now(UTC)
