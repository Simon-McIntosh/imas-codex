"""Bare ``[name]`` bracket normalization — source-of-truth (accept-path) fix.

The docs LLM sometimes writes a related-name mention as a bare ``[name]``
bracket despite the prompt rule; Markdown renders these broken. Historically a
per-rotation reconcile swept them, but a doc written by a late in-flight task
could finalize with a bare bracket after the sweep had run — occasionally an
*accepted* doc carried one. The fix normalizes the node AT acceptance
(:func:`persist_reviewed_docs`) so no accepted doc can carry a bare bracket,
regardless of when the doc was written, with the post-drain reconcile retained
as a belt-and-suspenders net.

Covers:
- scoped ``_normalize_bare_doc_links(gc, sn_id=...)`` links live names and
  strips dead/unknown tokens (single-node, no full-catalogue scan)
- ``persist_reviewed_docs`` invokes the scoped normalize ONLY when the doc
  promotes to ``accepted`` (not on ``reviewed`` / ``exhausted``)
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

_GC_PATH = "imas_codex.standard_names.graph_ops.GraphClient"


class _RecordingGC:
    """Fake GraphClient capturing every ``query(cypher, **params)`` call.

    ``responses`` is consumed in order, one per ``query`` call. The captured
    ``calls`` list lets a test assert what was written.
    """

    def __init__(self, responses: list[list[dict]]):
        self._responses = list(responses)
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def query(self, cypher: str, **params: Any) -> list[dict]:
        self.calls.append((cypher, params))
        if self._responses:
            return self._responses.pop(0)
        return []


def test_scoped_normalize_links_live_and_strips_dead():
    """A bare [live] becomes a link; a bare [dead]/[unknown] is stripped."""
    from imas_codex.standard_names.graph_ops import _normalize_bare_doc_links

    doc = "Related to [poloidal_flux] and the legacy [dead_name] term."
    gc = _RecordingGC(
        responses=[
            # fetch the single node's doc
            [{"id": "area_at_plasma_boundary", "docs": doc}],
            # liveness of the candidate tokens — only poloidal_flux is live
            [{"id": "poloidal_flux"}],
            # the UNWIND write (result unused)
            [],
        ]
    )

    n = _normalize_bare_doc_links(gc, sn_id="area_at_plasma_boundary")

    assert n == 1
    # The liveness query must be scoped to ONLY the doc's tokens (no full scan).
    liveness_cypher, liveness_params = gc.calls[1]
    assert "s.id IN $toks" in liveness_cypher
    assert set(liveness_params["toks"]) == {"poloidal_flux", "dead_name"}
    # The write must link the live name and strip the dead one to plain text.
    write_cypher, write_params = gc.calls[2]
    written = write_params["items"][0]["doc"]
    assert "[poloidal_flux](name:poloidal_flux)" in written
    assert "[dead_name]" not in written
    assert "legacy dead_name term" in written


def test_scoped_normalize_noop_when_no_brackets():
    """A doc with no bare brackets produces no write."""
    from imas_codex.standard_names.graph_ops import _normalize_bare_doc_links

    gc = _RecordingGC(responses=[[]])  # fetch returns nothing (no '[' docs)
    assert _normalize_bare_doc_links(gc, sn_id="plasma_current") == 0


def _patch_persist_gc(gc):
    return patch(_GC_PATH, return_value=gc)


def test_persist_reviewed_docs_normalizes_on_accept():
    """On accept, persist_reviewed_docs invokes the scoped bare-link normalize."""
    import imas_codex.standard_names.graph_ops as g

    gc = MagicMock()
    gc.__enter__ = MagicMock(return_value=gc)
    gc.__exit__ = MagicMock(return_value=False)
    gc.query = MagicMock(
        side_effect=[
            [{"docs_chain_length": 0}],  # readback
            [{"id": "area_at_plasma_boundary"}],  # SET committed → accepted
        ]
    )

    with (
        _patch_persist_gc(gc),
        patch.object(g, "_normalize_bare_doc_links") as norm,
        patch.object(g, "write_reviews"),
        patch.object(g, "bump_sn_run_counter"),
    ):
        result = g.persist_reviewed_docs(
            sn_id="area_at_plasma_boundary",
            claim_token="tok-1",
            score=0.95,
            model="test/model",
            min_score=0.85,
            rotation_cap=3,
            skip_review_node=True,
        )

    assert result == "accepted"
    norm.assert_called_once()
    # scoped to this node
    _args, kwargs = norm.call_args
    assert kwargs.get("sn_id") == "area_at_plasma_boundary"


def test_persist_reviewed_docs_skips_normalize_when_not_accepted():
    """A below-threshold review (→ reviewed) does NOT run the accept normalize."""
    import imas_codex.standard_names.graph_ops as g

    gc = MagicMock()
    gc.__enter__ = MagicMock(return_value=gc)
    gc.__exit__ = MagicMock(return_value=False)
    gc.query = MagicMock(
        side_effect=[
            [{"docs_chain_length": 0}],  # readback (chain 0 < cap → reviewed)
            [{"id": "area_at_plasma_boundary"}],  # SET committed → reviewed
        ]
    )

    with (
        _patch_persist_gc(gc),
        patch.object(g, "_normalize_bare_doc_links") as norm,
        patch.object(g, "write_reviews"),
        patch.object(g, "bump_sn_run_counter"),
    ):
        result = g.persist_reviewed_docs(
            sn_id="area_at_plasma_boundary",
            claim_token="tok-1",
            score=0.40,
            model="test/model",
            min_score=0.85,
            rotation_cap=3,
            skip_review_node=True,
        )

    assert result == "reviewed"
    norm.assert_not_called()
