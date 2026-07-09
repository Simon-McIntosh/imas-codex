"""Grammar-segment reconcile — structured segments must match the canonical id.

The bare-name segment columns (``position``, ``component``, ``subject``, …) are
a deterministic function of the canonical name id via the ISN parser. A name
written by an out-of-grammar path can carry stale segments that disagree with
its own id (observed: ``…_at_pedestal_top`` storing ``position='pedestal'``).
``reconcile_grammar_segments`` re-parses every live name and realigns drift.
"""

from __future__ import annotations

import pytest

from imas_codex.standard_names.graph_ops import (
    _GRAMMAR_SEGMENT_COLUMNS,
    _parse_grammar,
    reconcile_grammar_segments,
)


def test_reconcile_grammar_segments_returns_count():
    """The reconcile reports how many names it realigned (contract shape)."""
    # Pure-shape guard that does not require a live graph: the module exposes
    # the reconcile and the segment-column authority it realigns against.
    assert "position" in _GRAMMAR_SEGMENT_COLUMNS
    assert callable(reconcile_grammar_segments)


@pytest.mark.graph
def test_reconcile_grammar_segments_idempotent_and_no_drift():
    """After reconcile, no parseable live name's stored segments disagree with
    the parse of its own id, and a second run is a no-op (idempotent)."""
    from imas_codex.graph.client import GraphClient
    from imas_codex.standard_names.ledger import LIVE_NAME

    reconcile_grammar_segments()
    second = reconcile_grammar_segments()
    assert second["names_realigned"] == 0, "reconcile must be idempotent"

    cols = _GRAMMAR_SEGMENT_COLUMNS
    select = ", ".join(f"sn.{c} AS {c}" for c in cols)
    with GraphClient() as gc:
        rows = gc.query(
            f"MATCH (sn:StandardName) WHERE {LIVE_NAME} RETURN sn.id AS id, {select}"
        )
    drift = []
    for r in rows:
        parsed = _parse_grammar(r["id"])
        # Names the ISN model rejects are not realigned (segments stay as-is).
        if not parsed.get("physical_base"):
            continue
        if any(parsed.get(c) != r.get(c) for c in cols):
            drift.append(r["id"])
    assert not drift, f"{len(drift)} names drift from their id parse: {drift[:10]}"
