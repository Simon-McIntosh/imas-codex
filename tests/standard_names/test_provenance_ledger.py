"""Ledger-invariant tests: every live StandardName traces to >=1 source.

The graph is the provenance ledger. Two read-only invariant queries back the
gate and the rebuild:

- :func:`find_provenance_orphans` — live names (not superseded/exhausted) with
  no ``PRODUCED_NAME`` source, excluding deterministic error-siblings (which
  carry provenance on ``source_paths``, never a ``StandardNameSource``).
- :func:`find_edge_scalar_desyncs` — sources whose ``produced_sn_id`` scalar
  names a live name but whose ``PRODUCED_NAME`` edge is missing (reattachable).

The Cypher-shape tests are mock-based (no Neo4j). The ``graph``-marked tests
run READ-ONLY against the live graph and assert the ledger holds (0 orphans,
0 desyncs) — they are the acceptance proof for the one-time rebuild and must
be run, not excluded.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest


def _captured_query(gc: MagicMock) -> str:
    call = gc.query.call_args
    return call.args[0] if call.args else call.kwargs["query"]


# ---------------------------------------------------------------------------
# find_provenance_orphans — Cypher shape (mock-based)
# ---------------------------------------------------------------------------


def test_find_provenance_orphans_query_excludes_dead_and_error_siblings():
    """The orphan query must exclude superseded/exhausted names AND
    deterministic error-siblings, and select names with no PRODUCED_NAME.
    """
    from imas_codex.standard_names.ledger import find_provenance_orphans

    gc = MagicMock()
    gc.query.return_value = [
        {"sn_id": "orphan_a", "name_stage": "accepted", "origin": "catalog_edit"}
    ]
    orphans = find_provenance_orphans(gc=gc)

    assert orphans == [
        {"sn_id": "orphan_a", "name_stage": "accepted", "origin": "catalog_edit"}
    ]
    flat = " ".join(_captured_query(gc).split())
    # live = not superseded and not exhausted
    assert "'superseded'" in flat and "'exhausted'" in flat
    # error-siblings carry no StandardNameSource by design → excluded
    assert "deterministic:dd_error_modifier" in flat
    # selects names with NO incoming PRODUCED_NAME source
    assert "PRODUCED_NAME" in flat
    assert "NOT" in flat
