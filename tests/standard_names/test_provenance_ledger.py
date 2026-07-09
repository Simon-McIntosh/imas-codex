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

from unittest.mock import MagicMock, patch

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


# ---------------------------------------------------------------------------
# reattach_produced_name_edges — heal live scalar/missing-edge desyncs
# ---------------------------------------------------------------------------


def test_reattach_produced_name_edges_merges_missing_edge_for_live_scalar():
    """The reattach pass MERGEs a PRODUCED_NAME edge for every source whose
    ``produced_sn_id`` names a LIVE name but whose edge is missing.
    """
    from imas_codex.standard_names.ledger import reattach_produced_name_edges

    gc = MagicMock()
    gc.query.return_value = [{"reattached": 3}]
    n = reattach_produced_name_edges(gc=gc)

    assert n == 3
    flat = " ".join(_captured_query(gc).split())
    assert "MERGE (sns)-[:PRODUCED_NAME]->(sn)" in flat
    # only for live scalars whose edge is currently missing
    assert "'superseded'" in flat and "'exhausted'" in flat
    assert "NOT (sns)-[:PRODUCED_NAME]->(sn)" in flat


# ---------------------------------------------------------------------------
# reconcile_provenance — now also reattaches (3rd pass)
# ---------------------------------------------------------------------------


def test_reconcile_provenance_reports_edges_reattached():
    """reconcile_provenance runs the reattach pass and surfaces its count
    alongside the existing scalar-clear / orphan-delete counts.
    """
    from imas_codex.standard_names import graph_ops

    with (
        patch("imas_codex.standard_names.graph_ops.GraphClient") as MockGC,
        patch(
            "imas_codex.standard_names.graph_ops.reattach_produced_name_edges",
            return_value=7,
        ) as m_reattach,
    ):
        mock_gc = MockGC.return_value.__enter__.return_value
        # existing two passes: scalars_cleared=2, orphan_sources_deleted=1
        mock_gc.query.side_effect = [[{"n": 2}], [{"n": 1}]]
        result = graph_ops.reconcile_provenance()

    assert m_reattach.called
    assert result["edges_reattached"] == 7
    assert result["scalars_cleared"] == 2
    assert result["orphan_sources_deleted"] == 1


# ---------------------------------------------------------------------------
# Live ledger invariants (read-only) — the acceptance proof for the rebuild.
# These run against the live graph and MUST hold: 0 orphans, 0 desyncs.
# They are read-only (no mutation) and so are safe on the shared graph.
# ---------------------------------------------------------------------------


@pytest.mark.graph
def test_live_ledger_has_no_provenance_orphans():
    from imas_codex.standard_names.ledger import find_provenance_orphans

    orphans = find_provenance_orphans()
    assert orphans == [], (
        f"{len(orphans)} live name(s) have no PRODUCED_NAME source — the ledger "
        f"is incomplete. Run 'sn rebuild-provenance'. First: "
        f"{[o['sn_id'] for o in orphans[:20]]}"
    )


@pytest.mark.graph
def test_live_ledger_has_no_edge_scalar_desyncs():
    from imas_codex.standard_names.ledger import find_edge_scalar_desyncs

    desyncs = find_edge_scalar_desyncs()
    assert desyncs == [], (
        f"{len(desyncs)} source(s) name a live standard name via produced_sn_id "
        f"but have no PRODUCED_NAME edge. Run 'sn run --only reconcile'. First: "
        f"{[d['source_id'] for d in desyncs[:20]]}"
    )
