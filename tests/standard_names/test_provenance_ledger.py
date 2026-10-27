"""The source-loss probe consumes the shared graph classifier.

The scalar/edge desync query remains separate because it measures a source
whose recorded target lacks an edge. Live graph checks are read-only.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from imas_codex.standard_names.graph_store import graph_audits as _owning_graph_audits


def _captured_query(gc: MagicMock) -> str:
    call = gc.query.call_args
    return call.args[0] if call.args else call.kwargs["query"]


# ---------------------------------------------------------------------------
# find_provenance_orphans — Cypher shape (mock-based)
# ---------------------------------------------------------------------------


def test_find_provenance_orphans_reports_three_source_loss_populations():
    """The probe reports source loss at every stage and leaves orphans to retirement."""
    from imas_codex.standard_names.ledger import find_provenance_orphans

    gc = MagicMock()
    rows = [
        {"id": "dropped", "name_stage": "exhausted", "population": "dropped_edge"},
        {"id": "never", "name_stage": "accepted", "population": "never_bound"},
        {"id": "unchained", "name_stage": "pending", "population": "unchained_holder"},
        {"id": "parent", "name_stage": "accepted", "population": "derived_parent"},
        {
            "id": "retireable",
            "name_stage": "exhausted",
            "population": "orphan_awaiting_retirement",
        },
    ]
    with patch(
        "imas_codex.standard_names.graph_store.graph_supersession.classify_sourceless_standard_names",
        return_value=rows,
    ) as audit:
        orphans = find_provenance_orphans(gc=gc)

    audit.assert_called_once_with(gc)
    assert {
        (row["sn_id"], row["name_stage"], row["population"]) for row in orphans
    } == {
        ("dropped", "exhausted", "dropped_edge"),
        ("never", "accepted", "never_bound"),
        ("unchained", "pending", "unchained_holder"),
    }


@pytest.mark.parametrize(
    ("population", "stage", "expected_ids"),
    [
        ("dropped_edge", "exhausted", ["candidate"]),
        ("never_bound", "exhausted", ["candidate"]),
        ("unchained_holder", "exhausted", ["candidate"]),
        ("held_in_chain", "reviewed", []),
        ("orphan_awaiting_retirement", "exhausted", []),
        ("derived_parent", "accepted", []),
    ],
)
def test_find_provenance_orphans_uses_population_not_stage(
    population, stage, expected_ids
):
    from imas_codex.standard_names.ledger import find_provenance_orphans

    gc = MagicMock()
    with patch(
        "imas_codex.standard_names.graph_store.graph_supersession.classify_sourceless_standard_names",
        return_value=[
            {"id": "candidate", "name_stage": stage, "population": population}
        ],
    ):
        orphans = find_provenance_orphans(gc=gc)

    assert [item["sn_id"] for item in orphans] == expected_ids


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
        patch(
            "imas_codex.standard_names.graph_store.graph_foundation.GraphClient"
        ) as MockGC,
        patch(
            "imas_codex.standard_names.graph_store.graph_foundation.reattach_produced_name_edges",
            return_value=7,
        ) as m_reattach,
    ):
        mock_gc = MockGC.return_value.__enter__.return_value
        # existing two passes: scalars_cleared=2, orphan_sources_deleted=1
        mock_gc.query.side_effect = [[{"n": 2}], [{"n": 1}]]
        result = _owning_graph_audits.reconcile_provenance()

    assert m_reattach.called
    assert result["edges_reattached"] == 7
    assert result["scalars_cleared"] == 2
    assert result["orphan_sources_deleted"] == 1

    orphan_cleanup = mock_gc.query.call_args_list[-1].args[0]
    assert "sns.source_type = 'derived'" in orphan_cleanup
    assert "source_type IN ['derived', 'dd']" not in orphan_cleanup
    assert "MATCH (sns)-[:PRODUCED_NAME]->(:StandardName)" in orphan_cleanup
    assert "MATCH (sn:StandardName {id: sns.source_id})" in orphan_cleanup


# ---------------------------------------------------------------------------
# Live ledger views are read-only. Source loss remains reportable until repair.
# ---------------------------------------------------------------------------


@pytest.mark.graph
def test_live_source_loss_probe_matches_independent_count():
    from imas_codex.graph.client import GraphClient
    from imas_codex.standard_names.graph_store.graph_supersession import (
        classify_sourceless_standard_names,
    )
    from imas_codex.standard_names.ledger import find_provenance_orphans

    with GraphClient() as gc:
        sourced = gc.query(
            """MATCH (:StandardNameSource)-[:PRODUCED_NAME]->(name:StandardName)
            RETURN count(DISTINCT name) AS total"""
        )[0]["total"]
        count = gc.query(
            """MATCH (name:StandardName)
            WHERE NOT name.status IN ['deprecated', 'superseded']
              AND NOT EXISTS {
                (:StandardNameSource)-[:PRODUCED_NAME]->(name)
              }
            RETURN count(name) AS total"""
        )[0]["total"]
        classified = classify_sourceless_standard_names(gc)
        reported = find_provenance_orphans(gc=gc)
    assert sourced > 0
    assert len(classified) == count
    assert {row["sn_id"] for row in reported} == {
        row["id"]
        for row in classified
        if row["population"] in {"dropped_edge", "never_bound", "unchained_holder"}
    }


@pytest.mark.graph
def test_live_ledger_has_no_edge_scalar_desyncs():
    from imas_codex.standard_names.ledger import find_edge_scalar_desyncs

    desyncs = find_edge_scalar_desyncs()
    assert desyncs == [], (
        f"{len(desyncs)} source(s) name a live standard name via produced_sn_id "
        f"but have no PRODUCED_NAME edge. Run 'sn run --only reconcile'. First: "
        f"{[d['source_id'] for d in desyncs[:20]]}"
    )
