"""A single source-loss classification drives release, retirement and the probe."""

from __future__ import annotations

import asyncio
import os
from collections.abc import Iterator
from contextlib import nullcontext
from unittest.mock import patch

import pytest

from imas_codex.graph.client import GraphClient
from imas_codex.settings import get_graph_uri
from imas_codex.standard_names.graph_ops import (
    classify_sourceless_standard_names,
    supersede_exhausted_standard_name_orphans,
)
from imas_codex.standard_names.ledger import find_provenance_orphans
from imas_codex.standard_names.loop import _report_source_loss_probe


@pytest.fixture()
def planted_graph() -> Iterator[GraphClient]:
    """Plant six audit subjects and their source-carrying support nodes."""
    uri = os.environ.get("IMAS_CODEX_TEST_NEO4J_URI")
    if not uri:
        pytest.skip("a disposable Neo4j URI is required")
    if os.environ.get("IMAS_CODEX_TEST_NEO4J_EPHEMERAL") != "1":
        pytest.fail("source audit fixture requires a disposable graph")
    if uri == (os.environ.get("IMAS_CODEX_TEST_PROJECT_NEO4J_URI") or get_graph_uri()):
        pytest.fail("source audit fixture refuses the project graph")
    client = GraphClient(
        uri=uri,
        username="neo4j",
        password=os.environ.get("IMAS_CODEX_TEST_NEO4J_PASSWORD", ""),
        graph_name="source-audit-fixture",
    )
    client.query(
        """UNWIND [
          {id:'audit_dropped', stage:'exhausted', origin:'pipeline',
           paths:['dd:audit/dropped']},
          {id:'audit_never', stage:'exhausted', origin:'pipeline', paths:[]},
          {id:'audit_parent', stage:'accepted', origin:'derived', paths:[]},
          {id:'audit_orphan', stage:'exhausted', origin:'pipeline',
           paths:['dd:audit/orphan']},
          {id:'audit_unchained', stage:'exhausted', origin:'pipeline',
           paths:['dd:audit/unchained']},
          {id:'audit_sourced', stage:'accepted', origin:'pipeline', paths:[]},
          {id:'audit_child', stage:'accepted', origin:'pipeline', paths:[]},
          {id:'audit_holder', stage:'accepted', origin:'pipeline', paths:[]}
        ] AS row
        CREATE (:StandardName {id:row.id, name_stage:row.stage,
          origin:row.origin, source_paths:row.paths, status:'draft'})"""
    )
    client.query(
        """UNWIND [
          {id:'dd:audit/dropped', holder:null},
          {id:'dd:audit/orphan', holder:'audit_sourced'},
          {id:'dd:audit/unchained', holder:'audit_holder'},
          {id:'dd:audit/child', holder:'audit_child'}
        ] AS row
        CREATE (source:StandardNameSource {id:row.id, source_id:row.id})
        WITH source, row
        OPTIONAL MATCH (holder:StandardName {id:row.holder})
        FOREACH (_ IN CASE WHEN holder IS NULL THEN [] ELSE [1] END |
          CREATE (source)-[:PRODUCED_NAME]->(holder))"""
    )
    client.query(
        """MATCH (parent:StandardName {id:'audit_parent'}),
                     (child:StandardName {id:'audit_child'}),
                     (orphan:StandardName {id:'audit_orphan'}),
                     (successor:StandardName {id:'audit_sourced'})
        CREATE (child)-[:HAS_PARENT]->(parent)
        CREATE (successor)-[:REFINED_FROM]->(orphan)"""
    )
    try:
        yield client
    finally:
        client.query(
            "MATCH (node) WHERE node.id STARTS WITH 'audit_' "
            "OR node.id STARTS WITH 'dd:audit/' DETACH DELETE node"
        )
        client.close()


@pytest.mark.graph
def test_planted_names_follow_one_source_audit(planted_graph: GraphClient) -> None:
    rows = classify_sourceless_standard_names(planted_graph)
    populations = {row["id"]: row["population"] for row in rows}
    assert populations == {
        "audit_dropped": "dropped_edge",
        "audit_never": "never_bound",
        "audit_parent": "derived_parent",
        "audit_orphan": "orphan_awaiting_retirement",
        "audit_unchained": "unchained_holder",
    }
    assert "audit_sourced" not in populations
    assert planted_graph.query(
        """MATCH (name:StandardName)
        WHERE NOT EXISTS {(:StandardNameSource)-[:PRODUCED_NAME]->(name)}
        RETURN count(name) AS total"""
    ) == [{"total": len(rows)}]


@pytest.mark.graph
def test_derived_parent_requires_live_child_evidence(
    planted_graph: GraphClient,
) -> None:
    from imas_codex.standard_names.export import _fetch_export_population

    planted_graph.query(
        "MATCH (child:StandardName {id:'audit_child'}) SET child.status = 'deprecated'"
    )
    rows = classify_sourceless_standard_names(planted_graph)
    parent = next(row for row in rows if row["id"] == "audit_parent")
    assert parent["population"] == "never_bound"
    with patch(
        "imas_codex.graph.client.GraphClient", return_value=nullcontext(planted_graph)
    ):
        population = _fetch_export_population(require_docs_review=False)
    exported_parent = next(row for row in population if row["id"] == "audit_parent")
    assert exported_parent["_has_live_child"] is False


@pytest.mark.graph
@pytest.mark.parametrize(
    "operation",
    [
        "semantics_preserving_rename",
        "revert_rename",
        "backfill_refine",
        "campaign_docs_refine",
        "move_source_snapshot",
        "rebind_sources",
    ],
)
def test_recorded_identity_operations_link_transferred_sources(
    planted_graph: GraphClient, operation: str
) -> None:
    planted_graph.query(
        """MATCH (:StandardName {id:'audit_sourced'})
                  -[link:REFINED_FROM]->(:StandardName {id:'audit_orphan'})
        DELETE link"""
    )
    planted_graph.query(
        """CREATE (:StandardNameChange {id:'audit_change',
          from_name:'audit_orphan', to_name:'audit_sourced',
          operation:$operation})""",
        operation=operation,
    )
    rows = classify_sourceless_standard_names(planted_graph)
    orphan = next(row for row in rows if row["id"] == "audit_orphan")
    assert orphan["population"] == "orphan_awaiting_retirement"


@pytest.mark.graph
@pytest.mark.parametrize("skip_gate", [False, True])
def test_export_refuses_losses_before_population(
    planted_graph: GraphClient, tmp_path, skip_gate: bool
) -> None:
    from imas_codex.standard_names.export import run_export

    with (
        patch(
            "imas_codex.graph.client.GraphClient",
            return_value=nullcontext(planted_graph),
        ),
        patch(
            "imas_codex.standard_names.export._load_waived_source_paths",
            return_value=set(),
        ),
        patch("imas_codex.standard_names.export._fetch_export_population") as fetch,
    ):
        report = run_export(tmp_path, skip_gate=skip_gate, force=skip_gate)
    assert not report.all_gates_passed
    assert not fetch.called
    gate = next(gate for gate in report.gate_results if gate.gate == "source_audit")
    assert {issue["name"] for issue in gate.issues} == {"audit_dropped", "audit_never"}
    planted_graph.query(
        "MATCH (name:StandardName) WHERE name.id IN ['audit_dropped', 'audit_never'] "
        "DETACH DELETE name"
    )
    with (
        patch(
            "imas_codex.graph.client.GraphClient",
            return_value=nullcontext(planted_graph),
        ),
        patch(
            "imas_codex.standard_names.export._load_waived_source_paths",
            return_value=set(),
        ),
        patch("imas_codex.standard_names.export._fetch_export_population") as fetch,
    ):
        fetch.side_effect = RuntimeError("population reached")
        with pytest.raises(RuntimeError, match="population reached"):
            run_export(tmp_path, skip_gate=skip_gate, force=skip_gate)
    assert fetch.called


@pytest.mark.graph
def test_lifecycle_retires_only_recorded_chain_orphan(
    planted_graph: GraphClient,
) -> None:
    preview = supersede_exhausted_standard_name_orphans(
        reason="retire source-transferred predecessor", gc=planted_graph
    )
    assert preview["outcome"] == "would_apply"
    assert [action["name_id"] for action in preview["manifest"]["actions"]] == [
        "audit_orphan"
    ]
    applied = supersede_exhausted_standard_name_orphans(
        reason="retire source-transferred predecessor",
        apply=True,
        manifest_sha256=preview["manifest_sha256"],
        gc=planted_graph,
    )
    assert applied["outcome"] == "applied"
    assert applied["superseded"] == 1
    stages = {
        row["id"]: row["stage"]
        for row in planted_graph.query(
            "MATCH (name:StandardName) RETURN name.id AS id, name.name_stage AS stage"
        )
    }
    assert stages["audit_orphan"] == "superseded"
    for name in ("audit_dropped", "audit_never", "audit_unchained"):
        assert stages[name] == "exhausted"


@pytest.mark.graph
def test_probe_reports_exhausted_dropped_edge(planted_graph: GraphClient) -> None:
    orphans = find_provenance_orphans(gc=planted_graph)
    by_id = {row["sn_id"]: row["population"] for row in orphans}
    assert by_id == {
        "audit_dropped": "dropped_edge",
        "audit_never": "never_bound",
        "audit_unchained": "unchained_holder",
    }


def test_failed_probe_is_reported_as_not_run(caplog) -> None:
    def raise_read() -> None:
        raise RuntimeError("source graph unavailable")

    with caplog.at_level("WARNING"):
        result = asyncio.run(_report_source_loss_probe(raise_read))
    assert result is None
    assert "source-loss probe did not run: source graph unavailable" in caplog.text
