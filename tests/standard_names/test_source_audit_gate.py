"""A single source-loss classification drives release, retirement and the probe."""

from __future__ import annotations

import asyncio
import os
from collections.abc import Iterator
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from imas_codex.graph.client import GraphClient
from imas_codex.settings import get_graph_uri
from imas_codex.standard_names.graph_ops import (
    classify_sourceless_standard_names,
    supersede_exhausted_standard_name_orphans,
)
from imas_codex.standard_names.ledger import find_provenance_orphans
from imas_codex.standard_names.loop import _report_source_loss_probe
from imas_codex.standard_names.source_invariant import find_sourceless_live_names


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
def test_stale_binding_does_not_satisfy_writer_invariant(
    planted_graph: GraphClient,
) -> None:
    planted_graph.query(
        """CREATE (name:StandardName {id:'audit_stale',
          name_stage:'accepted', status:'draft', origin:'pipeline'})
        CREATE (source:StandardNameSource {id:'dd:audit/stale',
          source_id:'audit/stale', status:'stale'})
        CREATE (source)-[:PRODUCED_NAME]->(name)"""
    )
    assert find_sourceless_live_names(planted_graph, ["audit_stale"]) == ["audit_stale"]
    signed_count = """MATCH (source:StandardNameSource)-[:PRODUCED_NAME]->
                   (:StandardName {id:'audit_stale'})
        WHERE source.status <> 'stale'
        RETURN count(source) AS live_producers"""
    assert planted_graph.query(signed_count) == [{"live_producers": 0}]
    planted_graph.query(
        "MATCH (source:StandardNameSource {id:'dd:audit/stale'}) "
        "SET source.status = null"
    )
    assert find_sourceless_live_names(planted_graph, ["audit_stale"]) == ["audit_stale"]
    assert planted_graph.query(signed_count) == [{"live_producers": 0}]
    planted_graph.query(
        "MATCH (source:StandardNameSource {id:'dd:audit/stale'}) "
        "SET source.status = 'composed'"
    )
    assert find_sourceless_live_names(planted_graph, ["audit_stale"]) == []
    assert planted_graph.query(signed_count) == [{"live_producers": 1}]


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
def test_export_refuses_live_losses_before_population(
    planted_graph: GraphClient, tmp_path, skip_gate: bool
) -> None:
    from imas_codex.standard_names.export import run_export

    # A live-stage loss blocks the export; the exhausted loss stays dormant.
    planted_graph.query(
        "MATCH (name:StandardName {id:'audit_dropped'}) "
        "SET name.name_stage = 'accepted'"
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
        report = run_export(tmp_path, skip_gate=skip_gate, force=skip_gate)
    assert not report.all_gates_passed
    assert not fetch.called
    gate = next(gate for gate in report.gate_results if gate.gate == "source_audit")
    assert {issue["name"] for issue in gate.issues} == {"audit_dropped"}
    assert report.dormant_sourceless_names == [
        {
            "name": "audit_never",
            "name_stage": "exhausted",
            "population": "never_bound",
        }
    ]
    planted_graph.query(
        "MATCH (name:StandardName {id:'audit_dropped'}) DETACH DELETE name"
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


def _sourceless_row(name: str, stage: str, population: str = "dropped_edge") -> dict:
    """One classifier row for a live name that has no producing source."""
    return {
        "id": name,
        "name_stage": stage,
        "docs_stage": None,
        "validation_status": None,
        "model": None,
        "status": "draft",
        "origin": "pipeline",
        "population": population,
        "recorded_source_ids": [],
        "unresolved_tokens": [],
        "unchained_holders": [],
    }


def _candidate(name: str) -> dict:
    """An otherwise-eligible export candidate with a producing source."""
    return {
        "id": name,
        "name_stage": "accepted",
        "status": "draft",
        "validation_status": "valid",
        "_validation_observed_at": "2026-09-09T00:00:00Z",
        "review_quorum_shortfall": None,
        "docs_stage": "accepted",
        "_has_docs_review": True,
        "_has_winning_docs_review": True,
        "docs_review_quorum_shortfall": None,
        "reviewer_score_name": 0.95,
        "description": f"Description for {name}.",
        "documentation": f"Documentation for {name}.",
        "kind": "scalar",
        "unit": "1",
        "physics_domain": "general",
        "links": [],
        "_has_dd_source_binding": True,
        "_has_derived_producer": False,
        "_has_non_derived_producer": True,
        "_has_live_child": False,
        "_is_parent": False,
    }


class _EmptyGraphClient:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def query(self, cypher: str, **params):
        return []


def _run_export_with_audit(
    staging_dir: Path, population: list[dict], audit_rows: list[dict]
):
    """Run an export against a stubbed source audit and population."""
    from imas_codex.standard_names.export import run_export

    fetch = MagicMock(return_value=population)
    with (
        patch(
            "imas_codex.standard_names.graph_ops.classify_sourceless_standard_names",
            return_value=audit_rows,
        ),
        patch("imas_codex.standard_names.export._fetch_export_population", fetch),
        patch(
            "imas_codex.graph.client.GraphClient",
            return_value=_EmptyGraphClient(),
        ),
        patch(
            "imas_codex.standard_names.export._validate_entry",
            side_effect=lambda entry: entry,
        ),
        patch(
            "imas_codex.standard_names.export._fetch_deprecation_stubs",
            return_value=[],
        ),
        patch(
            "imas_codex.standard_names.export._fetch_ordering_edges_for_domain",
            return_value=([], set()),
        ),
        patch("imas_codex.standard_names.export._write_domain_yaml"),
    ):
        report = run_export(
            staging_dir,
            min_score=0.85,
            skip_gate=True,
            force=True,
            include_sources=False,
        )
    return report, fetch


def test_exhausted_sourceless_row_is_dormant_not_blocking(tmp_path: Path) -> None:
    """An exhausted name never ships, so its missing source is dormant."""
    report, fetch = _run_export_with_audit(
        tmp_path,
        [_candidate("live_sourced")],
        [_sourceless_row("dormant_exhausted", "exhausted")],
    )

    assert fetch.called
    assert report.all_gates_passed
    assert "source_audit" not in {gate.gate for gate in report.gate_results}
    assert report.to_dict()["dormant_sourceless"] == {
        "count": 1,
        "names": [
            {
                "name": "dormant_exhausted",
                "name_stage": "exhausted",
                "population": "dropped_edge",
            }
        ],
    }


def test_live_sourceless_row_blocks_the_export(tmp_path: Path) -> None:
    """A live name without a producing source still stops the export."""
    report, fetch = _run_export_with_audit(
        tmp_path, [], [_sourceless_row("live_loss", "accepted")]
    )

    assert not fetch.called
    assert not report.all_gates_passed
    gate = next(gate for gate in report.gate_results if gate.gate == "source_audit")
    assert {issue["name"] for issue in gate.issues} == {"live_loss"}
    assert gate.issues[0]["detail"] == "live standard name has no producing source"


def test_held_sourceless_identity_is_not_counted_and_not_exported(
    tmp_path: Path,
) -> None:
    """A held identity leaves through the hold path, not the source gate."""
    held = "electrostatic_potential_at_wall"
    report, fetch = _run_export_with_audit(
        tmp_path,
        [_candidate(held), _candidate("live_sourced")],
        [_sourceless_row(held, "accepted")],
    )

    assert fetch.called
    assert report.all_gates_passed
    assert "source_audit" not in {gate.gate for gate in report.gate_results}
    assert held not in report.exported_names
    assert report.to_dict()["dormant_sourceless"]["count"] == 0
    records = {record.standard_name_id: record for record in report.exclusion_records}
    assert records[held].reason == "release_hold_no_dd_source"
    assert records[held].stage == "release_authority"


def test_failed_probe_is_reported_as_not_run(caplog) -> None:
    def raise_read() -> None:
        raise RuntimeError("source graph unavailable")

    with caplog.at_level("WARNING"):
        result = asyncio.run(_report_source_loss_probe(raise_read))
    assert result is None
    assert "source-loss probe did not run: source graph unavailable" in caplog.text


def test_rebuild_excludes_exhausted_audit_findings_from_repair_and_retirement() -> None:
    import imas_codex.standard_names.provenance_rebuild as rebuild

    active = {"sn_id": "active_loss", "name_stage": "accepted", "origin": "pipeline"}
    exhausted = [
        {
            "sn_id": f"exhausted_{population}",
            "name_stage": "exhausted",
            "origin": "pipeline",
            "population": population,
        }
        for population in (
            "dropped_edge",
            "never_bound",
            "unchained_holder",
            "orphan_awaiting_retirement",
        )
    ]
    excluded = [
        {"sn_id": "scaffold", "name_stage": None, "origin": None},
        {
            "sn_id": "error_sibling",
            "name_stage": "accepted",
            "model": "deterministic:dd_error_modifier",
        },
    ]
    rows = [active, *exhausted, *excluded]
    gc = MagicMock()
    with (
        patch.object(
            rebuild,
            "find_provenance_orphans",
            side_effect=[rows, rows, rows, exhausted],
        ),
        patch.object(rebuild, "find_edge_scalar_desyncs", return_value=[]),
        patch.object(rebuild, "reattach_produced_name_edges", return_value=0),
        patch.object(rebuild, "_run_deterministic_fixpoints"),
        patch.object(rebuild, "find_orphan_parent_source_candidates", return_value=[]),
        patch.object(rebuild, "reconcile_orphan_parent_sources", return_value=0),
        patch.object(rebuild, "_fetch_dd_source_paths", return_value={}),
        patch.object(rebuild, "_fetch_change_history_sources", return_value={}),
        patch.object(rebuild, "_fetch_pending_source_names", return_value=set()),
        patch.object(rebuild, "bind_recovery_sources") as bind,
        patch.object(
            rebuild,
            "retire_unrecoverable_provenance_orphans",
            return_value=["active_loss"],
        ) as retire,
    ):
        summary = rebuild.rebuild_provenance(
            gc=gc,
            recovery_map={
                "exhausted_dropped_edge": [
                    {"id": "dd:example/path", "source_type": "dd"}
                ]
            },
            retire_unresolved=True,
        )
    assert summary["orphans_before"] == 1
    assert summary["retired_unresolved_names"] == ["active_loss"]
    assert summary["orphans_after"] == 0
    bind.assert_not_called()
    retire.assert_called_once_with(gc, ["active_loss"], include_accepted=False)
