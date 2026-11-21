"""Signed retirement uses an exhausted identity's own former-source evidence."""

from __future__ import annotations

import pytest

from iter_codex.graph.client import GraphClient
from iter_codex.standard_names.graph_store.graph_supersession import (
    ExhaustedOrphanSupersessionConflict,
    supersede_exhausted_standard_name_orphans,
)

pytestmark = pytest.mark.graph


@pytest.fixture()
def graph(disposable_neo4j_empty):
    with GraphClient(uri=disposable_neo4j_empty, username="neo4j", password="") as gc:
        yield gc


def _plant(graph, *, stage="exhausted", evidence="source_path"):
    graph.query(
        "CREATE (:StandardName {id:'retirement_subject', name_stage:$stage, "
        "status:'draft', origin:'pipeline', description:'Preserved description', "
        "llm_cost:0.25})",
        stage=stage,
    )
    if evidence in {"source_path", "source_paths", "detachment"}:
        graph.query(
            "CREATE (:StandardNameSource {id:'dd:retirement/value', "
            "source_id:'retirement/value', status:'extracted', attempt_count:0})"
        )
        if evidence == "detachment":
            graph.query(
                "CREATE (:StandardNameChange {id:'former_binding', "
                "from_name:'retirement/value', to_name:'retirement_subject', "
                "operation:'detach_inconsistent_attachment'})"
            )
        else:
            graph.query(
                "MATCH (n:StandardName {id:'retirement_subject'}) "
                "SET n.source_path=$path, n.source_paths=$paths",
                path="retirement/value" if evidence == "source_path" else None,
                paths=["dd:retirement/value"] if evidence == "source_paths" else [],
            )
    elif evidence == "migration":
        graph.query(
            "MATCH (n:StandardName {id:'retirement_subject'}) "
            "CREATE (n)-[:HAS_INTERNAL_CHANGE]->(:StandardNameChange {"
            "id:'former_binding', from_name:'retirement_predecessor', "
            "to_name:n.id, operation:'source_migration_manifest'})"
        )


@pytest.mark.parametrize(
    "evidence", ["source_path", "source_paths", "detachment", "migration"]
)
def test_signed_path_retires_exhausted_name_with_former_source(graph, evidence):
    _plant(graph, evidence=evidence)
    reason = "Retire an exhausted identity after its source returned to the pool"
    preview = supersede_exhausted_standard_name_orphans(
        ["retirement_subject"], reason=reason, gc=graph
    )
    assert preview["outcome"] == "would_apply", preview
    assert preview["counts"] == {"requested": 1, "admitted": 1, "refused": 0}
    authority = preview["manifest"]["participants"][0]["former_source_evidence"]
    if evidence == "migration":
        assert authority == {
            "released_sources": [],
            "migration_receipt_ids": ["former_binding"],
        }
    else:
        assert [row["id"] for row in authority["released_sources"]] == [
            "dd:retirement/value"
        ]
    assert graph.query("MATCH (n:StandardName) RETURN n.name_stage AS stage") == [
        {"stage": "exhausted"}
    ]
    applied = supersede_exhausted_standard_name_orphans(
        ["retirement_subject"],
        reason=reason,
        apply=True,
        manifest_sha256=preview["manifest_sha256"],
        gc=graph,
    )
    assert applied["outcome"] == "applied"
    assert applied["ledger_rows"] == 1
    assert graph.query(
        "MATCH (n:StandardName {id:'retirement_subject'}) "
        "RETURN n.name_stage AS stage, n.description AS description, n.llm_cost AS cost"
    ) == [{"stage": "superseded", "description": "Preserved description", "cost": 0.25}]
    assert graph.query(
        "MATCH (n:StandardName)-[:HAS_INTERNAL_CHANGE]->(c:StandardNameChange) "
        "WHERE c.operation='supersede_exhausted_orphan' "
        "RETURN c.manifest_sha256 AS digest"
    ) == [{"digest": preview["manifest_sha256"]}]
    assert (
        supersede_exhausted_standard_name_orphans(
            ["retirement_subject"],
            reason=reason,
            apply=True,
            manifest_sha256=preview["manifest_sha256"],
            gc=graph,
        )["outcome"]
        == "already_applied"
    )


@pytest.mark.parametrize(
    "stage", ["accepted", "refining", "contested", "drafted", "pending"]
)
@pytest.mark.parametrize("evidence", ["source_paths", "migration"])
def test_former_source_never_admits_non_exhausted_name(graph, stage, evidence):
    _plant(graph, stage=stage, evidence=evidence)
    preview = supersede_exhausted_standard_name_orphans(
        ["retirement_subject"], reason="Exhaustion is required", gc=graph
    )
    assert preview["outcome"] == "refused"
    assert preview["counts"]["admitted"] == 0
    assert "not 'exhausted'" in preview["refusals"][0]["reason"]


@pytest.mark.parametrize(
    "protection",
    ["no_record", "derived", "held_source", "unreleased", "missing_source", "claimed"],
)
def test_other_source_loss_is_still_refused(graph, protection):
    _plant(graph, evidence="none" if protection == "no_record" else "source_paths")
    if protection == "derived":
        graph.query("MATCH (n:StandardName) SET n.origin='derived'")
    elif protection == "held_source":
        graph.query(
            "MATCH (s:StandardNameSource) CREATE (holder:StandardName {"
            "id:'retirement_unrelated', name_stage:'accepted', status:'draft'}) "
            "CREATE (s)-[:PRODUCED_NAME]->(holder) SET s.status='composed'"
        )
    elif protection == "unreleased":
        graph.query("MATCH (s:StandardNameSource) SET s.status='composed'")
    elif protection == "missing_source":
        graph.query("MATCH (s:StandardNameSource) DELETE s")
    elif protection == "claimed":
        graph.query("MATCH (s:StandardNameSource) SET s.claim_token='compose-owner'")
    preview = supersede_exhausted_standard_name_orphans(
        ["retirement_subject"], reason="Require former source authority", gc=graph
    )
    assert preview["outcome"] == "refused"
    assert preview["counts"]["admitted"] == 0


def test_signed_former_source_authority_detects_drift(graph):
    _plant(graph)
    preview = supersede_exhausted_standard_name_orphans(
        ["retirement_subject"], reason="Retire exhausted source loss", gc=graph
    )
    assert preview["outcome"] == "would_apply"
    graph.query("MATCH (s:StandardNameSource) SET s.status='composed'")
    with pytest.raises(ExhaustedOrphanSupersessionConflict, match="signed hash"):
        supersede_exhausted_standard_name_orphans(
            ["retirement_subject"],
            reason="Retire exhausted source loss",
            apply=True,
            manifest_sha256=preview["manifest_sha256"],
            gc=graph,
        )
