"""Source drift requeues accepted documentation within the active run."""

from __future__ import annotations

import asyncio
import os
from collections.abc import Iterator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from imas_codex.graph.client import GraphClient
from imas_codex.standard_names import graph_ops, loop, source_refresh
from imas_codex.standard_names.graph_store import (
    graph_docs_persistence as _owning_graph_docs_persistence,
)


class _RefreshGraph:
    """Retain the name fields written by the refresh and its edit helper."""

    def __init__(self) -> None:
        self.name = {
            "id": "plasma_current",
            "name_stage": "accepted",
            "docs_stage": "accepted",
        }

    def query(self, cypher: str, **params: object) -> list[dict[str, object]]:
        if "SET sn.run_id = $scope_run_id" in cypher:
            self.name["run_id"] = params["scope_run_id"]
        return []


@pytest.mark.asyncio
async def test_drift_refresh_is_claimable_by_docs_pool_in_same_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = _RefreshGraph()
    active_run = "active-run"
    reason = source_refresh._format_reason(
        graph.name["id"],
        [{"field": "documentation", "old": "Old source.", "new": "New source."}],
    )
    drift = {
        "sn_id": graph.name["id"],
        "name_stage": "accepted",
        "docs_stage": "accepted",
        "new_path": "magnetics/ip/0",
        "resolved_dd_context": {},
        "renamed": False,
        "requires_steering": True,
        "deltas": [
            {"field": "documentation", "old": "Old source.", "new": "New source."}
        ],
    }

    monkeypatch.setattr(source_refresh, "stamp_source_snapshots", lambda *a, **k: 1)
    monkeypatch.setattr(source_refresh, "detect_source_drift", lambda **k: [drift])

    def apply_edit(**kwargs: object) -> MagicMock:
        graph.name.update(
            docs_stage="pending",
            edit_reason=kwargs["reason"],
            edit_requested_at="2026-09-02T15:00:00+00:00",
            run_id="private-edit-run",
        )
        return MagicMock(blocked=None)

    monkeypatch.setattr("imas_codex.standard_names.edit.apply_edit", apply_edit)

    refreshed = source_refresh.refresh_drifted_sources(
        scope_run_id=active_run,
        gc=graph,
    )

    assert refreshed["steered"] == 1
    assert graph.name["docs_stage"] == "pending"
    assert graph.name["edit_reason"] == reason
    assert graph.name["edit_requested_at"]
    assert graph.name["run_id"] == active_run

    claim_calls: list[str | None] = []

    def claim_generate_docs_batch(**kwargs: object) -> list[dict[str, object]]:
        scope = kwargs.get("scope_run_id")
        claim_calls.append(scope if isinstance(scope, str) else None)
        if (
            scope == graph.name["run_id"]
            and graph.name["name_stage"] == "accepted"
            and graph.name["docs_stage"] == "pending"
        ):
            return [dict(graph.name, claim_token="claim-token")]
        return []

    with (
        patch(
            "imas_codex.standard_names.graph_store.graph_docs_persistence.claim_generate_docs_batch",
            side_effect=claim_generate_docs_batch,
        ),
        patch("imas_codex.settings.get_pool_replicas", return_value=1),
        patch(
            "imas_codex.standard_names.worker_tasks.worker_generate_docs.process_generate_docs_batch",
            new_callable=AsyncMock,
        ),
    ):
        specs = loop._build_pool_specs(
            MagicMock(),
            asyncio.Event(),
            drift_scope_run_id=active_run,
            docs_only=True,
        )
        generate_docs = next(spec for spec in specs if spec.name == "generate_docs")
        claimed = await generate_docs.claim()

    assert claim_calls == [active_run]
    assert claimed is not None
    assert claimed["items"][0]["id"] == graph.name["id"]


@pytest.fixture
def sourced_name(
    disposable_neo4j_uri: str, monkeypatch: pytest.MonkeyPatch
) -> Iterator[GraphClient]:
    assert os.environ["IMAS_CODEX_TEST_NEO4J_EPHEMERAL"] == "1"

    def resolved(
        path: str, unit: str | None, documentation: str | None
    ) -> dict[str, object]:
        return {
            "unit": unit,
            "documentation": documentation,
            "published_dd_context": {"unit": unit, "documentation": documentation},
            "dd_resolution_ids": [],
            "dd_resolution_converged_ids": [],
            "dd_resolution_manifest_digest": "test-authority",
            "_dd_resolution_marker": "test-context",
        }

    monkeypatch.setattr(source_refresh, "_resolved_source_context", resolved)
    with GraphClient() as graph:
        assert graph.uri == disposable_neo4j_uri
        graph.query("MATCH (n) DETACH DELETE n")
        graph.query("""
            CREATE (o:IMASNode {id: 'magnetics/ip/0', unit: 'A',
                                documentation: 'Original source.'})
            CREATE (src:StandardNameSource {id: 'source', source_id: o.id})
            CREATE (sn:StandardName {id: 'plasma_current',
                name_stage: 'accepted', docs_stage: 'pending',
                claim_token: 'claim', docs_chain_length: 0,
                description: 'Original description.',
                documentation: 'Original documentation.'})
            CREATE (src)-[:PRODUCED_NAME]->(sn)
        """)
        assert (
            source_refresh.stamp_source_snapshots(sn_ids=["plasma_current"], gc=graph)
            == 1
        )
        graph.query("""
            MATCH (o:IMASNode {id: 'magnetics/ip/0'})
            SET o.documentation = 'Regeneration source.'
        """)
        assert (
            len(source_refresh.detect_source_drift(sn_ids=["plasma_current"], gc=graph))
            == 1
        )
        yield graph


def _write_docs(kind: str, *, token: str = "claim") -> object:
    common = {
        "sn_id": "plasma_current",
        "claim_token": token,
        "description": "Regenerated description.",
        "documentation": "Regenerated documentation.",
        "model": "test-model",
    }
    if kind == "generated":
        return _owning_graph_docs_persistence.persist_generated_docs(**common)
    return _owning_graph_docs_persistence.persist_refined_docs(
        **common,
        current_description="Original description.",
        current_documentation="Original documentation.",
    )


@pytest.mark.graph
@pytest.mark.timeout(180)
@pytest.mark.parametrize("kind", ["generated", "refined"])
def test_docs_regeneration_tracks_source_and_refuses_stale_claim(
    sourced_name: GraphClient, kind: str
) -> None:
    if kind == "refined":
        sourced_name.query(
            "MATCH (sn:StandardName {id: 'plasma_current'}) SET sn.docs_stage = 'refining'"
        )
    _write_docs(kind)
    row = sourced_name.query("""
        MATCH (sn:StandardName {id: 'plasma_current'})
        RETURN sn.source_documentation AS snapshot
    """)[0]
    assert row["snapshot"] == "Regeneration source."
    assert (
        source_refresh.detect_source_drift(sn_ids=["plasma_current"], gc=sourced_name)
        == []
    )
    if kind == "refined":
        sourced_name.query(
            "MATCH (sn:StandardName {id: 'plasma_current'}) SET sn.docs_stage = 'refining'"
        )
        assert _write_docs(kind, token="stale") == {
            "docs_chain_length": -1,
            "revision_id": "",
        }
    else:
        with pytest.raises(ValueError, match="token mismatch"):
            _write_docs(kind, token="stale")
    row = sourced_name.query("""
        MATCH (sn:StandardName {id: 'plasma_current'})
        RETURN sn.source_documentation AS snapshot, sn.claim_token AS token
    """)[0]
    assert row == {"snapshot": "Regeneration source.", "token": None}
    sourced_name.query("""
        MATCH (o:IMASNode {id: 'magnetics/ip/0'})
        SET o.documentation = 'Later source.'
    """)
    drift = source_refresh.detect_source_drift(
        sn_ids=["plasma_current"], gc=sourced_name
    )
    assert len(drift) == 1
    assert {delta["field"] for delta in drift[0]["deltas"]} >= {"documentation"}
