"""Disposable-graph coverage for structural provenance source reconciliation."""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest
from neo4j import GraphDatabase

from imas_codex.graph.client import GraphClient as RealGraphClient
from imas_codex.settings import get_graph_uri


@pytest.fixture(scope="module")
def disposable_neo4j() -> Iterator[tuple[str, str]]:
    uri = os.environ.get("IMAS_CODEX_TEST_NEO4J_URI")
    if not uri:
        pytest.skip("IMAS_CODEX_TEST_NEO4J_URI is not configured")
    if os.environ.get("IMAS_CODEX_TEST_NEO4J_EPHEMERAL") != "1":
        pytest.fail("source reconciliation requires a disposable graph")
    if uri == (os.environ.get("IMAS_CODEX_TEST_PROJECT_NEO4J_URI") or get_graph_uri()):
        pytest.fail("source reconciliation refuses the project graph")
    password = os.environ.get("IMAS_CODEX_TEST_NEO4J_PASSWORD", "")
    auth = ("neo4j", password) if password else None
    with GraphDatabase.driver(uri, auth=auth) as driver:
        driver.verify_connectivity()
    yield uri, password


@pytest.mark.graph
def test_stale_structural_sources_are_not_rebound(
    disposable_neo4j: tuple[str, str],
) -> None:
    from imas_codex.standard_names.graph_ops import (
        reconcile_orphan_parent_sources,
        reconcile_orphan_parent_sources_batched,
    )

    uri, password = disposable_neo4j
    parent_ids = ["electron_diffusivity", "ion_diffusivity"]
    source_ids = [f"derived:{parent_id}" for parent_id in parent_ids]
    child_ids = [f"radial_{parent_id}" for parent_id in parent_ids]
    client = RealGraphClient(
        uri=uri,
        username="neo4j",
        password=password,
        graph_name="disposable-parent-source-reconcile",
    )
    try:
        client.query(
            """
            UNWIND range(0, size($parent_ids) - 1) AS index
            CREATE (parent:StandardName {
              id: $parent_ids[index], name_stage: 'accepted', origin: 'derived'
            })
            CREATE (child:StandardName {
              id: $child_ids[index], name_stage: 'accepted', origin: 'pipeline'
            })
            CREATE (child)-[:HAS_PARENT]->(parent)
            CREATE (:StandardNameSource {
              id: $source_ids[index], status: 'stale', source_type: 'derived',
              source_id: $parent_ids[index], produced_sn_id: $parent_ids[index]
            })
            """,
            parent_ids=parent_ids,
            child_ids=child_ids,
            source_ids=source_ids,
        )

        batched_refused = False
        try:
            reconcile_orphan_parent_sources_batched(
                client,
                [{"parent_id": parent_ids[1]}],
            )
        except RuntimeError:
            batched_refused = True
        scalar_seeded = reconcile_orphan_parent_sources(
            gc=client,
            classification={
                "repairable": [{"parent_id": parent_ids[0]}],
                "rejected_derived": [],
            },
        )

        rows = client.query(
            """
            UNWIND $source_ids AS source_id
            MATCH (source:StandardNameSource {id: source_id})
            OPTIONAL MATCH (source)-[binding:PRODUCED_NAME]->(:StandardName)
            RETURN source.id AS source_id,
                   source.status AS status,
                   count(binding) AS bindings
            ORDER BY source_id
            """,
            source_ids=source_ids,
        )
        assert rows == [
            {
                "source_id": "derived:electron_diffusivity",
                "status": "stale",
                "bindings": 0,
            },
            {
                "source_id": "derived:ion_diffusivity",
                "status": "stale",
                "bindings": 0,
            },
        ]
        assert scalar_seeded == 0
        assert batched_refused
    finally:
        client.query(
            """
            MATCH (node)
            WHERE node.id IN $ids
            DETACH DELETE node
            """,
            ids=[*parent_ids, *source_ids, *child_ids],
        )
        client.close()
