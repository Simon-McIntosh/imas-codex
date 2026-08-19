"""Transactional coverage for exhausted orphan retirement."""

from __future__ import annotations

import json
import os
from collections.abc import Iterator

import pytest
from neo4j import GraphDatabase

from imas_codex.graph.client import GraphClient
from imas_codex.settings import get_graph_uri
from imas_codex.standard_names.graph_ops import (
    supersede_exhausted_standard_name_orphans,
)


@pytest.fixture(scope="module")
def disposable_neo4j() -> Iterator[tuple[str, str]]:
    uri = os.environ.get("IMAS_CODEX_TEST_NEO4J_URI")
    if not uri:
        pytest.skip("IMAS_CODEX_TEST_NEO4J_URI is not configured")
    if os.environ.get("IMAS_CODEX_TEST_NEO4J_EPHEMERAL") != "1":
        pytest.fail("orphan supersession requires a disposable graph")
    if uri == (os.environ.get("IMAS_CODEX_TEST_PROJECT_NEO4J_URI") or get_graph_uri()):
        pytest.fail("orphan supersession refuses the project graph")
    password = os.environ.get("IMAS_CODEX_TEST_NEO4J_PASSWORD", "")
    auth = ("neo4j", password) if password else None
    with GraphDatabase.driver(uri, auth=auth) as driver:
        driver.verify_connectivity()
    yield uri, password


def _client(endpoint: tuple[str, str], name: str) -> GraphClient:
    uri, password = endpoint
    return GraphClient(uri=uri, username="neo4j", password=password, graph_name=name)


def _seed_name(
    client: GraphClient,
    name_id: str,
    *,
    name_stage: str = "exhausted",
    origin: str = "pipeline",
    with_source: bool = False,
) -> None:
    client.query(
        "CREATE (name:StandardName {id: $name_id, name_stage: $name_stage, "
        "status: 'draft', origin: $origin, validation_status: 'valid'})",
        name_id=name_id,
        name_stage=name_stage,
        origin=origin,
    )
    if with_source:
        client.query(
            "MATCH (name:StandardName {id: $name_id}) "
            "CREATE (source:StandardNameSource {id: $source_id, source_type: 'dd', "
            "source_id: $path, status: 'composed', produced_sn_id: $name_id}) "
            "CREATE (source)-[:PRODUCED_NAME]->(name)",
            name_id=name_id,
            source_id=f"dd:{name_id}/value",
            path=f"{name_id}/value",
        )


def _cleanup(client: GraphClient, prefix: str) -> None:
    client.query(
        "MATCH (node) WHERE node.id STARTS WITH $prefix OR "
        "node.from_name STARTS WITH $prefix DETACH DELETE node",
        prefix=prefix,
    )


def _snapshot(client: GraphClient, prefix: str) -> bytes:
    nodes = client.query(
        "MATCH (node) WHERE node.id STARTS WITH $prefix OR "
        "node.from_name STARTS WITH $prefix "
        "RETURN elementId(node) AS element_id, labels(node) AS labels, "
        "properties(node) AS properties ORDER BY element_id",
        prefix=prefix,
    )
    relationships = client.query(
        "MATCH (start)-[relationship]->(end) "
        "WHERE start.id STARTS WITH $prefix OR end.id STARTS WITH $prefix "
        "RETURN elementId(relationship) AS element_id, type(relationship) AS type, "
        "properties(relationship) AS properties, elementId(start) AS start_id, "
        "elementId(end) AS end_id ORDER BY element_id",
        prefix=prefix,
    )
    return json.dumps(
        {"nodes": nodes, "relationships": relationships},
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode()


@pytest.mark.graph
def test_exhausted_orphan_is_superseded_with_ledger(
    disposable_neo4j: tuple[str, str],
) -> None:
    prefix = "orphanretire"
    name_id = f"{prefix}_name"
    client = _client(disposable_neo4j, prefix)
    _seed_name(client, name_id)
    try:
        preview = supersede_exhausted_standard_name_orphans(
            [name_id], reason="source-less exhausted identities are terminal", gc=client
        )
        assert preview["outcome"] == "would_apply"
        assert preview["counts"] == {"requested": 1, "admitted": 1, "refused": 0}
        applied = supersede_exhausted_standard_name_orphans(
            [name_id],
            reason="source-less exhausted identities are terminal",
            apply=True,
            manifest_sha256=preview["manifest_sha256"],
            gc=client,
        )
        assert applied["outcome"] == "applied"
        assert applied["superseded"] == 1
        assert applied["ledger_rows"] == 1
        assert client.query(
            "MATCH (name:StandardName {id: $name_id})-[:HAS_INTERNAL_CHANGE]->"
            "(change:StandardNameChange) RETURN name.name_stage AS name_stage, "
            "name.status AS status, change.operation AS operation, "
            "change.manifest_sha256 AS manifest_hash",
            name_id=name_id,
        ) == [
            {
                "name_stage": "superseded",
                "status": "superseded",
                "operation": "supersede_exhausted_orphan",
                "manifest_hash": preview["manifest_sha256"],
            }
        ]
    finally:
        _cleanup(client, prefix)


@pytest.mark.graph
@pytest.mark.parametrize(
    ("suffix", "stage", "origin", "with_source", "reason_fragment"),
    [
        ("sourced", "exhausted", "pipeline", True, "live producing source"),
        ("accepted", "accepted", "pipeline", False, "name_stage is 'accepted'"),
        ("derived", "exhausted", "derived", False, "origin is derived"),
    ],
)
def test_ineligible_orphan_is_refused(
    disposable_neo4j: tuple[str, str],
    suffix: str,
    stage: str,
    origin: str,
    with_source: bool,
    reason_fragment: str,
) -> None:
    prefix = f"orphanrefusal{suffix}"
    name_id = f"{prefix}_name"
    client = _client(disposable_neo4j, prefix)
    _seed_name(
        client,
        name_id,
        name_stage=stage,
        origin=origin,
        with_source=with_source,
    )
    try:
        preview = supersede_exhausted_standard_name_orphans(
            [name_id], reason="only terminal source-less names qualify", gc=client
        )
        assert preview["outcome"] == "refused"
        assert preview["counts"] == {"requested": 1, "admitted": 0, "refused": 1}
        assert reason_fragment in preview["refusals"][0]["reason"]
        assert client.query(
            "MATCH (name:StandardName {id: $name_id}) "
            "RETURN name.name_stage AS stage, "
            "COUNT { (:StandardNameSource)-[:PRODUCED_NAME]->(name) } AS sources",
            name_id=name_id,
        ) == [{"stage": stage, "sources": int(with_source)}]
    finally:
        _cleanup(client, prefix)


@pytest.mark.graph
def test_orphan_supersession_replay_is_write_free(
    disposable_neo4j: tuple[str, str],
) -> None:
    prefix = "orphanreplay"
    name_id = f"{prefix}_name"
    client = _client(disposable_neo4j, prefix)
    _seed_name(client, name_id)
    try:
        preview = supersede_exhausted_standard_name_orphans(
            [name_id], reason="source-less exhausted identities are terminal", gc=client
        )
        supersede_exhausted_standard_name_orphans(
            [name_id],
            reason="source-less exhausted identities are terminal",
            apply=True,
            manifest_sha256=preview["manifest_sha256"],
            gc=client,
        )
        before = _snapshot(client, prefix)
        replay = supersede_exhausted_standard_name_orphans(
            [name_id],
            reason="source-less exhausted identities are terminal",
            apply=True,
            manifest_sha256=preview["manifest_sha256"],
            gc=client,
        )
        after = _snapshot(client, prefix)
        assert replay["outcome"] == "already_applied"
        assert replay["changed"] == 0
        assert replay["persistent_writes"] == 0
        assert after == before
    finally:
        _cleanup(client, prefix)
