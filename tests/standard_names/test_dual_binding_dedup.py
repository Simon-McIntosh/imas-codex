"""Transactional coverage for scalar-selected source deduplication."""

from __future__ import annotations

import json
import os
from collections.abc import Iterator

import pytest
from neo4j import GraphDatabase

from imas_codex.graph.client import GraphClient
from imas_codex.settings import get_graph_uri
from imas_codex.standard_names.graph_ops import (
    deduplicate_scalar_selected_sources,
)


@pytest.fixture(scope="module")
def disposable_neo4j() -> Iterator[tuple[str, str]]:
    uri = os.environ.get("IMAS_CODEX_TEST_NEO4J_URI")
    if not uri:
        pytest.skip("IMAS_CODEX_TEST_NEO4J_URI is not configured")
    if os.environ.get("IMAS_CODEX_TEST_NEO4J_EPHEMERAL") != "1":
        pytest.fail("source deduplication requires a disposable graph")
    if uri == (os.environ.get("IMAS_CODEX_TEST_PROJECT_NEO4J_URI") or get_graph_uri()):
        pytest.fail("source deduplication refuses the project graph")
    password = os.environ.get("IMAS_CODEX_TEST_NEO4J_PASSWORD", "")
    auth = ("neo4j", password) if password else None
    with GraphDatabase.driver(uri, auth=auth) as driver:
        driver.verify_connectivity()
    yield uri, password


def _client(endpoint: tuple[str, str], name: str) -> GraphClient:
    uri, password = endpoint
    return GraphClient(uri=uri, username="neo4j", password=password, graph_name=name)


def _seed(
    client: GraphClient, prefix: str, *, scalar: str | None = None
) -> dict[str, str]:
    ids = {
        "source": f"dd:{prefix}/value",
        "path": f"{prefix}/value",
        "keep": f"{prefix}_selected",
        "remove": f"{prefix}_redundant",
    }
    client.query(
        "CREATE (keep:StandardName {id: $keep, name_stage: 'accepted', "
        "validation_status: 'valid', source_paths: ['dd:' + $path]}) "
        "CREATE (remove:StandardName {id: $remove, name_stage: 'accepted', "
        "validation_status: 'valid', source_paths: ['dd:' + $path]}) "
        "CREATE (backing:IMASNode {id: $path}) "
        "CREATE (source:StandardNameSource {id: $source, source_type: 'dd', "
        "source_id: $path, status: 'attached', produced_sn_id: $scalar}) "
        "CREATE (source)-[:FROM_DD_PATH]->(backing) "
        "CREATE (source)-[:PRODUCED_NAME]->(keep) "
        "CREATE (source)-[:PRODUCED_NAME]->(remove) "
        "CREATE (backing)-[:HAS_STANDARD_NAME]->(keep) "
        "CREATE (backing)-[:HAS_STANDARD_NAME]->(remove)",
        **ids,
        scalar=scalar if scalar is not None else ids["keep"],
    )
    return ids


def _cleanup(
    client: GraphClient, ids: dict[str, str], manifest: str | None = None
) -> None:
    client.query(
        "MATCH (node) WHERE node.id IN $ids OR node.manifest_sha256 = $manifest "
        "DETACH DELETE node",
        ids=list(ids.values()),
        manifest=manifest,
    )


def _snapshot_bytes(client: GraphClient, ids: list[str]) -> bytes:
    nodes = client.query(
        "MATCH (node) WHERE node.id IN $ids OR node.manifest_sha256 = $manifest "
        "RETURN elementId(node) AS element_id, labels(node) AS labels, "
        "properties(node) AS properties ORDER BY element_id",
        ids=ids,
        manifest=ids[-1],
    )
    relationships = client.query(
        "MATCH (start)-[relationship]->(end) "
        "WHERE start.id IN $ids OR end.id IN $ids "
        "RETURN elementId(relationship) AS element_id, "
        "type(relationship) AS relationship_type, "
        "properties(relationship) AS properties, "
        "elementId(start) AS start_element_id, elementId(end) AS end_element_id "
        "ORDER BY element_id",
        ids=ids,
    )
    return json.dumps(
        {"nodes": nodes, "relationships": relationships},
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode()


@pytest.mark.graph
def test_scalar_selected_dedup_removes_other_binding_and_projection(
    disposable_neo4j: tuple[str, str],
) -> None:
    client = _client(disposable_neo4j, "scalar-selected-dedup")
    ids = _seed(client, "selecteddedup")
    preview = None
    try:
        preview = deduplicate_scalar_selected_sources(
            [ids["source"]],
            reason="the scalar selects the surviving identity",
            gc=client,
        )
        assert preview["outcome"] == "would_apply"
        assert preview["counts"] == {
            "requested": 1,
            "admitted": 1,
            "refused": 0,
            "bindings_to_remove": 1,
            "projections_to_remove": 1,
        }

        applied = deduplicate_scalar_selected_sources(
            [ids["source"]],
            reason="the scalar selects the surviving identity",
            apply=True,
            manifest_sha256=preview["manifest_sha256"],
            gc=client,
        )
        assert applied["outcome"] == "applied"
        assert applied["sources_deduplicated"] == 1
        assert client.query(
            "MATCH (source:StandardNameSource {id: $source}), "
            "(backing:IMASNode {id: $path}), "
            "(keep:StandardName {id: $keep}), "
            "(remove:StandardName {id: $remove}) "
            "RETURN source.produced_sn_id AS scalar, "
            "[(source)-[:PRODUCED_NAME]->(name) | name.id] AS bindings, "
            "[(backing)-[:HAS_STANDARD_NAME]->(name) | name.id] AS projections, "
            "keep.source_paths AS keep_paths, remove.source_paths AS remove_paths",
            **ids,
        ) == [
            {
                "scalar": ids["keep"],
                "bindings": [ids["keep"]],
                "projections": [ids["keep"]],
                "keep_paths": [f"dd:{ids['path']}"],
                "remove_paths": [],
            }
        ]
    finally:
        _cleanup(client, ids, (preview or {}).get("manifest_sha256"))


@pytest.mark.graph
def test_scalar_disagreement_is_signed_refusal(
    disposable_neo4j: tuple[str, str],
) -> None:
    client = _client(disposable_neo4j, "scalar-disagreement-refusal")
    ids = _seed(client, "scalardisagreement", scalar="unbound_identity")
    try:
        preview = deduplicate_scalar_selected_sources(
            [ids["source"]], reason="the scalar must select a live identity", gc=client
        )
        assert preview["outcome"] == "refused"
        assert preview["counts"]["refused"] == 1
        assert preview["refusals"] == [
            {
                "source_id": ids["source"],
                "reason": "produced_sn_id does not select exactly one live binding",
            }
        ]
        assert client.query(
            "MATCH (source:StandardNameSource {id: $source}) "
            "RETURN COUNT { (source)-[:PRODUCED_NAME]->(:StandardName) } AS bindings",
            **ids,
        ) == [{"bindings": 2}]
    finally:
        _cleanup(client, ids)


@pytest.mark.graph
def test_missing_backing_projection_is_signed_refusal(
    disposable_neo4j: tuple[str, str],
) -> None:
    client = _client(disposable_neo4j, "missing-projection-refusal")
    ids = _seed(client, "missingprojection")
    try:
        client.query(
            "MATCH (backing:IMASNode {id: $path})"
            "-[projection:HAS_STANDARD_NAME]->"
            "(:StandardName {id: $remove}) DELETE projection",
            **ids,
        )
        preview = deduplicate_scalar_selected_sources(
            [ids["source"]], reason="every deletion needs backing authority", gc=client
        )
        assert preview["outcome"] == "refused"
        assert preview["refusals"] == [
            {
                "source_id": ids["source"],
                "reason": "non-selected binding has no signed backing projection",
            }
        ]
        assert client.query(
            "MATCH (:StandardNameSource {id: $source})"
            "-[binding:PRODUCED_NAME]->(:StandardName) "
            "RETURN count(binding) AS bindings",
            **ids,
        ) == [{"bindings": 2}]
    finally:
        _cleanup(client, ids)


@pytest.mark.graph
def test_duplicate_backing_projection_is_signed_refusal(
    disposable_neo4j: tuple[str, str],
) -> None:
    client = _client(disposable_neo4j, "duplicate-projection-refusal")
    ids = _seed(client, "duplicateprojection")
    try:
        client.query(
            "MATCH (backing:IMASNode {id: $path}), "
            "(remove:StandardName {id: $remove}) "
            "CREATE (backing)-[:HAS_STANDARD_NAME]->(remove)",
            **ids,
        )
        preview = deduplicate_scalar_selected_sources(
            [ids["source"]], reason="backing authority must be unique", gc=client
        )
        assert preview["outcome"] == "refused"
        assert preview["refusals"] == [
            {
                "source_id": ids["source"],
                "reason": (
                    "non-selected binding has duplicate signed backing projections"
                ),
            }
        ]
        assert client.query(
            "MATCH (:IMASNode {id: $path})"
            "-[projection:HAS_STANDARD_NAME]->"
            "(:StandardName {id: $remove}) "
            "RETURN count(projection) AS projections",
            **ids,
        ) == [{"projections": 2}]
    finally:
        _cleanup(client, ids)


@pytest.mark.graph
def test_signed_zero_projection_exclusion_allows_exact_cohort_apply(
    disposable_neo4j: tuple[str, str],
) -> None:
    client = _client(disposable_neo4j, "signed-projection-exclusion")
    admitted = _seed(client, "admittedsource")
    excluded = _seed(client, "excludedsource")
    preview = None
    exclusion_reason = "backing projection authority is absent; preserve this row"
    try:
        client.query(
            "MATCH (backing:IMASNode {id: $path})"
            "-[projection:HAS_STANDARD_NAME]->"
            "(:StandardName {id: $remove}) DELETE projection",
            **excluded,
        )
        source_ids = [admitted["source"], excluded["source"]]
        exclusions = {excluded["source"]: exclusion_reason}
        preview = deduplicate_scalar_selected_sources(
            source_ids,
            reason="deduplicate only rows with complete backing authority",
            excluded_source_reasons=exclusions,
            gc=client,
        )
        assert preview["outcome"] == "would_apply"
        assert preview["counts"]["admitted"] == 1
        assert preview["counts"]["refused"] == 1
        assert preview["refusals"] == [
            {
                "source_id": excluded["source"],
                "reason": exclusion_reason,
                "classification": "explicit_exclusion",
                "authority_reason": (
                    "non-selected binding has no signed backing projection"
                ),
            }
        ]

        applied = deduplicate_scalar_selected_sources(
            source_ids,
            reason="deduplicate only rows with complete backing authority",
            excluded_source_reasons=exclusions,
            apply=True,
            manifest_sha256=preview["manifest_sha256"],
            gc=client,
        )
        assert applied["outcome"] == "applied"
        assert applied["sources_deduplicated"] == 1
        assert client.query(
            "UNWIND $ids AS source_id "
            "MATCH (source:StandardNameSource {id: source_id}) "
            "RETURN source_id, "
            "COUNT { (source)-[:PRODUCED_NAME]->(:StandardName) } AS bindings "
            "ORDER BY source_id",
            ids=source_ids,
        ) == [
            {"source_id": admitted["source"], "bindings": 1},
            {"source_id": excluded["source"], "bindings": 2},
        ]
        replay = deduplicate_scalar_selected_sources(
            source_ids,
            reason="deduplicate only rows with complete backing authority",
            excluded_source_reasons=exclusions,
            apply=True,
            manifest_sha256=preview["manifest_sha256"],
            gc=client,
        )
        assert replay["outcome"] == "already_applied"
        assert replay["changed"] == 0
    finally:
        _cleanup(client, admitted, (preview or {}).get("manifest_sha256"))
        _cleanup(client, excluded)


@pytest.mark.graph
def test_replay_is_measured_write_free(disposable_neo4j: tuple[str, str]) -> None:
    client = _client(disposable_neo4j, "source-dedup-replay")
    ids = _seed(client, "dedupreplay")
    preview = None
    try:
        preview = deduplicate_scalar_selected_sources(
            [ids["source"]],
            reason="the scalar selects the surviving identity",
            gc=client,
        )
        deduplicate_scalar_selected_sources(
            [ids["source"]],
            reason="the scalar selects the surviving identity",
            apply=True,
            manifest_sha256=preview["manifest_sha256"],
            gc=client,
        )
        participant_ids = [*ids.values(), preview["manifest_sha256"]]
        before = _snapshot_bytes(client, participant_ids)
        changes_before = client.query(
            "MATCH (change:StandardNameChange {manifest_sha256: $manifest}) "
            "RETURN count(change) AS changes",
            manifest=preview["manifest_sha256"],
        )
        replay = deduplicate_scalar_selected_sources(
            [ids["source"]],
            reason="the scalar selects the surviving identity",
            apply=True,
            manifest_sha256=preview["manifest_sha256"],
            gc=client,
        )
        after = _snapshot_bytes(client, participant_ids)
        changes_after = client.query(
            "MATCH (change:StandardNameChange {manifest_sha256: $manifest}) "
            "RETURN count(change) AS changes",
            manifest=preview["manifest_sha256"],
        )
        assert replay["outcome"] == "already_applied"
        assert replay["changed"] == 0
        assert after == before
        assert changes_after == changes_before == [{"changes": 1}]
    finally:
        _cleanup(client, ids, (preview or {}).get("manifest_sha256"))
