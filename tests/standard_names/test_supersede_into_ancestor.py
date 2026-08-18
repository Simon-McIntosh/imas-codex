"""Transactional coverage for folding a name into its lineage ancestor."""

from __future__ import annotations

import os
from collections.abc import Iterator
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from neo4j import GraphDatabase

from imas_codex.graph.client import GraphClient
from imas_codex.settings import get_graph_uri
from imas_codex.standard_names.graph_ops import (
    SupersedeIntoAncestorConflict,
    supersede_into_ancestor,
)


@pytest.fixture(scope="module")
def disposable_neo4j() -> Iterator[tuple[str, str]]:
    uri = os.environ.get("IMAS_CODEX_TEST_NEO4J_URI")
    if not uri:
        pytest.skip("IMAS_CODEX_TEST_NEO4J_URI is not configured")
    if os.environ.get("IMAS_CODEX_TEST_NEO4J_EPHEMERAL") != "1":
        pytest.fail("ancestor supersession requires a disposable graph")
    if uri == (os.environ.get("IMAS_CODEX_TEST_PROJECT_NEO4J_URI") or get_graph_uri()):
        pytest.fail("ancestor supersession refuses the project graph")
    password = os.environ.get("IMAS_CODEX_TEST_NEO4J_PASSWORD", "")
    auth = ("neo4j", password) if password else None
    with GraphDatabase.driver(uri, auth=auth) as driver:
        driver.verify_connectivity()
    yield uri, password


def _client(endpoint: tuple[str, str], name: str) -> GraphClient:
    uri, password = endpoint
    return GraphClient(
        uri=uri,
        username="neo4j",
        password=password,
        graph_name=name,
    )


def _seed(
    client: GraphClient, prefix: str, *, multi_hop: bool = False
) -> dict[str, str]:
    ids = {
        "old": f"{prefix}_descendant",
        "middle": f"{prefix}_middle",
        "ancestor": f"{prefix}_ancestor",
        "path": f"{prefix}/value",
        "source": f"dd:{prefix}/value",
    }
    client.query(
        "CREATE (old:StandardName {id: $old, name_stage: 'accepted', "
        "validation_status: 'valid', status: 'active', origin: 'pipeline', "
        "unit: 'eV', source_paths: ['dd:' + $path]}) "
        "CREATE (ancestor:StandardName {id: $ancestor, name_stage: 'accepted', "
        "validation_status: 'valid', status: 'active', origin: 'pipeline', "
        "unit: 'eV', source_paths: []}) "
        "CREATE (dd:IMASNode {id: $path, units: 'eV'}) "
        "CREATE (source:StandardNameSource {id: $source, source_type: 'dd', "
        "source_id: $path, status: 'composed', produced_sn_id: $old}) "
        "CREATE (source)-[:FROM_DD_PATH]->(dd) "
        "CREATE (source)-[:PRODUCED_NAME]->(old) "
        "CREATE (dd)-[:HAS_STANDARD_NAME]->(old)",
        **ids,
    )
    if multi_hop:
        client.query(
            "MATCH (old:StandardName {id: $old}), "
            "(ancestor:StandardName {id: $ancestor}) "
            "CREATE (middle:StandardName {id: $middle, name_stage: 'superseded', "
            "validation_status: 'valid', status: 'superseded', origin: 'pipeline'}) "
            "CREATE (old)-[:REFINED_FROM]->(middle) "
            "CREATE (middle)-[:REFINED_FROM]->(ancestor)",
            **ids,
        )
    else:
        client.query(
            "MATCH (old:StandardName {id: $old}), "
            "(ancestor:StandardName {id: $ancestor}) "
            "CREATE (old)-[:REFINED_FROM]->(ancestor)",
            **ids,
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


@pytest.mark.graph
def test_direct_ancestor_fold_preserves_lineage_and_retargets_source(
    disposable_neo4j: tuple[str, str],
) -> None:
    client = _client(disposable_neo4j, "direct-ancestor-fold")
    ids = _seed(client, "direct")
    preview = None
    try:
        preview = supersede_into_ancestor(
            ids["old"], ids["ancestor"], reason="same physical quantity", gc=client
        )
        assert preview["changed"] == 0
        assert preview["counts"] == {"sources": 1, "retarget": 1, "deduplicate": 0}

        applied = supersede_into_ancestor(
            ids["old"],
            ids["ancestor"],
            reason="same physical quantity",
            apply=True,
            manifest_sha256=preview["manifest_sha256"],
            gc=client,
        )
        assert applied["changed"] == 1
        assert client.query(
            "MATCH (old:StandardName {id: $old})-[:REFINED_FROM]->"
            "(ancestor:StandardName {id: $ancestor}) "
            "MATCH (source:StandardNameSource {id: $source})-[:PRODUCED_NAME]->(ancestor) "
            "RETURN old.name_stage AS stage, source.produced_sn_id AS scalar, "
            "old.source_paths AS old_paths, ancestor.source_paths AS ancestor_paths, "
            "COUNT { (ancestor)-[:REFINED_FROM]->(old) } AS reverse",
            **ids,
        ) == [
            {
                "stage": "superseded",
                "scalar": ids["ancestor"],
                "old_paths": [],
                "ancestor_paths": [f"dd:{ids['path']}"],
                "reverse": 0,
            }
        ]
    finally:
        _cleanup(client, ids, (preview or {}).get("manifest_sha256"))


@pytest.mark.graph
def test_multi_hop_ancestor_fold_and_replay_are_idempotent(
    disposable_neo4j: tuple[str, str],
) -> None:
    client = _client(disposable_neo4j, "multi-hop-ancestor-fold")
    ids = _seed(client, "multihop", multi_hop=True)
    preview = None
    try:
        preview = supersede_into_ancestor(
            ids["old"], ids["ancestor"], reason="same physical quantity", gc=client
        )
        applied = supersede_into_ancestor(
            ids["old"],
            ids["ancestor"],
            reason="same physical quantity",
            apply=True,
            manifest_sha256=preview["manifest_sha256"],
            gc=client,
        )
        replay = supersede_into_ancestor(
            ids["old"],
            ids["ancestor"],
            reason="same physical quantity",
            apply=True,
            manifest_sha256=preview["manifest_sha256"],
            gc=client,
        )
        assert applied["changed"] == 1
        assert replay["outcome"] == "already_applied"
        assert replay["changed"] == 0
        assert client.query(
            "MATCH (:StandardName {id: $old})-[first:REFINED_FROM]->"
            "(:StandardName {id: $middle})-[second:REFINED_FROM]->"
            "(:StandardName {id: $ancestor}) RETURN count(first) + count(second) AS edges",
            **ids,
        ) == [{"edges": 2}]
    finally:
        _cleanup(client, ids, (preview or {}).get("manifest_sha256"))


@pytest.mark.graph
def test_non_ancestor_target_is_refused(disposable_neo4j: tuple[str, str]) -> None:
    client = _client(disposable_neo4j, "non-ancestor-refusal")
    ids = _seed(client, "nonancestor")
    outsider = "nonancestor_outsider"
    client.query(
        "CREATE (:StandardName {id: $id, name_stage: 'accepted', "
        "validation_status: 'valid'})",
        id=outsider,
    )
    try:
        with pytest.raises(SupersedeIntoAncestorConflict, match="not an ancestor"):
            supersede_into_ancestor(
                ids["old"], outsider, reason="invalid target", gc=client
            )
    finally:
        ids["outsider"] = outsider
        _cleanup(client, ids)


@pytest.mark.graph
def test_source_revalidation_refusal_rolls_back(
    disposable_neo4j: tuple[str, str],
) -> None:
    client = _client(disposable_neo4j, "revalidation-refusal")
    ids = _seed(client, "revalidation")
    try:
        rejected = SimpleNamespace(
            accepted_source_ids=[],
            rejected=[
                SimpleNamespace(source_node_id=ids["source"], reason="unit mismatch")
            ],
        )
        with (
            patch(
                "imas_codex.standard_names.attachment_audit.guard_source_pairings",
                return_value=rejected,
            ),
            pytest.raises(SupersedeIntoAncestorConflict, match="re-validation failed"),
        ):
            supersede_into_ancestor(
                ids["old"], ids["ancestor"], reason="same physical quantity", gc=client
            )
        assert client.query(
            "MATCH (source:StandardNameSource {id: $source})-[:PRODUCED_NAME]->"
            "(old:StandardName {id: $old}) RETURN old.name_stage AS stage, "
            "source.produced_sn_id AS scalar",
            **ids,
        ) == [{"stage": "accepted", "scalar": ids["old"]}]
    finally:
        _cleanup(client, ids)


@pytest.mark.graph
def test_dual_binding_is_deduplicated_without_reversing_lineage(
    disposable_neo4j: tuple[str, str],
) -> None:
    client = _client(disposable_neo4j, "dual-binding-deduplication")
    ids = _seed(client, "dual")
    preview = None
    try:
        client.query(
            "MATCH (source:StandardNameSource {id: $source}), "
            "(ancestor:StandardName {id: $ancestor}), (dd:IMASNode {id: $path}) "
            "CREATE (source)-[:PRODUCED_NAME]->(ancestor) "
            "CREATE (dd)-[:HAS_STANDARD_NAME]->(ancestor) "
            "SET source.produced_sn_id = $ancestor",
            **ids,
        )
        preview = supersede_into_ancestor(
            ids["old"], ids["ancestor"], reason="same physical quantity", gc=client
        )
        assert preview["counts"] == {"sources": 1, "retarget": 0, "deduplicate": 1}
        supersede_into_ancestor(
            ids["old"],
            ids["ancestor"],
            reason="same physical quantity",
            apply=True,
            manifest_sha256=preview["manifest_sha256"],
            gc=client,
        )
        assert client.query(
            "MATCH (source:StandardNameSource {id: $source}) "
            "RETURN source.produced_sn_id AS scalar, "
            "COUNT { (source)-[:PRODUCED_NAME]->(:StandardName {id: $ancestor}) } AS kept, "
            "COUNT { (source)-[:PRODUCED_NAME]->(:StandardName {id: $old}) } AS removed",
            **ids,
        ) == [{"scalar": ids["ancestor"], "kept": 1, "removed": 0}]
    finally:
        _cleanup(client, ids, (preview or {}).get("manifest_sha256"))
