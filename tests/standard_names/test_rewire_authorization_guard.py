"""Disposable-graph coverage for structural successor-rewire authority."""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest
from neo4j import GraphDatabase

from imas_codex.graph.client import GraphClient
from imas_codex.settings import get_graph_uri
from imas_codex.standard_names.graph_ops import (
    _rewire_has_parent_off_superseded,
)

SPECTRAL_CHILD = "spectral_signal_to_noise_ratio_of_spectrometer_channel"
SPECTRAL_OLD_PARENT = "signal_to_noise_ratio_of_spectrometer_channel"
SPECTRAL_TIP = "logarithm_of_spectral_signal_to_noise_ratio_of_spectrometer_channel"
SPECTRAL_CHAIN = [
    SPECTRAL_TIP,
    "logarithm_of_signal_to_noise_ratio_at_spectral_line",
    "ratio_of_spectral_power_to_reference_spectral_power",
    "spectral_signal_to_noise_ratio",
    "signal_to_noise_ratio",
    "reference_signal_to_noise_ratio",
    SPECTRAL_OLD_PARENT,
]

LEGITIMATE_CHILD = "maximum_of_electron_temperature"
LEGITIMATE_OLD_PARENT = "temperature_of_electrons"
LEGITIMATE_TIP = "electron_temperature"

TEST_NODE_IDS = [
    SPECTRAL_CHILD,
    *SPECTRAL_CHAIN,
    LEGITIMATE_CHILD,
    LEGITIMATE_OLD_PARENT,
    LEGITIMATE_TIP,
]


@pytest.fixture()
def disposable_graph() -> Iterator[GraphClient]:
    uri = os.environ.get("IMAS_CODEX_TEST_NEO4J_URI")
    if not uri:
        pytest.skip("IMAS_CODEX_TEST_NEO4J_URI is not configured")
    if os.environ.get("IMAS_CODEX_TEST_NEO4J_EPHEMERAL") != "1":
        pytest.fail("successor-rewire tests require a disposable graph")
    project_uri = os.environ.get("IMAS_CODEX_TEST_PROJECT_NEO4J_URI") or get_graph_uri()
    if uri == project_uri:
        pytest.fail("successor-rewire tests refuse the project graph")
    password = os.environ.get("IMAS_CODEX_TEST_NEO4J_PASSWORD", "")
    auth = ("neo4j", password) if password else None
    with GraphDatabase.driver(uri, auth=auth) as driver:
        driver.verify_connectivity()

    client = GraphClient(
        uri=uri,
        username="neo4j",
        password=password,
        graph_name="disposable-structural-rewire",
    )
    client.query(
        "MATCH (node) WHERE node.id IN $ids DETACH DELETE node",
        ids=TEST_NODE_IDS,
    )
    try:
        yield client
    finally:
        client.query(
            "MATCH (node) WHERE node.id IN $ids DETACH DELETE node",
            ids=TEST_NODE_IDS,
        )
        client.close()


def _parent_count(client: GraphClient, child_id: str, parent_id: str) -> int:
    return int(
        client.query(
            """
            MATCH (:StandardName {id: $child_id})-[edge:HAS_PARENT]->
                  (:StandardName {id: $parent_id})
            RETURN count(edge) AS count
            """,
            child_id=child_id,
            parent_id=parent_id,
        )[0]["count"]
    )


def _run_unguarded_rewire(client: GraphClient) -> int:
    result = client.query(
        """
        MATCH (old:StandardName)
        WHERE old.name_stage = 'superseded'
          AND EXISTS { (child)-[:HAS_PARENT]->(old) }
        MATCH path = (tip:StandardName)-[:REFINED_FROM*1..]->(old)
        WHERE tip.name_stage <> 'superseded'
          AND coalesce(tip.physical_base, '') = coalesce(old.physical_base, '')
          AND coalesce(tip.geometric_base, '') = coalesce(old.geometric_base, '')
          AND coalesce(tip.subject, '') = coalesce(old.subject, '')
          AND coalesce(tip.component, '') = coalesce(old.component, '')
        WITH old, tip, length(path) AS hops
        ORDER BY hops DESC
        WITH old, head(collect(tip)) AS tip
        WHERE tip IS NOT NULL
        MATCH (child)-[edge:HAS_PARENT]->(old)
        WITH tip, child, properties(edge) AS props, edge
        DELETE edge
        WITH tip, child, props
        WHERE tip.id <> child.id
        MERGE (child)-[replacement:HAS_PARENT]->(tip)
        SET replacement = props
        RETURN count(replacement) AS migrated
        """
    )
    return int(result[0]["migrated"] if result else 0)


@pytest.mark.graph
def test_six_hop_qualifier_relocation_is_refused(
    disposable_graph: GraphClient,
) -> None:
    client = disposable_graph
    client.query(
        """
        UNWIND $chain AS name_id
        CREATE (:StandardName {
            id: name_id,
            name_stage: CASE WHEN name_id = $tip THEN 'accepted' ELSE 'superseded' END,
            physical_base: 'signal_to_noise_ratio'
        })
        """,
        chain=SPECTRAL_CHAIN,
        tip=SPECTRAL_TIP,
    )
    client.query(
        """
        UNWIND range(0, size($chain) - 2) AS position
        MATCH (successor:StandardName {id: $chain[position]})
        MATCH (predecessor:StandardName {id: $chain[position + 1]})
        CREATE (successor)-[:REFINED_FROM]->(predecessor)
        """,
        chain=SPECTRAL_CHAIN,
    )
    client.query(
        """
        CREATE (child:StandardName {
            id: $child_id,
            name_stage: 'accepted',
            physical_base: 'signal_to_noise_ratio'
        })
        WITH child
        MATCH (old:StandardName {id: $old_parent_id})
        CREATE (child)-[:HAS_PARENT {
            operator: 'spectral', operator_kind: 'qualifier'
        }]->(old)
        """,
        child_id=SPECTRAL_CHILD,
        old_parent_id=SPECTRAL_OLD_PARENT,
    )

    assert _run_unguarded_rewire(client) == 1
    assert _parent_count(client, SPECTRAL_CHILD, SPECTRAL_TIP) == 1

    client.query(
        """
        MATCH (child:StandardName {id: $child_id})-[edge:HAS_PARENT]->()
        DELETE edge
        WITH child
        MATCH (old:StandardName {id: $old_parent_id})
        CREATE (child)-[:HAS_PARENT {
            operator: 'spectral', operator_kind: 'qualifier'
        }]->(old)
        """,
        child_id=SPECTRAL_CHILD,
        old_parent_id=SPECTRAL_OLD_PARENT,
    )

    assert _rewire_has_parent_off_superseded(client) == 0
    assert _parent_count(client, SPECTRAL_CHILD, SPECTRAL_OLD_PARENT) == 1
    assert _parent_count(client, SPECTRAL_CHILD, SPECTRAL_TIP) == 0


@pytest.mark.graph
def test_current_unary_prefix_parent_rewire_succeeds(
    disposable_graph: GraphClient,
) -> None:
    client = disposable_graph
    client.query(
        """
        CREATE (child:StandardName {
            id: $child_id, name_stage: 'accepted'
        })
        CREATE (old:StandardName {
            id: $old_parent_id,
            name_stage: 'superseded',
            physical_base: 'temperature',
            subject: 'electron'
        })
        CREATE (tip:StandardName {
            id: $tip_id,
            name_stage: 'accepted',
            physical_base: 'temperature',
            subject: 'electron'
        })
        CREATE (tip)-[:REFINED_FROM]->(old)
        CREATE (child)-[:HAS_PARENT {
            operator: 'maximum', operator_kind: 'unary_prefix'
        }]->(old)
        """,
        child_id=LEGITIMATE_CHILD,
        old_parent_id=LEGITIMATE_OLD_PARENT,
        tip_id=LEGITIMATE_TIP,
    )

    assert _rewire_has_parent_off_superseded(client) == 1
    assert _parent_count(client, LEGITIMATE_CHILD, LEGITIMATE_OLD_PARENT) == 0
    assert _parent_count(client, LEGITIMATE_CHILD, LEGITIMATE_TIP) == 1
