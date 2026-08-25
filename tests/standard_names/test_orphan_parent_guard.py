"""Disposable-graph coverage for orphan-parent provenance admission."""

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
        pytest.fail("orphan-parent admission requires a disposable graph")
    if uri == (os.environ.get("IMAS_CODEX_TEST_PROJECT_NEO4J_URI") or get_graph_uri()):
        pytest.fail("orphan-parent admission refuses the project graph")
    password = os.environ.get("IMAS_CODEX_TEST_NEO4J_PASSWORD", "")
    auth = ("neo4j", password) if password else None
    with GraphDatabase.driver(uri, auth=auth) as driver:
        driver.verify_connectivity()
    yield uri, password


@pytest.mark.graph
def test_reconcile_refuses_terminal_parent_and_source_bound_to_live_target(
    disposable_neo4j: tuple[str, str],
) -> None:
    from imas_codex.standard_names.graph_ops import (
        find_orphan_parent_source_candidates,
        reconcile_orphan_parent_sources,
    )

    uri, password = disposable_neo4j
    terminal_parent = "conductivity"
    terminal_without_source = "electrical_resistivity"
    live_parent = "thermal_conductivity"
    live_targets = ["electrical_conductivity", "effective_thermal_conductivity"]
    children = [
        "parallel_conductivity",
        "parallel_electrical_resistivity",
        "parallel_thermal_conductivity",
    ]
    source_ids = ["derived:conductivity", "derived:thermal_conductivity"]
    all_ids = [
        terminal_parent,
        terminal_without_source,
        live_parent,
        *live_targets,
        *children,
        *source_ids,
    ]
    client = RealGraphClient(
        uri=uri,
        username="neo4j",
        password=password,
        graph_name="disposable-orphan-parent-admission",
    )
    try:
        client.query(
            """
            CREATE (terminal:StandardName {
              id: $terminal_parent, name_stage: 'superseded', origin: 'derived'
            })
            CREATE (terminal_unbound:StandardName {
              id: $terminal_without_source, name_stage: 'superseded',
              origin: 'derived'
            })
            CREATE (live_parent:StandardName {
              id: $live_parent, name_stage: 'accepted', origin: 'derived'
            })
            WITH terminal, terminal_unbound, live_parent
            UNWIND range(0, 2) AS child_index
            CREATE (child:StandardName {
              id: $children[child_index], name_stage: 'accepted', origin: 'pipeline'
            })
            WITH terminal, terminal_unbound, live_parent, collect(child) AS children
            WITH terminal, terminal_unbound, live_parent,
                 children[0] AS terminal_child,
                 children[1] AS terminal_unbound_child,
                 children[2] AS live_child
            CREATE (terminal_child)-[:HAS_PARENT]->(terminal)
            CREATE (terminal_unbound_child)-[:HAS_PARENT]->(terminal_unbound)
            CREATE (live_child)-[:HAS_PARENT]->(live_parent)
            CREATE (terminal_target:StandardName {
              id: $live_targets[0], name_stage: 'accepted', origin: 'pipeline'
            })
            CREATE (bound_target:StandardName {
              id: $live_targets[1], name_stage: 'accepted', origin: 'pipeline'
            })
            CREATE (terminal_source:StandardNameSource {
              id: $source_ids[0], status: 'composed', source_type: 'derived',
              source_id: $terminal_parent, produced_sn_id: $live_targets[0]
            })
            CREATE (bound_source:StandardNameSource {
              id: $source_ids[1], status: 'composed', source_type: 'derived',
              source_id: $live_parent, produced_sn_id: $live_targets[1]
            })
            CREATE (terminal_source)-[:PRODUCED_NAME]->(terminal_target)
            CREATE (bound_source)-[:PRODUCED_NAME]->(bound_target)
            """,
            terminal_parent=terminal_parent,
            terminal_without_source=terminal_without_source,
            live_parent=live_parent,
            live_targets=live_targets,
            children=children,
            source_ids=source_ids,
        )

        def terminal_binding_count() -> int:
            rows = client.query(
                """
                MATCH (:StandardNameSource {id: $source_id})
                      -[binding:PRODUCED_NAME]->
                      (:StandardName {id: $parent_id})
                RETURN count(binding) AS count
                """,
                source_id=source_ids[0],
                parent_id=terminal_parent,
            )
            return int(rows[0]["count"])

        assert terminal_binding_count() == 0
        client.query(
            """
            MATCH (parent:StandardName {id: $parent_id})
            MATCH (source:StandardNameSource {id: $source_id})
            SET source.produced_sn_id = parent.id
            MERGE (source)-[:PRODUCED_NAME]->(parent)
            """,
            parent_id=terminal_parent,
            source_id=source_ids[0],
        )
        assert terminal_binding_count() == 1
        client.query(
            """
            MATCH (source:StandardNameSource {id: $source_id})
                  -[binding:PRODUCED_NAME]->
                  (:StandardName {id: $parent_id})
            DELETE binding
            SET source.produced_sn_id = $live_target
            """,
            source_id=source_ids[0],
            parent_id=terminal_parent,
            live_target=live_targets[0],
        )
        assert terminal_binding_count() == 0

        assert find_orphan_parent_source_candidates(gc=client) == []
        seeded = reconcile_orphan_parent_sources(
            gc=client,
            classification={
                "repairable": [
                    {"parent_id": terminal_parent},
                    {"parent_id": terminal_without_source},
                    {"parent_id": live_parent},
                ],
                "rejected_derived": [],
            },
        )

        assert seeded == 0
        assert terminal_binding_count() == 0
        rows = client.query(
            """
            UNWIND $source_ids AS source_id
            MATCH (source:StandardNameSource {id: source_id})
            OPTIONAL MATCH (source)-[:PRODUCED_NAME]->(target:StandardName)
            RETURN source.id AS source_id, source.produced_sn_id AS scalar,
                   collect(target.id) AS targets
            ORDER BY source_id
            """,
            source_ids=source_ids,
        )
        assert rows == [
            {
                "source_id": "derived:conductivity",
                "scalar": "electrical_conductivity",
                "targets": ["electrical_conductivity"],
            },
            {
                "source_id": "derived:thermal_conductivity",
                "scalar": "effective_thermal_conductivity",
                "targets": ["effective_thermal_conductivity"],
            },
        ]
        assert client.query(
            """
            MATCH (source:StandardNameSource {
              id: 'derived:electrical_resistivity'
            })
            RETURN count(source) AS count
            """
        ) == [{"count": 0}]
    finally:
        client.query(
            """
            MATCH (node)
            WHERE node.id IN $ids
            DETACH DELETE node
            """,
            ids=all_ids,
        )
        client.close()
