"""Every delete route leaves a receipt naming the name it removed.

Two routes terminate a ``StandardName`` without reaching the shared deletion
writer: the exact-reset scaffold retirement inside ``clear_standard_names``,
which wrote a bare ``StandardNameChange``, and ``purge_standard_names``, which
wrote no change row at all. Each test drives one route against the live graph
and reads the receipt back, so a removal that stops recording fails here rather
than going silent.

The suite is graph-marked: the exact-reset branch reads post-delete state
inside one statement.
"""

from __future__ import annotations

import uuid

import pytest

from imas_codex.graph.sn_cleanup import purge_standard_names
from imas_codex.standard_names.graph_ops import clear_standard_names

_PREFIX = "__delsnapshotroute__"


@pytest.fixture()
def _graph():
    try:
        from imas_codex.graph.client import GraphClient

        client = GraphClient()
        client.get_stats()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Neo4j not available: {exc}")

    yield client
    client.close()


@pytest.fixture()
def _clean(_graph):
    def wipe() -> None:
        # ``original_id`` is matched beside ``from_name``: a teardown keyed on
        # the change row alone strands the snapshot it was born with.
        _graph.query(
            """
            MATCH (node)
            WHERE node.id STARTS WITH $prefix
               OR node.source_id STARTS WITH $prefix
               OR node.from_name STARTS WITH $prefix
               OR node.original_id STARTS WITH $prefix
            DETACH DELETE node
            """,
            prefix=_PREFIX,
        )

    wipe()
    yield
    wipe()


def _id(label: str) -> str:
    return f"{_PREFIX}{label}_{uuid.uuid4().hex}"


def _exists(graph, label: str, node_id: str) -> bool:
    rows = graph.query(
        f"MATCH (node:{label} {{id: $id}}) RETURN count(node) AS count", id=node_id
    )
    return bool(rows and rows[0]["count"])


def _create_scaffold_parent(graph, parent_id: str) -> None:
    """A normalized derived parent whose only neighbour is its derived source."""
    graph.query(
        """
        CREATE (parent:StandardName {
            id: $parent_id,
            transformation: 'difference',
            aggregation: 'total',
            subject: 'neutral',
            physical_base: 'density'
        })
        CREATE (source:StandardNameSource {
            id: 'derived:' + $parent_id,
            source_type: 'derived',
            source_id: $parent_id,
            batch_key: 'derived_parent',
            status: 'composed',
            attempt_count: 0,
            produced_sn_id: $parent_id,
            created_at: datetime(),
            composed_at: datetime()
        })-[:PRODUCED_NAME]->(parent)
        """,
        parent_id=parent_id,
    )


def _snapshot_for(graph, node_id: str) -> dict | None:
    rows = graph.query(
        """
        MATCH (change:StandardNameChange)-[:HAS_DELETION_SNAPSHOT]->
              (snapshot:StandardNameDeletionSnapshot)
        WHERE snapshot.original_id = $node_id
        RETURN snapshot.id AS snapshot_id,
               snapshot.original_id AS original_id,
               change.operation AS operation,
               change.from_name AS from_name
        """,
        node_id=node_id,
    )
    return dict(rows[0]) if rows else None


def _snapshot_edge_neighbours(graph, snapshot_id: str) -> set[tuple[str, str]]:
    rows = graph.query(
        """
        MATCH (:StandardNameDeletionSnapshot {id: $snapshot_id})
              -[:HAS_EDGE_SNAPSHOT]->(edge)
        RETURN edge.relationship_type AS relationship_type,
               edge.neighbor_id AS neighbor_id
        """,
        snapshot_id=snapshot_id,
    )
    return {(row["relationship_type"], row["neighbor_id"]) for row in rows}


@pytest.mark.graph
def test_exact_reset_scaffold_leaves_a_deletion_snapshot(_graph, _clean) -> None:
    path = _id("path")
    candidate = _id("candidate")
    parent = _id("parent")

    _graph.query("CREATE (:IMASNode {id: $path})", path=path)
    _graph.query(
        """
        MATCH (path:IMASNode {id: $path})
        CREATE (candidate:StandardName {
            id: $candidate,
            origin: 'pipeline',
            name_stage: 'drafted',
            source_types: ['dd']
        })
        CREATE (path)-[:HAS_STANDARD_NAME]->(candidate)
        """,
        path=path,
        candidate=candidate,
    )
    _create_scaffold_parent(_graph, parent)
    _graph.query(
        """
        MATCH (candidate:StandardName {id: $candidate})
        MATCH (parent:StandardName {id: $parent})
        CREATE (candidate)-[:HAS_PARENT {operator_kind: 'binary'}]->(parent)
        """,
        candidate=candidate,
        parent=parent,
    )

    assert clear_standard_names(path_allowlist=[path]) == 2
    assert not _exists(_graph, "StandardName", candidate)
    assert not _exists(_graph, "StandardName", parent)

    scaffold = _snapshot_for(_graph, parent)
    assert scaffold is not None, "the retired scaffold left no deletion snapshot"
    assert scaffold["original_id"] == parent
    assert scaffold["from_name"] == parent
    assert scaffold["operation"] == "remove_skeleton_placeholder"
    assert scaffold["snapshot_id"].endswith(":node")
    removed_edges = _snapshot_edge_neighbours(_graph, scaffold["snapshot_id"])
    assert ("PRODUCED_NAME", f"derived:{parent}") in removed_edges, removed_edges

    # The candidate's receipt shares the statement and keeps its own operation,
    # which is what namespacing the scaffold parameters buys.
    selected = _snapshot_for(_graph, candidate)
    assert selected is not None
    assert selected["operation"] == "clear_selected_name"


@pytest.mark.graph
def test_purge_standard_names_leaves_a_deletion_snapshot(_graph, _clean) -> None:
    name = _id("quarantined")
    _graph.query(
        """
        CREATE (:StandardName {
            id: $name,
            origin: 'derived',
            name_stage: 'pending',
            validation_status: 'quarantined',
            source_paths: ['dd:pulse_schedule/x/reference']
        })
        """,
        name=name,
    )

    assert purge_standard_names(_graph, [name]) == 1
    assert not _exists(_graph, "StandardName", name)

    snapshot = _snapshot_for(_graph, name)
    assert snapshot is not None, "the purge left no deletion snapshot"
    assert snapshot["original_id"] == name
    assert snapshot["from_name"] == name
    assert snapshot["operation"] == "purge_quarantined_name"
