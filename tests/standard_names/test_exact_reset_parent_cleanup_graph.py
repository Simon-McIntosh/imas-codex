"""Live-graph behavior for exact reset cleanup of derived scaffolds.

Every test node carries a unique reserved prefix and the fixture removes only
that prefixed subgraph. The suite is graph-marked because the cleanup relies on
Neo4j read-after-delete semantics within one statement.
"""

from __future__ import annotations

import uuid

import pytest

from imas_codex.standard_names.graph_ops import clear_standard_names

_PREFIX = "__exact_reset_scaffold__"


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
        _graph.query(
            """
            MATCH (node)
            WHERE node.id STARTS WITH $prefix
               OR node.source_id STARTS WITH $prefix
               OR node.from_name STARTS WITH $prefix
            DETACH DELETE node
            """,
            prefix=_PREFIX,
        )

    wipe()
    yield
    wipe()


def _id(label: str) -> str:
    return f"{_PREFIX}{label}_{uuid.uuid4().hex}"


def _create_name(
    graph,
    name: str,
    *,
    origin: str = "derived",
    name_stage: str = "pending",
    extra_set: str = "",
) -> None:
    graph.query(
        f"""
        CREATE (sn:StandardName {{
            id: $name,
            origin: $origin,
            name_stage: $name_stage
        }})
        {extra_set}
        """,
        name=name,
        origin=origin,
        name_stage=name_stage,
    )


def _exists(graph, label: str, node_id: str) -> bool:
    rows = graph.query(
        f"MATCH (node:{label} {{id: $id}}) RETURN count(node) AS count",
        id=node_id,
    )
    return bool(rows and rows[0]["count"])


@pytest.mark.graph
def test_exact_reset_reaps_only_newly_orphaned_null_lifecycle_scaffold(
    _graph, _clean
) -> None:
    path = _id("path")
    candidate = _id("candidate")
    orphan = _id("orphan")
    accepted = _id("accepted")
    childful = _id("childful")
    lifecycle = _id("lifecycle")
    catalog = _id("catalog")
    claimed = _id("claimed")
    source_owned = _id("source_owned")
    malformed_source = _id("malformed_source")
    nonstructural = _id("nonstructural")
    selected_parent = _id("selected_parent")
    multi_producer_candidate = _id("multi_producer_candidate")
    multi_producer_parent = _id("multi_producer_parent")
    unrelated = _id("unrelated")
    surviving_child = _id("surviving_child")
    outside_path = _id("outside_path")

    _graph.query("CREATE (:IMASNode {id: $path})", path=path)
    _graph.query("CREATE (:IMASNode {id: $path})", path=outside_path)
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
    _create_name(_graph, orphan)
    _create_name(_graph, accepted, name_stage="accepted")
    _create_name(_graph, childful)
    _create_name(
        _graph,
        lifecycle,
        extra_set="SET sn.docs_stage = 'pending'",
    )
    _create_name(_graph, catalog, origin="catalog_edit")
    _create_name(
        _graph,
        claimed,
        extra_set="SET sn.claimed_at = datetime(), sn.claim_token = 'owner'",
    )
    _create_name(_graph, source_owned)
    _create_name(_graph, malformed_source)
    _create_name(_graph, nonstructural)
    _create_name(_graph, selected_parent)
    _create_name(
        _graph,
        multi_producer_candidate,
        origin="pipeline",
        name_stage="drafted",
    )
    _create_name(_graph, multi_producer_parent)
    _create_name(_graph, unrelated)
    _create_name(_graph, surviving_child, origin="pipeline", name_stage="drafted")

    _graph.query(
        """
        MATCH (candidate:StandardName {id: $candidate})
        UNWIND $parents AS parent_id
        MATCH (parent:StandardName {id: parent_id})
        CREATE (candidate)-[:HAS_PARENT {operator_kind: 'binary'}]->(parent)
        """,
        candidate=candidate,
        parents=[
            orphan,
            accepted,
            childful,
            lifecycle,
            catalog,
            claimed,
            source_owned,
            malformed_source,
            selected_parent,
        ],
    )
    _graph.query(
        """
        MATCH (candidate:StandardName {id: $candidate})
        MATCH (parent:StandardName {id: $parent})
        CREATE (candidate)-[:HAS_PARENT]->(parent)
        """,
        candidate=candidate,
        parent=nonstructural,
    )
    _graph.query(
        """
        MATCH (child:StandardName {id: $child})
        MATCH (parent:StandardName {id: $parent})
        CREATE (child)-[:HAS_PARENT {operator_kind: 'binary'}]->(parent)
        """,
        child=surviving_child,
        parent=childful,
    )
    _graph.query(
        """
        MATCH (parent:StandardName {id: $parent})
        CREATE (source:StandardNameSource {
            id: 'derived:' + $parent,
            source_type: 'derived',
            source_id: $parent,
            batch_key: 'derived_parent',
            attempt_count: 0,
            produced_sn_id: $parent
        })-[:PRODUCED_NAME]->(parent)
        """,
        parent=orphan,
    )
    _graph.query(
        """
        MATCH (parent:StandardName {id: $parent})
        CREATE (source:StandardNameSource {
            id: $source,
            source_type: 'dd',
            source_id: $source,
            status: 'composed',
            produced_sn_id: $parent
        })-[:PRODUCED_NAME]->(parent)
        """,
        parent=source_owned,
        source=_id("source"),
    )
    _graph.query(
        """
        MATCH (parent:StandardName {id: $parent})
        CREATE (source:StandardNameSource {
            id: 'derived:' + $parent,
            source_type: 'derived',
            source_id: $parent,
            batch_key: 'derived_parent',
            attempt_count: 0,
            produced_sn_id: $parent,
            unexpected_metadata: 'preserve'
        })-[:PRODUCED_NAME]->(parent)
        """,
        parent=malformed_source,
    )
    _graph.query(
        """
        MATCH (path:IMASNode {id: $path})
        MATCH (selected:StandardName {id: $selected})
        CREATE (path)-[:HAS_STANDARD_NAME]->(selected)
        """,
        path=path,
        selected=selected_parent,
    )
    _graph.query(
        """
        MATCH (inside:IMASNode {id: $inside})
        MATCH (outside:IMASNode {id: $outside})
        MATCH (candidate:StandardName {id: $candidate})
        MATCH (parent:StandardName {id: $parent})
        CREATE (inside)-[:HAS_STANDARD_NAME]->(candidate)
        CREATE (outside)-[:HAS_STANDARD_NAME]->(candidate)
        CREATE (candidate)-[:HAS_PARENT {operator_kind: 'binary'}]->(parent)
        """,
        inside=path,
        outside=outside_path,
        candidate=multi_producer_candidate,
        parent=multi_producer_parent,
    )

    assert clear_standard_names(path_allowlist=[path], dry_run=True) == 3
    assert _exists(_graph, "StandardName", candidate)
    assert _exists(_graph, "StandardName", orphan)

    assert clear_standard_names(path_allowlist=[path]) == 3

    assert not _exists(_graph, "StandardName", candidate)
    assert not _exists(_graph, "StandardName", orphan)
    assert not _exists(_graph, "StandardName", selected_parent)
    assert not _exists(_graph, "StandardNameSource", f"derived:{orphan}")
    for survivor in (
        accepted,
        childful,
        lifecycle,
        catalog,
        claimed,
        source_owned,
        malformed_source,
        nonstructural,
        multi_producer_candidate,
        multi_producer_parent,
        unrelated,
    ):
        assert _exists(_graph, "StandardName", survivor), survivor
