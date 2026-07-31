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


def _create_normalized_parent(
    graph,
    parent_id: str,
    *,
    origin: str | None = None,
    name_stage: str | None = None,
    docs_stage: str | None = None,
    produced_sn_id: str | None = None,
) -> None:
    graph.query(
        """
        CREATE (parent:StandardName {
            id: $parent_id,
            transformation: 'difference',
            aggregation: 'total',
            subject: 'neutral',
            physical_base: 'density'
        })
        FOREACH (_ IN CASE WHEN $origin IS NULL THEN [] ELSE [1] END |
            SET parent.origin = $origin)
        FOREACH (_ IN CASE WHEN $name_stage IS NULL THEN [] ELSE [1] END |
            SET parent.name_stage = $name_stage)
        FOREACH (_ IN CASE WHEN $docs_stage IS NULL THEN [] ELSE [1] END |
            SET parent.docs_stage = $docs_stage)
        CREATE (source:StandardNameSource {
            id: 'derived:' + $parent_id,
            source_type: 'derived',
            source_id: $parent_id,
            batch_key: 'derived_parent',
            status: 'composed',
            attempt_count: 0,
            produced_sn_id: coalesce($produced_sn_id, $parent_id),
            created_at: datetime(),
            composed_at: datetime()
        })-[:PRODUCED_NAME]->(parent)
        """,
        parent_id=parent_id,
        origin=origin,
        name_stage=name_stage,
        docs_stage=docs_stage,
        produced_sn_id=produced_sn_id,
    )


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


@pytest.mark.graph
def test_exact_reset_reaps_normalized_parent_and_preserves_owned_neighbors(
    _graph, _clean
) -> None:
    path = _id("normalized_path")
    outside_path = _id("normalized_outside_path")
    candidate = _id("normalized_candidate")
    eligible = _id("normalized_eligible")
    accepted = _id("normalized_accepted")
    catalog = _id("normalized_catalog")
    docs_reviewed = _id("normalized_docs_reviewed")
    live_child_parent = _id("normalized_live_child_parent")
    shared_parent = _id("normalized_shared_parent")
    extra_source_parent = _id("normalized_extra_source_parent")
    wrong_mirror_parent = _id("normalized_wrong_mirror_parent")
    unrelated_edge_parent = _id("normalized_unrelated_edge_parent")
    live_child = _id("normalized_live_child")
    shared_child = _id("normalized_shared_child")
    operand_a = _id("normalized_operand_a")
    operand_b = _id("normalized_operand_b")
    token_id = _id("normalized_token")

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
        CREATE (operand_a:StandardName {
            id: $operand_a,
            origin: 'catalog_edit',
            name_stage: 'accepted',
            docs_stage: 'accepted'
        })
        CREATE (operand_b:StandardName {
            id: $operand_b,
            origin: 'catalog_edit',
            name_stage: 'accepted',
            docs_stage: 'accepted'
        })
        CREATE (:GrammarToken {
            id: $token_id,
            value: 'total',
            segment: 'aggregation',
            version: 'test'
        })
        """,
        path=path,
        candidate=candidate,
        operand_a=operand_a,
        operand_b=operand_b,
        token_id=token_id,
    )

    _create_normalized_parent(_graph, eligible)
    _create_normalized_parent(
        _graph,
        accepted,
        origin="derived",
        name_stage="accepted",
    )
    _create_normalized_parent(
        _graph,
        catalog,
        origin="catalog_edit",
        name_stage="pending",
    )
    _create_normalized_parent(
        _graph,
        docs_reviewed,
        origin="derived",
        name_stage="pending",
        docs_stage="reviewed",
    )
    _create_normalized_parent(_graph, live_child_parent)
    _create_normalized_parent(_graph, shared_parent)
    _create_normalized_parent(_graph, extra_source_parent)
    _create_normalized_parent(
        _graph,
        wrong_mirror_parent,
        produced_sn_id=_id("wrong_target"),
    )
    _create_normalized_parent(_graph, unrelated_edge_parent)

    parents = [
        eligible,
        accepted,
        catalog,
        docs_reviewed,
        live_child_parent,
        shared_parent,
        extra_source_parent,
        wrong_mirror_parent,
        unrelated_edge_parent,
    ]
    _graph.query(
        """
        MATCH (candidate:StandardName {id: $candidate})
        MATCH (operand_a:StandardName {id: $operand_a})
        MATCH (operand_b:StandardName {id: $operand_b})
        MATCH (token:GrammarToken {id: $token_id})
        MERGE (locus:Locus {id: 'isotope'})
        WITH candidate, operand_a, operand_b, token, locus
        UNWIND $parents AS parent_id
        MATCH (parent:StandardName {id: parent_id})
        CREATE (candidate)-[:HAS_PARENT {
            operator: 'ratio',
            operator_kind: 'binary',
            role: 'b',
            separator: 'to'
        }]->(parent)
        CREATE (parent)-[:HAS_PARENT {
            operator: 'difference',
            operator_kind: 'binary',
            role: 'a',
            separator: 'and'
        }]->(operand_a)
        CREATE (parent)-[:HAS_PARENT {
            operator: 'difference',
            operator_kind: 'binary',
            role: 'b',
            separator: 'and'
        }]->(operand_b)
        CREATE (parent)-[:HAS_LOCUS {
            locus_token: 'isotope',
            locus_relation: 'of'
        }]->(locus)
        CREATE (parent)-[:HAS_SEGMENT {
            position: 0,
            segment: 'aggregation'
        }]->(token)
        CREATE (parent)-[:HAS_AGGREGATION]->(token)
        """,
        candidate=candidate,
        parents=parents,
        operand_a=operand_a,
        operand_b=operand_b,
        token_id=token_id,
    )
    _graph.query(
        """
        MATCH (parent:StandardName {id: $parent})
        CREATE (:StandardName {
            id: $child,
            origin: 'pipeline',
            name_stage: 'drafted'
        })-[:HAS_PARENT {
            operator_kind: 'binary',
            operator: 'ratio',
            role: 'b',
            separator: 'to'
        }]->(parent)
        """,
        parent=live_child_parent,
        child=live_child,
    )
    _graph.query(
        """
        MATCH (inside:IMASNode {id: $inside})
        MATCH (outside:IMASNode {id: $outside})
        MATCH (parent:StandardName {id: $parent})
        CREATE (shared:StandardName {
            id: $child,
            origin: 'pipeline',
            name_stage: 'drafted'
        })
        CREATE (inside)-[:HAS_STANDARD_NAME]->(shared)
        CREATE (outside)-[:HAS_STANDARD_NAME]->(shared)
        CREATE (shared)-[:HAS_PARENT {
            operator_kind: 'binary',
            operator: 'ratio',
            role: 'b',
            separator: 'to'
        }]->(parent)
        """,
        inside=path,
        outside=outside_path,
        parent=shared_parent,
        child=shared_child,
    )
    _graph.query(
        """
        MATCH (parent:StandardName {id: $parent})
        CREATE (:StandardNameSource {
            id: $source_id,
            source_type: 'dd',
            source_id: $source_id,
            status: 'composed'
        })-[:PRODUCED_NAME]->(parent)
        """,
        parent=extra_source_parent,
        source_id=_id("normalized_extra_source"),
    )
    _graph.query(
        """
        MATCH (parent:StandardName {id: $parent})
        MATCH (operand:StandardName {id: $operand})
        CREATE (parent)-[:HAS_SUCCESSOR]->(operand)
        """,
        parent=unrelated_edge_parent,
        operand=operand_a,
    )

    assert clear_standard_names(path_allowlist=[path], dry_run=True) == 2
    assert _exists(_graph, "StandardName", candidate)
    assert _exists(_graph, "StandardName", eligible)

    assert clear_standard_names(path_allowlist=[path]) == 2

    assert not _exists(_graph, "StandardName", candidate)
    assert not _exists(_graph, "StandardName", eligible)
    assert not _exists(_graph, "StandardNameSource", f"derived:{eligible}")
    for survivor in (
        accepted,
        catalog,
        docs_reviewed,
        live_child_parent,
        shared_parent,
        extra_source_parent,
        wrong_mirror_parent,
        unrelated_edge_parent,
        live_child,
        shared_child,
        operand_a,
        operand_b,
    ):
        assert _exists(_graph, "StandardName", survivor), survivor
    assert _exists(_graph, "Locus", "isotope")
