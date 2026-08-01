"""Live-graph coverage for write-local placeholder cleanup."""

from __future__ import annotations

import uuid
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from imas_codex.standard_names.derivation import DerivedEdge
from imas_codex.standard_names.graph_ops import write_standard_names

pytestmark = pytest.mark.graph

_PREFIX = "__write_local_placeholder__"


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


def _id(label: str) -> str:
    return f"{_PREFIX}{label}_{uuid.uuid4().hex}"


def _name(name_id: str, **relationships: str) -> dict:
    return {
        "id": name_id,
        "description": "Synthetic write-local placeholder cleanup quantity.",
        "kind": "scalar",
        "source_types": [],
        "source_id": None,
        "physics_domain": None,
        **relationships,
    }


def _exists(graph, label: str, node_id: str) -> bool:
    rows = graph.query(
        f"MATCH (node:{label} {{id: $id}}) RETURN count(node) AS count",
        id=node_id,
    )
    return bool(rows and rows[0]["count"])


def _cleanup(graph, node_ids: list[str], name_ids: list[str]) -> None:
    graph.query(
        """
        MATCH (change:StandardNameChange)
        WHERE change.from_name IN $name_ids
           OR change.to_name IN $name_ids
        DETACH DELETE change
        """,
        name_ids=name_ids,
    )
    graph.query(
        """
        MATCH (node)
        WHERE node.id IN $node_ids
        DETACH DELETE node
        """,
        node_ids=node_ids,
    )


def test_write_sweeps_only_its_unowned_placeholder_endpoints(_graph) -> None:
    unrelated = _id("unrelated")
    touched = _id("touched")
    accepted = _id("accepted")
    drafted = _id("drafted")
    reviewed = _id("reviewed")
    source_backed = _id("source_backed")
    source_id = _id("source")
    structural_parent = _id("structural_parent")
    closure_parent = _id("closure_parent")
    error_target = _id("error_target")

    written_touched = _id("written_touched")
    written_accepted = _id("written_accepted")
    written_drafted = _id("written_drafted")
    written_reviewed = _id("written_reviewed")
    written_source = _id("written_source")
    written_parent = _id("written_parent")
    written_error = _id("written_error")
    written_ids = [
        written_touched,
        written_accepted,
        written_drafted,
        written_reviewed,
        written_source,
        written_parent,
        written_error,
    ]
    target_ids = [
        unrelated,
        touched,
        accepted,
        drafted,
        reviewed,
        source_backed,
        structural_parent,
        closure_parent,
        error_target,
    ]
    name_ids = [*written_ids, *target_ids]
    node_ids = [*name_ids, source_id]

    edges = {
        written_parent: [
            DerivedEdge(
                edge_type="HAS_PARENT",
                from_name=written_parent,
                to_name=structural_parent,
                props={"operator_kind": "binary"},
            )
        ],
        structural_parent: [
            DerivedEdge(
                edge_type="HAS_PARENT",
                from_name=structural_parent,
                to_name=closure_parent,
                props={"operator_kind": "unary_prefix"},
            )
        ],
        written_error: [
            DerivedEdge(
                edge_type="HAS_ERROR",
                from_name=written_error,
                to_name=error_target,
                props={"error_type": "upper"},
            )
        ],
    }
    names = [
        _name(written_touched, predecessor=touched),
        _name(written_accepted, predecessor=accepted),
        _name(written_drafted, predecessor=drafted),
        _name(written_reviewed, predecessor=reviewed),
        _name(written_source, predecessor=source_backed),
        _name(written_parent),
        _name(written_error),
    ]

    try:
        _graph.query(
            """
            UNWIND $ids AS id
            CREATE (:StandardName {id: id})
            """,
            ids=[unrelated, touched, accepted, drafted, reviewed, source_backed],
        )
        _graph.query(
            """
            MATCH (accepted:StandardName {id: $accepted})
            MATCH (drafted:StandardName {id: $drafted})
            MATCH (reviewed:StandardName {id: $reviewed})
            SET accepted.name_stage = 'accepted',
                drafted.name_stage = 'drafted',
                reviewed.name_stage = 'reviewed'
            """,
            accepted=accepted,
            drafted=drafted,
            reviewed=reviewed,
        )
        _graph.query(
            """
            MATCH (target:StandardName {id: $target})
            CREATE (source:StandardNameSource {
                id: $source_id,
                produced_sn_id: $target
            })-[:PRODUCED_NAME]->(target)
            """,
            target=source_backed,
            source_id=source_id,
        )

        with (
            patch(
                "imas_codex.standard_names.protection.filter_protected",
                side_effect=lambda records, **_kwargs: (records, []),
            ),
            patch(
                "imas_standard_names.grammar.parse",
                side_effect=lambda name, **_kwargs: SimpleNamespace(ir=name),
            ),
            patch("imas_standard_names.grammar.compose", side_effect=lambda ir: ir),
            patch(
                "imas_codex.standard_names.graph_ops._parse_grammar",
                return_value={},
            ),
            patch(
                "imas_codex.standard_names.graph_ops._write_grammar_decomposition",
                return_value=[],
            ),
            patch(
                "imas_codex.standard_names.derivation.derive_edges",
                side_effect=lambda name: edges.get(name, []),
            ),
            patch(
                "imas_codex.standard_names.graph_ops._filter_admissible_parents",
                side_effect=lambda batch, _client, **_kwargs: batch,
            ),
        ):
            assert write_standard_names(names, gc=_graph) == len(names)

        assert _exists(_graph, "StandardName", unrelated)
        assert not _exists(_graph, "StandardName", touched)
        for durable in (accepted, drafted, reviewed, source_backed):
            assert _exists(_graph, "StandardName", durable)
        for structural in (structural_parent, closure_parent, error_target):
            assert _exists(_graph, "StandardName", structural)

        event_rows = _graph.query(
            """
            MATCH (change:StandardNameChange {
                from_name: $name,
                operation: 'remove_skeleton_placeholder'
            })
            RETURN count(change) AS count
            """,
            name=touched,
        )
        assert event_rows[0]["count"] == 1

        edge_rows = _graph.query(
            """
            MATCH (:StandardName {id: $child})-[:HAS_PARENT]->
                  (:StandardName {id: $parent})-[:HAS_PARENT]->
                  (:StandardName {id: $closure})
            MATCH (:StandardName {id: $error_source})-[:HAS_ERROR]->
                  (:StandardName {id: $error_target})
            RETURN count(*) AS count
            """,
            child=written_parent,
            parent=structural_parent,
            closure=closure_parent,
            error_source=written_error,
            error_target=error_target,
        )
        assert edge_rows[0]["count"] == 1
    finally:
        _cleanup(_graph, node_ids, name_ids)
        residue = _graph.query(
            """
            MATCH (node)
            WHERE node.id IN $node_ids
            RETURN count(node) AS count
            """,
            node_ids=node_ids,
        )
        changes = _graph.query(
            """
            MATCH (change:StandardNameChange)
            WHERE change.from_name IN $name_ids
               OR change.to_name IN $name_ids
            RETURN count(change) AS count
            """,
            name_ids=name_ids,
        )
        assert residue[0]["count"] == 0
        assert changes[0]["count"] == 0
