"""Every reader of a name's sources answers from the reader that owns them.

The graph holds a name's sources twice: as ``PRODUCED_NAME`` edges from the
source nodes, and as a ``source_paths`` list copied onto the name. The edges are
the authority, so a name whose copied list is empty while its edges are not must
still answer with its sources, and a name whose copied list is populated while
its edges are gone must answer with none.

The planted graph also carries two support names and one superseded name, so the
sourceless count is exercised without counting a superseded name and without
counting a derived parent as a loss.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest

from imas_codex.graph.client import GraphClient
from imas_codex.standard_names.search import (
    fetch_standard_names,
    search_standard_names,
)
from imas_codex.standard_names.source_reads import (
    count_live_names_without_source,
    read_source_paths,
)

READER_MODULES = (
    "search.py",
    "benchmark_roles.py",
    "judgment_goldens.py",
)


@pytest.fixture()
def planted_graph(disposable_neo4j_uri: str) -> Iterator[GraphClient]:
    """Plant seven names: edge-only, list-only, agreeing, derived parent,
    child, superseded and sourceless."""
    client = GraphClient(
        uri=disposable_neo4j_uri,
        username="neo4j",
        password=os.environ.get("IMAS_CODEX_TEST_NEO4J_PASSWORD", ""),
        graph_name="source-reads-fixture",
    )
    client.query(
        """UNWIND [
          {id:'reads_edge_only', stage:'accepted', origin:'pipeline',
           status:'draft', paths:[]},
          {id:'reads_list_only', stage:'accepted', origin:'pipeline',
           status:'draft', paths:['dd:reads/ghost']},
          {id:'reads_agrees', stage:'accepted', origin:'pipeline',
           status:'draft', paths:['dd:reads/agrees']},
          {id:'reads_derived_parent', stage:'accepted', origin:'derived',
           status:'draft', paths:[]},
          {id:'reads_child', stage:'accepted', origin:'pipeline',
           status:'draft', paths:[]},
          {id:'reads_superseded', stage:'accepted', origin:'pipeline',
           status:'superseded', paths:['dd:reads/ghost']},
          {id:'reads_sourceless', stage:'accepted', origin:'pipeline',
           status:'draft', paths:[]}
        ] AS row
        CREATE (:StandardName {id:row.id, name_stage:row.stage,
          origin:row.origin, source_paths:row.paths, status:row.status})"""
    )
    client.query(
        """UNWIND [
          {id:'dd:reads/edge_only', holder:'reads_edge_only'},
          {id:'dd:reads/agrees', holder:'reads_agrees'},
          {id:'dd:reads/child', holder:'reads_child'}
        ] AS row
        CREATE (source:StandardNameSource {id:row.id, source_id:row.id})
        WITH source, row
        MATCH (holder:StandardName {id:row.holder})
        CREATE (source)-[:PRODUCED_NAME]->(holder)"""
    )
    client.query(
        """MATCH (parent:StandardName {id:'reads_derived_parent'}),
                 (child:StandardName {id:'reads_child'})
        CREATE (child)-[:HAS_PARENT]->(parent)"""
    )
    try:
        yield client
    finally:
        client.query(
            "MATCH (node) WHERE node.id STARTS WITH 'reads_' "
            "OR node.id STARTS WITH 'dd:reads/' DETACH DELETE node"
        )
        client.close()


@pytest.mark.graph
def test_reader_answers_from_the_producer_edges(planted_graph: GraphClient) -> None:
    """An empty copied list does not hide a name's sources; a stale one does
    not invent them."""
    assert read_source_paths(planted_graph, "reads_edge_only") == ["dd:reads/edge_only"]
    assert read_source_paths(planted_graph, "reads_list_only") == []
    assert read_source_paths(planted_graph, "reads_agrees") == ["dd:reads/agrees"]


@pytest.mark.graph
def test_name_search_answers_from_the_producer_edges(
    planted_graph: GraphClient,
) -> None:
    """Search is a reader of the same place: the planted empty-list name comes
    back with its source and the planted stale-list name comes back with none."""
    rows = fetch_standard_names(
        ["reads_edge_only", "reads_list_only"],
        include_grammar=False,
        include_documentation=False,
        gc=planted_graph,
    )
    by_name = {row["name"]: row for row in rows}
    assert by_name["reads_edge_only"]["source_paths"] == ["dd:reads/edge_only"]
    assert by_name["reads_list_only"]["source_paths"] == []

    found = search_standard_names("reads_edge_only", k=5, gc=planted_graph)
    search_rows = {row["name"]: row for row in found}
    assert search_rows["reads_edge_only"]["source_paths"] == ["dd:reads/edge_only"]


@pytest.mark.graph
def test_sourceless_count_lists_every_unproduced_name_but_one(
    planted_graph: GraphClient,
) -> None:
    """Seven planted names yield two counted names and one derived parent
    listed apart, while the raw edge read counts all three."""
    report = count_live_names_without_source(planted_graph)

    assert report["live_names"] == 6
    assert report["live_with_producer"] == 3
    assert report["live_without_producer"] == 3
    assert report["count"] == 2
    assert [entry["id"] for entry in report["names"]] == [
        "reads_list_only",
        "reads_sourceless",
    ]
    assert [entry["id"] for entry in report["exempt_derived_parents"]] == [
        "reads_derived_parent"
    ]


def test_no_reader_projects_the_copied_list() -> None:
    """Each reader names its sources through the one place, not the copy.

    A reader that spells ``sn.source_paths`` into its own ``RETURN`` answers
    from the projection rather than from the edges, which is the defect this
    module exists to prevent, so the spelling is refused at the source.
    """
    package = Path(__file__).parents[2] / "imas_codex" / "standard_names"
    for module in READER_MODULES:
        text = (package / module).read_text(encoding="utf-8")
        assert "sn.source_paths AS" not in text, module
        assert "source_paths_projection" in text, module
