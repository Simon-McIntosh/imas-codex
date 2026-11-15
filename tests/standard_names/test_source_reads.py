"""Every reader of a name's sources answers from the reader that owns them.

The graph holds a name's sources twice: as ``PRODUCED_NAME`` edges from the
source nodes, and as a ``source_paths`` list copied onto the name. The edges are
the authority, so a name whose copied list is empty while its edges are not must
still answer with its sources, and a name whose copied list names a path no edge
gives must answer with none.

The planted graph holds exactly the seven shapes the sourceless count has to
separate: an empty copy with no edge, a populated copy with no edge, a derived
name with neither, a superseded status, a superseded stage, a derived parent
whose only producer is a derived record, and a data-dictionary producer with an
empty copy.
"""

from __future__ import annotations

import os
import re
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
    """Plant the seven source shapes the count and the readers must separate."""
    client = GraphClient(
        uri=disposable_neo4j_uri,
        username="neo4j",
        password=os.environ.get("IMAS_CODEX_TEST_NEO4J_PASSWORD", ""),
        graph_name="source-reads-fixture",
    )
    client.query(
        """UNWIND [
          {id:'reads_empty_list', status:'draft', stage:'accepted',
           origin:'pipeline', paths:[]},
          {id:'reads_stale_list', status:'draft', stage:'accepted',
           origin:'pipeline', paths:['dd:reads/ghost']},
          {id:'reads_derived_orphan', status:'draft', stage:'accepted',
           origin:'derived', paths:[]},
          {id:'reads_status_superseded', status:'superseded', stage:'accepted',
           origin:'pipeline', paths:[]},
          {id:'reads_stage_superseded', status:'draft', stage:'superseded',
           origin:'pipeline', paths:[]},
          {id:'reads_derived_sourced', status:'draft', stage:'accepted',
           origin:'derived', paths:[]},
          {id:'reads_dd_bound', status:'draft', stage:'accepted',
           origin:'pipeline', paths:[]}
        ] AS row
        CREATE (:StandardName {id:row.id, status:row.status,
          name_stage:row.stage, origin:row.origin, source_paths:row.paths})"""
    )
    client.query(
        """UNWIND [
          {id:'derived:reads/source', holder:'reads_derived_sourced'},
          {id:'dd:reads/bound', holder:'reads_dd_bound'}
        ] AS row
        CREATE (source:StandardNameSource {id:row.id, source_id:row.id})
        WITH source, row
        MATCH (holder:StandardName {id:row.holder})
        CREATE (source)-[:PRODUCED_NAME]->(holder)"""
    )
    client.query(
        """MATCH (orphan:StandardName {id:'reads_derived_orphan'}),
                 (sourced:StandardName {id:'reads_derived_sourced'}),
                 (stale:StandardName {id:'reads_stale_list'})
        CREATE (sourced)-[:HAS_PARENT]->(orphan)
        CREATE (stale)-[:HAS_PARENT]->(sourced)"""
    )
    try:
        yield client
    finally:
        client.query(
            "MATCH (node) WHERE node.id STARTS WITH 'reads_' "
            "OR node.id STARTS WITH 'dd:reads/' "
            "OR node.id STARTS WITH 'derived:reads/' DETACH DELETE node"
        )
        client.close()


@pytest.mark.graph
def test_reader_answers_from_the_producer_edges(planted_graph: GraphClient) -> None:
    """An empty copy does not hide a name's sources; a stale one does not
    invent them."""
    assert read_source_paths(planted_graph, "reads_dd_bound") == ["dd:reads/bound"]
    assert read_source_paths(planted_graph, "reads_stale_list") == []
    assert read_source_paths(planted_graph, "reads_empty_list") == []


@pytest.mark.graph
def test_name_search_answers_from_the_producer_edges(
    planted_graph: GraphClient,
) -> None:
    """Search is a reader of the same place: the planted producer comes back
    with its path and the planted over-reporting copy comes back with none."""
    rows = fetch_standard_names(
        ["reads_dd_bound", "reads_stale_list"],
        include_grammar=False,
        include_documentation=False,
        gc=planted_graph,
    )
    by_name = {row["name"]: row for row in rows}
    assert by_name["reads_dd_bound"]["source_paths"] == ["dd:reads/bound"]
    assert by_name["reads_stale_list"]["source_paths"] == []

    found = search_standard_names("reads_dd_bound", k=5, gc=planted_graph)
    search_rows = {row["name"]: row for row in found}
    assert search_rows["reads_dd_bound"]["source_paths"] == ["dd:reads/bound"]


@pytest.mark.graph
def test_sourceless_count_lists_two_of_three_and_exempts_the_third(
    planted_graph: GraphClient,
) -> None:
    """Seven planted names: five live, two produced, three not, of which the
    derived parent is listed apart rather than counted."""
    report = count_live_names_without_source(planted_graph)

    assert report["live_names"] == 5
    assert report["live_with_producer"] == 2
    assert report["live_without_producer"] == 3
    assert report["count"] == 2
    assert [entry["id"] for entry in report["names"]] == [
        "reads_empty_list",
        "reads_stale_list",
    ]
    assert [entry["id"] for entry in report["exempt_derived_parents"]] == [
        "reads_derived_orphan"
    ]


def test_each_reader_reads_sources_from_the_one_place() -> None:
    """A reader that spells the copied list, or the edge pattern, itself fails
    here: sources come from the projection that owns them."""
    package = Path(__file__).resolve().parents[2] / "imas_codex" / "standard_names"
    copied_list = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\.source_paths\b")

    for module in READER_MODULES:
        path = package / module
        assert path.is_file(), f"reader module is missing: {module}"
        text = path.read_text(encoding="utf-8")

        assert "source_paths_projection" in text, (
            f"{module} does not name the one place that reads a name's sources"
        )
        stale = copied_list.search(text)
        assert stale is None, (
            f"{module} reads the copied list: {stale.group(0) if stale else ''}"
        )
        assert "PRODUCED_NAME" not in text, (
            f"{module} spells the producer pattern itself instead of asking "
            "source_paths_projection()"
        )
