"""Every reader of a name's sources answers from the reader that owns them.

The graph holds a name's sources twice: as ``PRODUCED_NAME`` edges from the
source nodes, and as a ``source_paths`` list copied onto the name. The edges are
the authority, so a name whose copied list is empty while its edges are not must
still answer with its sources, and a name whose copied list names a path no edge
gives must answer with none.

The planted graph holds exactly the eight shapes the sourceless count has to
separate: an empty copy with no edge, a populated copy with no edge, a derived
name with neither, a superseded status, a superseded stage, a derived parent
whose only producer is a derived record, a data-dictionary producer with an
empty copy, and a name whose only producer is stale.
"""

from __future__ import annotations

import ast
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

#: The frozen-extraction records a committed corpus artifact is pinned to. Each
#: names the query that produced that artifact; it is kept verbatim so the
#: artifact can be checked against it, and reads nothing at runtime. They are
#: listed by name rather than recognised by suffix, because a live reader that
#: happens to sit in a constant named for a record is exactly what the guard
#: must still refuse.
FROZEN_EXTRACTION_RECORDS = frozenset({"_ADVISORY_CORPUS_EXTRACTION_RECORD"})


def _without_extraction_records(text: str) -> str:
    """Drop the frozen-extraction records a corpus is pinned to.

    Such a record is the query that produced a committed corpus artifact, kept
    verbatim so the artifact can be checked against it; it reads nothing at
    runtime. Everything else in the frozen modules is a live reader.
    """
    skipped: set[int] = set()
    for node in ast.parse(text).body:
        if not isinstance(node, ast.Assign):
            continue
        named = any(
            isinstance(target, ast.Name) and target.id in FROZEN_EXTRACTION_RECORDS
            for target in node.targets
        )
        if named:
            skipped.update(range(node.lineno, (node.end_lineno or node.lineno) + 1))
    return "\n".join(
        line
        for number, line in enumerate(text.splitlines(), 1)
        if number not in skipped
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
           origin:'pipeline', paths:[]},
          {id:'reads_stale_source', status:'draft', stage:'accepted',
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
        """MATCH (holder:StandardName {id:'reads_stale_source'})
        CREATE (source:StandardNameSource {id:'dd:reads/retired', status:'stale'})
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
def test_sourceless_count_reads_stale_as_no_producer(
    planted_graph: GraphClient,
) -> None:
    """Eight planted names: six live, two produced by a live source, four not,
    of which the derived parent is listed apart. The name whose only producer
    is stale counts here and does not count in the bare population."""
    report = count_live_names_without_source(planted_graph)

    assert report["live_names"] == 6
    assert report["live_with_producer"] == 2
    assert report["live_without_producer"] == 4
    assert report["count"] == 3
    assert [entry["id"] for entry in report["names"]] == [
        "reads_empty_list",
        "reads_stale_list",
        "reads_stale_source",
    ]
    assert [entry["has_edge"] for entry in report["names"]] == [False, False, True]
    assert [entry["id"] for entry in report["exempt_derived_parents"]] == [
        "reads_derived_orphan"
    ]

    assert report["bare_count"] == 2
    assert report["bare_names"] == ["reads_empty_list", "reads_stale_list"]
    assert report["differing_names"] == ["reads_stale_source"]


_COPIED_LIST = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\.source_paths\b")


@pytest.mark.parametrize("module", READER_MODULES)
def test_each_reader_reads_sources_from_the_one_place(module: str) -> None:
    """A reader that spells the copied list, or the edge pattern, itself fails
    here: sources come from the projection that owns them."""
    package = Path(__file__).resolve().parents[2] / "imas_codex" / "standard_names"
    path = package / module
    assert path.is_file(), f"reader module is missing: {module}"
    text = _without_extraction_records(path.read_text(encoding="utf-8"))

    assert "source_paths_projection" in text, (
        f"{module} does not name the one place that reads a name's sources"
    )
    copied = _COPIED_LIST.search(text)
    assert copied is None, (
        f"{module} reads the copied list: {copied.group(0) if copied else ''}"
    )
    assert "PRODUCED_NAME" not in text, (
        f"{module} spells the producer pattern itself instead of asking "
        "source_paths_projection()"
    )


def test_the_review_pipeline_reads_sources_from_the_one_place() -> None:
    """The review pipeline loads the catalog with its own query, so its
    projection is checked where it sits: it must ask the one place too."""
    cli = Path(__file__).resolve().parents[2] / "imas_codex" / "cli" / "sn.py"
    tree = ast.parse(cli.read_text(encoding="utf-8"))
    load_catalog = next(
        (
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "_load_catalog"
        ),
        None,
    )
    assert load_catalog is not None, "the review pipeline no longer loads a catalog"
    body = ast.get_source_segment(cli.read_text(encoding="utf-8"), load_catalog) or ""

    assert "source_paths_projection" in body, (
        "_load_catalog does not name the one place that reads a name's sources"
    )
    copied = _COPIED_LIST.search(body)
    assert copied is None, (
        f"_load_catalog reads the copied list: {copied.group(0) if copied else ''}"
    )


def test_a_reader_in_an_unlisted_record_constant_is_still_refused() -> None:
    """Only a record on the explicit list is skipped: a constant that merely
    ends the sanctioned suffix is a live reader and the guard refuses it."""
    source = (
        '_ADVISORY_CORPUS_EXTRACTION_RECORD = "RETURN sn.source_paths"\n'
        "_UNREVIEWED_CORPUS_EXTRACTION_RECORD = (\n"
        '    "MATCH (sn:StandardName)\\n"\n'
        '    "RETURN sn.source_paths AS source_paths"\n'
        ")\n"
    )
    text = _without_extraction_records(source)

    assert "sn.source_paths" not in text.split("_UNREVIEWED")[0]
    copied = _COPIED_LIST.search(text)
    assert copied is not None, (
        "the unlisted record was skipped although it reads the copied list"
    )
    assert copied.group(0) == "sn.source_paths"
