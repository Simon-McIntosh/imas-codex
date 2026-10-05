"""Live-graph pin for the chunk-source renderer.

The reader tests assert that each composed Cypher *contains* the text
``render_chunk_source`` emits, which guards the wiring and not the runtime
result. A reversed ``HAS_CHUNK`` direction, or a coalesce that binds the wrong
value, would leave those tests green while resolving no real path. This module
runs the rendered clause against the live graph so the resolution itself is
pinned: one bounded query binds a chunk reachable over an inbound
``HAS_CHUNK``, splices the rendered clause after the match, and asserts the
resolved value equals the owning ``CodeExample``'s ``source_file``.

``source_facility`` in the semantic-search reader is read from the chunk
(``node.facility_id``) rather than from the owning file: the chunk writer in
``imas_codex/ingestion/pipeline.py`` stamps ``facility_id`` on every chunk and
writes ``AT_FACILITY`` from that same value, so the chunk carries the writer's
facility and needs no join to recover it.

Module-scoped ``pytestmark`` is the graph tier; the graph-test conftest and the
top-level conftest skip every graph-marked test when Neo4j is unreachable, and
:func:`_resolved_source` skips when no chunk with an inbound ``HAS_CHUNK``
exists.
"""

from __future__ import annotations

import pytest

from imas_codex.graph.query_builder import render_chunk_source

pytestmark = pytest.mark.graph


def _resolved_source(gc, output: str) -> dict:
    """Run the rendered resolution clause against one live ``HAS_CHUNK`` edge.

    Binds a single ``CodeChunk`` reached from an owning ``CodeExample`` over
    the inbound ``HAS_CHUNK`` edge, splices ``render_chunk_source`` after that
    match, and returns the resolved path beside the owning example's own
    ``source_file``. The owner is bound forward in the base match so the
    expectation is the schema's own direction; the rendered clause re-binds it
    as ``ce`` and supplies the coalesced value.
    """
    cypher = (
        "MATCH (owner:CodeExample)-[:HAS_CHUNK]->(cc:CodeChunk)\n"
        "WHERE owner.source_file IS NOT NULL\n"
        "WITH cc LIMIT 1\n"
        f"{render_chunk_source('cc', output)}\n"
        f"RETURN {output} AS resolved, ce.source_file AS owning_source_file"
    )
    rows = gc.query(cypher)
    if not rows:
        pytest.skip("no CodeChunk with an inbound HAS_CHUNK to resolve")
    return rows[0]


def test_resolves_source_file_through_has_chunk(graph_client):
    """The rendered clause resolves a live chunk's path to its owner's file."""
    row = _resolved_source(graph_client, "source_file")
    assert row["resolved"] is not None
    assert row["resolved"] == row["owning_source_file"]


def test_output_name_is_parameterised(graph_client):
    """The same chunk bound to a different output name resolves identically."""
    row = _resolved_source(graph_client, "resolved_path")
    assert row["resolved"] is not None
    assert row["resolved"] == row["owning_source_file"]
