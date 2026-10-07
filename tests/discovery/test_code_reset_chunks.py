"""The code reset removes the example and chunk nodes an ingested file carries.

Resetting a CodeFile below ``ingested`` must not leave the CodeExample and
CodeChunk nodes the ingestion stage wrote for it, or a later re-ingest doubles them.  The
rendered reset query deletes them through the same traversal
``clear_facility_code`` uses, scoped to the files being reset.  These tests read
the query the reset actually issues rather than a live database, so they pin the
rendered Cypher, not the graph's state.
"""

from __future__ import annotations

import pytest

from imas_codex.discovery.base import reset as reset_module
from imas_codex.discovery.base.reset import CODE_RESET_SPECS, reset_to_status
from imas_codex.discovery.code import scorer

FACILITY = "jt-60sa"

# The reset stages that drop a file below ``ingested`` and so must delete the
# nodes ingestion wrote for it.
CASCADING_STAGES = ("discovered", "triaged")
# A file already at ``scored`` has no ingested chunks to remove.
NON_CASCADING_STAGES = ("scored",)


class _CapturingGraph:
    """Record the query the reset renders and answer with a reset count."""

    def __init__(self, count: int = 3):
        self.queries: list[str] = []
        self._count = count

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def query(self, cypher, **kwargs):
        self.queries.append(" ".join(cypher.split()))
        return [{"reset_count": self._count}]


@pytest.fixture
def captured(monkeypatch):
    graph = _CapturingGraph()
    monkeypatch.setattr("imas_codex.graph.GraphClient", lambda: graph)
    return graph


@pytest.mark.parametrize("stage", CASCADING_STAGES)
def test_code_reset_deletes_examples_and_chunks(captured, stage):
    count = reset_to_status(CODE_RESET_SPECS[stage], FACILITY)

    (query,) = captured.queries
    # The traversal is the one ``clear_facility_code`` uses, so a chunk is
    # reached only through the file's own CodeExample.
    assert "(n)<-[:FROM_FILE]-(ce:CodeExample)-[:HAS_CHUNK]->(cc:CodeChunk)" in query
    assert "DETACH DELETE" in query
    # The reset count is a count of files, not of the examples removed, so the
    # example fan-out is collapsed back to one row per file.
    assert "WITH DISTINCT n" in query
    assert "RETURN count(n) AS reset_count" in query
    assert count == 3


@pytest.mark.parametrize("stage", NON_CASCADING_STAGES)
def test_code_reset_above_ingestion_leaves_chunks(captured, stage):
    reset_to_status(CODE_RESET_SPECS[stage], FACILITY)

    (query,) = captured.queries
    assert "CodeChunk" not in query
    assert "DETACH DELETE" not in query


def test_code_reset_clear_fields_come_from_the_scorer_registry():
    """The fields a reset clears are the fields the decision arms write."""
    # Resetting to ``discovered`` clears the relevance arm's fields; resetting
    # to ``triaged`` keeps the name-arm relevance it is returning to.
    cleared = set(CODE_RESET_SPECS["discovered"].clear_fields)
    assert set(reset_module.CODE_RELEVANCE_FIELDS).issubset(cleared)
    kept = set(CODE_RESET_SPECS["triaged"].clear_fields)
    assert set(reset_module.CODE_RELEVANCE_FIELDS).isdisjoint(kept)
    assert set(scorer.RELEVANCE_FIELDS) == set(reset_module.CODE_RELEVANCE_FIELDS)
