"""Every admitted code file reaches a terminal state, and one bad file is one bad file.

Three behaviours are pinned here: an ingest batch records each file's own
outcome so a single write failure fails only that file; an admitted file whose
extraction yields no chunk is skipped with a reason rather than waiting at
``scored``; and an admitted file the claim can never reach — one above the
claim's line-count ceiling — is settled as skipped.  The reset that returns a
failed file to ``scored`` is checked alongside them.
"""

from __future__ import annotations

import asyncio

import pytest

from imas_codex.discovery.base.reset import CODE_RESET_SPECS, reset_to_status
from imas_codex.discovery.code.workers import (
    _mark_file_skipped,
    _settle_unclaimable_files,
)
from imas_codex.ingestion.pipeline import ingest_files

FACILITY = "jt-60sa"
GOOD_A = "/analysis/src/eqdb_io.py"
GOOD_B = "/analysis/src/mdbget.py"
BAD = "/analysis/src/setvtcx.py"
CONTENTS = {
    GOOD_A: "def a(shot):\n    return shot\n",
    GOOD_B: "def b(shot):\n    return shot\n",
    BAD: "def c(shot):\n    return shot\n",
}


def _fake_fetch(_facility, paths):
    for path in paths:
        yield path, CONTENTS[path], "python"


def _fake_split(content, language, metadata, use_text_splitter=False):
    return [
        {
            "text": content,
            "start_line": 1,
            "end_line": len(content.splitlines()),
            "source_file": metadata["source_file"],
            "facility_id": metadata["facility_id"],
            "language": language,
            "code_example_id": metadata["code_example_id"],
            "related_ids": [],
            "mdsplus_paths": [],
        }
    ]


class _FakeGraph:
    """A graph that fails the write for one file only."""

    def __init__(self, failing_path: str | None = None):
        self.failing_path = failing_path
        self.queries: list[str] = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def ensure_facility(self, facility_id):
        return None

    def create_nodes(self, label, items, create_relationships=True, **kwargs):
        if label == "CodeChunk":
            for item in items:
                if item.get("source_file") == self.failing_path:
                    raise RuntimeError("write failed for this file")

    def query(self, cypher, **params):
        self.queries.append(" ".join(cypher.split()))
        return []


def _patch_pipeline(monkeypatch, graph):
    monkeypatch.setattr(
        "imas_codex.ingestion.pipeline.GraphClient", lambda *a, **k: graph
    )
    monkeypatch.setattr("imas_codex.ingestion.pipeline.fetch_remote_files", _fake_fetch)
    monkeypatch.setattr("imas_codex.graph.meta.gate_ingestion", lambda *a, **k: None)


def test_one_bad_file_does_not_fail_the_batch(monkeypatch):
    graph = _FakeGraph(failing_path=BAD)
    _patch_pipeline(monkeypatch, graph)
    monkeypatch.setattr("imas_codex.ingestion.pipeline._split_and_extract", _fake_split)

    stats = asyncio.run(ingest_files(FACILITY, [GOOD_A, BAD, GOOD_B]))

    outcomes = stats["outcomes"]
    assert outcomes[GOOD_A]["status"] == "ingested"
    assert outcomes[GOOD_B]["status"] == "ingested"
    assert outcomes[BAD]["status"] == "failed"
    assert outcomes[BAD]["reason"]
    assert set(stats["failed"]) == {BAD}


def test_chunkless_file_is_skipped_with_a_reason(monkeypatch):
    graph = _FakeGraph()
    _patch_pipeline(monkeypatch, graph)

    def _some_chunks(content, language, metadata, use_text_splitter=False):
        if metadata["source_file"] == BAD:
            return []
        return _fake_split(content, language, metadata)

    monkeypatch.setattr(
        "imas_codex.ingestion.pipeline._split_and_extract", _some_chunks
    )

    stats = asyncio.run(ingest_files(FACILITY, [GOOD_A, BAD]))

    assert stats["outcomes"][BAD] == {
        "status": "skipped",
        "reason": "no chunks extracted",
    }
    assert stats["skipped_files"] == {BAD: "no chunks extracted"}


def test_extraction_failure_is_recorded_per_file(monkeypatch):
    graph = _FakeGraph()
    _patch_pipeline(monkeypatch, graph)

    def _boom(content, language, metadata, use_text_splitter=False):
        if metadata["source_file"] == BAD:
            raise ValueError("cannot parse")
        return _fake_split(content, language, metadata)

    monkeypatch.setattr("imas_codex.ingestion.pipeline._split_and_extract", _boom)

    stats = asyncio.run(ingest_files(FACILITY, [GOOD_A, BAD]))

    assert stats["outcomes"][GOOD_A]["status"] == "ingested"
    outcome = stats["outcomes"][BAD]
    assert outcome["status"] == "failed"
    assert "cannot parse" in outcome["reason"]


class _CapturingGraph:
    def __init__(self, rows=None):
        self.queries: list[str] = []
        self.params: list[dict] = []
        self._rows = rows if rows is not None else [{"settled": 2, "reset_count": 2}]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def query(self, cypher, **kwargs):
        self.queries.append(" ".join(cypher.split()))
        self.params.append(kwargs)
        return self._rows


@pytest.fixture
def captured(monkeypatch):
    graph = _CapturingGraph()
    monkeypatch.setattr("imas_codex.graph.GraphClient", lambda *a, **k: graph)
    return graph


def test_mark_file_skipped_writes_status_and_reason(captured):
    _mark_file_skipped("jt-60sa:/a/b.py", "no chunks extracted")

    (query,) = captured.queries
    assert "SET sf.status = 'skipped'" in query
    assert "sf.skip_reason = $reason" in query
    assert captured.params[0]["reason"] == "no chunks extracted"


def test_settle_unclaimable_files_names_the_line_ceiling(captured):
    count = _settle_unclaimable_files(FACILITY, 10000)

    (query,) = captured.queries
    # The same ceiling the claim applies, in the opposite direction: the claim
    # admits ``<=`` and this settles the ``>`` rows it can never reach.
    assert "coalesce(sf.line_count, 0) > $max_line_count" in query
    assert "SET sf.status = 'skipped'" in query
    assert "sf.skip_reason = 'exceeds max_line_count'" in query
    assert captured.params[0]["max_line_count"] == 10000
    assert count == 2


def test_claim_predicate_excludes_the_oversized_rows(captured):
    """The claim's own predicate is what leaves an oversized file unsettled."""
    from imas_codex.discovery.code.workers import _claim_code_files_for_ingestion

    # The claim reads its rows back by token; the fake returns nothing, so this
    # asserts the rendered predicate only.
    _claim_code_files_for_ingestion(FACILITY, limit=10)
    claim_query = captured.queries[0]
    assert "coalesce(sf.line_count, 0) <= $max_line_count" in claim_query
    assert captured.params[0]["max_line_count"] == 10000


def test_failed_file_resets_to_scored(captured):
    count = reset_to_status(CODE_RESET_SPECS["scored"], FACILITY)

    (query,) = captured.queries
    assert "SET n.status = $target_status" in query
    assert "n.error = null" in query
    assert "n.skip_reason = null" in query
    assert "failed" in captured.params[0]["source_statuses"]
    assert count == 2


def test_scored_spec_admits_failed_files():
    spec = CODE_RESET_SPECS["scored"]
    assert "failed" in spec.source_statuses
    assert "ingested" in spec.source_statuses
    assert "error" in spec.clear_fields
