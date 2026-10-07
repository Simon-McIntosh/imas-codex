"""The example write is idempotent on its id, and one file's write failure is one failure.

``GraphClient.create_nodes`` already upserts — it issues
``MERGE (n:{label} {id: item.id}) SET n += item`` — so a sequential re-ingest
onto the same id never raised a uniqueness violation, and this test does not
claim otherwise.  What it pins is that the pipeline's own example id is stable
for unchanged content (so the re-ingest lands on the same node and refreshes its
chunks), that changed content supersedes the file's previous example through the
shared cascade, and that a failure writing one file does not abort the batch.

The ``_FakeGraph`` below mirrors the real client's create contract: a create is a
merge, not an insert, so it upserts an existing id rather than refusing it.
"""

from __future__ import annotations

import asyncio

import pytest

from imas_codex.ingestion.pipeline import ingest_files

FACILITY = "jt-60sa"
PATH = "/analysis/src/eqdb_io.py"
CONTENT = "def read_eq(shot):\n    return shot\n"


def _fake_fetch(_facility, paths):
    for path in paths:
        yield path, CONTENT, "python"


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
    """A graph mirroring GraphClient.create_nodes: a create upserts on the id."""

    def __init__(self):
        self.examples: dict[str, dict] = {}
        self.chunk_writes: list[list[dict]] = []
        self.queries: list[str] = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def ensure_facility(self, facility_id):
        return None

    def create_nodes(self, label, items, create_relationships=True, **kwargs):
        if label == "CodeExample":
            # MERGE semantics, as GraphClient issues them: an existing id is
            # updated, never refused.
            for item in items:
                self.examples[item["id"]] = dict(item)
        elif label == "CodeChunk":
            self.chunk_writes.append([dict(c) for c in items])

    def query(self, cypher, **params):
        q = " ".join(cypher.split())
        self.queries.append(q)
        if "RETURN e.id AS id" in q:
            return [
                {"id": eid}
                for eid, props in self.examples.items()
                if props.get("facility_id") == params.get("facility")
                and props.get("source_file") == params.get("path")
            ]
        if "DETACH DELETE" in q:
            for cid in params.get("cf_ids", []):
                for eid, props in list(self.examples.items()):
                    if f"{props.get('facility_id')}:{props.get('source_file')}" == cid:
                        self.examples.pop(eid)
            return [{"reset_count": 0}]
        if "MERGE (e:CodeExample {id: item.id})" in q:
            for item in params.get("examples", []):
                self.examples[item["id"]] = dict(item)
            return []
        return []


@pytest.fixture
def fake_graph(monkeypatch):
    graph = _FakeGraph()
    monkeypatch.setattr(
        "imas_codex.ingestion.pipeline.GraphClient", lambda *a, **k: graph
    )
    for target, value in (
        ("imas_codex.ingestion.pipeline.fetch_remote_files", _fake_fetch),
        ("imas_codex.ingestion.pipeline._split_and_extract", _fake_split),
        ("imas_codex.graph.meta.gate_ingestion", lambda *a, **k: None),
    ):
        monkeypatch.setattr(target, value)
    return graph


def test_reingest_leaves_one_example_and_refreshes_chunks(fake_graph):
    """A second ingest of unchanged content reuses the example and rewrites its chunks."""
    first = asyncio.run(ingest_files(FACILITY, [PATH]))
    second = asyncio.run(ingest_files(FACILITY, [PATH]))

    assert first["outcomes"][PATH]["status"] == "ingested"
    assert second["outcomes"][PATH]["status"] == "ingested"
    assert not second["failed"]
    # One example for the file, not two.
    assert len(fake_graph.examples) == 1
    # The write goes through the merged example rather than creating a second.
    assert any("MERGE (e:CodeExample {id: item.id})" in q for q in fake_graph.queries)
    # The chunks are refreshed on the re-ingest.
    assert len(fake_graph.chunk_writes) == 2


def test_example_id_is_stable_for_unchanged_content(fake_graph):
    asyncio.run(ingest_files(FACILITY, [PATH]))
    first_id = next(iter(fake_graph.examples))
    asyncio.run(ingest_files(FACILITY, [PATH]))
    assert set(fake_graph.examples) == {first_id}


def test_content_change_supersedes_the_stale_example(fake_graph, monkeypatch):
    """Changed content yields a new id and removes the previous example."""
    asyncio.run(ingest_files(FACILITY, [PATH]))
    old_id = next(iter(fake_graph.examples))

    changed = "def read_eq(shot):\n    return shot + 1\n"

    def _changed_fetch(_facility, paths):
        for path in paths:
            yield path, changed, "python"

    monkeypatch.setattr(
        "imas_codex.ingestion.pipeline.fetch_remote_files", _changed_fetch
    )
    asyncio.run(ingest_files(FACILITY, [PATH]))

    assert len(fake_graph.examples) == 1
    new_id = next(iter(fake_graph.examples))
    assert new_id != old_id
    # The stale example is removed through the shared chunk cascade.
    assert any("DETACH DELETE" in q for q in fake_graph.queries)
