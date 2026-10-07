"""Re-ingesting a code file writes one example, not a second one.

The example id is derived from the file's content, so a re-ingest of unchanged
content lands on the same id and must merge onto the example already there.  The
``_FakeGraph`` below reproduces the contract the real graph enforces: creating a
``CodeExample`` whose id already exists raises the uniqueness violation.  The
pipeline never calls ``create_nodes`` for the example, so a regression to a plain
create fails here with that violation rather than in production.
"""

from __future__ import annotations

import asyncio

import pytest

from imas_codex.ingestion.pipeline import ingest_files

FACILITY = "jt-60sa"
PATH = "/analysis/src/eqdb_io.py"
CONTENT = "def read_eq(shot):\n    return shot\n"


class _UniquenessViolation(RuntimeError):
    """The error the graph raises when a created node's id already exists."""


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
    """A graph that enforces the CodeExample uniqueness constraint on create."""

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
            for item in items:
                if item["id"] in self.examples:
                    raise _UniquenessViolation(
                        "Node already exists with label CodeExample and id "
                        f"'{item['id']}'"
                    )
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


def test_reingest_merges_onto_the_existing_example(fake_graph):
    """A second ingest of unchanged content must not raise a uniqueness error."""
    first = asyncio.run(ingest_files(FACILITY, [PATH]))
    second = asyncio.run(ingest_files(FACILITY, [PATH]))

    # A plain create would raise the uniqueness violation on the second
    # ingest; this is the assertion the negative control fails.
    assert "Node already exists" not in str(
        second["outcomes"][PATH].get("reason", "")
    ), second["outcomes"][PATH]
    assert first["outcomes"][PATH]["status"] == "ingested"
    assert second["outcomes"][PATH]["status"] == "ingested"
    # One example for the file, not two.
    assert len(fake_graph.examples) == 1
    # The write is a merge, which is what makes the second ingest safe.
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
