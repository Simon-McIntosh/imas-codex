"""Drive the ingested-file re-judge through the fake decisions seam.

The re-judge takes code files that sit at ``status='ingested'`` with a
content-stage relevance and no recorded facet answer.  Their text already lives
in the graph as CodeChunks, so the pass rebuilds each file's content state from
its own stored chunk text -- ordered by reading position and cut to the same
length the fetched path uses -- in place of a remote fetch, and asks the content
arm the same question set through the same seat.  The judgment fields are
written back in place; the file's status, its CodeChunk text and the example it
belongs to are left as they are.  Admission is never re-decided, so a file whose
new composite and facets both fall below their gates stays ``ingested`` and is
reported.

Every decision here travels the module's HTTP seam (``_apost_decisions``) with a
fake transport, so no test opens the live endpoint; the autouse guard in
``tests/conftest.py`` refuses a real request.  The graph is stubbed so the
assertions read the queries the re-judge actually issued rather than a live
database.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from imas_codex.discovery.base import llm

FACILITY = "jt-60sa"
COST = 1.0e-4
SCOPE = (
    "loads_diagnostic_data",
    "processes_diagnostic_signals",
    "describes_machine_or_diagnostics",
    "maps_to_imas",
)


def _answers(
    *nouls: float,
    role: str = "diagnostic_data_access",
    facets: tuple[float, float, float, float] = (4.0, 3.0, 2.0, 1.0),
) -> dict:
    """Answer set with the four scope nouls set positionally.

    Carries the content arm's graded relevance and four facet Scores as well;
    *facets* sets the raw Scores for (data access, signal processing, machine
    description, imas mapping).  The facet Scores are independent of the scope
    nouls, so a file whose composite is weak can still carry a strong facet.
    """
    data_score, signal_score, machine_score, imas_score = facets
    out = {
        name: {"type": "noul", "noul": value}
        for name, value in zip(SCOPE, nouls, strict=True)
    }
    out["is_simulation"] = {"type": "noul", "noul": 0.05}
    other = (
        "infrastructure_or_utility"
        if role != "infrastructure_or_utility"
        else "diagnostic_data_access"
    )
    out["role"] = {
        "type": "choice",
        "choice": role,
        "probabilities": {role: 0.9, other: 0.1},
        "confidence": 0.8,
    }
    out.update(
        {
            "relevance_grade": {
                "type": "score",
                "score": 3.0,
                "probabilities": {0: 0.05, 1: 0.05, 2: 0.1, 3: 0.5, 4: 0.3},
                "confidence": 0.7,
            },
            "data_access_depth": {
                "type": "score",
                "score": data_score,
                "probabilities": {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 1.0},
                "confidence": 0.9,
            },
            "signal_processing_depth": {
                "type": "score",
                "score": signal_score,
                "probabilities": {0: 0.0, 1: 0.0, 2: 0.0, 3: 1.0},
                "confidence": 0.6,
            },
            "machine_description_depth": {
                "type": "score",
                "score": machine_score,
                "probabilities": {0: 0.0, 1: 0.0, 2: 1.0, 3: 0.0},
                "confidence": 0.5,
            },
            "imas_mapping_depth": {
                "type": "score",
                "score": imas_score,
                "probabilities": {0: 0.0, 1: 1.0, 2: 0.0, 3: 0.0},
                "confidence": 0.4,
            },
        }
    )
    return out


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload
        self.status_code = 200
        self.text = json.dumps(payload) if not isinstance(payload, str) else payload

    def json(self):
        return self._payload


def _payload(answers: dict) -> dict:
    return {"answers": answers, "usage": {"cost": COST}, "model": "typesafe/jev-1.13"}


def _file(path: str, **extra) -> dict:
    row = {
        "id": path,
        "path": path,
        "facility_id": FACILITY,
        "language": "fortran",
        "parent_path_id": "dir",
        "parent_path": "/analysis/src",
        "parent_description": "analysis sources",
    }
    row.update(extra)
    return row


class _RejudgeGraph:
    """Serve the chunk read and the judgment write the re-judge issues.

    The claim itself is stubbed at the function seam, so this graph only sees
    the chunk traversal (``HAS_CHUNK``) and the ``UNWIND $items`` judgment
    write.  It records both so a test can read the text the content arm was
    shown and the fields the writer persisted.
    """

    def __init__(self, chunks_by_file: dict[str, list[dict]]):
        self.chunks_by_file = chunks_by_file
        self.queries: list[str] = []
        self.writes: list[tuple[str, dict]] = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def query(self, cypher, **kwargs):
        text = " ".join(cypher.split())
        self.queries.append(text)
        if "HAS_CHUNK" in text:
            rows: list[dict] = []
            for fid in kwargs["ids"]:
                for chunk in self.chunks_by_file.get(fid, []):
                    rows.append(
                        {
                            "file_id": fid,
                            "start_line": chunk["start_line"],
                            "text": chunk["text"],
                        }
                    )
            return rows
        if "UNWIND $items AS item" in text:
            self.writes.append((text, kwargs))
            return []
        return []


def _run_rejudge(
    monkeypatch,
    files: list[dict],
    chunks_by_file: dict[str, list[dict]],
    answers_by_path: dict,
    *,
    requests: list | None = None,
):
    """Run the re-judge once over a single stubbed batch."""
    graph = _RejudgeGraph(chunks_by_file)
    released: list[str] = []
    claim_calls: list[dict] = []
    counter = {"n": 0}

    def claim_once(*args, **kwargs):
        claim_calls.append(kwargs)
        counter["n"] += 1
        return list(files) if counter["n"] == 1 else []

    async def fake_post(headers, body, timeout):
        if requests is not None:
            requests.append(body)
        path = body["state"]["file"]["path"]
        return _FakeResponse(_payload(answers_by_path[path]))

    monkeypatch.setenv("OPENROUTER_API_KEY_IMAS_CODEX", "test-key")
    monkeypatch.setattr(
        "imas_codex.discovery.base.facility.get_facility", lambda facility: {}
    )
    monkeypatch.setattr("imas_codex.settings.get_model", lambda section: "fake-model")
    monkeypatch.setattr("imas_codex.settings.get_code_ingest_threshold", lambda: 0.6)
    monkeypatch.setattr(
        "imas_codex.settings.get_code_facet_admission_threshold", lambda: 0.8
    )
    monkeypatch.setattr(
        "imas_codex.discovery.code.graph_ops.claim_files_for_scoring", claim_once
    )
    monkeypatch.setattr(
        "imas_codex.discovery.code.graph_ops.release_file_score_claims",
        lambda ids: released.extend(ids),
    )
    monkeypatch.setattr(
        "imas_codex.discovery.code.graph_ops.GraphClient", lambda: graph
    )
    monkeypatch.setattr("imas_codex.discovery.code.scorer.GraphClient", lambda: graph)
    monkeypatch.setattr(llm, "_apost_decisions", fake_post)

    from imas_codex.discovery.code.workers import rejudge_ingested_files

    result = asyncio.run(rejudge_ingested_files(FACILITY))
    return graph, result, claim_calls, released


# ---------------------------------------------------------------------------
# The re-judge presents stored chunk text and rewrites the judgment in place
# ---------------------------------------------------------------------------


def test_rejudge_presents_chunk_text_and_writes_facet_answers(monkeypatch):
    """The request carries the file's chunk text; the write carries the answer.

    The negative control for this test builds the state without the chunk text:
    the content head is then empty and the first assertion fails.
    """
    path = "/analysis/src/reader.f"
    chunks = [
        {"start_line": 20, "text": "    call ddaopen('jt60sa', 1)\n"},
        {"start_line": 1, "text": "program reader\n"},
    ]
    requests: list = []
    graph, result, _, _ = _run_rejudge(
        monkeypatch,
        [_file(path)],
        {path: chunks},
        {path: _answers(0.75, 0.4, 0.2, 0.1)},
        requests=requests,
    )

    # The request state carries the file's chunk text, in reading order.
    assert requests, "the re-judge issued a content decision request"
    head = requests[0]["state"]["file"]["content_head"]
    assert head == "program reader\n    call ddaopen('jt60sa', 1)\n"
    assert "ddaopen" in head

    # The facet fields carry the fake answers, each with its distribution and
    # confidence, exactly as the fetched content path writes them.
    ((_, kwargs),) = graph.writes
    (item,) = kwargs["items"]
    assert item["id"] == path
    assert item["score_data_access"] == 1.0
    assert item["score_signal_processing"] == 1.0
    assert item["score_machine_description"] == pytest.approx(0.6667, abs=1e-4)
    assert item["score_imas_mapping"] == pytest.approx(0.3333, abs=1e-4)
    assert item["relevance_grade"] == 3.0
    assert item["score_data_access_confidence"] == 0.9
    assert item["relevance_grade_probs"] == [0.05, 0.05, 0.1, 0.5, 0.3]
    assert item["relevance_stage"] == "content"
    assert result["rejudged"] == 1


def test_rejudge_asks_every_content_question(monkeypatch):
    """The re-judge asks the same question set the fetched path asks."""
    path = "/analysis/src/reader.f"
    requests: list = []
    _run_rejudge(
        monkeypatch,
        [_file(path)],
        {path: [{"start_line": 1, "text": "x\n"}]},
        {path: _answers(0.75, 0.4, 0.2, 0.1)},
        requests=requests,
    )

    asked = requests[0]["questions"]
    for question in (
        "data_access_depth",
        "signal_processing_depth",
        "machine_description_depth",
        "imas_mapping_depth",
        "relevance_grade",
    ):
        assert question in asked, f"re-judge request omitted {question}"


def test_rejudge_leaves_status_and_chunks_untouched(monkeypatch):
    """The write touches judgment fields only: no status, no chunk cascade."""
    path = "/analysis/src/reader.f"
    graph, _, _, _ = _run_rejudge(
        monkeypatch,
        [_file(path)],
        {path: [{"start_line": 1, "text": "x\n"}]},
        {path: _answers(0.75, 0.4, 0.2, 0.1)},
    )

    ((write_text, _),) = graph.writes
    assert "sf.status" not in write_text
    assert "DETACH DELETE" not in write_text
    # No query issued during the pass deletes or rewrites a chunk.
    assert all("DELETE" not in text for text in graph.queries)


def test_rejudge_reports_files_below_both_admission_gates(monkeypatch):
    """A file whose new composite and facets both miss stays ingested, reported."""
    path = "/analysis/src/plot.f"
    graph, result, _, _ = _run_rejudge(
        monkeypatch,
        [_file(path)],
        {path: [{"start_line": 1, "text": "plt.plot(x, y)\n"}]},
        # Weak composite (0.2 < 0.6) and weak facets (max 1/3 < 0.8).
        {path: _answers(0.2, 0.1, 0.1, 0.1, facets=(0.0, 1.0, 0.0, 0.0))},
    )

    assert result["below_gate"] == [path]
    ((_, kwargs),) = graph.writes
    # The judgment is still recorded; only the admission decision is left alone.
    assert kwargs["items"][0]["relevance_stage"] == "content"


def test_rejudge_releases_the_score_claim(monkeypatch):
    """A re-judged file's claim is released so the next pass can take it."""
    path = "/analysis/src/reader.f"
    _, _, _, released = _run_rejudge(
        monkeypatch,
        [_file(path)],
        {path: [{"start_line": 1, "text": "x\n"}]},
        {path: _answers(0.75, 0.4, 0.2, 0.1)},
    )

    assert released == [path]


# ---------------------------------------------------------------------------
# The claim reaches ingested content-stage files through the score claim
# ---------------------------------------------------------------------------


def _claim_query(monkeypatch, **claim_kwargs) -> str:
    captured: list[str] = []

    class _Recording:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def query(self, cypher, **kwargs):
            captured.append(" ".join(cypher.split()))
            return []

    monkeypatch.setattr(
        "imas_codex.discovery.code.graph_ops.GraphClient", lambda: _Recording()
    )
    from imas_codex.discovery.code.graph_ops import claim_files_for_scoring

    claim_files_for_scoring(FACILITY, **claim_kwargs)
    assert captured, "the claim issued a query"
    return captured[0]


def test_rejudge_claim_selects_ingested_content_stage_files(monkeypatch):
    """The re-judge claim takes ingested content-stage files, not triaged ones."""
    query = _claim_query(monkeypatch, ingested_rejudge=True)

    assert "sf.status = 'ingested'" in query
    assert "sf.relevance_stage = 'content'" in query
    # The drain key: a file with a recorded facet answer is not re-claimed.
    assert "coalesce(sf.score_data_access_confidence, 0.0) = 0.0" in query
    assert "sf.status = 'triaged'" not in query


def test_claim_without_rejudge_still_selects_triaged_enriched(monkeypatch):
    """The default claim is unchanged: first scoring takes triaged enriched files."""
    query = _claim_query(monkeypatch)

    assert "sf.status = 'triaged'" in query
    assert "sf.is_enriched = true" in query
    assert "sf.status = 'ingested'" not in query


# ---------------------------------------------------------------------------
# Chunk text assembly
# ---------------------------------------------------------------------------


def test_chunk_content_head_orders_by_start_line_and_cuts_to_the_fetched_length():
    """Chunks are joined in reading order and cut to the fetched path's length."""
    from imas_codex.discovery.code.scorer import (
        CONTENT_HEAD_CHARS,
        chunk_content_head,
    )

    chunks = [
        {"start_line": 30, "text": "c"},
        {"start_line": 10, "text": "a"},
        {"start_line": 20, "text": "b"},
    ]
    assert chunk_content_head(chunks) == "abc"
    # The cut matches the fetched path's content-head length.
    long = [{"start_line": 1, "text": "x" * (CONTENT_HEAD_CHARS + 500)}]
    assert len(chunk_content_head(long)) == CONTENT_HEAD_CHARS
