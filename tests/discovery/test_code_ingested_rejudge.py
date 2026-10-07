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
import importlib
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
    "reads_or_writes_reconstruction_db",
)


def _answers(
    *nouls: float,
    reconstruction_db: float = 0.0,
    role: str = "diagnostic_data_access",
    facets: tuple[float, float, float, float] = (4.0, 3.0, 2.0, 1.0),
) -> dict:
    """Answer set with the scope nouls set positionally.

    The four positional nouls fill the first four scope questions; the
    reconstruction-database noul is the fifth and is set by keyword so a caller
    passing the four legacy nouls still answers every scope question.  Carries
    the content arm's graded relevance and four facet Scores as well;
    *facets* sets the raw Scores for (data access, signal processing, machine
    description, imas mapping).  The facet Scores are independent of the scope
    nouls, so a file whose composite is weak can still carry a strong facet.
    """
    data_score, signal_score, machine_score, imas_score = facets
    values = (*nouls, reconstruction_db)
    out = {
        name: {"type": "noul", "noul": value}
        for name, value in zip(SCOPE, values, strict=True)
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


# ---------------------------------------------------------------------------
# A repeated pass resumes: the reset is once per request, not once per pass
# ---------------------------------------------------------------------------


class _MemoryRejudge:
    """The graph state a re-judge pass reads and writes, held in memory.

    Models the two ends the pass depends on: the reset clears each ingested
    content-stage file's *recorded answer*, and the judgment write records one.
    The claim is stubbed against this same state, so a second pass sees exactly
    the files the first left unanswered.  ``confidence`` stands for the facet
    confidence the claim drains on; a file whose decision fails never has it
    written, so it stays claimable.
    """

    def __init__(self, files: list[tuple[str, list[dict]]]):
        self.files = {
            path: {
                "id": path,
                "path": path,
                "confidence": 0.0,
                "claimed": False,
                "chunks": chunks,
            }
            for path, chunks in files
        }
        self.queries: list[str] = []
        self.reset_calls = 0
        self.writes: list[dict] = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def query(self, cypher, **kwargs):
        text = " ".join(cypher.split())
        self.queries.append(text)
        if "n.status = $target_status" in text and "reset_count" in text:
            # reset_to_status: clear the recorded answer wherever one is set.
            self.reset_calls += 1
            cleared = 0
            for f in self.files.values():
                if f["confidence"] != 0.0:
                    f["confidence"] = 0.0
                    cleared += 1
                f["claimed"] = False
            return [{"reset_count": cleared}]
        if "HAS_CHUNK" in text:
            rows: list[dict] = []
            for fid in kwargs["ids"]:
                for chunk in self.files.get(fid, {}).get("chunks", []):
                    rows.append(
                        {
                            "file_id": fid,
                            "start_line": chunk["start_line"],
                            "text": chunk["text"],
                        }
                    )
            return rows
        if "UNWIND $items AS item" in text:
            self.writes.append(kwargs)
            for item in kwargs["items"]:
                f = self.files.get(item["id"])
                if f is not None:
                    f["confidence"] = item.get("score_data_access_confidence", 0.0)
                    f["claimed"] = False
            return []
        return []

    def claim(self, facility, limit=100, path_prefixes=None, *, ingested_rejudge=False):
        """The claim's own selection: ingested content-stage files with no
        recorded answer, which is what makes a later pass resume."""
        if not ingested_rejudge:
            return []
        out: list[dict] = []
        for f in self.files.values():
            if f["confidence"] == 0.0 and not f["claimed"] and len(out) < limit:
                f["claimed"] = True
                out.append(_file(f["id"]))
        return out

    def release(self, file_ids):
        for fid in file_ids:
            f = self.files.get(fid)
            if f is not None:
                f["claimed"] = False


def _drive_passes(
    monkeypatch,
    memory: _MemoryRejudge,
    answers_by_path: dict,
    *,
    failing: tuple[str, ...] = (),
    batch_size: int = 10,
    passes: int = 1,
):
    """Run the re-judge ``passes`` times over the memory graph.

    Returns the per-pass result dicts and, for each pass, the paths whose
    decision was requested, so a test can assert what a pass did and did not
    judge.  A path in ``failing`` raises at the HTTP seam the *first* time its
    decision is requested, so the failure is transient: its answer is left
    unwritten on the first pass and the next pass can take it again.
    """
    from imas_codex.discovery.code import graph_ops

    requests: list[str] = []
    seen: dict[str, int] = {}

    async def fake_post(headers, body, timeout):
        path = body["state"]["file"]["path"]
        requests.append(path)
        count = seen.get(path, 0)
        seen[path] = count + 1
        if path in failing and count == 0:
            raise RuntimeError("simulated decision failure")
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
    monkeypatch.setattr(graph_ops, "claim_files_for_scoring", memory.claim)
    monkeypatch.setattr(graph_ops, "release_file_score_claims", memory.release)
    monkeypatch.setattr(graph_ops, "GraphClient", lambda: memory)
    monkeypatch.setattr("imas_codex.discovery.code.scorer.GraphClient", lambda: memory)
    monkeypatch.setattr("imas_codex.graph.GraphClient", lambda: memory)
    monkeypatch.setattr(llm, "_apost_decisions", fake_post)

    from imas_codex.discovery.code.workers import rejudge_ingested_files

    results: list[dict] = []
    per_pass: list[list[str]] = []
    for _ in range(passes):
        before = len(requests)
        results.append(
            asyncio.run(rejudge_ingested_files(FACILITY, batch_size=batch_size))
        )
        per_pass.append(requests[before:])
    return results, per_pass


def test_second_pass_judges_only_the_file_the_first_left_unanswered(monkeypatch):
    """Three ingested files, one decision fails on pass one.

    Pass one judges all three and records the two answers it got; the failed
    file keeps no answer.  Pass two takes only that file, because the claim
    drains on the recorded answer and no reset intervenes between the passes.
    """
    paths = [f"/analysis/src/f{i}.f" for i in range(3)]
    memory = _MemoryRejudge([(p, [{"start_line": 1, "text": "x\n"}]) for p in paths])
    answers = {p: _answers(0.75, 0.4, 0.2, 0.1) for p in paths}
    failed = paths[1]

    results, per_pass = _drive_passes(
        monkeypatch, memory, answers, failing=(failed,), passes=2
    )

    assert set(per_pass[0]) == set(paths)
    assert results[0]["rejudged"] == 2
    # The second pass resumes: only the file with no recorded answer is taken.
    assert per_pass[1] == [failed]
    assert results[1]["rejudged"] == 1
    # Neither pass resets: the reset is a separate, once-per-request step.
    assert memory.reset_calls == 0


def test_no_file_is_judged_twice_in_one_pass_and_count_matches(monkeypatch):
    """A batch that mixes an attempted file with a fresh one takes only the fresh.

    With a batch size of two over three files, pass one's first batch is
    ``[A, B]``; A's decision fails and leaves no answer, so the next claim
    returns ``[A, C]`` -- an already-attempted file beside a fresh one.  Only C
    is judged, so each file is asked once and ``rejudged`` counts the two
    distinct files whose answers were written, not the repeat.
    """
    paths = [f"/analysis/src/g{i}.f" for i in range(3)]
    memory = _MemoryRejudge([(p, [{"start_line": 1, "text": "x\n"}]) for p in paths])
    answers = {p: _answers(0.75, 0.4, 0.2, 0.1) for p in paths}
    failed = paths[0]

    results, per_pass = _drive_passes(
        monkeypatch, memory, answers, failing=(failed,), batch_size=2
    )

    judged = per_pass[0]
    assert len(judged) == len(set(judged)), f"a file was judged twice: {judged}"
    assert set(judged) == set(paths)
    assert results[0]["rejudged"] == 2


# ---------------------------------------------------------------------------
# The CLI reset is an explicit, once-per-request step, not a per-pass one
# ---------------------------------------------------------------------------


def _invoke_rejudge_cli(monkeypatch, args: list[str]):
    """Drive ``discover code``'s re-judge branch with its seams stubbed.

    The reset owner and the re-judge worker are replaced by recorders, so a test
    reads exactly whether the reset ran and when, without a live graph or LLM.
    """
    from click.testing import CliRunner

    code_mod = importlib.import_module("imas_codex.cli.discover.code")

    reset_calls: list = []
    rejudge_calls: list = []

    monkeypatch.setattr(
        "imas_codex.discovery.base.facility.get_facility",
        lambda facility: {"ssh_host": "host"},
    )
    monkeypatch.setattr("imas_codex.settings.get_discovery_threshold", lambda: 0.5)
    monkeypatch.setattr(
        "imas_codex.cli.discover.common.setup_logging", lambda *a, **k: None
    )
    monkeypatch.setattr(
        "imas_codex.cli.discover.common.make_log_print",
        lambda *a, **k: lambda msg: None,
    )
    monkeypatch.setattr("imas_codex.cli.discover.common.use_rich_output", lambda: False)
    monkeypatch.setattr(
        "imas_codex.discovery.base.reset.reset_to_status",
        lambda *a, **k: (reset_calls.append((a, k)), 0)[1],
    )

    async def fake_rejudge(facility, **kwargs):
        rejudge_calls.append(facility)
        return {"rejudged": 0, "below_gate": [], "cost": 0.0, "batches": 0}

    monkeypatch.setattr(
        "imas_codex.discovery.code.workers.rejudge_ingested_files", fake_rejudge
    )
    monkeypatch.setattr(
        "imas_codex.cli.shutdown.safe_asyncio_run", lambda coro: asyncio.run(coro)
    )

    result = CliRunner().invoke(code_mod.code, args)
    return result, reset_calls, rejudge_calls


def test_rejudge_ingested_alone_does_not_reset(monkeypatch):
    """A plain ``--rejudge-ingested`` resumes: it never clears recorded answers."""
    result, reset_calls, rejudge_calls = _invoke_rejudge_cli(
        monkeypatch, ["jt-60sa", "--rejudge-ingested"]
    )

    assert result.exit_code == 0, result.output
    assert reset_calls == [], "a plain re-judge must not reset"
    assert rejudge_calls == ["jt-60sa"]


def test_rejudge_ingested_with_reset_to_ingested_resets_once(monkeypatch):
    """The fresh-request reset rides the explicit ``--reset-to ingested`` step."""
    from imas_codex.discovery.base.reset import CODE_RESET_SPECS

    result, reset_calls, rejudge_calls = _invoke_rejudge_cli(
        monkeypatch, ["jt-60sa", "--reset-to", "ingested", "--rejudge-ingested"]
    )

    assert result.exit_code == 0, result.output
    assert len(reset_calls) == 1, "the reset runs exactly once per request"
    (spec, *_), _ = reset_calls[0]
    assert spec is CODE_RESET_SPECS["ingested"]
    assert rejudge_calls == ["jt-60sa"]
