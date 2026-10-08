"""Drive both code-relevance decision arms through the fake decisions seam.

The names arm (``triage_worker``) asks the decisions model about a discovered
file's identity and keeps the file only when its relevance -- the largest of
the four scope probabilities -- reaches the triage threshold.  The content arm
(``score_worker``) asks the same questions of the file's preview text and
records a content relevance the ingest claim reads.

Every decision here travels the module's HTTP seam (``_apost_decisions``) with
a fake transport, so no test opens the live endpoint; the autouse guard in
``tests/conftest.py`` refuses a real request.  The graph is stubbed so the
assertions read the queries the worker actually issued rather than a live
database.
"""

from __future__ import annotations

import asyncio
import json
import logging

import pytest

from imas_codex.discovery.base import llm
from imas_codex.discovery.code.scorer import (
    SCOPE_NOULS,
    FileScoreBatch,
    FileScoreResult,
    build_triage_questions,
    scope_relevance,
)
from imas_codex.discovery.code.state import FileDiscoveryState

FACILITY = "jt-60sa"
COST = 1.0e-4
SCOPE = (
    "loads_diagnostic_data",
    "processes_diagnostic_signals",
    "describes_machine_or_diagnostics",
    "maps_to_imas",
    "reads_or_writes_reconstruction_db",
)


def _relevance(row: dict) -> float:
    return max(
        float(row.get(key) or 0.0)
        for key in (
            "relevance_loads",
            "relevance_processes",
            "relevance_describes",
            "relevance_imas",
            "relevance_reconstruction_db",
        )
    )


def _facet(row: dict) -> float:
    """The strongest stored facet, the value the admission clause reads."""
    return max(
        float(row.get(key) or 0.0)
        for key in (
            "score_data_access",
            "score_signal_processing",
            "score_machine_description",
            "score_imas_mapping",
        )
    )


def _answers(
    *nouls: float,
    reconstruction_db: float = 0.0,
    role: str = "diagnostic_data_access",
    facets: tuple[float, float, float, float] = (4.0, 3.0, 2.0, 1.0),
) -> dict:
    """Answer set with the scope nouls set positionally.

    The four positional nouls fill the first four scope questions; the
    reconstruction-database noul is the fifth and is set by keyword, so a
    caller that passes the four legacy nouls still answers every scope
    question.  Carries the content arm's graded relevance and four facet Scores
    as well; the names arm's question set has no score questions, so its
    validator ignores them.  The facet Scores are independent of the scope
    nouls, so a file whose composite is weak can still carry a strong facet,
    which is what the admission clause reads.  *facets* sets the raw Scores for
    (data access, signal processing, machine description, imas mapping).
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


class _CapturingGraph:
    """Record every query the persistence helpers issue."""

    def __init__(self):
        self.queries: list[tuple[str, dict]] = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def query(self, cypher, **kwargs):
        self.queries.append((" ".join(cypher.split()), kwargs))
        return []

    def items_for(self, marker: str) -> list[dict]:
        """Every item persisted by a query whose text carries *marker*."""
        out: list[dict] = []
        for text, kwargs in self.queries:
            if marker in text:
                out.extend(kwargs.get("items", []))
        return out


def _post_by_path(answers_by_path: dict, requests: list | None = None):
    async def fake_post(headers, body, timeout):
        if requests is not None:
            requests.append(body)
        path = body["state"]["file"]["path"]
        return _FakeResponse(_payload(answers_by_path[path]))

    return fake_post


def _no_work(*args, **kwargs):
    return []


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


def _stub_common(monkeypatch, claims: list[dict], claim_calls: list | None = None):
    """Patch the graph and facility seams every worker test shares."""
    counter = {"n": 0}

    def claim_once(*args, **kwargs):
        if claim_calls is not None:
            claim_calls.append(kwargs)
        counter["n"] += 1
        return claims if counter["n"] == 1 else []

    monkeypatch.setenv("OPENROUTER_API_KEY_IMAS_CODEX", "test-key")
    monkeypatch.setattr(
        "imas_codex.discovery.base.facility.get_facility", lambda facility: {}
    )
    monkeypatch.setattr("imas_codex.settings.get_model", lambda section: "fake-model")
    monkeypatch.setattr(
        "imas_codex.settings.get_reasoning_effort", lambda section: None
    )
    monkeypatch.setattr("imas_codex.settings.get_code_ingest_threshold", lambda: 0.6)
    monkeypatch.setattr(
        "imas_codex.settings.get_code_facet_admission_threshold", lambda: 0.8
    )
    monkeypatch.setattr(
        "imas_codex.discovery.code.graph_ops.claim_files_for_triage", claim_once
    )
    monkeypatch.setattr(
        "imas_codex.discovery.code.graph_ops.claim_files_for_scoring", claim_once
    )
    monkeypatch.setattr(
        "imas_codex.discovery.code.scorer._group_files_by_parent",
        lambda files, include_siblings=False: [
            {
                "parent_path_id": "dir",
                "parent_path": "/analysis/src",
                "parent_description": "analysis sources",
                "files": list(files),
                "sibling_names": [f["path"] for f in files],
            }
        ],
    )


def _run_triage(monkeypatch, files, answers_by_path, *, released=None):
    graph = _CapturingGraph()
    released = released if released is not None else []
    _stub_common(monkeypatch, list(files))
    monkeypatch.setattr(llm, "_apost_decisions", _post_by_path(answers_by_path))
    monkeypatch.setattr(
        "imas_codex.discovery.code.graph_ops.release_file_triage_claims",
        lambda ids: released.extend(ids),
    )
    monkeypatch.setattr("imas_codex.settings.get_code_triage_threshold", lambda: 0.4)
    monkeypatch.setattr("imas_codex.discovery.code.scorer.GraphClient", lambda: graph)

    state = FileDiscoveryState(facility=FACILITY)
    seen: dict = {}

    def on_progress(message, stats, results):
        if results is not None:
            seen["results"] = results
        state.stop_requested = True

    from imas_codex.discovery.code.workers import triage_worker

    asyncio.run(triage_worker(state, on_progress=on_progress))
    return graph, released, state, seen


def _run_score(
    monkeypatch,
    files,
    answers_by_path,
    *,
    requests: list | None = None,
    path_prefixes: list[str] | None = None,
    claim_calls: list | None = None,
):
    graph = _CapturingGraph()
    released: list[str] = []
    description_calls: list[list[str]] = []
    _stub_common(monkeypatch, list(files), claim_calls)
    monkeypatch.setattr(
        llm, "_apost_decisions", _post_by_path(answers_by_path, requests)
    )

    def fake_description(**kwargs):
        # The user prompt names the files the local model was asked to
        # describe; record which paths reached the description call.
        described = [
            line.split(" ### ", 1)[1].split(" (", 1)[0]
            for line in kwargs["messages"][1]["content"].splitlines()
            if line.strip().startswith("### ")
        ]
        description_calls.append(described)
        batch = FileScoreBatch(
            results=[
                FileScoreResult(path=p, description="analysis helper")
                for p in described
            ]
        )
        return (batch, 1.0e-2, 10)

    monkeypatch.setattr(llm, "call_llm_structured", fake_description)
    monkeypatch.setattr(
        "imas_codex.discovery.code.graph_ops.release_file_score_claims",
        lambda ids: released.extend(ids),
    )
    monkeypatch.setattr(
        "imas_codex.discovery.code.scorer._build_score_system_prompt",
        lambda facility=None, focus=None: "system",
    )
    monkeypatch.setattr("imas_codex.discovery.code.scorer.GraphClient", lambda: graph)

    state = FileDiscoveryState(facility=FACILITY, path_prefixes=path_prefixes)
    state.cost_limit = 1000.0

    def on_progress(message, stats, results):
        state.stop_requested = True

    from imas_codex.discovery.code.workers import score_worker

    asyncio.run(score_worker(state, on_progress=on_progress))
    return graph, state, released, description_calls


# ---------------------------------------------------------------------------
# Names arm
# ---------------------------------------------------------------------------


def test_names_arm_triages_above_threshold_file(monkeypatch):
    path = "/analysis/src/reader.f"
    files = [_file(path)]
    answers = {path: _answers(0.8, 0.1, 0.2, 0.3)}
    graph, _, _, seen = _run_triage(monkeypatch, files, answers)

    items = graph.items_for("sf.status = 'triaged'")
    assert [item["id"] for item in items] == [path]
    assert items[0]["relevance_loads"] == 0.8
    assert items[0]["relevance_stage"] == "name"
    assert not graph.items_for("sf.status = 'skipped'")
    assert seen["results"][0]["skipped"] is False


def test_names_arm_skips_below_threshold_file_with_reason(monkeypatch):
    path = "/analysis/src/build.f"
    files = [_file(path)]
    answers = {path: _answers(0.1, 0.05, 0.2, 0.1, role="infrastructure_or_utility")}
    graph, _, _, seen = _run_triage(monkeypatch, files, answers)

    skipped = graph.items_for("sf.status = 'skipped'")
    assert [item["id"] for item in skipped] == [path]
    assert skipped[0]["relevance_role"] == "infrastructure_or_utility"
    reason = skipped[0]["reason"]
    assert "role=infrastructure_or_utility" in reason
    assert "relevance=0.20" in reason
    assert "infrastructure_or_utility=0.90" in reason
    assert not graph.items_for("sf.status = 'triaged'")
    assert seen["results"][0]["skipped"] is True


def test_names_arm_validation_failure_leaves_file_untouched(monkeypatch):
    path = "/analysis/src/bad.f"
    files = [_file(path)]
    # A noul outside [0, 1] is refused by the answer validator, so the worker
    # never receives a decision for this file.
    bad = _answers(0.9, 0.1, 0.1, 0.1)
    bad["loads_diagnostic_data"] = {"type": "noul", "noul": 1.5}
    released: list[str] = []
    graph, released, state, _ = _run_triage(
        monkeypatch, files, {path: bad}, released=released
    )

    assert graph.queries == [], "a refused decision must not touch the graph"
    assert released == [path], "a refused decision must release the claim"
    assert state.triage_stats.processed == 0


# ---------------------------------------------------------------------------
# Scope question set
# ---------------------------------------------------------------------------


def test_scope_question_set_has_five_nouls_in_both_arms():
    """The names and content arms both ask the five reconstruction scope nouls.

    The scope questions are shared by the two arms, so a noul missing from the
    shared block would be absent from both: assert the question set renders
    with every scope noul in the names arm (``with_content=False``) as well as
    the content arm.
    """
    names = build_triage_questions(with_content=False)
    content = build_triage_questions(with_content=True)

    assert set(SCOPE_NOULS) == {
        "loads_diagnostic_data",
        "processes_diagnostic_signals",
        "describes_machine_or_diagnostics",
        "maps_to_imas",
        "reads_or_writes_reconstruction_db",
    }
    for questions in (names, content):
        for question in SCOPE_NOULS:
            assert question in questions, f"{question} missing from an arm"
        assert questions["reads_or_writes_reconstruction_db"]["type"] == "noul"


def test_composite_is_the_largest_of_five_nouls():
    """A reconstruction-database noul alone can carry a file over the gate."""
    nouls = {
        "loads_diagnostic_data": 0.1,
        "processes_diagnostic_signals": 0.15,
        "describes_machine_or_diagnostics": 0.2,
        "maps_to_imas": 0.05,
        "reads_or_writes_reconstruction_db": 0.85,
    }
    assert scope_relevance(nouls) == 0.85


def test_names_arm_writes_the_reconstruction_db_field(monkeypatch):
    """A file read by the new noul is triaged and carries its field."""
    path = "/analysis/src/eqrd13.f"
    files = [_file(path)]
    answers = {
        path: _answers(0.1, 0.1, 0.2, 0.1, reconstruction_db=0.85),
    }
    graph, _, _, seen = _run_triage(monkeypatch, files, answers)

    (item,) = graph.items_for("sf.status = 'triaged'")
    assert item["id"] == path
    assert item["relevance_reconstruction_db"] == 0.85
    # The composite is the largest of the five, so the new noul alone carries
    # the file over the 0.4 triage threshold.
    assert item["score_composite"] == 0.85
    assert seen["results"][0]["skipped"] is False


# ---------------------------------------------------------------------------
# Content arm
# ---------------------------------------------------------------------------


def test_content_arm_marks_scored_with_content_relevance(monkeypatch):
    path = "/analysis/src/reader.f"
    files = [_file(path, preview_text="mdsopen('jt60sa', 12345)")]
    answers = {path: _answers(0.75, 0.4, 0.2, 0.1)}
    graph, _, _, _ = _run_score(monkeypatch, files, answers)

    scored_items = graph.items_for("sf.status = 'scored'")
    assert [item["id"] for item in scored_items] == [path]
    # The description and the content relevance land in the same write, so a
    # file that reaches 'scored' always carries a content-arm relevance.
    assert scored_items[0]["relevance_stage"] == "content"
    assert scored_items[0]["relevance_loads"] == 0.75
    assert scored_items[0]["score_reason"] == "analysis helper"


def test_scoped_stale_ingested_judgment_uses_chunks_without_remote_executor(
    monkeypatch,
):
    """An ordinary score worker updates stored judgment fields in place."""
    path = "/analysis/src/stored.f"
    file = _file(path, status="ingested", relevance_stage="content", preview_text="old")
    requests = []
    claim_calls = []
    monkeypatch.setattr(
        "imas_codex.discovery.code.graph_ops.fetch_file_chunk_text",
        lambda ids: {path: [{"start_line": 1, "text": "stored diagnostic source"}]},
    )

    def refuse_remote(*args, **kwargs):
        raise AssertionError("remote executor must not be invoked")

    monkeypatch.setattr("imas_codex.remote.executor.run_python_script", refuse_remote)
    monkeypatch.setattr(
        "imas_codex.remote.executor.async_run_python_script", refuse_remote
    )
    graph, state, released, descriptions = _run_score(
        monkeypatch,
        [file],
        {path: _answers(0.8, 0.2, 0.1, 0.1)},
        requests=requests,
        path_prefixes=["/analysis/src"],
        claim_calls=claim_calls,
    )
    assert claim_calls[0]["path_prefixes"] == ["/analysis/src"]
    assert requests and "stored diagnostic source" in str(requests[0]["state"])
    assert descriptions == []
    assert state.score_stats.cost == pytest.approx(COST)
    assert released == [path]
    assert graph.items_for(
        "sf.score_cost = coalesce(sf.score_cost, 0) + item.score_cost"
    )
    assert all("sf.status =" not in text for text, _ in graph.queries)
    assert all("DETACH DELETE" not in text for text, _ in graph.queries)


@pytest.mark.parametrize(
    ("prior_status", "scope", "facets", "expected_status"),
    [
        ("scored", 0.1, (0.0, 0.0, 0.0, 0.0), "skipped"),
        ("skipped", 0.8, (4.0, 3.0, 2.0, 1.0), "scored"),
    ],
)
def test_stale_content_judgment_updates_uningested_status(
    monkeypatch, prior_status, scope, facets, expected_status
):
    path = "/analysis/src/stale.f"
    file = _file(
        path, status=prior_status, relevance_stage="content", preview_text="stored text"
    )
    answers = {path: _answers(scope, 0.1, 0.1, 0.1, facets=facets)}

    graph, state, released, descriptions = _run_score(monkeypatch, [file], answers)

    items = graph.items_for("sf.status = item.status")
    assert len(items) == 1
    assert items[0]["status"] == expected_status
    assert items[0]["relevance_stage"] == "content"
    assert items[0]["score_cost"] == pytest.approx(COST)
    assert descriptions == []
    assert released == [path]
    assert state.score_stats.cost == pytest.approx(COST)


def test_content_arm_asks_every_content_question(monkeypatch):
    """The content request carries the facet questions and the grade.

    The content arm sends the file text, so it must ask the questions that
    read it.  A request built from the names-arm question set omits all five,
    and the persisted facet values then come from answers never asked for.
    """
    path = "/analysis/src/reader.f"
    files = [_file(path, preview_text="mdsopen('jt60sa', 12345)")]
    answers = {path: _answers(0.75, 0.4, 0.2, 0.1)}
    requests: list = []
    graph, _, _, _ = _run_score(monkeypatch, files, answers, requests=requests)

    assert requests, "the content arm issued a decisions request"
    asked = requests[0]["questions"]
    for question in (
        "data_access_depth",
        "signal_processing_depth",
        "machine_description_depth",
        "imas_mapping_depth",
        "relevance_grade",
    ):
        assert question in asked, f"content request omitted {question}"

    # The persisted facet values are the fake answers divided by their top
    # levels (data access 4, the other three 3).
    (item,) = graph.items_for("sf.status = 'scored'")
    assert item["score_data_access"] == 1.0
    assert item["score_signal_processing"] == 1.0
    assert item["score_machine_description"] == pytest.approx(0.6667, abs=1e-4)
    assert item["score_imas_mapping"] == pytest.approx(0.3333, abs=1e-4)
    assert item["relevance_grade"] == 3.0


def test_content_arm_writes_every_stage_field(monkeypatch):
    """Every content-arm field lands in the one write: facets, grade, role."""
    path = "/analysis/src/reader.f"
    files = [_file(path, preview_text="mdsopen('jt60sa', 12345)")]
    answers = {path: _answers(0.75, 0.4, 0.2, 0.1)}
    graph, _, _, _ = _run_score(monkeypatch, files, answers)

    (item,) = graph.items_for("sf.status = 'scored'")
    # Facet values are each Score divided by its top level (4, 3, 3, 3).
    assert item["score_composite"] == 0.75
    assert item["score_data_access"] == 1.0
    assert item["score_signal_processing"] == 1.0
    assert item["score_machine_description"] == pytest.approx(0.6667, abs=1e-4)
    assert item["score_imas_mapping"] == pytest.approx(0.3333, abs=1e-4)
    assert item["relevance_grade"] == 3.0
    # Each Score carries its distribution and confidence beside it.
    assert item["score_data_access_probs"] == [0.0, 0.0, 0.0, 0.0, 1.0]
    assert item["score_data_access_confidence"] == 0.9
    assert item["score_signal_processing_probs"] == [0.0, 0.0, 0.0, 1.0]
    assert item["score_signal_processing_confidence"] == 0.6
    assert item["score_machine_description_probs"] == [0.0, 0.0, 1.0, 0.0]
    assert item["score_machine_description_confidence"] == 0.5
    assert item["score_imas_mapping_probs"] == [0.0, 1.0, 0.0, 0.0]
    assert item["score_imas_mapping_confidence"] == 0.4
    assert item["relevance_grade_probs"] == [0.05, 0.05, 0.1, 0.5, 0.3]
    assert item["relevance_grade_confidence"] == 0.7
    # The role distribution is aligned to the fixed role enum order.
    assert len(item["relevance_role_probs"]) == 8
    assert item["relevance_role_probs"][0] == 0.9  # diagnostic_data_access
    assert item["relevance_role_confidence"] == 0.8


def test_content_arm_describes_only_a_passing_file(monkeypatch):
    """The local model is called only for files the ingest gate admits."""
    passing = "/analysis/src/reader.f"
    below = "/analysis/src/plot.f"
    files = [
        _file(passing, preview_text="mdsopen('jt60sa', 1)"),
        _file(below, preview_text="plt.plot(x, y)"),
    ]
    answers = {
        passing: _answers(0.75, 0.4, 0.2, 0.1),
        # Weak composite and weak facets: neither gate admits this file, so
        # it is scored but never described.
        below: _answers(0.2, 0.1, 0.1, 0.1, facets=(0.0, 1.0, 0.0, 0.0)),
    }
    graph, _, _, description_calls = _run_score(monkeypatch, files, answers)

    assert description_calls == [[passing]]
    scored = {item["id"] for item in graph.items_for("sf.status = 'scored'")}
    assert scored == {passing, below}
    # Only the admitted file carries a description.
    described = {
        item["id"]: item.get("score_reason")
        for item in graph.items_for("sf.status = 'scored'")
    }
    assert described[passing] == "analysis helper"
    assert not described[below]


def test_content_arm_failure_leaves_file_unscored_and_claimable(monkeypatch):
    path = "/analysis/src/reader.f"
    files = [_file(path, preview_text="content")]
    bad = _answers(0.9, 0.1, 0.1, 0.1)
    bad["role"] = {
        "type": "choice",
        "choice": "not_a_criterion",
        "probabilities": {"diagnostic_data_access": 1.0},
    }
    graph, _, released, _ = _run_score(monkeypatch, files, {path: bad})

    # A failed content decision writes nothing: the file stays at its prior
    # status, carries no content relevance, and its claim is released so the
    # next score pass reclaims and retries it.
    assert graph.queries == [], "a failed content decision writes nothing"
    assert released == [path], "the file's claim is released for a later retry"


def test_content_arm_reports_batch_failures_once(monkeypatch, caplog):
    """A batch's decision failures are logged once, with a count and a reason."""
    a = "/analysis/src/a.f"
    b = "/analysis/src/b.f"
    files = [_file(a, preview_text="x"), _file(b, preview_text="y")]
    bad = _answers(0.9, 0.1, 0.1, 0.1)
    bad["loads_diagnostic_data"] = {"type": "noul", "noul": 1.5}
    with caplog.at_level(logging.INFO, logger="imas_codex.discovery.code.workers"):
        _run_score(monkeypatch, files, {a: bad, b: bad})

    failures = [
        r.getMessage()
        for r in caplog.records
        if "content decision failed" in r.getMessage()
    ]
    assert len(failures) == 1, "one line per batch, not one per file"
    assert "2 of 2" in failures[0]


def test_content_arm_pairs_each_description_with_its_own_file(monkeypatch):
    """A failed decision mid-batch must not shift the files onto the wrong ones.

    The first file's content decision is refused by the answer validator, so it
    drops out of the batch's decisions while the other two survive.  Each
    surviving decision must stay paired with its own file by path: the two
    admitted files are described, the refused file is not, and the buggy
    positional zip -- which would describe the refused file and drop the last
    one -- is exactly what this asserts against.
    """
    refused = "/analysis/src/refused.f"
    first = "/analysis/src/reader.f"
    second = "/analysis/src/writer.f"
    files = [
        _file(refused, preview_text="bad"),
        _file(first, preview_text="mdsopen('jt60sa', 1)"),
        _file(second, preview_text="mdsput('jt60sa', 2)"),
    ]
    bad = _answers(0.9, 0.1, 0.1, 0.1)
    bad["loads_diagnostic_data"] = {"type": "noul", "noul": 1.5}
    answers = {
        refused: bad,
        first: _answers(0.9, 0.4, 0.2, 0.1),
        second: _answers(0.8, 0.3, 0.2, 0.1),
    }
    graph, _, _, description_calls = _run_score(monkeypatch, files, answers)

    # Only the two files whose decisions survived are described, and each is
    # the file its own decision admitted.
    assert description_calls == [[first, second]]
    scored = {
        item["id"]: item.get("score_reason")
        for item in graph.items_for("sf.status = 'scored'")
    }
    assert set(scored) == {first, second}
    assert scored[first] == "analysis helper"
    assert scored[second] == "analysis helper"
    assert refused not in scored, "the refused file is not scored on another's decision"


# ---------------------------------------------------------------------------
# Ingest claim
# ---------------------------------------------------------------------------


class _IngestStub:
    """Evaluate the seeded rows against the claim predicates the query carries."""

    def __init__(self, rows):
        self.rows = rows
        self.claimed: list[dict] = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def query(self, cypher, **kwargs):
        text = " ".join(cypher.split())
        if "SET sf.claimed_at" in text:
            require_stage = "sf.relevance_stage = 'content'" in text
            facet_gate = kwargs.get("min_facet_relevance")
            eligible = [
                r
                for r in self.rows
                if r["status"] == "scored"
                and (not require_stage or r.get("relevance_stage") == "content")
                and (
                    _relevance(r) >= kwargs["min_relevance"]
                    or (facet_gate is not None and _facet(r) >= facet_gate)
                )
                and r.get("line_count", 0) <= kwargs["max_line_count"]
            ]
            eligible.sort(key=_relevance, reverse=True)
            self.claimed = eligible[: kwargs["limit"]]
            for row in self.claimed:
                row["claim_token"] = kwargs["token"]
            return []
        if "claim_token: $token" in text:
            token = kwargs["token"]
            return [
                {
                    "id": r["id"],
                    "path": r["path"],
                    "language": r["language"],
                    "score_composite": r.get("score_composite"),
                    "content_hash": r.get("content_hash"),
                }
                for r in self.claimed
                if r.get("claim_token") == token
            ]
        return []


def test_ingest_claims_highest_relevance_first_and_refuses_below_threshold(
    monkeypatch,
):
    from imas_codex.discovery.code.workers import _claim_code_files_for_ingestion

    rows = [
        _file(
            "/analysis/src/low.f",
            status="scored",
            relevance_stage="content",
            relevance_loads=0.3,
        ),
        _file(
            "/analysis/src/mid.f",
            status="scored",
            relevance_stage="content",
            relevance_loads=0.7,
        ),
        _file(
            "/analysis/src/high.f",
            status="scored",
            relevance_stage="content",
            relevance_loads=0.95,
        ),
    ]
    monkeypatch.setattr("imas_codex.graph.GraphClient", lambda: _IngestStub(rows))

    claimed = _claim_code_files_for_ingestion(FACILITY, limit=10, min_relevance=0.5)

    assert [c["path"] for c in claimed] == [
        "/analysis/src/high.f",
        "/analysis/src/mid.f",
    ]
    assert "/analysis/src/low.f" not in [c["path"] for c in claimed]


def test_ingest_claim_orders_by_relevance_desc(monkeypatch):
    """The rendered claim orders by the shared relevance expression."""
    from imas_codex.discovery.code.workers import _claim_code_files_for_ingestion

    captured: list[str] = []

    class _Recording(_IngestStub):
        def query(self, cypher, **kwargs):
            captured.append(" ".join(cypher.split()))
            return super().query(cypher, **kwargs)

    rows = [
        _file(
            "/analysis/src/a.f",
            status="scored",
            relevance_stage="content",
            relevance_imas=0.8,
        )
    ]
    monkeypatch.setattr("imas_codex.graph.GraphClient", lambda: _Recording(rows))

    _claim_code_files_for_ingestion(FACILITY, limit=10, min_relevance=0.5)

    claim_query = next(t for t in captured if "SET sf.claimed_at" in t)
    assert "ORDER BY relevance DESC" in claim_query
    assert ">= $min_relevance" in claim_query


def test_ingest_claim_refuses_name_stage_relevance(monkeypatch):
    """A name-arm relevance never carries a file into ingestion."""
    from imas_codex.discovery.code.workers import _claim_code_files_for_ingestion

    rows = [
        _file(
            "/analysis/src/nameonly.f",
            status="scored",
            relevance_stage="name",
            relevance_loads=0.9,
        )
    ]
    monkeypatch.setattr("imas_codex.graph.GraphClient", lambda: _IngestStub(rows))

    claimed = _claim_code_files_for_ingestion(FACILITY, limit=10, min_relevance=0.5)

    assert claimed == [], "a file whose relevance came from the names arm is refused"
