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

import pytest

from imas_codex.discovery.base import llm
from imas_codex.discovery.code.scorer import FileScoreBatch, FileScoreResult
from imas_codex.discovery.code.state import FileDiscoveryState

FACILITY = "jt-60sa"
COST = 1.0e-4
SCOPE = (
    "loads_diagnostic_data",
    "processes_diagnostic_signals",
    "describes_machine_or_diagnostics",
    "maps_to_imas",
)


def _relevance(row: dict) -> float:
    return max(
        float(row.get(key) or 0.0)
        for key in (
            "relevance_loads",
            "relevance_processes",
            "relevance_describes",
            "relevance_imas",
        )
    )


def _answers(*nouls: float, role: str = "diagnostic_data_access") -> dict:
    """Six-question answer set with the four scope nouls set positionally."""
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


def _post_by_path(answers_by_path: dict):
    async def fake_post(headers, body, timeout):
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


def _stub_common(monkeypatch, claims: list[dict]):
    """Patch the graph and facility seams every worker test shares."""
    counter = {"n": 0}

    def claim_once(*args, **kwargs):
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


def _run_score(monkeypatch, files, answers_by_path):
    graph = _CapturingGraph()
    released: list[str] = []
    _stub_common(monkeypatch, list(files))
    monkeypatch.setattr(llm, "_apost_decisions", _post_by_path(answers_by_path))
    batch = FileScoreBatch(
        results=[
            FileScoreResult(
                path=f["path"],
                file_category="code",
                description="analysis helper",
                score_analysis_code=0.5,
            )
            for f in files
        ]
    )
    monkeypatch.setattr(llm, "call_llm_structured", lambda **kwargs: (batch, 0.01, 10))
    monkeypatch.setattr(
        "imas_codex.discovery.code.graph_ops.release_file_score_claims",
        lambda ids: released.extend(ids),
    )
    monkeypatch.setattr(
        "imas_codex.discovery.code.scorer._build_score_system_prompt",
        lambda facility=None, focus=None: "system",
    )
    monkeypatch.setattr(
        "imas_codex.discovery.code.scorer._build_score_user_prompt",
        lambda file_groups: "user",
    )
    monkeypatch.setattr("imas_codex.discovery.code.scorer.GraphClient", lambda: graph)

    state = FileDiscoveryState(facility=FACILITY)
    state.cost_limit = 1000.0

    def on_progress(message, stats, results):
        state.stop_requested = True

    from imas_codex.discovery.code.workers import score_worker

    asyncio.run(score_worker(state, on_progress=on_progress))
    return graph, state


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
# Content arm
# ---------------------------------------------------------------------------


def test_content_arm_records_relevance_without_changing_status(monkeypatch):
    path = "/analysis/src/reader.f"
    files = [_file(path, preview_text="mdsopen('jt60sa', 12345)")]
    answers = {path: _answers(0.75, 0.4, 0.2, 0.1)}
    graph, _ = _run_score(monkeypatch, files, answers)

    content_items = graph.items_for("relevance_stage = item.relevance_stage")
    assert [item["id"] for item in content_items] == [path]
    assert content_items[0]["relevance_stage"] == "content"
    assert content_items[0]["relevance_loads"] == 0.75
    # The score call itself still set status='scored' and never skipped.
    assert graph.items_for("sf.status = 'scored'")


def test_content_arm_failure_leaves_file_scored_without_relevance(monkeypatch):
    path = "/analysis/src/reader.f"
    files = [_file(path, preview_text="content")]
    bad = _answers(0.9, 0.1, 0.1, 0.1)
    bad["role"] = {
        "type": "choice",
        "choice": "not_a_criterion",
        "probabilities": {"diagnostic_data_access": 1.0},
    }
    graph, _ = _run_score(monkeypatch, files, {path: bad})

    assert graph.items_for("sf.status = 'scored'"), "the score itself still lands"
    content_items = graph.items_for("relevance_stage = item.relevance_stage")
    assert content_items == [], "a refused content decision writes no relevance"


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
            eligible = [
                r
                for r in self.rows
                if r["status"] == "scored"
                and _relevance(r) >= kwargs["min_relevance"]
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
        _file("/analysis/src/low.f", status="scored", relevance_loads=0.3),
        _file("/analysis/src/mid.f", status="scored", relevance_loads=0.7),
        _file("/analysis/src/high.f", status="scored", relevance_loads=0.95),
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

    rows = [_file("/analysis/src/a.f", status="scored", relevance_imas=0.8)]
    monkeypatch.setattr("imas_codex.graph.GraphClient", lambda: _Recording(rows))

    _claim_code_files_for_ingestion(FACILITY, limit=10, min_relevance=0.5)

    claim_query = next(t for t in captured if "SET sf.claimed_at" in t)
    assert "ORDER BY relevance DESC" in claim_query
    assert ">= $min_relevance" in claim_query
