"""The decisions-key skips in the pool rerank and the candidate stage.

With ``OPENROUTER_API_KEY_IMAS_CODEX`` unset no Jev decisions transport is
built for either the generic pool rerank or the candidate stage's routing and
judgment. ``rerank_pool`` returns the embedding order, a note naming the missing
variable and a zero cost; the candidate stage logs one warning for the stage and
leaves every source unjudged, exactly as a transport failure does. Each test
replaces the transport with a recorder that fails the test if it is ever
entered, so a skip that still made the call would be caught.
"""

from __future__ import annotations

import logging

import pytest

from imas_codex.discovery.base import judgment
from imas_codex.ids import candidates as cand

MODEL = "typesafe/jev-1.13"
SERVICE = "facility-discovery"

QUESTIONS = {
    "relevance_grade": {
        "type": "score",
        "instructions": "How relevant is this candidate?",
        "criteria": [
            "unrelated",
            "mentions the topic but does not help",
            "useful background",
            "answers a substantial part",
            "directly answers",
        ],
    }
}


@pytest.fixture(autouse=True)
def _no_decisions_key(monkeypatch):
    """Remove the decisions key and start each test with a fresh warning latch."""
    monkeypatch.delenv("OPENROUTER_API_KEY_IMAS_CODEX", raising=False)
    monkeypatch.setattr(cand, "_key_absence_warned", False)


def _state_for(query: str, candidate: dict) -> dict:
    return {
        "query": query,
        "candidate": {
            "locator": {"path": candidate["path"]},
            "text": candidate["text"],
        },
    }


def _pool(n: int) -> list[dict]:
    return [{"path": f"f{i}", "text": f"t{i}"} for i in range(n)]


class _FakeGraph:
    """A graph client answering only the IDS criteria query.

    Under the guard the query is never reached; with the guard removed the
    stage proceeds through the criteria to the decisions transport, so a fake
    that answers lets the negative control show the transport being entered.
    """

    def query(self, _cypher, **_params):
        return [{"id": "equilibrium", "description": "MHD equilibrium"}]


async def test_rerank_pool_skips_without_the_decisions_key(monkeypatch):
    calls: list[dict] = []

    async def refusing_transport(model, state, questions, *, service, **_kwargs):
        calls.append(state)
        raise AssertionError("the decisions transport must not be entered")

    monkeypatch.setattr(judgment, "acall_decisions", refusing_transport)

    pool = _pool(3)
    ordered, note, cost = await judgment.rerank_pool(
        "q",
        pool,
        state_for=_state_for,
        levels=QUESTIONS["relevance_grade"]["criteria"],
        instructions=QUESTIONS["relevance_grade"]["instructions"],
        model=MODEL,
        service=SERVICE,
    )

    assert [candidate["path"] for candidate in ordered] == ["f0", "f1", "f2"]
    assert calls == []
    assert note is not None and "OPENROUTER_API_KEY_IMAS_CODEX" in note
    assert cost == 0.0


def test_candidate_stage_leaves_every_source_unjudged(monkeypatch, caplog):
    calls: list[dict] = []

    def refusing_transport(model, state, questions, *, service, **_kwargs):
        calls.append(state)
        raise AssertionError("the decisions transport must not be entered")

    monkeypatch.setattr(cand, "call_decisions", refusing_transport)

    with caplog.at_level(logging.WARNING):
        routed = [cand.route_ids(f"source {i}", gc=_FakeGraph()) for i in range(4)]
        judged = cand.judge_candidates({}, {}, [object()])

    assert routed == [None, None, None, None]
    assert judged is None
    assert calls == []

    warnings = [
        record
        for record in caplog.records
        if "OPENROUTER_API_KEY_IMAS_CODEX" in record.getMessage()
    ]
    assert len(warnings) == 1
