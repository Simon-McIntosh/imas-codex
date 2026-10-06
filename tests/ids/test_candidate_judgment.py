"""Drive the candidate-stage judgments through the fake decisions seam.

``route_ids`` asks one Choice over the IDS criteria, ``judge_candidates`` asks
one batched ``same_quantity`` noul per candidate, and ``route`` turns the
judgments into a route. Every decision travels the shared decisions HTTP seam
(``_post_decisions``) with a fake transport, so no test opens the live
endpoint; the autouse guard in ``tests/conftest.py`` refuses a real request.
The graph is stubbed so the assertions read the criteria the routing choice
actually offered.
"""

from __future__ import annotations

import json

import pytest

from imas_codex.discovery.base import llm
from imas_codex.ids.candidates import (
    Candidate,
    PairJudgment,
    judge_candidates,
    route,
    route_ids,
)
from imas_codex.ids.mapping import PipelineCost
from imas_codex.models.constants import SearchMode
from imas_codex.search.search_strategy import SearchHit
from imas_codex.settings import RouteThresholds, get_mapping_route_thresholds

MODEL = "typesafe/jev-1.13"
COST = 2.0e-4


@pytest.fixture(autouse=True)
def _decisions_key(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY_IMAS_CODEX", "test-key")


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload
        self.status_code = 200
        self.text = json.dumps(payload)

    def json(self):
        return self._payload


def _payload(answers: dict) -> dict:
    return {"answers": answers, "usage": {"cost": COST}, "model": MODEL}


def _candidate(path: str, score: float) -> Candidate:
    hit = SearchHit(
        path=path,
        ids_name=path.split("/", 1)[0],
        documentation=f"documentation for {path}",
        score=score,
        rank=1,
        search_mode=SearchMode.AUTO,
    )
    return Candidate(hit=hit, parent_documentation=f"parent of {path}")


def _judgment(path: str, probability: float) -> PairJudgment:
    return PairJudgment(
        path=path,
        p_same_quantity=probability,
        model=MODEL,
        judged_at="2026-10-06T00:00:00+00:00",
    )


class _FakeGraph:
    """Graph client answering only the IDS criteria query."""

    def __init__(self, ids: dict[str, str]):
        self._ids = ids

    def query(self, cypher: str, **params):
        assert "MATCH (i:IDS)" in cypher
        return [
            {"id": name, "description": description}
            for name, description in self._ids.items()
        ]


def test_state_carries_facility_source_and_candidate_blocks(monkeypatch):
    captured: dict = {}

    def fake_post(headers, body, timeout):
        captured["body"] = body
        count = len(body["state"]["candidates"])
        answers = {
            f"same_quantity_{i}": {"type": "noul", "noul": 0.5} for i in range(count)
        }
        return _FakeResponse(_payload(answers))

    monkeypatch.setattr(llm, "_post_decisions", fake_post)

    candidates = [
        _candidate("equilibrium/time_slice/profiles_1d/psi", 0.9),
        _candidate("core_profiles/profiles_1d/electrons/temperature", 0.8),
    ]
    result = judge_candidates(
        {"id": "src-1", "description": "plasma poloidal flux"},
        {"facility_id": "jet"},
        candidates,
    )

    state = captured["body"]["state"]
    assert set(state) == {"facility", "signal_source", "candidates"}
    assert state["facility"]["facility_id"] == "jet"
    assert state["signal_source"]["id"] == "src-1"
    assert [c["path"] for c in state["candidates"]] == [c.hit.path for c in candidates]
    assert result is not None
    assert [j.path for j in result] == [c.hit.path for c in candidates]
    assert result[0].p_same_quantity == 0.5
    assert result[0].model == MODEL
    assert result[0].judged_at


def test_malformed_answer_is_rejected(monkeypatch):
    from imas_codex.discovery.base.llm import DecisionsValidationError

    def fake_post(headers, body, timeout):
        return _FakeResponse(
            _payload({"same_quantity_0": {"type": "noul", "noul": 1.5}})
        )

    monkeypatch.setattr(llm, "_post_decisions", fake_post)

    with pytest.raises(DecisionsValidationError):
        judge_candidates(
            {"id": "src-1", "description": "flux"},
            {"facility_id": "jet"},
            [_candidate("equilibrium/time_slice/profiles_1d/psi", 0.9)],
        )


def test_transport_error_yields_no_route(monkeypatch):
    def fake_post(headers, body, timeout):
        raise RuntimeError("connection refused")

    # One failed attempt: the retry policy is exercised in its own tests, and a
    # retryable error would otherwise sleep through the backoff here.
    monkeypatch.setattr(llm, "_decisions_retryable", lambda error: False)
    monkeypatch.setattr(llm, "_post_decisions", fake_post)

    candidates = [_candidate("equilibrium/time_slice/profiles_1d/psi", 0.9)]
    judgments = judge_candidates(
        {"id": "src-1", "description": "flux"}, {"facility_id": "jet"}, candidates
    )
    assert judgments is None
    assert route(judgments, get_mapping_route_thresholds()) is None

    graph = _FakeGraph({"equilibrium": "equilibrium IDS"})
    assert route_ids("flux", gc=graph) is None


def test_judge_candidates_adds_reported_cost_to_the_pipeline_cost(monkeypatch):
    def fake_post(headers, body, timeout):
        count = len(body["state"]["candidates"])
        answers = {
            f"same_quantity_{i}": {"type": "noul", "noul": 0.5} for i in range(count)
        }
        return _FakeResponse(_payload(answers))

    monkeypatch.setattr(llm, "_post_decisions", fake_post)

    cost = PipelineCost()
    candidates = [
        _candidate("equilibrium/time_slice/profiles_1d/psi", 0.9),
        _candidate("core_profiles/profiles_1d/electrons/temperature", 0.8),
    ]
    result = judge_candidates(
        {"id": "src-1", "description": "plasma poloidal flux"},
        {"facility_id": "jet"},
        candidates,
        cost=cost,
    )

    assert result is not None
    assert cost.steps["candidate_judgment"] == pytest.approx(COST)
    assert cost.total_usd == pytest.approx(COST)


def test_route_ids_adds_reported_cost_to_the_pipeline_cost(monkeypatch):
    graph = _FakeGraph(
        {
            "equilibrium": "equilibrium fields",
            "core_profiles": "core plasma profiles",
        }
    )

    def fake_post(headers, body, timeout):
        criteria = body["questions"]["ids_routing"]["criteria"]
        probabilities = dict.fromkeys(criteria, 0.0)
        probabilities["equilibrium"] = 0.9
        probabilities["core_profiles"] = 0.1
        return _FakeResponse(
            _payload(
                {
                    "ids_routing": {
                        "type": "choice",
                        "choice": "equilibrium",
                        "probabilities": probabilities,
                        "confidence": 0.8,
                    }
                }
            )
        )

    monkeypatch.setattr(llm, "_post_decisions", fake_post)

    cost = PipelineCost()
    result = route_ids("plasma poloidal flux", gc=graph, cost=cost)

    assert result == ["equilibrium", "core_profiles"]
    assert cost.steps["candidate_route"] == pytest.approx(COST)
    assert cost.total_usd == pytest.approx(COST)


def test_default_thresholds_escalate_and_shortlist_top_five_in_jev_order():
    thresholds = get_mapping_route_thresholds()
    assert thresholds.select_threshold is None
    assert thresholds.floor_threshold is None
    assert thresholds.shortlist_size == 5

    scores = [0.9, 0.7, 0.5, 0.3, 0.1, 0.05, 0.01]
    judgments = [_judgment(f"ids/p{i}", score) for i, score in enumerate(scores)]

    result = route(judgments, thresholds)
    assert result is not None
    assert result.decision == "escalated"
    assert [j.p_same_quantity for j in result.shortlist] == scores[:5]


def test_select_threshold_returns_each_route_including_margin_case():
    selected = route(
        [_judgment("ids/p0", 0.9), _judgment("ids/p1", 0.2)],
        RouteThresholds(
            select_threshold=0.5,
            select_margin=0.1,
            floor_threshold=None,
            shortlist_size=5,
        ),
    )
    assert selected is not None
    assert selected.decision == "selected"

    margin = route(
        [_judgment("ids/p0", 0.9), _judgment("ids/p1", 0.85)],
        RouteThresholds(
            select_threshold=0.5,
            select_margin=0.1,
            floor_threshold=None,
            shortlist_size=5,
        ),
    )
    assert margin is not None
    assert margin.decision == "escalated"

    no_candidate = route(
        [_judgment("ids/p0", 0.4), _judgment("ids/p1", 0.3)],
        RouteThresholds(
            select_threshold=0.5,
            select_margin=0.1,
            floor_threshold=0.5,
            shortlist_size=5,
        ),
    )
    assert no_candidate is not None
    assert no_candidate.decision == "no_candidate"

    below_select_above_floor = route(
        [_judgment("ids/p0", 0.5)],
        RouteThresholds(
            select_threshold=0.6,
            select_margin=0.1,
            floor_threshold=0.3,
            shortlist_size=5,
        ),
    )
    assert below_select_above_floor is not None
    assert below_select_above_floor.decision == "escalated"


def test_route_ids_returns_three_ids_drawn_from_the_criteria_it_offered(monkeypatch):
    graph = _FakeGraph(
        {
            "equilibrium": "equilibrium fields",
            "core_profiles": "core plasma profiles",
            "magnetics": "magnetic measurements",
            "pf_active": "poloidal field coils",
            "wall": "wall description",
        }
    )
    captured: dict = {}

    def fake_post(headers, body, timeout):
        captured["body"] = body
        criteria = body["questions"]["ids_routing"]["criteria"]
        probabilities = dict.fromkeys(criteria, 0.0)
        probabilities["equilibrium"] = 0.5
        probabilities["core_profiles"] = 0.3
        probabilities["magnetics"] = 0.15
        probabilities["pf_active"] = 0.05
        return _FakeResponse(
            _payload(
                {
                    "ids_routing": {
                        "type": "choice",
                        "choice": "equilibrium",
                        "probabilities": probabilities,
                        "confidence": 0.8,
                    }
                }
            )
        )

    monkeypatch.setattr(llm, "_post_decisions", fake_post)

    offered = set(graph._ids)
    result = route_ids("plasma poloidal flux", gc=graph)

    assert result == ["equilibrium", "core_profiles", "magnetics"]
    assert set(result) <= offered
    assert set(captured["body"]["questions"]["ids_routing"]["criteria"]) == offered
    assert captured["body"]["model"] == MODEL
