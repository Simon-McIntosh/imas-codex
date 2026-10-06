"""Drive the candidate judgment command and its worker.

The worker tests run ``candidate_worker`` against a candidate
:class:`CandidateDiscoveryState` with the claim, routing, retrieval, judgment
and persistence calls replaced, so the batch pipeline, the one-route-per-source
outcome and the retrieval scoping are measured without a live graph or a live
decisions endpoint. The cost-limit test wraps the decisions-cost seam to prove
the loop halts once the run's spend reaches the limit — the negative control
for the scoped-retrieval assertion lives here as
``test_candidate_worker_scopes_retrieval_to_routed_ids``.

The command tests invoke ``discover map`` with the engine replaced, asserting
the options reach the engine state, and cover the ``status --domain map`` route
counts and the ``clear --domain map`` removal counts.
"""

from __future__ import annotations

import asyncio
from contextlib import contextmanager

import pytest
from click.testing import CliRunner

from imas_codex.cli.discover import discover
from imas_codex.ids.candidates import PairJudgment
from imas_codex.ids.workers import CandidateDiscoveryState, candidate_worker
from imas_codex.settings import RouteThresholds

FACILITY = "jet"

# One routed IDS per fixture source; the description is the routing key.
_IDS_BY_DESCRIPTION = {
    "sel": ["equilibrium"],
    "esc": ["core_profiles"],
    "none": ["magnetics"],
}

# Per-source judged scores chosen so the real ``route`` yields one route each
# under the thresholds below: selected, escalated and no_candidate.
_SCORES = {
    "src-sel": [0.9, 0.2],
    "src-esc": [0.5, 0.5],
    "src-none": [0.1],
}

_THRESHOLDS = RouteThresholds(
    select_threshold=0.5,
    select_margin=0.1,
    floor_threshold=0.4,
    shortlist_size=5,
)

SOURCES = [
    {"id": "src-sel", "description": "sel"},
    {"id": "src-esc", "description": "esc"},
    {"id": "src-none", "description": "none"},
]


class _FakeGraph:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _judgment(path: str, probability: float) -> PairJudgment:
    return PairJudgment(
        path=path,
        p_same_quantity=probability,
        model="jev-1.13",
        judged_at="2026-10-06T00:00:00+00:00",
    )


class _Candidate:
    class _Hit:
        def __init__(self, path, score):
            self.path = path
            self.score = score
            self.ids_name = path.split("/", 1)[0]

    def __init__(self, path, score):
        self.hit = self._Hit(path, score)


def _patch_worker(monkeypatch, captured: dict, *, add_cost: float = 0.0):
    """Replace every graph and decisions call the candidate worker makes."""
    remaining = list(SOURCES)

    def fake_claim(facility, domains=None, batch_size=10):
        if not remaining:
            return []
        size = min(batch_size, len(remaining))
        batch, remaining[:] = remaining[:size], remaining[size:]
        return batch

    def fake_route_ids(description, *, gc, model=None):
        return list(_IDS_BY_DESCRIPTION[description])

    def fake_retrieve(sources, ids_by_source, *, gc, k=20, dd_version=None):
        captured.setdefault("ids_by_source", {}).update(ids_by_source)
        return {
            sid: ([_Candidate(f"{ids[0]}/field", 0.7)] if ids else [])
            for sid, ids in ids_by_source.items()
        }

    def fake_judge(source, facility, candidates, *, model=None):
        return [
            _judgment(f"{source['id']}/p{i}", p)
            for i, p in enumerate(_SCORES[source["id"]])
        ]

    def fake_write(source_id, records, route, gc):
        captured.setdefault("routes", {})[source_id] = route
        captured.setdefault("records", {})[source_id] = list(records)
        return len(records)

    monkeypatch.setattr(
        "imas_codex.ids.workers.claim_sources_for_candidates", fake_claim
    )
    monkeypatch.setattr("imas_codex.ids.workers.route_ids", fake_route_ids)
    monkeypatch.setattr("imas_codex.ids.workers.retrieve_candidates", fake_retrieve)
    monkeypatch.setattr("imas_codex.ids.workers.judge_candidates", fake_judge)
    monkeypatch.setattr("imas_codex.ids.workers.write_candidates", fake_write)
    monkeypatch.setattr(
        "imas_codex.ids.workers.get_mapping_route_thresholds", lambda: _THRESHOLDS
    )
    monkeypatch.setattr("imas_codex.ids.workers.GraphClient", lambda: _FakeGraph())

    if add_cost:

        @contextmanager
        def fake_capture(cost, step):
            yield
            cost.add(step, add_cost, 0)

        monkeypatch.setattr(
            "imas_codex.ids.workers._capture_decisions_cost", fake_capture
        )


def _new_state(**kwargs) -> CandidateDiscoveryState:
    # A finite source limit ends the loop deterministically without waiting on
    # the phase's idle/``has_work_fn`` completion path, which the supervision
    # loop drives in production.
    kwargs.setdefault("source_limit", 3)
    return CandidateDiscoveryState(facility=FACILITY, **kwargs)


def test_candidate_worker_writes_one_route_per_source(monkeypatch):
    captured: dict = {}
    _patch_worker(monkeypatch, captured)
    state = _new_state(batch_size=10)

    asyncio.run(candidate_worker(state))

    assert captured["routes"] == {
        "src-sel": "selected",
        "src-none": "no_candidate",
        "src-esc": "escalated",
    }
    assert state.sources_judged == 3
    assert state.candidates_written == 5
    # The selected source marks exactly its top candidate.
    selected = captured["records"]["src-sel"]
    assert [r["route"] for r in selected] == [True, False]


def test_candidate_worker_scopes_retrieval_to_routed_ids(monkeypatch):
    """Retrieval receives the routed IDSs, never an unscoped mapping.

    Negative control: make ``candidate_worker`` skip the ``route_ids`` call so
    ``routed`` stays empty; this assertion then fails because
    ``retrieve_candidates`` receives ``{}`` (or the sources are searched
    unscoped) instead of the per-source IDSs named here.
    """
    captured: dict = {}
    _patch_worker(monkeypatch, captured)
    state = _new_state(batch_size=10)

    asyncio.run(candidate_worker(state))

    assert captured["ids_by_source"] == {
        "src-sel": ["equilibrium"],
        "src-esc": ["core_profiles"],
        "src-none": ["magnetics"],
    }


def test_candidate_worker_ids_filter_narrows_routed_ids(monkeypatch):
    captured: dict = {}
    _patch_worker(monkeypatch, captured)
    state = _new_state(batch_size=10, ids_filter=["equilibrium", "magnetics"])

    asyncio.run(candidate_worker(state))

    # core_profiles is dropped, leaving src-esc routed to no IDS.
    assert captured["ids_by_source"] == {
        "src-sel": ["equilibrium"],
        "src-esc": [],
        "src-none": ["magnetics"],
    }


def test_cost_limit_stops_the_loop(monkeypatch):
    captured: dict = {}
    _patch_worker(monkeypatch, captured, add_cost=0.002)
    state = _new_state(batch_size=1, cost_limit=0.001)

    asyncio.run(candidate_worker(state))

    assert state.sources_judged == 1
    assert state.cost.total_usd >= 0.001


def test_source_limit_bounds_the_loop(monkeypatch):
    captured: dict = {}
    _patch_worker(monkeypatch, captured)
    state = _new_state(batch_size=10, source_limit=2)

    asyncio.run(candidate_worker(state))

    assert state.sources_judged == 2


# =============================================================================
# Command wiring
# =============================================================================


@pytest.fixture
def command_env(monkeypatch):
    monkeypatch.setattr("imas_codex.cli.discover.common.use_rich_output", lambda: False)
    monkeypatch.setattr(
        "imas_codex.discovery.base.facility.get_facility",
        lambda name: {"id": name},
    )


def test_map_command_builds_the_engine_state(monkeypatch, command_env, caplog):
    import logging

    caplog.set_level(logging.INFO, logger="imas_codex.discovery.map")
    captured: dict = {}

    async def fake_engine(state, *, stop_event=None, on_progress=None):
        captured["state"] = state
        state.sources_judged = 3
        state.candidates_written = 4
        state.cost.add("candidate", 0.25, 0)

    monkeypatch.setattr("imas_codex.ids.workers.run_candidate_engine", fake_engine)

    result = CliRunner().invoke(
        discover,
        [
            "map",
            FACILITY,
            "-d",
            "magnetics",
            "-i",
            "equilibrium",
            "-c",
            "2.0",
            "-n",
            "50",
            "--time",
            "5",
        ],
    )

    assert result.exit_code == 0, result.output
    state = captured["state"]
    assert state.facility == FACILITY
    assert state.domains == ["magnetics"]
    assert state.ids_filter == ["equilibrium"]
    assert state.cost_limit == 2.0
    assert state.source_limit == 50
    assert state.deadline is not None
    assert "3 sources judged" in caplog.text
    assert "$0.25" in caplog.text


def test_status_domain_map_prints_route_counts(monkeypatch):
    monkeypatch.setattr("imas_codex.cli.discover.common.use_rich_output", lambda: False)
    monkeypatch.setattr(
        "imas_codex.ids.workers.count_candidates_by_route",
        lambda facility: {"selected": 2, "escalated": 3, "pending": 5},
    )

    result = CliRunner().invoke(discover, ["status", FACILITY, "-d", "map"])

    assert result.exit_code == 0, result.output
    assert "selected: 2" in result.output
    assert "escalated: 3" in result.output
    assert "pending: 5" in result.output


def test_clear_domain_map_removes_candidates(monkeypatch):
    monkeypatch.setattr(
        "imas_codex.ids.workers.count_candidates_by_route",
        lambda facility: {"selected": 2, "pending": 5},
    )
    monkeypatch.setattr(
        "imas_codex.cli.discover.map.clear_facility_candidates",
        lambda facility: {"edges_removed": 7, "routes_reset": 2},
    )

    result = CliRunner().invoke(discover, ["clear", FACILITY, "-d", "map", "--force"])

    assert result.exit_code == 0, result.output
    assert "7 edges removed" in result.output
    assert "2 routes reset" in result.output
