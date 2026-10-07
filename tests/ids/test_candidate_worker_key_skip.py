"""The candidate worker ends its stage before claiming when the decisions key is absent.

With ``OPENROUTER_API_KEY_IMAS_CODEX`` unset ``candidate_worker`` must not claim
a single source: every claim it made could only fail to judge and be released,
so the stage would claim and release in a loop until its deadline. The worker
checks ``judgments_available`` once before the first claim, logs one warning
naming the missing variable, marks its phase done and returns. The routing and
judgment entry points are replaced with recorders whose assertions fail if they
are entered, so a stage that still claimed would be caught.

Negative control: remove the up-front ``judgments_available`` check from
``candidate_worker`` and this test fails — the claim recorder is entered and the
worker releases and reclaims in a loop until the deadline.
"""

from __future__ import annotations

import asyncio
import logging
import time

import pytest

from imas_codex.ids import candidates as cand
from imas_codex.ids.workers import CandidateDiscoveryState, candidate_worker

FACILITY = "jet"

# Far enough out that a stage which returns promptly is unambiguously "well
# before" its deadline, yet close enough that the negative control's
# claim/release loop terminates inside the test.
_DEADLINE_SECONDS = 10.0


@pytest.fixture(autouse=True)
def _no_decisions_key(monkeypatch):
    """Remove the decisions key and start each test with a fresh warning latch."""
    monkeypatch.delenv("OPENROUTER_API_KEY_IMAS_CODEX", raising=False)
    monkeypatch.setattr(cand, "_key_absence_warned", False)


class _FakeGraph:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _state() -> CandidateDiscoveryState:
    return CandidateDiscoveryState(
        facility=FACILITY, deadline=time.time() + _DEADLINE_SECONDS
    )


def test_candidate_worker_ends_before_claiming_without_the_decisions_key(
    monkeypatch, caplog
):
    claimed: list[str] = []
    routed: list[str] = []
    judged: list[str] = []
    released: list[str] = []

    def fake_claim(facility, domains=None, batch_size=10):
        claimed.append(facility)
        return [{"id": "src-1", "description": "plasma current"}]

    def fake_route_ids(description, *, gc, model=None, cost=None):
        routed.append(description)
        return None

    def fake_judge(source, facility, candidates, *, model=None, cost=None, step=None):
        judged.append(source["id"])
        return None

    monkeypatch.setattr(
        "imas_codex.ids.workers.claim_sources_for_candidates", fake_claim
    )
    monkeypatch.setattr("imas_codex.ids.workers.route_ids", fake_route_ids)
    monkeypatch.setattr("imas_codex.ids.workers.judge_candidates", fake_judge)
    monkeypatch.setattr(
        "imas_codex.ids.workers.release_mapping_claim",
        lambda source_id: released.append(source_id),
    )
    monkeypatch.setattr("imas_codex.ids.workers.GraphClient", lambda: _FakeGraph())

    state = _state()
    start = time.monotonic()
    with caplog.at_level(logging.WARNING):
        asyncio.run(candidate_worker(state))
    elapsed = time.monotonic() - start

    # No source is claimed, no routing or judgment entry point is entered and no
    # claim is released, because the stage ends before the first claim.
    assert claimed == []
    assert routed == []
    assert judged == []
    assert released == []
    assert state.candidate_phase.done
    # The stage returned promptly, not on its deadline.
    assert elapsed < _DEADLINE_SECONDS / 2

    warnings = [
        record
        for record in caplog.records
        if record.levelno == logging.WARNING
        and "OPENROUTER_API_KEY_IMAS_CODEX" in record.getMessage()
    ]
    assert len(warnings) == 1
