"""Drive the candidate stage function and its thin click wrapper.

The candidate stage drains the sources the signals stage seeded; it has no
seeding half. So ``--scan-only`` runs no worker (there is nothing to seed),
while ``--flush`` and the default run the single draining worker. ``--limit``
caps the sources judged this run, ``--focus`` is refused because the claim
query takes no item filter yet, and ``--topic`` is carried onto the stage for
a scorer to steer (candidates has no free-text focus target of its own, so the
topic reaches no worker here).

``run_discovery`` and ``run_candidate_engine`` are replaced, so each test
measures which half the stage selects without a live graph or decisions
endpoint. The negative control lives in the manifest's ``negative_control_log``.
"""

from __future__ import annotations

import logging

import click
import pytest
from click.testing import CliRunner

from imas_codex.cli.discover import discover
from imas_codex.cli.discover.map import (
    CandidatesStageOptions,
    run_candidates_stage,
)

FACILITY = "jet"


@pytest.fixture
def stage_env(monkeypatch):
    """Plain-text mode and a stub facility config, so the stage needs no graph."""
    monkeypatch.setattr("imas_codex.cli.discover.common.use_rich_output", lambda: False)
    monkeypatch.setattr(
        "imas_codex.discovery.base.facility.get_facility",
        lambda name: {"id": name},
    )


@pytest.fixture
def engine_calls(monkeypatch):
    """Replace the candidate engine with a recorder; return the call list.

    The recorder captures each ``CandidateDiscoveryState`` the stage builds, so
    a test can assert both whether the draining worker ran and the options that
    reached it.
    """
    calls: list = []

    async def fake_engine(state, *, stop_event=None, on_progress=None):
        calls.append(state)
        state.sources_judged = 2
        state.candidates_written = 3
        state.cost.add("candidate", 0.1, 0)

    monkeypatch.setattr("imas_codex.ids.workers.run_candidate_engine", fake_engine)
    return calls


def test_scan_only_runs_no_worker(stage_env, engine_calls):
    """A --scan-only pass has no seeding half, so no worker runs."""
    result = run_candidates_stage(FACILITY, CandidatesStageOptions(scan_only=True))

    assert engine_calls == []
    assert result["sources_judged"] == 0
    assert result["candidates_written"] == 0


def test_flush_runs_the_draining_worker(stage_env, engine_calls):
    """--flush selects the draining half: the candidate worker runs once."""
    run_candidates_stage(FACILITY, CandidatesStageOptions(flush=True))

    assert len(engine_calls) == 1
    assert engine_calls[0].facility == FACILITY


def test_default_runs_the_draining_worker(stage_env, engine_calls):
    """Neither flag selects the full stage, whose only half is the drain."""
    run_candidates_stage(FACILITY, CandidatesStageOptions())

    assert len(engine_calls) == 1


def test_limit_caps_items(stage_env, engine_calls):
    """--limit caps the sources judged this run (the engine's source_limit)."""
    run_candidates_stage(FACILITY, CandidatesStageOptions(limit=7))

    assert engine_calls[0].source_limit == 7


def test_stage_options_reach_the_engine_state(stage_env, engine_calls):
    run_candidates_stage(
        FACILITY,
        CandidatesStageOptions(
            physics_domain=("magnetics",),
            ids=("equilibrium",),
            cost_limit=2.0,
            limit=50,
            time_limit=5,
        ),
    )

    state = engine_calls[0]
    assert state.domains == ["magnetics"]
    assert state.ids_filter == ["equilibrium"]
    assert state.cost_limit == 2.0
    assert state.source_limit == 50
    assert state.deadline is not None


def test_focus_is_refused_naming_section_7(stage_env):
    """--focus is refused, never ignored silently, and names its plan section."""
    with pytest.raises(click.UsageError) as excinfo:
        run_candidates_stage(FACILITY, CandidatesStageOptions(focus=("equilibrium",)))

    message = str(excinfo.value)
    assert "facility-discovery-sequence" in message
    assert "section 7" in message


def test_topic_is_carried_to_the_stage(stage_env, engine_calls, caplog):
    """--topic is accepted and recorded rather than dropped.

    Candidates has no free-text focus target of its own (the map stage never
    had one), so the topic reaches no worker; the stage still carries and logs
    it so the settled surface is honoured uniformly.
    """
    caplog.set_level(logging.INFO, logger="imas_codex.discovery.map")

    run_candidates_stage(FACILITY, CandidatesStageOptions(topic="equilibrium"))

    assert len(engine_calls) == 1
    assert "Topic: equilibrium" in caplog.text


# =============================================================================
# The click command is a thin wrapper
# =============================================================================


def test_map_command_builds_options_and_delegates(monkeypatch):
    captured: dict = {}

    def fake_stage(facility, options):
        captured["facility"] = facility
        captured["options"] = options
        return {"sources_judged": 0, "candidates_written": 0, "cost": 0.0}

    monkeypatch.setattr("imas_codex.cli.discover.map.run_candidates_stage", fake_stage)

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
    assert captured["facility"] == FACILITY
    assert captured["options"] == CandidatesStageOptions(
        physics_domain=("magnetics",),
        ids=("equilibrium",),
        cost_limit=2.0,
        limit=50,
        time_limit=5,
    )
