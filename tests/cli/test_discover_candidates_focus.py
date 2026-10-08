"""A focused candidate run claims only the named SignalSource identities."""

from __future__ import annotations

import click
import pytest
from click.testing import CliRunner

from imas_codex.cli.discover import discover
from imas_codex.cli.discover.map import CandidatesStageOptions, run_candidates_stage

FACILITY = "jet"
SOURCES = ({"id": "source-a"}, {"id": "source-b"})


@pytest.fixture
def focus_env(monkeypatch):
    monkeypatch.setattr("imas_codex.cli.discover.common.use_rich_output", lambda: False)
    monkeypatch.setattr(
        "imas_codex.discovery.base.facility.get_facility",
        lambda name: {"id": name},
    )

    class Graph:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def query(self, cypher, **params):
            assert "SignalSource" in cypher
            assert params["facility"] == FACILITY
            return [
                {"id": item["id"]} for item in SOURCES if item["id"] in params["ids"]
            ]

    monkeypatch.setattr("imas_codex.graph.GraphClient", Graph)
    claimed = []

    def fake_claim(label, **kwargs):
        assert label == "SignalSource"
        assert kwargs["facility"] == FACILITY
        predicate = kwargs["status_predicate"]
        assert "$domainsAND" not in predicate
        selected = list(SOURCES)
        if "n.id IN $focus_ids" in predicate:
            selected = [
                source
                for source in selected
                if source["id"] in kwargs["status_params"]["focus_ids"]
            ]
        claimed.extend(source["id"] for source in selected)
        return selected

    monkeypatch.setattr("imas_codex.ids.workers.claim_batch", fake_claim)

    async def fake_engine(state, *, stop_event=None, on_progress=None):
        from imas_codex.ids.workers import claim_sources_for_candidates

        sources = claim_sources_for_candidates(
            state.facility,
            domains=state.domains or None,
            focus_ids=state.focus_ids or None,
        )
        state.sources_judged = len(sources)

    monkeypatch.setattr("imas_codex.ids.workers.run_candidate_engine", fake_engine)
    return claimed


def test_named_sources_are_the_only_claims(focus_env):
    result = run_candidates_stage(
        FACILITY,
        CandidatesStageOptions(focus=("source-a",), physics_domain=("magnetics",)),
    )

    assert focus_env == ["source-a"]
    assert result["sources_judged"] == 1


def test_manifest_produces_the_same_claims(focus_env, tmp_path):
    manifest = tmp_path / "signals.yaml"
    manifest.write_text("sources:\n  - source-a\n", encoding="utf-8")

    result = run_candidates_stage(
        FACILITY, CandidatesStageOptions(focus=(str(manifest),))
    )

    assert focus_env == ["source-a"]
    assert result["sources_judged"] == 1


def test_unknown_source_is_refused_by_name(focus_env):
    with pytest.raises(click.UsageError, match="missing-source"):
        run_candidates_stage(
            FACILITY, CandidatesStageOptions(focus=("source-a", "missing-source"))
        )

    assert focus_env == []


def test_map_command_passes_focus_to_the_stage(monkeypatch):
    received = []

    def fake_stage(facility, options):
        received.append((facility, options.focus))

    monkeypatch.setattr("imas_codex.cli.discover.map.run_candidates_stage", fake_stage)
    result = CliRunner().invoke(
        discover, ["map", FACILITY, "--focus", "source-a", "--focus", "source-b"]
    )

    assert result.exit_code == 0, result.output
    assert received == [(FACILITY, ("source-a", "source-b"))]
