"""Tests for the discover run stage registry and its access probes."""

from __future__ import annotations

import importlib
from unittest.mock import MagicMock

import pytest
from click.testing import CliRunner

from imas_codex.cli.discover import sequence
from imas_codex.cli.discover.common import DISCOVERY_DOMAINS

STAGE_ORDER = [
    "paths",
    "code",
    "documents",
    "wiki",
    "signals scan",
    "signals enrich",
    "signals check",
    "candidates",
    "mapping",
]


def _config(**over) -> dict:
    cfg = {
        "id": "jt-60sa",
        "ssh_host": "jt-60sa",
        "discovery_roots": ["/home"],
        "data_systems": {"edas": {}},
        "wiki_sites": [{"url": "https://wiki.example", "ssh_available": True}],
    }
    cfg.update(over)
    return cfg


def _stage(name: str) -> sequence.Stage:
    return next(s for s in sequence.STAGES if s.name == name)


def _outcome(outcomes, name: str) -> sequence.StageOutcome:
    return next(o for o in outcomes if o.stage == name)


@pytest.fixture
def healthy(monkeypatch):
    """Patch the three access checks healthy, and every predicate true."""
    monkeypatch = monkeypatch
    monkeypatch.setattr(sequence, "neo4j_health_check", lambda: (True, "graph"))
    monkeypatch.setattr(sequence, "ssh_health_check", lambda host: (True, host))
    monkeypatch.setattr(sequence, "wiki_auth_check", lambda url, host=None: (True, url))
    _patch_predicates(monkeypatch, True)
    return monkeypatch


def _patch_predicates(monkeypatch, value: bool) -> None:
    """Make every stage's pending-work predicate return ``value``."""
    for stage in sequence.STAGES:
        for spec in stage.pending:
            module_name, func = spec.rsplit(":", 1)
            module = importlib.import_module(module_name)
            monkeypatch.setattr(module, func, lambda facility, _v=value: _v)


def test_registry_lists_nine_stages_in_table_order():
    assert [s.name for s in sequence.STAGES] == STAGE_ORDER


def test_every_discovery_domain_has_a_stage():
    registered = {s.domain for s in sequence.STAGES if s.domain is not None}
    assert registered == set(DISCOVERY_DOMAINS)


def test_registry_records_the_documented_dependencies(healthy):
    assert _stage("code").reads == ("paths",)
    assert _stage("documents").reads == ("paths",)
    assert _stage("signals enrich").reads == ("signals scan",)
    assert set(_stage("signals enrich").context) == {"wiki", "code"}
    assert _stage("signals check").reads == ("signals enrich",)
    assert _stage("candidates").reads == ("signals enrich",)
    assert _stage("mapping").reads == ("candidates",)
    assert _stage("mapping").domain is None


def test_probe_access_calls_the_canonical_checks(monkeypatch):
    graph = MagicMock(return_value=(True, "g"))
    ssh = MagicMock(return_value=(True, "s"))
    wiki = MagicMock(return_value=(True, "w"))
    monkeypatch.setattr(sequence, "neo4j_health_check", graph)
    monkeypatch.setattr(sequence, "ssh_health_check", ssh)
    monkeypatch.setattr(sequence, "wiki_auth_check", wiki)

    cfg = _config()
    assert sequence.probe_access("graph", cfg) == (True, "g")
    assert sequence.probe_access("ssh", cfg) == (True, "s")
    assert sequence.probe_access("wiki", cfg) == (True, "w")
    graph.assert_called_once_with()
    ssh.assert_called_once_with("jt-60sa")
    wiki.assert_called_once_with("https://wiki.example", "jt-60sa")


def _probes(monkeypatch, *, graph=True, ssh=True, wiki=True):
    monkeypatch.setattr(sequence, "neo4j_health_check", lambda: (graph, "graph detail"))
    monkeypatch.setattr(sequence, "ssh_health_check", lambda host: (ssh, "ssh detail"))
    monkeypatch.setattr(
        sequence, "wiki_auth_check", lambda url, host=None: (wiki, "wiki detail")
    )


SSH_STAGES = ("paths", "code", "documents", "signals scan", "signals check")
GRAPH_STAGES = ("signals enrich", "candidates", "mapping")


def test_ssh_down_reads_ssh_stages_unreachable_and_graph_stages_runnable(monkeypatch):
    _probes(monkeypatch, graph=True, ssh=False, wiki=False)
    _patch_predicates(monkeypatch, True)

    outcomes = sequence.evaluate_plan("jt-60sa", _config())

    for name in SSH_STAGES:
        o = _outcome(outcomes, name)
        assert o.outcome == sequence.UNREACHABLE
        assert o.reason.startswith("ssh:")
    for name in GRAPH_STAGES:
        assert _outcome(outcomes, name).outcome == sequence.RUNNABLE


def test_wiki_without_sites_reads_not_configured(monkeypatch):
    _probes(monkeypatch, graph=True, ssh=True)
    _patch_predicates(monkeypatch, True)

    outcomes = sequence.evaluate_plan("jt-60sa", _config(wiki_sites=[]))

    o = _outcome(outcomes, "wiki")
    assert o.outcome == sequence.NOT_CONFIGURED
    assert "wiki_sites" in o.reason


def test_graph_check_governs_the_graph_only_stages(monkeypatch):
    _probes(monkeypatch, graph=False, ssh=True, wiki=True)
    _patch_predicates(monkeypatch, True)

    down = sequence.evaluate_plan("jt-60sa", _config())
    assert _outcome(down, "paths").outcome == sequence.RUNNABLE
    for name in GRAPH_STAGES:
        o = _outcome(down, name)
        assert o.outcome == sequence.UNREACHABLE
        assert o.reason.startswith("graph:")

    _probes(monkeypatch, graph=True, ssh=True, wiki=True)
    up = sequence.evaluate_plan("jt-60sa", _config())
    for name in GRAPH_STAGES:
        assert _outcome(up, name).outcome == sequence.RUNNABLE


@pytest.mark.parametrize("name", STAGE_ORDER)
def test_each_stage_follows_its_named_pending_predicate(monkeypatch, name):
    _probes(monkeypatch, graph=True, ssh=True, wiki=True)
    stage = _stage(name)

    _patch_predicates(monkeypatch, True)
    assert sequence.evaluate_stage(stage, "jt-60sa", _config()).outcome == (
        sequence.RUNNABLE
    )

    _patch_predicates(monkeypatch, False)
    o = sequence.evaluate_stage(stage, "jt-60sa", _config())
    assert o.outcome == sequence.NOTHING_TO_DO
    for pending_name in stage.pending_names:
        assert pending_name in o.reason


def test_raising_pending_query_reads_unreachable(monkeypatch):
    """A graph fault in the documents predicate reads unreachable, not empty."""
    _probes(monkeypatch, graph=True, ssh=True, wiki=True)

    from imas_codex.discovery.documents import pipeline

    def boom(facility: str) -> bool:
        raise RuntimeError("graph unavailable")

    monkeypatch.setattr(pipeline, "_has_pending_image_documents", boom)

    o = sequence.evaluate_stage(_stage("documents"), "jt-60sa", _config())

    assert o.outcome == sequence.UNREACHABLE
    assert "pending-work query failed" in o.reason


def test_dry_run_prints_every_stage_and_runs_nothing(monkeypatch, healthy):
    monkeypatch.setattr(
        "imas_codex.discovery.base.facility.get_facility", lambda f: _config()
    )
    monkeypatch.setattr(sequence, "stage_command", MagicMock())

    result = CliRunner().invoke(sequence.run, ["jt-60sa", "--dry-run"])

    assert result.exit_code == 0, result.output
    for name in STAGE_ORDER:
        assert name in result.output
    assert result.output.count(sequence.RUNNABLE) == len(STAGE_ORDER)
    assert sequence.stage_command.call_count == 0


def test_discover_exposes_the_run_command():
    from imas_codex.cli.discover import discover

    listed = CliRunner().invoke(discover, ["--help"])
    assert listed.exit_code == 0, listed.output
    assert "run" in listed.output

    helped = CliRunner().invoke(sequence.run, ["--help"])
    assert helped.exit_code == 0, helped.output
    assert "--dry-run" in helped.output
