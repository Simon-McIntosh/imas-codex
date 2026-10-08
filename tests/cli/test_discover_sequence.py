"""Tests for the discover run stage registry and its access probes."""

from __future__ import annotations

import importlib
from unittest.mock import MagicMock

import pytest
from click.testing import CliRunner

from imas_codex.cli.discover import sequence

STAGE_ORDER = [
    "paths",
    "code",
    "documents",
    "wiki",
    "signals scan",
    "signals enrich",
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


def test_registry_lists_stages_in_order():
    assert [s.name for s in sequence.STAGES] == STAGE_ORDER


def test_every_discovery_domain_has_a_stage():
    registered = {s.domain for s in sequence.STAGES if s.domain is not None}
    assert registered == set(sequence.DOMAINS)


def test_registry_records_the_documented_dependencies(healthy):
    assert _stage("code").reads == ("paths",)
    assert _stage("documents").reads == ("paths",)
    assert _stage("signals enrich").reads == ("signals scan",)
    assert set(_stage("signals enrich").context) == {"wiki", "code"}
    assert _stage("candidates").reads == ("signals enrich",)
    assert _stage("mapping").reads == ("candidates",)
    assert _stage("mapping").domain == "mapping"
    assert _stage("mapping").model_section == "ids-mapping"


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
    assert sequence.probe_access("wiki", cfg) == (True, "1 wiki sites reachable")
    graph.assert_called_once_with()
    ssh.assert_called_once_with("jt-60sa")
    wiki.assert_called_once_with("https://wiki.example", "jt-60sa")


def _probes(monkeypatch, *, graph=True, ssh=True, wiki=True):
    monkeypatch.setattr(sequence, "neo4j_health_check", lambda: (graph, "graph detail"))
    monkeypatch.setattr(sequence, "ssh_health_check", lambda host: (ssh, "ssh detail"))
    monkeypatch.setattr(
        sequence, "wiki_auth_check", lambda url, host=None: (wiki, "wiki detail")
    )


SSH_STAGES = ("paths", "code", "documents", "signals scan")
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


@pytest.mark.parametrize("name", [s.name for s in sequence.STAGES if s.pending])
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
    """A graph fault in a pending predicate reads unreachable, not empty."""
    _probes(monkeypatch, graph=True, ssh=True, wiki=True)

    from imas_codex.discovery.code import graph_ops

    def boom(facility: str) -> bool:
        raise RuntimeError("graph unavailable")

    monkeypatch.setattr(graph_ops, "has_pending_scan_work", boom)

    o = sequence.evaluate_stage(_stage("code"), "jt-60sa", _config())

    assert o.outcome == sequence.UNREACHABLE
    assert "pending-work query failed" in o.reason


def test_dry_run_prints_every_stage_and_runs_nothing(monkeypatch, healthy):
    monkeypatch.setattr(
        "imas_codex.discovery.base.facility.get_facility", lambda f: _config()
    )
    monkeypatch.setattr(sequence, "_stage_function", MagicMock())

    result = CliRunner().invoke(sequence.run, ["jt-60sa", "--dry-run"])

    assert result.exit_code == 0, result.output
    for domain in sequence.DOMAINS:
        assert domain in result.output
    assert result.output.count(sequence.RUNNABLE) == len(sequence.DOMAINS)
    assert sequence._stage_function.call_count == 0


def test_discover_hides_the_run_command():
    from imas_codex.cli.discover import discover

    listed = CliRunner().invoke(discover, ["--help"])
    assert listed.exit_code == 0, listed.output
    assert "run " not in listed.output

    helped = CliRunner().invoke(sequence.run, ["--help"])
    assert helped.exit_code == 0, helped.output
    assert "--dry-run" in helped.output


def test_bare_facility_routes_to_hidden_run_and_status_keeps_its_command(
    monkeypatch, healthy, tmp_path
):
    from imas_codex import discovery, settings
    from imas_codex.cli import logging as cli_logging
    from imas_codex.cli.discover import discover

    monkeypatch.setattr(cli_logging, "configure_cli_logging", lambda *a, **kw: None)
    monkeypatch.setattr(
        cli_logging, "get_log_file", lambda *a, **kw: tmp_path / "discover.log"
    )
    monkeypatch.setattr(
        "imas_codex.discovery.base.facility.get_facility", lambda f: _config()
    )
    dry = CliRunner().invoke(discover, ["jt-60sa", "--dry-run"])
    assert dry.exit_code == 0, dry.output
    assert "Discovery sequence for jt-60sa" in dry.output
    assert dry.output == (tmp_path / "discover.log").read_text()
    for domain in sequence.DOMAINS:
        assert domain in dry.output

    high_value = MagicMock(return_value=[])
    monkeypatch.setattr(discovery, "get_discovery_stats", lambda f: {"total": 0})
    monkeypatch.setattr(discovery, "get_high_value_paths", high_value)
    monkeypatch.setattr(settings, "get_path_scan_threshold", lambda: 0.63)
    status = CliRunner().invoke(
        discover, ["status", "jt-60sa", "--json", "-d", "paths"]
    )
    assert status.exit_code == 0, status.output
    assert '"high_value_paths"' in status.output
    high_value.assert_called_once_with("jt-60sa", min_score=0.63, limit=20)


def test_wiki_probe_checks_every_configured_site(monkeypatch):
    visited = []

    def check(url, host):
        visited.append((url, host))
        return (url != "https://down.example", "unreachable")

    monkeypatch.setattr(sequence, "wiki_auth_check", check)
    healthy, reason = sequence.probe_access(
        "wiki",
        _config(
            wiki_sites=[
                {"url": "https://up.example", "ssh_available": True},
                {"url": "https://down.example"},
            ]
        ),
    )
    assert not healthy
    assert "https://down.example" in reason
    assert visited == [
        ("https://up.example", "jt-60sa"),
        ("https://down.example", None),
    ]


def _recording_stages(monkeypatch, *, fail=None):
    calls = []

    def resolve(stage):
        def run(*args):
            calls.append((stage.name, args))
            if stage.name == fail:
                raise RuntimeError("stage crashed")
            return {"scanned": 2, "cost": 1.0, "remaining": 3}

        return run

    monkeypatch.setattr(sequence, "_stage_function", resolve)
    return calls


def test_stage_functions_run_in_order_with_remaining_limits(monkeypatch, healthy):
    calls = _recording_stages(monkeypatch)
    outcomes = sequence.run_sequence(
        "jt-60sa",
        config=_config(),
        options=sequence.SequenceOptions(cost_limit=12.0, time_limit=10),
    )
    assert [name for name, _ in calls] == STAGE_ORDER
    assert [outcome.outcome for outcome in outcomes] == [sequence.RAN] * len(
        STAGE_ORDER
    )
    assert [args[-2] for name, args in calls if name == "mapping"] == [5.0]
    assert 0 < calls[-1][1][-1] <= 10
    assert calls[0][1][1].cost_limit == 12.0
    assert calls[1][1][1].cost_limit == 11.0


def test_failure_skips_consumers_and_independent_domains_continue(monkeypatch, healthy):
    calls = _recording_stages(monkeypatch, fail="paths")
    outcomes = sequence.run_sequence("jt-60sa", config=_config())
    assert _outcome(outcomes, "paths").outcome == sequence.FAILED
    assert "stage crashed" in _outcome(outcomes, "paths").reason
    for name in ("code", "documents"):
        assert _outcome(outcomes, name).outcome == sequence.DEPENDENCY_FAILED
        assert name not in [called for called, _ in calls]
    assert _outcome(outcomes, "wiki").outcome == sequence.RAN
    assert _outcome(outcomes, "mapping").outcome == sequence.RAN


def test_selection_and_halves(monkeypatch, healthy):
    calls = _recording_stages(monkeypatch)
    selected = sequence.run_sequence(
        "jt-60sa",
        config=_config(),
        options=sequence.SequenceOptions(
            only=("signals", "candidates"), scan_only=True
        ),
    )
    assert [name for name, _ in calls] == ["signals scan"]
    assert calls[0][1][1].scan_only is True
    assert calls[0][1][1].flush is False
    assert _outcome(selected, "candidates").outcome == sequence.NOTHING_TO_SEED
    assert "nothing to seed" in _outcome(selected, "candidates").reason

    calls.clear()
    sequence.run_sequence(
        "jt-60sa",
        config=_config(),
        options=sequence.SequenceOptions(only=("signals",), flush=True),
    )
    assert [name for name, _ in calls] == ["signals enrich"]
    assert calls[0][1][1].flush is True
    assert calls[0][1][1].scan_only is False

    calls.clear()
    sequence.run_sequence(
        "jt-60sa",
        config=_config(),
        options=sequence.SequenceOptions(
            skip=("paths", "code", "documents", "wiki", "signals")
        ),
    )
    assert [name for name, _ in calls] == ["candidates", "mapping"]


def test_document_seeding_runs_before_any_document_is_pending(monkeypatch, healthy):
    _patch_predicates(monkeypatch, False)
    calls = _recording_stages(monkeypatch)
    sequence.run_sequence(
        "jt-60sa",
        config=_config(),
        options=sequence.SequenceOptions(only=("documents",), scan_only=True),
    )
    assert [name for name, _ in calls] == ["documents"]
    assert calls[0][1][1].scan_only is True


def test_cli_parses_repeated_and_comma_separated_domains(
    monkeypatch, healthy, tmp_path
):
    from imas_codex.cli import logging as cli_logging
    from imas_codex.cli.discover import discover

    monkeypatch.setattr(cli_logging, "configure_cli_logging", lambda *a, **kw: None)
    monkeypatch.setattr(
        cli_logging, "get_log_file", lambda *a, **kw: tmp_path / "discover.log"
    )
    monkeypatch.setattr(
        "imas_codex.discovery.base.facility.get_facility", lambda f: _config()
    )
    calls = _recording_stages(monkeypatch)
    result = CliRunner().invoke(
        discover,
        ["jt-60sa", "--only", "paths,signals", "--only", "wiki", "--skip", "wiki"],
    )
    assert result.exit_code == 0, result.output
    assert [name for name, _ in calls] == ["paths", "signals scan", "signals enrich"]


@pytest.mark.parametrize(
    "arguments",
    [
        ["--reset-to", "scanned"],
        ["--only", "paths,code", "--reset-to", "scanned"],
        ["--scan-only", "--flush"],
    ],
)
def test_conflicting_options_are_refused(arguments):
    result = CliRunner().invoke(sequence.run, ["jt-60sa", *arguments])
    assert result.exit_code == 2


def test_final_table_lists_every_domain(monkeypatch, healthy, capsys):
    _recording_stages(monkeypatch)
    sequence.run_sequence("jt-60sa", config=_config())
    report = capsys.readouterr().out
    assert report.count("Discovery sequence for jt-60sa") == 1
    for domain in sequence.DOMAINS:
        assert (
            sum(line.startswith(domain.ljust(10)) for line in report.splitlines()) == 1
        )
    assert "Done  Remaining  Cost     Time" in report


def test_mapping_calls_existing_pipeline_with_remaining_limits(monkeypatch):
    from imas_codex.cli.map import map_run

    callback = MagicMock()
    monkeypatch.setattr(map_run, "callback", callback)
    sequence.run_mapping_stage(
        "jt-60sa",
        sequence.SequenceOptions(physics_domain=("equilibrium",), ids=("pf_active",)),
        3.5,
        7,
    )
    callback.assert_called_once()
    assert callback.call_args.kwargs["cost_limit"] == 3.5
    assert callback.call_args.kwargs["time_limit"] == 7
    assert callback.call_args.kwargs["domains"] == ("equilibrium",)
    assert callback.call_args.kwargs["ids_names"] == ("pf_active",)


def test_no_candidate_route_skips_mapping_as_nothing_to_do(monkeypatch, healthy):
    from imas_codex.ids import workers

    monkeypatch.setattr(workers, "has_pending_mapping_work", lambda facility: False)
    monkeypatch.setattr(workers, "has_pending_validation_work", lambda facility: False)
    outcome = sequence.evaluate_stage(_stage("mapping"), "jt-60sa", _config())
    assert outcome.outcome == sequence.NOTHING_TO_DO


def test_mapping_focus_and_limit_restrict_ids_targets(monkeypatch, tmp_path):
    from imas_codex.cli.map import map_run
    from imas_codex.graph import client
    from imas_codex.ids import tools

    manifest = tmp_path / "ids.txt"
    manifest.write_text("equilibrium\npf_active\n")
    callback = MagicMock()
    monkeypatch.setattr(map_run, "callback", callback)
    monkeypatch.setattr(client, "GraphClient", MagicMock())
    selection = MagicMock(
        return_value={
            "ids_targets": [
                {"ids_name": "pf_active"},
                {"ids_name": "equilibrium"},
            ]
        }
    )
    monkeypatch.setattr(tools, "discover_mappable_ids", selection)

    result = sequence.run_mapping_stage(
        "jt-60sa",
        sequence.SequenceOptions(focus=(str(manifest),), limit=1),
        3.5,
        7,
    )
    assert selection.call_args.kwargs["ids_filter"] == ["equilibrium", "pf_active"]
    assert callback.call_args.kwargs["domains"] == ()
    assert callback.call_args.kwargs["ids_names"] == ("equilibrium",)
    assert result["remaining"] == 1


def test_signals_halves_resolve_to_the_signals_stage_function():
    from imas_codex.cli.discover.signals import run_signals_stage

    assert sequence._stage_function(_stage("signals scan")) is run_signals_stage
    assert sequence._stage_function(_stage("signals enrich")) is run_signals_stage


def test_stage_without_return_still_supplies_its_cost_receipt(monkeypatch):
    from imas_codex.cli.discover import common

    monkeypatch.setattr(
        common, "run_discovery", lambda *args, **kwargs: {"cost": 2.25, "scanned": 4}
    )

    def stage(facility, options):
        common.run_discovery(None, None)

    result = sequence._run_with_receipt(stage, "jt-60sa", object())
    assert result == {"cost": 2.25, "scanned": 4}


def test_failed_stage_receipt_reduces_independent_stage_budget(monkeypatch, healthy):
    from imas_codex.cli.discover import common

    monkeypatch.setattr(common, "run_discovery", lambda *a, **kw: {"cost": 2.0})
    received = []

    def resolve(stage):
        def run(facility, options):
            if stage.name == "paths":
                common.run_discovery(None, None)
                raise RuntimeError("failed after receipt")
            received.append((stage.name, options.cost_limit))
            return {"cost": 0.0}

        return run

    monkeypatch.setattr(sequence, "_stage_function", resolve)
    outcomes = sequence.run_sequence(
        "jt-60sa",
        config=_config(),
        options=sequence.SequenceOptions(only=("paths", "wiki"), cost_limit=10),
    )
    assert _outcome(outcomes, "paths").cost == 2.0
    assert ("wiki", 8.0) in received
