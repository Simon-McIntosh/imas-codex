"""CLI checks for the generated, validated, active mapping lifecycle."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from click.testing import CliRunner

from imas_codex.cli.discover import sequence
from imas_codex.cli.map import map_cmd


@pytest.fixture
def mapping_graph(monkeypatch):
    graph = MagicMock()
    graph.status = "generated"

    def query(statement, **params):
        if "SET m.status = 'active'" in statement:
            graph.status = "active"
            return [{"status": "active"}]
        if "SET m.status = 'validated'" in statement:
            graph.status = "validated"
            return [{"status": "validated"}]
        if "RETURN m.status AS status" in statement:
            return [{"status": graph.status}]
        raise AssertionError(f"unexpected graph query: {statement}")

    graph.query.side_effect = query
    monkeypatch.setattr("imas_codex.graph.client.GraphClient", lambda: graph)
    monkeypatch.setattr(
        "imas_codex.cli.map.configure_cli_logging", lambda *a, **kw: None
    )
    return graph


@pytest.fixture
def mapping_engine(monkeypatch, mapping_graph):
    from imas_codex.cli.discover import common
    from imas_codex.ids import tools, workers

    monkeypatch.setattr(common, "use_rich_output", lambda: False)
    monkeypatch.setattr(common, "setup_logging", lambda *a, **kw: None)
    monkeypatch.setattr(common, "make_log_print", lambda *a, **kw: lambda msg: None)
    monkeypatch.setattr(
        tools,
        "discover_mappable_ids",
        lambda *a, **kw: {
            "available_domains": ["magnetic_field_systems"],
            "ids_targets": [{"ids_name": "magnetics"}],
            "total_sources": 1,
        },
    )

    async def run(state, **kwargs):
        mapping_graph.status = "active" if state.activate else "generated"
        state.ids_results["magnetics"] = {"bindings": 1, "escalations": 0}

    monkeypatch.setattr(workers, "run_mapping_engine", run)
    return mapping_graph


def test_map_run_keeps_generated_status(mapping_engine):
    result = CliRunner().invoke(map_cmd, ["run", "jet", "-i", "magnetics"])

    assert result.exit_code == 0, result.output
    assert mapping_engine.status == "generated"
    assert "--no-activate" not in CliRunner().invoke(map_cmd, ["run", "--help"]).output


def test_activate_refuses_generated_mapping(mapping_graph):
    result = CliRunner().invoke(map_cmd, ["activate", "jet", "magnetics"])

    assert result.exit_code != 0
    assert (
        "Cannot activate mapping jet:magnetics with status 'generated'. "
        "Run map validate first."
    ) in result.output
    assert mapping_graph.status == "generated"


def test_activate_promotes_validated_mapping(mapping_graph):
    mapping_graph.status = "validated"
    result = CliRunner().invoke(map_cmd, ["activate", "jet", "magnetics"])

    assert result.exit_code == 0, result.output
    assert mapping_graph.status == "active"


@pytest.mark.parametrize("passed", [True, False])
def test_validate_records_only_a_pass(monkeypatch, mapping_graph, passed):
    from imas_codex.ids import tools, validation

    monkeypatch.setattr(
        tools,
        "search_existing_mappings",
        lambda *a, **kw: {
            "mapping": {"id": "jet:magnetics", "status": mapping_graph.status},
            "bindings": [{"source_id": "source", "target_id": "magnetics/path"}],
        },
    )
    check = SimpleNamespace(
        source_id="source",
        target_id="magnetics/path",
        source_exists=True,
        target_exists=True,
        transform_executes=True,
        units_compatible=True,
        error=None if passed else "target missing",
    )
    monkeypatch.setattr(
        validation,
        "validate_mapping",
        lambda bindings, gc: SimpleNamespace(
            binding_checks=[check], duplicate_targets=[], all_passed=passed
        ),
    )
    empty_coverage = SimpleNamespace(
        total_leaf_fields=0,
        total_enriched=0,
        total_enriched_matching=0,
        discovered_sources=0,
        multi_target_sources=0,
        total_sections=0,
        total_bindings=0,
    )
    for name in (
        "compute_coverage",
        "compute_signal_coverage",
        "compute_signal_source_coverage",
        "compute_assembly_coverage",
        "compute_confidence_distribution",
    ):
        monkeypatch.setattr(validation, name, lambda *a, **kw: empty_coverage)

    result = CliRunner().invoke(map_cmd, ["validate", "jet", "magnetics"])

    if passed:
        assert result.exit_code == 0, result.output
        assert mapping_graph.status == "validated"
        activated = CliRunner().invoke(map_cmd, ["activate", "jet", "magnetics"])
        assert activated.exit_code == 0, activated.output
        assert mapping_graph.status == "active"
    else:
        assert result.exit_code != 0
        assert "Validation failed" in result.output
        assert mapping_graph.status == "generated"


def test_sequence_mapping_stage_keeps_generated_status(mapping_engine):
    receipt = sequence.run_mapping_stage(
        "jet", sequence.SequenceOptions(ids=("magnetics",)), 2.0, None
    )

    assert receipt["bindings"] == 1
    assert mapping_engine.status == "generated"
