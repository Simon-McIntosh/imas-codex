"""A facility export can omit the data dictionary, leaving a plant package.

The CLI decides whether the filter runs and what it keeps; the generated
compute-node script is the artifact that performs the deletion, so its keep
list is checked here as well.
"""

from __future__ import annotations

import pytest
from click.testing import CliRunner

from iter_codex.cli.graph.data import graph_export
from iter_codex.graph.temp_neo4j import _build_filter_script, create_facility_dump


def test_without_dd_needs_a_facility():
    result = CliRunner().invoke(graph_export, ["--without-dd"])
    assert result.exit_code != 0
    assert "--facility" in result.output


def test_facility_dump_forwards_without_dd(monkeypatch):
    calls: dict = {}

    def _fake_slurm(source, output, *, mode, facility="", without_dd=False):
        calls.update(
            source=source,
            output=output,
            mode=mode,
            facility=facility,
            without_dd=without_dd,
        )

    monkeypatch.setattr("iter_codex.graph.temp_neo4j._should_use_slurm", lambda: True)
    monkeypatch.setattr(
        "iter_codex.graph.temp_neo4j._run_filter_via_slurm", _fake_slurm
    )
    create_facility_dump("in.dump", "demo", "out.dump", without_dd=True)
    assert calls["mode"] == "facility"
    assert calls["facility"] == "demo"
    assert calls["without_dd"] is True


def _script(without_dd: bool) -> str:
    return _build_filter_script(
        neo4j_image="/images/neo4j.sif",
        source_dump="/dumps/graph.dump",
        output_dump="/dumps/plant.dump",
        mode="facility",
        facility="demo",
        without_dd=without_dd,
    )


def test_plant_filter_deletes_every_facilityless_kind_but_the_kept_two():
    script = _script(without_dd=True)
    assert "x.facility_id IS NULL AND NOT x:Facility AND NOT x:GraphMeta" in script
    assert "MATCH (n:Facility) WHERE n.id <> 'demo'" in script
    assert "m.plant_only = true" in script


def test_facility_filter_alone_keeps_facilityless_kinds():
    script = _script(without_dd=False)
    assert "facility_id IS NULL AND NOT" not in script
    assert "m.plant_only = false" in script
    assert "m.plant_only = true" not in script


def test_plant_filter_keeps_the_facilitys_own_nodes():
    script = _script(without_dd=True)
    assert "n.facility_id <> 'demo'" in script


def test_public_registry_answer_makes_the_push_refuse(monkeypatch):
    """A registry that reports the package public stops the push."""
    import click

    from iter_codex.cli.graph import registry as registry_mod
    from iter_codex.graph import ghcr

    monkeypatch.setattr(ghcr, "get_package_visibility", lambda *a, **k: "public")
    # The registry refuses the write and keeps reporting public.
    monkeypatch.setattr(ghcr, "set_package_visibility", lambda *a, **k: "public")

    with pytest.raises(click.ClickException) as excinfo:
        registry_mod._report_plant_package_visibility(
            "ghcr.io/owner",
            "iter-codex-graph-iter-without-dd",
            None,
            facilities=("iter",),
            without_dd=True,
        )
    assert "must not be published publicly" in str(excinfo.value)


def test_private_registry_answer_is_reported_and_the_push_continues(
    monkeypatch, capsys
):
    from iter_codex.cli.graph import registry as registry_mod
    from iter_codex.graph import ghcr

    monkeypatch.setattr(ghcr, "get_package_visibility", lambda *a, **k: "private")
    registry_mod._report_plant_package_visibility(
        "ghcr.io/owner",
        "iter-codex-graph-iter-without-dd",
        None,
        facilities=("iter",),
        without_dd=True,
    )
    assert "private (registry-reported)" in capsys.readouterr().out


def test_a_facility_export_without_the_plant_flag_skips_the_visibility_report(
    monkeypatch, capsys
):
    from iter_codex.cli.graph import registry as registry_mod
    from iter_codex.graph import ghcr

    monkeypatch.setattr(
        ghcr,
        "get_package_visibility",
        lambda *a, **k: pytest.fail("visibility read for a full-graph package"),
    )
    registry_mod._report_plant_package_visibility(
        "ghcr.io/owner",
        "iter-codex-graph-dd",
        None,
        facilities=(),
        without_dd=False,
    )
    assert capsys.readouterr().out == ""


def test_plant_filter_measures_its_export_and_its_package():
    script = _script(without_dd=True)
    assert "measure_scope" in script
    assert "SCOPE_BEFORE=" in script
    assert "SCOPE_AFTER=" in script


def test_a_package_that_lost_a_kind_is_refused():
    import click

    from iter_codex.graph.plant_scope import PlantScope
    from iter_codex.graph.temp_neo4j import _verify_plant_package

    expected = PlantScope(
        node_labels={"FacilitySignal": 36},
        relationship_types={"DATA_ACCESS": 36},
        facilityless_labels={"Facility": 1},
    )
    actual = PlantScope(
        node_labels={"FacilitySignal": 35},
        relationship_types={"DATA_ACCESS": 36},
        facilityless_labels={"Facility": 1},
    )
    with pytest.raises(click.ClickException) as excinfo:
        _verify_plant_package(expected, actual, "iter")
    assert "does not match the graph it was filtered from" in str(excinfo.value)


def test_export_refuses_to_dump_a_mismatched_package(monkeypatch, tmp_path):
    """The export path runs the comparison, so no mismatched package is dumped."""
    import click

    import iter_codex.graph.temp_neo4j as temp_neo4j
    from iter_codex.graph.plant_scope import PlantScope

    source = tmp_path / "in.dump"
    source.write_bytes(b"")
    monkeypatch.setattr(temp_neo4j, "_should_use_slurm", lambda: False)
    monkeypatch.setattr(temp_neo4j, "_cleanup_stale_temp_neo4j", lambda *a: None)
    monkeypatch.setattr(
        temp_neo4j, "start_temp_neo4j", lambda *a, **k: (object(), tmp_path / "log")
    )
    monkeypatch.setattr(temp_neo4j, "stop_temp_neo4j", lambda *a: None)
    monkeypatch.setattr(
        temp_neo4j,
        "dump_temp_neo4j",
        lambda *a: pytest.fail("a mismatched package was dumped"),
    )
    monkeypatch.setattr(
        temp_neo4j,
        "_filter_to_facility",
        lambda *a, **k: (
            PlantScope(node_labels={"FacilitySignal": 36}),
            PlantScope(node_labels={"FacilitySignal": 35}),
        ),
    )

    with pytest.raises(click.ClickException):
        temp_neo4j.create_facility_dump(
            source, "iter", tmp_path / "out.dump", without_dd=True
        )


def test_the_compute_node_script_verifies_before_it_dumps():
    """The script's own refusal runs while no dump exists yet."""
    script = _script(without_dd=True)
    assert script.index("verify_scope\n") < script.index("Dumping filtered graph")
    assert script.index("verify_scope\n") < script.index("neo4j-admin database dump")


def test_the_verify_command_the_script_runs_refuses_a_mismatch():
    """The command the script executes, run on two measured scopes."""
    import json
    import subprocess
    import sys

    from iter_codex.graph.plant_scope import PlantScope
    from iter_codex.graph.temp_neo4j import VERIFY_SCOPE_PYTHON

    matched = PlantScope(
        node_labels={"FacilitySignal": 3},
        relationship_types={"DATA_ACCESS": 4},
        facilityless_labels={"Facility": 1, "GraphMeta": 1},
    ).to_dict()
    short = PlantScope(
        node_labels={"FacilitySignal": 2},
        relationship_types={"DATA_ACCESS": 4},
        facilityless_labels={"Facility": 1, "GraphMeta": 1},
    ).to_dict()

    def run(before: dict, after: dict):
        return subprocess.run(
            [
                sys.executable,
                "-c",
                VERIFY_SCOPE_PYTHON,
                json.dumps(before),
                json.dumps(after),
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )

    assert run(matched, matched).returncode == 0
    refused = run(matched, short)
    assert refused.returncode != 0
    assert "does not match" in refused.stderr
