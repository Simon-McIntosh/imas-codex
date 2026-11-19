"""A facility export can omit the data dictionary, leaving a plant package.

The CLI decides whether the filter runs and what it keeps; the generated
compute-node script is the artifact that performs the deletion, so its keep
list is checked here as well.
"""

from __future__ import annotations

import pytest
from click.testing import CliRunner

from imas_codex.cli.graph.data import graph_export
from imas_codex.graph.temp_neo4j import _build_filter_script, create_facility_dump


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

    monkeypatch.setattr("imas_codex.graph.temp_neo4j._should_use_slurm", lambda: True)
    monkeypatch.setattr(
        "imas_codex.graph.temp_neo4j._run_filter_via_slurm", _fake_slurm
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
    assert "n.facility_id IS NULL AND NOT n:Facility AND NOT n:GraphMeta" in script
    assert "MATCH (n:Facility) WHERE n.id <> 'demo'" in script
    assert "m.plant_only = true" in script


def test_facility_filter_alone_keeps_facilityless_kinds():
    script = _script(without_dd=False)
    assert "n.facility_id IS NULL AND NOT n:Facility" not in script
    assert "plant_only" not in script


def test_plant_filter_keeps_the_facilitys_own_nodes():
    script = _script(without_dd=True)
    assert "n.facility_id <> 'demo'" in script
