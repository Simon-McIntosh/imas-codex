"""Tests for the DD source-drift refresh (idempotent re-refine on source change)."""

from __future__ import annotations

import os
import re
import subprocess
from collections.abc import Iterator
from unittest.mock import MagicMock

import pytest

from imas_codex.graph.client import GraphClient
from imas_codex.graph.schema import GraphSchema
from imas_codex.standard_names import source_refresh as sr


class _FakeGC:
    """Minimal GraphClient stand-in: returns canned rows from ``query``."""

    def __init__(self, rows):
        self._rows = rows
        self.writes: list[tuple[str, dict]] = []

    def query(self, cypher, **kw):
        # Record writes; return canned rows for reads.
        if "SET" in cypher and "RETURN" not in cypher:
            self.writes.append((cypher, kw))
            return []
        return self._rows

    def close(self):
        pass


class _CaptureGC:
    """Capture a snapshot query and report one stamped name."""

    def __init__(self):
        self.calls: list[tuple[str, dict]] = []

    def query(self, cypher, **kw):
        self.calls.append((cypher, kw))
        if "RETURN sn.id AS sn_id" in cypher:
            return [
                {
                    "sn_id": "gas_flow",
                    "path": "gas_injection/pipe/flow_rate",
                    "unit": "s^-1",
                    "documentation": "Particle flow rate.",
                }
            ]
        return []


def test_norm_collapses_none_and_strips():
    assert sr._norm(None) == ""
    assert sr._norm("  x ") == "x"
    assert sr._norm("W.m^-2") == "W.m^-2"
    assert sr._norm(3) == "3"


def test_format_reason_reports_precise_delta():
    reason = sr._format_reason(
        "neutral_energy_flux_at_wall",
        [{"field": "units", "old": "m^-2.s^-1", "new": "W.m^-2"}],
    )
    assert "units" in reason
    assert "m^-2.s^-1" in reason and "W.m^-2" in reason
    assert "source-refresh" in reason  # framed as targeted refresh, not rewrite


def test_format_reason_truncates_long_documentation():
    long_old = "A" * 400
    reason = sr._format_reason(
        "x", [{"field": "documentation", "old": long_old, "new": "B"}]
    )
    assert "…" in reason  # long doc is truncated for the steering reason


def _row(
    old_unit,
    new_unit,
    old_doc="d",
    new_doc="d",
    old_path="wall/x",
    new_path="wall/x",
    anchor_id=None,
):
    current = sr._resolved_source_context(new_path, new_unit, new_doc)
    return {
        "sn_id": "some_name",
        "source_id": f"dd:{anchor_id or new_path}",
        "anchor_id": anchor_id if anchor_id is not None else new_path,
        "name_stage": "accepted",
        "docs_stage": "accepted",
        "old_unit": old_unit,
        "new_unit": new_unit,
        "old_doc": old_doc,
        "new_doc": new_doc,
        "old_path": old_path,
        "new_path": new_path,
        "old_raw_unit": current["raw_dd_context"]["unit"],
        "old_raw_doc": current["raw_dd_context"]["documentation"],
        "old_resolution_ids": current["dd_resolution_ids"],
        "old_converged_ids": current["dd_resolution_converged_ids"],
        "old_manifest_digest": current["dd_resolution_manifest_digest"],
        "old_resolution_marker": current["_dd_resolution_marker"],
    }


def test_detect_drift_units_change():
    out = sr.detect_source_drift(gc=_FakeGC([_row("m^-2.s^-1", "W.m^-2")]))
    assert len(out) == 1
    assert out[0]["deltas"] == [{"field": "units", "old": "m^-2.s^-1", "new": "W.m^-2"}]
    assert out[0]["renamed"] is False


def test_detect_drift_documentation_change():
    out = sr.detect_source_drift(
        gc=_FakeGC([_row("W.m^-2", "W.m^-2", "old doc", "new doc")])
    )
    assert len(out) == 1
    assert [d["field"] for d in out[0]["deltas"]] == ["documentation"]


def test_detect_drift_path_rename():
    out = sr.detect_source_drift(
        gc=_FakeGC(
            [
                _row(
                    "W.m^-2",
                    "W.m^-2",
                    old_path="x/torque_fast_tor",
                    new_path="x/torque_fast_phi",
                    anchor_id="x/torque_fast_tor",
                )
            ]
        )
    )
    assert len(out) == 1
    assert out[0]["renamed"] is True
    assert out[0]["deltas"][0]["field"] == "source_path"
    assert out[0]["new_path"] == "x/torque_fast_phi"


def test_format_reason_rename_labelled():
    reason = sr._format_reason(
        "x",
        [{"field": "source_path", "old": "a/tor", "new": "a/phi"}],
        renamed=True,
    )
    assert "path renamed" in reason


def test_format_reason_path_change_is_not_a_rename():
    # A source re-pointed to a different path with no RENAMED_TO lineage is a
    # path change; calling it a rename misdescribes the graph.
    reason = sr._format_reason(
        "x", [{"field": "source_path", "old": "wall/a", "new": "wall/b"}]
    )
    assert "path changed" in reason
    assert "renamed" not in reason


def test_detect_drift_upgrades_source_report_for_every_producing_source():
    # One drifted source is enough; the reported delta must come from the source
    # that drifted rather than from an arbitrary one of the name's sources.
    clean = _row(
        "W.m^-2",
        "W.m^-2",
        "same doc",
        "same doc",
        old_path="wall/clean",
        new_path="wall/clean",
    )
    clean["source_id"] = "dd:wall/clean"
    drifted = _row(
        "W.m^-2",
        "W.m^-2",
        "old doc",
        "new doc",
        old_path="wall/drifted",
        new_path="wall/drifted",
    )
    drifted["source_id"] = "dd:wall/drifted"

    out = sr.detect_source_drift(gc=_FakeGC([clean, drifted]))

    assert len(out) == 1
    assert out[0]["sn_id"] == "some_name"
    assert out[0]["source_id"] == "dd:wall/drifted"
    assert out[0]["new_path"] == "wall/drifted"
    assert [d["field"] for d in out[0]["deltas"]] == ["documentation"]


def test_detect_drift_reports_unresolvable_source_instead_of_dropping_it():
    unresolved = _row("W.m^-2", "W.m^-2", "old doc", "new doc")
    # source_id is not a DD path in the graph, so the OPTIONAL MATCH binds no
    # node: new_path/anchor_id come back null and there is nothing to compare.
    unresolved["new_path"] = None
    unresolved["anchor_id"] = None
    unchecked: list[dict[str, str]] = []

    out = sr.detect_source_drift(gc=_FakeGC([unresolved]), unchecked=unchecked)

    assert out == []
    assert unchecked == [{"sn_id": "some_name", "source_id": "dd:wall/x"}]


def test_detect_drift_ignores_whitespace_only_change():
    # normalised-equal values must not be reported as drift (idempotency guard)
    out = sr.detect_source_drift(gc=_FakeGC([_row("W.m^-2 ", "W.m^-2", "d ", " d")]))
    assert out == []


class _ScopeGraph:
    """Graph double that honours the ``sn.id IN $sn_ids`` scope filter.

    Two names have drifted; *in_scope* holds the one the run owns. The detect
    query returns only what the caller's id filter admits, exactly as Cypher
    does, so the refresh's scoping is exercised rather than assumed.
    """

    def __init__(self, in_scope: list[str] | None):
        self.in_scope = in_scope
        self.writes: list[tuple[str, dict]] = []
        self.detected_ids: list[str] = []

    def query(self, cypher, **kw):
        if "sn.source_path IS NULL" in cypher:
            return []  # nothing is unstamped: the baseline pass writes nothing
        if "n.id AS new_path" in cypher:
            admitted = kw.get("sn_ids")
            rows = [
                _row("W.m^-2", "W.m^-2", "old doc", "new doc", new_path="wall/x"),
                _row("W.m^-2", "W.m^-2", "old doc", "new doc", new_path="wall/other"),
            ]
            rows[0]["sn_id"] = "in_scope_name"
            rows[1]["sn_id"] = "out_of_scope_name"
            rows[0]["source_id"] = "dd:wall/x"
            rows[1]["source_id"] = "dd:wall/other"
            if admitted is not None:
                rows = [r for r in rows if r["sn_id"] in admitted]
            self.detected_ids = [r["sn_id"] for r in rows]
            return rows
        self.writes.append((cypher, kw))
        return []

    def close(self):
        pass


def test_scoped_refresh_never_steers_a_name_outside_its_scope(monkeypatch):
    graph = _ScopeGraph(in_scope=["in_scope_name"])
    steered: list[str] = []
    monkeypatch.setattr(
        "imas_codex.standard_names.edit.apply_edit",
        lambda **kw: steered.append(kw["target"]),
    )

    summary = sr.refresh_drifted_sources(
        sn_ids=["in_scope_name"], scope_run_id="scope-1", gc=graph
    )

    assert graph.detected_ids == ["in_scope_name"]
    assert steered == ["in_scope_name"]
    assert summary["detected"] == 1
    assert summary["steered"] == 1
    assert summary["names"] == ["in_scope_name"]
    # The outsider is neither stamped nor attached to this run.
    assert all("out_of_scope_name" not in str(kw) for _, kw in graph.writes), (
        graph.writes
    )


def test_unscoped_refresh_still_steers_every_drifted_name(monkeypatch):
    graph = _ScopeGraph(in_scope=None)
    steered: list[str] = []
    monkeypatch.setattr(
        "imas_codex.standard_names.edit.apply_edit",
        lambda **kw: steered.append(kw["target"]),
    )

    summary = sr.refresh_drifted_sources(gc=graph)

    assert sorted(graph.detected_ids) == ["in_scope_name", "out_of_scope_name"]
    assert sorted(steered) == ["in_scope_name", "out_of_scope_name"]
    assert summary["detected"] == 2
    for _, kw in graph.writes:
        updates = kw.get("updates") or []
        assert "out_of_scope_name" not in str(updates)


def test_refresh_no_drift_is_noop():
    # No rows -> nothing detected, nothing steered (safe on every run).
    summary = sr.refresh_drifted_sources(gc=_FakeGC([]), dry_run=True)
    assert summary["detected"] == 0
    assert summary["steered"] == 0


def test_stamp_source_snapshot_targets_only_gas_flow_cache():
    gc = _CaptureGC()

    stamped = sr.stamp_source_snapshots(["gas_flow"], gc=gc)

    assert stamped == 1
    assert len(gc.calls) == 2
    cypher, params = gc.calls[0]
    assert "sn.id IN $sn_ids" in cypher
    assert params["sn_ids"] == ["gas_flow"]
    write, write_params = gc.calls[1]
    assert "sn.source_unit = update.unit" in write
    [snapshot] = write_params["updates"]
    assert snapshot["sn_id"] == "gas_flow"
    assert snapshot["path"] == "gas_injection/pipe/flow_rate"
    assert snapshot["unit"] == "s^-1"
    assert snapshot["documentation"] == "Particle flow rate."
    assert snapshot["raw_unit"] == "s^-1"
    assert snapshot["raw_documentation"] == "Particle flow rate."
    assert snapshot["resolution_ids"] == []
    assert snapshot["converged_ids"] == []
    assert snapshot["manifest_digest"].startswith("sha256:")
    assert snapshot["resolution_marker"] == "resolved-dd-context"
    assert "sn.source_raw_unit = update.raw_unit" in write
    assert "sn.source_dd_resolution_marker = update.resolution_marker" in write


def test_stamped_source_provenance_is_declared_on_standard_name():
    gc = _CaptureGC()

    sr.stamp_source_snapshots(["gas_flow"], gc=gc)

    write = gc.calls[1][0]
    stamped_properties = set(re.findall(r"sn\.(source_[a-z_]+) =", write))
    declared = set(
        GraphSchema("imas_codex/schemas/standard_name.yaml")
        .get_all_slots("StandardName")
        .keys()
    )
    assert stamped_properties <= declared


def test_public_refresh_restamps_raw_convergence_without_steering(monkeypatch):
    resolution_id = "ddres:camera-direction-x"
    current_context = {
        "path": "camera_ir/channel/camera/direction/x",
        "unit": "1",
        "documentation": "Direction component.",
        "raw_dd_context": {
            "unit": "1",
            "documentation": "Direction component.",
        },
        "published_dd_context": {
            "unit": "m",
            "documentation": "Direction component.",
        },
        "dd_resolution_ids": [],
        "dd_resolution_converged_ids": [resolution_id],
        "dd_resolution_manifest_digest": "sha256:current",
        "_dd_resolution_marker": "resolved-dd-context",
    }
    drift_row = {
        "sn_id": "x_direction_unit_vector",
        "source_id": "dd:camera_ir/channel/camera/direction/x",
        "anchor_id": "camera_ir/channel/camera/direction/x",
        "name_stage": "accepted",
        "docs_stage": "accepted",
        "old_unit": "1",
        "new_unit": "1",
        "old_doc": "Direction component.",
        "new_doc": "Direction component.",
        "old_path": "camera_ir/channel/camera/direction/x",
        "new_path": "camera_ir/channel/camera/direction/x",
        "old_raw_unit": "m",
        "old_raw_doc": "Direction component.",
        "old_resolution_ids": [resolution_id],
        "old_converged_ids": [],
        "old_manifest_digest": "sha256:current",
        "old_resolution_marker": "resolved-dd-context",
    }

    class RefreshGraph:
        def __init__(self):
            self.writes: list[tuple[str, dict]] = []

        def query(self, query, **kwargs):
            if "SET sn.source_unit" in query:
                self.writes.append((query, kwargs))
                return []
            if "source_path IS NULL" in query:
                return []
            if "ORDER BY sn.id" in query:
                return [drift_row]
            if "sn.id IN $sn_ids" in query:
                return [
                    {
                        "sn_id": "x_direction_unit_vector",
                        "path": "camera_ir/channel/camera/direction/x",
                        "unit": "1",
                        "documentation": "Direction component.",
                    }
                ]
            return []

        def close(self):
            pass

    gc = RefreshGraph()
    apply_edit = MagicMock(
        side_effect=AssertionError("provenance drift must not steer")
    )
    monkeypatch.setattr(sr, "_resolved_source_context", lambda *_args: current_context)
    monkeypatch.setattr("imas_codex.standard_names.edit.apply_edit", apply_edit)

    summary = sr.refresh_drifted_sources(gc=gc)

    assert summary["detected"] == 1
    assert summary["restamped"] == 1
    assert summary["steered"] == 0
    assert summary["names"] == ["x_direction_unit_vector"]
    apply_edit.assert_not_called()
    assert len(gc.writes) == 1
    [update] = gc.writes[0][1]["updates"]
    assert update["raw_unit"] == "m"
    assert update["resolution_ids"] == []
    assert update["converged_ids"] == [resolution_id]


# ── Disposable-Neo4j contract ────────────────────────────────────────────────
# The scan is Cypher, so it is proved on a real graph: an empty Neo4j the test
# seeds itself. The endpoint, the environment naming it and the refusal of the
# project graph's URI are owned by the package's shared disposable fixture.


def _project_bolt_port() -> int:
    """The port this checkout's profile gives the project graph."""
    from imas_codex.graph.profiles import resolve_neo4j

    return resolve_neo4j(auto_tunnel=False).bolt_port


def _assert_refused(monkeypatch, address: str) -> None:
    """Require the refusal for *address*, from the check and the fixture body."""
    from tests.standard_names.conftest import (
        disposable_neo4j_uri,
        refuse_project_graph,
    )

    with pytest.raises(
        pytest.fail.Exception, match="refusing to run against"
    ) as refused:
        refuse_project_graph(address)
    print(f"refused {address}: {refused.value}")

    body = getattr(disposable_neo4j_uri, "__wrapped__", disposable_neo4j_uri)
    monkeypatch.setenv("IMAS_CODEX_TEST_NEO4J_URI", address)
    monkeypatch.setenv("IMAS_CODEX_TEST_NEO4J_EPHEMERAL", "1")
    with pytest.raises(pytest.fail.Exception, match="refusing to run against"):
        next(iter(body()))


def test_the_shared_fixture_refuses_the_project_loopback_address(monkeypatch):
    """The protection the disposable fixtures exist for, shown refusing.

    The fixture's whole value is that it cannot be pointed at production, and a
    guard nothing exercises is a guard nobody knows works. This requires the
    refusal for the loopback address the profile names, from the check and from
    the fixture body, and accepts a disposable endpoint in the same breath so
    the test cannot pass by refusing everything.
    """
    from tests.standard_names.conftest import refuse_project_graph

    _assert_refused(monkeypatch, f"bolt://localhost:{_project_bolt_port()}")
    refuse_project_graph("bolt://127.0.0.1:1")


def _scheduled_graph_node() -> str | None:
    """The node the scheduler reports for the project graph."""
    try:
        completed = subprocess.run(
            ["squeue", "-h", "-n", "codex-neo4j", "-o", "%N"],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    nodes = completed.stdout.split()
    return nodes[0].split(",")[0] if nodes else None


@pytest.mark.graph
def test_the_shared_fixture_refuses_the_scheduled_graph_address(monkeypatch):
    """The other spelling, read from the scheduler rather than from the profile.

    On a cluster whose graph runs as a service, the loopback port is not the
    address a compute node connects to, so a refusal that knows only the
    loopback protects nothing there. The address comes from ``squeue`` so that
    a resolver narrowing to the loopback spelling cannot keep this green -- a
    guard and its only test sharing one resolver move together. It needs a
    running service, hence the graph marker.
    """
    node = _scheduled_graph_node()
    assert node is not None, (
        "the scheduler reports no task named codex-neo4j, so the "
        "scheduled-instance arm of the refusal cannot be exercised"
    )
    _assert_refused(monkeypatch, f"bolt://{node}:{_project_bolt_port()}")


def test_a_failed_start_leaves_no_process_behind(monkeypatch, tmp_path):
    """A start that never reaches readiness stops the process it began.

    A container abandoned before readiness keeps its JVM and its memory on the
    node the run shares with every other session, so a failed start has to take
    the process group with it. The launch command is replaced by a long sleep
    so the arm costs nothing, but the process and its group are the real ones:
    the owner starts it in its own session and is the only thing that stops it.
    """
    from imas_codex.graph import temp_neo4j

    started = []
    real_popen = subprocess.Popen

    def _popen(command, **kwargs):
        process = real_popen(["sleep", "300"], **kwargs)
        started.append(process)
        return process

    def _never_ready(process, *args, **kwargs):
        assert process.poll() is None, "the arm must fail while the process is alive"
        raise temp_neo4j.click.ClickException("Bolt readiness never arrived")

    monkeypatch.setattr(temp_neo4j.subprocess, "Popen", _popen)
    monkeypatch.setattr(temp_neo4j, "_wait_for_bolt_ready", _never_ready)

    with pytest.raises(temp_neo4j.click.ClickException, match="Bolt readiness"):
        temp_neo4j.start_temp_neo4j(tmp_path / "instance", 1, 2, load_dump=False)

    assert len(started) == 1
    assert started[0].poll() is not None, "the failed start left its process running"


@pytest.fixture
def graph(disposable_neo4j_uri: str, monkeypatch) -> Iterator[GraphClient]:
    # An empty instance carries no DD-resolution authority, and the drift scan
    # reads one to stamp each source snapshot. The scan's Cypher is what these
    # tests prove, so the authority comes from the package's synthetic rows —
    # the same boundary the mock tier uses — while every query still runs on
    # the disposable graph.
    from imas_codex.standard_names import dd_resolutions
    from tests.standard_names.conftest import (
        SyntheticDDResolutionReader,
        neo4j_test_password,
    )
    from tests.standard_names.dd_resolution_test_data import (
        SYNTHETIC_GRAPH_RESOLUTION_ROWS,
    )

    monkeypatch.setattr(
        dd_resolutions,
        "dd_resolution_graph_reader",
        lambda: SyntheticDDResolutionReader(SYNTHETIC_GRAPH_RESOLUTION_ROWS),
    )
    client = GraphClient(
        uri=disposable_neo4j_uri,
        username="neo4j",
        password=neo4j_test_password(),
        graph_name="source-drift-refresh",
    )
    client.query("MATCH (node) DETACH DELETE node")
    yield client
    client.query("MATCH (node) DETACH DELETE node")
    client.close()


def _seed_name(
    graph: GraphClient,
    name_id: str,
    *,
    live_path: str,
    live_doc: str,
    snapshot_doc: str,
    snapshot_path: str | None = None,
    unit: str = "1",
    run_id: str | None = None,
) -> None:
    """Create a name carrying the snapshot of the live source it was built from.

    The snapshot's raw and resolution fields are read back from the resolver, so
    the only difference a test has to state is the documentation or path it wants
    to look drifted.
    """
    ctx = sr._resolved_source_context(live_path, unit, live_doc)
    published = ctx["published_dd_context"]
    graph.query(
        """
        MERGE (sn:StandardName {id: $name_id})
        SET sn.name_stage = 'accepted', sn.docs_stage = 'accepted',
            sn.run_id = $run_id,
            sn.source_path = $snapshot_path, sn.source_unit = $unit,
            sn.source_documentation = $snapshot_doc,
            sn.source_raw_unit = $raw_unit,
            sn.source_raw_documentation = $raw_doc,
            sn.source_dd_resolution_ids = $ids,
            sn.source_dd_resolution_converged_ids = $converged,
            sn.source_dd_resolution_manifest_digest = $digest,
            sn.source_dd_resolution_marker = $marker
        """,
        name_id=name_id,
        run_id=run_id,
        snapshot_path=snapshot_path or live_path,
        unit=unit,
        snapshot_doc=snapshot_doc,
        raw_unit=published["unit"],
        raw_doc=published["documentation"],
        ids=ctx["dd_resolution_ids"],
        converged=ctx["dd_resolution_converged_ids"],
        digest=ctx["dd_resolution_manifest_digest"],
        marker=ctx["_dd_resolution_marker"],
    )


def _attach_source(
    graph: GraphClient,
    name_id: str,
    *,
    source_id: str,
    live_path: str | None,
    live_doc: str = "",
    anchor_path: str | None = None,
    unit: str = "1",
) -> None:
    """Attach one producing source to a name.

    *anchor_path* makes the source's ``source_id`` an earlier identity that
    ``RENAMED_TO`` links to *live_path*, which is what a rename looks like.
    *live_path* of ``None`` leaves the source id unresolved in the graph.
    """
    if live_path is not None:
        graph.query(
            "CREATE (:IMASNode {id: $id, unit: $unit, documentation: $doc})",
            id=live_path,
            unit=unit,
            doc=live_doc,
        )
    if anchor_path is not None:
        graph.query("CREATE (:IMASNode {id: $id})", id=anchor_path)
        graph.query(
            """
            MATCH (o:IMASNode {id: $old}), (n:IMASNode {id: $new})
            CREATE (o)-[:RENAMED_TO]->(n)
            """,
            old=anchor_path,
            new=live_path,
        )
    graph.query(
        """
        MATCH (sn:StandardName {id: $name_id})
        CREATE (src:StandardNameSource {
          id: $source_id, source_type: 'dd', source_id: $source_id_body
        })
        CREATE (src)-[:PRODUCED_NAME]->(sn)
        """.replace("{id: $name_id)", "{id: $name_id}"),
        name_id=name_id,
        source_id=source_id,
        source_id_body=source_id[3:] if source_id.startswith("dd:") else source_id,
    )


@pytest.mark.graph
@pytest.mark.timeout(180)
def test_scoped_drift_scan_touches_only_the_runs_names(graph: GraphClient) -> None:
    _seed_name(
        graph,
        "in_scope_name",
        live_path="wall/a",
        live_doc="live a",
        snapshot_doc="stale a",
        run_id="scope-1",
    )
    _attach_source(
        graph,
        "in_scope_name",
        source_id="dd:wall/a",
        live_path="wall/a",
        live_doc="live a",
    )
    _seed_name(
        graph,
        "out_of_scope_name",
        live_path="wall/b",
        live_doc="live b",
        snapshot_doc="stale b",
    )
    _attach_source(
        graph,
        "out_of_scope_name",
        source_id="dd:wall/b",
        live_path="wall/b",
        live_doc="live b",
    )

    scoped = sr.detect_source_drift(sn_ids=["in_scope_name"], gc=graph)

    assert [d["sn_id"] for d in scoped] == ["in_scope_name"]
    unscoped = sr.detect_source_drift(gc=graph)
    assert sorted(d["sn_id"] for d in unscoped) == [
        "in_scope_name",
        "out_of_scope_name",
    ]


@pytest.mark.graph
@pytest.mark.timeout(180)
def test_scoped_refresh_leaves_an_out_of_scope_name_unsteered(
    graph: GraphClient, monkeypatch
) -> None:
    _seed_name(
        graph,
        "in_scope_name",
        live_path="wall/a",
        live_doc="live a",
        snapshot_doc="stale a",
        run_id="scope-1",
    )
    _attach_source(
        graph,
        "in_scope_name",
        source_id="dd:wall/a",
        live_path="wall/a",
        live_doc="live a",
    )
    _seed_name(
        graph,
        "out_of_scope_name",
        live_path="wall/b",
        live_doc="live b",
        snapshot_doc="stale b",
    )
    _attach_source(
        graph,
        "out_of_scope_name",
        source_id="dd:wall/b",
        live_path="wall/b",
        live_doc="live b",
    )
    steered: list[str] = []
    monkeypatch.setattr(
        "imas_codex.standard_names.edit.apply_edit",
        lambda **kw: steered.append(kw["target"]),
    )

    scoped = sr.refresh_drifted_sources(
        sn_ids=["in_scope_name"], scope_run_id="scope-1", gc=graph
    )

    assert steered == ["in_scope_name"]
    assert scoped["detected"] == 1
    rows = graph.query(
        """
        MATCH (sn:StandardName)
        RETURN sn.id AS id, sn.source_documentation AS doc, sn.run_id AS run_id
        ORDER BY sn.id
        """
    )
    by_id = {r["id"]: r for r in rows}
    # The outsider is neither re-stamped nor attached to the run.
    assert by_id["out_of_scope_name"]["doc"] == "stale b"
    assert by_id["out_of_scope_name"]["run_id"] is None
    # The in-scope name was steered and re-stamped so the change is idempotent.
    assert by_id["in_scope_name"]["doc"] == "live a"

    steered.clear()
    unscoped = sr.refresh_drifted_sources(gc=graph)

    assert steered == ["out_of_scope_name"]
    assert unscoped["detected"] == 1


@pytest.mark.graph
@pytest.mark.timeout(180)
def test_rename_is_reported_only_when_lineage_moved_the_anchor(
    graph: GraphClient,
) -> None:
    # A source re-anchored through RENAMED_TO is a rename.
    _seed_name(
        graph,
        "renamed_name",
        live_path="new/path",
        live_doc="doc",
        snapshot_doc="doc",
        snapshot_path="old/path",
    )
    _attach_source(
        graph,
        "renamed_name",
        source_id="dd:old/path",
        live_path="new/path",
        live_doc="doc",
        anchor_path="old/path",
    )
    # A source re-pointed to an unrelated path is a path change, not a rename.
    _seed_name(
        graph,
        "repointed_name",
        live_path="other/path",
        live_doc="doc",
        snapshot_doc="doc",
        snapshot_path="first/path",
    )
    _attach_source(
        graph,
        "repointed_name",
        source_id="dd:other/path",
        live_path="other/path",
        live_doc="doc",
    )

    out = {d["sn_id"]: d for d in sr.detect_source_drift(gc=graph)}

    assert out["renamed_name"]["renamed"] is True
    assert out["renamed_name"]["new_path"] == "new/path"
    assert "path renamed" in sr._format_reason(
        "renamed_name", out["renamed_name"]["deltas"], renamed=True
    )
    assert out["repointed_name"]["renamed"] is False
    assert "path changed" in sr._format_reason(
        "repointed_name", out["repointed_name"]["deltas"]
    )


@pytest.mark.graph
@pytest.mark.timeout(180)
def test_source_without_a_dd_node_is_reported_unchecked(graph: GraphClient) -> None:
    _seed_name(
        graph,
        "orphan_source_name",
        live_path="wall/a",
        live_doc="doc",
        snapshot_doc="stale",
    )
    _attach_source(
        graph, "orphan_source_name", source_id="dd:absent/path", live_path=None
    )

    unchecked: list[dict[str, str]] = []
    out = sr.detect_source_drift(gc=graph, unchecked=unchecked)

    assert out == []
    assert unchecked == [{"sn_id": "orphan_source_name", "source_id": "dd:absent/path"}]


@pytest.mark.graph
@pytest.mark.timeout(180)
def test_multi_source_name_is_compared_against_every_source(
    graph: GraphClient,
) -> None:
    _seed_name(
        graph,
        "multi_source_name",
        live_path="multi/a",
        live_doc="doc a",
        snapshot_doc="doc a",
    )
    _attach_source(graph, "multi_source_name", source_id="dd:multi/a", live_path=None)
    # Only the second source drifted; the first still matches the snapshot.
    _attach_source(
        graph,
        "multi_source_name",
        source_id="dd:multi/b",
        live_path="multi/b",
        live_doc="doc b",
    )

    out = sr.detect_source_drift(gc=graph)

    assert len(out) == 1
    assert out[0]["sn_id"] == "multi_source_name"
    assert out[0]["source_id"] == "dd:multi/b"
    assert "documentation" in [d["field"] for d in out[0]["deltas"]]
