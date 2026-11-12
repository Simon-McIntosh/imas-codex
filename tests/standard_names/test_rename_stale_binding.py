"""A rename migrates the live sources of a name and leaves its stale ones behind.

A source row goes stale when the DD path it points at is removed from the
current data dictionary. Such a binding can never satisfy a migration's
compare-and-set, so carrying it into the rename cohort blocks a rename that has
nothing wrong with it. The rename instead migrates every live binding and
records the stale ones on the successor, where a reader of the live name can
see that history stayed with the retired spelling. A name with no live binding
at all is still refused: there is nothing to carry the rename.

These tests drive the sanctioned rename writer on the disposable graph and read
both the successor's sources and the retired predecessor's back.
"""

from __future__ import annotations

import pytest

from imas_codex.graph.client import GraphClient
from imas_codex.standard_names.edit import apply_edit
from imas_codex.standard_names.source_invariant import MissingStandardNameSource

pytestmark = pytest.mark.graph

OLD = "toroidal_argon_velocity_at_separatrix"
NEW = "toroidal_argon_velocity_at_plasma_boundary"

LIVE_PATH = "summary/local/separatrix/velocity_phi/argon/value"
REMOVED_PATH = "summary/local/separatrix/velocity_tor/argon/value"


def _reset() -> None:
    with GraphClient() as gc:
        gc.query("MATCH (n) DETACH DELETE n")


def _name(gc: GraphClient, name_id: str, *, paths: list[str]) -> None:
    gc.query(
        """
        CREATE (sn:StandardName {
          id: $id, name_stage: 'accepted', status: 'accepted',
          origin: 'pipeline', description: 'the toroidal argon velocity',
          docs_stage: 'accepted', source_paths: $paths})
        """,
        id=name_id,
        paths=paths,
    )


def _source(gc: GraphClient, sn_id: str, path: str, *, status: str) -> None:
    gc.query(
        """
        MATCH (sn:StandardName {id: $sn_id})
        CREATE (source:StandardNameSource {
          id: $source_id, source_type: 'dd', source_id: $path,
          status: $status, produced_sn_id: $sn_id})
        CREATE (source)-[:PRODUCED_NAME]->(sn)
        CREATE (dd:IMASNode {id: $path, unit: 'm/s'})
        CREATE (source)-[:FROM_DD_PATH]->(dd)
        """,
        sn_id=sn_id,
        source_id=f"dd:{path}",
        path=path,
        status=status,
    )


def _read(name_id: str) -> dict:
    with GraphClient() as gc:
        rows = gc.query(
            """
            MATCH (sn:StandardName {id: $id})
            OPTIONAL MATCH (source:StandardNameSource)-[:PRODUCED_NAME]->(sn)
            RETURN sn.name_stage AS name_stage,
                   sn.stale_source_ids AS stale_source_ids,
                   [source IN collect(DISTINCT source) WHERE source IS NOT NULL
                     | source.id] AS sources
            """,
            id=name_id,
        )
    assert rows, f"{name_id} is missing from the graph"
    return dict(rows[0])


def _source_row(source_id: str) -> dict:
    with GraphClient() as gc:
        rows = gc.query(
            """
            MATCH (source:StandardNameSource {id: $id})
            OPTIONAL MATCH (source)-[:PRODUCED_NAME]->(holder:StandardName)
            RETURN source.status AS status,
                   source.produced_sn_id AS scalar,
                   collect(DISTINCT holder.id) AS holders
            """,
            id=source_id,
        )
    assert len(rows) == 1, f"{source_id} is missing from the graph"
    return dict(rows[0])


def _seed(gc: GraphClient, *, statuses: dict[str, str]) -> None:
    paths = list(statuses)
    _name(gc, OLD, paths=paths)
    for path, status in statuses.items():
        _source(gc, OLD, path, status=status)


def test_rename_migrates_the_live_binding_and_leaves_the_stale_one_behind(
    seed_dd_resolutions,
):
    """The live source follows the rename; the stale source keeps its record."""
    _reset()
    seed_dd_resolutions(
        [
            {"path": LIVE_PATH, "effective": "m/s"},
            {"path": REMOVED_PATH, "effective": "m/s"},
        ]
    )
    with GraphClient() as gc:
        _seed(gc, statuses={LIVE_PATH: "attached", REMOVED_PATH: "stale"})

    plan = apply_edit(
        target=OLD,
        rename=NEW,
        reason="the boundary locator is the ruled spelling at the plasma boundary",
        gc=GraphClient(),
    )
    assert plan.blocked is None, plan.blocked
    assert plan.applied is True

    live = _read(NEW)
    assert live["name_stage"] != "superseded"
    assert live["sources"] == [f"dd:{LIVE_PATH}"]
    assert live["stale_source_ids"] == [f"dd:{REMOVED_PATH}"]

    # The stale binding is untouched: it still carries its status and still
    # belongs to the retired spelling, which is where its history lives now.
    stale = _source_row(f"dd:{REMOVED_PATH}")
    assert stale["status"] == "stale"
    assert stale["holders"] == [OLD]
    assert stale["scalar"] == OLD
    assert _read(OLD)["name_stage"] == "superseded"


def test_rename_is_refused_when_the_only_binding_is_stale(seed_dd_resolutions):
    """A name with nothing live to migrate is refused, not silently renamed."""
    _reset()
    seed_dd_resolutions([{"path": REMOVED_PATH, "effective": "m/s"}])
    with GraphClient() as gc:
        _seed(gc, statuses={REMOVED_PATH: "stale"})

    # A live name must be produced by a live source, so the rename is refused
    # before it reaches the writer at all — the exclusion above never turns a
    # name with no live binding into a renamable one.
    with pytest.raises(MissingStandardNameSource):
        apply_edit(
            target=OLD,
            rename=NEW,
            reason=(
                "the boundary locator is the ruled spelling at the plasma boundary"
            ),
            gc=GraphClient(),
        )
    assert _read(OLD)["name_stage"] == "accepted"
    with GraphClient() as gc:
        assert (
            gc.query("MATCH (sn:StandardName {id: $id}) RETURN sn.id AS id", id=NEW)
            == []
        )
    assert _source_row(f"dd:{REMOVED_PATH}")["holders"] == [OLD]
