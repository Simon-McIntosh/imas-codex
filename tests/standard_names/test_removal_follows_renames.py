"""A removed DD path moves its name to the successor, or supersedes it.

A DD rebuild stamps every path it dropped with ``lifecycle_status='removed'``.
``reconcile_standard_name_sources`` marks the sources on such a path stale,
and marking alone strands the name those sources produced: no writer's
transaction runs, so the last-producing-source invariant never fires.

Reconciliation therefore resolves the name in the statement that marks the
source stale, following the removed path's ``RENAMED_TO`` successor: it moves
the binding to an unheld successor's source and keeps the name live, supersedes
the name onto a successor another live name holds, and retires the name when
no successor exists or the successor's unit disagrees.

Every fixture is prefix-isolated, so the cleanup touches only nodes this module
created.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

_PREFIX = "test_removal_follows_renames__"


@pytest.fixture()
def _gc(disposable_neo4j_uri: str):
    from imas_codex.graph.client import GraphClient

    client = GraphClient(
        uri=disposable_neo4j_uri,
        username="neo4j",
        password="",
        graph_name="removal-follows-renames-tests",
    )
    client.get_stats()
    yield client
    client.close()


@pytest.fixture()
def _clean(_gc):
    def _wipe() -> None:
        for label in (
            "StandardName",
            "StandardNameSource",
            "StandardNameChange",
            "IMASNode",
            "Unit",
        ):
            _gc.query(
                f"MATCH (n:{label}) WHERE n.id CONTAINS $p DETACH DELETE n",
                p=_PREFIX,
            )

    _wipe()
    yield
    _wipe()


def _path(suffix: str) -> str:
    return f"{_PREFIX}{suffix}"


def _seed_removed_source(
    gc,
    *,
    name: str,
    removed_path: str,
    unit: str = "m",
    name_stage: str = "draft",
    name_status: str = "draft",
) -> str:
    """Seed a live name produced only by a source on a removed DD path."""
    gc.query(
        """
        MERGE (removed:IMASNode {id: $removed_path})
        SET removed.lifecycle_status = 'removed'
        MERGE (sn:StandardName {id: $name})
        SET sn.name_stage = $stage, sn.status = $status, sn.unit = $unit
        MERGE (sns:StandardNameSource {id: 'dd:' + $removed_path})
        SET sns.source_type = 'dd', sns.source_id = $removed_path,
            sns.status = 'extracted'
        MERGE (sns)-[:FROM_DD_PATH]->(removed)
        MERGE (sns)-[:PRODUCED_NAME]->(sn)
        """,
        name=name,
        removed_path=removed_path,
        unit=unit,
        stage=name_stage,
        status=name_status,
    )
    return f"dd:{removed_path}"


def _reconcile(source_type: str = "dd") -> dict:
    from imas_codex.standard_names.graph_store.graph_audits import (
        reconcile_standard_name_sources,
    )

    return reconcile_standard_name_sources(source_type)


def _name_row(gc, name: str) -> dict:
    rows = gc.query(
        """
        MATCH (sn:StandardName {id: $name})
        RETURN sn.name_stage AS name_stage, sn.status AS status
        """,
        name=name,
    )
    return dict(rows[0]) if rows else {}


def _live_producers(gc, name: str) -> list[dict]:
    return [
        dict(row)
        for row in gc.query(
            """
            MATCH (sns:StandardNameSource)-[:PRODUCED_NAME]->(sn:StandardName {id: $name})
            WHERE coalesce(sns.status, '') <> 'stale'
            RETURN sns.id AS id, sns.source_id AS source_id
            """,
            name=name,
        )
    ]


@pytest.mark.graph
def test_removed_path_with_unheld_successor_moves_the_binding(_gc, _clean):
    """The name stays live and the successor's source becomes its producer."""
    name = _path("radial_coordinate")
    removed = _path("old_quantity/r")
    successor = _path("new_quantity/r")
    _seed_removed_source(_gc, name=name, removed_path=removed, unit="m")
    _gc.query(
        """
        MATCH (old:IMASNode {id: $removed})
        MERGE (succ:IMASNode {id: $successor})
        MERGE (old)-[:RENAMED_TO]->(succ)
        """,
        successor=successor,
        removed=removed,
    )
    # The removed path's replacement is a real source of the DD build.
    _gc.query(
        """
        MATCH (succ:IMASNode {id: $successor})
        MERGE (sns:StandardNameSource {id: 'dd:' + $successor})
        MERGE (sns)-[:FROM_DD_PATH]->(succ)
        SET sns.source_type = 'dd', sns.source_id = $successor,
            sns.status = 'extracted'
        """,
        successor=successor,
    )

    _reconcile()

    assert _name_row(_gc, name)["name_stage"] == "draft"
    producers = _live_producers(_gc, name)
    assert [row["source_id"] for row in producers] == [successor]

    successor_source = _gc.query(
        "MATCH (sns:StandardNameSource {id: 'dd:' + $successor}) "
        "RETURN sns.status AS status",
        successor=successor,
    )
    assert successor_source[0]["status"] == "extracted"


@pytest.mark.graph
def test_held_successor_supersedes_onto_the_holder_with_a_receipt(_gc, _clean):
    """A successor another live name holds retires this name onto that holder."""
    name = _path("loser_name")
    holder = _path("holder_name")
    removed = _path("old_quantity/phi")
    successor = _path("new_quantity/phi")
    _seed_removed_source(_gc, name=name, removed_path=removed, unit="m")
    _gc.query(
        """
        MATCH (old:IMASNode {id: $removed})
        MERGE (succ:IMASNode {id: $successor})
        MERGE (old)-[:RENAMED_TO]->(succ)
        MERGE (holder:StandardName {id: $holder})
        MERGE (held:StandardNameSource {id: 'dd:' + $successor})
        MERGE (held)-[:FROM_DD_PATH]->(succ)
        MERGE (held)-[:PRODUCED_NAME]->(holder)
        SET holder.name_stage = 'draft', holder.status = 'draft', holder.unit = 'm',
            held.source_type = 'dd', held.source_id = $successor,
            held.status = 'extracted'
        """,
        successor=successor,
        removed=removed,
        holder=holder,
    )

    _reconcile()

    assert _name_row(_gc, name)["name_stage"] == "superseded"
    edges = _gc.query(
        """
        MATCH (holder:StandardName {id: $holder})-[:REFINED_FROM]->(sn:StandardName {id: $name})
        RETURN count(*) AS n
        """,
        holder=holder,
        name=name,
    )
    assert edges[0]["n"] == 1
    receipts = _gc.query(
        """
        MATCH (sn:StandardName {id: $holder})-[:HAS_INTERNAL_CHANGE]->(c:StandardNameChange)
        RETURN c.from_name AS from_name, c.to_name AS to_name, c.operation AS operation
        """,
        holder=holder,
    )
    assert [dict(row) for row in receipts] == [
        {
            "from_name": name,
            "to_name": holder,
            "operation": "removed_path_successor_holder",
        }
    ]
    assert _live_producers(_gc, name) == []


@pytest.mark.graph
def test_removed_path_without_successor_retires_with_a_receipt(_gc, _clean):
    """No successor: the name is retired without a successor edge."""
    name = _path("orphan_name")
    removed = _path("vanished/quantity")
    _seed_removed_source(_gc, name=name, removed_path=removed, unit="m")

    _reconcile()

    assert _name_row(_gc, name)["name_stage"] == "superseded"
    edges = _gc.query(
        "MATCH (:StandardName)-[:REFINED_FROM]->(sn:StandardName {id: $name}) "
        "RETURN count(*) AS n",
        name=name,
    )
    assert edges[0]["n"] == 0
    receipts = _gc.query(
        """
        MATCH (sn:StandardName {id: $name})-[:HAS_INTERNAL_CHANGE]->(c:StandardNameChange)
        RETURN c.operation AS operation, c.to_name AS to_name
        """,
        name=name,
    )
    assert [dict(row) for row in receipts] == [
        {"operation": "removed_path_retirement", "to_name": name}
    ]
    assert _live_producers(_gc, name) == []


@pytest.mark.graph
def test_unit_mismatch_supersedes_instead_of_moving(_gc, _clean):
    """A successor in a different unit never takes over the binding."""
    name = _path("unit_mismatch")
    removed = _path("old_quantity/energy")
    successor = _path("new_quantity/energy")
    _seed_removed_source(_gc, name=name, removed_path=removed, unit="m")
    _gc.query(
        """
        MATCH (old:IMASNode {id: $removed})
        MERGE (succ:IMASNode {id: $successor})
        MERGE (old)-[:RENAMED_TO]->(succ)
        MERGE (sns:StandardNameSource {id: 'dd:' + $successor})
        MERGE (sns)-[:FROM_DD_PATH]->(succ)
        SET succ.unit = 'kg',
            sns.source_type = 'dd', sns.source_id = $successor,
            sns.status = 'extracted'
        """,
        successor=successor,
        removed=removed,
    )

    _reconcile()

    assert _name_row(_gc, name)["name_stage"] == "superseded"
    assert _live_producers(_gc, name) == []
    bound = _gc.query(
        "MATCH (:StandardNameSource {id: 'dd:' + $successor})"
        "-[:PRODUCED_NAME]->(sn:StandardName) RETURN count(sn) AS n",
        successor=successor,
    )
    assert bound[0]["n"] == 0


@pytest.mark.graph
def test_a_signal_source_draining_a_name_leaves_it_live(_gc, _clean):
    """A signals source names no IMASNode, so no successor rule applies to it."""
    name = _path("signal_backed_name")
    signal_id = _path("coil_current")
    _gc.query(
        """
        MERGE (sn:StandardName {id: $name})
        SET sn.name_stage = 'draft', sn.status = 'draft', sn.unit = 'A'
        MERGE (sns:StandardNameSource {id: 'signals:' + $signal_id})
        SET sns.source_type = 'signals', sns.source_id = $signal_id,
            sns.status = 'extracted'
        MERGE (sns)-[:PRODUCED_NAME]->(sn)
        """,
        name=name,
        signal_id=signal_id,
    )

    _reconcile("signals")

    source = _gc.query(
        "MATCH (sns:StandardNameSource {id: 'signals:' + $signal_id}) "
        "RETURN sns.status AS status",
        signal_id=signal_id,
    )
    assert [row["status"] for row in source] == ["stale"]
    assert _name_row(_gc, name) == {"name_stage": "draft", "status": "draft"}
    receipts = _gc.query(
        "MATCH (:StandardName {id: $name})-[:HAS_INTERNAL_CHANGE]->"
        "(c:StandardNameChange) RETURN count(c) AS n",
        name=name,
    )
    assert receipts[0]["n"] == 0


def test_dead_end_revival_reads_the_source_nodes_type():
    """A worker outcome omits the batch type; the source node owns it."""
    from imas_codex.standard_names.graph_store.graph_source_bindings import (
        _lock_claimed_name_bindings,
    )

    graph = MagicMock()
    graph.query.return_value = [
        {
            "id": "dd:test_removal_follows_renames__dust/value",
            "outcome": "winner",
            "binding_kind": "lifecycle_revival",
            "candidate_id": "test_removal_follows_renames__dust",
            "target_stage": "drafted",
        }
    ]
    batch = [
        {
            "sns_id": "dd:test_removal_follows_renames__dust/value",
            "source_id": "test_removal_follows_renames__dust/value",
            "sn_id": "test_removal_follows_renames__dust",
            "unit": "m",
            "claim_token": "worker-owner",
            "claim_seq": 3,
        }
    ]

    with patch(
        "imas_codex.standard_names.graph_store.graph_source_bindings."
        "_guard_existing_target_pairings",
        return_value={},
    ):
        _lock_claimed_name_bindings(
            graph,
            batch,
            allow_missing=True,
            allow_own_pending_reservation=True,
            allow_dead_end_revival=True,
        )

    cypher = next(
        call.args[0]
        for call in graph.query.call_args_list
        if "AS outcome" in call.args[0]
    )
    assert "sns.source_type = 'dd'" in cypher
    assert "b.source_type" not in cypher
