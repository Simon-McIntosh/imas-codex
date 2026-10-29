"""A write that moves a name's last producing source supersedes that name.

Every writer that moves a ``PRODUCED_NAME`` binding must leave the former
holder either still producing a source or superseded with a receipt-carrying
successor edge to the name that now holds the source. These tests drive each
writer on the disposable graph and read the resulting lifecycle back, so a
regression that leaves the name live and unsourced fails here rather than in
the catalog.
"""

from __future__ import annotations

import types

import pytest

from imas_codex.graph.client import GraphClient
from imas_codex.standard_names.provenance_lifecycle import (
    retarget_standard_name_sources,
)
from imas_codex.standard_names.provenance_rebuild import bind_recovery_sources
from imas_codex.standard_names.source_invariant import find_sourceless_live_names

pytestmark = pytest.mark.graph

OLD = "test-last-source-old"
NEW = "test-last-source-new"


def _seed(
    *,
    second_source: bool = False,
    source_status: str = "attached",
    claim_token: str | None = None,
    claim_seq: int | None = None,
) -> None:
    """Create a live holder, a successor, and the source(s) producing the holder."""
    with GraphClient() as gc:
        gc.query("MATCH (n) DETACH DELETE n")
        gc.query(
            """
            CREATE (old:StandardName {
              id: $old, name_stage: 'drafted', status: 'draft',
              origin: 'pipeline'})
            CREATE (new:StandardName {
              id: $new, name_stage: 'drafted', status: 'draft',
              origin: 'pipeline'})
            CREATE (d1:IMASNode {id: 'test/last_source/one'})
            CREATE (s1:StandardNameSource {
              id: 'dd:test/last_source/one', source_type: 'dd',
              source_id: 'test/last_source/one', status: $status,
              produced_sn_id: $old, claim_token: $token, claim_seq: $seq})
            CREATE (s1)-[:PRODUCED_NAME]->(old)
            CREATE (s1)-[:FROM_DD_PATH]->(d1)
            """,
            old=OLD,
            new=NEW,
            status=source_status,
            token=claim_token,
            seq=claim_seq,
        )
        if second_source:
            gc.query(
                """
                MATCH (old:StandardName {id: $old})
                CREATE (d2:IMASNode {id: 'test/last_source/two'})
                CREATE (s2:StandardNameSource {
                  id: 'dd:test/last_source/two', source_type: 'dd',
                  source_id: 'test/last_source/two', status: 'attached',
                  produced_sn_id: $old})
                CREATE (s2)-[:PRODUCED_NAME]->(old)
                CREATE (s2)-[:FROM_DD_PATH]->(d2)
                """,
                old=OLD,
            )


def _lifecycle(name_id: str) -> dict:
    with GraphClient() as gc:
        rows = gc.query(
            """
            MATCH (n:StandardName {id: $id})
            RETURN n.name_stage AS name_stage, n.status AS status,
                   COUNT { (succ:StandardName)-[:REFINED_FROM]->(n) }
                     AS successors,
                   COUNT { (n)-[:HAS_INTERNAL_CHANGE]->(
                     c:StandardNameChange) } AS receipts
            """,
            id=name_id,
        )
        return dict(rows[0]) if rows else {}


def _receipt_names() -> list[tuple[str, str]]:
    with GraphClient() as gc:
        rows = gc.query(
            """
            MATCH (c:StandardNameChange)
            WHERE c.operation = 'last_source_supersede'
            RETURN c.from_name AS from_name, c.to_name AS to_name
            ORDER BY from_name
            """
        )
        return [(str(r["from_name"]), str(r["to_name"])) for r in rows]


def _successor_id(name_id: str) -> str | None:
    with GraphClient() as gc:
        rows = gc.query(
            """
            MATCH (succ:StandardName)-[:REFINED_FROM]->(
              n:StandardName {id: $id})
            RETURN succ.id AS id
            """,
            id=name_id,
        )
        return str(rows[0]["id"]) if rows else None


def _producer_of(source_id: str) -> str | None:
    with GraphClient() as gc:
        rows = gc.query(
            """
            MATCH (sns:StandardNameSource {id: $sid})-[:PRODUCED_NAME]->(
              sn:StandardName)
            RETURN sn.id AS id
            """,
            sid=source_id,
        )
        return str(rows[0]["id"]) if rows else None


@pytest.mark.parametrize("operation", ["retarget", "refine"])
def test_source_migration_supersedes_the_drained_holder(
    disposable_neo4j_uri: str, operation: str
) -> None:
    """The shared move primitive supersedes a holder it leaves sourceless."""
    _seed()
    with GraphClient() as gc:
        moved = retarget_standard_name_sources(
            gc,
            OLD,
            NEW,
            operation=operation,
            source_ids=["dd:test/last_source/one"],
            expected_current_bindings={"dd:test/last_source/one": OLD},
            enforce_consistency=False,
        )
    assert moved == 1
    assert _producer_of("dd:test/last_source/one") == NEW
    assert _lifecycle(OLD)["name_stage"] == "superseded"
    assert _lifecycle(OLD)["successors"] == 1
    assert _receipt_names() == [(OLD, NEW)]
    assert _successor_id(OLD) == NEW


def test_recovery_bind_supersedes_the_drained_holder(
    disposable_neo4j_uri: str,
) -> None:
    """Recovery re-binds a canonical source, so it is a move, not an addition."""
    _seed()
    bound = bind_recovery_sources(
        NEW,
        [
            {
                "id": "dd:test/last_source/one",
                "source_type": "dd",
                "dd_path": "test/last_source/one",
                "status": "attached",
            }
        ],
    )
    assert bound == 1
    assert _producer_of("dd:test/last_source/one") == NEW
    assert _lifecycle(OLD)["name_stage"] == "superseded"
    assert _lifecycle(OLD)["successors"] == 1
    assert _successor_id(OLD) == NEW


def test_compose_bind_supersedes_the_drained_holder(
    disposable_neo4j_uri: str,
) -> None:
    """A compose bind that moves an extracted source drains its former holder."""
    from imas_codex.standard_names.graph_store.graph_source_bindings import (
        mark_sources_composed,
    )

    _seed(source_status="extracted", claim_token="tok", claim_seq=3)
    affected = mark_sources_composed(
        token="tok",
        claim_seq=3,
        source_ids=["dd:test/last_source/one"],
        standard_name_id=NEW,
        services=types.SimpleNamespace(GraphClient=GraphClient),
    )
    assert affected == 1
    assert _producer_of("dd:test/last_source/one") == NEW
    assert _lifecycle(OLD)["name_stage"] == "superseded"
    assert _lifecycle(OLD)["successors"] == 1
    assert _successor_id(OLD) == NEW


def test_control_holder_with_another_source_stays_live(
    disposable_neo4j_uri: str,
) -> None:
    """A holder that keeps a producing source is not superseded."""
    _seed(second_source=True)
    with GraphClient() as gc:
        retarget_standard_name_sources(
            gc,
            OLD,
            NEW,
            operation="retarget",
            source_ids=["dd:test/last_source/one"],
            expected_current_bindings={"dd:test/last_source/one": OLD},
            enforce_consistency=False,
        )
        sourceless = find_sourceless_live_names(gc, [OLD])
    assert sourceless == []
    assert _lifecycle(OLD)["name_stage"] == "drafted"
    assert _lifecycle(OLD)["successors"] == 0
    assert _producer_of("dd:test/last_source/two") == OLD
