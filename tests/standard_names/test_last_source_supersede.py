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


def _receipt_names(operation: str = "last_source_supersede") -> list[tuple[str, str]]:
    with GraphClient() as gc:
        rows = gc.query(
            """
            MATCH (c:StandardNameChange)
            WHERE c.operation = $operation
            RETURN c.from_name AS from_name, c.to_name AS to_name
            ORDER BY from_name
            """,
            operation=operation,
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


def test_source_migration_supersedes_the_drained_holder(
    disposable_neo4j_uri: str,
) -> None:
    """The shared move primitive supersedes a holder it leaves sourceless."""
    _seed()
    with GraphClient() as gc:
        moved = retarget_standard_name_sources(
            gc,
            OLD,
            NEW,
            operation="retarget",
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
    assert _receipt_names("composed_source_moved") == [(OLD, NEW)]


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


def _claim_seed(*, provisional: bool = True) -> None:
    """Seed a fenced, extracted claim whose provisional edge points at OLD."""
    _seed(source_status="extracted", claim_token="tok", claim_seq=3)
    with GraphClient() as gc:
        gc.query(
            """
            MATCH (s:StandardNameSource {id: 'dd:test/last_source/one'})
                  -[p:PRODUCED_NAME]->(:StandardName)
            SET p.provisional = $provisional,
                p.claim_token = 'tok', p.claim_seq = 3,
                p.created_target = true
            """,
            provisional=provisional,
        )


class _Services(types.SimpleNamespace):
    """Collaborators the claimed-binding writers reach through ``services``."""

    def __getattr__(self, name: str):
        from imas_codex.standard_names.graph_store import (
            graph_audits,
            graph_source_bindings,
        )

        for module in (graph_source_bindings, graph_audits):
            if hasattr(module, name):
                return getattr(module, name)
        raise AttributeError(name)


def _services():
    from imas_codex.standard_names.graph_store import graph_foundation

    return _Services(
        GraphClient=GraphClient,
        NameStage=graph_foundation.NameStage,
        _STABLE_BINDING_NAME_STAGES=graph_foundation._STABLE_BINDING_NAME_STAGES,
        _TransactionQuery=graph_foundation._TransactionQuery,
    )


def test_attached_source_moves_and_supersedes_the_drained_holder(
    disposable_neo4j_uri: str,
) -> None:
    """An attach that moves an extracted source drains its former holder."""
    from imas_codex.standard_names.graph_store.graph_source_bindings import (
        mark_sources_attached,
    )

    _claim_seed()
    affected = mark_sources_attached(
        token="tok",
        claim_seq=3,
        source_ids=["dd:test/last_source/one"],
        standard_name_id=NEW,
        services=types.SimpleNamespace(GraphClient=GraphClient),
    )
    assert affected == 1
    assert _producer_of("dd:test/last_source/one") == NEW
    assert _lifecycle(OLD)["name_stage"] == "superseded"
    assert _successor_id(OLD) == NEW
    assert _receipt_names("attached_source_moved") == [(OLD, NEW)]


def test_staged_candidate_produces_the_source_before_the_holder_is_superseded(
    disposable_neo4j_uri: str,
) -> None:
    """A staged candidate must produce the source it is named successor for.

    The supersession is written in the same transaction as the staging, so the
    binding has to exist first: otherwise a later rich write that fails leaves
    the former holder superseded towards a candidate that produces nothing.
    """
    from imas_codex.standard_names.graph_store.graph_source_bindings import (
        stage_claimed_generated_candidates,
    )

    _claim_seed()
    stage_claimed_generated_candidates(
        [
            {
                "sns_id": "dd:test/last_source/one",
                "source_id": "test/last_source/one",
                "sn_id": NEW,
                "model": "test/model",
                "claim_token": "tok",
                "claim_seq": 3,
            }
        ],
        services=_services(),
    )
    assert _lifecycle(OLD)["name_stage"] == "superseded"
    assert _successor_id(OLD) == NEW
    assert _producer_of("dd:test/last_source/one") == NEW
    assert _receipt_names("compose_rebind") == [(OLD, NEW)]


def test_claimed_lock_reports_the_binding_it_released(
    disposable_neo4j_uri: str,
) -> None:
    """The released-out plumbing reports each provisional edge it removed."""
    from imas_codex.standard_names.graph_store.graph_source_bindings import (
        _lock_claimed_name_bindings,
    )

    _claim_seed()
    released: list[dict[str, str]] = []
    with GraphClient() as gc:
        with gc.session() as session:
            tx = session.begin_transaction()
            from imas_codex.standard_names.graph_store import graph_foundation

            handle = graph_foundation._TransactionQuery(tx)
            _lock_claimed_name_bindings(
                handle,
                [
                    {
                        "sns_id": "dd:test/last_source/one",
                        "source_id": "test/last_source/one",
                        "sn_id": NEW,
                        "claim_token": "tok",
                        "claim_seq": 3,
                    }
                ],
                services=_services(),
                allow_missing=True,
                allow_own_pending_reservation=True,
                released_out=released,
            )
            tx.commit()
    assert released == [{"old_id": OLD, "sns_id": "dd:test/last_source/one"}]


def test_refined_name_moves_the_predecessors_source(
    disposable_neo4j_uri: str,
) -> None:
    """The refine writer itself is what moves the predecessor's source.

    Driving ``persist_refined_name`` directly keeps this case measuring the
    refine writer rather than the retarget primitive it is built on.
    """
    from imas_codex.standard_names.graph_store.graph_refinement import (
        persist_refined_name,
    )
    from imas_codex.standard_names.kind_derivation import derive_kind

    _seed()
    with GraphClient() as gc:
        gc.query("MATCH (n:StandardName {id: $new}) DETACH DELETE n", new=NEW)
    persist_refined_name(
        old_name=OLD,
        new_name=NEW,
        description="A refined name for the last-source test.",
        kind=derive_kind(NEW),
        expected_old_stage="drafted",
        services=_Services(GraphClient=GraphClient),
    )
    assert _producer_of("dd:test/last_source/one") == NEW
    assert _lifecycle(OLD)["name_stage"] == "superseded"
    assert _lifecycle(OLD)["successors"] == 1
    assert _successor_id(OLD) == NEW
