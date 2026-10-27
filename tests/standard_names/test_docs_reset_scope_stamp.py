"""A scoped docs reset binds its batch even when it resets nothing.

The reset and the scope stamp answer two different questions. A family anchor
that is already at ``docs_stage = 'pending'`` has nothing to reset, and still
must carry the run id its batch was minted under, or the scoped claim matches
no node and the drain halts the family with zero calls.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator

import pytest

from imas_codex.standard_names import graph_ops
from imas_codex.standard_names.graph_store import (
    graph_docs_persistence as _docs_persistence,
)

pytestmark = pytest.mark.graph

_PREFIX = "docs_reset_scope_test__"


def _uid(tag: str) -> str:
    return f"{_PREFIX}{tag}_{uuid.uuid4().hex[:8]}"


@pytest.fixture()
def _names(disposable_neo4j_uri: str) -> Iterator[list[str]]:
    """Seed two accepted, already-pending names and remove them afterwards."""
    from imas_codex.graph.client import GraphClient

    ids = [_uid("anchor"), _uid("sibling")]
    unit_id = _uid("unit")
    with GraphClient() as gc:
        # A shared unit lets the claim's batch expansion reach both names, so
        # the assertion covers the whole seeded batch rather than one seed.
        gc.query(
            """
            UNWIND $ids AS id
            MERGE (u:Unit {id: $unit_id})
            CREATE (sn:StandardName {id: id})
            SET sn.name_stage        = 'accepted',
                sn.docs_stage        = 'pending',
                sn.reviewer_score_name = 0.9,
                sn.physics_domain    = 'core_profiles',
                sn.documentation     = '',
                sn.kind              = 'scalar'
            MERGE (sn)-[:HAS_UNIT]->(u)
            """,
            ids=ids,
            unit_id=unit_id,
        )
    try:
        yield ids
    finally:
        with GraphClient() as gc:
            gc.query(
                "MATCH (n:DocsRevision) WHERE n.id STARTS WITH $p DETACH DELETE n",
                p=_PREFIX,
            )
            gc.query(
                "MATCH (n:StandardName) WHERE n.id STARTS WITH $p DETACH DELETE n",
                p=_PREFIX,
            )
            gc.query(
                "MATCH (n:Unit) WHERE n.id STARTS WITH $p DETACH DELETE n",
                p=_PREFIX,
            )


@pytest.fixture()
def _run_id() -> str:
    return f"scope-{uuid.uuid4()}"


def _read_run_id(gc, sn_id: str) -> str | None:
    rows = gc.query(
        "MATCH (sn:StandardName {id: $id}) RETURN sn.run_id AS run_id",
        id=sn_id,
    )
    return rows[0]["run_id"] if rows else None


def test_scoped_reset_stamps_names_that_need_no_reset(
    disposable_neo4j_uri: str, _names: list[str], _run_id: str
) -> None:
    """Zero eligible still binds the batch, and the scoped claim then sees it."""
    result = _docs_persistence.reset_standard_name_docs(sn_ids=_names, run_id=_run_id)

    assert result == {"eligible": 0, "reset": 0}

    from imas_codex.graph.client import GraphClient

    with GraphClient() as gc:
        stamped = [_read_run_id(gc, sn_id) for sn_id in _names]
    assert stamped == [_run_id, _run_id]

    claimed = _docs_persistence.claim_generate_docs_batch(
        batch_size=10, scope_run_id=_run_id
    )
    assert sorted(item["id"] for item in claimed) == sorted(_names)


def test_unscoped_reset_leaves_an_already_pending_name_alone(
    disposable_neo4j_uri: str, _names: list[str]
) -> None:
    """Without a run id there is no batch to bind, and nothing is written."""
    from imas_codex.graph.client import GraphClient

    stale_run = "earlier-run"
    with GraphClient() as gc:
        gc.query(
            "MATCH (sn:StandardName) WHERE sn.id IN $ids SET sn.run_id = $run_id",
            ids=_names,
            run_id=stale_run,
        )

    result = _docs_persistence.reset_standard_name_docs(sn_ids=_names)

    assert result == {"eligible": 0, "reset": 0}
    with GraphClient() as gc:
        assert [_read_run_id(gc, sn_id) for sn_id in _names] == [
            stale_run,
            stale_run,
        ]


def test_scoped_reset_of_an_eligible_name_still_resets_it(
    disposable_neo4j_uri: str, _run_id: str
) -> None:
    """An eligible name is snapshotted, reset to pending and bound, as before."""
    from imas_codex.graph.client import GraphClient

    sn_id = _uid("eligible")
    with GraphClient() as gc:
        gc.query(
            """
            CREATE (sn:StandardName {id: $id})
            SET sn.name_stage        = 'accepted',
                sn.docs_stage        = 'reviewed',
                sn.docs_chain_length = 0,
                sn.documentation     = 'Previous documentation text.',
                sn.physics_domain    = 'core_profiles',
                sn.kind              = 'scalar'
            """,
            id=sn_id,
        )
    try:
        result = _docs_persistence.reset_standard_name_docs(
            sn_ids=[sn_id], run_id=_run_id
        )

        assert result == {"eligible": 1, "reset": 1}
        with GraphClient() as gc:
            rows = gc.query(
                """
                MATCH (sn:StandardName {id: $id})
                OPTIONAL MATCH (sn)-[:DOCS_REVISION_OF]->(rev)
                RETURN sn.docs_stage AS docs_stage,
                       sn.docs_chain_length AS chain,
                       sn.run_id AS run_id,
                       collect(rev.id) AS revisions
                """,
                id=sn_id,
            )
        row = rows[0]
        assert row["docs_stage"] == "pending"
        assert row["chain"] == 1
        assert row["run_id"] == _run_id
        assert row["revisions"] == [f"{sn_id}#rev-0"]
    finally:
        with GraphClient() as gc:
            gc.query(
                "MATCH (n:DocsRevision) WHERE n.id STARTS WITH $p DETACH DELETE n",
                p=sn_id,
            )
            gc.query(
                "MATCH (n:StandardName {id: $id}) DETACH DELETE n",
                id=sn_id,
            )
