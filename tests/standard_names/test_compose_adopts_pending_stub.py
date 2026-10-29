"""A compose claim adopts an unowned pending projection stub of its own DD path.

A projection pass mints ``pending`` identities that carry a
``HAS_STANDARD_NAME`` edge from a DD node and no producing source. Composition
cannot bind such an identity — a pending name is a private reservation — so
every attempt collides, the source climbs to the claim cap, and the stub stays
live and sourceless for good.

These tests drive the two binding paths on the disposable graph: the batch
compose claim and the focused (explicit) claim. An unowned stub of the claimed
source's own DD path is adopted and enters ``drafted`` with a receipt; a stub
already owned by another claim is still refused; and a source parked at the
claim cap keeps the collision reason it was refused with.
"""

from __future__ import annotations

import pytest

from imas_codex.graph.client import GraphClient

pytestmark = pytest.mark.graph

PROBE_PATH = "test_adopt_projection/profiles_1d/psi"
PROBE_STUB = "test_adopt_projection_probe_stub"
PROBE_SOURCE = f"dd:{PROBE_PATH}"
TOKEN = "test-adopt-projection-token"
SEQ = 4
OWNER_SOURCE = "test_adopt_projection_owner"


@pytest.fixture()
def services():
    import imas_codex.standard_names.graph_ops  # noqa: F401 — wires the services
    from imas_codex.standard_names.graph_store import graph_foundation

    return graph_foundation.get_services()


def _wipe(gc: GraphClient) -> None:
    """Delete every row this suite creates, whatever its id namespace.

    The DD source id is ``dd:``-namespaced while the name and node ids are not,
    so the wipe matches anywhere in the id rather than at its start.
    """
    gc.query("MATCH (n) WHERE n.id CONTAINS 'test_adopt_projection' DETACH DELETE n")


@pytest.fixture()
def graph(disposable_neo4j_uri: str):
    with GraphClient() as gc:
        _wipe(gc)
        yield gc
        _wipe(gc)


def _seed_stub(gc: GraphClient, *, owner_token: str | None = None) -> None:
    """Create a pending projection stub plus the extracted source for its path."""
    gc.query(
        """
        MERGE (node:IMASNode {id: $path})
        ON CREATE SET node.node_category = 'scalar',
                      node.lifecycle_status = 'active'
        MERGE (stub:StandardName {id: $stub})
        ON CREATE SET stub.name_stage = 'pending', stub.status = 'draft',
                      stub.origin = 'projection'
        MERGE (node)-[:HAS_STANDARD_NAME]->(stub)
        """,
        path=PROBE_PATH,
        stub=PROBE_STUB,
    )
    gc.query(
        """
        MATCH (node:IMASNode {id: $path})
        MERGE (src:StandardNameSource {id: $source})
        ON CREATE SET src.source_type = 'dd', src.source_id = $path,
                      src.status = 'extracted', src.claim_token = $token,
                      src.claim_seq = $seq, src.attempt_count = 3
        MERGE (src)-[:FROM_DD_PATH]->(node)
        """,
        source=PROBE_SOURCE,
        path=PROBE_PATH,
        token=TOKEN,
        seq=SEQ,
    )
    if owner_token is not None:
        gc.query(
            """
            MATCH (stub:StandardName {id: $stub})
            MERGE (owner:StandardNameSource {id: $owner})
            SET owner.source_type = 'dd', owner.source_id = $path,
                owner.status = 'composed'
            MERGE (owner)-[produced:PRODUCED_NAME]->(stub)
            SET produced.provisional = true,
                produced.claim_token = $owner_token,
                produced.claim_seq = 1,
                stub.reservation_source_id = $owner
            """,
            stub=PROBE_STUB,
            owner=OWNER_SOURCE,
            path=PROBE_PATH,
            owner_token=owner_token,
        )


def _stub(gc: GraphClient) -> dict:
    rows = gc.query(
        """
        MATCH (stub:StandardName {id: $stub})
        RETURN stub.name_stage AS name_stage, stub.status AS status,
               COUNT { (stub)-[:HAS_INTERNAL_CHANGE]->(
                 c:StandardNameChange) } AS receipts
        """,
        stub=PROBE_STUB,
    )
    return dict(rows[0])


def _producers(gc: GraphClient) -> list[str]:
    return [
        str(row["id"])
        for row in gc.query(
            """
            MATCH (src:StandardNameSource)-[:PRODUCED_NAME]->(
              stub:StandardName {id: $stub})
            RETURN src.id AS id ORDER BY src.id
            """,
            stub=PROBE_STUB,
        )
    ]


def _adoption_receipts(gc: GraphClient) -> list[dict]:
    return [
        dict(row)
        for row in gc.query(
            """
            MATCH (stub:StandardName {id: $stub})-[:HAS_INTERNAL_CHANGE]->(
              c:StandardNameChange)
            WHERE c.operation = 'adopt_pending_projection'
            RETURN c.from_name AS from_name, c.to_name AS to_name,
                   c.adopted_source_id AS adopted_source_id
            """,
            stub=PROBE_STUB,
        )
    ]


def _candidate() -> dict:
    return {
        "sns_id": PROBE_SOURCE,
        "source_id": PROBE_PATH,
        "source_type": "dd",
        "sn_id": PROBE_STUB,
        "claim_token": TOKEN,
        "claim_seq": SEQ,
        "unit": None,
    }


def test_batch_claim_adopts_an_unowned_projection_stub(graph, services) -> None:
    """The batch compose claim binds a pending projection stub of its own path."""
    from imas_codex.standard_names.graph_store.graph_source_bindings import (
        stage_claimed_generated_candidates,
    )

    _seed_stub(graph)
    winners = stage_claimed_generated_candidates([_candidate()], services=services)
    assert winners == [PROBE_SOURCE], "the stub was not adopted: " + str(_stub(graph))

    stub = _stub(graph)
    assert stub["name_stage"] == "drafted"
    assert stub["receipts"] == 1
    assert _adoption_receipts(graph) == [
        {
            "from_name": "pending",
            "to_name": "drafted",
            "adopted_source_id": PROBE_SOURCE,
        }
    ]


def test_focused_claim_adopts_the_stub_and_writes_the_producing_edge(
    graph, services
) -> None:
    """The focused claim path adopts the stub and binds the source to it."""
    from imas_codex.standard_names.graph_store.graph_source_bindings import (
        persist_claimed_attachments,
    )

    _seed_stub(graph)
    winners = persist_claimed_attachments(
        [
            {
                "sns_id": PROBE_SOURCE,
                "source_id": PROBE_PATH,
                "source_type": "dd",
                "standard_name": PROBE_STUB,
                "claim_token": TOKEN,
                "claim_seq": SEQ,
                "unit": None,
            }
        ],
        services=services,
    )
    assert winners == [PROBE_SOURCE]
    assert _producers(graph) == [PROBE_SOURCE]
    stub = _stub(graph)
    assert stub["name_stage"] == "drafted"
    assert stub["receipts"] == 1

    status = graph.query(
        "MATCH (src:StandardNameSource {id: $id}) RETURN src.status AS status",
        id=PROBE_SOURCE,
    )
    assert status[0]["status"] == "attached"


def test_production_writer_adopts_an_unowned_projection_stub(graph, services) -> None:
    """The live compose writer adopts the stub and finalizes its source."""
    from imas_codex.standard_names.graph_store.graph_catalog import (
        persist_generated_name_batch,
    )

    _seed_stub(graph)
    winners = persist_generated_name_batch(
        [
            {
                "id": PROBE_STUB,
                "source_id": PROBE_PATH,
                "source_types": ["dd"],
                "unit": None,
                "source_claim_token": TOKEN,
                "source_claim_seq": SEQ,
            }
        ],
        compose_model="test-adopt-projection-model",
        return_winner_ids=True,
    )
    assert winners == [PROBE_SOURCE], "the stub was not adopted: " + str(_stub(graph))

    stub = _stub(graph)
    assert stub["name_stage"] == "drafted"
    assert _adoption_receipts(graph) == [
        {
            "from_name": "pending",
            "to_name": "drafted",
            "adopted_source_id": PROBE_SOURCE,
        }
    ]
    assert _producers(graph) == [PROBE_SOURCE]
    status = graph.query(
        "MATCH (src:StandardNameSource {id: $id}) RETURN src.status AS status",
        id=PROBE_SOURCE,
    )
    assert status[0]["status"] == "composed"


def test_stub_owned_by_another_claim_is_still_refused(graph, services) -> None:
    """An adoption candidate owned by another claim remains a collision."""
    from imas_codex.standard_names.graph_store.graph_source_bindings import (
        stage_claimed_generated_candidates,
    )

    _seed_stub(graph, owner_token="test-adopt-projection-other-claim")
    winners = stage_claimed_generated_candidates([_candidate()], services=services)
    assert winners == []
    stub = _stub(graph)
    assert stub["name_stage"] == "pending"
    assert stub["receipts"] == 0
    assert _adoption_receipts(graph) == []
    assert _producers(graph) == [OWNER_SOURCE]


def test_parked_source_keeps_the_collision_reason_beside_the_cap_message(
    graph, services
) -> None:
    """Parking at the claim cap preserves the refusal that led there."""
    from imas_codex.standard_names.graph_store.graph_cost_and_claims import (
        _MAX_COMPOSE_CLAIM_ATTEMPTS,
    )
    from imas_codex.standard_names.orphan_sweep import _orphan_sweep_tick

    collision = (
        'standard-name binding collision: candidate "test_adopt_projection_probe_stub" '
        'has non-bindable name_stage "pending" at composition attempt 5'
    )
    _seed_stub(graph)
    graph.query(
        """
        MATCH (src:StandardNameSource {id: $source})
        SET src.status = 'extracted',
            src.claim_token = null,
            src.claimed_at = null,
            src.attempt_count = $cap,
            src.last_error = $collision
        """,
        source=PROBE_SOURCE,
        cap=_MAX_COMPOSE_CLAIM_ATTEMPTS,
        collision=collision,
    )

    counts = _orphan_sweep_tick(timeout_s=600)
    assert counts["compose_attempt_cap"] == 1

    rows = graph.query(
        "MATCH (src:StandardNameSource {id: $id}) "
        "RETURN src.status AS status, src.last_error AS last_error",
        id=PROBE_SOURCE,
    )
    assert rows[0]["status"] == "failed"
    assert rows[0]["last_error"] == (
        "compose claim-attempt cap reached; last refusal: " + collision
    )
