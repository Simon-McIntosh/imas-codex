"""A compose claim adopts an unowned pending projection stub of its own DD path.

A projection pass mints ``pending`` identities that carry a
``HAS_STANDARD_NAME`` edge from a DD node and no producing source. An exact
claim may adopt an unowned projection, while a pending reservation remains
private to its owner. A different composed identity can replace the projection
only after the source's successor binding and the stub's ownership are checked.

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
PROBE_STUB = "electron_temperature"
PROBE_DESCRIPTION = "composed description carried into the adopted projection stub"
PROBE_UNIT = "eV"
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

    The DD source id is ``dd:``-namespaced while the node ids are not, so the
    wipe matches anywhere in the id rather than at its start. The stub carries
    its own grammar-valid identity, which shares no part of that namespace, so
    it is named as well. The disposable graph's receipt ledger is reset too:
    receipt ids are hashed or random, and their endpoints can be stage names.
    """
    gc.query(
        "MATCH (n) WHERE n.id CONTAINS 'test_adopt_projection' "
        "OR n.id IN [$stub, 'ion_temperature'] "
        "OR n:StandardNameChange "
        "DETACH DELETE n",
        stub=PROBE_STUB,
    )


def _receipt_count(gc: GraphClient) -> int:
    return gc.query(
        "MATCH (receipt:StandardNameChange) RETURN count(receipt) AS count"
    )[0]["count"]


@pytest.fixture()
def graph(disposable_neo4j_uri: str):
    with GraphClient() as gc:
        _wipe(gc)
        assert _receipt_count(gc) == 0
        yield gc
        before_cleanup = _receipt_count(gc)
        _wipe(gc)
        after_cleanup = _receipt_count(gc)
        assert after_cleanup == 0
        print(
            f"fixture_receipts: before_cleanup={before_cleanup}; "
            f"after_cleanup={after_cleanup}"
        )


def _seed_stub(gc: GraphClient, *, owner_token: str | None = None) -> None:
    """Create a pending projection stub plus the extracted source for its path."""
    gc.query(
        """
        MERGE (node:IMASNode {id: $path})
        ON CREATE SET node.node_category = 'scalar',
                      node.lifecycle_status = 'active'
        MERGE (stub:StandardName {id: $stub})
        ON CREATE SET stub.name_stage = 'pending', stub.status = 'draft',
                      stub.origin = 'projection', stub.unit = $unit
        MERGE (node)-[:HAS_STANDARD_NAME]->(stub)
        """,
        path=PROBE_PATH,
        stub=PROBE_STUB,
        unit=PROBE_UNIT,
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
               stub.description AS description, stub.unit AS unit,
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


def _persist_candidate(**overrides):
    from imas_codex.standard_names.graph_store.graph_catalog import (
        persist_generated_name_batch,
    )

    return persist_generated_name_batch(
        [
            {
                "id": PROBE_STUB,
                "source_id": PROBE_PATH,
                "source_types": ["dd"],
                "description": PROBE_DESCRIPTION,
                "unit": PROBE_UNIT,
                "source_claim_token": TOKEN,
                "source_claim_seq": SEQ,
                **overrides,
            }
        ],
        compose_model="test-adopt-projection-model",
        return_winner_ids=True,
    )


@pytest.mark.timeout(60)
def test_two_path_projection_stub_is_adopted(graph, services):
    _seed_stub(graph)
    graph.query(
        "MATCH (stub:StandardName {id: $stub}) "
        "MERGE (other:IMASNode {id: $path}) "
        "MERGE (other)-[:HAS_STANDARD_NAME]->(stub)",
        stub=PROBE_STUB,
        path=PROBE_PATH + "_other",
    )
    assert _persist_candidate() == [PROBE_SOURCE]
    assert _stub(graph)["name_stage"] == "drafted"
    assert len(_adoption_receipts(graph)) == 1


def test_structural_parent_projection_stub_is_adopted(graph, services):
    _seed_stub(graph)
    graph.query(
        "MATCH (stub:StandardName {id: $stub}) "
        "MERGE (parent:StandardName {id: $parent}) "
        "SET parent.origin = 'derived', parent.name_stage = 'accepted' "
        "MERGE (stub)-[:HAS_PARENT]->(parent) "
        "MERGE (authority:StructuralNameAuthority {id: $authority}) "
        "MERGE (authority)-[:ENTAILED_FROM_CHILD]->(stub)",
        stub=PROBE_STUB,
        parent="test_adopt_projection_parent",
        authority="test_adopt_projection_authority",
    )
    assert _persist_candidate() == [PROBE_SOURCE]
    assert _stub(graph)["name_stage"] == "drafted"
    assert len(_adoption_receipts(graph)) == 1


def test_second_claim_after_unpark_finalizes(graph, services, monkeypatch):
    from imas_codex.standard_names.graph_store.graph_source_bindings import (
        claim_explicit_standard_name_sources,
        retry_failed_sources,
    )

    _seed_stub(graph)
    graph.query(
        "MATCH (src:StandardNameSource {id: $source}) "
        "SET src.status = 'failed', src.attempt_count = 5, "
        "src.claim_token = null, src.claimed_at = null",
        source=PROBE_SOURCE,
    )
    result = retry_failed_sources(
        [PROBE_PATH], reason="retry fixture source", gc=graph, services=services
    )
    assert result["retried"] == 1
    claimed = claim_explicit_standard_name_sources([PROBE_PATH], services=services)
    assert len(claimed) == 1
    claim = claimed[0]
    assert claim["claim_seq"] > SEQ
    write = services.write_standard_names
    observed = []

    def observe_claim(names, **kwargs):
        result = write(names, **kwargs)
        observed.extend(
            kwargs["gc"].query(
                "MATCH (src:StandardNameSource {id: $source}) "
                "RETURN src.status AS status, src.claim_token AS token, src.claim_seq AS seq",
                source=PROBE_SOURCE,
            )
        )
        return result

    monkeypatch.setattr(services, "write_standard_names", observe_claim)
    assert _persist_candidate(
        id="ion_temperature",
        source_claim_token=claim["claim_token"],
        source_claim_seq=claim["claim_seq"],
    ) == [PROBE_SOURCE]
    assert observed == [
        {
            "status": "extracted",
            "token": claim["claim_token"],
            "seq": claim["claim_seq"],
        }
    ]
    row = graph.query(
        "MATCH (src:StandardNameSource {id: $source}) "
        "RETURN src.status AS status, src.claim_token AS token",
        source=PROBE_SOURCE,
    )[0]
    assert row == {"status": "composed", "token": None}
    assert _stub(graph)["name_stage"] == "superseded"
    assert graph.query(
        "MATCH (src:StandardNameSource {id: $source})-[:PRODUCED_NAME]->(name) "
        "RETURN collect(name.id) AS names",
        source=PROBE_SOURCE,
    )[0]["names"] == ["ion_temperature"]
    assert graph.query(
        "MATCH (dd:IMASNode {id: $path})-[:HAS_STANDARD_NAME]->(name) "
        "RETURN collect(name.id) AS names",
        path=PROBE_PATH,
    )[0]["names"] == ["ion_temperature"]
    assert (
        graph.query(
            "MATCH (name:StandardName {id: 'ion_temperature'})-[:HAS_INTERNAL_CHANGE]->(event) "
            "WHERE event.operation = 'source_migration_manifest' "
            "RETURN count(event) AS count"
        )[0]["count"]
        == 1
    )
    from imas_codex.standard_names.provenance_lifecycle import (
        retarget_standard_name_sources,
    )

    assert (
        retarget_standard_name_sources(
            graph,
            PROBE_STUB,
            "ion_temperature",
            operation="regenerate",
            source_ids=[PROBE_SOURCE],
            expected_current_bindings={PROBE_SOURCE: "ion_temperature"},
        )
        == 0
    )


def _seed_finalized_successor(graph):
    _seed_stub(graph)
    graph.query(
        "MATCH (src:StandardNameSource {id: $source})-[:FROM_DD_PATH]->(dd) "
        "CREATE (name:StandardName {id: 'ion_temperature', name_stage: 'drafted', "
        "status: 'draft', unit: $unit}) "
        "SET src.status = 'composed', src.claim_token = null, "
        "src.claimed_at = null, src.produced_sn_id = name.id "
        "CREATE (src)-[:PRODUCED_NAME]->(name) "
        "CREATE (dd)-[:HAS_STANDARD_NAME]->(name)",
        source=PROBE_SOURCE,
        unit=PROBE_UNIT,
    )


@pytest.mark.parametrize("stage", ["drafted", "reviewed", "accepted", "pending"])
@pytest.mark.parametrize("entrypoint", ["supersession", "retarget"])
def test_non_stub_predecessor_is_still_refused(graph, services, stage, entrypoint):
    from imas_codex.standard_names.graph_store.graph_supersession import (
        supersede_prior_source_names,
    )
    from imas_codex.standard_names.provenance_lifecycle import (
        retarget_standard_name_sources,
    )

    _seed_finalized_successor(graph)
    graph.query(
        "MATCH (stub:StandardName {id: $stub}) SET stub.name_stage = $stage",
        stub=PROBE_STUB,
        stage=stage,
    )
    if stage == "pending":
        graph.query(
            "MATCH (stub:StandardName {id: $stub}) "
            "CREATE (owner:StandardNameSource {id: $owner, status: 'composed'}) "
            "CREATE (owner)-[:PRODUCED_NAME]->(stub)",
            stub=PROBE_STUB,
            owner=OWNER_SOURCE,
        )
    refusal = (
        "source migration compare-and-set"
        if entrypoint == "supersession"
        else "heterogeneous|unowned pending projection-only predecessor"
    )
    with pytest.raises((ValueError, RuntimeError), match=refusal) as rejected:
        if entrypoint == "supersession":
            supersede_prior_source_names(
                [{"new_name": "ion_temperature", "source_id": PROBE_PATH}], gc=graph
            )
        else:
            retarget_standard_name_sources(
                graph,
                PROBE_STUB,
                "ion_temperature",
                operation="regenerate",
                source_ids=[PROBE_SOURCE],
                expected_current_bindings={PROBE_SOURCE: "ion_temperature"},
            )
    print(f"{entrypoint}/{stage} refused: {rejected.value}")
    assert _stub(graph)["name_stage"] == stage
    assert _stub(graph)["receipts"] == 0
    assert _producers(graph) == ([OWNER_SOURCE] if stage == "pending" else [])


def test_rich_write_then_adoption_keeps_the_ownership_guard(graph, services):
    from imas_codex.standard_names.graph_store.graph_source_bindings import (
        stage_claimed_generated_candidates,
    )

    _seed_stub(graph)
    assert services.write_standard_names(
        [
            {
                "id": PROBE_STUB,
                "source_types": ["dd"],
                "source_id": PROBE_PATH,
                "unit": PROBE_UNIT,
                "description": PROBE_DESCRIPTION,
            }
        ],
        gc=graph,
        return_written_ids=True,
    ) == [PROBE_STUB]
    after_write = graph.query(
        "MATCH (src:StandardNameSource {id: $source}) "
        "RETURN src.status AS status, src.claim_token AS token, src.claim_seq AS seq",
        source=PROBE_SOURCE,
    )[0]
    assert after_write["token"] == TOKEN
    assert after_write["seq"] == SEQ
    assert _producers(graph) == [PROBE_SOURCE]
    winners = stage_claimed_generated_candidates([_candidate()], services=services)
    print(
        f"rich_write_status={after_write['status']}; producers={[PROBE_SOURCE]}; adoption_winners={winners}"
    )
    assert winners == []
    assert _stub(graph)["name_stage"] == "pending"
    assert _adoption_receipts(graph) == []


def test_stolen_claim_aborts_transaction(graph, services, monkeypatch):
    _seed_stub(graph)
    write = services.write_standard_names

    def steal_claim(names, **kwargs):
        result = write(names, **kwargs)
        kwargs["gc"].query(
            "MATCH (src:StandardNameSource {id: $source}) "
            "SET src.claim_token = 'another-owner', src.claim_seq = src.claim_seq + 1",
            source=PROBE_SOURCE,
        )
        return result

    monkeypatch.setattr(services, "write_standard_names", steal_claim)
    with pytest.raises(RuntimeError, match="source claim changed"):
        _persist_candidate()
    assert _stub(graph)["name_stage"] == "pending"
    assert _producers(graph) == []
    assert _adoption_receipts(graph) == []


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
                "description": PROBE_DESCRIPTION,
                "unit": PROBE_UNIT,
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
    assert stub["description"] == PROBE_DESCRIPTION, (
        "the adopted stub did not take the composed description: " + str(stub)
    )
    assert stub["unit"] == PROBE_UNIT
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
