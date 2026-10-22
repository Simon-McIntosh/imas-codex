"""A detach that would strand its name is held instead of aborting the pass.

Executes the reconcile's own Cypher against an empty disposable Neo4j the test
plants its own fixtures in. The rejected attachment is a hardware-locus/device
mismatch: the name carries the locus ``coil_conductor`` and the source path
shares no device token with it, which is the reason the consistency guard
rejects the pair.
"""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest
from neo4j import GraphDatabase

from imas_codex.graph.client import GraphClient
from imas_codex.standard_names.attachment_audit import (
    _DETACH_QUERY,
    reconcile_attachment_consistency,
)
from imas_codex.standard_names.source_invariant import (
    MissingStandardNameSource,
    require_producing_sources,
)

#: A name whose locus token (``coil_conductor``) is a recognised hardware device.
_SN = "vertical_coordinate_of_coil_conductor"

#: Share no device token with the name's locus, so the guard rejects them.
_REJECTED_PATH = "ic_antennas/antenna/module/strap/geometry/arcs_of_circle/z"
_REJECTED_PATH_TWO = "summary/local/value"

#: Shares the ``coil`` device token, so the guard accepts it.
_ACCEPTED_PATH = "coil/geometry/arcs_of_circle/z"


@pytest.fixture(scope="module")
def disposable_neo4j() -> Iterator[tuple[str, str]]:
    uri = os.environ.get("IMAS_CODEX_TEST_NEO4J_URI")
    if not uri:
        pytest.skip("IMAS_CODEX_TEST_NEO4J_URI is not configured")
    if os.environ.get("IMAS_CODEX_TEST_NEO4J_EPHEMERAL") != "1":
        pytest.fail("stranding hold requires a disposable graph")
    password = os.environ.get("IMAS_CODEX_TEST_NEO4J_PASSWORD", "")
    auth = ("neo4j", password) if password else None
    print(f"GRAPH_ENDPOINT={uri}")
    with GraphDatabase.driver(uri, auth=auth) as driver:
        driver.verify_connectivity()
    yield uri, password


@pytest.fixture
def client(disposable_neo4j: tuple[str, str]) -> Iterator[GraphClient]:
    uri, password = disposable_neo4j
    graph = GraphClient(
        uri=uri,
        username="neo4j",
        password=password,
        graph_name="stranding-hold",
    )
    graph.query("MATCH (node) DETACH DELETE node")
    yield graph
    graph.query("MATCH (node) DETACH DELETE node")
    graph.close()


def _seed(
    client: GraphClient,
    sn_id: str,
    *,
    stage: str,
    source_paths: list[str],
) -> None:
    client.query(
        "CREATE (:StandardName {id: $sn_id, name_stage: $stage, "
        "status: 'draft', source_paths: []})",
        sn_id=sn_id,
        stage=stage,
    )
    for dd_path in source_paths:
        client.query(
            """
            MATCH (sn:StandardName {id: $sn_id})
            CREATE (src:StandardNameSource {
              id: $source_node_id, source_type: 'dd', source_id: $dd_path,
              dd_path: $dd_path, status: 'attached', produced_sn_id: $sn_id})
            CREATE (dd:IMASNode {id: $dd_path})
            CREATE (src)-[:FROM_DD_PATH]->(dd)
            CREATE (src)-[:PRODUCED_NAME]->(sn)
            CREATE (dd)-[:HAS_STANDARD_NAME]->(sn)
            SET sn.source_paths = sn.source_paths + ('dd:' + $dd_path)
            """,
            sn_id=sn_id,
            dd_path=dd_path,
            source_node_id=f"dd:{dd_path}",
        )


def _producer_count(client: GraphClient, sn_id: str) -> int:
    rows = client.query(
        "MATCH (:StandardNameSource)-[:PRODUCED_NAME]->"
        "(:StandardName {id: $sn_id}) RETURN count(*) AS n",
        sn_id=sn_id,
    )
    return int(rows[0]["n"])


@pytest.mark.graph
def test_sole_producer_rejected_attachment_is_held(client: GraphClient) -> None:
    _seed(client, _SN, stage="exhausted", source_paths=[_REJECTED_PATH])

    result = reconcile_attachment_consistency(client)

    assert result.skipped_stranding == 1
    assert result.detached == 0
    assert [v.sn_id for v in result.held_stranding] == [_SN]
    assert result.held_stranding[0].reason.startswith("locus/source device mismatch")
    summary = result.as_dict()
    assert summary["skipped_stranding"] == 1
    assert summary["held_stranding"] == 1
    assert summary["stranding"][0]["sn_id"] == _SN

    # The attachment is left exactly as found: edge, projection and mirror intact.
    assert _producer_count(client, _SN) == 1
    projections = client.query(
        "MATCH (:IMASNode {id: $dd_path})-[:HAS_STANDARD_NAME]->"
        "(:StandardName {id: $sn_id}) RETURN count(*) AS n",
        dd_path=_REJECTED_PATH,
        sn_id=_SN,
    )
    assert int(projections[0]["n"]) == 1


@pytest.mark.graph
def test_other_rejected_attachments_are_detached_in_the_same_pass(
    client: GraphClient,
) -> None:
    _seed(
        client,
        _SN,
        stage="exhausted",
        source_paths=[_REJECTED_PATH, _REJECTED_PATH_TWO, _ACCEPTED_PATH],
    )

    result = reconcile_attachment_consistency(client)

    assert result.skipped_stranding == 0
    assert result.detached == 2
    assert _producer_count(client, _SN) == 1


@pytest.mark.graph
def test_invariant_refuses_a_detach_forced_past_the_hold(client: GraphClient) -> None:
    _seed(client, _SN, stage="exhausted", source_paths=[_REJECTED_PATH])

    # Force exactly the detach the hold declines to perform.
    client.query(
        _DETACH_QUERY,
        items=[
            {
                "source_node_id": f"dd:{_REJECTED_PATH}",
                "dd_path": _REJECTED_PATH,
                "sn_id": _SN,
                "reroute": True,
            }
        ],
        historical=["superseded"],
    )

    assert _producer_count(client, _SN) == 0
    with pytest.raises(MissingStandardNameSource):
        require_producing_sources(client, {_SN})
