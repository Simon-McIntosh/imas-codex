"""Transactional coverage for signed provenance-orphan retirement."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
from neo4j import GraphDatabase

from imas_codex.graph.client import GraphClient
from imas_codex.standard_names.graph_ops import retire_signed_provenance_orphans

_AUTHORITY_PATH = (
    Path(__file__).parents[2]
    / "docs/evidence/sn-graph-wide-integrity/refused-target-orphan-adjudication.json"
)
_RETIRE_DISPOSITION = "retire_under_orphan_policy"


def _canonical_hash(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _authority() -> tuple[dict[str, object], str, list[dict[str, object]]]:
    authority = json.loads(_AUTHORITY_PATH.read_text())
    retirements = [
        row for row in authority["rows"] if row["disposition"] == _RETIRE_DISPOSITION
    ]
    assert len(retirements) == 16
    return authority, _canonical_hash(authority), retirements


@pytest.fixture(scope="module")
def disposable_neo4j() -> Iterator[tuple[str, str]]:
    uri = os.environ.get("IMAS_CODEX_TEST_NEO4J_URI")
    if not uri:
        pytest.skip("IMAS_CODEX_TEST_NEO4J_URI is not configured")
    if os.environ.get("IMAS_CODEX_TEST_NEO4J_EPHEMERAL") != "1":
        pytest.fail("signed orphan retirement requires a disposable graph")
    password = os.environ.get("IMAS_CODEX_TEST_NEO4J_PASSWORD", "")
    auth = ("neo4j", password) if password else None
    print(f"GRAPH_ENDPOINT={uri}")
    with GraphDatabase.driver(uri, auth=auth) as driver:
        driver.verify_connectivity()
        driver.execute_query("MATCH (node) DETACH DELETE node")
    yield uri, password


def _client(endpoint: tuple[str, str]) -> GraphClient:
    uri, password = endpoint
    return GraphClient(
        uri=uri,
        username="neo4j",
        password=password,
        graph_name="signed-orphan-retirement",
    )


def _seed_signed_targets(
    client: GraphClient, retirements: list[dict[str, object]]
) -> None:
    client.query(
        """
        UNWIND $rows AS row
        CREATE (:StandardName {
          id: row.name,
          name_stage: row.name_stage,
          status: 'draft',
          origin: row.origin,
          validation_status: row.validation_status
        })
        """,
        rows=retirements,
    )


def _cleanup(client: GraphClient, retirements: list[dict[str, object]]) -> None:
    names = [row["name"] for row in retirements]
    client.query(
        """
        MATCH (node)
        WHERE node.id IN $name_ids
           OR node.from_name IN $name_ids
           OR node.test_cohort = 'signed-orphan-retirement'
        DETACH DELETE node
        """,
        name_ids=names,
    )


def _snapshot(client: GraphClient) -> bytes:
    nodes = client.query(
        """
        MATCH (node)
        RETURN elementId(node) AS element_id, labels(node) AS labels,
               properties(node) AS properties
        ORDER BY element_id
        """
    )
    relationships = client.query(
        """
        MATCH (start)-[relationship]->(end)
        RETURN elementId(relationship) AS element_id,
               type(relationship) AS type,
               properties(relationship) AS properties,
               elementId(start) AS start_id,
               elementId(end) AS end_id
        ORDER BY element_id
        """
    )
    return json.dumps(
        {"nodes": nodes, "relationships": relationships},
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode()


@pytest.mark.graph
def test_signed_retirements_are_ledgered_once_and_replay_without_writes(
    disposable_neo4j: tuple[str, str],
) -> None:
    authority, authority_hash, retirements = _authority()
    client = _client(disposable_neo4j)
    _seed_signed_targets(client, retirements)
    reason = "signed identities have no remaining producing or structural authority"
    try:
        preview = retire_signed_provenance_orphans(
            authority,
            authority_sha256=authority_hash,
            reason=reason,
            gc=client,
        )
        assert preview["outcome"] == "would_apply"
        assert preview["counts"] == {
            "requested": 16,
            "admitted": 16,
            "refused": 0,
        }

        applied = retire_signed_provenance_orphans(
            authority,
            authority_sha256=authority_hash,
            reason=reason,
            apply=True,
            manifest_sha256=preview["manifest_sha256"],
            gc=client,
        )
        assert applied["outcome"] == "applied"
        assert applied["changed"] == 16
        assert applied["superseded"] == 16
        assert applied["ledger_rows"] == 16
        result = client.query(
            """
            MATCH (name:StandardName)
            WHERE name.id IN $name_ids
            OPTIONAL MATCH (name)-[:HAS_INTERNAL_CHANGE]->
              (change:StandardNameChange {
                operation: 'retire_signed_provenance_orphan'
              })
            RETURN count(name) AS names,
                   count(CASE WHEN name.name_stage = 'superseded'
                                   AND name.status = 'superseded'
                              THEN 1 END) AS superseded,
                   count(change) AS ledger_rows,
                   count(DISTINCT change.id) AS distinct_ledger_rows
            """,
            name_ids=[row["name"] for row in retirements],
        )
        assert result == [
            {
                "names": 16,
                "superseded": 16,
                "ledger_rows": 16,
                "distinct_ledger_rows": 16,
            }
        ]

        before = _snapshot(client)
        replay = retire_signed_provenance_orphans(
            authority,
            authority_sha256=authority_hash,
            reason=reason,
            apply=True,
            manifest_sha256=preview["manifest_sha256"],
            gc=client,
        )
        after = _snapshot(client)
        assert replay["outcome"] == "already_applied"
        assert replay["changed"] == 0
        assert replay["persistent_writes"] == 0
        assert after == before
    finally:
        _cleanup(client, retirements)


@pytest.mark.graph
def test_target_outside_signed_retirements_is_refused(
    disposable_neo4j: tuple[str, str],
) -> None:
    authority, authority_hash, retirements = _authority()
    client = _client(disposable_neo4j)
    try:
        receipt = retire_signed_provenance_orphans(
            authority,
            authority_sha256=authority_hash,
            name_ids=[*[row["name"] for row in retirements], "unsigned_identity"],
            reason="only the exact signed retirement cohort is authorized",
            gc=client,
        )
        assert receipt["outcome"] == "refused"
        assert receipt["changed"] == 0
        assert receipt["refusals"] == [
            {
                "name_id": "unsigned_identity",
                "reason": "target is outside signed retirement authority",
            }
        ]
    finally:
        _cleanup(client, retirements)


@pytest.mark.graph
@pytest.mark.parametrize("new_authority", ["producing_source", "structural_child"])
def test_new_live_authority_refuses_entire_signed_cohort(
    disposable_neo4j: tuple[str, str],
    new_authority: str,
) -> None:
    authority, authority_hash, retirements = _authority()
    client = _client(disposable_neo4j)
    _seed_signed_targets(client, retirements)
    target_id = str(retirements[0]["name"])
    add_authority: dict[str, Callable[[], object]] = {
        "producing_source": lambda: client.query(
            """
            MATCH (target:StandardName {id: $target_id})
            CREATE (source:StandardNameSource {
              id: 'test:signed-orphan-retirement:producer',
              status: 'composed',
              test_cohort: 'signed-orphan-retirement'
            })-[:PRODUCED_NAME]->(target)
            """,
            target_id=target_id,
        ),
        "structural_child": lambda: client.query(
            """
            MATCH (target:StandardName {id: $target_id})
            CREATE (child:StandardName {
              id: 'test_signed_orphan_retirement_child',
              name_stage: 'accepted',
              status: 'draft',
              test_cohort: 'signed-orphan-retirement'
            })-[:HAS_PARENT]->(target)
            """,
            target_id=target_id,
        ),
    }
    add_authority[new_authority]()
    try:
        preview = retire_signed_provenance_orphans(
            authority,
            authority_sha256=authority_hash,
            reason="new live authority blocks signed retirement",
            gc=client,
        )
        assert preview["outcome"] == "refused"
        assert preview["changed"] == 0
        assert preview["counts"] == {
            "requested": 16,
            "admitted": 15,
            "refused": 1,
        }
        expected_reason = {
            "producing_source": "name has a live producing source",
            "structural_child": "name has a live HAS_PARENT child",
        }[new_authority]
        assert preview["refusals"] == [
            {"name_id": target_id, "reason": expected_reason}
        ]
        assert client.query(
            """
            MATCH (name:StandardName)
            WHERE name.id IN $name_ids
            RETURN count(CASE WHEN name.name_stage = 'superseded'
                              THEN 1 END) AS superseded,
                   COUNT {
                     (change:StandardNameChange {
                       operation: 'retire_signed_provenance_orphan'
                     }) WHERE change.from_name IN $name_ids
                   } AS ledger_rows
            """,
            name_ids=[row["name"] for row in retirements],
        ) == [{"superseded": 0, "ledger_rows": 0}]
    finally:
        _cleanup(client, retirements)
