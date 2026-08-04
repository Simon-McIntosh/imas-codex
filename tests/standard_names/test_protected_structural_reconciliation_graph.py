"""Disposable-Neo4j contracts for protected structural reconciliation."""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest
from imas_standard_names import get_grammar_context, validate_round_trip
from neo4j import GraphDatabase

from imas_codex.graph.client import GraphClient
from imas_codex.settings import get_graph_uri
from imas_codex.standard_names import protected_structural_reconciliation as sut
from imas_codex.standard_names.grammar_segment_reconciliation import (
    _read_protected_source_sets,
    _west_source_ids,
)

pytestmark = pytest.mark.graph

_NEGATIVE_FIXTURE_LABELS = [
    {"path": "fixture/dd/average_field", "label": None},
    {"path": "fixture/dd/electron_temperature", "label": None},
]
_NEGATIVE_FIXTURE_LABELS_HASH = sut._negative_fixture_labels_hash(
    _NEGATIVE_FIXTURE_LABELS
)


@dataclass(frozen=True)
class _EphemeralNeo4j:
    uri: str

    def driver(self):
        return GraphDatabase.driver(self.uri, auth=None)

    def client(self) -> GraphClient:
        return GraphClient(
            uri=self.uri,
            username="neo4j",
            password="",
            graph_name="ephemeral-protected-structural",
        )


@pytest.fixture(scope="module")
def ephemeral_neo4j() -> Iterator[_EphemeralNeo4j]:
    uri = os.environ.get("IMAS_CODEX_TEST_NEO4J_URI")
    if not uri:
        pytest.skip("IMAS_CODEX_TEST_NEO4J_URI is not configured")
    if os.environ.get("IMAS_CODEX_TEST_NEO4J_EPHEMERAL") != "1":
        pytest.fail("protected structural tests require an ephemeral graph")
    project_uri = os.environ.get("IMAS_CODEX_TEST_PROJECT_NEO4J_URI") or get_graph_uri()
    if uri == project_uri:
        pytest.fail("protected structural tests refuse the configured project graph")
    driver = GraphDatabase.driver(uri, auth=None)
    driver.verify_connectivity()
    try:
        yield _EphemeralNeo4j(uri)
    finally:
        driver.close()


@pytest.fixture
def graph_client(ephemeral_neo4j: _EphemeralNeo4j) -> Iterator[GraphClient]:
    client = ephemeral_neo4j.client()
    client.query("MATCH (node) DETACH DELETE node")
    try:
        yield client
    finally:
        client.query("MATCH (node) DETACH DELETE node")
        assert client.query("MATCH (node) RETURN count(node) AS count") == [
            {"count": 0}
        ]
        client.close()


class _CountingTransaction:
    def __init__(self, transaction: Any, *, fail_mutation: bool) -> None:
        self._transaction = transaction
        self._fail_mutation = fail_mutation
        self.query_count = 0

    def run(self, cypher: str, **params: Any):
        self.query_count += 1
        result = self._transaction.run(cypher, **params)
        if self._fail_mutation and "PROTECTED_STRUCTURAL_RETIRE_APPLY" in cypher:
            list(result)
            raise RuntimeError("injected retirement failure")
        return result

    def commit(self) -> None:
        self._transaction.commit()

    def rollback(self) -> None:
        self._transaction.rollback()


class _CountingSession:
    def __init__(self, session: Any, owner: _CountingGraphClient) -> None:
        self._session = session
        self._owner = owner

    def begin_transaction(self) -> _CountingTransaction:
        transaction = _CountingTransaction(
            self._session.begin_transaction(), fail_mutation=self._owner.fail_mutation
        )
        self._owner.transactions.append(transaction)
        return transaction


class _CountingGraphClient:
    def __init__(self, client: GraphClient, *, fail_mutation: bool = False) -> None:
        self.client = client
        self.fail_mutation = fail_mutation
        self.transactions: list[_CountingTransaction] = []

    @contextmanager
    def session(self) -> Iterator[_CountingSession]:
        with self.client.session() as session:
            yield _CountingSession(session, self)

    @property
    def query_count(self) -> int:
        return sum(transaction.query_count for transaction in self.transactions)


def _target_names(count: int) -> list[str]:
    vocabularies = get_grammar_context()["grammar"]["vocabularies"]
    candidates = sorted(vocabularies["physical_bases"])
    valid = [name for name in candidates if validate_round_trip(name)]
    assert len(valid) >= count
    return valid[:count]


def _seed(client: GraphClient, action: str, *, count: int) -> list[dict[str, str]]:
    namespace = uuid4().hex
    targets = _target_names(count)
    rows = [
        {
            "old_id": f"invalid_identity_{namespace}_{index}",
            "target_id": target,
            "source_id": f"dd:test_review_entry__{namespace}_{index}",
            "backing_id": f"test/{namespace}/{index}/{target}",
            "downstream_label": "psi_like" if index % 2 == 0 else "ip_like",
        }
        for index, target in enumerate(targets)
    ]
    client.query(
        """
        CREATE (version:DDVersion {id: '4.1.1', is_current: true})
        CREATE (cocos:COCOS {id: 17})
        CREATE (version)-[:HAS_COCOS]->(cocos)
        CREATE (unit:Unit {id: 'm^2', symbol: 'm^2'})
        WITH unit
        UNWIND $negative_fixture_labels AS fixture
        CREATE (:IMASNode {
          id: fixture.path, cocos_transformation_type: fixture.label
        })
        WITH DISTINCT unit
        UNWIND $rows AS row
        CREATE (old:StandardName {
          id: row.old_id, name_stage: $old_stage,
          validation_status: 'quarantined', unit: 'm^2', cocos: 17,
          source_paths: [row.backing_id], claim_token: null, claimed_at: null
        })
        CREATE (target:StandardName {
          id: row.target_id, name_stage: 'accepted', validation_status: 'valid',
          unit: 'm^2', source_paths: [],
          claim_token: null, claimed_at: null
        })
        CREATE (source:StandardNameSource {
          id: row.source_id, source_type: 'dd', source_id: row.backing_id,
          status: $source_status, produced_sn_id: row.old_id, dd_version: '4.1.0',
          claim_token: null, claimed_at: null
        })
        CREATE (backing:IMASNode {
          id: row.backing_id, unit: 'm^2', node_category: 'quantity',
          standard_name_id: row.old_id,
          cocos_transformation_type: row.downstream_label
        })
        CREATE (old)-[:HAS_UNIT]->(unit)
        CREATE (target)-[:HAS_UNIT]->(unit)
        CREATE (backing)-[:HAS_UNIT]->(unit)
        CREATE (source)-[:FROM_DD_PATH]->(backing)
        CREATE (source)-[:PRODUCED_NAME]->(old)
        CREATE (backing)-[:HAS_STANDARD_NAME]->(old)
        """,
        rows=rows,
        old_stage=("reviewed" if action == sut.PROTECTED_IDENTITY_FOLD else "pending"),
        source_status=(
            "composed" if action == sut.PROTECTED_IDENTITY_FOLD else "stale"
        ),
        negative_fixture_labels=_NEGATIVE_FIXTURE_LABELS,
    )
    west_source_id = sorted(_west_source_ids())[0]
    client.query(
        """
        MATCH (target:StandardName {id: $target_id}), (unit:Unit {id: 'm^2'})
        CREATE (source:StandardNameSource {
          id: $source_id, source_type: 'dd', source_id: $backing_id,
          status: 'attached', produced_sn_id: target.id
        })
        CREATE (backing:IMASNode {
          id: $backing_id, unit: 'm^2', node_category: 'quantity',
          standard_name_id: target.id, cocos_transformation_type: 'psi_like'
        })
        CREATE (source)-[:FROM_DD_PATH]->(backing)
        CREATE (source)-[:PRODUCED_NAME]->(target)
        CREATE (backing)-[:HAS_STANDARD_NAME]->(target)
        CREATE (backing)-[:HAS_UNIT]->(unit)
        """,
        target_id=rows[0]["target_id"],
        source_id=west_source_id,
        backing_id=f"west-preserved/{namespace}",
    )
    return rows


def _snapshots_and_protection(
    client: GraphClient, rows: list[dict[str, str]]
) -> tuple[dict[str, dict[str, Any]], Any]:
    with client.session() as session:
        transaction = session.begin_transaction()
        try:
            snapshots = {}
            for raw in transaction.run(
                sut.PROTECTED_STRUCTURAL_SNAPSHOT_QUERY,
                pairs=[
                    {"old_id": row["old_id"], "target_id": row["target_id"]}
                    for row in rows
                ],
                live_stages=sorted(sut.identity_fold._FOLD_LIVE_STAGES),
            ):
                snapshot = sut._canonical(dict(raw))
                snapshot["_cas_signature"] = sut.identity_fold._fold_cas_signature(
                    dict(raw)
                )
                snapshots[snapshot["old_properties"]["id"]] = snapshot
            protected = _read_protected_source_sets(transaction)
            transaction.rollback()
        except BaseException:
            transaction.rollback()
            raise
    return snapshots, protected


def _authority(path: Path, rows: list[dict[str, Any]], *, disposition: str) -> str:
    evidence = {
        "authority_verdict": {
            "semantic_decision_remaining": False,
            "user_decision_remaining": False,
            "confidence": 0.97,
            "verdict": "equivalent_under_current_catalog",
        },
        "mutation_authorized": True,
        "final_disposition": disposition,
        "mutation_scopes": [
            {
                "operation": row["action"],
                "old_id": row["old_id"],
                "target_id": row["target_id"],
                "source_ids": row["source_ids"],
                "dd_version": "4.1.1",
                "cocos": 17,
                "mutation_authorized": True,
                "final_disposition": disposition,
            }
            for row in rows
        ],
        "cocos_contract": {
            "catalog_check_passed": True,
            "catalog_constant": 17,
            "change_made": False,
        },
        "negative_fixture_label_contract": {
            "dd_version": "4.1.1",
            "fixture_labels": _NEGATIVE_FIXTURE_LABELS,
            "fixture_labels_hash": _NEGATIVE_FIXTURE_LABELS_HASH,
        },
        "graph_evidence": {
            "raw_evidence": {
                "catalogs": [{"id": "4.1.1", "is_current": True, "cocos": 17}]
            }
        },
    }
    path.write_text(json.dumps(evidence, sort_keys=True))
    return sha256(path.read_bytes()).hexdigest()


def _manifest(
    tmp_path: Path,
    client: GraphClient,
    seeded: list[dict[str, str]],
    action: str,
) -> tuple[Path, str, list[dict[str, Any]]]:
    snapshots, protected = _snapshots_and_protection(client, seeded)
    evidence_path = tmp_path / f"{action}-authority.json"
    evidence_hash = "0" * 64
    provisional = [
        {
            "action": action,
            "old_id": item["old_id"],
            "target_id": item["target_id"],
            "source_ids": [item["source_id"]],
        }
        for item in seeded
    ]
    if action == sut.PROTECTED_IDENTITY_FOLD:
        evidence_hash = _authority(
            evidence_path,
            provisional,
            disposition=f"authorized_for_{action}",
        )
    rows = [
        sut.build_manifest_row(
            snapshots[item["old_id"]],
            action=action,
            protected=protected,
            reason="exact disposable graph repair",
            authority_evidence_sha256=(
                evidence_hash if action == sut.PROTECTED_IDENTITY_FOLD else None
            ),
            event_timestamp="2026-08-04T12:00:00+02:00",
            negative_fixture_labels=_NEGATIVE_FIXTURE_LABELS,
        )
        for item in seeded
    ]
    payload = sut.build_manifest_payload(
        rows,
        protected_set_hash=protected.protected_set_hash,
        authority_evidence_sha256=evidence_hash,
        authority_evidence_path=(
            evidence_path if action == sut.PROTECTED_IDENTITY_FOLD else None
        ),
        negative_fixture_labels=_NEGATIVE_FIXTURE_LABELS,
    )
    path = tmp_path / f"{action}-manifest.json"
    path.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
    return path, sha256(path.read_bytes()).hexdigest(), rows


def _event_ids(row: dict[str, Any]) -> list[str]:
    if row["action"] == sut.PROTECTED_IDENTITY_FOLD:
        return ["sn-change:protected-fold:" + sut._event_identity(row, "fold")]
    return [
        "source-authority-retirement:" + sut._event_identity(row, "source"),
        "sn-change:protected-retirement:" + sut._event_identity(row, "name"),
    ]


def _protected_identity(client: GraphClient, target_id: str) -> dict[str, Any]:
    rows = client.query(
        """
        MATCH (target:StandardName {id: $target_id})-[target_unit:HAS_UNIT]
              ->(unit:Unit)
        MATCH (west:StandardNameSource)-[binding:PRODUCED_NAME]->(target)
        WHERE west.id STARTS WITH 'dd:'
          AND NOT west.id STARTS WITH 'dd:test_review_entry__'
        MATCH (west)-[ownership:FROM_DD_PATH]->(backing:IMASNode)
        MATCH (backing)-[projection:HAS_STANDARD_NAME]->(target)
        MATCH (backing)-[backing_unit:HAS_UNIT]->(unit)
        RETURN elementId(target) AS target,
               properties(target) AS target_properties,
               elementId(west) AS west_source,
               properties(west) AS west_source_properties,
               elementId(backing) AS west_backing,
               properties(backing) AS west_backing_properties,
               elementId(unit) AS unit,
               elementId(binding) AS binding,
               elementId(ownership) AS ownership,
               elementId(projection) AS projection,
               elementId(target_unit) AS target_unit,
               elementId(backing_unit) AS backing_unit
        """,
        target_id=target_id,
    )
    assert len(rows) == 1
    identity = rows[0]
    identity["target_properties"].pop("source_paths", None)
    return identity


def _transactional_after_state(client: GraphClient, path: Path) -> dict[str, Any]:
    manifest = sut.load_protected_structural_manifest(path)
    with client.session() as session:
        transaction = session.begin_transaction()
        try:
            row = manifest.rows[0]
            snapshot = sut._read_snapshots(transaction, manifest)[row["row_key"]]
            mutation = (
                sut._fold_item(row, snapshot)
                if manifest.action == sut.PROTECTED_IDENTITY_FOLD
                else sut._retirement_item(row, snapshot)
            )
            query = (
                sut.FOLD_APPLY_QUERY
                if manifest.action == sut.PROTECTED_IDENTITY_FOLD
                else sut.RETIRE_APPLY_QUERY
            )
            list(transaction.run(query, items=[mutation]))
            if manifest.action == sut.PROTECTED_IDENTITY_FOLD:
                snapshot = sut._read_snapshots(transaction, manifest)[
                    manifest.rows[0]["row_key"]
                ]
                state = sut.identity_fold._fold_verification_state(
                    snapshot, fold_change_id=_event_ids(manifest.rows[0])[0]
                )
            else:
                state = sut._retirement_state_semantics(
                    sut._read_retirement_states(transaction, manifest)[
                        manifest.rows[0]["row_key"]
                    ]
                )
            transaction.rollback()
            return sut._bind_expected_after_contract(state, manifest.rows[0])
        except BaseException:
            transaction.rollback()
            raise


def _state_differences(actual: Any, expected: Any, path: str = "$config") -> list[str]:
    if type(actual) is not type(expected):
        return [f"{path}: {actual!r} != {expected!r}"]
    if isinstance(actual, dict):
        differences = []
        for key in sorted(set(actual) | set(expected)):
            if key not in actual or key not in expected:
                differences.append(
                    f"{path}.{key}: {actual.get(key)!r} != {expected.get(key)!r}"
                )
            else:
                differences.extend(
                    _state_differences(actual[key], expected[key], f"{path}.{key}")
                )
        return differences
    if isinstance(actual, list):
        differences = []
        if len(actual) != len(expected):
            differences.append(f"{path}.length: {len(actual)} != {len(expected)}")
        for index, (left, right) in enumerate(zip(actual, expected, strict=False)):
            differences.extend(_state_differences(left, right, f"{path}[{index}]"))
        return differences
    return [] if actual == expected else [f"{path}: {actual!r} != {expected!r}"]


def test_fold_dry_apply_second_apply_and_event_drift(
    tmp_path: Path, graph_client: GraphClient
) -> None:
    seeded = _seed(graph_client, sut.PROTECTED_IDENTITY_FOLD, count=1)
    path, digest, rows = _manifest(
        tmp_path, graph_client, seeded, sut.PROTECTED_IDENTITY_FOLD
    )
    dry_counter = _CountingGraphClient(graph_client)
    dry = sut.reconcile_protected_structure(path, gc=dry_counter)
    assert dry["mode"] == "dry_run"
    assert dry_counter.query_count == dry["query_audit"]["query_count"] == 2
    assert graph_client.query(
        "MATCH (event:StandardNameChange) RETURN count(event) AS count"
    ) == [{"count": 0}]
    actual_after = _transactional_after_state(graph_client, path)
    assert actual_after == rows[0]["expected_after"], _state_differences(
        actual_after, rows[0]["expected_after"]
    )

    protected_identity_before = _protected_identity(
        graph_client, seeded[0]["target_id"]
    )
    apply_counter = _CountingGraphClient(graph_client)
    receipt = sut.reconcile_protected_structure(
        path, apply=True, expected_manifest_hash=digest, gc=apply_counter
    )
    assert receipt["mode"] == "applied"
    assert apply_counter.query_count == receipt["query_audit"]["query_count"] == 9
    census = sut.census_protected_structural_release(
        path,
        receipt,
        expected_receipt_hash=receipt["receipt_hash"],
        gc=graph_client,
    )
    assert census["release_ready"] is True
    assert (
        _protected_identity(graph_client, seeded[0]["target_id"])
        == protected_identity_before
    )
    preserved = graph_client.query(
        """
        MATCH (target:StandardName {id: $target})-[:HAS_UNIT]->(unit:Unit)
        MATCH (west:StandardNameSource)-[:PRODUCED_NAME]->(target)
        WHERE west.id STARTS WITH 'dd:' AND NOT west.id STARTS WITH 'dd:test_review_entry__'
        MATCH (west)-[:FROM_DD_PATH]->(backing:IMASNode)
        MATCH (source:StandardNameSource {id: $source})
        RETURN target.cocos AS cocos, unit.id AS unit,
               backing.cocos_transformation_type AS label,
               backing.standard_name_id AS mirror,
               source.dd_version AS source_dd_version
        """,
        target=seeded[0]["target_id"],
        source=seeded[0]["source_id"],
    )
    assert preserved == [
        {
            "cocos": None,
            "unit": "m^2",
            "label": "psi_like",
            "mirror": seeded[0]["target_id"],
            "source_dd_version": "4.1.0",
        }
    ]

    repeated = sut.reconcile_protected_structure(
        path, apply=True, expected_manifest_hash=digest, gc=graph_client
    )
    assert repeated["mode"] == "already_current"
    assert graph_client.query(
        "MATCH (source:StandardNameSource {id: $source}) "
        "RETURN source.dd_version AS dd_version",
        source=seeded[0]["source_id"],
    ) == [{"dd_version": "4.1.0"}]
    event_id = _event_ids(rows[0])[0]
    graph_client.query(
        "MATCH (event {id: $id}) SET event.origin = 'tampered'", id=event_id
    )
    refused = sut.reconcile_protected_structure(path, gc=graph_client)
    assert refused["counts"]["refused"] == 1


def test_retirement_rollback_exact_postflight_and_second_apply(
    tmp_path: Path, graph_client: GraphClient
) -> None:
    seeded = _seed(graph_client, sut.RETIRE_STALE_SOURCE_BRANCH, count=1)
    path, digest, rows = _manifest(
        tmp_path, graph_client, seeded, sut.RETIRE_STALE_SOURCE_BRANCH
    )
    actual_after = _transactional_after_state(graph_client, path)
    assert actual_after == rows[0]["expected_after"], _state_differences(
        actual_after, rows[0]["expected_after"]
    )
    dry = sut.reconcile_protected_structure(path, gc=graph_client)
    assert dry["counts"]["planned"] == 1, dry["rows"][0]["unresolved"]
    protected_identity_before = _protected_identity(
        graph_client, seeded[0]["target_id"]
    )
    failing = _CountingGraphClient(graph_client, fail_mutation=True)
    with pytest.raises(RuntimeError, match="injected retirement failure"):
        sut.reconcile_protected_structure(
            path, apply=True, expected_manifest_hash=digest, gc=failing
        )
    assert graph_client.query(
        "MATCH (old:StandardName {id: $id}) RETURN count(old) AS count",
        id=seeded[0]["old_id"],
    ) == [{"count": 1}]
    assert graph_client.query(
        "MATCH (event) WHERE event.id IN $ids RETURN count(event) AS count",
        ids=_event_ids(rows[0]),
    ) == [{"count": 0}]

    receipt = sut.reconcile_protected_structure(
        path, apply=True, expected_manifest_hash=digest, gc=graph_client
    )
    assert receipt["mode"] == "applied"
    assert (
        _protected_identity(graph_client, seeded[0]["target_id"])
        == protected_identity_before
    )
    census = sut.census_protected_structural_release(
        path,
        receipt,
        expected_receipt_hash=receipt["receipt_hash"],
        gc=graph_client,
    )
    assert census["release_ready"] is True
    exact = graph_client.query(
        """
        MATCH (source:StandardNameSource {id: $source})
              -[:FROM_DD_PATH]->(backing:IMASNode {id: $backing})
        OPTIONAL MATCH (source)-[binding:PRODUCED_NAME]->()
        OPTIONAL MATCH (backing)-[projection:HAS_STANDARD_NAME]->()
        OPTIONAL MATCH (mirror)
        WHERE mirror.produced_sn_id = $old OR mirror.standard_name_id = $old
        RETURN source.status AS status, source.produced_sn_id AS source_mirror,
               source.skip_reason AS reason,
               backing.standard_name_id AS backing_mirror,
               backing.cocos_transformation_type AS label,
               count(DISTINCT binding) AS bindings,
               count(DISTINCT projection) AS projections,
               count(DISTINCT mirror) AS old_mirrors
        """,
        source=seeded[0]["source_id"],
        backing=seeded[0]["backing_id"],
        old=seeded[0]["old_id"],
    )
    assert exact == [
        {
            "status": "stale",
            "source_mirror": None,
            "reason": "stale_source_branch",
            "backing_mirror": None,
            "label": "psi_like",
            "bindings": 0,
            "projections": 0,
            "old_mirrors": 0,
        }
    ]
    assert graph_client.query(
        """
        MATCH (event)
        WHERE event.id IN $ids
        WITH event ORDER BY event.id
        RETURN collect(labels(event)) AS labels
        """,
        ids=_event_ids(rows[0]),
    )[0]["labels"] == [
        ["StandardNameChange"],
        ["StandardNameSourceAuthorityRetirement"],
    ]

    repeated = sut.reconcile_protected_structure(
        path, apply=True, expected_manifest_hash=digest, gc=graph_client
    )
    assert repeated["mode"] == "already_current"
    graph_client.query(
        "MATCH (event {id: $id}) SET event.reason = 'tampered'",
        id=_event_ids(rows[0])[1],
    )
    refused = sut.reconcile_protected_structure(path, gc=graph_client)
    assert refused["counts"]["refused"] == 1


def test_graph_manifest_rejects_shared_target(
    tmp_path: Path, graph_client: GraphClient
) -> None:
    seeded = _seed(graph_client, sut.RETIRE_STALE_SOURCE_BRANCH, count=2)
    _, _, rows = _manifest(
        tmp_path, graph_client, seeded, sut.RETIRE_STALE_SOURCE_BRANCH
    )
    rows[1]["target_id"] = rows[0]["target_id"]
    rows[1]["row_key"] = sut.payload_hash(
        {key: value for key, value in rows[1].items() if key != "row_key"}
    )
    _, protected = _snapshots_and_protection(graph_client, seeded)
    payload = sut.build_manifest_payload(
        rows,
        protected_set_hash=protected.protected_set_hash,
        authority_evidence_sha256="0" * 64,
        negative_fixture_labels=_NEGATIVE_FIXTURE_LABELS,
    )
    path = tmp_path / "shared-target.json"
    path.write_text(json.dumps(payload, sort_keys=True))

    with pytest.raises(ValueError, match="overlap"):
        sut.reconcile_protected_structure(path, gc=graph_client)


@pytest.mark.parametrize(
    ("action", "expected_dry_queries", "expected_apply_queries"),
    [
        (sut.PROTECTED_IDENTITY_FOLD, 2, 9),
        (sut.RETIRE_STALE_SOURCE_BRANCH, 2, 9),
    ],
)
@pytest.mark.parametrize("count", [1, 40])
def test_access_plan_is_constant_across_cohort_size(
    tmp_path: Path,
    graph_client: GraphClient,
    action: str,
    expected_dry_queries: int,
    expected_apply_queries: int,
    count: int,
) -> None:
    seeded = _seed(graph_client, action, count=count)
    path, digest, _ = _manifest(tmp_path, graph_client, seeded, action)
    dry_counter = _CountingGraphClient(graph_client)
    dry = sut.reconcile_protected_structure(path, gc=dry_counter)
    assert dry_counter.query_count == expected_dry_queries
    assert dry["query_audit"]["query_count"] == expected_dry_queries

    apply_counter = _CountingGraphClient(graph_client)
    applied = sut.reconcile_protected_structure(
        path, apply=True, expected_manifest_hash=digest, gc=apply_counter
    )
    assert applied["counts"]["applied"] == count
    assert apply_counter.query_count == expected_apply_queries
    assert applied["query_audit"]["query_count"] == expected_apply_queries

    census_counter = _CountingGraphClient(graph_client)
    census = sut.census_protected_structural_release(
        path,
        applied,
        expected_receipt_hash=applied["receipt_hash"],
        gc=census_counter,
    )
    assert census["release_ready"] is True, census["catalog_reasons"]
    assert census_counter.query_count == 1
    assert census["query_audit"]["query_count"] == 1
