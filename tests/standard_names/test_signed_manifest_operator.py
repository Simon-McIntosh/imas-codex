"""Transactional evidence for the generic signed-manifest operator."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from neo4j import GraphDatabase

from imas_codex.graph.client import GraphClient
from imas_codex.graph.profiles import resolve_neo4j
from imas_codex.standard_names import signed_manifest as operator
from imas_codex.standard_names.signed_manifest import (
    SignedManifestAuthorityError,
    SignedManifestConflict,
    apply_signed_manifest,
    signed_payload_sha256,
)

_SELECTION = {
    "id": "artifact-rows",
    "mode": "exact_complete_signed_cohort",
    "predicate": "artifact-rows",
}
_OPERATION = "test_signed_manifest_change"


def _guard(implementation: str) -> dict[str, Any]:
    kind = {
        "last-producing-source": "semantic_authority",
        "structural-legitimacy": "semantic_authority",
        "out-of-allowlist-immutability": "collateral_immutability",
    }[implementation]
    return {
        "id": implementation,
        "kind": kind,
        "implementation": implementation,
        "participant_ids": [],
    }


def _node_participant(node_id: str, label: str) -> dict[str, Any]:
    return {"id": node_id, "kind": "node", "graph_label": label}


def _relationship_participant(element_id: str) -> dict[str, Any]:
    return {
        "id": element_id,
        "kind": "relationship",
        "graph_label": "PRODUCED_NAME",
    }


def _row(
    row_id: str,
    *,
    kind: str,
    mutation_participant: dict[str, Any],
    participants: list[dict[str, Any]] | None = None,
    source_id: str | None = None,
    target_id: str | None = None,
) -> dict[str, Any]:
    guard_names = ["out-of-allowlist-immutability"]
    if kind == "detach":
        guard_names.append("last-producing-source")
    else:
        guard_names.append("structural-legitimacy")
    return {
        "id": row_id,
        "identity": {
            "id": row_id,
            "kind": "source" if source_id else "standard_name",
            "source_id": source_id,
            "target_id": target_id,
        },
        "participants": participants or [mutation_participant],
        "selection": _SELECTION,
        "mutations": [
            {
                "id": f"{row_id}:mutation",
                "order": 0,
                "kind": kind,
                "participant_id": mutation_participant["id"],
            }
        ],
        "guards": [_guard(name) for name in guard_names],
        "orphan_policy": "refuse",
    }


def _write_authority(path: Path, rows: list[dict[str, Any]]) -> tuple[str, str]:
    authority: dict[str, Any] = {
        "schema": "imas-codex.repair-authority.v1",
        "operation_id": "test-signed-manifest",
        "authority_mode": "external_reviewed",
        "rows": rows,
        "repair_rows": [row["id"] for row in rows],
        "selection": _SELECTION,
        "receipt_policy": {
            "id": "one-per-logical-change",
            "operation": _OPERATION,
            "cardinality": "per_target",
            "expected_count": "admitted_rows",
            "link_participant_kind": "standard_name",
            "replay_projection": ["manifest_sha256", "row_id"],
        },
        "orphan_policy": "refuse",
    }
    payload_sha256 = signed_payload_sha256(authority)
    authority["signature"] = {
        "canonicalization": "json-sort-keys-v1",
        "sha256": payload_sha256,
    }
    raw = json.dumps(authority, sort_keys=True, indent=2).encode()
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest(), payload_sha256


def _preview(
    path: Path,
    file_sha256: str,
    payload_sha256: str,
    client: GraphClient,
    reason: str = "exercise the signed generic transaction envelope",
) -> dict[str, Any]:
    return apply_signed_manifest(
        path,
        authority_file_sha256=file_sha256,
        authority_payload_sha256=payload_sha256,
        reason=reason,
        gc=client,
    )


def _apply(
    path: Path,
    file_sha256: str,
    payload_sha256: str,
    client: GraphClient,
    manifest_sha256: str,
    reason: str = "exercise the signed generic transaction envelope",
) -> dict[str, Any]:
    return apply_signed_manifest(
        path,
        authority_file_sha256=file_sha256,
        authority_payload_sha256=payload_sha256,
        reason=reason,
        apply=True,
        manifest_sha256=manifest_sha256,
        gc=client,
    )


def test_file_and_canonical_payload_digests_are_both_required(tmp_path: Path) -> None:
    row = _row(
        "digest-row",
        kind="supersede",
        mutation_participant=_node_participant("digest-target", "StandardName"),
        target_id="digest-target",
    )
    path = tmp_path / "authority.json"
    file_sha256, payload_sha256 = _write_authority(path, [row])

    with pytest.raises(
        SignedManifestAuthorityError, match="^authority file SHA-256 mismatch$"
    ):
        apply_signed_manifest(
            path,
            authority_file_sha256="0" * 64,
            authority_payload_sha256=payload_sha256,
            reason="file digest is independent authority",
        )
    with pytest.raises(
        SignedManifestAuthorityError,
        match="^canonical signed-payload SHA-256 mismatch$",
    ):
        apply_signed_manifest(
            path,
            authority_file_sha256=file_sha256,
            authority_payload_sha256="0" * 64,
            reason="payload digest is independent authority",
        )

    loaded = operator._load_authority(
        path,
        expected_file_sha256=file_sha256,
        expected_payload_sha256=payload_sha256,
    )
    assert [row.id for row in loaded.rows] == ["digest-row"]


@pytest.fixture(scope="module")
def disposable_neo4j() -> Iterator[tuple[str, str]]:
    uri = os.environ.get("IMAS_CODEX_TEST_NEO4J_URI")
    if not uri:
        pytest.skip("IMAS_CODEX_TEST_NEO4J_URI is not configured")
    if os.environ.get("IMAS_CODEX_TEST_NEO4J_EPHEMERAL") != "1":
        pytest.fail("signed-manifest tests require a disposable graph")
    project_uri = resolve_neo4j(auto_tunnel=False).uri
    if uri.rstrip("/") == project_uri.rstrip("/"):
        pytest.fail("signed-manifest tests refuse the project graph URI")
    password = os.environ.get("IMAS_CODEX_TEST_NEO4J_PASSWORD", "")
    auth = ("neo4j", password) if password else None
    with GraphDatabase.driver(uri, auth=auth) as driver:
        driver.verify_connectivity()
        driver.execute_query("MATCH (node) DETACH DELETE node")
    yield uri, password


@pytest.fixture
def client(disposable_neo4j: tuple[str, str]) -> Iterator[GraphClient]:
    uri, password = disposable_neo4j
    graph = GraphClient(
        uri=uri,
        username="neo4j",
        password=password,
        graph_name="signed-manifest-operator",
    )
    graph.query("MATCH (node) DETACH DELETE node")
    yield graph
    graph.query("MATCH (node) DETACH DELETE node")
    graph.close()


def _binding_id(client: GraphClient, source_id: str, target_id: str) -> str:
    return str(
        client.query(
            """
            MATCH (source:StandardNameSource {id: $source_id})
                  -[binding:PRODUCED_NAME]->
                  (target:StandardName {id: $target_id})
            RETURN elementId(binding) AS element_id
            """,
            source_id=source_id,
            target_id=target_id,
        )[0]["element_id"]
    )


def _seed_name(client: GraphClient, name_id: str) -> None:
    client.query(
        """
        CREATE (:StandardName {
          id: $name_id,
          name_stage: 'accepted',
          status: 'draft',
          source_paths: []
        })
        """,
        name_id=name_id,
    )


def _seed_binding(client: GraphClient, source_id: str, target_id: str) -> str:
    client.query(
        """
        MATCH (target:StandardName {id: $target_id})
        CREATE (:StandardNameSource {
          id: $source_id,
          status: 'composed',
          produced_sn_id: $target_id
        })-[:PRODUCED_NAME]->(target)
        """,
        source_id=source_id,
        target_id=target_id,
    )
    return _binding_id(client, source_id, target_id)


@pytest.mark.graph
def test_preview_derives_complete_rows_and_emits_would_apply_closure(
    client: GraphClient, tmp_path: Path
) -> None:
    _seed_name(client, "preview-target-a")
    _seed_name(client, "preview-target-b")
    _seed_name(client, "untouched-target")
    rows = [
        _row(
            row_id,
            kind="supersede",
            mutation_participant=_node_participant(target_id, "StandardName"),
            target_id=target_id,
        )
        for row_id, target_id in (
            ("preview-a", "preview-target-a"),
            ("preview-b", "preview-target-b"),
        )
    ]
    path = tmp_path / "preview-authority.json"
    file_sha256, payload_sha256 = _write_authority(path, rows)

    preview = _preview(path, file_sha256, payload_sha256, client)

    assert preview["outcome"] == "would_apply"
    assert preview["counts"] == {"authority_rows": 2, "admitted": 2, "refused": 0}
    assert preview["manifest"]["admitted_row_ids"] == ["preview-a", "preview-b"]
    assert len(preview["manifest"]["rows"]) == 2
    assert len(preview["manifest"]["collateral_rows"]) == 1
    assert client.query(
        "MATCH (change:StandardNameChange) RETURN count(change) AS n"
    ) == [{"n": 0}]


@pytest.mark.graph
@pytest.mark.parametrize("mutation_kind", ["delete", "supersede", "detach"])
def test_mutation_kinds_write_one_receipt_and_exact_replay_is_write_free(
    client: GraphClient,
    tmp_path: Path,
    mutation_kind: str,
) -> None:
    target_id = f"{mutation_kind}-target"
    _seed_name(client, target_id)
    if mutation_kind == "detach":
        binding_id = _seed_binding(client, "detached-source", target_id)
        _seed_binding(client, "retained-source", target_id)
        mutation_participant = _relationship_participant(binding_id)
        participants = [
            _node_participant("detached-source", "StandardNameSource"),
            _node_participant(target_id, "StandardName"),
            mutation_participant,
        ]
        source_id = "detached-source"
    else:
        mutation_participant = _node_participant(target_id, "StandardName")
        participants = None
        source_id = None
    row = _row(
        f"{mutation_kind}-row",
        kind=mutation_kind,
        mutation_participant=mutation_participant,
        participants=participants,
        source_id=source_id,
        target_id=target_id,
    )
    path = tmp_path / f"{mutation_kind}-authority.json"
    file_sha256, payload_sha256 = _write_authority(path, [row])
    preview = _preview(path, file_sha256, payload_sha256, client)

    applied = _apply(
        path,
        file_sha256,
        payload_sha256,
        client,
        preview["manifest_sha256"],
    )

    assert applied["outcome"] == "applied"
    assert applied["changed"] == 1
    assert applied["receipt_rows"] == 1
    receipt_count = client.query(
        """
        MATCH (change:StandardNameChange {
          operation: $operation,
          manifest_sha256: $manifest_sha256
        })
        RETURN count(change) AS count
        """,
        operation=_OPERATION,
        manifest_sha256=preview["manifest_sha256"],
    )[0]["count"]
    assert receipt_count == 1
    graph_before = client.query(
        """
        MATCH (node)
        OPTIONAL MATCH (node)-[relationship]->(other)
        RETURN elementId(node) AS node_id, properties(node) AS properties,
               collect({id: elementId(relationship), properties: properties(relationship),
                        other: elementId(other)}) AS relationships
        ORDER BY node_id
        """
    )
    replay = _apply(
        path,
        file_sha256,
        payload_sha256,
        client,
        preview["manifest_sha256"],
    )
    graph_after = client.query(
        """
        MATCH (node)
        OPTIONAL MATCH (node)-[relationship]->(other)
        RETURN elementId(node) AS node_id, properties(node) AS properties,
               collect({id: elementId(relationship), properties: properties(relationship),
                        other: elementId(other)}) AS relationships
        ORDER BY node_id
        """
    )
    assert replay["outcome"] == "already_applied"
    assert replay["changed"] == 0
    assert replay["persistent_writes"] == 0
    assert graph_after == graph_before


@pytest.mark.graph
def test_maximal_safe_subset_refuses_the_last_producer_with_exact_reason(
    client: GraphClient, tmp_path: Path
) -> None:
    target_id = "shared-target"
    _seed_name(client, target_id)
    binding_a = _seed_binding(client, "source-a", target_id)
    binding_b = _seed_binding(client, "source-b", target_id)
    rows = [
        _row(
            f"detach-{suffix}",
            kind="detach",
            mutation_participant=_relationship_participant(binding_id),
            participants=[
                _node_participant(f"source-{suffix}", "StandardNameSource"),
                _node_participant(target_id, "StandardName"),
                _relationship_participant(binding_id),
            ],
            source_id=f"source-{suffix}",
            target_id=target_id,
        )
        for suffix, binding_id in (("a", binding_a), ("b", binding_b))
    ]
    path = tmp_path / "maximal-authority.json"
    file_sha256, payload_sha256 = _write_authority(path, rows)

    preview = _preview(path, file_sha256, payload_sha256, client)

    assert preview["counts"] == {"authority_rows": 2, "admitted": 1, "refused": 1}
    assert preview["refusals"] == [
        {
            "row_id": "detach-b",
            "reason": "target would lose its last producing source",
        }
    ]
    applied = _apply(
        path,
        file_sha256,
        payload_sha256,
        client,
        preview["manifest_sha256"],
    )
    assert applied["changed"] == 1
    assert applied["receipt_rows"] == 1
    assert client.query(
        """
        MATCH (:StandardNameSource)-[binding:PRODUCED_NAME]->
              (:StandardName {id: $target_id})
        RETURN count(binding) AS count
        """,
        target_id=target_id,
    ) == [{"count": 1}]


@pytest.mark.graph
def test_structural_legitimacy_refuses_live_child_with_exact_reason(
    client: GraphClient, tmp_path: Path
) -> None:
    _seed_name(client, "structural-target")
    _seed_name(client, "live-child")
    client.query(
        """
        MATCH (target:StandardName {id: 'structural-target'}),
              (child:StandardName {id: 'live-child'})
        CREATE (child)-[:HAS_PARENT]->(target)
        """
    )
    row = _row(
        "structural-row",
        kind="supersede",
        mutation_participant=_node_participant("structural-target", "StandardName"),
        target_id="structural-target",
    )
    path = tmp_path / "structural-authority.json"
    file_sha256, payload_sha256 = _write_authority(path, [row])

    preview = _preview(path, file_sha256, payload_sha256, client)

    assert preview["outcome"] == "refused"
    assert preview["refusals"] == [
        {"row_id": "structural-row", "reason": "target has a live structural child"}
    ]


@pytest.mark.graph
def test_collateral_change_refuses_authorized_manifest_with_exact_reason(
    client: GraphClient, tmp_path: Path
) -> None:
    _seed_name(client, "collateral-target")
    _seed_name(client, "untouched-collateral")
    row = _row(
        "collateral-row",
        kind="supersede",
        mutation_participant=_node_participant("collateral-target", "StandardName"),
        target_id="collateral-target",
    )
    path = tmp_path / "collateral-authority.json"
    file_sha256, payload_sha256 = _write_authority(path, [row])
    preview = _preview(path, file_sha256, payload_sha256, client)
    client.query(
        "MATCH (name:StandardName {id: 'untouched-collateral'}) SET name.drift = true"
    )

    with pytest.raises(
        SignedManifestConflict,
        match=("^fresh signed-manifest closure does not match authorized SHA-256$"),
    ):
        _apply(
            path,
            file_sha256,
            payload_sha256,
            client,
            preview["manifest_sha256"],
        )


@pytest.mark.graph
def test_lock_then_rehash_refuses_participant_drift_and_rolls_back(
    client: GraphClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_name(client, "locking-target")
    row = _row(
        "locking-row",
        kind="supersede",
        mutation_participant=_node_participant("locking-target", "StandardName"),
        target_id="locking-target",
    )
    path = tmp_path / "locking-authority.json"
    file_sha256, payload_sha256 = _write_authority(path, [row])
    preview = _preview(path, file_sha256, payload_sha256, client)
    original_lock = operator._lock_participants

    def lock_and_drift(query: Any, current_preview: Any) -> None:
        original_lock(query, current_preview)
        query.query(
            "MATCH (name:StandardName {id: 'locking-target'}) SET name.drift = true"
        )

    monkeypatch.setattr(operator, "_lock_participants", lock_and_drift)
    with pytest.raises(
        SignedManifestConflict,
        match="^signed-manifest closure changed while locking$",
    ):
        _apply(
            path,
            file_sha256,
            payload_sha256,
            client,
            preview["manifest_sha256"],
        )
    assert client.query(
        "MATCH (name:StandardName {id: 'locking-target'}) RETURN name.drift AS drift"
    ) == [{"drift": None}]


@pytest.mark.graph
def test_receipt_and_provider_counters_are_measured_inside_apply(
    client: GraphClient, tmp_path: Path
) -> None:
    _seed_name(client, "counter-target")
    client.query("CREATE (:LLMCost {id: 'unrelated-cost', llm_cost: 0.0})")
    row = _row(
        "counter-row",
        kind="supersede",
        mutation_participant=_node_participant("counter-target", "StandardName"),
        target_id="counter-target",
    )
    path = tmp_path / "counter-authority.json"
    file_sha256, payload_sha256 = _write_authority(path, [row])
    preview = _preview(path, file_sha256, payload_sha256, client)
    before = client.query(
        """
        RETURN COUNT { (:StandardNameChange) } AS changes,
               COUNT { (:LLMCost) } AS llm_costs
        """
    )[0]

    applied = _apply(
        path,
        file_sha256,
        payload_sha256,
        client,
        preview["manifest_sha256"],
    )
    after = client.query(
        """
        RETURN COUNT { (:StandardNameChange) } AS changes,
               COUNT { (:LLMCost) } AS llm_costs
        """
    )[0]

    assert applied["receipt_rows"] == 1
    assert after["changes"] - before["changes"] == 1
    assert after["llm_costs"] == before["llm_costs"]
