"""Transactional coverage for signed source-disposition reconciliation."""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from pathlib import Path

import pytest
from neo4j import GraphDatabase

from imas_codex.graph.client import GraphClient
from imas_codex.settings import get_graph_uri
from imas_codex.standard_names.graph_ops import (
    SemanticMirrorRepairConflict,
    SignedSourceDispositionConflict,
    _catalog_edit_adjudication_signature_hash,
    apply_adjudicated_source_dispositions,
    repair_scalar_projection_mismatches,
)


@pytest.fixture(scope="module")
def disposable_neo4j() -> Iterator[tuple[str, str]]:
    uri = os.environ.get("IMAS_CODEX_TEST_NEO4J_URI")
    if not uri:
        pytest.skip("IMAS_CODEX_TEST_NEO4J_URI is not configured")
    if os.environ.get("IMAS_CODEX_TEST_NEO4J_EPHEMERAL") != "1":
        pytest.fail("source dispositions require a disposable graph")
    if uri == (os.environ.get("IMAS_CODEX_TEST_PROJECT_NEO4J_URI") or get_graph_uri()):
        pytest.fail("source dispositions refuse the project graph")
    password = os.environ.get("IMAS_CODEX_TEST_NEO4J_PASSWORD", "")
    auth = ("neo4j", password) if password else None
    with GraphDatabase.driver(uri, auth=auth) as driver:
        driver.verify_connectivity()
    yield uri, password


def _client(endpoint: tuple[str, str], name: str) -> GraphClient:
    uri, password = endpoint
    return GraphClient(uri=uri, username="neo4j", password=password, graph_name=name)


def _digest(value: object) -> str:
    return _catalog_edit_adjudication_signature_hash(value)


def _seed(
    client: GraphClient,
    prefix: str,
    *,
    scalar: str | None,
    preserve_removed_target: bool = True,
) -> dict[str, str]:
    ids = {
        "source": f"dd:{prefix}/value",
        "path": f"{prefix}/value",
        "catalog": f"{prefix}_catalog",
        "catalog_anchor": f"dd:{prefix}/catalog_anchor",
        "semantic": f"{prefix}_semantic",
    }
    client.query(
        "CREATE (catalog:StandardName {id: $catalog, name_stage: 'accepted', "
        "validation_status: 'valid', origin: 'catalog_edit', "
        "source_paths: ['dd:' + $path]}) "
        "CREATE (semantic:StandardName {id: $semantic, name_stage: 'accepted', "
        "validation_status: 'valid', origin: 'pipeline', "
        "source_paths: ['dd:' + $path]}) "
        "CREATE (backing:IMASNode {id: $path}) "
        "CREATE (source:StandardNameSource {id: $source, source_type: 'dd', "
        "source_id: $path, status: 'attached', produced_sn_id: $scalar}) "
        "CREATE (source)-[:FROM_DD_PATH]->(backing) "
        "CREATE (source)-[:PRODUCED_NAME]->(catalog) "
        "CREATE (source)-[:PRODUCED_NAME]->(semantic) "
        "CREATE (backing)-[:HAS_STANDARD_NAME]->(catalog) "
        "CREATE (backing)-[:HAS_STANDARD_NAME]->(semantic)",
        **ids,
        scalar=scalar,
    )
    if preserve_removed_target:
        client.query(
            "MATCH (catalog:StandardName {id: $catalog}) "
            "CREATE (anchor:StandardNameSource {id: $catalog_anchor, "
            "source_type: 'dd', source_id: $catalog_anchor, status: 'attached', "
            "produced_sn_id: $catalog}) "
            "CREATE (anchor)-[:PRODUCED_NAME]->(catalog)",
            **ids,
        )
    return ids


def _row(ids: dict[str, str], disposition: str, prior: str | None) -> dict[str, object]:
    survivor = ids["semantic"]
    row: dict[str, object] = {
        "authority": "dd_path_identity",
        "candidate_live_targets": sorted([ids["catalog"], survivor]),
        "disposition": disposition,
        "evidence": {"dd_source_id": ids["path"]},
        "family_key": "|".join(sorted([ids["catalog"], survivor])),
        "prior_scalar_target": prior,
        "rationale": "the exact DD source selects the semantic identity",
        "removed_targets": [ids["catalog"]],
        "source_id": ids["source"],
        "surviving_target": survivor,
    }
    row["row_signature_sha256"] = _digest(row)
    return row


def _seed_mirror_mismatch(
    client: GraphClient,
    prefix: str,
    *,
    scalar_matches: bool,
    projection_present: bool,
) -> dict[str, str]:
    ids = {
        "source": f"dd:{prefix}/value",
        "path": f"{prefix}/value",
        "target": f"{prefix}_target",
        "prior": f"{prefix}_prior",
    }
    client.query(
        "CREATE (target:StandardName {id: $target, name_stage: 'accepted', "
        "validation_status: 'valid', origin: 'pipeline'}) "
        "CREATE (prior:StandardName {id: $prior, name_stage: 'accepted', "
        "validation_status: 'valid', origin: 'pipeline'}) "
        "CREATE (backing:IMASNode {id: $path}) "
        "CREATE (source:StandardNameSource {id: $source, source_type: 'dd', "
        "source_id: $path, status: 'attached', produced_sn_id: CASE "
        "WHEN $scalar_matches THEN $target ELSE $prior END}) "
        "CREATE (source)-[:FROM_DD_PATH]->(backing) "
        "CREATE (source)-[:PRODUCED_NAME]->(target) "
        "FOREACH (_ IN CASE WHEN $projection_present THEN [1] ELSE [] END | "
        "CREATE (backing)-[:HAS_STANDARD_NAME]->(target))",
        **ids,
        scalar_matches=scalar_matches,
        projection_present=projection_present,
    )
    return ids


def _adjudication(rows: list[dict[str, object]]) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema": "imas-codex.catalog-edit-dual-binding-adjudication.v1",
        "rows": sorted(rows, key=lambda row: str(row["source_id"])),
    }
    return {
        **payload,
        "signature": {
            "algorithm": "sha256",
            "payload_sha256": _digest(payload),
        },
    }


def _structural_authority(
    ids: dict[str, str],
    signed_row: dict[str, object],
    *,
    disposition: str,
    child_id: str | None = None,
    relationship_id: str | None = None,
) -> tuple[dict[str, object], str]:
    disposition_counts = {
        "preserve_as_structural_identity": 0,
        "re_source_from_existing_dd_path": 0,
        "retain_competing_binding": 0,
        "retire_under_orphan_policy": 0,
    }
    disposition_counts[disposition] = 1
    row: dict[str, object] = {
        "name": ids["catalog"],
        "disposition": disposition,
        "mutation_authority": "classification_only",
        "current_removed_bindings": [
            {
                "source_id": ids["source"],
                "signed_row_sha256": signed_row["row_signature_sha256"],
                "signed_surviving_target": signed_row["surviving_target"],
            }
        ],
        "structural_closure": {
            "classification": (
                "structurally_legitimate_without_producing_source"
                if disposition == "preserve_as_structural_identity"
                else "no_live_structural_descendant"
            ),
            "has_live_has_parent_child": child_id is not None,
            "live_has_parent_children": (
                [{"name": child_id, "relationship_id": relationship_id}]
                if child_id is not None
                else []
            ),
        },
    }
    authority: dict[str, object] = {
        "schema": "imas-codex.refused-target-orphan-adjudication.v2",
        "read_only": True,
        "rows": [row],
        "summary": {
            "targets": 1,
            "disposition_counts": disposition_counts,
            "disposition_sum": 1,
        },
    }
    return authority, _digest(authority)


def _cleanup(
    client: GraphClient,
    ids: list[dict[str, str]],
    manifest_sha256: str | None = None,
) -> None:
    node_ids = [value for item in ids for value in item.values()]
    client.query(
        "MATCH (node) WHERE node.id IN $ids OR node.manifest_sha256 = $manifest "
        "DETACH DELETE node",
        ids=node_ids,
        manifest=manifest_sha256,
    )


def _snapshot(client: GraphClient, ids: list[str]) -> bytes:
    nodes = client.query(
        "MATCH (node) WHERE node.id IN $ids "
        "RETURN elementId(node) AS element_id, labels(node) AS labels, "
        "properties(node) AS properties ORDER BY element_id",
        ids=ids,
    )
    relationships = client.query(
        "MATCH (start)-[relationship]->(end) "
        "WHERE start.id IN $ids OR end.id IN $ids "
        "RETURN elementId(relationship) AS element_id, type(relationship) AS type, "
        "properties(relationship) AS properties, elementId(start) AS start, "
        "elementId(end) AS end ORDER BY element_id",
        ids=ids,
    )
    return json.dumps(
        {"nodes": nodes, "relationships": relationships},
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode()


@pytest.mark.graph
def test_signed_dispositions_apply_all_modes_and_preserve_outside_allowlist(
    disposable_neo4j: tuple[str, str],
) -> None:
    client = _client(disposable_neo4j, "signed-source-dispositions")
    retained = _seed(client, "dispositionretain", scalar="dispositionretain_semantic")
    retargeted = _seed(
        client, "dispositionretarget", scalar="dispositionretarget_catalog"
    )
    selected = _seed(client, "dispositionmissing", scalar=None)
    outside = _seed(client, "dispositionoutside", scalar="dispositionoutside_catalog")
    adjudication = _adjudication(
        [
            _row(retained, "retain_scalar_target", retained["semantic"]),
            _row(retargeted, "retarget_scalar_target", retargeted["catalog"]),
            _row(selected, "select_missing_scalar", None),
        ]
    )
    preview = None
    try:
        outside_ids = list(outside.values())
        outside_before = _snapshot(client, outside_ids)
        counters_before = client.query(
            "RETURN COUNT { (:StandardNameChange) } AS changes, "
            "COUNT { (:LLMCost) } AS costs"
        )
        preview = apply_adjudicated_source_dispositions(
            adjudication,
            reason="apply exact independently adjudicated source identities",
            gc=client,
        )
        assert preview["outcome"] == "would_apply"
        assert preview["counts"] == {
            "requested": 3,
            "admitted": 3,
            "refused": 0,
            "bindings_to_remove": 3,
            "projections_to_remove": 3,
            "scalars_to_change": 2,
        }
        assert _snapshot(client, outside_ids) == outside_before
        assert (
            client.query(
                "RETURN COUNT { (:StandardNameChange) } AS changes, "
                "COUNT { (:LLMCost) } AS costs"
            )
            == counters_before
        )

        applied = apply_adjudicated_source_dispositions(
            adjudication,
            reason="apply exact independently adjudicated source identities",
            apply=True,
            manifest_sha256=preview["manifest_sha256"],
            gc=client,
        )
        assert applied["outcome"] == "applied"
        assert applied["sources_reconciled"] == 3
        assert _snapshot(client, outside_ids) == outside_before
        assert client.query(
            "UNWIND $ids AS source_id "
            "MATCH (source:StandardNameSource {id: source_id}) "
            "OPTIONAL MATCH (source)-[:PRODUCED_NAME]->(target:StandardName) "
            "WITH source, collect(target.id) AS targets "
            "MATCH (source)-[:FROM_DD_PATH]->(backing:IMASNode) "
            "OPTIONAL MATCH (backing)-[:HAS_STANDARD_NAME]->(projected:StandardName) "
            "RETURN source.id AS source_id, source.produced_sn_id AS scalar, "
            "targets, collect(projected.id) AS projections ORDER BY source_id",
            ids=sorted([retained["source"], retargeted["source"], selected["source"]]),
        ) == [
            {
                "source_id": item["source"],
                "scalar": item["semantic"],
                "targets": [item["semantic"]],
                "projections": [item["semantic"]],
            }
            for item in sorted(
                [retained, retargeted, selected], key=lambda item: item["source"]
            )
        ]
        replay_before = _snapshot(
            client,
            [
                value
                for item in [retained, retargeted, selected, outside]
                for value in item.values()
            ],
        )
        replay = apply_adjudicated_source_dispositions(
            adjudication,
            reason="apply exact independently adjudicated source identities",
            apply=True,
            manifest_sha256=preview["manifest_sha256"],
            gc=client,
        )
        assert replay["outcome"] == "already_applied"
        assert replay["changed"] == 0
        assert (
            _snapshot(
                client,
                [
                    value
                    for item in [retained, retargeted, selected, outside]
                    for value in item.values()
                ],
            )
            == replay_before
        )
    finally:
        _cleanup(
            client,
            [retained, retargeted, selected, outside],
            (preview or {}).get("manifest_sha256"),
        )


@pytest.mark.graph
@pytest.mark.parametrize("drift", ["claim", "scalar"])
def test_apply_refuses_scalar_or_claim_drift_after_preview(
    disposable_neo4j: tuple[str, str], drift: str
) -> None:
    client = _client(disposable_neo4j, f"source-disposition-{drift}-drift")
    ids = _seed(client, f"disposition{drift}drift", scalar=None)
    adjudication = _adjudication([_row(ids, "select_missing_scalar", None)])
    preview = None
    try:
        preview = apply_adjudicated_source_dispositions(
            adjudication, reason="bind the exact adjudicated target", gc=client
        )
        if drift == "claim":
            client.query(
                "MATCH (source:StandardNameSource {id: $source}) "
                "SET source.claimed_at = datetime(), source.claim_token = 'other-worker'",
                **ids,
            )
        else:
            client.query(
                "MATCH (source:StandardNameSource {id: $source}) "
                "SET source.produced_sn_id = $catalog",
                **ids,
            )
        before = _snapshot(client, list(ids.values()))
        with pytest.raises(
            SignedSourceDispositionConflict,
            match="fresh source-disposition manifest does not match signed hash",
        ):
            apply_adjudicated_source_dispositions(
                adjudication,
                reason="bind the exact adjudicated target",
                apply=True,
                manifest_sha256=preview["manifest_sha256"],
                gc=client,
            )
        assert _snapshot(client, list(ids.values())) == before
        assert client.query(
            "MATCH (source:StandardNameSource {id: $source}) "
            "RETURN COUNT { (source)-[:PRODUCED_NAME]->(:StandardName) } AS bindings",
            **ids,
        ) == [{"bindings": 2}]
    finally:
        _cleanup(client, [ids], (preview or {}).get("manifest_sha256"))


@pytest.mark.graph
def test_preview_refuses_incomplete_projection_without_writes(
    disposable_neo4j: tuple[str, str],
) -> None:
    client = _client(disposable_neo4j, "source-disposition-projection-refusal")
    ids = _seed(client, "dispositionprojection", scalar=None)
    adjudication = _adjudication([_row(ids, "select_missing_scalar", None)])
    try:
        client.query(
            "MATCH (:IMASNode {id: $path})-[projection:HAS_STANDARD_NAME]->"
            "(:StandardName {id: $catalog}) DELETE projection",
            **ids,
        )
        before = _snapshot(client, list(ids.values()))
        preview = apply_adjudicated_source_dispositions(
            adjudication, reason="bind only complete source authority", gc=client
        )
        assert preview["outcome"] == "refused"
        assert preview["counts"]["admitted"] + preview["counts"]["refused"] == 1
        assert preview["refusals"] == [
            {
                "source_id": ids["source"],
                "reason": "live backing projection set changed from adjudication",
            }
        ]
        assert _snapshot(client, list(ids.values())) == before
    finally:
        _cleanup(client, [ids])


@pytest.mark.graph
def test_preview_refuses_removal_of_target_last_live_binding(
    disposable_neo4j: tuple[str, str],
) -> None:
    client = _client(disposable_neo4j, "source-disposition-last-binding-refusal")
    ids = _seed(
        client,
        "dispositionlastbinding",
        scalar=None,
        preserve_removed_target=False,
    )
    adjudication = _adjudication([_row(ids, "select_missing_scalar", None)])
    try:
        ids["stale_anchor"] = f"dd:{ids['path']}/stale_anchor"
        client.query(
            "MATCH (catalog:StandardName {id: $catalog}) "
            "CREATE (source:StandardNameSource {id: $stale_anchor, "
            "source_type: 'dd', source_id: $stale_anchor, status: 'stale', "
            "produced_sn_id: $catalog}) "
            "CREATE (source)-[:PRODUCED_NAME]->(catalog)",
            **ids,
        )
        before = _snapshot(client, list(ids.values()))
        preview = apply_adjudicated_source_dispositions(
            adjudication,
            reason="preserve the final live source of every removed target",
            gc=client,
        )
        assert preview["outcome"] == "refused"
        assert preview["counts"] == {
            "requested": 1,
            "admitted": 0,
            "refused": 1,
            "bindings_to_remove": 0,
            "projections_to_remove": 0,
            "scalars_to_change": 0,
        }
        assert preview["refusals"] == [
            {
                "source_id": ids["source"],
                "target_ids": [ids["catalog"]],
                "reason": (
                    "removal would leave target with zero live producing sources"
                ),
            }
        ]
        closure = preview["manifest"]["removed_target_closures"]
        assert len(closure) == 1
        assert closure[0]["target_id"] == ids["catalog"]
        assert {
            binding["source_id"] for binding in closure[0]["incoming_bindings"]
        } == {ids["source"], ids["stale_anchor"]}
        assert _snapshot(client, list(ids.values())) == before
    finally:
        _cleanup(client, [ids])


@pytest.mark.graph
def test_signed_structural_authority_admits_last_binding_removal(
    disposable_neo4j: tuple[str, str],
) -> None:
    client = _client(disposable_neo4j, "source-disposition-structural-authority")
    ids = _seed(
        client,
        "dispositionstructural",
        scalar=None,
        preserve_removed_target=False,
    )
    signed_row = _row(ids, "select_missing_scalar", None)
    ids["child"] = f"{ids['catalog']}_child"
    preview = None
    try:
        relationship = client.query(
            "MATCH (target:StandardName {id: $catalog}) "
            "CREATE (child:StandardName {id: $child, name_stage: 'accepted', "
            "validation_status: 'valid', origin: 'derived'}) "
            "CREATE (child)-[relationship:HAS_PARENT]->(target) "
            "RETURN elementId(relationship) AS relationship_id",
            **ids,
        )[0]
        authority, authority_sha256 = _structural_authority(
            ids,
            signed_row,
            disposition="preserve_as_structural_identity",
            child_id=ids["child"],
            relationship_id=relationship["relationship_id"],
        )
        adjudication = _adjudication([signed_row])
        preview = apply_adjudicated_source_dispositions(
            adjudication,
            reason="preserve signed direct-child authority after source repair",
            structural_authority=authority,
            structural_authority_sha256=authority_sha256,
            gc=client,
        )
        assert preview["outcome"] == "would_apply"
        assert preview["counts"] == {
            "requested": 1,
            "admitted": 1,
            "refused": 0,
            "bindings_to_remove": 1,
            "projections_to_remove": 1,
            "scalars_to_change": 1,
        }
        assert preview["manifest"]["structural_legitimacy_authority"] == {
            "schema": "imas-codex.refused-target-orphan-adjudication.v2",
            "payload_sha256": authority_sha256,
            "target_ids": [ids["catalog"]],
        }
        assert [
            exemption["target_id"]
            for exemption in preview["manifest"]["structural_exemptions"]
        ] == [ids["catalog"]]
        assert (
            preview["manifest"]["structural_exemptions"][0]["live_direct_children"][0][
                "child_id"
            ]
            == ids["child"]
        )

        applied = apply_adjudicated_source_dispositions(
            adjudication,
            reason="preserve signed direct-child authority after source repair",
            apply=True,
            manifest_sha256=preview["manifest_sha256"],
            structural_authority=authority,
            structural_authority_sha256=authority_sha256,
            gc=client,
        )
        assert applied["outcome"] == "applied"
        assert client.query(
            "MATCH (target:StandardName {id: $catalog}) "
            "RETURN COUNT { (:StandardNameSource)-[:PRODUCED_NAME]->(target) } "
            "AS producers, "
            "COUNT { (:StandardName {id: $child})-[:HAS_PARENT]->(target) } "
            "AS direct_children",
            **ids,
        ) == [{"producers": 0, "direct_children": 1}]
    finally:
        _cleanup(client, [ids], (preview or {}).get("manifest_sha256"))


@pytest.mark.graph
def test_signed_nonstructural_authority_still_refuses_last_binding_removal(
    disposable_neo4j: tuple[str, str],
) -> None:
    client = _client(disposable_neo4j, "source-disposition-nonstructural-refusal")
    ids = _seed(
        client,
        "dispositionnonstructural",
        scalar=None,
        preserve_removed_target=False,
    )
    signed_row = _row(ids, "select_missing_scalar", None)
    authority, authority_sha256 = _structural_authority(
        ids,
        signed_row,
        disposition="retire_under_orphan_policy",
    )
    try:
        before = _snapshot(client, list(ids.values()))
        preview = apply_adjudicated_source_dispositions(
            _adjudication([signed_row]),
            reason="refuse lifecycle authority in a source disposition",
            structural_authority=authority,
            structural_authority_sha256=authority_sha256,
            gc=client,
        )
        assert preview["outcome"] == "refused"
        assert preview["counts"]["admitted"] == 0
        assert preview["counts"]["refused"] == 1
        assert preview["manifest"]["structural_exemptions"] == []
        assert preview["refusals"] == [
            {
                "source_id": ids["source"],
                "target_ids": [ids["catalog"]],
                "reason": (
                    "removed target is outside signed structural legitimacy authority"
                ),
            }
        ]
        assert _snapshot(client, list(ids.values())) == before
    finally:
        _cleanup(client, [ids])


@pytest.mark.graph
def test_apply_refuses_global_incoming_binding_drift_after_preview(
    disposable_neo4j: tuple[str, str],
) -> None:
    client = _client(disposable_neo4j, "source-disposition-incoming-drift")
    ids = _seed(client, "dispositionincomingdrift", scalar=None)
    adjudication = _adjudication([_row(ids, "select_missing_scalar", None)])
    preview = None
    try:
        preview = apply_adjudicated_source_dispositions(
            adjudication,
            reason="bind the signed global incoming source closure",
            gc=client,
        )
        assert preview["outcome"] == "would_apply"
        ids["late_anchor"] = f"dd:{ids['path']}/late_anchor"
        client.query(
            "MATCH (catalog:StandardName {id: $catalog}) "
            "CREATE (source:StandardNameSource {id: $late_anchor, "
            "source_type: 'dd', source_id: $late_anchor, status: 'attached', "
            "produced_sn_id: $catalog}) "
            "CREATE (source)-[:PRODUCED_NAME]->(catalog)",
            **ids,
        )
        before = _snapshot(client, list(ids.values()))
        with pytest.raises(
            SignedSourceDispositionConflict,
            match="fresh source-disposition manifest does not match signed hash",
        ):
            apply_adjudicated_source_dispositions(
                adjudication,
                reason="bind the signed global incoming source closure",
                apply=True,
                manifest_sha256=preview["manifest_sha256"],
                gc=client,
            )
        assert _snapshot(client, list(ids.values())) == before
    finally:
        _cleanup(client, [ids], (preview or {}).get("manifest_sha256"))


@pytest.mark.graph
def test_admitted_subset_applies_only_safe_rows_and_replays_without_writes(
    disposable_neo4j: tuple[str, str],
) -> None:
    client = _client(disposable_neo4j, "source-disposition-admitted-subset")
    safe = _seed(client, "dispositionsubsetsafe", scalar=None)
    protected = _seed(
        client,
        "dispositionsubsetprotected",
        scalar=None,
        preserve_removed_target=False,
    )
    adjudication = _adjudication(
        [
            _row(safe, "select_missing_scalar", None),
            _row(protected, "select_missing_scalar", None),
        ]
    )
    preview = None
    try:
        protected_before = _snapshot(client, list(protected.values()))
        preview = apply_adjudicated_source_dispositions(
            adjudication,
            reason="apply only rows admitted by complete signed authority",
            admitted_subset=True,
            gc=client,
        )
        assert preview["outcome"] == "would_apply"
        assert preview["counts"] == {
            "requested": 1,
            "admitted": 1,
            "refused": 0,
            "bindings_to_remove": 1,
            "projections_to_remove": 1,
            "scalars_to_change": 1,
        }
        selection = preview["manifest"]["subset_selection"]
        assert selection["parent_counts"] == {
            "requested": 2,
            "admitted": 1,
            "refused": 1,
            "bindings_to_remove": 1,
            "projections_to_remove": 1,
            "scalars_to_change": 1,
        }
        assert selection["selected_source_ids"] == [safe["source"]]
        assert selection["excluded_source_ids"] == [protected["source"]]
        assert selection["excluded_refusals"] == [
            {
                "source_id": protected["source"],
                "target_ids": [protected["catalog"]],
                "reason": (
                    "removal would leave target with zero live producing sources"
                ),
            }
        ]
        assert _snapshot(client, list(protected.values())) == protected_before

        applied = apply_adjudicated_source_dispositions(
            adjudication,
            reason="apply only rows admitted by complete signed authority",
            apply=True,
            manifest_sha256=preview["manifest_sha256"],
            admitted_subset=True,
            gc=client,
        )
        assert applied["outcome"] == "applied"
        assert applied["sources_reconciled"] == 1
        assert applied["bindings_removed"] == 1
        assert applied["projections_removed"] == 1
        assert _snapshot(client, list(protected.values())) == protected_before

        replay_before = _snapshot(
            client, list(safe.values()) + list(protected.values())
        )
        replay = apply_adjudicated_source_dispositions(
            adjudication,
            reason="apply only rows admitted by complete signed authority",
            apply=True,
            manifest_sha256=preview["manifest_sha256"],
            admitted_subset=True,
            gc=client,
        )
        assert replay["outcome"] == "already_applied"
        assert replay["changed"] == 0
        assert (
            _snapshot(client, list(safe.values()) + list(protected.values()))
            == replay_before
        )
    finally:
        _cleanup(client, [safe, protected], (preview or {}).get("manifest_sha256"))


@pytest.mark.graph
def test_signed_mirror_repair_applies_scalar_and_projection_classes_exactly(
    disposable_neo4j: tuple[str, str],
) -> None:
    client = _client(disposable_neo4j, "semantic-mirror-repair")
    scalar = _seed_mirror_mismatch(
        client,
        "mirrorscalar",
        scalar_matches=False,
        projection_present=True,
    )
    projection = _seed_mirror_mismatch(
        client,
        "mirrorprojection",
        scalar_matches=True,
        projection_present=False,
    )
    both = _seed_mirror_mismatch(
        client,
        "mirrorboth",
        scalar_matches=False,
        projection_present=False,
    )
    outside = _seed_mirror_mismatch(
        client,
        "mirroroutside",
        scalar_matches=True,
        projection_present=True,
    )
    preview = None
    selected = sorted([scalar["source"], projection["source"], both["source"]])
    try:
        outside_before = _snapshot(client, list(outside.values()))
        unsourced_before = client.query(
            "MATCH (name:StandardName) "
            "WHERE NOT coalesce(name.name_stage, '') IN "
            "['superseded', 'exhausted', 'contested'] "
            "AND NOT EXISTS { "
            "MATCH (:StandardNameSource)-[:PRODUCED_NAME]->(name) } "
            "RETURN count(name) AS count"
        )[0]["count"]
        preview = repair_scalar_projection_mismatches(
            selected,
            reason="restore sole-live-target scalar and upstream projection parity",
            gc=client,
        )
        assert preview["outcome"] == "would_apply"
        assert preview["counts"] == {
            "requested": 3,
            "admitted": 3,
            "refused": 0,
            "already_clean": 0,
            "scalars_to_change": 2,
            "projections_to_add": 2,
        }
        assert _snapshot(client, list(outside.values())) == outside_before

        applied = repair_scalar_projection_mismatches(
            selected,
            reason="restore sole-live-target scalar and upstream projection parity",
            apply=True,
            manifest_sha256=preview["manifest_sha256"],
            gc=client,
        )
        assert applied["outcome"] == "applied"
        assert applied["sources_reconciled"] == 3
        assert applied["scalars_changed"] == 2
        assert applied["projections_added"] == 2
        assert applied["change_id"] == (
            "sn-change:semantic-mirror-repair:" + preview["manifest_sha256"]
        )
        assert client.query(
            "UNWIND $source_ids AS source_id "
            "MATCH (source:StandardNameSource {id: source_id}) "
            "MATCH (source)-[:PRODUCED_NAME]->(target:StandardName) "
            "MATCH (source)-[:FROM_DD_PATH]->(backing:IMASNode) "
            "RETURN source.id AS source_id, source.produced_sn_id AS scalar, "
            "collect(DISTINCT target.id) AS targets, "
            "COUNT { (backing)-[:HAS_STANDARD_NAME]->(target) } AS projections "
            "ORDER BY source_id",
            source_ids=selected,
        ) == [
            {
                "source_id": item["source"],
                "scalar": item["target"],
                "targets": [item["target"]],
                "projections": 1,
            }
            for item in sorted(
                [both, projection, scalar], key=lambda item: item["source"]
            )
        ]
        assert _snapshot(client, list(outside.values())) == outside_before
        assert (
            client.query(
                "MATCH (name:StandardName) "
                "WHERE NOT coalesce(name.name_stage, '') IN "
                "['superseded', 'exhausted', 'contested'] "
                "AND NOT EXISTS { "
                "MATCH (:StandardNameSource)-[:PRODUCED_NAME]->(name) } "
                "RETURN count(name) AS count"
            )[0]["count"]
            == unsourced_before
        )

        replay_before = _snapshot(
            client,
            [
                value
                for item in [scalar, projection, both, outside]
                for value in item.values()
            ],
        )
        replay = repair_scalar_projection_mismatches(
            selected,
            reason="restore sole-live-target scalar and upstream projection parity",
            apply=True,
            manifest_sha256=preview["manifest_sha256"],
            gc=client,
        )
        assert replay["outcome"] == "already_applied"
        assert replay["changed"] == 0
        assert (
            _snapshot(
                client,
                [
                    value
                    for item in [scalar, projection, both, outside]
                    for value in item.values()
                ],
            )
            == replay_before
        )
    finally:
        _cleanup(
            client,
            [scalar, projection, both, outside],
            (preview or {}).get("manifest_sha256"),
        )


@pytest.mark.graph
def test_signed_mirror_repair_refuses_non_unique_live_target_without_writes(
    disposable_neo4j: tuple[str, str],
) -> None:
    client = _client(disposable_neo4j, "semantic-mirror-ambiguity-refusal")
    ids = _seed(client, "mirrorambiguous", scalar="mirrorambiguous_semantic")
    try:
        before = _snapshot(client, list(ids.values()))
        preview = repair_scalar_projection_mismatches(
            [ids["source"]],
            reason="refuse mirror authority until one live target remains",
            gc=client,
        )
        assert preview["outcome"] == "refused"
        assert preview["counts"]["admitted"] == 0
        assert preview["counts"]["refused"] == 1
        assert preview["refusals"] == [
            {
                "source_id": ids["source"],
                "reason": "source does not have exactly one live target",
            }
        ]
        assert _snapshot(client, list(ids.values())) == before
    finally:
        _cleanup(client, [ids])


@pytest.mark.graph
def test_signed_mirror_repair_refuses_projection_drift_after_preview(
    disposable_neo4j: tuple[str, str],
) -> None:
    client = _client(disposable_neo4j, "semantic-mirror-projection-drift")
    ids = _seed_mirror_mismatch(
        client,
        "mirrorprojectiondrift",
        scalar_matches=True,
        projection_present=False,
    )
    preview = None
    try:
        preview = repair_scalar_projection_mismatches(
            [ids["source"]],
            reason="bind the exact missing upstream projection",
            gc=client,
        )
        client.query(
            "MATCH (backing:IMASNode {id: $path}), "
            "(target:StandardName {id: $target}) "
            "CREATE (backing)-[:HAS_STANDARD_NAME]->(target)",
            **ids,
        )
        before = _snapshot(client, list(ids.values()))
        with pytest.raises(
            SemanticMirrorRepairConflict,
            match="fresh semantic-mirror manifest does not match signed hash",
        ):
            repair_scalar_projection_mismatches(
                [ids["source"]],
                reason="bind the exact missing upstream projection",
                apply=True,
                manifest_sha256=preview["manifest_sha256"],
                gc=client,
            )
        assert _snapshot(client, list(ids.values())) == before
    finally:
        _cleanup(client, [ids], (preview or {}).get("manifest_sha256"))


def test_tampered_adjudication_is_rejected_before_graph_access() -> None:
    ids = {
        "source": "dd:tampered/value",
        "path": "tampered/value",
        "catalog": "tampered_catalog",
        "semantic": "tampered_semantic",
    }
    adjudication = _adjudication([_row(ids, "select_missing_scalar", None)])
    adjudication["rows"][0]["surviving_target"] = ids["catalog"]
    with pytest.raises(ValueError, match="adjudication signature does not match"):
        apply_adjudicated_source_dispositions(
            adjudication,
            reason="a tampered signed disposition must fail before graph access",
        )


def test_committed_adjudication_signature_contract_is_pinned() -> None:
    artifact_path = (
        Path(__file__).parents[2] / "docs/evidence/sn-graph-wide-integrity/"
        "catalog-edit-dual-binding-adjudication.json"
    )
    adjudication = json.loads(artifact_path.read_text())
    payload = {key: value for key, value in adjudication.items() if key != "signature"}
    assert _catalog_edit_adjudication_signature_hash(payload) == (
        "c227e70ec5cd940577ca778ce5ec63e4df3a63bf68c3e845eba92d0a4b9a0efb"
    )
    assert len(adjudication["rows"]) == 216
    assert all(
        _catalog_edit_adjudication_signature_hash(
            {key: value for key, value in row.items() if key != "row_signature_sha256"}
        )
        == row["row_signature_sha256"]
        for row in adjudication["rows"]
    )
