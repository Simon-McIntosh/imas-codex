"""Unit contracts for exact source-authority reconciliation."""

from __future__ import annotations

import copy
import json
from hashlib import sha256
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from imas_codex.standard_names import source_authority_reconciliation as reconciliation
from imas_codex.standard_names.source_authority import (
    SNAPSHOT_FIELDS,
    SNAPSHOT_MUTABLE_FIELDS,
    capture_source_authority_closure,
    participant_ids,
    payload_hash,
)


def _base_row(
    *,
    source_id_value: str | None,
    snapshot: str,
    node_category: str = "quantity",
    target_id: str | None = None,
    target_stage: str | None = None,
) -> dict[str, object]:
    path = "diagnostic/channel/value"
    node = {
        "element_id": "node-element",
        "labels": ["IMASNode"],
        "properties": {
            "id": path,
            "documentation": "Authoritative DD documentation",
            "description": "Enhanced DD description",
            "physics_domain": "diagnostics",
            "data_type": "FLT_0D",
            "unit": "V",
            "node_category": node_category,
            "lifecycle_status": "active",
            "lifecycle_version": "4.1.1",
            "enrichment_source": "template",
        },
        "units": [
            {
                "relationship_element_id": "unit-link",
                "relationship_properties": {},
                "element_id": "unit-element",
                "labels": ["Unit"],
                "id": "V",
                "properties": {"id": "V"},
            }
        ],
        "parents": [],
        "coordinates": [],
        "projections": [],
    }
    source_properties: dict[str, object] = {
        "id": f"dd:{path}",
        "source_type": "dd",
        "source_id": source_id_value,
        "status": "attached" if target_id else "failed",
        "produced_sn_id": target_id,
        "batch_key": "preserved",
    }
    if snapshot == "current":
        source_properties.update(
            {
                "dd_version": "4.1.1",
                "description": "Authoritative DD documentation",
                "physics_domain": "diagnostics",
                "dd_documentation": "Authoritative DD documentation",
                "dd_snapshot_pinned": True,
                "dd_parent_path": None,
                "dd_parent_documentation": None,
                "dd_data_type": "FLT_0D",
                "dd_unit": "V",
                "dd_coordinates": [],
                "dd_lifecycle_status": "active",
                "dd_lifecycle_version": "4.1.1",
                "enhanced_description": "Enhanced DD description",
                "enhancement_kind": "template",
            }
        )
    elif snapshot == "adopt":
        source_properties.update(
            {
                "dd_version": "4.1.1",
                "description": "Stale operational mirror",
                "physics_domain": "diagnostics",
                "dd_documentation": "Authoritative DD documentation",
                "dd_snapshot_pinned": True,
                "dd_parent_path": None,
                "dd_parent_documentation": None,
                "dd_data_type": "FLT_0D",
                "dd_unit": "V",
                "dd_coordinates": [],
                "dd_lifecycle_status": "active",
                "dd_lifecycle_version": "4.1.1",
                "enhanced_description": "Enhanced DD description",
                "enhancement_kind": "template",
            }
        )
    elif snapshot == "old":
        source_properties.update(
            {
                "dd_version": "4.1.0",
                "description": "Older DD documentation",
                "dd_snapshot_pinned": True,
            }
        )
    elif snapshot == "null":
        source_properties.update(dict.fromkeys(SNAPSHOT_FIELDS))
    else:
        raise AssertionError(snapshot)

    source = {
        "element_id": "source-element",
        "labels": ["StandardNameSource"],
        "properties": source_properties,
        "relationships": [
            {
                "element_id": "source-node-link",
                "type": "FROM_DD_PATH",
                "direction": "out",
                "properties": {"authority": "preserved"},
                "other_element_id": "node-element",
                "other_labels": ["IMASNode"],
                "other_id": path,
                "other_properties": node["properties"],
            }
        ],
        "ledger": [],
        "names": [],
    }
    if target_id:
        name = {
            "binding_element_id": "binding-element",
            "binding_properties": {},
            "element_id": "name-element",
            "labels": ["StandardName"],
            "properties": {"id": target_id, "name_stage": target_stage},
            "relationships": [],
        }
        source["relationships"].append(
            {
                "element_id": "binding-element",
                "type": "PRODUCED_NAME",
                "direction": "out",
                "properties": {},
                "other_element_id": "name-element",
                "other_labels": ["StandardName"],
                "other_id": target_id,
                "other_properties": name["properties"],
            }
        )
        source["names"].append(name)
        node["projections"].append(
            {
                "relationship_element_id": "projection-element",
                "relationship_properties": {},
                "element_id": "name-element",
                "labels": ["StandardName"],
                "id": target_id,
                "properties": name["properties"],
            }
        )
    return {
        "path": path,
        "versions": [
            {
                "element_id": "version-element",
                "labels": ["DDVersion"],
                "properties": {"id": "4.1.1", "is_current": True},
            }
        ],
        "sources": [source],
        "nodes": [node],
    }


def _manifest_for(
    row: dict[str, object],
    operation: str,
    *,
    duplicate: dict[str, object] | None = None,
    target_id: str | None = None,
) -> reconciliation.SourceAuthorityManifest:
    path = str(row["path"])
    operational = reconciliation._without_authority_relationships(row)
    preserved = operational
    mutable = SNAPSHOT_MUTABLE_FIELDS
    extras: dict[str, object] = {}
    if operation == reconciliation.REPAIR_IDENTITY_SCALAR:
        mutable = frozenset({"source_id"})
    elif operation == reconciliation.RETIRE_NONPARTICIPATING_SOURCE:
        assert target_id is not None
        preserved = reconciliation._retirement_preserved_row(row, target_id)
        mutable = frozenset(
            {
                "status",
                "produced_sn_id",
                "claimed_at",
                "claim_token",
                "drain_scope_id",
                "drain_scope_claimed_at",
                "drain_claim_scope_id",
                "drain_scope_actionable",
                "skip_reason",
                "skip_reason_detail",
            }
        )
        extras = {
            "expected_node_category": row["nodes"][0]["properties"]["node_category"],
            "expected_target_id": target_id,
            "expected_retirement_destructive_closure_hash": payload_hash(
                reconciliation._retirement_destructive_closure(row, target_id)
            ),
        }
    closure = capture_source_authority_closure(
        operational,
        manifest_hash="manifest-hash",
        authorized_source_ids=frozenset({f"dd:{path}"}),
        mutable_source_fields=mutable,
    )
    preserved_closure = capture_source_authority_closure(
        preserved,
        manifest_hash="manifest-hash",
        authorized_source_ids=frozenset({f"dd:{path}"}),
        mutable_source_fields=mutable,
    )
    if operation == reconciliation.FOLD_DUPLICATE_SOURCE_IDENTITY:
        assert duplicate is not None
        duplicate_identity = reconciliation.source_identity_payload(duplicate)
        extras = {
            "duplicate_source_id": duplicate["properties"]["id"],
            "expected_duplicate_source_element_id": duplicate["element_id"],
            "expected_duplicate_source_id": duplicate_identity["source_id"],
            "expected_duplicate_from_dd_path": path,
            "expected_duplicate_preserved_state_hash": payload_hash(
                reconciliation._duplicate_preserved_state(duplicate)
            ),
            "expected_duplicate_destructive_closure_hash": payload_hash(
                reconciliation._duplicate_destructive_closure(duplicate)
            ),
        }
    manifest_row = {
        "source_id": f"dd:{path}",
        "operation": operation,
        "expected_source_element_id": "source-element",
        "expected_source_id": row["sources"][0]["properties"]["source_id"],
        "expected_from_dd_path": path,
        "expected_before_snapshot_hash": payload_hash(closure.before_snapshot),
        "expected_authority_hash": closure.authority_hash,
        "expected_preserved_state_hash": preserved_closure.preserved_state_hash,
        "expected_participant_ids_hash": payload_hash(
            tuple(sorted(participant_ids(row)))
        ),
        "west_intersection": 0,
        "test_intersection": 0,
        **extras,
    }
    return reconciliation.SourceAuthorityManifest(
        path=Path("/tmp/authority.json"),
        manifest_hash="manifest-hash",
        operation=operation,
        rows=(manifest_row,),
        source_ids=(f"dd:{path}",),
        paths=(path,),
        allowlist_hash=payload_hash((f"dd:{path}",)),
    )


def _write_manifest(path: Path, rows: list[dict[str, object]]) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema": "imas-codex.source-authority-reconciliation-manifest",
                "schema_version": 1,
                "rows": rows,
            }
        )
    )
    return path


def _minimal_manifest_row(operation: str, source_id: str) -> dict[str, object]:
    path = source_id.removeprefix("dd:")
    row: dict[str, object] = {
        "source_id": source_id,
        "operation": operation,
        "expected_source_element_id": "source-element",
        "expected_source_id": path,
        "expected_from_dd_path": path,
        "expected_before_snapshot_hash": "a" * 64,
        "expected_authority_hash": "b" * 64,
        "expected_preserved_state_hash": "c" * 64,
        "expected_participant_ids_hash": "d" * 64,
        "west_intersection": 0,
        "test_intersection": 0,
    }
    if operation == reconciliation.FOLD_DUPLICATE_SOURCE_IDENTITY:
        row.update(
            {
                "duplicate_source_id": "dd:duplicate/path",
                "expected_duplicate_source_element_id": "duplicate-element",
                "expected_duplicate_source_id": path,
                "expected_duplicate_from_dd_path": path,
                "expected_duplicate_preserved_state_hash": "e" * 64,
                "expected_duplicate_destructive_closure_hash": "f" * 64,
            }
        )
    elif operation == reconciliation.RETIRE_NONPARTICIPATING_SOURCE:
        row.update(
            {
                "expected_node_category": "structural",
                "expected_target_id": "placeholder_name",
                "expected_retirement_destructive_closure_hash": "e" * 64,
            }
        )
    return row


def test_manifest_requires_one_homogeneous_operation(tmp_path: Path) -> None:
    manifest = _write_manifest(
        tmp_path / "mixed.json",
        [
            _minimal_manifest_row(
                reconciliation.REPAIR_IDENTITY_SCALAR, "dd:first/path"
            ),
            _minimal_manifest_row(
                reconciliation.ADOPT_CURRENT_SNAPSHOT, "dd:second/path"
            ),
        ],
    )

    with pytest.raises(ValueError, match="homogeneous"):
        reconciliation.load_source_authority_manifest(manifest)


def test_manifest_rejects_protected_and_nonexact_rows(tmp_path: Path) -> None:
    row = _minimal_manifest_row(
        reconciliation.REPAIR_IDENTITY_SCALAR, "dd:diagnostic/path"
    )
    row["west_intersection"] = 1
    manifest = _write_manifest(tmp_path / "protected.json", [row])
    with pytest.raises(ValueError, match="intersections"):
        reconciliation.load_source_authority_manifest(manifest)

    row["west_intersection"] = 0
    row["unexpected"] = True
    manifest = _write_manifest(tmp_path / "extra.json", [row])
    with pytest.raises(ValueError, match="fields are not exact"):
        reconciliation.load_source_authority_manifest(manifest)


def test_fold_manifest_requires_globally_disjoint_participants(tmp_path: Path) -> None:
    first = _minimal_manifest_row(
        reconciliation.FOLD_DUPLICATE_SOURCE_IDENTITY, "dd:first/path"
    )
    second = _minimal_manifest_row(
        reconciliation.FOLD_DUPLICATE_SOURCE_IDENTITY, "dd:second/path"
    )
    second["duplicate_source_id"] = first["duplicate_source_id"]
    repeated = _write_manifest(tmp_path / "repeated.json", [first, second])
    with pytest.raises(ValueError, match="globally unique"):
        reconciliation.load_source_authority_manifest(repeated)

    second["duplicate_source_id"] = first["source_id"]
    overlapping = _write_manifest(tmp_path / "overlap.json", [first, second])
    with pytest.raises(ValueError, match="disjoint"):
        reconciliation.load_source_authority_manifest(overlapping)


def test_apply_rejects_changed_manifest_before_graph_access(tmp_path: Path) -> None:
    row = _minimal_manifest_row(
        reconciliation.REPAIR_IDENTITY_SCALAR, "dd:diagnostic/path"
    )
    manifest = _write_manifest(tmp_path / "authority.json", [row])
    prior_hash = sha256(manifest.read_bytes()).hexdigest()
    payload = json.loads(manifest.read_text())
    payload["rows"][0]["expected_authority_hash"] = "f" * 64
    manifest.write_text(json.dumps(payload))

    with (
        patch.object(reconciliation, "GraphClient") as graph_client,
        pytest.raises(ValueError, match="does not match"),
    ):
        reconciliation.reconcile_source_authority(
            manifest,
            reason="repair exact authority",
            apply=True,
            expected_manifest_hash=prior_hash,
        )

    graph_client.assert_not_called()


def test_identity_repair_plans_then_requires_exact_event_for_idempotence() -> None:
    row = _base_row(source_id_value=None, snapshot="old")
    manifest = _manifest_for(row, reconciliation.REPAIR_IDENTITY_SCALAR)
    source_id = manifest.source_ids[0]

    plans, refusals = reconciliation._plan_rows(
        [row],
        manifest,
        events_by_source={source_id: []},
        duplicates_by_source={},
        reason="repair identity scalar",
        run_id="run",
        changed_at="2026-08-03T00:00:00+00:00",
    )

    assert not refusals
    assert plans[0]["status"] == "planned"
    event = plans[0]["event"]
    repaired = copy.deepcopy(row)
    repaired["sources"][0]["properties"]["source_id"] = row["path"]
    repaired["sources"][0]["relationships"].append(
        {
            "element_id": "event-link",
            "type": "HAS_IDENTITY_REPAIR",
            "direction": "out",
            "properties": {},
            "other_element_id": "event-element",
            "other_labels": ["StandardNameSourceIdentityRepair"],
            "other_id": event["id"],
            "other_properties": event,
        }
    )
    event_entry = {
        "relationship_type": "HAS_IDENTITY_REPAIR",
        "event_labels": ["StandardNameSourceIdentityRepair"],
        "event_properties": event,
    }
    repeated, repeated_refusals = reconciliation._plan_rows(
        [repaired],
        manifest,
        events_by_source={source_id: [event_entry]},
        duplicates_by_source={},
        reason="repair identity scalar",
        run_id="new-run",
        changed_at="2026-08-03T00:01:00+00:00",
    )

    assert not repeated_refusals
    assert repeated[0]["status"] == "already_current"


@pytest.mark.parametrize(
    ("operation", "row", "expected_reason"),
    [
        (
            reconciliation.REPAIR_IDENTITY_SCALAR,
            _base_row(source_id_value="wrong/path", snapshot="old"),
            "source.source_id is not null",
        ),
        (
            reconciliation.ADOPT_CURRENT_SNAPSHOT,
            _base_row(source_id_value="diagnostic/channel/value", snapshot="old"),
            "same-version adoption requires the unique current DD version",
        ),
        (
            reconciliation.ADMIT_CURRENT_SNAPSHOT,
            _base_row(source_id_value="diagnostic/channel/value", snapshot="adopt"),
            "snapshot admission requires every prior snapshot field to be null",
        ),
    ],
)
def test_operation_specific_refusals_are_fail_closed(
    operation: str, row: dict[str, object], expected_reason: str
) -> None:
    manifest = _manifest_for(row, operation)
    _, refusals = reconciliation._plan_rows(
        [row],
        manifest,
        events_by_source={manifest.source_ids[0]: []},
        duplicates_by_source={},
        reason="govern exact authority",
        run_id="run",
        changed_at="2026-08-03T00:00:00+00:00",
    )

    assert expected_reason in refusals[0]["reasons"]


def test_duplicate_fold_rejects_a_semantic_target() -> None:
    row = _base_row(source_id_value="diagnostic/channel/value", snapshot="current")
    duplicate = copy.deepcopy(row["sources"][0])
    duplicate["element_id"] = "duplicate-element"
    duplicate["properties"]["id"] = "dd:legacy/duplicate"
    duplicate["properties"]["source_id"] = row["path"]
    duplicate["properties"]["status"] = "failed"
    duplicate["names"] = [
        {
            "element_id": "live-name-element",
            "properties": {"id": "live_name", "name_stage": "accepted"},
            "relationships": [],
        }
    ]
    manifest = _manifest_for(
        row,
        reconciliation.FOLD_DUPLICATE_SOURCE_IDENTITY,
        duplicate=duplicate,
    )

    _, refusals = reconciliation._plan_rows(
        [row],
        manifest,
        events_by_source={manifest.source_ids[0]: []},
        duplicates_by_source={duplicate["properties"]["id"]: [duplicate]},
        reason="retire duplicate owner",
        run_id="run",
        changed_at="2026-08-03T00:00:00+00:00",
    )

    assert "duplicate source owns a semantic target" in refusals[0]["reasons"]


def test_duplicate_fold_binds_destructive_relationship_properties() -> None:
    row = _base_row(source_id_value="diagnostic/channel/value", snapshot="current")
    duplicate = copy.deepcopy(row["sources"][0])
    duplicate["element_id"] = "duplicate-element"
    duplicate["properties"]["id"] = "dd:legacy/duplicate"
    duplicate["properties"]["source_id"] = row["path"]
    duplicate["properties"]["status"] = "failed"
    manifest = _manifest_for(
        row,
        reconciliation.FOLD_DUPLICATE_SOURCE_IDENTITY,
        duplicate=duplicate,
    )
    duplicate["relationships"][0]["properties"]["authority"] = "drifted"

    _, refusals = reconciliation._plan_rows(
        [row],
        manifest,
        events_by_source={manifest.source_ids[0]: []},
        duplicates_by_source={duplicate["properties"]["id"]: [duplicate]},
        reason="retire duplicate owner",
        run_id="run",
        changed_at="2026-08-03T00:00:00+00:00",
    )

    assert (
        "manifest expected_duplicate_destructive_closure_hash drifted"
        in refusals[0]["reasons"]
    )


def test_retirement_binds_replacement_relationship_identity() -> None:
    row = _base_row(
        source_id_value="diagnostic/channel/value",
        snapshot="current",
        node_category="structural",
        target_id="null_placeholder",
    )
    manifest = _manifest_for(
        row,
        reconciliation.RETIRE_NONPARTICIPATING_SOURCE,
        target_id="null_placeholder",
    )
    row["sources"][0]["relationships"][1]["element_id"] = "replacement-binding"
    row["sources"][0]["names"][0]["binding_element_id"] = "replacement-binding"
    row["nodes"][0]["projections"][0]["relationship_element_id"] = (
        "replacement-projection"
    )

    _, refusals = reconciliation._plan_rows(
        [row],
        manifest,
        events_by_source={manifest.source_ids[0]: []},
        duplicates_by_source={},
        reason="retire nonparticipating source",
        run_id="run",
        changed_at="2026-08-03T00:00:00+00:00",
    )

    assert (
        "manifest expected_retirement_destructive_closure_hash drifted"
        in refusals[0]["reasons"]
    )


def test_retirement_rejects_a_generation_participating_category() -> None:
    row = _base_row(
        source_id_value="diagnostic/channel/value",
        snapshot="current",
        node_category="quantity",
        target_id="null_placeholder",
    )
    manifest = _manifest_for(
        row,
        reconciliation.RETIRE_NONPARTICIPATING_SOURCE,
        target_id="null_placeholder",
    )

    _, refusals = reconciliation._plan_rows(
        [row],
        manifest,
        events_by_source={manifest.source_ids[0]: []},
        duplicates_by_source={},
        reason="retire nonparticipating source",
        run_id="run",
        changed_at="2026-08-03T00:00:00+00:00",
    )

    assert (
        "backing node category participates in standard-name generation"
        in refusals[0]["reasons"]
    )


def test_retirement_accepts_a_prior_snapshot_admission_event() -> None:
    row = _base_row(
        source_id_value="diagnostic/channel/value",
        snapshot="current",
        node_category="structural",
        target_id="null_placeholder",
    )
    manifest = _manifest_for(
        row,
        reconciliation.RETIRE_NONPARTICIPATING_SOURCE,
        target_id="null_placeholder",
    )
    admission_event = {
        "relationship_type": "HAS_SNAPSHOT_ADMISSION",
        "event_labels": ["StandardNameSourceSnapshotAdmission"],
        "event_properties": {"id": "source-snapshot-admission:prior"},
    }

    plans, refusals = reconciliation._plan_rows(
        [row],
        manifest,
        events_by_source={manifest.source_ids[0]: [admission_event]},
        duplicates_by_source={},
        reason="retire nonparticipating source",
        run_id="run",
        changed_at="2026-08-03T00:00:00+00:00",
    )

    assert not refusals
    assert plans[0]["status"] == "planned"


def test_retirement_idempotence_uses_the_authorized_pre_retirement_authority() -> None:
    row = _base_row(
        source_id_value="diagnostic/channel/value",
        snapshot="current",
        node_category="structural",
        target_id="null_placeholder",
    )
    manifest = _manifest_for(
        row,
        reconciliation.RETIRE_NONPARTICIPATING_SOURCE,
        target_id="null_placeholder",
    )
    planned, refusals = reconciliation._plan_rows(
        [row],
        manifest,
        events_by_source={manifest.source_ids[0]: []},
        duplicates_by_source={},
        reason="retire nonparticipating source",
        run_id="run",
        changed_at="2026-08-03T00:00:00+00:00",
    )
    assert not refusals
    event = planned[0]["event"]

    retired = copy.deepcopy(row)
    source = retired["sources"][0]
    source["properties"].update(
        {
            "status": "stale",
            "skip_reason": "nonparticipating_dd_source",
            "skip_reason_detail": "retire nonparticipating source",
        }
    )
    for field in (
        "produced_sn_id",
        "claimed_at",
        "claim_token",
        "drain_scope_id",
        "drain_scope_claimed_at",
        "drain_claim_scope_id",
        "drain_scope_actionable",
    ):
        source["properties"].pop(field, None)
    source["relationships"] = [source["relationships"][0]]
    source["names"] = []
    retired["nodes"][0]["projections"] = []
    event_entry = {
        "relationship_type": "HAS_AUTHORITY_RETIREMENT",
        "event_labels": ["StandardNameSourceAuthorityRetirement"],
        "event_properties": event,
    }

    repeated, repeated_refusals = reconciliation._plan_rows(
        [retired],
        manifest,
        events_by_source={manifest.source_ids[0]: [event_entry]},
        duplicates_by_source={},
        reason="retire nonparticipating source",
        run_id="repeat",
        changed_at="2026-08-03T00:01:00+00:00",
    )

    assert not repeated_refusals
    assert repeated[0]["status"] == "already_current"
    assert repeated[0]["authority_hash"] == event["authority_hash"]


def test_relationship_lock_requires_exact_cardinality() -> None:
    transaction = Mock()
    transaction.run.return_value = [{"locked": 1}]

    with pytest.raises(
        reconciliation.SourceAuthorityReconciliationConflict,
        match="relationship set changed",
    ):
        reconciliation._lock_relationships(
            transaction,
            {"relationship-one", "relationship-two"},
        )

    cypher = transaction.run.call_args.args[0]
    assert "SET relationship._source_authority_lock" in cypher
    assert "REMOVE relationship._source_authority_lock" in cypher
