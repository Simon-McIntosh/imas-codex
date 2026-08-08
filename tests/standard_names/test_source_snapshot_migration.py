"""Unit checks for governed DD source snapshot migration."""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from imas_codex.standard_names import (
    source_authority as authority,
    source_snapshot_migration as migration,
)
from imas_codex.standard_names.source_snapshot_migration import (
    SourceSnapshotAllowlist,
    canonical_payload,
    classify_snapshot_change,
    load_source_snapshot_allowlist,
    migrate_source_snapshots,
)


def _record(
    source_id: str,
    *,
    scope_status: str = "executable",
    next_operator: str = "bounded_review",
    west: bool = False,
    test: bool = False,
) -> dict[str, object]:
    return {
        "scope_status": scope_status,
        "next_operator": next_operator,
        "participants": {"source_ids": [source_id], "name_ids": []},
        "scope_evidence": {
            "direct_west_source_hits": [source_id] if west else [],
            "direct_test_source_hits": [source_id] if test else [],
            "west_component_hits": ["component:west"] if west else [],
            "test_component_hits": ["component:test"] if test else [],
        },
    }


def _write_manifest(path: Path, records: list[dict[str, object]]) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema": "imas-codex.bounded-integrity-manifest",
                "schema_version": 2,
                "partitions": {"provenance": records},
                "special_checks": {
                    "declared_defect": {
                        "classification": "executable",
                        "next_operator": "DDGap_flag",
                        "source_id": "dd:declared/defect",
                    }
                },
            }
        )
    )
    return path


def test_allowlist_is_exact_and_excludes_deferred_or_defect_sources(
    tmp_path: Path,
) -> None:
    manifest = _write_manifest(
        tmp_path / "bounded.json",
        [
            _record("dd:kept/path"),
            _record("dd:test/path", test=True),
            _record("dd:record/defect", next_operator="DDGap_flag"),
            _record("dd:declared/defect"),
            _record("dd:deferred/path", scope_status="deferred_review"),
            _record("signals:facility:not-dd"),
        ],
    )

    allowlist = load_source_snapshot_allowlist(manifest)

    assert allowlist.source_ids == ("dd:kept/path",)
    assert allowlist.paths == ("kept/path",)
    assert len(allowlist.allowlist_hash) == 64
    assert allowlist.excluded_counts == {
        "dd_gap": 2,
        "non_dd": 1,
        "non_executable": 1,
        "test": 1,
    }
    assert allowlist.excluded_source_ids["test"] == ("dd:test/path",)


def test_allowlist_admits_facility_batch_sources(tmp_path: Path) -> None:
    """Facility batch membership is repairable, so it never subtracts a source."""
    manifest = _write_manifest(
        tmp_path / "bounded.json",
        [_record("dd:kept/path"), _record("dd:facility/path", west=True)],
    )

    allowlist = load_source_snapshot_allowlist(manifest)

    assert allowlist.source_ids == ("dd:facility/path", "dd:kept/path")
    assert "west" not in allowlist.excluded_counts
    assert "west" not in allowlist.excluded_source_ids


@pytest.mark.parametrize("protected_first", [False, True])
def test_allowlist_globally_subtracts_protected_source_regardless_of_order(
    tmp_path: Path, protected_first: bool
) -> None:
    duplicate = "dd:duplicated/protected"
    clean = _record(duplicate)
    protected = _record(duplicate, test=True)
    duplicate_records = [protected, clean] if protected_first else [clean, protected]
    manifest = _write_manifest(
        tmp_path / "bounded.json",
        [_record("dd:kept/path"), *duplicate_records],
    )

    allowlist = load_source_snapshot_allowlist(manifest)

    assert allowlist.source_ids == ("dd:kept/path",)
    assert allowlist.excluded_counts["test"] == 1
    assert allowlist.excluded_source_ids["test"] == (duplicate,)


def test_allowlist_subtracts_sources_protected_by_special_check(tmp_path: Path) -> None:
    manifest = _write_manifest(
        tmp_path / "bounded.json",
        [_record("dd:kept/path"), _record("dd:special/protected")],
    )
    payload = json.loads(manifest.read_text())
    payload["special_checks"]["protected_source"] = {
        "classification": "deferred_test_closure",
        "source_id": "dd:special/protected",
        "target_identity_test_closure": True,
    }
    manifest.write_text(json.dumps(payload))

    allowlist = load_source_snapshot_allowlist(manifest)

    assert allowlist.source_ids == ("dd:kept/path",)
    assert allowlist.excluded_source_ids["test"] == ("dd:special/protected",)


@pytest.mark.parametrize(
    "source_id",
    ["dd:", "dd:/leading", "dd:wild/*", "dd:space path", "dd:nested:identity"],
)
def test_allowlist_rejects_non_exact_dd_identity(
    tmp_path: Path, source_id: str
) -> None:
    manifest = _write_manifest(tmp_path / "bounded.json", [_record(source_id)])

    with pytest.raises(ValueError, match="exact DD source identity"):
        load_source_snapshot_allowlist(manifest)


def test_canonical_payload_preserves_types_and_normalizes_order() -> None:
    first = {"set": {"b", "a"}, "values": [2, 1], "flag": True, "none": None}
    second = {"none": None, "flag": True, "values": [2, 1], "set": {"a", "b"}}

    assert canonical_payload(first) == canonical_payload(second)


def test_source_authority_closure_has_exact_identity_and_participants() -> None:
    source = {
        "element_id": "source-element",
        "labels": ["StandardNameSource"],
        "properties": {
            "id": "dd:diagnostic/path",
            "source_type": "dd",
            "source_id": "diagnostic/path",
            "dd_version": "old-dd",
            "dd_snapshot_pinned": True,
            "batch_key": "preserved",
        },
        "relationships": [
            {
                "element_id": "source-node-link",
                "type": "FROM_DD_PATH",
                "direction": "out",
                "properties": {"owner": "preserved"},
                "other_element_id": "node-element",
                "other_labels": ["IMASNode"],
                "other_id": "diagnostic/path",
                "other_properties": {"id": "diagnostic/path"},
            }
        ],
        "ledger": [],
        "names": [],
    }
    node = {
        "element_id": "node-element",
        "labels": ["IMASNode"],
        "properties": {
            "id": "diagnostic/path",
            "documentation": "authoritative documentation",
            "unit": "Pa",
        },
        "units": [
            {
                "element_id": "unit-element",
                "labels": ["Unit"],
                "id": "Pa",
                "properties": {"id": "Pa"},
            }
        ],
        "parents": [],
        "coordinates": [],
        "projections": [],
    }
    row = {
        "path": "diagnostic/path",
        "versions": [
            {
                "element_id": "version-element",
                "labels": ["DDVersion"],
                "properties": {"id": "new-dd", "is_current": True},
            }
        ],
        "sources": [source],
        "nodes": [node],
    }

    closure = authority.capture_source_authority_closure(
        row,
        manifest_hash="manifest-digest",
        authorized_source_ids=frozenset({"dd:diagnostic/path"}),
    )

    assert closure.identity_payload == {
        "stable_id": "dd:diagnostic/path",
        "source_type": "dd",
        "source_id": "diagnostic/path",
        "from_dd_paths": [
            {
                "element_id": "source-node-link",
                "properties": {"owner": "preserved"},
                "other_element_id": "node-element",
                "other_labels": ["IMASNode"],
                "other_id": "diagnostic/path",
            }
        ],
    }
    assert closure.participant_ids == (
        "node-element",
        "source-element",
        "unit-element",
        "version-element",
    )
    assert closure.after_snapshot["dd_version"] == "new-dd"
    assert closure.after_snapshot["dd_unit"] == "Pa"
    assert closure.before_snapshot["dd_version"] == "old-dd"
    assert len(closure.authority_hash) == 64
    assert len(closure.precondition_hash) == 64
    assert len(closure.preserved_state_hash) == 64
    assert len(closure.participant_ids_hash) == 64


def test_snapshot_classification_distinguishes_byte_semantic_and_material() -> None:
    base = {
        "description": "operational mirror",
        "dd_documentation": "authoritative documentation",
        "dd_unit": "W.m^-2",
        "physics_domain": "wall",
    }

    assert classify_snapshot_change("wall/path", base, dict(base)) == "byte_unchanged"
    mirror_refresh = {**base, "description": "refreshed operational mirror"}
    assert (
        classify_snapshot_change("wall/path", base, mirror_refresh) == "byte_unchanged"
    )
    assert (
        sha256(canonical_payload(base).encode()).hexdigest()
        != sha256(canonical_payload(mirror_refresh).encode()).hexdigest()
    )
    assert (
        classify_snapshot_change(
            "wall/path",
            {**base, "dd_unit": "Hz"},
            {**base, "dd_unit": "s^-1"},
        )
        == "semantic_unchanged"
    )
    assert (
        classify_snapshot_change(
            "wall/path",
            {**base, "dd_unit": "Hz"},
            {
                **base,
                "description": "refreshed operational mirror",
                "dd_unit": "s^-1",
            },
        )
        == "semantic_unchanged"
    )
    assert (
        classify_snapshot_change(
            "wall/path",
            base,
            {
                **base,
                "description": "refreshed operational mirror",
                "physics_domain": "transport",
            },
        )
        == "changed"
    )
    assert (
        classify_snapshot_change(
            "wall/path",
            base,
            {
                **base,
                "description": "refreshed operational mirror",
                "dd_documentation": "changed authoritative documentation",
            },
        )
        == "changed"
    )


def test_preserved_state_normalizes_only_authorized_peer_snapshot_fields() -> None:
    source = {
        "element_id": "source-one-element",
        "labels": ["StandardNameSource"],
        "properties": {
            "id": "dd:shared/one",
            "dd_version": "old-dd",
            "description": "old documentation",
            "batch_key": "source-one-batch",
        },
        "relationships": [],
        "names": [
            {
                "element_id": "shared-name-element",
                "properties": {"id": "shared_name"},
                "relationships": [
                    {
                        "other_element_id": "source-one-element",
                        "other_labels": ["StandardNameSource"],
                        "other_id": "dd:shared/one",
                        "other_properties": {
                            "id": "dd:shared/one",
                            "dd_version": "old-dd",
                            "description": "old documentation",
                            "batch_key": "source-one-batch",
                        },
                    },
                    {
                        "other_element_id": "source-two-element",
                        "other_labels": ["StandardNameSource"],
                        "other_id": "dd:shared/two",
                        "other_properties": {
                            "id": "dd:shared/two",
                            "dd_version": "old-dd",
                            "description": "old documentation",
                            "batch_key": "source-two-batch",
                        },
                    },
                    {
                        "other_element_id": "external-source-element",
                        "other_labels": ["StandardNameSource"],
                        "other_id": "dd:external/path",
                        "other_properties": {
                            "id": "dd:external/path",
                            "dd_version": "old-dd",
                            "description": "external documentation",
                            "batch_key": "external-batch",
                        },
                    },
                ],
            }
        ],
    }

    preserved = authority.preserved_state(
        source,
        {"projections": []},
        authorized_source_ids=frozenset({"dd:shared/one", "dd:shared/two"}),
    )

    related = {
        item["other_id"]: item["other_properties"]
        for item in preserved["names"][0]["relationships"]
    }
    assert preserved["source_properties"] == {
        "id": "dd:shared/one",
        "batch_key": "source-one-batch",
    }
    assert related["dd:shared/one"] == {
        "id": "dd:shared/one",
        "batch_key": "source-one-batch",
    }
    assert related["dd:shared/two"] == {
        "id": "dd:shared/two",
        "batch_key": "source-two-batch",
    }
    assert related["dd:external/path"] == {
        "id": "dd:external/path",
        "dd_version": "old-dd",
        "description": "external documentation",
        "batch_key": "external-batch",
    }


def test_preserved_state_accepts_an_operation_specific_mutable_field() -> None:
    preserved = authority.preserved_state(
        {
            "element_id": "source-element",
            "labels": ["StandardNameSource"],
            "properties": {
                "id": "dd:diagnostic/path",
                "source_id": None,
                "dd_version": "old-dd",
                "batch_key": "preserved",
            },
            "relationships": [],
            "names": [],
        },
        {"projections": []},
        authorized_source_ids=frozenset({"dd:diagnostic/path"}),
        mutable_source_fields=frozenset({"source_id"}),
    )

    assert preserved["source_properties"] == {
        "id": "dd:diagnostic/path",
        "dd_version": "old-dd",
        "batch_key": "preserved",
    }


def test_receipt_bytes_remain_stable_for_one_exact_row() -> None:
    allowlist = SourceSnapshotAllowlist(
        manifest_path=Path("/tmp/exact-source-manifest.json"),
        manifest_hash="a" * 64,
        source_ids=("dd:diagnostic/path",),
        paths=("diagnostic/path",),
        allowlist_hash="b" * 64,
        excluded_counts={"test": 1},
        excluded_source_ids={"test": ("dd:test/path",)},
    )
    planned = [
        {
            "source_id": "dd:diagnostic/path",
            "path": "diagnostic/path",
            "status": "planned",
            "classification": "changed",
            "before_snapshot_hash": "c" * 64,
            "after_snapshot_hash": "d" * 64,
            "authority_hash": "e" * 64,
            "precondition_hash": "f" * 64,
            "preserved_state_hash": "0" * 64,
            "event": {"id": "source-snapshot-change:exact"},
        }
    ]

    receipt = migration._receipt(
        allowlist,
        planned,
        [],
        apply=False,
        run_id=None,
    )
    receipt_bytes = json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode()

    assert (
        sha256(receipt_bytes).hexdigest()
        == "cddeae3708bbf075e0d5d899a24cf85cd431b90ebcd2e96bf58b73189c4c50a4"
    )
    assert (
        receipt["receipt_hash"]
        == "11e802cf7f4b7684a5fdd1123fcbfa9d4e4bcea2fe4f0cf546e278dcff916739"
    )


@pytest.mark.parametrize("expected_hash", [None, "not-a-sha256", "0" * 64])
def test_apply_rejects_unbound_manifest_before_graph_access(
    tmp_path: Path, expected_hash: str | None
) -> None:
    manifest = _write_manifest(tmp_path / "bounded.json", [_record("dd:kept/path")])
    with (
        patch(
            "imas_codex.standard_names.source_snapshot_migration.GraphClient"
        ) as graph_client,
        pytest.raises(ValueError, match="manifest SHA-256"),
    ):
        migrate_source_snapshots(
            manifest,
            expected_from_version="old-dd",
            reason="refresh immutable authority",
            apply=True,
            expected_manifest_hash=expected_hash,
        )

    graph_client.assert_not_called()


def test_correct_manifest_hash_reaches_graph_session(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path / "bounded.json", [_record("dd:kept/path")])
    expected_hash = sha256(manifest.read_bytes()).hexdigest()
    graph = Mock()
    graph.session.side_effect = RuntimeError("graph reached")

    with pytest.raises(RuntimeError, match="graph reached"):
        migrate_source_snapshots(
            manifest,
            expected_from_version="old-dd",
            reason="refresh immutable authority",
            apply=True,
            expected_manifest_hash=expected_hash,
            gc=graph,
        )

    graph.session.assert_called_once_with()


def test_apply_rejects_manifest_changed_after_prior_plan(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path / "bounded.json", [_record("dd:kept/path")])
    planned_hash = sha256(manifest.read_bytes()).hexdigest()
    payload = json.loads(manifest.read_text())
    payload["generated_at"] = "changed after planning"
    manifest.write_text(json.dumps(payload))
    with (
        patch(
            "imas_codex.standard_names.source_snapshot_migration.GraphClient"
        ) as graph_client,
        pytest.raises(ValueError, match="does not match"),
    ):
        migrate_source_snapshots(
            manifest,
            expected_from_version="old-dd",
            reason="refresh immutable authority",
            apply=True,
            expected_manifest_hash=planned_hash,
        )

    graph_client.assert_not_called()
