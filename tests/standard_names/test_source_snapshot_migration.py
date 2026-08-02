"""Unit checks for governed DD source snapshot migration."""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from imas_codex.standard_names.source_snapshot_migration import (
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
            _record("dd:west/path", west=True),
            _record("dd:test/path", test=True),
            _record("dd:record/defect", next_operator="DDGap_flag"),
            _record("dd:declared/defect"),
            _record("dd:deferred/path", scope_status="deferred_west_closure"),
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
        "west": 2,
    }
    assert allowlist.excluded_source_ids["west"] == (
        "dd:deferred/path",
        "dd:west/path",
    )


@pytest.mark.parametrize("protected_kind", ["west", "test"])
@pytest.mark.parametrize("protected_first", [False, True])
def test_allowlist_globally_subtracts_protected_source_regardless_of_order(
    tmp_path: Path, protected_kind: str, protected_first: bool
) -> None:
    duplicate = "dd:duplicated/protected"
    clean = _record(duplicate)
    protected = _record(duplicate, **{protected_kind: True})
    duplicate_records = [protected, clean] if protected_first else [clean, protected]
    manifest = _write_manifest(
        tmp_path / "bounded.json",
        [_record("dd:kept/path"), *duplicate_records],
    )

    allowlist = load_source_snapshot_allowlist(manifest)

    assert allowlist.source_ids == ("dd:kept/path",)
    assert allowlist.excluded_counts[protected_kind] == 1
    assert allowlist.excluded_source_ids[protected_kind] == (duplicate,)


def test_allowlist_subtracts_sources_protected_by_special_check(tmp_path: Path) -> None:
    manifest = _write_manifest(
        tmp_path / "bounded.json",
        [_record("dd:kept/path"), _record("dd:special/protected")],
    )
    payload = json.loads(manifest.read_text())
    payload["special_checks"]["protected_source"] = {
        "classification": "deferred_west_closure",
        "source_id": "dd:special/protected",
        "target_identity_west_closure": True,
    }
    manifest.write_text(json.dumps(payload))

    allowlist = load_source_snapshot_allowlist(manifest)

    assert allowlist.source_ids == ("dd:kept/path",)
    assert allowlist.excluded_source_ids["west"] == ("dd:special/protected",)


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


def test_snapshot_classification_distinguishes_byte_semantic_and_material() -> None:
    base = {"dd_unit": "W.m^-2", "physics_domain": "wall"}

    assert classify_snapshot_change("wall/path", base, dict(base)) == "byte_unchanged"
    assert (
        classify_snapshot_change("wall/path", {"dd_unit": "Hz"}, {"dd_unit": "s^-1"})
        == "semantic_unchanged"
    )
    assert (
        classify_snapshot_change(
            "wall/path", base, {**base, "physics_domain": "transport"}
        )
        == "changed"
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
