"""Contracts for manifest-bound protected structural reconciliation."""

from __future__ import annotations

import copy
import json
from hashlib import sha256
from pathlib import Path
from typing import Any

import pytest

from imas_codex.standard_names import protected_structural_reconciliation as sut
from imas_codex.standard_names.grammar_segment_reconciliation import (
    ProtectedSourceSets,
)

_HASH = "a" * 64


def _row(
    action: str = sut.RETIRE_STALE_SOURCE_BRANCH,
    *,
    index: int = 0,
) -> dict[str, Any]:
    expected_after = {"cohort_index": index, "state": "expected"}
    row = {
        "row_key": "",
        "action": action,
        "old_id": f"invalid_identity_{index}",
        "target_id": f"accepted_identity_{index}",
        "source_ids": [f"dd:path_{index}"],
        "backing_ids": [f"path_{index}"],
        "expected_before_hash": _HASH,
        "expected_participant_ids_hash": _HASH,
        "expected_relationship_ids_hash": _HASH,
        "expected_protected_subclosure_hash": _HASH,
        "expected_mutation_hash": _HASH,
        "expected_after": expected_after,
        "expected_after_hash": sut.payload_hash(expected_after),
        "allowlisted_delta": {"action": action},
        "authority_evidence_sha256": (
            _HASH if action == sut.PROTECTED_IDENTITY_FOLD else None
        ),
        "event_timestamp": "2026-08-04T12:00:00+02:00",
        "reason": "exact structural repair",
    }
    row["row_key"] = sut.payload_hash(
        {key: value for key, value in row.items() if key != "row_key"}
    )
    return row


def _payload(
    rows: list[dict[str, Any]],
    *,
    authority_evidence_path: str | Path | None = None,
) -> dict[str, Any]:
    if authority_evidence_path is None and any(
        row["action"] == sut.PROTECTED_IDENTITY_FOLD for row in rows
    ):
        authority_evidence_path = "/authority-evidence/not-present.json"
    return sut.build_manifest_payload(
        rows,
        protected_set_hash=_HASH,
        authority_evidence_sha256=_HASH,
        authority_evidence_path=authority_evidence_path,
    )


def _write(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    authority_evidence_path: str | Path | None = None,
) -> tuple[Path, str]:
    payload = _payload(rows, authority_evidence_path=authority_evidence_path)
    path.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
    return path, sha256(path.read_bytes()).hexdigest()


def _protected() -> ProtectedSourceSets:
    return ProtectedSourceSets(
        west_source_ids=frozenset(),
        fixture_source_ids=frozenset(),
        present_source_ids=frozenset(),
        protected_set_hash=_HASH,
    )


def _catalog() -> dict[str, Any]:
    return {
        "versions": [{"properties": {"id": "4.1.1"}}],
        "cocos_nodes": [{"properties": {"id": 17}}],
    }


def test_operator_module_exposes_both_structural_actions() -> None:
    assert sut.PROTECTED_IDENTITY_FOLD in sut.PROTECTED_STRUCTURAL_ACTIONS
    assert sut.RETIRE_STALE_SOURCE_BRANCH in sut.PROTECTED_STRUCTURAL_ACTIONS


def test_manifest_is_exact_hash_bound_and_homogeneous(tmp_path: Path) -> None:
    rows = [_row(index=0), _row(index=1)]
    rows.sort(key=lambda item: item["row_key"])
    path, digest = _write(tmp_path / "manifest.json", rows)

    manifest = sut.load_protected_structural_manifest(path)

    assert manifest.manifest_hash == digest
    assert manifest.action == sut.RETIRE_STALE_SOURCE_BRANCH
    assert manifest.row_keys == tuple(row["row_key"] for row in rows)

    payload = json.loads(path.read_text())
    payload["unexpected"] = True
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="top-level fields"):
        sut.load_protected_structural_manifest(path)


def test_manifest_refuses_mixed_actions_and_authority_drift(tmp_path: Path) -> None:
    mixed = [_row(index=0), _row(sut.PROTECTED_IDENTITY_FOLD, index=1)]
    mixed.sort(key=lambda item: item["row_key"])
    path, _ = _write(tmp_path / "mixed.json", mixed)
    with pytest.raises(ValueError, match="homogeneous"):
        sut.load_protected_structural_manifest(path)

    fold = _row(sut.PROTECTED_IDENTITY_FOLD)
    fold["authority_evidence_sha256"] = "b" * 64
    path, _ = _write(tmp_path / "authority.json", [fold])
    with pytest.raises(ValueError, match="authority evidence"):
        sut.load_protected_structural_manifest(path)


def test_manifest_refuses_overlapping_targets_and_expected_after_drift(
    tmp_path: Path,
) -> None:
    first = _row(index=0)
    second = _row(index=1)
    second["target_id"] = first["target_id"]
    second["row_key"] = sut.payload_hash(
        {key: value for key, value in second.items() if key != "row_key"}
    )
    rows = sorted([first, second], key=lambda item: item["row_key"])
    path, _ = _write(tmp_path / "overlap.json", rows)
    with pytest.raises(ValueError, match="overlap"):
        sut.load_protected_structural_manifest(path)

    row = _row()
    row["expected_after"]["state"] = "tampered"
    path = tmp_path / "after-drift.json"
    path.write_text(json.dumps(_payload([row])))
    with pytest.raises(ValueError, match="expected_after_hash"):
        sut.load_protected_structural_manifest(path)


def test_manifest_refuses_cocos_dd_and_label_contract_drift(tmp_path: Path) -> None:
    path, _ = _write(tmp_path / "contract.json", [_row()])
    for field, value in (
        ("dd_version", "4.1.0"),
        ("cocos", 11),
        ("downstream_labels", ["psi_like"]),
    ):
        payload = _payload([_row()])
        payload["catalog_contract"][field] = value
        path.write_text(json.dumps(payload))
        with pytest.raises(ValueError, match="DD 4.1.1, COCOS 17, and labels"):
            sut.load_protected_structural_manifest(path)


def test_manifest_refuses_authority_policy_below_floor(tmp_path: Path) -> None:
    payload = _payload([_row(sut.PROTECTED_IDENTITY_FOLD)])
    payload["catalog_contract"]["minimum_authority_confidence"] = 0.5
    path = tmp_path / "weak-policy.json"
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="below the required floor"):
        sut.load_protected_structural_manifest(path)


class _Transaction:
    def __init__(self, *, fail_mutation: bool = False) -> None:
        self.fail_mutation = fail_mutation
        self.committed = False
        self.rolled_back = False
        self.mutation_queries = 0

    def run(self, cypher: str, **params: Any) -> list[dict[str, Any]]:
        if "PROTECTED_STRUCTURAL_FOLD_APPLY" in cypher:
            self.mutation_queries += 1
            return [
                {"results": [{"row_key": item["row_key"]} for item in params["items"]]}
            ]
        if "PROTECTED_STRUCTURAL_RETIRE_APPLY" in cypher:
            self.mutation_queries += 1
            if self.fail_mutation:
                raise RuntimeError("injected mutation failure")
            return [
                {"results": [{"row_key": item["row_key"]} for item in params["items"]]}
            ]
        if "PROTECTED_STRUCTURAL_RETIREMENT_STATE" in cypher:
            return [
                {
                    "row_key": item["row_key"],
                    "old_count": 0,
                    "events": [{"id": event_id} for event_id in item["event_ids"]],
                }
                for item in params["items"]
            ]
        raise AssertionError(f"unexpected query: {cypher[:80]}")

    def commit(self) -> None:
        self.committed = True

    def rollback(self) -> None:
        self.rolled_back = True


class _Session:
    def __init__(self, transaction: _Transaction) -> None:
        self.transaction = transaction

    def __enter__(self) -> _Session:
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def begin_transaction(self) -> _Transaction:
        return self.transaction


class _Graph:
    def __init__(self, *, fail_mutation: bool = False) -> None:
        self.transaction = _Transaction(fail_mutation=fail_mutation)
        self.session_calls = 0

    def session(self) -> _Session:
        self.session_calls += 1
        return _Session(self.transaction)


def _patch_retirement_execution(monkeypatch: pytest.MonkeyPatch) -> None:
    def read_preflight(
        _transaction: _Transaction,
        manifest: sut.ProtectedStructuralManifest,
    ) -> tuple[list[dict[str, Any]], ProtectedSourceSets, dict[str, Any]]:
        plans = []
        for row in manifest.rows:
            retirement_id = f"retirement:{row['row_key']}"
            deletion_id = f"deletion:{row['row_key']}"
            if row["action"] == sut.PROTECTED_IDENTITY_FOLD:
                event_ids = [f"fold:{row['row_key']}"]
                mutation = {
                    "row_key": row["row_key"],
                    "event": {"id": event_ids[0]},
                    "expected_after": {},
                }
            else:
                event_ids = [retirement_id, deletion_id]
                mutation = {
                    "row_key": row["row_key"],
                    "retirement_event": {"id": retirement_id},
                    "deletion_event": {"id": deletion_id},
                    "postflight_source_ids": [],
                    "postflight_backing_ids": [],
                }
            plans.append(
                {
                    "row_key": row["row_key"],
                    "status": "planned",
                    "unresolved": [],
                    "participant_ids": [f"node:{row['row_key']}"],
                    "relationship_ids": [f"edge:{row['row_key']}"],
                    "before_hash": row["expected_before_hash"],
                    "protected_subclosure_hash": row[
                        "expected_protected_subclosure_hash"
                    ],
                    "retirement_protected_hash": sut.payload_hash(
                        {"targets": [], "sources": [], "backings": []}
                    ),
                    "event_ids": event_ids,
                    "mutation": mutation,
                    "expected_after": row["expected_after"],
                    "expected_after_hash": row["expected_after_hash"],
                    "allowlisted_delta": row["allowlisted_delta"],
                }
            )
        return plans, _protected(), _catalog()

    monkeypatch.setattr(sut, "_read_preflight", read_preflight)
    monkeypatch.setattr(sut, "lock_participants", lambda *_args, **_kwargs: ())
    monkeypatch.setattr(sut, "_lock_relationships", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(sut, "_read_catalog_contract", lambda _tx: _catalog())
    monkeypatch.setattr(sut, "_read_protected_source_sets", lambda _tx: _protected())
    monkeypatch.setattr(
        sut,
        "_read_snapshots",
        lambda _tx, manifest: {row["row_key"]: {} for row in manifest.rows},
    )
    monkeypatch.setattr(
        sut.identity_fold, "_fold_verification_state", lambda *_args, **_kwargs: {}
    )
    monkeypatch.setattr(sut, "_validate_authority_evidence", lambda *_args: None)
    monkeypatch.setattr(sut, "_expected_after_matches", lambda *_args: True)


def test_dry_run_is_zero_write_and_apply_requires_exact_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, digest = _write(tmp_path / "dry-run.json", [_row()])
    _patch_retirement_execution(monkeypatch)
    graph = _Graph()

    receipt = sut.reconcile_protected_structure(path, gc=graph)

    assert receipt["mode"] == "dry_run"
    assert receipt["counts"]["planned"] == 1
    assert receipt["rows"][0]["expected_after"] == _row()["expected_after"]
    assert receipt["rows"][0]["allowlisted_delta"] == _row()["allowlisted_delta"]
    assert graph.transaction.mutation_queries == 0
    assert graph.transaction.rolled_back

    before_graph = _Graph()
    with pytest.raises(ValueError, match="SHA-256"):
        sut.reconcile_protected_structure(
            path,
            apply=True,
            expected_manifest_hash="0" * 64,
            gc=before_graph,
        )
    assert before_graph.session_calls == 0
    assert digest != "0" * 64


@pytest.mark.parametrize("size", [1, 40])
def test_apply_query_count_is_constant_in_cohort_size(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    size: int,
) -> None:
    rows = [_row(index=index) for index in range(size)]
    rows.sort(key=lambda item: item["row_key"])
    path, digest = _write(tmp_path / f"cohort-{size}.json", rows)
    _patch_retirement_execution(monkeypatch)
    graph = _Graph()

    receipt = sut.reconcile_protected_structure(
        path,
        apply=True,
        expected_manifest_hash=digest,
        gc=graph,
    )

    assert receipt["mode"] == "applied"
    assert receipt["counts"]["applied"] == size
    assert receipt["query_audit"] == {
        "query_count": 14,
        "cohort_size_independent": True,
    }
    assert graph.transaction.mutation_queries == 1
    assert graph.transaction.committed


@pytest.mark.parametrize("size", [1, 40])
def test_fold_apply_query_count_is_constant_in_cohort_size(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    size: int,
) -> None:
    rows = [_row(sut.PROTECTED_IDENTITY_FOLD, index=index) for index in range(size)]
    rows.sort(key=lambda item: item["row_key"])
    path, digest = _write(tmp_path / f"fold-cohort-{size}.json", rows)
    _patch_retirement_execution(monkeypatch)
    graph = _Graph()

    receipt = sut.reconcile_protected_structure(
        path,
        apply=True,
        expected_manifest_hash=digest,
        gc=graph,
    )

    assert receipt["counts"]["applied"] == size
    assert receipt["query_audit"]["query_count"] == 12
    assert graph.transaction.mutation_queries == 1
    assert graph.transaction.committed


def test_fold_requires_exact_authority_evidence_before_graph_access(
    tmp_path: Path,
) -> None:
    path, digest = _write(
        tmp_path / "fold-authority.json", [_row(sut.PROTECTED_IDENTITY_FOLD)]
    )
    graph = _Graph()

    with pytest.raises(ValueError, match="authority evidence is unavailable"):
        sut.reconcile_protected_structure(
            path,
            apply=True,
            expected_manifest_hash=digest,
            gc=graph,
        )
    assert graph.session_calls == 0

    evidence = tmp_path / "evidence.json"
    evidence.write_text("{}")
    path, digest = _write(
        tmp_path / "fold-authority-hash.json",
        [_row(sut.PROTECTED_IDENTITY_FOLD)],
        authority_evidence_path=evidence,
    )
    with pytest.raises(ValueError, match="SHA-256"):
        sut.reconcile_protected_structure(
            path,
            apply=True,
            expected_manifest_hash=digest,
            gc=graph,
        )
    assert graph.session_calls == 0


def test_fold_accepts_generic_current_catalog_authority_artifact(
    tmp_path: Path,
) -> None:
    row = _row(sut.PROTECTED_IDENTITY_FOLD)
    authorized_disposition = f"authorized_for_{sut.PROTECTED_IDENTITY_FOLD}"
    evidence = {
        "authority_verdict": {
            "semantic_decision_remaining": False,
            "user_decision_remaining": False,
            "confidence": 0.97,
            "verdict": "equivalent_to_catalog_identity",
        },
        "mutation_authorized": True,
        "final_disposition": authorized_disposition,
        "mutation_scopes": [
            {
                "operation": sut.PROTECTED_IDENTITY_FOLD,
                "old_id": row["old_id"],
                "target_id": row["target_id"],
                "source_ids": row["source_ids"],
                "dd_version": "4.1.1",
                "cocos": 17,
                "mutation_authorized": True,
                "final_disposition": authorized_disposition,
            }
        ],
        "cocos_contract": {
            "catalog_check_passed": True,
            "catalog_constant": 17,
            "change_made": False,
        },
        "graph_evidence": {
            "raw_evidence": {
                "catalogs": [{"id": "4.1.1", "is_current": True, "cocos": 17}]
            }
        },
    }
    evidence_path = tmp_path / "authority.json"
    evidence_path.write_text(json.dumps(evidence, sort_keys=True))
    evidence_hash = sha256(evidence_path.read_bytes()).hexdigest()
    row["authority_evidence_sha256"] = evidence_hash
    row["row_key"] = sut.payload_hash(
        {key: value for key, value in row.items() if key != "row_key"}
    )
    payload = sut.build_manifest_payload(
        [row],
        protected_set_hash=_HASH,
        authority_evidence_sha256=evidence_hash,
        authority_evidence_path=evidence_path,
        authority_verdict="equivalent",
        minimum_authority_confidence=0.95,
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(payload, sort_keys=True))

    manifest = sut.load_protected_structural_manifest(manifest_path)

    sut._validate_authority_evidence(manifest)


def test_read_only_authority_artifact_is_a_negative_fixture(tmp_path: Path) -> None:
    row = _row(sut.PROTECTED_IDENTITY_FOLD)
    evidence = {
        "authority_verdict": {
            "semantic_decision_remaining": False,
            "user_decision_remaining": True,
            "confidence": 0.99,
            "verdict": "equivalent_to_catalog_identity",
        },
        "mutation_authorized": False,
        "final_disposition": "equivalent_read_only",
        "mutation_scopes": [],
        "cocos_contract": {
            "catalog_check_passed": True,
            "catalog_constant": 17,
            "change_made": False,
        },
        "graph_evidence": {
            "raw_evidence": {
                "catalogs": [{"id": "4.1.1", "is_current": True, "cocos": 17}]
            }
        },
    }
    evidence_path = tmp_path / "read-only-authority.json"
    evidence_path.write_text(json.dumps(evidence, sort_keys=True))
    evidence_hash = sha256(evidence_path.read_bytes()).hexdigest()
    row["authority_evidence_sha256"] = evidence_hash
    row["row_key"] = sut.payload_hash(
        {key: value for key, value in row.items() if key != "row_key"}
    )
    payload = sut.build_manifest_payload(
        [row],
        protected_set_hash=_HASH,
        authority_evidence_sha256=evidence_hash,
        authority_evidence_path=evidence_path,
    )
    manifest_path = tmp_path / "read-only-manifest.json"
    manifest_path.write_text(json.dumps(payload, sort_keys=True))
    manifest = sut.load_protected_structural_manifest(manifest_path)

    with pytest.raises(ValueError, match="does not authorize"):
        sut._validate_authority_evidence(manifest)


def test_mutation_failure_rolls_back_without_partial_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, digest = _write(tmp_path / "rollback.json", [_row()])
    _patch_retirement_execution(monkeypatch)
    graph = _Graph(fail_mutation=True)

    with pytest.raises(RuntimeError, match="injected mutation failure"):
        sut.reconcile_protected_structure(
            path,
            apply=True,
            expected_manifest_hash=digest,
            gc=graph,
        )

    assert graph.transaction.rolled_back
    assert not graph.transaction.committed


def test_retirement_second_apply_requires_exact_events_and_after_closure(
    tmp_path: Path,
) -> None:
    row = _row()
    state = {
        "old_count": 0,
        "targets": [
            {"labels": ["StandardName"], "properties": {"id": row["target_id"]}}
        ],
        "sources": [],
        "backings": [],
        "events": [
            {
                "labels": ["StandardNameSourceAuthorityRetirement"],
                "properties": {
                    "id": "source-authority-retirement:"
                    + sut._event_identity(row, "source")
                },
            },
            {
                "labels": ["StandardNameChange"],
                "properties": {
                    "id": "sn-change:protected-retirement:"
                    + sut._event_identity(row, "name")
                },
            },
        ],
        "old_mirror_count": 0,
        "orphan_source_cache_count": 0,
        "orphan_backing_cache_count": 0,
    }
    row["expected_after"] = sut._retirement_state_semantics(state)
    row["expected_after_hash"] = sut.payload_hash(row["expected_after"])
    row["row_key"] = sut.payload_hash(
        {key: value for key, value in row.items() if key != "row_key"}
    )
    path, _ = _write(tmp_path / "already-current.json", [row])
    manifest = sut.load_protected_structural_manifest(path)

    current = sut._plan(manifest, {}, {row["row_key"]: state}, _protected(), _catalog())
    assert current[0]["status"] == "already_current"

    drifted = copy.deepcopy(state)
    drifted["events"][0]["properties"]["id"] = "tampered"
    refused = sut._plan(
        manifest, {}, {row["row_key"]: drifted}, _protected(), _catalog()
    )
    assert refused[0]["status"] == "refused"
    assert any("expected-after" in reason for reason in refused[0]["unresolved"])


def test_release_census_is_separate_and_receipt_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    row = _row()
    state = {
        "old_count": 0,
        "targets": [],
        "sources": [],
        "backings": [],
        "events": [],
        "old_mirror_count": 0,
        "orphan_source_cache_count": 0,
        "orphan_backing_cache_count": 0,
    }
    row["expected_after"] = sut._retirement_state_semantics(state)
    row["expected_after_hash"] = sut.payload_hash(row["expected_after"])
    row["row_key"] = sut.payload_hash(
        {key: value for key, value in row.items() if key != "row_key"}
    )
    path, _ = _write(tmp_path / "release.json", [row])
    manifest = sut.load_protected_structural_manifest(path)
    plan = {
        "row_key": row["row_key"],
        "status": "already_current",
        "unresolved": [],
        "event_ids": [],
        "expected_after": row["expected_after"],
        "expected_after_hash": row["expected_after_hash"],
        "allowlisted_delta": row["allowlisted_delta"],
    }
    receipt = sut._receipt(manifest, [plan], applied=True, query_count=4)
    assert receipt["release_postflight"] == {"required": True, "certified": False}
    monkeypatch.setattr(sut, "_read_snapshots", lambda *_args: {})
    monkeypatch.setattr(
        sut,
        "_read_retirement_states",
        lambda *_args: {row["row_key"]: state},
    )
    monkeypatch.setattr(sut, "_read_catalog_contract", lambda _tx: _catalog())
    monkeypatch.setattr(sut, "_read_protected_source_sets", lambda _tx: _protected())

    census = sut.census_protected_structural_release(
        path,
        receipt,
        expected_receipt_hash=receipt["receipt_hash"],
        gc=_Graph(),
    )

    assert census["release_ready"] is True
    assert census["receipt_hash"] == receipt["receipt_hash"]
    assert census["query_audit"]["query_count"] == 4


def test_protected_subclosure_hash_detects_target_and_west_drift() -> None:
    protected = ProtectedSourceSets(
        west_source_ids=frozenset({"dd:west"}),
        fixture_source_ids=frozenset(),
        present_source_ids=frozenset({"dd:west"}),
        protected_set_hash=_HASH,
    )
    snapshot = {
        "target_element_id": "node:target",
        "target_labels": ["StandardName"],
        "target_properties": {"id": "accepted", "cocos": 17},
        "target_units": [{"unit_id": "T"}],
        "sources": [
            {
                "id": "dd:west",
                "element_id": "node:source",
                "properties": {"status": "attached"},
                "backing_refs": [{"backing_element_id": "node:backing"}],
            }
        ],
        "backings": [
            {
                "id": "path",
                "element_id": "node:backing",
                "properties": {"cocos_transformation_type": None},
                "owners": [{"source_element_id": "node:source"}],
            }
        ],
        "relationships": [
            {
                "element_id": "edge:binding",
                "start_element_id": "node:source",
                "end_element_id": "node:target",
                "type": "PRODUCED_NAME",
            }
        ],
    }
    before = sut.payload_hash(sut._protected_subclosure(snapshot, protected))
    changed = copy.deepcopy(snapshot)
    changed["sources"][0]["properties"]["status"] = "stale"

    assert sut.payload_hash(sut._protected_subclosure(changed, protected)) != before


def test_catalog_contract_rejects_noncurrent_dd_and_nonseventeen_cocos() -> None:
    catalog = _catalog()
    assert sut._catalog_reasons(catalog) == []
    catalog["versions"][0]["properties"]["id"] = "4.1.0"
    catalog["cocos_nodes"][0]["properties"]["id"] = 11

    assert sut._catalog_reasons(catalog) == [
        "current DD version is not exactly 4.1.1",
        "global catalog COCOS is not exactly 17",
    ]
