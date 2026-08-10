"""Deterministic, read-only recovery planning for documentation evidence."""

from __future__ import annotations

import copy
import json
import multiprocessing
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from imas_codex.standard_names.graph_ops import (
    DOCS_EVIDENCE_RECOVERY_PROJECTION_VERSION,
    DocsEvidenceRecoveryConflict,
    build_docs_evidence_recovery_budget,
    build_docs_evidence_recovery_manifest,
    docs_review_admission_id,
    force_rollback_docs_review_admission,
    rollback_docs_rescore_admission,
    stage_docs_for_rescore,
)
from imas_codex.standard_names.review.audits import compute_review_input_hash


def _state(
    name_id: str,
    *,
    method: str | None = "quorum_consensus",
    cycles: int = 2,
    name_stage: str = "accepted",
    stored_method: str | None = None,
) -> dict:
    group_id = f"group-{name_id}"
    properties = {
        "id": name_id,
        "description": f"Description for {name_id}.",
        "documentation": f"Documentation for {name_id}.",
        "kind": "scalar",
        "unit": "1",
        "links": [],
        "physical_base": "efficiency",
        "subject": None,
        "component": None,
        "coordinate": None,
        "position": None,
        "process": None,
        "cocos_transformation_type": None,
        "source_paths": [f"dd:path/{name_id}"],
        "name_stage": name_stage,
        "docs_stage": "accepted",
        "status": "draft",
        "validation_status": "valid",
        "chain_length": 0,
        "docs_chain_length": 0,
        "edit_status": None,
        "claim_token": None,
        "claimed_at": None,
        "run_id": None,
        "drain_scope_id": None,
        "drain_scope_claimed_at": None,
        "drain_claim_scope_id": None,
        "reviewer_score_docs": 0.9,
        "reviewed_docs_at": "2026-08-09T12:00:00+00:00",
        "docs_review_resolution_method": stored_method,
        "docs_review_quorum_shortfall": None,
        "docs_review_quorum_shortfall_at": None,
    }
    properties["review_input_hash"] = compute_review_input_hash(properties)
    reviews = []
    for cycle in range(cycles):
        terminal = cycle == cycles - 1
        reviews.append(
            {
                "id": f"{name_id}:docs:{group_id}:{cycle}",
                "standard_name_id": name_id,
                "review_axis": "docs",
                "review_group_id": group_id,
                "cycle_index": cycle,
                "resolution_role": "primary" if cycle == 0 else "secondary",
                "resolution_method": method if terminal else None,
                "model": f"reviewer/{cycle}",
                "model_family": f"family/{cycle}",
                "is_canonical": cycle == 0,
                "score": 0.95 if cycle == 0 else 0.85,
                "scores_json": '{"clarity": 18}',
                "reviewed_at": "2026-08-09T12:00:00+00:00",
                "codex_version": "test-build",
                "isn_version": "test-catalog",
            }
        )
    return {
        "id": name_id,
        "standard_name": properties,
        "reviews": reviews,
        "source_bindings": [
            {
                "source_id": f"dd:{name_id}",
                "source_type": "dd",
                "path": f"path/{name_id}",
                "dd_path": f"path/{name_id}",
                "status": "attached",
                "relationship_id": f"rel-{name_id}",
            }
        ],
        "parents": [],
        "children": [],
    }


class _ReadTransaction:
    def __init__(self, states: list[dict]) -> None:
        self.states = {state["id"]: copy.deepcopy(state) for state in states}
        self.closed = False
        self.rolled_back = False
        self.queries: list[str] = []

    def run(self, cypher: str, **params):
        self.queries.append(cypher)
        if "RETURN sn.id AS id" in cypher and "properties(sn)" not in cypher:
            name_stages = set(params["name_stages"])
            docs_stage = params["docs_stage"]
            return [
                {"id": name_id}
                for name_id, state in sorted(self.states.items())
                if state["standard_name"]["docs_stage"] == docs_stage
                and state["standard_name"]["name_stage"] in name_stages
            ]
        if "properties(sn) AS standard_name" in cypher:
            return [
                copy.deepcopy(self.states[name_id])
                for name_id in params["ids"]
                if name_id in self.states
            ]
        raise AssertionError("recovery builder attempted a non-read query")

    def rollback(self) -> None:
        self.closed = True
        self.rolled_back = True


class _Session:
    def __init__(self, transaction: _ReadTransaction) -> None:
        self.transaction = transaction

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def begin_transaction(self):
        return self.transaction


class _Graph:
    def __init__(self, states: list[dict]) -> None:
        self.transaction = _ReadTransaction(states)

    def session(self):
        return _Session(self.transaction)


class _RescoreTransaction:
    def __init__(self, state: dict) -> None:
        self.state = copy.deepcopy(state)
        self.original = copy.deepcopy(state)
        self.closed = False
        self.committed = False
        self.rolled_back = False
        self.write_count = 0
        self.admissions: dict[str, dict] = {}
        self.original_admissions: dict[str, dict] = {}
        self.admission_links: set[str] = set()
        self.original_admission_links: set[str] = set()

    def run(self, cypher: str, **params):
        properties = self.state["standard_name"]
        if "_docs_rescore_lock" in cypher:
            return [{"standard_name": copy.deepcopy(properties)}]
        if "_docs_admission_recovery_lock" in cypher:
            admissions = [
                copy.deepcopy(admission)
                for admission in self.admissions.values()
                if admission.get("target_id") == params["target_id"]
                and admission.get("scope_id") == params["scope_id"]
            ]
            return [
                {
                    "standard_name": copy.deepcopy(properties),
                    "admissions": admissions,
                    "linked_admission_ids": sorted(self.admission_links),
                }
            ]
        if "OPTIONAL MATCH (a:DocsReviewAdmission" in cypher:
            return [
                {
                    "admissions": int(params["admission_id"] in self.admissions),
                    "other_targets": 0,
                }
            ]
        if "sn.docs_stage IN ['accepted', 'reviewed', 'exhausted']" in cypher:
            eligible = (
                properties["name_stage"] == "accepted"
                and properties["docs_stage"] in {"accepted", "reviewed", "exhausted"}
                and not properties.get("claim_token")
                and not properties.get("claimed_at")
                and not properties.get("drain_scope_id")
                and not properties.get("drain_scope_claimed_at")
                and not properties.get("drain_claim_scope_id")
            )
            if not eligible:
                return []
            prior_stage = properties["docs_stage"]
            if not params["dry_run"]:
                properties["docs_stage"] = "drafted"
                properties["reviewer_score_docs"] = None
                properties["run_id"] = params["run_id"]
                self.write_count += 1
            return [
                {
                    "prior_stage": prior_stage,
                    "description": properties["description"],
                    "documentation": properties["documentation"],
                }
            ]
        if "CREATE (a:DocsReviewAdmission" in cypher:
            if properties.get("docs_review_admission") is not None:
                return []
            admission = {
                "id": params["admission_id"],
                "target_id": params["target_id"],
                "scope_id": params["scope_id"],
                "request_identity": params["request_identity"],
                "docs_hash": params["docs_hash"],
                "review_input_hash": params["review_input_hash"],
                "mandatory_exposures_json": params["mandatory_exposures_json"],
                "conditional_exposure": params["conditional_exposure"],
                "expected_exposure": params["expected_exposure"],
                "prior_docs_stage": params["prior_docs_stage"],
                "prior_authority_json": params["prior_authority_json"],
                "review_group_id": params["review_group_id"],
                "created_review_ids": [],
                "status": "active",
                "receipt_identity": params["receipt_identity"],
                "claim_token": None,
                "claim_seq": None,
            }
            self.admissions[admission["id"]] = admission
            self.admission_links.add(admission["id"])
            properties["docs_review_admission"] = admission["id"]
            self.write_count += 1
            return [{"admission_id": admission["id"]}]
        if "_docs_review_admission_lock" in cypher:
            admission = self.admissions.get(params["admission_id"])
            if admission is None:
                return []
            owned = [
                review
                for review in self.state["reviews"]
                if review.get("docs_review_admission_id") == admission["id"]
            ]
            return [
                {
                    "admission": copy.deepcopy(admission),
                    "standard_name": copy.deepcopy(properties),
                    "owned_reviews": copy.deepcopy(owned),
                }
            ]
        if "deleted_reviews" in cypher:
            admission = self.admissions[params["admission_id"]]
            prior = json.loads(admission["prior_authority_json"])
            for field, value in prior.items():
                properties[field] = copy.deepcopy(value)
            properties["claimed_at"] = None
            properties["claim_token"] = None
            properties["docs_review_admission"] = None
            self.state["reviews"] = [
                review
                for review in self.state["reviews"]
                if review.get("docs_review_admission_id") != admission["id"]
            ]
            admission["status"] = "rolled_back"
            admission["created_review_ids"] = []
            self.admission_links.discard(admission["id"])
            self.write_count += 1
            return [{"id": properties["id"], "deleted_reviews": 1}]
        raise AssertionError("unexpected exact rescore query")

    def commit(self) -> None:
        self.committed = True
        self.closed = True
        self.original = copy.deepcopy(self.state)
        self.original_admissions = copy.deepcopy(self.admissions)
        self.original_admission_links = set(self.admission_links)

    def rollback(self) -> None:
        self.state = copy.deepcopy(self.original)
        self.admissions = copy.deepcopy(self.original_admissions)
        self.admission_links = set(self.original_admission_links)
        self.rolled_back = True
        self.closed = True


class _RescoreGraph:
    def __init__(self, state: dict) -> None:
        self.transaction = _RescoreTransaction(state)

    def session(self):
        return _Session(self.transaction)

    def close(self) -> None:
        pass


def _run_serialized_forced_rollback(input_path: str, output_path: str) -> None:
    """Reload staged graph state in a fresh interpreter and reconcile it."""
    payload = json.loads(Path(input_path).read_text())
    graph = _RescoreGraph(payload["state"])
    transaction = graph.transaction
    transaction.admissions = payload["admissions"]
    transaction.admission_links = set(payload["admission_links"])
    transaction.original_admissions = copy.deepcopy(transaction.admissions)
    transaction.original_admission_links = set(transaction.admission_links)
    with patch("imas_codex.standard_names.graph_ops.GraphClient", return_value=graph):
        result = force_rollback_docs_review_admission(**payload["rollback"])
    Path(output_path).write_text(
        json.dumps(
            {
                "result": result,
                "state": transaction.state,
                "admissions": transaction.admissions,
            },
            sort_keys=True,
        )
    )


def _row(manifest: dict, name_id: str) -> dict:
    return next(row for row in manifest["rows"] if row["id"] == name_id)


def test_exact_builder_is_deterministic_and_proves_strict_backfill() -> None:
    states = [_state("zeta_name"), _state("alpha_name")]

    first = build_docs_evidence_recovery_manifest(
        ["zeta_name", "alpha_name"], gc=_Graph(states)
    )
    second = build_docs_evidence_recovery_manifest(
        ["alpha_name", "zeta_name"], gc=_Graph(states)
    )

    assert first == second
    assert first["projection_version"] == DOCS_EVIDENCE_RECOVERY_PROJECTION_VERSION
    assert len(first["manifest_id"]) == 64
    assert [row["id"] for row in first["rows"]] == ["alpha_name", "zeta_name"]
    assert first["counts"] == {
        "already_authoritative": 0,
        "metadata_backfill_proven": 2,
        "rescore_required_current": 0,
        "historical_hold": 0,
        "evidence_ambiguous_hold": 0,
    }
    row = first["rows"][0]
    assert row["outcome"] == "metadata_backfill_proven"
    assert row["authority_projection"]["id"] == "alpha_name"
    assert row["description"] == "Description for alpha_name."
    assert row["source_bindings"][0]["path"] == "path/alpha_name"
    assert row["latest_review_group"][1]["score"] == 0.85


def test_canonical_stored_method_is_already_authoritative_not_backfill() -> None:
    state = _state("settled_name", stored_method="quorum_consensus")

    manifest = build_docs_evidence_recovery_manifest([state["id"]], gc=_Graph([state]))
    row = manifest["rows"][0]

    assert manifest["counts"] == {
        "already_authoritative": 1,
        "metadata_backfill_proven": 0,
        "rescore_required_current": 0,
        "historical_hold": 0,
        "evidence_ambiguous_hold": 0,
    }
    assert row["outcome"] == "already_authoritative"
    assert row["authority_status"] == "already_authoritative"
    assert row["authority_projection"]["expected_resolution_method"] == (
        "quorum_consensus"
    )


def test_lifecycle_selection_is_explicit_and_cardinality_fenced() -> None:
    states = [_state("accepted"), _state("old", name_stage="superseded")]
    selection = {
        "docs_stage": "accepted",
        "name_stages": ["accepted", "superseded"],
    }
    graph = _Graph(states)

    manifest = build_docs_evidence_recovery_manifest(
        lifecycle_selection=selection,
        expected_count=2,
        gc=graph,
    )

    assert manifest["selection"] == {
        "kind": "lifecycle",
        "docs_stage": "accepted",
        "name_stages": ["accepted", "superseded"],
        "expected_count": 2,
    }
    assert graph.transaction.rolled_back is True
    with pytest.raises(DocsEvidenceRecoveryConflict, match="cardinality"):
        build_docs_evidence_recovery_manifest(
            lifecycle_selection=selection,
            expected_count=3,
            gc=_Graph(states),
        )


@pytest.mark.parametrize(
    "name_ids,selection,expected_count",
    [
        (None, None, None),
        (["same", "same"], None, None),
        (["same"], {"docs_stage": "accepted", "name_stages": ["accepted"]}, 1),
        (None, {"docs_stage": "accepted", "name_stages": ["accepted"]}, None),
    ],
)
def test_selection_contract_fails_closed(name_ids, selection, expected_count) -> None:
    with pytest.raises(ValueError):
        build_docs_evidence_recovery_manifest(
            name_ids,
            lifecycle_selection=selection,
            expected_count=expected_count,
            gc=_Graph([_state("same")]),
        )


def test_current_single_review_is_exact_rescore_without_prose_change() -> None:
    state = _state("absorbed_wave_power", method="single_review", cycles=1)
    manifest = build_docs_evidence_recovery_manifest([state["id"]], gc=_Graph([state]))
    row = manifest["rows"][0]

    assert row["outcome"] == "rescore_required_current"
    assert row["reason_codes"] == ["incomplete_review_group"]
    assert row["rescore_input"] == {
        "sn_id": "absorbed_wave_power",
        "expected_docs_hash": row["docs_hash"],
        "expected_review_input_hash": row["current_review_input_hash"],
    }


def test_manifest_rescore_input_executes_content_bound_staging() -> None:
    state = _state("absorbed_wave_power", method="single_review", cycles=1)
    manifest = build_docs_evidence_recovery_manifest([state["id"]], gc=_Graph([state]))
    graph = _RescoreGraph(state)

    with patch("imas_codex.standard_names.graph_ops.GraphClient", return_value=graph):
        result = stage_docs_for_rescore(
            **manifest["rows"][0]["rescore_input"], run_id="exact-docs-run"
        )

    assert result["ok"] is True
    assert result["content_cas_verified"] is True
    assert graph.transaction.committed is True
    assert graph.transaction.write_count == 1
    assert graph.transaction.state["standard_name"]["docs_stage"] == "drafted"
    assert (
        graph.transaction.state["standard_name"]["description"]
        == (state["standard_name"]["description"])
    )
    assert (
        graph.transaction.state["standard_name"]["documentation"]
        == (state["standard_name"]["documentation"])
    )


def test_priced_staging_receipt_drives_exact_docs_axis_rollback() -> None:
    state = _state("absorbed_wave_power", method="single_review", cycles=1)
    manifest = build_docs_evidence_recovery_manifest([state["id"]], gc=_Graph([state]))
    graph = _RescoreGraph(state)

    with patch("imas_codex.standard_names.graph_ops.GraphClient", return_value=graph):
        staged = stage_docs_for_rescore(
            **manifest["rows"][0]["rescore_input"],
            run_id="exact-docs-run",
            request_identity="a" * 64,
            mandatory_exposures=[1.0, 1.0],
            conditional_exposure=1.0,
            expected_exposure=3.0,
        )
        graph.transaction.closed = False
        rolled_back = rollback_docs_rescore_admission(staged["admission_id"])

    assert rolled_back["ok"] is True
    after = graph.transaction.state["standard_name"]
    before = state["standard_name"]
    for field in (
        "docs_stage",
        "reviewer_score_docs",
        "reviewer_scores_docs",
        "reviewer_comments_docs",
        "reviewer_comments_per_dim_docs",
        "reviewer_model_docs",
        "docs_review_resolution_method",
        "docs_review_quorum_shortfall",
        "docs_review_quorum_shortfall_at",
        "reviewed_docs_at",
        "run_id",
    ):
        assert after.get(field) == before.get(field)
    assert after["description"] == before["description"]
    assert after["documentation"] == before["documentation"]
    assert after["docs_chain_length"] == before["docs_chain_length"]
    assert graph.transaction.state["reviews"] == state["reviews"]
    assert after["claim_token"] is None
    assert after["claimed_at"] is None
    assert after["docs_review_admission"] is None

    graph.transaction.closed = False
    with patch("imas_codex.standard_names.graph_ops.GraphClient", return_value=graph):
        repeated = rollback_docs_rescore_admission(staged["admission_id"])
    assert repeated == {"ok": True, "outcome": "already_rolled_back"}


def test_docs_review_admission_schema_declares_durable_authority() -> None:
    schema_path = (
        Path(__file__).parents[2] / "imas_codex" / "schemas" / "standard_name.yaml"
    )
    schema = yaml.safe_load(schema_path.read_text())
    admission = schema["classes"]["DocsReviewAdmission"]

    assert admission["attributes"]["id"]["identifier"] is True
    assert admission["attributes"]["status"]["range"] == ("DocsReviewAdmissionStatus")
    assert {
        "target_id",
        "scope_id",
        "request_identity",
        "docs_hash",
        "review_input_hash",
        "mandatory_exposures_json",
        "conditional_exposure",
        "expected_exposure",
        "claim_token",
        "claim_seq",
        "prior_docs_stage",
        "prior_authority_json",
        "review_group_id",
        "created_review_ids",
        "receipt_identity",
        "created_at",
        "expires_at",
        "created_reviews",
    } <= set(admission["attributes"])
    assert admission["attributes"]["created_reviews"] == {
        "description": "StandardNameReview records owned exclusively by this admission.",
        "multivalued": True,
        "range": "StandardNameReview",
        "inlined": False,
        "annotations": {"relationship_type": "CREATED_REVIEW"},
    }


def test_admission_identity_is_deterministic_and_scope_unique() -> None:
    assert docs_review_admission_id("scope-a") == docs_review_admission_id("scope-a")
    assert docs_review_admission_id("scope-a") != docs_review_admission_id("scope-b")
    assert len(docs_review_admission_id("scope-a")) == 64


def test_restart_rollback_deletes_only_admission_owned_partial_reviews() -> None:
    state = _state("absorbed_wave_power", method="single_review", cycles=1)
    manifest = build_docs_evidence_recovery_manifest([state["id"]], gc=_Graph([state]))
    graph = _RescoreGraph(state)
    with patch("imas_codex.standard_names.graph_ops.GraphClient", return_value=graph):
        staged = stage_docs_for_rescore(
            **manifest["rows"][0]["rescore_input"],
            run_id="restart-scope",
            request_identity="a" * 64,
            mandatory_exposures=[1.0, 1.0],
            conditional_exposure=1.0,
            expected_exposure=3.0,
        )
        admission_id = staged["admission_id"]
        graph.transaction.state["reviews"].append(
            {
                "id": "partial-review",
                "standard_name_id": state["id"],
                "review_axis": "docs",
                "review_group_id": staged["review_group_id"],
                "docs_review_admission_id": admission_id,
            }
        )
        graph.transaction.closed = False
        result = rollback_docs_rescore_admission(admission_id)

    assert result["ok"] is True
    assert [review["id"] for review in graph.transaction.state["reviews"]] == [
        state["reviews"][0]["id"]
    ]


def test_force_rollback_accepts_reclaimed_target_claim_without_rebinding() -> None:
    state = _state("absorbed_wave_power", method="single_review", cycles=1)
    manifest = build_docs_evidence_recovery_manifest([state["id"]], gc=_Graph([state]))
    graph = _RescoreGraph(state)
    with patch("imas_codex.standard_names.graph_ops.GraphClient", return_value=graph):
        staged = stage_docs_for_rescore(
            **manifest["rows"][0]["rescore_input"],
            run_id="crash-recovery-scope",
            request_identity="a" * 64,
            mandatory_exposures=[1.0, 1.0],
            conditional_exposure=1.0,
            expected_exposure=3.0,
        )
        admission = graph.transaction.admissions[staged["admission_id"]]
        admission["claim_token"] = "dead-worker-token"
        admission["claim_seq"] = 4
        target = graph.transaction.state["standard_name"]
        target["claim_token"] = "reclaimed-token"
        target["claim_seq"] = 5
        target["claimed_at"] = "now"
        graph.transaction.closed = False
        result = force_rollback_docs_review_admission(
            target_id=state["id"],
            scope_id="crash-recovery-scope",
            intended_admission_id=staged["admission_id"],
            current_claim_token="reclaimed-token",
            current_claim_seq=5,
        )

    assert result["ok"] is True
    assert result["outcome"] == "rolled_back"
    assert target["docs_stage"] == "accepted"
    assert target["claim_token"] is None
    assert admission["status"] == "rolled_back"


def test_staged_admission_survives_serialization_and_fresh_interpreter(
    tmp_path: Path,
) -> None:
    state = _state("absorbed_wave_power", method="single_review", cycles=1)
    manifest = build_docs_evidence_recovery_manifest([state["id"]], gc=_Graph([state]))
    graph = _RescoreGraph(state)
    with patch("imas_codex.standard_names.graph_ops.GraphClient", return_value=graph):
        staged = stage_docs_for_rescore(
            **manifest["rows"][0]["rescore_input"],
            run_id="serialized-scope",
            request_identity="a" * 64,
            mandatory_exposures=[1.0, 1.0],
            conditional_exposure=1.0,
            expected_exposure=3.0,
        )

    target = graph.transaction.state["standard_name"]
    target["claim_token"] = "replacement-token"
    target["claim_seq"] = 8
    target["claimed_at"] = "now"
    admission_id = staged["admission_id"]
    graph.transaction.admissions[admission_id]["claim_token"] = "crashed-token"
    graph.transaction.admissions[admission_id]["claim_seq"] = 7
    graph.transaction.state["reviews"].append(
        {
            "id": "serialized-partial-review",
            "docs_review_admission_id": admission_id,
        }
    )
    input_path = tmp_path / "durable-input.json"
    output_path = tmp_path / "durable-output.json"
    input_path.write_text(
        json.dumps(
            {
                "state": graph.transaction.state,
                "admissions": graph.transaction.admissions,
                "admission_links": sorted(graph.transaction.admission_links),
                "rollback": {
                    "target_id": state["id"],
                    "scope_id": "serialized-scope",
                    "intended_admission_id": admission_id,
                    "current_claim_token": "replacement-token",
                    "current_claim_seq": 8,
                },
            },
            sort_keys=True,
        )
    )

    process = multiprocessing.get_context("spawn").Process(
        target=_run_serialized_forced_rollback,
        args=(str(input_path), str(output_path)),
    )
    process.start()
    process.join(timeout=20)

    assert process.exitcode == 0
    recovered = json.loads(output_path.read_text())
    assert recovered["result"]["outcome"] == "rolled_back"
    assert recovered["state"]["standard_name"]["docs_stage"] == "accepted"
    assert recovered["state"]["standard_name"]["claim_token"] is None
    assert [review["id"] for review in recovered["state"]["reviews"]] == [
        state["reviews"][0]["id"]
    ]
    assert recovered["admissions"][admission_id]["status"] == "rolled_back"


def test_force_rollback_recovers_expired_unbound_admission_with_missing_scalar() -> (
    None
):
    state = _state("absorbed_wave_power", method="single_review", cycles=1)
    manifest = build_docs_evidence_recovery_manifest([state["id"]], gc=_Graph([state]))
    graph = _RescoreGraph(state)
    with patch("imas_codex.standard_names.graph_ops.GraphClient", return_value=graph):
        staged = stage_docs_for_rescore(
            **manifest["rows"][0]["rescore_input"],
            run_id="expired-scope",
            request_identity="a" * 64,
            mandatory_exposures=[1.0, 1.0],
            conditional_exposure=1.0,
            expected_exposure=3.0,
        )
        admission = graph.transaction.admissions[staged["admission_id"]]
        admission["expired"] = True
        target = graph.transaction.state["standard_name"]
        target["docs_review_admission"] = None
        target["claim_token"] = "current-token"
        target["claim_seq"] = 1
        graph.transaction.closed = False
        result = force_rollback_docs_review_admission(
            target_id=state["id"],
            scope_id="expired-scope",
            intended_admission_id=None,
            current_claim_token="current-token",
            current_claim_seq=1,
        )

    assert result["ok"] is True
    assert target["docs_stage"] == "accepted"
    assert target["claim_token"] is None


def test_force_rollback_refuses_duplicate_active_target_scope_authority() -> None:
    state = _state("absorbed_wave_power", method="single_review", cycles=1)
    manifest = build_docs_evidence_recovery_manifest([state["id"]], gc=_Graph([state]))
    graph = _RescoreGraph(state)
    with patch("imas_codex.standard_names.graph_ops.GraphClient", return_value=graph):
        staged = stage_docs_for_rescore(
            **manifest["rows"][0]["rescore_input"],
            run_id="duplicate-scope",
            request_identity="a" * 64,
            mandatory_exposures=[1.0, 1.0],
            conditional_exposure=1.0,
            expected_exposure=3.0,
        )
        duplicate = copy.deepcopy(graph.transaction.admissions[staged["admission_id"]])
        duplicate["id"] = "f" * 64
        graph.transaction.admissions[duplicate["id"]] = duplicate
        target_before = copy.deepcopy(graph.transaction.state["standard_name"])
        graph.transaction.closed = False
        result = force_rollback_docs_review_admission(
            target_id=state["id"],
            scope_id="duplicate-scope",
            intended_admission_id=staged["admission_id"],
            current_claim_token=None,
            current_claim_seq=None,
        )

    assert result["ok"] is False
    assert result["outcome"] == "duplicate_active_admission"
    assert graph.transaction.state["standard_name"] == target_before


@pytest.mark.parametrize("drift_field", ["description", "documentation", "kind"])
def test_manifest_rescore_input_refuses_content_or_review_input_drift(
    drift_field: str,
) -> None:
    state = _state("absorbed_wave_power", method="single_review", cycles=1)
    manifest = build_docs_evidence_recovery_manifest([state["id"]], gc=_Graph([state]))
    changed = copy.deepcopy(state)
    changed["standard_name"][drift_field] += " changed"
    graph = _RescoreGraph(changed)

    with patch("imas_codex.standard_names.graph_ops.GraphClient", return_value=graph):
        result = stage_docs_for_rescore(
            **manifest["rows"][0]["rescore_input"], run_id="exact-docs-run"
        )

    assert result["ok"] is False
    assert result["outcome"] == "content_drift"
    assert graph.transaction.write_count == 0
    assert graph.transaction.committed is False
    assert graph.transaction.rolled_back is True
    assert graph.transaction.state == graph.transaction.original


@pytest.mark.parametrize(
    "fault,reason",
    [
        ("missing_group", "missing_review_group"),
        ("stale_input", "stale_review_input"),
        ("aggregate", "aggregate_mismatch"),
        ("missing_method", "terminal_method_mismatch"),
        ("claim", "claim_scope_lifecycle_conflict"),
    ],
)
def test_ambiguous_evidence_never_becomes_recovered_or_rescore(fault, reason) -> None:
    state = _state(fault)
    if fault == "missing_group":
        state["reviews"] = []
    elif fault == "stale_input":
        state["standard_name"]["documentation"] += " Changed."
    elif fault == "aggregate":
        state["standard_name"]["reviewer_score_docs"] = 0.42
    elif fault == "missing_method":
        state["reviews"][-1]["resolution_method"] = None
    elif fault == "claim":
        state["standard_name"]["claim_token"] = "live-claim"

    manifest = build_docs_evidence_recovery_manifest([state["id"]], gc=_Graph([state]))
    row = manifest["rows"][0]

    assert row["outcome"] == "evidence_ambiguous_hold"
    assert reason in row["reason_codes"]
    assert row["authority_projection"] is None
    assert row["rescore_input"] is None


def test_superseded_history_is_zero_cost_hold_even_with_quorate_evidence() -> None:
    state = _state(
        "historical_name",
        name_stage="superseded",
        stored_method="quorum_consensus",
    )
    manifest = build_docs_evidence_recovery_manifest([state["id"]], gc=_Graph([state]))
    row = manifest["rows"][0]

    assert row["outcome"] == "historical_hold"
    assert row["reason_codes"] == ["superseded_history"]
    assert row["priority"] == 3
    assert row["rescore_input"] is None
    assert row["authority_status"] == "already_authoritative"


def test_budget_envelope_never_promises_unknown_cost_or_exceeds_caps() -> None:
    states = [
        _state(f"current_{index:02d}", method="single_review", cycles=1)
        for index in range(9)
    ]
    states.append(_state("historical", name_stage="superseded"))
    manifest = build_docs_evidence_recovery_manifest(
        [state["id"] for state in states], gc=_Graph(states)
    )

    budget = build_docs_evidence_recovery_budget(
        manifest,
        authorized_remaining_ceiling=190.092420,
        per_batch_hard_cap=50,
    )

    assert budget["authorized_remaining_ceiling_usd"] == 190.09242
    assert budget["total_hard_cap_usd"] <= 190.09242
    assert all(
        tranche["hard_cost_cap_usd"] <= 50 for tranche in budget["hard_cap_tranches"]
    )
    assert sum(
        tranche["hard_cost_cap_usd"] for tranche in budget["hard_cap_tranches"]
    ) == pytest.approx(190.09242)
    assert all(tranche["stop_and_remeasure"] for tranche in budget["hard_cap_tranches"])
    assert all(tranche["target_ids"] == [] for tranche in budget["hard_cap_tranches"])
    assert budget["reserved_exposure_usd"] == 0.0
    assert budget["expected_exposure_usd"] is None
    assert budget["expected_admission_mechanism"] == (
        "model_provider_exposure_after_request_render"
    )
    assert budget["provider_policy_ceiling_is_separate"] is True
    queued = [item["id"] for item in budget["prioritized_queue"]]
    assert queued == sorted(state["id"] for state in states[:-1])
    assert "historical" not in queued
    assert budget["historical_hold_count"] == 1


def test_budget_rejects_projection_or_identity_tampering() -> None:
    state = _state("candidate", method="single_review", cycles=1)
    manifest = build_docs_evidence_recovery_manifest([state["id"]], gc=_Graph([state]))
    manifest["projection_version"] += 1
    with pytest.raises(ValueError, match="projection version"):
        build_docs_evidence_recovery_budget(
            manifest,
            authorized_remaining_ceiling=10,
            per_batch_hard_cap=5,
        )
