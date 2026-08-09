"""Exact recovery of one stranded name-refinement claim."""

from __future__ import annotations

import copy
from contextlib import contextmanager
from typing import Any

import pytest

from imas_codex.standard_names import graph_ops


class _RefineClaimTransaction:
    def __init__(self, state: dict[str, Any]) -> None:
        self.state = copy.deepcopy(state)
        self._original = copy.deepcopy(state)
        self.closed = False
        self.committed = False
        self.rolled_back = False
        self.mutate_collateral = False

    def _eligible(self, params: dict[str, Any]) -> bool:
        properties = self.state["standard_name"]
        scope = params.get("scope_run_id")
        return (
            properties["id"] == params["target_id"]
            and properties.get("name_stage") == params["expected_stage"]
            and properties.get("claim_token") == params["claim_token"]
            and properties.get("claimed_at") is not None
            and (scope is None or properties.get("run_id") == scope)
        )

    def run(self, cypher: str, **params: Any):
        if "REFINE_CLAIM_RELEASE_LOCK" in cypher:
            return iter([{"id": params["target_id"]}] if self._eligible(params) else [])
        if "REFINE_CLAIM_RELEASE_READ" in cypher:
            if self.state["standard_name"]["id"] != params["target_id"]:
                return iter([])
            return iter([copy.deepcopy(self.state)])
        if "REFINE_CLAIM_RELEASE_WRITE" in cypher:
            properties = self.state["standard_name"]
            if not self._eligible(params):
                return iter([])
            if str(properties["claimed_at"]) != params["expected_claimed_at"]:
                return iter([])
            properties["name_stage"] = "reviewed"
            properties.pop("claim_token")
            properties.pop("claimed_at")
            if self.mutate_collateral:
                properties["documentation"] = "collateral mutation"
            return iter([{"id": properties["id"]}])
        raise AssertionError(f"unexpected query: {cypher}")

    def commit(self) -> None:
        self.committed = True
        self.closed = True

    def rollback(self) -> None:
        self.state = copy.deepcopy(self._original)
        self.rolled_back = True
        self.closed = True


class _RefineClaimGraph:
    def __init__(self, transaction: _RefineClaimTransaction) -> None:
        self.transaction = transaction

    @contextmanager
    def session(self):
        class _Session:
            def __init__(self, transaction: _RefineClaimTransaction) -> None:
                self.transaction = transaction

            def begin_transaction(self):
                return self.transaction

        yield _Session(self.transaction)


def _claim_state() -> dict[str, Any]:
    return {
        "standard_name": {
            "id": "electron_temperature",
            "name_stage": "refining",
            "docs_stage": "accepted",
            "claim_token": "exact-claim-token",
            "claimed_at": "2026-08-09T12:00:00+00:00",
            "run_id": "exact-scope",
            "drain_scope_id": "durable-drain-scope",
            "drain_scope_paths": ["core_profiles/electrons/temperature"],
            "embed_claim_token": "independent-embed-claim",
            "embed_claimed_at": "2026-08-09T11:59:00+00:00",
            "description": "Electron kinetic temperature.",
            "documentation": "The electron kinetic temperature $T_e$.",
            "validation_status": "valid",
            "unit": "eV",
            "kind": "scalar",
            "reviewer_score_name": 0.81,
            "reviewer_score_docs": 0.97,
            "source_paths": ["dd:core_profiles/electrons/temperature"],
        },
        "source_bindings": [
            {
                "source": {
                    "id": "dd:core_profiles/electrons/temperature",
                    "produced_sn_id": "electron_temperature",
                    "status": "composed",
                    "claim_token": None,
                    "drain_scope_id": "durable-drain-scope",
                },
                "relationship": {},
                "bound_names": [{"id": "electron_temperature"}],
            }
        ],
        "source_projections": [
            {
                "labels": ["IMASNode"],
                "node": {"id": "core_profiles/electrons/temperature"},
                "relationship": {},
            }
        ],
        "predecessor_ids": ["electron_temperature_old"],
        "successor_ids": [],
        "outgoing_relationships": [
            {
                "type": "REFINED_FROM",
                "relationship": {},
                "node_labels": ["StandardName"],
                "node": {"id": "electron_temperature_old"},
            }
        ],
        "incoming_relationships": [
            {
                "type": "HAS_REVIEW",
                "relationship": {},
                "node_labels": ["StandardNameReview"],
                "node": {"id": "review-name-1", "score": 0.81},
            }
        ],
    }


def _dry_run(transaction: _RefineClaimTransaction) -> dict[str, Any]:
    return graph_ops.release_stranded_refine_claim(
        "electron_temperature",
        claim_token="exact-claim-token",
        expected_stage="refining",
        scope_run_id="exact-scope",
        apply=False,
        gc=_RefineClaimGraph(transaction),
    )


def test_default_dry_run_rolls_back_exact_cas_and_emits_stable_manifest() -> None:
    first_transaction = _RefineClaimTransaction(_claim_state())
    second_transaction = _RefineClaimTransaction(_claim_state())

    first = _dry_run(first_transaction)
    second = _dry_run(second_transaction)

    assert first_transaction.rolled_back is True
    assert first_transaction.committed is False
    assert first_transaction.state == _claim_state()
    assert first["dry_run"] is True
    assert first["eligible"] is True
    assert first["would_release"] == 1
    assert first["changed"] == 0
    assert first["manifest_sha256"] == second["manifest_sha256"]
    assert first["manifest"] == second["manifest"]
    assert first["before"]["claim"]["token"] == "exact-claim-token"
    assert first["before"]["scope"]["run_id"] == "exact-scope"
    assert first["before"]["content_hashes"]["combined_sha256"]
    assert first["before"]["authority"] == {
        "kind": "scalar",
        "unit": "eV",
        "validation_status": "valid",
    }
    assert first["predicted_after"]["name_stage"] == "reviewed"
    assert first["predicted_after"]["claim"] == {
        "claimed_at": None,
        "token": None,
    }


def test_apply_preserves_every_nonrelease_projection() -> None:
    dry = _dry_run(_RefineClaimTransaction(_claim_state()))
    transaction = _RefineClaimTransaction(_claim_state())

    receipt = graph_ops.release_stranded_refine_claim(
        "electron_temperature",
        claim_token="exact-claim-token",
        expected_stage="refining",
        scope_run_id="exact-scope",
        apply=True,
        manifest_sha256=dry["manifest_sha256"],
        gc=_RefineClaimGraph(transaction),
    )

    assert transaction.committed is True
    assert transaction.rolled_back is False
    assert receipt["changed"] == 1
    assert receipt["would_release"] == 0
    assert receipt["postflight_verified"] is True
    assert receipt["collateral_proof"] == {
        "node_properties_except_release_fields_unchanged": True,
        "relationships_unchanged": True,
        "source_bindings_unchanged": True,
        "source_projections_unchanged": True,
        "lineage_unchanged": True,
    }
    after = transaction.state["standard_name"]
    assert after["name_stage"] == "reviewed"
    assert "claim_token" not in after
    assert "claimed_at" not in after
    expected_other = copy.deepcopy(_claim_state()["standard_name"])
    expected_other.pop("name_stage")
    expected_other.pop("claim_token")
    expected_other.pop("claimed_at")
    actual_other = copy.deepcopy(after)
    actual_other.pop("name_stage")
    assert actual_other == expected_other


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("claim_token", "wrong-token"),
        ("expected_stage", "reviewed"),
        ("scope_run_id", "wrong-scope"),
    ],
)
def test_dry_run_refuses_wrong_claim_authority(field: str, value: str) -> None:
    kwargs = {
        "claim_token": "exact-claim-token",
        "expected_stage": "refining",
        "scope_run_id": "exact-scope",
    }
    kwargs[field] = value
    if field == "expected_stage":
        with pytest.raises(ValueError, match="expected_stage must be 'refining'"):
            graph_ops.release_stranded_refine_claim(
                "electron_temperature",
                apply=False,
                gc=_RefineClaimGraph(_RefineClaimTransaction(_claim_state())),
                **kwargs,
            )
        return

    receipt = graph_ops.release_stranded_refine_claim(
        "electron_temperature",
        apply=False,
        gc=_RefineClaimGraph(_RefineClaimTransaction(_claim_state())),
        **kwargs,
    )
    assert receipt["eligible"] is False
    assert receipt["would_release"] == 0
    assert receipt["outcome"] == "no_exact_eligible_claim"


def test_apply_refuses_hash_drift_and_rolls_back() -> None:
    dry = _dry_run(_RefineClaimTransaction(_claim_state()))
    drifted = _claim_state()
    drifted["standard_name"]["documentation"] = "drifted documentation"
    transaction = _RefineClaimTransaction(drifted)
    with pytest.raises(graph_ops.RefineClaimReleaseConflict, match="manifest"):
        graph_ops.release_stranded_refine_claim(
            "electron_temperature",
            claim_token="exact-claim-token",
            expected_stage="refining",
            scope_run_id="exact-scope",
            apply=True,
            manifest_sha256=dry["manifest_sha256"],
            gc=_RefineClaimGraph(transaction),
        )
    assert transaction.rolled_back is True
    assert transaction.state == drifted


def test_multiple_current_source_bindings_are_refused() -> None:
    ambiguous = _claim_state()
    ambiguous["source_bindings"][0]["bound_names"].append({"id": "other_name"})
    transaction = _RefineClaimTransaction(ambiguous)

    with pytest.raises(graph_ops.RefineClaimReleaseConflict, match="exactly one"):
        _dry_run(transaction)
    assert transaction.rolled_back is True
    assert transaction.state == ambiguous


def test_apply_refuses_collateral_and_rolls_back() -> None:
    dry = _dry_run(_RefineClaimTransaction(_claim_state()))
    transaction = _RefineClaimTransaction(_claim_state())
    transaction.mutate_collateral = True

    with pytest.raises(graph_ops.RefineClaimReleaseConflict, match="collateral"):
        graph_ops.release_stranded_refine_claim(
            "electron_temperature",
            claim_token="exact-claim-token",
            expected_stage="refining",
            scope_run_id="exact-scope",
            apply=True,
            manifest_sha256=dry["manifest_sha256"],
            gc=_RefineClaimGraph(transaction),
        )
    assert transaction.rolled_back is True
    assert transaction.state == _claim_state()


def test_recovery_is_idempotent_and_cannot_reuse_manifest() -> None:
    dry = _dry_run(_RefineClaimTransaction(_claim_state()))
    transaction = _RefineClaimTransaction(_claim_state())
    graph = _RefineClaimGraph(transaction)
    graph_ops.release_stranded_refine_claim(
        "electron_temperature",
        claim_token="exact-claim-token",
        expected_stage="refining",
        scope_run_id="exact-scope",
        apply=True,
        manifest_sha256=dry["manifest_sha256"],
        gc=graph,
    )

    released = _RefineClaimTransaction(transaction.state)
    no_change = _dry_run(released)
    assert no_change["eligible"] is False
    assert no_change["changed"] == 0
    with pytest.raises(graph_ops.RefineClaimReleaseConflict, match="eligible"):
        graph_ops.release_stranded_refine_claim(
            "electron_temperature",
            claim_token="exact-claim-token",
            expected_stage="refining",
            scope_run_id="exact-scope",
            apply=True,
            manifest_sha256=dry["manifest_sha256"],
            gc=_RefineClaimGraph(_RefineClaimTransaction(transaction.state)),
        )
