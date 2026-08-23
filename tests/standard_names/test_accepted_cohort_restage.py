"""Accepted-name cohort restaging preserves identity authority for review."""

from __future__ import annotations

import json
from copy import deepcopy
from unittest.mock import patch

from click.testing import CliRunner

from imas_codex.cli.sn import sn
from imas_codex.standard_names.edit import restage_accepted_names_for_review


def _relationship(
    name_id: str,
    relationship_type: str,
    *,
    direction: str,
) -> dict[str, object]:
    other_id = f"{relationship_type.lower()}:{name_id}"
    return {
        "element_id": f"rel:{direction}:{relationship_type}:{name_id}",
        "relationship_type": relationship_type,
        "properties": {"authority": "fixture"},
        "other_element_id": f"node:{other_id}",
        "other_id": other_id,
        "other_labels": ["AuthorityFixture"],
    }


def _match(name_id: str, **overrides: object) -> dict[str, object]:
    properties: dict[str, object] = {
        "id": name_id,
        "name": name_id,
        "name_stage": "accepted",
        "validation_status": "valid",
        "reviewer_score_name": None,
        "reviewer_model_name": None,
        "claim_token": None,
        "claimed_at": None,
        "drain_scope_id": None,
        "drain_scope_claimed_at": None,
        "drain_claim_scope_id": None,
        "run_id": None,
        "unit": "T",
        "physics_domain": "magnetics",
    }
    properties.update(overrides)
    return {
        "element_id": f"sn:{name_id}",
        "properties": properties,
        "outgoing": [
            _relationship(name_id, "HAS_UNIT", direction="outgoing"),
            _relationship(name_id, "HAS_COCOS", direction="outgoing"),
            _relationship(name_id, "HAS_PARENT", direction="outgoing"),
        ],
        "incoming": [_relationship(name_id, "HAS_STANDARD_NAME", direction="incoming")],
    }


class _Transaction:
    def __init__(self, state: dict[str, list[dict[str, object]]]) -> None:
        self.state = state
        self.calls: list[tuple[str, dict[str, object]]] = []
        self.committed = False
        self.rolled_back = False

    def run(self, query: str, **params: object) -> list[dict[str, object]]:
        self.calls.append((query, params))
        if "ACCEPTED_REVIEW_RESTAGE_SNAPSHOT" in query:
            return [
                {
                    "requested_id": name_id,
                    "matches": deepcopy(self.state.get(name_id, [])),
                }
                for name_id in params["name_ids"]
            ]
        if "ACCEPTED_REVIEW_RESTAGE_LOCK" in query:
            targets = params["targets"]
            return [{"locked_ids": [target["id"] for target in targets]}]
        if "ACCEPTED_REVIEW_RESTAGE_MUTATION" in query:
            staged_ids: list[str] = []
            for target in params["targets"]:
                name_id = str(target["id"])
                match = self.state[name_id][0]
                properties = match["properties"]
                if (
                    properties["name_stage"] == "accepted"
                    and properties["validation_status"] == "valid"
                    and properties["reviewer_score_name"] is None
                ):
                    properties["name_stage"] = "drafted"
                    properties["run_id"] = params["run_id"]
                    staged_ids.append(name_id)
            return [{"staged_ids": staged_ids}]
        raise AssertionError(f"unexpected query: {query}")

    def commit(self) -> None:
        self.committed = True

    def rollback(self) -> None:
        self.rolled_back = True


class _Session:
    def __init__(self, transaction: _Transaction) -> None:
        self.transaction = transaction

    def begin_transaction(self) -> _Transaction:
        return self.transaction

    def __enter__(self) -> _Session:
        return self

    def __exit__(self, *_args: object) -> None:
        return None


class _Client:
    def __init__(self, state: dict[str, list[dict[str, object]]]) -> None:
        self.state = state
        self.transactions: list[_Transaction] = []

    def session(self) -> _Session:
        transaction = _Transaction(self.state)
        self.transactions.append(transaction)
        return _Session(transaction)


def test_operator_refuses_without_include_accepted_before_graph_access() -> None:
    client = _Client({"plasma_current": [_match("plasma_current")]})

    receipt = restage_accepted_names_for_review(
        ["plasma_current"],
        gc=client,
    )

    assert receipt["outcome"] == "refused"
    assert receipt["staged"] == 0
    assert "--include-accepted" in receipt["reason"]
    assert client.transactions == []


def test_noneligible_row_refuses_the_complete_cohort_without_mutation() -> None:
    state = {
        "magnetic_field": [_match("magnetic_field")],
        "plasma_current": [_match("plasma_current", name_stage="reviewed")],
    }
    client = _Client(state)

    receipt = restage_accepted_names_for_review(
        ["magnetic_field", "plasma_current"],
        include_accepted=True,
        dry_run=False,
        gc=client,
    )

    assert receipt["outcome"] == "refused"
    assert receipt["staged"] == 0
    assert receipt["refused_rows"] == [
        {
            "id": "plasma_current",
            "reason": "row is not accepted-valid-null-score and claim-free",
        }
    ]
    assert state["magnetic_field"][0]["properties"]["name_stage"] == "accepted"
    assert all(
        "ACCEPTED_REVIEW_RESTAGE_MUTATION" not in query
        for query, _params in client.transactions[0].calls
    )


def test_atomic_apply_preserves_identity_bindings_and_writes_no_score() -> None:
    state = {
        "magnetic_field": [_match("magnetic_field")],
        "plasma_current": [_match("plasma_current", unit="A")],
    }
    before = deepcopy(state)
    client = _Client(state)

    receipt = restage_accepted_names_for_review(
        ["plasma_current", "magnetic_field"],
        include_accepted=True,
        dry_run=False,
        gc=client,
    )

    assert receipt["outcome"] == "applied"
    assert receipt["requested"] == 2
    assert receipt["would_stage"] == 2
    assert receipt["staged"] == 2
    assert receipt["reviewer_scores_written"] == 0
    assert receipt["binding_counts_before"] == {
        "HAS_STANDARD_NAME": 2,
        "HAS_UNIT": 2,
        "HAS_COCOS": 2,
    }
    assert receipt["binding_counts_after"] == receipt["binding_counts_before"]
    assert receipt["relationship_count_before"] == 8
    assert receipt["relationship_count_after"] == 8
    assert (
        receipt["relationship_signature_after"]
        == receipt["relationship_signature_before"]
    )
    assert receipt["rows"] == [
        {
            "id": "magnetic_field",
            "before_stage": "accepted",
            "after_stage": "drafted",
            "changed": True,
        },
        {
            "id": "plasma_current",
            "before_stage": "accepted",
            "after_stage": "drafted",
            "changed": True,
        },
    ]
    for name_id in state:
        assert state[name_id][0]["properties"]["id"] == name_id
        assert state[name_id][0]["properties"]["name"] == name_id
        assert state[name_id][0]["properties"]["name_stage"] == "drafted"
        assert state[name_id][0]["properties"]["reviewer_score_name"] is None
        assert state[name_id][0]["properties"]["reviewer_model_name"] is None
        assert state[name_id][0]["outgoing"] == before[name_id][0]["outgoing"]
        assert state[name_id][0]["incoming"] == before[name_id][0]["incoming"]
    assert client.transactions[0].committed is True


def test_replay_of_the_same_cohort_is_idempotent() -> None:
    state = {
        "magnetic_field": [_match("magnetic_field")],
        "plasma_current": [_match("plasma_current", unit="A")],
    }
    client = _Client(state)
    first = restage_accepted_names_for_review(
        ["magnetic_field", "plasma_current"],
        include_accepted=True,
        dry_run=False,
        gc=client,
    )

    replay = restage_accepted_names_for_review(
        ["plasma_current", "magnetic_field"],
        include_accepted=True,
        dry_run=False,
        gc=client,
    )

    assert replay["outcome"] == "idempotent"
    assert replay["run_id"] == first["run_id"]
    assert replay["would_stage"] == 0
    assert replay["staged"] == 0
    assert replay["idempotent"] == 2
    assert replay["reviewer_scores_written"] == 0
    assert all(row["changed"] is False for row in replay["rows"])
    assert all(
        "ACCEPTED_REVIEW_RESTAGE_MUTATION" not in query
        for query, _params in client.transactions[1].calls
    )


def test_cli_refuses_without_include_accepted_and_explains_export_consequence() -> None:
    runner = CliRunner()

    refusal = runner.invoke(sn, ["restage-accepted", "plasma_current"])
    help_result = runner.invoke(sn, ["restage-accepted", "--help"])

    assert refusal.exit_code == 3
    receipt = json.loads(refusal.output)
    assert receipt["outcome"] == "refused"
    assert receipt["staged"] == 0
    assert "--include-accepted" in receipt["reason"]
    assert help_result.exit_code == 0
    assert "leave export eligibility" in help_result.output


def test_cli_reaches_atomic_operator() -> None:
    runner = CliRunner()
    receipt = {
        "schema": "imas-codex.accepted-review-restage-receipt",
        "schema_version": 1,
        "outcome": "would_apply",
        "dry_run": True,
        "run_id": "sn-review-restage-fixture",
        "requested": 2,
        "would_stage": 2,
        "staged": 0,
        "rows": [],
    }

    with patch(
        "imas_codex.standard_names.edit.restage_accepted_names_for_review",
        return_value=receipt,
    ) as operator:
        result = runner.invoke(
            sn,
            [
                "restage-accepted",
                "plasma_current",
                "magnetic_field",
                "--include-accepted",
            ],
        )

    assert result.exit_code == 0
    assert json.loads(result.output) == receipt
    operator.assert_called_once_with(
        ("plasma_current", "magnetic_field"),
        include_accepted=True,
        dry_run=True,
        run_id=None,
    )
