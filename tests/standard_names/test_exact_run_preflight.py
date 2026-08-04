"""One-query authorization evidence for exact paid name operations."""

from __future__ import annotations

from copy import deepcopy
from decimal import Decimal
from unittest.mock import patch

import pytest

from imas_codex.standard_names import graph_ops, run_preflight
from imas_codex.standard_names.defaults import (
    DETERMINISTIC_PARENT_DESCRIPTION_PLACEHOLDER,
)
from imas_codex.standard_names.run_preflight import (
    audit_exact_standard_name_preflight,
)

NAME = "poloidal_momentum_neutral_internal_state_flux_limiter_coefficient"
PATH = "edge_profiles/source/neutral/internal_state/flux_limiter_coefficient"


class _Client:
    def __init__(
        self, rows: list[dict[str, object]], *, error: Exception | None = None
    ):
        self.rows = rows
        self.error = error
        self.calls: list[tuple[str, dict[str, object]]] = []

    def query(self, query: str, **params: object) -> list[dict[str, object]]:
        self.calls.append((query, params))
        if self.error is not None:
            raise self.error
        return self.rows


def _row(label: str | None = None) -> dict[str, object]:
    return {
        "targets": [
            {
                "element_id": "target-1",
                "id": NAME,
                "name_stage": "reviewed",
                "docs_stage": "pending",
                "status": "draft",
                "validation_status": "valid",
                "description": "A reviewable DD-grounded quantity.",
                "reviewer_score_name": 0.5,
                "chain_length": 1,
                "review_resubmit_count": 0,
                "origin": "pipeline",
                "facility": None,
                "edit_mode": None,
                "edit_status": None,
                "unit": "m",
                "dd_version": None,
                "cocos": 17 if label is not None else None,
                "cocos_transformation_type": label,
                "source_paths": [f"dd:{PATH}"],
                # Durable history is evidence, never a live lease.
                "run_id": "historical-run",
                "last_run_id": "historical-worker-run",
                "claimed_at": None,
                "claim_token": None,
                "drain_scope_id": None,
                "drain_scope_claimed_at": None,
                "drain_claim_scope_id": None,
            }
        ],
        "action_count": 1,
        "action_element_ids": ["target-1"],
        "refine_action_count": 1,
        "review_action_count": 0,
        "sources": [
            {
                "element_id": "source-1",
                "id": f"dd:{PATH}",
                "source_id": PATH,
                "source_type": "dd",
                "status": "composed",
                "produced_sn_id": NAME,
                "produced_edge_id": "produced-1",
                "claimed_at": None,
                "claim_token": None,
                "drain_scope_id": None,
                "drain_scope_claimed_at": None,
                "drain_claim_scope_id": None,
                "dd_version": "4.1.1",
                "dd_snapshot_pinned": True,
                "dd_unit": "m",
                "dd_path": PATH,
                "backing_id": PATH,
                "backing_unit": "m",
                "backing_unit_ids": ["m"],
                "backing_unit_edge_ids": ["backing-unit-1"],
                "from_dd_edge_ids": ["from-dd-1"],
                "projection_edge_ids": ["projection-1"],
                "cocos_label": label,
            }
        ],
        "target_unit_ids": ["m"],
        "target_unit_edge_ids": ["target-unit-1"],
        "predecessor_ids": ["earlier_candidate"],
        "successor_ids": [],
        "accepted_or_protected_lineage_ids": [],
        "refinement_protected_source_ids": [],
        "protected_source_ids": [],
        "catalogs": [
            {
                "element_id": "catalog-1",
                "id": "4.1.1",
                "status": "built",
                "is_current": True,
                "cocos": 17,
            }
        ],
        "catalog_cocos_ids": [17],
        "catalog_cocos_edge_ids": ["catalog-cocos-1"],
    }


def _audit(row: dict[str, object], **overrides: object):
    client = _Client([row])
    kwargs: dict[str, object] = {
        "requested_cost_ceiling": "0.50",
        "cumulative_spend": "5.10",
        "authorized_budget": "200.00",
        "dd_version": "4.1.1",
        "gc": client,
    }
    kwargs.update(overrides)
    with patch(
        "imas_codex.standard_names.grammar_segment_reconciliation._west_source_ids",
        return_value=frozenset({"dd:west/protected"}),
    ):
        receipt = audit_exact_standard_name_preflight(NAME, **kwargs)
    return receipt, client


def _review_row(label: str | None = None) -> dict[str, object]:
    row = _row(label)
    target = row["targets"][0]  # type: ignore[index]
    target["name_stage"] = "drafted"
    target["reviewer_score_name"] = None
    target["chain_length"] = 2
    row["action_count"] = 1
    row["refine_action_count"] = 0
    row["review_action_count"] = 1
    row["accepted_or_protected_lineage_ids"] = ["accepted_predecessor"]
    return row


@pytest.mark.parametrize("label", [None, "psi_like", "ip_like"])
def test_pass_retains_exact_cocos_label_and_historical_run_id(
    label: str | None,
) -> None:
    receipt, client = _audit(_row(label))

    assert receipt.passed is True
    assert receipt.query_count == len(client.calls) == 1
    assert receipt.operation == "refine_name"
    assert receipt.action_count == receipt.refine_action_count == 1
    assert receipt.review_action_count == 0
    assert receipt.target_run_id == "historical-run"
    assert receipt.catalog_cocos == [17]
    assert receipt.per_path_cocos_labels == {PATH: label}
    assert receipt.budget_remaining_before == Decimal("194.90")
    assert receipt.budget_remaining_after == Decimal("194.40")


def test_claim_and_preflight_consume_the_same_eligibility_predicate() -> None:
    assert (
        run_preflight.REFINE_NAME_ELIGIBILITY_WHERE
        is graph_ops.REFINE_NAME_ELIGIBILITY_WHERE
    )
    assert (
        graph_ops.REFINE_NAME_ELIGIBILITY_WHERE
        in run_preflight._EXACT_STANDARD_NAME_PREFLIGHT_QUERY
    )
    assert (
        run_preflight.REVIEW_NAME_ELIGIBILITY_WHERE
        is graph_ops.REVIEW_NAME_ELIGIBILITY_WHERE
    )
    assert (
        graph_ops.REVIEW_NAME_ELIGIBILITY_WHERE
        in run_preflight._EXACT_STANDARD_NAME_PREFLIGHT_QUERY
    )
    assert (
        run_preflight._EXACT_STANDARD_NAME_PREFLIGHT_QUERY.count(
            "WITH head(target_matches) AS target"
        )
        == 4
    )
    assert (
        run_preflight._EXACT_STANDARD_NAME_PREFLIGHT_QUERY.count(
            "UNWIND target_matches"
        )
        == 1
    )
    with (
        patch.object(graph_ops, "_claim_sn_atomic", return_value=[]) as claim,
        patch.object(graph_ops, "_verify_name_claim_winners", return_value=[]),
    ):
        graph_ops.claim_refine_name_batch()
    assert (
        claim.call_args.kwargs["eligibility_where"]
        is graph_ops.REFINE_NAME_ELIGIBILITY_WHERE
    )
    with (
        patch.object(graph_ops, "_claim_sn_atomic", return_value=[]) as claim,
        patch.object(graph_ops, "_verify_name_claim_winners", return_value=[]),
    ):
        graph_ops.claim_review_name_batch(facility="west", drain_scope_id="drain")
    assert (
        claim.call_args.kwargs["eligibility_where"]
        is graph_ops.REVIEW_NAME_ELIGIBILITY_WHERE
    )
    assert claim.call_args.kwargs["query_params"] == {
        "parent_desc_placeholder": DETERMINISTIC_PARENT_DESCRIPTION_PLACEHOLDER,
        "drain_scope_id": "drain",
        "min_score": graph_ops.DEFAULT_MIN_SCORE,
        "facility": "west",
    }


@pytest.mark.parametrize("label", [None, "psi_like", "ip_like"])
def test_review_pass_retains_exact_cocos_label_and_accepted_predecessor(
    label: str | None,
) -> None:
    receipt, client = _audit(_review_row(label), operation="review_name")

    assert receipt.passed is True
    assert receipt.query_count == len(client.calls) == 1
    assert receipt.operation == "review_name"
    assert receipt.action_count == receipt.review_action_count == 1
    assert receipt.refine_action_count == 0
    assert receipt.accepted_or_protected_lineage_ids == ["accepted_predecessor"]
    assert receipt.refinement_protected_source_ids == []
    assert receipt.per_path_cocos_labels == {PATH: label}


@pytest.mark.parametrize("target_dd_version", [None, "4.0.0", "4.1.1"])
def test_target_dd_version_is_optional_historical_provenance(
    target_dd_version: str | None,
) -> None:
    row = _review_row()
    row["targets"][0]["dd_version"] = target_dd_version  # type: ignore[index]

    receipt, _client = _audit(row, operation="review_name")

    assert receipt.passed is True
    assert receipt.raw_evidence["targets"][0]["dd_version"] == target_dd_version


@pytest.mark.parametrize(
    "cached_source_ids",
    [
        [PATH],
        [],
        [f"dd:{PATH}", "dd:unexplained/path"],
    ],
)
def test_source_path_cache_refuses_noncanonical_missing_or_extra_ids(
    cached_source_ids: list[str],
) -> None:
    row = _review_row()
    row["targets"][0]["source_paths"] = cached_source_ids  # type: ignore[index]

    receipt, _client = _audit(row, operation="review_name")

    assert receipt.passed is False
    assert (
        "target source-ID mirror differs from producing sources" in receipt.diagnostics
    )


@pytest.mark.parametrize("operation", ["review_name", "refine_name"])
@pytest.mark.parametrize("location", ["predecessor", "successor", "self"])
def test_refinement_lineage_protected_source_refuses_both_operations(
    location: str,
    operation: str,
) -> None:
    row = _review_row() if operation == "review_name" else _row()
    protected_source = f"fixture:{location}"
    row["refinement_protected_source_ids"] = [protected_source]
    if location == "successor":
        row["successor_ids"] = ["later_candidate"]

    receipt, _client = _audit(row, operation=operation)

    assert receipt.passed is False
    assert receipt.refinement_protected_source_ids == [protected_source]
    assert (
        "refinement lineage intersects WEST or fixture sources" in receipt.diagnostics
    )


def test_review_explicit_drain_scope_accepts_already_passing_review() -> None:
    row = _review_row()
    target = row["targets"][0]  # type: ignore[index]
    target["name_stage"] = "reviewed"
    target["reviewer_score_name"] = 0.9

    receipt, _client = _audit(
        row,
        operation="review_name",
        drain_scope_id="exact-drain",
    )

    assert receipt.passed is True
    assert receipt.drain_scope_id == "exact-drain"


@pytest.mark.parametrize(
    ("mutation", "diagnostic"),
    [
        ("stage", "target is not staged for name review"),
        ("validation", "target validation status is not valid"),
        ("description", "target has no reviewable description"),
        (
            "placeholder",
            "target still carries the deterministic parent placeholder",
        ),
        ("derived", "derived names are structurally fixed"),
        ("facility", "target facility does not match review scope"),
        ("cardinality", "exact review_name action cardinality is 0"),
    ],
)
def test_review_specific_ineligibility_refuses(mutation: str, diagnostic: str) -> None:
    row = _review_row()
    target = row["targets"][0]  # type: ignore[index]
    kwargs: dict[str, object] = {"operation": "review_name"}
    if mutation == "stage":
        target["name_stage"] = "accepted"
    elif mutation == "validation":
        target["validation_status"] = "quarantined"
    elif mutation == "description":
        target["description"] = None
    elif mutation == "placeholder":
        target["description"] = DETERMINISTIC_PARENT_DESCRIPTION_PLACEHOLDER
    elif mutation == "derived":
        target["origin"] = "derived"
    elif mutation == "facility":
        target["facility"] = "iter"
        kwargs["facility"] = "west"
    row["review_action_count"] = 0

    receipt, _client = _audit(row, **kwargs)

    assert receipt.passed is False
    assert diagnostic in receipt.diagnostics


@pytest.mark.parametrize(
    ("mutation", "diagnostic"),
    [
        ("claim", "target has an active worker or drain lease"),
        ("protection", "structural lineage intersects WEST or fixture sources"),
        ("unit", "target unit property and relationship differ"),
        ("cocos", "target did not preserve the per-path COCOS label"),
        ("successor", "target already has a refined successor"),
        ("predecessors", "target has ambiguous refinement predecessors"),
    ],
)
def test_review_shared_safety_evidence_refuses(mutation: str, diagnostic: str) -> None:
    row = _review_row("psi_like")
    target = row["targets"][0]  # type: ignore[index]
    if mutation == "claim":
        target["claim_token"] = "occupied"
    elif mutation == "protection":
        row["protected_source_ids"] = ["dd:west/protected"]
    elif mutation == "unit":
        target["unit"] = "s"
    elif mutation == "cocos":
        target["cocos_transformation_type"] = "ip_like"
    elif mutation == "successor":
        row["successor_ids"] = ["later_candidate"]
    else:
        row["predecessor_ids"] = ["earlier_a", "earlier_b"]

    receipt, _client = _audit(row, operation="review_name")

    assert receipt.passed is False
    assert diagnostic in receipt.diagnostics


def test_review_budget_overflow_refuses() -> None:
    receipt, _client = _audit(
        _review_row(),
        operation="review_name",
        requested_cost_ceiling="5.01",
        cumulative_spend="195.00",
    )

    assert receipt.passed is False
    assert "requested cost ceiling exceeds authorized budget" in receipt.diagnostics


def test_invalid_operation_refuses_before_graph_access() -> None:
    client = _Client([_review_row()])

    with pytest.raises(ValueError, match="operation must be"):
        audit_exact_standard_name_preflight(
            NAME,
            operation="generate_name",  # type: ignore[arg-type]
            requested_cost_ceiling="0.5",
            cumulative_spend="5",
            authorized_budget="200",
            dd_version="4.1.1",
            gc=client,
        )

    assert client.calls == []


@pytest.mark.parametrize(
    ("field", "value", "diagnostic"),
    [
        ("name_stage", "drafted", "target name stage is not reviewed"),
        (
            "reviewer_score_name",
            0.85,
            "target name-review score is not below minimum",
        ),
        ("chain_length", 3, "target refinement chain reached the rotation cap"),
        ("origin", "derived", "derived names are structurally fixed"),
    ],
)
def test_ineligible_target_details_fail_closed(
    field: str, value: object, diagnostic: str
) -> None:
    row = _row()
    row["targets"][0][field] = value  # type: ignore[index]
    row["refine_action_count"] = 0

    receipt, _client = _audit(row)

    assert receipt.passed is False
    assert diagnostic in receipt.diagnostics
    assert "exact refine_name action cardinality is 0" in receipt.diagnostics


def test_capped_pinned_rename_is_ineligible() -> None:
    row = _row()
    target = row["targets"][0]  # type: ignore[index]
    target["edit_mode"] = "rename"
    target["review_resubmit_count"] = 3
    row["refine_action_count"] = 0

    receipt, _client = _audit(row)

    assert receipt.passed is False
    assert "pinned rename exhausted its re-review budget" in receipt.diagnostics


@pytest.mark.parametrize("target_count", [0, 2])
def test_missing_or_ambiguous_identity_still_uses_one_query(target_count: int) -> None:
    row = _row()
    row["targets"] = [deepcopy(row["targets"][0]) for _ in range(target_count)]  # type: ignore[index]
    row["refine_action_count"] = target_count

    receipt, client = _audit(row)

    assert receipt.passed is False
    assert receipt.identity_count == target_count
    assert receipt.query_count == len(client.calls) == 1
    assert any("identity resolved" in message for message in receipt.diagnostics)


@pytest.mark.parametrize(
    ("field", "value", "diagnostic"),
    [
        (
            "protected_source_ids",
            ["dd:west/protected"],
            "structural lineage intersects WEST or fixture sources",
        ),
        (
            "accepted_or_protected_lineage_ids",
            ["catalog_owned_name"],
            "refinement lineage intersects accepted or protected state",
        ),
    ],
)
def test_protected_lineage_refuses(field: str, value: object, diagnostic: str) -> None:
    row = _row()
    row[field] = value

    receipt, _client = _audit(row)

    assert receipt.passed is False
    assert diagnostic in receipt.diagnostics


@pytest.mark.parametrize(
    ("record", "field"),
    [("target", "claim_token"), ("target", "drain_scope_id"), ("source", "claimed_at")],
)
def test_real_claim_or_drain_fields_block_but_run_history_does_not(
    record: str, field: str
) -> None:
    row = _row()
    if record == "target":
        row["targets"][0][field] = "occupied"  # type: ignore[index]
    else:
        row["sources"][0][field] = "occupied"  # type: ignore[index]

    receipt, _client = _audit(row)

    assert receipt.passed is False
    assert any("lease" in diagnostic for diagnostic in receipt.diagnostics)


@pytest.mark.parametrize(
    ("mutation", "diagnostic"),
    [
        ("source_identity", "DD source identity mismatch"),
        ("dd_version", "source DD version is not current"),
        ("unit", "source/backing unit mismatch"),
        ("projection", "projection is not singular"),
    ],
)
def test_source_dd_unit_and_projection_mismatches_refuse(
    mutation: str, diagnostic: str
) -> None:
    row = _row()
    source = row["sources"][0]  # type: ignore[index]
    if mutation == "source_identity":
        source["id"] = "dd:wrong/path"
    elif mutation == "dd_version":
        source["dd_version"] = "4.0.0"
    elif mutation == "unit":
        source["dd_unit"] = "s"
    elif mutation == "projection":
        source["projection_edge_ids"] = []

    receipt, _client = _audit(row)

    assert receipt.passed is False
    assert any(diagnostic in message for message in receipt.diagnostics)


@pytest.mark.parametrize(
    ("field", "diagnostic"),
    [
        ("dd_unit", "source DD unit is missing"),
        ("backing_unit", "backing unit property is missing"),
        ("backing_unit_ids", "backing unit relationship is missing or ambiguous"),
    ],
)
def test_each_source_unit_authority_field_is_required(
    field: str, diagnostic: str
) -> None:
    row = _row()
    source = row["sources"][0]  # type: ignore[index]
    source[field] = [] if field == "backing_unit_ids" else None

    receipt, _client = _audit(row)

    assert receipt.passed is False
    assert any(diagnostic in message for message in receipt.diagnostics)


@pytest.mark.parametrize(
    ("field", "diagnostic"),
    [
        ("property", "target unit property is missing"),
        ("relationship", "target unit relationship is missing or ambiguous"),
    ],
)
def test_each_target_unit_authority_field_is_required(
    field: str, diagnostic: str
) -> None:
    row = _row()
    if field == "property":
        row["targets"][0]["unit"] = None  # type: ignore[index]
    else:
        row["target_unit_ids"] = []

    receipt, _client = _audit(row)

    assert receipt.passed is False
    assert diagnostic in receipt.diagnostics


def test_noncanonical_catalog_cocos_refuses() -> None:
    row = _row("psi_like")
    row["catalogs"][0]["cocos"] = 11  # type: ignore[index]
    row["catalog_cocos_ids"] = [11]

    receipt, _client = _audit(row)

    assert receipt.passed is False
    assert "global DD catalog COCOS is not exactly 17" in receipt.diagnostics


def test_changed_per_path_cocos_label_refuses() -> None:
    row = _row("psi_like")
    row["targets"][0]["cocos_transformation_type"] = "ip_like"  # type: ignore[index]

    receipt, _client = _audit(row)

    assert receipt.passed is False
    assert "target did not preserve the per-path COCOS label" in receipt.diagnostics


def test_budget_overflow_refuses_before_paid_work() -> None:
    receipt, _client = _audit(
        _row(), requested_cost_ceiling="5.01", cumulative_spend="195.00"
    )

    assert receipt.passed is False
    assert receipt.budget_remaining_after == Decimal("-0.01")
    assert "requested cost ceiling exceeds authorized budget" in receipt.diagnostics


@pytest.mark.parametrize(
    ("argument", "value"),
    [
        ("requested_cost_ceiling", "NaN"),
        ("cumulative_spend", "Infinity"),
        ("authorized_budget", "-Infinity"),
        ("min_score", "NaN"),
    ],
)
def test_non_finite_decimal_inputs_refuse_before_graph_access(
    argument: str, value: str
) -> None:
    client = _Client([_row()])
    kwargs: dict[str, object] = {
        "requested_cost_ceiling": "0.5",
        "cumulative_spend": "5",
        "authorized_budget": "200",
        "min_score": "0.85",
        "dd_version": "4.1.1",
        "gc": client,
    }
    kwargs[argument] = value

    with pytest.raises(ValueError, match="cost and score inputs must be finite"):
        audit_exact_standard_name_preflight(NAME, **kwargs)

    assert client.calls == []


def test_noisy_adverse_evidence_does_not_increase_query_count() -> None:
    row = _row()
    source = deepcopy(row["sources"][0])  # type: ignore[index]
    source["element_id"] = "source-2"
    source["from_dd_edge_ids"] = ["from-dd-1", "from-dd-2"]
    source["projection_edge_ids"] = ["projection-1", "projection-2"]
    row["sources"] = [source for _ in range(250)]
    client = _Client([row])

    with patch(
        "imas_codex.standard_names.grammar_segment_reconciliation._west_source_ids",
        return_value=frozenset(),
    ):
        receipt = audit_exact_standard_name_preflight(
            NAME,
            requested_cost_ceiling="0.5",
            cumulative_spend="5",
            authorized_budget="200",
            dd_version="4.1.1",
            gc=client,
        )

    assert receipt.passed is False
    assert receipt.query_count == len(client.calls) == 1
    query, params = client.calls[0]
    assert "MATCH (candidate:StandardName {id: $name_id})" in query
    assert params["name_id"] == NAME
    assert not any(
        token in query for token in (" SET ", " DELETE ", " MERGE ", " CREATE ")
    )


def test_query_failure_returns_one_query_fail_closed_receipt() -> None:
    client = _Client([], error=RuntimeError("graph unavailable"))
    with patch(
        "imas_codex.standard_names.grammar_segment_reconciliation._west_source_ids",
        return_value=frozenset(),
    ):
        receipt = audit_exact_standard_name_preflight(
            NAME,
            requested_cost_ceiling="0.5",
            cumulative_spend="5",
            authorized_budget="200",
            dd_version="4.1.1",
            gc=client,
        )

    assert receipt.passed is False
    assert receipt.query_count == len(client.calls) == 1
    assert any("query failed" in message for message in receipt.diagnostics)
