"""Quarantine status and its explanation are one atomic graph write."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

from imas_codex.standard_names import (
    campaign,
    edit,
    graph_ops,
    promote,
    signed_manifest,
)


class RecordingGraph:
    """Small query double that preserves statements and their parameters."""

    def __init__(self) -> None:
        self.writes: list[tuple[str, dict[str, Any]]] = []

    def __enter__(self) -> RecordingGraph:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def query(self, cypher: str, **params: Any) -> list[dict[str, Any]]:
        self.writes.append((cypher, params))
        if "RETURN count(sn) AS n" in cypher:
            return [{"n": len(params.get("ids", []))}]
        if "RETURN target_stage AS stage" in cypher:
            return [{"stage": "exhausted"}]
        if ("RETURN sn.id AS id" in cypher or "RETURN name.id AS id" in cypher) and (
            "deterministic:dd_error_modifier" in cypher
            or params.get("model") == "deterministic:dd_error_modifier"
        ):
            return [{"id": "upper_uncertainty_of_missing_temperature"}]
        return []


def _quarantine_statement(graph: RecordingGraph) -> tuple[str, dict[str, Any]]:
    return next(
        (cypher, params)
        for cypher, params in reversed(graph.writes)
        if "SET " in cypher and "validation_status" in cypher
    )


def test_campaign_quarantine_records_the_canonical_reason() -> None:
    graph = RecordingGraph()

    outcome = campaign.default_revalidate(graph, ["bad_name"], [])

    assert outcome == {"requarantined": 1, "confirmed": 0}
    cypher, params = _quarantine_statement(graph)
    assert "sn.validation_issues = [$reason]" in cypher
    assert params["reason"] == "campaign: banned prose persisted after refine"
    assert "quarantine_reason" not in cypher


def test_campaign_manifest_reports_a_reasonless_quarantine() -> None:
    target = campaign.match_target(
        {
            "id": "reasonless_name",
            "name": "reasonless_name",
            "description": "A stored name.",
            "documentation": "A stored name without a quarantine explanation.",
            "validation_status": "quarantined",
            "validation_issues": None,
            "physics_domain": "equilibrium",
        },
        campaign.CampaignSpec.parse("quarantined"),
    )

    assert target is not None
    assert target.matched_predicates["quarantined:reason_missing"] == [
        "validation_issues is null"
    ]
    manifest = campaign.build_manifest(
        campaign.CampaignSelection(
            spec=campaign.CampaignSpec.parse("quarantined"), targets=[target]
        ),
        sample_size=1,
    )
    assert manifest["per_predicate"]["quarantined:reason_missing"] == 1
    assert manifest["sample"][0]["matched_predicates"][
        "quarantined:reason_missing"
    ] == ["validation_issues is null"]


def test_approval_quarantine_records_the_canonical_reason() -> None:
    graph = RecordingGraph()

    promote._quarantine(
        "candidate_name",
        axis="name",
        score=0.4,
        reason="review score below approval threshold",
        gc=graph,
    )

    cypher, params = _quarantine_statement(graph)
    assert "sn.validation_issues = [$reason]" in cypher
    assert params["reason"] == "review score below approval threshold"
    assert "merge_quarantine_reason" not in cypher


def test_edit_validation_already_pairs_status_and_reason() -> None:
    graph = RecordingGraph()
    issues = ["grammar: invalid canonical name"]

    with patch(
        "imas_codex.standard_names.workers.validate_name_candidate",
        return_value=(issues, {}, "quarantined"),
    ):
        edit._stamp_successor_validation(
            graph,
            "candidate_name",
            {
                "kind": "scalar",
                "unit": "1",
                "description": "A candidate.",
                "physics_domain": "equilibrium",
                "source_paths": [],
            },
        )

    cypher, params = _quarantine_statement(graph)
    assert "sn.validation_status = $status" in cypher
    assert "sn.validation_issues = $issues" in cypher
    assert params["status"] == "quarantined"
    assert params["issues"] == issues


def test_signed_orphan_authority_records_the_canonical_reason() -> None:
    authority = signed_manifest._load_error_sibling_authority(
        mutation_kind=signed_manifest.RepairMutationKind.set_properties.value,
        guard_set=signed_manifest._ERROR_SIBLING_GUARDS,
    )

    assert authority.data["validation_issues"] == [
        signed_manifest._ERROR_SIBLING_REASON
    ]
    assert "quarantine_reason" not in authority.data


def test_signed_orphan_payload_records_the_canonical_reason() -> None:
    graph = RecordingGraph()

    rows = signed_manifest._error_sibling_rows(
        graph, signed_manifest._ERROR_SIBLING_REASON
    )

    assert len(rows) == 1
    properties = rows[0].mutations[0]["arguments"]["properties"]
    assert properties == {
        "validation_status": "quarantined",
        "validation_issues": [signed_manifest._ERROR_SIBLING_REASON],
    }


def test_signed_orphan_direct_write_records_the_canonical_reason() -> None:
    graph = RecordingGraph()

    assert signed_manifest._apply_error_sibling_query_handle(graph) == {
        "stale_marked": 1
    }

    cypher, _params = _quarantine_statement(graph)
    assert "sn.validation_issues = [$reason]" in cypher
    assert "quarantine_reason" not in cypher


def test_terminal_refine_quarantine_records_the_canonical_reason() -> None:
    graph = RecordingGraph()

    with patch.object(graph_ops, "GraphClient", return_value=graph):
        stage = graph_ops.stop_refine_name_attempt(
            sn_id="candidate_name",
            token="claim-token",
            reason="grammar_invalid",
            detail="canonical parsing failed",
        )

    assert stage == "exhausted"
    cypher, params = _quarantine_statement(graph)
    assert "sn.validation_issues = CASE" in cypher
    assert "[$reason]" in cypher
    assert params["reason"] == "grammar_invalid"


def test_validation_worker_already_pairs_status_and_reason() -> None:
    graph = RecordingGraph()
    issues = ["grammar: invalid canonical name"]

    with patch.object(graph_ops, "GraphClient", return_value=graph):
        graph_ops.mark_names_validated(
            "claim-token",
            [
                {
                    "id": "candidate_name",
                    "validation_status": "quarantined",
                    "validation_issues": issues,
                    "validation_layer_summary": "{}",
                }
            ],
        )

    cypher, params = _quarantine_statement(graph)
    assert "sn.validation_issues = b.issues" in cypher
    assert "sn.validation_status = b.validation_status" in cypher
    assert params["batch"][0]["issues"] == issues
