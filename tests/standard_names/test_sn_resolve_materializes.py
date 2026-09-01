"""Contested overrides materialize the stored reviewer proposal."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from imas_codex.standard_names.promote import (
    ApprovalChange,
    resolve_contested_override,
    run_approval,
)


class _ProposalGraph:
    def __init__(self) -> None:
        self.node: dict[str, Any] = {
            "id": "pulse_duration",
            "name_stage": "accepted",
            "description": "Initial pulse duration wording.",
            "documentation": "Initial pulse duration wording.",
        }
        self.statements: list[str] = []

    def query(self, statement: str, **parameters: Any) -> list[dict[str, Any]]:
        self.statements.append(statement)
        if "APPROVAL_MATCH_BY_ID" in statement:
            return [{"n": 1}]
        if "APPROVAL_CONTEST" in statement:
            self.node.update(
                name_stage="contested",
                edit_status="rejected",
                contested_reason=parameters["reason"],
                catalog_pr_number=parameters["pr_number"],
                catalog_pr_url=parameters["pr_url"],
                catalog_merge_commit_sha=parameters["merge_commit"],
                catalog_reviewer_actor=parameters["reviewer_actor"],
            )
            return []
        if "name_stage: 'contested'" in statement:
            if self.node["name_stage"] != "contested":
                return []
            if self.node.get("edit_mode") == "docs":
                proposal = self.node["docs_hint"]
                self.node["description"] = proposal
                self.node["documentation"] = proposal
                self.node["edit_status"] = "applied"
            elif self.node.get("edit_mode") == "rename":
                self.node["id"] = self.node["name_hint"]
                self.node["edit_status"] = "applied"
            self.node["name_stage"] = "approved"
            self.node["contested_resolution"] = parameters["reason"]
            return [{"id": self.node["id"]}]
        return []


def test_override_materializes_reviewer_wording_and_pr_provenance() -> None:
    reviewer_wording = "Reviewer-edited pulse duration wording."
    change = ApprovalChange(
        sn_id="pulse_duration",
        axis="docs",
        old_value="Initial pulse duration wording.",
        new_value=reviewer_wording,
    )
    provenance = {
        "catalog_pr_number": 3,
        "catalog_pr_url": "https://github.com/example/catalog/pull/3",
        "catalog_merge_commit_sha": "0123456789abcdef",
        "catalog_reviewer_actor": "catalog-reviewer",
    }
    graph = _ProposalGraph()

    def attach_proposal(**kwargs: Any) -> SimpleNamespace:
        graph.node.update(
            edit_mode="docs",
            docs_hint=kwargs["docs"],
            documentation=kwargs["docs"],
            edit_status="open",
        )
        return SimpleNamespace(blocked=None, successor=None, run_id="sn-edit-test")

    with (
        patch(
            "imas_codex.standard_names.promote.read_pr_changes",
            return_value=[change],
        ),
        patch(
            "imas_codex.standard_names.promote.apply_edit",
            side_effect=attach_proposal,
        ),
        patch("imas_codex.standard_names.promote._score_proposal", return_value=0.5),
    ):
        report = run_approval(
            isnc_dir="/unused/catalog",
            base_ref="submitted-candidate",
            threshold=0.85,
            gc=graph,
            **provenance,
        )

    assert report.contested == [
        {"sn_id": "pulse_duration", "target_id": "pulse_duration", "score": 0.5}
    ]
    assert resolve_contested_override(
        "pulse_duration", reason="Accept the reviewer's wording.", gc=graph
    )

    assert graph.node["description"] == reviewer_wording
    assert graph.node["name_stage"] == "approved"
    assert graph.node["contested_resolution"] == "Accept the reviewer's wording."
    assert {field: graph.node[field] for field in provenance} == provenance
    resolver = next(
        statement
        for statement in graph.statements
        if "name_stage: 'contested'" in statement
    )
    assert "sn.description = approved_description" in resolver
    assert "sn.id = approved_name" in resolver
