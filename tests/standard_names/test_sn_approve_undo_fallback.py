"""Approval undo falls back to frozen-batch membership when provenance is absent."""

from __future__ import annotations

from typing import Any

from imas_codex.standard_names.promote import undo_approval


class _UndoGraph:
    def __init__(self) -> None:
        self.names: dict[str, dict[str, Any]] = {
            "stamped_entry": {
                "id": "stamped_entry",
                "name_stage": "approved",
                "catalog_pr_number": 7,
                "catalog_pr_url": "https://github.com/example/catalog/pull/7",
                "catalog_merge_commit_sha": "merge-seven",
                "catalog_reviewer_actor": "reviewer",
                "catalog_approved_at": "approval-time",
            },
            "unstamped_batch_entry": {
                "id": "unstamped_batch_entry",
                "name_stage": "approved",
                "catalog_pr_number": None,
                "catalog_pr_url": "https://github.com/example/catalog/pull/7",
                "catalog_merge_commit_sha": "merge-seven",
                "catalog_reviewer_actor": "reviewer",
                "catalog_approved_at": "approval-time",
            },
            "unrelated_entry": {
                "id": "unrelated_entry",
                "name_stage": "approved",
                "catalog_pr_number": None,
                "catalog_pr_url": None,
                "catalog_merge_commit_sha": None,
                "catalog_reviewer_actor": None,
                "catalog_approved_at": "other-approval-time",
            },
        }
        self.statements: list[str] = []

    def query(self, statement: str, **parameters: Any) -> list[dict[str, str]]:
        self.statements.append(statement)
        if "name_stage: 'approved'" in statement:
            demoted = []
            for node in self.names.values():
                matches_pr = node["catalog_pr_number"] == parameters["pr"]
                unstamped_member = (
                    node["catalog_pr_number"] is None
                    and node["id"] in parameters["batch"]
                )
                if node["name_stage"] != "approved" or not (
                    matches_pr or unstamped_member
                ):
                    continue
                node.update(
                    name_stage="accepted",
                    catalog_pr_number=None,
                    catalog_pr_url=None,
                    catalog_merge_commit_sha=None,
                    catalog_reviewer_actor=None,
                    catalog_approved_at=None,
                )
                demoted.append({"id": node["id"]})
            return sorted(demoted, key=lambda row: row["id"])
        if "name_stage: 'contested'" in statement:
            return []
        raise AssertionError(f"unexpected query: {statement}")


def test_undo_demotes_stamped_and_unstamped_batch_approvals_only() -> None:
    graph = _UndoGraph()
    report = undo_approval(
        pr_number=7,
        batch=["stamped_entry", "unstamped_batch_entry"],
        gc=graph,
    )

    assert report.demoted == ["stamped_entry", "unstamped_batch_entry"]
    for name in report.demoted:
        node = graph.names[name]
        assert node["name_stage"] == "accepted"
        assert node["catalog_pr_number"] is None
        assert node["catalog_pr_url"] is None
        assert node["catalog_merge_commit_sha"] is None
        assert node["catalog_reviewer_actor"] is None
        assert node["catalog_approved_at"] is None

    assert graph.names["unrelated_entry"] == {
        "id": "unrelated_entry",
        "name_stage": "approved",
        "catalog_pr_number": None,
        "catalog_pr_url": None,
        "catalog_merge_commit_sha": None,
        "catalog_reviewer_actor": None,
        "catalog_approved_at": "other-approval-time",
    }
    approved_query = next(
        statement
        for statement in graph.statements
        if "name_stage: 'approved'" in statement
    )
    assert "sn.catalog_pr_number IS NULL AND sn.id IN $batch" in approved_query
