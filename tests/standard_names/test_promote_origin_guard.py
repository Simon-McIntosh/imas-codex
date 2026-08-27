"""Catalog approval records its receipt without rewriting name provenance."""

from __future__ import annotations

from typing import Any

from imas_codex.standard_names.promote import (
    mark_catalog_name_approved,
    resolve_contested_override,
)


class _ApprovalGraph:
    """Minimal stateful graph double for the two catalog-approval writes."""

    def __init__(self, **properties: Any) -> None:
        self.properties = properties
        self.statements: list[str] = []

    def query(self, statement: str, **parameters: Any) -> list[dict[str, str]]:
        self.statements.append(statement)
        if "sn.origin = 'catalog_edit'" in statement:
            self.properties["origin"] = "catalog_edit"

        if "name_stage: 'contested'" in statement:
            if self.properties.get("name_stage") != "contested":
                return []
            self.properties["name_stage"] = "approved"
            self.properties["contested_resolution"] = parameters["reason"]
        else:
            if self.properties.get("name_stage") not in {"accepted", "approved"}:
                return []
            if self.properties.get("docs_stage") != "accepted":
                return []
            self.properties.update(
                name_stage="approved",
                catalog_pr_number=parameters["pr_number"],
                catalog_pr_url=parameters["pr_url"],
                catalog_merge_commit_sha=parameters["merge_commit"],
            )

        self.properties.setdefault("catalog_approved_at", "catalog-approval-time")
        return [{"id": parameters["name"]}]

    def close(self) -> None:
        pass


def _pipeline_identity(**properties: Any) -> dict[str, Any]:
    return {
        "origin": "pipeline",
        "model": "configured-generation-seat",
        "generated_at": "2026-07-01T12:34:56Z",
        "chain_length": 3,
        **properties,
    }


def _assert_generation_provenance_retained(graph: _ApprovalGraph) -> None:
    assert graph.properties["origin"] == "pipeline"
    assert graph.properties["model"] == "configured-generation-seat"
    assert graph.properties["generated_at"] == "2026-07-01T12:34:56Z"
    assert graph.properties["chain_length"] == 3


def test_catalog_approval_preserves_generated_identity_provenance() -> None:
    graph = _ApprovalGraph(
        **_pipeline_identity(name_stage="accepted", docs_stage="accepted")
    )

    assert mark_catalog_name_approved(
        "plasma_current",
        catalog_pr_number=42,
        catalog_pr_url="https://example.invalid/pull/42",
        catalog_merge_commit_sha="abc123",
        gc=graph,
    )

    _assert_generation_provenance_retained(graph)
    assert graph.properties["catalog_pr_number"] == 42
    assert graph.properties["catalog_merge_commit_sha"] == "abc123"
    assert graph.properties["catalog_approved_at"] == "catalog-approval-time"
    assert all("sn.origin" not in statement for statement in graph.statements)


def test_contested_override_preserves_generated_identity_provenance() -> None:
    graph = _ApprovalGraph(
        **_pipeline_identity(
            name_stage="contested",
            catalog_pr_number=42,
            catalog_merge_commit_sha="abc123",
            catalog_approved_at="initial-approval-time",
        )
    )

    assert resolve_contested_override(
        "plasma_current", reason="expert approval", gc=graph
    )

    _assert_generation_provenance_retained(graph)
    assert graph.properties["catalog_pr_number"] == 42
    assert graph.properties["catalog_merge_commit_sha"] == "abc123"
    assert graph.properties["catalog_approved_at"] == "initial-approval-time"
    assert all("sn.origin" not in statement for statement in graph.statements)
